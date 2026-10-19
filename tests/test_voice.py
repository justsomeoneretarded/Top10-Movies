from contextlib import nullcontext
from types import SimpleNamespace
from unittest.mock import Mock

import numpy as np
import pytest

from movie_pipeline.core import PipelineError
from movie_pipeline.voice import Narrator


class Wave:
    def __init__(self, samples):
        self.samples = samples

    def squeeze(self):
        return self

    detach = squeeze
    cpu = squeeze

    def numpy(self):
        return self.samples


def setup_speaker(monkeypatch, tmp_path, seconds, samples):
    torch = SimpleNamespace(manual_seed=Mock(), inference_mode=nullcontext)
    soundfile = SimpleNamespace(write=lambda path, data, sr: path.write_bytes(b"raw"))
    monkeypatch.setitem(__import__("sys").modules, "torch", torch)
    monkeypatch.setitem(__import__("sys").modules, "soundfile", soundfile)
    monkeypatch.setattr("movie_pipeline.voice.duration", lambda path: seconds)
    monkeypatch.setattr("movie_pipeline.voice.ffmpeg", lambda *args: args[-1].write_bytes(b"normalized"))
    speaker = object.__new__(Narrator)
    speaker.reference_hash = "test-reference"
    speaker.model_name = "turbo"
    speaker.cache = tmp_path
    speaker.model = SimpleNamespace(sr=24000, generate=Mock(return_value=Wave(samples)))
    return speaker


@pytest.mark.parametrize("word_count, seconds", [(36, 6), (80, 75)])
def test_fast_or_slow_delivery_alone_does_not_abort(monkeypatch, tmp_path, word_count, seconds):
    speaker = setup_speaker(monkeypatch, tmp_path, seconds, np.full(24000, 0.1))
    path = speaker.speak(" ".join(["word"]*word_count))
    assert path.exists()
    assert speaker.model.generate.call_count == 1


@pytest.mark.parametrize("samples", [np.zeros(200), np.array([]), np.array([float("nan")])])
def test_empty_silent_or_invalid_audio_is_still_rejected(monkeypatch, tmp_path, samples):
    speaker = setup_speaker(monkeypatch, tmp_path, 6, samples)
    with pytest.raises(PipelineError, match="empty, silent"):
        speaker.speak("A closing sentence for the video.")
    assert speaker.model.generate.call_count == 3


def test_runaway_audio_is_rejected_with_diagnostics(monkeypatch, tmp_path):
    speaker = setup_speaker(monkeypatch, tmp_path, 600, np.full(24000, 0.1))
    with pytest.raises(PipelineError, match="runaway"):
        speaker.speak("Thanks for watching.")
    assert speaker.model.generate.call_count == 3
