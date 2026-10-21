"""Reaction selection and real moving-media rendering, without any network/model."""
import shutil

import numpy as np
import pytest
from PIL import Image

from movie_pipeline import cta, render
from movie_pipeline.core import PipelineError, duration, ffmpeg, probe


@pytest.mark.parametrize("title,theme,animated", [
    ("Top 5 Action Movies", "high-five", True),
    ("Top 5 Science Fiction Movies", "logical-cat", True),
    ("Top 5 Horror Movies", "crying-cat", False),
])
def test_reaction_joke_and_asset_are_a_matching_reproducible_pair(tmp_path, title, theme, animated):
    first = cta.prepare_cta(title, "A casual countdown.", "video-seed", tmp_path)
    assert first == cta.prepare_cta(title, "A casual countdown.", "video-seed", tmp_path)
    assert first["theme"] == theme
    assert first["animated"] is animated
    assert len(first["text"].split()) <= 16
    assert "like" in first["text"].lower() and "subscribe" in first["text"].lower()
    assert first["source"].startswith("https://github.com/")
    assert first["attribution"]
    with Image.open(first["path"]) as picture:
        assert (getattr(picture, "n_frames", 1) > 1) is animated
    if theme == "high-five":
        assert "five" in first["text"]
    elif theme == "logical-cat":
        assert "logical" in first["text"] and "cat" in first["text"]
    else:
        assert "cat" in first["text"]


def test_missing_asset_changes_the_joke_with_the_reaction(monkeypatch, tmp_path):
    assets = tmp_path / "assets"
    assets.mkdir()
    shutil.copy(cta.ASSET_ROOT / "spock-cat.gif.b64", assets)
    monkeypatch.setattr(cta, "ASSET_ROOT", assets)
    plan = cta.prepare_cta("Action movies", "", "seed", tmp_path / "work")
    assert plan["theme"] == "logical-cat"
    assert "logical" in plan["text"]
    assert "high five" not in plan["text"]


def test_missing_assets_use_a_film_background_without_failing(monkeypatch, tmp_path):
    monkeypatch.setattr(cta, "ASSET_ROOT", tmp_path / "unavailable")
    plan = cta.prepare_cta("Horror movies", "", "seed", tmp_path / "work")
    assert plan["path"] is None
    assert not plan["animated"]
    assert "cat" not in plan["text"].casefold()
    assert "like and subscribe" in plan["text"].casefold()


def test_optional_decode_failure_uses_the_selected_movie_and_exact_time(monkeypatch, tmp_path):
    source = tmp_path / "trailer.mp4"
    source.touch()
    audio = tmp_path / "voice.wav"
    plan = {"path": str(tmp_path / "bad.gif"), "animated": True, "headline": "HIGH FIVE?",
            "attribution": "Unused reaction credit", "source": "https://unused.example/reaction.gif"}
    recorded = {}
    monkeypatch.setattr(cta, "duration", lambda path: 2.0)

    def failed_reaction(*args):
        raise PipelineError("Invalid optional GIF")

    def record(source, audio, chosen, output, label, work, **kwargs):
        recorded.update(source=source, chosen=chosen, **kwargs)

    monkeypatch.setattr(cta, "_reaction_video", failed_reaction)
    monkeypatch.setattr(render, "speech_video", record)
    actual = cta.render_cta(plan, audio, tmp_path / "cta.mp4", tmp_path / "render",
                            background={"source": str(source), "start": 3.4})
    assert recorded["source"] == source
    assert recorded["chosen"] == [{"source": str(source), "start": 3.4, "duration": 2.0,
                                    "image": False, "animation": False, "still": True}]
    assert recorded["label_seconds"] == 2.0
    assert actual == {**recorded["chosen"][0], "attribution": "", "credit_source": ""}


def test_neutral_fallback_is_reported_as_an_uncredited_still(monkeypatch, tmp_path):
    monkeypatch.setattr(cta, "duration", lambda path: 2.0)
    monkeypatch.setattr(render, "speech_video", lambda *args, **kwargs: None)
    plan = {"path": None, "animated": False, "headline": "SAVE MOVIE NIGHT."}
    actual = cta.render_cta(plan, tmp_path / "voice.wav", tmp_path / "cta.mp4", tmp_path / "render")
    assert actual["source"].endswith("neutral-background.png")
    assert actual["image"] and actual["still"]
    assert not actual["animation"]
    assert actual["attribution"] == actual["credit_source"] == ""


@pytest.mark.skipif(not shutil.which("ffmpeg"), reason="FFmpeg is required")
def test_real_gif_stays_animated_and_complete_speech_uses_lossless_audio(monkeypatch, tmp_path):
    from movie_pipeline.shots import get_frame

    monkeypatch.setenv("VIDEO_RESOLUTION", "360")
    plan = cta.prepare_cta("Top 5 Action Movies", "", "seed", tmp_path / "assets")
    audio = tmp_path / "voice.wav"
    ffmpeg("-f", "lavfi", "-i", "sine=frequency=220:sample_rate=48000", "-t", 1.4, audio)
    target = tmp_path / "cta.mp4"
    actual = cta.render_cta(plan, audio, target, tmp_path / "render")
    assert actual == {"source": plan["path"], "start": 0, "duration": duration(audio),
                      "image": True, "animation": True, "still": False,
                      "attribution": plan["attribution"], "credit_source": plan["source"]}
    assert duration(audio) <= duration(target) < duration(audio) + 0.04
    streams = probe(target)["streams"]
    assert next(stream for stream in streams if stream["codec_type"] == "audio")["codec_name"] == "alac"
    assert (next(stream for stream in streams if stream["codec_type"] == "video")["width"],
            next(stream for stream in streams if stream["codec_type"] == "video")["height"]) == (640, 360)
    # Exclude the caption and its entrance animation: the underlying GIF itself
    # must change, rather than a static first frame with moving text.
    first = np.asarray(get_frame(target, 0.25), dtype=float)[30:240, 160:480]
    later = np.asarray(get_frame(target, 0.9), dtype=float)[30:240, 160:480]
    assert np.abs(first-later).mean() > 3
    ffmpeg("-xerror", "-i", target, "-f", "null", "-")
