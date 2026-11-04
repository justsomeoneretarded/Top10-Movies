"""Reference measurements use real media and never label a mix as isolated speech."""
from __future__ import annotations

import shutil
import subprocess
from pathlib import Path

import pytest

from movie_pipeline.core import PipelineError
from movie_pipeline import reference_analysis as ref


@pytest.fixture
def reference_video(tmp_path):
    if not shutil.which("ffmpeg") or not shutil.which("ffprobe"):
        pytest.skip("FFmpeg tools are required for the reference media fixture")
    path = tmp_path / "synthetic-reference.mp4"
    command = [
        "ffmpeg", "-hide_banner", "-loglevel", "error", "-y",
        "-f", "lavfi", "-i", "color=c=black:s=160x90:r=10:d=2",
        "-f", "lavfi", "-i", "color=c=white:s=160x90:r=10:d=2",
        "-f", "lavfi", "-i", "sine=frequency=440:sample_rate=48000:duration=4",
        "-filter_complex", "[0:v][1:v]concat=n=2:v=1:a=0[v]",
        "-map", "[v]", "-map", "2:a", "-c:v", "mpeg4", "-c:a", "aac",
        "-shortest", str(path),
    ]
    subprocess.run(command, check=True, capture_output=True)
    return path


def test_real_synthetic_video_has_cut_audio_report_and_clear_uncertainty(reference_video, tmp_path):
    report = ref.analyze(reference_video, tmp_path / "out")
    assert report["duration_seconds"] == pytest.approx(4, abs=0.2)
    assert any(abs(stamp - 2) <= 0.5 for stamp in report["visual"]["approximate_abrupt_boundaries_seconds"])
    assert report["mixed_audio"]["available"]
    assert report["mixed_audio"]["integrated_lufs"] is not None
    assert "entire mixed" in report["mixed_audio"]["scope"]
    assert not report["narrator_audio"]["available"]
    assert not report["music_audio"]["available"]
    assert (tmp_path / "out" / "contact-sheet.jpg").is_file()
    text = (tmp_path / "out" / "reference-analysis.md").read_text()
    assert "not separated" in text
    assert "not an automatic assessment" in text


def test_missing_reference_is_controlled(tmp_path):
    with pytest.raises(PipelineError, match="does not exist"):
        ref.analyze(tmp_path / "missing.mp4", tmp_path / "out")


def test_transcription_unavailable_does_not_discard_audio_or_video(reference_video, tmp_path, monkeypatch):
    monkeypatch.setattr(ref, "transcribe_audio",
                        lambda path: {"available": False, "segments": [], "reason": "unavailable"})
    report = ref.analyze(reference_video, tmp_path / "out", transcribe=True)
    assert not report["transcript"]["available"]
    assert report["mixed_audio"]["available"]
    assert report["visual"]["frames_decoded"] > 0
    assert (tmp_path / "out" / "reference-analysis.json").is_file()


def test_missing_stem_is_not_silently_interpreted_as_a_mix(reference_video, tmp_path):
    with pytest.raises(PipelineError, match="stem does not exist"):
        ref.analyze(reference_video, tmp_path / "out", voice_stem=tmp_path / "absent.wav")


@pytest.mark.parametrize("value", ["bad", None, "inf", "-inf", float("nan")])
def test_invalid_or_nonfinite_measurements_are_unknown(value):
    assert ref._number(value) is None


def test_provided_srt_and_untimed_text_are_reported_without_inventing_speaker(tmp_path):
    captions = tmp_path / "sample.srt"
    captions.write_text("1\n00:00:01,000 --> 00:00:03,200\nHello there.\n\n"
                        "2\n00:00:04,000 --> 00:00:05,500\nAnother line.\n")
    result = ref.transcript_from_file(captions)
    assert result["segments"][0] == {"start": 1.0, "end": 3.2, "text": "Hello there."}
    assert result["segments"][1]["start"] == 4.0
    assert "speaker identity is not inferred" in " ".join(result["limitations"])
    plain = tmp_path / "plain.txt"
    plain.write_text("This text has no timecodes.")
    result = ref.transcript_from_file(plain)
    assert result["segments"][0]["start"] is None


def test_invalid_caption_timing_does_not_pretend_to_be_valid(tmp_path):
    bad = tmp_path / "bad.vtt"
    bad.write_text("WEBVTT\n\n00:01 --> 00:02\nNot valid timestamps.\n")
    with pytest.raises(PipelineError, match="No valid timed cues"):
        ref.transcript_from_file(bad)


def test_comparison_only_contains_measured_values(reference_video, tmp_path):
    first = ref.analyze(reference_video, tmp_path / "ref")
    second = ref.analyze(reference_video, tmp_path / "own")
    comparison = ref.compare(first, second, tmp_path / "comparison")
    assert comparison["metrics"]["duration_seconds"]["difference"] == 0
    assert "narrator_audio_integrated_loudness" not in comparison["metrics"]
    assert "music_audio_integrated_loudness" not in comparison["metrics"]
    assert "transcribed_word_count" not in comparison["metrics"]
    assert "do not establish" in " ".join(comparison["limitations"])
    assert (tmp_path / "comparison" / "comparison.json").is_file()
    assert (tmp_path / "comparison" / "comparison.md").is_file()


def test_reference_workflow_configures_supported_javascript_runtime():
    """The yt-dlp Python API must explicitly enable the Node runner."""
    import yaml
    from pathlib import Path
    data = yaml.load(Path(".github/workflows/reference-analysis.yml").read_text(), Loader=yaml.BaseLoader)
    steps = data["jobs"]["review"]["steps"]
    assert any(step.get("uses", "").startswith("actions/setup-node@") for step in steps)
    download = next(step for step in steps if step.get("name") == "Download external reference when supplied")
    assert '"js_runtimes":{"node":{}}' in download["run"]


def test_contact_sheet_falls_back_to_ffmpeg_when_opencv_cannot_decode(reference_video, tmp_path, monkeypatch):
    class UnavailableCapture:
        def set(self, *args):
            pass
        def read(self):
            return False, None
        def release(self):
            pass
    monkeypatch.setattr(ref.cv2, "VideoCapture", lambda *args: UnavailableCapture())
    output = tmp_path / "contact.jpg"
    timestamps = ref._contact_sheet(reference_video, 4.0, output)
    assert timestamps and output.is_file()


def test_transcript_summary_merges_overlaps_and_labels_uncaptioned_gaps():
    transcript = {"available": True, "segments": [
        {"start": 1.0, "end": 3.0, "text": "One two"},
        {"start": 2.0, "end": 4.0, "text": "Three"},
        {"start": 8.0, "end": 9.0, "text": "Four"},
    ]}
    stats = ref.summarize_transcript(transcript, 12.0)
    assert stats["word_count"] == 4
    assert stats["captioned_speech_seconds"] == 4.0
    assert stats["first_caption_seconds"] == 1.0
    assert stats["uncaptioned_intervals_at_least_two_seconds"] == [
        {"start": 4.0, "end": 8.0}, {"start": 9.0, "end": 12.0},
    ]
    assert "not confirmed silence" in " ".join(stats["limitations"])


def test_untimed_or_missing_transcript_does_not_invent_speech_timing():
    assert ref.summarize_transcript({"available": False}, 10)["available"] is False
    stats = ref.summarize_transcript(
        {"available": True, "segments": [{"start": None, "end": None, "text": "Two words."}]}, 10)
    assert stats["word_count"] == 2 and stats["timed"] is False
    assert "first_caption_seconds" not in stats
