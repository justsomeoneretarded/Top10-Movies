import numpy as np
from PIL import Image

from movie_pipeline.core import ffmpeg
from movie_pipeline.review import build_review, loudness_spikes, repeated_shots, silent_captions


def test_checks_find_repeats_spikes_and_drift():
    pattern = Image.effect_mandelbrot((320, 180), (-2, -1, 1, 1), 60).convert("RGB")
    other = Image.linear_gradient("L").resize((320, 180)).convert("RGB")
    frames = [(0, pattern), (10, other), (40, pattern)]
    assert repeated_shots(frames, ignore_after=100) == [(0, 40)]
    assert repeated_shots(frames, ignore_after=30) == []
    profile = np.full(100, -30.0)
    profile[50] = -10
    profile[60:70] = -80
    assert loudness_spikes(profile) == [20.0]
    cues = [{"start": 24.0, "end": 27.6, "text": "said over silence"}, {"start": 2.0, "end": 4.0, "text": "fine"}]
    assert [c["text"] for c in silent_captions(cues, profile)] == ["said over silence"]


def test_review_page_is_written(tmp_path, monkeypatch):
    monkeypatch.setenv("SCENE_DESCRIPTIONS", "false")
    ffmpeg("-f", "lavfi", "-i", "testsrc2=size=640x360:rate=30", "-f", "lavfi", "-i", "sine=frequency=220:sample_rate=48000",
           "-t", 30, "-c:v", "libx264", "-preset", "ultrafast", "-c:a", "aac", tmp_path / "final-video.mp4")
    Image.new("RGB", (1280, 720), (90, 20, 20)).save(tmp_path / "thumbnail-01.jpg")
    summary = build_review(tmp_path, [{"start": 1, "end": 3, "text": "Hello."}], [{"start": 0, "title": "Intro"}], "horror", ending_seconds=5)
    page = (tmp_path / "review.html").read_text()
    assert "Contact sheet" in page and "thumbnail-01.jpg" in page and "data:image/jpeg;base64," in page
    assert summary["thumbnails"][0]["file"] == "thumbnail-01.jpg"


def test_intro_and_recap_repeats_are_not_counted():
    pattern = Image.effect_mandelbrot((320, 180), (-2, -1, 1, 1), 60).convert("RGB")
    other = Image.linear_gradient("L").resize((320, 180)).convert("RGB")
    frames = [(0, pattern), (10, other), (40, pattern)]
    assert repeated_shots(frames, ignore_after=100, ignore=[(35, 45)]) == []
    assert repeated_shots(frames, ignore_after=100, ignore=[(0, 5)]) == []



def test_thumbnail_scores_cover_all_numbered_options_in_numeric_order(tmp_path, monkeypatch):
    from movie_pipeline import scenes
    from movie_pipeline.review import thumbnail_scores

    monkeypatch.setattr(scenes, "enabled", lambda: False)
    for name in ("thumbnail-01.jpg", "thumbnail-02.jpg", "thumbnail-09.jpg",
                 "thumbnail-10.jpg", "thumbnail-11.jpg", "thumbnail-02-backup.jpg"):
        Image.new("RGB", (64, 36), (90, 20, 20)).save(tmp_path / name)
    (tmp_path / "thumbnail-12.jpg").mkdir()  # Ignore a similarly named directory.
    rows = thumbnail_scores(tmp_path, "horror")
    assert [row["file"] for row in rows] == [
        "thumbnail-01.jpg", "thumbnail-02.jpg", "thumbnail-09.jpg",
        "thumbnail-10.jpg", "thumbnail-11.jpg",
    ]


def test_repeat_review_reports_when_sampled_comparison_is_unavailable(tmp_path, monkeypatch):
    from movie_pipeline import review

    monkeypatch.setattr(review, "duration", lambda video: 30.0)
    monkeypatch.setattr(review, "_frames", lambda *args: [])
    monkeypatch.setattr(review, "black_frames", lambda video: [])
    monkeypatch.setattr(review, "loudness_profile", lambda video: np.zeros(0))
    monkeypatch.setattr(review, "thumbnail_scores", lambda *args: [])
    summary = review.build_review(tmp_path, [], [], "neutral", ending_seconds=5)
    page = (tmp_path / "review.html").read_text()
    assert summary["repeat_detection"]["available"] is False
    assert "Not enough comparable frames" in page
    assert "samples one frame every two seconds" in page
    assert "can miss shorter or visually different repeats" in page


def test_repeat_review_describes_the_limit_of_a_zero_result(tmp_path, monkeypatch):
    from movie_pipeline import review

    first = Image.effect_mandelbrot((320, 180), (-2, -1, 1, 1), 60).convert("RGB")
    second = Image.linear_gradient("L").resize((320, 180)).convert("RGB")
    monkeypatch.setattr(review, "duration", lambda video: 90.0)
    monkeypatch.setattr(review, "_frames", lambda *args: [(2.0, first), (42.0, second)])
    monkeypatch.setattr(review, "black_frames", lambda video: [])
    monkeypatch.setattr(review, "loudness_profile", lambda video: np.zeros(0))
    monkeypatch.setattr(review, "thumbnail_scores", lambda *args: [])
    summary = review.build_review(tmp_path, [], [], "neutral", ending_seconds=5)
    page = (tmp_path / "review.html").read_text()
    assert summary["repeated_shots"] == 0
    assert summary["repeat_detection"]["available"] is True
    assert "No matching sampled frames" in page
    assert "can miss shorter or visually different repeats" in page
