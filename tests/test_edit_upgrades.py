"""Delivery-based cut length, rank reveal transition and the end-screen plate."""
import shutil
from pathlib import Path

import numpy as np
import pytest
from PIL import Image

from movie_pipeline.core import duration, ffmpeg, probe
from movie_pipeline.render import end_screen_image, end_screen_video, trailer_video
from movie_pipeline.shots import PACE_LIMITS, ShotSelector, get_frame, pace_limit

needs_ffmpeg = pytest.mark.skipif(not shutil.which("ffmpeg"), reason="FFmpeg is required")


def selector(paths):
    selected = object.__new__(ShotSelector)
    selected.uses, selected.last_shots, selected.cut_count = {}, {}, 0
    selected.relevance = lambda *_: np.ones(24)
    selected.faces = {str(path): [False]*24 for path in paths}
    selected.playback_speed = 1
    return selected


def scenes():
    return [{"start": i*20, "end": i*20+15, "motion": 8} for i in range(24)]


@pytest.mark.parametrize("delivery", ["hype", "normal", "dramatic"])
def test_moving_shots_follow_the_line_delivery(delivery):
    path = Path("movie.mp4")
    selected = selector([path])
    selected.pace_limit = pace_limit(delivery)
    cuts = selected.choose(path, scenes(), "A specific line of narration.", 30, False)
    assert sum(cut["duration"] for cut in cuts) == pytest.approx(30)
    moving = [cut for cut in cuts if not cut["still"] and not cut["freeze"]]
    assert moving and max(cut["duration"] for cut in moving) <= PACE_LIMITS[delivery] + 1e-6


def test_hype_lines_cut_faster_than_dramatic_lines():
    path = Path("movie.mp4")
    fast, slow = selector([path]), selector([path])
    fast.pace_limit, slow.pace_limit = pace_limit("hype"), pace_limit("dramatic")
    assert len(fast.choose(path, scenes(), "Hook.", 30, False)) > len(slow.choose(path, scenes(), "Tension.", 30, False))
    assert pace_limit("hype", action=True) < pace_limit("hype")


@needs_ffmpeg
def test_rank_reveal_flashes_and_punches_in_then_settles(monkeypatch, tmp_path):
    monkeypatch.setenv("VIDEO_RESOLUTION", "640x360")
    source = tmp_path / "source.mp4"
    ffmpeg("-f", "lavfi", "-i", "testsrc2=size=640x360:rate=30:duration=4", "-c:v", "libx264", "-pix_fmt", "yuv420p", source)
    plain, revealed = tmp_path / "plain.mp4", tmp_path / "revealed.mp4"
    trailer_video(source, 0, 2, plain, muted=True)
    trailer_video(source, 0, 2, revealed, muted=True, reveal=True)
    assert duration(revealed) == pytest.approx(2, abs=.05)
    first_plain = np.asarray(get_frame(plain, 0.02), dtype=float)
    first_revealed = np.asarray(get_frame(revealed, 0.02), dtype=float)
    assert first_revealed.mean() > first_plain.mean() + 40  # white flash
    settled_plain = np.asarray(get_frame(plain, 1.5), dtype=float)
    settled_revealed = np.asarray(get_frame(revealed, 1.5), dtype=float)
    assert np.abs(settled_plain - settled_revealed).mean() < 4  # back to normal framing


@needs_ffmpeg
def test_end_screen_is_silent_calm_and_full_length(monkeypatch, tmp_path):
    monkeypatch.setenv("VIDEO_RESOLUTION", "640x360")
    plate = tmp_path / "plate.png"
    end_screen_image(plate, Image.new("RGB", (1920, 1080), (200, 60, 40)))
    with Image.open(plate) as image:
        assert image.size == (1280, 720)
        pixels = np.asarray(image, dtype=float)
    # Dimmed background; bright headline only near the top.
    assert pixels[300:].mean() < 90
    assert pixels[60:130].max() > 240
    end_screen_image(tmp_path / "plain.png", None)
    output = tmp_path / "end.mp4"
    end_screen_video(plate, 4, output)
    assert duration(output) == pytest.approx(4, abs=.05)
    audio = next(stream for stream in probe(output)["streams"] if stream["codec_type"] == "audio")
    assert audio["codec_name"] == "alac"
