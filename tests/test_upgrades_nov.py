"""Sizzle-reel intro, channel watermark, phrase-level voice, and stills instead of repeated footage."""
from pathlib import Path

from PIL import Image

from movie_pipeline.previews import SIZZLE_GENRES, sizzle_plan
from movie_pipeline.render import speech_video, watermark_image
from movie_pipeline.voice import phrases


class Selector:
    def __init__(self):
        self.reserved = []

    def card_kind(self, path, index):
        return "title" if index == 0 else None

    def reserve(self, path, start, seconds, kind):
        self.reserved.append((str(path), start, seconds))


def test_sizzle_opens_on_the_best_moment_and_rotates_titles():
    a, b = Path("a.mp4"), Path("b.mp4")
    lookup = {str(a): [{"start": 0, "end": 5, "peak": 1.0}, {"start": 10, "end": 14, "peak": 0.4}, {"start": 20, "end": 24, "peak": 0.2}],
              str(b): [{"start": 0, "end": 5, "peak": 0.1}, {"start": 30, "end": 34, "peak": 0.9}, {"start": 40, "end": 44, "peak": 0.5}]}
    selector = Selector()
    plan = sizzle_plan(selector, [[a], [b]], lookup, seconds=10.4, shot_seconds=2.6)
    assert len(plan) == 4
    assert [cut["film_index"] for cut in plan] == [1, 0, 1, 0]  # b's 0.9 beats a's best usable 0.4 (a's 1.0 is a title card)
    assert plan[0]["start"] >= 30 and not any(cut["source"] == "a.mp4" and cut["start"] < 5 for cut in plan)
    assert len(selector.reserved) == 4 and {"action", "adventure"} <= SIZZLE_GENRES and "comedy" not in SIZZLE_GENRES


def test_watermark_is_small_and_translucent(tmp_path):
    width, height = watermark_image(tmp_path / "w.png")
    image = Image.open(tmp_path / "w.png")
    assert width < 320 and height == 44 and image.getchannel("A").getextrema()[1] < 255


def test_phrases_keep_the_reveal_alone_and_join_short_sentences():
    parts = ["Number 5: White Chicks.", "Two agents go undercover.", "It gets weird fast.", "Then weirder."]
    assert phrases(parts, reveal=True) == ["Number 5: White Chicks.", "Two agents go undercover. It gets weird fast.", "Then weirder."]


def test_an_official_still_renders_with_a_smooth_zoom(tmp_path):
    still = tmp_path / "still.jpg"
    Image.new("RGB", (1920, 1080), (120, 80, 40)).save(still)
    audio = tmp_path / "voice.wav"
    import subprocess
    subprocess.run(["ffmpeg", "-loglevel", "error", "-f", "lavfi", "-i", "sine=f=220:d=3", "-ar", "48000", str(audio)], check=True)
    out = tmp_path / "part.mp4"
    speech_video(still, audio, [{"source": str(still), "image": True, "start": 0.0, "duration": 3.0, "still": True,
                                 "freeze": 0.0, "still_motion": "zoom_in"}], out, None, tmp_path / "render")
    assert out.stat().st_size > 1000
