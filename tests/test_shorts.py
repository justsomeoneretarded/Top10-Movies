import numpy as np

from movie_pipeline.core import ffmpeg, probe
from movie_pipeline.shorts import build_shorts, window

CHAPTERS = [{"start": 0, "title": "Intro"}, {"start": 3, "title": "#2 Second (2024)"},
            {"start": 30, "title": "#1 First (2025)"}, {"start": 58, "title": "Outro"}]
CUES = [{"start": 3.5, "end": 8.0, "text": "Number two: Second. A chase you will not forget."},
        {"start": 9.0, "end": 26.0, "text": "It keeps the tension going until the very last frame."},
        {"start": 30.5, "end": 40.0, "text": "Number one: First. The scariest night of the year."},
        {"start": 41.0, "end": 55.0, "text": "Watch it with the lights on."}]


def test_window_ends_on_a_finished_sentence_and_skips_short_entries():
    assert window(CHAPTERS, 1, CUES) == (30, 55.4)
    assert window(CHAPTERS, 2, CUES) == (3, 26.4)
    assert window(CHAPTERS, 3, CUES) is None


def test_shorts_are_vertical_with_burned_word_captions(tmp_path):
    video = tmp_path / "final.mp4"
    ffmpeg("-f", "lavfi", "-i", "testsrc2=size=1280x720:rate=30", "-f", "lavfi", "-i", "sine=frequency=330:sample_rate=48000",
           "-t", 60, "-c:v", "libx264", "-preset", "ultrafast", "-c:a", "aac", video)
    films = [{"title": "First", "year": 2025}, {"title": "Second", "year": 2024}]
    made = build_shorts(video, CHAPTERS, CUES, films, "Top 2 Scary Movies", tmp_path / "shorts", tmp_path / "work", count=2)
    assert [item["rank"] for item in made] == [1, 2]
    short = tmp_path / "shorts" / made[0]["file"]
    stream = next(s for s in probe(short)["streams"] if s["codec_type"] == "video")
    assert (stream["width"], stream["height"]) == (1080, 1920)
    assert 24 < float(probe(short)["format"]["duration"]) < 26.5
    assert "#shorts" in short.with_suffix(".txt").read_text()
    ass = (tmp_path / "work" / (short.stem + ".ass")).read_text()
    assert "#1 FIRST" in ass and "\\fscx112" in ass
