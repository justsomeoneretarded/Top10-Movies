
import pytest

from movie_pipeline.cli import choose_beat
from movie_pipeline.core import (
    PipelineError,
    narrator_from_brief,
    parse_movies,
    write_srt,
)
from movie_pipeline.footage import candidate_ok
from movie_pipeline.voice import spoken_coverage


def test_parse_titles_preserves_order_and_punctuation():
    movies = parse_movies("Spider-Man: No Way Home (2021); Arrival 2016\nCoherence (2013)")
    assert [(m.title, m.year) for m in movies] == [("Spider-Man: No Way Home", 2021), ("Arrival", 2016), ("Coherence", 2013)]


@pytest.mark.parametrize("text", ["Arrival", "Arrival (2016)", "Arrival (2016); Arrival (2016)", "Arrival (9999); Inception (2010)"])
def test_invalid_inputs_stop_early(text):
    with pytest.raises(PipelineError):
        parse_movies(text)


@pytest.mark.parametrize("brief, expected", [
    ("An energetic action countdown.", "warguy"),
    ("Use male audio.", "warguy"),
    ("Use Kit Laser for narration.", "kit-laser"),
    ("Use the female audio.", "female"),
    ("Female voice, slow and cinematic.", "female"),
    ("Don't use female audio.", "warguy"),
    ("Use male, not female.", "warguy"),
    ("Movies with female protagonists.", "warguy"),
    ("Use Warguy. Avoid the female voice.", "warguy"),
])
def test_narrator_selection(brief, expected):
    assert narrator_from_brief(brief) == expected


def test_ambiguous_narrator_rejected():
    with pytest.raises(PipelineError):
        narrator_from_brief("Use Kit Laser and the female voice")


def test_srt_is_uploadable_and_nonoverlapping(tmp_path):
    path = tmp_path / "subtitles.srt"
    write_srt(path, [{"start": 1.125, "end": 2.5, "text": "Hello world"}, {"start": 2.3, "end": 4, "text": "Next line"}])
    text = path.read_text()
    assert "00:00:01,125 --> 00:00:02,500" in text
    assert "00:00:02,500 --> 00:00:04,000" in text
    assert text.startswith("1\n")


def test_speech_verification_detects_missing_words():
    expected = "This movie takes us deep into space and asks a difficult question about humanity"
    assert spoken_coverage(expected, [{"text": expected}]) == 1
    assert spoken_coverage(expected, [{"text": "This movie takes us"}]) < 0.72


def test_trailer_rejects_fan_and_unverified_search_results():
    film = {"title": "Arrival"}
    base = {"title": "Arrival Official Trailer", "duration": 150, "channel_is_verified": True}
    assert candidate_ok(base, film)
    assert not candidate_ok({**base, "title": "Arrival Fan-Made Concept Trailer"}, film, curated=True)
    assert not candidate_ok({**base, "channel_is_verified": False}, film)
    assert not candidate_ok({**base, "title": "Interstellar Official Trailer"}, film)


def test_beat_avoids_promos_and_respects_bounds():
    cues = [{"start": 1, "end": 3, "text": "We have to go"}, {"start": 20, "end": 23, "text": "In theaters this summer"}, {"start": 30, "end": 33, "text": "We are running out of time"}]
    chosen = choose_beat(cues, [{"start": 29, "end": 34, "motion": 30}], 100, True)
    assert chosen["text"] == "We are running out of time"
