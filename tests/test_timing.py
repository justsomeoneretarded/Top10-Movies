import pytest

from movie_pipeline.core import PipelineError
from movie_pipeline.timing import (
    DurationSpec,
    narration_words,
    requested_duration,
    trailer_budget,
)


@pytest.mark.parametrize("brief, seconds", [
    ("Make a 5-minute video. Use Warguy.", 300),
    ("Duration: 5 min", 300), ("Make a five minute video", 300),
    ("Aim for 8–10 minutes", 540), ("Make a 15-minute video", 900),
    ("Make a 1.5 minute video", 90), ("Make a 1-hour video", 3600),
    ("Duration: 45 seconds", 45), ("A 30-second video", 30),
    ("Pause for 3 seconds during each trailer beat", 540),
])
def test_duration_from_brief(brief, seconds):
    assert requested_duration(brief).target == seconds


def test_default_preserves_original_eight_to_ten_minute_range():
    assert requested_duration("Cinematic and energetic") == DurationSpec()


def test_conflicting_durations_rejected_instead_of_silently_ignoring_one():
    with pytest.raises(PipelineError, match="conflicting"):
        requested_duration("Make a 5 minute video. Aim for 8-10 minutes.")


def test_shorter_duration_gets_shorter_total_script():
    five = narration_words(300, 5, 15)
    fifteen = narration_words(900, 5, 15)
    assert five[1] < fifteen[1]
    assert sum((five[0], five[1]*5, five[2])) < sum((fifteen[0], fifteen[1]*5, fifteen[2]))


def test_tiny_video_does_not_spend_its_duration_on_trailer_breaks():
    assert trailer_budget(requested_duration("30-second video"), 5) == 0
