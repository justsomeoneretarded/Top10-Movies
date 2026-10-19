from unittest.mock import Mock

import pytest

from movie_pipeline.core import PipelineError
from movie_pipeline.script import Writer

CLOSING = "a short closing thought and one audience question"


def test_fifty_word_closing_does_not_abort_video():
    writer = Writer()
    text = " ".join(["word"] * 50) + "."
    writer.ask = Mock(return_value={"paragraphs": [text]})
    assert writer.narration("Top 5", "Action", CLOSING, {}, 35) == [text]
    assert writer.ask.call_count == 1


@pytest.mark.parametrize("draft", [{"paragraphs": []}, {"paragraphs": ["Too short."]}, {"paragraphs": ["word " * 200]}])
def test_invalid_closing_uses_bounded_fallback(draft):
    writer = Writer()
    writer.ask = Mock(return_value=draft)
    result = writer.narration("Top 5", "Action", CLOSING, {}, 35)
    assert writer.ask.call_count == 3
    assert 18 <= len(" ".join(result).split()) <= 65
    assert "countdown" in result[0]


def test_movie_sections_still_require_substantial_narration():
    writer = Writer()
    writer.ask = Mock(return_value={"paragraphs": ["A short recommendation."]})
    with pytest.raises(PipelineError):
        writer.narration("Top 5", "Action", "rank 1: Example", {}, 216)
    assert writer.ask.call_count == 3


def test_completed_section_is_reused_after_later_failure(tmp_path):
    first = Writer(cache_dir=tmp_path)
    text = " ".join(["word"] * 50) + "."
    first.ask = Mock(return_value={"paragraphs": [text]})
    first.narration("Top 5", "Action", CLOSING, {}, 35)
    resumed = Writer(cache_dir=tmp_path)
    resumed.ask = Mock(side_effect=AssertionError("Should use checkpoint"))
    assert resumed.narration("Top 5", "Action", CLOSING, {}, 35) == [text]
    resumed.ask.assert_not_called()


@pytest.mark.parametrize("draft", [{"paragraphs": []}, {"paragraphs": ["Too short."]}, {"paragraphs": ["word " * 200]}])
def test_invalid_opening_uses_fallback_instead_of_aborting(draft):
    writer = Writer()
    writer.ask = Mock(return_value=draft)
    result = writer.narration("Top 5", "Action", "a punchy opening hook, without listing every title", {}, 65)
    assert writer.ask.call_count == 3
    assert 25 <= len(" ".join(result).split()) <= 110


def test_opening_can_be_natural_length():
    writer = Writer()
    text = " ".join(["word"] * 90) + "."
    writer.ask = Mock(return_value={"paragraphs": [text]})
    result = writer.narration("Top 5", "Action", "a punchy opening hook, without listing every title", {}, 65)
    assert result == [text]
    assert writer.ask.call_count == 1
