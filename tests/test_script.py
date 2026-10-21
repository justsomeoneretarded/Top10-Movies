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


def test_movie_length_variation_uses_best_draft_without_aborting():
    writer = Writer()
    writer.ask = Mock(return_value={"paragraphs": ["A short recommendation."]})
    assert writer.narration("Top 5", "Action", "rank 1: Example", {}, 216) == ["A short recommendation."]
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
    assert " ".join(result) == text
    assert all(len(p.split()) <= 65 for p in result)
    assert writer.ask.call_count == 1


def test_empty_movie_text_is_still_an_error():
    writer = Writer()
    writer.ask = Mock(return_value={"paragraphs": []})
    with pytest.raises(PipelineError, match="No usable narration text"):
        writer.narration("Top 5", "Action", "rank 1: Example", {}, 216)


FILMS = ["The Odyssey", "Spider-Man: Brand New Day", "Michael", "Toy Story 5", "Project Hail Mary"]


def test_intro_does_not_reveal_the_full_list_even_when_model_ignores_its_role():
    writer = Writer()
    bad = "Project Hail Mary is five. Toy Story 5 is four. Michael is three. Spider-Man: Brand New Day is two. The Odyssey is one. Which is your pick?"
    writer.ask = Mock(return_value={"paragraphs": [bad]})
    result = writer.narration("Top 5 Movies of 2026", "Action", "a punchy opening hook, without listing every title", {"titles":FILMS}, 65)
    assert writer.ask.call_count == 3
    assert not any(title in " ".join(result) for title in FILMS)
    assert "number 5" in " ".join(result)
    # The title list is kept for checking the response, not fed back as material to enumerate.
    assert all(title not in writer.ask.call_args_list[0].args[0] for title in FILMS)


def test_outro_is_a_goodbye_instead_of_a_second_countdown():
    writer = Writer()
    writer.ask = Mock(return_value={"paragraphs": ["Who's one? Let's count down. At number five, Project Hail Mary. Number four is Toy Story 5. Number three is Michael. Our movie list is about to begin."]})
    result = writer.narration("Top 5 Movies of 2026", "Action", CLOSING, {"titles":FILMS}, 35)
    text = " ".join(result)
    assert writer.ask.call_count == 3
    assert "subscribe" in text and "other recommendations" in text
    assert "Let's count down" not in text and "Who's one" not in text
    assert all(title not in text for title in FILMS)


def test_single_review_cannot_turn_into_the_whole_countdown():
    writer = Writer()
    writer.ask = Mock(return_value={"paragraphs": ["We're counting down the top five. Project Hail Mary is a space adventure. Next up, Toy Story 5 is family fun. Michael has music, then Spider-Man: Brand New Day brings action. Finally, The Odyssey is number one."]})
    facts = {"title":"Project Hail Mary", "rank":5, "overview":"A teacher wakes up aboard a spaceship with no memory of how he got there.",
             "other_titles":FILMS[:-1]}
    result = writer.narration("Top 5 Movies of 2026", "Keep it relaxed", "rank 5: Project Hail Mary", facts, 180)
    text = " ".join(result)
    assert writer.ask.call_count == 3
    assert "Project Hail Mary" in text and "teacher" in text
    assert all(title not in text for title in FILMS[:-1])
    assert all(len(paragraph.split()) <= 65 for paragraph in result)
    prompt = writer.ask.call_args_list[0].args[0]
    assert "SINGLE REVIEW" in prompt and all(title not in prompt for title in FILMS[:-1])


def test_one_movie_comparison_and_repeated_same_rank_remain_usable():
    writer = Writer()
    text = "Number five is Project Hail Mary. A teacher wakes up on a spaceship, far from home. This setup might appeal to fans of The Odyssey. At rank 5, the focus stays on this one adventure."
    writer.ask = Mock(return_value={"paragraphs":[text]})
    result = writer.narration("Top 5 Movies", "Relaxed", "rank 5: Project Hail Mary",
                              {"title":"Project Hail Mary", "rank":5, "other_titles":FILMS[:-1]}, 40)
    assert result == [text]
    assert writer.ask.call_count == 1
