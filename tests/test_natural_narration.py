"""Requested editorial changes: meaning-led writing and natural measured timing."""
from pathlib import Path
from unittest.mock import Mock

import numpy as np
import pytest

from movie_pipeline import cta
from movie_pipeline.script import Writer, OPENING, CLOSING, narration_material, plain_words
from movie_pipeline.sound_design import delivery_speed
from movie_pipeline.voice import thought_pause


def test_normalization_preserves_creative_words_and_titles():
    text = "Compelling evidence makes Significant Other unsettling."
    assert plain_words(text, protected=["Significant Other"]) == text


def test_story_material_is_distinct_from_metadata():
    facts = {"title": "Example", "overview": "Two friends lose their car.",
             "tagline": "A significant event.", "tmdb_rating": 8.1, "tmdb_votes": 20,
             "wiki_reception": "Remarkable chemistry.", "cast": ["An Actor as Sam"]}
    material = narration_material(facts)
    assert material == {"title": "Example", "overview": facts["overview"], "cast": facts["cast"]}
    assert facts["tagline"] == "A significant event."


def test_general_editor_can_shorten_far_beyond_old_percentage():
    writer = Writer()
    short = "Number 1: Example. Two friends lose their car."
    writer.ask = Mock(return_value={"paragraphs": [{"text": short, "delivery": "normal"}]})
    draft = ["Number 1: Example. "+"This offers a significant portrayal of remarkable themes. "*20]
    result, labels = writer.edit("Top movies", "Casual", "review", {"title": "Example"}, draft, rank=1)
    assert result == [short] and labels == ["normal"]
    assert len(short.split()) < len(draft[0].split())*.3
    prompt = writer.ask.call_args.args[0]
    assert "percentage" in prompt and "no minimum" in prompt
    assert "fact-check" not in prompt.casefold() and "spoiler" not in prompt.casefold()


def test_figures_names_and_plot_endings_are_not_factual_or_spoiler_gated():
    writer = Writer()
    draft = "Number 1: Example. In 2099, Ada finds the missing car and wins."
    writer.ask = Mock(return_value={"paragraphs": [draft]})
    assert writer.narration("Top movies", "Tell the story", "rank 1", {"title": "Example"}, 100) == [draft]
    assert writer.ask.call_count == 1
    prompt = writer.ask.call_args.args[0].casefold()
    assert "spoiler" not in prompt and "verified facts" not in prompt


def test_narration_uses_flexible_story_guidance_instead_of_rotating_templates():
    prompts = []
    writer = Writer()
    def ask(prompt, schema, **kwargs):
        prompts.append(prompt)
        return {"paragraphs": ["Number 1: Example. Two friends lose their car."]}
    writer.ask = ask
    writer.narration("Top movies", "Casual", "rank 1", {"title": "Example", "tagline": "MAGIC PHRASE"}, None)
    assert "Automatic length" in prompts[0]
    assert "MAGIC PHRASE" not in prompts[0]
    assert "Structure, in this order" not in prompts[0]
    assert "Choose the opening and order" in prompts[0]


@pytest.mark.parametrize("automatic", [True, False])
def test_auto_has_no_review_word_target_but_selected_minutes_remain_guidance(automatic):
    writer = Writer()
    writer.plan = lambda *args: {}
    targets = []
    def narration(topic, brief, label, facts, target):
        targets.append(target)
        return ["Number 1: Example. Two friends lose their car."]
    writer.narration = narration
    writer.edit = lambda topic, brief, role, facts, paragraphs, **kwargs: (paragraphs, ["normal"])
    result = writer.write("Top 2", "Casual", [{"title": "Example", "year": 2025}],
                          target_seconds=300, only={"film-0"}, automatic=automatic)
    assert targets == [None] if automatic else targets[0] > 0
    assert result["sections"][0]["automatic_runtime"] is automatic


@pytest.mark.parametrize("seconds,expected", [(12, 1.0), (18, 1.18), (7, .9)])
def test_recorded_pace_controls_adjustment_without_blind_multiplier(seconds, expected):
    assert delivery_speed(1.0, "normal", words=30, seconds=seconds) == expected


def test_already_fast_action_take_is_not_accelerated_again():
    assert delivery_speed(1.08, "hype", words=40, seconds=11) <= 1
    assert delivery_speed(1.08, "hype", words=30, seconds=11) == 1


def test_short_movie_announcements_preserve_articulation():
    assert delivery_speed(1.08, "hype", words=6, seconds=3) == 1


def test_phrase_pause_accounts_for_existing_boundary_silence_without_cutting_audio():
    rate = 1000
    voiced = np.ones(1000)*.1
    previous = np.r_[voiced, np.zeros(200)]
    following = np.r_[np.zeros(200), voiced]
    before = previous.copy(), following.copy()
    assert thought_pause(previous, following, rate, "One complete thought.") == 0
    assert .1 <= thought_pause(voiced, voiced, rate, "One complete thought.") <= .28
    assert np.array_equal(previous, before[0]) and np.array_equal(following, before[1])


def test_pause_follows_actual_phrase_duration():
    fast = np.ones(1000)*.1
    slow = np.ones(3000)*.1
    assert thought_pause(fast, fast, 1000, "One two three four five six.") < thought_pause(
        slow, fast, 1000, "One two three four five six.")


def test_bundled_subscribe_choices_are_more_than_three_reactions_and_eight_jokes(tmp_path):
    assert len(cta.REACTIONS) >= 9
    assert len({joke for reaction in cta.REACTIONS for joke in reaction["jokes"]}) >= 50
    for reaction in cta.REACTIONS:
        path = cta._asset(reaction, tmp_path)
        assert path.exists()
    assert len({cta._preferred("Movies", "Casual", str(i)) for i in range(100)}) > 3


def test_episode_specific_subscribe_line_is_used_before_backup_jokes(tmp_path):
    suggestion = {"theme": "approval", "text": "Like and subscribe. Keanu agrees that movie night needs better choices."}
    plan = cta.prepare_cta("Comedies", "Casual", "seed", tmp_path, suggestion=suggestion)
    assert plan["theme"] == "approval" and plan["text"] == suggestion["text"]
    assert cta.cta_variants(plan)[0] == plan
    assert len(cta.cta_variants(plan)) == 3


def test_invalid_optional_subscribe_suggestion_uses_a_complete_local_pair(tmp_path):
    plan = cta.prepare_cta("Movies", "", "seed", tmp_path,
                          suggestion={"theme": "missing", "text": "Something unrelated."})
    reaction = next(row for row in cta.REACTIONS if row["theme"] == plan["theme"])
    assert plan["text"] in reaction["jokes"]
    assert plan["path"]


def test_end_screen_shortens_without_changing_final_hold():
    from inspect import signature
    from movie_pipeline.cli import END_SCREEN_SECONDS
    from movie_pipeline.render import finish
    assert END_SCREEN_SECONDS == 10
    assert signature(finish).parameters["ending_hold"].default == 3



def test_stale_auto_checkpoint_migrates_without_old_word_quota():
    writer = Writer()
    writer.narration = Mock(return_value=["Number 1: Example. Two friends lose their car."])
    writer.edit = lambda *args, **kwargs: (args[4], ["normal"])
    draft = {"sections": [{"kind": "film", "film": 0, "paragraphs": ["Old review."]}],
             "package": {"script_policy_version": 1}}
    result = writer.refresh_editorial("Top movies", "Casual",
                                      [{"title": "Example", "year": 2025}], draft, automatic=True)
    assert result["sections"][0]["automatic_runtime"] is True
    assert writer.narration.call_args.args[4] is None
    assert "automatic_runtime" not in draft["sections"][0]


def test_exceptionally_long_sentence_is_bounded_without_losing_words():
    from movie_pipeline.voice import phrases
    text = " ".join(f"word{i}" for i in range(245))+"."
    chunks = phrases([text])
    assert all(len(chunk.split()) <= 90 for chunk in chunks)
    assert " ".join(chunks) == text


def test_exceptionally_long_sentence_prefers_existing_clause_boundaries():
    from movie_pipeline.voice import phrases
    clause = " ".join(["one"]*50)+","
    text = clause+" "+" ".join(["two"]*50)+"."
    assert phrases([text]) == [clause, " ".join(["two"]*50)+"."]


def test_episode_edit_includes_opening_and_closing_in_context_and_removes_repetition():
    writer = Writer()
    writer.ask = Mock(return_value={"remove": ["0:0:1", "1:0:1", "2:0:0"]})
    sections = [
        {"kind": "intro", "paragraphs": ["Three tense thrillers for your next movie night. These thrillers are tense."], "delivery": ["hype"]},
        {"kind": "film", "film": 0, "paragraphs": ["Number 1: Example. Tension fills every scene. A spy must rescue her brother."], "delivery": ["normal"]},
        {"kind": "outro", "paragraphs": ["These movies are tense. Which would you watch? Like and subscribe."], "delivery": ["calm"]},
    ]
    result = writer.episode_edit("Top thrillers", "Casual", sections)
    prompt = writer.ask.call_args.args[0]
    assert '"kind": "intro"' in prompt and '"kind": "outro"' in prompt
    assert result[0]["paragraphs"] == ["Three tense thrillers for your next movie night."]
    assert result[1]["paragraphs"] == ["Number 1: Example. A spy must rescue her brother."]
    assert result[2]["paragraphs"] == ["Which would you watch? Like and subscribe."]
    assert result[0]["delivery"] == ["hype"] and result[2]["delivery"] == ["calm"]
    assert sections[0]["paragraphs"][0].endswith("These thrillers are tense.")


def test_episode_edit_keeps_whole_intro_if_requested_to_delete_it_and_preserves_outro_cta():
    writer = Writer()
    writer.ask = Mock(return_value={"remove": ["0:0:0", "0:0:1", "1:0:0", "1:0:1", "2:0:0", "2:0:1"]})
    sections = [
        {"kind": "intro", "paragraphs": ["Watch these thrillers. They all pack a punch."]},
        {"kind": "film", "film": 0, "paragraphs": ["Number 1: Example. A spy goes undercover."]},
        {"kind": "outro", "paragraphs": ["Which would you watch? Like and subscribe."]},
    ]
    result = writer.episode_edit("Top thrillers", "", sections)
    assert result[0]["paragraphs"] == sections[0]["paragraphs"]
    assert result[1]["paragraphs"] == sections[1]["paragraphs"]
    assert result[2]["paragraphs"] == sections[2]["paragraphs"]


def test_episode_edit_leaves_non_editorial_sections_untouched_and_fails_open():
    writer = Writer()
    writer.ask = Mock(side_effect=ValueError("model unavailable"))
    sections = [
        {"kind": "intro", "paragraphs": ["One. Two."]},
        {"kind": "film", "film": 0, "paragraphs": ["Number 1: Example. Three. Four."]},
        {"kind": "outro", "paragraphs": ["Five. Six."]},
        {"kind": "subscribe", "paragraphs": ["Do not touch."]},
    ]
    assert writer.episode_edit("Top", "", sections) == sections


def test_restored_script_runs_full_episode_edit_once_without_rewriting_reviews(monkeypatch):
    from copy import deepcopy
    from movie_pipeline import script as script_module, packaging
    writer = Writer()
    writer.narration = Mock(side_effect=AssertionError("complete drafts should be reused"))
    writer.episode_edit = Mock(side_effect=lambda title, brief, sections: [
        {**sections[0], "paragraphs": ["A clear opening."]}, *deepcopy(sections[1:])
    ])
    monkeypatch.setattr(script_module, "is_fallback_opening", lambda *a, **kw: False)
    monkeypatch.setattr(script_module, "_section_issue", lambda *a, **kw: False)
    monkeypatch.setattr(script_module, "_editorial_issue", lambda *a, **kw: False)
    monkeypatch.setattr(packaging, "refresh_availability", lambda script, films: script)
    draft = {"sections": [
        {"kind": "intro", "paragraphs": ["A clear opening. Redundant opening."]},
        {"kind": "film", "film": 0, "paragraphs": ["Number 1: Example. A spy escapes."]},
        {"kind": "outro", "paragraphs": ["Which one? Like and subscribe."]},
    ], "package": {"script_policy_version": script_module.SCRIPT_POLICY_VERSION}}
    films = [{"title": "Example", "year": 2025}]
    refreshed = writer.refresh_editorial("Movies", "Casual", films, draft)
    assert refreshed["sections"][0]["paragraphs"] == ["A clear opening."]
    assert refreshed["sections"][1]["paragraphs"] == draft["sections"][1]["paragraphs"]
    assert refreshed["package"]["full_episode_edit_version"] == 1
    assert draft["sections"][0]["paragraphs"][0].endswith("Redundant opening.")
    assert writer.episode_edit.call_count == 1
    assert writer.refresh_editorial("Movies", "Casual", films, refreshed) == refreshed
    assert writer.episode_edit.call_count == 1
