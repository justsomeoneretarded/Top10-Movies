from copy import deepcopy
from pathlib import Path

import numpy as np
import pytest

from movie_pipeline.shots import ShotSelector
from movie_pipeline.source_pool import choose_from_sources
from movie_pipeline.script import Writer


def selector_for(paths):
    selector = object.__new__(ShotSelector)
    selector.uses, selector.last_shots, selector.cut_count = {}, {}, 0
    selector.features = {str(path): np.array([[1., 0.]]) for path in paths}
    selector._scene_ranges = {str(path): [{"start": 0., "end": 20., "motion": 25}] for path in paths}
    selector._scene_hashes = {str(path): [0 if i == 0 else (1 << 64) - 1] for i, path in enumerate(paths)}
    selector.relevance = lambda path, text: np.array([.5])
    selector.card_kind = lambda *args: None
    return selector


def test_unused_second_and_third_slices_are_not_visual_repeats():
    path = Path("film.mp4")
    selector = selector_for([path])
    selector.reserve(path, 0, 5, "narration")
    assert selector.visual_freshness(path, 5.18, 5) == 1
    assert selector.visual_reuse_penalty(path, 5.18, 5) == 0
    selector.reserve(path, 5.18, 5, "narration")
    assert selector.visual_freshness(path, 10.36, 5) == 1
    assert selector.visual_freshness(path, 4, 5) == 0
    assert not selector.continuation_orders(path, 19, 3)


def test_other_source_and_native_scene_duplicates_stay_blocked():
    path, other = Path("film.mp4"), Path("extra.mp4")
    selector = selector_for([path, other])
    selector._scene_hashes[str(other)] = [0]
    selector.reserve(other, 0, 5, "narration")
    selector.reserve(path, 0, 5, "narration")
    assert selector.visual_freshness(path, 5.18, 5) == 0
    assert selector.visual_reuse_penalty(path, 5.18, 5) > 0
    selector = selector_for([path])
    selector.reserve(path, 0, 5, "native-scene")
    assert selector.visual_freshness(path, 5.18, 5) == 0


def test_intervening_source_does_not_count_as_immediate_continuation():
    path, other = Path("film.mp4"), Path("extra.mp4")
    selector = selector_for([path, other])
    selector.reserve(path, 0, 5, "narration")
    selector.reserve(other, 0, 5, "narration")
    assert not selector.continuation_orders(path, 5.18, 5)


def test_source_pool_keeps_unused_same_shot_continuation(tmp_path):
    primary, extra = tmp_path / "primary.mp4", tmp_path / "extra.mp4"
    primary.write_bytes(b"source")
    extra.write_bytes(b"source")
    selector = selector_for([primary, extra])
    lookup = selector._scene_ranges
    first = choose_from_sources(selector, [primary, extra], lookup, "A pursuit.", 5, False)
    second = choose_from_sources(selector, [primary, extra], lookup, "The pursuit continues.", 5, False)
    assert first[0]["source"] == second[0]["source"] == str(primary)
    assert second[0]["start"] == pytest.approx(5.18)
    assert sum(c["duration"] for c in second) == pytest.approx(5)
    assert first[0]["start"] + first[0]["duration"] <= second[0]["start"]


def episode():
    return [
        {"kind": "film", "film": 1, "paragraphs": ["Number 2: Second.", "An agent flees a safe house."], "delivery": ["normal", "dramatic"]},
        {"kind": "film", "film": 0, "paragraphs": ["Number 1: First.", "A detective searches for her brother."], "delivery": ["normal", "calm"]},
    ]


def test_writer_keeps_section_edit_and_deletion_pass_without_extra_rewrites():
    writer = Writer()
    assert callable(writer.edit) and callable(writer.episode_edit)
    assert not hasattr(writer, "polish_episode")


def test_played_insert_breaks_continuation_without_forgetting_consumed_footage():
    path = Path("film.mp4")
    selector = selector_for([path])
    selector.reserve(path, 0, 5, "narration")
    history = deepcopy(selector._source_intervals)
    selector.break_continuation()
    assert not selector.continuation_orders(path, 5.18, 5)
    assert selector._source_intervals == history
    assert selector.freshness(path, 0, 5) == 0


def test_full_write_uses_baseline_edit_pass_without_experimental_polish(monkeypatch):
    from unittest.mock import Mock
    writer = Writer()
    films = [{"title": "First", "year": 2026, "overview": "A detective searches for her brother."},
             {"title": "Second", "year": 2025, "overview": "An agent escapes a safe house."}]
    writer.plan = lambda *args: {}
    writer.narration = lambda topic, brief, label, facts, words: [
        facts.get("overview", "Two stories worth your time.")]
    writer.edit = lambda *args, **kwargs: (args[4], ["normal"] * len(args[4]))
    writer.bridges = lambda *args: {}
    writer.thumbnail_concepts = lambda *args: []
    writer.polish_episode = Mock(side_effect=AssertionError("extra rewriting is disabled"))
    writer.episode_edit = Mock(side_effect=lambda topic, brief, sections: sections)
    writer.ask = lambda *args, **kwargs: {
        "titles": ["Two Spy Movies Worth Watching", "2 Great Spy Movies to Watch Tonight"],
        "hook": "Two spy movies.", "description": "Two tense stories.",
        "thumbnail_texts": ["TWO GREAT PICKS", "SPY MOVIES", "WATCH TONIGHT"],
        "tags": ["spy movies"], "subscribe": {"theme": "default", "text": "Like and subscribe."}}
    result = writer.write("Two Spy Movies Worth Watching", "", films, automatic=True)
    writer.polish_episode.assert_not_called()
    writer.episode_edit.assert_called_once()
    reviews = [section for section in result["sections"] if section["kind"] == "film"]
    assert len(reviews) == 2
    assert "agent escapes" in " ".join(reviews[0]["paragraphs"]).lower()
    assert all("planned_scene_ids" not in section for section in result["sections"])
