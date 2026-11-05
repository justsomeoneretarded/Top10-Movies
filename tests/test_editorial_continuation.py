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


def test_episode_critique_rewrites_only_flagged_review_and_reuses_critique(tmp_path, monkeypatch):
    writer = Writer(tmp_path)
    calls, edits = [], []
    def ask(*args, **kwargs):
        calls.append(args[0])
        return {"edits": [{"section": 0, "reason": "Explain the pursuit directly."}]}
    def edit(*args, **kwargs):
        edits.append(kwargs)
        return ["Number 2: Second.", "After an attack, an agent escapes with the man he was guarding."], ["normal", "dramatic"]
    monkeypatch.setattr(writer, "ask", ask)
    monkeypatch.setattr(writer, "edit", edit)
    films = [{"title": "First", "overview": "A detective searches for her brother."},
             {"title": "Second", "overview": "An agent flees a safe house."}]
    original = episode()
    result = writer.polish_episode("Spy stories", "", films, original)
    writer.polish_episode("Spy stories", "", films, original)
    assert original == episode()
    assert result[0]["paragraphs"] != original[0]["paragraphs"]
    assert result[1] == original[1]
    assert len(calls) == 1
    assert len(edits) == 2
    assert edits[0]["editorial_note"] == "Explain the pursuit directly."


def test_invalid_critique_and_unchanged_rewrite_keep_narration(monkeypatch):
    writer = Writer()
    original = episode()
    films = [{"title": "First"}, {"title": "Second"}]
    monkeypatch.setattr(writer, "ask", lambda *a, **kw: {"edits": [
        {"section": -1, "reason": "invalid"}, {"section": True, "reason": "invalid"},
        {"section": 0, "reason": ""}]})
    assert writer.polish_episode("Topic", "", films, original) == original
    monkeypatch.setattr(writer, "ask", lambda *a, **kw: {"edits": [{"section": 0, "reason": "Try a clearer opening."}]})
    monkeypatch.setattr(writer, "edit", lambda *a, **kw: (deepcopy(original[0]["paragraphs"]), ["hype", "hype"]))
    assert writer.polish_episode("Topic", "", films, original) == original


def test_provider_line_is_removed_before_episode_rewrite(monkeypatch):
    writer = Writer()
    original = episode()
    original[0]["availability_text"] = "You can find it on Prime Video."
    original[0]["paragraphs"].append(original[0]["availability_text"])
    films = [{"title": "First"}, {"title": "Second"}]
    prompts = []
    def ask(prompt, *args, **kwargs):
        prompts.append(prompt)
        return {"edits": []}
    monkeypatch.setattr(writer, "ask", ask)
    result = writer.polish_episode("Topic", "", films, original)
    assert "Prime Video" not in prompts[0]
    assert all("Prime Video" not in p for section in result for p in section["paragraphs"])


def test_played_insert_breaks_continuation_without_forgetting_consumed_footage():
    path = Path("film.mp4")
    selector = selector_for([path])
    selector.reserve(path, 0, 5, "narration")
    history = deepcopy(selector._source_intervals)
    selector.break_continuation()
    assert not selector.continuation_orders(path, 5.18, 5)
    assert selector._source_intervals == history
    assert selector.freshness(path, 0, 5) == 0
