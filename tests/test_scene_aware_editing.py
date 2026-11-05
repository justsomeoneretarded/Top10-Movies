from pathlib import Path

import numpy as np
import pytest

from movie_pipeline.footage_context import collect_visual_hints
from movie_pipeline.moments import _cuts_per_minute, _match_inserts, _usable, candidates, complete_window


def test_cut_density_counts_boundaries_and_remains_soft():
    shots = {
        "calm.mp4": [{"start": 0, "end": 9}],
        "fast.mp4": [{"start": 0, "end": 3}, {"start": 3, "end": 6}, {"start": 6, "end": 9}],
    }

    class Selector:
        features = {"calm.mp4": np.array([[1., 0.]]),
                    "fast.mp4": np.array([[1., 0.], [1., 0.], [1., 0.]])}

        def identity_relevance(self, source, query):
            return np.full(len(shots[str(source)]), .5)

    result = candidates(Selector(), [Path("fast.mp4"), Path("calm.mp4")], shots, {"title": "Film"})
    calm = next(x for x in result if x["source"] == "calm.mp4")
    fast = next(x for x in result if x["source"] == "fast.mp4" and x["duration"] == 9)
    assert fast["cuts_per_minute"] == pytest.approx(120 / 9)
    assert calm["score"] > fast["score"]
    assert fast in result
    assert _cuts_per_minute(shots["fast.mp4"], 5, 5) == 0


def test_expansion_updates_cut_density_and_empty_cues_are_not_verified_dialogue():
    shots = {"film.mp4": [{"start": 0, "end": 4}, {"start": 4, "end": 8}]}
    clip = {"source": "film.mp4", "start": 0, "duration": 8, "cuts_per_minute": 0}
    assert _usable(object(), clip, shots)
    assert clip["cuts_per_minute"] == pytest.approx(7.5)
    result = complete_window(clip, [], (0, 8))
    assert result is not None
    assert not result["dialogue_boundaries_verified"]


def test_insert_placement_prefers_stronger_paragraph_match_and_falls_back():
    shots = {"film.mp4": [{"start": 0., "end": 5.}, {"start": 25., "end": 30.}]}
    clips = [{"source": "film.mp4", "start": 0., "duration": 5.},
             {"source": "film.mp4", "start": 25., "duration": 5.}]

    class Selector:
        def identity_relevance(self, path, text):
            return np.array([.1, .8]) if text == "chase" else np.array([.8, .1])

    class Flat:
        def identity_relevance(self, path, text):
            return np.array([.25, .25])

    assert _match_inserts(Selector(), clips, [0, 1], ["chase", "talk"], shots) == clips[::-1]
    assert _match_inserts(Flat(), clips, [0, 1], ["chase", "talk"], shots) == clips
    assert _match_inserts(object(), clips, [0, 1], ["chase", "talk"], shots) == clips


def test_insert_matching_does_not_reverse_a_nearby_source_sequence():
    shots = {"film.mp4": [{"start": 0., "end": 5.}, {"start": 5., "end": 10.}]}
    clips = [{"source": "film.mp4", "start": 0., "duration": 5.},
             {"source": "film.mp4", "start": 5., "duration": 5.}]

    class Selector:
        def identity_relevance(self, path, text):
            return np.array([.1, .8]) if text == "chase" else np.array([.8, .1])

    assert _match_inserts(Selector(), clips, [0, 1], ["chase", "talk"], shots) == clips


def test_visual_hints_are_bounded_deduplicated_and_isolated_per_title(monkeypatch):
    from movie_pipeline import footage_context

    data = {
        "a.mp4": {"descriptions": [" a car   chase ", "A CAR CHASE", "rainy street", "x" * 150]},
        "b.mp4": {"descriptions": ["quiet library"]},
    }
    monkeypatch.setattr(footage_context.scenes, "load", lambda source, count: data.get(str(source)))
    pools = [[Path("a.mp4")], [Path("b.mp4")], [Path("missing.mp4")]]
    lookup = {str(path): [{"start": 0, "end": 1}] for pool in pools for path in pool}
    hints = collect_visual_hints(pools, lookup)
    assert hints == [["a car chase", "rainy street", "x" * 120], ["quiet library"], []]


def test_visual_hints_change_only_the_relevant_narration_cache(tmp_path, monkeypatch):
    from movie_pipeline.script import Writer

    writer = Writer(tmp_path)
    calls = []
    def ask(prompt, schema, **kwargs):
        calls.append(prompt)
        return {"paragraphs": ["Number 1: Film. An agent is followed through the city."]}
    monkeypatch.setattr(writer, "ask", ask)
    facts = {"title": "Film", "rank": 1, "overview": "An agent is followed."}
    hints = {**facts, "visual_hints": ["A person runs through a dark street"]}
    first = writer.narration("Spy movies", "", "rank 1: Film", facts, None)
    assert writer.narration("Spy movies", "", "rank 1: Film", facts, None) == first
    writer.narration("Spy movies", "", "rank 1: Film", hints, None)
    writer.narration("Spy movies", "", "rank 1: Film", hints, None)
    assert len(calls) == 2
    assert "uncertain single-frame guesses" in calls[1]
