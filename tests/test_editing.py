import math
from itertools import pairwise
from pathlib import Path
from unittest.mock import Mock

import numpy as np
import pytest

from movie_pipeline.shots import ShotSelector, intro_montage


class FakeSelector:
    def __init__(self):
        self.prompts = []

    def hook_score(self, path, shots, prompt, action):
        self.prompts.append(prompt)
        return int(path.stem)

    def choose(self, path, shots, prompt, seconds, action, maximum_shot=None):
        return [{"start": 50.0, "duration": seconds, "freeze": 0}]


@pytest.mark.parametrize("count, seconds", [(2, [10]), (5, [12]), (8, [1, 2]), (15, [0.7, 0.6])])
def test_intro_covers_at_least_half_of_movies_across_paragraphs(count, seconds):
    selector = FakeSelector()
    paths = [Path(f"{i}.mp4") for i in range(count)]
    plans = intro_montage(selector, paths, [[{"start": 50, "end": 55, "motion": 20}]]*count, seconds, "Top action movies", True)
    assert len({cut["film_index"] for plan in plans for cut in plan}) >= math.ceil(count/2)
    flattened = [cut for plan in plans for cut in plan]
    assert all(a["source"] != b["source"] for a, b in pairwise(flattened))
    for plan, duration in zip(plans, seconds):
        assert sum(round(c["duration"]*30) for c in plan) == math.ceil(duration*30)
    assert "fight" in selector.prompts[0]
    assert flattened[0]["start"] == 50  # A trailer's middle can open the video.


def bare_selector(scores):
    selector = object.__new__(ShotSelector)
    selector.relevance = Mock(return_value=np.array(scores))
    selector.uses = {}
    selector.last_shots = {}
    selector.cut_count = 0
    return selector


def test_shots_can_start_in_middle_and_are_not_played_as_original_sequence():
    selector = bare_selector([0.9, 0.8, 1.0])
    shots = [{"start": 0, "end": 4, "motion": 5}, {"start": 4, "end": 8, "motion": 5}, {"start": 40, "end": 44, "motion": 5}]
    cuts = selector.choose(Path("film.mp4"), shots, "cinematic action", 6, True)
    assert cuts[0]["start"] == 40
    assert cuts[1]["start"] == 0
    assert sum(c["duration"] for c in cuts) == pytest.approx(6)
    assert cuts[0]["freeze"] > 0
    assert cuts[1]["freeze"] == 0


def test_next_paragraph_avoids_replaying_the_same_or_sequential_source_shot():
    selector = bare_selector([1.0, 0.99, 0.9])
    shots = [{"start": 0, "end": 4, "motion": 5}, {"start": 4, "end": 8, "motion": 5}, {"start": 40, "end": 44, "motion": 5}]
    selector.choose(Path("film.mp4"), shots, "action", 2, True)
    second = selector.choose(Path("film.mp4"), shots, "action", 2, True)
    assert second[0]["start"] == 40
