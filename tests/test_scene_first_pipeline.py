from copy import deepcopy
from pathlib import Path
from types import SimpleNamespace
from concurrent.futures import ThreadPoolExecutor
import json
import threading
import time

import numpy as np
import pytest
from PIL import Image

from movie_pipeline import scene_plan, scenes, moments
from movie_pipeline.shots import ShotSelector
from movie_pipeline.source_pool import choose_from_sources
from movie_pipeline.download_access import youtube_dl
from movie_pipeline.script import Writer
from movie_pipeline.rank_intro import prepare_rank_intros


def tagged():
    return [{"start": float(i*12), "end": float((i+1)*12), "motion": 25,
             "detected_shot": 0, "continues_previous": i > 0} for i in range(3)]


def bare():
    selector = object.__new__(ShotSelector)
    selector.uses, selector.last_shots, selector.cut_count = {}, {}, 0
    selector.card_kind = lambda *args: None
    selector.relevance = lambda path, text: np.full(len(selector._scene_ranges[str(path)]), .8)
    return selector


def test_artificial_windows_join_without_reindexing():
    selector, path, rows = bare(), Path("film.mp4"), tagged()
    before = deepcopy(rows)
    assert selector.continuous_bounds(path, rows, 1) == (0., 36.)
    assert selector.continuous_end(path, rows, 0) == 22.
    assert rows == before
    assert moments._cuts_per_minute(rows, 0, 30) == 0


@pytest.mark.parametrize("boundary", ["cut", "gap", "card", "legacy"])
def test_join_stops_at_real_boundaries(boundary):
    selector, rows = bare(), tagged()
    if boundary == "cut":
        rows[1]["detected_shot"] = 1
    elif boundary == "gap":
        rows[1]["start"] = 12.25
    elif boundary == "card":
        rows[1]["text_card"] = "promo"
    else:
        for row in rows:
            row.pop("detected_shot")
            row.pop("continues_previous")
    assert selector.continuous_end(Path("film.mp4"), rows, 0) == 12.


def test_original_shot_continues_after_crossing_artificial_boundary():
    selector, path = bare(), Path("film.mp4")
    selector._scene_ranges = {str(path): tagged()}
    selector._scene_hashes = {str(path): [0, 1, 2]}
    selector.reserve(path, 0, 16, "narration")
    assert selector.continuation_orders(path, 16.18, 5)
    assert selector.visual_freshness(path, 16.18, 5) == 1


def test_pool_does_not_force_switch_at_artificial_boundary(tmp_path):
    primary, extra = tmp_path/"film.mp4", tmp_path/"extra.mp4"
    for path in (primary, extra):
        path.write_bytes(b"media")
    selector = bare()
    selector._scene_ranges = {str(primary): tagged(), str(extra): [{"start": 0., "end": 8., "motion": 25}]}
    selector._scene_hashes = {str(primary): [0, 1, 2], str(extra): [(1<<64)-1]}
    selector.features = {str(primary): np.ones((3, 2)), str(extra): np.ones((1, 2))}
    selector.relevance = lambda path, text: np.full(len(selector._scene_ranges[str(path)]), .8 if path == primary else .3)
    cuts = choose_from_sources(selector, [primary, extra], selector._scene_ranges, "A pursuit.", 16, False)
    assert len(cuts) == 1 and cuts[0]["source"] == str(primary)
    assert cuts[0]["duration"] == pytest.approx(16)


def map_fixture(monkeypatch, tmp_path):
    source = tmp_path/"film.mp4"
    source.write_bytes(b"original source bytes")
    lookup = {str(source): [{"start": 0., "end": 6.}, {"start": 6., "end": 12.}]}
    films = [{"title": "Example", "year": 2026, "kind": "movie", "id": 7}]
    monkeypatch.setattr(scene_plan.scenes, "load", lambda *args: {"descriptions": ["A car chase", "Two people argue"]})
    calls = []
    def enrich(candidate, dialogue, recognizer, work):
        assert work.is_dir()
        calls.append(candidate["start"])
        return {**candidate, "muted": False, "native_verified": True, "dialogue_boundaries_verified": True,
                "cues": [{"start": candidate["start"]+.1, "end": candidate["start"]+candidate["duration"]-.1, "text": "Run now."}]}
    monkeypatch.setattr(scene_plan, "_enrich", enrich)
    return source, lookup, films, calls


def test_scene_map_precedes_writing_and_reuses_exact_source_cache(monkeypatch, tmp_path):
    source, lookup, films, calls = map_fixture(monkeypatch, tmp_path)
    plan = scene_plan.prepare_scene_map([[source]], lookup, films, tmp_path/"plan", lambda: object())
    assert len(plan["films"]["0"]) == len(calls) == 2
    context = scene_plan.projected_context(plan)
    assert context[0][0]["id"] == plan["films"]["0"][0]["id"]
    assert "source" not in context[0][0]
    again = scene_plan.prepare_scene_map([[source]], lookup, films, tmp_path/"plan",
                                        lambda: pytest.fail("cached map must avoid ASR setup"))
    assert again["films"] == plan["films"] and len(calls) == 2
    source.write_bytes(b"changed source bytes")
    assert scene_plan.validated_priorities(plan, 0, [source], lookup, films[0]) == []


def test_scene_map_failures_and_zero_budget_are_optional(monkeypatch, tmp_path):
    source, lookup, films, calls = map_fixture(monkeypatch, tmp_path)
    empty = scene_plan.prepare_scene_map([[source]], lookup, films, tmp_path/"empty",
        lambda: pytest.fail("zero budget must not load a model"), budget_seconds=0)
    assert not empty["films"].get("0")
    setups = []
    def down():
        setups.append(1)
        raise RuntimeError("model unavailable")
    failed = scene_plan.prepare_scene_map([[source]], lookup, films, tmp_path/"down", down)
    assert len(setups) == 1 and not failed["films"]["0"]


def test_scene_map_rejects_wrong_film_promos_and_bad_context(monkeypatch, tmp_path):
    source, lookup, films, calls = map_fixture(monkeypatch, tmp_path)
    plan = scene_plan.prepare_scene_map([[source]], lookup, films, tmp_path/"plan", lambda: object())
    assert scene_plan.validated_priorities(plan, 0, [source], lookup, {"title": "Wrong"}) == []
    altered = deepcopy(lookup)
    for row in altered[str(source)]:
        row["text_card"] = "promo"
    assert scene_plan.validated_priorities(plan, 0, [source], altered, films[0]) == []
    assert scene_plan.projected_context({"film_count": 1, "films": []}) == []


def test_final_scene_selection_honors_preselected_ids(monkeypatch, tmp_path):
    source, lookup, films, calls = map_fixture(monkeypatch, tmp_path)
    plan = scene_plan.prepare_scene_map([[source]], lookup, films, tmp_path/"plan", lambda: object())
    selected = plan["films"]["0"][0]
    class Selector:
        def identity_relevance(self, path, query):
            return np.full(2, .1)
    monkeypatch.setattr(moments, "_enrich", lambda *args: None)
    monkeypatch.setattr("movie_pipeline.core.duration", lambda path: 10)
    result = moments.plan_episode(Selector(), [[source]], lookup, films,
        [{"kind": "film", "film": 0, "paragraphs": ["A specific story."], "planned_scene_ids": [selected["id"]]}],
        {(0, 0): Path("speech.wav")}, {}, None, tmp_path/"final", scene_map=plan)
    assert result["previews"][0]["id"] == selected["id"]
    assert result["report"]["planned_scenes_used"] == 1


def test_scene_context_affects_writing_without_speaking_ids(monkeypatch):
    writer = Writer()
    captured = []
    writer.plan = lambda *args: {}
    def narration(topic, brief, label, facts, target):
        captured.append(deepcopy(facts))
        return ["Number 1: Example. An agent must get home."]
    writer.narration = narration
    writer.edit = lambda *args, **kwargs: (args[4], ["normal"])
    context = [[{"id": "clip-secret-id", "description": "A chase", "duration": 6., "dialogue_excerpt": "Run."}]]
    films = [{"title": "Example", "year": 2026}]
    script = writer.write("Movies", "", films, only={"film-0"}, scene_context=context)
    prepared = prepare_rank_intros(script, films)
    assert captured[0]["available_scene_moments"] == context[0]
    assert prepared["sections"][0]["planned_scene_ids"] == ["clip-secret-id"]
    assert "clip-secret-id" not in " ".join(prepared["sections"][0]["paragraphs"])


def test_downloader_setup_serialized_but_extraction_stays_parallel():
    lock, barrier = threading.Lock(), threading.Barrier(2)
    active = peak = 0
    instances = []
    class Downloader:
        def __init__(self, options):
            nonlocal active, peak
            with lock:
                active += 1
                peak = max(peak, active)
                instances.append(self)
            time.sleep(.02)
            with lock:
                active -= 1
        def __enter__(self):
            return self
        def __exit__(self, *args):
            pass
        def extract_info(self, url):
            barrier.wait(timeout=3)
            return url
    module = SimpleNamespace(YoutubeDL=Downloader)
    def worker(index):
        with youtube_dl(module, {"worker": index}) as downloader:
            return downloader.extract_info(str(index))
    with ThreadPoolExecutor(max_workers=2) as executor:
        assert list(executor.map(worker, range(2))) == ["0", "1"]
    assert peak == 1 and len(instances) == 2 and instances[0] is not instances[1]


def test_downloader_failure_releases_lock_and_options_are_private():
    calls = []
    class Downloader:
        def __init__(self, options):
            calls.append(options)
            options["changed"] = True
            if len(calls) == 1:
                raise RuntimeError("setup unavailable")
    module, options = SimpleNamespace(YoutubeDL=Downloader), {"quiet": True}
    with pytest.raises(RuntimeError):
        youtube_dl(module, options)
    assert isinstance(youtube_dl(module, options), Downloader)
    assert options == {"quiet": True}


@pytest.mark.parametrize("payload", [
    {"response": "", "thinking": "private reasoning", "done_reason": "length"},
    {"response": "<think>private unfinished reasoning"},
    {"response": None}, {"response": {"caption": "not final text"}},
])
def test_vision_rejects_reasoning_and_malformed_answers(monkeypatch, capsys, payload):
    class Response:
        def raise_for_status(self):
            pass
        def json(self):
            return payload
    monkeypatch.setattr(scenes, "_VISION_READY", False)
    monkeypatch.setattr(scenes.requests, "post", lambda *args, **kwargs: Response())
    assert scenes._describe(Image.new("RGB", (64, 64))) == ""
    assert not scenes._VISION_READY
    assert "private" not in capsys.readouterr().out


def test_vision_uses_final_answer_and_checks_model_cache(tmp_path):
    assert scenes._answer_text({"response": "<think>private</think>A car enters a tunnel"}) == "A car enters a tunnel"
    path = tmp_path/"film.mp4"
    scenes.cache_path(path).write_text(json.dumps({"version": scenes.VERSION, "model": "old-thinking",
        "embed_model": scenes.EMBED_MODEL, "descriptions": ["A car"], "embeddings": [[1.]]}))
    assert scenes.load(path, 1) is None


@pytest.mark.parametrize("bad", [[], None, {"version": scenes.VERSION, "model": scenes.VISION_MODEL,
    "embed_model": scenes.EMBED_MODEL, "descriptions": None, "embeddings": []}])
def test_malformed_vision_cache_is_optional(tmp_path, bad):
    path = tmp_path/"film.mp4"
    scenes.cache_path(path).write_text(json.dumps(bad))
    assert scenes.load(path, 1) is None


def test_corrupt_preselected_cues_fall_back(monkeypatch, tmp_path):
    source, lookup, films, calls = map_fixture(monkeypatch, tmp_path)
    plan = scene_plan.prepare_scene_map([[source]], lookup, films, tmp_path/"plan", lambda: object())
    for clip in plan["films"]["0"]:
        clip["cues"] = [{"start": "bad", "text": "broken"}]
    assert scene_plan.validated_priorities(plan, 0, [source], lookup, films[0]) == []


def test_portrait_hold_stays_in_its_indexed_frame_window():
    selector, path = bare(), Path("film.mp4")
    rows = tagged()
    selector._scene_ranges = {str(path): rows}
    selector.faces = {str(path): [True, False, False]}
    selector.people = {str(path): ["Example Actor"]}
    cuts = selector.choose(path, rows, "Example Actor plays the agent.", 5, False)
    assert cuts[0]["still"]
    assert 0 <= cuts[0]["start"] <= 12
