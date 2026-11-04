import json
from types import SimpleNamespace

import numpy as np
import pytest
from PIL import Image

from movie_pipeline import scenes
from movie_pipeline.shots import ShotSelector, _zscore


@pytest.fixture(autouse=True)
def reset_scene_budget(monkeypatch):
    monkeypatch.setattr(scenes, "_DEADLINE", None)
    monkeypatch.setattr(scenes, "_READY", False)
    monkeypatch.setattr(scenes, "_UNAVAILABLE", False)
    monkeypatch.setattr(scenes, "_VISION_READY", False)

class FakeResponse:
    def __init__(self, data):
        self.data = data

    def raise_for_status(self):
        pass

    def json(self):
        return self.data


def fake_ollama(calls):
    def post(url, json=None, timeout=None):
        calls.append((url.rsplit("/", 1)[-1], json.get("model")))
        if url.endswith("/api/generate"):
            assert json["images"]
            return FakeResponse({"response": "close-up of an astronaut, tense, red cockpit light"})
        if url.endswith("/api/embed"):
            return FakeResponse({"embeddings": [[1.0, float(i)] for i, _ in enumerate(json["input"])]})
        return FakeResponse({})
    return post


def test_descriptions_are_generated_once_and_cached(monkeypatch, tmp_path):
    calls = []
    monkeypatch.setattr(scenes.requests, "post", fake_ollama(calls))
    monkeypatch.setattr("movie_pipeline.shots.get_frame", lambda path, seconds: Image.new("RGB", (1280, 720), "red"))
    trailer = tmp_path / "movie.mp4"
    shots = [{"start": 0, "end": 4}, {"start": 4, "end": 9}]
    data = scenes.describe_trailer(trailer, shots)
    assert data["descriptions"] == ["close-up of an astronaut, tense, red cockpit light"] * 2
    assert len(data["embeddings"]) == 2
    assert json.loads(scenes.cache_path(trailer).read_text())["version"] == scenes.VERSION
    calls.clear()
    assert scenes.describe_trailer(trailer, shots) == scenes.load(trailer, 2)
    assert calls == []  # cached
    assert scenes.load(trailer, 3) is None  # shot list changed


def test_descriptions_can_be_switched_off(monkeypatch, tmp_path):
    monkeypatch.setenv("SCENE_DESCRIPTIONS", "false")
    assert scenes.describe_trailer(tmp_path / "movie.mp4", [{"start": 0, "end": 4}]) is None


def test_unavailable_vision_model_falls_back_quietly(monkeypatch, tmp_path):
    import requests

    def down(*args, **kwargs):
        raise requests.ConnectionError("no server")
    monkeypatch.setattr(scenes.requests, "post", down)
    assert scenes.describe_trailer(tmp_path / "movie.mp4", [{"start": 0, "end": 4}]) is None


def test_relevance_blends_image_and_description_matches_on_clip_scale(monkeypatch):
    torch = pytest.importorskip("torch")
    selector = object.__new__(ShotSelector)
    selector.torch = torch
    path = "movie.mp4"
    selector.siglip = SimpleNamespace(text=lambda text: torch.tensor([[1.0, 0.0]]))
    selector.siglip_features = {path: torch.tensor([[1.0, 0.0], [0.0, 1.0], [0.7, 0.7]])}
    selector.scene_embeddings = {path: np.array([[0, 1], [1, 0], [0.7, 0.7]], dtype=np.float32)}
    monkeypatch.setattr(scenes, "embed", lambda texts: np.array([[1.0, 0.0]], dtype=np.float32))
    scores = selector.relevance(path, "An astronaut panics.")
    assert scores.shape == (3,)
    assert 0.15 < scores.min() and scores.max() < 0.35
    assert scores[2] == pytest.approx(scores.mean(), abs=0.02) or scores.argmax() in (0, 1)


def test_zscore_handles_flat_scores():
    assert np.all(_zscore([0.3, 0.3, 0.3]) == 0)


def test_siglip_loader_reports_and_falls_back_to_clip(monkeypatch, capsys):
    import builtins

    from movie_pipeline.shots import load_siglip
    real = builtins.__import__

    def blocked(name, *args, **kwargs):
        if name in ("open_clip", "transformers"):
            raise ImportError(name)
        return real(name, *args, **kwargs)
    monkeypatch.setattr(builtins, "__import__", blocked)
    assert load_siglip(object()) is None
    assert "SigLIP 2 unavailable" in capsys.readouterr().out


def test_vision_allows_slow_cold_start_then_uses_warm_timeout(monkeypatch):
    timeouts = []

    def post(url, json=None, timeout=None):
        timeouts.append(timeout)
        return FakeResponse({"response": "a car enters a tunnel"})

    monkeypatch.setattr(scenes.requests, "post", post)
    image = Image.new("RGB", (64, 64), "red")
    assert scenes._describe(image) == "a car enters a tunnel"
    assert scenes._describe(image) == "a car enters a tunnel"
    assert timeouts == [90.0, 60.0]


def test_scene_failures_report_types_without_leaking_details_and_keep_good_frames(monkeypatch, tmp_path, capsys):
    import requests

    calls = []
    def post(url, json=None, timeout=None):
        if url.endswith("/api/generate"):
            calls.append(timeout)
            if len(calls) == 1:
                raise requests.ReadTimeout("private https://secret.example/token")
            return FakeResponse({"response": "a car enters a tunnel"})
        if url.endswith("/api/embed"):
            return FakeResponse({"embeddings": [[1.0, 0.0] for _ in json["input"]]})
        return FakeResponse({})

    monkeypatch.setattr(scenes.requests, "post", post)
    monkeypatch.setattr("movie_pipeline.shots.get_frame", lambda path, seconds: Image.new("RGB", (64, 64), "red"))
    trailer = tmp_path / "movie.mp4"
    data = scenes.describe_trailer(trailer, [{"start": 0, "end": 4}, {"start": 4, "end": 9}])
    assert data is not None and len(data["described"]) == 1
    log = capsys.readouterr().out
    assert "ReadTimeout: 1" in log
    assert "secret.example" not in log
    assert len(calls) == 2 and all(timeout > 25 for timeout in calls)


def test_empty_vision_response_reports_fallback(monkeypatch, tmp_path, capsys):
    def post(url, json=None, timeout=None):
        if url.endswith("/api/generate"):
            return FakeResponse({"response": ""})
        return FakeResponse({})
    monkeypatch.setattr(scenes.requests, "post", post)
    monkeypatch.setattr("movie_pipeline.shots.get_frame", lambda path, seconds: Image.new("RGB", (64, 64), "red"))
    result = scenes.describe_trailer(tmp_path / "movie.mp4", [{"start": 0, "end": 4}])
    assert result is None
    assert "EmptyResponse: 1" in capsys.readouterr().out


def test_vision_timeout_is_configurable_and_bounded(monkeypatch):
    monkeypatch.setenv("SCENE_COLD_TIMEOUT_SECONDS", "999")
    assert scenes._vision_timeout() == 180.0
    monkeypatch.setenv("SCENE_COLD_TIMEOUT_SECONDS", "invalid")
    assert scenes._vision_timeout() == 90.0
    monkeypatch.setattr(scenes.time, "monotonic", lambda: 100.0)
    assert scenes._vision_timeout(deadline=104.0) == 4.0


def test_thumbnail_vision_failure_is_reported_without_leaking_details(monkeypatch, capsys):
    import requests

    def post(*args, **kwargs):
        raise requests.ReadTimeout("https://private.example/token")

    monkeypatch.setattr(scenes.requests, "post", post)
    assert scenes.rate_thumbnail(Image.new("RGB", (64, 64))) is None
    log = capsys.readouterr().out
    assert "ReadTimeout" in log and "image-based ranking" in log
    assert "private.example" not in log
