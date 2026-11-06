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


def _selector_for_relevance(scores):
    """A lightweight CLIP stand-in so coverage does not require a torch install."""
    from contextlib import nullcontext

    class ScoreTensor:
        def squeeze(self, _):
            return self
        def numpy(self):
            return np.asarray(scores, dtype=np.float32)

    class FeatureTensor:
        def __matmul__(self, _):
            return ScoreTensor()

    class TextTensor:
        @property
        def T(self):
            return self

    selector = object.__new__(ShotSelector)
    selector.torch = SimpleNamespace(inference_mode=nullcontext)
    selector.siglip = SimpleNamespace(text=lambda _: TextTensor())
    selector.siglip_features = {"movie.mp4": FeatureTensor()}
    selector.scene_embeddings = {}
    return selector


def test_single_or_constant_description_does_not_dilute_image_ranking(monkeypatch):
    selector = _selector_for_relevance([0.1, 0.3, 0.2])
    monkeypatch.setattr(scenes, "embed", lambda _: np.array([[1.0, 0.0]], dtype=np.float32))
    baseline = selector.relevance("movie.mp4", "a chase")
    selector.scene_embeddings["movie.mp4"] = np.array([[1, 0], [0, 0], [0, 0]], dtype=np.float32)
    np.testing.assert_allclose(selector.relevance("movie.mp4", "a chase"), baseline)
    selector.scene_embeddings["movie.mp4"] = np.array([[1, 0], [1, 0], [0, 0]], dtype=np.float32)
    np.testing.assert_allclose(selector.relevance("movie.mp4", "a chase"), baseline)


def test_multiple_distinct_descriptions_can_change_scene_ranking(monkeypatch):
    selector = _selector_for_relevance([0.1, 0.3, 0.2])
    monkeypatch.setattr(scenes, "embed", lambda _: np.array([[1.0, 0.0]], dtype=np.float32))
    baseline = selector.relevance("movie.mp4", "a chase")
    selector.scene_embeddings["movie.mp4"] = np.array([[1, 0], [0, 1], [0, 0]], dtype=np.float32)
    mixed = selector.relevance("movie.mp4", "a chase")
    assert not np.allclose(mixed, baseline)
    assert mixed[0] > baseline[0]


def test_relevance_falls_back_when_image_and_description_matching_are_unavailable(monkeypatch):
    selector = _selector_for_relevance([0.1, 0.3, 0.2])
    selector.siglip = None
    expected = np.array([0.25, 0.3, 0.2], dtype=np.float32)
    monkeypatch.setattr(selector, "identity_relevance", lambda path, text: expected)
    np.testing.assert_allclose(selector.relevance("movie.mp4", "a chase"), expected)


@pytest.mark.parametrize("payload", [
    {},
    {"large_face": "false", "emotion_intensity": 3, "action_or_spectacle": 2,
     "genre_fit": 4, "logo_or_text": False, "thumbnail_score": 5},
    {"large_face": True, "emotion_intensity": 11, "action_or_spectacle": 2,
     "genre_fit": 4, "logo_or_text": False, "thumbnail_score": 5},
])
def test_malformed_thumbnail_vision_is_unavailable_not_a_false_verdict(monkeypatch, payload):
    monkeypatch.setattr(scenes.requests, "post", lambda *args, **kwargs:
                        FakeResponse({"response": json.dumps(payload)}))
    assert scenes.rate_thumbnail(Image.new("RGB", (64, 64))) is None
    assert scenes._VISION_READY is False


def test_valid_thumbnail_vision_verdict_is_kept(monkeypatch):
    payload = {"large_face": True, "emotion_intensity": 8, "action_or_spectacle": 5,
               "genre_fit": 7, "logo_or_text": False, "thumbnail_score": 9}
    monkeypatch.setattr(scenes.requests, "post", lambda *args, **kwargs:
                        FakeResponse({"response": json.dumps(payload)}))
    result = scenes.rate_thumbnail(Image.new("RGB", (64, 64)))
    assert result["large_face"] is True and result["logo"] is False and result["score"] == 9
    assert scenes._VISION_READY is True


def test_thumbnail_installs_missing_instruct_model_before_rating(monkeypatch, tmp_path):
    calls = []
    verdict = {"large_face": True, "emotion_intensity": 8, "action_or_spectacle": 5,
               "genre_fit": 7, "logo_or_text": False, "thumbnail_score": 9}
    def post(url, json=None, timeout=None):
        calls.append((url.rsplit("/", 1)[-1], json, timeout))
        response = FakeResponse({"response": __import__("json").dumps(verdict)})
        response.status_code = 404 if url.endswith("/api/show") else 200
        return response
    monkeypatch.setattr(scenes.requests, "post", post)
    monkeypatch.setattr(scenes.shutil, "disk_usage", lambda path: SimpleNamespace(free=5_000_000_000))
    assert scenes.rate_thumbnail(Image.new("RGB", (64, 64)))["score"] == 9
    assert [entry[0] for entry in calls] == ["show", "pull", "generate"]
    assert calls[1][1] == {"model": scenes.VISION_MODEL, "stream": False}
    assert calls[1][2] == 90
    calls.clear()
    scenes.rate_thumbnail(Image.new("RGB", (64, 64)))
    assert [entry[0] for entry in calls] == ["generate"]


def test_missing_optional_thumbnail_model_preserves_render_disk(monkeypatch, capsys):
    calls = []
    def post(url, **kwargs):
        calls.append(url.rsplit("/", 1)[-1])
        response = FakeResponse({})
        response.status_code = 404
        return response
    monkeypatch.setattr(scenes.requests, "post", post)
    monkeypatch.setattr(scenes.shutil, "disk_usage", lambda path: SimpleNamespace(free=2_000_000_000))
    assert scenes.rate_thumbnail(Image.new("RGB", (64, 64))) is None
    assert calls == ["show"]
    assert "image-based ranking" in capsys.readouterr().out


def test_thumbnail_http_error_reports_status_without_private_url(monkeypatch, capsys):
    import requests
    def post(*args, **kwargs):
        response = SimpleNamespace(status_code=503)
        raise requests.HTTPError("https://private.example/token", response=response)
    monkeypatch.setattr(scenes.requests, "post", post)
    assert scenes.rate_thumbnail(Image.new("RGB", (64, 64))) is None
    log = capsys.readouterr().out
    assert "HTTP 503" in log and "private.example" not in log
