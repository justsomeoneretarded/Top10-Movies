"""Short AI descriptions of each trailer shot, used to match footage to narration.

A small local vision model (Qwen3-VL 2B through Ollama) describes the middle
frame of every analysed shot in a few plain words ("close-up of a woman in a
spacesuit, tense, red light"). A small local text-embedding model turns the
descriptions and each narration sentence into vectors, so a line about a
character's fear can find a frightened close-up. Everything is optional:
without descriptions the edit uses SigLIP image matching alone.
"""
from __future__ import annotations

import base64
from collections import Counter
import io
import json
import os
import time
from pathlib import Path

import numpy as np
import requests

from .core import PipelineError, save_json

VISION_MODEL = os.getenv("SCENE_MODEL", "qwen3-vl:2b")
EMBED_MODEL = os.getenv("EMBED_MODEL", "all-minilm")
VERSION = 2
_DEADLINE = None
_READY = False
_UNAVAILABLE = False
_VISION_READY = False
PROMPT = ("Describe this movie frame in at most 14 words for a video editor: who or what is shown, "
          "the action, the mood and the setting. Plain words, no names, no quotes.")


def enabled() -> bool:
    return os.getenv("SCENE_DESCRIPTIONS", "true").strip().lower() != "false"


def _url() -> str:
    return os.getenv("OLLAMA_URL", "http://127.0.0.1:11434").rstrip("/")


def _pull(model: str):
    requests.post(_url() + "/api/pull", json={"model": model, "stream": False}, timeout=180).raise_for_status()


def cache_path(trailer: Path) -> Path:
    return trailer.with_suffix(".scenes.json")


def load(trailer: Path, count: int) -> dict | None:
    """Cached descriptions and embeddings for a trailer, when they match its shot list."""
    path = cache_path(trailer)
    try:
        data = json.loads(path.read_text())
    except (OSError, ValueError):
        return None
    if data.get("version") != VERSION or len(data.get("descriptions", [])) != count or len(data.get("embeddings", [])) != count:
        return None
    return data


def embed(texts: list[str]) -> np.ndarray:
    response = requests.post(_url() + "/api/embed", json={"model": EMBED_MODEL, "input": texts}, timeout=300)
    response.raise_for_status()
    vectors = np.asarray(response.json()["embeddings"], dtype=np.float32)
    norms = np.linalg.norm(vectors, axis=1, keepdims=True)
    return vectors / np.maximum(norms, 1e-9)


def _vision_timeout(*, deadline=None) -> float:
    """Allow a cold CPU model more time while keeping requests bounded.

    The first useful response can include model loading. Once the vision model
    has returned a valid result, a shorter warm timeout keeps retries bounded.
    """
    name = "SCENE_WARM_TIMEOUT_SECONDS" if _VISION_READY else "SCENE_COLD_TIMEOUT_SECONDS"
    default = 60.0 if _VISION_READY else 90.0
    try:
        configured = float(os.getenv(name, str(default)))
    except (TypeError, ValueError):
        configured = default
    if not np.isfinite(configured):
        configured = default
    timeout = max(5.0, min(180.0, configured))
    if deadline is not None:
        timeout = min(timeout, max(0.1, deadline - time.monotonic()))
    return timeout


def _describe(image) -> str:
    global _VISION_READY
    small = image.convert("RGB").copy()
    small.thumbnail((448, 252))
    buffer = io.BytesIO()
    small.save(buffer, format="JPEG", quality=88)
    response = requests.post(_url() + "/api/generate", json={
        "model": VISION_MODEL, "prompt": PROMPT, "images": [base64.b64encode(buffer.getvalue()).decode()],
        "stream": False, "think": False, "keep_alive": "10m",
        "options": {"temperature": 0.2, "num_predict": 40, "num_ctx": 2048}}, timeout=_vision_timeout(deadline=_DEADLINE))
    response.raise_for_status()
    text = " ".join(str(response.json().get("response", "")).replace("\n", " ").split())
    text = text.strip(" \"'")[:160]
    if text:
        _VISION_READY = True
    return text


def describe_trailer(trailer: Path, shots: list[dict]) -> dict | None:
    """Describe representative frames within a shared optional-analysis budget."""
    global _DEADLINE, _READY, _UNAVAILABLE
    if not enabled() or not shots:
        return None
    cached = load(trailer, len(shots))
    if cached:
        return cached
    if _UNAVAILABLE:
        return None
    if _DEADLINE is None:
        _DEADLINE = time.monotonic() + max(30, float(os.getenv("SCENE_BUDGET_SECONDS", "420")))
    if time.monotonic() >= _DEADLINE:
        return None
    from .shots import shot_frame
    try:
        if not _READY:
            _pull(VISION_MODEL)
            _pull(EMBED_MODEL)
            _READY = True
        limit = max(1, min(16, int(os.getenv("SCENE_SAMPLE_FRAMES", "6"))))
        # Mix representative positions and impact moments; never caption the same frame twice.
        spread = np.linspace(0, len(shots)-1, min(limit, len(shots)), dtype=int).tolist()
        ranked = sorted(range(len(shots)), key=lambda i: -float(shots[i].get("peak", 0)))[:limit//2]
        indices = list(dict.fromkeys([*ranked, *spread]))[:limit]
        descriptions = [""] * len(shots)
        failures = Counter()
        for index in indices:
            if time.monotonic() >= _DEADLINE:
                break
            try:
                descriptions[index] = _describe(shot_frame(trailer, shots, index))
                if not descriptions[index]:
                    failures["EmptyResponse"] += 1
            except (PipelineError, requests.RequestException, ValueError) as exc:
                # Do not log exception text: request URLs can contain credentials.
                failures[type(exc).__name__] += 1
                continue
        known = [i for i, text in enumerate(descriptions) if text]
        if failures:
            details = ", ".join(f"{name}: {count}" for name, count in sorted(failures.items()))
            print(f"Scene analysis for {trailer.name}: {len(known)} descriptions; "
                  f"skipped {sum(failures.values())} frames ({details}). "
                  "Image matching remains available.", flush=True)
        if not known:
            return None
        embedded = embed([descriptions[i] for i in known])
        vectors = np.zeros((len(shots), embedded.shape[1]), dtype=np.float32)
        vectors[known] = embedded
    except (requests.RequestException, ValueError, KeyError) as exc:
        _UNAVAILABLE = True
        print(f"Optional scene description unavailable ({type(exc).__name__}); retaining image matching.", flush=True)
        return None
    data = {"version": VERSION, "model": VISION_MODEL, "embed_model": EMBED_MODEL,
            "descriptions": descriptions, "embeddings": vectors.round(5).tolist(), "described": known}
    save_json(cache_path(trailer), data)
    return data


def unload():
    global _VISION_READY
    _VISION_READY = False
    for model in (VISION_MODEL,):
        try:
            requests.post(_url() + "/api/generate", json={"model": model, "keep_alive": 0}, timeout=30)
        except requests.RequestException:
            pass




def rate_thumbnail(image, genre: str = "", genre_hint: str = "") -> dict | None:
    """Vision-model judgement of a frame as thumbnail material; None when the model is unavailable.

    ``genre_hint`` describes what the genre looks like ("fear, screaming, creatures, masks,
    blood, darkness"), so the model rates how instantly the frame reads as that genre.
    """
    global _VISION_READY
    small = image.convert("RGB").copy()
    small.thumbnail((512, 288))
    buffer = io.BytesIO()
    small.save(buffer, format="JPEG", quality=88)
    schema = {"type": "object", "properties": {
        "large_face": {"type": "boolean"}, "emotion_intensity": {"type": "integer", "minimum": 0, "maximum": 10},
        "action_or_spectacle": {"type": "integer", "minimum": 0, "maximum": 10},
        "genre_fit": {"type": "integer", "minimum": 0, "maximum": 10},
        "logo_or_text": {"type": "boolean"},
        "thumbnail_score": {"type": "integer", "minimum": 0, "maximum": 10}},
        "required": ["large_face", "emotion_intensity", "action_or_spectacle", "genre_fit", "logo_or_text", "thumbnail_score"]}
    prompt = THUMBNAIL_PROMPT
    if genre:
        prompt += (f" The video is a {genre} countdown. genre_fit: how instantly a viewer scrolling past would know this "
                   f"is a {genre} film from this frame alone ({genre_hint}); calm, smiling or neutral everyday frames score low.")
    try:
        response = requests.post(_url() + "/api/generate", json={
            "model": VISION_MODEL, "prompt": prompt, "images": [base64.b64encode(buffer.getvalue()).decode()],
            "format": schema, "stream": False, "think": False, "keep_alive": "10m",
            "options": {"temperature": 0.1, "num_predict": 100, "num_ctx": 2048}}, timeout=_vision_timeout())
        response.raise_for_status()
        data = json.loads(response.json().get("response", "{}"))
        # An empty or malformed verdict is not evidence that a frame has no
        # logo or a low score. Keep the existing image-based fallback instead.
        numeric = ("emotion_intensity", "action_or_spectacle", "genre_fit", "thumbnail_score")
        if (not isinstance(data, dict)
                or any(type(data.get(field)) is not bool for field in ("large_face", "logo_or_text"))
                or any(type(data.get(field)) is not int or not 0 <= data[field] <= 10 for field in numeric)):
            raise ValueError("Invalid thumbnail vision verdict")
        _VISION_READY = True
        return {"large_face": bool(data.get("large_face")),
                "emotion": max(0, min(10, int(data.get("emotion_intensity", 0)))),
                "action": max(0, min(10, int(data.get("action_or_spectacle", 0)))),
                "genre_fit": max(0, min(10, int(data.get("genre_fit", 5)))),
                "logo": bool(data.get("logo_or_text")),
                "score": max(0, min(10, int(data.get("thumbnail_score", 0))))}
    except (requests.RequestException, ValueError, TypeError, KeyError) as exc:
        print(f"Optional thumbnail vision unavailable ({type(exc).__name__}); "
              "retaining image-based ranking.", flush=True)
        return None


THUMBNAIL_PROMPT = ("You are choosing a YouTube thumbnail for a movie countdown. Judge this frame: "
                    "does it show a large, clear human face? how intense is the emotion (fear, rage, shock, focus)? "
                    "is there visible action, a weapon, a creature, an explosion or spectacle? "
                    "is any logo, watermark, channel bug or written text visible anywhere, even small in a corner? "
                    "Return JSON only.")
