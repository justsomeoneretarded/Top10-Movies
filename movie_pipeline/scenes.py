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
import io
import json
import os
from pathlib import Path

import numpy as np
import requests

from .core import PipelineError, save_json

VISION_MODEL = os.getenv("SCENE_MODEL", "qwen3-vl:2b")
EMBED_MODEL = os.getenv("EMBED_MODEL", "all-minilm")
VERSION = 1
PROMPT = ("Describe this movie frame in at most 14 words for a video editor: who or what is shown, "
          "the action, the mood and the setting. Plain words, no names, no quotes.")


def enabled() -> bool:
    return os.getenv("SCENE_DESCRIPTIONS", "true").strip().lower() != "false"


def _url() -> str:
    return os.getenv("OLLAMA_URL", "http://127.0.0.1:11434").rstrip("/")


def _pull(model: str):
    requests.post(_url() + "/api/pull", json={"model": model, "stream": False}, timeout=1800).raise_for_status()


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


def _describe(image) -> str:
    small = image.convert("RGB").copy()
    small.thumbnail((448, 252))
    buffer = io.BytesIO()
    small.save(buffer, format="JPEG", quality=88)
    response = requests.post(_url() + "/api/generate", json={
        "model": VISION_MODEL, "prompt": PROMPT, "images": [base64.b64encode(buffer.getvalue()).decode()],
        "stream": False, "think": False, "keep_alive": "10m",
        "options": {"temperature": 0.2, "num_predict": 40, "num_ctx": 2048}}, timeout=300)
    response.raise_for_status()
    text = " ".join(str(response.json().get("response", "")).replace("\n", " ").split())
    return text.strip(" \"'")[:160]


def describe_trailer(trailer: Path, shots: list[dict]) -> dict | None:
    """Describe every shot once and cache the result next to the trailer."""
    if not enabled() or not shots:
        return None
    cached = load(trailer, len(shots))
    if cached:
        return cached
    from .shots import get_frame
    try:
        _pull(VISION_MODEL)
        _pull(EMBED_MODEL)
        descriptions = []
        for shot in shots:
            try:
                frame = get_frame(trailer, (float(shot["start"]) + float(shot["end"])) / 2)
                descriptions.append(_describe(frame) or "a scene from the trailer")
            except (PipelineError, requests.RequestException, ValueError):
                descriptions.append("a scene from the trailer")
        vectors = embed(descriptions)
    except (requests.RequestException, ValueError, KeyError) as exc:
        print(f"Scene descriptions unavailable for {trailer.name} ({type(exc).__name__}); using image matching only.", flush=True)
        return None
    data = {"version": VERSION, "model": VISION_MODEL, "embed_model": EMBED_MODEL,
            "descriptions": descriptions, "embeddings": vectors.round(5).tolist()}
    save_json(cache_path(trailer), data)
    return data


def unload():
    for model in (VISION_MODEL,):
        try:
            requests.post(_url() + "/api/generate", json={"model": model, "keep_alive": 0}, timeout=30)
        except requests.RequestException:
            pass


THUMBNAIL_PROMPT = ("You are choosing a YouTube thumbnail for a movie countdown. Judge this frame: "
                    "does it show a large, clear human face? how intense is the emotion (fear, rage, shock, focus)? "
                    "is there visible action, a weapon, a creature, an explosion or spectacle? "
                    "Return JSON only.")


def rate_thumbnail(image) -> dict | None:
    """Vision-model judgement of a frame as thumbnail material; None when the model is unavailable."""
    small = image.convert("RGB").copy()
    small.thumbnail((512, 288))
    buffer = io.BytesIO()
    small.save(buffer, format="JPEG", quality=88)
    schema = {"type": "object", "properties": {
        "large_face": {"type": "boolean"}, "emotion_intensity": {"type": "integer", "minimum": 0, "maximum": 10},
        "action_or_spectacle": {"type": "integer", "minimum": 0, "maximum": 10},
        "thumbnail_score": {"type": "integer", "minimum": 0, "maximum": 10}},
        "required": ["large_face", "emotion_intensity", "action_or_spectacle", "thumbnail_score"]}
    try:
        response = requests.post(_url() + "/api/generate", json={
            "model": VISION_MODEL, "prompt": THUMBNAIL_PROMPT, "images": [base64.b64encode(buffer.getvalue()).decode()],
            "format": schema, "stream": False, "think": False, "keep_alive": "10m",
            "options": {"temperature": 0.1, "num_predict": 80, "num_ctx": 2048}}, timeout=300)
        response.raise_for_status()
        data = json.loads(response.json().get("response", "{}"))
        return {"large_face": bool(data.get("large_face")),
                "emotion": max(0, min(10, int(data.get("emotion_intensity", 0)))),
                "action": max(0, min(10, int(data.get("action_or_spectacle", 0)))),
                "score": max(0, min(10, int(data.get("thumbnail_score", 0))))}
    except (requests.RequestException, ValueError, TypeError, KeyError):
        return None
