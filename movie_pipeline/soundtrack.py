"""Bundled licensed recorded instrumentals; one quiet musical theme per episode."""
from __future__ import annotations

import math
import re
import wave
from pathlib import Path

import numpy as np

RATE = 48000


def theme_for(title: str, brief: str = "") -> str:
    text = title.casefold()
    # The title takes precedence over incidental examples in the creative brief.
    for source in (text, brief.casefold()):
        for theme, pattern in (("romance", r"romance|romantic|love stor"),
                               ("action", r"\baction\b|adventure"),
                               ("mystery", r"mystery|thriller|horror|sci.?fi"),
                               ("drama", r"\bdrama\b")):
            if re.search(pattern, source):
                return theme
    return "drama"


def narration_rate(theme: str) -> float:
    return {"action": 1.15, "mystery": 1.08, "drama": 1.0, "romance": 1.0}.get(theme, 1.0)


MUSIC_ROOT = Path(__file__).resolve().parents[1] / "assets" / "music"


def selected_track(theme: str) -> dict:
    import json
    name = {"action": "titan", "mystery": "celestial", "romance": "aurora", "drama": "aurora"}.get(theme, "aurora")
    return json.loads((MUSIC_ROOT / "catalog.json").read_text())[name]


def music_credit(theme: str) -> str:
    track = selected_track(theme)
    return (f"Music: '{track['title']}' by Scott Buckley — CC BY 4.0. "
            f"{track['source']} License: {track['license_url']} "
            "Excerpt, volume adjustment and crossfaded repetition.")


def music_bed(path: Path, seconds: float, action: bool = False, *, theme: str | None = None):
    """Quiet licensed recorded music, with smooth loops and no synthesized bed."""
    import hashlib

    import soundfile as sf

    from .core import PipelineError
    track = selected_track(theme or ("action" if action else "drama"))
    source = MUSIC_ROOT / track["file"]
    if hashlib.sha256(source.read_bytes()).hexdigest() != track["sha256"]:
        raise PipelineError("Bundled music failed its integrity check.")
    samples, rate = sf.read(source, dtype="float32", always_2d=True)
    if rate != RATE or len(samples) < RATE*8 or samples.shape[1] != 2:
        raise PipelineError("Bundled music must be stereo 48 kHz with a readable loop.")
    overlap = RATE*3
    weight = np.linspace(0, 1, overlap, dtype=np.float32)[:, None]
    loop = np.concatenate((samples[overlap:-overlap],
                           samples[-overlap:]*(1-weight)+samples[:overlap]*weight))
    rms = float(np.sqrt(np.mean(loop.astype(np.float64)**2)))
    peak = float(np.max(np.abs(loop)))
    gain = min(.0126/max(rms, 1e-9), .08/max(peak, 1e-9))
    with wave.open(str(path), "wb") as stream:
        stream.setnchannels(2)
        stream.setsampwidth(2)
        stream.setframerate(RATE)
        for block in range(math.ceil(seconds)):
            count = min(RATE, round(seconds*RATE)-block*RATE)
            indices = np.arange(count)+block*RATE
            time = indices/RATE
            envelope = np.minimum(1, time/2)*np.minimum(1, np.maximum(0, seconds-time)/2)
            audio = loop[indices % len(loop)]*gain*envelope[:, None]
            stream.writeframes((audio*32767).astype("<i2").tobytes())
