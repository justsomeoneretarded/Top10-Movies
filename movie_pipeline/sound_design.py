"""Segment-aware music levels and original synthesized transition effects.

The music bed is shaped by what is on screen: louder over the cold open, rank
reveals, title cards and the end screen; low under narration (where sidechain
ducking still applies); almost silent under trailer dialogue. Whooshes and
impacts mark rank reveals. Effects are generated here from noise and sine
sweeps, so there is no third-party audio or license involved.
"""
from __future__ import annotations

import math
import wave
from pathlib import Path

import numpy as np

RATE = 48000
CONTROL_RATE = 100  # gain automation points per second
RAMP_SECONDS = 0.45

# Multipliers on the quiet base bed (about 20 dB under narration).
MUSIC_LEVELS = {
    "cold-open": 4.0,
    "review-preview": 3.5,          # muted preview: music carries the reveal
    "review-preview-dialogue": 1.0,  # preview keeps a complete trailer line
    "title-card": 3.0,
    "subscribe": 1.8,
    "narration": 1.0,
    "narration-top": 1.25,           # number one gets a slightly bigger bed
    "trailer-audio": 0.35,
    "ending": 3.5,
}


def segment_level(entry: dict, top_title: str | None = None) -> float:
    kind = entry.get("type", "narration")
    if kind == "review-preview":
        shots = entry.get("shots") or [{}]
        return MUSIC_LEVELS["review-preview" if shots[0].get("muted", True) else "review-preview-dialogue"]
    if kind == "narration" and top_title and entry.get("section_kind") == "film" and entry.get("film") == top_title:
        return MUSIC_LEVELS["narration-top"]
    return MUSIC_LEVELS.get(kind, 1.0)


def gain_curve(timeline: list[dict], speed: float, total: float, ending_start: float,
               top_title: str | None = None) -> np.ndarray:
    """Smooth gain automation at CONTROL_RATE for the final (tempo-adjusted) timeline."""
    points = max(1, math.ceil(total * CONTROL_RATE) + 1)
    target = np.ones(points, dtype=np.float32)
    for entry in timeline:
        start = float(entry.get("start", 0)) / speed
        end = start + float(entry.get("duration", 0)) / speed
        a, b = max(0, round(start * CONTROL_RATE)), min(points, round(end * CONTROL_RATE))
        if b > a:
            target[a:b] = segment_level(entry, top_title)
    target[max(0, round(ending_start * CONTROL_RATE)):] = MUSIC_LEVELS["ending"]
    window = max(1, round(RAMP_SECONDS * CONTROL_RATE))
    padded = np.pad(target, (window, window), mode="edge")
    smooth = np.convolve(padded, np.ones(window, dtype=np.float32) / window, mode="same")[window:-window]
    return smooth.astype(np.float32)


def shape_music(path: Path, curve: np.ndarray):
    """Apply a gain curve to a 16-bit stereo WAV in place, one second at a time."""
    shaped = path.with_suffix(".shaped.wav")
    with wave.open(str(path), "rb") as source, wave.open(str(shaped), "wb") as target:
        target.setparams(source.getparams())
        rate = source.getframerate()
        position = 0
        control_times = np.arange(len(curve)) / CONTROL_RATE
        while True:
            raw = source.readframes(rate)
            if not raw:
                break
            block = np.frombuffer(raw, dtype="<i2").reshape(-1, source.getnchannels()).astype(np.float32)
            times = (position + np.arange(len(block))) / rate
            gain = np.interp(times, control_times, curve).astype(np.float32)
            block *= gain[:, None]
            target.writeframes(np.clip(block, -32767, 32767).astype("<i2").tobytes())
            position += len(block)
    shaped.replace(path)


def _one_pole_lowpass(signal: np.ndarray, cutoff: np.ndarray) -> np.ndarray:
    """Time-varying one-pole low-pass (cutoff in Hz per sample)."""
    alpha = 1 - np.exp(-2 * np.pi * np.clip(cutoff, 20, RATE / 2.2) / RATE)
    out = np.empty_like(signal)
    state = 0.0
    for index, (value, a) in enumerate(zip(signal, alpha)):
        state += a * (value - state)
        out[index] = state
    return out


def whoosh(seconds: float = 0.75, seed: int = 1) -> np.ndarray:
    """Stereo noise sweep that rises to a peak and falls away, panning left to right."""
    count = round(seconds * RATE)
    rng = np.random.default_rng(seed)
    t = np.linspace(0, 1, count, dtype=np.float32)
    noise = rng.standard_normal(count).astype(np.float32)
    peak = 0.62
    shape = np.where(t < peak, (t / peak) ** 2.2, np.exp(-(t - peak) * 9)).astype(np.float32)
    cutoff = 300 + 5200 * shape
    band = _one_pole_lowpass(noise, cutoff) - _one_pole_lowpass(noise, cutoff * 0.18)
    band *= shape
    band /= max(float(np.max(np.abs(band))), 1e-9)
    pan = 0.15 + 0.7 * t
    return np.column_stack((band * np.sqrt(1 - pan), band * np.sqrt(pan))).astype(np.float32)


def impact(seconds: float = 1.1, seed: int = 2) -> np.ndarray:
    """Low cinematic hit: a falling sine 'boom' plus a short filtered transient."""
    count = round(seconds * RATE)
    t = np.arange(count, dtype=np.float32) / RATE
    frequency = 38 + 62 * np.exp(-t * 7)
    phase = 2 * np.pi * np.cumsum(frequency) / RATE
    boom = np.sin(phase) * np.exp(-t * 3.2)
    rng = np.random.default_rng(seed)
    transient = _one_pole_lowpass(rng.standard_normal(count).astype(np.float32), np.full(count, 1800.0)) * np.exp(-t * 40)
    hit = boom + 0.6 * transient
    attack = np.minimum(1, t / 0.004)
    hit = (hit * attack).astype(np.float32)
    hit /= max(float(np.max(np.abs(hit))), 1e-9)
    return np.column_stack((hit, hit)).astype(np.float32)


def effect_events(timeline: list[dict], speed: float) -> list[tuple[float, str, float]]:
    """(time, effect, gain) in final-timeline seconds."""
    events = []
    for index, entry in enumerate(timeline):
        start = float(entry.get("start", 0)) / speed
        kind = entry.get("type")
        if kind == "cold-open" and index == 0:
            events.append((0.02, "impact", 0.55))
        elif kind == "review-preview":
            # The rank label slides in over about half a second; the hit lands with it.
            events.append((max(0.0, start - 0.32), "whoosh", 0.5))
            events.append((start + 0.5, "impact", 0.7))
        elif kind == "subscribe":
            events.append((max(0.0, start - 0.25), "whoosh", 0.3))
        elif kind == "narration" and entry.get("section_kind") == "outro" and \
                (index == 0 or timeline[index - 1].get("section_kind") != "outro"):
            events.append((max(0.0, start - 0.3), "whoosh", 0.25))
    return events


def effects_track(path: Path, timeline: list[dict], speed: float, total: float) -> int:
    """Write a stereo 16-bit WAV with every transition effect; returns the event count."""
    samples = np.zeros((max(1, round(total * RATE)), 2), dtype=np.float32)
    sounds = {"whoosh": whoosh(), "impact": impact()}
    events = effect_events(timeline, speed)
    for when, name, gain in events:
        sound = sounds[name]
        start = round(when * RATE)
        if start >= len(samples):
            continue
        end = min(len(samples), start + len(sound))
        samples[start:end] += sound[:end - start] * gain * 0.25  # about -12 dBFS peaks before the final limiter
    with wave.open(str(path), "wb") as stream:
        stream.setnchannels(2)
        stream.setsampwidth(2)
        stream.setframerate(RATE)
        stream.writeframes((np.clip(samples, -1, 1) * 32767).astype("<i2").tobytes())
    return len(events)


DELIVERY_PACE = {"hype": 1.05, "normal": 1.0, "dramatic": 0.94, "calm": 0.97}


def delivery_speed(theme_rate: float, delivery: str) -> float:
    """Narration tempo for one paragraph: the theme's pace shaped by its delivery label."""
    return round(min(1.18, max(0.9, theme_rate * DELIVERY_PACE.get(delivery, 1.0))), 3)
