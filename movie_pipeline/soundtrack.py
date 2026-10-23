"""Deterministic original instrumental beds with smooth, click-free note envelopes.

No downloaded music, noise layers, square pulses, or hard oscillator resets.
A single arrangement is chosen for the complete video's theme.
"""
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


def music_bed(path: Path, seconds: float, action: bool = False, *, theme: str | None = None):
    """Stream a softly plucked/piano-like arrangement with overlapping chords.

    Smooth attacks/releases ensure continuity at every note and WAV block.
    This is synthesized instrumental music, not a commercial recording.
    """
    theme = theme or ("action" if action else "drama")
    bpm = {"action": 108, "mystery": 76, "romance": 66, "drama": 74}[theme]
    beat = 60 / bpm
    minor = theme in {"action", "mystery"}
    roots = [45, 41, 48, 43] if minor else [48, 45, 41, 43]
    third = 3 if minor else 4
    events = []
    for bar in range(math.ceil(seconds / (beat * 4))):
        start = bar * beat * 4
        root = roots[(bar // 2) % len(roots)]
        # Overlapping soft chord voices, and a quiet arpeggiated upper line.
        for note in (root, root + third, root + 7):
            events.append((start, beat * 5, note, .024, "pad"))
        step = beat / 2 if theme == "action" else beat
        notes = (root + 12, root + 7, root + third + 12, root + 7)
        for j in range(round(beat * 4 / step)):
            events.append((start + j * step, beat * 2.5, notes[j % 4], .038, "pluck"))
    with wave.open(str(path), "wb") as stream:
        stream.setnchannels(2)
        stream.setsampwidth(2)
        stream.setframerate(RATE)
        for block in range(math.ceil(seconds)):
            count = min(RATE, round(seconds * RATE) - block * RATE)
            time = (np.arange(count) + block * RATE) / RATE
            signal = np.zeros(count, dtype=np.float64)
            for start, length, note, gain, instrument in events:
                if start >= time[-1] or start + length <= time[0]:
                    continue
                age = time - start
                valid = (age >= 0) & (age < length)
                t = age[valid]
                frequency = 440 * 2 ** ((note - 69) / 12)
                attack = .22 if instrument == "pad" else .025
                envelope = np.minimum(1, t / attack) ** 2
                envelope *= np.minimum(1, (length - t) / .35) ** 2
                if instrument == "pluck":
                    envelope *= np.exp(-t / (beat * .8))
                voice = (np.sin(2 * np.pi * frequency * t)
                         + .22 * np.sin(2 * np.pi * frequency * 2 * t)
                         + .07 * np.sin(2 * np.pi * frequency * 3 * t))
                signal[valid] += gain * envelope * voice
            signal *= np.minimum(1, time / 2) * np.minimum(1, np.maximum(0, seconds-time) / 2)
            stereo = np.column_stack((signal, signal * .98))
            stream.writeframes((np.clip(stereo, -.95, .95) * 32767).astype("<i2").tobytes())
