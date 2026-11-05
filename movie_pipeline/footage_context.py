"""Collect optional visual context from existing per-source scene caches.

Descriptions come from individual frames, not verified plot or selected scenes.
Do not load a vision model or make any network calls here.
"""
from __future__ import annotations

from pathlib import Path

from . import scenes


def collect_visual_hints(pools, shot_lookup, *, limit=3):
    """Return bounded, distinct frame descriptions for each title in pool order."""
    result = []
    for pool in pools:
        hints, seen = [], set()
        for source in pool:
            shots = shot_lookup.get(str(source), [])
            if not shots:
                continue
            try:
                data = scenes.load(Path(source), len(shots))
                descriptions = data.get("descriptions", []) if isinstance(data, dict) else []
                if not isinstance(descriptions, list):
                    continue
                for value in descriptions:
                    if not isinstance(value, str):
                        continue
                    text = " ".join(value.split())[:120].strip()
                    identity = text.casefold()
                    if not text or identity in seen:
                        continue
                    seen.add(identity)
                    hints.append(text)
                    if len(hints) >= max(1, limit):
                        break
            except (OSError, ValueError, TypeError, KeyError, AttributeError):
                continue
            if len(hints) >= max(1, limit):
                break
        result.append(hints)
    return result
