"""Continuous visual previews before narration, with conservative trailer audio.

CLIP relevance and motion are preferences only. A weak match never blocks a
usable preview; dialogue is muted whenever a clean audio window is uncertain.
"""
from __future__ import annotations

import math
from pathlib import Path

from .core import PipelineError

FPS = 30


def _number(value, default=0.0):
    try:
        value = float(value)
    except (TypeError, ValueError):
        return default
    return value if math.isfinite(value) else default


def _seconds(value: float) -> float:
    value = _number(value)
    if value <= 0:
        raise ValueError("Preview duration must be positive.")
    return max(1, round(value * FPS)) / FPS


def _relevance(selector, source: Path, text: str, count: int) -> list[float]:
    try:
        values = selector.relevance(source, text)
        return [max(-1.0, min(1.0, _number(values[i]))) for i in range(count)]
    except (AttributeError, IndexError, KeyError, RuntimeError, TypeError, ValueError):
        return [0.0] * count


def _dialogue(source: Path, dialogue: dict[str, list] | None) -> list[tuple[float, float]] | None:
    if dialogue is None or str(source) not in dialogue:
        return None
    cues = dialogue[str(source)]
    if not isinstance(cues, list):
        return None
    result = []
    for cue in cues:
        if not isinstance(cue, dict):
            return None
        start = _number(cue.get("start"), -1)
        end = _number(cue.get("end"), -1)
        if start < 0 or end <= start:
            return None
        result.append((start, end))
    return result


def _overlap(start: float, seconds: float, cues: list[tuple[float, float]]) -> bool:
    # A little room around recognized speech avoids cutting into its first consonant.
    return any(start < end + .12 and start + seconds > beginning - .12 for beginning, end in cues)


def _best_preview(selector, sources, shot_lookup, text, seconds, action, dialogue):
    if not sources:
        raise PipelineError("No footage source is available for a preview.")
    query = (("A gripping action highlight: a fight, chase, stunt or explosion. " if action else
              "A striking cinematic character or environment that establishes the mood. ") + text)
    candidates = []
    for source_index, value in enumerate(sources):
        source = Path(value)
        shots = shot_lookup.get(str(source), [])
        scores = _relevance(selector, source, query, len(shots))
        valid = []
        for index, shot in enumerate(shots):
            if not isinstance(shot, dict):
                continue
            start = max(0.0, _number(shot.get("start"), -1))
            end = _number(shot.get("end"), -1)
            if end <= start:
                continue
            valid.append((index, shot, start, end))
        maximum_end = max((end for _, _, _, end in valid), default=seconds)
        speech = _dialogue(source, dialogue)
        for index, shot, start, end in valid:
            available = end - start
            full_window = available >= seconds
            margin = min(.08, max(0.0, (available - seconds) / 2))
            initial = start + margin
            # If every source shot is short, use one continuous trailer window rather
            # than synthesizing several cuts or freezing the preview. Keep it inside
            # the sampled source extent whenever that extent is long enough.
            if not full_window:
                initial = min(initial, max(0.0, maximum_end - seconds))
            starts = [initial]
            if full_window and speech:
                starts += [last + .14 for _, last in speech if start <= last + .14 <= end - seconds]
            for candidate_start in dict.fromkeys(starts):
                muted = speech is None or _overlap(candidate_start, seconds, speech)
                motion = max(0.0, min(60.0, _number(shot.get("motion")))) / 60
                score = scores[index] * 1.25
                score += motion * .8 if action else -.18 * motion
                score += min(available, 8.0) * .025
                score += .16 if not muted else 0
                candidates.append((full_window, score, -source_index, -candidate_start,
                                   {"source": str(source), "start": candidate_start,
                                    "duration": seconds, "muted": muted}))
    if not candidates:
        # Analysis may be incomplete even though the source itself is usable.
        # A quiet continuous opening remains preferable to aborting production.
        return 0.0, {"source": str(sources[0]), "start": 0.0,
                     "duration": seconds, "muted": True}
    full, score, _source, _start, result = max(candidates, key=lambda value: value[:4])
    return score + (.5 if full else 0), result


def choose_preview(selector, sources: list[Path], shot_lookup: dict[str, list], text: str,
                   seconds: float, action: bool, dialogue: dict[str, list] | None = None) -> dict:
    """Choose one uninterrupted source interval before a movie's narration.

    Known speech-free intervals can retain trailer sound. Recognized speech,
    malformed transcripts and untranscribed extra trailers are muted; the final
    music bed fills those intervals without beginning halfway through dialogue.
    """
    return _best_preview(selector, sources, shot_lookup, text, _seconds(seconds), action, dialogue)[1]


def cold_open_plan(selector, pools: list[list[Path]], shot_lookup: dict[str, list], text: str,
                   action: bool, dialogue: dict[str, list] | None = None, seconds: float = 5) -> list[dict]:
    """Open on two strong, different titles in five seconds before any voiceover.

    Shorter requests use one continuous image instead of creating tiny flashes.
    Frame-based division keeps the returned duration exactly on the export grid.
    """
    total = _seconds(seconds)
    count = min(2, sum(bool(pool) for pool in pools)) if total >= 3 else 1
    if not any(pools):
        raise PipelineError("No footage is available for the opening preview.")
    frames = round(total * FPS)
    first_frames = math.ceil(frames / count)
    # All films are compared at the same duration so rankings remain comparable.
    choices = []
    for film_index, pool in enumerate(pools):
        if pool:
            score, shot = _best_preview(selector, pool, shot_lookup, text, first_frames/FPS, action, dialogue)
            choices.append((score, -film_index, film_index, shot))
    choices.sort(key=lambda value: value[:2], reverse=True)
    result = []
    remaining = frames
    for _, _, film_index, shot in choices[:count]:
        length = min(first_frames, remaining)
        result.append({**shot, "duration": length/FPS, "film_index": film_index})
        remaining -= length
    return result


def outro_recap_plan(selector, pools: list[list[Path]], shot_lookup: dict[str, list],
                     paragraph_seconds: list[float], text: str, action: bool = False) -> list[list[dict]]:
    """Revisit at most two titles calmly across the entire closing narration.

    Typical 14–20 second outros get two 7–10 second visual beats. Shorter
    closings stay on one source, and longer closings keep the same two titles
    rather than adding repetitive cuts. Paragraph boundaries advance within
    the current shot; they do not restart the scene or force a new movie.

    ``action`` is accepted for call-site consistency, but the outro deliberately
    favors stable, lower-motion imagery even for an action countdown.
    """
    paragraph_frames = [max(0, math.ceil(_number(seconds) * FPS)) for seconds in paragraph_seconds]
    total_frames = sum(paragraph_frames)
    if not total_frames:
        return [[] for _ in paragraph_frames]
    available = [(index, pool) for index, pool in enumerate(pools) if pool]
    if not available:
        raise PipelineError("No footage is available for the closing recap.")
    count = 2 if len(available) >= 2 and total_frames >= 14 * FPS else 1
    first_frames = math.ceil(total_frames / count)
    sample_seconds = min(8.0, first_frames / FPS)
    choices = []
    for film_index, pool in available:
        score, preview = _best_preview(selector, pool, shot_lookup, text, sample_seconds, False, None)
        choices.append((score, -film_index, film_index, preview))
    choices.sort(key=lambda value: value[:2], reverse=True)
    beats = []
    remaining = total_frames
    for _, _, film_index, preview in choices[:count]:
        frames = min(first_frames, remaining)
        shots = shot_lookup.get(preview["source"], [])
        enclosing = [_number(shot.get("end")) - preview["start"] for shot in shots
                     if isinstance(shot, dict) and _number(shot.get("start")) <= preview["start"] < _number(shot.get("end"))]
        # Hold the last frame of a short clean source shot instead of cycling to
        # another scene just to fill a few seconds of the closing sentence.
        playable = max(enclosing, default=sample_seconds)
        live_frames = min(frames, max(1, round(playable * FPS)))
        beats.append({"source": preview["source"], "start": preview["start"],
                      "film_index": film_index, "frames": frames, "live_frames": live_frames})
        remaining -= frames
    plans = []
    beat_index = 0
    used = 0
    for frames in paragraph_frames:
        plan = []
        remaining = frames
        while remaining:
            beat = beats[beat_index]
            take = min(remaining, beat["frames"] - used)
            live = max(0, min(take, beat["live_frames"] - used))
            start_frame = min(used, beat["live_frames"] - 1)
            plan.append({"source": beat["source"], "start": beat["start"] + start_frame / FPS,
                         "duration": take / FPS, "freeze": (take - live) / FPS if live else 0,
                         "still": live == 0, "film_index": beat["film_index"],
                         "visual_reason": "calm-outro-recap"})
            remaining -= take
            used += take
            if used == beat["frames"]:
                beat_index += 1
                used = 0
        plans.append(plan)
    return plans
