"""Use optional extra footage across paragraphs without increasing cut frequency."""
from __future__ import annotations

import math
from pathlib import Path

from .core import PipelineError

OPTIONAL_ERRORS = (PipelineError, RuntimeError, ValueError, KeyError, OSError, IndexError, TypeError, AttributeError)


def _available(selector, sources: list[Path], shot_sets: dict[str, list[dict]]) -> list[Path]:
    available = []
    features = getattr(selector, "features", None)
    seen = set()
    for index, value in enumerate(sources):
        path = Path(value)
        name = str(path)
        if name in seen or not shot_sets.get(name):
            continue
        seen.add(name)
        if index and (not path.is_file() or isinstance(features, dict) and name not in features):
            continue
        available.append(path)
    return available


def _rank_sources(selector, sources: list[Path], shot_sets: dict[str, list[dict]], text: str,
                  action: bool, *, uses: dict | None = None, last: str | None = None) -> list[Path]:
    ranked = []
    for index, path in enumerate(_available(selector, sources, shot_sets)):
        try:
            score = float(selector.hook_score(path, shot_sets[str(path)], text, action))
            if not math.isfinite(score):
                raise ValueError("Non-finite visual score")
        except OPTIONAL_ERRORS:
            if index:
                continue
            # Matching is optional; the primary remains usable even when similarity fails.
            score = 0.0
        if uses is not None:
            score -= uses.get(str(path), 0)*.03
            if str(path) == last:
                score -= .03
        try:
            fresh = float(selector.fresh_seconds(path,shot_sets[str(path)]))
            fresh_tier = int(math.isfinite(fresh) and fresh >= 1.5)
        except OPTIONAL_ERRORS:
            fresh_tier = 1
        ranked.append((fresh_tier,score,index,path))
    ranked.sort(key=lambda row:(-row[0],-row[1],row[2]))
    return [row[3] for row in ranked]


def _valid_plan(plan, seconds: float) -> bool:
    if not isinstance(plan, list) or not plan:
        return False
    try:
        durations = [float(shot["duration"]) for shot in plan]
        return (all(math.isfinite(value) and value > 0 for value in durations)
                and all(math.isfinite(float(shot["start"])) and float(shot["start"]) >= 0 for shot in plan)
                and abs(sum(durations)-seconds) <= .05)
    except (KeyError, TypeError, ValueError):
        return False


def choose_from_sources(selector, sources: list[Path], shot_sets: dict[str, list[dict]],
                        text: str, seconds: float, action: bool) -> list[dict]:
    """Select one source for a complete paragraph, then keep the existing steady edit.

    Small source reuse penalties encourage variety when visual relevance is close.
    Missing or failing optional footage always falls back to the primary source.
    """
    if seconds <= 0:
        return []
    if not sources:
        raise PipelineError("No primary footage was provided for this review.")
    primary = Path(sources[0])
    pool_key = tuple(str(path) for path in sources)
    all_uses = getattr(selector, "_source_pool_uses", {})
    last_sources = getattr(selector, "_source_pool_last", {})
    uses = all_uses.setdefault(pool_key, {})
    ranked = _rank_sources(selector, sources, shot_sets, text, False,
                           uses=uses, last=last_sources.get(pool_key))
    if primary not in ranked:
        ranked.append(primary)
    for path in ranked:
        try:
            plan = selector.choose(path, shot_sets.get(str(path), []), text, seconds, action)
            if not _valid_plan(plan, seconds):
                raise PipelineError("Footage selection did not cover its narration.")
        except OPTIONAL_ERRORS:
            if path == primary:
                raise
            continue
        uses[str(path)] = uses.get(str(path), 0)+1
        last_sources[pool_key] = str(path)
        selector._source_pool_uses = all_uses
        selector._source_pool_last = last_sources
        return [{**shot, "source": str(path)} for shot in plan]
    raise PipelineError("No usable primary shot plan was available.")


def best_intro_sources(selector, pools: list[list[Path]], shot_sets: dict[str, list[dict]],
                       prompt: str, action: bool) -> list[Path]:
    """Keep one source per film, so optional trailers do not inflate intro title counts."""
    result = []
    for sources in pools:
        if not sources:
            raise PipelineError("An intro title has no primary footage.")
        ranked = _rank_sources(selector, sources, shot_sets, prompt, action)
        result.append(ranked[0] if ranked else Path(sources[0]))
    return result


def primary_fallback_plan(selector, chosen: list[dict], trailers: list[Path],
                          primary_shots: list[list[dict]], text: str, action: bool,
                          default_film: int) -> list[dict]:
    """Replace optional video sources after a decode failure, keeping each beat's time.

    Intro beats retain their own film index; meme images and primary footage stay
    unchanged. A held optional frame remains one held frame from the correct
    film's primary trailer, rather than becoming a series of faster cuts.
    """
    result = []
    for shot in chosen:
        if shot.get("image"):
            result.append(dict(shot))
            continue
        film_index = shot.get("film_index", default_film)
        if not isinstance(film_index, int) or isinstance(film_index, bool) or not 0 <= film_index < len(trailers):
            film_index = default_film
        primary = Path(trailers[film_index])
        if Path(shot.get("source") or primary) == primary:
            result.append(dict(shot))
            continue
        seconds = float(shot["duration"])
        replacement = selector.choose(primary, primary_shots[film_index], text, seconds, action,
                                      maximum_shot=seconds)
        if not _valid_plan(replacement, seconds):
            raise PipelineError("Primary fallback footage did not cover its narration beat.")
        if shot.get("still"):
            replacement = [{**replacement[0], "duration": seconds, "still": True, "freeze": 0}]
        for index, cut in enumerate(replacement):
            # Preserve exact requested beat length even with floating-point rounding.
            length = seconds-sum(float(item["duration"]) for item in replacement[:index]) if index == len(replacement)-1 else cut["duration"]
            result.append({**shot, **cut, "duration": length, "source": str(primary),
                           "film_index": film_index, "visual_reason": "primary-after-secondary-render-failure"})
    return result
