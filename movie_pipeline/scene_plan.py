"""Bounded actual-scene preselection before script writing and voice generation."""
from __future__ import annotations

from copy import deepcopy
import gc
import hashlib
import json
import math
from pathlib import Path
import time

from .core import key, save_json
from .moments import _enrich
from . import scenes

VERSION = 1


def _film_key(film):
    return key(film.get("kind", "movie"), film.get("id"), film.get("title"), film.get("year"))


def _sha256(path, deadline=None):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        while True:
            if deadline is not None and time.monotonic() >= deadline:
                raise TimeoutError("Optional scene-plan budget reached")
            block = stream.read(1024 * 1024)
            if not block:
                break
            digest.update(block)
    return digest.hexdigest()


def _safe_window(clip, shots):
    try:
        start, seconds = float(clip["start"]), float(clip["duration"])
        end = start + seconds
        if not all(math.isfinite(v) for v in (start, seconds)) or start < 0 or not 3 <= seconds <= 22:
            return False
        covered = start
        for shot in sorted(shots, key=lambda row: float(row["start"])):
            low, high = float(shot["start"]), float(shot["end"])
            if not all(math.isfinite(v) for v in (low, high)) or high <= low:
                return False
            if high <= start or low >= end:
                continue
            if shot.get("text_card") or low > covered + .05:
                return False
            covered = max(covered, high)
        return covered >= end - .05
    except (KeyError, TypeError, ValueError):
        return False


def validated_priorities(scene_map, film_index, pool, shot_lookup, film, *, deadline=None, hash_cache=None):
    """Recheck source bytes, title identity and source windows before final placement."""
    if not isinstance(scene_map, dict) or scene_map.get("version") != VERSION:
        return []
    by_film = scene_map.get("films")
    if not isinstance(by_film, dict) or not isinstance(by_film.get(str(film_index), []), list):
        return []
    allowed = {str(Path(path)) for path in pool}
    hashes = {} if hash_cache is None else hash_cache
    result = []
    for row in by_film.get(str(film_index), [])[:2]:
        try:
            if not isinstance(row, dict) or row.get("film_key") != _film_key(film):
                continue
            source = str(Path(row["source"]))
            if source not in allowed or not _safe_window(row, shot_lookup.get(source, [])):
                continue
            if source not in hashes:
                hashes[source] = _sha256(Path(source), deadline)
            digest = hashes[source]
            identifier = key("scene-plan", VERSION, _film_key(film), digest, row["start"], row["duration"])
            if row.get("source_sha256") != digest or row.get("id") != identifier:
                continue
            cues = row.get("cues")
            if not isinstance(cues, list) or row.get("muted") is not False or row.get("native_verified") is not True:
                continue
            valid_cues = True
            for cue in cues:
                if not isinstance(cue, dict) or not isinstance(cue.get("text"), str):
                    valid_cues = False
                    break
                begin, end = float(cue["start"]), float(cue["end"])
                if (not all(math.isfinite(v) for v in (begin, end)) or begin < float(row["start"])
                        or not begin < end <= float(row["start"]) + float(row["duration"]) + .05):
                    valid_cues = False
                    break
            if not valid_cues:
                continue
            clip = deepcopy(row)
            clip["start"], clip["duration"] = float(row["start"]), float(row["duration"])
            clip["film_index"] = film_index
            clip["extent"] = max(float(shot["end"]) for shot in shot_lookup[source])
            result.append(clip)
        except (OSError, ValueError, TypeError, KeyError, RuntimeError):
            continue
    return result


def projected_context(scene_map):
    """Writing sees bounded clip descriptions and dialogue guesses, never local paths."""
    count = scene_map.get("film_count", 0) if isinstance(scene_map, dict) else 0
    by_film = scene_map.get("films", {}) if isinstance(scene_map, dict) else {}
    if type(count) is not int or not 0 <= count <= 15 or not isinstance(by_film, dict):
        return []
    result = []
    for index in range(count):
        rows = by_film.get(str(index), [])
        context = []
        for row in rows[:2] if isinstance(rows, list) else []:
            try:
                if not isinstance(row, dict) or not isinstance(row.get("id"), str):
                    continue
                seconds = float(row["duration"])
                if not math.isfinite(seconds) or not 3 <= seconds <= 22:
                    continue
                context.append({"id": row["id"], "description": str(row.get("description", ""))[:180],
                                "duration": round(seconds, 2),
                                "dialogue_excerpt": " ".join(str(cue.get("text", "")) for cue in row.get("cues", [])
                                    if isinstance(cue, dict))[:220]})
            except (ValueError, TypeError, KeyError):
                continue
        result.append(context)
    return result


def _candidate_groups(pool, shot_lookup):
    groups = []
    for source in pool:
        path = Path(source)
        shots = shot_lookup.get(str(path), [])
        if not shots:
            continue
        try:
            data = scenes.load(path, len(shots))
            descriptions = data.get("descriptions", []) if isinstance(data, dict) else []
        except (OSError, ValueError, TypeError, KeyError, AttributeError):
            descriptions = []
        rows = []
        extent = max(float(shot["end"]) for shot in shots)
        for index, shot in enumerate(shots):
            try:
                start, end = float(shot["start"]), float(shot["end"])
                last = index
                while last + 1 < len(shots):
                    nxt = shots[last+1]
                    if (nxt.get("text_card") or nxt.get("continues_previous") is not True
                            or shot.get("detected_shot") is None
                            or nxt.get("detected_shot") != shot.get("detected_shot")
                            or abs(float(shots[last]["end"])-float(nxt["start"])) > 1e-6):
                        break
                    end = min(start+22, float(nxt["end"]))
                    last += 1
                    if end >= start+22:
                        break
                seconds = end-start
                if shot.get("text_card") or not 3 <= seconds <= 22:
                    continue
                description = descriptions[index] if isinstance(descriptions, list) and index < len(descriptions) else ""
                description = " ".join(description.split())[:180] if isinstance(description, str) else ""
                peak = float(shot.get("peak", 0))
                peak = max(0, min(1, peak)) if math.isfinite(peak) else 0
                # Readability and actual visual context matter; loudness alone is a small signal.
                score = min(seconds, 12)/12 + .35*bool(description) + .15*peak
                rows.append({"source": str(path), "start": start, "duration": seconds,
                             "description": description, "score": score, "scene_index": index,
                             "scene_end": end, "extent": extent, "film_index": None,
                             "action_strength": 0., "motion": float(shot.get("motion", 0)),
                             "muted": True, "cues": []})
            except (ValueError, TypeError, KeyError):
                continue
        if rows:
            groups.append(sorted(rows, key=lambda row: -row["score"]))
    return sorted(groups, key=lambda rows: -rows[0]["score"])[:2]


def prepare_scene_map(pools, shot_lookup, films, work, recognizer_factory=None, *, budget_seconds=180):
    """Choose real sound/visual windows before writing; release ASR before the writer."""
    work = Path(work)
    result = {"version": VERSION, "film_count": len(films), "films": {}, "report": {}}
    deadline = time.monotonic() + max(0., float(budget_seconds))
    hashes = {}
    recognizer, attempted_factory = None, False
    cache = work / "scene-plan.json"
    try:
        old = json.loads(cache.read_text()) if cache.exists() else {}
    except (OSError, ValueError, TypeError):
        old = {}
    try:
        work.mkdir(parents=True, exist_ok=True)
        for index, (film, pool) in enumerate(zip(films, pools)):
            chosen = validated_priorities(old, index, pool, shot_lookup, film,
                                          deadline=deadline, hash_cache=hashes)
            groups = _candidate_groups(pool, shot_lookup)
            rows = [group[level] for level in range(2) for group in groups if level < len(group)]
            for candidate in rows:
                if len(chosen) >= 2 or time.monotonic() >= deadline:
                    break
                source = candidate["source"]
                if any(clip["source"] == source and clip["start"] < candidate["start"] + candidate["duration"]
                       and clip["start"] + clip["duration"] > candidate["start"] for clip in chosen):
                    continue
                try:
                    if source not in hashes:
                        hashes[source] = _sha256(Path(source), deadline)
                    if not attempted_factory:
                        attempted_factory = True
                        recognizer = recognizer_factory() if recognizer_factory is not None else None
                    if recognizer is None:
                        break
                    cue_work = work / "window-cues" / hashes[source][:16]
                    cue_work.mkdir(parents=True, exist_ok=True)
                    clip = _enrich(candidate, {}, recognizer, cue_work)
                    if clip is None or not _safe_window(clip, shot_lookup.get(source, [])):
                        continue
                    if any(old_clip["source"] == source and old_clip["start"] < clip["start"] + clip["duration"]
                           and old_clip["start"] + old_clip["duration"] > clip["start"] for old_clip in chosen):
                        continue
                    clip.update(film_index=index, film_key=_film_key(film), source_sha256=hashes[source])
                    clip["id"] = key("scene-plan", VERSION, clip["film_key"], hashes[source], clip["start"], clip["duration"])
                    chosen.append(clip)
                except (OSError, ValueError, TypeError, KeyError, RuntimeError):
                    continue
            result["films"][str(index)] = chosen[:2]
    except (OSError, ValueError, TypeError, KeyError, RuntimeError):
        pass
    finally:
        recognizer = None
        gc.collect()
    result["report"] = {"selected": sum(len(rows) for rows in result["films"].values()),
                        "hashed_sources": len(hashes), "budget_exhausted": time.monotonic() >= deadline}
    try:
        save_json(cache, result)
    except OSError:
        pass
    return result
