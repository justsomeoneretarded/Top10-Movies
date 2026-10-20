from __future__ import annotations

import json
import math
import os
import re
from pathlib import Path

from .core import PipelineError, duration, normalized, probe, save_json
from .video_settings import export_size, require_disk

REJECT = re.compile(r"\b(fan.?made|concept|ai.?generated|unofficial|reaction|review|explained)\b", re.IGNORECASE)
MAX_TRAILER_BYTES = 800_000_000
MAX_CANDIDATES = 8
MAX_QUALITY_DOWNLOADS = 3


def candidate_ok(info: dict, film: dict, curated: bool = False) -> bool:
    title = info.get("title", "")
    length = info.get("duration") or 0
    if REJECT.search(title) or info.get("is_live") or not 25 <= length <= 360:
        return False
    if curated:
        return True
    words = set(normalized(film["title"]).split())
    candidate_words = set(normalized(title).split())
    overlap = len(words & candidate_words) / max(1, len(words))
    return overlap >= 0.85 and "trailer" in candidate_words and bool(info.get("channel_is_verified"))


def effective_height(width, height) -> float:
    """Pixels available for a landscape 16:9 crop, rather than a height-only label."""
    if (isinstance(width, bool) or isinstance(height, bool) or not isinstance(width, (int, float))
            or not isinstance(height, (int, float)) or not math.isfinite(width) or not math.isfinite(height)
            or width <= 0 or height <= 0):
        return 0
    return min(float(height), float(width)*9/16)


def advertised_quality(info: dict, cap: int) -> float:
    """Prefer a source's best usable in-cap format without downloading every trailer."""
    formats = info.get("formats")
    rows = formats if isinstance(formats, list) else [info]
    choices = []
    for row in rows:
        if not isinstance(row, dict) or row.get("vcodec") == "none":
            continue
        width, height = row.get("width"), row.get("height")
        quality = effective_height(width, height)
        if not quality or height > cap:
            continue
        estimated = row.get("filesize") or row.get("filesize_approx")
        if isinstance(estimated, (int, float)) and estimated > MAX_TRAILER_BYTES:
            continue
        choices.append(quality)
    return max(choices, default=0)


def _bot_check(exc: Exception) -> bool:
    return "confirm you're not a bot" in str(exc).casefold().replace("’", "'")


def _blocked():
    return PipelineError(
        "YouTube blocked this runner with a sign-in/bot check. "
        "Trying more video IDs will not resolve a runner-wide block. "
        "Optionally configure the private YOUTUBE_COOKIES Actions secret with a valid "
        "Netscape cookies.txt export and start a new run from main. "
        "Cookies may expire or still be rejected on GitHub-hosted runners; "
        "use a self-hosted runner or a permitted alternative footage source if access remains blocked. "
        "See README: YouTube sign-in/bot errors."
    )


def _clean_candidate(path: Path):
    for partial in path.parent.glob(path.stem + ".*"):
        if partial.is_file():
            partial.unlink(missing_ok=True)


def _valid_cached(path: Path) -> bool:
    try:
        record = json.loads(path.with_suffix(".json").read_text())
        return (path.exists() and record.get("quality_selection_version") == 2
                and effective_height(record.get("source_width"), record.get("source_height")) > 0
                and 25 <= duration(path) <= 360 and path.stat().st_size <= MAX_TRAILER_BYTES)
    except (OSError, ValueError, TypeError, AttributeError, PipelineError):
        return False


def download(film: dict, root: Path) -> Path:
    import yt_dlp
    root.mkdir(parents=True, exist_ok=True)
    height = export_size()[1]
    path = root / f"{film['kind']}-{film['id']}-{height}p.mp4"
    if _valid_cached(path):
        return path
    require_disk(root)
    common = {"quiet": True, "no_warnings": True, "noplaylist": True, "socket_timeout": 30, "retries": 3, "fragment_retries": 3, "js_runtimes": {"node": {}}}
    cookies = os.getenv("YOUTUBE_COOKIE_FILE")
    if cookies:
        common["cookiefile"] = cookies
    candidates = []
    for trailer in film["trailers"][:MAX_CANDIDATES]:
        video_id = trailer.get("key", "")
        if re.fullmatch(r"[\w-]{11}", video_id):
            entry = (f"https://www.youtube.com/watch?v={video_id}", True)
            if entry not in candidates:
                candidates.append(entry)
    errors = []
    ranked = []
    blocked = False
    seen = set()

    def inspect(entries):
        nonlocal blocked
        for url, curated in entries:
            if url in seen or len(seen) >= MAX_CANDIDATES:
                continue
            seen.add(url)
            try:
                with yt_dlp.YoutubeDL(common) as ydl:
                    info = ydl.extract_info(url, download=False)
                if info and candidate_ok(info, film, curated):
                    ranked.append((advertised_quality(info, height), len(ranked), url, curated, info))
            except (yt_dlp.utils.DownloadError, PipelineError, OSError, ValueError) as exc:
                errors.append(type(exc).__name__)
                if _bot_check(exc):
                    blocked = True
                    break

    inspect(candidates)
    # If curated choices are missing or low quality, spend remaining metadata slots
    # on verified official search matches. Search results never bypass title checks.
    if not blocked and len(seen) < MAX_CANDIDATES and max((r[0] for r in ranked), default=0) < height:
        try:
            with yt_dlp.YoutubeDL({**common, "extract_flat": True}) as ydl:
                result = ydl.extract_info(f"ytsearch6:{film['title']} {film['year']} official trailer", download=False)
            discovered = []
            for entry in (result or {}).get("entries", []):
                if entry and re.fullmatch(r"[\w-]{11}", entry.get("id", "")):
                    discovered.append((f"https://www.youtube.com/watch?v={entry['id']}", False))
            inspect(discovered)
        except (yt_dlp.utils.DownloadError, PipelineError, OSError, ValueError) as exc:
            errors.append(type(exc).__name__)
            blocked = _bot_check(exc)
    if blocked and not ranked:
        raise _blocked()
    ranked.sort(key=lambda row: (-row[0], row[1]))
    best_quality = 0
    quality_downloads = 0
    selected = False
    for advertised, order, url, curated, info in ranked:
        if selected and (best_quality >= height or (advertised and advertised <= best_quality)
                         or quality_downloads >= MAX_QUALITY_DOWNLOADS):
            continue
        attempt = root / f"{path.stem}.candidate-{order:02}.mp4"
        _clean_candidate(attempt)
        try:
            options = {**common, "format": f"bv[height<={height}]+ba/b[height<={height}]",
                       "format_sort": ["res", "vcodec:h264"], "merge_output_format": "mp4",
                       "postprocessors": [{"key": "FFmpegVideoRemuxer", "preferedformat": "mp4"}],
                       "outtmpl": str(attempt.with_suffix(".%(ext)s")), "overwrites": True,
                       "max_filesize": MAX_TRAILER_BYTES}
            with yt_dlp.YoutubeDL(options) as ydl:
                ydl.download([url])
            if not attempt.exists() or not 25 <= duration(attempt) <= 360:
                raise PipelineError("Downloaded trailer failed file/duration checks.")
            if attempt.stat().st_size > MAX_TRAILER_BYTES:
                raise PipelineError("Trailer exceeds the 800 MB per-film storage budget.")
            metadata = probe(attempt)
            video = next((stream for stream in metadata.get("streams", []) if stream.get("codec_type") == "video"), {})
            actual = effective_height(video.get("width"), video.get("height"))
            if not actual:
                raise PipelineError("Downloaded trailer has no valid video dimensions.")
            quality_downloads += 1
            if actual > best_quality:
                attempt.replace(path)
                best_quality = actual
                selected = True
                save_json(path.with_suffix(".json"), {"source": url, "title": info.get("title"), "channel": info.get("channel"),
                          "curated": curated, "requested_height": height, "source_width": video.get("width"),
                          "source_height": video.get("height"), "quality_selection_version": 2})
        except (yt_dlp.utils.DownloadError, PipelineError, OSError, ValueError) as exc:
            errors.append(type(exc).__name__)
            if _bot_check(exc):
                if not selected:
                    raise _blocked() from None
                break
        finally:
            _clean_candidate(attempt)
    if selected:
        print(f"Selected trailer provides about {best_quality:.0f}p of landscape detail; final export quality will be chosen after all trailers are checked.", flush=True)
        return path
    path.unlink(missing_ok=True)
    path.with_suffix(".json").unlink(missing_ok=True)
    raise PipelineError(f"No usable trailer for {film['title']} ({film['year']}). Tried {len(seen)} candidates. YouTube access restrictions, unavailable trailers or format changes may be responsible. No movie was substituted. Failures: {', '.join(errors)}")
