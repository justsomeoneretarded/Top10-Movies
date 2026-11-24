from __future__ import annotations

import json
import math
import os
import re
from pathlib import Path

from . import download_access
from .core import PipelineError, duration, normalized, probe, save_json
from .video_settings import export_size, require_disk

from .media_discovery import (DiscoveryStats, REJECT, fair_candidates, identity_names,
                              identity_ok, rejection_reason)
MAX_TRAILER_BYTES = 800_000_000
MAX_CANDIDATES = 8
MAX_QUALITY_DOWNLOADS = 3


def search_query(film: dict, extra: str = "official trailer") -> str:
    """Legacy dated query helper; broad discovery uses editorial_policy.search_queries."""
    from .editorial_policy import season_hint
    return " ".join(str(part) for part in (film["title"], film["year"], season_hint(film), extra) if part)


def candidate_ok(info: dict, film: dict, curated: bool = False) -> bool:
    return rejection_reason(info, film, curated=curated) is None


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


class _RouteBlocked(Exception):
    """The current network route met a sign-in/bot check before any trailer succeeded."""


def _is_youtube(url: str) -> bool:
    return "youtube.com/" in url or "youtu.be/" in url


def trailer_url(trailer: dict) -> str | None:
    """Watch URL for a TMDB video record; YouTube first, Vimeo as a non-YouTube fallback."""
    key = str(trailer.get("key", ""))
    site = trailer.get("site", "YouTube")
    if site == "YouTube" and re.fullmatch(r"[\w-]{11}", key):
        return f"https://www.youtube.com/watch?v={key}"
    if site == "Vimeo" and re.fullmatch(r"\d{5,12}", key):
        return f"https://vimeo.com/{key}"
    return None


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
    stills_cached = False
    if _valid_cached(path):
        try:
            stills_cached = bool(json.loads(path.with_suffix(".json").read_text()).get("stills_fallback"))
        except (OSError, ValueError):
            stills_cached = False
        if not stills_cached:
            return path
        # A warm-up job had to use stills; try the real trailer once more before keeping them.
    require_disk(root)
    base = {"quiet": True, "no_warnings": True, "noplaylist": True, "socket_timeout": 30, "retries": 3, "fragment_retries": 3, "js_runtimes": {"node": {}}}
    tried = []
    state = {"attempts": 0}
    last_error: Exception | None = None
    for route in download_access.routes():
        if state["attempts"] >= MAX_QUALITY_DOWNLOADS:
            break
        try:
            result = _download_route(film, root, path, height, download_access.options(base, route), yt_dlp, state=state)
        except _RouteBlocked:
            last_error = _blocked()
            tried.append(route)
            if route == "direct" and download_access.fallback_proxy():
                print(f"YouTube bot check on the direct route for {film['title']}; retrying through the fallback route.", flush=True)
            continue
        except PipelineError as exc:
            # No usable copy on this route (for example every candidate errored); try the next route.
            last_error = exc
            tried.append(route)
            continue
        download_access.record_route_success(route)
        return result
    return _stills_or_raise(film, root, path, stills_cached, last_error or PipelineError("No trailer candidates were available on the configured routes."))


def _stills_or_raise(film: dict, root: Path, path: Path, stills_cached: bool, error: Exception) -> Path:
    """When no trailer can be downloaded, use the film's TMDB images instead of stopping the video."""
    if stills_cached and path.exists():
        return path
    if os.getenv("STILLS_FALLBACK", "true").strip().lower() == "false":
        raise error
    from .core import notice
    from .stills_trailer import build
    try:
        built = build(film, path, root)
    except Exception as exc:  # noqa: BLE001 -- report the original download failure
        print(f"Stills fallback unavailable for {film.get('title')} ({type(exc).__name__}).", flush=True)
        raise error from None
    notice("No trailer: using stills", f"{film.get('title')} ({film.get('year')}): no trailer could be downloaded "
           f"({type(error).__name__}); its section uses official TMDB images. Consider re-running or swapping this film.", "warning")
    return built


def _download_route(film: dict, root: Path, path: Path, height: int, common: dict, yt_dlp, *, state: dict | None = None) -> Path:
    state = state if state is not None else {"attempts": 0}
    candidates = []
    for trailer in film.get("trailers", [])[:MAX_CANDIDATES]:
        url = trailer_url(trailer)
        if url and url not in candidates:
            candidates.append(url)
    # Keep YouTube first, with Vimeo available after a route-specific bot check.
    candidates.sort(key=lambda url: not _is_youtube(url))
    stats = DiscoveryStats()
    ranked = []
    blocked = False
    seen = set()

    def search(query):
        if blocked:
            return []
        with yt_dlp.YoutubeDL({**common, "extract_flat": True}) as ydl:
            result = ydl.extract_info(f"ytsearch4:{query}", download=False)
        return [f"https://www.youtube.com/watch?v={entry['id']}"
                for entry in (result or {}).get("entries", []) if isinstance(entry, dict)
                and re.fullmatch(r"[\w-]{11}", str(entry.get("id", "")))]

    def search_error(exc):
        nonlocal blocked
        stats.error(exc)
        blocked = blocked or _bot_check(exc)

    from .editorial_policy import search_queries
    # One metadata result per dated, year-free and genre family before seconds.
    for url, curated in fair_candidates(candidates, search_queries(film, primary=True), search,
                                        limit=MAX_CANDIDATES, on_search_error=search_error):
        if blocked and _is_youtube(url):
            continue
        seen.add(url)
        stats.seen += 1
        try:
            with yt_dlp.YoutubeDL(common) as ydl:
                info = ydl.extract_info(url, download=False)
            reason = rejection_reason(info, film, curated=curated)
            if reason:
                stats.reject(reason)
                continue
            ranked.append((advertised_quality(info, height), len(ranked), url, curated, info))
        except (yt_dlp.utils.DownloadError, PipelineError, OSError, ValueError) as exc:
            stats.error(exc)
            if _bot_check(exc):
                blocked = True
    if blocked and not ranked:
        print(f"Primary footage discovery: {stats.summary()}.", flush=True)
        raise _RouteBlocked()
    ranked.sort(key=lambda row: (-row[0], row[1]))
    best_quality = 0
    selected = False
    for advertised, order, url, curated, info in ranked:
        if state["attempts"] >= MAX_QUALITY_DOWNLOADS:
            break
        if selected and (best_quality >= height or (advertised and advertised <= best_quality)):
            continue
        if blocked and _is_youtube(url):
            continue
        attempt = root / f"{path.stem}.candidate-{order:02}.mp4"
        _clean_candidate(attempt)
        try:
            options = {**common, "format": f"bv[height<={height}]+ba/b[height<={height}]",
                       "format_sort": ["res", "vcodec:h264"], "merge_output_format": "mp4",
                       "postprocessors": [{"key": "FFmpegVideoRemuxer", "preferedformat": "mp4"}],
                       "outtmpl": str(attempt.with_suffix(".%(ext)s")), "overwrites": True,
                       "max_filesize": MAX_TRAILER_BYTES}
            require_disk(root)
            state["attempts"] += 1
            stats.download_attempts += 1
            with yt_dlp.YoutubeDL(options) as ydl:
                ydl.download([url])
            if not attempt.exists() or not 25 <= duration(attempt) <= 360:
                stats.reject("download_file")
                raise PipelineError("Downloaded trailer failed file/duration checks.")
            if attempt.stat().st_size > MAX_TRAILER_BYTES:
                stats.reject("download_size")
                raise PipelineError("Trailer exceeds the 800 MB per-film storage budget.")
            metadata = probe(attempt)
            video = next((stream for stream in metadata.get("streams", []) if stream.get("codec_type") == "video"), {})
            actual = effective_height(video.get("width"), video.get("height"))
            if not actual:
                stats.reject("download_quality")
                raise PipelineError("Downloaded trailer has no valid video dimensions.")
            if actual > best_quality:
                attempt.replace(path)
                best_quality = actual
                selected = True
                save_json(path.with_suffix(".json"), {"source": url, "title": info.get("title"), "channel": info.get("channel"),
                          "curated": curated, "requested_height": height, "source_width": video.get("width"),
                          "source_height": video.get("height"), "quality_selection_version": 2})
        except (yt_dlp.utils.DownloadError, PipelineError, OSError, ValueError) as exc:
            stats.error(exc)
            if _bot_check(exc):
                blocked = True
                if selected:
                    break
        finally:
            _clean_candidate(attempt)
    print(f"Primary footage discovery: {stats.summary()}.", flush=True)
    if selected:
        print(f"Selected trailer provides about {best_quality:.0f}p of landscape detail; final export quality will be chosen after all trailers are checked.", flush=True)
        return path
    path.unlink(missing_ok=True)
    path.with_suffix(".json").unlink(missing_ok=True)
    if blocked:
        raise _RouteBlocked()
    raise PipelineError(f"No usable trailer for {film['title']} ({film['year']}). {stats.summary()}. No movie was substituted.")
