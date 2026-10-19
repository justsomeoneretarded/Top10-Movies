from __future__ import annotations

import os
import re
from pathlib import Path

from .core import PipelineError, duration, normalized, save_json

REJECT = re.compile(r"\b(fan.?made|concept|ai.?generated|unofficial|reaction|review|explained)\b", re.IGNORECASE)


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


def download(film: dict, root: Path) -> Path:
    import yt_dlp
    root.mkdir(parents=True, exist_ok=True)
    path = root / f"{film['kind']}-{film['id']}.mp4"
    if path.exists() and 25 <= duration(path) <= 360:
        return path
    common = {"quiet": True, "no_warnings": True, "noplaylist": True, "socket_timeout": 30, "retries": 3, "fragment_retries": 3, "js_runtimes": {"node": {}}}
    cookies = os.getenv("YOUTUBE_COOKIE_FILE")
    if cookies:
        common["cookiefile"] = cookies
    candidates = []
    for trailer in film["trailers"][:8]:
        video_id = trailer.get("key", "")
        if re.fullmatch(r"[\w-]{11}", video_id):
            candidates.append((f"https://www.youtube.com/watch?v={video_id}", True))
    # Search fallback is bounded and requires verified-channel metadata.
    if not candidates:
        with yt_dlp.YoutubeDL({**common, "extract_flat": True}) as ydl:
            result = ydl.extract_info(f"ytsearch6:{film['title']} {film['year']} official trailer", download=False)
            for entry in (result or {}).get("entries", []):
                if entry and re.fullmatch(r"[\w-]{11}", entry.get("id", "")):
                    candidates.append((f"https://www.youtube.com/watch?v={entry['id']}", False))
    errors = []
    for url, curated in candidates:
        try:
            with yt_dlp.YoutubeDL(common) as ydl:
                info = ydl.extract_info(url, download=False)
            if not info or not candidate_ok(info, film, curated):
                continue
            options = {**common, "format": "bv*[height<=720][vcodec^=avc1]+ba/b[height<=720]/bv*[height<=720]+ba", "merge_output_format": "mp4", "outtmpl": str(path.with_suffix(".%(ext)s")), "overwrites": True, "max_filesize": 300_000_000}
            with yt_dlp.YoutubeDL(options) as ydl:
                ydl.download([url])
            if not path.exists() or not 25 <= duration(path) <= 360:
                raise PipelineError("Downloaded trailer failed file/duration checks.")
            save_json(path.with_suffix(".json"), {"source": url, "title": info.get("title"), "channel": info.get("channel"), "curated": curated})
            return path
        except (yt_dlp.utils.DownloadError, PipelineError, OSError, ValueError) as exc:
            errors.append(type(exc).__name__)
            path.unlink(missing_ok=True)
    raise PipelineError(f"No usable trailer for {film['title']} ({film['year']}). Tried {len(candidates)} candidates. YouTube access restrictions, unavailable trailers or format changes may be responsible. No movie was substituted. Failures: {', '.join(errors)}")
