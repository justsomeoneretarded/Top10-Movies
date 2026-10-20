from __future__ import annotations

import os
import re
from pathlib import Path

from .core import PipelineError, duration, normalized, probe, save_json
from .video_settings import export_size, require_disk

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
    height = export_size()[1]
    path = root / f"{film['kind']}-{film['id']}-{height}p.mp4"
    if path.exists() and 25 <= duration(path) <= 360:
        return path
    require_disk(root)
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
            options = {**common, "format": f"bv[height<={height}]+ba/b[height<={height}]",
                       "format_sort": ["res", "vcodec:h264"], "merge_output_format": "mp4",
                       "postprocessors": [{"key": "FFmpegVideoRemuxer", "preferedformat": "mp4"}],
                       "outtmpl": str(path.with_suffix(".%(ext)s")), "overwrites": True,
                       "max_filesize": 800_000_000}
            with yt_dlp.YoutubeDL(options) as ydl:
                ydl.download([url])
            if not path.exists() or not 25 <= duration(path) <= 360:
                raise PipelineError("Downloaded trailer failed file/duration checks.")
            if path.stat().st_size > 800_000_000:
                raise PipelineError("Trailer exceeds the 800 MB per-film storage budget.")
            metadata = probe(path)
            video = next((stream for stream in metadata.get("streams", []) if stream.get("codec_type") == "video"), {})
            actual_height = video.get("height", 0)
            if actual_height < height:
                print(f"Trailer source is {actual_height}p; {height}p export will upscale it without adding real detail.", flush=True)
            save_json(path.with_suffix(".json"), {"source": url, "title": info.get("title"), "channel": info.get("channel"),
                      "curated": curated, "requested_height": height, "source_width": video.get("width"), "source_height": actual_height})
            return path
        except (yt_dlp.utils.DownloadError, PipelineError, OSError, ValueError) as exc:
            # Failed adaptive downloads can leave separate video/audio streams and .part files.
            for partial in root.glob(path.stem + ".*"):
                if partial.is_file():
                    partial.unlink(missing_ok=True)
            message = str(exc).casefold().replace("’", "'")
            if "confirm you're not a bot" in message or "confirm you’re not a bot" in message:
                raise PipelineError(
                    "YouTube blocked this runner with a sign-in/bot check. "
                    "Trying more video IDs will not resolve a runner-wide block. "
                    "Optionally configure the private YOUTUBE_COOKIES Actions secret with a valid "
                    "Netscape cookies.txt export and start a new run from main. "
                    "Cookies may expire or still be rejected on GitHub-hosted runners; "
                    "use a self-hosted runner or a permitted alternative footage source if access remains blocked. "
                    "See README: YouTube sign-in/bot errors."
                ) from None
            errors.append(type(exc).__name__)
            path.unlink(missing_ok=True)
    raise PipelineError(f"No usable trailer for {film['title']} ({film['year']}). Tried {len(candidates)} candidates. YouTube access restrictions, unavailable trailers or format changes may be responsible. No movie was substituted. Failures: {', '.join(errors)}")
