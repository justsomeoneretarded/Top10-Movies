"""Best-effort second trailer: different imagery, same title, and no quality downgrade."""
from __future__ import annotations

import json
import math
import re
from pathlib import Path
from urllib.parse import parse_qs, urlsplit

import cv2
import numpy as np

from .core import PipelineError, duration, normalized, probe, save_json
from . import download_access
from .footage import MAX_TRAILER_BYTES, _bot_check, _clean_candidate
from .video_settings import export_size, require_disk

REJECT = re.compile(r"\b(concept|ai[ -]?generated|reaction|review|explained|deepfake)\b", re.IGNORECASE)
FAN = re.compile(r"\b(?:fan[ -]?(?:made|edit)|unofficial)\b", re.IGNORECASE)
MAX_METADATA = 8
MAX_DOWNLOADS = 2


def secondary_candidate_ok(info: dict, film: dict, curated: bool = False) -> bool:
    title = info.get("title", "")
    if not isinstance(title, str) or REJECT.search(title) or info.get("is_live"):
        return False
    length = info.get("duration") or 0
    if not isinstance(length, (int, float)) or not 25 <= length <= 360:
        return False
    years = set(re.findall(r"\b(?:18|19|20)\d{2}\b", title))
    years -= set(re.findall(r"\b(?:18|19|20)\d{2}\b", film["title"]))
    if years and str(film["year"]) not in years:
        return False
    if curated:
        return True
    expected = set(normalized(film["title"]).split())
    found = set(normalized(title).split())
    overlap = len(expected & found)/max(1, len(expected))
    if overlap < .85 or "trailer" not in found:
        return False
    if info.get("channel_is_verified"):
        return True
    return bool(FAN.search(title)) and str(film["year"]) in title


def _hashes(path: Path) -> list[int]:
    """Sample useful frames and compute tiny perceptual difference fingerprints."""
    length = duration(path)
    capture = cv2.VideoCapture(str(path))
    hashes = []
    try:
        for seconds in np.linspace(length*.10, length*.85, 18):
            capture.set(cv2.CAP_PROP_POS_MSEC, float(seconds)*1000)
            ok, image = capture.read()
            if not ok:
                continue
            gray = cv2.cvtColor(cv2.resize(image, (160,90)), cv2.COLOR_BGR2GRAY)
            if float(gray.std()) < 8 or float(gray.mean()) < 12:
                continue
            small = cv2.resize(gray, (9,8), interpolation=cv2.INTER_AREA)
            bits = (small[:,1:] > small[:,:-1]).reshape(-1)
            value = sum(int(bit) << index for index, bit in enumerate(bits))
            # Count a held/repeated scene once, rather than treating every frame as new.
            if not any((value ^ old).bit_count() <= 4 for old in hashes):
                hashes.append(value)
    finally:
        capture.release()
    return hashes


def novelty_from_hashes(primary: list[int], candidate: list[int]) -> dict | None:
    if len(primary) < 4 or len(candidate) < 4:
        return None
    novel = sum(not any((value ^ old).bit_count() <= 8 for old in primary)
                for value in candidate)
    fraction = novel/len(candidate)
    return {"novel_frames": novel, "sampled_frames": len(candidate), "novel_fraction": fraction,
            "distinct": novel >= 3 and fraction >= .25}


def scene_novelty(primary: Path, candidate: Path) -> dict | None:
    """A visual diversity heuristic; this does not verify actor or movie identity."""
    return novelty_from_hashes(_hashes(primary), _hashes(candidate))


def _quality(info: dict, minimum_size: tuple[int,int], cap: int) -> float | None:
    width, height = minimum_size
    formats = info.get("formats")
    rows = formats if isinstance(formats, list) else [info]
    known = False
    choices = []
    for row in rows:
        if not isinstance(row, dict) or row.get("vcodec") == "none":
            continue
        w, h = row.get("width"), row.get("height")
        if not isinstance(w, (int,float)) or not isinstance(h, (int,float)) or not math.isfinite(w) or not math.isfinite(h):
            continue
        known = True
        size = row.get("filesize") or row.get("filesize_approx")
        if w >= width and height <= h <= cap and not (isinstance(size,(int,float)) and size > MAX_TRAILER_BYTES):
            choices.append(min(h,w*9/16))
    return max(choices) if choices else (None if known else 0)


def _primary_id(source: str) -> str:
    parsed = urlsplit(source)
    if parsed.hostname in {"youtube.com", "www.youtube.com"}:
        return parse_qs(parsed.query).get("v", [""])[0]
    if parsed.hostname == "youtu.be":
        return parsed.path.strip("/")
    return ""


def _valid_cache(path: Path, primary_source: str, minimum_size: tuple[int,int]) -> bool:
    try:
        record = json.loads(path.with_suffix('.json').read_text())
        return (record.get('secondary_version') == 1 and record.get('primary_source') == primary_source
                and record.get('novelty',{}).get('distinct') is True
                and record.get('source_width',0) >= minimum_size[0] and record.get('source_height',0) >= minimum_size[1]
                and path.exists() and 0 < path.stat().st_size <= MAX_TRAILER_BYTES and 25 <= duration(path) <= 360)
    except (OSError, ValueError, TypeError, AttributeError, PipelineError):
        return False


def download_secondary(film: dict, root: Path, primary: Path, minimum_size: tuple[int,int]) -> Path | None:
    """Keep at most one optional extra; every discovery or media failure returns None."""
    try:
        return _download_secondary(film, root, primary, minimum_size)
    except Exception:  # noqa: BLE001 -- optional enrichment must never discard the primary/video.
        print(f"No additional usable trailer for {film.get('title', 'this title')}; continuing with one.", flush=True)
        return None


def _download_secondary(film: dict, root: Path, primary: Path, minimum_size: tuple[int,int]) -> Path | None:
    import yt_dlp
    record = json.loads(primary.with_suffix('.json').read_text())
    if record.get('stills_fallback'):
        return None  # no trailer was reachable for this film; do not search again
    source = record.get('source')
    if not isinstance(source,str) or not source:
        return None
    root.mkdir(parents=True,exist_ok=True)
    target = root / f'{primary.stem}-extra.mp4'
    if _valid_cache(target,source,minimum_size):
        return target
    _clean_candidate(target)
    require_disk(root)
    cap = export_size()[1]
    # Optional extra footage uses whichever route already worked for primary trailers.
    common = download_access.options({'quiet':True,'no_warnings':True,'noplaylist':True,'socket_timeout':12,'retries':1,
                                      'fragment_retries':1,'js_runtimes':{'node':{}}}, download_access.routes()[0])
    seen = {_primary_id(source)}
    ranked = []
    blocked = False

    def inspect(entries):
        nonlocal blocked
        for identifier,curated in entries:
            if identifier in seen or len(seen)-1 >= MAX_METADATA or not re.fullmatch(r'[\w-]{11}', identifier):
                continue
            seen.add(identifier)
            url = f'https://www.youtube.com/watch?v={identifier}'
            try:
                with yt_dlp.YoutubeDL(common) as ydl:
                    info = ydl.extract_info(url,download=False)
                if info and secondary_candidate_ok(info,film,curated):
                    quality = _quality(info,minimum_size,cap)
                    if quality is not None:
                        ranked.append((quality,len(ranked),url,curated,info))
            except (yt_dlp.utils.DownloadError,PipelineError,OSError,ValueError) as exc:
                # Primary footage already exists. A secondary bot check is never fatal.
                if _bot_check(exc):
                    blocked = True
                    break
                continue

    inspect([(trailer.get('key',''),True) for trailer in film.get('trailers',[])[:MAX_METADATA] if trailer.get('site','YouTube') == 'YouTube'])
    # Reserve actual metadata capacity for fan edits when curated discovery has few options.
    if not blocked and len(seen)-1 < MAX_METADATA:
        try:
            with yt_dlp.YoutubeDL({**common,'extract_flat':True}) as ydl:
                result = ydl.extract_info(f"ytsearch6:{film['title']} {film['year']} trailer",download=False)
            inspect([(entry.get('id',''),False) for entry in (result or {}).get('entries',[]) if entry])
        except (yt_dlp.utils.DownloadError,PipelineError,OSError,ValueError):
            blocked = True
    ranked.sort(key=lambda row:(-row[0],row[1]))
    for _quality_value,order,url,curated,info in ranked[:MAX_DOWNLOADS]:
        attempt = root / f'{primary.stem}-extra-candidate-{order:02}.mp4'
        _clean_candidate(attempt)
        try:
            options = {**common,'format':f'bv[height<={cap}]+ba/b[height<={cap}]',
                       'format_sort':['res','vcodec:h264'],'merge_output_format':'mp4',
                       'postprocessors':[{'key':'FFmpegVideoRemuxer','preferedformat':'mp4'}],
                       'outtmpl':str(attempt.with_suffix('.%(ext)s')),'overwrites':True,'max_filesize':MAX_TRAILER_BYTES}
            with yt_dlp.YoutubeDL(options) as ydl:
                ydl.download([url])
            if not attempt.exists() or not 25 <= duration(attempt) <= 360 or attempt.stat().st_size > MAX_TRAILER_BYTES:
                continue
            metadata = probe(attempt)
            video = next((stream for stream in metadata.get('streams',[]) if stream.get('codec_type') == 'video'),{})
            width,height = video.get('width',0),video.get('height',0)
            if width < minimum_size[0] or height < minimum_size[1]:
                continue
            novelty = scene_novelty(primary,attempt)
            if not novelty or not novelty['distinct']:
                continue
            save_json(attempt.with_suffix('.json'), {'source':url,'primary_source':source,'title':info.get('title'),
                      'channel':info.get('channel'),'curated':curated,'fan_edit':bool(FAN.search(info.get('title',''))),
                      'source_width':width,'source_height':height,'secondary_version':1,'novelty':novelty})
            attempt.replace(target)
            attempt.with_suffix('.json').replace(target.with_suffix('.json'))
            print(f"Added a second trailer for {film['title']} with different sampled imagery.",flush=True)
            return target
        except (yt_dlp.utils.DownloadError,PipelineError,OSError,ValueError) as exc:
            if _bot_check(exc):
                break
        finally:
            _clean_candidate(attempt)
    if blocked or ranked:
        print(f"No distinct second trailer accepted for {film['title']}; continuing with one.",flush=True)
    return None
