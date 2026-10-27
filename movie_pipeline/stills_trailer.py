"""Fallback footage when no trailer can be downloaded: the film's official TMDB images.

If YouTube refuses every route (and no Vimeo copy exists), a film would otherwise stop
the whole video. Instead its TMDB backdrops become a calm "trailer": each image held for
about six seconds with a slow zoom or pan, hard cuts between images, and silent audio
(the music bed carries the section). The rest of the pipeline treats it like any trailer,
and the run, the description notes and review.html say which film used stills.
"""
from __future__ import annotations

import io
from pathlib import Path

import requests
from PIL import Image

from .core import PipelineError, duration, ffmpeg, key, save_json
from .video_settings import export_size

TMDB_ORIGINAL = "https://image.tmdb.org/t/p/original"
SECONDS_PER_IMAGE = 6.0
MIN_IMAGES = 5
MAX_IMAGES = 14
MOTIONS = (
    "z='min(zoom+0.0008,1.12)':x='iw/2-(iw/zoom/2)':y='ih/2-(ih/zoom/2)'",            # slow push in
    "z='1.12-on*0.0007':x='iw/2-(iw/zoom/2)':y='ih/2-(ih/zoom/2)'",                    # slow pull out
    "z='1.1':x='(iw-iw/zoom)*on/{frames}':y='ih/2-(ih/zoom/2)'",                        # pan left to right
    "z='1.1':x='(iw-iw/zoom)*(1-on/{frames})':y='ih/2-(ih/zoom/2)'",                    # pan right to left
)


def fetch_original(path: str, cache: Path) -> Image.Image | None:
    cache.mkdir(parents=True, exist_ok=True)
    target = cache / f"{key('tmdb-original', path)}.jpg"
    if not target.exists():
        try:
            response = requests.get(TMDB_ORIGINAL + path, timeout=40)
            response.raise_for_status()
            Image.open(io.BytesIO(response.content)).verify()
            target.write_bytes(response.content)
        except (requests.RequestException, OSError, ValueError):
            return None
    try:
        return Image.open(target).convert("RGB")
    except OSError:
        return None


def build(film: dict, path: Path, work: Path) -> Path:
    """Write ``path`` (and its .json record) from the film's TMDB images; raises when there are too few."""
    images = []
    for image_path in film.get("backdrops", [])[:MAX_IMAGES]:
        image = fetch_original(image_path, work / "stills-fallback")
        if image is not None and image.width >= 1280:
            images.append(image)
    if len(images) < MIN_IMAGES:
        raise PipelineError(f"Only {len(images)} usable TMDB images for {film.get('title')}; a stills fallback needs {MIN_IMAGES}.")
    width, height = export_size()
    fps = 30
    frames = int(SECONDS_PER_IMAGE * fps)
    inputs, filters = [], []
    folder = work / "stills-fallback" / f"{film.get('kind', 'movie')}-{film.get('id', 'x')}"
    folder.mkdir(parents=True, exist_ok=True)
    for index, image in enumerate(images):
        frame = folder / f"{index:02}.jpg"
        image.save(frame, quality=95)
        inputs += ["-i", frame]  # one picture in; zoompan makes all of its frames
        motion = MOTIONS[index % len(MOTIONS)].format(frames=frames)
        filters.append(f"[{index}:v]scale={width * 2}:{height * 2}:force_original_aspect_ratio=increase,"
                       f"crop={width * 2}:{height * 2},zoompan={motion}:d={frames}:s={width}x{height}:fps={fps},"
                       f"trim=end_frame={frames},setpts=PTS-STARTPTS,setsar=1,format=yuv420p[s{index}]")
    graph = ";".join(filters) + ";" + "".join(f"[s{i}]" for i in range(len(images))) + f"concat=n={len(images)}:v=1:a=0[v]"
    total = len(images) * SECONDS_PER_IMAGE
    ffmpeg(*inputs, "-f", "lavfi", "-t", total, "-i", "anullsrc=r=48000:cl=stereo", "-filter_complex", graph,
           "-map", "[v]", "-map", f"{len(images)}:a", "-c:v", "libx264", "-preset", "veryfast", "-crf", "18",
           "-c:a", "aac", "-b:a", "128k", "-t", total, path)
    if duration(path) < 25:
        raise PipelineError("Stills fallback came out too short.")
    save_json(path.with_suffix(".json"), {"source": f"TMDB images for {film.get('title')} (stills fallback; no trailer could be downloaded)",
              "title": film.get("title"), "stills_fallback": True, "images": len(images), "requested_height": height,
              "source_width": width, "source_height": height, "quality_selection_version": 2})
    print(f"Stills fallback built for {film.get('title')}: {len(images)} TMDB images, {total:.0f}s.", flush=True)
    return path
