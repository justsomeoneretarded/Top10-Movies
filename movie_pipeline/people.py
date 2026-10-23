"""Optional, metadata-identified portraits when narration explicitly names cast."""
from __future__ import annotations

import re
from pathlib import Path

import requests
from PIL import Image, ImageFilter, ImageOps


def portrait_for(text: str, film: dict, cache: Path) -> dict | None:
    """Never infer identity from a generic face detector or a CLIP face match."""
    for person in film.get("cast_profiles", []):
        name, profile = person.get("name", ""), person.get("profile_path", "")
        if not name or not re.search(r"(?<!\w)"+re.escape(name)+r"(?!\w)", text, re.IGNORECASE):
            continue
        if not re.fullmatch(r"/[A-Za-z0-9_-]+\.(?:jpg|png)", profile):
            continue
        cache.mkdir(parents=True, exist_ok=True)
        target = cache / profile[1:]
        try:
            if not target.exists():
                with requests.get("https://image.tmdb.org/t/p/w500"+profile, timeout=(3, 8), stream=True) as response:
                    response.raise_for_status()
                    content = bytearray()
                    for block in response.iter_content(65536):
                        content.extend(block)
                        if len(content) > 3_000_000:
                            raise ValueError("Oversized portrait")
                target.write_bytes(content)
            with Image.open(target) as image:
                image.verify()
            widescreen = target.with_name(target.stem+"-wide.jpg")
            if not widescreen.exists():
                with Image.open(target) as image:
                    image = image.convert("RGB")
                    canvas = ImageOps.fit(image, (1920, 1080)).filter(ImageFilter.GaussianBlur(35))
                    foreground = ImageOps.contain(image, (1920, 1080))
                    canvas.paste(foreground, ((1920-foreground.width)//2, (1080-foreground.height)//2))
                    canvas.save(widescreen, quality=95)
            return {"source": str(widescreen), "start": 0, "image": True, "still": True,
                    "still_motion": "zoom_in", "person": name,
                    "visual_reason": "metadata-identified-cast-portrait"}
        except (OSError, ValueError, requests.RequestException):
            target.unlink(missing_ok=True)
    return None


def insert_portrait(plan: list[dict], portrait: dict, seconds: float = 5.5, *, when: float = 0) -> list[dict]:
    """Replace time at the named person's mention, never add or repeat footage."""
    total = sum(shot["duration"] for shot in plan)
    beginning = max(0, min(when, max(0, total-4)))
    end = min(total, beginning+seconds)
    def window(low, high):
        result, clock = [], 0.0
        for shot in plan:
            start = max(low, clock)
            stop = min(high, clock+shot["duration"])
            if stop > start:
                revised = {**shot, "duration": stop-start}
                if not shot.get("still"):
                    revised["start"] += start-clock
                revised["freeze"] = min(revised.get("freeze", 0), revised["duration"])
                result.append(revised)
            clock += shot["duration"]
        return result
    return [*window(0, beginning), {**portrait, "duration": end-beginning}, *window(end, total)]
