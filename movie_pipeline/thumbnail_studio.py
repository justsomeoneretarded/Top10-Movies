"""Text-free, channel-quality thumbnails: huge faces, depth, panels and a cinematic grade.

Modelled on top movie-recommendation channels:
1. Hero face: number one's most intense face, cut out and enlarged over a blurred,
   darkened copy of the same scene, with a subtle rim light.
2. Diagonal panels: the strongest face from three titles in slanted panels with thin
   white dividers, which reads as "a list" without words.
3. Action composite: a cut-out character over a dramatic backdrop from another title.

Candidates come from TMDB stills and the best trailer frames; local checks score
faces, sharpness, exposure and colour and reject frames with burned-in text, and a
small local vision model (when available) rates emotion and action. One grading
pipeline gives every option the same punchy look. A service badge appears only when
number one is actually listed on that service. Text is off unless THUMBNAIL_TEXT=minimal.
"""
from __future__ import annotations

import math
import os
import re
from dataclasses import dataclass, field
from pathlib import Path

import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageEnhance, ImageFilter

from .core import PipelineError, save_json
from .render import ACCENT, HEIGHT, WIDTH, brand_font, wrapped
from .still_thumbnails import Cutter, faces, fetch_still, netflix_badge, save

SERVICE_COLORS = {"prime video": (0, 168, 225), "max": (0, 40, 255), "disney plus": (17, 60, 207),
                  "hulu": (28, 231, 131), "paramount plus": (0, 100, 255), "apple tv plus": (20, 20, 20),
                  "peacock": (20, 20, 20)}


@dataclass
class Candidate:
    image: Image.Image
    film: int
    source: str
    face: tuple[int, int, int, int] | None = None
    score: float = 0.0
    action: float = 0.0
    notes: dict = field(default_factory=dict)


# ---------------------------------------------------------------- candidates
def trim_bars(image: Image.Image, threshold: int = 18) -> Image.Image:
    """Remove black letterbox/pillarbox bars baked into trailer frames and stills."""
    gray = np.asarray(image.convert("L"), dtype=np.float32)
    rows = np.where(gray.mean(axis=1) > threshold)[0]
    cols = np.where(gray.mean(axis=0) > threshold)[0]
    if not len(rows) or not len(cols):
        return image
    top, bottom, left, right = rows[0], rows[-1] + 1, cols[0], cols[-1] + 1
    if (bottom - top) < image.height * 0.5 or (right - left) < image.width * 0.5:
        return image
    if top == 0 and left == 0 and bottom == image.height and right == image.width:
        return image
    return image.crop((left + 2, top + 2, right - 2, bottom - 2))


def _sharpness(gray: np.ndarray) -> float:
    return min(float(cv2.Laplacian(gray, cv2.CV_64F).var()), 900) / 900


def heuristic(image: Image.Image) -> tuple[float, tuple | None]:
    rgb = np.asarray(image.convert("RGB").resize((480, 270)))
    gray = cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY)
    boxes = faces(image)
    face = max(boxes, key=lambda b: b[2] * b[3]) if boxes else None
    face_fraction = (face[2] * face[3]) / (image.width * image.height) if face else 0.0
    # Best around 3-15% of the frame: big enough to read on a phone, small enough to crop around.
    face_score = 0.0 if not face else min(1.0, face_fraction / 0.03) * (1.0 if face_fraction < 0.2 else 0.7)
    hsv = cv2.cvtColor(rgb, cv2.COLOR_RGB2HSV)
    colorfulness = min(float(hsv[..., 1].mean()) / 110, 1.0)
    exposure = 1 - abs(float(gray.mean()) - 105) / 105
    contrast = min(float(gray.std()) / 60, 1.0)
    return face_score * 2.2 + _sharpness(gray) * 0.9 + contrast * 0.6 + colorfulness * 0.4 + exposure * 0.4, face


def has_burned_text(image: Image.Image) -> bool:
    from .shots import frame_text
    words = [w for w in re.findall(r"[A-Za-z]{3,}", frame_text(image))]
    return len(words) >= 3


def gather(films: list[dict], trailers: list[Path], shot_sets: list[list[dict]], work: Path,
           per_film: int = 3) -> list[Candidate]:
    from .shots import get_frame
    candidates: list[Candidate] = []
    for index, film in enumerate(films[:max(3, min(len(films), 5))]):
        pool = []
        for path in film.get("backdrops", [])[:5]:
            image = fetch_still(path, work / "stills")
            if image is not None:
                pool.append(Candidate(image, index, "tmdb-still"))
        if index < len(trailers) and index < len(shot_sets):
            shots = sorted(shot_sets[index], key=lambda s: -float(s.get("motion", 0)))[:10]
            for shot in shots:
                try:
                    frame = get_frame(trailers[index], (float(shot["start"]) + float(shot["end"])) / 2)
                except PipelineError:
                    continue
                if frame.width >= 960:
                    pool.append(Candidate(frame, index, "trailer-frame"))
        for candidate in pool:
            candidate.image = trim_bars(candidate.image)
            candidate.score, candidate.face = heuristic(candidate.image)
        pool.sort(key=lambda c: -c.score)
        kept = []
        for candidate in pool:
            if len(kept) >= per_film * 2:
                break
            if not has_burned_text(candidate.image):
                kept.append(candidate)
        candidates.extend(kept)
    return candidates


def rate_with_vision(candidates: list[Candidate], limit: int = 14) -> int:
    """Ask the local vision model about the most promising candidates; returns how many it rated."""
    from . import scenes
    if not scenes.enabled():
        return 0
    rated = 0
    for candidate in sorted(candidates, key=lambda c: -c.score)[:limit]:
        verdict = scenes.rate_thumbnail(candidate.image)
        if verdict is None:
            break
        rated += 1
        candidate.notes = verdict
        candidate.action = verdict["action"] / 10
        candidate.score += verdict["score"] * 0.25 + verdict["emotion"] * 0.12 + (0.6 if verdict["large_face"] else 0)
    return rated


# ---------------------------------------------------------------- grading
def cinematic_grade(image: Image.Image, strength: float = 1.0) -> Image.Image:
    """Local contrast, teal/orange split-toning, saturation, clarity, highlight bloom and vignette."""
    rgb = np.asarray(image.convert("RGB")).astype(np.float32)
    lab = cv2.cvtColor(rgb.astype(np.uint8), cv2.COLOR_RGB2LAB)
    clahe = cv2.createCLAHE(clipLimit=1.1 + 0.5 * strength, tileGridSize=(6, 6))
    lab[..., 0] = clahe.apply(lab[..., 0])
    rgb = cv2.cvtColor(lab, cv2.COLOR_LAB2RGB).astype(np.float32)
    luminance = rgb.mean(axis=2, keepdims=True) / 255
    shadows = np.clip(1 - luminance * 1.8, 0, 1)
    highlights = np.clip(luminance * 1.6 - 0.6, 0, 1)
    rgb += shadows * np.array([-7, 3, 10], dtype=np.float32) * strength
    rgb += highlights * np.array([12, 5, -9], dtype=np.float32) * strength
    hsv = cv2.cvtColor(np.clip(rgb, 0, 255).astype(np.uint8), cv2.COLOR_RGB2HSV).astype(np.float32)
    hsv[..., 1] = np.clip(hsv[..., 1] * (1 + 0.15 * strength), 0, 255)
    rgb = cv2.cvtColor(hsv.astype(np.uint8), cv2.COLOR_HSV2RGB).astype(np.float32)
    bright = np.clip((rgb - 200) * 3, 0, 255)
    bloom = cv2.GaussianBlur(bright, (0, 0), 18)
    rgb = 255 - (255 - rgb) * (255 - bloom * 0.12 * strength) / 255
    out = Image.fromarray(np.clip(rgb, 0, 255).astype(np.uint8))
    out = out.filter(ImageFilter.UnsharpMask(radius=1.6, percent=int(55 + 25 * strength), threshold=3))
    return vignette(out, 0.32 * strength)


def vignette(image: Image.Image, amount: float = 0.32) -> Image.Image:
    w, h = image.size
    y, x = np.ogrid[:h, :w]
    distance = np.sqrt(((x - w / 2) / (w / 2)) ** 2 + ((y - h / 2) / (h / 2)) ** 2) / math.sqrt(2)
    mask = 1 - amount * np.clip(distance - 0.25, 0, 1) ** 1.4 / 0.75 ** 1.4
    rgb = np.asarray(image.convert("RGB")).astype(np.float32) * mask[..., None]
    return Image.fromarray(np.clip(rgb, 0, 255).astype(np.uint8))


# ---------------------------------------------------------------- framing
def face_crop(candidate: Candidate, width: int = WIDTH, height: int = HEIGHT, *, face_height: float = 0.40,
              face_x: float = 0.62, face_y: float = 0.40) -> Image.Image:
    """Crop so the face fills about ``face_height`` of the frame at the chosen position."""
    image = candidate.image.convert("RGB")
    if candidate.face is None:
        scale = max(width / image.width, height / image.height)
        resized = image.resize((round(image.width * scale), round(image.height * scale)), Image.LANCZOS)
        left, top = (resized.width - width) // 2, (resized.height - height) // 2
        return resized.crop((left, top, left + width, top + height))
    x, y, w, h = candidate.face
    scale = max(width / image.width, height / image.height, min(2.4, face_height * height / max(h, 1)))
    resized = image.resize((round(image.width * scale), round(image.height * scale)), Image.LANCZOS)
    cx, cy = (x + w / 2) * scale, (y + h / 2) * scale
    left = int(min(max(0, cx - width * face_x), resized.width - width))
    top = int(min(max(0, cy - height * face_y), resized.height - height))
    return resized.crop((left, top, left + width, top + height))


def rim_light(subject: Image.Image, color=(255, 214, 160), width: int = 7) -> Image.Image:
    alpha = subject.getchannel("A")
    edge = Image.fromarray(np.clip(np.asarray(alpha.filter(ImageFilter.MaxFilter(width * 2 + 1)), dtype=np.int16)
                                   - np.asarray(alpha, dtype=np.int16), 0, 255).astype(np.uint8))
    glow = Image.new("RGBA", subject.size, color + (0,))
    glow.putalpha(edge.filter(ImageFilter.GaussianBlur(width)).point(lambda v: min(255, int(v * 0.9))))
    return glow


def depth_composite(subject: Image.Image, background: Image.Image, *, height_fraction: float = 1.05,
                    x_fraction: float = 0.62, blur: float = 14, darken: float = 0.55) -> Image.Image:
    """Cut-out subject over a blurred, darkened background, with a soft shadow and rim light."""
    canvas = background.convert("RGB").resize((WIDTH, HEIGHT), Image.LANCZOS).filter(ImageFilter.GaussianBlur(blur))
    canvas = ImageEnhance.Brightness(canvas).enhance(darken).convert("RGBA")
    scale = HEIGHT * height_fraction / subject.height
    subject = subject.resize((max(1, round(subject.width * scale)), max(1, round(subject.height * scale))), Image.LANCZOS)
    x = round(WIDTH * x_fraction - subject.width / 2)
    x = max(-subject.width // 4, min(WIDTH - subject.width * 3 // 4, x))
    y = HEIGHT - subject.height + round(HEIGHT * 0.06)
    shadow = Image.new("RGBA", subject.size, (0, 0, 0, 0))
    shadow.putalpha(subject.getchannel("A").point(lambda v: int(v * 0.55)).filter(ImageFilter.GaussianBlur(18)))
    canvas.alpha_composite(shadow, (x + 14, y + 10))
    canvas.alpha_composite(rim_light(subject), (x, y))
    canvas.alpha_composite(subject, (x, y))
    return canvas.convert("RGB")


def panels(candidates: list[Candidate]) -> Image.Image:
    """Two to four slanted panels separated by thin white diagonal lines."""
    count = len(candidates)
    canvas = Image.new("RGB", (WIDTH, HEIGHT), (10, 10, 12))
    slant = 90
    edges = [round(WIDTH * i / count) for i in range(count + 1)]
    for index, candidate in enumerate(candidates):
        left, right = edges[index], edges[index + 1]
        panel_width = right - left + slant * 2
        piece = face_crop(candidate, panel_width, HEIGHT, face_height=0.38, face_x=0.5, face_y=0.38)
        mask = Image.new("L", (WIDTH, HEIGHT), 0)
        top_left = left + (slant if index else -slant)
        top_right = right + (slant if index < count - 1 else slant * 2)
        bottom_left = left - (slant if index else slant * 2)
        bottom_right = right - (slant if index < count - 1 else -slant)
        ImageDraw.Draw(mask).polygon([(top_left, 0), (top_right, 0), (bottom_right, HEIGHT), (bottom_left, HEIGHT)], fill=255)
        layer = Image.new("RGB", (WIDTH, HEIGHT))
        layer.paste(piece, (left - slant, 0))
        canvas.paste(layer, (0, 0), mask)
    draw = ImageDraw.Draw(canvas)
    for index in range(1, count):
        x = edges[index]
        draw.line([(x + slant, -5), (x - slant, HEIGHT + 5)], fill=(255, 255, 255), width=10)
    return canvas


# ---------------------------------------------------------------- badges and text
def service_on_number_one(films: list[dict]) -> str | None:
    from .availability import services
    record = films[0].get("watch_availability", {}) if films else {}
    found = services(record, ("subscription", "free", "ads"), limit=1)
    return found[0] if found else None


def service_badge(canvas: Image.Image, service: str) -> Image.Image:
    canvas = canvas.convert("RGBA")
    if service.casefold() == "netflix":
        netflix_badge(canvas)
        return canvas
    draw = ImageDraw.Draw(canvas)
    face = brand_font("display", 40)
    text = service.upper()
    width = draw.textlength(text, font=face)
    color = SERVICE_COLORS.get(service.casefold(), (230, 33, 23))
    draw.rounded_rectangle((34, 30, 34 + width + 40, 92), radius=12, fill=color + (240,))
    draw.text((54, 61), text, font=face, anchor="lm", fill=(255, 255, 255))
    return canvas


def minimal_label(canvas: Image.Image, phrase: str) -> Image.Image:
    """Optional 2-3 word label along the bottom edge (THUMBNAIL_TEXT=minimal)."""
    canvas = canvas.convert("RGBA")
    shade = Image.new("RGBA", canvas.size, (0, 0, 0, 0))
    ImageDraw.Draw(shade).rectangle((0, HEIGHT - 170, WIDTH, HEIGHT), fill=(0, 0, 0, 150))
    canvas = Image.alpha_composite(canvas, shade.filter(ImageFilter.GaussianBlur(40)))
    draw = ImageDraw.Draw(canvas)
    words = " ".join(phrase.upper().split()[:3])
    size = 110
    while size > 50 and draw.textlength(words, font=brand_font("display", size)) > WIDTH - 120:
        size -= 4
    lines = wrapped(draw, words, brand_font("display", size), WIDTH - 120)[:1]
    draw.text((WIDTH // 2, HEIGHT - 70), lines[0] if lines else words, font=brand_font("display", size), anchor="mm",
              fill=(255, 255, 255), stroke_width=7, stroke_fill=(0, 0, 0))
    return canvas


def phone_check(image: Image.Image) -> float:
    """Contrast at phone-feed size (168x94); low values look muddy in the feed."""
    gray = np.asarray(image.convert("L").resize((168, 94), Image.LANCZOS), dtype=np.float32)
    return float(gray.std())


def finish(image: Image.Image) -> Image.Image:
    graded = cinematic_grade(image)
    if phone_check(graded) < 42:
        graded = cinematic_grade(image, strength=1.6)
    return graded


# ---------------------------------------------------------------- designs
def build(films: list[dict], trailers: list[Path], shot_sets: list[list[dict]], phrases: list[str],
          output: Path, work: Path) -> list[dict]:
    work.mkdir(parents=True, exist_ok=True)
    candidates = gather(films, trailers, shot_sets, work)
    if not any(c.film == 0 for c in candidates):
        raise PipelineError("No usable thumbnail image for number one.")
    rated = rate_with_vision(candidates)
    by_film: dict[int, list[Candidate]] = {}
    for candidate in sorted(candidates, key=lambda c: -c.score):
        by_film.setdefault(candidate.film, []).append(candidate)
    top = by_film[0]
    cutter = Cutter()
    report = []
    text_mode = os.getenv("THUMBNAIL_TEXT", "none").strip().lower()
    service = service_on_number_one(films)

    # 1. Hero face of number one, with depth when the cutout succeeds.
    hero = next((c for c in top if c.face), top[0])
    subject = cutter.cut(hero.image) if hero.face else None
    if subject is not None:
        image = depth_composite(subject, hero.image, height_fraction=1.12, x_fraction=0.6)
        style = "hero-depth"
    else:
        image = face_crop(hero, face_height=0.42)
        style = "hero-crop"
    canvas = finish(image)
    save(canvas, output / "thumbnail-01.jpg")
    report.append({"file": "thumbnail-01.jpg", "style": style, "source": hero.source, "film": films[0]["title"],
                   "vision": hero.notes, "cutout_model": cutter.model})

    # 2. Diagonal panels with the strongest face from up to three titles.
    picks = []
    for index in sorted(by_film)[:3]:
        options = by_film[index]
        picks.append(next((c for c in options if c.face and c is not hero), options[0]))
    if len(picks) < 2 and len(top) > 1:
        picks.append(top[1])
    canvas = finish(panels(picks) if len(picks) >= 2 else face_crop(top[0]))
    if text_mode == "minimal" and phrases:
        canvas = minimal_label(canvas, phrases[0])
    save(canvas, output / "thumbnail-02.jpg")
    report.append({"file": "thumbnail-02.jpg", "style": "diagonal-panels", "films": [films[p.film]["title"] for p in picks]})

    # 3. Action composite: a character from one title over another title's most spectacular backdrop.
    others = [c for c in candidates if c is not hero]
    action_pool = sorted((c for c in others if c.face), key=lambda c: -(c.score + c.action * 2)) or \
        sorted(others, key=lambda c: -(c.score + c.action * 2))
    character = action_pool[0] if action_pool else hero
    backdrops = sorted((c for c in candidates if c.film != character.film and c is not hero),
                       key=lambda c: -(c.action + c.score * 0.3))
    backdrop = backdrops[0] if backdrops else character
    subject = cutter.cut(character.image) if character.face else None
    if subject is not None:
        image = depth_composite(subject, backdrop.image, height_fraction=1.0, x_fraction=0.36, blur=4, darken=0.7)
        style = "action-composite"
    else:
        image = face_crop(character, face_height=0.36, face_x=0.4)
        style = "action-crop"
    canvas = finish(image)
    if service:
        canvas = service_badge(canvas, service)
    save(canvas, output / "thumbnail-03.jpg")
    report.append({"file": "thumbnail-03.jpg", "style": style, "character_film": films[character.film]["title"],
                   "backdrop_film": films[backdrop.film]["title"], "service_badge": service})
    save_json(work / "thumbnail-report.json", {"vision_rated": rated, "candidates": len(candidates), "thumbnails": report})
    print(f"Studio thumbnails: {', '.join(item['style'] for item in report)}; {rated} candidates rated by the vision model.", flush=True)
    return report
