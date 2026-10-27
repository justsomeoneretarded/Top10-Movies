"""Text-free, channel-quality thumbnails that read as the video's genre at a glance.

The list's genre (from the title, brief and the films' TMDB genres) drives everything:
- which frames are considered: the AI scene descriptions of every trailer shot are
  searched for that genre's signature moments (for horror: terrified faces, lurking
  figures, masks, creatures, blood, darkness), not just the busiest shots;
- how frames are scored: the local vision model rates how instantly a frame reads as
  the genre and flags any logo or text; exposure targets suit the genre (horror is dark);
- the look: a grade per genre (horror: crushed blacks, desaturated cold tones, reds kept,
  heavy vignette, grain), and the cut-out glow and panel dividers use the genre's colour.

Designs:
1. Hero: number one's most genre-defining face, cut out over a darkened, blurred scene.
2. Diagonal panels: the most genre-defining frame from three titles.
3. Signature frame: the single most genre-defining frame of the list, full bleed.
A service badge appears only when number one is listed on a recognised service.
Text is off unless THUMBNAIL_TEXT=minimal.
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



GENRE_PROFILES = {
    "horror": {
        "label": "horror",
        "words": ("horror", "scary", "scariest", "scare", "terrifying", "creepy", "haunt", "slasher", "nightmare",
                  "ruin your sleep", "sleep", "fear", "ghost", "demon", "possess", "zombie", "vampire", "killer"),
        "tmdb": ("Horror",),
        "hint": "fear, screaming, dread, a menacing figure, masks, creatures, blood, darkness",
        "queries": ("a terrified screaming face in the dark", "a creepy figure lurking in the shadows",
                    "a monster or creature attacking someone", "a masked killer holding a knife",
                    "a frightened face covered in blood", "glowing eyes in darkness",
                    "a possessed person with a disturbing expression"),
        "keywords": ("scream", "terrif", "fright", "fear", "horror", "creep", "creature", "monster", "mask", "blood",
                     "dark", "shadow", "ghost", "demon", "corpse", "knife", "eerie", "sinister", "menac", "possess",
                     "vampire", "zombie", "haunt", "evil", "disturb", "gore"),
        "avoid": ("smil", "laugh", "sunny", "bright day", "talking calmly", "conversation"),
        "exposure": 60, "grade": "horror", "rim": (190, 18, 18), "divider": (165, 12, 12),
    },
    "thriller": {
        "label": "thriller",
        "words": ("thriller", "mystery", "twist", "crime", "detective", "suspense", "mind-bending", "heist"),
        "tmdb": ("Thriller", "Mystery", "Crime"),
        "hint": "tension, suspicion, danger, a tense face, a weapon, a dark investigation",
        "queries": ("a tense close-up of a worried face", "a person watched from the shadows", "a detective in a dark room",
                    "someone holding a gun in a tense standoff"),
        "keywords": ("tense", "worried", "shadow", "dark", "gun", "detective", "suspic", "danger", "night", "fear"),
        "avoid": ("smil", "laugh", "sunny"),
        "exposure": 80, "grade": "cold", "rim": (120, 170, 255), "divider": (235, 235, 240),
    },
    "action": {
        "label": "action",
        "words": ("action", "explosive", "fight", "adrenaline", "war", "shootout", "martial"),
        "tmdb": ("Action", "War", "Western"),
        "hint": "explosions, fights, weapons, speed, a hero in combat",
        "queries": ("a huge explosion", "a hero fighting in combat", "a high speed car chase", "a soldier firing a gun"),
        "keywords": ("explo", "fight", "gun", "fire", "chase", "battle", "combat", "punch", "crash", "weapon"),
        "avoid": ("talking calmly",),
        "exposure": 105, "grade": "blockbuster", "rim": (255, 170, 60), "divider": (255, 255, 255),
    },
    "scifi": {
        "label": "sci-fi",
        "words": ("sci-fi", "science fiction", "space", "alien", "future", "robot", "time travel", "dystopia"),
        "tmdb": ("Science Fiction",),
        "hint": "spaceships, aliens, futuristic technology, space, glowing lights",
        "queries": ("a spaceship in space", "an astronaut in a spacesuit", "an alien creature", "a futuristic city at night"),
        "keywords": ("space", "alien", "robot", "futur", "astronaut", "planet", "laser", "glow", "ship"),
        "avoid": (),
        "exposure": 90, "grade": "scifi", "rim": (80, 210, 255), "divider": (255, 255, 255),
    },
    "comedy": {
        "label": "comedy",
        "words": ("comedy", "funny", "laugh", "hilarious", "comedies"),
        "tmdb": ("Comedy",),
        "hint": "big laughs, funny shocked reactions, chaos, bright colour",
        "queries": ("a person laughing hard", "a funny shocked reaction", "chaotic funny scene"),
        "keywords": ("laugh", "smil", "funny", "shock", "surpris", "grin", "chaos"),
        "avoid": ("dark", "blood"),
        "exposure": 130, "grade": "bright", "rim": (255, 255, 255), "divider": (255, 255, 255),
    },
    "drama": {
        "label": "drama",
        "words": ("drama", "emotional", "tearjerker", "romance", "love"),
        "tmdb": ("Drama", "Romance", "History"),
        "hint": "intense emotion, tears, heartbreak, an intimate close-up",
        "queries": ("an emotional close-up with tears", "two people in an intense emotional moment"),
        "keywords": ("tear", "cry", "emotion", "sad", "love", "embrace", "intense"),
        "avoid": (),
        "exposure": 100, "grade": "blockbuster", "rim": (255, 214, 160), "divider": (255, 255, 255),
    },
}
DEFAULT_GENRE = "drama"


def detect_genre(films: list[dict], title: str = "", brief: str = "") -> str:
    """The genre a viewer expects from this video: the title/brief wording first, then the films' TMDB genres."""
    text = f" {title} {brief} ".casefold()
    scores = {name: 0.0 for name in GENRE_PROFILES}
    for name, profile in GENRE_PROFILES.items():
        scores[name] += sum(3.0 for word in profile["words"] if word in text)
        for film in films:
            scores[name] += sum(1.0 for genre in film.get("genres", []) if genre in profile["tmdb"])
    best = max(scores, key=lambda name: scores[name])
    return best if scores[best] > 0 else DEFAULT_GENRE

@dataclass
class Candidate:
    image: Image.Image
    film: int
    source: str
    face: tuple[int, int, int, int] | None = None
    score: float = 0.0
    action: float = 0.0
    genre_fit: float = 0.0
    description: str = ""
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


def heuristic(image: Image.Image, exposure_target: float = 105) -> tuple[float, tuple | None]:
    rgb = np.asarray(image.convert("RGB").resize((480, 270)))
    gray = cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY)
    boxes = faces(image)
    face = max(boxes, key=lambda b: b[2] * b[3]) if boxes else None
    face_fraction = (face[2] * face[3]) / (image.width * image.height) if face else 0.0
    # Best around 3-15% of the frame: big enough to read on a phone, small enough to crop around.
    face_score = 0.0 if not face else min(1.0, face_fraction / 0.03) * (1.0 if face_fraction < 0.2 else 0.7)
    hsv = cv2.cvtColor(rgb, cv2.COLOR_RGB2HSV)
    colorfulness = min(float(hsv[..., 1].mean()) / 110, 1.0)
    exposure = 1 - min(1.0, abs(float(gray.mean()) - exposure_target) / 105)
    contrast = min(float(gray.std()) / 60, 1.0)
    color_weight = 0.4 if exposure_target >= 90 else 0.0  # dark genres are not meant to be colourful
    return face_score * 2.2 + _sharpness(gray) * 0.9 + contrast * 0.6 + colorfulness * color_weight + exposure * 0.6, face


LOGO_WORDS = re.compile(r"\b(hulu|netflix|max|hbo|prime|disney|peacock|paramount|apple|amazon|trailer|official|"
                        r"rated|coming soon|in theaters|in theatres|in cinemas|imax|studios|presents)\b", re.IGNORECASE)


def has_burned_text(image: Image.Image) -> bool:
    """Reject frames with captions, title cards, or a channel/streaming logo (even a small corner bug)."""
    from .shots import frame_text
    text = frame_text(image)
    if len(re.findall(r"[A-Za-z]{3,}", text)) >= 3 or LOGO_WORDS.search(text):
        return True
    w, h = image.size
    for box in ((0, 0, w * .26, h * .2), (w * .74, 0, w, h * .2), (0, h * .8, w * .26, h), (w * .74, h * .8, w, h)):
        corner = image.crop(tuple(round(v) for v in box)).resize((round(w * .26 * 2.5), round(h * .2 * 2.5)))
        found = frame_text(corner)
        if re.search(r"[A-Za-z]{4,}", found) or LOGO_WORDS.search(found):
            return True
    return False


def genre_shots(trailer: Path, shots: list[dict], profile: dict, limit: int = 8) -> list[tuple[dict, float, str]]:
    """Shots whose AI description best matches the genre's signature moments, with a 0-1 fit and the description."""
    from . import scenes
    data = scenes.load(trailer, len(shots)) if shots else None
    if not data:
        return []
    descriptions = data["descriptions"]
    fits = None
    try:
        queries = scenes.embed(list(profile["queries"]))
        vectors = np.asarray(data["embeddings"], dtype=np.float32)
        fits = (vectors @ queries.T).max(axis=1)
        fits = (fits - fits.min()) / max(1e-6, float(fits.max() - fits.min()))
    except Exception:  # noqa: BLE001 -- keyword matching below needs no model
        fits = None
    ranked = []
    for index, (shot, text) in enumerate(zip(shots, descriptions)):
        lowered = text.casefold()
        keyword = min(1.0, sum(0.34 for word in profile["keywords"] if word in lowered))
        keyword -= 0.5 * any(word in lowered for word in profile["avoid"])
        fit = 0.6 * float(fits[index]) + 0.4 * keyword if fits is not None else keyword
        ranked.append((shot, max(0.0, fit), text))
    ranked.sort(key=lambda row: -row[1])
    return ranked[:limit]


def gather(films: list[dict], trailers: list[Path], shot_sets: list[list[dict]], work: Path,
           per_film: int = 3, profile: dict | None = None) -> list[Candidate]:
    from .shots import get_frame
    profile = profile or GENRE_PROFILES[DEFAULT_GENRE]
    candidates: list[Candidate] = []
    for index, film in enumerate(films[:max(3, min(len(films), 5))]):
        pool = []
        for path in film.get("backdrops", [])[:4]:
            image = fetch_still(path, work / "stills")
            if image is not None:
                pool.append(Candidate(image, index, "tmdb-still"))
        if index < len(trailers) and index < len(shot_sets):
            chosen = genre_shots(trailers[index], shot_sets[index], profile)
            seen = {id(shot) for shot, _, _ in chosen}
            busiest = [s for s in sorted(shot_sets[index], key=lambda s: -float(s.get("motion", 0))) if id(s) not in seen][:4]
            for shot, fit, text in [*chosen, *((s, 0.0, "") for s in busiest)]:
                try:
                    frame = get_frame(trailers[index], (float(shot["start"]) + float(shot["end"])) / 2)
                except PipelineError:
                    continue
                if frame.width >= 960:
                    pool.append(Candidate(frame, index, "trailer-frame", genre_fit=fit, description=text))
        for candidate in pool:
            candidate.image = trim_bars(candidate.image)
            candidate.score, candidate.face = heuristic(candidate.image, profile["exposure"])
            candidate.score += candidate.genre_fit * 2.5
        pool.sort(key=lambda c: -c.score)
        kept = []
        for candidate in pool:
            if len(kept) >= per_film * 2:
                break
            if not has_burned_text(candidate.image):
                kept.append(candidate)
        if not kept and pool:
            kept = pool[:2]  # better than nothing; the vision check below still drops logo frames
        candidates.extend(kept)
    return candidates


def rate_with_vision(candidates: list[Candidate], limit: int = 18, profile: dict | None = None) -> int:
    """Ask the local vision model about the most promising candidates; returns how many it rated."""
    from . import scenes
    if not scenes.enabled():
        return 0
    profile = profile or GENRE_PROFILES[DEFAULT_GENRE]
    rated = 0
    for candidate in sorted(candidates, key=lambda c: -c.score)[:limit]:
        verdict = scenes.rate_thumbnail(candidate.image, profile["label"], profile["hint"])
        if verdict is None:
            break
        rated += 1
        candidate.notes = verdict
        candidate.action = verdict["action"] / 10
        candidate.genre_fit = max(candidate.genre_fit, verdict.get("genre_fit", 0) / 10)
        candidate.score += (verdict["score"] * 0.2 + verdict["emotion"] * 0.12 + verdict.get("genre_fit", 0) * 0.45
                            + (0.6 if verdict["large_face"] else 0))
        if verdict.get("logo"):
            candidate.score -= 100  # never use a frame with a logo or watermark
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


def genre_grade(image: Image.Image, grade: str, strength: float = 1.0) -> Image.Image:
    """The genre's look. Horror: crushed blacks, cold desaturated tones with reds kept, heavy vignette, grain."""
    if grade == "blockbuster":
        return cinematic_grade(image, strength)
    rgb = np.asarray(image.convert("RGB")).astype(np.uint8)
    lab = cv2.cvtColor(rgb, cv2.COLOR_RGB2LAB)
    lab[..., 0] = cv2.createCLAHE(clipLimit=1.4 + 0.6 * strength, tileGridSize=(6, 6)).apply(lab[..., 0])
    rgb = cv2.cvtColor(lab, cv2.COLOR_LAB2RGB)
    hsv = cv2.cvtColor(rgb, cv2.COLOR_RGB2HSV).astype(np.float32)
    hue, value = hsv[..., 0], hsv[..., 2] / 255
    if grade == "horror":
        red = ((hue < 10) | (hue > 170)).astype(np.float32)
        hsv[..., 1] *= 0.56 + 0.64 * red          # drain colour (cold skin stays), keep blood reds
        hsv[..., 2] = 255 * np.clip(value - 0.07, 0, 1) ** (1.22 + 0.1 * strength) / (0.93 ** (1.22 + 0.1 * strength))
        tint, vignette_amount, grain = np.array([-6, 4, 9]), 0.62, 7.0
    elif grade == "cold":
        hsv[..., 1] *= 0.72
        hsv[..., 2] = 255 * np.clip(value, 0, 1) ** 1.12
        tint, vignette_amount, grain = np.array([-6, 2, 12]), 0.48, 4.0
    elif grade == "scifi":
        hsv[..., 1] *= 1.12
        tint, vignette_amount, grain = np.array([-10, 6, 16]), 0.4, 2.0
    else:  # bright
        hsv[..., 1] *= 1.25
        hsv[..., 2] = 255 * np.clip(value * 1.06 + 0.03, 0, 1)
        tint, vignette_amount, grain = np.array([6, 3, -4]), 0.18, 0.0
    rgb = cv2.cvtColor(np.clip(hsv, 0, 255).astype(np.uint8), cv2.COLOR_HSV2RGB).astype(np.float32)
    shadows = np.clip(1 - rgb.mean(axis=2, keepdims=True) / 255 * 1.8, 0, 1)
    rgb += shadows * tint * strength
    if grain:
        noise = np.random.default_rng(7).normal(0, grain, rgb.shape[:2])[..., None]
        rgb += noise
    out = Image.fromarray(np.clip(rgb, 0, 255).astype(np.uint8))
    out = out.filter(ImageFilter.UnsharpMask(radius=1.8, percent=int(70 + 30 * strength), threshold=3))
    return vignette(out, vignette_amount * min(1.3, strength))


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


def rim_light(subject: Image.Image, color=(255, 214, 160), width: int = 4, strength: float = 0.6) -> Image.Image:
    alpha = subject.getchannel("A")
    edge = Image.fromarray(np.clip(np.asarray(alpha.filter(ImageFilter.MaxFilter(width * 2 + 1)), dtype=np.int16)
                                   - np.asarray(alpha, dtype=np.int16), 0, 255).astype(np.uint8))
    glow = Image.new("RGBA", subject.size, color + (0,))
    glow.putalpha(edge.filter(ImageFilter.GaussianBlur(width)).point(lambda v: min(255, int(v * strength))))
    return glow


def depth_composite(subject: Image.Image, background: Image.Image, *, height_fraction: float = 1.05,
                    x_fraction: float = 0.62, blur: float = 14, darken: float = 0.55,
                    rim=(255, 214, 160)) -> Image.Image:
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
    canvas.alpha_composite(rim_light(subject, rim), (x, y))
    canvas.alpha_composite(subject, (x, y))
    return canvas.convert("RGB")


def panels(candidates: list[Candidate], divider=(255, 255, 255)) -> Image.Image:
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
        draw.line([(x + slant, -5), (x - slant, HEIGHT + 5)], fill=divider, width=10)
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


def finish(image: Image.Image, grade: str = "blockbuster") -> Image.Image:
    graded = genre_grade(image, grade)
    # Dark genres are meant to be dark, but still need punch at phone size.
    if phone_check(graded) < (36 if grade == "horror" else 42):
        graded = genre_grade(image, grade, strength=1.6)
    return graded


# ---------------------------------------------------------------- designs
def build(films: list[dict], trailers: list[Path], shot_sets: list[list[dict]], phrases: list[str],
          output: Path, work: Path, title: str | None = None, brief: str | None = None) -> list[dict]:
    work.mkdir(parents=True, exist_ok=True)
    title = os.getenv("VIDEO_TITLE", "") if title is None else title
    brief = os.getenv("VIDEO_BRIEF", "") if brief is None else brief
    genre = detect_genre(films, title, brief)
    profile = GENRE_PROFILES[genre]
    candidates = gather(films, trailers, shot_sets, work, profile=profile)
    if not any(c.film == 0 for c in candidates):
        raise PipelineError("No usable thumbnail image for number one.")
    rated = rate_with_vision(candidates, profile=profile)
    candidates = [c for c in candidates if not c.notes.get("logo")] or candidates
    by_film: dict[int, list[Candidate]] = {}
    for candidate in sorted(candidates, key=lambda c: -c.score):
        by_film.setdefault(candidate.film, []).append(candidate)
    top = by_film.get(0) or sorted(candidates, key=lambda c: -c.score)
    cutter = Cutter()
    report = []
    text_mode = os.getenv("THUMBNAIL_TEXT", "none").strip().lower()
    service = service_on_number_one(films)
    grade = profile["grade"]

    # 1. Hero: number one's most genre-defining face, cut out over a darkened scene.
    hero = next((c for c in top if c.face), top[0])
    subject = cutter.cut(hero.image) if hero.face else None
    if subject is not None:
        # For dark genres the backdrop is the list's most genre-defining frame (the threat behind the face).
        backdrop = hero
        if grade in ("horror", "cold"):
            backdrop = max((c for c in candidates if c is not hero), key=lambda c: c.genre_fit, default=hero)
        image = depth_composite(subject, backdrop.image, height_fraction=1.12, x_fraction=0.6,
                                blur=6 if backdrop is not hero else 14, darken=0.42 if grade == "horror" else 0.55,
                                rim=profile["rim"])
        style = "hero-depth"
    else:
        image = face_crop(hero, face_height=0.42)
        style = "hero-crop"
    canvas = finish(image, grade)
    save(canvas, output / "thumbnail-01.jpg")
    report.append({"file": "thumbnail-01.jpg", "style": style, "source": hero.source, "film": films[hero.film]["title"],
                   "genre_fit": round(hero.genre_fit, 2), "description": hero.description, "vision": hero.notes,
                   "cutout_model": cutter.model})

    # 2. Diagonal panels: the most genre-defining frame from up to three titles.
    picks = []
    for index in sorted(by_film)[:3]:
        options = sorted(by_film[index], key=lambda c: -(c.genre_fit * 3 + c.score * 0.3))
        picks.append(next((c for c in options if c is not hero), options[0]))
    if len(picks) < 2 and len(top) > 1:
        picks.append(top[1])
    canvas = finish(panels(picks, profile["divider"]) if len(picks) >= 2 else face_crop(top[0]), grade)
    if text_mode == "minimal" and phrases:
        canvas = minimal_label(canvas, phrases[0])
    save(canvas, output / "thumbnail-02.jpg")
    report.append({"file": "thumbnail-02.jpg", "style": "diagonal-panels", "films": [films[p.film]["title"] for p in picks],
                   "descriptions": [p.description for p in picks]})

    # 3. Signature frame: the list's single most genre-defining frame, full bleed (never the hero image).
    used = {id(hero), *(id(p) for p in picks)}
    others = [c for c in candidates if id(c) not in used] or [c for c in candidates if c is not hero] or [hero]
    signature = max(others, key=lambda c: c.genre_fit * 3 + c.score * 0.3 + c.action)
    image = face_crop(signature, face_height=0.36, face_x=0.5)
    canvas = finish(image, grade)
    if service:
        canvas = service_badge(canvas, service)
    save(canvas, output / "thumbnail-03.jpg")
    report.append({"file": "thumbnail-03.jpg", "style": "signature-frame", "film": films[signature.film]["title"],
                   "genre_fit": round(signature.genre_fit, 2), "description": signature.description, "service_badge": service})
    save_json(work / "thumbnail-report.json", {"genre": genre, "vision_rated": rated, "candidates": len(candidates),
                                               "thumbnails": report})
    print(f"Studio thumbnails ({genre}): {', '.join(item['style'] for item in report)}; "
          f"{rated} candidates rated by the vision model.", flush=True)
    return report
