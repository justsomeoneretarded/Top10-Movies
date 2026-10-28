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
4. Service mark: number one with a large streaming-service mark (its real service when
   listed; a random big service only when no featured film has a recognised listing).
5. Headline: a two-to-four-word phrase from the title in huge letters beside a face.
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
        "words": ("thriller", "mystery", "twist", "crime", "detective", "suspense", "mind-bending", "heist", "whodunit", "serial killer", "psychological"),
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
        "words": ("action", "explosive", "fight", "adrenaline", "war", "shootout", "martial", "superhero", "western", "spy", "car chase"),
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
        "words": ("drama", "emotional", "tearjerker", "true events", "biopic", "oscar"),
        "tmdb": ("Drama", "History", "Music"),
        "hint": "intense emotion, tears, heartbreak, an intimate close-up",
        "queries": ("an emotional close-up with tears", "two people in an intense emotional moment"),
        "keywords": ("tear", "cry", "emotion", "sad", "love", "embrace", "intense"),
        "avoid": (),
        "exposure": 100, "grade": "blockbuster", "rim": (255, 214, 160), "divider": (255, 255, 255),
    },
    "adventure": {
        "label": "adventure",
        "words": ("adventure", "quest", "treasure", "expedition", "explore", "epic", "jungle", "voyage"),
        "tmdb": ("Adventure",),
        "hint": "epic landscapes, heroes on a quest, danger in the wild, awe and discovery",
        "queries": ("a hero standing before an epic landscape", "explorers in a dangerous jungle or desert",
                    "a daring escape across a cliff or bridge", "a person in awe of something huge"),
        "keywords": ("landscape", "mountain", "jungle", "desert", "ocean", "cliff", "explor", "ruins", "ship", "awe", "epic"),
        "avoid": ("office", "talking calmly"),
        "exposure": 110, "grade": "warm", "rim": (255, 190, 90), "divider": (255, 255, 255),
    },
    "family": {
        "label": "family",
        "words": ("family", "kids", "animated", "animation", "pixar", "disney", "cartoon", "children", "feel-good"),
        "tmdb": ("Family", "Animation"),
        "hint": "big joyful expressions, colourful characters, wonder, friendship, fun",
        "queries": ("a joyful character with a big smile", "colourful animated characters having fun",
                    "a child amazed by something magical", "friends laughing together"),
        "keywords": ("smil", "laugh", "colorful", "colourful", "cartoon", "animated", "joy", "wonder", "friend", "magic", "cute"),
        "avoid": ("blood", "gun", "dark"),
        "exposure": 135, "grade": "bright", "rim": (255, 255, 255), "divider": (255, 255, 255),
    },
    "romance": {
        "label": "romance",
        "words": ("romance", "romantic", "romantic comed", "love", "rom-com", "romcom", "date night", "valentine", "heartbreak"),
        "tmdb": ("Romance",),
        "hint": "two people close together, longing looks, a kiss, warm soft light, heartbreak",
        "queries": ("a couple about to kiss in warm light", "two people gazing at each other", "a tearful heartbroken face",
                    "a couple laughing together"),
        "keywords": ("couple", "kiss", "embrace", "gaz", "love", "romantic", "smil", "hold hands", "tear", "warm"),
        "avoid": ("gun", "blood", "explo"),
        "exposure": 120, "grade": "warm", "rim": (255, 150, 170), "divider": (255, 255, 255),
    },
    "fantasy": {
        "label": "fantasy",
        "words": ("fantasy", "magic", "dragon", "wizard", "kingdom", "mythical", "sword", "sorcer"),
        "tmdb": ("Fantasy",),
        "hint": "magic, dragons, glowing spells, castles, mythical creatures, epic battles",
        "queries": ("a wizard casting a glowing spell", "a dragon in the sky", "a castle in a magical landscape",
                    "a warrior with a sword facing a creature"),
        "keywords": ("magic", "glow", "dragon", "castle", "sword", "spell", "creature", "wizard", "kingdom", "mystic"),
        "avoid": ("office", "car"),
        "exposure": 95, "grade": "magic", "rim": (170, 120, 255), "divider": (255, 255, 255),
    },
    "documentary": {
        "label": "documentary",
        "words": ("documentary", "documentaries", "true story", "real story", "docuseries"),
        "tmdb": ("Documentary",),
        "hint": "real people, striking real places, an intense moment caught on camera",
        "queries": ("a real person speaking with emotion", "a striking real-world location", "archival footage of a dramatic event"),
        "keywords": ("person", "interview", "crowd", "real", "city", "nature", "archival", "emotion"),
        "avoid": (),
        "exposure": 105, "grade": "blockbuster", "rim": (255, 214, 160), "divider": (255, 255, 255),
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


_DETECTORS = {}


def _detector(name: str):
    if name not in _DETECTORS:
        path = Path(cv2.data.haarcascades) / name
        _DETECTORS[name] = cv2.CascadeClassifier(str(path)) if path.exists() else None
    return _DETECTORS[name]


def frame_sharpness(image: Image.Image) -> float:
    """Detail at a fixed 960-pixel width; below about 25 the frame is soft, blurred or bokeh."""
    gray = np.asarray(image.convert("L").resize((960, max(1, round(960 * image.height / image.width)))), dtype=np.uint8)
    return float(cv2.Laplacian(gray, cv2.CV_64F).var())


def face_quality(image: Image.Image) -> tuple[tuple[int, int, int, int] | None, float, bool]:
    """(face box, quality score, good) for the best front-facing face.

    Good means: big enough to read on a phone (about a tenth of the frame height or more),
    sharp, eyes visible, sensibly exposed and not cut off at the top.
    """
    detector = _detector("haarcascade_frontalface_default.xml")
    if detector is None:
        return None, 0.0, False
    scale = 640 / image.width
    small = np.asarray(image.convert("L").resize((640, max(1, round(image.height * scale)))), dtype=np.uint8)
    boxes = detector.detectMultiScale(small, 1.1, 6, minSize=(36, 36))
    if not len(boxes):
        return None, 0.0, False
    eyes = _detector("haarcascade_eye.xml")
    best, best_score, best_good = None, 0.0, False
    for x, y, w, h in boxes:
        face = small[y:y + h, x:x + w]
        size = h / small.shape[0]
        crisp = cv2.Laplacian(cv2.resize(face, (160, 160)), cv2.CV_64F).var()
        found_eyes = len(eyes.detectMultiScale(face[: h // 2 + h // 8], 1.1, 5, minSize=(max(8, w // 8), max(8, w // 8)))) if eyes is not None else 1
        exposure = 1 - min(1.0, abs(float(face.mean()) - 125) / 110)
        s_size, s_sharp = min(1.0, size / 0.22), min(1.0, crisp / 260)
        score = 2 * s_size + 2 * s_sharp + (1.0 if found_eyes else 0.3) + 0.5 * exposure
        good = size >= 0.1 and s_sharp >= 0.35 and found_eyes > 0 and y > 2 and exposure > 0.25
        if score > best_score:
            best = tuple(int(v / scale) for v in (x, y, w, h))
            best_score, best_good = score, good
    return best, best_score, best_good


def _dhash(image: Image.Image) -> int:
    gray = np.asarray(image.convert("L").resize((9, 8)), dtype=np.int16)
    return sum(int(bit) << i for i, bit in enumerate((gray[:, 1:] > gray[:, :-1]).reshape(-1)))


def gather(films: list[dict], trailers: list[Path], shot_sets: list[list[dict]], work: Path,
           per_film: int = 8, profile: dict | None = None) -> list[Candidate]:
    """Sharp, varied candidates per title: full-resolution TMDB stills first, then the best trailer frames."""
    from .shots import get_frame
    from .stills_trailer import fetch_original
    profile = profile or GENRE_PROFILES[DEFAULT_GENRE]
    candidates: list[Candidate] = []
    for index, film in enumerate(films[:max(3, min(len(films), 5))]):
        pool = []
        for path in film.get("backdrops", [])[:10]:
            image = fetch_original(path, work / "stills") or fetch_still(path, work / "stills")
            if image is not None:
                pool.append(Candidate(image, index, "tmdb-still"))
        if index < len(trailers) and index < len(shot_sets):
            chosen = genre_shots(trailers[index], shot_sets[index], profile)
            seen = {id(shot) for shot, _, _ in chosen}
            busiest = [s for s in sorted(shot_sets[index], key=lambda s: -float(s.get("peak", s.get("motion", 0)))) if id(s) not in seen][:4]
            for shot, fit, text in [*chosen, *((s, 0.0, "") for s in busiest)]:
                if shot.get("text_card"):
                    continue
                middle = (float(shot["start"]) + float(shot["end"])) / 2
                frames = []
                for offset in (0.0, 0.35):
                    try:
                        frames.append(get_frame(trailers[index], middle + offset))
                    except PipelineError:
                        continue
                if frames:
                    frame = max(frames, key=frame_sharpness)  # the steadier of two nearby frames
                    if frame.width >= 1280:
                        pool.append(Candidate(frame, index, "trailer-frame", genre_fit=fit, description=text,
                                              action=min(1.0, float(shot.get("peak", 0)))))
        kept, hashes = [], []
        for candidate in pool:
            candidate.image = trim_bars(candidate.image)
            if candidate.image.width > 2560:
                # 4K stills and frames are ~25 MB each in memory; 2560 px is plenty for a 1280x720 thumbnail.
                ratio = 2560 / candidate.image.width
                candidate.image = candidate.image.resize((2560, round(candidate.image.height * ratio)), Image.LANCZOS)
            sharp = frame_sharpness(candidate.image)
            if sharp < 25:
                continue  # soft, motion-blurred or out-of-focus frames never become thumbnails
            candidate.face, quality, good = face_quality(candidate.image)
            candidate.notes = {"sharpness": round(sharp), "face_quality": round(quality, 2), "good_face": good}
            base, _ = heuristic(candidate.image, profile["exposure"])
            candidate.score = base * 0.4 + quality * (1.2 if good else 0.5) + candidate.genre_fit * 2.0 + min(sharp, 400) / 400
            value = _dhash(candidate.image)
            if any((value ^ old).bit_count() <= 8 for old in hashes):
                continue  # near-duplicate of a frame already kept
            candidate.notes["hash"] = value
            kept.append(candidate)
            hashes.append(value)
        kept.sort(key=lambda c: -c.score)
        clean = [c for c in kept if not has_burned_text(c.image)][:per_film]
        candidates.extend(clean or kept[:2])
    return candidates


def good_face(candidate: Candidate) -> bool:
    return bool(candidate.face is not None and candidate.notes.get("good_face") and not candidate.notes.get("logo"))


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
        candidate.notes = {**candidate.notes, **verdict}
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
    rgb += shadows * np.array([-4, 2, 6], dtype=np.float32) * strength
    rgb += highlights * np.array([8, 4, -5], dtype=np.float32) * strength
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
    elif grade == "warm":
        hsv[..., 1] *= 1.08
        hsv[..., 2] = 255 * np.clip(value * 1.03 + 0.02, 0, 1)
        tint, vignette_amount, grain = np.array([10, 4, -8]), 0.3, 1.5
    elif grade == "magic":
        hsv[..., 1] *= 1.15
        tint, vignette_amount, grain = np.array([6, -2, 16]), 0.42, 1.5
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
def face_light(canvas: Image.Image, center: tuple[float, float], radius: float, amount: float = 0.14) -> Image.Image:
    """Lift the face slightly above its surroundings (the 'lit subject' look of channel thumbnails)."""
    w, h = canvas.size
    y, x = np.ogrid[:h, :w]
    distance = np.sqrt((x - center[0]) ** 2 + (y - center[1]) ** 2) / max(1.0, radius)
    gain = 1 + amount * np.clip(1 - distance, 0, 1) ** 1.5
    rgb = np.asarray(canvas.convert("RGB")).astype(np.float32) * gain[..., None]
    return Image.fromarray(np.clip(rgb, 0, 255).astype(np.uint8))


def face_crop(candidate: Candidate, width: int = WIDTH, height: int = HEIGHT, *, face_height: float = 0.40,
              face_x: float = 0.62, face_y: float = 0.40, light: bool = True) -> Image.Image:
    """Crop so the face fills about ``face_height`` of the frame at the chosen position.

    Enlargement is capped (1.6x) so faces never turn soft; the face gets a gentle light.
    """
    image = candidate.image.convert("RGB")
    if candidate.face is None:
        scale = max(width / image.width, height / image.height)
        resized = image.resize((round(image.width * scale), round(image.height * scale)), Image.LANCZOS)
        left, top = (resized.width - width) // 2, (resized.height - height) // 2
        return resized.crop((left, top, left + width, top + height))
    x, y, w, h = candidate.face
    cover = max(width / image.width, height / image.height)
    scale = max(cover, min(max(cover, 1.6), face_height * height / max(h, 1)))
    resized = image.resize((round(image.width * scale), round(image.height * scale)), Image.LANCZOS)
    cx, cy = (x + w / 2) * scale, (y + h / 2) * scale
    left = int(min(max(0, cx - width * face_x), resized.width - width))
    top = int(min(max(0, cy - height * face_y), resized.height - height))
    crop = resized.crop((left, top, left + width, top + height))
    if scale > 1.1:
        crop = crop.filter(ImageFilter.UnsharpMask(radius=1.0 + scale * 0.5, percent=int(min(120, 50 * scale)), threshold=2))
    if light:
        crop = face_light(crop, (cx - left, cy - top), h * scale * 1.3)
    return crop


def clean_cutout(subject: Image.Image | None) -> Image.Image | None:
    """Keep a cut-out only when its edge is clean: one solid shape, no stray blobs.

    The alpha edge is tightened by a pixel (removes the old background's colour fringe)
    and softened slightly so it never looks jagged.
    """
    if subject is None:
        return None
    alpha = np.asarray(subject.getchannel("A"))
    solid = (alpha > 128).astype(np.uint8)
    if solid.sum() < 1000:
        return None
    count, labels, stats, _ = cv2.connectedComponentsWithStats(solid, 8)
    largest = stats[1:, cv2.CC_STAT_AREA].max() if count > 1 else 0
    if largest < 0.93 * solid.sum():
        return None
    contours, _ = cv2.findContours(solid, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    contour = max(contours, key=cv2.contourArea)
    hull = cv2.contourArea(cv2.convexHull(contour))
    if hull <= 0 or cv2.contourArea(contour) / hull < 0.68:
        return None  # ragged, spiky or hollow shape: a failed cut-out
    refined = Image.fromarray(alpha).filter(ImageFilter.MinFilter(3)).filter(ImageFilter.GaussianBlur(1.1))
    result = subject.copy()
    result.putalpha(refined)
    return result


def rim_light(subject: Image.Image, color=(255, 214, 160), width: int = 4, strength: float = 0.6) -> Image.Image:
    alpha = subject.getchannel("A")
    edge = Image.fromarray(np.clip(np.asarray(alpha.filter(ImageFilter.MaxFilter(width * 2 + 1)), dtype=np.int16)
                                   - np.asarray(alpha, dtype=np.int16), 0, 255).astype(np.uint8))
    glow = Image.new("RGBA", subject.size, color + (0,))
    glow.putalpha(edge.filter(ImageFilter.GaussianBlur(width)).point(lambda v: min(255, int(v * strength))))
    return glow


def depth_composite(subject: Image.Image, background: Image.Image, *, height_fraction: float = 1.05,
                    x_fraction: float = 0.62, blur: float = 8, darken: float = 0.5, rim=None) -> Image.Image:
    """Cut-out subject over a different, softened and darkened scene, with a soft shadow (no glow)."""
    cover = max(WIDTH / background.width, HEIGHT / background.height)
    canvas = background.convert("RGB").resize((round(background.width * cover), round(background.height * cover)), Image.LANCZOS)
    left, top = (canvas.width - WIDTH) // 2, (canvas.height - HEIGHT) // 2
    canvas = canvas.crop((left, top, left + WIDTH, top + HEIGHT)).filter(ImageFilter.GaussianBlur(blur))
    canvas = ImageEnhance.Brightness(canvas).enhance(darken).convert("RGBA")
    scale = HEIGHT * height_fraction / subject.height
    subject = subject.resize((max(1, round(subject.width * scale)), max(1, round(subject.height * scale))), Image.LANCZOS)
    x = round(WIDTH * x_fraction - subject.width / 2)
    x = max(-subject.width // 4, min(WIDTH - subject.width * 3 // 4, x))
    y = HEIGHT - subject.height + round(HEIGHT * 0.06)
    shadow = Image.new("RGBA", subject.size, (0, 0, 0, 0))
    shadow.putalpha(subject.getchannel("A").point(lambda v: int(v * 0.6)).filter(ImageFilter.GaussianBlur(22)))
    canvas.alpha_composite(shadow, (x + 18, y + 12))
    if rim:
        canvas.alpha_composite(rim_light(subject, rim, width=2, strength=0.35), (x, y))
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
        piece = face_crop(candidate, panel_width, HEIGHT, face_height=0.42, face_x=0.5, face_y=0.40)
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


# Simple drawn wordmarks (our own lettering in each service's colours; no copied artwork).
SERVICE_MARKS = {
    "netflix": {"text": "N", "fg": (229, 9, 20), "bg": None},
    "hbo max": {"text": "HBO max", "fg": (255, 255, 255), "bg": (0, 26, 140), "bg2": (90, 40, 200)},
    "prime video": {"text": "prime video", "fg": (255, 255, 255), "bg": (0, 120, 200), "bg2": (0, 168, 225), "smile": True},
    "hulu": {"text": "hulu", "fg": (28, 231, 131), "bg": (10, 10, 12)},
    "disney plus": {"text": "Disney+", "fg": (255, 255, 255), "bg": (4, 26, 92), "bg2": (17, 60, 207)},
    "paramount plus": {"text": "Paramount+", "fg": (255, 255, 255), "bg": (0, 64, 200), "bg2": (0, 100, 255)},
    "peacock": {"text": "peacock", "fg": (255, 255, 255), "bg": (12, 12, 12)},
    "apple tv": {"text": "tv", "fg": (255, 255, 255), "bg": (12, 12, 12), "apple": True},
    "shudder": {"text": "SHUDDER", "fg": (255, 255, 255), "bg": (200, 10, 20)},
}
RANDOM_SERVICES = ("Netflix", "HBO Max", "Prime Video")


def service_mark(canvas: Image.Image, service: str, *, height: int = 150, corner: str = "left") -> Image.Image:
    """A large, eye-catching service mark in the upper corner (thumbnail 4)."""
    canvas = canvas.convert("RGBA")
    spec = SERVICE_MARKS.get(service.casefold())
    margin = 40
    if spec and spec["text"] == "N":
        mark = Image.new("RGBA", (125, 175))
        draw = ImageDraw.Draw(mark)
        draw.polygon([(0, 0), (36, 0), (36, 158), (0, 164)], fill="#ad0610")
        draw.polygon([(78, 0), (114, 0), (114, 164), (78, 158)], fill="#ad0610")
        draw.polygon([(0, 0), (36, 0), (114, 164), (78, 158)], fill="#e50914")
        scale = height * 1.5 / 175
        mark = mark.resize((round(125 * scale), round(175 * scale)), Image.LANCZOS)
    else:
        spec = spec or {"text": service, "fg": (255, 255, 255), "bg": SERVICE_COLORS.get(service.casefold(), (200, 20, 30))}
        font = brand_font("body", round(height * 0.5))
        probe = ImageDraw.Draw(Image.new("RGBA", (1, 1)))
        text_width = probe.textlength(spec["text"], font=font)
        width = round(text_width + height * 0.8 + (height * 0.45 if spec.get("apple") else 0))
        mark = Image.new("RGBA", (width, height), (0, 0, 0, 0))
        if spec.get("bg"):
            fill = Image.new("RGBA", (width, height), spec["bg"] + (255,))
            if spec.get("bg2"):
                ramp = np.linspace(0, 1, width)[None, :, None]
                colors = np.asarray(spec["bg"], np.float32) * (1 - ramp) + np.asarray(spec["bg2"], np.float32) * ramp
                fill = Image.fromarray(np.repeat(colors, height, axis=0).astype(np.uint8)).convert("RGBA")
            rounded = Image.new("L", (width, height), 0)
            ImageDraw.Draw(rounded).rounded_rectangle((0, 0, width - 1, height - 1), radius=height // 5, fill=255)
            mark.paste(fill, (0, 0), rounded)
        draw = ImageDraw.Draw(mark)
        x = height * 0.4
        if spec.get("apple"):
            r = height * 0.16
            cx, cy = x + r, height * 0.52
            draw.ellipse((cx - r, cy - r, cx + r, cy + r * 1.1), fill=spec["fg"])
            draw.ellipse((cx + r * 0.1, cy - r * 1.9, cx + r * 0.8, cy - r * 1.1), fill=spec["fg"])
            x += height * 0.45
        draw.text((x, height * (0.44 if spec.get("smile") else 0.5)), spec["text"], font=font, anchor="lm", fill=spec["fg"])
        if spec.get("smile"):
            draw.arc((x, height * 0.42, x + text_width * 0.6, height * 0.86), 20, 160, fill=spec["fg"], width=max(3, height // 22))
    glow = Image.new("RGBA", (mark.width + 80, mark.height + 80), (0, 0, 0, 0))
    glow.putalpha(Image.new("L", glow.size, 0))
    shadow = Image.new("RGBA", glow.size, (0, 0, 0, 0))
    shadow.paste((0, 0, 0, 170), (40, 40), mark.getchannel("A"))
    shadow = shadow.filter(ImageFilter.GaussianBlur(16))
    x = margin if corner == "left" else WIDTH - mark.width - margin
    canvas.alpha_composite(shadow, (x - 40 + 6, margin - 40 + 8))
    canvas.alpha_composite(mark, (x, margin))
    return canvas


def choose_service(films: list[dict], shown: list[int], seed: str = "") -> tuple[str, str]:
    """The service for thumbnail 4: number one's own; else a featured film's; else a random big one."""
    from .availability import services
    for index in [0, *[i for i in shown if i != 0], *range(len(films))]:
        if index < len(films):
            found = services(films[index].get("watch_availability", {}), ("subscription", "free", "ads"), limit=1)
            if found:
                return found[0], f"listed for {films[index].get('title', '')}"
    import hashlib
    pick = RANDOM_SERVICES[int(hashlib.sha256(seed.encode()).hexdigest()[:6], 16) % len(RANDOM_SERVICES)]
    return pick, "random (no recognised listing)"


def headline(canvas: Image.Image, phrase: str, color, *, side: str = "left") -> Image.Image:
    """Thumbnail 5: two to four huge words, stacked, in the genre colour over a dark fade."""
    canvas = canvas.convert("RGBA")
    words = [w for w in re.sub(r"[^\w'’ -]", " ", phrase).upper().split() if w][:4] or ["MUST", "WATCH"]
    # One word per line for up to three words (bigger letters); pairs for four.
    lines = words if len(words) <= 3 else [" ".join(words[:2]), " ".join(words[2:])]
    shade = Image.new("L", canvas.size, 0)
    ramp = np.clip(1.15 - np.linspace(0, 1.6, WIDTH), 0, 1)
    if side == "right":
        ramp = ramp[::-1]
    shade = Image.fromarray((np.repeat(ramp[None, :], HEIGHT, axis=0) * 205).astype(np.uint8))
    dark = Image.new("RGBA", canvas.size, (0, 0, 0, 255))
    dark.putalpha(shade)
    canvas = Image.alpha_composite(canvas, dark)
    draw = ImageDraw.Draw(canvas)
    box_width = WIDTH * 0.58
    size = 230
    while size > 70 and len(lines) * size * 0.98 > HEIGHT * 0.9:
        size -= 6
    while size > 70 and max(draw.textlength(line, font=brand_font("display", size)) for line in lines) > box_width:
        size -= 6
    font = brand_font("display", size)
    total = len(lines) * size * 0.98
    y = (HEIGHT - total) / 2 + size * 0.49
    for i, line in enumerate(lines):
        x = 60 if side == "left" else WIDTH - 60 - draw.textlength(line, font=font)
        fill = color if i == len(lines) - 1 and len(lines) > 1 else (255, 255, 255)
        draw.text((x, y), line, font=font, anchor="lm", fill=fill, stroke_width=max(4, size // 22), stroke_fill=(0, 0, 0))
        y += size * 0.98
    return canvas


def short_phrase(title: str, phrases: list[str], genre_label: str) -> str:
    """Two to four words for thumbnail 5, e.g. 'HORROR MOVIES YOU CAN'T MISS' -> 'CAN'T MISS HORROR'."""
    for phrase in phrases:
        words = str(phrase).split()
        if 2 <= len(words) <= 4 and len(str(phrase)) <= 28:
            return str(phrase)
    words = [w for w in re.findall(r"[A-Za-z']+", title) if w.casefold() not in {"the", "of", "that", "will", "from", "your", "a", "to"}]
    return " ".join(words[:3]) if words else f"{genre_label} must-watch"


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
def _similar(a: Candidate, b: Candidate) -> bool:
    return a is b or ("hash" in a.notes and "hash" in b.notes and (a.notes["hash"] ^ b.notes["hash"]).bit_count() <= 10)


class _Picker:
    """Hands out candidates so no image (or near-identical frame) appears in two designs."""

    def __init__(self, candidates: list[Candidate]):
        self.candidates = sorted(candidates, key=lambda c: -c.score)
        self.used: list[Candidate] = []

    def free(self, *, film: int | None = None, faces: bool = False) -> list[Candidate]:
        return [c for c in self.candidates if not any(_similar(c, u) for u in self.used)
                and (film is None or c.film == film) and (not faces or good_face(c))]

    def any(self, *, film: int | None = None, faces: bool = False) -> Candidate:
        """The best unused match; when everything has been used, the best overall (small lists)."""
        for options in (self.free(film=film, faces=faces), self.free(faces=faces), self.free(film=film), self.free()):
            if options:
                return options[0]
        return self.candidates[0]

    def take(self, candidate: Candidate | None) -> Candidate | None:
        if candidate is not None:
            self.used.append(candidate)
        return candidate


def _backdrop_for(picker: _Picker, subject: Candidate) -> Candidate | None:
    """A different scene for behind a cut-out: same film first, preferring frames without a big face."""
    options = [c for c in picker.free(film=subject.film) if not _similar(c, subject)] or \
              [c for c in picker.free() if not _similar(c, subject)]
    if not options:
        return None
    return max(options, key=lambda c: (c.face is None) * 2 + c.action + c.genre_fit)


def _face_design(picker: _Picker, cutter, candidate: Candidate, grade: str, *, face_x: float, height_fraction: float,
                 face_height: float = 0.5) -> tuple[Image.Image, str]:
    """Clean cut-out over another scene when possible, otherwise a tight, lit full-frame crop."""
    subject = clean_cutout(cutter.cut(candidate.image))
    backdrop = _backdrop_for(picker, candidate) if subject is not None else None
    if subject is not None and backdrop is not None:
        picker.take(backdrop)
        return depth_composite(subject, backdrop.image, height_fraction=height_fraction, x_fraction=face_x,
                               darken=0.42 if grade == "horror" else 0.5), "cutout"
    return face_crop(candidate, face_height=face_height, face_x=face_x, face_y=0.42), "crop"


def actor_cutouts(picker: _Picker, cutter, films: list[dict], limit_per_film: int = 3) -> list[tuple[Candidate, Image.Image]]:
    """Clean person cut-outs from the best good faces, number one first, at most a few per title."""
    from .thumbnail_composite import person_cutout
    result, per_film = [], {}
    ordered = sorted((c for c in picker.candidates if good_face(c)),
                     key=lambda c: (c.film, c.source != "tmdb-still", -c.score))
    for candidate in ordered:
        if per_film.get(candidate.film, 0) >= limit_per_film:
            continue
        cut = person_cutout(cutter, candidate.image, candidate.face, clean_cutout)
        if cut is not None:
            result.append((candidate, cut))
            per_film[candidate.film] = per_film.get(candidate.film, 0) + 1
    return result


def build(films: list[dict], trailers: list[Path], shot_sets: list[list[dict]], phrases: list[str],
          output: Path, work: Path, title: str | None = None, brief: str | None = None) -> list[dict]:
    from . import art
    from .thumbnail_composite import background, compose
    work.mkdir(parents=True, exist_ok=True)
    output.mkdir(parents=True, exist_ok=True)
    title = os.getenv("VIDEO_TITLE", "") if title is None else title
    brief = os.getenv("VIDEO_BRIEF", "") if brief is None else brief
    genre = detect_genre(films, title, brief)
    profile = GENRE_PROFILES[genre]
    candidates = gather(films, trailers, shot_sets, work, profile=profile)
    if not any(c.film == 0 for c in candidates):
        raise PipelineError("No usable thumbnail image for number one.")
    rated = rate_with_vision(candidates, profile=profile)
    candidates = [c for c in candidates if not c.notes.get("logo")] or candidates
    for candidate in candidates:
        if candidate.notes.get("large_face") is False and candidate.face is not None and not candidate.notes.get("good_face"):
            candidate.face = None  # the vision model sees no real face here
    picker = _Picker(candidates)
    cutter = Cutter()
    report = []
    grade = profile["grade"]
    scenes = art.available() if os.getenv("THUMBNAIL_COMPOSITE", "true").strip().lower() != "false" else {}
    actors = actor_cutouts(picker, cutter, films) if scenes else []
    used_actors: set[int] = set()
    grade_scene = (lambda image: genre_grade(image, grade, 0.6)) if grade != "blockbuster" else (lambda image: cinematic_grade(image, 0.6))

    def save_design(number: int, image: Image.Image, style: str, chosen: list[Candidate], **extra):
        save(image, output / f"thumbnail-0{number}.jpg")
        report.append({"file": f"thumbnail-0{number}.jpg", "style": style,
                       "films": [films[c.film]["title"] for c in chosen], "sources": [c.source for c in chosen],
                       "faces": [c.notes.get("face_quality") for c in chosen], **extra})

    def actor(film: int | None = None, *, fresh: bool = True):
        """The best unused actor cut-out (from ``film`` when given)."""
        for index, (candidate, cut) in enumerate(actors):
            if (film is None or candidate.film == film) and (not fresh or index not in used_actors):
                used_actors.add(index)
                picker.take(candidate)
                return candidate, cut
        return None

    def distinct_films(count: int, fresh: bool = True):
        chosen, films_seen = [], set()
        for index, (candidate, cut) in enumerate(actors):
            if (not fresh or index not in used_actors) and candidate.film not in films_seen:
                chosen.append((index, candidate, cut))
                films_seen.add(candidate.film)
            if len(chosen) == count:
                break
        if len(chosen) < count:
            return distinct_films(count, fresh=False) if fresh else None  # small lists: reuse a picture
        for index, candidate, _ in chosen:
            used_actors.add(index)
            picker.take(candidate)
        return [(candidate, cut) for _, candidate, cut in chosen]

    def finish_composite(image: Image.Image) -> Image.Image:
        return finish(image, grade) if grade in ("horror", "cold") else cinematic_grade(image, 0.7)

    # 1. Hero: number one's actor, huge, in the hero scene.
    hero_pick = actor(0) if "hero" in scenes else None
    if hero_pick:
        image = compose(background(scenes["hero"], grade_scene), [(hero_pick[1], 0.6, 1.18, 1.0)], genre, seed=1)
        save_design(1, finish_composite(image), "composite-hero", [hero_pick[0]], scene="hero")
    else:
        hero = picker.take(picker.any(film=0, faces=True))
        image, mode = _face_design(picker, cutter, hero, grade, face_x=0.6, height_fraction=1.15, face_height=0.52)
        save_design(1, finish(image, grade), f"hero-{mode}", [hero])

    # 2. Face-off: two actors from different titles at the edges, the scene's light between them.
    pair = distinct_films(2) if "faceoff" in scenes else None
    faces = []
    if pair:
        image = compose(background(scenes["faceoff"], grade_scene),
                        [(pair[0][1], 0.24, 1.08, 1.0), (pair[1][1], 0.77, 1.08, 1.0)], genre, seed=2)
        save_design(2, finish_composite(image), "composite-faceoff", [pair[0][0], pair[1][0]], scene="faceoff")
        faces = [pair[0][0], pair[1][0]]
    else:
        for film_index in sorted({c.film for c in picker.candidates}):
            option = next(iter(picker.free(film=film_index, faces=True)), None)
            if option is not None:
                faces.append(picker.take(option))
            if len(faces) == 3:
                break
        if len(faces) >= 2:
            save_design(2, finish(panels(faces, profile["divider"]), grade), f"faces-{len(faces)}", faces)
        else:
            for face in faces:
                picker.used.remove(face)
            single = picker.take(picker.any(faces=True))
            image, mode = _face_design(picker, cutter, single, grade, face_x=0.42, height_fraction=1.1)
            save_design(2, finish(image, grade), f"face-{mode}", [single])

    # 3. Ensemble: three titles' actors, the centre one biggest (team-poster layout); else the spectacle frame.
    trio = distinct_films(3) if "ensemble" in scenes else None
    if trio:
        image = compose(background(scenes["ensemble"], grade_scene),
                        [(trio[1][1], 0.2, 0.95, 0.85), (trio[2][1], 0.8, 0.95, 0.85), (trio[0][1], 0.5, 1.15, 1.0)],
                        genre, seed=3)
        save_design(3, finish_composite(image), "composite-ensemble", [c for c, _ in trio], scene="ensemble")
    else:
        scenes_left = picker.free() or picker.candidates
        spectacle = picker.take(max(scenes_left, key=lambda c: c.action * 2 + c.genre_fit * 2 + c.notes.get("score", 5) / 5
                                    + (0.0 if c.face is None else -0.4)))
        save_design(3, finish(face_crop(spectacle, face_height=0.4, face_x=0.5), grade), "spectacle", [spectacle])

    # 4. Streaming mark: number one again (another picture) in its scene, the service mark top-left.
    mark_service, mark_reason = choose_service(films, [c.film for c in faces], title)
    star = (actor(0) or actor() or actor(0, fresh=False)) if "service" in scenes else None
    if star:
        image = compose(background(scenes["service"], grade_scene), [(star[1], 0.67, 1.12, 1.0)], genre, seed=4)
        save_design(4, service_mark(finish_composite(image), mark_service), "composite-service", [star[0]],
                    scene="service", service=mark_service, service_reason=mark_reason)
    else:
        candidate = picker.take(picker.any(film=0, faces=True))
        image, mode = _face_design(picker, cutter, candidate, grade, face_x=0.66, height_fraction=1.1)
        save_design(4, service_mark(finish(image, grade), mark_service), f"service-{mode}", [candidate],
                    service=mark_service, service_reason=mark_reason)

    # 5. Headline: two to four huge words left, an actor right, in a mirrored scene.
    phrase = short_phrase(title, phrases, profile["label"])
    words_scene = scenes.get("hero") or scenes.get("service") or scenes.get("ensemble")
    face_pick = (actor() or actor(fresh=False)) if words_scene else None
    if face_pick:
        scene = background(words_scene, grade_scene).transpose(Image.FLIP_LEFT_RIGHT)
        image = compose(scene, [(face_pick[1], 0.75, 1.1, 1.0)], genre, seed=5)
        save_design(5, headline(finish_composite(image), phrase, profile["rim"]), "composite-headline", [face_pick[0]], text=phrase)
    else:
        candidate = picker.take(picker.any(faces=True))
        image, mode = _face_design(picker, cutter, candidate, grade, face_x=0.74, height_fraction=1.08)
        save_design(5, headline(finish(image, grade), phrase, profile["rim"]), f"headline-{mode}", [candidate], text=phrase)

    save_json(work / "thumbnail-report.json", {"genre": genre, "vision_rated": rated, "candidates": len(candidates),
              "good_faces": sum(good_face(c) for c in candidates), "scenes": sorted(scenes), "actor_cutouts": len(actors),
              "thumbnails": report,
              "pool": [{"film": films[c.film]["title"], "source": c.source, **{k: v for k, v in c.notes.items() if k != "hash"}}
                       for c in picker.candidates]})
    print(f"Studio thumbnails ({genre}): {', '.join(item['style'] for item in report)}; scenes {sorted(scenes)}; "
          f"{len(actors)} actor cut-outs; {sum(good_face(c) for c in candidates)} good faces among {len(candidates)} candidates; "
          f"{rated} rated by the vision model.", flush=True)
    return report
