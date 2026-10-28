"""Composite thumbnails: real actors from official stills placed into generated genre scenes.

The look of channel thumbnails made in Photoshop: a huge, sharp actor cut-out over a dramatic
scene, colour-matched to it, lit on one side by the scene's own light (rim light), with a soft
shadow and genre effects in front of and behind the actor (sparks and embers for action, fog
for horror, dust for adventure, bokeh for romance, particles for sci-fi and fantasy).
"""
from __future__ import annotations

import cv2
import numpy as np
from PIL import Image, ImageEnhance, ImageFilter

WIDTH, HEIGHT = 1280, 720

EFFECTS = {
    "action": ("embers", "smoke"), "adventure": ("dust", "rays"), "horror": ("fog", "dust"), "thriller": ("fog", "rain"),
    "scifi": ("particles", "rays"), "fantasy": ("particles", "fog"), "comedy": ("bokeh",), "family": ("bokeh", "particles"),
    "romance": ("bokeh", "rays"), "drama": ("dust", "rays"), "documentary": ("dust",),
}


def background(path, grade_fn=None) -> Image.Image:
    image = Image.open(path).convert("RGB")
    cover = max(WIDTH / image.width, HEIGHT / image.height)
    image = image.resize((round(image.width * cover), round(image.height * cover)), Image.LANCZOS)
    left, top = (image.width - WIDTH) // 2, (image.height - HEIGHT) // 2
    image = image.crop((left, top, left + WIDTH, top + HEIGHT)).filter(ImageFilter.UnsharpMask(radius=1.2, percent=60, threshold=2))
    return grade_fn(image) if grade_fn else image


def scene_light(scene: Image.Image) -> tuple[str, tuple[int, int, int]]:
    """Which side the scene is lit from, and the colour of its brightest light."""
    rgb = np.asarray(scene.convert("RGB").resize((320, 180)), dtype=np.float32)
    luminance = rgb.mean(axis=2)
    side = "left" if luminance[:, :107].mean() > luminance[:, 213:].mean() else "right"
    bright = rgb[luminance >= np.percentile(luminance, 97)]
    color = tuple(int(v) for v in (bright.mean(axis=0) if len(bright) else (255, 220, 180)))
    peak = max(color) or 1
    return side, tuple(min(255, int(v * 255 / peak)) for v in color)


def harmonize(subject: Image.Image, scene: Image.Image, amount: float = 0.18) -> Image.Image:
    """Pull the actor's colour and contrast toward the scene's (LAB statistics), keeping skin natural."""
    rgba = np.asarray(subject.convert("RGBA"))
    alpha = rgba[..., 3] > 128
    if alpha.sum() < 100:
        return subject
    sub_lab = cv2.cvtColor(rgba[..., :3], cv2.COLOR_RGB2LAB).astype(np.float32)
    scene_lab = cv2.cvtColor(np.asarray(scene.convert("RGB")), cv2.COLOR_RGB2LAB).astype(np.float32).reshape(-1, 3)
    mean_s, std_s = sub_lab[alpha].mean(axis=0), sub_lab[alpha].std(axis=0) + 1e-3
    mean_t, std_t = scene_lab.mean(axis=0), scene_lab.std(axis=0) + 1e-3
    weights = np.array([0.6, 0.7, 0.7]) * amount  # gentle: skin must stay natural
    target = (sub_lab - mean_s) * (1 + weights * (std_t / std_s - 1)) + mean_s + weights * (mean_t - mean_s)
    out = cv2.cvtColor(np.clip(target, 0, 255).astype(np.uint8), cv2.COLOR_LAB2RGB)
    result = Image.fromarray(np.dstack([out, rgba[..., 3]]), "RGBA")
    return result


def rim_light(subject: Image.Image, side: str, color: tuple[int, int, int], width: int = 5, strength: float = 0.45) -> Image.Image:
    """Light wrapping the actor's edge on the side the scene's light comes from."""
    alpha = np.asarray(subject.getchannel("A"), dtype=np.float32) / 255
    shift = width if side == "left" else -width
    shifted = np.roll(alpha, shift, axis=1)
    edge = np.clip(alpha - shifted, 0, 1)
    edge = cv2.GaussianBlur(edge, (0, 0), width / 2) * alpha
    layer = np.zeros((*alpha.shape, 4), dtype=np.uint8)
    layer[..., :3] = color
    layer[..., 3] = np.clip(edge * 255 * strength * 1.8, 0, 255).astype(np.uint8)
    return Image.fromarray(layer, "RGBA")


def _noise(shape, scale: float, seed: int) -> np.ndarray:
    rng = np.random.default_rng(seed)
    small = rng.random((max(2, int(shape[0] / scale)), max(2, int(shape[1] / scale)))).astype(np.float32)
    return cv2.resize(small, (shape[1], shape[0]), interpolation=cv2.INTER_CUBIC)


def effect(kind: str, light: tuple[int, int, int], seed: int, *, front: bool) -> Image.Image:
    """One genre effect layer (RGBA, full frame). Front layers are sparser and larger (closer to camera)."""
    rng = np.random.default_rng(seed + (7 if front else 0))
    layer = Image.new("RGBA", (WIDTH, HEIGHT), (0, 0, 0, 0))
    canvas = np.zeros((HEIGHT, WIDTH, 4), dtype=np.float32)
    if kind in ("embers", "particles", "dust"):
        count = {"embers": 140, "particles": 110, "dust": 160}[kind] // (3 if front else 1)
        colors = {"embers": [(255, 170, 60), (255, 120, 30), (255, 220, 120)], "particles": [light, (180, 220, 255)],
                  "dust": [(255, 240, 210)]}[kind]
        for _ in range(count):
            x, y = rng.uniform(0, WIDTH), rng.uniform(0, HEIGHT)
            radius = rng.uniform(1.5, 4.5) * (2.2 if front else 1)
            length = rng.uniform(4, 22) if kind == "embers" else 0
            color = colors[rng.integers(len(colors))]
            alpha = rng.uniform(0.35, 0.95) * (0.6 if kind == "dust" else 1)
            angle = rng.uniform(-0.6, 0.6) - np.pi / 2
            steps = max(1, int(length / 2))
            for step in range(steps):
                px, py = int(x + np.cos(angle) * step * 2), int(y + np.sin(angle) * step * 2)
                if 0 <= px < WIDTH and 0 <= py < HEIGHT:
                    cv2.circle(canvas, (px, py), max(1, int(radius * (1 - step / (steps + 1)))), (*color, alpha * 255), -1)
        canvas = cv2.GaussianBlur(canvas, (0, 0), 3.0 if front else 1.2)
        glow = cv2.GaussianBlur(canvas, (0, 0), 8)
        canvas = np.clip(canvas + glow * 0.8, 0, 255)
    elif kind in ("smoke", "fog"):
        density = _noise((HEIGHT, WIDTH), 90, seed) * 0.6 + _noise((HEIGHT, WIDTH), 30, seed + 1) * 0.4
        ramp = np.linspace(0.2, 1.0, HEIGHT)[:, None] if kind == "fog" else np.linspace(1.0, 0.4, HEIGHT)[:, None]
        amount = (0.35 if front else 0.55) * (0.6 if kind == "smoke" else 1)
        tone = (40, 40, 44) if kind == "smoke" else (200, 210, 225)
        canvas[..., :3] = tone
        canvas[..., 3] = np.clip((density - 0.35) * 2.2, 0, 1) * ramp * amount * 255
    elif kind == "bokeh":
        for _ in range(16 if front else 34):
            x, y = int(rng.uniform(0, WIDTH)), int(rng.uniform(0, HEIGHT))
            radius = int(rng.uniform(14, 46) * (1.8 if front else 1))
            color = [(255, 210, 160), (255, 170, 190), light][rng.integers(3)]
            cv2.circle(canvas, (x, y), radius, (*color, rng.uniform(40, 110)), -1)
        canvas = cv2.GaussianBlur(canvas, (0, 0), 4 if front else 2)
    elif kind == "rays" and not front:
        yy, xx = np.mgrid[0:HEIGHT, 0:WIDTH].astype(np.float32)
        origin = (WIDTH * rng.uniform(0.3, 0.7), -HEIGHT * 0.1)
        angle = np.arctan2(yy - origin[1], xx - origin[0])
        beams = (np.sin(angle * 38 + seed) * 0.5 + 0.5) ** 6
        fade = np.clip(1 - np.hypot(xx - origin[0], yy - origin[1]) / (HEIGHT * 1.3), 0, 1)
        canvas[..., :3] = light
        canvas[..., 3] = beams * fade * 70
    elif kind == "rain":
        for _ in range(220 if not front else 60):
            x, y = int(rng.uniform(0, WIDTH)), int(rng.uniform(0, HEIGHT))
            length = int(rng.uniform(12, 30) * (1.6 if front else 1))
            cv2.line(canvas, (x, y), (x - length // 6, y + length), (210, 220, 235, 90), 1)
        canvas = cv2.GaussianBlur(canvas, (0, 0), 0.8)
    return Image.fromarray(np.clip(canvas, 0, 255).astype(np.uint8), "RGBA") if canvas.any() else layer


def place(canvas: Image.Image, subject: Image.Image, *, x_center: float, height: float, side: str,
          light: tuple[int, int, int], darken: float = 1.0) -> Image.Image:
    """Scale the actor to ``height`` of the frame, bottom-anchored, with shadow and rim light."""
    scale = HEIGHT * height / subject.height
    subject = subject.resize((max(1, round(subject.width * scale)), max(1, round(subject.height * scale))), Image.LANCZOS)
    if darken != 1.0:
        alpha = subject.getchannel("A")
        subject = ImageEnhance.Brightness(subject.convert("RGB")).enhance(darken).convert("RGBA")
        subject.putalpha(alpha)
    x = round(WIDTH * x_center - subject.width / 2)
    # Headroom: the top of the head sits just below the frame edge; the body runs off the bottom.
    y = max(round(HEIGHT * 0.05), HEIGHT - subject.height + round(HEIGHT * 0.04))
    shadow = Image.new("RGBA", subject.size, (0, 0, 0, 0))
    shadow.putalpha(subject.getchannel("A").point(lambda v: int(v * 0.65)).filter(ImageFilter.GaussianBlur(26)))
    canvas.alpha_composite(shadow, (x + (-22 if side == "right" else 22), y + 14))
    canvas.alpha_composite(subject, (x, y))
    canvas.alpha_composite(rim_light(subject, side, light), (x, y))
    return canvas


def compose(scene: Image.Image, subjects: list[tuple[Image.Image, float, float, float]], genre: str, seed: int = 3) -> Image.Image:
    """``subjects``: (cut-out, x centre, height fraction, brightness) drawn back to front."""
    side, light = scene_light(scene)
    canvas = scene.convert("RGBA")
    kinds = EFFECTS.get(genre, ("dust",))
    for index, kind in enumerate(kinds):
        canvas.alpha_composite(effect(kind, light, seed + index, front=False))
    # Darken the scene a touch behind the actors so they pop.
    shade = np.zeros((HEIGHT, WIDTH, 4), dtype=np.uint8)
    for _, x_center, height, _ in subjects:
        cv2.ellipse(shade, (int(WIDTH * x_center), int(HEIGHT * 0.7)), (int(WIDTH * 0.22 * height), int(HEIGHT * 0.6)),
                    0, 0, 360, (0, 0, 0, 90), -1)
    canvas.alpha_composite(Image.fromarray(cv2.GaussianBlur(shade, (0, 0), 60), "RGBA"))
    for subject, x_center, height, brightness in subjects:
        place(canvas, harmonize(subject, scene), x_center=x_center, height=height, side=side, light=light, darken=brightness)
    for index, kind in enumerate(kinds[:1]):
        canvas.alpha_composite(effect(kind, light, seed + 10 + index, front=True))
    return canvas.convert("RGB")


def person_cutout(cutter, image: Image.Image, face: tuple[int, int, int, int], clean) -> Image.Image | None:
    """Cut out just the person whose face this is: crop around them first, then remove the background."""
    x, y, w, h = face
    left = max(0, int(x - w * 1.6))
    right = min(image.width, int(x + w * 2.6))
    top = max(0, int(y - h * 0.9))
    crop = image.crop((left, top, right, image.height))
    if crop.width < 200 or crop.height < 300:
        return None
    return clean(cutter.cut(crop))
