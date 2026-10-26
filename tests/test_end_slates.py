import shutil

import cv2
import numpy as np
import pytest
from PIL import Image, ImageDraw

from movie_pipeline.render import brand_font
from movie_pipeline.shots import analyse, promo_cutoff

pytestmark = pytest.mark.skipif(not shutil.which("tesseract"), reason="Tesseract OCR is required")


def trailer_with_end_slate(path, seconds=40, slate_from=30):
    writer = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*"mp4v"), 10, (640, 360))
    rng = np.random.default_rng(3)
    for frame in range(seconds * 10):
        t = frame / 10
        if t < slate_from:
            # Moving, textured "film" footage.
            image = Image.fromarray(np.uint8(np.clip(rng.normal(110 + 40 * np.sin(t), 45, (360, 640, 3)), 0, 255)))
        else:
            image = Image.new("RGB", (640, 360), (12, 30, 70))
            draw = ImageDraw.Draw(image)
            draw.text((40, 120), "Subscribe for the latest", font=brand_font("body", 34), fill="white")
            draw.text((40, 170), "movie trailers", font=brand_font("body", 34), fill="white")
        writer.write(cv2.cvtColor(np.asarray(image), cv2.COLOR_RGB2BGR))
    writer.release()


def test_promo_end_slate_is_found_and_excluded_from_every_shot(tmp_path):
    path = tmp_path / "trailer.mp4"
    trailer_with_end_slate(path)
    cutoff = promo_cutoff(path, 40)
    assert cutoff is not None and 29.5 <= cutoff <= 32
    shots = analyse(path, tmp_path / "trailer.shots.json")
    assert shots and max(shot["end"] for shot in shots) <= cutoff


def test_trailer_without_promo_text_keeps_default_window(tmp_path):
    path = tmp_path / "plain.mp4"
    trailer_with_end_slate(path, seconds=30, slate_from=99)
    assert promo_cutoff(path, 30) is None
