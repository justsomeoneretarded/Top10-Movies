import numpy as np
import pytest
from PIL import Image, ImageDraw

import movie_pipeline.thumbnail_studio as studio


def still(color, bars=False):
    image = Image.new("RGB", (1920, 1080), color)
    draw = ImageDraw.Draw(image)
    for i in range(0, 1920, 60):
        draw.line((i, 0, i + 300, 1080), fill=tuple(min(255, c + 60) for c in color), width=12)
    if bars:
        draw.rectangle((0, 0, 1920, 140), fill=(0, 0, 0))
        draw.rectangle((0, 940, 1920, 1080), fill=(0, 0, 0))
    return image


@pytest.fixture
def offline(monkeypatch):
    images = {"/a": still((150, 60, 40), bars=True), "/b": still((40, 90, 150)), "/c": still((60, 140, 70)), "/d": still((140, 130, 40))}
    monkeypatch.setattr(studio, "fetch_still", lambda path, cache: images.get(path))
    monkeypatch.setattr(studio, "has_burned_text", lambda image: False)
    monkeypatch.setattr("movie_pipeline.scenes.rate_thumbnail", lambda image: None)
    monkeypatch.setattr(studio.Cutter, "cut", lambda self, image: None)


def films(service=None):
    first = {"title": "One", "backdrops": ["/a", "/b"]}
    if service:
        first["watch_availability"] = {"subscription": [service]}
    return [first, {"title": "Two", "backdrops": ["/c"]}, {"title": "Three", "backdrops": ["/d"]}]


def test_letterbox_bars_are_removed():
    trimmed = studio.trim_bars(still((150, 60, 40), bars=True))
    assert trimmed.height < 820 and trimmed.width > 1900
    assert studio.trim_bars(still((150, 60, 40))).size == (1920, 1080)


def test_three_distinct_text_free_thumbnails(offline, tmp_path, monkeypatch):
    monkeypatch.delenv("THUMBNAIL_TEXT", raising=False)
    report = studio.build(films(), [], [], ["Watch these"], tmp_path, tmp_path / "work")
    assert [r["style"] for r in report] == ["hero-crop", "diagonal-panels", "action-crop"]
    pixels = [np.asarray(Image.open(tmp_path / f"thumbnail-0{i}.jpg"), dtype=float) for i in (1, 2, 3)]
    assert all(p.shape == (720, 1280, 3) for p in pixels)
    assert np.abs(pixels[0] - pixels[2]).mean() > 10  # option 3 is a different picture
    assert report[2]["service_badge"] is None


def test_panels_have_white_diagonal_dividers(offline):
    candidates = [studio.Candidate(still(c), i, "x") for i, c in enumerate([(150, 60, 40), (40, 90, 150), (60, 140, 70)])]
    canvas = np.asarray(studio.panels(candidates))
    assert canvas[360, 425:430].min() > 230  # a divider crosses the middle near the first third


def test_service_badge_only_when_number_one_is_listed(offline, tmp_path):
    report = studio.build(films("Paramount+ Amazon Channel"), [], [], ["x"], tmp_path, tmp_path / "work")
    assert report[2]["service_badge"] == "Paramount Plus"
    assert studio.service_on_number_one(films()) is None


def test_grade_keeps_size_and_adds_punch():
    flat = Image.new("RGB", (640, 360), (110, 110, 110))
    ImageDraw.Draw(flat).rectangle((200, 100, 440, 260), fill=(150, 150, 150))
    graded = studio.cinematic_grade(flat)
    assert graded.size == flat.size
    assert studio.phone_check(graded) >= studio.phone_check(flat) * 0.9
