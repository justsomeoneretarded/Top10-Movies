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
    monkeypatch.setattr("movie_pipeline.scenes.rate_thumbnail", lambda image, *args, **kw: None)
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
    assert [r["style"] for r in report] == ["hero-crop", "diagonal-panels", "signature-frame"]
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


def test_genre_comes_from_title_and_tmdb_genres():
    horror = [{"genres": ["Horror", "Mystery"]}, {"genres": ["Horror"]}, {"genres": ["Thriller"]}]
    assert studio.detect_genre(horror, "8 Movies That Will Ruin Your Sleep") == "horror"
    assert studio.detect_genre([{"genres": ["Action"]}] * 3, "Best movies") == "action"
    assert studio.detect_genre([{"genres": ["Drama"]}], "Top 5 scariest films") == "horror"
    assert studio.detect_genre([], "") == studio.DEFAULT_GENRE


def test_horror_grade_is_dark_cold_and_keeps_reds():
    image = Image.new("RGB", (640, 360), (150, 140, 110))
    ImageDraw.Draw(image).rectangle((260, 120, 380, 240), fill=(200, 20, 20))
    horror = np.asarray(studio.genre_grade(image, "horror"), dtype=float)
    action = np.asarray(studio.genre_grade(image, "blockbuster"), dtype=float)
    assert horror.mean() < action.mean() - 10                       # darker
    background = horror[40, 40]
    assert background.max() - background.min() < 40                 # colour drained from the background
    blood = horror[180, 320]
    assert blood[0] > blood[1] + 80 and blood[0] > blood[2] + 80     # reds survive


def test_logo_words_and_corner_bugs_reject_a_frame(monkeypatch):
    image = Image.new("RGB", (1280, 720), (30, 30, 30))
    monkeypatch.setattr("movie_pipeline.shots.frame_text", lambda img: "hulu" if img.size != (1280, 720) else "")
    assert studio.has_burned_text(image)                             # a lone corner logo
    monkeypatch.setattr("movie_pipeline.shots.frame_text", lambda img: "")
    assert not studio.has_burned_text(image)
    monkeypatch.setattr("movie_pipeline.shots.frame_text", lambda img: "Official Trailer" if img.size == (1280, 720) else "")
    assert studio.has_burned_text(image)


def test_scene_descriptions_pick_scary_shots(monkeypatch, tmp_path):
    shots = [{"start": i, "end": i + 1} for i in range(4)]
    descriptions = ["two friends smiling in a sunny park", "a terrified woman screams in a dark hallway",
                    "a man talking calmly at a desk", "a masked figure with a knife in the shadows"]
    monkeypatch.setattr("movie_pipeline.scenes.load", lambda trailer, count: {"descriptions": descriptions, "embeddings": [[0.0]] * 4})
    monkeypatch.setattr("movie_pipeline.scenes.embed", lambda texts: (_ for _ in ()).throw(RuntimeError("offline")))
    ranked = studio.genre_shots(tmp_path / "t.mp4", shots, studio.GENRE_PROFILES["horror"])
    assert {ranked[0][2], ranked[1][2]} == {descriptions[1], descriptions[3]}
    assert ranked[-1][1] == 0.0


def test_vision_logo_flag_excludes_a_candidate(monkeypatch):
    candidates = [studio.Candidate(still((80, 80, 80)), 0, "x", score=5), studio.Candidate(still((90, 90, 90)), 0, "y", score=4)]
    verdicts = iter([{"large_face": True, "emotion": 9, "action": 3, "genre_fit": 9, "logo": True, "score": 9},
                     {"large_face": True, "emotion": 8, "action": 3, "genre_fit": 8, "logo": False, "score": 8}])
    monkeypatch.setattr("movie_pipeline.scenes.rate_thumbnail", lambda image, *a, **k: next(verdicts))
    studio.rate_with_vision(candidates, profile=studio.GENRE_PROFILES["horror"])
    assert candidates[0].score < candidates[1].score
