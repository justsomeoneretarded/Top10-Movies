import numpy as np
import pytest
from PIL import Image, ImageDraw

from movie_pipeline import art, thumbnail_composite as tc
import movie_pipeline.thumbnail_studio as studio


def test_prompts_follow_genre_and_setting_and_never_ask_for_people():
    films = [{"title": "Dune", "keywords": ["desert", "spice"], "overview": "A war on a desert planet."},
             {"title": "Jumanji", "keywords": ["jungle", "video game"]}]
    prompts = art.prompts(films, "scifi")
    assert len(prompts) == 4 and "desert" in prompts[0] and "jungle" in prompts[1]
    assert all("without any people" in p and "no text" in p for p in prompts)
    assert "fog" in art.prompts([{"title": "X"}], "horror")[0]
    assert art.setting_for({"overview": "nothing visual"}, "romance") == art.DEFAULT_SETTING["romance"]


def person(color=(200, 150, 120)):
    image = Image.new("RGBA", (420, 700), (0, 0, 0, 0))
    draw = ImageDraw.Draw(image)
    draw.ellipse((120, 30, 300, 250), fill=color + (255,))
    draw.rectangle((60, 230, 360, 700), fill=(60, 70, 90, 255))
    return image


def test_compose_lights_the_actor_from_the_scene_and_adds_effects():
    scene = Image.new("RGB", (1280, 720), (20, 25, 40))
    ImageDraw.Draw(scene).rectangle((0, 0, 300, 720), fill=(255, 150, 60))  # light on the left
    side, color = tc.scene_light(scene)
    assert side == "left" and color[0] > color[2]
    out = tc.compose(scene, [(person(), 0.6, 1.1, 1.0)], "action")
    assert out.size == (1280, 720)
    pixels = np.asarray(out, dtype=float)
    assert pixels[60:200, 640:820].mean() > pixels[60:200, 1100:1250].mean()  # the actor reads brighter than the dark side
    for genre in tc.EFFECTS:
        tc.compose(scene, [(person(), 0.5, 1.0, 1.0)], genre)


def test_headroom_keeps_the_head_inside_the_frame():
    scene = Image.new("RGB", (1280, 720), (0, 0, 0))
    out = np.asarray(tc.compose(scene, [(person((250, 250, 250)), 0.5, 1.3, 1.0)], "drama"), dtype=float)
    assert out[:30, 600:680].mean() < 40     # nothing pasted in the top strip
    assert out[60:120, 600:680].mean() > 120  # the head starts just below it


def test_studio_uses_scenes_when_they_exist(monkeypatch, tmp_path):
    scenes = {}
    for name in art.SCENES:
        path = tmp_path / f"{name}.png"
        image = Image.new("RGB", (1024, 576), (30, 30, 60))
        ImageDraw.Draw(image).ellipse((300, 100, 700, 500), fill=(255, 140, 40))
        noise = (np.random.default_rng(1).random((576, 1024, 3)) * 60).astype(np.uint8)
        Image.fromarray(np.clip(np.asarray(image) + noise, 0, 255).astype(np.uint8)).save(path)
        scenes[name] = path
    monkeypatch.setattr(art, "available", lambda: scenes)
    stills = {f"/{i}": Image.new("RGB", (1920, 1080), (40 + 30 * i, 90, 120)) for i in range(6)}
    for i, image in list(stills.items()):
        ImageDraw.Draw(image).ellipse((700, 200, 1100, 700), fill=(220, 170, 140))
        grain = (np.random.default_rng(int(i[1:])).random((1080, 1920, 3)) * 70).astype(np.int16)
        stills[i] = Image.fromarray(np.clip(np.asarray(image, dtype=np.int16) + grain, 0, 255).astype(np.uint8))
    monkeypatch.setattr(studio, "fetch_still", lambda path, cache: stills.get(path))
    monkeypatch.setattr("movie_pipeline.stills_trailer.fetch_original", lambda path, cache: None)
    monkeypatch.setattr(studio, "has_burned_text", lambda image: False)
    monkeypatch.setattr("movie_pipeline.scenes.rate_thumbnail", lambda image, *a, **k: None)
    monkeypatch.setattr(studio, "face_quality", lambda image: ((800, 260, 300, 380), 4.0, True))
    monkeypatch.setattr(studio.Cutter, "cut", lambda self, image: person().resize((image.width // 2, image.height)))
    films = [{"title": f"T{k}", "backdrops": [f"/{k * 2 + j}" for j in range(2)]} for k in range(3)]
    report = studio.build(films, [], [], ["Big words"], tmp_path / "out", tmp_path / "work")
    assert [r["style"] for r in report] == ["composite-hero", "composite-faceoff", "composite-ensemble",
                                            "composite-service", "composite-headline"]
    assert len(report[2]["films"]) == 3 and len(set(report[1]["films"])) == 2


def test_reports_with_numpy_values_are_saved(tmp_path):
    from movie_pipeline.core import save_json
    save_json(tmp_path / "r.json", {"good": np.bool_(True), "score": np.float64(2.5), "n": np.int64(3)})
    assert (tmp_path / "r.json").read_text().count("true") == 1
