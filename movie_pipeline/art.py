"""Original background scenes for composite thumbnails (FLUX.2 [klein] 4B on the CPU).

Every video gets four new scenes made for its genre and setting (a burning city for an
action list, a foggy forest for horror, a desert for a sci-fi epic...). Scenes contain no
people: the real actors are composited in from official stills, so no one's likeness is
synthesized. Generation runs in parallel workflow jobs through stable-diffusion.cpp:
FLUX.2 [klein] 4B (Apache 2.0) with its VAE and the Qwen3-4B text encoder (Apache 2.0).

Usage in a workflow job: python -m movie_pipeline.art <scene index 0-3>
"""
from __future__ import annotations

import os
import re
import subprocess
import sys
import time
from pathlib import Path

ART_DIR = Path(os.getenv("ART_DIR", "work/art"))
SCENES = ("hero", "faceoff", "ensemble", "service")
MODEL_FILES = {
    "diffusion": ("unsloth/FLUX.2-klein-4B-GGUF", "flux-2-klein-4b-Q4_K_M.gguf"),
    "vae": ("Comfy-Org/vae-text-encorder-for-flux-klein-4b", "split_files/vae/flux2-vae.safetensors"),
    "llm": ("unsloth/Qwen3-4B-Instruct-2507-GGUF", "Qwen3-4B-Instruct-2507-Q4_K_M.gguf"),
}
WIDTH, HEIGHT = 1024, 576

STYLE = ("cinematic movie poster background plate, photorealistic, dramatic lighting, high detail, 16:9 wide shot, "
         "depth and atmosphere, empty scene without any people, no text, no logos")

# Per genre, one scene for each design: hero (one actor), face-off (two actors at the edges),
# ensemble (three actors), service (the number one film). Centres stay open for the actors.
GENRE_SCENES = {
    "action": (
        "a massive fiery explosion erupting behind {setting}, flying sparks and burning debris, thick smoke, orange fire light against a teal night sky",
        "{setting} split by a wall of fire in the centre, sparks and embers in the air, one side lit orange and the other cold blue",
        "a wide street of {setting} at night with helicopters, searchlights, wrecked cars burning and smoke drifting",
        "{setting} at dusk with a dramatic sky, distant explosions on the horizon and glowing embers floating",
    ),
    "adventure": (
        "an epic view over {setting} at golden hour, sun rays through clouds, ancient ruins and dust in the air",
        "{setting} with a glowing sunset in the centre between two cliffs, birds in the sky, warm haze",
        "a vast lost world of {setting}, waterfalls, jungle mist and a giant stone temple in the distance",
        "{setting} under a dramatic sky, a long winding path toward a glowing horizon",
    ),
    "horror": (
        "a dark foggy {setting} at night, a single flickering light, long shadows, cold blue moonlight, eerie",
        "a decaying hallway of {setting} with a door glowing red at the end, dust in the air, deep shadows",
        "an abandoned {setting} at midnight under a blood-red moon, mist creeping along the ground",
        "a lonely house on a hill above {setting}, a stormy sky with lightning, dead trees, cold fog",
    ),
    "thriller": (
        "a rain-soaked {setting} at night, neon reflections on wet ground, harsh sodium lights, tension",
        "a dark interrogation room in {setting} with one hanging lamp in the centre, smoke in the light beam",
        "an aerial view of {setting} at night, police lights glowing below, rain and fog",
        "an empty parking garage in {setting}, flickering lights, long shadows",
    ),
    "scifi": (
        "a colossal spaceship over {setting}, glowing engines, alien sky with two moons, dust storm",
        "a vast alien planet surface in {setting} with a bright portal of light in the centre, floating rocks",
        "a futuristic megacity of {setting} at night, flying vehicles, holographic glow, rain",
        "the edge of space above {setting}, a glowing planet horizon, stars and nebula",
    ),
    "fantasy": (
        "a towering castle above {setting} at twilight, a dragon silhouette in the sky, magical glowing particles",
        "an enchanted forest of {setting} with a glowing magical portal in the centre, floating lights",
        "a vast fantasy battlefield in {setting}, banners, glowing runes in the sky, mist",
        "a mystical mountain kingdom of {setting} under aurora lights",
    ),
    "comedy": (
        "a bright colourful {setting} on a sunny day, playful energy, confetti in the air, vivid colours",
        "a chaotic party scene in {setting} with balloons and streamers, bright fun lighting, empty of guests",
        "a sunny suburban street of {setting} with colourful houses, blue sky, lively and cheerful",
        "a glowing city of {setting} at sunset, cheerful lights and colours",
    ),
    "family": (
        "a magical colourful {setting} full of wonder, glowing lights, soft clouds, whimsical and joyful",
        "a sunny meadow in {setting} with a rainbow in the centre, floating bubbles, bright colours",
        "a cosy fairytale village of {setting} at golden hour, warm windows, gentle snow of sparkles",
        "a starry night sky over {setting}, glowing fireflies, warm and magical",
    ),
    "romance": (
        "a romantic {setting} at golden hour, warm soft light, gentle bokeh lights, flower petals in the air",
        "a city bridge in {setting} at sunset with the sun glowing in the centre, warm haze, soft bokeh",
        "a rainy evening street in {setting} with warm café lights and reflections, soft and intimate",
        "a beach in {setting} at sunset, pink and orange sky, calm sea",
    ),
    "drama": (
        "a moody {setting} at dusk, soft window light, rain on glass, emotional atmosphere",
        "a quiet {setting} at dawn with a single beam of light in the centre, dust in the air",
        "a wide empty {setting} under a heavy sky, cinematic and emotional",
        "a city of {setting} at blue hour with warm windows glowing",
    ),
    "documentary": (
        "a striking real-world view of {setting}, natural dramatic light, documentary style",
        "{setting} at sunrise with a beam of light in the centre, natural colours",
        "an aerial view of {setting}, real and grand",
        "{setting} at dusk, natural and atmospheric",
    ),
}

SETTINGS = {
    "desert": "a desert of endless dunes", "space": "deep space", "planet": "an alien planet", "jungle": "a dense jungle",
    "forest": "a dark forest", "ocean": "the open ocean", "sea": "a stormy sea", "island": "a tropical island",
    "city": "a modern city", "new york": "New York city", "los angeles": "Los Angeles", "london": "London",
    "paris": "Paris", "tokyo": "Tokyo", "war": "a war-torn city", "military": "a military base", "prison": "a prison",
    "school": "a high school", "hospital": "a hospital", "small town": "a small town", "village": "a village",
    "mountain": "snowy mountains", "snow": "a frozen wasteland", "arctic": "the frozen arctic", "castle": "a castle",
    "train": "a speeding train", "airplane": "an airliner", "ship": "a ship at sea", "submarine": "the deep sea",
    "highway": "a desert highway", "car": "a city highway", "heist": "a bank vault", "casino": "a casino",
    "haunted house": "a haunted house", "cabin": "a cabin in the woods", "zombie": "a ruined city", "vampire": "a gothic town",
    "dinosaur": "a prehistoric jungle", "robot": "a futuristic city", "alien": "an alien world", "time travel": "a futuristic city",
    "post-apocalyptic future": "a post-apocalyptic wasteland", "dystopia": "a dystopian city", "treasure": "an ancient temple",
    "pirate": "a pirate cove", "western": "the wild west", "superhero": "a city skyline", "spy": "a European city",
}
DEFAULT_SETTING = {"action": "a city at night", "adventure": "a lost jungle valley", "horror": "an old house",
                   "thriller": "a city at night", "scifi": "an alien world", "fantasy": "a mythical land",
                   "comedy": "a lively town", "family": "a magical land", "romance": "a seaside town",
                   "drama": "a quiet town", "documentary": "the real world"}


def setting_for(film: dict, genre: str) -> str:
    """A visual place for the scene, from the film's TMDB keywords and premise."""
    text = " ".join([*map(str, film.get("keywords", [])), str(film.get("overview", ""))]).casefold()
    for word, place in SETTINGS.items():
        if re.search(rf"\b{re.escape(word)}\b", text):
            return place
    return DEFAULT_SETTING.get(genre, "a cinematic landscape")


def prompts(films: list[dict], genre: str) -> list[str]:
    """Four scene prompts: hero and service from number one's setting, the others from the list's top titles."""
    scenes = GENRE_SCENES.get(genre, GENRE_SCENES["drama"])
    sources = [films[0], films[min(1, len(films) - 1)], films[min(2, len(films) - 1)], films[0]] if films else [{}] * 4
    return [f"{scene.format(setting=setting_for(film, genre))}, {STYLE}" for scene, film in zip(scenes, sources)]


def scene_path(index: int) -> Path:
    return ART_DIR / f"{SCENES[index]}.png"


def available() -> dict[str, Path]:
    """Scenes generated for this run (name -> file)."""
    found = {}
    for index, name in enumerate(SCENES):
        path = scene_path(index)
        if path.exists() and path.stat().st_size > 50_000:
            found[name] = path
    return found


def download_models(target: Path) -> dict[str, Path]:
    from huggingface_hub import hf_hub_download
    return {name: Path(hf_hub_download(repo, filename, local_dir=str(target))) for name, (repo, filename) in MODEL_FILES.items()}


def generate(index: int, binary: Path, models: dict[str, Path], prompt: str, seed: int = 11) -> Path:
    out = scene_path(index)
    out.parent.mkdir(parents=True, exist_ok=True)
    command = [str(binary), "--diffusion-model", str(models["diffusion"]), "--vae", str(models["vae"]), "--llm", str(models["llm"]),
               "--cfg-scale", "1.0", "--steps", "4", "-W", str(WIDTH), "-H", str(HEIGHT), "-s", str(seed + index),
               "-t", str(os.cpu_count() or 4), "-p", prompt, "-o", str(out)]
    subprocess.run(command, check=True, timeout=60 * 60)
    return out


def main() -> int:
    from .cli import setup_run
    from .core import notice
    from .thumbnail_studio import detect_genre
    index = int(sys.argv[1])
    started = time.time()
    title, brief = os.getenv("VIDEO_TITLE", ""), os.getenv("VIDEO_BRIEF", "")
    ctx = setup_run(title, brief, os.getenv("MOVIE_LIST", ""), Path("work/jobs"), Path("output"), started)
    genre = detect_genre(ctx.films, title, brief)
    prompt = prompts(ctx.films, genre)[index]
    try:
        models = download_models(Path(os.getenv("ART_MODELS", "work/art-models")))
        binary = Path(os.getenv("SD_BINARY", "sdcpp/build/bin/sd-cli"))
        generate(index, binary, models, prompt)
    except Exception as exc:  # noqa: BLE001 -- thumbnails fall back to trailer-frame designs
        notice("Thumbnail scene unavailable", f"{SCENES[index]}: {type(exc).__name__}: {str(exc)[-400:]}", "warning")
        return 0
    notice("Thumbnail scene", f"{SCENES[index]} ({genre}) in {(time.time() - started) / 60:.1f} min: {prompt[:300]}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
