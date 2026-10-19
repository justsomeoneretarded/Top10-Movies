from __future__ import annotations

import math
import os
import wave
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageEnhance, ImageFont, ImageOps

from .core import PipelineError, duration, ffmpeg
from .shots import get_frame

WIDTH, HEIGHT, FPS = 1280, 720, 30


def font(size: int):
    paths = [os.getenv("FONT_PATH", ""), "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf", "/usr/share/fonts/truetype/liberation2/LiberationSans-Bold.ttf"]
    for path in paths:
        if path and Path(path).exists():
            return ImageFont.truetype(path, size)
    return ImageFont.load_default(size=size)


def wrapped(draw, text, face, width):
    lines = []
    current = ""
    for word in text.split():
        trial = (current + " " + word).strip()
        if draw.textbbox((0, 0), trial, font=face)[2] > width and current:
            lines.append(current)
            current = word
        else:
            current = trial
    if current:
        lines.append(current)
    return lines


def label_image(path: Path, title: str, rank: int | None, year: int | None):
    image = Image.new("RGBA", (WIDTH, HEIGHT), (0, 0, 0, 0))
    draw = ImageDraw.Draw(image)
    draw.rounded_rectangle((48, 520, 1190, 680), radius=18, fill=(8, 12, 22, 225))
    draw.rounded_rectangle((48, 520, 61, 680), radius=5, fill=(255, 190, 55, 255))
    prefix = f"#{rank}  " if rank is not None else ""
    label = prefix + title
    size = 43
    while size > 24 and draw.textbbox((0, 0), label, font=font(size))[2] > 1060:
        size -= 1
    lines = wrapped(draw, label, font(size), 1050)
    draw.text((85, 543), lines[0], font=font(size), fill="white")
    draw.text((87, 609), str(year) if year else "THE COUNTDOWN", font=font(24), fill=(255, 190, 55))
    image.save(path)


def concat_file(paths: list[Path], listing: Path):
    # Paths are generated internally; apostrophe escaping follows FFmpeg concat syntax.
    listing.write_text("".join("file '" + str(p.resolve()).replace("'", "'\\''") + "'\n" for p in paths))


def speech_video(source: Path, audio: Path, chosen: list[dict], output: Path, label: Path | None, work: Path):
    work.mkdir(parents=True, exist_ok=True)
    total_frames = math.ceil(duration(audio) * FPS)
    remaining = total_frames
    clips = []
    for index, shot in enumerate(chosen):
        frames = remaining if index == len(chosen)-1 else min(remaining, max(1, round(shot["duration"] * FPS)))
        if frames <= 0:
            continue
        out = work / f"shot-{index:03}.mp4"
        filters = f"scale={WIDTH}:{HEIGHT}:force_original_aspect_ratio=increase,crop={WIDTH}:{HEIGHT},setsar=1,fps={FPS}"
        inputs = ["-ss", shot["start"], "-i", source]
        if label and index == 0:
            inputs += ["-i", label]
            filters = f"[0:v]{filters}[v];[v][1:v]overlay=0:0:enable='lt(t,3.5)'[out]"
            options = ["-filter_complex", filters, "-map", "[out]"]
        else:
            options = ["-vf", filters]
        ffmpeg(*inputs, *options, "-frames:v", frames, "-an", "-c:v", "libx264", "-preset", "veryfast", "-crf", "20", "-pix_fmt", "yuv420p", "-threads", "2", out)
        clips.append(out)
        remaining -= frames
        if remaining <= 0:
            break
    if remaining > 0:
        raise PipelineError("Shot plan is shorter than its narration.")
    listing = work / "concat.txt"
    concat_file(clips, listing)
    silent = work / "silent.mp4"
    ffmpeg("-f", "concat", "-safe", "0", "-i", listing, "-c", "copy", silent)
    seconds = total_frames / FPS
    ffmpeg("-i", silent, "-i", audio, "-map", "0:v", "-map", "1:a", "-c:v", "copy", "-af", "apad", "-t", seconds, "-ar", "48000", "-ac", "2", "-c:a", "aac", "-b:a", "192k", output)
    for clip in clips:
        clip.unlink(missing_ok=True)
    silent.unlink(missing_ok=True)


def trailer_video(source: Path, start: float, seconds: float, output: Path):
    frames = max(FPS, round(seconds * FPS))
    actual = frames / FPS
    ffmpeg("-ss", start, "-i", source, "-t", actual,
           "-vf", f"scale={WIDTH}:{HEIGHT}:force_original_aspect_ratio=increase,crop={WIDTH}:{HEIGHT},setsar=1,fps={FPS}",
           "-af", f"afade=t=in:d=0.08,afade=t=out:st={max(0,actual-0.15)}:d=0.15,loudnorm=I=-18:TP=-2:LRA=9,apad",
           "-c:v", "libx264", "-preset", "veryfast", "-crf", "20", "-pix_fmt", "yuv420p", "-threads", "2",
           "-ar", "48000", "-ac", "2", "-c:a", "aac", "-b:a", "192k", output)


def music_bed(path: Path, seconds: float, action: bool):
    """Generate an original, understated instrumental pad; no licensed music downloads."""
    rate = 48000
    with wave.open(str(path), "wb") as stream:
        stream.setnchannels(2)
        stream.setsampwidth(2)
        stream.setframerate(rate)
        for block in range(math.ceil(seconds)):
            count = min(rate, round(seconds * rate) - block * rate)
            t = (np.arange(count) + block * rate) / rate
            root = [110.0, 87.307, 103.826, 97.999][int(block // 8) % 4]
            envelope = np.minimum(1, t / 3) * np.minimum(1, np.maximum(0, (seconds - t) / 3))
            chord = sum(np.sin(2*np.pi*root*ratio*t) / (j+2) for j, ratio in enumerate((1, 1.5, 2)))
            pulse = np.exp(-((t * (2 if action else 1)) % 1) * 12) * np.sin(2*np.pi*55*t)
            values = (0.12 * chord + 0.07 * pulse) * envelope
            stereo = np.column_stack((values, values * 0.97))
            stream.writeframes((np.clip(stereo, -1, 1) * 32767).astype("<i2").tobytes())


def finish(parts: list[Path], output: Path, work: Path, action: bool) -> tuple[float, float]:
    listing = work / "final-concat.txt"
    concat_file(parts, listing)
    combined = work / "combined.mp4"
    ffmpeg("-f", "concat", "-safe", "0", "-i", listing, "-c", "copy", combined)
    original = duration(combined)
    # Keep natural speed when the video already falls within the requested range.
    speed = 1.0 if 480 <= original <= 600 else original / 540
    if not 0.85 <= speed <= 1.18:
        raise PipelineError(f"Narration produced {original/60:.1f} minutes; too far from 8–10 minutes for natural tempo adjustment. Tune the script word target or voice rate.")
    final_duration = original / speed
    music = work / "music.wav"
    music_bed(music, final_duration + 1, action)
    graph = (
        f"[0:v]setpts=(PTS-STARTPTS)/{speed}[v];"
        f"[0:a]atempo={speed},aresample=48000,asplit=2[voice][control];"
        "[1:a]volume=0.13[bed];"
        "[bed][control]sidechaincompress=threshold=0.025:ratio=6:attack=15:release=300[duck];"
        "[voice][duck]amix=inputs=2:duration=first:normalize=0,loudnorm=I=-16:TP=-1.5:LRA=10[a]"
    )
    ffmpeg("-i", combined, "-i", music, "-filter_complex", graph, "-map", "[v]", "-map", "[a]",
           "-c:v", "libx264", "-preset", "fast", "-crf", "19", "-pix_fmt", "yuv420p", "-r", FPS,
           "-threads", "2", "-ar", "48000", "-ac", "2", "-c:a", "aac", "-b:a", "192k", "-movflags", "+faststart", "-t", final_duration, output)
    music.unlink(missing_ok=True)
    combined.unlink(missing_ok=True)
    return duration(output), speed


def thumbnails(trailers: list[Path], shots: list[list[dict]], phrases: list[str], output: Path):
    """Three frame-based compositions with deterministic, readable typography."""
    for variant in range(3):
        canvas = Image.new("RGB", (WIDTH, HEIGHT), "#0b1020")
        indices = [(variant + j) % len(trailers) for j in range(2 if variant != 1 else 3)]
        panel_width = math.ceil(WIDTH / len(indices))
        for panel, index in enumerate(indices):
            ranked = sorted(shots[index], key=lambda s: s["motion"], reverse=True)
            shot = ranked[min(variant, len(ranked)-1)]
            frame = get_frame(trailers[index], (shot["start"] + shot["end"])/2)
            frame = ImageOps.fit(frame, (panel_width, HEIGHT), method=Image.Resampling.LANCZOS)
            frame = ImageEnhance.Contrast(frame).enhance(1.15)
            canvas.paste(frame, (panel*panel_width, 0))
        shade = Image.new("RGBA", (WIDTH, HEIGHT), (0, 0, 0, 0))
        pixels = np.zeros((HEIGHT, WIDTH, 4), dtype=np.uint8)
        pixels[:, :, 3] = np.linspace(0, 225, HEIGHT).astype(np.uint8)[:, None]
        shade = Image.fromarray(pixels)
        canvas = Image.alpha_composite(canvas.convert("RGBA"), shade)
        draw = ImageDraw.Draw(canvas)
        phrase = phrases[variant].upper()
        size = 100
        while size > 38:
            lines = wrapped(draw, phrase, font(size), 1150)
            if len(lines) <= 2:
                break
            size -= 2
        y = HEIGHT - len(lines) * (size + 12) - 55
        for line in lines:
            draw.text((55, y), line, font=font(size), fill=(255, 215, 65) if variant != 1 else "white", stroke_width=5, stroke_fill="black")
            y += size + 12
        draw.rounded_rectangle((45, 35, 220, 100), radius=12, fill="#f5bd37")
        draw.text((65, 43), f"TOP {len(trailers)}", font=font(32), fill="#111827")
        target = output / f"thumbnail-{variant+1:02}.jpg"
        canvas.convert("RGB").save(target, quality=91, optimize=True)
        if target.stat().st_size >= 2_000_000:
            canvas.convert("RGB").save(target, quality=80, optimize=True)
