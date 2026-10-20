from __future__ import annotations

import math
import os
import wave
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageFont

from .core import PipelineError, duration, ffmpeg
from .timing import DurationSpec
from .video_settings import export_size

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


def subscribe_image(path: Path):
    """A readable, steady CTA with thumb, subscribe button, and bell."""
    image = Image.new("RGBA", (WIDTH, HEIGHT))
    draw = ImageDraw.Draw(image)
    draw.rounded_rectangle((120, 200, 1160, 495), radius=30, fill=(8,12,22,235))
    draw.text((185, 240), "LIKE  +  SUBSCRIBE", font=font(58), fill="white")
    draw.text((185, 340), "Your next good movie starts here.", font=font(29), fill="#ffbe37")
    draw.rounded_rectangle((185, 402, 620, 460), radius=16, fill="#e50914")
    draw.text((210, 411), "SUBSCRIBE", font=font(31), fill="white")
    # Bell outline and ringing arcs, drawn locally without icon-font dependencies.
    draw.arc((960, 288, 1035, 365), 180, 360, fill="#ffbe37", width=7)
    draw.line((960,326,954,370,1041,370,1035,326), fill="#ffbe37", width=7)
    draw.ellipse((989, 376, 1007, 394), fill="#ffbe37")
    draw.arc((935, 272, 1057, 388), 190, 250, fill="white", width=5)
    draw.arc((935, 272, 1057, 388), 290, 350, fill="white", width=5)
    draw.polygon([(1050,428),(1065,428),(1076,400),(1086,402),(1086,419),(1116,419),(1110,452),(1074,452),(1065,446),(1050,446)], fill="white")
    image.save(path)


def embed_chapters(path: Path, chapters: list[dict], total: float, work: Path):
    def escaped(text):
        return str(text).replace("\\", "\\\\").replace("\n", " ").replace("\r", " ").replace("=", "\\=").replace(";", "\\;").replace("#", "\\#")
    lines = [";FFMETADATA1"]
    for index, chapter in enumerate(chapters):
        start = round(chapter["start"]*1000)
        end = round((chapters[index+1]["start"] if index+1 < len(chapters) else total)*1000)
        if end > start:
            lines.extend(["[CHAPTER]", "TIMEBASE=1/1000", f"START={start}", f"END={end}", f"title={escaped(chapter['title'])}"])
    metadata = work / "chapters.ffmeta"
    metadata.write_text("\n".join(lines)+"\n")
    target = work / "chaptered.mp4"
    ffmpeg("-i", path, "-f", "ffmetadata", "-i", metadata, "-map", "0", "-map_metadata", "1", "-map_chapters", "1", "-c", "copy", "-movflags", "+faststart", target)
    target.replace(path)


def concat_file(paths: list[Path], listing: Path):
    # Paths are generated internally; apostrophe escaping follows FFmpeg concat syntax.
    listing.write_text("".join("file '" + str(p.resolve()).replace("'", "'\\''") + "'\n" for p in paths))


def speech_video(source: Path, audio: Path, chosen: list[dict], output: Path, label: Path | None, work: Path, *, label_seconds: float = 3.5):
    work.mkdir(parents=True, exist_ok=True)
    width, height = export_size()
    total_frames = math.ceil(duration(audio) * FPS)
    remaining = total_frames
    clips = []
    for index, shot in enumerate(chosen):
        frames = remaining if index == len(chosen)-1 else min(remaining, max(1, round(shot["duration"] * FPS)))
        if frames <= 0:
            continue
        out = work / f"shot-{index:03}.mp4"
        filters = f"scale={width}:{height}:force_original_aspect_ratio=increase:flags=lanczos,crop={width}:{height},setsar=1,fps={FPS}"
        hold_frames = min(max(0, round(shot.get("freeze", 0)*FPS)), max(0, frames-1))
        if shot.get("still"):
            filters += f",trim=end_frame=1,setpts=PTS-STARTPTS,tpad=stop_mode=clone:stop_duration={(frames+1)/FPS}"
        elif hold_frames:
            filters += f",trim=end_frame={frames-hold_frames},setpts=PTS-STARTPTS,tpad=stop_mode=clone:stop_duration={(hold_frames+1)/FPS}"
        inputs = (["-loop", "1", "-framerate", FPS, "-i", shot["source"]] if shot.get("image") else
                  ["-ss", shot["start"], "-i", shot.get("source", source)])
        if label and index == 0:
            inputs += ["-i", label]
            filters = f"[0:v]{filters}[v];[1:v]scale={width}:{height}[label];[v][label]overlay=0:0:enable='lt(t,{label_seconds})'[out]"
            options = ["-filter_complex", filters, "-map", "[out]"]
        else:
            options = ["-vf", filters]
        ffmpeg(*inputs, *options, "-frames:v", frames, "-an", "-c:v", "libx264", "-preset", "veryfast", "-crf", "18", "-pix_fmt", "yuv420p", "-threads", "2", out)
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
    width, height = export_size()
    ffmpeg("-ss", start, "-i", source, "-t", actual,
           "-vf", f"scale={width}:{height}:force_original_aspect_ratio=increase:flags=lanczos,crop={width}:{height},setsar=1,fps={FPS}",
           "-af", f"afade=t=in:d=0.08,afade=t=out:st={max(0,actual-0.15)}:d=0.15,loudnorm=I=-18:TP=-2:LRA=9,apad",
           "-c:v", "libx264", "-preset", "veryfast", "-crf", "18", "-pix_fmt", "yuv420p", "-threads", "2",
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


def finish(parts: list[Path], output: Path, work: Path, action: bool, spec: DurationSpec | None = None, *, ending_hold: float = 3.0) -> tuple[float, float]:
    spec = spec or DurationSpec()
    listing = work / "final-concat.txt"
    concat_file(parts, listing)
    combined = work / "combined.mp4"
    ffmpeg("-f", "concat", "-safe", "0", "-i", listing, "-c", "copy", combined)
    original = duration(combined)
    # Keep natural speed when the video already falls within the requested range.
    speed = 1.0 if spec.minimum <= original+ending_hold <= spec.maximum else original / max(1, spec.target-ending_hold)
    if not 0.85 <= speed <= 1.18:
        raise PipelineError(f"Narration produced {original/60:.1f} minutes; too far from the requested {spec.target/60:.1f} minutes for natural tempo adjustment after automatic rewriting.")
    spoken_duration = original / speed
    final_duration = spoken_duration + ending_hold
    music = work / "music.wav"
    music_bed(music, final_duration + 1, action)
    graph = (
        f"[0:v]setpts=(PTS-STARTPTS)/{speed},fps={FPS},tpad=stop_mode=clone:stop_duration={ending_hold},fade=t=out:st={final_duration-1.2}:d=1.2[v];"
        f"[0:a]atempo={speed},aresample=48000,apad=pad_dur={ending_hold},asplit=2[voice][control];"
        "[1:a]volume=0.13[bed];"
        "[bed][control]sidechaincompress=threshold=0.025:ratio=6:attack=15:release=300[duck];"
        f"[voice][duck]amix=inputs=2:duration=first:normalize=0,loudnorm=I=-16:TP=-1.5:LRA=10,afade=t=out:st={final_duration-1.2}:d=1.2[a]"
    )
    ffmpeg("-i", combined, "-i", music, "-filter_complex", graph, "-map", "[v]", "-map", "[a]",
           "-c:v", "libx264", "-preset", "fast", "-crf", "18", "-pix_fmt", "yuv420p", "-r", FPS,
           "-threads", "2", "-ar", "48000", "-ac", "2", "-c:a", "aac", "-b:a", "192k", "-movflags", "+faststart", "-t", final_duration, output)
    music.unlink(missing_ok=True)
    combined.unlink(missing_ok=True)
    return duration(output), speed


def thumbnails(trailers: list[Path], shots: list[list[dict]], phrases: list[str], output: Path, **options):
    """Compose redesigned thumbnails; API use is explicitly enabled by the production caller."""
    from .thumbnails import build_thumbnails
    return build_thumbnails(trailers, shots, phrases, output, **options)
