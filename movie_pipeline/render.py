from __future__ import annotations

import math
import os
import wave
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageFont

from .core import PipelineError, duration, ffmpeg, probe
from .timing import DurationSpec, final_tempo
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
    """A restrained, content-only lower third; intro/outro footage needs no banner."""
    image = Image.new("RGBA", (WIDTH, HEIGHT), (0, 0, 0, 0))
    if rank is None:
        image.save(path)
        return
    draw = ImageDraw.Draw(image)
    left, top, right, bottom = 56, 566, 1224, 666
    title_left, title_width = 184, right-184-28
    size = 34
    while size > 24 and draw.textlength(title, font=font(size)) > title_width:
        size -= 1
    lines = wrapped(draw, title, font(size), title_width)
    # Long official titles remain complete rather than being silently cut at line one.
    if len(lines) > 1:
        top = 542
        while size > 18 and len(wrapped(draw, title, font(size), title_width)) > 2:
            size -= 1
        lines = wrapped(draw, title, font(size), title_width)
    content_width = max((draw.textlength(line, font=font(size)) for line in lines), default=0)
    right = min(right, max(650, round(title_left+content_width+30)))
    # Compact glass-dark backplate, hairline border, no badges/glows/extra copy.
    draw.rounded_rectangle((left, top, right, bottom), radius=9,
                           fill=(10, 12, 16, 220), outline=(255, 255, 255, 42), width=1)
    draw.line((158, top+21, 158, bottom-21), fill=(255, 255, 255, 65), width=1)
    draw.text((84, (top+bottom)//2), f"{rank:02d}", font=font(42), anchor="lm", fill=(245, 246, 249))
    title_y = top+16
    for line in lines:
        draw.text((title_left, title_y), line, font=font(size), fill=(248, 249, 251))
        title_y += size+4
    if year is not None:
        draw.text((title_left, bottom-28), str(year), font=font(18), fill=(174, 181, 192))
    image.save(path)


def _label_graph(input_index: int, seconds: float, width: int, height: int) -> str:
    """One consistent eased slide/fade animation, measured on the whole clip clock."""
    end = max(.1, seconds)
    delay = min(.12, end*.10)
    enter = min(.42, end*.30)
    leave = min(.34, end*.25)
    out_start = max(delay+enter, end-leave)
    # PNG has a full transparent canvas, so moving by one frame width fully hides it.
    x = (f"if(lt(t,{delay}),-{width},"
         f"if(lt(t,{delay+enter}),-{width}*pow(1-(t-{delay})/{enter},3),"
         f"if(lt(t,{out_start}),0,-{width}*pow((t-{out_start})/{leave},3))))")
    return (
        f"[{input_index}:v]scale={width}:{height}:flags=lanczos,format=rgba,"
        f"fade=t=in:st={delay}:d={enter}:alpha=1,"
        f"fade=t=out:st={out_start}:d={leave}:alpha=1[rank];"
        f"[v][rank]overlay=x='{x}':y=0:enable='between(t,{delay},{end})':eof_action=pass[out]"
    )


def subscribe_image(path: Path):
    """A quiet footer over reaction footage, leaving the meme as the focal point."""
    image = Image.new("RGBA", (WIDTH, HEIGHT))
    draw = ImageDraw.Draw(image)
    draw.rounded_rectangle((365, 608, 915, 672), radius=9,
                           fill=(10, 12, 16, 225), outline=(255, 255, 255, 45), width=1)
    draw.text((398, 626), "LIKE  +  SUBSCRIBE", font=font(27), fill=(248, 249, 251))
    # Small bell line icon rather than a full-screen faux interface.
    draw.arc((857, 623, 881, 649), 180, 360, fill=(240, 242, 247), width=2)
    draw.line((857,636,854,652,884,652,881,636), fill=(240, 242, 247), width=2)
    draw.ellipse((866, 655, 872, 661), fill=(240, 242, 247))
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
            filters += f",trim=end_frame=1,setpts=PTS-STARTPTS,tpad=stop_mode=clone:stop={frames+1}"
        elif hold_frames:
            filters += f",trim=end_frame={frames-hold_frames},setpts=PTS-STARTPTS,tpad=stop_mode=clone:stop={hold_frames+1}"
        else:
            # A selected window can approach the source's last decoded frame. Hold
            # that last picture to cover narration instead of shortening the export.
            filters += f",setpts=PTS-STARTPTS,tpad=stop_mode=clone:stop={frames+1}"
        if shot.get("animation"):
            inputs = ["-stream_loop", "-1", "-i", shot["source"]]
        elif shot.get("image"):
            inputs = ["-loop", "1", "-framerate", FPS, "-i", shot["source"]]
        else:
            inputs = ["-ss", shot["start"], "-i", shot.get("source", source)]
        ffmpeg(*inputs, "-vf", filters+f",setpts=N/({FPS}*TB)", "-frames:v", frames, "-r", FPS,
               "-an", "-c:v", "libx264", "-preset", "veryfast", "-crf", "18",
               "-pix_fmt", "yuv420p", "-threads", "2", out)
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
    inputs = ["-i", silent, "-i", audio]
    if label:
        inputs += ["-loop", "1", "-framerate", FPS, "-i", label]
        graph = f"[0:v]setpts=PTS-STARTPTS[v];{_label_graph(2, min(label_seconds, seconds), width, height)}"
        video_options = ["-filter_complex", graph, "-map", "[out]", "-c:v", "libx264",
                         "-preset", "veryfast", "-crf", "18", "-pix_fmt", "yuv420p", "-r", FPS, "-threads", "2"]
    else:
        video_options = ["-map", "0:v", "-c:v", "copy"]
    # Lossless intermediate audio avoids an independent AAC encoder delay at every
    # paragraph/preview boundary. Only the finished upload is encoded to AAC.
    ffmpeg(*inputs, *video_options, "-map", "1:a", "-af", "asetpts=PTS-STARTPTS,apad",
           "-t", seconds, "-ar", "48000", "-ac", "2", "-c:a", "alac", output)
    for clip in clips:
        clip.unlink(missing_ok=True)
    silent.unlink(missing_ok=True)


def trailer_video(source: Path, start: float, seconds: float, output: Path, *,
                  label: Path | None = None, label_seconds: float = 3.5, muted: bool = False):
    """A narration-free preview, optionally carrying the animated rank introduction."""
    frames = max(FPS, round(seconds * FPS))
    actual = frames / FPS
    width, height = export_size()
    inputs = ["-ss", start, "-i", source]
    has_audio = not muted and any(s.get("codec_type") == "audio" for s in probe(source).get("streams", []))
    audio_index = 0
    label_index = 1
    if not has_audio:
        inputs += ["-f", "lavfi", "-i", "anullsrc=r=48000:cl=stereo"]
        audio_index = 1
        label_index = 2
    graph = (
        f"[0:v]scale={width}:{height}:force_original_aspect_ratio=increase:flags=lanczos,"
        f"crop={width}:{height},setsar=1,setpts=PTS-STARTPTS,fps={FPS},"
        f"tpad=stop_mode=clone:stop={frames},setpts=N/({FPS}*TB)[v];"
        f"[{audio_index}:a]asetpts=PTS-STARTPTS,afade=t=in:d=0.08,"
        f"afade=t=out:st={max(0,actual-0.15)}:d=0.15,loudnorm=I=-18:TP=-2:LRA=9,apad[a]"
    )
    video = "[v]"
    if label:
        inputs += ["-loop", "1", "-framerate", FPS, "-i", label]
        graph += ";"+_label_graph(label_index, min(label_seconds, actual), width, height)
        video = "[out]"
    ffmpeg(*inputs, "-filter_complex", graph, "-map", video, "-map", "[a]", "-t", actual,
           "-c:v", "libx264", "-preset", "veryfast", "-crf", "18", "-pix_fmt", "yuv420p", "-r", FPS, "-threads", "2",
           "-ar", "48000", "-ac", "2", "-c:a", "alac", output)


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
    speed = final_tempo(original, spec, ending_hold)
    if not spec.automatic and not spec.minimum <= original/speed+ending_hold <= spec.maximum:
        print("Keeping the complete video at a natural pace outside the soft runtime target.", flush=True)
    spoken_duration = original / speed
    final_duration = spoken_duration + ending_hold
    music = work / "music.wav"
    music_bed(music, final_duration + 1, action)
    graph = (
        f"[0:v]setpts=(PTS-STARTPTS)/{speed},fps={FPS},tpad=stop_mode=clone:stop_duration={ending_hold},fade=t=out:st={final_duration-1.2}:d=1.2[v];"
        f"[0:a]asetpts=PTS-STARTPTS,aresample=48000,atempo={speed},"
        f"asetpts=N/SR/TB,apad=pad_dur={ending_hold},asplit=2[voice][control];"
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
