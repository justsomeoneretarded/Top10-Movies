from __future__ import annotations

import json
import math
import os
import shutil
import tempfile
import wave
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageFont

from .core import PipelineError, duration, ffmpeg, probe, run, save_json
from .timing import DurationSpec, final_tempo
from .video_settings import export_size

WIDTH, HEIGHT, FPS = 1280, 720, 30
LOSSLESS_AUDIO = ("-ar", "48000", "-ac", "2", "-channel_layout", "stereo",
                  "-c:a", "alac", "-sample_fmt", "s32p")
AUDIO_ALIGNMENT_TOLERANCE = .12
FOOTAGE_GRADE = "eq=contrast=1.015:saturation=1.015"


def _hold_motion(frames: int, live_frames: int, width: int, height: int, style: str) -> str:
    """Move only the frozen portion, by at most 3.5%, without altering time."""
    seconds = (frames-live_frames)/FPS
    if seconds < 1.5:
        return "null"  # A tiny timing hold should not turn into a quick zoom.
    amount = min(.035, seconds*.012)
    if live_frames:
        # A trailing freeze must start at the moving footage's existing framing.
        style = "zoom_in"
    progress = f"min(1,max(0,(on-{live_frames})/{max(1, frames-live_frames-1)}))"
    zoom = f"{1+amount}-{amount}*{progress}" if style == "zoom_out" else f"1+{amount}*{progress}"
    x, y = "(iw-iw/zoom)/2", "(ih-ih/zoom)/2"
    if style == "pan_left":
        zoom = str(1+amount)
        x = f"(iw-iw/zoom)*(0.7-0.4*{progress})"
    # With a trailing hold the moving footage stays at its original framing.
    if live_frames:
        zoom = f"if(lt(on,{live_frames}),1,{zoom})"
    return f"zoompan=z='{zoom}':x='{x}':y='{y}':d=1:s={width}x{height}:fps={FPS}"


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
        if not shot.get("image") and not shot.get("animation"):
            filters += ","+FOOTAGE_GRADE
        hold_frames = min(max(0, round(shot.get("freeze", 0)*FPS)), max(0, frames-1))
        if shot.get("still"):
            filters += f",trim=end_frame=1,setpts=PTS-STARTPTS,tpad=stop_mode=clone:stop={frames+1}"
        elif hold_frames:
            filters += f",trim=end_frame={frames-hold_frames},setpts=PTS-STARTPTS,tpad=stop_mode=clone:stop={hold_frames+1}"
        else:
            # A selected window can approach the source's last decoded frame. Hold
            # that last picture to cover narration instead of shortening the export.
            filters += f",setpts=PTS-STARTPTS,tpad=stop_mode=clone:stop={frames+1}"
        if (shot.get("still") or hold_frames) and not shot.get("animation"):
            live_frames = 0 if shot.get("still") else frames-hold_frames
            filters += ","+_hold_motion(frames, live_frames, width, height, shot.get("still_motion", "zoom_in"))
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
           "-t", seconds, *LOSSLESS_AUDIO, output)
    for clip in clips:
        clip.unlink(missing_ok=True)
    silent.unlink(missing_ok=True)


def trailer_video(source: Path, start: float, seconds: float, output: Path, *,
                  label: Path | None = None, label_seconds: float = 3.5, muted: bool = False,
                  still: bool = False, still_motion: str | None = None):
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
    hold = "trim=end_frame=1,setpts=PTS-STARTPTS," if still else ""
    motion = _hold_motion(frames, 0, width, height, still_motion)+"," if still and still_motion else ""
    graph = (
        f"[0:v]scale={width}:{height}:force_original_aspect_ratio=increase:flags=lanczos,"
        f"crop={width}:{height},setsar=1,{FOOTAGE_GRADE},setpts=PTS-STARTPTS,fps={FPS},"
        f"{hold}tpad=stop_mode=clone:stop={frames},{motion}setpts=N/({FPS}*TB)[v];"
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
           *LOSSLESS_AUDIO, output)


def _audio_stream(path: Path) -> dict:
    metadata = json.loads(run(["ffprobe", "-v", "error", "-select_streams", "a:0",
        "-show_entries", "stream=codec_name,sample_fmt,sample_rate,channels,channel_layout,bits_per_raw_sample,extradata,extradata_size",
        "-show_data", "-of", "json", path]))
    streams = metadata.get("streams", [])
    if not streams:
        raise PipelineError(f"No audio stream is available in {path.name}.")
    return streams[0]


def _canonical_audio(stream: dict) -> bool:
    # ALAC codec configuration is global in MP4. Identical codec names/rates
    # alone are insufficient: 16-bit packets cannot use a 24-bit stream header.
    return (stream.get("codec_name") == "alac" and stream.get("sample_fmt") == "s32p"
            and str(stream.get("sample_rate")) == "48000" and stream.get("channels") == 2
            and stream.get("channel_layout") == "stereo" and str(stream.get("bits_per_raw_sample")) == "24"
            and bool(stream.get("extradata")) and stream.get("extradata_size", 0) > 0)


def _normalized_audio_parts(parts: list[Path], work: Path) -> list[Path]:
    """Keep video bytes while making every audio codec configuration identical."""
    if not parts:
        raise PipelineError("Cannot assemble a video without media parts.")
    result, streams = [], []
    for index, value in enumerate(parts):
        path = Path(value)
        stream = _audio_stream(path)
        if not _canonical_audio(stream):
            normalized = work / f"part-{index:04}.mp4"
            ffmpeg("-xerror", "-err_detect", "explode", "-i", path, "-map", "0:v:0", "-map", "0:a:0",
                   "-c:v", "copy", "-af", "asetpts=PTS-STARTPTS", "-t", duration(path), *LOSSLESS_AUDIO, normalized)
            path, stream = normalized, _audio_stream(normalized)
        result.append(path)
        streams.append(stream)
    # A foreign ALAC encoder can use different predictor/configuration bytes
    # despite matching PCM parameters. Reencode that uncommon group uniformly.
    if len({stream.get("extradata") for stream in streams}) != 1:
        for index, path in enumerate(result):
            normalized = work / f"uniform-{index:04}.mp4"
            ffmpeg("-xerror", "-err_detect", "explode", "-i", path, "-map", "0:v:0", "-map", "0:a:0",
                   "-c:v", "copy", "-af", "asetpts=PTS-STARTPTS", "-t", duration(path), *LOSSLESS_AUDIO, normalized)
            result[index], streams[index] = normalized, _audio_stream(normalized)
    if not all(_canonical_audio(stream) for stream in streams) or len({stream.get("extradata") for stream in streams}) != 1:
        raise PipelineError("Intermediate audio codec configurations remain incompatible after lossless normalization.")
    return result


def _strict_audio_decode(path: Path):
    # Audio-only decoding is inexpensive even for long 4K exports. Treat packet
    # corruption as an error before a final encode can conceal it behind music.
    ffmpeg("-xerror", "-err_detect", "explode", "-i", path, "-map", "0:a:0", "-vn", "-f", "null", "-")


def validate_audio_video_duration(path: Path, expected_seconds: float | None = None) -> float:
    """Require complete, aligned audio/video streams, including the ending hold."""
    metadata = probe(path)
    video = next((stream for stream in metadata.get("streams", []) if stream.get("codec_type") == "video"), None)
    audio = next((stream for stream in metadata.get("streams", []) if stream.get("codec_type") == "audio"), None)
    if video is None or audio is None:
        raise PipelineError(f"Both audio and video streams are required in {path.name}.")
    try:
        video_start, audio_start = float(video.get("start_time", 0)), float(audio.get("start_time", 0))
        video_seconds, audio_seconds = float(video["duration"]), float(audio["duration"])
        total = float(metadata["format"]["duration"])
    except (KeyError, TypeError, ValueError) as exc:
        raise PipelineError(f"Cannot verify complete audio/video duration in {path.name}.") from exc
    values = (video_start, audio_start, video_seconds, audio_seconds, total)
    if not all(math.isfinite(value) for value in values) or min(video_seconds, audio_seconds, total) <= 0:
        raise PipelineError(f"Invalid audio/video duration in {path.name}.")
    tolerance = AUDIO_ALIGNMENT_TOLERANCE
    if (abs(video_start-audio_start) > tolerance or
            abs((video_start+video_seconds)-(audio_start+audio_seconds)) > tolerance):
        raise PipelineError(f"Audio/video timing is incomplete in {path.name}: video {video_seconds:.3f}s, audio {audio_seconds:.3f}s.")
    if expected_seconds is not None and (not math.isfinite(expected_seconds) or expected_seconds <= 0 or
            any(abs(value-expected_seconds) > tolerance for value in (video_seconds, audio_seconds, total))):
        raise PipelineError(f"Export streams do not cover the complete planned edit in {path.name}.")
    return total


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
    work.mkdir(parents=True, exist_ok=True)
    report = {"status": "pending", "canonical_audio": {"codec": "alac", "sample_format": "s32p",
              "bits_per_sample": 24, "sample_rate": 48000, "channels": 2, "channel_layout": "stereo"},
              "part_count": len(parts), "normalized_part_count": 0,
              "concatenated_strict_decode": "pending", "final_strict_decode": "pending"}

    def record(stage, **values):
        report.update(stage=stage, **values)
        save_json(work / "audio-assembly.json", report)

    def failed(exc):
        record(report["stage"], status="failed", error=type(exc).__name__, detail=str(exc)[-500:])

    record("normalizing")
    listing = work / "final-concat.txt"
    combined = work / "combined.mp4"
    normalization = Path(tempfile.mkdtemp(prefix="audio-normalized-", dir=work))
    try:
        normalized = _normalized_audio_parts(parts, normalization)
        record("concatenating", normalized_part_count=sum(Path(old) != new for old, new in zip(parts, normalized)))
        concat_file(normalized, listing)
        ffmpeg("-xerror", "-f", "concat", "-safe", "0", "-i", listing, "-c", "copy", combined)
    except Exception as exc:
        failed(exc)
        raise
    finally:
        shutil.rmtree(normalization, ignore_errors=True)
    try:
        record("checking-concatenated-audio", concatenated_strict_decode="running")
        _strict_audio_decode(combined)
        record("checking-concatenated-duration", concatenated_strict_decode="passed")
        original = validate_audio_video_duration(combined)
    except Exception as exc:
        failed(exc)
        raise
    speed = final_tempo(original, spec, ending_hold)
    if not spec.automatic and not spec.minimum <= original/speed+ending_hold <= spec.maximum:
        print("Keeping the complete video at a natural pace outside the soft runtime target.", flush=True)
    spoken_duration = original / speed
    final_duration = spoken_duration + ending_hold
    music = work / "music.wav"
    music_bed(music, final_duration + 1, action)
    graph = (
        f"[0:v]setpts=(PTS-STARTPTS)/{speed},fps={FPS},tpad=stop_mode=clone:stop_duration={ending_hold},fade=t=out:st={final_duration-1.2}:d=1.2[v];"
        # Honor timestamp-declared silence before converting the audio clock to
        # sample counts. Stream-looped AAC can contain small gaps between valid
        # packets; dropping those gaps would pull later speech forward. Strict
        # decoding and duration checks above already reject missing/corrupt audio.
        f"[0:a]asetpts=PTS-STARTPTS,aresample=48000:async=1:first_pts=0:min_hard_comp=0.0001,atempo={speed},"
        f"asetpts=N/SR/TB,apad=pad_dur={ending_hold},asplit=2[voice][control];"
        "[1:a]volume=0.13[bed];"
        "[bed][control]sidechaincompress=threshold=0.025:ratio=6:attack=15:release=300[duck];"
        f"[voice][duck]amix=inputs=2:duration=first:normalize=0,loudnorm=I=-16:TP=-1.5:LRA=10,afade=t=out:st={final_duration-1.2}:d=1.2[a]"
    )
    try:
        record("encoding", planned_duration=final_duration)
        ffmpeg("-xerror", "-err_detect", "explode", "-i", combined, "-i", music, "-filter_complex", graph, "-map", "[v]", "-map", "[a]",
               "-c:v", "libx264", "-preset", "fast", "-crf", "18", "-pix_fmt", "yuv420p", "-r", FPS,
               "-threads", "2", "-ar", "48000", "-ac", "2", "-c:a", "aac", "-b:a", "192k", "-movflags", "+faststart", "-t", final_duration, output)
        record("checking-final-audio", final_strict_decode="running")
        _strict_audio_decode(output)
        metadata = probe(output)
        record("checking-final-duration", final_strict_decode="passed", final_stream_durations={
            stream["codec_type"]: stream.get("duration") for stream in metadata.get("streams", [])
            if stream.get("codec_type") in ("audio", "video")})
        actual_duration = validate_audio_video_duration(output, final_duration)
    except Exception as exc:
        failed(exc)
        raise
    record("complete", status="passed", final_duration=actual_duration)
    music.unlink(missing_ok=True)
    combined.unlink(missing_ok=True)
    return actual_duration, speed


def thumbnails(trailers: list[Path], shots: list[list[dict]], phrases: list[str], output: Path, **options):
    """Compose redesigned thumbnails; API use is explicitly enabled by the production caller."""
    from .thumbnails import build_thumbnails
    return build_thumbnails(trailers, shots, phrases, output, **options)
