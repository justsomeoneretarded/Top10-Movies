"""Readable synchronized like/subscribe/bell interaction, with brief UI sounds."""
from __future__ import annotations

import math
import re
import wave
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw

from .core import duration, ffmpeg
from .render import FPS, LOSSLESS_AUDIO, font


def action_times(text: str, cues: list[dict], seconds: float) -> list[float]:
    times = []
    for word, fallback in (("like", .15), ("subscribe", .4), ("bell", .7)):
        cue = next((cue for cue in cues if re.search(r"\b"+word+r"\b", cue["text"], re.IGNORECASE)), None)
        if cue:
            match = re.search(r"\b"+word+r"\b", cue["text"], re.IGNORECASE)
            # Captions are short phrases, so use the word's proportional location.
            when = cue["start"]+(cue["end"]-cue["start"])*match.start()/max(1,len(cue["text"]))
        else:
            when = seconds*fallback
        times.append(min(max(.1, when), max(.1, seconds-.4)))
    times[1] = max(times[0]+.15, times[1])
    times[2] = min(max(times[1]+.45, times[2]), max(.1, seconds-.2))
    return times


def interaction_frame(t: float, times: list[float]) -> Image.Image:
    image = Image.new("RGBA", (900, 130), (0, 0, 0, 0))
    draw = ImageDraw.Draw(image)
    draw.rounded_rectangle((4, 4, 896, 126), radius=24, fill=(14, 18, 26, 240), outline=(90, 99, 115, 150), width=2)
    labels = (("LIKE", "LIKED"), ("SUBSCRIBE", "SUBSCRIBED"), ("BELL", "BELL ON"))
    for i, (before, after) in enumerate(labels):
        active = t >= times[i]
        x = 22+i*294
        fill = ((39, 119, 239, 255) if i != 1 else (220, 38, 49, 255)) if active else (42, 48, 61, 255)
        # A small expanding highlight makes each click visible without shaking.
        pulse = max(0, 1-abs(t-times[i])/.22)
        inset = round(3*pulse)
        draw.rounded_rectangle((x-inset, 24-inset, x+270+inset, 105+inset), radius=14, fill=fill)
        draw.text((x+135, 64), after if active else before, font=font(25), anchor="mm", fill="white")
        if times[i]-.18 <= t <= times[i]+.22:
            px, py = x+235, 95
            draw.polygon([(px,py),(px-7,py-28),(px+17,py-9),(px+5,py-8),(px+10,py+4)], fill="white", outline="black")
    return image


def add_cta_animation(video: Path, text: str, cues: list[dict], work: Path):
    work.mkdir(parents=True, exist_ok=True)
    seconds = duration(video)
    times = action_times(text, cues, seconds)
    for frame in range(math.ceil(seconds*FPS)):
        interaction_frame(frame/FPS, times).save(work / f"ui-{frame:04}.png")
    samples = np.zeros((round(seconds*48000), 2), dtype=np.float32)
    for i, when in enumerate(times):
        length = .055 if i < 2 else .32
        t = np.arange(round(length*48000))/48000
        envelope = np.sin(np.pi*np.minimum(1, t/.005)/2)**2*np.exp(-t/(.012 if i < 2 else .09))
        tone = .035*envelope*(np.sin(2*np.pi*(850 if i < 2 else 1300)*t)
                             + .25*np.sin(2*np.pi*2100*t))
        start = round(when*48000)
        stop = min(len(samples), start+len(tone))
        if stop > start:
            samples[start:stop] += tone[:stop-start,None]
    sound = work / "interaction.wav"
    with wave.open(str(sound), "wb") as stream:
        stream.setnchannels(2)
        stream.setsampwidth(2)
        stream.setframerate(48000)
        stream.writeframes((samples*32767).astype("<i2").tobytes())
    temporary = work / "with-interaction.mp4"
    # Scale the overlay relative to the output, preserving readability at 4K.
    visible_start = max(0, times[0]-.4)
    visible_end = min(seconds, max(times[-1]+1.2, visible_start+3.5))
    graph = ("[1:v][0:v]scale2ref=w=iw*0.72:h=ow*130/900[ui][base];"
             f"[base][ui]overlay=(W-w)/2:H-h-H*0.045:shortest=1:enable='between(t,{visible_start},{visible_end})'[v]")
    mixed, silent = work / "mixed.wav", work / "interaction-video.mp4"
    try:
        # FFmpeg 6 amix duration=first can end early with unequal input packet
        # sizes. Both inputs cover the full edit: drain both and mix independently,
        # then mux it without -shortest or an output -t. Never conceal missing
        # samples by padding the result after the fact.
        audio_graph = (
            "[0:a]aresample=48000,aformat=sample_fmts=fltp:channel_layouts=stereo,asetpts=N/SR/TB[voice];"
            "[1:a]aresample=48000,aformat=sample_fmts=fltp:channel_layouts=stereo,asetpts=N/SR/TB[sfx];"
            "[voice][sfx]amix=inputs=2:duration=longest:normalize=0[a]"
        )
        ffmpeg("-i", video, "-i", sound, "-filter_complex", audio_graph,
               "-map", "[a]", "-vn", "-c:a", "pcm_s24le", mixed)
        ffmpeg("-i", video, "-framerate", FPS, "-i", work / "ui-%04d.png",
               "-filter_complex_threads", "1", "-filter_complex", graph,
               "-map", "[v]", "-an", "-c:v", "libx264", "-preset", "fast", "-crf", "18",
               "-pix_fmt", "yuv420p", "-threads", "2", "-frames:v", round(seconds*FPS), silent)
        ffmpeg("-i", silent, "-i", mixed, "-map", "0:v", "-map", "1:a",
               "-c:v", "copy", *LOSSLESS_AUDIO, temporary)
        from .render import _strict_audio_decode, validate_audio_video_duration
        _strict_audio_decode(temporary)
        validate_audio_video_duration(temporary, seconds)
        temporary.replace(video)
    finally:
        for image in work.glob("ui-*.png"):
            image.unlink(missing_ok=True)
        sound.unlink(missing_ok=True)
        mixed.unlink(missing_ok=True)
        silent.unlink(missing_ok=True)
        temporary.unlink(missing_ok=True)
