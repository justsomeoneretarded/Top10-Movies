"""Vertical YouTube Shorts cut from the finished video.

The strongest entries (number one, two and three) each become a 1080x1920 Short of
up to 58 seconds: the review's own reveal, narration, impact moment and trailer sound,
reframed over a blurred fill, with a big title on top and large word-by-word captions
(the current word highlighted). Everything is taken from the finished export, so the
mix, grade and timing match the main video.
"""
from __future__ import annotations

import re
from pathlib import Path

from .core import ffmpeg, save_json
from .render import ACCENT, FONT_DIR

MAX_SECONDS = 58.0
MIN_SECONDS = 20.0
WIDTH, HEIGHT = 1080, 1920


def _ass_time(seconds: float) -> str:
    seconds = max(0.0, seconds)
    return f"{int(seconds // 3600)}:{int(seconds // 60 % 60):02}:{seconds % 60:05.2f}"


def _escape(text: str) -> str:
    return text.replace("\\", "").replace("{", "(").replace("}", ")")


def window(chapters: list[dict], rank: int, cues: list[dict]) -> tuple[float, float] | None:
    """The review of ``rank`` on the final clock, ended on a finished sentence within the limit."""
    titles = [str(chapter.get("title", "")) for chapter in chapters]
    index = next((i for i, title in enumerate(titles) if title.startswith(f"#{rank} ")), None)
    if index is None:
        return None
    start = float(chapters[index]["start"])
    end = float(chapters[index + 1]["start"]) if index + 1 < len(chapters) else start + MAX_SECONDS
    limit = min(end, start + MAX_SECONDS)
    endings = [float(cue["end"]) for cue in cues if start < float(cue["end"]) <= limit
               and re.search(r"[.!?][\"'’”)]*$", str(cue.get("text", "")).strip())]
    stop = (max(endings) + 0.4) if endings else limit
    stop = min(stop, limit)
    return (start, stop) if stop - start >= MIN_SECONDS else None


def captions_ass(path: Path, cues: list[dict], start: float, end: float, heading: str, accent=ACCENT):
    """Word-by-word captions: three-word groups, the spoken word highlighted and enlarged."""
    highlight = "&H00{:02X}{:02X}{:02X}&".format(accent[2], accent[1], accent[0])
    lines = [
        "[Script Info]", "ScriptType: v4.00+", f"PlayResX: {WIDTH}", f"PlayResY: {HEIGHT}", "WrapStyle: 2", "",
        "[V4+ Styles]",
        "Format: Name, Fontname, Fontsize, PrimaryColour, SecondaryColour, OutlineColour, BackColour, Bold, Italic, "
        "Underline, StrikeOut, ScaleX, ScaleY, Spacing, Angle, BorderStyle, Outline, Shadow, Alignment, MarginL, MarginR, MarginV, Encoding",
        "Style: Cap,Montserrat,84,&H00FFFFFF,&H00FFFFFF,&H00000000,&H64000000,1,0,0,0,100,100,0,0,1,7,3,2,60,60,430,1",
        "Style: Head,Anton,132,&H00FFFFFF,&H00FFFFFF,&H00000000,&H64000000,0,0,0,0,100,100,1,0,1,7,3,8,50,50,150,1",
        "", "[Events]", "Format: Layer, Start, End, Style, Name, MarginL, MarginR, MarginV, Effect, Text",
        f"Dialogue: 1,{_ass_time(0)},{_ass_time(end - start)},Head,,0,0,0,,{_escape(heading)}",
    ]
    for cue in cues:
        cue_start, cue_end = float(cue["start"]) - start, float(cue["end"]) - start
        if cue_end <= 0 or cue_start >= end - start:
            continue
        words = str(cue.get("text", "")).split()
        if not words:
            continue
        weights = [max(2, len(word)) for word in words]
        total = sum(weights)
        span = max(0.1, cue_end - cue_start)
        clock = cue_start
        times = []
        for weight in weights:
            times.append((clock, clock + span * weight / total))
            clock += span * weight / total
        for group in range(0, len(words), 3):
            chunk = words[group:group + 3]
            for offset, word in enumerate(chunk):
                begin, finish = times[group + offset]
                if finish <= 0:
                    continue
                text = " ".join(
                    (f"{{\\c{highlight}\\fscx112\\fscy112}}{_escape(w)}{{\\c&H00FFFFFF&\\fscx100\\fscy100}}" if i == offset else _escape(w))
                    for i, w in enumerate(chunk))
                lines.append(f"Dialogue: 0,{_ass_time(max(0, begin))},{_ass_time(min(end - start, finish))},Cap,,0,0,0,,{text}")
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def render_short(video: Path, start: float, end: float, cues: list[dict], heading: str, output: Path, work: Path):
    work.mkdir(parents=True, exist_ok=True)
    ass = work / (output.stem + ".ass")
    captions_ass(ass, cues, start, end, heading)
    length = end - start
    subtitle = str(ass).replace("\\", "/").replace(":", "\\:").replace("'", "\\'")
    fonts = str(FONT_DIR).replace("\\", "/").replace(":", "\\:")
    graph = (f"[0:v]split[a][b];"
             f"[a]scale={WIDTH}:{HEIGHT}:force_original_aspect_ratio=increase,crop={WIDTH}:{HEIGHT},boxblur=28:4,eq=brightness=-0.18[bg];"
             f"[b]scale=-2:900,crop={WIDTH}:900[fg];"
             f"[bg][fg]overlay=0:(H-h)/2-140,subtitles='{subtitle}':fontsdir='{fonts}',"
             f"fade=t=out:st={max(0, length - 0.5):.2f}:d=0.5[v];"
             f"[0:a]afade=t=out:st={max(0, length - 0.5):.2f}:d=0.5[a]")
    ffmpeg("-ss", f"{start:.3f}", "-t", f"{length:.3f}", "-i", video, "-filter_complex", graph, "-map", "[v]", "-map", "[a]",
           "-c:v", "libx264", "-preset", "fast", "-crf", "20", "-pix_fmt", "yuv420p", "-r", 30,
           "-c:a", "aac", "-b:a", "192k", "-movflags", "+faststart", output)


def build_shorts(video: Path, chapters: list[dict], cues: list[dict], films: list[dict], title: str,
                 output: Path, work: Path, count: int = 3) -> list[dict]:
    """Write up to ``count`` Shorts plus a text file with a title and description for each."""
    output.mkdir(parents=True, exist_ok=True)
    made = []
    for rank in range(1, min(count, len(films)) + 1):
        span = window(chapters, rank, cues)
        if not span:
            continue
        film = films[rank - 1]
        name = re.sub(r"[^a-z0-9]+", "-", film["title"].casefold()).strip("-")[:40] or f"rank-{rank}"
        path = output / f"short-{rank}-{name}.mp4"
        heading = f"#{rank} {film['title'].upper()}"
        try:
            render_short(video, span[0], span[1], cues, heading, path, work)
        except Exception as exc:  # noqa: BLE001 -- Shorts are a bonus; the main video stands
            print(f"Short for {film['title']} skipped ({type(exc).__name__}).", flush=True)
            continue
        short_title = f"#{rank} {film['title']} ({film.get('year', '')}) — {title}"
        if len(short_title) > 90:
            short_title = f"#{rank} {film['title']} — {title}"[:90]
        (path.with_suffix(".txt")).write_text(
            f"{short_title} #shorts\n\nNumber {rank} from the full countdown \"{title}\". Watch the whole list on the channel.\n\n"
            f"#shorts #{re.sub(r'[^A-Za-z0-9]', '', film['title'])} #movierecommendations\n", encoding="utf-8")
        made.append({"file": path.name, "film": film["title"], "rank": rank, "seconds": round(span[1] - span[0], 1)})
    save_json(work / "shorts.json", made)
    print(f"Shorts: {len(made)} made ({', '.join(item['file'] for item in made)}).", flush=True)
    return made
