"""Measure visible edit cadence and audio in an existing reference video.

This is an observation tool, not an automated taste or copyright judge. Cuts
are sampled and can be missed. Mixed soundtrack loudness is never labelled as
narrator-only or music-only loudness; supply separately isolated stems for that.
"""
from __future__ import annotations

import argparse
import io
import json
import math
import re
import statistics
import subprocess
from pathlib import Path

import cv2
import numpy as np
from PIL import Image, ImageDraw

from movie_pipeline.core import PipelineError, probe, save_json


def _number(value):
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    return round(number, 3) if math.isfinite(number) else None


def _run(command, timeout=600):
    try:
        result = subprocess.run(command, check=False, capture_output=True, text=True, timeout=timeout)
    except (OSError, subprocess.TimeoutExpired) as exc:
        raise PipelineError(f"Reference analysis could not run {command[0]} ({type(exc).__name__}).") from exc
    if result.returncode:
        raise PipelineError(f"Reference analysis failed in {command[0]}: {result.stderr[-1000:]}")
    return result.stderr


def analyze_mixed_audio(path: Path, media_seconds: float) -> dict:
    """Report measurements of the *whole* soundtrack, including any speech."""
    streams = probe(path).get("streams", [])
    if not any(stream.get("codec_type") == "audio" for stream in streams):
        return {"available": False, "reason": "No audio stream."}
    args = ["ffmpeg", "-nostats", "-hide_banner", "-i", str(path), "-map", "0:a:0",
            "-af", "ebur128=peak=true", "-f", "null", "-"]
    log = _run(args)
    summary = log.rsplit("Summary:", 1)[-1]
    loudness = re.search(r"\bI:\s*(-?(?:\d+(?:\.\d+)?|inf))\s*LUFS", summary, re.I)
    true_peak = re.search(r"\bPeak:\s*(-?(?:\d+(?:\.\d+)?|inf))\s*dBFS", summary, re.I)
    silence_log = _run(["ffmpeg", "-nostats", "-hide_banner", "-i", str(path),
                        "-map", "0:a:0", "-af", "silencedetect=noise=-45dB:d=0.3",
                        "-f", "null", "-"])
    silence = []
    pending = None
    for line in silence_log.splitlines():
        start = re.search(r"silence_start:\s*([\d.]+)", line)
        end = re.search(r"silence_end:\s*([\d.]+)", line)
        if start:
            pending = float(start.group(1))
        if end and pending is not None:
            stop = min(media_seconds, float(end.group(1)))
            if stop > pending:
                silence.append({"start": round(pending, 3), "end": round(stop, 3)})
            pending = None
    if pending is not None and pending < media_seconds:
        silence.append({"start": round(pending, 3), "end": round(media_seconds, 3)})
    return {
        "available": True,
        "scope": "entire mixed audio stream; not separate voice or music levels",
        "integrated_lufs": _number(loudness.group(1)) if loudness else None,
        "true_peak_dbfs": _number(true_peak.group(1)) if true_peak else None,
        "silence_threshold_db": -45,
        "silence_minimum_seconds": 0.3,
        "quiet_or_silent_intervals": silence,
        "limitations": ["Silence detection measures the whole mix; quiet narration may coexist with music.",
                        "Loudness and peak measurements cannot establish clarity, quality, or absence of noise."],
    }


def _difference(before: np.ndarray, after: np.ndarray) -> float:
    return float(np.mean(cv2.absdiff(before, after))) / 255.0


def analyze_visuals(path: Path, media_seconds: float, out: Path, sample_seconds: float = 0.5) -> dict:
    """Sample frames and estimate abrupt boundaries. Fast shots/fades may be missed."""
    if not math.isfinite(sample_seconds) or not 0.1 <= sample_seconds <= 5:
        raise ValueError("sample interval must be between 0.1 and 5 seconds")
    # Decode sequentially once, scaled before reaching Python. Seeking to every
    # half-second repeatedly decodes old GOPs and is much slower for long videos.
    command = ["ffmpeg", "-hide_banner", "-loglevel", "error", "-i", str(path),
               "-map", "0:v:0", "-vf", f"fps={1/sample_seconds:.8f},scale=128:72:flags=area,format=gray",
               "-pix_fmt", "gray", "-f", "rawvideo", "pipe:1"]
    try:
        result = subprocess.run(command, capture_output=True, timeout=900, check=False)
    except (OSError, subprocess.TimeoutExpired) as exc:
        raise PipelineError(f"Video sampling failed ({type(exc).__name__}).") from exc
    if result.returncode:
        raise PipelineError("Video sampling failed: " + result.stderr.decode("utf-8", errors="replace")[-500:])
    stride = 128 * 72
    boundaries, frame_times, before = [], [], None
    for index in range(len(result.stdout) // stride):
        timestamp = index * sample_seconds
        if timestamp >= media_seconds:
            break
        frame_times.append(timestamp)
        gray = np.frombuffer(result.stdout, dtype=np.uint8, count=stride, offset=index * stride).reshape((72, 128))
        if before is not None and _difference(before, gray) >= 0.27:
            if not boundaries or timestamp - boundaries[-1] >= 0.75:
                boundaries.append(round(timestamp, 3))
        before = gray
    if not frame_times:
        raise PipelineError("No frames could be decoded from the reference video.")
    edges = [0.0, *boundaries, media_seconds]
    lengths = [b-a for a,b in zip(edges, edges[1:]) if b > a]
    review_frames = _contact_sheet(path, media_seconds, out / "contact-sheet.jpg")
    return {
        "sample_interval_seconds": sample_seconds,
        "frames_decoded": len(frame_times),
        "approximate_abrupt_boundaries_seconds": boundaries,
        "estimated_segment_median_seconds": _number(statistics.median(lengths)) if lengths else None,
        "estimated_segment_count": len(lengths),
        "contact_sheet": review_frames,
        "limitations": ["Frame sampling misses some fast cuts and may mistake motion for a cut.",
                        "Fades, freeze frames and meaning of scenes require additional inspection.",
                        "These are apparent visual boundaries, not proof of an editor's original cuts."],
    }


def _contact_sheet(path: Path, seconds: float, output: Path) -> list[float]:
    times = [min(max(0, seconds-0.05), seconds * (i+0.5) / 12) for i in range(12)]
    sheet = Image.new("RGB", (960, 756), "#111114")
    draw = ImageDraw.Draw(sheet)
    cap = cv2.VideoCapture(str(path))
    accepted = []
    try:
        for index, stamp in enumerate(times):
            cap.set(cv2.CAP_PROP_POS_MSEC, stamp * 1000)
            ok, frame = cap.read()
            if ok:
                image = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
            else:
                # OpenCV may lack a codec that the system FFmpeg can decode.
                # Fall back per frame rather than silently omitting the sheet.
                command = ["ffmpeg", "-hide_banner", "-loglevel", "error", "-ss", str(stamp),
                           "-i", str(path), "-frames:v", "1", "-f", "image2pipe",
                           "-vcodec", "mjpeg", "pipe:1"]
                try:
                    extracted = subprocess.run(command, capture_output=True, timeout=30, check=False)
                    if extracted.returncode or not extracted.stdout:
                        continue
                    with Image.open(io.BytesIO(extracted.stdout)) as loaded:
                        image = loaded.convert("RGB")
                except (OSError, subprocess.TimeoutExpired, ValueError):
                    continue
            image.thumbnail((320, 180), Image.Resampling.LANCZOS)
            x, y = (index % 3) * 320, (index // 3) * 189
            sheet.paste(image, (x + (320-image.width)//2, y + (180-image.height)//2))
            draw.text((x+4, y+171), f"{stamp:.1f}s", fill="white", stroke_width=1, stroke_fill="black")
            accepted.append(round(stamp, 3))
    finally:
        cap.release()
    if accepted:
        output.parent.mkdir(parents=True, exist_ok=True)
        sheet.save(output, quality=85)
    return accepted


def transcript_from_file(path: Path) -> dict:
    """Read user-supplied SRT/VTT cues or an untimed plain-text transcript."""
    try:
        text = path.read_text(encoding="utf-8-sig")
    except (OSError, UnicodeError) as exc:
        raise PipelineError("The supplied transcript could not be read as UTF-8.") from exc
    if not text.strip():
        raise PipelineError("The supplied transcript is empty.")
    if "-->" not in text:
        return {"available": True, "source": "provided untimed transcript",
                "segments": [{"start": None, "end": None, "text": text.strip()}],
                "limitations": ["Provided transcript was not aligned or checked against the audio.",
                                "Speaker identity is not inferred."]}
    def seconds(value):
        parts = value.replace(",", ".").split(":")
        return sum(float(item) * (60 ** index) for index, item in enumerate(reversed(parts)))
    timestamp = r"(?:\d{1,2}:)?\d{2}:\d{2}[.,]\d{1,3}"
    cues = []
    blocks = re.split(r"\n\s*\n", text.replace("\r\n", "\n").replace("\r", "\n"))
    for block in blocks:
        lines = block.strip().splitlines()
        for index, line in enumerate(lines):
            found = re.search(rf"({timestamp})\s*-->\s*({timestamp})", line)
            if not found:
                continue
            start, end = seconds(found.group(1)), seconds(found.group(2))
            caption = " ".join(re.sub(r"<[^>]*>", "", item).strip() for item in lines[index+1:]).strip()
            if caption and end > start:
                cues.append({"start": round(start, 3), "end": round(end, 3), "text": caption})
            break
    if not cues:
        raise PipelineError("No valid timed cues were found in the supplied SRT/VTT file.")
    return {"available": True, "source": "provided captions", "segments": cues,
            "limitations": ["Provided captions may not match every spoken word.",
                            "Captions may include narration or other dialogue; speaker identity is not inferred."]}



def summarize_transcript(transcript: dict, media_seconds: float) -> dict:
    """Summarize speech timestamps without inferring speakers or quality."""
    if not transcript.get("available"):
        return {"available": False, "reason": "No transcript available."}
    segments = transcript.get("segments", [])
    words = sum(len(re.findall(r"\b[\w']+\b", str(segment.get("text", ""))))
                for segment in segments if isinstance(segment, dict))
    intervals = []
    for segment in segments:
        if not isinstance(segment, dict):
            continue
        start, end = segment.get("start"), segment.get("end")
        if (isinstance(start, bool) or isinstance(end, bool)
                or not isinstance(start, (int, float)) or not isinstance(end, (int, float))
                or not math.isfinite(start) or not math.isfinite(end)):
            continue
        start, end = max(0.0, float(start)), min(float(media_seconds), float(end))
        if end > start:
            intervals.append((start, end))
    if not intervals:
        return {"available": True, "word_count": words, "timed": False,
                "limitations": ["Untimed text does not establish when speech occurs."]}
    merged = []
    for start, end in sorted(intervals):
        if merged and start <= merged[-1][1]:
            merged[-1] = (merged[-1][0], max(end, merged[-1][1]))
        else:
            merged.append((start, end))
    spoken = sum(end - start for start, end in merged)
    gaps = []
    previous = 0.0
    for start, end in merged:
        if start - previous >= 2.0:
            gaps.append({"start": round(previous, 3), "end": round(start, 3)})
        previous = end
    if media_seconds - previous >= 2.0:
        gaps.append({"start": round(previous, 3), "end": round(media_seconds, 3)})
    return {"available": True, "timed": True, "word_count": words,
            "first_caption_seconds": round(merged[0][0], 3),
            "captioned_speech_seconds": round(spoken, 3),
            "captioned_fraction": round(spoken / media_seconds, 3),
            "approximate_words_per_captioned_minute": round(words * 60 / spoken, 1),
            "uncaptioned_intervals_at_least_two_seconds": gaps,
            "limitations": [
                "Speech may include movie dialogue, narration, or other speakers.",
                "ASR and captions can omit or mis-time speech; gaps are not confirmed silence.",
                "Word rate and caption coverage depend on the source and do not measure delivery quality.",
            ]}


def _comparable(reference, production):
    """Only compare metrics measured in both; do not imply a causal or quality verdict."""
    values = {
        "duration_seconds": (reference["duration_seconds"], production["duration_seconds"], "seconds"),
        "estimated_visual_interval_seconds": (reference["visual"]["estimated_segment_median_seconds"],
                                              production["visual"]["estimated_segment_median_seconds"], "seconds"),
        "mixed_integrated_loudness": (reference["mixed_audio"].get("integrated_lufs"),
                                      production["mixed_audio"].get("integrated_lufs"), "LUFS"),
        "mixed_true_peak": (reference["mixed_audio"].get("true_peak_dbfs"),
                             production["mixed_audio"].get("true_peak_dbfs"), "dBFS"),
    }
    for kind in ("narrator_audio", "music_audio"):
        a, b = reference[kind], production[kind]
        if a.get("available") and b.get("available"):
            values[kind + "_integrated_loudness"] = (a.get("integrated_lufs"), b.get("integrated_lufs"), "LUFS")
    for label, index in (("reference", reference), ("production", production)):
        transcript = index.get("transcript", {})
        if transcript.get("available"):
            count = len(re.findall(r"\b[\w']+\b", " ".join(seg["text"] for seg in transcript["segments"])))
            values.setdefault("transcribed_word_count", [None, None, "words"])[0 if label == "reference" else 1] = count
    metrics = {}
    for name, (before, after, units) in values.items():
        if before is not None and after is not None:
            metrics[name] = {"reference": before, "production": after,
                             "difference": round(after - before, 3), "unit": units}
    return {"schema_version": 1, "metrics": metrics,
            "limitations": [
                "Visual boundaries are sampled estimates and cannot establish editorial quality.",
                "Mixed-audio levels do not measure narrator/music balance.",
                "Transcript counts may include film dialogue or miss speech.",
                "Differences describe these two specific videos and do not establish an optimal setting.",
                "No production settings are changed by this comparison.",
            ]}


def compare(reference: dict, production: dict, output: Path) -> dict:
    result = _comparable(reference, production)
    output.mkdir(parents=True, exist_ok=True)
    save_json(output / "comparison.json", result)
    lines = ["# Reference vs production", "",
             "Observed measurements, not quality ratings or automatic editing instructions.", "",
             "| Measurement | Reference | Production | Difference | Unit |",
             "|---|---:|---:|---:|---|"]
    for name, values in result["metrics"].items():
        lines.append(f"| {name.replace('_', ' ')} | {values['reference']} | {values['production']} | "
                     f"{values['difference']} | {values['unit']} |")
    lines.extend(["", "## Limits", ""])
    lines.extend("- " + value for value in result["limitations"])
    (output / "comparison.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    return result


def transcribe_audio(path: Path) -> dict:
    """Optional CPU ASR; speech may include both narrator and film dialogue."""
    try:
        from faster_whisper import WhisperModel
        model = WhisperModel("base.en", device="cpu", compute_type="int8", cpu_threads=2)
        segments, _ = model.transcribe(str(path), beam_size=3, vad_filter=True)
        spoken = [{"start": round(segment.start, 3), "end": round(segment.end, 3),
                   "text": segment.text.strip()} for segment in segments if segment.text.strip()]
        return {"available": True, "source": "automatic CPU ASR", "segments": spoken,
                "limitations": ["Automatic transcript may miss or mishear words.",
                                "Speech may be narration, movie dialogue, or another speaker; no speaker identity is inferred."]}
    except Exception as exc:
        return {"available": False, "segments": [], "reason": type(exc).__name__,
                "limitations": ["Automatic transcription was unavailable; visual/audio measurements remain available."]}


def analyze(video: Path, output: Path, *, transcribe: bool = False, voice_stem: Path | None = None,
            music_stem: Path | None = None, transcript_path: Path | None = None, sample_seconds: float = 0.5) -> dict:
    video, output = Path(video), Path(output)
    if not video.is_file():
        raise PipelineError("Reference video does not exist.")
    metadata = probe(video)
    seconds = _number(metadata.get("format", {}).get("duration"))
    if not seconds or seconds <= 0 or seconds > 1800:
        raise PipelineError("Reference video must have a valid duration up to 30 minutes.")
    output.mkdir(parents=True, exist_ok=True)
    report = {
        "schema_version": 1, "reference_file": video.name, "duration_seconds": seconds,
        "visual": analyze_visuals(video, seconds, output, sample_seconds),
        "mixed_audio": analyze_mixed_audio(video, seconds),
        "narrator_audio": {"available": False, "reason": "No isolated narration stem supplied."},
        "music_audio": {"available": False, "reason": "No isolated music stem supplied."},
        "transcript": {"available": False, "reason": "Transcription not requested.", "segments": []},
        "limitations": ["Video measurements are observations, not an automatic assessment of taste or quality.",
                        "The original video is not modified. Reference results do not alter production settings."],
    }
    for name, stem in (("narrator_audio", voice_stem), ("music_audio", music_stem)):
        if stem is not None:
            if not Path(stem).is_file():
                raise PipelineError(f"Optional {name} stem does not exist.")
            stem_duration = _number(probe(Path(stem)).get("format", {}).get("duration"))
            if not stem_duration or stem_duration <= 0:
                raise PipelineError(f"Optional {name} stem has no valid duration.")
            report[name] = analyze_mixed_audio(Path(stem), stem_duration)
            report[name]["scope"] = "provided isolated stem; source separation quality is not verified"
    if transcript_path is not None:
        report["transcript"] = transcript_from_file(Path(transcript_path))
    elif transcribe and report["mixed_audio"]["available"]:
        report["transcript"] = transcribe_audio(video)
    report["transcript_summary"] = summarize_transcript(report["transcript"], seconds)
    save_json(output / "reference-analysis.json", report)
    write_report(output / "reference-analysis.md", report)
    return report


def write_report(path: Path, report: dict):
    visual, sound, transcript = report["visual"], report["mixed_audio"], report["transcript"]
    lines = ["# Reference video analysis", "",
             f"Duration: {report['duration_seconds']:.1f} seconds.",
             f"Apparent abrupt visual boundaries: {len(visual['approximate_abrupt_boundaries_seconds'])}.",
             f"Estimated median interval: {visual['estimated_segment_median_seconds']} seconds.",
             f"Boundaries (seconds): {', '.join(map(str, visual['approximate_abrupt_boundaries_seconds'])) or 'none detected'}.",
             "", "## Audio (whole mixed soundtrack)", "",
             f"Integrated loudness: {sound.get('integrated_lufs')} LUFS.",
             f"True peak: {sound.get('true_peak_dbfs')} dBFS.",
             "Narrator and music are not separated unless isolated stems are supplied.",
             "", "## Approximate transcript", ""]
    if transcript.get("available"):
        lines.extend(f"- {segment['start']:.1f}–{segment['end']:.1f}s: {segment['text']}" if segment["start"] is not None else f"- {segment['text']}" for segment in transcript["segments"])
    else:
        lines.append(str(transcript.get("reason", "Unavailable.")))
    stats = report.get("transcript_summary", {})
    if stats.get("available"):
        lines.extend(["", "## Approximate speech timing", "",
                      f"Words in transcript: {stats.get('word_count')}."])
        if stats.get("timed"):
            lines.extend([
                f"First caption: {stats['first_caption_seconds']} seconds.",
                f"Captioned fraction: {stats['captioned_fraction']:.1%}.",
                f"Approximate words per captioned minute: {stats['approximate_words_per_captioned_minute']}.",
                "Uncaptioned intervals of at least two seconds (not confirmed silence): "
                + (", ".join(f"{gap['start']}–{gap['end']}s"
                             for gap in stats['uncaptioned_intervals_at_least_two_seconds'][:30]) or "none"),
            ])
        lines.extend("- " + item for item in stats.get("limitations", []))
    lines.extend(["", "## Interpretation limits", ""])
    lines.extend("- " + item for item in report["limitations"] + visual["limitations"] +
                 sound.get("limitations", []) + transcript.get("limitations", []))
    lines.append("- Review the contact sheet and actual audio/video before making editorial changes.")
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main(argv=None):
    parser = argparse.ArgumentParser(description="Create an evidence-labelled report for one reference video.")
    parser.add_argument("--video", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--transcribe", action="store_true", help="Use optional CPU ASR; transcripts are approximate.")
    parser.add_argument("--voice-stem", type=Path, help="Optional genuinely isolated narrator audio.")
    parser.add_argument("--music-stem", type=Path, help="Optional genuinely isolated background music audio.")
    parser.add_argument("--transcript", type=Path, help="Optional SRT, VTT or untimed text transcript.")
    parser.add_argument("--compare-video", type=Path, help="Optional finished production to measure alongside the reference.")
    parser.add_argument("--comparison-voice-stem", type=Path)
    parser.add_argument("--comparison-music-stem", type=Path)
    parser.add_argument("--comparison-transcript", type=Path)
    parser.add_argument("--sample-seconds", type=float, default=0.5)
    args = parser.parse_args(argv)
    reference_dir = args.output_dir / "reference" if args.compare_video else args.output_dir
    report = analyze(args.video, reference_dir, transcribe=args.transcribe, voice_stem=args.voice_stem,
                     music_stem=args.music_stem, transcript_path=args.transcript, sample_seconds=args.sample_seconds)
    if args.compare_video:
        production = analyze(args.compare_video, args.output_dir / "production", transcribe=args.transcribe,
                             voice_stem=args.comparison_voice_stem, music_stem=args.comparison_music_stem,
                             transcript_path=args.comparison_transcript, sample_seconds=args.sample_seconds)
        compare(report, production, args.output_dir)
    print(f"Reference analysis complete: {report['duration_seconds']:.1f}s; report in {args.output_dir}.", flush=True)


if __name__ == "__main__":
    main()
