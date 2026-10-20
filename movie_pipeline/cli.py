from __future__ import annotations

import argparse
import gc
import json
import os
import re
import sys
from pathlib import Path

from .core import (
    PipelineError,
    duration,
    ffmpeg,
    key,
    narrator_from_brief,
    parse_movies,
    probe,
    save_json,
    voice_reference,
    write_srt,
)
from .timing import DurationSpec, requested_duration, trailer_budget
from .video_settings import export_size, require_disk, resolution_from_brief


def chapter_time(seconds):
    seconds = int(seconds)
    return f"{seconds // 3600}:{seconds // 60 % 60:02}:{seconds % 60:02}" if seconds >= 3600 else f"{seconds // 60}:{seconds % 60:02}"


def choose_beat(cues: list[dict], shots: list[dict], trailer_duration: float, action: bool, maximum: float = 4.5) -> dict | None:
    candidates = []
    for cue in cues:
        start = max(0, cue["start"] - 0.08)
        end = cue["end"] + 0.12
        if start < trailer_duration * 0.08 or end > trailer_duration * 0.85:
            continue
        if not 1.5 <= end - start <= maximum or len(cue["text"].split()) < 3:
            continue
        if re.search(r"subscribe|in theaters|coming soon|rated |thank you for watching", cue["text"], re.IGNORECASE):
            continue
        motion = max((s["motion"] for s in shots if s["start"] <= start <= s["end"]), default=0)
        score = (motion if action else -abs(motion - 12)) + min(len(cue["text"].split()), 10)
        candidates.append((score, {"start": start, "end": end, "text": cue["text"]}))
    return max(candidates, key=lambda x: x[0])[1] if candidates else None


def validate_package(output: Path, spec: DurationSpec | None = None):
    spec = spec or DurationSpec()
    required = ["final-video.mp4", "subtitles.srt", "description.txt", "recommended-title.txt", *[f"thumbnail-{n:02}.jpg" for n in range(1, 4)]]
    for name in required:
        path = output / name
        if not path.exists() or path.stat().st_size == 0:
            raise PipelineError(f"Missing deliverable: {name}")
    metadata = probe(output / "final-video.mp4")
    seconds = float(metadata["format"]["duration"])
    if not spec.minimum-0.5 <= seconds <= spec.maximum+0.5:
        raise PipelineError(f"Final video duration {seconds:.1f}s is outside the requested {spec.minimum/60:.1f}–{spec.maximum/60:.1f} minutes.")
    streams = metadata["streams"]
    video = next((s for s in streams if s["codec_type"] == "video"), None)
    if not video or (video["width"], video["height"]) != export_size() or not any(s["codec_type"] == "audio" for s in streams):
        raise PipelineError(f"Final export must have {export_size()[0]}x{export_size()[1]} video and an audio stream.")
    from PIL import Image
    for n in range(1, 4):
        path = output / f"thumbnail-{n:02}.jpg"
        with Image.open(path) as image:
            if image.size != (1280, 720) or path.stat().st_size >= 2_000_000:
                raise PipelineError("Thumbnail dimensions or file size failed validation.")
    return seconds


def produce(title: str, brief: str, movies: str, work: Path, output: Path):
    from .availability import enrich_availability
    from .editorial import place_intro_meme, subscription_boundary
    from .footage import download
    from .memes import prepare_intro_meme
    from .packaging import build_description, midroll_text, refresh_availability
    from .render import (
        embed_chapters,
        finish,
        label_image,
        speech_video,
        subscribe_image,
        thumbnails,
        trailer_video,
    )
    from .research import Research
    from .script import Writer
    from .shots import ShotSelector, analyse, intro_montage
    from .voice import Narrator, captions, spoken_coverage

    if not title.strip() or len(title) > 200 or not brief.strip() or len(brief) > 4000:
        raise PipelineError("Provide a title (max 200 characters) and creative brief (max 4,000 characters).")
    items = parse_movies(movies)
    count_match = re.search(r"\btop\s*(\d+)\b", title, re.IGNORECASE)
    if count_match and int(count_match[1]) != len(items):
        raise PipelineError("The title's countdown number must match the number of movies/shows.")
    os.environ["VIDEO_RESOLUTION"] = str(resolution_from_brief(brief))
    spec = requested_duration(brief)
    max_beat = trailer_budget(spec, len(items))
    expected_breaks = min(3, max_beat) * len(items)
    print(f"Requested runtime: {spec.target/60:.2f} minutes", flush=True)
    narrator = narrator_from_brief(brief)
    reference = voice_reference(narrator, Path(os.getenv("VOICE_DIR", "assets/voices")))
    # Include the writing model so a model upgrade never reuses an old model's script.
    script_model = os.getenv("SCRIPT_MODEL", "qwen3:4b")
    run_key = key("pipeline-v4-calm-casual", title, brief, movies, narrator, script_model, os.getenv("TTS_MODEL", "turbo"), key(reference.read_bytes().hex()))
    work = work / run_key
    work.mkdir(parents=True, exist_ok=True)
    output.mkdir(parents=True, exist_ok=True)
    save_json(work / "input.json", {"title": title, "brief": brief, "movies": [vars(i) for i in items], "narrator": narrator, "duration": vars(spec), "script_model": script_model})
    research = Research()
    facts_path = work / "research.json"
    films = json.loads(facts_path.read_text()) if facts_path.exists() else research.collect(items, facts_path)
    films = enrich_availability(films, research, work / "availability")
    trailers = []
    shot_sets = []
    for film in films:
        print(f"Footage: {film['title']}", flush=True)
        trailer = download(film, Path("work/footage"))
        trailers.append(trailer)
        shot_sets.append(analyse(trailer, trailer.with_suffix(".shots.json")))

    writer = Writer(cache_dir=work / "script-sections")
    print(f"Script model: {writer.model}", flush=True)
    script_path = work / "script.json"
    script = json.loads(script_path.read_text()) if script_path.exists() else writer.write(title, brief, films, target_seconds=spec.target, trailer_seconds=expected_breaks)
    refresh_availability(script, films)
    if script["package"].get("thumbnail_version") != 2:
        script["package"]["thumbnail_concepts"] = writer.thumbnail_concepts(title, brief, films, script["sections"])
        script["package"]["thumbnail_version"] = 2
    save_json(script_path, script)
    writer.unload()

    print(f"Narration: {narrator} / {os.getenv('TTS_MODEL', 'turbo')} on CPU", flush=True)
    word_scale = 1.0
    for revision in range(3):
        speaker = Narrator(reference, work / "speech")
        speech = {}
        for section_index, section in enumerate(script["sections"]):
            for paragraph_index, text in enumerate(section["paragraphs"]):
                print(f"Narrating section {section_index+1}, paragraph {paragraph_index+1}", flush=True)
                speech[(section_index, paragraph_index)] = speaker.speak(text)
        speaker.close()
        spoken_seconds = sum(duration(path) for path in speech.values())
        estimated = spoken_seconds + expected_breaks + 11
        if spec.minimum*0.98 <= estimated <= spec.maximum*1.02 or revision == 2:
            break
        print(f"Adjusting overall script to requested runtime; measured {estimated/60:.1f} minutes", flush=True)
        word_scale *= max(0.2, min(5.0, max(1, spec.target-expected_breaks-11)/max(1, spoken_seconds)))
        script = writer.write(title, brief, films, word_scale=word_scale, target_seconds=spec.target, trailer_seconds=expected_breaks)
        save_json(script_path, script)
        writer.unload()

    # Generate the brief CTA separately so its measured length can determine placement.
    cta_text = midroll_text(title+brief+movies)
    speaker = Narrator(reference, work / "speech")
    cta_audio = speaker.speak(cta_text)
    speaker.close()
    def shorten_cta(audio):
        if duration(audio) <= 8:
            return audio
        adjusted = work / "subscribe-short.wav"
        ratio = duration(audio)/8
        tempos = []
        while ratio > 2:
            tempos.append("atempo=2")
            ratio /= 2
        tempos.append(f"atempo={ratio}")
        ffmpeg("-i", audio, "-af", ",".join(tempos), adjusted)
        return adjusted

    cta_audio = shorten_cta(cta_audio)

    print("Aligning speech and analysing trailer dialogue", flush=True)
    from faster_whisper import WhisperModel
    recognizer = WhisperModel("base.en", device="cpu", compute_type="int8", cpu_threads=min(os.cpu_count() or 2, 4))
    aligned = {}
    pending = list(speech)
    for retry in range(3):
        bad = []
        for coordinates in pending:
            audio = speech[coordinates]
            index, paragraph = coordinates
            cache = audio.with_suffix(".captions.json")
            cues = json.loads(cache.read_text()) if cache.exists() else captions(audio, recognizer)
            if spoken_coverage(script["sections"][index]["paragraphs"][paragraph], cues) < 0.72:
                audio.unlink(missing_ok=True)
                cache.unlink(missing_ok=True)
                bad.append(coordinates)
            else:
                save_json(cache, cues)
                aligned[coordinates] = cues
        if not bad:
            break
        if retry == 2:
            raise PipelineError(f"Speech verification failed after automatic retries in {len(bad)} paragraphs.")
        print(f"Automatically regenerating {len(bad)} speech segments", flush=True)
        speaker = Narrator(reference, work / "speech")
        for coordinates in bad:
            i, p = coordinates
            speech[coordinates] = speaker.speak(script["sections"][i]["paragraphs"][p], attempt_offset=(retry+1)*3)
        speaker.close()
        pending = bad
    trailer_cues = []
    for trailer in trailers:
        cache = trailer.with_suffix(".dialogue.json")
        cues = json.loads(cache.read_text()) if cache.exists() else captions(trailer, recognizer)
        save_json(cache, cues)
        trailer_cues.append(cues)
    for retry in range(3):
        cta_cues = captions(cta_audio, recognizer)
        if spoken_coverage(cta_text, cta_cues) >= .72:
            break
        if retry == 2:
            raise PipelineError("Short subscribe speech did not pass transcript verification after retries.")
        cta_audio.unlink(missing_ok=True)
        speaker = Narrator(reference, work / "speech")
        cta_audio = shorten_cta(speaker.speak(cta_text, attempt_offset=(retry+1)*3))
        speaker.close()
    del recognizer
    gc.collect()

    action = bool(re.search(r"\b(action|energetic|fast.paced|fight|intense)\b", title + " " + brief, re.IGNORECASE))
    selector = ShotSelector()
    for film, trailer, shots in zip(films, trailers, shot_sets):
        selector.index(trailer, shots)
        selector.people[str(trailer)] = film.get("cast", [])
    beats = {i: choose_beat(trailer_cues[section["film"]], shot_sets[section["film"]], duration(trailers[section["film"]]), action, maximum=max_beat)
             for i, section in enumerate(script["sections"]) if section["kind"] == "film" and max_beat}
    lengths = {coordinates: duration(audio) for coordinates, audio in speech.items()}
    boundary, fraction = subscription_boundary(script["sections"], lengths,
        {i: beat["end"]-beat["start"] for i, beat in beats.items() if beat}, duration(cta_audio))
    print(f"Like/subscribe break before a review at {fraction:.0%} of the measured video.", flush=True)
    shifted = {}
    shifted_cues = {}
    for (i, p), audio in speech.items():
        shifted[(i+(i >= boundary), p)] = audio
        shifted_cues[(i+(i >= boundary), p)] = aligned[(i,p)]
    shifted[(boundary, 0)] = cta_audio
    shifted_cues[(boundary, 0)] = cta_cues
    speech, aligned = shifted, shifted_cues
    script["sections"].insert(boundary, {"kind": "subscribe", "film": script["sections"][boundary]["film"], "paragraphs": [cta_text]})
    beats = {i+(i >= boundary): beat for i, beat in beats.items()}
    require_disk(work, spec.target)
    intro_text = " ".join(script["sections"][0]["paragraphs"])
    meme = prepare_intro_meme(intro_text, title+brief+movies, work / "memes")
    meme_used = False
    parts = []
    all_cues = []
    chapters = []
    timeline = []
    offset = 0.0
    for section_index, section in enumerate(script["sections"]):
        film_index = section["film"]
        film = films[film_index]
        trailer = trailers[film_index]
        is_film = section["kind"] == "film"
        chapter = f"#{film_index+1} {film['title']} ({film['year']})" if is_film else section["kind"].title()
        if section["kind"] != "subscribe":
            chapters.append({"start": offset, "title": chapter})
        label = work / f"label-{section_index}.png"
        if section["kind"] == "subscribe":
            subscribe_image(label)
        else:
            label_image(label, film["title"] if is_film else title, film_index+1 if is_film else None, film["year"] if is_film else None)
        montage = None
        if section["kind"] == "intro":
            montage = intro_montage(selector, trailers, shot_sets,
                                    [duration(speech[(section_index, p)]) for p in range(len(section["paragraphs"]))],
                                    title + ". " + brief, action)
        midpoint = max(1, len(section["paragraphs"]) // 2)
        beat = beats.get(section_index)
        for paragraph_index, text in enumerate(section["paragraphs"]):
            print(f"Editing: {chapter}, paragraph {paragraph_index+1}", flush=True)
            audio = speech[(section_index, paragraph_index)]
            chosen = montage[paragraph_index] if montage is not None else selector.choose(trailer, shot_sets[film_index], text, duration(audio), action)
            if section["kind"] == "subscribe":
                held = max(chosen, key=lambda cut: cut["duration"])
                chosen = [{**held, "start": held["start"]+.2, "duration": duration(audio), "freeze": 0, "still": True}]
            if meme and montage is not None and not meme_used:
                chosen, meme_used = place_intro_meme(chosen, text, aligned[(section_index, paragraph_index)], meme)
            path = work / f"part-{len(parts):03}.mp4"
            speech_video(trailer, audio, chosen, path, label if paragraph_index == 0 else None, work / f"render-{len(parts):03}", label_seconds=duration(audio) if section["kind"] == "subscribe" else 3.5)
            seconds = duration(path)
            for cue in aligned[(section_index, paragraph_index)]:
                all_cues.append({"start": offset + cue["start"], "end": offset + min(cue["end"], seconds), "text": cue["text"]})
            timeline.append({"start": offset, "duration": seconds, "type": "narration", "film": "Intro montage" if montage is not None else film["title"], "films": sorted({films[cut.get("film_index", film_index)]["title"] for cut in chosen}), "text": text, "shots": chosen})
            parts.append(path)
            offset += seconds
            if beat and paragraph_index + 1 == midpoint:
                path = work / f"part-{len(parts):03}.mp4"
                trailer_video(trailer, beat["start"], beat["end"]-beat["start"], path)
                seconds = duration(path)
                # Align the exact retained audio rather than copy a potentially truncated cue.
                # Word timestamps from the source transcript are clipped to this insert.
                for cue in trailer_cues[film_index]:
                    start, end = max(cue["start"], beat["start"]), min(cue["end"], beat["end"])
                    if end > start:
                        all_cues.append({"start": offset+start-beat["start"], "end": offset+end-beat["start"], "text": cue["text"]})
                timeline.append({"start": offset, "duration": seconds, "type": "trailer-audio", "film": film["title"], "source_start": beat["start"]})
                parts.append(path)
                offset += seconds
    del selector
    gc.collect()
    print("Mixing and exporting final video", flush=True)
    final_duration, speed = finish(parts, output / "final-video.mp4", work, action, spec=spec)
    for cue in all_cues:
        cue["start"] /= speed
        cue["end"] = min(cue["end"] / speed, final_duration)
    write_srt(output / "subtitles.srt", all_cues)
    for chapter in chapters:
        chapter["start"] /= speed
    package = script["package"]
    (output / "recommended-title.txt").write_text(package["title"].strip() + "\n")
    source_links = [json.loads(t.with_suffix('.json').read_text())["source"] for t in trailers]
    (output / "description.txt").write_text(build_description(script, films, chapters, source_links))
    embed_chapters(output / "final-video.mp4", chapters, final_duration, work)
    credits = "Film metadata: TMDB — https://www.themoviedb.org/\nWatch listings: JustWatch via TMDB — https://www.justwatch.com/us\n"
    if meme_used:
        credits += str(meme.get("attribution", "Intro meme"))+" — "+str(meme.get("source", ""))+"\n"
    (output / "sources-and-credits.txt").write_text(credits)
    thumbnails(trailers, shot_sets, package["thumbnail_texts"], output,
               concepts=package.get("thumbnail_concepts"), title=title, brief=brief,
               work=work / "thumbnail-work", use_api=os.getenv("THUMBNAIL_AI", "true").lower() == "true",
               use_cutouts=True)
    save_json(work / "timeline.json", {"speed": speed, "events": timeline, "chapters": chapters})
    for part in parts:
        part.unlink(missing_ok=True)
    validate_package(output, spec=spec)
    # Full decode checks catch corrupt/truncated exports. Audio normalization is handled in render.finish.
    ffmpeg("-xerror", "-i", output / "final-video.mp4", "-f", "null", "-")
    print(f"Complete: {final_duration/60:.2f} minutes. Package: {output}", flush=True)


def main():
    parser = argparse.ArgumentParser(description="Produce an unattended movie countdown; request duration in the creative brief.")
    parser.add_argument("--title", default=os.getenv("VIDEO_TITLE", ""))
    parser.add_argument("--brief", default=os.getenv("VIDEO_BRIEF", ""))
    parser.add_argument("--movies", default=os.getenv("MOVIE_LIST", ""))
    parser.add_argument("--work", type=Path, default=Path("work/jobs"))
    parser.add_argument("--output", type=Path, default=Path("output"))
    args = parser.parse_args()
    try:
        produce(args.title, args.brief, args.movies, args.work, args.output)
    except Exception as exc:  # noqa: BLE001 -- CLI boundary reports failure without uploading partial output.
        # Failed runs must never upload an apparently finished partial package.
        if args.output.exists():
            (args.output / "final-video.mp4").unlink(missing_ok=True)
        print(f"Production failed ({type(exc).__name__}): {exc}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
