from __future__ import annotations

import argparse
import gc
import json
import math
import os
import re
import sys
import time
from pathlib import Path
from types import SimpleNamespace

from .core import (
    PipelineError,
    duration,
    ffmpeg,
    key,
    notice,
    parse_movies,
    probe,
    save_json,
    selected_content_type,
    selected_narrator,
    voice_reference,
    write_srt,
)
from .timing import DurationSpec, final_tempo, trailer_budget, workflow_duration
from .video_settings import (
    MAX_UPSCALE,
    choose_output_size,
    download_ceiling,
    export_size,
    inspect_sources,
    require_disk,
)

COLD_OPEN_SECONDS = 5.0
REVIEW_PREVIEW_SECONDS = 5.0
IMPACT_SECONDS = 3.2
TEASE_SECONDS = 3.0  # a flash of number one at the end of the intro: a reason to stay to the end


def number_one_tease_shot(shots: list[dict], is_card, taken: list[tuple[float, float]],
                          seconds: float = TEASE_SECONDS) -> dict | None:
    """The loudest, most striking moment of number one's trailer that the review itself won't show.

    ``is_card(index)`` marks text cards; ``taken`` holds (start, end) windows already used for
    number one (its preview, dialogue beats and impact moment), with a small margin.
    """
    options = []
    for index, shot in enumerate(shots):
        if is_card(index) or float(shot["end"]) - float(shot["start"]) < 1.2:
            continue
        centre = float(shot.get("punch_at", (float(shot["start"]) + float(shot["end"])) / 2))
        start = max(0.0, min(centre - seconds * 0.4, float(shot["end"]) - seconds))
        end = start + seconds
        if any(start < b + 1.0 and end > a - 1.0 for a, b in taken):
            continue
        score = float(shot.get("peak", 0)) * 2 + float(shot.get("motion", 0))
        options.append((score, start, end))
    if not options:
        return None
    _, start, end = max(options)
    return {"start": start, "end": end}
END_SCREEN_SECONDS = 17.0  # plus the final three-second hold: about 20 seconds for YouTube end-screen elements
INTRO_TARGET_SECONDS = 8.0


def note_visual_beat(selector, seconds, *, still=False):
    record = getattr(selector, "note_visual_beat", None)
    if callable(record):
        record(seconds, still=still)


def refine_intro_timing(writer, title, brief, films, script, speech, speaker_factory):
    """One editorial shortening pass; a soft intro target never rejects audio."""
    index = next((i for i, section in enumerate(script["sections"]) if section["kind"] == "intro"), None)
    if index is None:
        return script, speech
    coordinates = [(index, p) for p in range(len(script["sections"][index]["paragraphs"]))]
    if any(speech.get(coordinate) is None for coordinate in coordinates):
        return script, speech
    previous_seconds = sum(duration(speech[coordinate]) for coordinate in coordinates)
    if previous_seconds <= INTRO_TARGET_SECONDS+2:
        return script, speech
    words = sum(len(paragraph.split()) for paragraph in script["sections"][index]["paragraphs"])
    target = max(8, min(16, round(words*INTRO_TARGET_SECONDS/previous_seconds)))
    speaker = None
    try:
        candidate = writer.refresh_intro(title, brief, films, script, target_words=target)
        writer.unload()
        paragraphs = candidate["sections"][index]["paragraphs"]
        if paragraphs == script["sections"][index]["paragraphs"]:
            return script, speech
        speaker = speaker_factory()
        replacement = {(index, p): speaker.speak(text) for p, text in enumerate(paragraphs)}
        if sum(duration(audio) for audio in replacement.values()) < previous_seconds:
            updated = {coordinate: audio for coordinate, audio in speech.items() if coordinate[0] != index}
            updated.update(replacement)
            print("Using a shorter, complete opening; eight seconds remains a soft guide.", flush=True)
            return candidate, updated
    except Exception:  # noqa: BLE001 -- retain a complete take when optional editorial refinement is unavailable.
        print("Keeping the complete opening; intro timing guidance never aborts production.", flush=True)
    finally:
        if speaker is not None:
            speaker.close()
    return script, speech


def chapter_time(seconds):
    seconds = int(seconds)
    return f"{seconds // 3600}:{seconds // 60 % 60:02}:{seconds % 60:02}" if seconds >= 3600 else f"{seconds // 60}:{seconds % 60:02}"


def choose_beat(cues: list[dict], shots: list[dict], trailer_duration: float, action: bool,
                maximum: float = 4.5, *, excluded: list[dict] | None = None) -> dict | None:
    candidates = []
    for index, initial in enumerate(cues):
        if index and initial["start"]-cues[index-1]["end"] < .4 and not re.search(r"[.!?][\"’']?$", cues[index-1]["text"].strip()):
            continue  # Do not begin with the second subtitle chunk of a sentence.
        cue = dict(initial)
        for following in cues[index+1:]:
            if re.search(r"[.!?][\"’']?$", cue["text"].strip()) or following["start"]-cue["end"] >= .4:
                break
            cue["text"] += " "+following["text"]
            cue["end"] = following["end"]
        start = max(0, cue["start"] - 0.08)
        end = cue["end"] + 0.12
        if start < trailer_duration * 0.08 or end > trailer_duration * 0.85:
            continue
        if not 1.5 <= end - start <= maximum or len(cue["text"].split()) < 3:
            continue
        if any(start < shot["end"] and end > shot["start"] for shot in excluded or []):
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
    if os.getenv("COMPARISON_EXPORT", "false").lower() == "true":
        required.append("final-video-augmented.mp4")
    for name in required:
        path = output / name
        if not path.exists() or path.stat().st_size == 0:
            raise PipelineError(f"Missing deliverable: {name}")
    metadata = probe(output / "final-video.mp4")
    seconds = float(metadata["format"]["duration"])
    if not math.isfinite(seconds) or seconds <= 0:
        raise PipelineError("Final export has no valid media duration.")
    if not spec.automatic and not spec.minimum-.5 <= seconds <= spec.maximum+.5:
        print(f"Complete video is {seconds/60:.1f} minutes; outside the soft target, retained without padding or cutting a review.", flush=True)
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


def write_upload_extras(output: Path, package: dict, films: list[dict]):
    """Copy-paste helpers for YouTube Studio: title options, tags and a comment to pin."""
    options = [t for t in package.get("title_options", []) if isinstance(t, str) and t.strip()] or [package["title"]]
    (output / "title-options.txt").write_text(
        "Recommended title first; YouTube Studio's title test can compare up to three.\n\n"
        + "\n".join(f"{i}. {t.strip()} ({len(t.strip())} characters)" for i, t in enumerate(options, 1)) + "\n")
    tags = [t for t in package.get("tags", []) if isinstance(t, str) and t.strip()]
    if not tags:
        tags = [f"{f['title']} {f['year']}" for f in films]
    (output / "tags.txt").write_text(", ".join(tags) + "\n")
    question = package.get("pinned_comment") or "Which one would you add to this list?"
    shows = bool(films) and all(f.get("kind") == "tv" for f in films)
    followup = str(package.get("comment_followup", "")).strip()
    (output / "pinned-comment.txt").write_text(
        f"{question.strip()}\n\n" + (f"{followup}\n\n" if followup and followup != question.strip() else "")
        + f"Drop your pick below; the best {'show' if shows else 'movie'} suggestions might make a future list.\n")


def export_comparison(output: Path, work: Path, timeline: list[dict], speed: float, reference: Path):
    from faster_whisper import WhisperModel

    from .audio_review import verify_export_audio
    from .augmentation import export_augmented
    from .speech_recovery import SpeechRecovery
    from .voice import Narrator
    print("Exporting the augmented comparison with the same timeline", flush=True)
    protected = [(max(0, event["start"]/speed-.25), (event["start"]+event["duration"])/speed+.25)
                 for event in timeline if event.get("type") in {"narration", "subscribe"}]
    comparison = export_augmented(output / "final-video.mp4", output / "final-video-augmented.mp4", work / "comparison", protected_intervals=protected)
    comparison_recovery = SpeechRecovery(None, lambda: Narrator(reference, work / "speech"),
        work / "comparison-audio-checks", fallback_factory=lambda: WhisperModel(
            "small.en", device="cpu", compute_type="int8", cpu_threads=min(os.cpu_count() or 2, 4),
            download_root=os.getenv("SPEECH_VERIFY_MODEL_DIR") or None))
    try:
        report = verify_export_audio(output / "final-video-augmented.mp4", timeline, speed, comparison_recovery, work)
        save_json(work / "comparison-audio-review.json", report)
        if not report.get("accepted"):
            raise PipelineError("The augmented comparison did not preserve verified narration; see comparison-audio-review.json.")
    finally:
        comparison_recovery.close()
    save_json(output / "comparison-settings.json", comparison)
    (output / "comparison-readme.txt").write_text(
        "final-video.mp4: clean version.\nfinal-video-augmented.mp4: the same edit with visual/audio augmentation.\n"
        "Both use subtitles.srt and the same chapters. Narration is protected from audio augmentation. Augmentation is not anonymization or copyright protection.\n")


def setup_run(title: str, brief: str, movies: str, work: Path, output: Path, started: float) -> SimpleNamespace:
    """Validate inputs, resolve research and compute the run's cache folder (shared by every workflow job)."""
    from .availability import enrich_availability
    from .research import Research
    if not title.strip() or len(title) > 200 or not brief.strip() or len(brief) > 4000:
        raise PipelineError("Provide a title (max 200 characters) and creative brief (max 4,000 characters).")
    items = parse_movies(movies)
    content_type = selected_content_type()
    count_match = re.search(r"\btop\s*(\d+)\b", title, re.IGNORECASE)
    if count_match and int(count_match[1]) != len(items):
        raise PipelineError("The title's countdown number must match the number of movies/shows.")
    ceiling = download_ceiling()
    os.environ["VIDEO_RESOLUTION"] = str(ceiling)
    spec = workflow_duration(title, len(items))
    max_beat = min(7.0, trailer_budget(spec, len(items))*1.5)
    expected_breaks = min(3, max_beat) * len(items) + COLD_OPEN_SECONDS + REVIEW_PREVIEW_SECONDS*len(items)
    print(f"Requested runtime: {spec.target/60:.2f} minutes", flush=True)
    narrator = selected_narrator(brief)
    reference = voice_reference(narrator, Path(os.getenv("VOICE_DIR", "assets/voices")))
    # Include the resolver policy and writing model so stale identities/scripts are not reused.
    script_model = os.getenv("SCRIPT_MODEL", "gemma4:12b")
    run_key = key("pipeline-v13-three-trailer-rounds", title, brief, movies, content_type, narrator, script_model, vars(spec), os.getenv("TTS_MODEL", "turbo"), key(reference.read_bytes().hex()))
    work = work / run_key
    work.mkdir(parents=True, exist_ok=True)
    output.mkdir(parents=True, exist_ok=True)
    research = Research()
    # An explicit dropdown limits lookup to that medium. Local legacy runs can infer it.
    shows_title = re.search(r"\b(?:shows?|series|tv)\b", title, re.IGNORECASE)
    movies_title = re.search(r"\b(?:movies?|films?)\b", title, re.IGNORECASE)
    research.preferred_kind = content_type or ("tv" if shows_title and not movies_title else "movie")
    research.selected_kind = content_type
    facts_path = work / "research.json"
    films = json.loads(facts_path.read_text()) if facts_path.exists() else research.collect(items, facts_path)
    films = enrich_availability(films, research, work / "availability")
    notice("Research", f"{len(films)} titles resolved; Wikipedia notes for {sum(1 for f in films if f.get('wiki_summary'))}; "
           f"TMDB stills for {sum(1 for f in films if f.get('backdrops'))}. Elapsed {(time.time()-started)/60:.0f} min.")
    spec = workflow_duration(title, len(items), films)
    max_beat = min(7.0, trailer_budget(spec, len(items))*1.5)
    expected_breaks = min(3, max_beat)*len(items) + COLD_OPEN_SECONDS + REVIEW_PREVIEW_SECONDS*len(items)
    print(f"Runtime mode: {'Auto' if spec.automatic else 'selected minutes'}; soft plan {spec.target/60:.1f} minutes.", flush=True)
    save_json(work / "input.json", {"title": title, "brief": brief, "movies": [vars(i) for i in items], "content_type": content_type or "auto", "narrator": narrator, "duration": vars(spec), "script_model": script_model, "try_4k": ceiling == 2160})
    return SimpleNamespace(items=items, content_type=content_type, ceiling=ceiling, spec=spec, max_beat=max_beat,
                           expected_breaks=expected_breaks, narrator=narrator, reference=reference, script_model=script_model,
                           work=work, research=research, films=films)


def produce(title: str, brief: str, movies: str, work: Path, output: Path):
    started = time.time()
    from .audio_review import verify_export_audio
    from .availability import enrich_availability
    from .cta import cta_variants, prepare_cta, render_cta
    from .cta_animation import add_cta_animation
    from .editorial import place_intro_meme, subscription_boundary
    from .footage import download
    from .memes import prepare_intro_meme
    from .packaging import build_description, refresh_availability
    from .people import insert_trailer_hold, trailer_still_for
    from .previews import (
        choose_preview,
        choose_title_card,
        cold_open_plan,
        outro_recap_plan,
    )
    from .render import (
        embed_chapters,
        end_screen_image,
        end_screen_video,
        finish,
        j_cut,
        label_image,
        number_one_tease_image,
        reveal_style_for,
        speech_video,
        thumbnails,
        tracker_image,
        trailer_video,
        up_next_image,
    )
    from .research import Research
    from .script import Writer, aligned_delivery
    from .subtitles import script_cues
    from .dialogue import isolate_dialogue
    from .secondary_footage import download_secondary
    from .shots import ShotSelector, analyse, get_frame, intro_montage, pace_limit
    from .soundtrack import music_credit, narration_rate, theme_for
    from .source_pool import (
        best_intro_sources,
        choose_from_sources,
        primary_fallback_plan,
    )
    from .speech_recovery import SpeechRecovery, SpeechRecoveryError
    from .voice import Narrator, captions

    ctx = setup_run(title, brief, movies, work, output, started)
    from .thumbnail_studio import detect_genre
    genre = detect_genre(ctx.films, title, brief)
    os.environ["VIDEO_GENRE"] = genre  # drives the footage grade, music theme and reveal weighting
    items, content_type, ceiling, spec = ctx.items, ctx.content_type, ctx.ceiling, ctx.spec
    max_beat, expected_breaks, narrator, reference = ctx.max_beat, ctx.expected_breaks, ctx.narrator, ctx.reference
    script_model, work, research, films = ctx.script_model, ctx.work, ctx.research, ctx.films
    trailers = []
    shot_sets = []
    print("Footage round 1: fetch a primary trailer for every title.", flush=True)
    for film in films:
        print(f"Footage: {film['title']}", flush=True)
        trailer = download(film, Path("work/footage"))
        trailers.append(trailer)
        shot_sets.append(analyse(trailer, trailer.with_suffix(".shots.json")))

    sources = inspect_sources(trailers)
    width, height = choose_output_size(sources, ceiling)
    os.environ["VIDEO_RESOLUTION"] = f"{width}x{height}"
    pools = [[trailer] for trailer in trailers]
    shot_lookup = {str(trailer): shots for trailer, shots in zip(trailers, shot_sets)}
    for i, source in enumerate(sources):
        source.update(film=films[i]["title"], role="primary")
    for round_number in (2, 3):
        print(f"Footage round {round_number}: optional extra trailers, after the previous round finishes for all titles.", flush=True)
        for i, (film, primary) in enumerate(zip(films, trailers)):
            extra = None
            try:
                # The extra source is optional and must leave space for the final render.
                require_disk(work, spec.target)
                extra = (download_secondary(film, Path("work/footage"), primary, (width, height)) if round_number == 2 else
                             download_secondary(film, Path("work/footage"), primary, (width, height),
                                                existing=pools[i][1:], slot=3))
                if extra is None:
                    print(f"Keeping {len(pools[i])} trailer(s) for {film['title']}; optional trailer {round_number} was not accepted.", flush=True)
                    continue
                details = inspect_sources([extra])[0]
                if min(details["height"], details["width"]*9/16) * MAX_UPSCALE < height:
                    raise PipelineError("Optional trailer is too far below the selected export resolution.")
                extra_shots = analyse(extra, extra.with_suffix(".shots.json"))
                require_disk(work, spec.target)
                pools[i].append(extra)
                shot_lookup[str(extra)] = extra_shots
                details.update(film=film["title"], role="secondary" if round_number == 2 else "tertiary")
                sources.append(details)
                print(f"Using {len(pools[i])} trailers for {film['title']} to provide more scene variety.", flush=True)
            except Exception:  # noqa: BLE001 -- an optional extra trailer must never discard a usable first one.
                if extra is not None and extra != primary:
                    for path in extra.parent.glob(extra.stem+".*"):
                        if path.is_file():
                            path.unlink(missing_ok=True)
                print(f"Trailer {round_number} skipped for {film['title']}; keeping the accepted trailers.", flush=True)
    quality_report = {"try_4k": ceiling == 2160, "ceiling_height": ceiling,
                      "sources": sources, "export_width": width, "export_height": height}
    save_json(work / "quality-report.json", quality_report)
    print(f"Auto quality: {width}x{height}; highest shared primary-source resolution up to {ceiling}p. Optional extra trailers meet this quality.", flush=True)
    from . import scenes
    if scenes.enabled():
        described = 0
        for pool in pools:
            for source in pool:
                if scenes.describe_trailer(source, shot_lookup[str(source)]):
                    described += 1
        scenes.unload()
        print(f"Scene descriptions ready for {described} of {sum(len(pool) for pool in pools)} trailers.", flush=True)
    from . import download_access
    per_title = "; ".join(f"{source.get('film', '?')} ({source.get('role', '?')}): {source.get('width')}x{source.get('height')}" for source in sources)
    notice("Footage", f"Export {width}x{height}; {len(sources)} trailers accepted; route: {download_access.routes()[0]}. {per_title}. "
           f"Elapsed {(time.time()-started)/60:.0f} min.")

    writer = Writer(cache_dir=work / "script-sections")
    print(f"Script model: {writer.model}", flush=True)
    script_path = work / "script.json"
    script = json.loads(script_path.read_text()) if script_path.exists() else writer.write(title, brief, films, target_seconds=spec.target, trailer_seconds=expected_breaks)
    script = writer.refresh_editorial(title, brief, films, script)
    refresh_availability(script, films)
    if script["package"].get("thumbnail_version") != 2:
        script["package"]["thumbnail_concepts"] = writer.thumbnail_concepts(title, brief, films, script["sections"])
        script["package"]["thumbnail_version"] = 2
    save_json(script_path, script)
    writer.unload()
    intro_preview = " ".join(script["sections"][0]["paragraphs"])
    first_review = next((" ".join(s["paragraphs"]) for s in script["sections"] if s["kind"] == "film"), "")
    notice("Script", f"Model {writer.model}. Title: {script['package'].get('title')}. "
           f"Angle: {script['package'].get('episode_plan', {}).get('angle', '-')}. Intro: {intro_preview} "
           f"First review: {first_review[:900]} Elapsed {(time.time()-started)/60:.0f} min.")

    # Check the short intermission before spending an hour on full narration.
    # The same verifier then covers every hook, review, availability line and outro.
    from faster_whisper import WhisperModel
    recognizer = WhisperModel("base.en", device="cpu", compute_type="int8", cpu_threads=min(os.cpu_count() or 2, 4))
    recovery = SpeechRecovery(recognizer, lambda: Narrator(reference, work / "speech"),
        work / "speech-checks", fallback_factory=lambda: WhisperModel(
            "small.en", device="cpu", compute_type="int8", cpu_threads=min(os.cpu_count() or 2, 4),
            download_root=os.getenv("SPEECH_VERIFY_MODEL_DIR") or None))
    cta_plan = prepare_cta(title, brief, title+brief+movies, work / "memes")
    cta_error = None
    try:
        for candidate in cta_variants(cta_plan):
            try:
                cta_audio, cta_cues = recovery.recover(candidate["text"], label="subscribe break",
                    max_seconds=8, allow_split=False)
            except SpeechRecoveryError as exc:
                cta_error = exc
                print("Trying another short joke paired with the same reaction and narrator.", flush=True)
                continue
            cta_plan = candidate
            cta_text = candidate["text"]
            break
        else:
            raise SpeechRecoveryError(f"No complete short subscribe take could be verified. {cta_error}")
    finally:
        recovery.release_voice()
    print("Subscribe speech verified before full narration generation.", flush=True)

    print(f"Narration: {narrator} / {os.getenv('TTS_MODEL', 'turbo')} on CPU", flush=True)
    word_scale = 1.0
    best_draft = None
    intro_refined = False
    for revision in range(3):
        try:
            speaker = Narrator(reference, work / "speech")
            speech = {}
            try:
                for section_index, section in enumerate(script["sections"]):
                    labels = aligned_delivery(section)
                    for paragraph_index, text in enumerate(section["paragraphs"]):
                        print(f"Narrating section {section_index+1}, paragraph {paragraph_index+1}", flush=True)
                        try:
                            speech[(section_index, paragraph_index)] = speaker.speak(text, delivery=labels[paragraph_index])
                        except (RuntimeError, OSError, ValueError):
                            # Defer a bad take to the same bounded recovery used
                            # for missing words; never substitute silence or text.
                            speech[(section_index, paragraph_index)] = None
                            print("Initial voice take unavailable; queued for verified same-voice recovery.", flush=True)
            finally:
                speaker.close()
            if not intro_refined:
                script, speech = refine_intro_timing(writer, title, brief, films, script, speech,
                    lambda: Narrator(reference, work / "speech"))
                intro_refined = True
            spoken_seconds = sum(duration(path) if path is not None else
                len(script["sections"][i]["paragraphs"][p].split())*60/145
                for (i, p), path in speech.items())
        except Exception:
            if best_draft is None:
                raise
            print("Timing revision unavailable; retaining the earlier complete narration.", flush=True)
            break
        estimated = spoken_seconds+expected_breaks+11
        # Auto uses a broad editorial guide, not a required runtime window.
        comfortable = .5*spec.target <= estimated <= 1.6*spec.target if spec.automatic else spec.minimum <= estimated <= spec.maximum
        distance = abs(estimated-spec.target)+(0 if comfortable else spec.target)
        if best_draft is None or distance < best_draft[0]:
            best_draft = (distance, script, speech)
        if comfortable or revision == 2 or any(path is None for path in speech.values()):
            break
        print(f"Refining an unusually {'long' if estimated > spec.target else 'short'} draft; measured {estimated/60:.1f} minutes.", flush=True)
        word_scale *= max(.2, min(5.0, max(1, spec.target-expected_breaks-11)/max(1, spoken_seconds)))
        try:
            script = writer.write(title, brief, films, word_scale=word_scale, target_seconds=spec.target, trailer_seconds=expected_breaks)
            writer.unload()
        except Exception:  # noqa: BLE001 -- duration guidance is best effort; keep a complete usable draft.
            print("Timing rewrite unavailable; retaining a complete draft.", flush=True)
            break
    _distance, script, speech = best_draft
    save_json(script_path, script)

    print("Verifying narration and analysing trailer dialogue", flush=True)
    aligned = {}
    grouped = {}
    for coordinates, audio in speech.items():
        i, p = coordinates
        text = script["sections"][i]["paragraphs"][p]
        grouped.setdefault((audio, text), []).append(coordinates)
    try:
        for (audio, text), coordinates in grouped.items():
            i, p = coordinates[0]
            label = f"{script['sections'][i]['kind']} section {i+1}, paragraph {p+1}"
            verified, cues = recovery.recover(text, audio, label=label)
            for coordinate in coordinates:
                speech[coordinate] = verified
                aligned[coordinate] = cues
    finally:
        recovery.close()
    unpaced_take = next(iter(speech.values()), None)
    if unpaced_take is not None:
        ffmpeg("-i", unpaced_take, "-t", 12, "-c:a", "pcm_s24le", work / "voice-before-pacing.wav")
    music_theme = theme_for(title, brief)
    voice_speed = narration_rate(music_theme)
    from .script import aligned_delivery
    from .sound_design import delivery_speed
    from .voice import paced_take
    paced = {}
    delivery_report = {}
    for coordinate, audio in speech.items():
        section = script["sections"][coordinate[0]]
        label = aligned_delivery(section)[coordinate[1]]
        # Hooks and reveals move a little faster; tense or emotional lines breathe.
        paragraph_speed = delivery_speed(voice_speed, label)
        delivery_report[label] = delivery_report.get(label, 0) + 1
        if (audio, paragraph_speed) not in paced:
            paced[(audio, paragraph_speed)] = paced_take(audio, paragraph_speed)
        speech[coordinate] = paced[(audio, paragraph_speed)]
        aligned[coordinate] = [{**cue, "start": cue["start"]/paragraph_speed,
                                "end": cue["end"]/paragraph_speed} for cue in aligned[coordinate]]
    print(f"Narration delivery: {delivery_report}.", flush=True)
    notice("Narration", f"Delivery labels {delivery_report}; {sum(duration(a) for a in set(speech.values()))/60:.1f} min of speech. "
           f"Elapsed {(time.time()-started)/60:.0f} min.")
    if voice_speed != 1:
        cta_audio = paced_take(cta_audio, voice_speed)
        cta_cues = [{**cue, "start": cue["start"]/voice_speed,
                    "end": cue["end"]/voice_speed} for cue in cta_cues]
    paced_example = next(iter(speech.values()), None)
    if paced_example is not None:
        ffmpeg("-i", paced_example, "-t", 12, "-c:a", "pcm_s24le", work / "voice-after-pacing.wav")
    save_json(work / "soundtrack-report.json", {"theme": music_theme, "narration_rate": voice_speed, "delivery": delivery_report,
              "music_source": music_credit(music_theme), "trailer_under_narration": False})
    trailer_cues = []
    for trailer in trailers:
        cache = trailer.with_suffix(".dialogue.json")
        cues = json.loads(cache.read_text()) if cache.exists() else captions(trailer, recognizer)
        save_json(cache, cues)
        trailer_cues.append(cues)
    del recognizer
    gc.collect()

    action = bool(re.search(r"\b(action|energetic|fast.paced|fight|intense)\b", title + " " + brief, re.IGNORECASE)) or sum("Action" in film.get("genres", []) for film in films) >= max(1, len(films)//2)
    selector = ShotSelector()
    selector.peak_weight = {"horror": 1.1, "action": 1.1, "thriller": 1.0, "adventure": 0.9, "scifi": 0.9, "fantasy": 0.9}.get(genre, 0.6)
    selector.titles = {str(source): film["title"] for film, pool in zip(films, pools) for source in pool}
    for film, trailer, shots in zip(films, trailers, shot_sets):
        selector.index(trailer, shots)
        selector.people[str(trailer)] = film.get("cast", [])
    for film, pool in zip(films, pools):
        for extra in pool[1:]:
            try:
                selector.index(extra, shot_lookup[str(extra)])
                selector.people[str(extra)] = film.get("cast", [])
            except Exception:  # noqa: BLE001 -- optional analysis failure uses the primary source.
                pool.remove(extra)
                print(f"Optional trailer analysis skipped for {film['title']}.", flush=True)
    active_files = {path.name for pool in pools for path in pool}
    quality_report["sources"] = [source for source in sources if source["file"] in active_files]
    save_json(work / "quality-report.json", quality_report)
    notice("Shot matching", f"{'SigLIP 2' if getattr(selector, 'siglip', None) is not None else 'CLIP'} image matching; "
           f"AI scene descriptions for {len(getattr(selector, 'scene_embeddings', {}))} of {len(active_files)} trailers.")
    ending_cards = {i: choose_title_card(selector, pool, shot_lookup, films[i]["title"])
                    for i, pool in enumerate(pools)}
    card_kind = getattr(selector, "card_kind", None)
    excluded_cards = {film_index: [shot for index, shot in enumerate(shots)
                                  if callable(card_kind) and card_kind(trailers[film_index], index) is not None]
                      for film_index, shots in enumerate(shot_sets)}
    beats = {i: choose_beat(trailer_cues[section["film"]], shot_sets[section["film"]], duration(trailers[section["film"]]), action,
                           maximum=max_beat, excluded=excluded_cards[section["film"]])
             for i, section in enumerate(script["sections"]) if section["kind"] == "film" and max_beat}
    extra_beats = {}
    for i, section in enumerate(script["sections"]):
        first = beats.get(i)
        if section["kind"] == "film" and first and len(section["paragraphs"]) >= 3:
            film_index = section["film"]
            extra_beats[i] = choose_beat(trailer_cues[film_index], shot_sets[film_index],
                duration(trailers[film_index]), action, maximum=max_beat,
                excluded=[*excluded_cards[film_index], first])
    reserve = getattr(selector, "reserve", None)
    if callable(reserve):
        for i, beat in [*beats.items(), *extra_beats.items()]:
            if beat:
                reserve(trailers[script["sections"][i]["film"]], beat["start"], beat["end"]-beat["start"], "trailer-audio")
    # Impact moment: right after each review's hook, the trailer's own scare/impact beat plays with its sound.
    impacts = {}
    for i, section in enumerate(script["sections"]):
        if section["kind"] != "film" or len(section["paragraphs"]) < 3 or os.getenv("IMPACT_BEATS", "true") == "false":
            continue
        film_index = section["film"]
        taken = [beat for beat in (beats.get(i), extra_beats.get(i)) if beat]
        options = []
        for index, shot in enumerate(shot_sets[film_index]):
            if callable(card_kind) and card_kind(trailers[film_index], index) is not None:
                continue
            centre = float(shot.get("punch_at", (shot["start"] + shot["end"]) / 2))
            start = max(float(shot["start"]), centre - 1.1)
            end = min(float(shot["end"]), start + IMPACT_SECONDS)
            if end - start < 1.8 or any(start < b["end"] and end > b["start"] for b in taken):
                continue
            options.append((float(shot.get("peak", 0)), start, end))
        if options:
            peak, start, end = max(options)
            if peak >= 0.3:
                impacts[i] = {"start": start, "end": end, "peak": peak}
                if callable(reserve):
                    reserve(trailers[film_index], start, end - start, "trailer-audio")
    dialogue = {str(trailer): cues for trailer, cues in zip(trailers, trailer_cues)}
    opening = cold_open_plan(selector, pools, shot_lookup, title+". "+brief, action,
                             dialogue=dialogue, seconds=COLD_OPEN_SECONDS)
    previews = {i: choose_preview(selector, pools[section["film"]], shot_lookup,
                 films[section["film"]].get("overview", "")+". "+title+". "+brief,
                 REVIEW_PREVIEW_SECONDS, action, dialogue=dialogue)
                for i, section in enumerate(script["sections"]) if section["kind"] == "film"}
    tease = None
    top_index = next((i for i, section in enumerate(script["sections"]) if section["kind"] == "film" and section["film"] == 0), None)
    if top_index is not None and len(films) >= 3 and os.getenv("TEASE_NUMBER_ONE", "true").strip().lower() != "false":
        taken = [(previews[top_index]["start"], previews[top_index]["start"] + previews[top_index]["duration"])] \
            if Path(previews[top_index]["source"]) == trailers[0] else []
        taken += [(b["start"], b["end"]) for b in (beats.get(top_index), extra_beats.get(top_index), impacts.get(top_index)) if b]
        tease = number_one_tease_shot(shot_sets[0], lambda index: callable(card_kind) and card_kind(trailers[0], index) is not None, taken)
        if tease and callable(reserve):
            reserve(trailers[0], tease["start"], TEASE_SECONDS, "number-one-tease")
    lengths = {coordinates: duration(audio) for coordinates, audio in speech.items()}
    planned_seconds = (sum(lengths.values())+sum(preview["duration"] for preview in previews.values())
                       +sum(card["duration"] for card in ending_cards.values() if card)
                       +sum(beat["end"]-beat["start"] for beat in [*beats.values(), *extra_beats.values(), *impacts.values()] if beat)
                       +COLD_OPEN_SECONDS+duration(cta_audio)+(TEASE_SECONDS if tease else 0))
    selector.playback_speed = final_tempo(planned_seconds, spec)
    boundary, fraction = subscription_boundary(script["sections"], lengths,
        {i: preview["duration"] + (beats[i]["end"]-beats[i]["start"] if beats.get(i) else 0)
            + (extra_beats[i]["end"]-extra_beats[i]["start"] if extra_beats.get(i) else 0)
            + (impacts[i]["end"]-impacts[i]["start"] if impacts.get(i) else 0)
            + (ending_cards[script["sections"][i]["film"]]["duration"] if ending_cards.get(script["sections"][i]["film"]) else 0)
         for i, preview in previews.items()}, duration(cta_audio), opening_seconds=COLD_OPEN_SECONDS)
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
    previews = {i+(i >= boundary): preview for i, preview in previews.items()}
    extra_beats = {i+(i >= boundary): beat for i, beat in extra_beats.items()}
    impacts = {i+(i >= boundary): beat for i, beat in impacts.items()}
    overlays = []  # countdown tracker images with their (pre-tempo) start and end
    require_disk(work, spec.target)
    intro_text = " ".join(script["sections"][0]["paragraphs"])
    meme = prepare_intro_meme(intro_text, title+brief+movies, work / "memes")
    meme_used = False
    parts = []
    all_cues = []
    chapters = []
    timeline = []
    used_video_sources = set()
    cta_rendered = {}
    portraits_used = set()
    offset = 0.0
    film_position = 0  # entries shown so far, for rotating the rank reveal
    for shot in opening:
        film_index = shot["film_index"]
        source = Path(shot["source"])
        path = work / f"part-{len(parts):03}.mp4"
        try:
            trailer_video(source, shot["start"], shot["duration"], path, muted=shot["muted"],
                          still=bool(shot.get("still", False)), still_motion="zoom_in" if shot.get("still") else None)
        except (PipelineError, OSError, RuntimeError, ValueError):
            if source == trailers[film_index]:
                raise
            shot = {**choose_preview(selector, [trailers[film_index]], shot_lookup,
                    title+". "+brief, shot["duration"], action, dialogue=dialogue), "film_index": film_index}
            source = Path(shot["source"])
            trailer_video(source, shot["start"], shot["duration"], path, muted=shot["muted"],
                          still=bool(shot.get("still", False)), still_motion="zoom_in" if shot.get("still") else None)
        seconds = duration(path)
        timeline.append({"start": offset, "duration": seconds, "type": "cold-open", "film": films[film_index]["title"],
                         "source_start": shot["start"], "shots": [shot], "narration": False})
        parts.append(path)
        used_video_sources.add(str(source))
        offset += seconds
        note_visual_beat(selector, seconds, still=bool(shot.get("still")))
    for section_index, section in enumerate(script["sections"]):
        film_index = section["film"]
        film = films[film_index]
        trailer = trailers[film_index]
        is_film = section["kind"] == "film"
        chapter = f"#{film_index+1} {film['title']} ({film['year']})" if is_film else section["kind"].title()
        if section["kind"] != "subscribe":
            chapters.append({"start": 0.0 if section["kind"] == "intro" else offset, "title": chapter})
        section_start = offset
        if is_film:
            label = work / f"label-{section_index}.png"
            label_image(label, film["title"], film_index+1, film["year"])
            preview = previews[section_index]
            style = reveal_style_for(film_position, film_index+1)
            film_position += 1
            source = Path(preview["source"])
            path = work / f"part-{len(parts):03}.mp4"
            try:
                trailer_video(source, preview["start"], preview["duration"], path,
                              label=label, label_seconds=preview["duration"], muted=preview["muted"],
                              still=bool(preview.get("still", False)), still_motion="zoom_in" if preview.get("still") else None,
                              reveal=True, reveal_style=style, audio=None if preview["muted"] or preview.get("still") else
                              isolate_dialogue(source, preview["start"], preview["duration"], work / "dialogue"))
            except (PipelineError, OSError, RuntimeError, ValueError):
                if source == trailer:
                    raise
                preview = choose_preview(selector, [trailer], shot_lookup, title+". "+brief,
                                         REVIEW_PREVIEW_SECONDS, action, dialogue=dialogue)
                source = Path(preview["source"])
                trailer_video(source, preview["start"], preview["duration"], path,
                              label=label, label_seconds=preview["duration"], muted=preview["muted"],
                              still=bool(preview.get("still", False)), still_motion="zoom_in" if preview.get("still") else None,
                              reveal=True, reveal_style=style, audio=None if preview["muted"] or preview.get("still") else
                              isolate_dialogue(source, preview["start"], preview["duration"], work / "dialogue"))
            seconds = duration(path)
            preview_seconds = seconds
            timeline.append({"start": offset, "duration": seconds, "type": "review-preview", "film": film["title"],
                             "source_start": preview["start"], "shots": [preview], "narration": False, "reveal_style": style})
            parts.append(path)
            used_video_sources.add(str(source))
            offset += seconds
            note_visual_beat(selector, seconds, still=bool(preview.get("still")))
        montage = None
        if section["kind"] == "intro":
            intro_sources = best_intro_sources(selector, pools, shot_lookup, title+". "+brief, action)
            montage = intro_montage(selector, intro_sources, [shot_lookup[str(path)] for path in intro_sources],
                                    [duration(speech[(section_index, p)]) for p in range(len(section["paragraphs"]))],
                                    title + ". " + brief, action)
        elif section["kind"] == "outro":
            montage = outro_recap_plan(selector, pools, shot_lookup,
                [duration(speech[(section_index, p)]) for p in range(len(section["paragraphs"]))],
                " ".join(section["paragraphs"]), action)
        midpoint = max(1, len(section["paragraphs"]) // 2)
        beat = beats.get(section_index)
        for paragraph_index, text in enumerate(section["paragraphs"]):
            print(f"Editing: {chapter}, paragraph {paragraph_index+1}", flush=True)
            audio = speech[(section_index, paragraph_index)]
            selector.pace_limit = pace_limit(aligned_delivery(section)[paragraph_index], action)
            chosen = montage[paragraph_index] if montage is not None else choose_from_sources(selector, pools[film_index], shot_lookup, text, duration(audio), action)
            if is_film and duration(audio) >= 4:
                portrait = trailer_still_for(text, film, selector, pools[film_index], shot_lookup)
                if portrait and (film_index, portrait["person"]) not in portraits_used:
                    name = portrait["person"].split()[-1].casefold()
                    mention = next((cue["start"] for cue in aligned[(section_index, paragraph_index)]
                                    if name in cue["text"].casefold()), 0)
                    chosen = insert_trailer_hold(chosen, portrait, when=mention)
                    portraits_used.add((film_index, portrait["person"]))
                    reserve_hold = getattr(selector, "reserve", None)
                    if callable(reserve_hold):
                        reserve_hold(Path(portrait["source"]), portrait["start"], 1/30, "character-hold")
                    note_visual_beat(selector, min(5.5, duration(audio)), still=True)
            if meme and section["kind"] == "intro" and montage is not None and not meme_used:
                chosen, meme_used = place_intro_meme(chosen, text, aligned[(section_index, paragraph_index)], meme)
            path = work / f"part-{len(parts):03}.mp4"
            if section["kind"] == "subscribe":
                held = max(chosen, key=lambda cut: cut["duration"])
                cta_rendered = render_cta(cta_plan, audio, path, work / f"render-{len(parts):03}", background=held)
                chosen = [cta_rendered]
            else:
                try:
                    speech_video(trailer, audio, chosen, path, None, work / f"render-{len(parts):03}")
                except (PipelineError, OSError, RuntimeError, ValueError):
                    optional = {str(extra) for pool in pools for extra in pool[1:]}
                    failed = {cut.get("source") for cut in chosen if cut.get("source") in optional}
                    if not failed:
                        raise
                    print("Optional trailer could not render; retrying this paragraph with primary footage.", flush=True)
                    chosen = primary_fallback_plan(selector, chosen, trailers, shot_sets, text, action, film_index)
                    for pool in pools:
                        pool[:] = [source for source in pool if str(source) not in failed]
                    speech_video(trailer, audio, chosen, path, None, work / f"render-{len(parts):03}")
            if section["kind"] in {"subscribe", "outro"} and re.search(r"\bsubscribe\b", text, re.IGNORECASE):
                add_cta_animation(path, text, aligned[(section_index, paragraph_index)], work / f"cta-animation-{len(parts):03}")
            used_video_sources.update(str(cut.get("source", trailer)) for cut in chosen if not cut.get("image"))
            seconds = duration(path)
            # Captions use the exact script words; recognition supplies only the timing.
            for cue in script_cues(text, aligned[(section_index, paragraph_index)]):
                all_cues.append({"start": offset + cue["start"], "end": offset + min(cue["end"], seconds), "text": cue["text"]})
            timeline.append({"start": offset, "duration": seconds, "type": "subscribe" if section["kind"] == "subscribe" else "narration",
                             "section_kind": section["kind"], "source_audio": str(audio),
                             "film": section["kind"].title()+" montage" if montage is not None else film["title"], "films": sorted({films[cut.get("film_index", film_index)]["title"] for cut in chosen}), "text": text, "shots": chosen})
            parts.append(path)
            offset += seconds
            impact = impacts.get(section_index) if paragraph_index == 0 else None
            if impact:
                try:
                    j_cut(parts[-1], trailer, impact["start"])
                    path = work / f"part-{len(parts):03}.mp4"
                    trailer_video(trailer, impact["start"], impact["end"]-impact["start"], path)
                except (PipelineError, OSError, RuntimeError, ValueError):
                    print(f"Impact moment skipped for {film['title']}.", flush=True)
                else:
                    seconds = duration(path)
                    timeline.append({"start": offset, "duration": seconds, "type": "trailer-audio", "film": film["title"],
                                     "source_start": impact["start"], "separated": False, "impact": True})
                    parts.append(path)
                    used_video_sources.add(str(trailer))
                    offset += seconds
            active_beat = (beat if paragraph_index+1 == midpoint else
                           extra_beats.get(section_index) if paragraph_index+1 == max(midpoint+1, len(section["paragraphs"])-1) else None)
            if active_beat:
                beat = active_beat
                try:
                    j_cut(parts[-1], trailer, beat["start"])
                except (PipelineError, OSError, RuntimeError, ValueError):
                    pass
                path = work / f"part-{len(parts):03}.mp4"
                separated = isolate_dialogue(trailer, beat["start"], beat["end"]-beat["start"], work / "dialogue")
                trailer_video(trailer, beat["start"], beat["end"]-beat["start"], path, audio=separated)
                seconds = duration(path)
                # Align the exact retained audio rather than copy a potentially truncated cue.
                # Word timestamps from the source transcript are clipped to this insert.
                for cue in trailer_cues[film_index]:
                    start, end = max(cue["start"], beat["start"]), min(cue["end"], beat["end"])
                    if end > start:
                        all_cues.append({"start": offset+start-beat["start"], "end": offset+end-beat["start"], "text": cue["text"]})
                timeline.append({"start": offset, "duration": seconds, "type": "trailer-audio", "film": film["title"], "source_start": beat["start"],
                                 "separated": separated is not None})
                parts.append(path)
                offset += seconds
        card = ending_cards.get(film_index) if is_film else None
        if is_film:
            # Lower left, where the rank/title box sat: the small countdown tracker appears once that
            # box has gone and stays until the entry ends; in the entry's last seconds it gives way
            # to 'UP NEXT #4 Title' (number one stays secret), so viewers who don't care about the
            # next pick skip ahead with the chapters instead of leaving. Both vanish before the next
            # entry's rank/title box comes in.
            label_end = section_start + preview_seconds + 0.3
            switch = offset
            upcoming = next((s["film"] for s in script["sections"][section_index+1:] if s["kind"] == "film"), None)
            if upcoming is not None and offset - label_end > 9 and os.getenv("UP_NEXT", "true").strip().lower() != "false":
                switch = max(label_end + 5.0, offset - 4.5)
                pill = work / f"up-next-{film_index+1}.png"
                up_next_image(pill, upcoming+1, None if upcoming == 0 else films[upcoming]["title"])
                overlays.append((pill, switch, offset, "lower-left"))
            if len(films) >= 3 and switch - label_end > 2 and os.getenv("COUNTDOWN_TRACKER", "true") != "false":
                tracker = work / f"tracker-{film_index+1}.png"
                tracker_image(tracker, len(films), film_index+1)
                overlays.append((tracker, label_end, switch, "lower-left"))
        if card:
            source = Path(card["source"])
            path = work / f"part-{len(parts):03}.mp4"
            try:
                trailer_video(source, card["start"], card["duration"], path, muted=True,
                              still=bool(card.get("still", False)))
            except (PipelineError, OSError, RuntimeError, ValueError):
                print(f"Optional title card skipped for {film['title']}; its opening label remains available.", flush=True)
            else:
                seconds = duration(path)
                timeline.append({"start": offset, "duration": seconds, "type": "title-card",
                                 "section_kind": "film", "film": film["title"], "shots": [card], "narration": False})
                parts.append(path)
                used_video_sources.add(str(source))
                offset += seconds
                note_visual_beat(selector, seconds, still=True)
        if section["kind"] == "intro" and tease:
            # A three-second flash of number one, unnamed: the reason to stay until the end.
            plate = work / "number-one-tease.png"
            number_one_tease_image(plate)
            path = work / f"part-{len(parts):03}.mp4"
            try:
                trailer_video(trailers[0], tease["start"], TEASE_SECONDS, path, label=plate,
                              label_seconds=TEASE_SECONDS, reveal=True, reveal_style="flash")
            except (PipelineError, OSError, RuntimeError, ValueError):
                print("Number one teaser skipped.", flush=True)
            else:
                seconds = duration(path)
                timeline.append({"start": offset, "duration": seconds, "type": "teaser", "film": films[0]["title"],
                                 "source_start": tease["start"], "narration": False})
                parts.append(path)
                used_video_sources.add(str(trailers[0]))
                offset += seconds
                note_visual_beat(selector, seconds)
    # A calm end-screen plate gives YouTube's end-screen elements room (plus the final hold, about 20 seconds).
    try:
        top_section = next(i for i, section in enumerate(script["sections"]) if section["kind"] == "film" and section["film"] == 0)
        top_preview = previews[top_section]
        try:
            background = get_frame(Path(top_preview["source"]), float(top_preview["start"]) + 1)
        except Exception:  # noqa: BLE001 -- a plain dark plate is an acceptable background.
            background = None
        plate = work / "end-screen.png"
        end_screen_image(plate, background)
        path = work / f"part-{len(parts):03}.mp4"
        end_screen_video(plate, END_SCREEN_SECONDS, path)
        seconds = duration(path)
        timeline.append({"start": offset, "duration": seconds, "type": "end-screen", "narration": False})
        parts.append(path)
        offset += seconds
    except (PipelineError, OSError, StopIteration, KeyError, ValueError) as exc:
        print(f"End screen skipped ({type(exc).__name__}); the final hold remains.", flush=True)
    del selector
    gc.collect()
    print("Mixing and exporting final video", flush=True)
    final_duration, speed = finish(parts, output / "final-video.mp4", work, action, spec=spec, music_theme=music_theme,
                                   timeline=timeline, top_title=films[0]["title"], overlays=overlays)
    export_recovery = SpeechRecovery(None, lambda: Narrator(reference, work / "speech"),
        work / "export-audio-checks", fallback_factory=lambda: WhisperModel(
            "small.en", device="cpu", compute_type="int8", cpu_threads=min(os.cpu_count() or 2, 4),
            download_root=os.getenv("SPEECH_VERIFY_MODEL_DIR") or None))
    try:
        audio_report = verify_export_audio(output / "final-video.mp4", timeline, speed, export_recovery, work)
        save_json(work / "audio-review.json", audio_report)
        if not audio_report.get("accepted"):
            raise PipelineError("The finished mix lost verified narration; the damaged export will not be delivered. See audio-review.json.")
    finally:
        export_recovery.close()
    save_json(work / "duration-report.json", {"automatic": spec.automatic, "target_seconds": spec.target,
              "minimum_seconds": spec.minimum, "maximum_seconds": spec.maximum,
              "actual_seconds": final_duration, "inside_soft_range": spec.minimum <= final_duration <= spec.maximum,
              "tempo": speed})
    for cue in all_cues:
        cue["start"] /= speed
        cue["end"] = min(cue["end"] / speed, final_duration)
    write_srt(output / "subtitles.srt", all_cues)
    for chapter in chapters:
        chapter["start"] /= speed
    package = script["package"]
    (output / "recommended-title.txt").write_text(package["title"].strip() + "\n")
    write_upload_extras(output, package, films)
    credited_trailers = list(dict.fromkeys([path for pool in pools for path in pool]+[Path(source) for source in sorted(used_video_sources)]))
    active_files = {path.name for path in credited_trailers}
    quality_report["sources"] = [source for source in sources if source["file"] in active_files]
    save_json(work / "quality-report.json", quality_report)
    source_links = [json.loads(t.with_suffix('.json').read_text())["source"] for t in credited_trailers]
    (output / "description.txt").write_text(build_description(script, films, chapters, source_links, music_credit(music_theme)))
    embed_chapters(output / "final-video.mp4", chapters, final_duration, work)
    if os.getenv("COMPARISON_EXPORT", "false").lower() == "true":
        # Optional (repository variable COMPARISON_EXPORT=true): roughly doubles the final encode time.
        export_comparison(output, work, timeline, speed, reference)
    credits = music_credit(music_theme)+"\nFilm metadata: TMDB — https://www.themoviedb.org/\nWatch listings: JustWatch via TMDB — https://www.justwatch.com/us\n"
    if meme_used:
        credits += str(meme.get("attribution", "Intro meme"))+" — "+str(meme.get("source", ""))+"\n"
    if cta_rendered.get("credit_source"):
        credits += str(cta_rendered.get("attribution", "Subscribe reaction"))+" — "+str(cta_rendered["credit_source"])+"\n"
    (output / "sources-and-credits.txt").write_text(credits)
    thumbnails(trailers, shot_sets, package["thumbnail_texts"], output, films=films,
               concepts=package.get("thumbnail_concepts"), title=title, brief=brief,
               work=work / "thumbnail-work", use_api=os.getenv("THUMBNAIL_AI", "true").lower() == "true",
               use_cutouts=True)
    if os.getenv("REVIEW_REPORT", "true").strip().lower() != "false":
        from .review import build_review
        try:
            stills_films = []
            for film, trailer in zip(films, trailers):
                try:
                    if json.loads(trailer.with_suffix(".json").read_text()).get("stills_fallback"):
                        stills_films.append(film["title"])
                except (OSError, ValueError):
                    pass
            montage = [(entry["start"] / speed, (entry["start"] + entry["duration"]) / speed) for entry in timeline
                       if entry.get("type") in ("cold-open", "teaser") or entry.get("section_kind") in ("intro", "outro")]
            review = build_review(output, all_cues, chapters, genre, stills_films=stills_films, montage=montage)
            notice("Review report", f"review.html: {review['black_frames']} black-frame runs, {review['repeated_shots']} repeated shots, "
                   f"{review['loudness_spikes']} loudness spikes, {review['captions_on_silence']} captions over silence.")
        except Exception as exc:  # noqa: BLE001 -- the report is advisory
            notice("Review report skipped", f"{type(exc).__name__}: {str(exc)[-300:]}", "warning")
    save_json(work / "timeline.json", {"speed": speed, "events": timeline, "chapters": chapters})
    for part in parts:
        part.unlink(missing_ok=True)
    validate_package(output, spec=spec)
    # Full decode checks catch corrupt/truncated exports. Audio normalization is handled in render.finish.
    ffmpeg("-xerror", "-i", output / "final-video.mp4", "-f", "null", "-")
    assembly = json.loads((work / "audio-assembly.json").read_text()) if (work / "audio-assembly.json").exists() else {}
    loudness = assembly.get("loudness", {})
    notice("Complete", f"{final_duration/60:.2f} min video at tempo {speed:.3f}; loudness before mastering "
           f"{loudness.get('measured', {}).get('integrated', '?')} LUFS, gain {loudness.get('gain_db', '?')} dB; "
           f"{assembly.get('effect_events', 0)} transition effects. Total {(time.time()-started)/60:.0f} min.")
    print(f"Complete: {final_duration/60:.2f} minutes. Package: {output}", flush=True)


def warm_voice(sections: list[dict], ctx) -> int:
    """Record this section's narration in the parallel job (same cache the final job reads)."""
    if os.getenv("VOICE_WARM", "false").strip().lower() != "true" or not sections:
        return 0
    try:
        from .script import aligned_delivery
        from .voice import Narrator
        speaker = Narrator(ctx.reference, ctx.work / "speech")
    except Exception as exc:  # noqa: BLE001 -- the final job voices everything itself
        print(f"Voice warm-up skipped ({type(exc).__name__}: {str(exc)[:200]}).", flush=True)
        return 0
    voiced = 0
    try:
        for section in sections:
            labels = aligned_delivery(section)
            for index, text in enumerate(section.get("paragraphs", [])):
                try:
                    speaker.speak(text, delivery=labels[index])
                    voiced += 1
                except Exception as exc:  # noqa: BLE001 -- the final job retries with recovery
                    print(f"Voice warm-up: one paragraph left for the final job ({type(exc).__name__}).", flush=True)
    finally:
        speaker.close()
    print(f"Voice warm-up: {voiced} paragraphs recorded.", flush=True)
    return voiced


def warm(stage: str, part: str, title: str, brief: str, movies: str, work: Path, output: Path) -> dict:
    """Fill the run's caches in a parallel workflow job; the final produce job then reuses them.

    prepare  research; reports the parts, genre and score length for the other jobs
    plan     the episode plan every script section follows
    script   one script section: "intro", "outro" or "film-<index>" (0 is number one)
    footage  one title's primary trailer: download, end-slate scan, shot analysis, scene descriptions
    """
    from . import scenes
    from .script import Writer
    started = time.time()
    ctx = setup_run(title, brief, movies, work, output, started)
    films = ctx.films
    if stage == "plan":
        (ctx.work / "script-sections").mkdir(parents=True, exist_ok=True)  # packaged even without a plan
        writer = Writer(cache_dir=ctx.work / "script-sections")
        planned = bool(writer.plan(title, brief, films))
        writer.unload()
        return {"planned": planned}
    if stage == "prepare":
        # Research only: footage and music jobs start from this; the plan job follows.
        from .thumbnail_studio import detect_genre
        parts = {"script_parts": ["intro", "outro", *[f"film-{i}" for i in range(len(films))]],
                 "footage_parts": [str(i) for i in range(len(films))],
                 "genre": detect_genre(films, title, brief),
                 # ACE-Step stops any generation after 10 minutes (about 120 s of music on the CPU);
                 # 100 s stays safely inside that, and the music bed loops it with crossfades.
                 "music_seconds": 100}
        target = os.getenv("GITHUB_OUTPUT")
        if target:
            with open(target, "a", encoding="utf-8") as stream:
                for name, value in parts.items():
                    stream.write(f"{name}={value if isinstance(value, (str, int)) else json.dumps(value)}\n")
        return parts
    if stage == "script":
        writer = Writer(cache_dir=ctx.work / "script-sections")
        result = writer.write(title, brief, films, target_seconds=ctx.spec.target, trailer_seconds=ctx.expected_breaks, only={part})
        writer.unload()
        print(f"Warmed script section {part}.", flush=True)
        voiced = warm_voice(result.get("sections", []), ctx)
        return {"section": part, "voiced_paragraphs": voiced}
    if stage == "footage":
        from .footage import download
        from .shots import analyse
        film = films[int(part)]
        trailer = download(film, Path("work/footage"))
        shots = analyse(trailer, trailer.with_suffix(".shots.json"))
        scenes.describe_trailer(trailer, shots)
        # Extra trailers are fetched in global rounds by the final job, after
        # every title's primary has been fetched. Parallel warm-ups stay primary-only.
        extra = None
        scenes.unload()
        print(f"Warmed footage for {film['title']}: {trailer.name}, {len(shots)} shots; "
              f"optional extras deferred to the global download rounds.", flush=True)
        return {"trailer": str(trailer), "extra": str(extra) if extra else ""}
    raise PipelineError(f"Unknown stage {stage!r}.")


def main():
    parser = argparse.ArgumentParser(description="Produce an unattended movie/show countdown using the workflow selectors.")
    parser.add_argument("--title", default=os.getenv("VIDEO_TITLE", ""))
    parser.add_argument("--brief", default=os.getenv("VIDEO_BRIEF", ""))
    parser.add_argument("--movies", default=os.getenv("MOVIE_LIST", ""))
    parser.add_argument("--work", type=Path, default=Path("work/jobs"))
    parser.add_argument("--output", type=Path, default=Path("output"))
    parser.add_argument("--stage", choices=["produce", "prepare", "plan", "script", "footage"], default=os.getenv("PIPELINE_STAGE", "produce"))
    parser.add_argument("--part", default=os.getenv("PIPELINE_PART", ""))
    args = parser.parse_args()
    if args.stage != "produce":
        try:
            warm(args.stage, args.part, args.title, args.brief, args.movies, args.work, args.output)
        except Exception as exc:  # noqa: BLE001 -- warm-up jobs are optional; produce does any missing work
            notice(f"Warm-up {args.stage} {args.part}".strip(), f"{type(exc).__name__}: {str(exc)[-800:]}", "warning")
            return 1
        return 0
    try:
        produce(args.title, args.brief, args.movies, args.work, args.output)
    except Exception as exc:  # noqa: BLE001 -- CLI boundary reports failure without uploading partial output.
        # Failed runs must never upload an apparently finished partial package.
        if args.output.exists():
            (args.output / "final-video.mp4").unlink(missing_ok=True)
        print(f"Production failed ({type(exc).__name__}): {exc}", file=sys.stderr)
        import traceback
        notice("Production failed", f"{type(exc).__name__}: {str(exc)[-1500:]}\n{traceback.format_exc()[-1500:]}", "error")
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
