"""Exercise production timing/metadata orchestration without model or media downloads."""
from __future__ import annotations

import copy
import json
import sys
from itertools import pairwise
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock

import pytest
from PIL import Image

from movie_pipeline import (
    availability,
    cli,
    footage,
    memes,
    render,
    research,
    script,
    secondary_footage,
    shots,
    voice,
)
from movie_pipeline.core import PipelineError, key, save_json, write_srt
from movie_pipeline.packaging import MIDROLLS
from movie_pipeline.video_settings import export_size


@pytest.mark.parametrize("count,try_4k,source_height,expected_size,extra_mode,content_type,narrator", [
    (2,"true",2160,(3840,2160),"found",None,None),
    (5,"true",1080,(1920,1080),"missing","Shows","Female"),
    (2,"false",2160,(1920,1080),"raises","Movies","Kit Laser"),
    (5,"true",720,(1280,720),"found","Movies","Warguy"),
    (2,"true",2160,(3840,2160),"render-fails",None,None),
    (2,"true",2160,(3840,2160),"rewrite-fails",None,None),
    (2,"false",2160,(1920,1080),"shared-audio-retry","Movies","Warguy"),
])
def test_produce_keeps_cta_between_reviews_and_retimes_metadata(monkeypatch, tmp_path, count, try_4k, source_height, expected_size, extra_mode, content_type, narrator):
    monkeypatch.chdir(tmp_path)
    monkeypatch.setenv("VIDEO_RESOLUTION", "1080")
    monkeypatch.setenv("TRY_4K", try_4k)
    monkeypatch.setenv("SCRIPT_MODEL", "qwen3:4b")
    monkeypatch.setenv("VIDEO_MINUTES", "8" if extra_mode == "rewrite-fails" else "5")
    for name, selection in (("VIDEO_TYPE", content_type), ("NARRATOR", narrator)):
        if selection is None:
            monkeypatch.delenv(name, raising=False)
        else:
            monkeypatch.setenv(name, selection)
    expected_kind = "tv" if content_type == "Shows" else "movie"
    expected_narrator = {None:"warguy", "Warguy":"warguy", "Kit Laser":"kit-laser", "Female":"female"}[narrator]
    records = {}
    durations = {}
    text_for_audio = {}
    render_sizes = []
    written_cues = []
    embedded = []
    first = "Stop scrolling. Your next favorite movie might be here."
    second = "Two minutes of scrolling saved. Let's find your next watch."
    films = [{"title": f"Movie {i+1}", "year": 2025, "id": i+1, "kind": expected_kind,
              "cast": [f"Actor {i+1}"], "genres": ["Action"], "status": "Released"} for i in range(count)]
    reference = tmp_path / "reference.ogg"
    reference.write_bytes(b"reference")

    def make(path, seconds, data=b"test media"):
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(data)
        durations[str(path)] = seconds
        return path

    class FakeResearch:
        def collect(self, items, path):
            assert len(items) == count
            records['selected_kind'] = getattr(self, 'selected_kind', None)
            records['preferred_kind'] = self.preferred_kind
            save_json(path, films)
            return copy.deepcopy(films)

    def enrich(values, source, cache):
        assert len(values) == count
        return [{**film, "watch_availability": {"region": "US", "checked_at": "2026-10-20T12:00:00Z",
                 "status": "found", "subscription": ["Netflix"], "rent": ["Amazon Video"]}} for film in values]

    class FakeWriter:
        model = "qwen3:4b"

        def __init__(self, cache_dir):
            self.cache_dir = cache_dir

        def write(self, title, brief, values, **kwargs):
            records["writes"] = records.get("writes", 0)+1
            if extra_mode == "rewrite-fails" and records["writes"] > 1:
                raise RuntimeError("Optional timing rewrite unavailable")
            assert all(film['watch_availability']['region'] == 'US' for film in values)
            sections = [{"kind": "intro", "film": 0, "paragraphs": [first, second]}]
            sections += [{"kind": "film", "film": i, "paragraphs": [f"Review number {i+1} is easy to follow."]}
                         for i in reversed(range(count))]
            sections += [{"kind": "outro", "film": 0,
                          "paragraphs": ["Like, subscribe, and watch our other recommendations. Save your next movie night."]}]
            return {"sections": sections, "package": {"title": f"Top {count} Action Movies",
                    "description": "Looking for action movies to watch? Start with these picks.",
                    "thumbnail_texts": ["WATCH THESE", "NEXT MOVIE NIGHT", "ACTION PICKS"],
                    "thumbnail_concepts": [], "thumbnail_version": 2}}

        def unload(self):
            pass

    class FakeNarrator:
        def __init__(self, reference, cache):
            self.cache = cache

        def speak(self, text, **kwargs):
            if text in {first, second}:
                seconds = 6
            elif text.startswith("Review number"):
                seconds = 120 if count == 2 else 47
            elif text.startswith("It's listed"):
                seconds = 3
                if kwargs.get('attempt_offset'):
                    records['shared_regenerations'] = records.get('shared_regenerations', 0)+1
            elif text in MIDROLLS or text == records.get('cta_text'):
                seconds = 6
            else:
                seconds = 20
            path = make(self.cache / (key(text)+".wav"), seconds)
            text_for_audio[str(path)] = text
            return path

        def close(self):
            pass

    def download(film, root):
        records.setdefault("footage_order", []).append(("primary", film["id"]))
        path = make(root / f"movie-{film['id']}.mp4", 150)
        save_json(path.with_suffix('.json'), {'source': f"https://example.org/trailer/{film['id']}"})
        return path

    def secondary(film, root, primary, minimum_size):
        assert len([event for event in records["footage_order"] if event[0]=="primary"]) == count
        records["footage_order"].append(("secondary", film["id"]))
        records['extra_attempts'] = records.get('extra_attempts', 0)+1
        assert minimum_size == expected_size
        if extra_mode == 'raises':
            raise RuntimeError('Optional source unavailable')
        if extra_mode in {'missing', 'rewrite-fails', 'shared-audio-retry'}:
            return None
        path = make(root / f"movie-{film['id']}-extra.mp4", 150)
        save_json(path.with_suffix('.json'), {'source': f"https://example.org/extra/{film['id']}"})
        return path

    def captions(audio, recognizer):
        if str(audio) in text_for_audio:
            assert Path(audio).exists(), 'Shared speech must remain present while its coordinates are verified'
            text = text_for_audio[str(audio)]
            if extra_mode == 'shared-audio-retry' and text.startswith("It's listed"):
                records['shared_transcriptions'] = records.get('shared_transcriptions', 0)+1
                if not records.get('shared_regenerations'):
                    return [{'start': 0, 'end': durations[str(audio)], 'text': 'Missing the expected availability sentence.'}]
            return [{"start": 0, "end": durations[str(audio)], "text": text_for_audio[str(audio)]}]
        return [{"start": 30, "end": 32.9, "text": "We have to leave right now"}]

    class FakeSelector:
        def __init__(self):
            self.people = {}

        def index(self, path, choices):
            pass

        def hook_score(self, path, choices, text, action):
            return .5

        def choose(self, path, choices, text, seconds, action, **kwargs):
            return [{"start": 40, "duration": seconds, "freeze": 0}]

    def montage(selector, trailers, choices, lengths, brief, action):
        return [[{"start": 40, "duration": seconds, "freeze": 0,
                  "source": str(trailers[i % count]), "film_index": i % count}]
                for i, seconds in enumerate(lengths)]

    def speech_video(source, audio, selected, target, label, work, **kwargs):
        if extra_mode == 'render-fails' and any('-extra.mp4' in cut.get('source', '') for cut in selected):
            records['render_fallbacks'] = records.get('render_fallbacks', 0)+1
            raise PipelineError('Optional decoder failed')
        render_sizes.append(export_size())
        text = text_for_audio[str(audio)]
        assert label is None  # Rank labels belong to previews, never the spoken review.
        records.setdefault('spoken_renders', []).append({'text': text, 'target': str(target)})
        make(target, durations[str(audio)])

    def trailer_video(source, start, seconds, target, **kwargs):
        records.setdefault('trailer_renders', []).append({
            'source': str(source), 'start': start, 'duration': seconds,
            'target': str(target), **kwargs,
        })
        make(target, seconds)

    def cold_open_plan(selector, pools, shot_lookup, text, action, dialogue=None, seconds=5):
        assert seconds == 5
        records['cold_open_planned'] = True
        return [{'source': str(pools[0][0]), 'start': 40, 'duration': seconds, 'muted': True, 'film_index': 0}]

    def choose_preview(selector, sources, shot_lookup, text, seconds, action, dialogue=None):
        assert seconds == 3.5
        records.setdefault('review_previews', []).append(str(sources[0]))
        return {'source': str(sources[0]), 'start': 40, 'duration': seconds, 'muted': True}

    def prepare_cta(title, brief, seed, work):
        records['cta_text'] = MIDROLLS[0]
        plan = {'text': MIDROLLS[0], 'path': str(tmp_path / 'reaction.gif'),
                'animated': True, 'headline': 'LIKE + SUBSCRIBE', 'theme': 'movie-night',
                'source': 'https://example.org/reaction', 'attribution': 'Test reaction asset'}
        records['cta_prepared'] = copy.deepcopy(plan)
        return plan

    def render_cta(plan, audio, target, work, background=None):
        assert plan['animated'] is True
        assert plan == records['cta_prepared']
        assert text_for_audio[str(audio)] == plan['text']
        records['cta_render'] = {'plan': copy.deepcopy(plan), 'target': str(target), 'background': background}
        render_sizes.append(export_size())
        make(target, durations[str(audio)])
        return {'source': plan['path'], 'start': 0, 'duration': durations[str(audio)],
                'image': True, 'animation': True, 'still': False,
                'credit_source': plan['source'], 'attribution': plan['attribution']}

    def finish(parts, target, work, action, spec):
        records['ending_hold'] = 3
        records['parts'] = [str(part) for part in parts]
        seconds = (sum(durations[str(part)] for part in parts)+3)/1.02
        make(target, seconds)
        return seconds, 1.02

    def capture_captions(path, cues):
        written_cues.extend(copy.deepcopy(cues))
        write_srt(path, cues)

    def embed(path, chapters, total, work):
        embedded.extend(copy.deepcopy(chapters))
        records['final_duration'] = total

    def thumbnails(trailers, choices, phrases, output, **kwargs):
        for i in range(1, 4):
            Image.new('RGB', (1280, 720), '#234567').save(output / f'thumbnail-{i:02}.jpg')

    def reference_for(selected_narrator, root):
        records['narrator'] = selected_narrator
        return reference

    def label_image(path, title, rank=None, year=None):
        records.setdefault('rank_labels', []).append({'title': title, 'rank': rank, 'year': year})
        path.write_bytes(b'label')

    monkeypatch.setattr(cli, 'inspect_sources', lambda paths: [{"file": p.name, "width": round((source_height if i==0 else 2160)*16/9), "height": source_height if i==0 else 2160} for i,p in enumerate(paths)])
    monkeypatch.setattr(cli, 'voice_reference', reference_for)
    monkeypatch.setattr(cli, 'duration', lambda path: durations[str(path)])
    monkeypatch.setattr(cli, 'require_disk', Mock())
    monkeypatch.setattr(cli, 'ffmpeg', Mock())
    monkeypatch.setattr(cli, 'validate_package', Mock())
    monkeypatch.setattr(cli, 'write_srt', capture_captions)
    monkeypatch.setattr(research, 'Research', FakeResearch)
    monkeypatch.setattr(availability, 'enrich_availability', enrich)
    monkeypatch.setattr(footage, 'download', download)
    monkeypatch.setattr(secondary_footage, 'download_secondary', secondary)
    monkeypatch.setattr(script, 'Writer', FakeWriter)
    monkeypatch.setattr(voice, 'Narrator', FakeNarrator)
    monkeypatch.setattr(voice, 'captions', captions)
    monkeypatch.setattr(voice, 'spoken_coverage', lambda *args: 1)
    monkeypatch.setattr(shots, 'analyse', lambda *args: [{'start': 30, 'end': 90, 'motion': 8}])
    monkeypatch.setattr(shots, 'ShotSelector', FakeSelector)
    monkeypatch.setattr(shots, 'intro_montage', montage)
    monkeypatch.setattr(memes, 'prepare_intro_meme', lambda *args: None)
    monkeypatch.setattr(render, 'speech_video', speech_video)
    monkeypatch.setattr(render, 'trailer_video', trailer_video)
    monkeypatch.setattr(render, 'label_image', label_image)
    monkeypatch.setattr(render, 'subscribe_image', Mock(side_effect=AssertionError('The old static CTA card must not render')))
    monkeypatch.setattr(render, 'finish', finish)
    monkeypatch.setattr(render, 'embed_chapters', embed)
    monkeypatch.setattr(render, 'thumbnails', thumbnails)
    monkeypatch.setitem(sys.modules, 'movie_pipeline.previews', SimpleNamespace(
        cold_open_plan=cold_open_plan, choose_preview=choose_preview,
        outro_recap_plan=lambda selector, pools, shots, lengths, text, action: montage(
            selector, [pool[0] for pool in pools], shots, lengths, text, action),
    ))
    monkeypatch.setitem(sys.modules, 'movie_pipeline.cta', SimpleNamespace(
        prepare_cta=prepare_cta, render_cta=render_cta,
    ))
    monkeypatch.setitem(sys.modules, 'faster_whisper', SimpleNamespace(WhisperModel=Mock()))
    output = tmp_path / 'output'
    brief_voice = 'Use female audio.' if narrator == 'Warguy' else 'Use Warguy.'
    cli.produce(f'Top {count} Action Movies', 'Make a 5-minute video. '+brief_voice,
                '; '.join(f"{film['title']} ({film['year']})" for film in films), tmp_path / 'jobs', output)

    assert records['selected_kind'] == (expected_kind if content_type else None)
    assert records['preferred_kind'] == expected_kind
    assert records['narrator'] == expected_narrator
    saved_input = json.loads(next((tmp_path / 'jobs').glob('*/input.json')).read_text())
    assert saved_input['content_type'] == (expected_kind if content_type else 'auto')
    assert saved_input['narrator'] == expected_narrator
    # Cold opening and title previews add intentional screen time to the old plan.
    added_visual_seconds = 5 + count*3.5
    assert 285 <= records['final_duration']-added_visual_seconds/1.02 <= 315
    assert records['writes'] == (2 if extra_mode == "rewrite-fails" else 1)
    if extra_mode == 'shared-audio-retry':
        assert records['shared_transcriptions'] == 2  # one failed shared file, then its replacement
        assert records['shared_regenerations'] == 1  # reused by every repeated paragraph
    assert records['extra_attempts'] == count
    assert records["footage_order"] == [("primary",film["id"]) for film in films]+[("secondary",film["id"]) for film in films]
    if extra_mode == "render-fails":
        assert records['render_fallbacks'] == count
    assert len(json.loads(next((tmp_path / 'jobs').glob('*/quality-report.json')).read_text())['sources']) == count*(2 if extra_mode == 'found' else 1)
    assert render_sizes and set(render_sizes) == {expected_size}
    quality_report = json.loads(next((tmp_path / "jobs").glob("*/quality-report.json")).read_text())
    assert quality_report["try_4k"] == (try_4k == "true")
    assert (quality_report["export_width"], quality_report["export_height"]) == expected_size
    assert records['cta_render']['plan']['animated'] is True
    assert records['cta_render']['plan']['path'].endswith('.gif')
    assert len(records['review_previews']) == count
    assert records['rank_labels'] == [
        {'title': f'Movie {i+1}', 'rank': i+1, 'year': 2025} for i in reversed(range(count))
    ]
    timeline_file = next((tmp_path / 'jobs').glob('*/timeline.json'))
    timeline = json.loads(timeline_file.read_text())
    events = timeline['events']
    assert events[0]['type'] == 'cold-open'
    assert events[0]['start'] == 0
    assert sum(event['duration'] for event in events if event['type'] == 'cold-open') == pytest.approx(5)
    assert records['parts'][0] == records['trailer_renders'][0]['target']
    assert records['trailer_renders'][0]['muted'] is True
    assert min(cue['start'] for cue in written_cues) == pytest.approx(5/timeline['speed'])
    assert written_cues[0]['text'] == first
    previews = [event for event in events if event['type'] == 'review-preview']
    assert len(previews) == count
    assert all(event['duration'] == pytest.approx(3.5) for event in previews)
    labelled = [event for event in records['trailer_renders'] if event.get('label') is not None]
    assert len(labelled) == count
    assert all(event['label_seconds'] == 3.5 for event in labelled)
    ctas = [(i, event) for i, event in enumerate(events) if event.get('type') == 'subscribe']
    assert len(ctas) == 1
    index, cta = ctas[0]
    raw_duration = records['final_duration']*timeline['speed']
    assert .30 <= cta['start']/raw_duration <= .50
    assert cta['duration']/timeline['speed'] < 10
    assert events[index-1]['film'] != events[index+1]['film']
    assert events[index-1]['text'].startswith("It's listed")  # preceding review completed
    assert events[index+1]['type'] == 'review-preview'  # next review begins after the CTA
    assert events[index+2]['text'].startswith('Review number')
    assert [chapter['title'] for chapter in embedded] == ['Intro',
            *[f"#{i+1} Movie {i+1} (2025)" for i in reversed(range(count))], 'Outro']
    assert all(a['start'] < b['start'] for a, b in pairwise(embedded))
    assert not any('subscribe' in chapter['title'].casefold() for chapter in embedded)
    assert embedded == timeline['chapters']
    assert embedded[0]['start'] == 0
    for preview, chapter in zip(previews, embedded[1:-1]):
        assert chapter['start'] == pytest.approx(preview['start']/timeline['speed'])
        following = next(event for event in events if event['type'] == 'narration'
                         and event.get('section_kind') == 'film' and event['film'] == preview['film'])
        assert following['start'] == pytest.approx(preview['start']+3.5)
        cue = next(cue for cue in written_cues if cue['text'] == following['text'])
        assert cue['start'] == pytest.approx(following['start']/timeline['speed'])
    cta_cue = next(cue for cue in written_cues if cue['text'] in MIDROLLS)
    assert cta_cue['start'] == pytest.approx(cta['start']/timeline['speed'])
    assert cta_cue['end']-cta_cue['start'] < 10
    assert all(a['end'] <= b['start']+.001 for a,b in pairwise(written_cues))
    assert (records['final_duration']-written_cues[-1]['end'])*timeline['speed'] == pytest.approx(3)
    description = (output / 'description.txt').read_text()
    assert description.startswith(first+'\n\n'+second+'\n\n')
    assert '00:00 Intro' in description
    assert ('#ActionShows' if expected_kind == 'tv' else '#ActionMovies') in description
    if extra_mode == 'found':
        assert 'https://example.org/extra/1' in description
        assert any('-extra.mp4' in shot.get('source','') for event in events for shot in event.get('shots',[]))
    else:
        assert 'https://example.org/extra/' not in description
    assert ('Featured shows / series' if expected_kind == 'tv' else 'Featured movies') in description
    assert 'Stream: Netflix; Rent: Amazon Video' in description
    assert 'Film information: TMDB' not in description
    assert any('Netflix' in event.get('text', '') for event in events)
    assert all('US' not in event.get('text', '') for event in events)
    assert (output / 'subtitles.srt').exists()
    for i in range(1, 4):
        with Image.open(output / f'thumbnail-{i:02}.jpg') as thumbnail:
            assert thumbnail.size == (1280, 720)
    assert not list(timeline_file.parent.glob('part-*.mp4'))


def test_media_and_voice_dropdowns_isolate_cached_jobs(monkeypatch, tmp_path):
    """Identical text cannot restore another medium's research or another voice's speech."""
    monkeypatch.setenv('VIDEO_MINUTES', '5')
    monkeypatch.setenv('TRY_4K', 'false')
    reference = tmp_path / 'same-reference.ogg'
    reference.write_bytes(b'same reference bytes for every selected voice')
    monkeypatch.setattr(cli, 'voice_reference', lambda *args: reference)
    selected_jobs = []

    class StopAfterCacheSelection:
        def collect(self, items, path):
            selected_jobs.append((self.selected_kind, path.parent))
            raise RuntimeError('Stop after cache selection')

    monkeypatch.setattr(research, 'Research', StopAfterCacheSelection)
    selections = [('Movies', 'Warguy'), ('Shows', 'Warguy'),
                  ('Movies', 'Kit Laser'), ('Movies', 'Female'), ('Movies', 'Warguy')]
    for content_type, narrator in selections:
        monkeypatch.setenv('VIDEO_TYPE', content_type)
        monkeypatch.setenv('NARRATOR', narrator)
        with pytest.raises(RuntimeError, match='Stop after cache selection'):
            cli.produce('Top 2 Movies', 'Use Warguy. Keep it casual.',
                        'First (2025); Second (2025)', tmp_path / 'jobs', tmp_path / 'output')

    assert [kind for kind, _ in selected_jobs] == ['movie', 'tv', 'movie', 'movie', 'movie']
    paths = [path for _, path in selected_jobs]
    assert len(set(paths[:4])) == 4
    assert paths[0] == paths[-1]
