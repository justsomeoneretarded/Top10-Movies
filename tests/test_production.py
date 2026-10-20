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
    shots,
    voice,
)
from movie_pipeline.core import key, save_json, write_srt
from movie_pipeline.packaging import MIDROLLS
from movie_pipeline.video_settings import export_size


@pytest.mark.parametrize("count", [2, 5])
def test_produce_keeps_cta_between_reviews_and_retimes_metadata(monkeypatch, tmp_path, count):
    monkeypatch.chdir(tmp_path)
    monkeypatch.setenv("VIDEO_RESOLUTION", "1080")
    monkeypatch.setenv("SCRIPT_MODEL", "qwen3:4b")
    records = {}
    durations = {}
    text_for_audio = {}
    render_sizes = []
    written_cues = []
    embedded = []
    first = "Stop scrolling. Your next favorite movie might be here."
    second = "Two minutes of scrolling saved. Let's find your next watch."
    films = [{"title": f"Movie {i+1}", "year": 2025, "id": i+1, "kind": "movie",
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
            elif text in MIDROLLS:
                seconds = 6
            else:
                seconds = 20
            path = make(self.cache / (key(text)+".wav"), seconds)
            text_for_audio[str(path)] = text
            return path

        def close(self):
            pass

    def download(film, root):
        path = make(root / f"movie-{film['id']}.mp4", 150)
        save_json(path.with_suffix('.json'), {'source': f"https://example.org/trailer/{film['id']}"})
        return path

    def captions(audio, recognizer):
        if str(audio) in text_for_audio:
            return [{"start": 0, "end": durations[str(audio)], "text": text_for_audio[str(audio)]}]
        return [{"start": 30, "end": 32.9, "text": "We have to leave right now"}]

    class FakeSelector:
        def __init__(self):
            self.people = {}

        def index(self, path, choices):
            pass

        def choose(self, path, choices, text, seconds, action, **kwargs):
            return [{"start": 40, "duration": seconds, "freeze": 0}]

    def montage(selector, trailers, choices, lengths, brief, action):
        return [[{"start": 40, "duration": seconds, "freeze": 0,
                  "source": str(trailers[i % count]), "film_index": i % count}]
                for i, seconds in enumerate(lengths)]

    def speech_video(source, audio, selected, target, label, work, **kwargs):
        render_sizes.append(export_size())
        text = text_for_audio[str(audio)]
        if text in MIDROLLS:
            records['cta_plan'] = copy.deepcopy(selected)
            assert kwargs['label_seconds'] == durations[str(audio)]
        make(target, durations[str(audio)])

    def trailer_video(source, start, seconds, target):
        make(target, seconds)

    def finish(parts, target, work, action, spec):
        records['ending_hold'] = 3
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

    monkeypatch.setattr(cli, 'voice_reference', lambda *args: reference)
    monkeypatch.setattr(cli, 'duration', lambda path: durations[str(path)])
    monkeypatch.setattr(cli, 'require_disk', Mock())
    monkeypatch.setattr(cli, 'ffmpeg', Mock())
    monkeypatch.setattr(cli, 'validate_package', Mock())
    monkeypatch.setattr(cli, 'write_srt', capture_captions)
    monkeypatch.setattr(research, 'Research', FakeResearch)
    monkeypatch.setattr(availability, 'enrich_availability', enrich)
    monkeypatch.setattr(footage, 'download', download)
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
    monkeypatch.setattr(render, 'label_image', lambda path, *args: path.write_bytes(b'label'))
    monkeypatch.setattr(render, 'subscribe_image', lambda path: path.write_bytes(b'cta'))
    monkeypatch.setattr(render, 'finish', finish)
    monkeypatch.setattr(render, 'embed_chapters', embed)
    monkeypatch.setattr(render, 'thumbnails', thumbnails)
    monkeypatch.setitem(sys.modules, 'faster_whisper', SimpleNamespace(WhisperModel=Mock()))
    output = tmp_path / 'output'
    cli.produce(f'Top {count} Action Movies', 'Make a 5-minute video in 4K. Use male audio.',
                '; '.join(f"{film['title']} ({film['year']})" for film in films), tmp_path / 'jobs', output)

    assert 285 <= records['final_duration'] <= 315
    assert records['writes'] == 1
    assert render_sizes and set(render_sizes) == {(3840, 2160)}
    assert records['cta_plan'][0]['still'] is True
    assert len(records['cta_plan']) == 1
    timeline_file = next((tmp_path / 'jobs').glob('*/timeline.json'))
    timeline = json.loads(timeline_file.read_text())
    events = timeline['events']
    ctas = [(i, event) for i, event in enumerate(events) if event.get('text') in MIDROLLS]
    assert len(ctas) == 1
    index, cta = ctas[0]
    raw_duration = records['final_duration']*timeline['speed']
    assert .30 <= cta['start']/raw_duration <= .50
    assert cta['duration']/timeline['speed'] < 10
    assert events[index-1]['film'] != events[index+1]['film']
    assert events[index-1]['text'].startswith("It's listed")  # preceding review completed
    assert events[index+1]['text'].startswith('Review number')  # next review has not started
    assert [chapter['title'] for chapter in embedded] == ['Intro',
            *[f"#{i+1} Movie {i+1} (2025)" for i in reversed(range(count))], 'Outro']
    assert all(a['start'] < b['start'] for a, b in pairwise(embedded))
    assert not any('subscribe' in chapter['title'].casefold() for chapter in embedded)
    assert embedded == timeline['chapters']
    cta_cue = next(cue for cue in written_cues if cue['text'] in MIDROLLS)
    assert cta_cue['start'] == pytest.approx(cta['start']/timeline['speed'])
    assert cta_cue['end']-cta_cue['start'] < 10
    assert all(a['end'] <= b['start']+.001 for a,b in pairwise(written_cues))
    description = (output / 'description.txt').read_text()
    assert description.startswith(first+'\n\n'+second+'\n\n')
    assert '00:00 Intro' in description and '#ActionMovies' in description
    assert 'Featured movies' in description and 'Stream: Netflix; Rent: Amazon Video' in description
    assert 'Film information: TMDB' not in description
    assert any('Netflix' in event.get('text', '') for event in events)
    assert all('US' not in event.get('text', '') for event in events)
    assert (output / 'subtitles.srt').exists()
    for i in range(1, 4):
        with Image.open(output / f'thumbnail-{i:02}.jpg') as thumbnail:
            assert thumbnail.size == (1280, 720)
    assert not list(timeline_file.parent.glob('part-*.mp4'))
