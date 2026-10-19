"""Exercise real FFmpeg media assembly without downloading models or movie footage."""
import shutil

import pytest
from PIL import Image

from movie_pipeline.core import duration, ffmpeg, probe
from movie_pipeline.render import (
    label_image,
    music_bed,
    speech_video,
    thumbnails,
    trailer_video,
)
from movie_pipeline.shots import analyse

pytestmark = pytest.mark.skipif(not shutil.which("ffmpeg"), reason="FFmpeg is required")


def test_render_preserves_speech_duration_and_produces_thumbnails(tmp_path):
    source = tmp_path / "source.mp4"
    audio = tmp_path / "voice.wav"
    ffmpeg("-f", "lavfi", "-i", "testsrc2=size=640x360:rate=30", "-f", "lavfi", "-i", "sine=frequency=440:sample_rate=48000", "-t", 6, "-c:v", "libx264", "-preset", "ultrafast", "-c:a", "aac", source)
    ffmpeg("-f", "lavfi", "-i", "sine=frequency=220:sample_rate=48000", "-t", 2.125, audio)
    label = tmp_path / "label.png"
    label_image(label, "Arrival", 1, 2016)
    target = tmp_path / "speech.mp4"
    speech_video(source, audio, [{"start": 1, "duration": 1.1}, {"start": 3, "duration": 1.1}], target, label, tmp_path / "render")
    assert abs(duration(target) - duration(audio)) < 0.09
    assert any(s["codec_type"] == "audio" for s in probe(target)["streams"])
    shots = analyse(source, tmp_path / "shots.json")
    assert shots
    thumbnails([source, source], [shots, shots], ["MIND BENDING", "WATCH THESE", "EXPECT THE UNEXPECTED"], tmp_path)
    for i in range(1, 4):
        path = tmp_path / f"thumbnail-{i:02}.jpg"
        assert Image.open(path).size == (1280, 720)
        assert path.stat().st_size < 2_000_000
    insert = tmp_path / "insert.mp4"
    trailer_video(source, 1, 2.5, insert)
    assert abs(duration(insert) - 2.5) < 0.1
    music_bed(tmp_path / "music.wav", 3.125, True)
    assert abs(duration(tmp_path / "music.wav") - 3.125) < 0.001


@pytest.mark.parametrize("source_seconds, brief", [(480, ""), (330, "Make a 5-minute video")])
def test_final_export_respects_requested_duration_with_valid_audio(tmp_path, source_seconds, brief):
    from movie_pipeline.render import finish
    from movie_pipeline.timing import requested_duration
    spec = requested_duration(brief)
    # A repeated one-second synthetic scene exercises both default and five-minute exports,
    # sidechain music mix, loudness conversion, and final duration policy.
    seed = tmp_path / "seed.mp4"
    ffmpeg("-f", "lavfi", "-i", "color=c=0x16243a:size=1280x720:rate=30", "-f", "lavfi", "-i", "sine=frequency=180:sample_rate=48000", "-t", 1, "-c:v", "libx264", "-preset", "ultrafast", "-c:a", "aac", seed)
    long_source = tmp_path / "long.mp4"
    ffmpeg("-stream_loop", source_seconds-1, "-i", seed, "-t", source_seconds, "-c", "copy", long_source)
    target = tmp_path / "final.mp4"
    seconds, _speed = finish([long_source], target, tmp_path, False, spec=spec)
    assert spec.minimum-0.5 <= seconds <= spec.maximum+0.5
    if brief:
        assert abs(seconds-spec.target) < 0.1
        assert abs(_speed-1.1) < 0.01
    metadata = probe(target)
    audio = next(s for s in metadata["streams"] if s["codec_type"] == "audio")
    assert audio["sample_rate"] == "48000"
    assert audio["channels"] == 2
    ffmpeg("-xerror", "-i", target, "-f", "null", "-")
