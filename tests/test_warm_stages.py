import json
from pathlib import Path
from types import SimpleNamespace

from movie_pipeline import cli

FILMS = [{"title": "A", "year": 2020}, {"title": "B", "year": 2021}, {"title": "C", "year": 2022}]


def context(tmp_path):
    return SimpleNamespace(films=FILMS, work=tmp_path / "job", spec=SimpleNamespace(target=300), expected_breaks=40)


class FakeWriter:
    calls = []

    def __init__(self, cache_dir=None):
        self.cache_dir = cache_dir

    def plan(self, *args):
        FakeWriter.calls.append(("plan",))
        return {}

    def write(self, *args, **kwargs):
        FakeWriter.calls.append(("write", kwargs.get("only"), kwargs.get("target_seconds")))
        return {}

    def unload(self):
        pass


def test_prepare_plans_and_reports_parallel_parts(monkeypatch, tmp_path):
    output = tmp_path / "github-output"
    monkeypatch.setenv("GITHUB_OUTPUT", str(output))
    monkeypatch.setattr(cli, "setup_run", lambda *a: context(tmp_path))
    monkeypatch.setattr("movie_pipeline.script.Writer", FakeWriter)
    FakeWriter.calls = []
    parts = cli.warm("prepare", "", "t", "b", "m", tmp_path, tmp_path / "out")
    assert parts["script_parts"] == ["intro", "outro", "film-0", "film-1", "film-2"]
    assert parts["footage_parts"] == ["0", "1", "2"]
    assert ("plan",) in FakeWriter.calls
    lines = dict(line.split("=", 1) for line in output.read_text().splitlines())
    assert json.loads(lines["script_parts"]) == parts["script_parts"]


def test_script_stage_writes_only_its_section(monkeypatch, tmp_path):
    monkeypatch.setattr(cli, "setup_run", lambda *a: context(tmp_path))
    monkeypatch.setattr("movie_pipeline.script.Writer", FakeWriter)
    FakeWriter.calls = []
    cli.warm("script", "film-2", "t", "b", "m", tmp_path, tmp_path / "out")
    assert FakeWriter.calls == [("write", {"film-2"}, 300)]


def test_footage_stage_downloads_analyses_and_describes_one_title(monkeypatch, tmp_path):
    seen = {}
    monkeypatch.setattr(cli, "setup_run", lambda *a: context(tmp_path))
    monkeypatch.setattr("movie_pipeline.footage.download", lambda film, root: seen.setdefault("film", film) and Path("t.mp4"))
    monkeypatch.setattr("movie_pipeline.shots.analyse", lambda path, cache: [{"start": 0, "end": 5}])
    monkeypatch.setattr("movie_pipeline.scenes.describe_trailer", lambda path, shots: seen.setdefault("described", len(shots)))
    monkeypatch.setattr("movie_pipeline.scenes.unload", lambda: None)
    cli.warm("footage", "1", "t", "b", "m", tmp_path, tmp_path / "out")
    assert seen == {"film": FILMS[1], "described": 1}


def test_failed_warm_up_is_a_warning_not_a_crash(monkeypatch, tmp_path, capsys):
    def boom(*args):
        raise RuntimeError("network down")
    monkeypatch.setattr(cli, "setup_run", boom)
    monkeypatch.setattr("sys.argv", ["cli", "--stage", "footage", "--part", "0", "--title", "t", "--brief", "b", "--movies", "m"])
    assert cli.main() == 1
