"""The optional developer graph must never collect runtime or untracked files."""
from pathlib import Path
import subprocess

from scripts.graphify_trial import stage_tracked_sources


def test_corpus_is_scoped_to_tracked_code_and_workflows(tmp_path):
    root = tmp_path / "checkout"
    root.mkdir()
    subprocess.run(["git", "init", "-q", str(root)], check=True)
    files = {
        "movie_pipeline/voice.py": "def speak(): pass\n",
        "tests/test_voice.py": "def test_it(): pass\n",
        ".github/workflows/production.yml": "name: production\n",
        "scripts/prepare_cookies.py": "def prepare(): pass\n",
        "scripts/.env.py": "SECRET = 'not a real secret'\n",
        "scripts/reference.mp3": "fake media",
        "work/private.py": "private runtime file",
        "output/result.py": "output not code",
        ".env": "TOKEN=not-a-real-token",
    }
    for name, body in files.items():
        path = root / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(body)
    subprocess.run(["git", "-C", str(root), "add", "-f", "--", *files], check=True)
    (root / "scripts" / "untracked.py").write_text("untracked")
    target = tmp_path / "corpus"
    target.mkdir()
    included = stage_tracked_sources(root, target)
    assert set(included) == {
        "movie_pipeline/voice.py", "tests/test_voice.py",
        ".github/workflows/production.yml", "scripts/prepare_cookies.py",
    }
    assert not (target / "work").exists()
    assert not (target / "scripts" / ".env.py").exists()
    assert not (target / "scripts" / "untracked.py").exists()


def test_corpus_skips_tracked_symlinks(tmp_path):
    root = tmp_path / "checkout"
    root.mkdir()
    subprocess.run(["git", "init", "-q", str(root)], check=True)
    (root / "scripts").mkdir()
    outside = tmp_path / "external.py"
    outside.write_text("private")
    (root / "scripts" / "link.py").symlink_to(outside)
    subprocess.run(["git", "-C", str(root), "add", "--", "scripts/link.py"], check=True)
    target = tmp_path / "corpus"
    target.mkdir()
    assert stage_tracked_sources(root, target) == []
