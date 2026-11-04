"""Build an optional local, code-only Graphify map from tracked project sources.

The corpus excludes untracked files, runtime output, media, and environment files.
Graphify is a separate developer dependency and is never imported by production.
"""
from __future__ import annotations

import argparse
import importlib.metadata
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile

SOURCE_DIRS = ("movie_pipeline/", "scripts/", "tests/", ".github/workflows/")
SOURCE_SUFFIXES = {".py", ".sh", ".yml", ".yaml", ".toml"}
QUERIES = (
    ("Narration, subtitle timing, and export", "voice narration subtitles timing export"),
    ("Footage, shot selection, and editing", "footage source_pool shots previews editing render"),
)


def stage_tracked_sources(root: Path, corpus: Path) -> list[str]:
    """Copy selected tracked text source files without following symlinks."""
    root = root.resolve()
    raw = subprocess.check_output(["git", "-C", str(root), "ls-files", "-z"])
    copied = []
    for item in raw.split(b"\0"):
        if not item:
            continue
        name = os.fsdecode(item)
        relative = Path(name)
        if relative.is_absolute() or ".." in relative.parts:
            continue
        if not (name.startswith(SOURCE_DIRS) or name == "pyproject.toml"):
            continue
        if relative.suffix.lower() not in SOURCE_SUFFIXES:
            continue
        lowered = relative.name.lower()
        if lowered.startswith(".env") or "secret" in lowered or "credential" in lowered:
            continue
        source = root / relative
        # Also reject symlinked parent directories and paths outside the checkout.
        if source.is_symlink() or any((root / Path(*relative.parts[:i])).is_symlink()
                                      for i in range(1, len(relative.parts))):
            continue
        if not source.is_file() or not source.resolve().is_relative_to(root):
            continue
        target = corpus / relative
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(source, target)
        copied.append(relative.as_posix())
    return sorted(copied)


def build_graph(root: Path, output: Path) -> None:
    root, output = root.resolve(), output.resolve()
    # Never clobber an existing artifact; make each experiment reviewable.
    output.mkdir(parents=True, exist_ok=False)
    version = importlib.metadata.version("graphifyy")
    env = os.environ.copy()
    env["GRAPHIFY_BACKEND"] = "none"
    env["GRAPHIFY_QUERY_LOG_DISABLE"] = "1"
    with tempfile.TemporaryDirectory(prefix="movie-ranks-code-graph-") as temp:
        corpus = Path(temp) / "corpus"
        corpus.mkdir()
        included = stage_tracked_sources(root, corpus)
        if not included:
            raise RuntimeError("No tracked source files were selected for the graph.")
        commit = subprocess.check_output(["git", "-C", str(root), "rev-parse", "HEAD"],
                                         text=True).strip()
        command = [sys.executable, "-m", "graphify", "extract", str(corpus),
                   "--code-only", "--no-cluster", "--out", str(output)]
        subprocess.run(command, check=True, env=env, timeout=600)
        graph = output / "graphify-out" / "graph.json"
        payload = json.loads(graph.read_text(encoding="utf-8"))
        nodes = payload.get("nodes", [])
        edges = payload.get("edges", payload.get("links", []))
        if not nodes or not edges:
            raise RuntimeError("The code graph was empty; check the extraction output.")
        print(f"Graph contains {len(nodes)} nodes and {len(edges)} edges.", flush=True)
        evidence = []
        for label, question in QUERIES:
            result = subprocess.run([sys.executable, "-m", "graphify", "query", question,
                                     "--graph", str(graph), "--budget", "1500"],
                                    env=env, text=True, capture_output=True, timeout=60)
            evidence.append(f"## {label} (exit {result.returncode})\n"
                            f"{result.stdout}\n{result.stderr}\n")
            print(evidence[-1], flush=True)
        (output / "query-evidence.txt").write_text("\n".join(evidence), encoding="utf-8")
        (output / "manifest.json").write_text(json.dumps({
            "repository_commit": commit, "graphify_version": version,
            "included_files": included, "nodes": len(nodes), "edges": len(edges),
            "backend": "none", "scope": "tracked-code-only",
        }, indent=2) + "\n", encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=Path.cwd())
    parser.add_argument("--out", type=Path, default=Path("work/code-graph"))
    args = parser.parse_args()
    build_graph(args.root, args.out)


if __name__ == "__main__":
    main()
