# Developer code graph

This optional Graphify trial maps tracked source relationships for developers.
It is independent of video production, takes no provider key, and uses no
AI backend. It does not analyze reference videos or guarantee a complete
call graph. Dynamic calls, inferred links, and stale graphs still require
checking source, logs, tests, and the actual output.

From Actions choose **Developer code graph** and run it on the desired branch.
The downloadable artifact includes `graphify-out/graph.json`, two sample
queries, and a manifest with the source commit, included files and tool version.
Queries that find no match are reported rather than presented as proof. The
workflow uses a separate temporary corpus containing only tracked Python, shell,
YAML and TOML files from selected source/test/workflow directories. It omits
runtime files, video/audio, credentials and untracked files; inspect tracked
source for accidentally embedded secrets before sharing a graph.

For a local experiment, use a separate virtual environment and the same pinned
version (the production environment and its dependencies need not change):

```sh
python3 -m venv /tmp/movie-ranks-graphify
/tmp/movie-ranks-graphify/bin/python -m pip install graphifyy==0.9.102
/tmp/movie-ranks-graphify/bin/python -m scripts.graphify_trial
```

The default output is `work/code-graph/`, ignored by Git. Use a fresh `--out`
for each experiment; the script will not overwrite an existing directory.
For further questions, run Graphify's `query`, `explain` or `path` commands
with `--graph work/code-graph/graphify-out/graph.json`. Validate useful paths
against the source code; the graph is a map, not a test result. Rebuild it after
changes rather than assuming it reflects the latest commit.
