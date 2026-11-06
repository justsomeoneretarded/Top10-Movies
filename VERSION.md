# MOVIE RANKS — historical version

This repository runs the exact production code, workflows, tests and bundled assets from commit [`c6f82a62`](https://github.com/justsomeoneretarded/Top10-Movies/commit/c6f82a621b9de57a2c180a74cd9e37e17f2a6a92), immediately before Graphify was introduced.

Use **Generate Video** on the `main` branch in [Actions](https://github.com/justsomeoneretarded/Top10-Movies/actions/workflows/generate-video.yml).

The improved version is independently runnable in [Top10-Movies-2.0-ish](https://github.com/justsomeoneretarded/Top10-Movies-2.0-ish) using **Generate Video(2.0 -ish)**. It preserves improved source commit `0e2f4e9b4aafd46a73ede9622378f423f861b0f0`.

The historical tree was restored in a forward commit. Existing Git history, prior run records and repository caches are retained. The previous improved state is also preserved at branch `archive/improved-before-version-split-2026-11-06`.

For a comparison, enter the same title, brief, titles, narrator and duration in each repository. Discovery may accept different footage, and generated writing/speech can vary. Each repository owns its own secrets, caches and output artifacts; the new repository starts with cold model caches.

This version deliberately keeps its historical behavior and technical safeguards. Later fixes have not been silently backported. This document is the only addition to the historical source tree.
