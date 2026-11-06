# Quality recovery: November 2026

This revision removes experimental creative overrides introduced by PR #1 while retaining its technical fixes. It is a targeted recovery, not a claim that a quality multiplier has been measured.

## Runtime changes

- Research and the shared episode plan drive writing again. Approximate trailer descriptions, dialogue guesses and preselected scene IDs cannot enter spoken-script facts.
- Keep per-section editing and the final deletion-only episode edit; remove the extra critique/rewrite loop.
- Native moments compete in the ordinary ranking. Early scene-map IDs cannot take priority or reserve the strongest opening footage.
- Retain selected insertion order. Static-image paragraph similarity no longer permutes scenes.
- Internal cut density remains a reported metric, not a generic penalty against interesting action.
- Optional footage discovery tries every configured network route with fresh metadata on each route. Shared metadata/download limits, minimum quality, union novelty checks and reservations remain enforced.
- Rank announcements prefer a clean indexed frame from their verified source, using existing face/card evidence. This does not change the native preview interval or announcement audio.
- Thumbnail rating checks model availability even when scene captions came from warm caches. An optional bounded pull requires working-disk reserve. Errors retain image-based ranking and log exception class/HTTP status without private URLs.
- A new production cache namespace prevents reuse of experimental scripts. Python and model caches remain unchanged.

The scene-map, visual-context and insertion-matching modules remain available as offline utilities; production does not call them.

## Retained protections

Original-shot linking and immediate-continuation rules, face-frame bounds, played-insert barriers, initialism speech verification, serialized downloader construction, final-answer-only vision parsing, caption/chapter timing, audio decode and AV checks, source identity/novelty, bounded retries and cleanup remain active.

## Evidence and limits

Run 42105433308 completed but accepted only five primary footage files; optional downloads logged direct-route transport failures. Its export contact sheet includes a rank reveal over marketing text. Thumbnail vision logged HTTP errors; the old logs do not disclose status, so a missing model is a plausible integration failure rather than a proven diagnosis.

CI validates behavior and technical protections. Sampled frames are not a full playback or listening assessment. Actual scene coherence, delivery and thumbnail quality still require review of a fresh output using comparable inputs. Network route fallback cannot guarantee YouTube availability.
