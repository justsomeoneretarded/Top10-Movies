# Top10 Movies

An unattended, CPU-based GitHub Actions pipeline: enter a video title, creative brief, and movie/show names with years. Receive a 720p countdown at your requested duration (8–10 minutes by default), uploadable subtitles, description, recommended title, and three thumbnails.

**No paid inference APIs.** Scriptwriting uses a local Ollama model; narration uses Chatterbox; shot matching uses CLIP; captions use faster-whisper. GitHub account usage/storage limits still apply. This is an initial implementation, not a guarantee of editorial quality or successful downloads for every title. End-to-end production must be verified on your runner with your configured secrets.

## One-time setup

1. Obtain an **API Read Access Token** from [TMDB API settings](https://www.themoviedb.org/settings/api). Add it as repository Actions secret **`TMDB_READ_TOKEN`** in **Settings → Secrets and variables → Actions**. Use the long bearer/read token, not the short API key. TMDB attribution is included in descriptions; commercial use may require separate terms from TMDB.
2. The three narrator reference clips are already included in `assets/voices/`. **No voice secrets or audio uploads are needed.**
3. Enable Actions if GitHub asks. No paid LLM/TTS/image accounts are needed.

| Narrator | Included reference | Selection in creative brief |
|---|---|---|
| Warguy | `assets/voices/warguy.ogg` | Default, or “Use Warguy” / “Use male audio” |
| Kit Laser | `assets/voices/kit-laser.ogg` | “Use Kit Laser” |
| Female | `assets/voices/female.ogg` | “Use the female voice” |

These are public 12-second reference excerpts starting five seconds into each of the three recordings supplied by the owner, published with their approval. The original full-length recordings are not required for production. The clips are mono Opus files, normalized for cloning. Voice similarity has not been subjectively auditioned here.

To replace a reference, use `python scripts/prepare_voice.py NAME /path/to/audio.mp3 --start 20 --seconds 12`, then commit only the resulting `.ogg` file. Choose clear speech by one permitted speaker without music. Do not commit `.secrets.env` files or API tokens. Optional private base64 references remain supported when no local file exists; they are not needed for this repository.

## Generate a video

Open **[Actions → Generate Video](https://github.com/justsomeoneretarded/Top10-Movies/actions/workflows/generate-video.yml) → Run workflow**.

There are exactly three inputs:

```text
Title:
Top 5 Mind-Bending Science Fiction Movies

Description / creative brief:
Make a cinematic, spoiler-light countdown with a strong opening hook.
Use Kit Laser. Build tension, feature striking visuals, and let brief
trailer dialogue play between narration sections. Make the thumbnail
about mystery and impossible worlds. Avoid exaggerated claims.

Movies:
Arrival (2016); Inception (2010); Coherence (2013); Interstellar (2014); Predestination (2014)
```

List titles **best first** (#1 first); the rendered countdown plays from #5 down to #1. Use semicolons or newlines. Both movies and shows are resolved automatically; provide a show's first-release year. Supports 2–15 titles. A “Top N” title must match the list length. Upcoming titles work when TMDB has their release year, synopsis, and a usable trailer; the writer is instructed to avoid presenting them as watched/released.

No narrator instruction means **Warguy**. “Male” alone also means Warguy. Female is selected only by an explicit voice instruction; conflicting narrator choices fail early.

When the run succeeds, download **Artifacts → movie-video-N**. The ZIP contains:

```text
final-video.mp4
subtitles.srt
description.txt
recommended-title.txt
thumbnail-01.jpg
thumbnail-02.jpg
thumbnail-03.jpg
```

Upload the SRT separately in YouTube Studio. Captions cover narration and retained intelligible trailer dialogue; captions and chapter times are adjusted if final audio tempo changes. They are speech-recognized and can contain transcription errors. Subtitles are not burned into the picture. Artifacts expire after seven days.

## Production stages

1. Validate inputs and the selected voice before downloading models.
2. Resolve title/year against TMDB movies and TV. If the search year differs, check dated festival/regional release records. If still unresolved, verify an exact title and publication year through Wikidata, then map its IMDb ID back to TMDB. Record the evidence; never accept a nearby year without a supporting date. Reject genuinely ambiguous matches.
3. Discover TMDB-linked YouTube trailers/teasers. If TMDB has no candidates, search YouTube and require verified-channel metadata plus a matching title. Reject obvious fan/concept/reaction videos.
4. Download with yt-dlp and FFmpeg, then cache footage. No user-supplied URLs or footage are required. Only accessible non-DRM footage is supported.
5. Generate source-grounded, original recommendation scripts with `qwen3:1.7b` through local Ollama. Use word counts as planning guidance, not hard paragraph failure limits; no fabricated ratings, streaming availability, or awards are requested.
6. Clone the selected reference with **Chatterbox Turbo on CPU**. Generate speech in small chunks and retry faulty speech. Up to two overall script rewrites adapt the content to the requested runtime using measured narration pace.
7. Transcribe narration and trailer dialogue with CPU/int8 faster-whisper. Reject substantially missing/repeated narration and automatically regenerate bad segments, with bounded retries.
8. Sample scene changes and motion, embed frames/text with CLIP, and choose relevant shots with reuse and original-sequence penalties. Build the intro from multiple titles, covering at least half (rounded up). Action-oriented intros favor action-related imagery and short cuts.
9. Render reordered short clips, occasional brief freeze-frame accents, rank graphics, short trailer-audio breaks, and an original procedural instrumental bed. Narration stops during trailer breaks; source audio is removed under narration. Music ducks under speech.
10. Export H.264/AAC at 1280×720/30fps with normalized audio. If needed, use modest pitch-preserving tempo adjustment; stop if the requested duration still cannot be reached naturally after automatic rewrites.
11. Render three distinct frame-based thumbnail layouts with readable typography. There is no paid image generation; no diffusion model is required.
12. Validate required outputs, media streams, duration, thumbnail dimensions/size, and a full FFmpeg decode before uploading the package.

No approval or publication steps occur. Successful runs deliver downloadable files; this repository does not upload to YouTube.

## CPU performance and reliability

Turbo is the quality-oriented default, **not a benchmarked speed promise**. To use the smaller CPU-focused model, set repository **variable** `TTS_MODEL` to `nano`. Both use the same voice reference, and the pipeline never switches voices silently.

Public and private GitHub-hosted runners have different resource allowances. The workflow has a 350-minute timeout; first runs download several GB of models and dependencies. Model and job caches accelerate later runs, but GitHub may evict them. Free storage/minute quotas are finite; inspect your account's billing settings. Standard GitHub runners do not provide a free GPU. A failed/timed-out run delivers no partial package. Re-running a failed job can restore checkpoints saved by the failure handler; runner cancellation/timeouts may prevent cache saving.

Some upstream model/tool dependencies are pinned, while yt-dlp remains updateable to track changing sites. Ollama's installer and model tag track their upstream distribution. For strict reproducibility, pin their versions/model digests after a verified production run. Do not expect the first unbenchmarked run to have a guaranteed completion time.

YouTube can block data-center downloads or require authentication. Neither yt-dlp nor this workflow guarantees access. It does not bypass DRM or bot challenges. The pipeline stops clearly when required footage is missing instead of substituting a different film. A self-hosted runner is an alternative if hosted-runner access is unreliable. Publicly visible trailers are copyrighted; downloading and using them must comply with source terms and applicable rights. Short clips and commentary do not guarantee clearance or prevent Content ID claims.

Automated shot matching, trailer dialogue selection, factual accuracy, and narration quality remain imperfect. CLIP matching is a heuristic; the system does not understand every scene or guarantee spoiler avoidance. The three thumbnails are options, not tested CTR winners. The implementation uses clean cuts and rank graphics; it does not claim to reproduce a professional human editor's judgment.

## Local development

```bash
python3.11 -m venv .venv
source .venv/bin/activate
pip install -e '.[test]'
pytest -q

# Full production also needs FFmpeg, fonts, Ollama, and CPU speech dependencies.
pip install torch==2.6.0 torchaudio==2.6.0 --index-url https://download.pytorch.org/whl/cpu
pip install -e '.[voice]'
ollama serve
# In another terminal:
ollama pull qwen3:1.7b
export TMDB_READ_TOKEN='your-read-token'
# Narrator reference clips are already included in assets/voices/.
movie-pipeline --title 'Top 2 Sci-Fi Movies' \
  --brief 'Cinematic and spoiler-light. Use Warguy.' \
  --movies 'Arrival (2016); Interstellar (2014)'
```

Never put API tokens or voice base64 in workflow inputs. Inputs are visible in run metadata. The tests exercise input parsing, narrator defaults/negation, subtitles, trailer filtering, workflow safety, and actual FFmpeg rendering/thumbnail generation without downloading AI models or copyrighted media.

## YouTube sign-in/bot errors

If research succeeds but downloads say “Sign in to confirm you’re not a bot”, YouTube is rejecting the runner/session. It is not a TMDB-token or movie-list error. Switching video IDs or downloader forks does not reliably resolve it.

Optional authenticated access:

1. Export your YouTube session cookies as a **Netscape-format `cookies.txt`**, following [yt-dlp's current export guidance](https://github.com/yt-dlp/yt-dlp/wiki/Extractors#exporting-youtube-cookies). Use a trusted local export method; treat the export as account credentials. Do not share it in chat, issues, or public repository files.
2. Add a repository Actions **secret** named `YOUTUBE_COOKIES`. Paste the file's complete contents, not a filename, JSON, or base64. The secret must fit GitHub's 48 KB limit; export YouTube cookies only.
3. Start a **new Run workflow on main** so it uses the updated workflow. Re-running an older run uses that run's old workflow revision.

The workflow writes cookies to a permission-restricted file under `RUNNER_TEMP`, outside footage/model caches and output artifacts, and removes the file afterward. Cookies remain private even though narrator reference clips are public. Use authenticated downloads only where permitted by the source's terms and applicable rights.

**This is an authentication option, not a guaranteed fix or CAPTCHA bypass.** Sessions expire, and GitHub-hosted IP addresses may remain blocked even with valid cookies. In that case a self-hosted runner on a network where normal permitted access works, or another authorized footage source, is required. There is no dependable promise of unattended, free YouTube downloading forever from GitHub-hosted runners.

### Release-year differences and independent identity lookup

A title may premiere at a festival in one year and reach theaters in the next. The resolver preserves the year you entered and checks supporting release dates before rejecting it. If needed, a keyless Wikidata lookup checks the English label/aliases, publication-year claims, and IMDb identifier. TMDB's IMDb-ID lookup then retrieves the same film's metadata and trailer references. This is an independent **identity/year check**, not scraping IMDb reviews or plots. Synopses and trailer references still come from TMDB. Source outages and incomplete claims can still prevent verification; logs report that without assuming the input is wrong. No additional account or API secret is required.

## Request a different video length

Include one runtime instruction in the description/creative brief:

- `Make a 5-minute video. Use Warguy.`
- `Aim for 12–15 minutes. Use Kit Laser.`
- `Duration: 90 seconds. Use the female voice.`

No code changes or additional workflow input is required. If duration is omitted, the default remains 8–10 minutes. Single durations allow about 5% variation; explicit ranges use their stated bounds. Use one duration instruction and remove conflicting old instructions. Durations below 30 seconds are not supported by this countdown format.

The script budget scales with runtime, movie count, and planned trailer-audio pauses. Individual narration sections can vary in word count without aborting the job. Only empty/unusable narration is rejected after retries. Oversized paragraphs are split into smaller TTS chunks without deleting words. After narration, actual speech duration drives up to two whole-script adjustments; final modest tempo correction and duration validation use the requested runtime. Chapter and subtitle timing follow any final tempo change.

This does not promise unlimited video lengths on free hardware: longer videos require more generation/render time and storage, and the GitHub job timeout and account quotas still apply. No live model run has yet verified this duration feature; automated tests cover parsing, script-budget scaling, and real media exports.

## Intro montage and trailer editing

The opening is a montage drawn from at least half the supplied movies/shows, rounded up: for five titles, at least three appear. Where the intro has room, all titles can appear. Clips alternate between movies; they are selected from anywhere in the usable trailer footage rather than replaying trailer openings. Typical intro cuts are around 1–2 seconds, adjusted to the narration length and required title coverage. For action topics, semantic action prompts and motion scores favor fights, chases, stunts, and similar imagery. These are automated relevance estimates, not a guarantee of the film's best action scenes; only downloaded trailer footage is available.

Movie sections also use selected shots in a new order, with penalties against repeating a shot or continuing the trailer's original sequence when alternatives exist. Brief freeze-frame accents appear occasionally. Freeze holds replace part of a visual slot rather than extending it, so narration, subtitles, chapter times, and requested runtime remain synchronized. Intelligible trailer-dialogue inserts retain their selected short audio/video context.

Editing is for pacing and original commentary. Cuts, shuffling, frame holds, overlays, or music do not by themselves establish fair use, prevent Content ID claims, or guarantee YouTube monetization. Rights and YouTube's reused-content assessment are separate from the edit effects.

### Voice speed and failed-run recovery

Speaking-rate measurements are now logged as diagnostics. A fast or slow delivery alone does not reject a segment: the transcription stage checks whether the words were actually spoken, and final duration control handles overall pacing. Voice synthesis still rejects empty, silent, non-finite, or extremely long/runaway output after bounded retries.

A new workflow run can restore the latest earlier production checkpoint cache, including footage and completed narration, when GitHub retains it. Each job's research/script/audio paths are keyed by its inputs and selected voice; different inputs do not reuse the wrong script or narrator. Reuse is best effort, depends on the restored cache containing the matching job, and does not guarantee zero repeated work. Neither cookies nor voice secrets are included in this cache.
