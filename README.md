# Top10 Movies

An unattended, CPU-based GitHub Actions pipeline: enter a video title, creative brief, and movie/show names with years. Receive a countdown with automatically chosen source-matched resolution, up to 4K, with an Auto or selected video length, uploadable subtitles, description, recommended title, and three thumbnails.

**No paid inference APIs.** Scriptwriting uses a local Ollama model; narration uses Chatterbox; shot matching uses CLIP; captions use faster-whisper. GitHub account usage/storage limits still apply. This is an initial implementation, not a guarantee of editorial quality or successful downloads for every title. End-to-end production must be verified on your runner with your configured secrets.

## One-time setup

1. Obtain an **API Read Access Token** from [TMDB API settings](https://www.themoviedb.org/settings/api). Add it as repository Actions secret **`TMDB_READ_TOKEN`** in **Settings → Secrets and variables → Actions**. Use the long bearer/read token, not the short API key. Source and attribution notes are included in `sources-and-credits.txt`; the requested disclaimer is omitted from the video description. TMDB terms still apply, and commercial use may require separate terms from TMDB.
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

There are five inputs: title, description / creative brief, movies, a **Video length in minutes** selector, and a **Try 4K** checkbox. Length defaults to **Auto**; Try 4K is checked by default. Timing and resolution are selected in the workflow rather than written into the brief.

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

Video length in minutes:
Auto (default), or choose a number such as 8

Try 4K:
Checked (default)
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
sources-and-credits.txt
```

Upload the SRT separately in YouTube Studio. Captions cover narration and retained intelligible trailer dialogue; captions and chapter times are adjusted if final audio tempo changes. They are speech-recognized and can contain transcription errors. Subtitles are not burned into the picture. Artifacts expire after one day. The description includes the first two spoken paragraphs, a plain-language topic summary, 3–5 relevant hashtags, chapters, and a featured movies/shows list. Search-friendly metadata does not guarantee ranking. Chapters are also embedded in the MP4. To get chapters on YouTube, paste the timestamped description into your upload; the workflow does not upload or update YouTube videos.

## Production stages

1. Validate inputs, the video-length selection, the Try 4K setting, and the selected voice before downloading models.
2. Resolve title/year against TMDB movies and TV. If the search year differs, check dated festival/regional release records. If still unresolved, verify an exact title and publication year through Wikidata, then map its IMDb ID back to TMDB. Record the evidence; never accept a nearby year without a supporting date. Reject genuinely ambiguous matches.
3. Discover TMDB-linked YouTube trailers/teasers and bounded YouTube search matches. Prefer an official primary trailer. A second source may be a fan edit of the same movie/show when its title/year checks match; concept, AI-generated, fake, reaction, and review videos are excluded.
4. Fetch primary footage in a first round: download one usable primary trailer for every title before looking for second trailers. Choose the highest shared export resolution up to the checkbox ceiling. In a separate second round, try to add a trailer with different scenes for each title at that resolution. Discovery and retries are bounded; keep at most two trailers per title temporarily in `work/footage/`. One usable trailer is enough, and a missing or rejected second trailer never fails production. No user-supplied URLs or footage are required. Only accessible non-DRM footage is supported.
5. Look up US streaming providers using TMDB's JustWatch-powered watch listings, with a 24-hour cache. Keep subscription, free, ads, rental, and purchase options distinct. Missing or unavailable listings are skipped without failing production.
6. Generate original, factual recommendation scripts with `qwen3:4b` through local Ollama: casual language, a punchy genre-specific hook, clear movie premises, and an outro inviting likes, subscriptions, and other recommendations. Auto length scales the planning budget to the number and kind of titles. Selected minutes allow ±25%; word counts and timing are guidance rather than reasons to discard a usable video.
7. Clone the selected reference with **Chatterbox Turbo on CPU**. Generate speech in small chunks and retry faulty speech. For a selected length, bounded overall script adjustments use measured narration pace. Auto mode avoids chasing a fixed runtime.
8. Transcribe narration and primary-trailer dialogue with CPU/int8 faster-whisper. Verify that narration words were spoken. Place one funny like/subscribe break at a review boundary around 30–50% of measured runtime; shorten its audio to keep it below 10 seconds after final tempo adjustment.
9. Match frames from the accepted source pool loosely to the current narration using CLIP. Use both trailers when useful, retaining steadier, longer visual beats, occasional face holds, and a multi-title intro. Extra footage does not require faster cutting. Weak visual matches never cause a relevance-validation failure.
10. Render rank graphics, the like/subscribe/bell card, short trailer-audio breaks, and an original instrumental bed that ducks under speech. Keep the final image for three seconds after speech, with a gentle fade rather than an abrupt cut.
11. Export H.264/AAC at 30fps, using the highest source-matched resolution shared by all selected trailers, up to 4K when checked or 1080p when unchecked. Apply modest pitch-preserving tempo adjustment only when needed. Embed chapters and write synchronized SRT and description chapter timestamps.
12. Ask Qwen3 4B for six thumbnail concepts, select three styles, and compare up to two generated image candidates per selected concept. Complete cinematic scenes are the main path; local artwork and trailer cutouts remain a fallback. No paid inference fallback is used.
13. Validate output files, usable media streams and duration, thumbnail dimensions/size, and a full FFmpeg decode. A healthy export is not rejected solely for missing the length goal. Upload the package for one day, remove downloaded trailers and render intermediates, and save bounded speech/script checkpoints.

No approval or publication steps occur. Successful runs deliver downloadable files; this repository does not upload to YouTube.

## CPU performance and reliability

Turbo is the quality-oriented default, **not a benchmarked speed promise**. To use the smaller CPU-focused model, set repository **variable** `TTS_MODEL` to `nano`. Both use the same voice reference, and the pipeline never switches voices silently.

Public and private GitHub-hosted runners have different resource allowances. The workflow has a 350-minute timeout; first runs download several GB of models and dependencies. Model and job caches accelerate later runs, but GitHub may evict them. Free storage/minute quotas are finite; inspect your account's billing settings. Standard GitHub runners do not provide a free GPU. A failed/timed-out run delivers no partial package. Re-running a failed job can restore checkpoints saved by the failure handler; runner cancellation/timeouts may prevent cache saving.

Some upstream model/tool dependencies are pinned, while yt-dlp remains updateable to track changing sites. Ollama's installer and model tag track their upstream distribution. For strict reproducibility, pin their versions/model digests after a verified production run. Do not expect the first unbenchmarked run to have a guaranteed completion time.

YouTube can block data-center downloads or require authentication. Neither yt-dlp nor this workflow guarantees access. It does not bypass DRM or bot challenges. The pipeline stops clearly when required footage is missing instead of substituting a different film. A self-hosted runner is an alternative if hosted-runner access is unreliable. Publicly visible trailers are copyrighted; downloading and using them must comply with source terms and applicable rights. Short clips and commentary do not guarantee clearance or prevent Content ID claims.

Automated shot matching, trailer dialogue selection, factual accuracy, and narration quality remain imperfect. CLIP matching is a heuristic; the system does not understand every scene or guarantee spoiler avoidance. The three thumbnails are options, not tested CTR winners. The implementation uses clean cuts and rank graphics; it does not claim to reproduce a professional human editor's judgment.

## Resolution, disk space, and downloaded footage

The **Try 4K** checkbox is checked by default. Checked means “choose the best shared quality up to **3840×2160**”; unchecked caps the search and export at **1920×1080**. The brief no longer controls resolution, and no `VIDEO_RESOLUTION` repository variable is needed. Thumbnails remain 1280×720 JPEGs.

The pipeline downloads the best primary trailer for each title, checks their actual dimensions, then chooses the highest export resolution all primary sources support. All-4K sources can produce 4K; a mix of 4K and 1080p produces 1080p; 1440p, 720p, or lower sources lower the shared export accordingly. A weaker source is not stretched to force a 4K-sized file. If a low-resolution candidate would lower the result, the downloader first checks other candidate trailers for that movie and prefers a better source when available. It retains at most two trailers per title, with an 800 MB cap for each file.

After the first-trailer round has completed for every movie/show, the workflow starts a separate second-trailer round to enrich each title, without requiring another input. The second source must meet the already-selected export resolution, so optional extra footage cannot lower the result or force upscaling. Sampled-frame perceptual hashes estimate whether it adds different imagery instead of repeating a reupload. This is a scene-diversity heuristic, not proof that every scene is new or that a fan edit contains only footage from the intended title. Metadata checks still tie candidates to the same movie/show. Missing, inaccessible, too-similar, low-resolution, or unusable second trailers are skipped; production continues with the primary source. A working-disk reserve also takes priority over keeping extra footage.

Missing 4K footage does not cause a resolution error: production continues at the available shared quality. Logs and the cached job's quality-report JSON record accepted source dimensions, including optional second trailers, and the selected export resolution. All accepted trailer source URLs are included in the description. Website access and native 4K availability are not guaranteed.

4K is supported but expensive on a standard CPU runner: it has four times as many pixels as 1080p, can take substantially longer to encode, and may reach the 350-minute job timeout. Standard runners have finite temporary disk, and models, dependencies, trailers, rendered parts, concatenated input, and final output can coexist during production. Conservative free-space checks stop with an explanation when the working reserve is insufficient. 1080p is the practical choice for more dependable free runs; no 4K completion-time guarantee is made.

Artifacts expire after **one day**. That reduces how long finished packages occupy artifact storage; it does not increase disk space while a job is running. Models and checkpoint caches are separate from artifact storage and follow GitHub's own quotas and eviction rules. Repeated cache snapshots and older cache keys can still occupy repository cache space even though each new checkpoint snapshot is bounded.

During a run, trailers and their analysis files live under `work/footage/`; rendered intermediate clips live under `work/jobs/`. The cleanup step runs after success or failure, deletes both primary and optional second trailers plus rendered MP4s, and preserves `output/`, model weights, and the bounded speech/script checkpoints. Trailers must therefore be downloaded again on a new run. A cancelled job or runner shutdown can prevent cleanup or checkpoint saving, but GitHub-hosted runner disks are temporary and are discarded with the runner. These directories are never committed to the repository by the workflow.

## Streaming availability and descriptions

Watch listings use **US availability** from TMDB's JustWatch-powered endpoint. Successful lookups are cached for 24 hours and refreshed as needed. After a review, narration can say where the title is listed, distinguishing a subscription from a free-with-ads, rental, or purchase offer. The spoken line omits the country as requested. The description labels the listings as US and records the check date; availability can still change or be incomplete. Missing data or a provider outage omits the claim and does not fail the video.

`description.txt` begins with the first two spoken script paragraphs, followed by a simple topic summary and 3–5 relevant hashtags. It also includes timed chapters and a heading that matches the content: Featured movies, Featured shows / series, or a mixed heading. The former TMDB disclaimer sentence is omitted from this file as requested; source and attribution notes are kept separately in `sources-and-credits.txt`.

## Local development

```bash
python3.11 -m venv .venv
source .venv/bin/activate
pip install -e '.[test]'
pytest -q

# Full production also needs FFmpeg, fonts, Ollama, and CPU speech dependencies.
pip install torch==2.6.0 torchaudio==2.6.0 --index-url https://download.pytorch.org/whl/cpu
pip install -e '.[voice,thumbnails]'
ollama serve
# In another terminal:
ollama pull qwen3:4b
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

## Choose a video length

Use the workflow's **Video length in minutes** selector. The creative brief no longer controls runtime.

- **Auto (default):** plan an appropriate amount of commentary for the list, then let complete reviews determine the result. A top-five shows video is usually planned around 7–9 minutes; a top-ten movies video can reasonably be around 13–15 minutes. Auto does not turn a small list into an arbitrary 100-minute video or stretch useful narration to fill a fixed quota.
- **A selected number:** every minute option uses a **±25% soft target**. Selecting **8** means **6–10 minutes**; selecting **5** means **3:45–6:15**; selecting **15** means **11:15–18:45**. The selector offers every whole minute from 1 to 30.

The script budget scales with title count and the planned trailer pauses, subscribe break, and ending hold. Paragraph word counts stay flexible. Empty or unusable narration still gets retried; oversized paragraphs are split into smaller TTS chunks without deleting words. For selected minutes, measured speech can drive bounded script adjustments and a limited natural tempo correction. If those adjustments still miss the goal, the workflow keeps the healthy video rather than failing solely on length, cutting off a review, padding with filler, or forcing unnatural speech speed. Auto mode does not chase a fixed target after narration is generated. Chapters and subtitles follow the actual final timeline.

This does not remove real resource or media failures: longer videos require more generation/render time and storage, and GitHub's 350-minute job timeout and account quotas still apply. Missing required footage, broken media, or failed narration can still stop production. Automated tests check timing selection, script-budget scaling, and media export; full live model output still needs verification on the runner.

## Steadier editing, hooks, and calls to action

Reviews now aim for visual beats around 7.5–8.5 seconds where source footage allows, rather than the old rapid 3–4 second recuts. Longer stable source shots are preferred. When source shots are short, a readable held frame can give the narration time to finish. The intro uses roughly three-second images from several titles, covering at least half the list when its duration allows that without rapid flashes. The first visual can come from anywhere in a trailer, not just its opening.

CLIP matches the current thought in the script loosely to available frames across the accepted trailers; relevance is a preference, not a pass/fail condition. The intro can choose the stronger source for each movie/show, and review paragraphs can draw on both while keeping the same steady pacing. It is not a guarantee that every shot illustrates a sentence precisely. When narration discusses a performer or character and a suitable face frame is detected, the edit can pause on it for a few seconds. Face detection is not actor-identity verification. Short dialogue inserts use the primary trailer and keep their selected audio/video context, with narration silent during the insert. Second-trailer footage is muted under the original commentary.

Hooks use the owner's conversational, curiosity-driven examples as a style direction, adapted to the actual title count and genre. They avoid inventing viewing history, statistics, awards, or availability. Slang can appear naturally without being forced into every line. A rare optional “Absolute Cinema” meme image may be used only for a matching hook and approximately one in eight eligible jobs. The image is fetched from Imgflip; missing images or failed downloads simply skip the meme. It is never a required production dependency.

A funny like-and-subscribe message appears once between reviews, chosen from the actual measured timeline to target 30–50% of the video. It never interrupts a review. If no review boundary fits that window, the closest boundary is used instead of splitting a movie section. The message is kept below 10 seconds after tempo changes and uses a steady card with like, subscribe, and bell graphics. The outro asks viewers to like, subscribe, and watch other recommendations. Its last image remains for about three seconds after speech, then fades out gently.

Chapters include Intro, each movie/show with its timeline position, and Outro. They are adjusted after final tempo changes, written into `description.txt`, and embedded in the MP4. YouTube chapter display requires pasting the timestamped description into the video upload and remains subject to YouTube's chapter eligibility rules.

Editing is for pacing and original commentary. Cuts, shuffling, frame holds, overlays, memes, or music do not by themselves establish fair use, prevent Content ID claims, or guarantee YouTube monetization. Rights and YouTube's reused-content assessment are separate from the edit effects.

### Voice speed and failed-run recovery

Speaking-rate measurements are now logged as diagnostics. A fast or slow delivery alone does not reject a segment: the transcription stage checks whether the words were actually spoken, and final duration control handles overall pacing. Voice synthesis still rejects empty, silent, non-finite, or extremely long/runaway output after bounded retries.

A new workflow run can restore earlier research, script, and completed narration checkpoints when GitHub retains them. Downloaded footage and rendered MP4 intermediates are no longer included in checkpoint caches. Cleanup keeps at most the newest two job folders, bounded to 2 GB total before saving; models use a separate cache. Each job's paths are keyed by its inputs and selected voice, so different inputs do not reuse the wrong script or narrator. Reuse is best effort and does not guarantee zero repeated work. Neither cookies nor voice secrets are included in the checkpoint cache. Existing older `production-v1` caches are not automatically deleted; remove them in GitHub's Actions cache settings if you need that space back.

## Telegram completion reminder

After the video is generated, validated, and successfully uploaded to GitHub Artifacts, the workflow can send one formatted Telegram reminder with the recommended title and GitHub run link, then pin that same message. Five blank lines separate the heading, title, and download sections. It does not upload the video or other files to Telegram. Failed video runs do not send a success reminder. Missing credentials or a Telegram delivery error do not fail an otherwise successful video job.

One-time setup:

1. In Telegram, open the official **@BotFather** and send `/newbot`. Choose a name and a unique bot username. Copy the bot token privately.
2. Open your new bot's chat and press **Start**, or send `/start`. Bots cannot initiate a private conversation until you do this.
3. Get your chat ID: in a private browser session open `https://api.telegram.org/botYOUR_TOKEN/getUpdates`, replacing `YOUR_TOKEN` locally with your token. Find `message` → `chat` → `id` for your own `/start` message. If the result is empty, send another message to your bot and refresh. Never share the token-bearing URL or response; it may contain private chat details. Close the private browser afterward.
4. In repository **Settings → Secrets and variables → Actions**, add:
   - `TELEGRAM_BOT_TOKEN`: the token from BotFather.
   - `TELEGRAM_CHAT_ID`: your numeric chat ID.
5. Start a new Generate Video run on **main**. A reminder is sent only after its finished package uploads successfully.

If getUpdates reports a webhook conflict, use a dedicated new bot for this workflow instead of changing another application's bot. If a token is exposed, revoke it through BotFather and replace the GitHub secret. The linked artifacts retain the repository's GitHub access requirements and expire after one day. Delivery is best effort: a network failure can prevent confirmation, and re-running a successful production job can send another reminder.

### Channel pinning and spacing

Completion reminders have bold labels and five empty lines between their three main sections. The bot pins the message it just sent; it does not delete posts or unpin other messages. For a channel, make the bot an administrator with **Post Messages** and **Edit Messages** rights (the Bot API requires edit rights to pin in channels). For a group/supergroup, enable **Pin Messages**. Pinning failure leaves the sent reminder intact and does not fail video production. Telegram clients control exact text spacing and notification behavior; pinned-message alerts in channels may be silent.

## Redesigned thumbnail generation

The thumbnail pipeline now requests complete cinematic scenes, including subjects, lighting, and environment together. Qwen3 4B proposes six concepts from the title, brief, premises, and commentary. A lightweight composition-cue selector picks one concept per style. Each selected concept gets up to two image seeds (six image requests maximum); phone-size sharpness, contrast, exposure, and duplicate checks pick a candidate. These are visual heuristics, not a vision model, anatomy checker, semantic verifier, or CTR predictor.

The reference direction is expressive close-ups, intense genre-appropriate scenes, coherent dramatic lighting, strong depth, and minimal clutter. Horror gets dedicated dread/silhouette concepts; action gets chase/impact concepts. Generated scenes are illustrative artwork, not verified movie stills or guaranteed actor likenesses.

- **Option 1:** minimal-text cinematic hero.
- **Option 2:** bold headline, at most five words, with the generated subject on the right and readable text on the left.
- **Option 3:** cinematic alternative with the requested red Netflix-style N every time. The badge does not verify Netflix availability or endorsement.
- **Provider:** retains anonymous Pollinations Classic, with no keys or paid fallback. No unverified free FLUX service is claimed. Requests are sequential, at least 45 seconds apart, requested at 1024×576 and resized without cropping away provider branding. The additional candidates can add several minutes. Cached image requests are reused.
- **Fallback:** provider failure, blank images, or duplicate winners use local artwork and trailer subject cutouts. CPU rembg/U2Net and OpenCV extraction are now fallback tools only; successful complete scenes never get pasted trailer subjects over them.
- **Outputs:** three 1280×720 JPEGs below 2 MB each. Candidate scores, selected seeds, fallback methods, and concepts are recorded in `thumbnail-work/thumbnail-report.json`, outside the downloadable package. Updated concept/image cache versions prevent old background-only plans from being reused.

Audio stays Chatterbox Turbo by default; Pocket TTS remains excluded. Only generated visual prompts are sent to the public image provider, not voice recordings, tokens, or the whole script. Free service availability and quality are not guaranteed. Live provider quality must be checked on the GitHub runner; offline tests cover selection, composition, cache behavior, fallback, and file constraints.
