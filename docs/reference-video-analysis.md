# Reference video analysis

This optional analysis is separate from Generate Video. It measures an existing
reference and, optionally, a finished production. It does not change either video
or automatically change the production workflow's creative settings.

## Run from GitHub Actions

Open **Actions → Analyze Reference Video**. Provide exactly one of:

- A YouTube or Vimeo HTTPS reference URL, or
- A previous Generate Video run ID, optionally with an artifact name/pattern.

Optionally enter a comparison Generate Video run ID. Both artifact fields default
to `movie-video-*`. If an artifact contains more than one video, enter the
video's **relative path inside the downloaded artifact directory**. The workflow
rejects absolute paths and paths that escape this directory. It checks artifact
sizes and free disk before downloading. Reference clips must be up to 30 minutes.
Authenticated downloads can use the existing optional `YOUTUBE_COOKIES` secret;
some videos may remain unavailable on hosted runners.

Choose transcription if useful. Existing `subtitles.srt` files are preferred;
otherwise optional CPU speech recognition generates an approximate transcript.
Recognition may include film dialogue and need not represent the narrator alone.

The artifact `reference-analysis` lasts two days and includes contact sheets,
JSON and Markdown reports. With a comparison, it includes both video reports and
a comparison. Downloaded originals and private cookies are not packaged.

## Local usage

```sh
python -m movie_pipeline.reference_analysis --video /path/reference.mp4 \
  --compare-video /path/final-video.mp4 --output-dir reference-analysis \
  --transcript /path/reference.srt --comparison-transcript /path/subtitles.srt
```

Optional `--transcribe` performs CPU ASR if no transcript was provided.
`--transcript` accepts SRT, VTT, or a plain UTF-8 text transcript. Plain text
has no timings. The same rules apply to `--comparison-transcript`.

If you actually have isolated stems, `--voice-stem` and `--music-stem` measure
the reference stems; `--comparison-voice-stem` and `--comparison-music-stem`
measure production stems. A mixed soundtrack is never treated as an isolated
narrator or music track. Diagnostic snippets do not represent a whole episode.

## What is measured

- Approximate abrupt visual boundaries by sampling frames every 0.5 seconds,
  plus a contact sheet with timestamps.
- Integrated loudness and true peak of the mixed soundtrack, and intervals where
  the whole mixed soundtrack falls below the reported silence threshold.
- A provided or optionally recognized transcript with timestamps when available.
- Direct measurements from genuinely isolated voice or music stems, if supplied.

Cuts may be missed, and a sudden camera move can be mistaken for a cut. Fades,
story quality, emotion, visual relevance, narration quality, individual voices
and music-to-narrator balance cannot be established by these numbers alone.
Automatic speech recognition may miss words or mix movie dialogue with the host.
The comparison describes measured differences, **not** automatically ideal
settings or a quality ranking. Listen to and watch the video before changing
production decisions.
