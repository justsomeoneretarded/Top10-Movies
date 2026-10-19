"""Validate input and selected narrator before large model downloads."""
import os
from pathlib import Path

from movie_pipeline.core import (
    PipelineError,
    narrator_from_brief,
    parse_movies,
    voice_reference,
)
from movie_pipeline.timing import requested_duration

title, brief = os.getenv("VIDEO_TITLE", ""), os.getenv("VIDEO_BRIEF", "")
if not title.strip() or not brief.strip():
    raise PipelineError("Title and description/creative brief are required.")
films = parse_movies(os.getenv("MOVIE_LIST", ""))
if not os.getenv("TMDB_READ_TOKEN"):
    raise PipelineError("Configure the free TMDB_READ_TOKEN repository secret before running.")
requested_duration(brief)
name = narrator_from_brief(brief)
voice_reference(name, Path("assets/voices"))
print(f"Input valid: {len(films)} titles; narrator {name}.")
