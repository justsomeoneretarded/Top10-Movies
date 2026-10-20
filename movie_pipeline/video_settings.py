"""Export resolution and conservative scratch-space checks, independent of thumbnails."""
from __future__ import annotations

import os
import re
import shutil
from pathlib import Path

from .core import PipelineError

SIZES = {720: (1280, 720), 1080: (1920, 1080), 1440: (2560, 1440), 2160: (3840, 2160)}


def resolution_from_brief(brief: str) -> int:
    requested = set()
    for pattern, height in ((r"\b(?:4k|2160p)\b", 2160), (r"\b(?:1440p|2k|qhd)\b", 1440),
                            (r"\b(?:1080p|full[ -]?hd)\b", 1080), (r"\b720p\b", 720)):
        if re.search(pattern, brief, re.IGNORECASE):
            requested.add(height)
    return max(requested) if requested else export_size()[1]


def export_size() -> tuple[int, int]:
    value = os.getenv("VIDEO_RESOLUTION", "1080").strip().lower().removesuffix("p")
    value = {"4k": "2160", "2k": "1440"}.get(value, value)
    if not value.isdigit() or int(value) not in SIZES:
        raise PipelineError("VIDEO_RESOLUTION must be 720, 1080, 1440, or 2160 (4K).")
    return SIZES[int(value)]


def require_disk(path: Path, seconds: float = 0):
    # Parts, concatenated input, and final encode coexist. This is a reserve, not a size guarantee.
    height = export_size()[1]
    rate = {720: 6, 1080: 10, 1440: 16, 2160: 28}[height] * 1_000_000 / 8
    needed = max(1_500_000_000, seconds * rate * 3 + 1_000_000_000)
    free = shutil.disk_usage(path).free
    if free < needed:
        raise PipelineError(f"Not enough working disk for {height}p: {free/1e9:.1f} GB free, about {needed/1e9:.1f} GB reserved. Try 1080p or fewer titles; artifact expiry does not free running-job disk.")
