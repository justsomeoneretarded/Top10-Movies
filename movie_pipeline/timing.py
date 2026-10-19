"""Requested runtime is independent of individual paragraph word counts."""
from __future__ import annotations

import math
import re
from dataclasses import dataclass

from .core import PipelineError


@dataclass(frozen=True)
class DurationSpec:
    target: float = 540
    minimum: float = 480
    maximum: float = 600


def requested_duration(brief: str) -> DurationSpec:
    text = brief.casefold().replace("–", "-").replace("—", "-")
    words = {"one": 1, "two": 2, "three": 3, "four": 4, "five": 5, "six": 6,
             "seven": 7, "eight": 8, "nine": 9, "ten": 10, "fifteen": 15,
             "twenty": 20, "thirty": 30, "forty": 40, "sixty": 60}
    for word, number in words.items():
        text = re.sub(rf"\b{word}(?=[ -]+(?:minute|min|hour|hr|second|sec))", str(number), text)
    number = r"\d+(?:\.\d+)?"
    unit = r"(?:minutes?|mins?|hours?|hrs?)\b"
    matches = list(re.finditer(rf"\b({number})(?:\s*(?:-|to)\s*({number}))?[ -]*({unit})", text))
    # Seconds commonly describe editing pauses, so require an explicit runtime/video context.
    matches += list(re.finditer(rf"(?:duration|runtime|length)\s*[:=]?\s*({number})(?:\s*(?:-|to)\s*({number}))?[ -]*(seconds?|secs?)\b", text))
    matches += list(re.finditer(rf"\b({number})(?:\s*(?:-|to)\s*({number}))?[ -]*(seconds?|secs?)\s+video\b", text))
    specs = []
    for match in matches:
        factor = 3600 if match[3].startswith(("hour", "hr")) else (1 if match[3].startswith(("second", "sec")) else 60)
        low, high = float(match[1])*factor, float(match[2] or match[1])*factor
        if not math.isfinite(low) or not math.isfinite(high) or low < 30 or high < low:
            raise PipelineError("Request a video of at least 30 seconds, with the shorter duration first in a range.")
        target = (low+high)/2
        specs.append(DurationSpec(target, low if match[2] else target*0.95, high if match[2] else target*1.05))
    unique = set(specs)
    if len(unique) > 1:
        raise PipelineError("The brief contains conflicting video durations. Specify one duration, such as 'Make a 5-minute video'.")
    return specs[0] if specs else DurationSpec()


def trailer_budget(spec: DurationSpec, count: int) -> float:
    per_film = min(4.5, spec.target * 0.08 / count)
    return per_film if per_film >= 1.5 else 0.0


def narration_words(seconds: float, count: int, trailer_seconds: float = 0, scale: float = 1) -> tuple[int, int, int]:
    total = max(20, round(max(1, seconds-trailer_seconds) * 145 / 60 * scale))
    opening = min(65, max(5, round(total*0.055)))
    closing = min(35, max(4, round(total*0.03)))
    body = max(3, round((total-opening-closing)/count))
    return opening, body, closing
