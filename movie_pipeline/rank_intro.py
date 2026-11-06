"""Separate spoken rank cards from review bodies before generating any speech.

No waveform is cut: each announcement and each remaining review paragraph gets
its own verified take. A preview can then sit between two complete utterances.
"""
from __future__ import annotations

from copy import deepcopy
import re

RANK_INTRO_VERSION = 1


def _announcement(rank: int, title: str) -> str:
    return f"Number {rank}: {title}" + ("" if title.endswith((".", "!", "?")) else ".")


def _body_after_lead(text: str, title: str) -> str:
    """Remove a leading announcement, never a premise containing a title.

    Titles are matched literally (including colons, periods and parentheses).
    If the title is the subject of an unfinished sentence, retain that subject:
    'Number 2: Arrival follows a linguist' becomes 'Arrival follows a linguist'.
    """
    text = text.strip()
    lead = re.compile(
        r"^(?:(?:at\s+)?(?:number|no\.?|rank)\s+(?:\d{1,2}|[a-z]+)"
        r"(?:\s+on\s+(?:our|the)\s+list)?\s*[,.:—–-]*\s*"
        r"(?:(?:it[’']s|is|we\s+have)\s+)?|#\s*\d{1,2}\s*[:.,—–-]*\s*)",
        re.IGNORECASE,
    )
    match = lead.match(text)
    offset = match.end() if match else 0
    literal = re.match(re.escape(title), text[offset:], re.IGNORECASE)
    if literal is None:
        return text
    end = offset + literal.end()
    tail = text[end:]
    # Prevent partial matches such as 'It' in 'It's' or 'Up' in 'Upgrade'.
    if tail and tail[0].isalnum():
        return text
    year = re.match(r"^\s*\(\d{4}\)", tail)
    if year:
        end += year.end()
        tail = text[end:]
    boundary = re.match(r"^\s*[.!?:;—–-]+\s*", tail)
    if not tail.strip():
        return ""
    if boundary:
        return tail[boundary.end():].strip()
    if title.endswith((".", "!", "?")) and re.match(r"^\s+[A-Z0-9]", tail):
        # The title itself may supply the sentence-ending punctuation.
        return tail.strip()
    # A bare title can be part of the actual review; remove only a ranking prefix.
    return text[offset:].strip() if match else text


def prepare_rank_intros(script: dict, films: list[dict]) -> dict:
    """Return an idempotent, delivery-aligned script with one reveal per review."""
    from .script import aligned_delivery

    prepared = deepcopy(script)
    for section in prepared.get("sections", []):
        index = section.get("film")
        if section.get("kind") != "film" or not isinstance(index, int) or not 0 <= index < len(films):
            continue
        title = str(films[index]["title"]).strip()
        paragraphs = list(section.get("paragraphs", []))
        labels = aligned_delivery(section)
        body, delivery = [], []
        for p, paragraph in enumerate(paragraphs):
            cleaned = _body_after_lead(paragraph, title) if p == 0 else paragraph
            if cleaned.strip():
                body.append(cleaned)
                delivery.append(labels[p])
        section["paragraphs"] = [_announcement(index + 1, title), *body]
        section["delivery"] = ["hype", *delivery]
        section["rank_intro_version"] = RANK_INTRO_VERSION
    return prepared


def rank_still_time(selector, source, shots: list[dict], preview: dict) -> float:
    """Prefer a clean analysed frame for the rank card, preserving the preview.

    Only this verified footage source is searched. Face/card evidence belongs to
    the indexed midpoint, not an arbitrary frame elsewhere in a linked window.
    With no usable analysis, retain the established preview-start fallback.
    """
    start = float(preview["start"])
    end = start + float(preview["duration"])
    faces = getattr(selector, "faces", {}).get(str(source), [])
    texts = getattr(selector, "card_texts", {}).get(str(source), [])
    card_kind = getattr(selector, "card_kind", None)
    choices = []
    for index, shot in enumerate(shots):
        try:
            left, right = float(shot["start"]), float(shot["end"])
            if not (0 <= left < right):
                continue
            if shot.get("text_card") or (callable(card_kind) and card_kind(source, index)):
                continue
            if index < len(texts) and str(texts[index] or "").strip():
                continue
            middle = (left + right) / 2
            inside = start <= middle < end
            face = index < len(faces) and bool(faces[index])
            # Do not jump elsewhere on weak evidence; a clean face is useful
            # even if its shot is outside the native preview.
            if not inside and not face:
                continue
            priority = 0 if inside and face else 1 if face else 2
            choices.append((priority, abs(middle - (start + end) / 2), middle))
        except (TypeError, ValueError, KeyError, IndexError):
            continue
    return min(choices)[2] if choices else start
