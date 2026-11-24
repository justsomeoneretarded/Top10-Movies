"""Plot-only writing context, separate from full research and identity metadata.

This is a schema projection and removal of structural credit/catalog sentences,
not an adjective replacement list. Character actions and ordinary creative
wording are never rewritten to sound more impressive.
"""
from __future__ import annotations

from copy import deepcopy
import re

STORY_FIELDS = (
    "title", "year", "kind", "genres", "keywords", "characters", "overview",
    "premise", "plot", "plot_summary", "synopsis", "storyline", "story",
    "wiki_plot", "wiki_synopsis", "research_plot", "wiki_summary",
)
CONTEXT_FIELDS = (
    "rank", "titles", "other_titles", "count", "shared_genres", "requested_years",
    "opening_word_goal", "topic", "comment_question", "opening_hook_idea",
    "open_loop_idea", "list_angle", "video_promise", "episode_plan",
    "previous_narration_context",
)
TEXT_FIELDS = set(STORY_FIELDS) - {"title", "year", "kind", "genres", "keywords", "characters"}

# These patterns recognise the shape of a credit or catalogue record, not words
# such as "director" used for a fictional person's job within the actual plot.
_CREDIT_HEAD = re.compile(
    r"^(?:(?:the|this) (?:film|movie|show|series) (?:is|was) )?"
    r"(?:written|directed|created|produced|co-created|co-written|co-directed)"
    r"(?: and (?:written|directed|created|produced))? by\s+", re.I)
_STARRING_HEAD = re.compile(
    r"^(?:(?:the|this) (?:film|movie|show|series) )?(?:stars?|starring|features? performances? by)\s+",
    re.I)
_CATALOGUE = re.compile(
    r"^.{1,100}?\s+is\s+(?:an?\s+)?(?:18|19|20)\d{2}\b.{0,220}?"
    r"\b(?:film|movie|series|show)\b.{0,120}?\b(?:directed|written|created|produced|starring)\b",
    re.I)
_STORY_CONTINUATION = re.compile(
    r"\b((?:the|this) (?:film|movie|show|series|story)\s+"
    r"(?:follows|centres on|centers on|is about)\b.*)$", re.I)


def _story_sentence(sentence: str) -> str:
    text = sentence.strip()
    credit = _CREDIT_HEAD.match(text) or _STARRING_HEAD.match(text)
    catalogue = _CATALOGUE.match(text)
    # A lowercase object is often story-world prose: "Written by a killer,
    # the note leads her into a trap." It is not a filmmaker credit.
    if credit and (not text[credit.end():] or not text[credit.end():][0].isupper()):
        credit = None
    if not credit and not catalogue:
        return text
    continuation = _STORY_CONTINUATION.search(text)
    if continuation:
        retained = continuation.group(1)
        return retained[:1].upper() + retained[1:]
    return ""


def story_text(value: str) -> str:
    """Keep story sentences in a mixed source paragraph, without synonym swaps."""
    if not isinstance(value, str):
        return ""
    sentences = re.split(r"(?<=[.!?])\s+", value.strip())
    return " ".join(clean for sentence in sentences if (clean := _story_sentence(sentence)))


def remove_credit_sentences(paragraphs: list[str]) -> list[str]:
    """Keep complete plot paragraphs while removing standalone production credits."""
    return [clean for paragraph in paragraphs if isinstance(paragraph, str)
            and (clean := story_text(paragraph))]


def character_names(film: dict) -> list[str]:
    """Project fictional character names, never actor-as-character credit strings."""
    result = []
    supplied = film.get("characters", [])
    for value in supplied if isinstance(supplied, list) else []:
        if isinstance(value, str) and value.strip() and value.strip() not in result:
            result.append(value.strip())
    for person in film.get("cast_profiles", []) if isinstance(film.get("cast_profiles"), list) else []:
        value = person.get("character") if isinstance(person, dict) else None
        if isinstance(value, str) and value.strip() and value.strip() not in result:
            result.append(value.strip())
    return result[:8]


def plot_material(facts: dict) -> dict:
    """Default writing view: plot and section controls; full facts remain untouched."""
    result = {}
    for name in (*STORY_FIELDS, *CONTEXT_FIELDS):
        value = facts.get(name)
        if value in (None, "", []):
            continue
        if name in TEXT_FIELDS and isinstance(value, str):
            value = story_text(value)
            if not value:
                continue
        result[name] = deepcopy(value)
    people = character_names(facts)
    if people:
        result["characters"] = people
    return result
