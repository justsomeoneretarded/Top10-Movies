"""Plain-language metadata and grounded availability copy; no model calls or hard failures."""
from __future__ import annotations

import re
from collections import Counter

from .availability import availability_line
from .core import key

MIDROLLS = (
    "Like and subscribe. It's free. My movie opinions are the expensive part.",
    "Like and subscribe. Your watchlist needs help, and my ego needs supervision.",
    "Like and subscribe. Consider it insurance against another terrible movie night.",
    "Hit like and subscribe. Your finger could use the exercise.",
    "Like and subscribe. The button has been looking lonely lately.",
    "Subscribe. It's the only commitment I'm asking you to make today.",
    "Like and subscribe. My Wi-Fi deserves better than an audience of three.",
    "Subscribe. Consider it your post-credit scene. Back to the movies.",
)


def midroll_text(seed: str) -> str:
    """Most options request both actions; all stay short enough for a quick break."""
    return MIDROLLS[int(key(seed, "midroll-v1")[:8], 16) % len(MIDROLLS)]


def _provider_names(values) -> list[str]:
    if not isinstance(values, list):
        return []
    names = []
    for value in values:
        name = value if isinstance(value, str) else value.get("provider_name", value.get("name", "")) if isinstance(value, dict) else ""
        name = " ".join(str(name).split()).strip()
        if name and name not in names:
            names.append(name)
    return names


def _availability(film: dict) -> dict:
    data = film.get("watch_availability")
    return data if isinstance(data, dict) else {}


def refresh_availability(script: dict, films: list[dict]) -> dict:
    """Refresh only marked provider endlines; preserve cached original commentary."""
    for section in script.get("sections", []):
        index = section.get("film")
        if section.get("kind") != "film" or not isinstance(index, int) or not 0 <= index < len(films):
            continue
        paragraphs = section.get("paragraphs", [])
        old = section.get("availability_text", "")
        if old and paragraphs and paragraphs[-1] == old:
            paragraphs.pop()
        current = availability_line(films[index].get("watch_availability", {}))
        if current:
            paragraphs.append(current)
        section["availability_text"] = current
    return script


def featured_heading(films: list[dict]) -> str:
    kinds = {film.get("kind", "movie") for film in films}
    if kinds == {"tv"}:
        return "Featured shows / series"
    if kinds == {"movie"}:
        return "Featured movies"
    return "Featured movies and shows / series"


def chapter_time(seconds: float) -> str:
    total = max(0, int(seconds))
    if total >= 3600:
        return f"{total//3600}:{total//60%60:02}:{total%60:02}"
    return f"{total//60:02}:{total%60:02}"


def topic_hashtags(title: str, films: list[dict]) -> list[str]:
    """Choose 3–5 actual topic tags; these are relevance hints, not a ranking guarantee."""
    shows_only = bool(films) and all(film.get("kind") == "tv" for film in films)
    tags = ["#TVRecommendations" if shows_only else "#MovieRecommendations", "#WhatToWatch"]
    mapping = {
        "action": "ActionShows" if shows_only else "ActionMovies",
        "horror": "HorrorShows" if shows_only else "HorrorMovies",
        "science fiction": "SciFi", "sci-fi": "SciFi", "sci fi": "SciFi",
        "thriller": "Thrillers", "comedy": "Comedy", "drama": "Drama",
        "mystery": "Mystery", "animation": "Animation", "fantasy": "Fantasy",
    }
    topic = title.casefold()
    genres = Counter(str(genre).casefold() for film in films for genre in film.get("genres", []))
    candidates = [name for name in mapping if name in topic]
    candidates += [name for name, _count in genres.most_common() if name in mapping]
    for genre in candidates:
        tag = "#"+mapping[genre]
        if tag not in tags:
            tags.append(tag)
        if len(tags) == 5:
            break
    if len(tags) < 3:
        tags.append("#TVShows" if shows_only else "#MoviesToWatch")
    return tags


def build_description(script: dict, films: list[dict], chapters: list[dict], trailer_sources: list[str]) -> str:
    """Keep the first two spoken paragraphs verbatim, then readable search-friendly metadata."""
    package = script.get("package", {})
    spoken = [text for section in script.get("sections", []) if section.get("kind") != "subscribe"
              for text in section.get("paragraphs", []) if isinstance(text, str) and text.strip()]
    chunks = [text.strip() for text in spoken[:2]]
    summary = str(package.get("description", "")).strip()
    # Old checkpoint copy may contain the exact paragraph the owner asked to remove.
    summary = re.sub(r"Film information:\s*TMDB\.\s*This product uses the TMDB API but is not endorsed or certified by TMDB\.?", "", summary, flags=re.IGNORECASE).strip()
    if summary:
        chunks.append(summary)
    chunks.append(" ".join(topic_hashtags(str(package.get("title", "")), films)))
    if chapters:
        chunks.append("Chapters\n" + "\n".join(f"{chapter_time(c['start'])} {c['title']}" for c in chapters))
    chunks.append(featured_heading(films) + "\n" + "\n".join(f"{i+1}. {film['title']} ({film['year']})" for i, film in enumerate(films)))
    listings = []
    dates = set()
    labels = {"subscription": "Stream", "free": "Free", "ads": "With ads", "rent": "Rent", "buy": "Buy"}
    for film in films:
        data = _availability(film)
        groups = []
        for category, label in labels.items():
            names = _provider_names(data.get(category))[:3]
            if names:
                groups.append(label+": "+", ".join(names))
        if groups:
            listings.append(f"{film['title']}: " + "; ".join(groups))
            if data.get("checked_at"):
                dates.add(str(data["checked_at"])[:10])
    if listings:
        checked = ", ".join(sorted(dates))
        heading = "Where to watch (US listings" + (f", checked {checked}" if checked else "") + ")"
        chunks.append(heading + "\n" + "\n".join(listings) + "\nAvailability can change. Subscription, rental, and purchase options are listed separately.")
    if trailer_sources:
        chunks.append("Trailer sources\n" + "\n".join(dict.fromkeys(trailer_sources)))
    return "\n\n".join(chunk for chunk in chunks if chunk) + "\n"
