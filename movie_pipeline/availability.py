"""Optional US watch listings from TMDB's JustWatch-powered provider data.

Listings are checked close to production time. Missing data never blocks a video;
subscription, free-with-ads, rental, and purchase offers remain distinct.
"""
from __future__ import annotations

import json
from datetime import UTC, datetime, timedelta
from pathlib import Path
from urllib.parse import urlsplit

import requests
from requests.adapters import HTTPAdapter

from .core import PipelineError, save_json

CATEGORIES = {"subscription": "flatrate", "free": "free", "ads": "ads", "rent": "rent", "buy": "buy"}
TTL = timedelta(hours=24)


def _empty(now: datetime, status: str = "unavailable") -> dict:
    return {"region": "US", "checked_at": now.isoformat(), "status": status,
            "source": "TMDB / JustWatch", "link": "", **{name: [] for name in CATEGORIES}}


def _names(offers) -> list[str]:
    if not isinstance(offers, list):
        return []
    names = []
    seen = set()
    for offer in offers:
        if not isinstance(offer, dict):
            continue
        name = offer.get("provider_name")
        if not isinstance(name, str) or not 1 <= len(name.strip()) <= 100 or any(ord(c) < 32 for c in name):
            continue
        name = " ".join(name.split())
        if any(char in name for char in "<>{}") or name.casefold() in seen:
            continue
        seen.add(name.casefold())
        names.append(name)
    return names


def _link(value) -> str:
    if not isinstance(value, str):
        return ""
    try:
        parsed = urlsplit(value)
        if (parsed.scheme == "https" and parsed.hostname in {"www.themoviedb.org", "themoviedb.org"}
                and not parsed.username and not parsed.password and parsed.port in {None, 443}):
            return value
    except ValueError:
        pass
    return ""


def _cached(path: Path, now: datetime) -> dict | None:
    try:
        record = json.loads(path.read_text())
        checked = datetime.fromisoformat(record["checked_at"])
        if (record.get("region") == "US" and record.get("status") in {"found", "not-listed"}
                and checked.tzinfo is not None and timedelta(0) <= now-checked < TTL
                and all(isinstance(record.get(name), list) and all(isinstance(n, str) for n in record[name]) for name in CATEGORIES)):
            return record
    except (OSError, ValueError, TypeError, KeyError, AttributeError):
        pass
    return None


def enrich_availability(films: list[dict], research, cache: Path) -> list[dict]:
    """Return augmented film dictionaries; cache successful lookups for 24 hours.

    The optional lookup uses a short-lived session with the existing TMDB auth
    headers and no retries, so a provider outage cannot consume the whole run.
    Authentication is sent only to the fixed api.themoviedb.org endpoint.
    """
    now = datetime.now(UTC)
    cache.mkdir(parents=True, exist_ok=True)
    result = []
    with requests.Session() as session:
        session.headers.update(research.session.headers)
        session.mount("https://", HTTPAdapter(max_retries=0))
        for film in films:
            kind, identifier = film.get("kind"), film.get("id")
            if kind not in {"movie", "tv"} or not isinstance(identifier, int) or isinstance(identifier, bool) or identifier <= 0:
                result.append({**film, "watch_availability": _empty(now)})
                continue
            path = cache / f"{kind}-{identifier}-US-v1.json"
            record = _cached(path, now)
            if record is None:
                record = _empty(now)
                try:
                    response = session.get(f"https://api.themoviedb.org/3/{kind}/{identifier}/watch/providers", timeout=(5, 12), allow_redirects=False)
                    if response.status_code != 200:
                        raise PipelineError("Watch listing provider unavailable")
                    payload = response.json()
                    if not isinstance(payload, dict) or not isinstance(payload.get("results"), dict):
                        raise TypeError("Malformed watch listing response")
                    listings = payload["results"]
                    region = listings.get("US", {})
                    if not isinstance(region, dict):
                        raise TypeError("Malformed regional watch listing")
                    for name, api_name in CATEGORIES.items():
                        record[name] = _names(region.get(api_name))
                    record["link"] = _link(region.get("link"))
                    record["status"] = "found" if any(record[name] for name in CATEGORIES) else "not-listed"
                    save_json(path, record)
                except (requests.RequestException, PipelineError, ValueError, TypeError, OSError):
                    # Never log response bodies, auth headers, or exceptions containing URLs.
                    print(f"Watch listings unavailable for {film.get('title', 'this title')}; continuing without availability claims.", flush=True)
            result.append({**film, "watch_availability": record})
    return result


def availability_line(record: dict | None) -> str:
    """A short factual line for narration, omitting the country as requested."""
    if not isinstance(record, dict) or record.get("status") not in {"found", None}:
        return ""

    def providers(category):
        values = record.get(category, [])
        if not isinstance(values, list):
            return ""
        names = list(dict.fromkeys(n for n in values if isinstance(n, str) and n.strip()))[:2]
        return " and ".join(names)

    if names := providers("subscription"):
        return f"It's listed on {names} with a subscription."
    if names := providers("free"):
        return f"It's listed free to watch on {names}."
    if names := providers("ads"):
        return f"It's listed free with ads on {names}."
    rentals, purchases = providers("rent"), providers("buy")
    if rentals and rentals == purchases:
        return f"It's listed to rent or buy on {rentals}."
    if rentals:
        return f"It's listed to rent on {rentals}."
    if purchases:
        return f"It's listed to buy on {purchases}."
    return ""
