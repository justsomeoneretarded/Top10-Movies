from __future__ import annotations

import os
from difflib import SequenceMatcher
from pathlib import Path

import requests
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry

from .core import FilmInput, PipelineError, normalized, save_json


class Research:
    def __init__(self):
        self.token = os.getenv("TMDB_READ_TOKEN", "")
        if not self.token:
            raise PipelineError("Set the free TMDB_READ_TOKEN Actions secret (TMDB API Read Access Token).")
        self.session = requests.Session()
        self.session.headers.update({"Authorization": f"Bearer {self.token}"})
        self.session.mount("https://", HTTPAdapter(max_retries=Retry(total=4, backoff_factor=2, status_forcelist=[429, 500, 502, 503, 504])))

    def get(self, endpoint: str, **params):
        response = self.session.get("https://api.themoviedb.org/3/" + endpoint, params=params, timeout=40)
        # Avoid including request URLs or token-bearing headers in exception text.
        if response.status_code != 200:
            raise PipelineError(f"TMDB {endpoint} returned HTTP {response.status_code}. Check the token or retry later.")
        return response.json()

    def resolve(self, item: FilmInput) -> dict:
        candidates = []
        for kind in ("movie", "tv"):
            data = self.get(f"search/{kind}", query=item.title, language="en-US", include_adult="false")
            for found in data.get("results", []):
                date = found.get("release_date" if kind == "movie" else "first_air_date", "")
                title = found.get("title" if kind == "movie" else "name", "")
                original = found.get("original_title" if kind == "movie" else "original_name", "")
                similarity = max(SequenceMatcher(None, normalized(item.title), normalized(t)).ratio() for t in (title, original))
                if date[:4] == str(item.year) and similarity >= 0.88:
                    candidates.append((similarity, kind, found))
        candidates.sort(key=lambda x: x[0], reverse=True)
        if not candidates:
            raise PipelineError(f"No confident title/year match for {item.title} ({item.year}); correct the input.")
        if len(candidates) > 1 and candidates[0][0] - candidates[1][0] < 0.025:
            raise PipelineError(f"Ambiguous title/year: {item.title} ({item.year}). Use its full official title.")
        _, kind, match = candidates[0]
        details = self.get(f"{kind}/{match['id']}", append_to_response="videos,credits", language="en-US")
        videos = details.get("videos", {}).get("results", [])
        # Original-language trailers may exist when English-localized metadata has none.
        if not videos:
            videos = self.get(f"{kind}/{match['id']}/videos").get("results", [])
        videos = [v for v in videos if v.get("site") == "YouTube" and v.get("type") in ("Trailer", "Teaser")]
        videos.sort(key=lambda v: (v.get("official", False), v.get("type") == "Trailer"), reverse=True)
        overview = details.get("overview", "").strip()
        if not overview:
            raise PipelineError(f"No factual synopsis available for {item.title}; refusing to invent one.")
        crew = details.get("credits", {}).get("crew", [])
        return {
            "title": details.get("title", details.get("name")), "year": item.year,
            "kind": kind, "id": match["id"], "overview": overview,
            "release_date": details.get("release_date", details.get("first_air_date")),
            "genres": [g["name"] for g in details.get("genres", [])],
            "directors": [c["name"] for c in crew if c.get("job") == "Director"],
            "cast": [c["name"] for c in details.get("credits", {}).get("cast", [])[:5]],
            "status": details.get("status"), "trailers": videos,
            "source": f"https://www.themoviedb.org/{kind}/{match['id']}",
        }

    def collect(self, items: list[FilmInput], path: Path) -> list[dict]:
        films = []
        for item in items:
            print(f"Research: {item.title} ({item.year})", flush=True)
            films.append(self.resolve(item))
        save_json(path, films)
        return films
