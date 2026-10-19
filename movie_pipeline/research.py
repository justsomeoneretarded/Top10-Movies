from __future__ import annotations

import os
import re
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

    @staticmethod
    def title_score(item: FilmInput, found: dict, kind: str) -> float:
        names = (found.get("title" if kind == "movie" else "name", ""),
                 found.get("original_title" if kind == "movie" else "original_name", ""))
        return max(SequenceMatcher(None, normalized(item.title), normalized(name or "")).ratio() for name in names)

    def wikidata_candidates(self, item: FilmInput) -> list:
        """Independent identity fallback; never send the TMDB bearer token to Wikimedia."""
        endpoint = "https://www.wikidata.org/w/api.php"
        headers = {"User-Agent": "Top10Movies/0.1 (https://github.com/justsomeoneretarded/Top10-Movies)"}
        try:
            response = requests.get(endpoint, params={"action": "wbsearchentities", "search": item.title, "language": "en", "limit": 5, "format": "json"}, headers=headers, timeout=20)
            response.raise_for_status()
            ids = [x["id"] for x in response.json().get("search", [])]
            if not ids:
                return []
            response = requests.get(endpoint, params={"action": "wbgetentities", "ids": "|".join(ids), "props": "labels|aliases|claims", "languages": "en", "format": "json"}, headers=headers, timeout=20)
            response.raise_for_status()
            candidates = []
            for entity in response.json().get("entities", {}).values():
                names = [entity.get("labels", {}).get("en", {}).get("value", "")]
                names += [a.get("value", "") for a in entity.get("aliases", {}).get("en", [])]
                if normalized(item.title) not in {normalized(name) for name in names}:
                    continue
                claims = entity.get("claims", {})
                dates = [c.get("mainsnak", {}).get("datavalue", {}).get("value", {}) for c in claims.get("P577", []) if c.get("rank") != "deprecated"]
                if not any(isinstance(d, dict) and d.get("time", "")[1:5] == str(item.year) and d.get("precision", 0) >= 9 for d in dates):
                    continue
                for claim in claims.get("P345", []):
                    imdb = claim.get("mainsnak", {}).get("datavalue", {}).get("value", "")
                    if claim.get("rank") == "deprecated" or not isinstance(imdb, str) or not re.fullmatch(r"tt\d{7,10}", imdb):
                        continue
                    found = self.get(f"find/{imdb}", external_source="imdb_id", language="en-US")
                    for kind, group in (("movie", "movie_results"), ("tv", "tv_results")):
                        for row in found.get(group, []):
                            score = self.title_score(item, row, kind)
                            date = row.get("release_date" if kind == "movie" else "first_air_date", "")
                            if score >= 0.88 and date[:4].isdigit() and abs(int(date[:4]) - item.year) <= 2:
                                row = {**row, "_year_evidence": f"Wikidata {entity['id']} publication date", "_imdb_id": imdb, "_identity_source": f"https://www.wikidata.org/wiki/{entity['id']}"}
                                candidates.append((score, kind, row))
            return candidates
        except (requests.RequestException, ValueError, KeyError):
            print("Independent Wikidata lookup unavailable; no unverified year match accepted.", flush=True)
            return []

    def resolve(self, item: FilmInput) -> dict:
        candidates = []
        nearby = []
        for kind in ("movie", "tv"):
            data = self.get(f"search/{kind}", query=item.title, language="en-US", include_adult="false")
            for found in data.get("results", []):
                date = found.get("release_date" if kind == "movie" else "first_air_date", "")
                similarity = self.title_score(item, found, kind)
                if date[:4] == str(item.year) and similarity >= 0.88:
                    candidates.append((similarity, kind, found))
                elif kind == "movie" and date[:4].isdigit() and abs(int(date[:4]) - item.year) <= 2 and similarity >= 0.88:
                    nearby.append((similarity, kind, found))
        if not candidates:
            # Festival premieres and regional theatrical dates can precede the search year.
            for similarity, kind, found in sorted(nearby, key=lambda c: c[0], reverse=True)[:5]:
                releases = self.get(f"movie/{found['id']}/release_dates")
                dated = [r.get("release_date", "") for country in releases.get("results", []) for r in country.get("release_dates", [])]
                if any(date[:4] == str(item.year) for date in dated):
                    candidates.append((similarity, kind, {**found, "_year_evidence": "TMDB dated festival/regional release record"}))
        if not candidates:
            candidates = self.wikidata_candidates(item)
        # Search endpoints can return duplicate rows. A duplicate must never create ambiguity.
        candidates = list({(kind, found["id"]): (score, kind, found) for score, kind, found in candidates}.values())
        candidates.sort(key=lambda x: (x[0], x[2].get("vote_count", 0) or 0), reverse=True)
        if not candidates:
            raise PipelineError(f"Could not verify {item.title} ({item.year}) using search dates, dated release records, or the independent Wikidata/IMDb-ID fallback. This does not establish that your input is wrong.")
        contenders = [candidate for candidate in candidates if candidates[0][0] - candidate[0] < 0.025]
        if len(contenders) > 1:
            # Exact title/year matches outrank near matches; retain movie/TV neutrality.
            exact = [candidate for candidate in contenders if candidate[0] == 1.0]
            if exact:
                contenders = exact
        if len(contenders) > 1:
            # Mainstream features often share a title/year with obscure shorts or TV entries.
            # Require strong evidence, rather than choosing the API's first result or popularity alone.
            contenders.sort(key=lambda c: c[2].get("vote_count", 0) or 0, reverse=True)
            first_votes = contenders[0][2].get("vote_count", 0) or 0
            second_votes = contenders[1][2].get("vote_count", 0) or 0
            if first_votes < 50 or first_votes < 10 * max(1, second_votes):
                raise PipelineError(
                    f"Multiple similarly supported title/year matches for {item.title} ({item.year}). "
                    "Unable to choose confidently; check the official title and first-release year."
                )
            print(f"Resolved competing matches for {item.title} using a dominant audience-vote count.", flush=True)
        _, kind, match = contenders[0]
        if match.get("_year_evidence"):
            print(f"Verified {item.title} ({item.year}) via {match['_year_evidence']} (search release date: {match.get('release_date', '')}).", flush=True)
        details = self.get(f"{kind}/{match['id']}", append_to_response="videos,credits,external_ids", language="en-US")
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
            "year_evidence": match.get("_year_evidence", "TMDB search release date"),
            "imdb_id": match.get("_imdb_id") or details.get("imdb_id") or details.get("external_ids", {}).get("imdb_id"),
            "identity_source": match.get("_identity_source"),
            "source": f"https://www.themoviedb.org/{kind}/{match['id']}",
        }

    def collect(self, items: list[FilmInput], path: Path) -> list[dict]:
        films = []
        for item in items:
            print(f"Research: {item.title} ({item.year})", flush=True)
            films.append(self.resolve(item))
        save_json(path, films)
        return films
