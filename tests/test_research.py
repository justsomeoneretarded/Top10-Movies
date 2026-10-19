from unittest.mock import Mock

import pytest

from movie_pipeline.core import FilmInput, PipelineError
from movie_pipeline.research import Research


def resolve(movies, shows=()):
    research = object.__new__(Research)

    def get(endpoint, **params):
        if endpoint == "search/movie":
            return {"results": movies}
        if endpoint == "search/tv":
            return {"results": list(shows)}
        movie_id = int(endpoint.split("/")[1])
        match = next(x for x in movies + list(shows) if x["id"] == movie_id)
        return {**match, "overview": "An original science fiction premise.", "videos": {"results": [{"site": "YouTube", "type": "Trailer", "key": "example1234"}]}, "credits": {}}

    research.get = Mock(side_effect=get)
    return research.resolve(FilmInput("Arrival", 2016))


def film(identifier, votes, title="Arrival", year="2016"):
    return {"id": identifier, "title": title, "original_title": title, "release_date": year + "-09-01", "vote_count": votes}


def test_mainstream_arrival_not_rejected_for_obscure_namesake():
    assert resolve([film(1, 18000), film(2, 3)])["id"] == 1


def test_api_order_does_not_choose_obscure_namesake():
    assert resolve([film(2, 3), film(1, 18000)])["id"] == 1


def test_duplicate_search_records_are_not_ambiguous():
    assert resolve([film(1, 18000), film(1, 18000)])["id"] == 1


def test_truly_competing_records_still_fail_safely():
    with pytest.raises(PipelineError, match="Multiple similarly supported"):
        resolve([film(1, 100), film(2, 90)])


def test_zero_vote_upcoming_matches_not_selected_arbitrarily():
    with pytest.raises(PipelineError):
        resolve([film(1, 0), film(2, 0)])


def test_wrong_year_is_never_selected_for_popularity():
    assert resolve([film(2, 20000, year="2015"), film(1, 20)])["id"] == 1


def test_same_name_tv_record_does_not_override_dominant_movie():
    show = {"id": 2, "name": "Arrival", "original_name": "Arrival", "first_air_date": "2016-01-01", "vote_count": 1}
    assert resolve([film(1, 18000)], [show])["kind"] == "movie"


def coherence_research(release_year=None):
    research = object.__new__(Research)
    row = film(77, 2000, title="Coherence", year="2014")

    def get(endpoint, **params):
        if endpoint == "search/movie":
            return {"results": [row]}
        if endpoint == "search/tv":
            return {"results": []}
        if endpoint == "movie/77/release_dates":
            return {"results": [{"release_dates": [{"release_date": f"{release_year}-09-19T00:00:00Z"}]}]} if release_year else {"results": []}
        if endpoint == "find/tt2866360":
            return {"movie_results": [row]}
        return {**row, "overview": "Friends gather for dinner during an unusual astronomical event.", "videos": {"results": [{"site": "YouTube", "type": "Trailer"}]}, "credits": {}}

    research.get = Mock(side_effect=get)
    return research


def test_festival_release_year_is_valid_even_if_search_uses_theatrical_year():
    research = coherence_research(2013)
    research.wikidata_candidates = Mock(side_effect=AssertionError("Dated release already verifies the input"))
    result = research.resolve(FilmInput("Coherence", 2013))
    assert result["id"] == 77
    assert result["year"] == 2013
    assert result["release_date"].startswith("2014")
    assert "festival" in result["year_evidence"]


def test_nearby_year_alone_never_accepts_the_wrong_film():
    research = coherence_research(2014)
    research.wikidata_candidates = Mock(return_value=[])
    with pytest.raises(PipelineError, match="does not establish that your input is wrong"):
        research.resolve(FilmInput("Coherence", 2013))


def test_wikidata_year_and_imdb_identity_fallback(monkeypatch):
    research = coherence_research()
    claims = {"P577": [{"mainsnak": {"datavalue": {"value": {"time": "+2013-09-19T00:00:00Z", "precision": 11}}}}],
              "P345": [{"mainsnak": {"datavalue": {"value": "tt2866360"}}}]}
    payloads = [{"search": [{"id": "QTEST"}]}, {"entities": {"QTEST": {"id": "QTEST", "labels": {"en": {"value": "Coherence"}}, "claims": claims}}}]
    responses = [Mock(json=Mock(return_value=p)) for p in payloads]
    getter = Mock(side_effect=responses)
    monkeypatch.setattr("movie_pipeline.research.requests.get", getter)
    result = research.resolve(FilmInput("Coherence", 2013))
    assert result["imdb_id"] == "tt2866360"
    assert result["identity_source"] == "https://www.wikidata.org/wiki/QTEST"
    for call in getter.call_args_list:
        assert "Authorization" not in call.kwargs["headers"]


def test_unavailable_independent_source_does_not_invent_identity(monkeypatch):
    import requests
    research = coherence_research()
    monkeypatch.setattr("movie_pipeline.research.requests.get", Mock(side_effect=requests.ConnectionError("unavailable")))
    with pytest.raises(PipelineError, match="Could not verify"):
        research.resolve(FilmInput("Coherence", 2013))
