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
