from movie_pipeline.packaging import (
    MIDROLLS,
    build_description,
    featured_heading,
    midroll_text,
    topic_hashtags,
)


def test_description_keeps_two_spoken_paragraphs_and_correct_chapters():
    intro = "Stop scrolling. Your next favorite movie might be here."
    second = "Number two is a tense chase with a great lead performance."
    script = {"sections": [{"kind": "intro", "paragraphs": [intro]},
                           {"kind": "film", "paragraphs": [second, "A later paragraph."]}],
              "package": {"title": "Top 2 Action Movies", "description": "Looking for action movies to watch? Start with these two picks."}}
    films = [{"title": "First", "year": 2025, "kind": "movie", "genres": ["Action"]},
             {"title": "Second", "year": 2024, "kind": "movie", "genres": ["Thriller"]}]
    chapters = [{"start": 0, "title": "Intro"}, {"start": 12.7, "title": "#2 Second"},
                {"start": 80.5, "title": "#1 First"}, {"start": 150, "title": "Outro"}]
    result = build_description(script, films, chapters, ["https://example.org/trailer"])
    assert result.startswith(intro+"\n\n"+second+"\n\n")
    assert "Looking for action movies to watch?" in result
    assert "00:00 Intro\n00:12 #2 Second\n01:20 #1 First\n02:30 Outro" in result
    assert "Featured movies\n" in result
    assert "#ActionMovies" in result
    assert "Film information: TMDB" not in result
    assert "A later paragraph." not in result


def test_description_removes_requested_legacy_disclaimer_and_lists_rentals_separately():
    script = {"sections": [], "package": {"title": "Best Shows", "description": "Try these shows.\nFilm information: TMDB. This product uses the TMDB API but is not endorsed or certified by TMDB."}}
    films = [{"title": "A Show", "year": 2025, "kind": "tv", "genres": ["Drama"],
              "watch_availability": {"subscription": ["Hulu"], "rent": ["Prime Video"], "checked_at": "2026-10-20T12:00:00Z"}}]
    result = build_description(script, films, [], [])
    assert "TMDB" not in result
    assert "Featured shows / series" in result
    assert "Stream: Hulu; Rent: Prime Video" in result
    assert "US listings, checked 2026-10-20" in result
    assert "#TVRecommendations" in result


def test_ctas_are_short_and_most_request_like_and_subscribe():
    assert all(len(text.split()) <= 16 for text in MIDROLLS)
    assert sum("like" in text.casefold() and "subscribe" in text.casefold() for text in MIDROLLS) > len(MIDROLLS)/2
    assert midroll_text("Example") == midroll_text("Example")


def test_topic_tags_are_relevant_distinct_and_bounded():
    films = [{"kind": "movie", "genres": ["Horror", "Thriller", "Mystery", "Fantasy"]}]
    tags = topic_hashtags("Top 10 Horror Movies", films)
    assert 3 <= len(tags) <= 5
    assert len(tags) == len(set(tags))
    assert "#HorrorMovies" in tags
    assert "#ActionMovies" not in tags
    assert featured_heading([{"kind": "movie"}, {"kind": "tv"}]) == "Featured movies and shows / series"


def test_writer_appends_checked_availability_and_outro_cta():
    from unittest.mock import Mock

    from movie_pipeline.script import Writer
    writer = Writer()
    writer.narration = Mock(return_value=["A clear and casual recommendation."])
    writer.ask = Mock(return_value={"title": "Two movies to watch", "description": "Try these two movie picks.",
                                   "thumbnail_texts": ["WATCH THESE", "GREAT MOVIES", "NEXT WATCH"]})
    writer.thumbnail_concepts = Mock(return_value=[])
    films = [{"title": "First", "year": 2025, "status": "Released", "kind": "movie",
              "watch_availability": {"subscription": ["Hulu"]}},
             {"title": "Second", "year": 2024, "status": "Released", "kind": "movie"}]
    result = writer.write("Top 2 Movies", "Keep it relaxed", films, target_seconds=300)
    review = next(section for section in result['sections'] if section['kind'] == 'film' and section['film'] == 0)
    assert 'Hulu' in review['paragraphs'][-1]
    assert 'US' not in review['paragraphs'][-1]
    outro = ' '.join(result['sections'][-1]['paragraphs']).casefold()
    assert all(word in outro for word in ('like', 'subscribe', 'other picks'))
    assert result['sections'][1]['film'] == 1  # supplied best-first list plays in countdown order
    assert 'plain everyday search words' in writer.ask.call_args.args[0]


def test_refresh_availability_replaces_old_provider_line_and_keeps_review():
    from movie_pipeline.packaging import refresh_availability
    old = "It's listed on Hulu with a subscription."
    script = {'sections': [{'kind': 'film', 'film': 0, 'paragraphs': ['Original review.', old], 'availability_text': old}]}
    films = [{'watch_availability': {'subscription': ['Netflix']}}]
    refresh_availability(script, films)
    section = script['sections'][0]
    assert section['paragraphs'][0] == 'Original review.'
    assert len(section['paragraphs']) == 2
    assert 'Netflix' in section['paragraphs'][-1] and 'Hulu' not in section['paragraphs'][-1]
    refresh_availability(script, films)
    assert len(section['paragraphs']) == 2
    refresh_availability(script, [{'watch_availability': {}}])
    assert section['paragraphs'] == ['Original review.']
