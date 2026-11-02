from pathlib import Path
from types import SimpleNamespace
import pytest
from movie_pipeline.editorial_policy import explicit_genre, search_queries, image_queries, source_identity, clean_review
from movie_pipeline.moments import complete_window
from movie_pipeline.resource_budget import DownloadBudget

def test_unspecified_and_mixed_genres_are_neutral():
    assert explicit_genre("8 Movies of 2026", "A mixed list for your watchlist.") == "neutral"
    assert explicit_genre("8 Shows", "Medical drama, fantasy, sci-fi and horror.") == "neutral"
    assert explicit_genre("8 Horror Movies", "A mix of drama and mystery.") == "horror"

def test_queries_are_per_title_without_network_and_shows_without_year():
    film={"title":"Example Show","year":2023,"kind":"tv","genres":["Comedy"],"networks":["Do Not Search Me"]}
    queries=search_queries(film)
    assert "Example Show best scene" in queries
    assert any("funny scene" in q for q in queries)
    assert all("Do Not Search Me" not in q and "action scene" not in q for q in queries)
    assert any("official trailer" in q for q in search_queries(film,primary=True))
    assert any("best scene image" in q for q in image_queries(film))

@pytest.mark.parametrize("suffix",["","-extra","-extra-3","-extra-4","-extra-5"])
def test_source_identity_survives_all_extra_slots(suffix):
    assert source_identity(Path("movie-42"+suffix+".mp4"))=="movie-42"

def test_filler_removed_without_padding_or_losing_premise():
    paragraphs=["Two brothers get stranded. If you liked First Blood, see this.",
                "You can find it on Prime Video and Apple TV.","Their only route home crosses a flooded town."]
    assert clean_review(paragraphs)==["Two brothers get stranded.","Their only route home crosses a flooded town."]

def test_native_scene_grows_to_complete_dialogue():
    c={"start":10.0,"duration":5.0,"source":"x.mp4"}
    cues=[{"start":9.6,"end":15.5,"text":"We need to leave before the bridge falls."}]
    result=complete_window(c,cues,(0,30))
    assert result and result["start"]<9.6
    assert result["start"]+result["duration"]>15.5
    assert not result["muted"]

def test_incomplete_dialogue_is_not_used_as_an_important_native_beat():
    c={"start":10.0,"duration":5.0,"source":"x.mp4"}
    assert complete_window(c,[{"start":10.1,"end":14.9,"text":"If you think that we"}],(0,30)) is None
    assert complete_window(c,[{"start":9.6,"end":15.5,"text":"A complete thought."}],(9.6,16)) is None

def test_download_reservations_prevent_concurrent_overcommit(monkeypatch,tmp_path):
    monkeypatch.setattr("movie_pipeline.resource_budget.shutil.disk_usage",lambda _:SimpleNamespace(free=3_000_000_000))
    budget=DownloadBudget()
    with budget.claim(tmp_path) as first:
        assert first
        with budget.claim(tmp_path) as second:
            assert not second
    assert budget.pending==0

def test_image_url_filter_rejects_local_targets():
    from movie_pipeline.image_discovery import public_url
    assert not public_url("http://127.0.0.1/private")
    assert not public_url("file:///etc/passwd")
    assert not public_url("https://metadata.google.internal/x")
    assert public_url("https://example.com/picture.jpg")

def test_spoken_availability_removed_but_metadata_remains():
    from movie_pipeline.packaging import refresh_availability
    film={"watch_availability":{"subscription":["Netflix"]}}
    script={"sections":[{"kind":"film","film":0,"paragraphs":["A specific premise.","You can find it on Netflix."],
                         "availability_text":"You can find it on Netflix.","delivery":["dramatic","normal"]}]}
    refresh_availability(script,[film])
    assert script["sections"][0]["paragraphs"]==["A specific premise."]
    assert script["sections"][0]["delivery"]==["dramatic"]
    assert film["watch_availability"]["subscription"]==["Netflix"]

def test_ident_is_original_and_decodable(monkeypatch,tmp_path):
    import shutil
    if not shutil.which("ffmpeg"):
        pytest.skip("ffmpeg required")
    from movie_pipeline.branding import render_ident
    from movie_pipeline.core import duration, probe
    monkeypatch.setenv("VIDEO_RESOLUTION","320x180")
    output=tmp_path/"ident.mp4"
    render_ident(output,tmp_path/"ident")
    assert duration(output)==pytest.approx(2.3,abs=.05)
    assert {s["codec_type"] for s in probe(output)["streams"]}=={"audio","video"}
