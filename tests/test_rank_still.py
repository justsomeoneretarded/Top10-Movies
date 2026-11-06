from pathlib import Path
from types import SimpleNamespace

from movie_pipeline.rank_intro import rank_still_time


def test_rank_reveal_uses_clean_face_instead_of_trailer_marketing():
    source = Path("film-extra-4.mp4")
    shots = [{"start": 0., "end": 4., "text_card": "promo"},
             {"start": 4., "end": 8.}, {"start": 8., "end": 12.}]
    selector = SimpleNamespace(faces={str(source): [False, True, True]},
        card_texts={str(source): ["IN HISTORY", "", "IN CINEMAS"]},
        card_kind=lambda path, i: "promo" if i == 0 else None)
    preview = {"source": str(source), "start": 0., "duration": 10.}
    assert rank_still_time(selector, source, shots, preview) == 6.
    assert preview["start"] == 0. and preview["duration"] == 10.


def test_rank_reveal_never_borrows_faces_from_another_source():
    source = Path("film.mp4")
    selector = SimpleNamespace(faces={"unrelated.mp4": [True]})
    preview = {"start": 2., "duration": 3.}
    assert rank_still_time(selector, source, [], preview) == 2.


def test_rank_face_evidence_uses_indexed_midpoint_not_extended_window():
    source = Path("film.mp4")
    selector = SimpleNamespace(faces={str(source): [True, False]},
                               card_kind=lambda *args: None)
    shots = [{"start": 0., "end": 12.}, {"start": 12., "end": 24.}]
    preview = {"start": 14., "duration": 6.}
    assert rank_still_time(selector, source, shots, preview) == 6.


def test_rank_reveal_without_safe_frame_keeps_original_fallback():
    source = Path("film.mp4")
    preview = {"start": 2., "duration": 3.}
    selector = SimpleNamespace(card_kind=lambda *args: "promo")
    assert rank_still_time(selector, source, [{"start": 0., "end": 6.}], preview) == 2.
