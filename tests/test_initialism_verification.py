import pytest
from movie_pipeline.voice import speech_check, _aligned_tokens


@pytest.mark.parametrize("expected,heard", [
    ("Number 4: The Man from U.N.C.L.E.", "Number four. The Man from UNCLE."),
    ("The Man from UNCLE", "The Man from U N C L E"),
    ("A report from U.N.E.S.C.O.", "A report from UNESCO"),
    ("Agents of S.H.I.E.L.D.", "Agents of SHIELD"),
])
def test_initialism_segmentation_keeps_complete_speech(expected, heard):
    check = speech_check(expected, [{"start": 0, "end": 5, "text": heard}])
    assert check["accepted"]
    assert check["coverage"] == pytest.approx(1)


@pytest.mark.parametrize("heard", ["Number four. The Man from", "Number four. The Man from UNCLA"])
def test_missing_or_wrong_initialism_is_not_accepted(heard):
    assert not speech_check("Number 4: The Man from U.N.C.L.E.",
                            [{"start": 0, "end": 5, "text": heard}])["accepted"]


def test_long_initialism_join_does_not_join_ordinary_words():
    expected, heard = _aligned_tokens(["this", "is", "not", "spoken"], ["thisisnotspoken"])
    assert not expected and not heard
