import pytest

from movie_pipeline.voice import sound_key, speech_check


@pytest.fixture(autouse=True)
def no_custom_pronunciations(monkeypatch):
    monkeypatch.setattr("movie_pipeline.voice.custom_pronunciations", lambda: {})


def test_sound_alike_names_pass_the_speech_check():
    assert speech_check("Number 5: Dredd.", [{"start": 0, "end": 2, "text": "number five dread"}])["accepted"]
    assert speech_check("Number 4: Carrie.", [{"start": 0, "end": 2, "text": "Number four, carry."}])["accepted"]
    assert not speech_check("Number 5: Dredd.", [{"start": 0, "end": 2, "text": "number five bread"}])["accepted"]
    assert not speech_check("Number 4: Carrie.", [{"start": 0, "end": 2, "text": "number four, cat"}])["accepted"]


def test_voicing_confusions_pass_but_other_words_do_not():
    assert speech_check("Number 4: Bandidos.", [{"start": 0, "end": 2, "text": "Number four, Banditos."}])["accepted"]
    assert not speech_check("Number 4: Bandidos.", [{"start": 0, "end": 2, "text": "Number four, bananas."}])["accepted"]


@pytest.mark.parametrize("heard", [
    "Number 5: Bandidos.",
    "Number 5 Banditos",
    "Number five, Banditos.",
    "Number five, Bandidos.",
])
def test_failed_production_announcement_variants_match_completely(heard):
    result = speech_check("Number 5: Bandidos.", [{"start": 0, "end": 2, "text": heard}])
    assert result["accepted"], result
    assert result["coverage"] == 1
    assert result["precision"] == 1
    assert result["heard"] == heard


@pytest.mark.parametrize("heard", [
    "Number five",
    "Number five bananas",
    "Number four Banditos",
    "5. Banditos",
    "Banditos",
    "Number five Banditos. Number five Banditos.",
])
def test_sound_alikes_do_not_accept_missing_wrong_or_repeated_speech(heard):
    result = speech_check("Number 5: Bandidos.", [{"start": 0, "end": 2, "text": heard}])
    assert not result["accepted"], result


def test_short_sound_keys_do_not_create_unrelated_word_matches():
    assert sound_key("Bandidos") == sound_key("Banditos")
    assert sound_key("Bandidos") != sound_key("bananas")
    assert sound_key("cat") == ""
    assert sound_key("core") == ""
