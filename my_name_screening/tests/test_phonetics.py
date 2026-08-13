from my_name_screening.phonetics import (
    contains_arabic,
    double_metaphone_keys,
    phonetic_key,
    phonetic_keys,
    phonetically_equivalent,
    transliterate_to_latin,
)


def test_double_metaphone_returns_codes() -> None:
    keys = double_metaphone_keys("Mohammad Ali")

    assert keys
    assert all(isinstance(key, str) for key in keys)


def test_double_metaphone_matches_latin_spelling_variants() -> None:
    mohammad_keys = double_metaphone_keys("Mohammad")
    mohammed_keys = double_metaphone_keys("Mohammed")

    assert mohammad_keys.intersection(mohammed_keys)


def test_double_metaphone_handles_transliterated_arabic() -> None:
    keys = double_metaphone_keys("محمد")

    assert keys


def test_combined_method_matches_arabic_and_latin() -> None:
    arabic_keys = phonetic_keys("محمد علي")
    latin_keys = phonetic_keys("Mohammad Ali")

    assert arabic_keys.intersection(latin_keys)


def test_combined_method_handles_name_order() -> None:
    first = phonetic_keys("Mohammad Ali")
    second = phonetic_keys("Ali Mohammad")

    assert first.intersection(second)


def test_different_names_do_not_match() -> None:
    assert not phonetically_equivalent("Ahmad", "Khalid")


def test_arabic_diacritics_do_not_change_match() -> None:
    assert phonetically_equivalent("مُحَمَّد", "محمد")


def test_common_english_spelling_variation() -> None:
    assert phonetically_equivalent("Yousef", "Yusuf")


def test_arabic_and_english_khalid_match() -> None:
    assert phonetically_equivalent("خالد", "Khalid")


def test_empty_names_do_not_match() -> None:
    assert not phonetically_equivalent("", "")