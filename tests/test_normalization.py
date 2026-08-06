from name_screening.normalization import normalize_name, split_aliases, transliteration_variants


def test_normalize_name_removes_noise() -> None:
    assert normalize_name("  Táriq---Al  Hashimi  ") == "táriq al hashimi"


def test_split_aliases_pipe_delimited() -> None:
    assert split_aliases("A|B| C ") == ["A", "B", "C"]


def test_transliteration_variants_include_base() -> None:
    variants = transliteration_variants("Khaled")
    assert "khaled" in variants
