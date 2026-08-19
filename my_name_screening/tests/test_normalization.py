import pytest

from my_name_screening.normalization import name_comparison_forms, normalize_name


def test_removes_arabic_diacritics() -> None:
    assert normalize_name("مُحَمَّد") == "محمد"


def test_removes_tatweel() -> None:
    assert normalize_name("محمـــد") == "محمد"


def test_unifies_arabic_alef_forms() -> None:
    assert normalize_name("أحمد إبراهيم آدم") == "احمد ابراهيم ادم"


def test_unifies_taa_marbuta() -> None:
    assert normalize_name("فاطمة") == "فاطمه"


def test_unifies_alef_maqsura() -> None:
    assert normalize_name("مصطفى") == "مصطفي"


def test_casefolds_latin_names() -> None:
    assert normalize_name("MOHAMMAD ALI") == "mohammad ali"


def test_removes_punctuation() -> None:
    assert normalize_name("Mohammad, Ali!") == "mohammad ali"


def test_collapses_spaces() -> None:
    assert normalize_name("  Mohammad    Ali  ") == "mohammad ali"


def test_normalizes_al_and_el_particles() -> None:
    assert normalize_name("Hassan Al-Karim") == "hassan al karim"
    assert normalize_name("Hassan El Karim") == "hassan al karim"


def test_normalizes_ibn_to_bin() -> None:
    assert normalize_name("Omar Ibn Ahmad") == "omar bin ahmad"


def test_normalizes_abd_al_particle() -> None:
    assert normalize_name("Abd-el-Rahman") == "abd al rahman"
    assert normalize_name("Abd al Rahman") == "abd al rahman"


def test_rejects_non_string_input() -> None:
    with pytest.raises(TypeError):
        normalize_name(123)  # type: ignore[arg-type]

def test_name_comparison_forms_remove_al() -> None:
    assert normalize_name("Al Hashim") == "al hashim"

    assert name_comparison_forms("Al Hashim") == (
        "al hashim",
        "hashim",
    )

    assert name_comparison_forms("الهاشم") == (
        "الهاشم",
        "هاشم",
    )