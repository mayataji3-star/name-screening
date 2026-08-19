from __future__ import annotations

import re
import unicodedata


# Arabic vowel marks and pronunciation marks
_ARABIC_DIACRITICS_RE = re.compile(
    r"[\u0610-\u061A\u064B-\u065F\u0670\u06D6-\u06ED]"
)

# Arabic stretching character: ـ
_TATWEEL = "\u0640"

# Convert common Arabic letter forms to one form
_ARABIC_LETTER_MAP = str.maketrans(
    {
        "أ": "ا",
        "إ": "ا",
        "آ": "ا",
        "ٱ": "ا",
        "ة": "ه",
        "ى": "ي",
    }
)

_MULTI_SPACE_RE = re.compile(r"\s+")

_ARABIC_DEFINITE_ARTICLE_RE = re.compile(
    r"(?<!\S)ال(?=[\u0621-\u063A\u0641-\u064A]{2,})"
)

_LATIN_NAME_ARTICLE_RE = re.compile(
    r"\bal\s+(?=[a-z])"
)

def name_comparison_forms(value: str) -> tuple[str, ...]:
    """Return original and particle-insensitive comparison forms."""
    normalized = normalize_name(value)

    without_name_articles = _ARABIC_DEFINITE_ARTICLE_RE.sub(
        "",
        normalized,
    )

    # Remove separated article: al taji → taji
    without_name_articles = _LATIN_NAME_ARTICLE_RE.sub(
        "",
        without_name_articles,
    )

    # Remove attached article: altaji/eltaji → taji
    without_name_articles = _ATTACHED_LATIN_NAME_ARTICLE_RE.sub(
        "",
        without_name_articles,
    )

    return tuple(
        dict.fromkeys(
            (normalized, without_name_articles)
        )
    )

def _replace_punctuation_with_spaces(value: str) -> str:
    """Replace punctuation and symbols with spaces."""
    characters: list[str] = []

    for character in value:
        category = unicodedata.category(character)

        if category.startswith(("P", "S")):
        #checks whether the category starts with (punctuation) or (symbol)
            characters.append(" ")
        # if it does replace it with a space 
        else:
            characters.append(character)

    return "".join(characters)

_LATIN_NAME_ARTICLE_RE = re.compile(
    r"\bal\s+(?=[a-z])"
)

_ATTACHED_LATIN_NAME_ARTICLE_RE = re.compile(
    r"\b(?:al|el)(?=[a-z]{3,}\b)"
)

def _normalize_latin_particles(value: str) -> str:
    """Standardize common Latin-written Arabic name particles."""
    text = value

    # El-Karim and El Karim become al karim.
    text = re.sub(r"\b(?:al|el)[\s-]+", "al ", text)

    # Ibn Ahmad becomes bin ahmad.
    text = re.sub(r"\bibn\b", "bin", text)

    # Abd-el-Rahman and Abd al Rahman become abd al rahman.
    text = re.sub(
        r"\babd[\s-]+(?:al|el)[\s-]+",
        "abd al ",
        text,
    )

    return text




#the main function that other files should call
def normalize_name(value: str) -> str:
    """Return a consistent form of an Arabic or Latin name."""
    #check if the input is a string, if not raise a TypeError
    if not isinstance(value, str):
        raise TypeError("Name must be a string.")

    # Standardize how Unicode characters are stored.
    text = unicodedata.normalize("NFKC", value)
    #NFKC converts compatible Unicode forms into a standard form

    # Use a strong lowercase operation for Latin text
    text = text.casefold()

    # Remove Arabic vowel marks.
    text = _ARABIC_DIACRITICS_RE.sub("", text)

    # Remove Arabic stretching.
    text = text.replace(_TATWEEL, "")

    # Standardize Arabic letters.
    text = text.translate(_ARABIC_LETTER_MAP)
    #applies the replacement table created with str.maketrans()

    # Standardize Latin name particles before removing punctuation.
    text = _normalize_latin_particles(text)

    # Change punctuation into spaces.
    text = _replace_punctuation_with_spaces(text)

    # Replace many spaces with one and remove outside spaces.
    text = _MULTI_SPACE_RE.sub(" ", text).strip()

    return text


def name_comparison_forms(value: str) -> tuple[str, ...]:
    """Return original and particle-insensitive comparison forms."""
    normalized = normalize_name(value)

    without_name_articles = _ARABIC_DEFINITE_ARTICLE_RE.sub(
        "",
        normalized,
    )

    without_name_articles = _LATIN_NAME_ARTICLE_RE.sub(
        "",
        without_name_articles,
    )

    return tuple(
        dict.fromkeys(
            (normalized, without_name_articles)
        )
    )