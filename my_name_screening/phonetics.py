from __future__ import annotations

import re

from metaphone import doublemetaphone
from .normalization import normalize_name


_ARABIC_RE = re.compile(r"[\u0600-\u06FF]")
_NON_LATIN_RE = re.compile(r"[^a-z\s]")
_MULTI_SPACE_RE = re.compile(r"\s+")
_REPEATED_LETTER_RE = re.compile(r"(.)\1+")


# Basic Arabic-to-Latin sound mapping.
_ARABIC_TO_LATIN = {
    "ا": "a",
    "ب": "b",
    "ت": "t",
    "ث": "th",
    "ج": "j",
    "ح": "h",
    "خ": "kh",
    "د": "d",
    "ذ": "dh",
    "ر": "r",
    "ز": "z",
    "س": "s",
    "ش": "sh",
    "ص": "s",
    "ض": "d",
    "ط": "t",
    "ظ": "z",
    "ع": "a",
    "غ": "gh",
    "ف": "f",
    "ق": "q",
    "ك": "k",
    "ل": "l",
    "م": "m",
    "ن": "n",
    "ه": "h",
    "ء": "",
    "ؤ": "u",
    "ئ": "y",
}


def contains_arabic(value: str) -> bool:
    """Return True when the text contains Arabic characters."""
    return bool(_ARABIC_RE.search(value))


def _transliterate_arabic_word(word: str) -> str:
    """Convert one normalized Arabic word to a Latin form."""
    result: list[str] = []

    for position, character in enumerate(word):
        # ي usually sounds like y at the beginning and i elsewhere.
        if character == "ي":
            result.append("y" if position == 0 else "i")
            continue

        # و usually sounds like w at the beginning and u elsewhere.
        if character == "و":
            result.append("w" if position == 0 else "u")
            continue

        result.append(_ARABIC_TO_LATIN.get(character, character))

    return "".join(result)


def transliterate_to_latin(value: str) -> str:
    """Convert an Arabic or Latin name into normalized Latin text."""

    #calls the normalization function
    normalized = normalize_name(value)
    #removes diacritics, tatweel, punctuation and extra spaces

    words: list[str] = []

    for word in normalized.split():
        if contains_arabic(word):
            words.append(_transliterate_arabic_word(word))
        else:
            words.append(word)

    return " ".join(words)


def _canonicalize_latin_spelling(value: str) -> str:
    """Reduce common differences in Latin name spelling."""
    text = value

    replacements = (
        ("sch", "sh"),
        ("ph", "f"),
        ("kh", "x"),
        ("gh", "g"),
        ("sh", "c"),
        ("th", "t"),
        ("dh", "d"),
        ("ou", "u"),
        ("oo", "u"),
        ("ee", "i"),
        ("aa", "a"),
        ("q", "k"),
        ("ck", "k"),
    )

    for old, new in replacements:
        text = text.replace(old, new)

    return text



def double_metaphone_keys(value: str) -> set[str]:
    """Create Double Metaphone codes for an Arabic or Latin name."""
    latin_name = transliterate_to_latin(value)
    latin_name = _canonicalize_latin_spelling(latin_name)
    latin_name = _NON_LATIN_RE.sub(" ", latin_name)
    latin_name = _MULTI_SPACE_RE.sub(" ", latin_name).strip()

    if not latin_name:
        return set()

    word_codes: list[tuple[str, str]] = []

    for word in latin_name.split():
        primary, secondary = doublemetaphone(word)

        primary = primary or ""
        secondary = secondary or primary

        if primary:
            word_codes.append((primary, secondary))

    if not word_codes:
        return set()

    # Keep the original word order
    primary_code = "".join(primary for primary, _ in word_codes)
    secondary_code = "".join(secondary for _, secondary in word_codes)

    # Also create codes that ignore name order.
    sorted_primary_code = "".join(
        sorted(primary for primary, _ in word_codes)
    )
    sorted_secondary_code = "".join(
        sorted(secondary for _, secondary in word_codes)
    )

    return {
        code
        for code in {
            primary_code,
            secondary_code,
            sorted_primary_code,
            sorted_secondary_code,
        }
        if code
    }


def _word_phonetic_key(word: str) -> str:
    """Build a sound-based key for one Latin word."""
    text = _canonicalize_latin_spelling(word)

    # Keep only Latin letters.
    text = re.sub(r"[^a-z]", "", text)

    if not text:
        return ""

    # Remove vowels because their spelling changes frequently.
    text = re.sub(r"[aeiou]", "", text)

    # Remove repeated letters: mm becomes m
    text = _REPEATED_LETTER_RE.sub(r"\1", text)

    return text
#gained a phonetic key

def phonetic_key(value: str, *, sort_words: bool = False) -> str:
    """Create a comparable sound-based key for a person's name."""
    latin = transliterate_to_latin(value)
    latin = _NON_LATIN_RE.sub(" ", latin)
    latin = _MULTI_SPACE_RE.sub(" ", latin).strip()

    

    #Create a key for each word
    word_keys = [
        key
        for word in latin.split()
        if (key := _word_phonetic_key(word))
    ]

    if sort_words:
        word_keys.sort()

    return "".join(word_keys)


def phonetic_keys(value: str) -> set[str]:
    """Return custom and Double Metaphone phonetic keys."""
    custom_ordered = phonetic_key(value)
    custom_sorted = phonetic_key(value, sort_words=True)

    custom_keys = {
        key
        for key in {
            custom_ordered,
            custom_sorted,
        }
        if key
    }

    metaphone_keys = double_metaphone_keys(value)

    return custom_keys.union(metaphone_keys)
def phonetically_equivalent(left: str, right: str) -> bool:
    """Check whether two names share at least one phonetic key."""
    left_keys = phonetic_keys(left)
    right_keys = phonetic_keys(right)

    if not left_keys or not right_keys:
        return False

    return bool(left_keys.intersection(right_keys))