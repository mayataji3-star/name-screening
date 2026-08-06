from __future__ import annotations

import re
from difflib import SequenceMatcher

_MULTI_SPACE = re.compile(r"\s+")
_NON_WORD = re.compile(r"[^\w\u0600-\u06FF]+", flags=re.UNICODE)
_ARABIC_DIACRITICS = re.compile(r"[\u064B-\u065F\u0670]")
_NON_ALNUM = re.compile(r"[^a-z0-9]+")
_VOWELS = re.compile(r"[aeiou]")
_REPEATED = re.compile(r"(.)\1+")

_CHAR_EQUIVALENTS = {
    "أ": "ا",
    "إ": "ا",
    "آ": "ا",
    "ى": "ي",
    "ة": "ه",
}

_ARABIC_TO_LATIN = {
    "ا": "a",
    "أ": "a",
    "إ": "i",
    "آ": "a",
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
    "و": "u",
    "ؤ": "u",
    "ي": "i",
    "ى": "a",
    "ئ": "i",
    "ة": "h",
}


def normalize_text(value: str) -> str:
    text = value.strip().lower()
    text = _ARABIC_DIACRITICS.sub("", text)
    for source, target in _CHAR_EQUIVALENTS.items():
        text = text.replace(source, target)
    text = _NON_WORD.sub(" ", text)
    text = _MULTI_SPACE.sub(" ", text).strip()
    return text


def split_aliases(raw_aliases: str | None) -> list[str]:
    if not raw_aliases:
        return []
    values = [v.strip() for v in str(raw_aliases).replace(",", "|").split("|")]
    return [v for v in values if v]


def phonetic_key(value: str) -> str:
    base = normalize_text(value)
    if not base:
        return ""
    chars: list[str] = []
    for ch in base:
        chars.append(_ARABIC_TO_LATIN.get(ch, ch))
    latin = "".join(chars).lower()
    # Common transliteration harmonization.
    replacements = [
        ("ai", "ay"),
        ("ei", "ay"),
        ("ou", "u"),
        ("aa", "a"),
        ("ee", "i"),
        ("kh", "h"),
        ("ph", "f"),
        ("q", "k"),
        ("y", "i"),
    ]
    for left, right in replacements:
        latin = latin.replace(left, right)
    latin = _NON_ALNUM.sub("", latin)
    latin = _VOWELS.sub("", latin)
    latin = _REPEATED.sub(r"\1", latin)
    return latin


def fuzzy_ratio(left: str, right: str) -> float:
    a = normalize_text(left)
    b = normalize_text(right)
    if not a or not b:
        return 0.0
    return SequenceMatcher(None, a, b).ratio()


def equivalent_text(left: str, right: str, *, fuzzy_threshold: float = 0.86) -> bool:
    a = normalize_text(left)
    b = normalize_text(right)
    if not a or not b:
        return False
    if a == b:
        return True
    ka = phonetic_key(a)
    kb = phonetic_key(b)
    if ka and kb and ka == kb:
        return True
    return fuzzy_ratio(a, b) >= fuzzy_threshold

