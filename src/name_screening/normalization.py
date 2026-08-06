from __future__ import annotations

import re


_ARABIC_DIACRITICS_RE = re.compile(r"[\u064B-\u065F\u0670]")
_NON_WORD_SPACES_RE = re.compile(r"[^\w\u0600-\u06FF]+", flags=re.UNICODE)
_MULTI_SPACE_RE = re.compile(r"\s+")
_NON_ALNUM_RE = re.compile(r"[^a-z0-9]+")
_VOWELS_RE = re.compile(r"[aeiou]")
_REPEATED_CHAR_RE = re.compile(r"(.)\1+")

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
    "ئ": "y",
    "ة": "h",
}


def normalize_name(value: str) -> str:
    """Normalize Arabic/English names for robust matching."""
    text = value.strip().lower()
    text = _ARABIC_DIACRITICS_RE.sub("", text)
    text = _NON_WORD_SPACES_RE.sub(" ", text)
    text = _MULTI_SPACE_RE.sub(" ", text).strip()
    return text


def transliteration_variants(value: str) -> list[str]:
    """Generate lightweight transliteration-friendly variants."""
    base = normalize_name(value)
    variants = {base}
    replacements = [
        ("ph", "f"),
        ("q", "k"),
        ("kh", "h"),
        ("ou", "u"),
        ("aa", "a"),
        ("ee", "i"),
        ("-", " "),
    ]
    for left, right in replacements:
        variants.add(base.replace(left, right))
    return sorted(v for v in variants if v)


def phonetic_key(value: str) -> str:
    """Build a coarse Arabic/English comparable key."""
    base = normalize_name(value)
    if not base:
        return ""
    chars: list[str] = []
    for ch in base:
        chars.append(_ARABIC_TO_LATIN.get(ch, ch))
    latin = "".join(chars).lower()
    latin = _NON_ALNUM_RE.sub("", latin)
    latin = _VOWELS_RE.sub("", latin)
    latin = _REPEATED_CHAR_RE.sub(r"\1", latin)
    return latin


def equivalent_text(left: str, right: str) -> bool:
    left_norm = normalize_name(left)
    right_norm = normalize_name(right)
    if not left_norm or not right_norm:
        return False
    if left_norm == right_norm:
        return True
    return phonetic_key(left_norm) == phonetic_key(right_norm)


def split_aliases(raw_aliases: str | None) -> list[str]:
    if not raw_aliases:
        return []
    aliases = [a.strip() for a in str(raw_aliases).split("|")]
    return [a for a in aliases if a]
