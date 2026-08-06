from __future__ import annotations

from .normalization import normalize_name, split_aliases


def _payload(
    first_name: str,
    middle_name: str,
    last_name: str,
    dob: str,
    residency: str,
    aliases: str | None = None,
    relative_names: str | None = None,
    gender: str | None = None,
) -> str:
    name = " ".join(p for p in [first_name, middle_name, last_name] if p).strip()
    alias_tokens = split_aliases(aliases)
    relative_tokens = split_aliases(relative_names)
    alias_text = f" [ALIASES] {' | '.join(alias_tokens)}" if alias_tokens else ""
    relatives_text = (
        f" [RELATIVES] {' | '.join(relative_tokens)}" if relative_tokens else ""
    )
    gender_text = f" [GENDER] {gender}" if gender else ""
    return (
        f"[NAME] {name} [FIRST] {first_name} [MIDDLE] {middle_name} [LAST] {last_name} "
        f"[NAME_NORM] {normalize_name(name)} [DOB] {dob} [RESIDENCY] {residency}"
        f"{alias_text}{relatives_text}{gender_text}"
    )


def format_passage_record(
    first_name: str,
    middle_name: str,
    last_name: str,
    dob: str,
    residency: str,
    aliases: str | None = None,
    relative_names: str | None = None,
    gender: str | None = None,
) -> str:
    return (
        "passage: "
        + _payload(
            first_name,
            middle_name,
            last_name,
            dob,
            residency,
            aliases=aliases,
            relative_names=relative_names,
            gender=gender,
        )
    )


def format_query(
    first_name: str,
    middle_name: str,
    last_name: str,
    dob: str,
    residency: str,
    aliases: str | None = None,
    relative_names: str | None = None,
    gender: str | None = None,
) -> str:
    return (
        "query: "
        + _payload(
            first_name,
            middle_name,
            last_name,
            dob,
            residency,
            aliases=aliases,
            relative_names=relative_names,
            gender=gender,
        )
    )
