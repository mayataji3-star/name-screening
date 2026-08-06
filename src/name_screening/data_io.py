from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

WATCHLIST_COLUMNS = [
    "entity_id",
    "name",
    "first_name",
    "middle_name",
    "last_name",
    "dob",
    "residency",
    "nationality",
    "aliases",
    "relative_names",
    "gender",
    "risk_level",
]


def _split_name(value: str) -> tuple[str, str, str]:
    parts = [p for p in str(value).strip().split() if p]
    if not parts:
        return "", "", ""
    if len(parts) == 1:
        return parts[0], "", ""
    if len(parts) == 2:
        return parts[0], "", parts[1]
    return parts[0], " ".join(parts[1:-1]), parts[-1]


def _ensure_structured_watchlist(df: pd.DataFrame) -> pd.DataFrame:
    data = df.copy()
    if "name" not in data.columns:
        name_cols = ["first_name", "middle_name", "last_name"]
        data["name"] = data.apply(
            lambda row: " ".join(str(row.get(col, "")).strip() for col in name_cols if str(row.get(col, "")).strip()),
            axis=1,
        )
    if "first_name" not in data.columns or "last_name" not in data.columns:
        split_parts = data["name"].fillna("").map(_split_name)
        data["first_name"] = split_parts.map(lambda p: p[0])
        data["middle_name"] = split_parts.map(lambda p: p[1])
        data["last_name"] = split_parts.map(lambda p: p[2])
    if "nationality" not in data.columns:
        data["nationality"] = ""
    data["nationality"] = data["nationality"].astype(str)
    if "residency" not in data.columns:
        data["residency"] = ""
    data["residency"] = data["residency"].astype(str)
    data["residency"] = data["residency"].where(
        data["residency"].str.strip() != "", data["nationality"]
    )
    data["nationality"] = data["nationality"].where(
        data["nationality"].str.strip() != "", data["residency"]
    )
    for col in ["aliases", "relative_names", "gender", "risk_level", "dob", "entity_id"]:
        if col not in data.columns:
            data[col] = ""
    return data[WATCHLIST_COLUMNS].fillna("")


def _default_mock_rows() -> list[dict[str, str]]:
    base_rows = [
        (
            "Hassan",
            "Mahmoud",
            "Al-Karim",
            "حسن",
            "محمود",
            "الكريم",
            "Jordan",
            "الأردن",
            "Abu Ali",
            "Ali Hassan",
            "male",
            "high",
        ),
        (
            "Noor",
            "Sameer",
            "Darwish",
            "نور",
            "سمير",
            "درويش",
            "Egypt",
            "مصر",
            "Nour Darwish",
            "Mona Darwish",
            "female",
            "medium",
        ),
        (
            "Karim",
            "Nabil",
            "Mansour",
            "كريم",
            "نبيل",
            "منصور",
            "Syria",
            "سوريا",
            "Abu Nabil",
            "Samer Karim",
            "male",
            "high",
        ),
        (
            "Omar",
            "Hadi",
            "Qadri",
            "عمر",
            "هادي",
            "القادري",
            "Iraq",
            "العراق",
            "Abu Hadi",
            "Hadi Omar",
            "male",
            "high",
        ),
        (
            "Layla",
            "Fadi",
            "Shalabi",
            "ليلى",
            "فادي",
            "شلبي",
            "Jordan",
            "الأردن",
            "Laila Shalabi",
            "Rami Shalabi",
            "female",
            "medium",
        ),
        (
            "Yousef",
            "Aziz",
            "Hammoud",
            "يوسف",
            "عزيز",
            "حمود",
            "Libya",
            "ليبيا",
            "Yusuf Hamoud",
            "Aziz Yousef",
            "male",
            "high",
        ),
        (
            "Samir",
            "Rahman",
            "Qattan",
            "سمير",
            "رحمن",
            "قطان",
            "Morocco",
            "المغرب",
            "Samir Kattan",
            "Nabil Samir",
            "male",
            "high",
        ),
        (
            "Huda",
            "Karim",
            "Nadim",
            "هدى",
            "كريم",
            "نديم",
            "Sudan",
            "السودان",
            "Houda Nadim",
            "Karim Huda",
            "female",
            "medium",
        ),
        (
            "Anwar",
            "Imad",
            "Al-Yafi",
            "أنور",
            "عماد",
            "اليافي",
            "Yemen",
            "اليمن",
            "Anwar Yafi",
            "Imad Anwar",
            "male",
            "high",
        ),
        (
            "Rami",
            "Bashir",
            "Saad",
            "رامي",
            "بشير",
            "سعد",
            "Egypt",
            "مصر",
            "Rami S.",
            "Bashir Rami",
            "male",
            "medium",
        ),
    ]
    rows: list[dict[str, str]] = []
    idx = 1
    for (
        first_en,
        middle_en,
        last_en,
        first_ar,
        middle_ar,
        last_ar,
        res_en,
        res_ar,
        alias,
        relative,
        gender,
        risk,
    ) in base_rows:
        rows.append(
            {
                "entity_id": f"M{idx:03d}",
                "name": f"{first_en} {middle_en} {last_en}",
                "first_name": first_en,
                "middle_name": middle_en,
                "last_name": last_en,
                "dob": f"19{75 + (idx % 20):02d}-{1 + (idx % 12):02d}-{1 + ((idx * 3) % 28):02d}",
                "residency": res_en,
                "nationality": res_en,
                "aliases": f"{alias}|{first_ar} {last_ar}",
                "relative_names": relative,
                "gender": gender,
                "risk_level": risk,
            }
        )
        idx += 1
        rows.append(
            {
                "entity_id": f"M{idx:03d}",
                "name": f"{first_ar} {middle_ar} {last_ar}",
                "first_name": first_ar,
                "middle_name": middle_ar,
                "last_name": last_ar,
                "dob": f"19{75 + (idx % 20):02d}-{1 + (idx % 12):02d}-{1 + ((idx * 3) % 28):02d}",
                "residency": res_ar,
                "nationality": res_ar,
                "aliases": f"{alias}|{first_en} {last_en}",
                "relative_names": relative,
                "gender": gender,
                "risk_level": risk,
            }
        )
        idx += 1
    return rows


def _default_alias_rows() -> list[dict[str, str]]:
    return [
        {"alias": "Abu Ali", "canonical_name": "Hassan Mahmoud Al-Karim"},
        {"alias": "ابو علي", "canonical_name": "حسن محمود الكريم"},
        {"alias": "Nour Darwish", "canonical_name": "Noor Sameer Darwish"},
    ]


def ensure_mock_files(watchlist_path: Path, alias_map_path: Path) -> None:
    watchlist_path.parent.mkdir(parents=True, exist_ok=True)
    alias_map_path.parent.mkdir(parents=True, exist_ok=True)
    if not watchlist_path.exists():
        pd.DataFrame(_default_mock_rows(), columns=WATCHLIST_COLUMNS).to_csv(
            watchlist_path, index=False, encoding="utf-8"
        )
    if not alias_map_path.exists():
        pd.DataFrame(_default_alias_rows(), columns=["alias", "canonical_name"]).to_csv(
            alias_map_path, index=False, encoding="utf-8"
        )


def _expand_watchlist(df: pd.DataFrame, target_size: int = 500) -> pd.DataFrame:
    df = _ensure_structured_watchlist(df)
    if len(df) >= target_size:
        return df

    rows: list[dict[str, object]] = [r for r in df.to_dict(orient="records")]
    nationality_cycle = [
        "Yemen",
        "اليمن",
        "Syria",
        "سوريا",
        "Iraq",
        "العراق",
        "Sudan",
        "السودان",
        "Libya",
        "ليبيا",
        "Jordan",
        "الأردن",
        "Morocco",
        "المغرب",
        "Egypt",
        "مصر",
    ]
    risk_levels = ["high", "high", "high", "medium"]

    english_first = [
        "Tariq", "Khaled", "Samir", "Nabil", "Hassan", "Faris", "Omar", "Yousef",
        "Karim", "Adel", "Rami", "Majid", "Nader", "Imad", "Walid", "Basel",
        "Amin", "Ziad", "Murad", "Fadi", "Nasser", "Sami", "Jamal", "Anwar",
    ]
    english_middle = [
        "Ibrahim", "Mustafa", "Abdelrahman", "Mahmoud", "Saleh", "Kamal", "Hadi",
        "Yassin", "Rashid", "Hamdan", "Fouad", "Latif", "Qassem", "Tawfiq",
        "Aziz", "Naeem", "Bashir", "Mansour", "Rahman", "Hakim",
    ]
    english_last = [
        "Al-Hashimi", "Mansour", "Qadri", "Rahman", "Al-Karim", "Jabr", "Al-Sayeed",
        "Nadim", "Al-Farouq", "Barakat", "Hammoud", "Al-Masri", "Al-Yafi",
        "Shalabi", "Qattan", "Kanaan", "Darwish", "Saad", "Zahran", "Bakri",
    ]

    arabic_first = [
        "طارق", "خالد", "سمير", "نبيل", "حسن", "فارس", "عمر", "يوسف", "كريم",
        "عادل", "رامي", "ماجد", "نادر", "عماد", "وليد", "باسل", "أمين", "زياد",
        "مراد", "فادي", "ناصر", "سامي", "جمال", "أنور",
    ]
    arabic_middle = [
        "إبراهيم", "مصطفى", "عبدالرحمن", "محمود", "صالح", "كمال", "هادي", "ياسين",
        "رشيد", "حمدان", "فؤاد", "لطيف", "قاسم", "توفيق", "عزيز", "نعيم",
        "بشير", "منصور", "رحمن", "حكيم",
    ]
    arabic_last = [
        "الهاشمي", "منصور", "القادري", "الرحمن", "الكريم", "جبر", "السيد", "نديم",
        "الفاروق", "بركات", "حمود", "المصري", "اليافي", "شلبي", "قطان", "كنعان",
        "درويش", "سعد", "زهران", "بكري",
    ]

    i = len(rows)
    for ef_idx, ef in enumerate(english_first):
        for em_idx, em in enumerate(english_middle):
            for el_idx, el in enumerate(english_last):
                if len(rows) >= target_size:
                    return pd.DataFrame(rows)
                af = arabic_first[ef_idx % len(arabic_first)]
                am = arabic_middle[em_idx % len(arabic_middle)]
                al = arabic_last[el_idx % len(arabic_last)]

                full_en = f"{ef} {em} {el}"
                full_ar = f"{af} {am} {al}"
                year = 1965 + (i % 40)
                month = 1 + (i % 12)
                day = 1 + ((i * 5) % 28)
                rows.append(
                    {
                        "entity_id": f"E{i + 1:04d}",
                        "name": full_en if i % 2 == 0 else full_ar,
                        "first_name": ef if i % 2 == 0 else af,
                        "middle_name": em if i % 2 == 0 else am,
                        "last_name": el if i % 2 == 0 else al,
                        "dob": f"{year:04d}-{month:02d}-{day:02d}",
                        "residency": nationality_cycle[i % len(nationality_cycle)],
                        "nationality": nationality_cycle[i % len(nationality_cycle)],
                        "aliases": f"{full_ar}|{full_en}|ALT-{i + 1:04d}",
                        "relative_names": "",
                        "gender": "male" if i % 2 == 0 else "female",
                        "risk_level": risk_levels[i % len(risk_levels)],
                    }
                )
                i += 1

    return _ensure_structured_watchlist(pd.DataFrame(rows))


def _pick_first(values: object) -> str:
    if not isinstance(values, list) or not values:
        return ""
    return str(values[0]).strip()


def _pick_join(values: object) -> str:
    if not isinstance(values, list):
        return ""
    cleaned = [str(v).strip() for v in values if str(v).strip()]
    return "|".join(dict.fromkeys(cleaned))


def _to_watchlist_row(entity: dict[str, object]) -> dict[str, str]:
    props = entity.get("properties", {})
    if not isinstance(props, dict):
        props = {}

    name = _pick_first(props.get("name")) or str(entity.get("caption", "")).strip()
    dob = _pick_first(props.get("birthDate"))
    nationality = _pick_first(props.get("nationality")) or _pick_first(props.get("country"))
    aliases = _pick_join(props.get("alias"))

    return {
        "entity_id": str(entity.get("id", "")).strip(),
        "name": name or "UNKNOWN",
        "first_name": _split_name(name)[0],
        "middle_name": _split_name(name)[1],
        "last_name": _split_name(name)[2],
        "dob": dob or "UNKNOWN",
        "residency": nationality or "UNKNOWN",
        "nationality": nationality or "UNKNOWN",
        "aliases": aliases,
        "relative_names": "",
        "gender": "",
        "risk_level": "high",
    }


def _iter_ftm_entities(path: Path):
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            text = line.strip()
            if not text or text in {"[", "]"}:
                continue
            # Accept newline-delimited JSON and array-style JSON lines.
            if text.endswith(","):
                text = text[:-1]
            try:
                parsed = json.loads(text)
            except json.JSONDecodeError:
                continue
            if isinstance(parsed, dict):
                yield parsed


def _load_opensanctions_watchlist(
    path: str | Path,
    *,
    entity_scope: str = "people_only",
    max_records: int | None = None,
) -> pd.DataFrame:
    rows: list[dict[str, str]] = []
    source = Path(path)
    for entity in _iter_ftm_entities(source):
        schema = str(entity.get("schema", "")).lower()
        if entity_scope == "people_only" and schema != "person":
            continue
        row = _to_watchlist_row(entity)
        if not row["entity_id"] or row["name"] == "UNKNOWN":
            continue
        rows.append(row)
        if max_records is not None and len(rows) >= max_records:
            break

    return pd.DataFrame(
        rows,
        columns=WATCHLIST_COLUMNS,
    )


def load_alias_map(path: str | Path) -> dict[str, str]:
    source = Path(path)
    if not source.exists():
        return {}
    raw = pd.read_csv(source)
    alias_map: dict[str, str] = {}
    for row in raw.to_dict(orient="records"):
        alias = str(row.get("alias", "")).strip()
        canonical = str(row.get("canonical_name", "")).strip()
        if alias and canonical:
            alias_map[alias] = canonical
    return alias_map


def load_watchlist(
    path: str,
    *,
    target_size: int = 500,
    opensanctions_path: str | None = None,
    alias_map_path: str | None = None,
    use_mock_data: bool = True,
    entity_scope: str = "people_only",
    max_records: int | None = None,
) -> pd.DataFrame:
    watchlist_file = Path(path)
    if opensanctions_path:
        ftm = Path(opensanctions_path)
        if ftm.is_file():
            loaded = _load_opensanctions_watchlist(
                ftm, entity_scope=entity_scope, max_records=max_records
            )
            if not loaded.empty:
                return _ensure_structured_watchlist(loaded)

    if use_mock_data:
        alias_file = Path(alias_map_path or watchlist_file.parent / "alias_map_mock.csv")
        ensure_mock_files(watchlist_file, alias_file)
        return _ensure_structured_watchlist(pd.read_csv(watchlist_file))

    raw = pd.read_csv(watchlist_file)
    return _expand_watchlist(raw, target_size=target_size)


def load_eval_cases(path: str) -> pd.DataFrame:
    return pd.read_csv(path)
