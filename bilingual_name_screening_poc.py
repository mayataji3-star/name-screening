"""Thin demo wrapper around the Name Screening MVP modules."""

from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parent
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from name_screening.config import AppConfig
from name_screening.data_io import load_watchlist
from name_screening.models import ScreenRequest
from name_screening.screener import NameScreener


def run_test_case(screener: NameScreener, title: str, request: ScreenRequest) -> None:
    print(f"\n=== {title} ===")
    print(
        f"Input => name='{request.name}', dob='{request.dob}', "
        f"nationality='{request.nationality}'"
    )
    response = screener.screen(request)
    rows = [
        {
            "rank": m.rank,
            "entity_id": m.entity_id,
            "matched_name": m.matched_name,
            "matched_dob": m.matched_dob,
            "matched_nationality": m.matched_nationality,
            "cosine_similarity": round(m.cosine_similarity, 6),
            "similarity_pct": m.similarity_pct,
        }
        for m in response.matches
    ]
    print(pd.DataFrame(rows).to_string(index=False))
    print(
        f"Decision => {response.decision} "
        f"(top={response.top_score_pct:.1f}%, threshold={response.threshold_pct:.1f}%)"
    )


if __name__ == "__main__":
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")

    config = AppConfig()
    watchlist_df = load_watchlist(str(config.watchlist_path))
    screener = NameScreener.from_config(config)
    screener.ensure_index(watchlist_df)

    tests = [
        (
            "Test 1 - Exact Match (English)",
            ScreenRequest(name="Tariq Al-Hashimi", dob="1985-04-12", nationality="Yemen"),
        ),
        (
            "Test 2 - Exact Match (Arabic Script)",
            ScreenRequest(name="طارق الهاشمي", dob="1985-04-12", nationality="اليمن"),
        ),
        (
            "Test 3 - Contextual False Positive Check",
            ScreenRequest(name="Tariq Al-Hashimi", dob="1999-12-31", nationality="Morocco"),
        ),
        (
            "Test 4 - Near Match With Typo",
            ScreenRequest(name="Tarik Al Hashimi", dob="1985-04-12", nationality="Yemen"),
        ),
        (
            "Test 5 - Transliteration Variant",
            ScreenRequest(name="Tarek Al-Hashemi", dob="1985-04-12", nationality="Yemen"),
        ),
        (
            "Test 6 - Arabic Context Mismatch",
            ScreenRequest(name="طارق الهاشمي", dob="1997-01-01", nationality="المغرب"),
        ),
    ]

    for title, request in tests:
        run_test_case(screener, title, request)
