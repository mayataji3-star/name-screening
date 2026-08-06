from __future__ import annotations

from pathlib import Path

import pandas as pd

from sota_screening.feedback import append_feedback, summarize_feedback
from sota_screening.models import FeedbackEvent
from sota_screening.models import SotaScreenRequest
from sota_screening.normalization import equivalent_text, normalize_text, split_aliases
from sota_screening.policy import DecisionBandPolicy
from sota_screening.reranker import ConflictAwareReranker


def test_normalization_and_alias_split() -> None:
    assert normalize_text("  أحمَد   ") == "احمد"
    assert split_aliases("a|b, c") == ["a", "b", "c"]
    assert equivalent_text("Layla", "Laila")
    assert equivalent_text("Hassan", "Hasan")


def test_decision_band_policy() -> None:
    policy = DecisionBandPolicy(auto_clear_threshold=0.35, review_threshold=0.7, auto_hit_threshold=0.9)
    assert policy.decide(0.2) == "AUTO_CLEAR"
    assert policy.decide(0.8) == "REVIEW"
    assert policy.decide(0.95) == "AUTO_HIT"


def test_feedback_summary(tmp_path: Path) -> None:
    path = tmp_path / "feedback.jsonl"
    append_feedback(
        path,
        FeedbackEvent(
            input_name="Test Name",
            entity_id="N0001",
            analyst_label="MATCH",
            predicted_decision="REVIEW",
            predicted_score=0.8,
        ),
    )
    summary = summarize_feedback(path)
    assert summary["events"] == 1.0
    assert summary["match_rate"] == 1.0


def test_reranker_handles_transliteration_variants() -> None:
    reranker = ConflictAwareReranker()
    request = SotaScreenRequest(
        name="Layla Fadi Shalabi",
        aliases=["Laila Shalabi"],
        relative_names=["Ali Hassan"],
        residency="Jordan",
    )
    candidates = pd.DataFrame(
        [
            {
                "entity_id": "N1",
                "name": "Laila Fady Shalaby",
                "aliases": "Layla Shalabi|ليلى شلبي",
                "relative_names": "Ali Hasan",
                "residency": "Jordan",
                "gender": "",
                "dob": "UNKNOWN",
            }
        ]
    )
    scored = reranker.rerank(request, candidates)
    assert float(scored.iloc[0]["rerank_score"]) >= 0.6


def test_reranker_marks_hard_dob_conflict() -> None:
    reranker = ConflictAwareReranker()
    request = SotaScreenRequest(
        first_name="Layla",
        middle_name="Fadi",
        last_name="Shalabi",
        dob="1993-06-18",
        residency="Jordan",
        nationality="Jordan",
    )
    candidates = pd.DataFrame(
        [
            {
                "entity_id": "N2",
                "name": "Layla Fadi Shalabi",
                "first_name": "Layla",
                "middle_name": "Fadi",
                "last_name": "Shalabi",
                "aliases": "",
                "relative_names": "",
                "residency": "Jordan",
                "nationality": "Jordan",
                "gender": "",
                "dob": "1992-06-18",
            }
        ]
    )
    scored = reranker.rerank(request, candidates)
    assert bool(scored.iloc[0]["has_dob_conflict"]) is True
    assert float(scored.iloc[0]["rerank_score"]) == 0.0


def test_reranker_penalizes_middle_name_mismatch() -> None:
    reranker = ConflictAwareReranker()
    request = SotaScreenRequest(
        first_name="Laila",
        middle_name="Fadi",
        last_name="Shalaby",
        dob="1993-06-18",
        residency="Jordan",
        nationality="Jordan",
    )
    candidates = pd.DataFrame(
        [
            {
                "entity_id": "GOOD",
                "name": "Layla Fadi Shalabi",
                "first_name": "Layla",
                "middle_name": "Fadi",
                "last_name": "Shalabi",
                "aliases": "Laila Shalabi",
                "relative_names": "",
                "residency": "Jordan",
                "nationality": "Jordan",
                "gender": "",
                "dob": "1993-06-18",
            },
            {
                "entity_id": "NEAR",
                "name": "Laila Issa Shalaby",
                "first_name": "Laila",
                "middle_name": "Issa",
                "last_name": "Shalaby",
                "aliases": "Layla Shalabi",
                "relative_names": "",
                "residency": "Jordan",
                "nationality": "Jordan",
                "gender": "",
                "dob": "1993-06-18",
            },
        ]
    )
    scored = reranker.rerank(request, candidates)
    by_id = {str(r["entity_id"]): float(r["rerank_score"]) for _, r in scored.iterrows()}
    assert by_id["GOOD"] > by_id["NEAR"]
    near_row = scored[scored["entity_id"] == "NEAR"].iloc[0]
    assert "Middle name mismatch." in list(near_row["non_match_signals"])

