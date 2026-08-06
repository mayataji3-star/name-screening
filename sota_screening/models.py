from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, Field


class SotaScreenRequest(BaseModel):
    name: str = ""
    first_name: str = ""
    middle_name: str = ""
    last_name: str = ""
    dob: str = "UNKNOWN"
    residency: str = "UNKNOWN"
    nationality: str = "UNKNOWN"
    aliases: list[str] = Field(default_factory=list)
    relative_names: list[str] = Field(default_factory=list)
    gender: str = ""
    top_k: int = 5


class CandidateScore(BaseModel):
    entity_id: str
    name: str
    retrieval_score: float
    rerank_score: float
    policy_score: float
    final_score: float
    matched_aliases: list[str] = Field(default_factory=list)
    explanation: list[str] = Field(default_factory=list)
    non_match_signals: list[str] = Field(default_factory=list)
    llm_same_person_score: float | None = None
    llm_reason: str = ""
    rule_rerank_score: float | None = None


class SotaScreenResponse(BaseModel):
    decision: Literal["AUTO_CLEAR", "REVIEW", "AUTO_HIT"]
    top_score: float
    thresholds: dict[str, float]
    candidates: list[CandidateScore]


class FeedbackEvent(BaseModel):
    input_name: str
    entity_id: str
    analyst_label: Literal["MATCH", "NO_MATCH"]
    predicted_decision: str
    predicted_score: float

