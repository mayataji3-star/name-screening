"""Validated data models for LLM adjudication."""

from __future__ import annotations

from enum import Enum

from pydantic import BaseModel, ConfigDict, Field


class Verdict(str, Enum):
    """Allowed LLM adjudication verdicts."""

    MATCH = "MATCH"
    POSSIBLE = "POSSIBLE"
    NO_MATCH = "NO_MATCH"


class AdjudicationSignals(BaseModel):
    """Short explanations for the major evidence groups."""

    model_config = ConfigDict(extra="forbid")

    name: str
    dob: str
    geo: str


class AdjudicationDecision(BaseModel):
    """Strict response contract required from Groq."""

    model_config = ConfigDict(extra="forbid")

    same_person_score: float = Field(ge=0.0, le=1.0)
    verdict: Verdict
    reason: str = Field(min_length=1, max_length=500)
    signals: AdjudicationSignals


class AdjudicationResult(BaseModel):
    """Decision plus operational information about the Groq call."""

    model_config = ConfigDict(extra="forbid")

    decision: AdjudicationDecision
    llm_available: bool
    fallback_used: bool
    attempts: int = Field(ge=0)
    error: str | None = None
    cache_hit: bool = False
    model: str
    prompt_version: str
    input_tokens: int = Field(default=0, ge=0)
    output_tokens: int = Field(default=0, ge=0)
    total_tokens: int = Field(default=0, ge=0)
    latency_ms: float = Field(default=0.0, ge=0.0)