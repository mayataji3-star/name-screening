from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, Field, model_validator


def _split_full_name(value: str) -> tuple[str, str, str]:
    parts = [p for p in value.strip().split() if p]
    if not parts:
        return "", "", ""
    if len(parts) == 1:
        return parts[0], "", ""
    if len(parts) == 2:
        return parts[0], "", parts[1]
    return parts[0], " ".join(parts[1:-1]), parts[-1]


class StructuredNameMixin(BaseModel):
    first_name: str = ""
    middle_name: str = ""
    last_name: str = ""
    name: str = ""

    @property
    def full_name(self) -> str:
        parts = [self.first_name, self.middle_name, self.last_name]
        composed = " ".join(p.strip() for p in parts if p and p.strip())
        return composed or self.name.strip()

    @model_validator(mode="after")
    def _hydrate_name_fields(self):
        if (not self.first_name and not self.middle_name and not self.last_name) and self.name:
            first, middle, last = _split_full_name(self.name)
            self.first_name = first
            self.middle_name = middle
            self.last_name = last
        if not self.name:
            self.name = self.full_name
        return self


class WatchlistRecord(StructuredNameMixin):
    entity_id: str
    dob: str
    residency: str = ""
    nationality: str = ""
    aliases: list[str] = Field(default_factory=list)
    relative_names: list[str] = Field(default_factory=list)
    gender: str = ""
    risk_level: str = "high"


class ScreenRequest(StructuredNameMixin):
    dob: str
    residency: str = ""
    nationality: str = ""
    aliases: list[str] = Field(default_factory=list)
    relative_names: list[str] = Field(default_factory=list)
    gender: str = ""
    top_k: int = 3

    @model_validator(mode="after")
    def _fill_residency(self):
        if not self.residency and self.nationality:
            self.residency = self.nationality
        if not self.nationality and self.residency:
            self.nationality = self.residency
        return self


class MatchResult(BaseModel):
    rank: int
    entity_id: str
    matched_name: str
    matched_first_name: str = ""
    matched_middle_name: str = ""
    matched_last_name: str = ""
    matched_dob: str
    matched_residency: str = ""
    matched_gender: str = ""
    matched_aliases: list[str] = Field(default_factory=list)
    embedding_similarity: float
    embedding_similarity_pct: str
    weighted_field_score: float
    weighted_field_score_pct: str
    final_score: float
    final_score_pct: str


class ScreenResponse(BaseModel):
    input_name: str
    input_first_name: str = ""
    input_middle_name: str = ""
    input_last_name: str = ""
    input_dob: str
    input_residency: str = ""
    input_gender: str = ""
    top_score_pct: float
    threshold_pct: float
    decision: Literal["ALERT", "NO_ALERT"]
    reason: str
    matches: list[MatchResult]
