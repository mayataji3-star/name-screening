from __future__ import annotations

from dataclasses import dataclass

import pandas as pd


@dataclass(frozen=True)
class DecisionBandPolicy:
    auto_clear_threshold: float
    review_threshold: float
    auto_hit_threshold: float

    def decide(self, score: float) -> str:
        if score >= self.auto_hit_threshold:
            return "AUTO_HIT"
        if score >= self.review_threshold:
            return "REVIEW"
        if score <= self.auto_clear_threshold:
            return "AUTO_CLEAR"
        return "REVIEW"

    def apply(self, rows: pd.DataFrame) -> pd.DataFrame:
        data = rows.copy()
        data["decision_band"] = data["final_score"].map(self.decide)
        return data

