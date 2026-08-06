from __future__ import annotations

from dataclasses import dataclass

import pandas as pd

from .models import SotaScreenRequest
from .normalization import equivalent_text, normalize_text, split_aliases


@dataclass
class ConflictAwareReranker:
    first_name_weight: float = 0.22
    middle_name_weight: float = 0.14
    last_name_weight: float = 0.24
    alias_weight: float = 0.16
    residency_weight: float = 0.08
    nationality_weight: float = 0.08
    relatives_weight: float = 0.05
    gender_weight: float = 0.03

    @staticmethod
    def _split_name(name: str) -> tuple[str, str, str]:
        parts = [p for p in name.split() if p.strip()]
        if not parts:
            return "", "", ""
        if len(parts) == 1:
            return parts[0], "", ""
        if len(parts) == 2:
            return parts[0], "", parts[1]
        return parts[0], " ".join(parts[1:-1]), parts[-1]

    def _any_equivalent(
        self,
        left_values: list[str],
        right_values: list[str],
        *,
        threshold: float = 0.86,
    ) -> bool:
        for left in left_values:
            for right in right_values:
                if equivalent_text(left, right, fuzzy_threshold=threshold):
                    return True
        return False

    def _score_row(
        self, request: SotaScreenRequest, row: pd.Series
    ) -> tuple[float, list[str], list[str]]:
        reasons: list[str] = []
        non_match_signals: list[str] = []
        total = (
            self.first_name_weight
            + self.middle_name_weight
            + self.last_name_weight
            + self.alias_weight
            + self.residency_weight
            + self.nationality_weight
            + self.relatives_weight
            + self.gender_weight
        )
        score = 0.0

        query_first = request.first_name
        query_middle = request.middle_name
        query_last = request.last_name
        if not (query_first or query_middle or query_last):
            query_name = request.name or ""
            parts = [p for p in query_name.split() if p.strip()]
            if parts:
                query_first = parts[0]
            if len(parts) > 2:
                query_middle = " ".join(parts[1:-1])
            if len(parts) > 1:
                query_last = parts[-1]

        cand_first = str(row.get("first_name", ""))
        cand_middle = str(row.get("middle_name", ""))
        cand_last = str(row.get("last_name", ""))
        cand_name = str(row.get("name", ""))
        if not (cand_first or cand_middle or cand_last):
            cand_first, cand_middle, cand_last = self._split_name(cand_name)

        if query_first and not cand_first:
            non_match_signals.append("Candidate missing first name.")
        elif query_first and cand_first and equivalent_text(query_first, cand_first, fuzzy_threshold=0.84):
            score += self.first_name_weight
            reasons.append("First name equivalent.")
        elif query_first and cand_first:
            non_match_signals.append("First name mismatch.")
        if query_middle and not cand_middle:
            non_match_signals.append("Candidate missing middle name.")
        elif query_middle and cand_middle and equivalent_text(query_middle, cand_middle, fuzzy_threshold=0.85):
            score += self.middle_name_weight
            reasons.append("Middle name equivalent.")
        elif query_middle and cand_middle:
            non_match_signals.append("Middle name mismatch.")
        if query_last and not cand_last:
            non_match_signals.append("Candidate missing last name.")
        elif query_last and cand_last and equivalent_text(query_last, cand_last, fuzzy_threshold=0.84):
            score += self.last_name_weight
            reasons.append("Last name equivalent.")
        elif query_last and cand_last:
            non_match_signals.append("Last name mismatch.")

        # Fallback full-name hint to avoid losing score when split fields are absent.
        query_name = request.name or f"{query_first} {query_middle} {query_last}".strip()
        if query_name and equivalent_text(query_name, cand_name, fuzzy_threshold=0.83):
            reasons.append("Full name equivalent (transliteration/fuzzy).")

        req_aliases = [v for v in request.aliases if normalize_text(v)]
        cand_aliases = split_aliases(str(row.get("aliases", "")))
        if req_aliases and self._any_equivalent(req_aliases, cand_aliases, threshold=0.82):
            score += self.alias_weight
            reasons.append("Alias equivalence detected (transliteration/fuzzy).")
        elif req_aliases:
            non_match_signals.append("Alias mismatch or missing candidate aliases.")

        if request.residency and request.residency != "UNKNOWN" and equivalent_text(
            request.residency, str(row.get("residency", "")), fuzzy_threshold=0.88
        ):
            score += self.residency_weight
            reasons.append("Residency match.")
        elif request.residency and request.residency != "UNKNOWN":
            non_match_signals.append("Residency mismatch or missing.")

        if request.nationality and request.nationality != "UNKNOWN" and equivalent_text(
            request.nationality, str(row.get("nationality", "")), fuzzy_threshold=0.88
        ):
            score += self.nationality_weight
            reasons.append("Nationality match.")
        elif request.nationality and request.nationality != "UNKNOWN":
            non_match_signals.append("Nationality mismatch or missing.")

        req_relatives = [v for v in request.relative_names if normalize_text(v)]
        cand_relatives = split_aliases(str(row.get("relative_names", "")))
        if req_relatives and self._any_equivalent(req_relatives, cand_relatives, threshold=0.84):
            score += self.relatives_weight
            reasons.append("Relative-name equivalence detected.")
        elif req_relatives:
            non_match_signals.append("Relative names mismatch or missing.")

        if request.gender and equivalent_text(request.gender, str(row.get("gender", "")), fuzzy_threshold=0.95):
            score += self.gender_weight
            reasons.append("Gender match.")
        elif request.gender:
            non_match_signals.append("Gender mismatch or missing.")

        if request.dob and request.dob != "UNKNOWN":
            candidate_dob = str(row.get("dob", "UNKNOWN"))
            if candidate_dob not in {"", "UNKNOWN"} and candidate_dob != request.dob:
                reasons.append("DOB conflict found.")
                non_match_signals.append("DOB mismatch.")
                return 0.0, reasons, non_match_signals
            if candidate_dob == request.dob:
                reasons.append("DOB exact match.")
            if candidate_dob in {"", "UNKNOWN"}:
                non_match_signals.append("Candidate DOB missing.")

        return score / total if total else 0.0, reasons, non_match_signals

    def rerank(self, request: SotaScreenRequest, candidates: pd.DataFrame) -> pd.DataFrame:
        scored = candidates.copy()
        rerank_scores: list[float] = []
        explanations: list[list[str]] = []
        non_match_lists: list[list[str]] = []
        has_dob_conflict: list[bool] = []
        has_geo_conflict: list[bool] = []
        for _, row in scored.iterrows():
            score, reasons, non_match_signals = self._score_row(request, row)
            rerank_scores.append(score)
            explanations.append(reasons)
            non_match_lists.append(non_match_signals)
            has_dob_conflict.append(any("DOB conflict found." in r for r in reasons))
            req_res = normalize_text(request.residency)
            req_nat = normalize_text(request.nationality)
            cand_res = normalize_text(str(row.get("residency", "")))
            cand_nat = normalize_text(str(row.get("nationality", "")))
            geo_conflict = (
                (req_res and cand_res and req_res != cand_res)
                and (req_nat and cand_nat and req_nat != cand_nat)
            )
            has_geo_conflict.append(geo_conflict)
        scored["rerank_score"] = rerank_scores
        scored["explanation"] = explanations
        scored["non_match_signals"] = non_match_lists
        scored["has_dob_conflict"] = has_dob_conflict
        scored["has_geo_conflict"] = has_geo_conflict
        return scored.sort_values("rerank_score", ascending=False).reset_index(drop=True)

