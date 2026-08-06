from __future__ import annotations

from dataclasses import dataclass

import faiss
import pandas as pd

from .audit import append_audit_log
from .config import AppConfig
from .data_io import load_alias_map
from .embedding import EmbeddingService
from .formatting import format_passage_record, format_query
from .index_store import FaissIndexStore
from .models import MatchResult, ScreenRequest, ScreenResponse
from .normalization import equivalent_text, normalize_name, split_aliases

_RESIDENCY_EQUIVALENTS = {
    "jordan": "الأردن",
    "الأردن": "jordan",
    "yemen": "اليمن",
    "اليمن": "yemen",
    "egypt": "مصر",
    "مصر": "egypt",
    "syria": "سوريا",
    "سوريا": "syria",
    "iraq": "العراق",
    "العراق": "iraq",
}


@dataclass
class NameScreener:
    config: AppConfig
    embedding_service: EmbeddingService
    index_store: FaissIndexStore
    metadata_df: pd.DataFrame | None = None
    index: faiss.Index | None = None
    alias_map: dict[str, str] | None = None

    @classmethod
    def from_config(cls, config: AppConfig) -> "NameScreener":
        config.ensure_dirs()
        alias_map = load_alias_map(config.alias_map_path)
        return cls(
            config=config,
            embedding_service=EmbeddingService(config.model_name),
            index_store=FaissIndexStore(config.index_path, config.metadata_path),
            alias_map=alias_map,
        )

    def rebuild_index(self, watchlist_df: pd.DataFrame) -> None:
        watchlist_df = watchlist_df.reset_index(drop=True).copy()
        watchlist_df["passage_text"] = watchlist_df.apply(
            lambda row: format_passage_record(
                str(row.get("first_name", "")),
                str(row.get("middle_name", "")),
                str(row.get("last_name", "")),
                row["dob"],
                str(row.get("residency", row.get("nationality", ""))),
                aliases=str(row.get("aliases", "")),
                relative_names=str(row.get("relative_names", "")),
                gender=str(row.get("gender", "")),
            ),
            axis=1,
        )
        vectors = self.embedding_service.encode_texts(watchlist_df["passage_text"].tolist())
        index = faiss.IndexFlatIP(vectors.shape[1])
        index.add(vectors)
        self.index_store.save(index, watchlist_df)
        self.index = index
        self.metadata_df = watchlist_df

    def load_index(self) -> None:
        index, metadata_df = self.index_store.load()
        self.index = index
        self.metadata_df = metadata_df

    def ensure_index(self, watchlist_df: pd.DataFrame | None = None) -> None:
        if self.index is not None and self.metadata_df is not None:
            return
        if self.index_store.exists():
            self.load_index()
            return
        if watchlist_df is None:
            raise ValueError("Watchlist data required to build index.")
        self.rebuild_index(watchlist_df)

    def _resolve_request_name(self, request: ScreenRequest) -> ScreenRequest:
        if self.alias_map is None:
            return request
        candidate_aliases = [request.name] + request.aliases
        for alias in candidate_aliases:
            canonical = self.alias_map.get(alias)
            if canonical:
                request.aliases = list(dict.fromkeys([alias] + request.aliases))
                request.name = canonical
                parts = canonical.split()
                request.first_name = parts[0] if parts else ""
                request.middle_name = " ".join(parts[1:-1]) if len(parts) > 2 else ""
                request.last_name = parts[-1] if len(parts) > 1 else ""
                return request
        return request

    def _weighted_field_score(self, request: ScreenRequest, matched: pd.Series) -> float:
        total = (
            self.config.first_name_weight
            + self.config.middle_name_weight
            + self.config.last_name_weight
            + self.config.alias_weight
            + self.config.residency_weight
            + self.config.relatives_weight
            + self.config.gender_weight
        )
        if total <= 0:
            return 0.0

        def _equivalent(left: str, right: str) -> bool:
            if equivalent_text(left, right):
                return True
            left_norm = normalize_name(left)
            right_norm = normalize_name(right)
            return _RESIDENCY_EQUIVALENTS.get(left_norm) == right_norm

        def _any_equivalent(left_values: list[str], right_values: list[str]) -> bool:
            for left in left_values:
                for right in right_values:
                    if _equivalent(left, right):
                        return True
            return False

        score = 0.0
        if _equivalent(request.first_name, str(matched.get("first_name", ""))):
            score += self.config.first_name_weight
        if _equivalent(request.middle_name, str(matched.get("middle_name", ""))):
            score += self.config.middle_name_weight
        if _equivalent(request.last_name, str(matched.get("last_name", ""))):
            score += self.config.last_name_weight

        request_aliases = [a for a in request.aliases if a.strip()]
        matched_aliases = split_aliases(str(matched.get("aliases", "")))
        if _any_equivalent([request.name], matched_aliases) or _any_equivalent(
            request_aliases, matched_aliases
        ):
            score += self.config.alias_weight

        if _equivalent(request.residency, str(matched.get("residency", ""))):
            score += self.config.residency_weight

        request_relatives = [v for v in request.relative_names if v.strip()]
        matched_relatives = split_aliases(str(matched.get("relative_names", "")))
        if request_relatives and _any_equivalent(request_relatives, matched_relatives):
            score += self.config.relatives_weight

        if request.gender and _equivalent(request.gender, str(matched.get("gender", ""))):
            score += self.config.gender_weight

        return score / total

    def screen(self, request: ScreenRequest) -> ScreenResponse:
        if self.index is None or self.metadata_df is None:
            raise RuntimeError("Index is not loaded. Call ensure_index first.")

        request = self._resolve_request_name(request)
        query = format_query(
            request.first_name,
            request.middle_name,
            request.last_name,
            request.dob,
            request.residency or request.nationality,
            aliases="|".join(request.aliases),
            relative_names="|".join(request.relative_names),
            gender=request.gender,
        )
        query_vec = self.embedding_service.encode_texts([query])
        scores, indices = self.index.search(query_vec, request.top_k)
        score_row = scores[0]
        index_row = indices[0]

        rows: list[dict[str, object]] = []
        for score, idx in zip(score_row, index_row):
            if idx < 0:
                continue
            matched = self.metadata_df.iloc[int(idx)]
            embedding_score = max(0.0, min(1.0, float(score)))
            field_score = self._weighted_field_score(request, matched)
            final_score = max(
                0.0,
                min(
                    1.0,
                    (embedding_score * self.config.embedding_weight)
                    + (field_score * self.config.field_weight),
                ),
            )
            rows.append(
                {
                    "matched": matched,
                    "embedding_score": embedding_score,
                    "field_score": field_score,
                    "final_score": final_score,
                    "idx": int(idx),
                }
            )

        rows.sort(key=lambda row: float(row["final_score"]), reverse=True)
        matches: list[MatchResult] = []
        for rank, row in enumerate(rows, start=1):
            matched = row["matched"]
            idx = int(row["idx"])
            embedding_pct = float(row["embedding_score"]) * 100.0
            field_pct = float(row["field_score"]) * 100.0
            final_pct = float(row["final_score"]) * 100.0
            matches.append(
                MatchResult(
                    rank=rank,
                    entity_id=str(matched.get("entity_id", f"E{idx}")),
                    matched_name=str(matched["name"]),
                    matched_first_name=str(matched.get("first_name", "")),
                    matched_middle_name=str(matched.get("middle_name", "")),
                    matched_last_name=str(matched.get("last_name", "")),
                    matched_dob=str(matched["dob"]),
                    matched_residency=str(matched.get("residency", matched.get("nationality", ""))),
                    matched_gender=str(matched.get("gender", "")),
                    matched_aliases=split_aliases(str(matched.get("aliases", ""))),
                    embedding_similarity=float(row["embedding_score"]),
                    embedding_similarity_pct=f"{embedding_pct:.1f}%",
                    weighted_field_score=float(row["field_score"]),
                    weighted_field_score_pct=f"{field_pct:.1f}%",
                    final_score=float(row["final_score"]),
                    final_score_pct=f"{final_pct:.1f}%",
                )
            )

        top_score_pct = float(matches[0].final_score_pct.strip("%")) if matches else 0.0
        decision = (
            "ALERT"
            if top_score_pct >= self.config.alert_threshold_pct
            else "NO_ALERT"
        )
        reason = (
            f"Top score {top_score_pct:.1f}% compared to threshold "
            f"{self.config.alert_threshold_pct:.1f}%."
        )

        response = ScreenResponse(
            input_name=request.full_name,
            input_first_name=request.first_name,
            input_middle_name=request.middle_name,
            input_last_name=request.last_name,
            input_dob=request.dob,
            input_residency=request.residency or request.nationality,
            input_gender=request.gender,
            top_score_pct=top_score_pct,
            threshold_pct=self.config.alert_threshold_pct,
            decision=decision,
            reason=reason,
            matches=matches,
        )
        append_audit_log(self.config.audit_log_path, response.model_dump())
        return response

    def screen_batch(self, requests: list[ScreenRequest]) -> list[ScreenResponse]:
        return [self.screen(req) for req in requests]
