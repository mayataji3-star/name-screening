from __future__ import annotations

from dataclasses import dataclass

import pandas as pd

from name_screening.embedding import EmbeddingService

from .candidate_generation import CandidateGenerator
from .config import SotaConfig, effective_use_qwen_judge
from .models import CandidateScore, SotaScreenRequest, SotaScreenResponse
from .normalization import split_aliases
from .policy import DecisionBandPolicy
from .qwen.judge import QwenPairwiseJudge, build_query_payload_from_request
from .reranker import ConflictAwareReranker


@dataclass
class SotaPipeline:
    config: SotaConfig
    generator: CandidateGenerator
    reranker: ConflictAwareReranker
    policy: DecisionBandPolicy
    qwen_judge: QwenPairwiseJudge | None = None

    @classmethod
    def from_config(cls, config: SotaConfig) -> "SotaPipeline":
        config.ensure_dirs()
        emb = EmbeddingService(config.model_name)
        qwen: QwenPairwiseJudge | None = None
        if effective_use_qwen_judge(config):
            qwen = QwenPairwiseJudge(
                model_id=config.qwen_model_id,
                max_new_tokens=config.qwen_max_new_tokens,
            )
        return cls(
            config=config,
            generator=CandidateGenerator(emb),
            reranker=ConflictAwareReranker(),
            policy=DecisionBandPolicy(
                auto_clear_threshold=config.auto_clear_threshold,
                review_threshold=config.review_threshold,
                auto_hit_threshold=config.auto_hit_threshold,
            ),
            qwen_judge=qwen,
        )

    def build(self, watchlist: pd.DataFrame) -> None:
        self.generator.build(watchlist)

    def _blend_scores(self, rows: pd.DataFrame) -> pd.DataFrame:
        data = rows.copy()
        conflict_cap = self.config.conflict_score_cap
        if "has_dob_conflict" in data.columns:
            data.loc[data["has_dob_conflict"] == True, "rerank_score"] = data.loc[
                data["has_dob_conflict"] == True, "rerank_score"
            ].clip(upper=conflict_cap)
        if "has_geo_conflict" in data.columns:
            data.loc[data["has_geo_conflict"] == True, "rerank_score"] = data.loc[
                data["has_geo_conflict"] == True, "rerank_score"
            ].clip(upper=conflict_cap)
        data["policy_score"] = data["rerank_score"]
        data["final_score"] = (data["retrieval_score"] * 0.4) + (data["rerank_score"] * 0.6)
        return data.sort_values("final_score", ascending=False).reset_index(drop=True)

    def screen(self, request: SotaScreenRequest) -> SotaScreenResponse:
        query_name = request.name or f"{request.first_name} {request.middle_name} {request.last_name}"
        candidates = self.generator.retrieve(query_name, self.config.retrieval_top_k)
        if candidates.empty:
            return SotaScreenResponse(
                decision="AUTO_CLEAR",
                top_score=0.0,
                thresholds={
                    "auto_clear": self.config.auto_clear_threshold,
                    "review": self.config.review_threshold,
                    "auto_hit": self.config.auto_hit_threshold,
                },
                candidates=[],
            )
        reranked = self.reranker.rerank(request, candidates)
        reranked = self._apply_qwen_judge(request, reranked)
        reranked = reranked.head(max(request.top_k, self.config.rerank_top_n))
        blended = self._blend_scores(reranked)
        with_bands = self.policy.apply(blended)
        top_score = float(with_bands.iloc[0]["final_score"])
        decision = str(self.policy.decide(top_score))

        result_rows: list[CandidateScore] = []
        for _, row in with_bands.head(request.top_k).iterrows():
            result_rows.append(
                CandidateScore(
                    entity_id=str(row.get("entity_id", "")),
                    name=str(row.get("name", "")),
                    retrieval_score=float(row.get("retrieval_score", 0.0)),
                    rerank_score=float(row.get("rerank_score", 0.0)),
                    policy_score=float(row.get("policy_score", 0.0)),
                    final_score=float(row.get("final_score", 0.0)),
                    matched_aliases=split_aliases(str(row.get("aliases", ""))),
                    explanation=list(row.get("explanation", [])),
                    non_match_signals=list(row.get("non_match_signals", [])),
                    llm_same_person_score=row.get("llm_same_person_score"),  # type: ignore[arg-type]
                    llm_reason=str(row.get("llm_reason", "") or ""),
                    rule_rerank_score=row.get("rule_rerank_score"),  # type: ignore[arg-type]
                )
            )

        return SotaScreenResponse(
            decision=decision,  # type: ignore[arg-type]
            top_score=top_score,
            thresholds={
                "auto_clear": self.config.auto_clear_threshold,
                "review": self.config.review_threshold,
                "auto_hit": self.config.auto_hit_threshold,
            },
            candidates=result_rows,
        )

    def _apply_qwen_judge(self, request: SotaScreenRequest, scored: pd.DataFrame) -> pd.DataFrame:
        if self.qwen_judge is None or scored.empty:
            return scored
        n = min(self.config.rerank_top_n, len(scored))
        if n <= 0:
            return scored
        query_payload = build_query_payload_from_request(request)
        blend = self.config.llm_rule_blend
        min_rule = self.config.llm_min_rule_score
        max_rule = self.config.llm_max_rule_score
        df = scored.reset_index(drop=True).copy()
        df["rule_rerank_score"] = df["rerank_score"]
        df["llm_same_person_score"] = None
        df["llm_reason"] = ""
        for i in range(n):
            row = df.iloc[i]
            rule_s = float(row.get("rerank_score", 0.0))
            # LLM is only advisory for borderline records.
            if rule_s < min_rule or rule_s > max_rule:
                df.at[df.index[i], "llm_reason"] = "llm_skipped_outside_eligibility_band"
                continue
            if bool(row.get("has_dob_conflict", False)) or bool(row.get("has_geo_conflict", False)):
                df.at[df.index[i], "llm_reason"] = "llm_skipped_hard_conflict"
                continue
            try:
                llm_s, reason = self.qwen_judge.score_pair(query_payload, row)
            except Exception as exc:  # noqa: BLE001
                df.at[df.index[i], "llm_reason"] = f"llm_error:{exc}"
                continue
            combined = blend * llm_s + (1.0 - blend) * rule_s
            df.at[df.index[i], "llm_same_person_score"] = llm_s
            df.at[df.index[i], "llm_reason"] = reason
            df.at[df.index[i], "rerank_score"] = combined
        return df.sort_values("rerank_score", ascending=False).reset_index(drop=True)

