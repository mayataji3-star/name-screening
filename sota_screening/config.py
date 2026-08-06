from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class SotaConfig:
    model_name: str = "intfloat/multilingual-e5-base"
    watchlist_path: Path = Path("data/watchlist_neo4j.csv")
    artifacts_dir: Path = Path("sota_screening/data/artifacts")
    benchmark_path: Path = Path("sota_screening/data/benchmark_pairs.csv")
    calibration_path: Path = Path("sota_screening/data/calibration_thresholds.json")
    feedback_log_path: Path = Path("sota_screening/data/feedback_log.jsonl")
    retrieval_top_k: int = 50
    rerank_top_n: int = 10
    auto_clear_threshold: float = 0.35
    review_threshold: float = 0.70
    auto_hit_threshold: float = 0.90
    # Qwen2.5 LLM judge — lives in sota_screening/qwen/; enable via use_qwen_judge or env SOTA_USE_QWEN=1
    qwen_model_id: str = "Qwen/Qwen2.5-1.5B-Instruct"
    use_qwen_judge: bool = True
    qwen_max_new_tokens: int = 160
    llm_rule_blend: float = 0.25
    llm_min_rule_score: float = 0.35
    llm_max_rule_score: float = 0.75
    conflict_score_cap: float = 0.25

    def ensure_dirs(self) -> None:
        self.artifacts_dir.mkdir(parents=True, exist_ok=True)
        self.benchmark_path.parent.mkdir(parents=True, exist_ok=True)
        self.calibration_path.parent.mkdir(parents=True, exist_ok=True)
        self.feedback_log_path.parent.mkdir(parents=True, exist_ok=True)


def effective_use_qwen_judge(config: SotaConfig) -> bool:
    if config.use_qwen_judge:
        return True
    return os.getenv("SOTA_USE_QWEN", "").strip().lower() in {"1", "true", "yes"}
