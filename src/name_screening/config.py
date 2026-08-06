from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class AppConfig:
    model_name: str = "intfloat/multilingual-e5-base"
    top_k_default: int = 3
    alert_threshold_pct: float = 90.0
    watchlist_target_size: int = 200
    entity_scope: str = "people_only"
    opensanctions_max_records: int = 0
    use_mock_data: bool = True
    embedding_weight: float = 0.5
    field_weight: float = 0.5
    first_name_weight: float = 0.25
    middle_name_weight: float = 0.10
    last_name_weight: float = 0.25
    alias_weight: float = 0.15
    residency_weight: float = 0.10
    relatives_weight: float = 0.10
    gender_weight: float = 0.05
    data_dir: Path = Path("data")
    artifacts_dir: Path = Path("artifacts")
    watchlist_path: Path = Path("data/watchlist_neo4j.csv")
    alias_map_path: Path = Path("data/alias_map_mock.csv")
    opensanctions_path: Path = Path("")
    eval_cases_path: Path = Path("data/eval_cases.csv")
    index_path: Path = Path("artifacts/watchlist.faiss")
    metadata_path: Path = Path("artifacts/watchlist_metadata.csv")
    audit_log_path: Path = Path("artifacts/audit_log.jsonl")

    def ensure_dirs(self) -> None:
        self.data_dir.mkdir(parents=True, exist_ok=True)
        self.artifacts_dir.mkdir(parents=True, exist_ok=True)
