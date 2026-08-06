from __future__ import annotations

import json
from pathlib import Path

from .benchmark import run_benchmark
from .config import SotaConfig
from .pipeline import SotaPipeline


def calibrate_thresholds(config: SotaConfig, pipeline: SotaPipeline) -> dict[str, float]:
    metrics = run_benchmark(config.benchmark_path, pipeline)
    review_threshold = 0.7
    if metrics.precision < 0.85:
        review_threshold = 0.75
    if metrics.recall < 0.80:
        review_threshold = 0.65
    thresholds = {
        "auto_clear": 0.35,
        "review": review_threshold,
        "auto_hit": max(0.90, review_threshold + 0.15),
        "metrics_f1": metrics.f1,
        "metrics_review_rate": metrics.review_rate,
    }
    config.calibration_path.write_text(json.dumps(thresholds, indent=2), encoding="utf-8")
    return thresholds


def load_thresholds(path: Path) -> dict[str, float]:
    if not path.exists():
        return {}
    return json.loads(path.read_text(encoding="utf-8"))

