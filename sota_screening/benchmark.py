from __future__ import annotations

import csv
from dataclasses import dataclass
from pathlib import Path

from .models import SotaScreenRequest
from .pipeline import SotaPipeline


@dataclass(frozen=True)
class BenchmarkMetrics:
    precision: float
    recall: float
    f1: float
    review_rate: float


def ensure_benchmark_template(path: Path) -> None:
    if path.exists():
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=[
                "name",
                "dob",
                "residency",
                "gender",
                "expected_entity_id",
                "expected_match",
                "language_segment",
            ],
        )
        writer.writeheader()


def run_benchmark(path: Path, pipeline: SotaPipeline) -> BenchmarkMetrics:
    ensure_benchmark_template(path)
    rows = list(csv.DictReader(path.open("r", encoding="utf-8")))
    if not rows:
        return BenchmarkMetrics(0.0, 0.0, 0.0, 0.0)
    tp = fp = fn = review = 0
    for row in rows:
        response = pipeline.screen(
            SotaScreenRequest(
                name=row.get("name", ""),
                dob=row.get("dob", "UNKNOWN"),
                residency=row.get("residency", "UNKNOWN"),
                gender=row.get("gender", ""),
                top_k=1,
            )
        )
        predicted_match = bool(response.candidates and response.candidates[0].entity_id == row.get("expected_entity_id", ""))
        expected_match = str(row.get("expected_match", "")).strip().lower() == "true"
        if response.decision == "REVIEW":
            review += 1
        if predicted_match and expected_match:
            tp += 1
        elif predicted_match and not expected_match:
            fp += 1
        elif not predicted_match and expected_match:
            fn += 1
    precision = tp / (tp + fp) if (tp + fp) else 0.0
    recall = tp / (tp + fn) if (tp + fn) else 0.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) else 0.0
    return BenchmarkMetrics(precision, recall, f1, review / len(rows))

