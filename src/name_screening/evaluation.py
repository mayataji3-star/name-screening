from __future__ import annotations

from pathlib import Path

import pandas as pd

from .models import ScreenRequest
from .screener import NameScreener


def run_evaluation(screener: NameScreener, eval_df: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for row in eval_df.itertuples(index=False):
        expected_alert = str(row.expected_alert).strip().upper() == "ALERT"
        response = screener.screen(
            ScreenRequest(
                name=row.name,
                dob=row.dob,
                nationality=row.nationality,
                top_k=int(row.top_k) if hasattr(row, "top_k") else 3,
            )
        )
        predicted_alert = response.decision == "ALERT"
        rows.append(
            {
                "case_id": row.case_id,
                "name": row.name,
                "expected_alert": expected_alert,
                "predicted_alert": predicted_alert,
                "top_score_pct": response.top_score_pct,
                "decision": response.decision,
            }
        )
    return pd.DataFrame(rows)


def summarize_metrics(results_df: pd.DataFrame) -> dict[str, float]:
    tp = int(((results_df["expected_alert"]) & (results_df["predicted_alert"])).sum())
    fp = int((~results_df["expected_alert"] & results_df["predicted_alert"]).sum())
    fn = int((results_df["expected_alert"] & ~results_df["predicted_alert"]).sum())
    tn = int((~results_df["expected_alert"] & ~results_df["predicted_alert"]).sum())

    precision = tp / (tp + fp) if (tp + fp) else 0.0
    recall = tp / (tp + fn) if (tp + fn) else 0.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) else 0.0

    return {
        "tp": float(tp),
        "fp": float(fp),
        "fn": float(fn),
        "tn": float(tn),
        "precision": precision,
        "recall": recall,
        "f1": f1,
    }


def write_evaluation_reports(
    artifacts_dir: Path, results_df: pd.DataFrame, metrics: dict[str, float]
) -> tuple[Path, Path]:
    artifacts_dir.mkdir(parents=True, exist_ok=True)
    csv_path = artifacts_dir / "evaluation_results.csv"
    md_path = artifacts_dir / "evaluation_report.md"
    results_df.to_csv(csv_path, index=False)

    md_lines = [
        "# Evaluation Report",
        "",
        "## Metrics",
        f"- Precision: {metrics['precision']:.3f}",
        f"- Recall: {metrics['recall']:.3f}",
        f"- F1: {metrics['f1']:.3f}",
        f"- TP: {int(metrics['tp'])}",
        f"- FP: {int(metrics['fp'])}",
        f"- FN: {int(metrics['fn'])}",
        f"- TN: {int(metrics['tn'])}",
        "",
        f"Detailed results: `{csv_path}`",
    ]
    md_path.write_text("\n".join(md_lines), encoding="utf-8")
    return csv_path, md_path
