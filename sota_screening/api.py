from __future__ import annotations

import pandas as pd
from fastapi import FastAPI

from .benchmark import run_benchmark
from .calibration import calibrate_thresholds
from .config import SotaConfig, effective_use_qwen_judge
from .feedback import append_feedback, summarize_feedback
from .models import FeedbackEvent, SotaScreenRequest, SotaScreenResponse
from .pipeline import SotaPipeline

config = SotaConfig()
pipeline = SotaPipeline.from_config(config)
watchlist_df = pd.read_csv(config.watchlist_path)
pipeline.build(watchlist_df)

app = FastAPI(title="SOTA Name Screening API")


@app.get("/")
def root() -> dict[str, object]:
    return {
        "service": "sota_screening",
        "docs": "/docs",
        "qwen_judge_enabled": effective_use_qwen_judge(config),
        "qwen_model": config.qwen_model_id,
    }


@app.get("/health")
def health() -> dict[str, str]:
    return {"status": "ok"}


@app.post("/index/rebuild")
def rebuild_index() -> dict[str, str]:
    global watchlist_df
    watchlist_df = pd.read_csv(config.watchlist_path)
    pipeline.build(watchlist_df)
    return {"status": "rebuilt", "records": str(len(watchlist_df))}


@app.post("/screen", response_model=SotaScreenResponse)
def screen(request: SotaScreenRequest) -> SotaScreenResponse:
    return pipeline.screen(request)


@app.post("/benchmark")
def benchmark() -> dict[str, float]:
    result = run_benchmark(config.benchmark_path, pipeline)
    return result.__dict__


@app.post("/calibration/recompute")
def recompute_calibration() -> dict[str, float]:
    return calibrate_thresholds(config, pipeline)


@app.post("/feedback")
def write_feedback(event: FeedbackEvent) -> dict[str, float]:
    append_feedback(config.feedback_log_path, event)
    return summarize_feedback(config.feedback_log_path)

