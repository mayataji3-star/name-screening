from __future__ import annotations

import argparse
import json

import pandas as pd

from .benchmark import run_benchmark
from .calibration import calibrate_thresholds
from .config import SotaConfig
from .feedback import summarize_feedback
from .models import SotaScreenRequest
from .pipeline import SotaPipeline


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="SOTA Name Screening CLI")
    sub = parser.add_subparsers(dest="command", required=True)

    screen = sub.add_parser("screen")
    screen.add_argument("--name", required=True)
    screen.add_argument("--dob", default="UNKNOWN")
    screen.add_argument("--residency", default="UNKNOWN")
    screen.add_argument("--gender", default="")
    screen.add_argument("--top-k", type=int, default=5)

    sub.add_parser("rebuild-index")
    sub.add_parser("benchmark")
    sub.add_parser("calibrate")
    sub.add_parser("feedback-summary")
    return parser


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()
    config = SotaConfig()
    config.ensure_dirs()
    watchlist = pd.read_csv(config.watchlist_path)
    pipeline = SotaPipeline.from_config(config)
    pipeline.build(watchlist)

    if args.command == "rebuild-index":
        pipeline.build(watchlist)
        print(f"Index rebuilt. Records indexed: {len(watchlist)}")
        return

    if args.command == "screen":
        response = pipeline.screen(
            SotaScreenRequest(
                name=args.name,
                dob=args.dob,
                residency=args.residency,
                gender=args.gender,
                top_k=args.top_k,
            )
        )
        print(response.model_dump_json(indent=2))
        return

    if args.command == "benchmark":
        metrics = run_benchmark(config.benchmark_path, pipeline)
        print(json.dumps(metrics.__dict__, indent=2))
        return

    if args.command == "calibrate":
        updated = calibrate_thresholds(config, pipeline)
        print(json.dumps(updated, indent=2))
        return

    if args.command == "feedback-summary":
        print(json.dumps(summarize_feedback(config.feedback_log_path), indent=2))
        return


if __name__ == "__main__":
    main()

