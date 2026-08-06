from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import pandas as pd

from .config import AppConfig
from .data_io import load_eval_cases, load_watchlist
from .evaluation import run_evaluation, summarize_metrics, write_evaluation_reports
from .models import ScreenRequest
from .neo4j_import import import_names_to_watchlist
from .screener import NameScreener


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Bilingual Name Screening MVP CLI")
    sub = parser.add_subparsers(dest="command", required=True)

    rebuild = sub.add_parser("rebuild-index", help="Rebuild FAISS index from watchlist file")
    rebuild.add_argument(
        "--max-records",
        type=int,
        default=None,
        help="Optional cap for OpenSanctions ingestion during rebuild",
    )

    single = sub.add_parser("screen", help="Screen a single entity")
    single.add_argument("--name", required=True)
    single.add_argument("--dob", required=True)
    single.add_argument("--nationality", required=True)
    single.add_argument("--top-k", type=int, default=3)

    batch = sub.add_parser("screen-batch", help="Screen entities from CSV")
    batch.add_argument("--input-csv", required=True)

    import_names = sub.add_parser(
        "import-neo4j-names",
        help="Import distinct Neo4j Client names into watchlist CSV",
    )
    import_names.add_argument(
        "--limit",
        type=int,
        default=75,
        help="Distinct client names to import (recommended 50-100).",
    )
    import_names.add_argument(
        "--env-file",
        default="../Oracle_Neo4j_AML_ETL/.env",
        help="Path to env file containing NEO4J_URI/NEO4J_USER/NEO4J_PASSWORD.",
    )

    sub.add_parser("evaluate", help="Run evaluation on eval_cases.csv")
    return parser


def main() -> None:
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")

    parser = _build_parser()
    args = parser.parse_args()
    config = AppConfig()
    config.ensure_dirs()
    screener = NameScreener.from_config(config)
    watchlist_df = load_watchlist(
        str(config.watchlist_path),
        target_size=config.watchlist_target_size,
        opensanctions_path=str(config.opensanctions_path),
        alias_map_path=str(config.alias_map_path),
        use_mock_data=config.use_mock_data,
        entity_scope=config.entity_scope,
        max_records=config.opensanctions_max_records,
    )

    if args.command == "rebuild-index":
        if args.max_records is not None:
            watchlist_df = load_watchlist(
                str(config.watchlist_path),
                target_size=config.watchlist_target_size,
                opensanctions_path=str(config.opensanctions_path),
                alias_map_path=str(config.alias_map_path),
                use_mock_data=config.use_mock_data,
                entity_scope=config.entity_scope,
                max_records=args.max_records,
            )
        screener.rebuild_index(watchlist_df)
        print(f"Index rebuilt. Records indexed: {len(watchlist_df)}")
        return

    if args.command == "import-neo4j-names":
        imported = import_names_to_watchlist(
            neo4j_env_file=args.env_file,
            output_watchlist_path=config.watchlist_path,
            limit=max(1, int(args.limit)),
        )
        print(f"Imported {imported} names into {config.watchlist_path}")
        return

    screener.ensure_index(watchlist_df)

    if args.command == "screen":
        response = screener.screen(
            ScreenRequest(
                name=args.name,
                dob=args.dob,
                nationality=args.nationality,
                top_k=args.top_k,
            )
        )
        print(response.model_dump_json(indent=2))
        return

    if args.command == "screen-batch":
        df = pd.read_csv(args.input_csv)
        outputs = []
        for row in df.itertuples(index=False):
            response = screener.screen(
                ScreenRequest(
                    name=row.name,
                    dob=row.dob,
                    nationality=row.nationality,
                    top_k=int(getattr(row, "top_k", config.top_k_default)),
                )
            )
            outputs.append(response.model_dump())
        print(json.dumps(outputs, ensure_ascii=False, indent=2))
        return

    if args.command == "evaluate":
        eval_df = load_eval_cases(str(config.eval_cases_path))
        results_df = run_evaluation(screener, eval_df)
        metrics = summarize_metrics(results_df)
        csv_path, md_path = write_evaluation_reports(
            Path(config.artifacts_dir), results_df, metrics
        )
        print(f"Evaluation complete: {csv_path}")
        print(f"Report: {md_path}")
        return


if __name__ == "__main__":
    main()
