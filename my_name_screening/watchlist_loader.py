from __future__ import annotations

import csv
from pathlib import Path

from .retrieval import WatchlistRecord


def split_pipe_values(value: str) -> tuple[str, ...]:
    """Split pipe-separated values and remove empty entries."""
    return tuple(
        item.strip()
        for item in value.split("|")
        if item.strip()
    )


def load_watchlist(
    file_path: str | Path,
) -> list[WatchlistRecord]:
    """Load searchable watchlist records from a CSV file."""
    path = Path(file_path)

    if not path.exists():
        raise FileNotFoundError(
            f"Watchlist file was not found: {path}"
        )

    records: list[WatchlistRecord] = []

    with path.open(
        mode="r",
        encoding="utf-8-sig",
        newline="",
    ) as csv_file:
        reader = csv.DictReader(csv_file)

        required_columns = {
            "entity_id",
            "name",
            "aliases",
        }

        available_columns = set(reader.fieldnames or [])
        missing_columns = required_columns - available_columns

        if missing_columns:
            raise ValueError(
                "Missing required columns: "
                + ", ".join(sorted(missing_columns))
            )

        for row in reader:
            record_id = row["entity_id"].strip()
            name = row["name"].strip()
            aliases = split_pipe_values(
                row.get("aliases", "")
            )

            if not record_id or not name:
                continue

            records.append(
                WatchlistRecord(
                    record_id=record_id,
                    name=name,
                    aliases=aliases,
                )
            )

    if not records:
        raise ValueError(
            "The watchlist contains no usable records."
        )

    return records