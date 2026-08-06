from __future__ import annotations

import csv
from pathlib import Path

from dotenv import load_dotenv
from neo4j import GraphDatabase


def _split_name(value: str) -> tuple[str, str, str]:
    parts = [p for p in value.strip().split() if p]
    if not parts:
        return "", "", ""
    if len(parts) == 1:
        return parts[0], "", ""
    if len(parts) == 2:
        return parts[0], "", parts[1]
    return parts[0], " ".join(parts[1:-1]), parts[-1]


def import_names_to_watchlist(
    *,
    neo4j_env_file: str | Path,
    output_watchlist_path: str | Path,
    limit: int = 75,
) -> int:
    load_dotenv(neo4j_env_file)

    import os

    uri = os.getenv("NEO4J_URI", "")
    user = os.getenv("NEO4J_USER", "")
    password = os.getenv("NEO4J_PASSWORD", "")
    if not (uri and user and password):
        raise ValueError("Missing Neo4j credentials in provided env file.")

    driver = GraphDatabase.driver(uri, auth=(user, password))
    query = """
    MATCH (c:Client)
    WHERE c.name IS NOT NULL AND trim(toString(c.name)) <> ""
    RETURN DISTINCT toString(c.name) AS name
    ORDER BY name
    LIMIT $limit
    """
    rows: list[dict[str, str]] = []
    try:
        with driver.session() as session:
            names = [record["name"].strip() for record in session.run(query, limit=int(limit))]
        for i, name in enumerate(names, start=1):
            first, middle, last = _split_name(name)
            rows.append(
                {
                    "entity_id": f"N{i:04d}",
                    "name": name,
                    "first_name": first,
                    "middle_name": middle,
                    "last_name": last,
                    "dob": "UNKNOWN",
                    "residency": "UNKNOWN",
                    "nationality": "UNKNOWN",
                    "aliases": "",
                    "relative_names": "",
                    "gender": "",
                    "risk_level": "high",
                }
            )
    finally:
        driver.close()

    output_path = Path(output_watchlist_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "entity_id",
        "name",
        "first_name",
        "middle_name",
        "last_name",
        "dob",
        "residency",
        "nationality",
        "aliases",
        "relative_names",
        "gender",
        "risk_level",
    ]
    with output_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    return len(rows)
