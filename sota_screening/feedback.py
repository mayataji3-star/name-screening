from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path

from .models import FeedbackEvent


def append_feedback(path: Path, event: FeedbackEvent) -> None:
    payload = {"timestamp": datetime.now(timezone.utc).isoformat(), **event.model_dump()}
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(payload, ensure_ascii=False) + "\n")


def summarize_feedback(path: Path) -> dict[str, float]:
    if not path.exists():
        return {"events": 0.0, "match_rate": 0.0}
    lines = [line for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
    if not lines:
        return {"events": 0.0, "match_rate": 0.0}
    events = [json.loads(line) for line in lines]
    match_count = sum(1 for event in events if event.get("analyst_label") == "MATCH")
    return {"events": float(len(events)), "match_rate": float(match_count / len(events))}

