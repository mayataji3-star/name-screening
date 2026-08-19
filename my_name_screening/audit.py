"""Append-only JSONL audit logging for LLM adjudication."""

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from .llm_models import AdjudicationResult


class JsonlAuditLogger:
    """Write one safe audit record for each adjudication."""

    def __init__(
        self,
        path: str | Path = "artifacts/groq_audit.jsonl",
    ) -> None:
        self.path = Path(path)

    def log(
        self,
        *,
        query_hash: str,
        candidate_id: str,
        result: AdjudicationResult,
    ) -> None:
        """Append an adjudication event without secrets or raw query data."""
        record: dict[str, Any] = {
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "query_hash": query_hash,
            "candidate_id": candidate_id,
            "model": result.model,
            "prompt_version": result.prompt_version,
            "same_person_score": result.decision.same_person_score,
            "verdict": result.decision.verdict.value,
            "reason": result.decision.reason,
            "signals": result.decision.signals.model_dump(),
            "input_tokens": result.input_tokens,
            "output_tokens": result.output_tokens,
            "total_tokens": result.total_tokens,
            "latency_ms": result.latency_ms,
            "attempts": result.attempts,
            "cache_hit": result.cache_hit,
            "llm_available": result.llm_available,
            "fallback_used": result.fallback_used,
            "error": result.error,
        }

        self.path.parent.mkdir(parents=True, exist_ok=True)

        with self.path.open("a", encoding="utf-8") as audit_file:
            audit_file.write(
                json.dumps(record, ensure_ascii=False) + "\n"
            )