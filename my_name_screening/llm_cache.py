"""Small JSON-file cache for successful Groq adjudications."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

from .llm_models import AdjudicationDecision


class JsonLLMCache:
    """Persist successful LLM decisions between program runs."""

    def __init__(
        self,
        path: str | Path = "artifacts/groq_cache.json",
    ) -> None:
        self.path = Path(path)

    @staticmethod
    def query_hash(query: dict[str, Any]) -> str:
        """Create a stable privacy-friendly hash for a query."""
        normalized_query = json.dumps(
            query,
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
            default=str,
        )
        return hashlib.sha256(
            normalized_query.encode("utf-8")
        ).hexdigest()

    @staticmethod
    def make_key(
        query: dict[str, Any],
        candidate_id: str,
        model: str,
        prompt_version: str,
    ) -> str:
        """Create a stable cache key for one comparison."""
        query_hash = JsonLLMCache.query_hash(query)

        return "|".join(
            (
                query_hash,
                candidate_id,
                model,
                prompt_version,
            )
        )

    def _read_all(self) -> dict[str, Any]:
        if not self.path.exists():
            return {}

        try:
            content = self.path.read_text(encoding="utf-8")
            parsed = json.loads(content)
        except (OSError, json.JSONDecodeError):
            return {}

        return parsed if isinstance(parsed, dict) else {}

    def get(self, key: str) -> AdjudicationDecision | None:
        """Return a validated cached decision when available."""
        cached = self._read_all().get(key)

        if cached is None:
            return None

        try:
            return AdjudicationDecision.model_validate(cached)
        except ValueError:
            return None

    def set(
        self,
        key: str,
        decision: AdjudicationDecision,
    ) -> None:
        """Save one validated successful decision atomically."""
        entries = self._read_all()
        entries[key] = decision.model_dump(mode="json")

        self.path.parent.mkdir(parents=True, exist_ok=True)
        temporary_path = self.path.with_suffix(
            self.path.suffix + ".tmp"
        )
        temporary_path.write_text(
            json.dumps(
                entries,
                ensure_ascii=False,
                indent=2,
                sort_keys=True,
            ),
            encoding="utf-8",
        )
        temporary_path.replace(self.path)