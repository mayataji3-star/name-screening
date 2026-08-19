"""Pairwise LLM adjudication through the Groq API."""

from __future__ import annotations

import json
import os
import time
from collections.abc import Callable
from typing import Any

from dotenv import load_dotenv
from groq import (
    APIConnectionError,
    APIStatusError,
    APITimeoutError,
    Groq,
    RateLimitError,
)
from pydantic import ValidationError

from .llm_models import (
    AdjudicationDecision,
    AdjudicationResult,
    AdjudicationSignals,
    Verdict,
)
from .audit import JsonlAuditLogger
from .llm_cache import JsonLLMCache
from .prompts import PROMPT_VERSION, SYSTEM_PROMPT


DEFAULT_GROQ_MODEL = "openai/gpt-oss-20b"


class GroqJudge:
    """Ask Groq whether one query and candidate are the same person."""

    def __init__(
        self,
        *,
        model: str | None = None,
        timeout_seconds: float = 15.0,
        max_rate_limit_attempts: int = 3,
        initial_backoff_seconds: float = 1.0,
        client: Any | None = None,
        sleep: Callable[[float], None] = time.sleep,
        cache: JsonLLMCache | None = None,
        use_cache: bool = True,
        audit_logger: JsonlAuditLogger | None = None,
        enable_audit: bool = True,
    ) -> None:
        load_dotenv()

        self.model = model or os.getenv(
            "GROQ_MODEL",
            DEFAULT_GROQ_MODEL,
        )
        self.max_rate_limit_attempts = max_rate_limit_attempts
        self.initial_backoff_seconds = initial_backoff_seconds
        self._sleep = sleep
        self._cache = (
            cache or JsonLLMCache()
            if use_cache
            else None
        )
        self._audit_logger = (
            audit_logger or JsonlAuditLogger()
            if enable_audit
            else None
        )

        if max_rate_limit_attempts < 1:
            raise ValueError("max_rate_limit_attempts must be at least 1.")

        if initial_backoff_seconds < 0:
            raise ValueError("initial_backoff_seconds must not be negative.")

        if client is not None:
            self._client = client
            return

        api_key = os.getenv("GROQ_API_KEY")

        if not api_key:
            raise RuntimeError(
                "GROQ_API_KEY is missing. Add it to the private .env file."
            )

        self._client = Groq(
            api_key=api_key,
            timeout=timeout_seconds,
            max_retries=0,
        )

    @staticmethod
    def _build_user_message(
        query: dict[str, Any],
        candidate: dict[str, Any],
    ) -> str:
        """Build a grounded pairwise comparison request."""
        payload = {
            "QUERY": query,
            "WATCHLIST_CANDIDATE": candidate,
        }

        return json.dumps(
            payload,
            ensure_ascii=False,
            indent=2,
            default=str,
        )

    @staticmethod
    def _response_format() -> dict[str, Any]:
        """Return Groq's strict JSON Schema configuration."""
        return {
            "type": "json_schema",
            "json_schema": {
                "name": "name_adjudication",
                "strict": True,
                "schema": AdjudicationDecision.model_json_schema(),
            },
        }

    def _call_groq(
        self,
        query: dict[str, Any],
        candidate: dict[str, Any],
    ) -> tuple[AdjudicationDecision, int, int, int, float]:
        """Make one Groq request and validate the returned JSON."""
        started_at = time.perf_counter()
        response = self._client.chat.completions.create(
            model=self.model,
            messages=[
                {
                    "role": "system",
                    "content": SYSTEM_PROMPT,
                },
                {
                    "role": "user",
                    "content": self._build_user_message(
                        query,
                        candidate,
                    ),
                },
            ],
            temperature=0,
            response_format=self._response_format(),
        )

        content = response.choices[0].message.content
        latency_ms = (time.perf_counter() - started_at) * 1000

        if not content:
            raise ValueError("Groq returned an empty response.")

        decision = AdjudicationDecision.model_validate_json(content)
        usage = getattr(response, "usage", None)
        input_tokens = int(getattr(usage, "prompt_tokens", 0) or 0)
        output_tokens = int(
            getattr(usage, "completion_tokens", 0) or 0
        )
        total_tokens = int(
            getattr(usage, "total_tokens", 0)
            or input_tokens + output_tokens
        )

        return (
            decision,
            input_tokens,
            output_tokens,
            total_tokens,
            latency_ms,
        )

    def _record_audit(
        self,
        query: dict[str, Any],
        candidate_id: str,
        result: AdjudicationResult,
    ) -> AdjudicationResult:
        """Write the audit event and return the unchanged result."""
        if self._audit_logger is not None:
            self._audit_logger.log(
                query_hash=JsonLLMCache.query_hash(query),
                candidate_id=candidate_id,
                result=result,
            )

        return result

    @staticmethod
    def _fallback_decision(
        deterministic_score: float,
        deterministic_verdict: Verdict,
    ) -> AdjudicationDecision:
        """Build a clearly labelled deterministic fallback decision."""
        return AdjudicationDecision(
            same_person_score=deterministic_score,
            verdict=deterministic_verdict,
            reason=(
                "Groq was unavailable; the deterministic screening "
                "result was used."
            ),
            signals=AdjudicationSignals(
                name="Deterministic name evidence used",
                dob="Not adjudicated by the LLM",
                geo="Not adjudicated by the LLM",
            ),
        )

    def judge_pair(
        self,
        query: dict[str, Any],
        candidate: dict[str, Any],
        *,
        deterministic_score: float,
        deterministic_verdict: Verdict,
    ) -> AdjudicationResult:
        """Judge a pair, retry safely, and fall back on failure."""
        candidate_id = str(
            candidate.get("record_id")
            or candidate.get("entity_id")
            or ""
        )
        cache_key: str | None = None

        if self._cache is not None and candidate_id:
            cache_key = self._cache.make_key(
                query,
                candidate_id,
                self.model,
                PROMPT_VERSION,
            )
            cached_decision = self._cache.get(cache_key)

            if cached_decision is not None:
                result = AdjudicationResult(
                    decision=cached_decision,
                    llm_available=True,
                    fallback_used=False,
                    attempts=0,
                    cache_hit=True,
                    model=self.model,
                    prompt_version=PROMPT_VERSION,
                )
                return self._record_audit(query, candidate_id, result)

        attempts = 0
        malformed_responses = 0
        last_error: str | None = None

        while attempts < self.max_rate_limit_attempts:
            attempts += 1

            try:
                (
                    decision,
                    input_tokens,
                    output_tokens,
                    total_tokens,
                    latency_ms,
                ) = self._call_groq(query, candidate)

                if self._cache is not None and cache_key:
                    self._cache.set(cache_key, decision)

                result = AdjudicationResult(
                    decision=decision,
                    llm_available=True,
                    fallback_used=False,
                    attempts=attempts,
                    cache_hit=False,
                    model=self.model,
                    prompt_version=PROMPT_VERSION,
                    input_tokens=input_tokens,
                    output_tokens=output_tokens,
                    total_tokens=total_tokens,
                    latency_ms=latency_ms,
                )
                return self._record_audit(query, candidate_id, result)
            except (ValidationError, ValueError) as error:
                malformed_responses += 1
                last_error = type(error).__name__

                # One retry means at most two malformed responses.
                if malformed_responses >= 2:
                    break
            except RateLimitError as error:
                last_error = type(error).__name__

                if attempts >= self.max_rate_limit_attempts:
                    break

                delay = self.initial_backoff_seconds * (
                    2 ** (attempts - 1)
                )
                self._sleep(delay)
            except (
                APITimeoutError,
                APIConnectionError,
                APIStatusError,
            ) as error:
                last_error = type(error).__name__
                break

        fallback = self._fallback_decision(
            deterministic_score,
            deterministic_verdict,
        )

        result = AdjudicationResult(
            decision=fallback,
            llm_available=False,
            fallback_used=True,
            attempts=attempts,
            error=last_error or "GroqUnavailable",
            cache_hit=False,
            model=self.model,
            prompt_version=PROMPT_VERSION,
        )
        return self._record_audit(query, candidate_id, result)