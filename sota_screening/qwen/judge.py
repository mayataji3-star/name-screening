"""
Pairwise same-person scoring using Qwen2.5-1.5B-Instruct (local HF weights).

Conflict-first framing: treat name/script variation as normal; look for contradictory
identifiers or impossible attribute combinations. Output is JSON with same_person_score 0..1.
"""

from __future__ import annotations

import json
import logging
import re
from dataclasses import dataclass, field
from typing import Any

import pandas as pd

logger = logging.getLogger(__name__)

SYSTEM_PROMPT = """You are an expert entity-resolution assistant for AML name screening.
Decide whether QUERY and WATCHLIST_CANDIDATE refer to the same real-world person.

Rules:
- Name variation by transliteration (Arabic/Latin), spelling, or token order is common.
- Missing fields are normal and are not evidence of mismatch.
- Use contradictions to reduce score (DOB conflict, incompatible attributes).
- Do NOT claim "exact match" unless values truly match.

Output format:
Return exactly one JSON object with keys:
- same_person_score: float between 0.0 and 1.0
- brief_reason: short sentence grounded in compared fields
No markdown and no extra text.
"""


@dataclass
class QwenPairwiseJudge:
    model_id: str = "Qwen/Qwen2.5-1.5B-Instruct"
    max_new_tokens: int = 160
    _tokenizer: Any = field(default=None, repr=False)
    _model: Any = field(default=None, repr=False)

    def _ensure_loaded(self) -> None:
        if self._model is not None and self._tokenizer is not None:
            return
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer

        logger.info("Loading Qwen judge: %s", self.model_id)
        self._tokenizer = AutoTokenizer.from_pretrained(self.model_id, trust_remote_code=True)
        self._model = AutoModelForCausalLM.from_pretrained(
            self.model_id,
            torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
            device_map="auto" if torch.cuda.is_available() else None,
            trust_remote_code=True,
        )
        if not torch.cuda.is_available():
            self._model = self._model.to(torch.device("cpu"))
        if self._tokenizer.pad_token_id is None:
            self._tokenizer.pad_token = self._tokenizer.eos_token

    def _format_entity_block(self, label: str, payload: dict[str, Any]) -> str:
        def _is_effectively_empty(value: Any) -> bool:
            if value is None:
                return True
            if isinstance(value, str):
                return value.strip() == ""
            if isinstance(value, (list, tuple, set, dict)):
                return len(value) == 0
            try:
                import numpy as np  # local import to avoid hard dependency in module import path

                if isinstance(value, np.ndarray):
                    return value.size == 0
            except Exception:  # noqa: BLE001
                pass
            return False

        filtered: dict[str, Any] = {}
        for k, v in payload.items():
            if _is_effectively_empty(v):
                continue
            if isinstance(v, list):
                v = [str(x) for x in v]
            filtered[k] = v
        # JSON-serialized blocks are easier for the model to compare reliably.
        return f"{label}:\n{json.dumps(filtered, ensure_ascii=False, indent=2)}"

    def _build_user_prompt(self, query: dict[str, Any], candidate: dict[str, Any]) -> str:
        return (
            self._format_entity_block("QUERY", query)
            + "\n\n"
            + self._format_entity_block("WATCHLIST_CANDIDATE", candidate)
            + "\n\nOutput the JSON object now."
        )

    def score_pair(
        self,
        query_payload: dict[str, Any],
        candidate_row: pd.Series,
    ) -> tuple[float, str]:
        cand: dict[str, Any] = {
            "name": str(candidate_row.get("name", "")),
            "dob": str(candidate_row.get("dob", "")),
            "residency": str(candidate_row.get("residency", "")),
            "nationality": str(candidate_row.get("nationality", "")),
            "aliases": candidate_row.get("aliases", ""),
            "relative_names": candidate_row.get("relative_names", ""),
            "gender": str(candidate_row.get("gender", "")),
            "entity_id": str(candidate_row.get("entity_id", "")),
        }
        user_text = self._build_user_prompt(query_payload, cand)
        self._ensure_loaded()
        import torch

        messages = [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": user_text},
        ]
        text = self._tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
        )
        inputs = self._tokenizer(text, return_tensors="pt")
        dev = next(self._model.parameters()).device
        inputs = {k: v.to(dev) for k, v in inputs.items()}

        with torch.inference_mode():
            out = self._model.generate(
                **inputs,
                max_new_tokens=self.max_new_tokens,
                do_sample=False,
                pad_token_id=self._tokenizer.pad_token_id,
                eos_token_id=self._tokenizer.eos_token_id,
            )
        new_tokens = out[0, inputs["input_ids"].shape[1] :]
        raw = self._tokenizer.decode(new_tokens, skip_special_tokens=True)
        score, reason = _parse_llm_json(raw)
        return score, reason


def _parse_llm_json(raw: str) -> tuple[float, str]:
    text = raw.strip()
    if not text:
        return 0.5, "llm_parse_failed_empty"

    # Collect all brace-balanced JSON candidates from output.
    candidates: list[str] = []
    stack = 0
    start_idx = -1
    for i, ch in enumerate(text):
        if ch == "{":
            if stack == 0:
                start_idx = i
            stack += 1
        elif ch == "}":
            if stack > 0:
                stack -= 1
                if stack == 0 and start_idx >= 0:
                    candidates.append(text[start_idx : i + 1])
                    start_idx = -1

    if not candidates:
        return 0.5, "llm_parse_failed_no_json"

    parsed_objects: list[dict[str, Any]] = []
    for candidate in candidates:
        try:
            parsed = json.loads(candidate)
        except json.JSONDecodeError:
            continue
        if isinstance(parsed, dict):
            parsed_objects.append(parsed)

    if not parsed_objects:
        return 0.5, "llm_parse_failed_invalid_json"

    # Prefer the last valid object containing numeric score keys.
    obj: dict[str, Any] | None = None
    for parsed in reversed(parsed_objects):
        if "same_person_score" in parsed or "score" in parsed:
            obj = parsed
            break
    if obj is None:
        obj = parsed_objects[-1]

    score = obj.get("same_person_score", obj.get("score", 0.5))
    try:
        f = float(score)
    except (TypeError, ValueError):
        f = 0.5
    f = max(0.0, min(1.0, f))
    reason = str(obj.get("brief_reason", obj.get("reason", "")))[:500]
    if not reason:
        reason = "ok"
    return f, reason


def build_query_payload_from_request(req: Any) -> dict[str, Any]:
    name = req.name or f"{req.first_name} {req.middle_name} {req.last_name}".strip()
    return {
        "name": name,
        "first_name": req.first_name,
        "middle_name": req.middle_name,
        "last_name": req.last_name,
        "dob": req.dob,
        "residency": req.residency,
        "nationality": req.nationality,
        "aliases": ", ".join(req.aliases) if req.aliases else "",
        "relative_names": ", ".join(req.relative_names) if req.relative_names else "",
        "gender": req.gender,
    }
