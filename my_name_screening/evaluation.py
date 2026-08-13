from __future__ import annotations

from dataclasses import dataclass

from .retrieval import Candidate


@dataclass(frozen=True)
class EvalCase:
    """A query with all acceptable correct record IDs."""

    query: str
    expected_record_ids: frozenset[str]


@dataclass(frozen=True)
class RecallResult:
    """Recall result for one retrieval method."""

    method: str
    successful_queries: int
    total_queries: int
    recall_at_k: float


@dataclass(frozen=True)
class RetrievalDifference:
    """Differences between indexed and brute-force results."""

    query: str
    shared_ids: frozenset[str]
    indexed_only_ids: frozenset[str]
    brute_force_only_ids: frozenset[str]


def calculate_recall_at_k(
    method: str,
    retrieved_results: list[list[Candidate]],
    eval_cases: list[EvalCase],
    *,
    k: int = 20,
) -> RecallResult:
    """Calculate how often a correct record appears in top k."""
    if k < 1:
        raise ValueError("k must be at least 1.")

    if not eval_cases:
        raise ValueError(
            "Evaluation cases must not be empty."
        )

    if len(retrieved_results) != len(eval_cases):
        raise ValueError(
            "Results and evaluation cases must have "
            "the same length."
        )

    successful_queries = 0

    for case, candidates in zip(
        eval_cases,
        retrieved_results,
    ):
        retrieved_ids = {
            candidate.record_id
            for candidate in candidates[:k]
        }

        if case.expected_record_ids & retrieved_ids:
            successful_queries += 1

    recall = successful_queries / len(eval_cases)

    return RecallResult(
        method=method,
        successful_queries=successful_queries,
        total_queries=len(eval_cases),
        recall_at_k=recall,
    )


def compare_indexed_and_brute_force(
    query: str,
    indexed_candidates: list[Candidate],
    brute_force_candidates: list[Candidate],
    *,
    k: int = 20,
) -> RetrievalDifference:
    """Compare FAISS and brute-force candidate IDs."""
    indexed_ids = frozenset(
        candidate.record_id
        for candidate in indexed_candidates[:k]
    )

    brute_force_ids = frozenset(
        candidate.record_id
        for candidate in brute_force_candidates[:k]
    )

    return RetrievalDifference(
        query=query,
        shared_ids=indexed_ids & brute_force_ids,
        indexed_only_ids=indexed_ids - brute_force_ids,
        brute_force_only_ids=(
            brute_force_ids - indexed_ids
        ),
    )