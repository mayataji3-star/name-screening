from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, Protocol

from rapidfuzz import fuzz

from .normalization import normalize_name
from .phonetics import phonetic_keys, transliterate_to_latin


@dataclass(frozen=True)
class WatchlistRecord:
    """One simplified watchlist record."""

    record_id: str
    name: str
    aliases: tuple[str, ...] = ()


@dataclass(frozen=True)
class Candidate:
    """A watchlist record retrieved for detailed scoring."""

    record_id: str
    matched_name: str
    lexical_score: float | None = None
    semantic_score: float | None = None
    phonetic_match: bool = False
    retrieval_channels: tuple[str, ...] = ()


class SemanticRetrieverProtocol(Protocol):
    """Interface required from a semantic retriever."""

    def retrieve(
        self,
        query: str,
        *,
        top_k: int = 20,
        minimum_score: float = 0.50,
    ) -> list[Candidate]:
        """Retrieve semantic candidates."""
        ...


def lexical_similarity(left: str, right: str) -> float:
    """Return fuzzy similarity between two names from 0 to 100."""
    left_normalized = normalize_name(left)
    right_normalized = normalize_name(right)

    normalized_score = fuzz.token_set_ratio(
        left_normalized,
        right_normalized,
    )

    left_latin = transliterate_to_latin(left)
    right_latin = transliterate_to_latin(right)

    transliterated_score = fuzz.token_set_ratio(
        left_latin,
        right_latin,
    )

    return float(
        max(
            normalized_score,
            transliterated_score,
        )
    )


def has_phonetic_overlap(left: str, right: str) -> bool:
    """Return True when two names share a phonetic key."""
    left_keys = phonetic_keys(left)
    right_keys = phonetic_keys(right)

    return bool(left_keys & right_keys)




#This Function searches the watchlist using fuzzy spelling similarity and phonetic similarity
#It returns a list of candidates that match the query based on the specified minimum score and top_k parameters
def retrieve_lexical_candidates(
    query: str,
    watchlist: Iterable[WatchlistRecord],
    *,
    minimum_score: float = 65.0,
    top_k: int = 20,
) -> list[Candidate]:
    """Retrieve candidates using fuzzy and phonetic methods."""
    if not isinstance(query, str):
        raise TypeError("Query must be a string.")

    if not query.strip():
        raise ValueError("Query must not be empty.")

    if not 0 <= minimum_score <= 100:
        raise ValueError(
            "minimum_score must be between 0 and 100."
        )

    if top_k < 1:
        raise ValueError("top_k must be at least 1.")

    candidates: list[Candidate] = []

    for record in watchlist:
        possible_names = (
            record.name,
            *record.aliases,
        )

        best_candidate: Candidate | None = None

        for possible_name in possible_names:
            lexical_score = lexical_similarity(
                query,
                possible_name,
            )

            phonetic_match = has_phonetic_overlap(
                query,
                possible_name,
            )

            if (
                lexical_score < minimum_score
                and not phonetic_match
            ):
                continue

            channels: list[str] = []

            if lexical_score >= minimum_score:
                channels.append("lexical")

            if phonetic_match:
                channels.append("phonetic")

            candidate = Candidate(
                record_id=record.record_id,
                matched_name=possible_name,
                lexical_score=round(lexical_score, 2),
                phonetic_match=phonetic_match,
                retrieval_channels=tuple(channels),
            )

            if best_candidate is None:
                best_candidate = candidate
                continue

            candidate_rank = (
                candidate.phonetic_match,
                candidate.lexical_score or 0.0,
            )

            best_rank = (
                best_candidate.phonetic_match,
                best_candidate.lexical_score or 0.0,
            )

            if candidate_rank > best_rank:
                best_candidate = candidate

        if best_candidate is not None:
            candidates.append(best_candidate)

    candidates.sort(
        key=lambda candidate: (
            candidate.phonetic_match,
            candidate.lexical_score or 0.0,
        ),
        reverse=True,
    )

    return candidates[:top_k]

#this function combines candidates from two independent retrieval channels into a single list of candidates
def union_candidates(
    lexical_candidates: list[Candidate],
    semantic_candidates: list[Candidate],
    *,
    top_k: int = 20,
) -> list[Candidate]:
    """Combine candidates from independent retrieval channels."""
    if top_k < 1:
        raise ValueError("top_k must be at least 1.")

    combined: dict[str, Candidate] = {}

    for candidate in lexical_candidates:
        combined[candidate.record_id] = candidate

    for semantic_candidate in semantic_candidates:
        existing = combined.get(
            semantic_candidate.record_id
        )

        if existing is None:
            combined[semantic_candidate.record_id] = (
                semantic_candidate
            )
            continue

        channels = tuple(
            dict.fromkeys(
                existing.retrieval_channels
                + semantic_candidate.retrieval_channels
            )
        )

        combined[existing.record_id] = Candidate(
            record_id=existing.record_id,
            matched_name=existing.matched_name,
            lexical_score=existing.lexical_score,
            semantic_score=semantic_candidate.semantic_score,
            phonetic_match=existing.phonetic_match,
            retrieval_channels=channels,
        )

    results = list(combined.values())

    results.sort(
        key=lambda candidate: (
            len(candidate.retrieval_channels),
            candidate.phonetic_match,
            candidate.semantic_score or 0.0,
            (candidate.lexical_score or 0.0) / 100,
        ),
        reverse=True,
    )

    return results[:top_k]

#this function runs both retrieval channels (lexical and semantic) and combines their results into a single list of candidates
def retrieve_candidates(
    query: str,
    watchlist: list[WatchlistRecord],
    semantic_retriever: SemanticRetrieverProtocol,
    *,
    lexical_minimum: float = 65.0,
    semantic_minimum: float = 0.50,
    channel_top_k: int = 20,
    final_top_k: int = 20,
) -> list[Candidate]:
    """Run both retrieval channels and union their results."""
    lexical_candidates = retrieve_lexical_candidates(
        query,
        watchlist,
        minimum_score=lexical_minimum,
        top_k=channel_top_k,
    )

    semantic_candidates = semantic_retriever.retrieve(
        query,
        minimum_score=semantic_minimum,
        top_k=channel_top_k,
    )

    return union_candidates(
        lexical_candidates,
        semantic_candidates,
        top_k=final_top_k,
    )