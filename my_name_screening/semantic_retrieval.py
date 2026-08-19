from __future__ import annotations

from dataclasses import dataclass

import faiss
import numpy as np
from sentence_transformers import SentenceTransformer

from .retrieval import Candidate, WatchlistRecord


@dataclass(frozen=True)
class IndexedName:
    """Connect an embedded name to its watchlist record."""

    record_id: str
    name: str


class SemanticRetriever:
    """Retrieve candidates using multilingual E5 embeddings."""

    def __init__(
        self,
        model_name: str = "intfloat/multilingual-e5-base",
        *,
        local_files_only: bool = True,
    ) -> None:
        self.model = SentenceTransformer(
            model_name,
            local_files_only=local_files_only,
        )

        self.index: faiss.Index | None = None
        self.indexed_names: list[IndexedName] = []
        self.embeddings: np.ndarray | None = None

    def build_index(
        
        self,
        watchlist: list[WatchlistRecord],
    ) -> None:
        """Embed watchlist names and build a FAISS index."""
        if not watchlist:
            raise ValueError("Watchlist must not be empty.")

        self.indexed_names = []

        for record in watchlist:
            possible_names = (
                record.name,
                *record.aliases,
            )

            for name in possible_names:
                if not name.strip():
                    continue

                self.indexed_names.append(
                    IndexedName(
                        record_id=record.record_id,
                        name=name,
                    )
                )

        if not self.indexed_names:
            raise ValueError(
                "Watchlist contains no usable names."
            )

        passages = [
            f"passage: {indexed_name.name}"
            for indexed_name in self.indexed_names
        ]

        embeddings = self.model.encode(
            passages,
            convert_to_numpy=True,
            normalize_embeddings=True,
            show_progress_bar=False,
        )

        self.embeddings = np.asarray(
            embeddings,
            dtype=np.float32,
        )

        dimension = self.embeddings.shape[1]

        self.index = faiss.IndexFlatIP(dimension)
        self.index.add(self.embeddings)

    def _validate_retrieval_input(
        self,
        query: str,
        top_k: int,
        minimum_score: float,
    ) -> None:
        """Validate semantic retrieval arguments."""
        if not isinstance(query, str):
            raise TypeError("Query must be a string.")

        if not query.strip():
            raise ValueError("Query must not be empty.")

        if top_k < 1:
            raise ValueError("top_k must be at least 1.")

        if not 0 <= minimum_score <= 1:
            raise ValueError(
                "minimum_score must be between 0 and 1."
            )

        if self.embeddings is None or self.index is None:
            raise RuntimeError(
                "Call build_index() before retrieval."
            )

    def _create_query_embedding(
        self,
        query: str,
    ) -> np.ndarray:
        """Create a normalized E5 embedding for one query."""
        embedding = self.model.encode(
            [f"query: {query}"],
            convert_to_numpy=True,
            normalize_embeddings=True,
            show_progress_bar=False,
        )

        return np.asarray(
            embedding,
            dtype=np.float32,
        )

    def _deduplicate_results(
        self,
        scored_positions: list[tuple[float, int]],
        *,
        channel: str,
        minimum_score: float,
        top_k: int,
    ) -> list[Candidate]:
        """Keep the best matching name or alias per record."""
        best_by_record: dict[str, Candidate] = {}

        for score, position in scored_positions:
            if position < 0:
                continue

            if score < minimum_score:
                continue

            indexed_name = self.indexed_names[position]

            candidate = Candidate(
                record_id=indexed_name.record_id,
                matched_name=indexed_name.name,
                semantic_score=round(score, 4),
                retrieval_channels=(channel,),
            )

            existing = best_by_record.get(
                indexed_name.record_id
            )

            if (
                existing is None
                or (candidate.semantic_score or 0.0)
                > (existing.semantic_score or 0.0)
            ):
                best_by_record[indexed_name.record_id] = (
                    candidate
                )

        candidates = list(best_by_record.values())

        candidates.sort(
            key=lambda candidate: (
                candidate.semantic_score or 0.0
            ),
            reverse=True,
        )

        return candidates[:top_k]

    def retrieve(
        self,
        query: str,
        *,
        top_k: int = 20,
        minimum_score: float = 0.50,
    ) -> list[Candidate]:
        """Retrieve semantic candidates using FAISS."""
        self._validate_retrieval_input(
            query,
            top_k,
            minimum_score,
        )

        query_embedding = self._create_query_embedding(
            query
        )

        # Search more name/alias entries because several entries
        # may belong to the same record.
        search_size = min(
            max(top_k * 5, top_k),
            len(self.indexed_names),
        )

        assert self.index is not None

        scores, positions = self.index.search(
            query_embedding,
            search_size,
        )

        scored_positions = [
            (float(score), int(position))
            for score, position in zip(
                scores[0],
                positions[0],
            )
        ]

        return self._deduplicate_results(
            scored_positions,
            channel="semantic",
            minimum_score=minimum_score,
            top_k=top_k,
        )

    def retrieve_brute_force(
        self,
        query: str,
        *,
        top_k: int = 20,
        minimum_score: float = 0.50,
    ) -> list[Candidate]:
        """Compare the query directly with every stored embedding."""
        self._validate_retrieval_input(
            query,
            top_k,
            minimum_score,
        )

        query_embedding = self._create_query_embedding(
            query
        )

        assert self.embeddings is not None

        # Both sides are normalized, so inner product equals
        # cosine similarity.
        similarities = (
            self.embeddings @ query_embedding[0]
        )

        sorted_positions = np.argsort(
            similarities
        )[::-1]

        scored_positions = [
            (
                float(similarities[position]),
                int(position),
            )
            for position in sorted_positions
        ]

        return self._deduplicate_results(
            scored_positions,
            channel="semantic_brute_force",
            minimum_score=minimum_score,
            top_k=top_k,
        )