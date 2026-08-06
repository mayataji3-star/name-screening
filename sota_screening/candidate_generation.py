from __future__ import annotations

from dataclasses import dataclass

import faiss
import pandas as pd

from name_screening.embedding import EmbeddingService

from .normalization import normalize_text


@dataclass
class CandidateGenerator:
    embedding_service: EmbeddingService
    watchlist: pd.DataFrame | None = None
    index: faiss.Index | None = None

    def build(self, watchlist: pd.DataFrame) -> None:
        self.watchlist = watchlist.reset_index(drop=True).copy()
        texts = self.watchlist["name"].fillna("").map(normalize_text).tolist()
        vectors = self.embedding_service.encode_texts(texts)
        idx = faiss.IndexFlatIP(vectors.shape[1])
        idx.add(vectors)
        self.index = idx

    def retrieve(self, query_name: str, top_k: int) -> pd.DataFrame:
        if self.watchlist is None or self.index is None:
            raise RuntimeError("Candidate generator index is not built.")
        query_vec = self.embedding_service.encode_texts([normalize_text(query_name)])
        scores, indices = self.index.search(query_vec, top_k)
        rows: list[dict[str, object]] = []
        for score, idx in zip(scores[0], indices[0]):
            if idx < 0:
                continue
            row = self.watchlist.iloc[int(idx)].to_dict()
            row["retrieval_score"] = float(max(0.0, min(1.0, score)))
            rows.append(row)
        return pd.DataFrame(rows)

