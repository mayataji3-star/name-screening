from __future__ import annotations

from pathlib import Path

import faiss
import pandas as pd


class FaissIndexStore:
    def __init__(self, index_path: Path, metadata_path: Path) -> None:
        self.index_path = index_path
        self.metadata_path = metadata_path

    def save(self, index: faiss.Index, metadata_df: pd.DataFrame) -> None:
        self.index_path.parent.mkdir(parents=True, exist_ok=True)
        faiss.write_index(index, str(self.index_path))
        metadata_df.to_csv(self.metadata_path, index=False)

    def exists(self) -> bool:
        return self.index_path.is_file() and self.metadata_path.is_file()

    def load(self) -> tuple[faiss.Index, pd.DataFrame]:
        index = faiss.read_index(str(self.index_path))
        metadata = pd.read_csv(self.metadata_path)
        return index, metadata
