from __future__ import annotations

import hashlib

import pandas as pd

from name_screening.config import AppConfig
from name_screening.index_store import FaissIndexStore
from name_screening.models import ScreenRequest
from name_screening.screener import NameScreener


def _small_watchlist() -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "entity_id": "E001",
                "name": "Hassan Mahmoud Al-Karim",
                "first_name": "Hassan",
                "middle_name": "Mahmoud",
                "last_name": "Al-Karim",
                "dob": "1985-04-12",
                "residency": "Yemen",
                "nationality": "Yemen",
                "aliases": "Abu Ali|حسن الكريم",
                "relative_names": "Ali Hassan|Layla Hassan",
                "gender": "male",
                "risk_level": "high",
            },
            {
                "entity_id": "E002",
                "name": "Hassan Saleh Al-Karim",
                "first_name": "Hassan",
                "middle_name": "Saleh",
                "last_name": "Al-Karim",
                "dob": "1979-09-22",
                "residency": "Yemen",
                "nationality": "Yemen",
                "aliases": "Abu Omar",
                "relative_names": "Ali Hassan",
                "gender": "male",
                "risk_level": "high",
            },
        ]
    )


class FakeEmbeddingService:
    def encode_texts(self, texts: list[str]):
        import faiss
        import numpy as np

        vectors = []
        for text in texts:
            digest = hashlib.sha256(text.encode("utf-8")).digest()
            vals = [b / 255.0 for b in digest[:16]]
            vectors.append(vals)
        arr = np.array(vectors, dtype="float32")
        faiss.normalize_L2(arr)
        return arr


def test_screen_returns_matches(tmp_path) -> None:
    cfg = AppConfig(
        data_dir=tmp_path / "data",
        artifacts_dir=tmp_path / "artifacts",
        watchlist_path=tmp_path / "data" / "watchlist_mock.csv",
        alias_map_path=tmp_path / "data" / "alias_map_mock.csv",
        eval_cases_path=tmp_path / "data" / "eval.csv",
        index_path=tmp_path / "artifacts" / "watchlist.faiss",
        metadata_path=tmp_path / "artifacts" / "watchlist_metadata.csv",
        audit_log_path=tmp_path / "artifacts" / "audit_log.jsonl",
        use_mock_data=False,
    )
    cfg.ensure_dirs()
    (tmp_path / "data" / "alias_map_mock.csv").write_text(
        "alias,canonical_name\nAbu Ali,Hassan Mahmoud Al-Karim\n", encoding="utf-8"
    )
    screener = NameScreener(
        config=cfg,
        embedding_service=FakeEmbeddingService(),
        index_store=FaissIndexStore(cfg.index_path, cfg.metadata_path),
        alias_map={"Abu Ali": "Hassan Mahmoud Al-Karim"},
    )
    screener.ensure_index(_small_watchlist())

    response = screener.screen(
        ScreenRequest(
            name="Abu Ali",
            dob="1985-04-12",
            residency="Yemen",
            relative_names=["Ali Hassan"],
            gender="male",
        )
    )
    assert response.matches
    assert response.matches[0].entity_id == "E001"
    assert response.matches[0].weighted_field_score > response.matches[1].weighted_field_score
    assert "Abu Ali" in response.matches[0].matched_aliases
    assert response.top_score_pct > 0


def test_arabic_variant_gets_strong_field_score(tmp_path) -> None:
    cfg = AppConfig(
        data_dir=tmp_path / "data",
        artifacts_dir=tmp_path / "artifacts",
        watchlist_path=tmp_path / "data" / "watchlist_mock.csv",
        alias_map_path=tmp_path / "data" / "alias_map_mock.csv",
        eval_cases_path=tmp_path / "data" / "eval.csv",
        index_path=tmp_path / "artifacts" / "watchlist.faiss",
        metadata_path=tmp_path / "artifacts" / "watchlist_metadata.csv",
        audit_log_path=tmp_path / "artifacts" / "audit_log.jsonl",
        use_mock_data=False,
    )
    cfg.ensure_dirs()
    screener = NameScreener(
        config=cfg,
        embedding_service=FakeEmbeddingService(),
        index_store=FaissIndexStore(cfg.index_path, cfg.metadata_path),
        alias_map={"Abu Ali": "Hassan Mahmoud Al-Karim"},
    )
    watchlist = pd.DataFrame(
        [
            {
                "entity_id": "M001",
                "name": "Hassan Mahmoud Al-Karim",
                "first_name": "Hassan",
                "middle_name": "Mahmoud",
                "last_name": "Al-Karim",
                "dob": "1981-05-07",
                "residency": "Jordan",
                "nationality": "Jordan",
                "aliases": "Abu Ali|حسن الكريم",
                "relative_names": "Ali Hassan|Layla Hassan",
                "gender": "male",
                "risk_level": "high",
            },
            {
                "entity_id": "M002",
                "name": "حسن محمود الكريم",
                "first_name": "حسن",
                "middle_name": "محمود",
                "last_name": "الكريم",
                "dob": "1981-05-07",
                "residency": "الأردن",
                "nationality": "الأردن",
                "aliases": "Abu Ali|Hassan Al-Karim",
                "relative_names": "علي حسن|ليلى حسن",
                "gender": "male",
                "risk_level": "high",
            },
        ]
    )
    screener.ensure_index(watchlist)
    response = screener.screen(
        ScreenRequest(
            name="Abu Ali",
            dob="1981-05-07",
            residency="Jordan",
            relative_names=["Ali Hassan"],
            gender="male",
            top_k=2,
        )
    )
    by_id = {m.entity_id: m for m in response.matches}
    assert by_id["M002"].weighted_field_score >= 0.8
