from __future__ import annotations

from pathlib import Path

from name_screening.data_io import load_alias_map, load_watchlist


def test_load_watchlist_from_opensanctions_stream(tmp_path: Path) -> None:
    ftm = tmp_path / "entities.ftm.json"
    ftm.write_text(
        "\n".join(
            [
                '{"id":"os-1","schema":"Person","properties":{"name":["Jane Doe"],"alias":["J. Doe","جين دو"],"birthDate":["1988-02-03"],"nationality":["US"]}}',
                '{"id":"os-2","schema":"Organization","properties":{"name":["ACME Corp"]}}',
                '{"id":"os-3","schema":"Person","properties":{"name":["John Smith"],"country":["GB"]}}',
            ]
        ),
        encoding="utf-8",
    )

    csv_fallback = tmp_path / "seed.csv"
    csv_fallback.write_text(
        "entity_id,name,dob,nationality,aliases,risk_level\n"
        "E001,Tariq Ibrahim Al-Hashimi,1985-04-12,Yemen,a1,high\n",
        encoding="utf-8",
    )

    df = load_watchlist(
        str(csv_fallback),
        opensanctions_path=str(ftm),
        entity_scope="people_only",
        max_records=10,
    )

    assert len(df) == 2
    assert set(df["entity_id"]) == {"os-1", "os-3"}
    assert "Jane Doe" in set(df["name"])


def test_load_watchlist_mock_and_alias_map(tmp_path: Path) -> None:
    watchlist_file = tmp_path / "watchlist_mock.csv"
    alias_file = tmp_path / "alias_map_mock.csv"
    df = load_watchlist(
        str(watchlist_file),
        alias_map_path=str(alias_file),
        use_mock_data=True,
    )
    alias_map = load_alias_map(alias_file)

    assert {"first_name", "middle_name", "last_name", "residency", "gender"}.issubset(
        set(df.columns)
    )
    assert len(df) >= 20
    assert df["name"].nunique() == len(df)
    assert alias_map.get("Abu Ali") == "Hassan Mahmoud Al-Karim"
