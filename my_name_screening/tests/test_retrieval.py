from my_name_screening.retrieval import (
    WatchlistRecord,
    lexical_similarity,
    retrieve_lexical_candidates,
)


WATCHLIST = [
    WatchlistRecord(
        record_id="WL001",
        name="Mohammed Ali",
        aliases=("Muhammad Aly", "محمد علي"),
    ),
    WatchlistRecord(
        record_id="WL002",
        name="Ahmad Hassan",
        aliases=("Ahmed Hasan",),
    ),
    WatchlistRecord(
        record_id="WL003",
        name="John Smith",
    ),
]


def test_lexical_similarity_handles_spelling_variation() -> None:
    score = lexical_similarity(
        "Mohammad Ali",
        "Mohammed Aly",
    )

    assert score >= 70


def test_retrieval_finds_latin_spelling_variant() -> None:
    candidates = retrieve_lexical_candidates(
        "Mohammad Aly",
        WATCHLIST,
    )

    candidate_ids = {
        candidate.record_id
        for candidate in candidates
    }

    assert "WL001" in candidate_ids


def test_retrieval_finds_arabic_alias() -> None:
    candidates = retrieve_lexical_candidates(
        "محمد علي",
        WATCHLIST,
    )

    assert candidates
    assert candidates[0].record_id == "WL001"


def test_retrieval_uses_aliases() -> None:
    candidates = retrieve_lexical_candidates(
        "Ahmed Hasan",
        WATCHLIST,
    )

    assert candidates
    assert candidates[0].record_id == "WL002"
    assert candidates[0].matched_name == "Ahmed Hasan"


def test_retrieval_limits_results() -> None:
    candidates = retrieve_lexical_candidates(
        "Mohammed",
        WATCHLIST,
        minimum_score=20,
        top_k=1,
    )

    assert len(candidates) == 1


def test_retrieval_rejects_empty_query() -> None:
    try:
        retrieve_lexical_candidates("", WATCHLIST)
    except ValueError as error:
        assert str(error) == "Query must not be empty."
    else:
        raise AssertionError("Expected ValueError")