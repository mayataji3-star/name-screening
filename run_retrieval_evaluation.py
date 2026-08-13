from __future__ import annotations

from my_name_screening.evaluation import (
    EvalCase,
    calculate_recall_at_k,
    compare_indexed_and_brute_force,
)
from my_name_screening.retrieval import (
    retrieve_lexical_candidates,
    union_candidates,
)
from my_name_screening.semantic_retrieval import (
    SemanticRetriever,
)
from my_name_screening.watchlist_loader import (
    load_watchlist,
)


EVAL_CASES = [
    EvalCase(
        query="Hasan Mahmoud Al Karim",
        expected_record_ids=frozenset(
            {"M001", "M002"}
        ),
    ),
    EvalCase(
        query="حسن محمود الكريم",
        expected_record_ids=frozenset(
            {"M001", "M002"}
        ),
    ),
    EvalCase(
        query="Nour Samir Darwish",
        expected_record_ids=frozenset(
            {"M004", "M005"}
        ),
    ),
    EvalCase(
        query="نور سمير درويش",
        expected_record_ids=frozenset(
            {"M004", "M005"}
        ),
    ),
    EvalCase(
        query="Kareem Nabil Mansour",
        expected_record_ids=frozenset(
            {"M006", "M007"}
        ),
    ),
    EvalCase(
        query="كريم نبيل منصور",
        expected_record_ids=frozenset(
            {"M006", "M007"}
        ),
    ),
    EvalCase(
        query="Omar Hadi Al Qadri",
        expected_record_ids=frozenset(
            {"M008", "M009"}
        ),
    ),
    EvalCase(
        query="Layla Fadi Shalaby",
        expected_record_ids=frozenset(
            {"M010", "M011"}
        ),
    ),
    EvalCase(
        query="Yusuf Aziz Hamoud",
        expected_record_ids=frozenset(
            {"M012", "M013"}
        ),
    ),
    EvalCase(
        query="Samir Rahman Kattan",
        expected_record_ids=frozenset(
            {"M014", "M015"}
        ),
    ),
    EvalCase(
        query="Houda Karim Nadeem",
        expected_record_ids=frozenset(
            {"M016", "M017"}
        ),
    ),
    EvalCase(
        query="Anwar Imad Yafi",
        expected_record_ids=frozenset(
            {"M018", "M019"}
        ),
    ),
    EvalCase(
        query="Rami Bashir Saad",
        expected_record_ids=frozenset(
            {"M020", "M021"}
        ),
    ),
]


def main() -> None:
    """Run retrieval evaluation and compare FAISS with brute force."""
    watchlist = load_watchlist(
        "data/watchlist_mock.csv"
    )

    semantic_retriever = SemanticRetriever()
    semantic_retriever.build_index(watchlist)

    lexical_results = []
    indexed_results = []
    brute_force_results = []
    combined_results = []
    differences = []

    for case in EVAL_CASES:
        lexical = retrieve_lexical_candidates(
            case.query,
            watchlist,
            minimum_score=65.0,
            top_k=20,
        )

        indexed = semantic_retriever.retrieve(
            case.query,
            minimum_score=0.50,
            top_k=20,
        )

        brute_force = (
            semantic_retriever.retrieve_brute_force(
                case.query,
                minimum_score=0.50,
                top_k=20,
            )
        )

        combined = union_candidates(
            lexical,
            indexed,
            top_k=20,
        )

        lexical_results.append(lexical)
        indexed_results.append(indexed)
        brute_force_results.append(brute_force)
        combined_results.append(combined)

        differences.append(
            compare_indexed_and_brute_force(
                case.query,
                indexed,
                brute_force,
                k=20,
            )
        )

    methods = [
        (
            "Lexical + phonetic",
            lexical_results,
        ),
        (
            "Semantic FAISS",
            indexed_results,
        ),
        (
            "Semantic brute force",
            brute_force_results,
        ),
        (
            "Combined union",
            combined_results,
        ),
    ]

    for k in (1, 5, 10, 20):
        print(f"\nRecall@{k} results")
        print("-" * 60)

        for method_name, method_results in methods:
            result = calculate_recall_at_k(
                method_name,
                method_results,
                EVAL_CASES,
                k=k,
            )

            percentage = result.recall_at_k * 100

            print(
                f"{result.method:<25} "
                f"{result.successful_queries}/"
                f"{result.total_queries} "
                f"({percentage:.2f}%)"
            )

    print("\nFAISS versus brute-force differences")
    print("-" * 60)

    difference_found = False

    for difference in differences:
        if (
            difference.indexed_only_ids
            or difference.brute_force_only_ids
        ):
            difference_found = True

            print(f"\nQuery: {difference.query}")
            print(
                "FAISS only:",
                sorted(difference.indexed_only_ids),
            )
            print(
                "Brute-force only:",
                sorted(
                    difference.brute_force_only_ids
                ),
            )

    if not difference_found:
        print(
            "No candidate-ID differences were found "
            "in the top 20."
        )


if __name__ == "__main__":
    main()