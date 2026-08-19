"""Make one controlled Groq adjudication request."""

from __future__ import annotations

from my_name_screening.groq_judge import GroqJudge
from my_name_screening.llm_models import Verdict


query = {
    "name": "Hasan Mahmoud Al Karim",
    "dob": "1981-05-07",
    "nationality": "Jordan",
    "residency": "Jordan",
}

candidate = {
    "record_id": "M001",
    "name": "Hassan Mahmoud Al-Karim",
    "aliases": [
        "حسن الكريم",
        "Abu Ali",
    ],
    "dob": "1981-05-07",
    "nationality": "Jordan",
    "residency": "Jordan",
}

judge = GroqJudge()
result = judge.judge_pair(
    query,
    candidate,

    # Temporary example from the deterministic stages.
    deterministic_score=0.80,
    deterministic_verdict=Verdict.POSSIBLE,
)
print(
    result.model_dump_json(
        indent=2,
    )
)