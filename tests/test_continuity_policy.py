from mempol.backends.base import Hit, Unit
from mempol.policies.continuity import _deterministic_answer_fallback


def test_date_difference_fallback_for_blank_temporal_answer():
    evidence = [
        Hit(
            Unit(
                uid="moma",
                text="user: I just got back from a guided tour at the Museum of Modern Art.",
                metadata={"session_date": "2023/01/08 (Sun) 12:49", "dia_id": "D5:1"},
            ),
            1.0,
            "test",
        ),
        Hit(
            Unit(
                uid="met",
                text='user: I attended the "Ancient Civilizations" exhibit at the Metropolitan Museum of Art today.',
                metadata={"session_date": "2023/01/15 (Sun) 00:27", "dia_id": "D28:7"},
            ),
            1.0,
            "test",
        ),
    ]
    answer = _deterministic_answer_fallback(
        "How many days passed between my visit to MoMA and the Ancient Civilizations exhibit?",
        evidence,
    )
    assert answer == "7 days"

