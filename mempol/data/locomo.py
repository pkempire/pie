"""LoCoMo loader — yields normalized (Conversation, [QA]) pairs."""
from __future__ import annotations
import json
import re
from dataclasses import dataclass, field
from pathlib import Path

from .. import config

_CAT = {1: "single-hop", 2: "multi-hop", 3: "open-domain", 4: "temporal", 5: "adversarial"}


@dataclass
class Turn:
    dia_id: str            # e.g. "D1:3"
    session: int
    speaker: str
    text: str
    session_date: str      # human-readable, e.g. "8 May, 2023 1:56 pm"


@dataclass
class Conversation:
    sample_id: str
    speaker_a: str
    speaker_b: str
    turns: list[Turn]


@dataclass
class QA:
    sample_id: str
    qid: str
    question: str
    answer: str
    evidence: list[str]    # list of dia_ids
    category: int
    category_name: str = ""

    def __post_init__(self):
        self.category_name = _CAT.get(self.category, f"cat{self.category}")


def load(path: Path | None = None, n_convs: int | None = None) -> list[tuple[Conversation, list[QA]]]:
    """Load LoCoMo. Returns [(conv, qas), ...]."""
    path = path or config.LOCOMO_PATH
    raw = json.loads(Path(path).read_text())
    out = []
    for sample in (raw if n_convs is None else raw[:n_convs]):
        sid = sample["sample_id"]
        c = sample["conversation"]
        # collect sessions in order
        sess_ids = sorted(
            [int(m.group(1)) for k in c if (m := re.match(r"session_(\d+)$", k))]
        )
        turns: list[Turn] = []
        for n in sess_ids:
            session_list = c.get(f"session_{n}") or []
            date = c.get(f"session_{n}_date_time", "")
            for t in session_list:
                turns.append(
                    Turn(
                        dia_id=t.get("dia_id", f"D{n}:?"),
                        session=n,
                        speaker=t.get("speaker", "?"),
                        text=t.get("text", ""),
                        session_date=date,
                    )
                )
        conv = Conversation(
            sample_id=sid,
            speaker_a=c.get("speaker_a", "A"),
            speaker_b=c.get("speaker_b", "B"),
            turns=turns,
        )
        qas = []
        for i, q in enumerate(sample.get("qa", [])):
            qas.append(
                QA(
                    sample_id=sid,
                    qid=f"{sid}::q{i}",
                    question=str(q.get("question", "")),
                    answer=str(q.get("answer", "")),
                    evidence=list(q.get("evidence", []) or []),
                    category=int(q.get("category", 0)),
                )
            )
        out.append((conv, qas))
    return out
