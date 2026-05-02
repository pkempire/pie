"""LoCoMo loader — yields normalized (Conversation, [QA]) pairs."""
from __future__ import annotations
import json
import re
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path

from .. import config

_CAT = {
    # Local LoCoMo category ids:
    # 1 = multi-hop, 2 = temporal, 3 = open-domain, 4 = single-hop,
    # 5 = adversarial.
    1: "multi-hop",
    2: "temporal",
    3: "open-domain",
    4: "single-hop",
    5: "adversarial",
}

_DATE_RE_A = re.compile(
    r"(\d{1,2})\s+(\w+)\s+(\d{4})\s+at\s+(\d{1,2}):(\d{2})\s*(am|pm)",
    re.IGNORECASE,
)
_DATE_RE_B = re.compile(
    r"(\d{1,2}):(\d{2})\s*(am|pm)\s+on\s+(\d{1,2})\s+(\w+),?\s+(\d{4})",
    re.IGNORECASE,
)

MONTH_MAP = {
    "january": 1, "february": 2, "march": 3, "april": 4,
    "may": 5, "june": 6, "july": 7, "august": 8,
    "september": 9, "october": 10, "november": 11, "december": 12,
    "jan": 1, "feb": 2, "mar": 3, "apr": 4, "jun": 6, "jul": 7,
    "aug": 8, "sep": 9, "oct": 10, "nov": 11, "dec": 12,
}


def parse_locomo_date(date_str: str) -> float:
    """Parse the date formats used by LoCoMo into a UTC Unix timestamp."""
    if not date_str:
        return 0.0
    stripped = date_str.strip()
    m = _DATE_RE_A.match(stripped)
    if m:
        day = int(m.group(1))
        month_str = m.group(2).lower()
        year = int(m.group(3))
        hour = int(m.group(4))
        minute = int(m.group(5))
        ampm = m.group(6).lower()
    else:
        m = _DATE_RE_B.match(stripped)
        if not m:
            return 0.0
        hour = int(m.group(1))
        minute = int(m.group(2))
        ampm = m.group(3).lower()
        day = int(m.group(4))
        month_str = m.group(5).lower()
        year = int(m.group(6))
    month = MONTH_MAP.get(month_str)
    if not month:
        return 0.0
    if ampm == "pm" and hour != 12:
        hour += 12
    elif ampm == "am" and hour == 12:
        hour = 0
    return datetime(year, month, day, hour, minute, tzinfo=timezone.utc).timestamp()


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
                    answer=str(q.get("answer") or q.get("adversarial_answer", "")),
                    evidence=list(q.get("evidence", []) or []),
                    category=int(q.get("category", 0)),
                )
            )
        out.append((conv, qas))
    return out
