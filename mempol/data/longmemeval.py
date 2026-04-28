"""LongMemEval loader (xiaowu0162/longmemeval on HuggingFace).

LongMemEval is the benchmark Mastra reports 94.87% on with gpt-5-mini.

Variants:
  - longmemeval_s     ~115K tokens/q, 30-40 sessions  (fits 128K context)
  - longmemeval_m     ~1.5M tokens/q, 500 sessions    (very long context)
  - longmemeval_oracle  only evidence sessions        (control)

This loader downloads the data lazily (requires HF token for first run) and
yields the same `(Conversation, [QA])` shape as our LoCoMo loader so all
existing eval machinery works unchanged.

Usage:
    from mempol.data.longmemeval import load
    convs = load(variant="longmemeval_s", n_convs=5)
"""
from __future__ import annotations
import json
import os
from pathlib import Path
from typing import Literal

from .. import config
from .locomo import Conversation, QA, Turn

_VARIANTS = ("longmemeval_s", "longmemeval_m", "longmemeval_oracle")
_HF_REPO = "xiaowu0162/longmemeval-cleaned"
# The cleaned repo uses these specific filenames (per the dataset card):
_FILE_MAP = {
    "longmemeval_s":      "longmemeval_s_cleaned.json",
    "longmemeval_m":      "longmemeval_m_cleaned.json",
    "longmemeval_oracle": "longmemeval_oracle.json",        # already clean
}


def _local_path(variant: str) -> Path:
    return config.CACHE_DIR / f"{variant}.jsonl"


def _download(variant: str) -> Path:
    """Download cleaned dataset to CACHE_DIR. Requires HF_TOKEN."""
    from huggingface_hub import hf_hub_download
    target = _local_path(variant)
    if target.exists():
        return target
    target.parent.mkdir(parents=True, exist_ok=True)
    fname = _FILE_MAP[variant]
    print(f"[longmemeval] downloading {fname} from {_HF_REPO}…")
    fp = hf_hub_download(
        repo_id=_HF_REPO, filename=fname, repo_type="dataset",
        token=os.getenv("HF_TOKEN"),
    )
    raw = Path(fp).read_text()
    try:
        rows = json.loads(raw)
    except json.JSONDecodeError:
        rows = [json.loads(l) for l in raw.splitlines() if l.strip()]
    with target.open("w") as f:
        for r in rows:
            f.write(json.dumps(r) + "\n")
    print(f"  wrote {len(rows)} rows → {target}")
    return target


def _to_conv_qa(row: dict) -> tuple[Conversation, list[QA]]:
    """LongMemEval row → our Conversation/QA pair.

    Schema (per the dataset README):
      {
        "question_id": str,
        "question_type": str,        # info_extraction | multi-session | knowledge_update | temporal | abstention
        "question": str,
        "answer": str,
        "haystack_session_ids": [str],
        "haystack_dates": [str],
        "haystack_sessions": [[{"role": "user"|"assistant", "content": str}, ...], ...],
      }
    """
    sessions = row.get("haystack_sessions") or []
    dates = row.get("haystack_dates") or [""] * len(sessions)
    qid = str(row.get("question_id", "?"))
    qtype = str(row.get("question_type", "single-hop"))

    turns: list[Turn] = []
    for si, sess in enumerate(sessions):
        date = dates[si] if si < len(dates) else ""
        for ti, t in enumerate(sess):
            turns.append(Turn(
                dia_id=f"D{si+1}:{ti+1}",
                session=si + 1,
                speaker=str(t.get("role", "?")),
                text=str(t.get("content", "")),
                session_date=str(date),
            ))

    conv = Conversation(
        sample_id=qid,
        speaker_a="user",
        speaker_b="assistant",
        turns=turns,
    )
    qa = QA(
        sample_id=qid,
        qid=f"{qid}::q0",
        question=str(row.get("question", "")),
        answer=str(row.get("answer", "")),
        evidence=list(row.get("answer_evidence") or row.get("evidence") or []),
        category=_qtype_to_int(qtype),
    )
    qa.category_name = qtype
    return conv, [qa]


_QTYPE_MAP = {
    "single-session-user": 1, "single-session-assistant": 1, "single-session-preference": 1,
    "single-hop": 1, "multi-session": 2, "multi-hop": 2,
    "knowledge-update": 3, "knowledge_update": 3, "temporal-reasoning": 4,
    "temporal": 4, "abstention": 5, "info-extraction": 1, "info_extraction": 1,
}


def _qtype_to_int(qtype: str) -> int:
    return _QTYPE_MAP.get(qtype.lower(), 0)


def load(
    variant: Literal["longmemeval_s", "longmemeval_m", "longmemeval_oracle"] = "longmemeval_s",
    n_convs: int | None = None,
    download: bool = True,
) -> list[tuple[Conversation, list[QA]]]:
    if variant not in _VARIANTS:
        raise ValueError(f"variant must be one of {_VARIANTS}")
    path = _local_path(variant)
    if not path.exists():
        if not download:
            raise FileNotFoundError(f"{path} not found (set download=True or set HF_TOKEN)")
        _download(variant)

    out = []
    with path.open() as f:
        for i, line in enumerate(f):
            if not line.strip():
                continue
            if n_convs is not None and i >= n_convs:
                break
            row = json.loads(line)
            out.append(_to_conv_qa(row))
    return out


if __name__ == "__main__":
    convs = load(n_convs=2)
    print(f"loaded {len(convs)} convs")
    for c, qas in convs:
        print(f"  {c.sample_id}: {len(c.turns)} turns, {len(qas)} qas")
        print(f"    Q: {qas[0].question[:80]}")
        print(f"    GOLD: {qas[0].answer}")
