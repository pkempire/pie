"""Benchmark rows → MemoryDatum (analog of SearchR1Datum from tinker-cookbook).

For multi-source training we accept LoCoMo + LongMemEval + (optional) MSC + WildChat
synthetic. Each MemoryDatum is one (conversation, question, gold_answer) triple.

The conversation is the *knowledge base* for that env. The model only sees the
question; the env injects a backend pre-ingested with the conversation, and
exposes memory ops as tools.
"""
from __future__ import annotations
import json
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Iterator, Literal, TypedDict

from mempol.data.locomo import Conversation, QA, load as load_locomo
from mempol.data.longmemeval import load as load_longmemeval
from mempol.eval.runner import conv_to_units


class MemoryDatum(TypedDict):
    """One training example for the memory RL env."""
    question: str
    gold_answer: str
    category: str               # "single-hop" | "multi-hop" | "temporal" | ...
    data_source: str            # "locomo" | "longmemeval" | "msc" | "wildchat_synth"
    sample_id: str              # which conv/episode this Q belongs to
    qid: str
    # The conversation is passed by reference (a list of turn dicts) — the env
    # builder ingests it into a backend at env construction.
    conversation_units: list[dict]


def _conv_to_serializable_units(
    conv: Conversation,
    window: int = 6,
    stride: int = 3,
) -> list[dict]:
    """Chunked windowing over conversation turns.

    Each unit covers a sliding window of W consecutive turns within a session
    so embeddings have actual context (a single turn like 'Bye!' has no
    retrievable signal). Stride S < W gives overlap so important content isn't
    split across chunk boundaries.

    Defaults (W=6, S=3) on LoCoMo conv-30 (~350 turns) → ~120 chunks instead
    of 350 turn-units. Each chunk:
      - text:     "[YYYY-MM-DD HH:MM | session N]\\nA: turn1\\nB: turn2\\n..."
      - metadata: dia_ids in chunk, session number, session_date, first/last
                  dia_id (for evidence-recall in eval).
    """
    # Group turns by session, keep order
    by_session: dict = {}
    for t in conv.turns:
        by_session.setdefault(t.session, []).append(t)

    units = []
    for sess_n in sorted(by_session.keys()):
        session_turns = by_session[sess_n]
        if not session_turns:
            continue
        date = session_turns[0].session_date
        n = len(session_turns)
        # If session is shorter than window, single chunk; else slide
        if n <= window:
            ranges = [(0, n)]
        else:
            ranges = []
            i = 0
            while i < n:
                ranges.append((i, min(i + window, n)))
                if i + window >= n:
                    break
                i += stride

        for chunk_idx, (a, b) in enumerate(ranges):
            chunk_turns = session_turns[a:b]
            header = f"[{date} | session {sess_n}]"
            body = "\n".join(f"{t.speaker}: {t.text}" for t in chunk_turns)
            text = f"{header}\n{body}"
            uid = f"{conv.sample_id}::S{sess_n}::C{chunk_idx}"
            dia_ids = [t.dia_id for t in chunk_turns]
            units.append({
                "uid": uid,
                "text": text,
                "metadata": {
                    "session": sess_n,
                    "session_date": date,
                    "first_dia_id": chunk_turns[0].dia_id,
                    "last_dia_id": chunk_turns[-1].dia_id,
                    "dia_ids": dia_ids,
                    "speaker": chunk_turns[0].speaker,  # first speaker in chunk
                    "n_turns": len(chunk_turns),
                    "chunk_idx_in_session": chunk_idx,
                    # backward compat with older Backend code that expects "timestamp"
                    "timestamp": float(sess_n),
                    "dia_id": chunk_turns[0].dia_id,
                },
            })
    return units


def locomo_to_memory_data(
    n_convs: int | None = None,
    train_frac: float = 0.8,
    seed: int = 0,
) -> tuple[list[MemoryDatum], list[MemoryDatum]]:
    """Split LoCoMo at the conversation level (not Q level) so train/eval QAs
    come from disjoint conversations — the right OOD test."""
    convs = load_locomo(n_convs=n_convs)
    rng = random.Random(seed)
    rng.shuffle(convs)
    n_train = int(len(convs) * train_frac)

    def to_data(convs_split: list[tuple[Conversation, list[QA]]]) -> list[MemoryDatum]:
        out: list[MemoryDatum] = []
        for conv, qas in convs_split:
            units = _conv_to_serializable_units(conv)
            for qa in qas:
                out.append({
                    "question": qa.question,
                    "gold_answer": qa.answer,
                    "category": qa.category_name,
                    "data_source": "locomo",
                    "sample_id": conv.sample_id,
                    "qid": qa.qid,
                    "conversation_units": units,
                })
        return out

    return to_data(convs[:n_train]), to_data(convs[n_train:])


def longmemeval_jsonl_to_memory_data(jsonl_path: Path) -> list[MemoryDatum]:
    """Compatibility parser for raw LongMemEval JSONL files.

    Prefer `longmemeval_to_memory_data()` below for training: it goes through
    `mempol.data.longmemeval.load`, which preserves session dates and chunking in
    the same shape as LoCoMo.
    """
    if not jsonl_path.exists():
        return []
    out = []
    for line in jsonl_path.read_text().splitlines():
        try:
            row = json.loads(line)
        except Exception:
            continue
        # LongMemEval shape: question, answer, sessions, qid, category
        sessions = row.get("sessions") or row.get("haystack_sessions") or []
        units = []
        for si, sess in enumerate(sessions):
            for ti, t in enumerate(sess if isinstance(sess, list) else sess.get("turns", [])):
                txt = t.get("content") or t.get("text") or ""
                if not txt:
                    continue
                units.append({
                    "uid": f"{row.get('qid','q')}::S{si}T{ti}",
                    "text": f"{t.get('role','?')}: {txt}",
                    "metadata": {"session": si, "turn": ti, "speaker": t.get("role", "?")},
                })
        out.append({
            "question": row["question"],
            "gold_answer": str(row.get("answer", "")),
            "category": str(row.get("question_type") or row.get("category") or "single-hop"),
            "data_source": "longmemeval",
            "sample_id": str(row.get("question_id") or row.get("qid", "?")),
            "qid": str(row.get("question_id") or row.get("qid", "?")),
            "conversation_units": units,
        })
    return out


def _conv_qas_to_memory_data(
    conv_qas: list[tuple[Conversation, list[QA]]],
    data_source: str,
) -> list[MemoryDatum]:
    out: list[MemoryDatum] = []
    for conv, qas in conv_qas:
        units = _conv_to_serializable_units(conv)
        for qa in qas:
            out.append({
                "question": qa.question,
                "gold_answer": qa.answer,
                "category": qa.category_name,
                "data_source": data_source,
                "sample_id": conv.sample_id,
                "qid": qa.qid,
                "conversation_units": units,
            })
    return out


def _balanced_prefix(
    conv_qas: list[tuple[Conversation, list[QA]]],
    per_category: int,
) -> list[tuple[Conversation, list[QA]]]:
    if per_category <= 0:
        return conv_qas
    counts: dict[str, int] = {}
    kept: list[tuple[Conversation, list[QA]]] = []
    for conv, qas in conv_qas:
        category = qas[0].category_name if qas else "unknown"
        if counts.get(category, 0) >= per_category:
            continue
        kept.append((conv, qas))
        counts[category] = counts.get(category, 0) + 1
    return kept


def longmemeval_to_memory_data(
    variant: str = "longmemeval_s",
    n_rows: int | None = None,
    train_frac: float = 0.8,
    seed: int = 0,
    per_category: int = 0,
    download: bool = True,
) -> tuple[list[MemoryDatum], list[MemoryDatum]]:
    """Split LongMemEval at row level for read-policy RL training.

    Each LongMemEval row has one question and its own haystack sessions. We treat
    one row as one environment knowledge base, then split rows into train/eval.
    `per_category` is the training/eval equivalent of the matrix harness' balanced
    sampling knob.
    """
    load_n = None if per_category > 0 else n_rows
    rows = load_longmemeval(variant=variant, n_convs=load_n, download=download)
    rows = _balanced_prefix(rows, per_category=per_category)
    if n_rows is not None and n_rows > 0:
        rows = rows[:n_rows]
    rng = random.Random(seed)
    rng.shuffle(rows)
    n_train = max(1, int(len(rows) * train_frac)) if rows else 0
    train_rows = rows[:n_train]
    eval_rows = rows[n_train:]
    return (
        _conv_qas_to_memory_data(train_rows, data_source=variant),
        _conv_qas_to_memory_data(eval_rows, data_source=variant),
    )


def mix_sources(
    sources: dict[str, list[MemoryDatum]],
    weights: dict[str, float] | None = None,
    seed: int = 0,
) -> list[MemoryDatum]:
    """Interleave multiple sources with optional weighting.
    weights default = uniform per dataset (oversample smaller ones)."""
    rng = random.Random(seed)
    pools = {k: list(v) for k, v in sources.items() if v}
    for p in pools.values():
        rng.shuffle(p)
    if weights is None:
        weights = {k: 1.0 for k in pools}
    out: list[MemoryDatum] = []
    cursors = {k: 0 for k in pools}
    while any(cursors[k] < len(pools[k]) for k in pools):
        # weighted sampling
        candidates = [k for k in pools if cursors[k] < len(pools[k])]
        if not candidates:
            break
        w = [weights.get(k, 1.0) for k in candidates]
        chosen = rng.choices(candidates, weights=w, k=1)[0]
        out.append(pools[chosen][cursors[chosen]])
        cursors[chosen] += 1
    return out


def smoke():
    """Verify LoCoMo → MemoryDatum runs end to end."""
    train, eval_set = locomo_to_memory_data(n_convs=2, train_frac=0.5, seed=0)
    print(f"train={len(train)} eval={len(eval_set)}")
    print(f"first train Q: {train[0]['question'][:80]}")
    print(f"  gold: {train[0]['gold_answer']}")
    print(f"  category: {train[0]['category']}")
    print(f"  conv units: {len(train[0]['conversation_units'])}")
    return train, eval_set


if __name__ == "__main__":
    smoke()
