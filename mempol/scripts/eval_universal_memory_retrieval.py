"""LoCoMo evidence-retrieval eval for the universal memory/RL substrate.

This does not train a model. It measures whether the current substrate exposes
the evidence a write/read policy would need:

  1. raw_span_recall: question -> raw spans
  2. turn_memory_recall: question -> one MemoryState per turn

If these numbers are poor, full RL is bottlenecked by retrieval before the
policy can learn useful write/read behavior.
"""
from __future__ import annotations

import argparse
import json
import tempfile
from collections import defaultdict
from pathlib import Path
from statistics import mean

from mempol import config
from mempol.core.schema import Artifact, MemoryState, Span
from mempol.core.store import SQLiteMemoryStore, now_iso, stable_id
from mempol.data.locomo import load as load_locomo


def _turn_ids(sample_id: str, dia_id: str) -> tuple[str, str]:
    uid = f"{sample_id}_{dia_id.replace(':', '_')}"
    return f"locomo_artifact_{uid}", f"locomo_span_{uid}"


def _build_turn_store(conv) -> SQLiteMemoryStore:
    tmp = tempfile.NamedTemporaryFile(prefix="mempol_universal_retrieval_", suffix=".sqlite", delete=False)
    tmp.close()
    store = SQLiteMemoryStore(Path(tmp.name))
    for turn in conv.turns:
        aid, sid = _turn_ids(conv.sample_id, turn.dia_id)
        text = f"{turn.speaker}: {turn.text}"
        artifact = Artifact(
            id=aid,
            source="locomo",
            kind="conversation_turn",
            title=f"{conv.sample_id} {turn.dia_id}",
            content=text,
            created_at=turn.session_date,
            metadata={
                "sample_id": conv.sample_id,
                "dia_id": turn.dia_id,
                "session": turn.session,
                "speaker": turn.speaker,
                "session_date": turn.session_date,
            },
        )
        span = Span(
            id=sid,
            artifact_id=aid,
            text=text,
            locator=turn.dia_id,
            metadata={"dia_id": turn.dia_id, "speaker": turn.speaker},
        )
        memory = MemoryState(
            id=stable_id("turn_memory", sid),
            content=text,
            source_span_ids=[sid],
            created_at=now_iso(),
            updated_at=now_iso(),
            metadata={"adapter": "turn_memory_baseline", "dia_id": turn.dia_id},
        )
        store.upsert_artifact(artifact)
        store.upsert_span(span)
        store.upsert_memory_state(memory)
    store.commit()
    return store


def _recall(found: set[str], gold: list[str]) -> float:
    if not gold:
        return 0.0
    return len(found.intersection(gold)) / len(set(gold))


def run(n_convs: int, k: int, run_name: str) -> dict:
    rows = []
    by_category: dict[str, list[dict]] = defaultdict(list)
    for conv, qas in load_locomo(n_convs=n_convs):
        store = _build_turn_store(conv)
        span_to_dia = {
            f"locomo_span_{conv.sample_id}_{turn.dia_id.replace(':', '_')}": turn.dia_id
            for turn in conv.turns
        }
        for qa in qas:
            raw_hits = [
                h for h in store.retrieve(qa.question, k=k, include_spans=True)
                if h["kind"] == "span"
            ][:k]
            raw_found = {h.get("locator", "") for h in raw_hits}

            mem_hits = [
                h for h in store.retrieve(qa.question, k=k, include_spans=False)
                if h["kind"] == "memory_state"
            ][:k]
            mem_found = {
                span_to_dia.get(sid, "")
                for h in mem_hits
                for sid in h.get("source_span_ids", [])
            }
            row = {
                "sample_id": conv.sample_id,
                "qid": qa.qid,
                "category": qa.category_name,
                "question": qa.question,
                "gold_answer": qa.answer,
                "evidence": qa.evidence,
                "raw_recall": _recall(raw_found, qa.evidence),
                "raw_any": float(bool(raw_found.intersection(qa.evidence))),
                "turn_memory_recall": _recall(mem_found, qa.evidence),
                "turn_memory_any": float(bool(mem_found.intersection(qa.evidence))),
                "raw_hits": [
                    {"dia_id": h.get("locator"), "score": h.get("score"), "text": h.get("text", "")[:240]}
                    for h in raw_hits[:5]
                ],
                "memory_hits": [
                    {"source_span_ids": h.get("source_span_ids", []), "score": h.get("score"), "content": h.get("text", "")[:240]}
                    for h in mem_hits[:5]
                ],
            }
            rows.append(row)
            by_category[qa.category_name].append(row)
        store.close()

    summary = {
        "run_name": run_name,
        "dataset": "locomo",
        "n_convs": n_convs,
        "k": k,
        "qa_count": len(rows),
        "raw_recall_mean": mean([r["raw_recall"] for r in rows]) if rows else 0.0,
        "raw_any_mean": mean([r["raw_any"] for r in rows]) if rows else 0.0,
        "turn_memory_recall_mean": mean([r["turn_memory_recall"] for r in rows]) if rows else 0.0,
        "turn_memory_any_mean": mean([r["turn_memory_any"] for r in rows]) if rows else 0.0,
        "by_category": {
            cat: {
                "qa_count": len(cat_rows),
                "raw_recall_mean": mean([r["raw_recall"] for r in cat_rows]),
                "raw_any_mean": mean([r["raw_any"] for r in cat_rows]),
                "turn_memory_recall_mean": mean([r["turn_memory_recall"] for r in cat_rows]),
                "turn_memory_any_mean": mean([r["turn_memory_any"] for r in cat_rows]),
            }
            for cat, cat_rows in sorted(by_category.items())
        },
        "rows": rows,
    }
    out_dir = config.RESULTS_DIR / run_name
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "universal_memory_retrieval_eval.json").write_text(json.dumps(summary, indent=2, ensure_ascii=False))
    return summary


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--n-convs", type=int, default=1)
    ap.add_argument("--k", type=int, default=8)
    ap.add_argument("--run-name", default="universal_retrieval_eval")
    args = ap.parse_args()
    summary = run(n_convs=args.n_convs, k=args.k, run_name=args.run_name)
    compact = {k: v for k, v in summary.items() if k != "rows"}
    print(json.dumps(compact, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
