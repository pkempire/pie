"""Inspect one LongMemEval row and optional matrix outputs.

This is intentionally boring and explicit: it prints the real benchmark
question, raw sessions/turns, gold answer, and any strategy outputs already
written by `mempol.scripts.longmemeval_matrix`.
"""
from __future__ import annotations

import argparse
import json
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any

REPO = Path(__file__).resolve().parents[2]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

from mempol.data.longmemeval import load as load_lme
from mempol.scripts.longmemeval_matrix import _canonical_cell_name, _cell_label


def _compact(text: str, n: int = 500) -> str:
    text = (text or "").replace("\n", " ").strip()
    return text if len(text) <= n else text[:n] + " ..."


def _load_rows(results_dir: Path) -> dict[str, dict[str, dict[str, Any]]]:
    rows_path = results_dir / "rows.jsonl"
    out: dict[str, dict[str, dict[str, Any]]] = defaultdict(dict)
    if not rows_path.exists():
        return out
    for line in rows_path.read_text().splitlines():
        if not line.strip():
            continue
        row = json.loads(line)
        out[row["question_id"]][_canonical_cell_name(row["cell"])] = row
    return out


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--variant", default="longmemeval_s", choices=["longmemeval_s", "longmemeval_oracle", "longmemeval_m"])
    ap.add_argument("--question-id", default=None, help="LongMemEval question_id/sample_id. Defaults to first row.")
    ap.add_argument("--results", default=None, help="Optional matrix results directory containing rows.jsonl.")
    ap.add_argument("--max-sessions", type=int, default=2)
    ap.add_argument("--max-turns-per-session", type=int, default=8)
    ap.add_argument("--search", default=None, help="Optional substring filter for raw turns.")
    args = ap.parse_args()

    rows = load_lme(variant=args.variant, n_convs=None, download=False)
    selected = None
    for conv, qas in rows:
        if args.question_id is None or conv.sample_id == args.question_id or qas[0].qid == args.question_id:
            selected = (conv, qas[0])
            break
    if selected is None:
        raise SystemExit(f"question id not found: {args.question_id}")

    conv, qa = selected
    raw_chars = sum(len(t.text or "") for t in conv.turns)
    print(f"question_id: {conv.sample_id}")
    print(f"category:    {qa.category_name}")
    print(f"question:    {qa.question}")
    print(f"gold:        {qa.answer}")
    print(f"sessions:    {len({t.session for t in conv.turns})}")
    print(f"turns:       {len(conv.turns)}")
    print(f"chars:       {raw_chars:,} (~{raw_chars // 4:,} tokens)")

    if args.results:
        by_q = _load_rows(Path(args.results))
        strategy_rows = by_q.get(conv.sample_id, {})
        if strategy_rows:
            print("\n=== Strategy Outputs ===")
            for cell, row in sorted(strategy_rows.items()):
                print(f"\n[{_cell_label(cell)} / {cell}] score={row.get('score')} ctx={row.get('context_chars')} chars")
                print(f"answer: {_compact(row.get('answer', ''), 700)}")
                trace = row.get("trace") or {}
                steps = trace.get("steps") or []
                if steps:
                    print("steps:")
                    for step in steps:
                        print(f"  - {step.get('op')}: {step.get('args') or {}} -> {_compact(str(step.get('obs_summary', '')), 180)}")
                retrieved = trace.get("retrieved") or []
                if retrieved:
                    print("top retrieved:")
                    for hit in retrieved[:5]:
                        md = hit.get("metadata") or {}
                        print(
                            f"  - {hit.get('uid')} {md.get('session_date') or ''} "
                            f"{md.get('speaker') or md.get('name') or ''}: {_compact(hit.get('text', ''), 220)}"
                        )

    print("\n=== Raw Sessions / Turns ===")
    search = args.search.lower() if args.search else None
    sessions: dict[Any, list[Any]] = defaultdict(list)
    for t in conv.turns:
        if search and search not in (t.text or "").lower():
            continue
        sessions[t.session].append(t)
    for si, (session, turns) in enumerate(sessions.items()):
        if si >= args.max_sessions:
            break
        print(f"\nSession {session}")
        for t in turns[: args.max_turns_per_session]:
            print(f"  {t.dia_id} | {t.session_date} | {t.speaker}: {_compact(t.text, 300)}")


if __name__ == "__main__":
    main()
