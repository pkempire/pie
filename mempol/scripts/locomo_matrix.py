"""Canonical LoCoMo comparison matrix.

One command, one output directory, many memory variants. This is meant to stop
the repo from accumulating disconnected one-off evals.

Each row is:
  conversation × question × cell -> answer, score, retrieved evidence, steps

Outputs:
  rows.jsonl        machine-readable per-answer trace
  summary.json      accuracy by cell / category / conversation
  side_by_side.md   human-readable question-by-question comparison

The expensive cells are opt-in. Start with flat/pie/rlm on 3 convs, then add
mastra_inspired and hand/gepa consolidation when you want to spend calls.
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from collections import defaultdict
from dataclasses import asdict
from pathlib import Path
from typing import Callable

from pie.core.world_model import WorldModel

from mempol import config
from mempol.backends.base import Backend, Hit, Unit
from mempol.backends.flat import FlatBackend
from mempol.backends.mastra import MastraBackend
from mempol.backends.pie_kg import PIEBackend
from mempol.data.locomo import Conversation, QA, load
from mempol.eval.judge import judge
from mempol.eval.metrics import Result, summarise
from mempol.eval.runner import conv_to_units, evidence_recall
from mempol.policies.base import ReadPolicy, Trace
from mempol.policies.rlm_temporal import TemporalRLMPolicy
from mempol.policies.v0_naive import NaivePolicy
from mempol.policies.v1_heuristic import HeuristicPolicy


REPO = Path(__file__).resolve().parents[2]
SCRIPT_DIR = REPO / "scripts"
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))


def _compact(text: str, n: int = 1200) -> str:
    text = (text or "").replace("\n", " ").strip()
    return text if len(text) <= n else text[:n] + " ..."


def _hit_payload(hit: Hit) -> dict:
    md = dict(hit.unit.metadata or {})
    return {
        "uid": hit.unit.uid,
        "source": hit.source,
        "score": hit.score,
        "text": _compact(hit.unit.text, 700),
        "metadata": {
            "dia_id": md.get("dia_id"),
            "session": md.get("session"),
            "session_date": md.get("session_date"),
            "speaker": md.get("speaker"),
            "name": md.get("name"),
            "type": md.get("type"),
            "n_transitions": md.get("n_transitions"),
        },
    }


def _trace_payload(trace: Trace) -> dict:
    return {
        "policy": trace.policy,
        "backend": trace.backend,
        "answer": trace.answer,
        "n_steps": len(trace.steps),
        "n_retrievals": trace.n_retrievals,
        "steps": [asdict(s) for s in trace.steps],
        "retrieved": [_hit_payload(h) for h in trace.final_hits],
    }


class Cell:
    def __init__(
        self,
        name: str,
        build_backend: Callable[[Conversation, argparse.Namespace], Backend],
        build_policy: Callable[[argparse.Namespace], ReadPolicy],
        description: str,
        expensive: bool = False,
    ) -> None:
        self.name = name
        self.build_backend = build_backend
        self.build_policy = build_policy
        self.description = description
        self.expensive = expensive


def _flat_backend(conv: Conversation, _args: argparse.Namespace) -> Backend:
    b = FlatBackend()
    b.ingest(conv_to_units(conv))
    return b


def _pie_cached_backend(conv: Conversation, _args: argparse.Namespace) -> Backend:
    path = REPO / "benchmarks" / "locomo" / "cache" / f"{conv.sample_id}_wm.json"
    if not path.exists():
        raise FileNotFoundError(f"missing cached PIE world model: {path}")
    return PIEBackend(world_model=WorldModel(persist_path=str(path)))


def _mastra_inspired_backend(conv: Conversation, args: argparse.Namespace) -> Backend:
    b = MastraBackend(
        observer_token_threshold=args.mastra_observer_threshold,
        reflector_token_threshold=args.mastra_reflector_threshold,
        keep_recent_n=args.mastra_recent_turns,
    )
    b.ingest(conv_to_units(conv))
    return b


def _consolidated_flat_backend(conv: Conversation, args: argparse.Namespace, prompt_path: Path) -> Backend:
    from compare_pie_vs_gepa import consolidate_chunk, chunk_turns, entry_to_unit

    prompt = prompt_path.read_text()
    chunks = chunk_turns(conv.turns, args.consolidator_chunk_size)
    if args.max_chunks_per_conv:
        chunks = chunks[: args.max_chunks_per_conv]
    units: list[Unit] = []
    idx = 0
    for ci, ch in enumerate(chunks, start=1):
        print(f"    consolidating {conv.sample_id} chunk {ci}/{len(chunks)}", flush=True)
        for entry in consolidate_chunk(ch, prompt, args.consolidator_model):
            units.append(entry_to_unit(entry, idx))
            idx += 1
    if not units:
        raise RuntimeError(f"{prompt_path.name} produced zero consolidated units for {conv.sample_id}")
    b = FlatBackend()
    b.ingest(units)
    return b


def _hand_flat_backend(conv: Conversation, args: argparse.Namespace) -> Backend:
    return _consolidated_flat_backend(
        conv,
        args,
        REPO / "mempol" / "results" / "gepa_consolidator" / "prompt_original.txt",
    )


def _gepa_flat_backend(conv: Conversation, args: argparse.Namespace) -> Backend:
    return _consolidated_flat_backend(
        conv,
        args,
        REPO / "mempol" / "results" / "gepa_consolidator" / "prompt_optimized.txt",
    )


def _v0(_args: argparse.Namespace) -> ReadPolicy:
    return NaivePolicy(k=10)


def _v1(_args: argparse.Namespace) -> ReadPolicy:
    return HeuristicPolicy(do_reformulate=False, do_route=False, do_expand=True)


def _v1_expand(_args: argparse.Namespace) -> ReadPolicy:
    # Route is enabled because HeuristicPolicy only expands when the router
    # predicts the query needs expansion. This costs one extra LLM call per Q.
    return HeuristicPolicy(do_reformulate=False, do_route=True, do_expand=True)


def _rlm(args: argparse.Namespace) -> ReadPolicy:
    return TemporalRLMPolicy(
        first_k=args.rlm_first_k,
        final_k=args.rlm_final_k,
        expand_seed_k=args.rlm_expand_seed_k,
        force_timeline=args.rlm_force_timeline,
    )


CELLS: dict[str, Cell] = {
    "flat_v0": Cell(
        "flat_v0", _flat_backend, _v0,
        "Raw turns in FlatBackend; single hybrid retrieve; answer.",
    ),
    "flat_v1": Cell(
        "flat_v1", _flat_backend, _v1,
        "Raw turns in FlatBackend; hybrid retrieve + dense rerank; route disabled, so no expansion.",
    ),
    "flat_v1_expand": Cell(
        "flat_v1_expand", _flat_backend, _v1_expand,
        "Raw turns in FlatBackend; route may trigger adjacent-turn expansion; costs one router LLM call per Q.",
    ),
    "flat_rlm_temporal": Cell(
        "flat_rlm_temporal", _flat_backend, _rlm,
        "Raw turns in FlatBackend; broad retrieve + adjacent turns + LLM timeline reconstruction.",
    ),
    "pie_cached_v1": Cell(
        "pie_cached_v1", _pie_cached_backend, _v1,
        "Cached PIE temporal KG; hybrid entity retrieval + dense rerank; route disabled, so no KG expansion.",
    ),
    "pie_cached_v1_expand": Cell(
        "pie_cached_v1_expand", _pie_cached_backend, _v1_expand,
        "Cached PIE temporal KG; route may trigger KG neighbor expansion; costs one router LLM call per Q.",
    ),
    "mastra_inspired_v1": Cell(
        "mastra_inspired_v1", _mastra_inspired_backend, _v1,
        "Python Mastra-inspired Observer/Reflector notes; not official Mastra.",
        expensive=True,
    ),
    "hand_flat_v1": Cell(
        "hand_flat_v1", _hand_flat_backend, _v1,
        "Hand consolidator prompt -> FlatBackend; then v1 reader.",
        expensive=True,
    ),
    "gepa_flat_v1": Cell(
        "gepa_flat_v1", _gepa_flat_backend, _v1,
        "GEPA-optimized consolidator prompt -> FlatBackend; then v1 reader.",
        expensive=True,
    ),
}


def _selected_convs(args: argparse.Namespace) -> list[tuple[Conversation, list[QA]]]:
    convs = load()
    by_id = {c.sample_id: (c, qas) for c, qas in convs}
    if args.conv_ids:
        out = []
        for cid in [x.strip() for x in args.conv_ids.split(",") if x.strip()]:
            if cid not in by_id:
                raise SystemExit(f"unknown conv id {cid}; available={list(by_id)}")
            out.append(by_id[cid])
        return out
    return convs[: args.n_convs]


def _selected_qas(qas: list[QA], args: argparse.Namespace) -> list[QA]:
    out = [q for q in qas if q.answer]
    if args.categories:
        cats = {x.strip() for x in args.categories.split(",") if x.strip()}
        out = [q for q in out if q.category_name in cats]
    if args.max_qs_per_conv:
        out = out[: args.max_qs_per_conv]
    return out


def _load_done(rows_path: Path) -> dict[tuple[str, str, str], dict]:
    done = {}
    if not rows_path.exists():
        return done
    for line in rows_path.read_text().splitlines():
        try:
            row = json.loads(line)
            done[(row["cell"], row["conv_id"], row["qid"])] = row
        except Exception:
            continue
    return done


def run(args: argparse.Namespace) -> dict:
    out_dir = args.out_dir
    out_dir.mkdir(parents=True, exist_ok=True)
    rows_path = out_dir / "rows.jsonl"
    done = _load_done(rows_path)

    if args.summarize_only:
        rows = list(done.values())
        summary = _summarise_rows(rows, args)
        (out_dir / "summary.json").write_text(json.dumps(summary, indent=2))
        _write_side_by_side(rows, out_dir / "side_by_side.md", max_answer_chars=args.side_by_side_answer_chars)
        print(json.dumps(summary, indent=2))
        print(f"Wrote {out_dir / 'summary.json'}")
        print(f"Wrote {out_dir / 'side_by_side.md'}")
        return summary

    cells = [c.strip() for c in args.cells.split(",") if c.strip()]
    unknown = [c for c in cells if c not in CELLS]
    if unknown:
        raise SystemExit(f"unknown cells: {unknown}; choices={sorted(CELLS)}")
    if args.no_expensive:
        cells = [c for c in cells if not CELLS[c].expensive]

    convs = _selected_convs(args)
    print(f"[locomo_matrix] convs={[c.sample_id for c, _ in convs]}")
    print(f"[locomo_matrix] cells={cells}")
    print(f"[locomo_matrix] out={out_dir}")

    with rows_path.open("a", buffering=1) as f:
        for cell_name in cells:
            cell = CELLS[cell_name]
            print(f"\n[{cell_name}] {cell.description}", flush=True)
            policy = cell.build_policy(args)
            for ci, (conv, qas) in enumerate(convs, start=1):
                qas_run = _selected_qas(qas, args)
                remaining = [
                    qa for qa in qas_run
                    if (cell_name, conv.sample_id, qa.qid) not in done
                ]
                if not remaining:
                    print(f"  {conv.sample_id}: already complete ({len(qas_run)} qs)", flush=True)
                    continue
                print(
                    f"  building backend for {conv.sample_id} "
                    f"({ci}/{len(convs)}), {len(remaining)}/{len(qas_run)} remaining qs",
                    flush=True,
                )
                t_build = time.time()
                try:
                    backend = cell.build_backend(conv, args)
                except Exception as e:
                    print(f"  backend build failed: {e}", flush=True)
                    for qa in remaining:
                        row = {
                            "cell": cell_name,
                            "conv_id": conv.sample_id,
                            "qid": qa.qid,
                            "question": qa.question,
                            "gold": qa.answer,
                            "category": qa.category_name,
                            "score": 0.0,
                            "answer": "",
                            "error": f"backend_build_failed: {e}",
                        }
                        f.write(json.dumps(row, ensure_ascii=False) + "\n")
                        done[(cell_name, conv.sample_id, qa.qid)] = row
                    continue
                build_secs = time.time() - t_build

                scores = []
                for qi, qa in enumerate(remaining, start=1):
                    key = (cell_name, conv.sample_id, qa.qid)
                    try:
                        trace = policy.run(qa.question, backend)
                        score, reason = judge(qa.question, qa.answer, trace.answer)
                        rec = evidence_recall([h.unit.uid for h in trace.final_hits], qa.evidence)
                        row = {
                            "cell": cell_name,
                            "description": cell.description,
                            "conv_id": conv.sample_id,
                            "qid": qa.qid,
                            "question": qa.question,
                            "gold": qa.answer,
                            "category": qa.category_name,
                            "score": float(score),
                            "judge_reason": reason,
                            "evidence": qa.evidence,
                            "evidence_recall": rec,
                            "backend_build_secs": round(build_secs, 3),
                            **_trace_payload(trace),
                        }
                    except Exception as e:
                        row = {
                            "cell": cell_name,
                            "description": cell.description,
                            "conv_id": conv.sample_id,
                            "qid": qa.qid,
                            "question": qa.question,
                            "gold": qa.answer,
                            "category": qa.category_name,
                            "score": 0.0,
                            "answer": "",
                            "error": str(e),
                        }
                    f.write(json.dumps(row, ensure_ascii=False) + "\n")
                    done[key] = row
                    scores.append(float(row.get("score", 0.0)))
                    if qi % args.print_every == 0 or qi == len(remaining):
                        print(
                            f"    {conv.sample_id} q {qi}/{len(remaining)} "
                            f"run_acc={sum(scores) / max(1, len(scores)):.3f}",
                            flush=True,
                        )

    rows = list(_load_done(rows_path).values())
    summary = _summarise_rows(rows, args)
    (out_dir / "summary.json").write_text(json.dumps(summary, indent=2))
    _write_side_by_side(rows, out_dir / "side_by_side.md", max_answer_chars=args.side_by_side_answer_chars)
    print("\n=== SUMMARY ===")
    print(json.dumps(summary, indent=2))
    print(f"\nWrote {rows_path}")
    print(f"Wrote {out_dir / 'summary.json'}")
    print(f"Wrote {out_dir / 'side_by_side.md'}")
    return summary


def _summarise_rows(rows: list[dict], args: argparse.Namespace) -> dict:
    by_cell: dict[str, list[dict]] = defaultdict(list)
    by_cell_cat: dict[str, dict[str, list[dict]]] = defaultdict(lambda: defaultdict(list))
    by_cell_conv: dict[str, dict[str, list[dict]]] = defaultdict(lambda: defaultdict(list))
    for r in rows:
        by_cell[r["cell"]].append(r)
        by_cell_cat[r["cell"]][r.get("category", "unknown")].append(r)
        by_cell_conv[r["cell"]][r.get("conv_id", "unknown")].append(r)

    out = {
        "cells_requested": [x.strip() for x in args.cells.split(",") if x.strip()],
        "n_rows": len(rows),
        "by_cell": {},
    }
    for cell, rs in sorted(by_cell.items()):
        out["by_cell"][cell] = {
            "n": len(rs),
            "acc": sum(float(r.get("score", 0.0)) for r in rs) / max(1, len(rs)),
            "avg_evidence_recall": sum(float(r.get("evidence_recall", 0.0) or 0.0) for r in rs) / max(1, len(rs)),
            "errors": sum(1 for r in rs if r.get("error")),
            "by_category": {
                cat: {
                    "n": len(crs),
                    "acc": sum(float(r.get("score", 0.0)) for r in crs) / max(1, len(crs)),
                }
                for cat, crs in sorted(by_cell_cat[cell].items())
            },
            "by_conv": {
                conv: {
                    "n": len(crs),
                    "acc": sum(float(r.get("score", 0.0)) for r in crs) / max(1, len(crs)),
                }
                for conv, crs in sorted(by_cell_conv[cell].items())
            },
        }
    return out


def _write_side_by_side(rows: list[dict], path: Path, max_answer_chars: int = 900) -> None:
    by_q: dict[tuple[str, str], list[dict]] = defaultdict(list)
    for r in rows:
        by_q[(r.get("conv_id", ""), r.get("qid", ""))].append(r)
    lines = ["# LoCoMo Side-By-Side", ""]
    for (_conv, _qid), rs in sorted(by_q.items()):
        rs = sorted(rs, key=lambda r: r["cell"])
        first = rs[0]
        lines.append(f"## {first['conv_id']} / {first['qid']} / {first.get('category','')}")
        lines.append(f"**Q:** {first.get('question','')}")
        lines.append(f"**Gold:** {first.get('gold','')}")
        lines.append("")
        for r in rs:
            score = r.get("score", 0.0)
            err = f" ERROR: {r['error']}" if r.get("error") else ""
            lines.append(f"### {r['cell']} — score={score}{err}")
            lines.append(_compact(r.get("answer", ""), max_answer_chars) or "(empty)")
            retrieved = r.get("retrieved", [])[:4]
            if retrieved:
                lines.append("")
                lines.append("Top retrieved:")
                for h in retrieved:
                    md = h.get("metadata", {})
                    lines.append(
                        f"- `{h.get('uid')}` {md.get('session_date') or ''} "
                        f"{md.get('speaker') or md.get('name') or ''}: "
                        f"{_compact(h.get('text',''), 180)}"
                    )
            lines.append("")
        lines.append("---")
        lines.append("")
    path.write_text("\n".join(lines))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out-dir", type=Path, default=config.RESULTS_DIR / "locomo_matrix")
    ap.add_argument("--cells", default="flat_v0,flat_v1,flat_rlm_temporal,pie_cached_v1")
    ap.add_argument("--n-convs", type=int, default=3)
    ap.add_argument("--conv-ids", default="", help="comma-separated explicit conv ids, e.g. conv-26,conv-30")
    ap.add_argument("--max-qs-per-conv", type=int, default=0, help="0 = all")
    ap.add_argument("--categories", default="", help="comma subset: single-hop,multi-hop,temporal,open-domain")
    ap.add_argument("--no-expensive", action="store_true", help="drop cells marked expensive")
    ap.add_argument("--print-every", type=int, default=10)
    ap.add_argument("--side-by-side-answer-chars", type=int, default=900)
    ap.add_argument("--summarize-only", action="store_true",
                    help="read existing rows.jsonl and rebuild summary/side_by_side without running new evals")
    # RLM temporal reader knobs.
    ap.add_argument("--rlm-first-k", type=int, default=24)
    ap.add_argument("--rlm-final-k", type=int, default=12)
    ap.add_argument("--rlm-expand-seed-k", type=int, default=8)
    ap.add_argument("--rlm-force-timeline", action="store_true")
    # Mastra-inspired knobs.
    ap.add_argument("--mastra-observer-threshold", type=int, default=3000)
    ap.add_argument("--mastra-reflector-threshold", type=int, default=8000)
    ap.add_argument("--mastra-recent-turns", type=int, default=20)
    # Consolidator knobs.
    ap.add_argument("--consolidator-model", default="gpt-5-mini")
    ap.add_argument("--consolidator-chunk-size", type=int, default=30)
    ap.add_argument("--max-chunks-per-conv", type=int, default=0, help="0 = all chunks")
    args = ap.parse_args()
    run(args)


if __name__ == "__main__":
    main()
