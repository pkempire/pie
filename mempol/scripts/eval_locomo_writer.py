"""Chunked LoCoMo writer eval.

This is the first real number-producing path for the write side:

  chronological conversation -> chunked memory writes -> completed memory
  -> LoCoMo QA over that memory -> judge score / category breakdown

It is intentionally separate from `simulate_on_locomo.py`, which is only a
live write dashboard. This script answers the research question:

  Did the memory we wrote make future answers better under a storage/retrieval
  budget?

Usage:
  python -m mempol.scripts.eval_locomo_writer \\
    --n-convs 1 --chunk-size 12 --max-qs-per-conv 0 \\
    --run-name locomo_writer_chunked_v0
"""
from __future__ import annotations

import argparse
import json
import time
from dataclasses import asdict
from pathlib import Path
from typing import Iterable

from mempol import config
from mempol.backends.pie_kg import PIEBackend
from mempol.data.locomo import Conversation, QA, Turn, load, parse_locomo_date
from mempol.eval.judge import judge
from mempol.eval.metrics import Result, summarise
from mempol.policies.base import ReadPolicy
from mempol.policies.v0_naive import NaivePolicy
from mempol.policies.v1_heuristic import HeuristicPolicy
from mempol.policies.v1_write import HeuristicWritePolicy
from mempol.recipes.memory_rl.write_tools import WriteTool
from mempol.scripts.simulate_on_real_export import dump_world_model_md


def _chunks(turns: list[Turn], chunk_size: int, by_session: bool) -> Iterable[list[Turn]]:
    if by_session:
        current: list[Turn] = []
        current_session = None
        for t in turns:
            if current and t.session != current_session:
                yield current
                current = []
            current.append(t)
            current_session = t.session
        if current:
            yield current
        return
    for i in range(0, len(turns), chunk_size):
        yield turns[i:i + chunk_size]


def _chunk_text(chunk: list[Turn]) -> str:
    lines = []
    for t in chunk:
        lines.append(f"{t.dia_id} [{t.session_date}] {t.speaker}: {t.text}")
    return "\n".join(lines)


def _chunk_id(chunk: list[Turn]) -> str:
    return f"{chunk[0].dia_id}..{chunk[-1].dia_id}"


def _chunk_timestamp(chunk: list[Turn]) -> float:
    return parse_locomo_date(chunk[0].session_date) or float(chunk[0].session)


def _answer_qas(
    backend: PIEBackend,
    qas: list[QA],
    out_path: Path,
    policy: ReadPolicy,
    judge_mode: str = "llm",
    progress_every: int = 10,
) -> list[Result]:
    results: list[Result] = []
    with out_path.open("w") as f:
        for i, qa in enumerate(qas):
            trace = policy.run(qa.question, backend)
            trace.qid = qa.qid
            if judge_mode == "llm":
                score, reason = judge(qa.question, qa.answer, trace.answer)
            elif judge_mode == "none":
                score, reason = 0.0, "judge_disabled"
            else:
                raise ValueError(f"unknown judge_mode: {judge_mode}")
            result = Result(
                qid=qa.qid,
                category=qa.category,
                category_name=qa.category_name,
                score=score,
                n_retrievals=trace.n_retrievals,
                n_steps=len(trace.steps),
                answer=trace.answer,
                gold=qa.answer,
                judge_reason=reason,
                evidence_recall=None,
            )
            results.append(result)
            f.write(json.dumps({
                "qid": qa.qid,
                "question": qa.question,
                "gold": qa.answer,
                "answer": trace.answer,
                "score": score,
                "category": qa.category,
                "category_name": qa.category_name,
                "judge_reason": reason,
                "steps": [asdict(s) for s in trace.steps],
                "retrieved": [
                    {
                        "uid": h.unit.uid,
                        "text": h.unit.text[:800],
                        "score": h.score,
                        "source": h.source,
                        "metadata": h.unit.metadata,
                    }
                    for h in trace.final_hits
                ],
            }, ensure_ascii=False) + "\n")
            f.flush()
            if progress_every > 0 and (i + 1) % progress_every == 0:
                running = sum(r.score for r in results) / max(1, len(results))
                print(f"    qa {i+1}/{len(qas)} running_score={running:.3f}", flush=True)
    return results


def _make_reader(name: str, k: int) -> ReadPolicy:
    if name == "v0_naive":
        return NaivePolicy(k=k)
    if name == "v1_fast":
        return HeuristicPolicy(first_k=max(k, 2 * k), final_k=k, do_reformulate=False, do_expand=False, do_route=False)
    if name == "v1_heuristic":
        return HeuristicPolicy(first_k=max(k, 2 * k), final_k=k, do_reformulate=True, do_expand=True, do_route=True)
    raise ValueError(f"unknown read_policy: {name}")


def run(
    n_convs: int,
    out_dir: Path,
    chunk_size: int = 12,
    by_session: bool = False,
    max_turns_per_conv: int = 0,
    max_qs_per_conv: int = 0,
    max_chunks_per_conv: int = 0,
    checkpoint_eval_every_chunks: int = 0,
    skip_qa: bool = False,
    read_policy_name: str = "v1_fast",
    reader_k: int = 8,
    judge_mode: str = "llm",
    progress_every: int = 10,
) -> dict:
    out_dir.mkdir(parents=True, exist_ok=True)
    decisions_path = out_dir / "write_decisions.jsonl"
    qa_path = out_dir / "qa_results.jsonl"
    checkpoint_path = out_dir / "checkpoint_results.jsonl"
    summary_path = out_dir / "summary.json"

    write_policy = HeuristicWritePolicy()
    read_policy = _make_reader(read_policy_name, reader_k)
    all_results: list[Result] = []
    t0 = time.time()

    with decisions_path.open("w") as f_decisions, checkpoint_path.open("w") as f_ckpt:
        for ci, (conv, qas) in enumerate(load(n_convs=n_convs)):
            turns = conv.turns[:max_turns_per_conv] if max_turns_per_conv > 0 else conv.turns
            qas_to_run = qas[:max_qs_per_conv] if max_qs_per_conv > 0 else qas
            backend = PIEBackend()
            write_tool = WriteTool(backend=backend)
            recent_context = ""

            processed_dia_ids: set[str] = set()
            for chunk_idx, chunk in enumerate(_chunks(turns, chunk_size, by_session)):
                if max_chunks_per_conv > 0 and chunk_idx >= max_chunks_per_conv:
                    break
                cid = _chunk_id(chunk)
                text = _chunk_text(chunk)
                print(f"[write] conv={conv.sample_id} chunk={chunk_idx} {cid} entities={len(backend.wm.entities)}", flush=True)
                decision = write_policy.step(
                    turn_text=text,
                    dia_id=cid,
                    timestamp=_chunk_timestamp(chunk),
                    backend=backend,
                    write_tool=write_tool,
                    observation_time_text=chunk[0].session_date,
                    recent_context_text=recent_context,
                )
                row = {
                    "conv_id": conv.sample_id,
                    "chunk_idx": chunk_idx,
                    "chunk_id": cid,
                    "dia_ids": [t.dia_id for t in chunk],
                    "session_date": chunk[0].session_date,
                    "text": text[:2000],
                    "lookup_matches": [
                        {
                            "uid": m["uid"][:8],
                            "name": m["name"],
                            "type": m["type"],
                            "match_score": m.get("match_score"),
                        }
                        for m in decision.lookup_matches[:8]
                    ],
                    "raw_ops": decision.raw_ops,
                    "applied_ops": decision.applied_ops,
                    "errors": decision.errors,
                    "n_entities_so_far": len(backend.wm.entities),
                    "n_transitions_so_far": sum(len(backend.wm.get_transitions(uid)) for uid in backend.wm.entities),
                    "n_relationships_so_far": sum(len(backend.wm.get_relationships(uid)) for uid in backend.wm.entities),
                    "write_tool_stats": write_tool.write_stats(),
                }
                f_decisions.write(json.dumps(row, ensure_ascii=False) + "\n")
                f_decisions.flush()

                recent_context = text[-4000:]
                processed_dia_ids.update(t.dia_id for t in chunk)

                if checkpoint_eval_every_chunks > 0 and (chunk_idx + 1) % checkpoint_eval_every_chunks == 0:
                    ckpt_qas = [
                        qa for qa in qas_to_run
                        if all(did in processed_dia_ids for did in qa.evidence)
                    ]
                    if ckpt_qas:
                        ckpt_results = []
                        for qa in ckpt_qas:
                            trace = read_policy.run(qa.question, backend)
                            if judge_mode == "llm":
                                score, reason = judge(qa.question, qa.answer, trace.answer)
                            else:
                                score, reason = 0.0, "judge_disabled"
                            ckpt_results.append(score)
                            f_ckpt.write(json.dumps({
                                "conv_id": conv.sample_id,
                                "chunk_idx": chunk_idx,
                                "qid": qa.qid,
                                "score": score,
                                "question": qa.question,
                                "answer": trace.answer,
                                "gold": qa.answer,
                                "judge_reason": reason,
                            }, ensure_ascii=False) + "\n")
                        f_ckpt.flush()

            if not skip_qa:
                conv_qa_path = out_dir / f"{conv.sample_id}_qa_results.jsonl"
                print(f"[qa] conv={conv.sample_id} qas={len(qas_to_run)} reader={read_policy.name} judge={judge_mode}", flush=True)
                conv_results = _answer_qas(
                    backend,
                    qas_to_run,
                    conv_qa_path,
                    read_policy,
                    judge_mode=judge_mode,
                    progress_every=progress_every,
                )
                all_results.extend(conv_results)
                with qa_path.open("a") as f_all:
                    for line in conv_qa_path.read_text().splitlines():
                        f_all.write(line + "\n")

            dump_world_model_md(backend, out_dir / f"{conv.sample_id}_world_model.md")
            backend.save(out_dir / f"{conv.sample_id}_world_model.json")

    summary = summarise(all_results)
    summary.update({
        "n_convs": n_convs,
        "chunk_size": chunk_size,
        "by_session": by_session,
        "max_turns_per_conv": max_turns_per_conv,
        "max_qs_per_conv": max_qs_per_conv,
        "max_chunks_per_conv": max_chunks_per_conv,
        "skip_qa": skip_qa,
        "read_policy": read_policy.name,
        "reader_k": reader_k,
        "judge_mode": judge_mode,
        "wall_time_s": round(time.time() - t0, 2),
        "decisions_path": str(decisions_path),
        "qa_results_path": str(qa_path),
    })
    summary_path.write_text(json.dumps(summary, indent=2))
    return summary


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--n-convs", type=int, default=1)
    ap.add_argument("--chunk-size", type=int, default=12)
    ap.add_argument("--by-session", action="store_true")
    ap.add_argument("--max-turns-per-conv", type=int, default=0, help="0 = all")
    ap.add_argument("--max-qs-per-conv", type=int, default=0, help="0 = all")
    ap.add_argument("--max-chunks-per-conv", type=int, default=0, help="0 = all")
    ap.add_argument("--checkpoint-eval-every-chunks", type=int, default=0)
    ap.add_argument("--skip-qa", action="store_true")
    ap.add_argument("--read-policy", choices=["v0_naive", "v1_fast", "v1_heuristic"], default="v1_fast")
    ap.add_argument("--reader-k", type=int, default=8)
    ap.add_argument("--judge-mode", choices=["llm", "none"], default="llm")
    ap.add_argument("--progress-every", type=int, default=10)
    ap.add_argument("--run-name", default="locomo_writer_chunked")
    ap.add_argument("--out-dir", type=Path, default=None)
    args = ap.parse_args()

    out_dir = args.out_dir or (config.RESULTS_DIR / args.run_name)
    summary = run(
        n_convs=args.n_convs,
        out_dir=out_dir,
        chunk_size=args.chunk_size,
        by_session=args.by_session,
        max_turns_per_conv=args.max_turns_per_conv,
        max_qs_per_conv=args.max_qs_per_conv,
        max_chunks_per_conv=args.max_chunks_per_conv,
        checkpoint_eval_every_chunks=args.checkpoint_eval_every_chunks,
        skip_qa=args.skip_qa,
        read_policy_name=args.read_policy,
        reader_k=args.reader_k,
        judge_mode=args.judge_mode,
        progress_every=args.progress_every,
    )
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
