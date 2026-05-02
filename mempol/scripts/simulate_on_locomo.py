"""Live write-policy simulation on LoCoMo.

This is the easiest demo path for watching the write policy make decisions
without needing a private ChatGPT export. It writes the same `decisions.jsonl`
shape as `simulate_on_real_export.py`, so the Streamlit dashboard can tail it.

Usage:
  python -m mempol.scripts.simulate_on_locomo \\
      --conv-idx 0 --max-turns 40 --sleep-sec 0.5 \\
      --out-dir mempol/results/locomo_live
"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

from mempol import config
from mempol.backends.pie_kg import PIEBackend
from mempol.data.locomo import load, parse_locomo_date
from mempol.policies.v1_write import HeuristicWritePolicy
from mempol.recipes.memory_rl.write_tools import WriteTool
from mempol.scripts.simulate_on_real_export import dump_world_model_md


def run(
    conv_idx: int,
    out_dir: Path,
    max_turns: int = 40,
    sleep_sec: float = 0.0,
    context_turns: int = 6,
    checkpoint_every: int = 1,
) -> dict:
    convs = load(n_convs=conv_idx + 1)
    conv, qas = convs[conv_idx]
    turns = conv.turns[:max_turns]

    evidence_ids = {did for qa in qas for did in (qa.evidence or [])}

    out_dir.mkdir(parents=True, exist_ok=True)
    decisions_path = out_dir / "decisions.jsonl"
    summary_path = out_dir / "summary.json"

    backend = PIEBackend()
    write_tool = WriteTool(backend=backend)
    policy = HeuristicWritePolicy()
    op_counter: dict[str, int] = {}
    t0 = time.time()

    def _snapshot(n_turns_processed: int, md_path: Path | None = None) -> dict:
        if md_path is None:
            md_path = out_dir / "world_model.md"
        dump_world_model_md(backend, md_path)
        backend.save(out_dir / "world_model.json")
        summary = {
            "dataset": "locomo",
            "conv_id": conv.sample_id,
            "conv_idx": conv_idx,
            "max_turns": max_turns,
            "context_turns": context_turns,
            "n_turns_processed": n_turns_processed,
            "n_evidence_turns_seen": sum(1 for t in turns[:n_turns_processed] if t.dia_id in evidence_ids),
            "ops_applied": op_counter,
            "n_entities_final": len(backend.wm.entities),
            "n_transitions_final": sum(len(backend.wm.get_transitions(uid)) for uid in backend.wm.entities),
            "n_relationships_final": sum(len(backend.wm.get_relationships(uid)) for uid in backend.wm.entities),
            "wall_time_s": round(time.time() - t0, 2),
            "write_tool_stats": write_tool.write_stats(),
            "decisions_path": str(decisions_path),
            "world_model_md": str(md_path),
            "status": "complete" if n_turns_processed >= len(turns) else "running",
        }
        summary_path.write_text(json.dumps(summary, indent=2))
        return summary

    print(f"[simulate_on_locomo] {conv.sample_id}: {len(turns)} turns, "
          f"{len(qas)} qas, {len(evidence_ids)} evidence dia_ids")
    with decisions_path.open("w", buffering=1) as f:
        for ti, turn in enumerate(turns):
            recent = turns[max(0, ti - context_turns):ti]
            recent_context = "\n".join(
                f"{t.dia_id} [{t.session_date}] {t.speaker}: {t.text[:400]}"
                for t in recent
            )
            decision = policy.step(
                turn_text=f"{turn.speaker}: {turn.text}",
                dia_id=turn.dia_id,
                timestamp=parse_locomo_date(turn.session_date) or float(turn.session),
                backend=backend,
                write_tool=write_tool,
                observation_time_text=turn.session_date,
                recent_context_text=recent_context,
            )
            for op in decision.applied_ops:
                op_counter[op["op"]] = op_counter.get(op["op"], 0) + 1

            row = {
                "conv_id": conv.sample_id,
                "turn_idx": ti,
                "role": "speaker",
                "speaker": turn.speaker,
                "dia_id": turn.dia_id,
                "session_date": turn.session_date,
                "is_evidence": turn.dia_id in evidence_ids,
                "text": turn.text[:500],
                "lookup_matches": [
                    {"uid": m["uid"][:8], "name": m["name"], "type": m["type"],
                     "match_score": m["match_score"]}
                    for m in decision.lookup_matches[:5]
                ],
                "raw_ops": decision.raw_ops,
                "applied_ops": decision.applied_ops,
                "errors": decision.errors,
                "n_entities_so_far": len(backend.wm.entities),
                "n_transitions_so_far": sum(len(backend.wm.get_transitions(uid)) for uid in backend.wm.entities),
                "n_relationships_so_far": sum(len(backend.wm.get_relationships(uid)) for uid in backend.wm.entities),
                "write_tool_stats": write_tool.write_stats(),
                "context_turns_used": len(recent),
            }
            f.write(json.dumps(row, ensure_ascii=False) + "\n")
            f.flush()
            print(f"  {ti+1:>3}/{len(turns)} {turn.dia_id:<6} "
                  f"{'*' if row['is_evidence'] else ' '} "
                  f"ops={[op['op'] for op in decision.applied_ops]} "
                  f"entities={len(backend.wm.entities)}",
                  flush=True)
            if checkpoint_every > 0 and ((ti + 1) % checkpoint_every == 0 or ti + 1 == len(turns)):
                _snapshot(ti + 1)
            if sleep_sec > 0:
                time.sleep(sleep_sec)

    summary = _snapshot(len(turns))
    print("\n=== SUMMARY ===")
    print(json.dumps(summary, indent=2))
    return summary


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--conv-idx", type=int, default=0)
    ap.add_argument("--max-turns", type=int, default=40)
    ap.add_argument("--out-dir", type=Path, default=config.RESULTS_DIR / "locomo_live")
    ap.add_argument("--sleep-sec", type=float, default=0.0)
    ap.add_argument("--context-turns", type=int, default=6)
    ap.add_argument("--checkpoint-every", type=int, default=1)
    args = ap.parse_args()
    run(
        conv_idx=args.conv_idx,
        out_dir=args.out_dir,
        max_turns=args.max_turns,
        sleep_sec=args.sleep_sec,
        context_turns=args.context_turns,
        checkpoint_every=args.checkpoint_every,
    )


if __name__ == "__main__":
    main()
