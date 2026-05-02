"""Smoke test for Phase B (write policy).

Validates end-to-end without Tinker:
  1. Build WriteDatums from LoCoMo conv-26 (just turns where evidence exists).
  2. For each of N turns, simulate G=4 stochastic write trajectories using
     the heuristic write policy (LLM-based) — *not* a Tinker policy yet.
  3. After each W trajectory, check that PIEBackend has grown (entities > 0)
     and that WriteReward returns a sensible scalar.

Purpose: prove the deferred-reward signal is informative — i.e., different
W trajectories produce measurably different downstream R-accuracy. If they
don't, GRPO has nothing to learn and Phase B won't work.

Usage:
    python -m mempol.scripts.smoke_write --max-datums 3 --group-size 4
"""
from __future__ import annotations
import argparse
import asyncio
import json
import time
from pathlib import Path
from typing import Any

from mempol import config
from mempol.backends.pie_kg import PIEBackend
from mempol.data.locomo import load
from mempol.policies.v1_write import HeuristicWritePolicy
from mempol.recipes.memory_rl.write_env import WriteRLDatasetBuilder, WriteDatum
from mempol.recipes.memory_rl.write_reward import WriteReward
from mempol.recipes.memory_rl.write_tools import WriteTool


async def simulate_w_trajectory(
    datum: WriteDatum,
    write_policy: HeuristicWritePolicy,
    seed_offset: int,
) -> tuple[PIEBackend, list[dict], dict]:
    """Run one W trajectory using the heuristic write policy.
    Returns (mutated backend, fake_history_for_reward, ops_summary)."""
    backend = PIEBackend()
    wtool = WriteTool(backend=backend)

    decision = write_policy.step(
        turn_text=datum["turn_text"],
        dia_id=datum["turn_dia_id"],
        timestamp=float(datum["turn_idx"] + seed_offset),
        backend=backend,
        write_tool=wtool,
        observation_time_text=str(datum.get("session_date") or ""),
    )

    # Build a fake "history" so WriteReward can count ops via its tool_call regex.
    # Each applied op becomes one assistant message with a synthetic tool_call.
    fake_history = []
    for op in decision.applied_ops:
        fake_history.append({
            "role": "assistant",
            "content": f'<tool_call>{{"name":"{op["op"]}","arguments":{json.dumps(op["args"])}}}</tool_call>',
        })

    ops_summary = {
        "n_applied": len(decision.applied_ops),
        "ops": [op["op"] for op in decision.applied_ops],
        "errors": decision.errors,
    }
    return backend, fake_history, ops_summary


async def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--max-datums", type=int, default=3,
                    help="how many turns (with evidence) to test")
    ap.add_argument("--group-size", type=int, default=4)
    ap.add_argument("--n-convs", type=int, default=1)
    ap.add_argument("--out-dir", default=str(config.RESULTS_DIR / "smoke_write"))
    args = ap.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    log_path = out_dir / "trace.jsonl"

    # 1. Build a few WriteDatums via the dataset builder logic
    print(f"[smoke_write] loading {args.n_convs} LoCoMo conv(s)…")
    convs = load(n_convs=args.n_convs)

    # Mimic WriteRLDatasetBuilder.conv_to_write_data inline (skip Tinker chz)
    datums: list[WriteDatum] = []
    for conv, qas in convs:
        evidence_index: dict[str, list] = {}
        for qa in qas:
            for did in qa.evidence:
                evidence_index.setdefault(did, []).append(qa)
        for ti, t in enumerate(conv.turns):
            qas_for_turn = evidence_index.get(t.dia_id, [])
            if not qas_for_turn:
                continue
            datums.append({
                "conv_id": conv.sample_id, "turn_idx": ti,
                "turn_text": f"{t.speaker}: {t.text}",
                "turn_dia_id": t.dia_id,
                "session_date": t.session_date,
                "prior_turns_text": "(skipped in smoke)",
                "existing_entities_summary": "",
                "query_battery": [
                    (qa.question, qa.answer, list(qa.evidence or []))
                    for qa in qas_for_turn[:6]
                ],
            })
            if len(datums) >= args.max_datums:
                break
        if len(datums) >= args.max_datums:
            break

    print(f"  built {len(datums)} datums (turns with evidence)")
    write_policy = HeuristicWritePolicy()

    t0 = time.time()
    summary_rows = []
    with log_path.open("w") as f:
        for di, datum in enumerate(datums):
            print(f"\n=== datum {di+1}/{len(datums)} {datum['turn_dia_id']} ===")
            print(f"  turn: {datum['turn_text'][:120]}")
            print(f"  battery_size: {len(datum['query_battery'])}")

            rewards = []
            n_entities_per_traj = []
            for g in range(args.group_size):
                t_g = time.time()
                backend, fake_history, ops_summary = await simulate_w_trajectory(
                    datum, write_policy, seed_offset=g
                )
                reward_fn = WriteReward(
                    backend=backend,
                    query_battery=datum["query_battery"],
                )
                reward, metrics = await reward_fn(fake_history)
                rewards.append(reward)
                n_entities_per_traj.append(len(backend.wm.entities))

                print(f"    g={g}  ops={ops_summary['n_applied']:>2} "
                      f"({','.join(ops_summary['ops']) or 'noop':<30}) "
                      f"entities={metrics['n_entities']:>2.0f}  "
                      f"qa_mean={metrics['qa_mean']:.3f}  "
                      f"cost={metrics['cost_total']:.4f}  "
                      f"reward={reward:.3f}  ({time.time()-t_g:.1f}s)")

                f.write(json.dumps({
                    "datum_idx": di,
                    "g": g,
                    "turn_dia_id": datum["turn_dia_id"],
                    "battery_size": len(datum["query_battery"]),
                    "ops_applied": ops_summary["ops"],
                    "n_entities": int(metrics["n_entities"]),
                    "qa_mean": metrics["qa_mean"],
                    "reward": reward,
                }) + "\n")

            # GRPO-style group statistics
            if len(rewards) >= 2:
                mean_r = sum(rewards) / len(rewards)
                var_r = sum((r - mean_r) ** 2 for r in rewards) / len(rewards)
                std_r = var_r ** 0.5
                advs = [(r - mean_r) / max(std_r, 1e-6) for r in rewards]
                summary_rows.append({
                    "datum": di,
                    "rewards": rewards,
                    "mean_reward": mean_r,
                    "std_reward": std_r,
                    "advantages": advs,
                    "n_entities_per_traj": n_entities_per_traj,
                })
                print(f"  group: mean={mean_r:.3f} std={std_r:.3f}  "
                      f"advs={[round(a,2) for a in advs]}")

    print(f"\n=== SUMMARY ({time.time()-t0:.1f}s) ===")
    n_with_signal = sum(1 for r in summary_rows if r["std_reward"] > 0.01)
    print(f"  datums with non-trivial reward variance: "
          f"{n_with_signal}/{len(summary_rows)}")
    print(f"  → if this is 0, the deferred-reward signal is uninformative")
    print(f"  → if >50%, Phase B has signal to train on")
    print(f"\n  trace written to {log_path}")
    (out_dir / "summary.json").write_text(json.dumps({
        "n_datums": len(datums),
        "group_size": args.group_size,
        "datums_with_signal": n_with_signal,
        "rows": summary_rows,
    }, indent=2))


if __name__ == "__main__":
    asyncio.run(main())
