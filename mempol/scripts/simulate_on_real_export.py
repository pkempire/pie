"""Simulate the heuristic write policy on a real ChatGPT export.

This is the sanity check before any RL training spend. We walk through a
slice of the user's conversations.json, run v1_write on each user/assistant
turn, and log every (turn, op) decision. The final memory state is dumped as
markdown so we can eyeball the artefact.

Cost-controlled by `--max-convs` and `--max-turns-per-conv`. Default is small.

Usage:
    python -m mempol.scripts.simulate_on_real_export \\
        --conversations-json ~/Documents/pie22/conversations.json \\
        --max-convs 3 --max-turns-per-conv 40 \\
        --run-name simrun_001
"""
from __future__ import annotations
import argparse
import json
import time
from dataclasses import asdict
from pathlib import Path

from mempol import config
from mempol.backends.pie_kg import PIEBackend
from mempol.data.necessity_miner import _flatten_chatgpt_export
from mempol.policies.v1_write import HeuristicWritePolicy
from mempol.recipes.memory_rl.write_tools import WriteTool


def dump_world_model_md(backend: PIEBackend, out_path: Path) -> Path:
    md = ["# World model — built by v1_write on real export\n"]
    wm = backend.wm
    md.append(f"**{len(wm.entities)} entities** with "
              f"{sum(len(wm.get_transitions(uid)) for uid in wm.entities)} transitions, "
              f"{sum(len(wm.get_relationships(uid)) for uid in wm.entities)} relationship edges.\n")
    by_type: dict[str, list] = {}
    for e in wm.entities.values():
        by_type.setdefault(e.type.value, []).append(e)

    md.append("\n## Entities by type\n")
    for t, es in sorted(by_type.items(), key=lambda kv: -len(kv[1])):
        md.append(f"- **{t}**: {len(es)}")
    md.append("")

    for t in sorted(by_type, key=lambda x: -len(by_type[x])):
        md.append(f"\n## {t.title()} ({len(by_type[t])})\n")
        for e in sorted(by_type[t], key=lambda x: -len(wm.get_transitions(x.id))):
            transitions = wm.get_transitions(e.id)
            state = json.dumps(e.current_state, ensure_ascii=False)
            md.append(f"### {e.name}  `{e.id[:8]}`")
            md.append(f"- state: `{state[:300]}`")
            if e.aliases:
                md.append(f"- aliases: {', '.join(e.aliases)}")
            md.append(f"- transitions ({len(transitions)}):")
            for tr in transitions[-5:]:
                md.append(f"  - **{tr.transition_type.value}** at t={tr.timestamp:.0f} "
                          f"({tr.trigger_summary[:80]})")
            rels = wm.get_relationships(e.id)
            if rels:
                md.append(f"- relations ({len(rels)}):")
                for r in rels[:5]:
                    other = r.target_id if r.source_id == e.id else r.source_id
                    other_e = wm.entities.get(other)
                    other_name = other_e.name if other_e else other[:8]
                    rel_type = r.type.value if hasattr(r.type, "value") else str(r.type)
                    md.append(f"  - {rel_type} → {other_name}")
            md.append("")

    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text("\n".join(md))
    return out_path


def run(
    conversations_json: Path,
    out_dir: Path,
    max_convs: int = 3,
    max_turns_per_conv: int = 40,
    candidate_roles: tuple[str, ...] = ("user", "assistant"),
) -> dict:
    print(f"[simulate] flattening {conversations_json}…")
    turns = _flatten_chatgpt_export(conversations_json)
    print(f"  flattened {len(turns)} turns total")

    # Group by conv_id and pick the first max_convs
    by_conv: dict[str, list] = {}
    for t in turns:
        by_conv.setdefault(t.conv_id, []).append(t)
    conv_ids = list(by_conv.keys())[:max_convs]
    print(f"  selecting {len(conv_ids)} convs ({max_turns_per_conv} turns each)")

    out_dir.mkdir(parents=True, exist_ok=True)
    decisions_path = out_dir / "decisions.jsonl"
    summary_path = out_dir / "summary.json"

    backend = PIEBackend()
    write_tool = WriteTool(backend=backend)
    policy = HeuristicWritePolicy()

    op_counter: dict = {}
    n_total_turns = 0
    n_skipped = 0
    t0 = time.time()

    with decisions_path.open("w", buffering=1) as f:
        for ci, cid in enumerate(conv_ids):
            conv_turns = by_conv[cid][:max_turns_per_conv]
            for ti, turn in enumerate(conv_turns):
                if turn.role not in candidate_roles or not turn.text:
                    n_skipped += 1
                    continue
                n_total_turns += 1
                ts = float(ti)  # turn-index as time proxy
                decision = policy.step(
                    turn_text=turn.text[:1500],
                    dia_id=f"{cid[:8]}::T{ti}",
                    timestamp=ts,
                    backend=backend,
                    write_tool=write_tool,
                )
                f.write(json.dumps({
                    "conv_id": cid,
                    "turn_idx": ti,
                    "role": turn.role,
                    "text": turn.text[:300],
                    "lookup_matches": [
                        {"uid": m["uid"][:8], "name": m["name"], "type": m["type"],
                         "match_score": m["match_score"]}
                        for m in decision.lookup_matches[:3]
                    ],
                    "raw_ops": decision.raw_ops,
                    "applied_ops": decision.applied_ops,
                    "errors": decision.errors,
                }) + "\n")
                for op in decision.applied_ops:
                    op_counter[op["op"]] = op_counter.get(op["op"], 0) + 1
                if (n_total_turns) % 5 == 0:
                    print(f"  conv {ci+1}/{len(conv_ids)} turn {ti+1}/{len(conv_turns)} "
                          f"  entities={len(backend.wm.entities)} "
                          f"  ops={op_counter}", flush=True)

    # Persist world model + summary.
    md_path = dump_world_model_md(backend, out_dir / "world_model.md")
    json_path = out_dir / "world_model.json"
    backend.save(str(json_path))

    summary = {
        "conversations_json": str(conversations_json),
        "max_convs": max_convs,
        "max_turns_per_conv": max_turns_per_conv,
        "n_convs_processed": len(conv_ids),
        "n_turns_processed": n_total_turns,
        "n_turns_skipped_by_role": n_skipped,
        "ops_applied": op_counter,
        "n_entities_final": len(backend.wm.entities),
        "n_transitions_final": sum(len(backend.wm.get_transitions(uid))
                                    for uid in backend.wm.entities),
        "n_relationships_final": sum(len(backend.wm.get_relationships(uid))
                                      for uid in backend.wm.entities),
        "wall_time_s": round(time.time() - t0, 2),
        "write_tool_stats": write_tool.write_stats(),
    }
    summary_path.write_text(json.dumps(summary, indent=2))
    print("\n=== SUMMARY ===")
    print(json.dumps(summary, indent=2))
    print(f"\nFiles in {out_dir}/:")
    for p in (decisions_path, md_path, json_path, summary_path):
        print(f"  {p.name}")
    return summary


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--conversations-json", required=True, type=Path)
    ap.add_argument("--out-dir", default=None, type=Path,
                    help="defaults to mempol/results/simrun_<timestamp>")
    ap.add_argument("--max-convs", type=int, default=3)
    ap.add_argument("--max-turns-per-conv", type=int, default=40)
    ap.add_argument("--run-name", default=None)
    args = ap.parse_args()

    if args.out_dir is None:
        rn = args.run_name or f"simrun_{int(time.time())}"
        args.out_dir = config.RESULTS_DIR / rn

    run(
        conversations_json=args.conversations_json,
        out_dir=args.out_dir,
        max_convs=args.max_convs,
        max_turns_per_conv=args.max_turns_per_conv,
    )
