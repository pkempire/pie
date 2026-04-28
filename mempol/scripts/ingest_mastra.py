"""Step A: ingest a LoCoMo conv into Mastra OM and save the state to disk.

Usage:
    python -m mempol.scripts.ingest_mastra --conv-idx 0 --run-name mastra_c1
"""
from __future__ import annotations
import argparse
import json
import time
from pathlib import Path

from mempol import config
from mempol.backends.mastra import MastraBackend
from mempol.data.locomo import load as load_locomo
from mempol.data.longmemeval import load as load_lme
from mempol.eval.runner import conv_to_units


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", choices=["locomo", "longmemeval_s",
                                           "longmemeval_oracle", "longmemeval_m"],
                    default="locomo")
    ap.add_argument("--conv-idx", type=int, default=0)
    ap.add_argument("--run-name", default="mastra_c1")
    # Mastra's actual defaults: 30k / 40k. For LoCoMo (~21k tokens) we keep low
    # so the Observer fires multiple times — otherwise it barely engages.
    ap.add_argument("--observer-threshold", type=int, default=3000)
    ap.add_argument("--reflector-threshold", type=int, default=8000)
    args = ap.parse_args()

    if args.dataset == "locomo":
        convs = load_locomo(n_convs=args.conv_idx + 1)
    else:
        convs = load_lme(variant=args.dataset, n_convs=args.conv_idx + 1)
    conv, qas = convs[args.conv_idx]

    out_dir = config.RESULTS_DIR / args.run_name
    out_dir.mkdir(parents=True, exist_ok=True)
    state_path = out_dir / "mastra_state.pkl"
    log_path = out_dir / "memory_log.md"

    print(f"[{conv.sample_id}] ingesting {len(conv.turns)} turns...")
    t0 = time.time()
    b = MastraBackend(
        observer_token_threshold=args.observer_threshold,
        reflector_token_threshold=args.reflector_threshold,
    )
    b.ingest(conv_to_units(conv))
    elapsed = time.time() - t0
    print(f"  done in {elapsed:.1f}s. stats={b.stats()}")
    b.save(str(state_path))
    log_path.write_text(f"# Mastra OM log — {conv.sample_id}\n\n" + b.memory_log_md())
    print(f"  state saved to {state_path}")
    print(f"  log saved to {log_path}")
    (out_dir / "ingest_summary.json").write_text(json.dumps({
        "conv_id": conv.sample_id, "n_turns": len(conv.turns), "n_qas": len(qas),
        "wall_time_s": elapsed, "stats": b.stats(),
        "compression_ratio": b._stats["n_raw_chars_seen"] / max(1, b._stats["n_observation_chars"]),
    }, indent=2))


if __name__ == "__main__":
    main()
