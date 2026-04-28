"""Build an SFT dataset from the heuristic write policy's trajectories.

Why
---
The cold-start GRPO problem is real: at step 1 the LoRA is essentially
the base model, which means write rollouts default to whatever Qwen3
considers natural ("write a paragraph about the turn") rather than valid
tool-call sequences. This produces near-zero reward variance for many
steps and entropy collapses before useful exploration starts.

The Search-R1 recipe handles this by SFT-warming the LoRA on a few
hundred trajectories from a teacher policy before turning on GRPO. We do
the same thing here: run our HeuristicWritePolicy over ~500 LoCoMo turns,
record the (system + user prompt, assistant tool-call sequence, tool
results) triples, and emit them in the chat format Tinker's SFT recipe
consumes.

Usage:

    # 1. Generate the SFT data (~10 min, free; runs locally):
    python -m mempol.scripts.build_sft_dataset \\
        --n_convs 8 --turns_per_conv 100 \\
        --out runs/sft_warmup.jsonl

    # 2. SFT-train a Qwen3-4B LoRA on it (Tinker):
    python -m tinker_cookbook.recipes.sft.train \\
        model_name=Qwen/Qwen3-4B-Instruct-2507 \\
        lora_rank=32 \\
        dataset_path=runs/sft_warmup.jsonl \\
        n_epochs=2 batch_size=8 \\
        log_path=$HOME/mempol/runs/sft_warmup_$(date +%Y%m%d)

    # 3. Resume Phase B GRPO from that LoRA:
    SFT_CKPT=tinker://<paste-from-step-2-final-sampler>
    python -m tinker_cookbook.recipes.memory_rl.train_write \\
        ... \\
        load_checkpoint_path=$SFT_CKPT \\
        log_path=$HOME/mempol/runs/phaseB_warmstart_$(date +%Y%m%d)

Cost: ~$30 SFT training. GRPO afterward typically converges in
~½ to ⅓ the steps (per Search-R1 ablations).
"""
from __future__ import annotations
import argparse
import json
import logging
from pathlib import Path

from mempol.backends.pie_kg import PIEBackend
from mempol.data.locomo import load as load_locomo
from mempol.policies.v1_write import HeuristicWritePolicy
from mempol.recipes.memory_rl.write_env import WRITE_TASK_INSTRUCTIONS
from mempol.recipes.memory_rl.write_tools import WriteTool

logger = logging.getLogger(__name__)


def _build_user_block(turn_text: str, dia_id: str, session_date: str,
                      prior_turns_text: str,
                      existing_entities_summary: str) -> str:
    """Mirror the user-prompt formatter in write_env._initial_messages so
    SFT examples match the eventual GRPO env exactly."""
    return (
        f"Session: {session_date}\n"
        f"Recent turns leading to this one:\n{prior_turns_text}\n\n"
        f"FOCAL TURN ({dia_id}):\n{turn_text}\n\n"
        f"Existing entities (top-K nearby):\n"
        f"{existing_entities_summary or '(none)'}\n\n"
        f"What write ops should you emit for the focal turn? "
        f"Default to noop unless the turn carries durable, specific information."
    )


def _build_assistant_message(ops: list[dict]) -> dict:
    """Format the heuristic policy's chosen ops as a Qwen-style assistant
    message with tool_calls. The cookbook's SFT recipe accepts both
    structured `tool_calls` and inline `<tool_call>` text — we use the
    inline form so we don't depend on a specific renderer."""
    if not ops:
        return {"role": "assistant", "content":
                '<tool_call>{"name": "noop", "arguments": {"reason": "no durable info"}}</tool_call>'}
    chunks = []
    for op in ops:
        chunks.append("<tool_call>" + json.dumps({
            "name": op["name"],
            "arguments": op.get("arguments", {}),
        }, ensure_ascii=False) + "</tool_call>")
    return {"role": "assistant", "content": "\n".join(chunks)}


def _heuristic_ops_for_turn(
    policy: HeuristicWritePolicy,
    backend: PIEBackend,
    turn_text: str,
    dia_id: str,
    timestamp: float,
) -> list[dict]:
    """Run the heuristic write policy on one turn and return the ops it
    chose, formatted as Tinker-style tool_calls
    ({"name": ..., "arguments": ...}).

    The real API on HeuristicWritePolicy is `step(turn_text, dia_id,
    timestamp, backend, write_tool) -> WriteDecision`. We pass a fresh
    WriteTool wrapping the same backend so the policy's mutations land
    where we expect them, and we read `decision.raw_ops` (the LLM's
    chosen ops) instead of `applied_ops` (only ones that succeeded) —
    SFT learns the *intended* sequence, errors and all.
    """
    write_tool = WriteTool(backend=backend)
    decision = policy.step(
        turn_text=turn_text,
        dia_id=dia_id,
        timestamp=timestamp,
        backend=backend,
        write_tool=write_tool,
    )
    ops = decision.raw_ops or []
    if not ops:
        return [{"name": "noop", "arguments": {"reason": "no ops emitted"}}]
    return [
        {"name": op_spec.get("op", "noop"),
         "arguments": op_spec.get("args") or {}}
        for op_spec in ops
    ]


def build(out_path: Path, n_convs: int, turns_per_conv: int,
          n_prior_in_context: int = 2) -> int:
    convs = load_locomo(n_convs=n_convs)
    policy = HeuristicWritePolicy()
    out_path.parent.mkdir(parents=True, exist_ok=True)

    # Resume support: skip conversations already present in the output file.
    done_conv_ids: set[str] = set()
    if out_path.exists():
        try:
            for line in out_path.read_text().splitlines():
                meta = (json.loads(line).get("metadata") or {})
                cid = meta.get("conv_id")
                if cid:
                    done_conv_ids.add(cid)
            if done_conv_ids:
                logger.info("Resume mode: %d conversations already in %s; "
                            "skipping them.", len(done_conv_ids), out_path)
        except Exception as e:
            logger.warning("Could not parse existing output for resume: %s", e)

    written = 0
    # Append-mode so resume actually appends, not overwrites.
    with out_path.open("a") as f:
        for conv, _qas in convs:
            if conv.sample_id in done_conv_ids:
                continue
            backend = PIEBackend()                 # one shared backend per conv
            for ti, t in enumerate(conv.turns[:turns_per_conv]):
                prior = conv.turns[max(0, ti - n_prior_in_context):ti]
                prior_text = "\n".join(
                    f"  {p.dia_id} {p.speaker}: {p.text}" for p in prior
                ) or "  (none)"
                # Existing-entity hint: top-3 by recency (cheap, mirrors env)
                ent_summary_lines = []
                for e in list(backend.wm.entities.values())[-3:]:
                    ent_summary_lines.append(
                        f"  {e.id} {e.name} ({e.type.value if hasattr(e.type, 'value') else e.type})"
                    )
                ent_summary = "\n".join(ent_summary_lines)

                user_block = _build_user_block(
                    turn_text=f"{t.speaker}: {t.text}",
                    dia_id=t.dia_id,
                    session_date=t.session_date,
                    prior_turns_text=prior_text,
                    existing_entities_summary=ent_summary,
                )

                ops = _heuristic_ops_for_turn(
                    policy=policy, backend=backend,
                    turn_text=f"{t.speaker}: {t.text}",
                    dia_id=t.dia_id, timestamp=float(ti),
                )
                assistant_msg = _build_assistant_message(ops)

                example = {
                    "messages": [
                        {"role": "system", "content": WRITE_TASK_INSTRUCTIONS},
                        {"role": "user",   "content": user_block},
                        assistant_msg,
                    ],
                    "metadata": {
                        "conv_id": conv.sample_id,
                        "turn_idx": ti,
                        "dia_id": t.dia_id,
                        "n_ops": len(ops),
                    },
                }
                f.write(json.dumps(example) + "\n")
                written += 1
            logger.info("conv %s — %d turns processed", conv.sample_id,
                        min(turns_per_conv, len(conv.turns)))
    logger.info("Wrote %d SFT examples to %s", written, out_path)
    return written


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--n_convs", type=int, default=8)
    parser.add_argument("--turns_per_conv", type=int, default=100)
    parser.add_argument("--n_prior_in_context", type=int, default=2)
    parser.add_argument("--out", type=Path, default=Path("runs/sft_warmup.jsonl"))
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO,
                        format="%(asctime)s %(levelname)s %(message)s")

    n = build(out_path=args.out, n_convs=args.n_convs,
              turns_per_conv=args.turns_per_conv,
              n_prior_in_context=args.n_prior_in_context)
    print(f"OK — wrote {n} SFT examples to {args.out}")


if __name__ == "__main__":
    main()
