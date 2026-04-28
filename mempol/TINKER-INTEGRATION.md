# Tinker integration plan for mempol

*Project-specific recipe for using Tinker (Thinking Machines) as our RL training infra.*

---

## 1. Why Tinker (vs verl / OpenRLHF / TRL)

| Framework | Compute | Algo flexibility | Code overhead | LoRA | Ours? |
|---|---|---|---|---|---|
| **Tinker** (Thinking Machines, Oct 2025) | **Managed** distributed GPUs | **High** — primitives `forward_backward`, `optim_step`, `sample` | Low (write loop locally) | Built-in | **Primary** |
| **verl** (ByteDance) | Self-managed | Highest, agentic-ready | Highest (~32k LOC) | Yes | Fallback for full FT |
| **OpenRLHF** | Self-managed | High | Medium (~8.5k LOC) | Yes | Fallback |
| **TRL** (HuggingFace) | Self-managed | Medium | Low (canonical) | Yes | Fallback for SFT/DPO |

The decisive Tinker win: **we don't run GPUs**. We write the rollout loop on a laptop, Tinker farms the forward/backward/sample to managed clusters. For a 2-person research team that's a 5–10x reduction in ops overhead. Tinker is in private beta — sign up, it's free at the start.

The Tinker primitives we need (all of them are in `rl_loop.py` recipe):

- `service_client = tinker.ServiceClient(base_url=...)` — connects to the platform.
- `training_client = service_client.create_lora_training_client(base_model=..., rank=32)` — initialize LoRA-tuned policy.
- `sampling_client = training_client.save_weights_and_get_sampling_client()` — get a generator that uses current weights.
- `sampling_client.sample(prompt, num_samples=G, sampling_params=...)` — return Future of G samples for one prompt. **Critical: this is how we get group-relative samples for GRPO**.
- `training_client.forward_backward(datums, loss_fn="importance_sampling")` — gradient pass.
- `training_client.optim_step(adam_params)` — apply Adam step.

Fallback if Tinker access blocks: same script structure works on TRL `GRPOTrainer` with a few wrapper changes. We won't be locked in.

## 2. Mapping our setting onto Tinker

Tinker's `rl_loop.py` recipe trains GSM8K (math). Each "rollout" is one (prompt, sampled_tokens, reward) triple. **Our setting is structurally identical** — the only difference is what the prompt is, what the rollout produces, and how reward is computed.

| GSM8K recipe | Our memory-policy recipe |
|---|---|
| Prompt = math question | Prompt = state encoder output (`<task>`, `<query>`, `<top_hits>`, `<recent_ops>`) |
| Sample G completions | Sample G op-sequences (each a single JSON op for read-policy step, or full trajectory) |
| Reward = `\boxed{}` answer matches gold | Reward = `judge(question, gold, answer_built_from_traj) - λ·cost` |
| `advantage = reward - mean(rewards_G)` | Same |
| Datum = (model_input, target_tokens, logprobs, advantages) | Same |

**Critical design choice — single-step vs trajectory-level RL.** Tinker's recipe is single-completion. Our policy emits *multiple ops per query*. Two options:

1. **Single-step RL (recommended start)**: each rollout = one op decision. Model sees current state → outputs one op → state advances → loop. Reward propagates back to all ops in trajectory (with discount γ). Maps cleanly to Tinker — each op decision becomes one Datum.

2. **Trajectory-level (later)**: model emits full op sequence as one big completion. Reward is on the whole sequence. More like Search-R1's recipe. Tinker can do this — the GSM8K recipe already does multi-token sampling per "completion."

Start with (1). Move to (2) only if (1) is unstable.

## 3. The rollout loop — adapted from `rl_loop.py`

Below is the actual code shape we'll write, mirroring `tinker_cookbook/recipes/rl_loop.py` lines for lines but with our memory-specific bits. Save as `mempol/train/grpo_tinker.py` once Tinker access lands.

```python
"""GRPO over read-policy rollouts on LoCoMo. Mirrors tinker-cookbook/rl_loop.py."""
import time
import chz
import tinker
import torch
from tinker import types
from tinker.types.tensor_data import TensorData
from tinker_cookbook import checkpoint_utils, model_info, renderers
from tinker_cookbook.tokenizer_utils import get_tokenizer
from tinker_cookbook.utils import ml_log

from mempol.data.locomo import load
from mempol.backends.flat import FlatBackend
from mempol.eval.runner import conv_to_units
from mempol.eval.judge import judge
from mempol.policies.op_schema import build_state, parse_op_response
from mempol.rollout import run_trajectory_with_policy_tokens


@chz.chz
class Config:
    base_url: str | None = None
    log_path: str = "/tmp/mempol/grpo"
    model_name: str = "Qwen/Qwen2.5-3B-Instruct"
    batch_size: int = 16
    group_size: int = 8           # G — rollouts per query
    learning_rate: float = 5e-6
    lora_rank: int = 32
    save_every: int = 20
    max_tokens: int = 128         # per-op JSON output is small
    cost_lambda: float = 0.005    # cost penalty


def make_train_questions():
    """Yield (state_prompt, gold_answer, conv_id, qa) tuples from LoCoMo train."""
    convs = load(n_convs=8)  # hold out last 2 for eval
    for conv, qas in convs:
        backend = FlatBackend()
        backend.ingest(conv_to_units(conv))
        for qa in qas:
            yield {
                "question": qa.question,
                "gold": qa.answer,
                "backend": backend,
                "qid": qa.qid,
            }


def main(config: Config):
    ml_logger = ml_log.setup_logging(log_dir=config.log_path, config=config)

    tokenizer = get_tokenizer(config.model_name)
    renderer = renderers.get_renderer(
        model_info.get_recommended_renderer_name(config.model_name), tokenizer
    )

    service_client = tinker.ServiceClient(base_url=config.base_url)
    training_client = service_client.create_lora_training_client(
        base_model=config.model_name, rank=config.lora_rank
    )

    sampling_params = tinker.types.SamplingParams(
        max_tokens=config.max_tokens,
        stop=renderer.get_stop_sequences(),
        temperature=0.7,
    )
    adam_params = types.AdamParams(learning_rate=config.learning_rate, beta1=0.9, beta2=0.95, eps=1e-8)

    queries = list(make_train_questions())
    n_batches = len(queries) // config.batch_size

    for batch_idx in range(n_batches):
        t0 = time.time()
        sampling_client = training_client.save_weights_and_get_sampling_client()
        batch = queries[batch_idx * config.batch_size:(batch_idx + 1) * config.batch_size]

        # ----- Phase A: collect rollouts -----
        # For each query, run policy through (up to) max_steps op decisions.
        # Each step is one Tinker `sample()` call producing G alternatives.
        all_datums = []
        all_rewards = []
        for q in batch:
            traj_set = run_trajectory_with_policy_tokens(
                question=q["question"],
                backend=q["backend"],
                gold=q["gold"],
                sampling_client=sampling_client,
                renderer=renderer,
                sampling_params=sampling_params,
                G=config.group_size,
                cost_lambda=config.cost_lambda,
            )
            # traj_set is a list of G trajectories, each with .reward, .step_records
            rewards_G = [t.reward for t in traj_set]
            mean_r = sum(rewards_G) / len(rewards_G)
            advs_G = [r - mean_r for r in rewards_G]
            all_rewards.append(mean_r)

            if all(a == 0.0 for a in advs_G):
                continue

            # Each step in each trajectory contributes one Datum.
            for traj, adv in zip(traj_set, advs_G):
                for step_rec in traj.step_records:
                    # step_rec contains: prompt_tokens, sampled_tokens, logprobs
                    ob_len = len(step_rec.prompt_tokens) - 1
                    model_input = step_rec.model_input.append(
                        types.EncodedTextChunk(tokens=step_rec.sampled_tokens[:-1])
                    )
                    target_tokens = [0] * ob_len + step_rec.sampled_tokens
                    padded_logprobs = [0.0] * ob_len + step_rec.logprobs
                    padded_advs = [0.0] * ob_len + [adv] * (model_input.length - ob_len)
                    all_datums.append(types.Datum(
                        model_input=model_input,
                        loss_fn_inputs={
                            "target_tokens": TensorData.from_torch(torch.tensor(target_tokens)),
                            "logprobs": TensorData.from_torch(torch.tensor(padded_logprobs)),
                            "advantages": TensorData.from_torch(torch.tensor(padded_advs)),
                        },
                    ))

        # ----- Phase B: gradient step -----
        if all_datums:
            fwd = training_client.forward_backward(all_datums, loss_fn="importance_sampling")
            opt = training_client.optim_step(adam_params)
            fwd.result(); opt.result()

        ml_logger.log_metrics({
            "reward/mean": sum(all_rewards) / len(all_rewards),
            "n_datums": len(all_datums),
            "time/batch": time.time() - t0,
        }, step=batch_idx)

        if batch_idx % config.save_every == 0 and batch_idx > 0:
            checkpoint_utils.save_checkpoint(
                training_client=training_client,
                name=f"{batch_idx:06d}",
                log_path=config.log_path,
                kind="state",
                loop_state={"batch": batch_idx},
            )


if __name__ == "__main__":
    chz.nested_entrypoint(main)
```

The new piece is `run_trajectory_with_policy_tokens` which we ship in `mempol/rollout.py` (framework-agnostic). It:

1. Builds the initial state prompt for the question.
2. Repeatedly calls `sampling_client.sample(prompt, num_samples=G, sampling_params)` — but per-step. Track each branch independently.
3. Parses each sample's tokens as a JSON op via the constrained renderer.
4. Applies the op via the backend; updates state.
5. When `stop_and_answer` emitted, runs the answer LLM, scores via `judge()`, computes cost.
6. Returns G trajectories with rewards and per-step records (prompt tokens + sampled tokens + logprobs).

This is the same primitive whether we end up training on Tinker, TRL, or verl — only the Datum packaging changes.

## 4. SFT step on Tinker (week 5–6)

Same idea, simpler. Use `tinker_cookbook/recipes/sl_loop.py` as the template. We provide `(prompt, completion)` pairs from `mempol/traces/*.jsonl`:

```python
@chz.chz
class SFTConfig:
    model_name: str = "Qwen/Qwen2.5-1.5B-Instruct"
    train_jsonl: str = "mempol/data/sft.jsonl"
    learning_rate: float = 2e-5
    lora_rank: int = 32
    epochs: int = 3

# Per-batch:
service_client = tinker.ServiceClient()
training_client = service_client.create_lora_training_client(base_model=cfg.model_name, rank=32)
for epoch in range(cfg.epochs):
    for batch in batches_of(load_jsonl(cfg.train_jsonl), bs=8):
        datums = [build_sft_datum(row, renderer) for row in batch]
        fwd = training_client.forward_backward(datums, loss_fn="cross_entropy")
        opt = training_client.optim_step(adam_params)
        fwd.result(); opt.result()
```

## 5. The first thing to build (right now)

Before any RL, build **the data moat**: the future-query miner. Without it the reward signal isn't real. Two days of work, single Python file, no GPUs needed.

`mempol/data/backref_miner.py`:

```python
"""Mine (antecedent_turn, future_query, gold_answer) triples from ChatGPT export.

Two-pass:
  1. Regex prefilter: turns containing back-reference markers
     ("remember when", "what did I", "going back to", "earlier you said", etc.)
  2. LLM verification + grounding: identify the antecedent turn(s), extract gold

Output: data/backrefs.jsonl with rows
  {antecedent_dia_ids: [str], future_query: str, gold_span: str, confidence: float, source: str}
"""
```

Pseudocode for the miner:

```
load conversations.json (your ChatGPT export)
flatten into a single chronological list of turns

for each turn t with role="user":
    if not regex_match(t.text, BACKREF_PATTERNS):
        continue
    # gather candidate antecedents: turns t' from t-100 to t-1
    candidates = turns[max(0, idx-100):idx]
    # ask LLM: of these, which is the most likely antecedent?
    prompt = JUDGE_PROMPT.format(query=t.text, candidates=fmt(candidates))
    resp = llm.chat(prompt, json_mode=True)
    if resp.confidence < 0.5: continue
    write {
        antecedent_dia_ids: resp.antecedent_ids,
        future_query: t.text,
        gold_span: resp.gold_span,
        confidence: resp.confidence,
        source: t.id,
    }
```

Expected yield from a 2-year ChatGPT export: 3–10K triples after filter. That's enough for SFT-scale write-policy training. Cost: ~$5 for the LLM verification pass.

If yield <500: the data moat isn't real, pivot the paper away from "longitudinal back-references" toward synthetic reward generation (held-out turn k, generate question that requires turn t).

## 6. Order of operations (next ~3 weeks)

Week 1 (this week):
- Build `mempol/rollout.py` (framework-agnostic rollout + advantage). **Done before Tinker access.**
- Build `mempol/data/backref_miner.py` and run on author's ChatGPT export. Get the miner-yield number.
- Sign up for Tinker beta.

Week 2:
- Run `v1_heuristic` on full LoCoMo (1986 Qs) → first real Table-1 row.
- Ship Tree backend + Graph backend (Person B, see PAPER-SPEC §14).
- Convert v1 traces to SFT format (`mempol/data/sft.jsonl`).

Week 3:
- If Tinker access in: run `train/sft_tinker.py` on Qwen2.5-1.5B → v2_sft.
- If not: run TRL SFT locally as fallback → same artifact.
- Eval v2_sft on LoCoMo. Acceptance bar: ≥95% match on op type vs teacher.

After that: DPO (week 4), GRPO (week 5–6), co-train (week 7+). Same code structure, swap trainer.

## 7. References

- [Tinker docs](https://tinker-docs.thinkingmachines.ai/) — main entry
- [tinker-cookbook on GitHub](https://github.com/thinking-machines-lab/tinker-cookbook) — recipes including `rl_loop.py` (the GRPO template above)
- [First RL tutorial](https://tinker-docs.thinkingmachines.ai/tutorials/basics/first-rl/) — minimal walkthrough
- [Tinker announcement](https://thinkingmachines.ai/blog/announcing-tinker/) — what it is, who can sign up
- Search-R1 (arXiv:2503.09516) — closest precedent for retrieval-as-action GRPO
- DeepSeekMath (arXiv:2402.03300) — original GRPO paper

## 8. If Tinker access is blocked

Plan B: **TRL `GRPOTrainer` locally on Modal/RunPod with a single A100**. Code structure stays nearly identical — replace:
```python
service_client.create_lora_training_client(...)
training_client.forward_backward(...)
training_client.optim_step(...)
```
with TRL's `GRPOConfig(...)` + `trainer.train()`. The `rollout.py` core stays the same.

Modal cost estimate: ~$3/hr on A100 80GB. 50 hours of training = ~$150. Fits the budget in PAPER-SPEC §12.
