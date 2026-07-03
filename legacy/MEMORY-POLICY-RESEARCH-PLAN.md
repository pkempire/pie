# Memory as a Learned Policy — Research Plan & RL Tutorial

*A first-principles teaching pass on how to train and benchmark a learned memory policy. Built around your PIE / PIE22 codebase.*

---

## 0. What this document is

You asked four things:

1. **Graph vs tree (filesystem) for retrieval** — which is better, can the policy be representation-agnostic?
2. **A quick test run on LoCoMo** — concrete steps, not theory
3. **What "training a small policy" actually means**, step by step, from scratch
4. **All the RL methods (SFT, DPO, SimPO, GRPO, PPO)** — first principles to current SOTA, because RL is new to you

Plus the framing for the **joint write+read co-training paper** (option C from the previous discussion).

This document is built so you can read it linearly and end up understanding the whole stack. It assumes you're comfortable with Python and basic ML (gradient descent, neural nets) but not with RL.

---

## Part 1 — Graph vs Tree retrieval (~10 min read)

### The honest answer

The choice matters **less than people think**, because both end up searched the same way at the leaves: vector similarity + lexical match. What actually differs is where you put structure and what becomes cheap vs expensive.

| Aspect | Graph (PIE-style) | Tree/FS (PIE22-style) |
|---|---|---|
| Storage unit | Entity + typed transitions + edges | Markdown file in folder hierarchy |
| Primary retrieval | Embedding lookup → 1-hop expansion | FTS5 keyword + path filter + dense rerank |
| Multi-hop ("X did Y → who else") | Native — follow edges | Requires extracted joins |
| Locality (related items together) | Explicit (edges) | Implicit (folder co-residence) |
| Robustness | Brittle if entity resolution fails | Robust — files always exist |
| Update cost | Need to merge entities, propagate transitions | Append / move |
| Latency | Vector search + graph walk | FTS5 + path scan (very fast) |
| Best at | Multi-hop, relational, temporal-causal | Single-document recall, broad keyword |

### What the empirical data tells you

From your own audit: **Mastra (text-only, no graph at all) hits 94.87% on LongMemEval**. That single number is most of the answer. For pure fact-recall benchmarks, *the graph isn't load-bearing*. Most LoCoMo questions are factual recall + temporal — tree+FTS+dense will probably **tie or beat** KG on LoCoMo accuracy.

Where graph should win:
- Multi-hop queries where the *path* is the answer
- Temporal-causal reasoning across linked entities
- Behavioral / proactive surfacing (find related items the user didn't ask about)
- The thing TemporalBench is supposed to measure

### The right move for the policy

**Don't pick one.** Build the policy to emit ops in a backend-neutral middle language; have backend adapters compile to whichever store. A `expand_neighbor(node, edge_type)` op compiles to "follow link X" in graph, "look in adjacent folder Y" in FS. The policy learns *when* to expand vs not — the where is backend's problem.

This gives you a clean ablation in the paper:

> *"We train one policy over backend-agnostic ops, evaluate on three backends (KG, FS, flat), and show consistent wins."*

That's a Section 5 figure that wins reviewer 2.

---

## Part 2 — Getting on LoCoMo this week (concrete steps)

LoCoMo (Snap, 2024) has 50 long multi-session dialogues, ~19 sessions each, ~300 turns and 9K tokens average. Five question types: single-hop, multi-hop, temporal, adversarial, commonsense. Current SOTA on the leaderboard is in the 0.85+ range (MemMachine reported ~0.85; Mem0 / Zep / Memobase a bit below).

You already have it at `benchmarks/locomo/` in your repo. Here's the 3-day setup.

### Day 1 — sanity baselines

Goal: run two existing baselines to ground yourself in the harness and the metric. Don't write any new code today.

```bash
cd benchmarks/locomo

# Inspect the data
ls data/                    # what splits exist
head -50 data/<file>.json   # what's the shape — check questions, answers

# Run naive RAG baseline (just dense top-k over all turns)
python run_locomo.py --provider naive_rag --output results/baseline_naive

# Run Mem0 (or whatever's already wired)
python run_locomo.py --provider mem0 --output results/baseline_mem0
```

The point: get **two numbers in a results JSON**. If your existing harness returns 0% (which the audit said it does), spend day 1 fixing the harness. **No memory research is possible until you can produce a non-zero baseline.**

### Day 2 — wire two backends with a fixed read policy

Build the simplest possible adapters:

- **Tree backend**: every conversation session → one `.md` file at `<speaker>/<session_id>.md`. Build a SQLite FTS5 index. Add OpenAI embeddings in a sidecar table.
- **Graph backend**: point your existing PIE pipeline at LoCoMo dialogues, build the world model, expose retrieval via `pie.retrieval.hybrid_retriever`.

Then write the dumbest possible read policy in `bench/read_policy_v0.py`:

```python
def read_policy_v0(query, backend, k=5):
    # No reasoning. No reformulation. No iteration. Just retrieve and return.
    return backend.retrieve(query, k=k)

def answer_with_context(query, retrieved, llm):
    prompt = f"Answer using only these:\n{retrieved}\n\nQ: {query}\nA:"
    return llm(prompt)
```

Run this against both backends. Now you have **four numbers**: naive_RAG, Mem0, tree_v0, graph_v0. This is your floor.

### Day 3 — wire a heuristic-but-better policy as the teacher

This becomes your SFT teacher in week 2. Make it competitive but rule-based:

```python
def read_policy_heuristic(query, backend, llm):
    # 1. Reformulate
    reform = llm(f"Rewrite this as a search query: {query}")
    # 2. First-pass retrieve
    cands = backend.retrieve(reform, k=20)
    # 3. If query has temporal cue, filter by time window
    if has_temporal_marker(query):
        cands = filter_by_time(cands, extract_window(query))
    # 4. If multi-hop indicator, expand 1 hop
    if is_multihop(query):
        cands = expand_neighbors(cands, backend, max_hops=1)
    # 5. Rerank with embedding-cross-attention or just dense sim
    cands = rerank(cands, query, top_k=5)
    return cands
```

Run it. You should beat `read_policy_v0`. **This is the policy your learned model will eventually replace.**

### What you'll have by Friday

- Working LoCoMo harness with non-zero baselines
- Two backends behind a uniform `backend.retrieve(query, k)` interface
- A heuristic teacher that beats vanilla retrieval

That's the launchpad for the actual learning work. Without this, no amount of RL helps you.

---

## Part 3 — What "training a small policy" actually means, step by step

This section is the bridge to the RL tutorial. Read this even if you skip the math.

### What a "policy" is, mechanically

A **policy** is a function $\pi(a \mid s)$ that takes a state $s$ and outputs a distribution over actions $a$. In our case:

- **State $s$** = a structured description of "what's happening right now": the query, what's in memory, what we've already retrieved, how much budget is left.
- **Action $a$** = one of the memory ops from the vocab (`expand`, `route`, `rerank`, `summarize`, `stop_and_answer`, etc.) plus its arguments.

A **small policy** = a small language model (1.5B–7B parameters, e.g. Qwen2.5-1.5B, Phi-4-mini, Llama-3.2-3B) that you've fine-tuned to take a textual description of $s$ and output a JSON action.

### What "training" means here

Training the policy means: **given lots of (state, action, outcome) data, adjust the model's weights so it picks actions that lead to good outcomes more often**.

The whole RL/SFT/DPO/GRPO zoo is just *different recipes* for doing this adjustment. They differ in:
- What signal you train on (gold actions vs preferences vs scalar rewards)
- Whether you need a separate critic model
- How many copies of the model you keep in memory
- How stable training is

### The mental model: imitate → critique → optimize

Three stages, in order of increasing power and pain:

1. **Imitate** (Supervised Fine-Tuning, SFT). Show the model lots of (state, good_action) pairs from a teacher. Train it to copy the teacher's actions. Easy. Bounded by teacher quality.

2. **Critique** (Reward Modeling + Preference Optimization, DPO/SimPO). Show the model pairs of (state, action_A, action_B, "A was better"). Train it to prefer A. Better than imitation because preferences are easier to collect than gold actions.

3. **Optimize** (Online RL: PPO, GRPO). Let the model try actions, score them with a reward function, push the model toward higher-reward actions. Most powerful, most expensive, most unstable.

You will do all three, in that order. Each layer only makes sense once the previous one is working.

### What you actually need to assemble before any training starts

| Component | What it is | Who provides it |
|---|---|---|
| **Base model** | A small pretrained LM | HuggingFace (Qwen2.5-1.5B etc.) |
| **State encoder** | Function that turns `(query, memory, history)` into a string the LM can consume | You write it |
| **Action schema** | JSON schema for the op vocabulary | You write it |
| **Constrained decoder** | Forces LM output to match the schema | `outlines`, `vllm` grammar |
| **Heuristic teacher** | Rule-based policy that produces (state, action) pairs | You write it (Day 3 above) |
| **Reward function** | Function `reward(trajectory) → float` | You write it (correctness − cost) |
| **Reward model (optional)** | Small LM trained to predict reward without running the env | You train it later |
| **Trainer** | Code that does SFT/DPO/GRPO | `trl`, `verl`, `OpenRLHF` |
| **Rollout server** | LM serving for fast generation during RL | `vllm` |
| **Eval harness** | LoCoMo / LongMemEval / TemporalBench runner | Already in your repo |

That's the whole stack. Build them once, swap algorithms (SFT → DPO → GRPO) by changing only the trainer.

---

## Part 4 — RL methods from first principles to SOTA

This is the longest section. Read it linearly. Each method builds on the previous.

### 4.1 Foundation: language models as policies

A modern LLM is a function that takes a sequence of tokens and outputs a probability distribution over the next token: $p_\theta(x_t \mid x_{<t})$. To generate a long output, you sample tokens one at a time.

You can think of an LLM as **already a policy**. The "state" is the prompt so far; the "action" is the next token; the "policy distribution" is the softmax over the vocabulary. Everything that follows is just teaching this policy to produce *better* outputs (where "better" depends on the task).

Three training stages of a modern LLM:
1. **Pretraining**: predict the next token on a giant corpus. Produces the "base model."
2. **Supervised fine-tuning (SFT)**: predict the next token on curated `(prompt, ideal_response)` pairs. Aligns the model with a task format.
3. **Preference / RL post-training (RLHF, DPO, GRPO, etc.)**: adjust the model so its outputs are *preferred* by humans or by a reward function.

For our memory policy, stages 2 and 3 are what we'll use.

### 4.2 Supervised Fine-Tuning (SFT) — "imitation"

**The problem**: you have a base LLM that can generate any text, but you want it to specifically output JSON memory ops in your schema.

**The solution**: collect a dataset of `(state_string, ideal_action_json)` pairs from your heuristic teacher. Fine-tune the LLM on these as if they were any other (input, output) pair, using the standard cross-entropy loss on the output tokens.

**The loss** (this is just next-token prediction, masked to only count loss on the action tokens):
$$\mathcal{L}_{\text{SFT}} = -\sum_{t \in \text{action}} \log p_\theta(a_t \mid s, a_{<t})$$

**In practice**:
```python
# Pseudocode using TRL
from trl import SFTTrainer
trainer = SFTTrainer(
    model="Qwen/Qwen2.5-1.5B",
    train_dataset=teacher_traces,  # [{"prompt": state, "completion": action_json}]
    formatting_func=lambda x: f"{x['prompt']}\n{x['completion']}",
)
trainer.train()
```

**What it gets you**: a model that emits valid actions in your schema, roughly matching your teacher. **Ceiling = teacher quality.** Cannot exceed the teacher because you're literally copying.

**When to stop**: when the policy matches the teacher on ≥95% of held-out states. If it can't reach that, your action vocab or state encoding is broken — fix it before going further.

### 4.3 Reward signals — what does "good" mean?

After SFT you have a model that mimics the teacher. To **beat** the teacher, you need a different signal: feedback on *outcomes*, not on action choice.

For our memory policy, candidate reward signals:

| Signal | How to compute | Strength |
|---|---|---|
| **QA correctness** | Run policy → answer query → compare to gold | Clean, immediate |
| **Cost** | Tokens spent, retrievals made, latency | Computable for free |
| **Length penalty** | Discourage overlong rollouts | Trivial |
| **Future-query lift** | Did this write op help a future query? | Deferred, expensive |
| **Artifact adoption** | Did the user open / use the produced artifact? | Slow, sparse |

For the read-side policy on LoCoMo:
$$R(\tau) = \mathbb{1}[\text{answer correct}] - \lambda_c \cdot \text{cost}(\tau) - \lambda_l \cdot \text{length}(\tau)$$

with $\lambda_c, \lambda_l$ small (start ~0.01).

### 4.4 Reward models — the bridge

You can use the reward function directly during training (online RL). But computing it might be slow (running the QA pipeline every time). So a common trick: **train a smaller "reward model" (RM) to predict the reward from (state, action_sequence) without running the environment.**

How? Collect lots of `(trajectory_A, trajectory_B, "A scored higher")` pairs by running the heuristic teacher with random perturbations. Train a small LM with a scalar head:
$$\mathcal{L}_{\text{RM}} = -\log \sigma\big(r_\phi(\tau_A) - r_\phi(\tau_B)\big)$$

This is **Bradley-Terry preference modeling** — same math as Elo ratings in chess.

Once trained, $r_\phi$ can score any new trajectory in milliseconds.

### 4.5 PPO (Proximal Policy Optimization) — the original RLHF

**The full RLHF recipe (OpenAI, 2022) used PPO.** Conceptually:

1. Sample a batch of prompts.
2. For each prompt, the **policy** $\pi_\theta$ generates a response.
3. The **reward model** scores the response.
4. Update $\pi_\theta$ to make high-reward responses more likely — but not *too* much, to keep training stable.

**The "not too much" matters.** If you just maximize reward, the policy collapses (gives the same output every time, exploits weird reward-model quirks, etc.). PPO uses two regularizers:

1. **KL penalty against a reference model** $\pi_{\text{ref}}$ (your SFT model): keeps the new policy close to the old one in distribution.
2. **Clipped surrogate objective**: only updates within a "trust region" each step.

**The PPO loss** (simplified):
$$\mathcal{L}_{\text{PPO}} = -\mathbb{E}\big[\min(r_t \cdot A_t, \text{clip}(r_t, 1-\epsilon, 1+\epsilon) \cdot A_t)\big] + \beta \cdot \text{KL}(\pi_\theta \| \pi_{\text{ref}})$$

where $r_t = \pi_\theta / \pi_{\text{old}}$ (importance ratio) and $A_t$ is the **advantage** (how much better this action was than expected).

**The annoying part**: PPO needs **four model copies in memory** simultaneously:
- The policy $\pi_\theta$ (being trained)
- The reference $\pi_{\text{ref}}$ (frozen, for KL)
- The reward model $r_\phi$ (frozen, for scoring)
- The **value/critic model** $V_\psi$ (estimates expected reward — needed to compute advantage $A_t$)

Four copies = lots of GPUs. Plus PPO is famously unstable and hyperparameter-sensitive.

**Why we mention it**: every method that follows is essentially "PPO but cheaper / more stable."

### 4.6 DPO (Direct Preference Optimization) — the closed-form trick

**Paper**: Rafailov et al. 2023, "Your Language Model is Secretly a Reward Model."

**The insight**: in PPO the reward model trains *separately* from the policy, and then the policy learns to maximize its scores. DPO observes there's a closed-form mathematical relationship between the optimal policy and the reward function. **You can skip the reward model entirely** — directly train the policy on preference pairs using a clever loss.

**The DPO loss**:
$$\mathcal{L}_{\text{DPO}} = -\mathbb{E}\Big[\log \sigma\big(\beta \log \frac{\pi_\theta(a_w \mid s)}{\pi_{\text{ref}}(a_w \mid s)} - \beta \log \frac{\pi_\theta(a_l \mid s)}{\pi_{\text{ref}}(a_l \mid s)}\big)\Big]$$

where $a_w$ is the "winner" action, $a_l$ is the "loser." Don't memorize this — the **insight** is what matters: it's a binary classification loss that says "make the winner more likely than the loser, scaled by how far you've drifted from the reference."

**What you get**:
- Only **two model copies** (policy + frozen reference)
- No separate reward model, no critic, no rollouts
- A standard supervised loss (just over preference pairs)
- Stable, easy, GPU-friendly

**What you lose**:
- You need preference pairs *up front* — DPO is offline. Can't iteratively improve from environment feedback.
- Performance plateaus once the preference set is exhausted.

**In practice**:
```python
from trl import DPOTrainer
trainer = DPOTrainer(
    model=sft_model,
    ref_model=sft_model_frozen_copy,
    train_dataset=preferences,  # [{"prompt", "chosen", "rejected"}]
    beta=0.1,
)
trainer.train()
```

**Most labs default to DPO for offline preference tuning.** It's the workhorse.

### 4.7 SimPO, KTO, ORPO — the variants

You'll hear these names. One-line each:

- **SimPO** (2024): DPO without the reference model. Replaces $\log(\pi_\theta / \pi_{\text{ref}})$ with average log-prob, normalized by length. Cheaper (drops one model copy) and often slightly stronger. Default if memory-tight.
- **KTO** (Kahneman-Tversky Optimization, 2024): Doesn't need *paired* preferences — just (action, "good" or "bad") labels. Useful if your preference data is one-sided (e.g., you only have positives + random negatives).
- **ORPO** (2024): Combines SFT and preference learning in a single loss. Train from base model directly without separate SFT phase. Trendy; mixed evidence on whether it beats SFT→DPO.

Pick **DPO or SimPO** for your offline phase. Don't overthink it.

### 4.8 GRPO (Group Relative Policy Optimization) — the DeepSeek breakthrough

**Paper**: DeepSeekMath (Shao et al., 2024). Used in DeepSeek-R1.

**The insight**: PPO needs a critic to estimate the advantage $A_t$. The critic is hard to train and adds a model copy. **Why not estimate the advantage from the data itself?**

**The recipe**:
1. For each prompt $s$, sample a **group** of $G$ rollouts (e.g., $G=8$ or $G=64$) from the current policy.
2. Score each with the reward function: $r_1, r_2, \ldots, r_G$.
3. Compute **relative advantage** for each rollout: how much above/below average it scored, normalized:
   $$A_i = \frac{r_i - \text{mean}(r)}{\text{std}(r)}$$
4. Update policy with PPO-style clipped objective, but using these group-relative advantages instead of critic-estimated ones.

**What you get**:
- **No critic model** (one fewer model copy)
- The "baseline" (mean reward of the group) is computed from data, not learned — much more stable
- Scales beautifully with group size
- Works exceptionally well with verifiable rewards (math, code, retrieval)

**The full GRPO objective**:
$$\mathcal{L}_{\text{GRPO}} = -\frac{1}{G}\sum_{i=1}^{G} \min(r_t^i \cdot A_i, \text{clip}(r_t^i, 1-\epsilon, 1+\epsilon) \cdot A_i) + \beta \cdot \text{KL}(\pi_\theta \| \pi_{\text{ref}})$$

Same shape as PPO, but $A_i$ comes from group-relative scoring instead of a critic.

**Practical numbers (from DeepSeekMath)**: batch size 1024, with 16 prompts × 64 rollouts each. Single update per batch (PPO does 2-4). Their result: GSM8K 82.9% → 88.2%, MATH 46.8% → 51.7%.

**Why it matters for us**: retrieval is a perfect fit. Each query → sample 8 trajectories → score each by (correctness − cost) → group-relative advantage → update. **Search-R1 (Mar 2025) uses exactly this and gets +41% over RAG baselines on Qwen2.5-7B**. That paper is your direct precedent.

### 4.9 When to use which — the decision tree

```
Have curated (state, gold_action) pairs from a teacher?
    └─ Yes → SFT (do this first, always)

After SFT, do you have preference pairs (winner, loser)?
    └─ Yes → DPO (or SimPO if memory-tight)
    └─ No → can you generate them from heuristics? Yes → DPO

Do you have a verifiable reward (correctness, math, retrieval-as-correct)?
    └─ Yes → GRPO (best ROI)
    └─ No, only a learned RM → PPO or RLOO (rare)
```

For our memory policy: **SFT (week 3) → DPO (week 5) → GRPO (week 7+)** is the standard path.

---

## Part 5 — Joint write+read co-training (the actual paper)

This is the move. Read sections 1–4 first.

### The setup

You have two policies sharing the same op vocabulary, applied to the same memory backend:

- **Write policy** $\pi^W$: at ingestion time, takes a turn of conversation + current memory, emits write ops (`assign`, `merge`, `hoist`, `forget`, etc.). Modifies memory.
- **Read policy** $\pi^R$: at query time, takes a query + current memory, emits read ops (`route`, `expand`, `rerank`, `stop_and_answer`). Returns retrieved items, then an LLM produces the answer.

Both are small LMs trained from the same base (e.g., Qwen2.5-3B). Both emit JSON in the same schema with op-type as a top-level field.

### Why co-train?

- The **read policy** needs the write policy to have stored *useful* things (otherwise no good answer is reachable)
- The **write policy** needs feedback from the read policy (storing X is "useful" exactly when the read policy successfully uses it)
- Trained independently, they optimize for orthogonal goals
- Trained together, they discover a *coordinated memory protocol*

This is **the actual novelty**. Nobody currently co-trains write+read memory policies with shared reward.

### The algorithm (the meat)

Alternating optimization, like GAN training but cleaner:

```
# === SETUP ===
W = SFT_train(write_policy_base, teacher_write_traces)   # week 3
R = SFT_train(read_policy_base, teacher_read_traces)     # week 3
W = DPO_train(W, write_preference_pairs)                  # week 4
R = DPO_train(R, read_preference_pairs)                   # week 4

# === CO-TRAIN LOOP ===
for iteration in range(N):
    # Phase 1: freeze W, train R via GRPO
    freeze(W)
    for query_batch in train_queries:
        # Use W to build memory state from training conversations
        memory = build_memory_with(W, conversations)
        # Sample G read rollouts per query
        rollouts = [R.sample_trajectory(q, memory) for _ in range(G)]
        rewards = [reward_QA(traj, gold[q]) for traj in rollouts]
        R = grpo_update(R, query_batch, rollouts, rewards)
    
    # Phase 2: freeze R, train W via GRPO
    freeze(R)
    for conv_batch in train_conversations:
        # Sample G write rollouts per conversation
        write_rollouts = [W.sample(conv) for _ in range(G)]
        # Score each by: build memory with that write trajectory, then run R on a future-query battery
        rewards = []
        for w_traj in write_rollouts:
            mem = apply(w_traj, base_memory)
            qa_acc = sum(R(q, mem) == gold[q] for q in future_queries[conv]) / len(future_queries[conv])
            rewards.append(qa_acc - lambda_cost * cost(w_traj))
        W = grpo_update(W, conv_batch, write_rollouts, rewards)
    
    # Phase 3: evaluate on held-out
    evaluate(W, R, eval_set)
    if converged: break
```

### Key design questions you'll have to answer

1. **How big is N (the outer loop)?** Start with 3–5 iterations. Convergence is empirical.

2. **Where do "future queries" come from for the write reward?** Three sources:
   - Mined from the corpus (the back-reference trick: when you say "what did I decide about X" in turn $t'$, that's a future query for an earlier turn $t$)
   - Synthetic: hold out turn $t+k$, use it as the answer, generate the question by prompting "what would you ask now to need turn $t$?"
   - LongMemEval / LoCoMo train splits

3. **How to prevent collapse?** Three guards:
   - KL penalty against the SFT reference for both $W$ and $R$
   - Cost regularization on $W$ (penalize storing too much)
   - Replay buffer: keep old (state, action, reward) tuples to prevent forgetting

4. **Compute budget?** Realistic for a side-project: 2 × 3B-param policies, GRPO with G=8, ~5–10 outer iterations. Fits on 4×A100 or 2×H100. ~$2–5K of compute end-to-end.

### Why this is publishable at NeurIPS / ICLR

The pitch:

> *"Memory write and retrieval decisions are typically hand-coded or trained independently. We show that co-training both as a single op-language with shared outcome rewards yields X% improvement on LongMemEval, Y% on LoCoMo, and Z% on TemporalBench, while transferring zero-shot across two memory backends (graph and filesystem) with only A% degradation."*

That's a NeurIPS-shaped story. Three numbers, one transfer claim, one ablation table.

---

## Part 6 — Practical 6-week starter plan

| Week | Deliverable | Tooling |
|---|---|---|
| 1 | LoCoMo harness fixed, 3 baseline numbers logged | existing repo |
| 2 | Backend abstraction + 2 backends + heuristic teacher policy | your code |
| 3 | Trace collection + SFT of read policy on Qwen2.5-1.5B | `trl`, `transformers` |
| 4 | Build read preference pairs (perturb teacher rollouts, score by QA), DPO train | `trl` |
| 5 | Wire GRPO trainer + vLLM rollouts; run on LoCoMo | `verl` or `trl` |
| 6 | First real number that beats heuristic teacher; freeze, write up Section 1–3 of paper | — |

After week 6: add write policy (weeks 7–10), then co-training (weeks 11–14), then ablations + writeup (weeks 15–16).

### Key tooling decisions

- **Framework**: start with `trl` (HuggingFace, well-documented). Switch to `verl` (Volcengine) if GRPO+vLLM throughput becomes the bottleneck.
- **Inference**: `vLLM` with prefix-caching enabled for rollouts. ~5x cheaper than naive HF generation for memory tasks.
- **Base model**: Qwen2.5-1.5B for the gate, Qwen2.5-3B for the op policy. Qwen 2.5 series has the best instruction-following at small sizes as of 2026.
- **Constrained decoding**: `outlines` for SFT, vLLM grammar for rollouts. Non-negotiable — saves 30%+ training failures.
- **Logging**: `wandb` for training, custom JSONL for trajectory logs.

### What to NOT do early

- Don't try to train an end-to-end "memory + reasoning" model. Memory is a tool the reasoner uses; keep them separated.
- Don't skip SFT and jump to RL. Every successful RL paper does SFT first. Without it, RL is fitting noise.
- Don't co-train W and R until each works well on its own. Co-training amplifies bugs in either.
- Don't tune more than 3 hyperparameters at a time. RL is unstable; you'll lose the plot.

---

## Appendix A — Glossary

- **Policy** $\pi$: function from state to action distribution. Here, a small LM.
- **State** $s$: structured snapshot of "the situation right now."
- **Action** $a$: a memory op (write or read).
- **Trajectory** $\tau$: sequence of (state, action) pairs ending in a terminal action like `stop_and_answer`.
- **Reward** $r$: scalar score for a trajectory or step. Can be sparse (terminal only) or dense (per step).
- **Rollout**: one sampled trajectory from the policy.
- **Advantage** $A_t$: how much better an action was than the average / baseline.
- **Reference model** $\pi_{\text{ref}}$: frozen copy of the policy used for KL penalty. Prevents drift.
- **Critic / Value model** $V$: predicts expected future reward from a state. Used in PPO; **not** in GRPO/DPO.
- **Reward model (RM)** $r_\phi$: small LM trained to score (state, action) without running the environment.
- **KL divergence**: distance between two probability distributions. Used as regularizer to keep policy close to reference.
- **Bradley-Terry**: math underlying preference learning. "If A is preferred to B with prob p, then `score(A) - score(B) = log(p/(1-p))`."

---

## Appendix B — Reading list (in order)

If you want to ground in primary sources after this doc:

1. **DPO paper** — Rafailov et al. 2023. The closed-form trick.
2. **DeepSeekMath / GRPO paper** — Shao et al. 2024. The group-relative advantage idea.
3. **Search-R1** — Jin et al. 2025 (COLM 2025). Closest precedent for our retrieval policy.
4. **Mem0 paper** — for the prior-art baseline.
5. **MemGen** — for the continuous-memory contrast you'll cite.
6. **CoSearch** — for the "joint train ranker + reasoner" precedent on co-training.

That ordering takes a weekend. After that you'll be able to read the GRPO/Search-R1 source code directly.

---

*Last updated: 2026-04-27. Living document — extend as you ship.*
