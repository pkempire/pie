# RL Foundations for the Memory Policy

*A from-scratch teaching pass on what we're actually building, written so you could hand it to a smart 12-year-old and they'd get it. Concrete to our LoCoMo memory-policy setting throughout. No "imagine a robot in a maze" fluff — we're going to walk through one real training step end-to-end.*

---

## Part 1 — The game we're setting up for the LLM

A small LLM (let's say Qwen2.5-3B) is the **player**. The game is: *answer this question correctly using as few "moves" as possible*.

The state at any point in the game is shown to the LLM as a text prompt that looks like this:

```
<task>read</task>
<query>When did Caroline go to the LGBTQ support group?</query>
<budget>tokens=500 retrievals=3</budget>
<recent_ops>
  (none yet)
</recent_ops>
<top_hits>
  (none yet — haven't searched)
</top_hits>
<emit>
```

The LLM's job: emit one move (an "op") as a JSON object. The legal moves in our memory game are:

- `reformulate(rewrite the query)`
- `retrieve(query, k, source)` — search memory, get hits
- `expand(seed_uids, k_per)` — follow links / look at neighbors
- `rerank(strategy, k)` — re-order what we have
- `filter_by_time(window)` — drop stuff outside a time window
- `summarize(items)` — compress
- `stop_and_answer()` — terminal: produce the answer
- `ask_clarification(question)` — terminal: ask the user (rare)

It outputs (say): `{"op": "retrieve", "args": {"k": 12, "source": "hybrid"}}`. The environment runs that move — actually calls `backend.retrieve(...)` — and the new state is fed back in:

```
<task>read</task>
<query>When did Caroline go to the LGBTQ support group?</query>
<budget>tokens=500 retrievals=2</budget>
<recent_ops>
  retrieve(k=12, hybrid) → 12 hits
</recent_ops>
<top_hits>
  [1] D1:3 score=0.81 "Caroline: I went to a support group for LGBTQ youth..."
  [2] D4:13 score=0.62 "..."
  [3] ...
</top_hits>
<emit>
```

LLM picks another move. Repeat. When the LLM emits `stop_and_answer`, the environment:
1. Pulls the current top hits and runs an answer-LLM (gpt-4o-mini) over them to produce a final answer.
2. Compares the answer to the gold answer with an LLM-judge (also gpt-4o-mini), gets a score in {0.0, 0.5, 1.0}.
3. Computes a cost penalty: `λ_step × num_steps + λ_retrieve × num_retrievals`.
4. Returns the **reward**: `score − cost`.

That's the whole game. Concrete example for q0 in conv-26:

| Step | LLM emits | Environment does | New state |
|---|---|---|---|
| 0 | `reformulate` | rewrites query → "Caroline LGBTQ support group date" | recent_ops gains the reformulate line |
| 1 | `retrieve(k=12, hybrid)` | runs hybrid search, returns 12 hits | top_hits filled |
| 2 | `rerank(dense, k=6)` | re-orders by dense similarity to original q | top 6 |
| 3 | `stop_and_answer` | builds answer "Caroline went to the LGBTQ support group on 7 May, 2023" | terminal |

`judge(q, "7 May 2023", answer) = 1.0`. `cost = 0.005 × 4 + 0.01 × 2 = 0.04`. **Reward = 0.96.**

The LLM doesn't know any of this is happening. From its perspective, it's just doing what it always does: predict the next tokens, given a prompt. Our job is to **shape its weight matrix so the next-token distribution leans toward emitting moves that lead to higher reward.**

---

## Part 2 — What we want the LLM to learn

Three things, in increasing order of subtlety:

1. **Output the right move format.** The op JSON must be parseable. (Solved by SFT + constrained decoding — easy.)

2. **Pick moves a good teacher would pick.** When a state looks like "I have a temporal question and no hits yet," reformulate. When it looks like "I have 12 hits but the question is multi-hop," expand. (Solved by SFT — copy the heuristic teacher.)

3. **Discover moves the teacher *wouldn't* pick that lead to higher reward.** This is the only place RL beats SFT. Maybe sometimes the right move is "skip reformulate, go straight to retrieve" — or "retrieve once with k=20 instead of twice with k=10." A teacher would never have shown that, but the LLM, by trying many things and seeing what scores high, can find it. (This needs DPO or GRPO.)

The whole stack is just about getting (3) right. (1) and (2) are setup.

---

## Part 3 — Three training stages, each demystified

### Stage 1: SFT (Supervised Fine-Tuning) — "imitate the teacher"

We have a heuristic policy (`v1_heuristic` in `mempol/policies/`). It plays the game well-enough to get LoCoMo questions right ~70-80% of the time. We collect lots of (state, action) pairs from running it: every question × every step → one example.

A typical training example looks like:

```
PROMPT: "<task>read</task>\n<query>When did Caroline...</query>\n<recent_ops>(none)</recent_ops>\n<top_hits>(none)</top_hits>\n<emit>"
COMPLETION: '{"op": "reformulate", "args": {}}'
```

We fine-tune the LLM with the standard cross-entropy loss on the **completion tokens only** (we mask the prompt so we don't waste training signal teaching it to reproduce its own input).

**Loss per example:**
$$\mathcal{L} = -\sum_{t \in \text{completion}} \log p_\theta(\text{token}_t \mid \text{prompt}, \text{completion}_{<t})$$

That's literally just "predict the next token" — same loss the LLM was pretrained with. You're not doing RL yet. You're doing teacher-imitation in a fancy format.

**What it gets you:** an LLM that outputs valid op JSON, picks moves like the teacher, gets ~70-80% on LoCoMo. **Ceiling = teacher quality. Cannot exceed.**

**In Tinker code (the real thing):** see [tinker-cookbook/recipes/sl_loop.py](https://github.com/thinking-machines-lab/tinker-cookbook/blob/main/tinker_cookbook/recipes/sl_loop.py). The whole script is ~150 lines.

### Stage 2: DPO (Direct Preference Optimization) — "show me the better one"

After SFT, we want the LLM to *exceed* the teacher. SFT can't do that because it's just copying. So we collect a different kind of data: **preference pairs**.

For each prompt, we generate two trajectories by sampling the SFT model with different seeds:
- $\tau_W$ (winner) — got the answer right
- $\tau_L$ (loser) — got it wrong

Now we want: increase the LLM's probability of producing $\tau_W$ over $\tau_L$.

The naive thing would be to compute $\log p_\theta(\tau_W) - \log p_\theta(\tau_L)$ and increase that gap. But that's unstable — the LLM might drift far from its SFT checkpoint and break.

DPO's trick is a single line of math from Rafailov et al.:

$$\mathcal{L}_{\text{DPO}} = -\log \sigma\Big(\beta \cdot \big[\log \tfrac{p_\theta(\tau_W)}{p_{\text{ref}}(\tau_W)} - \log \tfrac{p_\theta(\tau_L)}{p_{\text{ref}}(\tau_L)}\big]\Big)$$

In English: "increase the gap between winner and loser, but measured **relative to the SFT reference model**." If the new model goes too far from the reference, the term inside sigmoid grows badly behaved and gradient pulls back. It's a self-anchoring loss.

**Why this is just classification:** $\sigma$ is sigmoid. The loss is the same shape as binary cross-entropy. You're literally training a classifier whose features are log-probability ratios.

**What it gets you:** an LLM that prefers high-reward trajectory shapes. Beats SFT by 2-5 points typically. Stable, fast, no GPU rollouts during training.

**Limitation:** it's offline. Once you exhaust your (winner, loser) pairs, learning stops.

### Stage 3: GRPO (Group Relative Policy Optimization) — "play the game many times, learn from wins"

This is where the real magic is. And it's actually simpler than DPO conceptually — the only hard part is implementation.

#### The intuition

For each question $q$, we let the LLM play the game $G$ times (usually $G=8$ or $G=16$). Different runs produce different trajectories because we sample with temperature > 0. Some of those $G$ trajectories will get higher reward, some lower.

For each trajectory $\tau_i$ in the group:
- $r_i$ = reward
- $\bar r$ = mean reward over the group
- **advantage** $A_i = r_i - \bar r$

Positive advantage = "this attempt was better than the average attempt on this question."
Negative advantage = "this attempt was worse than average."

(In practice we also divide by the std of rewards within the group — DeepSeek-style normalization. Makes training scale-invariant.)

We then update the LLM's weights so that:
- Trajectories with positive advantage become more likely (push log-prob up)
- Trajectories with negative advantage become less likely (push log-prob down)

The step size is bounded by a "trust region" (clipped surrogate objective, same as PPO) so we don't drift wildly. There's also a KL penalty against the SFT reference for stability.

**The full GRPO objective** (don't memorize this — read the structure):

$$\mathcal{L}_{\text{GRPO}} = -\frac{1}{G}\sum_{i=1}^{G} \min\Big( r_t^i \cdot A_i,\; \text{clip}(r_t^i,\,1-\epsilon,\,1+\epsilon) \cdot A_i \Big) \;+\; \beta \cdot D_{\text{KL}}(\pi_\theta \,\|\, \pi_{\text{ref}})$$

where $r_t^i = \pi_\theta(\tau_i) / \pi_{\text{old}}(\tau_i)$ is the importance-sampling ratio (basically: how much has the policy changed since we sampled this trajectory).

**That's it.** The whole thing is "sample, score, push toward winners, keep KL anchored."

#### Why "Group Relative" matters

Compare to PPO, the classic RLHF algorithm. PPO needs a separate model — the **critic** — that predicts "what reward should I expect from this state?" Then advantage = actual reward − critic's prediction. The critic is hard to train, adds another model copy in GPU memory, and is finicky.

GRPO says: "instead of learning what reward to expect, just sample 8 attempts and use the group average as the baseline." No critic. One less model. The "baseline" comes from data, not from a learned value function.

This is **the** big simplification that made DeepSeek-R1's training tractable. And it works exceptionally well for verifiable reward tasks — like "did this trajectory answer the question correctly?" Which is exactly our setting.

#### The full RL loop, once

For one batch of LoCoMo questions:

1. **Snapshot** the current policy weights → make a sampling client.
2. For each question $q$ in the batch:
   - Build the initial state prompt.
   - Sample $G$ rollouts. Each rollout = a sequence of (op, observation) until terminal.
   - Score each rollout: $r_i = \text{judge}(q, \text{gold}, \text{answer}_i) - \lambda \cdot \text{cost}_i$.
   - Compute advantages: $A_i = (r_i - \bar r) / \text{std}(r)$.
3. **Pack into Datums.** Each datum = (state prompt, sampled tokens, logprobs at sample time, advantages broadcast over the response tokens).
4. **One gradient step.** `forward_backward` computes loss + gradients; `optim_step` applies Adam.
5. **Log metrics**: mean reward, KL to reference, gradient norms.
6. Loop.

That's the entire training loop. The whole `mempol/train/grpo_tinker.py` we sketched earlier is exactly this, ~150 lines.

---

## Part 4 — One real training step, walked through end-to-end

To make this concrete, here's exactly what happens for one batch of one question.

**Setup:**
- LoCoMo question: "When did Caroline go to the LGBTQ support group?" gold = "7 May 2023"
- Backend: `FlatBackend` already ingested with conv-26's 419 turns
- Policy: Qwen2.5-3B with current LoRA weights (some SFT'd state)
- $G = 8$, $\epsilon = 0.2$, $\beta_{\text{KL}} = 0.04$

**Phase 1 — Build the initial state prompt** (in `mempol/policies/op_schema.py`):

```
<task>read</task>
<query>When did Caroline go to the LGBTQ support group?</query>
<budget>tokens=500 retrievals=3</budget>
<recent_ops>(none yet)</recent_ops>
<top_hits>(none yet)</top_hits>
<emit>
```

We tokenize this. Say it's 124 tokens.

**Phase 2 — Sample G=8 rollouts.**

We call `sampling_client.sample(prompt, num_samples=8, sampling_params=SamplingParams(temperature=0.7, max_tokens=128))`. Tinker returns 8 sequences:

```
1. {"op": "reformulate", "args": {}}
2. {"op": "retrieve", "args": {"k": 10, "source": "hybrid"}}
3. {"op": "retrieve", "args": {"k": 5, "source": "dense"}}
4. {"op": "reformulate", "args": {}}
5. {"op": "stop_and_answer"}                       # bad — no info yet
6. {"op": "retrieve", "args": {"k": 20, "source": "bm25"}}
7. {"op": "retrieve", "args": {"k": 10, "source": "hybrid"}}
8. {"op": "filter_by_time", "args": {"window": [...]}} # bad — no items yet
```

Each comes back with the per-token logprob trail. So for sample 2 we have something like:

```
sampled_tokens: [123, 456, 789, ...]   (the JSON encoded as token ids)
logprobs:       [-0.1, -0.3, -0.05, ...]
```

**Phase 3 — Each rollout continues to terminal.**

For each of the 8 first-step samples, we run the rest of the trajectory (currently in our `mempol/rollout.py`, soon to be in the Tinker env):

- Sample 1 (reformulate): then sample step 2, then step 3, ... until `stop_and_answer`. Final answer "Caroline went to the LGBTQ support group on 7 May, 2023." — judge=1.0, cost=0.04.
- Sample 5 (immediate stop_and_answer with no retrieval): answer "I don't have enough info." — judge=0.0, cost=0.005.
- Sample 8 (filter_by_time with no items): no-op effectively, then continues. Maybe ends up at judge=0.5, cost=0.05.

After all 8 finish:

| i | Reward | "How"  |
|---|---|---|
| 1 | 0.96 | reformulate → retrieve → answer correctly |
| 2 | 0.96 | retrieve → answer correctly |
| 3 | 0.95 | retrieve(dense, k=5) → answer correctly with one fewer hit |
| 4 | 0.50 | reformulate → reformulate → answer (judge gave partial) |
| 5 | -0.005 | stop too early |
| 6 | 0.96 | retrieve(bm25, k=20) → too noisy but answers |
| 7 | 0.96 | retrieve(hybrid, k=10) → standard win |
| 8 | 0.45 | wasted move on filter, then OK |

**Phase 4 — Group-relative advantage.**

```
mean = (0.96+0.96+0.95+0.50-0.005+0.96+0.96+0.45) / 8 ≈ 0.715
std  ≈ 0.32

advantages = (rewards - mean) / std
A_1 = (0.96 - 0.715) / 0.32 = +0.77   ← good
A_5 = (-0.005 - 0.715) / 0.32 = -2.25 ← terrible — push hard against
A_8 = (0.45 - 0.715) / 0.32 = -0.83   ← below average
```

**Phase 5 — Pack into Datums.**

For each rollout, we make a `Datum` with three things:

```python
Datum(
    model_input = prompt + sampled_tokens[:-1],           # what the LLM saw
    target_tokens = [0]*prompt_len + sampled_tokens,      # what to score
    logprobs = [0]*prompt_len + recorded_logprobs,        # for importance-ratio
    advantages = [0]*prompt_len + [A_i] * len(response),  # broadcast advantage
)
```

The 0s in front are masks — we only compute loss on the response tokens, not the prompt.

**Phase 6 — One gradient step.**

```python
fwd = training_client.forward_backward(datums, loss_fn="importance_sampling")
opt = training_client.optim_step(adam_params)
fwd.result(); opt.result()
```

Under the hood, the loss is the GRPO objective. Sample 5 (terrible) gets a strong negative gradient on its tokens — its log-prob will go *down* next step. Sample 1 (good) gets a positive gradient — its log-prob goes *up*. The KL term and clip term keep things from going off the rails.

**One batch done.** Move to next question. After ~2000 batches, the policy has internalized: "for temporal questions, retrieve immediately. For multi-hop, expand. Don't stop without retrieving. Don't filter empty sets."

---

## Part 5 — What Tinker abstracts vs what we still write

| Layer | Who owns it | Notes |
|---|---|---|
| **GPUs** | Tinker | Their managed cluster runs forward / backward / sample. Your code runs on a CPU. |
| **LoRA adapter** | Tinker | You pass `rank=32`. They handle creation, serialization, attention to base weights. |
| **Sampling efficiency** | Tinker | vLLM under the hood. Prefix caching free. |
| **Checkpoint persistence** | Tinker (with helpers) | `save_weights_and_get_sampling_client()`, `save_state`. |
| **Tokenization** | Tinker (via `tinker_cookbook.tokenizer_utils`) | Just call `get_tokenizer(model_name)`. |
| **Renderer / chat template** | Tinker (`renderers/`) | Knows how Qwen / Llama / etc. format messages. Plug-and-play. |
| **The training loop logic** | **You** | The `for batch_idx ...` with sample → score → datum → fwd → step. |
| **State encoder** | **You** | What goes into the prompt, format, budget tokens. |
| **Action space schema** | **You** | The op JSON, the constrained decoding grammar. |
| **Reward function** | **You** | `judge(q, gold, ans) - cost`. The most important thing in the whole project. |
| **Environment** | **You** | The thing that takes an op, applies it, returns observation. |
| **Memory backends** | **You** | Flat / Tree / Graph / etc. |
| **Eval harness** | **You** | LoCoMo runner, TemporalBench, etc. |

The slogan: **Tinker abstracts the pain (distributed GPUs), keeps the knobs (you control the algorithm).** Compare to OpenAI's fine-tuning API which abstracts both — you can't do GRPO there, you can only SFT a fixed objective.

---

## Part 6 — The Search-R1 recipe is *exactly* our paper's substrate

Tinker just released a Search-R1 replication recipe at [`tinker_cookbook/recipes/search_tool/`](https://github.com/thinking-machines-lab/tinker-cookbook/tree/main/tinker_cookbook/recipes/search_tool). It is almost literally what we want.

| Their files | What it does | Ours becomes |
|---|---|---|
| `search_env.py` | Multi-turn RL env: state = question + retrieved context, actions = "search" or "answer" | `mempol/env/memory_env.py` — state = question + retrieved memory, actions = our 8-op vocab |
| `tools.py` | `@tool` decorator over `ChromaTool.search()` and `answer()` | Our memory ops (`retrieve`, `expand`, etc.) as `@tool` definitions |
| `embedding.py` | Gemini embeddings + Chroma vector index | Our existing `mempol.llm.embed` + `FlatBackend` / `TreeBackend` |
| `train.py` | The GRPO loop tying it together | `mempol/train/grpo_tinker.py` with same shape, our env+tools |
| `offline_eval.py` | Eval on NQ / TriviaQA / HotpotQA / 2Wiki | Our LoCoMo runner |

**Their results** (from the README):

| Benchmark | Original Search-R1 paper | Tinker replication |
|---|---|---|
| Natural Questions | 42.9 | **51.6** |
| TriviaQA | 62.3 | **67.3** |
| HotpotQA | 38.6 | **49.7** |
| 2Wiki | 34.6 | **42.8** |

The Tinker GRPO recipe **beat the original paper on every benchmark** with the same model. That's our existence proof that this approach — Qwen 2.5/3 + GRPO over multi-turn retrieval — works. We just need to swap the search domain for personal-memory domain.

**Our concrete plan in light of this:**

1. Clone the recipe directory structure into `mempol/recipes/memory/`.
2. Replace `search_env.py` with our memory env (LoCoMo question → memory ops → answer).
3. Replace the Chroma+Gemini embedding stack with our `FlatBackend` (which already does dense + BM25 + RRF).
4. Replace the training set (NQ/TriviaQA/...) with LoCoMo train + LongMemEval train.
5. Run their `train.py` shape with our env. First number lands in <1 week of Tinker access.

This is *much* faster than building from scratch. The recipe even handles the gnarly bits like context-limit truncation in multi-turn rollouts and async sampling efficiency.

---

## Part 7 — Synthetic data: what we need and the principled way to make it

Three kinds of training data, each for a different stage:

### (A) SFT data — "what the teacher does"

For every (state, action) pair the heuristic teacher emits while running on LoCoMo / LongMemEval / MSC train splits.

**Source:** already automatic. `mempol/eval/runner.py` writes traces JSONL. Each trace step is one (state, action) pair.

**Volume:** ~5500 examples per LoCoMo run. Plus LongMemEval (~2000), MSC (~3000). ~10K total — plenty for LoRA SFT.

**Synthesis required:** none. Just collect.

### (B) Reward data for the read policy — "did the answer match?"

Standard QA pairs. Already in LoCoMo / LongMemEval — `(question, gold_answer)`. Reward is computed by the LLM-judge at rollout time.

**Source:** existing benchmarks.

**Synthesis required:** none.

### (C) Reward data for the write policy — "was this turn worth remembering?"

This is the only thing we need to synthesize, and it's the load-bearing piece for the joint write+read paper.

**Why we need it:** to train the write policy, we need a signal of "if I had stored this turn, would future questions have been answerable?" That's a counterfactual — it depends on future queries we don't have.

**The original miner I shipped used a hand-coded regex of back-reference phrases ("going back to", "earlier you said", etc.). That was wrong** — it's exactly the kind of hand-coded heuristic the paper's thesis attacks. A turn might be necessary for a future query without anyone saying "going back to" — the user can just re-ask in different words.

### The principled replacement: counterfactual necessity mining

Drop all regex. The criterion for "is this turn worth remembering" is **functional**: would removing this turn from the conversation history hurt the model's ability to answer future questions?

For each candidate antecedent turn $t$ in a conversation:

1. **Generate a synthetic future query.** Ask an LLM: "Given this turn, generate a follow-up question a user might naturally ask later that would require knowing the content of this turn. If the turn is too generic to support a meaningful follow-up, say so."
   - If the LLM says "too generic," skip.
   - Otherwise we get $(Q, \text{gold})$.

2. **Counterfactual answer test.** Build two versions of the conversation context:
   - $C^+$ = conversation including turn $t$
   - $C^-$ = conversation excluding turn $t$
   Run the answer LLM on both: `answer_with = LLM(Q, C+)`, `answer_without = LLM(Q, C-)`.

3. **Score both with the judge.**
   - $s^+ = \text{judge}(Q, \text{gold}, \text{answer\_with})$
   - $s^- = \text{judge}(Q, \text{gold}, \text{answer\_without})$

4. **Necessity gap.** $\Delta = s^+ - s^-$.
   - $\Delta > 0.5$: turn $t$ is functionally necessary for $Q$ — emit triple.
   - $\Delta \le 0.5$: turn $t$ wasn't needed — discard.

This is principled because:
- No lexical patterns. The criterion is "did removing this turn change the answer."
- It's exactly the right reward signal for write-policy training: "should I store this turn?" = "is there a plausible future query for which it's necessary?"
- It generalizes across phrasings, languages, conversation styles.
- The threshold (0.5) is the only knob, and it has a clear meaning (full-vs-half-correct gap).

Cost: 4 LLM calls per candidate (1 generate, 1 answer-with, 1 answer-without, 1 judge — gold can be derived from generate). At gpt-4o-mini ~$0.001/call → $0.004/candidate. To produce 5K positive triples at ~50% acceptance → 10K candidates → ~$40. Affordable, one-time.

Implementation: see `mempol/data/necessity_miner.py` (replaces the deleted `backref_miner.py`).

### How this data feeds training

For each `(turn t, query Q, gold)` triple:

- **Write-policy reward (Phase B of co-training):** when training W, sample G rollouts for the conversation containing $t$. For each rollout, run the read-policy on $Q$ using the memory state W produced. The advantage is whether the read-policy got $Q$ right. Turns that W "stored" and that helped → positive advantage. Turns it stored and that didn't help → negative.
  
- **Read-policy reward when isolated:** $Q$ becomes another QA item in our train pool, alongside LoCoMo. Just adds breadth.

That's the full data picture. There is no other synthetic data we need.

---

## Part 8 — The smallest mental model you can leave with

If you remember nothing else from this doc:

1. **The LLM is a player. Each prompt is a game state. Each completion is a move.**
2. **SFT** = "copy the teacher" via standard cross-entropy on (state, teacher_move) pairs. Stable, ceiling-bounded.
3. **DPO** = "preferred move beats unpreferred move" via a self-anchoring binary classifier loss. Stable, offline.
4. **GRPO** = "play 8 times, push toward the ones that scored above the group mean." Online, needs a reward function, no critic. Used by DeepSeek-R1 and Search-R1.
5. **Tinker** runs the GPUs that compute forward/backward/sample; you write the loop, the env, the reward, and the data.
6. **Our reward** = `judge(question, gold, answer) − λ × cost`. Both pieces are LLM calls in our setup.
7. **The only thing we synthesize** is `(turn, future_query, gold)` triples for the write-policy reward, via counterfactual necessity mining (no regex).
8. **The Search-R1 Tinker recipe is the existence proof.** Their numbers beat the original paper. We're doing the same thing applied to personal memory instead of Wikipedia.

---

## Part 9 — Reading order if you want to go deeper

1. [Tinker `rl_loop.py`](https://github.com/thinking-machines-lab/tinker-cookbook/blob/main/tinker_cookbook/recipes/rl_loop.py) — read this first, it's <250 lines and shows the whole GRPO loop.
2. [Tinker `search_tool/train.py`](https://github.com/thinking-machines-lab/tinker-cookbook/blob/main/tinker_cookbook/recipes/search_tool/train.py) — multi-turn RL, our actual template.
3. [DeepSeekMath / GRPO paper §4](https://arxiv.org/abs/2402.03300) — the formal derivation, ~3 pages.
4. [Search-R1 paper §3](https://arxiv.org/abs/2503.09516) — multi-turn retrieval RL formulation.
5. [DPO paper §4](https://arxiv.org/abs/2305.18290) — the closed-form derivation of why DPO works.

That's enough to walk in to a research team and contribute on day one.

---

*Last updated 2026-04-27. Ground truth lives in `mempol/`; this doc is the conceptual companion.*
