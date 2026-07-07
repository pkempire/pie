# Where the map points us — opportunity, models, benchmarks (2026-07-04)

*The field map isn't just content. It's a strategy diagram. This is what it says about where our
novelty is, which open-weight models to build on, which benchmarks are actually winnable, and the
experiments I'm most curious to run.*

## The map is a diagnosis of our own trap

We've spent this whole project in **token space** — PIE's KG, the eval matrix, the timeline reader,
the consolidator. Every one of those lives in the single most crowded cell on the map, competing
with a dozen funded companies whose only real differentiator is polish. The map's central finding is
also our strategic finding: **the token tier is commoditized; the cache and weight tiers are nearly
empty, and they're empty for an accident of access (you need open weights), not because they don't
work.** That accident is now cheap to overcome — Qwen3, Llama, GLM, gpt-oss are strong and rentable
by the hour.

**So the move is simple to state: get out of token space.** Everything novel we've discussed —
amortized critic, label-free next-interaction training, learning-to-study — was blocked because it
needs to touch weights or activations. The map says that block is now the opportunity, not the wall.

## The unlock: Cartridges

Cartridges (Eyuboglu, Arora, … Ré, Stanford, June 2025 — arXiv:2506.06266) is the most important
thing on the map for us, for three reasons:

1. **It's the cache tier's anchor and it's brand new** — one paper, open code, few followers yet.
   The competitive field is empty.
2. **Its mechanism is literally "self-study"** — generate synthetic conversations about a corpus,
   train a small KV cache by context-distillation. That is *our* studying/consolidation thesis, but
   with a proven, working substrate we didn't have to invent.
3. **It composes without retraining** and extends effective context 4×. That property is a research
   goldmine nobody has mined.

Cartridges turns our vague "REM / study while you sleep" vision into something concrete and
trainable *today*, on a rented GPU, with published code as the starting point.

## Five experiments I'm actually curious to run

Ranked by (novelty × feasibility-now × connection to a winnable benchmark).

### 1. Continual / updatable Cartridges — memory that grows
Cartridges are trained once per static corpus. Real memory isn't static — an agent trace grows, a
codebase changes. **Can you incrementally update a Cartridge as new events arrive, or compose a
"base" Cartridge with a small "recent" one, instead of retraining?** The paper says Cartridges
compose without retraining — nobody has tested that as a *continual-learning* mechanism. This is the
cleanest new idea on the board: static self-study caches → living memory. Substrate exists; the
continual angle is open.

### 2. Studying your own agent traces — "optimal memory from a long trace," made real
Take a multi-day agent trace, generate self-study Q&A over it, train a Cartridge, evaluate on
held-out questions about the trace vs. (a) raw-log-in-context and (b) a summary. This is the
concrete, trainable version of the "build optimal memory from a long trace" question the guide
answers in prose — and it's directly our REM loop with Cartridges as the consolidation substrate.

### 3. RL / GEPA over the self-study curriculum
Cartridges generate synthetic study data with a heuristic. **What if we optimize *what to study* —
which synthetic questions to generate — against downstream expertise?** That's learning-to-study
(our idea) applied to a working system, with GEPA (cheap, prompt-space) first and GRPO second. Reward
= held-out accuracy per unit of Cartridge size. Connects our GEPA infra to the cache tier.

### 4. A temporal Cartridge that crushes STALE
STALE (arXiv:2605.06527) tests "what was true at an earlier time"; production memory systems score
**5–8%**. That is enormous headroom. Combine our timeline-reconstruction reader with a Cartridge that
encodes *when* each fact held, and even a modestly good method could hit 40%+ — an order-of-magnitude
jump on a real benchmark. High-headroom, on-thesis (temporal), and a paper on its own.

### 5. The amortized write-utility critic — now unblocked
Our r=0.71 seed needed weight access to matter. On an open model via Tinker it's runnable. Position
against HiMPO/Memory-R2 ("they proved per-op counterfactual credit; we make it affordable"). Highest
prestige, highest risk; do it after 1–2 bank cheaper wins.

## Which open-weight models

- **Qwen3 (4B / 8B / 14B / 32B)** — the default substrate of the entire 2026 memory-RL literature
  (Memory-R1, Search-R1, MemBuilder all use Qwen). Best-supported by Tinker; start here. 4B/8B for
  fast iteration, 14B+ when a result needs to be credible.
- **GLM-4.6 / GLM-class** — strong, and you flagged it; good second model for a "does it transfer"
  check.
- **Llama-3.x / 4** — broad tooling, the safe baseline reviewers expect.
- **gpt-oss-20b** — OpenAI's open model; useful as a third point for generality claims.
- **Cartridges' own setup** — Llama-3.2-3B/8B-class; reuse their config to reproduce before we extend.

Tooling: **vLLM** (serving + free prefix caching + the KV access Cartridges needs), **Tinker**
(LoRA + GRPO; already wired into this repo via tinker-cookbook), the **open Cartridges repo** as the
starting point for experiments 1–4. All runnable on rented H100 time — no lab required.

## Which benchmarks are actually winnable

Avoid the saturated/broken ones as *primary* targets — LoCoMo (audited, 6% wrong key) and
LongMemEval (gamed, saturated) are table-stakes citations, not places to claim SOTA.

Target the high-headroom, on-thesis ones:

| Benchmark | Why it's winnable | Fits which experiment |
|---|---|---|
| **STALE** (temporal staleness) | vendors score **5–8%** — massive headroom | #4 temporal Cartridge |
| **StudyBench** (Machine Studying) | brand new, expertise-per-compute, few entries; *is* the "expert on a corpus" eval | #2, #3 self-study |
| **BEAM** (up to 10M tokens) | deliberately unsaturated (~48–64%); Cartridges' compression is the natural fit | #1 continual Cartridges |
| **HaluMem** (write-quality, op-level) | new; our counterfactual/verification machinery targets exactly this | #5 critic |
| **MemoryAgentBench** | the academic standard; credible to compete on | any, for external validity |

The sharpest single bet: **a temporal Cartridge on STALE (#4)** — largest headroom, on our
strongest theme, one clean result, and a substrate that already exists. Pair it with **StudyBench via
self-study on traces (#2)** for the competence story. Those two, on Qwen3 via Tinker + the Cartridges
repo, are a coherent research program the map directly motivates — and none of it is in token space.

## The one-line version

The map says: stop competing in the crowded token tier; take our studying/temporal/critic ideas —
which were only ever blocked by needing open weights — into the empty cache and weight tiers using
Cartridges + Qwen/GLM + Tinker, and aim the first shots at STALE and StudyBench, where the headroom
is real and the field is thin.
