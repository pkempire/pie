---
title: "Orientation + Next-Bet synthesis"
date: 2026-06-21
author: "comb-through pass (Claude) for Parth"
status: "DRAFT — §10 (literature landscape) and §11 (the bet) finalized after the mid-2026 lit refresh"
purpose: "One honest map of what actually exists, what works, what's vapor, and where to point next. Reconciled against PROJECT_ARTIFACT_INVENTORY.md, the openai_handoff/ postmortem, and a fresh full read of the code + every result JSON on disk."
---

# Orientation + Next-Bet — 2026-06-21

> This doc is the output of a full comb-through: ~7 deep passes over `mempol/`, `research/`,
> `docs/`, `paper/`, every `results/*/summary.json`, plus a mid-2026 literature refresh.
> It is written in your house style: honest over agreeable, numbers over adjectives,
> shipped-artifact over plan. Where a number is shaky, it says so.
>
> The single most important sentence: **your docs are 6–12 months ahead of your code, your
> paper sells a thesis you already killed, and your one genuinely interesting *positive*
> result (read-time timeline synthesis, 71.7%) is not the thing any of your write-ups lead with.**

---

## 1. The corrected reality (read this first)

Three facts that the HANDOFF (May 30) understates or misses:

1. **Git HEAD is frozen at May 2** (`f0c90e3`). *Everything* built since — `core/`, `ledger/`,
   `temporal/`, `applications/`, the GEPA result, `gepa_state.bin`, the continuity-teacher
   results, all `universal_*` RL code — is **uncommitted/untracked**. The `git am` session
   is still wedged. **This is the only emergency in the repo.** One bad `rm` or `git reset`
   and the last 7 weeks vanish. Fix before anything else (§7, R0).

2. **The project has pivoted three times**, and the artifacts from each era still coexist and
   contradict each other:
   - **PIE era** (Feb): typed temporal KG over personal exports. → demoted.
   - **mempol RL era** (Apr–May): per-op counterfactual write reward. → *you* declared it
     dead in `paper/openai_handoff/` (Jun 18). Zero real results ever produced.
   - **Temporal Context Engine / Research Ledger era** (May 23–Jun 16): the current north
     star. Real substrate shipped; flagship capability unproven on real data.

3. **The tooling environment changed since the Jun-6 code review.** `tinker`, `tinker_cookbook`,
   and `dspy` are now installed; the train CLIs now import clean. The binding constraint is no
   longer "no tinker" — it's `OPENAI_API_KEY` being unset + needing tinker compute to actually
   train. So "it imports" still ≠ "it runs."

---

## 2. Timeline of the actual work (filesystem mtimes, not git)

| When | What landed | Era |
|---|---|---|
| Feb 2026 | PIE world-model, benchmarks, `ARCHITECTURE-*`, MEMORY-LANDSCAPE | PIE |
| Apr 27–May 2 | mempol RL (`recipes/memory_rl/write_*`), per-op counterfactual reward, `paper/main.tex`, Pace proposal, architect/ | mempol-RL |
| **May 23** | `mempol/core/` — Universal Memory Core (Artifact/Span/MemoryState/TraceEvent) | TCE |
| May 25 | `dspy_consolidator/`, `universal_*` RL recipe | TCE |
| Jun 6 | `policies/rlm_temporal.py`; the brutal code review | TCE |
| **Jun 12** | `mempol/ledger/` — Research Ledger (repo+git ingest → SQLite) | TCE |
| Jun 13–14 | `mempol/temporal/` — Temporal Context Engine (store + read-time compiler) | TCE |
| Jun 15–16 | `applications/registry.py`, `policies/continuity.py`, continuity-teacher results | TCE |
| **Jun 18** | `paper/openai_handoff/` — the postmortem that kills the old thesis | TCE |

---

## 3. Honest status of every workstream

Legend: ✅ shipped & verified · 🟡 runs but tiny/unproven · 🔴 vapor / never ran on real data · ⚫ dead/abandoned

| Workstream | Status | The honest one-liner |
|---|---|---|
| **Research Ledger + Universal Core** | ✅ | Genuinely ingested this repo: 541 artifacts, 3124 spans, 4165 memberships, 16 days, correct parsed metrics. Disciplined, idempotent, tested. The real shipped thing. |
| **LongMemEval strategy matrix** | ✅ | n=240 balanced: **Timeline-Synthesis 71.7% > Turn-RAG 68.3 > Hybrid 62.5 > Rerank 61.7.** The one real *positive* finding. Read-time, not learned-write. |
| **LoCoMo matrix** | ✅ | n=1491: **flat 58.6 > flat-v1 53.6 > PIE-KG 46.2** (KG evidence_recall = 0.0). A real, honest *negative*: brittle typed-KG extraction loses to plain hybrid RAG. |
| **GEPA consolidator** | 🟡 | 0.6→0.8 **but train==val, 5 questions, 1 chunk.** ~29.5 hr / 1000 metric calls for that delta. The proper held-out run (train4/val4/q30) **crashed → 0-byte summary.** Not citable. |
| **Continuity-teacher (LongMemEval)** | 🟡 | 100%@n=6 → **collapses to 73.3% tie with timeline-synthesis @ n=120**, costing 7.3 tool-steps vs 0. Value = trace generator for distillation, not a better system. |
| **Temporal Context Engine** | 🔴 | The *named flagship*. Schema + store + read-time compiler exist and are unit-tested, but **never run on real data once.** The only on-disk "demo" hit the empty-store guard (action: refresh, nothing retrieved). |
| **Per-op counterfactual write RL (Phase B)** | 🔴 | The entire `paper/main.tex` thesis. Every RL run on disk is a **zero-signal smoke** (`std_reward = 0`). Reward collapses to 0 whenever the frozen reader can't answer; `w_cov_floor=0.05` exists only to paper over it. |
| **Phase A read-policy RL** | 🔴 | Never trained. R has been hand-coded `HeuristicPolicy(8,4)` the entire project. |
| **Trained-R-as-judge / co-training** | ⚫ | `make_tinker_r_runner` is a hard `NotImplementedError`; `cotrain.py` eval is a `TODO`. The co-training story is non-functional in code. |
| **`universal_*` RL (new substrate)** | 🔴 | Cleaner design (4 ops, single terminal judge, `freeze_raw_access` shaping). Wired, imports clean, **never run as RL.** Substrate retrieval recall is 0.15–0.31 → would be retrieval-bottlenecked before the policy learns. |
| **Critic counterfactual** | 🟡 | Toy: Pearson 0.707, MAE 0.03 on tiny sampled deltas. Seed for a cheap utility critic to replace brute-force counterfactual. |
| **`paper/main.tex`** | 🔴 | Fully scaffolded, 100% `[TBD]` result tables, sells the dead per-op thesis. Contradicted by your own Jun-18 handoff. |
| **PIE + `mcp_server.py`** | ✅ (maintain) | Live in Claude Desktop. Underperforms as a benchmark backend; fine as personal MCP + data source. |
| **architect/** | 🟡 | Streamlit + planner + component DB works, no users. |
| **Footnote** | 🟡 | MVP all 7 stages coded, never smoke-tested on a real video. Separate product. |

---

## 4. The real assets (what you can actually build on)

Stripped of aspiration, here is what genuinely exists and works:

1. **A working, balanced LongMemEval + LoCoMo eval harness** (`mempol/scripts/{longmemeval,locomo}_matrix.py`)
   with a 3-bucket judge, multiple strategy cells, and real n=240 / n=1491 runs. This is
   non-trivial infrastructure most people don't have.
2. **"Timeline Synthesis"** — a read-time method that reconstructs a dated timeline before
   answering, **71.7% on LongMemEval-S balanced (beats Turn-RAG and Hybrid).** This is your
   best card and almost nothing you've written leads with it.
3. **The Research Ledger** — real repo/git → SQLite ingestion with provenance, day-reports,
   and a context-pack compiler. Genuinely ran on this repo.
4. **The temporal substrate** — `TemporalState` with valid-time/supersession + a
   `decision_training_rows` join (decisions ↔ outcomes) ready for offline policy learning.
   Schema is good; it just hasn't met data.
5. **A curated 33-paper research wiki** with strict provenance discipline (abstract-verified
   vs table-only vs third-party). Rare and valuable as a related-work / content base.
6. **The honest negative results** — "typed-KG loses to flat RAG," "tiny-n wins evaporate at
   scale." These are publishable *findings*, not failures to hide.
7. **Read-side prototypes + the RLM reference program.** You have a `rlm_temporal_reconstruction`
   demo (flat 66.7% → reconstruction **83.3%** on temporal as-of questions, synthetic) and
   `policies/rlm_temporal.py`. Sitting next to it in `external/longmemeval-rlm/` is a full
   13-experiment RLM-on-LongMemEval program reaching **89.8%** (DSPy + Pydantic observational
   memory, Gemini-3-Flash, ~$0.035/q) — **note: this is the external raw.works RLM work, a
   reference clone, not your result.** But it's the strongest read-side recipe in the tree and
   its findings ("DSPy scaffolding 58%→87%", "context rot kills you when you add tools",
   "typed output > prompt instructions") are directly reusable.
8. **Two under-filed real assets in junk paths:** `output/experiments/critic_counterfactual.json`
   (the Pearson-0.707 utility-critic seed — the empirical core of the *current* paper, oddly not
   under `paper/`) and `runs/sft_warmup.jsonl` (a real cold-start SFT dataset with the full
   write-tool schema — the warm-start for any write-policy RL). Plus `backends/gitmem.py`, an
   untracked content-addressable/branchable memory backend that directly attacks PIE's three
   weaknesses (atomic bundles, branching contradictions, time-travel).

---

## 5. The current north star (vision vs shipped)

**Vision (from `docs/TEMPORAL_CONTEXT_ENGINE_TECHNICAL_SPEC.md`, Jun 13):** one system — a
*context+action compiler* for long-running agents — learning a policy
`πθ(C, a | T≤t, q, now, B)` that, given all prior traces, a task, the wall clock, and a
token/tool/latency budget, emits a structured **context pack** *and* an **action**
∈ {answer, refresh, wait, interrupt, replan}. Reward = task success + evidence support +
temporal validity + correct action timing − costs. Memory reframed from "what text exists"
to "what changed / what's still true / what's stale / what should happen next, with evidence."
First wedge: **long-running project/repo continuity**. The decisive bar is *cross-benchmark
generality*: one substrate that improves recall (LongMemEval/LoCoMo) **and** timing
(TicToc/Robotouille) **and** a product eval (repo-continuation) simultaneously.

**Shipped:** the substrate (core+ledger+temporal SQLite) and the harness. The *learned*
compiler, the critic, the action policy, the consolidator, the repo-continuation benchmark,
and the Pareto frontier — all unbuilt. Your own docs say it: *"until there is a Pareto table,
there is no paper."* That table does not exist yet.

---

## 6. Objectives the work implies (consolidated)

- **O1.** A learned, budgeted *context+action* compiler that beats raw-RAG / prompted-extraction
  / typed-KG on a **Pareto frontier** (task accuracy vs memory+retrieved tokens vs cost vs staleness).
- **O2.** Replace brute-force per-op counterfactuals with a **cheap learned utility critic**
  (the 0.707-Pearson toy is the seed).
- **O3.** Show **temporal validity/transition structure** beats current-state-only specifically
  on knowledge-update / stale / before-after question categories.
- **O4.** Make the repo itself the first dataset: a **repo-continuation benchmark** + the
  `decision_training_rows` join as the offline-RL/SFT corpus.
- **O5.** Distill the hand-written **continuity-teacher trace** into a cheaper learned read/write/action policy.
- **O6.** Ship the **research wiki + Working Memory ep 1** as the public/reputation artifact.

---

## 7. Problems & risks (ranked)

- **R0 — Uncommitted work (URGENT, non-research).** 7 weeks of code is untracked behind a stuck
  `git am`. Abort/continue the am, stage the new trees in logical commits, push. Until then every
  other item is at risk. (HANDOFF §7 has the commit grouping.)
- **R1 — The paper sells a dead thesis with zero results.** `main.tex` is 100% `[TBD]` and built
  on per-op counterfactual, which you killed in `openai_handoff/`. The Pace proposal is *staler
  still* (leads with "evidence coverage," a reward you set to zero). Don't submit/cite either as-is.
- **R2 — The headline GEPA number is a train==val, n=5 overfit.** Do not put 0.6→0.8 anywhere
  external until the held-out run (train≠val, ≥30 Q) actually completes. The one attempt crashed
  to a 0-byte file.
- **R3 — The flagship (Temporal Context Engine) has never touched real data.** Highest
  vision/reality gap in the repo. It's all scaffolding for a learned policy that doesn't exist.
- **R4 — Two disconnected memory stacks.** New `core.SQLiteMemoryStore` (lexical) and old
  `backends/FlatBackend` (hybrid/embedding) don't import each other. Benchmark numbers come from
  the *old* stack; the "universal" substrate has poor recall (0.15–0.31) and isn't wired to evals.
  Decide: reconcile or keep separate.
- **R5 — Fabricated-looking citations.** `applications/registry.py` and `refs.bib` contain
  future-dated/implausible arXiv IDs and 7 `author={Anonymous}` entries + duplicate keys. Will
  propagate as hallucinated refs if they feed a paper or planning agent. Verify or strip.
- **R6 — Stale docs contradict each other.** `QUALITY-REVIEW.md` calls PIE "9.2/10 production-ready";
  every later doc says PIE underperforms. Archive/flag the Feb snapshots.
- **R7 — Baseline completeness.** `BENCHMARK-AUDIT.md`: missing BM25/Contriever/Stella/time-aware
  baselines means the matrices aren't yet paper-grade comparable to the LongMemEval/LoCoMo originals.

---

## 8. Open questions to verify

- Does **bootstrap_repo + compile_temporal_context** actually produce a correct point-in-time
  context pack on real repo history? (The single most load-bearing unproven claim.)
- Does **Timeline-Synthesis's 71.7%** hold on the *full* LongMemEval-S (not just the 240 balanced),
  and against a full-context oracle + strong rerank baseline?
- Will **trace_events / context_decisions** ever accumulate at the volume needed to train a policy?
  (Currently ~1–2 rows per run — the training corpus is essentially empty.)
- Is the **utility critic (Pearson 0.707)** real signal or small-sample luck? Does it hold at scale?
- Does the **`freeze_raw_access`** shaping term actually force causal memory, or just suppress answers?
- **Does memory architecture even matter at frontier-model scale?** (Your recurring doubt:
  Hindsight's vector baseline beat its own KG; Amory +1.6pp over full-context.) → see §10.

---

## 9. Product ideas surfaced (for later, not the research bet)

- **Universal Context Compiler** — global hotkey detects active app/file/repo → compiles a cited
  context pack → preview sources → inject into Claude/Codex/GPT. The ledger's `compile_context`
  already emits exactly this. "Trust is the product wedge; the dashboard is the product."
- **Research OS / Corpus Compiler** — import papers+repos → "what's the frontier / is my idea novel
  / what did we try and why did it fail." Trustworthy deep research that compounds across sessions.
- **Footnote** — AI video annotation; separate shippable product, needs a real smoke.
- **Lucid → research residency** — the $12k 1-on-1 LLM-memory residency; the curriculum literally
  assigns LongMemEval → Mem0 → Auto-Dreamer (students would work your Goal-01 in parallel).

---

## 10. Mid-2026 literature landscape  — _[FINALIZE AFTER LIT REFRESH]_

> Deep-research workflow `wqvnls3ua` running. Will populate: updated LoCoMo/LongMemEval/ScienceWorld
> SOTA tables with dates + verification; which of your beliefs are now stale; and crowded-vs-open
> niches as of June 2026 (esp. GEPA×memory, offline-consolidation-RL, temporal-validity memory,
> learned read-time recursion).

---

## 11. THE NEXT BET — recommendation  — _[FINALIZE AFTER LIT REFRESH]_

> The framing is locked; the pick depends on what the literature says is still open. The candidates,
> ranked by current evidence-strength × defensibility:
>
> **A. Lead with read-time Timeline Synthesis** (your real 71.7% positive result). Cheapest path to
>    one defensible, reproducible benchmark result + a paper/video. Scale to full LongMemEval-S,
>    add strong baselines (full-context oracle, rerank), report a budget curve.
> **B. Offline consolidation + GEPA** (Goal-01, done right). Finish the held-out GEPA run; if it
>    holds, "reflective prompt evolution for memory consolidation at ~1/35th the compute of GRPO"
>    is a clean story — *iff* the niche is still open (Auto-Dreamer got close; check GEPA×memory).
> **C. Temporal-validity as the differentiator** — prove transition/validity structure beats
>    current-state-only on knowledge-update/stale categories; pair with the repo-continuation eval.
>    Highest ceiling, most unbuilt.
>
> Recommendation + sequencing finalized in the response once the lit review lands.

---

*Reconciles: `PROJECT_ARTIFACT_INVENTORY.md`, `openai_handoff/ai_memory_handoff_report.tex`,
`TEMPORAL_CONTEXT_ENGINE_TECHNICAL_SPEC.md`, `APPLICATION_EVAL_MAP.md`, and a full read of every
`mempol/results/*/summary.json`. Supersedes the optimistic framing in HANDOFF.md §0 where they conflict.*
