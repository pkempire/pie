# Master running notes — the substance of this repo

*Living synthesis, started 2026-07-06. Built from reading the actual CODE across every subfolder
(4 parallel ingestion passes), not the docs. Grouped: methods · problems · literature · applications
· experiment ideas. Grounded in file paths + real mechanisms. Update this as work continues; supersede
lines, don't append duplicates.*

## The through-line (what the code is actually building)
Across PIE, mempol, and the temporal/ledger layer, the code implements one thing: **learned,
temporally-aware memory with cheap credit assignment.** Three real, working mechanisms carry it — a
learned clock for staleness (PIE survival functions), a learned write policy graded by a per-op
counterfactual made affordable by an amortized critic (mempol), and a validity-native store that
reconstructs "what was true at T" as a query, not a snapshot (temporal). The rest is substrate,
baselines, and products around those.

---

## A. METHODS (the real mechanisms, with novelty vs. the frontier)

**A1. Survival-function staleness clock** — `pie/core/temporal.py`. *The buried standout.*
`survival(entity, t) = P(no state transition yet)`, built from **empirical survival tables**
(no Weibull/exponential assumption) normalized to *entity-relative* time `k = silence/mean_interval`,
with a 3-tier fallback (entity → type → global "universal curve"). Everything derives: `alive()`,
`expected_next()` (median remaining life via binary search), `anomaly()` (timing surprise),
`classify()` (active/dormant/fading/dead). **Novelty:** survival analysis applied to knowledge-entity
staleness is not in any 2026 memory system I've reviewed; it is *literally the agent clock* (temporal
awareness) the field says is missing — computed from learned rhythms, not a prompt timestamp.

**A2. Per-op counterfactual write reward + amortized critic** — `mempol/eval/counterfactual.py`,
`scripts/critic_counterfactual_smoke.py`. For each mutating write op, `_replay()` faithfully
reconstructs the trajectory from `ops_log` minus op_i and diffs the judge score → per-op credit.
Cost is `(K_mut+1)×Q`; the **critic** trains on ~8 exact deltas over *free* features (query-op sim,
retrieval uniqueness, evidence membership, supersession depth) to predict the rest (r≈0.71),
breaking the barrier. **Novelty:** per-op counterfactual itself got published (HiMPO/Memory-R2/Rosetta,
mid-2026) — but all pay replay cost; **the amortized critic is verified-open.** [[project-assets]]

**A3. Temporal validity as a query predicate** — `mempol/temporal/store.py`. `current_states(at=T)`
answers "what was true at T" via SQL range predicates (`valid_from ≤ T < valid_until AND status=active`);
`apply_transition()` marks superseded states and chains `supersedes_state_ids`. **The database is the
timeline** — time-travel without pre-computed snapshots. Read-side twin: `policies/rlm_temporal.py`
reconstructs a dated timeline before answering. **Novelty:** bi-temporal stores exist (Zep); the
*supersession-chain + as-of-T-as-query* framing is clean and underused.

**A4. Write decisions as learnable actions** — `mempol/recipes/memory_rl/write_tools.py`. Ops =
{create/update/merge/add_relation/mark_contradiction/forget}; the policy **must** `lookup_entity`
before `create` — there is **no hardcoded dedup resolver**, so the similarity threshold + when-to-merge
is learned end-to-end by GRPO, graded by downstream QA. **Novelty:** contrast Mem0's fixed
ADD/UPDATE/DELETE LLM-judge and PIE's 3-tier resolver — here resolution is *learned*.

**A5. Hard budget as structure, not cost** — `reader_overlap.py:56-88`. `k_max=12` eviction by
importance *before* scoring makes "store everything" impossible; converts memory into a budgeted-OS
optimization the policy can't brute-force around.

**A6. Learned consolidation via GEPA** — `mempol/dspy_consolidator/` + `scripts/run_gepa_consolidator.py`.
A one-call DSPy `ConsolidateSignature`; GEPA rewrites its docstring/prompt against a metric =
*retrieval-QA success under the consolidated memories* (not generation quality). **Novelty:** GEPA×memory
is verified-open. Current result is n=5 overfit — unproven. [[project-assets]]

**A7. Action selection (the clock → behavior)** — `mempol/temporal/context.py` `_choose_action()`:
interrupt (deadline passed) / replan (expected passed) / refresh (no evidence) / answer. Heuristic
defaults, **GEPA-learnable**. This is where A1's staleness clock would drive proactivity.

**A8. Supporting substrate** — `mempol/core/` (domain-light Artifact/Span/MemoryState/TraceEvent +
lexical-first retrieval, no embeddings required); PIE hybrid retrieval (BM25+dense+RRF+temporal-boost+
1-hop graph, broad-mode query decomposition into 12 sub-queries); dynamics model
(`pie/core/dynamics.py`, Dreamer/MuZero-style next-transition prediction); procedural memory
(`pie/analysis/procedural_memory.py`, n-grams of transition types = behavioral patterns).

---

## B. PROBLEMS (what these methods attack)

- **The agent has no clock** (temporal awareness) → A1 survival staleness, A3 validity, A7 action. Core.
- **Credit assignment for what to remember** → A2 counterfactual + critic, A4 learnable ops.
- **Memory calcifies / can't revise when the world changes** (continual-learning revision) → A3
  supersession chains + A1 staleness are the pieces; the *revision policy* is unbuilt. [[project-thesis]]
- **Extraction is lossy / merges distinct entities** → A4 learned dedup; PIE strict speaker attribution
  (Caroline's ≠ Melanie's, `benchmarks/locomo/baselines.py`) prevents the peer-conversation merge failure.
- **Consolidation is hand-coded** → A6 GEPA-learned.
- **Recall ≠ competence** (studying) → GEPA consolidation + the Cartridges direction. [[memory-field-map]]
- **Benchmarks are broken/expensive** → shared-world-model 200× speedup (`benchmarks/parallel_runner.py`);
  deterministic metrics preferred (TicToc, our `demos/`).

## C. LITERATURE (map methods ↔ papers; full wiki in `research/papers/`)
- Credit/RL: Memory-R1, Mem-α, HiMPO 2606.16285, Memory-R2 2605.21768, Rosetta, DeltaMem; GRPO 2402.03300; COMA (per-op lineage). ↔ A2/A4.
- Temporal: TicToc 2510.23853, Real-Time Deadlines 2601.13206, Zep 2501.13956, Peike Li "three clocks", Tan-Tan-Soatto "Can LLMs Perceive Time?". ↔ A1/A3/A7.
- Consolidation/studying: GEPA 2507.19457, Auto-Dreamer, Cartridges 2506.06266, on-policy distillation (TML), Machine Studying (Jacob Li). ↔ A6, experiments.
- Read-side: RLM 2512.24601, Search-R1 2503.09516, GraphRAG 2404.16130, Generative Agents 2304.03442, MemGPT 2310.08560, A-Mem 2502.12110. ↔ A8, the map.
- Eval: LoCoMo (audited, discredited), LongMemEval (gamed), STALE (vendors 5–8%), HaluMem, BEAM, StudyBench, MemoryAgentBench. Avoid LoCoMo/LongMemEval as primary.

## D. APPLICATIONS / PRODUCTS / CONTENT
- **Architect** (`architect/`) — live component KG + **MCP-Zero active discovery** (wish→retrieve→live-search→ingest→re-retrieve; unmatched wishes = product signal), **two-stage critic** (cheap prompt $0.001 / deep tool-using $0.05), planner loop, freshness decay (`importance·e^(−Δt/30d)`), pattern mining (algorithm ready, awaits data). Genuinely working; the discovery+critic ideas are under-documented.
- **Footnote** (`scripts/footnote/`) — 10-stage GEPA-optimizable video-overlay pipeline; orchestrator + typed artifacts done, stages partly stub.
- **Sales** (`sales/`) — Markov process-mining + what-if deal simulation, Flask MVP. Standalone.
- **BigBrother** (`visual-memory/`) — hackathon: semantic-then-spatial 4D video understanding. Standalone.
- **Content/artifacts** — field map (grid `memory-map.html` + scatter `memory-map-scatter.html` + systems `memory-map-companies.html`), blog "The Shape of Memory", operator buyer's guide, demos 01 (stale-memory) / 02 (TicToc time-as-state). Voice = Distill/researcher. [[project-current-bet]]

## E. EXPERIMENT IDEAS (grounded in what exists; ranked)
1. **Resurrect the survival clock as the temporal-awareness method.** A1 already computes learned
   staleness; wire it as the freshness signal for the TicToc decision (demo 02) — a *learned* clock vs
   raw timestamps. Directly tests "information ≠ awareness" with our own novel primitive. Cheapest, most
   differentiated, on the active bet.
2. **Amortized critic at scale (A2) on Qwen3 via Tinker.** Verified-open; r=0.71 seed exists; warm-start
   from `runs/sft_warmup.jsonl`. Position vs HiMPO/Memory-R2 ("we make per-op credit affordable").
3. **Revisable consolidation** — combine A3 supersession + A6 GEPA + Cartridges: a consolidated artifact
   that can be *superseded* when facts change. The continual-learning revision gap. [[project-thesis]]
4. **Survival-gated consolidation** — use A1 to decide *what to consolidate/forget* (staleness as the
   selection signal) — ties A1↔A6↔A5.
5. **StudyBench / self-study on our own ledger** — A6 + Cartridges on `mempol/ledger/` repo data.
Benchmarks to target: TicToc, STALE (5–8% headroom), StudyBench, HaluMem. Not LoCoMo/LongMemEval as primary.

## F. Asset reality (pointer)
Real-and-ran vs dead is in [[project-assets]]. Newly surfaced as *real and undervalued*: the survival-
function clock (A1), the learnable-dedup ops (A4), temporal-as-query (A3), architect discovery+critic.
Newly confirmed *dead/overfit*: write-RL smokes (zero signal), GEPA 0.6→0.8 (n=5), PIE-KG as SOTA backend.
