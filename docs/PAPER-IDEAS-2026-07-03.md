# Publishable paper directions — portfolio, 2026-07-03

*Fourteen paper proposals grounded in the July-2026 frontier review and this repo's real assets.
Each: the gap, the thesis, method, experiments+baselines, novelty vs current work, risk, what we
already have, effort, venue. Ends with a portfolio spine — which share infrastructure and which
order to do them in.*

Legend for **What we have**: ✅ exists & ran · 🟡 exists, unproven · 🔴 to build.
Legend for **Risk**: how likely the central claim survives a real experiment.

---

## Tier A — Flagship (full NeurIPS/ICLR/ACL, needs a strong result)

### A1. The amortized write-utility critic
**Gap.** Per-op counterfactual credit for memory writes is now published (HiMPO 2606.16285,
Memory-R2 2605.21768, Rosetta 2606.07711) — but *all* pay the `(K+1)×Q` replay/rescoring cost at
training time. Nobody replaces the replay with a learned value model. Verified-open.
**Thesis.** A learned critic predicts a write's counterfactual marginal utility from
(write, memory-state, question-battery) features in one forward pass, recovering ≥90% of true
per-op credit at a fraction of the compute — making per-op-credit RL affordable.
**Method.** (1) Collect exact leave-one-out deltas on a sample of trajectories. (2) Train the
critic (regression head over pooled write+state embeddings, or a small tuned model). (3) Use
critic-predicted advantage inside GRPO; periodically refresh with true replays (Dyna-style
corrective loop) to prevent drift/gaming. (4) Report the compute–quality frontier.
**Experiments / baselines.** Critic-vs-true correlation as sample scales; wall-clock and $/step
vs HiMPO/Memory-R2 replay; final policy quality at *matched compute* (the money plot); ablate
refresh frequency; adversarial check that the policy can't game the critic.
**Novelty.** HiMPO explicitly advertises *no critic*; this is the one differentiated wedge into
the crowded memory-RL literature — you complement rather than compete.
**Risk.** High. The r=0.71 toy seed must survive scale-up and resist gaming.
**What we have.** 🟡 critic seed (`scripts/critic_counterfactual_smoke.py`, r=0.71), the write
env + ops_log for exact replays, tinker GRPO loop.
**Effort.** 6–10 weeks + ~$400 compute. **Venue.** ICLR/NeurIPS main.

### A2. REM — the three-layer continual-learning hierarchy
**Gap.** Every 2026 system stops at token-space consolidation, hand-codes the consolidator, and
measures recall not competence. Nobody built the full CLS hierarchy (log→notes→KV→weights) with an
optimized studier and a competence metric.
**Thesis.** A hierarchy that graduates knowledge downward by stability×utility learns continually
and is measurably better week-over-week at *fixed inference cost*.
**Method.** The full [REM spec](REM-SPEC-2026-07-03.md): nightly studying (L1) → KV compile (L2)
→ utility-gated distillation (L3) → fixed-probe eval. Dogfood on this repo + StudyBench.
**Experiments.** Expertise-per-compute curve over N nights; ablate each layer; vs
retrieval-only and long-context baselines.
**Novelty.** The literal hippocampus→cortex loop with an efficiency objective; Letta's "memory
models" manifesto describes the vision and ships nothing.
**Risk.** High (big build); de-risked by the kill-criteria in the spec (ship token-space REM if L3
fails — still a paper).
**What we have.** 🟡 L1 pieces (ledger, temporal store, GEPA); 🔴 L2/L3.
**Effort.** The 90-day arc. **Venue.** Strong empirical paper / MLSys / NeurIPS.

---

## Tier B — Method papers (Findings/main, one clean result each)

### B1. Reflective prompt evolution for memory consolidation (GEPA studier)
**Gap.** No GEPA×memory paper. MemPro (2606.00619) took the TextGrad slot; the GEPA slot is empty.
**Thesis.** GEPA-evolved consolidation prompts match or beat RL-trained consolidators
(Auto-Dreamer-style) at ~1/35 the rollouts.
**Method.** GEPA over the consolidator prompt against downstream QA/expertise; head-to-head with
hand-coded, GRPO-trained, and TextGrad/MemPro.
**Experiments.** LongMemEval-S + StudyBench; the compute–quality frontier is the headline.
**Novelty.** Verified-open; the "35× cheaper than RL" claim ported to memory.
**Risk.** Medium (Auto-Dreamer is close on the RL side; comparison must be clean).
**What we have.** 🟡 `dspy_consolidator/` + `run_gepa_consolidator.py` + `gepa_live.py`; held-out
run is ~$100. **Effort.** 3–5 weeks. **Venue.** ACL/EMNLP Findings, COLM.

### B2. Studying produces *temporal* expertise
**Gap.** Machine Studying (jacobxli.com, 2026) proved studied cheatsheets beat fine-tuning — but
on *static* corpora. Fast-moving domains (deprecated APIs, versioned docs) are unaddressed.
**Thesis.** For evolving domains, expertise requires *temporal* consolidation; a validity-aware
studied artifact beats a flat cheatsheet on version-sensitive, post-cutoff questions.
**Method.** DSPy corpus (real API deprecations) as testbed; flat cheatsheet vs temporal cheatsheet;
score on version-sensitive questions at matched study/inference budget.
**Novelty.** Combines your temporal work with the field's freshest framing (June 2026); directly
comparable to StudyBench's published curves (same base model).
**Risk.** Low–medium. **What we have.** 🟡 temporal schema; 🔴 the study harness.
**Effort.** 3–4 weeks. **Venue.** COLM / workshop → main.

### B3. Reference-class forecasting for agents (the planning-fallacy fix)
**Gap.** LLMs estimate task cost in human-team-days with no grounding; the memory→planning link is
thin in the literature. Your `planning_fallacy.json` already shows GPT planning 8 days for a
30-minute script.
**Thesis.** Injecting an agent's own (task, estimate, actual, outcome) records at plan time
debiases duration/step/cost estimates and improves plan feasibility.
**Method.** Build the calibration table from the ledger; retrieve analogous past tasks (RCF);
measure estimation error and downstream success on DeepPlanning-style constrained tasks.
**Novelty.** Planning fallacy in LLMs is under-measured; nobody ships the calibration artifact.
**Risk.** Medium (needs a task suite with ground-truth actuals — the dogfood repo provides them).
**What we have.** 🟡 ledger records actuals; 🔴 the RCF injector + eval.
**Effort.** 3–5 weeks. **Venue.** ACL/EMNLP; agents workshop.

### B4. Utility-gated distillation — *what* to move into weights
**Gap.** RL-memory papers train policies; nobody studies the *selection* problem of which learned
knowledge should graduate to weights vs stay in context.
**Thesis.** Distilling only stable×used knowledge (critic-selected) beats distill-everything and
distill-recent on retention-vs-staleness at matched LoRA capacity.
**Method.** On-policy distillation with a utility gate (the A1 critic reused); compare gates
(recency, frequency, utility, random); measure retention, staleness, forgetting.
**Novelty.** Recasts the critic as a graduation policy — a second paper from the same machinery.
**Risk.** Medium. **What we have.** 🟡 critic seed, tinker distillation recipes.
**Effort.** 5–7 weeks. **Venue.** ACL/EMNLP; efficient-ML workshop.

### B5. Automatic contradiction surfacing as a proactive primitive
**Gap.** "Memory is a review problem" is loud 2026 discourse; belief revision (AGM/JTMS) is
under-applied to LLM memory. Systems store contradictions; none *surface* them unprompted.
**Thesis.** A consolidation pass that detects and surfaces contradictions/staleness ("this
conflicts with your March decision") measurably improves downstream decisions vs silent overwrite.
**Method.** Contradiction detection over the transition log; a proactive-delta protocol; eval on a
belief-revision task (does the agent avoid stale-fact errors STALE-style, and does surfacing beat
overwrite).
**Novelty.** The proactive/synthesis angle the user cares about, made measurable.
**Risk.** Medium. **What we have.** 🟡 temporal store with supersession; 🔴 the detector + eval.
**Effort.** 4–6 weeks. **Venue.** ACL/EMNLP Findings.

---

## Tier C — Benchmark & measurement (NeurIPS D&B track)

### C1. As-of-T state reconstruction benchmark (renamed — "TemporalBench" is taken twice)
**Gap.** No benchmark evaluates point-in-time state reconstruction. STALE (2605.06527) = invalidation
only; bi-temporal systems (MemStrata, Graphiti, Engram) ship as-of querying *explicitly
unevaluated*. Verified-open and hot.
**Thesis + deliverable.** A contamination-free benchmark + metric for as-of-T reconstruction over
multi-session histories with versioned valid-time ground truth; question types: point-in-time,
supersession, trajectory, stale-trap.
**Method.** Programmatic generation (TDBench-style temporal-SQL) for scalable, contamination-free
ground truth; deterministic + adversarially-validated scoring.
**Experiments.** Eval Mem0/Zep/full-context/RAG/our reader via the existing shims — expected
headline: production systems fail (STALE precedent: 6–8%).
**Novelty.** Verified-open; pair with the temporal-cheatsheet method (B2) so it's not "just a
benchmark."
**Risk.** Medium — adoption risk; mitigated by shipping a method that wins it and re-scoring
named systems. Move fast (STALE/MemTrace landed 6 weeks apart).
**What we have.** 🟡 temporal schema as ground-truth store, provider shims; 🔴 the generator.
**Effort.** 4–6 weeks. **Venue.** NeurIPS D&B, ACL.

### C2. Deterministic scoring & judge audit for memory QA
**Gap.** LoCoMo's judge accepts 63% of wrong answers; LLM judges flip verdicts 13.6% of runs
(Coin-Flip-Judge 2606.13685). No reusable, adversarially-validated scoring artifact exists for
memory QA specifically. You *observed this yourself* in demo 01.
**Thesis + deliverable.** A deterministic/adversarially-validated scoring protocol + harness;
re-score published memory results and show rank changes.
**Method.** Regex/structured scoring where possible; a "judge card" (wrong-answer acceptance,
flip rate, position bias) required for the rest; re-run vendors via shims.
**Novelty.** Demand proven (Penfield, Maximem); nobody packaged it for memory.
**Risk.** Low. **What we have.** 🟡 demo-01 deterministic scorer, shims. **Effort.** 3–4 weeks.
**Venue.** D&B, eval workshop.

### C3. Write-quality benchmark: marginal utility of a memory write
**Gap.** HaluMem (op-level hallucination) and MemAudit (write-as-compression-vs-oracle) are weeks
old and not interoperable. "Was this write worth keeping?" has no standard.
**Thesis + deliverable.** A benchmark scoring the *write path* directly — each candidate write's
marginal downstream utility — decoupled from retrieval and reader.
**Method.** Exactly the object your counterfactual machinery already computes; package it as a
shared oracle/candidate-set protocol others can plug into.
**Novelty.** Turns your internal reward signal into a community artifact; directly relevant to A1.
**Risk.** Medium. **What we have.** 🟡 counterfactual eval. **Effort.** 4–6 weeks. **Venue.** D&B.

---

## Tier D — Systems / efficiency (MLSys, efficiency workshops)

### D1. KV compilation as the deployment format of studying
**Gap.** Prompt caching is treated as infra, not a memory tier; the studier is never optimized to
produce cache-stable output.
**Thesis.** Compiling the studied artifact into a cache-stable KV prefix gives memory at near-zero
read cost; optimizing the studier for cache-stability (stable header / volatile tail) beats naive
ordering on cost-at-fixed-accuracy.
**Method.** Measure tokens/latency/accuracy: compiled-prefix vs retrieval vs long-context; ablate
cache-stable ordering; the economics of one-base-model + N-prefixes + N-LoRAs.
**Novelty.** Systems+economics result; makes per-user memory viable.
**Risk.** Low–medium. **What we have.** 🔴 (uses standard prefix-cache APIs). **Effort.** 4 weeks.
**Venue.** MLSys, efficiency workshop.

### D2. Weights with provenance / reversible distillation
**Gap.** Distilled knowledge goes stale silently in weights; no un-learning tied to source.
**Thesis.** Coupling each distilled fact to its token-space provenance enables targeted un-learning
on supersession, avoiding weight-staleness at bounded retraining cost.
**Method.** Shard-tracked distillation; on supersession, retrain-without-shard or negative
distillation; measure staleness vs retention vs cost.
**Novelty.** Nobody has provenance-linked, reversible weight updates for agents.
**Risk.** High (un-learning is hard). **What we have.** 🔴. **Effort.** 8+ weeks. **Venue.**
NeurIPS/ICLR if it lands; likely a longer arc.

---

## Tier E — Position / synthesis (COLM, position tracks, blog→paper)

### E1. "Recall is not competence": memory eval measures the wrong thing
**Thesis.** Current memory benchmarks measure retrieval/recall, not capability *gain*; propose
expertise-per-compute as the axis and show leaderboard systems don't improve capability. Synthesizes
the LoCoMo audit + LongMemEval saturation + Machine Studying into one argument.
**Risk.** Low (position). **What we have.** 🟡 the frontier review + eval matrix. **Effort.** 2–3
weeks. **Venue.** COLM / position track / a strong blog that becomes a citation.

### E2. The temporal-blindness trilogy — one framework
**Thesis.** Temporal *reasoning* (date math), *awareness* (sensing elapsed time/deadlines), and
*memory* (what-was-true-when) are three orthogonal gaps the field conflates; unify TicToc +
Real-Time-Deadlines + Robotouille + your reconstruction results into one framework + a diagnostic.
**Risk.** Low. **What we have.** 🟡 the concept doc + demo 01. **Effort.** 3 weeks. **Venue.**
survey/position; feeds the YouTube episode.

---

## The portfolio spine — how these connect

They are not 14 separate efforts; they share four pieces of machinery, so early work compounds.

```
                     ┌──────────────────────────────────────────────┐
   THE LEDGER  ──────┤ B3 calibration/planning   E1/E2 position      │
   (experience log)  └──────────────────────────────────────────────┘
                                    │
   THE COUNTERFACTUAL ──►  A1 amortized critic ──► B4 utility-gated distill ──► D2 reversible weights
   MACHINERY               │                                   │
                           └──► C3 write-quality benchmark     │
                                                               ▼
   THE TEMPORAL STORE ──► C1 as-of-T benchmark ──► B2 temporal studying ──┐
                          C2 deterministic scoring                        │
                                                                          ▼
   THE STUDIER (GEPA) ──► B1 GEPA consolidator ──► D1 KV compilation ──► A2 / REM (everything)
```

**Recommended order (each ships something publishable before the next needs it):**

1. **C2 + E1 first (3–4 wks, ~$50).** Deterministic scoring + the "recall≠competence" position.
   Cheap, low-risk, and they set the *evaluation ground* every later paper stands on. C2 is a
   citable artifact; E1 is the narrative spine (and the video).
2. **B1 GEPA studier (3–5 wks, ~$150).** Finish the held-out run; verified-open; your infra exists.
   First real method result.
3. **C1 + B2 as a pair (5–7 wks).** The as-of-T benchmark *and* the temporal-studying method that
   wins it — benchmark + method together dodges the "just another benchmark" critique you flagged.
4. **A1 the critic (6–10 wks, ~$400).** The flagship. Highest prestige, highest risk; do it once
   the cheaper wins are banked and the write-quality protocol (C3) can serve as its eval.
5. **A2/REM** absorbs all of the above into the systems paper + the hero video.

**Two-paper minimum viable portfolio if time is short:** C2/E1 (guaranteed, cheap) + B1 (verified-open,
infra-ready). Everything else is upside.

**What to *not* write:** anything headlining per-op counterfactual reward as the contribution
(scooped 3×), anything resting on raw LoCoMo numbers (discredited), backend-transfer of the write
policy (weak now), or PIE-as-benchmark (your own negative killed it).
