---
title: "Frontier review + repo triage — 2026-07-01"
purpose: "Finalizes §10 (literature landscape) and §11 (the next bet) of ORIENTATION-AND-NEXT-BET-2026-06-21.md. Full-repo audit reconciled against a fresh web literature review (academic, systems, and evaluation frontiers as of July 1, 2026)."
status: "FINAL — supersedes the open sections of the June 21 orientation doc"
method: "6 parallel research passes: (1) new-subsystem code audit, (2) June-6 self-doc distillation, (3) research-wiki inventory, (4) academic frontier web review, (5) commercial/systems landscape web review, (6) evaluation/benchmark frontier web review. Every external claim below carries a source."
---

# Frontier review — where this repo stands against the field, July 2026

> The one-sentence version: **your instincts were right and your timing is tight.** The thesis
> you killed (per-op counterfactual) got published by others within weeks of your killing it;
> two ideas you have on disk (the amortized utility critic, GEPA-on-consolidator) are
> verified-open as of today; and your temporal state-reconstruction idea is sitting on the
> single most confirmed-open, most-demanded gap in memory evaluation. The field moves on a
> ~4-week publication cadence in exactly your niche. The constraint is no longer ideas or
> code — it is finishing one thing and shipping it.

---

## 0. Scoreboard: every idea in this repo vs. the July 2026 frontier

| # | Your idea (source) | Frontier verdict | Evidence |
|---|---|---|---|
| 1 | **Per-op counterfactual write reward** (paper/main.tex thesis) | ⚫ **SCOOPED — correctly killed** | Memory-R2 (arXiv:2605.21768, May 20) does same-state local rerollouts per memory op; HiMPO (arXiv:2606.16285, Jun 15) does per-write local counterfactual utility + hindsight gating; Rosetta (arXiv:2606.07711) does operator-level counterfactual gain. You declared this dead Jun 18 — the field confirmed it was a real idea by publishing it three times. These are now mandatory related work for anything you write in this area. |
| 2 | **Learned critic that amortizes counterfactual replay** (critic_counterfactual smoke, Pearson r=0.707) | ✅ **VERIFIED OPEN** | Targeted searches find nothing. HiMPO explicitly advertises *avoiding* a critic; MemQ (2605.08374) learns Q-values read-side only; "Beyond Heuristics" (2512.21567) is decision-theoretic, not amortized. Now that per-op counterfactual credit is established literature that everyone pays replay cost for, "a value model that replaces the replay" is the natural next paper. Shelf life: short (~months) — this sub-area publishes every ~4 weeks. |
| 3 | **GEPA-optimized memory consolidator** (Goal 01) | ✅ **VERIFIED OPEN** | No GEPA×memory paper exists. Closest: MemPro (2606.00619, Jun) evolves memory programs with TextGrad — occupies the adjacent slot but leaves GEPA (ICLR 2026 Oral, 35× fewer rollouts than GRPO) unclaimed on memory. Your blocker is internal: the only result is train==val, n=5; the held-out run crashed to a 0-byte file. ~$100 finishes it. |
| 4 | **Temporal state-at-T reconstruction, benchmarked** (TemporalBench spec + rlm_temporal_reconstruction demo, 67%→83%) | ✅ **VERIFIED OPEN and HOT** | No benchmark evaluates as-of-time-T state reconstruction. STALE (2605.06527) covers only invalidation-detection — and Mem0 scores **8.3%**, Zep **6.0%** on it. MemStrata (2606.26511) ships as-of querying and literally says it's "a capability we build on but do not evaluate here"; same for Engram, Graphiti, GapTime. Every vendor (incl. Mem0 publicly) names temporal reasoning their weakest area. Two adjacent benchmarks (STALE, MemTrace) appeared within 6 weeks of each other — move fast. **Rename it**: "TemporalBench" is taken twice (2410.10818 video; 2602.13272 time-series). |
| 5 | **Timeline-Synthesis read policy** (71.7% LongMemEval-S balanced, n=240) | 🟡 **Real but not a headline alone** | Honest-number context: Mem0's claimed 93.4% reproduced at **73.8%** by a frozen-judge third-party harness (Maximem); Mastra OM 94.87% and RLM-scaffold 89.8% are the real read-side frontier. 71.7% honest is respectable, not SOTA. Its value: it's the *method leg* of the temporal-reconstruction story (80% on knowledge-update questions), not a standalone paper. |
| 6 | **Amortized-critic + temporal framing: "reasoning/awareness/memory" distinction** (time-aware-memory.md) | ✅ Open as a *framing* | TicToc / Real-Time Deadlines / Robotouille each test one leg; nobody has published the unified three-way decomposition. Good for the benchmark paper's intro and the YouTube episode; not a standalone contribution. |
| 7 | **Neutral memory-benchmark referee** (benchmarks/ + memory_providers/ + paper_leaderboard) | ✅ **OPEN, high demand, low effort** | LoCoMo audit: 6.4% of answer key wrong, judge accepts 63% of intentionally-wrong answers (github.com/dial481/locomo-audit); EverMemOS 92.3% claimed → 38.4% reproduced; Mem0 93.4% → 73.8% observed. The only reproduction harness (Maximem) is vendor-owned. Nobody neutral exists. You already own 80% of the needed infrastructure. |
| 8 | **Memory-as-learned-OS / LoRA write policy** (mempol Phase A/B) | ⚫ **CROWDED — do not resurrect as-is** | 2026 RL-memory wave: AgeMem, MemBuilder (84.2% LoCoMo w/ 4B model), Mem-T, MemPO, DeltaMem, Memory-R2, HiMPO. Letta's June "memory models" manifesto claims the framing commercially. Your Phase A was never run, Phase B never produced signal, trained-R-as-judge is a NotImplementedError. Competing here head-on as a solo undergrad against groups shipping monthly is the wrong fight. The *critic* (idea #2) is the differentiated wedge into this literature, not the policy itself. |
| 9 | **PIE typed-KG as memory substrate** | ⚫ **Dead as research; keep as product** | Your own n=1491 matrix: PIE-KG 46.2% < flat RAG 58.6% (evidence_recall 0.0). Field-level agreement: Hindsight's simpler vector baseline beats its own KG; substrate matters less than quality floor. Keep PIE as personal MCP + data source only. |
| 10 | **Research Ledger / repo-continuity + Temporal Context Engine** (mempol/ledger, mempol/temporal) | 🟡 **Aligned with real open gaps, but a 6-month bet** | Coding-agent memory is publicly unsolved (Cursor *removed* Memories; "memory is a review problem" discourse; CL-Bench notes SWE-Bench-CL is saturated-metric and unevaluated with frontier systems — a chronological coding-agent continual-learning benchmark is open). But the learned compiler πθ(C,a|...) doesn't exist, the training corpus is ~1-2 rows/run, and the flagship TCE has never touched real data. Right vision, premature as the *next* bet. The ledger substrate is a real asset — park it warm. |
| 11 | **Contract-based multi-agent orchestration** (Goal 04) | 🟡 Open but off-thesis | Still unclaimed infrastructure; still a distraction from the memory line. Keep parked. |
| 12 | **GitMem content-addressable backend** (backends/gitmem.py, untracked) | 🟡 Zeitgeist-aligned, undifferentiated | "Git for agent memory" / branches-diffs-rollback is loud 2026 discourse, and Letta moved to git-backed Context Repositories. Your backend is unwired and untested. Only worth reviving as the substrate for the review-tooling product idea, not as research. |

---

## 1. The mid-2026 landscape in ten load-bearing facts

1. **The RL-memory niche exploded.** Since January: AgeMem (step-level GRPO), MemBuilder (ADRPO, 84.23% LoCoMo from a 4B model), Mem-T (memory-op-tree credit), MemPO (per-step info-content advantage), DeltaMem (state-distance reward), Memory-R2 (same-state rerollouts), HiMPO (counterfactual + hindsight gate). Publication cadence in this sub-area: ~4 weeks.
2. **Per-op counterfactual credit is now literature** (Memory-R2, HiMPO, Rosetta) — but everyone pays for it at training time with rerollouts or log-prob scoring. **No one amortizes it with a learned critic.**
3. **GEPA×memory is unclaimed.** MemPro took the TextGrad slot; the GEPA slot (35× cheaper than GRPO, ICLR 2026 Oral) is empty.
4. **LoCoMo is discredited.** 6.4% wrong answer key, ~93.6% theoretical ceiling, judge accepts 63% of intentionally wrong answers, 22.5% of questions never evaluated by anyone (broken formatter). Usable only as a legacy secondary number with the audit cited.
5. **LongMemEval is saturated-by-eval-gaming.** Claimed-vs-observed scandals (Mem0 93.4→73.8; hidden judge rules, hardcoded date hints). Successor LongMemEval-V2 (115M-token agent trajectories, latency-accuracy frontier scoring) is the new target. Honest numbers in the 70s are competitive; nobody trusts the 90s.
6. **The write path is the new eval frontier.** HaluMem (operation-level extraction/update hallucination; extraction recall <60% across Mem0/Zep/etc.) and MemAudit (write-as-budgeted-compression vs oracle) landed in the last months. Write-quality benchmarking is where LoCoMo-style QA was in 2024.
7. **Temporal is the universally admitted weakness.** Mem0 says so publicly; STALE puts memory frameworks at 5-8%; bi-temporal systems ship as-of querying unevaluated. **No as-of-T reconstruction benchmark exists.**
8. **"Dreaming" became consensus.** OpenAI shipped ChatGPT "Dreaming" June 4; Letta, Honcho, Supermemory, MIRIX all have background consolidation. Sleep-time consolidation is now a productized mechanism, not a research novelty — the open question is *learned/optimized* consolidation (your Goal 01/02 framing survives).
9. **Read-side frontier = agentic compute over an organized log, not better embeddings.** Mastra OM 94.87% with no per-turn retrieval; RLM-scaffold 89.8%. Files/logs + agent curation + prompt-cache-stability is beating extraction-into-DB pipelines (Anthropic and Letta both moved toward files).
10. **Reviewer expectations hardened.** A credible 2026 memory paper: ≥2 benchmarks (one beyond-context-window), full-context + RAG + Mem0/Zep/A-Mem baselines, frozen judge with adversarial judge validation, multi-seed with std, cost/latency columns, full pipeline disclosure. (Exemplars: APEX-MEM, MemoryAgentBench, Memora.)

Sources for all of the above are in the three agent reports; keys: arXiv 2605.21768, 2606.16285, 2606.07711, 2606.00619, 2507.19457, 2605.06527, 2606.26511, 2511.03506, 2605.02199, 2605.12493, 2510.27246, github.com/dial481/locomo-audit, maximem.ai claimed-vs-observed, mastra.ai/research/observational-memory, openai.com/index/chatgpt-memory-dreaming.

---

## 2. Repo triage: what's relevant, what to fix, what to bury

### Keep and build on (the real assets)
- `mempol/scripts/{longmemeval,locomo}_matrix.py` + `mempol/strategies/` — the eval harness with real n=240/n=1491 runs. **This is the platform for everything below.**
- `mempol/policies/rlm_temporal.py` (Timeline-Synthesis, 71.7%) + `continuity.py` (trace generator).
- `mempol/temporal/schema.py` — valid-time/supersession schema. Repurpose: this is the *ground-truth generator* for the benchmark, not the flagship engine.
- `output/experiments/critic_counterfactual.json` (r=0.707 seed) + `mempol/scripts/critic_counterfactual_smoke.py` — the seed of the one open RL-adjacent contribution. Move under `paper/` or `mempol/results/`.
- `mempol/ledger/` + `mempol/core/` — shipped, tested substrate. Park warm for the repo-continuity bet later.
- `research/` wiki + verification discipline + `paper_leaderboard` — the credibility/distribution asset; feeds the neutral-referee play and the YouTube channel.
- `memory_providers/` shims (Mem0/Zep/Honcho/Supermemory) — needed to run vendors on the new benchmark.
- `benchmarks/parallel_runner.py`, `refs.bib` (after R5 cleanup), `runs/sft_warmup.jsonl`.

### Fix and run now (ordered, with cost)
1. **R0 — git (30 min, $0).** `git am --abort`, commit the ~2 months of untracked work in logical chunks (HANDOFF §7.4 grouping), push. Still the only emergency.
2. **GEPA held-out run (~$100, days).** train≠val, ≥30 Q, ≥2 held-out convs, 3 seeds. Converts Goal 01 from "overfit smoke" to citable-or-dead. Niche verified open — this is now cheap optionality on an unclaimed result.
3. **Timeline-Synthesis full eval (~$50-100).** Full LongMemEval-S + full-context oracle + strong rerank baseline + budget curve. Needed as the method leg of the benchmark paper regardless.
4. **Critic scale-up (~$100-200).** Regenerate per-op deltas at 10-20× the toy sample, test whether r=0.707 holds, on the existing write-trajectory infra. Decides whether the critic paper is real.
5. **R5 — strip/verify fabricated-looking citations** in `applications/registry.py` and `refs.bib` (7 Anonymous entries, future-dated IDs) before anything feeds a paper.

### Stop / bury (with reasons)
- **`paper/main.tex` as written** — sells a thesis that is now *other people's published work*. Do not rewrite it around Phase B. Salvage: refs.bib, the 2×2 related-work grid, the chunking subsection.
- **Phase A/B RL training runs as spec'd in paper/TODO.md §2** — the $300 Phase B run, backend transfer table, the five ablations: all serve the dead thesis. Do not spend this money.
- **PIE-as-benchmark-backend repair** (the `--load_cached_kg` flag, the init-signature fix): obsolete — your own matrix already produced the honest negative (KG < flat), and LoCoMo (its target) is discredited. PIE stays as personal MCP.
- **`universal_*` RL recipes** until substrate recall (0.15-0.31) is fixed — a policy can't learn through a broken retriever. Not on the critical path.
- **LoCoMo as a primary benchmark.** Report it only with the audit cited, or adopt/ship a corrected variant.
- **Temporal Context Engine as "the flagship."** The learned compiler needs a training corpus that doesn't exist. The schema survives inside the benchmark work; the engine waits.
- **architect/, Footnote, sales/, visual-memory/** — products, not research. Footnote's smoke is a separate decision; none of these touch the bet.

---

## 3. §11 finalized — THE NEXT BET

**Pick C, carried by A, with B as a cheap side quest.** Concretely:

### The bet: a temporal state-reconstruction benchmark + the method that wins it
One paper (and one video) with three legs:

1. **Benchmark** (~3 weeks): as-of-time-T state reconstruction over multi-session histories with versioned, valid-time ground truth. Generation trick: TDBench-style programmatic generation (contamination-free, scalable) + your `mempol/temporal/schema.py` as the ground-truth store; question types: point-in-time state ("what was true on D?"), supersession ("what replaced X and when?"), trajectory ("how did X change?"), stale-trap (STALE-style). NOT named TemporalBench (taken twice). Eval hygiene per 2026 reviewer bar: frozen judge + adversarial judge validation + multi-seed + cost columns.
2. **The finding** (~1 week, mostly existing shims): run Mem0, Zep/Graphiti, a full-context oracle, and flat RAG on it. Expected headline, based on STALE's 5-8% vendor scores and vendors' own admissions: **production memory systems catastrophically fail point-in-time queries.** A negative result of this shape is exactly what got Memora rewarded at ACL 2026.
3. **The method** (~2 weeks): Timeline-Synthesis / `reconstruct_state_at` reader (your 67%→83% synthetic demo, your 71.7% LongMemEval infrastructure, 80% on knowledge-update) as the first system that does well on it — plus its LongMemEval-S number as the external-validity anchor.

Why this over the alternatives: it's the only candidate that is simultaneously (a) verified open, (b) demanded — vendors publicly confess the weakness, systems ship the capability unbenchmarked, (c) achievable solo with infrastructure you already have, (d) durable — benchmarks compound citations for years while method papers in this niche age in weeks, and (e) it *leads with your one real positive result* instead of hiding it.

### The side quest: finish GEPA held-out (~$100)
If it holds ≥5pp: a clean short paper/blog ("reflective prompt evolution for memory consolidation at 1/35th the compute of GRPO") in a verified-open slot, and the Working Memory ep-1 live demo. If it doesn't: Goal 01 dies cheaply and honestly.

### The follow-up (post-benchmark): the amortized utility critic
Position against HiMPO/Memory-R2/Rosetta as "they proved per-op counterfactual credit works; we make it affordable." Requires the write-trajectory infra you already have and the r=0.707 seed to survive scale-up. Highest-risk, highest-prestige of the three; do not start it before the benchmark ships — the benchmark is also its evaluation vehicle.

### Sequencing (next ~10 weeks)
| Week | Action | Cost |
|---|---|---|
| 0 (today) | R0 git rescue; R5 citation cleanup | $0 |
| 0-1 | GEPA held-out run (side quest, decides Goal 01) | ~$100 |
| 1-4 | Benchmark v0: generator + 200-400 questions + judge card | ~$50 |
| 3-5 | Vendor runs (Mem0, Zep, oracle, RAG) via existing shims | ~$100 |
| 4-7 | Method leg: reconstruct_state_at reader + full LongMemEval-S validation | ~$150 |
| 7-9 | Paper + public leaderboard page (paper_leaderboard muscle) + Working Memory ep-1 anchored on it | time |
| 10+ | Critic scale-up → second paper | ~$200 |

Total cash: ~$600 — inside the $500-1000 envelope the HANDOFF already budgeted.

### What "done" looks like
An arXiv paper + public leaderboard where the field's named systems have embarrassing-but-fair numbers on temporal reconstruction, your reader is the reference method, the eval methodology is audit-proof by construction, and the YouTube episode walks through it. That single artifact simultaneously: ships Goal 03 (public content), retires the dead paper honestly, converts the eval harness into a community asset, and gives the critic paper a home benchmark.
