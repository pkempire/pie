# Research proposals — short form

*2026-07-07. Four proposals, each: problem / prior work / our idea / first experiment. All share
one substrate (anchored store + supersession + eval gate). Details: RESEARCH.md. Status of runs:
`ctxpack/results/`.*

---

## P1 — Compile, Don't Write: task-optimized context packs

**Problem.** Every coding agent reads a hand-written context file (CLAUDE.md/AGENTS.md). ETH
(arXiv:2602.11988) showed these files don't improve success and add >20% cost — LLM-generated
ones are net negative. Nobody measures or maintains them.
**Prior.** One-shot generators (/init, Copilot, Cursor, DeepWiki) — no eval. aider repo-map —
budgeted but heuristic, ephemeral. GEPA/ACE — optimize prompts/playbooks, not repo-derived
anchored packs. LLMLingua — token-level compression, no anchors, no gate.
**Our idea.** Treat the context file as a *compiled artifact*: budgeted map/reduce compile with
`[src:]` anchors per section; blame-decomposed evolution (fact missing = writer fault; present-
but-failed = organization fault — a deterministic containment check); an outer GEPA loop over
writer/reviser/reader prompts so the *policy* transfers to unseen corpora; diff-scoped
incremental recompile with the task suite as a regression gate.
**Evidence so far.** mempol corpus, matched 4k budget, held-out: handwritten 0–17% < RAG 41.7% <
evolved pack 83.3% (single seed, n=6 — directional). Evolution also demonstrated the failure the
gate prevents (r3 over-pruned, held-out collapsed).
**First real experiment.** Transfer: outer-loop on corpora A–C, frozen writer on unseen D vs
one-shot generation + repo-map + RAG at matched budgets; then maintenance on real git history.
**Falsifier.** Transferred writer ≤ one-shot generation on unseen corpora → maintenance-only
workshop paper.

## P2 — The Experiment Ledger: supersession-aware state for long-horizon agents

**Problem.** Research agents repeat their own failures. Google Co-Scientist's reports: "cycled
through variations of the same failed approaches," "insights from failed attempts are lost
across sessions," "confident rediscovery" of falsified hypotheses. Kosmos's world model is
additive-only; Sakana's tree is per-run.
**Prior.** StatefulDiscovery (2606.11851), Hypothesis-Tree Refinement (2606.11926), BeliefMem —
concurrent papers calling for exactly this; none ships a reusable component. Agent Trace spec
(Cognition+Cursor+Google) standardizes provenance with no compiler. Observability tools
(LangSmith Insights) cluster traces; none reconstructs epistemic state.
**Our idea.** Compile agent traces into a ledger of (belief, evidence, status) and (experiment,
result, artifact) objects with **supersession chains** — new results mark old beliefs superseded
rather than coexisting. Inject the ledger at decision points. Substrate already implemented
(`mempol/ledger` research objects + `mempol/temporal` valid-time/supersession).
**First experiment.** Long multi-session coding/research task; measure repeated-failed-attempt
rate and confident-rediscovery rate with ledger injection vs transcript-replay vs summary.
**Falsifier.** No reduction in repeat-failure rate → the primitive is decorative; publish the
negative.

## P3 — Learned Observational Memory (the LongMemEval play)

**Problem.** The best LongMemEval system (Mastra OM, 94.87%) is a *hand-tuned* Observer/Reflector
prompt pair — frozen, never corrected by outcomes. Learning the consolidator is the obvious next
step nobody has published (GEPA×memory verified open; MemPro took the TextGrad slot).
**Prior.** Mastra OM (hand-coded, SOTA); Memory-R1/Mem-α (RL, weights, trajectory rewards);
our GEPA consolidator infra (result was n=5 overfit — unproven, now redeemable).
**Our idea.** The ctxpack writer *is* an Observer/Reflector — so learn it: GEPA-evolve the
map/reduce prompts against held-out LongMemEval accuracy, with dated-transition pack format
(our timeline result: temporal structure wins on knowledge-update questions). One learned
policy, applied per-haystack at compile time — **no per-corpus retraining** (the policy
transfers; per-corpus you pay only the compile).
**Evidence so far.** Harness running (ctxpack/lme_pack_eval.py, n=30 in flight, full traces
saved). Expected shape: question-blind pack loses to query-adaptive RAG on needle-lookup,
competitive on preference/knowledge-update; the learned writer closes the gap.
**Falsifier.** If the GEPA-evolved writer can't beat the frozen v0 writer by a clear margin on
held-out haystacks, prompt-space learning is exhausted here → escalate to weights or stop.

## P4 — PIE v2: identity-safe entity resolution (the KG rebuild)

**Problem (measured, 2026-07-07).** The v1 world model (1,975 entities): "Guide to Cursor" holds
36 aliases including `Tor` and `PyTorch`; `Pinecone` absorbed `Weaviate`/`ChromaDB`; **57/283
aliases (20%) share zero tokens with their entity name**; 6 exact-duplicate entities. Cause:
Tier-2 substring/fuzzy containment + Tier-3 embedding similarity treat *relatedness* as
*identity*.
**Prior.** Classical entity resolution; Zep/Graphiti LLM dedup; our own mempol finding that
learnable dedup (lookup-before-create as policy) beats hardcoded cascades.
**Our idea (v2, releasable repo).** (i) Merges require an *identity* test (LLM verify with
evidence spans), never a similarity threshold; (ii) every alias carries provenance + confidence;
(iii) merges are supersession events — **revertible**, not destructive; (iv) an identity
test-suite as the regression gate for the ingestion pipeline (same gate pattern as ctxpack);
(v) related-but-distinct becomes a typed RELATED edge, not an alias. Then re-ingest the ChatGPT
export and ship the graph viewers on top.
**First experiment.** Re-resolve the existing 1,975 entities under v2 rules; report merge-error
rate vs v1 on a hand-labeled 100-pair identity set; then full re-ingest.

---

**Amortization (the retrain question, answered once):** policies are trained ONCE across corpora
and transfer; a new corpus/env costs only a compile (+ optional cheap artifact-evolution if a
task suite exists). Per-corpus retraining would kill both the economics and the paper claim —
transfer IS the claim.
