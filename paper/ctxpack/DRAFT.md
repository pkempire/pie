# Compile, Don't Retrieve: learned, maintained memory packs for long-horizon agents

*Draft skeleton — 2026-07-08. Each section names the experiment that fills it and its current
status. Numbers marked [n=30, 1 seed] are directional until the variance-controlled reruns (Q6).*

## The idea (one paragraph)

Agent memory today is either retrieval (fetch similar text per query — pays per query, returns
similarity not understanding) or a hand-written context file (CLAUDE.md/AGENTS.md — unmeasured,
and per ETH arXiv:2602.11988, net-negative while adding >20% cost). We treat memory as a
**compiled artifact**: a budgeted, cache-stable, provenance-anchored pack compiled from the
corpus/history *before* queries arrive, with (i) an **escalation path** to retrieval only when
the pack can't answer, (ii) a **learned compiler** — the writer policy improved not by lossy
whole-prompt mutation but by **structured rule accretion**: discrete rules with provenance,
measured deltas, gated admission, and supersession, and (iii) a **maintenance loop** — diff-
scoped invalidation + a task-suite regression gate, so the pack stays true as the corpus
changes. Compile once, answer cheaply forever, escalate rarely, update incrementally, audit
everything.

## Claims → experiments → status

**C1. Pack+escalate beats query-adaptive retrieval at matched budget, at ~1/3 marginal query
cost.** [LongMemEval-S paired n=30: escalate 60.0% > RAG 56.7% > pack-only 40.0%; escalation
rate 33% → 67% of queries answered from a cacheable prefix. Status: directional; needs Q6
variance controls + full-set run + BM25+dense retrieval upgrade.]

**C2. Structured rule accretion beats prompt mutation for self-improvement** — stability
(no destroy-and-regress), auditability (rules carry provenance + measured deltas), and
revertibility (supersession). [Head-to-head running, same splits/seed. Early artifacts: the
ledger's proposed rules are human-sensible; the admission gate exposed eval-noise limits —
now a finding about gate design. Status: in flight + needs variance-corrected rerun.]

**C3. The maintenance loop keeps compiled memory true under corpus change at a fraction of
recompile cost.** [Mechanism demonstrated on the mempol-corpus evolution (r3 collapse caught by
the gate; pack provenance corrected our own answer key). Status: full git-diff experiment
pending — the surviving piece of the maintenance program.]

**C4. Methodology: paired designs and deterministic scoring, or your numbers lie.** [Transplant
finding: SOTA prompts through a different pipeline score 26.7% vs their own pipeline's
reputation — prompt transplants are invalid baselines. Judge-flip and compile-stochasticity
variance measured (same config: 33% vs 67% held-out). LoCoMo audit as external context.
Status: material in hand.]

## Baselines table (paired, same 30 questions, same judge)
pack-only · RAG-lexical · pack+escalate · Mastra OM exact full pipeline (RUNNING) ·
[Q6: budgets 4k/8k/16k × {our writer, Mastra}; dense retrieval; multi-seed]

## Open items before submission
Full LongMemEval-S run (500q) with variance controls · rule-vs-mutation final table ·
transfer test (writer trained on split A → unseen haystacks B) · maintenance-on-git-diffs
experiment · calibrated escalation trigger (logprob, not string match) · cost accounting
figure (compile amortization vs per-query retrieval).

## Venue path
Workshop (agents/memory) with C1+C2+C4 → main conference (NeurIPS D&B or ACL) once C3 +
full-set + multi-seed land.
