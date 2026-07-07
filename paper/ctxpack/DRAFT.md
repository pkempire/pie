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

## Methodology in standard terms (no invented vocabulary)

What this system actually is, mapped to the literature by mechanism:

| Our working name | Standard mechanism | Closest prior work |
|---|---|---|
| "pack" / compile | memory consolidation via hierarchical summarization into a fixed, reusable working context | Generative Agents reflection (2304.03442); Mastra OM; sleep-time compute (2504.13171); the Machine Studying "cheatsheet" |
| "pack + escalate" | hierarchical memory: small always-in-context tier + fallback search over the raw archive | **MemGPT (2310.08560), exactly** — core memory + archival search |
| "rule ledger" | experiential insight extraction into a curated insight pool with add/edit/vote operations | **ExpeL (2308.10144)** — closest; also Voyager skill library, ACE playbooks, Reflexion |
| "writer-prompt mutation" | reflective prompt optimization | GEPA (2507.19457), OPRO, Promptbreeder |
| "temporal store / supersession" | bi-temporal fact validity + invalidation in a structured world model | Zep/Graphiti (2501.13956) |
| "maintenance loop / regression gate" | incremental view maintenance + CI regression testing, applied to consolidated memory artifacts | promptfoo-style CI; DB view maintenance — **the combination is unclaimed** (verified scans) |
| "blame decomposition" | containment-based credit attribution between content and organization | related: retrieval-usage attribution in MemBuilder's ADRPO |

**Honest novelty assessment by mechanism, not name:**
- *Not novel:* consolidation, hierarchical fallback, insight pools, prompt reflection — all
  established 2023–2025. We implement them; we do not claim them.
- *Incremental:* held-out-gated insight admission with per-batch measured deltas and provenance
  (= ExpeL + evaluation discipline); the confidence-gated escalation trigger.
- *Actually open (verified against mid-2026 literature):* (1) **maintenance** — diff-scoped
  invalidation + regression-gated updates of consolidated memory; (2) **the economics** —
  budget-matched, amortization-aware measurement of consolidation vs retrieval vs hierarchical
  fallback (compile-once-cacheable vs pay-per-query), with escalation rate as the key statistic;
  (3) **the noise audit** — our identical-config spread (33/67/75% held-out, n=12) implies many
  published small-n prompt-optimization gains may be evaluation noise; demonstrated
  systematically, this is a standalone finding.

**The system design, end to end, in standard terms:** (L1) append-only raw history + a
bi-temporal structured world model as ground truth; (L2) a consolidation policy that compiles a
budgeted, cache-stable working context from L1 per task distribution; (L3) confidence-gated
fallback search over L1; learning = insight-pool updates to the consolidation policy, admitted
only on held-out improvement; integrity = incremental invalidation on source change + a
regression suite before any updated artifact ships. **End result:** agent memory that is cheap
(cacheable), auditable (provenance end to end), self-improving (gated), and stays true under
change — evaluated with paired, budget-matched, variance-controlled protocols.

## Venue path
Workshop (agents/memory) with C1+C2+C4 → main conference (NeurIPS D&B or ACL) once C3 +
full-set + multi-seed land. Candidate spin-out: the noise-audit finding as its own short paper
("how much of prompt-optimization improvement is evaluation noise?").
