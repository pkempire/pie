# Half-Life Memory — the one paper + product

*2026-07-06. The convergence. One elegant technique, one real headline, one product. Everything
prior (survival clock, temporal validity, amortized critic, studying) is a component of this, not a
rival. Not taking the current repo literally — it's the parts bin; rebuild around the primitive below.*

## Why (the goal, above the code)
An AI that knows you or your project like a great chief of staff: a *current, correct* model of your
world that updates as things change and never confidently asserts what stopped being true. Every
memory product today rots — timeless facts pile up, contradictions accumulate, stale beliefs surface
as truth. **Temporal correctness — "what's still true" — is the most valuable, least-solved capability
in memory.** That is the whole point.

## The technique (simple, elegant)
Store **dated state-transitions, not facts.** Give every fact a **learned half-life** — a survival
curve `S(fact, t) = P(still true after elapsed t)`, learned per fact-type from the data (job status
changes ~monthly; birthplace never). One learned quantity yields three capabilities no system has together:
1. **Current-truth retrieval:** rank by *similarity × survival*, not similarity alone → return the
   fact most likely still true, not the most similar (which is usually stale).
2. **Staleness detection:** `S` below threshold ⇒ the belief is flagged stale.
3. **Self-triggered revision:** low `S` triggers re-check / ask-user / re-read-log. Memory that refreshes itself.

The primitive already exists as code (`pie/core/temporal.py`, empirical survival tables normalized to
entity-relative time). It was built, then abandoned in the RL pivot. Rehabilitate it as the centerpiece.

## The headline (real, falsifiable)
Benchmark: **STALE** (implicit stale-fact detection). Reported: Mem0 8.3%, Zep 6.0%, best frontier ~55%.
**Target: 50%+ — a multiple-x over every production system.** Honest falsifier: *if it can't beat the
best vendor by ≥3× on STALE, the technique is wrong — kill it.* Secondary axis: TicToc (the clock),
survival as the freshness signal. The pitch: "every AI memory will confidently tell you things that
stopped being true months ago; ours knows what's still true — 7% vs 55% on the benchmark for it."

## Novelty (honest)
Decay/forgetting memory exists (FadeMem, Ebbinghaus). The defensible seam: survival is **learned per
fact-type from data** (not a fixed curve), used for **truth-maintenance + revision** (not just
forgetting/compression), **validated on temporal correctness** (not compression ratio). Zep has
bi-temporal edges but hand-coded validity and no learned half-life. That gap is the contribution.

## Product (why it's not just a paper)
**"Paste your agent trace or ChatGPT export → a living knowledge graph that knows what's still true."**
Unique magic moment: it *proactively surfaces staleness* ("you said you lived in Boston 9 months ago —
likely stale, still true?"). No competitor does this because none has a per-fact half-life.
- Consumer: ingest ChatGPT/Claude export → your world model + a "what's probably stale about you" feed.
- Enterprise: paste week-long agent traces → current, non-contradictory project state instead of
  re-reading 200k tokens; the freshness gate on outgoing claims.
Ingestion path exists: `pie/ingestion/pipeline.py` (export → transitions → world model) + `pie/ui/`.

## Hypotheses (falsifiable, ranked)
- **H1 (core):** fact-types have learnable characteristic change-rates; ranking by similarity×survival
  beats recency and similarity-only on stale-fact benchmarks (STALE) by a large margin.
- **H2:** storing dated transitions + reconstructing state-at-T beats latest-value stores specifically
  on "what changed / what was true when" questions (we already see directional signal in demos).
- **H3 (label-free):** the next interaction is a free label for staleness; triggering/learning revision
  from next-interaction surprise beats fixed decay — trains on raw logs, no annotation.
- **H4 (clock):** survival injected as an explicit state beats raw timestamps on TicToc decisions.
- **H5 (competence):** a current-state consolidated artifact beats raw retrieval at fixed token budget.
- **H6 (credit):** which transitions to trust/keep is predictable cheaply via an amortized critic over
  survival + retrieval features (r=0.71 seed).
Paper spine = H1+H2+H3 (learned survival → truth-maintaining memory). H4 = the clock demo. H5/H6 = extensions.

## Where to pick up (rehabilitate, then rewrite around the primitive)
- `pie/core/temporal.py` — survival tables **built**; missing: wire `S` into retrieval ranking + revision trigger (~the core 50 lines).
- `pie/core/world_model.py`, `dynamics.py` — transitions + provenance, done.
- `mempol/temporal/store.py` — clean as-of-T SQL store to build the rewrite on.
- `mempol/temporal/context.py:_choose_action()` — refresh/interrupt action; gate on survival.
- `pie/ingestion/pipeline.py`, `pie/ui/` — the product ingestion + viewer.
- `demos/02-temporal-awareness` (TicToc), `scripts/critic_counterfactual_smoke.py` — clock demo + trust critic.
- Baselines: `memory_providers/` shims (Mem0/Zep) to run head-to-head; get external STALE + slice by state-change.

## First test (cheapest kill-shot for H1, ~$30)
On STALE (or a state-change slice we build from LongMemEval + our temporal schema): compare (a) similarity-only
retrieval, (b) recency-weighted, (c) **similarity × learned-survival**, scored on stale-fact questions.
If (c) doesn't clear ~2–3× (a)/(b), H1 is dead and we stop. If it does, that's the paper's core plot and
the product's differentiator — go. No training required for this test; survival tables + a reader are enough.

*Memory: this supersedes the "time-as-state / TicToc-first" framing in [[project-current-bet]] — TicToc/the
clock is now H4 (a demo), not the headline. Headline = learned half-life for truth-maintaining memory.*
