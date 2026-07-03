# Why agents don't get better — and what we're doing about it

*The problem statement and approach behind this repo. Last updated 2026-07-03.*

Every LLM agent you use today is a brilliant amnesiac. It starts each session from zero,
re-derives what it knew yesterday, repeats mistakes it already made, and plans like someone
who has never done the task before — because, from its point of view, it never has.

"Memory" products mostly bolt a search index onto this amnesiac: store what was said, retrieve
what sounds similar. That helps with *recall* and does nothing for *competence*. The agent that
retrieved your preferences is not an agent that learned your codebase, calibrated its plans, or
noticed that two of your decisions contradict each other.

This repo is about the second thing: **agents that turn experience into expertise.**

## The four problems

**1. Temporal blindness.** Facts change; similarity search doesn't know when. Ask a flat
vector store *"where did the user live in May?"* after an August move and it answers with the
current city — the newest, most-similar value ([Demo 01](demos/01-stale-memory/): 40% vs 100%
on as-of-the-past questions). This isn't a toy problem: on the STALE benchmark's
implicit-staleness scenarios, production memory frameworks score
[6–8%](https://arxiv.org/html/2605.06527), and even Mem0 publicly names temporal reasoning
[its weakest area](https://mem0.ai/blog/mem0-the-token-efficient-memory-algorithm). The fix is
structural, not more retrieval: store *transitions* (what changed, when, what it replaced) and
compute what-was-true-when by replaying them on demand.

**2. The planning fallacy, uncorrected.** Ask a frontier model to plan a 30-minute script and
it produces an "8-day, 5-phase" project plan — it estimates in human-team-days because that's
what its training data plans look like, and nothing in its context tells it what tasks
*actually* cost. Humans debias estimates with reference-class forecasting: "the last five tasks
like this took N." Agents can't, because nobody keeps their actuals. An experience log with
outcomes attached is the missing input; this repo's ledger (`mempol/ledger/`) records exactly
that — plans, runs, durations, results — from real repo history.

**3. Retrieval is not expertise.** The sharpest recent evidence:
[Machine Studying](https://jacobxli.com/blog/2026/machine-studying/) (2026) showed two
equally-capable models with search over the same corpus retrieve the *same* documents, yet the
one with more domain knowledge keeps the right ones and the other sets them aside — "nothing
failed in retrieval." Fine-tuning on the corpus failed too (memorization ≠ expertise). The only
method that worked was a **studied cheatsheet**: a compact artifact built by working through the
corpus before questions arrive. Which is to say: consolidation — done well — *is* how an agent
gets better, and [Letta](https://www.letta.com/blog/continual-learning),
[OpenAI's Dreaming](https://openai.com/index/chatgpt-memory-dreaming/), and every serious 2026
system converged on the same background-consolidation shape.

**4. Nothing improves the loop itself.** Every production system hand-writes its consolidation
logic once and freezes it. But "what should the cheatsheet contain, for this corpus and these
tasks?" is exactly the kind of question you optimize, not hand-code. Reflective prompt
evolution ([GEPA](https://arxiv.org/abs/2507.19457), ICLR 2026 — ~35× cheaper than RL) and
RL over memory operations (Memory-R1, HiMPO, and the 2026 wave) both work; nobody has shipped
an optimized *studying loop* end to end.

## The approach

One loop, three parts:

```
raw experience log  ──(sleep-time studying)──►  expertise artifact  ──►  cheaper, better work
      │                        ▲                      │
      │                        │                      ├── timeline of transitions (fixes #1)
      │              optimize the studier             ├── calibration table of actuals (fixes #2)
      │              (GEPA first, RL second)          ├── studied cheatsheet (fixes #3)
      └── everything, append-only, with timestamps    └── surfaced deltas: contradictions,
                                                          connections, "this went stale"
```

- **Append-only log** of what happened, with time attached (`mempol/core/`, `mempol/ledger/`).
- **A studying pass** that runs between sessions and compresses the log into a compact,
  human-readable expertise artifact: current state per entity *with validity*, calibration
  data, distilled how-tos. Proactivity falls out for free — the pass that compresses is the
  pass that notices contradictions and staleness.
- **Optimization of the studying pass itself** against downstream task performance per unit of
  inference cost — the [expertise-as-efficiency](https://jacobxli.com/blog/2026/machine-studying/)
  objective — using GEPA (cheap, prompt-space) before any RL (expensive, weight-space).

Token-space first, deliberately: readable, portable across models, debuggable, and — per both
Letta's argument and the Machine Studying results — currently *more effective* than fine-tuning
for this job.

## How we work

Bite-size, verifiable, or it doesn't ship. Every claim gets a [demo](demos/) you can run for
cents with deterministic scoring; larger results go through the eval matrix
(`mempol/scripts/longmemeval_matrix.py`, real n=240/n=1491 runs) with honest baselines. The
repo's own history is the first testbed: the ledger has already ingested it (541 artifacts,
3,124 spans), and the studying loop's job is to make the agent working on this repo measurably
better at it, week over week.
