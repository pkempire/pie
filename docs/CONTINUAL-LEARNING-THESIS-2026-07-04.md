# The continual-learning thesis — the science, not the evals

*2026-07-04. A reframe at the level of the underlying science: what problem this project is
actually part of (continual learning for agents), where the frontier is, the one unsolved question
our threads have been circling, and the research bet that follows. No toy eval numbers — those were
never the point.*

---

## 1. The problem is continual learning, and everything we care about is a face of it

"Memory" is the wrong frame; it points at recall. The real problem is **continual learning**: an
agent turning an ongoing stream of experience into durable competence, over time, without
forgetting. Every concrete thing we keep returning to is a sub-problem of it:

- **The planning fallacy** is an agent failing to learn from its own past outcomes — no
  accumulation of calibration.
- **Codebase / corpus expertise** is learning a body of material into competence (not lookup) —
  and keeping it current as the code changes.
- **Temporal awareness** is knowing *when* what you learned is still true.
- **Proactivity / synthesis** is a system that has consolidated enough experience to notice
  connections and contradictions unprompted.

These are not four projects. They are the acquisition, consolidation, validity, and use of learned
experience — the four moving parts of continual learning.

## 2. The actual science (stability–plasticity, and the three real questions)

Continual learning's central tension is the **stability–plasticity dilemma**: acquire new knowledge
(plasticity) without overwriting old (stability / catastrophic forgetting). For LLM agents this
resolves into three questions, two of which the frontier is attacking hard and one it is not:

**Q1 — Substrate: where does learned knowledge live?** Three answers, and the field has finally
made all three real:
- *Token / context* — no forgetting, but bounded and it doesn't compound; re-read every call.
- *Weights / parameters* — compounds, free at inference, but catastrophic forgetting, opaque, costly.
- *Activations / KV cache* — the new middle tier. **Cartridges** (Stanford, 2025) showed you can
  *compile* a corpus into a small KV cache via self-study, at ICL quality and ~38× less memory. This
  is continual learning in activation space, and it's the year's most important mechanism.

**Q2 — Consolidation: how does raw experience become durable knowledge?** The settled answer, across
neuroscience and 2026 ML, is *not* naive fine-tuning (Machine Studying proved next-token drilling on
a corpus underperforms and forgets). It's **replay + distillation**: generate structured synthetic
material *about* the experience and distill it into the durable substrate. Cartridges' "self-study,"
Thinking Machines' on-policy distillation, and the hippocampus→cortex consolidation story are the
same algorithm at different levels. This is active, crowded, and progressing.

**Q3 — Revision: how do you update durable knowledge when the world changes?** *Almost nobody is
working on this, and it is the one that breaks continual-learning agents in practice.* The instant
you consolidate experience into weights or a cache, you have frozen a **snapshot** of what was true
at consolidation time. Codebases change; policies change; facts get superseded. There is no accepted
mechanism to selectively un-learn or revise parametric/compiled knowledge — belief revision in
learned representations is essentially open. So an agent that continually consolidates months of
experience doesn't accumulate competence cleanly; it **accumulates stale, mutually-contradictory
beliefs it cannot revise.** Continual learning, done naively, is continual *calcification*.

## 3. Where our threads actually fit (at the level of the science)

This is the part I kept getting wrong by pointing at 10-question evals. At the level of the science,
the threads this project has circled for months are precisely the pieces of Q3, plus the machinery
for Q2:

- **Temporal validity** (`valid_from / superseded_by`, state-at-T) is *not* a QA trick. It is the
  representational substrate for **revision** — the only way a consolidated store can know *what*
  changed and *what to un-learn*. It's the stability side of the stability–plasticity dilemma made
  concrete. This is the deep reason the temporal thread has never died.
- **Learned consolidation** (the GEPA/DSPy consolidator, the "studying" idea) is our attempt at Q2 —
  the replay-and-distill mechanism, optimized rather than hand-coded.
- **Credit assignment / the amortized critic** is the *selection* problem inside both: of all the
  experience streaming in, which is worth consolidating into durable form, and which consolidated
  belief is worth revising? A value model over consolidation decisions.
- **The hierarchy (log → notes → cache → weights)** is our version of Q1 — and it maps to
  complementary learning systems: fast episodic capture, slow semantic consolidation.

So the project's real contribution isn't "a better memory system." It's aimed — whether we said it
this way or not — at **the revision problem of continual learning**: how to consolidate experience
into durable competence *and keep it correct as the world moves.*

## 4. The bet: revisable consolidation (temporally-valid continual learning)

The frontier can compile experience into durable knowledge (Cartridges, distillation) but produces
**static, un-revisable** artifacts. The open scientific question — the one that sits at the exact
intersection of continual learning, the temporal thread, and the real problems — is:

> **Can an agent consolidate its experience into a durable representation (a cache or an adapter)
> that carries validity, and be selectively revised when the world changes — without retraining
> from scratch and without catastrophic forgetting?**

Why this is the right bet, not another pivot:

- **It's genuinely open.** Cartridges are static; distillation is one-way; belief revision in learned
  representations is unsolved. Nobody has "revisable self-study."
- **It's frontier-native.** It builds directly on the year's key mechanism (self-study into cache)
  rather than around it, and inherits its open code and open-weight practicality.
- **It's the deep form of our durable thread.** Temporal validity finally does real work — as the
  revision interface, not a QA metric.
- **It resolves the real problems at their root:** a codebase Cartridge that updates as the repo
  changes; a planning system whose learned calibration revises as outcomes accrue; an assistant that
  updates, not calcifies, as your life changes.

## 5. The scientific questions to answer (not eval runs)

The research is defined by these questions, in order:

1. **Is compiled knowledge editable at all?** Cartridges reportedly *compose* without retraining.
   Does that composability extend to *invalidation* — can you subtract or override a specific belief
   in a compiled KV cache / adapter, or must you rebuild it? This is the crux experiment and it's
   conceptual, not benchmark-chasing.
2. **What is the right unit of revisable knowledge?** A validity-tagged "memory state" that compiles
   to a cache fragment? The temporal schema is the hypothesis for this representation.
3. **What is the objective for revision?** When new experience contradicts consolidated knowledge,
   what decides *update vs. keep vs. branch*? (Classical belief revision — AGM/JTMS — meets a learned
   value model.)
4. **Does revisable consolidation avoid the calcification failure** that static consolidation
   suffers as experience accumulates over long horizons? This is the payoff claim, and the right
   evaluation is a *long-horizon, world-changes-under-you* setting — not a static QA set.

## 6. What this means for the next move

The honest next step is not to run anything. It's to **pin the crux (question 1) against the actual
Cartridges mechanism**: get the Cartridges method running on an open-weight model and probe whether a
compiled representation can be selectively invalidated/updated, because the entire bet lives or dies
on that. If compiled knowledge is editable, revisable consolidation is a real research program with
a clear thesis. If it isn't, that itself is the finding, and it tells us revision has to happen one
tier up (in the notes that generate the cache), which is also publishable.

Everything we've built becomes machinery for this: the temporal schema is the validity
representation, the consolidator is the compile step, the critic is the revision-selection value
model, the ledger is the changing-world corpus. But the thesis is the science in §4 — continual
learning that revises instead of calcifies — not any single artifact.

---

*The reframe in one line: we are not building a memory system; we are working on the revision problem
of continual learning — how an agent consolidates experience into durable competence and keeps it
true as the world changes — which is the one part of the continual-learning frontier that self-study
and distillation have left open.*
