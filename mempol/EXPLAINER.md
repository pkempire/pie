# mempol, explained simply — and how to fix the reader

Two parts: (1) the whole system in plain words, (2) the learned-reader design (task #13).

---

## 1. What mempol is (6th-grade version)

An AI you chat with **forgets everything** when the conversation ends. So we give it a
notebook. But that just moves the problem: *what should go in the notebook?* Write down
too much and it's a messy, expensive pile. Too little and it forgets the thing you needed.

Most systems have a **human write the rules** for what to jot down. mempol's idea is
different: **teach the AI to take good notes by quizzing it later.** Like a teacher who
grades a student's notes not by how neat they look, but by whether the student can answer
questions with them a week later. If the notes led to right answers, that note-taking
habit gets rewarded. Do this thousands of times and the AI *learns* what's worth keeping.

The system has five parts:

1. **The writer** (write policy) — reads the conversation and decides: jot this down,
   update that, cross this out. *This is the part mempol trains.*
2. **The notebook** (backend / store) — where notes live. It can be a plain list
   (`flat`), a web of connected facts (`pie_kg`, a knowledge graph), or a tidy diary
   (`mastra`). Same notes, different filing systems.
3. **The reader** (read policy) — when you ask a question, it flips through the notebook
   and answers. Right now it's a simple "find the most similar-sounding notes" search.
   *This is the weak part — see Section 2.*
4. **The grader** (reward) — to know if the writer did well, we quiz the notebook with
   questions whose answers we already know. Good notes → right answers → high score.
5. **The hard part** (credit assignment) — say the AI answers 100 quiz questions and
   misses some. *Which note (or missing note) caused the miss?* mempol finds out by
   removing one note at a time and re-quizzing — accurate but **very** expensive
   ((notes+1) × questions re-quizzes). The **learned critic** predicts that score cheaply
   instead, so we don't have to re-quiz for every note.

**How the writer learns:** reinforcement learning (GRPO) — try many note-taking
strategies, keep the ones that lead to better answers. And **GEPA** — a cheaper trick that
*rewrites the instructions* by reflecting on mistakes, instead of retraining weights.

**The end product:** a small plug-in (a LoRA adapter) you can attach to an AI to give it
the good note-taking habit — without retraining the whole model.

### What "state reconstruction" means (plain)
A normal notebook stores **the latest answer**: "the user lives in NYC." Ask *"where did
they live in May?"* and it just hands back "NYC" — wrong, because they moved there in
August. **State reconstruction** keeps the whole **dated diary** of events and, when asked
about May, **rewinds the diary to May** and works out what was true *then*. It's the
difference between only seeing the last frame of a movie and being able to rewind to any
scene. "State-at-T" = what was true at time T; "reconstruct" = rebuild it from the log on
demand instead of trusting a saved value that may be stale.

### The whole thesis, in one breath
- **Read side (RLM):** rewind the diary to reconstruct what was true *then*, instead of
  trusting a possibly-stale saved answer.
- **Write side (learned policy):** keep the diary tidy so the reader isn't rewinding
  millions of lines every time — the writer compresses the raw stream.
- **Training (learned critic):** grading the writer is expensive; a critic predicts the
  grade cheaply so we can actually afford to train.
- **The thread through all of it:** *time / validity* — knowing what's true **when**, not
  just what's stored.

---

## 2. The learned reader — replacing the dumb BM25 hybrid (task #13)

**What's there now.** The reader is `HeuristicPolicy`: reformulate the question → hybrid
retrieve (BM25 keyword + dense embeddings, fused by RRF) → take the top-k → stuff them in
the prompt → answer. One shot. Fixed k. No reasoning about *what* to fetch, no follow-up,
no sense of time. Similarity is not relevance — "I will buy milk" and "I bought milk" look
identical to it. You're right that it's the weak link.

**What it should be: an agentic, tool-using, recursive reader.** Don't give the reader a
fixed top-k; give it **tools over the store** and let it *reason and fetch iteratively*,
like a researcher rather than a single lookup:

- `search(query)` — the current retrieval, but now *one tool among several*
- `get_transitions(entity)` — the change history of a fact (was → became)
- `reconstruct_state_at(entity, T)` — the RLM move: rebuild what was true at time T
- `expand(node)` / `get_neighbors(node)` — follow the graph for multi-hop questions
- `done(answer)` — stop and commit

The reader decides: do I have enough? If not, fetch more; if the question is temporal, call
`reconstruct_state_at`; if it's multi-hop, `expand`. For a huge log it **recurses** —
slice the log, sub-call itself on each slice, combine (the RLM pattern). This is strictly
more powerful than top-k, and it's the only design that can answer "what was true when."

**How to train it — cheap before expensive:**

1. **GEPA first** (~$50, no weight training). Evolve the reader's prompt + tool-use
   instructions against QA accuracy. This is the *exact* trick that took the consolidator
   0.6 → 0.8 — reflective prompt evolution, ~35× cheaper than RL. Likely gets most of the
   win on its own.
2. **GRPO next** (only if GEPA plateaus). RL-train the reader as a tool-using policy
   (Search-R1 style): reward = answer correctness; it learns *when* to search, expand,
   reconstruct, and stop. Make the per-step credit affordable with the **amortized critic**
   from `critic_counterfactual_smoke.py` (r=0.71) instead of brute-force re-rollouts.
3. **Substrate change:** stop treating BM25+dense+RRF top-k as *the* retrieval. Keep it as
   one tool; let the learned reader choose.

**Why this also fixes mempol's deepest crack.** The code review found the write reward is
judged by a *frozen heuristic reader* (`trained-R = NotImplementedError`). So today the
writer is being graded by a keyword-matcher — "did this note help a dumb reader?" Swap in a
**strong learned/RLM reader as R**, and the write reward finally means "did this note help a
*competent* reader." The learned reader isn't just a read-side upgrade; it's the unlock for
training the write side too.

**First experiment (the learned-reader smoke):** take `locomo_temporal_eval.py`, replace
the flat top-k reader with an agentic reader that has a `reconstruct_state_at` tool,
GEPA-optimize its prompt against the 37 real temporal questions, and measure the lift over
plain top-k. That's the cheapest path to "the learned reader beats the dumb one," on real
data.
