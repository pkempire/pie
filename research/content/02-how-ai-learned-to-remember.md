# How AI Learned to Remember
### From a model that forgets everything to one that reconstructs the past — the whole story, one rung at a time

*Working Memory. Script v1. Teaches memory from zero knowledge, as a ladder where each rung fixes the previous rung's failure, and ends exactly at the system we're building: RLM reconstruction on read, a learned write policy that compresses the log, and a learned critic that makes the training cheap. Real demos are marked [DEMO] and are runnable in `scripts/`. Production cues in [brackets].*

---

## 0. The goldfish

[SCREEN: a chat. Close the tab. Reopen. The model has no idea who you are.]

Here's the strange thing about the most powerful AI we've ever built: it has the memory of a goldfish. You can have an hour-long conversation with a language model, close the tab, and the moment you come back it has *no idea who you are.* Everything you told it is gone. Not misremembered — gone. The model that can pass the bar exam cannot remember your dog's name between Tuesday and Wednesday.

This isn't a bug someone forgot to fix. It's the default state of the technology. A language model is a function: text in, text out. When the function returns, nothing persists. There is no "yesterday."

So the entire field of *agent memory* is the story of bolting a past onto a thing that has none. And it turns out that's much harder, and much more interesting, than it sounds. Let me walk you up the ladder — because every step people tried fixes the last step's problem and exposes a new one, and the rung we're standing on right now is genuinely new.

---

## 1. Just paste it all in

[SCREEN: context window growing 4k → 128k → 1M tokens]

The first idea is the obvious one: if the model forgets when the conversation ends, *don't let it end.* Keep the whole history in the prompt and resend it every time. And for a while the industry just... made the prompt bigger. Context windows went from a couple thousand tokens to a million.

This works until it spectacularly doesn't. Three ways. **Cost:** you pay for every token, every turn — re-sending a million-token history on each message is ruinous. **"Lost in the middle":** even when it *fits*, models reliably miss facts buried in the center of a long context; more haystack, more missed needles. And **it still ends** — a lifetime of conversations exceeds any window, no matter how big. Bigger context is a faster horse, not a car. You need to store things *outside* the model and bring back only what matters.

---

## 2. The vector store — and why it's flat

[SCREEN: text → embedding arrow → point in space → top-k retrieval]

So: external memory. Write everything to a database; at query time, fetch the relevant bits and put *those* in the prompt. The trick that made this practical is the **embedding** — turn any piece of text into a point in space such that similar meanings land near each other. Store every message as a point. When a question comes in, embed it too, grab the nearest points, done. This is **RAG**, the vector store, and it is what almost every "AI with memory" runs on today. In this repo it's the `flat` backend: dense embeddings plus keyword search, fused. The honest name for it is *basic* — and that's the point.

Because here's the flaw, and it's deep. **A vector store only knows similarity. It knows nothing about truth, or time, or change.**

[DEMO — `scripts/temporal_memory_demo.py`] Tell it two things: "I'm so angry right now" and "I'm vegetarian." Both become points. A month later you ask, "is the user angry?" The store retrieves the angry sentence — it's a great *similarity* match — and the model says *yes.* It has no idea that anger has a half-life of minutes and a diet has a half-life of years. To a vector store, *"I will buy milk"* and *"I bought milk"* are near-twins; to an agent they're opposites — one is a goal, one is done. **Vectors rank by similarity. Agents need causality and time.** A flat store throws both away at the moment of writing.

---

## 3. Pull out the facts — knowledge graphs

[SCREEN: raw text → extracted (entity)—[relation]→(entity) graph]

If storing raw text loses structure, then extract the structure. Read the conversation and pull out *entities* and *relationships*: `(user)—lives_in→(Boston)`, `(user)—diet→(vegetarian)`. Now you have a **knowledge graph** — queryable, typed, and you can even attach validity ("true from January to July"). This is the lineage of Mem0, of Zep/Graphiti, and of this repo's `pie_kg` backend, which holds a real 4,000-entity, 6,000-transition graph. It's a real step up: structure the flat store never had.

Two problems remain. **Extraction is lossy** — Mem0's own paper reports a ~40% extraction-failure rate; whatever the extractor misses is gone forever. And the extractor is **hand-written** — a human wrote the prompt that decides what counts as a fact. Which raises the obvious question: *why are we hand-writing the most important part?*

---

## 4. Let the model reflect — observational memory

[SCREEN: raw turns → [Observer] → dated notes → [Reflector] → condensed memory]

The next rung makes the writing itself smart. Instead of regex-y extraction, you run a small LLM pass over the raw stream: an **Observer** turns turns into dated observations, and a **Reflector** condenses those into durable memory. This is Mastra's **Observational Memory**, the `mastra` backend here — and on the standard long-memory benchmark it posts some of the best published numbers, with no graph at all, just good reflection. The lesson: *how you write* matters more than *what you store it in.*

But the Reflector is still a frozen, hand-tuned prompt — a human's best guess at what's worth keeping, written once and never corrected by whether it actually *helped.* We've made the writer smart, but not *learned.*

---

## 5. Learn what to keep

[SCREEN: write decisions as actions; a reward signal flowing back from a future correct answer]

Here's the leap. Stop hand-writing the memory policy and **train it.** Treat every write — keep this, drop that, merge these, mark this stale — as an *action*, and reward the actions that make *future questions answerable.* This is Memory-R1, and it's what `mempol` does: a write policy trained with reinforcement learning against downstream answer accuracy. Now "what matters" isn't a human guess — it's learned from outcomes. *Facts live in the store; the strategy for managing them lives in the weights.*

The catch is brutal, and it's a real research problem: **credit assignment.** When a future answer is right, *which* of the hundred earlier write decisions deserves the credit? The honest way to find out is counterfactual — replay the memory without each write and see if the answer breaks. But that means re-running the reader for every write times every question: in this codebase, `(K+1)×Q` replays *per trajectory* — thousands of model calls per training step. Brute force.

[DEMO — `scripts/critic_counterfactual_smoke.py`] So we cheat, cleverly. Compute the expensive counterfactual for just a *few* writes, then train a tiny **critic** to *predict* the value of the rest from cheap features. In a smoke run, a 5-feature critic trained on 8 exact deltas predicts held-out per-op value at **r = 0.71** with zero extra replays — and it fails in exactly the spot theory says it should (redundant facts), which tells us where to look next. The expensive signal becomes a shrinking teacher. This is the 2026 frontier of RL credit assignment (CCPO, AgentPRM), pointed at memory.

---

## 6. The fork in the road: compress now, or reconstruct later

[SCREEN: two paths — left "compress at write time", right "store raw, recompute at read time"]

Step back, because the field just split in two. Everything so far spends effort at **write time** — reflect, extract, learn — to compress experience into a tidy store. Call it *compress-now.*

The other road spends effort at **read time.** Store the raw log, mostly untouched, and when a hard question comes, let the model **recurse over the log** — slice it, reason over each slice, combine — reconstructing the answer on demand. These are **Recursive Language Models**, and they're startlingly good: an RLM wrapping a *small* model beats a frontier model on long-context tasks. *Decompress-later.*

Why does read-time buy you something write-time can't? Because some questions can only be answered by *reconstructing the past*, and a compressed store has already thrown the past away.

[DEMO — `scripts/rlm_temporal_reconstruction.py`] Give both a year of someone's life — moved Boston→NYC in August, went vegetarian→pescatarian in July. Ask: *"where did they live in May?"* The flat store retrieves "Boston" and "NYC" with no sense of order and guesses **NYC — wrong.** The RLM reader rebuilds the timeline from the raw log and answers **"in May, Boston"** — correct, because it *reconstructed the state as of May* instead of looking up a current value. Validity isn't stored; it's recomputed. Flat 67%, reconstruction 83%, and the gap is entirely the as-of-the-past questions a compressed store structurally cannot answer.

---

## 7. Where the ladder actually leads

[SCREEN: the three pieces snapping together into one loop]

Now put the top three rungs together, because they aren't competitors — they're organs of one system:

- **Read side:** an RLM **reconstructs the state as of any time** from the raw log — nothing important is thrown away, and "what was true when" is answerable.
- **Write side:** a **learned policy compresses** the log so the reader isn't recursing over everything forever — amortizing the read-time cost.
- **Training:** a **learned critic** makes the per-decision reward cheap enough to actually train the write policy at scale.

And the thread running through every rung — the one the whole ladder kept tripping over — is **time.** The flat store ignored it. The graph bolted it on as metadata. Reflection summarized over it. The honest target isn't "remember more"; it's a memory that knows *what was true, when, and whether it still is* — and recomputes the answer rather than trusting a stale snapshot.

That's the system we're building, and the rung is new enough that the pieces don't have a name yet. The goldfish is learning to time-travel. That's the interesting part.

[SCREEN: end card]

---

### The real artifacts behind each rung (no mock data)
| Rung | In this repo | Demo / result |
|---|---|---|
| 2 — flat vector | `mempol/backends/flat.py` | `temporal_memory_demo.py`: flat 75% → temporal 100% |
| 3 — knowledge graph | `mempol/backends/pie_kg.py` (4k entities) | `reflector_backend_matrix.py` (kg cell) |
| 4 — observational | `mempol/backends/mastra.py` | `reflector_backend_matrix.py` (mastra cell) |
| 5 — learned write | `mempol/recipes/memory_rl/` + `eval/counterfactual.py` | `critic_counterfactual_smoke.py`: r=0.71 |
| 6 — RLM read | new | `rlm_temporal_reconstruction.py`: 67% → 83% |

### Honesty notes for the record
- The GEPA "reflector" result (0.6→0.8) is **in-sample, 5 questions, 1 chunk** of conv-26 — a pipeline proof, not a held-out claim.
- `mempol`'s write reward is currently scored against a **frozen heuristic reader** (trained-reader is `NotImplementedError`) — making R the RLM reconstructor is the next real step.
- Backend numbers for the matrix come from `reflector_backend_matrix.py` once run (command below).
