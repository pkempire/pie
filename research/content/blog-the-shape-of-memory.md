---
title: "The Shape of Memory"
subtitle: "Where AI keeps what it learns — and why the whole map is about to be redrawn"
author: "Working Memory"
status: "draft v1 — foundational essay, 2026-07-03"
target_length: "~3,200 words / 14 min read"
voice: "Distill / Gwern / Karpathy register. Concrete over abstract. No survey-speak. A point of view."
---

# The Shape of Memory

*Where AI keeps what it learns — and why the whole map is about to be redrawn.*

The most capable software ever built has the object permanence of an infant. You can spend an
hour teaching a language model the shape of your problem — your codebase, your constraints, the
three approaches you already ruled out — close the tab, and it is gone. Not misfiled. Gone. The
model that will out-argue a lawyer cannot remember, unprompted, that you are allergic to
penicillin.

This is not a bug anyone forgot to fix. It is the resting state of the technology. A language
model is a pure function: text in, text out, no side effects. When the call returns, the world
it briefly understood ceases to exist. There is no yesterday.

So the entire industry now labeled "agent memory" is one long attempt to bolt a past onto a thing
built without one. Billions of dollars, a dozen well-funded startups, first-party features from
every frontier lab. And yet if you talk to the people actually shipping agents, the mood is not
triumph — it's a low, persistent dissatisfaction. The memory works, technically, and the agents
still don't feel like they *learn*.

I think that dissatisfaction is diagnostic, and I want to explain it precisely. The field has
been answering two different questions as if they were one, optimizing hard for a goal that turns
out to be the wrong goal, and — right now, in mid-2026 — quietly discovering the goal it should
have had all along. This essay is a map of where we are and an argument about where we're going.
Its one claim, up front: **we built memory for recall, and what we actually wanted was learning.**

---

## Two questions, endlessly conflated

Every memory system is really an answer to two independent questions. Almost nobody separates
them, and the confusion is the source of half the field's arguments.

**Question one: where does what-the-agent-knows physically live?** There are exactly three answers,
and they form a hierarchy from hot to cold:

- **In the prompt — token space.** The knowledge is text, re-read from scratch on every call.
  Maximally flexible, impossible to corrupt silently, and expensive: you pay for every token,
  every turn.
- **In the cache — activation space.** The knowledge has been pre-digested into the model's
  key–value cache, so it's "already read" and reused across calls. Cheaper per query, but tied to
  a model and a moment.
- **In the weights — parameter space.** The knowledge has been trained into the network itself.
  Nearly free to use — zero context tokens — but expensive to write, opaque, and prone to
  forgetting everything around it.

**Question two: when do you spend compute to organize the knowledge?** This is a spectrum with two
poles:

- **At write time — compress now.** Do the work up front: extract the facts, reflect on the
  conversation, decide what's worth keeping. Reads are then cheap. If you got the extraction
  wrong, you're wrong forever.
- **At read time — reconstruct later.** Store the raw stream almost untouched and do the thinking
  when a question actually arrives — slice the log, reason over the pieces, rebuild the answer on
  demand. Nothing is lost, but every query pays the bill.

Put the two questions on two axes and you get the real map of the field. Here is where the actual
systems sit, mid-2026:

```
   WHERE it lives
   (parameter)  weights │                              Memory-R1, Mem-α       on-policy
                        │                              (research only)        distillation
                        │                                                     Letta "memory
                        │                                                     models" (vision)
   (activation) cache   │  MemOS "activation memory"           Ramp latent-briefing
                        │  (gestured at)                       (KV sharing)
                        │
                        │  ┌─────────────────────────────────────────────┐
   (token)     tokens   │  │ Mem0 · Zep/Graphiti · Mastra · Letta ·       │   RLM (recursive
                        │  │ Supermemory · Honcho · Cognee · Hindsight ·  │   LMs) · Search-R1 ·
                        │  │ MemOS · Basic Memory · LangMem  ← EVERYONE    │   agentic RAG
                        │  └─────────────────────────────────────────────┘
                        └───────────────────────────────────────────────────────────────
                            write-time  ◄─────── WHEN you organize ───────►  read-time
                            (compress now)                                   (reconstruct later)
```

Stare at that for a second, because the shape of it *is* the argument.

Almost the entire commercial field lives in a single cell: **token space, write time.** Mem0
extracts facts into a vector store ([mem0.ai](https://mem0.ai)). Zep builds a temporal knowledge
graph and keeps it in a database ([Graphiti](https://github.com/getzep/graphiti)). Mastra runs an
Observer and a Reflector to compress the log into dated notes
([observational memory](https://mastra.ai/research/observational-memory)). Letta writes
git-backed markdown files ([context repositories](https://www.letta.com/blog/context-repositories)).
Supermemory, Honcho, Cognee, Hindsight, MemOS, Basic Memory, LangMem — different data structures,
different pricing, genuinely different quality, but all fundamentally the same move: *do clever
work at write time to compress experience into text you retrieve later.* Even the celebrated 2026
innovation everyone converged on — background "dreaming," shipped by
[OpenAI](https://openai.com/index/chatgpt-memory-dreaming/), Letta, Honcho, and Supermemory within
months of each other — is a refinement inside this one cell: consolidate the text better, while
the user sleeps.

The other two rows are nearly empty. **Activation space** has a couple of gestures — MemOS calls
KV-cache injection "activation memory," Ramp shared a compressed-KV briefing trick — but no one has
made the cache a real memory tier. **Parameter space** is empty of products entirely; it exists
only in research (Memory-R1 and Mem-α train a memory-management policy;
[arXiv:2508.19828](https://arxiv.org/abs/2508.19828)) and in Letta's
["memory models" manifesto](https://www.letta.com/blog/towards-agents-that-learn/), which describes
the vision and ships nothing.

And the read-time column has essentially one serious inhabitant: Recursive Language Models
([RLM](https://arxiv.org/abs/2512.24601)), which keep the raw log and let a model recurse over it
at query time — and which, startlingly, let a *small* model beat a frontier model on long-context
tasks. Read-time is a real, underexplored road, and I'll come back to why.

So the map has a heavily overcrowded corner and vast open space. That alone should make you
suspicious. When an entire industry piles into one cell, it's usually because that cell is where
the easy wins were, not where the important problem is.

---

## The debate that isn't real

Because people conflate the two questions, they turn a placement decision into an ideology.

You'll read that "memory should live in context, not weights" — that's Letta's public position,
and [their argument](https://www.letta.com/blog/continual-learning) is genuinely good: token-space
memory is human-readable, portable across models, and debuggable, while weight updates are opaque
and lock you to one model. You'll read the opposite from the fine-tuning camp: real skill lives in
weights, and paying to re-read the same context forever is absurd. Google's
[Nested Learning](https://research.google/blog/introducing-nested-learning-a-new-ml-paradigm-for-continual-learning/)
tries to dissolve the distinction; Thinking Machines shows you can move knowledge from context into
weights cleanly with [on-policy distillation](https://thinkingmachines.ai/blog/on-policy-distillation/).

Here's the thing: **none of them is right in general, because it isn't a question with a general
answer.** Where a piece of knowledge should live is an economic decision about *that piece*. A fact
you'll use once belongs in the prompt. A fact you'll use ten thousand times, that never changes,
is wasting money in the prompt and belongs in the weights. A fact you'll reuse this week but that
might change belongs in the cache. Frequency, stability, and value decide placement — per item,
not per ideology.

Nobody has written down that decision rule. There is no crossover formula that says "distill this
once it's been used *k* times and survived *n* updates." The field is having a philosophical
argument about a question that wants an equation. That equation — a *placement law* for agent
knowledge — is, I'll argue, one of the two or three most valuable things left to derive here.

---

## The thread the whole map keeps tripping over: time

There's a defect that runs through every cell of the map, and it's worth isolating because it's the
cleanest illustration of why "storage and recall" is too weak a frame.

A store that retrieves by similarity has no concept of *when*. It knows that two statements are
related; it does not know which one superseded the other. So it fails a whole class of questions
that look trivial to a human.

Take a year of someone's life: they moved from Boston to New York in August, and went from
vegetarian to pescatarian in July. Ask a standard vector store, *"where did they live in May?"* It
retrieves "Boston" and "New York" — both are excellent similarity matches for a question about
where they live — and, having no sense of order, it hands you the most prominent one. Usually the
current one. Usually wrong.

I ran exactly this. A flat retrieval reader scores 20% on "as-of-the-past" questions like that one.
A reader that first rebuilds a *timeline* from the raw log — placing each fact on a line and reading
off the value that was true as of the month asked — scores 100%. (The full, runnable version is
[in the repo](https://github.com/pkempire/pie/tree/main/demos/01-stale-memory); it costs about a
cent to reproduce.) And here's the detail that should bother you: **a smarter base model makes the
flat store *worse*, not better.** Swap in a stronger model and its as-of-the-past accuracy drops,
because it reasons more confidently over evidence that has no timestamps. The failure isn't a lack
of intelligence at read time. It's structural. The store threw the timeline away at write time, and
no amount of cleverness downstream can recover what was never kept.

This is why the read-time road matters, and why the industry's write-time monoculture is a real
limitation and not just a stylistic choice. *Some questions can only be answered by reconstructing
the past, and a store optimized to compress the past has already discarded it.* On the standard
stale-memory benchmark, production memory frameworks — Mem0, Zep — score in the
[single digits](https://arxiv.org/html/2605.06527). Not because they're badly built. Because they're
built for recall, and this is a question about history.

Which brings us to the actual problem.

---

## Recall was never the goal

Here is the sentence the whole field is organized around without quite admitting it: *a memory
system is good if, when you ask it about something you told it, it can retrieve that thing.* Every
major benchmark — LoCoMo, LongMemEval — measures exactly this. Plant a fact, ask about it later,
check if it comes back.

And in 2026 that entire evaluation edifice quietly fell apart. An audit of LoCoMo, the most-cited
memory benchmark, found that
[6.4% of its answer key is simply wrong](https://github.com/dial481/locomo-audit), and that the
standard automatic judge accepts *63% of deliberately incorrect answers*. LongMemEval's top scores
turned out to be
[eval-stack engineering](https://www.maximem.ai/blog/state-of-ai-memory-2026-claimed-vs-observed) —
one vendor's claimed 93% reproduced at 74% under a frozen, honest judge. The leaderboards were
measuring the judge's leniency, not the memory.

It would be easy to read that as "the benchmarks are sloppy, let's build better benchmarks." That's
the small lesson. The large lesson is that *recall was the wrong target in the first place.*

The clearest demonstration I know came out this year, in a quiet post called
["Machine Studying"](https://jacobxli.com/blog/2026/machine-studying/). The author set up a simple
test: give an agent a corpus about a topic it doesn't know, and see if it can become genuinely
expert. Two findings mattered. First, retrieval is not expertise — two equally-capable models given
search over the *same* documents retrieved the same passages, and the one that already knew more
kept the right ones while the other discarded them. "Nothing failed in retrieval." The gap was
knowing which retrieved thing mattered, and that is not something you can look up. Second — and this
is the uncomfortable one — *fine-tuning on the corpus didn't work either.* Drilling the model on
synthetic questions made it better at closed-book recall and no better as an expert. Memorization is
not competence.

The only thing that worked was a studied artifact — a compressed, worked-through "cheatsheet" the
agent built by actually processing the corpus, which then made it dramatically more effective per
unit of thinking.

Sit with what that implies for the map. We have built an enormous industry to answer *"can you
retrieve what I told you"* — and the actual question, the one that determines whether an agent is
useful, is *"did you get better at the thing."* Those are not the same question, and a system tuned
for the first can be flatly bad at the second. An agent can recall every line of your codebase and
still not have learned it. Recall is a lookup. Competence is a change in the agent.

That is the redrawing. The map I showed you — every company, every repo, all that funding — is a map
of the **recall era.** It optimizes *where knowledge sits and when you tidy it.* The next era
optimizes something else entirely: *how much the agent improves from experience, per unit of compute
it spends.* Call it the competence axis. It runs perpendicular to the entire map, and almost nobody
is on it yet.

---

## What the competence era needs

If the goal is an agent that measurably improves — not one that retrieves better, one that *is*
better — then three problems become the whole game. None of them is solved. All three, I think, are
within reach, and I want to be concrete about each.

**A way to train memory without labels.** Every system today is tuned against a benchmark of
question–answer pairs. But real deployments have no answer key, and the benchmarks are broken
anyway. So the binding constraint on learned memory is embarrassing: *there is no signal for whether
a memory decision was good.* Except there is, and it's everywhere — the user's *next* message. If an
agent's memory is good, the next interaction should go better: the model should predict it, serve
it, need fewer clarifications. That's a free, abundant, un-gameable training signal sitting in every
interaction log on earth. Frame memory as keeping the *minimal* summary of the past that best
predicts the future, and you can train the whole thing on raw logs, no labels, no judge. This is,
plausibly, the "next-token prediction" moment for memory: a self-supervised objective that scales
with data instead of annotation. Nobody has shipped it. It might be the single highest-leverage
open problem in the field.

**A way to study, not just store.** The Machine Studying result says the valuable artifact is a
*studied* one — but it built that artifact with a fixed, hand-written procedure. The real skill is
knowing *how* to study a new domain: what to focus on, what to quiz yourself on, what to work
through. That's a policy, and it can be learned, graded by a brutally honest signal — how much
better the agent gets at real tasks it didn't study for. An agent that learns to study is an agent
you can point at a new codebase on Monday and trust on Friday. That capability — autonomous
onboarding to a domain — is precisely what every coding-agent team is failing at right now (Cursor
[removed its memory feature](https://forum.cursor.com/t/cant-clear-memories/148254) rather than fix
it; Devin's knowledge base is hand-curated). It's the applied prize.

**A rule for where knowledge should live.** The placement law from earlier. Once you're optimizing
for competence, the token/cache/weights question stops being ideology and becomes routing: send each
piece of knowledge to the cheapest tier that's sufficient, given how often it's used and how often it
changes. Derive the crossover conditions and you've written the "Chinchilla for memory" — the result
every agent builder needs and nobody has. It's also what finally puts the empty rows of the map to
work: the cache tier and the weight tier aren't rival ideologies, they're where stable, frequently-
used knowledge is *supposed* to migrate as it proves itself.

Notice these three fit together. A label-free signal tells you *whether* a memory decision helped. A
studying policy decides *what* to distill from raw experience into usable expertise. A placement law
decides *where* that expertise should live as it stabilizes. Together they describe not a memory
system but a *learning* one — an agent whose experience flows from raw log, to studied notes, to warm
cache, to weights, each step earned by evidence that it helped. That flow has a name in neuroscience:
it's roughly what your hippocampus and cortex do while you sleep. It does not yet have a name in AI,
because nobody has built it whole.

---

## Where this goes

The honest state of things: the recall era is mature, crowded, and slightly stuck. The systems are
real and useful — if you need an agent to remember a user's preferences, buy Mem0 or Zep and move on.
But the dissatisfaction that started this essay is correct. Those systems make agents that *retrieve*,
and the thing we want is agents that *learn*, and the gap between those is not an incremental one. It's
a different axis.

The next few years of this field will be the migration onto that axis: from memory-as-storage to
memory-as-learning, measured not by whether the agent can find what you said but by whether it got
better because you said it. The pieces are visibly assembling — the self-supervised training signal,
the studying policy, the placement law, the sleep-time consolidation loop everyone already built for
the wrong reason and can now repoint at the right one. Whoever puts them together first won't have
built a better memory product. They'll have built the first agent that actually learns on the job, and
that is a different kind of thing.

The goldfish was never going to be fixed by a bigger bowl. It gets fixed when it stops trying to hold
the water and starts learning to swim.

---

*Working Memory is a series on how language models remember, reason about time, and learn from
experience. The code, demos, and the full research map behind this essay are open at
[github.com/pkempire/pie](https://github.com/pkempire/pie). Next: how a memory system learns to study —
and the label-free objective that makes it trainable.*
