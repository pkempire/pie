---
title: "Time-aware memory"
year: 2026
category: "capability-gap"
our_status: "ship-tonight"
tags: ["temporal", "time", "memory", "agents", "video-script"]
---

# Time-aware memory

## Working title

The Clock Agents Cannot Feel

## One-sentence thesis

LLM agents do not fail at time because they cannot solve date puzzles. They fail because they do not maintain elapsed time as part of their own state, and memory systems make that failure permanent when they store changing human context as timeless facts.

## The distinction the whole video depends on

People say "temporal" and mean three different things.

**Temporal reasoning** is reasoning *about* time. Given dates, durations, and events, can the model compute what happened first, what happened next, or what was true on a stated date?

**Temporal awareness** is behaving *inside* time. Can the agent sense that eight hours passed since a tool call, that a deadline is approaching, that a background task should now be done, or that a cached answer is no longer safe?

**Temporal memory** is reconstructing state *across* time. Can the memory system answer not just "what is true now?" but "what was true in May?", "when did this change?", and "how stale is this belief?"

Most systems improve temporal reasoning and call it time awareness. That is the bug. Giving a model a timestamp gives it a number to read. It does not give the agent a clock it can feel.

## Recording script

Here is the problem in one example.

Imagine I tell my assistant two things:

```text
I am angry right now.
I am vegetarian.
```

Most memory systems write both of these the same way. They embed the sentence, store it in a vector database, maybe attach a timestamp, and later retrieve it if the query is similar.

But these are not the same kind of truth.

"I am angry" might stop being useful in twenty minutes. "I am vegetarian" might remain useful for years. A flat memory store does not know that. To the database, both are just facts. To a human, one is a mood and one is a durable preference. The difference is time.

This is why I think the next important layer in agent memory is not just better retrieval. It is time-aware state.

And this is where the field is confusing two different capabilities.

Temporal reasoning is when a model solves a puzzle about time. If I say, "The meeting was three days after Friday," can it compute Monday? That is reasoning over time as content.

Temporal awareness is different. It is the agent knowing that time is passing while it exists. If a user asked for the weather at 9 AM and asks again at 5 PM, should the agent reuse the cached weather or call the tool again? If a negotiation has a five-minute deadline, should it become more urgent as the deadline approaches? If a task started in the background, should the agent come back to it after enough time has elapsed?

That is not date math. That is a maintained state variable.

The evidence is now pretty direct.

In TicToc, from Cheng et al., the benchmark asks whether agents make human-aligned tool-use decisions when time has passed between turns. The paper reports that no model gets above 65% normalized alignment even when timestamp information is provided. The authors explicitly argue that naive prompt-based fixes have limited effectiveness. Source: [arXiv 2510.23853](https://arxiv.org/abs/2510.23853).

In Real-Time Deadlines, Sehgal, Guntuku, and Ungar put LLM negotiators under wall-clock deadlines. GPT-5.1 closes 4% of deals when it only knows the global time limit. If the remaining time is injected every turn, closure rises to 32%. The same models do almost perfectly under turn-based limits. That means the missing capability is not strategy. It is internal tracking of elapsed real time. Source: [arXiv 2601.13206](https://arxiv.org/abs/2601.13206).

In Robotouille, the same planning tasks become much harder when actions take time and overlap. ReAct with GPT-4o gets 47% on synchronous tasks and 11% on asynchronous ones. Same model, same broad task family, but now the world keeps moving while the agent acts. Source: [arXiv 2502.05227](https://arxiv.org/abs/2502.05227).

These are three different papers, but to me they point at one missing variable: elapsed time is not maintained by the agent.

A transformer sees tokens. It does not experience waiting. Between a message at 9 AM and a message at 5 PM, the model does not live through eight hours. It receives a new context window. You can paste "current_time: 5 PM" into the prompt, and that helps, but it is still just another token. The agent is reading a clock, not running on one.

That distinction matters for memory.

Because if the model is time-blind, and the memory system is also time-blind, the whole agent becomes time-blind twice. The model cannot feel duration, and the database stores old facts as if they are permanently current.

This is why conflict resolution by overwriting is so destructive.

Suppose a memory system has:

```text
2024: user is vegetarian
2026: user is pescatarian
```

A normal memory system says: contradiction detected, update the fact. Delete or overwrite vegetarian. Current state: pescatarian.

That seems reasonable. But it destroyed the most important information: the transition.

The useful memory is not just "user is pescatarian." The useful memory is:

```text
diet: vegetarian -> pescatarian
changed: 2026
previous state lasted: about two years
current confidence: high
future update rate: probably months to years, not minutes
```

The delta is the intelligence. The fact that the person changed tells you more than either snapshot alone.

This is the bridge from temporal awareness to temporal memory.

Temporal awareness is the agent's missing clock. Temporal memory is the store that preserves how the world changes across that clock.

The unit of memory should not be the fact. It should not even be the chunk. The unit of memory should be the state transition.

That means every retrieved memory should carry:

```text
what changed
when it changed
what it replaced
how long it is expected to remain valid
how confident we are now
what evidence supports it
```

Now retrieval changes.

A flat store returns:

```text
The user is vegetarian.
The user is pescatarian.
```

A temporal store returns:

```text
Diet changed from vegetarian to pescatarian in July.
The newer state supersedes the older one.
Diet facts usually change slowly, so current confidence is high.
```

That is not just more metadata. It changes the answer the model is able to give.

We built a small version of this in this repo.

In `scripts/temporal_memory_demo.py`, a flat store and a temporal store ingest the same facts. Thirty days later, the flat store still answers "yes" to "is the user angry right now?" because it retrieves the old anger sentence. The temporal store annotates the mood as expired and answers no. On that small controlled test, flat gets 75%, temporal gets 100%.

In `scripts/rlm_temporal_reconstruction.py`, the flat baseline retrieves whichever fact is semantically closest. When asked "Where did the user live in May?", it says NYC because the later NYC fact is highly relevant. The read-time reconstruction system builds a timeline and answers Boston. On that controlled test, flat gets 67%, RLM-style reconstruction gets 83%, and one of the misses is judge noise because the answer "vegetarian as of March" is actually correct.

The important part is not the tiny synthetic score. The important part is the failure mode. Flat retrieval has no natural way to answer "what was true at T?" once a fact changes. It retrieves facts; it does not reconstruct state.

This suggests a clean architecture:

```text
raw event log
  -> read-time reconstruction: what was true at T?
  -> write-time consolidation: what transitions are worth storing?
  -> temporal retrieval: return memories with validity, supersession, and confidence
```

Read-time reconstruction solves the question "what was true then?"

Write-time consolidation solves the storage problem: you cannot keep every raw event forever, so you compress the event log into durable transitions.

Temporal retrieval solves the prompt problem: the model does not just see text; it sees the temporal status of the text.

This is where the current research thread connects to GEPA and RL.

The hard question is: how do we learn which memories to write?

One path is brute-force counterfactuals: delete a candidate memory and rerun the whole future QA suite. If answers get worse, that memory was valuable. This is scientifically clean but computationally insane. It makes every write decision cost many model calls.

The better path is to learn a critic. Compute exact counterfactual deltas for a small sample of writes, train a cheap predictor of write utility, and use that critic to score the rest. This repo has a first controlled version in `scripts/critic_counterfactual_smoke.py`: exact per-op deltas for a few operations train a tiny critic that predicts held-out op advantage with much fewer rollouts. It is not the final result, but it points at the right optimization target.

The other path is GEPA: use language feedback from failed trajectories to evolve the consolidator prompt. GEPA is relevant because it extracts much more signal from each rollout than scalar RL. The GEPA paper reports that it beats GRPO by 6% on average across six tasks while using up to 35x fewer rollouts. Source: [arXiv 2507.19457](https://arxiv.org/abs/2507.19457).

That is why our strongest near-term experiment is not "train everything end-to-end immediately." It is:

```text
Compare raw retrieval, temporal KG, Mastra-style observational memory,
hand-written consolidation, and GEPA-learned consolidation
on the same LoCoMo questions, with the same reader and judge.
```

We already have partial numbers on conv-26:

```text
flat raw retrieval: 56.7%
cached PIE temporal KG: 70.0%
Mastra observational memory: 40.0%
```

The missing cells are hand-written consolidation and GEPA-learned consolidation. Those are currently running or runnable from `scripts/reflector_backend_matrix.py`.

So the video conclusion is not "I have solved time-aware agents."

The honest conclusion is better:

Current agents are missing a clock. Timestamp injection helps but does not solve the missing state variable. Flat memory makes the problem worse because it stores changing context as timeless facts. The fix is a temporal world model where memory is a trajectory of state transitions, and the research problem is learning which transitions to preserve under a budget.

That is the publishable idea.

Not "my vector DB has timestamps."

Not "my prompt says current date."

But:

```text
agents need a clock;
memory needs transitions;
retrieval needs temporal validity;
learning needs future utility under budget.
```

Once you see that, a lot of agent failures stop looking random.

The agent does not pick up a dropped thread because it has no model of how long the thread has been stale.

It does not know a cached answer expired because expiration is not represented.

It cannot orchestrate multi-agent work because it does not maintain live state for what each worker is doing over time.

It cannot be proactive because proactivity is choosing when to act, and choosing when to act requires a clock.

We have spent two years giving agents tools, retrieval, and longer context windows.

The next layer is time.

Not time as a string in the prompt.

Time as a state variable the system maintains.

Time as validity in memory.

Time as the thing that turns a pile of facts into a changing world.

## Demo beats for the video

Use these as screen inserts.

1. **Cold open:** two cards: "I am angry right now" vs "I am vegetarian." Draw half-life curves: mood decays fast, diet decays slowly.
2. **Paper evidence:** show TicToc 65%, Real-Time Deadlines 4% to 32%, Robotouille 47% to 11%.
3. **Your code demo 1:** run `python scripts/temporal_memory_demo.py`. Show flat memory saying the user is still angry. Show temporal memory marking the mood expired.
4. **Your code demo 2:** run `python scripts/rlm_temporal_reconstruction.py`. Show "Where did the user live in May?" Flat says NYC; reconstruction says Boston.
5. **Research result in progress:** show `scripts/reflector_backend_matrix.py` comparing flat, KG, Mastra, hand consolidation, GEPA consolidation.
6. **Closing visual:** event log becomes transition graph becomes time-aware retrieval prompt.

## Tonight publishing plan

If recording tonight, do not over-edit.

Record in sections:

1. Hook: angry vs vegetarian.
2. Distinction: reasoning vs awareness vs memory.
3. Evidence: three papers.
4. Demo: temporal memory and RLM reconstruction.
5. Research direction: transition memory + learned consolidation.
6. Close: agents need a clock.

For editing, use paper screenshots, terminal clips, and simple diagrams. Avoid full animation. This wants to feel like a serious researcher's field note, not a generic explainer.

## Related repo artifacts

- `scripts/temporal_memory_demo.py` - controlled flat vs temporal retrieval demo.
- `scripts/rlm_temporal_reconstruction.py` - read-time timeline reconstruction demo.
- `scripts/critic_counterfactual_smoke.py` - cheap learned critic for counterfactual write utility.
- `scripts/reflector_backend_matrix.py` - backend/reflector comparison matrix on LoCoMo.
- `scripts/compare_pie_vs_gepa.py` - PIE vs hand vs GEPA consolidator comparison.
- `mempol/results/gepa_consolidator/summary.json` - current tiny GEPA result: 60% baseline vs 80% GEPA on 5 questions.
- `mempol/results/reflector_matrix/` - partial backend matrix traces.
- `output/experiments/temporal_memory_demo.json` - current temporal demo output.
- `output/experiments/rlm_temporal_reconstruction.json` - current reconstruction demo output.

## Sources

- [Temporal Blindness in Multi-Turn LLM Agents](https://arxiv.org/abs/2510.23853) - TicToc; no model above 65% normalized alignment with timestamps.
- [Real-Time Deadlines Reveal Temporal Awareness Failures](https://arxiv.org/abs/2601.13206) - GPT-5.1 deal closure 4% without remaining-time updates vs 32% with updates; turn-based deadlines near-perfect.
- [Robotouille](https://arxiv.org/abs/2502.05227) - ReAct GPT-4o 47% synchronous vs 11% asynchronous.
- [Zep: A Temporal Knowledge Graph Architecture for Agent Memory](https://arxiv.org/abs/2501.13956) - temporal KG direction.
- [GEPA](https://arxiv.org/abs/2507.19457) - reflective prompt evolution, up to 35x fewer rollouts than GRPO in the paper's experiments.
