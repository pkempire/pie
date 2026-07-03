---
title: "Multi-agent delegation"
year: 2026
category: "architecture-pattern"
tags: ["multi-agent", "orchestration", "contracts", "checkpoints", "DeepPlanning"]
---

# Multi-agent delegation

When one LLM agent spawns others to do subtasks, almost everything that current frameworks do is fire-and-forget: orchestrator hands off a task, waits, receives a result string. This loses an enormous amount of structure and capability.

## What's missing in current frameworks

- **No live monitoring.** Orchestrator can't see if a subagent is stuck mid-execution. LangGraph, CrewAI, OpenAI Agents SDK, Claude Agent SDK all hand off synchronously.
- **No deadlines.** Subagents work until done. Can't express "I want a partial answer in 30 seconds, full in 5 minutes."
- **No partial-result extraction.** If you abort a subagent mid-reasoning, the 40% of work it did is discarded — there's no protocol for "give me whatever you have."
- **No mid-execution redirection.** If priorities shift, you can't tell a subagent "actually, focus on X instead."
- **No budget reallocation.** Subagent A finishing under-budget can't transfer remaining tokens to subagent B struggling on its budget.

## What the SOTA shows

[[2601.18137-deepplanning|DeepPlanning]] (Jan 2026): the hardest published long-horizon planning benchmark. GPT-5.2-high hits **44.6%** case-level accuracy. Claude-4.5-Opus at **22.7%** on travel planning. The failure mode is cascading constraint violations across multi-step plans — exactly what mid-execution coordination would catch.

MultiAgentBench (ACL 2025): canonical multi-agent eval with milestone-based KPIs across star/chain/tree/graph topologies. Standard target if you publish anything multi-agent.

[[2507.07957-mirix|MIRIX]] (Jul 2025): multi-agent memory architecture, 85.38% on LoCoMo. Closest existing published work to combining multi-agent orchestration with memory.

Google's "Science of Scaling Agent Systems" (2026): independent multi-agent amplifies errors **17.2×**; centralized orchestrator with validation contains amplification to **4.4×**. Empirical evidence the orchestrator-with-validation pattern is critical.

## The contract abstraction

Treat every delegated task as a **contract**: (task description, deadline, budget tokens, checkpoint interval, acceptable partial). Subagents checkpoint on schedule (fraction_done, summary, blockers, revised_eta). Orchestrator polls contracts, can interrupt with partial-result protocol, reallocates budget from completed contracts to stalled ones.

None of LangGraph / CrewAI / OpenAI Agents SDK / Claude Agent SDK have this primitive. Workflow engines (Temporal, Airflow) have checkpoint protocols but for deterministic pipelines, not LLM agents.

The combination — LLM agents with explicit contracts, checkpointing, hierarchical delegation, mid-execution interruption with partial results, and budget reallocation — is unclaimed infrastructure as of May 2026.

## Why this connects to memory

The orchestrator's *state* across a multi-step delegated job is memory. The contract DAG is the workflow's memory layer. Pausing and resuming requires checkpoint state. The same architecture that does sleep-consolidation for conversational memory is the substrate for orchestrator state.

## Parallelization wins contracts unlock

1. **Speculative branching.** Spawn N subagents with different framings; first to converge wins; kill others.
2. **Dynamic budget reallocation.** Move tokens from finished to struggling.
3. **Critical-path identification.** Checkpoints expose which subagent is the bottleneck.
4. **Backpressure on shared resources.** Stalled-on-rate-limit checkpoint prevents thundering herd.
5. **Pre-emption.** 0.05 fraction_done after 60% budget → kill and re-spawn with different framing.
6. **DAG-aware parallelization.** Contracts with no shared parent auto-parallelize.

## See also

- [[2601.18137-deepplanning|DeepPlanning]] — the hardest applied benchmark
- [[2507.07957-mirix|MIRIX]] — multi-agent memory architecture
- [[2502.05227-robotouille|Robotouille]] — async planning failure mode
- [[sleep-consolidation]] — same checkpointing primitive applied to memory
