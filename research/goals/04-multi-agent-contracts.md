---
title: "Goal 04 — Contract-based multi-agent orchestration"
status: "parked"
priority: 4
started: null
owner: "us"
budget: "n/a (parked)"
tags: ["multi-agent", "contracts", "orchestration", "parked"]
---

# Goal 04 — Contract-based multi-agent orchestration

## What this is

A protocol layer for LLM-agent delegation that treats every spawned task as a **contract** with explicit deadline, budget, checkpoint cadence, and acceptable-partial-result. The orchestrator polls contracts, can interrupt mid-execution, reallocates budget across subagents.

See [[multi-agent-delegation]] for the full concept page.

## Status: parked

Not the priority right now. Goal 01 (GEPA consolidator on LoCoMo) and Goal 02 (Auto-Dreamer reproduction) are ahead because they have direct paper-shaped output. This goal would require building substantial infrastructure (contract protocol, orchestrator loop, subagent SDK) before producing a single benchmark number.

## When we'd unpark

- If Goal 01 succeeds and we want a second-paper direction
- If we get external interest in the contract protocol as standalone infrastructure
- If [[2601.18137-deepplanning|DeepPlanning]]-shaped benchmarks become a clear publication target

## Related

- Concept: [[multi-agent-delegation]] — full design and parallelization wins
- Paper: [[2601.18137-deepplanning|DeepPlanning]] — the eval target
- Paper: [[2507.07957-mirix|MIRIX]] — closest existing multi-agent memory system
