---
title: "Substrate design space"
year: 2026
category: "design-space-axis"
tags: ["substrate", "storage", "KG", "vector", "git", "filesystem", "log"]
---

# Substrate design space

The data structure your memory lives in. Most architectural debates about LLM memory are actually substrate debates dressed up. Pick the substrate, much else follows.

## The six families

**1. Flat vector store (KV with embeddings).** Each entry is a chunk of text with an embedding. Retrieval = nearest neighbor.
- Examples: [[mem0]], naive RAG.
- Pro: simple, fast, well-tooled.
- Con: no structure, no relationships, no temporal axis.

**2. Hierarchical summary tree.** Levels of summaries with explicit parent-child links. Lower levels are detail; higher levels are abstraction.
- Examples: [[2601.02845-timem|TiMem]] (5-level temporal hierarchy), [[2510.16392-rgmem|RGMem]] (RG-flow inspired).
- Pro: matches how researchers actually organize.
- Con: hierarchy rigid; reorganization expensive.

**3. Typed knowledge graph.** Entities (typed nodes) + relationships (typed edges) + state transitions per entity.
- Examples: [[pie|PIE]], Zep/Graphiti (with bi-temporal edges).
- Pro: structured queries possible.
- Con: extraction is lossy and brittle; schema fights are constant.

**4. Observation log + consolidated layer.** Raw turns appended cheaply; consolidator builds structured layer asynchronously.
- Examples: [[2605.20616-auto-dreamer|Auto-Dreamer]] (RL-trained consolidator), Mastra OM (Observer + Reflector), SCM.
- Pro: hindsight in consolidation, defers expensive compute.
- Con: requires the consolidator to actually be good.

**5. Commit-graph / git-shaped.** Atomic bundles of changes (commits) with parent links, branches for divergent state, merges for reconciliation.
- Examples: [[gitmem]] (internal), Mesa filesystem (Apr 2026, agent-workflows).
- Pro: temporal queries trivial, contradictions are real branches, full audit trail.
- Con: substrate complexity overkill for most use cases.

**6. Filesystem / files.** Memory IS a directory tree of markdown/JSON files. Agents do `ls`, `cat`, `grep`, `write_file`.
- Examples: Letta filesystem, Mesa filesystem.
- Pro: human-readable, git-versionable, no schema battles.
- Con: navigation is the agent's burden.

## What the field is actually choosing

The high LoCoMo scores cluster in two families: **observation log + consolidated layer** (Auto-Dreamer, SCM, EverMemOS in spirit) and **typed knowledge graphs** (Zep, MIRIX, Amory). Vector stores (Mem0) trail. Filesystem and commit-graph are newer and underexplored.

A reasonable read: substrate matters less than people think above some quality floor. [[2512.12818-hindsight|Hindsight's]] Backboard baseline (90.0%) beats their full system (89.61%) on LoCoMo — that's a vector-store-plus-good-base-model beating a temporal-KG-plus-reflection-loop. Implies for many use cases, the substrate is not the bottleneck.

## When each substrate genuinely wins

- **Vector store**: high-QPS reads, no temporal axis needed, single-user.
- **Hierarchical summary**: human-readable navigation needed, multi-scale abstraction.
- **Typed KG**: explicit relationship queries needed, structured downstream tasks.
- **Observation log + consolidator**: long-horizon, want hindsight in writes, can defer compute.
- **Commit-graph**: branching/merging is first-class, audit trail matters, multi-collaborator.
- **Filesystem**: human-readable matters more than agent ergonomics, version control needed.

## What we'd build for our work

For the [[sleep-consolidation]] direction: substrate = observation log L1 + flat consolidated L3. Simple. The consolidator's prompt (optimized by [[gepa-vs-grpo|GEPA]]) does the work; the substrate doesn't need to be sophisticated.

For long-horizon project tracking specifically: filesystem + git is hard to beat. The substrate complements existing developer tooling.

## See also

- [[sleep-consolidation]] — observation log + consolidated layer family
- [[multi-agent-delegation]] — substrate for shared orchestrator state
- [[2605.20616-auto-dreamer|Auto-Dreamer]] — example of disciplined substrate-typing (semantic vs procedural entries)
