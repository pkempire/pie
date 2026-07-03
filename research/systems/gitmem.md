---
title: "GitMem (internal)"
year: 2026
category: "memory-system"
tags: ["internal", "git", "commits", "branches", "merges", "temporal"]
---

# GitMem

Internal commit-graph memory substrate. Built in May 2026, smoke-passes, not yet wired into any eval.

## Source

`/Users/parthkocheta/personal-intelligence-system/mempol/backends/gitmem.py` (~470 LOC)

## Architecture

- **Commit**: atomic bundle of ops from one turn. SHA, parent_shas, timestamp, dia_id, ops list, message.
- **Branch**: named pointer to a commit, optionally entity-scoped (e.g. `caroline_city_correction`).
- **Merge**: explicit 3-way reconciliation; commit with 2+ parents and reconciliation ops.
- **Primitives**: `commit`, `merge`, `checkout(sha)`, `state_at(timestamp, entity_uid)`, `diff`, `log`.

## Why this exists

The substrate was built when I was proposing a "trained write policy that emits git-like ops" thesis. That thesis is now superseded by [[sleep-consolidation]]. GitMem remains a useful substrate for cases where contradictions need branching and temporal queries need to be free.

## What it competes with

- Mesa filesystem (Apr 2026) is the closest published substrate — git-backed POSIX FS for enterprise agent workflows. Different problem (agent-produced files), same shape conclusion (git is right for versioned agent state).
- Zep/Graphiti's bi-temporal edges are the closest existing memory primitive — but they don't have branches or merges.

## Where it'd be used

Best fit: research-project memory where branches naturally encode competing hypotheses. Temporal queries on entity state become O(log n) via the `_commits_by_time` index.

Not the best fit: conversational memory like LoCoMo, where branching adds complexity without proportional benefit.

## See also

- [[substrate-design-space]] — where commit-graph fits
- [[time-aware-memory]] — temporal queries it enables for free
- [[belief-revision]] — branches as candidate-belief sets
