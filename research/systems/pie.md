---
title: "PIE (typed-transition KG)"
year: 2025
category: "memory-system"
tags: ["KG", "typed-transitions", "PIE", "internal", "extraction"]
---

# PIE

A typed-transition knowledge graph memory system, predates this research line. Entities + per-entity state transitions + typed relationships. Hand-coded extraction prompts; three-tier resolver for entity matching.

## Architecture

- **Entities**: typed nodes (`person`, `event`, `concept`, `organization`, `belief`, etc.). Each has a current state dict.
- **State transitions**: time-stamped events that modify an entity's state. Types: `creation`, `update`, `contradiction`, `resolution`, `archival`.
- **Relationships**: typed edges between entities.
- **Importance scoring**: per-entity float, hand-tuned heuristic.

## Numbers we have (conv-1 of LoCoMo, gpt-4.1 model + gpt-5-mini judge)

| Run | Overall | Adversarial | Multi-hop | Open-domain | Single-hop | Temporal |
|---|---|---|---|---|---|---|
| v1 (clean baseline) | **61.3%** | 29.8 | 56.2 | 80.8 | 72.1 | 78.4 |
| v4 (re-extracted, strict speaker) | **63.1%** | 10.6 | 65.6 | 46.2 | 87.9 | 86.5 |

PIE only wins on temporal (78-86%) — its one structural advantage from explicit timestamps + state transitions. Loses to naive RAG (66%) overall.

## Where it sits

- [[write-time-vs-read-time|Write-time compression]] (synchronous extraction per chunk)
- Unbounded scratchpad (374 entities for LoCoMo conv-26)
- Not [[noreplay-vs-retrieval|NoReplay]] (retrieves from KG at QA time, but no transcript replay — partial)
- Single-level (no [[sleep-consolidation|consolidation]])

## Why we used it (and what's wrong with it)

PIE was the substrate before pivoting to sleep-consolidation. The typed transitions give richer state than flat KV; the relationships enable multi-hop. But:
- Hand-coded extraction loses 12 points to full_context on conv-1
- Per-chunk LLM call for extraction is expensive
- Speaker confusion is the dominant failure mode (LoCoMo conversations are symmetric)
- No retention budget — graph grows without bound

## What we'd replace it with

For the [[sleep-consolidation]] direction: drop PIE as substrate, use raw observation log + flat consolidated facts. The consolidator (trained via [[gepa-vs-grpo|GEPA]]) does the work that PIE's extraction prompts did, but with hindsight across multiple chunks.

PIE-the-data-structure remains useful for cases where typed transitions are genuinely needed (temporal queries especially). For LoCoMo-style conversational memory, it's overengineering relative to what TiMem achieves with hierarchical summaries.

## See also

- [[mem0]] — simpler flat-KV alternative
- [[2601.02845-timem|TiMem]] — what beats PIE on the same benchmark
- [[gitmem]] — internal alternative with commit-graph substrate
