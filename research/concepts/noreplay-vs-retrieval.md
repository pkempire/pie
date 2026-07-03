---
title: "NoReplay vs retrieval-flavored memory"
year: 2026
category: "evaluation-axis"
tags: ["evaluation", "NoReplay", "retrieval", "RAG", "long-context"]
---

# NoReplay vs retrieval-flavored memory

A discipline for evaluating memory systems honestly. Proposed by Peter Yang (May 2026) to distinguish *memory* from *long-context reading* and *retrieval*.

## The rule

Once history is ingested, the system cannot replay the transcript. The benchmark's history is presented exactly once, in chronological order. The system gets a fixed-size scratchpad (e.g. 10,000 tokens). As history arrives the scratchpad can be freely updated. **At question time, the scratchpad is frozen.** Questions are answered using only:
1. The frozen scratchpad
2. The question
3. The model's inherent knowledge

No transcript access. No vector DBs over the transcript. No query-time retrieval. No hidden state. No second attempts. No judge-feedback loops.

## Why this matters

Current "memory" benchmarks lump together capabilities that are actually different:
- Long-context reading (stuff everything into context)
- Retrieval-augmented generation (vector search at query time)
- Session retrieval
- Scratchpad summarization (the actual memory capability)
- Hidden memory storage
- Agent-driven searches
- Strong base models compensating for poor memory

Mixing these gives leaderboard scores that don't measure memory. A system that searches the transcript when it sees a question is testing *retrieval*, not memory. A system that answers from a condensed pre-created state is testing *consolidation*. Both are valuable; they're different.

## How current systems map

| System | One-pass ingest | Bounded scratchpad | Transcript access at QA | NoReplay compliant |
|---|---|---|---|---|
| full_context (LoCoMo oracle) | n/a | no (full transcript) | yes (IS the transcript) | no |
| naive_rag | one pass | no (full chunk index) | yes (vector search at QA) | no |
| [[pie]] | one pass | unbounded (374 entities for conv-26) | no transcript, unbounded KG | partial |
| [[mastra-om]] | one pass | unbounded observation log | no transcript, queries log at QA | partial |
| [[mem0]] | one pass | unbounded vector store | no transcript, vector search at QA | partial |
| mempol w/ K_max=12 | one pass | bounded (~500-800 tokens) | no transcript, fixed reader | **yes, strict** |

Most published systems are "one-pass write but unbounded state." None of them are NoReplay-strict in Yang's sense. Mastra's 94.87% LongMemEval is impressive but not honestly comparable to a NoReplay system.

## What this means for evaluation

A proper memory benchmark should report a **budget curve**, not a single number:
- Score@1K tokens
- Score@4K
- Score@10K
- Score@32K
- Score@∞ (unbounded — equivalent to full_context)

The interesting comparison isn't "can you answer," it's "how much state do you need to answer well." A system that hits 75% at 4K tokens is in a different league from a system that hits 80% at 50K.

## Where this connects to our work

- [[sleep-consolidation]] makes NoReplay easier — the consolidator's job is to produce a bounded state. NoReplay forces honest accounting.
- The "budgeted decision problem" framing in mempol is implicitly NoReplay.
- [[2605.20616-auto-dreamer|Auto-Dreamer]] uses memory size as a first-class axis in their results table (active memory bank in tokens). Closest existing paper to NoReplay-discipline reporting.
- [[2601.02845-timem|TiMem]] reports recalled memory length as a metric (52.20% reduction on LoCoMo). Spiritually NoReplay-aligned.

## The honest test

If you can't answer a question without the transcript, you don't have memory. You have a search engine. Both are useful. Don't conflate them on leaderboards.

## See also

- [[write-time-vs-read-time]] — orthogonal axis
- [[2410.10813-longmemeval|LongMemEval]] — the benchmark NoReplay-discipline most affects
- [[memory-budget-curves]] — the reporting standard NoReplay implies
