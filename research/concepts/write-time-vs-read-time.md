---
title: "Write-time vs read-time compression"
year: 2026
category: "design-space-axis"
tags: ["compression", "RAG", "consolidation", "test-time-compute"]
---

# Write-time vs read-time compression

Memory systems are bifurcating into two camps based on **where the compute is spent**.

## Write-time compression (the traditional path)

Compute happens at ingestion. As turns arrive, an LLM (or a hand-coded extractor) decides what to commit to the structured store. The store is queried cheaply at read time.

Examples: [[mem0]], [[mastra-om]], [[zep]], [[pie]], [[2605.20616-auto-dreamer|Auto-Dreamer]], [[2601.02845-timem|TiMem]].

Pros:
- Reads are fast (the hard work is done)
- Storage is bounded by the extraction policy
- Suitable for high-QPS deployment (many reads per write)

Cons:
- Extraction is lossy (Mem0 reports ~40% extraction-failure rate)
- Decision quality at write time is limited by what's been seen so far (no hindsight)
- Wrong extractions are sticky — unless explicitly revised, they persist

## Read-time decompression (the new path)

No structured commit at write time. Store raw. At query time, the model uses tool calls to navigate, summarize, and reason over the raw store. The compute happens per query.

Examples: [[rlm|Recursive Language Models]], [[search-r1]], naive RAG with aggressive expansion.

Pros:
- No information loss (always reading source)
- Per-query budget can be tuned
- Works well for sparse query loads (a few hard questions, lots of context)

Cons:
- High per-query cost
- Latency scales with question hardness
- Each query rediscovers the same structure

## The numbers

Across LongMemEval (the standard benchmark):
- [[mastra-om|Mastra OM]] (write-time, gpt-5-mini): 94.87%
- [[rlm|RLM]] (read-time, Gemini 3 Flash): 89.8%
- [[mem0|Mem0]] (write-time, gpt-4o): ~69%
- Naive RAG (read-time): ~60%

Strong write-time systems beat strong read-time systems on this benchmark — but the comparison is muddy because write-time systems pre-pay LLM costs that don't appear in per-query budgets.

## The not-yet-popular third path: hybrid via sleep

[[sleep-consolidation]] is the synthesis. Write cheaply at ingestion (raw log, no LLM), consolidate asynchronously between sessions (offline compute, with hindsight), query the consolidated store cheaply at read time. Amortizes the LLM cost across many turns, gets the hindsight benefit, retains fast reads.

This is where the field is moving. Auto-Dreamer, TiMem, SCM all instantiate this pattern in different ways. The question is *how to train the consolidator* — see [[gepa-vs-grpo]].

## Implication for our work

When proposing a new memory system, locating it on this axis is the first question. mempol was originally framed as write-time-trained (RL on a write policy). The sleep-consolidation pivot moves us to the hybrid path — write cheaply, consolidate offline with [[gepa-vs-grpo|GEPA]].

## See also

- [[noreplay-vs-retrieval]] — the orthogonal evaluation axis
- [[sleep-consolidation]] — the architecture that bridges both
- [[gepa-vs-grpo]] — the training-method choice for the consolidator
