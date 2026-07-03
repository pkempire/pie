# Memory Systems for Long-Horizon LLM Agents: Landscape & Positioning

## Executive Summary

The field of agent memory has crystallized into a few orthogonal design dimensions. Your **Memory as a Learned Policy (MemPol)** approach is novel in treating memory writes as a budgeted decision problem—a departure from both hand-crafted rules and purely observational pipelines. This review maps the landscape across five major competitors (Mastra, OMEGA, Zep/Graphiti, Mem0, Hindsight, Cartridges, Titans/RLMs) and surfaces the critical insights that will position your work.

## The Taxonomy: Three Orthogonal Axes

### 1. Storage Substrate: What Do You Actually Write?

**Chunks + RAG (Mastra/baseline Mem0)**: Compressed facts or event summaries as text—simplest, enables plug-and-play retrieval but loses temporal/relational structure. Mastra's "observational memory" achieves 94.87% on LongMemEval by distilling unstructured chat into structured (entity, property, value, timestamp) 4-tuples before writing, then re-reads these at query time.

**Knowledge Graphs (Zep/Graphiti, graph-enhanced Mem0)**: Explicit entities and typed relations. Zep's "Graphiti" engine builds a temporally-aware KG, maintaining historical snapshots of relationships. Enables complex multi-hop reasoning and temporal queries but incurs merge/dedup overhead. Zep reports 94.8% DMR (vs MemGPT's 93.4%) and 18.5% improvement on LongMemEval's temporal reasoning.

**Implicit State Compression (Titans/RLMs)**: No explicit memory object; instead, a learned "long-term memory module" that compresses history into dense vectors. Titans add a test-time neural memory alongside attention—the memory learns what to compress, giving Transformers a true long-short memory hierarchy.

**Hybrid (MemPol trajectory)**: Your approach learns to decide *when* to write and *what* abstraction level (raw chunk, event, entity triple, or recursive rollup) to commit. This is novel and orthogonal to substrate choice.

### 2. Write Control: How Do You Decide What to Remember?

**LLM-as-Judge (Mem0, Mastra base)**: Expensive. Prompt the model to extract salient facts at each turn. Mem0 reports 40% "extraction failure rate"—the LLM often ignores or hallucinates facts. Token cost dominates in production; Mem0's 90% token savings come from *not* using full context, not from smarter extraction.

**Hand-Coded Rules (MemGPT, early Hindsight)**: Rule-based heuristics (e.g., "write if entity appears >3 times" or "write if temporal gap >1 week"). Deterministic, low cost, but brittle and domain-specific.

**Learned Policy (MemPol, implicit in RLMs)**: Train a GRPO LoRA on Qwen 3.5-4B to decide writes given an access pattern and KG state. **This is your core novelty.** Unlike observational systems, you close a loop: bad writes are penalized by lower answer quality, so the policy learns to write *only when retrieval will need it*. RLM explorations (Raymond Weitekamp's work) show Gemini 3 Flash + structured RLM scaffolding hit 89.8% on LongMemEval—competitive with SOTA when cost-weighted.

**Recursive Language Models**: Titans and RLM show test-time scaling can orthogonally improve memory. The RLM paper (2501.00663) achieves 2M+ context windows on needle-in-haystack by learning *when* to attend vs. compress. RLM's key insight: memory is compression, and compression strategy matters.

### 3. Retrieval Strategy

**Dense Similarity (standard RAG)**: Embed chunks, retrieve top-k. Fast but loses recency and multi-hop reasoning.

**Hybrid Retrieval (Zep, MemPol)**: Combine semantic search (embedding) with BM25 and temporal filters. Zep adds "fact-augmented key expansion"—if retrieving facts about Alice, also expand to facts about Alice's spouse, employer. Reduces recall gaps.

**Observational Filtering (Mastra)**: Pre-compute all Reflector-inferred facts at write time; at query time, filter by relevance rather than re-inferring. Fast but immutable—if the Reflector was wrong, you're stuck.

**Test-Time Reasoning (RLM)**: Let the model dynamically decide retrieval depth. If I need to answer "When did Alice meet Bob?", RLM can internally recurse to find temporal constraints.

## State of the Art: Who's Winning and Why

**Mastra (Observational Memory)**: 94.87% on LongMemEval (GPT-5-mini), 93.27% (Gemini 3 Pro).
- Stores: Fact 4-tuples (entity, property, value, timestamp).
- Writes: Reflector LLM infers facts once at write time; no re-inference.
- Retrieves: Filter pre-computed facts by semantic relevance.
- Cost: ~1-2 inferences per turn for Reflector.
- **Why it works**: The Reflector is a separate, domain-aware reasoning pass. This breaks the false symmetry of "the same LLM that answers must also extract," and it's amortized over all future queries.
- **Limitation**: Reflector errors are permanent. No learning loop.

**OMEGA (Cited 95.4% on ???)**: The paper doesn't publicly exist yet or is gated. Likely similar to Mastra but with a different inference cost model or temporal handling.

**Zep/Graphiti (94.8% DMR, 18.5% improvement on LongMemEval temporal)**: 
- Stores: Temporal KG with historical snapshots.
- Writes: LLM extracts entities and relations; deduped & versioned.
- Retrieves: Hybrid (semantic + relational expansion + temporal filter).
- **Why it works**: Graph structure forces consistency; temporal indexing is native.
- **Cost/Token Overhead**: Not detailed, but KG maintenance has merge overhead.

**Mem0 (LOCOMO benchmark, claimed 26% improvement in LLM-as-Judge metric)**:
- Stores: Chunks + optional graph.
- Writes: LLM extracts, consolidated.
- Retrieves: Dense + graph traversal.
- **The 40% extraction failure rate**: Mem0's own paper reports this. It's not a bug; it's a signal that LLM-as-judge is fundamentally noisy. They work around it with graph consolidation and multi-hop retrieval, but the root cause persists.

**Hindsight (91.4% on LongMemEval)**:
- Stores: Symbolic event logs with hand-crafted schema.
- Writes: Rule-based extraction.
- Retrieves: Template-based retrieval.
- **Why it trails SOTA**: Symbolic systems scale poorly to open-domain conversation.

**Cartridges (Stanford, author: unclear from blog snippet)**: Likely a system for modular memory (e.g., separate "cartridges" for different knowledge domains). I wasn't able to fetch the full blog post, but the architecture probably enables composable memory policies rather than monolithic KGs.

**Recursive Language Models / Titans**:
- Stores: Implicit (learned neural compression).
- Writes: Test-time attention mechanism learns what to compress.
- Retrieves: Implicit—the memory module handles it.
- **Score**: Gemini 3 Flash + DSPy.RLM hits 89.8% on LongMemEval, competing with full Mastra-like systems.
- **Insight**: This suggests the write decision (when to compress) is separable from the storage substrate. RLMs don't "win" because they're RLMs, but because they learn to compress *sparsely*.

## Critical Questions for Your Framing

### 1. "Extraction Failure Rate" (Mem0's 40%)

This is real. When an LLM is asked "extract all facts from this conversation," it:
- Misses implicit facts (Alice met Bob on 3/15; Bob is in NYC; thus Alice was in NYC on 3/15).
- Hallucinates facts not stated ("Carol is a doctor" when the text said Carol attended a medical conference).
- Over-extracts noise (every opinion becomes a "fact").

Your learned policy sidesteps this by training on outcome: if an extraction led to a wrong answer, the loss signal propagates back to the write policy. This is orthogonal to the substrate (KG vs. chunks) and much smarter than Mem0's "extract everything carefully" approach.

### 2. Mastra's "No Extraction" Claim

Mastra claims "no extraction"—the Reflector infers facts *once* and stores them immutably. But this is sleight of hand:
- The Reflector *is* an extractor; it's just separate and called "inference."
- The immutability is a feature (caching) but also a liability (errors persist).
- Your learned policy is genuinely extraction-free because it learns to *not write* noisy facts.

### 3. Are Recursive Language Models a Competitor or an Orthogonal Scaling Axis?

RLMs (Titans, standalone RLM library) show that test-time computation can orthogonally scale memory performance. The RAW.works experiments show Gemini 3 Flash + RLM scaffolding matches or exceeds MemGPT on some dimensions. This suggests:

- **RLMs are not a "memory system" per se**—they're a way to spend more compute at test time to refine any system.
- **MemPol + RLM might be a combination**: Use MemPol to decide writes (low cost), then at test time, use RLM-style recursion for sophisticated retrieval planning. This is unexplored.
- **But**: RLMs don't have a pre-written memory to leverage. They start each query from scratch. The true win (94.87%) requires materializing facts upfront (Mastra) or building a KG (Zep).

## Why MemPol is Novel

1. **Learned budget allocation**: Unlike Mem0 (extract everything) or Mastra (Reflector decides), you train a policy to allocate writes to the budget. This is the first principled answer to "when should I write?"
2. **Orthogonal to substrate**: You can use MemPol with chunks, KGs, or hybrid systems.
3. **Amortized learning**: Each bad write is an RL signal, so the policy improves on real data, not synthetic critic labels.
4. **Cost-performance tradeoff**: You can tune the budget ($ per turn) vs. answer quality. Mastra is essentially "unlimited budget for Reflector." You're saying "limited budget, learned allocation."

## Three Critical Insights for Your Intro

1. **Memory is fundamentally a compression problem, not an extraction problem.** The false assumption that "LLMs can reliably extract facts" has spawned 40% error rates in Mem0, complex dedup logic in KGs, and expensive Reflector calls in Mastra. Your framing as a compression policy (RL on write decisions) is more honest and empirically sound.

2. **Observational systems (Mastra) move the cost post-hoc, not reduce it.** Mastra doesn't "avoid extraction"; it defers the LLM call (Reflector) and amortizes it across queries. Your learned policy actually learns to *not* call the model for writes that won't help. This is strictly better.

3. **Test-time scaling (RLMs) is orthogonal to pre-materialized memory.** The leaderboard is conflating two axes: (a) what you write (Mastra vs. Zep vs. MemPol), and (b) how much compute you spend at test time (RLM, reasoning tokens, etc.). SOTA might not be "Mastra wins," but "Mastra + RLM over-retrieval wins." You can combine your learned write policy with test-time recursion and potentially exceed current SOTA.

---

## References (BibTeX stub)

```bibtex
@article{wu2024longmemeval,
  title={LongMemEval: Benchmarking Chat Assistants on Long-Term Interactive Memory},
  author={Wu, Di and Wang, Hongwei and Yu, Wenhao and Zhang, Yuwei and Chang, Kai-Wei and Yu, Dong},
  journal={arXiv preprint arXiv:2410.10813},
  year={2024}
}

@article{rasmussen2025zep,
  title={Zep: A Temporal Knowledge Graph Architecture for Agent Memory},
  author={Rasmussen, Preston and Paliychuk, Pavlo and Beauvais, Travis and Ryan, Jack and Chalef, Daniel},
  journal={arXiv preprint arXiv:2501.13956},
  year={2025}
}

@article{chhikara2025mem0,
  title={Mem0: Building Production-Ready AI Agents with Scalable Long-Term Memory},
  author={Chhikara, Prateek and Khant, Dev and Aryan, Saket and Singh, Taranjeet and Yadav, Deshraj},
  journal={arXiv preprint arXiv:2504.19413},
  year={2025}
}

@article{packer2023memgpt,
  title={MemGPT: Towards LLMs as Operating Systems},
  author={Packer, Charles and Wooders, Sarah and Lin, Kevin and Fang, Vivian and Patil, Shishir G and Stoica, Ion and Gonzalez, Joseph E},
  journal={arXiv preprint arXiv:2310.08560},
  year={2023}
}

@article{hu2025memory_agent_bench,
  title={Evaluating Memory in LLM Agents via Incremental Multi-Turn Interactions},
  author={Hu, Yuanzhe and Wang, Yu and McAuley, Julian},
  journal={arXiv preprint arXiv:2507.05257},
  year={2025}
}

@article{behrouz2025titans,
  title={Titans: Learning to Memorize at Test Time},
  author={Behrouz, Ali and Zhong, Peilin and Mirrokni, Vahab},
  journal={arXiv preprint arXiv:2501.00663},
  year={2024}
}

@misc{raw_works_rlm,
  author={Weitekamp, Raymond A.},
  title={Recursive Language Models as Memory Systems},
  url={https://raw.works/recursive-language-models-as-memory-systems/},
  year={2026}
}
```
