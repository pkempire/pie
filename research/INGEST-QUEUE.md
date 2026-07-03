# Ingest queue — papers/links to add to the wiki

Source: Parth's 2026-06-02 drop. Deduped against the 27 papers already in
`research/papers/`. Status: **HAVE** (already ingested), **NEW-DONE** (ingested
this pass from lit-review data), **NEW-QUEUED** (arXiv API was rate-limited/503;
fetch + ingest when it recovers), **WEB** (non-arXiv; fetch via web, not arXiv MCP).

To ingest a queued arXiv paper:
`python research/scripts/ingest.py <arxiv_id>` (or hand-author from SCHEMA.md), then
`python research/scripts/aggregate.py` to refresh STATUS.md.

## Temporal cluster (the user's #1 theme — reasoning-over-time vs behaving-in-time vs async-planning)

| arXiv | Short name | Status | Thread |
|---|---|---|---|
| 2510.23853 | Temporally Blind / TicToc | HAVE | behaving-in-time (awareness) |
| 2502.05227 | Robotouille | HAVE | planning-under-asynchrony |
| 2601.13206 | Real-Time Deadlines | NEW-DONE | behaving-in-time (awareness) |
| 2505.13508 | Time-R1 (comprehensive temporal reasoning) | NEW-DONE | reasoning-over-time |
| 2406.09170 | Test of Time | NEW-DONE | reasoning-over-time |
| 2508.02045 | TDBench (temporal databases for TSQA) | NEW-DONE | reasoning-over-time |
| 2401.14192 | STG-LLM (spatial-temporal data) | NEW-DONE | reasoning-over-time |
| 2503.13377 | Time-R1 (video temporal grounding) | NEW-QUEUED | reasoning-over-time (vision) |
| 2601.07468 | Beyond Dialogue Time (temporal-semantic memory) | NEW-QUEUED | time-aware-memory |

## RL-for-memory / tool-use / orchestration

| arXiv | Short name | Status |
|---|---|---|
| 2508.19828 | Memory-R1 | HAVE |
| 2503.09516 | Search-R1 | HAVE |
| 2507.21892 | Graph-R1 (agentic GraphRAG, end-to-end RL) | NEW-DONE |
| 2507.05257 | Evaluating Memory via Incremental Multi-Turn (MemoryAgentBench) | NEW-QUEUED |
| 2506.06266 | (unknown — fetch) | NEW-QUEUED |
| 2506.05790 | (unknown — fetch) | NEW-QUEUED |
| 2509.24527 | (unknown — fetch) | NEW-QUEUED |
| 2506.01622 | (unknown — fetch) | NEW-QUEUED |
| 2501.10674 | (unknown — fetch) | NEW-QUEUED |
| 2404.12353 | (unknown — fetch) | NEW-QUEUED |
| 2501.00663 | (unknown — fetch; arxiv html link given) | NEW-QUEUED |
| 2601.03236 | (unknown — fetch) | NEW-QUEUED |
| 2512.03627 | (unknown — fetch) | NEW-QUEUED |
| 2601.01885 | (unknown — fetch) | NEW-QUEUED |
| 2601.09465 | (unknown — fetch) | NEW-QUEUED |
| 2603.03290 | (unknown — fetch; future-dated, may not resolve) | NEW-QUEUED |
| 2604.17555 | (unknown — fetch; future-dated, may not resolve) | NEW-QUEUED |

## Already-have (in the user's list, for completeness)

2410.10813 LongMemEval · 2501.13956 Zep/Graphiti · 2601.02163 EverMemOS · 2601.02845 TiMem

## WEB (non-arXiv — fetch via browser/web_fetch, not arXiv MCP)

- Letta — *Context Repositories*: https://www.letta.com/blog/context-repositories
- Letta — *Continual Learning*: https://www.letta.com/blog/continual-learning
- Anthropic — *Effective Context Engineering for AI Agents*: https://www.anthropic.com/engineering/effective-context-engineering-for-ai-agents
- raw.works — *Recursive Language Models as Memory Systems* (already discussed in lit-review; make a `systems/` page): https://raw.works/recursive-language-models-as-memory-systems/
- GitHub — longmemeval-rlm (RLM reference impl): https://github.com/rawwerks/longmemeval-rlm
- Stanford Hazy Research — *Cartridges* (KV-cache compression; "facts in the weights"): https://hazyresearch.stanford.edu/blog/2025-06-08-cartridges
- Zep — Graphiti repo: https://github.com/getzep/graphiti
- OMEGA benchmark whitepaper: https://omegamax.co/benchmarks#whitepaper
- Honcho: https://honcho.dev/
- OpenReview jWBZdlU5Xl · OpenReview moWiYJuSGF
- UMD scholarly paper (Cavolowsky 2025): https://www.cs.umd.edu/sites/default/files/scholarly_papers/202501_Cavolowsky%2C_Mark_Scholarly_Paper.pdf

## Leaderboard scorecard the user pasted (verify + fold into paper_leaderboard)

| System | LongMemEval | LoCoMo | Note |
|---|---|---|---|
| OMEGA | 95.4% | — | #1, closed source |
| Mastra | 94.87% | — | Observational, text-only |
| Hindsight | 91.4% | — | Episodic + reflection |
| Zep/Graphiti | ~72% | 58.44%* | *corrected from claimed 84% |
| Mem0 | ~69% | +26% vs baseline | 40% extraction-failure documented |
| Naive RAG | ~60% | — | baseline most systems beat |
| Full context | varies | — | until context window exceeded |
