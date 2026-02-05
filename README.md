# PIE: Personal Intelligence Engine

**Temporal knowledge graphs for AI agents with long-term memory.**

PIE transforms unstructured conversation history into a continuously evolving world model — a graph of entities, state transitions, relationships, and procedures — enabling LLMs to reason about what changed, when, and why.

## What's Different

| System | Approach | Limitation |
|--------|----------|------------|
| **Mem0** | Facts extracted per-session | No temporal reasoning |
| **MemGPT** | Tiered storage with LLM-managed archival | No semantic time |
| **Zep** | Session summaries + knowledge graph | Graph is flat, no state evolution |
| **Graphiti** | Episode-based temporal graph | Requires structured episodes |
| **PIE** | **Continuous state tracking with semantic time** | — |

PIE tracks how entities evolve over time, detects contradictions, builds procedural memory from patterns, and compiles graph data into LLM-readable temporal context.

## Quick Start

```bash
# Install dependencies
pip install -r requirements.txt

# Set API key
export OPENAI_API_KEY=your_key

# Run extraction on ChatGPT export
python run.py --input ~/Downloads/conversations.json --batches 50

# View the graph
python -m http.server 8765 --directory output
# Open http://localhost:8765/viz.html
```

## Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                    RAW CONVERSATIONS                        │
│         (ChatGPT export, Slack, email, etc.)               │
└──────────────────────┬──────────────────────────────────────┘
                       │
                       ▼
┌─────────────────────────────────────────────────────────────┐
│                  INGESTION PIPELINE                         │
│  ┌──────────────┐  ┌──────────────┐  ┌──────────────┐      │
│  │ Daily Batch  │─▶│  Extraction  │─▶│  Resolution  │      │
│  │   80K chars  │  │   LLM call   │  │  3-tier      │      │
│  └──────────────┘  └──────────────┘  └──────────────┘      │
└──────────────────────┬──────────────────────────────────────┘
                       │
                       ▼
┌─────────────────────────────────────────────────────────────┐
│                    WORLD MODEL                              │
│  ┌─────────┐  ┌─────────────┐  ┌──────────────┐            │
│  │Entities │  │ Transitions │  │ Relationships│            │
│  │  591+   │  │   1,240+    │  │    656+      │            │
│  └─────────┘  └─────────────┘  └──────────────┘            │
└──────────────────────┬──────────────────────────────────────┘
                       │
                       ▼
┌─────────────────────────────────────────────────────────────┐
│               TEMPORAL CONTEXT COMPILER                     │
│  Graph → LLM-readable semantic temporal narratives         │
│  "MoMA visit (2024-03-15, ~3 weeks ago) led to..."        │
└─────────────────────────────────────────────────────────────┘
```

## Core Concepts

### Entities
Anything that persists and evolves: people, projects, beliefs, decisions, tools, organizations, life periods.

### State Transitions
Every state change is recorded, not overwritten. The transition history *is* the temporal model.

### Entity Resolution
Three-tier matching: string similarity → embedding similarity → LLM verification.

### Semantic Time
LLMs never see raw timestamps. Context is compiled with relative time ("~3 weeks ago"), periods ("during SF gap semester"), and ordering ("before X, after Y").

## Benchmarks

| Benchmark | naive_rag | PIE | Notes |
|-----------|-----------|-----|-------|
| LongMemEval | ~60% | TBD | Full 500q run in progress |
| Test of Time | 56% | 31% | PIE hurts date arithmetic |

**Key finding**: PIE helps ordering/succession reasoning but hurts point-in-time lookup. Task-adaptive preprocessing is needed.

## Project Structure

```
pie/
├── core/           # Data models, LLM client, world model
├── ingestion/      # Parser, pipeline, extraction prompts
├── resolution/     # Entity resolution, web grounding
└── eval/           # Quality metrics, ablations

benchmarks/
├── longmemeval/    # LongMemEval adapter + baselines
├── test-of-time/   # ToT benchmark runner
└── locomo/         # LoCoMo adapter (WIP)

sales/              # Sales Intelligence product (demo)
├── app.py          # Flask web app
├── process_mining.py   # Markov chains, bottlenecks
└── prospect_model.py   # Prospect state tracking
```

## Research

This project explores:
- When semantic temporal compilation helps vs. hurts
- Procedural memory extraction from state patterns
- Proactive uncertainty detection (surface unknowns, don't hallucinate)

Targeting EMNLP/NAACL 2025.

## License

MIT

## Contact

- GitHub: [@pkempire](https://github.com/pkempire)
- Twitter: [@parthkocheta](https://twitter.com/parthkocheta)

## Contributing

PRs welcome! See [CONTRIBUTING.md](CONTRIBUTING.md) for guidelines.
