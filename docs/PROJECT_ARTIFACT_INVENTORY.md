# Project Artifact Inventory

Status: first canonical inventory for the Research Ledger direction.

## Core Research / Memory Work

| Artifact / thread | Path(s) | Current status | Best-case value |
|---|---|---|---|
| GEPA consolidator | `mempol/dspy_consolidator/`, `scripts/run_gepa_consolidator.py`, `scripts/gepa_live.py`, `mempol/results/gepa_consolidator/summary.json` | Tiny smoke: baseline 60%, GEPA 80% on 5 questions. Too small and too expensive to claim as final. | Publishable result if scaled and reproduced: learned/offline consolidation improves memory quality over hand prompt. |
| LongMemEval matrix | `mempol/scripts/longmemeval_matrix.py`, `mempol/results/lme_core_shards_merged/summary.json` | Completed balanced run: Timeline Synthesis 71.7%, Turn RAG 68.3%, Hybrid 62.5%, Rerank 61.7%. | Strong benchmark artifact showing read-time temporal synthesis can outperform naive RAG on a popular benchmark. |
| LoCoMo matrix | `mempol/scripts/locomo_matrix.py`, `mempol/results/locomo_matrix_3conv_core/summary.json` | Completed 3-conversation run: flat 58.6%, flat-v1 53.6%, PIE cached 46.2%. | Honest baseline/failure table; useful for proving what does not work. |
| Universal memory core | `mempol/core/`, `mempol/scripts/core_ingest.py`, `mempol/scripts/core_query.py` | SQLite prototype with artifacts, spans, memory states, traces. Retrieval is lexical and simple. | Foundation for universal ingestion if upgraded into Research Ledger. |
| Research Ledger | `mempol/ledger/` | Initial implementation added: repo ingestion, Git history, day reports, context packs. | Product substrate for context manager app and long-running project memory. |
| Universal retrieval eval | `mempol/scripts/eval_universal_memory_retrieval.py`, `mempol/results/universal_retrieval_eval/` | One-conv retrieval result: turn-memory recall 30.9% vs raw 15.3%. | Baseline for retrieval/evidence improvements under token budget. |
| Critic counterfactual smoke | `scripts/critic_counterfactual_smoke.py`, `output/experiments/critic_counterfactual.json` | Toy result: Pearson 0.707, MAE 0.03 on tiny sampled deltas. | Seed for cheap utility critic replacing brute-force counterfactual reward. |
| RLM temporal reconstruction demo | `scripts/rlm_temporal_reconstruction.py`, `output/experiments/rlm_temporal_reconstruction.json` | Synthetic read-time timeline reconstruction; not a true RLM. Flat 66.7%, reconstruction 83.3% with one judging issue. | Strong educational demo for state-at-time reconstruction. |
| LoCoMo temporal eval | `scripts/locomo_temporal_eval.py`, `output/experiments/locomo_temporal_eval.json` | Small contradictory result: flat 83.3%, reconstruction 66.7%. | Needs cleanup or deletion; currently not reliable. |
| Reflector backend matrix | `scripts/reflector_backend_matrix.py`, `mempol/results/reflector_matrix/summary.json` | 30-question toy: kg_raw 70%, gepa_flat 68.3%, flat_raw 56.7%, hand 35%, mastra-inspired 40%. | Useful only if rerun with exact definitions and real Mastra caveat. |
| PIE cached KG | `pie/`, `mempol/backends/pie_kg.py`, `benchmarks/locomo/cache/` | Runs but underperforms; provenance/evidence recall broken in some paths. | Temporal/KG baseline and personal MCP if cleaned. |
| Phase A read policy | `mempol/recipes/memory_rl/` | Code exists; read policy has not been seriously trained. | Needed if returning to learned read/retrieve policies. |
| Phase B write policy | `mempol/recipes/memory_rl/`, `mempol/eval/counterfactual.py` | Prior counterfactual reward smoke had positive signal but thesis is no longer preferred. | Useful as critic-training infrastructure, not main paper headline. |

## Product / App Threads

| Artifact / thread | Path(s) | Current status | Best-case value |
|---|---|---|---|
| Context manager app | `mempol/ledger/`, `mempol/scripts/core_dashboard.py` | New direction; first ledger backend exists, polished app not yet built. | Product people actually use: inject perfect context into any LLM app. |
| PIE MCP | `mcp_server.py`, `pie/`, `pie/ui/` | Live older product wired to Claude Desktop. | Personal temporal memory demo and data source. |
| Architect planner | `architect/` | Working planner/dashboard/component DB; no users. | Demo that the ledger helps software architecture over long-running projects. |
| Footnote | `scripts/footnote/`, `remotion/footnote/` | MVP implemented end-to-end, not tested on real video. | Separate shippable AI video annotation product and public artifact generator. |
| Visual-memory / Big Brother | `visual-memory/` | Large hackathon artifact, frame/video/event DB. Not connected to mempol. | Multimodal temporal memory demo if revived. |
| Sales/Lucid side projects | `sales/`, `research/content/lucid-academy-research-residency.md` | Real business docs/products, not core. | Use as downstream project-memory examples later. |

## Research / Content Threads

| Artifact / thread | Path(s) | Current status | Best-case value |
|---|---|---|---|
| Research wiki | `research/` | Strong corpus of papers, systems, concepts, goals; static site exists. | Public artifact and source corpus for Research Ledger demo. |
| Working Memory episode 1 | `research/content/01-working-memory-intro-TOC-v2.md`, related drafts | TOC exists; body not final. | Public reputation artifact if paired with real ledger/benchmark demo. |
| Temporal awareness content | `research/content/temporal-awareness-*`, `research/concepts/time-aware-memory.md` | Multiple drafts and assets; thesis needs tightening. | Strong video/essay if tied to state transitions and elapsed-time agent failures. |
| Paper draft | `paper/main.tex`, `paper/lit-review/`, `paper/TODO.md` | Current thesis stale; per-op counterfactual framing no longer matches direction. | Rewrite around Research Ledger / learned context compiler after solid result. |
| Literature review | `paper/lit-review/`, `research/papers/`, `research/systems/` | Broad and useful, but scattered across docs. | Grounded related-work section and public bibliography. |

## External References In Repo

| Artifact / thread | Path(s) | Current status | Best-case value |
|---|---|---|---|
| RLM | `external/rlm/`, `external/longmemeval-rlm/` | Local reference implementation and LongMemEval experiments. | Read-time recursive context inspection policy reference. |
| H-Net | `external/hnet/` | Local reference implementation. | Learned/dynamic chunking experiment. |
| TimeM | `external/timem/` | Local reference implementation. | Temporal-memory baseline/reference. |

## Immediate Kill / Keep Guidance

Keep and finish:

- Research Ledger ingestion and context-pack compiler.
- LongMemEval matrix reporting.
- GEPA consolidator only if scaled reproducibly.
- Research wiki as source corpus.
- PIE as temporal baseline.

Demote:

- Per-op counterfactual as headline paper thesis.
- Tiny temporal/RLM toy results as research claims.
- Mastra-inspired results unless exact Mastra is wired or clearly renamed.

Archive after indexing:

- Old top-level architecture drafts.
- Duplicate paper/blog/video drafts.
- Result folders that are superseded by canonical summaries.
