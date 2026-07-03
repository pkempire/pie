# Personal Intelligence System

**Research code for agents that turn experience into expertise** — not just retrieval-memory,
but the loop behind it: an append-only experience log, a sleep-time *studying* pass that
compresses it into a compact expertise artifact (with time attached), and optimization of that
pass against downstream task performance.

Read [APPROACH.md](APPROACH.md) for the problem statement — temporal blindness, the uncorrected
planning fallacy, retrieval ≠ expertise, and the frozen consolidation loop — and how the pieces
here attack it.

## Start here: the demos

Each demo is one claim, one runnable script (~$0.01), deterministic scoring, committed results.

```bash
pip install -r requirements.txt
echo "OPENAI_API_KEY=sk-..." > .env
python demos/01-stale-memory/run.py
```

| Demo | Claim | Result |
|---|---|---|
| [01-stale-memory](demos/01-stale-memory/) | Similarity search returns yesterday's truth; timeline replay answers "as of when" | flat 40% → replay **100%** on as-of-past questions |

More in [demos/README.md](demos/README.md).

## Verified results at larger scale

All from the eval matrix (`mempol/scripts/longmemeval_matrix.py`, `locomo_matrix.py`) with a
3-bucket judge, honest baselines, and raw outputs on disk:

| Finding | Numbers | Where |
|---|---|---|
| Timeline-synthesis reading beats similarity retrieval on LongMemEval-S (balanced, n=240) | **71.7%** vs turn-RAG 68.3% vs hybrid 62.5% | `mempol/policies/rlm_temporal.py` |
| Tool-using "continuity teacher" ties it at 7× the steps — useful as a trace generator, not a system | 73.3% @ 7.3 steps vs 71.7% @ 4.6 | `mempol/policies/continuity.py` |
| Typed knowledge-graph extraction *loses* to flat RAG on LoCoMo (n=1,491) | KG 46.2% vs flat 58.6% | honest negative; `mempol/scripts/locomo_matrix.py` |
| Repo → memory-substrate ingestion works end to end | 541 artifacts, 3,124 spans, 16 days | `mempol/ledger/` |

Unverified-but-promising (clearly labeled, not results yet): GEPA-evolved consolidator
(+20pp on a 5-question smoke; held-out run pending), amortized write-utility critic
(r≈0.71 on a toy sample).

## The pieces

```text
demos/                  bite-size verified claims (start here)
mempol/
  core/                 universal Artifact/Span/MemoryState/TraceEvent substrate (SQLite)
  ledger/               repo + git history -> experience log with day reports & context packs
  temporal/             valid-time state store + as-of-T context compiler
  policies/             read-time policies incl. timeline reconstruction (rlm_temporal.py)
  strategies/           plugin registry of memory strategies for the eval matrix
  backends/             flat / KG / Mastra-style / provider memory backends
  dspy_consolidator/    DSPy consolidator + GEPA optimization (the studying pass)
  recipes/memory_rl/    RL environment + tooling for training memory policies (tinker)
  scripts/              eval matrices, dashboards, ingestion CLIs
benchmarks/             LoCoMo / LongMemEval loaders and runners
memory_providers/       Mem0 / Zep / Honcho / Supermemory shims (for head-to-head evals)
research/               structured lit-review wiki: 30+ papers with verification tiers
pie/                    the original temporal world model over personal exports (baseline + MCP)
paper/                  paper drafts, lit reviews, postmortems
docs/                   design docs, audits, the frontier review & current bet
demos/ scripts/         graduated vs. in-progress experiments (see demos/README.md)
```

## Common commands

Ingest this repo into the experience ledger, then compile an agent context pack:

```bash
python3 -m mempol.ledger.ingest_repo --root . --run-name dev --max-files 500 --max-commits 80
python3 -m mempol.ledger.day_report --run-name dev --day 2026-05-02 --limit 100
python3 -m mempol.ledger.compile_context --run-name dev \
  --task "What benchmark results should I trust?" --k 10 --token-budget 4500
```

Run the LongMemEval strategy matrix (reportable numbers, not demos):

```bash
python3 -m mempol.scripts.longmemeval_matrix \
  --variant longmemeval_s --out-dir mempol/results/lme_core --per-category 5 \
  --cells legacy_naive_rag_turn,flat_v0,flat_v1,flat_rlm_temporal \
  --answer-model gpt-5-mini --judge-model gpt-4o --embed-model text-embedding-3-large
```

Tests:

```bash
python3 -m pytest tests/
```

## House rules

- **Nothing is a "result" until it ran end to end with honest baselines.** Smoke tests and
  tiny-n runs are labeled as such, everywhere, always.
- No LLM judges where deterministic scoring works — judges flip verdicts between identical
  runs ([we've seen it](demos/01-stale-memory/README.md#why-no-llm-judge)).
- Commit source, tests, harnesses, and compact result summaries. Never commit `.env`, personal
  exports, run DBs, embedding caches, or full benchmark datasets.
- Every serious run records: command, git SHA, dataset, models, budget knobs, metrics, output path.
- If a run produces an insight, promote it to a tracked doc; if an experiment stalls, it stays
  out of `demos/` until it graduates.

## Environment

```bash
OPENAI_API_KEY=...   # most scripts
TINKER_API_KEY=...   # RL training recipes only
```

MIT licensed. Older exploratory drafts live in `legacy/`; the honest status audit of everything
in this repo is in `docs/ORIENTATION-AND-NEXT-BET-2026-06-21.md` and the July 2026 field review
in `docs/FRONTIER-REVIEW-2026-07-01.md`.
