# Demo 01 — Stale memory: similarity search returns yesterday's truth

**Claim.** A flat vector store retrieves by *similarity* and keeps no time order. Once a fact
changes, it cannot answer "what was true as of month T" — it confidently returns the newest
value. A reader that first **reconstructs a timeline** of state changes from the raw log, then
resolves the value at-or-before T, fixes this.

**Result** (10 questions over a 9-event life-log, gpt-5-mini, deterministic regex scoring, no LLM judge):

| Reader | Overall | "As of <past month>" questions only |
|---|---|---|
| Flat top-k retrieval | 60% | **20%** |
| Timeline replay | **100%** | **100%** |

A smarter model doesn't fix this — it makes it *worse* (gpt-4o-mini flat scored 40% on as-of-past;
gpt-5-mini scores 20%, because it reasons more confidently over timestamp-free memories). The
failure is structural: the store has no concept of time, so no amount of model capability at read
time can recover it.

Every flat failure is the same mechanism: asked *"where did the user live in May?"*, it answers
*"NYC"* — the user's current city, not their city in May. The information was never lost; the
retrieval scheme just has no concept of *when*.

```text
Q (May): Where did the user live?   [gold: Boston (moved to NYC in Aug)]
  FLAT   [X ] The user lived in NYC.
  REPLAY [OK] The user lived in Boston.
```

## Run it

```bash
# from repo root; needs OPENAI_API_KEY in env or .env
python demos/01-stale-memory/run.py
```

~60 seconds, ~$0.01 (gpt-5-mini + text-embedding-3-small; override with `DEMO_MODEL=...`).
Writes `results.json` (the committed copy is a real run from 2026-07-03).

## How it works

- **FLAT**: embed each log event, retrieve top-4 by cosine similarity, answer from those.
  This is the default memory pattern in most production systems.
- **REPLAY**: two steps. *Map*: an LLM pass over log windows extracts structured transitions
  `(attribute, value, month)` — e.g. `location = Boston @ Jan`, `location = NYC @ Aug`.
  *Reduce*: to answer "as of T", filter transitions to `month <= T` and take the latest value
  per attribute. Validity is computed on demand, never stored.

## Why no LLM judge

An earlier version scored answers with a gpt-4o-mini judge. On identical answers, its verdicts
flipped between runs (it marked "Yes, as of July, the user was dating Alex" wrong against gold
"yes, with Alex"). This is the same failure mode that corrupted LoCoMo's published numbers
([judge accepts up to 63% of intentionally wrong answers](https://github.com/dial481/locomo-audit)).
This demo scores with word-boundary regexes on the key fact: same answers → same score, always.

## Honest caveats

- Synthetic, n=10, one seed. This is a **mechanism demonstration**, not a benchmark. It shows
  *why* flat retrieval fails on time, in a form small enough to read in one sitting.
- The same idea evaluated at scale: our timeline-synthesis reader scores 71.7% on a balanced
  LongMemEval-S matrix (n=240) vs 68.3% turn-RAG and 62.5% hybrid search — see
  `mempol/policies/rlm_temporal.py` and `mempol/scripts/longmemeval_matrix.py`.
- The real-world severity of this failure class is documented by [STALE](https://arxiv.org/html/2605.06527)
  (May 2026): production memory frameworks score 6–8% on implicit-staleness scenarios.
