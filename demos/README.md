# Demos — one folder, one claim, one runnable script

Each demo is a bite-size, reproducible artifact: a **single claim** about how agent memory
fails or works, a script that anyone can run for cents, and a committed `results.json` from a
real run. Deterministic scoring wherever possible (no LLM judges — they flip verdicts between
identical runs).

The discipline: **nothing lives here unless it ran end-to-end and the committed results match
what the script produces.** Half-finished experiments stay in `scripts/` until they graduate.

| # | Demo | Claim | Status |
|---|---|---|---|
| 01 | [stale-memory](01-stale-memory/) | Similarity search returns yesterday's truth; timeline replay answers "as of when" (40% → 100% on as-of-past questions) | ✅ verified 2026-07-03 |
| 02 | planning-fallacy *(next)* | LLMs estimate task durations in human-team days with no grounding in actuals; a log of past actuals fixes calibration | 🔜 planned |
| 03 | studying-vs-retrieval *(next)* | Retrieving documents ≠ expertise: a studied cheatsheet beats raw retrieval at matched inference budget | 🔜 planned |
| 04 | judge-flakiness *(next)* | The standard LLM-judge protocol flips verdicts on identical answers; deterministic scoring doesn't | 🔜 planned |

Each planned demo maps to one of the core problems in [APPROACH.md](../APPROACH.md).
