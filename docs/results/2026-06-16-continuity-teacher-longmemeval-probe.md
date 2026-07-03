# Continuity Teacher LongMemEval Probe

Date: 2026-06-16

Status: first promising benchmark signal, not a final claim.

## Command

```bash
python3 -m mempol.scripts.longmemeval_matrix \
  --variant longmemeval_s \
  --out-dir mempol/results/lme_continuity_teacher_probe_final \
  --per-category 1 \
  --cells hybrid_search,continuity_teacher \
  --answer-model gpt-5-mini \
  --judge-model gpt-4o \
  --reformulate-model gpt-5-mini \
  --embed-model text-embedding-3-large \
  --side-by-side-max-questions 20
```

The final row was resumed with stricter continuity budgets after the teacher cell took too long on the shift-table row:

```bash
python3 -m mempol.scripts.longmemeval_matrix \
  --variant longmemeval_s \
  --out-dir mempol/results/lme_continuity_teacher_probe_final \
  --per-category 1 \
  --cells hybrid_search,continuity_teacher \
  --answer-model gpt-5-mini \
  --judge-model gpt-4o \
  --reformulate-model gpt-5-mini \
  --embed-model text-embedding-3-large \
  --continuity-turn-k 8 \
  --continuity-final-turn-k 6 \
  --continuity-session-k 1 \
  --continuity-max-session-chars 2500 \
  --side-by-side-max-questions 20
```

## Result

| Strategy | Rows | Accuracy | Avg retrieved tokens | Avg retrieval count | Avg steps |
|---|---:|---:|---:|---:|---:|
| Hybrid Search | 6 | 66.7% | 2,858 | 10.0 | 2.0 |
| Continuity Teacher | 6 | 100.0% | 4,835 | 11.17 | 7.67 |

By category:

| Category | Hybrid Search | Continuity Teacher |
|---|---:|---:|
| single-session-user | 100% | 100% |
| multi-session | 0% | 100% |
| single-session-preference | 100% | 100% |
| temporal-reasoning | 100% | 100% |
| knowledge-update | 0% | 100% |
| single-session-assistant | 100% | 100% |

Output files:

- `mempol/results/lme_continuity_teacher_probe_final/summary.json`
- `mempol/results/lme_continuity_teacher_probe_final/rows.jsonl`
- `mempol/results/lme_continuity_teacher_probe_final/side_by_side.md`

## What Changed

Added `continuity_teacher` to `mempol/scripts/longmemeval_matrix.py`.

It performs:

1. Route the question.
2. Search original question plus generated search queries.
3. Retrieve turn spans.
4. Retrieve session spans.
5. Expand adjacent turns.
6. Write temporary state objects from retrieved evidence.
7. Reconstruct a timeline when temporal/latest-state reasoning is needed.
8. Choose an action.
9. Answer from evidence/state/timeline.

This is a teacher trace generator, not a trained policy yet.

## Why This Is Interesting

Hybrid search failed exactly where continuity should matter:

- multi-session counting of pending obligations;
- knowledge update where a newer personal best superseded an older one.

Continuity Teacher fixed both by forcing an intermediate state-reconstruction step before answering.

The tradeoff is cost:

- more steps;
- more retrieved context;
- more model calls;
- slower wall time.

This points to the actual research direction: distill or optimize the teacher into a cheaper learned read/write/action policy.

## Not A Final Claim

This is only six balanced rows. It is not SOTA. It is enough to justify a scaled run.

Next scale target:

```bash
python3 -m mempol.scripts.longmemeval_matrix \
  --variant longmemeval_s \
  --out-dir mempol/results/lme_continuity_teacher_balanced_5 \
  --per-category 5 \
  --cells hybrid_search,turn_rag,timeline_synthesis,continuity_teacher \
  --answer-model gpt-5-mini \
  --judge-model gpt-4o \
  --reformulate-model gpt-5-mini \
  --embed-model text-embedding-3-large \
  --side-by-side-max-questions 80
```

If this holds on 30 rows, then run the shardable 60-120 row version and report budget curves.

