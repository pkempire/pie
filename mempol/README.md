# mempol

Research scaffold for **Memory as a Learned Policy**: training small policies that decide
what to write to / read from a long-horizon memory store.

## What's here

```
mempol/
  config.py                 paths, model names, env vars
  llm.py                    OpenAI wrapper + on-disk embedding cache
  data/
    locomo.py               LoCoMo loader → typed Conversation/QA
  backends/
    base.py                 Backend ABC (ingest / retrieve / expand / filter_by_time)
    flat.py                 in-memory dense + BM25 + RRF + adjacent-turn expand
    # tree.py (TODO)        SQLite FTS5 + dense sidecar (PIE22-style)
    # graph.py (TODO)       PIE-KG adapter
  policies/
    base.py                 ReadPolicy ABC + Trace/Step (every op logged)
    v0_naive.py             single hybrid retrieve → answer
    v1_heuristic.py         reformulate → retrieve → expand-if-multihop → rerank → answer
  eval/
    judge.py                LLM-as-judge (Bradley-Terry style scoring)
    metrics.py              accuracy by category, evidence recall, cost
    runner.py               ingest-then-answer pipeline, writes traces JSONL
results/                    per-run summary.json
traces/                     per-run JSONL: one row per question, full op trace
.cache/                     embeddings cache (sha1 → vector)
```

## Quickstart

```bash
# from repo root
export OPENAI_API_KEY=...
pip install openai numpy

# smoke test: 1 conv × 5 questions, ~$0.01, ~20s
python -m mempol.eval.runner --backend flat --policy v0_naive --max-qs 5 --run-name smoke_v0
python -m mempol.eval.runner --backend flat --policy v1_heuristic --max-qs 5 --run-name smoke_v1

# 1 full conv (199 questions), ~$0.20, ~10 min
python -m mempol.eval.runner --backend flat --policy v1_heuristic --max-qs 0 --run-name v1_conv1

# all 10 convs (1986 questions), ~$2-4, ~60 min
python -m mempol.eval.runner --backend flat --policy v1_heuristic --n-convs 10 --max-qs 0 --run-name v1_full
```

## What a trace row looks like (SFT-ready)

```json
{
  "qid": "conv-26::q0",
  "question": "When did Caroline go to the LGBTQ support group?",
  "gold": "7 May 2023",
  "answer": "Caroline went to the LGBTQ support group on 7 May, 2023.",
  "score": 1.0,
  "category": 2,
  "policy": "v1_heuristic",
  "backend": "flat",
  "steps": [
    {"op": "reformulate", "args": {}, "obs_summary": "Caroline LGBTQ support group date"},
    {"op": "retrieve",    "args": {"k": 12, "source": "hybrid"}, "obs_summary": "12 hits"},
    {"op": "rerank",      "args": {"strategy": "dense", "k": 6}, "obs_summary": "kept 6"},
    {"op": "stop_and_answer", "args": {}, "obs_summary": "..."}
  ],
  "retrieved_uids": ["conv-26::D1:3", ...],
  "evidence_recall": 1.0
}
```

Each `(question + accumulated observations up to step t, step_t.op)` pair is one SFT
training example for the read policy. See `PAPER-SPEC.md` §6 for the SFT data format.

## Verified working (smoke run, 2026-04-27)

```
conv conv-26: 419 turns, 5 qs
  v0_naive (flat):   acc=0.50, avg_steps=2.0, avg_retrievals=1.0, wall=18.6s
  v1_heuristic (flat): acc=0.50, avg_steps=4.2, avg_retrievals=2.0, wall=11.5s, evidence_recall=0.6
```

(Sample size too small for ranking — purpose was pipeline correctness. Real numbers
land at the 1-conv / 199-Q sweep.)

## Roadmap

See `PAPER-SPEC.md` for full plan. Next milestones:

1. Run v0/v1 on full LoCoMo (1986 Qs) → first publishable baseline numbers.
2. Add Tree backend (SQLite FTS5) → ablation across two memory representations.
3. SFT a Qwen2.5-1.5B policy on v1 traces → match teacher within 95% on held-out.
4. DPO over preference pairs from perturbed v1 rollouts.
5. GRPO on retrieval reward (correctness − cost) with vLLM rollouts.
6. Joint write+read co-training (the actual paper).
