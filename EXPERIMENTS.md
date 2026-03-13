# PIE Experiment Design: Validating the 4 Key Claims

## Overview

These experiments test PIE's core differentiating claims. Each is designed to run on LoCoMo (10 conversations, 1986 questions) where we build the world model once per conversation and reuse it.

**Run command:**
```bash
OPENAI_API_KEY="sk-..." python run_benchmark.py --benchmark locomo --baseline pie_temporal --debug
```

After running, open the generated `viewer_*.html` for full inspection of extraction, retrieval, and answers.

---

## Experiment 1: State Transitions vs Flat Fact Storage

**Claim:** Typed state transitions outperform flat fact storage for temporal reasoning.

**Method:**
- **Condition A (PIE):** Full state transition model — entities track CREATION → UPDATE → CONTRADICTION → RESOLUTION → ARCHIVAL. Context includes timeline with transition types.
- **Condition B (Flat Facts):** Same extraction but flatten all transitions into a single "latest state" per entity. No timeline, no transition types.
- **Condition C (Mastra-style Observations):** Replace PIE extraction with timestamped text observations (no structured entities/transitions).

**What to measure:**
- Overall accuracy on LoCoMo
- **Temporal question accuracy** (the category where transitions matter most)
- **Adversarial question accuracy** (where contradictions help detect false premises)

**How to implement Condition B:**
Modify `_compile_temporal_context()` to skip the Timeline section — only output "Current: {state}" per entity.

**How to implement Condition C:**
Add an `observation_baseline` that stores per-session summaries as text (LoCoMo already provides `session_summary` and `observation` fields we can use).

**Expected result:** A > B on temporal questions. A > C on contradiction/adversarial questions.

---

## Experiment 2: Contradiction Detection for Error Correction

**Claim:** Contradiction detection enables automatic error correction without fine-tuning.

**Method:**
- Take questions where the gold answer changed during the conversation (temporal questions, state changes)
- **Condition A:** PIE with contradiction tracking enabled — entity timelines show when facts changed
- **Condition B:** PIE with contradictions suppressed — only keep latest state, no contradiction markers

**What to measure:**
- Accuracy on temporal questions involving state changes (e.g., "What was X's job in April?" when they changed jobs in March)
- Count of questions where contradiction context helped vs. hurt

**How to identify relevant questions:**
Filter LoCoMo questions where `question_type == "temporal"` and the evidence spans multiple sessions (indicating a change over time).

**Expected result:** Condition A significantly outperforms B on questions requiring understanding of when/how facts changed.

---

## Experiment 3: Procedure Extraction Generalizes to New Situations

**Claim:** Procedure extraction from transition patterns generalizes to new situations.

**Method:**
This requires a different evaluation setup since LoCoMo doesn't test procedural memory.

- **Dataset:** Collect 20 multi-step tasks from real agent trajectories (e.g., web tasks, coding tasks)
- **Step 1:** Agent performs task 3 times with PIE recording state transitions
- **Step 2:** PIE extracts procedures from the transition patterns
- **Step 3:** Agent encounters a similar (but not identical) task
- **Condition A:** Agent has access to extracted procedures from PIE
- **Condition B:** Agent has access to raw trajectory logs
- **Condition C:** Agent has no memory of previous tasks

**What to measure:**
- Task success rate on novel similar tasks
- Number of steps to completion
- Number of errors/retries

**Alternative lightweight version:** Use PIE's existing procedure extraction on the personal ChatGPT export. Manually evaluate whether extracted procedures are (a) correct, (b) generalizable, (c) useful for predicting next actions.

**Expected result:** A > B > C. Structured procedures generalize better than raw logs.

---

## Experiment 4: Semantic Time vs Raw Timestamps

**Claim:** Temporal context compilation (semantic time) improves LLM reasoning over raw timestamps.

**Method:**
- Take 100 temporal questions from LoCoMo
- **Condition A (Semantic Time):** PIE's compiled context: "11 months ago, in the May 2023 session, changed from X to Y"
- **Condition B (Formatted Dates):** Same entities but dates as "May 8, 2023 — state changed from X to Y"
- **Condition C (Unix Timestamps):** Same entities but dates as "1683504000 — state changed from X to Y"
- **Condition D (No Dates):** Same entities with no temporal information at all

**What to measure:**
- Accuracy on temporal questions (ordering, "when did X happen", "what was X before Y")

**How to implement:** Create 4 variants of `_compile_temporal_context()` that format dates differently.

**Expected result:** A > B > D > C. Semantic time helps most. Unix timestamps actively hurt (worse than no dates).

---

## Experiment 5: BM25 Hybrid vs Embedding-Only Retrieval

**Claim:** BM25 + embedding hybrid retrieval via RRF outperforms embedding-only.

**Method:**
- **Condition A (Hybrid):** Current RRF fusion of BM25 + embedding
- **Condition B (Embedding-only):** Disable BM25, use only embedding cosine similarity
- **Condition C (BM25-only):** Disable embedding, use only BM25 keyword matching

**What to measure:**
- Overall accuracy
- Accuracy by question type (single-hop, multi-hop, temporal)
- Retrieval recall@k (what % of evidence entities appear in top-k)

**Expected result:** A > B > C for overall. C > B for keyword-heavy questions. B > C for semantic questions.

---

## Quick Ablation Checklist

| Variable | Default | Test | Metric |
|---|---|---|---|
| Batch size for extraction | 1 session | 1 vs 3 vs 5 | Entities extracted, JSON parse failures |
| top_k retrieval | 20 | 5, 10, 20, 50 | Accuracy, context size |
| max_context_chars | 30,000 | 10K, 20K, 30K, 50K | Accuracy, cost |
| Extraction model | gpt-4o-mini | gpt-4o, gpt-4o-mini | Entities extracted, accuracy, cost |
| Answer model | gpt-4o | gpt-4o, gpt-4o-mini, gpt-5-mini | Accuracy, cost |
| RRF k constant | 60 | 10, 30, 60, 100 | Retrieval quality |

---

## Implementation Priority

1. **Run LoCoMo with fixed extraction** (Exp 5 Condition A is the default) — get baseline numbers
2. **Ablate to embedding-only** (Exp 5 Condition B) — quick change, shows BM25 value
3. **Remove timelines** (Exp 1 Condition B) — quick change, shows transition value
4. **Semantic time variants** (Exp 4) — moderate effort, strongest novel claim
5. **Observation baseline** (Exp 1 Condition C) — compare to Mastra approach
6. **Procedure extraction eval** (Exp 3) — needs separate dataset, save for last
