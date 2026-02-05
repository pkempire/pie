# Evaluation Benchmarks Research for PIE

*Research compiled: February 2026*

This document analyzes three primary evaluation benchmarks (LongMemEval, LoCoMo, Test of Time) for evaluating PIE's temporal knowledge graph capabilities, plus newer/adjacent benchmarks for KG quality, entity resolution, and temporal completeness.

---

## Table of Contents

1. [LongMemEval](#1-longmemeval)
2. [LoCoMo](#2-locomo)
3. [Test of Time (ToT)](#3-test-of-time-tot)
4. [Gap Analysis: What These Benchmarks Miss vs. What PIE Does](#4-gap-analysis)
5. [Newer & Adjacent Benchmarks (2025)](#5-newer--adjacent-benchmarks-2025)
6. [Evaluation Strategy for PIE](#6-evaluation-strategy-for-pie)
7. [Key Comparable Systems Already Benchmarked](#7-key-comparable-systems-already-benchmarked)

---

## 1. LongMemEval

**Paper:** Wu et al., "Benchmarking Chat Assistants on Long-Term Interactive Memory" (ICLR 2025)
**Repo:** https://github.com/xiaowu0162/LongMemEval
**Dataset:** https://huggingface.co/datasets/xiaowu0162/longmemeval-cleaned
**Project Page:** https://xiaowu0162.github.io/long-mem-eval/

### What It Tests

500 high-quality questions testing **five core long-term memory abilities**:

| Ability | What It Measures | PIE Relevance |
|---------|-----------------|---------------|
| **Information Extraction** | Recall specific info from chat history (user-side and assistant-side) | ✅ Direct — PIE's entity/state extraction |
| **Multi-Session Reasoning** | Synthesize info across multiple sessions (aggregation, comparison) | ✅ Direct — PIE's cross-conversation graph queries |
| **Knowledge Updates** | Recognize changes in user info, update dynamically | ✅✅ Core — PIE's state transitions & contradiction detection |
| **Temporal Reasoning** | Awareness of time mentions and timestamp metadata | ✅✅ Core — PIE's temporal context compilation |
| **Abstention** | Refrain from answering when info isn't in history | ⚠️ Partial — depends on query layer, not graph |

### Question Types (7 types across those 5 abilities)

1. `single-session-user` — fact mentioned by user in one session
2. `single-session-assistant` — fact from assistant response
3. `single-session-preference` — implicit user preference
4. `temporal-reasoning` — requires time-aware reasoning
5. `knowledge-update` — user's info changed over time
6. `multi-session` — requires combining info across sessions
7. `*_abs` suffix — abstention variants (unanswerable)

### Data Format

Three JSON files, each containing 500 instances:

- **`longmemeval_oracle.json`** — only evidence sessions (oracle retrieval)
- **`longmemeval_s_cleaned.json`** — ~115k tokens of history (~40 sessions)
- **`longmemeval_m_cleaned.json`** — ~500 sessions (~1.5M tokens)

**Schema per instance:**
```json
{
  "question_id": "string",
  "question_type": "single-session-user | temporal-reasoning | knowledge-update | multi-session | ...",
  "question": "What is Parth's current GPA?",
  "answer": "expected answer string",
  "question_date": "2024-03-15",
  "haystack_session_ids": ["s1", "s2", ...],
  "haystack_dates": ["2023-06-01", "2023-07-15", ...],
  "haystack_sessions": [
    [
      {"role": "user", "content": "...", "has_answer": true},
      {"role": "assistant", "content": "..."}
    ]
  ],
  "answer_session_ids": ["s5", "s12"]
}
```

Key design: timestamped multi-session chat history in standard chat format. The `has_answer` field on individual turns enables turn-level recall evaluation.

### How to Run the Eval

**Step 1:** Feed timestamped history to your system, collect answers.

**Step 2:** Save output as JSONL with `question_id` and `hypothesis` fields.

**Step 3:** Run GPT-4o-as-judge evaluation:
```bash
export OPENAI_API_KEY=...
cd src/evaluation
python3 evaluate_qa.py gpt-4o your_hypothesis_file ../../data/longmemeval_oracle.json
```

**Step 4:** Aggregate scores:
```bash
python3 print_qa_metrics.py gpt-4o your_hypothesis_file.log ../../data/longmemeval_oracle.json
```

### Metrics

- **QA Accuracy** — per-question binary score via GPT-4o judge (is the hypothesis semantically correct?)
- **Memory Recall** — session-level and turn-level recall@k (did retrieval find the right evidence?)
- **Per-ability breakdown** — accuracy split by the 5 abilities
- Separate scores for S (115k) vs M (1.5M) difficulty levels

### How PIE Would Integrate

**Adaptation approach:**
1. **Ingest phase:** Feed each `haystack_session` as a conversation to PIE's ingestion pipeline. Each session has a timestamp from `haystack_dates`. PIE extracts entities, state transitions, and relationships.
2. **Query phase:** For each question, use PIE's semantic context compilation to generate a context document from the temporal knowledge graph. Pass context + question to an LLM reader.
3. **Output:** LLM generates `hypothesis` answer. Save as JSONL.

**What PIE brings to the table:**
- Knowledge-update questions should be PIE's **strongest** category — state transitions literally track these
- Temporal reasoning benefits from PIE's semantic time anchoring (periods, not raw timestamps)
- Multi-session reasoning benefits from entity-level aggregation across conversations
- Retrieval recall should be high because PIE indexes by entity/relationship, not just text similarity

**Adaptation complexity: LOW** — Standard QA eval. PIE just needs to serve as the memory/retrieval layer.

---

## 2. LoCoMo

**Paper:** Maharana et al., "Evaluating Very Long-Term Conversational Memory of LLM Agents" (ACL 2024)
**Repo:** https://github.com/snap-research/locomo
**Dataset:** `./data/locomo10.json` in the repo
**Project Page:** https://snap-research.github.io/locomo/

### What It Tests

Three evaluation tasks across **10 very long-term conversations** (each ~300 turns, ~9K tokens, up to 35 sessions):

**Task 1: Question Answering** — 1,986 questions across 5 reasoning types:

| Category ID | Type | Count | PIE Relevance |
|-------------|------|-------|---------------|
| 1 | Multi-Hop (synthesize across sessions) | 282 | ✅ Graph traversal |
| 2 | Temporal Reasoning (time-related cues) | 321 | ✅✅ Core |
| 3 | Open-Domain (requires world knowledge) | 96 | ⚠️ Partial |
| 4 | Single-Hop (direct fact recall) | 841 | ✅ Entity retrieval |
| 5 | Adversarial (designed to trick) | 446 | ⚠️ Not scored officially |

*Only categories 1-4 (1,540 questions) are used for official scoring.*

**Task 2: Event Graph Summarization** — Extract causal/temporal event graphs from conversations

**Task 3: Multi-modal Dialog Generation** — Generate contextually appropriate responses (includes images)

### Data Format

Single JSON file (`locomo10.json`), 10 conversation samples:

```json
{
  "sample_id": "string",
  "conversation": {
    "speaker_a": "Audrey",
    "speaker_b": "Andrew",
    "session_1": [...turns...],
    "session_1_date_time": "2023-01-15 10:30",
    "session_2": [...turns...],
    "session_2_date_time": "2023-01-22 14:00"
  },
  "observation": {
    "session_1_observation": "...",
    "session_2_observation": "..."
  },
  "session_summary": {
    "session_1_summary": "...",
    "session_2_summary": "..."
  },
  "event_summary": {
    "events_session_1": {
      "speaker_a": [...events...],
      "speaker_b": [...events...]
    }
  },
  "qa": [
    {
      "question": "When did Audrey get promoted?",
      "answer": "In March 2023",
      "category": 2,
      "evidence": ["dia_15", "dia_42"]
    }
  ]
}
```

Each turn within a session:
```json
{
  "speaker": "Audrey",
  "dia_id": "dia_15",
  "text": "I just got promoted at work!",
  "img_url": "optional_url",
  "blip_caption": "optional_caption",
  "search_query": "optional_query"
}
```

### How to Run the Eval

**QA Task (primary):**
```bash
# OpenAI models
bash scripts/evaluate_gpts.sh

# Claude models
bash scripts/evaluate_claude.sh

# HuggingFace models
bash scripts/evaluate_hf_llm.sh

# RAG-based evaluation
bash scripts/evaluate_rag_gpts.sh
```

**Scoring:** Uses LLM-as-judge (GPT-4o or GPT-4o-mini). Binary 0/1 score per question: does the predicted answer match ground truth semantically?

**Key metric:** `llm_score` — weighted average across categories by question count.

### Metrics

- **LLM Judge Score (llm_score)** — binary per question, weighted average across categories
- **F1 Score** — token-level overlap for answer prediction
- **Per-category breakdown** — separate scores for single-hop, multi-hop, temporal, open-domain
- **Event Summarization** — ROUGE/BERTScore for event graph extraction
- **MM-Relevance** — for multimodal dialog generation

### How PIE Would Integrate

**Adaptation approach:**
1. **Ingest:** Feed each conversation's sessions chronologically, with timestamps from `session_N_date_time`. PIE extracts entities, relationships, and state transitions for each speaker.
2. **QA Task:** For each question, query PIE's knowledge graph to compile relevant context. Use semantic search + graph traversal. Feed to LLM for answer generation.
3. **Event Summarization:** This is actually **very close to what PIE already does** — PIE's state transitions + entity relationships = an event graph. Compare PIE's extracted graph against the annotated `event_summary` ground truth.

**What PIE brings:**
- **Event summarization** is the most natural fit — PIE literally builds temporal event graphs
- Temporal questions benefit from PIE's period-based time reasoning
- Multi-hop benefits from PIE's entity-relationship graph enabling multi-step traversal
- The `observation` and `session_summary` data can be compared against PIE's extraction output

**Adaptation complexity: LOW-MEDIUM** — QA is straightforward. Event summarization requires mapping PIE's graph format to LoCoMo's event format.

---

## 3. Test of Time (ToT)

**Paper:** Fatemi et al., "Test of Time: A Benchmark for Evaluating LLMs on Temporal Reasoning" (ICLR 2025)
**Dataset:** https://huggingface.co/datasets/baharef/ToT
**Paper:** https://arxiv.org/abs/2406.09170
**Authors:** Google Research / Google DeepMind

### What It Tests

Two distinct temporal reasoning skills, evaluated independently:

**ToT-Semantic (2,800 questions):** Tests understanding of temporal semantics and logic using synthetic data.

| Question Type | Example | Count per config |
|---------------|---------|-----------------|
| EventAtTimeT | "Which entity had relation R with E at time T?" | 10 per (graph × sort) |
| EventAtWhatTime | "When did R between E1 and E2 start/end?" | 10 per config |
| NumberOfEventsInInterval | "How many entities had R with E between T1-T2?" | 10 per config |
| BeforeAfter | "Which entity had R with E1 right before/after E2?" | 10 per config |
| EventAtTimeOfAnother | "When E1 had R1 with E2, who had R2 with E3?" | 10 per config |
| FirstLast | "Which entity first had relation R with E?" | 10 per config |
| RelationDuration | "The k-th time R happened between E1&E2, how long?" | 10 per config |
| Timeline | "Sort entities that had R with E chronologically" | 10 per config |

Total: 7 graph types × 8 question types × 5 sort orders × 10 samples = **2,800 questions**

Graph types: Erdős-Rényi, Scale-Free, Barabási-Albert, Stochastic Block Model, Star, Complete, Anonymized WikiData Extract (AWE)

**ToT-Arithmetic (7 categories):** Tests practical time calculations.

| Category | Example |
|----------|---------|
| AddSubtract | "War started 360 BC, lasted 8 years. When did it end?" |
| Compare | Compare dates in different formats chronologically |
| Duration | Compute difference between two dates/times |
| Schedule | Find mutual free time across blocked schedules |
| Timezone | Convert times across timezones |
| Sequence | Determine order of events |
| Mixed | Combination of operations |

### Data Format

HuggingFace dataset with question-answer pairs:

**ToT-Arithmetic format:**
```json
{
  "category": "add_subtract",
  "question": "The war started in 360 BC and went on for 8 years. What year did the war end in? Format your answer as a JSON, like JSON = {\"explanation\": <step by step>, \"answer\": <year> <era>}",
  "answer": {"answer": "352 BC"}
}
```

**ToT-Semantic format:**
```
Context: E11 was the R21 of E23 from 1983 to 1985.
E23 was the R21 of E32 from 2007 to 2013. ...

Question: Which entity was the R17 of E23 at the time when E32 started being the R21 of E23?
Answer: E51
```

Key design choice: **All entities are anonymized** (E11, E23, R21) to prevent LLMs from using parametric knowledge. Tests pure temporal reasoning.

### How to Run the Eval

The paper mentions open-sourcing evaluation framework, but the primary release is the dataset on HuggingFace. Evaluation is:

1. Load questions from HuggingFace dataset
2. Prompt LLM with question (includes format instructions in question text)
3. Parse JSON response
4. Compare extracted answer to ground truth
5. Exact match or semantic equivalence scoring

No complex evaluation scripts — straightforward JSON answer comparison.

### Metrics

- **Accuracy** — exact match on extracted answer
- **Per-question-type breakdown** — 8 categories for semantic, 7 for arithmetic
- **Per-graph-type breakdown** — how does performance vary by graph structure?
- **Per-sort-order analysis** — effect of fact presentation order on accuracy
- **Problem-size scaling** — accuracy vs number of nodes/facts

### How PIE Would Integrate

**This benchmark is the LEAST natural fit for PIE.** Here's why:

ToT tests whether an LLM can reason about temporal facts **provided in the prompt**. PIE is a system for **extracting and storing** temporal knowledge from conversations, then compiling context for queries. ToT doesn't involve storage/retrieval — it's pure reasoning.

**Possible adaptation:**
1. Parse the synthetic facts into PIE's entity/relationship format
2. Build a temporal knowledge graph from the facts
3. Use PIE's temporal context compiler to generate context
4. Query the compiled context to answer the question

This would test whether PIE's **temporal context compilation** (converting raw temporal data into LLM-readable narratives) helps or hurts temporal reasoning compared to raw facts.

**Adaptation complexity: MEDIUM-HIGH** — Requires building synthetic ingestion path. Tests reasoning, not memory.

---

## 4. Gap Analysis

### What These Benchmarks Test vs. What PIE Does

| Capability | LongMemEval | LoCoMo | ToT | PIE Does This? |
|-----------|:-----------:|:------:|:---:|:---------------:|
| Fact retrieval from history | ✅ | ✅ | ❌ | ✅ Entity/state retrieval |
| Multi-session synthesis | ✅ | ✅ | ❌ | ✅ Graph traversal |
| Knowledge update tracking | ✅ | ❌ | ❌ | ✅✅ State transitions |
| Temporal reasoning | ✅ | ✅ | ✅✅ | ✅ Semantic time compilation |
| Temporal arithmetic | ❌ | ❌ | ✅ | ❌ Not a PIE concern |
| Abstention / unanswerable | ✅ | ✅ (adversarial) | ❌ | ⚠️ Depends on query layer |
| Entity resolution | ❌ | ❌ | ❌ | ✅✅ Core PIE capability |
| State contradiction detection | ❌ | ❌ | ❌ | ✅✅ Core PIE capability |
| Procedural memory | ❌ | ❌ | ❌ | ✅✅ Core PIE capability |
| Causal event graphs | ❌ | ✅ | ❌ | ✅ State transition chains |
| Graph quality / completeness | ❌ | ❌ | ❌ | Need custom eval |
| Multi-modal context | ❌ | ✅ | ❌ | ❌ Not in PIE scope |

### Critical Gaps — Things PIE Does That NO Benchmark Tests

1. **Entity Resolution Accuracy** — PIE resolves "SRA", "Science Research Academy", "scifair.tech", "the science fair platform" to a single entity. No benchmark specifically tests this in conversational data.

2. **State Transition Correctness** — PIE tracks `from_state → to_state` with `transition_type` (creation, update, contradiction, resolution, archival). No benchmark evaluates whether state transitions are correctly identified and typed.

3. **Procedural Memory** — PIE extracts recurring patterns ("every time Parth starts a project: deep research → narrow scope → build fast"). No benchmark tests procedure extraction from conversations.

4. **Temporal Context Compilation Quality** — PIE's semantic compiler converts graph data into LLM-readable narratives with periods, staleness, and change velocity. No benchmark evaluates compilation quality separately from answer quality.

5. **Knowledge Graph Completeness** — Given N conversations, did PIE extract all entities, relationships, and state changes? No benchmark provides a ground-truth knowledge graph for comparison.

6. **Contradiction Detection Precision/Recall** — PIE explicitly flags when new information contradicts prior state. No benchmark provides labeled contradictions for evaluation.

---

## 5. Newer & Adjacent Benchmarks (2025)

### 5.1 MemoryBench (Supermemory, 2025)

**Repo:** https://github.com/supermemoryai/memorybench
**What:** Unified benchmark framework for comparing memory providers (Supermemory, Mem0, Zep)
**Benchmarks included:** LoCoMo, LongMemEval, ConvoMem
**Why it matters:** Provides a **pluggable evaluation harness** with:
- Checkpointed pipeline: ingest → index → search → answer → evaluate
- Multi-provider comparison (same benchmark, different systems)
- Judge-agnostic (GPT-4o, Claude, Gemini)
- Web UI for inspection

**PIE integration:** Could add PIE as a provider in MemoryBench. This would give us standardized comparison against Supermemory, Mem0, and Zep on LoCoMo and LongMemEval with minimal effort.

### 5.2 TGB 2.0 — Temporal Graph Benchmark (2024-2025)

**Paper:** Gastinger et al., "TGB 2.0: A Benchmark for Learning on Temporal Knowledge Graphs"
**Website:** https://tgb.complexdatalab.com/
**What:** Benchmark for **link prediction** on temporal knowledge graphs and heterogeneous graphs
**Datasets:** 8 novel datasets across 5 domains, up to 53M edges
**Metric:** MRR (Mean Reciprocal Rank) with challenging negative edge sampling
**Why it matters for PIE:** Tests whether a temporal KG model can predict future links — relevant if PIE wants to do proactive suggestion ("based on past patterns, you might want to..."). Not directly relevant to PIE's current scope (extraction + retrieval, not prediction).

### 5.3 MEMTRACK (NeurIPS 2025 Workshop)

**Paper:** Deshpande et al., "MEMTRACK: Evaluating Long-Term Memory and State Tracking in Multi-Platform Dynamic Agent Environments"
**What:** Evaluates agents across **multi-platform** environments (not just chat). Tests state tracking across different interaction modalities.
**Why it matters:** Closer to PIE's real-world scenario (user interacts across multiple platforms). More realistic than single-conversation benchmarks.

### 5.4 StoryBench (2025)

**Paper:** Wan & Ma, "StoryBench: A Dynamic Benchmark for Evaluating Long-Term Memory with Multi Turns"
**What:** Emphasizes **long-term sequential reasoning** where memory must support inference, self-correction, and causal tracking.
**Why it matters:** Tests causal/sequential reasoning across turns — relevant to PIE's state transition chains and procedural memory.

### 5.5 NLPCC 2024 Shared Task — Dialogue Coreference Resolution

**What:** Benchmark for dialogue-level coreference resolution and relation extraction
**Why it matters:** Closest existing benchmark to PIE's entity resolution in conversations. Tests identifying that "he", "my brother", "Pranay" all refer to the same entity within dialogue.

### 5.6 TGB 2.0 Standard Metrics for TKG Quality

From the TKG completion literature, standard quality dimensions for knowledge graphs:

| Dimension | What It Measures | How to Evaluate for PIE |
|-----------|-----------------|------------------------|
| **Completeness** | Are all entities/relations captured? | Compare against manually labeled ground truth subset |
| **Accuracy** | Are extracted facts correct? | Spot-check sample of entity states against source conversations |
| **Timeliness** | Is temporal information current? | Verify state transitions against conversation timestamps |
| **Redundancy** | Are there duplicate entities? | Count entity clusters that should have been merged |
| **Consistency** | Are there contradictions? | Check if detected contradictions match ground truth |

Standard TKG metrics: **Hits@k**, **MRR** (Mean Reciprocal Rank), **MR** (Mean Rank) for link prediction.

---

## 6. Evaluation Strategy for PIE

### Tier 1: Use Existing Benchmarks (quickest to implement)

| Benchmark | What It Tests for PIE | Effort | Expected Advantage |
|-----------|----------------------|--------|-------------------|
| **LongMemEval** | End-to-end memory QA with temporal + knowledge-update focus | LOW | Knowledge updates, temporal reasoning |
| **LoCoMo QA** | Long-term conversational memory QA | LOW | Multi-hop, temporal |
| **LoCoMo Event Summary** | Temporal event graph extraction | MEDIUM | Direct comparison of PIE's graph vs ground truth |

**Recommended first step:** Integrate PIE as a provider in Supermemory's **MemoryBench** framework. This gives us standardized comparison against Mem0, Zep, and Supermemory on both LoCoMo and LongMemEval with minimal code.

### Tier 2: Custom Evaluations (PIE-specific capabilities)

These test what no existing benchmark covers:

**2a. Entity Resolution Accuracy**
- Take 50 conversations from PIE's training data
- Manually label all entity mentions and their ground-truth entity IDs
- Measure: precision, recall, F1 of entity resolution
- Also measure: number of false entity splits (same entity → two nodes) and false merges (two entities → one node)

**2b. State Transition Correctness**
- Take 20 entities with known state changes
- Manually label: what changed, when, what type (update vs contradiction)
- Measure: transition detection recall, type classification accuracy, timestamp accuracy

**2c. Temporal Context Compilation Quality**
- For 50 entities, generate PIE's compiled temporal context
- A/B test: LLM answers questions using (a) PIE's compiled context vs (b) raw conversation excerpts
- Measure: answer accuracy, response quality, hallucination rate

**2d. Knowledge Graph Completeness**
- For 10 conversations, manually extract all entities, relationships, and state changes
- Compare against PIE's automatic extraction
- Measure: entity recall, relationship recall, state change recall

### Tier 3: Novel Benchmark Contribution

If no existing benchmark tests PIE's core contributions, consider **creating a benchmark** focused on:

1. **Temporal State Tracking in Conversations** — Given N conversations over time, track how entity states evolve. Ground truth: labeled state timelines.
2. **Entity Resolution in Personal Conversations** — Given informal references across sessions, resolve to canonical entities. Ground truth: labeled entity clusters.
3. **Procedural Pattern Extraction** — Given conversation sequences showing repeated behaviors, extract procedures. Ground truth: labeled procedures.

This could itself be a paper contribution alongside PIE.

---

## 7. Key Comparable Systems Already Benchmarked

Understanding how comparable systems perform on these benchmarks helps set baselines:

### Zep (Temporal KG for Agent Memory)

**Most architecturally similar to PIE.** Uses Graphiti — a temporally-aware knowledge graph with:
- Episode subgraph (raw conversation data)
- Semantic entity subgraph (extracted entities + relationships)
- Community subgraph (clusters of related entities)
- Bi-temporal model (event time vs ingestion time)
- Entity resolution via embedding similarity + LLM

**Published results:**
- **DMR Benchmark:** 94.8% (vs MemGPT 93.4%)
- **LongMemEval:** Up to 18.5% accuracy improvement over baselines, 90% latency reduction
- **LoCoMo:** llm_score ~0.75 overall

**Key difference from PIE:** Zep is a production service focused on agent memory for enterprise. PIE is research-focused on semantic temporal reasoning over a personal knowledge worker's world state. PIE has state transitions with typed changes (creation/update/contradiction/resolution/archival) and procedural memory — Zep doesn't.

### Mem0 (Memory Layer for AI)

**Approach:** Extracts key conversational facts, maintains profile memory
**LoCoMo results:** llm_score varies by category, competitive on single-hop

### Supermemory (SOTA as of late 2025)

**Claims SOTA on LongMemEval_s**, particularly strong on:
- Temporal reasoning (where most systems struggle)
- Knowledge conflicts in high-noise environments

### MemMachine

**LoCoMo results:** 
- Overall: 0.8487
- Single-hop: 0.9334
- Multi-hop: 0.8050
- Temporal: 0.7259
- Open-domain: 0.6458

### Summary of Published Scores on LoCoMo

| System | Single-hop | Temporal | Multi-hop | Open-domain | Overall |
|--------|-----------|----------|-----------|-------------|---------|
| MemMachine | 0.933 | 0.726 | 0.805 | 0.646 | 0.849 |
| Memobase | 0.709 | 0.851 | 0.469 | 0.772 | 0.758 |
| Zep | 0.741 | 0.798 | 0.660 | 0.677 | 0.751 |
| Mem0 | — | — | — | — | ~0.70 (est) |

**PIE's expected advantages:** Knowledge-update and temporal reasoning categories, where PIE's explicit state transitions and semantic time compilation should outperform embedding-only retrieval.

---

## Appendix: Quick-Start Commands

### Running LongMemEval

```bash
# Clone and setup
git clone https://github.com/xiaowu0162/LongMemEval
cd LongMemEval
conda create -n longmemeval-lite python=3.9
conda activate longmemeval-lite
pip install -r requirements-lite.txt

# Download data
mkdir -p data/ && cd data/
wget https://huggingface.co/datasets/xiaowu0162/longmemeval-cleaned/resolve/main/longmemeval_oracle.json
wget https://huggingface.co/datasets/xiaowu0162/longmemeval-cleaned/resolve/main/longmemeval_s_cleaned.json

# Run evaluation on your outputs
cd src/evaluation
python3 evaluate_qa.py gpt-4o your_output.jsonl ../../data/longmemeval_oracle.json
python3 print_qa_metrics.py gpt-4o your_output.jsonl.log ../../data/longmemeval_oracle.json
```

### Running LoCoMo

```bash
# Clone
git clone https://github.com/snap-research/locomo
cd locomo

# Set API keys in scripts/env.sh
# Evaluate
bash scripts/evaluate_gpts.sh        # OpenAI models
bash scripts/evaluate_rag_gpts.sh    # RAG-based evaluation
```

### Running via MemoryBench (recommended for provider comparison)

```bash
# Clone
git clone https://github.com/supermemoryai/memorybench
cd memorybench
bun install
cp .env.example .env.local  # Add API keys

# Run single provider on LoCoMo
bun run src/index.ts run -p <your-provider> -b locomo

# Compare providers
bun run src/index.ts compare -p supermemory,mem0,zep -b locomo -s 5

# Run on LongMemEval
bun run src/index.ts run -p <your-provider> -b longmemeval
```

### Accessing Test of Time

```python
from datasets import load_dataset
tot = load_dataset("baharef/ToT")
# Splits available for arithmetic categories and semantic categories
```
