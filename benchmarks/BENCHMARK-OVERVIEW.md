# PIE Benchmark Suite Overview

This document describes the benchmarks used to evaluate PIE's temporal knowledge graph approach against standard baselines for long-context memory in conversational AI.

## Benchmarks Summary

| Benchmark | Focus | Size | Avg Turns | Question Types | Key Challenge |
|-----------|-------|------|-----------|----------------|---------------|
| **LongMemEval** | Temporal memory | ~500 Q | 53 sessions/Q | 6 types | Knowledge updates, temporal reasoning |
| **LoCoMo** | Very long-term | 10 convos | 300 turns | 5 types | 35+ sessions, causal understanding |
| **MSC** | Multi-session | 237K train | 5 sessions | 2 tasks | Persona consistency |

---

## 1. LongMemEval

**Source**: [LongMemEval (Xiaowu et al.)](https://github.com/xiaowu0162/LongMemEval)

### Description
LongMemEval tests long-term memory capabilities with synthetic conversations that span weeks/months. Each question comes with its own "haystack" of ~53 sessions representing a user's chat history.

### Key Characteristics
- **Format**: User + Assistant conversations
- **Scale**: ~500 questions, each with ~53 sessions
- **Temporal span**: Weeks to months of synthetic history
- **Question types**:
  - `single-session-user`: Facts from user's single message
  - `single-session-assistant`: Facts from assistant's response
  - `single-session-preference`: User preferences/opinions
  - `multi-session`: Info spanning multiple sessions
  - `knowledge-update`: Value that changed over time (test most recent)
  - `temporal-reasoning`: "When did X happen?", ordering questions

### Why It's Important for PIE
- **Knowledge updates** are the killer feature: Raw RAG retrieves old values
- **Temporal reasoning** requires understanding "before/after" relationships
- Tests PIE's state transition tracking and contradiction detection

### SOTA Scores (to beat)
| System | Overall | Temporal |
|--------|---------|----------|
| Emergence AI | 86% | — |
| Supermemory | 71.4% | 76.7% |
| Zep | 71.2% | — |
| Full context GPT-4o | 62% | — |

### Data Location
```
benchmarks/longmemeval/data/longmemeval_s_cleaned.json
```

---

## 2. LoCoMo (Long Conversation Memory)

**Source**: [LoCoMo (Snap Research, ACL 2024)](https://github.com/snap-research/locomo)

### Description
LoCoMo contains very long-term conversations between two AI agents, each with distinct personas and temporal event graphs. Conversations span up to 35 sessions with ~300 turns and 9K tokens average.

### Key Characteristics
- **Format**: Peer-to-peer chat (both speakers are "humans")
- **Scale**: 10 conversations, 200+ QA annotations
- **Temporal span**: Multiple weeks with causally-connected events
- **Question types**:
  - `single_hop`: Direct fact recall
  - `multi_hop`: Combining info from multiple sessions
  - `temporal`: When/ordering questions
  - `adversarial`: Trick questions testing hallucination
  - `commonsense`: Requires world knowledge + conversation context

### Why It's Important for PIE
- **Much longer** than LongMemEval (300 turns vs ~50)
- **Adversarial** questions catch hallucinations from full-context models
- **Causal event graphs** — tests if PIE captures relationships
- **Multimodal** (images with captions) — future extension opportunity

### Key Findings from Paper
- Long-context LLMs improve 22-66% over base but lag humans by 56%
- Temporal reasoning gap vs humans: 73%
- RAG with observations (extracted facts) performs best

### Data Location
```
benchmarks/locomo/data/locomo10.json
```
Download from: https://github.com/snap-research/locomo/blob/main/data/locomo10.json

---

## 3. MSC (Multi-Session Chat)

**Source**: [Facebook/Meta ParlAI](https://parl.ai/projects/msc/)

### Description
MSC is Meta's benchmark for long-term open-domain conversation. Human crowdworkers chat over 5 sessions, re-engaging after hours/days while maintaining persona consistency.

### Key Characteristics
- **Format**: Human-human conversation with assigned personas
- **Scale**: 237K training, 25K validation examples
- **Sessions**: Up to 5 sessions, ~14 utterances each
- **Tasks**:
  - `response_generation`: Generate next appropriate response
  - `memory_qa`: Recall facts from prior sessions (derived task)

### Why It's Important for PIE
- **Persona consistency** — tests if extracted beliefs/preferences are stable
- **Large scale** — good for statistical significance
- **Human-human** — more naturalistic than synthetic data
- **Session summaries** provided — can compare PIE extraction quality

### Key Findings from Paper
- Retrieval-augmented methods outperform standard encoder-decoder
- Methods with summarization + recall perform best
- Standard transformers struggle with multi-session context

### Data Location
```
benchmarks/msc/data/msc_valid.jsonl
```
Obtain via ParlAI:
```bash
parlai display_data -t msc -dt valid > msc_valid.txt
# Then convert to JSONL
```

---

## Baselines

### 1. Full Context (`full_context`)
Stuff the entire conversation history into the LLM prompt.

**Pros**: Simple, no retrieval errors
**Cons**: Context window limits, expensive, no structure

### 2. Naive RAG (`naive_rag`)
Embed chunks (turns or sessions), retrieve top-k by cosine similarity.

**Pros**: Scales to any history length
**Cons**: Retrieves based on surface similarity, misses temporal logic

### 3. PIE Temporal (`pie_temporal`)
Build a temporal knowledge graph, track entity states over time, compile semantic context.

**Pros**: Handles knowledge updates, temporal reasoning, explicit state tracking
**Cons**: Extraction overhead, potential entity resolution errors

---

## Running Evaluations

### Quick Sanity Check (5 samples per benchmark)
```bash
python -m benchmarks.eval_harness --baseline pie_temporal --subset 5
```

### Compare All Baselines (10 samples)
```bash
python -m benchmarks.eval_harness --baseline all --subset 10
```

### Full Evaluation (hours)
```bash
python -m benchmarks.eval_harness --baseline all
```

### Individual Benchmarks
```bash
# LongMemEval only
python -m benchmarks.longmemeval.runner --baseline pie_temporal --limit 20

# LoCoMo only
python -m benchmarks.locomo.runner --baseline pie_temporal --limit 20

# MSC only
python -m benchmarks.msc.runner --baseline pie_temporal --limit 20
```

---

## Results Dashboard

Open `benchmarks/results/dashboard.html` in a browser and load result JSON files to see:

- Comparison table across benchmarks and baselines
- Per-category breakdowns
- Win/lose analysis (PIE vs baselines)
- Score trends

---

## Key Metrics to Track

### Overall Accuracy
Percentage of questions answered correctly (via LLM-as-judge scoring).

### Per-Category Performance
- **Temporal-reasoning**: PIE's expected strength
- **Knowledge-update**: PIE's expected strength
- **Adversarial**: Tests hallucination resistance
- **Multi-hop**: Tests relationship understanding

### Efficiency Metrics
- Context tokens used (lower is better for cost)
- Retrieval count (fewer, more relevant = better)
- Latency (extraction overhead vs query time)

---

## Expected PIE Advantages

1. **Knowledge Updates**: When a fact changes over time, PIE tracks the transition and returns the most recent value. RAG returns whatever chunk has highest similarity (often the older mention).

2. **Temporal Reasoning**: PIE's context includes explicit timestamps and "X happened before Y" relationships. Full-context models must infer this from position.

3. **Hallucination Resistance**: PIE only returns facts explicitly extracted from conversation. Full-context models may hallucinate details.

4. **Efficiency**: PIE compiles a focused context (~12K chars) vs full-context (100K+ chars).

---

## Data Acquisition

### LongMemEval
Already included or download from the LongMemEval repo.

### LoCoMo
```bash
cd benchmarks/locomo/data
wget https://raw.githubusercontent.com/snap-research/locomo/main/data/locomo10.json
```

### MSC
```bash
# Requires ParlAI installation
pip install parlai
parlai display_data -t msc -dt valid --num-examples 1000 > msc_raw.txt
# Convert to JSONL format (script TBD)
```

---

## Future Benchmarks to Add

1. **DSTC11**: Task-oriented dialogue with slot tracking
2. **PersonaChat**: Persona-grounded conversation
3. **ConvAI2**: Competition benchmark for engaging conversation
4. **LIGHT**: Fantasy world conversation with world state
5. **MultiWOZ**: Multi-domain task-oriented dialogue

---

## References

- LongMemEval: https://github.com/xiaowu0162/LongMemEval
- LoCoMo: https://snap-research.github.io/locomo/
- MSC: https://parl.ai/projects/msc/
- LoCoMo Paper: https://arxiv.org/abs/2402.17753
- MSC Paper: https://arxiv.org/abs/2107.07567
