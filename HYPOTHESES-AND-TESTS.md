# PIE: Hypotheses, Sub-Tests, and Evaluation Plan

## Core Thesis
Temporal understanding → procedural memory → proactivity → agency → continual learning. Each depends on the previous. We start at the foundation.

---

## Hypothesis 1: Semantic Temporal Context > Raw Timestamps
**Claim:** LLMs reason more accurately about temporal information when it's compiled into natural language narratives ("22 months ago, during freshman year, changed 7 times") vs raw timestamps ("1693526400") or even formatted dates ("2023-09-01").

**Test Design:**
- Take 50 temporal reasoning questions over the same knowledge base
- Condition A: Raw timestamps (unix epoch)
- Condition B: Formatted dates ("March 15, 2024")  
- Condition C: Relative time ("10 months ago")
- Condition D: Semantic temporal context ("10 months ago, during UMD freshman year, changed 3 times at ~0.5x/month")
- Measure: Answer accuracy on temporal questions (ordering, duration, recency, change detection)
- Models: GPT-4o, GPT-5, Claude Opus, Llama 3.3

**Expected Result:** D >> C > B > A. The richer the temporal language context, the better the reasoning.

**Evaluation Metric:** Accuracy on temporal QA (exact match + LLM-judge for partial credit)

**Status:** Not started

---

## Hypothesis 2: Rolling Context Ingestion > Batch Isolation
**Claim:** Processing conversations chronologically with accumulating world model context produces better entity resolution and state change detection than processing each conversation in isolation.

**Test Design:**
- Take 50 consecutive conversations
- Condition A: Process each independently (no context)
- Condition B: Process chronologically, each sees accumulated entity list + recent state changes
- Measure: Entity resolution accuracy (F1), state change detection recall, duplicate entity rate

**Expected Result:** B significantly fewer duplicates, higher state change detection (because it knows prior state), better entity resolution.

**Evaluation Metric:** 
- Entity resolution F1 (manually annotated ground truth for 50 convos)
- Duplicate entity rate (unique entities / total extracted)
- State change recall (manually annotated changes detected / total actual changes)

**Status:** Test extraction run complete (10 convos, no rolling context). Need to build rolling context version.

---

## Hypothesis 3: Graph-Structural Importance > Heuristic Importance
**Claim:** Computing entity importance from graph structure (connectivity, transition count, neighbor importance — PageRank-like) produces better importance rankings than heuristic approaches (message count, recency, keyword matching).

**Test Design:**
- Build world model from 200+ conversations
- Compute importance both ways: graph-structural vs heuristic
- Have Parth rank 100 entities by actual importance to his life/work
- Measure rank correlation (Spearman) with each method

**Expected Result:** Graph-structural importance correlates more strongly with human judgment.

**Evaluation Metric:** Spearman rank correlation coefficient

**Status:** Not started. Requires substantial graph to test.

---

## Hypothesis 4: Procedural Memory Emerges from State Transitions
**Claim:** Meaningful procedural patterns (how the user approaches problems, makes decisions, evaluates tools) can be automatically extracted from state transition sequences across entity lifecycles.

**Test Design:**
- Build world model from all 2025 conversations
- Run procedure extraction across project entities
- Have Parth evaluate: "does this procedure accurately describe how you work?"
- Compare to: (a) no procedure extraction, (b) naive pattern matching on conversation text

**Expected Result:** LLM-extracted procedures from state transitions rated higher quality than text-pattern approaches.

**Evaluation Metric:** Human rating (1-5 scale: accuracy, completeness, actionability)

**Status:** Not started. Requires full pipeline.

---

## Hypothesis 5: World Model Retrieval > Standard RAG for Temporal Queries
**Claim:** Retrieving from a temporal knowledge graph (with state transitions, period nodes, and compiled temporal context) produces better answers to temporal queries than standard vector-search RAG over conversation chunks.

**Test Design:**
- Build both systems over same data:
  - System A: Standard RAG (chunk conversations, embed, retrieve top-k)
  - System B: PIE world model (temporal KG + semantic context compilation)
- Create 100 queries spanning:
  - Current state ("What's the status of project X?")
  - Temporal diff ("How has X changed since January?")
  - Temporal ordering ("What did I work on before Y?")
  - Contradiction detection ("Have I changed my mind about Z?")
  - Procedural ("How do I typically approach X?")
- Measure answer quality via LLM judge + human eval

**Expected Result:** System B significantly better on temporal diff, ordering, and contradiction queries. Comparable on current state.

**Evaluation Metric:** LLM-judge accuracy (GPT-5 as judge), human eval on 30-question subset

**Status:** Not started.

---

## Hypothesis 6: Tool-System Extraction as System Design Knowledge
**Claim:** Extracting tool usage patterns, architectures, and system designs from conversations creates a structured knowledge base useful for AI-assisted system design.

**Test Design:**
- Extract all tool/technology entities with their relationships (uses, replaces, integrates_with)
- Extract system architecture patterns (what tools are used together, in what configurations)
- Store separately from personal world model
- Test: Given a new system design problem, can retrieval from this KB improve design suggestions?
- Compare: (a) vanilla LLM, (b) LLM + tool KB retrieval, (c) LLM + full world model

**Expected Result:** System design suggestions from (b) reference more relevant, proven tool combinations.

**Evaluation Metric:** Human rating of design quality, relevance of tool suggestions

**Status:** Not started. Need to design tool extraction schema.

---

## Hypothesis 7: Significance Scoring Calibration
**Claim:** The extraction LLM can accurately distinguish between high-significance conversations (life decisions, project pivots) and low-significance ones (one-off questions, debugging).

**Test Design:**
- Parth rates 100 conversations on 0-1 significance scale
- Compare to model's significance scores
- Test different prompt strategies for significance scoring

**Expected Result:** Strong correlation (>0.7 Pearson) after prompt tuning.

**Evaluation Metric:** Pearson correlation, ROC-AUC for binary high/low classification

**Status:** Initial test shows tight clustering (0.3-0.7). Needs calibration.

---

## Hypothesis 8: Daily Batch + Activity Context > Per-Conversation Extraction
**Claim:** Processing conversations in daily batches with an activity-based context preamble (recently active entities, project states) produces more accurate entity extraction and better implicit connection detection than per-conversation extraction with mention-based context retrieval.

**Test Design:**
- Take 30 days of conversations (~180 convos)
- Condition A: Per-conversation extraction (no context)
- Condition B: Per-conversation with mention-based context retrieval
- Condition C: Daily batch with activity-based context preamble
- Manually annotate 50 "implicit connections" (subtask conversations that belong to a named project)
- Measure: implicit connection detection rate, entity duplication rate, LLM calls, cost

**Expected Result:** C detects >80% of implicit connections. A detects <20%. B somewhere in between.

**Evaluation Metric:** Implicit connection recall, duplicate entity rate, cost per conversation

**Status:** Not started. CONTEXT-WINDOW-FIX.md documents the approach.

---

## Sub-Tests (Implementation Validation)

### T1: Parser Correctness
- Verify tree linearization produces correct turn ordering
- Check: no content dropped, timestamps preserved, multimodal content handled
- Status: ✅ Basic parser working

### T2: Extraction Prompt Quality
- Run same 10 conversations with 3 different prompt variants
- Measure: entity count, type adherence, state change detection rate
- Current issue: over-extraction of noise entities, type drift ("function" not in schema)
- Status: 🔄 First run complete, needs iteration

### T3: Entity Resolution Accuracy
- Process 30 consecutive conversations, check for duplicate detection
- "Lucid Academy" = "Lucid" = "the summer program" = "SRA" should resolve
- Status: Not started

### T4: Scaling Characteristics
- Measure: tokens/conversation, cost/conversation, time/conversation
- Project total cost for 927 conversations
- Status: Partial (10 convos: avg 3,792 tokens/convo, ~$0.02/convo → ~$18.50 for all 927 with gpt-4o)

### T5: Personal vs General Knowledge Separation
- Design schema that separates:
  - Personal world model (entities specific to Parth's life)
  - General knowledge patterns (tool architectures, system designs, concept relationships)
- Test: Can we populate the general KB from extraction without leaking personal info?
- Status: Not started

---

## Evaluation Benchmarks Comparison

### LoCoMo (232 citations — most popular)
- **Paper:** "Evaluating Very Long-Term Conversational Memory of LLM Agents" (Maharana et al., Feb 2024, Snap Research)
- **What it tests:** Very long-term conversations (300 turns, 9K tokens avg, up to 35 sessions)
- **Temporal eval method:** Temporal reasoning questions embedded in long conversations. Tests ordering, recency, duration understanding.
- **Question types:** Single-hop, multi-hop, temporal, commonsense, world knowledge, adversarial
- **How it quantifies:** QA accuracy on each question type. Temporal questions specifically test if the model can order events and reason about when things happened.
- **Strengths:** Most cited, established baseline. Has temporal + causal dynamics.
- **Weaknesses:** Human-human conversations only (not human-AI). Relatively short context. LongMemEval authors note it "fails to evaluate recall of assistant-side information or reasoning with updated user info."

### LongMemEval (119 citations — ICLR 2025)
- **Paper:** "Benchmarking Chat Assistants on Long-Term Interactive Memory" (Wu et al., Oct 2024, UCLA/Tencent)
- **What it tests:** 5 core abilities: Information Extraction, Multi-Session Reasoning, Temporal Reasoning, Knowledge Updates, Abstention
- **Temporal eval method:** 
  - Temporal reasoning questions require leveraging explicit AND implicit time cues
  - Example: "What was the last restaurant I asked about?" requires tracking temporal ordering across sessions
  - Time-aware query expansion: extracting timestamps from facts, narrowing search by time range
  - Measures improvement from time-aware indexing (7-11% improvement on temporal subset)
- **Scale:** 500 questions, 115K tokens (LongMemEval_S) up to 1.5M tokens (LongMemEval_M)
- **How it quantifies:** Accuracy per ability category. GPT-4o as judge (97% agreement with humans). Also measures Recall@k and NDCG@k for retrieval quality.
- **Strengths:** Most comprehensive ability coverage. Scalable context length. Human-AI conversations (more realistic). Tests knowledge updates (critical for our use case). ICLR venue.
- **Weaknesses:** Newer, slightly fewer citations than LoCoMo.

### Test of Time (76 citations — ICLR 2025)
- **Paper:** "Test of Time: A Benchmark for Evaluating LLMs on Temporal Reasoning" (Fatemi et al., Jun 2024, Google Research)
- **What it tests:** Two distinct temporal skills independently:
  1. **ToT-Semantic:** Temporal semantics and logic (ordering, before/after, during/overlap, Allen's temporal relations)
  2. **ToT-Arithmetic:** Temporal calculations (duration computation, date math)
- **Temporal eval method:**
  - Synthetic datasets — entities are anonymized to prevent parametric knowledge shortcuts
  - Tests different graph structures (chains, trees, DAGs) at varying complexity levels
  - Controls for: problem size, question type, fact order, graph structure
  - Key finding: LLMs rely on parametric knowledge, not actual reasoning. Performance drops dramatically with anonymized entities.
- **How it quantifies:** Accuracy per question type and graph structure. Separately measures semantic understanding vs arithmetic ability.
- **Strengths:** Pure temporal reasoning isolation (no confounds). Synthetic = no data contamination. Reveals that LLMs fake temporal reasoning via memorization.
- **Weaknesses:** Doesn't test in context of long-term memory systems. Abstract/synthetic setting less directly applicable to real-world use.

### Recommendation for PIE
**Primary benchmark:** LongMemEval — closest to our use case (chat assistant memory), tests temporal reasoning + knowledge updates + multi-session reasoning. ICLR venue gives credibility.

**Secondary benchmark:** Test of Time — validates our core claim about semantic temporal context vs raw timestamps. The anonymization finding directly supports our thesis.

**Tertiary:** LoCoMo — most cited, good for baseline comparisons with existing work.

**Custom eval:** Ablation study (Hypothesis 1) is our strongest novel contribution. No existing benchmark specifically tests semantic temporal compilation.
