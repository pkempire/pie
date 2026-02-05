# Paper Outline: Semantic Temporal World Models for Knowledge Worker Memory

**Working Title Options:**
- "Beyond Timestamps: Semantic Temporal World Models for Long-Term Agent Memory"
- "PIE: Personal Intelligence through Temporal World Modeling"
- "From Conversations to World State: Temporal Knowledge Graphs for Knowledge Worker Memory"

---

## Abstract (~250 words)
- LLMs struggle with temporal reasoning (cite evidence: ToT, Real-Time Deadlines, LongMemEval)
- Existing memory systems pass raw timestamps → LLMs can't reason about them effectively
- We introduce semantic temporal context compilation: converting temporal graph data into natural language narratives
- We build a world model that tracks entities, state transitions, and procedures from 927 conversations
- Key contributions:
  1. Semantic temporal context compiler (novel)
  2. Rolling context ingestion for temporal knowledge graphs
  3. Procedural memory extraction from state transition patterns
  4. Comprehensive evaluation across 3 established benchmarks + custom ablation

---

## 1. Introduction (~1.5 pages)
- Knowledge workers accumulate thousands of conversations with AI assistants
- Current systems: Mem0 (facts), Honcho (user psychology), Zep/Graphiti (temporal KG)
- Missing piece: temporal reasoning over evolving world state
- The problem: LLMs can't reason about raw temporal data
  - ToT: performance drops when entities anonymized (not reasoning, memorizing)
  - Real-Time Deadlines: 4% vs 32% deal closure with/without time cues
  - LongMemEval: 30-60% accuracy drop on temporal + knowledge update tasks
- Our insight: compile temporal information into semantic narratives the LLM can parse as language
- Contributions enumerated

---

## 2. Related Work (~1.5 pages)

### 2.1 LLM Memory Systems
- Mem0: fact-level extraction and retrieval
- Honcho/Plastic Labs: user representation, dialectical dreaming
- Zep/Graphiti: temporal knowledge graphs with hybrid retrieval
- Supermemory: SOTA on LongMemEval_S
- Gap: none compile temporal context semantically

### 2.2 Temporal Reasoning in LLMs
- Test of Time (ICLR 2025): synthetic benchmark, parametric knowledge reliance
- Time-R1: 3-stage curriculum for temporal reasoning
- Real-Time Deadlines: temporal tracking failure in negotiations
- TV-LLM: temporal validity of rules in KG reasoning
- TISER: timeline self-reflection for temporal reasoning

### 2.3 World Models for Agents
- Kosmos: world models as structured JSON for agent coordination
- ExpeL: experience extraction from single tasks
- Process mining (see §2.4)
- Gap: no one builds continuous world models from conversational data

### 2.4 Process Mining and Procedural Knowledge
- Classical process mining: extracting process models from event logs (van der Aalst)
- LLMs for process mining: PM-LLM-Benchmark, natural language interfaces
- Key distinction: process mining works on structured event logs with defined activities
- Our approach: extracting implicit procedures from unstructured conversations
- Connection: state transitions in our graph ≈ events in process mining, but derived from natural language, not system logs

---

## 3. System Architecture (~3 pages)

### 3.1 Data Model
- Entities (typed: person, project, tool, belief, decision, concept, organization, period)
- State transitions (creation, update, contradiction, resolution, archival)
- Procedures (patterns across transition sequences)
- Relationships (typed edges)
- Period nodes (semantic time anchors)

### 3.2 Semantic Temporal Context Compiler
- Input: entity + transition history + current timestamp
- Output: natural language temporal narrative
- Components: relative time, period anchoring, change velocity, contradiction flagging
- Example transformations (raw → semantic)

### 3.3 Rolling Context Ingestion
- Chronological processing with accumulating world model
- Each conversation receives relevant subgraph context
- Entity mention detection → context retrieval → full extraction
- Entity resolution with embedding similarity + LLM verification

### 3.4 Procedural Memory Extraction
- Post-ingestion analysis of state transition sequences
- Pattern detection across entity lifecycles (projects, beliefs, decisions)
- Distinction from process mining: unstructured source, implicit activities
- Quality measurement

### 3.5 Tool-System Knowledge Extraction
- Separate extraction path for technology/architecture patterns
- Tool usage relationships, system configurations, integration patterns
- Stored in dedicated knowledge base (not mixed with personal world model)
- Application: AI-assisted system design

### 3.6 Forgetting and Consolidation
- Tiered archival (active → summarized → archived)
- Graph-structural importance scoring
- Periodic consolidation: merge, summarize, archive

---

## 4. Experimental Setup (~1.5 pages)

### 4.1 Dataset
- 927 ChatGPT conversations from a single knowledge worker (Jan-May 2025)
- ~2.6M tokens, spanning multiple projects, domains, life events
- Diverse model usage: GPT-4o, o3, o1, GPT-4.5, o4-mini

### 4.2 Benchmarks
- LongMemEval (primary): 5 abilities, temporal reasoning subset
- Test of Time (secondary): semantic vs arithmetic temporal reasoning
- LoCoMo (tertiary): long-term conversational memory

### 4.3 Custom Evaluations
- Semantic temporal context ablation (Hypothesis 1)
- Rolling context vs batch isolation (Hypothesis 2)
- End-to-end QA: world model retrieval vs standard RAG (Hypothesis 5)

### 4.4 Baselines
- Standard RAG (chunk + embed + retrieve)
- Mem0
- Graphiti/Zep (temporal KG without semantic compilation)
- Raw context window (all conversations in context)

---

## 5. Results (~2 pages)

### 5.1 Semantic Temporal Context Ablation
- Performance across 4 conditions: raw timestamps, formatted dates, relative time, full semantic
- Breakdown by temporal question type (ordering, duration, recency, change detection)

### 5.2 Rolling Context Ingestion
- Entity resolution accuracy: rolling vs batch
- State change detection recall
- Duplicate entity rates

### 5.3 Benchmark Results
- LongMemEval: per-ability accuracy (esp. temporal reasoning, knowledge updates)
- Test of Time: improvement with semantic compilation
- LoCoMo: comparison with existing systems

### 5.4 End-to-End Query Quality
- World model vs RAG on temporal, procedural, state, and contradiction queries
- Case studies showing specific improvements

### 5.5 Procedural Memory Quality
- Human evaluation of extracted procedures
- Comparison with text-pattern baselines

---

## 6. Analysis (~1 page)

### 6.1 What Semantic Compilation Captures
- Qualitative examples of semantic vs raw temporal context
- Failure modes and edge cases

### 6.2 Personal vs General Knowledge
- How the system separates personal world model from general tool/system knowledge
- Privacy implications and design choices

### 6.3 Limitations
- Single-user evaluation
- English-only
- LLM extraction errors propagate
- Cost considerations for full ingestion

---

## 7. Conclusion (~0.5 pages)
- Summary of contributions
- Semantic temporal compilation as a general technique
- Future work: multi-user, real-time ingestion, fine-tuning on temporal reasoning

---

## Appendix
- Full extraction prompt
- Entity type definitions and examples
- Temporal context compilation algorithm
- Additional benchmark results
- Cost analysis
