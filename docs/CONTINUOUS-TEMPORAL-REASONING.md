# Continuous Temporal Reasoning in LLMs: The Frontier

## The Problem

LLMs process text as discrete tokens. Time is continuous. This mismatch creates fundamental limitations:

| What LLMs Can Do | What They Can't Do Natively |
|------------------|---------------------------|
| "A happened before B" | Precise duration calculations |
| Parse dates from text | Temporal interval algebra |
| Relative ordering | "What was happening during X?" |
| "last week", "tomorrow" | Continuous decay of relevance |

## Current Approaches (What We Built)

### 1. Bi-Temporal Modeling
```
Entity {
  ingested_at: float  // When we learned about it (system time)
  valid_from: float   // When it started being true (event time)
  valid_to: float     // When it stopped being true
}
```

**Strengths:** Enables "what did I know on date X" vs "what was true on date X"
**Limitations:** Still discrete timestamps, no continuous interpolation

### 2. Graph-Aware Temporal Retrieval
```
Query → Intent Parsing → Seed Selection → Graph Traversal → Temporal Filter → Context
```

**Strengths:** Can follow temporal chains, find evolution patterns
**Limitations:** LLM does the reasoning, graph just retrieves

### 3. Semantic Temporal Compilation
Convert events to prose with relative time markers:
```
"Project X started in early January (3 weeks before query date).
 By mid-January, it had evolved to include Y.
 This contradicts the earlier state where Z was true."
```

**Strengths:** Plays to LLM's text understanding
**Limitations:** Loses precision in translation

## The Frontier: What's Possible But Not Widely Deployed

### 1. Temporal Point Processes + LLMs (TPP-LLM)

**Key papers:**
- [TPP-LLM](https://arxiv.org/abs/2410.02062): Fine-tune LLMs for event sequence modeling
- [Language-TPP](https://arxiv.org/abs/2502.07139): Integrate continuous-time TPPs with language models
- [LAMP](https://arxiv.org/abs/2501.14291): LLM abductive reasoning for event prediction

**Core idea:** Model events as a point process (continuous-time stochastic process), use LLM for semantic understanding of event content.

```python
# TPP gives you: P(next_event | history) as continuous function of time
# LLM gives you: semantic understanding of what the event means
# Combined: "Given the user visited the doctor on Jan 5, what's the 
#            probability they'll need a follow-up, and when?"
```

**Why this matters for PIE:**
- Could predict "you'll probably need to revisit this decision in ~2 weeks"
- Model decay of relevance (recent events matter more)
- Handle irregular time gaps naturally

### 2. Continuous-Time Attention

Standard attention: `Attention(Q, K, V)` — no time awareness

Time-aware attention:
```
Attention(Q, K, V, t) = softmax(QK^T / sqrt(d) + f(t_q - t_k)) V
```

Where `f(Δt)` is a learned function of time difference.

**Papers:**
- Time-Aware Transformers (2022)
- Temporal Fusion Transformers
- Neural ODE attention

**For PIE:** Weight retrieved context by temporal relevance, not just semantic similarity.

### 3. Neural ODEs for Memory Decay

```python
# Standard: memory[t] = static embedding
# Neural ODE: dh/dt = f(h, t; θ)
# Memory evolves continuously between observations
```

**What this enables:**
- "What was the state of project X as of March 15?" — interpolate between known states
- Graceful decay: old memories become fuzzier, recent ones sharp
- Handle missing data naturally

### 4. Temporal Knowledge Graph Completion

Standard KG: `(entity, relation, entity)` — static triples
Temporal KG: `(entity, relation, entity, [t_start, t_end])` — facts have validity windows

**Challenge:** Most KG completion methods assume static graphs
**Frontier:** Learn embeddings that evolve over time

```python
# Static: embed(entity) = fixed vector
# Temporal: embed(entity, t) = f(base_embed, t)
```

## What Would "True" Continuous Temporal Reasoning Look Like?

### The Ideal System

```python
class ContinuousTemporalMemory:
    def query(self, question: str, reference_time: float) -> str:
        """
        Answer question from perspective of any point in time.
        
        Internally:
        1. Parse temporal scope of question
        2. Retrieve events with continuous-time relevance weighting
        3. Interpolate entity states at query time
        4. Model uncertainty about past (decay) and future (prediction)
        5. Generate answer with temporal confidence bounds
        """
        
    def update(self, event: str, timestamp: float):
        """
        Integrate new information as continuous update, not discrete add.
        - Update entity state trajectories
        - Revise temporal predictions
        - Maintain consistency across time
        """
```

### Why We're Not There Yet

1. **Training data is discrete:** Conversations happen at discrete times, not continuously
2. **LLM architecture:** Transformers are fundamentally sequence models, not continuous-time
3. **Evaluation is hard:** No benchmarks for continuous temporal reasoning
4. **Computational cost:** Neural ODEs / TPPs are expensive at inference

## Practical Next Steps for PIE

### Short-term (launch-ready)
- [x] Bi-temporal entity model
- [x] Graph-aware retrieval with temporal filtering
- [ ] **Fix extraction to actually produce events** ← current blocker
- [ ] Temporal weighting in retrieval (recency bias)

### Medium-term (v2)
- [ ] Temporal decay function for relevance scoring
- [ ] Event prediction: "you'll probably revisit X by date Y"
- [ ] State interpolation: approximate entity state at arbitrary time

### Long-term (research direction)
- [ ] TPP-LLM integration for continuous event modeling
- [ ] Time-aware attention in retrieval
- [ ] Temporal KG embeddings that evolve

## The Honest Answer

**Is bi-temporal + graph traversal the best we can do?**

For production systems in 2025? Mostly yes. The alternatives (TPP-LLM, Neural ODEs, continuous-time attention) are:
- Research-stage, not production-ready
- Computationally expensive
- Require specialized training

But they represent where the field is heading. PIE's architecture is compatible with these advances — the bi-temporal model is the right foundation, we'd just swap the retrieval/reasoning layer.

**The real bottleneck right now isn't the architecture — it's that we're not extracting temporal data in the first place.**

Fix extraction → get events with dates → bi-temporal queries start working → then we can measure if fancier approaches help.
