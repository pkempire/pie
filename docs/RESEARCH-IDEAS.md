# PIE Research Ideas: Novel Directions for Temporal Memory Systems

## Current Landscape (2024-2026)

### Graph-RAG Evolution
The field has moved from naive RAG → structured graph RAG → RL-optimized graph reasoning:

| Paper | Key Innovation | Relevance to PIE |
|-------|---------------|------------------|
| **GraphRAG-R1** (WWW 2026) | RL with process-constrained rewards for multi-hop graph reasoning | Direct application — train retriever with GRPO |
| **T-GRAG** (2025) | Temporal Knowledge Graph + Temporal Query Decomposition + 3-layer interactive retrieval | Almost exactly our problem space |
| **R3-RAG** (EMNLP 2025) | Step-by-step reasoning + retrieval with process rewards | Process supervision beats outcome-only |
| **GNN-RAG** | Graph neural networks for retrieval routing | Learn which edges to traverse |

### Memory Systems
| System | Architecture | What We Can Learn |
|--------|-------------|-------------------|
| **Mem0** | Graph-based memory + dynamic consolidation | 91% lower latency vs full context |
| **MMS** (Multiple Memory Systems) | Cognitive psych-inspired: episodic + semantic + retrieval units | Separate retrieval units from context units |
| **A-MEM** | Zettelkasten-style dynamic indexing | Link memory notes like a knowledge graph |
| **SimpleMem** | Semantic density gating | Filter redundant memories before storage |

---

## RL Approaches We Can Apply

### 1. **GRPO for Subgraph Selection** (from GraphRAG-R1)

Train the retriever to select better subgraphs using Group Relative Policy Optimization:

```
State: query + current retrieved entities
Action: which edges to traverse / which entities to add
Reward: downstream answer correctness + retrieval cost penalty
```

**Novel twist for PIE**: Add temporal rewards:
- **Progressive Retrieval Attenuation (PRA)**: Penalize retrieving entities outside query's temporal scope
- **Temporal Coherence Reward**: Bonus for retrieving entities with consistent temporal relationships

### 2. **Process-Supervised RL** (from R3-RAG)

Don't just reward final answer correctness — reward intermediate retrieval steps:

```python
def process_reward(query, retrieved_entity, world_model):
    # Step-level reward, not just outcome
    relevance = embedding_similarity(query, entity)
    temporal_fit = temporal_scope_match(query, entity.transitions)
    path_validity = is_valid_graph_path(retrieved_so_far, entity)
    return α * relevance + β * temporal_fit + γ * path_validity
```

### 3. **Retriever Fine-tuning via Generation Loss** (from SmartRAG)

Use the gradient of generation loss to update the retriever:
- When LLM generates wrong answer → backprop signal to retriever
- Learn which retrievals lead to good answers

---

## Novel Research Directions

### Direction 1: **Temporal Query Decomposition with RL**

T-GRAG decomposes temporal queries manually. We can **learn** to decompose:

```
"How did my views on AI safety evolve from 2024 to 2025?"
    ↓ (learned decomposition)
Sub-queries:
  1. "AI safety beliefs in early 2024"
  2. "AI safety beliefs in late 2024"  
  3. "AI safety beliefs in 2025"
  4. "What changed between these?"
```

**Experiment**: Train decomposition policy with RL where reward = temporal QA accuracy

### Direction 2: **Contrastive Temporal Representation Learning**

Current: embed(entity) = embed(name + current_state)

Proposed: **Temporal-aware embeddings** that encode evolution:

```python
def temporal_embedding(entity, query_time):
    # Embedding that changes based on when you're asking about
    state_at_time = interpolate_state(entity.transitions, query_time)
    temporal_context = encode_evolution(entity.transitions)
    return concat(
        embed(entity.name),
        embed(state_at_time),
        temporal_context,
        positional_encode(query_time - entity.first_seen)
    )
```

**Contrastive objective**: 
- Positive: (query about X in 2024, X's state in 2024)
- Negative: (query about X in 2024, X's state in 2025)

### Direction 3: **Learned Graph Traversal Policy**

Instead of heuristic BFS, train a policy network:

```
Input: current_node, query_embedding, path_so_far
Output: distribution over edges to traverse

Train with: REINFORCE where reward = final answer quality
```

**Novel**: Add **edge-type attention** — learn which relationship types matter for which query types

### Direction 4: **Memory Consolidation via RL**

Inspired by MMS paper — but with learned consolidation:

When new information arrives, learn:
1. Should this update existing entity or create new one?
2. What old information should be "forgotten" (downweighted)?
3. What associations should be strengthened?

**Reward**: downstream retrieval performance + storage efficiency

### Direction 5: **Temporal Reasoning as Planning**

Cast temporal QA as a planning problem:

```
State: current_beliefs about timeline
Action: retrieve_entity(id), traverse_edge(from, to), infer_temporal_relation(A, B)
Goal: answer query correctly

Use: Monte Carlo Tree Search with LLM as rollout policy
```

This is similar to KnowGPT's approach but specialized for temporal graphs.

---

## Experiments to Run

### Experiment 1: Retrieval Method Ablation
- **Baseline**: embedding-only (current)
- **+Intent parsing**: add query understanding
- **+Graph traversal**: add BFS from seeds
- **+Temporal filtering**: add time-aware re-ranking
- **+RL-tuned**: fine-tune retriever with GRPO

Metric: retrieval recall@k + downstream QA accuracy on ToT/LongMemEval

### Experiment 2: Reward Signal Ablation
For RL training, compare:
- Outcome-only: reward = answer_correct
- +Relevance: reward = answer_correct + retrieval_precision
- +Temporal: reward = answer_correct + temporal_coherence
- +Cost: reward = answer_correct - retrieval_cost

### Experiment 3: Temporal Embedding Methods
Compare:
- Static embeddings (current)
- Time-interpolated state embeddings
- Trajectory embeddings (encode full history)
- Temporal positional encodings

### Experiment 4: Query Decomposition
Compare:
- No decomposition
- Rule-based decomposition
- LLM-prompted decomposition
- RL-learned decomposition

### Experiment 5: Cross-benchmark Generalization
Train on ToT → test on LongMemEval
Train on LongMemEval → test on LoCoMo
(Tests if temporal reasoning transfers)

---

## Technical Implementation Ideas

### Differentiable Retrieval
To enable gradient-based RL, make retrieval differentiable:
```python
# Soft top-k selection
def soft_retrieve(query_emb, entity_embs, tau=0.1):
    scores = cosine_similarity(query_emb, entity_embs)
    weights = softmax(scores / tau)  # soft selection
    return weighted_sum(entity_embs, weights)
```

### Temporal Positional Encoding
Encode "when" an entity was relevant:
```python
def temporal_position(timestamp, reference_time, max_time=3*365*24*3600):
    delta = (reference_time - timestamp) / max_time  # normalize to [-1, 1]
    return [sin(delta * freq) for freq in [1, 2, 4, 8, 16, 32]]
```

### Process Reward Model (PRM) for Retrieval
Train a model to predict whether a retrieval step is good:
```python
class RetrievalPRM(nn.Module):
    def forward(self, query, retrieved_entities, path_so_far):
        # Predict: will this retrieval lead to correct answer?
        return score
```

Train on (retrieval_trace, final_answer_correct) pairs.

---

## Publishable Contribution Angles

1. **"Temporal-Aware Graph Retrieval for Personal Memory"** — the T-GRAG approach but for personal knowledge graphs

2. **"Learning to Navigate Temporal Knowledge Graphs with RL"** — GRPO + temporal rewards

3. **"Process-Supervised Retrieval for Temporal Reasoning"** — R3-RAG ideas applied to temporal domain

4. **"When Does Temporal Compilation Help? Task-Adaptive Preprocessing for Memory QA"** — our existing finding, formalized

5. **"Contrastive Learning for Time-Aware Entity Representations"** — temporal embeddings

---

## Priority Order for Implementation

1. **Compare retrievers with logging** (already done) — establish baseline
2. **Add GRPO training loop** — most impactful, proven to work
3. **Temporal query decomposition** — low-hanging fruit
4. **Temporal embeddings** — requires training but could be big win
5. **Full planning-based approach** — most complex, save for later

---

## Key Papers to Deep-Read

1. GraphRAG-R1: https://arxiv.org/abs/2507.23581
2. T-GRAG: https://arxiv.org/abs/2508.01680
3. R3-RAG: https://arxiv.org/abs/2505.23794
4. ProRAG: https://arxiv.org/html/2601.21912
5. MMS: https://arxiv.org/html/2508.15294v1
6. Mem0: https://arxiv.org/abs/2504.19413

---

## Questions to Answer Experimentally

1. Does RL-trained retrieval outperform heuristic graph traversal for temporal queries?
2. What's the right reward balance (accuracy vs cost vs temporal coherence)?
3. Do temporal embeddings help more for some query types than others?
4. Can we learn query decomposition, or is prompted decomposition good enough?
5. How much does process supervision help vs outcome-only rewards?
6. Does training on one temporal benchmark generalize to others?
