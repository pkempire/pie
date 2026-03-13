# Critical Analysis: Temporal Reasoning SOTA — What's Real vs Hype

## The Honest Truth

None of the current approaches are directly applicable to our problem. Here's why:

| Approach | Designed For | Our Problem | Gap |
|----------|--------------|-------------|-----|
| TPP-TAL | Predicting next event in a stream (finance, healthcare) | Answering questions about past conversations | Different task entirely |
| MemoTime | QA over structured temporal KGs (Wikidata-style) | QA over unstructured conversation history | We don't have clean quadruples |
| Real-Time Deadlines | Real-time negotiation with deadlines | Long-term memory retrieval | Different temporal scale |

**Our actual problem:** Given 1000+ conversations over months, answer questions like "When did I decide to use React for the project?" or "What was my opinion on X before I changed my mind?"

---

## TPP-TAL: Deep Dive

### What It Actually Does

```python
# The "magic" is embarrassingly simple:
class TimeTextCrossAttention:
    def forward(self, text_tokens, time_token):
        # text_tokens: (L, D) - the event description
        # time_token: (D,) - embedding of the timestamp
        
        Q = self.q_linear(text_tokens)  # Query from text
        K = self.k_linear(time_token)   # Key from time
        V = self.v_linear(time_token)   # Value from time
        
        # Cross-attention: text attends to time
        attn = softmax(Q @ K.T / sqrt(dk))
        out = attn @ V
        
        return LayerNorm(text_tokens + out)
```

### What's Novel
1. **Cross-attention fusion** instead of concatenation — lets each text token dynamically weight temporal info
2. **Per-head temporal bias** — different attention heads learn different temporal scales (one for short-term, one for long-term)
3. **Log-bucketized time differences** — handles wide range of temporal distances stably

### What's NOT Novel
- The architecture is standard transformer with additive bias
- The idea of temporal embeddings is old (Time2Vec, etc.)
- They only show results on event prediction, not QA

### Applicability to PIE
**Partial.** We could use:
- Cross-attention to fuse temporal context into retrieval queries
- Per-head temporal bias in our attention over retrieved chunks

**But:** TPP-TAL operates on event streams. We have conversations with complex temporal references ("last week", "before the deadline"). Different problem.

---

## MemoTime: Deep Dive

### What It Actually Does

```
Input: "Who did Obama meet after visiting Berlin in 2015?"

Step 1: Extract topic entities [Obama, Berlin]
Step 2: Extract temporal constraint [after, 2015]
Step 3: Build "Tree of Time":
        Root: main question
        ├── Sub-Q1: When did Obama visit Berlin?
        │   └── Answer: 2015-06-07
        └── Sub-Q2: Who did Obama meet after 2015-06-07?
            └── Answer: [list from TKG]

Step 4: Execute with monotonic time constraints
Step 5: Store successful trace in experience memory
```

### What's Novel
1. **Tree of Time decomposition** — hierarchical, not linear, question decomposition
2. **Monotonic timestamp enforcement** — each hop must not go backwards in time
3. **Experience memory** — stores successful reasoning traces for reuse
4. **Operator-specific toolkits** — different retrieval strategies for before/after/during/first/last

### What's NOT Novel
- Question decomposition (many prior works)
- RAG with knowledge graphs (standard)
- Experience replay (from RL, applied here)

### Applicability to PIE
**High potential.** We could adapt:
- Tree of Time decomposition for complex temporal queries
- Monotonic constraints during multi-hop reasoning
- Experience memory for learning from successful retrievals

**But:** MemoTime assumes clean TKG quadruples (subject, relation, object, timestamp). We have messy conversations. We'd need to:
1. Extract temporal facts from conversations (our extraction pipeline)
2. Build a queryable temporal graph (our world model)
3. Apply MemoTime-style reasoning on top

**This is actually what PIE is building toward.**

---

## Temporal Contrastive Learning: The Missing Piece

Neither paper addresses this, but it's critical for retrieval:

**Problem:** When you ask "What did I think about React last month?", standard embeddings might retrieve:
- Conversations about React from 6 months ago (semantically similar, temporally wrong)
- Conversations from last month about Vue (temporally right, semantically wrong)

**Solution:** Train embeddings where temporal proximity matters:

```python
class TemporalContrastiveLoss:
    def forward(self, embeddings, timestamps):
        # Want: temporally close events have similar embeddings
        sim_matrix = cosine_similarity(embeddings, embeddings)
        time_dist = |timestamps - timestamps.T|
        
        # Target: similarity should correlate with temporal proximity
        target = exp(-time_dist / tau)
        
        return MSE(sim_matrix, target)
```

**This is unexplored territory.** No one has combined semantic + temporal embedding for conversational memory.

---

## What We Should Actually Build

### Phase 1: Enhance PIE's Temporal Retrieval (No Training)

```python
class TemporalAwareRetriever:
    def retrieve(self, query, query_time, top_k=10):
        # 1. Detect temporal references in query
        temporal_refs = extract_temporal_refs(query)
        # "last month" -> (query_time - 30d, query_time)
        
        # 2. Standard semantic retrieval
        semantic_results = self.embed_search(query, top_k * 3)
        
        # 3. Temporal filtering
        if temporal_refs:
            filtered = [r for r in semantic_results 
                       if r.timestamp in temporal_refs.range]
        else:
            filtered = semantic_results
        
        # 4. Temporal re-ranking (decay)
        def score(r):
            sem = r.semantic_score
            temp = exp(-|r.timestamp - query_time| / decay_rate)
            return sem * 0.7 + temp * 0.3
        
        return sorted(filtered, key=score)[:top_k]
```

### Phase 2: MemoTime-Style Tree of Time (No Training)

```python
class TreeOfTime:
    def answer(self, question, world_model):
        # 1. Decompose question
        sub_questions = self.decompose(question)
        # "When did I change my mind about X?" ->
        #   - "What was my original opinion on X?"
        #   - "What is my current opinion on X?"
        #   - "When did the change happen?"
        
        # 2. Execute with monotonic constraints
        results = []
        current_time_bound = None
        
        for sq in sub_questions:
            evidence = self.retrieve_with_constraint(sq, 
                time_after=current_time_bound)
            answer = self.llm_answer(sq, evidence)
            current_time_bound = answer.timestamp  # Monotonic!
            results.append(answer)
        
        # 3. Synthesize final answer
        return self.synthesize(question, results)
```

### Phase 3: Temporal Attention Bias (Minimal Training)

Adapt TPP-TAL's per-head bias for our retrieval attention:

```python
class TemporalBiasAttention(nn.Module):
    """Add to existing retrieval scoring."""
    
    def __init__(self, num_heads=8, num_buckets=32):
        self.time_bucket = nn.Embedding(num_buckets, 32)
        self.time_mlp = nn.Sequential(
            nn.Linear(32, 64),
            nn.SiLU(),
            nn.Linear(64, num_heads)
        )
    
    def forward(self, query_emb, doc_embs, doc_times, query_time):
        # Standard attention scores
        scores = query_emb @ doc_embs.T  # (1, N)
        
        # Temporal bias
        time_diffs = |doc_times - query_time|
        buckets = log_bucketize(time_diffs)
        bias = self.time_mlp(self.time_bucket(buckets))  # (N, H)
        
        # Per-head biased scores
        scores_per_head = scores.unsqueeze(-1) + bias.T  # (1, N, H)
        
        return scores_per_head.mean(dim=-1)  # Aggregate heads
```

### Phase 4: Temporal Contrastive Pre-training (Full Training)

If we have enough data, train embeddings that respect temporal proximity:

```python
def temporal_contrastive_training(conversations):
    for batch in conversations:
        # Positive pairs: temporally adjacent turns
        pos_pairs = [(batch[i], batch[i+1]) for i in range(len(batch)-1)]
        
        # Negative pairs: temporally distant turns
        neg_pairs = [(batch[i], batch[j]) for i, j in 
                     random_distant_pairs(batch)]
        
        # Contrastive loss
        loss = 0
        for (a, b) in pos_pairs:
            loss -= log(sim(embed(a), embed(b)))
        for (a, b) in neg_pairs:
            loss += log(1 - sim(embed(a), embed(b)))
        
        loss.backward()
```

---

## Path to SOTA

### Benchmark: LongMemEval Temporal Subset (133 questions)

Current baselines:
- naive_rag: ~60%
- Emergence AI: ~86% (but not open, unclear method)
- Supermemory: ~71% 

Our path:
1. **Phase 1** (1 week): Temporal-aware retriever → target 65-70%
2. **Phase 2** (2 weeks): Tree of Time decomposition → target 75%
3. **Phase 3** (1 week): Temporal attention bias → target 78-80%
4. **Phase 4** (4 weeks): Contrastive training → target 85%+

### What Would Be Genuinely Novel

1. **Temporal contrastive embeddings for conversational memory** — no one has done this
2. **Semantic temporal compilation + Tree of Time** — combining PIE's approach with MemoTime's
3. **Adaptive temporal granularity** — automatically adjusting temporal resolution based on query type

---

## Concrete Next Steps

1. **Today:** Implement Phase 1 temporal-aware retriever in PIE
2. **This week:** Run ablations on LongMemEval temporal subset
3. **Next week:** Implement Tree of Time decomposition
4. **Week 3-4:** Train temporal attention bias on our conversation data
5. **Month 2:** Temporal contrastive pre-training if data supports it

The goal isn't to copy TPP-TAL or MemoTime — it's to take their best ideas and adapt them to conversational memory, where no one has done this work yet.
