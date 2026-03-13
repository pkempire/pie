# PIE Benchmark Audit

## Current State: What We Have vs What Papers Use

### LongMemEval (Wu et al., ICLR 2025)

**Paper baselines:**
| Baseline | Description | Status |
|----------|-------------|--------|
| Full Context | Stuff all sessions into prompt | ✅ `full_context` |
| BM25 | Sparse retrieval (turn/session) | ❌ Missing |
| Contriever | Dense retrieval (Facebook model) | ❌ Missing |
| Stella-1.5B | Dense retrieval | ❌ Missing |
| GTE-Qwen2-7B | Dense retrieval | ❌ Missing |
| OpenAI Embeddings | Dense retrieval (text-embedding-3) | ✅ `naive_rag` |
| Key Expansion | LLM-expanded keys | ❌ Missing |
| Time-Aware Query | Temporal search pruning | ❌ Missing |
| **PIE Temporal** | Our approach | ✅ `pie_temporal` |

**Paper official repo:** https://github.com/xiaowu0162/LongMemEval
- Has BM25, Contriever, Stella implementations
- Has session decomposition, key expansion
- Has time-aware query expansion

**Our implementation gaps:**
1. No BM25 baseline (need `rank_bm25` or paper's implementation)
2. No Contriever baseline (need HuggingFace model)
3. No key expansion or time-aware pruning

---

### LoCoMo (Maharana et al., ACL 2024)

**Paper baselines:**
| Baseline | Description | Status |
|----------|-------------|--------|
| BM25 | Sparse retrieval | ❌ Missing |
| DPR | Dense Passage Retrieval | ❌ Missing |
| Contriever | Facebook dense retriever | ❌ Missing |
| MPNet | Sentence transformer | ❌ Missing |
| OpenAI Embeddings | Dense retrieval | ✅ `naive_rag` |
| Full Context | Long-context baseline | ✅ `full_context` |

---

### Test of Time (Fatemi et al., ICLR 2025)

**Paper categories:**
| Category | Description | Status |
|----------|-------------|--------|
| Semantic - before_after | Ordering questions | ✅ |
| Semantic - first_last | First/last in sequence | ✅ |
| Semantic - timeline | Order multiple events | ✅ |
| Arithmetic - duration | Calculate time spans | ✅ |
| Arithmetic - event_at_time_t | Point-in-time lookup | ✅ |
| Arithmetic - simultaneous | Find concurrent events | ✅ |

**Our implementation:** Running both ToT-Semantic and ToT-Arithmetic splits.

---

### MemoryBench (Supermemory)

**Benchmarks covered:**
- LoCoMo ✅ (we have it)
- LongMemEval ✅ (we have it)
- ConvoMem ❌ (not implemented)

**Providers tested:**
- Supermemory, Mem0, Zep

**We should integrate:** Use their framework for cross-system comparison.

---

### MemoryAgentBench (ICLR 2026)

**Key benchmarks:**
| Benchmark | Description | Status |
|----------|-------------|--------|
| Accurate Retrieval (AR) | Fact recall | ❌ |
| Test-Time Learning (TTL) | Learning from context | ❌ |
| Long-Range Understanding (LRU) | Cross-session reasoning | ❌ |
| Conflict Resolution (CR) | Handle contradictions | ❌ |
| EventQA | Temporal event questions | ❌ |
| FactConsolidation | Fact merging | ❌ |

**Paper:** https://arxiv.org/abs/2507.05257
**Repo:** https://github.com/HUST-AI-HYZ/MemoryAgentBench

---

### Other Benchmarks to Consider

| Benchmark | Description | Priority |
|-----------|-------------|----------|
| MSC (Multi-Session Chat) | ParlAI persona consistency | ⬛ Medium |
| HELMET | Long-context eval | ⬛ Medium |
| InfBench | Infinitely long context | ⬜ Low |
| PersistBench | Persistence across sessions | ⬜ Low |
| DMR (Deep Memory Retrieval) | MemGPT benchmark | ⬛ Medium |

---

## PIE Code Quality Assessment

### Strengths
- ✅ Clean architecture (parser → ingestion → resolution → world model)
- ✅ Proper entity typing (EntityType, TransitionType enums)
- ✅ Sliding window context for temporal coherence
- ✅ Web grounding for entity canonicalization
- ✅ Batch embedding computation

### Issues
- ⚠️ No tests (0 test files)
- ⚠️ No type checking setup (mypy/pyright)
- ⚠️ Hardcoded model names in places
- ⚠️ No async/parallel processing
- ⚠️ No proper error recovery/retry logic
- ⚠️ No streaming support for large files

### Missing for Production
- [ ] Unit tests
- [ ] Integration tests
- [ ] Type annotations (partial)
- [ ] Proper logging configuration
- [ ] Rate limiting / retry logic
- [ ] Async API calls
- [ ] Streaming ingestion
- [ ] Checkpoint/resume for crashes
- [ ] Database backend (currently JSON file)

---

## Recommended Fixes

### Priority 1: Add Paper Baselines (for fair comparison)

```python
# benchmarks/baselines/bm25.py
from rank_bm25 import BM25Okapi

def bm25_baseline(item, top_k=10, chunk_by="session"):
    # Tokenize chunks
    # Build BM25 index
    # Retrieve top-k
    # Generate answer
    pass

# benchmarks/baselines/contriever.py
from sentence_transformers import SentenceTransformer

def contriever_baseline(item, top_k=10):
    model = SentenceTransformer('facebook/contriever')
    # Embed and retrieve
    pass
```

### Priority 2: Integrate MemoryBench

```bash
# Clone and use their framework
git clone https://github.com/supermemoryai/memorybench
# Implement PIE as a provider
```

### Priority 3: Add MemoryAgentBench

```bash
git clone https://github.com/HUST-AI-HYZ/MemoryAgentBench
# Run their benchmarks with PIE
```

---

## Revised Test Suite

```bash
# Full comparison run
python run_full_suite.py \
  --baselines naive_rag_turn naive_rag_session bm25 contriever pie_temporal \
  --benchmarks longmemeval locomo tot \
  --skip-extraction  # if world model already built
```
