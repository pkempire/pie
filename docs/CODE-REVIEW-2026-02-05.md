# PIE Code Review — 2026-02-05

## Architecture Overview

```
pie/
├── core/
│   ├── models.py      # Data models (Entity, Conversation, Turn, etc.)
│   ├── world_model.py # In-memory graph + JSON persistence
│   ├── parser.py      # ChatGPT JSON export parser  
│   └── llm.py         # OpenAI API wrapper
├── ingestion/
│   ├── pipeline.py    # Main orchestrator
│   └── prompts.py     # Extraction prompts
├── resolution/
│   ├── resolver.py    # 3-tier entity resolution
│   └── web_grounder.py# Brave Search verification
├── eval/              # Ablation/quality eval modules
└── config.py          # Dataclass configs

benchmarks/
├── longmemeval/       # LongMemEval-S benchmark
├── locomo/            # LoCoMo benchmark  
├── msc/               # Multi-session chat
├── test-of-time/      # Test of Time (temporal)
├── tot/               # DUPLICATE - same as test-of-time?
└── eval_harness.py    # Unified runner
```

---

## 🐛 Bugs Found

### 1. Action Name Mismatch (resolver.py → pipeline.py)

**Location**: `resolver.py:_resolve_single()` returns `action="new"`, but `pipeline.py:_apply_to_world_model()` checks for `action="create"`

```python
# resolver.py line ~218
return ResolvedEntity(
    extracted=extracted,
    matched_entity_id=None,
    action="create",   # ← This is correct in the current code
    ...
)
```

**Status**: Actually fixed — I initially thought I saw "new" but it's "create". False alarm.

### 2. No Retry Logic on Embedding Calls

**Location**: `llm.py:embed()` and `embed_single()`

```python
def embed(self, texts: list[str], ...) -> list[list[float]]:
    # No retry — will fail on transient errors
    response = self.client.embeddings.create(...)
```

**Fix**: Add same retry pattern as `chat()`:
```python
for attempt in range(MAX_RETRIES):
    try:
        response = self.client.embeddings.create(...)
        break
    except Exception as e:
        if attempt == MAX_RETRIES - 1:
            raise
        time.sleep(RETRY_DELAY * (attempt + 1))
```

### 3. Cache Key Collision in Web Grounder

**Location**: `web_grounder.py:ground()`

```python
cache_key = f"{entity.name.lower()}:{entity.type}"
```

**Problem**: Entity names can contain colons (e.g., "C++: Advanced" + "tool" → "c++: advanced:tool")

**Fix**: Use a delimiter that's unlikely in names:
```python
cache_key = f"{entity.name.lower()}||{entity.type}"
```

### 4. Silent Batch Processing Failures

**Location**: `pipeline.py:run()`

```python
except Exception as e:
    logger.error(f"  → ERROR processing batch {batch.date}: {e}")
    self._errors.append({...})
    continue  # ← Just continues, doesn't surface the error
```

**Problem**: Partial failures go unnoticed. If 50% of batches fail, you still get "INGESTION COMPLETE".

**Fix**: Add option to fail-fast or at minimum surface error count prominently:
```python
if self._errors:
    logger.error(f"⚠️ COMPLETED WITH {len(self._errors)} ERRORS")
```

---

## 🔧 Code Quality Issues

### 1. Hardcoded Model Names

**Location**: Multiple files

```python
# llm.py
NO_TEMP_MODELS = {"gpt-5-mini", "gpt-5-nano", ...}

# config.py  
extraction_model: str = "gpt-5-mini"

# resolver.py
model="gpt-5-nano"  # hardcoded for binary classification
```

**Problem**: Not everyone has access to these models. Should use config or env vars.

### 2. Magic Thresholds Without Validation

**Location**: `config.py:ResolutionConfig`

```python
string_match_threshold: float = 0.85
embedding_similarity_threshold: float = 0.70
embedding_ambiguous_threshold: float = 0.78
```

**Problem**: These are "feels right" numbers, not empirically validated. The resolution_ablation.py exists but doesn't seem to have been used to tune these.

### 3. Duplicate Benchmark Folders

Both `benchmarks/test-of-time/` and `benchmarks/tot/` exist. Should consolidate.

### 4. Unused/Dead Code Patterns

**Location**: `resolver.py:_precompute_embeddings()`

```python
self._embedding_cache = {}  # Set as instance var
...
# Later accessed via getattr with default:
embedding = getattr(self, '_embedding_cache', {}).get(extracted.name)
```

The `getattr` with default suggests defensive coding against missing attribute, but it's always set. Either trust it exists or initialize in `__init__`.

---

## 📊 Missing Test Coverage

No unit tests found (only `test_extraction.py` which looks like a manual test script, not pytest).

**Should have**:
- `tests/test_parser.py` — ChatGPT format edge cases
- `tests/test_resolver.py` — Resolution tier testing
- `tests/test_world_model.py` — Graph operations
- `tests/test_pipeline.py` — End-to-end integration

---

## 🎯 Benchmark Coverage Analysis

### Currently Implemented
| Benchmark | Type | Status |
|-----------|------|--------|
| LongMemEval-S | Temporal/multi-session QA | ✅ Working |
| LoCoMo | Long-term conversational memory | ✅ Working |
| MSC | Persona consistency | ✅ Working |
| Test of Time | Pure temporal reasoning | ✅ Working |

### Missing (High Priority)
| Benchmark | Paper | Why Important |
|-----------|-------|---------------|
| **MemoryAgentBench** | ICLR 2026 (2507.05257) | 4 core competencies, multi-turn incremental, SOTA comparison |
| **MemoryBench** | 2510.17281 | Continual learning, user feedback simulation |
| **PersistBench** | 2602.01146 | Memory forgetting evaluation |
| **Real-time Temporal** | 2601.13206 | Deadline awareness in dialogues |

---

## 🚀 Recommendations

### Immediate Fixes
1. Add retry logic to embedding calls
2. Fix web grounder cache key collision
3. Surface batch errors prominently
4. Consolidate ToT benchmark folders

### Short-term
1. Add MemoryAgentBench — it's comprehensive and has public code/data
2. Make model names configurable via env vars
3. Add basic pytest coverage

### Medium-term
1. Run resolution ablation to validate thresholds
2. Add MemoryBench for continual learning eval
3. Build unified leaderboard/dashboard for all benchmarks

---

## Summary

**Code Quality**: 7/10 — Clean architecture, good separation of concerns, but lacks tests and has some hardcoded magic numbers.

**Benchmark Coverage**: 6/10 — Good foundation but missing recent SOTA benchmarks (MemoryAgentBench is essential).

**Production Readiness**: 5/10 — No retry on embeddings, silent failures, hardcoded models would break for other users.
