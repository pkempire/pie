# PIE Evaluation Commands

## Quick Reference

### Local Testing (5-10 questions)
```bash
cd personal-intelligence-system

# Test single approach
python3 modal_eval.py --benchmark tot --approach naive_rag --n 5

# Test all approaches on ToT
for approach in naive_rag pie_temporal graph_aware hybrid; do
  python3 modal_eval.py --benchmark tot --approach $approach --n 10
done
```

### Modal Full Runs

```bash
# Setup (once)
pip install modal
modal setup  # Authenticate with Modal

# Create OpenAI secret in Modal dashboard:
#   modal.com → Settings → Secrets → Create → "openai-api-key"
#   Add OPENAI_API_KEY=sk-...

# Single benchmark/approach (50 questions)
modal run modal_eval.py --benchmark tot --approach naive_rag --n 50

# Full benchmark suite (all approaches, 100 questions each)
modal run modal_eval.py --all --n 100

# Full run (all questions)
modal run modal_eval.py --all
```

### Existing Benchmark Runners

```bash
# LongMemEval (uses benchmarks/longmemeval/)
python3 -m benchmarks.longmemeval.runner \
  --world-model output/world_model.json \
  --method naive_rag \
  --n 100

# Test of Time (uses benchmarks/tot/)
python3 benchmarks/tot/baselines_runner.py \
  --n 80 \
  --methods baseline,naive_rag,pie_temporal

# LoCoMo
python3 -m benchmarks.locomo.runner \
  --world-model output/world_model.json \
  --n 50

# MSC
python3 -m benchmarks.msc.runner \
  --world-model output/world_model.json \
  --n 50
```

---

## Approaches

| Approach | Description | Status |
|----------|-------------|--------|
| `naive_rag` | Embedding similarity over entities | ✅ Working |
| `pie_temporal` | Semantic temporal context compilation | ✅ Working |
| `graph_aware` | LLM intent parsing + graph traversal | ✅ Working (new) |
| `hybrid` | Embedding + graph combined | ✅ Working (new) |

---

## Benchmarks

| Benchmark | Questions | Focus | Data Path |
|-----------|-----------|-------|-----------|
| `longmemeval` | 500 | Multi-session reasoning | `benchmarks/longmemeval/data/` |
| `tot` | 800 | Temporal arithmetic | `benchmarks/tot/tot_semantic.json` |
| `locomo` | 1986 | Long conversation | `benchmarks/locomo/data/` |
| `msc` | ~500 | Persona consistency | `benchmarks/msc/data/` |

---

## Results

Results are saved to:
```
benchmarks/results/YYYYMMDD_HHMMSS/
  ├── longmemeval_naive_rag.json
  ├── longmemeval_pie_temporal.json
  ├── tot_naive_rag.json
  └── ...
```

Each result file contains:
```json
{
  "benchmark": "tot",
  "approach": "naive_rag",
  "accuracy": 0.56,
  "total": 80,
  "correct": 45,
  "by_type": {...},
  "elapsed_seconds": 120.5,
  "timestamp": "2026-02-07T13:22:56"
}
```

---

## Expected Results (from prior runs)

| Benchmark | naive_rag | pie_temporal | Notes |
|-----------|-----------|--------------|-------|
| LongMemEval | **66.3%** | TBD | 500 questions |
| Test of Time | **56.2%** | 31.2% | PIE hurts date arithmetic |
| LoCoMo | 58% | TBD | Temporal subset weak |
| MSC | 46% | TBD | 76% partial credit |

---

## Running on Modal GPU (faster embedding)

```python
# In modal_eval.py, change image to include GPU:
image = modal.Image.debian_slim(python_version="3.11").pip_install(
    "openai>=1.0.0",
    "numpy",
    "sentence-transformers",  # for local embeddings
)

@app.function(
    image=image,
    gpu="T4",  # Add GPU
    mounts=[project_mount],
    secrets=[modal.Secret.from_name("openai-api-key")],
    timeout=3600,
)
def modal_run_eval(...):
    ...
```

---

## Debugging

```bash
# Check world model
python3 -c "
import json
with open('output/world_model.json') as f:
    d = json.load(f)
print(f'Entities: {len(d[\"entities\"])}')
print(f'Transitions: {len(d[\"transitions\"])}')
"

# Test retriever directly
python3 -c "
from pie.retrieval.graph_retriever import retrieve_subgraph
from pie.core.world_model import WorldModel
from pie.core.llm import LLMClient

wm = WorldModel('output/world_model.json')
llm = LLMClient()
result = retrieve_subgraph('What projects have I worked on?', wm, llm)
print(f'Retrieved {len(result.entity_ids)} entities')
for eid in result.entity_ids[:5]:
    e = wm.get_entity(eid)
    print(f'  - {e.name}')
"

# Compare retrievers
python3 -m pie.retrieval.compare_retrievers "How has my tech stack evolved?"
```

---

## Cost Estimates

| Benchmark | Questions | Est. Cost (gpt-4o-mini) |
|-----------|-----------|------------------------|
| LongMemEval | 500 | ~$2 |
| ToT | 800 | ~$3 |
| LoCoMo | 1986 | ~$8 |
| MSC | 500 | ~$2 |
| **Full suite** | 3786 | ~$15 |

With all 4 approaches: ~$60 total.
