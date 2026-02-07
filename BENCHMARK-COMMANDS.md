# Benchmark Commands

Run these in terminal with `OPENAI_API_KEY` set.

## Prerequisites
```bash
export OPENAI_API_KEY="your-key-here"
cd ~/.openclaw/workspace/personal-intelligence-system
```

---

## 1. MemoryAgentBench (ICLR 2026)
**Paper:** arxiv:2507.05257
**Model:** gpt-4o-mini (as specified in paper)
**Competencies:** AR (Accurate Retrieval), TTL (Test-Time Learning), LRU (Long-Range Understanding), CR (Conflict Resolution)

```bash
# Setup (first time only)
cd benchmarks/memoryagentbench/repo
pip install -r requirements.txt

# Create .env
echo "OPENAI_API_KEY=$OPENAI_API_KEY" > .env
echo "LLM_MODEL=gpt-4o-mini" >> .env

# Run long-context baseline
python main.py \
  --agent_config configs/agent_conf/Long_Context_Agents/gpt-4o-mini/long_context.yaml \
  --dataset_config configs/data_conf/nfcats.yaml

# Run RAG baseline  
python main.py \
  --agent_config configs/agent_conf/RAG_Agents/gpt-4o-mini/naive_rag.yaml \
  --dataset_config configs/data_conf/nfcats.yaml
```

---

## 2. Test of Time (ToT)
**Paper:** Temporal reasoning benchmark
**Model:** gpt-4o-mini

```bash
cd benchmarks/tot

# Edit samples per type if needed (default 10)
# SAMPLES_PER_TYPE = 2  # for quick test

python runner.py
```

---

## 3. PersistBench
**Paper:** arxiv:2602.01146
**Tests:** Cross-domain leakage, Sycophancy

```bash
cd benchmarks/persistbench/repo

# Install
pip install -e .

# Run baseline
uv run benchmark run configs/baseline_gpt4o.json

# Or use our PIE runner
cd ..
python pie_runner.py --samples 50 --model gpt-4o-mini
```

---

## 4. Temporal Awareness (Negotiation)
**Paper:** arxiv:2601.13206
**Tests:** Time pressure handling in negotiations

```bash
cd benchmarks/temporal_awareness

# Quick mock test
python run_benchmark.py --mock --quick

# Real run (20 trials per condition)
python run_benchmark.py --model gpt-4o-mini --trials 20
```

---

## 5. LongMemEval (existing)
**Our baseline results:** 66.3%

```bash
cd benchmarks/longmemeval
python baselines.py --method naive_rag --limit 100
```

---

## 6. LoCoMo (existing)
**Our baseline results:** 58%

```bash
cd benchmarks/locomo
python runner.py --limit 100
```

---

## Notes
- All benchmarks use **gpt-4o-mini** as specified in their papers
- Some use **gpt-4o** as judge for evaluation
- Results will vary based on API response consistency
- For PIE comparison, run the `pie_temporal` baseline variant where available
