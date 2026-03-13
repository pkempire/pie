# Benchmark Commands

Run these from `~/personal-intelligence-system`:

## Quick Sanity Check (5 samples each)
```bash
cd ~/personal-intelligence-system
python3 -m benchmarks.eval_harness --benchmarks longmemeval locomo msc --baseline naive_rag --subset 5
```

## Full Benchmark Runs

### 1. LongMemEval (500 questions, ~2-3 hours)
```bash
cd ~/personal-intelligence-system
python3 -m benchmarks.eval_harness --benchmarks longmemeval --baseline naive_rag 2>&1 | tee benchmarks/results/longmemeval_$(date +%Y%m%d_%H%M%S).log
```

### 2. LoCoMo (200 questions, ~1 hour)
```bash
cd ~/personal-intelligence-system
python3 -m benchmarks.eval_harness --benchmarks locomo --baseline naive_rag 2>&1 | tee benchmarks/results/locomo_$(date +%Y%m%d_%H%M%S).log
```

### 3. MSC (50 questions default, ~30 min)
```bash
cd ~/personal-intelligence-system
python3 -m benchmarks.eval_harness --benchmarks msc --baseline naive_rag --subset 50 2>&1 | tee benchmarks/results/msc_$(date +%Y%m%d_%H%M%S).log
```

### 4. Test of Time (semantic + arithmetic)
```bash
cd ~/personal-intelligence-system
python3 benchmarks/tot/runner.py --subset 100 2>&1 | tee benchmarks/results/tot_$(date +%Y%m%d_%H%M%S).log
```

### 5. All Baselines Comparison (subset for speed)
```bash
cd ~/personal-intelligence-system
python3 -m benchmarks.eval_harness --benchmarks longmemeval locomo --baseline all --subset 50 2>&1 | tee benchmarks/results/comparison_$(date +%Y%m%d_%H%M%S).log
```

## Results Location
- `benchmarks/results/` — timestamped run folders
- `benchmarks/results/dashboard.html` — HTML comparison dashboard (if generated)

## Notes
- Results auto-save to `benchmarks/results/YYYYMMDD_HHMMSS/`
- Use `--subset N` to limit questions for faster iteration
- Use `--baseline all` to compare naive_rag, pie_temporal, full_context
