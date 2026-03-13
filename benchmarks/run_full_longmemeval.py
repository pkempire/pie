#!/usr/bin/env python3
"""
Full LongMemEval Benchmark Run

Runs all 500 questions with multiple baselines:
1. naive_rag_turn - turn-level embedding retrieval (k=10)
2. naive_rag_session - session-level embedding retrieval (k=5)
3. full_context - stuff all sessions (for comparison)

Outputs: benchmarks/results/full_longmemeval_{timestamp}/
"""

import json
import os
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from collections import defaultdict

# Load env
env_path = Path(__file__).parent.parent / ".env"
if env_path.exists():
    for line in env_path.read_text().splitlines():
        if "=" in line and not line.startswith("#"):
            k, v = line.split("=", 1)
            os.environ.setdefault(k.strip(), v.strip())

sys.path.insert(0, str(Path(__file__).parent.parent))

from pie.core.llm import LLMClient
from benchmarks.longmemeval.baselines import naive_rag, full_context
from benchmarks.longmemeval.runner import judge_answer

def run_baseline(dataset, llm, baseline_name, baseline_fn, output_dir, **kwargs):
    """Run a baseline on full dataset and save results."""
    results = []
    scores_by_type = defaultdict(lambda: {"count": 0, "score": 0.0})
    total_score = 0
    
    print(f"\n{'='*70}")
    print(f"Running {baseline_name} on {len(dataset)} questions")
    print(f"{'='*70}")
    
    for i, item in enumerate(dataset):
        t0 = time.time()
        
        try:
            result = baseline_fn(item, llm=llm, **kwargs)
            hypothesis = result.hypothesis
            latency = result.latency_ms
        except Exception as e:
            hypothesis = f"Error: {e}"
            latency = (time.time() - t0) * 1000
        
        # Judge
        try:
            score, reason = judge_answer(
                question=item["question"],
                gold_answer=item["answer"],
                hypothesis=hypothesis,
                llm=llm,
                model="gpt-4o"
            )
        except Exception as e:
            score, reason = 0.0, f"Judge error: {e}"
        
        total_score += score
        qtype = item["question_type"]
        scores_by_type[qtype]["count"] += 1
        scores_by_type[qtype]["score"] += score
        
        results.append({
            "question_id": item["question_id"],
            "question_type": qtype,
            "question": item["question"],
            "gold_answer": item["answer"],
            "hypothesis": hypothesis,
            "score": score,
            "reason": reason,
            "latency_ms": latency,
        })
        
        # Progress
        emoji = "✅" if score == 1.0 else "🟡" if score == 0.5 else "❌"
        running_acc = 100 * total_score / (i + 1)
        print(f"{emoji} [{i+1}/{len(dataset)}] {qtype[:15]:15} | {running_acc:5.1f}% | {item['question'][:40]}...")
        
        # Save periodically
        if (i + 1) % 50 == 0:
            _save_results(results, scores_by_type, total_score, baseline_name, output_dir)
    
    # Final save
    _save_results(results, scores_by_type, total_score, baseline_name, output_dir)
    
    # Print summary
    print(f"\n{'='*70}")
    print(f"{baseline_name} FINAL RESULTS")
    print(f"{'='*70}")
    print(f"Overall: {total_score}/{len(dataset)} = {100*total_score/len(dataset):.1f}%")
    for qtype, data in sorted(scores_by_type.items()):
        acc = 100 * data["score"] / data["count"] if data["count"] > 0 else 0
        print(f"  {qtype:30} {acc:5.1f}% ({int(data['score'])}/{data['count']})")
    
    return total_score / len(dataset)

def _save_results(results, scores_by_type, total_score, baseline_name, output_dir):
    """Save intermediate results."""
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Save detailed results
    with open(output_dir / f"{baseline_name}_results.jsonl", "w") as f:
        for r in results:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    
    # Save summary
    summary = {
        "baseline": baseline_name,
        "total": len(results),
        "total_score": total_score,
        "accuracy": 100 * total_score / len(results) if results else 0,
        "by_type": {t: {"accuracy": 100*d["score"]/d["count"] if d["count"] else 0, **d} 
                   for t, d in scores_by_type.items()},
        "timestamp": datetime.now(timezone.utc).isoformat(),
    }
    with open(output_dir / f"{baseline_name}_summary.json", "w") as f:
        json.dump(summary, f, indent=2)

def main():
    # Load data
    data_path = Path("benchmarks/longmemeval/data/longmemeval_s_cleaned.json")
    with open(data_path) as f:
        dataset = json.load(f)
    
    print(f"Loaded {len(dataset)} questions")
    
    # Create output directory
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = Path(f"benchmarks/results/full_longmemeval_{timestamp}")
    
    llm = LLMClient()
    
    # Run baselines
    baselines = [
        ("naive_rag_turn", naive_rag, {"chunk_by": "turn", "top_k": 10}),
        ("naive_rag_session", naive_rag, {"chunk_by": "session", "top_k": 5}),
    ]
    
    all_results = {}
    for name, fn, kwargs in baselines:
        acc = run_baseline(dataset, llm, name, fn, output_dir, **kwargs)
        all_results[name] = acc
    
    # Print comparison
    print(f"\n{'='*70}")
    print("COMPARISON")
    print(f"{'='*70}")
    for name, acc in all_results.items():
        print(f"  {name:30} {100*acc:.1f}%")

if __name__ == "__main__":
    main()
