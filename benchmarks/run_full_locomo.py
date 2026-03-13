#!/usr/bin/env python3
"""
Full LoCoMo Benchmark Run - 1986 QA pairs across 10 conversations
"""

import json
import os
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from collections import defaultdict

env_path = Path(__file__).parent.parent / ".env"
if env_path.exists():
    for line in env_path.read_text().splitlines():
        if "=" in line and not line.startswith("#"):
            k, v = line.split("=", 1)
            os.environ.setdefault(k.strip(), v.strip())

sys.path.insert(0, str(Path(__file__).parent.parent))

from pie.core.llm import LLMClient
from benchmarks.locomo.baselines import naive_rag
from benchmarks.locomo.runner import judge_answer
from benchmarks.locomo.adapter import flatten_qa

# Category mapping from LoCoMo paper
CATEGORY_MAP = {1: "single_hop", 2: "multi_hop", 3: "temporal", 4: "adversarial", 5: "commonsense"}

def run_baseline(items, llm, baseline_name, output_dir, **kwargs):
    results = []
    scores_by_cat = defaultdict(lambda: {"count": 0, "score": 0.0})
    total_score = 0
    
    print(f"\n{'='*70}")
    print(f"Running {baseline_name} on {len(items)} questions")
    print(f"{'='*70}")
    
    for i, item in enumerate(items):
        try:
            result = naive_rag(item, llm=llm, **kwargs)
            hypothesis = result.hypothesis
        except Exception as e:
            hypothesis = f"Error: {e}"
        
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
        cat = CATEGORY_MAP.get(item.get("category", 0), "unknown")
        scores_by_cat[cat]["count"] += 1
        scores_by_cat[cat]["score"] += score
        
        results.append({
            "question_id": item.get("question_id", f"q{i}"),
            "category": cat,
            "question": item["question"],
            "gold_answer": item["answer"],
            "hypothesis": hypothesis,
            "score": score,
        })
        
        emoji = "✅" if score == 1.0 else "🟡" if score == 0.5 else "❌"
        running_acc = 100 * total_score / (i + 1)
        print(f"{emoji} [{i+1}/{len(items)}] {cat[:12]:12} | {running_acc:5.1f}% | {item['question'][:40]}...")
        
        if (i + 1) % 100 == 0:
            _save(results, scores_by_cat, total_score, baseline_name, output_dir)
    
    _save(results, scores_by_cat, total_score, baseline_name, output_dir)
    
    print(f"\n{baseline_name}: {100*total_score/len(items):.1f}%")
    for cat, d in sorted(scores_by_cat.items()):
        print(f"  {cat:15} {100*d['score']/d['count']:.1f}%")
    
    return total_score / len(items)

def _save(results, scores_by_cat, total_score, name, output_dir):
    output_dir.mkdir(parents=True, exist_ok=True)
    with open(output_dir / f"{name}_results.jsonl", "w") as f:
        for r in results:
            f.write(json.dumps(r) + "\n")
    summary = {"total": len(results), "accuracy": 100*total_score/len(results), 
               "by_cat": {c: 100*d["score"]/d["count"] for c,d in scores_by_cat.items()}}
    with open(output_dir / f"{name}_summary.json", "w") as f:
        json.dump(summary, f, indent=2)

def main():
    with open("benchmarks/locomo/data/locomo10.json") as f:
        data = json.load(f)
    
    items = flatten_qa(data)
    print(f"Loaded {len(items)} QA pairs from {len(data)} conversations")
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = Path(f"benchmarks/results/full_locomo_{timestamp}")
    
    llm = LLMClient()
    
    for chunk_by in ["turn", "session"]:
        run_baseline(items, llm, f"naive_rag_{chunk_by}", output_dir, chunk_by=chunk_by)

if __name__ == "__main__":
    main()
