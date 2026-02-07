#!/usr/bin/env python3
"""
MemoryAgentBench Benchmark Runner for PIE

Main entry point for running the MemoryAgentBench benchmark.

Usage:
    # Run PIE on 5 questions (quick test)
    python -m benchmarks.memoryagentbench.runner --baseline pie_temporal --limit 5

    # Run naive RAG on Accurate Retrieval competency
    python -m benchmarks.memoryagentbench.runner --baseline naive_rag --competency AR

    # Run all baselines on all competencies  
    python -m benchmarks.memoryagentbench.runner --baseline all

    # Run with specific model
    python -m benchmarks.memoryagentbench.runner --baseline long_context --model gpt-4o

Paper Baselines (Table 3):
    Long Context (GPT-4o-mini):  ~50-60% average across competencies
    Naive RAG (text-embedding-3-small): ~40-55%
    Target: Beat naive RAG, approach long-context performance with less context
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import sys
import time
from collections import defaultdict
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any

# Add project root to path
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from pie.core.llm import LLMClient

from benchmarks.memoryagentbench.adapter import (
    load_dataset,
    dataset_stats,
    filter_by_competency,
    BenchmarkItem,
)
from benchmarks.memoryagentbench.baselines import (
    BaselineResult,
    long_context,
    naive_rag,
    pie_temporal,
    NaiveRAGRetriever,
    PIEWorldModelBuilder,
    BASELINES,
)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("pie.bench.memoryagentbench")


# ── LLM-as-Judge Evaluation ──────────────────────────────────────────────────

JUDGE_SYSTEM_PROMPT = """\
You are evaluating whether a predicted answer is correct given a reference answer.
Score the prediction on a scale:
  1.0 = Fully correct — captures the key information from the reference
  0.5 = Partially correct — some relevant info but incomplete or slightly wrong
  0.0 = Incorrect — wrong, irrelevant, or "I don't know"

Be generous with phrasing differences — focus on semantic correctness.
The reference may be a list; if so, the prediction should match any item in the list.
For exact matches (names, numbers), require close alignment.

Respond with ONLY a JSON object: {"score": <0.0|0.5|1.0>, "reason": "<brief explanation>"}"""

JUDGE_USER_TEMPLATE = """\
Question: {question}
Reference answer: {gold_answer}
Predicted answer: {hypothesis}

Score (0.0, 0.5, or 1.0):"""


def judge_answer(
    question: str,
    gold_answer: Any,
    hypothesis: str,
    llm: LLMClient,
    model: str = "gpt-4o-mini",
) -> tuple[float, str]:
    """Use LLM-as-judge to score a predicted answer."""
    # Quick check for obvious failures
    hyp_lower = hypothesis.lower().strip()
    if hyp_lower.startswith("error:") or hyp_lower in ["i don't know.", "i don't know"]:
        return 0.0, "No answer provided"

    # Format gold answer
    gold_str = gold_answer if isinstance(gold_answer, str) else str(gold_answer)
    if isinstance(gold_answer, list) and len(gold_answer) > 0:
        if isinstance(gold_answer[0], str):
            gold_str = " | ".join(gold_answer)  # Show alternatives

    messages = [
        {"role": "system", "content": JUDGE_SYSTEM_PROMPT},
        {
            "role": "user",
            "content": JUDGE_USER_TEMPLATE.format(
                question=question,
                gold_answer=gold_str,
                hypothesis=hypothesis,
            ),
        },
    ]

    try:
        result = llm.chat(messages=messages, model=model, json_mode=True)
        parsed = result["content"]
        if isinstance(parsed, str):
            parsed = json.loads(parsed)
        
        score = float(parsed.get("score", 0.0))
        reason = parsed.get("reason", "")
        
        # Clamp to valid values
        if score >= 0.75:
            score = 1.0
        elif score >= 0.25:
            score = 0.5
        else:
            score = 0.0
        
        return score, reason

    except Exception as e:
        logger.warning(f"Judge failed: {e}")
        return 0.0, f"Judge error: {e}"


# ── Score Aggregation ─────────────────────────────────────────────────────────


@dataclass
class BenchmarkScores:
    """Aggregated scores across questions."""
    total: int = 0
    correct: int = 0
    partial: int = 0
    incorrect: int = 0
    total_score: float = 0.0
    by_competency: dict = field(default_factory=lambda: defaultdict(lambda: {"total": 0, "score": 0.0}))
    by_source: dict = field(default_factory=lambda: defaultdict(lambda: {"total": 0, "score": 0.0}))

    def add(self, result: BaselineResult, score: float):
        self.total += 1
        self.total_score += score
        
        if score >= 0.9:
            self.correct += 1
        elif score >= 0.4:
            self.partial += 1
        else:
            self.incorrect += 1
            
        self.by_competency[result.competency]["total"] += 1
        self.by_competency[result.competency]["score"] += score
        
        self.by_source[result.source]["total"] += 1
        self.by_source[result.source]["score"] += score

    @property
    def accuracy(self) -> float:
        return (self.total_score / self.total * 100) if self.total > 0 else 0.0

    def accuracy_by_competency(self) -> dict[str, float]:
        return {
            k: (v["score"] / v["total"] * 100) if v["total"] > 0 else 0.0
            for k, v in self.by_competency.items()
        }

    def to_dict(self) -> dict:
        return {
            "total": self.total,
            "correct": self.correct,
            "partial": self.partial,
            "incorrect": self.incorrect,
            "accuracy": round(self.accuracy, 2),
            "by_competency": {
                k: round((v["score"] / v["total"] * 100) if v["total"] > 0 else 0.0, 2)
                for k, v in self.by_competency.items()
            },
        }


# ── Main Runner ───────────────────────────────────────────────────────────────


def run_benchmark(
    baseline: str,
    competency: str | None = None,
    model: str = "gpt-4o-mini",
    limit: int | None = None,
    output_dir: Path | None = None,
    skip_judge: bool = False,
    top_k: int = 10,
    chunk_size: int = 4096,
) -> dict:
    """
    Run benchmark evaluation.
    
    Args:
        baseline: Baseline to run (long_context, naive_rag, pie_temporal, all)
        competency: Filter to specific competency (AR, TTL, LRU, CR) or None for all
        model: LLM model to use
        limit: Maximum number of questions to evaluate
        output_dir: Directory to save results
        skip_judge: Skip LLM judge and just collect predictions
        top_k: Number of chunks to retrieve (for RAG/PIE)
        chunk_size: Chunk size for RAG
        
    Returns:
        Dictionary with results and scores
    """
    llm = LLMClient()
    
    # Load dataset
    logger.info("Loading MemoryAgentBench dataset...")
    items = load_dataset(competency)
    stats = dataset_stats(items)
    logger.info(f"Dataset: {stats['total_items']} items, {stats['total_questions']} total questions")
    logger.info(f"By competency: {stats['by_competency']}")
    
    # Determine which baselines to run
    baselines_to_run = list(BASELINES.keys()) if baseline == "all" else [baseline]
    
    all_results = {}
    
    for baseline_name in baselines_to_run:
        logger.info(f"\n{'='*60}")
        logger.info(f"Running baseline: {baseline_name}")
        logger.info(f"{'='*60}")
        
        results: list[BaselineResult] = []
        scores = BenchmarkScores()
        question_count = 0
        
        for item_idx, item in enumerate(items):
            logger.info(f"\nItem {item_idx + 1}/{len(items)} [{item.competency}] {item.source}")
            logger.info(f"  Context: {len(item.context):,} chars, {item.num_questions} questions")
            
            # Build retriever/builder once per item (shared across questions)
            retriever = None
            builder = None
            
            if baseline_name == "naive_rag":
                retriever = NaiveRAGRetriever(llm, chunk_size=chunk_size)
                t0 = time.time()
                retriever.index_context(item.context)
                logger.info(f"  Indexed in {time.time()-t0:.1f}s ({len(retriever.chunks)} chunks)")
                
            elif baseline_name == "pie_temporal":
                builder = PIEWorldModelBuilder(llm, model)
                t0 = time.time()
                try:
                    builder.build_from_context(item.context)
                    logger.info(f"  Built world model in {time.time()-t0:.1f}s ({len(builder.world_model.entities)} entities)")
                except Exception as e:
                    logger.error(f"  Failed to build world model: {e}")
                    builder = None
            
            # Process each question
            for q_idx, question, answer, qa_pair_id in item.iter_qa_pairs():
                if limit and question_count >= limit:
                    break
                
                # Run baseline
                if baseline_name == "long_context":
                    result = long_context(item, q_idx, llm=llm, model=model)
                elif baseline_name == "naive_rag":
                    result = naive_rag(item, q_idx, retriever=retriever, llm=llm, model=model, top_k=top_k)
                elif baseline_name == "pie_temporal":
                    result = pie_temporal(item, q_idx, builder=builder, llm=llm, model=model, top_k=top_k)
                else:
                    raise ValueError(f"Unknown baseline: {baseline_name}")
                
                results.append(result)
                
                # Judge answer
                if not skip_judge:
                    score, reason = judge_answer(
                        result.question,
                        result.gold_answer,
                        result.hypothesis,
                        llm,
                        model,
                    )
                    scores.add(result, score)
                    
                    # Log progress
                    logger.info(f"  Q{q_idx+1}: score={score} | {result.hypothesis[:50]}...")
                
                question_count += 1
                
            if limit and question_count >= limit:
                logger.info(f"\nReached limit of {limit} questions")
                break
        
        # Store results for this baseline
        all_results[baseline_name] = {
            "results": [r.to_dict() for r in results],
            "scores": scores.to_dict() if not skip_judge else None,
            "config": {
                "model": model,
                "top_k": top_k,
                "chunk_size": chunk_size,
                "competency": competency,
                "limit": limit,
            },
        }
        
        # Print summary
        if not skip_judge:
            logger.info(f"\n{baseline_name} Results:")
            logger.info(f"  Overall: {scores.accuracy:.1f}% ({scores.correct}/{scores.total} correct)")
            for comp, acc in scores.accuracy_by_competency().items():
                logger.info(f"  {comp}: {acc:.1f}%")
    
    # Save results
    if output_dir:
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_path = output_dir / f"memoryagentbench_{baseline}_{timestamp}.json"
        
        with open(output_path, "w") as f:
            json.dump(all_results, f, indent=2)
        logger.info(f"\nResults saved to {output_path}")
    
    return all_results


def main():
    parser = argparse.ArgumentParser(
        description="Run MemoryAgentBench benchmark",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--baseline", "-b",
        type=str,
        default="naive_rag",
        choices=["long_context", "naive_rag", "pie_temporal", "all"],
        help="Baseline to run (default: naive_rag)",
    )
    parser.add_argument(
        "--competency", "-c",
        type=str,
        default=None,
        choices=["AR", "TTL", "LRU", "CR"],
        help="Filter to specific competency",
    )
    parser.add_argument(
        "--model", "-m",
        type=str,
        default="gpt-4o-mini",
        help="LLM model to use (default: gpt-4o-mini)",
    )
    parser.add_argument(
        "--limit", "-l",
        type=int,
        default=None,
        help="Maximum number of questions to evaluate",
    )
    parser.add_argument(
        "--output", "-o",
        type=str,
        default="benchmarks/memoryagentbench/results",
        help="Output directory for results",
    )
    parser.add_argument(
        "--top-k", "-k",
        type=int,
        default=10,
        help="Number of chunks to retrieve (default: 10)",
    )
    parser.add_argument(
        "--chunk-size",
        type=int,
        default=4096,
        help="Chunk size for RAG (default: 4096)",
    )
    parser.add_argument(
        "--skip-judge",
        action="store_true",
        help="Skip LLM judge (just collect predictions)",
    )
    parser.add_argument(
        "--debug",
        action="store_true",
        help="Enable debug logging",
    )
    
    args = parser.parse_args()
    
    if args.debug:
        logging.getLogger().setLevel(logging.DEBUG)
    
    results = run_benchmark(
        baseline=args.baseline,
        competency=args.competency,
        model=args.model,
        limit=args.limit,
        output_dir=Path(args.output),
        skip_judge=args.skip_judge,
        top_k=args.top_k,
        chunk_size=args.chunk_size,
    )
    
    # Print final summary
    print("\n" + "="*60)
    print("BENCHMARK COMPLETE")
    print("="*60)
    for baseline_name, data in results.items():
        if data["scores"]:
            print(f"\n{baseline_name}:")
            print(f"  Overall Accuracy: {data['scores']['accuracy']:.1f}%")
            for comp, acc in data['scores']['by_competency'].items():
                print(f"  {comp}: {acc:.1f}%")


if __name__ == "__main__":
    main()
