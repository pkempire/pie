#!/usr/bin/env python3
"""
LongMemEval Benchmark Runner for PIE

Main entry point for running the LongMemEval benchmark against PIE's temporal
knowledge graph and comparison baselines.

Usage:
    # Run PIE on 5 questions (quick test)
    python -m benchmarks.longmemeval.runner --baseline pie_temporal --limit 5

    # Run all baselines on temporal-reasoning questions
    python -m benchmarks.longmemeval.runner --category temporal-reasoning

    # Run single question with debug output
    python -m benchmarks.longmemeval.runner --question-id e47becba --debug

    # Compare all baselines on full dataset
    python -m benchmarks.longmemeval.runner --baseline all --output results/

    # Resume from cached world models
    python -m benchmarks.longmemeval.runner --baseline pie_temporal --cache-dir cache/

SOTA scores to beat:
    Emergence AI:    86% overall
    Supermemory:     71.4% (temporal-reasoning: 76.7%)
    Zep:             71.2%
    Full context:    60-64%
    Our target:      beat Zep (71.2%) overall, beat all on temporal-reasoning
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
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

# Add project root to path
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from pie.core.llm import LLMClient
from pie.core.world_model import WorldModel

from benchmarks.longmemeval.adapter import (
    load_dataset,
    load_oracle_dataset,
    filter_by_category,
    filter_by_ids,
    dataset_stats,
    parse_question_date,
    format_date_for_context,
)
from benchmarks.longmemeval.baselines import (
    BaselineResult,
    full_context,
    naive_rag,
    pie_temporal,
    pie_temporal_cached,
    PIETemporalCachedBaseline,
    BASELINES,
)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("pie.bench.longmemeval")


# ── LLM-as-Judge Evaluation ──────────────────────────────────────────────────

JUDGE_SYSTEM_PROMPT = """\
You are evaluating whether a predicted answer is correct given a reference answer.
Score the prediction on a scale:
  1.0 = Fully correct — captures the key information from the reference
  0.5 = Partially correct — some relevant info but incomplete or slightly wrong
  0.0 = Incorrect — wrong, irrelevant, or "I don't know"

Be generous with phrasing differences — focus on semantic correctness.
"Business Administration" and "a degree in Business Admin" are both correct.
But "Computer Science" when the answer is "Business Administration" is wrong.

For questions about preferences or opinions, the specific stated preference must match.
For temporal questions, the ordering/timing must be correct.
For knowledge-update questions, the MOST RECENT value must be given.

Respond with ONLY a JSON object: {"score": <0.0|0.5|1.0>, "reason": "<brief explanation>"}"""

JUDGE_USER_TEMPLATE = """\
Question: {question}
Reference answer: {gold_answer}
Predicted answer: {hypothesis}

Score (0.0, 0.5, or 1.0):"""


def judge_answer(
    question: str,
    gold_answer: str,
    hypothesis: str,
    llm: LLMClient,
    model: str = "gpt-4o",
) -> tuple[float, str]:
    """
    Use LLM-as-judge to score a predicted answer.
    
    Returns (score, reason) where score is 0.0, 0.5, or 1.0.
    Compatible with LongMemEval's official GPT-4o judge format.
    """
    # Quick check: if hypothesis is an error or "I don't know"
    hyp_lower = hypothesis.lower().strip()
    if hyp_lower.startswith("error:") or hyp_lower == "i don't know." or hyp_lower == "i don't know":
        return 0.0, "No answer provided"

    messages = [
        {"role": "system", "content": JUDGE_SYSTEM_PROMPT},
        {
            "role": "user",
            "content": JUDGE_USER_TEMPLATE.format(
                question=question,
                gold_answer=gold_answer,
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


# ── World Model Caching ──────────────────────────────────────────────────────


def _cache_path(cache_dir: Path, question_id: str) -> Path:
    """Get the cache file path for a question's world model."""
    return cache_dir / f"{question_id}_world_model.json"


def save_world_model_cache(
    world_model: WorldModel,
    cache_dir: Path,
    question_id: str,
) -> None:
    """Save a built world model to disk for reuse."""
    cache_dir.mkdir(parents=True, exist_ok=True)
    path = _cache_path(cache_dir, question_id)
    world_model.persist_path = path
    world_model.save()
    logger.debug(f"  Cached world model to {path}")


def load_world_model_cache(
    cache_dir: Path,
    question_id: str,
) -> WorldModel | None:
    """Load a cached world model if it exists."""
    path = _cache_path(cache_dir, question_id)
    if path.exists():
        wm = WorldModel(persist_path=path)
        if wm.entities:
            logger.info(f"  Loaded cached world model ({len(wm.entities)} entities)")
            return wm
    return None


# ── Score Aggregation ─────────────────────────────────────────────────────────


@dataclass
class BenchmarkScores:
    """Aggregated benchmark scores."""
    total: int = 0
    total_score: float = 0.0
    by_category: dict[str, dict] = field(default_factory=lambda: defaultdict(
        lambda: {"count": 0, "score": 0.0, "scores": []}
    ))
    results: list[dict] = field(default_factory=list)

    @property
    def overall_accuracy(self) -> float:
        return self.total_score / self.total if self.total > 0 else 0.0

    def add(self, result: BaselineResult, score: float, reason: str):
        self.total += 1
        self.total_score += score
        cat = result.question_type
        self.by_category[cat]["count"] += 1
        self.by_category[cat]["score"] += score
        self.by_category[cat]["scores"].append(score)
        self.results.append({
            **result.to_dict(),
            "judge_score": score,
            "judge_reason": reason,
        })

    def summary(self) -> dict:
        summary = {
            "overall": {
                "accuracy": round(self.overall_accuracy * 100, 1),
                "total": self.total,
                "total_score": round(self.total_score, 1),
            },
            "by_category": {},
        }
        for cat, data in sorted(self.by_category.items()):
            count = data["count"]
            score = data["score"]
            accuracy = score / count if count > 0 else 0.0
            summary["by_category"][cat] = {
                "accuracy": round(accuracy * 100, 1),
                "count": count,
                "score": round(score, 1),
            }
        return summary

    def print_report(self, baseline_name: str):
        """Print a formatted benchmark report."""
        s = self.summary()
        print("\n" + "=" * 70)
        print(f"  LongMemEval Results — {baseline_name}")
        print("=" * 70)
        print(f"  Overall Accuracy: {s['overall']['accuracy']}%  "
              f"({int(s['overall']['total_score'])}/{s['overall']['total']})")
        print()

        # SOTA comparison
        print("  Comparison to SOTA:")
        sota = [
            ("Emergence AI", 86.0),
            ("Supermemory", 71.4),
            ("Zep", 71.2),
            ("Full context GPT-4o", 62.0),
        ]
        our = s["overall"]["accuracy"]
        for name, acc in sota:
            delta = our - acc
            marker = "✅" if delta >= 0 else "  "
            print(f"    {marker} {name}: {acc}% (delta: {delta:+.1f}%)")

        print()
        print("  Per-Category Breakdown:")
        print(f"  {'Category':<30} {'Accuracy':>8} {'Count':>6} {'Score':>7}")
        print("  " + "-" * 55)
        for cat, data in sorted(s["by_category"].items()):
            print(
                f"  {cat:<30} {data['accuracy']:>7.1f}% "
                f"{data['count']:>6} {data['score']:>7.1f}"
            )
        print("=" * 70)


# ── Single-Question Runner ────────────────────────────────────────────────────


def run_single_question(
    item: dict[str, Any],
    baseline_name: str = "pie_temporal",
    llm: LLMClient | None = None,
    judge_llm: LLMClient | None = None,
    model: str = "gpt-4o",
    extraction_model: str = "gpt-4o-mini",
    judge_model: str = "gpt-4o",
    cache_dir: Path | None = None,
    debug: bool = False,
) -> tuple[BaselineResult, float, str]:
    """
    Run a single question through a baseline and judge the result.
    
    Returns (result, score, reason).
    """
    llm = llm or LLMClient()
    judge_llm = judge_llm or llm

    qid = item["question_id"]
    qtype = item["question_type"]

    if debug:
        print(f"\n{'─'*60}")
        print(f"  Question: {item['question']}")
        print(f"  Type: {qtype}")
        print(f"  Gold: {item['answer']}")
        print(f"  Date: {item['question_date']}")
        print(f"{'─'*60}")

    # Check cache for PIE baseline
    cached_wm = None
    if baseline_name == "pie_temporal" and cache_dir:
        cached_wm = load_world_model_cache(cache_dir, qid)

    # Run the baseline
    if baseline_name == "full_context":
        result = full_context(item, llm=llm, model=model)
    elif baseline_name == "naive_rag":
        result = naive_rag(item, llm=llm, model=model)
    elif baseline_name == "naive_rag_session":
        result = naive_rag(item, llm=llm, model=model, chunk_by="session")
    elif baseline_name == "pie_temporal":
        result = pie_temporal(
            item,
            world_model=cached_wm,
            llm=llm,
            model=model,
            extraction_model=extraction_model,
        )
    elif baseline_name == "pie_temporal_cached":
        result = pie_temporal_cached(
            item,
            cache_dir=cache_dir,
            llm=llm,
            model=model,
            extraction_model=extraction_model,
        )
    else:
        raise ValueError(f"Unknown baseline: {baseline_name}")

    # Cache the world model if PIE and we built a new one
    if baseline_name == "pie_temporal" and cache_dir and cached_wm is None:
        # The world model was built internally — we need to rebuild to cache
        # (since pie_temporal doesn't expose it). For efficiency, we mark this
        # for the next run.
        pass

    if debug:
        print(f"  Hypothesis: {result.hypothesis}")
        print(f"  Latency: {result.latency_ms:.0f}ms")
        if result.context_chars:
            print(f"  Context: {result.context_chars} chars")
        if result.retrieval_count:
            print(f"  Retrieved: {result.retrieval_count} items")

    # Judge the answer
    score, reason = judge_answer(
        question=item["question"],
        gold_answer=item["answer"],
        hypothesis=result.hypothesis,
        llm=judge_llm,
        model=judge_model,
    )

    if debug:
        emoji = "✅" if score == 1.0 else "🟡" if score == 0.5 else "❌"
        print(f"  Score: {score} {emoji}")
        print(f"  Reason: {reason}")

    return result, score, reason


# ── Batch Runner ──────────────────────────────────────────────────────────────


def run_benchmark(
    dataset: list[dict[str, Any]],
    baseline_name: str = "pie_temporal",
    model: str = "gpt-4o",
    extraction_model: str = "gpt-4o-mini",
    judge_model: str = "gpt-4o",
    cache_dir: Path | None = None,
    output_dir: Path | None = None,
    debug: bool = False,
    save_every: int = 10,
) -> BenchmarkScores:
    """
    Run the full benchmark on a dataset.
    
    Args:
        dataset: List of question items.
        baseline_name: Which baseline to run.
        model: LLM model for answering.
        extraction_model: LLM model for PIE extraction (cheaper).
        judge_model: LLM model for judging answers.
        cache_dir: Directory for caching world models.
        output_dir: Directory for saving results.
        debug: Print detailed output per question.
        save_every: Save intermediate results every N questions.
    
    Returns:
        BenchmarkScores with per-category breakdown.
    """
    llm = LLMClient()
    judge_llm = llm  # reuse same client

    scores = BenchmarkScores()
    total = len(dataset)
    t0 = time.time()

    # Prepare output directory
    if output_dir:
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

    logger.info(f"Running LongMemEval benchmark: {baseline_name}")
    logger.info(f"  Questions: {total}")
    logger.info(f"  Model: {model}")
    if baseline_name == "pie_temporal":
        logger.info(f"  Extraction model: {extraction_model}")
    logger.info(f"  Judge model: {judge_model}")
    if cache_dir:
        logger.info(f"  Cache dir: {cache_dir}")

    for i, item in enumerate(dataset):
        qid = item["question_id"]
        qtype = item["question_type"]

        logger.info(
            f"[{i+1}/{total}] {qid} ({qtype}): "
            f"{item['question'][:60]}..."
        )

        try:
            result, score, reason = run_single_question(
                item=item,
                baseline_name=baseline_name,
                llm=llm,
                judge_llm=judge_llm,
                model=model,
                extraction_model=extraction_model,
                judge_model=judge_model,
                cache_dir=cache_dir,
                debug=debug,
            )
            scores.add(result, score, reason)

            emoji = "✅" if score == 1.0 else "🟡" if score == 0.5 else "❌"
            running_acc = scores.overall_accuracy * 100
            logger.info(
                f"  {emoji} {score} | Running: {running_acc:.1f}% "
                f"({int(scores.total_score)}/{scores.total}) | "
                f"{result.latency_ms:.0f}ms"
            )

        except Exception as e:
            logger.error(f"  ❌ Error: {e}")
            # Record error as 0 score
            error_result = BaselineResult(
                question_id=qid,
                question_type=qtype,
                question=item["question"],
                gold_answer=item["answer"],
                hypothesis=f"Error: {e}",
                baseline_name=baseline_name,
                model=model,
                error=str(e),
            )
            scores.add(error_result, 0.0, f"Error: {e}")

        # Periodic save
        if output_dir and (i + 1) % save_every == 0:
            _save_intermediate(scores, output_dir, baseline_name)

    # Final timing
    total_time = time.time() - t0
    logger.info(
        f"\nCompleted in {total_time:.0f}s ({total_time/60:.1f}m) — "
        f"avg {total_time/max(total,1):.1f}s/question"
    )

    # Print report
    scores.print_report(baseline_name)

    # Save final results
    if output_dir:
        _save_results(scores, output_dir, baseline_name)

    return scores


def _save_intermediate(
    scores: BenchmarkScores,
    output_dir: Path,
    baseline_name: str,
):
    """Save intermediate results."""
    path = output_dir / f"{baseline_name}_intermediate.jsonl"
    with open(path, "w") as f:
        for r in scores.results:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    logger.debug(f"  Saved intermediate results to {path}")


def _save_results(
    scores: BenchmarkScores,
    output_dir: Path,
    baseline_name: str,
):
    """Save final results and summary."""
    # JSONL results (LongMemEval compatible format)
    results_path = output_dir / f"{baseline_name}_results.jsonl"
    with open(results_path, "w") as f:
        for r in scores.results:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")

    # Summary JSON
    summary_path = output_dir / f"{baseline_name}_summary.json"
    summary = scores.summary()
    summary["baseline"] = baseline_name
    summary["timestamp"] = datetime.now(timezone.utc).isoformat()
    summary["num_results"] = len(scores.results)
    with open(summary_path, "w") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)

    # LongMemEval-compatible hypothesis file (for their eval scripts)
    hypothesis_path = output_dir / f"{baseline_name}_hypothesis.jsonl"
    with open(hypothesis_path, "w") as f:
        for r in scores.results:
            f.write(json.dumps({
                "question_id": r["question_id"],
                "hypothesis": r["hypothesis"],
            }, ensure_ascii=False) + "\n")

    logger.info(f"Results saved to {output_dir}/")
    logger.info(f"  {results_path.name} — full results with scores")
    logger.info(f"  {summary_path.name} — aggregate summary")
    logger.info(f"  {hypothesis_path.name} — LongMemEval-compatible format")


# ── PIE Temporal Cached (New Optimized Runner) ────────────────────────────────


def run_pie_cached(
    dataset: list[dict[str, Any]],
    cache_dir: Path,
    model: str = "gpt-4o",
    extraction_model: str = "gpt-4o-mini",
    judge_model: str = "gpt-4o",
    output_dir: Path | None = None,
    debug: bool = False,
    save_every: int = 10,
) -> BenchmarkScores:
    """
    Run PIE temporal cached baseline — optimized for benchmark runs.
    
    Key difference from run_pie_with_caching:
    - Uses PIETemporalCachedBaseline class that caches embeddings
    - Single baseline instance reused across all questions
    - 10-20x faster after initial world model build
    """
    llm = LLMClient()
    cache_dir = Path(cache_dir)
    cache_dir.mkdir(parents=True, exist_ok=True)
    
    scores = BenchmarkScores()
    total = len(dataset)
    t0 = time.time()
    
    if output_dir:
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
    
    logger.info(f"Running PIE temporal cached (optimized)")
    logger.info(f"  Questions: {total}")
    logger.info(f"  Cache dir: {cache_dir}")
    logger.info(f"  Model: {model}, Extraction: {extraction_model}")
    
    # Create single baseline instance for all questions
    baseline = PIETemporalCachedBaseline(
        cache_dir=cache_dir,
        llm=llm,
        model=model,
        extraction_model=extraction_model,
    )
    
    # Count cached world models
    cached_count = sum(
        1 for item in dataset
        if (cache_dir / f"{item['question_id']}_world_model.json").exists()
    )
    logger.info(f"  Cached world models: {cached_count}/{total}")
    
    for i, item in enumerate(dataset):
        qid = item["question_id"]
        qtype = item["question_type"]
        
        logger.info(
            f"[{i+1}/{total}] {qid} ({qtype}): "
            f"{item['question'][:60]}..."
        )
        
        try:
            # Run using cached baseline
            result = baseline.run(item)
            
            # Judge
            score, reason = judge_answer(
                question=item["question"],
                gold_answer=item["answer"],
                hypothesis=result.hypothesis,
                llm=llm,
                model=judge_model,
            )
            
            scores.add(result, score, reason)
            
            emoji = "✅" if score == 1.0 else "🟡" if score == 0.5 else "❌"
            running_acc = scores.overall_accuracy * 100
            logger.info(
                f"  {emoji} {score} | Running: {running_acc:.1f}% "
                f"({int(scores.total_score)}/{scores.total}) | "
                f"{result.latency_ms:.0f}ms"
            )
            
            if debug:
                print(f"  Q: {item['question']}")
                print(f"  Gold: {item['answer']}")
                print(f"  Pred: {result.hypothesis}")
                print(f"  Reason: {reason}")
        
        except Exception as e:
            logger.error(f"  ❌ Error: {e}")
            error_result = BaselineResult(
                question_id=qid,
                question_type=qtype,
                question=item["question"],
                gold_answer=item["answer"],
                hypothesis=f"Error: {e}",
                baseline_name="pie_temporal_cached",
                model=model,
                error=str(e),
            )
            scores.add(error_result, 0.0, f"Error: {e}")
        
        # Periodic save
        if output_dir and (i + 1) % save_every == 0:
            _save_intermediate(scores, output_dir, "pie_temporal_cached")
    
    total_time = time.time() - t0
    logger.info(
        f"\nCompleted in {total_time:.0f}s ({total_time/60:.1f}m) — "
        f"avg {total_time/max(total,1):.1f}s/question"
    )
    
    # Print cache stats
    baseline.print_stats()
    
    scores.print_report("pie_temporal_cached")
    
    if output_dir:
        _save_results(scores, output_dir, "pie_temporal_cached")
    
    return scores


# ── PIE Temporal with Caching (Original) ──────────────────────────────────────


def run_pie_with_caching(
    dataset: list[dict[str, Any]],
    cache_dir: Path,
    model: str = "gpt-4o",
    extraction_model: str = "gpt-4o-mini",
    judge_model: str = "gpt-4o",
    output_dir: Path | None = None,
    debug: bool = False,
    save_every: int = 10,
) -> BenchmarkScores:
    """
    Run PIE temporal baseline with world model caching.
    
    Two-phase approach:
      Phase 1: Build world models for all questions (expensive, cached)
      Phase 2: Query and evaluate (cheap, uses cached models)
    
    This is much more efficient for iterating on the query/context compilation
    without re-running extraction every time.
    """
    llm = LLMClient()
    cache_dir = Path(cache_dir)
    cache_dir.mkdir(parents=True, exist_ok=True)

    scores = BenchmarkScores()
    total = len(dataset)
    t0 = time.time()

    if output_dir:
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

    logger.info(f"Running PIE temporal with caching")
    logger.info(f"  Questions: {total}")
    logger.info(f"  Cache dir: {cache_dir}")
    logger.info(f"  Model: {model}, Extraction: {extraction_model}")

    # Count cached vs needs-building
    cached_count = sum(
        1 for item in dataset
        if _cache_path(cache_dir, item["question_id"]).exists()
    )
    logger.info(f"  Cached: {cached_count}/{total} world models")

    for i, item in enumerate(dataset):
        qid = item["question_id"]
        qtype = item["question_type"]

        logger.info(
            f"[{i+1}/{total}] {qid} ({qtype}): "
            f"{item['question'][:60]}..."
        )

        try:
            # Phase 1: Get or build world model
            wm = load_world_model_cache(cache_dir, qid)
            if wm is None:
                logger.info(f"  Building world model ({len(item['haystack_sessions'])} sessions)...")
                t_build = time.time()
                from benchmarks.longmemeval.baselines import _build_world_model_for_question
                wm = _build_world_model_for_question(item, llm, extraction_model)
                build_time = time.time() - t_build
                logger.info(
                    f"  Built in {build_time:.1f}s: "
                    f"{len(wm.entities)} entities, "
                    f"{len(wm.transitions)} transitions"
                )
                save_world_model_cache(wm, cache_dir, qid)

            # Phase 2: Query and evaluate
            result = pie_temporal(
                item,
                world_model=wm,
                llm=llm,
                model=model,
            )

            # Judge
            score, reason = judge_answer(
                question=item["question"],
                gold_answer=item["answer"],
                hypothesis=result.hypothesis,
                llm=llm,
                model=judge_model,
            )

            scores.add(result, score, reason)

            emoji = "✅" if score == 1.0 else "🟡" if score == 0.5 else "❌"
            running_acc = scores.overall_accuracy * 100
            logger.info(
                f"  {emoji} {score} | Running: {running_acc:.1f}% "
                f"({int(scores.total_score)}/{scores.total}) | "
                f"{result.latency_ms:.0f}ms"
            )

            if debug:
                print(f"  Q: {item['question']}")
                print(f"  Gold: {item['answer']}")
                print(f"  Pred: {result.hypothesis}")
                print(f"  Reason: {reason}")

        except Exception as e:
            logger.error(f"  ❌ Error: {e}")
            error_result = BaselineResult(
                question_id=qid,
                question_type=qtype,
                question=item["question"],
                gold_answer=item["answer"],
                hypothesis=f"Error: {e}",
                baseline_name="pie_temporal",
                model=model,
                error=str(e),
            )
            scores.add(error_result, 0.0, f"Error: {e}")

        # Periodic save
        if output_dir and (i + 1) % save_every == 0:
            _save_intermediate(scores, output_dir, "pie_temporal_cached")

    total_time = time.time() - t0
    logger.info(
        f"\nCompleted in {total_time:.0f}s ({total_time/60:.1f}m)"
    )

    scores.print_report("pie_temporal (cached)")

    if output_dir:
        _save_results(scores, output_dir, "pie_temporal_cached")

    return scores


# ── Compare All Baselines ────────────────────────────────────────────────────


def compare_baselines(
    dataset: list[dict[str, Any]],
    baselines: list[str] | None = None,
    model: str = "gpt-4o",
    extraction_model: str = "gpt-4o-mini",
    judge_model: str = "gpt-4o",
    cache_dir: Path | None = None,
    output_dir: Path | None = None,
    debug: bool = False,
) -> dict[str, BenchmarkScores]:
    """
    Run and compare multiple baselines on the same dataset.
    
    Returns dict of baseline_name → BenchmarkScores.
    """
    baselines = baselines or ["full_context", "naive_rag", "pie_temporal"]
    all_scores = {}

    for baseline_name in baselines:
        logger.info(f"\n{'='*70}")
        logger.info(f"  Running baseline: {baseline_name}")
        logger.info(f"{'='*70}\n")

        if baseline_name == "pie_temporal" and cache_dir:
            scores = run_pie_with_caching(
                dataset=dataset,
                cache_dir=cache_dir,
                model=model,
                extraction_model=extraction_model,
                judge_model=judge_model,
                output_dir=output_dir,
                debug=debug,
            )
        else:
            scores = run_benchmark(
                dataset=dataset,
                baseline_name=baseline_name,
                model=model,
                extraction_model=extraction_model,
                judge_model=judge_model,
                output_dir=output_dir,
                debug=debug,
            )

        all_scores[baseline_name] = scores

    # Print comparison table
    if len(all_scores) > 1:
        _print_comparison(all_scores)

    return all_scores


def _print_comparison(all_scores: dict[str, BenchmarkScores]):
    """Print a comparison table across baselines."""
    print("\n" + "=" * 80)
    print("  BASELINE COMPARISON")
    print("=" * 80)

    # Get all categories
    all_cats = set()
    for scores in all_scores.values():
        all_cats.update(scores.by_category.keys())
    cats_sorted = sorted(all_cats)

    # Header
    header = f"{'Category':<28}"
    for name in all_scores:
        header += f" {name:>14}"
    print(header)
    print("-" * (28 + 15 * len(all_scores)))

    # Overall
    row = f"{'OVERALL':<28}"
    for name, scores in all_scores.items():
        acc = scores.overall_accuracy * 100
        row += f" {acc:>13.1f}%"
    print(row)
    print("-" * (28 + 15 * len(all_scores)))

    # Per-category
    for cat in cats_sorted:
        row = f"{cat:<28}"
        for name, scores in all_scores.items():
            data = scores.by_category.get(cat)
            if data and data["count"] > 0:
                acc = (data["score"] / data["count"]) * 100
                row += f" {acc:>13.1f}%"
            else:
                row += f" {'N/A':>14}"
        print(row)

    print("=" * 80)


# ── CLI Entry Point ───────────────────────────────────────────────────────────


def main():
    parser = argparse.ArgumentParser(
        description="LongMemEval Benchmark Runner for PIE",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
SOTA scores to beat:
  Emergence AI:    86% overall
  Supermemory:     71.4% (temporal-reasoning: 76.7%)
  Zep:             71.2%
  Full context:    60-64%

Examples:
  # Quick test on 5 questions
  python -m benchmarks.longmemeval.runner --baseline pie_temporal --limit 5

  # Run on temporal-reasoning only
  python -m benchmarks.longmemeval.runner --category temporal-reasoning --limit 10

  # Compare baselines
  python -m benchmarks.longmemeval.runner --baseline all --limit 20 --output results/

  # Debug single question
  python -m benchmarks.longmemeval.runner --question-id e47becba --debug
        """,
    )

    # Dataset options
    parser.add_argument(
        "--dataset",
        type=Path,
        default=PROJECT_ROOT / "benchmarks/longmemeval/data/longmemeval_s_cleaned.json",
        help="Path to LongMemEval dataset JSON",
    )
    parser.add_argument(
        "--oracle",
        action="store_true",
        help="Use oracle dataset (minimal sessions, only evidence)",
    )
    parser.add_argument(
        "--limit", "-n",
        type=int,
        default=None,
        help="Limit to first N questions",
    )
    parser.add_argument(
        "--category", "-c",
        type=str,
        default=None,
        choices=[
            "single-session-user",
            "single-session-assistant",
            "single-session-preference",
            "multi-session",
            "knowledge-update",
            "temporal-reasoning",
        ],
        help="Filter to specific question category",
    )
    parser.add_argument(
        "--question-id", "-q",
        type=str,
        default=None,
        help="Run single question by ID",
    )

    # Baseline options
    parser.add_argument(
        "--baseline", "-b",
        type=str,
        default="pie_temporal",
        choices=["full_context", "naive_rag", "naive_rag_session", "pie_temporal", "pie_temporal_cached", "all"],
        help="Which baseline to run (default: pie_temporal)",
    )

    # Model options
    parser.add_argument(
        "--model", "-m",
        type=str,
        default="gpt-4o",
        help="LLM model for answering (default: gpt-4o)",
    )
    parser.add_argument(
        "--extraction-model",
        type=str,
        default="gpt-4o-mini",
        help="LLM model for PIE extraction (default: gpt-4o-mini)",
    )
    parser.add_argument(
        "--judge-model",
        type=str,
        default="gpt-4o",
        help="LLM model for judging (default: gpt-4o)",
    )

    # Output options
    parser.add_argument(
        "--output", "-o",
        type=Path,
        default=None,
        help="Output directory for results",
    )
    parser.add_argument(
        "--cache-dir",
        type=Path,
        default=None,
        help="Directory for caching PIE world models",
    )

    # Misc
    parser.add_argument(
        "--debug", "-d",
        action="store_true",
        help="Print detailed debug output per question",
    )
    parser.add_argument(
        "--stats",
        action="store_true",
        help="Print dataset statistics and exit",
    )
    parser.add_argument(
        "--verbose", "-v",
        action="store_true",
        help="Enable verbose logging",
    )

    args = parser.parse_args()

    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)

    # Load dataset
    if args.oracle:
        oracle_path = args.dataset.parent / "longmemeval_oracle.json"
        logger.info(f"Loading oracle dataset from {oracle_path}")
        dataset = load_oracle_dataset(oracle_path)
    else:
        logger.info(f"Loading dataset from {args.dataset}")
        dataset = load_dataset(args.dataset)

    logger.info(f"Loaded {len(dataset)} questions")

    # Print stats and exit if requested
    if args.stats:
        stats = dataset_stats(dataset)
        print(json.dumps(stats, indent=2))
        return

    # Apply filters
    if args.question_id:
        dataset = filter_by_ids(dataset, [args.question_id])
        if not dataset:
            logger.error(f"Question ID {args.question_id} not found")
            sys.exit(1)
        args.debug = True  # auto-enable debug for single question

    if args.category:
        dataset = filter_by_category(dataset, args.category)
        logger.info(f"Filtered to {len(dataset)} {args.category} questions")

    if args.limit:
        dataset = dataset[:args.limit]
        logger.info(f"Limited to {args.limit} questions")

    if not dataset:
        logger.error("No questions to process after filtering")
        sys.exit(1)

    # Set default output dir if not specified
    if args.output is None:
        args.output = (
            PROJECT_ROOT / "benchmarks" / "longmemeval" / "results"
            / datetime.now().strftime("%Y%m%d_%H%M%S")
        )

    # Set default cache dir for PIE
    if args.cache_dir is None and args.baseline in ("pie_temporal", "pie_temporal_cached", "all"):
        args.cache_dir = (
            PROJECT_ROOT / "benchmarks" / "longmemeval" / "cache"
        )

    # Run
    if args.baseline == "all":
        compare_baselines(
            dataset=dataset,
            model=args.model,
            extraction_model=args.extraction_model,
            judge_model=args.judge_model,
            cache_dir=args.cache_dir,
            output_dir=args.output,
            debug=args.debug,
        )
    elif args.baseline == "pie_temporal_cached":
        # Use optimized cached runner
        run_pie_cached(
            dataset=dataset,
            cache_dir=args.cache_dir,
            model=args.model,
            extraction_model=args.extraction_model,
            judge_model=args.judge_model,
            output_dir=args.output,
            debug=args.debug,
        )
    elif args.baseline == "pie_temporal" and args.cache_dir:
        run_pie_with_caching(
            dataset=dataset,
            cache_dir=args.cache_dir,
            model=args.model,
            extraction_model=args.extraction_model,
            judge_model=args.judge_model,
            output_dir=args.output,
            debug=args.debug,
        )
    else:
        run_benchmark(
            dataset=dataset,
            baseline_name=args.baseline,
            model=args.model,
            extraction_model=args.extraction_model,
            judge_model=args.judge_model,
            cache_dir=args.cache_dir,
            output_dir=args.output,
            debug=args.debug,
        )


if __name__ == "__main__":
    main()
