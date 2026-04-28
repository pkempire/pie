#!/usr/bin/env python3
"""
LoCoMo Benchmark Runner for PIE

Main entry point for running the LoCoMo benchmark against PIE's temporal
knowledge graph and comparison baselines.

Usage:
    # Run PIE on 5 questions (quick test)
    python -m benchmarks.locomo.runner --baseline pie_temporal --limit 5

    # Run on temporal questions only
    python -m benchmarks.locomo.runner --category temporal --limit 10

    # Run single question with debug
    python -m benchmarks.locomo.runner --question-id conv-26_q0 --debug

    # Compare all baselines
    python -m benchmarks.locomo.runner --baseline all --output results/

LoCoMo key metrics (from paper):
    Long-context LLMs: 22-66% improvement over base
    RAG with observations: Best balanced performance
    Human performance: ~56% better than best models on temporal reasoning
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

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

_env_path = PROJECT_ROOT / ".env"
if _env_path.exists():
    for _line in _env_path.read_text().splitlines():
        if "=" in _line and not _line.startswith("#"):
            _k, _v = _line.split("=", 1)
            os.environ.setdefault(_k.strip(), _v.strip())

from pie.core.llm import LLMClient

from benchmarks.locomo.adapter import (
    load_dataset,
    flatten_qa,
    filter_by_category,
    filter_by_ids,
    dataset_stats,
)
from benchmarks.locomo.baselines import (
    BaselineResult,
    full_context,
    naive_rag,
    pie_temporal_hybrid,
    pie_temporal,
    _build_world_model_for_conversation,
    BASELINES,
)
from benchmarks.formatting import format_qa_result, format_summary_header

logging.basicConfig(
    level=logging.WARNING,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("pie.bench.locomo")


# ── LLM-as-Judge ──────────────────────────────────────────────────────────────

JUDGE_SYSTEM = """\
You are evaluating whether a predicted answer is correct given a reference answer.
Score the prediction:
  1.0 = Fully correct — captures the key information
  0.5 = Partially correct — some relevant info but incomplete
  0.0 = Incorrect — wrong, irrelevant, or "I don't know"

Be generous with phrasing differences — focus on semantic correctness.
For temporal questions, dates/ordering must be correct.

Respond with ONLY JSON: {"score": <0.0|0.5|1.0>, "reason": "<brief>"}"""

JUDGE_USER = """\
Question: {question}
Reference answer: {gold}
Predicted answer: {hypothesis}

Score:"""


def judge_answer(
    question: str,
    gold_answer: str,
    hypothesis: str,
    llm: LLMClient,
    model: str = "gpt-4o",
) -> tuple[float, str]:
    """Use LLM-as-judge to score a predicted answer."""
    hyp_lower = hypothesis.lower().strip()
    if hyp_lower.startswith("error:") or "i don't know" in hyp_lower:
        return 0.0, "No answer provided"

    messages = [
        {"role": "system", "content": JUDGE_SYSTEM},
        {
            "role": "user",
            "content": JUDGE_USER.format(
                question=question,
                gold=gold_answer,
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

    total_tokens_prompt: int = 0
    total_tokens_completion: int = 0

    def add(self, result: BaselineResult, score: float, reason: str):
        self.total += 1
        self.total_score += score
        self.total_tokens_prompt += result.tokens_prompt
        self.total_tokens_completion += result.tokens_completion
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
        print(f"  LoCoMo Results — {baseline_name}")
        print("=" * 70)
        print(f"  Overall Accuracy: {s['overall']['accuracy']}%  "
              f"({int(s['overall']['total_score'])}/{s['overall']['total']})")
        total_tok = self.total_tokens_prompt + self.total_tokens_completion
        if total_tok > 0 and self.total > 0:
            avg_tok = total_tok / self.total
            print(f"  Token Usage (answer-gen): {total_tok:,} total  "
                  f"({self.total_tokens_prompt:,}p + {self.total_tokens_completion:,}c)  "
                  f"avg {avg_tok:.0f}/q")
        print()
        print("  Per-Category Breakdown:")
        print(f"  {'Category':<25} {'Accuracy':>8} {'Count':>6} {'Score':>7}")
        print("  " + "-" * 50)
        for cat, data in sorted(s["by_category"].items()):
            print(
                f"  {cat:<25} {data['accuracy']:>7.1f}% "
                f"{data['count']:>6} {data['score']:>7.1f}"
            )
        print("=" * 70)


# ── Batch Runner ──────────────────────────────────────────────────────────────


def run_benchmark(
    items: list[dict[str, Any]],
    baseline_name: str = "pie_temporal",
    model: str = "gpt-4o",
    extraction_model: str = "gpt-4o-mini",
    judge_model: str = "gpt-4o",
    output_dir: Path | None = None,
    cache_dir: Path | None = None,
    debug: bool = False,
    save_every: int = 10,
) -> BenchmarkScores:
    """Run the full benchmark on a list of QA items.

    For PIE baselines: processes one conversation at a time —
    builds world model → answers all its questions → saves → moves on.
    Never blocks on building all world models before starting Q&A.
    """
    llm = LLMClient()
    scores = BenchmarkScores()
    total = len(items)
    t0 = time.time()

    if output_dir:
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

    print(format_summary_header("LoCoMo", baseline_name, total))

    uses_pie = baseline_name in ("pie_temporal", "pie_temporal_hybrid")

    if uses_pie:
        # Group items by conversation so we can build + answer per conversation
        from collections import OrderedDict
        conv_groups: dict[str, list[dict]] = OrderedDict()
        for item in items:
            sid = item.get("sample_id", item["question_id"].split("_q")[0])
            conv_groups.setdefault(sid, []).append(item)

        n_convos = len(conv_groups)
        print(f"\n📋 {n_convos} conversations, {total} questions total", flush=True)
        print(f"   Strategy: build world model → answer questions → save → next conversation\n", flush=True)

        global_i = 0  # running question index across all conversations
        for conv_i, (sid, conv_items) in enumerate(conv_groups.items(), 1):
            representative_item = conv_items[0]
            conv = representative_item.get("conversation", {})
            n_sessions = sum(1 for k in conv if k.startswith("session_") and not k.endswith("_date_time"))
            n_q = len(conv_items)

            print(f"\n{'─'*70}", flush=True)
            print(f"  Conversation {conv_i}/{n_convos}: '{sid[:20]}' — {n_sessions} sessions, {n_q} questions", flush=True)
            print(f"{'─'*70}", flush=True)

            # Build (or load cached) world model for this conversation
            _t_wm = time.time()
            world_model = None
            wm_cache_path: Path | None = None
            if cache_dir:
                cache_dir = Path(cache_dir)
                cache_dir.mkdir(parents=True, exist_ok=True)
                wm_cache_path = cache_dir / f"{sid}_wm.json"

            from pie.core.world_model import WorldModel as _WM
            if wm_cache_path and wm_cache_path.exists():
                try:
                    world_model = _WM(persist_path=str(wm_cache_path))
                    elapsed = time.time() - _t_wm
                    print(f"  ✅ Loaded cached world model for '{sid}' — {len(world_model.entities)} entities in {elapsed:.1f}s", flush=True)
                except Exception as e:
                    logger.warning(f"Cache load failed for {sid}: {e}, rebuilding...")
                    world_model = None

            if world_model is None:
                print(f"  ⏳ Building world model ({n_sessions} sessions)...", end="", flush=True)
                try:
                    world_model = _build_world_model_for_conversation(
                        representative_item, llm, extraction_model
                    )
                    elapsed = time.time() - _t_wm
                    print(f" ✓ {len(world_model.entities)} entities, {len(world_model.transitions)} transitions in {elapsed:.0f}s", flush=True)
                    if wm_cache_path:
                        world_model.persist_path = wm_cache_path
                        world_model.save()
                except Exception as e:
                    elapsed = time.time() - _t_wm
                    print(f" ✗ FAILED in {elapsed:.0f}s: {e}", flush=True)
                    logger.warning(f"World model build failed for {sid}: {e}")

            # Answer all questions for this conversation immediately
            for item in conv_items:
                global_i += 1
                qid = item["question_id"]
                qtype = item["question_type"]

                try:
                    if baseline_name == "pie_temporal":
                        result = pie_temporal(
                            item,
                            world_model=world_model,
                            llm=llm,
                            model=model,
                            extraction_model=extraction_model,
                        )
                    else:
                        result = pie_temporal_hybrid(
                            item,
                            world_model=world_model,
                            llm=llm,
                            model=model,
                            extraction_model=extraction_model,
                        )

                    score, reason = judge_answer(
                        question=item["question"],
                        gold_answer=item["answer"],
                        hypothesis=result.hypothesis,
                        llm=llm,
                        model=judge_model,
                    )
                    scores.add(result, score, reason)
                    print(format_qa_result(
                        idx=global_i,
                        total=total,
                        question=item["question"],
                        expected=item["answer"],
                        predicted=result.hypothesis,
                        score=score,
                        qtype=qtype,
                        latency_ms=result.latency_ms,
                    ))

                except Exception as e:
                    logger.error(f"❌ {qid}: {e}")
                    scores.add(BaselineResult(
                        question_id=qid, question_type=qtype,
                        question=item["question"], gold_answer=item["answer"],
                        hypothesis=f"Error: {e}", baseline_name=baseline_name,
                        model=model, error=str(e),
                    ), 0.0, f"Error: {e}")

            # Save after every conversation — never lose more than one convo's work
            if output_dir:
                _save_results(scores, output_dir, baseline_name, intermediate=True)
                running_acc = scores.overall_accuracy * 100
                print(f"\n  💾 Saved. Running accuracy: {running_acc:.1f}% ({int(scores.total_score)}/{scores.total})", flush=True)

    else:
        # Non-PIE baselines: simple flat loop
        for i, item in enumerate(items):
            qid = item["question_id"]
            qtype = item["question_type"]

            try:
                if baseline_name == "full_context":
                    result = full_context(item, llm=llm, model=model)
                elif baseline_name == "naive_rag":
                    result = naive_rag(item, llm=llm, model=model)
                else:
                    raise ValueError(f"Unknown baseline: {baseline_name}")

                score, reason = judge_answer(
                    question=item["question"],
                    gold_answer=item["answer"],
                    hypothesis=result.hypothesis,
                    llm=llm,
                    model=judge_model,
                )
                scores.add(result, score, reason)
                print(format_qa_result(
                    idx=i + 1, total=total,
                    question=item["question"], expected=item["answer"],
                    predicted=result.hypothesis, score=score,
                    qtype=qtype, latency_ms=result.latency_ms,
                ))

            except Exception as e:
                logger.error(f"❌ Error: {e}")
                scores.add(BaselineResult(
                    question_id=qid, question_type=qtype,
                    question=item["question"], gold_answer=item["answer"],
                    hypothesis=f"Error: {e}", baseline_name=baseline_name,
                    model=model, error=str(e),
                ), 0.0, f"Error: {e}")

            if output_dir and (i + 1) % save_every == 0:
                _save_results(scores, output_dir, baseline_name, intermediate=True)

    total_time = time.time() - t0
    print(f"\nCompleted in {total_time:.0f}s ({total_time/60:.1f}m)", flush=True)

    scores.print_report(baseline_name)

    if output_dir:
        _save_results(scores, output_dir, baseline_name)

    return scores


def _save_results(
    scores: BenchmarkScores,
    output_dir: Path,
    baseline_name: str,
    intermediate: bool = False,
):
    """Save results to files."""
    suffix = "_intermediate" if intermediate else ""

    # JSONL results
    results_path = output_dir / f"{baseline_name}{suffix}_results.jsonl"
    with open(results_path, "w") as f:
        for r in scores.results:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")

    if not intermediate:
        # Summary JSON
        summary_path = output_dir / f"{baseline_name}_summary.json"
        summary = scores.summary()
        summary["baseline"] = baseline_name
        summary["benchmark"] = "locomo"
        summary["timestamp"] = datetime.now(timezone.utc).isoformat()
        with open(summary_path, "w") as f:
            json.dump(summary, f, indent=2)

        logger.info(f"Results saved to {output_dir}/")


# ── CLI ───────────────────────────────────────────────────────────────────────


def main():
    parser = argparse.ArgumentParser(
        description="LoCoMo Benchmark Runner for PIE",
    )

    parser.add_argument(
        "--dataset",
        type=Path,
        default=PROJECT_ROOT / "benchmarks/locomo/data/locomo10.json",
        help="Path to LoCoMo dataset JSON",
    )
    parser.add_argument("--limit", "-n", type=int, default=None)
    parser.add_argument(
        "--category", "-c",
        type=str,
        default=None,
        choices=["single_hop", "multi_hop", "temporal", "adversarial", "commonsense"],
    )
    parser.add_argument("--question-id", "-q", type=str, default=None)
    parser.add_argument(
        "--baseline", "-b",
        type=str,
        default="pie_temporal",
        choices=["full_context", "naive_rag", "pie_temporal", "pie_temporal_hybrid", "all"],
    )
    parser.add_argument("--model", "-m", type=str, default="gpt-4o")
    parser.add_argument("--extraction-model", type=str, default="gpt-4o-mini")
    parser.add_argument("--judge-model", type=str, default="gpt-4o")
    parser.add_argument("--output", "-o", type=Path, default=None)
    parser.add_argument(
        "--cache-dir",
        type=Path,
        default=PROJECT_ROOT / "benchmarks/locomo/cache",
        help="Directory to save/load world model cache (default: benchmarks/locomo/cache). Pass 'none' to disable.",
    )
    parser.add_argument("--debug", "-d", action="store_true")
    parser.add_argument("--stats", action="store_true")

    args = parser.parse_args()

    # Normalize --cache-dir
    if hasattr(args, "cache_dir") and str(args.cache_dir).lower() == "none":
        args.cache_dir = None

    # Load dataset
    logger.info(f"Loading dataset from {args.dataset}")
    dataset = load_dataset(args.dataset)
    logger.info(f"Loaded {len(dataset)} conversations")

    if args.stats:
        stats = dataset_stats(dataset)
        print(json.dumps(stats, indent=2))
        return

    # Flatten to QA items
    items = flatten_qa(dataset)
    logger.info(f"Flattened to {len(items)} QA items")

    # Apply filters
    if args.question_id:
        items = filter_by_ids(items, [args.question_id])
        args.debug = True

    if args.category:
        items = filter_by_category(items, args.category)
        logger.info(f"Filtered to {len(items)} {args.category} questions")

    if args.limit:
        items = items[:args.limit]
        logger.info(f"Limited to {args.limit} questions")

    if not items:
        logger.error("No questions to process")
        sys.exit(1)

    # Set default output
    if args.output is None:
        args.output = (
            PROJECT_ROOT / "benchmarks" / "locomo" / "results"
            / datetime.now().strftime("%Y%m%d_%H%M%S")
        )

    # Run
    if args.baseline == "all":
        for baseline in ["full_context", "naive_rag", "pie_temporal", "pie_temporal_hybrid"]:
            logger.info(f"\n{'='*70}\n  Running: {baseline}\n{'='*70}")
            run_benchmark(
                items=items,
                baseline_name=baseline,
                model=args.model,
                extraction_model=args.extraction_model,
                judge_model=args.judge_model,
                output_dir=args.output,
                cache_dir=args.cache_dir,
                debug=args.debug,
            )
    else:
        run_benchmark(
            items=items,
            baseline_name=args.baseline,
            model=args.model,
            extraction_model=args.extraction_model,
            judge_model=args.judge_model,
            output_dir=args.output,
            cache_dir=args.cache_dir,
            debug=args.debug,
        )


if __name__ == "__main__":
    main()
