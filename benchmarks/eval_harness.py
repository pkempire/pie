#!/usr/bin/env python3
"""
Unified Evaluation Harness for PIE Benchmarks

Run all benchmarks with consistent configuration and output unified results.

Usage:
    # Run all benchmarks with PIE
    python -m benchmarks.eval_harness --baseline pie_temporal

    # Quick sanity check (5 samples per benchmark)
    python -m benchmarks.eval_harness --baseline pie_temporal --subset 5

    # Compare all baselines
    python -m benchmarks.eval_harness --baseline all --subset 10

    # Run specific benchmarks
    python -m benchmarks.eval_harness --benchmarks longmemeval locomo --subset 10

Supported benchmarks:
    - longmemeval: Long-term memory with haystack (LongMemEval-S)
    - locomo: Very long-term conversational memory (LoCoMo-10)
    - msc: Multi-session chat (MSC)

Baselines:
    - full_context: Stuff all context into prompt
    - naive_rag: Embed + retrieve top-k chunks
    - pie_temporal: PIE's temporal knowledge graph
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
import time
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("pie.bench.harness")


# ── Benchmark Registry ────────────────────────────────────────────────────────

BENCHMARKS = {
    "longmemeval": {
        "name": "LongMemEval",
        "description": "Long-term memory with temporal reasoning",
        "module": "benchmarks.longmemeval",
        "default_dataset": "benchmarks/longmemeval/data/longmemeval_s_cleaned.json",
        "question_types": [
            "single-session-user",
            "single-session-assistant",
            "single-session-preference",
            "multi-session",
            "knowledge-update",
            "temporal-reasoning",
        ],
    },
    "locomo": {
        "name": "LoCoMo",
        "description": "Very long-term conversational memory (300+ turns)",
        "module": "benchmarks.locomo",
        "default_dataset": "benchmarks/locomo/data/locomo10.json",
        "question_types": [
            "single_hop",
            "multi_hop",
            "temporal",
            "adversarial",
            "commonsense",
        ],
    },
    "msc": {
        "name": "MSC",
        "description": "Multi-session chat with persona consistency",
        "module": "benchmarks.msc",
        "default_dataset": "benchmarks/msc/data/msc_valid.jsonl",
        "question_types": [
            "response_generation",
            "memory_qa",
        ],
    },
}

BASELINES = ["full_context", "naive_rag", "pie_temporal"]


# ── Unified Results ───────────────────────────────────────────────────────────


@dataclass
class UnifiedResults:
    """Unified results across all benchmarks."""
    results: dict[str, dict[str, Any]] = field(default_factory=dict)
    start_time: float = field(default_factory=time.time)
    config: dict = field(default_factory=dict)

    def add_benchmark_result(
        self,
        benchmark: str,
        baseline: str,
        summary: dict,
        raw_results: list[dict] | None = None,
    ):
        """Add results from a single benchmark run."""
        if benchmark not in self.results:
            self.results[benchmark] = {}
        self.results[benchmark][baseline] = {
            "summary": summary,
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "raw_count": len(raw_results) if raw_results else 0,
        }

    def get_comparison_table(self) -> dict:
        """Generate a comparison table across benchmarks and baselines."""
        table = {
            "benchmarks": {},
            "baselines": {},
            "matrix": {},  # benchmark -> baseline -> score
        }

        all_baselines = set()
        for benchmark, baseline_results in self.results.items():
            table["benchmarks"][benchmark] = BENCHMARKS.get(benchmark, {}).get("name", benchmark)
            table["matrix"][benchmark] = {}
            for baseline, data in baseline_results.items():
                all_baselines.add(baseline)
                score = data["summary"].get("overall", {}).get("accuracy", 0)
                if score == 0:
                    score = data["summary"].get("overall", {}).get("score", 0)
                table["matrix"][benchmark][baseline] = score

        for baseline in sorted(all_baselines):
            table["baselines"][baseline] = baseline

        return table

    def print_report(self):
        """Print a formatted comparison report."""
        table = self.get_comparison_table()

        print("\n" + "=" * 80)
        print("  PIE BENCHMARK SUITE — UNIFIED RESULTS")
        print("=" * 80)
        print()

        # Header
        baselines = sorted(table["baselines"].keys())
        header = f"{'Benchmark':<20}"
        for baseline in baselines:
            header += f" {baseline:>15}"
        print(header)
        print("-" * (20 + 16 * len(baselines)))

        # Rows
        for benchmark, scores in sorted(table["matrix"].items()):
            name = table["benchmarks"].get(benchmark, benchmark)
            row = f"{name:<20}"
            for baseline in baselines:
                score = scores.get(baseline, "-")
                if isinstance(score, (int, float)):
                    row += f" {score:>14.1f}%"
                else:
                    row += f" {str(score):>15}"
            print(row)

        print("-" * (20 + 16 * len(baselines)))

        # Averages
        avg_row = f"{'AVERAGE':<20}"
        for baseline in baselines:
            scores = [
                table["matrix"][b].get(baseline, 0)
                for b in table["matrix"]
                if isinstance(table["matrix"][b].get(baseline), (int, float))
            ]
            if scores:
                avg = sum(scores) / len(scores)
                avg_row += f" {avg:>14.1f}%"
            else:
                avg_row += f" {'-':>15}"
        print(avg_row)

        print("=" * 80)

        # Win/Lose analysis
        if len(baselines) > 1 and "pie_temporal" in baselines:
            print("\n  PIE Temporal vs Baselines:")
            for benchmark, scores in sorted(table["matrix"].items()):
                pie_score = scores.get("pie_temporal", 0)
                for baseline in baselines:
                    if baseline == "pie_temporal":
                        continue
                    other_score = scores.get(baseline, 0)
                    if isinstance(pie_score, (int, float)) and isinstance(other_score, (int, float)):
                        delta = pie_score - other_score
                        emoji = "✅" if delta > 0 else "❌" if delta < 0 else "➖"
                        print(f"    {emoji} {benchmark} vs {baseline}: {delta:+.1f}%")
            print()

    def save(self, output_dir: Path):
        """Save unified results to files."""
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        # Full results JSON
        results_path = output_dir / "unified_results.json"
        with open(results_path, "w") as f:
            json.dump({
                "config": self.config,
                "results": self.results,
                "comparison": self.get_comparison_table(),
                "timestamp": datetime.now(timezone.utc).isoformat(),
                "total_time_seconds": time.time() - self.start_time,
            }, f, indent=2)

        # Comparison table (simple CSV for easy analysis)
        table = self.get_comparison_table()
        csv_path = output_dir / "comparison.csv"
        with open(csv_path, "w") as f:
            baselines = sorted(table["baselines"].keys())
            f.write("benchmark," + ",".join(baselines) + "\n")
            for benchmark, scores in sorted(table["matrix"].items()):
                row = [benchmark]
                for baseline in baselines:
                    score = scores.get(baseline, "")
                    row.append(str(score) if score else "")
                f.write(",".join(row) + "\n")

        logger.info(f"Results saved to {output_dir}/")
        logger.info(f"  {results_path.name} — full results")
        logger.info(f"  {csv_path.name} — comparison table")


# ── Benchmark Runners ─────────────────────────────────────────────────────────


def run_longmemeval(
    baseline: str,
    limit: int | None,
    model: str,
    extraction_model: str,
    judge_model: str,
    output_dir: Path,
    debug: bool,
) -> tuple[dict, list[dict]]:
    """Run LongMemEval benchmark."""
    from benchmarks.longmemeval.adapter import load_dataset
    from benchmarks.longmemeval.runner import run_benchmark

    dataset_path = PROJECT_ROOT / "benchmarks/longmemeval/data/longmemeval_s_cleaned.json"
    if not dataset_path.exists():
        logger.warning(f"LongMemEval dataset not found at {dataset_path}")
        return {"error": "Dataset not found"}, []

    dataset = load_dataset(dataset_path)
    if limit:
        dataset = dataset[:limit]

    scores = run_benchmark(
        dataset=dataset,
        baseline_name=baseline,
        model=model,
        extraction_model=extraction_model,
        judge_model=judge_model,
        output_dir=output_dir / "longmemeval",
        debug=debug,
    )

    return scores.summary(), scores.results


def run_locomo(
    baseline: str,
    limit: int | None,
    model: str,
    extraction_model: str,
    judge_model: str,
    output_dir: Path,
    debug: bool,
) -> tuple[dict, list[dict]]:
    """Run LoCoMo benchmark."""
    from benchmarks.locomo.adapter import load_dataset, flatten_qa
    from benchmarks.locomo.runner import run_benchmark

    dataset_path = PROJECT_ROOT / "benchmarks/locomo/data/locomo10.json"
    if not dataset_path.exists():
        logger.warning(f"LoCoMo dataset not found at {dataset_path}")
        return {"error": "Dataset not found"}, []

    dataset = load_dataset(dataset_path)
    items = flatten_qa(dataset)
    if limit:
        items = items[:limit]

    scores = run_benchmark(
        items=items,
        baseline_name=baseline,
        model=model,
        extraction_model=extraction_model,
        judge_model=judge_model,
        output_dir=output_dir / "locomo",
        debug=debug,
    )

    return scores.summary(), scores.results


def run_msc(
    baseline: str,
    limit: int | None,
    model: str,
    extraction_model: str,
    judge_model: str,
    output_dir: Path,
    debug: bool,
) -> tuple[dict, list[dict]]:
    """Run MSC benchmark."""
    from benchmarks.msc.adapter import (
        load_msc_personas,
        create_persona_test_case,
        format_conversations_as_text,
    )
    from benchmarks.msc.runner import run_benchmark

    # Try to load personas, or use synthetic ones
    personas_path = PROJECT_ROOT / "benchmarks/msc/data/msc_personas.json"
    if personas_path.exists():
        personas = load_msc_personas(personas_path)
    else:
        logger.warning(f"MSC personas not found, using synthetic test data")
        personas = [
            ["I love hiking in the mountains.", "I work as a software engineer.", "I have two cats."],
            ["I enjoy cooking Italian food.", "I'm studying medicine.", "I play piano."],
            ["I'm passionate about photography.", "I work from home.", "I have a golden retriever."],
        ]

    # Create eval items from persona test cases
    items = []
    num_personas = min(limit or 10, len(personas))
    for i in range(num_personas):
        test_case = create_persona_test_case(personas, num_sessions=3, seed=i)
        for q in test_case["test_questions"]:
            items.append({
                "item_id": f"msc_{i}_{q['question_id']}",
                "item_type": "memory_qa",
                "conversations": test_case["conversations"],
                "personas": test_case["persona"],
                "question": q["question"],
                "answer": q["answer"],
                "expected_response": q["answer"],
                "context_text": format_conversations_as_text(test_case["conversations"]),
            })

    if limit:
        items = items[:limit]

    scores = run_benchmark(
        items=items,
        baseline_name=baseline,
        model=model,
        extraction_model=extraction_model,
        judge_model=judge_model,
        output_dir=output_dir / "msc",
        debug=debug,
    )

    return scores.summary(), scores.results


BENCHMARK_RUNNERS = {
    "longmemeval": run_longmemeval,
    "locomo": run_locomo,
    "msc": run_msc,
}


# ── Main Harness ──────────────────────────────────────────────────────────────


def run_eval_harness(
    benchmarks: list[str],
    baselines: list[str],
    subset: int | None = None,
    model: str = "gpt-4o",
    extraction_model: str = "gpt-4o-mini",
    judge_model: str = "gpt-4o",
    output_dir: Path | None = None,
    debug: bool = False,
) -> UnifiedResults:
    """
    Run the full evaluation harness.

    Args:
        benchmarks: List of benchmark names to run.
        baselines: List of baseline names to compare.
        subset: Limit samples per benchmark (for quick testing).
        model: LLM model for answering.
        extraction_model: LLM model for PIE extraction.
        judge_model: LLM model for judging.
        output_dir: Directory for output files.
        debug: Enable debug output.

    Returns:
        UnifiedResults with all benchmark results.
    """
    results = UnifiedResults()
    results.config = {
        "benchmarks": benchmarks,
        "baselines": baselines,
        "subset": subset,
        "model": model,
        "extraction_model": extraction_model,
        "judge_model": judge_model,
    }

    if output_dir is None:
        output_dir = PROJECT_ROOT / "benchmarks" / "results" / datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    total_benchmarks = len(benchmarks)
    total_baselines = len(baselines)

    for bi, benchmark in enumerate(benchmarks):
        if benchmark not in BENCHMARK_RUNNERS:
            logger.warning(f"Unknown benchmark: {benchmark}")
            continue

        logger.info(f"\n{'='*70}")
        logger.info(f"  BENCHMARK {bi+1}/{total_benchmarks}: {BENCHMARKS.get(benchmark, {}).get('name', benchmark)}")
        logger.info(f"{'='*70}")

        runner = BENCHMARK_RUNNERS[benchmark]

        for bj, baseline in enumerate(baselines):
            logger.info(f"\n  Baseline {bj+1}/{total_baselines}: {baseline}")
            logger.info(f"  {'-'*60}")

            try:
                summary, raw = runner(
                    baseline=baseline,
                    limit=subset,
                    model=model,
                    extraction_model=extraction_model,
                    judge_model=judge_model,
                    output_dir=output_dir,
                    debug=debug,
                )

                results.add_benchmark_result(
                    benchmark=benchmark,
                    baseline=baseline,
                    summary=summary,
                    raw_results=raw,
                )

            except Exception as e:
                logger.error(f"Failed to run {benchmark}/{baseline}: {e}")
                results.add_benchmark_result(
                    benchmark=benchmark,
                    baseline=baseline,
                    summary={"error": str(e)},
                )

    # Print and save final report
    results.print_report()
    results.save(output_dir)

    return results


# ── CLI ───────────────────────────────────────────────────────────────────────


def main():
    parser = argparse.ArgumentParser(
        description="Unified Evaluation Harness for PIE Benchmarks",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    # Quick sanity check
    python -m benchmarks.eval_harness --baseline pie_temporal --subset 5

    # Compare all baselines
    python -m benchmarks.eval_harness --baseline all --subset 10

    # Run specific benchmarks
    python -m benchmarks.eval_harness --benchmarks longmemeval locomo --baseline all

    # Full evaluation (takes hours)
    python -m benchmarks.eval_harness --baseline all
        """,
    )

    parser.add_argument(
        "--benchmarks",
        nargs="+",
        default=list(BENCHMARKS.keys()),
        choices=list(BENCHMARKS.keys()),
        help="Benchmarks to run (default: all)",
    )
    parser.add_argument(
        "--baseline", "-b",
        type=str,
        default="pie_temporal",
        choices=["full_context", "naive_rag", "pie_temporal", "all"],
        help="Baseline to run (default: pie_temporal)",
    )
    parser.add_argument(
        "--subset", "-n",
        type=int,
        default=None,
        help="Limit samples per benchmark (for quick testing)",
    )
    parser.add_argument("--model", "-m", type=str, default="gpt-4o")
    parser.add_argument("--extraction-model", type=str, default="gpt-4o-mini")
    parser.add_argument("--judge-model", type=str, default="gpt-4o")
    parser.add_argument("--output", "-o", type=Path, default=None)
    parser.add_argument("--debug", "-d", action="store_true")
    parser.add_argument(
        "--list-benchmarks",
        action="store_true",
        help="List available benchmarks and exit",
    )

    args = parser.parse_args()

    if args.list_benchmarks:
        print("\nAvailable Benchmarks:")
        print("-" * 60)
        for name, info in BENCHMARKS.items():
            print(f"  {name:<15} {info['description']}")
            print(f"  {' '*15} Types: {', '.join(info['question_types'][:3])}...")
        print()
        return

    # Determine baselines to run
    if args.baseline == "all":
        baselines = BASELINES
    else:
        baselines = [args.baseline]

    # Run harness
    run_eval_harness(
        benchmarks=args.benchmarks,
        baselines=baselines,
        subset=args.subset,
        model=args.model,
        extraction_model=args.extraction_model,
        judge_model=args.judge_model,
        output_dir=args.output,
        debug=args.debug,
    )


if __name__ == "__main__":
    main()
