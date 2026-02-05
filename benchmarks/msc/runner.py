#!/usr/bin/env python3
"""
MSC (Multi-Session Chat) Benchmark Runner for PIE

Usage:
    # Run PIE on 5 examples
    python -m benchmarks.msc.runner --baseline pie_temporal --limit 5

    # Response generation task
    python -m benchmarks.msc.runner --task response_generation --limit 10

    # Memory QA task
    python -m benchmarks.msc.runner --task memory_qa --limit 10

MSC evaluates:
    - Persona consistency across sessions
    - Memory of facts from prior conversations
    - Natural response generation
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
import time
from collections import defaultdict
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from pie.core.llm import LLMClient

from benchmarks.msc.adapter import (
    load_msc_personas,
    create_persona_test_case,
    dataset_stats,
    parse_msc_example,
    format_conversations_as_text,
)
from benchmarks.msc.baselines import (
    BaselineResult,
    full_context,
    naive_rag,
    pie_temporal,
    BASELINES,
)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("pie.bench.msc")


# ── Evaluation ────────────────────────────────────────────────────────────────

RESPONSE_JUDGE_SYSTEM = """\
You are evaluating the quality of a conversational response.
Score on multiple dimensions:

1. Fluency (0-1): Is the response natural and grammatically correct?
2. Relevance (0-1): Does it appropriately continue the conversation?
3. Consistency (0-1): Is it consistent with the speaker's persona/history?
4. Engagingness (0-1): Is it interesting and engaging?

Output JSON: {"fluency": X, "relevance": X, "consistency": X, "engagingness": X, "overall": X, "reason": "..."}
Overall should be the average of the four scores."""

RESPONSE_JUDGE_USER = """\
Persona: {personas}

Conversation context:
{context}

Generated response: {response}

Reference response (if available): {reference}

Score the generated response:"""

QA_JUDGE_SYSTEM = """\
Evaluate if the predicted answer is correct:
  1.0 = Correct
  0.5 = Partially correct
  0.0 = Incorrect

Output JSON: {"score": X, "reason": "..."}"""


def judge_response(
    context: str,
    response: str,
    reference: str,
    personas: list[str],
    llm: LLMClient,
    model: str = "gpt-4o",
) -> tuple[float, dict]:
    """Judge a response generation result."""
    messages = [
        {"role": "system", "content": RESPONSE_JUDGE_SYSTEM},
        {
            "role": "user",
            "content": RESPONSE_JUDGE_USER.format(
                personas="\n".join(personas) if personas else "None",
                context=context[:2000],
                response=response,
                reference=reference or "Not available",
            ),
        },
    ]

    try:
        result = llm.chat(messages=messages, model=model, json_mode=True)
        parsed = result["content"]
        if isinstance(parsed, str):
            parsed = json.loads(parsed)

        overall = float(parsed.get("overall", 0.5))
        return overall, parsed

    except Exception as e:
        logger.warning(f"Judge failed: {e}")
        return 0.5, {"error": str(e)}


def judge_qa(
    question: str,
    gold: str,
    hypothesis: str,
    llm: LLMClient,
    model: str = "gpt-4o",
) -> tuple[float, str]:
    """Judge a QA result."""
    hyp_lower = hypothesis.lower().strip()
    if "i don't know" in hyp_lower:
        return 0.0, "No answer"

    messages = [
        {"role": "system", "content": QA_JUDGE_SYSTEM},
        {
            "role": "user",
            "content": f"Question: {question}\nExpected: {gold}\nPredicted: {hypothesis}",
        },
    ]

    try:
        result = llm.chat(messages=messages, model=model, json_mode=True)
        parsed = result["content"]
        if isinstance(parsed, str):
            parsed = json.loads(parsed)

        score = float(parsed.get("score", 0.0))
        reason = parsed.get("reason", "")

        if score >= 0.75:
            score = 1.0
        elif score >= 0.25:
            score = 0.5
        else:
            score = 0.0

        return score, reason

    except Exception as e:
        return 0.0, f"Error: {e}"


# ── Score Aggregation ─────────────────────────────────────────────────────────


@dataclass
class BenchmarkScores:
    """Aggregated scores."""
    total: int = 0
    total_score: float = 0.0
    by_type: dict[str, dict] = field(default_factory=lambda: defaultdict(
        lambda: {"count": 0, "score": 0.0}
    ))
    results: list[dict] = field(default_factory=list)

    @property
    def overall_accuracy(self) -> float:
        return self.total_score / self.total if self.total > 0 else 0.0

    def add(self, result: BaselineResult, score: float, details: Any = None):
        self.total += 1
        self.total_score += score
        itype = result.item_type
        self.by_type[itype]["count"] += 1
        self.by_type[itype]["score"] += score
        self.results.append({
            **result.to_dict(),
            "judge_score": score,
            "judge_details": details,
        })

    def summary(self) -> dict:
        return {
            "overall": {
                "score": round(self.overall_accuracy * 100, 1),
                "total": self.total,
            },
            "by_type": {
                t: {
                    "score": round(d["score"] / d["count"] * 100, 1) if d["count"] > 0 else 0,
                    "count": d["count"],
                }
                for t, d in self.by_type.items()
            },
        }

    def print_report(self, baseline_name: str):
        s = self.summary()
        print("\n" + "=" * 60)
        print(f"  MSC Results — {baseline_name}")
        print("=" * 60)
        print(f"  Overall Score: {s['overall']['score']}% ({s['overall']['total']} items)")
        for t, d in s["by_type"].items():
            print(f"    {t}: {d['score']}% ({d['count']} items)")
        print("=" * 60)


# ── Runner ────────────────────────────────────────────────────────────────────


def run_benchmark(
    items: list[dict[str, Any]],
    baseline_name: str = "pie_temporal",
    model: str = "gpt-4o",
    extraction_model: str = "gpt-4o-mini",
    judge_model: str = "gpt-4o",
    output_dir: Path | None = None,
    debug: bool = False,
) -> BenchmarkScores:
    """Run the MSC benchmark."""
    llm = LLMClient()
    scores = BenchmarkScores()
    total = len(items)
    t0 = time.time()

    if output_dir:
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

    logger.info(f"Running MSC benchmark: {baseline_name}")
    logger.info(f"  Items: {total}")

    for i, item in enumerate(items):
        item_id = item["item_id"]
        item_type = item.get("item_type", "unknown")

        logger.info(f"[{i+1}/{total}] {item_id} ({item_type})")

        try:
            # Run baseline
            if baseline_name == "full_context":
                result = full_context(item, llm=llm, model=model)
            elif baseline_name == "naive_rag":
                result = naive_rag(item, llm=llm, model=model)
            elif baseline_name == "pie_temporal":
                result = pie_temporal(
                    item, llm=llm, model=model, extraction_model=extraction_model
                )
            else:
                raise ValueError(f"Unknown baseline: {baseline_name}")

            # Judge
            if item_type == "response_generation":
                score, details = judge_response(
                    context=item.get("context_text", "")[:2000],
                    response=result.hypothesis,
                    reference=item.get("expected_response", ""),
                    personas=item.get("personas", []),
                    llm=llm,
                    model=judge_model,
                )
            else:
                score, details = judge_qa(
                    question=item.get("question", ""),
                    gold=item.get("answer", ""),
                    hypothesis=result.hypothesis,
                    llm=llm,
                    model=judge_model,
                )

            scores.add(result, score, details)

            emoji = "✅" if score >= 0.7 else "🟡" if score >= 0.4 else "❌"
            logger.info(f"  {emoji} {score:.2f} | {result.latency_ms:.0f}ms")

            if debug:
                print(f"  Response: {result.hypothesis[:100]}...")

        except Exception as e:
            logger.error(f"  ❌ Error: {e}")
            error_result = BaselineResult(
                item_id=item_id,
                item_type=item_type,
                question=item.get("question", ""),
                gold_answer=item.get("expected_response", item.get("answer", "")),
                hypothesis=f"Error: {e}",
                baseline_name=baseline_name,
                model=model,
                error=str(e),
            )
            scores.add(error_result, 0.0)

    total_time = time.time() - t0
    logger.info(f"\nCompleted in {total_time:.0f}s")

    scores.print_report(baseline_name)

    if output_dir:
        _save_results(scores, output_dir, baseline_name)

    return scores


def _save_results(scores: BenchmarkScores, output_dir: Path, baseline_name: str):
    """Save results."""
    # JSONL
    with open(output_dir / f"{baseline_name}_results.jsonl", "w") as f:
        for r in scores.results:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")

    # Summary
    summary = scores.summary()
    summary["baseline"] = baseline_name
    summary["benchmark"] = "msc"
    summary["timestamp"] = datetime.now(timezone.utc).isoformat()
    with open(output_dir / f"{baseline_name}_summary.json", "w") as f:
        json.dump(summary, f, indent=2)

    logger.info(f"Results saved to {output_dir}/")


# ── CLI ───────────────────────────────────────────────────────────────────────


def main():
    parser = argparse.ArgumentParser(description="MSC Benchmark Runner")

    parser.add_argument(
        "--dataset",
        type=Path,
        default=PROJECT_ROOT / "benchmarks/msc/data/msc_valid.jsonl",
    )
    parser.add_argument(
        "--task",
        type=str,
        default="response_generation",
        choices=["response_generation", "memory_qa"],
    )
    parser.add_argument("--limit", "-n", type=int, default=None)
    parser.add_argument(
        "--baseline", "-b",
        type=str,
        default="pie_temporal",
        choices=["full_context", "naive_rag", "pie_temporal", "all"],
    )
    parser.add_argument("--model", "-m", type=str, default="gpt-4o")
    parser.add_argument("--extraction-model", type=str, default="gpt-4o-mini")
    parser.add_argument("--judge-model", type=str, default="gpt-4o")
    parser.add_argument("--output", "-o", type=Path, default=None)
    parser.add_argument("--debug", "-d", action="store_true")
    parser.add_argument("--stats", action="store_true")

    args = parser.parse_args()

    # Load dataset (personas for synthetic tests)
    personas_path = PROJECT_ROOT / "benchmarks/msc/data/msc_personas.json"
    if personas_path.exists():
        logger.info(f"Loading personas from {personas_path}")
        personas = load_msc_personas(personas_path)
        logger.info(f"Loaded {len(personas)} personas")
    else:
        # Generate synthetic personas for testing
        logger.warning(f"Personas not found at {personas_path}")
        logger.info("Using synthetic test personas")
        personas = [
            ["I love hiking in the mountains.", "I work as a software engineer.", "I have two cats."],
            ["I enjoy cooking Italian food.", "I'm studying medicine.", "I play piano."],
            ["I'm passionate about photography.", "I work from home.", "I have a golden retriever."],
        ]

    if args.stats:
        stats = dataset_stats(personas)
        print(json.dumps(stats, indent=2))
        return

    # Create eval items from persona test cases
    items = []
    for i, _ in enumerate(personas[:args.limit or 10]):
        test_case = create_persona_test_case(personas, num_sessions=3, seed=i)
        for q in test_case["test_questions"]:
            items.append({
                "item_id": f"msc_{i}_{q['question_id']}",
                "item_type": args.task,
                "conversations": test_case["conversations"],
                "personas": test_case["persona"],
                "question": q["question"],
                "answer": q["answer"],
                "expected_response": q["answer"],
                "context_text": format_conversations_as_text(test_case["conversations"]),
            })
    logger.info(f"Created {len(items)} eval items for {args.task}")

    if args.limit:
        items = items[:args.limit]

    if not items:
        logger.error("No items to evaluate")
        sys.exit(1)

    # Output dir
    if args.output is None:
        args.output = (
            PROJECT_ROOT / "benchmarks" / "msc" / "results"
            / datetime.now().strftime("%Y%m%d_%H%M%S")
        )

    # Run
    if args.baseline == "all":
        for baseline in ["full_context", "naive_rag", "pie_temporal"]:
            run_benchmark(
                items=items,
                baseline_name=baseline,
                model=args.model,
                extraction_model=args.extraction_model,
                judge_model=args.judge_model,
                output_dir=args.output,
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
            debug=args.debug,
        )


if __name__ == "__main__":
    main()
