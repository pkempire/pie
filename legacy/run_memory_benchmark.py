#!/usr/bin/env python3
"""
Memory Provider Benchmark Suite
===============================

Run all memory providers against all benchmarks.

Usage:
    # Test locally with a few examples
    python run_memory_benchmark.py --test
    
    # Run specific provider on specific benchmark
    python run_memory_benchmark.py --provider pie --benchmark longmemeval --limit 50
    
    # Run all providers on all benchmarks
    python run_memory_benchmark.py --all
    
    # List available providers and benchmarks
    python run_memory_benchmark.py --list

Providers: pie, zep, mem0, supermemory, honcho
Benchmarks: longmemeval, locomo, tot, msc
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import sys
import time
from datetime import datetime
from pathlib import Path
from collections import defaultdict

# Add project root
PROJECT_ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(PROJECT_ROOT))

# Load .env
env_path = PROJECT_ROOT / ".env"
if env_path.exists():
    for line in env_path.read_text().splitlines():
        if "=" in line and not line.startswith("#"):
            k, v = line.split("=", 1)
            os.environ.setdefault(k.strip(), v.strip())

from memory_providers import get_provider, list_providers, MemoryProviderConfig

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("benchmark")

# Auto-logging to RESULTS.json
RESULTS_FILE = PROJECT_ROOT / "benchmark_results" / "RESULTS.json"


def log_result(benchmark: str, provider: str, result: dict):
    """Auto-log a benchmark result to RESULTS.json"""
    try:
        if RESULTS_FILE.exists():
            with open(RESULTS_FILE) as f:
                data = json.load(f)
        else:
            data = {"meta": {}, "benchmarks": {}, "manual_tests": []}
        
        # Update timestamp
        data["meta"]["last_updated"] = datetime.now().isoformat()
        
        # Add to benchmark runs
        if benchmark not in data["benchmarks"]:
            data["benchmarks"][benchmark] = {"runs": []}
        
        run_entry = {
            "provider": provider,
            "accuracy": result.get("accuracy"),
            "total": result.get("total"),
            "by_category": result.get("by_category", result.get("by_type", {})),
            "timestamp": datetime.now().isoformat(),
        }
        data["benchmarks"][benchmark]["runs"].append(run_entry)
        
        # Save
        RESULTS_FILE.parent.mkdir(parents=True, exist_ok=True)
        with open(RESULTS_FILE, "w") as f:
            json.dump(data, f, indent=2)
        
        logger.info(f"Result logged to {RESULTS_FILE}")
    except Exception as e:
        logger.warning(f"Failed to log result: {e}")


# ══════════════════════════════════════════════════════════════════════════════
# Benchmark Loaders
# ══════════════════════════════════════════════════════════════════════════════

def load_longmemeval(limit: int | None = None) -> list[dict]:
    """Load LongMemEval dataset."""
    path = PROJECT_ROOT / "benchmarks/longmemeval/data/longmemeval_s_cleaned.json"
    if not path.exists():
        logger.warning(f"LongMemEval not found at {path}")
        return []
    
    with open(path) as f:
        data = json.load(f)
    
    if limit:
        data = data[:limit]
    
    return data


def load_locomo(limit: int | None = None) -> list[dict]:
    """Load LoCoMo dataset.

    LoCoMo format:
      conversation: dict with speaker_a, speaker_b, session_1..N, session_N_date_time
      qa: list of {question, answer, category (1-5), evidence}
    """
    path = PROJECT_ROOT / "benchmarks/locomo/data/locomo10.json"
    if not path.exists():
        logger.warning(f"LoCoMo not found at {path}")
        return []

    with open(path) as f:
        conversations = json.load(f)

    CATEGORY_MAP = {1: "single_hop", 2: "multi_hop", 3: "temporal", 4: "adversarial", 5: "commonsense"}

    items = []
    for conv in conversations:
        conv_data = conv.get("conversation", {})
        speaker_a = conv_data.get("speaker_a", "Speaker A")
        speaker_b = conv_data.get("speaker_b", "Speaker B")

        # Extract sessions (session_1, session_2, ...) and their dates
        sessions = []
        dates = []
        for i in range(1, 30):
            session_key = f"session_{i}"
            date_key = f"session_{i}_date_time"
            if session_key not in conv_data:
                break

            # Convert turns: {speaker, text} → {role, content}
            raw_turns = conv_data[session_key]
            formatted_turns = []
            for turn in raw_turns:
                speaker = turn.get("speaker", "")
                role = "user" if speaker == speaker_a else "assistant"
                formatted_turns.append({
                    "role": role,
                    "content": f"[{speaker}]: {turn.get('text', '')}",
                })
            sessions.append(formatted_turns)
            dates.append(conv_data.get(date_key, f"Session {i}"))

        # QA pairs use "qa" key, not "qa_pairs"
        # Some items (category 5 = commonsense) use "adversarial_answer" instead of "answer"
        conv_id = conv.get("sample_id", f"conv_{len(items)}")
        for qa in conv.get("qa", []):
            answer = qa.get("answer", qa.get("adversarial_answer", ""))
            items.append({
                "question": qa["question"],
                "answer": str(answer),
                "category": CATEGORY_MAP.get(qa.get("category", 0), "unknown"),
                "sessions": sessions,
                "dates": dates,
                "conv_id": conv_id,
            })

    if limit:
        items = items[:limit]

    return items


def load_tot(limit: int | None = None) -> list[dict]:
    """Load Test of Time dataset."""
    arith_path = PROJECT_ROOT / "benchmarks/tot/tot_arithmetic.json"
    sem_path = PROJECT_ROOT / "benchmarks/tot/tot_semantic.json"
    
    items = []
    
    if arith_path.exists():
        with open(arith_path) as f:
            arith = json.load(f)
        for item in arith[:limit or 100]:
            items.append({
                "question": item.get("question", ""),
                "answer": str(item.get("label", "")),
                "category": f"arithmetic_{item.get('question_type', 'unknown')}",
                "prompt": item.get("prompt", ""),
            })
    
    if sem_path.exists():
        with open(sem_path) as f:
            sem = json.load(f)
        # Filter by length and sample
        sem = [s for s in sem if len(s.get("prompt", "")) < 16000]
        for item in sem[:limit or 100]:
            items.append({
                "question": item.get("question", ""),
                "answer": item.get("label", ""),
                "category": f"semantic_{item.get('question_type', 'unknown')}",
                "prompt": item.get("prompt", ""),
            })
    
    return items


BENCHMARKS = {
    "longmemeval": load_longmemeval,
    "locomo": load_locomo,
    "tot": load_tot,
}


# ══════════════════════════════════════════════════════════════════════════════
# Judging
# ══════════════════════════════════════════════════════════════════════════════

def judge_answer(question: str, gold: str, hypothesis: str) -> tuple[float, str]:
    """Judge if hypothesis matches gold answer."""
    from openai import OpenAI
    client = OpenAI()
    
    try:
        response = client.chat.completions.create(
            model="gpt-4o",
            messages=[{
                "role": "user",
                "content": f"""Judge if the hypothesis correctly answers the question.

Question: {question}
Gold Answer: {gold}
Hypothesis: {hypothesis}

Score:
- 1.0 = Fully correct (captures the key information)
- 0.5 = Partially correct (some relevant info but incomplete)
- 0.0 = Wrong or irrelevant

Reply with ONLY the score (1.0, 0.5, or 0.0):"""
            }],
            max_tokens=10,
        )
        
        score_text = response.choices[0].message.content.strip()
        if "1.0" in score_text or "1" == score_text:
            return 1.0, "correct"
        elif "0.5" in score_text:
            return 0.5, "partial"
        else:
            return 0.0, "wrong"
            
    except Exception as e:
        return 0.0, f"judge_error: {e}"


# ══════════════════════════════════════════════════════════════════════════════
# Main Benchmark Runner
# ══════════════════════════════════════════════════════════════════════════════

def run_benchmark(
    provider_name: str,
    benchmark_name: str,
    limit: int | None = None,
    output_dir: Path | None = None,
) -> dict:
    """
    Run a single provider on a single benchmark.
    
    Returns dict with accuracy and detailed results.
    """
    logger.info(f"\n{'='*60}")
    logger.info(f"  {provider_name.upper()} on {benchmark_name.upper()}")
    logger.info(f"{'='*60}")
    
    # Load benchmark data
    if benchmark_name not in BENCHMARKS:
        raise ValueError(f"Unknown benchmark: {benchmark_name}")
    
    items = BENCHMARKS[benchmark_name](limit)
    if not items:
        return {"error": "No data loaded"}
    
    logger.info(f"Loaded {len(items)} items")
    
    # Initialize provider
    api_key_map = {
        "mem0": os.environ.get("MEM0_API_KEY"),
        "zep": os.environ.get("ZEP_API_KEY"),
        "graphiti": os.environ.get("ZEP_API_KEY"),
        "supermemory": os.environ.get("SUPERMEMORY_API_KEY"),
    }
    config = MemoryProviderConfig(
        api_key=api_key_map.get(provider_name.lower()),
        model="gpt-4o",
    )
    provider = get_provider(provider_name, config)

    # ── For LoCoMo: group items by conversation to avoid re-ingesting ──
    # All QA items from the same conversation share sessions, so ingest once.
    _last_conv_id = None

    # Run evaluation
    results = []
    scores_by_category = defaultdict(lambda: {"count": 0, "score": 0.0})
    total_score = 0

    for i, item in enumerate(items):
        t0 = time.time()
        gold_answer = item.get("answer", str(item.get("gold", "")))

        try:
            # Ingest context for this item
            if benchmark_name == "longmemeval":
                sessions = item.get("haystack_sessions", [])
                dates = item.get("haystack_dates", [])
                provider.clear()
                provider.ingest(sessions, dates)
            elif benchmark_name == "locomo":
                conv_id = item.get("conv_id", "")
                if conv_id != _last_conv_id:
                    sessions = item.get("sessions", [])
                    dates = item.get("dates")
                    logger.info(f"Ingesting conversation {conv_id} ({len(sessions)} sessions)...")
                    provider.clear()
                    if sessions:
                        provider.ingest(sessions, dates)
                    _last_conv_id = conv_id
            elif benchmark_name == "tot":
                # ToT doesn't need ingestion - context is in prompt
                pass

            # Generate answer
            if benchmark_name == "tot":
                # ToT is self-contained (facts + question in prompt).
                # Bypass the provider pipeline — just send to LLM directly.
                from openai import OpenAI
                _oa = OpenAI()
                full_prompt = f"{item.get('prompt', '')}\n\n{item['question']}\n\nAnswer with ONLY the answer value, nothing else."
                _resp = _oa.chat.completions.create(
                    model="gpt-4o",
                    messages=[{"role": "user", "content": full_prompt}],
                    max_tokens=100,
                )
                hypothesis = _resp.choices[0].message.content.strip()
            else:
                hypothesis = provider.answer(
                    item["question"],
                    item.get("question_date"),
                )
            
            latency = time.time() - t0

            # Judge
            score, reason = judge_answer(item["question"], gold_answer, hypothesis)
            
        except Exception as e:
            hypothesis = f"Error: {e}"
            score = 0.0
            reason = "error"
            latency = time.time() - t0
        
        total_score += score
        category = item.get("category") or item.get("question_type", "unknown")
        scores_by_category[category]["count"] += 1
        scores_by_category[category]["score"] += score
        
        results.append({
            "question": item["question"][:100],
            "gold": gold_answer[:100],
            "hypothesis": hypothesis[:200],
            "score": score,
            "category": category,
            "latency_ms": round(latency * 1000),
        })
        
        # Progress
        emoji = "✅" if score == 1.0 else "🟡" if score == 0.5 else "❌"
        running_acc = 100 * total_score / (i + 1)
        logger.info(f"{emoji} [{i+1}/{len(items)}] {running_acc:.1f}% | {category[:15]}")
    
    # Summary
    final_acc = 100 * total_score / len(items)
    
    summary = {
        "provider": provider_name,
        "benchmark": benchmark_name,
        "accuracy": final_acc,
        "total": len(items),
        "by_category": {
            cat: 100 * d["score"] / d["count"] if d["count"] > 0 else 0
            for cat, d in scores_by_category.items()
        },
        "results": results,
    }
    
    logger.info(f"\n{provider_name} on {benchmark_name}: {final_acc:.1f}%")
    for cat, acc in summary["by_category"].items():
        logger.info(f"  {cat}: {acc:.1f}%")
    
    # Save results
    if output_dir:
        output_dir.mkdir(parents=True, exist_ok=True)
        path = output_dir / f"{provider_name}_{benchmark_name}.json"
        with open(path, "w") as f:
            json.dump(summary, f, indent=2)
        logger.info(f"Saved to {path}")
    
    # Auto-log to central RESULTS.json
    log_result(benchmark_name, provider_name, summary)
    
    return summary


def run_manual_test(provider_name: str):
    """
    Run a quick manual test to verify provider works.
    
    Tests basic ingest → search → answer flow.
    """
    logger.info(f"\n{'='*60}")
    logger.info(f"  MANUAL TEST: {provider_name.upper()}")
    logger.info(f"{'='*60}")
    
    # Sample data
    sessions = [
        [
            {"role": "user", "content": "I'm working on a project called PIE - it's a personal intelligence engine."},
            {"role": "assistant", "content": "That sounds interesting! What does PIE do?"},
            {"role": "user", "content": "It builds a temporal knowledge graph from my conversations to help me recall things."},
        ],
        [
            {"role": "user", "content": "I just visited the MoMA yesterday. The modern art exhibit was amazing."},
            {"role": "assistant", "content": "Which exhibit did you see?"},
            {"role": "user", "content": "The Picasso collection. I spent 3 hours there."},
        ],
        [
            {"role": "user", "content": "I've been learning Rust lately. It's challenging but rewarding."},
            {"role": "assistant", "content": "What made you choose Rust?"},
            {"role": "user", "content": "The memory safety guarantees. I want to use it for my next project."},
        ],
    ]
    dates = ["2025-01-15", "2025-01-20", "2025-01-25"]
    
    # Initialize provider with API keys from environment
    api_key_map = {
        "mem0": os.environ.get("MEM0_API_KEY"),
        "zep": os.environ.get("ZEP_API_KEY"),
        "graphiti": os.environ.get("ZEP_API_KEY"),
        "supermemory": os.environ.get("SUPERMEMORY_API_KEY"),
    }
    
    config = MemoryProviderConfig(
        api_key=api_key_map.get(provider_name.lower()),
        model="gpt-4o"
    )
    provider = get_provider(provider_name, config)
    
    # Test ingestion
    logger.info("\n1. INGEST")
    logger.info(f"   Ingesting {len(sessions)} sessions...")
    provider.ingest(sessions, dates)
    stats = provider.stats()
    logger.info(f"   Stats: {stats}")
    
    # Test search
    logger.info("\n2. SEARCH")
    queries = [
        "What project am I working on?",
        "When did I visit the museum?",
        "What programming language am I learning?",
    ]
    
    for query in queries:
        results = provider.search(query, top_k=3)
        logger.info(f"\n   Query: {query}")
        for r in results[:2]:
            logger.info(f"   → [{r.score:.2f}] {r.content[:80]}...")
    
    # Test answer
    logger.info("\n3. ANSWER")
    questions = [
        "What is PIE?",
        "What museum did I visit and when?",
        "Why am I learning Rust?",
    ]
    
    for question in questions:
        answer = provider.answer(question)
        logger.info(f"\n   Q: {question}")
        logger.info(f"   A: {answer[:150]}...")
    
    logger.info("\n✅ Manual test complete!")


# ══════════════════════════════════════════════════════════════════════════════
# CLI
# ══════════════════════════════════════════════════════════════════════════════

def main():
    parser = argparse.ArgumentParser(description="Memory Provider Benchmark Suite")
    parser.add_argument("--provider", "-p", type=str, help="Provider to test")
    parser.add_argument("--benchmark", "-b", type=str, help="Benchmark to run")
    parser.add_argument("--limit", "-n", type=int, help="Limit items per benchmark")
    parser.add_argument("--output", "-o", type=Path, help="Output directory")
    parser.add_argument("--test", action="store_true", help="Run quick manual test")
    parser.add_argument("--all", action="store_true", help="Run all providers on all benchmarks")
    parser.add_argument("--list", action="store_true", help="List providers and benchmarks")
    
    args = parser.parse_args()
    
    if args.list:
        print("\nProviders:")
        for p in list_providers():
            print(f"  - {p}")
        print("\nBenchmarks:")
        for b in BENCHMARKS.keys():
            print(f"  - {b}")
        return
    
    if args.test:
        provider = args.provider or "pie"
        run_manual_test(provider)
        return
    
    # Determine what to run
    providers = [args.provider] if args.provider else list_providers()
    benchmarks = [args.benchmark] if args.benchmark else list(BENCHMARKS.keys())
    
    if args.all:
        providers = list_providers()
        benchmarks = list(BENCHMARKS.keys())
    
    if not args.provider and not args.all:
        parser.print_help()
        return
    
    # Output directory
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = args.output or (PROJECT_ROOT / "benchmark_results" / timestamp)
    
    # Run benchmarks
    all_results = {}
    
    for provider_name in providers:
        for benchmark_name in benchmarks:
            try:
                result = run_benchmark(
                    provider_name=provider_name,
                    benchmark_name=benchmark_name,
                    limit=args.limit,
                    output_dir=output_dir,
                )
                all_results[f"{provider_name}_{benchmark_name}"] = result
            except Exception as e:
                logger.error(f"Failed {provider_name}/{benchmark_name}: {e}")
                all_results[f"{provider_name}_{benchmark_name}"] = {"error": str(e)}
    
    # Print summary
    print("\n" + "=" * 70)
    print("  FINAL RESULTS")
    print("=" * 70)
    print(f"{'Provider':<15} {'Benchmark':<15} {'Accuracy':>10}")
    print("-" * 70)
    
    for key, result in all_results.items():
        parts = key.split("_", 1)
        provider = parts[0]
        benchmark = parts[1] if len(parts) > 1 else "unknown"
        acc = result.get("accuracy", "N/A")
        if isinstance(acc, float):
            acc = f"{acc:.1f}%"
        print(f"{provider:<15} {benchmark:<15} {acc:>10}")
    
    print("=" * 70)
    print(f"\nResults saved to: {output_dir}/")


if __name__ == "__main__":
    main()
