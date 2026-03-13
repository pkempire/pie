#!/usr/bin/env python3
"""
Benchmark Smoke Test
====================

Quick validation that all benchmark datasets load, parse correctly,
and baseline functions can be called. No API calls made.

Run: python3 experiments/benchmark_smoke_test.py
"""

import sys
import json
import time
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))


def test_longmemeval():
    """Test LongMemEval dataset loads and parses correctly."""
    print("\n--- LongMemEval ---")
    from benchmarks.longmemeval.adapter import load_dataset, dataset_stats

    dataset_path = PROJECT_ROOT / "benchmarks/longmemeval/data/longmemeval_s_cleaned.json"
    if not dataset_path.exists():
        print(f"  ❌ Dataset not found: {dataset_path}")
        return False

    t0 = time.time()
    dataset = load_dataset(dataset_path)
    load_time = time.time() - t0

    stats = dataset_stats(dataset)
    print(f"  ✅ Loaded {len(dataset)} questions in {load_time:.1f}s")
    print(f"  Categories: {json.dumps(stats.get('by_category', {}), indent=6)}")

    # Validate first item structure
    item = dataset[0]
    required_keys = ["question_id", "question", "answer", "question_type",
                     "question_date", "haystack_sessions", "haystack_dates"]
    missing = [k for k in required_keys if k not in item]
    if missing:
        print(f"  ❌ Missing keys in first item: {missing}")
        return False

    print(f"  ✅ First question: {item['question'][:80]}...")
    print(f"     Type: {item['question_type']}")
    print(f"     Haystack sessions: {len(item['haystack_sessions'])}")
    print(f"     Answer: {item['answer'][:80]}...")

    # Check haystack structure
    first_session = item["haystack_sessions"][0]
    if isinstance(first_session, list) and len(first_session) > 0:
        first_turn = first_session[0]
        print(f"  ✅ First turn has keys: {list(first_turn.keys())}")
    else:
        print(f"  ⚠️ Unexpected session structure: {type(first_session)}")

    return True


def test_locomo():
    """Test LoCoMo dataset loads and parses correctly."""
    print("\n--- LoCoMo ---")
    from benchmarks.locomo.adapter import load_dataset, flatten_qa

    dataset_path = PROJECT_ROOT / "benchmarks/locomo/data/locomo10.json"
    if not dataset_path.exists():
        print(f"  ❌ Dataset not found: {dataset_path}")
        return False

    t0 = time.time()
    dataset = load_dataset(dataset_path)
    load_time = time.time() - t0

    print(f"  ✅ Loaded {len(dataset)} conversations in {load_time:.1f}s")

    # Flatten to QA items
    items = flatten_qa(dataset)
    print(f"  ✅ Flattened to {len(items)} QA items")

    if items:
        item = items[0]
        print(f"  ✅ First question: {item.get('question', 'N/A')[:80]}...")
        print(f"     Category: {item.get('category', 'N/A')}")
        print(f"     Answer: {str(item.get('answer', 'N/A'))[:80]}...")
        print(f"     Keys: {list(item.keys())}")

    # Category distribution
    from collections import Counter
    cats = Counter(item.get("category", "unknown") for item in items)
    print(f"  Categories: {dict(cats)}")

    return True


def test_msc():
    """Test MSC dataset loads and parses correctly."""
    print("\n--- MSC ---")
    from benchmarks.msc.adapter import load_msc_personas, create_persona_test_case

    personas_path = PROJECT_ROOT / "benchmarks/msc/data/msc_personas.json"
    if not personas_path.exists():
        print(f"  ❌ Personas not found: {personas_path}")
        return False

    t0 = time.time()
    personas = load_msc_personas(personas_path)
    load_time = time.time() - t0

    print(f"  ✅ Loaded {len(personas)} personas in {load_time:.1f}s")

    if personas:
        print(f"  ✅ First persona traits: {personas[0][:3]}...")

        # Create a test case
        test_case = create_persona_test_case(personas, num_sessions=3, seed=42)
        print(f"  ✅ Test case created:")
        print(f"     Persona traits: {len(test_case['persona'])}")
        print(f"     Conversations: {len(test_case['conversations'])}")
        print(f"     Test questions: {len(test_case['test_questions'])}")
        if test_case["test_questions"]:
            q = test_case["test_questions"][0]
            print(f"     First Q: {q.get('question', 'N/A')[:80]}...")

    return True


def test_baseline_imports():
    """Test that all baseline functions can be imported."""
    print("\n--- Baseline Imports ---")

    try:
        from benchmarks.longmemeval.baselines import (
            full_context, naive_rag, pie_temporal, pie_temporal_cached,
            PIETemporalCachedBaseline, BASELINES
        )
        print(f"  ✅ LongMemEval baselines imported: {list(BASELINES.keys())}")
    except Exception as e:
        print(f"  ❌ LongMemEval baselines import failed: {e}")
        return False

    try:
        from benchmarks.locomo.baselines import BASELINES as LOCOMO_BASELINES
        print(f"  ✅ LoCoMo baselines imported: {list(LOCOMO_BASELINES.keys())}")
    except Exception as e:
        print(f"  ❌ LoCoMo baselines import failed: {e}")
        return False

    try:
        from benchmarks.msc.baselines import BASELINES as MSC_BASELINES
        print(f"  ✅ MSC baselines imported: {list(MSC_BASELINES.keys())}")
    except Exception as e:
        print(f"  ❌ MSC baselines import failed: {e}")
        return False

    return True


def test_eval_harness():
    """Test that the eval harness can be imported and configured."""
    print("\n--- Eval Harness ---")

    try:
        from benchmarks.eval_harness import (
            run_eval_harness, BENCHMARKS, BASELINES, BENCHMARK_RUNNERS
        )
        print(f"  ✅ Eval harness imported")
        print(f"     Benchmarks: {list(BENCHMARKS.keys())}")
        print(f"     Baselines: {BASELINES}")
        print(f"     Runners: {list(BENCHMARK_RUNNERS.keys())}")
    except Exception as e:
        print(f"  ❌ Eval harness import failed: {e}")
        return False

    return True


def test_pie_core():
    """Test that PIE core modules load correctly."""
    print("\n--- PIE Core ---")

    try:
        from pie.core.models import Entity, EntityType, TransitionType, StateTransition
        print(f"  ✅ Models imported")
        print(f"     Entity types: {[e.value for e in EntityType]}")
        print(f"     Transition types: {[t.value for t in TransitionType]}")
    except Exception as e:
        print(f"  ❌ Models import failed: {e}")
        return False

    try:
        from pie.core.world_model import WorldModel
        wm = WorldModel()
        print(f"  ✅ WorldModel created (empty)")
    except Exception as e:
        print(f"  ❌ WorldModel creation failed: {e}")
        return False

    try:
        from pie.core.llm import LLMClient
        # Don't actually create client (needs API key)
        print(f"  ✅ LLMClient importable")
    except Exception as e:
        print(f"  ❌ LLMClient import failed: {e}")
        return False

    return True


def test_api_key():
    """Check if OpenAI API key is configured."""
    print("\n--- API Configuration ---")
    import os

    # Load .env
    env_path = PROJECT_ROOT / ".env"
    if env_path.exists():
        for line in env_path.read_text().splitlines():
            if "=" in line and not line.startswith("#"):
                k, v = line.split("=", 1)
                os.environ.setdefault(k.strip(), v.strip())
        print(f"  ✅ .env loaded")

    key = os.environ.get("OPENAI_API_KEY", "")
    if key:
        print(f"  ✅ OPENAI_API_KEY set ({key[:8]}...{key[-4:]})")
        return True
    else:
        print(f"  ❌ OPENAI_API_KEY not set")
        return False


if __name__ == "__main__":
    print("╔══════════════════════════════════════════════════════════════════════╗")
    print("║  BENCHMARK INFRASTRUCTURE SMOKE TEST                                ║")
    print("║  Validates all datasets, imports, and configuration                  ║")
    print("╚══════════════════════════════════════════════════════════════════════╝")

    results = {}
    results["pie_core"] = test_pie_core()
    results["longmemeval"] = test_longmemeval()
    results["locomo"] = test_locomo()
    results["msc"] = test_msc()
    results["baseline_imports"] = test_baseline_imports()
    results["eval_harness"] = test_eval_harness()
    results["api_key"] = test_api_key()

    print("\n" + "=" * 70)
    print("SMOKE TEST RESULTS")
    print("=" * 70)
    all_pass = True
    for name, passed in results.items():
        emoji = "✅" if passed else "❌"
        print(f"  {emoji} {name}")
        if not passed:
            all_pass = False

    if all_pass:
        print(f"\n  🎉 ALL TESTS PASSED — Ready to run benchmarks!")
        print(f"\n  Quick start commands:")
        print(f"  # Run naive_rag on 5 LongMemEval questions (fast, ~2 min)")
        print(f"  python -m benchmarks.eval_harness -b naive_rag -n 5 --benchmarks longmemeval")
        print(f"")
        print(f"  # Run all baselines on 10 questions each (moderate, ~15 min)")
        print(f"  python -m benchmarks.eval_harness -b all -n 10")
        print(f"")
        print(f"  # Run PIE temporal cached with caching (first run slow, then fast)")
        print(f"  python -m benchmarks.longmemeval.runner -b pie_temporal_cached -n 5 --cache-dir benchmarks/longmemeval/cache")
        print(f"")
        print(f"  # Full benchmark suite (hours)")
        print(f"  python -m benchmarks.eval_harness -b all")
    else:
        print(f"\n  ⚠️ Some tests failed — fix issues before running benchmarks")

    sys.exit(0 if all_pass else 1)
