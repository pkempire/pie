#!/usr/bin/env python3
"""
Experiment runner — decouples extraction from retrieval/answering.

Architecture:
  1. Build world model ONCE per conversation → save to cache
  2. Run N experiments on retrieval+answering in parallel
  3. Each experiment logs: per-question retrieval context, scores, timing

Usage:
  # Step 1: Build and cache world models (slow, do once)
  python run_experiments.py cache --debug

  # Step 2: Run experiments against cached world models (fast, do many times)
  python run_experiments.py run --experiment baseline
  python run_experiments.py run --experiment concise_prompt
  python run_experiments.py run --experiment type_filter
  python run_experiments.py run --experiment top_k_10

  # Step 3: Compare all experiments
  python run_experiments.py compare

  # Run ALL experiments in parallel:
  python run_experiments.py run-all
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
import hashlib
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path
from collections import Counter, defaultdict
from dataclasses import dataclass, field, asdict
from typing import Any

sys.path.insert(0, str(Path(__file__).parent))


# ── World Model Cache ────────────────────────────────────────────────────────

CACHE_DIR = Path("benchmarks/locomo/cache")
RESULTS_DIR = Path("benchmarks/results/experiments")


def cache_world_models(debug: bool = False, num_convos: int = 0):
    """Build and cache world models for all LoCoMo conversations."""
    from benchmarks.locomo.adapter import load_dataset, flatten_qa
    from benchmarks.locomo.baselines import _build_world_model_for_conversation
    from pie.core.llm import LLMClient

    CACHE_DIR.mkdir(parents=True, exist_ok=True)

    data = load_dataset()
    qa_items = flatten_qa(data)

    # Group by conversation
    by_convo = defaultdict(list)
    for item in qa_items:
        by_convo[item["sample_id"]].append(item)

    if num_convos > 0:
        by_convo = dict(list(by_convo.items())[:num_convos])

    llm = LLMClient()

    for convo_id, items in by_convo.items():
        cache_path = CACHE_DIR / f"{convo_id}_wm.json"
        questions_path = CACHE_DIR / f"{convo_id}_questions.json"

        if cache_path.exists():
            # Verify cached file actually has entities (guard against empty saves)
            try:
                with open(cache_path) as f:
                    cached = json.load(f)
                n_cached = len(cached.get("entities", {}))
                if n_cached > 0:
                    print(f"  {convo_id}: cached ({n_cached} entities)")
                    # Always update questions
                    with open(questions_path, "w") as f:
                        json.dump([{
                            "question_id": q["question_id"],
                            "question_type": q["question_type"],
                            "question": q["question"],
                            "answer": str(q["answer"]),
                            "sample_id": q["sample_id"],
                        } for q in items], f, indent=2)
                    continue
                else:
                    print(f"  {convo_id}: cached file EMPTY — rebuilding...")
                    cache_path.unlink()
            except Exception:
                print(f"  {convo_id}: cached file CORRUPT — rebuilding...")
                cache_path.unlink()

        print(f"\n  {convo_id}: building world model ({len(items)} questions)...")
        t0 = time.time()
        extraction_log = []
        wm = _build_world_model_for_conversation(
            items[0], llm, "gpt-4o-mini",
            debug=debug,
            debug_log=extraction_log if debug else None,
        )
        elapsed = time.time() - t0
        print(f"  → {len(wm.entities)} entities, {len(wm.transitions)} transitions in {elapsed:.1f}s")

        # Save world model
        wm.persist_path = cache_path
        wm.save()

        # Verify save was successful
        with open(cache_path) as f:
            verify = json.load(f)
        n_saved = len(verify.get("entities", {}))
        if n_saved != len(wm.entities):
            print(f"  ⚠ WARNING: saved {n_saved} entities but WM has {len(wm.entities)}!")
        else:
            print(f"  → Cached to {cache_path} ({n_saved} entities verified)")

        # Save questions
        with open(questions_path, "w") as f:
            json.dump([{
                "question_id": q["question_id"],
                "question_type": q["question_type"],
                "question": q["question"],
                "answer": str(q["answer"]),
                "sample_id": q["sample_id"],
            } for q in items], f, indent=2)

        # Save extraction log
        log_path = CACHE_DIR / f"{convo_id}_extraction_log.json"
        with open(log_path, "w") as f:
            json.dump(extraction_log, f, indent=2, default=str)

    print(f"\nAll world models cached in {CACHE_DIR}")


# ── Experiment Definitions ───────────────────────────────────────────────────

EXPERIMENTS = {
    "baseline": {
        "description": "Current defaults: top_k=25, gpt-4o answer, hybrid BM25+embedding",
        "top_k": 25,
        "answer_model": "gpt-4o",
        "answer_prompt": "default",
        "type_filter": False,
        "ablation": None,
    },
    "concise_v2": {
        "description": "Ultra-concise answer prompt, force short answers",
        "top_k": 25,
        "answer_model": "gpt-4o",
        "answer_prompt": "concise_v2",
        "type_filter": False,
        "ablation": None,
    },
    "type_filter": {
        "description": "Filter entities by inferred question type before retrieval",
        "top_k": 25,
        "answer_model": "gpt-4o",
        "answer_prompt": "default",
        "type_filter": True,
        "ablation": None,
    },
    "top_k_10": {
        "description": "Fewer entities (top_k=10) — less noise",
        "top_k": 10,
        "answer_model": "gpt-4o",
        "answer_prompt": "default",
        "type_filter": False,
        "ablation": None,
    },
    "top_k_40": {
        "description": "More entities (top_k=40) — more coverage",
        "top_k": 40,
        "answer_model": "gpt-4o",
        "answer_prompt": "default",
        "type_filter": False,
        "ablation": None,
    },
    "no_bm25": {
        "description": "Embedding-only retrieval (no BM25)",
        "top_k": 25,
        "answer_model": "gpt-4o",
        "answer_prompt": "default",
        "type_filter": False,
        "ablation": "no-bm25",
    },
    "no_timeline": {
        "description": "No timeline in context (flat facts only)",
        "top_k": 25,
        "answer_model": "gpt-4o",
        "answer_prompt": "default",
        "type_filter": False,
        "ablation": "no-timeline",
    },
    "mini_answer": {
        "description": "Use gpt-4o-mini for answering (cheaper, faster)",
        "top_k": 25,
        "answer_model": "gpt-4o-mini",
        "answer_prompt": "default",
        "type_filter": False,
        "ablation": None,
    },
    "aggressive_answer": {
        "description": "Never say IDK — always attempt an answer",
        "top_k": 25,
        "answer_model": "gpt-4o",
        "answer_prompt": "never_idk",
        "type_filter": False,
        "ablation": None,
    },
}


# ── Answer Prompts ───────────────────────────────────────────────────────────

ANSWER_PROMPTS = {
    "default": """\
You are answering questions about a conversation between two people.
You are given structured knowledge extracted from their chat history.

## ANSWER FORMAT
- Be CONCISE. For factual questions, answer in 1-2 sentences max.
- Single word/name/date/phrase answers are PREFERRED:
  Q: "When did X happen?" → "July 2023"
  Q: "What is X's pet?" → "A dog named Rex"
  Q: "Where did X move from?" → "Sweden"
  Q: "What is X's relationship status?" → "Single"
- Use absolute dates from the Timeline when available (e.g. "July 6, 2023").
- For list questions ("what activities", "what hobbies"), scan ALL entities and combine everything into one comma-separated list.

## ADVERSARIAL / TRICKY QUESTIONS
- The question may DELIBERATELY name the WRONG person. If the question asks about Person A but the context only has that fact about Person B → ANSWER ANYWAY using Person B's information.
- NEVER say "I don't know" just because the person's name doesn't match. Look for the FACT across ALL entities.

## TEMPORAL REASONING
- The MOST RECENT state is the current truth
- State changes marked with ⚠ mean the NEWER value replaced the old one
- Use Timeline dates to answer "when" questions

## CRITICAL RULES
1. NEVER say "I don't know", "the context does not provide", "not mentioned", or "no information". FORBIDDEN.
2. If ANY entity has even PARTIAL information, USE IT.
3. Make reasonable inferences. "tough breakup" → likely "single".
4. Search ALL entities for relevant facts, not just entities matching the person named in the question.
5. For list questions, aggregate from EVERY entity.
6. If truly zero relevant info, give your best guess from context clues.""",

    "concise_v2": """\
Answer questions about a conversation between two people using the knowledge base below.

RULES:
- Give the SHORTEST possible correct answer.
- For "when" questions: give just the date (e.g., "July 2023" or "June 5, 2023").
- For "what" questions: give just the fact (e.g., "pottery, painting, camping").
- For "who" questions: give just the name.
- Convert relative dates to absolute using timeline dates. "yesterday" on July 2 = "July 1, 2023".
- If information is in the context, USE IT. Do not say "I don't know" unless zero relevant info.
- For lists, aggregate from ALL entities about that person.""",

    "never_idk": """\
You are answering questions about a conversation between two people.
You are given structured knowledge extracted from their chat history.

CRITICAL: You must ALWAYS provide your best answer. NEVER say "I don't know" or
"the context does not provide". If the context has any related information at all,
use it to construct an answer. Make reasonable inferences.

Be concise — 1-2 sentences max for factual questions.
Convert all relative dates to absolute dates using the timeline dates shown.
For list questions, aggregate from ALL entities about that person.""",
}


# ── Run Single Experiment ────────────────────────────────────────────────────

@dataclass
class QuestionResult:
    question_id: str
    question_type: str
    question: str
    gold: str
    predicted: str
    match: bool
    retrieval_count: int
    context_chars: int
    context_tokens_est: int  # estimated tokens (chars/4)
    time_s: float
    retrieved_entities: list[dict] = field(default_factory=list)  # name, score, type
    context_preview: str = ""  # first 500 chars of context


def run_experiment(experiment_name: str, config: dict, num_convos: int = 0):
    """Run a single experiment against cached world models."""
    from pie.core.world_model import WorldModel
    from pie.core.llm import LLMClient
    from benchmarks.locomo.baselines import (
        _retrieve_entities_for_question, _compile_temporal_context,
    )

    # Set ablation env vars
    if config.get("ablation"):
        os.environ["PIE_ABLATION"] = config["ablation"]
    else:
        os.environ.pop("PIE_ABLATION", None)

    os.environ["PIE_TOP_K"] = str(config["top_k"])

    llm = LLMClient()
    answer_prompt = ANSWER_PROMPTS[config["answer_prompt"]]
    answer_model = config["answer_model"]

    # Find cached world models
    wm_files = sorted(CACHE_DIR.glob("*_wm.json"))
    if num_convos > 0:
        wm_files = wm_files[:num_convos]

    all_results: list[QuestionResult] = []
    t_total = time.time()

    for wm_path in wm_files:
        convo_id = wm_path.stem.replace("_wm", "")
        questions_path = CACHE_DIR / f"{convo_id}_questions.json"

        if not questions_path.exists():
            print(f"  {convo_id}: no questions file, skipping")
            continue

        # Load world model
        wm = WorldModel(persist_path=wm_path)
        with open(questions_path) as f:
            questions = json.load(f)

        print(f"  {convo_id}: {len(wm.entities)} entities, {len(questions)} questions")

        for q in questions:
            t0 = time.time()

            # Retrieve
            retrieved = _retrieve_entities_for_question(
                question=q["question"],
                world_model=wm,
                llm=llm,
                top_k=config["top_k"],
            )

            # Compile context
            context = _compile_temporal_context(
                retrieved=retrieved,
                world_model=wm,
            )

            # Answer
            messages = [
                {"role": "system", "content": answer_prompt},
                {"role": "user", "content": f"Knowledge base:\n\n{context}\n\n---\n\nQuestion: {q['question']}\n\nAnswer:"},
            ]
            result = llm.chat(messages=messages, model=answer_model, max_tokens=500)
            predicted = result["content"].strip()

            elapsed = time.time() - t0

            # Rough match (with punctuation stripping)
            import re as _re_match
            def _norm(t):
                t = t.lower().strip()
                t = _re_match.sub(r'["\'\.\,\!\?\;\:\(\)\[\]\{\}]', ' ', t)
                t = _re_match.sub(r'\s+', ' ', t).strip()
                return t
            gold_lower = _norm(q["answer"])
            pred_lower = _norm(predicted)
            match = gold_lower in pred_lower or any(
                w in pred_lower for w in gold_lower.split() if len(w) > 3
            )

            # Log retrieved entities
            retrieved_info = [
                {"name": entity.name, "type": entity.type.value, "score": round(score, 4)}
                for eid, entity, score in retrieved[:10]
            ]

            all_results.append(QuestionResult(
                question_id=q["question_id"],
                question_type=q["question_type"],
                question=q["question"],
                gold=q["answer"],
                predicted=predicted,
                match=match,
                retrieval_count=len(retrieved),
                context_chars=len(context),
                context_tokens_est=len(context) // 4,
                time_s=round(elapsed, 2),
                retrieved_entities=retrieved_info,
                context_preview=context[:500],
            ))

    total_time = time.time() - t_total

    # Save results
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    out_path = RESULTS_DIR / f"{experiment_name}_{int(time.time())}.json"

    # Compute summary
    type_counts = Counter(r.question_type for r in all_results)
    type_pass = Counter(r.question_type for r in all_results if r.match)

    summary = {
        "experiment": experiment_name,
        "config": config,
        "total_questions": len(all_results),
        "total_time_s": round(total_time, 1),
        "accuracy": {
            "overall": f"{sum(r.match for r in all_results)}/{len(all_results)}",
            "by_type": {
                qt: f"{type_pass.get(qt, 0)}/{type_counts[qt]} ({100*type_pass.get(qt,0)/type_counts[qt]:.0f}%)"
                for qt in sorted(type_counts.keys())
            },
        },
        "context_stats": {
            "avg_chars": sum(r.context_chars for r in all_results) // max(len(all_results), 1),
            "avg_tokens_est": sum(r.context_tokens_est for r in all_results) // max(len(all_results), 1),
            "max_chars": max((r.context_chars for r in all_results), default=0),
        },
        "failure_modes": _classify_failures(all_results),
    }

    with open(out_path, "w") as f:
        json.dump({
            "summary": summary,
            "results": [asdict(r) for r in all_results],
        }, f, indent=2, default=str)

    # Print summary
    print(f"\n{'='*70}")
    print(f"EXPERIMENT: {experiment_name}")
    print(f"  {config['description']}")
    print(f"{'='*70}")
    total_pass = sum(r.match for r in all_results)
    print(f"  Overall: {total_pass}/{len(all_results)} = {100*total_pass/max(len(all_results),1):.0f}%")
    for qt in sorted(type_counts.keys()):
        p = type_pass.get(qt, 0)
        t = type_counts[qt]
        print(f"  {qt:15s}: {p:3d}/{t:3d} = {100*p/t:.0f}%")
    print(f"  Avg context: {summary['context_stats']['avg_chars']} chars (~{summary['context_stats']['avg_tokens_est']} tokens)")
    fm = summary["failure_modes"]
    print(f"  Failures: {fm.get('idk',0)} IDK, {fm.get('wrong',0)} wrong, {fm.get('truncated',0)} truncated")
    print(f"  Time: {total_time:.0f}s")
    print(f"  Results: {out_path}")

    return summary


def _classify_failures(results: list[QuestionResult]) -> dict:
    """Classify failure modes."""
    modes = {"idk": 0, "wrong": 0, "truncated": 0}
    for r in results:
        if r.match or r.question_type == "commonsense":
            continue
        pred = r.predicted.lower()
        if "i don't know" in pred or "does not provide" in pred or "no information" in pred:
            modes["idk"] += 1
        elif len(r.predicted) > 480:
            modes["truncated"] += 1
        else:
            modes["wrong"] += 1
    return modes


# ── Compare Experiments ──────────────────────────────────────────────────────

def compare_experiments():
    """Compare all experiment results."""
    result_files = sorted(RESULTS_DIR.glob("*.json"))
    if not result_files:
        print("No experiment results found.")
        return

    # Group by experiment name (take latest for each)
    latest = {}
    for f in result_files:
        name = "_".join(f.stem.split("_")[:-1])  # strip timestamp
        latest[name] = f

    print(f"\n{'='*90}")
    print(f"EXPERIMENT COMPARISON")
    print(f"{'='*90}")
    print(f"{'Experiment':<20s} {'Overall':>8s} {'Single':>8s} {'Multi':>8s} {'Temporal':>8s} {'Adver':>8s} {'Ctx':>6s} {'IDK':>5s}")
    print(f"{'─'*90}")

    for name in sorted(latest.keys()):
        with open(latest[name]) as f:
            data = json.load(f)
        s = data["summary"]
        by_type = s["accuracy"]["by_type"]

        def _pct(s):
            if "(" in s:
                return s.split("(")[1].rstrip(")")
            parts = s.split("/")
            if len(parts) == 2:
                n, d = int(parts[0]), int(parts[1])
                return f"{100*n/d:.0f}%" if d else "N/A"
            return s

        print(f"{name:<20s} "
              f"{_pct(s['accuracy']['overall']):>8s} "
              f"{_pct(by_type.get('single_hop', 'N/A')):>8s} "
              f"{_pct(by_type.get('multi_hop', 'N/A')):>8s} "
              f"{_pct(by_type.get('temporal', 'N/A')):>8s} "
              f"{_pct(by_type.get('adversarial', 'N/A')):>8s} "
              f"{s['context_stats']['avg_tokens_est']:>5d}t "
              f"{s['failure_modes'].get('idk', 0):>4d}")

    print(f"\nDetailed results in {RESULTS_DIR}/")


# ── Run All Experiments ──────────────────────────────────────────────────────

def run_all_experiments(num_convos: int = 0):
    """Run all experiments sequentially (parallel requires separate processes)."""
    for name, config in EXPERIMENTS.items():
        print(f"\n{'━'*70}")
        print(f"  Starting experiment: {name}")
        print(f"{'━'*70}")
        try:
            run_experiment(name, config, num_convos=num_convos)
        except Exception as e:
            print(f"  FAILED: {e}")
            import traceback
            traceback.print_exc()

    compare_experiments()


# ── Trace Mode: Inspect what context the model sees ──────────────────────────

def trace_questions(question_substr: str = "", question_type: str = "", num_convos: int = 0):
    """Trace retrieval + context for specific questions. Shows everything."""
    from pie.core.world_model import WorldModel
    from pie.core.llm import LLMClient
    from benchmarks.locomo.baselines import (
        _retrieve_entities_for_question, _compile_temporal_context,
    )

    llm = LLMClient()

    wm_files = sorted(CACHE_DIR.glob("*_wm.json"))
    if num_convos > 0:
        wm_files = wm_files[:num_convos]

    for wm_path in wm_files:
        convo_id = wm_path.stem.replace("_wm", "")
        questions_path = CACHE_DIR / f"{convo_id}_questions.json"
        if not questions_path.exists():
            continue

        wm = WorldModel(persist_path=wm_path)
        with open(questions_path) as f:
            questions = json.load(f)

        for q in questions:
            if question_substr and question_substr.lower() not in q["question"].lower():
                continue
            if question_type and q["question_type"] != question_type:
                continue

            print(f"\n{'='*80}")
            print(f"Q: {q['question']}")
            print(f"Type: {q['question_type']} | Gold: {q['answer']}")
            print(f"{'='*80}")

            # Retrieve
            retrieved = _retrieve_entities_for_question(
                question=q["question"],
                world_model=wm,
                llm=llm,
                top_k=25,
            )

            print(f"\n── RETRIEVED ENTITIES ({len(retrieved)}) ──")
            for i, (eid, entity, score) in enumerate(retrieved[:15]):
                state = entity.current_state
                desc = state.get("description", str(state)[:100]) if isinstance(state, dict) else str(state)[:100]
                print(f"  [{i+1:2d}] {score:.4f} | {entity.name} ({entity.type.value})")
                print(f"       State: {desc[:120]}")

            # Compile context
            context = _compile_temporal_context(
                retrieved=retrieved,
                world_model=wm,
            )

            print(f"\n── CONTEXT ({len(context)} chars, ~{len(context)//4} tokens) ──")
            print(context[:2000])
            if len(context) > 2000:
                print(f"\n  ... ({len(context) - 2000} more chars)")


# ── Main ─────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="PIE Experiment Runner",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
WORKFLOW:
  # 1. Cache world models (do once, ~5 min per conversation)
  python run_experiments.py cache --debug

  # 2. Run experiments (fast, ~30s each against cache)
  python run_experiments.py run --experiment baseline
  python run_experiments.py run --experiment never_idk
  python run_experiments.py run --experiment top_k_10

  # 3. Run ALL experiments
  python run_experiments.py run-all

  # 4. Compare results
  python run_experiments.py compare

  # 5. Trace specific questions (see exact retrieval + context)
  python run_experiments.py trace --question "picnic"
  python run_experiments.py trace --type multi_hop
""",
    )
    sub = parser.add_subparsers(dest="command")

    # cache
    p_cache = sub.add_parser("cache", help="Build and cache world models")
    p_cache.add_argument("--debug", action="store_true")
    p_cache.add_argument("--num-convos", type=int, default=0)

    # run
    p_run = sub.add_parser("run", help="Run a single experiment")
    p_run.add_argument("--experiment", required=True, choices=list(EXPERIMENTS.keys()))
    p_run.add_argument("--num-convos", type=int, default=0)

    # run-all
    p_all = sub.add_parser("run-all", help="Run all experiments")
    p_all.add_argument("--num-convos", type=int, default=0)

    # compare
    sub.add_parser("compare", help="Compare all experiment results")

    # trace
    p_trace = sub.add_parser("trace", help="Trace retrieval for specific questions")
    p_trace.add_argument("--question", default="", help="Substring to match in question text")
    p_trace.add_argument("--type", default="", help="Question type filter")
    p_trace.add_argument("--num-convos", type=int, default=0)

    # list
    sub.add_parser("list", help="List available experiments")

    args = parser.parse_args()

    if not args.command:
        parser.print_help()
        return

    if args.command == "list":
        print("Available experiments:")
        for name, config in EXPERIMENTS.items():
            print(f"  {name:<20s} — {config['description']}")
        return

    # Check API key
    if args.command in ("cache", "run", "run-all", "trace"):
        if not os.environ.get("OPENAI_API_KEY"):
            print("ERROR: Set OPENAI_API_KEY")
            sys.exit(1)

    if args.command == "cache":
        cache_world_models(debug=args.debug, num_convos=args.num_convos)

    elif args.command == "run":
        config = EXPERIMENTS[args.experiment]
        run_experiment(args.experiment, config, num_convos=args.num_convos)

    elif args.command == "run-all":
        run_all_experiments(num_convos=args.num_convos)

    elif args.command == "compare":
        compare_experiments()

    elif args.command == "trace":
        trace_questions(
            question_substr=args.question,
            question_type=args.type,
            num_convos=args.num_convos,
        )


if __name__ == "__main__":
    main()
