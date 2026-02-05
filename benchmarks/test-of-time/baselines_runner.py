#!/usr/bin/env python3
"""
Test of Time (ToT) Benchmark - Baselines Runner
=================================================

Runs three baselines on ToT-Semantic:
  1. baseline: Raw temporal facts with dates (as provided in dataset)
  2. naive_rag: Embed facts, retrieve top-k by similarity, answer
  3. pie_temporal: PIE-style semantic temporal narrative reformulation

Usage:
  python baselines_runner.py --limit 50              # run all baselines
  python baselines_runner.py --limit 100 --baseline naive_rag
  python baselines_runner.py --limit 20 --dry-run    # show prompts only
"""

import argparse
import asyncio
import json
import os
import re
import sys
import time
from collections import defaultdict
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Optional
import numpy as np

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

SCRIPT_DIR = Path(__file__).parent
DATA_PATH = SCRIPT_DIR / "data" / "tot_semantic" / "test.json"

# Get API key from environment
def get_api_key():
    """Get OpenAI API key from environment variable."""
    key = os.environ.get("OPENAI_API_KEY")
    if not key:
        raise RuntimeError(
            "OPENAI_API_KEY environment variable not set. "
            "Please set it before running the benchmark."
        )
    return key

# ---------------------------------------------------------------------------
# Data structures
# ---------------------------------------------------------------------------

@dataclass
class TemporalFact:
    """A parsed temporal fact."""
    subject: str
    role: str
    obj: str
    start: int
    end: int
    duration: int = 0
    text: str = ""
    embedding: Optional[list] = None

    def __post_init__(self):
        self.duration = self.end - self.start
        self.text = f"{self.subject} was the {self.role} of {self.obj} from {self.start} to {self.end}."


@dataclass
class BaselineResult:
    """Result from running a baseline on one question."""
    question_id: int
    question_type: str
    question: str
    label: str
    hypothesis: str
    baseline_name: str
    model: str
    correct: bool = False
    latency_ms: float = 0.0
    context_chars: int = 0
    retrieval_count: int = 0
    error: Optional[str] = None


# ---------------------------------------------------------------------------
# Fact parsing
# ---------------------------------------------------------------------------

FACT_PATTERN = re.compile(
    r"(E\d+) was the (R\d+) of (E\d+) from (\d{4}) to (\d{4})\."
)


def parse_facts(prompt: str) -> list[TemporalFact]:
    """Extract all temporal facts from a prompt string."""
    facts = []
    for m in FACT_PATTERN.finditer(prompt):
        facts.append(TemporalFact(
            subject=m.group(1),
            role=m.group(2),
            obj=m.group(3),
            start=int(m.group(4)),
            end=int(m.group(5)),
        ))
    return facts


# ---------------------------------------------------------------------------
# PIE-style semantic temporal reformulation (from runner.py)
# ---------------------------------------------------------------------------

def duration_description(years: int) -> str:
    """Human-readable duration."""
    if years == 0:
        return "less than a year"
    if years == 1:
        return "1 year"
    if years <= 2:
        return f"{years} years (brief)"
    if years <= 5:
        return f"{years} years"
    if years <= 10:
        return f"{years} years (extended)"
    return f"{years} years (long-standing)"


def reformulate_pie_style(facts: list[TemporalFact], question: str) -> str:
    """
    Reformulate temporal facts into PIE's semantic temporal narrative format.
    
    Key transformations:
    1. Group facts by entity → entity timelines
    2. Add duration descriptions (brief, extended, long-standing)
    3. Relative time anchoring between facts
    4. Overlap markers between concurrent facts
    5. Temporal ordering narrative
    6. Key temporal relationships (succession chains)
    """
    if not facts:
        return "No temporal facts provided."

    # Adaptive verbosity
    n_facts = len(facts)
    compact = n_facts > 150
    max_overlaps_per_fact = 1 if compact else 3
    show_full_ordering = not compact

    # Global stats
    all_starts = [f.start for f in facts]
    all_ends = [f.end for f in facts]
    global_start = min(all_starts)
    global_end = max(all_ends)
    total_span = global_end - global_start

    # Group by object entity
    by_object: dict[str, list[TemporalFact]] = defaultdict(list)
    for f in facts:
        by_object[f.obj].append(f)

    lines = []
    lines.append(f"TEMPORAL CONTEXT ({global_start}–{global_end}, spanning {total_span} years)")
    lines.append("")

    # Section 1: Entity Timelines
    lines.append("═══ ENTITY TIMELINES ═══")
    lines.append("")

    for obj_entity in sorted(by_object.keys(), key=lambda e: int(e[1:])):
        obj_facts = sorted(by_object[obj_entity], key=lambda f: (f.start, f.end))
        lines.append(f"▸ Timeline for {obj_entity}:")

        role_holders: dict[str, list[TemporalFact]] = defaultdict(list)
        for f in obj_facts:
            role_holders[f.role].append(f)

        prev_fact = None
        for f in obj_facts:
            dur = duration_description(f.duration)
            line_parts = [f"  [{f.start}–{f.end}] {f.subject} as {f.role} — {dur}"]

            if prev_fact:
                gap = f.start - prev_fact.end
                if gap > 0:
                    line_parts.append(f"({gap}yr gap)")
                elif gap == 0:
                    line_parts.append("(immediately following)")
                else:
                    line_parts.append(f"(began {abs(gap)}yr before prev ended)")

            lines.append(" ".join(line_parts))

            # Check overlaps
            if max_overlaps_per_fact > 0:
                overlapping = []
                for other in obj_facts:
                    if other is f:
                        continue
                    o_start = max(f.start, other.start)
                    o_end = min(f.end, other.end)
                    if o_start < o_end:
                        overlapping.append(
                            f"    ↔ concurrent with {other.subject} as {other.role} ({o_start}–{o_end})"
                        )
                for ol in overlapping[:max_overlaps_per_fact]:
                    lines.append(ol)

            prev_fact = f

        # Summary
        roles_summary = []
        for role, holders in sorted(role_holders.items()):
            holder_names = [h.subject for h in holders]
            roles_summary.append(f"{role}: {', '.join(holder_names)}")
        lines.append(f"  Summary: {'; '.join(roles_summary)}")
        lines.append("")

    # Section 2: Temporal ordering (skip for large)
    if show_full_ordering:
        lines.append("═══ TEMPORAL ORDERING (all facts by start time) ═══")
        lines.append("")

        all_sorted = sorted(facts, key=lambda f: (f.start, f.end))
        for i, f in enumerate(all_sorted):
            seq_parts = [f"  {i+1}. [{f.start}→{f.end}] {f.subject} was {f.role} of {f.obj} ({duration_description(f.duration)})"]
            if i > 0:
                prev = all_sorted[i - 1]
                diff = f.start - prev.start
                if diff == 0:
                    seq_parts.append("— same start as previous")
                else:
                    seq_parts.append(f"— starts {diff}yr after #{i}")
            lines.append(" ".join(seq_parts))
        lines.append("")

    # Section 3: Key temporal relationships
    lines.append("═══ KEY TEMPORAL RELATIONSHIPS ═══")
    lines.append("")

    for obj_entity in sorted(by_object.keys(), key=lambda e: int(e[1:])):
        role_holders = defaultdict(list)
        for f in by_object[obj_entity]:
            role_holders[f.role].append(f)

        for role in sorted(role_holders.keys()):
            holders = sorted(role_holders[role], key=lambda f: f.start)
            if len(holders) >= 2:
                first = holders[0]
                last = holders[-1]
                lines.append(
                    f"  {obj_entity}/{role}: first={first.subject} ({first.start}), "
                    f"last={last.subject} ({last.start}), "
                    f"{len(holders)} total holders"
                )
                # Succession chain
                for j in range(len(holders) - 1):
                    curr = holders[j]
                    nxt = holders[j + 1]
                    gap = nxt.start - curr.end
                    if gap == 0:
                        trans = "immediately succeeded by"
                    elif gap > 0:
                        trans = f"succeeded {gap}yr later by"
                    else:
                        trans = f"overlapped ({abs(gap)}yr) then succeeded by"
                    lines.append(f"    {curr.subject} ({curr.start}–{curr.end}) {trans} {nxt.subject} ({nxt.start}–{nxt.end})")

    lines.append("")

    narrative = "\n".join(lines)

    # Hard cap
    MAX_CHARS = 400_000
    if len(narrative) > MAX_CHARS:
        narrative = narrative[:MAX_CHARS] + "\n... (truncated for length)"

    return narrative


# ---------------------------------------------------------------------------
# LLM Client
# ---------------------------------------------------------------------------

class LLMClient:
    """Simple OpenAI client for chat and embeddings."""
    
    def __init__(self, api_key: str = None):
        from openai import OpenAI
        self.client = OpenAI(api_key=api_key or get_api_key())
        self._embed_cache = {}
    
    def chat(self, messages: list, model: str = "gpt-4o-mini", max_tokens: int = 200, temperature: float = 0.0) -> str:
        response = self.client.chat.completions.create(
            model=model,
            messages=messages,
            temperature=temperature,
            max_tokens=max_tokens,
        )
        return response.choices[0].message.content.strip()
    
    def embed(self, texts: list[str], model: str = "text-embedding-3-small") -> list[list[float]]:
        """Embed a batch of texts."""
        # Check cache
        uncached = []
        uncached_idx = []
        results = [None] * len(texts)
        
        for i, t in enumerate(texts):
            key = (t[:500], model)  # Use truncated text as key
            if key in self._embed_cache:
                results[i] = self._embed_cache[key]
            else:
                uncached.append(t[:8000])  # Truncate for embedding
                uncached_idx.append(i)
        
        if uncached:
            response = self.client.embeddings.create(
                model=model,
                input=uncached,
            )
            for j, emb in enumerate(response.data):
                idx = uncached_idx[j]
                results[idx] = emb.embedding
                key = (texts[idx][:500], model)
                self._embed_cache[key] = emb.embedding
        
        return results
    
    def embed_single(self, text: str, model: str = "text-embedding-3-small") -> list[float]:
        return self.embed([text], model)[0]


def cosine_similarity(a: list[float], b: list[float]) -> float:
    """Compute cosine similarity between two vectors."""
    a = np.array(a)
    b = np.array(b)
    return float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b)))


# ---------------------------------------------------------------------------
# Answer checking
# ---------------------------------------------------------------------------

def normalize_answer(answer: str) -> str:
    """Normalize an answer for comparison."""
    answer = answer.strip()

    # Try JSON extraction
    if answer.startswith('{'):
        try:
            parsed = json.loads(answer)
            if isinstance(parsed, dict) and 'answer' in parsed:
                answer = str(parsed['answer']).strip()
        except json.JSONDecodeError:
            m = re.search(r'"answer"\s*:\s*"([^"]+)"', answer)
            if m:
                answer = m.group(1).strip()
            else:
                m = re.search(r'"answer"\s*:\s*(\d+)', answer)
                if m:
                    answer = m.group(1).strip()

    answer = answer.strip().strip('.').strip('"').strip("'").strip()

    for prefix in ["answer:", "the answer is", "answer is", "final answer:"]:
        if answer.lower().startswith(prefix):
            answer = answer[len(prefix):].strip()

    answer = answer.rstrip('.')
    return answer


def check_answer(predicted: str, label: str) -> bool:
    """Check if predicted answer matches label."""
    pred = normalize_answer(predicted)
    lab = normalize_answer(label)

    if pred.lower() == lab.lower():
        return True

    # For comma-separated lists
    if ',' in lab:
        pred_parts = [p.strip() for p in pred.split(',')]
        lab_parts = [p.strip() for p in lab.split(',')]
        return pred_parts == lab_parts

    return False


# ---------------------------------------------------------------------------
# Baseline 1: Raw facts (original prompt)
# ---------------------------------------------------------------------------

BASELINE_SYSTEM = """You are a precise temporal reasoning assistant. Answer the question based ONLY on the provided temporal facts. Give ONLY the final answer with no explanation.

Rules:
- For entity questions: answer with just the entity ID (e.g., E42)
- For time questions: answer with just the year (e.g., 1985)
- For duration questions: answer with just the number (e.g., 5)
- For counting questions: answer with just the number (e.g., 3)
- For timeline/list questions: answer with a comma-separated list (e.g., E59,E57,E43)
- No extra text, no reasoning, just the answer."""


def run_baseline(item: dict, llm: LLMClient, model: str = "gpt-4o-mini") -> BaselineResult:
    """Run the raw baseline (original prompt with dates)."""
    t0 = time.time()
    
    question = item['question']
    label = str(item['label'])
    prompt = item['prompt']
    
    # Use the original prompt (raw temporal facts)
    user_content = f"{prompt}\n\nQuestion: {question}\nAnswer:"
    
    try:
        answer = llm.chat(
            messages=[
                {"role": "system", "content": BASELINE_SYSTEM},
                {"role": "user", "content": user_content},
            ],
            model=model,
            max_tokens=100,
        )
        
        correct = check_answer(answer, label)
        
        return BaselineResult(
            question_id=hash(question) % 100000,
            question_type=item['question_type'],
            question=question,
            label=label,
            hypothesis=answer,
            baseline_name="baseline",
            model=model,
            correct=correct,
            latency_ms=(time.time() - t0) * 1000,
            context_chars=len(prompt),
        )
    except Exception as e:
        return BaselineResult(
            question_id=hash(question) % 100000,
            question_type=item['question_type'],
            question=question,
            label=label,
            hypothesis=f"Error: {e}",
            baseline_name="baseline",
            model=model,
            correct=False,
            latency_ms=(time.time() - t0) * 1000,
            error=str(e),
        )


# ---------------------------------------------------------------------------
# Baseline 2: Naive RAG
# ---------------------------------------------------------------------------

def run_naive_rag(
    item: dict,
    llm: LLMClient,
    model: str = "gpt-4o-mini",
    top_k: int = 20,
) -> BaselineResult:
    """
    Naive RAG baseline: embed facts, retrieve top-k by similarity, answer.
    """
    t0 = time.time()
    
    question = item['question']
    label = str(item['label'])
    prompt = item['prompt']
    
    try:
        # Parse facts
        facts = parse_facts(prompt)
        
        if not facts:
            return BaselineResult(
                question_id=hash(question) % 100000,
                question_type=item['question_type'],
                question=question,
                label=label,
                hypothesis="No facts to search.",
                baseline_name="naive_rag",
                model=model,
                correct=False,
                latency_ms=(time.time() - t0) * 1000,
            )
        
        # Embed question
        query_emb = llm.embed_single(question)
        
        # Embed all facts
        fact_texts = [f.text for f in facts]
        fact_embeddings = llm.embed(fact_texts)
        
        # Score by cosine similarity
        scored = []
        for fact, emb in zip(facts, fact_embeddings):
            sim = cosine_similarity(query_emb, emb)
            scored.append((fact, sim))
        
        scored.sort(key=lambda x: x[1], reverse=True)
        top_facts = scored[:top_k]
        
        # Sort retrieved facts chronologically
        top_facts_sorted = sorted(top_facts, key=lambda x: (x[0].start, x[0].end))
        
        # Build context from retrieved facts
        context_parts = []
        for fact, score in top_facts_sorted:
            context_parts.append(f"[relevance: {score:.2f}] {fact.text}")
        context = "\n".join(context_parts)
        
        user_content = f"Retrieved temporal facts:\n{context}\n\nQuestion: {question}\nAnswer:"
        
        answer = llm.chat(
            messages=[
                {"role": "system", "content": BASELINE_SYSTEM},
                {"role": "user", "content": user_content},
            ],
            model=model,
            max_tokens=100,
        )
        
        correct = check_answer(answer, label)
        
        return BaselineResult(
            question_id=hash(question) % 100000,
            question_type=item['question_type'],
            question=question,
            label=label,
            hypothesis=answer,
            baseline_name="naive_rag",
            model=model,
            correct=correct,
            latency_ms=(time.time() - t0) * 1000,
            context_chars=len(context),
            retrieval_count=len(top_facts),
        )
    except Exception as e:
        return BaselineResult(
            question_id=hash(question) % 100000,
            question_type=item['question_type'],
            question=question,
            label=label,
            hypothesis=f"Error: {e}",
            baseline_name="naive_rag",
            model=model,
            correct=False,
            latency_ms=(time.time() - t0) * 1000,
            error=str(e),
        )


# ---------------------------------------------------------------------------
# Baseline 3: PIE Temporal
# ---------------------------------------------------------------------------

PIE_SYSTEM = """You are a precise temporal reasoning assistant. Answer the question based ONLY on the provided temporal context. The context has been pre-processed to highlight:
- Entity timelines with durations
- Temporal relationships (succession, overlap)
- Key first/last holders of roles

Give ONLY the final answer with no explanation.

Rules:
- For entity questions: answer with just the entity ID (e.g., E42)
- For time questions: answer with just the year (e.g., 1985)
- For duration questions: answer with just the number (e.g., 5)
- For counting questions: answer with just the number (e.g., 3)
- For timeline/list questions: answer with a comma-separated list (e.g., E59,E57,E43)
- No extra text, no reasoning, just the answer."""


def run_pie_temporal(
    item: dict,
    llm: LLMClient,
    model: str = "gpt-4o-mini",
) -> BaselineResult:
    """
    PIE temporal baseline: reformulate facts into semantic temporal narrative.
    """
    t0 = time.time()
    
    question = item['question']
    label = str(item['label'])
    prompt = item['prompt']
    
    try:
        # Parse and reformulate
        facts = parse_facts(prompt)
        pie_context = reformulate_pie_style(facts, question)
        
        user_content = f"{pie_context}\n\nUsing the temporal context above, answer:\nQuestion: {question}\nAnswer:"
        
        answer = llm.chat(
            messages=[
                {"role": "system", "content": PIE_SYSTEM},
                {"role": "user", "content": user_content},
            ],
            model=model,
            max_tokens=100,
        )
        
        correct = check_answer(answer, label)
        
        return BaselineResult(
            question_id=hash(question) % 100000,
            question_type=item['question_type'],
            question=question,
            label=label,
            hypothesis=answer,
            baseline_name="pie_temporal",
            model=model,
            correct=correct,
            latency_ms=(time.time() - t0) * 1000,
            context_chars=len(pie_context),
        )
    except Exception as e:
        return BaselineResult(
            question_id=hash(question) % 100000,
            question_type=item['question_type'],
            question=question,
            label=label,
            hypothesis=f"Error: {e}",
            baseline_name="pie_temporal",
            model=model,
            correct=False,
            latency_ms=(time.time() - t0) * 1000,
            error=str(e),
        )


# ---------------------------------------------------------------------------
# Main runner
# ---------------------------------------------------------------------------

BASELINES = {
    "baseline": run_baseline,
    "naive_rag": run_naive_rag,
    "pie_temporal": run_pie_temporal,
}


def run_benchmark(
    limit: int = 50,
    question_type: Optional[str] = None,
    baselines: list[str] = None,
    model: str = "gpt-4o-mini",
    dry_run: bool = False,
):
    """Run the benchmark with specified baselines."""
    
    baselines = baselines or ["baseline", "naive_rag", "pie_temporal"]
    
    # Load data
    print(f"Loading data from {DATA_PATH}...")
    with open(DATA_PATH) as f:
        all_items = json.load(f)
    
    print(f"Total items: {len(all_items)}")
    
    # Filter by question type
    if question_type:
        all_items = [item for item in all_items if item['question_type'] == question_type]
        print(f"Filtered to {len(all_items)} items of type '{question_type}'")
    
    # Sample evenly across question types
    if not question_type:
        by_type = defaultdict(list)
        for item in all_items:
            by_type[item['question_type']].append(item)
        
        per_type = max(1, limit // len(by_type))
        remainder = limit - per_type * len(by_type)
        
        selected = []
        for qt in sorted(by_type.keys()):
            n = per_type + (1 if remainder > 0 else 0)
            remainder -= 1
            selected.extend(by_type[qt][:n])
        
        items = selected[:limit]
    else:
        items = all_items[:limit]
    
    print(f"Running {len(items)} questions")
    print(f"Model: {model}")
    print(f"Baselines: {baselines}")
    print(f"Dry run: {dry_run}")
    print()
    
    if dry_run:
        # Show sample reformulations
        for i, item in enumerate(items[:3]):
            facts = parse_facts(item['prompt'])
            pie = reformulate_pie_style(facts, item['question'])
            print(f"{'='*80}")
            print(f"[{i+1}] Question ({item['question_type']}): {item['question']}")
            print(f"    Label: {item['label']}")
            print(f"    Facts: {len(facts)}")
            print()
            print("--- BASELINE PROMPT (first 500 chars) ---")
            print(item['prompt'][:500] + "...")
            print()
            print("--- PIE PROMPT (first 1500 chars) ---")
            print(pie[:1500] + "...")
            print()
        return
    
    # Initialize LLM client
    llm = LLMClient()
    
    # Run baselines
    all_results: dict[str, list[BaselineResult]] = {b: [] for b in baselines}
    
    start_time = time.time()
    
    for idx, item in enumerate(items):
        for baseline_name in baselines:
            baseline_fn = BASELINES[baseline_name]
            result = baseline_fn(item, llm, model)
            all_results[baseline_name].append(result)
            
            status = "✓" if result.correct else "✗"
            print(f"  [{idx+1}/{len(items)}] {result.question_type[:20]:20s} | "
                  f"{baseline_name:12s}: {status} | "
                  f"pred={result.hypothesis[:30]:30s} | "
                  f"label={result.label}")
    
    elapsed = time.time() - start_time
    
    # ---------------------------------------------------------------------------
    # Results summary
    # ---------------------------------------------------------------------------
    print()
    print("=" * 100)
    print("RESULTS SUMMARY")
    print("=" * 100)
    
    # Overall by baseline
    print(f"\n{'Baseline':<15} {'Correct':>10} {'Total':>10} {'Accuracy':>12}")
    print("-" * 50)
    
    for baseline_name in baselines:
        results = all_results[baseline_name]
        correct = sum(1 for r in results if r.correct)
        total = len(results)
        acc = correct / total * 100 if total else 0
        print(f"  {baseline_name:<13} {correct:>10} {total:>10} {acc:>10.1f}%")
    
    # By question type
    print(f"\n\nBreakdown by Question Type:")
    print(f"{'Question Type':<45}", end="")
    for b in baselines:
        print(f" {b[:10]:>10}", end="")
    print()
    print("-" * (45 + len(baselines) * 11))
    
    question_types = sorted(set(r.question_type for r in all_results[baselines[0]]))
    
    for qt in question_types:
        print(f"  {qt:<43}", end="")
        for baseline_name in baselines:
            results = [r for r in all_results[baseline_name] if r.question_type == qt]
            correct = sum(1 for r in results if r.correct)
            total = len(results)
            acc = correct / total * 100 if total else 0
            print(f" {acc:>9.1f}%", end="")
        print()
    
    # Comparison with published results
    print(f"\n\n{'='*100}")
    print("COMPARISON WITH PUBLISHED RESULTS (ToT Paper - ICLR 2025)")
    print("="*100)
    print("""
Published results on ToT-Semantic (GPT-4 Turbo):
  - Complete graphs: ~40% accuracy
  - ER graphs: ~45-55% accuracy  
  - AWE graphs (wikidata-style): ~90% accuracy
  - SFN (scale-free): ~55% accuracy
  - Star graphs: ~65% accuracy

Note: Our dataset uses ER graph generation primarily.
Typical GPT-4o-mini performance: 35-50% on ER graphs.

Question type difficulty (from paper):
  - event_at_what_time: Easiest (simple lookup)
  - event_at_time_t: Moderate (requires time arithmetic)
  - timeline: Hardest (multi-entity ordering)
""")
    
    print(f"\nElapsed: {elapsed:.1f}s ({elapsed/len(items):.1f}s per question × {len(baselines)} baselines)")
    
    # Save results
    results_dir = SCRIPT_DIR / "results"
    results_dir.mkdir(exist_ok=True)
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    qt_suffix = f"_{question_type}" if question_type else ""
    results_file = results_dir / f"baselines_{timestamp}_n{len(items)}{qt_suffix}.json"
    
    output = {
        "meta": {
            "timestamp": timestamp,
            "model": model,
            "total_questions": len(items),
            "question_type_filter": question_type,
            "baselines": baselines,
            "elapsed_seconds": round(elapsed, 1),
        },
        "overall": {},
        "by_type": {},
        "details": {},
    }
    
    for baseline_name in baselines:
        results = all_results[baseline_name]
        correct = sum(1 for r in results if r.correct)
        total = len(results)
        acc = correct / total * 100 if total else 0
        
        output["overall"][baseline_name] = {
            "correct": correct,
            "total": total,
            "accuracy": round(acc, 2),
        }
        
        output["by_type"][baseline_name] = {}
        for qt in question_types:
            qt_results = [r for r in results if r.question_type == qt]
            qt_correct = sum(1 for r in qt_results if r.correct)
            qt_total = len(qt_results)
            output["by_type"][baseline_name][qt] = {
                "correct": qt_correct,
                "total": qt_total,
                "accuracy": round(qt_correct / qt_total * 100, 2) if qt_total else 0,
            }
        
        output["details"][baseline_name] = [
            {
                "question_type": r.question_type,
                "question": r.question,
                "label": r.label,
                "hypothesis": r.hypothesis,
                "correct": r.correct,
                "latency_ms": round(r.latency_ms, 1),
                "context_chars": r.context_chars,
                "retrieval_count": r.retrieval_count,
            }
            for r in results
        ]
    
    with open(results_file, 'w') as f:
        json.dump(output, f, indent=2)
    
    print(f"\nResults saved to: {results_file}")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="ToT Benchmark: Compare baselines (raw, naive_rag, pie_temporal)"
    )
    parser.add_argument(
        "--limit", type=int, default=50,
        help="Number of questions to evaluate (default: 50)"
    )
    parser.add_argument(
        "--question-type", type=str, default=None,
        choices=[
            "before_after", "event_at_time_t", "event_at_what_time",
            "first_last", "event_at_the_time_of_another_event",
            "number_of_events_in_time_interval", "relation_duration", "timeline"
        ],
        help="Filter to a specific question type"
    )
    parser.add_argument(
        "--baseline", type=str, default=None,
        choices=["baseline", "naive_rag", "pie_temporal"],
        help="Run only a specific baseline (default: all)"
    )
    parser.add_argument(
        "--dry-run", action="store_true",
        help="Show reformulated prompts without making LLM calls"
    )
    parser.add_argument(
        "--model", type=str, default="gpt-4o-mini",
        help="OpenAI model to use (default: gpt-4o-mini)"
    )

    args = parser.parse_args()
    
    baselines = [args.baseline] if args.baseline else None
    
    run_benchmark(
        limit=args.limit,
        question_type=args.question_type,
        baselines=baselines,
        model=args.model,
        dry_run=args.dry_run,
    )


if __name__ == "__main__":
    main()
