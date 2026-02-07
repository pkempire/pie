#!/usr/bin/env python3
"""
Hybrid Temporal Context Runner for ToT Benchmark
=================================================
Tests the hypothesis that HYBRID context (raw + derived + semantic) beats both
pure raw and pure semantic approaches.

Three conditions:
  A (baseline): Raw temporal facts
  B (pie_semantic): Semantic reformulation only (current PIE)
  C (hybrid): Raw facts + derived facts + optional semantic context

Usage:
  python hybrid_runner.py --limit 50 --dry-run
  python hybrid_runner.py --limit 100
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

# Import from existing runner
from runner import (
    TemporalFact,
    parse_facts,
    normalize_answer,
    check_answer,
    SYSTEM_PROMPT,
)

# ---------------------------------------------------------------------------
# Query Classification
# ---------------------------------------------------------------------------

def classify_query(question: str) -> str:
    """Classify temporal query type to determine optimal context strategy."""
    q = question.lower()
    
    # Compound temporal queries (need broader context)
    # Pattern: "At the [start/end] time of X, Y was..."
    if ("at the" in q and "time of" in q) or ("at the start" in q) or ("at the end" in q):
        return "compound_time"
    
    # Point-in-time lookup (needs exact dates)
    if re.search(r'in \d{4}', q) or re.search(r'during \d{4}', q):
        return "point_in_time"
    
    # End time queries
    if any(kw in q for kw in ["stop", "end", "cease", "no longer", "finish"]):
        return "end_time"
    
    # Start time queries  
    if any(kw in q for kw in ["start", "begin", "first time", "initially"]):
        return "start_time"
    
    # Ordering queries
    if any(kw in q for kw in ["first", "last", "before", "after", "earlier", "later"]):
        return "ordering"
    
    # Duration queries
    if any(kw in q for kw in ["how long", "duration", "years", "span", "period"]):
        return "duration"
    
    # Count queries
    if any(kw in q for kw in ["how many", "count", "number of"]):
        return "count"
    
    # Timeline queries
    if "timeline" in q or "sequence" in q or "order" in q:
        return "timeline"
    
    return "general"


def is_compound_query(question: str) -> bool:
    """Check if this is a compound temporal query needing broader context."""
    q = question.lower()
    return (
        ("at the" in q and "time of" in q) or
        ("at the start" in q) or 
        ("at the end" in q) or
        q.count(",") >= 1  # Multiple clauses
    )


# ---------------------------------------------------------------------------
# Derived Fact Generation
# ---------------------------------------------------------------------------

def derive_facts(fact: TemporalFact, query_years: set[int] = None) -> list[str]:
    """Generate explicit derived facts from a temporal fact.
    
    Args:
        fact: The temporal fact
        query_years: Optional set of years mentioned in the query to explicitly state
    """
    derived = []
    
    # Explicit start/stop times (CRITICAL for temporal lookup)
    derived.append(f"{fact.subject} started being the {fact.role} of {fact.obj} in {fact.start}.")
    derived.append(f"{fact.subject} stopped being the {fact.role} of {fact.obj} in {fact.end}.")
    
    # Duration
    derived.append(f"{fact.subject} was the {fact.role} of {fact.obj} for {fact.duration} years ({fact.start}-{fact.end}).")
    
    # Point-in-time assertions for boundary years
    derived.append(f"In {fact.start}, {fact.subject} was the {fact.role} of {fact.obj}.")
    if fact.end != fact.start:
        derived.append(f"In {fact.end}, {fact.subject} was the {fact.role} of {fact.obj}.")
    
    # For any query-relevant years within this fact's range, add explicit assertions
    if query_years:
        for year in query_years:
            if fact.start <= year <= fact.end and year not in {fact.start, fact.end}:
                derived.append(f"In {year}, {fact.subject} was the {fact.role} of {fact.obj}.")
    
    return derived


def extract_years_from_question(question: str) -> set[int]:
    """Extract any years mentioned in the question."""
    years = set()
    for match in re.findall(r'\b(19\d{2}|20\d{2})\b', question):
        years.add(int(match))
    return years


# ---------------------------------------------------------------------------
# Entity Filtering  
# ---------------------------------------------------------------------------

def extract_entities_from_question(question: str) -> tuple[set[str], set[str]]:
    """Extract entity IDs and role IDs from a question."""
    entities = set(re.findall(r'E\d+', question))
    roles = set(re.findall(r'R\d+', question))
    return entities, roles


def filter_facts_by_relevance(facts: list[TemporalFact], question: str) -> list[TemporalFact]:
    """Filter facts to only those relevant to the question.
    
    Direction-aware prioritization:
    1. Exact structure match: subject=E1, role=R, object=E2 when question asks about E1 as R of E2
    2. Facts involving role AND at least one entity
    3. Facts involving any mentioned entity
    
    For compound queries, use broader filtering to capture multi-hop reasoning.
    """
    entities, roles = extract_entities_from_question(question)
    
    if not entities and not roles:
        return facts  # Can't filter, return all
    
    # Compound queries need broader context
    compound = is_compound_query(question)
    
    # Try to detect direction from question
    subject_hint = None
    object_hint = None
    
    entity_list = sorted(entities, key=lambda e: question.find(e))
    
    if len(entity_list) >= 2 and not compound:
        # For simple queries, use direction hints
        subject_hint = entity_list[0]
        object_hint = entity_list[-1]
    
    # Tier 0: EXACT structure match
    tier0 = []
    # Tier 1: Good structural match  
    tier1 = []
    # Tier 2: Role + entity
    tier2 = []
    # Tier 3: Any entity
    tier3 = []
    
    for f in facts:
        entities_in_fact = {f.subject, f.obj}
        matching_entities = entities & entities_in_fact
        role_match = f.role in roles
        
        # Check for exact structural match
        exact_match = (
            subject_hint and object_hint and role_match and
            f.subject == subject_hint and f.obj == object_hint
        )
        
        if exact_match:
            tier0.append(f)
        elif role_match and len(matching_entities) >= 2:
            tier1.append(f)
        elif role_match and matching_entities:
            tier2.append(f)
        elif len(matching_entities) >= 2:
            tier1.append(f)
        elif matching_entities:
            tier3.append(f)
    
    # Combine tiers in priority order
    relevant = tier0 + tier1 + tier2 + tier3
    
    # Cap at reasonable size — larger for compound queries
    max_facts = 150 if compound else 80
    if len(relevant) > max_facts:
        priority = tier0 + tier1
        if len(priority) >= max_facts:
            relevant = priority[:max_facts]
        else:
            remaining = max_facts - len(priority)
            relevant = priority + (tier2 + tier3)[:remaining]
    
    return relevant if relevant else facts[:50]


# ---------------------------------------------------------------------------
# Hybrid Context Builder
# ---------------------------------------------------------------------------

def build_hybrid_context(facts: list[TemporalFact], question: str, include_semantic: bool = True) -> str:
    """Build hybrid context with raw + derived + optional semantic layers."""
    
    qtype = classify_query(question)
    relevant_facts = filter_facts_by_relevance(facts, question)
    compound = is_compound_query(question)
    
    # Extract years from question and from key facts
    query_years = extract_years_from_question(question)
    
    # For compound queries, also extract start/end years from primary relationship facts
    # These might be needed to answer the secondary part of the query
    if compound:
        entities, roles = extract_entities_from_question(question)
        for f in relevant_facts:
            if f.role in roles:
                query_years.add(f.start)
                query_years.add(f.end)
    
    sections = []
    
    # === Section 1: Raw Facts (always include, preserves precision) ===
    sections.append("=== RAW TEMPORAL FACTS ===")
    for f in relevant_facts:
        sections.append(f.raw)
    
    # === Section 2: Derived Facts ===
    sections.append("\n=== DERIVED TEMPORAL FACTS ===")
    seen_derived = set()
    for f in relevant_facts:
        for derived in derive_facts(f, query_years):
            if derived not in seen_derived:
                sections.append(derived)
                seen_derived.add(derived)
    
    # For compound queries, add explicit compound reasoning hints
    if compound:
        sections.append("\n=== COMPOUND QUERY REASONING ===")
        # Find the primary relationship mentioned first
        entities_list = sorted(extract_entities_from_question(question)[0], 
                               key=lambda e: question.find(e))
        if len(entities_list) >= 2:
            sections.append(f"Note: First identify the relevant time from the first relationship, then find what held at that time.")
    
    # === Section 3: Semantic Context (only for certain query types) ===
    if include_semantic and qtype in ["ordering", "duration", "timeline", "general"]:
        sections.append("\n=== SEMANTIC TEMPORAL CONTEXT ===")
        sections.append(build_ordering_context(relevant_facts, question))
    
    return "\n".join(sections)


def build_ordering_context(facts: list[TemporalFact], question: str) -> str:
    """Build semantic ordering context for ordering/timeline queries."""
    if not facts:
        return "No temporal ordering context available."
    
    # Group by object entity
    by_obj = defaultdict(list)
    for f in facts:
        by_obj[f.obj].append(f)
    
    lines = []
    
    for obj in sorted(by_obj.keys(), key=lambda e: int(e[1:])):
        obj_facts = sorted(by_obj[obj], key=lambda f: (f.start, f.end))
        
        # Group by role
        by_role = defaultdict(list)
        for f in obj_facts:
            by_role[f.role].append(f)
        
        for role in sorted(by_role.keys()):
            holders = sorted(by_role[role], key=lambda f: f.start)
            if len(holders) >= 1:
                first = holders[0]
                last = holders[-1]
                
                lines.append(f"{obj}/{role}:")
                lines.append(f"  - First holder: {first.subject} (started {first.start})")
                lines.append(f"  - Last holder: {last.subject} (ended {last.end})")
                lines.append(f"  - Total holders: {len(holders)}")
                
                if len(holders) > 1:
                    succession = []
                    for i in range(len(holders) - 1):
                        curr = holders[i]
                        nxt = holders[i + 1]
                        succession.append(f"{curr.subject} ({curr.end}) → {nxt.subject} ({nxt.start})")
                    lines.append(f"  - Succession: {' → '.join([h.subject for h in holders])}")
    
    return "\n".join(lines) if lines else "No ordering context available."


# ---------------------------------------------------------------------------
# LLM Query
# ---------------------------------------------------------------------------

async def query_llm(prompt: str, question: str, model: str = "gpt-4o-mini") -> str:
    """Send a question with context to the LLM."""
    from openai import AsyncOpenAI
    client = AsyncOpenAI()
    
    user_content = f"{prompt}\n\nQuestion: {question}\nAnswer:"
    
    response = await client.chat.completions.create(
        model=model,
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": user_content},
        ],
        temperature=0.0,
        max_tokens=100,
    )
    
    return response.choices[0].message.content.strip()


# ---------------------------------------------------------------------------
# Benchmark Runner
# ---------------------------------------------------------------------------

@dataclass
class HybridResult:
    question_type: str
    question: str
    label: str
    baseline_answer: str = ""
    hybrid_answer: str = ""
    baseline_correct: bool = False
    hybrid_correct: bool = False


async def run_hybrid_benchmark(
    data_path: str,
    limit: int = 50,
    question_type: Optional[str] = None,
    dry_run: bool = False,
    model: str = "gpt-4o-mini",
    concurrency: int = 5,
):
    """Run hybrid benchmark comparison."""
    
    print(f"Loading data from {data_path}...")
    with open(data_path) as f:
        all_items = json.load(f)
    
    print(f"Total items: {len(all_items)}")
    
    # Filter by question type if specified
    if question_type:
        all_items = [item for item in all_items if item['question_type'] == question_type]
        print(f"Filtered to {len(all_items)} items of type '{question_type}'")
    
    # Sample evenly across question types
    by_type = defaultdict(list)
    for item in all_items:
        by_type[item['question_type']].append(item)
    
    per_type = max(1, limit // len(by_type))
    selected = []
    for qt in sorted(by_type.keys()):
        selected.extend(by_type[qt][:per_type])
    items = selected[:limit]
    
    print(f"Selected {len(items)} items across {len(by_type)} question types")
    
    # Dry run: show example contexts
    if dry_run:
        for item in items[:3]:
            facts = parse_facts(item['prompt'])
            print(f"\n{'='*60}")
            print(f"Question: {item['question']}")
            print(f"Label: {item['label']}")
            print(f"Query type: {classify_query(item['question'])}")
            print(f"Total facts: {len(facts)}")
            
            relevant = filter_facts_by_relevance(facts, item['question'])
            print(f"Relevant facts: {len(relevant)}")
            
            print(f"\n--- HYBRID CONTEXT ---")
            hybrid = build_hybrid_context(facts, item['question'])
            print(hybrid[:3000])
            print(f"\n{'='*60}")
        return
    
    # Run benchmark
    results = []
    semaphore = asyncio.Semaphore(concurrency)
    
    async def process_item(item):
        async with semaphore:
            facts = parse_facts(item['prompt'])
            
            # Baseline: raw facts
            baseline_prompt = "\n".join([f.raw for f in facts])
            
            # Hybrid: raw + derived + semantic
            hybrid_prompt = build_hybrid_context(facts, item['question'])
            
            # Query both
            baseline_ans = await query_llm(baseline_prompt, item['question'], model)
            hybrid_ans = await query_llm(hybrid_prompt, item['question'], model)
            
            result = HybridResult(
                question_type=item['question_type'],
                question=item['question'],
                label=item['label'],
                baseline_answer=baseline_ans,
                hybrid_answer=hybrid_ans,
                baseline_correct=check_answer(baseline_ans, item['label']),
                hybrid_correct=check_answer(hybrid_ans, item['label']),
            )
            
            # Progress
            status = "✅" if result.hybrid_correct else "❌"
            baseline_status = "✅" if result.baseline_correct else "❌"
            print(f"{item['question_type']}: baseline={baseline_status} hybrid={status}")
            
            return result
    
    tasks = [process_item(item) for item in items]
    results = await asyncio.gather(*tasks)
    
    # Aggregate results
    by_type_results = defaultdict(lambda: {"baseline": [], "hybrid": []})
    for r in results:
        by_type_results[r.question_type]["baseline"].append(r.baseline_correct)
        by_type_results[r.question_type]["hybrid"].append(r.hybrid_correct)
    
    print(f"\n{'='*70}")
    print("HYBRID BENCHMARK RESULTS")
    print(f"{'='*70}")
    print(f"{'Question Type':<40} {'Baseline':>10} {'Hybrid':>10} {'Delta':>10}")
    print("-" * 70)
    
    total_baseline = 0
    total_hybrid = 0
    total_count = 0
    
    for qtype in sorted(by_type_results.keys()):
        baseline_acc = sum(by_type_results[qtype]["baseline"]) / len(by_type_results[qtype]["baseline"]) * 100
        hybrid_acc = sum(by_type_results[qtype]["hybrid"]) / len(by_type_results[qtype]["hybrid"]) * 100
        delta = hybrid_acc - baseline_acc
        
        total_baseline += sum(by_type_results[qtype]["baseline"])
        total_hybrid += sum(by_type_results[qtype]["hybrid"])
        total_count += len(by_type_results[qtype]["baseline"])
        
        delta_str = f"+{delta:.1f}%" if delta > 0 else f"{delta:.1f}%"
        print(f"{qtype:<40} {baseline_acc:>9.1f}% {hybrid_acc:>9.1f}% {delta_str:>10}")
    
    print("-" * 70)
    overall_baseline = total_baseline / total_count * 100
    overall_hybrid = total_hybrid / total_count * 100
    overall_delta = overall_hybrid - overall_baseline
    delta_str = f"+{overall_delta:.1f}%" if overall_delta > 0 else f"{overall_delta:.1f}%"
    print(f"{'OVERALL':<40} {overall_baseline:>9.1f}% {overall_hybrid:>9.1f}% {delta_str:>10}")
    print(f"{'='*70}")
    
    # Save results
    output_path = Path(__file__).parent / "results" / f"hybrid_{datetime.now().strftime('%Y%m%d_%H%M%S')}_n{limit}.json"
    output_path.parent.mkdir(exist_ok=True)
    
    with open(output_path, 'w') as f:
        json.dump({
            "meta": {
                "model": model,
                "limit": limit,
                "timestamp": datetime.now().isoformat(),
            },
            "overall": {
                "baseline_accuracy": overall_baseline,
                "hybrid_accuracy": overall_hybrid,
                "delta": overall_delta,
            },
            "by_type": {
                qtype: {
                    "baseline_accuracy": sum(by_type_results[qtype]["baseline"]) / len(by_type_results[qtype]["baseline"]) * 100,
                    "hybrid_accuracy": sum(by_type_results[qtype]["hybrid"]) / len(by_type_results[qtype]["hybrid"]) * 100,
                    "count": len(by_type_results[qtype]["baseline"]),
                }
                for qtype in by_type_results
            },
            "details": [
                {
                    "question_type": r.question_type,
                    "question": r.question,
                    "label": r.label,
                    "baseline_answer": r.baseline_answer,
                    "hybrid_answer": r.hybrid_answer,
                    "baseline_correct": r.baseline_correct,
                    "hybrid_correct": r.hybrid_correct,
                }
                for r in results
            ],
        }, f, indent=2)
    
    print(f"\nResults saved to: {output_path}")


def main():
    parser = argparse.ArgumentParser(description="Hybrid Temporal Context Benchmark")
    parser.add_argument("--limit", type=int, default=50)
    parser.add_argument("--question-type", type=str, default=None)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--model", type=str, default="gpt-4o-mini")
    parser.add_argument("--concurrency", type=int, default=5)
    parser.add_argument("--data", type=str, default="data/tot_semantic/test.json")
    
    args = parser.parse_args()
    
    data_path = Path(__file__).parent / args.data
    
    asyncio.run(run_hybrid_benchmark(
        data_path=str(data_path),
        limit=args.limit,
        question_type=args.question_type,
        dry_run=args.dry_run,
        model=args.model,
        concurrency=args.concurrency,
    ))


if __name__ == "__main__":
    main()
