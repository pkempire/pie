#!/usr/bin/env python3
"""
Test of Time (ToT) Benchmark Runner
====================================
Tests whether PIE's semantic temporal context compilation improves LLM temporal reasoning.

Two conditions:
  A (baseline): Raw temporal facts with dates
  B (PIE-style): Reformulated into semantic temporal narratives with relative time,
                  durations, overlap markers, entity timelines, and ordering context

Usage:
  python runner.py --limit 20 --dry-run          # verify reformulation
  python runner.py --limit 50                     # run real evaluation
  python runner.py --limit 100 --question-type before_after
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

# ---------------------------------------------------------------------------
# Data structures
# ---------------------------------------------------------------------------

@dataclass
class TemporalFact:
    """A parsed temporal fact: subject held role for object from start to end."""
    subject: str      # E98
    role: str          # R82
    obj: str           # E59
    start: int         # 1978
    end: int           # 1983
    duration: int = 0  # computed

    def __post_init__(self):
        self.duration = self.end - self.start

    @property
    def raw(self) -> str:
        return f"{self.subject} was the {self.role} of {self.obj} from {self.start} to {self.end}."


@dataclass
class QuestionItem:
    question: str
    question_type: str
    label: str
    prompt: str
    facts: list = field(default_factory=list)


@dataclass
class Result:
    question_type: str
    question: str
    label: str
    baseline_answer: str = ""
    pie_answer: str = ""
    baseline_correct: bool = False
    pie_correct: bool = False


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
# PIE-style semantic temporal reformulation
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


def relative_time_desc(ref_year: int, target_year: int) -> str:
    """Describe target_year relative to ref_year."""
    diff = target_year - ref_year
    if diff == 0:
        return "at the same time"
    if diff > 0:
        return f"{diff} year{'s' if diff != 1 else ''} later"
    return f"{abs(diff)} year{'s' if abs(diff) != 1 else ''} earlier"


def compute_overlaps(fact: TemporalFact, others: list[TemporalFact]) -> list[str]:
    """Find facts that temporally overlap with the given fact."""
    overlaps = []
    for o in others:
        if o is fact:
            continue
        # Check overlap: intervals [a.start, a.end) and [b.start, b.end)
        overlap_start = max(fact.start, o.start)
        overlap_end = min(fact.end, o.end)
        if overlap_start < overlap_end:
            overlap_years = overlap_end - overlap_start
            overlaps.append(
                f"overlapped with {o.subject}'s role as {o.role} of {o.obj} "
                f"for {overlap_years} year{'s' if overlap_years != 1 else ''} "
                f"({overlap_start}–{overlap_end})"
            )
    return overlaps


def reformulate_pie_style(facts: list[TemporalFact], question: str) -> str:
    """
    Reformulate temporal facts into PIE's semantic temporal narrative format.

    Key transformations (from PIE ARCHITECTURE §4.1):
    1. Group facts by entity → entity timelines
    2. Add duration descriptions (brief, extended, long-standing)
    3. Relative time anchoring between facts
    4. Overlap markers between concurrent facts
    5. Temporal ordering narrative (sequential context)
    6. Period-style grouping (early period, middle period, recent)

    Adapts verbosity to fact count to stay within context limits.
    """
    if not facts:
        return "No temporal facts provided."

    # --- Adaptive verbosity ---
    n_facts = len(facts)
    # Compact mode for very large prompts, full mode for small ones
    compact = n_facts > 150
    max_overlaps_per_fact = 1 if compact else 3
    show_full_ordering = not compact  # skip full ordering for huge sets

    # --- Gather global stats ---
    all_starts = [f.start for f in facts]
    all_ends = [f.end for f in facts]
    global_start = min(all_starts)
    global_end = max(all_ends)
    total_span = global_end - global_start

    # Define era boundaries (thirds of the total span)
    era_size = max(total_span // 3, 1)
    def era_label(year: int) -> str:
        offset = year - global_start
        if offset < era_size:
            return "early period"
        elif offset < 2 * era_size:
            return "middle period"
        else:
            return "later period"

    # --- Group facts by object entity (the entity things happen TO) ---
    by_object: dict[str, list[TemporalFact]] = defaultdict(list)
    for f in facts:
        by_object[f.obj].append(f)

    # --- Build the narrative ---
    lines = []
    lines.append(f"TEMPORAL CONTEXT ({global_start}–{global_end}, spanning {total_span} years)")
    lines.append("")

    # === Section 1: Entity Timelines (grouped by object) ===
    lines.append("═══ ENTITY TIMELINES ═══")
    lines.append("")

    for obj_entity in sorted(by_object.keys(), key=lambda e: int(e[1:])):
        obj_facts = sorted(by_object[obj_entity], key=lambda f: (f.start, f.end))
        lines.append(f"▸ Timeline for {obj_entity}:")

        # Track roles by type for this entity
        role_holders: dict[str, list[TemporalFact]] = defaultdict(list)
        for f in obj_facts:
            role_holders[f.role].append(f)

        prev_fact = None
        for i, f in enumerate(obj_facts):
            dur = duration_description(f.duration)

            # Build the fact line
            line_parts = [f"  [{f.start}–{f.end}] {f.subject} as {f.role} — {dur}"]

            # Relative to previous fact on same entity
            if prev_fact:
                gap = f.start - prev_fact.end
                if gap > 0:
                    line_parts.append(f"({gap}yr gap)")
                elif gap == 0:
                    line_parts.append("(immediately following)")
                else:
                    line_parts.append(f"(began {abs(gap)}yr before prev ended)")

            lines.append(" ".join(line_parts))

            # Check overlaps (capped)
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

        # Summary stats for this entity
        roles_summary = []
        for role, holders in sorted(role_holders.items()):
            holder_names = [h.subject for h in holders]
            roles_summary.append(f"{role}: {', '.join(holder_names)}")
        lines.append(f"  Summary: {'; '.join(roles_summary)}")
        lines.append("")

    # === Section 2: Temporal ordering (skip for huge fact sets) ===
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

    # === Section 3: Key temporal relationships (ALWAYS included — critical for reasoning) ===
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
                # Show succession chain
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

    # --- Hard cap: truncate if still too long (~100K chars ≈ ~25K tokens) ---
    MAX_CHARS = 400_000
    if len(narrative) > MAX_CHARS:
        narrative = narrative[:MAX_CHARS] + "\n... (truncated for length)"

    return narrative


# ---------------------------------------------------------------------------
# LLM interaction
# ---------------------------------------------------------------------------

SYSTEM_PROMPT = """You are a precise temporal reasoning assistant. Answer the question based ONLY on the provided temporal facts. Give ONLY the final answer with no explanation.

Rules:
- For entity questions: answer with just the entity ID (e.g., E42)
- For time questions: answer with just the year (e.g., 1985)
- For duration questions: answer with just the number (e.g., 5)
- For counting questions: answer with just the number (e.g., 3)
- For timeline/list questions: answer with a comma-separated list (e.g., E59,E57,E43)
- No extra text, no reasoning, just the answer."""


async def query_llm(prompt: str, question: str, model: str = "gpt-4o-mini") -> str:
    """Send a question with context to the LLM and get a short answer."""
    try:
        from openai import AsyncOpenAI
    except ImportError:
        print("ERROR: openai package not installed. Run: pip install openai", file=sys.stderr)
        sys.exit(1)

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

    answer = response.choices[0].message.content.strip()
    return answer


def normalize_answer(answer: str) -> str:
    """Normalize an answer for comparison. Handles JSON, prose, and clean answers."""
    answer = answer.strip()

    # Try to extract from JSON response (model sometimes returns {"answer": "X", ...})
    if answer.startswith('{'):
        try:
            parsed = json.loads(answer)
            if isinstance(parsed, dict) and 'answer' in parsed:
                answer = str(parsed['answer']).strip()
        except json.JSONDecodeError:
            # Try regex extraction as fallback
            m = re.search(r'"answer"\s*:\s*"([^"]+)"', answer)
            if m:
                answer = m.group(1).strip()
            else:
                # Try number answer
                m = re.search(r'"answer"\s*:\s*(\d+)', answer)
                if m:
                    answer = m.group(1).strip()

    # Strip whitespace, periods, quotes
    answer = answer.strip().strip('.').strip('"').strip("'").strip()

    # Remove any prefix like "Answer: " or "The answer is "
    for prefix in ["answer:", "the answer is", "answer is", "final answer:"]:
        if answer.lower().startswith(prefix):
            answer = answer[len(prefix):].strip()

    # Remove trailing punctuation
    answer = answer.rstrip('.')

    return answer


def check_answer(predicted: str, label: str) -> bool:
    """Check if predicted answer matches label."""
    pred = normalize_answer(predicted)
    lab = normalize_answer(label)

    # Direct match
    if pred.lower() == lab.lower():
        return True

    # For comma-separated lists, normalize spacing
    if ',' in lab:
        pred_parts = [p.strip() for p in pred.split(',')]
        lab_parts = [p.strip() for p in lab.split(',')]
        return pred_parts == lab_parts

    return False


# ---------------------------------------------------------------------------
# Main runner
# ---------------------------------------------------------------------------

async def run_benchmark(
    data_path: str,
    limit: int = 50,
    question_type: Optional[str] = None,
    dry_run: bool = False,
    model: str = "gpt-4o-mini",
    concurrency: int = 5,
):
    """Run the benchmark comparison."""
    # Load data
    print(f"Loading data from {data_path}...")
    with open(data_path) as f:
        all_items = json.load(f)

    print(f"Total items: {len(all_items)}")

    # Filter by question type if specified
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
    print(f"Dry run: {dry_run}")
    print()

    results: list[Result] = []
    semaphore = asyncio.Semaphore(concurrency)

    async def process_item(idx: int, item: dict) -> Result:
        question = item['question']
        q_type = item['question_type']
        label = str(item['label'])
        prompt = item['prompt']

        # Parse facts
        facts = parse_facts(prompt)

        # Build PIE-style reformulation
        pie_prompt = reformulate_pie_style(facts, question)

        result = Result(
            question_type=q_type,
            question=question,
            label=label,
        )

        if dry_run:
            # Show reformulation for first few items
            if idx < 5:
                print(f"{'='*80}")
                print(f"[{idx+1}] Question ({q_type}): {question}")
                print(f"    Label: {label}")
                print(f"    Facts parsed: {len(facts)}")
                print()
                print("--- BASELINE PROMPT (first 500 chars) ---")
                print(prompt[:500])
                print("...")
                print()
                print("--- PIE-STYLE PROMPT (first 1500 chars) ---")
                print(pie_prompt[:1500])
                print("...")
                print()
            return result

        # Run both conditions
        async with semaphore:
            try:
                baseline_answer, pie_answer = await asyncio.gather(
                    query_llm(prompt, question, model),
                    query_llm(pie_prompt + "\n\nUsing the temporal context above, answer:", question, model),
                )
            except Exception as e:
                print(f"  ERROR on item {idx+1}: {e}", file=sys.stderr)
                return result

        result.baseline_answer = baseline_answer
        result.pie_answer = pie_answer
        result.baseline_correct = check_answer(baseline_answer, label)
        result.pie_correct = check_answer(pie_answer, label)

        status_b = "✓" if result.baseline_correct else "✗"
        status_p = "✓" if result.pie_correct else "✗"
        print(f"  [{idx+1}/{len(items)}] {q_type[:20]:20s} | "
              f"Baseline: {status_b} ({baseline_answer[:30]:30s}) | "
              f"PIE: {status_p} ({pie_answer[:30]:30s}) | "
              f"Label: {label}")

        return result

    # Process all items
    start_time = time.time()

    if dry_run:
        for idx, item in enumerate(items):
            r = await process_item(idx, item)
            results.append(r)
    else:
        print("Running evaluations...")
        print("-" * 100)
        tasks = [process_item(idx, item) for idx, item in enumerate(items)]
        results = await asyncio.gather(*tasks)
        results = list(results)

    elapsed = time.time() - start_time

    if dry_run:
        print(f"\n{'='*80}")
        print(f"DRY RUN complete. {len(items)} items processed in {elapsed:.1f}s")
        print(f"No LLM calls made. Review the reformulations above.")
        return

    # ---------------------------------------------------------------------------
    # Analyze and display results
    # ---------------------------------------------------------------------------
    print()
    print("=" * 100)
    print("RESULTS")
    print("=" * 100)

    # Overall accuracy
    baseline_correct = sum(1 for r in results if r.baseline_correct)
    pie_correct = sum(1 for r in results if r.pie_correct)
    total = len(results)

    baseline_acc = baseline_correct / total * 100 if total else 0
    pie_acc = pie_correct / total * 100 if total else 0
    delta = pie_acc - baseline_acc

    print(f"\nOverall Accuracy ({total} questions):")
    print(f"  Baseline (raw dates):  {baseline_correct}/{total} = {baseline_acc:.1f}%")
    print(f"  PIE-style (semantic):  {pie_correct}/{total} = {pie_acc:.1f}%")
    print(f"  Delta:                 {'+' if delta >= 0 else ''}{delta:.1f}%")
    print()

    # Breakdown by question type
    by_type: dict[str, list[Result]] = defaultdict(list)
    for r in results:
        by_type[r.question_type].append(r)

    print(f"{'Question Type':<45} {'N':>4} {'Baseline':>10} {'PIE':>10} {'Delta':>8}")
    print("─" * 82)

    for qt in sorted(by_type.keys()):
        type_results = by_type[qt]
        n = len(type_results)
        b_correct = sum(1 for r in type_results if r.baseline_correct)
        p_correct = sum(1 for r in type_results if r.pie_correct)
        b_pct = b_correct / n * 100 if n else 0
        p_pct = p_correct / n * 100 if n else 0
        d = p_pct - b_pct

        d_str = f"{'+' if d >= 0 else ''}{d:.1f}%"
        print(f"  {qt:<43} {n:>4} {b_pct:>8.1f}%  {p_pct:>8.1f}%  {d_str:>8}")

    print("─" * 82)
    d_str = f"{'+' if delta >= 0 else ''}{delta:.1f}%"
    print(f"  {'TOTAL':<43} {total:>4} {baseline_acc:>8.1f}%  {pie_acc:>8.1f}%  {d_str:>8}")

    # Show interesting cases: PIE correct but baseline wrong
    flips_to_pie = [r for r in results if r.pie_correct and not r.baseline_correct]
    flips_to_base = [r for r in results if r.baseline_correct and not r.pie_correct]

    print(f"\n\nFlip Analysis:")
    print(f"  PIE rescued (baseline wrong → PIE correct):  {len(flips_to_pie)}")
    print(f"  PIE hurt (baseline correct → PIE wrong):     {len(flips_to_base)}")

    if flips_to_pie:
        print(f"\n  Examples where PIE helped:")
        for r in flips_to_pie[:5]:
            print(f"    [{r.question_type}] {r.question[:70]}")
            print(f"      Label: {r.label} | Baseline said: {r.baseline_answer} | PIE said: {r.pie_answer}")

    if flips_to_base:
        print(f"\n  Examples where PIE hurt:")
        for r in flips_to_base[:5]:
            print(f"    [{r.question_type}] {r.question[:70]}")
            print(f"      Label: {r.label} | Baseline said: {r.baseline_answer} | PIE said: {r.pie_answer}")

    print(f"\nElapsed: {elapsed:.1f}s ({elapsed/total:.1f}s per question pair)")

    # ---------------------------------------------------------------------------
    # Save results
    # ---------------------------------------------------------------------------
    results_dir = Path(__file__).parent / "results"
    results_dir.mkdir(exist_ok=True)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    qt_suffix = f"_{question_type}" if question_type else ""
    results_file = results_dir / f"results_{timestamp}_n{total}{qt_suffix}.json"

    output = {
        "meta": {
            "timestamp": timestamp,
            "model": model,
            "total_questions": total,
            "question_type_filter": question_type,
            "elapsed_seconds": round(elapsed, 1),
        },
        "overall": {
            "baseline_accuracy": round(baseline_acc, 2),
            "pie_accuracy": round(pie_acc, 2),
            "delta": round(delta, 2),
            "baseline_correct": baseline_correct,
            "pie_correct": pie_correct,
            "total": total,
        },
        "by_type": {},
        "flips": {
            "pie_rescued": len(flips_to_pie),
            "pie_hurt": len(flips_to_base),
        },
        "details": [
            {
                "question_type": r.question_type,
                "question": r.question,
                "label": r.label,
                "baseline_answer": r.baseline_answer,
                "pie_answer": r.pie_answer,
                "baseline_correct": r.baseline_correct,
                "pie_correct": r.pie_correct,
            }
            for r in results
        ],
    }

    for qt in sorted(by_type.keys()):
        type_results = by_type[qt]
        n = len(type_results)
        b = sum(1 for r in type_results if r.baseline_correct)
        p = sum(1 for r in type_results if r.pie_correct)
        output["by_type"][qt] = {
            "n": n,
            "baseline_correct": b,
            "baseline_accuracy": round(b / n * 100, 2) if n else 0,
            "pie_correct": p,
            "pie_accuracy": round(p / n * 100, 2) if n else 0,
            "delta": round((p - b) / n * 100, 2) if n else 0,
        }

    with open(results_file, 'w') as f:
        json.dump(output, f, indent=2)

    print(f"\nResults saved to: {results_file}")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="ToT Benchmark: Baseline vs PIE-style temporal reasoning"
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
        "--dry-run", action="store_true",
        help="Show reformulated prompts without making LLM calls"
    )
    parser.add_argument(
        "--model", type=str, default="gpt-4o-mini",
        help="OpenAI model to use (default: gpt-4o-mini)"
    )
    parser.add_argument(
        "--concurrency", type=int, default=5,
        help="Max concurrent LLM requests (default: 5)"
    )
    parser.add_argument(
        "--data", type=str,
        default=str(Path(__file__).parent / "data" / "tot_semantic" / "test.json"),
        help="Path to test.json"
    )

    args = parser.parse_args()

    if not args.dry_run and not os.environ.get("OPENAI_API_KEY"):
        print("ERROR: OPENAI_API_KEY not set", file=sys.stderr)
        sys.exit(1)

    asyncio.run(run_benchmark(
        data_path=args.data,
        limit=args.limit,
        question_type=args.question_type,
        dry_run=args.dry_run,
        model=args.model,
        concurrency=args.concurrency,
    ))


if __name__ == "__main__":
    main()


def reformulate_hybrid(facts: list[TemporalFact], question: str) -> str:
    """
    Hybrid reformulation: raw facts + ordering insights.
    
    Keeps exact dates visible for arithmetic, adds first/last/succession for ordering.
    """
    if not facts:
        return "No temporal facts provided."
    
    # Section 1: Raw facts (compact, preserves exact dates)
    lines = ["=== TEMPORAL FACTS ===", ""]
    for f in sorted(facts, key=lambda x: (x.start, x.end)):
        lines.append(f"{f.subject} was the {f.role} of {f.obj} from {f.start} to {f.end}.")
    
    # Section 2: Ordering insights (only for multi-holder roles)
    lines.append("")
    lines.append("=== ORDERING INSIGHTS ===")
    
    by_obj_role = defaultdict(list)
    for f in facts:
        by_obj_role[(f.obj, f.role)].append(f)
    
    for (obj, role), holders in sorted(by_obj_role.items()):
        if len(holders) < 2:
            continue
        holders_sorted = sorted(holders, key=lambda f: f.start)
        first = holders_sorted[0]
        last = holders_sorted[-1]
        
        lines.append(f"")
        lines.append(f"{obj}/{role}: FIRST={first.subject} ({first.start}), LAST={last.subject} ({last.end})")
        
        # Succession chain
        for i in range(len(holders_sorted) - 1):
            curr = holders_sorted[i]
            nxt = holders_sorted[i + 1]
            gap = nxt.start - curr.end
            if gap == 0:
                rel = "immediately before"
            elif gap > 0:
                rel = f"{gap}yr before"
            else:
                rel = f"overlapped {abs(gap)}yr with"
            lines.append(f"  {curr.subject} ({curr.end}) {rel} {nxt.subject} ({nxt.start})")
    
    return "\n".join(lines)
