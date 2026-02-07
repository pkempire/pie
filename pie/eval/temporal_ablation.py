"""
Temporal Ablation — the core experiment for Hypothesis 1.

Claim: LLMs reason more accurately about temporal information when it's
compiled into semantic narratives vs raw timestamps.

Design (from HYPOTHESES-AND-TESTS.md):
    Condition A: Raw timestamps (unix epoch)
    Condition B: Formatted dates ("March 15, 2024")
    Condition C: Relative time ("10 months ago")
    Condition D: Semantic temporal context ("10 months ago, during UMD
                 freshman year, changed 3 times at ~0.5x/month")

For each entity with 3+ state transitions, we generate temporal reasoning
queries and test whether the LLM answers more accurately with richer
temporal context.

Usage:
    python3 -m pie.eval.temporal_ablation [--world-model output/world_model.json]

    # Dry run (generate queries without calling LLM)
    python3 -m pie.eval.temporal_ablation --dry-run

    # Custom model
    python3 -m pie.eval.temporal_ablation --model gpt-4o-mini
"""

from __future__ import annotations
import argparse
import datetime
import json
import logging
import os
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path
from statistics import mean

logging.basicConfig(level=logging.INFO, format="%(message)s")
logger = logging.getLogger("pie.eval.temporal_ablation")


# ── Config ────────────────────────────────────────────────────────────────────

MIN_TRANSITIONS = 3        # entities must have at least this many transitions
MAX_ENTITIES = 30          # cap on entities to test
QUERIES_PER_ENTITY = 3     # queries generated per entity

# The four temporal presentation conditions
CONDITIONS = ["raw_timestamps", "formatted_dates", "relative_time", "semantic_context"]

# Query templates — each generates a different type of temporal reasoning
QUERY_TEMPLATES = [
    {
        "id": "evolution",
        "template": "How has {entity_name} evolved over time? Describe its key changes chronologically.",
        "type": "temporal_evolution",
        "scoring_hint": "Must mention multiple state changes in correct chronological order.",
    },
    {
        "id": "when_changed",
        "template": "When did {entity_name} undergo its most significant change, and what was it?",
        "type": "temporal_identification",
        "scoring_hint": "Must correctly identify the timing and nature of a major state change.",
    },
    {
        "id": "pattern",
        "template": "What pattern exists in how {entity_name} has developed? Is it changing faster or slower over time?",
        "type": "temporal_pattern",
        "scoring_hint": "Must describe a meaningful pattern and correctly assess change velocity.",
    },
]


# ── Temporal context formatters ───────────────────────────────────────────────

def _humanize_delta(seconds: float) -> str:
    """Convert seconds to human-readable duration."""
    days = seconds / 86400
    if days < 1:
        return "today"
    elif days < 2:
        return "yesterday"
    elif days < 7:
        return f"{int(days)} days ago"
    elif days < 30:
        weeks = int(days / 7)
        return f"{weeks} week{'s' if weeks != 1 else ''} ago"
    elif days < 365:
        months = int(days / 30)
        return f"about {months} month{'s' if months != 1 else ''} ago"
    else:
        years = days / 365
        if years < 2:
            months = int(days / 30)
            return f"about {months} months ago"
        return f"about {years:.1f} years ago"


def _guess_period(timestamp: float) -> str:
    """
    Guess a life period from timestamp. In production, this uses the actual
    Period entities from the world model. For eval, we use date-based heuristics.
    """
    dt = datetime.datetime.fromtimestamp(timestamp, tz=datetime.timezone.utc)
    year = dt.year
    month = dt.month

    if year == 2023 and month < 6:
        return "early 2023"
    elif year == 2023 and month < 9:
        return "summer 2023"
    elif year == 2023:
        return "fall 2023"
    elif year == 2024 and month < 6:
        return "early 2024"
    elif year == 2024 and month < 9:
        return "summer 2024"
    elif year == 2024:
        return "fall 2024"
    elif year == 2025 and month < 6:
        return "early 2025"
    elif year == 2025 and month < 9:
        return "summer 2025"
    elif year == 2025:
        return "fall 2025"
    elif year == 2026:
        return "early 2026"
    else:
        return f"{year}"


def format_condition_a(entity: dict, transitions: list[dict], now: float) -> str:
    """Condition A: Raw unix timestamps."""
    lines = [f"Entity: {entity['name']} (type: {entity['type']})"]
    lines.append(f"First seen: {entity.get('first_seen', 0)}")
    lines.append(f"Last seen: {entity.get('last_seen', 0)}")
    lines.append(f"Total transitions: {len(transitions)}")
    lines.append("")
    lines.append("State transitions:")
    for t in transitions:
        ttype = t.get("transition_type", "update")
        lines.append(
            f"  [{t.get('timestamp', 0)}] ({ttype}) {t.get('trigger_summary', 'no summary')}"
        )
    return "\n".join(lines)


def format_condition_b(entity: dict, transitions: list[dict], now: float) -> str:
    """Condition B: Formatted dates (human-readable but no relative context)."""
    lines = [f"Entity: {entity['name']} (type: {entity['type']})"]

    first_dt = datetime.datetime.fromtimestamp(
        entity.get("first_seen", 0), tz=datetime.timezone.utc
    )
    last_dt = datetime.datetime.fromtimestamp(
        entity.get("last_seen", 0), tz=datetime.timezone.utc
    )
    lines.append(f"First seen: {first_dt.strftime('%B %d, %Y')}")
    lines.append(f"Last seen: {last_dt.strftime('%B %d, %Y')}")
    lines.append(f"Total transitions: {len(transitions)}")
    lines.append("")
    lines.append("State transitions:")
    for t in transitions:
        ts = t.get("timestamp", 0)
        dt = datetime.datetime.fromtimestamp(ts, tz=datetime.timezone.utc)
        ttype = t.get("transition_type", "update")
        lines.append(
            f"  [{dt.strftime('%B %d, %Y')}] ({ttype}) {t.get('trigger_summary', 'no summary')}"
        )
    return "\n".join(lines)


def format_condition_c(entity: dict, transitions: list[dict], now: float) -> str:
    """Condition C: Relative time (e.g., '10 months ago')."""
    lines = [f"Entity: {entity['name']} (type: {entity['type']})"]

    first_ago = _humanize_delta(now - entity.get("first_seen", now))
    last_ago = _humanize_delta(now - entity.get("last_seen", now))
    lines.append(f"First appeared: {first_ago}")
    lines.append(f"Last referenced: {last_ago}")
    lines.append(f"Total transitions: {len(transitions)}")
    lines.append("")
    lines.append("State transitions:")
    for t in transitions:
        t_ago = _humanize_delta(now - t.get("timestamp", now))
        ttype = t.get("transition_type", "update")
        lines.append(
            f"  [{t_ago}] ({ttype}) {t.get('trigger_summary', 'no summary')}"
        )
    return "\n".join(lines)


def format_condition_d(entity: dict, transitions: list[dict], now: float) -> str:
    """
    Condition D: Full semantic temporal context.

    This is the PIE format from ARCHITECTURE-FINAL.md — the context compiler
    output. Includes periods, change velocity, contradiction markers, and
    natural language narrative.
    """
    name = entity["name"]
    etype = entity["type"]

    first_seen = entity.get("first_seen", now)
    last_seen = entity.get("last_seen", now)
    first_ago = _humanize_delta(now - first_seen)
    last_ago = _humanize_delta(now - last_seen)
    first_period = _guess_period(first_seen)

    change_count = len(transitions)
    months_span = max((last_seen - first_seen) / (30 * 86400), 1)
    change_velocity = change_count / months_span

    lines = []
    lines.append(
        f"{name} ({etype}) — first appeared {first_ago} ({first_period}), "
        f"last referenced {last_ago}."
    )
    lines.append(
        f"Changed state {change_count} times (~{change_velocity:.1f}x/month)."
    )

    if change_velocity > 1.5:
        lines.append("⚡ High change velocity — actively evolving.")
    elif change_velocity < 0.2:
        lines.append("🔒 Low change velocity — relatively stable.")

    lines.append("")
    lines.append("Timeline:")
    for t in transitions:
        ts = t.get("timestamp", now)
        t_ago = _humanize_delta(now - ts)
        t_period = _guess_period(ts)
        ttype = t.get("transition_type", "update")
        summary = t.get("trigger_summary", "no summary")

        lines.append(f"  • {t_ago} ({t_period}): {summary}")
        if ttype == "contradiction":
            lines.append("    ⚠ This contradicted the previous state.")
        elif ttype == "creation":
            lines.append("    [first appearance]")

    # Add state summary
    state = entity.get("current_state", {})
    if state:
        state_desc = state.get("description", "")
        if not state_desc and isinstance(state, dict):
            state_desc = "; ".join(f"{k}: {v}" for k, v in state.items() if k != "description")
        if state_desc:
            lines.append(f"\nCurrent state: {state_desc}")

    return "\n".join(lines)


CONDITION_FORMATTERS = {
    "raw_timestamps": format_condition_a,
    "formatted_dates": format_condition_b,
    "relative_time": format_condition_c,
    "semantic_context": format_condition_d,
}


# ── Query generation ──────────────────────────────────────────────────────────

@dataclass
class TemporalQuery:
    """A single temporal reasoning query."""
    query_id: str
    entity_id: str
    entity_name: str
    query_text: str
    query_type: str  # evolution | when_changed | pattern
    scoring_hint: str

    # Ground truth derived from the transitions
    ground_truth: str = ""

    # Context presented in each condition
    contexts: dict[str, str] = field(default_factory=dict)  # condition -> formatted context

    # LLM responses per condition
    responses: dict[str, str] = field(default_factory=dict)  # condition -> LLM answer

    # Scores per condition (0.0 - 1.0, from LLM judge)
    scores: dict[str, float] = field(default_factory=dict)  # condition -> score


def build_ground_truth(entity: dict, transitions: list[dict], query_type: str) -> str:
    """
    Build ground truth answer from the transitions.
    Used by the LLM judge to score responses.
    """
    name = entity["name"]
    if query_type == "evolution":
        parts = []
        for t in transitions:
            ttype = t.get("transition_type", "update")
            summary = t.get("trigger_summary", "")
            parts.append(f"({ttype}) {summary}")
        return f"{name} went through {len(transitions)} changes: " + " → ".join(parts)

    elif query_type == "when_changed":
        # Find the transition with the most significance (contradictions > updates > creation)
        priority = {"contradiction": 3, "update": 2, "creation": 1, "resolution": 2, "archival": 1}
        best = max(transitions, key=lambda t: priority.get(t.get("transition_type", ""), 0))
        ts = best.get("timestamp", 0)
        dt = datetime.datetime.fromtimestamp(ts, tz=datetime.timezone.utc)
        return (
            f"The most significant change to {name} was on "
            f"{dt.strftime('%B %d, %Y')}: {best.get('trigger_summary', 'unknown')}"
        )

    elif query_type == "pattern":
        first_seen = entity.get("first_seen", 0)
        last_seen = entity.get("last_seen", 0)
        months_span = max((last_seen - first_seen) / (30 * 86400), 1)
        velocity = len(transitions) / months_span

        # Compute inter-transition intervals
        timestamps = sorted(t.get("timestamp", 0) for t in transitions)
        intervals = []
        for i in range(1, len(timestamps)):
            intervals.append(timestamps[i] - timestamps[i - 1])

        if intervals:
            avg_interval_days = mean(intervals) / 86400
            trend = ""
            if len(intervals) >= 3:
                first_half = mean(intervals[: len(intervals) // 2])
                second_half = mean(intervals[len(intervals) // 2 :])
                if second_half < first_half * 0.7:
                    trend = "accelerating (changes getting more frequent)"
                elif second_half > first_half * 1.3:
                    trend = "decelerating (changes getting less frequent)"
                else:
                    trend = "roughly constant rate"
            return (
                f"{name} changes approximately every {avg_interval_days:.0f} days "
                f"(~{velocity:.1f}x/month). Pattern: {trend}"
            )

    return ""


def select_entities(data: dict, min_transitions: int, max_entities: int) -> list[tuple[dict, list[dict]]]:
    """
    Select entities with enough transitions for temporal reasoning queries.
    Returns list of (entity_dict, sorted_transitions) tuples.
    """
    entities = data.get("entities", {})
    transitions = data.get("transitions", {})

    # Group transitions by entity
    trans_by_entity: dict[str, list[dict]] = {}
    for tid, t in transitions.items():
        eid = t.get("entity_id", "")
        if eid not in trans_by_entity:
            trans_by_entity[eid] = []
        trans_by_entity[eid].append(t)

    # Filter and sort
    candidates = []
    for eid, entity in entities.items():
        t_list = trans_by_entity.get(eid, [])
        if len(t_list) >= min_transitions:
            t_list.sort(key=lambda t: t.get("timestamp", 0))
            candidates.append((entity, t_list))

    # Sort by transition count (most interesting first), take top N
    candidates.sort(key=lambda x: len(x[1]), reverse=True)
    selected = candidates[:max_entities]

    logger.info(
        f"Selected {len(selected)} entities with ≥{min_transitions} transitions "
        f"(from {len(candidates)} candidates)"
    )
    return selected


def generate_queries(
    entities_with_transitions: list[tuple[dict, list[dict]]],
    now: float,
) -> list[TemporalQuery]:
    """Generate temporal reasoning queries for selected entities."""
    queries = []

    for entity, transitions in entities_with_transitions:
        eid = entity.get("id", entity.get("name", "unknown"))
        name = entity.get("name", "unknown")

        for tmpl in QUERY_TEMPLATES:
            query_text = tmpl["template"].format(entity_name=name)
            ground_truth = build_ground_truth(entity, transitions, tmpl["id"])

            # Build context in all four conditions
            contexts = {}
            for condition, formatter in CONDITION_FORMATTERS.items():
                contexts[condition] = formatter(entity, transitions, now)

            queries.append(TemporalQuery(
                query_id=f"{eid}_{tmpl['id']}",
                entity_id=eid,
                entity_name=name,
                query_text=query_text,
                query_type=tmpl["id"],
                scoring_hint=tmpl["scoring_hint"],
                ground_truth=ground_truth,
                contexts=contexts,
            ))

    logger.info(f"Generated {len(queries)} queries ({len(entities_with_transitions)} entities × {len(QUERY_TEMPLATES)} templates)")
    return queries


# ── LLM answering and judging ────────────────────────────────────────────────

def _get_llm_client():
    """Lazy import and create LLM client."""
    from pie.core.llm import LLMClient
    return LLMClient()


def answer_query(
    llm,
    query: TemporalQuery,
    condition: str,
    model: str = "gpt-4o-mini",
) -> str:
    """Have the LLM answer a query given context in a specific condition."""
    context = query.contexts[condition]

    messages = [
        {
            "role": "system",
            "content": (
                "You are answering questions about an entity based on its temporal data. "
                "Use ONLY the provided context to answer. Be specific about timing and changes. "
                "Answer in 2-4 sentences."
            ),
        },
        {
            "role": "user",
            "content": f"Context:\n{context}\n\nQuestion: {query.query_text}",
        },
    ]

    result = llm.chat(messages=messages, model=model, max_tokens=300)
    return result["content"]


def judge_response(
    llm,
    query: TemporalQuery,
    condition: str,
    response: str,
    model: str = "gpt-4o-mini",
) -> float:
    """
    LLM-as-judge: score the response on temporal accuracy.
    Returns 0.0, 0.5, or 1.0.
    """
    messages = [
        {
            "role": "system",
            "content": (
                "You are a judge evaluating temporal reasoning accuracy. "
                "Score the response as: 1.0 (correct), 0.5 (partially correct), or 0.0 (incorrect). "
                "Focus on: (1) chronological accuracy, (2) identification of key changes, "
                "(3) correct temporal patterns/trends. "
                "Respond with ONLY the numeric score: 0.0, 0.5, or 1.0"
            ),
        },
        {
            "role": "user",
            "content": (
                f"Question: {query.query_text}\n\n"
                f"Ground truth: {query.ground_truth}\n\n"
                f"Scoring criteria: {query.scoring_hint}\n\n"
                f"Response to evaluate:\n{response}\n\n"
                f"Score (0.0, 0.5, or 1.0):"
            ),
        },
    ]

    result = llm.chat(messages=messages, model=model, max_tokens=10)
    content = result["content"].strip()

    # Parse score
    for val in ["1.0", "0.5", "0.0"]:
        if val in content:
            return float(val)

    # Fuzzy fallback
    content_lower = content.lower()
    if "1" in content_lower or "correct" in content_lower:
        return 1.0
    elif "0.5" in content_lower or "partial" in content_lower:
        return 0.5
    return 0.0


def run_experiment(
    queries: list[TemporalQuery],
    model: str = "gpt-4o-mini",
    judge_model: str = "gpt-4o-mini",
    conditions: list[str] | None = None,
) -> list[TemporalQuery]:
    """
    Run the full ablation experiment: for each query × condition,
    get an LLM answer and judge its accuracy.
    """
    conditions = conditions or CONDITIONS
    llm = _get_llm_client()
    total = len(queries) * len(conditions)
    done = 0

    logger.info(f"\nRunning experiment: {len(queries)} queries × {len(conditions)} conditions = {total} LLM calls")
    logger.info(f"  Answer model: {model}")
    logger.info(f"  Judge model:  {judge_model}")

    for qi, query in enumerate(queries):
        for condition in conditions:
            done += 1
            logger.info(f"  [{done}/{total}] {query.entity_name} / {query.query_type} / {condition}")

            try:
                # Get answer
                response = answer_query(llm, query, condition, model=model)
                query.responses[condition] = response

                # Judge answer
                score = judge_response(llm, query, condition, response, model=judge_model)
                query.scores[condition] = score

                logger.info(f"    → score={score}")
            except Exception as e:
                logger.error(f"    → ERROR: {e}")
                query.responses[condition] = f"ERROR: {e}"
                query.scores[condition] = 0.0

            # Light rate limiting
            time.sleep(0.1)

    logger.info(f"\nExperiment complete. LLM stats: {llm.stats}")
    return queries


# ── Results analysis ──────────────────────────────────────────────────────────

@dataclass
class AblationResults:
    """Aggregated experiment results."""
    # Per-condition average scores
    condition_scores: dict[str, float] = field(default_factory=dict)

    # Per-query-type × condition scores
    type_condition_scores: dict[str, dict[str, float]] = field(default_factory=dict)

    # Per-query breakdown
    per_query: list[dict] = field(default_factory=list)

    # Summary
    best_condition: str = ""
    worst_condition: str = ""
    semantic_vs_raw_delta: float = 0.0

    def to_dict(self) -> dict:
        return {
            "hypothesis": "H1: Semantic temporal context > Raw timestamps",
            "expected": "semantic_context >> relative_time > formatted_dates > raw_timestamps",
            "condition_scores": {k: round(v, 3) for k, v in self.condition_scores.items()},
            "type_condition_scores": {
                qtype: {cond: round(score, 3) for cond, score in cond_scores.items()}
                for qtype, cond_scores in self.type_condition_scores.items()
            },
            "best_condition": self.best_condition,
            "worst_condition": self.worst_condition,
            "semantic_vs_raw_delta": round(self.semantic_vs_raw_delta, 3),
            "per_query": self.per_query,
        }


def analyze_results(queries: list[TemporalQuery]) -> AblationResults:
    """Analyze experiment results."""
    results = AblationResults()

    # Per-condition aggregation
    condition_scores_all: dict[str, list[float]] = {c: [] for c in CONDITIONS}
    type_condition_all: dict[str, dict[str, list[float]]] = {}

    for query in queries:
        query_result = {
            "query_id": query.query_id,
            "entity": query.entity_name,
            "query_type": query.query_type,
            "query": query.query_text,
            "scores": query.scores,
        }
        results.per_query.append(query_result)

        for condition, score in query.scores.items():
            condition_scores_all[condition].append(score)

            if query.query_type not in type_condition_all:
                type_condition_all[query.query_type] = {c: [] for c in CONDITIONS}
            type_condition_all[query.query_type][condition].append(score)

    # Compute averages
    for condition, scores in condition_scores_all.items():
        results.condition_scores[condition] = mean(scores) if scores else 0.0

    for qtype, cond_scores in type_condition_all.items():
        results.type_condition_scores[qtype] = {}
        for condition, scores in cond_scores.items():
            results.type_condition_scores[qtype][condition] = mean(scores) if scores else 0.0

    # Summary
    if results.condition_scores:
        results.best_condition = max(results.condition_scores, key=results.condition_scores.get)
        results.worst_condition = min(results.condition_scores, key=results.condition_scores.get)
        raw = results.condition_scores.get("raw_timestamps", 0)
        semantic = results.condition_scores.get("semantic_context", 0)
        results.semantic_vs_raw_delta = semantic - raw

    return results


def print_results(results: AblationResults):
    """Print experiment results."""
    print("\n" + "=" * 70)
    print("  TEMPORAL ABLATION RESULTS — Hypothesis 1")
    print("  Semantic Temporal Context vs. Raw Timestamps")
    print("=" * 70)

    print(f"\n{'─'*50}")
    print("  OVERALL ACCURACY BY CONDITION")
    print(f"{'─'*50}")

    # Sort by score
    sorted_conditions = sorted(results.condition_scores.items(), key=lambda x: -x[1])
    for condition, score in sorted_conditions:
        bar = "█" * int(score * 40)
        marker = " ← BEST" if condition == results.best_condition else ""
        marker = " ← WORST" if condition == results.worst_condition else marker
        print(f"  {condition:<20} {score:.3f}  {bar}{marker}")

    delta = results.semantic_vs_raw_delta
    direction = "+" if delta >= 0 else ""
    print(f"\n  Semantic vs Raw delta: {direction}{delta:.3f}")
    if delta > 0.1:
        print("  ✅ Hypothesis SUPPORTED: Semantic context significantly outperforms raw timestamps")
    elif delta > 0:
        print("  ⚠️  Weak support: Semantic context slightly better")
    else:
        print("  ❌ Hypothesis NOT SUPPORTED: Raw timestamps performed as well or better")

    if results.type_condition_scores:
        print(f"\n{'─'*50}")
        print("  ACCURACY BY QUERY TYPE × CONDITION")
        print(f"{'─'*50}")
        for qtype, cond_scores in results.type_condition_scores.items():
            print(f"\n  {qtype}:")
            for condition, score in sorted(cond_scores.items(), key=lambda x: -x[1]):
                bar = "█" * int(score * 30)
                print(f"    {condition:<20} {score:.3f}  {bar}")

    print("\n" + "=" * 70)


# ── CLI entry point ───────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="PIE Temporal Ablation — test H1: semantic temporal context vs raw timestamps"
    )
    parser.add_argument(
        "--world-model",
        type=Path,
        default=Path("output/world_model.json"),
        help="Path to world_model.json",
    )
    parser.add_argument(
        "--model",
        type=str,
        default="gpt-4o-mini",
        help="LLM model for answering queries (default: gpt-4o-mini)",
    )
    parser.add_argument(
        "--judge-model",
        type=str,
        default="gpt-4o-mini",
        help="LLM model for judging responses (default: gpt-4o-mini)",
    )
    parser.add_argument(
        "--min-transitions",
        type=int,
        default=MIN_TRANSITIONS,
        help=f"Minimum transitions per entity (default: {MIN_TRANSITIONS})",
    )
    parser.add_argument(
        "--max-entities",
        type=int,
        default=MAX_ENTITIES,
        help=f"Max entities to test (default: {MAX_ENTITIES})",
    )
    parser.add_argument(
        "--conditions",
        nargs="+",
        choices=CONDITIONS,
        default=None,
        help="Specific conditions to test (default: all four)",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Generate queries and contexts without calling the LLM",
    )
    parser.add_argument(
        "--json",
        action="store_true",
        help="Output results as JSON",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="Write results to file",
    )
    args = parser.parse_args()

    # Load world model
    if not args.world_model.exists():
        logger.error(f"World model not found at {args.world_model}")
        logger.error("Run the ingestion pipeline first.")
        sys.exit(1)

    with open(args.world_model) as f:
        data = json.load(f)

    logger.info(f"Loaded world model: {len(data.get('entities', {}))} entities, "
                f"{len(data.get('transitions', {}))} transitions")

    now = time.time()

    # Select entities and generate queries
    entities = select_entities(data, args.min_transitions, args.max_entities)
    if not entities:
        logger.error("No entities with enough transitions. Run more ingestion first.")
        sys.exit(1)

    queries = generate_queries(entities, now)

    if args.dry_run:
        # Print sample contexts for inspection
        print("\n" + "=" * 70)
        print("  DRY RUN — Sample Contexts")
        print("=" * 70)
        sample = queries[:3]
        for q in sample:
            print(f"\n{'─'*50}")
            print(f"Query: {q.query_text}")
            print(f"Type:  {q.query_type}")
            print(f"Truth: {q.ground_truth}")
            for cond in CONDITIONS:
                print(f"\n  ── {cond} ──")
                print(f"  {q.contexts[cond][:300]}...")
        print(f"\n\nTotal queries ready: {len(queries)}")
        print("Re-run without --dry-run to execute the experiment.")
        return

    # Run the experiment
    conditions = args.conditions or CONDITIONS
    queries = run_experiment(queries, model=args.model, judge_model=args.judge_model, conditions=conditions)

    # Analyze and display
    results = analyze_results(queries)

    if args.json:
        print(json.dumps(results.to_dict(), indent=2))
    else:
        print_results(results)

    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        with open(args.output, "w") as f:
            json.dump(results.to_dict(), f, indent=2)
        logger.info(f"\nResults saved to {args.output}")


if __name__ == "__main__":
    main()
