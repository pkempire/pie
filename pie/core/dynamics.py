"""
Transition Dynamics Module — learning P(next_state | current_state, context)

This is PIE's equivalent of the dynamics model in Dreamer/MuZero, but for
non-physical state spaces (knowledge, beliefs, relationships, trajectories).

Traditional world models: s_{t+1} = f(s_t, a_t)  — predict next physical state
PIE dynamics:             s_{t+1} = f(s_t, trigger, entity_type, history)  — predict next knowledge state

Three levels of capability:
  Level 1 — Statistics: empirical transition rates, volatility, staleness
  Level 2 — Patterns:   recurring transition sequences, causal co-occurrence
  Level 3 — Prediction: LLM-based forward simulation using learned patterns

The key insight from Dreamer/MuZero that applies here: a world model is not
just a database — it's a *simulator*. It should predict what happens next,
not just record what happened. For knowledge state spaces, this means
predicting which entities will change, when, and how.
"""

from __future__ import annotations

import math
import logging
from collections import defaultdict, Counter
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any

from .models import (
    Entity, EntityType, StateTransition, TransitionType,
    Relationship, Procedure,
)
from .world_model import WorldModel

logger = logging.getLogger("pie.dynamics")


# ── Level 1: Transition Statistics ───────────────────────────────────────────


@dataclass
class EntityDynamicsProfile:
    """Dynamics profile for a single entity — its behavioral signature."""
    entity_id: str
    entity_name: str
    entity_type: EntityType

    # Transition statistics
    total_transitions: int = 0
    transition_type_counts: dict[str, int] = field(default_factory=dict)

    # Temporal dynamics
    first_transition_ts: float = 0.0
    last_transition_ts: float = 0.0
    mean_interval_s: float = 0.0       # average seconds between transitions
    volatility: float = 0.0            # coefficient of variation of intervals

    # State dynamics
    unique_states_observed: int = 0
    contradiction_count: int = 0
    has_been_archived: bool = False

    # Derived scores
    staleness_score: float = 0.0       # 0=fresh, 1=very stale (hasn't changed when it should)
    predictability: float = 0.0        # 0=random, 1=very regular transition pattern

    @property
    def is_volatile(self) -> bool:
        """Entity changes frequently relative to its type's baseline."""
        return self.volatility > 0.5 and self.total_transitions > 3

    @property
    def is_stale(self) -> bool:
        """Entity hasn't transitioned in a while but historically does."""
        return self.staleness_score > 0.7 and self.total_transitions > 2


@dataclass
class TypeDynamicsProfile:
    """Aggregate dynamics for an entity type — the "prior" for that type."""
    entity_type: EntityType
    entity_count: int = 0
    mean_transitions_per_entity: float = 0.0
    mean_interval_s: float = 0.0
    median_interval_s: float = 0.0
    contradiction_rate: float = 0.0     # fraction of transitions that are contradictions
    archival_rate: float = 0.0          # fraction of entities that get archived
    common_transition_sequences: list[tuple[str, ...]] = field(default_factory=list)


@dataclass
class TransitionCooccurrence:
    """Two entities that tend to transition together (causal signal)."""
    entity_a_id: str
    entity_b_id: str
    entity_a_name: str
    entity_b_name: str
    cooccurrence_count: int = 0
    typical_lag_s: float = 0.0          # B typically transitions lag_s after A
    confidence: float = 0.0


@dataclass
class DynamicsReport:
    """Full dynamics analysis of a world model."""
    entity_profiles: dict[str, EntityDynamicsProfile] = field(default_factory=dict)
    type_profiles: dict[str, TypeDynamicsProfile] = field(default_factory=dict)
    cooccurrences: list[TransitionCooccurrence] = field(default_factory=list)

    # Predictions
    stale_entities: list[str] = field(default_factory=list)         # entity_ids likely needing update
    volatile_entities: list[str] = field(default_factory=list)      # entity_ids that change a lot
    predicted_next_transitions: list[dict] = field(default_factory=list)  # {entity_id, predicted_type, confidence, reason}


class TransitionDynamics:
    """
    Analyzes and predicts state transitions in a world model.

    This is the core of PIE's "learning" capability — it doesn't just
    store what happened, it learns *patterns* in how things change and
    uses those patterns to predict what will change next.
    """

    def __init__(self, world_model: WorldModel):
        self.wm = world_model

    def analyze(self, reference_time: float | None = None) -> DynamicsReport:
        """
        Full dynamics analysis of the world model.

        Args:
            reference_time: "now" for staleness calculation. Defaults to
                            the latest transition timestamp.

        Returns:
            DynamicsReport with per-entity profiles, type profiles,
            co-occurrence patterns, and predictions.
        """
        report = DynamicsReport()

        if not self.wm.entities:
            return report

        # Find reference time
        if reference_time is None:
            all_ts = [t.timestamp for t in self.wm.transitions.values()]
            reference_time = max(all_ts) if all_ts else 0.0

        # ── Per-entity analysis ──
        for eid, entity in self.wm.entities.items():
            profile = self._analyze_entity(eid, entity, reference_time)
            report.entity_profiles[eid] = profile

            if profile.is_stale:
                report.stale_entities.append(eid)
            if profile.is_volatile:
                report.volatile_entities.append(eid)

        # ── Per-type aggregation ──
        type_groups: dict[EntityType, list[EntityDynamicsProfile]] = defaultdict(list)
        for profile in report.entity_profiles.values():
            type_groups[profile.entity_type].append(profile)

        for etype, profiles in type_groups.items():
            report.type_profiles[etype.value] = self._aggregate_type(etype, profiles)

        # ── Co-occurrence detection ──
        report.cooccurrences = self._detect_cooccurrences()

        # ── Predictions ──
        report.predicted_next_transitions = self._predict_next(
            report.entity_profiles, report.type_profiles, reference_time
        )

        return report

    def _analyze_entity(
        self,
        eid: str,
        entity: Entity,
        reference_time: float,
    ) -> EntityDynamicsProfile:
        """Compute dynamics profile for a single entity."""
        transitions = self.wm.get_transitions(eid, ordered=True)

        profile = EntityDynamicsProfile(
            entity_id=eid,
            entity_name=entity.name,
            entity_type=entity.type,
            total_transitions=len(transitions),
        )

        if not transitions:
            return profile

        # Transition type counts
        type_counts: Counter = Counter()
        states_seen = set()
        for t in transitions:
            type_counts[t.transition_type.value] += 1
            state_key = str(t.to_state)[:100]  # rough dedup
            states_seen.add(state_key)
            if t.transition_type == TransitionType.ARCHIVAL:
                profile.has_been_archived = True

        profile.transition_type_counts = dict(type_counts)
        profile.unique_states_observed = len(states_seen)
        profile.contradiction_count = type_counts.get("contradiction", 0)

        # Temporal dynamics
        timestamps = sorted(t.timestamp for t in transitions)
        profile.first_transition_ts = timestamps[0]
        profile.last_transition_ts = timestamps[-1]

        if len(timestamps) >= 2:
            intervals = [
                timestamps[i + 1] - timestamps[i]
                for i in range(len(timestamps) - 1)
            ]
            mean_interval = sum(intervals) / len(intervals)
            profile.mean_interval_s = mean_interval

            # Volatility = coefficient of variation
            if mean_interval > 0 and len(intervals) >= 2:
                variance = sum((iv - mean_interval) ** 2 for iv in intervals) / len(intervals)
                std = math.sqrt(variance)
                profile.volatility = std / mean_interval
            else:
                profile.volatility = 0.0

            # Predictability = 1 - volatility (clamped)
            profile.predictability = max(0.0, min(1.0, 1.0 - profile.volatility))

        # Staleness: how overdue is this entity for a transition?
        if len(timestamps) >= 2 and profile.mean_interval_s > 0:
            time_since_last = reference_time - profile.last_transition_ts
            expected_transitions = time_since_last / profile.mean_interval_s
            # Sigmoid-like staleness: 0 at expected=0, ~0.5 at expected=1, ~0.9 at expected=3
            profile.staleness_score = 1.0 - (1.0 / (1.0 + expected_transitions))
        else:
            profile.staleness_score = 0.0

        return profile

    def _aggregate_type(
        self,
        etype: EntityType,
        profiles: list[EntityDynamicsProfile],
    ) -> TypeDynamicsProfile:
        """Compute aggregate dynamics for an entity type."""
        tp = TypeDynamicsProfile(
            entity_type=etype,
            entity_count=len(profiles),
        )

        if not profiles:
            return tp

        # Mean transitions per entity
        total_t = sum(p.total_transitions for p in profiles)
        tp.mean_transitions_per_entity = total_t / len(profiles)

        # Mean interval across entities that have intervals
        intervals = [p.mean_interval_s for p in profiles if p.mean_interval_s > 0]
        if intervals:
            tp.mean_interval_s = sum(intervals) / len(intervals)
            sorted_intervals = sorted(intervals)
            tp.median_interval_s = sorted_intervals[len(sorted_intervals) // 2]

        # Contradiction rate
        total_contradictions = sum(p.contradiction_count for p in profiles)
        if total_t > 0:
            tp.contradiction_rate = total_contradictions / total_t

        # Archival rate
        archived = sum(1 for p in profiles if p.has_been_archived)
        tp.archival_rate = archived / len(profiles)

        return tp

    def _detect_cooccurrences(
        self,
        window_s: float = 86400.0,  # 24 hours
        min_count: int = 2,
    ) -> list[TransitionCooccurrence]:
        """
        Find pairs of entities that tend to transition close together in time.

        This is a causal signal: if entity A transitioning often precedes
        entity B transitioning within `window_s` seconds, there's likely a
        causal or correlational relationship.
        """
        # Build timeline: [(timestamp, entity_id)]
        timeline: list[tuple[float, str]] = []
        for tid, t in self.wm.transitions.items():
            timeline.append((t.timestamp, t.entity_id))
        timeline.sort()

        # Count co-occurrences within window
        pair_counts: Counter = Counter()
        pair_lags: dict[tuple[str, str], list[float]] = defaultdict(list)

        for i, (ts_a, eid_a) in enumerate(timeline):
            for j in range(i + 1, len(timeline)):
                ts_b, eid_b = timeline[j]
                lag = ts_b - ts_a
                if lag > window_s:
                    break
                if eid_a == eid_b:
                    continue
                pair = (eid_a, eid_b) if eid_a < eid_b else (eid_b, eid_a)
                pair_counts[pair] += 1
                pair_lags[pair].append(lag)

        # Filter to significant co-occurrences
        results = []
        for (eid_a, eid_b), count in pair_counts.most_common(20):
            if count < min_count:
                break
            lags = pair_lags[(eid_a, eid_b)]
            mean_lag = sum(lags) / len(lags)

            entity_a = self.wm.entities.get(eid_a)
            entity_b = self.wm.entities.get(eid_b)
            if not entity_a or not entity_b:
                continue

            # Confidence based on count relative to total transitions of each entity
            trans_a = len(self.wm.get_transitions(eid_a))
            trans_b = len(self.wm.get_transitions(eid_b))
            min_trans = min(trans_a, trans_b) if min(trans_a, trans_b) > 0 else 1
            confidence = min(1.0, count / min_trans)

            results.append(TransitionCooccurrence(
                entity_a_id=eid_a,
                entity_b_id=eid_b,
                entity_a_name=entity_a.name,
                entity_b_name=entity_b.name,
                cooccurrence_count=count,
                typical_lag_s=mean_lag,
                confidence=confidence,
            ))

        return results

    def _predict_next(
        self,
        entity_profiles: dict[str, EntityDynamicsProfile],
        type_profiles: dict[str, TypeDynamicsProfile],
        reference_time: float,
    ) -> list[dict]:
        """
        Predict which entities are likely to transition next.

        Uses:
          1. Entity's own transition history (interval regularity)
          2. Type-level priors (how often this type of entity changes)
          3. Staleness score
          4. Volatility

        Returns list of predictions sorted by confidence.
        """
        predictions = []

        for eid, profile in entity_profiles.items():
            if profile.total_transitions < 2:
                continue

            # Time since last transition
            time_since = reference_time - profile.last_transition_ts
            if time_since <= 0:
                continue

            # Expected number of transitions based on entity's own rate
            if profile.mean_interval_s > 0:
                expected = time_since / profile.mean_interval_s
            else:
                continue

            # Confidence increases with overdue-ness and predictability
            confidence = min(1.0, expected * profile.predictability * 0.5)

            if confidence < 0.1:
                continue

            # Predict most likely transition type (excluding creation)
            likely_type = "update"  # default
            if profile.transition_type_counts:
                non_creation = {
                    k: v for k, v in profile.transition_type_counts.items()
                    if k != "creation"
                }
                if non_creation:
                    likely_type = max(non_creation, key=non_creation.get)

            # Build reason string
            days_since = time_since / 86400.0
            avg_days = profile.mean_interval_s / 86400.0 if profile.mean_interval_s > 0 else 0

            reason = (
                f"Last transition {days_since:.0f} days ago "
                f"(avg interval: {avg_days:.0f} days, "
                f"volatility: {profile.volatility:.2f})"
            )

            predictions.append({
                "entity_id": eid,
                "entity_name": profile.entity_name,
                "predicted_type": likely_type,
                "confidence": round(confidence, 3),
                "staleness": round(profile.staleness_score, 3),
                "reason": reason,
            })

        predictions.sort(key=lambda x: x["confidence"], reverse=True)
        return predictions[:20]

    # ── Level 2: Transition Sequence Patterns ────────────────────────────────

    def extract_transition_sequences(
        self,
        min_length: int = 2,
        min_support: int = 2,
    ) -> list[dict]:
        """
        Find recurring transition sequences across entities.

        This discovers patterns like:
          - "new job" → "city move" → "apartment search" (career change cascade)
          - "argument" → "silence" → "resolution" (relationship pattern)
          - "error" → "retry" → "success" (agent learning pattern)

        These are the precursors to procedural knowledge.
        """
        # Build per-entity transition type sequences
        sequences: dict[str, list[tuple[float, str]]] = defaultdict(list)
        for tid, t in self.wm.transitions.items():
            sequences[t.entity_id].append((t.timestamp, t.transition_type.value))

        # Sort each by timestamp
        for eid in sequences:
            sequences[eid].sort()

        # Extract n-grams of transition types per entity
        ngram_counts: Counter = Counter()
        ngram_examples: dict[tuple, list[str]] = defaultdict(list)

        for eid, seq in sequences.items():
            types = [t for _, t in seq]
            for n in range(min_length, min(len(types) + 1, 6)):
                for i in range(len(types) - n + 1):
                    ngram = tuple(types[i:i + n])
                    ngram_counts[ngram] += 1
                    if len(ngram_examples[ngram]) < 3:
                        ngram_examples[ngram].append(eid)

        # Filter to significant patterns
        patterns = []
        for ngram, count in ngram_counts.most_common(30):
            if count < min_support:
                break
            example_names = [
                self.wm.entities[eid].name
                for eid in ngram_examples[ngram]
                if eid in self.wm.entities
            ]
            patterns.append({
                "sequence": list(ngram),
                "count": count,
                "example_entities": example_names,
            })

        return patterns

    # ── Level 3: LLM-based Forward Simulation ────────────────────────────────

    def build_simulation_prompt(
        self,
        entity_id: str,
        scenario: str = "",
    ) -> str:
        """
        Build a prompt for LLM-based forward simulation.

        Given an entity's full transition history and an optional scenario
        ("what if the user gets a new job?"), generate a prompt that asks
        the LLM to predict the next state.

        This is the equivalent of Dreamer's "imagination" step, but using
        an LLM as the dynamics model instead of a learned neural network.
        """
        entity = self.wm.entities.get(entity_id)
        if not entity:
            return ""

        transitions = self.wm.get_transitions(entity_id, ordered=True)

        # Build timeline narrative
        timeline_parts = []
        for t in transitions:
            dt = datetime.fromtimestamp(t.timestamp, tz=timezone.utc)
            date_str = dt.strftime("%B %d, %Y")
            ttype = t.transition_type.value.upper()
            state_desc = ""
            if isinstance(t.to_state, dict):
                state_desc = "; ".join(f"{k}: {v}" for k, v in t.to_state.items() if v)
            else:
                state_desc = str(t.to_state)[:200]

            timeline_parts.append(f"  [{date_str}] {ttype}: {state_desc}")
            if t.trigger_summary:
                timeline_parts.append(f"    Trigger: {t.trigger_summary}")

        timeline_str = "\n".join(timeline_parts)

        # Dynamics statistics
        profile = self._analyze_entity(
            entity_id, entity,
            transitions[-1].timestamp if transitions else 0.0
        )

        stats = (
            f"Transitions: {profile.total_transitions} | "
            f"Avg interval: {profile.mean_interval_s / 86400:.0f} days | "
            f"Volatility: {profile.volatility:.2f} | "
            f"Contradictions: {profile.contradiction_count}"
        )

        prompt = f"""You are a world model dynamics predictor. Given an entity's complete
transition history, predict what will happen next.

## Entity: {entity.name} ({entity.type.value})
## Current State: {entity.current_state}

## Transition History:
{timeline_str}

## Dynamics Statistics:
{stats}

{"## Scenario: " + scenario if scenario else ""}

## Task:
Based on the transition history and dynamics, predict:
1. What is the most likely NEXT transition for this entity?
2. When is it likely to happen (relative to the last transition)?
3. What will the new state be?
4. What would trigger this transition?
5. What other entities might be affected (co-occurrence)?

Respond as JSON:
{{
  "predicted_transition_type": "update|contradiction|resolution|archival",
  "predicted_timeframe": "description of when",
  "predicted_new_state": {{"key": "value"}},
  "likely_trigger": "what would cause this",
  "affected_entities": ["entity names that might also change"],
  "confidence": 0.0-1.0,
  "reasoning": "explanation"
}}"""
        return prompt

    # ── Utility: Human-readable report ───────────────────────────────────────

    def summarize(self, reference_time: float | None = None) -> str:
        """Generate a human-readable dynamics summary."""
        report = self.analyze(reference_time)

        lines = [
            "# World Model Dynamics Report",
            f"\nEntities: {len(report.entity_profiles)}",
            f"Stale: {len(report.stale_entities)}",
            f"Volatile: {len(report.volatile_entities)}",
            f"Co-occurrences found: {len(report.cooccurrences)}",
        ]

        # Type profiles
        if report.type_profiles:
            lines.append("\n## Entity Type Dynamics")
            for tname, tp in sorted(report.type_profiles.items()):
                avg_days = tp.mean_interval_s / 86400.0 if tp.mean_interval_s > 0 else 0
                lines.append(
                    f"  {tname}: {tp.entity_count} entities, "
                    f"{tp.mean_transitions_per_entity:.1f} transitions avg, "
                    f"~{avg_days:.0f} days between changes, "
                    f"{tp.contradiction_rate:.0%} contradiction rate"
                )

        # Stale entities
        if report.stale_entities:
            lines.append("\n## Stale Entities (likely needing update)")
            for eid in report.stale_entities[:10]:
                p = report.entity_profiles[eid]
                lines.append(f"  - {p.entity_name}: staleness={p.staleness_score:.2f}")

        # Volatile entities
        if report.volatile_entities:
            lines.append("\n## Volatile Entities (high change frequency)")
            for eid in report.volatile_entities[:10]:
                p = report.entity_profiles[eid]
                lines.append(f"  - {p.entity_name}: volatility={p.volatility:.2f}")

        # Co-occurrences
        if report.cooccurrences:
            lines.append("\n## Transition Co-occurrences (causal signals)")
            for co in report.cooccurrences[:10]:
                lag_hours = co.typical_lag_s / 3600.0
                lines.append(
                    f"  - {co.entity_a_name} ↔ {co.entity_b_name}: "
                    f"{co.cooccurrence_count}x (lag: {lag_hours:.1f}h, "
                    f"conf: {co.confidence:.2f})"
                )

        # Predictions
        if report.predicted_next_transitions:
            lines.append("\n## Predicted Next Transitions")
            for pred in report.predicted_next_transitions[:10]:
                lines.append(
                    f"  - {pred['entity_name']}: likely {pred['predicted_type']} "
                    f"(conf: {pred['confidence']:.2f}) — {pred['reason']}"
                )

        # Transition sequences
        sequences = self.extract_transition_sequences()
        if sequences:
            lines.append("\n## Recurring Transition Patterns")
            for seq in sequences[:5]:
                pattern_str = " → ".join(seq["sequence"])
                lines.append(
                    f"  - {pattern_str} ({seq['count']}x) "
                    f"e.g. {', '.join(seq['example_entities'][:2])}"
                )

        return "\n".join(lines)
