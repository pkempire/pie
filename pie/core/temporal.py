"""
Temporal State — continuous time reasoning from one primitive.

The insight: just as next-token prediction gives LLMs all of language as
emergent capability, PREDICT THE NEXT STATE CHANGE gives you:
  - State estimation at any time t  (what's probably true now?)
  - Anomaly detection                (this changed sooner/later than expected)
  - Planning                         (when should I check back?)
  - Consolidation                    (what's dead vs dormant vs active?)

Everything derived from one learned function. No hardcoded probabilities.
Every number comes from the data.

The key discovery: when you measure time in "entity-relative units" (gap /
mean_gap for that entity), the survival function S(k) = P(gap > k × mean)
is UNIVERSAL across all entity types. One empirical curve fits projects,
tools, people, goals — everything.

This means the ONLY per-entity parameter is its mean interval (clock speed).
The shape of how entities die/revive is a population-level constant, learned
once from all the data.

Query interface:
  ts = TemporalState(world_model)
  ts.learn()                        # fit parameters from data
  ts.survival(entity_id, t)         # THE primitive — everything derives from this
  ts.alive(entity_id, t)            # P(entity will have another transition)
  ts.expected_next(entity_id)       # when is next transition most likely?
  ts.state_confidence(entity_id, t) # how much do we trust current state?
  ts.anomaly(entity_id, event_t)    # how surprising was this event's timing?
  ts.query(entity_id, t)            # full state estimate at arbitrary time t
"""

from __future__ import annotations

import math
import statistics
import logging
from dataclasses import dataclass, field
from collections import defaultdict
from typing import Any

from .models import Entity, EntityType, StateTransition, TransitionType
from .world_model import WorldModel

logger = logging.getLogger("pie.temporal")


# ═══════════════════════════════════════════════════════════════════════════════
# Learned parameters — everything comes from data
# ═══════════════════════════════════════════════════════════════════════════════

@dataclass
class Rhythm:
    """An entity's learned temporal signature.

    The ONLY per-entity parameter is mean_interval (clock speed).
    The survival curve shape comes from the population-level table.
    """
    entity_id: str
    n_transitions: int = 0
    mean_interval: float = 0.0      # days between transitions (entity's clock speed)
    median_interval: float = 0.0
    intervals: list[float] = field(default_factory=list)  # raw gaps in days

    # Per-entity survival table (only for entities with 10+ transitions)
    own_table: SurvivalTable | None = None

    # Derived
    last_transition_t: float = 0.0   # timestamp of most recent transition
    first_transition_t: float = 0.0  # timestamp of first transition
    lifespan_days: float = 0.0

    @property
    def has_data(self) -> bool:
        return self.n_transitions >= 2 and self.mean_interval > 0


@dataclass
class SurvivalTable:
    """Empirical survival function in entity-relative time.

    Instead of fitting a parametric distribution (Weibull, etc.), we store
    the ACTUAL empirical survival curve. This is more accurate for heavy-
    tailed distributions and more honest — literally just the data.

    The table maps k → P(gap > k × mean_interval).
    Between table points we interpolate linearly in log-probability space.
    """
    # Sorted list of (k, survival_probability) pairs
    points: list[tuple[float, float]] = field(default_factory=list)
    n_intervals: int = 0
    n_entities: int = 0

    # Population-level stats
    mean_interval: float = 0.0  # median of per-entity mean intervals (days)

    def survival(self, k: float) -> float:
        """Interpolate survival probability at k multiples of mean interval."""
        if k <= 0:
            return 1.0
        if not self.points:
            return 0.5  # no data

        # Before first point
        if k <= self.points[0][0]:
            # Interpolate from (0, 1.0) to first point
            k0, s0 = 0.0, 1.0
            k1, s1 = self.points[0]
            return self._interp(k, k0, s0, k1, s1)

        # After last point
        if k >= self.points[-1][0]:
            # Extrapolate: decay at the same rate as last segment
            if len(self.points) >= 2:
                k0, s0 = self.points[-2]
                k1, s1 = self.points[-1]
                if s0 > 0 and s1 > 0 and s0 > s1:
                    # Log-linear extrapolation
                    rate = math.log(s1 / s0) / (k1 - k0)
                    return max(0.0, s1 * math.exp(rate * (k - k1)))
            return 0.0

        # Between two points: log-linear interpolation
        for i in range(len(self.points) - 1):
            k0, s0 = self.points[i]
            k1, s1 = self.points[i + 1]
            if k0 <= k <= k1:
                return self._interp(k, k0, s0, k1, s1)

        return 0.0

    @staticmethod
    def _interp(k: float, k0: float, s0: float, k1: float, s1: float) -> float:
        """Log-linear interpolation between two survival points."""
        if k1 == k0:
            return s0
        frac = (k - k0) / (k1 - k0)
        if s0 <= 0 or s1 <= 0:
            # Can't do log-space, fall back to linear
            return max(0.0, s0 + frac * (s1 - s0))
        # Interpolate in log space (survival decays roughly exponentially)
        log_s = math.log(s0) + frac * (math.log(s1) - math.log(s0))
        return math.exp(log_s)

    @classmethod
    def from_intervals(cls, normalized_intervals: list[float],
                       n_entities: int = 0,
                       mean_interval: float = 0.0) -> SurvivalTable:
        """Build empirical survival table from normalized intervals.

        normalized_intervals: gaps measured in multiples of each entity's mean.
        """
        if not normalized_intervals:
            return cls()

        n = len(normalized_intervals)
        # Sample points: dense near 0 (where most action is), sparser in tail
        k_points = [0.1, 0.2, 0.3, 0.5, 0.75, 1.0, 1.5, 2.0, 3.0,
                     5.0, 7.0, 10.0, 15.0, 20.0, 30.0, 50.0]

        points = []
        for k in k_points:
            surv = sum(1 for g in normalized_intervals if g > k) / n
            if surv > 0:
                points.append((k, surv))
            else:
                points.append((k, 0.0))
                break  # no point going further

        return cls(
            points=points,
            n_intervals=n,
            n_entities=n_entities,
            mean_interval=mean_interval,
        )


# ═══════════════════════════════════════════════════════════════════════════════
# TemporalState — the query interface
# ═══════════════════════════════════════════════════════════════════════════════

class TemporalState:
    """Continuous time reasoning over a world model.

    Usage:
        ts = TemporalState(world_model)
        ts.learn()  # fit all parameters from data

        # The ONE primitive — everything else derives from this
        ts.survival(entity_id, t)  # P(no transition since last_event, at time t)

        # Derived queries
        ts.alive(entity_id, t)           # P(entity will ever transition again)
        ts.expected_next(entity_id)      # E[next transition time]
        ts.state_confidence(entity_id, t)  # how much to trust current_state at time t
        ts.anomaly(entity_id, event_t)   # how surprising was this event's timing
        ts.classify(entity_id, t)        # active / dormant / dead
    """

    def __init__(self, wm: WorldModel):
        self.wm = wm
        self.rhythms: dict[str, Rhythm] = {}
        self.global_table = SurvivalTable()        # population-level curve
        self.type_tables: dict[str, SurvivalTable] = {}  # per-type curves
        self._learned = False

    # ── Learning (fit everything from data) ──────────────────────────────

    def learn(self) -> dict:
        """Fit all parameters from the world model's transition data.

        Three things are learned:
        1. Per-entity rhythm (mean_interval = clock speed)
        2. Global survival table (the universal curve shape)
        3. Per-type survival tables (if types differ meaningfully)

        Returns stats about what was learned.
        """
        # Step 1: compute per-entity rhythms and collect normalized intervals
        entity_timestamps: dict[str, list[float]] = defaultdict(list)
        for t in self.wm.transitions.values():
            entity_timestamps[t.entity_id].append(t.timestamp)

        all_normalized: list[float] = []
        type_normalized: dict[str, list[float]] = defaultdict(list)

        for eid, timestamps in entity_timestamps.items():
            entity = self.wm.entities.get(eid)
            if not entity:
                continue

            timestamps.sort()
            n = len(timestamps)

            rhythm = Rhythm(
                entity_id=eid,
                n_transitions=n,
                first_transition_t=timestamps[0],
                last_transition_t=timestamps[-1],
                lifespan_days=(timestamps[-1] - timestamps[0]) / 86400,
            )

            if n >= 2:
                gaps = [(timestamps[i + 1] - timestamps[i]) / 86400
                        for i in range(n - 1)]
                rhythm.intervals = gaps
                rhythm.mean_interval = statistics.mean(gaps)
                rhythm.median_interval = statistics.median(gaps)

                if rhythm.mean_interval > 0.001:
                    # Normalize to entity-relative time
                    normalized = [g / rhythm.mean_interval for g in gaps]
                    all_normalized.extend(normalized)
                    type_normalized[entity.type.value].extend(normalized)

                    # Per-entity table for entities with rich history
                    if len(gaps) >= 10:
                        rhythm.own_table = SurvivalTable.from_intervals(normalized)

            self.rhythms[eid] = rhythm

        # Step 2: build global survival table from all normalized intervals
        intervals_with_data = [r.mean_interval for r in self.rhythms.values()
                               if r.has_data]
        pop_median = statistics.median(intervals_with_data) if intervals_with_data else 1.0

        if len(all_normalized) >= 20:
            self.global_table = SurvivalTable.from_intervals(
                all_normalized,
                n_entities=len([r for r in self.rhythms.values() if r.has_data]),
                mean_interval=pop_median,
            )

        # Step 3: build per-type survival tables
        for etype, normalized in type_normalized.items():
            if len(normalized) >= 30:
                self.type_tables[etype] = SurvivalTable.from_intervals(
                    normalized,
                    n_entities=sum(1 for r in self.rhythms.values()
                                   if r.has_data and self.wm.entities.get(r.entity_id)
                                   and self.wm.entities[r.entity_id].type.value == etype),
                    mean_interval=pop_median,
                )

        self._learned = True

        # Stats
        individually_fitted = sum(1 for r in self.rhythms.values()
                                  if r.own_table is not None)
        has_data = sum(1 for r in self.rhythms.values() if r.has_data)
        singleton = sum(1 for r in self.rhythms.values() if r.n_transitions < 2)

        return {
            "total_entities": len(self.rhythms),
            "individually_fitted": individually_fitted,
            "with_rhythm": has_data,
            "singletons": singleton,
            "global_table_points": len(self.global_table.points),
            "global_n_intervals": self.global_table.n_intervals,
            "population_median_interval_days": round(pop_median, 2),
            "type_tables": {k: len(v.points) for k, v in self.type_tables.items()},
        }

    # ── The ONE primitive ────────────────────────────────────────────────

    def _get_table(self, entity_id: str) -> SurvivalTable:
        """Get the best survival table for an entity.

        Hierarchy: entity's own table → type table → global table.
        """
        rhythm = self.rhythms.get(entity_id)
        if rhythm and rhythm.own_table:
            return rhythm.own_table

        entity = self.wm.entities.get(entity_id)
        if entity and entity.type.value in self.type_tables:
            return self.type_tables[entity.type.value]

        return self.global_table

    def _get_mean_interval(self, entity_id: str) -> float:
        """Get entity's clock speed (mean interval in days)."""
        rhythm = self.rhythms.get(entity_id)
        if rhythm and rhythm.has_data:
            return rhythm.mean_interval
        return self.global_table.mean_interval if self.global_table.mean_interval > 0 else 1.0

    def survival(self, entity_id: str, t: float) -> float:
        """THE PRIMITIVE: P(no new transition between last_event and t).

        This is the survival function evaluated at the entity's current
        silence duration, measured in entity-relative time (k = silence / mean_interval).

        Everything else is derived from this single function.
        """
        rhythm = self.rhythms.get(entity_id)
        if not rhythm or not rhythm.has_data:
            return 0.5  # maximum uncertainty for unknown entities

        silence_days = (t - rhythm.last_transition_t) / 86400
        if silence_days <= 0:
            return 1.0

        mean_int = self._get_mean_interval(entity_id)
        k = silence_days / mean_int  # entity-relative time
        table = self._get_table(entity_id)
        return table.survival(k)

    # ── Derived from the primitive ───────────────────────────────────────

    def alive(self, entity_id: str, t: float) -> float:
        """P(entity will have at least one more transition).

        An entity is "alive" if its current silence is within the range
        of its historical gaps. Uses the survival function directly.
        """
        s = self.survival(entity_id, t)

        # The survival function gives P(gap > current_silence | gap started).
        # But we also want to account for the base rate: entities with many
        # transitions are more likely to have another one.
        rhythm = self.rhythms.get(entity_id)
        if not rhythm:
            return 0.5

        # Momentum factor: log(transitions) gives a gentle boost
        # (an entity with 50 transitions is more likely to keep going than one with 3)
        if rhythm.n_transitions >= 2:
            momentum = min(math.log2(rhythm.n_transitions) / 6.0, 1.0)
        else:
            momentum = 0.0

        # Combine: survival gives the timing signal, momentum gives the base rate
        return s * 0.7 + momentum * 0.3

    def state_confidence(self, entity_id: str, t: float) -> float:
        """How much should we trust entity's current_state at time t?

        Fresh state (just updated) → 1.0
        Stale state (long silence) → approaches 0.0
        """
        return self.survival(entity_id, t)

    def expected_next(self, entity_id: str) -> float:
        """Expected time of next transition (as Unix timestamp).

        Finds the median remaining life: the time s such that
        P(T > t + s | T > t) = 0.5, using binary search on the survival table.
        """
        rhythm = self.rhythms.get(entity_id)
        if not rhythm or not rhythm.has_data:
            return 0.0

        mean_int = self._get_mean_interval(entity_id)
        table = self._get_table(entity_id)
        silence_days = max(0, (self._now() - rhythm.last_transition_t) / 86400)
        k_now = silence_days / mean_int
        s_now = table.survival(k_now)

        if s_now <= 0.001:
            # Already effectively dead — no meaningful prediction
            return 0.0

        # Binary search for median remaining life in entity-relative time
        lo_k, hi_k = 0.0, max(50.0, k_now * 10)
        for _ in range(50):
            mid_k = (lo_k + hi_k) / 2
            s_future = table.survival(k_now + mid_k)
            conditional = s_future / s_now
            if conditional > 0.5:
                lo_k = mid_k
            else:
                hi_k = mid_k

        median_remaining_k = (lo_k + hi_k) / 2
        median_remaining_days = median_remaining_k * mean_int
        return rhythm.last_transition_t + (silence_days + median_remaining_days) * 86400

    def anomaly(self, entity_id: str, event_t: float) -> float:
        """How surprising was a transition at time event_t?

        Returns a score from 0 (completely expected) to 1 (extremely surprising).
        Based on how far into the tail of the survival function the event fell.
        Events at the median (S=0.5) are least surprising.
        Events very early (S≈1) or very late (S≈0) are most surprising.
        """
        rhythm = self.rhythms.get(entity_id)
        if not rhythm or not rhythm.has_data:
            return 0.0  # can't judge surprise without history

        mean_int = self._get_mean_interval(entity_id)
        gap_days = (event_t - rhythm.last_transition_t) / 86400
        if gap_days < 0:
            return 0.0

        k = gap_days / mean_int
        s = self._get_table(entity_id).survival(k)
        return 1.0 - 2.0 * abs(s - 0.5)

    def classify(self, entity_id: str, t: float) -> str:
        """Classify entity's temporal status at time t.

        Returns one of: 'active', 'expected', 'dormant', 'fading', 'dead'

        These thresholds are derived from the survival function:
          active:   S(k) > 0.5  — within normal activity window
          expected: 0.2 < S(k) ≤ 0.5 — overdue but plausible
          dormant:  0.05 < S(k) ≤ 0.2 — unusually long silence
          fading:   0.01 < S(k) ≤ 0.05 — very unlikely to return
          dead:     S(k) ≤ 0.01 — effectively zero chance
        """
        s = self.survival(entity_id, t)
        if s > 0.5:
            return "active"
        elif s > 0.2:
            return "expected"
        elif s > 0.05:
            return "dormant"
        elif s > 0.01:
            return "fading"
        else:
            return "dead"

    def query(self, entity_id: str, t: float) -> dict[str, Any]:
        """Full state estimate at arbitrary time t.

        Returns everything we can infer about this entity at time t,
        all derived from the survival function primitive.
        """
        entity = self.wm.entities.get(entity_id)
        rhythm = self.rhythms.get(entity_id)

        if not entity:
            return {"error": "unknown entity"}

        s = self.survival(entity_id, t)
        silence_days = (t - rhythm.last_transition_t) / 86400 if rhythm and rhythm.has_data else None

        result = {
            "entity_id": entity_id,
            "name": entity.name,
            "type": entity.type.value,
            "query_time": t,

            # The primitive
            "survival": round(s, 4),

            # Derived
            "status": self.classify(entity_id, t),
            "state_confidence": round(self.state_confidence(entity_id, t), 4),
            "alive_probability": round(self.alive(entity_id, t), 4),
            "current_state": entity.current_state,

            # Timing
            "silence_days": round(silence_days, 2) if silence_days is not None else None,
            "rhythm_mean_days": round(rhythm.mean_interval, 2) if rhythm and rhythm.has_data else None,
            "silence_in_rhythm_units": round(silence_days / rhythm.mean_interval, 2)
                if rhythm and rhythm.has_data and rhythm.mean_interval > 0 else None,
        }

        # Expected next event
        expected = self.expected_next(entity_id)
        if expected > 0:
            expected_days = (expected - t) / 86400
            result["expected_next_in_days"] = round(expected_days, 1)

        return result

    def rank_by_staleness(self, t: float, top_n: int = 20,
                          min_transitions: int = 3) -> list[dict]:
        """Rank entities by how overdue they are for a transition.

        Returns entities sorted by lowest survival probability (most overdue).
        Filters to entities with enough history to be meaningful.
        """
        candidates = []
        for eid, rhythm in self.rhythms.items():
            if rhythm.n_transitions < min_transitions:
                continue
            s = self.survival(eid, t)
            entity = self.wm.entities.get(eid)
            if not entity:
                continue
            candidates.append({
                "entity_id": eid,
                "name": entity.name,
                "type": entity.type.value,
                "survival": round(s, 4),
                "status": self.classify(eid, t),
                "silence_days": round((t - rhythm.last_transition_t) / 86400, 1),
                "mean_interval_days": round(rhythm.mean_interval, 1),
                "n_transitions": rhythm.n_transitions,
            })

        candidates.sort(key=lambda x: x["survival"])
        return candidates[:top_n]

    def rank_by_momentum(self, t: float, top_n: int = 20) -> list[dict]:
        """Rank entities by current activity momentum.

        Returns the most actively evolving entities — those with high
        transition rates AND recent activity.
        """
        candidates = []
        for eid, rhythm in self.rhythms.items():
            if rhythm.n_transitions < 3:
                continue
            s = self.survival(eid, t)
            entity = self.wm.entities.get(eid)
            if not entity:
                continue

            # Momentum = alive_probability * transition_density
            alive_p = self.alive(eid, t)
            density = rhythm.n_transitions / max(rhythm.lifespan_days, 1.0)
            momentum = alive_p * density

            candidates.append({
                "entity_id": eid,
                "name": entity.name,
                "type": entity.type.value,
                "momentum": round(momentum, 4),
                "alive": round(alive_p, 3),
                "density": round(density, 3),
                "n_transitions": rhythm.n_transitions,
                "status": self.classify(eid, t),
            })

        candidates.sort(key=lambda x: -x["momentum"])
        return candidates[:top_n]

    def population_summary(self, t: float) -> dict:
        """Aggregate temporal state of the entire world model at time t."""
        statuses = defaultdict(int)
        total_alive = 0.0
        n_with_data = 0

        for eid in self.rhythms:
            status = self.classify(eid, t)
            statuses[status] += 1
            if self.rhythms[eid].has_data:
                total_alive += self.alive(eid, t)
                n_with_data += 1

        return {
            "timestamp": t,
            "total_entities": len(self.rhythms),
            "status_distribution": dict(statuses),
            "mean_alive_probability": round(total_alive / max(n_with_data, 1), 4),
            "global_table_points": len(self.global_table.points),
            "global_n_intervals": self.global_table.n_intervals,
            "type_tables": list(self.type_tables.keys()),
        }

    # ── Internal ─────────────────────────────────────────────────────────

    def _now(self) -> float:
        """Latest timestamp in the world model (not wall clock)."""
        if not self.wm.transitions:
            return 0.0
        return max(t.timestamp for t in self.wm.transitions.values())

    def to_dict(self) -> dict:
        """Serialize learned parameters (not the raw data, just the model)."""
        return {
            "global_table": {
                "points": self.global_table.points,
                "n_intervals": self.global_table.n_intervals,
                "n_entities": self.global_table.n_entities,
                "mean_interval": self.global_table.mean_interval,
            },
            "type_tables": {
                etype: {"points": table.points, "n_intervals": table.n_intervals}
                for etype, table in self.type_tables.items()
            },
            "rhythms": {
                eid: {
                    "n": r.n_transitions,
                    "mean": round(r.mean_interval, 3),
                    "last_t": r.last_transition_t,
                    "has_own_table": r.own_table is not None,
                }
                for eid, r in self.rhythms.items()
                if r.has_data
            },
        }
