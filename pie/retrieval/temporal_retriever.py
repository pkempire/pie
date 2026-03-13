"""
Temporal-Aware Retriever for PIE

Combines semantic similarity with temporal filtering and decay.
No magic numbers — all constants from config with documented derivations.

Usage:
    from pie.config import PIEConfig
    retriever = TemporalRetriever(world_model, config.retrieval, config.temporal)
    results = retriever.retrieve(query, query_time)
"""

from __future__ import annotations

import re
import math
from datetime import datetime, timedelta
from dataclasses import dataclass
from typing import Optional

import numpy as np

from pie.config import RetrievalConfig, TemporalConfig


# =============================================================================
# Temporal Reference Extraction
# =============================================================================

@dataclass
class TemporalRef:
    """Extracted temporal reference from a query."""
    start: Optional[datetime]
    end: Optional[datetime]
    operator: str  # "during", "before", "after", "around"
    raw_text: str
    confidence: float  # From config, not arbitrary


@dataclass
class RetrievalResult:
    """A single retrieval result with scores."""
    entity_id: str
    entity_name: str
    entity_type: str
    content: str
    
    # Temporal data (both raw and semantic)
    timestamp: Optional[datetime]
    date_iso: Optional[str]
    relative_time: Optional[str]
    period: Optional[str]
    
    # Scores (for debugging/analysis)
    semantic_score: float
    temporal_score: float
    recency_score: float
    combined_score: float


class TemporalRefExtractor:
    """
    Extract temporal references from natural language queries.
    
    Confidence values come from TemporalConfig, not hardcoded.
    """
    
    # Pattern definitions (the patterns themselves are not arbitrary —
    # they're standard English temporal expressions)
    RELATIVE_PATTERNS = [
        (r"(\d+)\s*(day|week|month|year)s?\s*ago", "ago"),
        (r"last\s*(day|week|month|year)", "last"),
        (r"yesterday", "yesterday"),
        (r"today", "today"),
        (r"this\s*(week|month|year)", "this"),
        (r"past\s*(\d+)?\s*(day|week|month|year)s?", "past"),
    ]
    
    ABSOLUTE_PATTERNS = [
        (r"in\s*(january|february|march|april|may|june|july|august|september|october|november|december)\s*(\d{4})?", "month"),
        (r"on\s*(\w+)\s+(\d{1,2})(?:st|nd|rd|th)?(?:,?\s*(\d{4}))?", "date"),
        (r"in\s*(\d{4})", "year"),
        (r"(q[1-4])\s*(\d{4})?", "quarter"),
    ]
    
    OPERATOR_PATTERNS = [
        (r"before\s+", "before"),
        (r"after\s+", "after"),
        (r"during\s+", "during"),
        (r"around\s+", "around"),
        (r"since\s+", "after"),
        (r"until\s+", "before"),
    ]
    
    MONTH_MAP = {
        "january": 1, "february": 2, "march": 3, "april": 4,
        "may": 5, "june": 6, "july": 7, "august": 8,
        "september": 9, "october": 10, "november": 11, "december": 12
    }
    
    def __init__(self, config: TemporalConfig, reference_time: Optional[datetime] = None):
        self.config = config
        self.reference_time = reference_time or datetime.now()
    
    def extract(self, query: str) -> Optional[TemporalRef]:
        """Extract temporal reference from query."""
        query_lower = query.lower()
        
        # Check for operator first
        operator = "during"  # default
        for pattern, op in self.OPERATOR_PATTERNS:
            if re.search(pattern, query_lower):
                operator = op
                break
        
        # Try relative patterns (highest confidence)
        for pattern, ptype in self.RELATIVE_PATTERNS:
            match = re.search(pattern, query_lower)
            if match:
                return self._parse_relative(match, ptype, operator, 
                                           confidence=self.config.relative_confidence)
        
        # Try absolute patterns (highest confidence)
        for pattern, ptype in self.ABSOLUTE_PATTERNS:
            match = re.search(pattern, query_lower)
            if match:
                return self._parse_absolute(match, ptype, operator,
                                           confidence=self.config.explicit_confidence)
        
        # Check for implicit temporal cues (lowest confidence)
        if any(word in query_lower for word in ["recently", "lately", "just"]):
            return TemporalRef(
                start=self.reference_time - timedelta(days=self.config.implicit_recent_days),
                end=self.reference_time,
                operator="during",
                raw_text="recently",
                confidence=self.config.implicit_confidence
            )
        
        if any(word in query_lower for word in ["earlier", "previously", "before"]):
            return TemporalRef(
                start=None,
                end=self.reference_time,
                operator="before",
                raw_text="earlier",
                confidence=self.config.implicit_confidence
            )
        
        return None
    
    def _parse_relative(self, match, ptype: str, operator: str, confidence: float) -> TemporalRef:
        """Parse relative time expressions."""
        ref = self.reference_time
        
        if ptype == "yesterday":
            start = ref - timedelta(days=1)
            end = start + timedelta(days=1)
        elif ptype == "today":
            start = ref.replace(hour=0, minute=0, second=0, microsecond=0)
            end = ref
        elif ptype == "ago":
            num = int(match.group(1))
            unit = match.group(2)
            delta = self._unit_to_delta(unit, num)
            start = ref - delta
            end = ref
        elif ptype == "last":
            unit = match.group(1)
            delta = self._unit_to_delta(unit, 1)
            start = ref - delta
            end = ref
        elif ptype == "this":
            unit = match.group(1)
            if unit == "week":
                start = ref - timedelta(days=ref.weekday())
                end = ref
            elif unit == "month":
                start = ref.replace(day=1)
                end = ref
            elif unit == "year":
                start = ref.replace(month=1, day=1)
                end = ref
            else:
                start = ref - timedelta(days=7)
                end = ref
        elif ptype == "past":
            num = int(match.group(1)) if match.group(1) else 1
            unit = match.group(2)
            delta = self._unit_to_delta(unit, num)
            start = ref - delta
            end = ref
        else:
            start = ref - timedelta(days=7)
            end = ref
        
        return TemporalRef(
            start=start,
            end=end,
            operator=operator,
            raw_text=match.group(0),
            confidence=confidence
        )
    
    def _parse_absolute(self, match, ptype: str, operator: str, confidence: float) -> TemporalRef:
        """Parse absolute time expressions."""
        ref = self.reference_time
        
        if ptype == "month":
            month_name = match.group(1).lower()
            month = self.MONTH_MAP.get(month_name, 1)
            year = int(match.group(2)) if match.group(2) else ref.year
            start = datetime(year, month, 1)
            if month == 12:
                end = datetime(year + 1, 1, 1)
            else:
                end = datetime(year, month + 1, 1)
        elif ptype == "year":
            year = int(match.group(1))
            start = datetime(year, 1, 1)
            end = datetime(year + 1, 1, 1)
        elif ptype == "quarter":
            q = int(match.group(1)[1])
            year = int(match.group(2)) if match.group(2) else ref.year
            start_month = (q - 1) * 3 + 1
            start = datetime(year, start_month, 1)
            end_month = start_month + 3
            if end_month > 12:
                end = datetime(year + 1, 1, 1)
            else:
                end = datetime(year, end_month, 1)
        else:
            start = ref - timedelta(days=30)
            end = ref
        
        return TemporalRef(
            start=start,
            end=end,
            operator=operator,
            raw_text=match.group(0),
            confidence=confidence
        )
    
    def _unit_to_delta(self, unit: str, num: int) -> timedelta:
        """Convert time unit to timedelta."""
        if unit == "day":
            return timedelta(days=num)
        elif unit == "week":
            return timedelta(weeks=num)
        elif unit == "month":
            return timedelta(days=num * 30)  # Approximation
        elif unit == "year":
            return timedelta(days=num * 365)  # Approximation
        else:
            return timedelta(days=num)


# =============================================================================
# Main Retriever
# =============================================================================

class TemporalRetriever:
    """
    Temporal-aware retriever.
    
    Scoring: combined = α·semantic + β·temporal + γ·recency
    Where α, β, γ come from RetrievalConfig (to be learned).
    """
    
    def __init__(
        self,
        world_model,  # WorldModel, not imported to avoid circular
        retrieval_config: RetrievalConfig,
        temporal_config: TemporalConfig,
    ):
        self.world_model = world_model
        self.r_config = retrieval_config
        self.t_config = temporal_config
        self.extractor = TemporalRefExtractor(temporal_config)
    
    def retrieve(
        self,
        query: str,
        query_time: Optional[datetime] = None,
        top_k: int = 10,
        semantic_oversample: int = 3,
    ) -> list[RetrievalResult]:
        """
        Retrieve temporally-aware results.
        
        Args:
            query: Natural language query
            query_time: Reference time (defaults to now)
            top_k: Number of results to return
            semantic_oversample: Fetch more candidates for reranking
        """
        query_time = query_time or datetime.now()
        self.extractor.reference_time = query_time
        
        # 1. Extract temporal reference
        temporal_ref = self.extractor.extract(query)
        
        # 2. Semantic retrieval (oversample for reranking)
        candidates = self._semantic_search(query, top_k * semantic_oversample)
        
        # 3. Score each candidate
        results = []
        for entity, semantic_score in candidates:
            timestamp = self._get_entity_timestamp(entity)
            
            # Compute component scores
            temporal_score = self._compute_temporal_score(timestamp, temporal_ref, query_time)
            recency_score = self._compute_recency_score(timestamp, query_time)
            
            # Weighted combination (weights from config)
            combined = (
                self.r_config.semantic_weight * semantic_score +
                self.r_config.temporal_weight * temporal_score +
                self.r_config.recency_weight * recency_score
            )
            
            # Build result with dual temporal representation
            results.append(RetrievalResult(
                entity_id=entity.get("id", ""),
                entity_name=entity.get("name", "unknown"),
                entity_type=entity.get("type", "unknown"),
                content=self._build_content(entity),
                timestamp=timestamp,
                date_iso=timestamp.strftime("%Y-%m-%d") if timestamp else None,
                relative_time=self._humanize_delta(query_time - timestamp) if timestamp else None,
                period=self._get_period(timestamp) if timestamp else None,
                semantic_score=semantic_score,
                temporal_score=temporal_score,
                recency_score=recency_score,
                combined_score=combined,
            ))
        
        # 4. Sort by combined score
        results.sort(key=lambda r: r.combined_score, reverse=True)
        return results[:top_k]
    
    def _semantic_search(self, query: str, top_k: int) -> list[tuple[dict, float]]:
        """Perform semantic search over world model entities."""
        if not self.world_model.entities:
            return []
        
        # Use world model's search if available
        if hasattr(self.world_model, 'find_related'):
            return self.world_model.find_related(query, limit=top_k)
        
        # Fallback: embedding search
        if hasattr(self.world_model, 'embed'):
            query_emb = self.world_model.embed(query)
            scored = []
            for eid, entity in self.world_model.entities.items():
                if entity.embedding:
                    score = self._cosine_similarity(query_emb, entity.embedding)
                    scored.append((entity.__dict__, score))
            scored.sort(key=lambda x: x[1], reverse=True)
            return scored[:top_k]
        
        return []
    
    def _get_entity_timestamp(self, entity: dict) -> Optional[datetime]:
        """Extract timestamp from entity dict."""
        for field in ['last_seen', 'timestamp', 'first_seen']:
            if field in entity:
                ts = entity[field]
                if isinstance(ts, datetime):
                    return ts
                if isinstance(ts, (int, float)) and ts > 0:
                    return datetime.fromtimestamp(ts)
        
        # Check state.date for events
        state = entity.get('current_state', entity.get('state', {}))
        if isinstance(state, dict) and 'date' in state:
            try:
                return datetime.fromisoformat(state['date'])
            except (ValueError, TypeError):
                pass
        
        return None
    
    def _compute_temporal_score(
        self,
        timestamp: Optional[datetime],
        temporal_ref: Optional[TemporalRef],
        query_time: datetime,
    ) -> float:
        """Compute temporal constraint satisfaction score."""
        if timestamp is None:
            return 0.5  # Neutral for undated entities
        
        if temporal_ref is None:
            # No constraint — use recency-based fallback
            days_ago = (query_time - timestamp).total_seconds() / 86400
            return math.exp(-days_ago / 90)  # 90-day decay (TODO: config?)
        
        # Check if timestamp in range
        in_range = True
        if temporal_ref.start and timestamp < temporal_ref.start:
            in_range = False
        if temporal_ref.end and timestamp > temporal_ref.end:
            in_range = False
        
        if in_range:
            return 1.0 * temporal_ref.confidence
        
        # Out of range — compute distance and decay
        if temporal_ref.start and timestamp < temporal_ref.start:
            dist_days = (temporal_ref.start - timestamp).total_seconds() / 86400
        elif temporal_ref.end and timestamp > temporal_ref.end:
            dist_days = (timestamp - temporal_ref.end).total_seconds() / 86400
        else:
            dist_days = 0
        
        # Decay score based on distance (30-day characteristic scale)
        decay = math.exp(-dist_days / 30)
        return decay * 0.5  # Max 0.5 for out-of-range
    
    def _compute_recency_score(
        self,
        timestamp: Optional[datetime],
        query_time: datetime,
    ) -> float:
        """Compute recency score with exponential decay."""
        if timestamp is None:
            return 0.5
        
        age_seconds = (query_time - timestamp).total_seconds()
        if age_seconds < 0:
            return 0.3  # Future events slightly penalized
        
        # Exponential decay with halflife from config
        halflife = self.r_config.recency_halflife_seconds
        return math.exp(-age_seconds * math.log(2) / halflife)
    
    def _build_content(self, entity: dict) -> str:
        """Build content string from entity."""
        parts = [f"[{entity.get('type', 'entity')}] {entity.get('name', 'unknown')}"]
        
        state = entity.get('current_state', entity.get('state', {}))
        if isinstance(state, dict):
            for k, v in list(state.items())[:5]:  # Limit fields
                if k not in ['embedding']:
                    parts.append(f"{k}: {str(v)[:100]}")
        
        return " | ".join(parts)
    
    def _humanize_delta(self, delta: timedelta) -> str:
        """Convert timedelta to human-readable string."""
        days = abs(delta.days)
        if days == 0:
            return "today"
        elif days == 1:
            return "yesterday" if delta.days > 0 else "tomorrow"
        elif days < 7:
            return f"{days} days ago"
        elif days < 30:
            weeks = days // 7
            return f"{weeks} week{'s' if weeks > 1 else ''} ago"
        elif days < 365:
            months = days // 30
            return f"{months} month{'s' if months > 1 else ''} ago"
        else:
            years = days // 365
            return f"{years} year{'s' if years > 1 else ''} ago"
    
    def _get_period(self, timestamp: datetime) -> str:
        """Get semantic period for timestamp. TODO: implement properly."""
        # Placeholder — should query world model for Period entities
        return timestamp.strftime("%B %Y")
    
    def _cosine_similarity(self, a, b) -> float:
        """Cosine similarity between two vectors."""
        a, b = np.array(a), np.array(b)
        return float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-9))
