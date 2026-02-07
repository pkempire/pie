"""Bi-temporal utilities for PIE.

Bi-temporal modeling separates two time dimensions:
- SYSTEM TIME (ingested_at): When we learned about something
- EVENT TIME (valid_at/valid_from/valid_to): When it actually happened

This enables queries like:
- "What did I know on Jan 15?" → filter by system time
- "What was happening in December?" → filter by event time
- "When did I first learn about X?" → system time on entity
- "When did X actually start?" → event time on entity
"""

from __future__ import annotations
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Iterator, TypeVar, Callable
import re

T = TypeVar('T')


@dataclass
class TemporalFilter:
    """A bi-temporal filter specification."""
    
    # Event time constraints (when things happened)
    event_after: float | None = None   # valid_at/valid_from >= this
    event_before: float | None = None  # valid_at/valid_to <= this
    
    # System time constraints (when we learned about it)
    ingested_after: float | None = None
    ingested_before: float | None = None
    
    @classmethod
    def from_query(cls, query: str, reference_date: float | None = None) -> 'TemporalFilter':
        """Parse temporal constraints from a natural language query.
        
        Examples:
            "What happened in December 2024?" → event time filter
            "What did I know as of January 15?" → system time filter
            "First thing I did last week" → event time filter
        """
        ref = reference_date or datetime.now(timezone.utc).timestamp()
        ref_dt = datetime.fromtimestamp(ref, tz=timezone.utc)
        
        filter = cls()
        query_lower = query.lower()
        
        # Parse "in [month] [year]" patterns → event time
        month_match = re.search(r'\bin\s+(january|february|march|april|may|june|july|august|september|october|november|december)(?:\s+(\d{4}))?', query_lower)
        if month_match:
            month_name = month_match.group(1)
            year = int(month_match.group(2)) if month_match.group(2) else ref_dt.year
            month_num = ['january', 'february', 'march', 'april', 'may', 'june', 
                        'july', 'august', 'september', 'october', 'november', 'december'].index(month_name) + 1
            
            start = datetime(year, month_num, 1, tzinfo=timezone.utc)
            if month_num == 12:
                end = datetime(year + 1, 1, 1, tzinfo=timezone.utc)
            else:
                end = datetime(year, month_num + 1, 1, tzinfo=timezone.utc)
            
            filter.event_after = start.timestamp()
            filter.event_before = end.timestamp()
        
        # Parse "last week/month" → event time
        if 'last week' in query_lower:
            filter.event_after = ref - (7 * 24 * 3600)
            filter.event_before = ref
        elif 'last month' in query_lower:
            filter.event_after = ref - (30 * 24 * 3600)
            filter.event_before = ref
        elif 'yesterday' in query_lower:
            filter.event_after = ref - (24 * 3600)
            filter.event_before = ref
        
        # Parse "as of [date]" → system time filter (what we knew then)
        as_of_match = re.search(r'as of\s+(\w+\s+\d+|\d{4}-\d{2}-\d{2})', query_lower)
        if as_of_match:
            try:
                date_str = as_of_match.group(1)
                if '-' in date_str:
                    dt = datetime.strptime(date_str, '%Y-%m-%d').replace(tzinfo=timezone.utc)
                else:
                    dt = datetime.strptime(date_str, '%B %d').replace(year=ref_dt.year, tzinfo=timezone.utc)
                filter.ingested_before = dt.timestamp()
            except ValueError:
                pass
        
        # Parse "before/after [date]"
        before_match = re.search(r'before\s+(\w+\s+\d+,?\s*\d*|\d{4}-\d{2}-\d{2})', query_lower)
        after_match = re.search(r'after\s+(\w+\s+\d+,?\s*\d*|\d{4}-\d{2}-\d{2})', query_lower)
        
        if before_match:
            ts = _parse_date_string(before_match.group(1), ref_dt)
            if ts:
                filter.event_before = ts
                
        if after_match:
            ts = _parse_date_string(after_match.group(1), ref_dt)
            if ts:
                filter.event_after = ts
        
        return filter
    
    def matches_entity(self, entity) -> bool:
        """Check if an entity matches this temporal filter."""
        # System time checks
        if self.ingested_after and entity.first_seen < self.ingested_after:
            return False
        if self.ingested_before and entity.first_seen > self.ingested_before:
            return False
        
        # Event time checks
        valid_from = getattr(entity, 'valid_from', None)
        valid_to = getattr(entity, 'valid_to', None)
        
        # If no event time on entity, fall back to system time
        if valid_from is None and valid_to is None:
            # Use first_seen as proxy for event time
            if self.event_after and entity.first_seen < self.event_after:
                return False
            if self.event_before and entity.first_seen > self.event_before:
                return False
        else:
            # Check event time overlap
            if self.event_after and valid_to and valid_to < self.event_after:
                return False
            if self.event_before and valid_from and valid_from > self.event_before:
                return False
        
        return True
    
    def matches_transition(self, transition) -> bool:
        """Check if a state transition matches this temporal filter."""
        ingested = getattr(transition, 'ingested_at', None) or transition.timestamp
        valid = getattr(transition, 'valid_at', None) or ingested
        
        if self.ingested_after and ingested < self.ingested_after:
            return False
        if self.ingested_before and ingested > self.ingested_before:
            return False
        if self.event_after and valid < self.event_after:
            return False
        if self.event_before and valid > self.event_before:
            return False
        
        return True
    
    def filter_items(self, items: Iterator[T], time_getter: Callable[[T], tuple[float, float]]) -> Iterator[T]:
        """Generic filter for items with (ingested_at, valid_at) times."""
        for item in items:
            ingested, valid = time_getter(item)
            
            if self.ingested_after and ingested < self.ingested_after:
                continue
            if self.ingested_before and ingested > self.ingested_before:
                continue
            if self.event_after and valid < self.event_after:
                continue
            if self.event_before and valid > self.event_before:
                continue
            
            yield item


def _parse_date_string(date_str: str, ref_dt: datetime) -> float | None:
    """Parse various date string formats."""
    date_str = date_str.strip().rstrip(',')
    
    # Try YYYY-MM-DD
    try:
        dt = datetime.strptime(date_str, '%Y-%m-%d').replace(tzinfo=timezone.utc)
        return dt.timestamp()
    except ValueError:
        pass
    
    # Try "Month Day" or "Month Day Year"
    for fmt in ['%B %d %Y', '%B %d']:
        try:
            dt = datetime.strptime(date_str, fmt)
            if '%Y' not in fmt:
                dt = dt.replace(year=ref_dt.year)
            dt = dt.replace(tzinfo=timezone.utc)
            return dt.timestamp()
        except ValueError:
            continue
    
    return None


def timestamp_to_date(ts: float) -> str:
    """Convert Unix timestamp to YYYY-MM-DD."""
    return datetime.fromtimestamp(ts, tz=timezone.utc).strftime('%Y-%m-%d')


def date_to_timestamp(date_str: str) -> float:
    """Convert YYYY-MM-DD to Unix timestamp."""
    return datetime.strptime(date_str, '%Y-%m-%d').replace(tzinfo=timezone.utc).timestamp()


def format_temporal_context(
    entities: list,
    reference_date: float | None = None,
    max_entities: int = 20
) -> str:
    """Format entities with temporal context for LLM consumption.
    
    Groups entities by validity period and shows temporal relationships.
    """
    ref = reference_date or datetime.now(timezone.utc).timestamp()
    
    # Separate by temporal status
    ongoing = []
    past = []
    future = []
    unknown = []
    
    for e in entities[:max_entities]:
        valid_from = getattr(e, 'valid_from', None)
        valid_to = getattr(e, 'valid_to', None)
        
        if valid_from is None and valid_to is None:
            unknown.append(e)
        elif valid_to and valid_to < ref:
            past.append(e)
        elif valid_from and valid_from > ref:
            future.append(e)
        else:
            ongoing.append(e)
    
    parts = []
    
    if ongoing:
        parts.append("=== Currently Active ===")
        for e in ongoing:
            parts.append(f"- {e.name} ({e.type.value})")
    
    if past:
        parts.append("\n=== Past ===")
        for e in sorted(past, key=lambda x: getattr(x, 'valid_to', 0) or 0, reverse=True):
            ended = getattr(e, 'valid_to', None)
            end_str = f" [ended {timestamp_to_date(ended)}]" if ended else ""
            parts.append(f"- {e.name} ({e.type.value}){end_str}")
    
    if future:
        parts.append("\n=== Upcoming ===")
        for e in sorted(future, key=lambda x: getattr(x, 'valid_from', 0) or 0):
            starts = getattr(e, 'valid_from', None)
            start_str = f" [starts {timestamp_to_date(starts)}]" if starts else ""
            parts.append(f"- {e.name} ({e.type.value}){start_str}")
    
    if unknown:
        parts.append("\n=== Atemporal ===")
        for e in unknown:
            parts.append(f"- {e.name} ({e.type.value})")
    
    return "\n".join(parts)
