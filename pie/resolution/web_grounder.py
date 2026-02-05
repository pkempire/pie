"""
Web Grounding — verify and enrich entities using web search.

Scoped to tools, organizations, and concepts.
Uses search snippets only — no deep scraping.
"""

from __future__ import annotations
import logging
import re
from dataclasses import dataclass

from pie.core.models import ExtractedEntity

logger = logging.getLogger("pie.web_grounder")


@dataclass
class WebGrounding:
    """Result of web grounding for an entity."""
    query: str
    canonical_name: str | None = None
    description: str | None = None
    url: str | None = None
    verified: bool = False
    raw_snippets: list[str] | None = None


class WebGrounder:
    """
    Verify entities against web search results.
    Uses Brave Search API via the search function.
    """
    
    def __init__(self, search_fn=None):
        """
        Args:
            search_fn: Callable that takes a query string and returns search results.
                        If None, web grounding is disabled (returns unverified).
        """
        self.search_fn = search_fn
        self._cache: dict[str, WebGrounding] = {}
        self._stats = {"queries": 0, "verified": 0, "not_found": 0, "cached": 0}
    
    @property
    def stats(self) -> dict:
        return self._stats.copy()
    
    def ground(self, entity: ExtractedEntity) -> WebGrounding:
        """
        Verify an entity against web search.
        Only grounds tools, organizations, and concepts.
        """
        # Skip entity types we don't ground
        groundable_types = {"tool", "organization", "concept"}
        if entity.type not in groundable_types:
            return WebGrounding(query="", verified=False)
        
        # Skip very generic names
        generic_names = {
            "python", "javascript", "html", "css", "sql", "git",
            "machine learning", "ai", "api", "database", "web",
        }
        if entity.name.lower() in generic_names:
            return WebGrounding(
                query=entity.name,
                canonical_name=entity.name,
                verified=True,  # We know these are real
                description=f"Well-known {entity.type}",
            )
        
        # Check cache
        cache_key = f"{entity.name.lower()}:{entity.type}"
        if cache_key in self._cache:
            self._stats["cached"] += 1
            return self._cache[cache_key]
        
        if not self.search_fn:
            return WebGrounding(query=entity.name, verified=False)
        
        # Build search query
        type_hint = {
            "tool": "software tool framework",
            "organization": "company organization",
            "concept": "concept technology",
        }.get(entity.type, "")
        
        query = f"{entity.name} {type_hint}".strip()
        
        try:
            self._stats["queries"] += 1
            results = self.search_fn(query)
            
            if results and len(results) > 0:
                # Extract info from top results
                snippets = []
                canonical_name = entity.name
                description = None
                url = None
                
                for r in results[:3]:
                    title = r.get("title", "")
                    snippet = r.get("description", r.get("snippet", ""))
                    result_url = r.get("url", "")
                    snippets.append(f"{title}: {snippet}")
                    
                    # Try to get canonical name from title
                    if not description and snippet:
                        description = snippet[:200]
                    if not url and result_url:
                        url = result_url
                    
                    # Check if the entity name appears in results (basic verification)
                    if entity.name.lower() in title.lower() or entity.name.lower() in snippet.lower():
                        canonical_name = _extract_canonical_from_title(title, entity.name)
                
                grounding = WebGrounding(
                    query=query,
                    canonical_name=canonical_name,
                    description=description,
                    url=url,
                    verified=True,
                    raw_snippets=snippets,
                )
                self._stats["verified"] += 1
            else:
                grounding = WebGrounding(
                    query=query,
                    verified=False,
                )
                self._stats["not_found"] += 1
            
            self._cache[cache_key] = grounding
            return grounding
            
        except Exception as e:
            logger.warning(f"Web grounding failed for {entity.name}: {e}")
            return WebGrounding(query=query, verified=False)


def _extract_canonical_from_title(title: str, entity_name: str) -> str:
    """
    Try to extract the canonical name from a search result title.
    e.g., "FalkorDB - Redis-based Graph Database" → "FalkorDB"
    """
    # Common separators in titles
    for sep in [" - ", " | ", " — ", " · ", ": "]:
        if sep in title:
            parts = title.split(sep)
            # The canonical name is usually the first part
            candidate = parts[0].strip()
            if entity_name.lower() in candidate.lower() and len(candidate) < 50:
                return candidate
    
    return entity_name
