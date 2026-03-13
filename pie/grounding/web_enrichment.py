"""
Web Enrichment — Live grounding for entities.

Uses web search to:
1. Verify entity identity (is "React" the library or something else?)
2. Enrich with current info (latest version, recent news)
3. Resolve ambiguity (which "Python" — language, snake, Monty?)

Designed to work with both PIE and the sales product.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Callable, Optional

logger = logging.getLogger("pie.grounding")


@dataclass
class WebEnrichment:
    """Result of web enrichment for an entity."""
    canonical_name: str         # Official/canonical name
    description: str            # One-line description
    entity_type: str            # Verified type
    url: Optional[str]          # Primary URL (website, LinkedIn, etc.)
    
    # Additional context
    recent_info: Optional[str]  # Recent news/updates
    verified: bool              # Was this verified against web?
    confidence: float           # How confident are we?
    
    # Raw search results for debugging
    search_query: str
    result_count: int


class WebEnricher:
    """
    Enrich entities with web information.
    
    Uses a search function (Brave, Google, etc.) to find info,
    then LLM to extract structured data.
    """
    
    def __init__(
        self,
        search_fn: Callable[[str], list[dict]],  # query -> [{title, url, snippet}]
        llm_fn: Callable[[str], str],            # prompt -> response
    ):
        self.search = search_fn
        self.llm = llm_fn
    
    def enrich(
        self,
        name: str,
        entity_type: str,
        context: Optional[str] = None,
    ) -> WebEnrichment:
        """
        Enrich an entity with web information.
        
        Args:
            name: Entity name to search
            entity_type: Expected type (tool, organization, person, etc.)
            context: Additional context to disambiguate (e.g., "JavaScript library")
        """
        # Build search query
        query = self._build_query(name, entity_type, context)
        
        # Search
        try:
            results = self.search(query)
        except Exception as e:
            logger.warning(f"Web search failed for {name}: {e}")
            return self._fallback(name, entity_type, query, str(e))
        
        if not results:
            return self._fallback(name, entity_type, query, "No results")
        
        # Extract structured info via LLM
        return self._extract_from_results(name, entity_type, query, results)
    
    def _build_query(self, name: str, entity_type: str, context: Optional[str]) -> str:
        """Build search query for entity."""
        if entity_type == "tool":
            return f"{name} software tool"
        elif entity_type == "organization":
            return f"{name} company organization"
        elif entity_type == "person":
            if context:
                return f"{name} {context}"
            return f"{name} person"
        elif entity_type == "concept":
            return f"{name} definition meaning"
        else:
            if context:
                return f"{name} {context}"
            return name
    
    def _extract_from_results(
        self,
        name: str,
        entity_type: str,
        query: str,
        results: list[dict],
    ) -> WebEnrichment:
        """Use LLM to extract structured info from search results."""
        
        # Format results for LLM
        results_text = "\n".join([
            f"- [{r.get('title', 'No title')}]({r.get('url', '')}): {r.get('snippet', '')}"
            for r in results[:5]
        ])
        
        prompt = f"""Extract information about "{name}" from these search results.

Search results:
{results_text}

Expected type: {entity_type}

Respond with JSON:
{{
    "canonical_name": "official/full name",
    "description": "one sentence description",
    "verified_type": "tool|organization|person|concept|other",
    "primary_url": "main website or null",
    "recent_info": "any recent news/updates or null",
    "confidence": 0.0-1.0
}}"""
        
        try:
            response = self.llm(prompt)
            import json
            data = json.loads(response)
            
            return WebEnrichment(
                canonical_name=data.get("canonical_name", name),
                description=data.get("description", ""),
                entity_type=data.get("verified_type", entity_type),
                url=data.get("primary_url"),
                recent_info=data.get("recent_info"),
                verified=True,
                confidence=float(data.get("confidence", 0.8)),
                search_query=query,
                result_count=len(results),
            )
        except Exception as e:
            logger.warning(f"LLM extraction failed: {e}")
            return self._fallback(name, entity_type, query, str(e))
    
    def _fallback(self, name: str, entity_type: str, query: str, error: str) -> WebEnrichment:
        """Return unverified enrichment when web lookup fails."""
        return WebEnrichment(
            canonical_name=name,
            description=f"[unverified] {error}",
            entity_type=entity_type,
            url=None,
            recent_info=None,
            verified=False,
            confidence=0.3,
            search_query=query,
            result_count=0,
        )


# =============================================================================
# Sales-specific enrichment
# =============================================================================

@dataclass
class ProspectEnrichment:
    """Enriched prospect/company data for sales."""
    company_name: str
    industry: str
    size: str  # startup, SMB, enterprise
    hq_location: str
    
    # People
    key_contacts: list[dict]  # [{name, title, linkedin}]
    
    # Tech stack (for tech sales)
    tech_stack: list[str]
    
    # Recent signals
    recent_news: list[str]
    funding_info: Optional[str]
    hiring_signals: list[str]
    
    # Confidence
    verified: bool
    confidence: float


class SalesEnricher(WebEnricher):
    """
    Specialized enricher for sales prospects.
    
    Extracts company info, key contacts, tech stack, recent signals.
    """
    
    def enrich_prospect(
        self,
        company_name: str,
        domain: Optional[str] = None,
    ) -> ProspectEnrichment:
        """Enrich a sales prospect with web data."""
        
        # Search for company info
        query = f"{company_name} company"
        if domain:
            query += f" {domain}"
        
        try:
            results = self.search(query)
        except Exception as e:
            logger.warning(f"Prospect search failed: {e}")
            return self._empty_prospect(company_name)
        
        if not results:
            return self._empty_prospect(company_name)
        
        # Also search for recent news
        news_query = f"{company_name} news funding hiring"
        try:
            news_results = self.search(news_query)
        except:
            news_results = []
        
        # Extract via LLM
        return self._extract_prospect(company_name, results, news_results)
    
    def _extract_prospect(
        self,
        company_name: str,
        company_results: list[dict],
        news_results: list[dict],
    ) -> ProspectEnrichment:
        """Extract prospect info via LLM."""
        
        company_text = "\n".join([
            f"- {r.get('title', '')}: {r.get('snippet', '')}"
            for r in company_results[:5]
        ])
        
        news_text = "\n".join([
            f"- {r.get('title', '')}: {r.get('snippet', '')}"
            for r in news_results[:5]
        ]) if news_results else "No recent news found"
        
        prompt = f"""Extract sales-relevant information about {company_name}.

Company search results:
{company_text}

News/funding search results:
{news_text}

Respond with JSON:
{{
    "industry": "their industry",
    "size": "startup|smb|enterprise",
    "hq_location": "city, country",
    "key_contacts": [{{"name": "...", "title": "...", "linkedin": null}}],
    "tech_stack": ["tool1", "tool2"],
    "recent_news": ["headline1", "headline2"],
    "funding_info": "latest funding round or null",
    "hiring_signals": ["role1", "role2"],
    "confidence": 0.0-1.0
}}"""
        
        try:
            response = self.llm(prompt)
            import json
            data = json.loads(response)
            
            return ProspectEnrichment(
                company_name=company_name,
                industry=data.get("industry", "unknown"),
                size=data.get("size", "unknown"),
                hq_location=data.get("hq_location", "unknown"),
                key_contacts=data.get("key_contacts", []),
                tech_stack=data.get("tech_stack", []),
                recent_news=data.get("recent_news", []),
                funding_info=data.get("funding_info"),
                hiring_signals=data.get("hiring_signals", []),
                verified=True,
                confidence=float(data.get("confidence", 0.7)),
            )
        except Exception as e:
            logger.warning(f"Prospect extraction failed: {e}")
            return self._empty_prospect(company_name)
    
    def _empty_prospect(self, company_name: str) -> ProspectEnrichment:
        """Return empty prospect when enrichment fails."""
        return ProspectEnrichment(
            company_name=company_name,
            industry="unknown",
            size="unknown",
            hq_location="unknown",
            key_contacts=[],
            tech_stack=[],
            recent_news=[],
            funding_info=None,
            hiring_signals=[],
            verified=False,
            confidence=0.0,
        )
