"""
Web Enrichment — automatically research prospects and companies.

Uses Brave Search API to gather:
- Company info, news, press releases
- LinkedIn profile snippets
- Marketing channels and tech stack
- Funding, hiring, product launches
- Industry trends
"""

from __future__ import annotations
import json
import os
import logging
import time
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional

import requests

logger = logging.getLogger("sales.enrichment")


def make_brave_search_fn(api_key: str | None = None) -> Callable | None:
    """Create a Brave Search function."""
    key = api_key or os.environ.get("BRAVE_API_KEY")
    if not key:
        logger.warning("BRAVE_API_KEY not set — web enrichment disabled")
        return None
    
    def search(query: str, count: int = 5) -> list[dict]:
        resp = requests.get(
            "https://api.search.brave.com/res/v1/web/search",
            headers={"X-Subscription-Token": key},
            params={"q": query, "count": count},
            timeout=10,
        )
        resp.raise_for_status()
        data = resp.json()
        results = []
        for r in data.get("web", {}).get("results", []):
            results.append({
                "title": r.get("title", ""),
                "description": r.get("description", ""),
                "url": r.get("url", ""),
            })
        return results
    
    return search


@dataclass
class EnrichmentResult:
    """Results of web enrichment for a prospect/company."""
    company: dict = field(default_factory=dict)
    person: dict = field(default_factory=dict)
    industry: dict = field(default_factory=dict)
    tech_stack: dict = field(default_factory=dict)
    marketing: dict = field(default_factory=dict)
    raw_searches: list[dict] = field(default_factory=list)
    
    def to_dict(self) -> dict:
        return {
            "company": self.company,
            "person": self.person,
            "industry": self.industry,
            "tech_stack": self.tech_stack,
            "marketing": self.marketing,
        }


class WebEnricher:
    """
    Enrich prospect and company intelligence using web search.
    
    Runs multiple targeted searches and synthesizes results.
    """
    
    def __init__(self, search_fn: Callable | None = None, llm=None):
        self.search_fn = search_fn or make_brave_search_fn()
        self.llm = llm
        self._cache: dict[str, list[dict]] = {}
        self.stats = {"queries": 0, "cached": 0, "errors": 0}
    
    def _search(self, query: str, count: int = 5) -> list[dict]:
        """Execute a search with caching."""
        if query in self._cache:
            self.stats["cached"] += 1
            return self._cache[query]
        
        if not self.search_fn:
            return []
        
        try:
            self.stats["queries"] += 1
            results = self.search_fn(query, count=count)
            self._cache[query] = results
            time.sleep(1.0)  # rate limit courtesy (Brave free tier is strict)
            return results
        except Exception as e:
            logger.warning(f"Search failed for '{query}': {e}")
            self.stats["errors"] += 1
            return []
    
    def enrich_prospect(
        self,
        name: str,
        title: str,
        company: str,
    ) -> EnrichmentResult:
        """
        Run full enrichment for a prospect.
        
        Searches for company info, person info, industry, tech stack, marketing.
        """
        result = EnrichmentResult()
        
        logger.info(f"Enriching: {name} ({title}) at {company}")
        
        # 1. Company overview
        company_results = self._search(f"{company} company about")
        result.company["name"] = company
        result.company["description"] = self._extract_best_snippet(company_results, company)
        result.company["url"] = self._extract_best_url(company_results, company)
        result.raw_searches.append({"query": f"{company} company about", "results": company_results})
        
        # 2. Company recent news
        news_results = self._search(f"{company} news announcements 2025 2026")
        result.company["recent_news"] = [
            r["description"] for r in news_results[:3] if r.get("description")
        ]
        result.raw_searches.append({"query": f"{company} news", "results": news_results})
        
        # 3. Company funding/hiring
        funding_results = self._search(f"{company} funding hiring growth")
        result.company["funding_hiring"] = [
            r["description"] for r in funding_results[:3] if r.get("description")
        ]
        
        # 4. Person LinkedIn
        person_results = self._search(f"{name} {title} {company} LinkedIn")
        result.person["name"] = name
        result.person["title"] = title
        result.person["linkedin_snippets"] = [
            r["description"] for r in person_results[:3] if r.get("description")
        ]
        result.person["linkedin_url"] = self._extract_linkedin_url(person_results)
        
        # 5. Tech stack / data platform
        tech_results = self._search(f"{company} data platform technology stack engineering")
        result.tech_stack["signals"] = [
            r["description"] for r in tech_results[:3] if r.get("description")
        ]
        
        # 6. Job postings (tech stack signals)
        jobs_results = self._search(f"{company} data engineer job posting")
        result.tech_stack["job_posting_signals"] = [
            r["description"] for r in jobs_results[:3] if r.get("description")
        ]
        
        # 7. Industry trends
        industry = self._infer_industry(company, company_results)
        result.industry["name"] = industry
        trend_results = self._search(f"{industry} data platform trends 2025 2026")
        result.industry["trends"] = [
            r["description"] for r in trend_results[:3] if r.get("description")
        ]
        
        # 8. Marketing channels
        marketing_results = self._search(f"{company} blog content marketing")
        result.marketing["channels"] = []
        for r in marketing_results[:3]:
            url = r.get("url", "")
            if "blog" in url.lower():
                result.marketing["channels"].append("blog")
            if "linkedin" in url.lower():
                result.marketing["channels"].append("linkedin")
            if "twitter" in url.lower() or "x.com" in url.lower():
                result.marketing["channels"].append("twitter")
        result.marketing["content_signals"] = [
            r["description"] for r in marketing_results[:2] if r.get("description")
        ]
        
        # Use LLM to synthesize if available
        if self.llm and self.search_fn:
            result = self._llm_synthesize(result, name, title, company)
        
        logger.info(
            f"Enrichment complete for {name}: "
            f"{self.stats['queries']} queries, {self.stats['cached']} cached"
        )
        
        return result
    
    def _llm_synthesize(
        self,
        result: EnrichmentResult,
        name: str,
        title: str,
        company: str,
    ) -> EnrichmentResult:
        """Use LLM to synthesize raw search results into clean intelligence."""
        raw_data = json.dumps(result.to_dict(), indent=2, default=str)[:3000]
        
        messages = [
            {
                "role": "system",
                "content": (
                    "You are a sales intelligence analyst. Given raw web search results about a prospect "
                    "and their company, synthesize into clean, actionable intelligence.\n\n"
                    "Return JSON with:\n"
                    "- company: {description, industry, size_estimate, recent_developments: [str], website}\n"
                    "- person: {background, experience_signals: [str]}\n"
                    "- tech_stack: {known_technologies: [str], infrastructure_signals: [str]}\n"
                    "- industry: {name, key_trends: [str], challenges: [str]}\n"
                    "- marketing: {channels: [str], content_themes: [str]}\n"
                    "- talking_points: [str] — 3-5 things the sales rep should mention based on this research\n\n"
                    "Only include information you can verify from the search results. "
                    "Mark anything uncertain with '(unverified)'."
                ),
            },
            {
                "role": "user",
                "content": f"Prospect: {name}, {title} at {company}\n\nRaw search data:\n{raw_data}",
            },
        ]
        
        try:
            llm_result = self.llm.chat(
                messages=messages,
                model="gpt-4o-mini",
                json_mode=True,
                max_tokens=16000,
            )
            
            synthesized = llm_result["content"]
            
            # Merge synthesized data into result
            if "company" in synthesized:
                result.company.update(synthesized["company"])
            if "person" in synthesized:
                result.person.update(synthesized["person"])
            if "tech_stack" in synthesized:
                result.tech_stack.update(synthesized["tech_stack"])
            if "industry" in synthesized:
                result.industry.update(synthesized["industry"])
            if "marketing" in synthesized:
                result.marketing.update(synthesized["marketing"])
            if "talking_points" in synthesized:
                result.company["talking_points"] = synthesized["talking_points"]
            
        except Exception as e:
            logger.warning(f"LLM synthesis failed: {e}")
        
        return result
    
    def _extract_best_snippet(self, results: list[dict], entity_name: str) -> str:
        """Get the most relevant snippet from search results."""
        for r in results:
            desc = r.get("description", "")
            if entity_name.lower() in desc.lower() and len(desc) > 50:
                return desc[:300]
        if results:
            return results[0].get("description", "")[:300]
        return ""
    
    def _extract_best_url(self, results: list[dict], entity_name: str) -> str:
        """Get the most relevant URL (prefer the company's own domain)."""
        name_parts = entity_name.lower().replace(" ", "").replace("-", "")
        for r in results:
            url = r.get("url", "")
            if name_parts[:6] in url.lower().replace(" ", "").replace("-", ""):
                return url
        if results:
            return results[0].get("url", "")
        return ""
    
    def _extract_linkedin_url(self, results: list[dict]) -> str:
        """Extract LinkedIn URL from results."""
        for r in results:
            url = r.get("url", "")
            if "linkedin.com" in url:
                return url
        return ""
    
    def _infer_industry(self, company: str, results: list[dict]) -> str:
        """Infer industry from company name and search results."""
        company_lower = company.lower()
        
        # Common industry keywords
        industry_map = {
            "energy": "Energy & Utilities",
            "grid": "Energy & Utilities",
            "solar": "Energy & Utilities",
            "biotech": "Biotechnology & Pharma",
            "pharma": "Biotechnology & Pharma",
            "bio": "Biotechnology & Pharma",
            "health": "Healthcare",
            "medical": "Healthcare",
            "bank": "Financial Services",
            "fintech": "Financial Services",
            "payment": "Financial Services",
            "insurance": "Insurance",
            "telecom": "Telecommunications",
            "media": "Media & Entertainment",
            "travel": "Travel & Hospitality",
            "retail": "Retail & E-commerce",
            "manufacturing": "Manufacturing",
            "logistics": "Logistics & Supply Chain",
            "shipping": "Logistics & Supply Chain",
            "aerospace": "Aerospace & Defense",
            "education": "Education Technology",
            "government": "Government",
            "food": "Food & Beverage",
            "fashion": "Fashion & Apparel",
            "real estate": "Real Estate",
        }
        
        for keyword, industry in industry_map.items():
            if keyword in company_lower:
                return industry
        
        # Check search results for industry clues
        all_text = " ".join(r.get("description", "") for r in results).lower()
        for keyword, industry in industry_map.items():
            if keyword in all_text:
                return industry
        
        return "Technology"
