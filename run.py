#!/usr/bin/env python3
"""
PIE Runner — CLI entry point for the ingestion pipeline.

Usage:
    python run.py                          # Full ingestion (all data)
    python run.py --test                   # Test run (5 batches)
    python run.py --test --batches 20      # Test run (20 batches)
    python run.py --year 2025              # Only 2025+ conversations
    python run.py --no-web                 # Skip web grounding
    python run.py --no-context             # Skip sliding window context
    python run.py --model gpt-5-mini     # Use a different model
"""

import argparse
import logging
import sys
import os
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent))

from pie.config import PIEConfig
from pie.ingestion.pipeline import IngestionPipeline


def setup_logging(verbose: bool = True):
    """Configure logging."""
    level = logging.INFO if verbose else logging.WARNING
    
    formatter = logging.Formatter(
        "%(asctime)s [%(name)s] %(message)s",
        datefmt="%H:%M:%S",
    )
    
    handler = logging.StreamHandler()
    handler.setFormatter(formatter)
    
    root_logger = logging.getLogger("pie")
    root_logger.setLevel(level)
    root_logger.addHandler(handler)


def make_web_search_fn():
    """
    Create a web search function.
    Uses Brave Search API if available, otherwise returns None.
    """
    api_key = os.environ.get("BRAVE_API_KEY")
    if not api_key:
        logging.getLogger("pie").warning("BRAVE_API_KEY not set — web grounding disabled")
        return None
    
    import requests
    
    def search(query: str, count: int = 3) -> list[dict]:
        """Search via Brave API."""
        resp = requests.get(
            "https://api.search.brave.com/res/v1/web/search",
            headers={"X-Subscription-Token": api_key},
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


def main():
    parser = argparse.ArgumentParser(description="PIE Ingestion Pipeline")
    parser.add_argument("--test", action="store_true", help="Test mode (5 batches)")
    parser.add_argument("--batches", type=int, default=None, help="Limit number of batches")
    parser.add_argument("--conversations", type=int, default=None, help="Limit conversations parsed")
    parser.add_argument("--year", type=int, default=2025, help="Minimum year (default: 2025)")
    parser.add_argument("--model", type=str, default="gpt-5-mini", help="Extraction model")
    parser.add_argument("--no-web", action="store_true", help="Disable web grounding")
    parser.add_argument("--no-context", action="store_true", help="Disable sliding window context")
    parser.add_argument("--input", type=str, default=None, help="Path to conversations.json")
    parser.add_argument("--output", type=str, default="./output", help="Output directory")
    parser.add_argument("--save-every", type=int, default=5, help="Save checkpoint every N batches")
    parser.add_argument("--skip", type=int, default=0, help="Skip first N batches (for resuming)")
    parser.add_argument("--quiet", action="store_true", help="Less output")
    
    args = parser.parse_args()
    setup_logging(verbose=not args.quiet)
    
    logger = logging.getLogger("pie")
    
    # Build config
    config = PIEConfig(
        output_dir=Path(args.output),
        use_web_grounding=not args.no_web,
        use_sliding_window=not args.no_context,
    )
    config.llm.extraction_model = args.model
    config.ingestion.year_min = args.year
    
    if args.input:
        config.conversations_path = Path(args.input)
    
    # Determine limits
    limit_batches = args.batches
    if args.test and limit_batches is None:
        limit_batches = 5
    
    logger.info("=" * 60)
    logger.info("PIE: Personal Intelligence Engine")
    logger.info("=" * 60)
    logger.info(f"Model: {config.llm.extraction_model}")
    logger.info(f"Input: {config.conversations_path}")
    logger.info(f"Output: {config.output_dir}")
    logger.info(f"Year filter: >= {config.ingestion.year_min}")
    logger.info(f"Sliding window: {config.use_sliding_window}")
    logger.info(f"Web grounding: {config.use_web_grounding}")
    if limit_batches:
        logger.info(f"Batch limit: {limit_batches}")
    logger.info("=" * 60)
    
    # Build web search function
    web_search_fn = None if args.no_web else make_web_search_fn()
    
    # Create and run pipeline
    pipeline = IngestionPipeline(config=config, web_search_fn=web_search_fn)
    
    pipeline.run(
        year_min=args.year,
        limit_batches=limit_batches,
        limit_conversations=args.conversations,
        save_every=args.save_every,
        skip_batches=args.skip,
    )


if __name__ == "__main__":
    main()
