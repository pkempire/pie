"""Enrich a single component by name (and optional URLs).

Usage:
    python -m architect.scripts.enrich_one "Browserbase"
    python -m architect.scripts.enrich_one "Stagehand" \\
        --homepage https://www.stagehand.dev \\
        --github https://github.com/browserbase/stagehand
"""
from __future__ import annotations
import argparse
import json
import logging

from ..ingestion import enrich


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("name", help="Component display name, e.g. 'Browserbase'")
    parser.add_argument("--homepage", default="", help="Homepage URL (optional; otherwise resolved)")
    parser.add_argument("--github", default="",   help="GitHub URL (optional; otherwise resolved)")
    parser.add_argument("--context", default="",  help="Optional disambiguation hint for resolution")
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO,
                        format="%(asctime)s %(levelname)s %(message)s")

    card = enrich.enrich_component(
        name=args.name,
        homepage_url=args.homepage,
        github_url=args.github,
        context=args.context,
    )
    if not card:
        raise SystemExit("enrichment failed (no card returned)")
    print(json.dumps(
        {k: v for k, v in card.items() if k != "embedding_json"},
        indent=2,
    ))


if __name__ == "__main__":
    main()
