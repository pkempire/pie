"""CLI for the planner agent.

Usage:
    python -m architect.scripts.plan "I want to scrape competitor pricing nightly and Slack-ping me on changes"

    python -m architect.scripts.plan "Build a long-running customer-support agent with persistent memory" \\
        --format cursor --no-discovery
"""
from __future__ import annotations
import argparse
import logging

from ..agent import planner


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("spec", help="What you want to build")
    parser.add_argument("--format", choices=["markdown", "cursor"], default="markdown")
    parser.add_argument("--no-discovery", action="store_true",
                        help="Skip live web search if KG matches are weak.")
    parser.add_argument("--deep-critic", action="store_true",
                        help="Run the expensive verification step (slower, costs more).")
    parser.add_argument("--max-revisions", type=int, default=2)
    parser.add_argument("--email", default="", help="Optional email for analytics.")
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO,
                        format="%(asctime)s %(levelname)s %(message)s")

    plan = planner.plan_for_spec(
        spec=args.spec,
        format=args.format,
        allow_live_search=not args.no_discovery,
        deep_critic=args.deep_critic,
        max_revisions=args.max_revisions,
        user_email=args.email,
    )
    print(plan.rendered)
    if plan.critic_report:
        print(f"\n--- critic: severity={plan.critic_report.severity}, "
              f"revisions={plan.revisions} ---")


if __name__ == "__main__":
    main()
