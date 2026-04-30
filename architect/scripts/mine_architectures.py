"""Discover real-world systems that use a known component, ingest their
architecture cards, and reinforce the co-occurrence graph.

Usage:
    python -m architect.scripts.mine_architectures Stagehand --max_repos 30
    python -m architect.scripts.mine_architectures Browserbase --max_repos 50
"""
from __future__ import annotations
import argparse
import logging

from ..architecture_miner import mine_for_component


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("component", help="Display name of an existing component")
    parser.add_argument("--max_repos", type=int, default=30)
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO,
                        format="%(asctime)s %(levelname)s %(message)s")

    n = mine_for_component(args.component, max_repos=args.max_repos)
    print(f"OK — ingested {n} architectures for {args.component}")


if __name__ == "__main__":
    main()
