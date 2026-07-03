"""Print the canonical application/evaluation map."""
from __future__ import annotations

import argparse
import json
from dataclasses import asdict

from mempol.applications.registry import application_targets, to_markdown


def main() -> None:
    ap = argparse.ArgumentParser(description="Show application targets and evaluation wedges.")
    ap.add_argument("--in-scope-only", action="store_true")
    ap.add_argument("--format", choices=["markdown", "json"], default="markdown")
    args = ap.parse_args()

    rows = application_targets(in_scope_only=args.in_scope_only)
    if args.format == "json":
        print(json.dumps([asdict(r) for r in rows], indent=2))
    else:
        print(to_markdown(rows))


if __name__ == "__main__":
    main()

