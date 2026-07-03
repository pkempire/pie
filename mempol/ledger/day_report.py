"""Export a day-level report from the Research Ledger."""
from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path
from typing import Any

from mempol import config
from mempol.core.store import SQLiteMemoryStore, estimate_tokens
from mempol.ledger.store import LedgerStore, ledger_for_run


def _target_preview(core: SQLiteMemoryStore, target_type: str, target_id: str) -> dict[str, Any]:
    if target_type == "artifact":
        artifact = core.get_artifact(target_id)
        if not artifact:
            return {"title": target_id, "text": ""}
        return {
            "title": artifact.title,
            "source": artifact.source,
            "kind": artifact.kind,
            "uri": artifact.uri,
            "text": artifact.content[:800],
            "metadata": artifact.metadata,
        }
    if target_type == "span":
        span = core.get_span(target_id)
        if not span:
            return {"title": target_id, "text": ""}
        artifact = core.get_artifact(span.artifact_id)
        return {
            "title": artifact.title if artifact else span.artifact_id,
            "source": artifact.source if artifact else "",
            "kind": "span",
            "uri": artifact.uri if artifact else "",
            "text": span.text[:800],
            "metadata": {**span.metadata, "locator": span.locator},
        }
    if target_type == "memory_state":
        state = core.get_memory_state(target_id)
        if not state:
            return {"title": target_id, "text": ""}
        return {
            "title": state.metadata.get("rel_path") or state.id,
            "source": state.metadata.get("adapter", "memory_state"),
            "kind": "memory_state",
            "uri": "",
            "text": state.content[:800],
            "metadata": state.metadata,
        }
    return {"title": target_id, "text": ""}


def available_days(run_name: str) -> list[dict[str, Any]]:
    ledger = ledger_for_run(run_name)
    rows = ledger.available_days()
    ledger.close()
    return rows


def build_day_report(run_name: str, day: str, limit: int = 200) -> dict[str, Any]:
    core_path = config.RESULTS_DIR / run_name / "core_memory.sqlite"
    ledger_path = config.RESULTS_DIR / run_name / "ledger.sqlite"
    core = SQLiteMemoryStore(core_path)
    ledger = LedgerStore(ledger_path)
    rows = ledger.conn.execute(
        """
        SELECT * FROM memberships
        WHERE substr(valid_from, 1, 10)=?
        ORDER BY thread_id, target_type, created_at
        LIMIT ?
        """,
        (day, limit),
    ).fetchall()
    groups: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for r in rows:
        row = dict(r)
        preview = _target_preview(core, row["target_type"], row["target_id"])
        groups[row["thread_id"]].append({**row, "preview": preview})
    report = {
        "run_name": run_name,
        "day": day,
        "n_memberships": len(rows),
        "threads": dict(groups),
        "core_stats": core.stats(),
        "ledger_stats": ledger.stats(),
    }
    core.close()
    ledger.close()
    return report


def format_markdown(report: dict[str, Any]) -> str:
    lines = [
        f"# Research Ledger Day Report: {report['day']}",
        "",
        f"- Run: `{report['run_name']}`",
        f"- Membership rows shown: {report['n_memberships']}",
        f"- Core artifacts: {report['core_stats']['artifacts']}",
        f"- Ledger memberships: {report['ledger_stats']['memberships']}",
        "",
    ]
    for thread_id, rows in sorted(report["threads"].items(), key=lambda kv: (-len(kv[1]), kv[0])):
        lines.extend([f"## {thread_id}", ""])
        by_type: dict[str, list[dict[str, Any]]] = defaultdict(list)
        for row in rows:
            by_type[row["target_type"]].append(row)
        for target_type, typed_rows in sorted(by_type.items()):
            lines.extend([f"### {target_type} ({len(typed_rows)})", ""])
            for row in typed_rows[:25]:
                preview = row["preview"]
                meta = preview.get("metadata") or {}
                title = preview.get("title") or row["target_id"]
                source = preview.get("source") or ""
                kind = preview.get("kind") or ""
                locator = meta.get("locator") or meta.get("rel_path") or meta.get("commit") or ""
                tokens = estimate_tokens(preview.get("text", ""))
                lines.append(f"- `{kind}` **{title}**")
                lines.append(f"  - source: `{source}` · locator: `{locator}` · tokens~{tokens}")
                if row.get("rationale"):
                    lines.append(f"  - assignment: {row['rationale']} ({row['confidence']:.2f})")
                text = " ".join((preview.get("text") or "").split())
                if text:
                    lines.append(f"  - preview: {text[:240]}")
            lines.append("")
    return "\n".join(lines).rstrip() + "\n"


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--run-name", default="research_ledger_repo")
    ap.add_argument("--day", default="", help="YYYY-MM-DD. If omitted, prints available days.")
    ap.add_argument("--limit", type=int, default=200)
    ap.add_argument("--out", type=Path, default=None)
    args = ap.parse_args()

    if not args.day:
        print(json.dumps(available_days(args.run_name)[:100], indent=2, ensure_ascii=False))
        return

    report = build_day_report(args.run_name, args.day, limit=args.limit)
    md = format_markdown(report)
    out = args.out or (config.RESULTS_DIR / args.run_name / f"day_{args.day}.md")
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(md, encoding="utf-8")
    (out.with_suffix(".json")).write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
    print(md)
    print(f"\nWrote {out}")


if __name__ == "__main__":
    main()
