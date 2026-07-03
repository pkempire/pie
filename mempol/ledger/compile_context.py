"""Compile a cited context pack for a project/thread task."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from mempol import config
from mempol.core.retrieval import format_context, retrieve_budgeted
from mempol.core.schema import TraceEvent
from mempol.core.store import estimate_tokens, now_iso, store_for_run, trace_id
from mempol.ledger.schema import ContextPack
from mempol.ledger.store import ledger_for_run


def compile_context(
    run_name: str,
    task: str,
    project_id: str = "memory_context_systems",
    thread_id: str = "",
    k: int = 12,
    token_budget: int = 5000,
) -> dict[str, Any]:
    core = store_for_run(run_name)
    ledger = ledger_for_run(run_name)
    scoped_query = task
    if thread_id:
        scoped_query = f"{task}\nthread:{thread_id}"
    hits, metrics = retrieve_budgeted(core, scoped_query, k=k, token_budget=token_budget, include_spans=True)

    source_span_ids: list[str] = []
    for hit in hits:
        if hit["kind"] == "span":
            source_span_ids.append(hit["id"])
        else:
            source_span_ids.extend(hit.get("source_span_ids") or [])
    source_span_ids = list(dict.fromkeys(source_span_ids))

    context = format_context(core, hits, provenance_limit=4)
    markdown = "\n".join(
        [
            f"# Context Pack: {task}",
            "",
            f"- Project: `{project_id}`",
            f"- Thread: `{thread_id or 'all'}`",
            f"- Token budget: {token_budget}",
            f"- Retrieved items: {len(hits)}",
            f"- Estimated context tokens: {estimate_tokens(context)}",
            "",
            "## Instructions For The Agent",
            "Use the evidence below. Preserve uncertainty. If recommending actions, distinguish run-ready commands from speculative ideas.",
            "",
            "## Retrieved Evidence",
            "",
            context,
        ]
    )
    pack = ContextPack(
        id=trace_id("context_pack", run_name, project_id, thread_id, task),
        project_id=project_id,
        thread_id=thread_id,
        task=task,
        markdown=markdown,
        source_span_ids=source_span_ids,
        token_budget=token_budget,
        token_estimate=estimate_tokens(markdown),
        created_at=now_iso(),
        metrics=metrics,
        metadata={"retrieved_ids": [h["id"] for h in hits]},
    )
    ledger.upsert_context_pack(pack)
    core.log_trace(
        TraceEvent(
            id=trace_id("ledger_compile_context", run_name),
            run_name=run_name,
            op="ledger_compile_context",
            input={"task": task, "project_id": project_id, "thread_id": thread_id, "k": k, "token_budget": token_budget},
            output={"context_pack_id": pack.id, "retrieved_ids": [h["id"] for h in hits]},
            metrics={**metrics, "context_pack_tokens_est": pack.token_estimate},
            created_at=now_iso(),
        )
    )
    core.commit()
    ledger.commit()
    out_dir = config.RESULTS_DIR / run_name
    out_dir.mkdir(parents=True, exist_ok=True)
    out_md = out_dir / "latest_context_pack.md"
    out_json = out_dir / "latest_context_pack.json"
    out_md.write_text(markdown, encoding="utf-8")
    out_json.write_text(
        json.dumps(
            {
                "context_pack_id": pack.id,
                "run_name": run_name,
                "project_id": project_id,
                "thread_id": thread_id,
                "task": task,
                "metrics": {**metrics, "context_pack_tokens_est": pack.token_estimate},
                "retrieved": hits,
                "source_span_ids": source_span_ids,
                "markdown_path": str(out_md),
            },
            indent=2,
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )
    core.close()
    ledger.close()
    return {"markdown": markdown, "json_path": str(out_json), "markdown_path": str(out_md), "metrics": metrics}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--run-name", default="research_ledger_repo")
    ap.add_argument("--task", required=True)
    ap.add_argument("--project-id", default="memory_context_systems")
    ap.add_argument("--thread-id", default="")
    ap.add_argument("--k", type=int, default=12)
    ap.add_argument("--token-budget", type=int, default=5000)
    ap.add_argument("--out", type=Path, default=None)
    args = ap.parse_args()
    result = compile_context(args.run_name, args.task, args.project_id, args.thread_id, args.k, args.token_budget)
    if args.out:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(result["markdown"], encoding="utf-8")
    print(result["markdown"])
    print(f"\nWrote {result['markdown_path']}")


if __name__ == "__main__":
    main()
