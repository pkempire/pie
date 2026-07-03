"""Ingest existing repo data into the universal memory core.

Examples:
  python -m mempol.scripts.core_ingest --source pie_output --input output/world_model.json --run-name universal_smoke
  python -m mempol.scripts.core_ingest --source architect --run-name universal_smoke
  python -m mempol.scripts.core_ingest --source locomo --run-name universal_smoke --limit 50
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from mempol import config
from mempol.core.adapters import architect_seed_items, locomo_items, pie_world_model_items
from mempol.core.schema import TraceEvent
from mempol.core.store import now_iso, store_for_run, trace_id


def _default_input(source: str) -> Path | None:
    if source == "pie_output":
        return config.ROOT / "output" / "world_model.json"
    if source == "architect":
        return config.ROOT / "architect" / "db" / "seed_components.json"
    return None


def ingest(source: str, run_name: str, input_path: Path | None = None, limit: int = 0, n_convs: int = 1) -> dict:
    store = store_for_run(run_name)
    input_path = input_path or _default_input(source)

    if source == "pie_output":
        if not input_path or not input_path.exists():
            raise FileNotFoundError(f"PIE input not found: {input_path}")
        iterator = pie_world_model_items(input_path, limit=limit)
    elif source == "architect":
        if not input_path or not input_path.exists():
            raise FileNotFoundError(f"architect input not found: {input_path}")
        iterator = architect_seed_items(input_path, limit=limit)
    elif source == "locomo":
        iterator = locomo_items(n_convs=n_convs, limit=limit)
    else:
        raise ValueError(f"unknown source: {source}")

    n_artifacts = n_spans = n_states = 0
    for artifact, spans, state in iterator:
        store.upsert_artifact(artifact)
        n_artifacts += 1
        for span in spans:
            store.upsert_span(span)
            n_spans += 1
        store.upsert_memory_state(state)
        n_states += 1

    summary = {
        "run_name": run_name,
        "source": source,
        "input": str(input_path) if input_path else "",
        "artifacts_ingested": n_artifacts,
        "spans_ingested": n_spans,
        "memory_states_ingested": n_states,
        "store": store.stats(),
    }
    store.log_trace(
        TraceEvent(
            id=trace_id("ingest", source, run_name),
            run_name=run_name,
            op="ingest",
            input={"source": source, "input": str(input_path) if input_path else "", "limit": limit},
            output=summary,
            metrics={
                "artifacts": n_artifacts,
                "spans": n_spans,
                "memory_states": n_states,
            },
            created_at=now_iso(),
        )
    )
    store.commit()
    summary["store"] = store.stats()

    out_dir = config.RESULTS_DIR / run_name
    out_dir.mkdir(parents=True, exist_ok=True)
    summary_path = out_dir / "core_ingest_summary.json"
    existing = []
    if summary_path.exists():
        try:
            existing = json.loads(summary_path.read_text())
            if not isinstance(existing, list):
                existing = [existing]
        except Exception:
            existing = []
    existing.append(summary)
    summary_path.write_text(json.dumps(existing, indent=2, ensure_ascii=False))
    store.close()
    return summary


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--source", choices=["pie_output", "architect", "locomo"], required=True)
    ap.add_argument("--input", type=Path, default=None)
    ap.add_argument("--run-name", default="universal_smoke")
    ap.add_argument("--limit", type=int, default=0, help="0 = no limit")
    ap.add_argument("--n-convs", type=int, default=1, help="LoCoMo only")
    args = ap.parse_args()
    summary = ingest(args.source, args.run_name, args.input, args.limit, args.n_convs)
    print(json.dumps(summary, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
