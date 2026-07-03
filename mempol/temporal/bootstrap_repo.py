"""Bootstrap temporal state from an existing research-ledger repo ingest."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from mempol.core.store import now_iso, stable_id, store_for_run
from mempol.ledger.store import LedgerStore
from mempol import config

from .schema import ActiveProcess, StateTransition, TemporalState
from .store import temporal_store_for_run


def _rows(conn, sql: str, params: tuple = ()) -> list[dict[str, Any]]:
    return [dict(r) for r in conn.execute(sql, params).fetchall()]


def _load_json(raw: str | None, default: Any) -> Any:
    if not raw:
        return default
    try:
        return json.loads(raw)
    except Exception:
        return default


def _scope_for_thread(thread_id: str) -> str:
    return f"project:{thread_id or 'memory'}"


def bootstrap_repo_temporal(run_name: str, scope_prefix: str = "project") -> dict[str, Any]:
    core = store_for_run(run_name)
    ledger_path = config.RESULTS_DIR / run_name / "ledger.sqlite"
    if not ledger_path.exists():
        core.close()
        raise FileNotFoundError(
            f"No ledger.sqlite found for run `{run_name}`. Run: "
            f"python3 -m mempol.ledger.ingest_repo --root . --run-name {run_name}"
        )
    ledger = LedgerStore(ledger_path)
    temporal = temporal_store_for_run(run_name)

    n_states = 0
    n_transitions = 0
    n_processes = 0

    memberships = _rows(
        ledger.conn,
        """
        SELECT * FROM memberships
        WHERE target_type IN ('memory_state', 'artifact')
        ORDER BY valid_from
        """,
    )
    for row in memberships:
        meta = _load_json(row.get("metadata_json"), {})
        thread_id = row.get("thread_id") or "memory"
        scope_id = _scope_for_thread(thread_id)
        rel_path = meta.get("rel_path") or row["target_id"]
        kind = meta.get("kind") or row["target_type"]
        key = f"repo.{kind}.{rel_path}"

        content = ""
        source_span_ids: list[str] = []
        if row["target_type"] == "memory_state":
            mem = core.get_memory_state(row["target_id"])
            if not mem:
                continue
            content = mem.content
            source_span_ids = mem.source_span_ids
        elif row["target_type"] == "artifact":
            artifact = core.get_artifact(row["target_id"])
            if not artifact:
                continue
            content = f"{artifact.kind}: {artifact.title}\nsource={artifact.source}\ncreated_at={artifact.created_at}"
        observed_at = row.get("valid_from") or now_iso()
        state_id = stable_id("temporal_repo_state", run_name, scope_id, key, row["target_id"])
        state = TemporalState(
            id=state_id,
            scope_id=scope_id,
            key=key,
            content=content,
            state_type="repo_artifact",
            valid_from=observed_at,
            observed_at=observed_at,
            source_span_ids=source_span_ids[:12],
            metadata={
                "bootstrap": "repo_ledger",
                "target_type": row["target_type"],
                "target_id": row["target_id"],
                "thread_id": thread_id,
                "rel_path": rel_path,
                "kind": kind,
            },
        )
        temporal.apply_transition(
            StateTransition(
                id=stable_id("temporal_repo_transition", state_id),
                scope_id=scope_id,
                transition_type="create",
                new_state_id=state_id,
                reason=f"Bootstrapped from repo ledger membership for {rel_path}",
                observed_at=observed_at,
                source_span_ids=source_span_ids[:12],
                metadata={"bootstrap": "repo_ledger", "membership_id": row["id"]},
            ),
            new_state=state,
        )
        n_states += 1
        n_transitions += 1

    runs = _rows(ledger.conn, "SELECT * FROM runs ORDER BY started_at")
    for run in runs:
        thread_id = run.get("thread_id") or "memory"
        scope_id = _scope_for_thread(thread_id)
        started_at = run.get("started_at") or now_iso()
        process_id = stable_id("temporal_repo_process", run_name, run["id"])
        temporal.upsert_process(
            ActiveProcess(
                id=process_id,
                scope_id=scope_id,
                kind="repo_run",
                description=f"{run.get('title') or 'repo run'} ({run.get('status') or 'unknown'})",
                status="done" if run.get("status") in {"committed", "completed"} else "active",
                started_at=started_at,
                expected_at=run.get("ended_at") or "",
                last_checked_at=run.get("ended_at") or started_at,
                metadata={
                    "bootstrap": "repo_ledger",
                    "run_id": run["id"],
                    "command": run.get("command") or "",
                    "thread_id": thread_id,
                    "metrics": _load_json(run.get("metrics_json"), {}),
                },
            )
        )
        n_processes += 1

    temporal.commit()
    summary = {
        "run_name": run_name,
        "states_bootstrapped": n_states,
        "transitions_bootstrapped": n_transitions,
        "processes_bootstrapped": n_processes,
        "temporal_store": temporal.stats(),
        "core_store": core.stats(),
    }
    out_dir = config.RESULTS_DIR / run_name
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "temporal_bootstrap_summary.json").write_text(json.dumps(summary, indent=2, ensure_ascii=False))
    core.close()
    ledger.close()
    temporal.close()
    return summary


def main() -> None:
    ap = argparse.ArgumentParser(description="Bootstrap temporal states from repo ledger artifacts.")
    ap.add_argument("--run-name", required=True)
    args = ap.parse_args()
    print(json.dumps(bootstrap_repo_temporal(args.run_name), indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
