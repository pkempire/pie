"""Ingest local repo artifacts and Git history into the Research Ledger.

This creates an inspectable day-by-day record of what changed in a coding or
research project.  It does not summarize or "understand" the repo yet; it
preserves evidence and project/thread membership so later policies can learn on
top of real traces.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import mimetypes
import subprocess
from collections.abc import Iterable
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from mempol import config
from mempol.core.schema import Artifact, MemoryState, Span, TraceEvent
from mempol.core.store import estimate_tokens, now_iso, stable_id, store_for_run, trace_id
from mempol.ledger.schema import Membership, Project, ResearchObject, RunRecord
from mempol.ledger.store import ledger_for_run, membership_id
from mempol.ledger.tagger import DEFAULT_PROJECT, assign_thread, seed_threads


TEXT_EXTS = {
    ".py", ".md", ".txt", ".json", ".jsonl", ".csv", ".ts", ".tsx", ".js", ".jsx",
    ".html", ".css", ".toml", ".yaml", ".yml", ".sql", ".tex", ".bib", ".sh",
    ".rb", ".rs", ".go", ".java", ".c", ".cpp", ".h", ".hpp",
}

SKIP_PARTS = {
    ".git", "__pycache__", ".pytest_cache", "node_modules", ".next", ".venv", "venv",
    ".cache", "frames", "frames_ironsite", "sample_frames", "videos", "annotated_segments",
}

SKIP_EXTS = {".pyc", ".png", ".jpg", ".jpeg", ".mp4", ".mov", ".pdf", ".sqlite", ".db", ".npy", ".DS_Store"}


def sha256_text(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8", errors="ignore")).hexdigest()


def _run_git(root: Path, args: list[str]) -> str:
    try:
        return subprocess.check_output(["git", *args], cwd=root, text=True, stderr=subprocess.DEVNULL)
    except Exception:
        return ""


def _git_last_touch(root: Path, rel_path: str) -> dict[str, str]:
    raw = _run_git(root, ["log", "-1", "--format=%H%x1f%aI%x1f%s", "--", rel_path]).strip()
    if not raw:
        return {}
    parts = raw.split("\x1f")
    if len(parts) < 3:
        return {}
    return {"git_commit": parts[0], "git_author_date": parts[1], "git_subject": parts[2]}


def _git_commits(root: Path, max_commits: int) -> Iterable[dict[str, Any]]:
    raw = _run_git(
        root,
        [
            "log",
            f"-{max_commits}",
            "--date=iso-strict",
            "--pretty=format:@@COMMIT@@%n%H%n%aI%n%an%n%s%n%b%n@@FILES@@",
            "--name-only",
        ],
    )
    if not raw:
        return []
    chunks = [c for c in raw.split("@@COMMIT@@") if c.strip()]
    commits = []
    for chunk in chunks:
        header, _, files_blob = chunk.partition("@@FILES@@")
        lines = [ln.rstrip() for ln in header.strip().splitlines()]
        if len(lines) < 4:
            continue
        commit_hash, authored_at, author, subject = lines[:4]
        body = "\n".join(lines[4:]).strip()
        files = [ln.strip() for ln in files_blob.splitlines() if ln.strip()]
        commits.append({
            "hash": commit_hash,
            "authored_at": authored_at,
            "author": author,
            "subject": subject,
            "body": body,
            "files": files,
        })
    return commits


def _is_text_candidate(path: Path, root: Path, max_file_bytes: int) -> bool:
    try:
        rel_parts = path.relative_to(root).parts
    except ValueError:
        return False
    if any(part in SKIP_PARTS for part in rel_parts):
        return False
    if path.suffix in SKIP_EXTS:
        return False
    if path.suffix not in TEXT_EXTS:
        return False
    try:
        return path.stat().st_size <= max_file_bytes
    except OSError:
        return False


def _read_text(path: Path) -> str:
    return path.read_text(encoding="utf-8", errors="replace")


def _mtime_iso(path: Path) -> str:
    try:
        return datetime.fromtimestamp(path.stat().st_mtime, tz=timezone.utc).isoformat().replace("+00:00", "Z")
    except OSError:
        return now_iso()


def _json_timestamp(text: str) -> str:
    try:
        obj = json.loads(text)
    except Exception:
        return ""
    if not isinstance(obj, dict):
        return ""
    for key in ("timestamp", "created_at", "run_at", "date", "started_at"):
        val = obj.get(key)
        if isinstance(val, str) and len(val) >= 10:
            return val
    return ""


def _line_spans(text: str, max_chars: int = 3500) -> list[tuple[int, int, str]]:
    lines = text.splitlines()
    spans = []
    start_line = 1
    buf: list[str] = []
    size = 0
    for idx, line in enumerate(lines, 1):
        add = len(line) + 1
        if buf and size + add > max_chars:
            spans.append((start_line, idx - 1, "\n".join(buf)))
            start_line = idx
            buf = []
            size = 0
        buf.append(line)
        size += add
    if buf:
        spans.append((start_line, len(lines), "\n".join(buf)))
    return spans or [(1, 1, text)]


def _artifact_kind(path: Path) -> str:
    if path.name in {"summary.json", "core_ingest_summary.json"}:
        return "result_summary"
    if path.suffix == ".jsonl" or "rows.jsonl" in path.name:
        return "result_rows"
    if path.suffix == ".md":
        return "markdown_doc"
    if path.suffix in {".py", ".ts", ".tsx", ".js", ".jsx"}:
        return "source_file"
    if path.suffix in {".tex", ".bib"}:
        return "paper_source"
    return "text_file"


def _result_research_object(rel_path: str, text: str, project_id: str, thread_id: str, span_ids: list[str]) -> ResearchObject | None:
    if not rel_path.endswith(".json"):
        return None
    try:
        obj = json.loads(text)
    except Exception:
        return None
    if not isinstance(obj, dict):
        return None
    interesting = {k: obj.get(k) for k in ("overall_acc", "acc", "delta", "baseline_mean", "gepa_mean", "n_rows", "variant") if k in obj}
    overall = obj.get("overall")
    if isinstance(overall, dict):
        for k in ("accuracy", "total", "total_score"):
            if k in overall:
                interesting[f"overall_{k}"] = overall[k]
    by_cell = obj.get("by_cell")
    if isinstance(by_cell, dict):
        interesting["cells"] = {k: v.get("overall_acc") or v.get("acc") for k, v in by_cell.items() if isinstance(v, dict)}
    if not interesting:
        return None
    content = f"Result summary from {rel_path}: {json.dumps(interesting, ensure_ascii=False, sort_keys=True)}"
    return ResearchObject(
        id=stable_id("research_result", rel_path, content),
        project_id=project_id,
        thread_id=thread_id,
        role="run_result",
        content=content,
        source_span_ids=span_ids[:3],
        status="observed",
        metadata={"path": rel_path, "parsed_metrics": interesting},
    )


def ingest_repo(
    root: Path,
    run_name: str,
    max_files: int = 0,
    max_file_bytes: int = 300_000,
    max_commits: int = 250,
) -> dict[str, Any]:
    root = Path(root).resolve()
    core = store_for_run(run_name)
    ledger = ledger_for_run(run_name)

    project = Project(
        id=DEFAULT_PROJECT.id,
        title=DEFAULT_PROJECT.title,
        objective=DEFAULT_PROJECT.objective,
        metadata={"root": str(root), "ingest_kind": "local_repo"},
    )
    ledger.upsert_project(project)
    for thread in seed_threads(project.id):
        ledger.upsert_thread(thread)

    n_files = n_spans = n_states = n_memberships = n_objects = n_commits = 0
    day_counts: dict[str, int] = {}

    for path in sorted(root.rglob("*")):
        if max_files and n_files >= max_files:
            break
        if not path.is_file() or not _is_text_candidate(path, root, max_file_bytes):
            continue
        rel_path = path.relative_to(root).as_posix()
        try:
            text = _read_text(path)
        except Exception:
            continue
        if not text.strip():
            continue
        checksum = sha256_text(text)
        git_meta = _git_last_touch(root, rel_path)
        observed_at = git_meta.get("git_author_date") or _json_timestamp(text) or _mtime_iso(path)
        day_counts[observed_at[:10]] = day_counts.get(observed_at[:10], 0) + 1
        thread_id, rationale, confidence = assign_thread(rel_path)
        artifact_id = stable_id("repo_artifact", str(root), rel_path, checksum)
        mime_type = mimetypes.guess_type(rel_path)[0] or "text/plain"
        artifact = Artifact(
            id=artifact_id,
            source="local_repo",
            kind=_artifact_kind(path),
            title=rel_path,
            content=text,
            uri=str(path),
            created_at=observed_at,
            metadata={
                "root": str(root),
                "rel_path": rel_path,
                "checksum_sha256": checksum,
                "size_bytes": path.stat().st_size,
                "mime_type": mime_type,
                **git_meta,
            },
        )
        core.upsert_artifact(artifact)
        n_files += 1

        span_ids: list[str] = []
        for start, end, chunk in _line_spans(text):
            span_id = stable_id("repo_span", artifact_id, start, end, sha256_text(chunk))
            span = Span(
                id=span_id,
                artifact_id=artifact_id,
                text=chunk,
                locator=f"{rel_path}:{start}-{end}",
                start=start,
                end=end,
                metadata={
                    "rel_path": rel_path,
                    "line_start": start,
                    "line_end": end,
                    "observed_at": observed_at,
                    "thread_id": thread_id,
                },
            )
            core.upsert_span(span)
            span_ids.append(span_id)
            n_spans += 1
            ledger.upsert_membership(
                Membership(
                    id=membership_id("span", span_id, project.id, thread_id),
                    target_type="span",
                    target_id=span_id,
                    project_id=project.id,
                    thread_id=thread_id,
                    confidence=confidence,
                    assigned_by="rule",
                    rationale=rationale,
                    valid_from=observed_at,
                    metadata={"rel_path": rel_path, "locator": span.locator},
                )
            )
            n_memberships += 1

        state_content = (
            f"Repo file {rel_path}\n"
            f"Kind: {artifact.kind}\n"
            f"Last touched: {observed_at}\n"
            f"Thread: {thread_id}\n"
            f"Git subject: {git_meta.get('git_subject', '')}\n"
            f"Token estimate: {estimate_tokens(text)}"
        )
        state_id = stable_id("repo_state", artifact_id)
        core.upsert_memory_state(
            MemoryState(
                id=state_id,
                content=state_content,
                source_span_ids=span_ids[:12],
                created_at=observed_at,
                updated_at=observed_at,
                metadata={
                    "adapter": "repo_ledger",
                    "rel_path": rel_path,
                    "thread_id": thread_id,
                    "view_tags": [artifact.kind, thread_id],
                },
            )
        )
        n_states += 1
        ledger.upsert_membership(
            Membership(
                id=membership_id("artifact", artifact_id, project.id, thread_id),
                target_type="artifact",
                target_id=artifact_id,
                project_id=project.id,
                thread_id=thread_id,
                confidence=confidence,
                assigned_by="rule",
                rationale=rationale,
                valid_from=observed_at,
                metadata={"rel_path": rel_path, "kind": artifact.kind},
            )
        )
        ledger.upsert_membership(
            Membership(
                id=membership_id("memory_state", state_id, project.id, thread_id),
                target_type="memory_state",
                target_id=state_id,
                project_id=project.id,
                thread_id=thread_id,
                confidence=confidence,
                assigned_by="rule",
                rationale=rationale,
                valid_from=observed_at,
                metadata={"rel_path": rel_path, "kind": artifact.kind},
            )
        )
        n_memberships += 2

        result_obj = _result_research_object(rel_path, text, project.id, thread_id, span_ids)
        if result_obj:
            ledger.upsert_research_object(result_obj)
            n_objects += 1

    for commit in _git_commits(root, max_commits=max_commits):
        n_commits += 1
        text = "\n".join(
            [
                f"Commit: {commit['hash']}",
                f"Date: {commit['authored_at']}",
                f"Author: {commit['author']}",
                f"Subject: {commit['subject']}",
                "",
                commit.get("body") or "",
                "",
                "Changed files:",
                "\n".join(commit.get("files") or []),
            ]
        )
        checksum = sha256_text(text)
        artifact_id = stable_id("git_commit", commit["hash"])
        thread_id, rationale, confidence = assign_thread(" ".join(commit.get("files") or []) or commit["subject"])
        artifact = Artifact(
            id=artifact_id,
            source="git",
            kind="commit",
            title=f"{commit['hash'][:10]} {commit['subject']}",
            content=text,
            uri=f"git:{commit['hash']}",
            created_at=commit["authored_at"],
            metadata={**commit, "checksum_sha256": checksum},
        )
        core.upsert_artifact(artifact)
        span_id = stable_id("git_commit_span", commit["hash"])
        core.upsert_span(
            Span(
                id=span_id,
                artifact_id=artifact_id,
                text=text,
                locator=commit["hash"],
                metadata={"commit": commit["hash"], "observed_at": commit["authored_at"], "thread_id": thread_id},
            )
        )
        ledger.upsert_membership(
            Membership(
                id=membership_id("artifact", artifact_id, project.id, thread_id),
                target_type="artifact",
                target_id=artifact_id,
                project_id=project.id,
                thread_id=thread_id,
                confidence=confidence,
                assigned_by="rule",
                rationale=f"git commit files: {rationale}",
                valid_from=commit["authored_at"],
                metadata={"commit": commit["hash"], "files": commit.get("files") or []},
            )
        )
        ledger.upsert_membership(
            Membership(
                id=membership_id("span", span_id, project.id, thread_id),
                target_type="span",
                target_id=span_id,
                project_id=project.id,
                thread_id=thread_id,
                confidence=confidence,
                assigned_by="rule",
                rationale=f"git commit files: {rationale}",
                valid_from=commit["authored_at"],
                metadata={"commit": commit["hash"], "files": commit.get("files") or []},
            )
        )
        n_memberships += 2
        ledger.upsert_run(
            RunRecord(
                id=stable_id("git_run", commit["hash"]),
                project_id=project.id,
                thread_id=thread_id,
                title=commit["subject"],
                started_at=commit["authored_at"],
                ended_at=commit["authored_at"],
                actor=commit["author"],
                command=f"git commit {commit['hash']}",
                status="committed",
                artifact_ids=[artifact_id],
                metadata={"commit": commit["hash"], "files": commit.get("files") or []},
            )
        )

    summary = {
        "run_name": run_name,
        "root": str(root),
        "files_ingested": n_files,
        "git_commits_ingested": n_commits,
        "spans_ingested": n_spans + n_commits,
        "memory_states_ingested": n_states,
        "memberships": n_memberships,
        "research_objects": n_objects,
        "day_counts": dict(sorted(day_counts.items())),
        "core_store": core.stats(),
        "ledger_store": ledger.stats(),
    }
    core.log_trace(
        TraceEvent(
            id=trace_id("ledger_ingest_repo", run_name),
            run_name=run_name,
            op="ledger_ingest_repo",
            input={"root": str(root), "max_files": max_files, "max_commits": max_commits},
            output=summary,
            metrics={
                "files": n_files,
                "commits": n_commits,
                "spans": n_spans + n_commits,
                "memberships": n_memberships,
            },
            created_at=now_iso(),
        )
    )
    core.commit()
    ledger.commit()

    out_dir = config.RESULTS_DIR / run_name
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "ledger_ingest_summary.json").write_text(json.dumps(summary, indent=2, ensure_ascii=False))
    core.close()
    ledger.close()
    return summary


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", type=Path, default=config.ROOT)
    ap.add_argument("--run-name", default="research_ledger_repo")
    ap.add_argument("--max-files", type=int, default=0, help="0 = no limit")
    ap.add_argument("--max-file-bytes", type=int, default=300_000)
    ap.add_argument("--max-commits", type=int, default=250)
    args = ap.parse_args()
    summary = ingest_repo(args.root, args.run_name, args.max_files, args.max_file_bytes, args.max_commits)
    print(json.dumps(summary, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
