from __future__ import annotations

import subprocess
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from mempol.ledger.compile_context import compile_context
from mempol.ledger.day_report import build_day_report, format_markdown
from mempol.ledger.ingest_repo import ingest_repo
from mempol.ledger.store import ledger_for_run
from mempol.temporal.bootstrap_repo import bootstrap_repo_temporal
from mempol.temporal.context import compile_temporal_context


def test_research_ledger_ingests_repo_day_and_context(tmp_path, monkeypatch):
    repo = tmp_path / "repo"
    repo.mkdir()
    (repo / "mempol").mkdir()
    (repo / "mempol" / "notes.md").write_text("# Memory\n\nGEPA consolidator improved a memory prompt.\n")
    (repo / "research").mkdir()
    (repo / "research" / "idea.md").write_text("# Idea\n\nContext packs should cite evidence.\n")

    subprocess.run(["git", "init"], cwd=repo, check=True, stdout=subprocess.DEVNULL)
    subprocess.run(["git", "config", "user.email", "test@example.com"], cwd=repo, check=True)
    subprocess.run(["git", "config", "user.name", "Test User"], cwd=repo, check=True)
    subprocess.run(["git", "add", "."], cwd=repo, check=True)
    subprocess.run(
        ["git", "commit", "-m", "Add memory research notes"],
        cwd=repo,
        check=True,
        stdout=subprocess.DEVNULL,
    )

    import mempol.config as config

    monkeypatch.setattr(config, "RESULTS_DIR", tmp_path / "results")
    run_name = "ledger_test"
    summary = ingest_repo(repo, run_name=run_name, max_commits=10)
    assert summary["files_ingested"] == 2
    assert summary["git_commits_ingested"] == 1
    assert summary["ledger_store"]["projects"] == 1
    assert summary["ledger_store"]["threads"] >= 1

    ledger = ledger_for_run(run_name)
    days = ledger.available_days()
    ledger.close()
    assert days

    report = build_day_report(run_name, days[0]["day"])
    md = format_markdown(report)
    assert "Research Ledger Day Report" in md
    assert "notes.md" in md or "idea.md" in md

    context = compile_context(run_name, "What improved the memory prompt?", k=4, token_budget=2000)
    assert "GEPA consolidator" in context["markdown"]

    temporal_summary = bootstrap_repo_temporal(run_name)
    assert temporal_summary["states_bootstrapped"] > 0

    temporal_context = compile_temporal_context(
        run_name,
        scope_id="project:mempol_memory_policy",
        task="What improved the memory prompt?",
        k=4,
        token_budget=2500,
    )
    assert temporal_context["selected_states"]
    assert "GEPA consolidator" in temporal_context["markdown"]


def test_temporal_context_empty_run_has_actionable_error(tmp_path, monkeypatch):
    import mempol.config as config

    monkeypatch.setattr(config, "RESULTS_DIR", tmp_path / "results")
    result = compile_temporal_context(
        "empty_run",
        scope_id="project:memory",
        task="What should happen next?",
    )
    assert result["error"] == "empty_run"
    assert "mempol.ledger.ingest_repo" in result["markdown"]
    assert "mempol.temporal.bootstrap_repo" in result["markdown"]
