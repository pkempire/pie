"""Two-stage critic for proposed plans.

A "plan" here is the list of components the planner has selected for a
user's spec, plus the role each one plays. The critic answers: would
this actually work?

Two stages because the trade-off matters:

  STAGE 1 — Cheap, prompt-only sanity check.
    Run on every plan. ~$0.001, ~1 sec.
    Catches structural problems an LLM can spot from the plan alone:
      • missing slots (no auth, no storage, no error handling)
      • language / runtime mismatches (Python tool + Node-only build)
      • redundant components (two memory layers)
      • known anti-patterns (raw Playwright + Browserbase together is
        almost always wrong)
      • obvious version skew

  STAGE 2 — Expensive, tool-using verification.
    Run when stage 1 passes but stakes are high (paid tier, complex
    plan). ~$0.05, ~30 sec.
    Calls real tools to verify claims:
      • DeepWiki on each component to confirm capability claims
      • Resolves package names to current PyPI / npm versions
      • Cross-checks homepage `last_verified_at` to flag stale entries
      • Looks for archived / deprecated repos via GitHub API

When the critic finds issues, it returns a `CritiqueReport` with both
the issues and concrete revision suggestions the planner can act on.

The planner uses this in a loop: propose → critique → revise → re-critique
until clean or max_revisions hit.

API:

    report = critique_cheap(plan, user_spec)
    if report.severity in ("major", "blocking"):
        plan = planner.revise(plan, report)
        report = critique_cheap(plan, user_spec)
    if user_tier == "paid":
        deep = critique_deep(plan, user_spec)
        ...
"""
from __future__ import annotations
import json
import logging
from dataclasses import dataclass, field
from typing import Literal

from .. import db
from ..ingestion import github_client
from mempol import llm, config

logger = logging.getLogger(__name__)


Severity = Literal["clean", "minor", "major", "blocking"]


# ─── Data model ─────────────────────────────────────────────────────────────
@dataclass
class PlanComponent:
    slug: str
    name: str
    role: str                                          # "browser_runtime", "memory_layer", ...


@dataclass
class Plan:
    user_spec: str
    components: list[PlanComponent]
    architecture_pattern: str = ""                     # optional pattern this matches


@dataclass
class CritiqueIssue:
    severity: Severity
    category: str                                      # "missing_slot" | "redundant" | "anti_pattern" | ...
    message: str                                       # human-readable
    suggestion: str = ""                               # concrete fix (slug to add/remove/swap)


@dataclass
class CritiqueReport:
    severity: Severity                                 # worst issue's severity
    issues: list[CritiqueIssue] = field(default_factory=list)
    summary: str = ""

    def is_blocking(self) -> bool:
        return self.severity == "blocking"


# ─── Stage 1: cheap, prompt-only ────────────────────────────────────────────
_CHEAP_SYSTEM = """You are a senior staff engineer reviewing a proposed
software architecture for a developer's stated goal. List concrete
issues with the plan if there are any.

For each issue, classify severity:
  - clean      :  no issue (don't return clean issues; only return real ones)
  - minor      :  cosmetic / nit (e.g. version preference)
  - major      :  the plan will be harder than necessary or fragile
  - blocking   :  the plan will not work as stated

For each issue, propose a concrete fix: which component to add, remove,
or swap. Be specific.

Common categories:
  - missing_slot       : no component fills a required role (auth, storage, error handling)
  - redundant          : two components fill the same role unnecessarily
  - anti_pattern       : the combination is known to fail or is overkill
  - language_mismatch  : components in incompatible runtimes / ecosystems
  - version_skew       : implied version mismatch between dependencies
  - unjustified        : a component is included but not used by the plan

Return JSON: {"issues": [...], "summary": "..."}. Empty list if clean."""


def _format_plan_for_critic(plan: Plan, with_one_liners: bool = True) -> str:
    """Render the plan in a compact form for the critic prompt."""
    lines = [f"USER SPEC:\n{plan.user_spec.strip()}", "", "PLAN:"]
    if plan.architecture_pattern:
        lines.append(f"  pattern: {plan.architecture_pattern}")
    with db.connect() as conn:
        for c in plan.components:
            row = db.get_component(conn, c.slug)
            ol = ""
            if with_one_liners and row:
                ol = f" — {row['one_liner']}"
            lines.append(f"  {c.role:24s} {c.name}{ol}")
    return "\n".join(lines)


def critique_cheap(plan: Plan) -> CritiqueReport:
    """Stage-1 critic. Prompt-only structural sanity check."""
    user = _format_plan_for_critic(plan)
    msgs = [
        {"role": "system", "content": _CHEAP_SYSTEM},
        {"role": "user",   "content": user},
    ]
    raw = llm.chat(msgs, model=config.OBSERVER_MODEL or "gpt-4o-mini",
                    json_mode=True)
    try:
        obj = json.loads(raw)
    except Exception as e:
        logger.warning("cheap-critic parse fail: %s; raw=%r", e, raw[:200])
        return CritiqueReport(severity="clean", summary="critic parse error")

    issues = []
    worst: Severity = "clean"
    severity_order = {"clean": 0, "minor": 1, "major": 2, "blocking": 3}
    for it in (obj.get("issues") or []):
        sev: Severity = it.get("severity", "minor")
        if sev not in severity_order:
            sev = "minor"
        if severity_order[sev] > severity_order[worst]:
            worst = sev
        issues.append(CritiqueIssue(
            severity=sev,
            category=it.get("category", "unspecified"),
            message=it.get("message", "")[:500],
            suggestion=it.get("suggestion", "")[:500],
        ))
    return CritiqueReport(
        severity=worst,
        issues=issues,
        summary=obj.get("summary", "")[:500],
    )


# ─── Stage 2: expensive, tool-using verification ────────────────────────────
def _check_component_freshness(slug: str, max_age_days: int = 60) -> CritiqueIssue | None:
    """Flag components whose last_verified_at is stale or whose underlying
    repo is archived. We only flag, not block — a stale card might still
    point at a working tool."""
    from datetime import datetime, timedelta
    with db.connect() as conn:
        row = db.get_component(conn, slug)
    if not row:
        return CritiqueIssue(
            severity="major", category="unknown_component",
            message=f"component {slug!r} is not in the index",
            suggestion="re-run discovery for this slot",
        )
    lv = row["last_verified_at"]
    if lv:
        try:
            age = datetime.utcnow() - datetime.fromisoformat(lv)
            if age > timedelta(days=max_age_days):
                return CritiqueIssue(
                    severity="minor", category="stale_card",
                    message=f"{slug} card last verified {age.days} days ago",
                    suggestion=f"re-enrich {slug} before publishing the plan",
                )
        except Exception:
            pass
    # GitHub archived check
    if row["github_url"]:
        try:
            meta = github_client.get_repo_meta(row["github_url"])
            if meta.get("archived"):
                return CritiqueIssue(
                    severity="major", category="repo_archived",
                    message=f"{slug}'s GitHub repo is archived",
                    suggestion=f"swap {slug} for an alternative_to",
                )
        except Exception:
            pass
    return None


_DEEP_SYSTEM = """You are verifying a proposed software architecture
against authoritative sources for each component. For each component,
state whether the role assigned to it is actually supported by the
component, citing the source. If a component's claimed role is not
supported, flag it as an issue.

You will receive: the user spec, the list of components with assigned
roles, and a short fetched snippet from each component's docs/README.

Return JSON: {"issues": [...], "verified_roles": [{slug, role, ok, evidence}]}."""


def critique_deep(plan: Plan) -> CritiqueReport:
    """Stage-2 critic. Verifies each component's claimed role against
    the actual README/docs we have on file. Slower but high-precision."""
    issues: list[CritiqueIssue] = []
    severity_order = {"clean": 0, "minor": 1, "major": 2, "blocking": 3}
    worst: Severity = "clean"

    # 1. Freshness / archived checks (one DB+API call per component)
    for c in plan.components:
        issue = _check_component_freshness(c.slug)
        if issue:
            issues.append(issue)
            if severity_order[issue.severity] > severity_order[worst]:
                worst = issue.severity

    # 2. Role-vs-capability cross-check via LLM grounded on stored summary
    snippets = []
    with db.connect() as conn:
        for c in plan.components:
            row = db.get_component(conn, c.slug)
            if not row:
                continue
            extras = json.loads(row["extras_json"] or "{}")
            cap = (row["capability_long"] or row["summary"] or "")[:1200]
            ex = "; ".join(
                e.get("description", "")[:80]
                for e in (extras.get("canonical_examples") or [])[:2]
            )
            snippets.append(
                f"  [{c.slug} as {c.role!r}]\n"
                f"    summary:  {(row['summary'] or '')[:200]}\n"
                f"    capability: {cap[:600]}\n"
                f"    examples: {ex or '(none)'}"
            )

    user = (
        f"USER SPEC:\n{plan.user_spec}\n\n"
        f"COMPONENTS WITH SNIPPETS:\n" + "\n\n".join(snippets)
    )
    msgs = [
        {"role": "system", "content": _DEEP_SYSTEM},
        {"role": "user",   "content": user},
    ]
    raw = llm.chat(msgs, model=config.OBSERVER_MODEL or "gpt-4o-mini",
                    json_mode=True)
    try:
        obj = json.loads(raw)
    except Exception as e:
        logger.warning("deep-critic parse fail: %s", e)
        return CritiqueReport(severity=worst, issues=issues,
                                summary="deep critic parse error; freshness checks only")

    for it in (obj.get("issues") or []):
        sev: Severity = it.get("severity", "minor")
        if sev not in severity_order:
            sev = "minor"
        if severity_order[sev] > severity_order[worst]:
            worst = sev
        issues.append(CritiqueIssue(
            severity=sev,
            category=it.get("category", "role_mismatch"),
            message=it.get("message", "")[:500],
            suggestion=it.get("suggestion", "")[:500],
        ))
    return CritiqueReport(
        severity=worst, issues=issues,
        summary=f"deep critique: {len(issues)} issue(s) — "
                 f"worst severity {worst}",
    )
