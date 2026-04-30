"""Planner agent — turns a user spec into a concrete implementation plan.

The planner is one LLM with a tight tool surface. It does NOT see the
full component index in its prompt; it only sees the components the
discovery loop returns for each capability requirement. This is the
MCP-Zero principle in action: small, focused tool surface per step.

Loop
====

  1. Decompose: LLM splits the user spec into capability requirements.
     Each requirement carries a short capability description, a runtime
     hint (python/typescript/hosted/...), and a required vs optional
     flag. Output: list[CapabilityRequirement].

  2. Discover: for each requirement, build a ToolWish and call
     discovery.discover(). Returns top-3 candidates per slot.

  3. Compose: LLM picks one component per slot from the discovered
     candidates and assembles a Plan (component slugs + roles +
     architectural pattern label).

  4. Critique: critic.critique_cheap(plan). If blocking, send the
     issues back into Compose with the critic's suggestions and try
     again (max 2 revisions).

  5. (Optional, paid tier) critic.critique_deep(plan).

  6. Render: emit the final plan as markdown / n8n template / cursor
     spec / .txt. Also persist the (query, plan) pair into user_queries
     for analytics + future RL signal.

Run via:
    plan = plan_for_spec(
        "I want to scrape competitor pricing nightly and Slack-ping me
         on changes",
        format="markdown",
    )
    print(plan.rendered)
"""
from __future__ import annotations
import json
import logging
from dataclasses import dataclass, field
from datetime import datetime
from typing import Literal

from .. import db
from . import discovery, critic
from mempol import llm, config

logger = logging.getLogger(__name__)


# ─── Data model ────────────────────────────────────────────────────────────
@dataclass
class CapabilityRequirement:
    role: str                                   # "browser_runtime", "scheduler", ...
    capability: str                             # what the slot has to do
    required: bool = True
    runtime_hint: str = ""                      # "python" | "typescript" | "hosted"
    must_avoid: list[str] = field(default_factory=list)


@dataclass
class FinalPlan:
    user_spec: str
    components: list[critic.PlanComponent]
    pattern: str = ""
    requirements: list[CapabilityRequirement] = field(default_factory=list)
    critic_report: critic.CritiqueReport | None = None
    revisions: int = 0
    rendered: str = ""
    discovery_candidates: dict[str, list[discovery.ComponentMatch]] = field(default_factory=dict)


# ─── 1. Decompose ──────────────────────────────────────────────────────────
_DECOMPOSE_SYSTEM = """You decompose a developer's natural-language goal
into the minimum set of CAPABILITY REQUIREMENTS needed to build it.

Each requirement is one slot the system needs filled. Pick roles that
recur across software architectures (e.g. browser_runtime, scheduler,
storage, llm_provider, vector_store, notification_channel, auth,
queue, frontend_framework, observability). Don't invent boutique roles
unless the spec demands it.

Be parsimonious: 4-7 requirements is the right shape for most specs.
Mark optional slots accordingly.

Return JSON:
{
  "requirements": [
    {"role": "...", "capability": "1-sentence what-it-does",
     "required": true|false,
     "runtime_hint": "python|typescript|hosted|any",
     "must_avoid": []}
  ],
  "pattern_guess": "5-8 word architectural pattern label, or '' if unclear"
}"""


def decompose(spec: str) -> tuple[list[CapabilityRequirement], str]:
    """Return (requirements, pattern_guess)."""
    msgs = [
        {"role": "system", "content": _DECOMPOSE_SYSTEM},
        {"role": "user",   "content": "USER SPEC:\n" + spec.strip()},
    ]
    raw = llm.chat(msgs, model=config.OBSERVER_MODEL or "gpt-4o-mini",
                    json_mode=True)
    try:
        obj = json.loads(raw)
    except Exception as e:
        logger.warning("decompose parse fail: %s; raw=%r", e, raw[:300])
        return [], ""
    reqs = []
    for r in (obj.get("requirements") or []):
        reqs.append(CapabilityRequirement(
            role=r.get("role", "unspecified")[:60],
            capability=r.get("capability", "")[:300],
            required=bool(r.get("required", True)),
            runtime_hint=r.get("runtime_hint", "")[:30],
            must_avoid=list(r.get("must_avoid") or [])[:5],
        ))
    return reqs, obj.get("pattern_guess", "")[:80]


# ─── 2. Discover ───────────────────────────────────────────────────────────
def discover_for_requirements(
    reqs: list[CapabilityRequirement],
    top_k: int = 3,
    allow_live_search: bool = True,
) -> dict[str, list[discovery.ComponentMatch]]:
    """One discovery call per requirement. Returns {role: [matches]}."""
    out: dict[str, list[discovery.ComponentMatch]] = {}
    for r in reqs:
        nice = []
        if r.runtime_hint and r.runtime_hint != "any":
            nice.append(f"runtime: {r.runtime_hint}")
        wish = discovery.ToolWish(
            capability=r.capability,
            context=f"role={r.role!r}; required={r.required}",
            nice_to_have=nice,
            must_avoid=r.must_avoid,
        )
        try:
            out[r.role] = discovery.discover(
                wish, top_k=top_k, allow_live_search=allow_live_search,
            )
        except Exception as e:
            logger.warning("discovery failed for role=%s: %s", r.role, e)
            out[r.role] = []
    return out


# ─── 3. Compose ────────────────────────────────────────────────────────────
_COMPOSE_SYSTEM = """You assemble a concrete implementation plan from a
user spec, a list of capability requirements, and the candidate
components for each requirement (3 per slot).

For each REQUIRED slot, pick exactly one component. For OPTIONAL slots,
include only if it adds value to this specific spec. Prefer components
whose runtime / deployment matches the user's apparent constraints.

You MAY decline a slot by emitting role with component_slug = null and
note explaining why (e.g. "not needed for this spec"). Don't pad.

Return JSON:
{
  "pattern": "concrete architectural pattern label",
  "components": [
    {"role": "...", "component_slug": "...", "rationale": "≤ 1 sentence"}
  ],
  "rendering_hints": {
    "format_preference": "markdown|n8n|cursor",
    "implementation_order": ["role1", "role2", ...]
  }
}"""


def _format_candidates_block(
    discoveries: dict[str, list[discovery.ComponentMatch]],
) -> str:
    lines = []
    for role, matches in discoveries.items():
        lines.append(f"  ROLE: {role}")
        for m in matches:
            tag = "🆕" if m.fresh else m.confidence
            lines.append(f"    - {m.slug:24s} [{tag} {m.score:.2f}]"
                         f"  {m.name} — {m.rationale[:120]}")
        if not matches:
            lines.append("    (no candidates discovered)")
    return "\n".join(lines)


def compose(
    spec: str,
    reqs: list[CapabilityRequirement],
    pattern_guess: str,
    discoveries: dict[str, list[discovery.ComponentMatch]],
    revision_feedback: str = "",
) -> tuple[list[critic.PlanComponent], str]:
    """LLM-assemble a plan from the discovered candidates."""
    user_parts = [
        f"USER SPEC:\n{spec.strip()}",
        f"PATTERN GUESS: {pattern_guess or '(none)'}",
        "",
        "CAPABILITY REQUIREMENTS:",
    ]
    for r in reqs:
        flag = "REQUIRED" if r.required else "optional"
        user_parts.append(f"  - {r.role} [{flag}]: {r.capability}"
                          + (f"  (runtime hint: {r.runtime_hint})" if r.runtime_hint else ""))
    user_parts.append("")
    user_parts.append("CANDIDATES PER ROLE:")
    user_parts.append(_format_candidates_block(discoveries))
    if revision_feedback:
        user_parts.append("\nREVISION FEEDBACK from critic:")
        user_parts.append(revision_feedback)

    msgs = [
        {"role": "system", "content": _COMPOSE_SYSTEM},
        {"role": "user",   "content": "\n".join(user_parts)},
    ]
    raw = llm.chat(msgs, model=config.OBSERVER_MODEL or "gpt-4o-mini",
                    json_mode=True)
    try:
        obj = json.loads(raw)
    except Exception as e:
        logger.warning("compose parse fail: %s", e)
        return [], pattern_guess

    pattern = obj.get("pattern", pattern_guess)[:80]
    components: list[critic.PlanComponent] = []
    name_lookup: dict[str, dict] = {}
    with db.connect() as conn:
        for c in (obj.get("components") or []):
            slug = c.get("component_slug")
            if not slug:
                continue
            row = db.get_component(conn, slug)
            if not row:
                logger.warning("compose chose unknown slug %r — skipping", slug)
                continue
            components.append(critic.PlanComponent(
                slug=slug, name=row["name"], role=c.get("role", "unspecified"),
            ))
    return components, pattern


# ─── 4. Critique loop with revisions ───────────────────────────────────────
def _format_critic_feedback(report: critic.CritiqueReport) -> str:
    lines = [f"Critic severity: {report.severity}"]
    for it in report.issues:
        lines.append(f"  [{it.severity}] {it.category}: {it.message}")
        if it.suggestion:
            lines.append(f"      suggestion: {it.suggestion}")
    return "\n".join(lines)


# ─── 5. Render ─────────────────────────────────────────────────────────────
def _render_markdown(plan: FinalPlan) -> str:
    parts = [
        "# Implementation plan",
        "",
        f"**Goal.** {plan.user_spec.strip()}",
        "",
        f"**Pattern.** {plan.pattern or '(unlabeled)'}",
        "",
        "## Components",
    ]
    with db.connect() as conn:
        for c in plan.components:
            row = db.get_component(conn, c.slug)
            ol = row["one_liner"] if row else ""
            url = row["homepage_url"] if row else ""
            link = f"[{c.name}]({url})" if url else c.name
            parts.append(f"- **{c.role}** — {link}. {ol}")
    parts.extend(["", "## How it fits together"])
    parts.append("Implementation order:")
    for i, c in enumerate(plan.components, 1):
        parts.append(f"{i}. Wire up **{c.name}** as `{c.role}`.")
    if plan.critic_report and plan.critic_report.issues:
        parts.extend(["", "## Caveats from review"])
        for it in plan.critic_report.issues:
            parts.append(f"- *{it.severity}*: {it.message}"
                          + (f"  → {it.suggestion}" if it.suggestion else ""))
    return "\n".join(parts)


def _render_cursor_spec(plan: FinalPlan) -> str:
    """A copy-paste prompt for Cursor / Claude Code."""
    parts = [
        "You are implementing the following system. Use the listed components.",
        "",
        f"GOAL: {plan.user_spec.strip()}",
        f"PATTERN: {plan.pattern}",
        "",
        "STACK:",
    ]
    with db.connect() as conn:
        for c in plan.components:
            row = db.get_component(conn, c.slug)
            if not row:
                continue
            parts.append(
                f"  - {c.role} := {c.name}  ({row['homepage_url'] or ''})"
            )
    parts.extend(["", "Implementation order:"])
    for i, c in enumerate(plan.components, 1):
        parts.append(f"  {i}. {c.role} ({c.name})")
    parts.extend([
        "",
        "Constraints: use the exact components listed (do not substitute).",
        "Start with a minimal end-to-end vertical slice; we'll add features after.",
    ])
    return "\n".join(parts)


def render(plan: FinalPlan, format: Literal["markdown", "cursor"] = "markdown") -> str:
    if format == "cursor":
        return _render_cursor_spec(plan)
    return _render_markdown(plan)


# ─── 6. Persist for analytics + future RL signal ────────────────────────────
def _persist_query(plan: FinalPlan, format: str, user_email: str = "") -> None:
    with db.connect() as conn:
        comp_ids = []
        for c in plan.components:
            row = db.get_component(conn, c.slug)
            if row:
                comp_ids.append(row["id"])
        conn.execute(
            "INSERT INTO user_queries (query, plan_components_json, "
            "plan_format, user_email) VALUES (?, ?, ?, ?)",
            (plan.user_spec, json.dumps(comp_ids), format, user_email),
        )
        # Reinforce importance + last_referenced_at on each chosen component
        for cid in comp_ids:
            conn.execute(
                "UPDATE components SET "
                "  importance = MIN(1.5, importance + 0.05), "
                "  last_referenced_at = ? "
                "WHERE id = ?",
                (datetime.utcnow().isoformat(timespec="seconds"), cid),
            )


# ─── Main entry point ──────────────────────────────────────────────────────
def plan_for_spec(
    spec: str,
    format: Literal["markdown", "cursor"] = "markdown",
    allow_live_search: bool = True,
    deep_critic: bool = False,
    max_revisions: int = 2,
    user_email: str = "",
) -> FinalPlan:
    """End-to-end planning. Returns a FinalPlan with .rendered populated."""
    # 1. Decompose
    reqs, pattern_guess = decompose(spec)
    if not reqs:
        return FinalPlan(user_spec=spec, components=[], pattern="",
                          rendered="(planner could not decompose spec)")

    # 2. Discover
    discoveries = discover_for_requirements(
        reqs, top_k=3, allow_live_search=allow_live_search,
    )

    # 3. Compose + 4. Critique loop
    components, pattern = compose(spec, reqs, pattern_guess, discoveries)
    plan = FinalPlan(
        user_spec=spec, components=components, pattern=pattern,
        requirements=reqs, discovery_candidates=discoveries,
    )

    for revision in range(max_revisions + 1):
        report = critic.critique_cheap(critic.Plan(
            user_spec=spec, components=components, architecture_pattern=pattern,
        ))
        plan.critic_report = report
        plan.revisions = revision
        if report.severity in ("clean", "minor"):
            break
        if revision >= max_revisions:
            break
        # Send critic feedback back to compose
        feedback = _format_critic_feedback(report)
        components, pattern = compose(
            spec, reqs, pattern_guess, discoveries,
            revision_feedback=feedback,
        )
        plan.components = components
        plan.pattern = pattern

    # 5. Optional deep critic
    if deep_critic:
        deep = critic.critique_deep(critic.Plan(
            user_spec=spec, components=components, architecture_pattern=pattern,
        ))
        # Merge: keep the worse severity, append issues
        if deep.issues:
            existing = plan.critic_report or critic.CritiqueReport(severity="clean")
            severity_order = {"clean": 0, "minor": 1, "major": 2, "blocking": 3}
            new_sev = max(existing.severity, deep.severity,
                           key=lambda s: severity_order.get(s, 0))
            plan.critic_report = critic.CritiqueReport(
                severity=new_sev,
                issues=existing.issues + deep.issues,
                summary=(existing.summary + " | " + deep.summary).strip(" |"),
            )

    # 6. Render + persist
    plan.rendered = render(plan, format=format)
    try:
        _persist_query(plan, format=format, user_email=user_email)
    except Exception as e:
        logger.warning("persist_query failed: %s", e)
    return plan
