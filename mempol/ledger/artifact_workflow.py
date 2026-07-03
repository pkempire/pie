"""Versioned LLM artifact workflow for research/content drafts.

This is the agentic layer above deterministic ingestion. It lets a model create
or revise a durable artifact from selected evidence, asks specialist reviewers
to critique it, then asks a meta-reviewer to compare versions. All outputs are
plain files so they can be inspected, committed, and ingested back into the
ledger.
"""
from __future__ import annotations

import argparse
import json
import os
import re
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

from mempol import config
from mempol.core.store import now_iso
from mempol.llm import chat


DEFAULT_WRITER_PROMPT = """You are writing a serious public technical artifact.

Audience:
- frontier AI researchers, systems builders, and technical founders
- readers who understand transformers, retrieval, agents, and evals

Style:
- clear, direct, professional
- technically precise without sounding academic for its own sake
- no hype, no buzzwords, no anthropomorphic filler, no vague motivational framing
- define terms before making claims
- explain architecture first when architecture is the causal mechanism
- use examples only when they sharpen the mechanism
- preserve uncertainty and avoid overclaiming

Task:
Use the supplied sources to write the requested artifact. Keep the strongest
argument. Cut weak framing. Add citations or source labels where claims depend
on evidence. Prefer one clean technical argument over many loosely connected ideas.
"""


REVIEWER_PROMPTS: dict[str, str] = {
    "taste": """You are a strict editor with excellent taste.
Review for tone, clarity, flow, credibility, and cringe. Flag vague claims,
buzzwords, weak openings, corny lines, and places where the prose feels like
AI slop. Give concrete edits, not encouragement.""",
    "science": """You are a careful AI research reviewer.
Review factual accuracy, strength of evidence, novelty, missing citations,
overclaims, and places where temporal reasoning, temporal awareness, and
temporal memory are being confused. Be precise.""",
    "product": """You are a product strategist for agent infrastructure.
Review whether this artifact explains a real pain point, a credible product
direction, and a concrete demo. Flag places where the work sounds theoretical
without showing what can be built or used.""",
}


META_PROMPT = """You are the meta-reviewer.
Compare the prior version, the new version, and reviewer notes.

Return:
1. verdict: keep_new, keep_old, or revise_again
2. strongest improvements
3. regressions
4. required next edits
5. a concise release-readiness score from 0 to 10

Be direct. Do not flatter the draft.
"""


@dataclass
class SourceDoc:
    path: str
    title: str
    text: str
    chars: int


@dataclass
class WorkflowResult:
    artifact_id: str
    run_name: str
    out_dir: str
    created_at: str
    objective: str
    sources: list[dict[str, Any]]
    draft_path: str
    reviews: dict[str, str]
    comparison_path: str
    manifest_path: str
    dry_run: bool


def _slug(text: str) -> str:
    s = re.sub(r"[^a-zA-Z0-9]+", "-", text.strip().lower()).strip("-")
    return s[:80] or "artifact"


def _read_source(path: Path, max_chars: int) -> SourceDoc:
    text = path.read_text(encoding="utf-8", errors="replace")
    if len(text) > max_chars:
        text = text[:max_chars] + "\n\n[TRUNCATED]"
    return SourceDoc(path=str(path), title=path.name, text=text, chars=len(text))


def _format_sources(sources: list[SourceDoc]) -> str:
    chunks = []
    for i, src in enumerate(sources, 1):
        chunks.append(
            f"## Source {i}: {src.path}\n\n"
            f"{src.text.strip()}\n"
        )
    return "\n\n---\n\n".join(chunks)


def _write(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")


def _call(model: str, system: str, user: str, max_tokens: int) -> str:
    return chat(
        [
            {"role": "system", "content": system},
            {"role": "user", "content": user},
        ],
        model=model,
        max_tokens=max_tokens,
    )


def run_workflow(
    *,
    run_name: str,
    artifact_id: str,
    objective: str,
    sources: list[Path],
    prior: Path | None = None,
    model: str = "gpt-5-mini",
    review_model: str = "gpt-5-mini",
    compare_model: str = "gpt-5-mini",
    max_source_chars: int = 18000,
    max_draft_tokens: int = 5000,
    dry_run: bool = False,
) -> WorkflowResult:
    created_at = now_iso()
    slug = _slug(artifact_id)
    out_dir = config.RESULTS_DIR / run_name / "artifact_workflows" / slug
    source_docs = [_read_source(path, max_source_chars) for path in sources]
    source_text = _format_sources(source_docs)
    prior_text = prior.read_text(encoding="utf-8", errors="replace") if prior and prior.exists() else ""

    writer_user = (
        f"Objective:\n{objective}\n\n"
        f"Prior version, if any:\n{prior_text or '[none]'}\n\n"
        f"Sources:\n{source_text}\n\n"
        "Write the artifact now."
    )

    prompts_dir = out_dir / "prompts"
    _write(prompts_dir / "writer.md", f"# System\n\n{DEFAULT_WRITER_PROMPT}\n\n# User\n\n{writer_user}")

    if dry_run:
        draft = (
            "# Dry Run\n\n"
            "Set `OPENAI_API_KEY` and run without `--dry-run` to generate the artifact.\n"
        )
    else:
        if not os.getenv("OPENAI_API_KEY"):
            raise RuntimeError("OPENAI_API_KEY is required unless --dry-run is set")
        draft = _call(model, DEFAULT_WRITER_PROMPT, writer_user, max_draft_tokens)

    draft_path = out_dir / "draft.md"
    _write(draft_path, draft)

    review_paths: dict[str, str] = {}
    review_texts: dict[str, str] = {}
    for name, reviewer_prompt in REVIEWER_PROMPTS.items():
        review_user = (
            f"Objective:\n{objective}\n\n"
            f"Draft:\n{draft}\n\n"
            f"Sources:\n{source_text}\n\n"
            "Review this draft against the objective."
        )
        _write(prompts_dir / f"review_{name}.md", f"# System\n\n{reviewer_prompt}\n\n# User\n\n{review_user}")
        if dry_run:
            review = f"# Dry Run Review: {name}\n\nNo model call made."
        else:
            review = _call(review_model, reviewer_prompt, review_user, 1800)
        path = out_dir / f"review_{name}.md"
        _write(path, review)
        review_paths[name] = str(path)
        review_texts[name] = review

    comparison_user = (
        f"Objective:\n{objective}\n\n"
        f"Prior version:\n{prior_text or '[none]'}\n\n"
        f"New version:\n{draft}\n\n"
        f"Reviewer notes:\n{json.dumps(review_texts, indent=2, ensure_ascii=False)}"
    )
    _write(prompts_dir / "meta_compare.md", f"# System\n\n{META_PROMPT}\n\n# User\n\n{comparison_user}")
    if dry_run:
        comparison = "# Dry Run Comparison\n\nNo model call made."
    else:
        comparison = _call(compare_model, META_PROMPT, comparison_user, 1800)
    comparison_path = out_dir / "comparison.md"
    _write(comparison_path, comparison)

    result = WorkflowResult(
        artifact_id=artifact_id,
        run_name=run_name,
        out_dir=str(out_dir),
        created_at=created_at,
        objective=objective,
        sources=[asdict(src) for src in source_docs],
        draft_path=str(draft_path),
        reviews=review_paths,
        comparison_path=str(comparison_path),
        manifest_path=str(out_dir / "manifest.json"),
        dry_run=dry_run,
    )
    _write(out_dir / "manifest.json", json.dumps(asdict(result), indent=2, ensure_ascii=False))
    return result


def main() -> None:
    ap = argparse.ArgumentParser(description="Generate/review a versioned research artifact.")
    ap.add_argument("--run-name", default="artifact_workflow")
    ap.add_argument("--artifact-id", required=True)
    ap.add_argument("--objective", required=True)
    ap.add_argument("--source", action="append", required=True, help="Source file. Repeatable.")
    ap.add_argument("--prior", default="")
    ap.add_argument("--model", default="gpt-5-mini")
    ap.add_argument("--review-model", default="gpt-5-mini")
    ap.add_argument("--compare-model", default="gpt-5-mini")
    ap.add_argument("--max-source-chars", type=int, default=18000)
    ap.add_argument("--max-draft-tokens", type=int, default=5000)
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()

    result = run_workflow(
        run_name=args.run_name,
        artifact_id=args.artifact_id,
        objective=args.objective,
        sources=[Path(p) for p in args.source],
        prior=Path(args.prior) if args.prior else None,
        model=args.model,
        review_model=args.review_model,
        compare_model=args.compare_model,
        max_source_chars=args.max_source_chars,
        max_draft_tokens=args.max_draft_tokens,
        dry_run=args.dry_run,
    )
    print(json.dumps(asdict(result), indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
