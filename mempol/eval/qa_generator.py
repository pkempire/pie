"""Generate evaluation QAs from a conversation when no gold labels exist.

Use when running on personal data (ChatGPT export, Slack threads, etc.)
where there are no LoCoMo-style `evidence` annotations. The generator
prompts a strong LLM (default gpt-4o) to read the full conversation and
emit comprehension questions whose answers it can verify against the
same conversation. Each question is also labelled with one of the LoCoMo
categories so the per-category eval pipeline still works.

Cost: roughly one gpt-4o call per conversation, ~$0.05 per ~30k-token
conversation. Cache results to avoid re-running.
"""
from __future__ import annotations
import hashlib
import json
import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterable

from .. import config, llm

logger = logging.getLogger(__name__)


_GEN_SYSTEM = (
    "You are generating evaluation questions for a long conversational "
    "memory system. Read the conversation, then write questions whose "
    "answers can be verified against the conversation alone (no outside "
    "knowledge). Cover a mix of categories. Return ONLY a JSON array."
)

_GEN_USER_TEMPLATE = """Conversation transcript:

{transcript}

Produce {n} questions about this conversation. Each entry must be:
{{"question": "...",
  "gold_answer": "...",
  "category": one of "single-hop" | "multi-hop" | "temporal" | "open-domain" | "adversarial",
  "evidence_text": "verbatim sentence(s) from the conversation that justify the answer"}}

Aim for the same proportions as LoCoMo: 50% single-hop, 25% multi-hop,
15% temporal, 7% open-domain, 3% adversarial. Adversarial means a
question that *sounds* plausible but the conversation actually contradicts
or never says — gold_answer should reflect the contradiction or
"not stated in conversation."

Return ONLY the JSON array, no commentary."""


@dataclass
class GeneratedQA:
    question: str
    gold_answer: str
    category: str                                  # see GEN_USER for valid set
    evidence_text: str                             # verbatim quote
    evidence_dia_ids: list[str] = field(default_factory=list)
    # ↑ filled in by `link_evidence_to_dia_ids` below if the conv has
    # dia_ids; left empty otherwise.


def generate(
    transcript: str,
    n: int = 10,
    model: str | None = None,
    cache_dir: Path | None = None,
) -> list[GeneratedQA]:
    """Generate n evaluation QAs from a transcript string.

    The transcript should be plain text with speaker prefixes
    (e.g. "Caroline: ...\nMelanie: ...\n...") — same shape we use in
    the chunked-window backends.

    Caching: if cache_dir is given, results are keyed by sha256(transcript)
    and reused on later calls.
    """
    model = model or config.OBSERVER_MODEL or "gpt-4o-mini"

    if cache_dir is not None:
        cache_dir = Path(cache_dir)
        cache_dir.mkdir(parents=True, exist_ok=True)
        key = hashlib.sha256(
            (transcript + f"::n={n}::m={model}").encode("utf-8")
        ).hexdigest()[:16]
        cache_path = cache_dir / f"qa_{key}.json"
        if cache_path.exists():
            try:
                cached = json.loads(cache_path.read_text())
                return [GeneratedQA(**c) for c in cached]
            except Exception:
                pass

    msgs = [
        {"role": "system", "content": _GEN_SYSTEM},
        {"role": "user",   "content": _GEN_USER_TEMPLATE.format(
            transcript=transcript[:120_000],         # truncate at ~30k tokens
            n=n)},
    ]
    raw = llm.chat(msgs, model=model, json_mode=True)
    try:
        data = json.loads(raw)
        if isinstance(data, dict) and "questions" in data:
            data = data["questions"]
    except Exception as e:
        logger.warning("QA generator parse failure on model=%s: %s; raw=%r",
                       model, e, raw[:200])
        return []

    out: list[GeneratedQA] = []
    for d in data if isinstance(data, list) else []:
        try:
            out.append(GeneratedQA(
                question=str(d["question"]),
                gold_answer=str(d["gold_answer"]),
                category=str(d.get("category", "single-hop")),
                evidence_text=str(d.get("evidence_text", "")),
            ))
        except Exception:
            continue

    if cache_dir is not None:
        cache_path.write_text(json.dumps([qa.__dict__ for qa in out], indent=2))
    return out


def link_evidence_to_dia_ids(
    qas: list[GeneratedQA],
    turns: Iterable,
) -> list[GeneratedQA]:
    """Map each QA's evidence_text back to dia_ids in the source conversation.

    Heuristic: longest substring overlap. Works because the QA generator
    quotes verbatim. Used only when we have dia_ids on hand (e.g. running
    Mode B on top of LoCoMo conversations as a sanity check vs. Mode A
    gold labels).
    """
    turn_list = list(turns)
    for qa in qas:
        if not qa.evidence_text:
            continue
        ev_norm = qa.evidence_text.lower()
        scored = []
        for t in turn_list:
            txt = (getattr(t, "text", "") or "").lower()
            if not txt:
                continue
            # crude longest-common-substring proxy: count shared 5-grams
            grams = {ev_norm[i:i+5] for i in range(len(ev_norm) - 4)}
            score = sum(1 for g in grams if g in txt) / max(len(grams), 1)
            scored.append((score, getattr(t, "dia_id", "")))
        scored.sort(reverse=True)
        qa.evidence_dia_ids = [d for s, d in scored[:2] if s > 0.2 and d]
    return qas


def _smoke():
    transcript = """\
Caroline: I'm thinking of moving to Boston next month for the new job at Whoop.
Melanie: That's huge! When does it start?
Caroline: April 12th. I'm freaking out about packing.
Melanie: I'll come help that weekend. Wait did you find an apartment?
Caroline: Yeah, in Cambridge. Two bedroom, but I'll be alone for the first month.
"""
    qas = generate(transcript, n=3, model="gpt-4o-mini")
    print(f"generated {len(qas)} QAs")
    for qa in qas:
        print(f"  [{qa.category}] {qa.question}")
        print(f"    gold: {qa.gold_answer}")
        print(f"    ev:   {qa.evidence_text[:80]}...")


if __name__ == "__main__":
    _smoke()
