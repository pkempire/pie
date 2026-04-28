"""Head-to-head: learned-W vs hardcoded PIE extraction on a ChatGPT export.

This is the eval that does NOT depend on LoCoMo's gold evidence labels.
For each conversation in a ChatGPT export:

  1. Generate K evaluation QAs from the conversation (Mode B).
  2. Build a memory state two ways:
       a. Hardcoded PIE prompt-based extraction (the existing
          pie/ingestion/pipeline.py path).
       b. Learned write policy — either the heuristic teacher
          (mempol.policies.v1_write) or, once we have a checkpoint,
          a Tinker-trained LoRA invoked through the same write tools.
  3. Run the heuristic read policy R against each memory state to
     answer the K QAs.
  4. Judge both answers against the QA's gold (gpt-4o judge with the
     LongMemEval bucketing protocol).
  5. Report win rate + per-category breakdown + memory-size comparison.

Usage:

    python -m mempol.scripts.headtohead_chatgpt \\
        --export ~/Downloads/conversations.json \\
        --n_convs 20 --qas_per_conv 8 \\
        --out runs/headtohead_$(date +%Y%m%d).json

The export argument expects the standard ChatGPT export format
(`conversations.json` from the Settings → Data export flow).
"""
from __future__ import annotations
import argparse
import json
import logging
from collections import defaultdict
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

from mempol import config
from mempol.backends.base import Unit
from mempol.backends.pie_kg import PIEBackend
from mempol.eval.judge import judge as _judge
from mempol.eval.qa_generator import GeneratedQA, generate as generate_qas
from mempol.policies.v1_heuristic import HeuristicPolicy
from mempol.policies.v1_write import HeuristicWritePolicy

logger = logging.getLogger(__name__)


# ─── ChatGPT export parsing ─────────────────────────────────────────────────
def _parse_chatgpt_export(path: Path) -> list[dict]:
    """Yield {title, turns:[(role, text)], created_at} per conversation."""
    raw = json.loads(path.read_text())
    convs = []
    for entry in raw if isinstance(raw, list) else raw.get("conversations", []):
        title = entry.get("title", "")
        created = entry.get("create_time") or entry.get("created_at")
        mapping = entry.get("mapping") or {}
        turns: list[tuple[str, str]] = []
        # Walk the message tree in order. Mapping is a dict keyed by node id
        # with parent/children pointers; we follow children depth-first.
        roots = [k for k, v in mapping.items()
                 if v.get("parent") is None and (v.get("message") or {}).get("content")]
        if not roots and mapping:
            # Fall back: just iterate in insertion order.
            roots = [next(iter(mapping))]
        seen = set()
        for r in roots:
            stack = [r]
            while stack:
                node_id = stack.pop()
                if node_id in seen:
                    continue
                seen.add(node_id)
                node = mapping.get(node_id) or {}
                msg = node.get("message") or {}
                role = (msg.get("author") or {}).get("role", "")
                content_block = msg.get("content") or {}
                parts = content_block.get("parts") or []
                text = "\n".join(str(p) for p in parts if isinstance(p, str)).strip()
                if role in ("user", "assistant") and text:
                    turns.append((role, text))
                stack.extend(node.get("children") or [])
        if turns:
            convs.append({
                "title": title,
                "created_at": created,
                "turns": turns,
            })
    return convs


# ─── Ingestion arms ─────────────────────────────────────────────────────────
def _ingest_with_pie_prompt(turns: list[tuple[str, str]]) -> PIEBackend:
    """Use the existing prompted extraction pipeline. Re-uses PIE's own
    extraction prompts and 3-tier resolver; this is the hand-tuned baseline."""
    from pie.ingestion.pipeline import IngestionPipeline
    from pie.core.world_model import WorldModel
    wm = WorldModel()
    pipeline = IngestionPipeline(wm)
    transcript = "\n".join(f"{r}: {t}" for r, t in turns)
    pipeline.ingest_text(transcript, conversation_id="chatgpt_export")
    return PIEBackend(world_model=wm)


def _ingest_with_heuristic_w(turns: list[tuple[str, str]]) -> PIEBackend:
    """Use the existing heuristic write policy as a stand-in for the
    learned write policy (until we have a Tinker checkpoint to plug in)."""
    backend = PIEBackend()
    write_policy = HeuristicWritePolicy()
    for ti, (role, text) in enumerate(turns):
        write_policy.process_turn(
            backend=backend,
            turn_text=f"{role}: {text}",
            dia_id=f"D1:{ti}",
            timestamp=float(ti),
        )
    return backend


# ─── Run the eval ───────────────────────────────────────────────────────────
@dataclass
class ConvResult:
    title: str
    n_turns: int
    n_qas: int
    pie_score: float = 0.0
    heuristic_w_score: float = 0.0
    pie_n_entities: int = 0
    heuristic_w_n_entities: int = 0
    per_category_pie:    dict[str, float] = field(default_factory=dict)
    per_category_heur_w: dict[str, float] = field(default_factory=dict)


def _score_one_qa(qa: GeneratedQA, backend: PIEBackend, reader: HeuristicPolicy) -> float:
    trace = reader.run(qa.question, backend)
    answer = trace.answer or "not in context"
    score, _ = _judge(qa.question, qa.gold_answer, answer)
    return float(score)


def run_headtohead(
    export_path: Path,
    n_convs: int = 5,
    qas_per_conv: int = 8,
    qa_model: str = "gpt-4o-mini",
    cache_dir: Path | None = None,
) -> list[ConvResult]:
    convs = _parse_chatgpt_export(export_path)
    logger.info("Loaded %d conversations from %s", len(convs), export_path)
    convs = [c for c in convs if len(c["turns"]) >= 6][:n_convs]
    logger.info("Filtered to %d conversations with ≥6 turns", len(convs))

    reader = HeuristicPolicy(first_k=8, final_k=4,
                              do_reformulate=True, do_expand=True)

    results: list[ConvResult] = []
    for ci, conv in enumerate(convs, 1):
        title = conv["title"][:60] or f"conv_{ci}"
        turns = conv["turns"]
        transcript = "\n".join(f"{r}: {t}" for r, t in turns)
        logger.info("[%d/%d] %s — %d turns", ci, len(convs), title, len(turns))

        qas = generate_qas(
            transcript=transcript,
            n=qas_per_conv,
            model=qa_model,
            cache_dir=cache_dir,
        )
        if not qas:
            logger.warning("  no QAs generated, skipping")
            continue

        logger.info("  building PIE-prompt KG...")
        kb_pie = _ingest_with_pie_prompt(turns)
        logger.info("  building heuristic-W KG...")
        kb_heur = _ingest_with_heuristic_w(turns)

        per_cat_pie:  dict[str, list[float]] = defaultdict(list)
        per_cat_heur: dict[str, list[float]] = defaultdict(list)
        pie_total, heur_total = 0.0, 0.0
        for qa in qas:
            sp = _score_one_qa(qa, kb_pie, reader)
            sh = _score_one_qa(qa, kb_heur, reader)
            pie_total += sp; heur_total += sh
            per_cat_pie[qa.category].append(sp)
            per_cat_heur[qa.category].append(sh)

        result = ConvResult(
            title=title,
            n_turns=len(turns),
            n_qas=len(qas),
            pie_score=pie_total / len(qas),
            heuristic_w_score=heur_total / len(qas),
            pie_n_entities=len(kb_pie.wm.entities),
            heuristic_w_n_entities=len(kb_heur.wm.entities),
            per_category_pie={k: sum(v)/len(v) for k, v in per_cat_pie.items()},
            per_category_heur_w={k: sum(v)/len(v) for k, v in per_cat_heur.items()},
        )
        results.append(result)
        logger.info("  pie=%.3f  heuristic_w=%.3f  Δ=%+.3f",
                    result.pie_score, result.heuristic_w_score,
                    result.heuristic_w_score - result.pie_score)
    return results


def _summarise(results: list[ConvResult]) -> dict:
    if not results:
        return {"n": 0}
    n = len(results)
    pie_avg  = sum(r.pie_score for r in results) / n
    heur_avg = sum(r.heuristic_w_score for r in results) / n
    wins_heur = sum(1 for r in results if r.heuristic_w_score > r.pie_score)
    cat_pie:  dict[str, list[float]] = defaultdict(list)
    cat_heur: dict[str, list[float]] = defaultdict(list)
    for r in results:
        for k, v in r.per_category_pie.items():    cat_pie[k].append(v)
        for k, v in r.per_category_heur_w.items(): cat_heur[k].append(v)
    return {
        "n_convs": n,
        "avg_pie_score": pie_avg,
        "avg_heuristic_w_score": heur_avg,
        "delta": heur_avg - pie_avg,
        "win_rate_heuristic_w": wins_heur / n,
        "per_category": {
            k: {
                "pie":         sum(cat_pie[k])  / max(len(cat_pie[k]), 1),
                "heuristic_w": sum(cat_heur[k]) / max(len(cat_heur[k]), 1),
            }
            for k in sorted(set(list(cat_pie) + list(cat_heur)))
        },
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--export", required=True, type=Path,
                        help="path to conversations.json from a ChatGPT data export")
    parser.add_argument("--n_convs", type=int, default=5)
    parser.add_argument("--qas_per_conv", type=int, default=8)
    parser.add_argument("--qa_model", default="gpt-4o-mini")
    parser.add_argument("--cache_dir", type=Path, default=Path("mempol/.cache/headtohead_qa"))
    parser.add_argument("--out", type=Path, default=Path("runs/headtohead.json"))
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO,
                        format="%(asctime)s %(levelname)s %(message)s")
    args.out.parent.mkdir(parents=True, exist_ok=True)

    results = run_headtohead(
        export_path=args.export,
        n_convs=args.n_convs,
        qas_per_conv=args.qas_per_conv,
        qa_model=args.qa_model,
        cache_dir=args.cache_dir,
    )
    summary = _summarise(results)

    payload = {
        "summary": summary,
        "per_conv": [asdict(r) for r in results],
    }
    args.out.write_text(json.dumps(payload, indent=2))
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
