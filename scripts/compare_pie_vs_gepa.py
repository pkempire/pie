"""3-way apples-to-apples comparison on LoCoMo conv-26:

    1. PIE  (cached typed-KG)      — load benchmarks/locomo/cache/conv-26_wm.json
    2. HAND (hand-coded consolidator) — prompt_original.txt  -> FlatBackend
    3. GEPA (GEPA-evolved consolidator) — prompt_optimized.txt -> FlatBackend

All three are read by the SAME frozen reader (HeuristicPolicy) and scored by
the SAME judge, on the SAME question set, so the only thing that varies is the
write/storage layer. This reuses artifacts already on disk (the cached KG and
the two saved prompts) so it does NOT re-run the hours-long GEPA optimization.

Why this script exists
----------------------
The Cowork sandbox caps every shell command at 45s and won't persist long jobs,
so the full GEPA *optimization* can't run there. But the *comparison* of the
already-evolved prompt against PIE is cheap and is the number Parth actually
asked for. Run this on a normal machine:

    pip install openai numpy
    export OPENAI_API_KEY=...            # or rely on .env
    python scripts/compare_pie_vs_gepa.py --max-questions 30 --max-chunks 8

Results stream to mempol/results/pie_vs_gepa/ as JSONL (resumable) plus a
summary.json + a printed table. Estimated cost at --max-questions 30:
~$2-4 with gpt-5-mini answers + gpt-4o-mini judge (embeddings are cheap).
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

_REPO = Path(__file__).resolve().parent.parent
if str(_REPO) not in sys.path:
    sys.path.insert(0, str(_REPO))

# Load .env if present (so OPENAI_API_KEY is available without exporting).
_envf = _REPO / ".env"
if _envf.exists():
    for line in _envf.read_text().splitlines():
        line = line.strip()
        if line and not line.startswith("#") and "=" in line:
            k, v = line.split("=", 1)
            import os
            os.environ.setdefault(k.strip(), v.strip().strip('"').strip("'"))

from mempol import config, llm                              # noqa: E402
from mempol.backends.base import Unit                       # noqa: E402
from mempol.backends.flat import FlatBackend                # noqa: E402
from mempol.backends.pie_kg import PIEBackend               # noqa: E402
from mempol.data import locomo                              # noqa: E402
from mempol.eval.judge import judge                         # noqa: E402
from mempol.policies.v1_heuristic import HeuristicPolicy    # noqa: E402
from pie.core.world_model import WorldModel                 # noqa: E402

CACHE_KG = _REPO / "benchmarks/locomo/cache/conv-26_wm.json"
PROMPT_ORIG = _REPO / "mempol/results/gepa_consolidator/prompt_original.txt"
PROMPT_OPT = _REPO / "mempol/results/gepa_consolidator/prompt_optimized.txt"
OUT_DIR = config.RESULTS_DIR / "pie_vs_gepa"

_ANSWER_SYS = (
    "You answer questions about a long-running conversation between two people "
    "using only the consolidated memory entries provided. "
    "Be concise (one sentence). If the answer is not present, say 'Not in context'."
)


def _mempol_model_name(model: str) -> str:
    """The local OpenAI wrapper expects raw OpenAI model ids, not DSPy provider ids."""
    return model.removeprefix("openai/")


def chunk_turns(turns, size=30):
    return [turns[i:i + size] for i in range(0, len(turns), size)]


def _format_turns(turns) -> str:
    lines = []
    for t in turns:
        sd = getattr(t, "session_date", "") or getattr(t, "date", "") or ""
        lines.append(f"[{t.dia_id} | {t.speaker} | {sd}] {t.text}")
    return "\n".join(lines)


def consolidate_chunk(turns, system_prompt: str, model: str) -> list[dict]:
    """Run a consolidator (defined purely by `system_prompt`) over one chunk.
    Returns a list of entry dicts. No DSPy needed — raw JSON-mode LLM call."""
    user = (
        f"working_region (one conversation session):\n{_format_turns(turns)}\n\n"
        "Return ONLY a JSON object of the form "
        '{"entries": [ {"entry_type": "...", "name": "...", "summary": "...", '
        '"details": "...", "steps": [], "speaker": "...", "source_turn_ids": [...]} ]}'
    )
    msgs = [{"role": "system", "content": system_prompt},
            {"role": "user", "content": user}]
    try:
        raw = llm.chat(msgs, model=_mempol_model_name(model), json_mode=True)
        obj = json.loads(raw)
        ents = obj.get("entries", obj if isinstance(obj, list) else [])
        return ents if isinstance(ents, list) else []
    except Exception as e:
        print(f"    [warn] consolidate failed: {e}")
        return []


def entry_to_unit(e: dict, idx: int) -> Unit:
    etype = e.get("entry_type", "semantic")
    if etype == "procedural":
        body = "\n".join(f"- {s}" for s in e.get("steps", []) or [])
    else:
        body = e.get("details", "") or ""
    text = (f"[{etype}] {e.get('name','')}\n"
            f"speaker: {e.get('speaker','')}\n"
            f"{e.get('summary','')}\n{body}").strip()
    return Unit(uid=f"conv-26::entry_{idx}", text=text,
                metadata={"entry_type": etype, "speaker": e.get("speaker", ""),
                          "source_turn_ids": e.get("source_turn_ids", []),
                          "name": e.get("name", "")})


def build_consolidated_backend(turns, system_prompt, model, max_chunks) -> FlatBackend:
    chunks = chunk_turns(turns, size=30)
    if max_chunks:
        chunks = chunks[:max_chunks]
    units, idx = [], 0
    for ci, ch in enumerate(chunks):
        ents = consolidate_chunk(ch, system_prompt, model)
        for e in ents:
            units.append(entry_to_unit(e, idx)); idx += 1
        print(f"    chunk {ci+1}/{len(chunks)}: +{len(ents)} entries (total {idx})")
    be = FlatBackend()
    be.ingest(units)
    return be


def answer(question, hits, model) -> str:
    ctx = "\n\n".join(f"[{i}] {h.unit.text}" for i, h in enumerate(hits, 1)) or "(none)"
    msgs = [{"role": "system", "content": _ANSWER_SYS},
            {"role": "user",
             "content": f"Memory entries:\n{ctx}\n\nQuestion: {question}\nAnswer:"}]
    try:
        return llm.chat(msgs, model=model).strip()
    except Exception as e:
        return f"error:{e}"


def eval_backend(name, backend, qas, answer_model, ckpt_path) -> list[dict]:
    """Read+judge each question. Resumable: skips qids already in ckpt_path."""
    done = {}
    if ckpt_path.exists():
        for line in ckpt_path.read_text().splitlines():
            try:
                r = json.loads(line); done[r["qid"]] = r
            except Exception:
                pass
    pol = HeuristicPolicy(do_reformulate=False, do_route=False, do_expand=True)
    results = []
    with ckpt_path.open("a", buffering=1) as f:
        for i, qa in enumerate(qas):
            if qa.qid in done:
                results.append(done[qa.qid]); continue
            t0 = time.time()
            tr = pol.run(qa.question, backend)
            # HeuristicPolicy may return its own answer; if not, answer from hits.
            ans = getattr(tr, "answer", None)
            if not ans:
                ans = answer(qa.question, getattr(tr, "hits", []) or [], answer_model)
            score, reason = judge(qa.question, qa.answer, ans)
            r = {"system": name, "qid": qa.qid,
                 "category_name": getattr(qa, "category_name", ""),
                 "question": qa.question, "gold": qa.answer, "answer": ans,
                 "score": score, "reason": reason, "secs": round(time.time() - t0, 1)}
            results.append(r); f.write(json.dumps(r) + "\n")
            if (i + 1) % 5 == 0:
                acc = sum(x["score"] for x in results) / len(results)
                print(f"    [{name}] q{i+1}/{len(qas)} acc={acc:.3f}")
    return results


def summarize(name, results) -> dict:
    from collections import defaultdict
    by = defaultdict(list)
    for r in results:
        by[r["category_name"]].append(r["score"])
    n = max(1, len(results))
    return {"system": name, "n": len(results),
            "overall_acc": round(sum(r["score"] for r in results) / n, 4),
            "by_category": {k: round(sum(v) / len(v), 4) for k, v in sorted(by.items())}}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--max-questions", type=int, default=30)
    ap.add_argument("--max-chunks", type=int, default=8,
                    help="Chunks of conv-26 fed to the consolidators (0=all).")
    ap.add_argument("--consolidator-model", default="gpt-5-mini")
    ap.add_argument("--answer-model", default="gpt-5-mini")
    ap.add_argument("--systems", default="pie,hand,gepa",
                    help="Comma list subset of {pie,hand,gepa}.")
    args = ap.parse_args()
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    want = set(args.systems.split(","))

    conv, qas = [(c, q) for c, q in locomo.load() if c.sample_id == "conv-26"][0]
    qas = [q for q in qas if q.answer][:args.max_questions]
    print(f"conv-26: {len(conv.turns)} turns, evaluating {len(qas)} questions")

    summaries = {}

    if "pie" in want:
        print("\n[PIE] loading cached typed-KG...")
        wm = WorldModel(persist_path=str(CACHE_KG))
        be = PIEBackend(world_model=wm)
        print(f"  KG: {len(wm.entities)} entities")
        res = eval_backend("pie", be, qas, args.answer_model, OUT_DIR / "pie.jsonl")
        summaries["pie"] = summarize("pie", res)

    if "hand" in want:
        print("\n[HAND] building store from hand-coded consolidator prompt...")
        be = build_consolidated_backend(conv.turns, PROMPT_ORIG.read_text(),
                                        args.consolidator_model, args.max_chunks)
        res = eval_backend("hand", be, qas, args.answer_model, OUT_DIR / "hand.jsonl")
        summaries["hand"] = summarize("hand", res)

    if "gepa" in want:
        print("\n[GEPA] building store from GEPA-evolved consolidator prompt...")
        be = build_consolidated_backend(conv.turns, PROMPT_OPT.read_text(),
                                        args.consolidator_model, args.max_chunks)
        res = eval_backend("gepa", be, qas, args.answer_model, OUT_DIR / "gepa.jsonl")
        summaries["gepa"] = summarize("gepa", res)

    (OUT_DIR / "summary.json").write_text(json.dumps(summaries, indent=2))
    print("\n==================  COMPARISON (conv-26)  ==================")
    print(f"{'system':<8}{'n':>4}{'overall_acc':>14}")
    for k in ("pie", "hand", "gepa"):
        if k in summaries:
            s = summaries[k]
            print(f"{k:<8}{s['n']:>4}{s['overall_acc']:>14.3f}")
    print("============================================================")
    print(f"Files: {OUT_DIR}/")


if __name__ == "__main__":
    main()
