"""Reflector × backend comparison matrix — on REAL LoCoMo data, REAL backends.

Answers two questions at once, on the same conversation, same reader, same judge:
  - which STORE wins:  flat vector  vs  knowledge-graph  vs  Mastra observational  vs git-tree
  - which REFLECTOR wins: raw turns (no reflection)  vs  hand-coded consolidator  vs  GEPA-learned

Each cell = ingest conv-26 with that (method, backend) -> read every question with the
SAME HeuristicPolicy -> judge -> accuracy. Reuses the real backends in mempol/backends/
and the two saved consolidator prompts (no DSPy needed — we replay the evolved prompt).

Run (on a machine with the deps, ~$5-12, ~20-40 min for all cells):
    pip install openai numpy
    python scripts/reflector_backend_matrix.py --max-questions 30
    # subsets:
    python scripts/reflector_backend_matrix.py --cells flat_raw,kg_raw,mastra,gepa_flat

Writes: mempol/results/reflector_matrix/summary.json
"""
from __future__ import annotations
import argparse, json, sys, time
from pathlib import Path

_REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_REPO)); sys.path.insert(0, str(_REPO/"scripts"))
_envf=_REPO/".env"
if _envf.exists():
    import os
    for line in _envf.read_text().splitlines():
        if line.strip() and not line.startswith("#") and "=" in line:
            k,v=line.split("=",1); os.environ.setdefault(k.strip(), v.strip().strip('"').strip("'"))

from mempol import config
from mempol.backends.flat import FlatBackend
from mempol.backends.pie_kg import PIEBackend
from mempol.backends.mastra import MastraBackend
from mempol.data import locomo
from mempol.eval.judge import judge
from mempol.eval.runner import conv_to_units
from mempol.policies.v1_heuristic import HeuristicPolicy
from pie.core.world_model import WorldModel
# reuse the real, tested consolidator helpers from the 3-way script
from compare_pie_vs_gepa import consolidate_chunk, entry_to_unit, chunk_turns, PROMPT_ORIG, PROMPT_OPT

CACHE_KG = _REPO/"benchmarks/locomo/cache/conv-26_wm.json"
OUT = config.RESULTS_DIR/"reflector_matrix"

def build_consolidated(turns, prompt, model, max_chunks):
    chunks = chunk_turns(turns, 30)[:max_chunks] if max_chunks else chunk_turns(turns,30)
    units, idx = [], 0
    for ci, ch in enumerate(chunks, start=1):
        print(f"    consolidating chunk {ci}/{len(chunks)}...", flush=True)
        entries = consolidate_chunk(ch, prompt, model)
        print(f"    chunk {ci}: {len(entries)} entries", flush=True)
        for e in entries:
            units.append(entry_to_unit(e, idx)); idx+=1
    if not units:
        raise RuntimeError(
            "consolidator produced zero entries; check model id/API errors before scoring this cell"
        )
    be=FlatBackend(); be.ingest(units); return be

def make_backend(cell, conv, model, max_chunks):
    """Return a ready-to-read backend for the named cell."""
    if cell=="flat_raw":                       # plain RAG baseline (today's default)
        be=FlatBackend(); be.ingest(conv_to_units(conv)); return be
    if cell=="kg_raw":                         # knowledge graph (cached PIE world model)
        return PIEBackend(world_model=WorldModel(persist_path=str(CACHE_KG)))
    if cell=="mastra":                         # Mastra Observational Memory (observer+reflector)
        be=MastraBackend(); be.ingest(conv_to_units(conv)); return be
    if cell=="hand_flat":                      # hand-coded reflector -> flat
        return build_consolidated(conv.turns, PROMPT_ORIG.read_text(), model, max_chunks)
    if cell=="gepa_flat":                      # GEPA-learned reflector -> flat
        return build_consolidated(conv.turns, PROMPT_OPT.read_text(), model, max_chunks)
    raise SystemExit(f"unknown cell {cell}")

ALL_CELLS=["flat_raw","kg_raw","mastra","hand_flat","gepa_flat"]

def eval_cell(name, backend, qas, ckpt):
    done={}
    if ckpt.exists():
        for ln in ckpt.read_text().splitlines():
            try: r=json.loads(ln); done[r["qid"]]=r
            except: pass
    pol=HeuristicPolicy(do_reformulate=False, do_route=False, do_expand=True)
    res=[]
    with ckpt.open("a",buffering=1) as f:
        for qa in qas:
            if qa.qid in done: res.append(done[qa.qid]); continue
            tr=pol.run(qa.question, backend)
            s,reason=judge(qa.question, qa.answer, tr.answer)
            r={"qid":qa.qid,"score":s,"q":qa.question,"ans":tr.answer}
            res.append(r); f.write(json.dumps(r)+"\n")
    return sum(r["score"] for r in res)/max(1,len(res))

def summarize_checkpoint(ckpt):
    if not ckpt.exists():
        return None
    rows = []
    for ln in ckpt.read_text().splitlines():
        try:
            rows.append(json.loads(ln))
        except Exception:
            pass
    if not rows:
        return None
    return {"acc": round(sum(r["score"] for r in rows) / len(rows), 4), "n": len(rows)}

def main():
    ap=argparse.ArgumentParser()
    ap.add_argument("--max-questions",type=int,default=30)
    ap.add_argument("--max-chunks",type=int,default=0,help="0=all chunks for consolidators")
    ap.add_argument("--conv",default="conv-26")
    ap.add_argument("--model",default="gpt-5-mini")
    ap.add_argument("--cells",default=",".join(ALL_CELLS))
    args=ap.parse_args()
    OUT.mkdir(parents=True,exist_ok=True)
    conv,qas=[(c,q) for c,q in locomo.load() if c.sample_id==args.conv][0]
    qas=[q for q in qas if q.answer]
    if args.max_questions:
        qas = qas[:args.max_questions]
    print(f"{args.conv}: {len(conv.turns)} turns, {len(qas)} questions\n")
    rows={}
    summary_path = OUT/"summary.json"
    if summary_path.exists():
        try:
            rows.update(json.loads(summary_path.read_text()))
        except Exception:
            pass
    for cell in args.cells.split(","):
        t0=time.time()
        print(f"[{cell}] building + evaluating...")
        be=make_backend(cell, conv, args.model, args.max_chunks or None)
        acc=eval_cell(cell, be, qas, OUT/f"{cell}.jsonl")
        rows[cell]={"acc":round(acc,4),"secs":round(time.time()-t0,1),"n":len(qas)}
        print(f"  -> {acc*100:.1f}%  ({rows[cell]['secs']}s)")
    for cell in ALL_CELLS:
        if cell not in rows:
            saved = summarize_checkpoint(OUT/f"{cell}.jsonl")
            if saved:
                rows[cell] = saved
    summary_path.write_text(json.dumps(rows,indent=2))
    print("\n=========== REFLECTOR × BACKEND (conv-26) ===========")
    label={"flat_raw":"flat vector (raw turns)","kg_raw":"knowledge graph (raw)",
           "mastra":"Mastra observational","hand_flat":"hand reflector -> flat","gepa_flat":"GEPA reflector -> flat"}
    for c in ALL_CELLS:
        if c in rows: print(f"  {label[c]:32} {rows[c]['acc']*100:5.1f}%")
    print("=====================================================")

if __name__=="__main__": main()
