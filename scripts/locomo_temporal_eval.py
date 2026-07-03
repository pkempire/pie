"""Real-data temporal eval: flat retrieval vs RLM state-reconstruction on LoCoMo.

This is the REAL-DATA port of the synthetic temporal/RLM demos. No mock data:
it uses LoCoMo conv-26's actual timestamped sessions and its 37 real 'temporal'
questions with real gold dates (e.g. "When did Caroline go to the LGBTQ support
group?" -> "7 May 2023"). Answering them requires resolving relative time
("yesterday" relative to the session's date) and ordering events — exactly where
a flat vector store has nothing to stand on.

  FLAT : every turn -> a dated unit ("[8 May 2023] Caroline: ..."), embed, top-k,
         answer. The store sees timestamps as text but has no timeline.
  RLM  : recurse over the log by SESSION, extract dated events (resolving relative
         dates against each session's timestamp) into one ordered timeline, then
         answer each question from that reconstructed timeline.

Models: current (June 2026). Default gpt-5.4-mini for the loop; embeddings 3-large.

Run a real number on all 37:
    python scripts/locomo_temporal_eval.py --max-questions 0
Quick smoke:
    python scripts/locomo_temporal_eval.py --max-questions 6
Writes: output/experiments/locomo_temporal_eval.json
"""
from __future__ import annotations
import argparse, json, os, math
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
for line in (REPO/".env").read_text().splitlines() if (REPO/".env").exists() else []:
    if line.strip() and not line.startswith("#") and "=" in line:
        k,v=line.split("=",1); os.environ.setdefault(k.strip(), v.strip().strip('"').strip("'"))
import sys; sys.path.insert(0, str(REPO))
from openai import OpenAI
from mempol.data import locomo
client=OpenAI()
EMB="text-embedding-3-large"

def chat(model, msgs, max_out=1200, json_mode=False):
    kw={"model":model,"messages":msgs,"max_completion_tokens":max_out}   # no temperature: 5.x reasoning models reject !=1
    if json_mode: kw["response_format"]={"type":"json_object"}
    return client.chat.completions.create(**kw).choices[0].message.content or ""

def embed(texts): return [d.embedding for d in client.embeddings.create(model=EMB,input=texts).data]
def cosine(a,b):
    dot=sum(x*y for x,y in zip(a,b)); na=math.sqrt(sum(x*x for x in a)); nb=math.sqrt(sum(y*y for y in b)); return dot/(na*nb+1e-9)
def judge(model,q,gold,pred):
    raw=chat(model,[{"role":"system","content":'Return JSON {"correct":true|false}. Correct iff the predicted date/time matches the gold (same day/month/year; phrasing may differ).'},
                    {"role":"user","content":f"Q:{q}\nGold:{gold}\nPred:{pred}"}],max_out=400,json_mode=True)
    try:return bool(json.loads(raw).get("correct"))
    except:return False

def main():
    ap=argparse.ArgumentParser()
    ap.add_argument("--max-questions",type=int,default=6)
    ap.add_argument("--model",default="gpt-5.4-nano")   # fast + current
    ap.add_argument("--top-k",type=int,default=6)
    args=ap.parse_args()
    OUT=REPO/"output/experiments"; OUT.mkdir(parents=True,exist_ok=True)
    ck=OUT/"locomo_temporal_eval.jsonl"
    EMB_CACHE=OUT/"_locomo_uembs.json"; TL_CACHE=OUT/"_locomo_timeline.jsonl"

    conv,qas=[(c,q) for c,q in locomo.load() if c.sample_id=="conv-26"][0]
    temporal=[q for q in qas if q.category_name=="temporal" and q.answer]
    if args.max_questions: temporal=temporal[:args.max_questions]
    units=[f"[{t.session_date}] {t.speaker}: {t.text}" for t in conv.turns]
    print(f"conv-26: {len(conv.turns)} turns, {len(temporal)} REAL temporal questions")

    # dated unit embeddings (cache once) ---------------------------------------
    if EMB_CACHE.exists():
        uembs=json.loads(EMB_CACHE.read_text())
    else:
        uembs=embed(units); EMB_CACHE.write_text(json.dumps(uembs)); print("cached embeddings")

    # RLM reconstruction: map over sessions -> timeline (resumable per session) -
    from collections import defaultdict
    sess=defaultdict(list)
    for t in conv.turns: sess[t.session].append(t)
    done_sessions=set(); timeline=[]
    if TL_CACHE.exists():
        for ln in TL_CACHE.read_text().splitlines():
            try: e=json.loads(ln); timeline.append(e); done_sessions.add(e.get("_session"))
            except: pass
    with TL_CACHE.open("a",buffering=1) as tf:
        for s in sorted(sess):
            if s in done_sessions: continue
            date=sess[s][0].session_date
            block="\n".join(f"{t.speaker}: {t.text}" for t in sess[s])
            raw=chat(args.model,[{"role":"system","content":(
                f"This conversation session happened on: {date}. Extract dated EVENTS as JSON "
                '{"events":[{"what":"...","date":"resolved calendar date, ISO if possible"}]}. '
                "Resolve relative times (yesterday, last week, last year) against the session date.")},
                {"role":"user","content":block}],max_out=900,json_mode=True)
            try:
                for e in json.loads(raw).get("events",[]):
                    ev={"_session":s,"session_date":date,**e}; timeline.append(ev); tf.write(json.dumps(ev)+"\n")
            except: pass
    tl_text="\n".join(f"- {e.get('what')}  (date: {e.get('date')})" for e in timeline)
    print(f"RECONSTRUCTED TIMELINE: {len(timeline)} dated events ({len(done_sessions)}/{len(sess)} sessions cached)\n")

    done={}
    if ck.exists():
        for ln in ck.read_text().splitlines():
            try: r=json.loads(ln); done[r["qid"]]=r
            except: pass
    res={"flat":[],"rlm":[]}
    with ck.open("a",buffering=1) as f:
        for q in temporal:
            if q.qid in done:
                res["flat"].append(done[q.qid]["f"]); res["rlm"].append(done[q.qid]["r"]); continue
            # FLAT
            qe=embed([q.question])[0]
            top=sorted(range(len(units)),key=lambda i:cosine(qe,uembs[i]),reverse=True)[:args.top_k]
            ctx="\n".join(units[i] for i in top)
            fa=chat(args.model,[{"role":"system","content":"Answer WHEN, using only these dated memories. Resolve relative dates. Give the date."},
                                {"role":"user","content":f"Memories:\n{ctx}\n\nQuestion: {q.question}"}],max_out=600)
            # RLM
            ra=chat(args.model,[{"role":"system","content":"Answer WHEN using this reconstructed dated timeline. Give the date."},
                                {"role":"user","content":f"Timeline:\n{tl_text}\n\nQuestion: {q.question}"}],max_out=600)
            fok=judge(args.model,q.question,q.answer,fa); rok=judge(args.model,q.question,q.answer,ra)
            fr={"qid":q.qid,"q":q.question,"gold":q.answer,"a":fa,"ok":fok}
            rr={"qid":q.qid,"q":q.question,"gold":q.answer,"a":ra,"ok":rok}
            res["flat"].append(fr); res["rlm"].append(rr)
            f.write(json.dumps({"qid":q.qid,"f":fr,"r":rr})+"\n")
            print(f"Q: {q.question[:58]}  gold: {q.answer}")
            print(f"  FLAT [{'OK' if fok else 'X '}] {fa[:60]}")
            print(f"  RLM  [{'OK' if rok else 'X '}] {ra[:60]}")
    fs=sum(x['ok'] for x in res['flat'])/len(temporal); rs=sum(x['ok'] for x in res['rlm'])/len(temporal)
    (OUT/"locomo_temporal_eval.json").write_text(json.dumps({"flat_acc":fs,"rlm_acc":rs,"n":len(temporal),"timeline":timeline,"results":res},indent=2))
    print(f"\n==== REAL LoCoMo temporal questions (n={len(temporal)}, {args.model}) ====")
    print(f"FLAT vector store:     {fs*100:.0f}%")
    print(f"RLM reconstruction:    {rs*100:.0f}%   ({'+' if rs>=fs else ''}{(rs-fs)*100:.0f}pp)")

if __name__=="__main__": main()
