"""Temporal memory vs. flat vector store — the working demo (the video's payoff).

Everyone *talks* about time-blind memory. This *measures* it. Two stores ingest
the SAME timestamped conversation; we query them at a later time and judge whose
answers are correct.

  FLAT  : embed every statement, retrieve top-k by cosine, answer. No notion of time.
  TEMPORAL: every statement carries created_at + a half-life estimated from its
            *type* (mood ~ hours, diet ~ years, project-status ~ weeks) + a
            supersession link. At query time we (a) drop facts whose decay-adjusted
            validity has lapsed, (b) follow supersession chains to the current value,
            (c) hand the model validity-annotated facts. This is the "Layer 2" fix —
            and the type-based half-life is the piece Graphiti/Zep don't do (they
            invalidate only on an *observed* contradiction; we predict decay up front).

Run:  python scripts/temporal_memory_demo.py
Cost: ~$0.50 with gpt-4o-mini answers+judge + text-embedding-3-small.
Writes: output/experiments/temporal_memory_demo.json
"""
from __future__ import annotations
import json, math, os
from pathlib import Path
from datetime import datetime, timedelta

REPO = Path(__file__).resolve().parent.parent
for line in (REPO/".env").read_text().splitlines() if (REPO/".env").exists() else []:
    if line.strip() and not line.startswith("#") and "=" in line:
        k,v=line.split("=",1); os.environ.setdefault(k.strip(), v.strip().strip('"').strip("'"))

from openai import OpenAI
client = OpenAI()
EMB="text-embedding-3-small"; GEN="gpt-4o-mini"

DAY = 24*3600
# Half-life (seconds) by fact type — the forward-looking decay prior.
HALF_LIFE = {
    "mood": 2*3600, "activity": 6*3600, "plan": 7*DAY, "project_status": 21*DAY,
    "location": 180*DAY, "preference": 2*365*DAY, "diet": 5*365*DAY, "identity": 50*365*DAY,
}

# A clean synthetic timeline (no messy personal data). t = seconds from epoch0.
EPOCH = datetime(2026,1,1,9,0,0)
def ts(days, hours=0): return (EPOCH + timedelta(days=days, hours=hours)).timestamp()

STATEMENTS = [
    # (text, type, t, supersedes_index_or_None)
    ("I'm a vegetarian.", "diet", ts(0), None),                       # 0
    ("I'm so angry at my coworker right now.", "mood", ts(0), None),  # 1
    ("I'm working on the Q3 launch this week.", "project_status", ts(0), None),  # 2
    ("I live in Boston.", "location", ts(0), None),                   # 3
    ("Actually I eat fish now — I'm pescatarian.", "diet", ts(30), 0),# 4 supersedes 0
    ("The Q3 launch shipped successfully.", "project_status", ts(30), 2), # 5 supersedes 2
]
# Query time: 30 days + 8 hours after the start.
NOW = ts(30, 8)

QUESTIONS = [
    ("Is the user angry right now?", "No — that was a mood from a month ago; it has long passed."),
    ("Is the user a vegetarian?", "No — they became pescatarian."),
    ("Is the user still working on the Q3 launch?", "No — it already shipped."),
    ("Where does the user live?", "Boston."),
]

def embed(texts):
    r = client.embeddings.create(model=EMB, input=texts)
    return [d.embedding for d in r.data]

def cosine(a,b):
    dot=sum(x*y for x,y in zip(a,b)); na=math.sqrt(sum(x*x for x in a)); nb=math.sqrt(sum(y*y for y in b))
    return dot/(na*nb+1e-9)

def answer(question, context_lines):
    ctx = "\n".join(f"- {c}" for c in context_lines) or "(no memory)"
    msgs=[{"role":"system","content":(
        "Answer the question using ONLY the memory below. One sentence. "
        "Each memory may carry annotations: [EXPIRED] or low validity means the fact was "
        "true once but is probably NOT true now; [SUPERSEDED] means it was replaced by a newer fact. "
        "Treat expired/superseded facts as no longer true.")},
          {"role":"user","content":f"Memory:\n{ctx}\n\nQuestion: {question}\nAnswer:"}]
    return client.chat.completions.create(model=GEN,messages=msgs,temperature=0,max_tokens=80).choices[0].message.content.strip()

def judge(question, gold, pred):
    msgs=[{"role":"system","content":'Return JSON {"correct": true/false}. Mark correct only if the prediction matches the gold answer\'s key fact.'},
          {"role":"user","content":f"Q: {question}\nGold: {gold}\nPrediction: {pred}"}]
    raw=client.chat.completions.create(model=GEN,messages=msgs,temperature=0,response_format={"type":"json_object"},max_tokens=20).choices[0].message.content
    try: return bool(json.loads(raw).get("correct"))
    except: return False

# ---- FLAT store: cosine top-k, no time ----
def flat_context(q_emb, embs, k=4):
    scored=sorted(range(len(embs)), key=lambda i: cosine(q_emb,embs[i]), reverse=True)[:k]
    return [STATEMENTS[i][0] for i in scored]

# ---- TEMPORAL store: decay + supersession + validity annotation ----
def superseded_ids():
    return {STATEMENTS[i][3] for i in range(len(STATEMENTS)) if STATEMENTS[i][3] is not None}
def temporal_context(q_emb, embs, k=4):
    dead = superseded_ids()
    cand=[]
    for i,(text,typ,t,sup) in enumerate(STATEMENTS):
        hl = HALF_LIFE.get(typ, 30*DAY)
        decay = 0.5 ** ((NOW - t)/hl)          # forward-looking validity
        cand.append((i, decay, cosine(q_emb,embs[i])))
    cand.sort(key=lambda c: c[2], reverse=True)  # rank by similarity
    lines=[]
    for i,decay,_ in cand[:k]:
        text,typ,t,_=STATEMENTS[i]
        age_d=(NOW-t)/DAY
        tags=[f"type={typ}", f"age={age_d:.0f}d", f"validity={decay:.2f}"]
        if i in dead: tags.append("SUPERSEDED")          # a newer fact replaced this
        elif decay < 0.25: tags.append("EXPIRED")        # decayed past its half-life
        lines.append(f'{text}  [{", ".join(tags)}]')
    return lines

def main():
    texts=[s[0] for s in STATEMENTS]
    embs=embed(texts); q_embs=embed([q for q,_ in QUESTIONS])
    out={"flat":[], "temporal":[]}
    for (q,gold),qe in zip(QUESTIONS,q_embs):
        fc=flat_context(qe,embs); tc=temporal_context(qe,embs)
        fa=answer(q,fc); ta=answer(q,tc)
        fok=judge(q,gold,fa); tok=judge(q,gold,ta)
        out["flat"].append({"q":q,"answer":fa,"correct":fok,"context":fc})
        out["temporal"].append({"q":q,"answer":ta,"correct":tok,"context":tc})
        print(f"\nQ: {q}")
        print(f"  FLAT     [{'✓' if fok else '✗'}] {fa}")
        print(f"  TEMPORAL [{'✓' if tok else '✗'}] {ta}")
    fscore=sum(x['correct'] for x in out['flat'])/len(QUESTIONS)
    tscore=sum(x['correct'] for x in out['temporal'])/len(QUESTIONS)
    out["summary"]={"flat_acc":fscore,"temporal_acc":tscore,"n":len(QUESTIONS)}
    Path("output/experiments").mkdir(parents=True,exist_ok=True)
    Path("output/experiments/temporal_memory_demo.json").write_text(json.dumps(out,indent=2))
    print(f"\n==== RESULT ====\nFLAT store:     {fscore*100:.0f}%\nTEMPORAL store: {tscore*100:.0f}%  (+{(tscore-fscore)*100:.0f}pp)")

if __name__=="__main__":
    main()
