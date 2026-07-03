"""RLM-read: temporal state reconstruction vs. a flat vector store.

The genuinely new piece. A flat vector store retrieves by similarity and has no
time order, so it CANNOT answer "what was true on date T" once a fact has
changed. An RLM-style reader recurses over the raw event log to *reconstruct*
the state as of T — validity is computed on demand, never stored.

Two readers, same timestamped life-log:
  FLAT : embed each statement, retrieve top-k by cosine, answer.
  RLM  : (map) recurse over time-windows of the log, extracting structured
         (attribute -> value @ month) transitions; (reduce) for each query,
         resolve the latest value at-or-before T from the reconstructed timeline.

Run:  python scripts/rlm_temporal_reconstruction.py
Writes: output/experiments/rlm_temporal_reconstruction.json
"""
from __future__ import annotations
import json, math, os
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
for line in (REPO/".env").read_text().splitlines() if (REPO/".env").exists() else []:
    if line.strip() and not line.startswith("#") and "=" in line:
        k,v=line.split("=",1); os.environ.setdefault(k.strip(), v.strip().strip('"').strip("'"))
from openai import OpenAI
client=OpenAI(); EMB="text-embedding-3-small"; GEN="gpt-4o-mini"

# A timestamped life-log (natural language, month index 1..12 of 2025).
LOG = [
    (1,  "I just moved to Boston for a new job at a fintech startup."),
    (2,  "Loving the vegetarian cafes near my new place."),
    (4,  "Started dating someone named Alex."),
    (6,  "Work's been rough lately, thinking about leaving the startup."),
    (7,  "I actually eat fish now — went pescatarian."),
    (8,  "Big news: I moved to NYC and took a job at a bigger company."),
    (9,  "Alex and I broke up, unfortunately."),
    (10, "Been getting really into rock climbing."),
    (11, "Got promoted to senior engineer at the new job."),
]
# (question, as-of month T, gold)
QUESTIONS = [
    ("Where did the user live?",            5,  "Boston (they moved to NYC in August)."),
    ("Where does the user live?",           12, "NYC."),
    ("Was the user a vegetarian?",          3,  "Yes (they became pescatarian in July)."),
    ("What is the user's diet?",            12, "Pescatarian."),
    ("Was the user in a relationship?",     7,  "Yes, with Alex (they broke up in September)."),
    ("Is the user dating Alex?",            12, "No, they broke up in September."),
]
MONTH={1:"Jan",2:"Feb",3:"Mar",4:"Apr",5:"May",6:"Jun",7:"Jul",8:"Aug",9:"Sep",10:"Oct",11:"Nov",12:"Dec"}

def chat(msgs, **kw): return client.chat.completions.create(model=GEN,messages=msgs,temperature=0,**kw).choices[0].message.content
def embed(texts): return [d.embedding for d in client.embeddings.create(model=EMB,input=texts).data]
def cosine(a,b):
    dot=sum(x*y for x,y in zip(a,b)); na=math.sqrt(sum(x*x for x in a)); nb=math.sqrt(sum(y*y for y in b)); return dot/(na*nb+1e-9)
def judge(q,gold,pred):
    raw=chat([{"role":"system","content":'JSON {"correct":bool}. Correct iff prediction matches the gold key fact (including the time/as-of aspect).'},
              {"role":"user","content":f"Q:{q}\nGold:{gold}\nPred:{pred}"}],response_format={"type":"json_object"},max_tokens=20)
    try:return bool(json.loads(raw).get("correct"))
    except:return False

# ---------- FLAT reader ----------
def flat_answer(q, T, embs):
    qe=embed([q])[0]
    order=sorted(range(len(LOG)),key=lambda i:cosine(qe,embs[i]),reverse=True)[:4]
    ctx="\n".join(f"- {LOG[i][1]}" for i in order)
    return chat([{"role":"system","content":"Answer in one sentence using only these memories. No timestamps are given."},
                 {"role":"user","content":f"Memories:\n{ctx}\n\nQuestion (as of {MONTH[T]}): {q}"}],max_tokens=60)

# ---------- RLM reader: recurse over the log to reconstruct a timeline ----------
def rlm_reconstruct():
    """map: split the log into windows, extract structured transitions from each."""
    windows=[LOG[0:3],LOG[3:6],LOG[6:9]]
    timeline=[]
    for w in windows:
        block="\n".join(f"[{MONTH[m]}] {t}" for m,t in w)
        raw=chat([{"role":"system","content":(
            "Extract durable state changes as JSON: "
            '{"transitions":[{"attribute":"location|diet|relationship|job","value":"...","month":1-12}]}. '
            "Only include facts that set or change a state.")},
            {"role":"user","content":block}],response_format={"type":"json_object"},max_tokens=300)
        try: timeline+=json.loads(raw).get("transitions",[])
        except: pass
    return sorted(timeline,key=lambda x:x.get("month",0))

def rlm_answer(q,T,timeline):
    """reduce: resolve the state at-or-before T from the reconstructed timeline."""
    asof=[t for t in timeline if t.get("month",99)<=T]
    tl="\n".join(f"[{MONTH[t['month']]}] {t['attribute']} = {t['value']}" for t in asof) or "(nothing before T)"
    full="\n".join(f"[{MONTH[t['month']]}] {t['attribute']} = {t['value']}" for t in timeline)
    return chat([{"role":"system","content":(
        "You have a reconstructed timeline of state changes. Answer the question AS OF the given month: "
        "use the most recent value at-or-before that month. One sentence.")},
        {"role":"user","content":f"Full timeline:\n{full}\n\nState known as of {MONTH[T]}:\n{tl}\n\nQuestion (as of {MONTH[T]}): {q}"}],max_tokens=60)

def main():
    embs=embed([t for _,t in LOG])
    timeline=rlm_reconstruct()
    print("RECONSTRUCTED TIMELINE:")
    for t in timeline: print(f"  [{MONTH.get(t.get('month',0),'?')}] {t.get('attribute')} = {t.get('value')}")
    out={"flat":[],"rlm":[],"timeline":timeline}
    for q,T,gold in QUESTIONS:
        fa=flat_answer(q,T,embs); ra=rlm_answer(q,T,timeline)
        fok=judge(q,gold,fa); rok=judge(q,gold,ra)
        out["flat"].append({"q":q,"T":MONTH[T],"a":fa,"ok":fok})
        out["rlm"].append({"q":q,"T":MONTH[T],"a":ra,"ok":rok})
        print(f"\nQ ({MONTH[T]}): {q}\n  FLAT [{'✓' if fok else '✗'}] {fa}\n  RLM  [{'✓' if rok else '✗'}] {ra}")
    fs=sum(x['ok'] for x in out['flat'])/len(QUESTIONS); rs=sum(x['ok'] for x in out['rlm'])/len(QUESTIONS)
    out["summary"]={"flat_acc":fs,"rlm_acc":rs}
    Path("output/experiments").mkdir(parents=True,exist_ok=True)
    Path("output/experiments/rlm_temporal_reconstruction.json").write_text(json.dumps(out,indent=2))
    print(f"\n==== RESULT ====\nFLAT: {fs*100:.0f}%   RLM-reconstruct: {rs*100:.0f}%  (+{(rs-fs)*100:.0f}pp)")

if __name__=="__main__": main()
