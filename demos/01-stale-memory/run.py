"""Demo 01 — Stale memory: similarity search returns yesterday's truth.

THE CLAIM
    A flat vector store retrieves by similarity and keeps no time order, so once a
    fact changes it cannot answer "what was true as of month T" — it confidently
    returns the most similar (usually newest or most-repeated) value. A reader that
    first RECONSTRUCTS a timeline of state changes from the raw log, then resolves
    the value at-or-before T, answers both "now" and "as of T" questions.

THE SETUP
    One synthetic 9-event timestamped life-log where four attributes change over a
    year (location, diet, relationship, job). 10 questions, half asked "as of" a
    month BEFORE a change, half asked about the present.

    FLAT reader  : embed each event, retrieve top-4 by cosine, answer.
    REPLAY reader: (map) extract (attribute -> value @ month) transitions from log
                   windows; (reduce) resolve latest value at-or-before T.

SCORING
    Deterministic regex match on the key fact — no LLM judge. LLM judges flip
    verdicts between identical runs (we observed it on this very demo), which is
    the same failure mode that broke LoCoMo's answer key. Anyone who runs this
    gets the same score for the same answers.

Run:    python demos/01-stale-memory/run.py     (from repo root; needs OPENAI_API_KEY or .env)
Writes: demos/01-stale-memory/results.json
Cost:   ~$0.01 (gpt-4o-mini + text-embedding-3-small)
"""
from __future__ import annotations
import json, math, os, re
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent.parent
for line in (REPO/".env").read_text().splitlines() if (REPO/".env").exists() else []:
    if line.strip() and not line.startswith("#") and "=" in line:
        k,v=line.split("=",1); os.environ.setdefault(k.strip(), v.strip().strip('"').strip("'"))
from openai import OpenAI
client=OpenAI(); EMB="text-embedding-3-small"; GEN=os.environ.get("DEMO_MODEL","gpt-5-mini")

# A timestamped life-log (month index 1..12 of one year).
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

# (question, as-of month T, accept regex, reject regex, gold note)
QUESTIONS = [
    ("Where did the user live?",              5,  r"\bboston\b",                 r"\b(nyc|new york)\b",     "Boston (moved to NYC in Aug)"),
    ("Where does the user live?",             12, r"\b(nyc|new york)\b",         r"\bboston\b",             "NYC"),
    ("What was the user's diet?",             3,  r"\bvegetarian\b",             r"\b(pescatarian|fish)\b", "vegetarian (went pescatarian in Jul)"),
    ("What is the user's diet?",              12, r"\b(pescatarian|fish)\b",     r"\bvegetarian\b",         "pescatarian"),
    ("Was the user in a relationship?",       7,  r"\b(yes|dating|with alex)\b", r"\b(no|not|broke)\b",     "yes, with Alex (broke up in Sep)"),
    ("Is the user dating Alex?",              12, r"\b(no|not|broke)\b",         r"^yes\b",                 "no, they broke up in Sep"),
    ("Where did the user work?",              6,  r"\b(startup|fintech)\b",      r"\bbigger\b",             "the fintech startup (changed jobs in Aug)"),
    ("What is the user's job level?",         12, r"\b(senior|promoted)\b",      r"",                       "senior engineer (promoted in Nov)"),
    ("Was the user into rock climbing?",      6,  r"\b(no|not)\b",               r"^yes\b",                 "no (started in Oct)"),
    ("What hobby is the user into?",          12, r"\bclimbing\b",               r"",                       "rock climbing"),
]
MONTH={1:"Jan",2:"Feb",3:"Mar",4:"Apr",5:"May",6:"Jun",7:"Jul",8:"Aug",9:"Sep",10:"Oct",11:"Nov",12:"Dec"}
ANSWER_STYLE="Answer in one sentence. If the question is yes/no, start with 'Yes' or 'No'."

def chat(msgs, **kw):
    kw.setdefault("max_completion_tokens", kw.pop("max_tokens", 600))
    return client.chat.completions.create(model=GEN,messages=msgs,reasoning_effort="minimal",**kw).choices[0].message.content
def embed(texts): return [d.embedding for d in client.embeddings.create(model=EMB,input=texts).data]
def cosine(a,b):
    dot=sum(x*y for x,y in zip(a,b)); na=math.sqrt(sum(x*x for x in a)); nb=math.sqrt(sum(y*y for y in b)); return dot/(na*nb+1e-9)

def score(pred, accept, reject):
    p=pred.lower()
    if reject and re.search(reject,p): return False
    return bool(re.search(accept,p))

# ---------- FLAT reader: top-k cosine similarity, no time order ----------
def flat_answer(q, T, embs):
    qe=embed([q])[0]
    order=sorted(range(len(LOG)),key=lambda i:cosine(qe,embs[i]),reverse=True)[:4]
    ctx="\n".join(f"- {LOG[i][1]}" for i in order)
    return chat([{"role":"system","content":f"Use only these memories. No timestamps are given. {ANSWER_STYLE}"},
                 {"role":"user","content":f"Memories:\n{ctx}\n\nQuestion (as of {MONTH[T]}): {q}"}],max_tokens=60)

# ---------- REPLAY reader: reconstruct a timeline, resolve value at-or-before T ----------
def reconstruct():
    """map: split the log into windows, extract structured transitions from each."""
    windows=[LOG[0:3],LOG[3:6],LOG[6:9]]
    timeline=[]
    for w in windows:
        block="\n".join(f"[{MONTH[m]}] {t}" for m,t in w)
        raw=chat([{"role":"system","content":(
            "Extract durable state changes as JSON: "
            '{"transitions":[{"attribute":"location|diet|relationship|job|hobby","value":"...","month":1-12}]}. '
            "Only include facts that set or change a state.")},
            {"role":"user","content":block}],response_format={"type":"json_object"},max_tokens=300)
        try: timeline+=json.loads(raw).get("transitions",[])
        except: pass
    return sorted(timeline,key=lambda x:x.get("month",0))

def replay_answer(q,T,timeline):
    """reduce: resolve the state at-or-before T from the reconstructed timeline."""
    asof=[t for t in timeline if t.get("month",99)<=T]
    tl="\n".join(f"[{MONTH[t['month']]}] {t['attribute']} = {t['value']}" for t in asof) or "(nothing before T)"
    return chat([{"role":"system","content":(
        "You have a timeline of state changes known as of the given month. Answer AS OF that month "
        f"using the most recent value at-or-before it. {ANSWER_STYLE}")},
        {"role":"user","content":f"State known as of {MONTH[T]}:\n{tl}\n\nQuestion (as of {MONTH[T]}): {q}"}],max_tokens=60)

def main():
    embs=embed([t for _,t in LOG])
    timeline=reconstruct()
    print("RECONSTRUCTED TIMELINE:")
    for t in timeline: print(f"  [{MONTH.get(t.get('month',0),'?')}] {t.get('attribute')} = {t.get('value')}")
    out={"flat":[],"replay":[],"timeline":timeline}
    for q,T,acc,rej,gold in QUESTIONS:
        fa=flat_answer(q,T,embs); ra=replay_answer(q,T,timeline)
        fok=score(fa,acc,rej); rok=score(ra,acc,rej)
        out["flat"].append({"q":q,"T":MONTH[T],"a":fa,"ok":fok,"gold":gold})
        out["replay"].append({"q":q,"T":MONTH[T],"a":ra,"ok":rok,"gold":gold})
        print(f"\nQ ({MONTH[T]}): {q}   [gold: {gold}]\n  FLAT   [{'OK' if fok else 'X '}] {fa}\n  REPLAY [{'OK' if rok else 'X '}] {ra}")
    n=len(QUESTIONS)
    fs=sum(x['ok'] for x in out['flat'])/n; rs=sum(x['ok'] for x in out['replay'])/n
    past=[i for i,(q,T,a,r,g) in enumerate(QUESTIONS) if T<12]
    fp=sum(out['flat'][i]['ok'] for i in past)/len(past); rp=sum(out['replay'][i]['ok'] for i in past)/len(past)
    out["summary"]={"n":n,"flat_acc":fs,"replay_acc":rs,
                    "flat_asof_past":fp,"replay_asof_past":rp,
                    "scoring":"deterministic regex, no LLM judge"}
    (REPO/"demos/01-stale-memory/results.json").write_text(json.dumps(out,indent=2))
    print(f"\n==== RESULT (n={n}, deterministic scoring) ====")
    print(f"overall     — FLAT: {fs*100:.0f}%   REPLAY: {rs*100:.0f}%  ({(rs-fs)*100:+.0f}pp)")
    print(f"as-of past  — FLAT: {fp*100:.0f}%   REPLAY: {rp*100:.0f}%  ({(rp-fp)*100:+.0f}pp)")

if __name__=="__main__": main()
