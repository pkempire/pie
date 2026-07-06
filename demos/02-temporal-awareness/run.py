"""TicToc smoke: does the model FEEL elapsed time? (information != awareness)

Task: given a timestamped tool-use conversation, decide CALL-TOOL (refresh) vs
ANSWER-DIRECTLY, scored against human preference labels. Three conditions:
  A  blind   : timestamps stripped from the transcript
  B  raw     : ISO timestamps present (what TicToc tests -> still <65%)
  C  state   : timestamps present + a computed freshness line (age of the data as of now)
Hypothesis: A ~= B (raw timestamps don't help -> blindness), C > B (time-as-state helps).
"""
import json, os, re, random
from pathlib import Path
from datetime import datetime

REPO = Path("/Users/parthkocheta/personal-intelligence-system")
for line in (REPO/".env").read_text().splitlines() if (REPO/".env").exists() else []:
    if line.strip() and not line.startswith("#") and "=" in line:
        k,v=line.split("=",1); os.environ.setdefault(k.strip(), v.strip().strip('"').strip("'"))
from openai import OpenAI
cli=OpenAI(); MODEL="gpt-5-mini"

# Get the data:  git clone https://github.com/chengez/TicToc  (or set TICTOC_DIR)
TICTOC=Path(os.environ.get("TICTOC_DIR", Path(__file__).parent/"TicToc"))
DATA=TICTOC/"merged_fully_labeled_data_test.json"
if not DATA.exists():
    raise SystemExit(f"TicToc data not found at {DATA}. Run: git clone https://github.com/chengez/TicToc (into {TICTOC.parent}) or set TICTOC_DIR.")
samples=json.load(open(DATA))
random.seed(7)

# binary human label
def human(s):
    p=s["preference"]
    if p in ("tool","lean_tool"): return "TOOL"
    if p in ("direct","lean_direct"): return "DIRECT"
    return None

pool=[s for s in samples if human(s)]
# balance TOOL/DIRECT, spread across scenarios
random.shuffle(pool)
tool=[s for s in pool if human(s)=="TOOL"][:18]
direct=[s for s in pool if human(s)=="DIRECT"][:18]
batch=tool+direct; random.shuffle(batch)

def parse_t(t):
    try: return datetime.fromisoformat(t.replace("Z","+00:00"))
    except: return None

def fmt_history(s, show_time):
    lines=[]
    for m in s["history"]:
        role=m["role"]; t=m.get("time","")
        if role=="system": continue
        stamp=f"[{t}] " if (show_time and t) else ""
        if role=="user": lines.append(f"{stamp}User: {m['content']}")
        elif role=="tool": lines.append(f"{stamp}Tool result: {m['content']}")
        elif role=="assistant":
            if m.get("tool_calls"):
                tc=m["tool_calls"][0]["function"]
                lines.append(f"{stamp}Assistant called {tc['name']}({tc['arguments']})")
            elif m.get("content"): lines.append(f"{stamp}Assistant: {m['content']}")
    return "\n".join(lines)

def freshness(s):
    # age of the last tool result as of the final (now) message
    times=[parse_t(m.get("time","")) for m in s["history"] if m.get("time")]
    times=[t for t in times if t]
    now=times[-1] if times else None
    last_tool=None
    for m in s["history"]:
        if m["role"]=="tool" and m.get("time"):
            last_tool=parse_t(m["time"])
    if now and last_tool:
        secs=(now-last_tool).total_seconds()
        if secs<90: age=f"{secs:.0f} seconds"
        elif secs<5400: age=f"{secs/60:.0f} minutes"
        elif secs<172800: age=f"{secs/3600:.1f} hours"
        else: age=f"{secs/86400:.1f} days"
        return f"Time context: it is now {now.isoformat()}. The data you retrieved is {age} old."
    return "Time context: unavailable."

TOOLS=lambda s: "\n".join(f"- {f['function']['name']}: {f['function'].get('description','')}" for f in s["function"])

def decide(s, cond):
    show = cond!="blind"
    hist=fmt_history(s, show)
    extra = ("\n"+freshness(s)) if cond=="state" else ""
    sys=("You are the assistant. Choose your next action to serve the user's latest message well: either "
         "TOOL (call a tool to fetch up-to-date data) or DIRECT (answer from information already available). "
         "Weigh how quickly this kind of information goes out of date against how long it has been since you got it. "
         'Respond as JSON: {"action":"TOOL"|"DIRECT"}.')
    usr=f"Available tools:\n{TOOLS(s)}\n\nConversation so far:\n{hist}{extra}\n\nDecision JSON:"
    r=cli.chat.completions.create(model=MODEL,messages=[{"role":"system","content":sys},{"role":"user","content":usr}],
                                  reasoning_effort="low",response_format={"type":"json_object"},max_completion_tokens=800)
    out=(r.choices[0].message.content or "").upper()
    return "TOOL" if '"ACTION": "TOOL"' in out or '"ACTION":"TOOL"' in out else "DIRECT"

conds=["blind","raw","state"]
score={c:0 for c in conds}; n=0
for s in batch:
    h=human(s); n+=1
    row=f"[{human(s):6}] {s['id'][:34]:34}"
    for c in conds:
        try: d=decide(s,c)
        except Exception as e: d="DIRECT"
        ok=(d==h); score[c]+=ok
        row+=f"  {c}:{'OK' if ok else '..'}"
    print(row)
print("\n==== TicToc alignment (n=%d, gpt-5-mini) ===="%n)
for c in conds:
    print(f"  {c:6}: {score[c]/n*100:5.1f}%  aligned with human preference")
print("\nblind vs raw  = does showing timestamps help? (TicToc: barely)")
print("raw   vs state= does computing freshness-as-state help? (our hypothesis)")
out={"model":MODEL,"n":n,"seed":7,"reasoning":"low",
     "alignment":{c:round(score[c]/n,4) for c in conds},
     "note":"smoke, single seed; direction not significance"}
(Path(__file__).parent/"results.json").write_text(json.dumps(out,indent=2))
