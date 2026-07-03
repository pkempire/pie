"""Learned-critic counterfactual — the clever fix for mempol's per-op cost, made real.

mempol's per-op counterfactual costs (K_mut+1)*Q reader+judge calls PER trajectory
(verified in mempol/CODE-REVIEW-2026-06-06.md). That's the brute-force tax.

The clever alternative (cf. CCPO arXiv 2603.21563 — "counterfactual advantages without
repeated re-rollouts"): compute the EXACT per-op delta for only a *few* ops, fit a tiny
critic that predicts per-op advantage from CHEAP features (retrieval similarity, evidence
overlap, redundancy — all free, no extra rollouts), and use the critic for the rest.

This smoke isolates the mechanism: can a critic trained on k exact deltas predict the
held-out ops' per-op advantage? Ground-truth correctness here is deterministic
(required-evidence-op-in-top-k retrieval) — a stand-in for the LLM judge so we measure the
CRITIC, not judge noise. Real embeddings (OpenAI) drive the features + retrieval.

Run:  python scripts/critic_counterfactual_smoke.py
"""
from __future__ import annotations
import json, os, math, random
from pathlib import Path
import numpy as np

REPO = Path(__file__).resolve().parent.parent
for line in (REPO/".env").read_text().splitlines() if (REPO/".env").exists() else []:
    if line.strip() and not line.startswith("#") and "=" in line:
        k,v=line.split("=",1); os.environ.setdefault(k.strip(), v.strip().strip('"').strip("'"))
from openai import OpenAI
client=OpenAI(); EMB="text-embedding-3-small"; random.seed(0); np.random.seed(0)

# ---- a synthetic but realistic memory: facts (ops) + questions w/ known gold evidence ----
FACTS = [
    "The user lives in Boston.", "The user moved to Boston for a fintech job.",   # 0,1 redundant (location)
    "The user is vegetarian.",                                                    # 2 unique (diet)
    "The user is dating Alex.",                                                   # 3 unique (relationship)
    "The user's manager is named Priya.",                                         # 4 unique
    "The user uses Python daily.", "The user codes mostly in Python.",            # 5,6 redundant (language)
    "The user has a dog named Mochi.",                                            # 7 unique
    "The user enjoys rock climbing.",                                             # 8 unique
    "The user's startup is called Tensor.",                                       # 9 unique
    "The user drinks coffee every morning.",                                      # 10 noise
    "The user watched a movie last night.",                                       # 11 noise
    "The user is allergic to peanuts.",                                           # 12 unique
    "The user's sister lives in Chicago.",                                        # 13 unique
    "The user prefers tea over soda.",                                            # 14 noise
    "The user graduated from UMD.",                                               # 15 unique
]
# question -> set of fact indices that satisfy it (gold evidence; >1 = redundant)
QUESTIONS = [
    ("Where does the user live?", {0,1}),
    ("What is the user's diet?", {2}),
    ("Who is the user dating?", {3}),
    ("What language does the user code in?", {5,6}),
    ("Does the user have a pet?", {7}),
    ("What is the user allergic to?", {12}),
    ("Where did the user go to college?", {15}),
]
TOPK=3

def embed(texts):
    return np.array([d.embedding for d in client.embeddings.create(model=EMB,input=texts).data])

def retrieve_topk(q_emb, fact_embs, alive):
    sims=[(i, float(q_emb@fact_embs[i])) for i in alive]
    sims.sort(key=lambda x:x[1], reverse=True)
    return [i for i,_ in sims[:TOPK]]

def battery_acc(fact_embs, q_embs, alive):
    ok=0
    for (q,gold),qe in zip(QUESTIONS,q_embs):
        top=set(retrieve_topk(qe,fact_embs,alive))
        ok += 1 if (top & gold) else 0
    return ok/len(QUESTIONS)

def exact_delta(i, fact_embs, q_embs, base):
    alive=[j for j in range(len(FACTS)) if j!=i]
    return base - battery_acc(fact_embs, q_embs, alive)   # how much accuracy drops if op i removed

def features(i, fact_embs, q_embs):
    sims=[float(q_embs[m]@fact_embs[i]) for m in range(len(QUESTIONS))]
    # cheap features available WITHOUT any leave-one-out rollout:
    max_sim=max(sims); mean_sim=sum(sims)/len(sims)
    # is op i the *unique* top-1 evidence for some question, or redundant?
    is_top1=0; redundancy=0
    for (q,gold),qe in zip(QUESTIONS,q_embs):
        top=retrieve_topk(qe,fact_embs,list(range(len(FACTS))))
        if top and top[0]==i: is_top1=1
        if i in gold: redundancy=max(redundancy, len(gold))   # 1 = unique evidence, >1 = redundant
    evidence_member=1 if any(i in gold for _,gold in QUESTIONS) else 0
    uniqueness=1.0/redundancy if redundancy else 0.0          # unique evidence -> 1, redundant -> 0.5, none -> 0
    return [max_sim, mean_sim, float(is_top1), float(evidence_member), uniqueness]

def main():
    fact_embs=embed(FACTS); q_embs=embed([q for q,_ in QUESTIONS])
    base=battery_acc(fact_embs,q_embs,list(range(len(FACTS))))
    X=np.array([features(i,fact_embs,q_embs) for i in range(len(FACTS))])
    y=np.array([exact_delta(i,fact_embs,q_embs,base) for i in range(len(FACTS))])

    idx=list(range(len(FACTS))); random.shuffle(idx)
    k=8; train,test=idx[:k],idx[k:]                          # critic sees exact deltas for only 8 ops
    Xtr=np.c_[X[train],np.ones(len(train))]; Xte=np.c_[X[test],np.ones(len(test))]
    w,*_=np.linalg.lstsq(Xtr,y[train],rcond=None)            # tiny linear critic
    pred=Xte@w
    # correlation on held-out ops (predicted with ZERO rollouts)
    if np.std(pred)>0 and np.std(y[test])>0:
        r=float(np.corrcoef(pred,y[test])[0,1])
    else: r=float("nan")
    mae=float(np.mean(np.abs(pred-y[test])))

    print("base battery accuracy:", round(base,3))
    print("\nop | exact_delta | features[max_sim,mean_sim,is_top1,evid,uniq]")
    for i in range(len(FACTS)):
        tag="train" if i in train else "TEST"
        print(f" {i:2d} [{tag:5}] Δ={y[i]:+.3f}  {FACTS[i][:34]:34} {np.round(X[i],2)}")
    print(f"\nHeld-out per-op advantage — critic vs exact (NO rollouts used for these {len(test)} ops):")
    for i,p in zip(test,pred):
        print(f"  op{i:2d}: predicted {p:+.3f}   exact {y[i]:+.3f}")
    print(f"\n==== RESULT ====")
    print(f"critic trained on {k} exact deltas predicts {len(test)} held-out ops")
    print(f"  Pearson r = {r:.2f}   MAE = {mae:.3f}")
    n=len(FACTS); Q=len(QUESTIONS)
    print(f"cost: brute-force per-op = (K+1)*Q = {(n+1)*Q} oracle calls; "
          f"critic = {k} exact (+free features) -> ~{k}/{n} = {100*k//n}% of the rollouts")
    Path("output/experiments").mkdir(parents=True,exist_ok=True)
    Path("output/experiments/critic_counterfactual.json").write_text(json.dumps(
        {"base":base,"deltas":y.tolist(),"train":train,"test":test,"pred":pred.tolist(),
         "pearson_r":r,"mae":mae,"weights":w.tolist()},indent=2))

if __name__=="__main__": main()
