"""Counterfactual necessity miner — principled replacement for the deleted regex miner.

Goal: produce (antecedent_turn, future_query, gold_answer, necessity_score) triples
for training the write policy. The criterion for "the antecedent should be stored"
is functional, not lexical: if removing the turn from context measurably reduces
an LLM's ability to answer a synthetic future query, the turn is necessary.

No hand-coded patterns. Every decision is grounded in model behaviour.

Pipeline per candidate antecedent turn t in conversation C:

  1. SYNTHESIZE: ask an LLM to generate a plausible later user question Q
     that requires the content of t. Reject if t is too generic.
  2. DERIVE GOLD: ask the LLM for the gold answer to Q given t.
  3. ABLATE: build C^- = C \ {t}.
  4. ANSWER BOTH: run an answer-LLM on (Q, C^+) and (Q, C^-).
  5. JUDGE: score s+ and s- against gold.
  6. NECESSITY: Δ = s+ − s−. Keep iff Δ ≥ τ_necessity (default 0.5).

Cost: ~4 LLM calls per candidate at gpt-4o-mini (~$0.004). For 5K positive
triples assuming ~50% acceptance, expect ~10K candidates, ~$40 total.

Usage:
  python -m mempol.data.necessity_miner \
      --conversations-json ~/Documents/pie22/conversations.json \
      --out mempol/data/necessity_triples.jsonl \
      --max-pairs 1000
"""
from __future__ import annotations
import argparse
import json
from dataclasses import dataclass
from pathlib import Path

from .. import llm, config
from ..eval.judge import judge


# ── ChatGPT export parsing (no regex over content; just structural). ──
@dataclass
class _Turn:
    idx: int
    role: str        # "user" | "assistant" | "system"
    text: str
    msg_id: str
    conv_id: str


def _flatten_chatgpt_export(path: Path) -> list[_Turn]:
    """Robust to ChatGPT's two common export shapes (mapping / messages).
    Structural only — does not read content."""
    raw = json.loads(Path(path).read_text())
    out: list[_Turn] = []
    convs = raw if isinstance(raw, list) else raw.get("conversations", [])
    for ci, conv in enumerate(convs):
        cid = str(conv.get("id") or conv.get("conversation_id") or f"c{ci}")
        items: list[tuple[float, str, str, str]] = []
        if isinstance(conv.get("mapping"), dict):
            for nid, node in conv["mapping"].items():
                msg = node.get("message")
                if not msg or not msg.get("content"):
                    continue
                ct = msg["content"]
                parts = ct.get("parts") if isinstance(ct, dict) else None
                if not parts:
                    continue
                text = "\n".join(str(p) for p in parts if isinstance(p, str)).strip()
                if not text:
                    continue
                role = msg.get("author", {}).get("role", "")
                ts = float(msg.get("create_time") or 0)
                items.append((ts, role, text, msg.get("id", nid)))
        elif isinstance(conv.get("messages"), list):
            for m in conv["messages"]:
                text = (m.get("content") or "").strip()
                if not text:
                    continue
                ts = float(m.get("create_time") or m.get("timestamp") or 0)
                items.append((ts, m.get("role", ""), text, str(m.get("id", ""))))
        items.sort(key=lambda x: x[0])
        for ts, role, text, mid in items:
            out.append(_Turn(idx=len(out), role=role, text=text, msg_id=mid, conv_id=cid))
    return out


# ── LLM steps (all calibrated by behaviour, not patterns). ──
_GENERATE_SYS = (
    "You see one turn from a conversation between a user and an AI assistant. "
    "Decide whether this turn carries specific, durable information that a user "
    "could plausibly need to recall later (a fact, decision, plan, preference, "
    "result, name, time, or constraint). If yes, write a single later user "
    "question Q a real user might ask weeks/months later that would require "
    "this turn to answer correctly, and the gold answer derivable from this "
    "turn. If the turn is too generic, conversational, or lacks specific "
    "information, set is_specific=false. Strict JSON only."
)
_GENERATE_USR = (
    "Turn role: {role}\n"
    "Turn text: {text}\n\n"
    "Return JSON with keys: is_specific (bool), future_query (string|null), "
    "gold_answer (string|null), reason (one sentence)."
)


def _generate_future_query(turn: _Turn) -> dict | None:
    raw = llm.chat(
        [
            {"role": "system", "content": _GENERATE_SYS},
            {"role": "user", "content": _GENERATE_USR.format(role=turn.role, text=turn.text[:1500])},
        ],
        model=config.REFORMULATE_MODEL,
        json_mode=True,
    )
    try:
        obj = json.loads(raw)
    except Exception:
        return None
    if not obj.get("is_specific"):
        return None
    q = obj.get("future_query")
    g = obj.get("gold_answer")
    if not q or not g:
        return None
    return {"future_query": str(q), "gold": str(g), "reason": str(obj.get("reason", ""))[:200]}


_ANSWER_SYS = (
    "You are a careful assistant answering a user's question using ONLY the "
    "provided conversation history. Be concise. If the history doesn't support "
    "an answer, reply: 'not in context'."
)


def _answer_with_context(future_query: str, history: list[_Turn]) -> str:
    fmt = "\n".join(f"[{i}] {t.role}: {t.text[:400]}" for i, t in enumerate(history))
    msgs = [
        {"role": "system", "content": _ANSWER_SYS},
        {"role": "user", "content": f"Conversation history:\n{fmt}\n\nQuestion: {future_query}\nAnswer:"},
    ]
    return llm.chat(msgs, model=config.ANSWER_MODEL).strip()


# ── Necessity test. ──
@dataclass
class NecessityResult:
    necessity: float
    score_with: float
    score_without: float
    answer_with: str
    answer_without: str


def measure_necessity(
    target: _Turn,
    context_window: list[_Turn],
    future_query: str,
    gold: str,
) -> NecessityResult:
    history_with = list(context_window)
    history_without = [t for t in context_window if t.msg_id != target.msg_id]
    a_plus = _answer_with_context(future_query, history_with)
    a_minus = _answer_with_context(future_query, history_without)
    s_plus, _ = judge(future_query, gold, a_plus)
    s_minus, _ = judge(future_query, gold, a_minus)
    return NecessityResult(
        necessity=s_plus - s_minus,
        score_with=s_plus,
        score_without=s_minus,
        answer_with=a_plus,
        answer_without=a_minus,
    )


# ── Mining loop. ──
def mine(
    conversations_json: Path,
    out_path: Path,
    max_pairs: int = 1000,
    window_back: int = 30,
    window_forward: int = 5,
    necessity_threshold: float = 0.5,
    candidate_roles: tuple[str, ...] = ("assistant", "user"),
    sample_every: int = 1,
) -> dict:
    """Walk the export and mine necessity triples. No pattern matching used."""
    turns = _flatten_chatgpt_export(conversations_json)
    print(f"[necessity_miner] flattened {len(turns)} turns from export")

    out_path.parent.mkdir(parents=True, exist_ok=True)
    n_kept = 0
    n_rejected_generate = 0
    n_rejected_necessity = 0
    n_scanned = 0

    with out_path.open("w") as f:
        for i, t in enumerate(turns):
            if t.role not in candidate_roles:
                continue
            if i % sample_every != 0:
                continue
            n_scanned += 1
            if n_kept >= max_pairs:
                break

            # 1+2. synthesize Q + gold. Skip if turn is too generic.
            qg = _generate_future_query(t)
            if qg is None:
                n_rejected_generate += 1
                continue

            # 3+4+5. counterfactual answer
            ctx_lo = max(0, t.idx - window_back)
            ctx_hi = min(len(turns), t.idx + window_forward + 1)
            window = [u for u in turns[ctx_lo:ctx_hi] if u.conv_id == t.conv_id]
            if t not in window:
                window.append(t)
            window.sort(key=lambda u: u.idx)

            res = measure_necessity(t, window, qg["future_query"], qg["gold"])
            if res.necessity < necessity_threshold:
                n_rejected_necessity += 1
                continue

            f.write(json.dumps({
                "antecedent_msg_id": t.msg_id,
                "antecedent_role": t.role,
                "antecedent_text": t.text[:2000],
                "conv_id": t.conv_id,
                "future_query": qg["future_query"],
                "gold": qg["gold"],
                "necessity": res.necessity,
                "score_with": res.score_with,
                "score_without": res.score_without,
                "answer_with": res.answer_with[:500],
                "answer_without": res.answer_without[:500],
                "generator_reason": qg["reason"],
            }) + "\n")
            n_kept += 1

            if n_scanned % 25 == 0:
                print(f"  scanned={n_scanned} kept={n_kept} "
                      f"rej_generate={n_rejected_generate} rej_necessity={n_rejected_necessity}")

    stats = {
        "scanned": n_scanned,
        "kept": n_kept,
        "rejected_generate": n_rejected_generate,
        "rejected_necessity": n_rejected_necessity,
        "out_path": str(out_path),
    }
    print(f"[necessity_miner] done: {json.dumps(stats)}")
    return stats


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--conversations-json", required=True, type=Path)
    ap.add_argument("--out", default=Path("mempol/data/necessity_triples.jsonl"), type=Path)
    ap.add_argument("--max-pairs", default=1000, type=int)
    ap.add_argument("--window-back", default=30, type=int)
    ap.add_argument("--window-forward", default=5, type=int)
    ap.add_argument("--necessity-threshold", default=0.5, type=float)
    ap.add_argument("--sample-every", default=1, type=int,
                    help="sample 1 of every N candidate turns to control cost")
    args = ap.parse_args()
    mine(
        conversations_json=args.conversations_json,
        out_path=args.out,
        max_pairs=args.max_pairs,
        window_back=args.window_back,
        window_forward=args.window_forward,
        necessity_threshold=args.necessity_threshold,
        sample_every=args.sample_every,
    )


if __name__ == "__main__":
    main()
