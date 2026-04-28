"""LLM-as-judge for free-form QA. Same protocol as the LoCoMo paper."""
from __future__ import annotations
import json

from .. import llm, config


_JUDGE_SYS = (
    "You are a strict judge of question-answering. Compare the predicted answer "
    "to the gold answer and decide if the prediction conveys the same factual "
    "content. Minor wording differences are OK. Return JSON with keys 'score' "
    "(1.0 fully correct, 0.5 partial, 0.0 wrong) and 'reason' (one sentence)."
)
_JUDGE_USR = (
    "Question: {question}\nGold answer: {gold}\nPredicted answer: {pred}\n\n"
    "Return JSON only."
)


def judge(question: str, gold: str, pred: str) -> tuple[float, str]:
    p = (pred or "").strip().lower()
    if not p or p.startswith("not in context") or p.startswith("error"):
        return 0.0, "no answer"
    msgs = [
        {"role": "system", "content": _JUDGE_SYS},
        {"role": "user", "content": _JUDGE_USR.format(question=question, gold=gold, pred=pred)},
    ]
    raw = llm.chat(msgs, model=config.JUDGE_MODEL, json_mode=True)
    try:
        obj = json.loads(raw)
        s = float(obj.get("score", 0.0))
        # Bucketing follows the LongMemEval paper protocol (Wu et al. 2024,
        # §3.2): 1.0 = fully correct, 0.5 = partial, 0.0 = wrong.
        if s >= 0.75:
            s = 1.0
        elif s >= 0.25:
            s = 0.5
        else:
            s = 0.0
        return s, str(obj.get("reason", ""))[:200]
    except Exception as e:
        # FIX (audit #4): surface judge parse failures to logs instead of
        # silently returning 0. A repeated stream of judge_err messages
        # usually means the model is rate-limited or returning markdown.
        import logging
        logging.getLogger("mempol.judge").warning(
            "judge JSON parse failed (model=%s): %s; raw=%r",
            config.JUDGE_MODEL, e, str(raw)[:120],
        )
        return 0.0, f"judge_err:{e}"
