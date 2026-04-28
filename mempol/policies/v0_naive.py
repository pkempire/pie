"""v0: single-shot dense+BM25 hybrid retrieve, then answer. The floor."""
from __future__ import annotations

from .. import llm, config
from ..backends.base import Backend
from .base import ReadPolicy, Step, Trace


_ANSWER_SYS = (
    "You answer questions using ONLY the provided conversation excerpts. "
    "Be concise. If the excerpts don't contain the answer, reply 'not in context'."
)


def _format_context(hits) -> str:
    lines = []
    for h in hits:
        m = h.unit.metadata
        lines.append(f"[{m.get('dia_id')} | {m.get('session_date','')}] {m.get('speaker','?')}: {h.unit.text}")
    return "\n".join(lines)


def answer_with_context(question: str, hits) -> str:
    ctx = _format_context(hits)
    msgs = [
        {"role": "system", "content": _ANSWER_SYS},
        {"role": "user", "content": f"Conversation excerpts:\n{ctx}\n\nQuestion: {question}\nAnswer:"},
    ]
    return llm.chat(msgs, model=config.ANSWER_MODEL).strip()


class NaivePolicy(ReadPolicy):
    name = "v0_naive"

    def __init__(self, k: int = 10) -> None:
        self.k = k

    def run(self, question: str, backend: Backend) -> Trace:
        t = Trace(qid="", question=question, backend=backend.name, policy=self.name)
        hits = backend.retrieve(question, k=self.k, source="hybrid")
        t.steps.append(Step("retrieve", {"k": self.k, "source": "hybrid"}, f"got {len(hits)} hits"))
        t.n_retrievals += 1
        t.final_hits = hits
        t.answer = answer_with_context(question, hits)
        t.steps.append(Step("stop_and_answer", {}, t.answer[:120]))
        return t
