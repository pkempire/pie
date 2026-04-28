"""Reward function — Tinker-cookbook compatible signature.

Mirrors search_tool/tools.py:TextAnswerReward exactly:
  async __call__(history: list[Message]) -> tuple[float, dict[str, float]]

Reward = format_coef * (correct_format - 1) + correct_answer
  - correct_format = 1 if the assistant emitted "Answer: ..." else 0
  - correct_answer = LLM-judge score (1.0 / 0.5 / 0.0)

We use the LLM-judge instead of normalized exact-match because LongMemEval /
LoCoMo answers are free-form sentences. Reuses our existing mempol.eval.judge.
"""
from __future__ import annotations
import asyncio
from dataclasses import dataclass

from mempol.eval.judge import judge as _judge_sync


def _extract_answer(text: str) -> str | None:
    """Pull the final "Answer:" line out of an assistant message."""
    if not isinstance(text, str) or "Answer:" not in text:
        return None
    parts = text.split("Answer:")
    if len(parts) < 2:
        return None
    # Take everything after the LAST "Answer:" — the policy might emit
    # "Answer:" inside a thought.
    return parts[-1].strip()


@dataclass
class JudgeReward:
    """LLM-judge reward over the agent's "Answer: ..." line."""
    gold_answer: str
    question: str
    format_coef: float = 0.1
    correct_reward: float = 1.0
    partial_reward: float = 0.5
    wrong_reward: float = 0.0

    async def __call__(self, history) -> tuple[float, dict[str, float]]:
        # Find last assistant message
        final = None
        for msg in reversed(history):
            try:
                role = msg.get("role")
            except AttributeError:
                role = getattr(msg, "role", None)
            if role == "assistant":
                final = msg
                break
        if final is None:
            return 0.0, {"format": 0.0, "correct": 0.0}

        # Pull text content. tinker_cookbook.renderers.get_text_content handles
        # thinking-model formats (o1, gpt-5). Fall back to dict access otherwise.
        try:
            from tinker_cookbook.renderers import get_text_content  # type: ignore
            content = get_text_content(final)
        except Exception:
            if isinstance(final, dict):
                content = final.get("content", "")
            else:
                content = getattr(final, "content", "") or ""
        if not isinstance(content, str):
            content = str(content) if content is not None else ""

        ans = _extract_answer(content)
        correct_format = float(ans is not None)
        if ans is None:
            return self.format_coef * (correct_format - 1.0), {
                "format": correct_format, "correct": 0.0,
            }

        # LLM-judge is sync — run in a thread so we don't block the event loop
        loop = asyncio.get_running_loop()
        judge_score, _ = await loop.run_in_executor(
            None, _judge_sync, self.question, self.gold_answer, ans
        )
        if judge_score >= 0.75:
            correct_answer = self.correct_reward
        elif judge_score >= 0.25:
            correct_answer = self.partial_reward
        else:
            correct_answer = self.wrong_reward

        reward = self.format_coef * (correct_format - 1.0) + correct_answer
        return reward, {"format": correct_format, "correct": correct_answer}
