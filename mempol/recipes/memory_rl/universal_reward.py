"""Reward for universal-memory RL episodes."""
from __future__ import annotations

import asyncio
from dataclasses import dataclass
from typing import Any

from mempol.eval.judge import judge as _judge_sync
from mempol.recipes.memory_rl.reward import _extract_answer
from mempol.recipes.memory_rl.universal_tools import UniversalMemoryTool


@dataclass
class UniversalMemoryReward:
    question: str
    gold_answer: str
    tool: UniversalMemoryTool
    format_coef: float = 0.1
    write_bonus: float = 0.05
    token_cost_coef: float = 0.0005
    raw_left_open_penalty: float = 0.15

    async def __call__(self, history) -> tuple[float, dict[str, float]]:
        final_text = ""
        for msg in reversed(history):
            role = msg.get("role") if isinstance(msg, dict) else getattr(msg, "role", None)
            if role == "assistant":
                try:
                    from tinker_cookbook.renderers import get_text_content  # type: ignore
                    final_text = get_text_content(msg)
                except Exception:
                    final_text = msg.get("content", "") if isinstance(msg, dict) else getattr(msg, "content", "")
                break

        ans = _extract_answer(final_text or "")
        format_ok = float(ans is not None)
        if ans is None:
            score = 0.0
        else:
            loop = asyncio.get_running_loop()
            judge_score, _ = await loop.run_in_executor(None, _judge_sync, self.question, self.gold_answer, ans)
            score = float(judge_score)

        stats = self.tool.stats()
        # Reward memory use, but only lightly; the main signal must remain task correctness.
        write_bonus = self.write_bonus if stats["writes"] > 0 else 0.0
        cost = self.token_cost_coef * float(stats["token_cost"])
        raw_penalty = self.raw_left_open_penalty if stats["raw_enabled"] else 0.0
        reward = score + self.format_coef * (format_ok - 1.0) + write_bonus - cost - raw_penalty
        return reward, {
            "judge_score": score,
            "format": format_ok,
            "writes": float(stats["writes"]),
            "raw_searches": float(stats["raw_searches"]),
            "memory_searches": float(stats["memory_searches"]),
            "token_cost": float(stats["token_cost"]),
            "raw_left_open": float(stats["raw_enabled"]),
            "cost_penalty": cost,
            "reward": reward,
        }
