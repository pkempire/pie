"""Memory environment + EnvGroupBuilder + RLDataset.

Mirrors `tinker_cookbook/recipes/search_tool/search_env.py` line-for-line in
shape. Drop this file into `tinker_cookbook/recipes/memory_rl/` after cloning
the cookbook — the imports below assume that layout.

Differences from the search recipe:
  - Each env builds a per-conversation Backend (we don't share one Chroma).
  - Tools come from MemoryTool (4 ops vs their 1).
  - Reward = JudgeReward (LLM-judge over conversational answer) instead of
    string-match Wikipedia answer.
"""
from __future__ import annotations
import random
from collections.abc import Sequence
from typing import TYPE_CHECKING

# Try to import tinker_cookbook; provide minimal stubs if standalone.
try:
    import chz                                              # type: ignore
    from tinker_cookbook import model_info, tokenizer_utils
    from tinker_cookbook.renderers import get_renderer
    from tinker_cookbook.renderers.base import Message, Renderer
    from tinker_cookbook.rl.types import (
        Env, EnvGroupBuilder, RLDataset, RLDatasetBuilder,
    )
    HAS_TINKER = True
except Exception:
    HAS_TINKER = False
    # Minimal stubs so this file imports + tests pass standalone. The real
    # training run requires the cookbook to be installed; calling
    # `make_envs()` without it will raise via build_agent_tool_env's stub.

    class chz:                                             # type: ignore
        @staticmethod
        def chz(cls):
            return cls

        @staticmethod
        def field(default_factory=None):
            return default_factory() if callable(default_factory) else None

    class _Stub:
        pass

    Env = _Stub
    EnvGroupBuilder = _Stub
    RLDataset = _Stub
    RLDatasetBuilder = _Stub
    Message = dict
    Renderer = _Stub
    model_info = _Stub
    tokenizer_utils = _Stub
    def get_renderer(name, tokenizer):
        raise RuntimeError("tinker_cookbook not installed")

# tinker_compat handles @tool + build_agent_tool_env in either mode
from mempol.recipes.memory_rl.tinker_compat import build_agent_tool_env

# Project imports — adjust path if mempol/ is not on PYTHONPATH:
from mempol.backends.flat import FlatBackend
from mempol.backends.base import Backend, Unit

from mempol.recipes.memory_rl.data import MemoryDatum, locomo_to_memory_data
from mempol.recipes.memory_rl.tools import MemoryTool
from mempol.recipes.memory_rl.reward import JudgeReward


MEMORY_TASK_INSTRUCTIONS = """You are an expert assistant who answers questions about a long conversation between two people, spread over many sessions across weeks or months. You cannot see the conversation directly. You query a memory store of CHUNKS — each chunk is ~6 consecutive turns from one session, prefixed with a header like:

    [1:56 pm on 8 May, 2023 | session 1]
    Caroline: Hey Mel! ...
    Melanie: ...
    ...

TOOLS
  memory_search(query, k=10, source="hybrid")  — search chunks. source ∈ {bm25, dense, hybrid}.
  memory_expand(seed_uids, k_per=2)            — pull adjacent chunks from the same session.
  memory_filter(predicate, value)              — filter current hits. predicate ∈ {session_lt, session_gt, session_eq, speaker_eq}.
  memory_rerank(strategy, query)               — reorder. strategy ∈ {dense, session_desc, session_asc}.

HOW TO SOLVE
1. Read the question carefully. Note any temporal cues ("yesterday", "last week", "in May").
2. Choose ONE good search query and call memory_search. Use 2-3 specific noun phrases from the question.
3. Read the chunk headers — they tell you the session DATE in absolute form. If the question mentions "yesterday" and the relevant chunk says "[8 May 2023 | session N]", the event was on 7 May 2023. Always resolve relative dates against the chunk's session header.
4. If the first search misses, refine the query with different keywords or use memory_expand on the most promising hit to see surrounding turns.
5. Stop searching once a chunk clearly contains the answer. More searches cost more.
6. Write your final answer on its own line: Answer: <answer>
7. Keep the answer concise — one short phrase or sentence. Match the format the question implies (a date, a name, a number, etc).

Avoid wasted searches. Each tool call has a cost; reward = correctness − tool-use cost.
"""


def _backend_from_units(units_dicts: list[dict]) -> Backend:
    """Rehydrate a per-env FlatBackend. Cheap because embeddings are cached on disk."""
    b = FlatBackend()
    units = [Unit(uid=u["uid"], text=u["text"], metadata=u["metadata"]) for u in units_dicts]
    b.ingest(units)
    return b


def _initial_messages(
    datum: MemoryDatum,
    renderer: Renderer,
    memory_tool: MemoryTool,
) -> list[Message]:
    """Tool schemas + system prompt + the question."""
    # In tinker-cookbook the @tool decorator gives each method a .to_spec().
    # When you copy this into the cookbook, add @tool to each method in tools.py.
    tool_schemas = [
        memory_tool.memory_search.to_spec(),  # type: ignore[attr-defined]
        memory_tool.memory_expand.to_spec(),  # type: ignore[attr-defined]
        memory_tool.memory_filter.to_spec(),  # type: ignore[attr-defined]
        memory_tool.memory_rerank.to_spec(),  # type: ignore[attr-defined]
    ]
    prefix = renderer.create_conversation_prefix_with_tools(
        tools=tool_schemas, system_prompt=MEMORY_TASK_INSTRUCTIONS
    )
    return prefix + [{"role": "user", "content": datum["question"]}]


class MemoryEnvGroupBuilder(EnvGroupBuilder):
    """Build G envs for one (conversation, question) datum — the GRPO group."""

    def __init__(
        self,
        datum: MemoryDatum,
        model_name: str,
        renderer_name: str | None,
        max_turns: int,
        group_size: int,
        format_coef: float = 0.1,
        max_trajectory_tokens: int = 32 * 1024,
        max_generation_tokens: int | None = None,
        context_overflow_reward: float = -0.1,
    ):
        self.datum = datum
        self.model_name = model_name
        self.renderer_name = renderer_name
        self.max_turns = max_turns
        self.group_size = group_size
        self.format_coef = format_coef
        self.max_trajectory_tokens = max_trajectory_tokens
        self.max_generation_tokens = max_generation_tokens
        self.context_overflow_reward = context_overflow_reward

    async def make_envs(self) -> Sequence[Env]:
        tokenizer = tokenizer_utils.get_tokenizer(self.model_name)
        renderer_name = self.renderer_name or model_info.get_recommended_renderer_name(self.model_name)
        renderer = get_renderer(renderer_name, tokenizer)

        envs: list[Env] = []
        for _ in range(self.group_size):
            # IMPORTANT: each env in the group gets its own backend — different
            # rollouts must not share mutable state on `last_hits`.
            backend = _backend_from_units(self.datum["conversation_units"])
            mtool = MemoryTool(backend=backend)
            initial_messages = _initial_messages(self.datum, renderer, mtool)
            reward_fn = JudgeReward(
                gold_answer=self.datum["gold_answer"],
                question=self.datum["question"],
                format_coef=self.format_coef,
            )
            envs.append(build_agent_tool_env(
                renderer=renderer,
                tools=[
                    mtool.memory_search,
                    mtool.memory_expand,
                    mtool.memory_filter,
                    mtool.memory_rerank,
                ],
                initial_messages=initial_messages,
                reward_fn=reward_fn,
                max_turns=self.max_turns,
                max_trajectory_tokens=self.max_trajectory_tokens,
                max_generation_tokens=self.max_generation_tokens,
                context_overflow_reward=self.context_overflow_reward,
            ))
        return envs

    def logging_tags(self) -> list[str]:
        return [self.datum.get("data_source", "unknown"), self.datum.get("category", "?")]


class MemoryRLDataset(RLDataset):
    def __init__(self, env_group_builders: list[MemoryEnvGroupBuilder], batch_size: int):
        self.env_group_builders = env_group_builders
        self.batch_size = batch_size

    def get_batch(self, index: int) -> Sequence[EnvGroupBuilder]:
        s = index * self.batch_size
        return self.env_group_builders[s:s + self.batch_size]

    def __len__(self) -> int:
        return len(self.env_group_builders) // self.batch_size


@chz.chz
class MemoryRLDatasetBuilder(RLDatasetBuilder):
    """Top-level builder Tinker calls. Returns (train_dataset, eval_dataset)."""
    model_name_for_tokenizer: str
    n_convs: int = 8
    train_frac: float = 0.8
    batch_size: int = 16
    group_size: int = 8
    renderer_name: str | None = None
    max_turns: int = 6
    format_coef: float = 0.1
    max_trajectory_tokens: int = 16 * 1024
    seed: int = 0

    async def __call__(self) -> tuple[RLDataset, RLDataset | None]:
        train_data, eval_data = locomo_to_memory_data(
            n_convs=self.n_convs, train_frac=self.train_frac, seed=self.seed,
        )
        rng = random.Random(self.seed)
        rng.shuffle(train_data)

        def to_builders(data: list[MemoryDatum]) -> list[MemoryEnvGroupBuilder]:
            return [
                MemoryEnvGroupBuilder(
                    datum=d,
                    model_name=self.model_name_for_tokenizer,
                    renderer_name=self.renderer_name,
                    max_turns=self.max_turns,
                    group_size=self.group_size,
                    format_coef=self.format_coef,
                    max_trajectory_tokens=self.max_trajectory_tokens,
                )
                for d in data
            ]

        train_ds = MemoryRLDataset(to_builders(train_data), batch_size=self.batch_size)
        eval_ds = MemoryRLDataset(to_builders(eval_data), batch_size=self.batch_size)
        return train_ds, eval_ds
