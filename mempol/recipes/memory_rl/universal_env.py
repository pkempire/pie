"""Universal-memory GRPO env for Tinker.

This is the full-RL path for the universal substrate:

  raw artifacts/spans -> policy writes freeform MemoryStates -> policy retrieves
  MemoryStates -> final answer -> reward

The important constraint is `freeze_raw_access`: raw-span search can be used to
build memory, but the policy is penalized if it answers without freezing raw
access. That prevents the trivial solution "just do raw RAG" and makes written
memory causally useful.
"""
from __future__ import annotations

import random
import tempfile
from collections.abc import Sequence
from pathlib import Path
from typing import TypedDict

try:
    import chz  # type: ignore
    from tinker_cookbook import model_info, tokenizer_utils
    from tinker_cookbook.renderers import get_renderer
    from tinker_cookbook.renderers.base import Message, Renderer
    from tinker_cookbook.rl.types import Env, EnvGroupBuilder, RLDataset, RLDatasetBuilder
    HAS_TINKER = True
except Exception:
    HAS_TINKER = False

    class chz:  # type: ignore
        @staticmethod
        def chz(cls): return cls
    class _Stub: pass
    Env = EnvGroupBuilder = RLDataset = RLDatasetBuilder = _Stub
    Message = dict
    Renderer = _Stub
    model_info = tokenizer_utils = _Stub
    def get_renderer(*a, **kw): raise RuntimeError("tinker_cookbook not installed")

from mempol.core.schema import Artifact, Span
from mempol.core.store import SQLiteMemoryStore, now_iso
from mempol.data.locomo import load as load_locomo
from mempol.recipes.memory_rl.tinker_compat import build_agent_tool_env
from mempol.recipes.memory_rl.universal_reward import UniversalMemoryReward
from mempol.recipes.memory_rl.universal_tools import UniversalMemoryTool


UNIVERSAL_MEMORY_INSTRUCTIONS = """You are training a universal memory policy.

You do not get the full corpus directly. You have tools:

  search_raw_spans(query, k)
      Search raw evidence spans. Use this to discover information.

  write_memory_state(content, source_span_ids)
      Write a compact, useful memory backed by raw evidence.

  freeze_raw_access(reason)
      Disable raw search once you have built enough memory. You should do this
      before answering; otherwise reward is penalized.

  retrieve_memory_states(query, k)
      Search compressed memory states. Use this after writing memory.

Goal:
  Answer the question correctly while writing the smallest useful memory and
  using few tokens/actions.

Rules:
  - Every written memory must cite source_span_ids returned by search_raw_spans.
  - Prefer compact memory that will help answer the question.
  - Do not answer from raw spans alone. Write memory, freeze raw access, retrieve
    memory, then answer.
  - Final answer format must be exactly:
      Answer: <short answer>
"""


class UniversalDatum(TypedDict):
    question: str
    gold_answer: str
    qid: str
    sample_id: str
    source: str
    raw_spans: list[dict]


def _locomo_raw_span_data(n_convs: int | None = None, train_frac: float = 0.8, seed: int = 0) -> tuple[list[UniversalDatum], list[UniversalDatum]]:
    convs = load_locomo(n_convs=n_convs)
    rng = random.Random(seed)
    rng.shuffle(convs)
    n_train = int(len(convs) * train_frac)

    def convert(split):
        rows: list[UniversalDatum] = []
        for conv, qas in split:
            raw_spans = []
            for t in conv.turns:
                aid = f"locomo_artifact_{conv.sample_id}_{t.dia_id.replace(':', '_')}"
                sid = f"locomo_span_{conv.sample_id}_{t.dia_id.replace(':', '_')}"
                raw_spans.append({
                    "artifact": {
                        "id": aid,
                        "source": "locomo",
                        "kind": "conversation_turn",
                        "title": f"{conv.sample_id} {t.dia_id}",
                        "content": f"{t.speaker}: {t.text}",
                        "created_at": t.session_date,
                        "metadata": {
                            "sample_id": conv.sample_id,
                            "dia_id": t.dia_id,
                            "session": t.session,
                            "speaker": t.speaker,
                            "session_date": t.session_date,
                        },
                    },
                    "span": {
                        "id": sid,
                        "artifact_id": aid,
                        "text": f"{t.speaker}: {t.text}",
                        "locator": t.dia_id,
                        "metadata": {"dia_id": t.dia_id, "speaker": t.speaker},
                    },
                })
            for qa in qas:
                rows.append({
                    "question": qa.question,
                    "gold_answer": qa.answer,
                    "qid": qa.qid,
                    "sample_id": conv.sample_id,
                    "source": "locomo",
                    "raw_spans": raw_spans,
                })
        return rows

    return convert(convs[:n_train]), convert(convs[n_train:])


def _build_store_for_datum(datum: UniversalDatum) -> SQLiteMemoryStore:
    tmp = tempfile.NamedTemporaryFile(prefix="mempol_universal_env_", suffix=".sqlite", delete=False)
    tmp.close()
    store = SQLiteMemoryStore(Path(tmp.name))
    for row in datum["raw_spans"]:
        a = row["artifact"]
        s = row["span"]
        store.upsert_artifact(Artifact(**a))
        store.upsert_span(Span(**s))
    store.commit()
    return store


def _initial_messages(datum: UniversalDatum, renderer: Renderer, tool: UniversalMemoryTool) -> list[Message]:
    tool_schemas = [
        tool.search_raw_spans.to_spec(),  # type: ignore[attr-defined]
        tool.write_memory_state.to_spec(),  # type: ignore[attr-defined]
        tool.freeze_raw_access.to_spec(),  # type: ignore[attr-defined]
        tool.retrieve_memory_states.to_spec(),  # type: ignore[attr-defined]
    ]
    prefix = renderer.create_conversation_prefix_with_tools(
        tools=tool_schemas,
        system_prompt=UNIVERSAL_MEMORY_INSTRUCTIONS,
    )
    return prefix + [{"role": "user", "content": datum["question"]}]


class UniversalMemoryEnvGroupBuilder(EnvGroupBuilder):
    def __init__(
        self,
        datum: UniversalDatum,
        model_name: str,
        renderer_name: str | None,
        group_size: int,
        max_turns: int = 10,
        max_trajectory_tokens: int = 12 * 1024,
        max_generation_tokens: int | None = None,
        context_overflow_reward: float = -0.2,
    ):
        self.datum = datum
        self.model_name = model_name
        self.renderer_name = renderer_name
        self.group_size = group_size
        self.max_turns = max_turns
        self.max_trajectory_tokens = max_trajectory_tokens
        self.max_generation_tokens = max_generation_tokens
        self.context_overflow_reward = context_overflow_reward

    async def make_envs(self) -> Sequence[Env]:
        if not HAS_TINKER:
            raise RuntimeError("UniversalMemoryEnvGroupBuilder requires tinker_cookbook")
        tokenizer = tokenizer_utils.get_tokenizer(self.model_name)
        renderer_name = self.renderer_name or model_info.get_recommended_renderer_name(self.model_name)
        renderer = get_renderer(renderer_name, tokenizer)

        envs: list[Env] = []
        for _ in range(self.group_size):
            store = _build_store_for_datum(self.datum)
            tool = UniversalMemoryTool(store=store)
            reward_fn = UniversalMemoryReward(
                question=self.datum["question"],
                gold_answer=self.datum["gold_answer"],
                tool=tool,
            )
            envs.append(build_agent_tool_env(
                renderer=renderer,
                tools=[
                    tool.search_raw_spans,
                    tool.write_memory_state,
                    tool.freeze_raw_access,
                    tool.retrieve_memory_states,
                ],
                initial_messages=_initial_messages(self.datum, renderer, tool),
                reward_fn=reward_fn,
                max_turns=self.max_turns,
                max_trajectory_tokens=self.max_trajectory_tokens,
                max_generation_tokens=self.max_generation_tokens,
                context_overflow_reward=self.context_overflow_reward,
            ))
        return envs

    def logging_tags(self) -> list[str]:
        return [self.datum["source"], self.datum["sample_id"]]


class UniversalMemoryRLDataset(RLDataset):
    def __init__(self, builders: list[UniversalMemoryEnvGroupBuilder], batch_size: int):
        self.builders = builders
        self.batch_size = batch_size

    def get_batch(self, index: int) -> Sequence[EnvGroupBuilder]:
        s = index * self.batch_size
        return self.builders[s:s + self.batch_size]

    def __len__(self) -> int:
        if not self.builders:
            return 0
        return max(1, len(self.builders) // self.batch_size)


@chz.chz
class UniversalMemoryRLDatasetBuilder(RLDatasetBuilder):
    model_name_for_tokenizer: str
    n_convs: int = 2
    train_frac: float = 0.8
    batch_size: int = 2
    group_size: int = 4
    renderer_name: str | None = None
    max_turns: int = 10
    max_trajectory_tokens: int = 12 * 1024
    seed: int = 0

    async def __call__(self) -> tuple[RLDataset, RLDataset | None]:
        train_rows, eval_rows = _locomo_raw_span_data(
            n_convs=self.n_convs,
            train_frac=self.train_frac,
            seed=self.seed,
        )
        rng = random.Random(self.seed)
        rng.shuffle(train_rows)

        def make(rows: list[UniversalDatum]) -> UniversalMemoryRLDataset:
            return UniversalMemoryRLDataset([
                UniversalMemoryEnvGroupBuilder(
                    datum=row,
                    model_name=self.model_name_for_tokenizer,
                    renderer_name=self.renderer_name,
                    group_size=self.group_size,
                    max_turns=self.max_turns,
                    max_trajectory_tokens=self.max_trajectory_tokens,
                )
                for row in rows
            ], batch_size=self.batch_size)

        return make(train_rows), make(eval_rows)


def smoke() -> None:
    train, eval_rows = _locomo_raw_span_data(n_convs=1, train_frac=1.0)
    print(f"train={len(train)} eval={len(eval_rows)}")
    store = _build_store_for_datum(train[0])
    tool = UniversalMemoryTool(store=store)
    print(tool.search_raw_spans_impl("LGBTQ support group", k=3))
    store.close()


if __name__ == "__main__":
    smoke()
