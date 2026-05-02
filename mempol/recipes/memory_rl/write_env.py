"""WriteEnv — Tinker RL env for the write policy.

Mirrors the read-side `memory_env.py` but with three structural differences:

  1. Per-turn episodes. This is the current smoke/SFT/GRPO scaffold, not the
     final continual-memory training environment. The final loop should score
     chronological chunk/session episodes against future QA, because memory
     quality is a property of an accumulated state, not a single isolated turn.

  2. Tools are write-side: lookup_entity, create_entity, update_state,
     merge_entities, add_relation, mark_contradiction, forget, noop. Plus
     lookup_relation for graph-aware decisions.

  3. Reward is deferred: WriteReward (write_reward.py) runs a frozen R
     against a held-out QA battery on the resulting memory state. The
     battery is pre-computed per conversation turn from LoCoMo's evidence
     labels — only questions whose `evidence` includes a dia_id from THIS
     turn (or its session) are in the battery. This is the natural
     counterfactual signal: "did W's writes preserve enough information to
     answer the questions that depend on this turn?"

The WriteRLDatasetBuilder iterates over (conv, turn_idx) pairs across the
training conversations.
"""
from __future__ import annotations
import logging
import random
from collections.abc import Sequence
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any, TypedDict

# ── Tinker imports (guarded for standalone testing) ─────────────────────────
try:
    import chz                                                     # type: ignore
    from tinker_cookbook import model_info, tokenizer_utils
    from tinker_cookbook.renderers import get_renderer
    from tinker_cookbook.renderers.base import Message, Renderer
    from tinker_cookbook.rl.types import (
        Env, EnvGroupBuilder, RLDataset, RLDatasetBuilder,
    )
    HAS_TINKER = True
except Exception:
    HAS_TINKER = False

    class chz:                                                     # type: ignore
        @staticmethod
        def chz(cls): return cls
        @staticmethod
        def field(default_factory=None):
            return default_factory() if callable(default_factory) else None
    class _Stub: pass
    Env = EnvGroupBuilder = RLDataset = RLDatasetBuilder = _Stub
    Message = dict; Renderer = _Stub
    model_info = _Stub; tokenizer_utils = _Stub
    def get_renderer(*a, **kw): raise RuntimeError("tinker_cookbook not installed")

from mempol.backends.flat import FlatBackend
from mempol.backends.base import Unit
from mempol.backends.pie_kg import PIEBackend
from mempol.data.locomo import Conversation, QA, Turn, load as load_locomo
from mempol.recipes.memory_rl.data import _conv_to_serializable_units
from mempol.recipes.memory_rl.tinker_compat import build_agent_tool_env
from mempol.recipes.memory_rl.write_tools import WriteTool
from mempol.recipes.memory_rl.write_reward import (
    WriteReward, resolve_r_runner_from_env,
)

logger = logging.getLogger(__name__)


# ── System prompt for the write policy ──────────────────────────────────────
WRITE_TASK_INSTRUCTIONS = """You maintain a knowledge graph of facts extracted from a long-running conversation. You see ONE turn of conversation at a time, plus the state of the existing knowledge graph for context. Your job is to decide what (if anything) to store from this turn.

# Tools

  lookup_entity(query, type=None, top_k=5)
      → list of nearby entities already in the graph. CALL FIRST if you might
        be about to create a duplicate.
  create_entity(name, type, state)
      → add a new entity. type ∈ {person, project, tool, organization, belief,
        decision, concept, period, event, goal}. state is a dict of attributes.
  update_state(uid, new_state, transition_type, trigger_summary)
      → update an existing entity. transition_type ∈ {update, contradiction,
        resolution, archival}.
  merge_entities(canonical_uid, alias_uid)
      → collapse two entities. Use when lookup returns a high-similarity dup.
  add_relation(source_uid, target_uid, rel_type, description)
      → link two entities. rel_type ∈ {uses, works_on, collaborates_with,
        related_to, part_of, caused_by, during, replaces, integrates_with}.
  mark_contradiction(uid, contradicting_state)
      → flag a conflict. Both states retained.
  forget(uid, reason) → archive. Soft-delete with audit trail.
  noop(reason="") → this turn isn't memory-worthy.

# Strategy

1. Default to noop. Most turns are filler, acknowledgments, or repeats.
2. Store only durable, specific information: identities, decisions, plans,
   deadlines, names, concrete facts. NOT pleasantries.
3. ALWAYS call lookup_entity before create_entity if you might be about to
   create a duplicate.
4. Prefer update_state over create_entity if the entity exists.
5. Use merge_entities aggressively when lookup finds a high-similarity match.

# Output format

For each tool call:
<tool_call>
{"name": "<tool_name>", "arguments": {<args>}}
</tool_call>

Emit as many tool calls as are useful, then stop. The runtime may still impose
a max trajectory/token budget to prevent infinite loops, but there is no
semantic "four calls only" rule. Reward is computed AFTER the episode by
querying your stored memory against held-out questions.
"""


# ── Datum / Dataset shapes ──────────────────────────────────────────────────
class WriteDatum(TypedDict):
    """One training example for the W policy.

    Keys:
        conv_id: conversation identifier (string).
        turn_idx: index of the focal turn within the conversation.
        turn_text: "Speaker: text" of the focal turn.
        turn_dia_id: e.g. "D5:3".
        session_date: human-readable date of the session containing this turn.
        prior_turns_text: text of the K turns immediately preceding this one
            (for the W policy's context, NOT given to W as a tool result).
        existing_entities_summary: a compact string of pre-existing entities
            in this datum's PIEBackend (empty for v1 — each WriteEnv
            currently starts with an empty PIEBackend; future variants will
            seed with an in-progress KG snapshot).
        query_battery: list of (question, gold_answer) pairs that depend on
            this turn (per LoCoMo's `evidence` labels).
    """
    conv_id: str
    turn_idx: int
    turn_text: str
    turn_dia_id: str
    session_date: str
    prior_turns_text: str
    existing_entities_summary: str
    # (question, gold_answer, evidence_dia_ids). evidence is unused by the
    # reader-overlap reward but kept for backward-compat logging of the
    # legacy coverage signal as a control.
    query_battery: list[tuple[str, str, list[str]]]
    # Full-conversation text backend (FlatBackend). The reader queries this
    # both for reader-overlap (legacy) and for the random-K baseline used
    # in answer-gain. Built once per conv.
    full_text_backend: Any
    # Per-conv cache of question -> set of dia_ids the reader retrieves
    # from full text. Same dict reference shared across all turns of one
    # conv so the per-question full-text retrieval is cached.
    full_text_cache: dict
    # Per-conv cache of (conv_id, q, K) -> baseline judge score for
    # answer-gain. Same dict reference shared across all rollouts so the
    # random-K baseline is judged at most once per (conv, q, K) over the
    # whole training run.
    baseline_cache: dict


class WriteEnvGroupBuilder(EnvGroupBuilder):
    """Build G envs for one WriteDatum — the GRPO group.

    Each env in the group gets its own fresh PIEBackend (no shared state
    across rollouts). The frozen R for reward is shared (read-only).

    Inherits compute_group_rewards (default: zero group reward, episode
    rewards via reward_fn already cover us) and cleanup (no-op) from the
    base class. We override make_envs and logging_tags only.
    """

    def __init__(
        self,
        datum: WriteDatum,
        model_name: str,
        renderer_name: str | None,
        group_size: int,
    max_turns: int = 32,
        format_coef: float = 0.0,
        max_trajectory_tokens: int = 4 * 1024,
        max_generation_tokens: int | None = None,
        context_overflow_reward: float = -0.1,
    ):
        self.datum = datum
        self.model_name = model_name
        self.renderer_name = renderer_name
        self.group_size = group_size
        self.max_turns = max_turns
        self.format_coef = format_coef
        self.max_trajectory_tokens = max_trajectory_tokens
        self.max_generation_tokens = max_generation_tokens
        self.context_overflow_reward = context_overflow_reward

    async def make_envs(self) -> Sequence[Env]:
        if not HAS_TINKER:
            raise RuntimeError("WriteEnvGroupBuilder requires tinker_cookbook")
        tokenizer = tokenizer_utils.get_tokenizer(self.model_name)
        renderer_name = (
            self.renderer_name
            or model_info.get_recommended_renderer_name(self.model_name)
        )
        renderer = get_renderer(renderer_name, tokenizer)
        envs: list[Env] = []
        for _ in range(self.group_size):
            backend = PIEBackend()                       # fresh per group member
            wtool = WriteTool(backend=backend)
            wtool.current_turn_text = self.datum["turn_text"]
            wtool.current_dia_id = self.datum["turn_dia_id"]
            wtool.current_timestamp = float(self.datum["turn_idx"])
            initial_messages = _initial_messages(self.datum, renderer, wtool)
            # If MEMPOL_R_CHECKPOINT is set, the trained R LoRA is used as
            # the QA-judge backbone; otherwise the heuristic R is the
            # default (resolved inside WriteReward.__post_init__).
            r_runner = resolve_r_runner_from_env()
            # Reward-mix knobs. Defaults reflect the v3 reward (per-op
            # counterfactual marginal utility as the primary dense signal,
            # absolute judge as an anchor). Legacy answer-gain and
            # reader-overlap weights default to 0; bump them via env vars
            # for ablation runs.
            import os as _os
            w_cf        = float(_os.environ.get("MEMPOL_W_CF",        "0.7"))
            w_qa        = float(_os.environ.get("MEMPOL_W_QA",        "0.3"))
            w_gain      = float(_os.environ.get("MEMPOL_W_GAIN",      "0.0"))
            w_overlap   = float(_os.environ.get("MEMPOL_W_OVERLAP",   "0.0"))
            w_cov_floor = float(_os.environ.get("MEMPOL_W_COV_FLOOR", "0.05"))
            k_max       = int(_os.environ.get("MEMPOL_K_MAX",         "12"))
            reward_fn = WriteReward(
                backend=backend,
                query_battery=self.datum["query_battery"],
                full_text_backend=self.datum.get("full_text_backend"),
                r_runner=r_runner,
                write_tool=wtool,
                conv_id=self.datum.get("conv_id", ""),
                w_cf=w_cf,
                w_qa=w_qa,
                w_gain=w_gain,
                w_overlap=w_overlap,
                w_cov_floor=w_cov_floor,
                k_max=k_max,
                full_text_cache=self.datum.get("full_text_cache"),
                baseline_cache=self.datum.get("baseline_cache"),
            )
            envs.append(build_agent_tool_env(
                renderer=renderer,
                tools=[
                    wtool.lookup_entity, wtool.lookup_relation,
                    wtool.create_entity, wtool.update_state,
                    wtool.merge_entities, wtool.add_relation,
                    wtool.mark_contradiction, wtool.forget, wtool.noop,
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
        return ["write", f"battery_size_{len(self.datum['query_battery'])}"]


def _initial_messages(datum: WriteDatum, renderer, wtool: WriteTool) -> list[dict]:
    """System prompt + tool schemas + the per-turn user prompt."""
    tool_schemas = [
        wtool.lookup_entity.to_spec(),
        wtool.lookup_relation.to_spec(),
        wtool.create_entity.to_spec(),
        wtool.update_state.to_spec(),
        wtool.merge_entities.to_spec(),
        wtool.add_relation.to_spec(),
        wtool.mark_contradiction.to_spec(),
        wtool.forget.to_spec(),
        wtool.noop.to_spec(),
    ]
    prefix = renderer.create_conversation_prefix_with_tools(
        tools=tool_schemas, system_prompt=WRITE_TASK_INSTRUCTIONS,
    )
    user_block = (
        f"Session: {datum['session_date']}\n"
        f"Recent turns leading to this one:\n{datum['prior_turns_text']}\n\n"
        f"FOCAL TURN ({datum['turn_dia_id']}):\n{datum['turn_text']}\n\n"
        f"Existing entities (top-K nearby):\n{datum['existing_entities_summary'] or '(none)'}\n\n"
        f"What write ops should you emit for the focal turn? "
        f"Default to noop unless the turn carries durable, specific information."
    )
    return prefix + [{"role": "user", "content": user_block}]


class WriteRLDataset(RLDataset):
    """Dataset of write episodes. One episode per (conv, turn_idx) pair."""

    def __init__(self, builders: list[WriteEnvGroupBuilder], batch_size: int):
        self.builders = builders
        self.batch_size = batch_size

    def get_batch(self, index: int) -> Sequence[EnvGroupBuilder]:
        s = index * self.batch_size
        return self.builders[s:s + self.batch_size]

    def __len__(self) -> int:
        # Floor-divide can return 0 when filters knock the corpus below
        # batch_size — Tinker then iterates zero batches and the run is a
        # silent no-op. Guarantee at least one partial batch.
        if not self.builders:
            return 0
        return max(1, len(self.builders) // self.batch_size)


@chz.chz
class WriteRLDatasetBuilder(RLDatasetBuilder):
    """Top-level builder. Tinker calls this to get train + eval datasets.

    Iterates LoCoMo conversations and produces WriteDatums:
      For each conv, for each user/assistant turn:
        - Find QAs whose `evidence` mentions a dia_id in this turn's session
        - If at least one such QA exists, build a WriteDatum
        - Otherwise skip (noop-only episodes give no learning signal)
    """
    model_name_for_tokenizer: str
    n_convs: int = 8
    train_frac: float = 0.8
    batch_size: int = 8
    group_size: int = 4
    max_turns: int = 4
    max_battery_per_turn: int = 0        # 0 = use all Qs for this turn
    min_battery_per_turn: int = 1
    n_prior_turns_in_context: int = 2
    seed: int = 0
    renderer_name: str | None = None

    async def __call__(self) -> tuple[RLDataset, RLDataset | None]:
        convs = load_locomo(n_convs=self.n_convs)
        rng = random.Random(self.seed)
        rng.shuffle(convs)
        n_train = max(1, int(len(convs) * self.train_frac))

        def conv_to_write_data(conv_qas: list[tuple[Conversation, list[QA]]]) -> list[WriteDatum]:
            data: list[WriteDatum] = []
            for conv, qas in conv_qas:
                # Build dia_id → list of QAs whose evidence contains this dia_id.
                evidence_index: dict[str, list[QA]] = {}
                for qa in qas:
                    for did in qa.evidence:
                        evidence_index.setdefault(did, []).append(qa)

                # Build the full-text backend once per conversation. Shared
                # across all turn-episodes of this conv so the reader's
                # full-text retrieval is consistent and the cache hits.
                unit_dicts = _conv_to_serializable_units(conv)
                full_text_backend = FlatBackend()
                full_text_backend.ingest([
                    Unit(uid=u["uid"], text=u["text"], metadata=u["metadata"])
                    for u in unit_dicts
                ])
                full_text_cache: dict = {}              # shared per conv
                baseline_cache:  dict = {}              # shared per conv (answer-gain)

                for ti, t in enumerate(conv.turns):
                    qas_for_turn = evidence_index.get(t.dia_id, [])
                    if len(qas_for_turn) < self.min_battery_per_turn:
                        continue                       # too few Qs → no GRPO variance
                    battery = (
                        qas_for_turn
                        if self.max_battery_per_turn <= 0
                        else qas_for_turn[: self.max_battery_per_turn]
                    )
                    prior_turns = conv.turns[max(0, ti - self.n_prior_turns_in_context):ti]
                    prior_text = "\n".join(
                        f"  {pt.dia_id} {pt.speaker}: {pt.text}" for pt in prior_turns
                    ) or "  (none)"

                    data.append(WriteDatum(
                        conv_id=conv.sample_id,
                        turn_idx=ti,
                        turn_text=f"{t.speaker}: {t.text}",
                        turn_dia_id=t.dia_id,
                        session_date=t.session_date,
                        prior_turns_text=prior_text,
                        existing_entities_summary="",     # v1: fresh KG per env
                        query_battery=[
                            (qa.question, qa.answer, list(qa.evidence or []))
                            for qa in battery
                        ],
                        full_text_backend=full_text_backend,
                        full_text_cache=full_text_cache,
                        baseline_cache=baseline_cache,
                    ))
            return data

        train_data = conv_to_write_data(convs[:n_train])
        eval_data = conv_to_write_data(convs[n_train:])
        rng.shuffle(train_data)
        logger.info("WriteRLDatasetBuilder: %d train datums, %d eval datums",
                    len(train_data), len(eval_data))

        def to_builders(data: list[WriteDatum]) -> list[WriteEnvGroupBuilder]:
            return [
                WriteEnvGroupBuilder(
                    datum=d,
                    model_name=self.model_name_for_tokenizer,
                    renderer_name=self.renderer_name,
                    group_size=self.group_size,
                    max_turns=self.max_turns,
                )
                for d in data
            ]

        train_ds = WriteRLDataset(to_builders(train_data), batch_size=self.batch_size)
        eval_ds = WriteRLDataset(to_builders(eval_data), batch_size=self.batch_size)
        return train_ds, eval_ds
