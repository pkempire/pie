"""Mastra-inspired Observational Memory backend (Observer + Reflector).

This is a Python baseline inspired by Mastra's Observational Memory, not the
official Mastra implementation. Use it for ablations, not reproduction claims.
Exact Mastra comparison should run the official TypeScript package.

NOT vector retrieval. The architecture is:

    raw turns → [Observer] → dated bullet observations → [Reflector] → condensed reflections

At query time the agent sees:

    {reflections (condensed long-term)}
    {observations (recent observer bullets)}
    {recent raw turns (last K)}

There is no per-query retrieval. The context is *stable* — that's Mastra's whole
prompt-caching pitch. We keep the `Backend.retrieve()` interface for compatibility
by returning the full log as a sequence of Hits, but the policy can also call
`get_full_context()` directly for the proper Mastra-shaped prompt.

Official Mastra defaults are 30k / 40k. For LoCoMo's ~21k-token conversations
this backend defaults lower so the Observer fires multiple times across a
single conversation — otherwise it would barely engage and we'd be measuring
almost only the recent raw window.
"""
from __future__ import annotations
import json
import re
import time
from dataclasses import dataclass, field
from typing import Any

import numpy as np

from .. import config, llm
from .base import Backend, Hit, Unit


# ── Helpers ──
def _estimate_tokens(text: str) -> int:
    """Cheap proxy: ~4 chars per token (roughly OpenAI-like English ratio)."""
    return max(1, len(text) // 4)


# ── Observer / Reflector prompts (modelled on Mastra's documented behaviour) ──
_OBSERVER_SYS = """You are the OBSERVER in Mastra's Observational Memory system.

You are given a chunk of recent conversation between a user and an AI assistant (or between two peers). Your job is to write CONCISE, DATED, PRIORITIZED observations capturing what happened, what was decided, and what changed — NOT a transcript.

Format strictly:

Date: <YYYY-MM-DD>

- 🔴 <HH:MM> <observation: a single durable fact, decision, plan, identity attribute, preference, deadline, or change>
  - 🔴 <HH:MM> <sub-observation if it logically nests>
  - 🟡 <HH:MM> <medium-priority detail>
- 🟡 <HH:MM> <observation>
- 🟢 <HH:MM> <low-priority but worth keeping>

Priority key:
  🔴 HIGH — identity, decisions, plans with deadlines, named entities, durable facts
  🟡 MED  — preferences, ongoing topics, named relationships
  🟢 LOW  — minor details that may matter later

Rules:
- Each line is ONE complete observation (no run-ons).
- Use the SPEAKER's name when known (e.g. "Caroline went to ...").
- Keep dates from the conversation (e.g. "1:56 pm on 8 May, 2023" → "2023-05-08").
- Skip pure pleasantries, acknowledgments, and chitchat.
- Compression target: 5–10× shorter than the input.

After the bullet list, on a new line, write:
  Current task: <one short phrase summarising what the user is currently working on>
  Suggested response: <one short phrase suggesting what the assistant might do next>

Return ONLY the markdown observation block. No JSON, no preamble.
"""

_REFLECTOR_SYS = """You are the REFLECTOR in Mastra's Observational Memory.

You are given a list of observations produced by the Observer over time. Your job is to CONDENSE them into a smaller list that preserves the durable, important content while collapsing related items.

Strategy:
- Group related observations into single composite lines.
- Keep dates and priority emojis.
- Drop redundant restatements.
- If something was contradicted later, keep the latest with a (← prior) note.
- Preserve named entities, decisions, identity, deadlines, and preferences.

Output the same dated bullet markdown format as the Observer. Return ONLY markdown.
Compression target: another 2–3× over the input.
"""


# ── Data classes ──
@dataclass
class _RawTurn:
    uid: str
    text: str
    metadata: dict[str, Any]


@dataclass
class ObservationBlock:
    """One Observer call's output. Held verbatim — Mastra's log is markdown."""
    date_label: str
    markdown: str
    source_uids: list[str] = field(default_factory=list)
    current_task: str = ""
    suggested_response: str = ""
    n_input_chars: int = 0
    n_output_chars: int = 0


@dataclass
class ReflectionBlock:
    """One Reflector call's output."""
    markdown: str
    n_observations_consumed: int = 0
    n_input_chars: int = 0
    n_output_chars: int = 0


# ── Backend ──
class MastraBackend(Backend):
    name = "mastra_om"

    def __init__(
        self,
        observer_token_threshold: int = 3_000,
        reflector_token_threshold: int = 8_000,
        keep_recent_n: int = 20,
    ):
        self.observer_token_threshold = observer_token_threshold
        self.reflector_token_threshold = reflector_token_threshold
        self.keep_recent_n = keep_recent_n

        self._raw_buffer: list[_RawTurn] = []   # turns since last Observer call
        self._all_turns: list[_RawTurn] = []    # for keep_recent_n window
        self.observations: list[ObservationBlock] = []
        self.reflections: list[ReflectionBlock] = []

        # Embedding index for semantic retrieval over observations
        self._obs_embeddings: Any = None  # np.ndarray (N, D) or None
        self._obs_index_dirty: bool = True

        self._stats = {
            "n_observer_runs": 0,
            "n_reflector_runs": 0,
            "n_raw_turns_seen": 0,
            "n_raw_chars_seen": 0,
            "n_observation_chars": 0,
            "n_reflection_chars": 0,
        }

    # ── Public Backend API ──
    def ingest(self, units: list[Unit]) -> None:
        for u in units:
            self._ingest_one(u)
        # End-of-batch: flush the buffer if anything's left so the log is complete.
        self._maybe_run_observer(force=True)

    def retrieve(self, query: str, k: int = 10, source: str = "hybrid") -> list[Hit]:
        """Semantic retrieval over observations + always-include reflections + recent raw.

        Embeds the query, scores against observation embeddings via cosine
        similarity, and returns top-k observations. Reflections and recent
        raw turns are always appended as bonus context.
        """
        out: list[Hit] = []

        # 1. Always include reflections (condensed long-term) — they're small
        for i, r in enumerate(self.reflections):
            out.append(Hit(
                unit=Unit(
                    uid=f"reflect::{i}", text=r.markdown,
                    metadata={"kind": "reflection", "section_idx": i},
                ),
                score=1.0, source="reflection",
            ))

        # 2. Semantic search over observations
        if self.observations:
            self._reindex_observations()
            if self._obs_embeddings is not None and len(self._obs_embeddings) > 0:
                q_emb = llm.embed([query])[0]
                q_emb = q_emb / (np.linalg.norm(q_emb) + 1e-8)
                scores = np.dot(self._obs_embeddings, q_emb)
                top_k = min(k, len(scores))
                top_idx = np.argsort(scores)[-top_k:][::-1]
                for idx in top_idx:
                    o = self.observations[int(idx)]
                    out.append(Hit(
                        unit=Unit(
                            uid=f"observ::{idx}", text=o.markdown,
                            metadata={
                                "kind": "observation", "section_idx": int(idx),
                                "current_task": o.current_task, "date": o.date_label,
                            },
                        ),
                        score=float(scores[idx]), source="observation",
                    ))

        # 3. Recent raw turns (always appended as fallback)
        for t in self._all_turns[-self.keep_recent_n:]:
            out.append(Hit(
                unit=Unit(
                    uid=t.uid, text=t.text,
                    metadata={"kind": "raw_turn", **t.metadata},
                ),
                score=0.4, source="recent_raw",
            ))

        return out

    def _reindex_observations(self) -> None:
        """(Re)build embedding index for all observation blocks."""
        if not self._obs_index_dirty:
            return
        if not self.observations:
            self._obs_embeddings = None
            self._obs_index_dirty = False
            return
        texts = [o.markdown for o in self.observations]
        try:
            embs = llm.embed(texts)
            # L2-normalize for cosine similarity via dot product
            norms = np.linalg.norm(embs, axis=1, keepdims=True) + 1e-8
            self._obs_embeddings = embs / norms
            self._obs_index_dirty = False
        except Exception as e:
            print(f"[mastra] embedding index build failed: {e}")

    def expand(self, seed_uids: list[str], k_per: int = 2) -> list[Hit]:
        """Mastra has no graph. We approximate by returning observations
        adjacent to the one a seed came from — useful when policies want
        more context after a hit."""
        out: list[Hit] = []
        seen = set(seed_uids)
        for uid in seed_uids:
            if not uid.startswith("observ::"):
                continue
            try:
                idx = int(uid.split("::", 1)[1])
            except Exception:
                continue
            for j in (idx - 1, idx + 1):
                if 0 <= j < len(self.observations):
                    new_uid = f"observ::{j}"
                    if new_uid in seen:
                        continue
                    seen.add(new_uid)
                    o = self.observations[j]
                    out.append(Hit(
                        unit=Unit(uid=new_uid, text=o.markdown,
                                  metadata={"kind": "observation", "section_idx": j}),
                        score=0.5, source="expand_om",
                    ))
        return out[: k_per * len(seed_uids)]

    # ── The Mastra-shaped context the answer LLM should see ──
    def get_full_context(self) -> str:
        """The agent's view of memory at query time — stable, replaces raw history.
        Concatenates: reflections → observations → recent raw turns."""
        parts: list[str] = []
        if self.reflections:
            parts.append("## Reflections (condensed long-term memory)\n")
            for r in self.reflections:
                parts.append(r.markdown.strip())
        if self.observations:
            parts.append("\n## Observations (Observer log)\n")
            for o in self.observations:
                parts.append(o.markdown.strip())
        recent = self._all_turns[-self.keep_recent_n:]
        if recent:
            parts.append("\n## Recent raw turns (last "
                         f"{len(recent)} of {len(self._all_turns)})\n")
            for t in recent:
                speaker = t.metadata.get("speaker", "?")
                dia = t.metadata.get("dia_id", "?")
                parts.append(f"- **{speaker}** `{dia}`: {t.text}")
        return "\n".join(parts).strip()

    def memory_log_md(self) -> str:
        """Human-readable dump of the entire Mastra memory state."""
        md = [
            "# Mastra Observational Memory log",
            "",
            f"**{len(self.observations)} observation blocks**, "
            f"**{len(self.reflections)} reflection blocks**, "
            f"**{len(self._all_turns)} raw turns** seen total "
            f"(last {min(self.keep_recent_n, len(self._all_turns))} kept verbatim).",
            "",
            f"Compression: raw_chars={self._stats['n_raw_chars_seen']} → "
            f"observation_chars={self._stats['n_observation_chars']} → "
            f"reflection_chars={self._stats['n_reflection_chars']}.",
            "",
            "---",
            "",
        ]
        if self.reflections:
            md.append("## Reflections (condensed)\n")
            for i, r in enumerate(self.reflections):
                md.append(f"### Reflection #{i} ({r.n_observations_consumed} obs consumed)")
                md.append(r.markdown.strip())
                md.append("")
        if self.observations:
            md.append("\n## Observations\n")
            for i, o in enumerate(self.observations):
                md.append(f"### Block #{i}  ·  {o.date_label}  "
                          f"·  {len(o.source_uids)} source turns")
                md.append(o.markdown.strip())
                if o.current_task:
                    md.append(f"_current_task_: {o.current_task}")
                if o.suggested_response:
                    md.append(f"_suggested_response_: {o.suggested_response}")
                md.append("")
        recent = self._all_turns[-self.keep_recent_n:]
        if recent:
            md.append("\n## Recent raw turns (verbatim window)\n")
            for t in recent:
                md.append(f"- **{t.metadata.get('speaker', '?')}** "
                          f"`{t.metadata.get('dia_id', '?')}`: {t.text}")
        return "\n".join(md)

    def stats(self) -> dict:
        return dict(self._stats)

    # ── Persistence ──
    def save(self, path: str) -> None:
        """Pickle the in-memory state. Embeddings cached separately on disk."""
        import pickle
        from pathlib import Path as _P
        _P(path).parent.mkdir(parents=True, exist_ok=True)
        state = {
            "observer_token_threshold": self.observer_token_threshold,
            "reflector_token_threshold": self.reflector_token_threshold,
            "keep_recent_n": self.keep_recent_n,
            "raw_buffer": self._raw_buffer,
            "all_turns": self._all_turns,
            "observations": self.observations,
            "reflections": self.reflections,
            "stats": self._stats,
        }
        with open(path, "wb") as f:
            pickle.dump(state, f)

    @classmethod
    def load(cls, path: str) -> "MastraBackend":
        import pickle
        with open(path, "rb") as f:
            state = pickle.load(f)
        b = cls(
            observer_token_threshold=state["observer_token_threshold"],
            reflector_token_threshold=state["reflector_token_threshold"],
            keep_recent_n=state["keep_recent_n"],
        )
        b._raw_buffer = state["raw_buffer"]
        b._all_turns = state["all_turns"]
        b.observations = state["observations"]
        b.reflections = state["reflections"]
        b._stats = state["stats"]
        return b

    # ── Internal: ingestion + observer/reflector loop ──
    def _ingest_one(self, unit: Unit) -> None:
        t = _RawTurn(uid=unit.uid, text=unit.text, metadata=dict(unit.metadata))
        self._raw_buffer.append(t)
        self._all_turns.append(t)
        self._stats["n_raw_turns_seen"] += 1
        self._stats["n_raw_chars_seen"] += len(t.text)
        self._maybe_run_observer()
        self._maybe_run_reflector()

    def _buffer_tokens(self) -> int:
        return sum(_estimate_tokens(t.text) for t in self._raw_buffer)

    def _observation_tokens(self) -> int:
        return sum(_estimate_tokens(o.markdown) for o in self.observations)

    def _maybe_run_observer(self, force: bool = False) -> None:
        tok = self._buffer_tokens()
        if not self._raw_buffer:
            return
        if (not force) and tok < self.observer_token_threshold:
            return
        block = self._call_observer(self._raw_buffer)
        if block:
            self.observations.append(block)
            self._stats["n_observer_runs"] += 1
            self._stats["n_observation_chars"] = sum(
                len(o.markdown) for o in self.observations
            )
            self._obs_index_dirty = True
        # whether we got a block or not, drain the buffer to avoid loops
        self._raw_buffer = []

    def _maybe_run_reflector(self) -> None:
        if self._observation_tokens() < self.reflector_token_threshold:
            return
        block = self._call_reflector(self.observations)
        if block:
            self.reflections.append(block)
            self._stats["n_reflector_runs"] += 1
            self._stats["n_reflection_chars"] = sum(
                len(r.markdown) for r in self.reflections
            )
            # Mastra keeps a rolling tail; drop the consumed observations.
            keep_tail = max(2, len(self.observations) // 4)
            self.observations = self.observations[-keep_tail:]

    # ── LLM calls ──
    @staticmethod
    def _format_buffer_for_observer(buffer: list[_RawTurn]) -> str:
        lines = []
        for t in buffer:
            speaker = t.metadata.get("speaker", "?")
            dia = t.metadata.get("dia_id", "?")
            date = t.metadata.get("session_date", "")
            lines.append(f"[{dia} | {date} | {speaker}] {t.text}")
        return "\n".join(lines)

    def _call_observer(self, buffer: list[_RawTurn]) -> ObservationBlock | None:
        block_in = self._format_buffer_for_observer(buffer)
        try:
            md = llm.chat(
                [
                    {"role": "system", "content": _OBSERVER_SYS},
                    {"role": "user", "content":
                        "Conversation chunk to observe:\n\n" + block_in +
                        "\n\nReturn only the dated bullet observations + "
                        "Current task + Suggested response."},
                ],
                model=config.OBSERVER_MODEL,
            ).strip()
        except Exception as e:
            print(f"[mastra] observer error: {e}")
            return None
        if not md:
            return None
        # Pull current_task / suggested_response if present (best-effort regex)
        ct_m = re.search(r"Current task:\s*(.+?)(?:\n|$)", md, re.IGNORECASE)
        sr_m = re.search(r"Suggested response:\s*(.+?)(?:\n|$)", md, re.IGNORECASE)
        date_m = re.search(r"Date:\s*(\S.+?)(?:\n|$)", md)
        return ObservationBlock(
            date_label=(date_m.group(1).strip() if date_m else ""),
            markdown=md,
            source_uids=[t.uid for t in buffer],
            current_task=(ct_m.group(1).strip() if ct_m else ""),
            suggested_response=(sr_m.group(1).strip() if sr_m else ""),
            n_input_chars=len(block_in),
            n_output_chars=len(md),
        )

    def _call_reflector(self, observations: list[ObservationBlock]) -> ReflectionBlock | None:
        block_in = "\n\n---\n\n".join(o.markdown for o in observations)
        try:
            md = llm.chat(
                [
                    {"role": "system", "content": _REFLECTOR_SYS},
                    {"role": "user", "content":
                        "Observations to consolidate:\n\n" + block_in +
                        "\n\nReturn only condensed dated bullet markdown."},
                ],
                model=config.REFLECTOR_MODEL,
            ).strip()
        except Exception as e:
            print(f"[mastra] reflector error: {e}")
            return None
        if not md:
            return None
        return ReflectionBlock(
            markdown=md,
            n_observations_consumed=len(observations),
            n_input_chars=len(block_in),
            n_output_chars=len(md),
        )
