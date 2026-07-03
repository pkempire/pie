"""Future-query eval for a generic active-memory manager.

This is the closest current script to the actual continual-learning product:

  past raw episodes -> budgeted active memory bank
  future user turns -> answer using active memory + optional raw-log search
  reward/report     -> task score, active-memory size, raw retrieval cost

The storage format is intentionally plain text with provenance. There is no
domain schema for projects/science/sales/etc. The model decides what compressed
state to keep, while the raw event log remains immutable and searchable.

Use `--dry-run` first. Full runs make many LLM calls.
"""
from __future__ import annotations

import argparse
import json
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

from mempol import config, llm
from mempol.backends.base import Hit, Unit
from mempol.backends.flat import FlatBackend
from mempol.eval.judge import judge
from mempol.policies.v0_naive import answer_with_context
from mempol.scripts.future_eval import (
    ChronoTurn,
    _answer_backend,
    _build_raw,
    _estimate_tokens,
    _is_eval_query,
    _load_turns,
    _not_in_context,
    _retrieved_chars,
)


@dataclass
class ActiveMemory:
    id: str
    text: str
    source_ids: list[str] = field(default_factory=list)
    created_episode: int = 0
    updated_episode: int = 0
    archived: bool = False


@dataclass
class Episode:
    idx: int
    start_turn_idx: int
    end_turn_idx: int
    title: str
    text: str
    source_ids: list[str]


_WRITE_SYS = """You manage a small active memory bank for a long-running AI agent.

The raw event log is never deleted and can be searched later. Active memory is
only the compressed state worth keeping hot because it may help future tasks.

Do not use a fixed schema. Write compact natural-language memories with source
ids. Prefer information that changes what the agent should do later, prevents
duplicated work, preserves decisions/evidence, or keeps track of stale/open
threads. Merge/update existing memories instead of creating duplicates.

Return strict JSON:
{
  "edits": [
    {"op":"create","text":"...","source_ids":["..."],"reason":"..."},
    {"op":"update","id":"mem_1","text":"...","source_ids":["..."],"reason":"..."},
    {"op":"archive","id":"mem_2","reason":"..."},
    {"op":"noop","reason":"..."}
  ]
}
"""


_COMPRESS_SYS = """Rewrite an active memory bank to fit a token budget.

Keep the most useful future-task state. Merge duplicates. Preserve source ids.
Drop weak, stale, or unsupported notes. Do not introduce unsupported facts.

Return strict JSON:
{"memories":[{"text":"...","source_ids":["..."]}]}
"""


_READ_PLAN_SYS = """You answer a future user turn using active memory and, if useful, raw-log search.

Decide whether raw-log search is needed. Active memory is compressed and may be
enough; raw search costs tokens, so only request it when needed.

Return strict JSON:
{"need_raw": bool, "raw_queries": ["search query 1", "search query 2"], "reason": "..."}
"""


_ANSWER_SYS = """Answer the future user turn using active memory and raw evidence.

If the evidence is insufficient, say "not in context". Be specific and cite
memory/source ids in prose when useful. Do not invent unsupported facts.
"""


def _load_json(raw: str, default: Any) -> Any:
    try:
        return json.loads(raw)
    except Exception:
        return default


def _turn_label(t: ChronoTurn) -> str:
    return f"{t.conv_id}::T{t.idx}"


def _format_turn(t: ChronoTurn, max_chars: int = 900) -> str:
    stamp = time.strftime("%Y-%m-%d", time.gmtime(t.timestamp)) if t.timestamp else ""
    text = t.text.replace("\n", " ").strip()
    if len(text) > max_chars:
        text = text[:max_chars] + " ..."
    return f"[{_turn_label(t)} | {stamp} | {t.role} | {t.title[:80]}] {text}"


def build_episodes(
    turns: list[ChronoTurn],
    episode_turns: int,
    max_episode_chars: int,
) -> list[Episode]:
    episodes: list[Episode] = []
    cur: list[ChronoTurn] = []
    cur_chars = 0

    def flush() -> None:
        if not cur:
            return
        idx = len(episodes)
        text = "\n".join(_format_turn(t) for t in cur)
        title = cur[-1].title or cur[0].title or f"episode {idx}"
        episodes.append(Episode(
            idx=idx,
            start_turn_idx=cur[0].idx,
            end_turn_idx=cur[-1].idx,
            title=title,
            text=text,
            source_ids=[_turn_label(t) for t in cur],
        ))

    for t in turns:
        t_chars = len(t.text)
        if cur and (len(cur) >= episode_turns or cur_chars + t_chars > max_episode_chars):
            flush()
            cur = []
            cur_chars = 0
        cur.append(t)
        cur_chars += t_chars
    flush()
    return episodes


class ActiveMemoryManager:
    def __init__(self, token_budget: int, model: str) -> None:
        self.token_budget = token_budget
        self.model = model
        self.memories: list[ActiveMemory] = []
        self.trace: list[dict] = []
        self._next_id = 1

    def active(self) -> list[ActiveMemory]:
        return [m for m in self.memories if not m.archived]

    def active_text(self, limit_chars: int = 12000) -> str:
        rows = []
        for m in self.active():
            src = ",".join(m.source_ids[:8])
            rows.append(f"{m.id}: {m.text}\n  sources: {src}")
        text = "\n\n".join(rows) or "(empty)"
        return text[:limit_chars]

    def active_tokens(self) -> int:
        return _estimate_tokens(self.active_text(limit_chars=10_000_000))

    def ingest_episode(self, ep: Episode) -> None:
        raw = llm.chat(
            [
                {"role": "system", "content": _WRITE_SYS},
                {
                    "role": "user",
                    "content": (
                        f"Token budget for active memory: {self.token_budget}\n\n"
                        f"Current active memory:\n{self.active_text()}\n\n"
                        f"New episode {ep.idx} ({ep.start_turn_idx}-{ep.end_turn_idx}):\n"
                        f"{ep.text}\n\n"
                        "Return memory edits."
                    ),
                },
            ],
            model=self.model,
            json_mode=True,
        )
        obj = _load_json(raw, {"edits": [{"op": "noop", "reason": "json_parse_failed"}]})
        edits = obj.get("edits") or []
        applied = self.apply_edits(edits, ep.idx)
        before_compress = self.active_tokens()
        compressed = False
        if before_compress > self.token_budget:
            self.compress(ep.idx)
            compressed = True
        self.trace.append({
            "event": "ingest_episode",
            "episode_idx": ep.idx,
            "turn_range": [ep.start_turn_idx, ep.end_turn_idx],
            "applied": applied,
            "active_memories": len(self.active()),
            "active_tokens": self.active_tokens(),
            "compressed": compressed,
        })

    def apply_edits(self, edits: list[dict], episode_idx: int) -> list[dict]:
        applied: list[dict] = []
        by_id = {m.id: m for m in self.memories}
        for edit in edits:
            op = str(edit.get("op", "noop")).lower()
            if op == "create":
                text = str(edit.get("text", "")).strip()
                if not text:
                    continue
                mem = ActiveMemory(
                    id=f"mem_{self._next_id}",
                    text=text,
                    source_ids=[str(s) for s in edit.get("source_ids", [])],
                    created_episode=episode_idx,
                    updated_episode=episode_idx,
                )
                self._next_id += 1
                self.memories.append(mem)
                applied.append({"op": "create", "id": mem.id})
            elif op == "update":
                mem = by_id.get(str(edit.get("id", "")))
                text = str(edit.get("text", "")).strip()
                if mem is None or not text:
                    continue
                mem.text = text
                mem.updated_episode = episode_idx
                mem.source_ids = sorted(set(mem.source_ids + [str(s) for s in edit.get("source_ids", [])]))
                applied.append({"op": "update", "id": mem.id})
            elif op == "archive":
                mem = by_id.get(str(edit.get("id", "")))
                if mem is None:
                    continue
                mem.archived = True
                mem.updated_episode = episode_idx
                applied.append({"op": "archive", "id": mem.id})
            elif op == "noop":
                applied.append({"op": "noop", "reason": str(edit.get("reason", ""))[:160]})
        return applied

    def compress(self, episode_idx: int) -> None:
        raw = llm.chat(
            [
                {"role": "system", "content": _COMPRESS_SYS},
                {
                    "role": "user",
                    "content": (
                        f"Budget: {self.token_budget} tokens.\n\n"
                        f"Current active memory:\n{self.active_text(limit_chars=30000)}"
                    ),
                },
            ],
            model=self.model,
            json_mode=True,
        )
        obj = _load_json(raw, {"memories": []})
        memories = obj.get("memories") or []
        for m in self.active():
            m.archived = True
            m.updated_episode = episode_idx
        for item in memories:
            text = str(item.get("text", "")).strip()
            if not text:
                continue
            mem = ActiveMemory(
                id=f"mem_{self._next_id}",
                text=text,
                source_ids=[str(s) for s in item.get("source_ids", [])],
                created_episode=episode_idx,
                updated_episode=episode_idx,
            )
            self._next_id += 1
            self.memories.append(mem)


def answer_with_active_memory(
    query: str,
    manager: ActiveMemoryManager,
    raw_backend: FlatBackend,
    raw_k: int,
    model: str,
) -> tuple[str, list[Hit], dict]:
    plan_raw = llm.chat(
        [
            {"role": "system", "content": _READ_PLAN_SYS},
            {
                "role": "user",
                "content": f"Active memory:\n{manager.active_text()}\n\nFuture user turn:\n{query}",
            },
        ],
        model=model,
        json_mode=True,
    )
    plan = _load_json(plan_raw, {"need_raw": True, "raw_queries": [query], "reason": "json_parse_failed"})
    raw_hits: list[Hit] = []
    if plan.get("need_raw", True):
        raw_queries = [str(q).strip() for q in plan.get("raw_queries", []) if str(q).strip()]
        if not raw_queries:
            raw_queries = [query]
        seen = set()
        for rq in raw_queries[:3]:
            for h in raw_backend.retrieve(rq, k=raw_k, source="hybrid"):
                if h.unit.uid in seen:
                    continue
                seen.add(h.unit.uid)
                raw_hits.append(h)
    evidence = "\n".join(
        f"[{h.unit.uid} | {h.unit.metadata.get('title','')}] {h.unit.text}"
        for h in raw_hits
    )
    answer = llm.chat(
        [
            {"role": "system", "content": _ANSWER_SYS},
            {
                "role": "user",
                "content": (
                    f"Active memory:\n{manager.active_text(limit_chars=18000)}\n\n"
                    f"Raw evidence retrieved:\n{evidence or '(none)'}\n\n"
                    f"Future user turn:\n{query}\n\nAnswer:"
                ),
            },
        ],
        model=model,
    ).strip()
    return answer, raw_hits, plan


def _write_memory_md(path: Path, memories: list[ActiveMemory]) -> None:
    lines = ["# Active Memory Bank", ""]
    for m in memories:
        status = "archived" if m.archived else "active"
        lines.append(f"## {m.id} · {status}")
        lines.append(m.text)
        if m.source_ids:
            lines.append(f"sources: {', '.join(m.source_ids[:20])}")
        lines.append("")
    path.write_text("\n".join(lines))


def run(
    conversations_json: Path,
    out_dir: Path,
    year_min: int,
    cutoff_turns: int,
    future_turns: int,
    max_queries: int,
    episode_turns: int,
    max_episode_chars: int,
    max_episodes: int,
    memory_budget_tokens: int,
    teacher_k: int,
    raw_k: int,
    model: str,
    dry_run: bool,
) -> dict:
    turns = _load_turns(conversations_json, year_min=year_min)
    if cutoff_turns >= len(turns):
        raise ValueError(f"cutoff_turns={cutoff_turns} but only loaded {len(turns)} turns")

    past = turns[:cutoff_turns]
    future = turns[cutoff_turns:cutoff_turns + future_turns]
    episodes = build_episodes(past, episode_turns=episode_turns, max_episode_chars=max_episode_chars)
    if max_episodes > 0:
        episodes = episodes[:max_episodes]

    queries = [
        t for t in future
        if _is_eval_query(t, min_chars=12, max_chars=1200)
    ][:max_queries]

    out_dir.mkdir(parents=True, exist_ok=True)
    if dry_run:
        payload = {
            "dry_run": True,
            "loaded_turns": len(turns),
            "past_turns": len(past),
            "future_turns": len(future),
            "episodes": len(episodes),
            "episode_preview": [asdict(e) | {"text": e.text[:800]} for e in episodes[:3]],
            "queries": [{"idx": q.idx, "title": q.title, "text": q.text[:500]} for q in queries[:10]],
        }
        (out_dir / "summary.json").write_text(json.dumps(payload, indent=2))
        print(json.dumps(payload, indent=2))
        return payload

    raw_backend, raw_stats = _build_raw(past)
    manager = ActiveMemoryManager(token_budget=memory_budget_tokens, model=model)

    for ep in episodes:
        print(f"[memory] episode {ep.idx + 1}/{len(episodes)} turns={ep.start_turn_idx}-{ep.end_turn_idx}", flush=True)
        manager.ingest_episode(ep)
        print(
            f"  active={len(manager.active())} tokens={manager.active_tokens()}",
            flush=True,
        )

    rows = []
    rows_path = out_dir / "rows.jsonl"
    with rows_path.open("w", buffering=1) as f:
        for i, qt in enumerate(queries, start=1):
            teacher_answer, teacher_hits = _answer_backend(qt.text, raw_backend, k=teacher_k)
            if _not_in_context(teacher_answer):
                continue
            answer, raw_hits, plan = answer_with_active_memory(
                qt.text, manager, raw_backend, raw_k=raw_k, model=model
            )
            score, reason = judge(qt.text, teacher_answer, answer)
            row = {
                "query_i": i,
                "query_idx": qt.idx,
                "query": qt.text,
                "teacher_answer": teacher_answer,
                "answer": answer,
                "score": score,
                "judge_reason": reason,
                "read_plan": plan,
                "raw_retrieved_units": len(raw_hits),
                "raw_retrieved_chars": _retrieved_chars(raw_hits),
                "teacher_retrieved_chars": _retrieved_chars(teacher_hits),
                "active_memory_tokens": manager.active_tokens(),
                "active_memory_count": len(manager.active()),
            }
            rows.append(row)
            f.write(json.dumps(row, ensure_ascii=False) + "\n")
            print(f"[query {i}/{len(queries)}] score={score} raw_hits={len(raw_hits)} {qt.text[:80]}", flush=True)

    summary = {
        "n_queries_scored": len(rows),
        "avg_score_vs_raw_teacher": sum(r["score"] for r in rows) / max(1, len(rows)),
        "active_memory_tokens": manager.active_tokens(),
        "active_memory_count": len(manager.active()),
        "raw_past_tokens_est": raw_stats.stored_chars // 4,
        "active_to_raw_token_ratio": manager.active_tokens() / max(1, raw_stats.stored_chars // 4),
        "avg_raw_retrieved_tokens": (
            sum(_estimate_tokens("x" * r["raw_retrieved_chars"]) for r in rows) / max(1, len(rows))
        ),
        "config": {
            "conversations_json": str(conversations_json),
            "year_min": year_min,
            "cutoff_turns": cutoff_turns,
            "future_turns": future_turns,
            "max_queries": max_queries,
            "episode_turns": episode_turns,
            "max_episode_chars": max_episode_chars,
            "max_episodes": max_episodes,
            "memory_budget_tokens": memory_budget_tokens,
            "teacher_k": teacher_k,
            "raw_k": raw_k,
            "model": model,
        },
    }
    (out_dir / "summary.json").write_text(json.dumps(summary, indent=2))
    (out_dir / "memory_trace.jsonl").write_text("\n".join(json.dumps(x, ensure_ascii=False) for x in manager.trace) + "\n")
    _write_memory_md(out_dir / "active_memory.md", manager.memories)
    print(json.dumps(summary, indent=2))
    print(f"Wrote {out_dir}")
    return summary


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--conversations-json", required=True, type=Path)
    ap.add_argument("--out-dir", type=Path, default=config.RESULTS_DIR / "future_memory_agent")
    ap.add_argument("--year-min", type=int, default=2023)
    ap.add_argument("--cutoff-turns", type=int, default=1000)
    ap.add_argument("--future-turns", type=int, default=300)
    ap.add_argument("--max-queries", type=int, default=20)
    ap.add_argument("--episode-turns", type=int, default=24)
    ap.add_argument("--max-episode-chars", type=int, default=24000)
    ap.add_argument("--max-episodes", type=int, default=0, help="0 = all episodes before cutoff")
    ap.add_argument("--memory-budget-tokens", type=int, default=4000)
    ap.add_argument("--teacher-k", type=int, default=50)
    ap.add_argument("--raw-k", type=int, default=8)
    ap.add_argument("--model", default=config.ANSWER_MODEL)
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()
    run(**vars(args))


if __name__ == "__main__":
    main()
