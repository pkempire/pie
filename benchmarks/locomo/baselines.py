"""
LoCoMo Baselines — comparison approaches for the benchmark.

Three baselines:
  1. full_context  — stuff all sessions into context, ask LLM directly
  2. naive_rag     — embed chunks, retrieve top-k by similarity, answer
  3. pie_temporal  — PIE's approach: ingest → build world model → compile context → answer

Each baseline takes a QA item and returns an answer string.

LoCoMo differences from LongMemEval:
  - Peer-to-peer chat (both speakers are humans)
  - Much longer conversations (~300 turns, up to 35 sessions)
  - 5 question types: single-hop, multi-hop, temporal, adversarial, commonsense
"""

from __future__ import annotations
import logging
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Any

from pie.core.llm import LLMClient
from pie.core.world_model import WorldModel, cosine_similarity
from pie.core.models import (
    Conversation, Turn, EntityType, TransitionType,
)

from .adapter import (
    sample_to_conversations,
    format_conversation_as_text,
    format_date_for_context,
    parse_locomo_date,
    get_session_observations,
    get_session_summaries,
)

logger = logging.getLogger("pie.bench.locomo")


# ── Shared Types ──────────────────────────────────────────────────────────────


@dataclass
class BaselineResult:
    """Result from running a baseline on one question."""
    question_id: str
    question_type: str
    question: str
    gold_answer: str
    hypothesis: str
    baseline_name: str
    model: str
    latency_ms: float = 0.0
    context_chars: int = 0
    retrieval_count: int = 0
    error: str | None = None

    def to_dict(self) -> dict:
        return {
            "question_id": self.question_id,
            "question_type": self.question_type,
            "question": self.question,
            "gold_answer": self.gold_answer,
            "hypothesis": self.hypothesis,
            "baseline_name": self.baseline_name,
            "model": self.model,
            "latency_ms": round(self.latency_ms, 1),
            "context_chars": self.context_chars,
            "retrieval_count": self.retrieval_count,
            "error": self.error,
        }


# ── QA Prompt ─────────────────────────────────────────────────────────────────

ANSWER_SYSTEM_PROMPT = """\
You are a helpful assistant answering questions about a conversation between two people.
You will be given context from their chat history, and a question.
Answer the question based ONLY on the provided context.
Be concise and specific. If the context doesn't contain the answer, say "I don't know."
Do NOT make up information."""

ANSWER_USER_TEMPLATE = """\
Conversation history:

{context}

---

Question: {question}

Answer concisely:"""


def _ask_llm(
    context: str,
    question: str,
    llm: LLMClient,
    model: str = "gpt-4o",
) -> str:
    """Ask an LLM to answer a question given context."""
    messages = [
        {"role": "system", "content": ANSWER_SYSTEM_PROMPT},
        {
            "role": "user",
            "content": ANSWER_USER_TEMPLATE.format(
                context=context,
                question=question,
            ),
        },
    ]
    result = llm.chat(messages=messages, model=model, max_tokens=300)
    return result["content"].strip()


# ═══════════════════════════════════════════════════════════════════════════════
# Baseline 1: Full Context
# ═══════════════════════════════════════════════════════════════════════════════


def full_context(
    item: dict[str, Any],
    llm: LLMClient | None = None,
    model: str = "gpt-4o",
    max_context_chars: int = 120_000,
) -> BaselineResult:
    """
    Full-context baseline: stuff ALL conversation into the prompt.
    """
    llm = llm or LLMClient()
    t0 = time.time()

    try:
        context = format_conversation_as_text(
            item["conversation"],
            max_chars=max_context_chars,
        )

        answer = _ask_llm(
            context=context,
            question=item["question"],
            llm=llm,
            model=model,
        )

        return BaselineResult(
            question_id=item["question_id"],
            question_type=item["question_type"],
            question=item["question"],
            gold_answer=item["answer"],
            hypothesis=answer,
            baseline_name="full_context",
            model=model,
            latency_ms=(time.time() - t0) * 1000,
            context_chars=len(context),
        )

    except Exception as e:
        return BaselineResult(
            question_id=item["question_id"],
            question_type=item["question_type"],
            question=item["question"],
            gold_answer=item["answer"],
            hypothesis=f"Error: {e}",
            baseline_name="full_context",
            model=model,
            latency_ms=(time.time() - t0) * 1000,
            error=str(e),
        )


# ═══════════════════════════════════════════════════════════════════════════════
# Baseline 2: Naive RAG
# ═══════════════════════════════════════════════════════════════════════════════


def naive_rag(
    item: dict[str, Any],
    llm: LLMClient | None = None,
    model: str = "gpt-4o",
    embed_model: str = "text-embedding-3-large",
    top_k: int = 10,
    use_observations: bool = False,
) -> BaselineResult:
    """
    Naive RAG baseline: embed chunks, retrieve top-k by cosine similarity.

    Can use either raw dialog text or pre-computed observations.
    """
    llm = llm or LLMClient()
    t0 = time.time()

    try:
        # Build chunks
        chunks = _build_rag_chunks(
            item["conversation"],
            use_observations=use_observations,
        )

        if not chunks:
            return BaselineResult(
                question_id=item["question_id"],
                question_type=item["question_type"],
                question=item["question"],
                gold_answer=item["answer"],
                hypothesis="No chunks to search.",
                baseline_name="naive_rag",
                model=model,
                latency_ms=(time.time() - t0) * 1000,
            )

        # Embed question and chunks
        query_emb = llm.embed_single(item["question"], model=embed_model)

        chunk_texts = [c["text"] for c in chunks]
        chunk_embeddings = _batch_embed(chunk_texts, llm, embed_model)

        # Rank by similarity
        scored = []
        for chunk, emb in zip(chunks, chunk_embeddings):
            sim = cosine_similarity(query_emb, emb)
            scored.append((chunk, sim))

        scored.sort(key=lambda x: x[1], reverse=True)
        top_chunks = scored[:top_k]

        # Compile context
        top_chunks_sorted = sorted(top_chunks, key=lambda x: x[0].get("timestamp", 0))
        context_parts = []
        for chunk, score in top_chunks_sorted:
            date = chunk.get("date", "Unknown date")
            context_parts.append(f"[{date} | relevance: {score:.2f}]\n{chunk['text']}")
        context = "\n\n".join(context_parts)

        # Ask LLM
        answer = _ask_llm(
            context=context,
            question=item["question"],
            llm=llm,
            model=model,
        )

        return BaselineResult(
            question_id=item["question_id"],
            question_type=item["question_type"],
            question=item["question"],
            gold_answer=item["answer"],
            hypothesis=answer,
            baseline_name="naive_rag",
            model=model,
            latency_ms=(time.time() - t0) * 1000,
            context_chars=len(context),
            retrieval_count=len(top_chunks),
        )

    except Exception as e:
        return BaselineResult(
            question_id=item["question_id"],
            question_type=item["question_type"],
            question=item["question"],
            gold_answer=item["answer"],
            hypothesis=f"Error: {e}",
            baseline_name="naive_rag",
            model=model,
            latency_ms=(time.time() - t0) * 1000,
            error=str(e),
        )


def _build_rag_chunks(
    conversation: dict[str, Any],
    use_observations: bool = False,
) -> list[dict]:
    """Build text chunks for RAG retrieval."""
    chunks = []

    session_keys = sorted(
        [k for k in conversation.keys() if k.startswith("session_") and not k.endswith("_date_time")],
        key=lambda x: int(x.split("_")[1])
    )

    for i, session_key in enumerate(session_keys):
        session = conversation[session_key]
        date_key = f"{session_key}_date_time"
        date_str = conversation.get(date_key, "")

        try:
            timestamp = parse_locomo_date(date_str) if date_str else 0
        except Exception:
            timestamp = 0

        human_date = format_date_for_context(date_str) if date_str else f"Session {i + 1}"

        # Format session as chunk
        lines = []
        for turn in session:
            name = turn.get("name", "Unknown")
            text = turn.get("text", "").strip()
            if text:
                lines.append(f"{name}: {text}")

        if lines:
            chunks.append({
                "text": "\n".join(lines),
                "date": human_date,
                "timestamp": timestamp,
                "session_index": i,
            })

    return chunks


def _batch_embed(
    texts: list[str],
    llm: LLMClient,
    model: str = "text-embedding-3-large",
    batch_size: int = 512,
) -> list[list[float]]:
    """Embed texts in batches."""
    all_embeddings = []
    for i in range(0, len(texts), batch_size):
        batch = texts[i : i + batch_size]
        batch = [t[:8000] for t in batch]  # truncate
        embeddings = llm.embed(batch, model=model)
        all_embeddings.extend(embeddings)
    return all_embeddings


# ═══════════════════════════════════════════════════════════════════════════════
# Baseline 3: PIE Temporal
# ═══════════════════════════════════════════════════════════════════════════════


def pie_temporal(
    item: dict[str, Any],
    world_model: WorldModel | None = None,
    llm: LLMClient | None = None,
    model: str = "gpt-4o",
    extraction_model: str = "gpt-4o-mini",
    top_k_entities: int = 15,
    max_context_chars: int = 12_000,
) -> BaselineResult:
    """
    PIE's temporal approach:
      1. Ingest conversation sessions into world model
      2. Build temporal knowledge graph
      3. Retrieve relevant entities
      4. Compile semantic temporal context
      5. Answer using context
    """
    llm = llm or LLMClient()
    t0 = time.time()

    try:
        if world_model is None:
            world_model = _build_world_model_for_conversation(
                item, llm, extraction_model
            )

        # Retrieve relevant entities
        question = item["question"]

        retrieved = _retrieve_entities_for_question(
            question=question,
            world_model=world_model,
            llm=llm,
            top_k=top_k_entities,
        )

        if not retrieved:
            return BaselineResult(
                question_id=item["question_id"],
                question_type=item["question_type"],
                question=question,
                gold_answer=item["answer"],
                hypothesis="I don't have enough information.",
                baseline_name="pie_temporal",
                model=model,
                latency_ms=(time.time() - t0) * 1000,
            )

        # Compile temporal context
        context = _compile_temporal_context(
            retrieved=retrieved,
            world_model=world_model,
            max_chars=max_context_chars,
        )

        # Ask LLM
        answer = _ask_llm_temporal(
            context=context,
            question=question,
            llm=llm,
            model=model,
        )

        return BaselineResult(
            question_id=item["question_id"],
            question_type=item["question_type"],
            question=question,
            gold_answer=item["answer"],
            hypothesis=answer,
            baseline_name="pie_temporal",
            model=model,
            latency_ms=(time.time() - t0) * 1000,
            context_chars=len(context),
            retrieval_count=len(retrieved),
        )

    except Exception as e:
        logger.exception(f"PIE temporal failed for {item['question_id']}")
        return BaselineResult(
            question_id=item["question_id"],
            question_type=item["question_type"],
            question=item["question"],
            gold_answer=item["answer"],
            hypothesis=f"Error: {e}",
            baseline_name="pie_temporal",
            model=model,
            latency_ms=(time.time() - t0) * 1000,
            error=str(e),
        )


# ── PIE Helpers ──────────────────────────────────────────────────────────────

PIE_EXTRACTION_PROMPT = """\
You are extracting structured knowledge from a conversation between two people.

Extract:
1. **Entities**: People, places, events, preferences, activities.
   - name: canonical name
   - type: person|project|tool|organization|belief|concept|preference|event
   - state: dict of key-value attributes
   - For EVENTS, include: {date: "YYYY-MM-DD", description: "..."}

2. **State changes**: If something changed from a previous value.
   - entity_name, what_changed, old_state, new_state, is_contradiction

3. **Relationships**: How entities relate.
   - source, target, type, description

Focus on FACTS about the speakers' lives, preferences, activities.
Skip generic chit-chat.

Output JSON:
{
  "entities": [...],
  "state_changes": [...],
  "relationships": []
}"""


def _build_world_model_for_conversation(
    item: dict[str, Any],
    llm: LLMClient,
    extraction_model: str = "gpt-4o-mini",
) -> WorldModel:
    """Build a world model from a conversation's sessions."""
    import json

    wm = WorldModel()

    # Convert to PIE conversations
    # Create a minimal sample structure
    sample = {
        "sample_id": item.get("sample_id", item["question_id"]),
        "conversation": item["conversation"],
    }
    conversations = sample_to_conversations(sample)

    # Process in batches
    batch_size = 5
    for batch_start in range(0, len(conversations), batch_size):
        batch = conversations[batch_start : batch_start + batch_size]

        batch_text = _format_conversations_for_extraction(batch)

        try:
            result = llm.chat(
                messages=[
                    {"role": "system", "content": PIE_EXTRACTION_PROMPT},
                    {"role": "user", "content": batch_text},
                ],
                model=extraction_model,
                json_mode=True,
            )

            raw = result["content"]
            if isinstance(raw, str):
                raw = json.loads(raw)

            _apply_extraction_to_world_model(
                raw, wm, batch[0].created_at, batch[0].id
            )

        except Exception as e:
            logger.warning(f"Extraction failed for batch {batch_start}: {e}")
            continue

    return wm


def _format_conversations_for_extraction(conversations: list[Conversation]) -> str:
    """Format conversations for extraction prompt."""
    parts = []
    for convo in conversations:
        dt = datetime.fromtimestamp(convo.created_at, tz=timezone.utc)
        date_str = dt.strftime("%B %d, %Y at %H:%M")
        parts.append(f"[Session — {date_str}]")
        for turn in convo.turns:
            role = turn.role.capitalize()
            text = turn.text[:2000]
            parts.append(f"{role}: {text}")
        parts.append("")
    return "\n".join(parts)


def _apply_extraction_to_world_model(
    raw: dict,
    wm: WorldModel,
    timestamp: float,
    convo_id: str,
) -> None:
    """Apply extraction output to world model."""
    for entity_data in raw.get("entities", []):
        name = entity_data.get("name", "")
        if not name:
            continue

        etype_str = entity_data.get("type", "concept").lower()
        try:
            etype = EntityType(etype_str)
        except ValueError:
            type_map = {
                "preference": EntityType.BELIEF,
                "place": EntityType.CONCEPT,
                "location": EntityType.CONCEPT,
                "event": EntityType.CONCEPT,
            }
            etype = type_map.get(etype_str, EntityType.CONCEPT)

        state = entity_data.get("state", {})
        if isinstance(state, str):
            state = {"description": state}

        existing = wm.find_by_name(name)
        if existing:
            wm.update_entity_state(
                entity_id=existing.id,
                new_state=state,
                source_conversation_id=convo_id,
                timestamp=timestamp,
                trigger_summary="Updated from session",
            )
        else:
            wm.create_entity(
                name=name,
                type=etype,
                state=state,
                source_conversation_id=convo_id,
                timestamp=timestamp,
            )


def _retrieve_entities_for_question(
    question: str,
    world_model: WorldModel,
    llm: LLMClient,
    top_k: int = 15,
) -> list[tuple[str, Any, float]]:
    """Retrieve relevant entities via embedding similarity."""
    if not world_model.entities:
        return []

    query_emb = llm.embed_single(question)

    # Compute missing embeddings
    needs_embed = []
    needs_embed_ids = []
    for eid, entity in world_model.entities.items():
        if entity.embedding is None:
            state = entity.current_state
            desc = state.get("description", str(state)[:200]) if isinstance(state, dict) else str(state)[:200]
            text = f"{entity.name} ({entity.type.value}): {desc}"
            needs_embed.append(text)
            needs_embed_ids.append(eid)

    if needs_embed:
        try:
            embeddings = _batch_embed(needs_embed, llm)
            for eid, emb in zip(needs_embed_ids, embeddings):
                world_model.entities[eid].embedding = emb
        except Exception as e:
            logger.warning(f"Batch embedding failed: {e}")

    # Score entities
    scored = []
    for eid, entity in world_model.entities.items():
        if entity.embedding:
            sim = cosine_similarity(query_emb, entity.embedding)
            scored.append((eid, entity, sim))

    scored.sort(key=lambda x: x[2], reverse=True)
    return [(eid, e, s) for eid, e, s in scored[:top_k]]


def _compile_temporal_context(
    retrieved: list[tuple[str, Any, float]],
    world_model: WorldModel,
    max_chars: int = 12_000,
) -> str:
    """Compile PIE's semantic temporal context."""
    parts = []
    total_chars = 0

    for eid, entity, relevance in retrieved:
        transitions = world_model.get_transitions(eid, ordered=True)

        lines = []
        name = entity.name
        etype = entity.type.value

        lines.append(f"## {name} ({etype})")

        # Current state
        state = entity.current_state
        if state:
            desc = state.get("description", "")
            if not desc and isinstance(state, dict):
                desc = "; ".join(f"{k}: {v}" for k, v in state.items() if v)
            if desc:
                lines.append(f"Current: {desc}")

        # Timeline
        if transitions and len(transitions) > 1:
            lines.append("\nTimeline:")
            for t in transitions:
                dt = datetime.fromtimestamp(t.timestamp, tz=timezone.utc)
                date_str = dt.strftime("%Y-%m-%d")
                ttype = t.transition_type
                prefix = "  ⚠" if ttype == TransitionType.CONTRADICTION else "  •"
                if t.trigger_summary:
                    lines.append(f"{prefix} {date_str}: {t.trigger_summary}")

        part = "\n".join(lines)

        if total_chars + len(part) > max_chars:
            break

        parts.append(part)
        total_chars += len(part)

    return "\n\n".join(parts)


PIE_ANSWER_SYSTEM = """\
You are answering questions about a conversation between two people.
You are given structured knowledge extracted from their chat history.

Use the temporal information to answer time-sensitive questions accurately.
Pay attention to:
- The MOST RECENT state when asking about "current" info
- State changes marked with ⚠ — the NEWER value is correct
- Chronological ordering for temporal questions

Answer based ONLY on the provided context. Be concise.
If the answer isn't in the context, say "I don't know."
Do NOT make up information."""


def _ask_llm_temporal(
    context: str,
    question: str,
    llm: LLMClient,
    model: str = "gpt-4o",
) -> str:
    """Ask LLM with PIE's temporal context."""
    messages = [
        {"role": "system", "content": PIE_ANSWER_SYSTEM},
        {
            "role": "user",
            "content": f"Knowledge base:\n\n{context}\n\n---\n\nQuestion: {question}\n\nAnswer:",
        },
    ]
    result = llm.chat(messages=messages, model=model, max_tokens=300)
    return result["content"].strip()


# ═══════════════════════════════════════════════════════════════════════════════
# Baseline registry
# ═══════════════════════════════════════════════════════════════════════════════

BASELINES = {
    "full_context": full_context,
    "naive_rag": naive_rag,
    "pie_temporal": pie_temporal,
}
