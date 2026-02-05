"""
MSC (Multi-Session Chat) Baselines

Baselines for the MSC benchmark:
  1. full_context  — stuff all sessions into context
  2. naive_rag     — embed turns, retrieve top-k
  3. pie_temporal  — PIE's temporal knowledge graph approach

MSC focuses on:
  - Response generation (continuing conversation)
  - Persona consistency across sessions
  - Memory of facts from prior sessions
"""

from __future__ import annotations
import logging
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Any

from pie.core.llm import LLMClient
from pie.core.world_model import WorldModel, cosine_similarity
from pie.core.models import Conversation, Turn, EntityType, TransitionType

from .adapter import (
    parse_msc_example,
    format_conversations_as_text,
    format_personas,
)

logger = logging.getLogger("pie.bench.msc")


@dataclass
class BaselineResult:
    """Result from running a baseline."""
    item_id: str
    item_type: str
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
            "item_id": self.item_id,
            "item_type": self.item_type,
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


# ── Prompts ───────────────────────────────────────────────────────────────────

RESPONSE_SYSTEM = """\
You are continuing a multi-session conversation between two people.
They have been chatting over multiple days and know each other.
Generate a natural, persona-consistent response.
Be engaging and reference relevant context from prior conversations when appropriate."""

RESPONSE_USER = """\
{personas}

Conversation history:
{context}

---

Generate the next response in this conversation. Be natural and engaging."""


QA_SYSTEM = """\
You are answering questions about a multi-session conversation.
Answer based ONLY on the provided context.
Be concise. If the answer isn't in the context, say "I don't know."
Do NOT make up information."""

QA_USER = """\
{personas}

Conversation history:
{context}

---

Question: {question}

Answer:"""


def _generate_response(
    context: str,
    personas: list[str],
    llm: LLMClient,
    model: str = "gpt-4o",
) -> str:
    """Generate a continuation response."""
    persona_text = format_personas(personas)

    messages = [
        {"role": "system", "content": RESPONSE_SYSTEM},
        {
            "role": "user",
            "content": RESPONSE_USER.format(
                personas=persona_text,
                context=context,
            ),
        },
    ]
    result = llm.chat(messages=messages, model=model, max_tokens=200)
    return result["content"].strip()


def _answer_qa(
    context: str,
    question: str,
    personas: list[str],
    llm: LLMClient,
    model: str = "gpt-4o",
) -> str:
    """Answer a memory QA question."""
    persona_text = format_personas(personas)

    messages = [
        {"role": "system", "content": QA_SYSTEM},
        {
            "role": "user",
            "content": QA_USER.format(
                personas=persona_text,
                context=context,
                question=question,
            ),
        },
    ]
    result = llm.chat(messages=messages, model=model, max_tokens=200)
    return result["content"].strip()


# ═══════════════════════════════════════════════════════════════════════════════
# Baseline 1: Full Context
# ═══════════════════════════════════════════════════════════════════════════════


def full_context(
    item: dict[str, Any],
    llm: LLMClient | None = None,
    model: str = "gpt-4o",
    max_context_chars: int = 50_000,
) -> BaselineResult:
    """Full-context baseline: stuff all sessions into prompt."""
    llm = llm or LLMClient()
    t0 = time.time()

    try:
        context = item.get("context_text", "")
        if len(context) > max_context_chars:
            context = context[:max_context_chars] + "\n[... truncated ...]"

        personas = item.get("personas", [])
        item_type = item.get("item_type", "response_generation")

        if item_type == "response_generation":
            answer = _generate_response(
                context=context,
                personas=personas,
                llm=llm,
                model=model,
            )
        else:  # memory_qa
            answer = _answer_qa(
                context=context,
                question=item.get("question", ""),
                personas=personas,
                llm=llm,
                model=model,
            )

        return BaselineResult(
            item_id=item["item_id"],
            item_type=item_type,
            question=item.get("question", ""),
            gold_answer=item.get("expected_response", item.get("answer", "")),
            hypothesis=answer,
            baseline_name="full_context",
            model=model,
            latency_ms=(time.time() - t0) * 1000,
            context_chars=len(context),
        )

    except Exception as e:
        return BaselineResult(
            item_id=item["item_id"],
            item_type=item.get("item_type", "unknown"),
            question=item.get("question", ""),
            gold_answer=item.get("expected_response", item.get("answer", "")),
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
) -> BaselineResult:
    """Naive RAG baseline: retrieve relevant turns."""
    llm = llm or LLMClient()
    t0 = time.time()

    try:
        conversations = item.get("conversations", [])
        personas = item.get("personas", [])
        item_type = item.get("item_type", "response_generation")
        query = item.get("question", "")

        # Build chunks (one per turn)
        chunks = []
        for convo in conversations:
            dt = datetime.fromtimestamp(convo.created_at, tz=timezone.utc)
            date_str = dt.strftime("%B %d, %Y")
            for turn in convo.turns:
                chunks.append({
                    "text": f"{turn.role.capitalize()}: {turn.text}",
                    "date": date_str,
                    "timestamp": turn.timestamp,
                })

        if not chunks:
            return BaselineResult(
                item_id=item["item_id"],
                item_type=item_type,
                question=query,
                gold_answer=item.get("expected_response", item.get("answer", "")),
                hypothesis="No context available.",
                baseline_name="naive_rag",
                model=model,
                latency_ms=(time.time() - t0) * 1000,
            )

        # For response generation, use last few turns as query
        if item_type == "response_generation" and len(chunks) > 3:
            query = " ".join(c["text"] for c in chunks[-3:])

        # Embed and retrieve
        query_emb = llm.embed_single(query, model=embed_model)
        chunk_texts = [c["text"] for c in chunks]
        chunk_embs = llm.embed(chunk_texts[:500], model=embed_model)  # limit

        scored = []
        for chunk, emb in zip(chunks[:500], chunk_embs):
            sim = cosine_similarity(query_emb, emb)
            scored.append((chunk, sim))

        scored.sort(key=lambda x: x[1], reverse=True)
        top_chunks = scored[:top_k]

        # Build context
        top_sorted = sorted(top_chunks, key=lambda x: x[0].get("timestamp", 0))
        context_parts = []
        for chunk, score in top_sorted:
            context_parts.append(f"[{chunk['date']}] {chunk['text']}")
        context = "\n".join(context_parts)

        # Generate answer
        if item_type == "response_generation":
            answer = _generate_response(context, personas, llm, model)
        else:
            answer = _answer_qa(context, query, personas, llm, model)

        return BaselineResult(
            item_id=item["item_id"],
            item_type=item_type,
            question=query,
            gold_answer=item.get("expected_response", item.get("answer", "")),
            hypothesis=answer,
            baseline_name="naive_rag",
            model=model,
            latency_ms=(time.time() - t0) * 1000,
            context_chars=len(context),
            retrieval_count=len(top_chunks),
        )

    except Exception as e:
        return BaselineResult(
            item_id=item["item_id"],
            item_type=item.get("item_type", "unknown"),
            question=item.get("question", ""),
            gold_answer=item.get("expected_response", item.get("answer", "")),
            hypothesis=f"Error: {e}",
            baseline_name="naive_rag",
            model=model,
            latency_ms=(time.time() - t0) * 1000,
            error=str(e),
        )


# ═══════════════════════════════════════════════════════════════════════════════
# Baseline 3: PIE Temporal
# ═══════════════════════════════════════════════════════════════════════════════


def pie_temporal(
    item: dict[str, Any],
    world_model: WorldModel | None = None,
    llm: LLMClient | None = None,
    model: str = "gpt-4o",
    extraction_model: str = "gpt-4o-mini",
    top_k_entities: int = 10,
) -> BaselineResult:
    """PIE's temporal knowledge graph approach."""
    llm = llm or LLMClient()
    t0 = time.time()

    try:
        conversations = item.get("conversations", [])
        personas = item.get("personas", [])
        item_type = item.get("item_type", "response_generation")
        query = item.get("question", "")

        if world_model is None:
            world_model = _build_world_model(conversations, personas, llm, extraction_model)

        # Retrieve relevant entities
        if item_type == "response_generation" and conversations:
            # Use recent context as query
            recent = conversations[-1] if conversations else None
            if recent and recent.turns:
                query = " ".join(t.text for t in recent.turns[-3:])

        retrieved = _retrieve_entities(query, world_model, llm, top_k_entities)

        # Compile context
        context = _compile_context(retrieved, world_model, personas)

        # Generate answer
        if item_type == "response_generation":
            answer = _generate_response(context, personas, llm, model)
        else:
            answer = _answer_qa(context, query, personas, llm, model)

        return BaselineResult(
            item_id=item["item_id"],
            item_type=item_type,
            question=query,
            gold_answer=item.get("expected_response", item.get("answer", "")),
            hypothesis=answer,
            baseline_name="pie_temporal",
            model=model,
            latency_ms=(time.time() - t0) * 1000,
            context_chars=len(context),
            retrieval_count=len(retrieved),
        )

    except Exception as e:
        logger.exception(f"PIE temporal failed")
        return BaselineResult(
            item_id=item["item_id"],
            item_type=item.get("item_type", "unknown"),
            question=item.get("question", ""),
            gold_answer=item.get("expected_response", item.get("answer", "")),
            hypothesis=f"Error: {e}",
            baseline_name="pie_temporal",
            model=model,
            latency_ms=(time.time() - t0) * 1000,
            error=str(e),
        )


def _build_world_model(
    conversations: list[Conversation],
    personas: list[str],
    llm: LLMClient,
    extraction_model: str,
) -> WorldModel:
    """Build world model from conversations and personas."""
    import json

    wm = WorldModel()

    # Add persona facts as entities
    for i, persona in enumerate(personas):
        wm.create_entity(
            name=f"persona_fact_{i}",
            type=EntityType.BELIEF,
            state={"description": persona},
            source_conversation_id="persona",
            timestamp=0,
        )

    # Extract from conversations
    extraction_prompt = """\
Extract key facts from this conversation:
- Preferences, opinions, beliefs
- Mentioned people, places, activities
- Important events or plans

Output JSON: {"entities": [{"name": "...", "type": "concept|person|belief", "state": {"description": "..."}}]}"""

    for convo in conversations:
        text = "\n".join(f"{t.role}: {t.text}" for t in convo.turns)
        try:
            result = llm.chat(
                messages=[
                    {"role": "system", "content": extraction_prompt},
                    {"role": "user", "content": text[:4000]},
                ],
                model=extraction_model,
                json_mode=True,
            )
            raw = result["content"]
            if isinstance(raw, str):
                raw = json.loads(raw)

            for entity_data in raw.get("entities", []):
                name = entity_data.get("name", "")
                if not name:
                    continue

                etype_str = entity_data.get("type", "concept").lower()
                try:
                    etype = EntityType(etype_str)
                except ValueError:
                    etype = EntityType.CONCEPT

                state = entity_data.get("state", {})
                if isinstance(state, str):
                    state = {"description": state}

                existing = wm.find_by_name(name)
                if existing:
                    wm.update_entity_state(
                        entity_id=existing.id,
                        new_state=state,
                        source_conversation_id=convo.id,
                        timestamp=convo.created_at,
                    )
                else:
                    wm.create_entity(
                        name=name,
                        type=etype,
                        state=state,
                        source_conversation_id=convo.id,
                        timestamp=convo.created_at,
                    )

        except Exception as e:
            logger.warning(f"Extraction failed: {e}")
            continue

    return wm


def _retrieve_entities(
    query: str,
    world_model: WorldModel,
    llm: LLMClient,
    top_k: int = 10,
) -> list[tuple[str, Any, float]]:
    """Retrieve relevant entities."""
    if not world_model.entities:
        return []

    query_emb = llm.embed_single(query)

    # Compute missing embeddings
    for eid, entity in world_model.entities.items():
        if entity.embedding is None:
            state = entity.current_state
            desc = state.get("description", str(state)[:200]) if isinstance(state, dict) else str(state)[:200]
            text = f"{entity.name}: {desc}"
            try:
                entity.embedding = llm.embed_single(text)
            except:
                continue

    # Score
    scored = []
    for eid, entity in world_model.entities.items():
        if entity.embedding:
            sim = cosine_similarity(query_emb, entity.embedding)
            scored.append((eid, entity, sim))

    scored.sort(key=lambda x: x[2], reverse=True)
    return scored[:top_k]


def _compile_context(
    retrieved: list[tuple[str, Any, float]],
    world_model: WorldModel,
    personas: list[str],
) -> str:
    """Compile context from retrieved entities."""
    parts = []

    # Add personas
    if personas:
        parts.append("Speaker facts:")
        for p in personas:
            parts.append(f"  - {p}")
        parts.append("")

    # Add retrieved entities
    if retrieved:
        parts.append("Relevant context:")
        for eid, entity, score in retrieved:
            state = entity.current_state
            desc = state.get("description", "") if isinstance(state, dict) else str(state)
            if desc:
                parts.append(f"  - {entity.name}: {desc}")

    return "\n".join(parts)


BASELINES = {
    "full_context": full_context,
    "naive_rag": naive_rag,
    "pie_temporal": pie_temporal,
}
