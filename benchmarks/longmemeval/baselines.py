"""
LongMemEval Baselines — comparison approaches for the benchmark.

Three baselines:
  1. full_context  — stuff all sessions into context, ask LLM directly (~60% baseline)
  2. naive_rag     — embed turns, retrieve top-k by similarity, answer
  3. pie_temporal  — PIE's approach: ingest → build world model → compile temporal
                     context → answer

Each baseline takes a question item and returns an answer string.
"""

from __future__ import annotations
import datetime
import logging
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, TYPE_CHECKING

if TYPE_CHECKING:
    from benchmarks.common.cache import CachedWorldModel

from pie.core.llm import LLMClient
from pie.core.world_model import WorldModel, cosine_similarity
from pie.core.models import (
    Conversation, Turn, EntityType, TransitionType,
)

from .adapter import (
    haystack_to_conversations,
    format_haystack_as_text,
    format_date_for_context,
    parse_longmemeval_date,
    parse_question_date,
)

logger = logging.getLogger("pie.bench.longmemeval")


# ── Shared Types ──────────────────────────────────────────────────────────────


@dataclass
class BaselineResult:
    """Result from running a baseline on one question."""
    question_id: str
    question_type: str
    question: str
    gold_answer: str
    hypothesis: str            # the baseline's generated answer
    baseline_name: str
    model: str
    latency_ms: float = 0.0
    context_chars: int = 0
    retrieval_count: int = 0   # number of items retrieved (for RAG/PIE)
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


# ── QA Prompt (shared across baselines) ──────────────────────────────────────

ANSWER_SYSTEM_PROMPT = """\
You are a helpful assistant answering questions about a user's past conversations.
You will be given context from the user's chat history, and a question.
Answer the question based ONLY on the provided context.
Be concise and specific. If the context doesn't contain the answer, say "I don't know."
Do NOT make up information."""

ANSWER_USER_TEMPLATE = """\
Context from the user's chat history:

{context}

---

Question (asked on {question_date}): {question}

Answer concisely:"""


def _ask_llm(
    context: str,
    question: str,
    question_date: str,
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
                question_date=format_date_for_context(question_date),
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
    Full-context baseline: stuff ALL haystack sessions into the prompt.
    
    This is the simplest approach and the ~60% baseline from the paper.
    Limited by context window — may need truncation for large haystacks.
    """
    llm = llm or LLMClient()
    t0 = time.time()

    try:
        context = format_haystack_as_text(
            item["haystack_sessions"],
            item["haystack_dates"],
            max_chars=max_context_chars,
        )

        answer = _ask_llm(
            context=context,
            question=item["question"],
            question_date=item["question_date"],
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
    chunk_by: str = "turn",  # "turn" or "session"
) -> BaselineResult:
    """
    Naive RAG baseline: embed chunks, retrieve top-k by cosine similarity.
    
    Two chunking strategies:
      - "turn": each user/assistant turn is a chunk (finer-grained)
      - "session": each session is a chunk (preserves conversation flow)
    """
    llm = llm or LLMClient()
    t0 = time.time()

    try:
        # Step 1: Build chunks with metadata
        chunks = _build_rag_chunks(
            item["haystack_sessions"],
            item["haystack_dates"],
            chunk_by=chunk_by,
        )

        if not chunks:
            return BaselineResult(
                question_id=item["question_id"],
                question_type=item["question_type"],
                question=item["question"],
                gold_answer=item["answer"],
                hypothesis="No chunks to search.",
                baseline_name=f"naive_rag_{chunk_by}",
                model=model,
                latency_ms=(time.time() - t0) * 1000,
            )

        # Step 2: Embed question and all chunks
        query_emb = llm.embed_single(item["question"], model=embed_model)

        # Batch embed all chunks (may need sub-batching for large haystacks)
        chunk_texts = [c["text"] for c in chunks]
        chunk_embeddings = _batch_embed(chunk_texts, llm, embed_model)

        # Step 3: Rank by cosine similarity
        scored = []
        for chunk, emb in zip(chunks, chunk_embeddings):
            sim = cosine_similarity(query_emb, emb)
            scored.append((chunk, sim))

        scored.sort(key=lambda x: x[1], reverse=True)
        top_chunks = scored[:top_k]

        # Step 4: Compile context from top chunks (preserve chronological order)
        top_chunks_sorted = sorted(top_chunks, key=lambda x: x[0]["timestamp"])
        context_parts = []
        for chunk, score in top_chunks_sorted:
            context_parts.append(
                f"[{chunk['date']} | relevance: {score:.2f}]\n{chunk['text']}"
            )
        context = "\n\n".join(context_parts)

        # Step 5: Ask LLM
        answer = _ask_llm(
            context=context,
            question=item["question"],
            question_date=item["question_date"],
            llm=llm,
            model=model,
        )

        return BaselineResult(
            question_id=item["question_id"],
            question_type=item["question_type"],
            question=item["question"],
            gold_answer=item["answer"],
            hypothesis=answer,
            baseline_name=f"naive_rag_{chunk_by}",
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
            baseline_name=f"naive_rag_{chunk_by}",
            model=model,
            latency_ms=(time.time() - t0) * 1000,
            error=str(e),
        )


def _build_rag_chunks(
    haystack_sessions: list[list[dict]],
    haystack_dates: list[str],
    chunk_by: str = "turn",
) -> list[dict]:
    """Build text chunks for RAG retrieval."""
    chunks = []

    for i, (session, date_str) in enumerate(zip(haystack_sessions, haystack_dates)):
        timestamp = parse_longmemeval_date(date_str)
        human_date = format_date_for_context(date_str)

        if chunk_by == "session":
            # Entire session as one chunk
            lines = []
            for turn in session:
                role = turn.get("role", "user").capitalize()
                content = turn.get("content", "").strip()
                if content:
                    lines.append(f"{role}: {content}")
            if lines:
                chunks.append({
                    "text": "\n".join(lines),
                    "date": human_date,
                    "timestamp": timestamp,
                    "session_index": i,
                })

        elif chunk_by == "turn":
            # Each turn as a separate chunk (with session context)
            for j, turn in enumerate(session):
                content = turn.get("content", "").strip()
                if not content:
                    continue
                role = turn.get("role", "user").capitalize()
                # Include preceding turn for context if it's a response
                context_prefix = ""
                if j > 0 and role == "Assistant":
                    prev = session[j - 1]
                    prev_content = prev.get("content", "").strip()
                    if prev_content:
                        context_prefix = f"User: {prev_content[:200]}\n"
                chunks.append({
                    "text": f"{context_prefix}{role}: {content}",
                    "date": human_date,
                    "timestamp": timestamp + (j * 5),
                    "session_index": i,
                    "turn_index": j,
                })

    return chunks


def _batch_embed(
    texts: list[str],
    llm: LLMClient,
    model: str = "text-embedding-3-large",
    batch_size: int = 512,
) -> list[list[float]]:
    """Embed texts in batches to avoid API limits."""
    all_embeddings = []
    for i in range(0, len(texts), batch_size):
        batch = texts[i : i + batch_size]
        # Truncate very long texts for embedding
        batch = [t[:8000] for t in batch]
        embeddings = llm.embed(batch, model=model)
        all_embeddings.extend(embeddings)
    return all_embeddings


# ═══════════════════════════════════════════════════════════════════════════════
# Baseline 3: PIE Temporal (the approach we're testing)
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
      1. Ingest haystack sessions into a fresh world model
      2. Build temporal knowledge graph (entities, transitions, relationships)
      3. Retrieve relevant entities for the question
      4. Compile semantic temporal context
      5. Ask LLM to answer using compiled context
    
    If a pre-built world_model is provided, skip ingestion (for cached runs).
    """
    llm = llm or LLMClient()
    t0 = time.time()

    try:
        # Step 1: Build or reuse world model
        if world_model is None:
            world_model = _build_world_model_for_question(
                item, llm, extraction_model
            )

        # Step 2: Retrieve relevant entities
        question = item["question"]
        question_ts = parse_question_date(item["question_date"])

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
                hypothesis="I don't have enough information to answer this question.",
                baseline_name="pie_temporal",
                model=model,
                latency_ms=(time.time() - t0) * 1000,
            )

        # Step 3: Compile semantic temporal context
        context = _compile_temporal_context(
            retrieved=retrieved,
            world_model=world_model,
            question_ts=question_ts,
            max_chars=max_context_chars,
        )

        # Step 4: Ask LLM with temporal context
        answer = _ask_llm_temporal(
            context=context,
            question=question,
            question_date=item["question_date"],
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


# ── PIE Temporal Helpers ─────────────────────────────────────────────────────


PIE_EXTRACTION_PROMPT = """\
You are extracting structured knowledge from a user's chat history.

Each session header shows the EXACT DATE in format: [Session — Month DD, YYYY at HH:MM]
Use this to compute exact dates from relative references!

For each session below, extract:
1. **Entities**: People, places, projects, preferences, AND USER ACTIVITIES/EVENTS.
   - name: canonical name (for events, use descriptive name like "MoMA visit" or "friend's wedding")
   - type: person|project|tool|organization|belief|concept|preference|event
   - state: dict of key-value attributes
   - For EVENTS, ALWAYS include: {date: "YYYY-MM-DD", description: "...", location: "..." (if known)}

2. **State changes**: If an entity's state changed from a previous value.
   - entity_name, what_changed, old_state (if known), new_state, is_contradiction

3. **Relationships**: How entities relate to each other.
   - source, target, type (related_to|uses|works_on|has|prefers|located_in|attended|participated_in), description

CRITICAL FOR TEMPORAL QUESTIONS — EXTRACT EVENTS WITH EXACT DATES:

**PRIORITY 1: USER'S COMPLETED ACTIVITIES (MOST IMPORTANT!)**
Look for phrases where the USER says they DID something:
- "I just [verb]" → event happened TODAY (session date)
- "I [verb] today" → event happened TODAY (session date)
- "I [verb] yesterday" → event happened YESTERDAY (session date - 1 day)
- "I [verb] last week" → event happened ~7 days before session
- "I recently [verb]" → event happened around session date

Examples of USER ACTIVITIES to extract as events:
- "I just got back from a tour at MoMA" → MoMA visit event, date = session date
- "I just helped my friend prepare a nursery today" → nursery preparation event, date = session date
- "I just ordered a customized phone case" → phone case order event, date = session date
- "I attended the exhibit yesterday" → exhibit visit event, date = session date - 1 day
- "I helped my cousin pick out stuff for her baby shower" → baby shower shopping event

**PRIORITY 2: ALL OTHER USER ACTIVITIES**
- Extract ALL user activities: visits, meetings, purchases, trips, appointments, dinners, helping friends, etc.
- ALWAYS compute the EXACT DATE in YYYY-MM-DD format
- For relative references, compute from the session date:
  * "today" → session date
  * "yesterday" → subtract 1 day from session date
  * "last Tuesday" → find the most recent Tuesday before session date
  * "last week" → subtract 7 days
  * "two weeks ago" → subtract 14 days
  * "last month" → same day, previous month

**PRIORITY 3: Other facts about the user**
- Focus on FACTS ABOUT THE USER — their life, preferences, relationships.
- Skip generic conversational content and assistant recommendations.

Output JSON:
{
  "entities": [...],
  "state_changes": [...],
  "relationships": [...],
  "summary": "one-line summary of key facts"
}"""


def _build_world_model_for_question(
    item: dict[str, Any],
    llm: LLMClient,
    extraction_model: str = "gpt-4o-mini",
) -> WorldModel:
    """
    Build a fresh world model from a question's haystack sessions.
    
    This is the expensive part — each question requires processing ~53 sessions
    through the extraction LLM to build the knowledge graph.
    """
    from pie.core.models import ExtractedEntity, ExtractedStateChange, ExtractedRelationship
    from pie.core.llm import parse_extraction_result

    wm = WorldModel()
    conversations = haystack_to_conversations(
        item["haystack_sessions"],
        item["haystack_dates"],
        item["question_id"],
    )

    # Process sessions in chronological batches (group ~5 sessions per LLM call
    # for efficiency while maintaining chronological context)
    batch_size = 5
    for batch_start in range(0, len(conversations), batch_size):
        batch = conversations[batch_start : batch_start + batch_size]

        # Format batch for extraction
        batch_text = _format_conversations_for_extraction(batch)

        # Build context preamble from current world model state
        context = ""
        if wm.entities:
            context = wm.build_context_preamble(batch[0].created_at)

        user_msg = ""
        if context:
            user_msg += f"=== CURRENT KNOWN FACTS ===\n{context}\n\n"
        user_msg += f"=== CONVERSATIONS TO PROCESS ===\n{batch_text}"

        try:
            result = llm.chat(
                messages=[
                    {"role": "system", "content": PIE_EXTRACTION_PROMPT},
                    {"role": "user", "content": user_msg},
                ],
                model=extraction_model,
                json_mode=True,
            )

            raw = result["content"]
            if isinstance(raw, str):
                import json
                raw = json.loads(raw)

            # Apply extracted entities to world model
            _apply_extraction_to_world_model(
                raw, wm, batch[0].created_at, batch[0].id
            )

        except Exception as e:
            logger.warning(
                f"Extraction failed for batch starting at session "
                f"{batch_start}: {e}"
            )
            continue

    return wm


def _format_conversations_for_extraction(conversations: list[Conversation]) -> str:
    """Format conversations for the extraction prompt."""
    import datetime

    parts = []
    for convo in conversations:
        dt = datetime.datetime.fromtimestamp(
            convo.created_at, tz=datetime.timezone.utc
        )
        date_str = dt.strftime("%B %d, %Y at %H:%M")
        parts.append(f"[Session — {date_str}]")
        for turn in convo.turns:
            role = turn.role.capitalize()
            text = turn.text[:2000]  # cap per-turn length
            parts.append(f"{role}: {text}")
        parts.append("")
    return "\n".join(parts)


def _apply_extraction_to_world_model(
    raw: dict,
    wm: WorldModel,
    timestamp: float,
    convo_id: str,
) -> None:
    """Apply raw extraction output to the world model."""
    for entity_data in raw.get("entities", []):
        name = entity_data.get("name", "")
        if not name:
            continue

        etype_str = entity_data.get("type", "concept").lower()
        is_event = etype_str == "event"
        try:
            etype = EntityType(etype_str)
        except ValueError:
            # Map non-standard types
            type_map = {
                "preference": EntityType.BELIEF,
                "place": EntityType.CONCEPT,
                "location": EntityType.CONCEPT,
                "event": EntityType.CONCEPT,  # Will be marked via state
                "skill": EntityType.CONCEPT,
                "hobby": EntityType.CONCEPT,
                "food": EntityType.CONCEPT,
                "pet": EntityType.CONCEPT,
                "vehicle": EntityType.CONCEPT,
                "item": EntityType.CONCEPT,
            }
            etype = type_map.get(etype_str, EntityType.CONCEPT)

        state = entity_data.get("state", {})
        if isinstance(state, str):
            state = {"description": state}

        # Mark events and preserve their date
        if is_event:
            state["_is_event"] = True
            # If date wasn't in state, try to extract it from entity_data
            if "date" not in state and entity_data.get("date"):
                state["date"] = entity_data.get("date")

        # Check if entity already exists
        existing = wm.find_by_name(name)
        if existing:
            wm.update_entity_state(
                entity_id=existing.id,
                new_state=state,
                source_conversation_id=convo_id,
                timestamp=timestamp,
                trigger_summary=f"Updated from session",
            )
        else:
            wm.create_entity(
                name=name,
                type=etype,
                state=state,
                source_conversation_id=convo_id,
                timestamp=timestamp,
            )

    # Apply state changes
    for sc in raw.get("state_changes", []):
        entity_name = sc.get("entity_name", "")
        entity = wm.find_by_name(entity_name)
        if entity:
            new_state_val = sc.get("new_state") or sc.get("to_state", "")
            wm.update_entity_state(
                entity_id=entity.id,
                new_state={"description": new_state_val} if isinstance(new_state_val, str) else new_state_val or {},
                source_conversation_id=convo_id,
                timestamp=timestamp,
                trigger_summary=sc.get("what_changed", "state changed"),
                is_contradiction=sc.get("is_contradiction", False),
            )

    # Apply relationships
    for rel in raw.get("relationships", []):
        source_name = rel.get("source", "")
        target_name = rel.get("target", "")
        source_entity = wm.find_by_name(source_name)
        target_entity = wm.find_by_name(target_name)
        if source_entity and target_entity:
            from pie.core.models import RelationshipType
            try:
                rtype = RelationshipType(rel.get("type", "related_to").lower())
            except ValueError:
                rtype = RelationshipType.RELATED_TO
            wm.add_relationship(
                source_id=source_entity.id,
                target_id=target_entity.id,
                rel_type=rtype,
                description=rel.get("description", ""),
                source_conversation_id=convo_id,
                timestamp=timestamp,
            )


def _retrieve_entities_for_question(
    question: str,
    world_model: WorldModel,
    llm: LLMClient,
    top_k: int = 15,
) -> list[tuple[str, dict, float]]:
    """
    Retrieve relevant entities for a question using embedding similarity.
    
    Returns list of (entity_id, entity_dict, similarity) sorted by relevance.
    """
    if not world_model.entities:
        return []

    # Embed the question
    query_emb = llm.embed_single(question)

    # Compute embeddings for entities that lack them (batch)
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

    # Score entities by embedding similarity
    scored = []
    for eid, entity in world_model.entities.items():
        if entity.embedding:
            sim = cosine_similarity(query_emb, entity.embedding)
            scored.append((eid, entity, sim))

    scored.sort(key=lambda x: x[2], reverse=True)
    return [(eid, e, s) for eid, e, s in scored[:top_k]]


def _humanize_delta(seconds: float) -> str:
    """Convert seconds to human-readable duration."""
    if seconds < 0:
        return "in the future"
    days = seconds / 86400
    if days < 1:
        return "today"
    elif days < 2:
        return "yesterday"
    elif days < 7:
        return f"{int(days)} days ago"
    elif days < 30:
        weeks = int(days / 7)
        return f"~{weeks} week{'s' if weeks != 1 else ''} ago"
    elif days < 365:
        months = int(days / 30)
        return f"~{months} month{'s' if months != 1 else ''} ago"
    else:
        years = days / 365
        return f"~{years:.1f} years ago"


def _guess_period(timestamp: float) -> str:
    """Guess a human-readable period from a timestamp."""
    import datetime
    dt = datetime.datetime.fromtimestamp(timestamp, tz=datetime.timezone.utc)
    return dt.strftime("%B %Y")


def _compile_temporal_context(
    retrieved: list[tuple[str, Any, float]],
    world_model: WorldModel,
    question_ts: float,
    max_chars: int = 12_000,
) -> str:
    """
    Compile PIE's semantic temporal context from retrieved entities.
    
    This is the core of PIE's approach — converting graph data into
    LLM-readable temporal narratives. The LLM NEVER sees raw timestamps.
    """
    parts = []
    total_chars = 0

    for eid, entity, relevance in retrieved:
        transitions = world_model.get_transitions(eid, ordered=True)
        relationships = world_model.get_relationships(eid)

        lines = []
        name = entity.name
        etype = entity.type.value
        state = entity.current_state or {}

        # Header with temporal metadata - INCLUDE EXACT DATES for arithmetic
        import datetime
        first_seen = entity.first_seen
        last_seen = entity.last_seen
        first_ago = _humanize_delta(question_ts - first_seen)
        last_ago = _humanize_delta(question_ts - last_seen)
        first_period = _guess_period(first_seen)
        
        # Add exact dates for temporal arithmetic
        first_date = datetime.datetime.fromtimestamp(first_seen, tz=datetime.timezone.utc).strftime("%Y-%m-%d")
        last_date = datetime.datetime.fromtimestamp(last_seen, tz=datetime.timezone.utc).strftime("%Y-%m-%d")

        change_count = len(transitions)
        months_span = max((last_seen - first_seen) / (30 * 86400), 1)
        velocity = change_count / months_span

        # Check if this is an event with an explicit date
        is_event = state.get("_is_event", False) or state.get("date")
        event_date = state.get("date", "")
        
        # FALLBACK: If this looks like an event but has no date, use first_seen timestamp
        # This handles cases where extraction identified an event but didn't compute the date
        if is_event and not event_date:
            event_date = first_date  # first_date is already computed from first_seen
        
        if is_event and event_date:
            # For events, prominently show the EVENT DATE
            lines.append(f"## {name} (EVENT)")
            # Compute relative time from event date
            try:
                event_dt = datetime.datetime.strptime(event_date, "%Y-%m-%d").replace(tzinfo=datetime.timezone.utc)
                event_ts = event_dt.timestamp()
                event_ago = _humanize_delta(question_ts - event_ts)
                lines.append(f"**Event date: {event_date} ({event_ago})**")
            except:
                lines.append(f"**Event date: {event_date}**")
            
            # Add description and location if available
            desc = state.get("description", "")
            location = state.get("location", "")
            if desc:
                lines.append(f"Description: {desc}")
            if location:
                lines.append(f"Location: {location}")
        else:
            # Regular entity handling
            lines.append(f"## {name} ({etype})")
            lines.append(
                f"First mentioned: {first_date} ({first_ago}), "
                f"last mentioned: {last_date} ({last_ago})."
            )
            if change_count > 1:
                lines.append(
                    f"State changed {change_count} times "
                    f"(~{velocity:.1f}x/month)."
                )

            # Current state
            if state:
                desc = state.get("description", "")
                if not desc and isinstance(state, dict):
                    # Build description from state dict (exclude internal fields)
                    desc = "; ".join(
                        f"{k}: {v}" for k, v in state.items()
                        if k not in ("description", "_is_event") and v
                    )
                if desc:
                    lines.append(f"Current state: {desc}")

        # Timeline of changes (most relevant for temporal/knowledge-update)
        if transitions and len(transitions) > 1:
            lines.append("")
            lines.append("Timeline:")
            for t in transitions:
                t_ago = _humanize_delta(question_ts - t.timestamp)
                t_date = datetime.datetime.fromtimestamp(t.timestamp, tz=datetime.timezone.utc).strftime("%Y-%m-%d")
                ttype = t.transition_type

                prefix = "  •"
                if ttype == TransitionType.CONTRADICTION:
                    prefix = "  ⚠ [CHANGED]"
                elif ttype == TransitionType.CREATION:
                    prefix = "  ★"

                summary = t.trigger_summary
                if summary:
                    lines.append(f"{prefix} {t_date} ({t_ago}): {summary}")

        # Relationships
        if relationships:
            rel_strs = []
            for r in relationships[:5]:
                other_id = (
                    r.target_id if r.source_id == eid else r.source_id
                )
                other = world_model.get_entity(other_id)
                if other:
                    rel_strs.append(
                        f"{r.type.value}: {other.name}"
                        + (f" ({r.description})" if r.description else "")
                    )
            if rel_strs:
                lines.append(f"\nRelated: {'; '.join(rel_strs)}")

        part = "\n".join(lines)

        if total_chars + len(part) > max_chars:
            break

        parts.append(part)
        total_chars += len(part)

    return "\n\n".join(parts)


PIE_ANSWER_SYSTEM_PROMPT = """\
You are a personal knowledge assistant answering questions about a user's life
and conversations. You are given a structured knowledge base compiled from
their chat history, organized by entity with temporal information.

Key features of the context:
- Each entity shows WHEN it first appeared and was last mentioned
- State changes are tracked chronologically with timestamps
- Contradictions (where info changed) are marked with ⚠ [CHANGED]
- Relationships between entities are shown

Use this temporal information to answer time-sensitive questions accurately.
Pay special attention to:
- The MOST RECENT state when asking about "current" info
- State changes marked as contradictions — the NEWER value is correct
- Chronological ordering for "first", "last", "before", "after" questions

Answer based ONLY on the provided context. Be concise and specific.
If the answer isn't in the context, say "I don't know."
Do NOT make up information."""


def _ask_llm_temporal(
    context: str,
    question: str,
    question_date: str,
    llm: LLMClient,
    model: str = "gpt-4o",
) -> str:
    """Ask LLM with PIE's temporal context format."""
    messages = [
        {"role": "system", "content": PIE_ANSWER_SYSTEM_PROMPT},
        {
            "role": "user",
            "content": (
                f"Knowledge base (as of {format_date_for_context(question_date)}):\n\n"
                f"{context}\n\n---\n\n"
                f"Question: {question}\n\nAnswer concisely:"
            ),
        },
    ]
    result = llm.chat(messages=messages, model=model, max_tokens=300)
    return result["content"].strip()


# ═══════════════════════════════════════════════════════════════════════════════
# Baseline 4: PIE Temporal Cached (optimized for benchmark runs)
# ═══════════════════════════════════════════════════════════════════════════════


class PIETemporalCachedBaseline:
    """
    PIE Temporal baseline with caching for efficient benchmark runs.
    
    Key optimizations:
    - World model built once and cached to disk
    - Entity embeddings computed once and cached
    - Each question only does: embed query → retrieve → compile context → answer
    
    Usage:
        baseline = PIETemporalCachedBaseline(
            cache_dir=Path("cache/"),
            llm=llm,
            model="gpt-4o",
        )
        
        # For each question in benchmark
        result = baseline.run(item)
    """
    
    def __init__(
        self,
        cache_dir: Path | str,
        llm: LLMClient | None = None,
        model: str = "gpt-4o",
        extraction_model: str = "gpt-4o-mini",
        embed_model: str = "text-embedding-3-large",
        top_k_entities: int = 15,
        max_context_chars: int = 12_000,
    ):
        from pathlib import Path
        from benchmarks.common.cache import CachedWorldModel
        
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        
        self.llm = llm or LLMClient()
        self.model = model
        self.extraction_model = extraction_model
        self.embed_model = embed_model
        self.top_k_entities = top_k_entities
        self.max_context_chars = max_context_chars
        
        # Cache of loaded CachedWorldModels by question_id
        self._cached_models: dict[str, CachedWorldModel] = {}
    
    def _get_cached_world_model(self, item: dict[str, Any]) -> "CachedWorldModel":
        """Get or build cached world model for a question."""
        from benchmarks.common.cache import CachedWorldModel
        
        qid = item["question_id"]
        
        if qid in self._cached_models:
            return self._cached_models[qid]
        
        cache_path = self.cache_dir / f"{qid}_world_model.json"
        
        def build_fn():
            return _build_world_model_for_question(
                item, self.llm, self.extraction_model
            )
        
        cached_wm = CachedWorldModel.load_or_build(
            cache_path=cache_path,
            build_fn=build_fn,
            llm=self.llm,
            embed_model=self.embed_model,
        )
        
        self._cached_models[qid] = cached_wm
        return cached_wm
    
    def run(self, item: dict[str, Any]) -> BaselineResult:
        """
        Run PIE temporal baseline on a single question using cached world model.
        """
        t0 = time.time()
        
        try:
            # Get cached world model (builds if needed)
            cached_wm = self._get_cached_world_model(item)
            
            question = item["question"]
            question_ts = parse_question_date(item["question_date"])
            
            # Retrieve relevant entities (uses cached embeddings)
            retrieved = cached_wm.retrieve(question, top_k=self.top_k_entities)
            
            if not retrieved:
                return BaselineResult(
                    question_id=item["question_id"],
                    question_type=item["question_type"],
                    question=question,
                    gold_answer=item["answer"],
                    hypothesis="I don't have enough information to answer this question.",
                    baseline_name="pie_temporal_cached",
                    model=self.model,
                    latency_ms=(time.time() - t0) * 1000,
                )
            
            # Compile temporal context
            context = _compile_temporal_context_cached(
                retrieved=retrieved,
                cached_wm=cached_wm,
                question_ts=question_ts,
                max_chars=self.max_context_chars,
            )
            
            # Ask LLM
            answer = _ask_llm_temporal(
                context=context,
                question=question,
                question_date=item["question_date"],
                llm=self.llm,
                model=self.model,
            )
            
            return BaselineResult(
                question_id=item["question_id"],
                question_type=item["question_type"],
                question=question,
                gold_answer=item["answer"],
                hypothesis=answer,
                baseline_name="pie_temporal_cached",
                model=self.model,
                latency_ms=(time.time() - t0) * 1000,
                context_chars=len(context),
                retrieval_count=len(retrieved),
            )
        
        except Exception as e:
            logger.exception(f"PIE temporal cached failed for {item['question_id']}")
            return BaselineResult(
                question_id=item["question_id"],
                question_type=item["question_type"],
                question=item["question"],
                gold_answer=item["answer"],
                hypothesis=f"Error: {e}",
                baseline_name="pie_temporal_cached",
                model=self.model,
                latency_ms=(time.time() - t0) * 1000,
                error=str(e),
            )
    
    def print_stats(self):
        """Print caching statistics."""
        print(f"\nPIETemporalCachedBaseline Stats:")
        print(f"  Cached models loaded: {len(self._cached_models)}")
        for qid, cached_wm in self._cached_models.items():
            print(f"  [{qid}]")
            cached_wm.print_stats()


def _compile_temporal_context_cached(
    retrieved: list[tuple[str, Any, float]],
    cached_wm: "CachedWorldModel",
    question_ts: float,
    max_chars: int = 12_000,
) -> str:
    """
    Compile temporal context using CachedWorldModel.
    
    Same as _compile_temporal_context but uses cached_wm interface.
    """
    import datetime
    
    parts = []
    total_chars = 0
    
    for eid, entity, relevance in retrieved:
        transitions = cached_wm.get_transitions(eid, ordered=True)
        relationships = cached_wm.get_relationships(eid)
        
        lines = []
        name = entity.name
        etype = entity.type.value
        state = entity.current_state or {}
        
        # Header with temporal metadata
        first_seen = entity.first_seen
        last_seen = entity.last_seen
        first_ago = _humanize_delta(question_ts - first_seen)
        last_ago = _humanize_delta(question_ts - last_seen)
        
        first_date = datetime.datetime.fromtimestamp(
            first_seen, tz=datetime.timezone.utc
        ).strftime("%Y-%m-%d")
        last_date = datetime.datetime.fromtimestamp(
            last_seen, tz=datetime.timezone.utc
        ).strftime("%Y-%m-%d")
        
        change_count = len(transitions)
        months_span = max((last_seen - first_seen) / (30 * 86400), 1)
        velocity = change_count / months_span
        
        # Check if this is an event
        is_event = state.get("_is_event", False) or state.get("date")
        event_date = state.get("date", "")
        
        if is_event and not event_date:
            event_date = first_date
        
        if is_event and event_date:
            lines.append(f"## {name} (EVENT)")
            try:
                event_dt = datetime.datetime.strptime(
                    event_date, "%Y-%m-%d"
                ).replace(tzinfo=datetime.timezone.utc)
                event_ts = event_dt.timestamp()
                event_ago = _humanize_delta(question_ts - event_ts)
                lines.append(f"**Event date: {event_date} ({event_ago})**")
            except:
                lines.append(f"**Event date: {event_date}**")
            
            desc = state.get("description", "")
            location = state.get("location", "")
            if desc:
                lines.append(f"Description: {desc}")
            if location:
                lines.append(f"Location: {location}")
        else:
            lines.append(f"## {name} ({etype})")
            lines.append(
                f"First mentioned: {first_date} ({first_ago}), "
                f"last mentioned: {last_date} ({last_ago})."
            )
            if change_count > 1:
                lines.append(
                    f"State changed {change_count} times "
                    f"(~{velocity:.1f}x/month)."
                )
            
            if state:
                desc = state.get("description", "")
                if not desc and isinstance(state, dict):
                    desc = "; ".join(
                        f"{k}: {v}" for k, v in state.items()
                        if k not in ("description", "_is_event") and v
                    )
                if desc:
                    lines.append(f"Current state: {desc}")
        
        # Timeline
        if transitions and len(transitions) > 1:
            lines.append("")
            lines.append("Timeline:")
            for t in transitions:
                t_ago = _humanize_delta(question_ts - t.timestamp)
                t_date = datetime.datetime.fromtimestamp(
                    t.timestamp, tz=datetime.timezone.utc
                ).strftime("%Y-%m-%d")
                ttype = t.transition_type
                
                prefix = "  •"
                if ttype == TransitionType.CONTRADICTION:
                    prefix = "  ⚠ [CHANGED]"
                elif ttype == TransitionType.CREATION:
                    prefix = "  ★"
                
                summary = t.trigger_summary
                if summary:
                    lines.append(f"{prefix} {t_date} ({t_ago}): {summary}")
        
        # Relationships
        if relationships:
            rel_strs = []
            for r in relationships[:5]:
                other_id = r.target_id if r.source_id == eid else r.source_id
                other = cached_wm.get_entity(other_id)
                if other:
                    rel_strs.append(
                        f"{r.type.value}: {other.name}"
                        + (f" ({r.description})" if r.description else "")
                    )
            if rel_strs:
                lines.append(f"\nRelated: {'; '.join(rel_strs)}")
        
        part = "\n".join(lines)
        
        if total_chars + len(part) > max_chars:
            break
        
        parts.append(part)
        total_chars += len(part)
    
    return "\n\n".join(parts)


def pie_temporal_cached(
    item: dict[str, Any],
    baseline: PIETemporalCachedBaseline | None = None,
    cache_dir: str | Path | None = None,
    llm: LLMClient | None = None,
    model: str = "gpt-4o",
    extraction_model: str = "gpt-4o-mini",
    top_k_entities: int = 15,
    max_context_chars: int = 12_000,
) -> BaselineResult:
    """
    Function wrapper for PIETemporalCachedBaseline.
    
    If baseline is provided, uses it directly (for batch runs).
    Otherwise creates a new baseline instance (for single question).
    """
    from pathlib import Path
    
    if baseline is not None:
        return baseline.run(item)
    
    # Create baseline for single-question use
    if cache_dir is None:
        cache_dir = Path(__file__).parent / "cache"
    
    baseline = PIETemporalCachedBaseline(
        cache_dir=cache_dir,
        llm=llm,
        model=model,
        extraction_model=extraction_model,
        top_k_entities=top_k_entities,
        max_context_chars=max_context_chars,
    )
    return baseline.run(item)


# ═══════════════════════════════════════════════════════════════════════════════
# Baseline 4: Graph-Aware Retrieval
# ═══════════════════════════════════════════════════════════════════════════════


def graph_aware(
    item: dict[str, Any],
    world_model: WorldModel | None = None,
    llm: LLMClient | None = None,
    model: str = "gpt-4o",
    extraction_model: str = "gpt-4o-mini",
    max_entities: int = 15,
    max_hops: int = 2,
    max_context_chars: int = 12_000,
) -> BaselineResult:
    """
    Hybrid graph-aware retrieval baseline:
      1. Parse query intent with LLM
      2. Select seed entities (named, embedding, type-constrained)
      3. Traverse graph along relevant edges
      4. Also retrieve raw conversation chunks (RAG fallback)
      5. Combine entity context + raw chunks
      6. Answer using hybrid context
    
    This hybrid approach uses graph structure for structured info
    while keeping RAG for simple facts that weren't extracted as entities.
    """
    from pie.retrieval.graph_retriever import retrieve_subgraph
    from pie.core.temporal import format_temporal_context
    
    llm = llm or LLMClient()
    t0 = time.time()
    
    try:
        # Step 1: Build or reuse world model
        if world_model is None:
            world_model = _build_world_model_for_question(
                item, llm, extraction_model
            )
        
        question = item["question"]
        question_ts = parse_question_date(item["question_date"])
        
        # Step 2: Graph-aware retrieval
        subgraph = retrieve_subgraph(
            query=question,
            world_model=world_model,
            llm=llm,
            max_entities=max_entities,
            max_hops=max_hops,
        )
        
        # Step 3: Also do RAG on raw conversation chunks
        # This catches simple facts not extracted as entities
        chunks = _chunk_haystack(
            item["haystack_sessions"],
            item["haystack_dates"],
            chunk_by="turn",
        )
        
        # Embed and retrieve top chunks
        query_embedding = llm.embed([question])[0]
        chunk_texts = [c["text"][:2000] for c in chunks]
        chunk_embeddings = _batch_embed(chunk_texts, llm)
        
        chunk_scores = []
        for i, emb in enumerate(chunk_embeddings):
            sim = cosine_similarity(query_embedding, emb)
            chunk_scores.append((i, sim, chunks[i]))
        
        chunk_scores.sort(key=lambda x: x[1], reverse=True)
        top_chunks = chunk_scores[:5]  # Top 5 raw chunks
        
        # Step 4: Compile hybrid context
        context_parts = []
        
        # Part A: Entity-based context (if entities found)
        if subgraph.entity_ids:
            entities = subgraph.get_entities(world_model)
            
            intent = subgraph.intent
            if intent.temporal_pattern:
                context_parts.append(f"[Query analysis: type={intent.query_type}, temporal={intent.temporal_pattern}]")
            
            context_parts.append("=== Extracted Knowledge ===")
            for entity in entities[:max_entities]:
                lines = [f"\n• {entity.name} ({entity.type.value})"]
                
                state = entity.current_state
                if isinstance(state, dict):
                    for k, v in list(state.items())[:5]:
                        if v and k not in ("_is_event",):
                            lines.append(f"  {k}: {v}")
                
                context_parts.append("\n".join(lines))
        
        # Part B: Raw conversation chunks (RAG fallback)
        context_parts.append("\n=== Relevant Conversations ===")
        for idx, score, chunk in top_chunks:
            context_parts.append(f"\n[{chunk['date']}] (relevance: {score:.2f})")
            context_parts.append(chunk["text"][:1500])
        
        context = "\n".join(context_parts)[:max_context_chars]
        
        # Step 5: Answer
        answer = _ask_llm_temporal(
            context=context,
            question=question,
            question_date=item["question_date"],
            llm=llm,
            model=model,
        )
        
        return BaselineResult(
            question_id=item["question_id"],
            question_type=item["question_type"],
            question=question,
            gold_answer=item["answer"],
            hypothesis=answer,
            baseline_name="graph_aware",
            model=model,
            latency_ms=(time.time() - t0) * 1000,
            context_chars=len(context),
            retrieval_count=len(subgraph.entity_ids) + len(top_chunks),
        )
    
    except Exception as e:
        logger.exception(f"Graph-aware retrieval failed for {item['question_id']}")
        return BaselineResult(
            question_id=item["question_id"],
            question_type=item["question_type"],
            question=item["question"],
            gold_answer=item["answer"],
            hypothesis=f"Error: {e}",
            baseline_name="graph_aware",
            model=model,
            latency_ms=(time.time() - t0) * 1000,
            error=str(e),
        )


# ═══════════════════════════════════════════════════════════════════════════════
# Baseline 5: Graph-Aware Cached (fast version with pre-built world models)
# ═══════════════════════════════════════════════════════════════════════════════


class GraphAwareCachedBaseline:
    """
    Graph-aware retrieval with cached world models for faster evaluation.
    
    Uses the same caching infrastructure as PIETemporalCachedBaseline but
    adds graph traversal and bi-temporal filtering.
    """
    
    def __init__(
        self,
        cache_dir: str | Path,
        llm: LLMClient | None = None,
        model: str = "gpt-4o",
        extraction_model: str = "gpt-4o-mini",
        embed_model: str = "text-embedding-3-large",
        max_entities: int = 15,
        max_hops: int = 2,
        max_context_chars: int = 12_000,
    ):
        from pathlib import Path
        from benchmarks.common.cache import CachedWorldModel
        
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        
        self.llm = llm or LLMClient()
        self.model = model
        self.extraction_model = extraction_model
        self.embed_model = embed_model
        self.max_entities = max_entities
        self.max_hops = max_hops
        self.max_context_chars = max_context_chars
        
        self._cached_models: dict[str, "CachedWorldModel"] = {}
    
    def _get_cached_world_model(self, item: dict[str, Any]) -> "CachedWorldModel":
        from benchmarks.common.cache import CachedWorldModel
        
        qid = item["question_id"]
        if qid in self._cached_models:
            return self._cached_models[qid]
        
        cache_path = self.cache_dir / f"{qid}_world_model.json"
        
        def build_fn():
            return _build_world_model_for_question(
                item, self.llm, self.extraction_model
            )
        
        cached_wm = CachedWorldModel.load_or_build(
            cache_path=cache_path,
            build_fn=build_fn,
            llm=self.llm,
            embed_model=self.embed_model,
        )
        
        self._cached_models[qid] = cached_wm
        return cached_wm
    
    def run(self, item: dict[str, Any]) -> BaselineResult:
        from pie.retrieval.graph_retriever import retrieve_subgraph
        
        t0 = time.time()
        
        try:
            cached_wm = self._get_cached_world_model(item)
            question = item["question"]
            question_ts = parse_question_date(item["question_date"])
            
            # Graph-aware retrieval using cached world model
            subgraph = retrieve_subgraph(
                query=question,
                world_model=cached_wm.world_model,
                llm=self.llm,
                max_entities=self.max_entities,
                max_hops=self.max_hops,
            )
            
            # Also do RAG on chunks (hybrid approach)
            chunks = _chunk_haystack(
                item["haystack_sessions"],
                item["haystack_dates"],
                chunk_by="turn",
            )
            
            query_embedding = self.llm.embed([question])[0]
            chunk_texts = [c["text"][:2000] for c in chunks]
            chunk_embeddings = _batch_embed(chunk_texts, self.llm)
            
            chunk_scores = []
            for i, emb in enumerate(chunk_embeddings):
                sim = cosine_similarity(query_embedding, emb)
                chunk_scores.append((i, sim, chunks[i]))
            
            chunk_scores.sort(key=lambda x: x[1], reverse=True)
            top_chunks = chunk_scores[:5]
            
            # Compile hybrid context
            context_parts = []
            
            if subgraph.entity_ids:
                entities = subgraph.get_entities(cached_wm.world_model)
                context_parts.append("=== Extracted Knowledge ===")
                for entity in entities[:self.max_entities]:
                    lines = [f"• {entity.name} ({entity.type.value})"]
                    state = entity.current_state
                    if isinstance(state, dict):
                        for k, v in list(state.items())[:5]:
                            if v and k not in ("_is_event",):
                                lines.append(f"  {k}: {v}")
                    context_parts.append("\n".join(lines))
            
            context_parts.append("\n=== Relevant Conversations ===")
            for idx, score, chunk in top_chunks:
                context_parts.append(f"\n[{chunk['date']}] (relevance: {score:.2f})")
                context_parts.append(chunk["text"][:1500])
            
            context = "\n".join(context_parts)[:self.max_context_chars]
            
            answer = _ask_llm_temporal(
                context=context,
                question=question,
                question_date=item["question_date"],
                llm=self.llm,
                model=self.model,
            )
            
            return BaselineResult(
                question_id=item["question_id"],
                question_type=item["question_type"],
                question=question,
                gold_answer=item["answer"],
                hypothesis=answer,
                baseline_name="graph_aware_cached",
                model=self.model,
                latency_ms=(time.time() - t0) * 1000,
                context_chars=len(context),
                retrieval_count=len(subgraph.entity_ids) + len(top_chunks),
            )
        
        except Exception as e:
            logger.exception(f"Graph-aware cached failed for {item['question_id']}")
            return BaselineResult(
                question_id=item["question_id"],
                question_type=item["question_type"],
                question=item["question"],
                gold_answer=item["answer"],
                hypothesis=f"Error: {e}",
                baseline_name="graph_aware_cached",
                model=self.model,
                latency_ms=(time.time() - t0) * 1000,
                error=str(e),
            )


def graph_aware_cached(
    item: dict[str, Any],
    baseline: GraphAwareCachedBaseline | None = None,
    cache_dir: str | Path | None = None,
    llm: LLMClient | None = None,
    model: str = "gpt-4o",
    extraction_model: str = "gpt-4o-mini",
) -> BaselineResult:
    """Function wrapper for GraphAwareCachedBaseline."""
    from pathlib import Path
    
    if baseline is not None:
        return baseline.run(item)
    
    if cache_dir is None:
        cache_dir = Path(__file__).parent / "cache"
    
    baseline = GraphAwareCachedBaseline(
        cache_dir=cache_dir,
        llm=llm,
        model=model,
        extraction_model=extraction_model,
    )
    return baseline.run(item)


# ═══════════════════════════════════════════════════════════════════════════════
# Baseline registry (for runner)
# ═══════════════════════════════════════════════════════════════════════════════

BASELINES = {
    "full_context": full_context,
    "naive_rag": naive_rag,
    "pie_temporal": pie_temporal,
    "pie_temporal_cached": pie_temporal_cached,
    "graph_aware": graph_aware,
    "graph_aware_cached": graph_aware_cached,
}
