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
Answer the question based on the provided context. Make reasonable inferences when the context provides relevant clues.
For example:
- If the user mentions "Valentine's Day", that means February 14th
- If they mention an event related to their stated interests (animal welfare, health, etc.), connect the dots
- If a name/event appears that could match the question, use that information
Be concise and specific. Only say "I don't know" if there's truly NO relevant information.
Do NOT fabricate completely unrelated information."""

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
    max_context_chars: int = 30_000,
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
You are extracting ALL factual information about a user from their chat history.
Your goal is TOTAL RECALL — capture every single fact the user reveals about themselves,
no matter how small or casual. Benchmark accuracy depends on not missing anything.

Each session header shows the EXACT DATE: [Session — Month DD, YYYY at HH:MM]

## WHAT TO EXTRACT (capture ALL of these)

### 1. PERSONAL FACTS (HIGHEST PRIORITY)
Extract EVERY fact the user reveals about themselves:
- Education: degree, school, major, graduation year, GPA, courses
- Career: job title, employer, industry, salary, commute, work hours, office location
- Demographics: age, birthday, hometown, current city, nationality
- Family: spouse/partner name, children, parents, siblings, pets (name, breed, age)
- Health: conditions, medications, diet, allergies, doctor visits
- Daily life: commute time, routine, habits, schedule
- Hobbies: sports, instruments, games, collections, activities
- Preferences: favorite foods, restaurants, colors, brands, music, movies, books
- Skills: languages spoken, certifications, abilities
- Finances: budget mentions, purchases, subscriptions
- Home: type of dwelling, roommates, neighborhood
- Vehicle: car make/model, transportation mode

Format each as an entity:
  name: descriptive key (e.g., "user's degree", "user's commute", "user's pet")
  type: "concept" (for facts) or "person"/"organization" (for named entities)
  state: {fact: "the actual value", context: "how it was mentioned"}

### 2. PEOPLE & RELATIONSHIPS
- Names of family, friends, colleagues, doctors, etc.
- How they relate to the user
- Any facts about them (job, age, where they live, etc.)

### 3. EVENTS & ACTIVITIES
- Things the user DID (visits, purchases, trips, meetings, appointments)
- Compute EXACT DATE from session date + relative reference:
  * "today"/"just" → session date
  * "yesterday" → session date - 1 day
  * "last week" → session date - 7 days
  * "last month" → session date - ~30 days
  * "last Tuesday" → most recent Tuesday before session date

Format: name: descriptive (e.g., "MoMA visit"), type: "event",
  state: {date: "YYYY-MM-DD", description: "...", location: "..."}

### 4. PROJECTS & TOOLS
- Things the user is building or working on
- Tools, apps, technologies they use or are evaluating

### 5. STATE CHANGES
- If any previously extracted fact CHANGES (e.g., new job, moved cities)

## RULES
- Extract from the USER's messages, not the assistant's generic advice
- When the user says "I have a..." or "my ... is..." — ALWAYS extract it
- When the user mentions a name, ALWAYS extract that person
- When the user describes an activity, ALWAYS extract it as an event
- DO NOT skip "small" facts — the question might be about ANY detail
- DO NOT skip preferences, hobbies, daily routines, or personal details
- If unsure whether something is important, EXTRACT IT ANYWAY

Output JSON:
{
  "entities": [
    {"name": "str", "type": "str", "state": {"key": "value"}}
  ],
  "state_changes": [
    {"entity_name": "str", "what_changed": "str", "old_state": "str", "new_state": "str", "is_contradiction": false}
  ],
  "relationships": [
    {"source": "str", "target": "str", "type": "str", "description": "str"}
  ],
  "summary": "one-line summary of key facts extracted"
}"""


def _build_world_model_for_question(
    item: dict[str, Any],
    llm: LLMClient,
    extraction_model: str = "gpt-4o-mini",
    debug: bool = False,
    debug_log: list | None = None,
) -> WorldModel:
    """
    Build a fresh world model from a question's haystack sessions.

    Processing one session at a time to avoid truncated JSON output from the LLM.
    """
    from pie.core.models import ExtractedEntity, ExtractedStateChange, ExtractedRelationship
    from pie.core.llm import parse_extraction_result

    wm = WorldModel()
    conversations = haystack_to_conversations(
        item["haystack_sessions"],
        item["haystack_dates"],
        item["question_id"],
    )

    if debug:
        print(f"    Processing {len(conversations)} sessions...")

    MAX_INPUT_CHARS = 6_000

    for i, convo in enumerate(conversations):
        # Format single session for extraction
        batch_text = _format_conversations_for_extraction([convo])

        if len(batch_text) > MAX_INPUT_CHARS:
            batch_text = batch_text[:MAX_INPUT_CHARS] + "\n\n[... session truncated ...]"

        # Build context preamble from current world model state
        context = ""
        if wm.entities:
            context = wm.build_context_preamble(convo.created_at)

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
                max_tokens=4096,
            )

            raw = result["content"]
            if isinstance(raw, str):
                import json
                raw = json.loads(raw)

            n_before = len(wm.entities)
            _apply_extraction_to_world_model(
                raw, wm, convo.created_at, convo.id
            )
            n_new = len(wm.entities) - n_before

            if debug:
                print(f"      Session {i+1}/{len(conversations)}: "
                      f"{len(batch_text):,} chars → "
                      f"{len(raw.get('entities', []))} extracted, {n_new} new "
                      f"(total: {len(wm.entities)})")

            if debug_log is not None:
                debug_log.append({
                    "session_index": i,
                    "input_chars": len(batch_text),
                    "raw_extraction": raw,
                    "entities_found": len(raw.get("entities", [])),
                    "entities_new": n_new,
                    "total_entities": len(wm.entities),
                })

        except Exception as e:
            logger.warning(
                f"Extraction failed for session {i}: {e}"
            )
            if debug:
                print(f"      Session {i+1}/{len(conversations)}: FAILED — {e}")
            if debug_log is not None:
                debug_log.append({
                    "session_index": i,
                    "input_chars": len(batch_text),
                    "error": str(e),
                })
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
    top_k: int = 20,
) -> list[tuple[str, dict, float]]:
    """
    Retrieve relevant entities using hybrid BM25 + embedding retrieval.

    Uses Reciprocal Rank Fusion (RRF) to combine sparse (BM25) and dense
    (embedding) scores. This catches both semantic matches AND exact keyword
    matches (names, numbers, specific terms).

    Returns list of (entity_id, entity_dict, combined_score) sorted by relevance.
    """
    if not world_model.entities:
        return []

    # Build text representations for all entities
    entity_texts = {}
    for eid, entity in world_model.entities.items():
        state = entity.current_state
        if isinstance(state, dict):
            state_str = "; ".join(
                f"{k}: {v}" for k, v in state.items()
                if k not in ("_is_event", "embedding") and v
            )
        else:
            state_str = str(state)[:300]
        entity_texts[eid] = f"{entity.name} ({entity.type.value}): {state_str}"

    # --- BM25 scoring ---
    bm25_ranks = {}
    try:
        from rank_bm25 import BM25Okapi
        eids_list = list(entity_texts.keys())
        tokenized_docs = [entity_texts[eid].lower().split() for eid in eids_list]
        tokenized_query = question.lower().split()
        bm25 = BM25Okapi(tokenized_docs)
        bm25_scores = bm25.get_scores(tokenized_query)
        # Rank by BM25 score
        bm25_ranked = sorted(
            zip(eids_list, bm25_scores), key=lambda x: x[1], reverse=True
        )
        for rank, (eid, _score) in enumerate(bm25_ranked):
            bm25_ranks[eid] = rank
    except ImportError:
        logger.warning("rank_bm25 not installed, falling back to embedding-only")
        for rank, eid in enumerate(entity_texts.keys()):
            bm25_ranks[eid] = rank

    # --- Embedding scoring ---
    query_emb = llm.embed_single(question)

    needs_embed = []
    needs_embed_ids = []
    for eid, entity in world_model.entities.items():
        if entity.embedding is None:
            needs_embed.append(entity_texts[eid])
            needs_embed_ids.append(eid)

    if needs_embed:
        try:
            embeddings = _batch_embed(needs_embed, llm)
            for eid, emb in zip(needs_embed_ids, embeddings):
                world_model.entities[eid].embedding = emb
        except Exception as e:
            logger.warning(f"Batch embedding failed: {e}")

    embed_ranked = []
    for eid, entity in world_model.entities.items():
        if entity.embedding:
            sim = cosine_similarity(query_emb, entity.embedding)
            embed_ranked.append((eid, sim))
    embed_ranked.sort(key=lambda x: x[1], reverse=True)
    embed_ranks = {eid: rank for rank, (eid, _) in enumerate(embed_ranked)}

    # --- Reciprocal Rank Fusion (k=60) ---
    k = 60  # standard RRF constant
    rrf_scores = {}
    for eid in entity_texts:
        bm25_r = bm25_ranks.get(eid, len(entity_texts))
        embed_r = embed_ranks.get(eid, len(entity_texts))
        rrf_scores[eid] = (1.0 / (k + bm25_r)) + (1.0 / (k + embed_r))

    # Sort by RRF score, return top_k
    sorted_eids = sorted(rrf_scores.keys(), key=lambda eid: rrf_scores[eid], reverse=True)
    results = []
    for eid in sorted_eids[:top_k]:
        entity = world_model.entities[eid]
        results.append((eid, entity, rrf_scores[eid]))
    return results


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
    max_chars: int = 30_000,
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
You are a personal knowledge assistant answering questions about a user's life.
You are given a structured knowledge base compiled from their chat history.

Key features of the context:
- Each entity shows WHEN it first appeared and was last mentioned
- State changes are tracked chronologically
- Contradictions (where info changed) are marked with ⚠ [CHANGED]
- Relationships between entities are shown

Use temporal information to answer time-sensitive questions accurately:
- The MOST RECENT state is correct for "current" questions
- State changes marked as contradictions — the NEWER value overrides
- Use chronological ordering for "first", "last", "before", "after" questions

IMPORTANT: Answer concisely with just the fact requested.
- If the context contains the answer, give it directly (e.g., "Business Administration", "45 minutes")
- Make reasonable inferences from available context
- Only say "I don't know" if there is truly ZERO relevant information
- Do NOT repeat the question or add unnecessary explanation"""


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
        max_context_chars: int = 30_000,
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
    max_chars: int = 30_000,
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
    max_context_chars: int = 30_000,
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
# Baseline registry (for runner)
# ═══════════════════════════════════════════════════════════════════════════════

BASELINES = {
    "full_context": full_context,
    "naive_rag": naive_rag,
    "pie_temporal": pie_temporal,
    "pie_temporal_cached": pie_temporal_cached,
}
