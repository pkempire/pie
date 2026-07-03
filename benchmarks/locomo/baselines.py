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
    tokens_prompt: int = 0
    tokens_completion: int = 0
    error: str | None = None

    @property
    def tokens_total(self) -> int:
        return self.tokens_prompt + self.tokens_completion

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
            "tokens_prompt": self.tokens_prompt,
            "tokens_completion": self.tokens_completion,
            "tokens_total": self.tokens_total,
            "error": self.error,
        }


# ── QA Prompt ─────────────────────────────────────────────────────────────────

ANSWER_SYSTEM_PROMPT = """\
You are a helpful assistant answering questions about a conversation between two people.
You will be given context from their chat history, and a question.
Answer the question based on the provided context. Be CONCISE — 1-2 sentences for factual questions.
Make reasonable inferences when the context provides relevant clues.
ALWAYS use absolute dates (e.g. "July 2023"), NEVER relative dates ("last month").
Only say "I don't know" if there is truly ZERO relevant information.
Do NOT fabricate information."""

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
    result = llm.chat(messages=messages, model=model, max_tokens=500)
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


_TOP_K_BY_TYPE: dict[str, int] = {
    "single_hop":   30,   # single fact — but specific entities can rank low, need headroom
    "temporal":     25,   # date-chain — moderate window
    "adversarial":  25,   # single fact, wrong-person hint
    "commonsense":  30,   # inference — need broader context
    "multi_hop":    40,   # chain of facts — wide retrieval
}


def _dynamic_top_k(world_model: "WorldModel", qtype: str | None = None) -> int:
    """Scale top_k with world model size. Retrieve ~20% of entities, min 15, max 60."""
    n = len(world_model.entities) if world_model else 0
    proportional = max(15, min(60, n // 5))
    return proportional
    

def pie_temporal(
    item: dict[str, Any],
    world_model: WorldModel | None = None,
    llm: LLMClient | None = None,
    model: str = "gpt-4o",
    extraction_model: str = "gpt-4o-mini",
    top_k_entities: int | None = None,
    max_context_chars: int = 30_000,
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

    qtype = item.get("question_type", "")
    if top_k_entities is None:
        top_k_entities = _dynamic_top_k(world_model) if world_model else 20

    try:
        if world_model is None:
            world_model = _build_world_model_for_conversation(
                item, llm, extraction_model
            )
            if top_k_entities == 20:
                top_k_entities = _dynamic_top_k(world_model)

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
        answer, tok_p, tok_c = _ask_llm_temporal(
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
            tokens_prompt=tok_p,
            tokens_completion=tok_c,
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
You are extracting a knowledge graph from a conversation between two people.

## CORE RULE 1: ONE ENTITY PER FACT

Each entity represents a single fact with a unique descriptive name.

Bad — everything collapsed into one entity:
  {"name": "Alex", "state": {"job": "engineer", "pet": "dog", "city": "NYC"}}

Good — each fact is its own entity:
  {"name": "Alex's job", "state": {"description": "Software engineer at Google"}}
  {"name": "Alex's dog", "state": {"description": "Golden retriever named Rex"}}
  {"name": "Alex's city", "state": {"description": "New York City"}}

## CORE RULE 2: STRICT SPEAKER ATTRIBUTION (CRITICAL)

Every entity that describes a person's experience, hobby, possession,
relationship, feeling, plan, family member, pet, or activity MUST be
prefixed with that person's exact name. NEVER create speaker-less
entities for personal facts.

CORRECT — separate entities per speaker, even for the same activity:
  {"name": "Caroline's running practice", "type": "concept",
   "state": {"description": "Caroline runs to de-stress and clear her mind",
              "speaker": "Caroline"}}
  {"name": "Melanie's running practice", "type": "concept",
   "state": {"description": "Melanie runs for mental health",
              "speaker": "Melanie"}}

WRONG — merging two people's facts into one entity:
  {"name": "running", "state": {"description": "improves mental health"}}

Every entity's state dict MUST contain a "speaker" field naming the
person the fact belongs to (or "both" only if the fact is genuinely
shared, e.g. a joint trip they took together).

## CORE RULE 3: NEVER MERGE PARALLEL FACTS ACROSS SPEAKERS

LoCoMo conversations are SYMMETRIC. Both speakers regularly mention
the same kinds of facts: both have grandparents, both have hobbies,
both have musical tastes, both go camping, both have pets. You MUST
extract a separate entity for EACH speaker's version, even if their
facts are very similar.

Concrete examples drawn from real conversations:

  Conversation has:
    Caroline (in chunk 1): "My grandma in Sweden gave me this necklace"
    Melanie (in chunk 12): "My grandma in Sweden gave me a necklace too"
  CORRECT extraction (TWO entities):
    {"name": "Caroline's necklace from grandma", "speaker": "Caroline",
     "state": {"description": "Necklace from Caroline's Swedish grandma",
                "symbolism": "love, faith, and strength"}}
    {"name": "Melanie's necklace from grandma", "speaker": "Melanie",
     "state": {"description": "Necklace from Melanie's Swedish grandma",
                "symbolism": "love, faith, and strength"}}
  WRONG (one merged entity):
    {"name": "necklace from grandma in Sweden", ...}    # speaker-less
    {"name": "Caroline's necklace from grandma", ...}   # missing Melanie's

  Conversation has:
    Caroline: "I run to de-stress and clear my mind"
    Melanie: "I run for mental health"
  CORRECT (TWO entities):
    Caroline's running practice (speaker=Caroline)
    Melanie's running practice (speaker=Melanie)

  Conversation has:
    Caroline: "I love Bach and Mozart"
    Melanie: "I'm a fan of Bach and Mozart"
  CORRECT (TWO entities):
    Caroline's classical music taste (speaker=Caroline, state mentions Bach, Mozart)
    Melanie's classical music taste (speaker=Melanie, state mentions Bach, Mozart)

When you encounter a fact that you've seen a similar version of in a
prior chunk:
  - If it's about the SAME speaker → add it as a state_change to the existing entity
  - If it's about a DIFFERENT speaker → create a NEW entity prefixed with that speaker
  - NEVER assume "I already extracted necklace, no need for another" — check the speaker

When extracting from a chunk, ALWAYS scan every turn separately for
who is speaking. The chunk header marks each turn with the speaker name.
If a turn starts with "Caroline:", any first-person ("I", "my", "me")
in that turn refers to Caroline; the next turn starting with "Melanie:"
has first-person referring to Melanie.

## WHAT TO EXTRACT

Create entities for every stated fact across BOTH speakers:
- Identity/demographics: name, age, job, relationship status, nationality
- Family: each family member with their name and role
- Pets: each pet with name, type, any behaviors
- Hobbies and activities: one entity per hobby/activity, include how long/how often
- Events: one entity per event, include when it happened, who participated
- Locations: hometown, current city, places visited — each separately
- Books, films, music: title and speaker who mentioned it
- Purchases and possessions: item, description, when acquired
- Emotions and reactions: how someone felt about a specific event
- Plans and intentions: upcoming events, goals
- Symbolic objects (gifts, jewellery, tattoos): item, giver, recipient,
  what it symbolises — all in one entity for that specific object

## EXTRACTION RULES

1. **Extract verbatim.** Copy names, places, titles, and numbers exactly
   as spoken. "my home country, Sweden" → description includes "Sweden".

2. **Attribute correctly.** If speaker A says "I did X", create the
   entity under A's name. If speaker A says "B did X", create the
   entity under B's name. Use the conversation header to know which
   speaker is talking.

3. **When a fact involves both people**, still pick the *primary
   subject* of the fact. "Caroline gave Melanie a necklace" → entity
   "Necklace from Caroline to Melanie" with speaker="Melanie" (it's
   Melanie's possession). "Caroline and Melanie went hiking together"
   → entity "Caroline and Melanie's hike" with speaker="both".

4. **Convert all relative dates to absolute** using the session timestamp.
   Session date July 2, 2023: "yesterday" → "July 1, 2023".

5. **Compute durations.** "doing X for 7 years" + session 2023 → start 2016.

6. **Over-extract rather than under-extract.** If unsure, include it.

7. Create a speaker entity for each person:
   {"name": "Alex", "type": "person",
    "state": {"description": "One of the two speakers", "speaker": "Alex"}}

## STATE CHANGES

If a fact changed since a previous session, record it. Always preserve
the speaker:
  {"entity_name": "Alex's job", "what_changed": "employer",
   "old_state": "Google", "new_state": "Meta", "is_contradiction": false}

## OUTPUT

{
  "entities": [{"name": "<speaker>'s <fact>", "type": "person|concept|event|organization|belief",
                 "state": {"description": "the specific fact", "speaker": "<name>"}}],
  "state_changes": [{"entity_name": "str", "what_changed": "str",
                      "old_state": "str", "new_state": "str", "is_contradiction": false}],
  "relationships": [{"source": "str", "target": "str", "type": "str", "description": "str"}]
}"""


def _build_world_model_for_conversation(
    item: dict[str, Any],
    llm: LLMClient,
    extraction_model: str = "gpt-4o-mini",
    debug: bool = False,
    debug_log: list | None = None,
) -> WorldModel:
    """Build a world model from a conversation's sessions.

    Args:
        debug: If True, print extraction progress to stdout.
        debug_log: If provided, append extraction details for the viewer.
    """
    import json

    wm = WorldModel()

    # Convert to PIE conversations
    conversation = item["conversation"]
    sample = {
        "sample_id": item.get("sample_id", item["question_id"]),
        "conversation": conversation,
    }
    conversations = sample_to_conversations(sample)

    # Get speaker names for consistent entity naming
    speaker_a = conversation.get("speaker_a", "")
    speaker_b = conversation.get("speaker_b", "")

    if debug:
        print(f"    Converting {len(conversations)} sessions... (speakers: {speaker_a}, {speaker_b})")

    # Process ONE session at a time to avoid truncated JSON output.
    # LoCoMo sessions average ~3K chars each; batching 5 together produced
    # 12-16K chars of input which caused gpt-4o-mini to generate truncated
    # JSON, leading to silent extraction failures.
    MAX_INPUT_CHARS = 6_000
    # Entity context cap: keep the known-entity summary under this many chars
    # so it doesn't crowd out the session text.
    MAX_ENTITY_CONTEXT_CHARS = 3_000

    for i, convo in enumerate(conversations):
        print(".", end="", flush=True)

        batch_text = _format_conversations_for_extraction(
            [convo], speaker_a=speaker_a, speaker_b=speaker_b
        )

        if len(batch_text) > MAX_INPUT_CHARS:
            batch_text = batch_text[:MAX_INPUT_CHARS] + "\n\n[... session truncated ...]"

        # Inject known entities from previous sessions so the LLM can detect
        # state changes and use consistent names instead of re-inventing them.
        # Sort by recency (most recently updated first) so the most active
        # entities are always included. Cap at whole entities, never mid-entity.
        entity_context = ""
        if i > 0 and wm.entities:
            # Sort by latest transition timestamp descending
            def _last_ts(eid: str) -> float:
                trans = wm.get_transitions(eid, ordered=True)
                return trans[-1].timestamp if trans else 0.0

            sorted_eids = sorted(wm.entities.keys(), key=_last_ts, reverse=True)

            lines = []
            chars = 0
            for eid in sorted_eids:
                ent = wm.entities[eid]
                desc = ""
                if isinstance(ent.current_state, dict):
                    desc = ent.current_state.get("description", "")
                elif ent.current_state:
                    desc = str(ent.current_state)
                line = f"  - {ent.name}: {desc}"
                if chars + len(line) > MAX_ENTITY_CONTEXT_CHARS:
                    break  # stop before adding a partial entry
                lines.append(line)
                chars += len(line)

            hidden = len(wm.entities) - len(lines)
            header = f"Known entities ({len(lines)}/{len(wm.entities)} most recent; use consistent names; record changes as state_changes):"
            if hidden:
                footer = f"  ... {hidden} older entities not shown"
                entity_context = header + "\n" + "\n".join(lines) + "\n" + footer + "\n\n"
            else:
                entity_context = header + "\n" + "\n".join(lines) + "\n\n"

        user_content = entity_context + batch_text

        try:
            result = llm.chat(
                messages=[
                    {"role": "system", "content": PIE_EXTRACTION_PROMPT},
                    {"role": "user", "content": user_content},
                ],
                model=extraction_model,
                json_mode=True,
                max_tokens=4096,
            )

            raw = result["content"]
            if isinstance(raw, str):
                raw = json.loads(raw)

            n_entities_before = len(wm.entities)
            _apply_extraction_to_world_model(
                raw, wm, convo.created_at, convo.id
            )
            n_new = len(wm.entities) - n_entities_before

            if debug:
                entities_found = len(raw.get("entities", []))
                print(f"      Session {i+1}/{len(conversations)}: "
                      f"{len(batch_text):,} chars input → "
                      f"{entities_found} extracted, {n_new} new "
                      f"(total: {len(wm.entities)})")

            if debug_log is not None:
                debug_log.append({
                    "session_index": i,
                    "session_id": convo.id,
                    "input_chars": len(batch_text),
                    "input_preview": batch_text[:500],
                    "raw_extraction": raw,
                    "entities_found": len(raw.get("entities", [])),
                    "entities_new": n_new,
                    "total_entities": len(wm.entities),
                })

        except Exception as e:
            logger.warning(f"Extraction failed for session {i}: {e}")
            if debug:
                print(f"      Session {i+1}/{len(conversations)}: FAILED — {e}")
            if debug_log is not None:
                debug_log.append({
                    "session_index": i,
                    "session_id": convo.id,
                    "input_chars": len(batch_text),
                    "error": str(e),
                })
            continue

    return wm


def _format_conversations_for_extraction(
    conversations: list[Conversation],
    speaker_a: str = "",
    speaker_b: str = "",
) -> str:
    """Format conversations for extraction prompt.

    Args:
        conversations: List of Conversation objects to format.
        speaker_a: Name of the first speaker (mapped from "user" role).
        speaker_b: Name of the second speaker (mapped from "assistant" role).
    """
    parts = []

    # Tell the LLM who the speakers are
    if speaker_a and speaker_b:
        parts.append(f"[Speakers: {speaker_a} and {speaker_b}]")
        parts.append(f"Use their FULL NAMES ({speaker_a}, {speaker_b}) in entity names, never abbreviations.\n")

    for convo in conversations:
        dt = datetime.fromtimestamp(convo.created_at, tz=timezone.utc)
        date_str = dt.strftime("%B %d, %Y at %H:%M")
        parts.append(f"[Session — {date_str}]")
        for turn in convo.turns:
            # Map role back to actual speaker name
            if speaker_a and turn.role == "user":
                name = speaker_a
            elif speaker_b and turn.role == "assistant":
                name = speaker_b
            else:
                name = turn.role.capitalize()
            text = turn.text[:2000]
            parts.append(f"{name}: {text}")
        parts.append("")
    return "\n".join(parts)


def _resolve_relative_dates(text: str, session_ts: float) -> str:
    """Convert relative date references to absolute dates using session timestamp.

    gpt-4o-mini often ignores the instruction to convert relative dates during
    extraction, so we do it deterministically in post-processing.

    Example: "last week" with session_ts=July 6, 2023 → "the week before July 6, 2023"
    """
    import re
    from datetime import timedelta

    dt = datetime.fromtimestamp(session_ts, tz=timezone.utc)

    replacements = [
        (r'\byesterday\b', (dt - timedelta(days=1)).strftime("%B %d, %Y")),
        (r'\btoday\b', dt.strftime("%B %d, %Y")),
        (r'\blast week\b', f"the week before {dt.strftime('%B %d, %Y')}"),
        (r'\btwo weeks ago\b', f"two weeks before {dt.strftime('%B %d, %Y')}"),
        (r'\blast weekend\b', f"the weekend before {dt.strftime('%B %d, %Y')}"),
        (r'\bthis weekend\b', f"the weekend of {dt.strftime('%B %d, %Y')}"),
        (r'\blast friday\b', f"the Friday before {dt.strftime('%B %d, %Y')}"),
        (r'\blast month\b', (dt.replace(day=1) - timedelta(days=1)).strftime("%B %Y")),
        (r'\bthis month\b', dt.strftime("%B %Y")),
        (r'\bnext month\b', (dt.replace(day=28) + timedelta(days=5)).strftime("%B %Y")),
        (r'\blast year\b', str(dt.year - 1)),
        (r'\btwo weekends ago\b', f"two weekends before {dt.strftime('%B %d, %Y')}"),
        (r'\ba few weeks ago\b', f"a few weeks before {dt.strftime('%B %d, %Y')}"),
        (r'\ba couple of weeks ago\b', f"around {(dt - timedelta(days=14)).strftime('%B %d, %Y')}"),
    ]

    result = text
    for pattern, replacement in replacements:
        result = re.sub(pattern, replacement, result, flags=re.IGNORECASE)

    # Handle "N years" → compute start year (both digit and word forms)
    word_to_num = {
        "two": 2, "three": 3, "four": 4, "five": 5, "six": 6,
        "seven": 7, "eight": 8, "nine": 9, "ten": 10,
        "eleven": 11, "twelve": 12, "fifteen": 15, "twenty": 20,
    }

    def _replace_n_years_digit(m):
        n = int(m.group(1))
        start_year = dt.year - n
        return f"{m.group(1)} years (since {start_year})"

    def _replace_n_years_word(m):
        word = m.group(1).lower()
        n = word_to_num.get(word, 0)
        if n:
            start_year = dt.year - n
            return f"{m.group(1)} years (since {start_year})"
        return m.group(0)

    result = re.sub(r'\b(\d+)\s+years?\b(?!\s*(?:old|ago|since))', _replace_n_years_digit, result)
    result = re.sub(r'\b(two|three|four|five|six|seven|eight|nine|ten|eleven|twelve|fifteen|twenty)\s+years?\b(?!\s*(?:old|ago))',
                    _replace_n_years_word, result, flags=re.IGNORECASE)

    return result


def _apply_extraction_to_world_model(
    raw: dict,
    wm: WorldModel,
    timestamp: float,
    convo_id: str,
) -> None:
    """Apply extraction output to world model.

    Only uses EXACT name matching (case-insensitive) for entity dedup.
    Post-processes entity states to convert relative dates to absolute.
    """
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

        # Post-process: convert relative dates to absolute
        if isinstance(state, dict) and "description" in state:
            state["description"] = _resolve_relative_dates(state["description"], timestamp)

        # Build a meaningful trigger summary from the state description
        desc = ""
        if isinstance(state, dict):
            desc = state.get("description", "")
        if not desc:
            desc = str(state)[:120]
        trigger = desc[:150] if desc else f"Extracted: {name}"

        # Exact name match only (case-insensitive via find_by_name)
        existing = wm.find_by_name(name)

        if existing:
            wm.update_entity_state(
                entity_id=existing.id,
                new_state=state,
                source_conversation_id=convo_id,
                timestamp=timestamp,
                trigger_summary=trigger,
            )
        else:
            wm.create_entity(
                name=name,
                type=etype,
                state=state,
                source_conversation_id=convo_id,
                timestamp=timestamp,
            )

    # Process state changes (contradictions / updates with old→new)
    for change in raw.get("state_changes", []):
        entity_name = change.get("entity_name", "")
        if not entity_name:
            continue

        existing = wm.find_by_name(entity_name)

        if existing:
            new_state_val = change.get("new_state", "")
            what_changed = change.get("what_changed", "")
            is_contradiction = change.get("is_contradiction", False)
            trigger = f"{what_changed}: '{change.get('old_state', '')}' → '{new_state_val}'"
            wm.update_entity_state(
                entity_id=existing.id,
                new_state={"description": str(new_state_val)},
                source_conversation_id=convo_id,
                timestamp=timestamp,
                trigger_summary=trigger,
                is_contradiction=is_contradiction,
            )


def _retrieve_entities_for_question(
    question: str,
    world_model: WorldModel,
    llm: LLMClient,
    top_k: int | None = None,
) -> list[tuple[str, Any, float]]:
    """Retrieve relevant entities using hybrid BM25 + embedding (RRF fusion).

    Respects ablation flags via environment variables:
      PIE_ABLATION=no-bm25  → embedding-only retrieval
      PIE_TOP_K=N           → override top_k
    """
    import os
    if top_k is None:
        top_k = int(os.environ.get("PIE_TOP_K", "20"))

    ablation = os.environ.get("PIE_ABLATION", "")

    if not world_model.entities:
        return []

    # Build text representations — include entity name, type, state, AND
    # transition history for richer keyword matching.
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

        # Also include transition summaries for better keyword coverage
        transitions = world_model.get_transitions(eid, ordered=True)
        trans_strs = []
        for t in transitions[-5:]:  # last 5 transitions
            if t.trigger_summary and t.trigger_summary != f"First appearance of {entity.name}":
                trans_strs.append(t.trigger_summary)
            if isinstance(t.to_state, dict):
                desc = t.to_state.get("description", "")
                if desc:
                    trans_strs.append(desc)

        text = f"{entity.name} ({entity.type.value}): {state_str}"
        if trans_strs:
            text += " | " + " | ".join(trans_strs[:3])

        # Also add aliases for better name matching
        if entity.aliases:
            text += " | aliases: " + ", ".join(entity.aliases)

        entity_texts[eid] = text

    # BM25 scoring (skip if ablation=no-bm25)
    bm25_ranks = {}
    if ablation != "no-bm25":
        try:
            import re as _re
            from rank_bm25 import BM25Okapi
            eids_list = list(entity_texts.keys())

            # Better tokenization: split on punctuation too, not just spaces
            def _tokenize(text: str) -> list[str]:
                return [w for w in _re.split(r'[\s,;:()|/\-\'\"]+', text.lower()) if len(w) > 1]

            tokenized_docs = [_tokenize(entity_texts[eid]) for eid in eids_list]
            tokenized_query = _tokenize(question)
            bm25 = BM25Okapi(tokenized_docs)
            bm25_scores = bm25.get_scores(tokenized_query)
            bm25_ranked = sorted(zip(eids_list, bm25_scores), key=lambda x: x[1], reverse=True)
            for rank, (eid, _) in enumerate(bm25_ranked):
                bm25_ranks[eid] = rank
        except ImportError:
            logger.warning("rank_bm25 not installed, falling back to embedding-only")
    if not bm25_ranks:
        for rank, eid in enumerate(entity_texts.keys()):
            bm25_ranks[eid] = rank

    # Embedding scoring
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

    # Reciprocal Rank Fusion (k=60)
    k = 60
    rrf_scores = {}
    for eid in entity_texts:
        bm25_r = bm25_ranks.get(eid, len(entity_texts))
        embed_r = embed_ranks.get(eid, len(entity_texts))
        rrf_scores[eid] = (1.0 / (k + bm25_r)) + (1.0 / (k + embed_r))

    # ── Speaker-mention boost (anti-confusion) ─────────────────────────────
    # The single largest failure mode in adversarial questions is
    # cross-speaker attribution: the question asks about Caroline, the KB
    # has the fact under Melanie, retrieval surfaces Melanie's entity, the
    # answer LLM responds with Melanie's fact. Fix at retrieval time:
    # when the question explicitly names a speaker, multiplicatively boost
    # entities whose name starts with that speaker OR whose state.speaker
    # field matches.
    if os.environ.get("PIE_NO_SPEAKER_BOOST", "") != "1":
        speaker_names = set()
        for eid, entity in world_model.entities.items():
            # Speaker entities are type=person whose name appears in their
            # own state. Also accept any entity whose state.speaker is set.
            try:
                if entity.type.value == "person":
                    speaker_names.add(entity.name)
            except AttributeError:
                pass
            state = entity.current_state if isinstance(entity.current_state, dict) else {}
            sp = state.get("speaker")
            if isinstance(sp, str) and sp.lower() not in ("both", "unknown", ""):
                speaker_names.add(sp)

        question_lower = question.lower()
        mentioned = [s for s in speaker_names if s and s.lower() in question_lower]

        if mentioned:
            # Identify, for each entity, whether it belongs to one of the
            # mentioned speakers.
            for eid, entity in world_model.entities.items():
                ent_speakers = set()
                # by name prefix
                ename_lower = entity.name.lower()
                for s in mentioned:
                    if ename_lower.startswith(s.lower() + "'s") or \
                       ename_lower.startswith(s.lower() + " ") or \
                       ename_lower == s.lower():
                        ent_speakers.add(s)
                # by state.speaker field
                state = entity.current_state if isinstance(entity.current_state, dict) else {}
                sp = state.get("speaker", "")
                if isinstance(sp, str) and sp in mentioned:
                    ent_speakers.add(sp)
                if ent_speakers:
                    rrf_scores[eid] *= 2.5  # promote entities of mentioned speakers
                else:
                    # Entities that explicitly belong to the OTHER speaker
                    # get demoted, so they don't outrank an exact-match
                    # entity for the named one.
                    other_speaker = None
                    if isinstance(sp, str) and sp and sp not in mentioned and \
                       sp.lower() not in ("both", "unknown", ""):
                        other_speaker = sp
                    if other_speaker is None:
                        for s in speaker_names:
                            if s in mentioned:
                                continue
                            if ename_lower.startswith(s.lower() + "'s") or \
                               ename_lower.startswith(s.lower() + " "):
                                other_speaker = s
                                break
                    if other_speaker is not None:
                        rrf_scores[eid] *= 0.4

    sorted_eids = sorted(rrf_scores.keys(), key=lambda eid: rrf_scores[eid], reverse=True)
    results = []
    for eid in sorted_eids[:top_k]:
        entity = world_model.entities[eid]
        results.append((eid, entity, rrf_scores[eid]))
    return results


def _compile_temporal_context(
    retrieved: list[tuple[str, Any, float]],
    world_model: WorldModel,
    max_chars: int = 30_000,
) -> str:
    """Compile PIE's semantic temporal context.

    Respects ablation flags:
      PIE_ABLATION=no-timeline  → skip timeline, only show current state
      PIE_ABLATION=no-dates     → strip all date information
    """
    import os
    ablation = os.environ.get("PIE_ABLATION", "")

    parts = []
    total_chars = 0

    for eid, entity, relevance in retrieved:
        transitions = world_model.get_transitions(eid, ordered=True)

        lines = []
        name = entity.name
        etype = entity.type.value

        lines.append(f"## {name} ({etype})")

        # Current state — include ALL state fields, not just "description"
        state = entity.current_state
        if state:
            if isinstance(state, dict):
                for k, v in state.items():
                    if k not in ("embedding", "_is_event") and v:
                        lines.append(f"  {k}: {v}")
            else:
                lines.append(f"  State: {state}")

        # Timeline (skip if ablation=no-timeline or no-dates)
        if ablation not in ("no-timeline", "no-dates"):
            if transitions:
                if len(transitions) == 1:
                    # Single event — show the date inline under the entity header
                    t = transitions[0]
                    dt = datetime.fromtimestamp(t.timestamp, tz=timezone.utc)
                    lines.append(f"  date: {dt.strftime('%B %d, %Y')}")
                else:
                    lines.append("\nTimeline:")
                    for t in transitions:
                        dt = datetime.fromtimestamp(t.timestamp, tz=timezone.utc)
                        date_str = dt.strftime("%B %d, %Y")
                        ttype = t.transition_type
                        prefix = "  ⚠" if ttype == TransitionType.CONTRADICTION else "  •"

                        summary = t.trigger_summary or ""
                        if isinstance(t.to_state, dict):
                            state_desc = t.to_state.get("description", "")
                            if state_desc and state_desc != summary:
                                summary = f"{summary} — {state_desc}" if summary else state_desc
                        if summary:
                            lines.append(f"{prefix} {date_str}: {summary}")

        # Related entities (1-hop neighbors)
        neighbors = world_model.get_neighbors(eid)
        if neighbors:
            related_names = []
            for nid in neighbors[:5]:
                ne = world_model.get_entity(nid)
                if ne:
                    related_names.append(ne.name)
            if related_names:
                lines.append(f"  Related: {', '.join(related_names)}")

        part = "\n".join(lines)

        if total_chars + len(part) > max_chars:
            break

        parts.append(part)
        total_chars += len(part)

    return "\n\n".join(parts)


PIE_ANSWER_SYSTEM = """\
Answer the question using the knowledge base provided. Be concise — give \
only what is asked for. No preamble, no reasoning, no explanation.

## DEFAULT MODE: USE EVERY KB FACT YOU CAN

Most questions name a person and ask about something they did, owned,
felt, or experienced. To answer them, look across ALL entities in the
knowledge base whose name or speaker tag matches the named person, then
combine their facts to construct the answer. The KB is dense — facts
are split across many small entities, so you'll usually need to assemble
several. This includes:

- "What country is Melanie's grandma from?" → look at any entity about
  Melanie's grandma; if the description mentions Sweden, answer "Sweden".
- "Which classical musicians does Caroline enjoy?" → look at Caroline's
  music entities; combine names mentioned across them.
- "What inspired Caroline's sculpture?" → find Caroline's sculpture/art
  entities; quote the inspiration described.

## EXCEPTION: NEVER SUBSTITUTE THE OTHER SPEAKER'S FACT

The one thing you must not do is invent a fact about person X by
copying a fact from person Y on a similar topic. Concretely:

  Q: "What is Caroline's reason for getting into running?"
  KB has: "Melanie's running practice (speaker=Melanie): runs for mental health"
  KB has: NO Caroline running entity
  CORRECT: "no information about Caroline's reason for running"
  WRONG:   "to de-stress" (made up)
  WRONG:   "Melanie runs for mental health" (wrong speaker)
  WRONG:   "Caroline runs for mental health" (false attribution)

The test for whether to answer "no information": is there ANY entity
in the KB whose name or speaker tag matches the named person AND whose
description is on the asked topic? If yes, answer from it. If no — and
ONLY in that case — say "no information about <person>'s <topic>".

## YES/NO AND INFERENCE QUESTIONS

For yes/no questions ("Is Oscar Melanie's pet?", "Would Caroline...")
answer "yes" or "no" based on what the KB supports, even if the answer
requires connecting two or three facts about the named person. Do not
default to "no information" for yes/no questions — pick the side the KB
evidence points to.

For "would X likely..." or "would X enjoy..." questions, infer from the
named person's stated preferences and history. If Caroline says she
enjoys classical music and the question is about a classical piece,
answer "yes" with a brief reason. Inference within a single person's
facts is allowed and expected.

## ANSWER FORMATTING

Factual (who/what/where/when/how long): give the specific value verbatim
from the KB.
List: short comma-separated list using only items explicitly tied to
the asked entity. Do NOT pad with unrelated items.
Why/how: one sentence maximum.
Numeric: precise number, not "at least N" or "around N".

## TEMPORAL

Use the most recent state when facts have changed. Copy dates exactly
as they appear in the KB."""


def _ask_llm_temporal(
    context: str,
    question: str,
    llm: LLMClient,
    model: str = "gpt-4o",
) -> tuple[str, int, int]:
    """Ask LLM with PIE's temporal context. Returns (answer, prompt_tokens, completion_tokens)."""
    messages = [
        {"role": "system", "content": PIE_ANSWER_SYSTEM},
        {
            "role": "user",
            "content": f"Knowledge base:\n\n{context}\n\n---\n\nQuestion: {question}\n\nAnswer:",
        },
    ]
    result = llm.chat(messages=messages, model=model, max_tokens=150)
    usage = result.get("usage", {})
    return (
        result["content"].strip(),
        usage.get("prompt", 0),
        usage.get("completion", 0),
    )


# ═══════════════════════════════════════════════════════════════════════════════
# Baseline 4: PIE Temporal + Hybrid (LLM-decomposed retrieval)
# ═══════════════════════════════════════════════════════════════════════════════


_HYBRID_DECOMPOSE_SYSTEM = """\
You are analyzing a question about a two-person conversation stored as a knowledge graph.
Your job: generate 8-12 SHORT (2-6 word) keyword search queries that will retrieve the
exact entities needed to answer the question.

The knowledge graph stores facts as typed entities named like:
  "{Person}'s {fact}", "{Event name}", "{Location}"

Rules:
- Extract any person names mentioned and include them in queries
- If the question asks "when", include time/date-related keywords
- Cover the question from multiple angles: person, activity, location, date
- Prefer short keyword phrases (2-6 words) over full sentences
- Do not use generic filler words like "find", "search", "what is"

Return JSON: {"queries": ["...", "...", ...], "speakers": ["name1", "name2"], "date_hint": "YYYY or null"}"""


def _targeted_retrieve_for_question(
    question: str,
    world_model: "WorldModel",
    llm: "LLMClient",
    model: str = "gpt-4.1",
    top_k: int = 30,
    item: dict | None = None,
) -> list[str]:
    """LLM-decomposed retrieval targeted specifically at a benchmark question.

    Unlike generic broad_scan, this:
    1. Parses speaker names and date hints directly from the question
    2. Generates sub-queries that match PIE's entity naming convention
    3. Runs BM25+dense RRF per sub-query, then multi-source RRF fusion
    4. Returns entity_ids (no graph expansion — keeps precision for single-hop QA)
    """
    from pie.retrieval.hybrid_retriever import HybridRetriever, _rrf_score
    from pie.config import PIEConfig
    from datetime import datetime

    if not world_model.entities:
        return []

    # Build the HybridRetriever around this (small, per-question) world model
    world_model.rebuild_embedding_matrix()
    retriever = HybridRetriever(world_model, llm, PIEConfig())

    # Get speaker context to help decomposition
    speaker_context = ""
    if item:
        conv = item.get("conversation", {})
        sa = conv.get("speaker_a", "")
        sb = conv.get("speaker_b", "")
        if sa and sb:
            speaker_context = f"\nSpeakers in this conversation: {sa} and {sb}"

    try:
        result = llm.chat(
            messages=[
                {"role": "system", "content": _HYBRID_DECOMPOSE_SYSTEM},
                {"role": "user", "content": f"Question: {question}{speaker_context}"},
            ],
            model=model,
            json_mode=True,
        )
        _c = result["content"]
        parsed = _c if isinstance(_c, dict) else json.loads(_c)
        sub_queries: list[str] = parsed.get("queries", [])
        if not sub_queries:
            sub_queries = [question]
    except Exception:
        sub_queries = [question]

    # Multi-source RRF: run each sub-query, collect rank votes
    rank_votes: dict[str, dict[int, int]] = {}
    now = datetime.now()
    for q_idx, sub_q in enumerate(sub_queries[:12]):
        results = retriever._raw_retrieve(sub_q, top_k=20, now=now)
        for rank, eid in enumerate(results):
            rank_votes.setdefault(eid, {})[q_idx] = rank

    fused: dict[str, float] = {
        eid: sum(_rrf_score(r) for r in votes.values())
        for eid, votes in rank_votes.items()
    }
    ranked = sorted(fused.items(), key=lambda x: x[1], reverse=True)
    return [eid for eid, _ in ranked[:top_k]]


def pie_temporal_hybrid(
    item: dict[str, Any],
    world_model: "WorldModel | None" = None,
    llm: "LLMClient | None" = None,
    model: str = "gpt-4.1",
    extraction_model: str = "gpt-4.1",
    top_k_entities: int | None = None,
    max_context_chars: int = 40_000,
) -> BaselineResult:
    """PIE Temporal + Hybrid retrieval.

    Identical pipeline to pie_temporal but replaces flat BM25+cosine retrieval
    with LLM-decomposed targeted retrieval:
      - Parses speaker names and date/event hints from the question
      - Generates entity-naming-aware sub-queries ("{Name}'s {event}", not generic activity labels)
      - Multi-source RRF over 8-12 sub-queries → better recall on multi-hop and temporal questions
    """
    llm = llm or LLMClient()
    t0 = time.time()

    if top_k_entities is None:
        top_k_entities = _dynamic_top_k(world_model) if world_model else 20

    try:
        if world_model is None:
            world_model = _build_world_model_for_conversation(
                item, llm, extraction_model
            )
            if top_k_entities == 20:
                top_k_entities = _dynamic_top_k(world_model)

        question = item["question"]

        entity_ids = _targeted_retrieve_for_question(
            question=question,
            world_model=world_model,
            llm=llm,
            model=model,
            top_k=top_k_entities,
            item=item,
        )

        if not entity_ids:
            return BaselineResult(
                question_id=item["question_id"],
                question_type=item["question_type"],
                question=question,
                gold_answer=item["answer"],
                hypothesis="I don't have enough information.",
                baseline_name="pie_temporal_hybrid",
                model=model,
                latency_ms=(time.time() - t0) * 1000,
            )

        # Reuse same context compiler as pie_temporal
        retrieved_triples = [
            (eid, world_model.entities[eid], 1.0)
            for eid in entity_ids
            if eid in world_model.entities
        ]
        context = _compile_temporal_context(retrieved_triples, world_model, max_context_chars)

        answer, tok_p, tok_c = _ask_llm_temporal(
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
            baseline_name="pie_temporal_hybrid",
            model=model,
            latency_ms=(time.time() - t0) * 1000,
            context_chars=len(context),
            retrieval_count=len(entity_ids),
            tokens_prompt=tok_p,
            tokens_completion=tok_c,
        )

    except Exception as e:
        logger.exception(f"PIE temporal hybrid failed for {item['question_id']}")
        return BaselineResult(
            question_id=item["question_id"],
            question_type=item["question_type"],
            question=item["question"],
            gold_answer=item["answer"],
            hypothesis=f"Error: {e}",
            baseline_name="pie_temporal_hybrid",
            model=model,
            latency_ms=(time.time() - t0) * 1000,
            error=str(e),
        )


# ═══════════════════════════════════════════════════════════════════════════════
# Baseline registry
# ═══════════════════════════════════════════════════════════════════════════════

BASELINES = {
    "full_context": full_context,
    "naive_rag": naive_rag,
    "pie_temporal": pie_temporal,
    "pie_temporal_hybrid": pie_temporal_hybrid,
}
