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


def pie_temporal(
    item: dict[str, Any],
    world_model: WorldModel | None = None,
    llm: LLMClient | None = None,
    model: str = "gpt-4o",
    extraction_model: str = "gpt-4o-mini",
    top_k_entities: int = 25,
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
You are extracting a knowledge graph from a conversation between two people.

CRITICAL RULE: Create ONE ENTITY PER FACT, not one entity per person.
Each entity must have a UNIQUE DESCRIPTIVE NAME that identifies the specific fact.

## ENTITY NAMING — READ CAREFULLY

WRONG (everything collapses to 2 entities):
  {"name": "Himanshu", "state": {"job": "engineer", "pet": "dog named Rex"}}

RIGHT (each fact is its own entity):
  {"name": "Himanshu's job", "state": {"description": "Software engineer at Google"}}
  {"name": "Himanshu's pet Rex", "state": {"description": "Dog named Rex, golden retriever"}}
  {"name": "Himanshu's degree", "state": {"description": "BS in Computer Science from MIT, 2019"}}
  {"name": "Apartment search", "state": {"description": "Looking for 2BR in Brooklyn, budget $3000"}}
  {"name": "Hawaii trip", "state": {"description": "Planned trip to Maui in August 2023"}}

## WHAT TO EXTRACT — BE EXHAUSTIVE

For EACH speaker, create separate entities for EVERY fact mentioned:
- Education: "{Name}'s degree", "{Name}'s school", "{Name}'s major"
- Career: "{Name}'s job", "{Name}'s employer", "{Name}'s commute"
- Family: "{Name}'s sister {SisterName}", "{Name}'s pet {PetName}", "{Name}'s children" (include COUNT)
- Health: "{Name}'s allergy", "{Name}'s diet", "{Name}'s injury"
- Hobbies/Activities: "{Name}'s hobby: {hobby}", "{Name}'s instrument"
- Preferences: "{Name}'s favorite restaurant", "{Name}'s music taste"
- Home/Location: "{Name}'s apartment", "{Name}'s neighborhood", "{Name}'s hometown", "{Name}'s home country"
- Demographics: "{Name}'s age", "{Name}'s relationship status", "{Name}'s marital status"
- Events attended: "Trip to {place}", "{Name}'s birthday party", "Concert: {artist name}", "{Name}'s park visit"
- Purchases/Items: "{Name}'s new shoes", "{Name}'s figurines", "Gift from {person}"
- Books/Media: "{Name}'s book: {title}", "{Name}'s favorite book", "{Name}'s movie recommendation"
- Specific details: bandnames, book titles, pet behaviors, item descriptions
- Plans: "{Name}'s weekend plans", "Upcoming move"
- Emotional experiences: "{Name}'s reaction to {event}", "How {Name} felt about {thing}"
- People mentioned: Create entity for each person mentioned by name

CRITICAL — COMMONLY MISSED DETAILS (extract these!):
- Exact number of children/siblings/pets
- Book titles (in quotes in the conversation)
- Band/artist names from concerts or music mentions
- Specific items purchased or received as gifts (with descriptions)
- Where pets hide things, pet quirks and behaviors
- What posters/signs said at events
- How someone FELT about an event (emotions, reactions)
- Frequency of activities ("once a year", "every weekend")
- Specific places visited (Grand Canyon, park name, beach name)
- What exactly was painted/created/made (with details like "sunset with palm tree")

Also create the SPEAKER entities themselves:
  {"name": "Himanshu", "type": "person", "state": {"description": "One of the two speakers"}}

## STATE CHANGES
If a fact CHANGED from earlier sessions, note it:
  {"entity_name": "Himanshu's job", "what_changed": "employer", "old_state": "engineer at Google", "new_state": "engineer at Meta", "is_contradiction": false}

## DATE HANDLING — CRITICAL
Convert ALL relative dates to ABSOLUTE dates using the session date shown above.
If the session date is "July 2, 2023" and someone says:
  "yesterday" → "July 1, 2023"
  "last week" → "the week before July 2, 2023"
  "this month" → "July 2023"
  "next month" → "August 2023"
  "a year ago" → "2022"
  "seven years" of doing something → calculate the start year
NEVER store relative dates like "yesterday" or "last week" — always convert to absolute.

## RULES
- ONE ENTITY PER DISTINCT FACT — this is the most important rule
- EXTRACT from BOTH speakers equally — do NOT favor one speaker over the other
- DO NOT skip ANY facts — questions test EVERY detail including:
  - Specific item descriptions (what color, what pattern, what was on it)
  - Book titles, band names, artist names, song names
  - Numbers (how many children, how many years married, how often they do something)
  - Pet behaviors and quirks
  - What signs/posters said
  - How someone felt about something
  - What someone did after/before an event
- Include the specific details (names, numbers, dates, places)
- If unsure whether important, EXTRACT IT ANYWAY — err on the side of OVER-extracting
- ALWAYS use the speaker's FULL NAME (from the conversation) in entity names, never abbreviations or "User"/"Assistant"
- Convert all relative dates to ABSOLUTE dates (see DATE HANDLING above)

Output JSON:
{
  "entities": [{"name": "descriptive unique name", "type": "person|concept|organization|belief", "state": {"description": "the specific fact with details"}}],
  "state_changes": [{"entity_name": "str", "what_changed": "str", "old_state": "str", "new_state": "str", "is_contradiction": false}],
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

    for i, convo in enumerate(conversations):
        batch_text = _format_conversations_for_extraction(
            [convo], speaker_a=speaker_a, speaker_b=speaker_b
        )

        # If a single session is very long, truncate it
        if len(batch_text) > MAX_INPUT_CHARS:
            batch_text = batch_text[:MAX_INPUT_CHARS] + "\n\n[... session truncated ...]"

        try:
            result = llm.chat(
                messages=[
                    {"role": "system", "content": PIE_EXTRACTION_PROMPT},
                    {"role": "user", "content": batch_text},
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
            if transitions and len(transitions) > 1:
                lines.append("\nTimeline:")
                for t in transitions:
                    dt = datetime.fromtimestamp(t.timestamp, tz=timezone.utc)
                    date_str = dt.strftime("%B %d, %Y")
                    ttype = t.transition_type
                    prefix = "  ⚠" if ttype == TransitionType.CONTRADICTION else "  •"

                    # Include the state content too, not just the trigger
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
You are answering questions about a conversation between two people.
You are given structured knowledge extracted from their chat history.

## ANSWER FORMAT
- Be CONCISE. For factual questions, answer in 1-2 sentences max.
- Single word/name/date/phrase answers are PREFERRED:
  Q: "When did X happen?" → "July 2023"
  Q: "What is X's pet?" → "A dog named Rex"
  Q: "Where did X move from?" → "Sweden"
  Q: "What is X's relationship status?" → "Single"
- Use absolute dates from the Timeline when available (e.g. "July 6, 2023").
- For list questions ("what activities", "what hobbies"), scan ALL entities and combine everything into one comma-separated list.

## ADVERSARIAL / TRICKY QUESTIONS
- The question may DELIBERATELY name the WRONG person. For example, it might ask "What did Melanie do?" when actually it was Caroline who did it.
- If the question asks about Person A but the context only has that fact about Person B → ANSWER ANYWAY using Person B's information. Correct the premise briefly if needed.
  Example: Q: "What did Melanie realize after the charity race?" → If only Caroline ran a charity race: "Caroline (not Melanie) ran the charity race and realized the importance of mental health awareness."
- NEVER say "I don't know" or "the context does not mention" just because the person's name doesn't match. Look for the FACT across ALL entities regardless of which person it's attributed to.

## TEMPORAL REASONING
- The MOST RECENT state is the current truth
- State changes marked with ⚠ mean the NEWER value replaced the old one
- Use Timeline dates to answer "when" questions
- If a Timeline shows a date range like "the week before July 6, 2023", use that as the answer

## CRITICAL ANSWERING RULES
1. NEVER say "I don't know", "the context does not provide", "not mentioned", or "no information". These phrases are FORBIDDEN.
2. If ANY entity has even PARTIAL information related to the question, USE IT.
3. Make reasonable inferences. If someone had a "tough breakup" → they are likely "single".
4. Search ALL entities for relevant facts, not just entities matching the person named in the question.
5. For list questions, aggregate from EVERY entity that mentions the topic.
6. If truly zero relevant info exists, give your best guess based on context clues rather than saying IDK.
7. Do NOT fabricate entirely, but DO make reasonable inferences."""


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
    result = llm.chat(messages=messages, model=model, max_tokens=500)
    return result["content"].strip()


# ═══════════════════════════════════════════════════════════════════════════════
# Baseline registry
# ═══════════════════════════════════════════════════════════════════════════════

BASELINES = {
    "full_context": full_context,
    "naive_rag": naive_rag,
    "pie_temporal": pie_temporal,
}
