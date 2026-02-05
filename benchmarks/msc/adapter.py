"""
MSC (Multi-Session Chat) → PIE Format Adapter

Converts MSC persona-grounded dialogues into PIE Conversation objects
for persona consistency evaluation.

MSC (Meta's Multi-Session Chat) dataset:
  - Multi-session dialogues grounded in persona descriptions
  - Each conversation has multiple sessions spread over time
  - Tests persona consistency across long conversations

The MSC personas from LoCoMo repo can be used to create synthetic
persona consistency tests.
"""

from __future__ import annotations
import json
import random
from datetime import datetime, timezone, timedelta
from pathlib import Path
from typing import Any

from pie.core.models import Turn, Conversation


# ── MSC Personas Loading ──────────────────────────────────────────────────────


def load_msc_personas(
    path: str | Path = "benchmarks/msc/data/msc_personas.json",
) -> list[list[str]]:
    """
    Load the MSC personas from the LoCoMo repository.
    
    Each persona has 4-8 facts about a person (e.g., hobbies, job, family).
    Returns a flat list of persona fact lists.
    """
    path = Path(path)
    with open(path) as f:
        data = json.load(f)
    
    # MSC format: {"train": [...], "test": [...], "valid": [...]}
    # Each entry has "Speaker 1" and "Speaker 2" persona lists
    all_personas = []
    
    for split in ["train", "test", "valid"]:
        if split not in data:
            continue
        for entry in data[split]:
            if "Speaker 1" in entry:
                all_personas.append(entry["Speaker 1"])
            if "Speaker 2" in entry:
                all_personas.append(entry["Speaker 2"])
    
    # If old format (just a list), return as-is
    if not all_personas and isinstance(data, list):
        return data
    
    return all_personas


def create_persona_test_case(
    personas: list[list[str]],
    num_sessions: int = 3,
    turns_per_session: int = 6,
    seed: int | None = None,
) -> dict:
    """
    Create a synthetic persona consistency test case.
    
    Returns a dict with:
      - persona: the persona facts
      - conversations: list of session conversations
      - test_questions: questions about persona facts
    """
    if seed is not None:
        random.seed(seed)
    
    # Pick a random persona (list of fact strings)
    persona = random.choice(personas)
    if isinstance(persona, list):
        persona_facts = [str(f) for f in persona if f]
    elif isinstance(persona, dict):
        persona_facts = persona.get("personas", persona.get("facts", []))
    else:
        persona_facts = [str(persona)]
    
    # Ensure we have at least some facts
    if not persona_facts:
        persona_facts = ["I enjoy spending time outdoors.", "I like reading books."]
    
    # Generate synthetic conversations mentioning persona facts
    conversations = []
    base_time = datetime(2023, 1, 1, 10, 0, tzinfo=timezone.utc)
    
    for session_idx in range(num_sessions):
        session_time = base_time + timedelta(days=session_idx * 7)
        turns = []
        
        # Weave persona facts into conversation
        for turn_idx in range(turns_per_session):
            if turn_idx % 2 == 0:
                role = "user"
            else:
                role = "assistant"
            
            # Include persona fact reference
            fact_idx = (session_idx + turn_idx) % len(persona_facts)
            fact = persona_facts[fact_idx]
            
            if role == "user":
                text = f"I was thinking about {fact.lower()}"
            else:
                text = f"That's interesting! Tell me more about that."
            
            turns.append(Turn(
                role=role,
                text=text,
                timestamp=session_time.timestamp() + (turn_idx * 30),
            ))
        
        convo = Conversation(
            id=f"msc_test_session_{session_idx}",
            title=f"Session {session_idx + 1}",
            created_at=session_time.timestamp(),
            updated_at=session_time.timestamp() + (turns_per_session * 30),
            model=None,
            turns=turns,
        )
        conversations.append(convo)
    
    # Generate test questions about persona consistency
    test_questions = []
    for i, fact in enumerate(persona_facts[:3]):
        test_questions.append({
            "question_id": f"persona_q{i}",
            "question": f"What do you know about the user's {_extract_topic(fact)}?",
            "answer": fact,
            "question_type": "persona_consistency",
        })
    
    return {
        "persona": persona_facts,
        "conversations": conversations,
        "test_questions": test_questions,
    }


def _extract_topic(fact: str) -> str:
    """Extract a topic from a persona fact for question generation."""
    topics = ["hobby", "job", "family", "interest", "background"]
    fact_lower = fact.lower()
    
    if any(word in fact_lower for word in ["work", "job", "career"]):
        return "job"
    if any(word in fact_lower for word in ["hobby", "like", "enjoy", "love"]):
        return "interests"
    if any(word in fact_lower for word in ["family", "children", "married"]):
        return "family"
    return "background"


# ── Dataset Statistics ───────────────────────────────────────────────────────


def dataset_stats(personas: list) -> dict:
    """Compute summary statistics for the MSC personas dataset."""
    total = len(personas)
    
    # Count facts per persona
    fact_counts = []
    for p in personas:
        if isinstance(p, list):
            fact_counts.append(len(p))
        elif isinstance(p, dict):
            facts = p.get("personas", p.get("facts", []))
            fact_counts.append(len(facts))
        else:
            fact_counts.append(1)
    
    # Collect sample facts for inspection
    sample_facts = []
    for p in personas[:3]:
        if isinstance(p, list):
            sample_facts.extend(p[:2])
    
    return {
        "total_personas": total,
        "facts_per_persona": {
            "mean": round(sum(fact_counts) / len(fact_counts), 1) if fact_counts else 0,
            "min": min(fact_counts) if fact_counts else 0,
            "max": max(fact_counts) if fact_counts else 0,
        },
        "sample_facts": sample_facts[:5],
    }


# ── Persona Consistency Evaluation ───────────────────────────────────────────


def generate_consistency_questions(
    persona_facts: list[str],
    num_questions: int = 5,
) -> list[dict]:
    """
    Generate questions to test persona consistency.
    
    These questions probe whether the system correctly recalls
    persona facts mentioned across sessions.
    """
    questions = []
    
    templates = [
        ("What is one of the user's hobbies or interests?", "hobby"),
        ("What does the user do for work?", "job"),
        ("What did the user mention about their family?", "family"),
        ("What is something the user enjoys doing?", "interest"),
        ("What has the user shared about their background?", "background"),
    ]
    
    for i, fact in enumerate(persona_facts[:num_questions]):
        topic = _extract_topic(fact)
        template = next((t[0] for t in templates if t[1] == topic), templates[0][0])
        
        questions.append({
            "question_id": f"consistency_{i}",
            "question": template,
            "answer": fact,
            "question_type": "persona_consistency",
            "evidence_fact": fact,
        })
    
    return questions


def format_persona_as_context(persona_facts: list[str]) -> str:
    """Format persona facts as context for LLM evaluation."""
    if not persona_facts:
        return "No persona information available."
    
    lines = ["User Persona:"]
    for i, fact in enumerate(persona_facts, 1):
        lines.append(f"  {i}. {fact}")
    return "\n".join(lines)


# ── Baseline Helpers ─────────────────────────────────────────────────────────


def parse_msc_example(example: dict) -> dict:
    """
    Parse an MSC example into a standardized format.
    
    Returns dict with:
      - conversations: list of Conversation objects
      - personas: list of persona strings
      - item_id: unique identifier
      - item_type: 'response_generation' or 'memory_qa'
    """
    # For synthetic tests, the example is already in the right format
    if "conversations" in example:
        return example
    
    # Otherwise, try to parse from raw MSC format
    conversations = []
    personas = example.get("persona", example.get("personas", []))
    
    if "dialog" in example:
        # Convert dialog to conversation
        turns = []
        for i, utterance in enumerate(example["dialog"]):
            turns.append(Turn(
                role="user" if i % 2 == 0 else "assistant",
                text=utterance,
                timestamp=1672531200 + i * 60,  # Fixed base timestamp
            ))
        
        if turns:
            conversations.append(Conversation(
                id="msc_dialog",
                title="Dialog",
                created_at=1672531200,
                updated_at=1672531200 + len(turns) * 60,
                model=None,
                turns=turns,
            ))
    
    return {
        "conversations": conversations,
        "personas": personas,
        "item_id": example.get("id", "unknown"),
        "item_type": example.get("type", "response_generation"),
    }


def format_conversations_as_text(
    conversations: list[Conversation],
    max_chars: int | None = None,
) -> str:
    """
    Format a list of conversations as a single text document.
    """
    parts = []
    total_chars = 0
    
    for i, convo in enumerate(conversations):
        dt = datetime.fromtimestamp(convo.created_at, tz=timezone.utc)
        date_str = dt.strftime("%B %d, %Y at %H:%M")
        header = f"\n[Session {i + 1} — {date_str}]"
        parts.append(header)
        total_chars += len(header)
        
        for turn in convo.turns:
            role = turn.role.capitalize()
            line = f"{role}: {turn.text}"
            
            if max_chars and total_chars + len(line) > max_chars:
                parts.append("\n[... truncated ...]")
                return "\n".join(parts)
            
            parts.append(line)
            total_chars += len(line)
        
        parts.append("")  # blank line between sessions
    
    return "\n".join(parts)


def format_personas(personas: list[str]) -> str:
    """Format persona facts for prompt."""
    if not personas:
        return ""
    
    lines = ["Speaker facts:"]
    for fact in personas:
        lines.append(f"  - {fact}")
    return "\n".join(lines)
