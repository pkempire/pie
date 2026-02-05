"""ChatGPT export parser. Handles the tree-structured JSON format."""

from __future__ import annotations
import json
import datetime
from collections import defaultdict
from pathlib import Path

from .models import Turn, Conversation, DailyBatch


# Content types we actually want
KEEP_CONTENT_TYPES = {"text", "multimodal_text"}

# Roles we care about
KEEP_ROLES = {"user", "assistant"}

# Skip these — they're system/tool internals
SKIP_CONTENT_TYPES = {
    "thoughts", "reasoning_recap", "code", "execution_output",
    "tether_browsing_display", "tether_quote", "computer_output",
    "system_error", "model_editable_context",
}


def _extract_text_parts(parts: list) -> str:
    """Extract text from content parts, skipping non-text (images, etc)."""
    texts = []
    for part in parts:
        if isinstance(part, str):
            texts.append(part)
        elif isinstance(part, dict):
            # Some multimodal parts have text in them
            if "text" in part:
                texts.append(part["text"])
    return "\n".join(texts).strip()


def linearize_conversation(mapping: dict, current_node: str) -> list[Turn]:
    """
    Walk from current_node to root via parent pointers, then reverse.
    This gives us the actual conversation path the user saw.
    """
    messages = []
    node_id = current_node
    visited = set()  # guard against cycles
    
    while node_id and node_id not in visited:
        visited.add(node_id)
        node = mapping.get(node_id)
        if not node:
            break
        
        msg = node.get("message")
        if msg:
            role = msg.get("author", {}).get("role", "unknown")
            content = msg.get("content", {})
            content_type = content.get("content_type", "")
            parts = content.get("parts", [])
            
            if (role in KEEP_ROLES 
                and content_type in KEEP_CONTENT_TYPES
                and content_type not in SKIP_CONTENT_TYPES):
                
                text = _extract_text_parts(parts)
                if text:
                    messages.append(Turn(
                        role=role,
                        text=text,
                        timestamp=msg.get("create_time"),
                    ))
        
        node_id = node.get("parent")
    
    messages.reverse()
    return messages


def parse_conversations(
    path: str | Path,
    year_min: int = 2023,
) -> list[Conversation]:
    """
    Parse ChatGPT JSON export into sorted Conversation objects.
    
    Args:
        path: Path to conversations.json
        year_min: Earliest year to include
    
    Returns:
        Chronologically sorted list of conversations
    """
    path = Path(path)
    with open(path) as f:
        raw = json.load(f)
    
    conversations = []
    skipped = {"no_timestamp": 0, "too_old": 0, "no_turns": 0}
    
    for c in raw:
        ts = c.get("create_time")
        if not ts:
            skipped["no_timestamp"] += 1
            continue
        
        dt = datetime.datetime.fromtimestamp(ts)
        if dt.year < year_min:
            skipped["too_old"] += 1
            continue
        
        turns = linearize_conversation(c["mapping"], c["current_node"])
        if not turns:
            skipped["no_turns"] += 1
            continue
        
        conversations.append(Conversation(
            id=c["conversation_id"],
            title=c.get("title", "Untitled"),
            created_at=ts,
            updated_at=c.get("update_time"),
            model=c.get("default_model_slug", "unknown"),
            turns=turns,
        ))
    
    # Chronological sort — critical for rolling context
    conversations.sort(key=lambda c: c.created_at)
    
    return conversations


def group_into_daily_batches(
    conversations: list[Conversation],
    max_chars_per_batch: int = 80_000,  # ~20K tokens — safe for most models
) -> list[DailyBatch]:
    """
    Group chronologically sorted conversations into daily batches.
    Heavy days (>max_chars_per_batch) get sub-batched.
    """
    by_date: dict[str, list[Conversation]] = defaultdict(list)
    
    for convo in conversations:
        date_str = datetime.datetime.fromtimestamp(convo.created_at).strftime("%Y-%m-%d")
        by_date[date_str].append(convo)
    
    batches = []
    for date_str in sorted(by_date.keys()):
        day_convos = by_date[date_str]
        total_chars = sum(c.total_chars for c in day_convos)
        
        if total_chars <= max_chars_per_batch:
            batches.append(DailyBatch(
                date=date_str,
                conversations=day_convos,
            ))
        else:
            # Sub-batch heavy days
            current_batch: list[Conversation] = []
            current_chars = 0
            sub_idx = 0
            
            for convo in day_convos:
                if current_chars + convo.total_chars > max_chars_per_batch and current_batch:
                    sub_idx += 1
                    batches.append(DailyBatch(
                        date=f"{date_str}-{sub_idx}",
                        conversations=current_batch,
                    ))
                    current_batch = []
                    current_chars = 0
                
                current_batch.append(convo)
                current_chars += convo.total_chars
            
            if current_batch:
                sub_idx += 1
                batches.append(DailyBatch(
                    date=f"{date_str}-{sub_idx}" if sub_idx > 1 else date_str,
                    conversations=current_batch,
                ))
    
    return batches


def get_stats(conversations: list[Conversation]) -> dict:
    """Get summary statistics about parsed conversations."""
    if not conversations:
        return {"count": 0}
    
    total_turns = sum(c.turn_count for c in conversations)
    total_chars = sum(c.total_chars for c in conversations)
    models = defaultdict(int)
    for c in conversations:
        models[c.model or "unknown"] += 1
    
    first = datetime.datetime.fromtimestamp(conversations[0].created_at)
    last = datetime.datetime.fromtimestamp(conversations[-1].created_at)
    
    return {
        "count": len(conversations),
        "total_turns": total_turns,
        "total_chars": total_chars,
        "avg_turns": total_turns / len(conversations),
        "avg_chars": total_chars / len(conversations),
        "date_range": f"{first.strftime('%Y-%m-%d')} → {last.strftime('%Y-%m-%d')}",
        "models": dict(models),
    }
