"""
PIE Small Test — Extract entities and state changes from 10 conversations.
Goal: See what extraction output looks like, validate approach, identify issues.
"""

import json
import datetime
import os
import sys
from openai import OpenAI

# --- Config ---
CONVERSATIONS_PATH = os.path.expanduser("~/Downloads/conversations.json")
OUTPUT_PATH = os.path.join(os.path.dirname(__file__), "test_output.json")
NUM_TEST = 10
MODEL = "gpt-4o"  # cheaper for testing, upgrade to gpt-5 for real run

client = OpenAI()

# --- Parser ---
def linearize_conversation(mapping, current_node):
    """Walk from current_node back to root, then reverse."""
    messages = []
    node_id = current_node
    while node_id:
        node = mapping.get(node_id)
        if not node:
            break
        if node.get("message"):
            msg = node["message"]
            role = msg.get("author", {}).get("role", "unknown")
            content_type = msg.get("content", {}).get("content_type", "")
            parts = msg.get("content", {}).get("parts", [])
            text_parts = [p for p in parts if isinstance(p, str)]
            text = "\n".join(text_parts).strip()
            
            if text and role in ("user", "assistant") and content_type in ("text", "multimodal_text"):
                messages.append({
                    "role": role,
                    "text": text,
                    "timestamp": msg.get("create_time"),
                })
        node_id = node.get("parent")
    messages.reverse()
    return messages


def parse_and_filter(path, year_min=2025, limit=None):
    """Parse ChatGPT export, filter to year_min+, sort chronologically."""
    with open(path) as f:
        raw = json.load(f)
    
    conversations = []
    for c in raw:
        ts = c.get("create_time")
        if not ts:
            continue
        dt = datetime.datetime.fromtimestamp(ts)
        if dt.year < year_min:
            continue
        
        turns = linearize_conversation(c["mapping"], c["current_node"])
        if not turns:
            continue
        
        conversations.append({
            "id": c["conversation_id"],
            "title": c["title"],
            "created_at": ts,
            "created_date": dt.isoformat(),
            "model": c.get("default_model_slug", "unknown"),
            "turns": turns,
            "turn_count": len(turns),
            "total_chars": sum(len(t["text"]) for t in turns),
        })
    
    conversations.sort(key=lambda c: c["created_at"])
    if limit:
        conversations = conversations[:limit]
    return conversations


# --- Extraction ---
EXTRACTION_PROMPT = """You are an entity and knowledge extractor for a personal intelligence system.

You are processing a conversation from a knowledge worker's ChatGPT history. Extract structured information about their world.

For each conversation, extract:

1. **Entities**: People, projects, tools, organizations, beliefs, decisions, concepts mentioned.
   - For each: name, type, current state as described, whether it seems new or ongoing.

2. **State Changes**: If any entity's state evolved or changed during/because of this conversation.
   - What changed, from what to what (if known), whether it contradicts prior state.

3. **Relationships**: How entities relate to each other.

4. **Temporal Context**: What period of the user's life this belongs to, what's happening around them.

5. **Significance**: How important is this conversation for understanding who this person is and what they're working on? (0.0 = trivial/one-off, 1.0 = life-defining decision)

6. **User State**: What cognitive/emotional state is the user in? (exploratory, decisive, frustrated, learning, building, etc.)

Respond in this exact JSON format:
{
  "entities": [
    {
      "name": "string",
      "type": "person|project|tool|organization|belief|decision|concept",
      "state": "current state description",
      "is_new_mention": true/false
    }
  ],
  "state_changes": [
    {
      "entity_name": "string",
      "what_changed": "description",
      "from_state": "previous state or null",
      "to_state": "new state",
      "is_contradiction": false
    }
  ],
  "relationships": [
    {
      "source": "entity name",
      "target": "entity name", 
      "type": "uses|works_on|collaborates_with|related_to|part_of|caused_by",
      "description": "brief description"
    }
  ],
  "temporal_context": "what period/phase this belongs to",
  "significance": 0.0,
  "user_state": "string",
  "summary": "one paragraph summary of what happened"
}

Be precise. Don't hallucinate entities that aren't mentioned. Extract what's actually there."""


def extract_from_conversation(convo):
    """Run extraction on a single conversation."""
    # Format turns
    turns_text = ""
    for t in convo["turns"]:
        # Truncate very long messages to avoid context limits
        text = t["text"][:3000] if len(t["text"]) > 3000 else t["text"]
        ts_str = ""
        if t.get("timestamp"):
            ts_str = f" [{datetime.datetime.fromtimestamp(t['timestamp']).strftime('%Y-%m-%d %H:%M')}]"
        turns_text += f"\n{t['role'].upper()}{ts_str}:\n{text}\n"
    
    user_msg = f"""Conversation to analyze:
Title: {convo['title']}
Date: {convo['created_date'][:10]}
Model used: {convo['model']}
Turns: {convo['turn_count']}

{turns_text}"""

    response = client.chat.completions.create(
        model=MODEL,
        messages=[
            {"role": "system", "content": EXTRACTION_PROMPT},
            {"role": "user", "content": user_msg},
        ],
        response_format={"type": "json_object"},
        temperature=0.1,
    )
    
    result = json.loads(response.choices[0].message.content)
    result["_meta"] = {
        "conversation_id": convo["id"],
        "title": convo["title"],
        "date": convo["created_date"][:10],
        "model": convo["model"],
        "turn_count": convo["turn_count"],
        "total_chars": convo["total_chars"],
        "tokens_used": {
            "prompt": response.usage.prompt_tokens,
            "completion": response.usage.completion_tokens,
            "total": response.usage.total_tokens,
        },
    }
    return result


def main():
    print(f"Parsing conversations from {CONVERSATIONS_PATH}...")
    conversations = parse_and_filter(CONVERSATIONS_PATH, year_min=2025)
    print(f"Found {len(conversations)} conversations from 2025+")
    
    # Select 10 diverse conversations (spread across time, varying sizes)
    # Pick every ~90th conversation to spread across the dataset
    step = max(1, len(conversations) // NUM_TEST)
    selected = []
    for i in range(0, len(conversations), step):
        selected.append(conversations[i])
        if len(selected) >= NUM_TEST:
            break
    
    # If we didn't get enough, pad from the end
    while len(selected) < NUM_TEST and len(selected) < len(conversations):
        selected.append(conversations[len(selected)])
    
    print(f"\nSelected {len(selected)} conversations for test extraction:")
    for i, c in enumerate(selected):
        print(f"  [{i+1}] {c['created_date'][:10]} | {c['model']:15s} | {c['turn_count']:3d} turns | {c['total_chars']:6d} chars | {c['title'][:50]}")
    
    print(f"\nRunning extraction with {MODEL}...")
    results = []
    total_tokens = 0
    for i, convo in enumerate(selected):
        print(f"  [{i+1}/{len(selected)}] {convo['title'][:50]}...", end="", flush=True)
        try:
            result = extract_from_conversation(convo)
            results.append(result)
            tokens = result["_meta"]["tokens_used"]["total"]
            total_tokens += tokens
            entities = len(result.get("entities", []))
            changes = len(result.get("state_changes", []))
            print(f" → {entities} entities, {changes} state changes, {tokens} tokens")
        except Exception as e:
            print(f" → ERROR: {e}")
            results.append({"error": str(e), "_meta": {"title": convo["title"]}})
    
    # Save results
    with open(OUTPUT_PATH, "w") as f:
        json.dump(results, f, indent=2)
    
    print(f"\n=== RESULTS ===")
    print(f"Processed: {len(results)}")
    print(f"Total tokens: {total_tokens:,}")
    print(f"Estimated cost: ${total_tokens * 5 / 1_000_000:.3f}")  # rough gpt-4o pricing
    print(f"Output saved to: {OUTPUT_PATH}")
    
    # Summary stats
    all_entities = []
    all_types = {}
    for r in results:
        if "error" in r:
            continue
        for e in r.get("entities", []):
            all_entities.append(e["name"])
            t = e.get("type", "unknown")
            all_types[t] = all_types.get(t, 0) + 1
    
    print(f"\nTotal entities extracted: {len(all_entities)}")
    print(f"Unique entities: {len(set(all_entities))}")
    print(f"Entity types: {json.dumps(all_types, indent=2)}")
    
    # Show significance distribution
    sigs = [r.get("significance", 0) for r in results if "error" not in r]
    print(f"\nSignificance scores: {sigs}")
    print(f"Average significance: {sum(sigs)/len(sigs):.2f}" if sigs else "N/A")


if __name__ == "__main__":
    main()
