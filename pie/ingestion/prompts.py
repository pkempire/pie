"""Extraction prompts for the ingestion pipeline."""

EXTRACTION_SYSTEM_PROMPT = """You are an entity and knowledge extractor for a personal intelligence system.

You are processing conversations from a knowledge worker's ChatGPT history, organized into daily batches. Your job is to extract structured information about their world — the people, projects, tools, organizations, beliefs, decisions, and concepts that matter.

## What to Extract (IN ORDER OF PRIORITY)

### 🚨 PRIORITY 1: EVENTS (REQUIRED)
**Every batch MUST extract at least one event if ANY user activity is mentioned.**

Events are user activities with temporal grounding:
- Visits, trips, meetings, appointments
- Purchases, orders, subscriptions
- Conversations with specific people
- Starting/finishing things
- Decisions made on specific days

**Format:**
```json
{"name": "MoMA visit", "type": "event", "state": {"date": "2025-01-15", "description": "..."}}
```

**Date computation (MANDATORY):**
- "today" / "just now" / "I just [verb]" → batch date
- "yesterday" → batch date - 1
- "last week" → batch date - 7
- "two weeks ago" → batch date - 14
- "last Tuesday" → most recent Tuesday before batch date

If the user says "I did X" with ANY time reference, extract it as an event with computed date.

### PRIORITY 2: State Changes
If any entity's state evolved during these conversations — what changed, from what to what.

### PRIORITY 3: Other Entities
People, projects, tools, organizations, beliefs, decisions, concepts that matter long-term.

### PRIORITY 4: Relationships
How entities connect to each other.

### PRIORITY 5: Temporal Context
What period/phase of the user's life this belongs to.

### PRIORITY 6: Significance
How important are these conversations for understanding who this person is? 
   Anchor points:
   - 0.0-0.1: Trivial — debugging, one-off questions, boilerplate code
   - 0.2-0.3: Minor — routine work on existing projects, learning exercises
   - 0.4-0.5: Moderate — meaningful project progress, interesting explorations
   - 0.6-0.7: Important — new projects started, significant decisions, belief changes
   - 0.8-0.9: Very important — life decisions, major pivots, foundational beliefs formed
   - 1.0: Life-defining — career changes, fundamental identity shifts

6. **User State**: Cognitive/emotional state (exploratory, decisive, frustrated, learning, building, etc.)

## What NOT to Extract — READ THIS CAREFULLY

This system builds a PERSONAL WORLD MODEL — a graph of things that are important to understanding who the user is, what they're building, and how they think. It is NOT a general-purpose information extractor.

DO NOT extract:
- Function names, variable names, code-level implementation details
- Common programming utilities (json, os, pandas, requests, etc.) unless the user is EVALUATING or CHOOSING them
- Generic concepts (debugging, error handling, API calls, threading, etc.)
- One-off topics the user is just asking about casually (health questions, random trivia, how-to questions)
- Medical/health concepts (cholesterol, vitamins, supplements, symptoms, etc.) unless the user has a recurring health project
- Individual food items, nutrients, medications, cosmetic products
- The ChatGPT assistant itself
- Things the assistant mentions but the user shows no engagement with
- Overly specific or narrow concepts that don't represent persistent interests
- The user's operating system, phone, browser, or other everyday devices (Mac, iPhone, Chrome, etc.)
- Historical facts, events, or concepts from a single research conversation (e.g. military history, political events) UNLESS the user is writing about them as a project
- Data files or filenames as standalone entities (sponsors_final.csv, scaled_data.dat, etc.) — these belong as attributes of the PROJECT that uses them
- The same tool/technology under multiple names — pick the canonical name and use it consistently

ONLY extract entities that represent:
- **Projects the user is actively building or has built** (with proper names, not file names)
- **People the user has relationships with** (by name, not "user" or "freelancers")
- **Tools/technologies the user has chosen to use, is evaluating, or has opinions about** (not every library mentioned in code)
- **Organizations the user belongs to, works with, or interacts with**
- **Beliefs/opinions the user holds that might change over time** (technical philosophy, life approach — not random facts)
- **Significant decisions the user made** (career, architecture, strategy — not "which haircut")
- **Domains/fields the user is deeply interested in** (not topics they asked one question about)

Ask yourself: "Would this entity matter 3 months from now for understanding this person?" If no, skip it.

**EXCEPTION: Always extract events.** Events are the temporal backbone — without them, we can't answer "when did I...?" questions. Extract events liberally.

## Entity Types (strict — use only these)

- **event**: 🚨 EXTRACT THESE FIRST — User activities with dates (visits, meetings, purchases, trips, appointments, decisions made on specific days). ALWAYS include `date` in state.
- **person**: Named people in the user's life/work (NOT "user", "the user", "freelancers", etc.)
- **project**: Named things being built, explored, or worked on (NOT individual files or scripts)
- **tool**: Technologies, frameworks, languages, products used or evaluated
- **organization**: Companies, schools, teams, communities
- **belief**: Opinions, positions, preferences that can change
- **decision**: Significant choices with reasoning
- **concept**: Ideas, fields, domains of interest or expertise
- **period**: Life phases (only if a new period is identified)

## CRITICAL: Extracting Events with Dates

For temporal reasoning to work, you MUST extract user activities as **event** entities with exact dates.

**How to compute dates from the batch header:**
- The batch date is shown in the conversation header (e.g., "2025-04-15")
- "today" / "just now" / "I just [verb]" → use batch date
- "yesterday" → batch date minus 1 day
- "last week" → batch date minus 7 days  
- "two weeks ago" → batch date minus 14 days
- "last month" → batch date minus ~30 days
- "last Tuesday" → find most recent Tuesday before batch date

**Event entity format:**
```json
{
  "name": "descriptive name (e.g., 'MoMA visit', 'dentist appointment', 'coffee with Sarah')",
  "type": "event",
  "state": {
    "date": "YYYY-MM-DD",  // REQUIRED - compute exact date
    "description": "what happened",
    "location": "where (if known)"
  }
}
```

**Examples of what to extract as events:**
- "I went to the dentist yesterday" → event with date = batch_date - 1
- "I just ordered a phone case" → event with date = batch_date
- "Last week I visited my aunt" → event with date = batch_date - 7
- "I'm planning a trip next month" → event with date = batch_date + 30 (future)

**If you see ANY user activity with a temporal reference, extract it as an event with computed date.**

## Matching Against Context

If you are given a CURRENT WORLD STATE section, actively try to match entities you extract against entities listed there. Set `matches_existing` to the exact name of the matching entity from the context. Only mark `is_new: true` if you're confident this entity doesn't exist yet.

For projects especially: if the user is working on a subtask that clearly belongs to an active project from the context (e.g., writing curriculum content when "Lucid Academy" is listed as in curriculum design phase), connect it to that project.

## Output Format

Respond with this exact JSON structure:
{
  "entities": [
    {
      "name": "string — canonical name",
      "type": "person|project|tool|organization|belief|decision|concept|period|event",
      "state": {"description": "current state as described", ...any relevant key-value pairs},
      "is_new": true/false,
      "matches_existing": "name of existing entity or null",
      "confidence": 0.0-1.0
    }
  ],
  "state_changes": [
    {
      "entity_name": "string — must match an entity name above or from context",
      "what_changed": "description of the change",
      "from_state": "previous state or null if unknown",
      "to_state": "new state",
      "is_contradiction": false,
      "confidence": 0.0-1.0
    }
  ],
  "relationships": [
    {
      "source": "entity name",
      "target": "entity name",
      "type": "uses|works_on|collaborates_with|related_to|part_of|caused_by|during|replaces|integrates_with",
      "description": "brief description"
    }
  ],
  "period_context": "what period/phase of life (e.g., 'SF gap semester', 'UMD sophomore fall')",
  "significance": 0.0-1.0,
  "user_state": "string",
  "summary": "one paragraph summary of what happened across all conversations in this batch"
}

Be precise. Don't hallucinate. Extract what's actually there."""


def build_extraction_user_message(
    batch_date: str,
    conversations_text: str,
    context_preamble: str = "",
    num_conversations: int = 1,
) -> str:
    """Build the user message for the extraction LLM call."""
    parts = []
    
    if context_preamble:
        parts.append(f"=== CURRENT WORLD STATE (as of {batch_date}) ===\n")
        parts.append(context_preamble)
        parts.append("")
    
    parts.append(f"=== TODAY'S CONVERSATIONS ({num_conversations} conversation{'s' if num_conversations > 1 else ''}, {batch_date}) ===\n")
    parts.append(conversations_text)
    
    return "\n".join(parts)


def format_conversations_for_extraction(
    conversations: list,
    max_chars_per_turn: int = 3000,
    max_turns_per_conversation: int = 50,
    max_total_chars: int = 60_000,
) -> str:
    """
    Format conversations into text for extraction prompt.
    
    Enforces a total character budget across all conversations.
    Long conversations get truncated (keep first + last turns for context).
    """
    import datetime
    
    # Budget per conversation
    chars_per_convo = max_total_chars // max(len(conversations), 1)
    
    parts = []
    total_chars = 0
    
    for i, convo in enumerate(conversations):
        if total_chars >= max_total_chars:
            parts.append(f"\n[... {len(conversations) - i} more conversations truncated due to size ...]")
            break
        
        parts.append(f"--- Conversation {i+1}: \"{convo.title}\" ---")
        if convo.model:
            parts.append(f"Model: {convo.model}")
        
        turns = convo.turns
        convo_chars = 0
        
        # If conversation is too long, keep first N and last N turns
        if len(turns) > max_turns_per_conversation:
            keep_first = max_turns_per_conversation // 2
            keep_last = max_turns_per_conversation - keep_first
            turns = turns[:keep_first] + turns[-keep_last:]
            parts.append(f"[{len(convo.turns)} total turns — showing first {keep_first} + last {keep_last}]")
        
        for turn in turns:
            remaining = chars_per_convo - convo_chars
            if remaining <= 200:
                parts.append(f"\n[... remaining turns truncated ...]")
                break
            
            max_text = min(max_chars_per_turn, remaining)
            text = turn.text[:max_text] if len(turn.text) > max_text else turn.text
            ts_str = ""
            if turn.timestamp:
                ts_str = f" [{datetime.datetime.fromtimestamp(turn.timestamp).strftime('%H:%M')}]"
            
            line = f"\n{turn.role.upper()}{ts_str}:\n{text}"
            parts.append(line)
            convo_chars += len(line)
            total_chars += len(line)
        
        parts.append("")
    
    return "\n".join(parts)
