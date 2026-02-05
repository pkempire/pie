# PIE: Personal Intelligence Engine — Final Architecture

## What This Is

A system that ingests 4758 ChatGPT conversations (16M tokens, April 2023 → January 2026) into a temporal world model — a continuously evolving graph of entities, states, relationships, and procedures — queryable by any LLM via MCP.

The novel contribution: **semantic temporal reasoning over a knowledge worker's world state.** Not just "what facts do I know about the user" (Mem0) or "what's the user's psychology" (Honcho) — but "what is the full state of the user's world, how has it changed over time, and what patterns exist in those changes."

The thesis: temporal understanding → procedural memory → proactivity → agency → continual learning. Each step depends on the previous. We start at the foundation.

---

## Core Data Model

Everything in the world model is one of three things: **entities**, **state transitions**, or **procedures**.

### Entities

An entity is anything that persists across conversations and has evolving state. The graph stores them as typed nodes.

```python
class Entity(BaseModel):
    id: str                          # stable UUID
    type: EntityType                 # person | project | belief | decision | tool | concept | organization | period
    name: str                        # canonical name ("Science Research Academy")
    aliases: list[str]               # ["SRA", "scifair.tech", "the science fair platform"]
    current_state: dict              # latest known state (mutable, overwritten)
    created_from: str                # conversation_id where first extracted
    first_seen: float                # unix timestamp
    last_seen: float                 # unix timestamp
    importance: float                # 0.0 - 1.0, computed (not heuristic)
    embedding: list[float]           # vector of name + current_state summary
```

**EntityType breakdown:**

| Type | What it captures | Examples from your data |
|------|-----------------|----------------------|
| `person` | People in your life/work, their roles and relationships | Pranay (brother, co-founder), professors, collaborators |
| `project` | Things being built, explored, or worked on | Science Research Academy, PIE, Vidbreeze, AI Innovator's Program, Lucid Academy |
| `belief` | Opinions, positions, preferences that can change | "knowledge graphs > pure vector", "local-first > cloud" |
| `decision` | Choices made with reasoning and context | "use FalkorDB", "take gap semester in SF", "pivot SRA to curriculum" |
| `tool` | Technologies, frameworks, languages used | Python, GPT-4o, Neo4j, FalkorDB, Graphiti |
| `concept` | Ideas, fields, topics of interest | temporal KGs, agent memory, procedural memory, quantum consciousness |
| `organization` | Companies, schools, teams | UMD, Sanofi, CMU AirLab, Edison Scientific |
| `period` | Life phases that anchor time semantically | "freshman year", "SF gap semester", "Sanofi internship #2" |

### State Transitions

Every time an entity's state changes, the old state isn't overwritten in history — a transition is recorded.

```python
class StateTransition(BaseModel):
    id: str
    entity_id: str                   # which entity changed
    from_state: dict | None          # previous state (None if entity creation)
    to_state: dict                   # new state
    transition_type: TransitionType  # creation | update | contradiction | resolution | archival
    trigger_conversation_id: str     # which conversation caused this
    trigger_summary: str             # LLM-generated: "Parth mentioned SRA is now doing research curriculum"
    timestamp: float                 # when the source conversation happened
    confidence: float                # how confident the extraction is
```

**TransitionType:**
- `creation` — entity first appears
- `update` — state evolves naturally (project gained users, person changed role)
- `contradiction` — new state conflicts with previous (belief changed, decision reversed)
- `resolution` — a previously flagged contradiction was resolved
- `archival` — entity marked inactive/stale by consolidation

The transition history IS the temporal model. You don't need raw timestamps passed to the LLM — you have a chain of transitions with causal links and semantic descriptions.

### Procedures

Extracted from repeated patterns across state transitions. This is procedural memory.

```python
class Procedure(BaseModel):
    id: str
    description: str                 # "How Parth evaluates new technologies"
    pattern: list[str]               # ["deep research phase", "compare alternatives", "narrow to one", "build fast"]
    evidence: list[str]              # conversation_ids where this pattern was observed
    times_observed: int              # how many times this sequence appeared
    first_observed: float            # timestamp
    last_observed: float             # timestamp
    domain: str | None               # "technology evaluation", "project management", etc.
    confidence: float                # higher with more observations
```

Procedures aren't extracted per-conversation. They emerge from the consolidation engine analyzing state transition sequences across many entities. "Every time Parth starts a new project, the sequence is: deep research → narrow scope → build fast → iterate." That's a procedure extracted from observing the state transitions of Project A, Project B, Project C.

---

## Graph Structure

FalkorDB stores this as a labeled property graph. Nodes are entities, edges are relationships and transitions.

### Edge Types

```
# Relationship edges (between entities)
(:Person)-[:COLLABORATES_WITH]->(:Person)
(:Person)-[:WORKS_ON {role: "founder"}]->(:Project)
(:Project)-[:USES]->(:Tool)
(:Decision)-[:ABOUT]->(:Project)
(:Decision)-[:CAUSED_BY]->(:Belief)
(:Belief)-[:CONTRADICTS]->(:Belief)
(:Concept)-[:RELATED_TO]->(:Concept)
(:Entity)-[:DURING]->(:Period)              # semantic time anchor

# Temporal edges (state transitions as edges)
(:Entity)-[:TRANSITIONED {
    from_state: {...},
    to_state: {...},
    type: "update",
    trigger_conversation: "abc123",
    trigger_summary: "...",
    confidence: 0.9
}]->(:Entity)                               # self-edge with transition data

# Provenance edges (link to source)
(:Entity)-[:EXTRACTED_FROM]->(:Conversation)
(:StateTransition)-[:TRIGGERED_BY]->(:Conversation)
```

### Semantic Time Anchors

Instead of raw dates, the graph maintains **Period** nodes that represent meaningful life phases:

```
(:Period {name: "high school senior year", start: 1693526400, end: 1717200000})
(:Period {name: "UMD freshman year", start: 1724544000, end: 1747699200})
(:Period {name: "SF gap semester", start: 1735689600, end: ...})
(:Period {name: "Sanofi internship #1", start: ..., end: ...})
(:Period {name: "Sanofi internship #2", start: ..., end: ...})
(:Period {name: "CMU AirLab research", start: ..., end: ...})
```

Events and entities link to periods via `DURING` edges. The temporal context compiler translates graph data into LLM-readable context:

```python
def compile_temporal_context(entity: Entity, transitions: list[StateTransition], now: float) -> str:
    """
    Convert raw graph temporal data into semantic context an LLM can reason about.
    The LLM NEVER sees raw timestamps.
    """
    age = humanize_delta(now - entity.first_seen)                    # "about 16 months"
    staleness = humanize_delta(now - entity.last_seen)               # "last mentioned 3 weeks ago"
    period = get_period_for_timestamp(entity.first_seen)             # "during SF gap semester"
    change_count = len(transitions)
    change_velocity = change_count / max(months_between(entity.first_seen, entity.last_seen), 1)
    
    # Build narrative
    lines = []
    lines.append(f"{entity.name} — first appeared {age} ago ({period.name}), last referenced {staleness} ago.")
    lines.append(f"Changed state {change_count} times (~{change_velocity:.1f}x/month).")
    
    for t in transitions:
        t_period = get_period_for_timestamp(t.timestamp)
        t_ago = humanize_delta(now - t.timestamp)
        lines.append(f"  • {t_ago} ago ({t_period.name}): {t.trigger_summary}")
        if t.transition_type == "contradiction":
            lines.append(f"    ⚠ This contradicted the previous state.")
    
    return "\n".join(lines)
```

Example output that an LLM would see:

```
Science Research Academy — first appeared about 22 months ago (high school senior year),
last referenced 3 weeks ago.
Changed state 7 times (~0.3x/month).
  • 22 months ago (high school senior year): Founded with Pranay as educational platform for science fair students
  • 18 months ago (summer before UMD): Launched at scifair.tech, 30 students signed up
  • 14 months ago (UMD freshman year): Pivoted from mentoring to research curriculum
    ⚠ This contradicted the previous state.
  • 9 months ago (UMD freshman year): Expanded to 100+ students
  • 5 months ago (SF gap semester): Pranay running day-to-day, Parth in advisory role
```

The LLM gets a narrative it can reason about. No date parsing required.

---

## Ingestion Pipeline

### Phase 1: Parse

Convert ChatGPT export format into a linear sequence of conversations.

```python
def parse_chatgpt_export(path: str) -> list[Conversation]:
    """
    Parse the ChatGPT JSON export.
    
    The export uses a tree structure (mapping with parent/children pointers).
    We linearize by walking from current_node backwards to root.
    
    Skip non-text content: thoughts, reasoning_recap, code, execution_output,
    tether_browsing_display, tether_quote, computer_output, system_error.
    
    Keep: user:text, user:multimodal_text (text parts only), assistant:text
    """
    with open(path) as f:
        raw = json.load(f)
    
    conversations = []
    for c in raw:
        messages = linearize_tree(c["mapping"], c["current_node"])
        
        # Filter to substantive messages
        filtered = []
        for msg in messages:
            role = msg["author"]["role"]
            content_type = msg["content"]["content_type"]
            
            if role not in ("user", "assistant"):
                continue
            if content_type not in ("text", "multimodal_text"):
                continue
            
            text = extract_text_parts(msg["content"]["parts"])
            if not text.strip():
                continue
            
            filtered.append(Turn(
                role=role,
                text=text,
                timestamp=msg.get("create_time"),
            ))
        
        conversations.append(Conversation(
            id=c["conversation_id"],
            title=c["title"],
            created_at=c["create_time"],
            updated_at=c["update_time"],
            model=c.get("default_model_slug"),
            turns=filtered,
        ))
    
    # Sort chronologically — critical for rolling context ingestion
    conversations.sort(key=lambda c: c.created_at)
    return conversations
```

From the actual data: 4758 conversations, ~17K user messages, ~24K assistant messages, ~16M tokens total. Sorted chronologically from April 2023 → January 2026.

### Phase 2: Extract (LLM, every conversation, with rolling context)

Every single conversation gets processed by GPT-5. No filtering. No heuristics. The LLM receives:

1. **The conversation itself**
2. **Rolling world model context** — a summary of the current world model state relevant to entities mentioned in this conversation
3. **An extraction schema** — what to extract

```python
async def ingest_conversation(
    conversation: Conversation,
    world_model: WorldModel,
    llm: LLM,
) -> ExtractionResult:
    """
    Process one conversation through GPT-5 with full world model context.
    """
    
    # Step 1: Quick entity mention detection (LLM, not regex)
    # Ask the LLM what entities are mentioned so we can pull relevant context
    mentioned = await llm.extract_mentions(conversation)
    
    # Step 2: Pull relevant world model context for those entities
    # This is NOT the whole world model — just the subgraph around mentioned entities
    context = world_model.get_context_for_entities(mentioned)
    
    # Step 3: Full extraction with context
    extraction = await llm.extract(
        system=EXTRACTION_PROMPT,
        context=f"""
        Current world model state for relevant entities:
        {context}
        
        Conversation to process:
        Title: {conversation.title}
        Date: {format_as_period(conversation.created_at)}  # "March 2024, during UMD freshman year"
        Model: {conversation.model}
        
        {format_turns(conversation.turns)}
        """,
        response_model=ExtractionResult,  # structured output
    )
    
    return extraction
```

**The extraction prompt asks for:**

```python
class ExtractionResult(BaseModel):
    """What GPT-5 extracts from each conversation."""
    
    # What entities appear? (new or existing)
    entities: list[ExtractedEntity]
    
    # What relationships exist between entities?
    relationships: list[ExtractedRelationship]
    
    # What state changes happened to existing entities?
    state_changes: list[ExtractedStateChange]
    
    # What new beliefs or opinions were expressed?
    beliefs: list[ExtractedBelief]
    
    # What decisions were made?
    decisions: list[ExtractedDecision]
    
    # What period of life does this conversation belong to?
    period_context: str
    
    # One-paragraph summary of what happened in this conversation
    summary: str
    
    # How significant is this conversation for understanding the user's world?
    # NOT based on length — based on content significance
    significance: float  # 0.0 - 1.0
    
    # What is the user's emotional/cognitive state in this conversation?
    user_state: str | None  # "exploratory", "frustrated", "decisive", "learning", etc.

class ExtractedEntity(BaseModel):
    name: str                        # canonical name
    type: EntityType
    state: dict                      # current state as described in this conversation
    is_new: bool                     # does this entity already exist in the world model?
    matches_existing: str | None     # if not new, which existing entity does this match?

class ExtractedStateChange(BaseModel):
    entity_name: str
    what_changed: str                # human-readable description
    old_state_aspect: str | None     # what it was before (if known from context)
    new_state_aspect: str            # what it is now
    is_contradiction: bool           # does this conflict with the known state?
    confidence: float
```

### Phase 3: Entity Resolution

After extraction, match extracted entities to existing graph nodes.

```python
async def resolve_entities(
    extracted: list[ExtractedEntity],
    world_model: WorldModel,
    llm: LLM,
) -> list[ResolvedEntity]:
    """
    Match extracted entities to existing graph nodes.
    Uses embedding similarity + LLM verification.
    """
    resolved = []
    
    for entity in extracted:
        if entity.is_new and entity.matches_existing is None:
            # LLM thinks this is new — verify by checking similar existing entities
            candidates = world_model.find_similar_entities(
                name=entity.name,
                type=entity.type,
                top_k=5,
            )
            
            if candidates and candidates[0].similarity > 0.7:
                # Possible match — ask LLM to verify
                is_same = await llm.verify_entity_match(
                    new=entity,
                    candidate=candidates[0],
                )
                if is_same:
                    resolved.append(ResolvedEntity(
                        extracted=entity,
                        matched_to=candidates[0].id,
                        action="update",
                    ))
                    continue
            
            # Genuinely new entity
            resolved.append(ResolvedEntity(
                extracted=entity,
                matched_to=None,
                action="create",
            ))
        else:
            # LLM already identified the match
            existing = world_model.find_by_name(entity.matches_existing)
            resolved.append(ResolvedEntity(
                extracted=entity,
                matched_to=existing.id if existing else None,
                action="update" if existing else "create",
            ))
    
    return resolved
```

### Phase 4: World Model Update

Apply extracted data to the graph.

```python
async def update_world_model(
    conversation: Conversation,
    extraction: ExtractionResult,
    resolved_entities: list[ResolvedEntity],
    world_model: WorldModel,
):
    """Apply extraction results to the world model graph."""
    
    # Store the conversation node itself (for provenance)
    convo_node = world_model.create_node(
        type="conversation",
        name=conversation.title,
        properties={
            "summary": extraction.summary,
            "significance": extraction.significance,
            "user_state": extraction.user_state,
            "model": conversation.model,
        },
        timestamp=conversation.created_at,
    )
    
    # Create/update entities
    for resolved in resolved_entities:
        if resolved.action == "create":
            node_id = world_model.create_entity(
                type=resolved.extracted.type,
                name=resolved.extracted.name,
                initial_state=resolved.extracted.state,
                source_conversation=convo_node.id,
                timestamp=conversation.created_at,
            )
        else:
            # Update existing entity — create state transition
            world_model.update_entity_state(
                entity_id=resolved.matched_to,
                new_state=resolved.extracted.state,
                source_conversation=convo_node.id,
                timestamp=conversation.created_at,
            )
    
    # Add relationships
    for rel in extraction.relationships:
        world_model.add_relationship(
            from_entity=rel.source,
            to_entity=rel.target,
            type=rel.relationship_type,
            properties=rel.properties,
            source_conversation=convo_node.id,
            timestamp=conversation.created_at,
        )
    
    # Handle state changes (including contradictions)
    for change in extraction.state_changes:
        entity = world_model.find_by_name(change.entity_name)
        if entity:
            world_model.record_transition(
                entity_id=entity.id,
                from_state=change.old_state_aspect,
                to_state=change.new_state_aspect,
                transition_type="contradiction" if change.is_contradiction else "update",
                trigger_conversation=convo_node.id,
                trigger_summary=change.what_changed,
                timestamp=conversation.created_at,
                confidence=change.confidence,
            )
    
    # Handle beliefs (first-class entities with their own transitions)
    for belief in extraction.beliefs:
        world_model.upsert_belief(
            statement=belief.statement,
            domain=belief.domain,
            confidence=belief.confidence,
            source_conversation=convo_node.id,
            timestamp=conversation.created_at,
        )
    
    # Handle decisions
    for decision in extraction.decisions:
        world_model.record_decision(
            summary=decision.summary,
            reasoning=decision.reasoning,
            alternatives=decision.alternatives_considered,
            related_entities=decision.related_entity_names,
            source_conversation=convo_node.id,
            timestamp=conversation.created_at,
        )
    
    # Link conversation to period
    period = world_model.get_or_create_period(
        extraction.period_context,
        conversation.created_at,
    )
    world_model.add_relationship(convo_node.id, period.id, "DURING")
```

### Phase 5: Full Ingestion Orchestrator

```python
async def ingest_all(export_path: str, world_model: WorldModel, llm: LLM):
    """
    Process all 4758 conversations chronologically.
    Rolling context: each conversation sees the world model state
    built from all prior conversations.
    """
    conversations = parse_chatgpt_export(export_path)
    # Already sorted chronologically
    
    total = len(conversations)
    for i, convo in enumerate(conversations):
        logger.info(f"[{i+1}/{total}] {convo.title} ({format_period(convo.created_at)})")
        
        # Extract with current world model as context
        extraction = await ingest_conversation(convo, world_model, llm)
        
        # Resolve entities
        resolved = await resolve_entities(extraction.entities, world_model, llm)
        
        # Update world model
        await update_world_model(convo, extraction, resolved, world_model)
        
        # Periodic consolidation (every 100 conversations)
        if i > 0 and i % 100 == 0:
            logger.info(f"Running consolidation pass at conversation {i}...")
            await consolidate(world_model, llm)
    
    # Final consolidation
    logger.info("Running final consolidation...")
    await consolidate(world_model, llm)
    
    # Extract procedures from state transition patterns
    logger.info("Extracting procedures...")
    await extract_procedures(world_model, llm)
    
    stats = world_model.get_stats()
    logger.info(f"Done. {stats.entity_count} entities, {stats.transition_count} transitions, "
                f"{stats.relationship_count} relationships, {stats.procedure_count} procedures.")
```

**Cost estimate for full ingestion (GPT-5, no shortcuts):**
- 4758 conversations × ~2 LLM calls each (mention detection + full extraction)
- Average ~5K tokens input per call (conversation + rolling context), ~1K output
- ~4758 × 2 × 6K = ~57M tokens total
- At GPT-5 pricing (~$10/M input, ~$30/M output): ~$57 input + ~$285 output ≈ **~$350**
- Plus entity resolution calls, consolidation calls: probably **~$500 total**

---

## Forgetting & Consolidation

### How Importance is Computed

Every entity has an importance score. It's NOT a heuristic based on message count. It's computed from graph structure:

```python
def compute_importance(entity: Entity, world_model: WorldModel) -> float:
    """
    Importance is a function of graph structure, not surface features.
    """
    # How connected is this entity? (PageRank-like)
    connectivity = world_model.get_degree(entity.id, weighted=True)
    max_connectivity = world_model.get_max_degree()
    connectivity_score = connectivity / max_connectivity
    
    # How many state transitions? (entities that change are important)
    transitions = world_model.get_transitions(entity.id)
    transition_score = min(len(transitions) / 10, 1.0)
    
    # How recently referenced?
    recency = 1.0 / (1.0 + days_since(entity.last_seen) / 30)  # decays over months
    
    # Is it connected to other important entities?
    neighbor_importance = mean([
        world_model.get_entity(n).importance 
        for n in world_model.get_neighbors(entity.id)
    ]) if world_model.get_neighbors(entity.id) else 0
    
    # Was it ever queried/retrieved? (access-based reinforcement)
    access_score = min(entity.access_count / 5, 1.0)
    
    # Weighted combination
    return (
        0.25 * connectivity_score +
        0.20 * transition_score +
        0.20 * recency +
        0.20 * neighbor_importance +
        0.15 * access_score
    )
```

### What Forgetting Looks Like

Forgetting is NOT deletion. It's tiered archival:

**Tier 1: Active** — Full detail in graph. All transitions preserved. Retrieved in queries.

**Tier 2: Summarized** — Individual transitions merged into a summary. Entity still in graph but with compressed history. "Had 15 conversations about CSS debugging in Q3 2024 → resolved various styling issues, no lasting impact."

**Tier 3: Archived** — Removed from main graph, stored in cold archive. Still recoverable but never surfaces in queries.

Consolidation decides what moves between tiers:

```python
async def consolidate(world_model: WorldModel, llm: LLM):
    """
    Periodic consolidation: merge, summarize, archive.
    Run every ~100 conversations during ingestion, and on schedule post-ingestion.
    """
    
    # 1. MERGE similar low-importance entities
    # Find entities with high embedding similarity that might be duplicates
    # or closely related enough to merge
    similar_pairs = world_model.find_similar_entity_pairs(threshold=0.90)
    for a, b in similar_pairs:
        should_merge = await llm.should_merge_entities(a, b)
        if should_merge:
            world_model.merge_entities(a, b)  # preserves all transitions from both
    
    # 2. SUMMARIZE low-importance entities with many transitions
    verbose_entities = world_model.find_entities(
        min_transitions=10,
        max_importance=0.3,
    )
    for entity in verbose_entities:
        transitions = world_model.get_transitions(entity.id)
        summary = await llm.summarize_transitions(entity, transitions)
        world_model.compress_transitions(entity.id, summary)  # replace many transitions with one summary
    
    # 3. ARCHIVE stale low-importance entities
    stale = world_model.find_entities(
        max_importance=0.1,
        last_seen_before=days_ago(180),
    )
    for entity in stale:
        world_model.archive_entity(entity.id)
    
    # 4. RECOMPUTE importance scores (after structural changes)
    for entity in world_model.get_all_active_entities():
        entity.importance = compute_importance(entity, world_model)
        world_model.update_entity(entity)
    
    # 5. DETECT contradictions that haven't been resolved
    contradictions = world_model.find_unresolved_contradictions()
    for c in contradictions:
        resolution = await llm.attempt_contradiction_resolution(c)
        if resolution.resolved:
            world_model.resolve_contradiction(c.id, resolution)
    
    # 6. UPDATE period nodes
    # As more conversations are processed, period boundaries become clearer
    await update_period_boundaries(world_model, llm)
```

### What Happens to "Random Temp Tasks"

A conversation like "fix this CSS bug" produces:
- Maybe one entity: a Tool node for the specific CSS property, or nothing at all
- Low significance score from extraction LLM
- No connections to important entities
- No state transitions on important entities

After consolidation:
- If similar CSS conversations exist, they merge: "Various CSS debugging, Q3 2024"
- If isolated, importance decays to <0.1 after 6 months
- Gets archived in Tier 3
- Never surfaces in queries

A conversation like "actually let's pivot SRA to research curriculum" produces:
- State transition on Science Research Academy (high-connectivity entity)
- Contradiction flag (previous approach was "mentoring")
- Decision entity with reasoning
- High significance score from extraction LLM

After consolidation:
- Importance stays high (connected to important entities, has contradictions)
- Transitions preserved in full detail
- Surfaces whenever SRA is queried

The system doesn't need hand-crafted rules to distinguish these. The graph structure itself separates signal from noise.

---

## Procedural Memory Extraction

Procedures are extracted AFTER initial ingestion, by analyzing state transition patterns across entities.

```python
async def extract_procedures(world_model: WorldModel, llm: LLM):
    """
    Find recurring patterns in state transition sequences.
    These are the user's 'how I do things' patterns.
    """
    
    # Get all entities of types that have meaningful lifecycles
    projects = world_model.get_entities(type="project")
    
    # For each project, get the sequence of state transitions
    project_lifecycles = []
    for project in projects:
        transitions = world_model.get_transitions(project.id, ordered=True)
        lifecycle = [t.trigger_summary for t in transitions]
        project_lifecycles.append({
            "entity": project.name,
            "transitions": lifecycle,
            "duration": project.last_seen - project.first_seen,
        })
    
    # Ask LLM to find recurring patterns across lifecycles
    procedures = await llm.extract_patterns(
        prompt="""
        Below are the state transition histories for multiple projects.
        Identify recurring patterns — sequences of events or phases that 
        appear across multiple projects. These represent the user's 
        habitual approach to projects.
        
        For each pattern found:
        - Describe the pattern
        - List which projects exhibit it
        - Note any evolution (did the pattern change over time?)
        """,
        data=project_lifecycles,
        response_model=list[Procedure],
    )
    
    for procedure in procedures:
        world_model.store_procedure(procedure)
    
    # Same for beliefs — how do beliefs typically evolve?
    beliefs = world_model.get_entities(type="belief")
    belief_evolutions = []
    for belief in beliefs:
        transitions = world_model.get_transitions(belief.id, ordered=True)
        if len(transitions) >= 2:
            belief_evolutions.append({
                "belief": belief.name,
                "evolution": [t.trigger_summary for t in transitions],
            })
    
    if belief_evolutions:
        belief_patterns = await llm.extract_patterns(
            prompt="""
            Below are the evolution histories of the user's beliefs.
            Identify patterns in how beliefs change — what triggers changes,
            how long beliefs persist, what kinds of beliefs are most stable vs volatile.
            """,
            data=belief_evolutions,
            response_model=list[Procedure],
        )
        for p in belief_patterns:
            world_model.store_procedure(p)
    
    # Same for decisions — how does the user typically make decisions?
    decisions = world_model.get_entities(type="decision")
    # Group decisions by domain/context and look for patterns in reasoning
    decision_patterns = await llm.extract_patterns(
        prompt="""
        Below are decisions the user made over time, with their reasoning.
        Identify patterns in decision-making — what factors consistently 
        drive decisions, how alternatives are evaluated, what biases appear.
        """,
        data=[{
            "decision": d.name,
            "reasoning": d.current_state.get("reasoning"),
            "period": get_period_for_timestamp(d.first_seen).name,
        } for d in decisions],
        response_model=list[Procedure],
    )
    for p in decision_patterns:
        world_model.store_procedure(p)
```

---

## Query Layer (MCP Server)

The MCP server is the interface. Any LLM client connects and queries.

### Core Operations

```python
@mcp.tool()
async def query_world(query: str) -> str:
    """
    Primary query interface. Takes a natural language question,
    retrieves relevant subgraph, compiles semantic temporal context,
    returns LLM-ready text.
    """
    # 1. Understand the query intent
    intent = await llm.classify_query(query)
    # → "temporal_diff" | "current_state" | "entity_lookup" | "procedural" | "relationship" | "contradiction"
    
    # 2. Identify relevant entities
    entities = await llm.extract_query_entities(query)
    
    # 3. Retrieve relevant subgraph
    if intent == "temporal_diff":
        # "How did X change between period A and period B?"
        subgraph = world_model.get_transitions_in_range(entities, time_a, time_b)
    elif intent == "current_state":
        # "What's the current state of X?"
        subgraph = world_model.get_current_state(entities, with_neighbors=True)
    elif intent == "procedural":
        # "How do I usually approach X?"
        subgraph = world_model.get_procedures(domain=infer_domain(query))
    elif intent == "relationship":
        # "How is X connected to Y?"
        subgraph = world_model.get_paths_between(entities)
    elif intent == "contradiction":
        # "What conflicts exist about X?"
        subgraph = world_model.get_contradictions(entities)
    else:
        # General: hybrid retrieval
        subgraph = world_model.hybrid_retrieve(query, top_k=20)
    
    # 4. Compile semantic temporal context (NO raw dates)
    context = compile_temporal_context_for_subgraph(subgraph)
    
    # 5. Add provenance links
    context += "\n\nSources:\n"
    for source in subgraph.source_conversations:
        context += f"  - '{source.title}' ({format_period(source.created_at)})\n"
    
    return context

@mcp.tool()
async def get_world_snapshot(period: str | None = None) -> str:
    """
    Get a snapshot of the entire world model state at a point in time.
    If period is None, returns current state.
    """
    if period:
        # Reconstruct state at that period by replaying transitions
        snapshot = world_model.reconstruct_state_at_period(period)
    else:
        snapshot = world_model.get_current_state_all()
    
    return compile_world_snapshot(snapshot)

@mcp.tool()
async def get_entity_timeline(entity_name: str) -> str:
    """
    Get the full evolution history of an entity,
    compiled as semantic temporal narrative.
    """
    entity = world_model.find_by_name(entity_name)
    transitions = world_model.get_transitions(entity.id, ordered=True)
    return compile_temporal_context(entity, transitions, now=time.time())

@mcp.tool()
async def get_procedures(domain: str | None = None) -> str:
    """
    Get extracted procedural patterns.
    Optionally filter by domain.
    """
    procedures = world_model.get_procedures(domain=domain)
    return format_procedures(procedures)

@mcp.tool()
async def get_contradictions() -> str:
    """Get all unresolved contradictions in the world model."""
    contradictions = world_model.find_unresolved_contradictions()
    return format_contradictions(contradictions)

@mcp.tool()
async def diff_periods(period_a: str, period_b: str) -> str:
    """
    What changed in the world model between two periods?
    E.g., diff_periods("UMD freshman year", "SF gap semester")
    """
    state_a = world_model.reconstruct_state_at_period(period_a)
    state_b = world_model.reconstruct_state_at_period(period_b)
    
    diff = compute_world_diff(state_a, state_b)
    return format_diff(diff)

@mcp.tool()
async def ingest_episode(content: str, source: str = "manual") -> str:
    """
    Ingest a new piece of information into the world model.
    Used for real-time ingestion from live conversations.
    """
    extraction = await extract_from_text(content, world_model, llm)
    resolved = await resolve_entities(extraction.entities, world_model, llm)
    await update_world_model_from_extraction(extraction, resolved, world_model)
    return f"Ingested. Updated {len(resolved)} entities."
```

---

## Dreaming Engine

Background process that reasons over the world model to surface insights not explicit in any single conversation.

Runs on a schedule, not per-query.

```python
class DreamingEngine:
    """
    Three tiers of dreaming:
    - micro: after each new ingestion batch (quick, cheap model)
    - deep: nightly (full graph analysis, strong model)
    - mega: weekly (cross-domain synthesis, strongest model)
    """
    
    async def micro_dream(self, recent_conversations: list[str]):
        """Run after each batch of new conversations."""
        # What new connections can we make from the latest data?
        recent_entities = world_model.get_recently_updated(hours=1)
        for entity in recent_entities:
            # Check: does this entity connect to anything unexpected?
            neighbors = world_model.get_neighbors(entity.id, max_hops=2)
            distant_similar = world_model.find_similar_entities(
                entity.name, top_k=5, exclude_neighbors=True
            )
            if distant_similar:
                for similar in distant_similar:
                    insight = await llm_cheap.reason(
                        f"Entity '{entity.name}' was just updated. "
                        f"It seems related to '{similar.name}' but they're not connected in the graph. "
                        f"Is there a meaningful connection?",
                    )
                    if insight.is_meaningful:
                        world_model.add_relationship(entity.id, similar.id, "RELATED_TO",
                                                      properties={"discovered_by": "dreaming", "reason": insight.reason})
    
    async def deep_dream(self):
        """Nightly: full graph analysis."""
        # 1. Consolidation pass
        await consolidate(world_model, llm_strong)
        
        # 2. Procedure extraction/update
        await extract_procedures(world_model, llm_strong)
        
        # 3. Belief evolution analysis
        beliefs = world_model.get_entities(type="belief", min_transitions=2)
        for belief in beliefs:
            history = world_model.get_transitions(belief.id, ordered=True)
            evolution = await llm_strong.analyze_belief_evolution(belief, history)
            world_model.update_entity_metadata(belief.id, {"evolution_summary": evolution})
        
        # 4. Stale entity detection
        # What entities haven't been referenced in a while but were once important?
        potentially_stale = world_model.find_entities(
            min_importance=0.3,  # was important
            last_seen_before=days_ago(60),  # but not seen recently
        )
        if potentially_stale:
            world_model.flag_for_review(potentially_stale)
    
    async def mega_dream(self):
        """Weekly: cross-domain synthesis."""
        # What high-level patterns span multiple projects and time periods?
        all_procedures = world_model.get_procedures()
        all_periods = world_model.get_periods()
        
        synthesis = await llm_strongest.synthesize(
            prompt="""
            Given the user's full world model — their projects, beliefs, decisions,
            and extracted procedures — what high-level observations can you make?
            
            Think about:
            - How has the user's thinking evolved across all domains?
            - What blind spots exist (topics never explored, perspectives never considered)?
            - What connections between different life domains are under-explored?
            - What predictions can you make about where the user is headed?
            """,
            context=world_model.get_global_summary(),
        )
        
        world_model.store_mega_dream_insight(synthesis)
```

---

## Tech Stack

| Component | Choice | Rationale |
|-----------|--------|-----------|
| **Graph DB** | FalkorDB | Redis-based, vector search built in, Graphiti supports it, one Docker command |
| **Extraction LLM** | GPT-5 | Best extraction quality, no cost constraints |
| **Reasoning LLM** | GPT-5 / Claude Opus | Dreaming and synthesis need the best reasoning |
| **Cheap LLM** | GPT-5-mini or Haiku | Micro-dreams, quick classifications |
| **Embeddings** | text-embedding-3-large | Best quality for entity resolution and similarity |
| **MCP Framework** | Python MCP SDK | Standard protocol, any client connects |
| **Graph Library** | Graphiti-core | Temporal KG with hybrid retrieval already built |
| **Orchestration** | Python + asyncio | Simple, proven for batch + background processing |

**Not needed:**
- Separate vector DB (FalkorDB has vector search built in)
- ChromaDB / Qdrant / Pinecone (unnecessary complexity)
- SQLite episode store (graph stores provenance natively)
- Jina late chunking (we're not chunking — each conversation is a unit)

---

## What's Novel

1. **Semantic temporal context compilation.** Nobody converts raw timestamps into LLM-readable temporal narratives. Every existing system passes raw dates. We compile "22 months ago, during high school senior year, 7 state changes at ~0.3x/month." Testable claim: this improves temporal reasoning accuracy on existing benchmarks.

2. **World model as temporal state machine for knowledge work.** Kosmos proved world models work for one-shot research runs. We extend this to continuous, indefinite knowledge work. The world model accumulates and evolves over years, not hours.

3. **Procedural memory from state transition patterns.** Nobody extracts "how the user does things" from conversation history. ExpeL extracts procedures from single tasks. We extract them from *patterns across entity lifecycles.*

4. **Rolling context ingestion.** Processing each conversation with awareness of the full world model state built from all prior conversations. Not batch extraction in isolation — chronological ingestion where context accumulates.

5. **Tiered forgetting with graph-structural importance.** Importance isn't a heuristic — it's computed from graph connectivity, transition count, recency, and neighbor importance. PageRank for personal memory.

---

## Build Sequence

### Week 1: Foundation + Parser
- Project structure, dependencies, Docker (FalkorDB)
- ChatGPT export parser (linearize tree, extract turns, sort chronologically)
- Basic world model store (create/read entities and transitions in FalkorDB)
- Extraction prompt engineering and structured output schema

### Week 2: Ingestion Pipeline
- Full ingestion orchestrator (rolling context, entity resolution, state tracking)
- Run on all 4758 conversations
- Temporal context compiler (semantic time, not raw dates)
- Period detection and semantic anchoring

### Week 3: Query Layer + MCP
- MCP server with all query tools
- Hybrid retrieval (vector + graph traversal + temporal)
- World snapshot and diff operations
- Test with Claude Desktop / Cursor

### Week 4: Dreaming + Procedures
- Consolidation engine (merge, summarize, archive)
- Procedural memory extraction
- Dreaming engine (micro/deep/mega)
- Contradiction detection and tracking

### Week 5+: Polish + Release
- Benchmark on temporal reasoning (LoCoMo, BEAM temporal subset)
- Open source release with documentation
- Blog post / demo video
- Obsidian vault materialization (optional)
