# PIE Integration Guide

## Using PIE with OpenClaw/Dev

PIE can be integrated with OpenClaw to give the agent (me) persistent memory of your conversations and projects.

### Option 1: MCP Server (Recommended)

PIE exposes an MCP server that any LLM client can connect to.

```bash
# Start PIE MCP server
cd ~/personal-intelligence-system
python -m pie.mcp.server

# In OpenClaw config (config.yaml):
mcp_servers:
  - name: pie
    command: ["python", "-m", "pie.mcp.server"]
    cwd: "~/personal-intelligence-system"
```

### Option 2: Direct Integration

For tighter integration, PIE can be called directly from OpenClaw:

```python
# In an OpenClaw skill or hook:
from pie.core.world_model import WorldModel
from pie.retrieval.temporal_retriever import TemporalRetriever
from pie.retrieval.context_compiler import compile_subgraph

# Load world model
wm = WorldModel("~/personal-intelligence-system/output/world_model.json")

# Query
retriever = TemporalRetriever(wm, config.retrieval, config.temporal)
results = retriever.retrieve("What was my opinion on React last month?")

# Compile to markdown for LLM
context = compile_subgraph(results, ...)
# Include in LLM context
```

### Option 3: Memory Search Integration

Add PIE as a memory source in OpenClaw's `memory_search`:

```yaml
# In config.yaml
memory_search:
  extra_paths:
    - ~/personal-intelligence-system/output/world_model.json
```

---

## Entity Granularity by Use Case

### Personal Assistant (OpenClaw/Dev)
**Goal:** Remember user's projects, decisions, opinions over time

**Entity types to use:**
- `project`: Things user is building
- `decision`: Choices made with reasoning
- `belief`: Opinions that evolve
- `person`: People user mentions
- `period`: Life phases for temporal anchoring
- `event`: Specific dated occurrences

**Granularity:** Coarse. One entity per project, not per task.

### Sales Intelligence
**Goal:** Track prospects, deals, competitive intelligence

**Entity types to use:**
- `organization`: Companies (prospects, competitors)
- `person`: Contacts, stakeholders
- `decision`: Deal stages, objections
- `event`: Meetings, demos, follow-ups

**Granularity:** Fine. Separate entities for each stakeholder, each objection.

### Knowledge Base
**Goal:** Store domain knowledge with evolution

**Entity types to use:**
- `concept`: Core ideas
- `tool`: Technologies mentioned
- `belief`: Expert opinions (can evolve as field evolves)

**Granularity:** Medium. One entity per major concept.

---

## Live Web Grounding

PIE uses web search to verify and enrich entities.

### Setup with Brave API

```python
from pie.grounding.web_enrichment import WebEnricher

def brave_search(query: str) -> list[dict]:
    # Use OpenClaw's web_search tool
    from openclaw.tools import web_search
    return web_search(query)

def llm_extract(prompt: str) -> str:
    # Use OpenClaw's LLM
    from openclaw.tools import llm_chat
    return llm_chat(prompt)

enricher = WebEnricher(brave_search, llm_extract)

# Enrich an entity
result = enricher.enrich(
    name="FalkorDB",
    entity_type="tool",
    context="graph database Redis"
)
print(result.canonical_name)  # "FalkorDB"
print(result.description)      # "Real-time graph database built on Redis..."
```

### Sales Enrichment

```python
from pie.grounding.web_enrichment import SalesEnricher

enricher = SalesEnricher(brave_search, llm_extract)

# Enrich a prospect
prospect = enricher.enrich_prospect("Stripe")
print(prospect.industry)       # "fintech"
print(prospect.tech_stack)     # ["Ruby", "Go", "React", ...]
print(prospect.key_contacts)   # [{"name": "Patrick Collison", "title": "CEO"}, ...]
```

---

## Eliminating Hardcoded Thresholds

### The Problem

Traditional approach:
```python
if similarity > 0.85:
    match = True  # Why 0.85? Arbitrary.
```

### The Solution: LLM-Native Decisions

Instead of thresholds, ask the LLM:

```python
from pie.resolution.llm_resolver import LLMEntityResolver

resolver = LLMEntityResolver(world_model, llm_client)

# No threshold! LLM decides.
result = resolver.resolve(extracted_entity)
print(result.matched_entity)  # The match, or None
print(result.confidence)      # LLM's stated confidence
print(result.reasoning)       # "These are the same project, just renamed"
```

### Cost Tradeoff

| Approach | Cost per resolution | Accuracy | Interpretability |
|----------|---------------------|----------|------------------|
| Hardcoded thresholds | ~$0.0001 (embedding) | Varies | Low |
| LLM-native | ~$0.001 (API call) | Higher | High |

For a knowledge graph with ~1000 entities, processing 200 conversations:
- Hardcoded: ~200 × $0.0001 = $0.02
- LLM-native: ~200 × 5 candidates × $0.001 = $1.00

**Recommendation:** Use LLM-native for entity resolution (decisions matter).
Use embeddings for retrieval (just finding candidates).

### When to Use Hardcoded Thresholds

Still useful for:
1. **Candidate retrieval:** Embedding similarity to narrow candidates before LLM
2. **Confidence gating:** "If LLM confidence < 0.7, don't auto-merge"
3. **Rate limiting:** "Max 5 LLM calls per entity resolution"

---

## Retrieval Weight Learning

Instead of hardcoded weights (0.6 semantic, 0.25 temporal, 0.15 recency):

### Option 1: LLM Reranking

```python
def llm_rerank(query: str, candidates: list[RetrievalResult]) -> list[RetrievalResult]:
    """Let LLM rerank candidates instead of using fixed weights."""
    
    prompt = f"""Given this query: "{query}"
    
Rank these entities by relevance (1 = most relevant):

{format_candidates(candidates)}

Return JSON: [{{"entity": "name", "rank": 1}}, ...]"""
    
    response = llm(prompt)
    rankings = parse_rankings(response)
    
    return sorted(candidates, key=lambda c: rankings.get(c.entity_name, 999))
```

### Option 2: Learn Weights from Feedback

```python
class LearnedRetriever:
    def __init__(self):
        self.weights = {"semantic": 0.33, "temporal": 0.33, "recency": 0.33}
        self.feedback_log = []
    
    def log_feedback(self, query: str, entity: str, relevant: bool):
        """Log user feedback for weight learning."""
        self.feedback_log.append({
            "query": query,
            "entity": entity,
            "relevant": relevant,
            "features": self._get_features(query, entity),
        })
    
    def update_weights(self):
        """Learn weights from feedback via logistic regression."""
        from sklearn.linear_model import LogisticRegression
        
        X = [f["features"] for f in self.feedback_log]
        y = [f["relevant"] for f in self.feedback_log]
        
        model = LogisticRegression().fit(X, y)
        
        # New weights from learned coefficients
        self.weights = {
            "semantic": model.coef_[0],
            "temporal": model.coef_[1],
            "recency": model.coef_[2],
        }
```

---

## Quick Start: Using PIE Today

1. **Load existing world model:**
```python
from pie.core.world_model import WorldModel
wm = WorldModel("~/personal-intelligence-system/output/world_model.json")
print(f"Loaded {len(wm.entities)} entities")
```

2. **Query it:**
```python
# Find entities by name
entity = wm.find_by_name("Science Research Academy")
print(entity.current_state)

# Get timeline
transitions = wm.get_transitions(entity.id)
for t in transitions:
    print(f"{t.timestamp}: {t.trigger_summary}")
```

3. **Add new info:**
```python
# From a new conversation
from pie.core.llm import LLMClient
from pie.ingestion.pipeline import IngestionPipeline

pipeline = IngestionPipeline(config)
pipeline.ingest_single(conversation)
```

4. **Compile context for LLM:**
```python
from pie.retrieval.context_compiler import compile_entity_context
from datetime import datetime

context = compile_entity_context(entity, transitions, relationships, datetime.now())
print(context)
# Include in your LLM prompt
```
