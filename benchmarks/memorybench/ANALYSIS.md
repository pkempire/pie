# MemoryBench Analysis

## Overview

**Paper**: MemoryBench: A Benchmark for Memory and Continual Learning in LLM Systems  
**arXiv**: 2510.17281  
**GitHub**: https://github.com/LittleDinoC/MemoryBench  
**Dataset**: https://huggingface.co/datasets/THUIR/MemoryBench

## Key Concepts

### Memory Taxonomy

MemoryBench uniquely tests two types of memory:

1. **Declarative Memory** - Factual knowledge independent of tasks
   - **Semantic Memory**: User-independent facts (Wikipedia, textbooks)
   - **Episodic Memory**: User-dependent info (conversation history, personal profiles)

2. **Procedural Memory** - Non-factual knowledge about task execution
   - User feedback logs containing info about past system performance
   - Workflow patterns, rewards, lessons learned

### Feedback Types

1. **Explicit Feedback**
   - **Verbose**: Natural language critiques and follow-up conversations
   - **Action**: Like/dislike button clicks

2. **Implicit Feedback**
   - Copy button clicks, session terminations, prompt refinements
   - Requires analysis to extract reliable signals

### Task Formats

| Format | Input | Output | Examples |
|--------|-------|--------|----------|
| LiSo | Long | Short | Locomo, DialSim, IdeaBench |
| LiLo | Long | Long | HelloBench, WritingBench, JuDGE |
| SiLo | Short | Long | WritingPrompts, HelloBench-QA |
| SiSo | Short | Short | NFCats, LexEval-QA, SciTechNews |

### Domains

- **Open-Domain**: Locomo, DialSim, WritingPrompts, HelloBench, NFCats
- **Legal**: JuDGE, LexEval variants, WritingBench-Politics&Law
- **Academic**: IdeaBench, LimitGen, HelloBench-Academic, WritingBench-Academic

## Baseline Results (from paper)

### Key Finding: SOTA memory systems don't generalize

| Method | Open-Domain | Legal | Academic |
|--------|-------------|-------|----------|
| Vanilla (no memory) | 0.45 | 0.38 | 0.42 |
| BM25-Message | 0.52 | 0.48 | 0.51 |
| BM25-Dialog | 0.51 | 0.47 | 0.50 |
| Embed-Message | 0.54 | 0.49 | 0.53 |
| Embed-Dialog | 0.53 | 0.48 | 0.52 |
| A-Mem | 0.50 | 0.45 | 0.49 |
| Mem0 | - | 0.44 | 0.46 |
| MemoryOS | 0.48 | 0.43 | 0.47 |

*Note: Min-max normalized scores. Mem0 fails on Open-Domain due to long contexts.*

### Critical Insights

1. **Naive RAG beats sophisticated memory systems** on most tasks
2. **Existing systems treat all memory as declarative** - they can't distinguish procedural memory
3. **Efficiency is poor** - MemoryOS takes 17+ seconds per case for memory construction
4. **Performance fluctuates during training** - especially in vertical domains

## PIE Analysis

### How PIE Handles This

PIE's world model approach is fundamentally different from existing baselines:

| Feature | BM25/RAG | A-Mem/Mem0 | PIE |
|---------|----------|------------|-----|
| Memory Structure | Flat documents | Graph/hierarchy | Typed entity graph |
| State Tracking | None | Implicit | Explicit transitions |
| Contradiction Handling | None | Rewrite | StateTransition(CONTRADICTION) |
| Temporal Awareness | None | Limited | first_seen/last_seen + history |
| Entity Resolution | BM25/embedding | Embedding | String + embedding + context |

### PIE's Strengths for MemoryBench

1. **State Transitions** - PIE explicitly models state changes, which is exactly what procedural memory captures (feedback about past performance = state change observations)

2. **Contradiction Detection** - When user feedback says "that was wrong", PIE can record this as a `TransitionType.CONTRADICTION` rather than silently updating

3. **Typed Entities** - PIE distinguishes person, project, tool, belief, decision, concept, period, event - this helps separate semantic vs episodic memory

4. **Temporal Context** - `build_context_preamble()` provides activity-based context, which naturally handles the "what happened recently" aspect of procedural memory

### PIE's Weaknesses for MemoryBench

1. **No Procedural Learning** - PIE records observations but doesn't learn policies from them. The `Procedure` class exists but isn't implemented.

2. **Extraction-Only** - PIE is designed for batch extraction from conversations, not interactive feedback integration

3. **No Retrieval** - PIE doesn't have a RAG-style retrieval mechanism; it relies on context preamble

4. **Scale** - MemoryBench has ~20k cases; PIE hasn't been tested at this scale

## Implementation Plan

### Phase 1: Adapt PIE as MemoryBench Baseline

```python
class PIEAgent(BaseAgent):
    """PIE-based memory system for MemoryBench."""
    
    def __init__(self, config):
        self.world_model = WorldModel()
        self.llm = LlmFactory.create(...)
    
    def add_conversation_to_memory(self, messages, conversation_idx):
        # Extract entities and state changes from feedback dialog
        extraction = self.extract(messages)
        
        for entity in extraction.entities:
            self.world_model.create_entity(...)
        
        for change in extraction.state_changes:
            # This is the key: treat feedback as state transitions
            self.world_model.update_entity_state(
                is_contradiction=change.is_contradiction
            )
    
    def generate_response(self, messages, retrieve_k=10):
        # Build context from world model
        context = self.world_model.build_context_preamble(time.time())
        
        # Find relevant entities
        query = messages[-1]['content']
        matches = self.world_model.find_by_string_match(query)
        
        # Inject context into prompt
        prompt = f"""
Context from memory:
{context}

Relevant entities:
{self._format_entities(matches)}

User query:
{query}
"""
        return self.llm.generate_response(prompt)
```

### Phase 2: Enhance PIE for Procedural Learning

The key insight: **User feedback = Evidence of system behavior patterns**

```python
@dataclass
class TaskFeedback:
    """Feedback on a specific task execution."""
    task_type: str
    task_query: str
    response: str
    feedback_type: str  # verbose, action, implicit
    feedback_content: str
    is_positive: bool
    timestamp: float

class ProceduralMemory:
    """Track patterns in task execution and feedback."""
    
    def __init__(self):
        self.task_patterns: dict[str, list[TaskFeedback]] = {}
    
    def record_feedback(self, feedback: TaskFeedback):
        """Record feedback for pattern learning."""
        key = self._pattern_key(feedback.task_type, feedback.task_query)
        self.task_patterns[key].append(feedback)
    
    def get_similar_experiences(self, query: str, k: int = 5):
        """Find past experiences similar to current query."""
        # Return both successes and failures
        pass
```

### Phase 3: Continual Learning Loop

```python
def continual_learning_step(
    world_model: WorldModel,
    procedural: ProceduralMemory,
    task_batch: list[dict],
):
    """One step of continual learning."""
    
    for task in task_batch:
        # 1. Build context from both memory types
        declarative_context = world_model.build_context_preamble(task['timestamp'])
        procedural_context = procedural.get_similar_experiences(task['query'])
        
        # 2. Generate response
        response = generate_with_context(
            task['query'],
            declarative_context,
            procedural_context,
        )
        
        # 3. Get feedback
        feedback = simulate_user_feedback(task, response)
        
        # 4. Update memories
        if feedback.is_verbose:
            # Extract entities/state changes from feedback
            extraction = extract_from_feedback(feedback)
            world_model.update_from_extraction(extraction)
        
        # 5. Record procedural memory
        procedural.record_feedback(TaskFeedback(
            task_type=task['type'],
            task_query=task['query'],
            response=response,
            feedback_type=feedback.type,
            feedback_content=feedback.content,
            is_positive=feedback.is_positive,
            timestamp=task['timestamp'],
        ))
```

## Hypotheses to Test

### H1: PIE's state tracking improves contradiction handling
- **Test**: Run on tasks with conflicting information in feedback
- **Metric**: Accuracy on questions about corrected facts

### H2: Typed entities improve domain-specific tasks
- **Test**: Legal domain (entities = laws, cases, parties)
- **Metric**: Performance on LexEval, JuDGE

### H3: Temporal context helps on long-context tasks
- **Test**: LiSo and LiLo tasks with temporal dependencies
- **Metric**: Compare to BM25 on DialSim, Locomo

### H4: Explicit procedural memory improves from feedback
- **Test**: Track success/failure patterns across task types
- **Metric**: Performance improvement over time in on-policy setting

## Experiments to Run

### Experiment 1: Baseline Reproduction
- [ ] Set up MemoryBench environment
- [ ] Run BM25-Message baseline
- [ ] Verify results match paper

### Experiment 2: PIE Adapter
- [ ] Implement PIEAgent class
- [ ] Run on small subset (Open-Domain, LiSo)
- [ ] Compare to BM25

### Experiment 3: Procedural Enhancement
- [ ] Implement ProceduralMemory class
- [ ] Add feedback-to-procedure extraction
- [ ] Run on-policy experiments

### Experiment 4: Full Benchmark
- [ ] Run on all domains/tasks
- [ ] Collect timing statistics
- [ ] Generate comparison figures

## Notes

### Why This Matters for PIE

MemoryBench reveals a critical gap: **nobody knows how to learn from user feedback**.

PIE's current architecture is read-only - it extracts knowledge but doesn't adapt based on performance feedback. The benchmark shows this is a universal problem.

**The opportunity**: PIE's explicit state tracking could be the foundation for better procedural learning:
1. Record predictions + feedback as state transitions
2. Detect patterns in successes/failures
3. Build procedural knowledge from patterns
4. Use procedural knowledge to guide future responses

This is exactly what humans do - we remember not just facts, but also our past attempts and what worked.

---

*Analysis by PIE benchmark runner - Last updated: 2026-02-05*
