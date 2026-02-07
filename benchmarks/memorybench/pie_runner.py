#!/usr/bin/env python3
"""
PIE Runner for MemoryBench

This script adapts PIE (Personal Intelligence Engine) to run on MemoryBench,
comparing its world-model approach to naive RAG baselines.
"""

import os
import sys
import json
import time
import copy
import logging
from pathlib import Path
from dataclasses import dataclass, field
from typing import List, Dict, Any, Literal, Optional
from concurrent.futures import ThreadPoolExecutor, as_completed
from tqdm import tqdm

# Add PIE to path
PIE_PATH = Path(__file__).parent.parent.parent / "pie"
sys.path.insert(0, str(PIE_PATH.parent))

from pie.core.world_model import WorldModel
from pie.core.models import (
    Entity, EntityType, StateTransition, TransitionType,
    Relationship, RelationshipType,
)

# Add MemoryBench repo to path
MEMORYBENCH_PATH = Path(__file__).parent / "repo"
sys.path.insert(0, str(MEMORYBENCH_PATH))

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("pie_runner")


# ============================================================================
# PIE Agent for MemoryBench
# ============================================================================

@dataclass
class PIEAgentConfig:
    """Configuration for PIE-based memory agent."""
    llm_provider: Literal["openai", "vllm"] = "openai"
    llm_config: dict = field(default_factory=lambda: {
        "model": "gpt-4o-mini",
        "temperature": 0.3,
    })
    memory_cache_dir: str = "./pie_memory"
    retrieve_k: int = 10
    extraction_model: str = "gpt-4o-mini"


class PIEAgent:
    """
    PIE-based memory agent for MemoryBench.
    
    Key differences from other baselines:
    1. Extracts typed entities from feedback dialogs
    2. Tracks state transitions (including contradictions)
    3. Uses temporal context for retrieval
    4. Distinguishes declarative vs procedural memory
    """
    
    def __init__(self, config: PIEAgentConfig):
        self.config = config
        self.world_model = WorldModel()
        self.feedback_log: List[Dict] = []  # Procedural memory
        
        # Initialize LLM
        try:
            from src.llms import LlmFactory
            self.llm = LlmFactory.create(
                provider_name=config.llm_provider,
                config=config.llm_config
            )
        except ImportError:
            # Fallback to direct OpenAI
            import openai
            self.llm = None
            self.openai_client = openai.OpenAI()
    
    def _llm_generate(self, messages: List[Dict]) -> str:
        """Generate response using LLM."""
        if self.llm:
            return self.llm.generate_response(messages=messages)
        else:
            response = self.openai_client.chat.completions.create(
                model=self.config.llm_config.get("model", "gpt-4o-mini"),
                messages=messages,
                temperature=self.config.llm_config.get("temperature", 0.3),
            )
            return response.choices[0].message.content
    
    def _extract_from_feedback(self, messages: List[Dict]) -> Dict:
        """
        Extract entities and state changes from a feedback dialog.
        
        This is the key innovation: treating feedback as evidence of
        system behavior patterns (procedural memory).
        """
        dialog_text = "\n".join([
            f"{m['role'].upper()}: {m['content']}" for m in messages
        ])
        
        extraction_prompt = f"""Analyze this conversation between a user and an AI system.
Extract:
1. Any factual information mentioned (entities, preferences, facts)
2. Any corrections or contradictions the user made
3. Whether the user was satisfied (positive/negative/neutral)
4. What type of task this was (QA, writing, summarization, etc.)

Conversation:
{dialog_text}

Respond in JSON:
{{
    "entities": [
        {{"name": "...", "type": "person|project|tool|concept|belief", "info": "..."}}
    ],
    "corrections": [
        {{"what": "...", "wrong": "...", "right": "..."}}
    ],
    "satisfaction": "positive|negative|neutral",
    "task_type": "...",
    "key_feedback": "..."
}}
"""
        
        try:
            response = self._llm_generate([
                {"role": "system", "content": "You are an expert at analyzing conversations to extract structured information."},
                {"role": "user", "content": extraction_prompt}
            ])
            
            # Parse JSON from response
            import re
            json_match = re.search(r'\{[\s\S]*\}', response)
            if json_match:
                return json.loads(json_match.group())
        except Exception as e:
            logger.warning(f"Extraction failed: {e}")
        
        return {
            "entities": [],
            "corrections": [],
            "satisfaction": "neutral",
            "task_type": "unknown",
            "key_feedback": ""
        }
    
    def add_conversation_to_memory(
        self, 
        messages: List[Dict[str, str]], 
        conversation_idx: int = 0,
    ):
        """
        Add a feedback conversation to memory.
        
        This processes the dialog to:
        1. Extract entities for declarative memory
        2. Track corrections/contradictions
        3. Record feedback patterns for procedural memory
        """
        timestamp = time.time() + conversation_idx  # Use idx for ordering
        
        # Extract structured information
        extraction = self._extract_from_feedback(messages)
        
        # Add entities to world model
        type_map = {
            "person": EntityType.PERSON,
            "project": EntityType.PROJECT,
            "tool": EntityType.TOOL,
            "concept": EntityType.CONCEPT,
            "belief": EntityType.BELIEF,
        }
        
        for entity_data in extraction.get("entities", []):
            entity_type = type_map.get(
                entity_data.get("type", "concept"),
                EntityType.CONCEPT
            )
            
            # Check if entity already exists
            existing = self.world_model.find_by_name(entity_data["name"])
            if existing:
                # Update state
                self.world_model.update_entity_state(
                    entity_id=existing.id,
                    new_state={"description": entity_data.get("info", "")},
                    source_conversation_id=str(conversation_idx),
                    timestamp=timestamp,
                    trigger_summary=f"Updated from feedback #{conversation_idx}",
                )
            else:
                # Create new entity
                self.world_model.create_entity(
                    name=entity_data["name"],
                    type=entity_type,
                    state={"description": entity_data.get("info", "")},
                    source_conversation_id=str(conversation_idx),
                    timestamp=timestamp,
                )
        
        # Record corrections as contradictions
        for correction in extraction.get("corrections", []):
            # Find related entity if exists
            entities = self.world_model.find_by_string_match(
                correction.get("what", ""),
                threshold=0.7
            )
            if entities:
                entity, score = entities[0]
                self.world_model.update_entity_state(
                    entity_id=entity.id,
                    new_state={
                        "corrected": correction.get("right", ""),
                        "was_wrong": correction.get("wrong", ""),
                    },
                    source_conversation_id=str(conversation_idx),
                    timestamp=timestamp,
                    trigger_summary=f"Correction: {correction.get('what', '')}",
                    is_contradiction=True,
                )
        
        # Record procedural memory (feedback pattern)
        self.feedback_log.append({
            "conversation_idx": conversation_idx,
            "task_type": extraction.get("task_type", "unknown"),
            "satisfaction": extraction.get("satisfaction", "neutral"),
            "key_feedback": extraction.get("key_feedback", ""),
            "messages": messages,
            "timestamp": timestamp,
        })
    
    def retrieve_memory(self, query: str, k: int = 10) -> List[str]:
        """
        Retrieve relevant memory for a query.
        
        Combines:
        1. Entity matching (declarative)
        2. Recent transitions (temporal)
        3. Similar feedback patterns (procedural)
        """
        results = []
        
        # 1. Find matching entities
        entity_matches = self.world_model.find_by_string_match(query, threshold=0.6)
        for entity, score in entity_matches[:k//2]:
            state_summary = json.dumps(entity.current_state, ensure_ascii=False)
            results.append(f"[{entity.type.value}] {entity.name}: {state_summary}")
            
            # Get transitions for context
            transitions = self.world_model.get_transitions(entity.id)
            for t in transitions[-3:]:  # Last 3 transitions
                results.append(f"  - {t.trigger_summary}")
        
        # 2. Get recent activity context
        context_preamble = self.world_model.build_context_preamble(time.time())
        if context_preamble:
            results.append(f"RECENT CONTEXT:\n{context_preamble}")
        
        # 3. Find similar feedback patterns (procedural memory)
        query_lower = query.lower()
        relevant_feedback = []
        for fb in self.feedback_log:
            # Simple keyword matching for now
            if any(word in query_lower for word in fb.get("key_feedback", "").lower().split()):
                relevant_feedback.append(fb)
        
        if relevant_feedback:
            results.append("SIMILAR PAST EXPERIENCES:")
            for fb in relevant_feedback[-3:]:
                results.append(f"  - Task: {fb['task_type']}, Feedback: {fb['satisfaction']}")
                if fb.get("key_feedback"):
                    results.append(f"    Note: {fb['key_feedback']}")
        
        return results[:k]
    
    def generate_response(
        self,
        messages: List[Dict[str, str]],
        lang: Literal["en", "zh"] = "en",
        retrieve_k: int = None,
    ) -> str:
        """Generate response using memory context."""
        if retrieve_k is None:
            retrieve_k = self.config.retrieve_k
        
        question = messages[-1]['content']
        memory_context = self.retrieve_memory(question, k=retrieve_k)
        context = "\n".join(memory_context)
        
        if lang == "en":
            user_prompt = f"""Memory Context:
{context}

User Query:
{question}

Based on the memory context provided (which includes past interactions, learned facts, and feedback patterns), respond naturally and appropriately to the user's query above."""
        else:  # zh
            user_prompt = f"""记忆上下文：
{context}

用户查询：
{question}

请根据提供的记忆上下文（包括过去的交互、学习到的事实和反馈模式）准确、自然地回答用户的查询。"""
        
        messages_with_context = copy.deepcopy(messages)
        messages_with_context[-1]["content"] = user_prompt
        
        return self._llm_generate(messages_with_context)
    
    def save_memories(self):
        """Save memory to disk."""
        if self.config.memory_cache_dir:
            os.makedirs(self.config.memory_cache_dir, exist_ok=True)
            
            # Save world model
            self.world_model.persist_path = Path(self.config.memory_cache_dir) / "world_model.json"
            self.world_model.save()
            
            # Save feedback log (procedural memory)
            with open(Path(self.config.memory_cache_dir) / "feedback_log.json", "w") as f:
                json.dump(self.feedback_log, f, indent=2, ensure_ascii=False)
            
            logger.info(f"Saved PIE memory: {self.world_model.stats}")
    
    def load_memories(self):
        """Load memory from disk."""
        if self.config.memory_cache_dir:
            wm_path = Path(self.config.memory_cache_dir) / "world_model.json"
            fb_path = Path(self.config.memory_cache_dir) / "feedback_log.json"
            
            if wm_path.exists():
                self.world_model = WorldModel(persist_path=wm_path)
                logger.info(f"Loaded world model: {self.world_model.stats}")
            
            if fb_path.exists():
                with open(fb_path) as f:
                    self.feedback_log = json.load(f)
                logger.info(f"Loaded {len(self.feedback_log)} feedback entries")


# ============================================================================
# PIE Solver (MemoryBench interface)
# ============================================================================

class PIESolver:
    """PIE-based solver compatible with MemoryBench framework."""
    
    AGENT_CLASS = PIEAgent
    MAX_THREADS = 2  # Lower due to LLM extraction calls
    
    def __init__(self, config: PIEAgentConfig, memory_cache_dir: str):
        config.memory_cache_dir = memory_cache_dir
        self.config = config
        self.memory_cache_dir = memory_cache_dir
        self.method_name = "pie"
        self.agent = self.AGENT_CLASS(config)
    
    def create_or_load_memory(self, dialogs: List[Dict]):
        """Create or load memory from dialogs."""
        cache_marker = Path(self.memory_cache_dir) / ".cached"
        
        if not cache_marker.exists():
            logger.info(f"Creating PIE memory at {self.memory_cache_dir}")
            
            for dialog in tqdm(dialogs, desc="Processing dialogs with PIE"):
                try:
                    self.agent.add_conversation_to_memory(
                        dialog["dialog"],
                        dialog["test_idx"]
                    )
                except Exception as e:
                    logger.warning(f"Failed to process dialog {dialog['test_idx']}: {e}")
            
            self.agent.save_memories()
            cache_marker.touch()
        else:
            logger.info(f"Loading PIE memory from {self.memory_cache_dir}")
            self.agent.load_memories()
    
    def predict_single_data(self, dataset, data) -> Dict:
        """Predict response for a single data point."""
        input_messages = dataset.get_initial_chat_messages(data["test_idx"])
        
        try:
            messages = copy.deepcopy(input_messages)
            response = self.agent.generate_response(
                messages=messages,
                lang=data["lang"],
                retrieve_k=self.config.retrieve_k,
            )
        except Exception as e:
            logger.error(f"Failed to generate response for {data['test_idx']}: {e}")
            response = f"Error: {str(e)}"
        
        return {
            "test_idx": data["test_idx"],
            "messages": messages,
            "response": response,
        }
    
    def predict_test(self, dataset, split_name: str = "test") -> List[Dict]:
        """Predict on test set."""
        results = []
        
        with ThreadPoolExecutor(max_workers=self.MAX_THREADS) as executor:
            futures = [
                executor.submit(self.predict_single_data, dataset, data)
                for data in dataset.dataset[split_name].to_list()
            ]
            for future in tqdm(as_completed(futures), total=len(futures), desc="PIE prediction"):
                results.append(future.result())
        
        return sorted(results, key=lambda x: x["test_idx"])


# ============================================================================
# Main Runner
# ============================================================================

def run_pie_benchmark(
    dataset_type: str = "single",
    set_name: str = "NFCats",
    output_dir: str = "pie_results",
    sample_size: int = 50,
):
    """Run PIE on MemoryBench."""
    from memorybench import load_memory_bench, evaluate, summary_results
    
    # Load dataset
    logger.info(f"Loading {dataset_type}/{set_name}...")
    if dataset_type == "single":
        datasets = [load_memory_bench(dataset_type, set_name)]
    else:
        datasets = load_memory_bench(dataset_type, set_name)
    
    # Collect training dialogs
    total_dialogs = []
    for dataset in datasets:
        for data in dataset.dataset["train"].to_list()[:sample_size]:
            total_dialogs.append({
                "test_idx": data["test_idx"],
                "dialog": data.get("dialog", []),
                "dataset": dataset.dataset_name,
            })
    
    logger.info(f"Collected {len(total_dialogs)} training dialogs")
    
    # Create PIE solver
    config = PIEAgentConfig()
    memory_cache_dir = f"{output_dir}/memory_cache/{dataset_type}/{set_name}/pie"
    solver = PIESolver(config, memory_cache_dir)
    
    # Build memory
    solver.create_or_load_memory(total_dialogs)
    
    # Run predictions
    all_predictions = []
    for dataset in datasets:
        logger.info(f"Predicting on {dataset.dataset_name}...")
        predictions = solver.predict_test(dataset)
        for pred in predictions:
            pred["dataset"] = dataset.dataset_name
            all_predictions.append(pred)
    
    # Save predictions
    os.makedirs(f"{output_dir}/{dataset_type}/{set_name}", exist_ok=True)
    with open(f"{output_dir}/{dataset_type}/{set_name}/predictions.json", "w") as f:
        json.dump(all_predictions, f, indent=2, ensure_ascii=False)
    
    # Evaluate
    logger.info("Evaluating...")
    eval_results = evaluate(dataset_type, set_name, all_predictions)
    
    with open(f"{output_dir}/{dataset_type}/{set_name}/eval_results.json", "w") as f:
        json.dump(eval_results, f, indent=2, ensure_ascii=False)
    
    # Summary
    summary = summary_results(dataset_type, set_name, all_predictions, eval_results)
    
    with open(f"{output_dir}/{dataset_type}/{set_name}/summary.json", "w") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)
    
    logger.info(f"Results saved to {output_dir}/{dataset_type}/{set_name}/")
    logger.info(f"Summary: {json.dumps(summary.get('summary', {}), indent=2)}")
    
    return summary


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Run PIE on MemoryBench")
    parser.add_argument("--dataset_type", type=str, default="single",
                       choices=["single", "domain", "task"])
    parser.add_argument("--set_name", type=str, default="NFCats")
    parser.add_argument("--output_dir", type=str, default="pie_results")
    parser.add_argument("--sample_size", type=int, default=50,
                       help="Number of training samples to use")
    
    args = parser.parse_args()
    
    run_pie_benchmark(
        dataset_type=args.dataset_type,
        set_name=args.set_name,
        output_dir=args.output_dir,
        sample_size=args.sample_size,
    )
