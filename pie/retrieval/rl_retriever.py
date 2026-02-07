"""
RL-trained Graph Retriever — GRPO for subgraph selection.

Implements Group Relative Policy Optimization for learning
which subgraphs to retrieve for temporal queries.

Architecture:
  - Policy: LLM generates retrieval actions (which entities to fetch)
  - Reward: downstream answer quality + temporal coherence + cost penalty
  - Training: GRPO with process supervision

Based on GraphRAG-R1 and R3-RAG approaches.
"""

from __future__ import annotations
import json
import random
import logging
from dataclasses import dataclass, field
from typing import Callable
from pathlib import Path

logger = logging.getLogger("pie.retrieval.rl")


# ── Reward Functions ──────────────────────────────────────────────────────────

@dataclass
class RetrievalReward:
    """Composite reward for retrieval actions."""
    answer_correct: float = 0.0      # 0 or 1
    retrieval_precision: float = 0.0  # fraction of retrieved entities that were useful
    temporal_coherence: float = 0.0   # how well entities match query's time scope
    cost_penalty: float = 0.0         # penalty for # of retrievals
    
    @property
    def total(self) -> float:
        # Weights can be tuned
        return (
            1.0 * self.answer_correct +
            0.3 * self.retrieval_precision +
            0.2 * self.temporal_coherence -
            0.1 * self.cost_penalty
        )


def compute_temporal_coherence(
    retrieved_entity_ids: list[str],
    query_time_range: tuple[float, float] | None,
    world_model,
) -> float:
    """
    How well do retrieved entities match the query's temporal scope?
    
    Returns 0-1 score:
    - 1.0 if all entities have transitions in the query time range
    - 0.0 if no entities have relevant transitions
    """
    if not query_time_range or not retrieved_entity_ids:
        return 0.5  # neutral if no temporal constraint
    
    start, end = query_time_range
    in_range_count = 0
    
    for eid in retrieved_entity_ids:
        entity = world_model.get_entity(eid)
        if not entity:
            continue
        
        # Check if entity has any transition in range
        transitions = world_model.get_transitions(eid)
        for t in transitions:
            if start <= t.timestamp <= end:
                in_range_count += 1
                break
    
    return in_range_count / len(retrieved_entity_ids) if retrieved_entity_ids else 0.0


def compute_retrieval_precision(
    retrieved_entity_ids: list[str],
    gold_entity_ids: set[str],
) -> float:
    """Fraction of retrieved entities that are in the gold set."""
    if not retrieved_entity_ids:
        return 0.0
    hits = sum(1 for eid in retrieved_entity_ids if eid in gold_entity_ids)
    return hits / len(retrieved_entity_ids)


# ── GRPO Implementation ───────────────────────────────────────────────────────

@dataclass
class GRPOConfig:
    """Configuration for Group Relative Policy Optimization."""
    group_size: int = 4          # number of rollouts per query
    temperature: float = 0.7      # sampling temperature
    baseline: str = "mean"        # "mean" or "min" for advantage computation
    lr: float = 1e-5
    kl_coef: float = 0.1          # KL penalty coefficient
    max_retrieval_steps: int = 5  # max entities to retrieve per query


@dataclass
class RetrievalRollout:
    """One retrieval trajectory."""
    query: str
    actions: list[str]  # entity IDs retrieved, in order
    reward: RetrievalReward
    log_probs: list[float]  # log probability of each action


def grpo_advantage(rollouts: list[RetrievalRollout], baseline: str = "mean") -> list[float]:
    """
    Compute advantages for GRPO.
    
    In GRPO, we compare rollouts within a group rather than using
    a learned value function. This is simpler and works well for
    discrete action spaces.
    """
    rewards = [r.reward.total for r in rollouts]
    
    if baseline == "mean":
        baseline_value = sum(rewards) / len(rewards)
    elif baseline == "min":
        baseline_value = min(rewards)
    else:
        baseline_value = 0.0
    
    return [r - baseline_value for r in rewards]


class RLRetriever:
    """
    Retriever that can be trained with RL.
    
    The policy is parameterized by an LLM that outputs
    retrieval actions (entity IDs or "STOP").
    """
    
    def __init__(
        self,
        world_model,
        llm_client,
        config: GRPOConfig = None,
    ):
        self.world_model = world_model
        self.llm = llm_client
        self.config = config or GRPOConfig()
        
        # For training, we'd fine-tune the LLM
        # For now, we use prompting and can collect training data
        self.training_data: list[tuple[str, list[RetrievalRollout]]] = []
    
    def _get_retrieval_prompt(
        self,
        query: str,
        retrieved_so_far: list[str],
        available_entities: list[tuple[str, str, float]],  # (id, name, score)
    ) -> str:
        """Build prompt for retrieval action."""
        
        retrieved_names = []
        for eid in retrieved_so_far:
            entity = self.world_model.get_entity(eid)
            if entity:
                retrieved_names.append(entity.name)
        
        # Top candidates not yet retrieved
        candidates = [
            (eid, name, score) 
            for eid, name, score in available_entities 
            if eid not in retrieved_so_far
        ][:10]
        
        prompt = f"""You are retrieving entities from a knowledge graph to answer this query:

Query: {query}

Already retrieved: {retrieved_names if retrieved_names else "(none yet)"}

Available entities to add (with relevance scores):
{chr(10).join(f"  {i+1}. {name} (score: {score:.2f})" for i, (eid, name, score) in enumerate(candidates))}

Choose the NEXT entity to retrieve, or respond "STOP" if you have enough.
Respond with just the number (1-{len(candidates)}) or "STOP"."""
        
        return prompt
    
    def retrieve_with_policy(
        self,
        query: str,
        temperature: float = None,
    ) -> tuple[list[str], list[float]]:
        """
        Run the retrieval policy to get a subgraph.
        
        Returns: (entity_ids, log_probs) for the trajectory
        """
        from pie.retrieval.graph_retriever import retrieve_subgraph, parse_query_intent
        
        temperature = temperature or self.config.temperature
        
        # Get initial candidates via embedding search
        intent = parse_query_intent(query, self.llm)
        
        # Get all entities scored by embedding similarity
        query_emb = self.llm.embed_single(query)
        scored_entities = self.world_model.find_by_embedding(
            query_emb, 
            top_k=50,
        )
        available = [(e.id, e.name, s) for e, s in scored_entities]
        
        retrieved = []
        log_probs = []
        
        for step in range(self.config.max_retrieval_steps):
            prompt = self._get_retrieval_prompt(query, retrieved, available)
            
            # Get LLM action
            # In real training, we'd use log_prob from the model
            result = self.llm.chat(
                messages=[{"role": "user", "content": prompt}],
                model="gpt-4o-mini",
                max_tokens=10,
                temperature=temperature,
            )
            action = result["content"].strip()
            
            if action.upper() == "STOP":
                log_probs.append(0.0)  # placeholder
                break
            
            try:
                idx = int(action) - 1
                candidates = [
                    (eid, name, score) 
                    for eid, name, score in available 
                    if eid not in retrieved
                ][:10]
                
                if 0 <= idx < len(candidates):
                    eid = candidates[idx][0]
                    retrieved.append(eid)
                    log_probs.append(0.0)  # placeholder — real impl would get from model
            except:
                continue
        
        return retrieved, log_probs
    
    def collect_rollouts(
        self,
        query: str,
        answer_fn: Callable[[str, list[str]], tuple[bool, set[str]]],
        query_time_range: tuple[float, float] | None = None,
    ) -> list[RetrievalRollout]:
        """
        Collect multiple rollouts for a query using the current policy.
        
        answer_fn: Given (query, retrieved_entity_ids), returns:
          - (answer_correct, gold_entity_ids)
        """
        rollouts = []
        
        for _ in range(self.config.group_size):
            entity_ids, log_probs = self.retrieve_with_policy(
                query, 
                temperature=self.config.temperature,
            )
            
            # Compute reward
            correct, gold_entities = answer_fn(query, entity_ids)
            
            reward = RetrievalReward(
                answer_correct=1.0 if correct else 0.0,
                retrieval_precision=compute_retrieval_precision(entity_ids, gold_entities),
                temporal_coherence=compute_temporal_coherence(
                    entity_ids, query_time_range, self.world_model
                ),
                cost_penalty=len(entity_ids) / self.config.max_retrieval_steps,
            )
            
            rollouts.append(RetrievalRollout(
                query=query,
                actions=entity_ids,
                reward=reward,
                log_probs=log_probs,
            ))
        
        return rollouts
    
    def compute_grpo_loss(
        self,
        rollouts: list[RetrievalRollout],
    ) -> dict:
        """
        Compute GRPO loss for a batch of rollouts.
        
        In practice, this would update model parameters.
        For now, we just compute what the loss would be.
        """
        advantages = grpo_advantage(rollouts, self.config.baseline)
        
        # Policy gradient loss: -advantage * log_prob
        pg_losses = []
        for rollout, adv in zip(rollouts, advantages):
            for log_prob in rollout.log_probs:
                pg_losses.append(-adv * log_prob)
        
        return {
            "pg_loss": sum(pg_losses) / len(pg_losses) if pg_losses else 0.0,
            "mean_reward": sum(r.reward.total for r in rollouts) / len(rollouts),
            "mean_advantage": sum(advantages) / len(advantages),
        }
    
    def save_training_data(self, path: Path):
        """Save collected training data for offline training."""
        data = []
        for query, rollouts in self.training_data:
            for r in rollouts:
                data.append({
                    "query": r.query,
                    "actions": r.actions,
                    "reward": {
                        "answer_correct": r.reward.answer_correct,
                        "retrieval_precision": r.reward.retrieval_precision,
                        "temporal_coherence": r.reward.temporal_coherence,
                        "cost_penalty": r.reward.cost_penalty,
                        "total": r.reward.total,
                    },
                })
        
        with open(path, "w") as f:
            for item in data:
                f.write(json.dumps(item) + "\n")
        
        logger.info(f"Saved {len(data)} training examples to {path}")


# ── Process Reward Model ──────────────────────────────────────────────────────

class ProcessRewardModel:
    """
    Learned reward model for retrieval steps.
    
    Predicts whether a given retrieval action will lead
    to a correct answer. Used for dense reward shaping.
    
    Based on R3-RAG's process supervision approach.
    """
    
    def __init__(self, llm_client):
        self.llm = llm_client
        # In practice, this would be a trained model
        # For now, we use LLM-as-judge
    
    def score_retrieval_step(
        self,
        query: str,
        entity_name: str,
        entity_state: dict,
        retrieved_so_far: list[str],
    ) -> float:
        """
        Score a single retrieval step.
        
        Returns 0-1 probability that this retrieval will help
        answer the query correctly.
        """
        prompt = f"""Rate how relevant this entity is for answering the query.

Query: {query}

Entity being retrieved: {entity_name}
Entity state: {json.dumps(entity_state, default=str)[:500]}

Already retrieved: {retrieved_so_far}

Rate 0-10 (0=irrelevant, 10=essential):"""

        try:
            result = self.llm.chat(
                messages=[{"role": "user", "content": prompt}],
                model="gpt-4o-mini",
                max_tokens=10,
                temperature=0,
            )
            score = float(result["content"].strip()) / 10.0
            return min(max(score, 0), 1)
        except:
            return 0.5


# ── Training Loop ─────────────────────────────────────────────────────────────

def train_retriever(
    retriever: RLRetriever,
    train_queries: list[dict],  # [{query, answer, time_range, gold_entities}]
    num_epochs: int = 3,
    log_interval: int = 10,
):
    """
    Main training loop for RL retriever.
    
    train_queries: list of dicts with:
      - query: str
      - answer: str (for checking correctness)
      - time_range: (start, end) optional
      - gold_entities: list of entity IDs that should be retrieved
    """
    logger.info(f"Training on {len(train_queries)} queries for {num_epochs} epochs")
    
    for epoch in range(num_epochs):
        random.shuffle(train_queries)
        epoch_rewards = []
        
        for i, item in enumerate(train_queries):
            query = item["query"]
            expected_answer = item.get("answer", "")
            time_range = item.get("time_range")
            gold_entities = set(item.get("gold_entities", []))
            
            def answer_fn(q, retrieved_ids):
                # In practice, run full QA pipeline and check correctness
                # For now, use gold entity overlap as proxy
                overlap = len(gold_entities & set(retrieved_ids))
                correct = overlap >= len(gold_entities) * 0.5
                return correct, gold_entities
            
            rollouts = retriever.collect_rollouts(
                query, 
                answer_fn,
                query_time_range=time_range,
            )
            
            # Store for offline training
            retriever.training_data.append((query, rollouts))
            
            # Compute loss (would update model in real impl)
            loss_info = retriever.compute_grpo_loss(rollouts)
            epoch_rewards.append(loss_info["mean_reward"])
            
            if (i + 1) % log_interval == 0:
                recent_reward = sum(epoch_rewards[-log_interval:]) / log_interval
                logger.info(
                    f"Epoch {epoch+1}, Step {i+1}/{len(train_queries)}: "
                    f"mean_reward={recent_reward:.3f}"
                )
        
        epoch_mean = sum(epoch_rewards) / len(epoch_rewards)
        logger.info(f"Epoch {epoch+1} complete: mean_reward={epoch_mean:.3f}")


# ── CLI ───────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import sys
    
    print("RL Retriever — Training data collector")
    print("This module provides the training loop for GRPO-based retrieval learning.")
    print("\nUsage:")
    print("  from pie.retrieval.rl_retriever import RLRetriever, train_retriever")
    print("  retriever = RLRetriever(world_model, llm_client)")
    print("  train_retriever(retriever, train_queries)")
