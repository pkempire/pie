"""
MemoryAgentBench Data Adapter

Loads the HuggingFace dataset and adapts it for benchmark evaluation.
Supports all 4 competencies:
  - Accurate Retrieval (AR)
  - Test-Time Learning (TTL)
  - Long-Range Understanding (LRU)
  - Conflict Resolution (CR)
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Any, Iterator

logger = logging.getLogger("pie.bench.memoryagentbench")


@dataclass
class BenchmarkItem:
    """A single context with its associated questions."""
    context: str
    questions: list[str]
    answers: list[Any]
    qa_pair_ids: list[str]
    competency: str  # AR, TTL, LRU, CR
    source: str
    metadata: dict = field(default_factory=dict)
    
    @property
    def num_questions(self) -> int:
        return len(self.questions)
    
    def iter_qa_pairs(self) -> Iterator[tuple[int, str, Any, str]]:
        """Iterate over (index, question, answer, qa_pair_id) tuples."""
        for i, (q, a, qid) in enumerate(zip(self.questions, self.answers, self.qa_pair_ids)):
            yield i, q, a, qid


def load_dataset(competency: str | None = None) -> list[BenchmarkItem]:
    """
    Load MemoryAgentBench from HuggingFace.
    
    Args:
        competency: Filter to specific competency (AR, TTL, LRU, CR) or None for all
        
    Returns:
        List of BenchmarkItem objects
    """
    from datasets import load_dataset as hf_load
    
    ds = hf_load("ai-hyz/MemoryAgentBench")
    
    # Map split names to competency codes
    competency_map = {
        "Accurate_Retrieval": "AR",
        "Test_Time_Learning": "TTL", 
        "Long_Range_Understanding": "LRU",
        "Conflict_Resolution": "CR",
    }
    
    items = []
    
    for split_name, comp_code in competency_map.items():
        if competency and comp_code != competency:
            continue
            
        if split_name not in ds:
            logger.warning(f"Split {split_name} not found in dataset")
            continue
            
        split_data = ds[split_name]
        
        for row in split_data:
            # Extract metadata
            metadata = row.get("metadata", {}) or {}
            qa_pair_ids = metadata.get("qa_pair_ids", [])
            source = metadata.get("source", split_name)
            
            # Ensure qa_pair_ids has same length as questions
            questions = row.get("questions", [])
            answers = row.get("answers", [])
            
            if not qa_pair_ids:
                qa_pair_ids = [f"{source}_q{i}" for i in range(len(questions))]
            elif len(qa_pair_ids) < len(questions):
                qa_pair_ids.extend([f"{source}_q{i}" for i in range(len(qa_pair_ids), len(questions))])
            
            items.append(BenchmarkItem(
                context=row["context"],
                questions=questions,
                answers=answers,
                qa_pair_ids=qa_pair_ids,
                competency=comp_code,
                source=source,
                metadata=metadata,
            ))
    
    logger.info(f"Loaded {len(items)} items from MemoryAgentBench")
    return items


def dataset_stats(items: list[BenchmarkItem]) -> dict:
    """Get statistics about the loaded dataset."""
    from collections import Counter
    
    competency_counts = Counter(item.competency for item in items)
    total_questions = sum(item.num_questions for item in items)
    total_context_chars = sum(len(item.context) for item in items)
    
    return {
        "total_items": len(items),
        "total_questions": total_questions,
        "total_context_chars": total_context_chars,
        "by_competency": dict(competency_counts),
        "avg_questions_per_item": total_questions / len(items) if items else 0,
        "avg_context_chars": total_context_chars / len(items) if items else 0,
    }


def filter_by_competency(items: list[BenchmarkItem], competency: str) -> list[BenchmarkItem]:
    """Filter items to a specific competency."""
    return [item for item in items if item.competency == competency]


def filter_by_source(items: list[BenchmarkItem], source_pattern: str) -> list[BenchmarkItem]:
    """Filter items by source pattern (substring match)."""
    return [item for item in items if source_pattern.lower() in item.source.lower()]


def chunk_context(context: str, chunk_size: int = 4096, overlap: int = 256) -> list[str]:
    """
    Split context into overlapping chunks.
    
    Args:
        context: The full context string
        chunk_size: Maximum size of each chunk in characters
        overlap: Number of characters to overlap between chunks
        
    Returns:
        List of context chunks
    """
    if len(context) <= chunk_size:
        return [context]
    
    chunks = []
    start = 0
    
    while start < len(context):
        end = min(start + chunk_size, len(context))
        
        # Try to break at sentence boundary
        if end < len(context):
            # Look for sentence end within last 20% of chunk
            search_start = end - int(chunk_size * 0.2)
            for punct in ['. ', '.\n', '! ', '? ']:
                last_punct = context.rfind(punct, search_start, end)
                if last_punct > search_start:
                    end = last_punct + 1
                    break
        
        chunks.append(context[start:end])
        
        if end >= len(context):
            break
            
        start = end - overlap
    
    return chunks


def format_context_for_prompt(context: str, max_chars: int = 100_000) -> str:
    """Format context for inclusion in an LLM prompt, with truncation if needed."""
    if len(context) <= max_chars:
        return context
    
    # Truncate from middle, keeping start and end
    half = max_chars // 2 - 50  # Leave room for truncation notice
    return (
        context[:half] + 
        "\n\n[... content truncated due to length ...]\n\n" + 
        context[-half:]
    )
