"""
MemoryAgentBench Baselines

Three baseline approaches:
  1. long_context  — stuff full context into LLM prompt (up to context limit)
  2. naive_rag     — chunk context, embed, retrieve top-k, answer
  3. pie_temporal  — PIE's world model approach: ingest → build entities → query

Designed to reproduce paper baselines and compare with PIE.
"""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from pie.core.llm import LLMClient
from pie.core.world_model import WorldModel, cosine_similarity

from .adapter import BenchmarkItem, chunk_context, format_context_for_prompt

logger = logging.getLogger("pie.bench.memoryagentbench")


# ── Result Types ──────────────────────────────────────────────────────────────


@dataclass
class BaselineResult:
    """Result from running a baseline on one question."""
    qa_pair_id: str
    competency: str
    source: str
    question: str
    gold_answer: Any
    hypothesis: str
    baseline_name: str
    model: str
    latency_ms: float = 0.0
    context_chars: int = 0
    retrieval_count: int = 0
    error: str | None = None

    def to_dict(self) -> dict:
        return {
            "qa_pair_id": self.qa_pair_id,
            "competency": self.competency,
            "source": self.source,
            "question": self.question,
            "gold_answer": self.gold_answer if isinstance(self.gold_answer, str) else str(self.gold_answer),
            "hypothesis": self.hypothesis,
            "baseline_name": self.baseline_name,
            "model": self.model,
            "latency_ms": round(self.latency_ms, 1),
            "context_chars": self.context_chars,
            "retrieval_count": self.retrieval_count,
            "error": self.error,
        }


# ── Shared QA Prompts ─────────────────────────────────────────────────────────

ANSWER_SYSTEM_PROMPT = """\
You are a helpful assistant answering questions based on provided context.
Answer the question based ONLY on the context provided.
Be concise and specific. If the context doesn't contain the answer, say "I don't know."
Do NOT make up information.
Give only the direct answer without explanation unless asked."""

ANSWER_USER_TEMPLATE = """\
Context:
{context}

---

Question: {question}

Answer concisely:"""


def _ask_llm(
    context: str,
    question: str,
    llm: LLMClient,
    model: str = "gpt-4o-mini",
    max_tokens: int = 200,
) -> str:
    """Ask an LLM to answer a question given context."""
    messages = [
        {"role": "system", "content": ANSWER_SYSTEM_PROMPT},
        {
            "role": "user",
            "content": ANSWER_USER_TEMPLATE.format(
                context=context,
                question=question,
            ),
        },
    ]
    result = llm.chat(messages=messages, model=model, max_tokens=max_tokens)
    return result["content"].strip()


# ═══════════════════════════════════════════════════════════════════════════════
# Baseline 1: Long Context (Full Context in Prompt)
# ═══════════════════════════════════════════════════════════════════════════════


def long_context(
    item: BenchmarkItem,
    question_idx: int,
    llm: LLMClient | None = None,
    model: str = "gpt-4o-mini",
    max_context_chars: int = 120_000,
) -> BaselineResult:
    """
    Long-context baseline: stuff full context into the prompt.
    
    This is the simplest approach — limited only by context window.
    For very long contexts, truncates to fit.
    """
    llm = llm or LLMClient()
    t0 = time.time()
    
    question = item.questions[question_idx]
    answer = item.answers[question_idx]
    qa_pair_id = item.qa_pair_ids[question_idx]

    try:
        context = format_context_for_prompt(item.context, max_context_chars)
        
        hypothesis = _ask_llm(
            context=context,
            question=question,
            llm=llm,
            model=model,
        )

        return BaselineResult(
            qa_pair_id=qa_pair_id,
            competency=item.competency,
            source=item.source,
            question=question,
            gold_answer=answer,
            hypothesis=hypothesis,
            baseline_name="long_context",
            model=model,
            latency_ms=(time.time() - t0) * 1000,
            context_chars=len(context),
        )

    except Exception as e:
        logger.error(f"Long context error: {e}")
        return BaselineResult(
            qa_pair_id=qa_pair_id,
            competency=item.competency,
            source=item.source,
            question=question,
            gold_answer=answer,
            hypothesis=f"Error: {e}",
            baseline_name="long_context",
            model=model,
            latency_ms=(time.time() - t0) * 1000,
            error=str(e),
        )


# ═══════════════════════════════════════════════════════════════════════════════
# Baseline 2: Naive RAG
# ═══════════════════════════════════════════════════════════════════════════════


class NaiveRAGRetriever:
    """Simple embedding-based retriever for naive RAG baseline."""
    
    # Embedding model dimensions
    MODEL_DIMS = {
        "text-embedding-3-small": 1536,
        "text-embedding-3-large": 3072,
        "text-embedding-ada-002": 1536,
    }
    
    def __init__(
        self,
        llm: LLMClient,
        embed_model: str = "text-embedding-3-small",
        chunk_size: int = 4096,
        chunk_overlap: int = 256,
    ):
        self.llm = llm
        self.embed_model = embed_model
        self.embed_dims = self.MODEL_DIMS.get(embed_model, 1536)
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.chunks: list[str] = []
        self.embeddings: list[list[float]] = []
        
    def index_context(self, context: str):
        """Chunk and embed the context."""
        self.chunks = chunk_context(context, self.chunk_size, self.chunk_overlap)
        
        # Embed all chunks
        self.embeddings = []
        for chunk in self.chunks:
            emb = self.llm.embed([chunk], model=self.embed_model, dimensions=self.embed_dims)
            self.embeddings.append(emb[0])
            
        logger.debug(f"Indexed {len(self.chunks)} chunks")
        
    def retrieve(self, query: str, top_k: int = 5) -> list[str]:
        """Retrieve top-k most relevant chunks for a query."""
        if not self.chunks:
            return []
            
        # Embed query (must use same dimensions as index)
        query_emb = self.llm.embed([query], model=self.embed_model, dimensions=self.embed_dims)[0]
        
        # Score all chunks
        scores = [cosine_similarity(query_emb, chunk_emb) for chunk_emb in self.embeddings]
        
        # Get top-k indices
        sorted_indices = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)
        top_indices = sorted_indices[:top_k]
        
        return [self.chunks[i] for i in top_indices]


def naive_rag(
    item: BenchmarkItem,
    question_idx: int,
    retriever: NaiveRAGRetriever | None = None,
    llm: LLMClient | None = None,
    model: str = "gpt-4o-mini",
    embed_model: str = "text-embedding-3-small",
    top_k: int = 10,
    chunk_size: int = 4096,
) -> BaselineResult:
    """
    Naive RAG baseline: chunk context, embed, retrieve top-k, answer.
    
    Pass a pre-built retriever to avoid re-indexing for each question
    on the same context.
    """
    llm = llm or LLMClient()
    t0 = time.time()
    
    question = item.questions[question_idx]
    answer = item.answers[question_idx]
    qa_pair_id = item.qa_pair_ids[question_idx]

    try:
        # Build retriever if not provided
        if retriever is None:
            retriever = NaiveRAGRetriever(llm, embed_model, chunk_size)
            retriever.index_context(item.context)
        
        # Retrieve relevant chunks
        retrieved = retriever.retrieve(question, top_k)
        
        # Build context from retrieved chunks
        context = "\n\n---\n\n".join(retrieved)
        
        # Generate answer
        hypothesis = _ask_llm(
            context=context,
            question=question,
            llm=llm,
            model=model,
        )

        return BaselineResult(
            qa_pair_id=qa_pair_id,
            competency=item.competency,
            source=item.source,
            question=question,
            gold_answer=answer,
            hypothesis=hypothesis,
            baseline_name="naive_rag",
            model=model,
            latency_ms=(time.time() - t0) * 1000,
            context_chars=len(context),
            retrieval_count=len(retrieved),
        )

    except Exception as e:
        logger.error(f"Naive RAG error: {e}")
        return BaselineResult(
            qa_pair_id=qa_pair_id,
            competency=item.competency,
            source=item.source,
            question=question,
            gold_answer=answer,
            hypothesis=f"Error: {e}",
            baseline_name="naive_rag",
            model=model,
            latency_ms=(time.time() - t0) * 1000,
            error=str(e),
        )


# ═══════════════════════════════════════════════════════════════════════════════
# Baseline 3: PIE Temporal (World Model Approach)
# ═══════════════════════════════════════════════════════════════════════════════


class PIEWorldModelBuilder:
    """Build PIE world model from context for answering questions."""
    
    def __init__(
        self,
        llm: LLMClient,
        model: str = "gpt-4o-mini",
        chunk_size: int = 8192,
    ):
        self.llm = llm
        self.model = model
        self.chunk_size = chunk_size
        self.world_model: WorldModel | None = None
        
    def build_from_context(self, context: str) -> WorldModel:
        """Build a world model from context text."""
        from pie.core.models import Conversation, Turn
        
        # Create world model
        self.world_model = WorldModel(llm=self.llm, model=self.model)
        
        # Chunk the context
        chunks = chunk_context(context, self.chunk_size, overlap=512)
        
        # Process each chunk to extract entities and relationships
        for i, chunk in enumerate(chunks):
            # Create a pseudo-conversation for ingestion
            conv = Conversation(
                conversation_id=f"context_chunk_{i}",
                source="memoryagentbench",
                turns=[
                    Turn(
                        turn_id=f"chunk_{i}",
                        role="context",
                        content=chunk,
                        timestamp=None,
                    )
                ],
            )
            
            # Ingest into world model
            self.world_model.ingest_conversation(conv)
            
            logger.debug(f"Processed chunk {i+1}/{len(chunks)}")
        
        logger.info(f"Built world model with {len(self.world_model.entities)} entities")
        return self.world_model
    
    def query(self, question: str, top_k: int = 10) -> str:
        """Query the world model and compile context for answering."""
        if self.world_model is None:
            raise ValueError("World model not built yet")
        
        # Get relevant entities
        relevant = self.world_model.query(question, top_k=top_k)
        
        # Compile context from entities
        context_parts = []
        for entity, score in relevant:
            if entity.summary:
                context_parts.append(f"• {entity.name}: {entity.summary}")
            for fact in entity.facts[:5]:  # Limit facts per entity
                context_parts.append(f"  - {fact}")
        
        return "\n".join(context_parts)


def pie_temporal(
    item: BenchmarkItem,
    question_idx: int,
    builder: PIEWorldModelBuilder | None = None,
    llm: LLMClient | None = None,
    model: str = "gpt-4o-mini",
    top_k: int = 10,
) -> BaselineResult:
    """
    PIE temporal baseline: build world model from context, query for relevant
    entities, compile context, and answer.
    
    Pass a pre-built builder to avoid re-building world model for each question.
    """
    llm = llm or LLMClient()
    t0 = time.time()
    
    question = item.questions[question_idx]
    answer = item.answers[question_idx]
    qa_pair_id = item.qa_pair_ids[question_idx]

    try:
        # Build world model if not provided
        if builder is None:
            builder = PIEWorldModelBuilder(llm, model)
            builder.build_from_context(item.context)
        
        # Query world model for relevant context
        context = builder.query(question, top_k)
        
        if not context.strip():
            # Fallback: use first chunk of raw context
            context = item.context[:10000]
        
        # Generate answer
        hypothesis = _ask_llm(
            context=context,
            question=question,
            llm=llm,
            model=model,
        )

        return BaselineResult(
            qa_pair_id=qa_pair_id,
            competency=item.competency,
            source=item.source,
            question=question,
            gold_answer=answer,
            hypothesis=hypothesis,
            baseline_name="pie_temporal",
            model=model,
            latency_ms=(time.time() - t0) * 1000,
            context_chars=len(context),
            retrieval_count=top_k,
        )

    except Exception as e:
        logger.error(f"PIE temporal error: {e}")
        return BaselineResult(
            qa_pair_id=qa_pair_id,
            competency=item.competency,
            source=item.source,
            question=question,
            gold_answer=answer,
            hypothesis=f"Error: {e}",
            baseline_name="pie_temporal",
            model=model,
            latency_ms=(time.time() - t0) * 1000,
            error=str(e),
        )


# ── Registry ──────────────────────────────────────────────────────────────────

BASELINES = {
    "long_context": long_context,
    "naive_rag": naive_rag,
    "pie_temporal": pie_temporal,
}
