"""
Paper Baselines for Fair Comparison

Implements retrieval methods from:
- LongMemEval (Wu et al., ICLR 2025): BM25, Contriever, Stella
- LoCoMo (Maharana et al., ACL 2024): BM25, DPR, Contriever

These are the standard baselines papers use for comparison.
"""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass
from typing import Any, Callable

import numpy as np

logger = logging.getLogger("pie.bench.baselines")


@dataclass
class RetrievalResult:
    """Result from retrieval baseline."""
    question_id: str
    question_type: str
    question: str
    gold_answer: str
    hypothesis: str
    baseline_name: str
    model: str
    latency_ms: float = 0.0
    context_chars: int = 0
    retrieval_count: int = 0
    error: str | None = None


# ═══════════════════════════════════════════════════════════════════════════════
# BM25 Baseline (Sparse Retrieval)
# ═══════════════════════════════════════════════════════════════════════════════

def bm25_retrieval(
    item: dict[str, Any],
    llm_fn: Callable,
    top_k: int = 10,
    chunk_by: str = "session",
) -> RetrievalResult:
    """
    BM25 sparse retrieval baseline.
    
    Standard baseline from LongMemEval and LoCoMo papers.
    Uses rank_bm25 for efficient BM25Okapi implementation.
    """
    try:
        from rank_bm25 import BM25Okapi
    except ImportError:
        raise ImportError("Install rank_bm25: pip install rank_bm25")
    
    t0 = time.time()
    
    try:
        # Build chunks
        chunks = _build_chunks(
            item["haystack_sessions"],
            item.get("haystack_dates", []),
            chunk_by=chunk_by,
        )
        
        if not chunks:
            return RetrievalResult(
                question_id=item["question_id"],
                question_type=item.get("question_type", "unknown"),
                question=item["question"],
                gold_answer=item["answer"],
                hypothesis="No chunks to search.",
                baseline_name=f"bm25_{chunk_by}",
                model="bm25",
                latency_ms=(time.time() - t0) * 1000,
            )
        
        # Tokenize
        tokenized_chunks = [_tokenize(c["text"]) for c in chunks]
        tokenized_query = _tokenize(item["question"])
        
        # Build BM25 index
        bm25 = BM25Okapi(tokenized_chunks)
        scores = bm25.get_scores(tokenized_query)
        
        # Get top-k
        top_indices = np.argsort(scores)[::-1][:top_k]
        top_chunks = [(chunks[i], scores[i]) for i in top_indices if scores[i] > 0]
        
        # Sort by timestamp for context
        top_chunks.sort(key=lambda x: x[0].get("timestamp", 0))
        
        # Build context
        context_parts = []
        for chunk, score in top_chunks:
            context_parts.append(f"[{chunk.get('date', 'Unknown')} | BM25: {score:.2f}]\n{chunk['text']}")
        context = "\n\n".join(context_parts)
        
        # Generate answer
        answer = llm_fn(context, item["question"], item.get("question_date", ""))
        
        return RetrievalResult(
            question_id=item["question_id"],
            question_type=item.get("question_type", "unknown"),
            question=item["question"],
            gold_answer=item["answer"],
            hypothesis=answer,
            baseline_name=f"bm25_{chunk_by}",
            model="bm25+gpt-4o",
            latency_ms=(time.time() - t0) * 1000,
            context_chars=len(context),
            retrieval_count=len(top_chunks),
        )
        
    except Exception as e:
        return RetrievalResult(
            question_id=item["question_id"],
            question_type=item.get("question_type", "unknown"),
            question=item["question"],
            gold_answer=item["answer"],
            hypothesis=f"Error: {e}",
            baseline_name=f"bm25_{chunk_by}",
            model="bm25",
            latency_ms=(time.time() - t0) * 1000,
            error=str(e),
        )


# ═══════════════════════════════════════════════════════════════════════════════
# Contriever Baseline (Dense Retrieval - Facebook)
# ═══════════════════════════════════════════════════════════════════════════════

def contriever_retrieval(
    item: dict[str, Any],
    llm_fn: Callable,
    top_k: int = 10,
    chunk_by: str = "session",
    model_name: str = "facebook/contriever",
) -> RetrievalResult:
    """
    Contriever dense retrieval baseline.
    
    Facebook's unsupervised dense retriever, used in both LongMemEval and LoCoMo.
    """
    try:
        from sentence_transformers import SentenceTransformer
    except ImportError:
        raise ImportError("Install sentence-transformers: pip install sentence-transformers")
    
    t0 = time.time()
    
    try:
        # Load model (cached after first load)
        model = _get_contriever_model(model_name)
        
        # Build chunks
        chunks = _build_chunks(
            item["haystack_sessions"],
            item.get("haystack_dates", []),
            chunk_by=chunk_by,
        )
        
        if not chunks:
            return RetrievalResult(
                question_id=item["question_id"],
                question_type=item.get("question_type", "unknown"),
                question=item["question"],
                gold_answer=item["answer"],
                hypothesis="No chunks to search.",
                baseline_name=f"contriever_{chunk_by}",
                model=model_name,
                latency_ms=(time.time() - t0) * 1000,
            )
        
        # Embed
        chunk_texts = [c["text"] for c in chunks]
        chunk_embeddings = model.encode(chunk_texts, convert_to_numpy=True, show_progress_bar=False)
        query_embedding = model.encode([item["question"]], convert_to_numpy=True)[0]
        
        # Compute cosine similarity
        scores = np.dot(chunk_embeddings, query_embedding) / (
            np.linalg.norm(chunk_embeddings, axis=1) * np.linalg.norm(query_embedding)
        )
        
        # Get top-k
        top_indices = np.argsort(scores)[::-1][:top_k]
        top_chunks = [(chunks[i], scores[i]) for i in top_indices]
        
        # Sort by timestamp
        top_chunks.sort(key=lambda x: x[0].get("timestamp", 0))
        
        # Build context
        context_parts = []
        for chunk, score in top_chunks:
            context_parts.append(f"[{chunk.get('date', 'Unknown')} | sim: {score:.3f}]\n{chunk['text']}")
        context = "\n\n".join(context_parts)
        
        # Generate answer
        answer = llm_fn(context, item["question"], item.get("question_date", ""))
        
        return RetrievalResult(
            question_id=item["question_id"],
            question_type=item.get("question_type", "unknown"),
            question=item["question"],
            gold_answer=item["answer"],
            hypothesis=answer,
            baseline_name=f"contriever_{chunk_by}",
            model=f"{model_name}+gpt-4o",
            latency_ms=(time.time() - t0) * 1000,
            context_chars=len(context),
            retrieval_count=len(top_chunks),
        )
        
    except Exception as e:
        return RetrievalResult(
            question_id=item["question_id"],
            question_type=item.get("question_type", "unknown"),
            question=item["question"],
            gold_answer=item["answer"],
            hypothesis=f"Error: {e}",
            baseline_name=f"contriever_{chunk_by}",
            model=model_name,
            latency_ms=(time.time() - t0) * 1000,
            error=str(e),
        )


# ═══════════════════════════════════════════════════════════════════════════════
# Stella Baseline (Dense Retrieval - paper's top performer)
# ═══════════════════════════════════════════════════════════════════════════════

def stella_retrieval(
    item: dict[str, Any],
    llm_fn: Callable,
    top_k: int = 10,
    chunk_by: str = "session",
    model_name: str = "dunzhang/stella_en_1.5B_v5",
) -> RetrievalResult:
    """
    Stella V5 dense retrieval baseline.
    
    LongMemEval paper's best-performing retriever.
    """
    # Same implementation as Contriever, just different model
    return contriever_retrieval(
        item=item,
        llm_fn=llm_fn,
        top_k=top_k,
        chunk_by=chunk_by,
        model_name=model_name,
    )


# ═══════════════════════════════════════════════════════════════════════════════
# Helpers
# ═══════════════════════════════════════════════════════════════════════════════

_CONTRIEVER_CACHE: dict = {}


def _get_contriever_model(model_name: str):
    """Load and cache sentence transformer model."""
    if model_name not in _CONTRIEVER_CACHE:
        from sentence_transformers import SentenceTransformer
        logger.info(f"Loading {model_name}...")
        _CONTRIEVER_CACHE[model_name] = SentenceTransformer(model_name)
    return _CONTRIEVER_CACHE[model_name]


def _tokenize(text: str) -> list[str]:
    """Simple whitespace tokenization for BM25."""
    import re
    # Lowercase, remove punctuation, split on whitespace
    text = text.lower()
    text = re.sub(r'[^\w\s]', ' ', text)
    return text.split()


def _build_chunks(
    haystack_sessions: list[list[dict]],
    haystack_dates: list[str],
    chunk_by: str = "session",
) -> list[dict]:
    """Build text chunks from haystack sessions."""
    chunks = []
    
    for i, session in enumerate(haystack_sessions):
        date = haystack_dates[i] if i < len(haystack_dates) else f"Session {i+1}"
        
        if chunk_by == "session":
            # Entire session as one chunk
            lines = []
            for turn in session:
                role = turn.get("role", "user").capitalize()
                content = turn.get("content", "").strip()
                if content:
                    lines.append(f"{role}: {content}")
            if lines:
                chunks.append({
                    "text": "\n".join(lines),
                    "date": date,
                    "timestamp": i,
                })
        
        elif chunk_by == "turn":
            for j, turn in enumerate(session):
                content = turn.get("content", "").strip()
                if not content:
                    continue
                role = turn.get("role", "user").capitalize()
                chunks.append({
                    "text": f"{role}: {content}",
                    "date": date,
                    "timestamp": i * 1000 + j,
                })
    
    return chunks


# ═══════════════════════════════════════════════════════════════════════════════
# Baseline Registry
# ═══════════════════════════════════════════════════════════════════════════════

PAPER_BASELINES = {
    "bm25_session": lambda item, llm_fn: bm25_retrieval(item, llm_fn, chunk_by="session"),
    "bm25_turn": lambda item, llm_fn: bm25_retrieval(item, llm_fn, chunk_by="turn"),
    "contriever_session": lambda item, llm_fn: contriever_retrieval(item, llm_fn, chunk_by="session"),
    "contriever_turn": lambda item, llm_fn: contriever_retrieval(item, llm_fn, chunk_by="turn"),
    "stella_session": lambda item, llm_fn: stella_retrieval(item, llm_fn, chunk_by="session"),
    "stella_turn": lambda item, llm_fn: stella_retrieval(item, llm_fn, chunk_by="turn"),
}
