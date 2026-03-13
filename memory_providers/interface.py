"""
Unified Memory Provider Interface

All providers implement this interface for fair benchmark comparison.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any, Optional, List, Dict


@dataclass
class MemoryProviderConfig:
    """Configuration for a memory provider."""
    api_key: Optional[str] = None
    model: str = "gpt-4o"
    embedding_model: str = "text-embedding-3-large"
    max_tokens: int = 8000
    extra: Dict = field(default_factory=dict)


@dataclass
class SearchResult:
    """Result from memory search."""
    content: str
    score: float
    metadata: Dict = field(default_factory=dict)


@dataclass
class MemoryStats:
    """Statistics about the memory store."""
    num_memories: int = 0
    num_entities: int = 0
    num_relationships: int = 0
    extra: Dict = field(default_factory=dict)


class MemoryProvider(ABC):
    """
    Abstract interface for memory providers.
    
    All providers must implement:
        - ingest(): Add conversations/episodes to memory
        - search(): Retrieve relevant context for a query
        - answer(): Generate answer using memory context
        - stats(): Get memory statistics
        - clear(): Reset memory state
    """
    
    name: str = "base"
    
    def __init__(self, config: Optional[MemoryProviderConfig] = None):
        self.config = config or MemoryProviderConfig()
    
    @abstractmethod
    def ingest(self, sessions: List[List[Dict]], dates: Optional[List[str]] = None) -> None:
        """
        Ingest conversation sessions into memory.
        
        Args:
            sessions: List of sessions, each session is a list of turns
                      Turn format: {"role": "user"|"assistant", "content": str}
            dates: Optional list of dates for each session
        """
        pass
    
    @abstractmethod
    def search(self, query: str, top_k: int = 10) -> List[SearchResult]:
        """
        Search memory for relevant context.
        
        Args:
            query: Search query
            top_k: Number of results to return
            
        Returns:
            List of SearchResult objects
        """
        pass
    
    @abstractmethod
    def answer(self, question: str, question_date: Optional[str] = None) -> str:
        """
        Answer a question using memory context.
        
        This is the end-to-end method used in benchmarks:
        1. Search for relevant context
        2. Compile context for LLM
        3. Generate answer
        
        Args:
            question: Question to answer
            question_date: Optional date context for the question
            
        Returns:
            Generated answer string
        """
        pass
    
    @abstractmethod
    def stats(self) -> MemoryStats:
        """Get statistics about the memory store."""
        pass
    
    @abstractmethod
    def clear(self) -> None:
        """Clear all memory state."""
        pass
    
    def __repr__(self) -> str:
        return f"{self.__class__.__name__}(name={self.name})"
