"""Base class and metadata for MemoryStrategy plugins.

A MemoryStrategy is the atomic unit of comparison in the evaluation matrix.
Each strategy encapsulates:
  - Paper metadata (arXiv ID, title, URL)
  - Method taxonomy (tags)
  - build_backend(): ingest a conversation into memory
  - run(): answer a question from that memory

This design keeps the evaluation harness (experiments/run_matrix.py) fully
decoupled from strategy implementations — adding a new method is a single
class addition to implementations.py plus a registry entry.
"""
from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from mempol.backends.base import Backend
    from mempol.data.locomo import Conversation, QA


@dataclass
class PaperRef:
    """Citation metadata for a strategy's primary paper."""

    arxiv_id: str
    title: str
    url: str = ""

    def __post_init__(self) -> None:
        if not self.url and self.arxiv_id:
            self.url = f"https://arxiv.org/abs/{self.arxiv_id}"

    def __str__(self) -> str:
        return f"arXiv:{self.arxiv_id} — {self.title}"


class MemoryStrategy(ABC):
    """
    Plugin interface for memory strategies.

    Each strategy must declare:
    - name: short snake_case identifier
    - label: human-readable display name
    - paper: PaperRef for the paper this implements
    - tags: list of method categories e.g. ["RAG", "temporal-aware", "write-time"]
    - runnable: True if this can run today with just OPENAI_API_KEY

    And implement:
    - build_backend(conv: Conversation) -> tuple[Backend, dict]:
        Build the memory store from a conversation.
        Returns (backend, storage_metrics_dict).
    - run(question: str, backend: Backend, qa: QA) -> tuple[str, dict]:
        Answer the question. Returns (answer, trace_dict).
    """

    name: str
    label: str
    paper: PaperRef
    tags: list[str] = field(default_factory=list)
    runnable: bool = True

    @abstractmethod
    def build_backend(self, conv: "Conversation") -> "tuple[Backend, dict]":
        """Ingest conv into memory. Returns (backend, storage_metrics_dict)."""
        ...

    @abstractmethod
    def run(
        self, question: str, backend: "Backend", qa: "QA"
    ) -> "tuple[str, dict]":
        """Answer question from backend. Returns (answer, trace_dict)."""
        ...
