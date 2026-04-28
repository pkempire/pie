"""Read policy interface and Trace data class.

A policy emits a sequence of ops (the "trajectory"). Every step is logged so
that the trace can later be used as supervised data for SFT, or as a sample
for DPO/GRPO.
"""
from __future__ import annotations
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any

from ..backends.base import Backend, Hit


@dataclass
class Step:
    """One op the policy emitted."""
    op: str                    # "reformulate" | "retrieve" | "expand" | ...
    args: dict[str, Any]
    obs_summary: str           # short text summarising what came back


@dataclass
class Trace:
    qid: str
    question: str
    backend: str
    policy: str
    steps: list[Step] = field(default_factory=list)
    final_hits: list[Hit] = field(default_factory=list)
    answer: str = ""
    cost_tokens: int = 0
    n_retrievals: int = 0


class ReadPolicy(ABC):
    name: str = "base"

    @abstractmethod
    def run(self, question: str, backend: Backend) -> Trace: ...
