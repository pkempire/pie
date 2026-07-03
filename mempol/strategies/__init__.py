"""Memory strategy plugin system.

Quick start:
    from mempol.strategies import REGISTRY
    from mempol.strategies.registry import describe_all

    describe_all()
    strategy = REGISTRY["chronos"]
    backend, metrics = strategy.build_backend(conv)
    answer, trace = strategy.run(question, backend, qa)
"""
from .registry import REGISTRY, describe_all

__all__ = ["REGISTRY", "describe_all"]
