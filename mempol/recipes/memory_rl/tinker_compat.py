"""Tinker cookbook compatibility shim.

Goal: a single import path used by tools.py / write_tools.py / memory_env.py
that works in BOTH modes:

  1. Standalone (no tinker-cookbook installed) — for unit tests and smoke runs.
     `@tool` becomes a no-op pass-through. `build_agent_tool_env` raises a
     clear error if you try to use it.

  2. Inside a tinker-cookbook clone — for real GRPO training.
     Imports the real `@tool` decorator and `build_agent_tool_env` factory.

Usage:
    from mempol.recipes.memory_rl.tinker_compat import tool, build_agent_tool_env, HAS_TINKER

    class MemoryTool:
        @tool
        def memory_search(self, query: str, k: int = 10) -> str:
            ...
"""
from __future__ import annotations
import functools
import inspect
from typing import Any, Callable

HAS_TINKER = False

try:
    from tinker_cookbook.tool_use import (  # type: ignore
        tool as _tool_real,
        build_agent_tool_env as _build_agent_tool_env_real,
        simple_tool_result as _simple_tool_result_real,
        ToolResult as _ToolResult_real,
    )
    HAS_TINKER = True

    def tool(fn=None, **kwargs):
        """Pass-through to tinker_cookbook.tool_use.tool (real decorator)."""
        if fn is None:
            return lambda f: _tool_real(f, **kwargs)
        return _tool_real(fn, **kwargs)

    def build_agent_tool_env(**kwargs):
        return _build_agent_tool_env_real(**kwargs)

    def simple_tool_result(content, **kwargs):
        return _simple_tool_result_real(content, **kwargs)

    ToolResult = _ToolResult_real

except Exception:
    # Cookbook not available — provide no-op fallbacks.
    def tool(fn=None, **kwargs):
        """No-op decorator. Mirrors tinker_cookbook.tool_use.tool's signature
        so callsites identical to the real one work in either mode.

        Adds a synthetic `.to_spec()` method to the wrapped function so
        memory_env.py's tool registration works without the cookbook."""
        def _wrap(f: Callable) -> Callable:
            sig = inspect.signature(f)
            params = []
            for name, p in sig.parameters.items():
                if name == "self":
                    continue
                ann = p.annotation if p.annotation is not inspect.Parameter.empty else "Any"
                default = None if p.default is inspect.Parameter.empty else p.default
                params.append({"name": name, "annotation": str(ann), "default": default})
            spec = {
                "name": f.__name__,
                "description": (f.__doc__ or "").strip()[:300],
                "parameters": params,
            }

            @functools.wraps(f)
            def wrapped(*args, **kw):
                return f(*args, **kw)

            wrapped.to_spec = lambda _spec=spec: _spec  # type: ignore[attr-defined]
            wrapped._mempol_tool = True  # type: ignore[attr-defined]
            return wrapped

        if fn is None:
            return _wrap
        return _wrap(fn)

    def build_agent_tool_env(**kwargs):  # type: ignore[no-redef]
        raise RuntimeError(
            "build_agent_tool_env is unavailable: tinker_cookbook is not "
            "importable. Clone https://github.com/thinking-machines-lab/"
            "tinker-cookbook and `pip install -e .` to use real envs."
        )

    # Standalone stubs so tools.py / write_tools.py imports don't blow up
    # outside the cookbook. These are NEVER used at training time — only for
    # unit tests that import the modules.
    class ToolResult(dict):                                # type: ignore[no-redef]
        def __init__(self, content="", **kwargs):
            super().__init__(content=content, **kwargs)

    def simple_tool_result(content, **kwargs):              # type: ignore[no-redef]
        return ToolResult(content=content, **kwargs)


__all__ = ["tool", "build_agent_tool_env", "simple_tool_result", "ToolResult", "HAS_TINKER"]
