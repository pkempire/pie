"""Thin OpenAI wrapper with embedding cache. No heavy deps."""
from __future__ import annotations
import hashlib
import json
import os
import time
from pathlib import Path
from typing import Iterable

import numpy as np
from openai import OpenAI

from . import config

_client: OpenAI | None = None


def client() -> OpenAI:
    global _client
    if _client is None:
        _client = OpenAI(api_key=config.OPENAI_API_KEY)
    return _client


def _embed_cache_path(model: str) -> Path:
    return config.CACHE_DIR / f"emb_{model.replace('/', '_')}.jsonl"


_emb_mem: dict[str, dict[str, list[float]]] = {}


def _load_emb_cache(model: str) -> dict[str, list[float]]:
    if model in _emb_mem:
        return _emb_mem[model]
    path = _embed_cache_path(model)
    cache: dict[str, list[float]] = {}
    if path.exists():
        for line in path.read_text().splitlines():
            try:
                obj = json.loads(line)
                cache[obj["k"]] = obj["v"]
            except Exception:
                continue
    _emb_mem[model] = cache
    return cache


def _key(text: str) -> str:
    return hashlib.sha1(text.encode("utf-8")).hexdigest()


def embed(texts: list[str], model: str | None = None, batch: int = 64) -> np.ndarray:
    """Batched embedding with on-disk cache. Returns (N, D) float32 array."""
    model = model or config.EMBED_MODEL
    cache = _load_emb_cache(model)
    keys = [_key(t) for t in texts]
    missing_idx = [i for i, k in enumerate(keys) if k not in cache]
    if missing_idx:
        path = _embed_cache_path(model)
        with path.open("a") as f:
            for s in range(0, len(missing_idx), batch):
                chunk_idx = missing_idx[s : s + batch]
                resp = client().embeddings.create(
                    model=model, input=[texts[i] for i in chunk_idx]
                )
                for i, d in zip(chunk_idx, resp.data):
                    cache[keys[i]] = d.embedding
                    f.write(json.dumps({"k": keys[i], "v": d.embedding}) + "\n")
    return np.array([cache[k] for k in keys], dtype=np.float32)


def _is_reasoning_model(model: str) -> bool:
    """gpt-5*, o1*, o3*, o4* only support the default temperature (1.0).
    They also error on top_p / presence_penalty / frequency_penalty when set."""
    m = model.lower()
    return m.startswith("gpt-5") or m.startswith("o1") or m.startswith("o3") or m.startswith("o4")


def chat(
    messages: list[dict], model: str | None = None, json_mode: bool = False, **kw
) -> str:
    model = model or config.ANSWER_MODEL
    reasoning = _is_reasoning_model(model)
    if not reasoning:
        kw.setdefault("temperature", 0.0)
    else:
        # Strip any sampler params the user might have passed — gpt-5/o-series reject them
        for bad in ("temperature", "top_p", "presence_penalty", "frequency_penalty"):
            kw.pop(bad, None)
        if "max_tokens" in kw and "max_completion_tokens" not in kw:
            kw["max_completion_tokens"] = kw.pop("max_tokens")
    if json_mode:
        kw["response_format"] = {"type": "json_object"}
    for attempt in range(3):
        try:
            resp = client().chat.completions.create(model=model, messages=messages, **kw)
            return resp.choices[0].message.content or ""
        except Exception as e:
            if attempt == 2:
                raise
            time.sleep(1.0 * (attempt + 1))
    return ""
