"""LLM interface — wraps OpenAI API with structured output and retry logic."""

from __future__ import annotations
import json
import time
import logging
from typing import Any
from openai import OpenAI
from .models import ExtractionResult, ExtractedEntity, ExtractedStateChange, ExtractedRelationship

logger = logging.getLogger("pie.llm")

# Retry config
MAX_RETRIES = 3
RETRY_DELAY = 2.0


class LLMClient:
    """Thin wrapper around OpenAI for PIE's needs."""
    
    def __init__(self, api_key: str | None = None):
        self.client = OpenAI(api_key=api_key) if api_key else OpenAI()
        self._total_tokens = 0
        self._total_calls = 0
    
    @property
    def stats(self) -> dict:
        return {
            "total_tokens": self._total_tokens,
            "total_calls": self._total_calls,
        }
    
    # Models that don't support custom temperature
    NO_TEMP_MODELS = {"gpt-5-mini", "gpt-5-nano", "gpt-5-mini-2025-08-07", "gpt-5-nano-2025-08-07"}
    
    def chat(
        self,
        messages: list[dict],
        model: str = "gpt-5-mini",
        temperature: float = 0.1,
        json_mode: bool = False,
        max_tokens: int | None = None,
    ) -> dict:
        """
        Make a chat completion call. Returns parsed JSON if json_mode, else raw content.
        """
        kwargs: dict[str, Any] = {
            "model": model,
            "messages": messages,
        }
        # Some models don't support custom temperature — skip for those
        model_base = model.split("-2025")[0] if "-2025" in model else model
        if model_base not in self.NO_TEMP_MODELS:
            kwargs["temperature"] = temperature
        if json_mode:
            kwargs["response_format"] = {"type": "json_object"}
        if max_tokens:
            # gpt-5-mini/nano use max_completion_tokens instead of max_tokens
            if model_base in self.NO_TEMP_MODELS:
                kwargs["max_completion_tokens"] = max_tokens
            else:
                kwargs["max_tokens"] = max_tokens
        
        for attempt in range(MAX_RETRIES):
            try:
                response = self.client.chat.completions.create(**kwargs)
                self._total_tokens += response.usage.total_tokens
                self._total_calls += 1
                
                content = response.choices[0].message.content
                
                # Reasoning models may return None/empty content on some calls
                if json_mode and (content is None or content.strip() == ""):
                    logger.warning(f"Empty content from {model} (attempt {attempt+1}), retrying...")
                    if attempt < MAX_RETRIES - 1:
                        time.sleep(RETRY_DELAY * (attempt + 1))
                        continue
                    raise RuntimeError(f"Model {model} returned empty content after {MAX_RETRIES} attempts")
                
                result = {
                    "content": json.loads(content) if json_mode else content,
                    "tokens": {
                        "prompt": response.usage.prompt_tokens,
                        "completion": response.usage.completion_tokens,
                        "total": response.usage.total_tokens,
                    },
                }
                return result
                
            except json.JSONDecodeError as e:
                logger.warning(f"JSON parse error (attempt {attempt+1}): {e}")
                if attempt == MAX_RETRIES - 1:
                    raise
            except Exception as e:
                logger.warning(f"LLM call failed (attempt {attempt+1}): {e}")
                if attempt == MAX_RETRIES - 1:
                    raise
                time.sleep(RETRY_DELAY * (attempt + 1))
        
        raise RuntimeError("LLM call failed after all retries")
    
    def embed(
        self,
        texts: list[str],
        model: str = "text-embedding-3-large",
        dimensions: int = 3072,
    ) -> list[list[float]]:
        """Get embeddings for a list of texts."""
        # OpenAI allows batching up to 2048 inputs
        response = self.client.embeddings.create(
            model=model,
            input=texts,
            dimensions=dimensions,
        )
        self._total_tokens += response.usage.total_tokens
        self._total_calls += 1
        return [item.embedding for item in response.data]
    
    def embed_single(
        self,
        text: str,
        model: str = "text-embedding-3-large",
        dimensions: int = 3072,
    ) -> list[float]:
        """Get embedding for a single text."""
        return self.embed([text], model=model, dimensions=dimensions)[0]


def parse_extraction_result(raw: dict, conversation_ids: list[str], tokens: dict) -> ExtractionResult:
    """Parse raw LLM JSON output into ExtractionResult."""
    
    entities = []
    for e in raw.get("entities", []):
        # Validate entity type
        etype = e.get("type", "concept").lower()
        valid_types = {"person", "project", "tool", "organization", "belief", "decision", "concept", "period", "event"}
        if etype not in valid_types:
            etype = "concept"  # default fallback
        
        entities.append(ExtractedEntity(
            name=e.get("name", ""),
            type=etype,
            state=e.get("state", {}) if isinstance(e.get("state"), dict) else {"description": str(e.get("state", ""))},
            is_new=e.get("is_new", True),
            matches_existing=e.get("matches_existing"),
            confidence=e.get("confidence", 1.0),
        ))
    
    state_changes = []
    for sc in raw.get("state_changes", []):
        state_changes.append(ExtractedStateChange(
            entity_name=sc.get("entity_name", ""),
            what_changed=sc.get("what_changed", ""),
            old_state=sc.get("from_state") or sc.get("old_state"),
            new_state=sc.get("to_state") or sc.get("new_state", ""),
            is_contradiction=sc.get("is_contradiction", False),
            confidence=sc.get("confidence", 1.0),
        ))
    
    relationships = []
    for r in raw.get("relationships", []):
        relationships.append(ExtractedRelationship(
            source=r.get("source", ""),
            target=r.get("target", ""),
            type=r.get("type", "related_to"),
            description=r.get("description", ""),
        ))
    
    return ExtractionResult(
        entities=entities,
        state_changes=state_changes,
        relationships=relationships,
        period_context=raw.get("period_context", raw.get("temporal_context", "")),
        summary=raw.get("summary", ""),
        significance=raw.get("significance", 0.0),
        user_state=raw.get("user_state"),
        conversation_ids=conversation_ids,
        tokens_used=tokens,
    )
