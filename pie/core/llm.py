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


def _recover_partial_json(content: str) -> str:
    """Attempt to recover a truncated JSON string by closing open structures.

    When finish_reason == "length", the JSON was cut mid-stream. We try:
    1. Direct parse (maybe it's coincidentally complete)
    2. Truncate at the last complete top-level array element (last `},`)
    3. Close open braces/brackets to make valid JSON with what we have
    """
    if not content:
        return "{}"
    try:
        json.loads(content)
        return content  # Already valid
    except json.JSONDecodeError:
        pass

    # Strategy: find last well-formed item boundary inside a JSON array
    # Most PIE extractions are {"entities": [...], ...} — try to close the array
    last_complete = content.rfind("},")
    if last_complete > 0:
        truncated = content[: last_complete + 1]
        # Count open brackets/braces and close them
        depth_brace = truncated.count("{") - truncated.count("}")
        depth_bracket = truncated.count("[") - truncated.count("]")
        closed = truncated
        closed += "]" * max(0, depth_bracket)
        closed += "}" * max(0, depth_brace)
        try:
            json.loads(closed)
            logger.warning("Partial JSON recovery succeeded (truncated at last complete item)")
            return closed
        except json.JSONDecodeError:
            pass

    # Last resort: return minimal valid skeleton
    logger.warning("Partial JSON recovery failed — returning empty extraction skeleton")
    return '{"entities": [], "state_changes": [], "relationships": [], "summary": "extraction truncated"}'


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
    
    # Reasoning/thinking models that don't support temperature
    NO_TEMP_MODELS = {"gpt-5-mini", "gpt-5-nano", "o1", "o1-mini", "o3", "o3-mini"}

    @staticmethod
    def _uses_completion_tokens(model: str) -> bool:
        """gpt-5.x and o-series models require max_completion_tokens, not max_tokens."""
        base = model.split(":")[0].lower()
        return base.startswith("gpt-5") or base.startswith("o1") or base.startswith("o3")

    def chat(
        self,
        messages: list[dict],
        model: str = "gpt-5.4",
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
        # Strip version suffix (e.g. "-2025-08-07") for set lookup
        model_base = model.split("-2025")[0] if "-2025" in model else model
        is_reasoning = model_base in self.NO_TEMP_MODELS
        if not is_reasoning:
            kwargs["temperature"] = temperature
        else:
            kwargs["reasoning_effort"] = "low"
        if json_mode:
            kwargs["response_format"] = {"type": "json_object"}
            # OpenAI rejects response_format=json_object unless the literal
            # word 'json' appears in at least one message. Some PIE prompts
            # describe schemas with braces but not the word. Inject it.
            try:
                msgs_text = " ".join(m.get("content", "") for m in messages
                                      if isinstance(m, dict) and isinstance(m.get("content"), str))
            except Exception:
                msgs_text = ""
            if "json" not in msgs_text.lower():
                kwargs["messages"] = (
                    [{"role": "system", "content": "Respond strictly as a JSON object."}]
                    + list(messages)
                )
        if max_tokens:
            # gpt-5.x and o-series require max_completion_tokens
            if self._uses_completion_tokens(model):
                kwargs["max_completion_tokens"] = max_tokens
            else:
                kwargs["max_tokens"] = max_tokens
        
        for attempt in range(MAX_RETRIES):
            try:
                response = self.client.chat.completions.create(**kwargs)
                self._total_tokens += response.usage.total_tokens
                self._total_calls += 1
                
                content = response.choices[0].message.content
                finish_reason = response.choices[0].finish_reason
                
                # Reasoning models may return None/empty content on some calls
                if json_mode and (content is None or content.strip() == ""):
                    logger.warning(f"Empty content from {model} (attempt {attempt+1}), retrying...")
                    if attempt < MAX_RETRIES - 1:
                        time.sleep(RETRY_DELAY * (attempt + 1))
                        continue
                    raise RuntimeError(f"Model {model} returned empty content after {MAX_RETRIES} attempts")

                # If output was truncated mid-JSON, try partial recovery before retrying
                if json_mode and finish_reason == "length":
                    logger.warning(f"Response truncated (finish_reason=length) on attempt {attempt+1}, trying partial JSON recovery")
                    content = _recover_partial_json(content)

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
        name = (e.get("name") or "").strip()
        if not name:
            continue  # Skip entities with empty/missing names

        # Validate entity type
        etype = e.get("type", "concept").lower()
        valid_types = {"person", "project", "tool", "organization", "belief", "decision", "concept", "period", "event", "goal"}
        if etype not in valid_types:
            etype = "concept"  # default fallback

        entities.append(ExtractedEntity(
            name=name,
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
