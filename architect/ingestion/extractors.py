"""LLM extraction prompts for the architect component knowledge graph.

We separate "I have a homepage + README" from "I just have a name." The
former goes through full extraction; the latter first goes through a
search-and-resolve step so we end up with the URLs we need.

All extractors emit JSON. We use OpenAI structured-output mode where
possible. We re-use mempol/llm.py because it's already battle-tested
(retries, json_mode, response caching, key loading).
"""
from __future__ import annotations
import json
import logging
from textwrap import dedent

from mempol import llm, config

logger = logging.getLogger(__name__)


# ─── 1. Resolve a name to its canonical homepage + GitHub URL ────────────────
RESOLVE_SYSTEM = dedent("""\
    You resolve an AI software component's name to its canonical URLs.
    Given just a NAME (e.g. "Stagehand", "Browserbase", "n8n"), return
    JSON with the most authoritative homepage_url, github_url (if any),
    and docs_url (if distinct from homepage). Use your knowledge of
    AI/agent tooling. If you are not confident a tool with that exact
    name exists, return all-empty strings — don't fabricate URLs.
""").strip()

RESOLVE_USER_TEMPLATE = dedent("""\
    Component name: {name}
    Optional context: {context}

    Return JSON:
    {{
      "canonical_name":   "...",
      "homepage_url":     "https://...",
      "github_url":       "https://github.com/org/repo",
      "docs_url":         "https://docs....",
      "type_guess":       "tool|library|api|mcp_server|model_api|framework|template|infra|sdk",
      "confidence":       0.0
    }}
""").strip()


def resolve_name(name: str, context: str = "") -> dict:
    """Use the LLM's prior to resolve a bare name to URLs + a type guess."""
    msgs = [
        {"role": "system", "content": RESOLVE_SYSTEM},
        {"role": "user", "content":
            RESOLVE_USER_TEMPLATE.format(name=name, context=context or "(none)")},
    ]
    raw = llm.chat(msgs, model=config.OBSERVER_MODEL or "gpt-4o-mini",
                    json_mode=True)
    try:
        return json.loads(raw)
    except Exception as e:
        logger.warning("resolve_name parse failed for %s: %s; raw=%r",
                       name, e, raw[:200])
        return {}


# ─── 2. Full structured extraction from a homepage + README ──────────────────
EXTRACT_SYSTEM = dedent("""\
    You produce the structured "card" for an AI software component, the
    way a senior staff engineer would describe it for a colleague who
    has 60 seconds to decide whether to use it.

    You will receive the component's name, its homepage HTML/text, and
    its GitHub README. Synthesise these into a JSON object with the
    fields below. Be specific. Do not write marketing copy. If a field
    cannot be determined from the inputs, return "" (empty string) or
    [] for that field — do not invent.

    Tone: terse, accurate, opinionated where evidence supports.
""").strip()

EXTRACT_USER_TEMPLATE = dedent("""\
    Component name:    {name}
    Homepage URL:      {homepage_url}
    GitHub URL:        {github_url}

    --- HOMEPAGE TEXT (first 8k chars) ---
    {homepage_text}

    --- GITHUB README (first 12k chars) ---
    {readme_text}

    Return JSON with the following shape:
    {{
      "slug":           "short-kebab-case-id",
      "canonical_name": "Display Name",
      "aliases":        ["alt-name-1", "alt-name-2"],
      "type":           "tool|library|api|mcp_server|model_api|framework|template|infra|sdk",
      "one_liner":      "≤140 chars; what it does in one sentence",
      "summary":        "2-3 sentence description for a search result card",
      "capability_long":"1-2 paragraph deeper description: when to reach for this, what problems it solves, what category it falls into",
      "homepage_url":   "...",
      "github_url":     "...",
      "docs_url":       "...",
      "mcp_url":        "",   // fill only if it ships an MCP server
      "pricing_model":  "free|freemium|paid|oss|usage_based",
      "hosted_or_self": "hosted|self_hosted|both",
      "license":        "MIT|Apache-2.0|proprietary|...",
      "tags":           ["browser-agent", "scraping", "memory", ...],   // 2-5 capability concepts
      "integrates_with":["component-name-1", "component-name-2"],       // products it explicitly works with
      "alternative_to": ["component-name-1"],                            // direct competitors
      "depends_on":     ["component-name-1"],                            // upstream deps
      "canonical_examples": [
        {{"description": "minimal usage", "code": "..."}},
        {{"description": "common pattern", "code": "..."}}
      ]
    }}

    Rules:
    - Tags should be drawn from a small vocabulary of capability concepts.
      Examples: browser-agent, web-scraping, memory-layer, vector-store,
      llm-orchestration, mcp-server, prompt-management, eval-framework,
      agent-framework, voice-synthesis, image-generation, code-execution,
      hosted-runtime, observability, fine-tuning. Invent new tags only when
      none of the obvious fit.
    - For relationships (integrates_with / alternative_to / depends_on)
      use the names as written on the homepage / README. Do not normalise.
    - canonical_examples should be 1-3 short code snippets pulled from the
      README's quickstart, not invented.
""").strip()


def extract_card(name: str, homepage_url: str, github_url: str,
                  homepage_text: str, readme_text: str) -> dict:
    msgs = [
        {"role": "system", "content": EXTRACT_SYSTEM},
        {"role": "user", "content": EXTRACT_USER_TEMPLATE.format(
            name=name,
            homepage_url=homepage_url or "(none)",
            github_url=github_url or "(none)",
            homepage_text=(homepage_text or "")[:8000],
            readme_text=(readme_text or "")[:12000],
        )},
    ]
    raw = llm.chat(msgs, model=config.OBSERVER_MODEL or "gpt-4o-mini",
                    json_mode=True)
    try:
        return json.loads(raw)
    except Exception as e:
        logger.warning("extract_card parse failed for %s: %s; raw=%r",
                       name, e, raw[:300])
        return {}


# ─── 3. Architecture-mining extractor: what does this repo do, what does it use?
ARCH_SYSTEM = dedent("""\
    You read a GitHub repository's README and other context (top-level
    file list, package.json or pyproject snippet) and produce a
    structured "architecture card" describing what the project does and
    which AI/dev tooling components it depends on.

    Be conservative: only list components that are explicitly imported
    or named in the inputs. Do not infer based on category similarity.
""").strip()

ARCH_USER_TEMPLATE = dedent("""\
    Repo:        {repo_url}
    Stars:       {stars}
    Description: {description}

    --- README (first 16k chars) ---
    {readme_text}

    --- Imports / dependencies (subset) ---
    {imports_text}

    Return JSON:
    {{
      "name":     "App / project name",
      "summary":  "1-2 sentence description of what this app does",
      "pattern":  "short architectural pattern label, e.g. 'browser-agent + slack-alerting + cron'",
      "components_used": [
        {{"name": "Stagehand",     "role": "browser_runtime",  "evidence": "import {{...}} from 'stagehand'"}},
        {{"name": "Browserbase",   "role": "headless_infra",   "evidence": "BROWSERBASE_API_KEY"}},
        {{"name": "OpenAI SDK",    "role": "llm_provider",     "evidence": "openai==1.x in requirements"}}
      ],
      "is_template_or_demo": false,    // true if this is a starter / template / demo, not a real app
      "quality_signal":      0.0       // a [0..1] qualitative score: production-shape > demo > stub
    }}

    Rules:
    - "components_used" must list ONLY items mentioned in the inputs. Use
      the canonical product name (e.g. "Browserbase", not "browserbase
      python sdk"). Include short evidence snippets (≤80 chars).
    - "pattern" should be 4-8 words, naming the architectural shape.
    - "quality_signal" should reflect whether this looks like real
      production code (high) vs. a tutorial / demo / hackathon project
      (low). It's qualitative.
""").strip()


def extract_architecture(repo_url: str, stars: int, description: str,
                          readme_text: str, imports_text: str = "") -> dict:
    msgs = [
        {"role": "system", "content": ARCH_SYSTEM},
        {"role": "user", "content": ARCH_USER_TEMPLATE.format(
            repo_url=repo_url,
            stars=stars,
            description=description or "",
            readme_text=(readme_text or "")[:16000],
            imports_text=(imports_text or "")[:4000],
        )},
    ]
    raw = llm.chat(msgs, model=config.OBSERVER_MODEL or "gpt-4o-mini",
                    json_mode=True)
    try:
        return json.loads(raw)
    except Exception as e:
        logger.warning("extract_architecture parse failed for %s: %s",
                       repo_url, e)
        return {}
