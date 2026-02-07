"""
Content Extraction — Extract structured elements from raw sales content using LLM.

Extracts: claims, proof points, pains addressed, objections handled, CTA, tone.
"""

from __future__ import annotations
import json
import logging
import os
import time
from typing import Optional

from openai import OpenAI

from sales.content_model import SalesContent, ContentType, ContentTone

logger = logging.getLogger("sales.content_extraction")

# Initialize OpenAI client
client = OpenAI(api_key=os.environ.get("OPENAI_API_KEY"))

EXTRACTION_PROMPT = """You are a sales content analyst. Extract structured information from the following sales content.

## Content Type
{content_type}

## Raw Content
{content}

## Instructions
Extract the following elements:

1. **Claims**: Value propositions, promises, or benefits stated in the content. Be specific.
2. **Proof Points**: Evidence supporting claims — case studies, statistics, customer quotes, social proof.
3. **Pains Addressed**: What customer pain points or challenges does this content speak to?
4. **Objections Handled**: What potential objections does this content preemptively address?
5. **CTA (Call to Action)**: What action does the content ask the reader to take?
6. **Tone**: One of: professional, casual, urgent, educational, consultative, challenger
7. **Subject Line**: If this is an email, extract or suggest a subject line.

## Output Format
Return valid JSON with this structure:
```json
{{
    "claims": ["claim 1", "claim 2", ...],
    "proof_points": ["proof point 1", "proof point 2", ...],
    "pains_addressed": ["pain 1", "pain 2", ...],
    "objections_handled": ["objection 1", "objection 2", ...],
    "cta": "the call to action",
    "tone": "professional|casual|urgent|educational|consultative|challenger",
    "subject_line": "suggested or extracted subject line"
}}
```

Be thorough but concise. Only include items that are clearly present or implied in the content."""


def extract_content_elements(
    content: SalesContent,
    model: str = "gpt-4o-mini",
) -> SalesContent:
    """
    Extract structured elements from a SalesContent object using LLM.
    
    Updates the content object in-place and returns it.
    """
    
    if not content.raw_content:
        logger.warning(f"No raw content to extract from: {content.id}")
        return content
    
    logger.info(f"Extracting elements from: {content.name or content.id}")
    
    prompt = EXTRACTION_PROMPT.format(
        content_type=content.type.value,
        content=content.raw_content,
    )
    
    try:
        response = client.chat.completions.create(
            model=model,
            messages=[
                {"role": "system", "content": "You are a precise sales content analyst. Return only valid JSON."},
                {"role": "user", "content": prompt},
            ],
            temperature=0.3,
            response_format={"type": "json_object"},
        )
        
        result_text = response.choices[0].message.content
        result = json.loads(result_text)
        
        # Update content object
        content.claims = result.get("claims", [])
        content.proof_points = result.get("proof_points", [])
        content.pains_addressed = result.get("pains_addressed", [])
        content.objections_handled = result.get("objections_handled", [])
        content.cta = result.get("cta", "")
        
        # Parse tone
        tone_str = result.get("tone", "professional").lower()
        try:
            content.tone = ContentTone(tone_str)
        except ValueError:
            content.tone = ContentTone.PROFESSIONAL
        
        # Subject line
        if result.get("subject_line"):
            content.subject_line = result["subject_line"]
        
        # Mark as extracted
        content.extracted = True
        content.extraction_model = model
        content.extraction_timestamp = time.time()
        content.updated_at = time.time()
        
        logger.info(
            f"Extracted from {content.name}: "
            f"{len(content.claims)} claims, "
            f"{len(content.proof_points)} proof points, "
            f"{len(content.pains_addressed)} pains, "
            f"{len(content.objections_handled)} objections"
        )
        
        return content
        
    except Exception as e:
        logger.error(f"Extraction failed for {content.id}: {e}")
        raise


def extract_from_raw_text(
    text: str,
    name: str = "",
    content_type: ContentType = ContentType.EMAIL,
    model: str = "gpt-4o-mini",
) -> SalesContent:
    """
    Create and extract a SalesContent object from raw text.
    
    Convenience function for the common case of uploading text and extracting immediately.
    """
    content = SalesContent(
        name=name or f"Content {time.strftime('%Y-%m-%d %H:%M')}",
        type=content_type,
        raw_content=text,
    )
    
    return extract_content_elements(content, model=model)


def batch_extract(
    contents: list[SalesContent],
    model: str = "gpt-4o-mini",
    skip_extracted: bool = True,
) -> list[SalesContent]:
    """
    Extract elements from multiple content objects.
    
    Args:
        contents: List of SalesContent objects
        model: LLM model to use
        skip_extracted: Skip content that's already been extracted
        
    Returns:
        List of updated SalesContent objects
    """
    results = []
    
    for i, content in enumerate(contents):
        if skip_extracted and content.extracted:
            logger.info(f"Skipping already extracted: {content.name}")
            results.append(content)
            continue
        
        try:
            extracted = extract_content_elements(content, model=model)
            results.append(extracted)
            
            # Rate limiting
            if i < len(contents) - 1:
                time.sleep(0.5)
                
        except Exception as e:
            logger.error(f"Failed to extract {content.id}: {e}")
            results.append(content)  # Return unextracted
    
    return results


# ─────────────────────────────────────────────────────────────────────────────
# Content Comparison Analysis
# ─────────────────────────────────────────────────────────────────────────────

COMPARISON_PROMPT = """Compare these two pieces of sales content and analyze their differences.

## Content A: {name_a}
Type: {type_a}
---
{content_a}
---

## Content B: {name_b}
Type: {type_b}
---
{content_b}
---

## Analysis
Provide a structured comparison:

1. **Approach**: How do the messaging approaches differ?
2. **Pain Focus**: Which pain points does each address?
3. **Tone**: Compare the tone and style
4. **Strengths**: What is each content piece best at?
5. **Best For**: What type of prospect is each best suited for?
6. **Risk Factors**: What could go wrong with each approach?

Return valid JSON:
```json
{{
    "approach_comparison": "...",
    "pain_focus": {{
        "content_a": ["pain1", "pain2"],
        "content_b": ["pain1", "pain2"],
        "overlap": ["shared pains"]
    }},
    "tone_comparison": "...",
    "strengths": {{
        "content_a": ["strength1", "strength2"],
        "content_b": ["strength1", "strength2"]
    }},
    "best_for": {{
        "content_a": "persona/situation description",
        "content_b": "persona/situation description"
    }},
    "risk_factors": {{
        "content_a": ["risk1"],
        "content_b": ["risk1"]
    }}
}}
```"""


def compare_content(
    content_a: SalesContent,
    content_b: SalesContent,
    model: str = "gpt-4o-mini",
) -> dict:
    """
    Compare two pieces of content and analyze their differences.
    
    Useful for A/B testing decisions and sequence planning.
    """
    
    prompt = COMPARISON_PROMPT.format(
        name_a=content_a.name,
        type_a=content_a.type.value,
        content_a=content_a.raw_content,
        name_b=content_b.name,
        type_b=content_b.type.value,
        content_b=content_b.raw_content,
    )
    
    try:
        response = client.chat.completions.create(
            model=model,
            messages=[
                {"role": "system", "content": "You are a sales messaging expert. Return only valid JSON."},
                {"role": "user", "content": prompt},
            ],
            temperature=0.3,
            response_format={"type": "json_object"},
        )
        
        return json.loads(response.choices[0].message.content)
        
    except Exception as e:
        logger.error(f"Comparison failed: {e}")
        return {"error": str(e)}
