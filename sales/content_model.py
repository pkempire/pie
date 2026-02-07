"""
Content Model — Represents sales content that can be simulated against prospect world models.

Content types: email sequences, call scripts, pitch decks, proposals, case studies.
Each piece of content is extracted for: claims, proof points, pains addressed, tone, CTA.
"""

from __future__ import annotations
import json
import logging
import os
import sys
import uuid
import time
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Any, Literal, Optional

logger = logging.getLogger("sales.content_model")


class ContentType(str, Enum):
    EMAIL = "email"
    CALL_SCRIPT = "call_script"
    PITCH_DECK = "pitch_deck"
    PROPOSAL = "proposal"
    CASE_STUDY = "case_study"
    FOLLOW_UP = "follow_up"
    OBJECTION_HANDLER = "objection_handler"


class ContentTone(str, Enum):
    PROFESSIONAL = "professional"
    CASUAL = "casual"
    URGENT = "urgent"
    EDUCATIONAL = "educational"
    CONSULTATIVE = "consultative"
    CHALLENGER = "challenger"


@dataclass
class SalesContent:
    """A piece of sales content that can be simulated against prospect world models."""
    
    id: str = field(default_factory=lambda: str(uuid.uuid4()))
    name: str = ""
    type: ContentType = ContentType.EMAIL
    raw_content: str = ""  # Original text
    
    # Extracted elements (filled by LLM extraction)
    claims: list[str] = field(default_factory=list)  # Value props, promises
    proof_points: list[str] = field(default_factory=list)  # Case studies, stats, social proof
    pains_addressed: list[str] = field(default_factory=list)  # Which pain points this addresses
    objections_handled: list[str] = field(default_factory=list)  # Which objections this preempts
    cta: str = ""  # Call to action
    tone: ContentTone = ContentTone.PROFESSIONAL
    
    # Metadata
    subject_line: str = ""  # For emails
    word_count: int = 0
    reading_time_seconds: int = 0
    
    # Timestamps
    created_at: float = field(default_factory=time.time)
    updated_at: float = field(default_factory=time.time)
    
    # Extraction metadata
    extracted: bool = False
    extraction_model: str = ""
    extraction_timestamp: float = 0
    
    def __post_init__(self):
        if self.raw_content:
            self.word_count = len(self.raw_content.split())
            self.reading_time_seconds = int(self.word_count / 200 * 60)  # ~200 wpm
    
    def to_dict(self) -> dict:
        return {
            "id": self.id,
            "name": self.name,
            "type": self.type.value if isinstance(self.type, ContentType) else self.type,
            "raw_content": self.raw_content,
            "claims": self.claims,
            "proof_points": self.proof_points,
            "pains_addressed": self.pains_addressed,
            "objections_handled": self.objections_handled,
            "cta": self.cta,
            "tone": self.tone.value if isinstance(self.tone, ContentTone) else self.tone,
            "subject_line": self.subject_line,
            "word_count": self.word_count,
            "reading_time_seconds": self.reading_time_seconds,
            "created_at": self.created_at,
            "updated_at": self.updated_at,
            "extracted": self.extracted,
            "extraction_model": self.extraction_model,
            "extraction_timestamp": self.extraction_timestamp,
        }
    
    @classmethod
    def from_dict(cls, data: dict) -> "SalesContent":
        content = cls()
        content.id = data.get("id", str(uuid.uuid4()))
        content.name = data.get("name", "")
        content.type = ContentType(data.get("type", "email"))
        content.raw_content = data.get("raw_content", "")
        content.claims = data.get("claims", [])
        content.proof_points = data.get("proof_points", [])
        content.pains_addressed = data.get("pains_addressed", [])
        content.objections_handled = data.get("objections_handled", [])
        content.cta = data.get("cta", "")
        content.tone = ContentTone(data.get("tone", "professional"))
        content.subject_line = data.get("subject_line", "")
        content.word_count = data.get("word_count", 0)
        content.reading_time_seconds = data.get("reading_time_seconds", 0)
        content.created_at = data.get("created_at", time.time())
        content.updated_at = data.get("updated_at", time.time())
        content.extracted = data.get("extracted", False)
        content.extraction_model = data.get("extraction_model", "")
        content.extraction_timestamp = data.get("extraction_timestamp", 0)
        return content


@dataclass
class ContentSequence:
    """A sequence of content pieces (e.g., nurture campaign, follow-up sequence)."""
    
    id: str = field(default_factory=lambda: str(uuid.uuid4()))
    name: str = ""
    description: str = ""
    
    # Sequence items
    items: list[SalesContent] = field(default_factory=list)
    
    # Timing between items (in days) — items[i] sent timing[i] days after previous
    # timing[0] is ignored (first item sent immediately)
    timing_days: list[int] = field(default_factory=list)
    
    # Target persona
    target_persona: str = ""  # e.g., "technical buyer", "executive sponsor"
    target_stage: str = ""  # e.g., "discovery", "evaluation"
    
    # Metadata
    created_at: float = field(default_factory=time.time)
    updated_at: float = field(default_factory=time.time)
    
    @property
    def total_duration_days(self) -> int:
        """Total duration of sequence in days."""
        return sum(self.timing_days) if self.timing_days else 0
    
    @property
    def item_count(self) -> int:
        return len(self.items)
    
    def add_item(self, content: SalesContent, days_after_previous: int = 3):
        """Add a content item to the sequence."""
        self.items.append(content)
        self.timing_days.append(days_after_previous)
        self.updated_at = time.time()
    
    def to_dict(self) -> dict:
        return {
            "id": self.id,
            "name": self.name,
            "description": self.description,
            "items": [item.to_dict() for item in self.items],
            "timing_days": self.timing_days,
            "target_persona": self.target_persona,
            "target_stage": self.target_stage,
            "total_duration_days": self.total_duration_days,
            "item_count": self.item_count,
            "created_at": self.created_at,
            "updated_at": self.updated_at,
        }
    
    @classmethod
    def from_dict(cls, data: dict) -> "ContentSequence":
        seq = cls()
        seq.id = data.get("id", str(uuid.uuid4()))
        seq.name = data.get("name", "")
        seq.description = data.get("description", "")
        seq.items = [SalesContent.from_dict(item) for item in data.get("items", [])]
        seq.timing_days = data.get("timing_days", [])
        seq.target_persona = data.get("target_persona", "")
        seq.target_stage = data.get("target_stage", "")
        seq.created_at = data.get("created_at", time.time())
        seq.updated_at = data.get("updated_at", time.time())
        return seq


class ContentLibrary:
    """
    Manages a library of sales content and sequences.
    
    Persists to JSON files in the output directory.
    """
    
    def __init__(self, output_dir: str | Path = "sales/output"):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        self.content_file = self.output_dir / "content_library.json"
        self.sequences_file = self.output_dir / "content_sequences.json"
        
        self.content: dict[str, SalesContent] = {}  # id -> content
        self.sequences: dict[str, ContentSequence] = {}  # id -> sequence
        
        self._load()
    
    def _load(self):
        """Load library from disk."""
        if self.content_file.exists():
            try:
                data = json.loads(self.content_file.read_text())
                for item in data.get("content", []):
                    content = SalesContent.from_dict(item)
                    self.content[content.id] = content
                logger.info(f"Loaded {len(self.content)} content items")
            except Exception as e:
                logger.error(f"Error loading content library: {e}")
        
        if self.sequences_file.exists():
            try:
                data = json.loads(self.sequences_file.read_text())
                for item in data.get("sequences", []):
                    seq = ContentSequence.from_dict(item)
                    self.sequences[seq.id] = seq
                logger.info(f"Loaded {len(self.sequences)} sequences")
            except Exception as e:
                logger.error(f"Error loading sequences: {e}")
    
    def _save(self):
        """Persist library to disk."""
        # Save content
        content_data = {
            "content": [c.to_dict() for c in self.content.values()],
            "updated_at": time.time(),
        }
        self.content_file.write_text(json.dumps(content_data, indent=2))
        
        # Save sequences
        seq_data = {
            "sequences": [s.to_dict() for s in self.sequences.values()],
            "updated_at": time.time(),
        }
        self.sequences_file.write_text(json.dumps(seq_data, indent=2))
    
    def add_content(self, content: SalesContent) -> SalesContent:
        """Add content to library."""
        self.content[content.id] = content
        self._save()
        logger.info(f"Added content: {content.name} ({content.type.value})")
        return content
    
    def get_content(self, content_id: str) -> Optional[SalesContent]:
        """Get content by ID."""
        return self.content.get(content_id)
    
    def list_content(self, content_type: Optional[ContentType] = None) -> list[SalesContent]:
        """List all content, optionally filtered by type."""
        items = list(self.content.values())
        if content_type:
            items = [c for c in items if c.type == content_type]
        return sorted(items, key=lambda c: c.created_at, reverse=True)
    
    def delete_content(self, content_id: str) -> bool:
        """Delete content by ID."""
        if content_id in self.content:
            del self.content[content_id]
            self._save()
            return True
        return False
    
    def add_sequence(self, sequence: ContentSequence) -> ContentSequence:
        """Add sequence to library."""
        self.sequences[sequence.id] = sequence
        self._save()
        logger.info(f"Added sequence: {sequence.name} ({sequence.item_count} items)")
        return sequence
    
    def get_sequence(self, sequence_id: str) -> Optional[ContentSequence]:
        """Get sequence by ID."""
        return self.sequences.get(sequence_id)
    
    def list_sequences(self) -> list[ContentSequence]:
        """List all sequences."""
        return sorted(self.sequences.values(), key=lambda s: s.created_at, reverse=True)
    
    def delete_sequence(self, sequence_id: str) -> bool:
        """Delete sequence by ID."""
        if sequence_id in self.sequences:
            del self.sequences[sequence_id]
            self._save()
            return True
        return False


# Demo content for testing
def get_demo_content() -> list[SalesContent]:
    """Generate demo content for testing."""
    
    return [
        SalesContent(
            id="demo-email-1",
            name="Initial Outreach — Pain-Focused",
            type=ContentType.EMAIL,
            subject_line="Struggling with [Pain Point]? You're not alone.",
            raw_content="""Hi {{first_name}},

I noticed that {{company}} has been growing rapidly — congrats! With that growth, I imagine you're dealing with the challenges that come with scaling: disconnected systems, manual processes, and data that lives in silos.

We've helped companies like Acme Corp and TechStart reduce their operational overhead by 40% while actually improving visibility across teams.

Would you be open to a 15-minute call to see if we might be able to help {{company}} too?

Best,
{{sender_name}}""",
            claims=[
                "Reduce operational overhead by 40%",
                "Improve visibility across teams",
            ],
            proof_points=[
                "Helped Acme Corp achieve results",
                "Helped TechStart achieve results",
            ],
            pains_addressed=[
                "Disconnected systems",
                "Manual processes",
                "Data silos",
                "Scaling challenges",
            ],
            objections_handled=[],
            cta="15-minute discovery call",
            tone=ContentTone.CONSULTATIVE,
            extracted=True,
        ),
        SalesContent(
            id="demo-email-2",
            name="Follow-up — ROI Focused",
            type=ContentType.EMAIL,
            subject_line="Quick question about {{company}}'s priorities",
            raw_content="""Hi {{first_name}},

I wanted to follow up on my earlier note. I've been doing some research on {{company}} and it looks like you're in a critical growth phase.

Companies at your stage typically face a choice: keep adding headcount to manage complexity, or invest in systems that scale without linear cost increases.

Our customers typically see:
- 3x faster time-to-insight on key metrics
- 60% reduction in manual data reconciliation
- Full ROI within 6 months

Is improving operational efficiency on your radar for this quarter?

Best,
{{sender_name}}""",
            claims=[
                "3x faster time-to-insight",
                "60% reduction in manual data reconciliation",
                "Full ROI within 6 months",
            ],
            proof_points=[
                "Based on customer outcomes",
            ],
            pains_addressed=[
                "Rising headcount costs",
                "Complexity management",
                "Manual data reconciliation",
            ],
            objections_handled=[
                "ROI uncertainty",
                "Time to value concerns",
            ],
            cta="Discuss operational efficiency priorities",
            tone=ContentTone.PROFESSIONAL,
            extracted=True,
        ),
        SalesContent(
            id="demo-email-3",
            name="Objection Handler — Budget",
            type=ContentType.OBJECTION_HANDLER,
            raw_content="""Hi {{first_name}},

I completely understand budget is tight — that's true for most teams right now.

A few thoughts:

1. **Cost of inaction**: What's it costing {{company}} today to operate without a solution? Our customers were spending 20+ hours/week on manual work before switching.

2. **Phased approach**: We can start with a pilot focused on one high-impact area. Prove the value, then expand.

3. **Flexible terms**: We offer quarterly billing for teams who need to manage cash flow carefully.

Would it help to see a quick ROI calculator based on {{company}}'s situation?

Best,
{{sender_name}}""",
            claims=[
                "Customers were spending 20+ hours/week before switching",
                "Pilot approach available",
                "Quarterly billing available",
            ],
            proof_points=[
                "Customer time savings data",
            ],
            pains_addressed=[],
            objections_handled=[
                "Budget constraints",
                "Cash flow concerns",
                "Risk of large commitment",
            ],
            cta="ROI calculator walkthrough",
            tone=ContentTone.CONSULTATIVE,
            extracted=True,
        ),
        SalesContent(
            id="demo-case-study-1",
            name="Case Study — TechStart",
            type=ContentType.CASE_STUDY,
            raw_content="""# TechStart: From Chaos to Clarity in 90 Days

## The Challenge
TechStart, a Series B startup with 150 employees, was drowning in spreadsheets. Their sales, marketing, and customer success teams all had different versions of the truth. Leadership couldn't get accurate pipeline data without a 2-day manual reconciliation process.

## The Solution
We implemented our platform in phases:
- Week 1-2: Core integrations (Salesforce, HubSpot, Stripe)
- Week 3-4: Custom dashboards for each team lead
- Week 5-8: Automated alerts and anomaly detection
- Week 9-12: Advanced forecasting and scenario planning

## The Results
- **Time to insight**: Reduced from 2 days to 15 minutes
- **Data accuracy**: Improved from ~70% to 99%+
- **Forecast accuracy**: Improved by 35%
- **Hours saved**: 25 hours/week across leadership team

## Quote
"We went from flying blind to having a cockpit view of the entire business. I can't imagine going back." — Sarah Chen, CEO, TechStart""",
            claims=[
                "Implementation in 90 days",
                "Time to insight: 2 days to 15 minutes",
                "Data accuracy: 70% to 99%+",
                "Forecast accuracy improved 35%",
                "25 hours/week saved",
            ],
            proof_points=[
                "TechStart case study",
                "Sarah Chen CEO testimonial",
                "Series B startup, 150 employees",
            ],
            pains_addressed=[
                "Spreadsheet chaos",
                "Multiple versions of truth",
                "Manual reconciliation",
                "Inaccurate pipeline data",
            ],
            objections_handled=[
                "Implementation complexity",
                "Time to value",
            ],
            cta="See similar results",
            tone=ContentTone.PROFESSIONAL,
            extracted=True,
        ),
    ]


def get_demo_sequences() -> list[ContentSequence]:
    """Generate demo sequences for testing."""
    
    demo_content = get_demo_content()
    
    # Pain-focused nurture sequence
    pain_sequence = ContentSequence(
        id="demo-seq-pain",
        name="Pain-Focused Nurture",
        description="3-touch sequence focused on pain points and challenges",
        target_persona="Operations leader",
        target_stage="discovery",
    )
    pain_sequence.add_item(demo_content[0], days_after_previous=0)  # Initial outreach
    pain_sequence.add_item(demo_content[1], days_after_previous=3)  # ROI follow-up
    pain_sequence.add_item(demo_content[3], days_after_previous=5)  # Case study
    
    # ROI-focused sequence
    roi_sequence = ContentSequence(
        id="demo-seq-roi",
        name="ROI-Focused Sequence",
        description="2-touch sequence emphasizing ROI and cost savings",
        target_persona="Finance / budget holder",
        target_stage="qualification",
    )
    roi_sequence.add_item(demo_content[1], days_after_previous=0)  # ROI email
    roi_sequence.add_item(demo_content[2], days_after_previous=4)  # Budget objection handler
    
    return [pain_sequence, roi_sequence]
