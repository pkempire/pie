#!/usr/bin/env python3
"""
Import Box sales data into Prism demo.

Reads:
- Sep 1 to Dec 11 Outreach Stats.xls → sequence performance data
- Sales Play Example *.boxnote → email templates
- Prompt for Sales Outreach Agent.boxnote → system context

Creates realistic prospects based on Box's public sector & enterprise focus.
"""

import json
import pandas as pd
import re
from pathlib import Path
from datetime import datetime, timedelta
import random

DATA_DIR = Path.home() / "Downloads" / "Data Validation Platform"
OUTPUT_DIR = Path(__file__).parent / "data"

# Box's target verticals (from the sales plays)
VERTICALS = [
    "Federal Government",
    "State Government", 
    "Local Government",
    "Healthcare",
    "Financial Services",
    "Manufacturing",
    "Legal",
    "Education",
]

# Pain points from Box messaging
BOX_PAIN_POINTS = [
    "Legacy ECM systems (OpenText, Hyland) are expensive to maintain",
    "Content scattered across SharePoint, shared drives, point solutions",
    "No visibility into sensitive content across the organization",
    "Manual classification and compliance processes",
    "Security alerts overwhelming the team",
    "Ransomware and malware threats increasing",
    "Need FedRAMP High / DoD IL4 compliance",
    "FOIA/PRA request response times too slow",
    "External collaboration with vendors is risky",
    "AI adoption blocked by security concerns",
]

# Objections from the outreach data
BOX_OBJECTIONS = [
    "Already using Microsoft 365 / SharePoint",
    "Budget constraints this fiscal year",
    "Currently under contract with competitor",
    "Need to involve procurement",
    "Timing - other priorities right now",
    "Security review required",
    "Need executive sponsorship",
]

# Box value props
BOX_VALUE_PROPS = [
    "Single secure content cloud",
    "FedRAMP High authorized",
    "Box AI for intelligent content",
    "Box Shield Pro for threat detection",
    "Unlimited e-signatures included",
    "Replace legacy ECM at lower TCO",
]


def extract_boxnote_text(filepath: Path) -> str:
    """Extract text content from a .boxnote file."""
    try:
        with open(filepath, 'r') as f:
            data = json.load(f)
        
        def walk(node):
            text = ''
            if isinstance(node, dict):
                if node.get('type') == 'text':
                    text += node.get('text', '')
                if node.get('type') in ('paragraph', 'heading'):
                    text += '\n'
                for child in node.get('content', []):
                    text += walk(child)
            elif isinstance(node, list):
                for item in node:
                    text += walk(item)
            return text
        
        return walk(data.get('doc', {}))
    except:
        return ""


def parse_outreach_stats() -> pd.DataFrame:
    """Load and parse the outreach stats Excel file."""
    xlsx_path = DATA_DIR / "Sep 1 to Dec 11 Outreach Stats.xls"
    if not xlsx_path.exists():
        print(f"Warning: {xlsx_path} not found")
        return pd.DataFrame()
    
    df = pd.read_excel(xlsx_path, engine='xlrd')
    return df


def extract_email_templates(boxnote_path: Path) -> list[dict]:
    """Extract email templates from a sales play boxnote."""
    content = extract_boxnote_text(boxnote_path)
    templates = []
    
    # Find email sections
    email_pattern = r'(\d+(?:st|nd|rd|th) TOUCH.*?)(?=\d+(?:st|nd|rd|th) TOUCH|$)'
    matches = re.findall(email_pattern, content, re.DOTALL | re.IGNORECASE)
    
    for i, match in enumerate(matches[:3]):  # Max 3 touches
        # Extract subject line
        subject_match = re.search(r'Subject line\(?s?\)?[:\s]*["""]([^"""]+)["""]', match)
        subject = subject_match.group(1) if subject_match else f"Touch {i+1}"
        
        # Extract body (everything after subject until next section)
        body_match = re.search(r'Hi \{\{.*?\}\}[,\s]*(.*?)(?:Best,|$)', match, re.DOTALL)
        body = body_match.group(1).strip() if body_match else match[:500]
        
        templates.append({
            "touch": i + 1,
            "subject": subject,
            "body": body[:1000],
            "type": "email",
        })
    
    return templates


def generate_box_prospects() -> list[dict]:
    """Generate realistic Box prospects based on their target market."""
    
    # Real-ish companies in Box's verticals
    prospects = [
        {
            "id": "box-001",
            "name": "Sarah Chen",
            "title": "Director of IT Modernization",
            "company": "California Department of Transportation",
            "vertical": "State Government",
            "stage": "evaluation",
            "deal_value": 450000,
            "pain_points": [
                "Legacy FileNet system costs $800K/year to maintain",
                "Content scattered across 47 regional offices",
                "FOIA response times averaging 45 days",
            ],
            "objections": [
                "Need FedRAMP authorization verification",
                "Current FileNet contract runs through June",
            ],
            "stakeholders": [
                {"name": "Sarah Chen", "role": "Champion", "title": "Director of IT Modernization"},
                {"name": "Michael Torres", "role": "Economic Buyer", "title": "Deputy CIO"},
                {"name": "Jennifer Walsh", "role": "Technical Evaluator", "title": "Enterprise Architect"},
            ],
            "next_step": "Security architecture review with Jennifer",
            "created_at": (datetime.now() - timedelta(days=45)).isoformat(),
        },
        {
            "id": "box-002", 
            "name": "Marcus Williams",
            "title": "Chief Information Security Officer",
            "company": "First National Bank",
            "vertical": "Financial Services",
            "stage": "proposal",
            "deal_value": 780000,
            "pain_points": [
                "1.3M security alerts per month from current tools",
                "Manual PII classification taking 200 hours/month",
                "Failed last OCC audit on document retention",
            ],
            "objections": [
                "Board approval required for cloud vendors",
                "Need proof of SOC 2 Type II compliance",
            ],
            "stakeholders": [
                {"name": "Marcus Williams", "role": "Champion", "title": "CISO"},
                {"name": "David Park", "role": "Economic Buyer", "title": "CFO"},
                {"name": "Amanda Foster", "role": "Blocker", "title": "Chief Risk Officer"},
            ],
            "next_step": "Present to Risk Committee next Tuesday",
            "created_at": (datetime.now() - timedelta(days=30)).isoformat(),
        },
        {
            "id": "box-003",
            "name": "Dr. Patricia Okonkwo",
            "title": "VP of Research Administration",
            "company": "Johns Hopkins Applied Physics Lab",
            "vertical": "Education",
            "stage": "discovery",
            "deal_value": 320000,
            "pain_points": [
                "ITAR/CUI content mixed with unclassified research",
                "External collaboration with contractors blocked",
                "Researchers using unauthorized cloud tools",
            ],
            "objections": [
                "Need DoD IL4 authorization",
                "Academic procurement process is 6+ months",
            ],
            "stakeholders": [
                {"name": "Dr. Patricia Okonkwo", "role": "Champion", "title": "VP Research Admin"},
                {"name": "Col. James Mitchell (Ret.)", "role": "Technical Evaluator", "title": "Security Director"},
            ],
            "next_step": "Demo with research team leads",
            "created_at": (datetime.now() - timedelta(days=12)).isoformat(),
        },
        {
            "id": "box-004",
            "name": "Robert Kim",
            "title": "Deputy Director of Enforcement",
            "company": "OSHA Region 5",
            "vertical": "Federal Government",
            "stage": "negotiation",
            "deal_value": 195000,
            "pain_points": [
                "Case files stored on legacy Documentum system",
                "Field inspectors can't access files remotely",
                "Evidence chain-of-custody documentation gaps",
            ],
            "objections": [
                "Need CJIS-equivalent security controls",
                "Union approval for new tools required",
            ],
            "stakeholders": [
                {"name": "Robert Kim", "role": "Champion", "title": "Deputy Director"},
                {"name": "Lisa Thompson", "role": "Economic Buyer", "title": "Regional Administrator"},
            ],
            "next_step": "Final pricing review with contracting",
            "created_at": (datetime.now() - timedelta(days=60)).isoformat(),
        },
        {
            "id": "box-005",
            "name": "Jennifer Martinez",
            "title": "City Manager",
            "company": "City of Austin",
            "vertical": "Local Government",
            "stage": "lead",
            "deal_value": 125000,
            "pain_points": [
                "Public records requests up 300% post-pandemic",
                "No central repository for city documents",
                "Council meeting recordings hard to search",
            ],
            "objections": [],
            "stakeholders": [
                {"name": "Jennifer Martinez", "role": "Champion", "title": "City Manager"},
            ],
            "next_step": "Initial discovery call scheduled",
            "created_at": (datetime.now() - timedelta(days=3)).isoformat(),
        },
    ]
    
    return prospects


def generate_box_content() -> list[dict]:
    """Generate content items from Box sales plays."""
    content = []
    
    # ECM Replacement Play
    content.append({
        "id": "content-001",
        "name": "ECM Modernization - First Touch",
        "type": "email",
        "subject_line": "Rethinking ECM for the AI era",
        "raw_content": """Hi {{first_name}},

IT teams overspend by a whopping ~30% to maintain outdated ECM systems – like OpenText and Hyland OnBase – that store content, but can't truly activate it.

Instead of adding single-purpose tools to fill those gaps, Box helps teams unlock more value from their content with:

• Intelligent search → built-in AI-powered search supported across all documents and data
• Built-in workflow automation → extract key information, route approvals, trigger workflows  
• Integrated security & compliance → advanced threat detection and auto-classification
• Unlimited e-signatures → unlimited signature requests at no additional cost within Box

All this functionality happens in one place, via one platform – not only lowering tool costs, but also minimizing friction users face day-to-day.

Would you be open to exploring whether {{company}} could achieve similar gains with Box?

Best,
{{sender.first_name}}""",
        "claims": [
            "30% cost savings vs legacy ECM",
            "AI-powered search across all content",
            "Built-in workflow automation",
            "Unlimited e-signatures included",
        ],
        "pains_addressed": [
            "Legacy ECM maintenance costs",
            "Content activation gaps",
            "Tool sprawl and friction",
        ],
        "tone": "professional",
    })
    
    content.append({
        "id": "content-002",
        "name": "Shield Pro - Security Play",
        "type": "email",
        "subject_line": "Staying ahead of content growth",
        "raw_content": """Hi {{first_name}},

{{company}}'s content volume has grown significantly – and as content generation increases, visibility and control become harder to maintain.

That's why we've built Box Shield Pro – strengthening protection with:

• Adaptive AI Classification – auto-classify sensitive data as it's created
• Intelligent Threat Analysis – precise, high-impact insights with actionable remediation
• Advanced Ransomware Protection – detect and stop attacks before encryption spreads

These enhancements extend your existing security foundation without adding new friction.

Would you be open to a quick walkthrough this week or next?

Best,
{{sender.first_name}}""",
        "claims": [
            "AI-powered auto-classification",
            "Real-time ransomware detection",
            "Actionable threat remediation",
        ],
        "pains_addressed": [
            "Content visibility gaps",
            "Security alert overload",
            "Ransomware threats",
        ],
        "tone": "professional",
    })
    
    content.append({
        "id": "content-003",
        "name": "Public Sector - CMMC 2.0 Play",
        "type": "email", 
        "subject_line": "CMMC 2.0 readiness",
        "raw_content": """Hi {{first_name}},

With CMMC 2.0 requirements taking effect, defense contractors need to demonstrate compliance across their entire content ecosystem.

Box is positioned to help {{company}} meet these requirements:

• FedRAMP High authorized – highest level of federal security certification
• DoD IL4 compliant – cleared for CUI handling
• NIST 800-171 aligned – built-in controls for CMMC requirements
• Audit-ready logging – complete chain of custody documentation

Many defense contractors are consolidating onto Box to simplify their compliance posture.

Would a brief call to discuss your CMMC timeline make sense?

Best,
{{sender.first_name}}""",
        "claims": [
            "FedRAMP High authorized",
            "DoD IL4 compliant",
            "NIST 800-171 aligned",
        ],
        "pains_addressed": [
            "CMMC 2.0 compliance requirements",
            "CUI handling gaps",
            "Audit documentation",
        ],
        "tone": "professional",
    })
    
    return content


def generate_sequences() -> list[dict]:
    """Generate content sequences from Box plays."""
    return [
        {
            "id": "seq-001",
            "name": "ECM Modernization - 3 Touch",
            "description": "Enterprise Content Management replacement sequence targeting IT leaders",
            "target_stage": "discovery",
            "target_persona": "IT Director / CIO",
            "items": [
                {"content_id": "content-001", "day_offset": 0},
                {"content_id": "content-002", "day_offset": 4},
                {"content_id": "content-003", "day_offset": 9},
            ],
        },
        {
            "id": "seq-002", 
            "name": "Shield Pro - Security Leaders",
            "description": "Security-focused sequence for CISOs and security directors",
            "target_stage": "lead",
            "target_persona": "CISO / Security Director",
            "items": [
                {"content_id": "content-002", "day_offset": 0},
                {"content_id": "content-001", "day_offset": 3},
            ],
        },
    ]


def save_demo_data():
    """Save all generated demo data."""
    OUTPUT_DIR.mkdir(exist_ok=True)
    
    # Load outreach stats for performance data
    stats_df = parse_outreach_stats()
    if not stats_df.empty:
        # Get top performing sequences
        top_sequences = stats_df.nlargest(20, 'Meetings booked')[
            ['Sequence name', 'Owner name', 'Meetings booked', 'Open rate', 'Reply rate', 'Prospects invited']
        ].to_dict('records')
        
        with open(OUTPUT_DIR / 'outreach_performance.json', 'w') as f:
            json.dump(top_sequences, f, indent=2, default=str)
        print(f"✓ Saved {len(top_sequences)} top sequences to outreach_performance.json")
    
    # Save prospects
    prospects = generate_box_prospects()
    with open(OUTPUT_DIR / 'box_prospects.json', 'w') as f:
        json.dump(prospects, f, indent=2)
    print(f"✓ Saved {len(prospects)} Box prospects to box_prospects.json")
    
    # Save content
    content = generate_box_content()
    with open(OUTPUT_DIR / 'box_content.json', 'w') as f:
        json.dump(content, f, indent=2)
    print(f"✓ Saved {len(content)} content items to box_content.json")
    
    # Save sequences
    sequences = generate_sequences()
    with open(OUTPUT_DIR / 'box_sequences.json', 'w') as f:
        json.dump(sequences, f, indent=2)
    print(f"✓ Saved {len(sequences)} sequences to box_sequences.json")
    
    print("\n✅ Box demo data import complete!")
    print(f"   Output directory: {OUTPUT_DIR}")


if __name__ == "__main__":
    save_demo_data()
