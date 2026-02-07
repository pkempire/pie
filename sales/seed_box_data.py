#!/usr/bin/env python3
"""
Seed the sales app with real Box data from the Downloads folder.
Extracts sales plays, outreach stats, and creates initial content.
"""

import json
import os
import sys
import re
import pandas as pd
from pathlib import Path
from datetime import datetime

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from pie.core.llm import LLMClient

DATA_DIR = Path.home() / "Downloads" / "Data Validation Platform"
OUTPUT_DIR = Path(__file__).parent / "seed_data"
OUTPUT_DIR.mkdir(exist_ok=True)


def extract_boxnote_text(filepath: Path) -> str:
    """Extract text from boxnote JSON."""
    with open(filepath, 'r') as f:
        try:
            data = json.load(f)
        except:
            return f.read()
    
    def walk(node):
        text = ''
        if isinstance(node, dict):
            if node.get('type') == 'text':
                text += node.get('text', '')
            if node.get('type') in ('paragraph', 'heading', 'list_item'):
                text += '\n'
            for child in node.get('content', []):
                text += walk(child)
        elif isinstance(node, list):
            for item in node:
                text += walk(item)
        return text
    
    return walk(data.get('doc', {}))


def extract_emails_from_play(content: str) -> list[dict]:
    """Extract individual email templates from a sales play."""
    emails = []
    
    # Pattern to find email touches
    touch_pattern = r'(\d+(?:st|nd|rd|th) TOUCH - [A-Z])\s*\n*Subject [Ll]ine\(?s?\)?[:\s]*["""]?([^""""\n]+)["""]?\s*\n+(.*?)(?=\d+(?:st|nd|rd|th) TOUCH|\Z)'
    
    matches = re.findall(touch_pattern, content, re.DOTALL | re.IGNORECASE)
    
    for match in matches:
        touch_id, subject, body = match
        # Clean up body
        body = body.strip()
        # Remove template variables explanation
        body = re.sub(r'\{\{[^}]+\}\}', '[PERSONALIZE]', body)
        
        emails.append({
            "touch": touch_id.strip(),
            "subject": subject.strip().strip('"'),
            "body": body[:1500],  # Truncate
        })
    
    return emails


def parse_sales_plays() -> list[dict]:
    """Parse all sales play boxnotes."""
    plays = []
    
    play_files = [
        ("Sales Play Example 1.boxnote", "ECM Modernization", "Replace legacy ECM systems"),
        ("Sales Play Example 2.boxnote", "Shield Pro Security", "Advanced security for content"),
        ("Sales Play Example 3.boxnote", "CMMC 2.0 Compliance", "Defense contractor compliance"),
    ]
    
    for filename, name, description in play_files:
        filepath = DATA_DIR / filename
        if not filepath.exists():
            print(f"Warning: {filepath} not found")
            continue
        
        content = extract_boxnote_text(filepath)
        emails = extract_emails_from_play(content)
        
        plays.append({
            "id": filename.replace(" ", "_").replace(".boxnote", "").lower(),
            "name": name,
            "description": description,
            "source_file": filename,
            "emails": emails,
            "raw_content": content[:5000],
        })
        print(f"Extracted {len(emails)} emails from {name}")
    
    return plays


def parse_outreach_stats() -> list[dict]:
    """Parse outreach stats Excel."""
    filepath = DATA_DIR / "Sep 1 to Dec 11 Outreach Stats.xls"
    if not filepath.exists():
        print(f"Warning: {filepath} not found")
        return []
    
    df = pd.read_excel(filepath, engine='xlrd')
    
    # Get top performing sequences
    top = df.nlargest(20, 'Meetings booked')
    
    stats = []
    for _, row in top.iterrows():
        stats.append({
            "sequence_name": row['Sequence name'],
            "owner": row['Owner name'],
            "meetings_booked": int(row['Meetings booked']) if pd.notna(row['Meetings booked']) else 0,
            "open_rate": float(row['Open rate']) if pd.notna(row['Open rate']) else 0,
            "reply_rate": float(row['Reply rate']) if pd.notna(row['Reply rate']) else 0,
            "prospects_invited": int(row['Prospects invited']) if pd.notna(row['Prospects invited']) else 0,
        })
    
    print(f"Extracted {len(stats)} top sequences from outreach stats")
    return stats


def parse_agent_prompt() -> dict:
    """Parse the agent prompt/instructions."""
    filepath = DATA_DIR / "Prompt for Sales Outreach Agent.boxnote"
    if not filepath.exists():
        return {}
    
    content = extract_boxnote_text(filepath)
    
    return {
        "system_prompt": content,
        "modes": ["Campaign QA & Coaching", "Generate New Campaign"],
        "value_props": [
            "FedRAMP High, DoD IL4, CJIS, IRS 1075, HIPAA compliance",
            "Box Shield Pro - malware detection, anomaly detection",
            "Box AI, Box AI Agents, Metadata Extraction",
            "Replace legacy ECM: FileNet, OpenText, Laserfiche",
            "Unlimited e-signatures included",
        ],
        "target_personas": [
            "Federal IT Directors",
            "State CIOs/CISOs", 
            "Defense Contractors",
            "Public Safety/Courts",
            "Healthcare/HHS",
        ],
    }


def generate_sample_prospects(llm: LLMClient) -> list[dict]:
    """Use LLM to generate realistic Box public sector prospects."""
    
    prompt = """Generate 5 realistic public sector sales prospects for Box (the content cloud company).
    
Each prospect should be a government or defense organization that would benefit from:
- FedRAMP High compliant content management
- Replacing legacy ECM (FileNet, OpenText)
- CMMC 2.0 compliance
- Secure collaboration

Return JSON array:
[
    {
        "name": "Contact Name",
        "title": "Job Title",
        "company": "Organization Name",
        "vertical": "Federal|State|Local|Defense|Healthcare",
        "stage": "discovery|evaluation|proposal",
        "deal_value": 250000,
        "pain_points": ["pain 1", "pain 2"],
        "current_solution": "Current ECM/system",
        "trigger_event": "Why now - budget, mandate, etc."
    }
]"""
    
    response = llm.chat(prompt, model="gpt-4o-mini", response_format={"type": "json_object"})
    
    try:
        data = json.loads(response)
        if isinstance(data, dict) and 'prospects' in data:
            return data['prospects']
        elif isinstance(data, list):
            return data
    except:
        pass
    
    return []


def save_seed_data():
    """Save all extracted data."""
    
    # Parse Box files
    plays = parse_sales_plays()
    stats = parse_outreach_stats()
    agent_config = parse_agent_prompt()
    
    # Generate sample prospects
    print("Generating sample prospects with LLM...")
    try:
        llm = LLMClient()
        prospects = generate_sample_prospects(llm)
        print(f"Generated {len(prospects)} sample prospects")
    except Exception as e:
        print(f"LLM generation failed: {e}")
        prospects = []
    
    # Save everything
    seed_data = {
        "generated_at": datetime.now().isoformat(),
        "sales_plays": plays,
        "outreach_stats": stats,
        "agent_config": agent_config,
        "prospects": prospects,
    }
    
    with open(OUTPUT_DIR / "box_seed_data.json", 'w') as f:
        json.dump(seed_data, f, indent=2)
    
    print(f"\n✅ Saved seed data to {OUTPUT_DIR / 'box_seed_data.json'}")
    print(f"   - {len(plays)} sales plays")
    print(f"   - {len(stats)} outreach stats")
    print(f"   - {len(prospects)} sample prospects")
    
    return seed_data


if __name__ == "__main__":
    save_seed_data()
