#!/usr/bin/env python3
"""
Sales Intelligence Demo — end-to-end CLI.

Takes transcript files → extracts entities → builds prospect models →
enriches from web → detects uncertainties → outputs full intelligence view.

Usage:
    python sales/demo.py
    python sales/demo.py --transcripts /path/to/transcripts/
    python sales/demo.py --no-web        # Skip web enrichment
    python sales/demo.py --no-simulate   # Skip simulation prompt
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
import os
import time
from pathlib import Path
from typing import Optional, Tuple

# Add project root to path
project_root = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(project_root))

from pie.core.llm import LLMClient

from sales.extraction import extract_sales_entities
from sales.prospect_model import ProspectWorldModel
from sales.enrichment import WebEnricher
from sales.uncertainty import detect_uncertainties, add_rule_based_uncertainties


def setup_logging():
    formatter = logging.Formatter(
        "%(asctime)s [%(name)s] %(message)s",
        datefmt="%H:%M:%S",
    )
    handler = logging.StreamHandler()
    handler.setFormatter(formatter)
    
    for name in ["sales", "pie"]:
        logger = logging.getLogger(name)
        logger.setLevel(logging.INFO)
        logger.addHandler(handler)


def load_transcript(path: str | Path) -> tuple[str, str]:
    """Load a transcript file. Returns (filename, content)."""
    path = Path(path)
    content = path.read_text(encoding="utf-8")
    return path.stem, content


def print_section(title: str, char: str = "═"):
    width = 70
    print(f"\n{char * width}")
    print(f"  {title}")
    print(f"{char * width}")


def print_subsection(title: str):
    print(f"\n  ▸ {title}")
    print(f"  {'─' * 50}")


def format_entity_graph(model) -> str:
    """Format the entity graph for display."""
    lines = []
    
    # Prospect node
    lines.append(f"  ╔══ PROSPECT: {model.name}")
    lines.append(f"  ║  Title: {model.title}")
    lines.append(f"  ║  Company: {model.company}")
    lines.append(f"  ║  Email: {model.email}")
    if model.personality_traits:
        # Handle both string list and dict list formats
        traits = []
        for t in model.personality_traits:
            if isinstance(t, dict):
                traits.append(t.get("trait", t.get("name", str(t))))
            else:
                traits.append(str(t))
        lines.append(f"  ║  Traits: {', '.join(traits)}")
    
    # Handle dict or string for communication/decision style
    comm = model.communication_style
    if isinstance(comm, dict):
        comm = comm.get("style", str(comm))
    dec = model.decision_style
    if isinstance(dec, dict):
        dec = dec.get("style", str(dec))
    
    lines.append(f"  ║  Communication: {comm}")
    lines.append(f"  ║  Decision Style: {dec}")
    
    # Pain points
    if model.pain_points:
        lines.append(f"  ║")
        lines.append(f"  ╠══ PAIN POINTS ({len(model.pain_points)})")
        for pp in model.pain_points:
            severity = pp.get("severity", "?").upper()
            emoji = {"CRITICAL": "🔴", "HIGH": "🟠", "MEDIUM": "🟡", "LOW": "🟢"}.get(severity, "⚪")
            lines.append(f"  ║  {emoji} [{severity}] {pp.get('description', '')}")
            if pp.get("evolution"):
                lines.append(f"  ║     ↳ Evolution: {pp['evolution']}")
    
    # Buying signals
    if model.buying_signals:
        lines.append(f"  ║")
        lines.append(f"  ╠══ BUYING SIGNALS ({len(model.buying_signals)})")
        for sig in model.buying_signals:
            strength = sig.get("strength", "?")
            emoji = {"strong": "💪", "moderate": "👍", "weak": "👋"}.get(strength, "❓")
            lines.append(f"  ║  {emoji} [{strength}] {sig.get('signal', '')}")
    
    # Objections
    if model.objections:
        lines.append(f"  ║")
        lines.append(f"  ╠══ OBJECTIONS ({len(model.objections)})")
        for obj in model.objections:
            status = obj.get("resolution_status", "?")
            emoji = {"resolved": "✅", "partially_resolved": "🔄", "unresolved": "❌"}.get(status, "❓")
            lines.append(f"  ║  {emoji} [{obj.get('type', '?')}] {obj.get('description', '')}")
            if obj.get("what_would_resolve"):
                lines.append(f"  ║     ↳ Resolve: {obj['what_would_resolve']}")
    
    # Stakeholders
    if model.stakeholders:
        lines.append(f"  ║")
        lines.append(f"  ╠══ STAKEHOLDERS ({len(model.stakeholders)})")
        for s in model.stakeholders:
            emoji = {"positive": "👍", "neutral": "😐", "negative": "👎", "unknown": "❓"}.get(
                s.get("sentiment", "unknown"), "❓"
            )
            lines.append(
                f"  ║  {emoji} {s.get('name', '?')} — {s.get('role', '?')} "
                f"({s.get('influence', '?')})"
            )
    
    # Competitive context
    if model.competitive_context:
        lines.append(f"  ║")
        lines.append(f"  ╠══ COMPETITIVE CONTEXT ({len(model.competitive_context)})")
        for cc in model.competitive_context:
            lines.append(
                f"  ║  ⚔️  {cc.get('competitor_or_alternative', '?')} "
                f"({cc.get('type', '?')}) — {cc.get('details', '')}"
            )
    
    # Next steps
    if model.next_steps:
        lines.append(f"  ║")
        lines.append(f"  ╠══ NEXT STEPS ({len(model.next_steps)})")
        for ns in model.next_steps:
            commitment = ns.get("commitment_level", "?")
            emoji = {"firm": "🎯", "tentative": "🔄", "suggested": "💡"}.get(commitment, "❓")
            lines.append(
                f"  ║  {emoji} {ns.get('action', '?')} "
                f"(owner: {ns.get('owner', '?')}, commitment: {commitment})"
            )
    
    lines.append(f"  ╚{'═' * 60}")
    
    return "\n".join(lines)


def format_temporal_evolution(model) -> str:
    """Format temporal evolution for display."""
    lines = []
    
    metadata = model.meeting_metadata
    meetings = metadata.get("meetings", [])
    
    lines.append(f"  Deal Stage: {metadata.get('overall_deal_stage', 'unknown')}")
    lines.append(f"  Momentum: {metadata.get('momentum', 'unknown')}")
    lines.append(f"  Meetings: {metadata.get('meeting_count', '?')}")
    
    for m in meetings:
        lines.append(f"    Meeting {m.get('number', '?')}: {m.get('topic', '?')} (tone: {m.get('tone', '?')})")
    
    if model.evolutions:
        lines.append(f"\n  State Changes:")
        for evo in model.evolutions:
            if evo.changed or len(evo.states) > 1:
                lines.append(f"    • [{evo.entity_type}] {evo.entity_key}")
                lines.append(f"      {evo.evolution_summary}")
    
    return "\n".join(lines)


def format_web_enrichment(enrichment: dict) -> str:
    """Format web enrichment for display."""
    lines = []
    
    company = enrichment.get("company", {})
    if company:
        lines.append(f"  Company: {company.get('name', '?')}")
        if company.get("description"):
            lines.append(f"    {company['description'][:200]}")
        if company.get("website") or company.get("url"):
            lines.append(f"    🌐 {company.get('website', company.get('url', ''))}")
        if company.get("industry"):
            lines.append(f"    🏭 Industry: {company['industry']}")
        if company.get("recent_developments"):
            lines.append(f"    📰 Recent:")
            for dev in company["recent_developments"][:3]:
                lines.append(f"       • {dev[:120]}")
        elif company.get("recent_news"):
            lines.append(f"    📰 Recent:")
            for news in company["recent_news"][:3]:
                lines.append(f"       • {news[:120]}")
        if company.get("talking_points"):
            lines.append(f"    💬 Talking Points:")
            for tp in company["talking_points"]:
                lines.append(f"       • {tp}")
    
    person = enrichment.get("person", {})
    if person and person.get("linkedin_url"):
        lines.append(f"\n  Person:")
        lines.append(f"    🔗 {person.get('linkedin_url', '')}")
        if person.get("background"):
            lines.append(f"    {person['background'][:200]}")
    
    tech = enrichment.get("tech_stack", {})
    if tech:
        if tech.get("known_technologies"):
            lines.append(f"\n  Tech Stack:")
            lines.append(f"    🔧 {', '.join(tech['known_technologies'][:8])}")
        if tech.get("signals") and not tech.get("known_technologies"):
            lines.append(f"\n  Tech Signals:")
            for sig in tech["signals"][:2]:
                lines.append(f"    • {sig[:120]}")
    
    industry = enrichment.get("industry", {})
    if industry and industry.get("key_trends"):
        lines.append(f"\n  Industry Trends ({industry.get('name', '?')}):")
        for trend in industry["key_trends"][:3]:
            lines.append(f"    📈 {trend}")
    
    return "\n".join(lines) if lines else "  (No web enrichment data)"


def format_uncertainties(uncertainties: dict) -> str:
    """Format uncertainty flags for display."""
    lines = []
    
    confidence = uncertainties.get("overall_confidence", "?")
    lines.append(f"  Overall Confidence: {confidence}")
    
    top_q = uncertainties.get("top_3_questions", [])
    if top_q:
        lines.append(f"\n  🔝 TOP QUESTIONS:")
        for i, q in enumerate(top_q, 1):
            lines.append(f"    {i}. {q}")
    
    flags = uncertainties.get("uncertainties", [])
    if flags:
        lines.append(f"\n  ⚠️  UNCERTAINTY FLAGS ({len(flags)}):")
        
        # Group by impact
        high = [f for f in flags if f.get("impact") == "high"]
        medium = [f for f in flags if f.get("impact") == "medium"]
        low = [f for f in flags if f.get("impact") == "low"]
        
        for impact_group, label, emoji in [
            (high, "HIGH IMPACT", "🔴"),
            (medium, "MEDIUM IMPACT", "🟡"),
            (low, "LOW IMPACT", "🟢"),
        ]:
            if impact_group:
                lines.append(f"\n    {emoji} {label}:")
                for f in impact_group:
                    lines.append(f"    ┌ [{f.get('type', '?')}] {f.get('entity_type', '?')}: {f.get('entity_reference', '?')}")
                    lines.append(f"    │ ❓ {f.get('question', '?')}")
                    lines.append(f"    │ 📋 Context: {f.get('context', '')[:120]}")
                    lines.append(f"    └ 💡 Action: {f.get('suggested_action', '')[:120]}")
                    lines.append(f"")
    
    return "\n".join(lines)


def run_demo(
    transcript_dir: str | Path,
    output_dir: str | Path = "sales/output",
    skip_web: bool = False,
    skip_simulate: bool = False,
    model: str = "gpt-4o-mini",
):
    """Run the full demo pipeline."""
    transcript_dir = Path(transcript_dir)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Initialize
    llm = LLMClient()
    world_model = ProspectWorldModel(output_dir=output_dir)
    enricher = WebEnricher(llm=llm) if not skip_web else None
    
    # Find transcript files
    transcript_files = sorted(transcript_dir.glob("*transcript*"))
    if not transcript_files:
        transcript_files = sorted(transcript_dir.glob("*.txt"))
    
    if not transcript_files:
        print(f"❌ No transcript files found in {transcript_dir}")
        return
    
    print_section("SALES INTELLIGENCE ENGINE", "█")
    print(f"  Transcripts: {len(transcript_files)}")
    print(f"  Web Enrichment: {'ON' if not skip_web else 'OFF'}")
    print(f"  Model: {model}")
    print(f"  Output: {output_dir}")
    
    all_prospects = []
    total_tokens = {"prompt": 0, "completion": 0, "total": 0}
    
    for tf in transcript_files:
        filename, content = load_transcript(tf)
        
        print_section(f"PROCESSING: {filename}")
        
        # Step 1: Extract entities
        print_subsection("Step 1: Entity Extraction")
        start = time.time()
        extraction_result = extract_sales_entities(content, llm, model=model)
        extraction = extraction_result["extraction"]
        tokens = extraction_result["tokens"]
        for k in total_tokens:
            total_tokens[k] += tokens.get(k, 0)
        
        elapsed = time.time() - start
        print(f"  ✅ Extracted in {elapsed:.1f}s ({tokens['total']} tokens)")
        
        # Step 2: Build prospect model
        print_subsection("Step 2: Build Prospect Model")
        prospect = world_model.ingest_extraction(extraction_result)
        print(f"  ✅ Built model for {prospect.name} ({prospect.company})")
        
        # Step 3: Web enrichment
        if enricher and not skip_web:
            print_subsection("Step 3: Web Enrichment")
            start = time.time()
            try:
                enrichment = enricher.enrich_prospect(
                    name=prospect.name,
                    title=prospect.title,
                    company=prospect.company,
                )
                prospect.web_enrichment = enrichment.to_dict()
                elapsed = time.time() - start
                print(f"  ✅ Enriched in {elapsed:.1f}s ({enricher.stats['queries']} searches)")
            except Exception as e:
                print(f"  ⚠️  Enrichment failed: {e}")
                prospect.web_enrichment = {"error": str(e)}
        else:
            print_subsection("Step 3: Web Enrichment (SKIPPED)")
        
        # Step 4: Uncertainty detection
        print_subsection("Step 4: Uncertainty Detection")
        start = time.time()
        prospect_dict = prospect.to_dict()
        uncertainties = detect_uncertainties(prospect_dict, llm, model=model)
        tokens_u = uncertainties.get("tokens", {})
        
        # Add rule-based checks
        uncertainties = add_rule_based_uncertainties(prospect_dict, uncertainties)
        prospect.uncertainties = uncertainties.get("uncertainties", [])
        
        elapsed = time.time() - start
        print(f"  ✅ {len(prospect.uncertainties)} flags in {elapsed:.1f}s")
        
        all_prospects.append(prospect)
        
        # === Display Results ===
        
        print_section(f"INTELLIGENCE: {prospect.name}", "─")
        
        print_subsection("A) Entity Graph")
        print(format_entity_graph(prospect))
        
        print_subsection("B) Temporal Evolution")
        print(format_temporal_evolution(prospect))
        
        print_subsection("C) Web Enrichment")
        print(format_web_enrichment(prospect.web_enrichment))
        
        print_subsection("D) Uncertainty Flags")
        print(format_uncertainties(uncertainties))
        
        print_subsection("E) Simulation Prompt")
        sim_prompt = world_model.build_simulation_prompt(
            prospect.name,
            pitch="We'd like to start with a focused 30-day pilot on your top 3 critical pipelines. We'll define clear SLAs together and I'll personally ensure your compliance team has everything they need."
        )
        print(f"  [Simulation prompt built: {len(sim_prompt)} chars]")
        print(f"  Preview (first 500 chars):")
        print(f"  {'─' * 50}")
        for line in sim_prompt[:500].split("\n"):
            print(f"  │ {line}")
        print(f"  │ ...")
        
        # Run simulation if not skipped
        if not skip_simulate:
            print_subsection("E.1) Simulated Response")
            start = time.time()
            sim_messages = [
                {"role": "system", "content": sim_prompt},
                {
                    "role": "user",
                    "content": (
                        "We'd like to start with a focused 30-day pilot on your top 3 critical pipelines. "
                        "We'll define clear SLAs together and I'll personally ensure your compliance team "
                        "has everything they need."
                    ),
                },
            ]
            sim_result = llm.chat(sim_messages, model=model, max_tokens=500)
            elapsed = time.time() - start
            print(f"  🎭 {prospect.name} would likely respond:")
            print(f"  {'─' * 50}")
            for line in sim_result["content"].split("\n"):
                print(f"  │ {line}")
            print(f"  {'─' * 50}")
            print(f"  (Generated in {elapsed:.1f}s)")
        
        # Save individual prospect
        path = world_model.save_individual(prospect.name)
        print(f"\n  💾 Saved to {path}")
    
    # === Save combined output ===
    print_section("COMBINED OUTPUT")
    
    # Save all prospects
    combined_path = world_model.save()
    print(f"  💾 All prospects: {combined_path}")
    
    # Save demo data for HTML viz
    demo_data = {
        "generated_at": time.time(),
        "prospects": {},
        "contacts_csv": str(transcript_dir / "data_platform_contacts.csv"),
    }
    
    for prospect in all_prospects:
        p_dict = prospect.to_dict()
        # Add simulation prompt
        p_dict["simulation_prompt"] = world_model.build_simulation_prompt(prospect.name)
        demo_data["prospects"][prospect.name] = p_dict
    
    demo_path = output_dir / "demo_data.json"
    with open(demo_path, "w") as f:
        json.dump(demo_data, f, indent=2, default=str)
    print(f"  💾 Demo data: {demo_path}")
    
    # Token stats
    print_section("STATS")
    print(f"  LLM calls: {llm.stats['total_calls']}")
    print(f"  Total tokens: {llm.stats['total_tokens']:,}")
    if enricher:
        print(f"  Web searches: {enricher.stats['queries']}")
    print(f"  Prospects processed: {len(all_prospects)}")


def main():
    parser = argparse.ArgumentParser(description="Sales Intelligence Demo")
    parser.add_argument(
        "--transcripts",
        type=str,
        default=str(Path.home() / "Downloads" / "demoData"),
        help="Directory containing transcript files",
    )
    parser.add_argument("--output", type=str, default="sales/output", help="Output directory")
    parser.add_argument("--no-web", action="store_true", help="Skip web enrichment")
    parser.add_argument("--no-simulate", action="store_true", help="Skip simulation")
    parser.add_argument("--model", type=str, default="gpt-4o-mini", help="LLM model")
    
    args = parser.parse_args()
    setup_logging()
    
    run_demo(
        transcript_dir=args.transcripts,
        output_dir=args.output,
        skip_web=args.no_web,
        skip_simulate=args.no_simulate,
        model=args.model,
    )


if __name__ == "__main__":
    main()
