"""
Prism - Box Sales Campaign Intelligence

PIE-powered sales intelligence:
- Extract prospects from call transcripts (PIE extraction)
- Enrich with web data (Brave API grounding)
- Simulate email sequences against prospects
- Generate personalized campaigns
"""

from __future__ import annotations
import json
import logging
import os
import sys
import uuid
from datetime import datetime
from pathlib import Path
from dataclasses import asdict

from flask import Flask, render_template, request, jsonify, redirect, url_for, flash

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from sales.prospect_model import ProspectWorldModel, ProspectModel, ProspectExtractor, ProspectEnricher, PainPoint, Objection, Stakeholder, Activity
from sales.content_model import SalesContent, ContentType, ContentTone
from sales.interaction_simulator import InteractionSimulator, InteractionResult

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = Flask(__name__)
app.secret_key = os.environ.get("FLASK_SECRET_KEY", "dev-key-change-in-prod")

# Data paths
SEED_DATA_PATH = Path(__file__).parent / "seed_data" / "box_seed_data.json"
PROSPECTS_DIR = Path(__file__).parent / "prospects"
PROSPECTS_DIR.mkdir(exist_ok=True)

# Initialize PIE components (lazy)
_extractor = None
_enricher = None
_simulator = None

def get_extractor():
    global _extractor
    if _extractor is None:
        _extractor = ProspectExtractor()
    return _extractor

def get_enricher():
    global _enricher
    if _enricher is None:
        _enricher = ProspectEnricher()
    return _enricher

def get_simulator():
    global _simulator
    if _simulator is None:
        _simulator = InteractionSimulator(model="gpt-4o-mini")
    return _simulator


def load_seed_data():
    """Load cached Box seed data."""
    if SEED_DATA_PATH.exists():
        with open(SEED_DATA_PATH) as f:
            return json.load(f)
    return {"sales_plays": [], "outreach_stats": [], "prospects": [], "agent_config": {}}


SEED_DATA = load_seed_data()


# ============================================================================
# ROUTES
# ============================================================================

@app.route("/")
def dashboard():
    """Dashboard with PIE-powered insights."""
    prospects = ProspectWorldModel.list_all()
    
    # Stats
    active = [p for p in prospects if p.stage not in ['closed_won', 'closed_lost']]
    total_value = sum(p.deal_value for p in active)
    total_probability_weighted = sum(p.deal_value * p.probability for p in active)
    
    return render_template(
        "dashboard.html",
        sales_plays=SEED_DATA.get('sales_plays', []),
        outreach_stats=SEED_DATA.get('outreach_stats', [])[:10],
        prospects=prospects[:5],
        total_value=total_value,
        weighted_value=total_probability_weighted,
        total_prospects=len(prospects),
        total_plays=len(SEED_DATA.get('sales_plays', [])),
        total_emails=sum(len(p.get('emails', [])) for p in SEED_DATA.get('sales_plays', [])),
    )


@app.route("/plays")
def plays():
    """View all sales plays."""
    return render_template(
        "plays.html",
        sales_plays=SEED_DATA.get('sales_plays', []),
    )


@app.route("/play/<play_id>")
def play_detail(play_id):
    """View single play."""
    play = next((p for p in SEED_DATA.get('sales_plays', []) if p['id'] == play_id), None)
    if not play:
        flash("Play not found", "error")
        return redirect(url_for('plays'))
    return render_template("play_detail.html", play=play)


@app.route("/stats")
def stats():
    """Outreach statistics."""
    return render_template(
        "stats.html",
        outreach_stats=SEED_DATA.get('outreach_stats', []),
    )


# ============================================================================
# PROSPECTS (PIE-POWERED)
# ============================================================================

@app.route("/prospects")
def prospects_list():
    """List all prospects."""
    prospects = ProspectWorldModel.list_all()
    return render_template("prospects.html", prospects=prospects)


@app.route("/prospect/<prospect_id>")
def prospect_detail(prospect_id):
    """Prospect detail with PIE world model."""
    prospect = ProspectWorldModel.load(prospect_id)
    if not prospect:
        flash("Prospect not found", "error")
        return redirect(url_for('prospects_list'))
    
    return render_template(
        "prospect_detail.html", 
        prospect=prospect,
        sales_plays=SEED_DATA.get('sales_plays', []),
    )


@app.route("/add", methods=["GET", "POST"])
def add_prospect():
    """Add prospect manually or from transcript."""
    if request.method == "POST":
        # Check if transcript extraction
        transcript = request.form.get('transcript', '').strip()
        
        if transcript:
            # PIE extraction
            try:
                extractor = get_extractor()
                prospect = extractor.extract_from_transcript(transcript)
                flash(f"Extracted prospect: {prospect.name} from {prospect.company}", "success")
                return redirect(url_for('prospect_detail', prospect_id=prospect.id))
            except Exception as e:
                logger.error(f"Extraction failed: {e}")
                flash(f"Extraction failed: {e}", "error")
                return redirect(url_for('add_prospect'))
        
        # Manual add
        prospect = ProspectWorldModel(
            id=str(uuid.uuid4())[:8],
            name=request.form.get('name', 'Unknown'),
            title=request.form.get('title', ''),
            company=request.form.get('company', ''),
            stage=request.form.get('stage', 'lead'),
            deal_value=float(request.form.get('deal_value', 0) or 0),
        )
        
        # Parse pain points
        pain_text = request.form.get('pain_points', '')
        for line in pain_text.strip().split('\n'):
            if line.strip():
                prospect.pain_points.append(PainPoint(
                    description=line.strip(),
                    severity="medium",
                    category="technical",
                ))
        
        prospect.calculate_probability()
        prospect.save()
        flash(f"Added prospect: {prospect.name}", "success")
        return redirect(url_for('prospect_detail', prospect_id=prospect.id))
    
    return render_template("add_prospect.html")


@app.route("/prospect/<prospect_id>/enrich", methods=["POST"])
def enrich_prospect(prospect_id):
    """Web-enrich prospect with Brave API."""
    prospect = ProspectWorldModel.load(prospect_id)
    if not prospect:
        flash("Prospect not found", "error")
        return redirect(url_for('prospects_list'))
    
    try:
        enricher = get_enricher()
        prospect = enricher.enrich_company(prospect)
        prospect = enricher.enrich_stakeholders(prospect)
        flash(f"Enriched {prospect.company} with web data", "success")
    except Exception as e:
        logger.error(f"Enrichment failed: {e}")
        flash(f"Enrichment failed: {e}", "error")
    
    return redirect(url_for('prospect_detail', prospect_id=prospect_id))


@app.route("/prospect/<prospect_id>/delete", methods=["POST"])
def delete_prospect(prospect_id):
    """Delete prospect."""
    path = PROSPECTS_DIR / f"{prospect_id}.json"
    if path.exists():
        path.unlink()
        flash("Prospect deleted", "success")
    return redirect(url_for('prospects_list'))


# ============================================================================
# SIMULATION (PIE CORE FEATURE)
# ============================================================================

@app.route("/simulate/<prospect_id>", methods=["GET", "POST"])
def simulate(prospect_id):
    """Simulate email sequence against prospect."""
    prospect = ProspectWorldModel.load(prospect_id)
    if not prospect:
        flash("Prospect not found", "error")
        return redirect(url_for('prospects_list'))
    
    simulation_result = None
    
    if request.method == "POST":
        # Get email content to simulate
        subject = request.form.get('subject', '')
        body = request.form.get('body', '')
        play_id = request.form.get('play_id', '')
        
        if play_id:
            # Use email from a play
            play = next((p for p in SEED_DATA.get('sales_plays', []) if p['id'] == play_id), None)
            if play and play.get('emails'):
                email_idx = int(request.form.get('email_idx', 0))
                email = play['emails'][email_idx]
                subject = email['subject']
                body = email['body']
        
        if subject and body:
            # Build content model
            content = SalesContent(
                id=str(uuid.uuid4())[:8],
                name=subject[:50],
                type=ContentType.EMAIL,
                subject_line=subject,
                raw_content=body,
                tone=ContentTone.PROFESSIONAL,
            )
            
            # Convert ProspectWorldModel to ProspectModel for simulator
            prospect_for_sim = ProspectModel.from_world_model(prospect)
            
            try:
                simulator = get_simulator()
                result = simulator.simulate_interaction(prospect_for_sim, content)
                simulation_result = result.to_dict()
                flash("Simulation complete!", "success")
            except Exception as e:
                logger.error(f"Simulation failed: {e}")
                flash(f"Simulation failed: {e}", "error")
    
    return render_template(
        "simulate.html",
        prospect=prospect,
        sales_plays=SEED_DATA.get('sales_plays', []),
        simulation_result=simulation_result,
    )


# ============================================================================
# CAMPAIGN GENERATION
# ============================================================================

@app.route("/generate", methods=["GET", "POST"])
def generate_campaign():
    """Generate new campaign with AI."""
    prospects = ProspectWorldModel.list_all()
    generated = None
    
    if request.method == "POST":
        persona = request.form.get('persona', '')
        agency = request.form.get('agency', '')
        use_case = request.form.get('use_case', 'ecm')
        prospect_id = request.form.get('prospect_id', '')
        
        # Get prospect context if selected
        prospect_context = ""
        if prospect_id:
            prospect = ProspectWorldModel.load(prospect_id)
            if prospect:
                pains = [p.description for p in prospect.pain_points[:3]]
                prospect_context = f"""
Target Prospect: {prospect.name}, {prospect.title} at {prospect.company}
Stage: {prospect.stage}
Pain Points: {'; '.join(pains)}
"""
        
        # Generate with LLM
        from pie.core.llm import LLMClient
        llm = LLMClient()
        
        prompt = f"""Generate a 3-touch email sequence for Box Public Sector sales.

Target: {persona or 'IT Director'}
Agency: {agency or 'State Government'}
Use Case: {use_case}
{prospect_context}

Box Value Props to include:
- FedRAMP High, DoD IL4, CJIS compliance
- Replace legacy ECM (FileNet, OpenText, Hyland)
- Box Shield Pro (malware detection, anomaly detection)
- Box AI and AI Agents
- Unlimited e-signatures

Generate 3 emails with increasing urgency. Return JSON:
{{
    "emails": [
        {{"touch": "1st Touch", "subject": "...", "body": "..."}},
        {{"touch": "2nd Touch", "subject": "...", "body": "..."}},
        {{"touch": "3rd Touch", "subject": "...", "body": "..."}}
    ]
}}"""
        
        try:
            result = llm.chat(
                messages=[{"role": "user", "content": prompt}],
                model="gpt-4o-mini",
                json_mode=True
            )
            generated = result["content"] if isinstance(result["content"], dict) else json.loads(result["content"])
            flash("Campaign generated!", "success")
        except Exception as e:
            logger.error(f"Generation failed: {e}")
            flash(f"Generation failed: {e}", "error")
    
    return render_template(
        "generate.html",
        prospects=prospects,
        agent_config=SEED_DATA.get('agent_config', {}),
        generated=generated,
    )


# ============================================================================
# API
# ============================================================================

@app.route("/api/prospects")
def api_prospects():
    prospects = ProspectWorldModel.list_all()
    return jsonify([asdict(p) for p in prospects])


@app.route("/api/prospect/<prospect_id>")
def api_prospect(prospect_id):
    prospect = ProspectWorldModel.load(prospect_id)
    if not prospect:
        return jsonify({"error": "Not found"}), 404
    return jsonify(asdict(prospect))


@app.route("/api/extract", methods=["POST"])
def api_extract():
    """Extract prospect from transcript."""
    data = request.json
    transcript = data.get('transcript', '')
    if not transcript:
        return jsonify({"error": "No transcript provided"}), 400
    
    try:
        extractor = get_extractor()
        prospect = extractor.extract_from_transcript(transcript)
        return jsonify(asdict(prospect))
    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route("/api/simulate", methods=["POST"])
def api_simulate():
    """Simulate content against prospect."""
    data = request.json
    prospect_id = data.get('prospect_id')
    subject = data.get('subject', '')
    body = data.get('body', '')
    
    prospect = ProspectWorldModel.load(prospect_id)
    if not prospect:
        return jsonify({"error": "Prospect not found"}), 404
    
    content = SalesContent(
        id=str(uuid.uuid4())[:8],
        name=subject[:50],
        type=ContentType.EMAIL,
        subject_line=subject,
        raw_content=body,
        tone=ContentTone.PROFESSIONAL,
    )
    
    prospect_for_sim = ProspectModel.from_world_model(prospect)
    
    try:
        simulator = get_simulator()
        result = simulator.simulate_interaction(prospect_for_sim, content)
        return jsonify(result.to_dict())
    except Exception as e:
        return jsonify({"error": str(e)}), 500


# ============================================================================
# MAIN
# ============================================================================

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    print(f"""
╔══════════════════════════════════════════════════════════════════╗
║  🎯 PRISM - PIE-Powered Sales Intelligence                       ║
║                                                                  ║
║  Features:                                                       ║
║  • Extract prospects from transcripts (PIE)                      ║
║  • Web enrichment (Brave API)                                    ║
║  • Simulate sequences against prospects                          ║
║  • Generate personalized campaigns                               ║
║                                                                  ║
║  Data: {len(SEED_DATA.get('sales_plays', []))} plays • {sum(len(p.get('emails', [])) for p in SEED_DATA.get('sales_plays', []))} emails • {len(SEED_DATA.get('outreach_stats', []))} sequences        ║
║                                                                  ║
║  → http://localhost:{port}                                         ║
╚══════════════════════════════════════════════════════════════════╝
    """)
    app.run(host="0.0.0.0", port=port, debug=True)
