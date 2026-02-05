"""
Sales Intelligence Web Application

Premium dark-themed dashboard for sales process intelligence.
Features: Pipeline view, process funnel, prospect details, what-if simulation.
"""

from __future__ import annotations
import json
import logging
import os
import sys
import uuid
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

from flask import Flask, render_template, request, jsonify, redirect, url_for, flash
from werkzeug.utils import secure_filename

# Add parent for imports
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from sales.process_mining import (
    SalesProcessMiner,
    generate_demo_pipeline,
    SALES_STAGES,
    STAGE_DISPLAY,
)
from sales.prospect_model import ProspectWorldModel

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("sales.app")

# Flask app
app = Flask(__name__)
app.secret_key = os.environ.get("FLASK_SECRET_KEY", "sales-intel-dev-key-change-in-prod")

# Config
UPLOAD_FOLDER = Path(__file__).parent / "uploads"
OUTPUT_FOLDER = Path(__file__).parent / "output"
ALLOWED_EXTENSIONS = {"txt", "json", "csv"}

UPLOAD_FOLDER.mkdir(exist_ok=True)
OUTPUT_FOLDER.mkdir(exist_ok=True)

app.config["UPLOAD_FOLDER"] = str(UPLOAD_FOLDER)
app.config["MAX_CONTENT_LENGTH"] = 16 * 1024 * 1024  # 16MB max

# Global state (in production, use a proper database)
pipeline_data: List[dict] = []
process_analysis: Dict[str, Any] = {}
miner = SalesProcessMiner()


def allowed_file(filename: str) -> bool:
    return "." in filename and filename.rsplit(".", 1)[1].lower() in ALLOWED_EXTENSIONS


def load_existing_prospects():
    """Load existing prospect data from output folder."""
    global pipeline_data
    
    prospects = []
    
    # Load individual prospect files
    for json_file in OUTPUT_FOLDER.glob("prospect_*.json"):
        try:
            with open(json_file) as f:
                data = json.load(f)
                prospects.append(data)
                logger.info(f"Loaded prospect: {data.get('name', 'Unknown')}")
        except Exception as e:
            logger.error(f"Error loading {json_file}: {e}")
    
    # If no prospects found, generate demo data
    if not prospects:
        logger.info("No existing prospects found, generating demo pipeline...")
        prospects = generate_demo_pipeline()
    
    pipeline_data = prospects
    return prospects


def refresh_analysis():
    """Refresh process mining analysis with current pipeline data."""
    global process_analysis
    process_analysis = miner.get_full_analysis(pipeline_data)


# Load data on startup
load_existing_prospects()
refresh_analysis()


# =============================================================================
# ROUTES
# =============================================================================

@app.route("/")
def dashboard():
    """Main dashboard view."""
    return render_template(
        "dashboard.html",
        summary=process_analysis.get("summary", {}),
        funnel=process_analysis.get("funnel", []),
        bottlenecks=process_analysis.get("bottlenecks", [])[:5],
        timelines=process_analysis.get("timelines", [])[:10],
        stage_distribution=process_analysis.get("stage_distribution", {}),
        stages=SALES_STAGES,
        stage_display=STAGE_DISPLAY,
    )


@app.route("/pipeline")
def pipeline():
    """Pipeline view with all deals."""
    # Group by stage
    by_stage = {stage: [] for stage in SALES_STAGES}
    
    for timeline in process_analysis.get("timelines", []):
        stage = timeline.get("current_stage", "lead")
        if stage in by_stage:
            by_stage[stage].append(timeline)
    
    return render_template(
        "pipeline.html",
        by_stage=by_stage,
        stages=SALES_STAGES,
        stage_display=STAGE_DISPLAY,
        total_deals=len(pipeline_data),
    )


@app.route("/prospect/<prospect_id>")
def prospect_detail(prospect_id: str):
    """Individual prospect detail view."""
    # Find prospect
    prospect = None
    timeline = None
    
    for p in pipeline_data:
        if p.get("id") == prospect_id:
            prospect = p
            break
    
    for t in process_analysis.get("timelines", []):
        if t.get("prospect_id") == prospect_id:
            timeline = t
            break
    
    if not prospect:
        flash("Prospect not found", "error")
        return redirect(url_for("pipeline"))
    
    # Get transition matrix for simulation
    transition_matrix = process_analysis.get("transition_matrix", {})
    
    return render_template(
        "prospect_detail.html",
        prospect=prospect,
        timeline=timeline,
        transition_matrix=transition_matrix,
        stages=SALES_STAGES,
        stage_display=STAGE_DISPLAY,
    )


@app.route("/upload", methods=["GET", "POST"])
def upload():
    """Upload transcript files."""
    if request.method == "POST":
        if "file" not in request.files:
            flash("No file selected", "error")
            return redirect(request.url)
        
        file = request.files["file"]
        
        if file.filename == "":
            flash("No file selected", "error")
            return redirect(request.url)
        
        if file and allowed_file(file.filename):
            filename = secure_filename(file.filename)
            filepath = Path(app.config["UPLOAD_FOLDER"]) / filename
            file.save(filepath)
            
            # Process the file
            try:
                content = filepath.read_text()
                
                if filename.endswith(".json"):
                    data = json.loads(content)
                    if isinstance(data, list):
                        pipeline_data.extend(data)
                    else:
                        pipeline_data.append(data)
                else:
                    # Text transcript - would need LLM extraction
                    # For now, create a placeholder
                    flash("Transcript uploaded. LLM extraction would process this.", "info")
                
                refresh_analysis()
                flash(f"Successfully uploaded {filename}", "success")
                
            except Exception as e:
                flash(f"Error processing file: {e}", "error")
            
            return redirect(url_for("dashboard"))
        else:
            flash("Invalid file type. Allowed: txt, json, csv", "error")
    
    return render_template("upload.html")


@app.route("/simulation", methods=["GET", "POST"])
def simulation():
    """What-if scenario simulation."""
    results = None
    selected_prospect = None
    
    if request.method == "POST":
        prospect_id = request.form.get("prospect_id")
        
        # Find prospect timeline
        timeline = None
        for t in miner.deal_timelines:
            if t.prospect_id == prospect_id:
                timeline = t
                break
        
        if timeline:
            selected_prospect = timeline.prospect_name
            
            # Get scenario parameters
            changes = {}
            if request.form.get("rep_change"):
                changes["rep_change"] = True
            if request.form.get("timing_speedup"):
                changes["timing_speedup"] = float(request.form.get("timing_speedup", 1.0))
            if request.form.get("remove_objection"):
                changes["remove_objection"] = True
            if request.form.get("add_champion"):
                changes["add_champion"] = True
            
            # Run simulation
            transition_matrix = process_analysis.get("transition_matrix", {})
            results = miner.simulate_scenario(timeline, changes, transition_matrix)
    
    return render_template(
        "simulation.html",
        timelines=process_analysis.get("timelines", []),
        results=results,
        selected_prospect=selected_prospect,
    )


@app.route("/process")
def process_view():
    """Process mining visualization."""
    return render_template(
        "process.html",
        transition_matrix=process_analysis.get("transition_matrix", {}),
        bottlenecks=process_analysis.get("bottlenecks", []),
        funnel=process_analysis.get("funnel", []),
        stages=SALES_STAGES,
        stage_display=STAGE_DISPLAY,
    )


# =============================================================================
# API ENDPOINTS
# =============================================================================

@app.route("/api/pipeline")
def api_pipeline():
    """Get pipeline data as JSON."""
    return jsonify({
        "success": True,
        "data": process_analysis,
    })


@app.route("/api/prospect/<prospect_id>")
def api_prospect(prospect_id: str):
    """Get single prospect data."""
    for p in pipeline_data:
        if p.get("id") == prospect_id:
            return jsonify({"success": True, "data": p})
    
    return jsonify({"success": False, "error": "Prospect not found"}), 404


@app.route("/api/simulate", methods=["POST"])
def api_simulate():
    """Run simulation via API."""
    data = request.get_json()
    
    prospect_id = data.get("prospect_id")
    changes = data.get("changes", {})
    
    # Find timeline
    timeline = None
    for t in miner.deal_timelines:
        if t.prospect_id == prospect_id:
            timeline = t
            break
    
    if not timeline:
        return jsonify({"success": False, "error": "Prospect not found"}), 404
    
    transition_matrix = process_analysis.get("transition_matrix", {})
    results = miner.simulate_scenario(timeline, changes, transition_matrix)
    
    return jsonify({"success": True, "data": results})


@app.route("/api/refresh", methods=["POST"])
def api_refresh():
    """Refresh analysis."""
    refresh_analysis()
    return jsonify({"success": True, "message": "Analysis refreshed"})


# =============================================================================
# TEMPLATE FILTERS
# =============================================================================

@app.template_filter("stage_color")
def stage_color_filter(stage: str) -> str:
    """Return color class for a stage."""
    colors = {
        "lead": "slate",
        "discovery": "blue",
        "qualification": "indigo",
        "demo": "violet",
        "evaluation": "purple",
        "proposal": "fuchsia",
        "negotiation": "amber",
        "closed_won": "emerald",
        "closed_lost": "red",
    }
    return colors.get(stage, "gray")


@app.template_filter("severity_color")
def severity_color_filter(severity: str) -> str:
    """Return color class for severity."""
    colors = {
        "critical": "red",
        "high": "orange",
        "medium": "yellow",
        "low": "green",
    }
    return colors.get(severity, "gray")


@app.template_filter("format_date")
def format_date_filter(date_str: str) -> str:
    """Format ISO date string."""
    if not date_str:
        return "—"
    try:
        dt = datetime.fromisoformat(date_str.replace("Z", "+00:00"))
        return dt.strftime("%b %d, %Y")
    except:
        return date_str


# =============================================================================
# MAIN
# =============================================================================

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    debug = os.environ.get("FLASK_ENV") == "development"
    
    print(f"""
╔══════════════════════════════════════════════════════════════╗
║                                                              ║
║   🎯  S A L E S   I N T E L L I G E N C E                   ║
║                                                              ║
║   Process Mining • Pipeline Analytics • Deal Intelligence    ║
║                                                              ║
║   → http://localhost:{port}                                    ║
║                                                              ║
╚══════════════════════════════════════════════════════════════╝
    """)
    
    app.run(host="0.0.0.0", port=port, debug=debug)
