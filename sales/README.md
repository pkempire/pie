# 🎯 Sales Intelligence Platform

A premium sales intelligence system with process mining, pipeline analytics, and deal simulation capabilities.

![Dashboard](https://img.shields.io/badge/UI-Dark%20Theme-1a1a24)
![Python](https://img.shields.io/badge/Python-3.9+-3776ab)
![Flask](https://img.shields.io/badge/Flask-3.0+-000000)

## ✨ Features

### 📊 Process Mining
- **Stage Extraction**: Automatically extract sales process stages from unstructured transcripts
- **Markov Chain Analysis**: Build transition probability graphs across: Lead → Discovery → Qualification → Demo → Evaluation → Proposal → Negotiation → Close
- **Bottleneck Detection**: Identify stages where deals commonly stall or get lost
- **Conversion Funnel**: Visualize stage-by-stage conversion rates

### 🎨 Premium Web Interface
- **Dashboard**: Pipeline overview, funnel visualization, key metrics at a glance
- **Pipeline View**: Kanban-style board with deals grouped by stage
- **Prospect Detail**: Complete intelligence profile with personality traits, pain points, objections, buying signals
- **Simulation**: "What if" scenario modeling to predict deal outcomes
- **Upload**: Drag-and-drop transcript upload (txt, json, csv)

### 🔮 Simulation Engine
Model deal outcomes with scenarios like:
- **New Rep Assignment**: +15% win probability
- **Resolve Main Objection**: +25% win probability  
- **Gain Internal Champion**: +35% win probability
- **Sales Cycle Speedup**: Variable impact based on timing

## 🚀 Quick Start

### Prerequisites
- Python 3.9+
- pip

### Installation

```bash
# Navigate to sales directory
cd personal-intelligence-system/sales

# Create virtual environment (recommended)
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install flask werkzeug

# For full functionality (LLM extraction)
pip install -r ../requirements.txt
```

### Run the Application

```bash
# Development mode
python app.py

# Or with Flask CLI
FLASK_ENV=development flask run --port 5000
```

Open http://localhost:5000 in your browser.

## 📁 Project Structure

```
sales/
├── app.py                  # Flask web application
├── process_mining.py       # Process mining & Markov chain engine
├── extraction.py           # LLM-based entity extraction
├── uncertainty.py          # Uncertainty & contradiction detection
├── prospect_model.py       # Prospect world model
├── enrichment.py           # Web enrichment for prospects
├── demo.py                 # Demo data generation
├── templates/              # Jinja2 HTML templates
│   ├── base.html          # Base template with navigation
│   ├── dashboard.html     # Main dashboard view
│   ├── pipeline.html      # Kanban pipeline view
│   ├── prospect_detail.html # Individual prospect view
│   ├── process.html       # Process mining visualization
│   ├── simulation.html    # What-if simulation
│   └── upload.html        # File upload interface
├── output/                 # Stored prospect data
│   ├── prospect_*.json    # Individual prospect files
│   └── prospects.json     # Combined prospects
├── uploads/               # Uploaded transcripts
└── README.md              # This file
```

## 📖 Usage Guide

### 1. Dashboard
The main dashboard shows:
- **Key Metrics**: Total deals, active deals, stalled deals, stall rate
- **Conversion Funnel**: Visual representation of deals at each stage
- **Stage Distribution**: Breakdown of deals by current stage
- **Process Bottlenecks**: Stages with high stall or loss rates
- **Recent Deals**: Quick access to recent prospects

### 2. Pipeline View
Kanban-style board where you can:
- See all deals organized by stage
- Identify stalled deals (marked with amber badge)
- Click any deal to view full details
- Track deal progress visually

### 3. Prospect Detail
Complete intelligence profile including:
- **Personality Profile**: Traits, communication style, decision style
- **Pain Points**: Issues and their severity
- **Objections**: Concerns and resolution status
- **Buying Signals**: Positive indicators with strength ratings
- **Stakeholders**: Key people involved in the decision
- **Key Questions**: Uncertainties to resolve
- **Next Steps**: Action items with owners

### 4. Process Mining
Visualize your sales process:
- **Process Flow**: Markov chain visualization
- **Transition Matrix**: Stage-to-stage probabilities
- **Bottleneck Analysis**: Risk scores by stage
- **Conversion Chart**: Stage-by-stage conversion rates

### 5. Simulation
Model "what if" scenarios:
1. Select a prospect/deal
2. Toggle scenario variables (new rep, resolve objection, add champion)
3. Adjust sales cycle timing
4. View projected win probability change
5. Get AI-powered recommendations

### 6. Upload
Add new data to the system:
- Drag & drop or click to upload
- Supports `.txt` (transcripts), `.json` (structured data), `.csv` (bulk import)
- Automatic processing and analysis refresh

## 🔌 API Endpoints

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/api/pipeline` | GET | Full pipeline analysis data |
| `/api/prospect/<id>` | GET | Single prospect data |
| `/api/simulate` | POST | Run simulation scenario |
| `/api/refresh` | POST | Refresh analysis |

### Example: Run Simulation via API

```bash
curl -X POST http://localhost:5000/api/simulate \
  -H "Content-Type: application/json" \
  -d '{
    "prospect_id": "uuid-here",
    "changes": {
      "rep_change": true,
      "add_champion": true,
      "timing_speedup": 0.8
    }
  }'
```

## 🎨 Design System

The UI uses a custom dark theme with:
- **Background**: `#0a0a0f` (dark-900) to `#2d2d3a` (dark-500)
- **Accent**: Indigo (`#6366f1`) to Purple (`#a855f7`) gradient
- **Typography**: Inter (sans) + JetBrains Mono (code)
- **Stage Colors**: Each pipeline stage has a unique color from slate to amber

## 🧪 Demo Data

The system includes demo data for two real prospects:
- **Mateo Alvarez** (VP, Data & AI Operations @ GreenGrid Energy)
- **Nina Petrov** (Director, Data Platform & Security @ Aurora Biotech)

Additional demo pipeline data is auto-generated if no prospects exist.

## 🔧 Configuration

Environment variables:
- `FLASK_SECRET_KEY`: Session encryption key (set in production!)
- `FLASK_ENV`: Set to `development` for debug mode
- `PORT`: Server port (default: 5000)

## 📈 Future Enhancements

- [ ] Real-time LLM transcript processing
- [ ] CRM integrations (Salesforce, HubSpot)
- [ ] Team analytics and rep performance
- [ ] Email/meeting integration
- [ ] Custom stage configuration
- [ ] Export to PDF reports

## 🤝 Contributing

This is part of the Personal Intelligence System (PIE) project. Contributions welcome!

---

Built with ❤️ for sales teams who want to win more deals.
