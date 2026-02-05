"""
Realistic Demo Data for Sales Intelligence Platform

Hand-crafted B2B sales prospects across industries with realistic:
- Sales cycle timelines
- Stakeholder dynamics
- Objections and buying signals
- Process mining data showing stage transitions
"""

from datetime import datetime, timedelta
import uuid

# Base dates for realistic timelines
NOW = datetime.now()


def days_ago(n: int) -> str:
    """Return ISO date string for n days ago."""
    return (NOW - timedelta(days=n)).isoformat()


DEMO_PROSPECTS = [
    # ==========================================================================
    # DEAL 1: Hot SaaS deal in negotiation - likely to close
    # ==========================================================================
    {
        "id": "deal-001-nexgen",
        "name": "Marcus Chen",
        "title": "VP of Engineering",
        "company": "NexGen Analytics",
        "industry": "SaaS / Data Analytics",
        "email": "marcus.chen@nexgenanalytics.io",
        "deal_value": 185000,
        "deal_value_display": "$185,000 ARR",
        
        "personality_traits": ["data-driven", "technical", "decisive", "fast-mover"],
        "communication_style": "direct",
        "decision_style": "ROI-focused",
        "preferred_contact": "Slack, then email",
        "meeting_preference": "30-min focused calls, no fluff",
        
        "pain_points": [
            {
                "description": "Current observability stack (Datadog + custom) costs $400K/year and engineers still can't debug production issues quickly",
                "severity": "critical",
                "category": "cost",
                "business_impact": "3-4 hours average incident resolution time, causing SLA breaches",
                "meeting_number": 1,
            },
            {
                "description": "Data pipeline reliability is hurting customer trust - 2 major outages last quarter",
                "severity": "critical",
                "category": "reliability",
                "business_impact": "Lost $2M deal because prospect saw status page during demo",
                "meeting_number": 2,
            },
        ],
        
        "objections": [
            {
                "description": "Concerned about migration complexity from existing Datadog setup",
                "type": "technical",
                "severity": "concern",
                "resolution_status": "resolved",
                "resolution_notes": "Showed parallel-run capability, they can keep Datadog during transition",
                "meeting_number": 3,
            },
            {
                "description": "CFO wants to see 12-month ROI projection before signing",
                "type": "budget",
                "severity": "concern",
                "resolution_status": "partially_resolved",
                "resolution_notes": "Sent ROI calculator, Marcus is building internal business case",
                "meeting_number": 4,
            },
        ],
        
        "buying_signals": [
            {
                "signal": "Asked for reference calls with similar-sized SaaS companies",
                "strength": "strong",
                "type": "validation",
                "meeting_number": 3,
            },
            {
                "signal": "Introduced us to their CFO without being asked",
                "strength": "strong", 
                "type": "champion",
                "meeting_number": 4,
            },
            {
                "signal": "Mentioned they need to make a decision before Q2 budget lock",
                "strength": "strong",
                "type": "timeline",
                "meeting_number": 5,
            },
            {
                "signal": "Engineering team already created Slack channel #our-product-eval",
                "strength": "moderate",
                "type": "adoption",
                "meeting_number": 4,
            },
        ],
        
        "stakeholders": [
            {
                "name": "Marcus Chen",
                "role": "VP Engineering",
                "influence": "champion",
                "sentiment": "strong_positive",
                "notes": "Driving this initiative, wants to be hero who fixes reliability",
            },
            {
                "name": "Diana Reyes",
                "role": "CFO",
                "influence": "economic_buyer",
                "sentiment": "neutral_positive",
                "notes": "Cares about cost savings, needs ROI proof. Met in meeting 4.",
            },
            {
                "name": "Tom Bradley",
                "role": "CTO",
                "influence": "decision_maker",
                "sentiment": "positive",
                "notes": "Trusts Marcus. Will sign if CFO approves budget.",
            },
            {
                "name": "Aisha Patel",
                "role": "Senior SRE",
                "influence": "evaluator",
                "sentiment": "positive",
                "notes": "Did technical eval, loved the product. Key internal advocate.",
            },
        ],
        
        "competitive_context": [
            {
                "competitor": "Datadog",
                "status": "incumbent",
                "notes": "Currently paying $400K/year. Contract renews in 3 months.",
            },
            {
                "competitor": "Honeycomb",
                "status": "evaluated",
                "notes": "Looked at 6 months ago, found it too expensive for their scale",
            },
        ],
        
        "next_steps": [
            {
                "action": "Send final contract with negotiated 15% discount",
                "owner": "rep",
                "due_date": days_ago(-2),
                "commitment_level": "firm",
            },
            {
                "action": "Marcus presents ROI case to exec team Monday",
                "owner": "prospect",
                "due_date": days_ago(-5),
                "commitment_level": "firm",
            },
        ],
        
        "meeting_metadata": {
            "meeting_count": 5,
            "overall_deal_stage": "negotiation",
            "momentum": "accelerating",
            "meetings": [
                {"number": 1, "date": days_ago(45), "topic": "Discovery - observability pain", "tone": "engaged", "duration_min": 45},
                {"number": 2, "date": days_ago(38), "topic": "Technical deep-dive with SRE team", "tone": "technical", "duration_min": 60},
                {"number": 3, "date": days_ago(28), "topic": "Demo + POC planning", "tone": "excited", "duration_min": 45},
                {"number": 4, "date": days_ago(14), "topic": "POC results + CFO intro", "tone": "positive", "duration_min": 30},
                {"number": 5, "date": days_ago(5), "topic": "Pricing negotiation", "tone": "collaborative", "duration_min": 25},
            ],
        },
        
        "stage_transitions": [
            {"stage": "lead", "entered": days_ago(52), "exited": days_ago(45), "days": 7},
            {"stage": "discovery", "entered": days_ago(45), "exited": days_ago(38), "days": 7},
            {"stage": "qualification", "entered": days_ago(38), "exited": days_ago(30), "days": 8},
            {"stage": "demo", "entered": days_ago(30), "exited": days_ago(28), "days": 2},
            {"stage": "evaluation", "entered": days_ago(28), "exited": days_ago(14), "days": 14},
            {"stage": "proposal", "entered": days_ago(14), "exited": days_ago(7), "days": 7},
            {"stage": "negotiation", "entered": days_ago(7), "exited": None, "days": 7},
        ],
        
        "deal_notes": "Strong deal. Marcus is a great champion. Main risk is CFO wanting more discount. Hold firm at 15%.",
    },
    
    # ==========================================================================
    # DEAL 2: Healthcare - stuck in evaluation (compliance concerns)
    # ==========================================================================
    {
        "id": "deal-002-meridian",
        "name": "Dr. Sarah Okonkwo",
        "title": "Chief Medical Information Officer",
        "company": "Meridian Health Systems",
        "industry": "Healthcare",
        "email": "sokonkwo@meridianhealth.org",
        "deal_value": 340000,
        "deal_value_display": "$340,000 ARR",
        
        "personality_traits": ["methodical", "risk-averse", "consensus-builder", "patient-focused"],
        "communication_style": "formal",
        "decision_style": "committee-driven",
        "preferred_contact": "Email with detailed follow-up",
        "meeting_preference": "Scheduled calls with agenda sent 48h in advance",
        
        "pain_points": [
            {
                "description": "EHR data integration is fragmented across 12 different systems",
                "severity": "critical",
                "category": "integration",
                "business_impact": "Nurses spend 2+ hours per shift on duplicate data entry",
                "meeting_number": 1,
            },
            {
                "description": "Patient readmission predictions are inaccurate, hurting CMS quality scores",
                "severity": "high",
                "category": "analytics",
                "business_impact": "Lost $3.2M in value-based care bonuses last year",
                "meeting_number": 2,
            },
            {
                "description": "IT team is stretched thin managing legacy integrations",
                "severity": "medium",
                "category": "efficiency",
                "business_impact": "6-month backlog on integration requests from clinical teams",
                "meeting_number": 1,
            },
        ],
        
        "objections": [
            {
                "description": "HIPAA BAA terms need review by our legal team - they're backed up 8 weeks",
                "type": "legal",
                "severity": "hard_blocker",
                "resolution_status": "unresolved",
                "resolution_notes": "Escalated to their GC. Offered to have our HIPAA counsel do joint review.",
                "meeting_number": 4,
            },
            {
                "description": "Need to validate SOC 2 Type II and HITRUST certification",
                "type": "compliance",
                "severity": "concern",
                "resolution_status": "resolved",
                "resolution_notes": "Sent full audit reports. Security team approved.",
                "meeting_number": 3,
            },
            {
                "description": "Board wants to see case study from similar health system",
                "type": "validation",
                "severity": "concern",
                "resolution_status": "partially_resolved",
                "resolution_notes": "Connected them with Lakeside Medical (similar size). Call scheduled for next week.",
                "meeting_number": 5,
            },
        ],
        
        "buying_signals": [
            {
                "signal": "CMIO asked about multi-year pricing and implementation timeline",
                "strength": "strong",
                "type": "timeline",
                "meeting_number": 2,
            },
            {
                "signal": "IT Director spent 3 hours in technical sandbox unprompted",
                "strength": "strong",
                "type": "adoption",
                "meeting_number": 3,
            },
            {
                "signal": "Asked if we can present to their board in Q2",
                "strength": "moderate",
                "type": "process",
                "meeting_number": 5,
            },
        ],
        
        "stakeholders": [
            {
                "name": "Dr. Sarah Okonkwo",
                "role": "Chief Medical Information Officer",
                "influence": "champion",
                "sentiment": "positive",
                "notes": "Clinical leader. Sees this as key to patient outcomes improvement.",
            },
            {
                "name": "Robert Kim",
                "role": "CIO",
                "influence": "decision_maker",
                "sentiment": "neutral",
                "notes": "Cautious. Burned by failed Epic implementation 3 years ago. Needs convincing.",
            },
            {
                "name": "Janet Morrison",
                "role": "General Counsel",
                "influence": "blocker",
                "sentiment": "unknown",
                "notes": "Haven't met directly. She's the bottleneck on BAA review. Very risk-averse.",
            },
            {
                "name": "David Park",
                "role": "VP Finance",
                "influence": "economic_buyer",
                "sentiment": "positive",
                "notes": "Excited about ROI from reduced readmissions. Strong ally.",
            },
            {
                "name": "Michelle Torres",
                "role": "IT Director",
                "influence": "evaluator",
                "sentiment": "strong_positive",
                "notes": "Technical champion. Wants us to win. Very helpful in security review.",
            },
        ],
        
        "competitive_context": [
            {
                "competitor": "Epic (internal expansion)",
                "status": "alternative",
                "notes": "CIO's fallback is to expand Epic. More expensive, but 'safe' choice.",
            },
        ],
        
        "next_steps": [
            {
                "action": "Reference call between their CIO and Lakeside Medical CIO",
                "owner": "rep",
                "due_date": days_ago(-7),
                "commitment_level": "firm",
            },
            {
                "action": "Follow up with GC office on BAA timeline",
                "owner": "rep",
                "due_date": days_ago(-1),
                "commitment_level": "firm",
            },
            {
                "action": "Dr. Okonkwo to champion internally for board presentation slot",
                "owner": "prospect",
                "due_date": days_ago(-14),
                "commitment_level": "tentative",
            },
        ],
        
        "meeting_metadata": {
            "meeting_count": 5,
            "overall_deal_stage": "evaluation",
            "momentum": "stalling",
            "meetings": [
                {"number": 1, "date": days_ago(75), "topic": "Discovery - data integration challenges", "tone": "engaged", "duration_min": 60},
                {"number": 2, "date": days_ago(60), "topic": "Clinical use case deep-dive", "tone": "excited", "duration_min": 90},
                {"number": 3, "date": days_ago(45), "topic": "Technical/security review with IT", "tone": "thorough", "duration_min": 120},
                {"number": 4, "date": days_ago(30), "topic": "Compliance requirements discussion", "tone": "cautious", "duration_min": 45},
                {"number": 5, "date": days_ago(21), "topic": "Deal progress check-in", "tone": "hopeful", "duration_min": 30},
            ],
        },
        
        "stage_transitions": [
            {"stage": "lead", "entered": days_ago(82), "exited": days_ago(75), "days": 7},
            {"stage": "discovery", "entered": days_ago(75), "exited": days_ago(62), "days": 13},
            {"stage": "qualification", "entered": days_ago(62), "exited": days_ago(50), "days": 12},
            {"stage": "demo", "entered": days_ago(50), "exited": days_ago(45), "days": 5},
            {"stage": "evaluation", "entered": days_ago(45), "exited": None, "days": 45},  # STALLED
        ],
        
        "deal_notes": "Deal is stuck in legal review. CMIO is champion but GC is bottleneck. Need to find way to accelerate BAA review or deal may die. Healthcare sales cycles are long but this is pushing even those limits.",
    },
    
    # ==========================================================================
    # DEAL 3: Manufacturing - early stage, high potential
    # ==========================================================================
    {
        "id": "deal-003-steelworks",
        "name": "James Kowalski",
        "title": "Director of Operations",
        "company": "Great Lakes Steelworks",
        "industry": "Manufacturing",
        "email": "jkowalski@greatlakessteelworks.com",
        "deal_value": 95000,
        "deal_value_display": "$95,000 ARR",
        
        "personality_traits": ["pragmatic", "no-nonsense", "cost-conscious", "hands-on"],
        "communication_style": "direct",
        "decision_style": "ROI-focused",
        "preferred_contact": "Phone calls, early morning",
        "meeting_preference": "On-site visits preferred. Wants to see things work.",
        
        "pain_points": [
            {
                "description": "Production line sensors generate data but nobody analyzes it in real-time",
                "severity": "high",
                "category": "analytics",
                "business_impact": "Missed 3 preventable equipment failures last year, $800K in repairs + downtime",
                "meeting_number": 1,
            },
            {
                "description": "Quality control is manual and reactive - catching defects too late",
                "severity": "critical",
                "category": "quality",
                "business_impact": "2% scrap rate, industry best is 0.5%. Losing $1.2M/year to waste.",
                "meeting_number": 2,
            },
        ],
        
        "objections": [
            {
                "description": "Our IT team is tiny - just 2 people. Can we really implement this?",
                "type": "resource",
                "severity": "concern",
                "resolution_status": "unresolved",
                "resolution_notes": "Need to show managed service options and white-glove implementation",
                "meeting_number": 2,
            },
            {
                "description": "Plant floor has spotty WiFi and legacy PLCs",
                "type": "technical",
                "severity": "concern",
                "resolution_status": "partially_resolved",
                "resolution_notes": "Discussed edge deployment options. Need site survey.",
                "meeting_number": 2,
            },
        ],
        
        "buying_signals": [
            {
                "signal": "Shared actual production data for us to analyze",
                "strength": "strong",
                "type": "trust",
                "meeting_number": 2,
            },
            {
                "signal": "Mentioned CFO is pushing digital transformation initiative",
                "strength": "moderate",
                "type": "budget",
                "meeting_number": 1,
            },
        ],
        
        "stakeholders": [
            {
                "name": "James Kowalski",
                "role": "Director of Operations",
                "influence": "champion",
                "sentiment": "positive",
                "notes": "Old-school but sees the value. Needs simple ROI story.",
            },
            {
                "name": "Patricia Reeves",
                "role": "CFO",
                "influence": "economic_buyer",
                "sentiment": "unknown",
                "notes": "Haven't met. James says she's driving digital spend. Key to unlock budget.",
            },
            {
                "name": "Steve Brennan",
                "role": "Plant Manager",
                "influence": "evaluator",
                "sentiment": "skeptical",
                "notes": "30-year veteran. Doesn't trust 'tech solutions'. Need to win him over.",
            },
        ],
        
        "competitive_context": [
            {
                "competitor": "Status quo (manual)",
                "status": "incumbent",
                "notes": "They've always done it this way. Biggest competitor is inertia.",
            },
            {
                "competitor": "Siemens MindSphere",
                "status": "researching",
                "notes": "James mentioned they got a cold call. Too expensive and complex per James.",
            },
        ],
        
        "next_steps": [
            {
                "action": "Schedule on-site visit to plant floor",
                "owner": "rep",
                "due_date": days_ago(-10),
                "commitment_level": "firm",
            },
            {
                "action": "Prepare custom ROI analysis using their production data",
                "owner": "rep",
                "due_date": days_ago(-7),
                "commitment_level": "firm",
            },
            {
                "action": "James to get meeting with CFO on calendar",
                "owner": "prospect",
                "due_date": days_ago(-14),
                "commitment_level": "tentative",
            },
        ],
        
        "meeting_metadata": {
            "meeting_count": 2,
            "overall_deal_stage": "qualification",
            "momentum": "steady",
            "meetings": [
                {"number": 1, "date": days_ago(18), "topic": "Intro - predictive maintenance interest", "tone": "curious", "duration_min": 30},
                {"number": 2, "date": days_ago(8), "topic": "Technical discovery + data review", "tone": "engaged", "duration_min": 60},
            ],
        },
        
        "stage_transitions": [
            {"stage": "lead", "entered": days_ago(25), "exited": days_ago(18), "days": 7},
            {"stage": "discovery", "entered": days_ago(18), "exited": days_ago(10), "days": 8},
            {"stage": "qualification", "entered": days_ago(10), "exited": None, "days": 10},
        ],
        
        "deal_notes": "Classic manufacturing deal. Long sales cycle but sticky once won. Key is winning over the plant manager and showing ROI in terms he understands (uptime, scrap rate). CFO meeting is critical next step.",
    },
    
    # ==========================================================================
    # DEAL 4: FinTech - fast-moving, competitive situation
    # ==========================================================================
    {
        "id": "deal-004-paystream",
        "name": "Priya Sharma",
        "title": "Head of Data Engineering",
        "company": "PayStream Technologies",
        "industry": "FinTech",
        "email": "priya.sharma@paystream.io",
        "deal_value": 275000,
        "deal_value_display": "$275,000 ARR",
        
        "personality_traits": ["analytical", "competitive", "innovative", "fast-paced"],
        "communication_style": "technical",
        "decision_style": "data-driven",
        "preferred_contact": "Slack DMs or quick Zoom calls",
        "meeting_preference": "Async video updates between calls",
        
        "pain_points": [
            {
                "description": "Real-time fraud detection latency is 800ms, competitors are at 50ms",
                "severity": "critical",
                "category": "performance",
                "business_impact": "Losing enterprise deals to competitors who can demo faster detection",
                "meeting_number": 1,
            },
            {
                "description": "Data warehouse costs $180K/month and queries still take minutes",
                "severity": "high",
                "category": "cost",
                "business_impact": "Finance team can't get same-day reports for reconciliation",
                "meeting_number": 2,
            },
            {
                "description": "SOC 2 audit found data lineage gaps",
                "severity": "high",
                "category": "compliance",
                "business_impact": "Need to remediate before next audit in 4 months",
                "meeting_number": 3,
            },
        ],
        
        "objections": [
            {
                "description": "Your pricing is 20% higher than Competitor X",
                "type": "budget",
                "severity": "hard_blocker",
                "resolution_status": "partially_resolved",
                "resolution_notes": "Showed TCO comparison including operational overhead they'd need with Competitor X. Priya is building internal comparison.",
                "meeting_number": 4,
            },
            {
                "description": "Can you match Competitor X's latency guarantees?",
                "type": "technical",
                "severity": "concern",
                "resolution_status": "resolved",
                "resolution_notes": "Ran benchmark on their data. Our p99 was actually better. Shared results.",
                "meeting_number": 3,
            },
        ],
        
        "buying_signals": [
            {
                "signal": "Asked about annual vs monthly billing (annual saves money)",
                "strength": "strong",
                "type": "budget",
                "meeting_number": 4,
            },
            {
                "signal": "Invited their VP of Eng to next call",
                "strength": "strong",
                "type": "champion",
                "meeting_number": 3,
            },
            {
                "signal": "Shared that Competitor X's POC had issues",
                "strength": "strong",
                "type": "competitive",
                "meeting_number": 4,
            },
            {
                "signal": "Asked about customer success and dedicated support",
                "strength": "moderate",
                "type": "commitment",
                "meeting_number": 4,
            },
        ],
        
        "stakeholders": [
            {
                "name": "Priya Sharma",
                "role": "Head of Data Engineering",
                "influence": "champion",
                "sentiment": "positive",
                "notes": "Technical champion. Prefers us but under pressure to justify premium.",
            },
            {
                "name": "Kevin O'Brien",
                "role": "VP of Engineering",
                "influence": "decision_maker",
                "sentiment": "neutral",
                "notes": "Numbers guy. Will go with best TCO story. Joining next call.",
            },
            {
                "name": "Lisa Chang",
                "role": "CTO",
                "influence": "economic_buyer",
                "sentiment": "unknown",
                "notes": "Final sign-off. Very strategic thinker. Priya says she values innovation.",
            },
        ],
        
        "competitive_context": [
            {
                "competitor": "Competitor X (Databricks)",
                "status": "active_evaluation",
                "notes": "Running parallel POC. Their POC had reliability issues per Priya.",
            },
            {
                "competitor": "Snowflake",
                "status": "incumbent",
                "notes": "Current warehouse. Will keep for some workloads regardless.",
            },
        ],
        
        "next_steps": [
            {
                "action": "VP of Eng call - focus on TCO and strategic roadmap",
                "owner": "both",
                "due_date": days_ago(-3),
                "commitment_level": "firm",
            },
            {
                "action": "Send custom pricing proposal with annual discount",
                "owner": "rep",
                "due_date": days_ago(-1),
                "commitment_level": "firm",
            },
        ],
        
        "meeting_metadata": {
            "meeting_count": 4,
            "overall_deal_stage": "proposal",
            "momentum": "accelerating",
            "meetings": [
                {"number": 1, "date": days_ago(28), "topic": "Intro - real-time processing needs", "tone": "interested", "duration_min": 30},
                {"number": 2, "date": days_ago(21), "topic": "Technical architecture review", "tone": "technical", "duration_min": 60},
                {"number": 3, "date": days_ago(14), "topic": "POC kickoff + benchmark planning", "tone": "collaborative", "duration_min": 45},
                {"number": 4, "date": days_ago(5), "topic": "POC results + pricing discussion", "tone": "positive", "duration_min": 40},
            ],
        },
        
        "stage_transitions": [
            {"stage": "lead", "entered": days_ago(35), "exited": days_ago(28), "days": 7},
            {"stage": "discovery", "entered": days_ago(28), "exited": days_ago(22), "days": 6},
            {"stage": "qualification", "entered": days_ago(22), "exited": days_ago(18), "days": 4},
            {"stage": "demo", "entered": days_ago(18), "exited": days_ago(14), "days": 4},
            {"stage": "evaluation", "entered": days_ago(14), "exited": days_ago(5), "days": 9},
            {"stage": "proposal", "entered": days_ago(5), "exited": None, "days": 5},
        ],
        
        "deal_notes": "Hot competitive deal. Databricks is main threat but their POC had issues. Need to nail the VP of Eng call with strong TCO story. Priya is great champion but needs ammunition to justify premium pricing internally.",
    },
    
    # ==========================================================================
    # DEAL 5: E-commerce - CLOSED WON 🎉
    # ==========================================================================
    {
        "id": "deal-005-urbanstyle",
        "name": "Rachel Torres",
        "title": "VP of Technology",
        "company": "UrbanStyle Retail",
        "industry": "E-commerce",
        "email": "rachel.torres@urbanstyle.com",
        "deal_value": 125000,
        "deal_value_display": "$125,000 ARR",
        
        "personality_traits": ["collaborative", "customer-obsessed", "innovative", "decisive"],
        "communication_style": "warm",
        "decision_style": "fast-mover",
        "preferred_contact": "Email or text",
        "meeting_preference": "Video calls, casual vibe",
        
        "pain_points": [
            {
                "description": "Personalization engine recommendations have 2% CTR, industry avg is 5%",
                "severity": "critical",
                "category": "analytics",
                "business_impact": "Leaving $4M+ in annual revenue on the table",
                "meeting_number": 1,
            },
            {
                "description": "Inventory forecasting is inaccurate - 15% overstock on slow items",
                "severity": "high",
                "category": "operations",
                "business_impact": "$2M tied up in excess inventory, heavy discounting to clear",
                "meeting_number": 2,
            },
        ],
        
        "objections": [
            {
                "description": "Worried about Black Friday readiness - can't risk new system during peak",
                "type": "timing",
                "severity": "concern",
                "resolution_status": "resolved",
                "resolution_notes": "Proposed phased rollout: non-critical features first, full go-live after Jan 15",
                "meeting_number": 3,
            },
        ],
        
        "buying_signals": [
            {
                "signal": "Asked to accelerate timeline to beat Q1 planning cycle",
                "strength": "strong",
                "type": "timeline",
                "meeting_number": 4,
            },
            {
                "signal": "CEO joined final call to say thanks and share vision",
                "strength": "strong",
                "type": "executive_sponsorship",
                "meeting_number": 5,
            },
        ],
        
        "stakeholders": [
            {
                "name": "Rachel Torres",
                "role": "VP of Technology",
                "influence": "champion",
                "sentiment": "strong_positive",
                "notes": "Drove the whole deal. Great partner.",
            },
            {
                "name": "Michael Huang",
                "role": "CEO",
                "influence": "economic_buyer",
                "sentiment": "positive",
                "notes": "Trusted Rachel completely. Wanted quick win for board.",
            },
            {
                "name": "Jennifer Walsh",
                "role": "CMO",
                "influence": "influencer",
                "sentiment": "positive",
                "notes": "Excited about personalization improvements.",
            },
        ],
        
        "competitive_context": [
            {
                "competitor": "Algolia",
                "status": "lost_early",
                "notes": "Evaluated but too narrow in scope for their needs",
            },
        ],
        
        "next_steps": [
            {
                "action": "Kickoff call scheduled",
                "owner": "both",
                "due_date": days_ago(-7),
                "commitment_level": "firm",
            },
        ],
        
        "meeting_metadata": {
            "meeting_count": 5,
            "overall_deal_stage": "closed_won",
            "momentum": "closed",
            "closed_date": days_ago(3),
            "meetings": [
                {"number": 1, "date": days_ago(42), "topic": "Discovery - personalization challenges", "tone": "excited", "duration_min": 45},
                {"number": 2, "date": days_ago(35), "topic": "Technical review + use cases", "tone": "collaborative", "duration_min": 60},
                {"number": 3, "date": days_ago(25), "topic": "Demo + implementation planning", "tone": "positive", "duration_min": 50},
                {"number": 4, "date": days_ago(14), "topic": "Pricing + timeline negotiation", "tone": "businesslike", "duration_min": 35},
                {"number": 5, "date": days_ago(3), "topic": "Contract signing + CEO intro", "tone": "celebratory", "duration_min": 20},
            ],
        },
        
        "stage_transitions": [
            {"stage": "lead", "entered": days_ago(50), "exited": days_ago(42), "days": 8},
            {"stage": "discovery", "entered": days_ago(42), "exited": days_ago(36), "days": 6},
            {"stage": "qualification", "entered": days_ago(36), "exited": days_ago(30), "days": 6},
            {"stage": "demo", "entered": days_ago(30), "exited": days_ago(25), "days": 5},
            {"stage": "evaluation", "entered": days_ago(25), "exited": days_ago(18), "days": 7},
            {"stage": "proposal", "entered": days_ago(18), "exited": days_ago(10), "days": 8},
            {"stage": "negotiation", "entered": days_ago(10), "exited": days_ago(3), "days": 7},
            {"stage": "closed_won", "entered": days_ago(3), "exited": None, "days": 3},
        ],
        
        "outcome": "closed_won",
        "deal_notes": "Textbook deal. Strong champion, clear ROI, fast decision-making. Rachel was a pleasure to work with. Good reference potential.",
    },
    
    # ==========================================================================
    # DEAL 6: CLOSED LOST - Lost to competitor
    # ==========================================================================
    {
        "id": "deal-006-capitaledge",
        "name": "Thomas Brennan",
        "title": "CTO",
        "company": "Capital Edge Partners",
        "industry": "Financial Services",
        "email": "tbrennan@capitaledge.com",
        "deal_value": 420000,
        "deal_value_display": "$420,000 ARR",
        
        "personality_traits": ["conservative", "detail-oriented", "relationship-driven", "slow-moving"],
        "communication_style": "formal",
        "decision_style": "consensus-builder",
        "preferred_contact": "Email with formal tone",
        "meeting_preference": "In-person when possible",
        
        "pain_points": [
            {
                "description": "Risk modeling takes 4 hours to run, need real-time",
                "severity": "critical",
                "category": "performance",
                "business_impact": "Missing trading opportunities due to slow risk calculations",
                "meeting_number": 1,
            },
            {
                "description": "Data silos between trading desks",
                "severity": "high",
                "category": "integration",
                "business_impact": "Duplicate positions and compliance gaps",
                "meeting_number": 2,
            },
        ],
        
        "objections": [
            {
                "description": "Our incumbent vendor has 15-year relationship with our CEO",
                "type": "relationship",
                "severity": "hard_blocker",
                "resolution_status": "unresolved",
                "resolution_notes": "Could never get direct CEO access. Thomas tried but CEO deferred to existing vendor.",
                "meeting_number": 5,
            },
            {
                "description": "Switching costs are very high - need to retrain 200 analysts",
                "type": "process",
                "severity": "hard_blocker",
                "resolution_status": "unresolved",
                "resolution_notes": "Offered free training program but wasn't enough to overcome inertia",
                "meeting_number": 4,
            },
        ],
        
        "buying_signals": [
            {
                "signal": "Thomas personally championed us in vendor review committee",
                "strength": "strong",
                "type": "champion",
                "meeting_number": 3,
            },
            {
                "signal": "Got to final two vendors",
                "strength": "moderate",
                "type": "process",
                "meeting_number": 5,
            },
        ],
        
        "stakeholders": [
            {
                "name": "Thomas Brennan",
                "role": "CTO",
                "influence": "champion",
                "sentiment": "positive",
                "notes": "Wanted us to win. Didn't have enough political capital.",
            },
            {
                "name": "William Hayes",
                "role": "CEO",
                "influence": "decision_maker",
                "sentiment": "negative",
                "notes": "Long relationship with incumbent vendor CEO. Golfing buddies.",
            },
            {
                "name": "Margaret Chen",
                "role": "CFO",
                "influence": "economic_buyer",
                "sentiment": "neutral",
                "notes": "Focused on costs. We were actually cheaper but didn't matter.",
            },
        ],
        
        "competitive_context": [
            {
                "competitor": "Bloomberg (incumbent)",
                "status": "won",
                "notes": "15-year relationship. CEO-to-CEO connection we couldn't overcome.",
            },
        ],
        
        "next_steps": [],
        
        "meeting_metadata": {
            "meeting_count": 6,
            "overall_deal_stage": "closed_lost",
            "momentum": "closed",
            "closed_date": days_ago(15),
            "loss_reason": "Incumbent relationship - CEO-level loyalty to Bloomberg",
            "meetings": [
                {"number": 1, "date": days_ago(95), "topic": "Discovery - risk modeling pain", "tone": "interested", "duration_min": 45},
                {"number": 2, "date": days_ago(82), "topic": "Technical deep-dive", "tone": "engaged", "duration_min": 75},
                {"number": 3, "date": days_ago(68), "topic": "Demo to broader team", "tone": "positive", "duration_min": 90},
                {"number": 4, "date": days_ago(50), "topic": "POC review + pricing", "tone": "cautious", "duration_min": 60},
                {"number": 5, "date": days_ago(35), "topic": "Vendor committee presentation", "tone": "formal", "duration_min": 60},
                {"number": 6, "date": days_ago(15), "topic": "Decision call - lost", "tone": "disappointing", "duration_min": 15},
            ],
        },
        
        "stage_transitions": [
            {"stage": "lead", "entered": days_ago(100), "exited": days_ago(95), "days": 5},
            {"stage": "discovery", "entered": days_ago(95), "exited": days_ago(85), "days": 10},
            {"stage": "qualification", "entered": days_ago(85), "exited": days_ago(75), "days": 10},
            {"stage": "demo", "entered": days_ago(75), "exited": days_ago(68), "days": 7},
            {"stage": "evaluation", "entered": days_ago(68), "exited": days_ago(45), "days": 23},
            {"stage": "proposal", "entered": days_ago(45), "exited": days_ago(35), "days": 10},
            {"stage": "negotiation", "entered": days_ago(35), "exited": days_ago(15), "days": 20},
            {"stage": "closed_lost", "entered": days_ago(15), "exited": None, "days": 15},
        ],
        
        "outcome": "closed_lost",
        "deal_notes": "Lost to incumbent relationship. Thomas did everything right as a champion but CEO had personal loyalty to Bloomberg. Lesson: need to identify CEO-level relationships early. Could have disqualified earlier and saved 3 months.",
    },
    
    # ==========================================================================
    # DEAL 7: SaaS - Discovery stage, promising
    # ==========================================================================
    {
        "id": "deal-007-cloudshift",
        "name": "Amanda Liu",
        "title": "Director of Platform Engineering",
        "company": "CloudShift Solutions",
        "industry": "SaaS / DevOps",
        "email": "amanda.liu@cloudshift.dev",
        "deal_value": 78000,
        "deal_value_display": "$78,000 ARR",
        
        "personality_traits": ["technical", "curious", "builder-mindset", "collaborative"],
        "communication_style": "casual-technical",
        "decision_style": "bottom-up",
        "preferred_contact": "Slack or Twitter DM",
        "meeting_preference": "Pairing sessions over formal demos",
        
        "pain_points": [
            {
                "description": "Deployment pipeline takes 45 minutes, developers are frustrated",
                "severity": "high",
                "category": "efficiency",
                "business_impact": "Engineers shipping less frequently, velocity down 30%",
                "meeting_number": 1,
            },
        ],
        
        "objections": [],
        
        "buying_signals": [
            {
                "signal": "Already tested our open-source version in personal project",
                "strength": "strong",
                "type": "adoption",
                "meeting_number": 1,
            },
            {
                "signal": "Asked about team/enterprise pricing unprompted",
                "strength": "moderate",
                "type": "budget",
                "meeting_number": 1,
            },
        ],
        
        "stakeholders": [
            {
                "name": "Amanda Liu",
                "role": "Director of Platform Engineering",
                "influence": "champion",
                "sentiment": "strong_positive",
                "notes": "Already a fan. Needs help making business case.",
            },
            {
                "name": "Unknown",
                "role": "VP Engineering",
                "influence": "decision_maker",
                "sentiment": "unknown",
                "notes": "Haven't met yet. Amanda says he's supportive of platform investments.",
            },
        ],
        
        "competitive_context": [
            {
                "competitor": "CircleCI (current)",
                "status": "incumbent",
                "notes": "Currently using CircleCI. Pain is real but switching has friction.",
            },
        ],
        
        "next_steps": [
            {
                "action": "Technical deep-dive call with Amanda's team",
                "owner": "both",
                "due_date": days_ago(-5),
                "commitment_level": "firm",
            },
            {
                "action": "Amanda to quantify pipeline time costs internally",
                "owner": "prospect",
                "due_date": days_ago(-10),
                "commitment_level": "tentative",
            },
        ],
        
        "meeting_metadata": {
            "meeting_count": 1,
            "overall_deal_stage": "discovery",
            "momentum": "accelerating",
            "meetings": [
                {"number": 1, "date": days_ago(5), "topic": "Intro - she found us on Twitter", "tone": "enthusiastic", "duration_min": 40},
            ],
        },
        
        "stage_transitions": [
            {"stage": "lead", "entered": days_ago(8), "exited": days_ago(5), "days": 3},
            {"stage": "discovery", "entered": days_ago(5), "exited": None, "days": 5},
        ],
        
        "deal_notes": "Inbound from product-led growth. Amanda is already technical champion. Need to help her build business case and get to VP. Good land-and-expand potential.",
    },
    
    # ==========================================================================
    # DEAL 8: Healthcare - CLOSED LOST (budget cut)
    # ==========================================================================
    {
        "id": "deal-008-wellness",
        "name": "Dr. Michael Foster",
        "title": "Chief Digital Officer",
        "company": "Wellness Partners Network",
        "industry": "Healthcare",
        "email": "mfoster@wellnesspartners.health",
        "deal_value": 210000,
        "deal_value_display": "$210,000 ARR",
        
        "personality_traits": ["visionary", "innovative", "executive-presence", "strategic"],
        "communication_style": "polished",
        "decision_style": "strategic",
        "preferred_contact": "EA schedules everything",
        "meeting_preference": "Exec briefings, short and strategic",
        
        "pain_points": [
            {
                "description": "Patient engagement app has 12% monthly active rate, target is 40%",
                "severity": "critical",
                "category": "product",
                "business_impact": "Board pressure on digital health metrics",
                "meeting_number": 1,
            },
        ],
        
        "objections": [
            {
                "description": "Board just cut all discretionary tech spend due to reimbursement cuts",
                "type": "budget",
                "severity": "hard_blocker",
                "resolution_status": "unresolved",
                "resolution_notes": "External factor. Budget frozen for at least 6 months.",
                "meeting_number": 4,
            },
        ],
        
        "buying_signals": [
            {
                "signal": "CDO was personally sponsoring the project",
                "strength": "strong",
                "type": "champion",
                "meeting_number": 2,
            },
        ],
        
        "stakeholders": [
            {
                "name": "Dr. Michael Foster",
                "role": "Chief Digital Officer",
                "influence": "champion",
                "sentiment": "positive",
                "notes": "Wanted this badly. Budget got cut from under him.",
            },
            {
                "name": "Board of Directors",
                "role": "Economic Buyer",
                "influence": "decision_maker",
                "sentiment": "negative",
                "notes": "Froze all discretionary spend. External reimbursement pressures.",
            },
        ],
        
        "competitive_context": [],
        
        "next_steps": [
            {
                "action": "Reconnect in 6 months when budget cycle resets",
                "owner": "rep",
                "due_date": days_ago(-180),
                "commitment_level": "tentative",
            },
        ],
        
        "meeting_metadata": {
            "meeting_count": 4,
            "overall_deal_stage": "closed_lost",
            "momentum": "closed",
            "closed_date": days_ago(22),
            "loss_reason": "Budget freeze - external healthcare reimbursement cuts",
            "meetings": [
                {"number": 1, "date": days_ago(65), "topic": "Discovery - patient engagement", "tone": "visionary", "duration_min": 45},
                {"number": 2, "date": days_ago(52), "topic": "Strategic alignment call", "tone": "excited", "duration_min": 50},
                {"number": 3, "date": days_ago(38), "topic": "Demo + roadmap review", "tone": "positive", "duration_min": 60},
                {"number": 4, "date": days_ago(22), "topic": "Budget update - bad news", "tone": "apologetic", "duration_min": 15},
            ],
        },
        
        "stage_transitions": [
            {"stage": "lead", "entered": days_ago(72), "exited": days_ago(65), "days": 7},
            {"stage": "discovery", "entered": days_ago(65), "exited": days_ago(55), "days": 10},
            {"stage": "qualification", "entered": days_ago(55), "exited": days_ago(45), "days": 10},
            {"stage": "demo", "entered": days_ago(45), "exited": days_ago(38), "days": 7},
            {"stage": "evaluation", "entered": days_ago(38), "exited": days_ago(22), "days": 16},
            {"stage": "closed_lost", "entered": days_ago(22), "exited": None, "days": 22},
        ],
        
        "outcome": "closed_lost",
        "deal_notes": "Lost to budget freeze, not competition or product. CDO was great champion. Keep warm - this is a H2 opportunity if healthcare market stabilizes. Not a reflection of sales execution.",
    },
    
    # ==========================================================================
    # DEAL 9: Manufacturing - Demo stage
    # ==========================================================================
    {
        "id": "deal-009-precisionparts",
        "name": "Robert Andersen",
        "title": "VP of Manufacturing",
        "company": "Precision Parts International",
        "industry": "Manufacturing / Aerospace",
        "email": "r.andersen@precisionparts.com",
        "deal_value": 165000,
        "deal_value_display": "$165,000 ARR",
        
        "personality_traits": ["precise", "quality-focused", "methodical", "safety-conscious"],
        "communication_style": "formal",
        "decision_style": "committee-driven",
        "preferred_contact": "Email with documentation",
        "meeting_preference": "Structured meetings with clear agendas",
        
        "pain_points": [
            {
                "description": "Traceability for aerospace parts requires manual documentation",
                "severity": "critical",
                "category": "compliance",
                "business_impact": "Failed FAA audit last year, nearly lost major contract",
                "meeting_number": 1,
            },
            {
                "description": "Quality inspection data lives in Excel spreadsheets",
                "severity": "high",
                "category": "efficiency",
                "business_impact": "2 FTEs dedicated to manual data compilation",
                "meeting_number": 2,
            },
        ],
        
        "objections": [
            {
                "description": "AS9100 certification requirements - can you meet them?",
                "type": "compliance",
                "severity": "hard_blocker",
                "resolution_status": "partially_resolved",
                "resolution_notes": "Shared AS9100 compliance documentation. Under review by their quality team.",
                "meeting_number": 3,
            },
        ],
        
        "buying_signals": [
            {
                "signal": "Quality Director joined call unprompted",
                "strength": "strong",
                "type": "stakeholder",
                "meeting_number": 3,
            },
            {
                "signal": "Asked about implementation timeline for their facility",
                "strength": "moderate",
                "type": "timeline",
                "meeting_number": 3,
            },
        ],
        
        "stakeholders": [
            {
                "name": "Robert Andersen",
                "role": "VP of Manufacturing",
                "influence": "champion",
                "sentiment": "positive",
                "notes": "Driving initiative. Burned by FAA audit failure.",
            },
            {
                "name": "Carol Washington",
                "role": "Quality Director",
                "influence": "evaluator",
                "sentiment": "cautiously_positive",
                "notes": "Key validator. Needs AS9100 compliance proof.",
            },
            {
                "name": "George Martinez",
                "role": "COO",
                "influence": "decision_maker",
                "sentiment": "unknown",
                "notes": "Final approval. Haven't met. Robert says he's supportive.",
            },
        ],
        
        "competitive_context": [
            {
                "competitor": "SAP (considering)",
                "status": "researching",
                "notes": "SAP is their ERP. Considering SAP quality module but it's expensive.",
            },
        ],
        
        "next_steps": [
            {
                "action": "Demo of traceability features to quality team",
                "owner": "both",
                "due_date": days_ago(-4),
                "commitment_level": "firm",
            },
            {
                "action": "Send AS9100 compliance matrix",
                "owner": "rep",
                "due_date": days_ago(-2),
                "commitment_level": "firm",
            },
        ],
        
        "meeting_metadata": {
            "meeting_count": 3,
            "overall_deal_stage": "demo",
            "momentum": "steady",
            "meetings": [
                {"number": 1, "date": days_ago(32), "topic": "Discovery - traceability pain", "tone": "serious", "duration_min": 45},
                {"number": 2, "date": days_ago(22), "topic": "Technical requirements review", "tone": "detailed", "duration_min": 75},
                {"number": 3, "date": days_ago(12), "topic": "Initial demo + compliance discussion", "tone": "thorough", "duration_min": 90},
            ],
        },
        
        "stage_transitions": [
            {"stage": "lead", "entered": days_ago(40), "exited": days_ago(32), "days": 8},
            {"stage": "discovery", "entered": days_ago(32), "exited": days_ago(24), "days": 8},
            {"stage": "qualification", "entered": days_ago(24), "exited": days_ago(15), "days": 9},
            {"stage": "demo", "entered": days_ago(15), "exited": None, "days": 15},
        ],
        
        "deal_notes": "Aerospace manufacturing = long sales cycle but very sticky. AS9100 compliance is table stakes - need to nail this. Quality Director is key gatekeeper. Good opportunity if we pass compliance review.",
    },
    
    # ==========================================================================
    # DEAL 10: E-commerce - Lead stage (just came in)
    # ==========================================================================
    {
        "id": "deal-010-freshgoods",
        "name": "Sophie Martinez",
        "title": "CTO",
        "company": "FreshGoods Market",
        "industry": "E-commerce / Grocery",
        "email": "sophie@freshgoodsmarket.com",
        "deal_value": 55000,
        "deal_value_display": "$55,000 ARR",
        
        "personality_traits": ["startup-mindset", "scrappy", "data-curious", "fast-paced"],
        "communication_style": "casual",
        "decision_style": "fast-mover",
        "preferred_contact": "Text or WhatsApp",
        "meeting_preference": "Quick calls, walking meetings",
        
        "pain_points": [
            {
                "description": "Grocery demand forecasting is terrible - 25% waste on perishables",
                "severity": "critical",
                "category": "operations",
                "business_impact": "$500K/year in spoilage for a company their size",
                "meeting_number": None,  # From inbound form
            },
        ],
        
        "objections": [],
        
        "buying_signals": [
            {
                "signal": "Filled out detailed inbound form with specific pain points",
                "strength": "moderate",
                "type": "intent",
                "meeting_number": None,
            },
        ],
        
        "stakeholders": [
            {
                "name": "Sophie Martinez",
                "role": "CTO",
                "influence": "decision_maker",
                "sentiment": "unknown",
                "notes": "Inbound lead. Startup so probably CTO makes tech decisions.",
            },
        ],
        
        "competitive_context": [],
        
        "next_steps": [
            {
                "action": "Discovery call - understand forecasting pain in detail",
                "owner": "rep",
                "due_date": days_ago(-2),
                "commitment_level": "firm",
            },
        ],
        
        "meeting_metadata": {
            "meeting_count": 0,
            "overall_deal_stage": "lead",
            "momentum": "new",
            "source": "Inbound - website form",
            "meetings": [],
        },
        
        "stage_transitions": [
            {"stage": "lead", "entered": days_ago(2), "exited": None, "days": 2},
        ],
        
        "deal_notes": "Fresh inbound lead. Grocery/perishables vertical is interesting - spoilage pain is real. Smaller deal but could be good case study. Series B startup, growing fast.",
    },
]


def get_demo_prospects() -> list:
    """Return the list of demo prospects."""
    return DEMO_PROSPECTS


def get_prospect_by_id(prospect_id: str):
    """Find a prospect by ID."""
    for p in DEMO_PROSPECTS:
        if p["id"] == prospect_id:
            return p
    return None


def get_prospects_by_stage(stage: str) -> list:
    """Get all prospects in a given stage."""
    return [
        p for p in DEMO_PROSPECTS 
        if p.get("meeting_metadata", {}).get("overall_deal_stage") == stage
    ]


def get_pipeline_summary() -> dict:
    """Get summary stats for the pipeline."""
    total = len(DEMO_PROSPECTS)
    by_stage = {}
    total_value = 0
    weighted_value = 0
    
    stage_weights = {
        "lead": 0.05,
        "discovery": 0.10,
        "qualification": 0.20,
        "demo": 0.35,
        "evaluation": 0.50,
        "proposal": 0.65,
        "negotiation": 0.80,
        "closed_won": 1.0,
        "closed_lost": 0.0,
    }
    
    for p in DEMO_PROSPECTS:
        stage = p.get("meeting_metadata", {}).get("overall_deal_stage", "lead")
        by_stage[stage] = by_stage.get(stage, 0) + 1
        
        value = p.get("deal_value", 0)
        total_value += value
        weighted_value += value * stage_weights.get(stage, 0)
    
    return {
        "total_deals": total,
        "by_stage": by_stage,
        "total_pipeline_value": total_value,
        "weighted_pipeline_value": round(weighted_value),
        "closed_won_value": sum(
            p.get("deal_value", 0) for p in DEMO_PROSPECTS 
            if p.get("outcome") == "closed_won"
        ),
    }


if __name__ == "__main__":
    # Quick test
    import json
    
    summary = get_pipeline_summary()
    print("Pipeline Summary:")
    print(json.dumps(summary, indent=2))
    
    print(f"\nTotal prospects: {len(DEMO_PROSPECTS)}")
    for p in DEMO_PROSPECTS:
        stage = p["meeting_metadata"]["overall_deal_stage"]
        print(f"  - {p['name']} ({p['company']}) - {stage} - {p['deal_value_display']}")
