"""
Negotiation scenarios for temporal awareness benchmark.
Based on "New Recruit" and "Rubbermind" case studies used in the paper.
"""

from dataclasses import dataclass
from typing import Dict, List, Any

@dataclass
class PayoffTable:
    """Private payoff table for an agent."""
    issues: Dict[str, Dict[str, int]]  # issue -> {option: payoff}
    batna: int  # Best Alternative to Negotiated Agreement
    
    def calculate_payoff(self, contract: Dict[str, str]) -> int:
        """Calculate total payoff for a contract."""
        return sum(self.issues[issue][option] for issue, option in contract.items())


# ============================================================================
# NEW RECRUIT SCENARIO
# Adapted from SHRM Workplace Dispute Resolution case studies
# ============================================================================

NEW_RECRUIT_ISSUES = {
    "salary": ["$90,000", "$95,000", "$100,000", "$105,000", "$110,000"],
    "signing_bonus": ["$0", "$5,000", "$10,000", "$15,000", "$20,000"],
    "vacation_days": ["15 days", "20 days", "25 days", "30 days"],
    "start_date": ["Immediate", "2 weeks", "1 month", "2 months"]
}

# Hiring Manager payoffs (prefers lower salary, earlier start, fewer vacation days)
MANAGER_PAYOFFS = {
    "salary": {
        "$90,000": 4000, "$95,000": 3000, "$100,000": 2000, 
        "$105,000": 1000, "$110,000": 0
    },
    "signing_bonus": {
        "$0": 2000, "$5,000": 1500, "$10,000": 1000, 
        "$15,000": 500, "$20,000": 0
    },
    "vacation_days": {
        "15 days": 1200, "20 days": 800, "25 days": 400, "30 days": 0
    },
    "start_date": {
        "Immediate": 1800, "2 weeks": 1200, "1 month": 600, "2 months": 0
    }
}

# Candidate payoffs (prefers higher salary, later start, more vacation)
CANDIDATE_PAYOFFS = {
    "salary": {
        "$90,000": 0, "$95,000": 1000, "$100,000": 2000, 
        "$105,000": 3000, "$110,000": 4000
    },
    "signing_bonus": {
        "$0": 0, "$5,000": 500, "$10,000": 1000, 
        "$15,000": 1500, "$20,000": 2000
    },
    "vacation_days": {
        "15 days": 0, "20 days": 600, "25 days": 1200, "30 days": 1800
    },
    "start_date": {
        # Note: This is INTEGRATIVE - both parties prefer later start (candidate wants prep time)
        # This creates opportunity for win-win trades
        "Immediate": 0, "2 weeks": 200, "1 month": 400, "2 months": 600
    }
}

NEW_RECRUIT_SCENARIO = {
    "name": "New Recruit",
    "description": "Hiring negotiation between a manager and job candidate for a software engineering position.",
    "issues": NEW_RECRUIT_ISSUES,
    "manager": PayoffTable(MANAGER_PAYOFFS, batna=3000),  # Alternative candidate exists
    "candidate": PayoffTable(CANDIDATE_PAYOFFS, batna=2500),  # Other job offer
    "context": {
        "manager": """You are a hiring manager at TechCorp negotiating a compensation package with a promising job candidate.
You have a budget constraint but flexibility on non-salary items. Another qualified candidate exists (your BATNA = 3000 points).
Your goal is to hire this candidate at terms favorable to the company.""",
        "candidate": """You are a job candidate negotiating a compensation package with a hiring manager at TechCorp.
You have another offer on the table (your BATNA = 2500 points) but prefer this company.
Your goal is to negotiate the best possible package while securing this position."""
    }
}


# ============================================================================
# RUBBERMIND SCENARIO
# Adapted from Kellogg DRRC case studies - more challenging integrative structure
# ============================================================================

RUBBERMIND_ISSUES = {
    "price_per_unit": ["$8.00", "$8.50", "$9.00", "$9.50", "$10.00"],
    "volume_commitment": ["10,000", "15,000", "20,000", "25,000", "30,000"],
    "contract_length": ["1 year", "2 years", "3 years"],
    "quality_tier": ["Standard", "Premium", "Elite"],
    "payment_terms": ["Net 15", "Net 30", "Net 45", "Net 60"]
}

SUPPLIER_PAYOFFS = {
    "price_per_unit": {
        "$8.00": 0, "$8.50": 500, "$9.00": 1000, "$9.50": 1500, "$10.00": 2000
    },
    "volume_commitment": {
        "10,000": 0, "15,000": 400, "20,000": 800, "25,000": 1200, "30,000": 1600
    },
    "contract_length": {
        "1 year": 0, "2 years": 600, "3 years": 1200
    },
    "quality_tier": {
        "Standard": 800, "Premium": 400, "Elite": 0
    },
    "payment_terms": {
        "Net 15": 600, "Net 30": 400, "Net 45": 200, "Net 60": 0
    }
}

BUYER_PAYOFFS = {
    "price_per_unit": {
        "$8.00": 2000, "$8.50": 1500, "$9.00": 1000, "$9.50": 500, "$10.00": 0
    },
    "volume_commitment": {
        "10,000": 800, "15,000": 600, "20,000": 400, "25,000": 200, "30,000": 0
    },
    "contract_length": {
        # INTEGRATIVE: Buyer also prefers longer term (price stability)
        "1 year": 0, "2 years": 400, "3 years": 800
    },
    "quality_tier": {
        "Standard": 0, "Premium": 600, "Elite": 1200
    },
    "payment_terms": {
        "Net 15": 0, "Net 30": 200, "Net 45": 400, "Net 60": 600
    }
}

RUBBERMIND_SCENARIO = {
    "name": "Rubbermind",
    "description": "B2B supply contract negotiation between a supplier and buyer of industrial components.",
    "issues": RUBBERMIND_ISSUES,
    "supplier": PayoffTable(SUPPLIER_PAYOFFS, batna=2800),
    "buyer": PayoffTable(BUYER_PAYOFFS, batna=2500),
    "context": {
        "supplier": """You are a sales representative for Rubbermind Inc., a manufacturer of industrial rubber components.
You are negotiating a supply contract with a potential major client.
Your alternative is to focus on smaller accounts (BATNA = 2800 points).
Your goal is to secure a profitable long-term contract.""",
        "buyer": """You are a procurement manager for AutoParts Corp., sourcing industrial rubber components.
You are negotiating with Rubbermind Inc., a quality supplier.
You have alternative suppliers available (BATNA = 2500 points).
Your goal is to secure favorable terms for your company."""
    }
}


SCENARIOS = {
    "new_recruit": NEW_RECRUIT_SCENARIO,
    "rubbermind": RUBBERMIND_SCENARIO
}


def get_scenario(name: str) -> dict:
    """Get a scenario by name."""
    return SCENARIOS.get(name, NEW_RECRUIT_SCENARIO)


def format_payoff_table(payoff_table: PayoffTable) -> str:
    """Format payoff table for display in prompt."""
    lines = ["Your private payoff table (points for each option):"]
    for issue, options in payoff_table.issues.items():
        lines.append(f"\n{issue.replace('_', ' ').title()}:")
        for option, points in options.items():
            lines.append(f"  {option}: {points} points")
    lines.append(f"\nYour BATNA (walk away value): {payoff_table.batna} points")
    return "\n".join(lines)
