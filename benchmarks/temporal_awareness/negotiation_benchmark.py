"""
Temporal Awareness Negotiation Benchmark

Implements the benchmark from arxiv:2601.13206:
"Real-Time Deadlines Reveal Temporal Awareness Failures in LLM Strategic Dialogues"

This tests whether LLMs can track elapsed time and adapt their strategy accordingly
in multi-issue negotiations under real-time deadlines.
"""

import json
import time
from dataclasses import dataclass, field
from enum import Enum
from typing import Dict, List, Optional, Any, Tuple
from datetime import datetime

# Assuming these will be implemented or imported
try:
    from scenarios import get_scenario, format_payoff_table, PayoffTable
except ImportError:
    from .scenarios import get_scenario, format_payoff_table, PayoffTable


class Condition(Enum):
    """Experimental conditions."""
    CONTROL = "control"               # Time limit stated once at start
    TIME_AWARE = "time_aware"         # Remaining time shown each turn
    URGENCY = "urgency"               # Generic urgency cue each turn
    TURN_BASED = "turn_based"         # Fixed turn limit (no time pressure)
    PIE_TEMPORAL = "pie_temporal"     # PIE: inject remaining time as entity state


class TerminationReason(Enum):
    """Why a negotiation ended."""
    DEAL = "deal"              # Agreement reached
    BATNA = "batna"            # One party walked away
    TIME_EXPIRED = "time_expired"  # Ran out of time
    TURN_EXPIRED = "turn_expired"  # Ran out of turns
    ERROR = "error"            # Parsing/API error


@dataclass
class Action:
    """A negotiation action from an agent."""
    message: str                      # Natural language message
    offer: Optional[Dict[str, str]]   # Proposed contract terms (or None)
    accept: bool = False              # Accept the current offer on the table
    invoke_batna: bool = False        # Walk away to outside option
    
    @classmethod
    def from_json(cls, data: dict) -> 'Action':
        return cls(
            message=data.get("message", ""),
            offer=data.get("offer"),
            accept=data.get("accept", False),
            invoke_batna=data.get("invoke_batna", False)
        )
    
    def to_json(self) -> dict:
        return {
            "message": self.message,
            "offer": self.offer,
            "accept": self.accept,
            "invoke_batna": self.invoke_batna
        }


@dataclass
class Turn:
    """A single turn in the negotiation."""
    agent_role: str
    action: Action
    remaining_time: float
    turn_number: int
    word_count: int
    time_consumed: float  # Time deducted for this utterance


@dataclass
class NegotiationResult:
    """Result of a complete negotiation."""
    scenario_name: str
    condition: Condition
    time_limit: float
    turn_limit: Optional[int]
    
    # Outcome
    termination_reason: TerminationReason
    deal_reached: bool
    final_offer: Optional[Dict[str, str]]
    
    # Payoffs
    agent1_payoff: Optional[int]
    agent2_payoff: Optional[int]
    joint_payoff: Optional[int]
    
    # Process metrics
    turns: List[Turn] = field(default_factory=list)
    total_words: int = 0
    total_time_consumed: float = 0.0
    actual_duration: float = 0.0  # Wall clock time
    
    # Metadata
    model_name: str = ""
    timestamp: str = ""
    
    def to_dict(self) -> dict:
        return {
            "scenario": self.scenario_name,
            "condition": self.condition.value,
            "time_limit": self.time_limit,
            "turn_limit": self.turn_limit,
            "termination_reason": self.termination_reason.value,
            "deal_reached": self.deal_reached,
            "final_offer": self.final_offer,
            "agent1_payoff": self.agent1_payoff,
            "agent2_payoff": self.agent2_payoff,
            "joint_payoff": self.joint_payoff,
            "num_turns": len(self.turns),
            "total_words": self.total_words,
            "total_time_consumed": self.total_time_consumed,
            "model": self.model_name,
            "timestamp": self.timestamp
        }


class NegotiationEngine:
    """
    Engine for running temporal awareness negotiation experiments.
    """
    
    # Speech latency: 150 words per minute (from paper)
    WORDS_PER_MINUTE = 150
    SECONDS_PER_WORD = 60.0 / WORDS_PER_MINUTE  # 0.4 seconds per word
    
    def __init__(
        self,
        llm_client,  # Generic LLM client interface
        scenario_name: str = "new_recruit",
        condition: Condition = Condition.CONTROL,
        time_limit: float = 300.0,  # seconds
        turn_limit: Optional[int] = None,
        model_name: str = "unknown",
        pie_temporal_tracker: Optional[Any] = None  # For PIE integration
    ):
        self.llm_client = llm_client
        self.scenario = get_scenario(scenario_name)
        self.condition = condition
        self.time_limit = time_limit
        self.turn_limit = turn_limit
        self.model_name = model_name
        self.pie_tracker = pie_temporal_tracker
        
        # Get roles from scenario
        if scenario_name == "new_recruit":
            self.roles = ["manager", "candidate"]
            self.payoffs = {
                "manager": self.scenario["manager"],
                "candidate": self.scenario["candidate"]
            }
        else:  # rubbermind
            self.roles = ["supplier", "buyer"]
            self.payoffs = {
                "supplier": self.scenario["supplier"],
                "buyer": self.scenario["buyer"]
            }
    
    def _build_system_prompt(self, role: str) -> str:
        """Build system prompt for an agent."""
        context = self.scenario["context"][role]
        payoff_table = format_payoff_table(self.payoffs[role])
        
        issues_str = json.dumps(self.scenario["issues"], indent=2)
        
        prompt = f"""You are participating in a negotiation simulation.

{context}

ISSUES TO NEGOTIATE:
{issues_str}

{payoff_table}

RULES:
1. You must respond with a JSON object containing:
   - "message": Your natural language response to the other party
   - "offer": A complete proposal (all issues) or null if not making an offer
   - "accept": true if you accept the current offer on the table, false otherwise
   - "invoke_batna": true if you want to walk away to your outside option

2. Your goal is to maximize your payoff while reaching an agreement.
3. The negotiation ends when: someone accepts an offer, someone invokes BATNA, or time/turns expire.
4. If no agreement is reached, you get your BATNA payoff.

"""
        
        # Add time constraint information based on condition
        if self.condition == Condition.CONTROL:
            prompt += f"""TIME CONSTRAINT:
You have a total of {self.time_limit:.0f} seconds to reach an agreement.
Each message you send consumes time based on its length (approximately 150 words per minute).
If time expires without agreement, you both receive your BATNA payoffs.
"""
        
        elif self.condition == Condition.TIME_AWARE:
            prompt += f"""TIME CONSTRAINT:
You have a total of {self.time_limit:.0f} seconds to reach an agreement.
Each message you send consumes time based on its length (approximately 150 words per minute).
You will be informed of the remaining time at each turn.
If time expires without agreement, you both receive your BATNA payoffs.
"""
        
        elif self.condition == Condition.URGENCY:
            prompt += f"""TIME CONSTRAINT:
You have a total of {self.time_limit:.0f} seconds to reach an agreement.
Each message you send consumes time based on its length (approximately 150 words per minute).
If time expires without agreement, you both receive your BATNA payoffs.
"""
        
        elif self.condition == Condition.TURN_BASED:
            prompt += f"""TURN CONSTRAINT:
You have a total of {self.turn_limit} turns (combined between both parties) to reach an agreement.
If turns expire without agreement, you both receive your BATNA payoffs.
"""
        
        elif self.condition == Condition.PIE_TEMPORAL:
            prompt += f"""TIME CONSTRAINT:
You have a total of {self.time_limit:.0f} seconds to reach an agreement.
Each message consumes time (approximately 150 words per minute).
IMPORTANT: Pay attention to the [TEMPORAL_STATE] information provided - this shows accurate remaining time.
If time expires without agreement, you both receive your BATNA payoffs.
"""
        
        prompt += """
RESPONSE FORMAT (JSON only):
{
  "message": "Your response to the other party",
  "offer": {"issue1": "value1", "issue2": "value2", ...} or null,
  "accept": false,
  "invoke_batna": false
}

Respond ONLY with valid JSON. No additional text before or after the JSON."""
        
        return prompt
    
    def _calculate_time_consumed(self, message: str) -> float:
        """Calculate time consumed by a message based on word count."""
        word_count = len(message.split())
        return word_count * self.SECONDS_PER_WORD
    
    def _format_turn_prefix(self, remaining_time: float, turn_number: int) -> str:
        """Format the prefix shown to agents based on condition."""
        if self.condition == Condition.TIME_AWARE:
            return f"[{remaining_time:.0f} seconds remaining]"
        elif self.condition == Condition.URGENCY:
            return "(Deadline approaching--act with urgency.)"
        elif self.condition == Condition.TURN_BASED:
            remaining_turns = self.turn_limit - turn_number
            return f"[{remaining_turns} turns remaining]"
        elif self.condition == Condition.PIE_TEMPORAL:
            # PIE-style entity state injection
            return f"[TEMPORAL_STATE: remaining_time={remaining_time:.1f}s, urgency={'HIGH' if remaining_time < 60 else 'MEDIUM' if remaining_time < 120 else 'LOW'}]"
        else:  # CONTROL
            return ""
    
    def _parse_action(self, response: str) -> Action:
        """Parse LLM response into an Action."""
        try:
            # Try to find JSON in the response
            response = response.strip()
            
            # Handle markdown code blocks
            if response.startswith("```json"):
                response = response[7:]
            if response.startswith("```"):
                response = response[3:]
            if response.endswith("```"):
                response = response[:-3]
            
            data = json.loads(response.strip())
            return Action.from_json(data)
        except (json.JSONDecodeError, KeyError) as e:
            # Return a default action on parse failure
            return Action(
                message=response[:500] if response else "...",
                offer=None,
                accept=False,
                invoke_batna=False
            )
    
    async def _get_agent_response(
        self,
        role: str,
        history: List[Dict[str, str]],
        remaining_time: float,
        turn_number: int
    ) -> Action:
        """Get a response from an agent."""
        system_prompt = self._build_system_prompt(role)
        
        # Add turn prefix to the last message in history
        formatted_history = history.copy()
        if formatted_history and self.condition != Condition.CONTROL:
            prefix = self._format_turn_prefix(remaining_time, turn_number)
            if prefix:
                last_msg = formatted_history[-1]
                if last_msg["role"] == "user":
                    formatted_history[-1] = {
                        "role": "user",
                        "content": f"{prefix}\n\n{last_msg['content']}"
                    }
        
        # Call LLM
        response = await self.llm_client.chat(
            system=system_prompt,
            messages=formatted_history
        )
        
        return self._parse_action(response)
    
    async def run_negotiation(self) -> NegotiationResult:
        """Run a complete negotiation and return results."""
        start_time = time.time()
        
        remaining_time = self.time_limit
        turn_number = 0
        turns: List[Turn] = []
        
        # Conversation history for each agent
        history_agent1: List[Dict[str, str]] = []
        history_agent2: List[Dict[str, str]] = []
        
        current_offer: Optional[Dict[str, str]] = None
        termination_reason: Optional[TerminationReason] = None
        deal_reached = False
        final_offer: Optional[Dict[str, str]] = None
        
        # Alternate between agents
        current_agent_idx = 0
        
        while True:
            turn_number += 1
            current_role = self.roles[current_agent_idx]
            other_role = self.roles[1 - current_agent_idx]
            
            # Check termination conditions
            if self.condition == Condition.TURN_BASED:
                if turn_number > self.turn_limit:
                    termination_reason = TerminationReason.TURN_EXPIRED
                    break
            else:
                if remaining_time <= 0:
                    termination_reason = TerminationReason.TIME_EXPIRED
                    break
            
            # Get current agent's history
            history = history_agent1 if current_agent_idx == 0 else history_agent2
            
            # Get agent response
            try:
                action = await self._get_agent_response(
                    current_role, history, remaining_time, turn_number
                )
            except Exception as e:
                termination_reason = TerminationReason.ERROR
                break
            
            # Calculate time consumed
            time_consumed = self._calculate_time_consumed(action.message)
            word_count = len(action.message.split())
            
            # Update remaining time (only for time-based conditions)
            if self.condition != Condition.TURN_BASED:
                remaining_time -= time_consumed
            
            # Record turn
            turn = Turn(
                agent_role=current_role,
                action=action,
                remaining_time=remaining_time,
                turn_number=turn_number,
                word_count=word_count,
                time_consumed=time_consumed
            )
            turns.append(turn)
            
            # Check for termination actions
            if action.accept and current_offer is not None:
                deal_reached = True
                final_offer = current_offer
                termination_reason = TerminationReason.DEAL
                break
            
            if action.invoke_batna:
                termination_reason = TerminationReason.BATNA
                break
            
            # Update offer if one was made
            if action.offer is not None:
                current_offer = action.offer
            
            # Update histories for both agents
            agent_msg = json.dumps(action.to_json())
            
            # Add to current agent's history as assistant
            if current_agent_idx == 0:
                history_agent1.append({"role": "assistant", "content": agent_msg})
                history_agent2.append({"role": "user", "content": agent_msg})
            else:
                history_agent2.append({"role": "assistant", "content": agent_msg})
                history_agent1.append({"role": "user", "content": agent_msg})
            
            # Switch agents
            current_agent_idx = 1 - current_agent_idx
            
            # Safety limit
            if turn_number > 50:
                termination_reason = TerminationReason.ERROR
                break
        
        # Calculate payoffs
        agent1_payoff: Optional[int] = None
        agent2_payoff: Optional[int] = None
        joint_payoff: Optional[int] = None
        
        if deal_reached and final_offer:
            try:
                agent1_payoff = self.payoffs[self.roles[0]].calculate_payoff(final_offer)
                agent2_payoff = self.payoffs[self.roles[1]].calculate_payoff(final_offer)
                joint_payoff = agent1_payoff + agent2_payoff
            except (KeyError, TypeError):
                pass
        else:
            # BATNA payoffs
            agent1_payoff = self.payoffs[self.roles[0]].batna
            agent2_payoff = self.payoffs[self.roles[1]].batna
            joint_payoff = agent1_payoff + agent2_payoff
        
        actual_duration = time.time() - start_time
        total_words = sum(t.word_count for t in turns)
        total_time_consumed = sum(t.time_consumed for t in turns)
        
        return NegotiationResult(
            scenario_name=self.scenario["name"],
            condition=self.condition,
            time_limit=self.time_limit,
            turn_limit=self.turn_limit,
            termination_reason=termination_reason or TerminationReason.ERROR,
            deal_reached=deal_reached,
            final_offer=final_offer,
            agent1_payoff=agent1_payoff,
            agent2_payoff=agent2_payoff,
            joint_payoff=joint_payoff,
            turns=turns,
            total_words=total_words,
            total_time_consumed=total_time_consumed,
            actual_duration=actual_duration,
            model_name=self.model_name,
            timestamp=datetime.now().isoformat()
        )


class MockLLMClient:
    """Mock LLM client for testing without API calls."""
    
    def __init__(self, behavior: str = "cooperative"):
        self.behavior = behavior
        self.turn_count = 0
    
    async def chat(self, system: str, messages: List[Dict[str, str]]) -> str:
        """Generate a mock response."""
        self.turn_count += 1
        
        if self.behavior == "cooperative":
            # Cooperative agent that makes reasonable offers and accepts after a few turns
            if self.turn_count <= 2:
                return json.dumps({
                    "message": "Let's work together to find a mutually beneficial agreement.",
                    "offer": {
                        "salary": "$100,000",
                        "signing_bonus": "$10,000",
                        "vacation_days": "20 days",
                        "start_date": "2 weeks"
                    },
                    "accept": False,
                    "invoke_batna": False
                })
            else:
                return json.dumps({
                    "message": "I think this is a fair deal. Let's close this.",
                    "offer": None,
                    "accept": True,
                    "invoke_batna": False
                })
        else:
            # Uncooperative agent that never concedes
            return json.dumps({
                "message": "I need better terms.",
                "offer": {
                    "salary": "$110,000",
                    "signing_bonus": "$20,000", 
                    "vacation_days": "30 days",
                    "start_date": "2 months"
                },
                "accept": False,
                "invoke_batna": False
            })


# Example usage
if __name__ == "__main__":
    import asyncio
    
    async def test_mock():
        client = MockLLMClient(behavior="cooperative")
        engine = NegotiationEngine(
            llm_client=client,
            scenario_name="new_recruit",
            condition=Condition.TIME_AWARE,
            time_limit=300,
            model_name="mock"
        )
        result = await engine.run_negotiation()
        print(f"Deal reached: {result.deal_reached}")
        print(f"Turns: {len(result.turns)}")
        print(f"Joint payoff: {result.joint_payoff}")
        print(json.dumps(result.to_dict(), indent=2))
    
    asyncio.run(test_mock())
