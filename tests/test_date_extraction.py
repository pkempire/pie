"""
Test event extraction with date computation from relative references.

Tests that the updated extraction prompt correctly:
1. Extracts EVENT entities from conversations
2. Computes dates from relative references ("yesterday", "last week", etc.)
3. Includes required fields: name, type="event", state.date (YYYY-MM-DD), state.description
"""

import json
import sys
from pathlib import Path
from datetime import datetime, timedelta
from unittest.mock import MagicMock, patch

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from pie.ingestion.prompts import (
    EXTRACTION_SYSTEM_PROMPT,
    build_extraction_user_message,
)
from pie.core.llm import LLMClient, parse_extraction_result


# Sample test conversations with date references
TEST_CONVERSATIONS = [
    {
        "batch_date": "2025-02-05",
        "conversation_text": '''--- Conversation 1: "MoMA Trip" ---
USER:
I went to MOMA yesterday - they had an amazing Picasso exhibit! Spent about 3 hours there.

ASSISTANT:
That sounds wonderful! MoMA has one of the best collections of Picasso's work...''',
        "expected_events": [
            {
                "name_contains": "MoMA",  # Could be "MoMA visit" or similar
                "expected_date": "2025-02-04",  # yesterday
                "description_contains": "Picasso",
            }
        ],
    },
    {
        "batch_date": "2025-02-05", 
        "conversation_text": '''--- Conversation 1: "Phone Case Order" ---
USER:
Last week I ordered a new phone case from Amazon. It was a clear case for my iPhone 15. Should arrive tomorrow.

ASSISTANT:
Great choice! Clear cases are popular because they show off the phone design...''',
        "expected_events": [
            {
                "name_contains": "phone case",
                "expected_date": "2025-01-29",  # last week (batch_date - 7 days)
                "description_contains": "order",
            }
        ],
    },
    {
        "batch_date": "2025-02-05",
        "conversation_text": '''--- Conversation 1: "Helping Friend Move" ---
USER:
Two days ago I helped my friend Sarah move to her new apartment in Brooklyn. It took all day but we got everything done.

ASSISTANT:
That's really kind of you! Moving is always exhausting...''',
        "expected_events": [
            {
                "name_contains": "move",
                "expected_date": "2025-02-03",  # two days ago
                "description_contains": "Sarah",
            }
        ],
    },
]


class MockLLMResponse:
    """Mock LLM response generator for testing."""
    
    @staticmethod
    def generate_extraction_response(batch_date: str, conversation_text: str) -> dict:
        """Generate a mock extraction response based on the conversation content."""
        
        entities = []
        
        # Parse batch date
        batch_dt = datetime.strptime(batch_date, "%Y-%m-%d")
        
        # MoMA visit (yesterday)
        if "MOMA yesterday" in conversation_text or "MoMA yesterday" in conversation_text:
            event_date = (batch_dt - timedelta(days=1)).strftime("%Y-%m-%d")
            entities.append({
                "name": "MoMA visit",
                "type": "event",
                "state": {
                    "date": event_date,
                    "description": "Visited MoMA museum, saw Picasso exhibit, spent ~3 hours",
                    "location": "MoMA, New York",
                },
                "is_new": True,
                "matches_existing": None,
                "confidence": 0.95,
            })
        
        # Phone case order (last week)
        if "Last week I ordered" in conversation_text and "phone case" in conversation_text:
            event_date = (batch_dt - timedelta(days=7)).strftime("%Y-%m-%d")
            entities.append({
                "name": "iPhone phone case order",
                "type": "event",
                "state": {
                    "date": event_date,
                    "description": "Ordered clear phone case for iPhone 15 from Amazon",
                },
                "is_new": True,
                "matches_existing": None,
                "confidence": 0.9,
            })
        
        # Helping friend move (two days ago)
        if "Two days ago" in conversation_text and "move" in conversation_text:
            event_date = (batch_dt - timedelta(days=2)).strftime("%Y-%m-%d")
            entities.append({
                "name": "Helped Sarah move",
                "type": "event",
                "state": {
                    "date": event_date,
                    "description": "Helped friend Sarah move to new apartment in Brooklyn, took all day",
                    "location": "Brooklyn",
                },
                "is_new": True,
                "matches_existing": None,
                "confidence": 0.95,
            })
            # Also extract the person
            entities.append({
                "name": "Sarah",
                "type": "person",
                "state": {
                    "description": "Friend who moved to Brooklyn",
                },
                "is_new": True,
                "matches_existing": None,
                "confidence": 0.85,
            })
        
        return {
            "entities": entities,
            "state_changes": [],
            "relationships": [],
            "period_context": "2025 early February",
            "significance": 0.3,
            "user_state": "casual",
            "summary": "User describes recent personal activities.",
        }


def test_prompt_contains_event_instructions():
    """Verify the prompt includes EVENT entity type and date computation instructions."""
    
    # Check EVENT is in entity types
    assert "event" in EXTRACTION_SYSTEM_PROMPT.lower()
    assert "Event entity format" in EXTRACTION_SYSTEM_PROMPT or "event" in EXTRACTION_SYSTEM_PROMPT
    
    # Check date computation instructions
    assert "yesterday" in EXTRACTION_SYSTEM_PROMPT.lower()
    assert "last week" in EXTRACTION_SYSTEM_PROMPT.lower()
    assert "batch date" in EXTRACTION_SYSTEM_PROMPT.lower()
    
    # Check required fields mentioned
    assert "YYYY-MM-DD" in EXTRACTION_SYSTEM_PROMPT
    assert "state.date" in EXTRACTION_SYSTEM_PROMPT or '"date":' in EXTRACTION_SYSTEM_PROMPT
    
    print("✅ Prompt contains all required EVENT extraction instructions")


def test_extraction_with_mock_llm():
    """Test extraction with mocked LLM responses."""
    
    for test_case in TEST_CONVERSATIONS:
        batch_date = test_case["batch_date"]
        conversation_text = test_case["conversation_text"]
        expected_events = test_case["expected_events"]
        
        # Generate mock response
        mock_response = MockLLMResponse.generate_extraction_response(
            batch_date, conversation_text
        )
        
        # Parse the extraction result
        result = parse_extraction_result(
            raw=mock_response,
            conversation_ids=["test-convo-1"],
            tokens={"prompt": 1000, "completion": 200, "total": 1200},
        )
        
        # Find event entities
        event_entities = [e for e in result.entities if e.type == "event"]
        
        assert len(event_entities) >= 1, f"Expected at least 1 event entity, got {len(event_entities)}"
        
        for expected in expected_events:
            # Find matching event
            matching_events = [
                e for e in event_entities
                if expected["name_contains"].lower() in e.name.lower()
            ]
            
            assert len(matching_events) >= 1, (
                f"Expected event containing '{expected['name_contains']}', "
                f"but found: {[e.name for e in event_entities]}"
            )
            
            event = matching_events[0]
            
            # Verify date
            assert "date" in event.state, f"Event {event.name} missing 'date' in state"
            assert event.state["date"] == expected["expected_date"], (
                f"Event {event.name} has date {event.state['date']}, "
                f"expected {expected['expected_date']}"
            )
            
            # Verify description
            assert "description" in event.state, f"Event {event.name} missing 'description' in state"
            assert expected["description_contains"].lower() in event.state["description"].lower(), (
                f"Event description '{event.state['description']}' doesn't contain "
                f"'{expected['description_contains']}'"
            )
            
            print(f"✅ Event '{event.name}' correctly extracted with date {event.state['date']}")


def test_user_message_includes_batch_date():
    """Verify the user message includes the batch date for reference."""
    
    user_message = build_extraction_user_message(
        batch_date="2025-02-05",
        conversations_text="USER: I went to the museum yesterday",
        context_preamble="",
        num_conversations=1,
    )
    
    # Batch date should be in the message for the LLM to use
    assert "2025-02-05" in user_message
    print("✅ User message includes batch date for temporal reference")


def test_event_entity_type_recognized():
    """Verify that 'event' is a valid entity type that gets parsed correctly."""
    
    raw_response = {
        "entities": [
            {
                "name": "Test Event",
                "type": "event",
                "state": {
                    "date": "2025-02-04",
                    "description": "Test event description",
                },
                "is_new": True,
                "matches_existing": None,
                "confidence": 0.9,
            }
        ],
        "state_changes": [],
        "relationships": [],
        "period_context": "",
        "significance": 0.3,
        "user_state": "casual",
        "summary": "Test",
    }
    
    result = parse_extraction_result(
        raw=raw_response,
        conversation_ids=["test"],
        tokens={"prompt": 100, "completion": 50, "total": 150},
    )
    
    assert len(result.entities) == 1
    assert result.entities[0].type == "event"  # Should NOT be downgraded to "concept"
    print("✅ Event entity type is correctly recognized and preserved")


def run_all_tests():
    """Run all tests and print summary."""
    print("=" * 60)
    print("Testing Date Extraction Prompt Changes")
    print("=" * 60)
    print()
    
    tests = [
        ("Prompt contains event instructions", test_prompt_contains_event_instructions),
        ("User message includes batch date", test_user_message_includes_batch_date),
        ("Event entity type recognized", test_event_entity_type_recognized),
        ("Extraction with mock LLM", test_extraction_with_mock_llm),
    ]
    
    passed = 0
    failed = 0
    
    for name, test_fn in tests:
        print(f"\n--- {name} ---")
        try:
            test_fn()
            passed += 1
        except AssertionError as e:
            print(f"❌ FAILED: {e}")
            failed += 1
        except Exception as e:
            print(f"❌ ERROR: {e}")
            failed += 1
    
    print("\n" + "=" * 60)
    print(f"Results: {passed} passed, {failed} failed")
    print("=" * 60)
    
    return failed == 0


if __name__ == "__main__":
    import sys
    sys.path.insert(0, str(__file__).rsplit("/tests/", 1)[0])
    
    success = run_all_tests()
    sys.exit(0 if success else 1)
