from mempol.temporal import (
    ActiveProcess,
    ContextDecision,
    OutcomeEvent,
    StateTransition,
    TemporalMemoryStore,
    TemporalState,
)
from mempol.core.schema import Artifact, Span
from mempol.core.store import SQLiteMemoryStore
from mempol.temporal.context import compile_temporal_context_from_stores


def test_temporal_state_supersession_and_state_at_time(tmp_path):
    store = TemporalMemoryStore(tmp_path / "temporal.sqlite")

    s1 = TemporalState(
        id="state_diet_1",
        scope_id="user:p",
        key="user.diet",
        content="vegetarian",
        valid_from="2026-01-01T00:00:00Z",
        observed_at="2026-01-01T00:00:00Z",
        source_span_ids=["span_1"],
    )
    store.upsert_state(s1)
    store.apply_transition(
        StateTransition(
            id="tr_1",
            scope_id="user:p",
            transition_type="create",
            new_state_id=s1.id,
            observed_at=s1.observed_at,
            source_span_ids=s1.source_span_ids,
        )
    )

    s2 = TemporalState(
        id="state_diet_2",
        scope_id="user:p",
        key="user.diet",
        content="pescatarian",
        valid_from="2026-03-12T00:00:00Z",
        observed_at="2026-03-12T00:00:00Z",
        source_span_ids=["span_2"],
        supersedes_state_ids=[s1.id],
    )
    store.apply_transition(
        StateTransition(
            id="tr_2",
            scope_id="user:p",
            transition_type="supersede",
            old_state_ids=[s1.id],
            new_state_id=s2.id,
            observed_at=s2.observed_at,
            source_span_ids=s2.source_span_ids,
        ),
        new_state=s2,
    )
    store.commit()

    jan = store.current_states("user:p", at="2026-02-01T00:00:00Z", include_stale=True)
    now = store.current_states("user:p", at="2026-04-01T00:00:00Z")

    assert [s.content for s in jan] == ["vegetarian"]
    assert [s.content for s in now] == ["pescatarian"]
    assert store.state_history("user:p", "user.diet")[0].id == "state_diet_2"
    assert store.state_history("user:p", "user.diet")[1].valid_until == "2026-03-12T00:00:00Z"

    store.close()


def test_active_processes_and_decision_outcomes(tmp_path):
    store = TemporalMemoryStore(tmp_path / "temporal.sqlite")

    store.upsert_process(
        ActiveProcess(
            id="proc_followup",
            scope_id="account:a",
            kind="sales_followup",
            description="Send security docs after procurement call.",
            status="waiting",
            started_at="2026-06-10T00:00:00Z",
            deadline_at="2026-06-12T00:00:00Z",
        )
    )
    store.log_context_decision(
        ContextDecision(
            id="decide_1",
            scope_id="account:a",
            task="Should we follow up?",
            now="2026-06-13T00:00:00Z",
            action="interrupt",
            selected_process_ids=["proc_followup"],
            token_budget=1000,
            token_estimate=200,
        )
    )
    store.log_outcome(
        OutcomeEvent(
            id="outcome_1",
            decision_id="decide_1",
            scope_id="account:a",
            score=1.0,
            outcome_type="human",
            feedback="Correctly surfaced overdue follow-up.",
        )
    )
    store.commit()

    due = store.due_processes("account:a", now="2026-06-13T00:00:00Z")
    rows = store.decision_training_rows("account:a")

    assert len(due) == 1
    assert due[0].id == "proc_followup"
    assert rows[0]["action"] == "interrupt"
    assert rows[0]["outcome_score"] == 1.0
    assert store.stats()["outcome_events"] == 1

    store.close()


def test_temporal_context_compiler_logs_decision(tmp_path):
    core = SQLiteMemoryStore(tmp_path / "core.sqlite")
    temporal = TemporalMemoryStore(tmp_path / "temporal.sqlite")

    artifact = Artifact(
        id="artifact_call",
        source="crm",
        kind="call",
        title="Procurement call",
        content="The buyer moved to procurement review and asked for security docs.",
        created_at="2026-06-10T00:00:00Z",
    )
    span = Span(
        id="span_call_1",
        artifact_id=artifact.id,
        text="The buyer moved to procurement review and asked for security docs.",
        locator="call:1",
    )
    core.upsert_artifact(artifact)
    core.upsert_span(span)
    core.commit()

    state = TemporalState(
        id="state_stage",
        scope_id="account:acme",
        key="account.stage",
        content="Account moved to procurement review.",
        state_type="account",
        valid_from="2026-06-10T00:00:00Z",
        observed_at="2026-06-10T00:00:00Z",
        source_span_ids=[span.id],
    )
    temporal.apply_transition(
        StateTransition(
            id="tr_stage",
            scope_id="account:acme",
            transition_type="create",
            new_state_id=state.id,
            observed_at=state.observed_at,
            source_span_ids=[span.id],
            reason="Call moved account stage.",
        ),
        new_state=state,
    )
    temporal.upsert_process(
        ActiveProcess(
            id="proc_security_docs",
            scope_id="account:acme",
            kind="sales_followup",
            description="Send requested security docs.",
            status="waiting",
            started_at="2026-06-10T00:00:00Z",
            deadline_at="2026-06-12T00:00:00Z",
            source_span_ids=[span.id],
        )
    )
    temporal.commit()

    result = compile_temporal_context_from_stores(
        core=core,
        temporal=temporal,
        run_name="temporal_test",
        scope_id="account:acme",
        task="Should we follow up with procurement?",
        now="2026-06-13T00:00:00Z",
        k=4,
        token_budget=2500,
    )

    rows = temporal.decision_training_rows("account:acme")

    assert result["action"] == "interrupt"
    assert "procurement review" in result["markdown"]
    assert "security docs" in result["markdown"]
    assert rows[0]["action"] == "interrupt"
    assert rows[0]["selected_process_ids"] == ["proc_security_docs"]

    core.close()
    temporal.close()
