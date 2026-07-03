from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from mempol.core.schema import Artifact, MemoryState, Span, TraceEvent
from mempol.core.store import SQLiteMemoryStore


def test_universal_memory_persistence_and_provenance(tmp_path):
    store = SQLiteMemoryStore(tmp_path / "memory.sqlite")
    artifact = Artifact(
        id="artifact_1",
        source="test",
        kind="note",
        title="Memory project note",
        content="The memory system should use one universal substrate.",
    )
    span = Span(
        id="span_1",
        artifact_id=artifact.id,
        text="The memory system should use one universal substrate.",
        locator="line 1",
    )
    state = MemoryState(
        id="state_1",
        content="Use one universal memory substrate across apps.",
        source_span_ids=[span.id],
    )
    store.upsert_artifact(artifact)
    store.upsert_span(span)
    store.upsert_memory_state(state)
    store.log_trace(
        TraceEvent(
            id="trace_1",
            run_name="test",
            op="ingest",
            input={"source": "test"},
            output={"memory_state": state.id},
        )
    )
    store.commit()

    stats = store.stats()
    assert stats["artifacts"] == 1
    assert stats["spans"] == 1
    assert stats["memory_states"] == 1
    assert stats["trace_events"] == 1

    hits = store.retrieve("universal memory substrate", k=3)
    assert hits
    assert hits[0]["id"] in {"state_1", "span_1"}

    loaded = store.get_memory_state("state_1")
    assert loaded is not None
    prov = store.provenance_for_state(loaded)
    assert prov[0]["span_id"] == "span_1"
    assert prov[0]["artifact_title"] == "Memory project note"
    store.close()
