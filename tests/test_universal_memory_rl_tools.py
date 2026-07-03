from __future__ import annotations

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from mempol.core.schema import Artifact, Span
from mempol.core.store import SQLiteMemoryStore
from mempol.recipes.memory_rl.universal_tools import UniversalMemoryTool


def _tool_content(result):
    if isinstance(result, dict):
        return result["content"]
    if hasattr(result, "messages"):
        return result.messages[0]["content"]
    return result.content


def test_universal_rl_tools_force_memory_use(tmp_path):
    store = SQLiteMemoryStore(tmp_path / "env.sqlite")
    artifact = Artifact(
        id="a1",
        source="test",
        kind="note",
        title="Support group note",
        content="Caroline went to an LGBTQ support group yesterday.",
    )
    span = Span(
        id="s1",
        artifact_id="a1",
        text="Caroline went to an LGBTQ support group yesterday.",
        locator="D1:3",
    )
    store.upsert_artifact(artifact)
    store.upsert_span(span)
    store.commit()

    tool = UniversalMemoryTool(store=store)
    raw = json.loads(_tool_content(tool.search_raw_spans_impl("LGBTQ support group", k=3)))
    assert raw["hits"][0]["span_id"] == "s1"

    written = json.loads(
        _tool_content(tool.write_memory_state_impl(
            content="Caroline attended an LGBTQ support group.",
            source_span_ids=["s1"],
        ))
    )
    assert written["written_state_id"]

    frozen = json.loads(_tool_content(tool.freeze_raw_access_impl("memory built")))
    assert frozen["raw_enabled"] is False

    blocked = json.loads(_tool_content(tool.search_raw_spans_impl("LGBTQ", k=1)))
    assert blocked["error"] == "raw_span_search_disabled_after_memory_build"

    mem = json.loads(_tool_content(tool.retrieve_memory_states_impl("Caroline LGBTQ group", k=3)))
    assert mem["hits"][0]["memory_state_id"] == written["written_state_id"]
    assert tool.stats()["writes"] == 1
    store.close()
