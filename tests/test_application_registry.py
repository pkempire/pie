from mempol.applications.registry import application_targets, to_markdown


def test_application_registry_has_core_targets():
    rows = application_targets()
    names = {r.name for r in rows}
    assert "Long-running project continuity" in names
    assert "Temporal tool-use / stale context" in names
    assert "AI scientist / auto-research memory" in names


def test_application_registry_markdown_contains_benchmarks():
    md = to_markdown(application_targets(in_scope_only=True))
    assert "LongMemEval" in md
    assert "Robotouille" in md
    assert "SWE-bench" in md
