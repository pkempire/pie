"""Offline tests for the egocentric worldline compiler (no API calls)."""
from __future__ import annotations

from datetime import datetime, timezone

from mempol.temporal.worldline import (
    Event,
    arithmetic_card,
    calendar_delta,
    classify_question,
    compile_worldline,
    exact_delta_phrase,
    humanize_delta,
    offset_phrase,
)


def _ts(y, mo, d, h=12, mi=0):
    return datetime(y, mo, d, h, mi, tzinfo=timezone.utc).timestamp()


NOW = _ts(2023, 10, 20, 17, 2)


def test_calendar_delta_exact():
    a = datetime(2023, 5, 8, tzinfo=timezone.utc)
    b = datetime(2023, 10, 20, tzinfo=timezone.utc)
    assert calendar_delta(a, b) == (0, 5, 12)
    # symmetric
    assert calendar_delta(b, a) == (0, 5, 12)
    # year boundary with day borrow
    a = datetime(2022, 12, 31, tzinfo=timezone.utc)
    b = datetime(2024, 1, 1, tzinfo=timezone.utc)
    assert calendar_delta(a, b) == (1, 0, 1)


def test_exact_delta_phrase():
    assert exact_delta_phrase(_ts(2023, 5, 8), NOW) == "5 months, 12 days"
    assert exact_delta_phrase(_ts(2023, 10, 20, 10), NOW) == "0 days"


def test_offset_phrase():
    assert offset_phrase(_ts(2023, 5, 8), NOW) == "5 months, 12 days ago"
    assert offset_phrase(_ts(2023, 10, 20, 9), NOW) == "today"
    assert offset_phrase(_ts(2023, 11, 1), NOW).startswith("in ")


def test_humanize_delta_bands():
    assert humanize_delta(30) == "moments"
    assert humanize_delta(3 * 3600) == "3 hours"
    assert humanize_delta(5 * 86400) == "5 days"
    assert humanize_delta(25 * 86400) == "3 weeks, 4 days"
    assert humanize_delta(100 * 86400).startswith("3 months")
    assert humanize_delta(500 * 86400).startswith("1 year")


def test_worldline_ordering_gaps_and_staleness():
    events = [
        Event(ts=_ts(2023, 8, 1), text="second", source_id="D2:1", speaker="A", session="2"),
        Event(ts=_ts(2023, 5, 8), text="first", source_id="D1:1", speaker="A", session="1"),
        Event(ts=0.0, text="undated note", source_id="D9:9"),
    ]
    out = compile_worldline(events, NOW)
    # chronological order regardless of input order
    assert out.index("first") < out.index("second")
    # gap marker between May and August, flagged as long
    assert "pass -- long gap" in out
    # egocentric offset present
    assert "5 months, 12 days ago" in out
    # NOW header and weekday rendering
    assert "NOW = Fri 2023-10-20" in out
    # undated evidence sectioned off, not silently dropped
    assert "UNDATED EVIDENCE" in out and "undated note" in out


def test_short_gap_not_flagged_volatile():
    events = [
        Event(ts=_ts(2023, 5, 8), text="a"),
        Event(ts=_ts(2023, 5, 12), text="b"),
    ]
    out = compile_worldline(events, NOW)
    assert "( 4 days pass )" in out
    assert "long gap" not in out


def test_arithmetic_card_now_and_reference():
    events = [
        Event(ts=_ts(2023, 5, 8), text="a"),
        Event(ts=_ts(2023, 8, 1), text="b"),
    ]
    card = arithmetic_card(events, NOW, reference_ts=_ts(2023, 6, 1), reference_label="in June")
    assert "5 months, 12 days before NOW (165 days)" in card
    assert "2023-05-08 -> Tue 2023-08-01" in card or "-> " in card  # consecutive gap section exists
    assert "85 days" in card  # May 8 -> Aug 1
    assert 'REFERENCE TIME = Thu 2023-06-01 ("in June")' in card
    assert "before the reference time" in card and "after the reference time" in card


def test_classify_duration():
    f = classify_question("How long has Caroline been vegetarian?")
    assert f.is_temporal and f.operator == "duration"


def test_classify_locate():
    f = classify_question("When did Melanie adopt her dog?")
    assert f.operator == "locate"


def test_classify_current_state():
    f = classify_question("Does John still work at the bakery?")
    assert f.operator == "current_state"


def test_classify_change():
    f = classify_question("Has Caroline changed her diet since last year?")
    assert f.operator == "change"


def test_classify_frequency():
    f = classify_question("How many times did they go hiking?")
    assert f.operator == "frequency"


def test_classify_non_temporal():
    f = classify_question("What is Caroline's favorite color?")
    assert not f.is_temporal and f.operator == "none"


def test_reference_resolution_full_date():
    ts_list = [_ts(2023, 5, 8), _ts(2023, 8, 1)]
    f = classify_question("What did Caroline say on 8 May 2023?", event_ts=ts_list, now_ts=NOW)
    assert f.operator == "point_in_time"
    assert f.reference_granularity == "day"
    ref = datetime.fromtimestamp(f.reference_ts, tz=timezone.utc)
    assert (ref.year, ref.month, ref.day) == (2023, 5, 8)


def test_reference_resolution_month_without_year():
    # "in May" must resolve to the May inside the worldline span, not need a year.
    ts_list = [_ts(2023, 5, 8), _ts(2023, 8, 1)]
    f = classify_question("What was Melanie planning in May?", event_ts=ts_list, now_ts=NOW)
    assert f.reference_ts is not None
    ref = datetime.fromtimestamp(f.reference_ts, tz=timezone.utc)
    assert (ref.year, ref.month) == (2023, 5)
    assert f.reference_granularity == "month"


def test_reference_resolution_ignores_future_month():
    # NOW is Oct 2023; "in December" should resolve to a past-or-near December,
    # not fabricate one outside the span. Dec 2022 not in candidates -> Dec 2023
    # is within the +31d tolerance? It's >31d after Oct 20 -> falls back to no pick
    # only if no candidate; 2023-12 is 42+ days out, so expect None or a past year.
    ts_list = [_ts(2022, 12, 5), _ts(2023, 8, 1)]
    f = classify_question("What happened in December?", event_ts=ts_list, now_ts=NOW)
    assert f.reference_ts is not None
    ref = datetime.fromtimestamp(f.reference_ts, tz=timezone.utc)
    assert (ref.year, ref.month) == (2022, 12)


if __name__ == "__main__":
    import sys
    import traceback

    fns = [v for k, v in sorted(globals().items()) if k.startswith("test_") and callable(v)]
    failed = 0
    for fn in fns:
        try:
            fn()
            print(f"PASS {fn.__name__}")
        except Exception:
            failed += 1
            print(f"FAIL {fn.__name__}")
            traceback.print_exc()
    print(f"\n{len(fns) - failed}/{len(fns)} passed")
    sys.exit(1 if failed else 0)
