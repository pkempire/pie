#!/usr/bin/env python3
"""
PIE World Model Browser — zero-dependency offline viewer.

No API keys, no LLM calls, no embeddings. Just reads the JSON and lets you explore.

Usage:
    python3 -m pie.browse                              # interactive mode
    python3 -m pie.browse --world-model output/world_model.json

Commands (in interactive mode):
    stats                          — overall world model stats
    entities [type]                — list entities (optionally filter by type)
    entity <name>                  — show entity detail + timeline
    recent [n]                     — show n most recently changed entities
    stale [n]                      — show n most stale entities (long time since change)
    search <term>                  — fuzzy search entities by name/alias
    relationships <name>           — show all relationships for an entity
    threads                        — show entity clusters that change together
    types                          — show entity type distribution
    timeline [days]                — show all transitions in last N days
    quit                           — exit
"""

from __future__ import annotations
import argparse
import json
import sys
import datetime
from pathlib import Path
from collections import Counter, defaultdict
from difflib import SequenceMatcher


def _normalize(s: str) -> str:
    return s.lower().strip().replace("-", " ").replace("_", " ")


def _fuzzy(a: str, b: str) -> float:
    return SequenceMatcher(None, _normalize(a), _normalize(b)).ratio()


def _ts_to_str(ts: float) -> str:
    """Timestamp to human string."""
    if ts <= 0:
        return "unknown"
    try:
        dt = datetime.datetime.fromtimestamp(ts, tz=datetime.timezone.utc)
        return dt.strftime("%Y-%m-%d")
    except (ValueError, OSError):
        return "unknown"


def _ago(ts: float, now: float) -> str:
    """Time since timestamp as human string."""
    if ts <= 0:
        return "unknown"
    days = (now - ts) / 86400
    if days < 0:
        return "in the future"
    if days < 1:
        return "today"
    if days < 2:
        return "yesterday"
    if days < 7:
        return f"{int(days)}d ago"
    if days < 30:
        return f"{int(days/7)}w ago"
    if days < 365:
        return f"{int(days/30)}mo ago"
    return f"{days/365:.1f}y ago"


class WorldModelBrowser:
    def __init__(self, data: dict):
        self.entities = data.get("entities", {})
        self.transitions = data.get("transitions", {})
        self.relationships = data.get("relationships", {})
        self.now = __import__("time").time()

        # Index transitions by entity
        self.trans_by_entity: dict[str, list[dict]] = defaultdict(list)
        for tid, t in self.transitions.items():
            eid = t.get("entity_id", "")
            self.trans_by_entity[eid].append(t)
        for eid in self.trans_by_entity:
            self.trans_by_entity[eid].sort(key=lambda t: t.get("timestamp", 0))

        # Index relationships by entity
        self.rels_by_entity: dict[str, list[dict]] = defaultdict(list)
        for rid, r in self.relationships.items():
            for eid in (r.get("source_id", ""), r.get("target_id", "")):
                if eid:
                    self.rels_by_entity[eid].append(r)

        # Name → entity_id index
        self.name_index: dict[str, str] = {}
        for eid, e in self.entities.items():
            self.name_index[_normalize(e.get("name", ""))] = eid
            for alias in e.get("aliases", []):
                self.name_index[_normalize(alias)] = eid

    def _find_entity(self, name: str) -> tuple[str, dict] | None:
        """Find entity by name or alias (exact then fuzzy)."""
        norm = _normalize(name)
        # Exact match
        if norm in self.name_index:
            eid = self.name_index[norm]
            return eid, self.entities[eid]
        # Fuzzy match
        best_score, best_id = 0.0, None
        for n, eid in self.name_index.items():
            score = _fuzzy(norm, n)
            if score > best_score:
                best_score = score
                best_id = eid
        if best_score > 0.5 and best_id:
            return best_id, self.entities[best_id]
        return None

    def cmd_stats(self):
        """Show overall stats."""
        types = Counter(e.get("type", "unknown") for e in self.entities.values())
        ttypes = Counter(t.get("transition_type", "unknown") for t in self.transitions.values())

        # Time range
        all_ts = [t.get("timestamp", 0) for t in self.transitions.values() if t.get("timestamp", 0) > 0]
        first = min(all_ts) if all_ts else 0
        last = max(all_ts) if all_ts else 0

        print(f"\n{'='*60}")
        print(f"  PIE World Model")
        print(f"{'='*60}")
        print(f"  Entities:      {len(self.entities):,}")
        print(f"  Transitions:   {len(self.transitions):,}")
        print(f"  Relationships: {len(self.relationships):,}")
        print(f"  Time range:    {_ts_to_str(first)} → {_ts_to_str(last)}")
        if all_ts:
            span_days = (last - first) / 86400
            print(f"  Span:          {span_days:.0f} days ({span_days/30:.1f} months)")
            print(f"  Avg changes/day: {len(all_ts) / max(span_days, 1):.1f}")
        print(f"\n  Entity types:")
        for t, c in types.most_common():
            print(f"    {t:20s} {c:5d}")
        print(f"\n  Transition types:")
        for t, c in ttypes.most_common():
            print(f"    {t:20s} {c:5d}")

    def cmd_types(self):
        """Show entity type distribution."""
        types = Counter(e.get("type", "unknown") for e in self.entities.values())
        print(f"\n  Entity type distribution:")
        for t, c in types.most_common():
            bar = "█" * min(c // 5, 50)
            print(f"    {t:20s} {c:5d}  {bar}")

    def cmd_entities(self, entity_type: str | None = None):
        """List entities, optionally filtered by type."""
        items = []
        for eid, e in self.entities.items():
            if entity_type and e.get("type", "") != entity_type:
                continue
            last_seen = e.get("last_seen", 0)
            n_trans = len(self.trans_by_entity.get(eid, []))
            items.append((e.get("name", "?"), e.get("type", "?"), last_seen, n_trans, eid))

        items.sort(key=lambda x: x[2], reverse=True)

        type_str = f" (type={entity_type})" if entity_type else ""
        print(f"\n  {len(items)} entities{type_str}, sorted by last seen:\n")
        print(f"  {'Name':<40s} {'Type':<15s} {'Last seen':<12s} {'Changes':>7s}")
        print(f"  {'─'*40} {'─'*15} {'─'*12} {'─'*7}")
        for name, etype, last_seen, n_trans, eid in items[:50]:
            print(f"  {name[:40]:<40s} {etype:<15s} {_ago(last_seen, self.now):<12s} {n_trans:>7d}")
        if len(items) > 50:
            print(f"\n  ... and {len(items)-50} more. Use 'search <term>' to find specific entities.")

    def cmd_entity(self, name: str):
        """Show detailed entity info + timeline."""
        result = self._find_entity(name)
        if not result:
            print(f"  Entity not found: '{name}'. Try 'search {name}'")
            return
        eid, entity = result
        transitions = self.trans_by_entity.get(eid, [])
        relationships = self.rels_by_entity.get(eid, [])

        print(f"\n{'='*60}")
        print(f"  {entity.get('name', '?')} ({entity.get('type', '?')})")
        print(f"{'='*60}")

        # Aliases
        aliases = entity.get("aliases", [])
        if aliases:
            print(f"  Aliases: {', '.join(aliases[:5])}")
            if len(aliases) > 5:
                print(f"           ... and {len(aliases)-5} more")

        # Temporal info
        first_seen = entity.get("first_seen", 0)
        last_seen = entity.get("last_seen", 0)
        print(f"  First seen:  {_ts_to_str(first_seen)} ({_ago(first_seen, self.now)})")
        print(f"  Last seen:   {_ts_to_str(last_seen)} ({_ago(last_seen, self.now)})")
        print(f"  Transitions: {len(transitions)}")

        # Current state
        state = entity.get("current_state", {})
        if state:
            print(f"\n  Current state:")
            if isinstance(state, dict):
                for k, v in state.items():
                    v_str = str(v)[:100]
                    print(f"    {k}: {v_str}")
            else:
                print(f"    {str(state)[:200]}")

        # Timeline
        if transitions:
            print(f"\n  Timeline ({len(transitions)} transitions):")
            for t in transitions[-20:]:  # last 20
                ts = t.get("timestamp", 0)
                ttype = t.get("transition_type", "?")
                summary = t.get("trigger_summary", "")[:80]
                icon = {"creation": "★", "contradiction": "⚠", "update": "•", "resolution": "✓", "archival": "†"}.get(ttype, "•")
                print(f"    {icon} {_ts_to_str(ts)} ({_ago(ts, self.now):>8s})  {summary}")
            if len(transitions) > 20:
                print(f"    ... {len(transitions)-20} earlier transitions omitted")

        # Relationships
        if relationships:
            print(f"\n  Relationships ({len(relationships)}):")
            for r in relationships[:10]:
                src = r.get("source_id", "")
                tgt = r.get("target_id", "")
                rtype = r.get("type", "related_to")
                desc = r.get("description", "")[:60]
                other_id = tgt if src == eid else src
                other = self.entities.get(other_id, {})
                other_name = other.get("name", "?")
                direction = "→" if src == eid else "←"
                print(f"    {direction} {rtype}: {other_name}" + (f" ({desc})" if desc else ""))
            if len(relationships) > 10:
                print(f"    ... and {len(relationships)-10} more")

    def cmd_recent(self, n: int = 15):
        """Show most recently changed entities."""
        items = []
        for eid, e in self.entities.items():
            last = e.get("last_seen", 0)
            if last > 0:
                items.append((last, e.get("name", "?"), e.get("type", "?"), len(self.trans_by_entity.get(eid, []))))
        items.sort(reverse=True)

        print(f"\n  {min(n, len(items))} most recently changed entities:\n")
        print(f"  {'Last seen':<14s} {'Name':<40s} {'Type':<15s} {'Changes':>7s}")
        print(f"  {'─'*14} {'─'*40} {'─'*15} {'─'*7}")
        for last, name, etype, ntrans in items[:n]:
            print(f"  {_ts_to_str(last)} ({_ago(last, self.now):>6s})  {name[:40]:<40s} {etype:<15s} {ntrans:>7d}")

    def cmd_stale(self, n: int = 15):
        """Show most stale entities (longest since last change, but historically active)."""
        items = []
        for eid, e in self.entities.items():
            last = e.get("last_seen", 0)
            ntrans = len(self.trans_by_entity.get(eid, []))
            if last > 0 and ntrans >= 3:  # only entities with history
                staleness = self.now - last
                items.append((staleness, e.get("name", "?"), e.get("type", "?"), ntrans, last))
        items.sort(reverse=True)

        print(f"\n  {min(n, len(items))} most stale entities (≥3 transitions, sorted by time since last change):\n")
        print(f"  {'Last seen':<14s} {'Stale':<10s} {'Name':<35s} {'Type':<15s} {'Changes':>7s}")
        print(f"  {'─'*14} {'─'*10} {'─'*35} {'─'*15} {'─'*7}")
        for staleness, name, etype, ntrans, last in items[:n]:
            print(f"  {_ts_to_str(last)} ({_ago(last, self.now):>6s})  {name[:35]:<35s} {etype:<15s} {ntrans:>7d}")

    def cmd_search(self, term: str):
        """Fuzzy search entities by name/alias."""
        norm = _normalize(term)
        matches = []
        seen = set()
        for eid, e in self.entities.items():
            best = 0.0
            name = e.get("name", "")
            # Substring match
            if norm in _normalize(name):
                best = max(best, 0.95)
            for alias in e.get("aliases", []):
                if norm in _normalize(alias):
                    best = max(best, 0.90)
            # Fuzzy
            best = max(best, _fuzzy(norm, name))
            for alias in e.get("aliases", []):
                best = max(best, _fuzzy(norm, alias) * 0.95)

            if best > 0.4 and eid not in seen:
                seen.add(eid)
                matches.append((best, name, e.get("type", "?"), eid))

        matches.sort(reverse=True)
        if not matches:
            print(f"  No matches for '{term}'")
            return

        print(f"\n  {len(matches)} matches for '{term}':\n")
        for score, name, etype, eid in matches[:20]:
            print(f"    {score:.2f}  {name:<40s} ({etype})")

    def cmd_timeline(self, days: int = 7):
        """Show all transitions in last N days."""
        cutoff = self.now - days * 86400
        recent = []
        for tid, t in self.transitions.items():
            ts = t.get("timestamp", 0)
            if ts > cutoff:
                eid = t.get("entity_id", "")
                ename = self.entities.get(eid, {}).get("name", "?")
                recent.append((ts, ename, t.get("transition_type", "?"), t.get("trigger_summary", "")[:60]))

        recent.sort(reverse=True)
        print(f"\n  {len(recent)} transitions in last {days} days:\n")
        for ts, name, ttype, summary in recent[:50]:
            icon = {"creation": "★", "contradiction": "⚠", "update": "•", "resolution": "✓"}.get(ttype, "•")
            print(f"    {_ts_to_str(ts)}  {icon} {name:<30s}  {summary}")
        if len(recent) > 50:
            print(f"\n  ... and {len(recent)-50} more")

    def cmd_relationships(self, name: str):
        """Show all relationships for an entity."""
        result = self._find_entity(name)
        if not result:
            print(f"  Entity not found: '{name}'")
            return
        eid, entity = result
        rels = self.rels_by_entity.get(eid, [])
        print(f"\n  Relationships for {entity.get('name', '?')} ({len(rels)}):\n")
        for r in rels:
            src = r.get("source_id", "")
            tgt = r.get("target_id", "")
            rtype = r.get("type", "related_to")
            desc = r.get("description", "")
            other_id = tgt if src == eid else src
            other = self.entities.get(other_id, {})
            other_name = other.get("name", "?")
            direction = "→" if src == eid else "←"
            print(f"    {direction} {rtype}: {other_name}")
            if desc:
                print(f"      {desc[:100]}")


def interactive(browser: WorldModelBrowser):
    """Interactive REPL."""
    print("\n" + "=" * 60)
    print("  PIE World Model Browser")
    print("  Type 'help' for commands, 'quit' to exit")
    print("=" * 60)
    browser.cmd_stats()

    while True:
        try:
            raw = input("\npie> ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\nBye!")
            break

        if not raw:
            continue

        parts = raw.split(maxsplit=1)
        cmd = parts[0].lower()
        arg = parts[1].strip() if len(parts) > 1 else ""

        if cmd in ("quit", "exit", "q"):
            print("Bye!")
            break
        elif cmd == "help":
            print(__doc__)
        elif cmd == "stats":
            browser.cmd_stats()
        elif cmd == "types":
            browser.cmd_types()
        elif cmd in ("entities", "ls"):
            browser.cmd_entities(arg if arg else None)
        elif cmd in ("entity", "show", "e"):
            if not arg:
                print("  Usage: entity <name>")
            else:
                browser.cmd_entity(arg)
        elif cmd == "recent":
            browser.cmd_recent(int(arg) if arg.isdigit() else 15)
        elif cmd == "stale":
            browser.cmd_stale(int(arg) if arg.isdigit() else 15)
        elif cmd in ("search", "find", "s"):
            if not arg:
                print("  Usage: search <term>")
            else:
                browser.cmd_search(arg)
        elif cmd in ("relationships", "rels"):
            if not arg:
                print("  Usage: relationships <entity name>")
            else:
                browser.cmd_relationships(arg)
        elif cmd == "timeline":
            browser.cmd_timeline(int(arg) if arg.isdigit() else 7)
        elif cmd == "threads":
            print("  (Coming soon — co-occurrence clusters)")
        else:
            # Treat unknown input as entity search
            browser.cmd_search(raw)


def main():
    parser = argparse.ArgumentParser(description="PIE World Model Browser — offline viewer")
    parser.add_argument("--world-model", type=Path, default=Path("output/world_model.json"),
                        help="Path to world_model.json")
    parser.add_argument("--cmd", type=str, default=None,
                        help="Run a single command and exit (e.g. 'stats', 'recent 10', 'entity PIE')")
    args = parser.parse_args()

    if not args.world_model.exists():
        # Try backup
        backup = Path("output/world_model_backup_20260212.json")
        if backup.exists():
            args.world_model = backup
        else:
            print(f"World model not found at {args.world_model}")
            sys.exit(1)

    print(f"Loading {args.world_model}...")
    with open(args.world_model) as f:
        data = json.load(f)

    browser = WorldModelBrowser(data)

    if args.cmd:
        parts = args.cmd.split(maxsplit=1)
        cmd = parts[0].lower()
        arg = parts[1] if len(parts) > 1 else ""
        dispatch = {
            "stats": lambda: browser.cmd_stats(),
            "types": lambda: browser.cmd_types(),
            "entities": lambda: browser.cmd_entities(arg if arg else None),
            "entity": lambda: browser.cmd_entity(arg),
            "recent": lambda: browser.cmd_recent(int(arg) if arg.isdigit() else 15),
            "stale": lambda: browser.cmd_stale(int(arg) if arg.isdigit() else 15),
            "search": lambda: browser.cmd_search(arg),
            "relationships": lambda: browser.cmd_relationships(arg),
            "timeline": lambda: browser.cmd_timeline(int(arg) if arg.isdigit() else 7),
        }
        fn = dispatch.get(cmd)
        if fn:
            fn()
        else:
            browser.cmd_search(args.cmd)
    else:
        interactive(browser)


if __name__ == "__main__":
    main()
