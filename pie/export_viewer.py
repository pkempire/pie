#!/usr/bin/env python3
"""
Export world model to a self-contained HTML viewer.

Generates a single HTML file with embedded JSON data and a full
interactive UI — search, filter, entity detail, timeline, graph stats.
No server needed. Just open in a browser.

Usage:
    python3 -m pie.export_viewer                           # default paths
    python3 -m pie.export_viewer --world-model output/world_model.json -o viewer.html
"""

from __future__ import annotations
import argparse
import json
import sys
import time
import datetime
from pathlib import Path


def _slim_data(data: dict, max_entities: int = 500) -> dict:
    """
    Slim down the world model for embedding in HTML.
    Keep top entities by transition count + recency.
    """
    entities = data.get("entities", {})
    transitions = data.get("transitions", {})
    relationships = data.get("relationships", {})

    # Count transitions per entity
    trans_count = {}
    for tid, t in transitions.items():
        eid = t.get("entity_id", "")
        trans_count[eid] = trans_count.get(eid, 0) + 1

    # Score entities: transitions * recency
    now = time.time()
    scored = []
    for eid, e in entities.items():
        last = e.get("last_seen", 0)
        recency = max(0, 1.0 - (now - last) / (365 * 86400)) if last > 0 else 0
        tc = trans_count.get(eid, 0)
        score = tc * (0.5 + recency)
        scored.append((score, eid))

    scored.sort(reverse=True)
    keep_ids = set(eid for _, eid in scored[:max_entities])

    # Slim entities (drop embeddings, trim state)
    slim_entities = {}
    for eid in keep_ids:
        e = entities[eid]
        slim = {
            "id": eid,
            "name": e.get("name", ""),
            "type": e.get("type", ""),
            "aliases": e.get("aliases", [])[:5],
            "first_seen": e.get("first_seen", 0),
            "last_seen": e.get("last_seen", 0),
        }
        # Slim current_state
        state = e.get("current_state", {})
        if isinstance(state, dict):
            desc = state.get("description", "")
            if desc:
                slim["state"] = desc[:300]
            else:
                slim["state"] = "; ".join(f"{k}: {str(v)[:50]}" for k, v in list(state.items())[:5])
        else:
            slim["state"] = str(state)[:300]

        slim_entities[eid] = slim

    # Slim transitions (only for kept entities)
    slim_transitions = []
    for tid, t in transitions.items():
        eid = t.get("entity_id", "")
        if eid in keep_ids:
            slim_transitions.append({
                "entity_id": eid,
                "timestamp": t.get("timestamp", 0),
                "type": t.get("transition_type", "update"),
                "summary": (t.get("trigger_summary", "") or "")[:120],
            })
    slim_transitions.sort(key=lambda t: t["timestamp"])

    # Slim relationships (only between kept entities)
    slim_rels = []
    for rid, r in relationships.items():
        src = r.get("source_id", "")
        tgt = r.get("target_id", "")
        if src in keep_ids and tgt in keep_ids:
            slim_rels.append({
                "source": src,
                "target": tgt,
                "type": r.get("type", "related_to"),
                "desc": (r.get("description", "") or "")[:80],
            })

    return {
        "entities": slim_entities,
        "transitions": slim_transitions,
        "relationships": slim_rels,
        "meta": {
            "total_entities": len(entities),
            "total_transitions": len(transitions),
            "total_relationships": len(relationships),
            "shown_entities": len(slim_entities),
            "exported_at": datetime.datetime.now(datetime.timezone.utc).isoformat(),
        }
    }


HTML_TEMPLATE = """<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="utf-8">
<title>PIE World Model Viewer</title>
<style>
  * { box-sizing: border-box; margin: 0; padding: 0; }
  body { font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', system-ui, sans-serif;
         background: #0a0a0f; color: #e0e0e0; }
  .header { background: #12121a; padding: 16px 24px; border-bottom: 1px solid #222;
            display: flex; align-items: center; gap: 16px; }
  .header h1 { font-size: 18px; color: #fff; font-weight: 600; }
  .header .meta { color: #888; font-size: 13px; }
  .container { display: flex; height: calc(100vh - 57px); }
  .sidebar { width: 380px; border-right: 1px solid #222; display: flex; flex-direction: column; }
  .search-box { padding: 12px; border-bottom: 1px solid #222; }
  .search-box input { width: 100%; padding: 8px 12px; background: #1a1a24; border: 1px solid #333;
                      border-radius: 6px; color: #fff; font-size: 14px; outline: none; }
  .search-box input:focus { border-color: #5a5aff; }
  .filters { padding: 8px 12px; border-bottom: 1px solid #222; display: flex; flex-wrap: wrap; gap: 4px; }
  .filter-btn { padding: 3px 10px; border-radius: 12px; border: 1px solid #333; background: transparent;
                color: #aaa; font-size: 11px; cursor: pointer; transition: all 0.15s; }
  .filter-btn:hover { border-color: #5a5aff; color: #fff; }
  .filter-btn.active { background: #5a5aff; border-color: #5a5aff; color: #fff; }
  .entity-list { flex: 1; overflow-y: auto; }
  .entity-item { padding: 10px 16px; border-bottom: 1px solid #1a1a24; cursor: pointer; transition: background 0.1s; }
  .entity-item:hover { background: #16161f; }
  .entity-item.selected { background: #1a1a2e; border-left: 3px solid #5a5aff; }
  .entity-item .name { font-size: 14px; font-weight: 500; color: #fff; }
  .entity-item .info { font-size: 11px; color: #666; margin-top: 2px; }
  .entity-item .type-badge { display: inline-block; padding: 1px 6px; border-radius: 8px;
                             font-size: 10px; font-weight: 500; margin-left: 6px; }
  .main { flex: 1; overflow-y: auto; padding: 24px; }
  .detail-header { margin-bottom: 20px; }
  .detail-header h2 { font-size: 22px; color: #fff; margin-bottom: 4px; }
  .detail-header .subtitle { color: #888; font-size: 13px; }
  .section { margin-bottom: 24px; }
  .section h3 { font-size: 14px; color: #5a5aff; margin-bottom: 10px; text-transform: uppercase;
                letter-spacing: 0.5px; }
  .state-box { background: #12121a; padding: 14px; border-radius: 8px; font-size: 13px;
               line-height: 1.5; border: 1px solid #222; }
  .timeline-item { display: flex; gap: 12px; padding: 8px 0; border-bottom: 1px solid #1a1a1a; font-size: 13px; }
  .timeline-date { color: #666; min-width: 90px; font-size: 12px; }
  .timeline-icon { width: 20px; text-align: center; }
  .timeline-text { color: #ccc; flex: 1; }
  .rel-item { padding: 6px 0; font-size: 13px; border-bottom: 1px solid #1a1a1a; }
  .rel-type { color: #5a5aff; font-weight: 500; }
  .rel-name { color: #fff; cursor: pointer; }
  .rel-name:hover { text-decoration: underline; }
  .stat-grid { display: grid; grid-template-columns: repeat(auto-fill, minmax(160px, 1fr)); gap: 12px; margin-bottom: 20px; }
  .stat-card { background: #12121a; padding: 16px; border-radius: 8px; border: 1px solid #222; }
  .stat-card .value { font-size: 28px; font-weight: 700; color: #fff; }
  .stat-card .label { font-size: 12px; color: #666; margin-top: 4px; }
  .aliases { display: flex; flex-wrap: wrap; gap: 4px; margin-top: 8px; }
  .alias-tag { padding: 2px 8px; background: #1a1a2e; border-radius: 10px; font-size: 11px; color: #aaa; }
  .empty-state { text-align: center; padding: 60px 20px; color: #444; }
  .empty-state h2 { font-size: 18px; margin-bottom: 8px; color: #666; }
  .type-colors .person { background: #1a2e1a; color: #6f6; }
  .type-colors .project { background: #2e2e1a; color: #ff6; }
  .type-colors .tool { background: #1a1a2e; color: #66f; }
  .type-colors .organization { background: #2e1a2e; color: #f6f; }
  .type-colors .goal { background: #2e1a1a; color: #f66; }
  .type-colors .decision { background: #1a2e2e; color: #6ff; }
  .type-colors .belief { background: #2e2a1a; color: #fa6; }
  .type-colors .concept { background: #222; color: #aaa; }
  .type-colors .event { background: #1a2a2e; color: #6af; }
  .type-colors .period { background: #2a1a2e; color: #a6f; }
  ::-webkit-scrollbar { width: 6px; }
  ::-webkit-scrollbar-track { background: transparent; }
  ::-webkit-scrollbar-thumb { background: #333; border-radius: 3px; }
</style>
</head>
<body class="type-colors">
<div class="header">
  <h1>PIE World Model</h1>
  <span class="meta" id="header-meta"></span>
</div>
<div class="container">
  <div class="sidebar">
    <div class="search-box">
      <input type="text" id="search" placeholder="Search entities..." autofocus>
    </div>
    <div class="filters" id="filters"></div>
    <div class="entity-list" id="entity-list"></div>
  </div>
  <div class="main" id="main">
    <div class="empty-state">
      <h2>Select an entity to view details</h2>
      <p>Use search or click an entity in the sidebar</p>
    </div>
  </div>
</div>

<script>
const DATA = __DATA_PLACEHOLDER__;

const entities = DATA.entities;
const transitions = DATA.transitions;
const relationships = DATA.relationships;
const meta = DATA.meta;

// Index
const transByEntity = {};
transitions.forEach(t => {
  if (!transByEntity[t.entity_id]) transByEntity[t.entity_id] = [];
  transByEntity[t.entity_id].push(t);
});

const relsByEntity = {};
relationships.forEach(r => {
  if (!relsByEntity[r.source]) relsByEntity[r.source] = [];
  relsByEntity[r.source].push(r);
  if (!relsByEntity[r.target]) relsByEntity[r.target] = [];
  relsByEntity[r.target].push(r);
});

// Sort entities by last_seen desc
const sortedIds = Object.keys(entities).sort((a, b) =>
  (entities[b].last_seen || 0) - (entities[a].last_seen || 0)
);

// Types
const typeCounts = {};
sortedIds.forEach(id => {
  const t = entities[id].type || 'unknown';
  typeCounts[t] = (typeCounts[t] || 0) + 1;
});

// Header
document.getElementById('header-meta').textContent =
  `${meta.shown_entities} of ${meta.total_entities} entities | ${meta.total_transitions} transitions | ${meta.total_relationships} relationships`;

// Filters
let activeFilter = null;
const filtersEl = document.getElementById('filters');
const allTypes = Object.keys(typeCounts).sort((a,b) => typeCounts[b] - typeCounts[a]);
allTypes.forEach(type => {
  const btn = document.createElement('button');
  btn.className = 'filter-btn';
  btn.textContent = `${type} (${typeCounts[type]})`;
  btn.onclick = () => {
    if (activeFilter === type) { activeFilter = null; btn.classList.remove('active'); }
    else {
      document.querySelectorAll('.filter-btn').forEach(b => b.classList.remove('active'));
      activeFilter = type; btn.classList.add('active');
    }
    renderList();
  };
  filtersEl.appendChild(btn);
});

// Helpers
function ago(ts) {
  if (!ts || ts <= 0) return 'unknown';
  const days = (Date.now()/1000 - ts) / 86400;
  if (days < 0) return 'future';
  if (days < 1) return 'today';
  if (days < 2) return 'yesterday';
  if (days < 7) return Math.floor(days) + 'd ago';
  if (days < 30) return Math.floor(days/7) + 'w ago';
  if (days < 365) return Math.floor(days/30) + 'mo ago';
  return (days/365).toFixed(1) + 'y ago';
}

function dateStr(ts) {
  if (!ts || ts <= 0) return 'unknown';
  return new Date(ts * 1000).toISOString().split('T')[0];
}

function renderList() {
  const query = document.getElementById('search').value.toLowerCase();
  const listEl = document.getElementById('entity-list');
  listEl.innerHTML = '';

  let ids = sortedIds;
  if (activeFilter) ids = ids.filter(id => entities[id].type === activeFilter);
  if (query) {
    ids = ids.filter(id => {
      const e = entities[id];
      const haystack = (e.name + ' ' + (e.aliases||[]).join(' ')).toLowerCase();
      return haystack.includes(query);
    });
  }

  ids.slice(0, 200).forEach(id => {
    const e = entities[id];
    const tc = (transByEntity[id] || []).length;
    const div = document.createElement('div');
    div.className = 'entity-item';
    div.innerHTML = `
      <div class="name">${esc(e.name)} <span class="type-badge ${e.type}">${e.type}</span></div>
      <div class="info">${ago(e.last_seen)} · ${tc} changes</div>
    `;
    div.onclick = () => selectEntity(id);
    listEl.appendChild(div);
  });

  if (ids.length > 200) {
    const more = document.createElement('div');
    more.className = 'entity-item';
    more.innerHTML = `<div class="info">... ${ids.length - 200} more entities</div>`;
    listEl.appendChild(more);
  }
}

function esc(s) { const d = document.createElement('div'); d.textContent = s; return d.innerHTML; }

function selectEntity(id) {
  document.querySelectorAll('.entity-item').forEach(el => el.classList.remove('selected'));
  // Find and highlight
  const items = document.querySelectorAll('.entity-item');
  items.forEach(el => { if (el.onclick && el.textContent.includes(entities[id].name)) el.classList.add('selected'); });

  const e = entities[id];
  const trans = transByEntity[id] || [];
  const rels = relsByEntity[id] || [];
  const mainEl = document.getElementById('main');

  let html = `
    <div class="detail-header">
      <h2>${esc(e.name)} <span class="type-badge ${e.type}">${e.type}</span></h2>
      <div class="subtitle">First seen: ${dateStr(e.first_seen)} (${ago(e.first_seen)}) · Last: ${dateStr(e.last_seen)} (${ago(e.last_seen)}) · ${trans.length} transitions</div>
      ${e.aliases && e.aliases.length ? `<div class="aliases">${e.aliases.map(a => `<span class="alias-tag">${esc(a)}</span>`).join('')}</div>` : ''}
    </div>
  `;

  if (e.state) {
    html += `<div class="section"><h3>Current State</h3><div class="state-box">${esc(e.state)}</div></div>`;
  }

  if (trans.length) {
    const icons = { creation: '★', contradiction: '⚠', update: '•', resolution: '✓', archival: '†' };
    html += `<div class="section"><h3>Timeline (${trans.length})</h3>`;
    trans.slice(-30).reverse().forEach(t => {
      html += `<div class="timeline-item">
        <span class="timeline-date">${dateStr(t.timestamp)}</span>
        <span class="timeline-icon">${icons[t.type] || '•'}</span>
        <span class="timeline-text">${esc(t.summary || '(no summary)')}</span>
      </div>`;
    });
    if (trans.length > 30) html += `<div class="timeline-item"><span class="timeline-text" style="color:#555">... ${trans.length-30} earlier</span></div>`;
    html += `</div>`;
  }

  if (rels.length) {
    html += `<div class="section"><h3>Relationships (${rels.length})</h3>`;
    rels.slice(0, 20).forEach(r => {
      const otherId = r.source === id ? r.target : r.source;
      const other = entities[otherId];
      const otherName = other ? other.name : '?';
      const dir = r.source === id ? '→' : '←';
      html += `<div class="rel-item">${dir} <span class="rel-type">${esc(r.type)}</span>: <span class="rel-name" onclick="selectEntity('${otherId}')">${esc(otherName)}</span>${r.desc ? ` <span style="color:#555">(${esc(r.desc)})</span>` : ''}</div>`;
    });
    html += `</div>`;
  }

  mainEl.innerHTML = html;
}

document.getElementById('search').addEventListener('input', renderList);
renderList();
</script>
</body>
</html>"""


def export_viewer(data: dict, output: Path, max_entities: int = 500):
    """Export world model to self-contained HTML viewer."""
    slim = _slim_data(data, max_entities=max_entities)
    json_str = json.dumps(slim, separators=(',', ':'))
    html = HTML_TEMPLATE.replace('__DATA_PLACEHOLDER__', json_str)
    output.write_text(html)
    size_mb = output.stat().st_size / (1024 * 1024)
    print(f"Exported viewer to {output} ({size_mb:.1f} MB)")
    print(f"  {slim['meta']['shown_entities']} entities, "
          f"{len(slim['transitions'])} transitions, "
          f"{len(slim['relationships'])} relationships")


def main():
    parser = argparse.ArgumentParser(description="Export PIE world model to HTML viewer")
    parser.add_argument("--world-model", type=Path, default=Path("output/world_model.json"))
    parser.add_argument("-o", "--output", type=Path, default=Path("output/viewer.html"))
    parser.add_argument("--max-entities", type=int, default=500,
                        help="Max entities to include (default: 500, sorted by activity)")
    args = parser.parse_args()

    if not args.world_model.exists():
        print(f"World model not found at {args.world_model}")
        sys.exit(1)

    print(f"Loading {args.world_model}...")
    with open(args.world_model) as f:
        data = json.load(f)

    export_viewer(data, args.output, max_entities=args.max_entities)
    print(f"\nOpen in browser: file://{args.output.resolve()}")


if __name__ == "__main__":
    main()
