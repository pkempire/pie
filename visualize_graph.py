"""
PIE Graph Visualizer — builds an interactive HTML visualization from extraction output.
Shows entities as nodes, relationships as edges, color-coded by type.
Grows incrementally to show the graph being built over time.
"""

import json
import os
from pyvis.network import Network

OUTPUT_DIR = os.path.dirname(__file__)
EXTRACTION_PATH = os.path.join(OUTPUT_DIR, "test_output.json")
VIZ_PATH = os.path.join(OUTPUT_DIR, "graph_viz.html")

# Color scheme by entity type
TYPE_COLORS = {
    "person": "#e74c3c",       # red
    "project": "#3498db",      # blue
    "tool": "#2ecc71",         # green
    "organization": "#9b59b6", # purple
    "belief": "#f39c12",       # orange
    "decision": "#e67e22",     # dark orange
    "concept": "#1abc9c",      # teal
    "period": "#34495e",       # dark gray
    "function": "#95a5a6",     # light gray (schema drift — flag these)
}

# Shape by entity type
TYPE_SHAPES = {
    "person": "dot",
    "project": "diamond",
    "tool": "triangle",
    "organization": "square",
    "belief": "star",
    "decision": "star",
    "concept": "dot",
    "period": "box",
    "function": "triangleDown",
}

REL_COLORS = {
    "uses": "#2ecc71",
    "works_on": "#3498db",
    "collaborates_with": "#e74c3c",
    "related_to": "#95a5a6",
    "part_of": "#9b59b6",
    "caused_by": "#f39c12",
    "decorates": "#1abc9c",
    "used_by": "#2ecc71",
    "executes": "#e67e22",
}


def build_graph(results):
    net = Network(
        height="900px",
        width="100%",
        bgcolor="#1a1a2e",
        font_color="#ffffff",
        directed=True,
        notebook=False,
    )
    
    # Physics settings for readable layout
    net.set_options("""
    {
        "nodes": {
            "borderWidth": 2,
            "borderWidthSelected": 4,
            "font": {
                "size": 14,
                "face": "Inter, system-ui, sans-serif"
            }
        },
        "edges": {
            "arrows": {
                "to": {"enabled": true, "scaleFactor": 0.5}
            },
            "color": {"inherit": false},
            "smooth": {"type": "continuous"},
            "font": {
                "size": 10,
                "color": "#aaaaaa",
                "strokeWidth": 0
            }
        },
        "physics": {
            "forceAtlas2Based": {
                "gravitationalConstant": -100,
                "centralGravity": 0.01,
                "springLength": 200,
                "springConstant": 0.02,
                "damping": 0.4
            },
            "solver": "forceAtlas2Based",
            "stabilization": {
                "iterations": 200
            }
        },
        "interaction": {
            "hover": true,
            "tooltipDelay": 100,
            "navigationButtons": true,
            "keyboard": true
        }
    }
    """)
    
    entity_nodes = {}  # name -> node data
    conversation_order = []
    
    for result in results:
        if "error" in result:
            continue
        
        meta = result["_meta"]
        conv_date = meta["date"]
        conv_title = meta["title"]
        significance = result.get("significance", 0.5)
        
        # Add conversation node (smaller, gray)
        conv_id = f"conv_{meta['conversation_id'][:8]}"
        net.add_node(
            conv_id,
            label=f"📄 {conv_title[:25]}",
            title=f"<b>{conv_title}</b><br>Date: {conv_date}<br>Significance: {significance}<br>User state: {result.get('user_state', 'unknown')}",
            color="#4a4a6a",
            size=10 + significance * 15,
            shape="box",
            font={"size": 10, "color": "#888888"},
        )
        conversation_order.append(conv_id)
        
        # Add entity nodes
        for entity in result.get("entities", []):
            name = entity["name"]
            etype = entity.get("type", "concept")
            state = entity.get("state", "")
            
            if name not in entity_nodes:
                color = TYPE_COLORS.get(etype, "#95a5a6")
                shape = TYPE_SHAPES.get(etype, "dot")
                
                net.add_node(
                    name,
                    label=name,
                    title=f"<b>{name}</b><br>Type: {etype}<br>State: {state}<br>First seen: {conv_date}",
                    color=color,
                    size=15,
                    shape=shape,
                    font={"size": 12},
                )
                entity_nodes[name] = {"type": etype, "conversations": [conv_date]}
            else:
                # Entity seen again — increase size
                entity_nodes[name]["conversations"].append(conv_date)
                current_size = 15 + len(entity_nodes[name]["conversations"]) * 3
                net.get_node(name)  # pyvis doesn't have great update, but nodes deduplicate
            
            # Edge from conversation to entity (extracted_from)
            net.add_edge(
                conv_id, name,
                color="#4a4a6a",
                width=0.5,
                dashes=True,
            )
        
        # Add relationship edges between entities
        for rel in result.get("relationships", []):
            source = rel.get("source", "")
            target = rel.get("target", "")
            rtype = rel.get("type", "related_to")
            desc = rel.get("description", "")
            
            if source in entity_nodes and target in entity_nodes:
                color = REL_COLORS.get(rtype, "#95a5a6")
                net.add_edge(
                    source, target,
                    label=rtype,
                    title=desc,
                    color=color,
                    width=2,
                )
        
        # Highlight state changes
        for sc in result.get("state_changes", []):
            entity_name = sc.get("entity_name", "")
            if entity_name in entity_nodes:
                # Add a visual indicator — we can't easily modify existing nodes in pyvis,
                # so add a small annotation node
                sc_id = f"sc_{entity_name}_{conv_date}"
                what = sc.get("what_changed", "")[:50]
                net.add_node(
                    sc_id,
                    label=f"⚡ {what}",
                    title=f"<b>State Change</b><br>Entity: {entity_name}<br>From: {sc.get('from_state', 'unknown')}<br>To: {sc.get('to_state', 'unknown')}<br>Contradiction: {sc.get('is_contradiction', False)}",
                    color="#f39c12" if not sc.get("is_contradiction") else "#e74c3c",
                    size=8,
                    shape="triangle",
                    font={"size": 9, "color": "#f39c12"},
                )
                net.add_edge(entity_name, sc_id, color="#f39c12", width=1, dashes=True)
    
    # Connect conversations chronologically
    for i in range(1, len(conversation_order)):
        net.add_edge(
            conversation_order[i-1],
            conversation_order[i],
            color="#2a2a4a",
            width=1,
            dashes=[5, 10],
        )
    
    return net


def add_legend_html(html_path):
    """Inject a legend into the generated HTML."""
    legend_html = """
    <div style="position: fixed; top: 10px; right: 10px; background: #1a1a2e; border: 1px solid #4a4a6a; 
                border-radius: 8px; padding: 15px; z-index: 1000; font-family: Inter, system-ui, sans-serif;
                color: white; font-size: 12px; max-width: 200px;">
        <div style="font-weight: bold; margin-bottom: 10px; font-size: 14px;">PIE World Model</div>
        <div style="margin-bottom: 5px;">🔴 Person</div>
        <div style="margin-bottom: 5px;">🔵 Project</div>
        <div style="margin-bottom: 5px;">🟢 Tool</div>
        <div style="margin-bottom: 5px;">🟣 Organization</div>
        <div style="margin-bottom: 5px;">🟠 Belief/Decision</div>
        <div style="margin-bottom: 5px;">🩵 Concept</div>
        <div style="margin-bottom: 5px;">⬜ Conversation</div>
        <div style="margin-bottom: 5px;">⚡ State Change</div>
        <hr style="border-color: #4a4a6a;">
        <div style="font-size: 10px; color: #888;">10 conversations · Jan-May 2025</div>
    </div>
    """
    
    with open(html_path, "r") as f:
        html = f.read()
    
    html = html.replace("<body>", f"<body>{legend_html}")
    
    with open(html_path, "w") as f:
        f.write(html)


def main():
    with open(EXTRACTION_PATH) as f:
        results = json.load(f)
    
    print(f"Building graph from {len(results)} extraction results...")
    net = build_graph(results)
    
    print(f"Saving to {VIZ_PATH}...")
    net.save_graph(VIZ_PATH)
    add_legend_html(VIZ_PATH)
    
    # Stats
    print(f"\nGraph stats:")
    print(f"  Nodes: {len(net.nodes)}")
    print(f"  Edges: {len(net.edges)}")
    print(f"  Saved to: {VIZ_PATH}")
    print(f"\nOpen in browser: file://{VIZ_PATH}")


if __name__ == "__main__":
    main()
