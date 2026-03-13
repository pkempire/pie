# PIE World Model Explorer - User Guide

## Overview

The **PIE World Model Explorer** is a comprehensive, self-contained HTML dashboard for analyzing and exploring the Personal Intelligence Environment (PIE) world model. It provides deep analytical capabilities for understanding entity relationships, state transitions, and system evolution over time.

**Location:** `/mnt/personal-intelligence-system/explorer.html`

## Quick Start

1. Start a local HTTP server in the personal-intelligence-system directory:
   ```bash
   python3 -m http.server 8000
   ```

2. Open in browser:
   ```
   http://localhost:8000/explorer.html
   ```

3. Or with custom world model path:
   ```
   http://localhost:8000/explorer.html?wm=path/to/world_model.json
   ```

## Features

### 1. Overview Dashboard
The first tab displays system-level metrics and visualizations:

- **Key Metrics Cards:**
  - Total Entities: 1,447 entities in the world model
  - Total Relationships: 1,412 directed connections
  - Total Transitions: 3,033 state changes tracked
  - Entity Types: 9 different entity categories
  - Most Active: Entity with highest transition count
  - Timespan: Date range of entity creation

- **Type Distribution Chart:** Bar chart showing entity counts by type
  - person (👤), project (📋), tool (🔧)
  - organization (🏢), belief (💡), decision (⚖️)
  - concept (🧠), period (📅), event (📍)

- **Activity Timeline Chart:** Time-series visualization of transitions per day
  - Identifies peak activity periods
  - Shows system engagement over time

### 2. Entity Explorer
Interactive searchable database of all entities with detailed views:

**Search & Filter Controls:**
- Free-text search across entity names and aliases
- Filter by entity type
- Sort options:
  - By name (A-Z)
  - By type
  - By transition count (most active first)
  - By first seen (newest first)
  - By last seen (recently modified)

**Entity Detail Panel** (shown on right when entity selected):
- **Header Info:** Name, type badge, creation/modification dates
- **Current State:** Formatted JSON view of entity's current properties
  - Syntax highlighted JSON display
  - Scrollable for large states
  
- **Relationships Section:** All incoming and outgoing connections
  - Relationship type (e.g., "uses", "integrates_with", "implements")
  - Direction indicators (→ for outgoing, ← for incoming)
  - Full relationship descriptions
  - Clickable entity links to navigate

- **Transition History:** Complete chronological record
  - Transition type (creation, update, contradiction, etc.)
  - Timestamp
  - Trigger conversation summary
  - Reverse chronological (newest first)

### 3. Timeline View
Temporal visualization of entity creation and evolution:

**Horizontal Timeline:**
- Each bar represents a creation date
- Color-coded by entity type
- Shows count of entities created on that date
- Click bars to see details

**Use Cases:**
- Identify when major system components were introduced
- Understand system evolution phases
- Track when different entity types emerged

### 4. Relationship Explorer
Graph-based view of entity connections:

**How It Works:**
1. Search and select an entity from left panel
2. Center node shows selected entity with color-coded type
3. Surrounding nodes show connected entities
4. Lines represent relationships with type labels
5. Click any connected entity to re-center and explore further

**Visual Elements:**
- Entity icons and colors match type badges
- Relationship labels show connection type
- All connections are clickable to drill down
- Hover highlights relationships

**Navigation:**
- Start with high-level entities
- Click connected entities to explore branches
- Useful for understanding dependency chains

### 5. Insights Panel
Computed analytics and patterns discovered in world model:

**Most Active Entities:** Top 10 entities by transition count
- Shows which elements change most frequently
- Indicates core system components

**Recently Changed Entities:** Latest 10 modifications
- Recent updates and active development areas
- Useful for tracking current work

**Contradiction Hotspots:** Entities with conflicting states
- Identifies areas needing resolution
- Shows belief conflicts or decision reversals
- Count of contradictions per entity

**Orphan Entities:** Entities with no relationships
- Isolated concepts or one-off events
- May indicate incomplete model or independent items
- Useful for data quality assessment

**Belief Evolution:** Statistics on belief entities
- Total beliefs in system
- Count of belief state transitions
- Average transitions per belief
- Indicates conviction changes

**Relationship Statistics:**
- Total relationships in model
- Average connections per entity
- Shows network density and interconnectedness

## Data Structure

The explorer loads from `output/world_model.json`:

```json
{
  "entities": {
    "uuid": {
      "id": "uuid",
      "type": "person|project|tool|organization|belief|decision|concept|period|event",
      "name": "string",
      "aliases": ["string"],
      "current_state": { ... },
      "created_from": "conversation_id",
      "first_seen": unix_timestamp,
      "last_seen": unix_timestamp
    }
  },
  "relationships": {
    "uuid": {
      "id": "uuid",
      "source_id": "entity_uuid",
      "target_id": "entity_uuid",
      "type": "uses|integrates_with|implements|describes|etc",
      "description": "string",
      "timestamp": unix_timestamp
    }
  },
  "transitions": {
    "uuid": {
      "id": "uuid",
      "entity_id": "entity_uuid",
      "from_state": { ... },
      "to_state": { ... },
      "transition_type": "creation|update|contradiction|etc",
      "timestamp": unix_timestamp
    }
  }
}
```

## Color Scheme

Entity types use consistent colors throughout:

| Type | Color | Icon | Use |
|------|-------|------|-----|
| person | #4fc3f7 (cyan) | 👤 | People, users |
| project | #81c784 (green) | 📋 | Projects, applications |
| tool | #ffb74d (orange) | 🔧 | Tools, libraries |
| organization | #ce93d8 (purple) | 🏢 | Companies, teams |
| belief | #ef5350 (red) | 💡 | Beliefs, opinions |
| decision | #ffd54f (yellow) | ⚖️ | Decisions made |
| concept | #90a4ae (gray) | 🧠 | Abstract concepts |
| period | #4db6ac (teal) | 📅 | Time periods |
| event | #7986cb (indigo) | 📍 | Events, occurrences |

## Performance

The explorer handles the full dataset efficiently:
- 1,447 entities loaded instantly
- 1,412 relationships rendered interactively
- 3,033 transitions indexed for fast lookup
- Vanilla JavaScript with no external dependencies
- Responsive filtering and sorting
- Canvas-based charts render in real-time

## Tips for Effective Exploration

### Understanding Entity States
- JSON viewer uses syntax highlighting for readability
- States are immutable records of entity properties at points in time
- Use state diffs between timeline entries to see what changed

### Finding Key Entities
1. Sort by transitions in Entity Explorer
2. Check Insights panel for "Most Active"
3. Use Relationship Explorer to find hubs with many connections

### Discovering Contradictions
1. Check Insights panel "Contradiction Hotspots"
2. Click entity to see detail panel
3. Review transition history for conflicting states

### Analyzing Evolution
- Timeline view shows creation patterns
- Activity Timeline shows engagement over time
- Recently Changed entities show current focus
- Belief Evolution shows conviction changes

### Relationship Patterns
- Orphans indicate isolated concepts
- High connectivity indicates critical infrastructure
- Relationship type diversity shows complexity

## Technical Details

**Single File Design:**
- All code (HTML, CSS, JavaScript) in one file
- No external dependencies (no npm, no CDN)
- Fetch-based JSON loading (requires HTTP server)
- Works in any modern browser

**Browser Requirements:**
- ES6 JavaScript support
- Canvas API for charts
- Fetch API for loading data
- CSS Grid/Flexbox for layout

**Performance Optimizations:**
- Lazy rendering of entity lists
- Efficient sorting and filtering
- Canvas charts render on demand
- DOM updates only for visible content

## Troubleshooting

**"Failed to load world model"**
- Ensure HTTP server is running
- Check path to world_model.json is correct
- Check browser console for CORS errors

**Slow performance with large entity lists**
- Use search/filter to narrow results
- Sort by relevant field to find target faster
- Click entity to open detail panel (doesn't reload)

**Charts not rendering**
- Check browser supports Canvas API
- Try refreshing page
- Check browser console for errors

## Future Enhancements

Potential additions:
- Export capabilities (CSV, JSON)
- Relationship type filtering
- State comparison view
- Bulk entity operations
- Custom analytics queries
- Relationship strength metrics
- Network centrality analysis
- Entity clustering visualization
