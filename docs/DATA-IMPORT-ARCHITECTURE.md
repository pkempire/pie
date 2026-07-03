# Data Import Architecture

PIE Sales Intelligence needs to ingest data from multiple CRM/sales platforms and build a unified world model for process mining and simulation.

## Supported Data Sources

### Tier 1 - Core CRM
| Source | Data Types | Integration Method |
|--------|------------|-------------------|
| **Salesforce** | Opportunities, Accounts, Contacts, Activities, Emails, Tasks | REST API (OAuth 2.0) |
| **HubSpot** | Deals, Companies, Contacts, Engagements, Emails, Meetings | REST API (OAuth 2.0) |
| **Pipedrive** | Deals, Persons, Organizations, Activities | REST API (API key) |

### Tier 2 - Engagement Platforms
| Source | Data Types | Integration Method |
|--------|------------|-------------------|
| **Gong** | Call transcripts, topics, action items, talk ratios | REST API (OAuth 2.0) |
| **Chorus** | Call recordings, moments, trackers | REST API |
| **Outreach** | Sequences, mailings, opens, clicks, replies | REST API |
| **Salesloft** | Cadences, emails, calls, meetings | REST API |

### Tier 3 - Revenue Intelligence
| Source | Data Types | Integration Method |
|--------|------------|-------------------|
| **Stripe** | Subscriptions, invoices, charges, refunds, churn | REST API + Webhooks |
| **Chargebee** | Subscriptions, invoices, customer lifecycle | REST API |
| **Kit.com (ConvertKit)** | Subscribers, sequences, broadcasts, conversions | REST API |

### Tier 4 - Communication
| Source | Data Types | Integration Method |
|--------|------------|-------------------|
| **Gmail** | Emails (with deal context) | Gmail API |
| **Slack** | Deal room messages, notifications | Slack API |
| **LinkedIn Sales Nav** | InMails, profile views, connections | (Manual export / scraping) |

---

## Unified Data Model

All sources map to a common schema:

```
┌─────────────────────────────────────────────────────────────────┐
│                        DEAL ENTITY                               │
├─────────────────────────────────────────────────────────────────┤
│ id: string (uuid)                                                │
│ external_ids: {salesforce: "...", hubspot: "...", ...}          │
│ name: string                                                     │
│ company: -> Company                                              │
│ value: number (USD)                                              │
│ stage: string                                                    │
│ probability: float                                               │
│ expected_close: date                                             │
│ owner: -> Person                                                 │
│ stakeholders: [-> Stakeholder]                                   │
│ created_at: timestamp                                            │
│ updated_at: timestamp                                            │
└─────────────────────────────────────────────────────────────────┘

┌─────────────────────────────────────────────────────────────────┐
│                      STAKEHOLDER ENTITY                          │
├─────────────────────────────────────────────────────────────────┤
│ id: string                                                       │
│ name: string                                                     │
│ title: string                                                    │
│ email: string                                                    │
│ role: enum {champion, blocker, economic_buyer, technical_buyer,  │
│             influencer, end_user, legal, procurement}            │
│ sentiment: float (-1 to 1)                                       │
│ engagement_score: float (0 to 1)                                 │
│ last_contact: timestamp                                          │
│ communication_style: string (inferred)                           │
│ personality_traits: [string] (inferred)                          │
└─────────────────────────────────────────────────────────────────┘

┌─────────────────────────────────────────────────────────────────┐
│                       ACTIVITY ENTITY                            │
├─────────────────────────────────────────────────────────────────┤
│ id: string                                                       │
│ deal: -> Deal                                                    │
│ type: enum {email, call, meeting, demo, proposal, contract,      │
│             objection, follow_up, internal}                      │
│ timestamp: datetime                                              │
│ participants: [-> Stakeholder]                                   │
│ outcome: enum {positive, neutral, negative, unknown}             │
│ summary: string (LLM-generated)                                  │
│ key_moments: [{time, text, sentiment}]                           │
│ next_steps: [string]                                             │
│ objections_raised: [{objection, resolved, resolution}]           │
│ raw_content: string (transcript/email body)                      │
└─────────────────────────────────────────────────────────────────┘

┌─────────────────────────────────────────────────────────────────┐
│                    STAGE_TRANSITION ENTITY                       │
├─────────────────────────────────────────────────────────────────┤
│ deal: -> Deal                                                    │
│ from_stage: string                                               │
│ to_stage: string                                                 │
│ timestamp: datetime                                              │
│ trigger_activity: -> Activity (what caused the transition)       │
│ duration_days: float (time in from_stage)                        │
└─────────────────────────────────────────────────────────────────┘
```

---

## Import Pipeline

```
┌──────────────┐     ┌──────────────┐     ┌──────────────┐     ┌──────────────┐
│   Source     │────▶│   Adapter    │────▶│   Mapper     │────▶│ World Model  │
│   API        │     │   (auth,     │     │   (schema    │     │   (graph     │
│              │     │    fetch)    │     │    unify)    │     │    store)    │
└──────────────┘     └──────────────┘     └──────────────┘     └──────────────┘
                                                │
                                                ▼
                                         ┌──────────────┐
                                         │   Enricher   │
                                         │   (LLM +     │
                                         │    Web)      │
                                         └──────────────┘
```

### 1. Adapter Layer
Each source has an adapter that handles:
- Authentication (OAuth flow or API key)
- Pagination
- Rate limiting
- Incremental sync (track last sync timestamp)

```python
class SalesforceAdapter:
    def __init__(self, credentials: dict):
        self.client = Salesforce(**credentials)
    
    def fetch_opportunities(self, since: datetime = None) -> Iterator[dict]:
        query = "SELECT Id, Name, Amount, StageName, ... FROM Opportunity"
        if since:
            query += f" WHERE LastModifiedDate > {since.isoformat()}"
        for record in self.client.query_all(query)['records']:
            yield record
    
    def fetch_activities(self, opportunity_id: str) -> Iterator[dict]:
        # Tasks, Events, EmailMessages related to opportunity
        ...
```

### 2. Mapper Layer
Transforms source-specific schemas to unified model:

```python
class SalesforceMapper:
    STAGE_MAP = {
        "Prospecting": "lead",
        "Qualification": "qualification", 
        "Needs Analysis": "discovery",
        "Value Proposition": "demo",
        "Proposal/Price Quote": "proposal",
        "Negotiation/Review": "negotiation",
        "Closed Won": "closed_won",
        "Closed Lost": "closed_lost",
    }
    
    def map_opportunity(self, sf_opp: dict) -> Deal:
        return Deal(
            external_ids={"salesforce": sf_opp["Id"]},
            name=sf_opp["Name"],
            value=sf_opp.get("Amount", 0),
            stage=self.STAGE_MAP.get(sf_opp["StageName"], sf_opp["StageName"]),
            ...
        )
```

### 3. Enricher Layer
LLM-powered extraction from unstructured content:

```python
class ActivityEnricher:
    def enrich_call_transcript(self, transcript: str, deal: Deal) -> Activity:
        # Use LLM to extract:
        # - Summary
        # - Key moments with timestamps
        # - Objections raised
        # - Next steps
        # - Sentiment shifts
        # - Stakeholder personality insights
        
        extraction = self.llm.extract(
            transcript,
            prompt=CALL_EXTRACTION_PROMPT,
            deal_context=deal.to_context()
        )
        return Activity(**extraction)
```

---

## Simulation Engine

The core value prop: don't just track deals, **simulate outcomes**.

### Process Mining
Build Markov chain from historical stage transitions:

```python
class ProcessMiner:
    def build_transition_matrix(self, deals: list[Deal]) -> dict:
        """
        Build probability matrix: P(to_stage | from_stage, context)
        
        Context factors:
        - Has champion? (+15% to advance)
        - Blocker identified? (-20% if unresolved)
        - Days in stage (decay function)
        - Industry match to ICP
        - Deal size vs historical median
        - Engagement recency
        """
        transitions = defaultdict(lambda: defaultdict(int))
        for deal in deals:
            for t in deal.stage_transitions:
                key = (t.from_stage, self._context_bucket(deal, t))
                transitions[key][t.to_stage] += 1
        
        # Normalize to probabilities
        return self._normalize(transitions)
```

### Monte Carlo Simulation
Run thousands of simulations with different scenarios:

```python
class DealSimulator:
    def simulate(self, deal: Deal, scenario: Scenario, n_runs: int = 10000) -> SimulationResult:
        """
        Scenario examples:
        - "add_champion": Add a champion stakeholder
        - "remove_blocker": Resolve blocking objection
        - "accelerate_demo": Move demo up by 1 week
        - "increase_touchpoints": 2x meeting frequency
        """
        outcomes = []
        for _ in range(n_runs):
            modified_deal = scenario.apply(deal)
            outcome = self._run_single(modified_deal)
            outcomes.append(outcome)
        
        return SimulationResult(
            p_win=sum(1 for o in outcomes if o.won) / n_runs,
            expected_days=np.mean([o.days for o in outcomes]),
            expected_value=np.mean([o.value for o in outcomes]),
            confidence_interval=self._compute_ci(outcomes)
        )
```

---

## API Design

```
POST /api/import/connect
  body: {source: "salesforce", credentials: {...}}
  -> {connection_id, status, sync_started}

GET /api/import/status/{connection_id}
  -> {records_synced, last_sync, errors}

POST /api/import/sync/{connection_id}
  body: {full: false}  # incremental by default
  -> {job_id}

POST /api/simulate
  body: {
    deal_id: "...",
    scenario: {
      type: "add_champion",
      params: {name: "...", title: "VP Engineering"}
    },
    n_simulations: 10000
  }
  -> {
    baseline: {p_win: 0.35, days: 45},
    modified: {p_win: 0.52, days: 38},
    delta: {p_win: +0.17, days: -7},
    confidence: 0.95
  }

GET /api/deals/{id}/forecast
  -> {
    scenarios: [
      {name: "current_trajectory", p_win: 0.35, days: 45},
      {name: "best_case", p_win: 0.65, days: 28},
      {name: "worst_case", p_win: 0.12, days: 90},
    ],
    recommendations: [
      {action: "Identify champion", impact: "+17% win rate"},
      {action: "Address security objection", impact: "-12 days cycle time"},
    ]
  }
```

---

## Implementation Priority

### Phase 1 - Core Import (Week 1-2)
- [ ] Salesforce adapter + mapper
- [ ] HubSpot adapter + mapper
- [ ] Manual CSV/JSON upload
- [ ] Unified deal/activity storage

### Phase 2 - Enrichment (Week 3-4)
- [ ] Call transcript extraction (Gong format)
- [ ] Email thread summarization
- [ ] Stakeholder inference from activities
- [ ] Objection tracking

### Phase 3 - Simulation (Week 5-6)
- [ ] Process mining from historical data
- [ ] Scenario modeling
- [ ] Monte Carlo engine
- [ ] Recommendation generation

### Phase 4 - Scale (Week 7-8)
- [ ] Incremental sync
- [ ] Webhook listeners
- [ ] Background job queue
- [ ] Multi-tenant isolation

---

## Open Questions

1. **Stage normalization** - Different orgs use different stage names. Do we:
   - Force a standard taxonomy?
   - Let users map their stages?
   - Use LLM to infer stage meanings?

2. **Historical data depth** - How much history do we need for reliable process mining?
   - Minimum viable: 50 closed deals
   - Recommended: 200+ deals with full activity history

3. **Real-time vs batch** - Should simulations update live as activities come in?
   - Webhooks for real-time stage changes
   - Batch enrichment for transcripts (expensive)

4. **Privacy** - Call transcripts contain sensitive info
   - On-prem option?
   - PII redaction before storage?
   - SOC2 compliance path?
