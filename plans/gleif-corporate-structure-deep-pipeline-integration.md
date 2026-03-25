# GLEIF Corporate Structure -- Deep Pipeline Integration

How GLEIF parent/subsidiary data can enhance every stage of the Operator 1 pipeline.

---

## What We Now Have

[`operator1/clients/gleif.py`](operator1/clients/gleif.py) provides `CorporateStructureResult` with:
- `ultimate_parent`: CorporateEntity with LEI, name, country
- `direct_parent`: CorporateEntity with LEI, name, country
- `subsidiaries`: list of CorporateEntity objects

[`main.py`](main.py:1358) Step 5e.1 injects `parent_companies` and `subsidiaries` into the `relationships` dict after LLM entity discovery.

---

## Integration Map -- 12 Touch Points

### 1. Graph Risk -- Asymmetric Contagion Edges

**Where**: [`operator1/models/graph_risk.py`](operator1/models/graph_risk.py:128) `build_entity_graph()`

**Current**: All edges have the same contagion probability (0.3). A supplier link and a parent-subsidiary link carry equal weight.

**Enhancement**: When relationship group is `parent_companies` or `subsidiaries`, assign asymmetric edge weights:

```
parent -> target:    contagion_prob = 0.85  -- parent distress directly impacts subsidiary
target -> parent:    contagion_prob = 0.20  -- subsidiary distress weakly impacts parent
target -> subsidiary: contagion_prob = 0.70 -- target controls subsidiary fate
subsidiary -> target: contagion_prob = 0.15 -- subsidiary distress mildly impacts parent
```

The graph already iterates over all relationship groups, so parent_companies and subsidiaries already appear as nodes. The change is adding group-aware edge weights to the SIR contagion simulation.

**Impact**: Contagion simulation becomes realistic. A parent bankruptcy should infect subsidiaries with near-certainty, not 30%.

---

### 2. Survival Mode -- Parent Bailout Factor

**Where**: [`operator1/analysis/survival_mode.py`](operator1/analysis/survival_mode.py:54) `compute_company_survival_flag()`

**Current**: Survival flag is based only on the target's own ratios (current_ratio, debt_to_equity, fcf_yield, drawdown).

**Enhancement**: New concept -- **parent strength modifier**. If the target is a subsidiary of a financially strong parent:
- Fetch parent's key ratios from PIT API (if available) or use GLEIF metadata
- If parent has strong balance sheet (current_ratio > 2.0, debt_to_equity < 1.0), reduce subsidiary's survival probability by 20-30%
- Rationale: a strong parent can inject capital, guarantee debt, or restructure the subsidiary

Conversely, if the parent is in distress:
- If parent triggers survival mode on its own metrics, amplify subsidiary survival probability by 15-25%
- A distressed parent may liquidate subsidiaries, cut funding, or drag them into bankruptcy

This connects to the continuous `compute_survival_probability()` function (line 141) -- the sigmoid probability should be modulated by parent health.

**Data requirement**: Need to fetch parent's financials via PIT client. Could be a single API call per pipeline run (fetch parent's latest balance sheet).

---

### 3. Fuzzy Protection -- Protection Inheritance

**Where**: [`operator1/analysis/fuzzy_protection.py`](operator1/analysis/fuzzy_protection.py:222) `compute_fuzzy_protection()`

**Current**: Protection degree based on: (1) sector strategicness, (2) market_cap/GDP ratio, (3) policy responsiveness.

**Enhancement**: If the target is a subsidiary of a company with high protection degree:
- Compute parent's fuzzy protection score (sector + market_cap/GDP)
- Use `max(target_degree, parent_degree * 0.7)` as the floor
- A subsidiary of a defense contractor inherits strategic importance
- A subsidiary of a bank in a bailout zone inherits financial protection

The parent's protection can be computed cheaply from GLEIF metadata alone (parent name -> infer sector from name heuristics, parent country -> lookup GDP).

---

### 4. Conflict Risk -- Cross-Border Subsidiary Exposure

**Where**: [`operator1/features/conflict_risk.py`](operator1/features/conflict_risk.py:1) `assess_conflict_risk()`

**Current already works**: The linked entity conflict propagation at Step 5g.5 checks all relationship groups. Since GLEIF subsidiaries are injected as a relationship group, they already flow through `assess_linked_entity_conflict()`.

**But we can do better**: A subsidiary in a conflict zone is different from a supplier in a conflict zone. The subsidiary is owned by the target -- its assets are the target's assets. The risk multiplier should be higher:
- Supplier in conflict zone: supply_chain_risk_score weight = 0.3
- Subsidiary in conflict zone: supply_chain_risk_score weight = 0.8 (it IS your supply chain)

Similarly, a parent in a sanctioned country means the target itself may face secondary sanctions.

---

### 5. Ownership Contagion -- Controlling vs Portfolio

**Where**: [`operator1/models/ownership_contagion.py`](operator1/models/ownership_contagion.py:1)

**Current**: MHHI, crowding, and liquidation analysis assume all holders are portfolio investors (institutional holders).

**Enhancement**: Separate controlling ownership (parent owns 100%) from portfolio holdings. A parent is NOT an institutional investor -- it's a controlling owner. This changes the analysis:
- MHHI: parent ownership should be excluded (MHHI measures anticompetitive common ownership among portfolio investors, not controlling ownership)
- Crowding: controlling ownership is not a crowded trade -- it's strategic, not speculative
- Liquidation: a parent selling a subsidiary is a corporate restructuring event, not a portfolio rebalancing -- different dynamics entirely

New signal: **cross-holding risk** -- if the parent also owns competitors (common in Asian conglomerates like Samsung Group, Tata Group), that creates a different type of competitive dynamic than institutional common ownership.

---

### 6. Economic Planes -- Conglomerate Cross-Plane Mapping

**Where**: [`operator1/analysis/economic_planes.py`](operator1/analysis/economic_planes.py:39) `classify_economic_plane()`

**Current**: Target is mapped to one plane based on its sector.

**Enhancement**: If the target has a parent or subsidiaries in different sectors, map the entire corporate tree across planes:
- TotalEnergies SE: Energy plane (primary)
  - TotalEnergies Renewables SAS: Energy plane (same)
  - TotalEnergies Marketing France: Consumption plane (retail)
  - TotalEnergies Petrochemicals: Manufacturing plane (chemical)

This creates explicit inter-plane linkages that the 5-plane Sudoku model should capture. When the Energy plane enters crisis, the Consumption subsidiary is partially shielded but the Manufacturing subsidiary is directly exposed.

---

### 7. Entity Discovery -- LLM Validation

**Where**: [`operator1/steps/entity_discovery.py`](operator1/steps/entity_discovery.py:28)

**Current**: LLM proposes linked entities; PIT client searches and resolves them with fuzzy matching.

**Enhancement**: Use GLEIF subsidiaries to validate LLM proposals. If the LLM says "Company X is a supplier" but GLEIF shows Company X is actually a subsidiary, the relationship should be reclassified. This catches LLM hallucinations about entity relationships.

Also: GLEIF subsidiaries that the LLM missed can be added as additional linked entities. The LLM may not know about obscure holding companies or recently acquired subsidiaries.

---

### 8. Linked Aggregates -- Subsidiary Financial Aggregation

**Where**: [`operator1/features/linked_aggregates.py`](operator1/features/linked_aggregates.py)

**Current**: Computes avg/median of financial metrics across entity groups (competitors_avg_return_1d, suppliers_median_volatility_21d, etc.).

**Enhancement**: New aggregate columns:
- `subsidiaries_total_revenue` -- sum of all subsidiary revenues (consolidated revenue proxy)
- `subsidiaries_avg_survival_prob` -- average survival probability across subsidiaries
- `parent_survival_prob` -- parent's survival probability (single value, not aggregate)
- `group_debt_total` -- total debt across entire corporate group

These give temporal models signals about group-level health that individual company data misses.

---

### 9. Game Theory -- Vertical Integration Dynamics

**Where**: [`operator1/models/game_theory.py`](operator1/models/game_theory.py)

**Current**: Cournot/Bertrand/Stackelberg models between target and competitors.

**Enhancement**: If the target owns subsidiaries that are suppliers or customers to its competitors, the competitive dynamic changes:
- Vertically integrated company has cost advantages (captured supply chain margins)
- Can use subsidiary pricing to squeeze competitors (predatory pricing via subsidiary)
- Stackelberg leadership is stronger with vertical integration

New metric: `vertical_integration_score` -- fraction of the supply chain that the corporate group controls internally.

---

### 10. Financial Health -- Group Liquidity

**Where**: [`operator1/models/financial_health.py`](operator1/models/financial_health.py)

**Current**: Financial health scores based on individual company ratios.

**Enhancement**: New sub-score: `group_liquidity_buffer` -- if the parent has excess cash, the subsidiary has an implicit liquidity backstop. This moderates the Tier 1 liquidity score:
- Target cash_ratio < 1.0 but parent cash_ratio > 3.0: effective liquidity is higher than it appears
- Altman Z-Score adjustment: subsidiary of strong parent has lower bankruptcy risk than standalone company with same ratios

---

### 11. Report Generator -- Corporate Structure Section

**Where**: [`operator1/report/report_generator.py`](operator1/report/report_generator.py)

**Enhancement**: New report section (Section 19.6 or equivalent):
- Corporate group structure visualization (parent -> target -> subsidiaries tree)
- Cross-border subsidiary exposure map
- Group-level risk aggregation summary
- Protection inheritance chain

---

### 12. Profile Builder -- GLEIF Metadata

**Where**: [`operator1/report/profile_builder.py`](operator1/report/profile_builder.py)

**Enhancement**: New profile section `corporate_structure`:
```json
{
  "corporate_structure": {
    "available": true,
    "target_lei": "529900S21EQ1BO4ESM68",
    "ultimate_parent": {"name": "TotalEnergies SE", "lei": "...", "country": "FR"},
    "direct_parent": null,
    "n_subsidiaries": 15,
    "subsidiaries_countries": ["FR", "PT", "..."],
    "cross_border_exposure": 0.73,
    "group_plane_distribution": {"energy": 0.6, "manufacturing": 0.25, "consumption": 0.15}
  }
}
```

---

## Implementation Priority

| # | Enhancement | Effort | Impact | Dependencies |
|---|-------------|--------|--------|--------------|
| 1 | Graph risk asymmetric edges | Low | High | Already have data in relationships |
| 2 | Parent bailout factor in survival | Medium | High | Need parent financials fetch |
| 3 | Protection inheritance | Low | Medium | GLEIF metadata only |
| 4 | Conflict subsidiary exposure weights | Low | Medium | Already flows through |
| 5 | Ownership controlling vs portfolio | Medium | Medium | Conceptual change in ownership_contagion |
| 6 | Economic planes cross-mapping | Low | Low | Name-based sector heuristics |
| 7 | LLM validation | Low | Medium | Compare GLEIF vs LLM results |
| 8 | Subsidiary aggregates | Medium | Medium | Need subsidiary financials |
| 9 | Vertical integration dynamics | Medium | Low | Game theory model change |
| 10 | Group liquidity buffer | Medium | Medium | Need parent financials |
| 11 | Report section | Low | Low | Template + profile section |
| 12 | Profile builder section | Low | Low | JSON structure addition |

**Recommended order**: 1, 3, 4, 12, 11, 7, 6, 2, 5, 8, 10, 9

Items 1, 3, 4, 12, 11 can be done quickly with existing data. Items 2, 8, 10 require fetching parent/subsidiary financials (additional PIT API calls). Item 5 is a conceptual refactor of the ownership model. Item 9 is a game theory model enhancement.
