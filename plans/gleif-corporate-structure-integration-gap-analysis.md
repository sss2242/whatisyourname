# GLEIF Corporate Structure Integration -- Gap Analysis

## The Insight

When analyzing a large company like TotalEnergies, Siemens, or Nestle, the pipeline currently treats it as an isolated entity. But GLEIF reveals that these companies have **corporate parents and subsidiaries** -- and this structural information is missing from every downstream model.

The question: **Should a company's survival analysis, contagion risk, and financial health consider whether it has a parent that could bail it out (or drag it down), or subsidiaries whose distress could cascade upward?**

The answer is yes. This is a genuine gap.

---

## Current Pipeline Flow -- What Exists

```
Entity Discovery (LLM) --> relationships dict
  |                          {competitors, suppliers, customers,
  |                           financial_institutions, logistics, regulators}
  |
  +--> Graph Risk (C11) -- builds network, computes centrality, PageRank, contagion
  +--> Game Theory (C12) -- competitive dynamics between target and competitors
  +--> Linked Aggregates (C7) -- avg/median of peer financial metrics
  +--> Ownership Contagion -- MHHI, crowding, liquidation from institutional holders
  +--> Conflict Propagation -- supply chain/revenue/competitive risk from conflicts
```

**Missing from entity discovery**: `parent_companies` and `subsidiaries` relationship groups. The LLM is asked for competitors, suppliers, customers, financial_institutions, logistics, regulators -- but never for corporate parent/subsidiary structure.

---

## The Gap -- Where Corporate Structure Matters

### 1. Survival Mode (survival_mode.py)

**Current**: Company survival flag is triggered by the target's own ratios (current_ratio, debt_to_equity, fcf_yield, drawdown).

**Gap**: If the target is a subsidiary of a financially strong parent, its survival risk is lower -- the parent can inject capital, guarantee debt, or restructure. Conversely, if the parent is in distress, the subsidiary faces contagion risk even if its own ratios look healthy.

**GLEIF data needed**: Ultimate parent LEI --> fetch parent's financials --> assess parent health --> adjust subsidiary survival probability.

### 2. Graph Risk (graph_risk.py)

**Current**: Network built from LLM-discovered relationships (competitors, suppliers, customers). Contagion simulation runs on this graph.

**Gap**: Parent-subsidiary edges are the STRONGEST contagion channels -- a parent's bankruptcy directly impacts all subsidiaries. These edges should have much higher contagion probability (0.8-0.9) than supplier/customer edges (0.3).

**GLEIF data needed**: Add parent/subsidiary nodes to the entity graph with high-weight edges.

### 3. Ownership Contagion (ownership_contagion.py)

**Current**: Models institutional overlap (MHHI) between target and competitors. Computes crowding and liquidation risk.

**Gap**: A corporate parent IS the ultimate owner. If TotalEnergies SE owns 100% of TotalEnergies Renewables SAS, that's not institutional overlap -- it's controlling ownership. This changes the entire contagion dynamic: the parent can reallocate capital, merge entities, or wind down subsidiaries at will.

**GLEIF data needed**: Separate controlling ownership from institutional portfolio holdings.

### 4. Economic Planes (economic_planes.py)

**Current**: Maps target to one of 5 economic planes based on sector.

**Gap**: A conglomerate parent may span multiple planes (TotalEnergies: energy + renewables + chemical). Subsidiaries in different planes create inter-plane linkages that the 5-plane Sudoku model should capture.

**GLEIF data needed**: Map parent and each subsidiary to their respective planes; create inter-plane edges.

### 5. Fuzzy Protection (fuzzy_protection.py)

**Current**: Government protection score based on sector strategicness + market cap/GDP + policy responsiveness.

**Gap**: A subsidiary of a strategically important parent inherits protection. If Siemens AG is government-protected, Siemens Energy inherits some of that protection. The fuzzy membership should propagate through the corporate tree.

**GLEIF data needed**: Parent's protection degree as a floor for subsidiary protection.

---

## Proposed Integration Points

### Step 1: Enrich Entity Discovery with GLEIF (Low effort, High impact)

In [`main.py`](main.py:1333) Step 5e, after LLM entity discovery, call GLEIF to add parent/subsidiary relationships:

```
relationships["parent_companies"] = [gleif_ultimate_parent, gleif_direct_parent]
relationships["subsidiaries"] = [gleif_direct_children[:10]]
```

This feeds directly into graph_risk, linked_aggregates, and conflict_propagation with zero model changes -- they already iterate over all relationship groups.

### Step 2: Add Parent Health Score to Survival Mode (Medium effort, High impact)

New survival trigger: if the parent company is in distress (fetch parent's key ratios from PIT API), amplify the subsidiary's survival probability.

### Step 3: High-Weight Contagion Edges for Parent-Sub (Low effort, Medium impact)

In graph_risk, when building edges, assign:
- parent->subsidiary: contagion_prob = 0.85 (controlling ownership)
- subsidiary->parent: contagion_prob = 0.40 (subsidiary distress weakens parent)

### Step 4: Protection Inheritance in Fuzzy Logic (Low effort, Medium impact)

In fuzzy_protection, if the target is a subsidiary, compute parent's protection degree and use `max(target_degree, parent_degree * 0.7)` as the floor.

---

## What GLEIF Provides vs What Each Model Needs

| Model | Needs from GLEIF | Currently Available | Gap |
|-------|-----------------|--------------------|----|
| Entity Discovery | parent/subsidiary LEIs + names | Yes (get_holders returns them) | Need to inject into relationships dict |
| Graph Risk | parent/sub nodes with high-weight edges | No | New edge type with 0.85 contagion prob |
| Survival Mode | parent financial health assessment | No | New survival trigger: parent_distress_flag |
| Ownership Contagion | controlling vs portfolio ownership distinction | No | New category: controlling_owner |
| Economic Planes | parent/sub cross-plane mapping | No | Inter-plane edges from corporate tree |
| Fuzzy Protection | parent protection score inheritance | No | Protection floor from parent |
| Conflict Risk | parent/sub in conflict zones | Partially (linked entity conflict) | Already works if added to relationships |

---

## Recommendation

**Step 1 is the quick win**: inject GLEIF parent/subsidiary data into the `relationships` dict after LLM entity discovery. This immediately flows through graph_risk, linked_aggregates, conflict_propagation, and peer_ranking with zero model changes. The graph_risk contagion simulation already handles arbitrary relationship groups -- adding "parent_companies" and "subsidiaries" groups just adds nodes with different edge semantics.

Steps 2-4 are model enhancements that would be separate PRs.
