# Product Analysis Methods: Expert Techniques & Integration Plan

*Research compiled: 2026-04-06*

How professional analysts, hedge funds, and quantitative researchers analyze a company's products, and how each method maps into Operator 1's existing pipeline architecture.

---

## Part 1: Popular Methods (Widely Used by Analysts)

### M1. Revenue Segment Decomposition

**What it is**: Breaking total revenue into product lines and tracking each line's growth rate, margin, and contribution to total.

**Who uses it**: Every sell-side analyst, all equity research reports. The first thing anyone does when analyzing Apple is separate iPhone from Services from Mac.

**How experts do it**:
- Extract from 10-K Item 1/Item 7 (US), annual report segment notes (IFRS)
- Build a segment revenue waterfall: `Total Revenue = sum(Product_i * Unit_Price_i * Volume_i)`
- Track quarter-over-quarter growth PER segment, not just aggregate
- Compute revenue Herfindahl-Hirschman Index: `HHI = sum((Revenue_i / Total_Revenue)^2)` -- higher = more concentrated = more risk

**Where it fits in Operator 1**:
- **Data source**: SimFin API (`/api/v3/companies/id/{id}/statements/derived?statement=segments`) or XBRL segment dimensions from edgartools
- **Feature module**: New `operator1/features/product_segments.py`
  - Computes: `segment_hhi`, `segment_growth_rates`, `segment_margin_spreads`, `dominant_segment_pct`
  - Runs at Step 4d (after cache build, before derived variables)
- **HF integration**: Directly feeds `fcf_quality.py` (FCF by segment), `growth_quality.py` (organic vs M&A per segment), `momentum_composite.py` (per-segment revenue acceleration)
- **Cache columns**: `segment_hhi`, `dominant_segment_growth`, `segment_count`, `segment_diversification_delta`

---

### M2. Product Lifecycle Stage Classification (Bass Model)

**What it is**: Classifying where each product is on the S-curve: Introduction, Growth, Maturity, Decline.

**Who uses it**: Consumer goods analysts, tech analysts forecasting iPhone upgrade cycles, pharma analysts modeling drug adoption curves.

**How experts do it**:
- Bass diffusion model: `f(t) = (p + q*F(t)) * (1 - F(t))` where p = innovation coefficient, q = imitation coefficient, F(t) = cumulative adoption
- Fit Bass curve to product segment revenue history (needs 8+ quarterly data points)
- Classify stage: `F(t) < 0.1` = Introduction, `0.1-0.5` = Growth, `0.5-0.9` = Maturity, `>0.9` = Decline
- Key signal: a product transitioning from Growth to Maturity means revenue growth will decelerate even if the product is "doing well"

**Where it fits in Operator 1**:
- **Feature module**: Enhancement to `operator1/features/product_catalysts.py`
  - Add `classify_product_lifecycle()`: fits Bass model to each segment's revenue trajectory
  - Output: per-segment lifecycle stage, growth runway estimate (quarters until peak), replacement risk flag
- **HF integration**: `growth_quality.py` -- organic growth score adjusted by lifecycle stage. A company with 80% revenue in Maturity gets a lower growth quality score even if current QoQ is positive.
- **Cache columns**: `product_lifecycle_stage` (dominant product), `growth_runway_quarters`, `maturity_concentration` (% revenue in Maturity/Decline)

---

### M3. Gross Margin Bridge Analysis

**What it is**: Decomposing gross margin changes into product mix shifts, pricing changes, and cost changes.

**Who uses it**: Industrials, consumer staples, and semiconductor analysts. Critical for understanding whether margin expansion comes from pricing power (sustainable) or cost reduction (temporary).

**How experts do it**:
- Build margin bridge: `delta_GM = (Mix Effect) + (Price Effect) + (Cost Effect)`
- Mix effect: `sum((weight_i_new - weight_i_old) * margin_i_old)` -- did high-margin products gain share?
- Price effect: `sum(weight_i_new * (price_i_new - price_i_old) / revenue_i_old)` -- did pricing improve?
- Cost effect: `sum(weight_i_new * (cost_i_old - cost_i_new) / revenue_i_old)` -- did costs fall?
- If margin expansion comes entirely from mix (selling more high-margin services), it's sustainable. If from cost cuts, it's temporary.

**Where it fits in Operator 1**:
- **Feature module**: New function in `operator1/features/derived_variables.py` or `product_segments.py`
  - Computes margin bridge when segment data is available
  - When segment data unavailable: proxy via `gross_margin` delta decomposition using revenue growth vs COGS growth (already in cache)
- **HF integration**: `accruals_forensics.py` -- mix-driven margin expansion is legitimate; cost-cut-driven expansion is a manipulation risk flag
- **Cache columns**: `margin_mix_effect`, `margin_price_effect`, `margin_cost_effect`, `margin_sustainability_score`

---

### M4. TAM/SAM/SOM Market Share Analysis

**What it is**: Total Addressable Market -> Serviceable Addressable Market -> Serviceable Obtainable Market. What % of the market does this company capture?

**Who uses it**: Growth equity, venture capital, technology analysts. Core of every IPO pitch deck.

**How experts do it**:
- TAM from industry reports (Gartner, IDC, Statista) or compute from macro data: `TAM = GDP_sector * penetration_rate`
- SAM = TAM * geographic_reach * capability_match
- SOM = actual revenue / SAM
- Track SOM trend: rising = gaining share (bullish), falling = losing share (bearish)
- Key signal: SOM > 30% means further growth requires taking share from entrenched competitors (harder)

**Where it fits in Operator 1**:
- **Data source**: Macro data already available (GDP by sector from FRED/World Bank). Revenue from cache. Industry classification from `economic_planes.py`.
- **Feature module**: Enhancement to `operator1/features/market_buying_power.py`
  - Add `estimate_market_share()`: computes SOM from company revenue / sector GDP proxy
  - Trend: `som_trend = SOM.pct_change(4)` (QoQ share change)
- **HF integration**: `valuation_quality.py` -- high SOM + low growth = value trap risk. Low SOM + high growth = growth opportunity.
- **Cache columns**: `estimated_market_share`, `som_trend_4q`, `tam_penetration`

---

### M5. Customer Concentration Risk (Revenue Dependency)

**What it is**: How dependent is the company on its top customers? SEC requires disclosure of any customer >10% of revenue.

**Who uses it**: Credit analysts, small-cap analysts, supply chain analysts. A company with 50% revenue from one customer is one phone call away from disaster.

**How experts do it**:
- Extract from 10-K: "Our largest customer, [Retailer], accounted for approximately 15% of our revenues"
- Customer HHI: `sum((customer_share_i)^2)` -- same concept as product HHI but for customers
- Cross-reference with customer's financial health (if Walmart is your 30% customer and Walmart is healthy, lower risk than if your 30% customer is a struggling retailer)
- Key signal: customer concentration + product concentration = double risk

**Where it fits in Operator 1**:
- **Data source**: SEC filing text extraction (edgartools or LLM filing extractor). The `us_edgar.py` client already fetches full filing text.
- **Feature module**: New `operator1/features/customer_concentration.py`
  - LLM-assisted extraction of customer names and revenue percentages from 10-K Item 1
  - Fallback: fuzzy_pdf_parser with customer-specific concept dictionary
  - Computes: `customer_hhi`, `top_customer_pct`, `customer_count`, `customer_health_weighted_risk`
- **HF integration**: `leverage_stress.py` -- stress scenarios should model top customer loss. `obs_risk.py` -- customer concentration is a hidden off-balance-sheet risk.
- **Cache columns**: `customer_hhi`, `top_customer_pct`, `customer_concentration_risk`

---

### M6. Product Pricing Power Index

**What it is**: Can the company raise prices without losing volume? Measured by comparing revenue growth to volume growth.

**Who uses it**: Consumer staples analysts (Procter & Gamble, Coca-Cola), luxury goods analysts, SaaS analysts (net revenue retention).

**How experts do it**:
- Price/volume decomposition: `Revenue Growth = Price Growth + Volume Growth + Mix`
- If Price Growth > Inflation: real pricing power (moat indicator)
- If Volume Growth > Price Growth: volume-driven (less sustainable, margin pressure)
- For SaaS: Net Revenue Retention (NRR) > 120% = strong pricing power. < 100% = churn problem.
- Proxy when not disclosed: `Real Revenue Growth = Nominal Revenue Growth - Industry PPI`

**Where it fits in Operator 1**:
- **Data source**: Revenue from cache. PPI (Producer Price Index) from FRED macro data. CPI already in macro_provider.
- **Feature module**: Enhancement to `operator1/features/derived_variables.py`
  - `pricing_power_index = (revenue_growth_yoy - industry_ppi) / revenue_growth_yoy`
  - Positive = real pricing power, Negative = volume-dependent growth
- **HF integration**: `momentum_composite.py` -- pricing power is the highest-quality form of revenue momentum. `valuation_quality.py` -- companies with pricing power deserve higher PE multiples.
- **Cache columns**: `pricing_power_index`, `real_revenue_growth`, `price_vs_volume_ratio`

---

## Part 2: Unpopular/Unconventional Methods (Used by Quants & Niche Analysts)

### M7. Patent Citation Network Analysis (Forward Citations)

**What it is**: Not just patent counts, but the CITATION NETWORK. How many times are a company's patents cited by OTHER companies' patents? This measures innovation influence.

**Who uses it**: Technology-focused hedge funds (ARK Invest, Tiger Global). Academic researchers (Kogan et al. 2017 "Technological Innovation, Resource Allocation, and Growth").

**How experts do it**:
- Forward citation count: how many future patents cite patent X? High forward citations = foundational innovation
- Citation velocity: how fast are citations accumulating? Accelerating = hot research area
- Self-citation ratio: what % of citations are from the company's own patents? High self-citation = incremental innovation. Low self-citation + high total citations = breakthrough innovation cited by competitors.
- Technology class diversification: Shannon entropy of CPC codes. Increasing entropy = company exploring new product areas.

**Where it fits in Operator 1**:
- **Data source**: PatentsView API (free, unlimited). Already in the research file.
- **Feature module**: New `operator1/features/patent_pipeline.py`
  - `compute_patent_innovation_score()`: forward citation velocity, self-citation ratio, technology diversification
  - Runs at Step 5 (after cache build, with company name/ticker for patent lookup)
- **HF integration**: `growth_quality.py` -- patent innovation score as a leading indicator of organic growth quality. `product_catalysts.py` -- high citation velocity = upcoming product catalyst.
- **Cache columns**: `patent_citation_velocity`, `patent_self_citation_ratio`, `patent_tech_diversification`, `patent_innovation_score`

---

### M8. Employee Review Sentiment by Product Team (Glassdoor/Blind)

**What it is**: Track employee sentiment by product team. A product team with declining Glassdoor ratings is likely building a product that will underperform.

**Who uses it**: Some hedge funds (Point72, Two Sigma) reportedly use alternative data including employee sentiment. Academic: Green et al. (2019) "The Value of Employee Satisfaction in Disastrous Times."

**How experts do it**:
- Scrape Glassdoor reviews with department/team tags
- Compute sentiment delta: team sentiment now vs 6 months ago
- Cross-reference with product launch timeline: declining team sentiment 6-12 months before launch = bad product incoming
- Key signal: engineering team sentiment declining while marketing team sentiment rising = company knows the product is weak and is compensating with marketing spend

**Where it fits in Operator 1**:
- **Data source**: Glassdoor API (limited), Blind (anonymous employee reviews), or LLM analysis of scraped reviews
- **Integration**: Enhancement to `operator1/features/news_sentiment.py`
  - New `compute_employee_sentiment()` function alongside news sentiment
  - Separate cache columns from news sentiment
- **HF integration**: `growth_quality.py` -- employee sentiment as a 6-12 month leading indicator of product quality
- **Cache columns**: `employee_sentiment_score`, `employee_sentiment_delta_6m`, `product_team_morale`
- **Note**: Data availability is limited. Best suited as an optional enhancement.

---

### M9. Supply Chain Bill of Materials (BOM) Cost Tracking

**What it is**: Track the cost of a product's raw materials and components. When input costs rise, margin compresses unless the company can pass through pricing.

**Who uses it**: Industrials analysts, auto analysts, semiconductor analysts. Apple analysts track NAND flash prices, display costs, and chip costs to forecast iPhone margins.

**How experts do it**:
- Build BOM cost model: `iPhone BOM = Display($70) + Processor($40) + Memory($35) + Camera($25) + Battery($10) + Assembly($8) + Other($62) = $250`
- Track component price indices: DRAM Spot Price (TrendForce), NAND Price (DRAMeXchange), Display Panel Price (DSCC)
- Compute BOM cost trend vs ASP trend: if BOM rising faster than ASP, margins will compress
- Key signal: BOM cost spike 2 quarters before earnings = margin miss incoming

**Where it fits in Operator 1**:
- **Data source**: FRED commodity prices (already available), sector-specific indices. No free API for component-level BOM data (proprietary: IHS Markit, TechInsights).
- **Feature module**: Enhancement to `operator1/features/market_buying_power.py`
  - `compute_input_cost_pressure()`: tracks relevant commodity/component indices for the company's sector
  - Sector mapping: Technology -> semiconductor indices, Energy -> crude oil, Materials -> metals
- **HF integration**: `fcf_quality.py` -- input cost pressure as a leading indicator of FCF degradation. `leverage_stress.py` -- stress scenarios should include input cost spikes.
- **Cache columns**: `input_cost_pressure`, `bom_margin_squeeze_flag`, `commodity_exposure_score`

---

### M10. Satellite/Geospatial Product Activity Proxies

**What it is**: Use satellite imagery or geolocation data to estimate real-time product activity (factory output, retail foot traffic, shipping container counts).

**Who uses it**: Quant hedge funds (Citadel, Renaissance). Providers: Orbital Insight, SpaceKnow, Descartes Labs, Placer.ai.

**How experts do it**:
- Satellite: count cars in Walmart parking lots -> estimate quarterly revenue 2-3 weeks before earnings
- Shipping: track container ship GPS via MarineTraffic/VesselFinder -> estimate import/export volumes
- Factory: nighttime light intensity at manufacturing facilities -> estimate production levels
- Retail: mobile phone geolocation foot traffic -> estimate store visits and conversion

**Where it fits in Operator 1**:
- **Data source**: No free satellite APIs with sufficient resolution. MarineTraffic has limited free API. Placer.ai has academic access.
- **Integration**: Too expensive for free-tier pipeline. Note for future premium data layer.
- **Realistic alternative**: Google Trends as a free proxy for product interest. `pytrends` library.

---

### M11. Cannibalization Rate Analysis

**What it is**: Does a new product eat into an existing product's revenue? iPhone SE cannibalizes iPhone 15 sales. Tesla Model 3 cannibalized Model S.

**Who uses it**: Multi-product company analysts, auto industry analysts, consumer electronics analysts.

**How experts do it**:
- Compare segment revenue changes after new product launch
- Cannibalization rate: `CR = (delta_old_product_revenue) / new_product_revenue`
- If CR > 50%: most of the "new" revenue is just moving from old product (net neutral)
- If CR < 20%: genuinely new demand (net positive)
- Cross-elasticity estimation: regress old product sales on new product price/availability

**Where it fits in Operator 1**:
- **Data source**: Requires multi-period segment data (SimFin quarterly segments)
- **Feature module**: Enhancement to `operator1/features/product_segments.py` (new module from M1)
  - `detect_cannibalization()`: compares sequential segment revenue changes around product launch dates
  - Uses product_catalysts.py launch detection as trigger
- **HF integration**: `growth_quality.py` -- high cannibalization = lower organic growth quality. `momentum_composite.py` -- adjust revenue acceleration for cannibalization.
- **Cache columns**: `cannibalization_rate`, `net_new_revenue_pct`

---

### M12. Network Effect Strength Measurement

**What it is**: For platform companies (Meta, Uber, Airbnb), measuring how strong the network effect is. Strong network effects = winner-take-all = durable moat.

**Who uses it**: Technology growth investors, venture capital (Metcalfe's Law, Sarnoff's Law).

**How experts do it**:
- Metcalfe coefficient: regress `Revenue ~ n^alpha` where n = user count. Alpha > 1.5 = strong network effect, Alpha < 1.0 = weak/no network effect.
- Cross-side elasticity (two-sided platforms): does adding drivers attract riders? `elasticity = %delta_riders / %delta_drivers`
- Engagement depth: DAU/MAU ratio (>50% = sticky), time-on-platform trend
- Proxy from financials: `Revenue per User growth` with constant or declining `Customer Acquisition Cost` = network effect strengthening

**Where it fits in Operator 1**:
- **Data source**: Limited from public filings. MAU/DAU sometimes disclosed in 10-K for tech companies. Revenue per user derivable from revenue / active users if disclosed.
- **Feature module**: Enhancement to `operator1/features/product_catalysts.py`
  - `estimate_network_effect()`: computes revenue/user growth vs CAC trend from available data
  - Falls back to revenue acceleration pattern analysis when user metrics unavailable
- **HF integration**: `growth_quality.py` -- network effect strength as a moat durability indicator. `valuation_quality.py` -- companies with strong network effects deserve higher valuation multiples.
- **Cache columns**: `network_effect_score`, `revenue_per_user_growth`, `moat_durability`

---

## Part 3: Integration Architecture

### New Module: `operator1/features/product_segments.py`

Core module that aggregates all product-level analysis. Runs at **Step 4d** (after cache build, before derived variables).

```
Inputs:
  - SimFin API segment data (or XBRL segment dimensions)
  - Cache (for revenue, margins, growth rates)
  - target_profile (sector, industry for BOM/TAM mapping)
  - patent_data (from PatentsView, optional)

Outputs (cache columns):
  - segment_hhi                    # Revenue concentration (0-1)
  - segment_count                  # Number of product lines
  - dominant_segment_pct           # % revenue from largest segment
  - dominant_segment_growth        # QoQ growth of largest segment
  - segment_diversification_delta  # Change in HHI over 4Q
  - product_lifecycle_stage        # Intro/Growth/Maturity/Decline (dominant)
  - growth_runway_quarters         # Estimated quarters until peak (Bass model)
  - maturity_concentration         # % revenue in Maturity+Decline
  - margin_mix_effect              # Gross margin bridge: mix component
  - margin_sustainability_score    # Mix-driven vs cost-driven margin change
  - pricing_power_index            # Real revenue growth vs industry PPI
  - cannibalization_rate           # New product eating old product sales
  - net_new_revenue_pct            # Genuine new demand vs cannibalized
  - customer_hhi                   # Customer concentration (10-K extraction)
  - top_customer_pct               # Largest customer % of revenue
  - estimated_market_share         # SOM estimate from revenue/sector GDP
  - som_trend_4q                   # Market share trend
  - input_cost_pressure            # BOM/commodity cost trend vs margins
  - patent_innovation_score        # Forward citation velocity + diversification
  - network_effect_score           # Revenue/user growth vs CAC (tech only)
```

### HF Pipeline Integration

| HF Module | Product Features Consumed | Impact |
|-----------|--------------------------|--------|
| `fcf_quality.py` | segment_hhi, input_cost_pressure, customer_hhi | FCF by segment, BOM pressure signal |
| `accruals_forensics.py` | margin_mix_effect, margin_sustainability | Distinguish legitimate vs suspect margin changes |
| `growth_quality.py` | product_lifecycle_stage, cannibalization_rate, patent_innovation_score, network_effect_score, som_trend | Decompose growth into organic/inorganic/cannibalized per segment |
| `momentum_composite.py` | dominant_segment_growth, pricing_power_index, net_new_revenue_pct | Per-segment revenue acceleration |
| `leverage_stress.py` | customer_hhi, top_customer_pct, input_cost_pressure | Top customer loss + BOM spike stress scenarios |
| `obs_risk.py` | customer_hhi, segment_hhi | Customer + product concentration as hidden risks |
| `valuation_quality.py` | pricing_power_index, network_effect_score, estimated_market_share | Pricing power premium, network effect premium, market share valuation anchor |
| `dcf_valuation.py` | product_lifecycle_stage, growth_runway_quarters, maturity_concentration | Per-segment terminal growth rates (Maturity segments get lower terminal growth) |
| `earnings_surprise.py` | input_cost_pressure, dominant_segment_growth | BOM cost pressure as earnings miss predictor |

### Main Pipeline Integration

| Step | Integration Point | What Changes |
|------|-------------------|--------------|
| Step 4d (NEW) | After cache build | `product_segments.py` fetches segment data, computes all product features |
| Step 4a.3 | `conflict_risk.py` | Product-weighted geographic exposure for tariff/trade risk |
| Step 5 | `derived_variables.py` | `pricing_power_index` and `segment_hhi` added to compute stages |
| Step 5i.5 | `product_catalysts.py` | Lifecycle classification, network effect, patent pipeline |
| Step 6-HF | `hedge_fund/engine.py` | All 9 HF modules consume product features (see table above) |
| Step 6 | `monte_carlo.py` | Per-segment survival simulation using segment-specific risk profiles |
| Step 6 | `_extra_vars` filter | Add `segment_`, `pricing_`, `patent_`, `customer_`, `som_` prefixes |

### Data Source Priority

| Source | Coverage | Cost | Latency | Quality |
|--------|----------|------|---------|---------|
| 1. XBRL segment dimensions (edgartools) | US SEC filers (8K+) | Free | Real-time (filing date) | High (regulatory data) |
| 2. SimFin segments API | US (3K+) | Free (2K/day) | Quarterly | High |
| 3. LLM extraction from 10-K text | US (all), Tier 2 (with PDFs) | LLM API cost | Post-filing | Medium |
| 4. fuzzy_pdf_parser with segment dictionary | Tier 2 markets | Free | Post-filing | Medium |
| 5. PatentsView | Global (US patents) | Free | Monthly | High |
| 6. FRED PPI/CPI | US | Free | Monthly | High |

---

## Summary: 12 Methods Ranked by Impact

| Rank | Method | Impact on Error | Data Available | Effort | Priority |
|------|--------|----------------|----------------|--------|----------|
| 1 | M1: Revenue Segment Decomposition | High (3-4pp) | SimFin/XBRL | Low | P0 |
| 2 | M6: Pricing Power Index | High (2-3pp) | FRED PPI + cache | Low | P0 |
| 3 | M3: Gross Margin Bridge | Medium (1-2pp) | Cache (derived) | Low | P1 |
| 4 | M5: Customer Concentration | Medium (1-2pp) | 10-K text/LLM | Medium | P1 |
| 5 | M2: Product Lifecycle (Bass) | Medium (1-2pp) | SimFin segments | Medium | P1 |
| 6 | M4: TAM/Market Share | Medium (1pp) | Macro + cache | Low | P1 |
| 7 | M7: Patent Citations | Medium (1pp) | PatentsView | Medium | P2 |
| 8 | M9: BOM Cost Tracking | Medium (1pp) | FRED commodities | Low | P2 |
| 9 | M11: Cannibalization | Low-Medium | SimFin segments | Medium | P2 |
| 10 | M12: Network Effects | Low-Medium | 10-K/limited | Medium | P2 |
| 11 | M8: Employee Sentiment | Low | Glassdoor/limited | High | P3 |
| 12 | M10: Satellite/Geospatial | Low | Paid only | High | P3 |

**Top 2 to implement first**: M1 (Revenue Segments) and M6 (Pricing Power) -- both are high-impact, low-effort, and use free data sources already available in the pipeline.
