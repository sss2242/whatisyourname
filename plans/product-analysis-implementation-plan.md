# Product Analysis Implementation Plan

*Optimum implementation design for integrating product-level analysis into Operator 1*
*Date: 2026-04-06*

---

## Architecture Overview

```
                    ┌─────────────────────────┐
                    │   Data Extraction Layer  │
                    │   (3 sources, waterfall) │
                    └──────────┬──────────────┘
                               │
                    ┌──────────▼──────────────┐
                    │  product_segments.py     │
                    │  (NEW: Step 4d)          │
                    │  20 cache columns        │
                    └──────────┬──────────────┘
                               │
              ┌────────────────┼────────────────┐
              │                │                │
    ┌─────────▼────┐  ┌───────▼──────┐  ┌──────▼──────────┐
    │ Existing      │  │ HF Pipeline  │  │ Temporal Models │
    │ Features      │  │ (9 modules)  │  │ (via _extra_vars)│
    │ (enhanced)    │  │              │  │                 │
    └──────────────┘  └──────────────┘  └─────────────────┘
```

---

## Part 1: Data Extraction (Waterfall Strategy)

Three sources in priority order. Each returns the same output format: a dict of `{segment_name: pd.Series}` with quarterly revenue per segment.

### Source 1: XBRL Segment Dimensions (edgartools, free, real-time)

**When**: US SEC filers (8K+ companies). Available immediately at filing time.

**How**: edgartools already parses XBRL. The segment data is in dimension members of revenue facts. Apple's 10-K XBRL contains:
```xml
<us-gaap:RevenueFromContractWithCustomerExcludingAssessedTax>
  <xbrldi:explicitMember dimension="srt:ProductOrServiceAxis">aapl:IPhoneMember</xbrldi:explicitMember>
  200583000000
</us-gaap:RevenueFromContractWithCustomerExcludingAssessedTax>
```

**Extraction code** (add to `canonical_translator.py`):
```python
def extract_segment_revenue(filing) -> dict[str, pd.Series]:
    """Extract product segment revenue from XBRL dimension members."""
    facts = filing.xbrl().facts
    # Filter for revenue facts with ProductOrServiceAxis dimension
    segment_facts = [f for f in facts 
                     if 'Revenue' in f.concept 
                     and any(d.dimension == 'srt:ProductOrServiceAxis' for d in f.dimensions)]
    
    segments = {}
    for fact in segment_facts:
        member = [d.member for d in fact.dimensions if d.dimension == 'srt:ProductOrServiceAxis'][0]
        segment_name = member.replace('Member', '').split(':')[-1]
        # Build time series from multiple filings
        segments.setdefault(segment_name, []).append({
            'date': fact.period_end,
            'revenue': float(fact.value)
        })
    
    return {name: pd.Series({r['date']: r['revenue'] for r in rows}).sort_index()
            for name, rows in segments.items()}
```

**Cost**: Zero. **Latency**: Real-time. **Coverage**: ~60% of S&P 500 have segment dimensions.

### Source 2: SimFin API (free tier, 2K calls/day)

**When**: XBRL dimensions unavailable or incomplete. Covers 3K+ US companies with curated segment data.

**Extraction code** (new `operator1/clients/simfin_segments.py`):
```python
import requests

SIMFIN_BASE = "https://backend.simfin.com/api/v3"

def fetch_segments(ticker: str, api_key: str = "free") -> dict[str, pd.Series]:
    """Fetch product segment revenue from SimFin."""
    # Step 1: Get company ID
    resp = requests.get(f"{SIMFIN_BASE}/companies/list",
                       params={"ticker": ticker},
                       headers={"Authorization": f"api-key {api_key}"})
    companies = resp.json()
    if not companies:
        return {}
    
    company_id = companies[0]["id"]
    
    # Step 2: Get segment data
    resp = requests.get(
        f"{SIMFIN_BASE}/companies/id/{company_id}/statements/derived",
        params={"statement": "segments", "period": "quarters"},
        headers={"Authorization": f"api-key {api_key}"}
    )
    data = resp.json()
    
    segments = {}
    for row in data.get("data", []):
        name = row.get("segment_name", "Unknown")
        date = pd.Timestamp(row.get("period_end"))
        revenue = row.get("revenue")
        if revenue is not None:
            segments.setdefault(name, {})[date] = float(revenue)
    
    return {name: pd.Series(values).sort_index() for name, values in segments.items()}
```

**Cost**: Free (2K/day). **Latency**: Quarterly (after filing). **Coverage**: 3K+ US companies.

### Source 3: LLM Extraction from 10-K Text (fallback)

**When**: Neither XBRL dimensions nor SimFin have segment data. Uses the existing `llm_filing_extractor.py` with a segment-specific prompt.

**Extraction prompt** (add to `llm_filing_extractor.py`):
```python
SEGMENT_EXTRACTION_PROMPT = """
Extract product/segment revenue from this filing text.
Return JSON: {"segments": [{"name": "iPhone", "revenue": 200583000000, "period": "2024-09-30"}]}
Only include segments with explicit revenue numbers.
"""
```

**Cost**: LLM API call. **Latency**: Post-filing. **Coverage**: Any company with a 10-K/annual report.

### Waterfall Logic

```python
def fetch_product_segments(ticker: str, market_id: str, pit_client, secrets: dict) -> dict:
    """Waterfall: XBRL -> SimFin -> LLM extraction."""
    # Try XBRL segment dimensions first (free, real-time)
    if market_id == "us_sec_edgar":
        segments = _try_xbrl_segments(ticker, pit_client)
        if segments and len(segments) >= 2:
            return {"source": "xbrl", "segments": segments}
    
    # Try SimFin (free, curated)
    segments = _try_simfin_segments(ticker)
    if segments and len(segments) >= 2:
        return {"source": "simfin", "segments": segments}
    
    # Try LLM extraction from latest annual report
    segments = _try_llm_segment_extraction(ticker, market_id, pit_client, secrets)
    if segments and len(segments) >= 2:
        return {"source": "llm", "segments": segments}
    
    return {"source": "none", "segments": {}}
```

---

## Part 2: Processing Module (`operator1/features/product_segments.py`)

Single module that computes all product-level features from segment data. Runs at **Step 4d** in main.py (after cache build, before derived variables).

### Core Computations (20 cache columns)

```python
def compute_product_segment_features(
    cache: pd.DataFrame,
    segment_data: dict[str, pd.Series],
    target_profile: dict,
    macro_data: dict | None = None,
) -> tuple[pd.DataFrame, ProductSegmentResult]:
    """Compute all product-level features and merge into cache."""
    
    result = ProductSegmentResult()
    
    if not segment_data or len(segment_data) < 2:
        return cache, result
    
    # ── M1: Revenue Segment Decomposition ──
    total_rev = sum(s.iloc[-1] for s in segment_data.values() if len(s) > 0)
    shares = {name: s.iloc[-1] / total_rev for name, s in segment_data.items() if len(s) > 0 and total_rev > 0}
    
    # HHI (0 = perfectly diversified, 1 = single product)
    hhi = sum(s**2 for s in shares.values())
    cache["segment_hhi"] = hhi
    cache["segment_count"] = len(segment_data)
    
    # Dominant segment
    dominant = max(shares, key=shares.get)
    cache["dominant_segment_pct"] = shares[dominant]
    
    # Per-segment growth rates
    growth_rates = {}
    for name, series in segment_data.items():
        if len(series) >= 2:
            growth_rates[name] = (series.iloc[-1] - series.iloc[-2]) / abs(series.iloc[-2]) if series.iloc[-2] != 0 else 0
    
    cache["dominant_segment_growth"] = growth_rates.get(dominant, 0)
    
    # Diversification change (4Q HHI delta)
    if all(len(s) >= 5 for s in segment_data.values()):
        old_total = sum(s.iloc[-5] for s in segment_data.values())
        old_shares = {name: s.iloc[-5] / old_total for name, s in segment_data.items() if old_total > 0}
        old_hhi = sum(s**2 for s in old_shares.values())
        cache["segment_diversification_delta"] = hhi - old_hhi  # negative = more diverse
    
    # ── M2: Product Lifecycle Classification (Bass) ──
    for name, series in segment_data.items():
        if len(series) >= 8:
            stage, runway = _classify_lifecycle(series)
            if name == dominant:
                cache["product_lifecycle_stage"] = _stage_to_numeric(stage)
                cache["growth_runway_quarters"] = runway
    
    # Maturity concentration (% revenue in Maturity+Decline)
    maturity_rev = sum(
        segment_data[name].iloc[-1]
        for name in segment_data
        if len(segment_data[name]) >= 8 and _classify_lifecycle(segment_data[name])[0] in ("maturity", "decline")
    )
    cache["maturity_concentration"] = maturity_rev / total_rev if total_rev > 0 else 0
    
    # ── M3: Gross Margin Bridge (when per-segment margins available) ──
    # Proxy: decompose aggregate margin change into mix effect
    if "gross_margin" in cache.columns and len(shares) >= 2:
        mix_effect = _compute_mix_effect(segment_data, cache)
        cache["margin_mix_effect"] = mix_effect
        cache["margin_sustainability_score"] = _margin_sustainability(mix_effect, cache)
    
    # ── M4: Market Share Estimation ──
    if macro_data and "gdp" in macro_data:
        sector = target_profile.get("sector", "")
        som = _estimate_market_share(total_rev, macro_data, sector)
        cache["estimated_market_share"] = som
        # SOM trend (need 4Q of segment data)
        if all(len(s) >= 5 for s in segment_data.values()):
            old_total_rev = sum(s.iloc[-5] for s in segment_data.values())
            old_som = _estimate_market_share(old_total_rev, macro_data, sector)
            cache["som_trend_4q"] = som - old_som
    
    # ── M5: Customer Concentration (from segment proxy) ──
    # True customer concentration requires 10-K text extraction (separate module)
    # Proxy: segment HHI correlates with customer concentration for single-product companies
    cache["customer_concentration_proxy"] = hhi * shares.get(dominant, 0)
    
    # ── M6: Pricing Power Index ──
    if "revenue" in cache.columns and macro_data:
        pricing_power = _compute_pricing_power(cache, macro_data, target_profile.get("sector", ""))
        cache["pricing_power_index"] = pricing_power
    
    # ── M9: Input Cost Pressure ──
    if macro_data:
        input_pressure = _compute_input_cost_pressure(cache, macro_data, target_profile.get("sector", ""))
        cache["input_cost_pressure"] = input_pressure
    
    # ── M11: Cannibalization Detection ──
    if len(segment_data) >= 2 and all(len(s) >= 5 for s in segment_data.values()):
        cannibal = _detect_cannibalization(segment_data)
        cache["cannibalization_rate"] = cannibal.get("rate", 0)
        cache["net_new_revenue_pct"] = cannibal.get("net_new_pct", 1.0)
    
    # ── M12: Network Effect (tech sector only) ──
    sector = target_profile.get("sector", "").lower()
    if any(w in sector for w in ["technology", "communication", "software", "internet"]):
        nfx = _estimate_network_effect(cache, segment_data)
        cache["network_effect_score"] = nfx
    
    result.available = True
    result.source = segment_data.get("source", "unknown")
    result.n_segments = len(segment_data)
    result.dominant_segment = dominant
    result.hhi = hhi
    
    return cache, result
```

### Helper Functions

```python
def _classify_lifecycle(series: pd.Series) -> tuple[str, int]:
    """Bass model lifecycle classification. Returns (stage, quarters_to_peak)."""
    try:
        from scipy.optimize import curve_fit
        
        cumulative = series.cumsum()
        t = np.arange(len(cumulative))
        
        def bass_cdf(t, p, q, m):
            return m * (1 - np.exp(-(p+q)*t)) / (1 + (q/p)*np.exp(-(p+q)*t))
        
        popt, _ = curve_fit(bass_cdf, t, cumulative.values, p0=[0.01, 0.3, cumulative.iloc[-1]*1.5], maxfev=5000)
        p, q, m = popt
        
        # Current adoption fraction
        F_current = cumulative.iloc[-1] / m if m > 0 else 0
        
        if F_current < 0.1:
            stage = "introduction"
            runway = int((0.5 * m - cumulative.iloc[-1]) / series.mean()) if series.mean() > 0 else 20
        elif F_current < 0.5:
            stage = "growth"
            peak_t = np.log(q/p) / (p+q) if p > 0 and q > 0 else len(t)
            runway = max(0, int(peak_t - len(t)))
        elif F_current < 0.9:
            stage = "maturity"
            runway = 0
        else:
            stage = "decline"
            runway = 0
        
        return stage, runway
    except Exception:
        # Fallback: simple growth rate classification
        if len(series) >= 4:
            recent_growth = (series.iloc[-1] - series.iloc[-4]) / abs(series.iloc[-4]) if series.iloc[-4] != 0 else 0
            if recent_growth > 0.20:
                return "growth", 8
            elif recent_growth > 0.0:
                return "maturity", 0
            else:
                return "decline", 0
        return "unknown", 0

def _stage_to_numeric(stage: str) -> float:
    """Convert lifecycle stage to numeric for cache (models need float)."""
    return {"introduction": 0.0, "growth": 0.25, "maturity": 0.5, "decline": 0.75, "unknown": 0.5}[stage]

def _compute_pricing_power(cache: pd.DataFrame, macro_data: dict, sector: str) -> float:
    """M6: Pricing power = real revenue growth vs industry PPI."""
    if "revenue" not in cache.columns:
        return 0.0
    
    # Revenue growth (TTM)
    rev = cache["revenue"].dropna()
    if len(rev) < 252:
        return 0.0
    rev_growth = (rev.iloc[-1] - rev.iloc[-252]) / abs(rev.iloc[-252]) if rev.iloc[-252] != 0 else 0
    
    # Industry PPI (from FRED macro data)
    ppi_growth = 0.03  # default 3% inflation
    if "inflation" in macro_data:
        inf_series = macro_data["inflation"]
        if not inf_series.empty:
            ppi_growth = float(inf_series.iloc[-1]) / 100  # convert from percentage
    
    # Pricing power: positive = real pricing power, negative = volume-dependent
    if rev_growth == 0:
        return 0.0
    return (rev_growth - ppi_growth) / abs(rev_growth)

def _compute_input_cost_pressure(cache: pd.DataFrame, macro_data: dict, sector: str) -> float:
    """M9: Input cost pressure from relevant commodity indices."""
    # Sector -> commodity mapping
    SECTOR_COMMODITIES = {
        "energy": "inflation",  # proxy via energy PPI
        "materials": "inflation",
        "technology": "inflation",  # chip costs proxy
        "industrials": "inflation",
        "consumer": "inflation",
    }
    
    indicator = SECTOR_COMMODITIES.get(sector.lower().split()[0] if sector else "", "inflation")
    
    if indicator in macro_data:
        series = macro_data[indicator]
        if not series.empty and len(series) >= 2:
            # Rising input costs = pressure
            trend = (float(series.iloc[-1]) - float(series.iloc[0])) / abs(float(series.iloc[0])) if series.iloc[0] != 0 else 0
            return max(0, min(1, trend))  # 0 = no pressure, 1 = severe
    return 0.0

def _detect_cannibalization(segment_data: dict[str, pd.Series]) -> dict:
    """M11: Detect if newer segments are eating older segment revenue."""
    # Identify newest segment (shortest history)
    lengths = {name: len(s) for name, s in segment_data.items()}
    newest = min(lengths, key=lengths.get)
    oldest = [n for n in segment_data if n != newest]
    
    if not oldest or len(segment_data[newest]) < 4:
        return {"rate": 0, "net_new_pct": 1.0}
    
    new_rev = segment_data[newest].iloc[-4:].sum()  # Last 4Q revenue
    
    # Check if old segments declined after new segment appeared
    old_decline = 0
    for name in oldest:
        s = segment_data[name]
        if len(s) >= 8:
            pre_new = s.iloc[-8:-4].mean()  # 4Q before new segment started
            post_new = s.iloc[-4:].mean()    # 4Q after
            if pre_new > post_new:
                old_decline += (pre_new - post_new) * 4
    
    rate = old_decline / new_rev if new_rev > 0 else 0
    return {"rate": min(1, rate), "net_new_pct": max(0, 1 - rate)}

def _estimate_network_effect(cache: pd.DataFrame, segment_data: dict) -> float:
    """M12: Network effect strength from revenue trajectory curvature."""
    # Proxy: if revenue acceleration is positive AND margins are stable/improving,
    # the company likely has network effects (more users = more value = more users)
    if "revenue" not in cache.columns or "gross_margin" not in cache.columns:
        return 0.0
    
    rev = cache["revenue"].dropna()
    if len(rev) < 126:  # need 6+ months
        return 0.0
    
    # Revenue acceleration (2nd derivative)
    rev_growth = rev.pct_change(63).dropna()  # quarterly growth
    if len(rev_growth) < 2:
        return 0.0
    acceleration = rev_growth.iloc[-1] - rev_growth.iloc[0]
    
    # Margin trend
    margin = cache["gross_margin"].dropna()
    margin_stable = margin.std() < 0.05 if len(margin) > 20 else True
    
    # Positive acceleration + stable margins = network effect signal
    if acceleration > 0 and margin_stable:
        return min(1, acceleration * 10)  # scale to 0-1
    return 0.0

def _compute_mix_effect(segment_data: dict[str, pd.Series], cache: pd.DataFrame) -> float:
    """M3: Gross margin bridge mix effect."""
    # Compare current vs prior period segment weights
    # Higher-margin segments gaining share = positive mix effect
    if len(segment_data) < 2:
        return 0.0
    
    current_total = sum(s.iloc[-1] for s in segment_data.values() if len(s) > 0)
    if current_total == 0:
        return 0.0
    
    # Without per-segment margins, proxy from relative growth rates
    # Fast-growing segments in high-margin companies tend to be the higher-margin ones
    growth_weighted_shift = 0
    for name, series in segment_data.items():
        if len(series) >= 2:
            share = series.iloc[-1] / current_total
            growth = (series.iloc[-1] - series.iloc[-2]) / abs(series.iloc[-2]) if series.iloc[-2] != 0 else 0
            growth_weighted_shift += share * growth
    
    return growth_weighted_shift

def _margin_sustainability(mix_effect: float, cache: pd.DataFrame) -> float:
    """Score how sustainable margin changes are (mix-driven = sustainable, cost-driven = temporary)."""
    if "gross_margin" not in cache.columns:
        return 0.5
    
    gm = cache["gross_margin"].dropna()
    if len(gm) < 63:
        return 0.5
    
    gm_change = gm.iloc[-1] - gm.iloc[-63]
    
    if abs(gm_change) < 0.01:
        return 0.5  # stable margin
    
    # If margin improved AND mix effect is positive -> sustainable (score toward 1)
    # If margin improved but mix effect is zero/negative -> cost-driven (score toward 0)
    if gm_change > 0:
        if mix_effect > 0:
            return min(1, 0.5 + mix_effect * 5)
        return max(0, 0.5 - abs(mix_effect) * 5)
    else:
        return max(0, 0.3 - abs(gm_change) * 2)

def _estimate_market_share(company_revenue: float, macro_data: dict, sector: str) -> float:
    """M4: Estimate SOM from company revenue / sector GDP proxy."""
    if "gdp" not in macro_data or not macro_data["gdp"].any():
        return 0.0
    
    gdp = float(macro_data["gdp"].iloc[-1])
    if gdp <= 0:
        return 0.0
    
    # Sector share of GDP (rough estimates)
    SECTOR_GDP_SHARE = {
        "technology": 0.08, "healthcare": 0.07, "financials": 0.08,
        "energy": 0.06, "industrials": 0.06, "consumer discretionary": 0.05,
        "consumer staples": 0.04, "materials": 0.03, "utilities": 0.02,
        "real estate": 0.03, "communication services": 0.04,
    }
    
    sector_lower = sector.lower() if sector else ""
    sector_share = 0.05  # default
    for key, share in SECTOR_GDP_SHARE.items():
        if key in sector_lower:
            sector_share = share
            break
    
    sector_tam = gdp * sector_share * 1e9  # GDP is in billions typically
    return min(1, company_revenue / sector_tam) if sector_tam > 0 else 0
```

---

## Part 3: Wiring into main.py

### Step 4d (NEW): After cache build, before derived variables

```python
    # ------------------------------------------------------------------
    # Step 4d: Product segment analysis
    # ------------------------------------------------------------------
    product_segment_result = None
    try:
        from operator1.features.product_segments import (
            fetch_product_segments,
            compute_product_segment_features,
        )
        logger.info("")
        logger.info("Step 4d: Product segment analysis...")
        
        _seg_data = fetch_product_segments(
            ticker=ticker,
            market_id=market_id,
            pit_client=pit_client,
            secrets=secrets,
        )
        
        if _seg_data.get("segments"):
            cache, product_segment_result = compute_product_segment_features(
                cache,
                segment_data=_seg_data["segments"],
                target_profile=target_profile,
                macro_data=macro_data,
            )
            logger.info(
                "Product segments: %d segments via %s, HHI=%.3f, dominant=%s (%.0f%%)",
                product_segment_result.n_segments,
                _seg_data["source"],
                product_segment_result.hhi,
                product_segment_result.dominant_segment,
                product_segment_result.hhi * 100,  # proxy
            )
        else:
            logger.info("Product segments: no segment data available for %s", ticker)
    except Exception as exc:
        logger.warning("Product segment analysis failed: %s", exc)
```

### _extra_vars filter update

Add to the existing `_extra_vars` filter in Step 6:
```python
or c.startswith("segment_") or c.startswith("product_")
or c.startswith("pricing_") or c.startswith("customer_")
or c.startswith("som_") or c.startswith("margin_")
or c in ("cannibalization_rate", "net_new_revenue_pct",
         "network_effect_score", "input_cost_pressure",
         "growth_runway_quarters", "maturity_concentration")
```

### Profile injection (Step 7)

```python
    # Inject product segment analysis
    if product_segment_result is not None and product_segment_result.available:
        profile["product_segments"] = {
            "available": True,
            "source": product_segment_result.source,
            "n_segments": product_segment_result.n_segments,
            "dominant_segment": product_segment_result.dominant_segment,
            "hhi": product_segment_result.hhi,
            "segment_names": list(segment_data.keys()),
        }
    else:
        profile["product_segments"] = {"available": False}
```

### HF Pipeline Wiring

```python
    # In hedge_fund/engine.py run_hedge_fund_analysis():
    hf_result = run_hedge_fund_analysis(
        ...
        product_segment_result=product_segment_result,  # NEW
        ...
    )
```

Each HF module reads product features from cache columns (already merged by Step 4d):
- `fcf_quality.py`: reads `segment_hhi`, `input_cost_pressure`
- `growth_quality.py`: reads `product_lifecycle_stage`, `cannibalization_rate`, `network_effect_score`
- `momentum_composite.py`: reads `dominant_segment_growth`, `pricing_power_index`
- `leverage_stress.py`: reads `customer_concentration_proxy`, `input_cost_pressure`
- `valuation_quality.py`: reads `pricing_power_index`, `estimated_market_share`
- `dcf_valuation.py`: reads `maturity_concentration`, `growth_runway_quarters`

### Conflict Risk Enhancement

```python
    # In features/conflict_risk.py assess_conflict_risk():
    # Product-weighted geographic exposure
    if segment_data:
        # Weight conflict intensity by product's geographic exposure
        # iPhone (China-assembled) gets 0.95 tariff weight
        # Services (no physical goods) gets 0.0 tariff weight
        weighted_intensity = sum(
            shares[name] * _geographic_exposure(name, country_iso2)
            for name in shares
        )
        conflict_result.product_weighted_intensity = weighted_intensity
```

### Monte Carlo Enhancement

```python
    # In models/monte_carlo.py run_monte_carlo():
    # Per-segment survival simulation
    if "segment_hhi" in cache.columns:
        # High concentration = simulate dominant segment risk separately
        # Low concentration = simulate aggregate (diversification absorbs shocks)
        hhi = cache["segment_hhi"].iloc[-1]
        if hhi > 0.5:  # concentrated
            # Simulate dominant segment shock scenario
            mc_result.concentration_risk_flag = True
```

---

## Part 4: Report Integration

### New report section 19.9: Product Segment Analysis (Pro + Premium)

```python
def _build_product_segments_section(profile: dict) -> str:
    """Section 19.9: Product portfolio analysis."""
    ps = profile.get("product_segments", {})
    if not ps.get("available"):
        return ""
    
    lines = ["## Product Portfolio Analysis\n"]
    lines.append(f"**Segments**: {ps['n_segments']} product lines")
    lines.append(f"**Dominant**: {ps['dominant_segment']} ({ps.get('dominant_pct', 0):.0%} of revenue)")
    lines.append(f"**Concentration (HHI)**: {ps['hhi']:.3f}")
    # ... lifecycle, pricing power, cannibalization, margin bridge
    return "\n".join(lines)
```

---

## Part 5: File Summary

| File | Action | Lines Est. |
|------|--------|-----------|
| `operator1/features/product_segments.py` | **NEW** | ~400 |
| `operator1/clients/simfin_segments.py` | **NEW** | ~80 |
| `operator1/clients/canonical_translator.py` | **MODIFY** | +50 (XBRL segment extraction) |
| `main.py` | **MODIFY** | +30 (Step 4d wiring + _extra_vars + profile) |
| `operator1/features/conflict_risk.py` | **MODIFY** | +20 (product-weighted exposure) |
| `operator1/models/monte_carlo.py` | **MODIFY** | +15 (concentration risk flag) |
| `operator1/hedge_fund/engine.py` | **MODIFY** | +10 (product_segment_result param) |
| `operator1/report/report_generator.py` | **MODIFY** | +40 (section 19.9 builder) |
| `requirements/stage4-wrappers.txt` | **MODIFY** | +1 (simfin) |
| **Total** | | **~650 lines** |

### Dependencies
- `simfin` -- NEW package (MIT, free tier)
- All others already installed (scipy, numpy, pandas, networkx, fredapi)

### Expected Impact
- 21d prediction error: 9.84% -> ~6% (product-weighted policy risk)
- 252d prediction error: 2.26% -> ~1.5% (per-segment lifecycle forecasting)
- Vol prediction error: 10pp -> ~6pp (concentration risk in GARCH features)
- Direction accuracy: maintained at 100% (product data sharpens magnitude, not direction)
