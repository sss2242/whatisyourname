# Middleware Deep Architecture -- How Data Actually Flows

A detailed analysis of the 20 middleware modules, how they transform data at each stage, what invariants they maintain, where the design is strong, and where it has structural tensions.

---

## The Core Problem These Modules Solve

The pipeline ingests data from 25 different government filing APIs across 10 regions. Each API speaks a different language:
- SEC EDGAR returns US-GAAP XBRL concepts like `RevenueFromContractWithCustomerExcludingAssessedTax`
- DART returns Korean names like `매출액`
- CVM returns hierarchical account codes like `3.01`
- MOPS returns Traditional Chinese like `營業收入合計`
- Some return long-format (one row per concept per filing), others return wide-format tables

The middleware's job is to take all of this, normalize it into a single daily-frequency DataFrame with canonical column names, fill gaps intelligently, and produce a survival-state-annotated time series that 25+ temporal models can consume without knowing which country the data came from.

---

## Stage 1: Data Acquisition Layer

### The Two Paths

Data enters the middleware through two distinct paths:

**Path A -- Structured API data** (Tier 1 markets + some Tier 2):
```
PIT Client API call
    -> raw DataFrame (region-specific columns)
    -> Canonical Translator (concept mapping)
    -> pivot_to_canonical_wide()
    -> wide DataFrame (report_date as index, canonical columns)
```

**Path B -- PDF filing extraction** (Tier 2 markets without structured APIs):
```
Filing Discoverer (exchange announcement search)
    -> PDF download (validated: %PDF magic bytes)
    -> LLM Filing Extractor (Gemini/Claude with taxonomy hints)
       OR Fuzzy PDF Parser (camelot + fuzzy string matching, no LLM)
    -> canonical long-format DataFrame
    -> pivot_to_canonical_wide()
    -> wide DataFrame (same output contract as Path A)
```

**Path C -- Synthetic derivation** (Switzerland only):
```
SIX APIs (dividend history + capital structure + notices)
    -> six_derived_proxies.py
    -> 22 canonical fields derived mathematically
       (Kalman, PELT, Merton structural model, L1 reconstruction)
    -> wide DataFrame (same output contract)
```

**What this means architecturally**: The Canonical Translator and the pivot function are the *convergence point*. Regardless of whether data came from XBRL parsing, PDF extraction, or mathematical derivation, everything exits as a wide DataFrame with the same 30+ canonical column names. This is the critical contract that makes the rest of the pipeline source-agnostic.

### How Concept Mapping Actually Works

The translator maintains 8 static dictionaries (one per accounting standard) totaling ~500 explicit mappings. When a raw concept arrives:

1. **Exact match** against the market-specific dictionary
2. **Namespace-stripped match** -- `ifrs-full:Revenue` becomes `Revenue`, retry
3. **Alias resolution** -- handles variant spellings (`sga_expense` -> `sga_expenses`)
4. **Case-insensitive match** -- catches FRS102/UK-GAAP casing variations
5. **Identity check** -- the concept might already BE a canonical name
6. **LLM fallback** (optional) -- batches up to 30 unmapped concepts, asks Gemini/Claude, caches results to disk

The LLM fallback is interesting because it only fires when important canonical fields are missing after static mapping. It does not re-run for concepts that were already resolved. The disk cache (`cache/llm_concept_map.json`) means the LLM is never asked the same question twice across pipeline runs.

**Structural observation**: The CAS (Chinese) mapping handles a subtle issue -- China reports selling expenses and admin expenses separately instead of combined SGA. The pivot function detects `sga_selling` + `sga_admin` columns and sums them into `sga_expenses`. This kind of cross-standard normalization is baked into specific points in the pipeline rather than being a general framework.

---

## Stage 2: Validation and Cleaning

### Data Reconciliation

Before entering the cache, each statement DataFrame passes through 4 validation steps:

1. **Field name normalization** -- 101 aliases covering camelCase API variants
2. **Filing date validation** -- asserts `filing_date >= report_date`. Violations are corrected to `report_date + 1 day` (not dropped -- the data is still valuable, the date was just wrong)
3. **Duplicate removal** -- keeps latest `filing_date` per `report_date` (amendments supersede originals)
4. **Staleness detection** -- flags data > 180 days old

**Why this matters**: Filing date validation is the PIT (Point-in-Time) guarantee. If a Q4 report with `report_date=2025-12-31` was filed on `filing_date=2026-02-15`, the data cannot be used for any analysis of days before February 15th. The reconciliation layer catches cases where upstream APIs return filing dates that precede the report period (which would be a time-travel violation).

**Tension**: The 180-day staleness threshold is hardcoded. Annual-only filers (some German and Swiss companies) legitimately have 12+ months between filings. The filing calendar module handles this with market-specific thresholds later in the pipeline, but the reconciliation layer still logs warnings. Not a bug -- just noise in the logs.

---

## Stage 3: Cache Construction

### The Daily Spine

The cache is built around a DatetimeIndex of business days. The OHLCV (price) data forms the spine because it is the highest-frequency data available:

```
OHLCV DataFrame (daily: date, open, high, low, close, volume)
    -> set as DatetimeIndex
    -> this becomes the "spine" of the daily cache
```

When OHLCV is unavailable (common for Tier 2 markets or private companies), the spine is a synthetic business-day range from `DATE_START` to `DATE_END`. This means downstream models always see a continuous daily index, even when there is no price data -- they just see NaN for price columns.

### Statement Merge -- The Key Design Decision

Financial statements arrive as periodic data (quarterly, semi-annual, annual). The pipeline needs to get this onto a daily index. Here is where a critical design decision was made:

**Merge key = `report_date`, NOT `filing_date`**

This seems counterintuitive for a PIT system. The filing_date is when the information became public. But the pipeline uses report_date because:
- Filing dates can be duplicated (a 10-K filing contains both Q4 and annual data, same filing_date, different report_dates)
- The as-of join logic (`report_date <= t`) still satisfies PIT constraints because `report_date <= filing_date` always holds
- Using filing_date would create duplicate rows in the index (two reports filed on the same day)

The PIT guarantee is maintained through a different mechanism: the LookAheadError check in cache_builder.py validates that no statement row is applied to a day earlier than its report_date.

### Frequency-Aware Interpolation vs Flat Forward-Fill

This is architecturally significant. The pipeline has two interpolation strategies:

**Flat forward-fill** (fallback, single-filing, or missing interpolator):
```
Q1: revenue = 100M  ->  Day 1-63: revenue = 100M
Q2: revenue = 120M  ->  Day 64-126: revenue = 120M
```

**Frequency-aware interpolation** (primary, when >= 2 filings):
- Stock variables (balance sheet): LINEAR interpolation between filings. Total assets change gradually, not in steps.
- Flow variables (income/cashflow): DISTRIBUTE period total across business days. $100M quarterly revenue becomes ~$1.59M/day across 63 business days.

**Why this matters for models**: Flat forward-fill creates artificial discontinuities on filing days that confuse gradient-based models (LSTM, Transformer) and inflate GARCH volatility estimates. Linear interpolation for stock variables and daily distribution for flow variables create smooth trajectories that better represent economic reality.

**The confidence layer**: Each interpolated value gets a confidence score based on:
- Distance from nearest filing (closer = higher confidence)
- Filing frequency (quarterly = 0.85 base, annual = 0.50 base)
- Variable type (stock = 1.0x, flow = 0.85x)

These confidence scores flow into the estimator and can be consumed by models that support sample weighting.

---

## Stage 4: Estimation Engine

### The Three-Phase Architecture

The estimator is the most algorithmically complex middleware module. Its job: fill remaining NaN values using increasingly sophisticated methods.

**Phase 1 -- Deterministic Identity Fill**

Four accounting identities are applied iteratively (up to 5 rounds):
```
total_assets = total_liabilities + total_equity
free_cash_flow = operating_cash_flow - |capex|
net_debt = total_debt - cash_and_equivalents
total_debt = short_term_debt + long_term_debt
```

Each identity can solve for any ONE missing variable when the other two are present. Cascading: filling total_equity in round 1 might enable computing current_ratio in round 2. This is the "Sudoku" metaphor -- each fill reveals more information.

**Phase 2 -- Missingness Classification**

Each remaining NaN is classified:
- **MCAR/MAR**: The gap is structural (API didn't return it, filing period hasn't arrived yet). The missing value is independent of the true value.
- **MNAR**: The company is likely hiding data. Evidence: the field was reported in prior periods but vanished, OR >80% of peer companies report this field but this company doesn't.

This classification drives which estimator handles the imputation:

**Phase 3a -- MAR path** (MICE + GP + Matrix Completion):
Three methods run independently and their results are combined via inverse-variance-weighted median. The inter-method agreement drives the confidence score. If MICE says revenue is $50M, GP says $48M, and Matrix Completion says $52M, the tight agreement yields high confidence. If they disagree wildly, confidence drops.

**Phase 3b -- MNAR path** (Heckman + Pattern-Mixture + Sensitivity Bounds + GAIN):
The Heckman selection model is particularly clever -- it models WHY the data is missing (using company size, prior reporting rates, overall data quality as selection variables), then corrects for the selection bias in the imputation. The sensitivity bounds compute tipping points: "the hidden value would need to be below X to change the survival classification."

**What gets estimated**: 22 variables across all three statement types. These are the core financial fields that downstream models need: revenue, total_assets, cash_and_equivalents, operating_cash_flow, etc.

**Critical invariant**: Observed values are NEVER overwritten. The estimator adds parallel columns (`{var}_observed`, `{var}_estimated`, `{var}_final`, `{var}_source`, `{var}_confidence`) so models can distinguish real data from estimates.

---

## Stage 5: Survival State Machine

### Base Timeline

The survival timeline is a state machine that classifies every day into one of 6 modes based on 3 binary inputs:

```
company_survival_flag = f(current_ratio, debt_to_equity, fcf_yield, drawdown, conflict, sanctions)
country_survival_flag = f(credit_spread, unemployment_rise, yield_curve, fx_volatility)
country_protected_flag = f(strategic_sector, market_cap/GDP, emergency_rate_cut)
```

The state transition logic:
```
if !company AND !country -> "normal"
if company AND !country -> "company_only"
if !company AND country AND protected -> "country_protected"
if !company AND country AND !protected -> "country_exposed"
if company AND country AND protected -> "both_protected"
if company AND country AND !protected -> "both_unprotected"
```

### Enriched Timeline (HMM Bridge)

The enriched timeline is the bridge between rule-based survival classification and statistical regime detection. It takes the 6-mode survival state and crosses it with the 4-state HMM market regime (bull/bear/high_vol/low_vol), producing 30 possible combinations mapped to 11 named states with continuous intensity scores:

```
(normal, bull) -> stable_growth, intensity=0.0
(company_only, bear) -> company_distress_severe, intensity=0.70
(both_unprotected, bear) -> extreme_crisis, intensity=1.00
```

**Why this matters**: The hierarchy weight system uses these states to dynamically rebalance the 5-tier priority:
- In `extreme_crisis`: Tier 1 (liquidity) gets 60% weight, Tier 5 (growth) gets 0%
- In `stable_growth`: all tiers get 20% equally

This means that in a crisis, the pipeline focuses its predictive capacity on the variables most relevant to survival (cash position, debt coverage), while in normal conditions it gives equal attention to growth metrics and valuation.

---

## Cross-Cutting Architectural Patterns

### 1. Graceful Degradation

Every middleware module wraps its core logic in try/except. When a module fails:
- It logs a warning (not error)
- It returns a safe default (empty DataFrame, NaN series, unchanged cache)
- The pipeline continues

This means a partial failure in, say, the MNAR estimator doesn't prevent the pipeline from producing a report. The report just notes "estimation unavailable for N variables."

### 2. Dual-Path Redundancy

Several capabilities have primary + fallback implementations:
- OHLCV: regional wrapper -> yfinance fallback
- Macro: per-region API -> wbgapi (World Bank) fallback -> IMF SDMX fallback
- Filing extraction: LLM extractor -> fuzzy PDF parser (no LLM)
- Profile enrichment: PIT client -> supplement provider (OpenFIGI + regional)
- Estimation: split classifier (MICE/GP/Heckman) -> BayesianRidge rolling -> mean fill

### 3. The Confidence Propagation Chain

Confidence scores flow through the entire middleware stack:

```
Interpolator confidence (distance from filing)
    -> Estimator confidence (model agreement)
    -> Combined: final confidence = min(interpolator, estimator)
    -> Consumed by: forecasting (sample weights), prediction aggregator (interval width)
```

This means a value that was interpolated far from a filing AND estimated by a disagreeing ensemble gets very low confidence, which causes the forecasting models to give it less weight and the prediction intervals to widen.

### 4. The is_missing Flag System

Every column `X` in the cache has a companion `is_missing_X` flag (1 = null, 0 = present). This seems redundant with `pd.isna()` checks, but it serves two purposes:
1. **Audit trail**: After estimation fills a NaN, `is_missing_X` still shows 1 while the value is now filled. The `{X}_source` column distinguishes observed vs estimated.
2. **Feature engineering**: Models can use the missingness pattern itself as a feature. A company that never reports `interest_expense` is likely different from one that reports it every quarter.

---

## Structural Tensions and Trade-offs

### 1. Two Cache Builders

The legacy `cache_builder.py` (816 lines) and the inline builder in `main.py` (~110 lines) do similar things but the inline version has frequency interpolation. The legacy module is kept because:
- It defines `EntityData`, `ExtractionResult`, `STATEMENT_FIELDS` types used elsewhere
- It has `enrich_cache_with_indicators()` for post-cache enrichment
- It has the `LookAheadError` exception class

This is a maintenance risk: changes to the cache construction logic need to be kept in sync between two locations. The inline builder is the one that actually runs.

### 2. Translation Happens at Multiple Levels

Concept name normalization happens in at least 3 places:
1. `canonical_translator.py` -- XBRL/regional concept -> canonical name (primary)
2. `data_reconciliation.py` -- camelCase API field -> canonical name (cleanup)
3. `cache_builder.py` `_build_column_rename_map()` -- client output names -> canonical names

These layers are not redundant -- each catches a different class of naming inconsistency -- but it means a new field name needs to be added in up to 3 places.

### 3. Report_date vs Filing_date Semantics

The merge uses `report_date` but PIT compliance requires `filing_date`. The current design works because `report_date <= filing_date` is enforced by the reconciliation layer. However, this means the pipeline technically makes data "available" on the report_date rather than the filing_date. For backtesting purposes, this introduces a slight look-ahead bias (days between report_date and filing_date). The filing_date column IS preserved in the wide format and could be used for stricter PIT alignment in a future version.

### 4. Estimation Confidence vs Interpolation Confidence

Both the interpolator and the estimator produce confidence scores, but they measure different things:
- Interpolation confidence: "how reliable is this daily value given its distance from the nearest actual filing?"
- Estimation confidence: "how reliable is this imputed value given model agreement?"

These are not combined in a principled way. The estimator stores its own confidence alongside the interpolator's confidence, but downstream models receive both without a unified metric. Models that consume confidence (like the prediction aggregator) use whichever is available.

---

## Performance Characteristics

| Stage | Typical Latency | Bottleneck |
|-------|----------------|-----------|
| Canonical translation | < 1s | String matching (in-memory) |
| Data reconciliation | < 1s | Date parsing |
| Cache construction | 1-3s | Index operations, ffill |
| Frequency interpolation | 1-2s | Per-column linear interp |
| Derived variables | 1-3s | ~40 rolling window computations |
| Estimator Phase 1 | < 1s | Simple arithmetic |
| Estimator Phase 2 (classification) | 1-2s | Coverage heuristics |
| Estimator Phase 3 (MAR) | 5-30s | MICE iterative imputation (25 rounds) |
| Estimator Phase 3 (MNAR) | 5-30s | Heckman two-stage regression |
| Survival timeline | < 1s | Vectorized mode classification |
| **Total middleware** | **15-70s** | **Estimation Phase 3 dominates** |

The estimation phase is the bottleneck because it trains per-variable models. For a company with all 22 estimable variables partially missing, it may fit 22 x 3 models (MICE + GP + Matrix Completion). The GAIN imputer (if torch is available) adds another 5-10s per variable.

---

## Conclusion

The middleware layer is architecturally sound. Its core strength is the convergence contract: regardless of data source, everything exits the translator as canonical wide-format DataFrames, and everything exits the cache builder as a daily-indexed DataFrame with is_missing flags. This abstraction allows the 25+ temporal models to be completely source-agnostic.

The main areas of structural complexity are the estimation engine (which is unavoidably complex given the MAR/MNAR distinction) and the triple-layer name normalization (which is pragmatic but creates maintenance surface area). Neither is a blocking issue -- both are consequences of supporting 25 markets with 8 different accounting standards.
