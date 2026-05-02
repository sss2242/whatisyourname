# Frequency Separator Upgrade Plan

*2026-05-02*

## Current State

[`operator1/clients/frequency_separator.py`](operator1/clients/frequency_separator.py) (415 lines) handles the critical problem of mixed-frequency financial statement data. PIT clients return annual and quarterly filings in a single DataFrame; naively merging them produces inflated ratios (e.g., annual revenue / quarterly COGS = 300%+ gross margin).

**Current approach:**
1. **Separation:** Split by `period_type` column or infer from date gaps (<120d = quarterly, 120-250d = semi-annual, 250-500d = annual)
2. **Backfill:** Missing columns in high-freq DF filled from low-freq using revenue-ratio scaling for flow variables, direct copy for stock variables
3. **Priority:** Quarterly > semi-annual > annual

**Called from:** [`main.py:838`](main.py:838) and [`backtest_runner.py:230`](backtest_runner.py:230)

## Current Weaknesses

| # | Weakness | Impact |
|---|----------|--------|
| W1 | **Gap-based frequency detection uses fixed thresholds** (120d, 250d, 500d). Fiscal calendars vary: Japan uses April fiscal years (gaps shift by 1-2 months), India has March fiscal years, some EU companies report Q1/H1/Q3/FY (irregular). | Misclassification of semi-annual as quarterly (or vice versa) for non-calendar fiscal years |
| W2 | **Revenue-ratio scaling assumes uniform seasonality.** `Q4_gross_profit = annual_gross_profit * (Q4_revenue / annual_revenue)` works for companies with proportional cost structures but fails for seasonal businesses (retail Q4 has 40% of revenue but 25% of SGA). | Incorrect quarterly estimates for seasonal companies (retail, agriculture, tourism) |
| W3 | **No handling of amended/restated filings.** If a company restates Q2 and the PIT client returns both the original and amended filing, both end up in the same frequency group with different values for the same period. | Duplicate rows with conflicting values |
| W4 | **No detection of fiscal year changes.** When a company changes its fiscal year end (e.g., from December to September), date gaps become irregular and the gap classifier produces wrong labels. | Entire year's data potentially misclassified |
| W5 | **`_merge_column` iterates row-by-row** with `target.at[idx, col]`. For DataFrames with hundreds of rows and dozens of columns, this is O(n*m) with Python loop overhead. | Performance drag on large datasets |
| W6 | **No cross-validation of flow variable totals.** After backfilling, Q1+Q2+Q3+Q4 revenue should approximate annual revenue. No check exists. | Silent errors from bad scaling |
| W7 | **Binary stock/flow classification misses semi-flow variables.** EPS is flow-like (earned over period) but not a simple total (it's an average). Shares outstanding is stock but can change mid-quarter (buybacks). | Incorrect interpolation for edge-case variables |

## Upgrade Methods (from multiple expert domains)

### Method 1: Bayesian Frequency Detection (Econometrics)

**Source:** Geweke (1977) "The Dynamic Factor Analysis of Economic Time Series", adapted for filing date analysis.

**Instead of:** Fixed gap thresholds (current: <120d = quarterly)

**Do:** Compute posterior probability of each frequency class given observed date gaps using Bayesian inference with informative priors from the market's filing regime.

```
P(freq=Q | gaps) proportional to P(gaps | freq=Q) * P(freq=Q | market_id)
```

Where:
- `P(gaps | freq=Q)` = likelihood using Gaussian around expected gap (90 +/- 15 days for quarterly)
- `P(freq=Q | market_id)` = market-specific prior (US = 0.8 quarterly, EU ESEF = 0.3 quarterly / 0.4 annual)

**Advantage:** Handles irregular fiscal calendars (Japan April FY, India March FY) and fiscal year changes gracefully. The prior encodes market knowledge; the likelihood handles company-specific irregularities.

**Config-driven:** Priors stored in `config/frequency_separator.yml` per market.

### Method 2: Temporal Disaggregation (National Accounts / Econometrics)

**Source:** Chow-Lin (1971), Denton (1971), Litterman (1983) -- standard methods used by national statistics agencies (BEA, Eurostat, ONS) to convert quarterly GDP to monthly estimates.

**Instead of:** Simple revenue-ratio scaling (`Q4_gp = annual_gp * Q4_rev/annual_rev`)

**Do:** Use the Chow-Lin temporal disaggregation method which finds the quarterly series that:
1. Sums to the annual total (additive constraint)
2. Has maximum correlation with a related indicator (revenue as the indicator for gross_profit, operating_income, etc.)
3. Minimizes the deviation from a smooth trajectory

```
minimize  sum((q_t - q_{t-1})^2)  [Denton smoothness]
subject to  sum(q_i for i in year) = annual_total  [aggregation constraint]
correlate with  revenue_quarterly  [Chow-Lin indicator]
```

**Advantage:** Produces quarterly estimates that are internally consistent (they sum to the annual total), smooth (no jumps), and informed by the seasonal pattern of the indicator variable (revenue). This is exactly what national accounts economists do when they need to produce monthly GDP from quarterly data.

**Implementation:** `scipy.optimize.minimize` with linear equality constraints. The Chow-Lin formula has a closed-form solution (GLS regression with AR(1) residuals).

### Method 3: Denton Proportional Benchmarking (Statistics Canada)

**Source:** Denton (1971) "Adjustment of Monthly or Quarterly Series to Annual Totals" -- the gold standard for statistical agencies.

**Instead of:** Direct copy for stock variables

**Do:** Even stock variables benefit from benchmarking. The Denton proportional method adjusts a preliminary quarterly series (from interpolation) so that:
1. Annual averages match annual reports exactly
2. Quarter-to-quarter movements are as smooth as possible

```
minimize  sum((q_t/p_t - q_{t-1}/p_{t-1})^2)  [proportional first differences]
subject to  mean(q_i for i in year) = annual_value
```

Where `p_t` is the preliminary (interpolated) series.

**Advantage:** Better handling of shares_outstanding changes (buybacks create mid-quarter jumps that linear interpolation misses). The preliminary series from frequency_interpolator.py provides the `p_t`.

### Method 4: Expectation-Maximization for Mixed Frequencies (Signal Processing)

**Source:** Shumway & Stoffer (2000) "Time Series Analysis and Its Applications" -- EM algorithm for incomplete/mixed-frequency time series.

**Instead of:** Treating each frequency independently then merging

**Do:** Use an EM algorithm that jointly models all frequencies as observations of the same underlying latent process at different sampling rates:

- **E-step:** Given current parameter estimates, compute the expected value of missing quarterly observations using the Kalman smoother
- **M-step:** Given the completed data, update the AR(1) process parameters

**Advantage:** Produces maximum-likelihood estimates that optimally combine information from all available frequencies. A quarterly filing + annual filing for the same company year are not independent -- the EM approach respects their joint distribution.

**Implementation:** Uses the existing Kalman filter infrastructure from `operator1/models/forecasting.py` (already has `fit_kalman`).

### Method 5: Cross-Validation Reconciliation (Audit / Accounting)

**Source:** Standard audit practice -- "analytical review procedures" (ISA 520, AU-C 520).

**Instead of:** No post-backfill validation

**Do:** After backfilling, apply accounting identity cross-checks:

```python
# Flow variable reconciliation
for year in fiscal_years:
    quarterly_sum = sum(Q1, Q2, Q3, Q4)
    annual_total = annual_report_value
    discrepancy = abs(quarterly_sum - annual_total) / abs(annual_total)
    if discrepancy > 0.05:  # 5% tolerance
        # Adjust quarters proportionally (pro-rata)
        adjustment = annual_total / quarterly_sum
        Q1, Q2, Q3, Q4 = [q * adjustment for q in quarters]

# Cross-statement consistency
assert total_assets == total_liabilities + total_equity  (within tolerance)
assert free_cash_flow == operating_cf - abs(capex)  (within tolerance)
```

**Advantage:** Catches scaling errors before they propagate to derived variables and downstream models. Pure audit logic -- no statistical assumptions.

### Method 6: Fiscal Calendar Registry (Domain Engineering)

**Source:** Exchange filing calendars (SEC, DART, MOPS, etc.) -- codified in a config file.

**Instead of:** Inferring fiscal year end from date gaps

**Do:** Maintain a registry of known fiscal calendar patterns per market:

```yaml
# config/frequency_separator.yml
fiscal_calendars:
  US:
    common_fy_ends: ["12-31", "09-30", "06-30", "03-31"]
    default_filing_freq: quarterly
    typical_filing_lag_days: 40  # 10-K filed ~40 days after FY end
  JP:
    common_fy_ends: ["03-31"]
    default_filing_freq: quarterly
    typical_filing_lag_days: 45
  UK:
    common_fy_ends: ["12-31", "03-31", "06-30"]
    default_filing_freq: semiannual
    typical_filing_lag_days: 120
```

Use the company's actual fiscal year end (from profile data) to anchor the classification rather than relying solely on date gaps.

**Advantage:** Eliminates misclassification for known markets. Combined with Bayesian detection for unknown patterns.

### Method 7: Amendment Deduplication (Database Engineering)

**Source:** Standard SCD (Slowly Changing Dimension) Type 2 handling from data warehousing.

**Instead of:** Keeping both original and amended filings

**Do:** For each (company, report_date, statement_type) tuple, keep only the filing with the latest `filing_date`:

```python
# Deduplicate by report_date, keeping latest amendment
stmt_df = stmt_df.sort_values("filing_date").drop_duplicates(
    subset=["report_date"], keep="last"
)
```

**Advantage:** Simple, correct, and prevents the root cause of duplicate-row confusion. Already partially implemented in `data_reconciliation.py` but not in frequency_separator.

## Implementation Plan

| # | Step | Method | Complexity | Impact |
|---|------|--------|------------|--------|
| 1 | Create `config/frequency_separator.yml` with fiscal calendar registry + Bayesian priors | M6, M1 | Low | Fixes W1, W4 |
| 2 | Replace `_classify_gap` with Bayesian frequency detection using market priors + company FY end | M1 | Medium | Fixes W1, W4 |
| 3 | Add amendment deduplication before separation | M7 | Low | Fixes W3 |
| 4 | Replace revenue-ratio scaling with Chow-Lin temporal disaggregation | M2 | Medium | Fixes W2 |
| 5 | Add Denton benchmarking for stock variables | M3 | Medium | Fixes W7 |
| 6 | Add cross-validation reconciliation after backfill | M5 | Low | Fixes W6 |
| 7 | Vectorize `_merge_column` using pd.merge_asof | N/A | Low | Fixes W5 |
| 8 | Add semi-flow variable category (EPS, per-share metrics) | N/A | Low | Fixes W7 |

## Files to Modify

| File | Change |
|------|--------|
| `config/frequency_separator.yml` | NEW -- fiscal calendar registry, Bayesian priors, tolerance thresholds |
| `operator1/clients/frequency_separator.py` | All 7 methods applied to existing functions |
| `operator1/estimation/frequency_interpolator.py` | Add `SEMI_FLOW_VARIABLES` classification, Denton benchmarking |

## Dependency Notes

- Chow-Lin and Denton: `scipy.optimize.minimize` (already installed)
- Bayesian detection: `numpy` only (already installed)
- Kalman EM: reuse existing `filterpy` / custom Kalman (already in codebase)
- No new pip dependencies needed
