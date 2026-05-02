# Frequency Separator: GitHub Research Findings

*2026-05-02 -- Detailed implementation patterns from open-source projects*

For each of the 7 upgrade methods proposed in the frequency separator plan, I researched existing GitHub implementations and examined their source code for patterns we can adopt.

---

## Method 1: Bayesian Frequency Detection

### Repo: hildensia/bayesian_changepoint_detection

**URL:** https://github.com/hildensia/bayesian_changepoint_detection
**Stars:** ~1K | **Language:** Python + PyTorch

**What they do:** Online Bayesian changepoint detection (Adams & MacKay 2007). Detects when a time series changes its generating distribution -- exactly what we need to detect when filing dates shift from quarterly to semi-annual.

**Key implementation patterns found:**

1. **Hazard function design** (`hazard_functions.py`): Uses a `constant_hazard(lam, r)` function where `lam` is the expected run length. For our case: `lam` would be the expected number of filings between frequency changes (e.g., `lam=8` means we expect ~8 filings before a frequency change).

2. **Student-t likelihood** (`online_likelihoods.py`): Uses Normal-Gamma conjugate prior for unknown mean and variance. The predictive distribution is Student-t. For filing date gaps, our "data" is the gap in days between consecutive filings. The model learns the mean gap (e.g., ~90 for quarterly) and flags when it shifts to ~180 (semi-annual).

3. **Prior-based classification**: The `priors.py` module provides `const_prior(t, p)` -- a constant prior probability for changepoints. For our use case, we'd use market-specific priors instead:
   ```python
   # US: 80% chance of quarterly filing regime
   # EU: 40% quarterly, 40% annual, 20% semi-annual
   # JP: 90% quarterly (TSE requires quarterly)
   ```

**What we should adopt:**
- The constant hazard function is the right model for filing frequency changes (changes are rare, ~1 per company lifetime)
- The Student-t likelihood handles the natural variability in filing gap lengths (Q4 might be 95 days, Q1 might be 88 days)
- We do NOT need their PyTorch/GPU infrastructure -- our data is tiny (8-20 filing dates per company). A pure numpy implementation is sufficient

**Adaptation for our use case:**
```python
# Instead of changepoint detection on a time series,
# we compute posterior probability of each frequency class:
# P(freq=Q | gaps) = P(gaps | freq=Q) * P(freq=Q | market)
# where P(gaps | freq=Q) = product of Gaussian(gap_i; mu=90, sigma=15)
# This is simpler than full BOCD but uses the same Bayesian framework
```

---

## Method 2: Chow-Lin Temporal Disaggregation

### Repo: jaimevera1107/tempdisagg

**URL:** https://github.com/jaimevera1107/tempdisagg
**Version:** 0.2.13 | **License:** MIT | **Language:** Python
**Already installed in our environment**

**What they do:** Full implementation of 8 temporal disaggregation methods: OLS, Denton, Denton-Cholette, Chow-Lin (4 variants), Litterman (2 variants), Fernandez, and a fast approximation.

**Key implementation patterns found:**

1. **Conversion matrix builder** (`preprocessing/conversion_matrix_builder.py`): Builds the C matrix that maps high-frequency to low-frequency observations. Supports `"sum"` (flow variables), `"average"`, `"first"`, and `"last"` (stock variables). This is exactly our stock/flow distinction.

2. **Chow-Lin with rho optimization** (`model/models_handler.py`): The `chow_lin_opt_estimation()` method uses `scipy.optimize.minimize_scalar` to find the optimal AR(1) autocorrelation parameter `rho` that maximizes the log-likelihood. The formula:
   ```
   Sigma = rho^|i-j| (AR(1) covariance)
   Q = C @ Sigma @ C'
   beta = (X'C'Q^{-1}CX)^{-1} X'C'Q^{-1} y_l
   y_hat = X*beta + Sigma*C'*Q^{-1}*(y_l - C*X*beta)
   ```

3. **Retropolarizer** (`TempDisaggModel` constructor has `use_retropolarizer` option): When the indicator variable (revenue for us) has missing values at the start of the series, the retropolarizer back-extrapolates using linear regression. This handles the case where quarterly revenue starts later than annual data.

4. **Post-estimation adjustment** (`model/tempdisagg_adjuster.py`): After disaggregation, ensures non-negativity and consistency with the aggregation method. This is the reconciliation step we need.

**What we should adopt:**
- Use `tempdisagg` directly as a dependency (already installed, MIT license, 47KB wheel, scipy/numpy/pandas only)
- The `TempDisaggModel(method="chow-lin-opt", conversion="sum")` call does exactly what we need for flow variables
- For stock variables, use `conversion="last"` (balance sheet = last value in period)
- The rho optimizer (`RhoOptimizer`) finds the best AR(1) parameter automatically
- The adjuster ensures non-negative outputs (revenue can't be negative)

**Integration pattern:**
```python
from tempdisagg import TempDisaggModel

# Disaggregate annual gross_profit to quarterly using quarterly revenue as indicator
model = TempDisaggModel(
    method="chow-lin-opt",
    conversion="sum",  # flow variable
    grain_col="quarter",  # high-freq identifier
    index_col="year",  # low-freq identifier
    y_col="gross_profit",  # target to disaggregate
    X_col="revenue",  # indicator variable
)
model.fit(df)
quarterly_gp = model.predict()
```

---

## Method 3: Denton Proportional Benchmarking

### Repo: jaimevera1107/tempdisagg (same as above)

**Key implementation patterns found:**

1. **Denton estimation** (`models_handler.py:82-118`): Core formula using differencing matrix D:
   ```python
   D = I - shift_matrix  # first-difference operator
   D_h = D^h  # h-th order differencing
   Sigma_D = (D_h' D_h)^{-1}  # smoothness covariance
   Q = C @ Sigma_D @ C'
   y_hat = X*beta + Sigma_D*C'*Q^{-1}*(y_l - C*X*beta)
   ```
   The key insight: the Denton penalty minimizes the first differences of the disaggregated series relative to the indicator, producing the smoothest possible series that still sums to the annual total.

2. **Denton-Cholette variant** (`models_handler.py:365-437`): An improvement over basic Denton. Uses a base series (our interpolated values from frequency_interpolator.py) and adjusts it minimally to match annual totals:
   ```python
   P = W + D'D  # penalty matrix (weights + smoothness)
   lambda = solve(C @ P^{-1} @ C', y_l - C @ base_series)
   y_hat = base_series + P^{-1} @ C' @ lambda
   ```
   This is better for us because we already have a preliminary series from the frequency interpolator -- Denton-Cholette adjusts it rather than building from scratch.

**What we should adopt:**
- Use `TempDisaggModel(method="denton-cholette")` for stock variables where we have a preliminary interpolation
- Use `TempDisaggModel(method="fast")` as fallback (fast approximation of Denton-Cholette)
- Pass our frequency_interpolator output as the `base_series` parameter

---

## Method 4: EM for Mixed Frequencies

### Built-in: statsmodels.tsa.statespace.dynamic_factor_mq

**URL:** https://www.statsmodels.org/stable/generated/statsmodels.tsa.statespace.dynamic_factor_mq.DynamicFactorMQ.html
**Already installed in our environment** (statsmodels 0.14.6)

**What it does:** `DynamicFactorMQ` (Mixed-frequency Quarterly) is a state-space model that jointly estimates factors from mixed-frequency data. It handles monthly + quarterly data natively using the Kalman filter/smoother.

**Key implementation patterns found:**

1. **Constructor accepts mixed-frequency data directly:**
   ```python
   DynamicFactorMQ(
       endog=monthly_data,        # monthly observations
       endog_quarterly=quarterly_data,  # quarterly observations
       k_endog_monthly=n_monthly_vars,
       factors=1,                 # number of latent factors
       factor_orders=1,           # AR order for factor
       idiosyncratic_ar1=True,    # AR(1) for each variable's error
   )
   ```

2. **Quarterly variables treated as latent monthly stocks:** The model internally converts quarterly observations to monthly using a state-space formulation where the quarterly value is the sum (or average) of 3 monthly values. The Kalman smoother fills in the unobserved months.

3. **Standardization built-in:** `standardize=True` handles the scale differences between variables automatically.

**What we should adopt:**
- For companies with both annual and quarterly data: use DynamicFactorMQ with quarterly as "monthly" and annual as "quarterly" (3:1 ratio maps to 4:1 ratio). The math is identical -- the model just needs the aggregation ratio.
- The Kalman smoother produces maximum-likelihood estimates of the missing quarters
- This replaces our simple revenue-ratio scaling with a statistically optimal approach

**However:** DynamicFactorMQ is designed for macro nowcasting (mixing monthly GDP indicators with quarterly GDP). For financial statement disaggregation, the simpler tempdisagg Chow-Lin is more appropriate because:
- We have a clear indicator variable (revenue)
- We know the aggregation relationship (sum for flows, last for stocks)
- DynamicFactorMQ adds unnecessary complexity for our 2-variable case

**Recommendation:** Use DynamicFactorMQ only as a fallback when no indicator variable is available (no revenue at high frequency). Primary path: tempdisagg Chow-Lin.

---

## Method 5: Cross-Validation Reconciliation

### Standard: Audit ISA 520 / AU-C 520

No single GitHub repo implements this because it's domain-specific accounting logic. The patterns come from audit practice:

**Key patterns from financial data validation libraries:**

1. **Accounting identity checks** (from our own `operator1/estimation/estimator.py` Phase 1):
   ```python
   # Already implemented in estimator.py:
   total_assets == total_liabilities + total_equity
   free_cash_flow == operating_cash_flow - abs(capex)
   net_debt == total_debt - cash_and_equivalents
   ```

2. **Temporal consistency checks** (new for frequency_separator):
   ```python
   # After disaggregation, verify:
   sum(Q1, Q2, Q3, Q4) == annual_total  # within 5% tolerance
   # For each variable, across all fiscal years
   ```

3. **Cross-statement consistency** (new):
   ```python
   # Balance sheet change should match cash flow:
   delta_cash == operating_cf + investing_cf + financing_cf  # within tolerance
   ```

**What we should adopt:**
- Post-disaggregation validation layer that checks flow variable sums match annual totals
- Pro-rata adjustment when sum deviates by more than 5%: `Q_adjusted = Q * (annual / sum_of_quarters)`
- Log warnings when deviations exceed 10% (likely a data quality issue, not a disaggregation issue)

---

## Method 6: Fiscal Calendar Registry

### Repo: quantopian/exchange_calendars (now maintained as exchange-calendars)

**URL:** https://github.com/gerrymanoim/exchange_calendars
**Already installed in our environment** (exchange_calendars package, 102 exchanges)

**What it does:** Provides trading calendars for 102 exchanges worldwide. Handles holidays, half-days, early closes, and ad-hoc closures.

**Key patterns found:**

1. **Per-exchange calendar access:**
   ```python
   import exchange_calendars as ec
   nyse = ec.get_calendar("XNYS")  # NYSE
   tse = ec.get_calendar("XTKS")   # Tokyo Stock Exchange
   lse = ec.get_calendar("XLON")   # London Stock Exchange
   ```

2. **Trading day count between dates** (useful for period classification):
   ```python
   sessions = nyse.sessions_in_range("2024-01-01", "2024-03-31")
   # 62 trading days in Q1 2024 (vs 90 calendar days)
   ```

3. **Market-to-exchange mapping** (we need to build this):
   ```python
   MARKET_TO_EXCHANGE = {
       "us_sec_edgar": "XNYS",
       "uk_companies_house": "XLON",
       "jp_jquants": "XTKS",
       "kr_dart": "XKRX",
       "tw_mops": "XTAI",
       "br_cvm": "BVMF",
       # ... 25 markets -> exchanges
   }
   ```

**What we should adopt:**
- Use exchange_calendars to count trading days between filing dates (more accurate than calendar days for gap classification)
- Build a `config/frequency_separator.yml` mapping market_id -> exchange + common fiscal year ends + filing frequency priors
- Use the company's actual fiscal year end from profile data (already available in `target_profile["fiscal_year_end"]`)

**Integration with Bayesian detection:**
```python
# Gap classification using trading days instead of calendar days:
# 58-67 trading days = quarterly (vs 85-95 calendar days)
# 120-130 trading days = semi-annual
# 245-255 trading days = annual
# Trading day counts are more stable across markets
```

---

## Method 7: Amendment Deduplication

### Standard: SCD Type 2 (Data Warehousing)

This is a standard database pattern. No specialized repo needed.

**Key patterns from data engineering:**

1. **Simple dedup (our primary need):**
   ```python
   # For each report_date, keep only the latest filing_date
   stmt_df = stmt_df.sort_values("filing_date").drop_duplicates(
       subset=["report_date"], keep="last"
   )
   ```

2. **Amendment tracking** (for audit trail):
   ```python
   # Track which filings were amendments
   stmt_df["is_amendment"] = stmt_df.duplicated(subset=["report_date"], keep="first")
   # Count amendments per period
   n_amendments = stmt_df.groupby("report_date")["is_amendment"].sum()
   ```

3. **Value change detection** (for quality flagging):
   ```python
   # When an amendment changes a value by >10%, flag it
   # This indicates a material restatement
   for report_date, group in stmt_df.groupby("report_date"):
       if len(group) > 1:
           original = group.iloc[0]
           amended = group.iloc[-1]
           for col in numeric_cols:
               pct_change = abs(amended[col] - original[col]) / abs(original[col])
               if pct_change > 0.10:
                   logger.warning("Material restatement: %s changed %d%% on %s",
                                  col, pct_change * 100, report_date)
   ```

**What we should adopt:**
- Simple dedup before frequency separation (keep latest filing_date per report_date)
- Material restatement detection (>10% change = warning in profile)
- Already partially implemented in `data_reconciliation.py` -- just needs to be called earlier in the pipeline (before frequency_separator runs)

---

## Summary: What to Use and What to Build

| Method | Best Implementation | Use As-Is? | Notes |
|--------|-------------------|------------|-------|
| M1: Bayesian freq detection | Custom (numpy only) | Build | Adapt hazard function pattern from hildensia/bayesian_changepoint_detection |
| M2: Chow-Lin | `tempdisagg` library | Yes | Already installed, MIT, 47KB. Use `TempDisaggModel(method="chow-lin-opt", conversion="sum")` |
| M3: Denton benchmarking | `tempdisagg` library | Yes | Use `TempDisaggModel(method="denton-cholette")` with frequency_interpolator output as base_series |
| M4: EM mixed-frequency | `statsmodels.DynamicFactorMQ` | Fallback only | Too complex for our case. Use only when no indicator variable available |
| M5: Reconciliation | Custom | Build | Simple sum checks + pro-rata adjustment + accounting identity validation |
| M6: Fiscal calendar | `exchange_calendars` | Yes | Already installed, 102 exchanges. Need market_id -> exchange mapping config |
| M7: Amendment dedup | Standard pandas | Build | 3-line dedup + material restatement detection |

### New dependency: `tempdisagg==0.2.13`
- Already installed in this environment
- MIT license, pure Python, 47KB wheel
- Dependencies: numpy, pandas, scipy, matplotlib, scikit-learn (all already installed)
- Should be added to `requirements/stage2-ml.txt`
