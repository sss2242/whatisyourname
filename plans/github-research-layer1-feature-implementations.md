# GitHub Research: Layer 1 Feature Implementation Findings

*Researched 2026-05-01 -- GitHub code search + direct source review of 15+ repositories*

For each of the 47 proposed features in the Layer 1 Expert Methods Update Plan, this document catalogs the best open-source implementation found on GitHub, highlights useful patterns that would benefit our codebase, and flags pitfalls discovered in existing implementations.

---

## Domain 1: Market Microstructure (5 features)

### Feature 1: Corwin-Schultz Bid-Ask Spread Estimator

**Best implementation found:** [`RiskLabAI/RiskLabAI.py`](https://github.com/RiskLabAI/RiskLabAI.py) (107 stars)
- File: `RiskLabAI/features/microstructural_features/corwin_schultz.py`
- Full working implementation (unlike `mlfinlab` which has stubs only behind paywall)
- Clean decomposition into `beta_estimates()`, `gamma_estimates()`, `alpha_estimates()`, then final `corwin_schultz_spread()`

**Also found in:** [`mgao6767/frds`](https://github.com/mgao6767/frds) (101 stars) -- `src/frds/measures/spread.py` has quoted, effective, and realized spread estimators but NOT Corwin-Schultz (focuses on tick-data spreads)

**Also found in:** [`OpenSourceAP/CrossSection`](https://github.com/OpenSourceAP/CrossSection) (963 stars) -- SAS implementation in `Signals/pyCode/PrepScripts/corwin_schultz_edit.sas` (Chen & Zimmermann 2020 replication)

**Useful patterns for us:**
1. RiskLabAI decomposes the formula into 3 clear steps matching the paper's notation (beta, gamma, alpha). We should follow this structure.
2. The constant `_DENOMINATOR = 3 - 2*sqrt(2)` is precomputed at module level -- avoids repeated computation inside the rolling loop.
3. Alpha is floored at 0 (negative spread is economically meaningless). This is critical and easy to forget.
4. Beta uses `rolling(window=2).sum()` then `rolling(window=window_span).mean()` -- double rolling pattern that is more numerically stable than a single-pass approach.

**Pitfalls found:**
- The `mlfinlab` implementation is pay-walled (all functions return `pass`). Don't use it.
- Some implementations forget the floor on alpha, producing negative spreads which cause downstream NaN cascades.
- The window_span for beta averaging should match our adaptive windows (not hardcoded to 20).

---

### Feature 2: Kyle's Lambda (Price Impact)

**Best implementation found:** [`mgao6767/frds`](https://github.com/mgao6767/frds) (101 stars)
- File: `src/frds/measures/_kyle_lambda.py`
- Extremely clean: 15 lines, OLS via `np.linalg.lstsq`
- Returns lambda * 1,000,000 (standard scaling for interpretation)

**Key insight from implementation:**
```python
def kyle_lambda(returns, signed_dollar_volume):
    y = np.asarray(returns)
    x = np.asarray(signed_dollar_volume)[:, np.newaxis]
    a, _, _, _ = np.linalg.lstsq(x, y, rcond=None)
    return a[0] * 1_000_000
```

**Useful patterns for us:**
1. The original Kyle (1985) requires signed order flow (buy vs sell). With daily data only, we must approximate: `signed_volume = volume * sign(return_1d)` -- this is the standard daily proxy (Hasbrouck 2009).
2. FRDS uses `lstsq` instead of `statsmodels.OLS` -- 10x faster for simple univariate regression with no intercept.
3. The 1M scaling is conventional in the literature -- keeps the number in a readable range.
4. We should compute this as a ROLLING regression (21d window) to get a daily time series, not a single scalar.

**Pitfalls found:**
- `pymicrostructure` (BBieganowski) implements Kyle Lambda for simulated LOB data -- not useful for our daily OHLCV context.
- Without signed volume approximation, this feature is meaningless. The sign(return) proxy is imperfect but standard.

---

### Feature 3: Parkinson Volatility

**Best implementation found:** [`scikit-portfolio/scikit-portfolio`](https://github.com/scikit-portfolio/scikit-portfolio) (58 stars)
- File: `skportfolio/metrics/volatiltiy.py`
- Clean, well-documented implementation with formula in docstring

**Also found in:** [`Jensenberg/volatility-and-option`](https://github.com/Jensenberg/volatility-and-option) (125 stars)
- File: `volatilty measurements/volatility.py` -- contains ALL 6 volatility estimators (realized, Parkinson, Garman-Klass, Rogers-Satchell, GK-Yang-Zhang, Yang-Zhang) in a single file

**Key implementation pattern (from Jensenberg):**
```python
def parkinson(high, low, N=240):
    sum_hl = sum(log(H_t / L_t) ** 2 for H_t, L_t in zip(high, low))
    return sqrt(sum_hl * N / (4 * len(high) * log(2)))
```

**Useful patterns for us:**
1. Jensenberg's file is the single best reference for ALL volatility estimators -- all 6 methods in 120 lines with consistent API. We should model our implementation on this pattern.
2. The annualization factor N=240 (Chinese market) should be parameterized -- our pipeline covers 25 markets with different trading day counts (252 for US, 240 for CN, ~247 for EU, etc.).
3. scikit-portfolio's docstring explains WHY Parkinson is better: "uses high and low price of the day... close to close prices could show little difference while large price movements could have happened during the day."
4. For rolling computation, we need to vectorize these -- the generator expressions above are for single-window calculation. Use pandas `.rolling().apply()` or vectorized numpy.

**Pitfalls found:**
- Parkinson systematically underestimates volatility because it doesn't account for overnight gaps. That's exactly why we also need Yang-Zhang (Feature 4).
- Some implementations forget the `4 * n * ln(2)` denominator, producing volatility estimates that are off by a constant factor.

---

### Feature 4: Yang-Zhang Volatility

**Best implementation found:** [`Jensenberg/volatility-and-option`](https://github.com/Jensenberg/volatility-and-option) (125 stars)
- Complete Yang-Zhang implementation that composes Rogers-Satchell as a sub-component
- Key insight: `k = 0.34 / (1.34 + (n+1)/(n-1))` -- the optimal weighting factor between overnight, open-to-close, and Rogers-Satchell components

**Also found in:** [`Acelogic/Earnings-Volatility-Calculator`](https://github.com/Acelogic/Earnings-Volatility-Calculator) (79 stars)
- Uses Yang-Zhang specifically for earnings event volatility analysis

**Key implementation pattern:**
```python
def yang_zhang(open, high, low, close, N=240):
    # Overnight returns: open_t / close_{t-1}
    oc = [log(O_t / C_prev) for O_t, C_prev in zip(open[1:], close[:-1])]
    oc_var = var(oc) * N
    
    # Close-to-open returns: close_t / open_t
    co = [log(C_t / O_t) for O_t, C_t in zip(open[1:], close[1:])]
    co_var = var(co) * N
    
    # Rogers-Satchell component (intraday range-based)
    rs_var = roger_satchell(open, high, low, close) ** 2
    
    # Optimal blend: k balances open-close vs overnight vs intraday
    k = 0.34 / (1.34 + (n + 1) / (n - 1))
    return sqrt(oc_var + k * co_var + (1 - k) * rs_var)
```

**Useful patterns for us:**
1. Yang-Zhang decomposes into 3 variance components: overnight (open/prev_close), open-to-close (close/open), and Rogers-Satchell (intraday high-low). Each component tells a different story.
2. The optimal k comes from the paper itself -- don't change it without reason.
3. We should compute ALL THREE sub-components as separate features, not just the composite. The overnight variance component is a leading indicator of gap risk.
4. Rogers-Satchell is a dependency of Yang-Zhang -- implement it first (it's also useful on its own as a drift-independent vol estimator).

**Pitfalls found:**
- Markets without pre-market trading (some Asian exchanges) have no meaningful overnight returns. For those, fall back to Rogers-Satchell only.
- Our existing Garman-Klass computation in `adaptive_model_params.py` should be consolidated with this.

---

### Feature 5: Volume Clock Intensity

**No direct GitHub implementation found** for the exact "volume clock" feature as proposed.

**Related implementations:** The concept comes from Lopez de Prado's "Advances in Financial Machine Learning" (2018). The closest implementations are in `mlfinlab` (paywalled) and `RiskLabAI` (tick bars, volume bars -- not directly applicable to daily data).

**Useful approach for us:**
- Simplify to: `volume_clock_intensity = volume / volume_avg_21d` -- this is just a z-scored volume anomaly. The original volume clock from AFML requires tick data.
- The daily version captures the same signal: abnormal volume days relative to recent history.
- Apply z-score normalization: `(volume - volume_avg_21d) / volume_std_21d` for better distributional properties.

---

## Domain 2: Econometrics / Stationarity (4 features)

### Feature 6: Fractional Differentiation

**Best implementation found:** [`pushkarkumarvats/OmniQuant`](https://github.com/pushkarkumarvats/OmniQuant) -- `src/feature_engineering/advanced_features.py`
- Clean Python-only implementation with configurable d and threshold
- Uses convolution of fractional weights -- the "fixed-width window" approach from AFML

**Also found:** [`fracdiff/fracdiff`](https://github.com/fracdiff/fracdiff) (335 stars)
- Dedicated package with sklearn and PyTorch integrations
- `fracdiff/fdiff.py` contains the core logic

**Key implementation pattern (from OmniQuant):**
```python
def fractional_differentiation(series, d=0.5, threshold=1e-5):
    weights = [1.0]
    for k in range(1, len(series)):
        weight = -weights[-1] * (d - k + 1) / k
        if abs(weight) < threshold:
            break
        weights.append(weight)
    weights = np.array(weights[::-1])
    result = pd.Series(index=series.index, dtype=float)
    for i in range(len(weights), len(series)):
        result.iloc[i] = np.dot(weights, series.iloc[i-len(weights)+1:i+1])
    return result
```

**Useful patterns for us:**
1. The weight threshold (1e-5) controls window width -- lower threshold = longer memory but heavier computation. For our 504-day cache, 1e-5 is fine.
2. OmniQuant's implementation is ~20 lines and has zero dependencies beyond numpy. We should inline this rather than adding the `fracdiff` package.
3. The optimal d should be found via ADF test: binary search for minimum d that achieves stationarity (ADF p < 0.05). This is a one-time computation per series, not rolling.
4. The `fracdiff` package has a `fracdiff.fdiff_coef()` function that just computes weights, useful for validation.

**Pitfalls found:**
- The naive loop implementation above is O(n*w) where w = weight window size. For 504 rows this is fine; for 10K+ it needs vectorization via `np.convolve()`.
- d=0.5 is a common default but often not optimal. For close prices, typical optimal d is 0.3-0.5.
- Must preserve the original index and NaN handling -- the convolution approach loses the first `len(weights)` rows.

---

### Feature 7: Rolling Hurst Exponent

**Best implementation found:** [`Mottl/hurst`](https://github.com/Mottl/hurst) (336 stars)
- Dedicated package with multiple methods (simplified R/S, standard R/S, corrected R/S)
- Well-tested, handles edge cases (S=0, R=0)

**Also found in:** [`Nixtla/tsfeatures`](https://github.com/Nixtla/tsfeatures) (443 stars)
- `tsfeatures/utils.py` has a compact Hurst implementation using cumulative sum deviation method

**Key implementation pattern (from tsfeatures -- compact):**
```python
def hurst_exponent(x):
    n = x.size
    t = np.arange(1, n + 1)
    y = x.cumsum()
    mean_t = y / t
    s_t = np.sqrt(np.array([np.mean((x[:i+1] - mean_t[i])**2) for i in range(n)]))
    r_t = np.array([np.ptp(y[:i+1] - t[:i+1] * mean_t[i]) for i in range(n)])
    r_s = r_t / s_t
    r_s = np.log(r_s)[1:]
    n_log = np.log(t)[1:]
    hurst, _ = np.linalg.lstsq(np.column_stack((n_log, np.ones(n_log.size))), r_s, rcond=-1)[0]
    return hurst
```

**Useful patterns for us:**
1. tsfeatures' version is more compact and numerically stable than the `hurst` package for short windows.
2. For rolling computation, apply this to 63-day windows. The `hurst` package supports only full-series computation.
3. Key interpretation: H > 0.5 = trending (momentum), H < 0.5 = mean-reverting, H ~ 0.5 = random walk. This directly informs model selection.
4. Our existing computation in `adaptive_model_params.py` uses a similar R/S approach but computes it once. We need it as a daily rolling feature.

**Pitfalls found:**
- For very short windows (< 20 points), the Hurst estimator is unreliable. Use 63d minimum.
- The `hurst` package's `compute_Hc()` function has 3 "kinds": `random_walk`, `price`, `change`. For our return series, use `change`.

---

### Features 8-9: Autocorrelation at Lag 1 and Lag 5

**No dedicated GitHub project needed** -- this is standard pandas.

**Key implementation:**
```python
autocorr_lag1 = return_1d.rolling(63).apply(lambda x: x.autocorr(lag=1), raw=False)
autocorr_lag5 = return_1d.rolling(63).apply(lambda x: x.autocorr(lag=5), raw=False)
```

**Useful insight from academic literature (Lo & MacKinlay 1988):**
- Lag-1 autocorrelation of returns is a direct measure of short-term predictability. Positive = momentum, negative = reversal.
- Lag-5 (weekly) autocorrelation captures the weekly reversal effect documented by Jegadeesh (1990).
- The rolling window should match our adaptive windows (63d for quarterly filers).

**Pitfall:** `pandas.Series.autocorr()` uses Pearson correlation, which is undefined when variance is zero. Our `safe_ratio` pattern should wrap this.

---

## Domain 3: Behavioral Finance (5 features)

### Features 10-11: 52-Week High/Low Anchoring

**No dedicated GitHub implementation found** -- but the formula is trivial:
```python
anchoring_52w_high = close / close.rolling(252).max()
anchoring_52w_low = close / close.rolling(252).min()
```

**Useful insight from academic literature (George & Hwang 2004, Journal of Finance):**
- The 52-week high ratio subsumes standard momentum. Stocks near their 52-week high tend to continue rising.
- The ratio is bounded [0, 1] for high and [1, inf) for low -- natural normalization.
- This is one of the most replicated and persistent anomalies in finance.

---

### Feature 12: Disposition Effect Proxy

**Found:** [`David-Woroniuk/crypto_trading_strategies`](https://github.com/David-Woroniuk/crypto_trading_strategies) has behavioral bias features but not disposition-specific.

**Implementation approach for our pipeline:**
```python
disposition_effect_proxy = return_1d.rolling(63).corr(inst_flow_momentum)
```
If correlation is strongly negative (institutions sell when returns are positive), it suggests disposition effect (selling winners too early).

---

### Feature 13: Attention Spike

**No dedicated implementation needed.** Simple threshold on normalized volume:
```python
volume_zscore = (volume - volume_avg_21d) / volume.rolling(21).std()
attention_spike = (volume_zscore > 2.0).astype(int)
```

**Insight from Barber & Odean (2008):** Attention-driven buying predicts short-term overreaction followed by reversal.

---

### Feature 14: Lottery Characteristics

**Found in academic factor replication:** [`OpenSourceAP/CrossSection`](https://github.com/OpenSourceAP/CrossSection) (963 stars) -- Chen & Zimmermann (2020) replication of 200+ cross-sectional predictors includes lottery demand characteristics.

**Implementation:**
```python
idio_vol = residual_vol_from_beta_regression  # see Feature 28
skewness_63d = return_1d.rolling(63).skew()
low_price = (close < close.expanding().quantile(0.20)).astype(int)
lottery_score = (idio_vol_zscore + skewness_63d_zscore + low_price) / 3
```

---

## Domain 4: Credit Risk / Distress (5 features)

### Feature 15: Cash Burn Rate

**No GitHub implementation needed** -- simple formula:
```python
cash_burn_rate_monthly = np.maximum(0, -operating_cash_flow) / 30
```

### Feature 16: Debt Maturity Pressure

```python
debt_maturity_pressure = short_term_debt / total_debt_asof
```

### Feature 17: Cash Conversion Cycle

**Found in multiple repos** but all are simple accounting formulas. Best pattern:
```python
DSO = receivables / (revenue / 90)     # Days Sales Outstanding
DIO = inventory / (cost_of_revenue / 90)  # Days Inventory Outstanding
DPO = payables / (cost_of_revenue / 90)   # Days Payable Outstanding
CCC = DSO + DIO - DPO
```

**Useful insight:** When COGS is not available (some non-US markets), use `revenue * 0.6` as a proxy (industry median cost ratio). Our estimator module could fill this.

### Feature 18: Altman Z Momentum

```python
altman_z_momentum_63d = fh_altman_z_score.diff(63)
```
Directional change in Z-score catches deterioration BEFORE crossing the 1.81 threshold.

### Feature 19: Covenant Proximity Score

```python
# For each survival trigger, compute distance to threshold as 0-1 score
proximity_scores = []
for var, threshold, direction in survival_triggers:
    if direction == 'below':
        score = max(0, min(1, (threshold - actual) / threshold))
    else:
        score = max(0, min(1, (actual - threshold) / threshold))
    proximity_scores.append(score)
covenant_proximity = max(proximity_scores)
```

---

## Domain 5: Information Theory (4 features)

### Feature 20: Sample Entropy

**Best implementation found:** [`raphaelvallat/antropy`](https://github.com/raphaelvallat/antropy) (367 stars)
- Uses numba JIT compilation for the Chebyshev distance inner loop
- Falls back to sklearn KDTree for non-Chebyshev metrics
- Returns NaN when no matching templates found (instead of arbitrary value)

**Also found in:** [`blue-yonder/tsfresh`](https://github.com/blue-yonder/tsfresh) (9,183 stars)
- `tsfresh/feature_extraction/feature_calculators.py` has a simpler implementation using `_into_subchunks()` helper
- No numba dependency but slower

**Key implementation insight from antropy:**
- Parameters: `order=2, tolerance=0.2*std(x)` are standard choices from Richman & Moorman (2000)
- The numba-accelerated path is ~100x faster for len(x) < 5000
- For our rolling windows (63-126 points), the numba path is ideal

**Useful patterns for us:**
1. antropy's implementation handles NaN input gracefully (returns NaN immediately).
2. The tolerance `r = 0.2 * std(x)` auto-scales to the signal amplitude -- critical for financial data where magnitude varies wildly.
3. For rolling computation, we need to call this on each window. With numba JIT, 504 windows of 63 points each takes ~0.5s.
4. The tsfresh implementation is dependency-free but 5-10x slower. We should use the numba path since numba is already a transitive dependency (via `llvmlite` from `numba` which is pulled by `stumpy`).

**Recommendation:** Inline the antropy numba implementation (~30 lines of the hot path) rather than adding antropy as a dependency.

---

### Feature 21: Approximate Entropy

**Best implementation found:** [`raphaelvallat/antropy`](https://github.com/raphaelvallat/antropy) -- `app_entropy()` function
- Adapted from `mne-features` package
- Uses sklearn KDTree for neighbor counting

**Key difference from Sample Entropy:** ApEn counts self-matches (biased) while SampEn doesn't. SampEn is preferred for short series (our case). However, ApEn on PRICE (not returns) captures a different signal: price predictability vs return predictability.

---

### Feature 22: Lempel-Ziv Complexity

**Best implementation found:** [`raphaelvallat/antropy`](https://github.com/raphaelvallat/antropy) -- `lziv_complexity()` function

**Key implementation insight:**
```python
# Input must be binarized first
binary_returns = ''.join(['1' if r > 0 else '0' for r in returns])
lz = lziv_complexity(binary_returns, normalize=True)
```

**Critical warning from antropy docs:** "Float and integer arrays are cast to uint32 before processing (values are truncated, not discretized into bins). For continuous-valued signals, binarize the sequence first."

**Useful patterns for us:**
1. Binarize returns as >0/<=0 BEFORE calling LZ complexity.
2. Use `normalize=True` to get length-independent complexity measure.
3. Normalized LZ ranges from 0 (perfectly regular) to 1 (random). Values near 1 indicate random-walk behavior where models can't help.

---

### Feature 23: Permutation Entropy

**Best implementations found:**
- [`raphaelvallat/antropy`](https://github.com/raphaelvallat/antropy) -- `perm_entropy()` with optimized comparison-based fast path for order=3 and 4
- [`blue-yonder/tsfresh`](https://github.com/blue-yonder/tsfresh) -- also has `permutation_entropy` calculator
- [`nikdon/pyEntropy`](https://github.com/nikdon/pyEntropy) (369 stars) -- standalone entropy library with `time_delay_embedding()` helper

**Key insight from antropy's optimized implementation:**
- For order=3, uses lookup table with bit-packed pairwise comparisons -- 5-7x faster than argsort-based general path
- Pre-computed `_PE3_LOOKUP` and `_PE4_LOOKUP` tables avoid sorting entirely
- Handles ties via positional epsilon jitter: `x + i * eps` breaks ties by position (matches argsort behavior)

**Useful patterns for us:**
1. Order=3, delay=1 is the standard choice for financial returns (Bandt & Pompe 2002).
2. antropy's bit-packed lookup approach is worth inlining -- it's the fastest known Python implementation for our use case.
3. Permutation entropy is the most robust of the 4 entropy features for short, noisy financial time series because it only uses ordinal patterns (rank order), not magnitudes.

---

## Domain 6: Factor Investing (6 features)

### Features 24-26: Industry-Adjusted Ratios

**Reference implementation:** [`OpenSourceAP/CrossSection`](https://github.com/OpenSourceAP/CrossSection) (963 stars) -- Chen & Zimmermann (2020) replicate 200+ cross-sectional factors including industry-adjusted metrics. SAS code but logic is transferable.

**Implementation approach:** Straightforward subtraction of peer median from target value. We already have peer data from linked_caches.

### Feature 27: Momentum 12-1

**Found in:** Many factor libraries. Canonical implementation:
```python
momentum_12_1 = (close / close.shift(252)) - (close / close.shift(21))
```

**Insight from Novy-Marx (2012):** The last month is excluded because short-term reversal contaminates momentum. This single feature has produced 8-12% annual alpha across decades of data.

### Feature 28: Idiosyncratic Volatility

**Found in:** [`Stefan-Jansen/machine-learning-for-trading`](https://github.com/Stefan-Jansen/machine-learning-for-trading) (17,213 stars) -- Chapter on factor models

**Implementation:**
```python
residual = return_1d - beta_252d * benchmark_return_1d
idiosyncratic_vol = residual.rolling(63).std()
```

**Insight from Ang et al. (2006):** High idiosyncratic vol predicts low future returns. This is one of the most puzzling anomalies in finance (the "idiosyncratic volatility puzzle").

### Feature 29: Earnings Revision Proxy

```python
earnings_revision_proxy = (eps_calc - eps_calc.shift(63)) / abs(eps_calc.shift(63))
```

**Insight from Chan, Jegadeesh & Lakonishok (1996):** Earnings revision momentum is a stronger predictor than price momentum for fundamental stocks.

---

## Domain 7: Operations Research (5 features)

### Features 30-34: Turnover Ratios + SGA Efficiency + CapEx Intensity

**No specialized GitHub implementations needed** -- these are standard accounting ratios. All inputs are already in our cache.

**Key implementation notes:**
- Use `safe_ratio()` for all divisions (already available in our codebase)
- COGS proxy when not available: `revenue - gross_profit` (exact identity) or `revenue * 0.6` (sector median estimate)
- CapEx intensity = `abs(capex) / revenue` -- the `abs()` is important because capex is typically negative in cash flow statements

---

## Domain 8: Tail Risk (5 features)

### Features 35-39: Skewness, Kurtosis, Tail Ratio, Max Loss, Vol-of-Vol

**Found in:** [`ranaroussi/quantstats`](https://github.com/ranaroussi/quantstats) (7,053 stars) -- has tail ratio, skewness, kurtosis computations
**Found in:** [`quantopian/empyrical`](https://github.com/quantopian/empyrical) (1,481 stars) -- has tail ratio and downside risk metrics

**Implementation:** All 5 use standard pandas rolling operations:
```python
skewness_63d = return_1d.rolling(63).skew()
kurtosis_63d = return_1d.rolling(63).kurt()
tail_ratio = abs(return_1d.rolling(63).quantile(0.05)) / return_1d.rolling(63).quantile(0.95)
max_daily_loss_63d = return_1d.rolling(63).min()
vol_of_vol = volatility_21d.rolling(21).std()
```

**Useful insight from empyrical:** They compute tail ratio as `abs(P95/P5)` not `abs(P5/P95)` -- the convention matters for interpretation. We should follow empyrical's convention where >1 means positive skew (right tail larger).

---

## Domain 9: Normalization (4 feature patterns)

### Features 40-43: Z-Scores, Percentiles, Regime-Conditional, Changes

**No specialized implementation needed** -- standard pandas operations.

**Key insight from practice:** The regime-conditional z-score (Feature 42) requires per-regime statistics:
```python
# Group by survival_regime and compute expanding stats
for regime in ['normal', 'company_survival', 'modified_survival', 'extreme_survival']:
    mask = cache['survival_regime'] == regime
    regime_mean = cache.loc[mask, var].expanding().mean()
    regime_std = cache.loc[mask, var].expanding().std()
    cache.loc[mask, f'{var}_regime_zscore'] = (cache.loc[mask, var] - regime_mean) / regime_std.clip(lower=EPSILON)
```

---

## Domain 10: Forensic Accounting (4 features)

### Features 44-47: Revenue-Receivables Divergence, CapEx/Depreciation, Soft Assets, OCF Ratio

**Found in:** Our own HF modules already compute some of these at the quarterly level. The novelty is computing them as DAILY features in the cache (interpolated from quarterly filings).

**Key implementations:**
```python
# Revenue-receivables divergence (Lev & Thiagarajan 1993)
rev_growth = revenue.pct_change(252)
rec_growth = receivables.pct_change(252)
revenue_receivables_divergence = rev_growth - rec_growth  # positive = healthy

# CapEx/Depreciation ratio
depreciation_proxy = ebitda - ebit  # ebit = operating_income
capex_depreciation_ratio = abs(capex) / depreciation_proxy.clip(lower=EPSILON)

# Soft asset ratio (Barton & Simko 2002)
hard_assets = cash_and_equivalents.fillna(0)  # could add net PPE if available
soft_assets = total_assets - hard_assets
soft_asset_ratio = soft_assets / total_assets.clip(lower=EPSILON)

# OCF ratio
ocf_ratio = operating_cash_flow / net_income.where(net_income.abs() > EPSILON)
```

---

## Summary: Key Repos by Usefulness

| Repo | Stars | What We Should Use |
|------|-------|----|
| [`raphaelvallat/antropy`](https://github.com/raphaelvallat/antropy) | 367 | Inline the numba-accelerated sample_entropy, lziv_complexity, perm_entropy implementations (~80 lines total) |
| [`RiskLabAI/RiskLabAI.py`](https://github.com/RiskLabAI/RiskLabAI.py) | 107 | Corwin-Schultz spread (full implementation) + Bekker-Parkinson volatility (spread-adjusted Parkinson) |
| [`mgao6767/frds`](https://github.com/mgao6767/frds) | 101 | Kyle's Lambda (15-line OLS implementation) |
| [`Jensenberg/volatility-and-option`](https://github.com/Jensenberg/volatility-and-option) | 125 | All 6 volatility estimators in a single file (Parkinson, Garman-Klass, Rogers-Satchell, Yang-Zhang) |
| [`Nixtla/tsfeatures`](https://github.com/Nixtla/tsfeatures) | 443 | Compact Hurst exponent implementation (10 lines, numerically stable) |
| [`pushkarkumarvats/OmniQuant`](https://github.com/pushkarkumarvats/OmniQuant) | - | Fractional differentiation (20-line inline implementation, no external dependency) |
| [`blue-yonder/tsfresh`](https://github.com/blue-yonder/tsfresh) | 9,183 | Reference for sample_entropy + permutation_entropy (dependency-free fallback path) |
| [`OpenSourceAP/CrossSection`](https://github.com/OpenSourceAP/CrossSection) | 963 | Academic reference for 200+ cross-sectional factors (SAS but logic is universal) |
| [`scikit-portfolio/scikit-portfolio`](https://github.com/scikit-portfolio/scikit-portfolio) | 58 | Clean Parkinson + Garman-Klass with good docstrings |
| [`Mottl/hurst`](https://github.com/Mottl/hurst) | 336 | Production-grade Hurst exponent with edge case handling |

## Key Implementation Recommendations

1. **Inline over dependency:** For entropy features, inline antropy's numba-accelerated implementations rather than adding it as a pip dependency. The hot paths are short (~30 lines each) and numba is already available via stumpy's dependency chain.

2. **Consolidate volatility estimators:** Jensenberg's single-file pattern (all 6 vol estimators in 120 lines) is the right model. Create one `_compute_advanced_volatility()` stage in derived_variables.py that computes Parkinson, Rogers-Satchell, and Yang-Zhang alongside the existing close-to-close and EWMA volatility.

3. **Use OmniQuant's fractional diff:** The 20-line inline implementation is production-ready. No need for the fracdiff package.

4. **Adapt tsfeatures' Hurst:** The compact 10-line version from tsfeatures is better suited for rolling windows than the full hurst package.

5. **Follow RiskLabAI's decomposition pattern:** Their Corwin-Schultz splits cleanly into beta/gamma/alpha sub-functions with clear docstrings matching the paper's notation. This pattern should be followed for all paper-based implementations.

6. **frds for Kyle Lambda:** The 15-line OLS implementation is perfect. Just wrap it in a rolling window.

7. **No new pip dependencies required:** All features can be implemented inline using numpy, pandas, scipy (already installed), and numba (already a transitive dependency).
