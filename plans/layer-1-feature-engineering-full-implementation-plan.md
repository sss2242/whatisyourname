# Layer 1 Feature Engineering -- Full Implementation Plan

*Combines: Expert Methods Research + GitHub Implementation Findings + Debug Scan Edit Points*

This is the step-by-step implementation plan for adding 47 new features across 10 expert domains to the Layer 1 feature engineering pipeline. Each step includes the exact file, line numbers, code patterns from GitHub research, input/output contracts, and downstream wiring instructions.

---

## Implementation Phases

The work is divided into 7 phases, each independently testable and committable.

---

## Phase 1: Derived Variables Enhancement (5 new stages, 23 features)

**Files edited:** [`operator1/features/derived_variables.py`](operator1/features/derived_variables.py:1)

### Step 1.1: Add Stage 18 -- Microstructure Signals

**Insert after** [`_compute_merton_distance_to_default`](operator1/features/derived_variables.py:888) (line ~938)

**Features to implement (5):**

```
def _compute_microstructure_signals(df: pd.DataFrame) -> pd.DataFrame:
```

| Feature | Implementation Source | Key Pattern |
|---------|---------------------|-------------|
| `corwin_schultz_spread` | RiskLabAI/RiskLabAI.py `corwin_schultz.py` | 3-step decomposition: beta_estimates -> gamma_estimates -> alpha_estimates. Constant `_DENOMINATOR = 3 - 2*sqrt(2)`. Floor alpha at 0. `spread = 2*(exp(alpha) - 1) / (1 + exp(alpha))` |
| `kyle_lambda` | mgao6767/frds `_kyle_lambda.py` | Daily proxy: `signed_vol = volume * sign(return_1d)`. Rolling 21d OLS via `np.linalg.lstsq`. Scale by 1e6. |
| `parkinson_vol_21d` | scikit-portfolio `volatiltiy.py` | `sqrt(1/(4*n*ln2) * sum(ln(H/L)^2))` over rolling 21d window |
| `yang_zhang_vol_21d` | Jensenberg/volatility-and-option `volatility.py` | 3-component: overnight_var + k*close_open_var + (1-k)*rogers_satchell_var. `k = 0.34 / (1.34 + (n+1)/(n-1))` |
| `volume_clock_intensity` | Custom (AFML concept simplified for daily) | `(volume - volume_avg_21d) / volume.rolling(21).std()` |

**Input requirements:** `open`, `high`, `low`, `close`, `volume`, `return_1d`, `volume_avg_21d` -- all available from prior stages.

**Implementation detail for Corwin-Schultz (from RiskLabAI):**
```python
_CS_DENOM = 3 - 2 * (2 ** 0.5)

def _cs_beta(high, low, window=20):
    log_hl_sq = np.log(high / low) ** 2
    return log_hl_sq.rolling(2).sum().rolling(window).mean()

def _cs_gamma(high, low):
    h2 = high.rolling(2).max()
    l2 = low.rolling(2).min()
    return np.log(h2 / l2) ** 2

def _cs_alpha(beta, gamma):
    t1 = (2**0.5 - 1) * np.sqrt(beta) / _CS_DENOM
    t2 = np.sqrt(gamma / _CS_DENOM)
    return np.maximum(t1 - t2, 0)

# Final spread
alpha = _cs_alpha(_cs_beta(high, low), _cs_gamma(high, low))
df["corwin_schultz_spread"] = 2 * (np.exp(alpha) - 1) / (1 + np.exp(alpha))
```

### Step 1.2: Add Stage 19 -- Stationarity Features

**Insert after** Stage 18

**Features to implement (4):**

```
def _compute_stationarity_features(df: pd.DataFrame) -> pd.DataFrame:
```

| Feature | Implementation Source | Key Pattern |
|---------|---------------------|-------------|
| `close_frac_diff` | OmniQuant `advanced_features.py` | Weight series: `w[k] = -w[k-1] * (d-k+1)/k`, truncate at threshold 1e-5. Convolve with close. Default d=0.4. |
| `hurst_exponent_rolling` | Nixtla/tsfeatures `utils.py` | R/S analysis: cumsum deviations, ptp range, std normalization. Rolling 63d. OLS slope of log(R/S) vs log(n). |
| `autocorr_lag1` | Standard pandas | `return_1d.rolling(63).apply(lambda x: x.autocorr(lag=1), raw=False)` |
| `autocorr_lag5` | Standard pandas | `return_1d.rolling(63).apply(lambda x: x.autocorr(lag=5), raw=False)` |

**Implementation detail for fractional diff (from OmniQuant, 20 lines):**
```python
def _frac_diff(series, d=0.4, threshold=1e-5):
    weights = [1.0]
    for k in range(1, len(series)):
        w = -weights[-1] * (d - k + 1) / k
        if abs(w) < threshold:
            break
        weights.append(w)
    weights = np.array(weights[::-1])
    result = np.full(len(series), np.nan)
    for i in range(len(weights), len(series)):
        result[i] = np.dot(weights, series.values[i - len(weights) + 1:i + 1])
    return pd.Series(result, index=series.index)
```

**Implementation detail for Hurst (from tsfeatures, 10 lines):**
```python
def _hurst(x):
    n = x.size
    t = np.arange(1, n + 1)
    y = x.cumsum()
    mean_t = y / t
    s_t = np.sqrt(np.array([np.mean((x[:i+1] - mean_t[i])**2) for i in range(n)]))
    r_t = np.array([np.ptp(y[:i+1] - t[:i+1] * mean_t[i]) for i in range(n)])
    with np.errstate(invalid="ignore"):
        r_s = np.log(r_t / s_t)[1:]
    n_log = np.log(t)[1:]
    h, _ = np.linalg.lstsq(np.column_stack((n_log, np.ones(n_log.size))), r_s, rcond=-1)[0]
    return float(np.clip(h, 0, 1))
```

### Step 1.3: Add Stage 20 -- Credit Signals

**Features to implement (5):**

```
def _compute_credit_signals(df: pd.DataFrame) -> pd.DataFrame:
```

| Feature | Formula | Fallback |
|---------|---------|----------|
| `cash_burn_rate_monthly` | `max(0, -operating_cash_flow) / 30` | NaN if OCF missing |
| `debt_maturity_pressure` | `short_term_debt / total_debt_asof` | NaN if either missing |
| `cash_conversion_cycle` | `DSO + DIO - DPO` where DSO=receivables/(revenue/90), DIO=inventory/(COGS/90), DPO=payables/(COGS/90) | COGS fallback: `revenue - gross_profit`. NaN if receivables/inventory missing. |
| `altman_z_momentum_63d` | `fh_altman_z_score.diff(63)` | NaN if `fh_altman_z_score` not in cache (FH runs after derived_vars). **SOLUTION:** This feature should be computed in a SECOND pass or in financial_health.py itself. For derived_variables.py, skip if column missing. |
| `covenant_proximity_score` | `max(proximity for each survival trigger)` where proximity = normalized distance to threshold (0-1) | Uses thresholds from scoring_weights.yml |

**Critical note on `altman_z_momentum_63d`:** The Altman Z-score is computed in [`financial_health.py`](operator1/models/financial_health.py:424) at Step 5d, which runs AFTER derived_variables at Step 5. Options:
- **Option A (recommended):** Compute it inside `compute_financial_health()` after the Z-score is available
- **Option B:** Run `_compute_credit_signals` a second time after FH (adds complexity)
- **Option C:** Move it to `feature_normalization.py` which runs last

### Step 1.4: Add Stage 22 -- Tail Risk Features

**Features to implement (5):**

```
def _compute_tail_risk_features(df: pd.DataFrame) -> pd.DataFrame:
```

| Feature | Formula | Notes |
|---------|---------|-------|
| `skewness_63d` | `return_1d.rolling(63).skew()` | Standard pandas |
| `kurtosis_63d` | `return_1d.rolling(63).kurt()` | Standard pandas (excess kurtosis, normal=0) |
| `tail_ratio_63d` | `abs(return_1d.rolling(63).quantile(0.95) / return_1d.rolling(63).quantile(0.05))` | Follow empyrical convention: >1 = positive skew |
| `max_daily_loss_63d` | `return_1d.rolling(63).min()` | Always negative |
| `vol_of_vol_21d` | `volatility_21d.rolling(21).std()` | Requires volatility_21d from Stage 1 |

### Step 1.5: Add Stage 24 -- Forensic Signals

**Features to implement (4):**

```
def _compute_forensic_signals(df: pd.DataFrame) -> pd.DataFrame:
```

| Feature | Formula |
|---------|---------|
| `revenue_receivables_divergence` | `revenue.pct_change(252) - receivables.pct_change(252)` -- positive = healthy |
| `capex_depreciation_ratio` | `abs(capex) / max(ebitda - ebit, EPSILON)` -- depreciation proxy |
| `soft_asset_ratio` | `(total_assets - cash_and_equivalents.fillna(0)) / total_assets` |
| `ocf_ratio` | `operating_cash_flow / net_income.where(abs(net_income) > EPSILON)` |

### Step 1.6: Update `_COMPUTE_STAGES` tuple

At [`line 942`](operator1/features/derived_variables.py:942), add:
```python
_COMPUTE_STAGES = (
    _compute_returns_and_risk,        # Stage 1
    _compute_solvency,                # Stage 2
    _compute_liquidity,               # Stage 3
    _compute_interest_coverage,       # Stage 4
    _compute_cash_reality,            # Stage 5
    _compute_profitability,           # Stage 6
    _compute_roa,                     # Stage 7
    _compute_valuation,               # Stage 8
    _compute_ttm_and_growth,          # Stage 9
    _compute_volume_avg,              # Stage 10
    _compute_per_share,               # Stage 11
    _compute_technical_indicators,    # Stage 12
    _compute_recovery_time,           # Stage 13
    _compute_beta,                    # Stage 14
    _compute_earnings_quality_signals,# Stage 15
    _compute_realized_vol_decomposition,  # Stage 16
    _compute_merton_distance_to_default,  # Stage 17
    _compute_microstructure_signals,  # Stage 18 (NEW)
    _compute_stationarity_features,   # Stage 19 (NEW)
    _compute_credit_signals,          # Stage 20 (NEW)
    _compute_tail_risk_features,      # Stage 22 (NEW)
    _compute_forensic_signals,        # Stage 24 (NEW)
)
```

### Step 1.7: Update `DERIVED_VARIABLES` tuple

At [`line 963`](operator1/features/derived_variables.py:963), add all new variable names. This is consumed by [`data_quality.py:343`](operator1/quality/data_quality.py:343).

---

## Phase 2: New Module -- Behavioral Signals (5 features)

**Create file:** `operator1/features/behavioral_signals.py` (~150 lines)

### Step 2.1: Implement the module

```python
"""Behavioral finance signals -- anomalies from investor psychology.

Features:
  - anchoring_52w_high/low: George & Hwang 2004
  - disposition_effect_proxy: Shefrin & Statman 1985
  - attention_spike: Barber & Odean 2008
  - lottery_characteristics: Bali, Cakici & Whitelaw 2011

Pipeline step: Step 5i.7
"""

def compute_behavioral_signals(cache: pd.DataFrame) -> pd.DataFrame:
    result = cache.copy()
    # 1. Anchoring
    result["anchoring_52w_high"] = result["close"] / result["close"].rolling(252).max()
    result["anchoring_52w_low"] = result["close"] / result["close"].rolling(252).min()
    
    # 2. Disposition effect (needs inst_flow_momentum)
    if "inst_flow_momentum" in result.columns and "return_1d" in result.columns:
        result["disposition_effect_proxy"] = (
            result["return_1d"].rolling(63).corr(result["inst_flow_momentum"])
        )
    
    # 3. Attention spike
    if "volume" in result.columns and "volume_avg_21d" in result.columns:
        vol_std = result["volume"].rolling(21).std()
        vol_z = (result["volume"] - result["volume_avg_21d"]) / vol_std.clip(lower=EPSILON)
        result["attention_spike"] = (vol_z > 2.0).astype(int)
    
    # 4. Lottery characteristics
    if "return_1d" in result.columns and "beta_252d" in result.columns:
        benchmark = result.get("benchmark_return_1d")
        if benchmark is not None:
            residual = result["return_1d"] - result["beta_252d"] * benchmark
            idio_vol = residual.rolling(63).std()
        else:
            idio_vol = result["return_1d"].rolling(63).std()
        skew = result["return_1d"].rolling(63).skew()
        low_price = (result["close"] < result["close"].expanding().quantile(0.20)).astype(float)
        # Normalize each component
        for s in [idio_vol, skew, low_price]:
            s_z = (s - s.expanding().mean()) / s.expanding().std().clip(lower=EPSILON)
        result["lottery_characteristics"] = (idio_vol + skew + low_price) / 3
    
    return result
```

### Step 2.2: Wire into main.py

At [`main.py`](main.py) after Step 5i.5 (product_catalysts, ~line 2125), add:
```python
# Step 5i.7: Behavioral finance signals
try:
    from operator1.features.behavioral_signals import compute_behavioral_signals
    cache = compute_behavioral_signals(cache)
except Exception as exc:
    logger.warning("Behavioral signals failed: %s", exc)
```

### Step 2.3: Wire into backtest_runner.py

At [`backtest_runner.py`](backtest_runner.py) after product_catalysts block (~line 784), add matching call.

---

## Phase 3: New Module -- Complexity Signals (4 features)

**Create file:** `operator1/features/complexity_signals.py` (~200 lines)

### Step 3.1: Implement entropy functions

Inline the core algorithms from antropy (numba-accelerated) and tsfresh (fallback):

**Sample entropy** (from antropy, ~30 lines of numba hot path):
```python
@numba.jit(nopython=True)
def _sample_entropy_numba(x, m, r):
    """Fast sample entropy via Chebyshev distance."""
    n = len(x)
    # ... (inline antropy's _sampen numba kernel)
```

**Permutation entropy** (from antropy, ~20 lines with lookup table):
```python
_PE3_LOOKUP = np.zeros(8, dtype=np.int8)
# ... (inline antropy's comparison-based fast path for order=3)
```

**LZ complexity** (from antropy, ~15 lines):
```python
def _lz_complexity(binary_string):
    """Lempel-Ziv 1976 complexity."""
    # ... (standard LZ76 algorithm)
```

### Step 3.2: Implement the module entry point

```python
def compute_complexity_signals(cache: pd.DataFrame, window: int = 63) -> pd.DataFrame:
    result = cache.copy()
    returns = result.get("return_1d")
    if returns is None or returns.notna().sum() < window + 10:
        return result
    
    # Rolling computations
    result["sample_entropy_21d"] = returns.rolling(window).apply(
        lambda x: _sample_entropy(x.values, order=2, r=0.2*np.std(x)),
        raw=False
    )
    result["perm_entropy_21d"] = returns.rolling(window).apply(
        lambda x: _perm_entropy(x.values, order=3, delay=1),
        raw=False
    )
    # LZ complexity on binarized returns
    result["lz_complexity"] = returns.rolling(window).apply(
        lambda x: _lz_complexity("".join(["1" if v > 0 else "0" for v in x])),
        raw=False
    )
    # ApEn on close prices (different signal than return entropy)
    close = result.get("close")
    if close is not None:
        result["approx_entropy_price"] = close.rolling(126).apply(
            lambda x: _app_entropy(x.values, order=2, r=0.2*np.std(x)),
            raw=False
        )
    return result
```

### Step 3.3: Wire into main.py and backtest_runner.py

Add as Step 5i.8 after behavioral_signals.

---

## Phase 4: New Module -- Feature Normalization (40 features)

**Create file:** `operator1/features/feature_normalization.py` (~150 lines)

### Step 4.1: Implement the module

```python
"""Feature normalization -- regime-aware z-scores and percentiles.

MUST run LAST in the feature pipeline (Step 5i.9), after all other
feature modules have populated the cache, and after hierarchy_weights
has computed survival_regime.

10 key variables x 4 normalization types = ~40 new columns.
"""

_NORMALIZE_VARS = [
    "current_ratio", "debt_to_equity_abs", "fcf_yield", "gross_margin",
    "volatility_21d", "pe_ratio_calc", "revenue_growth_yoy",
    "sentiment_score", "merton_dd", "fh_composite_score",
]

_REGIME_VARS = [
    "current_ratio", "debt_to_equity_abs", "fcf_yield",
    "drawdown_252d", "volatility_21d",
]

def compute_feature_normalization(cache: pd.DataFrame) -> pd.DataFrame:
    result = cache.copy()
    
    for var in _NORMALIZE_VARS:
        if var not in result.columns:
            continue
        s = result[var]
        
        # 1. Rolling z-score (63d)
        mean_63 = s.rolling(63, min_periods=10).mean()
        std_63 = s.rolling(63, min_periods=10).std().clip(lower=EPSILON)
        result[f"{var}_zscore_63d"] = (s - mean_63) / std_63
        
        # 2. Expanding percentile (252d)
        result[f"{var}_percentile_252d"] = s.rolling(252, min_periods=20).apply(
            lambda x: pd.Series(x).rank(pct=True).iloc[-1], raw=False
        )
        
        # 3. Level change (21d)
        result[f"{var}_change_21d"] = s.diff(21)
    
    # 4. Regime-conditional z-score (for survival variables only)
    if "survival_regime" in result.columns:
        for var in _REGIME_VARS:
            if var not in result.columns:
                continue
            col = f"{var}_regime_zscore"
            result[col] = np.nan
            for regime in result["survival_regime"].dropna().unique():
                mask = result["survival_regime"] == regime
                s_regime = result.loc[mask, var]
                rmean = s_regime.expanding(min_periods=5).mean()
                rstd = s_regime.expanding(min_periods=5).std().clip(lower=EPSILON)
                result.loc[mask, col] = (s_regime - rmean) / rstd
    
    return result
```

### Step 4.2: Wire into main.py

Add as Step 5i.9 -- the LAST feature module call before temporal models:
```python
# Step 5i.9: Feature normalization (MUST run last)
try:
    from operator1.features.feature_normalization import compute_feature_normalization
    cache = compute_feature_normalization(cache)
except Exception as exc:
    logger.warning("Feature normalization failed: %s", exc)
```

---

## Phase 5: Enhanced Existing Modules (8 features)

### Step 5.1: Enhance peer_ranking.py -- Add 3 industry-adjusted ratios

At [`operator1/features/peer_ranking.py`](operator1/features/peer_ranking.py), add after the existing `compute_peer_ranking()` function:

```python
def compute_industry_adjusted_ratios(
    cache: pd.DataFrame, linked_caches: dict
) -> pd.DataFrame:
    """Compute industry-adjusted PE, margin, and vol."""
    result = cache.copy()
    _ADJUST_VARS = {
        "pe_industry_adjusted": "pe_ratio_calc",
        "margin_industry_adjusted": "gross_margin",
        "vol_industry_adjusted": "volatility_21d",
    }
    for adj_name, source_var in _ADJUST_VARS.items():
        if source_var not in result.columns:
            continue
        peer_values = []
        for eid, pcache in linked_caches.items():
            if source_var in pcache.columns:
                last = pcache[source_var].dropna().iloc[-1] if pcache[source_var].notna().any() else None
                if last is not None:
                    peer_values.append(last)
        if peer_values:
            peer_median = np.median(peer_values)
            result[adj_name] = result[source_var] - peer_median
        else:
            # Fallback: own-history percentile
            result[adj_name] = result[source_var] - result[source_var].expanding().median()
    return result
```

Wire: call from main.py after `compute_peer_ranking()` at Step 5h.

### Step 5.2: Enhance product_metrics.py -- Add 5 operational features

At [`operator1/features/product_metrics.py`](operator1/features/product_metrics.py), add:

```python
def compute_operational_efficiency(cache: pd.DataFrame) -> pd.DataFrame:
    """Compute turnover ratios and operational efficiency metrics."""
    result = cache.copy()
    
    # COGS proxy
    cogs = result.get("cost_of_revenue")
    if cogs is None or cogs.isna().all():
        rev = result.get("revenue")
        gp = result.get("gross_profit")
        if rev is not None and gp is not None:
            cogs = (rev - gp).clip(lower=0)
    
    rev = result.get("revenue")
    if rev is None:
        return result
    
    for num_col, name in [
        ("inventory", "inventory_turnover"),
        ("receivables", "receivables_turnover"),
        ("payables", "payables_turnover"),
    ]:
        num = result.get(num_col)
        denom = cogs if name == "payables_turnover" else rev
        if num is not None and denom is not None:
            safe_num = num.where(num.abs() > EPSILON)
            result[name] = denom / safe_num
    
    # SGA efficiency
    sga = result.get("sga_expenses")
    if sga is not None:
        safe_sga = sga.where(sga.abs() > EPSILON)
        result["sga_efficiency"] = rev / safe_sga
    
    # CapEx intensity
    capex = result.get("capex")
    if capex is not None:
        safe_rev = rev.where(rev.abs() > EPSILON)
        result["capex_intensity"] = capex.abs() / safe_rev
    
    return result
```

Wire: call from main.py at Step 5i.6 after `compute_product_metrics()`.

### Step 5.3: Add momentum and factor features to derived_variables.py

Inside Stage 19 (`_compute_stationarity_features`) or as separate Stage 21:

```python
# Momentum 12-1 (Jegadeesh & Titman 1993)
if "close" in df.columns and df["close"].notna().sum() > 260:
    ret_12m = df["close"] / df["close"].shift(252) - 1
    ret_1m = df["close"] / df["close"].shift(21) - 1
    df["momentum_12_1"] = ret_12m - ret_1m

# Idiosyncratic volatility (Ang et al. 2006)
if "return_1d" in df.columns and "beta_252d" in df.columns:
    bench = df.get("benchmark_return_1d")
    if bench is not None:
        residual = df["return_1d"] - df["beta_252d"] * bench
        df["idiosyncratic_vol_63d"] = residual.rolling(63, min_periods=20).std()

# Earnings revision proxy (Chan et al. 1996)
eps = df.get("eps_calc")
if eps is None:
    eps = df.get("net_income_ttm_asof")
if eps is not None and eps.notna().sum() > 70:
    df["earnings_revision_proxy"] = eps.pct_change(63)
```

---

## Phase 6: Downstream Wiring (10 files, minor edits)

### Step 6.1: Update `_init_extra_vars()` in stage3_temporal.py

At [`operator1/stages/stage3_temporal.py:44`](operator1/stages/stage3_temporal.py:44), add new prefix patterns:

```python
state.extra_vars = [
    c for c in cache.columns
    if (c.startswith("fh_") or c.startswith("sentiment_")
        or c.startswith("peer_") or c.startswith("macro_")
        or c.startswith("inst_")
        or c.startswith("buying_power_") or c.startswith("catalyst_")
        or c.startswith("conflict_") or c.startswith("demand_")
        or c.startswith("merton_") or c.startswith("rv_")
        or c.startswith("policy_risk_") or c.startswith("sector_leader_")
        or c.startswith("segment_") or c.startswith("product_")
        or c.startswith("pricing_") or c.startswith("margin_")
        or c.startswith("som_") or c.startswith("customer_")
        # NEW prefixes for Layer 1 enhancement
        or c.startswith("corwin_") or c.startswith("kyle_")
        or c.startswith("parkinson_") or c.startswith("yang_zhang_")
        or c.startswith("hurst_") or c.startswith("autocorr_")
        or c.startswith("anchoring_") or c.startswith("disposition_")
        or c.startswith("lottery_") or c.startswith("attention_")
        or c.startswith("cash_burn_") or c.startswith("debt_maturity_")
        or c.startswith("covenant_") or c.startswith("ccc_")
        or c.startswith("sample_entropy") or c.startswith("perm_entropy")
        or c.startswith("lz_") or c.startswith("approx_entropy")
        or c.startswith("skewness_") or c.startswith("kurtosis_")
        or c.startswith("tail_ratio") or c.startswith("vol_of_vol")
        or c.startswith("revenue_rec_") or c.startswith("capex_depr")
        or c.startswith("soft_asset") or c.startswith("ocf_ratio")
        or c.endswith("_zscore_63d") or c.endswith("_percentile_252d")
        or c.endswith("_regime_zscore") or c.endswith("_change_21d")
        # ... existing named columns ...
```

### Step 6.2: Update signal_ic.py

At [`operator1/analysis/signal_ic.py:37`](operator1/analysis/signal_ic.py:37), add new signals:

```python
_SIGNAL_SPEED = {
    # ... existing 39 signals ...
    # NEW slow signals (fundamental)
    "cash_conversion_cycle": "slow",
    "debt_maturity_pressure": "slow",
    "soft_asset_ratio": "slow",
    "ocf_ratio": "slow",
    "capex_intensity": "slow",
    "sga_efficiency": "slow",
    "revenue_receivables_divergence": "slow",
    # NEW medium signals
    "covenant_proximity_score": "medium",
    "altman_z_momentum_63d": "medium",
    "earnings_revision_proxy": "medium",
    "disposition_effect_proxy": "medium",
    "pe_industry_adjusted": "medium",
    # NEW fast signals
    "corwin_schultz_spread": "fast",
    "parkinson_vol_21d": "fast",
    "yang_zhang_vol_21d": "fast",
    "hurst_exponent_rolling": "fast",
    "sample_entropy_21d": "fast",
    "skewness_63d": "fast",
    "kurtosis_63d": "fast",
    "anchoring_52w_high": "fast",
    "momentum_12_1": "fast",
    "attention_spike": "fast",
}
```

### Step 6.3: Update survival_hierarchy.yml

Add new variables to appropriate tiers (see debug scan document section 2c).

### Step 6.4: Update model_tests.py

Add 3 new smoke test entries to the `MODEL_TESTS` dict at [`line 320`](operator1/monitoring/model_tests.py:320):

```python
"behavioral_signals": {"layer": "features", "fn": _test_behavioral_signals},
"complexity_signals": {"layer": "features", "fn": _test_complexity_signals},
"feature_normalization": {"layer": "features", "fn": _test_feature_normalization},
```

### Step 6.5: Update multi_frequency_runner.py

At [`operator1/steps/multi_frequency_runner.py:202`](operator1/steps/multi_frequency_runner.py:202), add calls to new modules after `compute_derived_variables()`.

---

## Phase 7: Testing

### Step 7.1: Unit tests for each new stage

Create `tests/test_layer1_enhancements.py`:
- Test each `_compute_*` stage with synthetic cache (known OHLCV values)
- Verify formulas produce expected output for known inputs
- Test all-NaN input produces NaN output (no crashes)
- Test data-sparse input (50 rows) for graceful degradation

### Step 7.2: Integration test

- Run `compute_derived_variables()` on real AAPL cache
- Verify no regression on existing 67 features
- Verify new features produce non-NaN values for at least 50% of rows

### Step 7.3: Signal IC validation

After implementation, measure information coefficient of each new feature vs 5d/21d forward returns on historical data (AAPL, Samsung, Petrobras).

---

## Implementation Checklist

```
[ ] Phase 1: derived_variables.py (23 features in 5 new stages)
    [ ] Stage 18: _compute_microstructure_signals (5 features)
    [ ] Stage 19: _compute_stationarity_features (4 features)
    [ ] Stage 20: _compute_credit_signals (5 features)
    [ ] Stage 22: _compute_tail_risk_features (5 features)
    [ ] Stage 24: _compute_forensic_signals (4 features)
    [ ] Update _COMPUTE_STAGES tuple
    [ ] Update DERIVED_VARIABLES tuple

[ ] Phase 2: behavioral_signals.py (5 features)
    [ ] Create module
    [ ] Wire into main.py Step 5i.7
    [ ] Wire into backtest_runner.py

[ ] Phase 3: complexity_signals.py (4 features)
    [ ] Inline entropy implementations (numba + fallback)
    [ ] Create module
    [ ] Wire into main.py Step 5i.8
    [ ] Wire into backtest_runner.py

[ ] Phase 4: feature_normalization.py (~40 features)
    [ ] Create module (MUST run LAST)
    [ ] Wire into main.py Step 5i.9
    [ ] Wire into backtest_runner.py

[ ] Phase 5: Enhance existing modules (8 features)
    [ ] peer_ranking.py: 3 industry-adjusted ratios
    [ ] product_metrics.py: 5 operational efficiency features
    [ ] derived_variables.py: momentum_12_1, idiosyncratic_vol, earnings_revision

[ ] Phase 6: Downstream wiring
    [ ] stage3_temporal.py: _init_extra_vars prefix patterns
    [ ] signal_ic.py: add ~20 new signals to _SIGNAL_SPEED
    [ ] survival_hierarchy.yml: tier membership
    [ ] scoring_weights.yml: new thresholds (if any)
    [ ] model_tests.py: 3 new smoke tests
    [ ] multi_frequency_runner.py: new module calls

[ ] Phase 7: Testing
    [ ] Unit tests for each new stage
    [ ] Integration test on real data
    [ ] Signal IC validation
```
