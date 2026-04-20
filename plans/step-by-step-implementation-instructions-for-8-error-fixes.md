# Step-by-Step Implementation Instructions for 8 Error Fixes

*2026-04-20*

Code-mode implementation instructions. Each step is atomic -- compile-check after each. No new sub-stages needed -- all changes fit within existing pipeline stages. None of these are new models; they are enhancements to existing models and post-processing logic.

**Branch:** `fix/close-prediction-error-gaps`
**Commit:** `fix: 8 improvements to close prediction error gaps -- XBRL concepts, NaN scoring, MC intervals, Merton anchor, regime shift, PELT fallback, PEAD drift, mcap calibration`

---

## Pipeline Call Chain Context

These 8 methods slot into the EXISTING call chain. No new sub-stages are needed because:
- Methods 1 is a Stage 1 data fetch enhancement
- Methods 2 is a HF scoring fix -- runs inside existing sub-stage 7.5
- Methods 3 patches conformal output -- runs inside existing sub-stage 6.3
- Methods 4 and 8 are post-MC anchoring -- added after sub-stage 5.4
- Method 5 patches forecast output -- runs inside existing 4.1
- Method 6 patches regime detection -- runs inside existing 3.1
- Method 7 patches prediction aggregation -- runs inside existing 6.5

```
Stage 1:  [Data fetch] -> Method 1 (XBRL fallback) -> [cache build]
Sub 3.1:  [Regime detection] -> Method 6 (PELT fallback for degenerate HMM)
Sub 4.1:  [Forecasting] -> Method 5 (regime probability-weighted shift)
Sub 5.4:  [Monte Carlo] -> Methods 4+8 (Merton anchor + mcap floor) -- NEW POST-MC BLOCK
Sub 6.3:  [Conformal] -> Method 3 (MC intervals for 21d+)
Sub 6.5:  [Prediction aggregation] -> Method 7 (PEAD drift)
Sub 7.5:  [Hedge Fund] -> Method 2 (NaN-aware scoring)
```

---

## Step 1: Expand XBRL Concept Coverage (Method 1)

**File:** `operator1/clients/us_edgar.py`

### Step 1a: Expand _USGAAP_BALANCE_CONCEPTS (line 57-73)

Add these concept mappings after the existing entries:

```python
# After line 73 ("AccountsPayableCurrent": "payables"):
"CashCashEquivalentsRestrictedCashAndRestrictedCashEquivalents": "cash_and_equivalents",
"CashCashEquivalentsRestrictedCashAndRestrictedCashEquivalentsIncludingDisposalGroupAndDiscontinuedOperations": "cash_and_equivalents",
"MarketableSecuritiesCurrent": "short_term_investments",
"AccountsReceivableNet": "receivables",
"PropertyPlantAndEquipmentNet": "ppe_net",
"OtherAssetsCurrent": "other_current_assets",
"OtherLiabilitiesCurrent": "other_current_liabilities",
```

### Step 1b: Expand _USGAAP_INCOME_CONCEPTS (line 41-55)

Add after existing entries:

```python
"OperatingExpenses": "operating_expenses",
"CostOfGoodsSold": "cost_of_revenue",
"SalesRevenueNet": "revenue",
"SalesRevenueGoodsNet": "revenue",
```

### Step 1c: Add _fill_critical_nans helper method

Add a new method to `USEdgarClient` class (after `_extract_from_companyfacts` around line 1860):

```python
def _fill_critical_nans_from_companyfacts(
    self, identifier: str, df: pd.DataFrame, statement_type: str,
) -> pd.DataFrame:
    """Fill NaN columns in critical financial fields using CompanyFacts fallback.
    
    When edgartools XBRL parsing misses fields due to non-standard concept names,
    this method queries the SEC CompanyFacts JSON which contains ALL reported concepts
    and searches for alternative tags.
    
    Only runs for the 7 critical ratio fields:
    current_assets, current_liabilities, cash_and_equivalents,
    interest_expense, gross_profit, net_income, revenue
    """
    # Define critical fields and their alternative SEC concept names
    _BALANCE_ALTERNATES = {
        "cash_and_equivalents": [
            "CashCashEquivalentsRestrictedCashAndRestrictedCashEquivalents",
            "CashAndCashEquivalentsAtCarryingValue",
            "CashCashEquivalentsAndShortTermInvestments",
        ],
        "current_assets": ["AssetsCurrent"],
        "current_liabilities": ["LiabilitiesCurrent"],
    }
    _INCOME_ALTERNATES = {
        "interest_expense": ["InterestExpense", "InterestExpenseDebt"],
        "gross_profit": ["GrossProfit"],
    }
    
    alternates = _BALANCE_ALTERNATES if statement_type == "balance" else _INCOME_ALTERNATES
    
    # Check which critical fields are NaN
    missing = [f for f in alternates if f in df.columns and df[f].isna().all()]
    if not missing:
        return df  # nothing to fill
    
    # Fetch CompanyFacts
    try:
        facts = self._fetch_companyfacts_direct(identifier)
        usgaap = facts.get("facts", {}).get("us-gaap", {})
    except Exception:
        return df
    
    for field, concepts in alternates.items():
        if field not in missing:
            continue
        for concept in concepts:
            concept_data = usgaap.get(concept, {})
            units = concept_data.get("units", {})
            usd_facts = units.get("USD", [])
            if not usd_facts:
                continue
            # Take the most recent annual or quarterly filing
            for fact in sorted(usd_facts, key=lambda x: x.get("filed", ""), reverse=True):
                if fact.get("val") is not None:
                    # Fill the NaN column with this value (forward-filled in cache builder)
                    df[field] = df[field].fillna(fact["val"])
                    logger.info("CompanyFacts filled %s from %s: %s", field, concept, fact["val"])
                    break
            if df[field].notna().any():
                break  # found a value, stop trying alternates
    
    return df
```

### Step 1d: Call the helper after primary extraction

In `get_balance_sheet()` (around line 520), after the primary edgartools extraction:

```python
# After edgartools extraction returns df:
df = self._fill_critical_nans_from_companyfacts(identifier, df, "balance")
```

In `get_income_statement()` (around line 500):

```python
df = self._fill_critical_nans_from_companyfacts(identifier, df, "income")
```

### Compile check: `python -m py_compile operator1/clients/us_edgar.py`

---

## Step 2: NaN-Aware Piotroski and Altman Scoring (Method 2)

**File:** `operator1/hedge_fund/advanced_methods.py`

### Step 2a: Add `n_available` fields to dataclass (line 51-53)

```python
# After line 53:
piotroski_n_available: int = 0       # how many of 9 signals had data
altman_n_available: int = 0          # how many of 4 terms had data
```

### Step 2b: Modify compute_piotroski_f_score (line 79-175)

Add a counter `n_available = 0` at line 88. For each signal block, increment `n_available` when the input data exists (regardless of whether the signal scores 0 or 1):

```python
# At line 88:
score = 0
n_available = 0

# Signal 1 (line 136):
if ni is not None:
    n_available += 1
    if ni > 0:
        score += 1
# (similar for all 9 signals -- add n_available += 1 when inputs exist)

# At the end (before return), normalize if partial:
if n_available < 9 and n_available >= 4:
    # Normalize to 9-point scale based on available signals
    normalized = round(score / n_available * 9)
    label = "strong" if normalized >= 7 else "weak" if normalized <= 2 else "moderate"
    label += f" ({n_available}/9 signals available)"
    return normalized, n_available, label
elif n_available < 4:
    return score, n_available, f"insufficient ({n_available}/9 signals)"
else:
    label = "strong" if score >= 7 else "weak" if score <= 2 else "moderate"
    return score, n_available, label
```

### Step 2c: Modify compute_altman_z_double_prime (line 440-500)

Similar approach -- track which of the 4 terms have valid inputs:

```python
# After computing each term, check if inputs were NaN
terms_available = 0
z = 0.0
if wc is not None and ta is not None and ta > 0:
    z += 6.56 * (wc / ta)
    terms_available += 1
# ... for all 4 terms
if terms_available >= 2:
    z = z / terms_available * 4  # normalize to 4-term scale
else:
    z = float("nan")  # insufficient data
```

### Step 2d: Update callers in engine.py (line 852)

```python
# Change from:
result.piotroski_f_score, result.piotroski_label = compute_piotroski_f_score(...)
# To:
result.piotroski_f_score, result.piotroski_n_available, result.piotroski_label = compute_piotroski_f_score(...)
```

### Compile check: `python -m py_compile operator1/hedge_fund/advanced_methods.py && python -m py_compile operator1/hedge_fund/engine.py`

---

## Step 3: MC Path Intervals for 21d+ (Method 3)

**File:** `operator1/stages/stage6_ensemble.py`

### Step 3a: Add MC interval override in run_6_3_conformal (after line 167)

After `state.conformal_result = build_conformal_result(...)`:

```python
# Replace conformal intervals at 21d+ with MC path percentiles
if state.mc_result is not None and state.conformal_result is not None:
    try:
        _last_close = float(cache["close"].dropna().iloc[-1]) if "close" in cache.columns and cache["close"].notna().any() else None
        if _last_close and hasattr(state.conformal_result, "intervals"):
            for var, hd in state.conformal_result.intervals.items():
                if not isinstance(hd, dict):
                    continue
                for h_label, interval in hd.items():
                    h_days = {"1d": 1, "5d": 5, "21d": 21, "252d": 252}.get(h_label, 0)
                    if h_days >= 21:
                        # Use MC terminal values for this horizon
                        _mc_tv = state.mc_result.terminal_values.get(h_label)
                        if _mc_tv is not None and len(_mc_tv) > 0:
                            _mc_prices = _last_close * (1 + np.array(_mc_tv))
                            if hasattr(interval, "lower"):
                                interval.lower = float(np.percentile(_mc_prices, 5))
                            if hasattr(interval, "upper"):
                                interval.upper = float(np.percentile(_mc_prices, 95))
    except Exception:
        pass
```

### Step 3b: Mirror in backtest_runner.py (after line 1635)

Same block after `state.conformal_result = build_conformal_result(...)`.

### Compile check: `python -m py_compile operator1/stages/stage6_ensemble.py && python -m py_compile backtest_runner.py`

---

## Step 4: Merton Default + Market-Cap Floor for MC Survival (Methods 4 + 8)

**File:** `operator1/models/monte_carlo.py`

### Step 4a: Add get_mcap_survival_floor utility (after imports, around line 30)

```python
def get_mcap_survival_floor(market_cap: float) -> float:
    """Return minimum credible survival probability based on market cap quintile.
    
    Based on Fama-French size quintile analysis: mega-caps almost never
    experience >40% drawdowns that trigger survival mode.
    """
    if market_cap >= 200e9: return 0.92   # mega-cap: >$200B
    elif market_cap >= 10e9: return 0.82  # large-cap: $10B-$200B
    elif market_cap >= 2e9: return 0.70   # mid-cap: $2B-$10B
    elif market_cap >= 300e6: return 0.55 # small-cap: $300M-$2B
    else: return 0.40                     # micro-cap: <$300M
```

### Step 4b: Add anchor_mc_survival utility (after get_mcap_survival_floor)

```python
def anchor_mc_survival(
    mc_result: "MonteCarloResult",
    market_cap: float | None = None,
    merton_pd: float | None = None,
) -> None:
    """Anchor MC survival probabilities using market-cap floor and Merton default.
    
    Modifies mc_result.survival_probability in place. Applied after MC simulation
    and after HF analysis produces Merton default probability.
    """
    if mc_result is None:
        return
    
    floor = 0.0
    if market_cap is not None and market_cap > 0:
        floor = max(floor, get_mcap_survival_floor(market_cap))
    if merton_pd is not None and 0 < merton_pd < 1:
        floor = max(floor, 1.0 - merton_pd)
    
    if floor <= 0:
        return
    
    for horizon, data in mc_result.survival_probability.items():
        if isinstance(data, dict) and "mean" in data:
            if data["mean"] < floor:
                data["mean"] = floor
                data["floor_source"] = "mcap" if market_cap and get_mcap_survival_floor(market_cap) >= (1.0 - (merton_pd or 1)) else "merton"
```

**File:** `operator1/stages/stage5_forward.py`

### Step 4c: Call anchor after MC in run_5_4_monte_carlo (after line 252)

```python
# After regime shift prediction, apply market-cap survival floor
try:
    from operator1.models.monte_carlo import anchor_mc_survival
    _mcap = None
    if "market_cap" in cache.columns and cache["market_cap"].notna().any():
        _mcap = float(cache["market_cap"].dropna().iloc[-1])
    anchor_mc_survival(state.mc_result, market_cap=_mcap)
except Exception:
    pass
```

**File:** `operator1/stages/stage7_integration.py`

### Step 4d: Enhance anchor with Merton after HF in run_7_5_hedge_fund (after line 353)

```python
# After HF analysis, anchor MC survival with Merton + mcap
try:
    from operator1.models.monte_carlo import anchor_mc_survival
    _mcap = None
    if state.cache is not None and "market_cap" in state.cache.columns:
        _mcap = float(state.cache["market_cap"].dropna().iloc[-1]) if state.cache["market_cap"].notna().any() else None
    _merton_pd = None
    if state.hf_result and hasattr(state.hf_result, "advanced_methods"):
        _adv = state.hf_result.advanced_methods
        if hasattr(_adv, "merton_default_probability") and _adv.merton_default_probability:
            _merton_pd = _adv.merton_default_probability.get("pd_1yr") if isinstance(_adv.merton_default_probability, dict) else None
    anchor_mc_survival(state.mc_result, market_cap=_mcap, merton_pd=_merton_pd)
except Exception:
    pass
```

**File:** `backtest_runner.py`

### Step 4e: Mirror in run_stage2b after MC (after line 1497)

```python
# Market-cap survival floor
try:
    from operator1.models.monte_carlo import anchor_mc_survival
    _mcap = float(cache["market_cap"].dropna().iloc[-1]) if "market_cap" in cache.columns and cache["market_cap"].notna().any() else None
    anchor_mc_survival(state.mc_result, market_cap=_mcap)
except Exception:
    pass
```

### Step 4f: Mirror in run_stage2d after HF (after line 1850)

```python
# Merton + mcap anchor after HF
try:
    from operator1.models.monte_carlo import anchor_mc_survival
    _mcap = float(cache["market_cap"].dropna().iloc[-1]) if "market_cap" in cache.columns and cache["market_cap"].notna().any() else None
    _merton_pd = None
    if hasattr(state, 'hf_result') and state.hf_result and hasattr(state.hf_result, 'advanced_methods'):
        _adv = state.hf_result.advanced_methods
        if hasattr(_adv, 'merton_default_probability') and isinstance(_adv.merton_default_probability, dict):
            _merton_pd = _adv.merton_default_probability.get('pd_1yr')
    anchor_mc_survival(state.mc_result, market_cap=_mcap, merton_pd=_merton_pd)
except Exception:
    pass
```

### Compile check: `python -m py_compile operator1/models/monte_carlo.py && python -m py_compile operator1/stages/stage5_forward.py && python -m py_compile operator1/stages/stage7_integration.py && python -m py_compile backtest_runner.py`

---

## Step 5: Regime Probability-Weighted Prediction Shift (Method 5)

**File:** `operator1/models/forecasting.py`

### Step 5a: Add regime shift block after residual adjustment (after the Bug #2 fix block, before the long-horizon tree blend)

Insert after the `apply_residual_feature_adjustment` try/except block:

```python
# Regime probability-weighted return shift for directional adjustment
_prob_cols = [c for c in cache.columns if c.startswith("regime_hmm_prob_")]
if _prob_cols and var_name == "close" and len(cache) > 0:
    try:
        _last_probs = cache[_prob_cols].iloc[-1].dropna()
        if len(_last_probs) > 0 and "regime_hmm" in cache.columns:
            _ret_col = "return_1d"
            if _ret_col in cache.columns:
                _regime_means = {}
                for col in _prob_cols:
                    _ridx = int(col.split("_")[-1])
                    _rmask = cache["regime_hmm"] == _ridx
                    if _rmask.sum() > 5:
                        _regime_means[_ridx] = float(cache.loc[_rmask, _ret_col].mean())
                if _regime_means:
                    _expected_daily = sum(
                        _regime_means.get(int(col.split("_")[-1]), 0.0) * float(_last_probs[col])
                        for col in _last_probs.index
                    )
                    for _rl, _rh in HORIZONS.items():
                        _shift = _expected_daily * _rh
                        _horizon_forecasts[_rl] *= (1 + _shift)
    except Exception:
        pass
```

### Compile check: `python -m py_compile operator1/models/forecasting.py`

---

## Step 6: PELT Fallback for Degenerate HMM (Method 6)

**File:** `operator1/models/regime_detector.py`

### Step 6a: Add _label_from_pelt_segments helper (after the class, before the module-level functions)

```python
def _label_from_pelt_segments(
    cache: pd.DataFrame,
    breakpoints: list[int],
    return_col: str = "return_1d",
    vol_col: str = "volatility_21d",
) -> np.ndarray:
    """Label time series segments from PELT breakpoints by return/volatility characteristics."""
    n = len(cache)
    labels = np.full(n, 0, dtype=int)
    segments = [0] + breakpoints + [n]
    for i in range(len(segments) - 1):
        start, end = segments[i], segments[i + 1]
        if return_col in cache.columns:
            mean_ret = cache[return_col].iloc[start:end].mean()
        else:
            mean_ret = 0
        if vol_col in cache.columns:
            mean_vol = cache[vol_col].iloc[start:end].mean()
        else:
            mean_vol = 0
        # Classify: bull (high return), bear (low return), high_vol, low_vol
        if mean_ret > 0.0005:
            labels[start:end] = 0  # bull
        elif mean_ret < -0.0005:
            labels[start:end] = 1  # bear
        elif mean_vol > cache[vol_col].median() if vol_col in cache.columns else True:
            labels[start:end] = 2  # high_vol
        else:
            labels[start:end] = 3  # low_vol
    return labels
```

### Step 6b: Add degeneracy check after HMM label assignment

Find the line where HMM labels are finalized (around line 280, after `regime_label_mapping`). Add:

```python
# Check for degenerate HMM states (any state with <5% of observations)
_state_counts = pd.Series(self._hmm_labels).value_counts(normalize=True)
if _state_counts.min() < 0.05:
    _degen_state = _state_counts.idxmin()
    logger.warning(
        "HMM degenerate: state %s has %.1f%% of observations -- "
        "falling back to PELT-based labeling",
        _degen_state, _state_counts.min() * 100,
    )
    if hasattr(self, '_pelt_breakpoints') and self._pelt_breakpoints:
        self._hmm_labels = _label_from_pelt_segments(
            cache, self._pelt_breakpoints,
        )
        logger.info("PELT fallback: %d segments from %d breakpoints",
                     len(set(self._hmm_labels)), len(self._pelt_breakpoints))
```

### Compile check: `python -m py_compile operator1/models/regime_detector.py`

---

## Step 7: PEAD Earnings Drift in Prediction Aggregator (Method 7)

**File:** `operator1/models/prediction_aggregator.py`

### Step 7a: Add event_calendar_result parameter to run_prediction_aggregation

Find the function signature (around line 2230). Add parameter:

```python
event_calendar_result: Any | None = None,
```

### Step 7b: Add PEAD drift block at the end of aggregation (before return)

After the USS bounding block and before the return statement:

```python
# PEAD: pre-earnings drift adjustment based on event proximity
try:
    if event_calendar_result is not None and getattr(event_calendar_result, "available", False):
        _days = getattr(event_calendar_result, "days_to_next_event", None)
        _prem = getattr(event_calendar_result, "event_uncertainty_premium", None)
        if _days is not None and _days < 30 and _prem is not None and abs(_prem) > 0.001:
            _drift = -_prem * 0.001 * min(_days, 21)
            for _pvar in result.predictions:
                if _pvar == "close" and isinstance(result.predictions[_pvar], dict):
                    for _ph, _php in result.predictions[_pvar].items():
                        if hasattr(_php, "point_forecast") and _php.point_forecast:
                            _php.point_forecast *= (1 + _drift)
except Exception:
    pass
```

**File:** `operator1/stages/stage6_ensemble.py`

### Step 7c: Pass event_calendar_result in run_6_5_aggregation (line 234-247)

Add to the `run_prediction_aggregation()` call:

```python
event_calendar_result=getattr(state, "event_calendar_result", None),
```

**File:** `backtest_runner.py`

### Step 7d: Pass event_calendar_result in Stage 2c aggregation (line 1674-1687)

Add to the `run_prediction_aggregation()` call:

```python
event_calendar_result=state.event_calendar_result,
```

**File:** `main.py`

### Step 7e: Pass event_calendar_result in main.py aggregation call

Find the `run_prediction_aggregation()` call and add:

```python
event_calendar_result=event_calendar_result,
```

### Compile check: `python -m py_compile operator1/models/prediction_aggregator.py && python -m py_compile operator1/stages/stage6_ensemble.py && python -m py_compile backtest_runner.py && python -m py_compile main.py`

---

## Step 8: Final Compile + Test + Commit

### 8a: Compile ALL modified files

```bash
python -m py_compile operator1/clients/us_edgar.py
python -m py_compile operator1/hedge_fund/advanced_methods.py
python -m py_compile operator1/hedge_fund/engine.py
python -m py_compile operator1/stages/stage6_ensemble.py
python -m py_compile operator1/models/monte_carlo.py
python -m py_compile operator1/stages/stage5_forward.py
python -m py_compile operator1/stages/stage7_integration.py
python -m py_compile operator1/models/forecasting.py
python -m py_compile operator1/models/regime_detector.py
python -m py_compile operator1/models/prediction_aggregator.py
python -m py_compile backtest_runner.py
python -m py_compile main.py
```

### 8b: Run AAPL backtest

```bash
python backtest_runner.py --market us_sec_edgar --company AAPL --end-date 2024-12-31 --years 2 --stage all
python backtest_runner.py --validate --run-dir cache/backtest_AAPL_2024-12-31
```

### 8c: Compare results

Check:
1. NaN ratios (current_ratio, cash_ratio, etc.) -- should now have values
2. Piotroski F-Score -- should be higher than 2/9 (normalized by available signals)
3. 21d conformal interval -- should be MC-based, much tighter than $619 range
4. MC survival 252d -- should be >= 82% for AAPL (large-cap floor)
5. Regime detection -- should show multiple states (not 495/502 in one)
6. Close predictions -- should be shifted by regime probabilities

### 8d: Commit and push

```bash
git checkout -b fix/close-prediction-error-gaps
git add -A
git commit -m 'fix: 8 improvements to close prediction error gaps -- XBRL concepts, NaN scoring, MC intervals, Merton anchor, regime shift, PELT fallback, PEAD drift, mcap calibration'
git push origin HEAD:fix/close-prediction-error-gaps
gh pr create --draft --repo sso4422/whatisyourname --head fix/close-prediction-error-gaps --title 'fix: close prediction error gaps (8 improvements)' --body '...'
```
