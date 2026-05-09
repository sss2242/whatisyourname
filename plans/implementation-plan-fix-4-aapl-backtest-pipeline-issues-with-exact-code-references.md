# Implementation Plan: Fix 4 AAPL Backtest Pipeline Issues

*Based on debug scan of all callers, inputs, outputs, and data flow paths.*

---

## Fix 1: Burn-Out Regime Distributions Filter to Return Variable Only

**Problem:** `ExponentialGradientWeightLearner` stores actuals for ALL variables (close=$250, cash=$30B, return_1d=0.001) in `_regime_weighted_returns`. MC receives billion-scale distributions and overflows.

**Data flow traced:**
```
forward_pass predictions_log entries have {"variable": var_name, "actual": value} at L4320
  -> burn-out iterates ALL entries at L4873
  -> learner.update(regime, per_model, actual) at L4879
  -> _regime_weighted_returns[regime].append(actual) at L4608 (NO variable filter)
  -> get_regime_distributions() at L4627 computes mean/std across ALL variables
  -> BurnoutResult.regime_distributions at L4887
  -> stage5_forward.py L190-201 passes to run_monte_carlo()
  -> monte_carlo.py L1354-1364 overrides RegimeDistribution with billion-scale values
  -> exp(cumsum(billions)) overflow at L748
```

### Changes (2 files, ~15 lines)

**File 1: `operator1/models/forecasting.py`**

**Change 1a** -- L4873-4879: Filter predictions_log to return variable before feeding learner

```python
# BEFORE (L4873-4879):
for entry in predictions_log:
    regime = entry.get("regime", "unknown")
    per_model = entry.get("per_model", {})
    actual = entry.get("actual", 0.0)
    if isinstance(per_model, dict) and per_model:
        learner.update(regime, per_model, actual)

# AFTER:
for entry in predictions_log:
    regime = entry.get("regime", "unknown")
    per_model = entry.get("per_model", {})
    actual = entry.get("actual", 0.0)
    variable = entry.get("variable", "")
    if isinstance(per_model, dict) and per_model:
        learner.update(regime, per_model, actual, variable=variable)
```

**Change 1b** -- L4570 (step/update method signature): Add variable parameter

```python
# BEFORE:
def update(self, regime, per_model_preds, actual):

# AFTER:
def update(self, regime, per_model_preds, actual, variable: str = ""):
```

**Change 1c** -- L4605-4608: Store variable name alongside actual

```python
# BEFORE:
self._regime_weighted_returns[regime] = []
...
self._regime_weighted_returns[regime].append(actual)

# AFTER:
self._regime_weighted_returns[regime] = []
...
self._regime_weighted_returns[regime].append({"actual": actual, "variable": variable})
```

**Change 1d** -- L4620-4641: Filter to return_1d in get_regime_distributions()

```python
# BEFORE:
def get_regime_distributions(self) -> dict:
    for regime, returns in self._regime_weighted_returns.items():
        if len(returns) >= 5:
            arr = np.array(returns)

# AFTER:
def get_regime_distributions(self, variable: str = "return_1d") -> dict:
    for regime, entries in self._regime_weighted_returns.items():
        filtered = [e["actual"] for e in entries
                    if isinstance(e, dict) and e.get("variable") == variable]
        if not filtered:
            # Fallback: use entries where actual is in return-like range
            filtered = [e["actual"] if isinstance(e, dict) else e
                        for e in entries
                        if (isinstance(e, dict) and abs(e.get("actual", 999)) < 1.0)
                        or (not isinstance(e, dict) and abs(e) < 1.0)]
        if len(filtered) >= 5:
            arr = np.array(filtered)
```

**File 2: `operator1/models/monte_carlo.py`**

**Change 1e** -- L1354-1364: Add scale validation guard

```python
# AFTER L1356 "for regime, params in burnout_distributions.items():"
# Add before the override:
if abs(params.get("mean", 0)) > 1.0 or abs(params.get("std", 0)) > 1.0:
    logger.warning(
        "MC: rejecting burn-out override for regime '%s' "
        "(mean=%.2f, std=%.2f -- non-return-scale values detected)",
        regime, params.get("mean", 0), params.get("std", 0),
    )
    continue
```

---

## Fix 2: Cash Adequacy Absolute Floor for Mega-Caps

**Problem:** Cash/mcap ratio test (>5%) fails for mega-caps where $30B cash / $3.8T mcap = 0.79%.

**Data flow:** Single location, no downstream callers affected.

### Changes (2 files, ~8 lines)

**File 1: `operator1/analysis/survival_mode.py`**

**Change 2a** -- L223-224: Add absolute cash floor

```python
# BEFORE:
_cash_adequate = (_cash > 0) & (_mcap > 0) & (_cash / _mcap > 0.05)

# AFTER:
try:
    from operator1.scoring_weights import get_weight
    _cash_ratio_thresh = float(get_weight("survival_thresholds.cash_adequacy_ratio", 0.05))
    _cash_abs_thresh = float(get_weight("survival_thresholds.cash_adequacy_absolute", 10_000_000_000))
except Exception:
    _cash_ratio_thresh = 0.05
    _cash_abs_thresh = 10_000_000_000
_cash_ratio_ok = (_cash > 0) & (_mcap > 0) & (_cash / _mcap > _cash_ratio_thresh)
_cash_absolute_ok = _cash > _cash_abs_thresh
_cash_adequate = _cash_ratio_ok | _cash_absolute_ok
```

**File 2: `config/scoring_weights.yml`**

**Change 2b** -- Under `survival_thresholds`:

```yaml
survival_thresholds:
  current_ratio: 1.0
  # ... existing keys ...
  cash_adequacy_ratio: 0.05           # Cash/market_cap ratio floor
  cash_adequacy_absolute: 10000000000  # $10B absolute cash floor for mega-caps
```

---

## Fix 3: Sector-Aware Survival Thresholds

**Problem:** Global `current_ratio < 1.0` triggers survival for tech mega-caps with deliberate lean balance sheets.

**Callers that need `sector` parameter (20 total, 9 in production code, 7 in tests):**

| # | File | Line | Current call | Needs sector |
|---|------|------|-------------|-------------|
| 1 | `backtest_runner.py` | 771 | `compute_company_survival_flag(cache)` | Yes - `state.target_profile.get("sector", "")` |
| 2 | `backtest_runner.py` | 979 | `compute_company_survival_flag(cache, thresholds=adapted)` | Yes - same |
| 3 | `main.py` | 1547 | `compute_company_survival_flag(cache)` | Yes - `target_profile.get("sector", "")` |
| 4 | `main.py` | 2326 | `compute_company_survival_flag(cache, ...)` | Yes - same |
| 5 | `survival_mode.py` | 863 | `compute_company_survival_flag(df)` | No - internal test helper |
| 6 | `model_tests.py` | 161 | `compute_company_survival_flag(cache.copy())` | No - synthetic test data |
| 7 | `stage2_freq_pipeline.py` | 197 | `compute_company_survival_flag(cache, freq="Q")` | Yes - from state |
| 8 | `multi_frequency_runner.py` | 234 | `compute_company_survival_flag(cache, freq=freq)` | Yes - from kwargs |
| 9 | Tests (7 calls) | various | `compute_company_survival_flag(df)` | No - keep as-is |

### Changes (5 files, ~30 lines)

**File 1: `operator1/analysis/survival_mode.py`**

**Change 3a** -- L71-75: Add `sector` parameter

```python
# BEFORE:
def compute_company_survival_flag(
    df: pd.DataFrame,
    thresholds: dict[str, float] | None = None,
    freq: str = "D",
) -> pd.Series:

# AFTER:
def compute_company_survival_flag(
    df: pd.DataFrame,
    thresholds: dict[str, float] | None = None,
    freq: str = "D",
    sector: str = "",
) -> pd.Series:
```

**Change 3b** -- L96 (after `t = thresholds or _COMPANY_THRESHOLDS`): Apply sector overrides

```python
t = dict(thresholds or _COMPANY_THRESHOLDS)  # copy to avoid mutating defaults

# Apply sector overrides from scoring_weights.yml
if sector:
    try:
        from operator1.scoring_weights import get_weight
        _sector_key = sector.lower().replace(" ", "_").replace("&", "and")
        _overrides = get_weight(f"survival_thresholds.sector_overrides.{_sector_key}", {})
        if isinstance(_overrides, dict):
            for k, v in _overrides.items():
                # Map config keys to threshold dict keys
                if k == "current_ratio":
                    t["current_ratio_lt"] = float(v)
                elif k == "debt_to_equity":
                    t["debt_to_equity_abs_gt"] = float(v)
    except Exception:
        pass
```

**File 2: `config/scoring_weights.yml`**

**Change 3c** -- Under `survival_thresholds`:

```yaml
survival_thresholds:
  current_ratio: 1.0
  # ... existing keys ...
  sector_overrides:
    technology:
      current_ratio: 0.7
      debt_to_equity: 4.0
    communication_services:
      current_ratio: 0.8
```

**File 3: `backtest_runner.py`**

**Change 3d** -- L771, L979: Pass sector

```python
# L771:
cache["company_survival_mode_flag"] = compute_company_survival_flag(
    cache, sector=state.target_profile.get("sector", ""))

# L979:
cache["company_survival_mode_flag"] = compute_company_survival_flag(
    cache, thresholds=adapted, sector=state.target_profile.get("sector", ""))
```

**File 4: `main.py`**

**Change 3e** -- L1547, L2326: Pass sector

```python
# L1547:
cache["company_survival_mode_flag"] = compute_company_survival_flag(
    cache, sector=target_profile.get("sector", ""))

# L2326: already has freq="Q", add sector
cache["company_survival_mode_flag"] = compute_company_survival_flag(
    cache, freq="Q", sector=target_profile.get("sector", ""))
```

**File 5: `operator1/stages/stage2_freq_pipeline.py`**

**Change 3f** -- L197: Pass sector from state

```python
cache["company_survival_mode_flag"] = compute_company_survival_flag(
    cache, freq="Q", sector=state.target_profile.get("sector", ""))
```

---

## Fix 4: SIC Peer Fallback to Static Map + yfinance

**Problem:** SEC EDGAR SIC peer lookup returns 0 matches for AAPL (SIC 3571) because competitors have different SIC codes.

**Callers:** Only `entity_discovery.py` L289 calls `get_peers()` for peer fallback.

### Changes (2 files, ~40 lines)

**File 1: `operator1/clients/us_edgar.py`**

**Change 4a** -- After L900 (fallback section): Add static peer map + yfinance sector fallback

```python
# After existing "Fallback to 2-digit SIC" block at ~L900:

# Fallback 2: Static peer map for major tickers
_STATIC_PEERS = {
    "AAPL": ["MSFT", "GOOG", "AMZN", "META", "NVDA"],
    "MSFT": ["AAPL", "GOOG", "AMZN", "ORCL", "CRM"],
    "GOOG": ["AAPL", "MSFT", "META", "AMZN", "NFLX"],
    "AMZN": ["AAPL", "MSFT", "GOOG", "WMT", "SHOP"],
    "META": ["GOOG", "AAPL", "SNAP", "PINS", "MSFT"],
    "NVDA": ["AMD", "INTC", "AAPL", "MSFT", "TSM"],
    "TSLA": ["F", "GM", "RIVN", "NIO", "BYD"],
}

if len(peers) < 3 and target_ticker in _STATIC_PEERS:
    static = [p for p in _STATIC_PEERS[target_ticker] if p != target_ticker]
    peers.extend(static[:max(0, 5 - len(peers))])
    logger.info("Static peer map fallback: %d peers for %s", len(static), target_ticker)

# Fallback 3: yfinance sector peers
if len(peers) < 3:
    try:
        import yfinance as yf
        sector_obj = yf.Sector(yf.Ticker(target_ticker).info.get("sectorKey", ""))
        if hasattr(sector_obj, "top_companies") and sector_obj.top_companies is not None:
            yf_peers = [str(t) for t in sector_obj.top_companies.index
                        if str(t).upper() != target_ticker][:10]
            peers.extend(yf_peers[:max(0, 5 - len(peers))])
            logger.info("yfinance sector fallback: %d peers", len(yf_peers))
    except Exception as exc:
        logger.debug("yfinance peer fallback failed: %s", exc)
```

**File 2: `operator1/steps/entity_discovery.py`**

**Change 4b** -- L310 area: Also check static peers when LLM + SIC both fail

```python
# After the existing "Static competitor fallback" block:
# Also try us_edgar static peer map
if not competitors and hasattr(pit_client, "get_peers"):
    try:
        static_peers = pit_client.get_peers(target_isin)
        if static_peers:
            # ... existing resolution logic ...
    except Exception:
        pass
```

---

## Execution Checklist

```
[ ] Fix 1a: forecasting.py L4873-4879 -- pass variable to learner.update()
[ ] Fix 1b: forecasting.py L4570 -- add variable param to update()
[ ] Fix 1c: forecasting.py L4605-4608 -- store variable in _regime_weighted_returns
[ ] Fix 1d: forecasting.py L4620-4641 -- filter get_regime_distributions()
[ ] Fix 1e: monte_carlo.py L1354 -- add scale validation guard
[ ] Fix 2a: survival_mode.py L223-224 -- absolute cash floor
[ ] Fix 2b: scoring_weights.yml -- add cash_adequacy_absolute key
[ ] Fix 3a: survival_mode.py L71 -- add sector parameter
[ ] Fix 3b: survival_mode.py L96 -- apply sector overrides
[ ] Fix 3c: scoring_weights.yml -- add sector_overrides
[ ] Fix 3d: backtest_runner.py L771, L979 -- pass sector
[ ] Fix 3e: main.py L1547, L2326 -- pass sector
[ ] Fix 3f: stage2_freq_pipeline.py L197 -- pass sector
[ ] Fix 4a: us_edgar.py L900 -- static peers + yfinance fallback
[ ] Fix 4b: entity_discovery.py L310 -- use static peers from get_peers()
[ ] Verify: AST parse all modified files
[ ] Verify: Import check all modified modules
[ ] Verify: Run AAPL backtest to confirm fixes
```

## Files Changed Summary

| File | Fix | Lines Changed |
|------|-----|---------------|
| `operator1/models/forecasting.py` | 1a,1b,1c,1d | ~20 |
| `operator1/models/monte_carlo.py` | 1e | ~7 |
| `operator1/analysis/survival_mode.py` | 2a,3a,3b | ~20 |
| `config/scoring_weights.yml` | 2b,3c | ~10 |
| `backtest_runner.py` | 3d | ~4 |
| `main.py` | 3e | ~4 |
| `operator1/stages/stage2_freq_pipeline.py` | 3f | ~2 |
| `operator1/clients/us_edgar.py` | 4a | ~25 |
| `operator1/steps/entity_discovery.py` | 4b | ~5 |
| **Total** | **9 files** | **~97 lines** |
