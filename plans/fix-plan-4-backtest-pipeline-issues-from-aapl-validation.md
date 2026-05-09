# Fix Plan: 4 Backtest Pipeline Issues from AAPL Validation

*From AAPL backtest 2024-12-31 -> 2025 validation. Root causes verified in source code.*

---

## Fix 1: Burn-Out Regime Distributions Pass Wrong-Scale Values to MC

**Severity:** CRITICAL -- causes MC overflow and 0% survival probability
**File:** `operator1/models/forecasting.py` line 4608
**Root cause:** The `ExponentialGradientLearner.step()` stores `actual` (the raw forecasted variable value) into `_regime_weighted_returns`. When burn-out runs across multiple variables (close=$250, cash=$30B, return_1d=0.001), the distributions mix all scales. MC then uses these as daily return parameters, causing `exp(cumsum(30B))` overflow.

### Fix

In `get_regime_distributions()` (line 4620), filter to only include observations from the return variable. Add a `target_variable` parameter:

```python
# forecasting.py line 4600-4608
# BEFORE:
self._regime_weighted_returns[regime].append(actual)

# AFTER:
self._regime_weighted_returns[regime].append({
    'variable': self._current_variable,
    'actual': actual,
})
```

```python
# forecasting.py line 4620
# BEFORE:
def get_regime_distributions(self) -> dict:
    for regime, returns in self._regime_weighted_returns.items():
        arr = np.array(returns)

# AFTER:
def get_regime_distributions(self, variable: str = 'return_1d') -> dict:
    for regime, entries in self._regime_weighted_returns.items():
        # Filter to target variable only
        arr = np.array([e['actual'] for e in entries if e['variable'] == variable])
```

Also add a safety bound in MC at the override point (line 1357-1364):

```python
# monte_carlo.py line 1357
# Add scale validation before override
if abs(params['mean']) > 1.0 or abs(params['std']) > 1.0:
    logger.warning(
        'MC: rejecting burn-out override for regime %s '
        '(mean=%.2f, std=%.2f -- values suggest non-return variable leak)',
        regime, params['mean'], params['std'],
    )
    continue
```

| Step | File | Change |
|------|------|--------|
| 1a | `operator1/models/forecasting.py:4602-4608` | Store variable name alongside actual value |
| 1b | `operator1/models/forecasting.py:4620-4641` | Add `variable` filter parameter to `get_regime_distributions()` |
| 1c | `operator1/models/forecasting.py:4887` | Pass `variable='return_1d'` when calling `get_regime_distributions()` |
| 1d | `operator1/models/monte_carlo.py:1357` | Add scale validation guard before override |

---

## Fix 2: Cash Adequacy Floor Too High for Mega-Caps

**Severity:** MAJOR -- causes Apple (and all mega-cap tech) to be flagged as distressed
**File:** `operator1/analysis/survival_mode.py` line 224
**Root cause:** Cash adequacy threshold is `cash/market_cap > 5%`. Apple has $30B/$3.8T = 0.79%. The ratio test works for small/mid caps but fails for mega-caps where absolute cash is enormous but the ratio is small due to trillion-dollar market caps.

### Fix

Add an absolute cash floor alongside the ratio check:

```python
# survival_mode.py line 223-224
# BEFORE:
_cash_adequate = (_cash > 0) & (_mcap > 0) & (_cash / _mcap > 0.05)

# AFTER:
_cash_ratio_adequate = (_cash > 0) & (_mcap > 0) & (_cash / _mcap > 0.05)
_cash_absolute_adequate = _cash > 10_000_000_000  # $10B absolute floor
_cash_adequate = _cash_ratio_adequate | _cash_absolute_adequate
```

Make the $10B threshold configurable via `scoring_weights.yml`:

```yaml
survival_thresholds:
  cash_adequacy_ratio: 0.05
  cash_adequacy_absolute: 10000000000  # $10B absolute floor for mega-caps
```

| Step | File | Change |
|------|------|--------|
| 2a | `operator1/analysis/survival_mode.py:224` | Add absolute cash floor with OR logic |
| 2b | `operator1/analysis/survival_mode.py:220` | Read both thresholds from scoring_weights |
| 2c | `config/scoring_weights.yml` | Add `cash_adequacy_absolute` key |

---

## Fix 3: Sector-Aware Survival Thresholds

**Severity:** MAJOR -- current_ratio=0.92 triggers survival for Apple despite being a deliberate capital structure choice
**File:** `operator1/analysis/survival_mode.py` line 52
**Root cause:** Single global `current_ratio < 1.0` threshold with no per-sector override. Tech mega-caps (Apple, Google, Meta, Amazon) deliberately run current ratios below 1.0 -- their current liabilities include deferred revenue and short-term commercial paper that are continuously refinanced, not genuine liquidity crises.

### Fix

Add sector override map in `scoring_weights.yml` and consult it in `compute_company_survival_flag()`:

```yaml
# scoring_weights.yml
survival_thresholds:
  current_ratio: 1.0  # global default
  sector_overrides:
    technology:
      current_ratio: 0.7
    communication_services:
      current_ratio: 0.8
```

```python
# survival_mode.py -- new parameter
def compute_company_survival_flag(
    df: pd.DataFrame,
    thresholds: dict | None = None,
    freq: str = 'D',
    sector: str = '',  # NEW
) -> pd.Series:
    t = thresholds or _COMPANY_THRESHOLDS

    # Apply sector overrides from scoring_weights.yml
    if sector:
        try:
            from operator1.scoring_weights import get_weight
            sector_key = sector.lower().replace(' ', '_')
            sector_overrides = get_weight(
                f'survival_thresholds.sector_overrides.{sector_key}', {}
            )
            if sector_overrides:
                for k, v in sector_overrides.items():
                    t_key = f'{k}_lt' if k in ('current_ratio', 'fcf_yield') else f'{k}_gt'
                    t[t_key] = float(v)
        except Exception:
            pass
```

Wire the sector parameter from the caller:

```python
# backtest_runner.py (and main.py) where compute_company_survival_flag is called:
cache['company_survival_mode_flag'] = compute_company_survival_flag(
    cache, freq='Q',
    sector=state.target_profile.get('sector', ''),  # NEW
)
```

| Step | File | Change |
|------|------|--------|
| 3a | `config/scoring_weights.yml` | Add `sector_overrides` sub-key under `survival_thresholds` |
| 3b | `operator1/analysis/survival_mode.py:71` | Add `sector` parameter to function signature |
| 3c | `operator1/analysis/survival_mode.py:96` | Apply sector overrides to threshold dict |
| 3d | `backtest_runner.py` (6 call sites) | Pass `sector=state.target_profile.get('sector', '')` |
| 3e | `main.py` (3 call sites) | Pass `sector=target_profile.get('sector', '')` |
| 3f | `operator1/stages/stage2_freq_pipeline.py:197` | Pass sector from state |

---

## Fix 4: SIC Peer Lookup Fallback to yfinance Sector

**Severity:** MODERATE -- disables all peer-dependent features when SIC yields 0 matches
**File:** `operator1/clients/us_edgar.py` line 864
**Root cause:** `get_peers()` only uses SIC code matching. Apple's SIC 3571 returns 0 peers because competitors have different SIC codes (MSFT=7372, GOOG=7375, AMZN=5961).

### Fix

Add a yfinance sector peer fallback when SIC yields fewer than 3 matches:

```python
# us_edgar.py get_peers() -- after SIC lookup
if len(peers) < 3 and not self._yf_peer_attempted:
    self._yf_peer_attempted = True
    try:
        import yfinance as yf
        ticker_obj = yf.Ticker(target_ticker)
        info = ticker_obj.info or {}
        sector = info.get('sector', '')
        industry = info.get('industry', '')
        # Use yfinance recommended symbols as peer candidates
        recs = getattr(ticker_obj, 'recommendations', None)
        # Also try screener for same sector
        if hasattr(yf, 'Sector') and sector:
            sector_obj = yf.Sector(sector)
            top_companies = sector_obj.top_companies
            if top_companies is not None:
                yf_peers = [
                    str(idx) for idx in top_companies.index
                    if str(idx).upper() != target_ticker
                ][:10]
                peers.extend(yf_peers[:max(0, 5 - len(peers))])
                logger.info(
                    'yfinance sector fallback: %d peers from sector=%s',
                    len(yf_peers), sector,
                )
    except Exception as exc:
        logger.debug('yfinance peer fallback failed: %s', exc)
```

Also add a static tech-peer mapping as an ultimate fallback:

```python
_STATIC_PEER_MAP = {
    'AAPL': ['MSFT', 'GOOG', 'AMZN', 'META', 'NVDA'],
    'MSFT': ['AAPL', 'GOOG', 'AMZN', 'META', 'ORCL'],
    'GOOG': ['AAPL', 'MSFT', 'META', 'AMZN', 'NFLX'],
    # Add more as needed
}
```

| Step | File | Change |
|------|------|--------|
| 4a | `operator1/clients/us_edgar.py:900` | Add yfinance sector fallback after SIC lookup |
| 4b | `operator1/clients/us_edgar.py` (top) | Add static peer map for major tickers |
| 4c | `operator1/steps/entity_discovery.py` | Use static peers when both LLM and SIC fail |

---

## Implementation Order

```
Fix 1 (burn-out scale) -> highest impact, prevents MC overflow
Fix 2 (cash adequacy)  -> enables proper mega-cap handling
Fix 3 (sector thresholds) -> reduces false survival triggers
Fix 4 (SIC fallback)   -> enables peer-dependent features
```

Fixes 1-3 should be in a single PR (they interact: fixing survival triggers changes hierarchy weights which changes FH composite which changes MC inputs). Fix 4 can be a separate PR.

## Estimated Changes

| Fix | Files | Lines |
|-----|-------|-------|
| Fix 1 | forecasting.py + monte_carlo.py | ~25 |
| Fix 2 | survival_mode.py + scoring_weights.yml | ~10 |
| Fix 3 | survival_mode.py + scoring_weights.yml + backtest_runner.py + main.py + stage2_freq_pipeline.py | ~40 |
| Fix 4 | us_edgar.py + entity_discovery.py | ~35 |
| **Total** | **7 files** | **~110 lines** |

## Expected Outcome After Fixes

For the AAPL backtest:
- Survival flag should drop from 83.5% to ~5-10% (only drawdown/vol spikes)
- FH composite should rise from 33 (Weak) to ~65-75 (Healthy)
- MC should produce valid survival probabilities (not 0%)
- Close predictions should show directional accuracy instead of flat ~$250
- Entity discovery should find 5+ tech peers via yfinance fallback
