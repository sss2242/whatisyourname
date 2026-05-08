# Fix Guide: Annualization Regression + Missed Wiring

## Debug Scan Results

3 critical regression bugs + 5 missed wiring points found in the frequency-first pipeline implementation.

---

## BUG 1 (CRITICAL): _ANNUALIZE_MULT at D/W/M causes 252x regression

**File:** `operator1/features/derived_variables.py:49`

**Problem:** `_ANNUALIZE_MULT = {"D": 252.0, "W": 52.0, "M": 12.0, ...}` assumes flow variables at D/W/M freq are at a uniform per-period rate. But on the daily cache, flow variables are at MIXED scales:
- Some (revenue, net_income) are daily rates from the interpolator (~$84M/day)
- Some (ebit, ebitda) are raw quarterly values forward-filled (~$42.8B)
- There is NO way to know which is which at runtime

Multiplying by 252 on daily data makes every ratio WORSE than the original broken values:
- ROA before: `NI_daily / TA` = 0.0003 (too small by ~1000x)
- ROA after BUG: `NI_daily * 252 / TA` = 0.075 (wrong direction, depends on whether NI is rate or raw)
- If NI is raw quarterly ($24B): `24B * 252 / 364B` = 16.6 (absurd)

**Fix:** Set D/W/M multipliers to 1.0 (no annualization = backward compatible). Annualization ONLY makes sense at Q/A/S where values are at known filing scale.

```python
# BEFORE (WRONG):
_ANNUALIZE_MULT: dict[str, float] = {
    "D": 252.0, "W": 52.0, "M": 12.0, "Q": 4.0, "S": 2.0, "A": 1.0,
}

# AFTER (CORRECT):
_ANNUALIZE_MULT: dict[str, float] = {
    "D": 1.0, "W": 1.0, "M": 1.0, "Q": 4.0, "S": 2.0, "A": 1.0,
}
```

**Affected formulas in derived_variables.py:**
- `_compute_valuation`: PE = `close / (eps * _mult)`, EV/EBITDA = `EV / (ebitda * _mult)`, P/S = `mcap / (rev * _mult)`
- `_compute_roa`: `(NI * _mult) / TA`
- `_compute_profitability`: ROE = `(NI * _mult) / equity`
- `_compute_earnings_quality_signals`: accruals = `((NI-OCF) * _mult) / TA`

---

## BUG 2 (CRITICAL): Altman Z annualization regression at D freq

**File:** `operator1/models/financial_health.py:633`

**Problem:** Same `_annualize` dict with `"D": 252.0` inside `compute_altman_z_score`. x3 = `ebit * 252 / TA` and x5 = `revenue * 252 / TA` at D freq is a regression vs the original `ebit / TA`.

**Fix:** Set D/W/M to 1.0 in the local dict.

```python
# BEFORE (WRONG):
_annualize = {"D": 252.0, "W": 52.0, "M": 12.0, "Q": 4.0, "S": 2.0, "A": 1.0}

# AFTER (CORRECT):
_annualize = {"D": 1.0, "W": 1.0, "M": 1.0, "Q": 4.0, "S": 2.0, "A": 1.0}
```

---

## BUG 3 (CRITICAL): Runway period divisor regression at D freq

**File:** `operator1/models/financial_health.py:822`

**Problem:** `_months_in_period = {"D": 1/30, ...}` means at D freq: `monthly_burn = |OCF| / (1/30) = |OCF| * 30`. If OCF is a raw quarterly value ($52.5B), monthly_burn = $1.575T which is garbage. Before the change it was `|OCF| / 12.0` which at least assumed annual.

**Fix:** Set D/W/M to use the old backward-compatible divisor (12.0 for D, assuming the forward-filled OCF is roughly annual-scale in the daily cache).

```python
# BEFORE (WRONG):
_months_in_period = {"A": 12.0, "S": 6.0, "Q": 3.0, "M": 1.0, "W": 7/30, "D": 1/30}

# AFTER (CORRECT):
_months_in_period = {"A": 12.0, "S": 6.0, "Q": 3.0, "M": 1.0, "W": 1.0, "D": 12.0}
```

At D freq: `monthly_burn = |OCF| / 12.0` (backward compatible, assumes OCF in cache is annual-ish from forward-fill of quarterly or annual filing).

---

## MISSED WIRING (5 call sites not passing freq)

All use default `freq="D"` which is CORRECT for the daily cache after fixing BUG 1-3. No code change needed for backward compatibility. However, for clarity and documentation, these should be explicit:

| # | File:Line | Call | Status |
|---|-----------|------|--------|
| 1 | `main.py:1507` | `compute_derived_variables(cache)` | OK (default D) |
| 2 | `main.py:1547` | `compute_company_survival_flag(cache)` | OK (default D, skips fcf_yield) |
| 3 | `main.py:1703` | `compute_financial_health(cache, ...)` | OK after BUG 1-3 fix |
| 4 | `backtest_runner.py:742` | `compute_derived_variables(cache)` | OK (default D) |
| 5 | `backtest_runner.py:828` | `compute_financial_health(cache, ...)` | OK after BUG 1-3 fix |

**Decision:** No code changes needed for missed wiring. Default "D" is correct.

---

## MINOR: Stage order inconsistency

`_compute_solvency` (stage 2) references `ebitda_ttm_asof` which is created by `_compute_ttm_and_growth` (stage 9). At D freq, the code falls back to raw `ebitda` via `df.get("ebitda_ttm_asof", ebitda)`. This is correct behavior. No fix needed.

---

## Execution Order

1. Fix `_ANNUALIZE_MULT` in `derived_variables.py:49` -- set D/W/M to 1.0
2. Fix `_annualize` dict in `financial_health.py:648` (Altman Z) -- set D/W/M to 1.0
3. Fix `_months_in_period` dict in `financial_health.py:838` (runway) -- set D to 12.0, W to 1.0, M to 1.0
4. Verify: `python3.12 -c "from operator1.features.derived_variables import _ANNUALIZE_MULT; print(_ANNUALIZE_MULT)"`
5. Commit + push

## Expected Results After Fix

| Metric | Before Our Changes | After BUG (wrong) | After Fix (correct) |
|--------|-------------------|-------------------|---------------------|
| D-freq PE | 178.87 | 250/(1.4*252)=0.71 | 178.87 (backward compat, fixed by Q forward-fill) |
| D-freq ROA | 0.0003 | 0.0003*252=0.076 | 0.0003 (backward compat, fixed by Q forward-fill) |
| D-freq Altman x3 | ebit/TA | ebit*252/TA | ebit/TA (backward compat) |
| Q-freq PE | N/A (not computed before) | 250/(1.4*4)=44.6 | 250/(1.4*4)=44.6 (CORRECT) |
| Q-freq ROA | N/A | NI*4/TA=28% | NI*4/TA=28% (CORRECT) |

The D-freq ratios stay backward compatible (broken but same as before). The Q-freq ratios are CORRECT. Stage 2.F forward-fills the correct Q values into the daily cache, replacing the broken D values.
