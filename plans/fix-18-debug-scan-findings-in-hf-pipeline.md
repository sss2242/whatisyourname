# Fix 18 Debug Scan Findings in HF Pipeline

All findings from the debug scan, grouped by priority and implementation order.

---

## Phase 1: Logic Bug Fixes (B1-B5) -- CRITICAL

These produce wrong results. Must fix before any wiring.

### B1: FCF quality inflated for loss-making companies

**File:** `operator1/hedge_fund/fcf_quality.py` line ~48
**Bug:** When NI is negative, OCF/NI ratio is positive (neg/neg) and scores as "healthy ~1.0"
**Fix:**
```python
# Before computing ratio, check NI sign
if float(ni_vals.mean()) < 0:
    # Loss-making company: OCF/NI ratio is meaningless
    # Score based on whether OCF is at least positive
    ocf_mean = float(ocf_vals.mean())
    if ocf_mean > 0:
        ocf_ni_score = 40.0  # losing money but generating cash
    else:
        ocf_ni_score = 10.0  # losing money AND burning cash
else:
    # Normal path: ratio scoring
    ...existing code...
```

### B2: Dividend burn risk=1 for negative FCF + dividends

**File:** `operator1/hedge_fund/engine.py` `_compute_dividend_burn()` line ~115
**Bug:** `min(100, ratio * 60)` produces negative when FCF < 0 and divs > 0
**Fix:**
```python
if len(div_coverage) > 0:
    mean_cov = float(div_coverage.mean())
    result.dividend_fcf_coverage = mean_cov
    # CRITICAL: if FCF is negative and paying dividends, automatic high risk
    mean_fcf = float(fcf.mean())
    mean_div = float(div_abs.mean())
    if mean_fcf < 0 and mean_div > 0:
        div_score = 95.0  # paying dividends from debt/asset liquidation
    elif mean_cov > 1.5:
        div_score = min(100, mean_cov * 40)  # payout > 150% of FCF
    else:
        div_score = min(100, max(0, mean_cov * 60))
```

### B3: Leverage stress -60x for negative EBITDA

**File:** `operator1/hedge_fund/engine.py` `_compute_leverage_stress()` line ~213
**Bug:** Check is `abs(ebitda) < 1e-6` which misses negative EBITDA
**Fix:**
```python
if not all([debt, ebitda, rev]):
    return result
# CRITICAL: negative EBITDA = automatic distress
if ebitda <= 0:
    result.base_case = StressScenario(
        name="base_case",
        debt_to_ebitda=None,  # meaningless when negative
        interest_coverage=safe_divide(ebit, ie) if ebit and ie else None,
        covenant_breach=True,  # automatic breach
    )
    result.base_case.cash_runway_months = safe_divide(
        (extract_latest_value(balance_df, "cash_and_equivalents") or 0),
        abs(ebitda) / 12,
    )
    # Skip revenue_miss and systemic (already in crisis)
    result.revenue_miss = result.base_case
    result.systemic_crisis = result.base_case
    result.refinancing_risk = True
    result.narrative = f"Negative EBITDA ({ebitda:.0f}): automatic distress"
    result.available = True
    return result
```

### B4: Scorecard B+ for distressed company

**File:** `operator1/hedge_fund/engine.py` `_build_scorecard()`
**Bug:** Cascading from B1+B2+B3. Fix B1-B3 and the scorecard automatically corrects.
**Additional safeguard:** Add a distress override:
```python
# After computing overall score, check for hard distress signals
if (hf.leverage_stress.available and hf.leverage_stress.base_case.covenant_breach
        and hf.dividend_burn.risk_score > 80
        and hf.fcf_quality.score < 30):
    # Override: company is clearly in distress regardless of other scores
    sc.investment_grade = min(sc.investment_grade, "D")  # cap at D
    sc.conviction = min(sc.conviction, 2)
```

### B5: Position signal always 0 without forecast_result

**File:** `operator1/hedge_fund/engine.py` `_compute_position_signal()` line ~425
**Bug:** No fallback alpha when forecast is unavailable
**Fix:**
```python
# Alpha base from forecast OR fallback from scorecard
alpha = 0.0
if forecast_result is not None and hasattr(forecast_result, "forecasts"):
    ...existing code...

# Fallback: derive alpha from scorecard grade
if alpha == 0.0 and hf.scorecard.available:
    grade_to_alpha = {
        "A+": 0.08, "A": 0.06, "B+": 0.03, "B": 0.01,
        "C+": 0.0, "C": -0.01, "D": -0.03, "F": -0.06,
    }
    alpha = grade_to_alpha.get(hf.scorecard.investment_grade, 0.0)
```

---

## Phase 2: Data Flow Fixes (D1-D3)

### D1: fcf_quality.py missing balance_df parameter

**File:** `operator1/hedge_fund/fcf_quality.py`
**Bug:** Function tries to get total_assets from income_df (never there). Needs balance_df.
**Fix:** Add `balance_df` parameter:
```python
def compute_fcf_quality(
    income_df: pd.DataFrame,
    cashflow_df: pd.DataFrame,
    balance_df: pd.DataFrame | None = None,  # NEW
    cache: pd.DataFrame | None = None,
) -> FCFQualityResult:
```
And use it for total_assets:
```python
if balance_df is not None:
    ta_series = extract_quarterly_series(balance_df, "total_assets", n_periods)
```
Update call in engine.py:
```python
hf.fcf_quality = compute_fcf_quality(income_df, cashflow_df, balance_df, cache)
```

### D2: Multi-frequency results not consumed

**File:** `operator1/hedge_fund/engine.py`
**Bug:** `multi_frequency_result` parameter accepted but unused
**Fix:** Pass to modules that benefit:
- `_compute_momentum()`: use MF weekly trend for price momentum divergence
- `_compute_dcf()`: use MF annual forecast bounds for terminal growth constraint
- `_compute_leverage_stress()`: use MF weekly rate environment for refinancing

For now, add a `_extract_mf_context()` helper that safely pulls per-frequency summaries:
```python
def _extract_mf_context(mf_result, frequency: str) -> dict:
    if mf_result is None or not hasattr(mf_result, "results"):
        return {}
    freq_result = mf_result.results.get(frequency)
    if freq_result is None:
        return {}
    return {
        "regime": getattr(freq_result, "regime_label", "unknown"),
        "trend": getattr(freq_result, "trend_direction", "flat"),
        "survival": getattr(freq_result, "survival_probability", 1.0),
    }
```

### D3: Dividend burn WC index mismatch

**File:** `operator1/hedge_fund/engine.py` `_compute_dividend_burn()`
**Bug:** WC uses balance_df dates but the `common` index is from cashflow_df dates
**Fix:** Compute WC independently with its own date alignment:
```python
# WC computation uses its own index (balance sheet dates)
if len(ca) >= 4 and len(cl) >= 4:
    wc_common = ca.index.intersection(cl.index)  # already correct
    # Don't intersect with `common` from cashflow
```
Actually this is already correct on re-examination -- `ca` and `cl` use their own index intersection. The bug was a false positive. Mark as NOT A BUG.

---

## Phase 3: Wiring (W1-W7) -- Connect to Pipeline

### W1: Wire into main.py Step 6-HF

**File:** `main.py` (after Step 6.7 multi-frequency, before Step 7 profile)
**Add ~30 lines:**
```python
# Step 6-HF: Hedge Fund Analysis
hf_result = None
if not args.skip_models:
    try:
        from operator1.hedge_fund.engine import run_hedge_fund_analysis
        hf_result = run_hedge_fund_analysis(
            income_df=income_df,
            balance_df=balance_df,
            cashflow_df=cashflow_df,
            cache=cache,
            target_profile=target_profile,
            forecast_result=forecast_result,
            mc_result=mc_result,
            scenario_result=scenario_result,
            multi_frequency_result=multi_frequency_result,
            signal_ic_result=signal_ic_result,
            filing_calendar_result=filing_calendar_result,
            fh_result=fh_result,
            peer_ranking_result=peer_ranking_result,
            sentiment_result=sentiment_result,
            survival_controller=survival_controller,
            linked_caches=linked_caches,
            macro_data=macro_data,
        )
    except Exception as exc:
        logger.warning("Hedge Fund Analysis failed: %s", exc)
```

### W2: Extend profile_builder.py

**File:** `operator1/report/profile_builder.py`
**In `build_company_profile()`, after existing profile sections:**
No changes needed in profile_builder.py itself -- the injection happens in main.py Step 7:
```python
# In main.py Step 7, after profile = build_company_profile(...)
if hf_result is not None and hf_result.available:
    profile["hedge_fund"] = hf_result.to_profile_dict()
else:
    profile["hedge_fund"] = {"available": False}
```

### W3: Extend report_generator.py

**File:** `operator1/report/report_generator.py`
**Add 7 new section builders + wire into TIER_SECTIONS:**

New builder functions:
- `_build_hf_earnings_quality(profile)` -> section 23
- `_build_hf_cash_flow(profile)` -> section 24
- `_build_hf_balance_sheet(profile)` -> section 25
- `_build_hf_inflection(profile)` -> section 26
- `_build_hf_valuation(profile)` -> section 27
- `_build_hf_scorecard(profile)` -> section 28
- `_build_hf_position_signal(profile)` -> section 29

Wire into `TIER_SECTIONS[ReportTier.PREMIUM]` and `_section_builders` dict.
Each reads from `profile["hedge_fund"]` and formats as markdown tables.

### W4: profile_schema.py

**File:** `operator1/report/profile_schema.py`
**Add to `_OPTIONAL_SECTIONS`:**
```python
"hedge_fund",
```

### W5: dashboard.py

**File:** `dashboard.py`
**In `render_home()`, after existing summary cards:**
```python
# HF Thesis Scorecard (if available)
hf = profile.get("hedge_fund", {})
if hf.get("available"):
    sc = hf.get("scorecard", {})
    ui.separator()
    ui.label("Hedge Fund Thesis").classes("text-lg font-bold mt-3")
    with ui.row().classes("gap-4"):
        _card("Grade", sc.get("investment_grade", "N/A"), "", "school")
        _card("Conviction", f'{sc.get("conviction", 0)}/10', "", "psychology")
        pos = hf.get("position", {})
        _card("Signal", f'{pos.get("signal", 0):+.2f} ({pos.get("label", "hold")})', "", "trending_up")
```

### W6: scoring_weights.py

**File:** `operator1/scoring_weights.py`
**Add function to load HF weights alongside existing weights:**
```python
def get_hedge_fund_weights() -> dict[str, Any]:
    """Return hedge fund analysis weights from config/hedge_fund_weights.yml."""
    from operator1.hedge_fund.helpers import load_hf_config
    return load_hf_config()
```
Also extend the dashboard Scoring Weights tab to show HF weights.

### W7: backtest_runner.py

**File:** `backtest_runner.py`
**In `run_stage2()` (after temporal models) or `run_stage3()` (before profile build):**
```python
# HF Analysis
try:
    from operator1.hedge_fund.engine import run_hedge_fund_analysis
    state.hf_result = run_hedge_fund_analysis(
        income_df=state._income_df,
        balance_df=state._balance_df,
        cashflow_df=state._cashflow_df,
        cache=cache,
        ...
    )
except Exception as exc:
    logger.debug("HF analysis skipped: %s", exc)
```
Add `hf_result` field to `BacktestState`.

---

## Phase 4: Style Cleanup (S1-S3) -- LOW PRIORITY

### S1: Deduplicate _sf() / _safe_float()
Keep `_sf()` in types.py (used internally by HF), don't change profile_builder.py.

### S2: Deduplicate safe_divide() / safe_ratio()
Keep both -- they serve different purposes (safe_divide returns float, safe_ratio returns Series + flags).

### S3: Type annotations
Change `fh_result: object` to `fh_result: Any` (already imported). Minor cleanup.

---

## Implementation Order

```
[ ] Phase 1a: Fix B1 (FCF quality negative NI)
[ ] Phase 1b: Fix B2 (Dividend burn negative FCF)
[ ] Phase 1c: Fix B3 (Leverage stress negative EBITDA)
[ ] Phase 1d: Fix B4 (Scorecard distress override)
[ ] Phase 1e: Fix B5 (Position signal fallback alpha)
[ ] Phase 2a: Fix D1 (Add balance_df to fcf_quality)
[ ] Phase 2b: Fix D2 (Add MF context extraction)
[ ] Phase 3a: Wire W1 (main.py Step 6-HF)
[ ] Phase 3b: Wire W2 (profile hedge_fund section in main.py)
[ ] Phase 3c: Wire W3 (report_generator 7 new sections)
[ ] Phase 3d: Wire W4 (profile_schema)
[ ] Phase 3e: Wire W5 (dashboard HF cards)
[ ] Phase 3f: Wire W6 (scoring_weights HF accessor)
[ ] Phase 3g: Wire W7 (backtest_runner HF step)
[ ] Phase 4: Style cleanup (S1-S3)
[ ] Commit + push + verify edge case tests pass
```
