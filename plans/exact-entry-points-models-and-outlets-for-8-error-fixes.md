# Exact Entry Points, Models, and Outlets for 8 Error Fixes

*2026-04-20*

For each method: the exact file, line, function, data inlet, data outlet, and downstream consumers that will change.

---

## Method 1: SEC CompanyFacts API Fallback for NaN Ratios

**Goal:** When edgartools XBRL parsing misses critical fields for AAPL, fall back to SEC CompanyFacts to discover non-standard concept names.

### Entry Point
- **File:** [`operator1/clients/us_edgar.py`](operator1/clients/us_edgar.py:495)
- **Function:** `get_income_statement()` / `get_balance_sheet()` / `get_cashflow_statement()` (lines ~495, ~520, ~535)
- **Trigger:** After edgartools extraction returns a DataFrame with NaN in critical fields

### Model / Logic
- **Existing:** `_extract_from_companyfacts()` at line 1860 already handles companyfacts fallback
- **Existing:** `_fetch_companyfacts_direct()` at line 1964 fetches raw JSON
- **Change needed:** Expand `_USGAAP_BALANCE_CONCEPTS` dict at line 57 to add Apple's non-standard tags:
  ```
  Line 57-73: Add to _USGAAP_BALANCE_CONCEPTS:
    "CashCashEquivalentsRestrictedCashAndRestrictedCashEquivalents": "cash_and_equivalents",
    "CashCashEquivalentsRestrictedCashAndRestrictedCashEquivalentsIncludingDisposalGroupAndDiscontinuedOperations": "cash_and_equivalents",
    "NoncurrentAssets": "non_current_assets",  (for identity: current_assets = total_assets - non_current_assets)
    "AccountsReceivableNet": "receivables",  (broader match)
    "InventoryNet": "inventory",  (confirm present)
  ```
- **New function:** `_fill_critical_nans_from_companyfacts(self, identifier, existing_df)` -- for each NaN column in critical_fields, query CompanyFacts for alternative concepts

### Data Inlet
- `identifier` (CIK/ticker) from `run_stage1` at [`backtest_runner.py:475`](backtest_runner.py:475)
- Raw XBRL DataFrame from edgartools with NaN in `current_assets`, `current_liabilities`, `cash_and_equivalents`

### Data Outlet
- Enriched DataFrame with NaN fields filled from CompanyFacts
- Flows to: cache builder (line 630-658), then derived_variables, then survival_mode, then HF scoring

### Downstream Consumers
- [`operator1/features/derived_variables.py`](operator1/features/derived_variables.py) -- `current_ratio`, `cash_ratio`, `net_margin`, etc.
- [`operator1/analysis/survival_mode.py:49`](operator1/analysis/survival_mode.py:49) -- survival triggers use `current_ratio`
- [`operator1/hedge_fund/advanced_methods.py:79`](operator1/hedge_fund/advanced_methods.py:79) -- Piotroski uses `current_assets/current_liabilities`
- [`operator1/hedge_fund/advanced_methods.py:440`](operator1/hedge_fund/advanced_methods.py:440) -- Altman uses `current_assets/total_assets`

### Lines Changed
- `operator1/clients/us_edgar.py` lines 57-73: add ~8 concept mappings
- `operator1/clients/us_edgar.py` new function ~30 lines: `_fill_critical_nans_from_companyfacts()`
- `operator1/clients/us_edgar.py` lines ~500, ~525: call the new function after primary extraction

---

## Method 2: Score Normalization by Available Data

**Goal:** Piotroski F-Score and Altman Z'' should score on available components only, not penalize NaN.

### Entry Point
- **File:** [`operator1/hedge_fund/advanced_methods.py`](operator1/hedge_fund/advanced_methods.py:79)
- **Function:** `compute_piotroski_f_score()` at line 79
- **Function:** `compute_altman_z_double_prime()` at line 440

### Model / Logic
- **Piotroski (line 79-175):** Currently scores 9 binary signals. When a signal's inputs are NaN, it scores 0. Change: track `n_available` alongside `score`. Return `(score, n_available, label)`. Normalize: `normalized = round(score / n_available * 9)` if `n_available >= 4` else mark as insufficient.
- **Altman Z'' (line 440-500):** Currently uses 4 coefficients. When a ratio's inputs are NaN, the term is 0. Change: sum only terms where inputs are available, divide by number of available terms, multiply by 4 to normalize. Or skip the term entirely and note the reduced confidence.

### Data Inlet
- `income_df`, `balance_df`, `cashflow_df` from [`operator1/hedge_fund/engine.py:1255`](operator1/hedge_fund/engine.py:1255)
- `extract_latest_value()` returns `None` for NaN fields

### Data Outlet
- `AdvancedMethodsResult.piotroski_f_score` (int, now normalized)
- `AdvancedMethodsResult.piotroski_label` (str, now includes data quality note)
- `AdvancedMethodsResult.altman_z_double_prime` (float, now normalized)
- New field: `AdvancedMethodsResult.piotroski_n_available` (int, 0-9)
- New field: `AdvancedMethodsResult.altman_n_available` (int, 0-4)

### Downstream Consumers
- [`operator1/hedge_fund/engine.py:1255`](operator1/hedge_fund/engine.py:1255) -- logs the score
- [`operator1/hedge_fund/engine.py`](operator1/hedge_fund/engine.py) -- `_build_scorecard()` uses Piotroski for earnings quality tier
- [`backtest_runner.py`](backtest_runner.py) Stage 3 profile injection
- [`operator1/report/report_generator.py`](operator1/report/report_generator.py) -- report sections display the score

### Lines Changed
- `operator1/hedge_fund/advanced_methods.py` lines 79-175: add `n_available` counter (~10 lines changed)
- `operator1/hedge_fund/advanced_methods.py` lines 440-500: add normalization (~10 lines changed)
- `operator1/hedge_fund/advanced_methods.py` lines 51-53: add `piotroski_n_available`, `altman_n_available` fields

---

## Method 3: MC Path Intervals Instead of Conformal for 21d+

**Goal:** Replace the degenerate conformal interval at 21d (-$60 to $559) with MC P5/P95 path percentiles.

### Entry Point
- **File:** [`operator1/stages/stage6_ensemble.py`](operator1/stages/stage6_ensemble.py:87) `run_6_3_conformal()`
- **Also:** [`backtest_runner.py`](backtest_runner.py:1611) Stage 2c conformal section
- **Also:** [`operator1/models/conformal.py`](operator1/models/conformal.py:378) `build_conformal_result()`

### Model / Logic
- **Current:** `build_conformal_result()` at line 378 computes intervals for ALL horizons using the conformal calibrator
- **Change:** After computing conformal intervals, for horizons >= 21d, replace with MC path percentiles if `mc_result` is available:
  ```python
  if horizon_days >= 21 and mc_result is not None:
      # Get MC terminal values for this horizon
      mc_paths = mc_result.terminal_values.get(horizon_label)
      if mc_paths is not None:
          last_close = cache["close"].dropna().iloc[-1]
          mc_prices = last_close * (1 + mc_paths)
          interval.lower = float(np.percentile(mc_prices, 5))
          interval.upper = float(np.percentile(mc_prices, 95))
  ```

### Data Inlet
- `state.mc_result.terminal_values` from [`operator1/models/monte_carlo.py:1423`](operator1/models/monte_carlo.py:1423)
- `state.conformal_result.intervals` from `build_conformal_result()`
- `cache["close"]` for converting returns to prices

### Data Outlet
- `state.conformal_result.intervals[var][horizon].lower/upper` -- replaced with MC-based bounds for 21d+
- Flows to: prediction aggregator at [`stage6_ensemble.py:159`](operator1/stages/stage6_ensemble.py:159)

### Downstream Consumers
- [`operator1/models/prediction_aggregator.py`](operator1/models/prediction_aggregator.py) -- uses conformal intervals for uncertainty bands
- [`operator1/report/report_generator.py`](operator1/report/report_generator.py) -- displays intervals in Section 9
- [`backtest_runner.py`](backtest_runner.py) validation -- checks `actual_in_bounds`

### Lines Changed
- `operator1/stages/stage6_ensemble.py` after line 167: add MC interval override (~15 lines)
- `backtest_runner.py` after line 1635: mirror the same override (~15 lines)

---

## Method 4: Merton Default -> MC Survival Anchor

**Goal:** When Merton says P(default)=0.01%, MC survival should be >= 99%, not 12%.

### Entry Point
- **File:** [`operator1/stages/stage7_integration.py`](operator1/stages/stage7_integration.py:328) `run_7_5_hedge_fund()` -- HF runs AFTER MC
- **Also:** [`backtest_runner.py`](backtest_runner.py:1814) `run_stage2d()` -- HF runs in 2d
- **Better insertion point:** After MC completes and before profile building. In the staged pipeline: after 5.4 (MC) and after 7.5 (HF). Or: add MC anchoring as a post-processing step in profile building.

### Model / Logic
- **Current:** MC survival stored in `mc_result.survival_probability` as dict `{horizon: {mean, p5, p95}}`
- **Change:** After HF analysis produces `merton_default_probability`, compute a floor:
  ```python
  merton_pd = hf_result.advanced_methods.merton_default_probability  # e.g. 0.001
  survival_floor = 1.0 - merton_pd  # 0.999
  # Also use market_cap_quintile_floor as secondary anchor
  mcap = cache["market_cap"].dropna().iloc[-1]
  mcap_floor = get_mcap_survival_floor(mcap)
  floor = max(survival_floor, mcap_floor)
  # Apply floor
  for horizon_data in mc_result.survival_probability.values():
      if horizon_data["mean"] < floor:
          horizon_data["mean"] = floor
          horizon_data["anchored"] = True
  ```

### Data Inlet
- `state.hf_result.advanced_methods.merton_default_probability` from [`advanced_methods.py`](operator1/hedge_fund/advanced_methods.py)
- `cache["market_cap"]` from OHLCV data
- `state.mc_result.survival_probability` from [`monte_carlo.py:1423`](operator1/models/monte_carlo.py:1423)

### Data Outlet
- Modified `state.mc_result.survival_probability` with floor-adjusted values
- Flows to: profile builder, report generator, triage card

### Downstream Consumers
- [`operator1/report/profile_builder.py`](operator1/report/profile_builder.py) -- `_build_monte_carlo_section()`
- [`operator1/report/report_generator.py`](operator1/report/report_generator.py) -- Section 9 Predictions
- [`operator1/report/triage_card.py`](operator1/report/triage_card.py) -- survival probabilities

### Lines Changed
- `backtest_runner.py` after HF analysis in `run_stage2d()` (~line 1850): add anchoring block (~15 lines)
- `operator1/stages/stage7_integration.py` after `run_7_5_hedge_fund()` completes: add same block (~15 lines)
- New utility function `get_mcap_survival_floor()` in `operator1/models/monte_carlo.py` (~10 lines)

---

## Method 5: Regime Probability-Weighted Prediction Shift

**Goal:** Use HMM state probabilities to adjust Kalman/baseline predictions directionally.

### Entry Point
- **File:** [`operator1/models/forecasting.py`](operator1/models/forecasting.py:2362)
- **Location:** After `_horizon_forecasts` is populated (line 2362-2365), before the long-horizon tree blend

### Model / Logic
- **Existing data:** `regime_hmm_prob_*` columns in cache (line 2246), `regime_map` at line 2249
- **Change:** After populating `_horizon_forecasts` and after the residual feature adjustment (Bug #2 fix), add:
  ```python
  # Regime probability-weighted return shift
  _prob_cols = [c for c in cache.columns if c.startswith("regime_hmm_prob_")]
  if _prob_cols and len(cache) > 0:
      _last_probs = cache[_prob_cols].iloc[-1]
      _regime_means = {}
      for col in _prob_cols:
          idx = int(col.replace("regime_hmm_prob_", ""))
          _regime_means[idx] = cache["return_1d"].where(cache["regime_hmm"] == idx).mean()
      expected_daily_return = sum(
          _regime_means.get(i, 0) * float(_last_probs.iloc[i])
          for i in range(len(_last_probs)) if not np.isnan(_last_probs.iloc[i])
      )
      for label, h in HORIZONS.items():
          shift = expected_daily_return * h
          _horizon_forecasts[label] *= (1 + shift)
  ```

### Data Inlet
- `cache["regime_hmm_prob_0"]` through `cache["regime_hmm_prob_3"]` from regime_detector
- `cache["regime_hmm"]` labels from regime_detector
- `cache["return_1d"]` for computing per-regime mean returns

### Data Outlet
- Modified `_horizon_forecasts` dict (per-horizon close predictions)
- Flows to: `result.forecasts[var_name]`, then prediction aggregator

### Downstream Consumers
- [`operator1/models/prediction_aggregator.py`](operator1/models/prediction_aggregator.py) -- consumes `forecast_result.forecasts`
- [`backtest_runner.py`](backtest_runner.py) Stage 3 validation

### Lines Changed
- `operator1/models/forecasting.py` after line ~2387 (after residual adjustment, before horizon-specific tree blend): ~15 lines

---

## Method 6: PELT Fallback When HMM Degenerate

**Goal:** When any HMM state has <5% of observations, fall back to PELT-based regime labeling.

### Entry Point
- **File:** [`operator1/models/regime_detector.py`](operator1/models/regime_detector.py:207)
- **Function:** The HMM fitting block (around line 207-260)
- **Trigger:** After HMM labels are assigned, check for degeneracy

### Model / Logic
- **Current:** HMM always uses `n_components=4` (line 208). Labels assigned via emission mean ranking (line ~280).
- **Change:** After HMM labels are assigned, check state distribution:
  ```python
  state_counts = pd.Series(hmm_labels).value_counts(normalize=True)
  if state_counts.min() < 0.05:
      logger.warning("HMM degenerate: state '%s' has %.1f%% -- using PELT fallback",
                      state_counts.idxmin(), state_counts.min() * 100)
      # Use PELT breakpoints to segment time series
      # Label each segment by mean return + volatility
      pelt_labels = _label_from_pelt_segments(cache, pelt_breakpoints)
      hmm_labels = pelt_labels
  ```
- **Existing:** PELT is already computed in the same function (line ~340-360). The breakpoints are available.

### Data Inlet
- `hmm_labels` array from HMM fit
- `pelt_breakpoints` from PELT detection (already computed in same function)
- `cache["return_1d"]` and `cache["volatility_21d"]` for segment characterization

### Data Outlet
- `cache["regime_label"]` -- corrected labels
- `cache["regime_hmm"]` -- corrected labels
- `regime_detector.result` -- updated result object

### Downstream Consumers
- [`operator1/stages/stage3_temporal.py:96`](operator1/stages/stage3_temporal.py:96) -- uses regime_label
- [`operator1/models/monte_carlo.py`](operator1/models/monte_carlo.py) -- per-regime return distributions
- [`operator1/models/forecasting.py:2246`](operator1/models/forecasting.py:2246) -- regime probabilities for Kalman
- [`operator1/stages/stage6_ensemble.py:131`](operator1/stages/stage6_ensemble.py:131) -- vol_ratio for conformal widening

### Lines Changed
- `operator1/models/regime_detector.py` after HMM label assignment (~line 280): add degeneracy check + PELT fallback (~15 lines)
- New helper `_label_from_pelt_segments()` in same file (~20 lines)

---

## Method 7: PEAD Earnings Drift -> Price Prediction

**Goal:** Adjust price predictions based on earnings surprise probability and proximity to filing.

### Entry Point
- **File:** [`operator1/models/forecasting.py`](operator1/models/forecasting.py:2362)
- **Also:** [`backtest_runner.py`](backtest_runner.py:1672) prediction aggregation
- **Better insertion:** In prediction_aggregator after ensemble weights are computed

### Model / Logic
- **Data source:** `hf_result.earnings_surprise` with `p_beat`, `p_miss`, `days_to_next_filing` from [`engine.py:569-602`](operator1/hedge_fund/engine.py:569)
- **But HF runs AFTER forecasting.** So this needs to be wired differently.
- **Alternative inlet:** `event_calendar_result.earnings_proximity` and `event_calendar_result.days_to_next_event` which are computed in Stage 1 (before forecasting)
- **Change in prediction_aggregator:** After ensemble aggregation, apply PEAD drift:
  ```python
  if event_calendar_result and event_calendar_result.available:
      days_to_event = event_calendar_result.days_to_next_event
      if days_to_event and days_to_event < 30:
          # Apply pre-earnings drift based on event_uncertainty_premium
          prem = event_calendar_result.event_uncertainty_premium or 0
          drift = -prem * 0.001 * min(days_to_event, 21)  # uncertainty = slight downward pressure
          for var in result.predictions:
              if var == "close" and isinstance(result.predictions[var], dict):
                  for h, hp in result.predictions[var].items():
                      if hasattr(hp, "point_forecast") and hp.point_forecast:
                          hp.point_forecast *= (1 + drift)
  ```

### Data Inlet
- `state.event_calendar_result.days_to_next_event` from [`backtest_runner.py:767`](backtest_runner.py:767)
- `state.event_calendar_result.event_uncertainty_premium` from event_calendar features

### Data Outlet
- Modified `state.pred_result.predictions["close"]` with drift-adjusted forecasts

### Downstream Consumers
- Profile builder, report generator, validation

### Lines Changed
- `operator1/models/prediction_aggregator.py` `run_prediction_aggregation()`: add `event_calendar_result` parameter + drift block (~20 lines)
- `operator1/stages/stage6_ensemble.py` `run_6_5_aggregation()`: pass `state.event_calendar_result` (~1 line)
- `backtest_runner.py` Stage 2c prediction aggregation call: pass event_calendar_result (~1 line)
- `main.py` prediction aggregation call: pass event_calendar_result (~1 line)

---

## Method 8: Market-Cap Quintile Survival Calibration

**Goal:** Set minimum survival probability based on market cap quintile, preventing absurd 12% for $3T AAPL.

### Entry Point
- Same location as Method 4 -- applied after MC completes
- **File:** [`operator1/models/monte_carlo.py`](operator1/models/monte_carlo.py:1423)
- **Alternative:** Post-MC processing in `backtest_runner.py` and `stage5_forward.py`

### Model / Logic
- **New function** in `operator1/models/monte_carlo.py`:
  ```python
  def get_mcap_survival_floor(market_cap: float) -> float:
      if market_cap >= 200e9: return 0.92
      elif market_cap >= 10e9: return 0.82
      elif market_cap >= 2e9: return 0.70
      elif market_cap >= 300e6: return 0.55
      else: return 0.40
  ```
- Combined with Method 4 (Merton anchor) in a single post-MC block

### Data Inlet
- `cache["market_cap"]` from OHLCV/profile data
- `state.mc_result.survival_probability` from MC simulation

### Data Outlet
- Same as Method 4 -- floor-adjusted `mc_result.survival_probability`

### Downstream Consumers
- Same as Method 4

### Lines Changed
- `operator1/models/monte_carlo.py`: add `get_mcap_survival_floor()` (~10 lines)
- Merged with Method 4 implementation (shared post-MC block)

---

## Summary: All Files and Lines

| File | Methods | Total Lines Changed |
|------|---------|-------------------|
| `operator1/clients/us_edgar.py` | #1 | ~40 (8 concepts + 30-line function + 2 call sites) |
| `operator1/hedge_fund/advanced_methods.py` | #2 | ~25 (Piotroski normalization + Altman normalization + 2 fields) |
| `operator1/stages/stage6_ensemble.py` | #3, #7 | ~17 (MC interval override + event_calendar pass-through) |
| `operator1/models/monte_carlo.py` | #4, #8 | ~15 (mcap_floor function + anchoring utility) |
| `operator1/models/forecasting.py` | #5 | ~15 (regime probability-weighted shift) |
| `operator1/models/regime_detector.py` | #6 | ~35 (degeneracy check + PELT fallback helper) |
| `operator1/models/prediction_aggregator.py` | #7 | ~22 (event_calendar parameter + PEAD drift) |
| `backtest_runner.py` | #3, #4, #7, #8 | ~35 (MC interval mirror + Merton/mcap anchor + event_calendar pass) |
| `main.py` | #7 | ~2 (event_calendar pass-through) |
| `operator1/stages/stage7_integration.py` | #4 | ~15 (Merton/mcap anchor after HF) |
| **Total** | **8 methods** | **~221 lines across 10 files** |
