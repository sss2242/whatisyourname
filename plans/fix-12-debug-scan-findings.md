# Fix Plan: 12 Debug Scan Findings

## Category 1: Report Generator Missing 5 Profile Keys

These profile keys are stored in `company_profile.json` but have no report sections. Each needs a `_build_*()` template function in `report_generator.py` and a section number in `TIER_SECTIONS`.

### F1. Corporate Structure section
- **File:** `operator1/report/report_generator.py`
- **Action:** Add `_build_corporate_structure_section(profile)` that renders GLEIF parent/subsidiary data
- **Section number:** 19.9 (Pro+ tier)
- **Content:** Parent companies table, subsidiary count + countries, cross-border flag
- **Profile key:** `profile["corporate_structure"]`

### F2. Filing Calendar section
- **File:** `operator1/report/report_generator.py`  
- **Action:** Add `_build_filing_calendar_section(profile)` that renders filing frequency and staleness
- **Section number:** 6.5 (Pro+ tier, after Survival Mode)
- **Content:** Filing frequency, coverage ratio, gaps, staleness warning, predicted next filing date
- **Profile key:** `profile["filing_calendar"]`

### F3. Macro Indicators section
- **File:** `operator1/report/report_generator.py`
- **Action:** Add `_build_macro_indicators_section(profile)` that renders macro data summary
- **Section number:** 18.5 (Pro+ tier, after Macro Quadrant)
- **Content:** Table of indicators with latest value, date, observation count
- **Profile key:** `profile["macro_indicators"]`

### F4. Supply Chain Stress section
- **File:** `operator1/report/report_generator.py`
- **Action:** Add `_build_supply_chain_stress_section(profile)` that renders supply chain stress
- **Section number:** 12.5 (Premium tier, after Supply Chain and Contagion Risk)
- **Content:** Stress flag, score, stress sources list
- **Profile key:** `profile["supply_chain_stress"]`

### F5. Synergies Applied section
- **File:** `operator1/report/report_generator.py`
- **Action:** Add `_build_synergies_section(profile)` that renders synergy metadata
- **Section number:** 19.99 (Premium tier, in Advanced Insights)
- **Content:** Cycle features added, causal network density, pattern drift multiplier
- **Profile key:** `profile["synergies_applied"]`

### All 5 sections:
- Add section numbers to `TIER_SECTIONS` for appropriate tiers
- Add builders to `_section_builders` dict
- Add to `_embed_charts_in_markdown` if any produce charts

---

## Category 2: Profile Schema Validation Gaps (3 keys)

### F6-F8. Add 3 keys to profile_schema.py
- **File:** `operator1/report/profile_schema.py`
- **Action:** Add `predicted_regime_shifts`, `model_diagnostics`, and `supply_chain_stress` to the optional keys validation list (with `available` flag check)
- **Where:** In the `validate_profile()` function, add to the optional sections list alongside existing ones like `graph_risk`, `game_theory`, etc.

---

## Category 3: Backtest Runner Module Gaps (4 items)

### F9. Add adaptive_model_params + adaptive_windows to backtest Stage 1
- **File:** `backtest_runner.py`
- **Action:** In `_run_stage1()`, after adaptive thresholds, add:
  - Import and call `compute_blend_weights()`, `compute_regime_risk_multiplier()`, etc. from `adaptive_model_params`
  - Import and call `compute_adaptive_windows()`, `compute_nn_hyperparams()` from `adaptive_windows`
  - Store results in `state._adaptive_model_params` and `state._adaptive_tier3`

### F10. Add linked_aggregates + ownership_contagion to backtest Stage 1
- **File:** `backtest_runner.py`
- **Action:** In `_run_stage1()`, after linked entity data fetch, add:
  - Import and call `compute_linked_aggregates()` and `compute_relative_metrics()` from `linked_aggregates`
  - Import and call `compute_ownership_contagion()` from `ownership_contagion`
  - Merge linked_agg_df columns into cache
  - Store `state.linked_agg_df` and contagion_result

### F11. Add regime_shift_predictor to backtest Stage 2
- **File:** `backtest_runner.py`
- **Action:** In `_run_stage2()`, after Monte Carlo, add:
  - Import and call `predict_regime_shifts()` from `regime_shift_predictor`
  - Store result in `state.regime_shift_result`
  - Pass to profile builder in Stage 3

### F12. Add fetch_benchmark_returns (beta_252d) to backtest Stage 1
- **File:** `backtest_runner.py`
- **Action:** In `_run_stage1()`, after cache build, add:
  - Import `fetch_benchmark_returns` from `ohlcv_provider`
  - Fetch benchmark returns for the market_id
  - Merge into cache as `benchmark_return_1d`
  - This enables `beta_252d` computation in `compute_derived_variables()`

---

## Implementation Order

1. **Profile schema** (F6-F8) -- smallest change, 3 lines
2. **Report generator** (F1-F5) -- 5 new section builders, ~150 lines total
3. **Backtest runner** (F9-F12) -- 4 module integrations, ~80 lines total

## Testing Strategy

- No tests to run per instructions, but all changes should be syntax-checked with `python -m py_compile`
- Report sections should produce valid markdown for available and unavailable data
- Backtest additions should be wrapped in try/except like existing code
