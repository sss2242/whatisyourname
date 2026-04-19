# Staged Pipeline Sub-Stage Data Flow Map + Gap Analysis

*2026-04-19*

## Part 1: How Sub-Stages Save and Pass Data

### Serialization Mechanism

The `PipelineState` object is the single shared state bag. Between sub-stages, `runner.py` calls `state.save(sub_id)` which persists:

1. **`cache.parquet`** -- the main daily DataFrame (overwritten each checkpoint)
2. **`linked_caches/`** -- per-entity Parquet files + `_ids.json` manifest
3. **`linked_agg.parquet`** -- linked aggregate DataFrame
4. **`income_df.parquet`, `balance_df.parquet`, `cashflow_df.parquet`, `quotes_df.parquet`** -- raw statement DFs
5. **`state_{sub_id}.pkl`** -- pickle of ALL non-DataFrame state (model results, dicts, scalars)
6. **`config.json`** -- human-readable metadata (market_id, company, end_date, sub_stage_completed)

When resuming, `state.load_checkpoint(prev_sub_id)` reads Parquet files back into DataFrames and unpickles model results into the state object.

### Sub-Stage Data Flow Chain

```
main.py Steps 1-5k.2 (pre-temporal)
    |
    | Populates PipelineState with: cache, target_profile, weights, fh_result,
    | fuzzy_result, relationships, linked_caches, linked_agg_df, graph_risk_result,
    | game_theory_result, contagion_result, peer_ranking_result, sentiment_result,
    | catalyst_result, signal_ic_result, prediction_log_summary, enriched_timeline_result,
    | early_regime_result, regime_detector, survival_controller, adaptive_thresholds,
    | adaptive_model_params, adaptive_tier3, income_df, balance_df, cashflow_df, quotes_df,
    | options_signal_result, cross_asset_result, event_calendar_result
    |
    | state.save("2.9") -- checkpoint before temporal stages
    v
[3.1 Regime Detection]
    Reads: state.cache, state.regime_detector
    Writes: state.cache (+ regime columns), state.regime_detector, state.extra_vars
    Passes to 3.2: regime_label in cache, regime_detector object
    |
    v
[3.2 Dual Regime]
    Reads: state.cache (regime_label), state.adaptive_thresholds
    Writes: state.dual_regime_result
    Passes to 6.5: dual_regime_result for prediction aggregation
    |
    v
[3.3 Granger Causality]
    Reads: state.cache, state.extra_vars
    Writes: state.granger_result, state.extra_vars (pruned)
    Passes to 3.7/6.5/6.8: granger_result, pruned extra_vars
    |
    v
[3.4 Transfer Entropy]
    Reads: state.cache
    Writes: state.transfer_entropy_result
    Passes to 3.7: transfer_entropy_result for unified causal network
    |
    v
[3.5 Cycle Decomposition]
    Reads: state.cache
    Writes: state.cycle_result
    Passes to 3.7/6.10: cycle_result for phase features + OHLC drift
    |
    v
[3.6 Pattern Detection]
    Reads: state.cache (OHLC)
    Writes: state.pattern_result
    Passes to 6.10: pattern_result for pattern drift adjustment
    |
    v
[3.7 Synergies]
    Reads: state.cache, state.extra_vars, state.cycle_result, state.granger_result,
           state.transfer_entropy_result, state.linked_caches, state.economic_plane
    Writes: state.cache (+ synergy features), state.extra_vars (updated),
            state.synergy_meta, state.economic_plane
    Passes to 4.1: enriched cache + pruned extra_vars
    |
    v
[4.1 Forecasting]
    Reads: state.cache, state.extra_vars, state.adaptive_tier3
    Writes: state.forecast_result (forecasts, metrics, residuals, model_used)
    Passes to 5.1/6.1/6.3/6.5/6.9/6.10/7.1: forecast_result
    |
    v
[5.1 Forward Pass]
    Reads: state.cache, state.weights, state.extra_vars, regime_labels from cache
    Writes: state.forward_pass_result (errors, model_states, predictions_log,
            conformal_calibrator)
    Passes to 5.2/5.3/6.3/6.5/6.6: forward_pass_result
    |
    v
[5.2 Burn-Out]
    Reads: state.cache, state.weights, state.extra_vars, state.forward_pass_result
    Writes: state.burnout_result (regime_weights, regime_distributions)
    Passes to 5.3/5.4: burnout calibrated weights + distributions
    |
    v
[5.3 Walk-Forward + MCS + FixedShare]
    Reads: state.cache, state.forward_pass_result, state.burnout_result
    Writes: state.walk_forward_result, state.mode_weights
    Passes to 6.5: walk_forward_result + mode_weights for ensemble weighting
    |
    v
[5.4 Monte Carlo + Regime Shift]
    Reads: state.cache, state.adaptive_thresholds, state.adaptive_model_params,
           state.burnout_result
    Writes: state.mc_result, state.regime_shift_result
    Passes to 6.5/6.8/6.10/7.1/7.5: mc_result + regime_shift_result
    |
    v
[5.5 Copula]
    Reads: state.cache
    Writes: state.copula_result
    Passes to 6.5/6.8/7.3: copula_result (tail dependence, correlation)
    |
    v
[6.1 Transformer]
    Reads: state.cache, state.forecast_result
    Writes: state.transformer_result; injects forecasts into state.forecast_result
    Passes to 6.5: enhanced forecast_result with transformer predictions
    |
    v
[6.2 Particle Filter]
    Reads: state.cache
    Writes: state.particle_filter_result
    |
    v
[6.3 Conformal]
    Reads: state.cache, state.forecast_result, state.forward_pass_result,
           state.regime_detector
    Writes: state.conformal_result (prediction intervals)
    Passes to 6.5: conformal_result for uncertainty bands
    |
    v
[6.4 DTW Analogs]
    Reads: state.cache, state.linked_caches, state.catalyst_result
    Writes: state.dtw_result
    Passes to 6.5/7.3: dtw_result for analog forecasts
    |
    v
[6.5 Prediction Aggregation]
    Reads: state.cache, state.forecast_result, state.mc_result, state.mode_weights,
           state.signal_ic_result, state.prediction_log_summary, state.conformal_result,
           state.dual_regime_result, state.copula_result, state.dtw_result,
           state.granger_result, state.shap_result, state.walk_forward_result,
           state.survival_controller
    Writes: state.pred_result (final ensemble predictions)
    Passes to 6.6/7.x: pred_result
    |
    v
[6.6 SHAP]
    Reads: state.cache, state.pred_result, state.forward_pass_result
    Writes: state.shap_result
    |
    v
[6.7 Sobol]
    Reads: state.cache, state.weights
    Writes: state.sobol_result, state.weights (adjusted)
    |
    v
[6.8 TV Granger + MV MC]
    Reads: state.cache, state.copula_result
    Writes: state.tv_granger_result, state.mv_mc_result
    |
    v
[6.9 Genetic Optimizer]
    Reads: state.cache, state.forecast_result
    Writes: state.ga_result
    |
    v
[6.10 OHLC Predictor]
    Reads: state.cache, state.forecast_result, state.mc_result, state.pattern_result,
           state.cycle_result
    Writes: state.ohlc_result, state.pattern_drift, state.pattern_result (predicted
            patterns)
    |
    v
[7.1 USS + Scenario]
    Reads: state.cache, state.survival_controller, state.forecast_result
    Writes: state.scenario_result
    |
    v
[7.2 Retro Calibration]
    Reads: state.cache, state.linked_caches, state.relationships,
           state.walk_forward_result, state.forecast_result, state.sobol_result,
           state.target_profile
    Writes: state.retro_params
    |
    v
[7.3 Model Diagnostics]
    Reads: state.cache, state.forecast_result, state.mc_result, state.copula_result,
           state.granger_result, state.cycle_result, state.dtw_result,
           state.conformal_result
    Writes: state.model_diagnostics_result
    |
    v
[7.4 Multi-Frequency] (7.4.0 -> 7.4.1-5 -> 7.4.6)
    Reads: state.cache, state.income_df, state.balance_df, state.cashflow_df,
           state.quotes_df
    Writes: state.multi_frequency_result
    |
    v
[7.5 Hedge Fund]
    Reads: state.income_df, state.balance_df, state.cashflow_df, state.cache,
           state.target_profile, state.forecast_result, state.mc_result,
           state.scenario_result, state.multi_frequency_result, state.signal_ic_result,
           state.filing_calendar_result, state.fh_result, state.peer_ranking_result,
           state.sentiment_result, state.survival_controller, state.linked_caches,
           state.macro_data
    Writes: state.hf_result
```

---

## Part 2: The Staged Pipeline Commit History

### Key Commit: Staged Architecture Created

- **`05b210c` (2026-04-16):** `feat: staged pipeline architecture with per-model sub-stages (Stages 3-6)` -- Created pipeline_state.py, runner.py, stage3-6 modules. 30 sub-stages.
- **`d92066c` (2026-04-16):** `feat: add Stage 7 integration sub-stages` -- stage7_integration.py (7.1-7.5 + 7.4.0-7.4.6)
- **`ff4a320` (2026-04-16):** `feat: split multi-frequency pipeline into per-frequency sub-stages`

### Post-Stage Commits (features added AFTER staged architecture)

| Commit | Date | Feature | What It Added |
|--------|------|---------|---------------|
| `ee0d913` | 04-18 | Gap 1: Options signals | `features/options_signals.py` (NEW), 6 features in cache |
| `cdb6ddc` | 04-18 | Gap 2: Geographic supply chain | `features/product_metrics.py` enhanced, 5 features in cache |
| `1ef161f` | 04-18 | Gap 3: Cross-asset rotation | `features/cross_asset_signals.py` (NEW), 6 features in cache |
| `3cb5d24` | 04-18 | Gap 4: Event calendar | `features/event_calendar.py` (NEW), 5 features in cache |
| `cd73002` | 04-18 | Gap 5: DCF calibration | `hedge_fund/engine.py` improvements |
| `7215a1f` | 04-18 | Gap 6: Ensemble diversity | `models/prediction_aggregator.py` (routing + reject) |
| `cfffea1` | 04-18 | Wire routing + reject | `models/prediction_aggregator.py` |
| `83afe15` | 04-18 | **Sync _extra_vars** | `stages/stage3_temporal.py` -- added 22 Gap 1-4 vars |
| `5db1d69` | 04-18 | Copy Gap results to MF | `backtest_runner.py` |
| `7970855` | 04-18 | Render in reports | `report/report_generator.py`, `dashboard.py` |

---

## Part 3: Models That Need Proper Sub-Stage Arrangement

### Currently CORRECT

All 30+ original sub-stages (3.1-7.5) are properly wired. Each model reads from PipelineState and writes results back. The `save(sub_id)` call persists everything via pickle.

### Gap 1-4: Already Flowing Correctly

Gap 1-4 features are computed in main.py Steps 4-5 (pre-temporal). They produce:
1. **Cache columns** (e.g., `put_call_ratio`, `geo_hhi`, `sector_relative_strength`, `days_to_next_event`) -- saved in `cache.parquet` at checkpoint 2.9
2. **Result objects** (e.g., `options_signal_result`, `cross_asset_result`, `event_calendar_result`) -- pickled in `state_2.9.pkl`

The staged runner picks up both because:
- `_init_extra_vars()` was synced with 22 new column names in commit `83afe15`
- Result objects survive pickle/unpickle via PipelineState serialization

### Gap 5-6: Already Inside Existing Sub-Stages

- **Gap 5** (DCF calibration) -- changes are inside `hedge_fund/engine.py` which runs in sub-stage 7.5
- **Gap 6** (ensemble diversity) -- changes are inside `prediction_aggregator.py` which runs in sub-stage 6.5

### Verdict: No Models Currently Need Rearrangement

All Gap features flow correctly through the staged pipeline:
- Gap 1-4: pre-temporal features in cache + pickle at 2.9 checkpoint
- Gap 5: inside sub-stage 7.5 (hedge fund)
- Gap 6: inside sub-stage 6.5 (prediction aggregation)
- The `_extra_vars` sync (commit 83afe15) ensures staged 3.1 picks up all 22 new feature columns

### Future Work: Pre-Temporal Sub-Stages (Stages 1-2)

If we wanted to make Steps 1-5k.2 resumable too, we would need ~14 new sub-stages:

| Proposed | Model | Current Location |
|----------|-------|-----------------|
| 1.1 | Data fetch (PIT client) | main.py Step 3 |
| 1.2 | Data reconciliation | main.py Step 3b |
| 1.3 | Cache build + interpolation | main.py Step 4 |
| 1.4 | Macro fetch + quadrant | main.py Step 4a |
| 1.5 | Conflict risk + buying power | main.py Steps 4a.3-4a.4 |
| 1.6 | Options + cross-asset + events (Gaps 1/3/4) | main.py Steps 4a.7-4c.1 |
| 2.1 | Estimation | main.py Step 4b |
| 2.2 | Filing calendar | main.py Step 4c |
| 2.3 | Derived variables + survival + hierarchy | main.py Step 5 |
| 2.4 | USS + fuzzy + financial health + vanity | main.py Steps 5-USS to 5d |
| 2.5 | Entity discovery + linked fetch + contagion | main.py Steps 5e-5g |
| 2.6 | Peer ranking + sentiment + catalysts + segments | main.py Steps 5h-5i.6 |
| 2.7 | Adaptive calibration (thresholds + params + windows) | main.py Steps 5j-5k.2 |
| 2.8 | Enriched survival timeline + signal IC | main.py Steps 5.5-5j.6 |

This would make the entire pipeline resumable, not just the temporal modeling portion.
