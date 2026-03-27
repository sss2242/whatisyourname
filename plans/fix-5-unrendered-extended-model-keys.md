# Fix: 5 Unrendered Extended Model Keys

## Problem

5 model results are computed, stored in `company_profile.json` under `extended_models`, but never displayed in either the report or the dashboard.

## What Each Key Contains

### 1. `extended_models.candlestick_patterns`
**Source:** `detect_patterns()` in `pattern_detector.py`
**Data:** Detected candlestick patterns (doji, hammer, engulfing), motifs (recurring patterns via Matrix Profile), discords (anomalies)
**Profile keys:** `available`, `patterns` list, `motifs` list, `discords` list, `n_patterns`

### 2. `extended_models.walk_forward`
**Source:** `run_walk_forward()` in `walk_forward.py`
**Data:** Model evaluation: which forecasting model works best in each survival mode
**Profile keys:** `available`, `overall_mae`, `overall_best_model`, `n_retrains`, `mode_scores` dict, `best_model_by_mode` dict

### 3. `extended_models.burnout`
**Source:** `run_burnout()` in `forecasting.py`
**Data:** Online weight calibration via exponential gradient learning
**Profile keys:** `available`, `iterations_completed`, `converged`, `calibrated`, `weight_stability`, `calibration_steps`, `regime_weights` dict, `regime_distributions` dict

### 4. `extended_models.time_varying_granger`
**Source:** `compute_time_varying_granger()` in `granger_causality.py`
**Data:** Rolling-window causal graph showing which relationships are emerging or disappearing over time
**Profile keys:** `available`, `n_windows`, `emerging_pairs` list, `disappearing_pairs` list

### 5. `extended_models.multivariate_monte_carlo`
**Source:** `run_multivariate_monte_carlo()` in `monte_carlo.py`
**Data:** Joint simulation of financial ratios via copula correlation, with direct ratio-based survival triggers
**Profile keys:** `available`, `survival_probability`, `variables_simulated` list

## Plan

### A. Report Generator (`report_generator.py`)

Add 5 new subsections to `_build_advanced_insights()` (Section 19), right after the existing Granger causality subsection. Each follows the same pattern: check `ext.get(key, {})`, guard on `available`, render descriptive text + key metrics.

| # | Subsection Title | Insert After | Key Metrics to Display |
|---|-----------------|-------------|----------------------|
| 1 | Candlestick Pattern Detection | Granger causality | n_patterns, top patterns with signal direction, motif count, discord count |
| 2 | Walk-Forward Model Evaluation | Candlestick patterns | overall_best_model, overall_mae, best_model_by_mode table, n_retrains |
| 3 | Burn-Out Weight Calibration | Walk-forward | converged, iterations, calibrated, regime_weights table |
| 4 | Time-Varying Causal Dynamics | Burn-out | n_windows, emerging_pairs list, disappearing_pairs list |
| 5 | Multivariate Monte Carlo | Time-varying Granger | survival_probability, variables_simulated list |

### B. Dashboard (`dashboard.py`)

Add a new "Advanced Models" card to the Report page Summary tab. The dashboard currently shows 3 summary cards (Health Score, Survival Prob, Regime). Add a 4th card showing the count of available extended models, and in the Charts tab add a mini table listing each model's key result.

| Component | Location | Content |
|-----------|----------|---------|
| Summary card | Report > Summary tab | "Models: X/17 available" badge |
| Model results table | Report > Summary tab (below existing cards) | Table: model name, status, key metric |

### Files Modified

| File | Change |
|------|--------|
| `operator1/report/report_generator.py` | Add 5 subsection builders to `_build_advanced_insights()` (~120 lines) |
| `dashboard.py` | Add extended models summary card + table to Report page (~40 lines) |
