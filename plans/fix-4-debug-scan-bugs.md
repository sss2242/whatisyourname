# Fix Plan: 4 Bugs from Debug Scan (2026-04-19)

## Bug #1: Multi-Frequency Pipeline Double Execution (HIGH)

**Location:** `operator1/stages/stage7_integration.py` line 358, `STAGE_7_SUBSTAGES`

**Root cause:** The registry contains BOTH the wrapper `("7.4", run_7_4_multi_frequency)` AND individual sub-stages `("7.4.0", ...)` through `("7.4.6", ...)`. When `--stage all` or `--stage 7` runs, the runner matches both 7.4 (wrapper) and 7.4.0-7.4.6 (individual), executing the multi-frequency pipeline twice.

**Fix:** Remove the wrapper entry `("7.4", run_7_4_multi_frequency)` from `STAGE_7_SUBSTAGES`. Keep only the individual 7.4.0-7.4.6 entries. Users who run `--stage 7.4` will still match all 7.4.x sub-stages via the `_matches_stage()` prefix matching in `runner.py`.

**File changes:**
- `operator1/stages/stage7_integration.py`: Remove line `("7.4", run_7_4_multi_frequency),` from `STAGE_7_SUBSTAGES` list

**Verification:** After fix, `--stage all` should list 7.4.0 through 7.4.6 but NOT 7.4. `--stage 7.4` should still match 7.4.0-7.4.6.

---

## Bug #2: Dashboard Missing product_segments Display (LOW)

**Location:** `dashboard.py`, `render_home()` function

**Root cause:** The Home page renders 20+ profile sections but never reads `profile["product_segments"]`. This section was added in the April 11-14 product segment extraction work but the dashboard was not updated.

**Fix:** Add a Product Segments card group in `render_home()`, after the existing Multi-Frequency Fusion section. Display: `n_segments`, `dominant_segment` with percentage, `segment_hhi` concentration metric.

**File changes:**
- `dashboard.py`: Add ~15 lines in `render_home()` to display product_segments data from the profile JSON

**Verification:** Run dashboard, load a profile that has product_segments data, confirm the cards appear.

---

## Bug #3: Gap 6 routing_weights Computed But Not Applied (MEDIUM)

**Location:** `operator1/models/prediction_aggregator.py` line 2794

**Root cause:** `compute_model_routing_weights(cache)` returns a dict of model-type weights based on market characteristics (e.g., trending markets favor Kalman, volatile favor GARCH). But the result is only stored in `result.metadata` -- it never modifies the actual ensemble weights used for point forecast aggregation.

**Fix:** Apply routing_weights as a multiplicative adjustment to the base ensemble weights. After computing the base inverse-RMSE weights (or FixedShare weights), multiply each model's weight by its routing_weight, then re-normalize to sum to 1.0. This should happen BEFORE the weighted average computation in `run_prediction_aggregation()`.

**File changes:**
- `operator1/models/prediction_aggregator.py`: In `run_prediction_aggregation()`, after line 2798 where routing_weights are stored in metadata, add logic to apply them:
  ```python
  # Apply routing weights to ensemble
  if routing_weights:
      for var in result.predictions:
          for h, hp in result.predictions[var].items():
              if hasattr(hp, 'model_weights') and hp.model_weights:
                  for model_name in hp.model_weights:
                      if model_name in routing_weights:
                          hp.model_weights[model_name] *= routing_weights[model_name]
                  # Re-normalize
                  total = sum(hp.model_weights.values())
                  if total > 0:
                      hp.model_weights = {k: v/total for k, v in hp.model_weights.items()}
  ```

**Verification:** Run pipeline with `--verbose`, check that routing weights appear in prediction metadata AND that they actually change the ensemble output vs without them.

---

## Bug #4: Gap 1-6 Models Missing from scoring_weights.yml (LOW)

**Location:** `config/scoring_weights.yml`

**Root cause:** The 6 Gap models (options_signals, cross_asset_signals, event_calendar, geographic_metrics, ensemble_diversity, model_routing) all use hardcoded default values for their internal weights/thresholds. They are not configurable via the scoring_weights.yml file that the dashboard panel reads.

**Fix:** Add configuration sections for each Gap model to `scoring_weights.yml` with their current hardcoded defaults. Then update each module to read from config with fallback to current hardcoded values.

**File changes:**
- `config/scoring_weights.yml`: Add 6 new top-level sections:
  - `options_signals`: component weights for put/call ratio, risk reversal, etc.
  - `cross_asset_signals`: sector rotation thresholds, stress composite weights
  - `event_calendar`: proximity decay rates, uncertainty premium multipliers
  - `geographic_metrics`: geo_hhi thresholds, china_pct threshold
  - `model_routing`: trend/volatile/mean_revert model affinity weights
  - `ensemble_diversity`: reject_option confidence threshold

- `operator1/features/options_signals.py`: Add `get_weight()` reads with fallback
- `operator1/features/cross_asset_signals.py`: Add `get_weight()` reads with fallback
- `operator1/features/event_calendar.py`: Add `get_weight()` reads with fallback
- `operator1/features/product_metrics.py`: Add `get_weight()` reads with fallback
- `operator1/models/prediction_aggregator.py`: Add `get_weight()` reads for routing + reject thresholds

**Verification:** Modify a value in scoring_weights.yml, run pipeline, confirm the changed value is used (via logging or output change).

---

## Implementation Order

1. **Bug #1** (HIGH, 1 line change) -- fixes double execution, immediate impact
2. **Bug #3** (MEDIUM, ~15 lines) -- makes Gap 6 routing functional
3. **Bug #2** (LOW, ~15 lines) -- dashboard display improvement
4. **Bug #4** (LOW, ~60 lines across 7 files) -- config externalization
