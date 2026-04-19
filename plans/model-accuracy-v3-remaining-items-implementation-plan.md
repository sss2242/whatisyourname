# Model Accuracy v3 -- Remaining Items

Cross-reference of the 6-Gap plan against what was implemented in PRs #1-#6 and our current PR, identifying exactly what remains.

---

## Status Summary

| Gap | Title | Core Feature | Wiring (main.py) | Wiring (backtest) | Wiring (stages) | Report | Dashboard | Profile |
|-----|-------|-------------|-------------------|-------------------|-----------------|--------|-----------|---------|
| 1 | Options Signals | DONE (PR #1) | DONE | DONE | DONE (our PR) | DONE (our PR, S31) | DONE (our PR) | DONE (our PR) |
| 2 | Geographic Supply Chain | PARTIAL | DONE | DONE | DONE (our PR) | Not needed (data flows via extra_vars) | Not needed | DONE (via product_segments) |
| 3 | Cross-Asset Rotation | DONE (PR #3) | DONE | DONE | DONE (our PR) | DONE (our PR, S32) | DONE (our PR) | DONE (our PR) |
| 4 | Event Calendar | DONE (PR #4) | DONE | DONE | DONE (our PR) | DONE (our PR, S33) | DONE (our PR) | DONE (our PR) |
| 5 | DCF Calibration | DONE (PR #5) | DONE (automatic) | DONE (automatic) | DONE (in 7.5) | DONE (existing HF sections) | Not needed | DONE (hedge_fund.dcf) |
| 6 | Ensemble Diversity | DONE (PR #6) | DONE (automatic) | DONE (automatic) | DONE (in 6.5) | Not needed | Not needed | Not needed |

---

## Remaining Items (Gap 2 incomplete)

### Gap 2 Remaining: EU ESEF Geographic Dimensions

**Status:** US EDGAR geographic extraction is DONE (our B5 fix). But 2 items from the plan remain:

#### 2a. EU ESEF Geographic Axis Parsing

**File:** `operator1/clients/eu_esef_wrapper.py`
**What:** Add `ifrs-full:GeographicAreasAxis` to the XBRL dimension search in `_extract_segments_from_xbrl_dimensions()`
**Why:** EU companies report geographic segments under IFRS 8 using the `GeographicAreasAxis` dimension. Currently only `OperatingSegmentsMember` is searched.
**Scope:** ~15 lines -- add axis to the existing dimension search loop and return `geo_segments` alongside `segments`

#### 2b. Trade Policy Uncertainty Index

**File:** `operator1/clients/macro_provider.py`
**What:** Add `fetch_trade_policy_uncertainty()` function
**Source:** `policyuncertainty.com/data/Trade_Policy_Uncertainty_Index.csv` (free, no key)
**Why:** The `trade_policy_uncertainty` and `tariff_exposure_score` cache columns are computed in `product_metrics.compute_geographic_metrics()` but `trade_policy_uncertainty` currently comes from nowhere -- it's always NaN. The plan specified fetching this from the Baker-Bloom-Davis TPU index.
**Scope:** ~40 lines -- download CSV, parse, resample to daily, cache to disk
**Wiring:** Call from `main.py` after macro fetch, inject into cache. Also in `backtest_runner.py` `run_stage1()`

---

## Items Already Done (no action needed)

### Gap 1 -- Options Signals: COMPLETE
- `options_signals.py` exists (PR #1)
- Wired in main.py Step 4a.7
- Wired in backtest_runner.py run_stage1()
- Wired in stage3_temporal.py _extra_vars (our PR commit 4)
- Report section 31 (our PR commit 2)
- Dashboard card (our PR commit 2)
- Profile key injection in backtest_runner.py (our PR commit 2)

### Gap 2 -- Geographic Supply Chain: MOSTLY COMPLETE
- `compute_geographic_metrics()` exists in product_metrics.py (earlier PR)
- US EDGAR `_extract_geo_segments_from_text()` added (our PR commit 3)
- `_compute_geo_hhi()` renamed (our PR commit 1)
- Wired in main.py Step 5i.6
- Wired in backtest_runner.py run_stage1()
- Stage3 _extra_vars synced (our PR commit 4)
- **REMAINING:** EU ESEF geographic axis + Trade Policy Uncertainty fetch

### Gap 3 -- Cross-Asset Rotation: COMPLETE
- `cross_asset_signals.py` exists (PR #3)
- All wiring done
- Report section 32 (our PR)
- Dashboard card (our PR)

### Gap 4 -- Event Calendar: COMPLETE
- `event_calendar.py` exists (PR #4)
- `config/event_calendar.json` exists
- All wiring done
- Report section 33 (our PR)
- Dashboard card (our PR)

### Gap 5 -- DCF Calibration: COMPLETE
- All 5 DCF fixes implemented in `hedge_fund/engine.py` (PR #5):
  1. Growth prior from company CAGR (not regime distribution)
  2. WACC cap at sector median + 2%
  3. 3-stage model for high-growth companies
  4. Reverse DCF (implied growth rate)
  5. Sanity gate (DCF/price ratio check)

### Gap 6 -- Ensemble Diversity: COMPLETE
- All 4 changes implemented (PR #6):
  1. Feature-driven model router (`_compute_model_routing_weights`)
  2. Adaptive FixedShare (share parameter scales with online_change_score)
  3. Equal-weight fallback in GA (diversity mechanism when single model dominates)
  4. Reject option (interval > 2x price = "no opinion")

---

## Implementation Plan for Remaining Items

### Step 1: EU ESEF Geographic Axis (Gap 2a)

**File:** `operator1/clients/eu_esef_wrapper.py`

Find the `_extract_segments_from_xbrl_dimensions()` method (or equivalent segment extraction). Add `ifrs-full:GeographicAreasAxis` alongside the existing operating segment axes. When geographic dimension members are found, store them in a separate `geo_segments` dict and return alongside `segments`.

Pattern:
```
existing: segments["Product A"] = 1000000
new:      geo_segments["Europe"] = 500000, geo_segments["Americas"] = 300000
```

### Step 2: Trade Policy Uncertainty Index (Gap 2b)

**File:** `operator1/clients/macro_provider.py` (or new utility)

Add `fetch_trade_policy_uncertainty()`:
1. Download CSV from `policyuncertainty.com/data/Trade_Policy_Uncertainty_Index.csv`
2. Parse date + TPU value columns
3. Resample to daily via forward-fill
4. Cache to `cache/tpu_index.parquet` with 30-day refresh

Wire in `main.py` after macro fetch (Step 4a) and in `backtest_runner.py` `run_stage1()`:
- Inject as `cache["trade_policy_uncertainty"]`
- `tariff_exposure_score` in `compute_geographic_metrics()` will then have non-NaN TPU values

### Step 3: Verify and Test

1. AST parse all modified files
2. Import validation for new functions
3. Run partial Stage 1 to verify TPU fetch works
4. Commit, push, update PR

---

## Execution Order

```
[ ] Step 1: eu_esef_wrapper.py -- add geographic axis to XBRL dimension search
[ ] Step 2: macro_provider.py -- add fetch_trade_policy_uncertainty()
[ ] Step 3: main.py -- wire TPU fetch after macro data
[ ] Step 4: backtest_runner.py -- wire TPU fetch in run_stage1()
[ ] Step 5: Verify all files parse cleanly
[ ] Step 6: Commit, push
```

All other items from the v3 plan are already implemented across PRs #1-#6 and our current PR.
