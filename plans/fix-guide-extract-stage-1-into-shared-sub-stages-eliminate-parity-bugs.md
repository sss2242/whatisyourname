# Fix Guide: Extract Stage 1 Into Shared Sub-Stages

*Created: 2026-05-13 | Priority: High (structural, eliminates recurring bug class)*

## Problem

Since the staged-first refactor on 2026-04-30 (commit `43ed751`), there have been **13 separate parity fix commits** for `backtest_runner.py`. Every time a feature or parameter is added to `main.py` Stage 1, the same change must be manually replicated in `backtest_runner.py` -- and is frequently missed.

**Examples of parity bugs found:**
- `sector=` parameter missing from `compute_financial_health()` call (fixed 2026-05-13)
- Forward pass KeyError 0 crash (fixed 2026-05-03)
- CopulaResult attribute mismatches (fixed 2026-05-03)
- Frequency separator not wired (fixed 2026-05-02)
- MC jump params not passed (fixed 2026-05-02)
- DERIVED_VARIABLES sync (fixed 2026-05-01)
- 13 HMM look-ahead + missing kwargs (fixed 2026-04-15)
- Signal IC + prediction log not in state (fixed 2026-04-03)

**Root cause:** `backtest_runner.py` has its own ~1,000 line `run_stage1()` function that duplicates `main.py` Steps 1-5. Stages 2-7 don't have this problem because they're in shared modules (`operator1/stages/stage2-7_*.py`) that both entry points call.

## Architecture: Current vs Target

```
CURRENT (two code paths):

main.py Steps 1-5           backtest_runner.py run_stage1()
  |                           |
  | (~1,800 lines inline)     | (~1,000 lines, partial copy)
  |                           |
  v                           v
operator1/stages/stage2-7    operator1/stages/stage2-7
  (shared, no parity bugs)    (shared, no parity bugs)


TARGET (single code path):

main.py                     backtest_runner.py
  |                           |
  v                           v
operator1/stages/stage1_*   operator1/stages/stage1_*
  (shared sub-stage modules)  (same modules)
  |                           |
  v                           v
operator1/stages/stage2-7   operator1/stages/stage2-7
  (shared, no parity bugs)    (shared, no parity bugs)
```

## Plan: 10 Sub-Stage Modules

Extract `main.py` Steps 1-5 into 10 shared sub-stage functions following the exact same pattern as Stages 2-7.

### New File: `operator1/stages/stage1_acquisition.py`

```
STAGE_1_SUBSTAGES = [
    ("1.1", run_1_1_profile),
    ("1.2", run_1_2_financials),
    ("1.3", run_1_3_ohlcv_holders),
    ("1.4a", run_1_4a_cache_build),
    ("1.4b", run_1_4b_macro_risk),
    ("1.5", run_1_5_estimation_features),
    ("1.6", run_1_6_entity_discovery),
    ("1.7", run_1_7_adaptive_calibration),
    ("1.8a", run_1_8a_regime_timeline),
    ("1.8b", run_1_8b_finalization),
]
```

### Sub-Stage Mapping (main.py -> shared module)

| Sub-Stage | main.py Lines | Content | PipelineState Fields Written |
|-----------|--------------|---------|------------------------------|
| **1.1** | 555-688 | Select market, create PIT client, search company, fetch profile, supplement enrichment | `target_profile`, `market_id`, `company` |
| **1.2** | 710-830 | Fetch income/balance/cashflow/quotes in parallel, reconcile, pivot, frequency separate | `income_df`, `balance_df`, `cashflow_df`, `quotes_df`, `income_freq_groups`, `balance_freq_groups`, `cashflow_freq_groups` |
| **1.3** | 690-709, 757-770 | OHLCV fallback, benchmark returns, IV, sector leaders, holders, insiders, segments | `quotes_df`, `target_holders`, `target_insiders`, `seg_result`, `ohlcv_source_label` |
| **1.4a** | 896-1110 | Build daily cache from OHLCV spine + statement merge + shares_outstanding injection + backtest filter | `cache` |
| **1.4b** | 1116-1230 | Cross-asset signals, options signals, macro fetch, macro quadrant, validation, institutional ownership | `macro_data`, `macro_dataset`, `macro_quadrant_result`, `conflict_result`, `buying_power_result`, `cross_asset_result`, `options_signal_result` |
| **1.5** | 1344-1532 | Pre-estimation ratios, estimation engine, filing calendar, event calendar, derived variables, survival mode, Cox PH, FH scoring, vanity | `estimation_coverage`, `filing_calendar_result`, `event_calendar_result`, `fh_result`, `weights` |
| **1.6** | 1736-2110 | LLM client creation, entity discovery, GLEIF, graph risk, game theory, linked entity data fetch, ownership contagion, linked aggregates, peer ranking, sentiment, catalysts, segments | `relationships`, `linked_caches`, `linked_agg_df`, `graph_risk_result`, `game_theory_result`, `contagion_result`, `peer_ranking_result`, `sentiment_result`, `catalyst_result` |
| **1.7** | 2333-2663 | Adaptive thresholds, threshold registry, USS refresh, signal IC, prediction log, adaptive model params, adaptive windows | `adaptive_thresholds`, `adaptive_model_params`, `adaptive_tier3`, `signal_ic_result`, `prediction_log_summary`, `survival_controller` |
| **1.8a** | 2452-2549 | Early regime detection, ChangeFinder, enriched survival timeline | `early_regime_result`, `regime_detector`, `enriched_timeline_result` |
| **1.8b** | 2111-2324 | Linked conflict, supply chain stress, peer ranking, behavioral signals, complexity signals, feature normalization | Various feature columns in cache |

### Implementation Steps

**Step 1: Create `operator1/stages/stage1_acquisition.py`** (~1,200 lines)

Extract each sub-stage function from `main.py`. Each function:
- Takes `PipelineState` as its only argument
- Reads inputs from `state.*` fields
- Writes outputs to `state.*` fields
- Follows the same try/except + logger.warning pattern as stage3-7 modules

**Step 2: Update `operator1/stages/runner.py`**

Add `STAGE_1_SUBSTAGES` to the registry in `_build_registry()`:

```python
def _build_registry():
    from operator1.stages.stage1_acquisition import STAGE_1_SUBSTAGES
    from operator1.stages.stage2_preprocessing import STAGE_2_SUBSTAGES
    ...
    return STAGE_1_SUBSTAGES + STAGE_2_SUBSTAGES + ...
```

**Step 3: Simplify `main.py`**

Replace the ~1,800 lines of inline Stage 1 code with:

```python
from operator1.stages.runner import run_stages
_ps = PipelineState(market_id=..., company=..., ...)
run_stages(_ps, "all", save_checkpoints=bool(args.stage))
```

This means `main.py` goes from ~3,700 lines to ~700 lines (just CLI parsing + profile build + report).

**Step 4: Simplify `backtest_runner.py`**

Replace `run_stage1()` (~1,000 lines) with:

```python
def run_stage1(state):
    from operator1.stages.runner import run_stages
    run_stages(state, "1")
```

This means `backtest_runner.py` goes from ~2,332 lines to ~1,332 lines.

**Step 5: Add `_CRITICAL_SUBSTAGES` and `_SUBSTAGE_REQUIREMENTS` for Stage 1**

```python
_CRITICAL_SUBSTAGES.update({"1.1", "1.2", "1.4a"})
_SUBSTAGE_REQUIREMENTS["1.2"] = ["target_profile"]
_SUBSTAGE_REQUIREMENTS["1.4a"] = ["quotes_df"]
_SUBSTAGE_REQUIREMENTS["1.5"] = ["cache"]
_SUBSTAGE_REQUIREMENTS["1.6"] = ["cache", "target_profile"]
```

**Step 6: Update `run_backtest_staged.py`**

The `ALL_STAGES` list already has the right sub-stage IDs (1.1-1.8b). Just need to ensure `backtest_runner.py` routes these to the shared modules instead of its own `run_stage1()`.

### PipelineState Changes

Add 2 new fields needed for Stage 1 sub-stages:

```python
# In PipelineState.__init__():
self._pit_client: Any = None     # already exists (transient)
self._llm_client: Any = None     # already exists (transient)
self.interactive: bool = False    # new: controls interactive prompts
self.cli_args: dict = {}          # new: parsed CLI arguments
```

### Risk Assessment

**Medium risk.** This is a large refactoring that touches the two most critical files (`main.py` and `backtest_runner.py`). Mitigations:

1. **Incremental approach:** Extract one sub-stage at a time, test after each extraction
2. **Backward compatibility:** Keep the `backtest_runner.py` Stage 1 functions as thin wrappers that call the shared modules, so the staged compiler doesn't need changes
3. **Test with AAPL backtest:** Run the full 63-stage backtest after each sub-stage extraction to catch regressions
4. **Feature branch:** All work on a dedicated branch with PR review

### Expected Outcome

- **0 parity bugs** for Stage 1 going forward (single code path)
- `main.py` reduced from ~3,700 to ~700 lines
- `backtest_runner.py` reduced from ~2,332 to ~1,332 lines
- Stage 1 gets checkpoint save/resume (currently only Stages 2-7 have this)
- Stage 1 sub-stages get graceful degradation + timeout protection

### Estimated Scope

- 1 new file: `operator1/stages/stage1_acquisition.py` (~1,200 lines, extracted from main.py)
- 3 modified files: `main.py` (-1,800 lines), `backtest_runner.py` (-1,000 lines), `operator1/stages/runner.py` (+10 lines)
- Total net: ~-1,600 lines (deletion of duplicate code)
