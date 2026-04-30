# Staged-First Architecture: Fix 11 Bugs + Eliminate Dual-Path

## Problem

main.py has TWO execution paths for temporal models (Step 6):

1. **Inline path** (lines 2607-3501): 900 lines of inline model calls when `--stage` is NOT set
2. **Staged path** (lines 2519-2605): Delegates to `run_stages()` when `--stage` IS set

These two paths must be kept in sync. They aren't. The debug scan found 11 bugs caused by this dual architecture:

- 5 bugs: recursive aggregator missing from various paths
- 2 bugs: main.py staged copy-back missing results
- 4 bugs: backtest_runner missing profile injections and state attrs

**Root cause:** Every new model added requires changes in 4 places (staged module, main.py inline, main.py copy-back, backtest_runner). Any miss creates a silent data loss bug.

## Solution: Staged-First

Make the staged pipeline the ONLY execution path. Delete the 900-line inline path from main.py. All three entry points (main.py, backtest_runner.py, dashboard.py) use the same staged runner.

```
BEFORE:
  main.py --stage    --> run_stages() --> PipelineState
  main.py (no flag)  --> 900 lines inline code --> local vars
  backtest_runner.py --> own stage functions --> BacktestState

AFTER:
  main.py            --> run_stages() --> PipelineState --> profile
  backtest_runner.py --> run_stages() --> PipelineState --> profile
  dashboard.py       --> subprocess main.py --> same path
```

## Implementation Steps

### Phase 1: Fix the 11 bugs (immediate, no architecture change)

These are targeted fixes that work regardless of the staged-first migration.

**1.1** `stage6_ensemble.py:468-476` -- Remove `ensemble_weights` variable and parameter from `run_6_11_recursive_predictions()`. Just delete lines 468-470 and remove `ensemble_weights=ensemble_weights` from the call at line 473.

**1.2** `main.py:2588` -- Add `recursive_result = _ps.recursive_result` after `ohlc_result = _ps.ohlc_result`

**1.3** `main.py:2588` -- Add line to copy feature_selection_result back. Currently it goes into the staged runner but the local var for profile injection at line 4238 is never set from _ps. Add: `feature_selection_result = _ps.feature_selection_result` (Note: check if it's already handled via a different variable name).

**1.4** `backtest_runner.py BacktestState.__init__()` -- Add three missing attributes:
```python
self.recursive_result = None
self.regime_shift_result = None  # was local var in run_stage2b
self.model_diagnostics_result = None  # was local _diag in run_stage2d
```

**1.5** `backtest_runner.py run_stage2b()` -- Change `regime_shift_result = predict_regime_shifts(...)` to `state.regime_shift_result = predict_regime_shifts(...)`

**1.6** `backtest_runner.py run_stage2c()` -- Add recursive aggregator call after OHLC predictor (lines ~1937), matching main.py pattern:
```python
# Recursive day-by-day predictions
try:
    from operator1.models.recursive_aggregator import run_recursive_predictions
    if state.forward_pass_result is not None and hasattr(state.forward_pass_result, 'model_states') and state.forward_pass_result.model_states:
        _rc_tm = getattr(state.mc_result, 'transition_matrix', None) if state.mc_result else None
        _rc_ro = getattr(state.mc_result, 'regime_order', None) if state.mc_result else None
        state.recursive_result = run_recursive_predictions(
            cache=cache, model_states=state.forward_pass_result.model_states,
            transition_matrix=_rc_tm, regime_order=_rc_ro,
        )
except Exception:
    pass
```

**1.7** `backtest_runner.py run_stage2d()` -- Store model_diagnostics on state:
Change `_diag = compute_model_diagnostics(...)` to `state.model_diagnostics_result = compute_model_diagnostics(...)`

**1.8** `backtest_runner.py run_stage3()` -- Add 4 missing profile injections:
```python
# recursive_predictions
if getattr(state, 'recursive_result', None) and state.recursive_result.available:
    profile["extended_models"]["recursive_predictions"] = state.recursive_result.to_dict()

# predicted_regime_shifts
if getattr(state, 'regime_shift_result', None) and state.regime_shift_result.available:
    profile["predicted_regime_shifts"] = state.regime_shift_result.to_dict()
else:
    profile["predicted_regime_shifts"] = {"available": False}

# feature_selection
if getattr(state, 'feature_selection_result', None) and state.feature_selection_result.fitted:
    profile["feature_selection"] = {
        "available": True,
        "n_input": state.feature_selection_result.n_input,
        "n_output": state.feature_selection_result.n_output,
        "boruta_confirmed": state.feature_selection_result.boruta_confirmed[:20],
        "mrmr_selected": state.feature_selection_result.mrmr_selected[:15],
    }
else:
    profile["feature_selection"] = {"available": False}

# model_diagnostics
if getattr(state, 'model_diagnostics_result', None) and state.model_diagnostics_result.available:
    profile["model_diagnostics"] = state.model_diagnostics_result.to_dict()
else:
    profile["model_diagnostics"] = {"available": False}
```

### Phase 2: Make main.py staged-first (architecture change)

**2.1** In main.py `main()`, after Step 5k.2, ALWAYS create PipelineState and delegate to `run_stages()`:

Replace the `if args.stage and not args.skip_models:` / `elif not args.skip_models:` fork (lines 2519-3501, ~980 lines) with:

```python
if not args.skip_models:
    from operator1.pipeline_state import PipelineState
    from operator1.stages.runner import run_stages

    _run_dir = args.run_dir or f"{args.output_dir}/{args.company}_{args.end_date or 'latest'}"
    _stage_spec = args.stage if args.stage else "all"

    # Build PipelineState from current local variables
    _ps = PipelineState(...)  # same as current lines 2524-2561
    
    # populate _ps with all Step 1-5 results (same as current)
    
    # Run temporal stages
    run_stages(_ps, _stage_spec, save_checkpoints=bool(args.stage))
    
    # Copy ALL results back from PipelineState
    # (single authoritative list -- no more forgetting to add new results)
    for attr in [
        'forecast_result', 'forward_pass_result', 'walk_forward_result',
        'burnout_result', 'mc_result', 'pred_result', 'transfer_entropy_result',
        'cycle_result', 'pattern_result', 'copula_result', 'conformal_result',
        'dtw_result', 'shap_result', 'sobol_result', 'particle_filter_result',
        'transformer_result', 'dual_regime_result', 'granger_result',
        'ga_result', 'ohlc_result', 'recursive_result', 'regime_shift_result',
        'scenario_result', 'retro_params', 'model_diagnostics_result',
        'multi_frequency_result', 'hf_result', 'mode_weights',
        'tv_granger_result', 'mv_mc_result', 'feature_selection_result',
        'synergy_meta', 'pattern_drift', 'weights', 'regime_detector',
        'economic_plane',
    ]:
        val = getattr(_ps, attr, None)
        if val is not None:
            locals()[attr] = val  # or assign to named vars explicitly
```

This eliminates the 900-line inline path entirely. The staged runner IS the execution engine.

**2.2** Keep `save_checkpoints=False` when no `--stage` flag is passed (no disk I/O overhead for normal runs). When `--stage` is set, `save_checkpoints=True` for resume support.

**2.3** Delete the entire `elif not args.skip_models:` block (lines 2607-3501). This is the 900-line inline path that duplicates everything the staged runner already does.

### Phase 3: Unify backtest_runner with PipelineState

**3.1** Replace `BacktestState` with `PipelineState`. The `BacktestState` class is a 58-attribute copy of `PipelineState` with different naming conventions and missing attributes. Replace it entirely:

```python
from operator1.pipeline_state import PipelineState

# BacktestState is now just PipelineState with backtest-specific config
state = PipelineState(
    market_id=args.market,
    company=args.company,
    end_date=args.end_date,
    years=args.years,
    output_dir=run_dir,
)
```

**3.2** Update `run_stage1()` to populate PipelineState attributes instead of BacktestState-specific names (e.g., `state._income_df` -> `state.income_df`, `state._is_private` -> `state.is_private`).

**3.3** Replace `run_stage2a/2b/2c/2d` with calls to `run_stages()`:
```python
# Instead of:
run_stage2a1(state)
run_stage2a2(state)
run_stage2b(state)
run_stage2c(state)
run_stage2d(state)

# Use:
run_stages(state, "3-7", save_checkpoints=True)
```

This eliminates the SECOND copy of every model call. backtest_runner becomes a thin wrapper that does Stage 1 (data fetch) then delegates temporal analysis to the same staged runner that main.py uses.

## Files Changed

| Phase | File | Change |
|-------|------|--------|
| 1.1 | `operator1/stages/stage6_ensemble.py` | Remove ensemble_weights from 6.11 |
| 1.2-1.3 | `main.py` | Add 2 missing copy-back lines |
| 1.4-1.8 | `backtest_runner.py` | Add 3 state attrs, store regime_shift/diagnostics on state, add recursive call, add 4 profile injections |
| 2.1-2.3 | `main.py` | Delete 900-line inline path, always use run_stages() |
| 3.1-3.3 | `backtest_runner.py` | Replace BacktestState with PipelineState, replace stage2a-2d with run_stages() |

## Risk Assessment

- **Phase 1** is zero-risk: targeted fixes, no architecture change
- **Phase 2** is medium-risk: deleting 900 lines, but the staged runner is already tested and covers all models
- **Phase 3** is higher-risk: backtest_runner Stage 1 has backtest-specific logic (date filtering, CompanyFacts fallback) that doesn't exist in the staged runner. Need to either add these to PipelineState or keep Stage 1 as custom code.

## Recommendation

Implement Phase 1 first (fix all 11 bugs). Then Phase 2 (staged-first main.py). Phase 3 can be done later since backtest_runner is a secondary path.
