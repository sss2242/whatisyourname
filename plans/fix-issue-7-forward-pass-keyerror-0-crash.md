# Fix Issue 7: Forward Pass KeyError: 0 Crash

## Problem

The forward pass in sub-stage 5.1 crashes with `KeyError: 0` at `forecasting.py:4039`, causing `forward_pass_result = None`. This cascades to:
- Conformal prediction (6.3) skipped
- SHAP explanations (6.6) empty
- Burn-out calibration (5.2) empty
- Prediction bands too narrow

## Root Cause

At `forecasting.py:3892`:
```python
tier_num = int(tier_key.replace("tier", "")) if "tier" in tier_key else 0
```

Variables not mapped to a survival tier (extra_vars like `return_1d`, `volatility_21d`, etc.) get `tier_num = 0`. But `ForwardPassResult.errors_by_tier` only has keys 1-5.

At line 4039: `result.errors_by_tier[tier_num].append(weighted_error)` raises `KeyError: 0`.

## Fix: Two-line change

**Line 4039** -- Use `setdefault` to handle tier 0 gracefully:
```python
result.errors_by_tier.setdefault(tier_num, []).append(weighted_error)
```

This is the safest fix because:
1. It doesn't change the tier assignment logic (tier 0 = "no tier" is semantically correct)
2. It doesn't change the survival hierarchy (tiers 1-5 are the meaningful ones)
3. Tier 0 errors are collected but don't affect hierarchy weight recalibration (which only looks at tiers 1-5)
4. Zero risk of breaking any downstream consumer

## Verification

After fix, re-run sub-stage 5.1:
```bash
python3.12 backtest_runner.py --stage 5.1 --run-dir cache/backtest_AAPL_2024-12-31
```

Expected:
- `forward_pass_result` is non-None in `state_5.1.pkl`
- `total_days` > 0
- `predictions_log` has entries
- Sub-stage 6.3 (conformal) no longer skipped
