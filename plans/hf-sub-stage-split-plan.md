# HF Sub-Stage Split Plan

*Split monolithic sub-stage 7.5 into 4 sub-stages with per-stage timeouts and graceful degradation.*

---

## Current State

Sub-stage 7.5 runs `run_hedge_fund_analysis()` which executes ~68 methods in sequence:
- 15 base metrics (Tiers 1-5)
- 15 advanced methods (P1/P2/P3)
- 8+3 fusion methods
- 30 new methods from this PR
- Scorecard + position signal

**Single timeout: 300s.** If any method hangs, everything after it is lost.

## Proposed Split

| Sub-Stage | ID | Methods | Timeout | Critical? | Est. Time |
|-----------|-----|---------|---------|-----------|-----------|
| **HF Base** | 7.5.1 | 15 base metrics + EVA + SOTP + CVaR + SGR + scorecard + position | 180s | Yes | ~80s |
| **HF Advanced** | 7.5.2 | 15+7 advanced methods (Piotroski, Merton term, credit migration, moat, factor exposure, equity duration, governance, capital allocation) | 120s | No | ~30s |
| **HF Multi-Freq** | 7.5.3 | 3 MF variants (FCF quality MF, accruals MF, growth quality MF) with coherence testing | 180s | No | ~60s |
| **HF Fusion** | 7.5.4 | 8+3 fusion methods (HRP, Brier, anomaly routing, meta-ensemble, Bayesian, etc.) | 60s | No | ~10s |

### Why This Split

1. **7.5.1 is critical** -- the base scorecard + position signal are core pipeline outputs. If advanced methods fail, the scorecard still has the 15 base metrics.

2. **7.5.2 is non-critical** -- advanced methods add depth but the HF thesis works without them. Factor exposure OLS can be slow. Merton term structure involves root-finding.

3. **7.5.3 is the heaviest** -- each MF variant runs a full per-frequency pipeline (derived variables + survival + regime + forecasting + MC). With 2-3 frequencies, this is 3 x 20s = 60s. If it times out, base metrics from 7.5.1 are unaffected.

4. **7.5.4 is fast but depends on all prior** -- fusion reads from base + advanced + MF. If 7.5.2 or 7.5.3 failed, fusion still works with whatever is available (graceful degradation already built in).

## Implementation

### Files Modified

| File | Changes |
|------|---------|
| [`operator1/hedge_fund/engine.py`](operator1/hedge_fund/engine.py) | Split `run_hedge_fund_analysis()` into 4 public functions: `run_hf_base()`, `run_hf_advanced()`, `run_hf_multi_freq()`, `run_hf_fusion()` |
| [`operator1/stages/stage7_integration.py`](operator1/stages/stage7_integration.py) | Replace single `run_7_5_hedge_fund` with 4 sub-stage functions + register in `STAGE_7_SUBSTAGES` |
| [`operator1/stages/runner.py`](operator1/stages/runner.py) | Add 7.5.1-7.5.4 to `_SUBSTAGE_TIMEOUTS` and `_CRITICAL_SUBSTAGES` |
| [`run_backtest_staged.py`](run_backtest_staged.py) | Add 4 HF sub-stages to `ALL_STAGES` list |

### engine.py Changes

```python
# NEW: 4 public entry points replacing single run_hedge_fund_analysis()

def run_hf_base(state) -> None:
    """7.5.1: Run 15 base metrics + EVA + SOTP + CVaR + SGR + scorecard + position."""
    # Contains: Tier 1-5 + new Phase 2/4/5 inline methods
    # Stores: state.hf_result (HedgeFundResult with base metrics filled)

def run_hf_advanced(state) -> None:
    """7.5.2: Run 15+7 advanced methods."""
    # Contains: run_advanced_methods() with all P1/P2/P3 + new Phase 3/5/7/8 methods
    # Requires: state.hf_result from 7.5.1
    # Stores: state.hf_result.advanced dict updated

def run_hf_multi_freq(state) -> None:
    """7.5.3: Run 3 MF variants with coherence testing."""
    # Contains: FCF quality MF, accruals MF, growth quality MF
    # Requires: state.hf_result from 7.5.1, state.*_freq_groups
    # Stores: state.hf_result with MF-enhanced scores

def run_hf_fusion(state) -> None:
    """7.5.4: Run 11 fusion methods."""
    # Contains: run_fusion() with all 11 methods
    # Requires: state.hf_result from 7.5.1 (minimum), 7.5.2/7.5.3 (optional)
    # Stores: state.hf_result.fusion dict
```

### stage7_integration.py Changes

```python
# REPLACE:
("7.5", run_7_5_hedge_fund),

# WITH:
("7.5.1", run_7_5_1_hf_base),
("7.5.2", run_7_5_2_hf_advanced),
("7.5.3", run_7_5_3_hf_multi_freq),
("7.5.4", run_7_5_4_hf_fusion),
```

### runner.py Changes

```python
# Add to _SUBSTAGE_TIMEOUTS:
"7.5.1": 180,   # base metrics + scorecard
"7.5.2": 120,   # advanced methods
"7.5.3": 180,   # MF variants (per-frequency pipelines)
"7.5.4": 60,    # fusion

# Add to _CRITICAL_SUBSTAGES:
"7.5.1"  # base scorecard is essential (replace "7.5")

# Remove from _CRITICAL_SUBSTAGES:
"7.5"    # no longer exists

# Add to _SUBSTAGE_REQUIREMENTS:
"7.5.1": ["cache", "income_df", "balance_df", "cashflow_df"],
"7.5.2": ["cache"],  # hf_result checked at runtime
"7.5.3": ["cache"],  # freq_groups checked at runtime
"7.5.4": ["cache"],  # hf_result checked at runtime
```

### run_backtest_staged.py Changes

```python
# REPLACE in ALL_STAGES:
("7.5", "Hedge Fund Analysis (15 metrics + fusion)"),

# WITH:
("7.5.1", "Hedge Fund: Base Metrics + Scorecard (15 metrics)"),
("7.5.2", "Hedge Fund: Advanced Methods (22 techniques)"),
("7.5.3", "Hedge Fund: Multi-Frequency Variants (3 MF metrics)"),
("7.5.4", "Hedge Fund: Cross-Pipeline Fusion (11 methods)"),
```

### State Persistence Between Sub-Stages

The key change: `state.hf_result` must persist between 7.5.1 -> 7.5.2 -> 7.5.3 -> 7.5.4. This already works via `PipelineState.save()` / `load_checkpoint()` since `hf_result` is a field on PipelineState and serializes via pickle.

### Execution Order

```
7.5.1 (CRITICAL, 180s)
  |
  +-> state.hf_result = HedgeFundResult(
  |     fcf_quality, accruals_forensic, smoothing,
  |     dividend_burn, return_spread, operating_leverage,
  |     obs_risk, asset_quality, leverage_stress,
  |     momentum, growth_quality, earnings_surprise,
  |     dcf, valuation_quality, peg_composite,
  |     eva, sotp, risk_attribution,
  |     scorecard, position
  |   )
  v
7.5.2 (non-critical, 120s)
  |
  +-> state.hf_result.advanced = {
  |     piotroski, balance_sheet_velocity, earnings_persistence,
  |     forensic_cashflow, ou_mean_reversion, accruals_rank,
  |     altman_z_double_prime, garch_vol, insider_alignment,
  |     capital_cycle, earnings_torpedo, vrp_proxy,
  |     implied_cost_of_capital, cross_asset_regime,
  |     merton_term_structure, credit_migration, moat,
  |     factor_exposure, equity_duration, governance,
  |     capital_allocation
  |   }
  v
7.5.3 (non-critical, 180s)
  |  [ONLY if _has_freq_groups]
  +-> Overrides: hf.fcf_quality, hf.accruals_forensic,
  |   hf.growth_quality with MF-enhanced versions
  |   Re-runs: scorecard + position with MF data
  v
7.5.4 (non-critical, 60s)
  |
  +-> state.hf_result.fusion = {
       fused_signal, fused_label, fused_conviction,
       anomaly_overrides, hrp_weights, calibrated_conviction,
       belief_posterior, action, primary_risk, next_catalyst
     }
```

### Graceful Degradation Matrix

| 7.5.1 | 7.5.2 | 7.5.3 | 7.5.4 | Result |
|-------|-------|-------|-------|--------|
| OK | OK | OK | OK | Full HF analysis |
| OK | FAIL | OK | OK | Missing advanced methods, fusion still works |
| OK | OK | FAIL | OK | Base metrics used (no MF enhancement) |
| OK | OK | OK | FAIL | No fused signal, HF scorecard + position still available |
| OK | FAIL | FAIL | FAIL | Base scorecard + position still available |
| FAIL | skip | skip | skip | Pipeline continues with `hf_result = None` |

## Checklist

- [ ] Split `run_hedge_fund_analysis()` into 4 functions in `engine.py`
- [ ] Create 4 sub-stage wrapper functions in `stage7_integration.py`
- [ ] Update `STAGE_7_SUBSTAGES` registry
- [ ] Update `_SUBSTAGE_TIMEOUTS`, `_CRITICAL_SUBSTAGES`, `_SUBSTAGE_REQUIREMENTS` in `runner.py`
- [ ] Update `ALL_STAGES` in `run_backtest_staged.py`
- [ ] Verify state persistence between sub-stages
- [ ] Test: 7.5.2 failure doesn't block 7.5.3/7.5.4
