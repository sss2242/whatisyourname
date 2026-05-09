# Batch C: Full Implementation Plan -- Integration Test Suite

*Problem 5: Prevent regressions with golden data + layer-by-layer validation*
*Based on debug scan of all module connections + 16 patterns from 7 open-source projects*

---

## Debug Scan Summary

### What we're testing (module connection map)

```
Golden AAPL Cache (frozen parquet, ~500 rows x ~400 cols)
    |
    v
[Layer 1] compute_derived_variables(cache, freq="D")
    |       -> ~96 new columns + ~126 companion flags
    |       -> file: operator1/features/derived_variables.py:1667
    |       -> returns: pd.DataFrame (augmented cache)
    |
[Layer 2a] compute_company_survival_flag(cache, thresholds=None, freq="D", sector="")
    |       -> cache["company_survival_mode_flag"] (int 0/1)
    |       -> file: operator1/analysis/survival_mode.py:71
    |       -> threshold source: _COMPANY_THRESHOLDS (scoring_weights.yml)
    |       -> sector overrides: scoring_weights.yml sector_overrides
    |
[Layer 2b] compute_survival_probability(cache)
    |       -> cache["survival_probability"] (float 0-1)
    |       -> file: operator1/analysis/survival_mode.py:373
    |       -> uses SAME thresholds as 2a (critical consistency check)
    |
[Layer 2c] compute_hierarchy_weights(cache)
    |       -> cache["hierarchy_tier1-5_weight"], cache["survival_regime"]
    |       -> file: operator1/analysis/hierarchy_weights.py:289
    |
[Layer 2d] compute_financial_health(cache, hierarchy_weights, freq="D", sector="")
    |       -> cache + 19 fh_* columns, FinancialHealthResult
    |       -> file: operator1/models/financial_health.py:909
    |       -> FinancialHealthResult.latest_composite: float (0-100)
    |       -> FinancialHealthResult.latest_label: str
    |
[Layer 3a] run_monte_carlo(cache, survival_thresholds=DEFAULT_SURVIVAL_THRESHOLDS)
    |       -> MonteCarloResult.survival_probability: dict[str, float]
    |       -> file: operator1/models/monte_carlo.py:1264
    |       -> threshold source: DEFAULT_SURVIVAL_THRESHOLDS + SECTOR_SURVIVAL_OVERRIDES
    |       -> ** CRITICAL: must agree with 2a thresholds **
    |
[Layer 3b] run_forecasting(cache, extra_variables=[...])
    |       -> (cache, ForecastResult)
    |       -> file: operator1/models/forecasting.py:2148
    |       -> ForecastResult.forecasts: dict[str, dict[str, float]]
    |       -> ForecastResult.metrics: dict[str, ModelMetrics]
    |
[Layer 3c] predict_ohlc_series(cache, forecast_result, mc_result)
    |       -> OHLCResult with next_day, next_week, next_month, next_year
    |       -> file: operator1/models/ohlc_predictor.py:97
    |
[Layer 4] run_hedge_fund_analysis(income_df, balance_df, cashflow_df, cache, ...)
    |       -> HedgeFundResult.scorecard.investment_grade: str (A+ to F)
    |       -> HedgeFundResult.position.signal: float (-1 to +1)
    |       -> file: operator1/hedge_fund/engine.py:1511
    |
[Profile] build_company_profile(verified_target, cache, ...)
            -> profile dict with 88 profile[] key injections in main.py
            -> file: operator1/report/profile_builder.py:1121
```

### Existing test gaps (what's NOT tested)

| Bug Pattern | Times Occurred | Current Test Coverage |
|-------------|---------------|---------------------|
| FH score inversion (Apple=28 "Weak") | 3x | `test_financial_health.py` tests synthetic data only, never real company |
| MC drawdown NaN | 2x | `test_phase6_monte_carlo.py` tests synthetic 2-regime data, never checks NaN |
| Survival non-monotonic (1d=0%, 21d=99%) | 1x | No test |
| OHLC year-end crash ($78 from $245) | 1x | No test |
| Threshold disagreement (survival vs MC) | 4x | No test |
| Burn-out scale contamination | 2x | No test |
| Profile key missing/empty | 5x | No test (profile_schema.py exists but no test calls it) |
| Result class not serializable | 3x | No test |
| Column count regression | 2x | No test |

### Key numbers

- **90 Result classes** across `operator1/`
- **33 to_dict/to_profile_dict methods** (57 Result classes lack serialization -- potential profile gaps)
- **88 profile key injections** in `main.py`
- **No conftest.py** in `tests/`
- **No fixtures/ directory** in `tests/`
- **79 existing test files** (all use synthetic data or mock objects)

---

## Implementation Plan

### Phase 1: Golden Test Data Infrastructure

#### 1.1 Create fixture directory and conftest.py

**New file:** `tests/conftest.py`

```python
"""Session-scoped fixtures for integration testing.

Pattern: Deepchecks session-scoped model fixtures + GE validator architecture.
Expensive computations (FH scoring, forecasting) run once per pytest session.
"""
import pytest
import pandas as pd
from pathlib import Path

FIXTURE_DIR = Path(__file__).parent / "fixtures"

@pytest.fixture(scope="session")
def aapl_cache():
    """Frozen AAPL daily cache after L1 feature engineering (Step 5)."""
    path = FIXTURE_DIR / "aapl_cache.parquet"
    if not path.exists():
        pytest.skip("Golden fixture not generated. Run: python tests/generate_golden_data.py")
    return pd.read_parquet(path)

@pytest.fixture(scope="session")
def aapl_statements():
    """Frozen AAPL raw statement DataFrames."""
    stmts = {}
    for name in ("income", "balance", "cashflow"):
        path = FIXTURE_DIR / f"aapl_{name}.parquet"
        stmts[name] = pd.read_parquet(path) if path.exists() else pd.DataFrame()
    return stmts

@pytest.fixture(scope="session")
def aapl_profile():
    """Frozen AAPL target profile dict."""
    import json
    path = FIXTURE_DIR / "aapl_profile.json"
    if not path.exists():
        return {"name": "Apple Inc", "ticker": "AAPL", "sector": "Technology"}
    return json.loads(path.read_text())

@pytest.fixture(scope="session")
def aapl_fh_result(aapl_cache):
    """Session-scoped FH result (computed once, reused across tests)."""
    from operator1.models.financial_health import compute_financial_health
    cache, result = compute_financial_health(
        aapl_cache.copy(), sector="Technology"
    )
    return cache, result

@pytest.fixture(scope="session")
def aapl_survival(aapl_cache):
    """Session-scoped survival flag + probability (computed once)."""
    from operator1.analysis.survival_mode import (
        compute_company_survival_flag, compute_survival_probability,
    )
    flag = compute_company_survival_flag(aapl_cache, sector="Technology")
    prob = compute_survival_probability(aapl_cache)
    return flag, prob
```

**Connections exercised:**
- [`compute_financial_health()`](operator1/models/financial_health.py:909) -- L2d entry point
- [`compute_company_survival_flag()`](operator1/analysis/survival_mode.py:71) -- L2a entry point
- [`compute_survival_probability()`](operator1/analysis/survival_mode.py:373) -- L2b entry point

#### 1.2 Create golden data generator

**New file:** `tests/generate_golden_data.py` (~40 lines)

```python
"""Generate frozen AAPL test fixtures from live pipeline.

Run: python tests/generate_golden_data.py
Requires: EDGAR_IDENTITY env var (SEC email) + internet.
Run manually when test data needs refreshing (rarely).
"""
```

Calls `main.py` logic inline through Step 5 (feature engineering), then saves:
- `tests/fixtures/aapl_cache.parquet` -- daily cache after derived_variables
- `tests/fixtures/aapl_income.parquet` -- raw income statement
- `tests/fixtures/aapl_balance.parquet` -- raw balance sheet  
- `tests/fixtures/aapl_cashflow.parquet` -- raw cashflow statement
- `tests/fixtures/aapl_profile.json` -- target profile dict

**Connections exercised:**
- SEC EDGAR wrapper ([`operator1/clients/us_edgar.py`]) -- data source
- Canonical translator ([`operator1/clients/canonical_translator.py`]) -- field mapping
- Cache builder (inline in [`main.py`](main.py:900)) -- OHLCV spine + merge
- [`compute_derived_variables()`](operator1/features/derived_variables.py:1667) -- feature engineering

#### 1.3 Create expected ranges config

**New file:** `tests/fixtures/expected_ranges.yml` (~80 lines)

Pattern: Great Expectations declarative expectation suites.

```yaml
# Expected output ranges for golden AAPL data.
# Update when intentional improvements change scores (e.g., Batch B FH redesign).
# Format: {metric: {min: float, max: float}} or {metric: {not_in: [values]}}

aapl:
  # Layer 1: Feature Engineering
  l1_column_count: {min: 350, max: 600}
  l1_return_1d_abs_max: {min: 0, max: 0.5}
  l1_core_nan_rate_max: {value: 0.20}

  # Layer 2: Analysis
  l2_fh_composite: {min: 40, max: 85}
  l2_fh_label: {not_in: ["Critical", "Poor"]}
  l2_survival_flag_rate: {min: 0.0, max: 0.10}
  l2_survival_prob_latest: {min: 0.70, max: 1.0}

  # Layer 3: Temporal Models
  l3_mc_survival_1d: {min: 0.90, max: 1.0}
  l3_mc_survival_252d: {min: 0.50, max: 1.0}
  l3_mc_drawdown_not_nan: {value: true}
  l3_mc_survival_monotonic: {value: true}
  l3_ohlc_year_pct_change: {min: -0.50, max: 0.50}

  # Layer 4: Hedge Fund
  l4_hf_grade: {not_in: ["F", "D"]}
  l4_hf_conviction: {min: 1, max: 10}

  # Cross-layer: Threshold Consistency
  cross_survival_mc_agree: {value: true}
```

---

### Phase 2: Layer-by-Layer Validation Tests

#### 2.1 Layer 1: Feature Engineering Tests

**New file:** `tests/test_integration_l1_features.py` (~100 lines)

Tests exercise:
- [`compute_derived_variables()`](operator1/features/derived_variables.py:1667) on golden cache
- Column count regression (>350 columns after feature engineering)
- Core column NaN rates (<20% for return_1d, close, current_ratio, gross_margin)
- Return scale validation (return_1d in [-0.5, 0.5], not price-scale)
- No future data leak (filing_freshness <= 1.0 at end)
- Companion flag existence (every ratio has `is_missing_*` or `invalid_math_*`)

**Connections verified:**
- [`safe_ratio()`](operator1/features/derived_variables.py) -- denominator guards
- [`_compute_returns_and_risk()`](operator1/features/derived_variables.py) -- Stage 1 of 24
- [`_compute_microstructure_signals()`](operator1/features/derived_variables.py) -- Stage 18 (PR #1 addition)
- `ta` library integration (ADX, OBV, BB, MACD)

#### 2.2 Layer 2: Analysis Tests

**New file:** `tests/test_integration_l2_analysis.py` (~120 lines)

Tests exercise:
- [`compute_company_survival_flag()`](operator1/analysis/survival_mode.py:71) -- Apple should NOT be in survival >10% of days
- [`compute_survival_probability()`](operator1/analysis/survival_mode.py:373) -- Apple latest prob > 0.70
- [`compute_financial_health()`](operator1/models/financial_health.py:909) -- Apple composite > 40, label not "Critical"/"Poor"
- [`compute_hierarchy_weights()`](operator1/analysis/hierarchy_weights.py:289) -- weights sum to ~100
- Sector-aware thresholds: `sector="Technology"` uses current_ratio threshold 0.7 (from [`scoring_weights.yml`](config/scoring_weights.yml) `sector_overrides.technology.current_ratio`)
- Cash adequacy floor: Apple's $30B+ cash should suppress liquidity survival trigger

**Connections verified:**
- [`_COMPANY_THRESHOLDS`](operator1/analysis/survival_mode.py:68) -- threshold loading from config
- [`_load_company_thresholds()`](operator1/analysis/survival_mode.py) -- scoring_weights.yml consumption
- [`FinancialHealthResult`](operator1/models/financial_health.py:320) -- result dataclass fields
- [`_normalize_series()`](operator1/models/financial_health.py) -- the self-referential scorer (will change in Batch B)

#### 2.3 Layer 3: Temporal Model Tests

**New file:** `tests/test_integration_l3_temporal.py` (~150 lines)

Tests exercise:
- [`run_monte_carlo()`](operator1/models/monte_carlo.py:1264) with `n_paths=1000` (reduced for speed)
  - Drawdown not NaN at any horizon
  - Survival probability monotonically non-increasing (1d >= 5d >= 21d >= 252d)
  - Survival probability > 0.50 at 1d for healthy Apple
- [`predict_ohlc_series()`](operator1/models/ohlc_predictor.py:97) -- year-end within +/-50% of current
- Burn-out distribution scale: if `ExponentialGradientWeightLearner` exists in forward_pass, verify [`get_regime_distributions()`](operator1/models/forecasting.py) returns |mean| < 0.1

**Connections verified:**
- [`DEFAULT_SURVIVAL_THRESHOLDS`](operator1/models/monte_carlo.py:150) -- MC threshold dict
- [`SECTOR_SURVIVAL_OVERRIDES`](operator1/models/monte_carlo.py:164) -- sector override dict
- [`get_sector_aware_thresholds()`](operator1/models/monte_carlo.py) -- threshold merger
- [`MonteCarloResult`](operator1/models/monte_carlo.py:252) -- result dataclass
- [`OHLCCandle`](operator1/models/ohlc_predictor.py:29) -- candle dataclass

#### 2.4 Layer 4: Hedge Fund Tests

**New file:** `tests/test_integration_l4_hedge_fund.py` (~80 lines)

Tests exercise:
- [`run_hedge_fund_analysis()`](operator1/hedge_fund/engine.py:1511) with golden statements + cache
  - Grade not F/D for Apple
  - Conviction > 0
  - FCF quality score > 30 (Apple has strong OCF/NI ratio)
  - Data readiness gate: [`assess_hf_readiness()`](operator1/hedge_fund/helpers.py) passes for Apple
- [`HedgeFundResult`](operator1/hedge_fund/types.py:456) -- available=True
- [`ThesisScorecard`](operator1/hedge_fund/types.py:400) -- all 5 tier scores > 0

**Connections verified:**
- [`FCFQualityResult`](operator1/hedge_fund/types.py:58)
- [`AccrualsForensicResult`](operator1/hedge_fund/types.py:74)
- [`AdvancedMethodsResult`](operator1/hedge_fund/advanced_methods.py:46)

---

### Phase 3: Cross-Layer Consistency Tests

#### 3.1 Threshold Consistency Test

**New file:** `tests/test_integration_threshold_consistency.py` (~80 lines)

This is the **single most important test** -- catches the #1 bug pattern.

Tests exercise:
- Extract thresholds from [`_COMPANY_THRESHOLDS`](operator1/analysis/survival_mode.py:68) (survival_mode source)
- Extract thresholds from [`DEFAULT_SURVIVAL_THRESHOLDS`](operator1/models/monte_carlo.py:150) (MC source)
- Assert they agree on the same trigger variables and directions
- Run both on AAPL cache: if survival_mode says "normal", MC 1d survival must be > 0.50
- If survival_mode says "distress", MC must show < 0.80

**Connections verified:**
- [`scoring_weights.yml`](config/scoring_weights.yml) `survival_thresholds` section
- Both consumers of threshold config read from the same YAML keys
- Sector overrides applied consistently in both paths

#### 3.2 Profile Completeness Test

**New file:** `tests/test_integration_profile_completeness.py` (~60 lines)

Tests exercise:
- Run [`build_company_profile()`](operator1/report/profile_builder.py:1121) with golden data
- Assert all 26 required profile keys exist (from [`profile_schema.py`](operator1/report/profile_schema.py))
- Assert no profile value is `float('nan')` or `float('inf')` (JSON serialization check)
- Run `json.dumps(profile, default=str)` without error

**Connections verified:**
- [`validate_profile()`](operator1/report/profile_schema.py) -- the existing schema checker
- 88 `profile[...]` assignments in [`main.py`](main.py) Step 7

---

### Phase 4: Result Class Serialization Tests

#### 4.1 Auto-Discover and Validate Result Classes

**New file:** `tests/test_integration_result_serialization.py` (~60 lines)

Pattern: Evidently auto-discovery via introspection.

Tests exercise:
- Discover all 90 `*Result` dataclasses in `operator1/`
- For each class with `to_dict()` or `to_profile_dict()`: create minimal instance, call method, verify JSON-serializable
- For classes WITHOUT serialization methods: log warning (57 of 90 lack serialization -- these are internal, but some may need it)

**Connections verified:**
- All 33 `to_dict`/`to_profile_dict` methods across the codebase
- `json.dumps()` compatibility of all result outputs

---

### Phase 5: Graceful Degradation Tests

#### 5.1 Parametrized Fault Injection

**New file:** `tests/test_integration_graceful_degradation.py` (~100 lines)

Pattern: Evidently parametrized edge cases.

Tests exercise (all parametrized):
- FH with missing columns: drop 1-3 columns, assert composite > 0 (not crash)
- MC with no regime labels: cache without `regime_label`, assert survival_probability is not None
- MC with minimal data (50 rows instead of 500): assert no NaN/Inf
- Survival flag with all NaN inputs: assert returns all-zero (not crash)
- HF with empty statement DFs: assert `available=False` (not crash)
- Profile builder with None results: assert profile JSON-serializable

**Connections verified:**
- Every module's error handling path
- The `try/except` guards in [`main.py`](main.py) Steps 4-7
- [`assess_hf_readiness()`](operator1/hedge_fund/helpers.py) -- data gate

---

### Phase 6: Regression Guard Tests

#### 6.1 Column Count and NaN Rate Regression

**New file:** `tests/test_integration_regression_guards.py` (~60 lines)

Pattern: QuantConnect/Lean DataPoints regression.

Tests exercise:
- Cache column count after feature engineering (>350, <600)
- Core columns NaN rate (<20%): return_1d, close, current_ratio, gross_margin, pe_ratio_calc
- Derived variable companion flag existence: for every ratio column, `is_missing_*` or `invalid_math_*` exists
- No duplicate columns in cache
- DatetimeIndex is monotonically increasing

**Connections verified:**
- [`STATEMENT_FIELDS`](operator1/types.py:124) -- expected column names
- [`QUOTE_FIELDS`](operator1/types.py:118) -- expected OHLCV columns
- [`PROFILE_FIELDS`](operator1/types.py:112) -- expected profile fields

---

### Phase 7: CI Integration

#### 7.1 GitHub Actions Workflow

**New file:** `.github/workflows/integration-tests.yml` (~35 lines)

```yaml
name: Integration Tests
on: [pull_request]
jobs:
  integration:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v4
      - uses: jdx/mise-action@v2
      - run: pip install --timeout 300 -r requirements/stage1-core.txt
      - run: pip install --timeout 300 -r requirements/stage2-ml.txt
      - run: pytest tests/test_integration_*.py -v --tb=short -x
```

Note: Stage 3 (torch/pymc) and Stage 4 (wrappers) NOT installed for CI -- integration tests are designed to work with Stage 1+2 only. MC and HF tests that require Stage 3 models use `pytest.importorskip("torch")`.

---

## File Summary

| File | Type | Lines | Patterns Used |
|------|------|-------|---------------|
| `tests/conftest.py` | New | ~60 | Deepchecks session fixtures, GE validator |
| `tests/generate_golden_data.py` | New | ~40 | NASA golden test data |
| `tests/fixtures/expected_ranges.yml` | New | ~80 | GE declarative expectations |
| `tests/test_integration_l1_features.py` | New | ~100 | Pandera schema, Lean DataPoints |
| `tests/test_integration_l2_analysis.py` | New | ~120 | Deepchecks check conditions |
| `tests/test_integration_l3_temporal.py` | New | ~150 | Lean AlgorithmStatisticsTest |
| `tests/test_integration_l4_hedge_fund.py` | New | ~80 | Deepchecks model validation |
| `tests/test_integration_threshold_consistency.py` | New | ~80 | Lean regression, custom |
| `tests/test_integration_profile_completeness.py` | New | ~60 | Syrupy structure snapshot |
| `tests/test_integration_result_serialization.py` | New | ~60 | Evidently auto-discovery |
| `tests/test_integration_graceful_degradation.py` | New | ~100 | Evidently parametrize, Netflix fault injection |
| `tests/test_integration_regression_guards.py` | New | ~60 | Lean DataPoints regression |
| `.github/workflows/integration-tests.yml` | New | ~35 | CI gate |
| `tests/fixtures/aapl_cache.parquet` | Binary | ~2MB | Golden data |
| `tests/fixtures/aapl_*.parquet` | Binary x3 | ~1MB | Golden statements |
| `tests/fixtures/aapl_profile.json` | JSON | ~200 | Golden profile |
| **Total** | **16 files** | **~1,025 lines + fixtures** | |

---

## Execution Order

1. **Phase 1** first (fixtures + conftest): everything depends on golden data
2. **Phase 2** next (L1-L4 tests): validates each layer independently
3. **Phase 3** then (cross-layer): catches threshold disagreements
4. **Phase 4** (serialization): catches profile wiring gaps
5. **Phase 5** (degradation): catches crash bugs on edge cases
6. **Phase 6** (regression guards): catches silent data loss
7. **Phase 7** last (CI): gates all PRs

---

## Validation Criteria

After implementation, ALL of these must pass:

1. `pytest tests/test_integration_*.py -v` passes with 0 failures
2. Apple FH composite > 40 (currently 28.97 -- will fail until Batch B fixes scoring)
3. Apple survival probability latest > 0.70
4. MC drawdown never NaN at any horizon
5. MC survival monotonically non-increasing
6. OHLC year-end within +/-50% of current price
7. HF grade not F/D for Apple
8. survival_mode and MC agree on Apple's survival status
9. All 33 `to_dict`/`to_profile_dict` methods produce JSON-serializable output
10. Profile `json.dumps()` succeeds without error
11. Cache has >350 columns after feature engineering
12. Core column NaN rate < 20%

**Note:** Criterion #2 will fail on current code (FH=28.97). This is intentional -- the test documents the bug. After Batch B (cross-sectional scoring) merges, the test will pass. Tests should be committed with `@pytest.mark.xfail(reason="Batch B not yet merged")` for known-broken metrics.
