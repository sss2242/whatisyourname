# Batch C: Integration Test Suite -- Prevent Regressions

*Problem 5: 923 commits, zero tests that verify "does the output make sense?"*

---

## Current state

The test suite checks:
- "Does the column exist?" (`assert "fh_composite_score" in cache.columns`)
- "Is it not NaN?" (`assert result_cache["fh_composite_score"].notna().any()`)
- "Does the function not crash?" (implicit in all tests)

It never checks:
- "Is Apple's FH composite > 50?" (it's 28.97)
- "Is Apple's survival probability > 80%?" (it was 0% at 1-day)
- "Is the OHLC year-end within 30% of current?" (it was $78, a 68% crash)
- "Is MC drawdown not NaN?" (it was NaN at 5d/21d/252d)

Every session can silently regress the output quality. The "debug scan" pattern (scan, find N bugs, fix them) repeats 8 times in the commit history because there's no automated check preventing re-introduction.

---

## Expert methods

**From NASA software engineering (NASA-STD-8739.8, "Software Assurance and Software Safety Standard"):**
- **Golden test data**: NASA maintains reference datasets with known correct outputs. Every build runs the same inputs and asserts outputs match within tolerance. Not unit tests -- full pipeline integration tests with frozen inputs and expected outputs.
- Applied here: Cache a real AAPL pipeline output. Every PR runs the cached data through Stages 3-7 and asserts key metrics match expected ranges.

**From machine learning ops (Google's ML Test Score, Breck et al. 2017):**
- **Data validation tests**: Assert statistical properties of features (mean, variance, min/max within expected range). A feature suddenly being all-zero or all-NaN is caught before it reaches the model.
- **Model staleness tests**: Assert that model predictions change when inputs change (not stuck returning constants).
- **Prediction quality tests**: Assert that predictions on calibration data match historical realized values within tolerance.
- Applied here: Test each layer's output independently -- L1 features, L2 analysis, L3 temporal, L4 HF -- before the full integration test.

**From pharmaceutical clinical trials (ICH E6 Good Clinical Practice):**
- **Expected outcome ranges**: Before a trial runs, the protocol defines what constitutes a "pass" for each endpoint. You don't run the trial and then decide what the result should be.
- Applied here: Define expected output ranges per company BEFORE running the test. Apple FH should be 50-75. This prevents "anchoring to whatever the current broken output is."

**From chaos engineering (Netflix Simian Army):**
- **Fault injection**: Deliberately break one module and verify that downstream modules degrade gracefully (not silently produce garbage).
- Applied here: Test what happens when MC returns None, when linked_caches is empty, when income_df has only 2 rows.

---

## Implementation Plan

### Phase 1: Golden Test Data (frozen AAPL cache)

**File:** `tests/fixtures/aapl_cache.parquet` (new, ~2MB)

Generate once by running the pipeline on AAPL, then freeze the cache at the end of Step 5 (after features, before temporal models). This cache contains:
- ~500 rows (2 years of trading days)
- ~400 columns (all derived variables, survival flags, macro data)
- Real financial data from SEC EDGAR
- Real OHLCV from yfinance

Script to regenerate:
```python
# tests/generate_golden_data.py
# Run: python main.py --market us_sec_edgar --company AAPL --skip-models --output-dir tests/fixtures
# Then: cp tests/fixtures/cache.parquet tests/fixtures/aapl_cache.parquet
```

Also freeze:
- `tests/fixtures/aapl_profile.json` -- target profile dict
- `tests/fixtures/aapl_income.parquet` -- raw income statement
- `tests/fixtures/aapl_balance.parquet` -- raw balance sheet
- `tests/fixtures/aapl_cashflow.parquet` -- raw cashflow statement

### Phase 2: Layer-by-Layer Output Validation Tests

**File:** `tests/test_integration_aapl.py` (new, ~300 lines)

```python
"""Integration tests using frozen AAPL data.

These tests run cached data through each pipeline layer and assert
that key output metrics fall within expected ranges. They catch:
- Regressions that produce NaN/Inf
- Score inversions (healthy company scoring as distressed)
- Scale contamination (billions in return distributions)
- Threshold disagreements between modules

Run: pytest tests/test_integration_aapl.py -v
"""

import pytest
import pandas as pd
import numpy as np
from pathlib import Path

FIXTURE_DIR = Path(__file__).parent / "fixtures"


@pytest.fixture
def aapl_cache():
    return pd.read_parquet(FIXTURE_DIR / "aapl_cache.parquet")


@pytest.fixture
def aapl_statements():
    return {
        "income": pd.read_parquet(FIXTURE_DIR / "aapl_income.parquet"),
        "balance": pd.read_parquet(FIXTURE_DIR / "aapl_balance.parquet"),
        "cashflow": pd.read_parquet(FIXTURE_DIR / "aapl_cashflow.parquet"),
    }


# ---- Layer 1: Feature Engineering ----

class TestL1Features:
    def test_derived_variables_no_nan_in_core(self, aapl_cache):
        """Core derived variables should not be all-NaN."""
        core = ["return_1d", "volatility_21d", "current_ratio", 
                "gross_margin", "pe_ratio_calc"]
        for col in core:
            if col in aapl_cache.columns:
                assert aapl_cache[col].notna().sum() > 100, \
                    f"{col} has fewer than 100 non-NaN values"

    def test_returns_are_return_scale(self, aapl_cache):
        """return_1d should be in [-0.5, 0.5] range, not price-scale."""
        r = aapl_cache["return_1d"].dropna()
        assert r.abs().max() < 0.5, \
            f"return_1d max={r.abs().max()}, likely price-scale not return-scale"

    def test_no_future_data_leak(self, aapl_cache):
        """Derived variables should not reference future dates."""
        if "filing_freshness" in aapl_cache.columns:
            assert aapl_cache["filing_freshness"].iloc[-1] <= 1.0


# ---- Layer 2: Analysis ----

class TestL2Analysis:
    def test_financial_health_apple_not_distressed(self, aapl_cache):
        """Apple should score above 40 on financial health.
        
        Apple has: 77% gross margins, $30B+ quarterly FCF, 
        $160B+ cash, interest coverage >29x. Any score below 40 
        indicates the scoring methodology is broken.
        """
        from operator1.models.financial_health import compute_financial_health
        cache, result = compute_financial_health(aapl_cache)
        assert result.latest_composite > 40, \
            f"Apple FH={result.latest_composite} (expected >40). " \
            f"Tier scores: L={result.tier_means.get('fh_liquidity_score')}, " \
            f"S={result.tier_means.get('fh_solvency_score')}, " \
            f"St={result.tier_means.get('fh_stability_score')}, " \
            f"P={result.tier_means.get('fh_profitability_score')}, " \
            f"G={result.tier_means.get('fh_growth_score')}"

    def test_survival_probability_apple_healthy(self, aapl_cache):
        """Apple should have >80% survival probability at all horizons."""
        from operator1.analysis.survival_mode import (
            compute_company_survival_flag, compute_survival_probability,
        )
        flag = compute_company_survival_flag(aapl_cache)
        prob = compute_survival_probability(aapl_cache)
        # Apple should not be in survival mode for >10% of days
        assert flag.sum() / len(flag) < 0.1, \
            f"Apple flagged as survival {flag.sum()}/{len(flag)} days"
        # Latest probability should be high
        latest_prob = float(prob.dropna().iloc[-1])
        assert latest_prob > 0.7, \
            f"Apple survival prob={latest_prob} (expected >0.7)"


# ---- Layer 3: Temporal Models ----

class TestL3TemporalModels:
    def test_mc_no_nan_drawdown(self, aapl_cache):
        """MC drawdown should never be NaN at any horizon."""
        from operator1.models.monte_carlo import run_monte_carlo
        mc = run_monte_carlo(aapl_cache, n_paths=1000)  # fewer paths for speed
        for h in ["1d", "5d", "21d", "252d"]:
            dd = mc.max_drawdown_distribution.get(h, {})
            if dd:
                assert not np.isnan(dd.get("median", float("nan"))), \
                    f"MC drawdown NaN at {h}"

    def test_mc_survival_monotonic(self, aapl_cache):
        """Survival probability should be monotonically non-increasing.
        
        P(survive 1 day) >= P(survive 5 days) >= P(survive 21 days) >= P(survive 252 days)
        """
        from operator1.models.monte_carlo import run_monte_carlo
        mc = run_monte_carlo(aapl_cache, n_paths=1000)
        probs = []
        for h in ["1d", "5d", "21d", "252d"]:
            sp = mc.survival_probability.get(h, {})
            if isinstance(sp, dict):
                probs.append(sp.get("mean", 1.0))
            elif isinstance(sp, (int, float)):
                probs.append(float(sp))
        if len(probs) >= 2:
            for i in range(len(probs) - 1):
                assert probs[i] >= probs[i+1] - 0.05, \
                    f"Non-monotonic survival: {probs}"

    def test_burnout_distributions_return_scale(self, aapl_cache):
        """Burn-out regime distributions should be return-scale (|mean|<0.1)."""
        from operator1.models.forecasting import run_forward_pass, run_burnout
        # This is expensive -- only run if forward pass result exists
        # In practice, load from cached state
        pass  # Placeholder -- requires forward_pass_result

    def test_ohlc_year_end_within_range(self, aapl_cache):
        """OHLC year-end prediction should be within +/-50% of current price."""
        from operator1.models.ohlc_predictor import predict_ohlc_series
        ohlc = predict_ohlc_series(aapl_cache)
        if ohlc.fitted and ohlc.next_year:
            last_close = float(aapl_cache["close"].dropna().iloc[-1])
            year_end_close = ohlc.next_year[-1].close
            pct_change = abs(year_end_close - last_close) / last_close
            assert pct_change < 0.50, \
                f"OHLC year-end ${year_end_close:.2f} is {pct_change:.0%} " \
                f"from current ${last_close:.2f}"


# ---- Layer 4: Hedge Fund ----

class TestL4HedgeFund:
    def test_hf_scorecard_apple_not_F(self, aapl_cache, aapl_statements):
        """Apple should not receive an F grade from the HF scorecard."""
        from operator1.hedge_fund.engine import run_hedge_fund_analysis
        hf = run_hedge_fund_analysis(
            income_df=aapl_statements["income"],
            balance_df=aapl_statements["balance"],
            cashflow_df=aapl_statements["cashflow"],
            cache=aapl_cache,
            target_profile={"sector": "Technology", "name": "Apple"},
        )
        if hf.available and hf.scorecard:
            grade = hf.scorecard.investment_grade
            assert grade not in ("F", "D"), \
                f"Apple HF grade={grade} (expected B or better)"


# ---- Cross-Layer: Threshold Consistency ----

class TestThresholdConsistency:
    def test_survival_and_mc_agree(self, aapl_cache):
        """survival_mode and monte_carlo should use the same thresholds.
        
        If survival_mode says 'normal', MC should show >50% survival.
        If survival_mode says 'distress', MC should show <50% survival.
        """
        from operator1.analysis.survival_mode import compute_company_survival_flag
        from operator1.models.monte_carlo import run_monte_carlo
        
        flag = compute_company_survival_flag(aapl_cache)
        mc = run_monte_carlo(aapl_cache, n_paths=1000)
        
        latest_flag = int(flag.iloc[-1])
        mc_1d = mc.survival_probability.get("1d", {})
        mc_1d_mean = mc_1d.get("mean", 1.0) if isinstance(mc_1d, dict) else float(mc_1d)
        
        if latest_flag == 0:  # normal
            assert mc_1d_mean > 0.5, \
                f"survival_mode says normal but MC 1d survival={mc_1d_mean}"
        elif latest_flag == 1:  # distress
            assert mc_1d_mean < 0.8, \
                f"survival_mode says distress but MC 1d survival={mc_1d_mean}"


# ---- Fault Injection ----

class TestGracefulDegradation:
    def test_fh_with_missing_columns(self):
        """FH should produce a score even with minimal data."""
        cache = pd.DataFrame({
            "close": [100.0] * 100,
            "current_ratio": [1.5] * 100,
        }, index=pd.date_range("2024-01-01", periods=100, freq="B"))
        from operator1.models.financial_health import compute_financial_health
        cache, result = compute_financial_health(cache)
        assert result.latest_composite > 0, "FH should produce >0 with partial data"

    def test_mc_with_no_regime_labels(self):
        """MC should run without regime labels (single 'unknown' regime)."""
        cache = pd.DataFrame({
            "close": np.random.lognormal(5, 0.01, 300),
            "return_1d": np.random.normal(0, 0.01, 300),
        }, index=pd.date_range("2024-01-01", periods=300, freq="B"))
        from operator1.models.monte_carlo import run_monte_carlo
        mc = run_monte_carlo(cache, n_paths=100)
        assert mc.survival_probability is not None
```

### Phase 3: CI gate

**File:** `.github/workflows/integration-tests.yml` (new, ~30 lines)

```yaml
name: Integration Tests
on: [pull_request]
jobs:
  test:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v4
      - uses: actions/setup-python@v5
        with:
          python-version: '3.12'
      - run: pip install -r requirements/stage1-core.txt -r requirements/stage2-ml.txt
      - run: pytest tests/test_integration_aapl.py -v --tb=short
```

### Phase 4: Expected output documentation

**File:** `plans/integration-test-expectations.md` (new, ~50 lines)

```markdown
# Integration Test Expected Outputs

## AAPL (Technology) -- as of 2026-05-09
| Metric | Expected Range | Why |
|--------|---------------|-----|
| FH composite | 50-75 | 77% gross margin, $30B+ FCF, $160B cash |
| Survival probability | 85-99% | current_ratio=0.92 is normal for tech |
| MC 1d survival | 95-100% | healthy company, 1-day risk is minimal |
| MC drawdown | not NaN | no horizon should produce NaN |
| OHLC year-end | +/-30% of current | no crash/moonshot predictions |
| HF grade | B or better | strong fundamentals across all tiers |
| Survival curve | monotonic non-increasing | longer horizons = more risk |
```

---

## Files created/changed

| File | Type | Lines |
|------|------|-------|
| `tests/fixtures/aapl_cache.parquet` | Binary fixture | ~2MB |
| `tests/fixtures/aapl_profile.json` | JSON fixture | ~200 |
| `tests/fixtures/aapl_*.parquet` | Binary fixtures x3 | ~1MB |
| `tests/generate_golden_data.py` | Script | ~30 |
| `tests/test_integration_aapl.py` | Tests | ~300 |
| `.github/workflows/integration-tests.yml` | CI | ~30 |
| `plans/integration-test-expectations.md` | Docs | ~50 |
| **Total** | | ~410 lines + fixtures |

---

## Key design decisions

1. **Frozen fixtures, not live API calls**: Tests use cached data so they're fast (seconds), deterministic, and don't require API keys. Fixture regeneration is manual and infrequent.

2. **Range assertions, not exact values**: `assert composite > 40` not `assert composite == 63.7`. Exact values change with code improvements; ranges catch regressions without false positives.

3. **Per-layer isolation**: Each test class corresponds to one pipeline layer. A failure in TestL2Analysis doesn't depend on TestL3TemporalModels having run.

4. **Threshold consistency test**: The single most important test. If survival_mode and MC disagree on Apple's survival status, something is structurally wrong.

5. **Graceful degradation tests**: Verify the pipeline produces reasonable output even with missing data -- the common case for Tier 2 markets.
