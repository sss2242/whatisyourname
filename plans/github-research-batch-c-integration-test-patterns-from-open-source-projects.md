# GitHub Research: Integration Test Patterns from Open-Source Projects

*Research for Batch C: Integration Test Suite (Problem 5)*
*Date: 2026-05-09*

Research across 7 open-source projects to identify implementation patterns for golden data fixtures, range-based validation, layer isolation, fault injection, snapshot testing, and financial pipeline regression testing. Each pattern evaluated for applicability to Operator 1's specific needs: a 78-module financial analysis pipeline with ~516 cache columns and ~346 result fields.

---

## Project 1: Great Expectations (11,467 stars)

**Repo:** `great-expectations/great_expectations`
**Relevance:** The standard for data validation in production pipelines. Their "expectation" concept maps directly to our "assert output makes sense" need.

### Key Patterns Found

**Pattern 1: Expectation Suites as Declarative Contracts**

Great Expectations defines data quality checks as JSON-serializable "expectation suites" -- not inline code assertions. Each expectation is a named object with parameters:

```python
# GE approach: declarative expectations stored as JSON
{
    "expectation_type": "expect_column_values_to_be_between",
    "kwargs": {"column": "revenue", "min_value": 0, "max_value": 1e12}
}
```

**What we can adopt:** Instead of hardcoding `assert result.latest_composite > 40` in test files, define expected ranges in a YAML config file (`tests/fixtures/expected_ranges.yml`). This separates "what to check" from "how to check," making it easy to add new companies or update ranges without touching test code.

```yaml
# Proposed: expected_ranges.yml
aapl:
  fh_composite: {min: 40, max: 85}
  survival_probability: {min: 0.70, max: 1.0}
  mc_survival_1d: {min: 0.90, max: 1.0}
  hf_grade: {not_in: ["F", "D"]}
```

**Pattern 2: Fixture-Based Test Data with Session Scope**

GE's `conftest.py` uses `@pytest.fixture(scope='session')` for expensive data loads. The fixture is computed once and reused across all tests in the session. They also use `@pytest.fixture(scope='function')` for test-specific state.

**What we can adopt:** Load the AAPL parquet fixture once per session (not per test), since reading a 2MB parquet takes ~100ms but running FH scoring takes ~2s. Session-scoped fixtures prevent redundant I/O.

**Pattern 3: Validator-Based Architecture**

GE separates the validation engine from the data source. Validators receive a batch of data and apply expectations. Results are structured objects with `success`, `result`, and `exception_info` fields.

**What we can adopt:** Create a `PipelineValidator` class that receives a cache DataFrame and returns structured validation results. This is cleaner than 30 independent test functions and allows the dashboard to call the same validator for runtime checks.

---

## Project 2: Deepchecks (4,012 stars)

**Repo:** `deepchecks/deepchecks`
**Relevance:** Purpose-built for ML model and data validation. Their tabular checks are the closest analogy to our per-layer output validation.

### Key Patterns Found

**Pattern 4: Train/Test Split Fixture with Model Fitting**

Deepchecks' `conftest.py` uses session-scoped fixtures that load real datasets (iris, diabetes), split them, fit a model, and make the fitted model available to all tests. This avoids re-fitting models in each test:

```python
@pytest.fixture(scope='session')
def diabetes_model(diabetes):
    clf = GradientBoostingRegressor(random_state=0)
    train, _ = diabetes
    clf.fit(train.data[train.features], train.data[train.label_name])
    return clf
```

**What we can adopt:** For Layer 3 temporal model tests, create a session-scoped fixture that runs forecasting on the golden AAPL cache once, then reuses the `ForecastResult` across all downstream tests (MC, conformal, prediction aggregator). This converts a 5-minute test suite into a 30-second one.

```python
@pytest.fixture(scope='session')
def aapl_forecast(aapl_cache):
    from operator1.models.forecasting import run_forecasting
    _, result = run_forecasting(aapl_cache, extra_variables=[...])
    return result
```

**Pattern 5: hamcrest Matchers for Expressive Assertions**

Deepchecks uses PyHamcrest matchers (`close_to`, `greater_than`, `has_entries`) instead of raw `assert`. This produces much more readable test output on failure:

```python
assert_that(result['scores']['f1_score']['Origin'], close_to(0.94, 0.05))
```

**What we can adopt:** Use `pytest.approx()` (built-in, no extra dependency) for numerical range checks. For structured dict checks, use helper functions that produce informative failure messages.

**Pattern 6: Check-Based Architecture with Conditions**

Each Deepchecks check is a class with a `run()` method that returns a `CheckResult`. Conditions are added post-hoc: `check.add_condition_greater_than("metric", 0.5)`. This separates computation from assertion.

**What we can adopt:** Structure tests as "compute result, then assert conditions" rather than mixing computation with assertions. This makes it trivial to add new assertions without re-running expensive computations.

---

## Project 3: Pandera (4,329 stars)

**Repo:** `unionai-oss/pandera`
**Relevance:** DataFrame schema validation. Directly applicable to validating our ~516-column cache DataFrame.

### Key Patterns Found

**Pattern 7: Schema Definitions for DataFrames**

Pandera defines expected DataFrame schemas as Python classes with type annotations and constraints:

```python
class CacheSchema(pa.DataFrameModel):
    close: pa.typing.Series[float] = pa.Field(gt=0, nullable=True)
    return_1d: pa.typing.Series[float] = pa.Field(ge=-0.5, le=0.5, nullable=True)
    current_ratio: pa.typing.Series[float] = pa.Field(ge=0, le=50, nullable=True)
    fh_composite_score: pa.typing.Series[float] = pa.Field(ge=0, le=100, nullable=True)
```

**What we can adopt:** Define a `CacheSchema` for our daily cache that validates:
- Column existence (all 96 derived variables present)
- Value ranges (return_1d between -0.5 and 0.5, not price-scale)
- Type correctness (no string values in numeric columns)
- NaN ratio bounds (return_1d should have <5% NaN for AAPL)

This catches the exact bugs we've seen: return values in price-scale ($30B in return distributions), columns silently all-NaN, and type mismatches.

**Pattern 8: Hypothesis-Based Property Testing**

Pandera integrates with Hypothesis for property-based testing -- generating random DataFrames that match a schema and verifying that pipeline functions handle them correctly.

**What we can adopt (lower priority):** For fault injection tests, use Hypothesis to generate degenerate DataFrames (all-NaN columns, single-row DataFrames, extreme values) and verify graceful degradation. This is more thorough than manually constructing edge cases.

---

## Project 4: Evidently (7,466 stars)

**Repo:** `evidentlyai/evidently`
**Relevance:** ML monitoring and data drift detection. Their parametrized test patterns are directly applicable.

### Key Patterns Found

**Pattern 9: Parametrized Edge Case Testing**

Evidently's `test_data_integration.py` uses `@pytest.mark.parametrize` extensively to test the same function against many edge cases in a compact format:

```python
@pytest.mark.parametrize(
    "dataset, expected_missed",
    (
        (pd.DataFrame(), 0),
        (pd.DataFrame({"feature": []}), 0),
        (pd.DataFrame({"feature": [1, 2, 3]}), 0),
        (pd.DataFrame({"feature1": [1, None, pd.NA], "feature2": [np.nan, None, pd.NaT]}), 5),
    ),
)
def test_get_number_of_all_pandas_missed_values(dataset, expected_missed):
    assert get_number_of_all_pandas_missed_values(dataset) == expected_missed
```

**What we can adopt:** For fault injection tests, use parametrized test cases instead of separate test functions. One parametrized test for "FH with partial data" covers 5 edge cases in 10 lines:

```python
@pytest.mark.parametrize("missing_cols,expected_min", [
    ([], 40),                      # full data
    (["revenue"], 20),             # missing profitability input
    (["close", "volume"], 15),     # no OHLCV
    (["current_ratio"], 25),       # missing liquidity
    (["total_assets"], 10),        # missing solvency
])
def test_fh_with_missing_data(aapl_cache, missing_cols, expected_min):
    cache = aapl_cache.drop(columns=missing_cols, errors="ignore")
    _, result = compute_financial_health(cache)
    assert result.latest_composite > expected_min
```

**Pattern 10: MetricResult Schema Validation via Introspection**

Evidently's `test_metric_results.py` uses Python introspection to discover ALL MetricResult subclasses in the codebase, then validates their serialization config. This catches configuration errors automatically as new metrics are added.

**What we can adopt:** Write a test that discovers all `*Result` dataclasses in `operator1/` (via `importlib` + `inspect`) and validates that each has `to_dict()` or `to_profile_dict()` methods, and that calling them doesn't raise on the golden data. This prevents the "new model result not wired into profile" bug pattern we've seen 5+ times.

```python
def test_all_result_classes_serializable():
    """Every *Result dataclass should serialize without error."""
    for cls in discover_result_classes():
        instance = create_minimal_instance(cls)
        d = instance.to_dict() if hasattr(instance, 'to_dict') else vars(instance)
        json.dumps(d, default=str)  # Must be JSON-serializable
```

---

## Project 5: whylogs (2,817 stars)

**Repo:** `whylabs/whylogs`
**Relevance:** Data profiling and constraint-based validation. Their constraint factory pattern is elegant for our per-column validation needs.

### Key Patterns Found

**Pattern 11: Constraint Factories with Named Constraints**

whylogs defines constraints as composable factories with human-readable names:

```python
builder.add_constraint(is_in_range(column_name="weight", lower=1.1, upper=3.2))
builder.add_constraint(is_non_negative(column_name="legs"))
builder.add_constraint(mean_between_range(column_name="age", lower=20, upper=80))
builder.add_constraint(quantile_between_range(column_name="score", quantile=0.95, lower=0, upper=100))
```

Each constraint produces a report: `{name: "weight is in range [1.1,3.2]", passed: 0/1, failed: 1/0}`.

**What we can adopt:** Build a constraint-based validation layer for the cache that produces a structured report. Instead of pytest assertions (which stop at first failure), constraints run exhaustively and produce a full report:

```python
# Proposed: cache_constraints.py
constraints = [
    column_in_range("return_1d", -0.5, 0.5),
    column_in_range("fh_composite_score", 0, 100),
    column_non_negative("survival_probability"),
    column_not_all_nan("close"),
    column_not_all_nan("current_ratio"),
    mean_in_range("return_1d", -0.01, 0.01),
    quantile_in_range("volatility_21d", 0.95, 0, 0.5),
]

report = validate_cache(aapl_cache, constraints)
# report: [{name: "return_1d in [-0.5, 0.5]", passed: True}, ...]
```

**Pattern 12: Skip Missing Columns Gracefully**

whylogs constraints accept a `skip_missing=True` parameter that returns "passed" when a column doesn't exist. This prevents tests from failing on Tier 2 markets where some columns (options signals, cross-asset signals) may not be available.

**What we can adopt:** Add `required=True/False` to cache constraints. Required columns (return_1d, close) must exist and pass. Optional columns (options signals, cross-asset) pass automatically when absent.

---

## Project 6: Syrupy (845 stars)

**Repo:** `syrupy-project/syrupy`
**Relevance:** Pytest snapshot testing plugin. Instead of manually specifying expected values, tests compare against stored snapshots.

### Key Patterns Found

**Pattern 13: Snapshot Testing for Complex Outputs**

Syrupy stores the first test output as a "snapshot" file. Subsequent runs compare against the snapshot. If the output changes, the test fails until the developer explicitly updates the snapshot with `--snapshot-update`.

```python
def test_profile_structure(snapshot):
    profile = build_company_profile(aapl_data)
    assert profile == snapshot  # compares against stored JSON snapshot
```

**What we can adopt (with caution):** Snapshot testing is powerful for catching unintended changes to profile structure (new keys, removed keys, type changes). However, for numerical values that legitimately change with code improvements, snapshots are too brittle. 

**Recommended hybrid:** Use snapshots for profile STRUCTURE (key names, nesting, types) but range assertions for VALUES. This catches "profile key accidentally removed" without false-positiving on "FH score improved from 55 to 62."

```python
def test_profile_structure(snapshot):
    profile = build_company_profile(aapl_data)
    # Snapshot only the keys and types, not values
    structure = extract_structure(profile)  # {key: type_name, nested...}
    assert structure == snapshot

def test_profile_values(aapl_cache):
    # Range assertions for values (separate from structure test)
    assert profile["financial_health"]["latest_composite"] > 40
```

**Pattern 14: Diff-Based Snapshots**

Syrupy supports `diff` mode where a snapshot stores only the delta from a base snapshot. Useful when testing incremental changes.

**What we can adopt:** For regression testing across pipeline versions, store the baseline AAPL profile as a snapshot. When a PR changes model behavior, the diff shows exactly which profile sections changed and by how much.

---

## Project 7: QuantConnect/Lean (18,878 stars)

**Repo:** `QuantConnect/Lean`
**Relevance:** Production-grade algorithmic trading engine with the most mature regression testing framework in finance.

### Key Patterns Found

**Pattern 15: AlgorithmStatisticsTestParameters -- the Gold Standard**

Lean's regression test framework is the most directly applicable pattern. Each algorithm has a test that:
1. Defines expected statistics as a dictionary
2. Runs the full backtest
3. Compares actual statistics against expected

```csharp
var parameters = new AlgorithmStatisticsTestParameters(
    "SelectUniverseSymbolsFromIDRegressionAlgorithm",
    new Dictionary<string, string> { 
        {PerformanceMetrics.TotalOrders, "0"} 
    },
    Language.Python,
    AlgorithmStatus.Completed
);
AlgorithmRunner.RunLocalBacktest(parameters.Algorithm,
    parameters.Statistics, parameters.Language, parameters.ExpectedFinalStatus);
```

**What we can adopt:** This is exactly what Batch C needs. Define per-company expected statistics, run the pipeline, compare. The key Lean design decisions:
- Expected values are **exact** for deterministic metrics (TotalOrders) and **ranges** for stochastic metrics
- Tests verify the **final status** (Completed vs Error) separately from statistics
- Non-deterministic data points use `-1` as a "skip" sentinel
- Each regression algorithm is a standalone test case

**Proposed implementation for Operator 1:**

```python
@dataclass
class PipelineRegressionParams:
    company: str
    market_id: str
    expected: dict[str, tuple[float, float]]  # metric -> (min, max)
    expected_status: str = "completed"
    
AAPL_REGRESSION = PipelineRegressionParams(
    company="AAPL",
    market_id="us_sec_edgar",
    expected={
        "fh_composite": (40, 85),
        "survival_probability": (0.70, 1.0),
        "mc_survival_1d": (0.90, 1.0),
        "mc_drawdown_252d_not_nan": (1, 1),  # boolean as (1,1)
        "hf_grade_not_F_D": (1, 1),
        "ohlc_year_pct_change": (-0.50, 0.50),
    },
)
```

**Pattern 16: DataPoints Regression Guard**

Lean tracks `DataPoints` (total data consumed) and `AlgorithmHistoryDataPoints` as regression metrics. If a code change accidentally reduces data consumption (e.g., a filter bug), the test catches it.

**What we can adopt:** Track cache column count and non-NaN ratios as regression metrics. If a code change accidentally drops columns (from ~516 to ~500), or increases NaN rate (from 5% to 30%), the test catches it.

```python
def test_cache_column_count_regression(aapl_cache):
    """Cache should have ~400+ columns after feature engineering."""
    n_cols = len(aapl_cache.columns)
    assert n_cols > 350, f"Column count regressed: {n_cols} (expected >350)"

def test_cache_nan_rate_regression(aapl_cache):
    """Core columns should have <20% NaN."""
    for col in CORE_COLUMNS:
        if col in aapl_cache.columns:
            nan_rate = aapl_cache[col].isna().mean()
            assert nan_rate < 0.20, f"{col} NaN rate: {nan_rate:.1%}"
```

---

## Synthesis: Recommended Implementation for Operator 1

Based on the 16 patterns discovered, here's the recommended test architecture:

### Architecture

```
tests/
  fixtures/
    aapl_cache.parquet          # Frozen golden cache (Pattern 1, 15)
    aapl_statements/            # Raw statement DFs
    expected_ranges.yml         # Declarative range specs (Pattern 1)
  conftest.py                   # Session-scoped fixtures (Pattern 2, 4)
  test_integration_aapl.py      # Per-layer validation (Pattern 6, 9)
  test_cache_schema.py          # DataFrame schema validation (Pattern 7)
  test_cache_constraints.py     # Constraint-based validation (Pattern 11)
  test_profile_structure.py     # Snapshot structure test (Pattern 13)
  test_regression_metrics.py    # Column count + NaN regression (Pattern 16)
  test_graceful_degradation.py  # Parametrized fault injection (Pattern 9)
  test_result_serialization.py  # Auto-discover result classes (Pattern 10)
  test_threshold_consistency.py # Cross-module agreement (Pattern 15)
```

### Priority Ranking

| Priority | Pattern | Source | Why |
|----------|---------|--------|-----|
| P1 | Golden fixture + session scope | GE, Deepchecks | Foundation for all other tests |
| P1 | Range assertions in YAML | GE | Separates spec from code |
| P1 | AlgorithmStatisticsTestParameters | Lean | Exact regression framework we need |
| P1 | Threshold consistency | Lean | Catches the #1 bug pattern (threshold disagreement) |
| P2 | DataFrame schema validation | Pandera | Catches scale contamination (return vs price) |
| P2 | Constraint-based validation | whylogs | Exhaustive reporting (not stop-at-first-failure) |
| P2 | Auto-discover result classes | Evidently | Catches "new model not wired into profile" |
| P2 | Column count + NaN regression | Lean | Catches silent data loss |
| P3 | Parametrized fault injection | Evidently | Covers edge cases compactly |
| P3 | Snapshot structure testing | Syrupy | Catches profile key removal/rename |
| P4 | Hypothesis property testing | Pandera | Thorough but slower |
| P4 | Diff-based snapshots | Syrupy | For cross-version regression tracking |

### Key Design Decisions Informed by Research

1. **Session-scoped fixtures over per-test computation** (Deepchecks pattern): The golden AAPL cache should be loaded once. Expensive computations (FH scoring, forecasting) should run once in session fixtures and be reused across all assertion tests. This is the difference between a 5-minute and 30-second test suite.

2. **Declarative ranges in YAML over inline assertions** (GE pattern): Expected ranges should live in `expected_ranges.yml`, not scattered across test functions. When Apple's FH improves from 55 to 65 due to the Batch B cross-sectional scoring fix, update one YAML entry instead of 5 test files.

3. **Exhaustive constraint reports over fail-fast assertions** (whylogs pattern): Run ALL checks and produce a report, not stop at the first failure. When debugging, knowing "5 of 12 checks failed" is more useful than "first check failed, 11 unknown."

4. **Structure snapshots + value ranges** (Syrupy + GE hybrid): Test profile structure (keys, types, nesting) with snapshots that break on unintended changes. Test profile values (scores, probabilities) with ranges that accommodate intentional improvements.

5. **Auto-discovery for coverage** (Evidently pattern): Don't manually list every Result class to test. Discover them via introspection. When someone adds a new model, the test automatically checks it serializes correctly.

6. **skip_missing for optional features** (whylogs pattern): Tier 2 markets may lack options signals, cross-asset signals, or peer data. Tests should pass with `skip_missing=True` for non-core columns, failing only on columns that every market must have.

### New Dependencies

None required. All patterns use existing pytest features (`fixtures`, `parametrize`, `approx`). Pandera and Syrupy are optional enhancements but not needed for the core test suite.

### Estimated Implementation Size

| Component | Lines | Source Patterns |
|-----------|-------|-----------------|
| `conftest.py` (session fixtures) | ~60 | GE, Deepchecks |
| `expected_ranges.yml` | ~80 | GE |
| `test_integration_aapl.py` | ~250 | Lean, Deepchecks |
| `test_cache_schema.py` | ~80 | Pandera |
| `test_cache_constraints.py` | ~100 | whylogs |
| `test_profile_structure.py` | ~40 | Syrupy |
| `test_regression_metrics.py` | ~60 | Lean |
| `test_graceful_degradation.py` | ~80 | Evidently |
| `test_result_serialization.py` | ~50 | Evidently |
| `test_threshold_consistency.py` | ~60 | Lean |
| **Total** | **~860** | |

This is larger than the original Batch C estimate (~410 lines) because the research revealed additional test categories (schema validation, constraint reports, auto-discovery, regression metrics) that address real bug patterns we've experienced.
