"""Pipeline model smoke tests.

Tests each analytical model (not wrappers) by importing it, building
a small synthetic cache, and calling the main entry point. Reports
pass/fail, latency, and error messages.

Usage:
    from operator1.monitoring.model_tests import run_model_tests
    results = run_model_tests()  # all models
    results = run_model_tests(["derived_variables", "survival_mode"])  # specific
"""

from __future__ import annotations

import logging
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass, field
from typing import Any

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


@dataclass
class ModelTestResult:
    """Result of a single model smoke test."""
    name: str = ""
    layer: str = ""          # features, analysis, temporal
    status: str = "unknown"  # ok, fail, skip
    latency_ms: int = 0
    error: str = ""
    detail: str = ""

    def to_dict(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "layer": self.layer,
            "status": self.status,
            "latency_ms": self.latency_ms,
            "error": self.error,
            "detail": self.detail,
        }


def _build_synthetic_cache(n_rows: int = 100) -> pd.DataFrame:
    """Build a realistic synthetic daily cache for smoke tests."""
    np.random.seed(42)
    dates = pd.bdate_range("2024-01-01", periods=n_rows, freq="B")

    close = 100 + np.cumsum(np.random.randn(n_rows) * 0.5)
    volume = np.random.randint(1_000_000, 10_000_000, n_rows).astype(float)

    cache = pd.DataFrame({
        "open": close + np.random.randn(n_rows) * 0.3,
        "high": close + abs(np.random.randn(n_rows) * 0.5),
        "low": close - abs(np.random.randn(n_rows) * 0.5),
        "close": close,
        "volume": volume,
        "revenue": np.random.uniform(1e9, 5e9, n_rows),
        "net_income": np.random.uniform(1e8, 1e9, n_rows),
        "total_assets": np.random.uniform(5e9, 20e9, n_rows),
        "total_liabilities": np.random.uniform(2e9, 10e9, n_rows),
        "total_equity": np.random.uniform(2e9, 10e9, n_rows),
        "current_assets": np.random.uniform(1e9, 5e9, n_rows),
        "current_liabilities": np.random.uniform(5e8, 3e9, n_rows),
        "cash_and_equivalents": np.random.uniform(1e8, 2e9, n_rows),
        "total_debt": np.random.uniform(1e9, 8e9, n_rows),
        "operating_cash_flow": np.random.uniform(1e8, 2e9, n_rows),
        "capex": np.random.uniform(-5e8, -1e7, n_rows),
        "interest_expense": np.random.uniform(1e7, 2e8, n_rows),
        "ebitda": np.random.uniform(5e8, 3e9, n_rows),
        "shares_outstanding": np.full(n_rows, 1e9),
    }, index=dates)
    cache.index.name = "date"
    return cache


def _test_import(module_path: str, names: list[str]) -> tuple[bool, str]:
    """Test that a module and its symbols can be imported."""
    try:
        import importlib
        mod = importlib.import_module(module_path)
        for name in names:
            if not hasattr(mod, name):
                return False, f"Missing symbol: {name}"
        return True, "Import OK"
    except Exception as exc:
        return False, f"Import failed: {str(exc)[:120]}"


# ---------------------------------------------------------------------------
# Test definitions
# ---------------------------------------------------------------------------

def _test_derived_variables(cache: pd.DataFrame) -> str:
    from operator1.features.derived_variables import compute_derived_variables
    result = compute_derived_variables(cache.copy())
    n_new = len(result.columns) - len(cache.columns)
    return f"{n_new} new columns added"


def _test_conflict_risk(cache: pd.DataFrame) -> str:
    from operator1.features.conflict_risk import assess_conflict_risk
    result = assess_conflict_risk("US")
    return f"flag={result.country_conflict_flag}, intensity={result.conflict_intensity_score:.3f}"


def _test_filing_calendar(cache: pd.DataFrame) -> str:
    from operator1.features.filing_calendar import analyze_filing_calendar
    result = analyze_filing_calendar(cache.copy(), market_id="us_sec_edgar")
    return f"freq={result.detected_frequency}, stale={result.is_stale}"


def _test_macro_quadrant(cache: pd.DataFrame) -> str:
    from operator1.features.macro_quadrant import compute_macro_quadrant
    c, result = compute_macro_quadrant(cache.copy(), macro_data=None)
    return f"quadrant={getattr(result, 'latest_quadrant', 'N/A')}"


def _test_news_sentiment(cache: pd.DataFrame) -> str:
    from operator1.features.news_sentiment import compute_news_sentiment
    c, result = compute_news_sentiment(cache.copy(), llm_client=None, symbol="TEST",
                                        market_id="us_sec_edgar", company_name="Test Corp")
    return f"articles={result.n_articles_fetched}"


def _test_peer_ranking(cache: pd.DataFrame) -> str:
    from operator1.features.peer_ranking import compute_peer_ranking
    c, result = compute_peer_ranking(cache.copy(), linked_caches={})
    return f"peers={result.n_peers}, vars={result.n_variables_ranked}"


def _test_private_company(cache: pd.DataFrame) -> str:
    from operator1.features.private_company_proxies import is_private_company
    return f"is_private={is_private_company(cache)}"


def _test_institutional_flow(cache: pd.DataFrame) -> str:
    from operator1.features.institutional_flow import compute_institutional_flow
    result = compute_institutional_flow(cache.copy())
    return f"cols={len(result.columns)}"


def _test_market_buying_power(cache: pd.DataFrame) -> str:
    from operator1.features.market_buying_power import compute_market_buying_power
    c, result = compute_market_buying_power(cache.copy(), sector="Technology", country_iso2="US")
    return f"bpi={result.buying_power_index:.0f}, momentum={result.sector_demand_momentum:+.2f}"


def _test_product_catalysts(cache: pd.DataFrame) -> str:
    from operator1.features.product_catalysts import detect_product_catalysts
    c, result = detect_product_catalysts(cache.copy(), profile={"sector": "Technology"})
    return f"score={result.catalyst_score:.2f}, type={result.catalyst_type}"


def _test_survival_mode(cache: pd.DataFrame) -> str:
    from operator1.analysis.survival_mode import compute_company_survival_flag
    flags = compute_company_survival_flag(cache.copy())
    return f"flagged={int(flags.sum())}/{len(flags)}"


def _test_hierarchy_weights(cache: pd.DataFrame) -> str:
    from operator1.analysis.hierarchy_weights import compute_hierarchy_weights
    c = cache.copy()
    c["company_survival_mode_flag"] = 0
    c["country_survival_mode_flag"] = 0
    c["country_protected_flag"] = 0
    result = compute_hierarchy_weights(c)
    return f"cols added: {sum(1 for col in result.columns if col.startswith('hierarchy_'))}"


def _test_fuzzy_protection(cache: pd.DataFrame) -> str:
    from operator1.analysis.fuzzy_protection import compute_fuzzy_protection
    c = cache.copy()
    c["company_survival_mode_flag"] = 0
    result = compute_fuzzy_protection(c, sector="Technology")
    return f"degree={result['fuzzy_protection_degree'].mean():.3f}"


def _test_financial_health(cache: pd.DataFrame) -> str:
    from operator1.models.financial_health import compute_financial_health
    c = cache.copy()
    from operator1.features.derived_variables import compute_derived_variables
    c = compute_derived_variables(c)
    c, result = compute_financial_health(c, hierarchy_weights={})
    return f"composite={result.latest_composite:.1f}, label={result.latest_label}"


def _test_vanity(cache: pd.DataFrame) -> str:
    from operator1.analysis.vanity import compute_vanity_score
    result = compute_vanity_score(cache.copy())
    has_score = "vanity_score" in result.columns
    return f"vanity_score present: {has_score}"


def _test_economic_planes(cache: pd.DataFrame) -> str:
    from operator1.analysis.economic_planes import classify_economic_plane
    result = classify_economic_plane("Technology", "Software")
    return f"plane={result.get('primary_plane', '?')}"


def _test_adaptive_thresholds(cache: pd.DataFrame) -> str:
    from operator1.analysis.adaptive_thresholds import compute_adaptive_thresholds
    result = compute_adaptive_thresholds(cache.copy())
    return f"adapted={result.adapted}"


def _test_survival_timeline(cache: pd.DataFrame) -> str:
    from operator1.analysis.survival_timeline import compute_survival_timeline
    c = cache.copy()
    c["company_survival_mode_flag"] = 0
    c["country_survival_mode_flag"] = 0
    c["country_protected_flag"] = 0
    result = compute_survival_timeline(c)
    return f"n_switches={result.n_switches}"


def _test_regime_detector(cache: pd.DataFrame) -> str:
    from operator1.models.regime_detector import run_early_regime_detection
    c = cache.copy()
    from operator1.features.derived_variables import compute_derived_variables
    c = compute_derived_variables(c)
    c, result = run_early_regime_detection(c)
    return f"fitted={result.fitted}" if result else "no result"


def _test_regime_mixer(cache: pd.DataFrame) -> str:
    from operator1.models.regime_mixer import compute_dual_regimes
    result = compute_dual_regimes(cache.copy())
    return f"fitted={result.fitted}" if result else "no result"


def _test_granger_causality(cache: pd.DataFrame) -> str:
    from operator1.models.granger_causality import compute_granger_causality
    result = compute_granger_causality(cache.copy())
    return f"pairs={len(result.significant_pairs)}, density={result.network_density:.3f}"


def _test_transfer_entropy(cache: pd.DataFrame) -> str:
    from operator1.models.causality import compute_transfer_entropy
    result = compute_transfer_entropy(cache.copy())
    return f"available={getattr(result, 'available', False)}"


def _test_cycle_decomposition(cache: pd.DataFrame) -> str:
    from operator1.models.cycle_decomposition import run_cycle_decomposition
    result = run_cycle_decomposition(cache.copy())
    n = len(result.dominant_cycles) if hasattr(result, "dominant_cycles") else 0
    return f"cycles={n}"


def _test_pattern_detector(cache: pd.DataFrame) -> str:
    from operator1.models.pattern_detector import detect_patterns
    result = detect_patterns(cache.copy())
    return f"patterns={getattr(result, 'n_patterns', 0)}"


def _test_forecasting(cache: pd.DataFrame) -> str:
    from operator1.models.forecasting import run_forecasting
    from operator1.features.derived_variables import compute_derived_variables
    c = compute_derived_variables(cache.copy())
    c, result = run_forecasting(c)
    n_vars = len(result.forecasts) if result and result.forecasts else 0
    return f"vars_forecast={n_vars}"


def _test_monte_carlo(cache: pd.DataFrame) -> str:
    from operator1.models.monte_carlo import run_monte_carlo
    from operator1.features.derived_variables import compute_derived_variables
    c = compute_derived_variables(cache.copy())
    result = run_monte_carlo(c)
    surv = getattr(result, "survival_probability_mean", None)
    return f"surv_prob={surv:.4f}" if surv is not None else "no result"


def _test_copula(cache: pd.DataFrame) -> str:
    from operator1.models.copula import run_copula_analysis
    result = run_copula_analysis(cache.copy())
    return f"available={getattr(result, 'available', False)}"


def _test_dtw_analogs(cache: pd.DataFrame) -> str:
    from operator1.models.dtw_analogs import find_historical_analogs
    result = find_historical_analogs(cache.copy())
    return f"available={getattr(result, 'available', False)}"


def _test_particle_filter(cache: pd.DataFrame) -> str:
    from operator1.models.particle_filter import run_particle_filter
    from operator1.features.derived_variables import compute_derived_variables
    c = compute_derived_variables(cache.copy())
    result = run_particle_filter(c, variables=["current_ratio"])
    return f"available={getattr(result, 'available', False)}"


def _test_sensitivity(cache: pd.DataFrame) -> str:
    from operator1.models.sensitivity import run_sensitivity_analysis
    from operator1.features.derived_variables import compute_derived_variables
    c = compute_derived_variables(cache.copy())
    result = run_sensitivity_analysis(c, target_variable="return_1d")
    return f"available={getattr(result, 'available', False)}"


# Tests that only check importability (need too much setup for smoke test)
_IMPORT_ONLY_TESTS = {
    "walk_forward": ("operator1.models.walk_forward", ["run_walk_forward"]),
    "conformal": ("operator1.models.conformal", ["ConformalCalibrator", "build_conformal_result"]),
    "transformer_forecaster": ("operator1.models.transformer_forecaster", ["train_transformer"]),
    "explainability": ("operator1.models.explainability", ["compute_shap_explanations"]),
    "genetic_optimizer": ("operator1.models.genetic_optimizer", ["run_genetic_optimization"]),
    "prediction_aggregator": ("operator1.models.prediction_aggregator", ["run_prediction_aggregation"]),
    "model_synergies": ("operator1.models.model_synergies", ["apply_pre_forecasting_synergies"]),
}


# Full smoke tests
MODEL_TESTS: dict[str, dict[str, Any]] = {
    # Layer 1: Features
    "derived_variables":    {"layer": "features", "fn": _test_derived_variables},
    "conflict_risk":        {"layer": "features", "fn": _test_conflict_risk},
    "filing_calendar":      {"layer": "features", "fn": _test_filing_calendar},
    "macro_quadrant":       {"layer": "features", "fn": _test_macro_quadrant},
    "news_sentiment":       {"layer": "features", "fn": _test_news_sentiment},
    "peer_ranking":         {"layer": "features", "fn": _test_peer_ranking},
    "private_company":      {"layer": "features", "fn": _test_private_company},
    "institutional_flow":   {"layer": "features", "fn": _test_institutional_flow},
    "market_buying_power":  {"layer": "features", "fn": _test_market_buying_power},
    "product_catalysts":    {"layer": "features", "fn": _test_product_catalysts},
    # Layer 2: Analysis
    "survival_mode":        {"layer": "analysis", "fn": _test_survival_mode},
    "hierarchy_weights":    {"layer": "analysis", "fn": _test_hierarchy_weights},
    "fuzzy_protection":     {"layer": "analysis", "fn": _test_fuzzy_protection},
    "financial_health":     {"layer": "analysis", "fn": _test_financial_health},
    "vanity":               {"layer": "analysis", "fn": _test_vanity},
    "economic_planes":      {"layer": "analysis", "fn": _test_economic_planes},
    "adaptive_thresholds":  {"layer": "analysis", "fn": _test_adaptive_thresholds},
    "survival_timeline":    {"layer": "analysis", "fn": _test_survival_timeline},
    # Layer 3: Temporal
    "regime_detector":      {"layer": "temporal", "fn": _test_regime_detector},
    "regime_mixer":         {"layer": "temporal", "fn": _test_regime_mixer},
    "granger_causality":    {"layer": "temporal", "fn": _test_granger_causality},
    "transfer_entropy":     {"layer": "temporal", "fn": _test_transfer_entropy},
    "cycle_decomposition":  {"layer": "temporal", "fn": _test_cycle_decomposition},
    "pattern_detector":     {"layer": "temporal", "fn": _test_pattern_detector},
    "forecasting":          {"layer": "temporal", "fn": _test_forecasting},
    "monte_carlo":          {"layer": "temporal", "fn": _test_monte_carlo},
    "copula":               {"layer": "temporal", "fn": _test_copula},
    "dtw_analogs":          {"layer": "temporal", "fn": _test_dtw_analogs},
    "particle_filter":      {"layer": "temporal", "fn": _test_particle_filter},
    "sensitivity":          {"layer": "temporal", "fn": _test_sensitivity},
}


def run_single_test(name: str, cache: pd.DataFrame | None = None) -> ModelTestResult:
    """Run a single model smoke test."""
    if cache is None:
        cache = _build_synthetic_cache()

    # Import-only tests
    if name in _IMPORT_ONLY_TESTS:
        mod_path, symbols = _IMPORT_ONLY_TESTS[name]
        t0 = time.time()
        ok, detail = _test_import(mod_path, symbols)
        return ModelTestResult(
            name=name,
            layer="temporal",
            status="ok" if ok else "fail",
            latency_ms=int((time.time() - t0) * 1000),
            detail=detail,
            error="" if ok else detail,
        )

    # Full smoke tests
    config = MODEL_TESTS.get(name)
    if not config:
        return ModelTestResult(name=name, status="skip", error=f"Unknown test: {name}")

    result = ModelTestResult(name=name, layer=config["layer"])
    t0 = time.time()
    try:
        detail = config["fn"](cache)
        result.status = "ok"
        result.detail = detail
    except Exception as exc:
        result.status = "fail"
        result.error = str(exc)[:200]
    result.latency_ms = int((time.time() - t0) * 1000)
    return result


def run_model_tests(
    names: list[str] | None = None,
    max_workers: int = 1,
) -> list[ModelTestResult]:
    """Run model smoke tests.

    Parameters
    ----------
    names : list[str], optional
        Specific test names. Defaults to all tests.
    max_workers : int
        Parallelism level. Default 1 (sequential) since many models
        share global state.

    Returns
    -------
    List of ModelTestResult sorted by layer then name.
    """
    if names is None:
        names = list(MODEL_TESTS.keys()) + list(_IMPORT_ONLY_TESTS.keys())

    cache = _build_synthetic_cache()
    results: list[ModelTestResult] = []

    # Sequential by default (models may share state)
    for name in names:
        logger.info("Testing %s ...", name)
        r = run_single_test(name, cache)
        results.append(r)
        icon = "+" if r.status == "ok" else "!" if r.status == "fail" else "~"
        logger.info("  [%s] %s: %s (%dms) %s", icon, name, r.status, r.latency_ms,
                     r.error[:60] if r.error else r.detail[:60])

    # Sort by layer order then name
    layer_order = {"features": 0, "analysis": 1, "temporal": 2}
    results.sort(key=lambda r: (layer_order.get(r.layer, 9), r.name))
    return results
