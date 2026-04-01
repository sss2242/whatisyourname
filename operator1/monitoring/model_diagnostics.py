"""Model Expected Path vs Actual Path Diagnostics.

For each major model in the pipeline, pre-computes an "expected path"
based on data characteristics (return distribution shape, autocorrelation
structure, filing frequency, survival flags), then compares it against
what the model actually produced.  Outputs per-model diagnostic cards
with deviation scores and robustness ratings.

Similar to how the survival timeline pre-analyzes macro data to predict
regime transitions, this module pre-analyzes the cache to predict what
each model SHOULD produce, then measures how far reality deviated.

Top-level entry point:
    ``compute_model_diagnostics(cache, model_results)``
"""

from __future__ import annotations

import logging
import math
from dataclasses import dataclass, field
from typing import Any

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Data characteristic extraction
# ---------------------------------------------------------------------------


def _extract_data_characteristics(cache: pd.DataFrame) -> dict[str, Any]:
    """Extract statistical characteristics from cache for model expectations.

    Computes: skewness, kurtosis, bimodality coefficient, ACF at lag 1,
    ARCH LM proxy, stationarity proxy, data density, filing frequency.
    """
    chars: dict[str, Any] = {}

    returns = cache.get("return_1d")
    if returns is not None:
        clean = returns.dropna()
        if len(clean) > 30:
            chars["n_observations"] = len(clean)
            chars["skewness"] = float(clean.skew())
            chars["kurtosis"] = float(clean.kurtosis())  # excess kurtosis

            # Bimodality coefficient: BC = (skew^2 + 1) / kurtosis
            # BC > 0.555 suggests bimodal distribution (2+ regimes)
            kurt_adj = chars["kurtosis"] + 3  # raw kurtosis
            if kurt_adj > 0:
                chars["bimodality_coeff"] = (chars["skewness"] ** 2 + 1) / kurt_adj
            else:
                chars["bimodality_coeff"] = 0.0

            # ACF at lag 1 (autocorrelation)
            if len(clean) > 5:
                chars["acf_lag1"] = float(clean.autocorr(lag=1))
            else:
                chars["acf_lag1"] = 0.0

            # ARCH LM proxy: autocorrelation of squared returns
            # High value suggests GARCH should fit well
            sq_returns = clean ** 2
            if len(sq_returns) > 5:
                chars["arch_proxy"] = float(sq_returns.autocorr(lag=1))
            else:
                chars["arch_proxy"] = 0.0

            # Volatility clustering: ratio of max 21d vol to min 21d vol
            vol = cache.get("volatility_21d")
            if vol is not None:
                vol_clean = vol.dropna()
                if len(vol_clean) > 42:
                    vol_max = float(vol_clean.max())
                    vol_min = float(vol_clean.min())
                    if vol_min > 1e-8:
                        chars["vol_ratio"] = vol_max / vol_min
                    else:
                        chars["vol_ratio"] = 1.0
                else:
                    chars["vol_ratio"] = 1.0
            else:
                chars["vol_ratio"] = 1.0

    # Survival flags active
    if "company_survival_mode_flag" in cache.columns:
        sf = cache["company_survival_mode_flag"].dropna()
        chars["survival_days_pct"] = float(sf.mean()) if len(sf) > 0 else 0.0
    else:
        chars["survival_days_pct"] = 0.0

    # Data completeness (non-NaN ratio across key columns)
    key_cols = ["close", "revenue", "total_assets", "net_income", "operating_cash_flow"]
    available = [c for c in key_cols if c in cache.columns]
    if available:
        coverage = sum(cache[c].notna().mean() for c in available) / len(available)
        chars["data_coverage"] = float(coverage)
    else:
        chars["data_coverage"] = 0.0

    # Filing frequency (from filing_calendar if available)
    if "filing_freshness" in cache.columns:
        freshness = cache["filing_freshness"].dropna()
        chars["avg_filing_freshness"] = float(freshness.mean()) if len(freshness) > 0 else 0.5
    else:
        chars["avg_filing_freshness"] = 0.5

    return chars


# ---------------------------------------------------------------------------
# Per-model expected path computation
# ---------------------------------------------------------------------------


def _expected_regime_detector(chars: dict) -> dict[str, Any]:
    """Predict what the regime detector should find."""
    expected = {
        "model": "regime_detector",
        "expected_n_regimes": 2,
        "expected_dominant": "unknown",
        "reasoning": [],
    }

    bc = chars.get("bimodality_coeff", 0.0)
    skew = chars.get("skewness", 0.0)
    kurt = chars.get("kurtosis", 0.0)
    vol_ratio = chars.get("vol_ratio", 1.0)

    # Bimodality suggests multiple regimes
    if bc > 0.555:
        expected["expected_n_regimes"] = 3
        expected["reasoning"].append(
            f"Bimodal returns (BC={bc:.3f}>0.555) suggest 3+ distinct regimes"
        )
    else:
        expected["reasoning"].append(
            f"Unimodal returns (BC={bc:.3f}) suggest 2 regimes (normal + stressed)"
        )

    # High kurtosis suggests a high-vol regime exists
    if kurt > 3:
        expected["expected_n_regimes"] = max(expected["expected_n_regimes"], 3)
        expected["reasoning"].append(
            f"Fat tails (kurtosis={kurt:.1f}>3) indicate a high-volatility regime"
        )

    # Negative skew suggests bear regime
    if skew < -0.5:
        expected["expected_dominant"] = "bear"
        expected["reasoning"].append(
            f"Negative skew ({skew:.2f}) indicates asymmetric downside risk"
        )
    elif skew > 0.3:
        expected["expected_dominant"] = "bull"
        expected["reasoning"].append(
            f"Positive skew ({skew:.2f}) indicates upward drift dominance"
        )

    # High vol ratio suggests distinct vol regimes
    if vol_ratio > 3:
        expected["reasoning"].append(
            f"High vol ratio ({vol_ratio:.1f}x) confirms distinct volatility regimes"
        )

    return expected


def _expected_forecasting(chars: dict) -> dict[str, Any]:
    """Predict which forecasting model should dominate."""
    expected = {
        "model": "forecasting",
        "expected_best_model": "baseline",
        "reasoning": [],
    }

    acf = chars.get("acf_lag1", 0.0)
    arch = chars.get("arch_proxy", 0.0)
    n_obs = chars.get("n_observations", 0)

    # High ACF -> Kalman (linear state-space model)
    if abs(acf) > 0.1:
        expected["expected_best_model"] = "kalman"
        expected["reasoning"].append(
            f"Significant autocorrelation (ACF1={acf:.3f}) favors Kalman filter"
        )
    # ARCH effects -> GARCH
    elif arch > 0.15:
        expected["expected_best_model"] = "garch"
        expected["reasoning"].append(
            f"Volatility clustering (ARCH proxy={arch:.3f}) favors GARCH"
        )
    # Enough data for deep learning
    elif n_obs > 300:
        expected["expected_best_model"] = "lstm"
        expected["reasoning"].append(
            f"Sufficient data ({n_obs} obs) for LSTM/tree non-linear patterns"
        )
    else:
        expected["reasoning"].append(
            f"Limited data ({n_obs} obs) and weak structure -- baseline expected"
        )

    return expected


def _expected_monte_carlo(chars: dict) -> dict[str, Any]:
    """Predict what Monte Carlo survival probability should be."""
    expected = {
        "model": "monte_carlo",
        "expected_survival_range": [0.0, 1.0],
        "reasoning": [],
    }

    survival_pct = chars.get("survival_days_pct", 0.0)
    kurt = chars.get("kurtosis", 0.0)

    if survival_pct > 0.3:
        expected["expected_survival_range"] = [0.2, 0.6]
        expected["reasoning"].append(
            f"High survival flag rate ({survival_pct:.0%}) suggests low MC survival"
        )
    elif survival_pct > 0.05:
        expected["expected_survival_range"] = [0.5, 0.85]
        expected["reasoning"].append(
            f"Moderate survival flag rate ({survival_pct:.0%}) suggests medium MC survival"
        )
    else:
        expected["expected_survival_range"] = [0.85, 1.0]
        expected["reasoning"].append(
            f"Low survival flag rate ({survival_pct:.0%}) suggests high MC survival"
        )

    if kurt > 5:
        # Fat tails increase tail risk
        expected["expected_survival_range"][0] -= 0.1
        expected["reasoning"].append(
            f"Extreme kurtosis ({kurt:.1f}) increases tail risk in simulations"
        )

    return expected


def _expected_financial_health(chars: dict) -> dict[str, Any]:
    """Predict expected financial health score range."""
    expected = {
        "model": "financial_health",
        "expected_score_range": [30, 70],
        "reasoning": [],
    }

    survival_pct = chars.get("survival_days_pct", 0.0)
    coverage = chars.get("data_coverage", 0.0)

    if survival_pct > 0.3:
        expected["expected_score_range"] = [0, 30]
        expected["reasoning"].append(
            "Frequent survival triggers indicate poor financial health"
        )
    elif survival_pct > 0.05:
        expected["expected_score_range"] = [20, 50]
        expected["reasoning"].append(
            "Occasional survival triggers suggest stressed health"
        )
    else:
        expected["expected_score_range"] = [40, 80]
        expected["reasoning"].append(
            "No significant survival triggers suggest moderate-to-good health"
        )

    if coverage < 0.5:
        expected["reasoning"].append(
            f"Low data coverage ({coverage:.0%}) may inflate score uncertainty"
        )

    return expected


def _expected_estimation(chars: dict) -> dict[str, Any]:
    """Predict expected estimation coverage."""
    expected = {
        "model": "estimation",
        "expected_estimated_pct_range": [0.0, 1.0],
        "reasoning": [],
    }

    freshness = chars.get("avg_filing_freshness", 0.5)
    coverage = chars.get("data_coverage", 0.0)

    if coverage > 0.8:
        expected["expected_estimated_pct_range"] = [0.0, 0.3]
        expected["reasoning"].append(
            f"High data coverage ({coverage:.0%}) means most values are observed"
        )
    elif coverage > 0.5:
        expected["expected_estimated_pct_range"] = [0.2, 0.6]
        expected["reasoning"].append(
            f"Medium coverage ({coverage:.0%}) requires moderate estimation"
        )
    else:
        expected["expected_estimated_pct_range"] = [0.5, 0.95]
        expected["reasoning"].append(
            f"Low coverage ({coverage:.0%}) means heavy reliance on estimation"
        )

    return expected


def _expected_copula(chars: dict) -> dict[str, Any]:
    """Predict which copula type should win AIC."""
    expected = {
        "model": "copula",
        "expected_best_copula": "gaussian",
        "reasoning": [],
    }
    kurt = chars.get("kurtosis", 0.0)
    if kurt > 3:
        expected["expected_best_copula"] = "student_t"
        expected["reasoning"].append(
            f"Fat tails (kurtosis={kurt:.1f}) favor Student-t copula (tail dependence)"
        )
    elif kurt > 1:
        expected["reasoning"].append(
            f"Moderate tails (kurtosis={kurt:.1f}) -- Gaussian copula likely adequate"
        )
    else:
        expected["reasoning"].append("Thin tails suggest Gaussian copula")
    return expected


def _expected_granger(chars: dict) -> dict[str, Any]:
    """Predict Granger causality network density."""
    expected = {
        "model": "granger_causality",
        "expected_density_range": [0.0, 0.5],
        "reasoning": [],
    }
    n_obs = chars.get("n_observations", 0)
    if n_obs > 200:
        expected["expected_density_range"] = [0.05, 0.4]
        expected["reasoning"].append(f"Sufficient data ({n_obs} obs) for causal detection")
    elif n_obs > 50:
        expected["expected_density_range"] = [0.0, 0.2]
        expected["reasoning"].append(f"Limited data ({n_obs} obs) -- sparse causal network expected")
    else:
        expected["expected_density_range"] = [0.0, 0.05]
        expected["reasoning"].append(f"Very short series ({n_obs} obs) -- minimal causality detectable")
    return expected


def _expected_cycle(chars: dict) -> dict[str, Any]:
    """Predict dominant cycle period from filing frequency."""
    expected = {
        "model": "cycle_decomposition",
        "expected_dominant_period_range": [40, 130],
        "reasoning": [],
    }
    freshness = chars.get("avg_filing_freshness", 0.5)
    if freshness > 0.7:
        expected["expected_dominant_period_range"] = [50, 80]
        expected["reasoning"].append("High filing freshness suggests quarterly cycle (~63 days)")
    elif freshness > 0.3:
        expected["expected_dominant_period_range"] = [80, 180]
        expected["reasoning"].append("Medium freshness suggests semi-annual cycle")
    else:
        expected["expected_dominant_period_range"] = [180, 280]
        expected["reasoning"].append("Low freshness suggests annual cycle")
    return expected


def _expected_dtw(chars: dict) -> dict[str, Any]:
    """Predict DTW analog match quality."""
    expected = {
        "model": "dtw_analogs",
        "expected_match_quality": "medium",
        "reasoning": [],
    }
    n_obs = chars.get("n_observations", 0)
    vol_ratio = chars.get("vol_ratio", 1.0)
    if n_obs > 300:
        expected["expected_match_quality"] = "high"
        expected["reasoning"].append(f"Long history ({n_obs} days) provides rich analog pool")
    elif n_obs > 100:
        expected["reasoning"].append(f"Moderate history ({n_obs} days) for analog matching")
    else:
        expected["expected_match_quality"] = "low"
        expected["reasoning"].append(f"Short history ({n_obs} days) limits analog quality")
    if vol_ratio > 3:
        expected["reasoning"].append(f"High vol ratio ({vol_ratio:.1f}) -- analogs from similar vol regimes preferred")
    return expected


def _expected_conformal(chars: dict) -> dict[str, Any]:
    """Predict conformal prediction interval behavior."""
    expected = {
        "model": "conformal_prediction",
        "expected_coverage_near_target": True,
        "expected_width": "normal",
        "reasoning": [],
    }
    vol_ratio = chars.get("vol_ratio", 1.0)
    if vol_ratio > 3:
        expected["expected_width"] = "wide"
        expected["reasoning"].append(
            f"High vol ratio ({vol_ratio:.1f}) requires wider intervals for coverage"
        )
    else:
        expected["reasoning"].append("Stable volatility should allow tight, accurate intervals")
    return expected


# ---------------------------------------------------------------------------
# Compare expected vs actual
# ---------------------------------------------------------------------------


def _compare_copula(expected: dict, copula_result: Any) -> dict[str, Any]:
    """Compare expected copula type against actual."""
    result = {**expected, "actual": {}, "deviation": 0.0, "robustness": "unknown"}
    if copula_result is None:
        result["actual"]["status"] = "not_run"
        result["robustness"] = "n/a"
        return result
    best = getattr(copula_result, "best_copula", None)
    if best is None:
        result["actual"]["status"] = "no_copula_fitted"
        result["robustness"] = "low"
        return result
    result["actual"]["best_copula"] = str(best)
    exp = expected.get("expected_best_copula", "gaussian")
    result["deviation"] = 0.0 if str(best).lower() == exp.lower() else 0.5
    result["robustness"] = "high" if result["deviation"] == 0 else "medium"
    return result


def _compare_granger(expected: dict, granger_result: Any) -> dict[str, Any]:
    """Compare expected Granger density against actual."""
    result = {**expected, "actual": {}, "deviation": 0.0, "robustness": "unknown"}
    if granger_result is None:
        result["actual"]["status"] = "not_run"
        result["robustness"] = "n/a"
        return result
    density = getattr(granger_result, "network_density", None)
    if density is None:
        result["actual"]["status"] = "no_density"
        result["robustness"] = "low"
        return result
    result["actual"]["network_density"] = round(float(density), 4)
    lo, hi = expected.get("expected_density_range", [0, 1])
    if density < lo:
        deviation = (lo - density) / max(hi - lo, 0.01)
    elif density > hi:
        deviation = (density - hi) / max(hi - lo, 0.01)
    else:
        deviation = 0.0
    result["deviation"] = round(min(deviation, 2.0), 3)
    result["robustness"] = "high" if deviation < 0.3 else "medium" if deviation < 1.0 else "low"
    return result


def _compare_cycle(expected: dict, cycle_result: Any) -> dict[str, Any]:
    """Compare expected cycle period against actual."""
    result = {**expected, "actual": {}, "deviation": 0.0, "robustness": "unknown"}
    if cycle_result is None:
        result["actual"]["status"] = "not_run"
        result["robustness"] = "n/a"
        return result
    cycles = getattr(cycle_result, "dominant_cycles", [])
    if not cycles:
        result["actual"]["status"] = "no_cycles_detected"
        result["robustness"] = "medium"
        return result
    # Use the first (strongest) cycle
    if isinstance(cycles[0], dict):
        period = cycles[0].get("period", 0)
    elif hasattr(cycles[0], "period"):
        period = cycles[0].period
    else:
        period = 0
    result["actual"]["dominant_period"] = round(float(period), 1) if period else 0
    lo, hi = expected.get("expected_dominant_period_range", [40, 130])
    if period and lo <= period <= hi:
        deviation = 0.0
    elif period:
        mid = (lo + hi) / 2
        deviation = abs(period - mid) / max(hi - lo, 1)
    else:
        deviation = 0.5
    result["deviation"] = round(min(deviation, 2.0), 3)
    result["robustness"] = "high" if deviation < 0.3 else "medium" if deviation < 1.0 else "low"
    return result


def _compare_dtw(expected: dict, dtw_result: Any) -> dict[str, Any]:
    """Compare expected DTW match quality against actual."""
    result = {**expected, "actual": {}, "deviation": 0.0, "robustness": "unknown"}
    if dtw_result is None:
        result["actual"]["status"] = "not_run"
        result["robustness"] = "n/a"
        return result
    matches = getattr(dtw_result, "matches", []) or getattr(dtw_result, "analogs", [])
    n_matches = len(matches) if matches else 0
    result["actual"]["n_matches"] = n_matches
    exp_quality = expected.get("expected_match_quality", "medium")
    quality_map = {"high": 5, "medium": 3, "low": 1}
    exp_n = quality_map.get(exp_quality, 3)
    actual_quality = "high" if n_matches >= 5 else "medium" if n_matches >= 2 else "low"
    result["actual"]["match_quality"] = actual_quality
    result["deviation"] = 0.0 if actual_quality == exp_quality else 0.3 if abs(quality_map.get(actual_quality, 0) - exp_n) <= 2 else 0.7
    result["robustness"] = "high" if result["deviation"] < 0.2 else "medium" if result["deviation"] < 0.5 else "low"
    return result


def _compare_conformal(expected: dict, conformal_result: Any) -> dict[str, Any]:
    """Compare expected conformal coverage against actual."""
    result = {**expected, "actual": {}, "deviation": 0.0, "robustness": "unknown"}
    if conformal_result is None:
        result["actual"]["status"] = "not_run"
        result["robustness"] = "n/a"
        return result
    coverage = getattr(conformal_result, "empirical_coverage", None)
    if coverage is None:
        # Try alternate attributes
        coverage = getattr(conformal_result, "coverage", None)
    if coverage is not None:
        result["actual"]["empirical_coverage"] = round(float(coverage), 3)
        deviation = abs(float(coverage) - 0.9) / 0.1  # distance from 90% target
        result["deviation"] = round(min(deviation, 2.0), 3)
        result["robustness"] = "high" if deviation < 0.3 else "medium" if deviation < 1.0 else "low"
    else:
        result["actual"]["status"] = "coverage_not_available"
        result["robustness"] = "medium"
    return result


def _compare_regime_detector(
    expected: dict, cache: pd.DataFrame,
) -> dict[str, Any]:
    """Compare expected regime detection against actual."""
    result = {**expected, "actual": {}, "deviation": 0.0, "robustness": "unknown"}

    if "regime_label" not in cache.columns:
        result["actual"]["status"] = "not_run"
        result["robustness"] = "n/a"
        return result

    regimes = cache["regime_label"].dropna()
    if regimes.empty:
        result["actual"]["status"] = "no_labels"
        result["robustness"] = "low"
        return result

    n_actual = len(regimes.unique())
    result["actual"]["n_regimes"] = n_actual
    result["actual"]["regime_distribution"] = regimes.value_counts(normalize=True).round(3).to_dict()

    # Deviation: how far from expected regime count
    exp_n = expected.get("expected_n_regimes", 2)
    deviation = abs(n_actual - exp_n) / max(exp_n, 1)
    result["deviation"] = round(deviation, 3)
    result["robustness"] = "high" if deviation < 0.34 else "medium" if deviation < 0.67 else "low"

    return result


def _compare_forecasting(
    expected: dict, forecast_result: Any,
) -> dict[str, Any]:
    """Compare expected forecasting model against actual."""
    result = {**expected, "actual": {}, "deviation": 0.0, "robustness": "unknown"}

    if forecast_result is None:
        result["actual"]["status"] = "not_run"
        result["robustness"] = "n/a"
        return result

    model_used = getattr(forecast_result, "model_used", {})
    if not model_used:
        result["actual"]["status"] = "no_models_fitted"
        result["robustness"] = "low"
        return result

    # Find most common model type
    model_counts: dict[str, int] = {}
    for var, model in model_used.items():
        base = model.split("_")[0] if model else "baseline"
        model_counts[base] = model_counts.get(base, 0) + 1

    dominant = max(model_counts, key=model_counts.get) if model_counts else "baseline"
    result["actual"]["dominant_model"] = dominant
    result["actual"]["model_distribution"] = model_counts

    # Deviation: 0 if expected matches actual, 1 if totally different
    exp_best = expected.get("expected_best_model", "baseline")
    result["deviation"] = 0.0 if dominant == exp_best else 0.5
    result["robustness"] = "high" if result["deviation"] == 0 else "medium"

    return result


def _compare_monte_carlo(
    expected: dict, mc_result: Any,
) -> dict[str, Any]:
    """Compare expected MC survival against actual."""
    result = {**expected, "actual": {}, "deviation": 0.0, "robustness": "unknown"}

    if mc_result is None:
        result["actual"]["status"] = "not_run"
        result["robustness"] = "n/a"
        return result

    actual_surv = getattr(mc_result, "survival_probability_mean", float("nan"))
    if math.isnan(actual_surv):
        result["actual"]["status"] = "no_survival_computed"
        result["robustness"] = "low"
        return result

    result["actual"]["survival_probability_mean"] = round(actual_surv, 4)

    # Deviation: how far actual is from expected range
    lo, hi = expected.get("expected_survival_range", [0, 1])
    if actual_surv < lo:
        deviation = (lo - actual_surv) / max(hi - lo, 0.01)
    elif actual_surv > hi:
        deviation = (actual_surv - hi) / max(hi - lo, 0.01)
    else:
        deviation = 0.0

    result["deviation"] = round(min(deviation, 2.0), 3)
    result["robustness"] = "high" if deviation < 0.3 else "medium" if deviation < 1.0 else "low"

    return result


def _compare_financial_health(
    expected: dict, cache: pd.DataFrame,
) -> dict[str, Any]:
    """Compare expected FH score against actual."""
    result = {**expected, "actual": {}, "deviation": 0.0, "robustness": "unknown"}

    if "fh_composite_score" not in cache.columns:
        result["actual"]["status"] = "not_run"
        result["robustness"] = "n/a"
        return result

    fh = cache["fh_composite_score"].dropna()
    if fh.empty:
        result["actual"]["status"] = "empty"
        result["robustness"] = "low"
        return result

    actual_score = float(fh.iloc[-1])
    result["actual"]["composite_score"] = round(actual_score, 1)

    lo, hi = expected.get("expected_score_range", [30, 70])
    mid = (lo + hi) / 2
    width = max(hi - lo, 1)
    deviation = abs(actual_score - mid) / width
    # Only count deviation if outside expected range
    if lo <= actual_score <= hi:
        deviation = 0.0

    result["deviation"] = round(min(deviation, 2.0), 3)
    result["robustness"] = "high" if deviation < 0.3 else "medium" if deviation < 1.0 else "low"

    return result


def _compare_estimation(
    expected: dict, cache: pd.DataFrame,
) -> dict[str, Any]:
    """Compare expected estimation coverage against actual."""
    result = {**expected, "actual": {}, "deviation": 0.0, "robustness": "unknown"}

    # Count estimated vs observed across source columns
    source_cols = [c for c in cache.columns if c.endswith("_source")]
    if not source_cols:
        result["actual"]["status"] = "no_estimation_run"
        result["robustness"] = "n/a"
        return result

    total = 0
    estimated = 0
    for col in source_cols:
        vals = cache[col].dropna()
        total += len(vals)
        estimated += (vals != "observed").sum()

    if total > 0:
        est_pct = estimated / total
    else:
        est_pct = 0.0

    result["actual"]["estimated_pct"] = round(est_pct, 3)
    result["actual"]["n_source_columns"] = len(source_cols)

    lo, hi = expected.get("expected_estimated_pct_range", [0, 1])
    if est_pct < lo:
        deviation = (lo - est_pct) / max(hi - lo, 0.01)
    elif est_pct > hi:
        deviation = (est_pct - hi) / max(hi - lo, 0.01)
    else:
        deviation = 0.0

    result["deviation"] = round(min(deviation, 2.0), 3)
    result["robustness"] = "high" if deviation < 0.3 else "medium" if deviation < 1.0 else "low"

    return result


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


@dataclass
class ModelDiagnosticsResult:
    """Container for all model diagnostic results."""

    available: bool = False
    data_characteristics: dict[str, Any] = field(default_factory=dict)
    models: dict[str, dict[str, Any]] = field(default_factory=dict)
    overall_robustness: str = "unknown"
    n_models_assessed: int = 0
    n_models_on_track: int = 0
    n_models_deviated: int = 0
    error: str | None = None

    def to_dict(self) -> dict[str, Any]:
        """JSON-serializable dict for profile injection."""
        return {
            "available": self.available,
            "data_characteristics": self.data_characteristics,
            "models": self.models,
            "overall_robustness": self.overall_robustness,
            "n_models_assessed": self.n_models_assessed,
            "n_models_on_track": self.n_models_on_track,
            "n_models_deviated": self.n_models_deviated,
            "error": self.error,
        }


def compute_model_diagnostics(
    cache: pd.DataFrame,
    forecast_result: Any = None,
    mc_result: Any = None,
    copula_result: Any = None,
    granger_result: Any = None,
    cycle_result: Any = None,
    dtw_result: Any = None,
    conformal_result: Any = None,
) -> ModelDiagnosticsResult:
    """Compute expected path vs actual path diagnostics for all models.

    For each major model, pre-computes what it SHOULD produce based on
    data characteristics, then compares against actual results.

    Parameters
    ----------
    cache:
        Daily cache with all derived variables and model outputs.
    forecast_result:
        ``ForecastResult`` from ``run_forecasting()``.
    mc_result:
        ``MonteCarloResult`` from ``run_monte_carlo()``.

    Returns
    -------
    ModelDiagnosticsResult with per-model diagnostic cards.
    """
    result = ModelDiagnosticsResult()

    try:
        if cache is None or cache.empty or len(cache) < 10:
            result.error = "insufficient cache data"
            return result

        # Step 1: Extract data characteristics
        chars = _extract_data_characteristics(cache)
        result.data_characteristics = {
            k: round(v, 4) if isinstance(v, float) else v
            for k, v in chars.items()
        }

        # Step 2: Compute expected paths for each model
        expectations = {
            "regime_detector": _expected_regime_detector(chars),
            "forecasting": _expected_forecasting(chars),
            "monte_carlo": _expected_monte_carlo(chars),
            "financial_health": _expected_financial_health(chars),
            "estimation": _expected_estimation(chars),
            "copula": _expected_copula(chars),
            "granger_causality": _expected_granger(chars),
            "cycle_decomposition": _expected_cycle(chars),
            "dtw_analogs": _expected_dtw(chars),
            "conformal_prediction": _expected_conformal(chars),
        }

        # Step 3: Compare expected vs actual
        comparisons = {
            "regime_detector": _compare_regime_detector(
                expectations["regime_detector"], cache,
            ),
            "forecasting": _compare_forecasting(
                expectations["forecasting"], forecast_result,
            ),
            "monte_carlo": _compare_monte_carlo(
                expectations["monte_carlo"], mc_result,
            ),
            "financial_health": _compare_financial_health(
                expectations["financial_health"], cache,
            ),
            "estimation": _compare_estimation(
                expectations["estimation"], cache,
            ),
            "copula": _compare_copula(
                expectations["copula"], copula_result,
            ),
            "granger_causality": _compare_granger(
                expectations["granger_causality"], granger_result,
            ),
            "cycle_decomposition": _compare_cycle(
                expectations["cycle_decomposition"], cycle_result,
            ),
            "dtw_analogs": _compare_dtw(
                expectations["dtw_analogs"], dtw_result,
            ),
            "conformal_prediction": _compare_conformal(
                expectations["conformal_prediction"], conformal_result,
            ),
        }

        result.models = comparisons
        result.n_models_assessed = len(comparisons)

        # Count on-track vs deviated
        on_track = sum(
            1 for m in comparisons.values()
            if m.get("robustness") in ("high", "n/a")
        )
        deviated = sum(
            1 for m in comparisons.values()
            if m.get("robustness") == "low"
        )
        result.n_models_on_track = on_track
        result.n_models_deviated = deviated

        # Overall robustness
        if deviated == 0:
            result.overall_robustness = "high"
        elif deviated <= 1:
            result.overall_robustness = "medium"
        else:
            result.overall_robustness = "low"

        result.available = True

        logger.info(
            "Model diagnostics: %d models assessed, %d on track, %d deviated, "
            "overall=%s",
            result.n_models_assessed,
            result.n_models_on_track,
            result.n_models_deviated,
            result.overall_robustness,
        )

    except Exception as exc:
        result.error = str(exc)
        logger.warning("Model diagnostics failed: %s", exc)

    return result
