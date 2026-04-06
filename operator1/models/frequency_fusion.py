"""Cross-frequency fusion for multi-scope forecasting.

Reconciles predictions, regimes, and survival probabilities from
5 frequency pipelines (annual -> daily) into a single coherent output.

Fusion methods:
  - Inverse-variance weighted prediction reconciliation
  - Regime consensus across frequencies (majority vote + disagreement detection)
  - Survival probability via harmonic mean (weakest-link principle)
  - Natural horizon matching (each frequency forecasts at its native horizon)
"""

from __future__ import annotations

import logging
import math
from dataclasses import dataclass, field
from typing import Any

import numpy as np

from operator1.scoring_weights import get_weight

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Horizon-to-frequency contribution weights
# ---------------------------------------------------------------------------

# For each prediction horizon, which frequencies contribute and with what
# base weight.  Faster frequencies dominate short horizons, slower
# frequencies dominate long horizons.
#
# These defaults can be overridden via config/scoring_weights.yml section
# ``frequency_fusion``.  The dashboard Scoring Weights > Frequency tab
# edits these values.
_DEFAULT_HORIZON_WEIGHTS: dict[str, dict[str, float]] = {
    "1d":  {"D": 1.0},
    "5d":  {"D": 0.7, "W": 0.3},
    "1w":  {"D": 0.4, "W": 0.6},
    "21d": {"D": 0.3, "W": 0.4, "M": 0.3},
    "1m":  {"W": 0.3, "M": 0.7},
    "3m":  {"W": 0.15, "M": 0.35, "Q": 0.50},
    "6m":  {"M": 0.25, "Q": 0.50, "A": 0.25},
    "1y":  {"M": 0.15, "Q": 0.35, "A": 0.50},
    "2y":  {"Q": 0.30, "A": 0.70},
}


def _get_horizon_weights() -> dict[str, dict[str, float]]:
    """Load horizon weights from scoring_weights config, falling back to defaults."""
    configured = get_weight("frequency_fusion", None)
    if configured and isinstance(configured, dict):
        # Merge configured values over defaults
        merged = dict(_DEFAULT_HORIZON_WEIGHTS)
        for horizon, freq_weights in configured.items():
            if isinstance(freq_weights, dict):
                merged[horizon] = {k: float(v) for k, v in freq_weights.items() if v}
        return merged
    return _DEFAULT_HORIZON_WEIGHTS


# ---------------------------------------------------------------------------
# Result containers
# ---------------------------------------------------------------------------

@dataclass
class RegimeConsensus:
    """Multi-frequency regime consensus."""

    consensus_regime: str = "unknown"
    agreement_ratio: float = 0.0       # 0-1, fraction of frequencies that agree
    frequency_regimes: dict[str, str] = field(default_factory=dict)
    disagreements: list[str] = field(default_factory=list)
    interpretation: str = ""


@dataclass
class FusedSurvival:
    """Multi-frequency survival probability fusion."""

    fused_probability: float = 1.0
    harmonic_mean: float = 1.0
    per_frequency: dict[str, float] = field(default_factory=dict)
    weakest_frequency: str = ""
    interpretation: str = ""


@dataclass
class FusedPrediction:
    """Fused prediction for a single variable at a single horizon."""

    variable: str = ""
    horizon: str = ""
    point_forecast: float | None = None
    lower_bound: float | None = None
    upper_bound: float | None = None
    contributing_frequencies: dict[str, float] = field(default_factory=dict)
    confidence: float = 0.0


@dataclass
class FrequencyFusionResult:
    """Complete fusion result across all frequencies."""

    regime_consensus: RegimeConsensus = field(default_factory=RegimeConsensus)
    survival: FusedSurvival = field(default_factory=FusedSurvival)
    predictions: list[FusedPrediction] = field(default_factory=list)
    frequency_summary: dict[str, dict[str, Any]] = field(default_factory=dict)
    n_frequencies_used: int = 0
    available: bool = False

    def to_profile_dict(self) -> dict[str, Any]:
        """Convert to JSON-serializable dict for the profile."""
        return {
            "available": self.available,
            "n_frequencies_used": self.n_frequencies_used,
            "regime_consensus": {
                "consensus_regime": self.regime_consensus.consensus_regime,
                "agreement_ratio": round(self.regime_consensus.agreement_ratio, 3),
                "frequency_regimes": self.regime_consensus.frequency_regimes,
                "disagreements": self.regime_consensus.disagreements,
                "interpretation": self.regime_consensus.interpretation,
            },
            "survival": {
                "fused_probability": round(self.survival.fused_probability, 4),
                "harmonic_mean": round(self.survival.harmonic_mean, 4),
                "per_frequency": {
                    k: round(v, 4) for k, v in self.survival.per_frequency.items()
                },
                "weakest_frequency": self.survival.weakest_frequency,
                "interpretation": self.survival.interpretation,
            },
            "predictions": [
                {
                    "variable": p.variable,
                    "horizon": p.horizon,
                    "point_forecast": round(p.point_forecast, 6) if p.point_forecast is not None else None,
                    "confidence": round(p.confidence, 3),
                    "contributing_frequencies": {
                        k: round(v, 3) for k, v in p.contributing_frequencies.items()
                    },
                }
                for p in self.predictions[:20]  # cap for profile size
            ],
            "frequency_summary": self.frequency_summary,
        }


# ---------------------------------------------------------------------------
# Regime consensus
# ---------------------------------------------------------------------------

def compute_regime_consensus(
    frequency_results: dict[str, Any],
) -> RegimeConsensus:
    """Build regime consensus from multi-frequency pipeline results.

    Parameters
    ----------
    frequency_results:
        Dict of frequency code -> FrequencyResult.
    """
    regimes: dict[str, str] = {}
    for freq, result in frequency_results.items():
        if hasattr(result, "regime_label") and result.regime_label != "unknown":
            regimes[freq] = result.regime_label

    if not regimes:
        return RegimeConsensus(interpretation="No regime data from any frequency.")

    # Majority vote
    from collections import Counter
    counts = Counter(regimes.values())
    most_common, most_count = counts.most_common(1)[0]
    agreement = most_count / len(regimes)

    # Detect disagreements
    disagreements = []
    if len(set(regimes.values())) > 1:
        for freq, regime in regimes.items():
            if regime != most_common:
                disagreements.append(
                    f"{freq} says '{regime}' while consensus is '{most_common}'"
                )

    # Generate interpretation
    interpretation = ""
    if agreement >= 0.8:
        interpretation = f"Strong consensus: {most_common} across {len(regimes)} frequencies."
    elif agreement >= 0.6:
        interpretation = (
            f"Moderate consensus: {most_common} ({agreement:.0%}). "
            f"Disagreements: {'; '.join(disagreements)}."
        )
    else:
        # Check for the common pattern: daily bear + longer-term bull
        daily_regime = regimes.get("D", "")
        slower_regimes = [r for f, r in regimes.items() if f != "D"]
        if daily_regime and slower_regimes:
            slower_consensus = Counter(slower_regimes).most_common(1)[0][0]
            if "bear" in daily_regime.lower() and "bull" in slower_consensus.lower():
                interpretation = (
                    "Short-term correction in a longer-term uptrend. "
                    "Daily data shows bearish signals but weekly/monthly remain bullish."
                )
            elif "bull" in daily_regime.lower() and "bear" in slower_consensus.lower():
                interpretation = (
                    "Bear market rally. Daily shows bullish momentum but "
                    "longer-term frequencies indicate structural bear market."
                )
            else:
                interpretation = (
                    f"Mixed signals: daily={daily_regime}, "
                    f"longer-term={slower_consensus}."
                )
        else:
            interpretation = f"Weak consensus ({agreement:.0%}). No clear regime agreement."

    return RegimeConsensus(
        consensus_regime=most_common,
        agreement_ratio=agreement,
        frequency_regimes=regimes,
        disagreements=disagreements,
        interpretation=interpretation,
    )


# ---------------------------------------------------------------------------
# Survival probability fusion
# ---------------------------------------------------------------------------

def fuse_survival_probabilities(
    frequency_results: dict[str, Any],
) -> FusedSurvival:
    """Fuse survival probabilities using harmonic mean.

    Harmonic mean is used because survival is a "weakest link" problem:
    if ANY frequency shows high distress probability, that matters more
    than others showing safety.  The harmonic mean naturally weights
    toward the lowest value.
    """
    probs: dict[str, float] = {}
    for freq, result in frequency_results.items():
        if hasattr(result, "survival_probability"):
            p = result.survival_probability
            if p is not None and not math.isnan(p):
                probs[freq] = max(0.001, min(1.0, p))  # clamp to avoid division by zero

    if not probs:
        return FusedSurvival(interpretation="No survival data from any frequency.")

    # Harmonic mean
    n = len(probs)
    reciprocal_sum = sum(1.0 / p for p in probs.values())
    harmonic = n / reciprocal_sum if reciprocal_sum > 0 else 1.0

    # Weighted version: slower frequencies get slightly more weight for survival
    # (quarterly fundamentals are more reliable than daily volatility-driven survival)
    freq_weights = {"A": 1.5, "Q": 1.3, "M": 1.0, "W": 0.8, "D": 0.7}
    weighted_sum = sum(probs[f] * freq_weights.get(f, 1.0) for f in probs)
    weight_total = sum(freq_weights.get(f, 1.0) for f in probs)
    weighted_avg = weighted_sum / weight_total if weight_total > 0 else 1.0

    # Use the more conservative of harmonic mean and weighted average
    fused = min(harmonic, weighted_avg)

    weakest = min(probs, key=probs.get)

    # Interpretation
    if fused > 0.9:
        interpretation = f"Low distress risk across all frequencies (fused={fused:.1%})."
    elif fused > 0.7:
        interpretation = (
            f"Moderate risk. Weakest signal from {weakest} "
            f"({probs[weakest]:.1%}). Monitor closely."
        )
    elif fused > 0.5:
        interpretation = (
            f"Elevated risk. {weakest} frequency shows "
            f"{probs[weakest]:.1%} survival probability."
        )
    else:
        interpretation = (
            f"High distress risk. {weakest} frequency shows "
            f"only {probs[weakest]:.1%} survival probability. "
            f"Multi-frequency consensus confirms distress."
        )

    return FusedSurvival(
        fused_probability=round(fused, 4),
        harmonic_mean=round(harmonic, 4),
        per_frequency=probs,
        weakest_frequency=weakest,
        interpretation=interpretation,
    )


# ---------------------------------------------------------------------------
# Prediction fusion
# ---------------------------------------------------------------------------

def fuse_predictions(
    frequency_results: dict[str, Any],
) -> list[FusedPrediction]:
    """Fuse predictions across frequencies for each horizon.

    Each horizon gets contributions from its natural frequencies
    weighted by the ``_HORIZON_WEIGHTS`` table, further adjusted
    by each frequency's walk-forward accuracy (if available).
    """
    fused: list[FusedPrediction] = []

    # Collect all variable/horizon pairs across all frequencies
    all_forecasts: dict[str, dict[str, dict[str, float]]] = {}
    # Structure: {variable: {horizon: {frequency: value}}}

    for freq, result in frequency_results.items():
        if not hasattr(result, "forecast_summary"):
            continue
        for key, value in result.forecast_summary.items():
            if value is None:
                continue
            # Key format: "variable_horizon" (e.g., "close_1d", "revenue_21d")
            parts = key.rsplit("_", 1)
            if len(parts) != 2:
                continue
            var, horizon = parts
            all_forecasts.setdefault(var, {}).setdefault(horizon, {})[freq] = value

    for var, horizons in all_forecasts.items():
        for horizon, freq_values in horizons.items():
            if not freq_values:
                continue

            # Get base weights for this horizon (from config or defaults)
            hw = _get_horizon_weights()
            base_weights = hw.get(horizon, {})
            if not base_weights:
                # Use equal weights if horizon not in table
                base_weights = {f: 1.0 for f in freq_values}

            # Compute weighted prediction
            total_weight = 0.0
            weighted_sum = 0.0
            contributing = {}

            for freq, value in freq_values.items():
                w = base_weights.get(freq, 0.1)
                # Adjust by walk-forward accuracy if available
                result = frequency_results.get(freq)
                if result and hasattr(result, "walk_forward_mae") and result.walk_forward_mae:
                    mae = result.walk_forward_mae
                    if mae > 0:
                        w *= 1.0 / mae  # inverse-MAE weighting
                weighted_sum += value * w
                total_weight += w
                contributing[freq] = w

            if total_weight > 0:
                point = weighted_sum / total_weight
                # Normalize contributing weights to sum to 1
                for f in contributing:
                    contributing[f] /= total_weight
                confidence = min(1.0, len(freq_values) / 3)  # more frequencies = more confident

                fused.append(FusedPrediction(
                    variable=var,
                    horizon=horizon,
                    point_forecast=point,
                    contributing_frequencies=contributing,
                    confidence=confidence,
                ))

    return fused


# ---------------------------------------------------------------------------
# Main fusion function
# ---------------------------------------------------------------------------

def fuse_multi_frequency_results(
    multi_result: Any,
) -> FrequencyFusionResult:
    """Fuse results from all frequency pipelines into a unified output.

    Parameters
    ----------
    multi_result:
        ``MultiFrequencyResult`` from ``run_multi_frequency_pipeline()``.

    Returns
    -------
    FrequencyFusionResult with regime consensus, survival fusion, and
    reconciled predictions.
    """
    results = multi_result.results if hasattr(multi_result, "results") else {}

    if not results:
        return FrequencyFusionResult(available=False)

    # 1. Regime consensus
    regime_consensus = compute_regime_consensus(results)
    logger.info(
        "Regime consensus: %s (agreement=%.0f%%, %d frequencies)",
        regime_consensus.consensus_regime,
        regime_consensus.agreement_ratio * 100,
        len(regime_consensus.frequency_regimes),
    )

    # 2. Survival probability fusion
    survival = fuse_survival_probabilities(results)
    logger.info(
        "Survival fusion: %.1f%% (harmonic=%.1f%%, weakest=%s at %.1f%%)",
        survival.fused_probability * 100,
        survival.harmonic_mean * 100,
        survival.weakest_frequency,
        survival.per_frequency.get(survival.weakest_frequency, 0) * 100,
    )

    # 3. Prediction fusion
    predictions = fuse_predictions(results)
    logger.info("Prediction fusion: %d fused predictions", len(predictions))

    # F2: Multi-frequency constraint propagation.
    # Slower frequencies constrain faster frequencies: annual forecast
    # bounds quarterly, which bounds monthly, which bounds daily.
    # Prevents fast-frequency models from producing forecasts that
    # are inconsistent with slower-frequency structural trends.
    _n_constrained = 0
    try:
        # Extract forecast bounds from each frequency's context
        _freq_bounds: dict[str, dict[str, tuple[float, float]]] = {}
        _freq_order = ["A", "Q", "M", "W", "D"]  # slow to fast

        for freq, result in results.items():
            _ctx = getattr(result, "context", None)
            if _ctx and hasattr(_ctx, "forecast_bounds"):
                _bounds = _ctx.forecast_bounds
                if isinstance(_bounds, dict) and _bounds:
                    _freq_bounds[freq] = _bounds

        # Apply constraints: for each fused prediction, check if it violates
        # the bounds from any slower frequency
        if _freq_bounds and predictions:
            for pred in predictions:
                _var = pred.variable
                _horizon = pred.horizon
                _point = pred.point_forecast

                if _point is None or (_point != _point):  # NaN check
                    continue

                # Find the slowest frequency that has bounds for this variable
                for slow_freq in _freq_order:
                    if slow_freq in _freq_bounds and _var in _freq_bounds[slow_freq]:
                        _lo, _hi = _freq_bounds[slow_freq][_var]
                        if _lo is not None and _hi is not None and _hi > _lo:
                            # Scale bounds by horizon fraction
                            # (annual bound constrains daily proportionally)
                            _orig = _point
                            if _point > _hi:
                                pred.point_forecast = _hi
                                _n_constrained += 1
                            elif _point < _lo:
                                pred.point_forecast = _lo
                                _n_constrained += 1
                            break  # slowest constraint wins

        if _n_constrained > 0:
            logger.info(
                "F2 constraint propagation: %d predictions constrained by slower frequencies",
                _n_constrained,
            )
    except Exception as _exc:
        logger.debug("F2 constraint propagation skipped: %s", _exc)

    # 4. Build frequency summary
    freq_summary = {}
    for freq, result in results.items():
        freq_summary[freq] = {
            "label": result.label,
            "n_periods": result.n_periods,
            "trend_direction": result.trend_direction,
            "regime_label": result.regime_label,
            "survival_probability": round(result.survival_probability, 4),
            "elapsed_seconds": round(result.elapsed_seconds, 1),
        }

    return FrequencyFusionResult(
        regime_consensus=regime_consensus,
        survival=survival,
        predictions=predictions,
        frequency_summary=freq_summary,
        n_frequencies_used=len(results),
        available=True,
    )
