"""Predicted regime shifts -- forward-looking regime change probabilities.

Synthesizes the HMM transition matrix, stability score, and transition
half-life into dated regime shift predictions:

  - Probability of leaving the current regime within N days
  - Expected next regime (most probable transition target)
  - Estimated transition window (date range)

This fills the gap identified in App core idea Section E.5:
``pred_regime_shifts_next_year`` -- a list of predicted regime changes
with approximate dates.

The transition matrix is computed by ``monte_carlo.estimate_transition_matrix``
from historical regime labels.  The stability score and transition half-life
come from ``adaptive_model_params`` and the enriched survival timeline.

Top-level entry point:
    ``predict_regime_shifts(...)`` -> ``RegimeShiftPrediction``
"""

from __future__ import annotations

import logging
import math
from dataclasses import dataclass, field
from datetime import date, timedelta
from typing import Any

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


@dataclass
class PredictedShift:
    """A single predicted regime transition."""

    from_regime: str
    to_regime: str
    probability: float          # cumulative probability of this transition
    expected_day: int           # trading days from now
    expected_date: str          # approximate calendar date (ISO format)
    confidence: float           # how reliable is this prediction (0-1)


@dataclass
class RegimeShiftPrediction:
    """Container for all forward-looking regime shift predictions."""

    available: bool = False
    current_regime: str = ""
    n_regimes: int = 0

    # Per-horizon cumulative probability of leaving current regime
    prob_exit_5d: float = 0.0
    prob_exit_21d: float = 0.0
    prob_exit_63d: float = 0.0
    prob_exit_252d: float = 0.0

    # Expected days until next regime change (geometric distribution)
    expected_days_to_shift: float = float("inf")

    # Most probable next regime
    most_probable_next_regime: str = ""
    most_probable_next_prob: float = 0.0

    # Dated shift predictions (up to 4 horizons)
    predicted_shifts: list[PredictedShift] = field(default_factory=list)

    # Stability context
    stability_score: float = 1.0
    transition_halflife: int = 63

    error: str | None = None

    def to_dict(self) -> dict[str, Any]:
        """JSON-serializable dict for profile injection."""
        return {
            "available": self.available,
            "current_regime": self.current_regime,
            "n_regimes": self.n_regimes,
            "prob_exit_5d": round(self.prob_exit_5d, 4),
            "prob_exit_21d": round(self.prob_exit_21d, 4),
            "prob_exit_63d": round(self.prob_exit_63d, 4),
            "prob_exit_252d": round(self.prob_exit_252d, 4),
            "expected_days_to_shift": (
                round(self.expected_days_to_shift, 1)
                if not math.isinf(self.expected_days_to_shift)
                else None
            ),
            "most_probable_next_regime": self.most_probable_next_regime,
            "most_probable_next_prob": round(self.most_probable_next_prob, 4),
            "predicted_shifts": [
                {
                    "from_regime": s.from_regime,
                    "to_regime": s.to_regime,
                    "probability": round(s.probability, 4),
                    "expected_day": s.expected_day,
                    "expected_date": s.expected_date,
                    "confidence": round(s.confidence, 3),
                }
                for s in self.predicted_shifts
            ],
            "stability_score": round(self.stability_score, 3),
            "transition_halflife": self.transition_halflife,
            "error": self.error,
        }


def predict_regime_shifts(
    cache: pd.DataFrame,
    transition_matrix: np.ndarray | None = None,
    regime_order: list[str] | None = None,
    stability_score: float | None = None,
    transition_halflife: int | None = None,
    reference_date: date | None = None,
) -> RegimeShiftPrediction:
    """Predict future regime shifts from the HMM transition matrix.

    Uses the Markov property: given the current regime, the probability
    of being in regime *j* after *n* steps is ``(T^n)[i, j]`` where *T*
    is the one-step transition matrix and *i* is the current regime index.

    The probability of exiting the current regime within *n* days is:
    ``1 - (T^n)[i, i]`` (complement of staying in the same regime).

    The expected time to exit is the mean of the geometric distribution:
    ``1 / (1 - T[i, i])`` days.

    Parameters
    ----------
    cache:
        Daily cache with ``regime_label`` column.
    transition_matrix:
        Row-stochastic transition matrix from
        ``monte_carlo.estimate_transition_matrix()``.  If None, estimated
        from cache ``regime_label`` column directly.
    regime_order:
        List of regime labels corresponding to matrix rows/columns.
    stability_score:
        Latest 21-day stability score from enriched timeline (0-1).
    transition_halflife:
        Estimated half-life of regime transitions in trading days.
    reference_date:
        Date to use as "today" for calendar date predictions.
        Defaults to the last date in the cache.

    Returns
    -------
    RegimeShiftPrediction with forward-looking regime change estimates.
    """
    result = RegimeShiftPrediction()

    try:
        # Determine current regime from cache
        if "regime_label" not in cache.columns:
            result.error = "no regime_label in cache"
            return result

        regime_labels = cache["regime_label"].dropna()
        if len(regime_labels) < 30:
            result.error = "insufficient regime labels (<30 days)"
            return result

        current_regime = str(regime_labels.iloc[-1])
        result.current_regime = current_regime

        # Build or validate transition matrix
        if transition_matrix is None or regime_order is None:
            transition_matrix, regime_order = _estimate_transition_matrix(
                regime_labels.values
            )

        if transition_matrix is None or len(regime_order) < 2:
            result.error = "could not estimate transition matrix"
            return result

        result.n_regimes = len(regime_order)

        # Find current regime index
        if current_regime not in regime_order:
            # Try matching by prefix or substring
            matched = False
            for i, r in enumerate(regime_order):
                if str(r) == current_regime or current_regime in str(r):
                    current_idx = i
                    matched = True
                    break
            if not matched:
                result.error = f"current regime '{current_regime}' not in transition matrix"
                return result
        else:
            current_idx = regime_order.index(current_regime)

        T = transition_matrix
        n_regimes = T.shape[0]

        # Self-transition probability (probability of staying)
        p_stay = float(T[current_idx, current_idx])
        p_exit_1d = 1.0 - p_stay

        # Cumulative exit probabilities via matrix power
        # P(exit within n days) = 1 - (T^n)[i, i]
        for horizon, attr in [
            (5, "prob_exit_5d"),
            (21, "prob_exit_21d"),
            (63, "prob_exit_63d"),
            (252, "prob_exit_252d"),
        ]:
            try:
                T_n = np.linalg.matrix_power(T, horizon)
                p_stay_n = float(T_n[current_idx, current_idx])
                setattr(result, attr, round(1.0 - p_stay_n, 6))
            except Exception:
                setattr(result, attr, 1.0 - p_stay ** horizon)

        # Expected days to next regime change (geometric distribution)
        if p_exit_1d > 1e-8:
            result.expected_days_to_shift = 1.0 / p_exit_1d
        else:
            result.expected_days_to_shift = float("inf")

        # Most probable next regime (excluding self-transition)
        transition_probs = T[current_idx].copy()
        transition_probs[current_idx] = 0.0  # exclude self
        total_exit = transition_probs.sum()
        if total_exit > 0:
            transition_probs /= total_exit  # normalize exit probabilities
            best_next_idx = int(np.argmax(transition_probs))
            result.most_probable_next_regime = str(regime_order[best_next_idx])
            result.most_probable_next_prob = float(transition_probs[best_next_idx])

        # Stability context
        if stability_score is not None:
            result.stability_score = stability_score
        elif "stability_score_21d" in cache.columns:
            stab = cache["stability_score_21d"].dropna()
            if len(stab) > 0:
                result.stability_score = float(stab.iloc[-1])

        if transition_halflife is not None:
            result.transition_halflife = transition_halflife

        # Reference date for calendar predictions
        if reference_date is None:
            if hasattr(cache.index[-1], "date"):
                reference_date = cache.index[-1].date()
            else:
                reference_date = date.today()

        # Build predicted shift list (one per possible target regime)
        for j in range(n_regimes):
            if j == current_idx:
                continue
            target_regime = str(regime_order[j])
            # Probability of transitioning to this specific regime
            # within one step, given that we exit
            if total_exit > 0:
                p_this_target = float(T[current_idx, j]) / p_exit_1d if p_exit_1d > 0 else 0
            else:
                p_this_target = 0

            if p_this_target < 0.05:
                continue  # skip negligible transitions

            # Expected days to this specific transition
            # Approximate: expected_exit_days / p_this_target_given_exit
            if p_exit_1d > 1e-8 and p_this_target > 0:
                expected_day = int(result.expected_days_to_shift / max(p_this_target, 0.01))
            else:
                expected_day = 252

            expected_day = min(expected_day, 504)  # cap at 2 years

            # Calendar date
            expected_date = reference_date + timedelta(days=int(expected_day * 1.4))  # trading -> calendar
            expected_date_str = expected_date.isoformat()

            # Confidence: higher when stability is low (regime already shaky)
            # and when the transition probability is high
            instability = 1.0 - result.stability_score
            confidence = min(0.95, p_this_target * 0.6 + instability * 0.3 + 0.1)

            # Cumulative probability of this transition within 252 days
            cumulative_prob = result.prob_exit_252d * p_this_target

            result.predicted_shifts.append(PredictedShift(
                from_regime=current_regime,
                to_regime=target_regime,
                probability=cumulative_prob,
                expected_day=expected_day,
                expected_date=expected_date_str,
                confidence=confidence,
            ))

        # Sort by probability descending
        result.predicted_shifts.sort(key=lambda s: s.probability, reverse=True)

        result.available = True
        logger.info(
            "Regime shift prediction: current=%s, P(exit 21d)=%.1f%%, "
            "P(exit 252d)=%.1f%%, expected_days=%.0f, next=%s (%.1f%%)",
            current_regime,
            result.prob_exit_21d * 100,
            result.prob_exit_252d * 100,
            result.expected_days_to_shift,
            result.most_probable_next_regime,
            result.most_probable_next_prob * 100,
        )

    except Exception as exc:
        result.error = str(exc)
        logger.warning("Regime shift prediction failed: %s", exc)

    return result


def _estimate_transition_matrix(
    regime_labels: np.ndarray,
) -> tuple[np.ndarray | None, list[str]]:
    """Estimate transition matrix from regime label sequence.

    Counts transitions between consecutive days and normalizes rows
    to get a row-stochastic matrix.
    """
    # Clean labels
    clean = [str(r) for r in regime_labels if r is not None and str(r) != "nan"]
    if len(clean) < 30:
        return None, []

    unique = sorted(set(clean))
    n = len(unique)
    if n < 2:
        return None, unique

    idx_map = {r: i for i, r in enumerate(unique)}
    counts = np.zeros((n, n), dtype=np.float64)

    for t in range(len(clean) - 1):
        i = idx_map[clean[t]]
        j = idx_map[clean[t + 1]]
        counts[i, j] += 1

    # Row-normalize (add small epsilon to avoid zero rows)
    row_sums = counts.sum(axis=1, keepdims=True)
    row_sums = np.maximum(row_sums, 1e-10)
    T = counts / row_sums

    return T, unique
