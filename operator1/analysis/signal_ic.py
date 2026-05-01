"""Signal Information Coefficient (IC) measurement and decay analysis.

Computes rolling Spearman rank correlation between every derived signal
and forward returns at multiple horizons.  Produces IC, ICIR (IC
Information Ratio), and per-signal decay curves.

The IC matrix is the single most important feedback mechanism in the
pipeline: it tells us which signals actually predict future returns and
which are noise.  Signals with |ICIR| < threshold are pruned from the
ensemble.

References:
    - Qian, Hua & Sorensen (2007). Quantitative Equity Portfolio Management.
    - Grinold & Kahn (2000). Active Portfolio Management, 2nd ed.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Any

import numpy as np
import pandas as pd
from scipy import stats

from operator1.config_loader import get_global_config

logger = logging.getLogger(__name__)

# Default horizons for IC computation (business days)
_DEFAULT_HORIZONS = [5, 21, 63]

# Signal speed classification for decay weighting
_SIGNAL_SPEED: dict[str, str] = {
    # Slow (fundamentals, balance sheet)
    "current_ratio": "slow",
    "debt_to_equity_abs": "slow",
    "cash_ratio": "slow",
    "interest_coverage": "slow",
    "net_debt": "slow",
    "total_debt": "slow",
    "fh_composite_score": "slow",
    "fh_altman_z_score": "slow",
    "fh_beneish_m_score": "slow",
    "fh_runway_months": "slow",
    "gross_margin": "slow",
    "operating_margin": "slow",
    "net_margin": "slow",
    "ebitda_margin": "slow",
    "fcf_yield": "slow",
    "pe_ratio_calc": "slow",
    "ev_to_ebitda": "slow",
    "accruals_signal": "slow",
    "debt_service_coverage": "slow",
    "vanity_score": "slow",
    # Medium (earnings momentum, PEAD, institutional flow)
    "sue_score": "medium",
    "pead_signal": "medium",
    "revenue_ttm": "medium",
    "net_income_ttm": "medium",
    "eps_surprise_proxy": "medium",
    "inst_flow_momentum": "medium",
    "inst_smart_money_signal": "medium",
    "sentiment_score": "medium",
    "catalyst_score": "medium",
    "buying_power_index": "medium",
    "sector_demand_momentum": "medium",
    # Fast (technicals, volatility, patterns)
    "return_1d": "fast",
    "return_5d": "fast",
    "volatility_21d": "fast",
    "volatility_63d": "fast",
    "drawdown_252d": "fast",
    "adx": "fast",
    "obv": "fast",
    "bb_width": "fast",
    "macd_histogram": "fast",
    "online_change_score": "fast",
    "beta_252d": "fast",
    # Layer 2 enhancements: survival velocity, uncertainty, duration, distress
    # Slow (fundamental, quarterly frequency)
    "survival_prob_liquidity": "slow",
    "survival_prob_solvency": "slow",
    "fh_ensemble_distress_prob": "slow",
    "fh_cvar_composite": "slow",
    # Medium (derived, changes with regimes)
    "survival_deterioration_rate": "medium",
    "dominant_risk_channel": "medium",
    "mode_exit_probability_21d": "medium",
    "covenant_proximity_score": "medium",
    # Fast (daily updates)
    "survival_velocity_flag": "fast",
    "survival_uncertainty": "fast",
    "early_warning_score": "fast",
}


@dataclass
class SignalICResult:
    """Result from IC computation."""

    available: bool = False
    n_signals: int = 0
    n_horizons: int = 0
    n_observations: int = 0

    # IC matrix: {signal_name: {horizon_str: ic_value}}
    ic_matrix: dict[str, dict[str, float]] = field(default_factory=dict)

    # ICIR matrix: {signal_name: {horizon_str: icir_value}}
    icir_matrix: dict[str, dict[str, float]] = field(default_factory=dict)

    # Signals that pass the ICIR threshold for the primary target
    strong_signals: list[str] = field(default_factory=list)
    weak_signals: list[str] = field(default_factory=list)

    # Primary target IC summary
    primary_target: str = "return_5d"
    primary_horizon: int = 5
    best_signal: str = ""
    best_ic: float = 0.0

    # Signal speed classifications
    signal_speeds: dict[str, str] = field(default_factory=dict)

    error: str = ""

    def to_profile_dict(self) -> dict[str, Any]:
        """Convert to JSON-serializable dict for the profile."""
        if not self.available:
            return {"available": False, "error": self.error}
        return {
            "available": True,
            "n_signals": self.n_signals,
            "n_horizons": self.n_horizons,
            "n_observations": self.n_observations,
            "primary_target": self.primary_target,
            "best_signal": self.best_signal,
            "best_ic": round(self.best_ic, 4),
            "n_strong_signals": len(self.strong_signals),
            "n_weak_signals": len(self.weak_signals),
            "strong_signals": self.strong_signals[:10],
            "ic_matrix": {
                s: {h: round(v, 4) for h, v in horizons.items()}
                for s, horizons in list(self.ic_matrix.items())[:20]
            },
        }


def classify_signal_speed(signal_name: str) -> str:
    """Classify a signal as slow/medium/fast for decay weighting.

    Returns "slow", "medium", or "fast".
    """
    if signal_name in _SIGNAL_SPEED:
        return _SIGNAL_SPEED[signal_name]
    # Heuristics for unknown signals
    name_lower = signal_name.lower()
    if any(k in name_lower for k in ("ratio", "margin", "yield", "score", "z_", "m_", "runway")):
        return "slow"
    if any(k in name_lower for k in ("sue", "pead", "surprise", "momentum", "sentiment", "inst_")):
        return "medium"
    if any(k in name_lower for k in ("return", "vol", "drawdown", "adx", "obv", "macd", "bb_", "beta")):
        return "fast"
    return "medium"  # conservative default


def compute_signal_ic(
    cache: pd.DataFrame,
    horizons: list[int] | None = None,
    min_observations: int = 60,
    rolling_window: int = 126,
) -> SignalICResult:
    """Compute rolling Spearman IC for all numeric signals vs forward returns.

    Parameters
    ----------
    cache:
        Daily cache with derived variables.
    horizons:
        Forward return horizons in business days (default: [5, 21, 63]).
    min_observations:
        Minimum non-NaN observations to compute IC.
    rolling_window:
        Rolling window for IC computation (126 = ~6 months).

    Returns
    -------
    SignalICResult with IC and ICIR matrices.
    """
    if horizons is None:
        horizons = _DEFAULT_HORIZONS

    cfg = get_global_config()
    contract = cfg.get("prediction_contract", {})
    ic_threshold = contract.get("ic_inclusion_threshold", 0.02)
    icir_threshold = contract.get("icir_inclusion_threshold", 0.3)
    primary_target = contract.get("primary_target", "return_5d")

    result = SignalICResult(
        primary_target=primary_target,
        n_horizons=len(horizons),
    )

    if cache.empty or len(cache) < min_observations:
        result.error = f"Insufficient data: {len(cache)} rows (need {min_observations})"
        return result

    # Compute forward returns for each horizon
    forward_returns: dict[int, pd.Series] = {}
    if "close" in cache.columns:
        close = cache["close"].dropna()
        if len(close) >= min_observations:
            for h in horizons:
                fr = close.pct_change(h).shift(-h)  # forward-looking
                forward_returns[h] = fr
    elif "return_1d" in cache.columns:
        # Construct from cumulative returns
        ret1d = cache["return_1d"].fillna(0)
        for h in horizons:
            forward_returns[h] = ret1d.rolling(h).sum().shift(-h)

    if not forward_returns:
        result.error = "No close or return_1d column for forward return computation"
        return result

    # Identify signal columns (numeric, non-trivial)
    signal_cols = [
        c for c in cache.select_dtypes(include=["number"]).columns
        if (cache[c].notna().sum() >= min_observations
            and cache[c].nunique() > 3
            and not c.startswith("is_missing_")
            and not c.startswith("invalid_math_")
            and not c.startswith("interp_confidence_")
            and not c.endswith("_observed")
            and not c.endswith("_estimated")
            and not c.endswith("_source")
            and not c.endswith("_confidence")
            and c not in ("is_partial_period",))
    ]

    if not signal_cols:
        result.error = "No signal columns with sufficient data"
        return result

    ic_matrix: dict[str, dict[str, float]] = {}
    icir_matrix: dict[str, dict[str, float]] = {}
    signal_speeds: dict[str, str] = {}

    for col in signal_cols:
        signal = cache[col]
        ic_matrix[col] = {}
        icir_matrix[col] = {}
        signal_speeds[col] = classify_signal_speed(col)

        for h in horizons:
            fr = forward_returns.get(h)
            if fr is None:
                continue

            # Align signal and forward return
            aligned = pd.DataFrame({"signal": signal, "forward": fr}).dropna()
            if len(aligned) < min_observations:
                continue

            # Rolling Spearman IC
            rolling_ics = []
            for start in range(0, len(aligned) - rolling_window + 1, rolling_window // 4):
                window = aligned.iloc[start:start + rolling_window]
                if len(window) >= min_observations // 2:
                    ic, _ = stats.spearmanr(window["signal"], window["forward"])
                    if not np.isnan(ic):
                        rolling_ics.append(ic)

            if rolling_ics:
                mean_ic = float(np.mean(rolling_ics))
                std_ic = float(np.std(rolling_ics)) if len(rolling_ics) > 1 else 1.0
                icir = mean_ic / std_ic if std_ic > 1e-8 else 0.0

                h_label = f"{h}d"
                ic_matrix[col][h_label] = mean_ic
                icir_matrix[col][h_label] = icir

    # Determine primary horizon label
    primary_h = int(primary_target.replace("return_", "").replace("d", "")) if "return_" in primary_target else 5
    primary_h_label = f"{primary_h}d"

    # Classify signals as strong or weak based on primary target ICIR
    strong_signals = []
    weak_signals = []
    best_signal = ""
    best_ic = 0.0

    for col in signal_cols:
        ic_val = ic_matrix.get(col, {}).get(primary_h_label, 0.0)
        icir_val = icir_matrix.get(col, {}).get(primary_h_label, 0.0)

        if abs(icir_val) >= icir_threshold and abs(ic_val) >= ic_threshold:
            strong_signals.append(col)
            if abs(ic_val) > abs(best_ic):
                best_ic = ic_val
                best_signal = col
        else:
            weak_signals.append(col)

    # Sort strong signals by |IC| descending
    strong_signals.sort(key=lambda s: abs(ic_matrix.get(s, {}).get(primary_h_label, 0.0)), reverse=True)

    result.available = True
    result.n_signals = len(signal_cols)
    result.n_observations = len(cache)
    result.ic_matrix = ic_matrix
    result.icir_matrix = icir_matrix
    result.strong_signals = strong_signals
    result.weak_signals = weak_signals
    result.best_signal = best_signal
    result.best_ic = best_ic
    result.signal_speeds = signal_speeds
    result.primary_horizon = primary_h

    logger.info(
        "Signal IC: %d signals, %d strong (|ICIR|>=%.1f), best=%s (IC=%.4f at %s)",
        len(signal_cols), len(strong_signals), icir_threshold,
        best_signal, best_ic, primary_h_label,
    )

    return result


def get_ic_weighted_signals(
    ic_result: SignalICResult,
    extra_vars: list[str],
) -> list[str]:
    """Filter extra_vars to only include signals with strong IC.

    Parameters
    ----------
    ic_result:
        Result from ``compute_signal_ic()``.
    extra_vars:
        Current list of extra variables for temporal models.

    Returns
    -------
    Filtered list keeping only IC-strong signals + always-keep variables.
    """
    if not ic_result.available or not ic_result.strong_signals:
        return extra_vars  # no filtering if IC not computed

    strong_set = set(ic_result.strong_signals)

    # Always keep these regardless of IC
    always_keep = {
        "survival_probability", "company_survival_mode_flag",
        "fh_composite_score", "stability_score_21d",
        "online_change_score",
    }

    filtered = [
        v for v in extra_vars
        if v in strong_set or v in always_keep
    ]

    n_pruned = len(extra_vars) - len(filtered)
    if n_pruned > 0:
        logger.info(
            "IC filter: %d -> %d variables (%d pruned for low IC)",
            len(extra_vars), len(filtered), n_pruned,
        )

    return filtered
