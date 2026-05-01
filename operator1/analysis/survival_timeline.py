"""Phase 2 -- Survival timeline pre-analysis.

Classifies every day in the 2-year daily cache into one of six survival
modes based on the combination of company and country survival flags
plus the country protection flag.

**Survival modes:**

1. ``normal``           -- no survival flags active
2. ``company_only``     -- company survival flag active, country normal
3. ``country_protected``-- country survival flag active, company normal,
                          BUT company is government-protected
4. ``country_exposed``  -- country survival flag active, company normal,
                          company is NOT protected
5. ``both_unprotected`` -- both company and country flags active,
                          company NOT protected
6. ``both_protected``   -- both company and country flags active,
                          company IS protected

**Additional computed columns:**

- ``switch_point``       -- 1 on days where the mode changes from the
                           previous day, 0 otherwise
- ``days_in_mode``       -- running counter of consecutive days in the
                           current mode (resets at each switch point)
- ``stability_score_21d``-- 21-day rolling stability score: fraction of
                           the last 21 days that share the same mode as
                           the current day (1.0 = fully stable)

Top-level entry point:
    ``compute_survival_timeline(daily_cache) -> SurvivalTimelineResult``
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Any

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Survival mode labels
# ---------------------------------------------------------------------------

SURVIVAL_MODES = (
    "normal",
    "company_only",
    "country_protected",
    "country_exposed",
    "both_unprotected",
    "both_protected",
)

# Map mode label -> integer code for efficient storage.
MODE_TO_CODE: dict[str, int] = {mode: idx for idx, mode in enumerate(SURVIVAL_MODES)}
CODE_TO_MODE: dict[int, str] = {idx: mode for idx, mode in enumerate(SURVIVAL_MODES)}

# Stability rolling window (business days).
_STABILITY_WINDOW: int = 21


# ---------------------------------------------------------------------------
# Result container
# ---------------------------------------------------------------------------


@dataclass
class SurvivalTimelineResult:
    """Output of the survival timeline computation."""

    # Full timeline DataFrame with mode, switch_point, days_in_mode,
    # stability_score_21d columns added.
    timeline: pd.DataFrame = field(default_factory=pd.DataFrame)

    # Switch points: list of (date, from_mode, to_mode) tuples.
    switch_points: list[dict[str, Any]] = field(default_factory=list)

    # Mode distribution: {mode_label: fraction_of_days}.
    mode_distribution: dict[str, float] = field(default_factory=dict)

    # Total number of switch points.
    n_switches: int = 0

    # Mean stability score across all days.
    mean_stability: float = float("nan")

    # Whether the computation succeeded.
    fitted: bool = False
    error: str | None = None


# ---------------------------------------------------------------------------
# Mode classification
# ---------------------------------------------------------------------------


def classify_survival_mode(
    company_flag: int,
    country_flag: int,
    protected_flag: int,
) -> str:
    """Classify a single day into one of six survival modes.

    Parameters
    ----------
    company_flag:
        1 if company is in survival mode, 0 otherwise.
    country_flag:
        1 if country is in survival mode, 0 otherwise.
    protected_flag:
        1 if the company is government-protected, 0 otherwise.

    Returns
    -------
    One of the six mode labels from ``SURVIVAL_MODES``.
    """
    comp = bool(company_flag)
    ctry = bool(country_flag)
    prot = bool(protected_flag)

    if not comp and not ctry:
        return "normal"
    if comp and not ctry:
        return "company_only"
    if not comp and ctry:
        return "country_protected" if prot else "country_exposed"
    # Both active
    return "both_protected" if prot else "both_unprotected"


def _classify_series(
    company_flags: pd.Series,
    country_flags: pd.Series,
    protected_flags: pd.Series,
) -> pd.Series:
    """Vectorised mode classification across the full daily index.

    Returns a Series of mode label strings.
    """
    comp = company_flags.fillna(0).astype(bool)
    ctry = country_flags.fillna(0).astype(bool)
    prot = protected_flags.fillna(0).astype(bool)

    # Default to normal
    modes = pd.Series("normal", index=company_flags.index, dtype="object")

    # company_only: comp=True, ctry=False
    modes.loc[comp & ~ctry] = "company_only"

    # country_protected: comp=False, ctry=True, prot=True
    modes.loc[~comp & ctry & prot] = "country_protected"

    # country_exposed: comp=False, ctry=True, prot=False
    modes.loc[~comp & ctry & ~prot] = "country_exposed"

    # both_unprotected: comp=True, ctry=True, prot=False
    modes.loc[comp & ctry & ~prot] = "both_unprotected"

    # both_protected: comp=True, ctry=True, prot=True
    modes.loc[comp & ctry & prot] = "both_protected"

    return modes


# ---------------------------------------------------------------------------
# Switch points
# ---------------------------------------------------------------------------


def _compute_switch_points(modes: pd.Series) -> pd.Series:
    """Return a binary series: 1 on days where the mode changes.

    The first day is always 0 (no prior day to compare).
    """
    shifted = modes.shift(1)
    switch = (modes != shifted).astype(int)
    # First day is not a switch
    if len(switch) > 0:
        switch.iloc[0] = 0
    return switch


def _extract_switch_list(
    modes: pd.Series,
    switch_flags: pd.Series,
) -> list[dict[str, Any]]:
    """Extract structured switch point records."""
    switches = []
    switch_dates = switch_flags.index[switch_flags == 1]
    for dt in switch_dates:
        loc = modes.index.get_loc(dt)
        from_mode = str(modes.iloc[loc - 1]) if loc > 0 else "unknown"
        to_mode = str(modes.iloc[loc])
        switches.append({
            "date": str(dt.date()) if hasattr(dt, "date") else str(dt),
            "from_mode": from_mode,
            "to_mode": to_mode,
        })
    return switches


# ---------------------------------------------------------------------------
# Days-in-mode counter
# ---------------------------------------------------------------------------


def _compute_days_in_mode(modes: pd.Series) -> pd.Series:
    """Running counter of consecutive days in the current mode.

    Resets to 1 each time the mode changes.
    """
    counter = pd.Series(0, index=modes.index, dtype=int)
    if len(modes) == 0:
        return counter

    count = 1
    prev = modes.iloc[0]
    counter.iloc[0] = 1

    for i in range(1, len(modes)):
        current = modes.iloc[i]
        if current == prev:
            count += 1
        else:
            count = 1
            prev = current
        counter.iloc[i] = count

    return counter


# ---------------------------------------------------------------------------
# Stability score
# ---------------------------------------------------------------------------


def _compute_stability_score(
    modes: pd.Series,
    window: int = _STABILITY_WINDOW,
) -> pd.Series:
    """21-day rolling stability score.

    For each day t, the score is the fraction of the last ``window``
    days (including t) that have the same mode as day t.
    Values range from 1/window (completely unstable) to 1.0 (fully
    stable -- same mode for the entire window).
    """
    if len(modes) == 0:
        return pd.Series(dtype=float)

    # Encode modes as integers for fast comparison
    mode_codes = modes.map(MODE_TO_CODE).fillna(-1).astype(int)

    stability = pd.Series(np.nan, index=modes.index, dtype=float)

    for i in range(len(mode_codes)):
        start = max(0, i - window + 1)
        window_slice = mode_codes.iloc[start:i + 1]
        current_code = mode_codes.iloc[i]
        n_same = (window_slice == current_code).sum()
        stability.iloc[i] = n_same / len(window_slice)

    return stability


# ---------------------------------------------------------------------------
# Semi-Markov Duration Modeling (Barbu & Limnios 2008)
# ---------------------------------------------------------------------------


def _compute_semi_markov_exit(
    modes: pd.Series,
    days_in_mode: pd.Series,
    switch_points: pd.Series,
    horizon: int = 21,
) -> tuple[pd.Series, pd.Series]:
    """Compute duration-aware mode exit probability and expected remaining time.

    Uses a Weibull distribution (generalizes geometric/Markov) fitted on
    historical dwell times per mode. Shape < 1 = decreasing hazard (distress
    trap), Shape > 1 = increasing hazard (recovery more likely over time).

    Falls back to geometric distribution (standard Markov) when fewer than
    3 dwell episodes are available for a mode.

    Parameters
    ----------
    modes:
        Daily survival mode labels.
    days_in_mode:
        Running day count in current mode.
    switch_points:
        Binary series (1 on mode switch days).
    horizon:
        Forecast horizon for exit probability (default 21 days).

    Returns
    -------
    tuple[pd.Series, pd.Series]
        (expected_remaining_days_in_mode, mode_exit_probability_21d)
    """
    import numpy as np

    expected_remaining = pd.Series(np.nan, index=modes.index, dtype=float)
    exit_prob = pd.Series(np.nan, index=modes.index, dtype=float)

    # Collect completed dwell times per mode from history
    mode_dwells: dict[str, list[float]] = {}
    current_mode = None
    current_dwell = 0
    for i in range(len(modes)):
        m = modes.iloc[i]
        if m != current_mode:
            if current_mode is not None and current_dwell > 0:
                mode_dwells.setdefault(current_mode, []).append(float(current_dwell))
            current_mode = m
            current_dwell = 1
        else:
            current_dwell += 1

    # Fit Weibull per mode and compute exit probabilities
    try:
        from scipy.stats import weibull_min
    except ImportError:
        logger.debug("scipy not available for semi-Markov; returning NaN")
        return expected_remaining, exit_prob

    weibull_params: dict[str, tuple[float, float]] = {}  # mode -> (shape, scale)
    for mode_name, dwells in mode_dwells.items():
        if len(dwells) < 3:
            # Fallback: geometric distribution (mean dwell time)
            mean_d = np.mean(dwells) if dwells else 63.0
            weibull_params[mode_name] = (1.0, mean_d)  # shape=1 = geometric
        else:
            try:
                shape, _loc, scale = weibull_min.fit(dwells, floc=0)
                shape = float(np.clip(shape, 0.1, 10.0))
                scale = float(np.clip(scale, 1.0, 1000.0))
                weibull_params[mode_name] = (shape, scale)
            except Exception:
                mean_d = np.mean(dwells)
                weibull_params[mode_name] = (1.0, mean_d)

    # Compute per-day exit probability and expected remaining days
    for i in range(len(modes)):
        mode = modes.iloc[i]
        d = float(days_in_mode.iloc[i])
        params = weibull_params.get(mode)
        if params is None:
            continue

        shape, scale = params
        try:
            # Hazard rate at current dwell time d
            sf_d = weibull_min.sf(d, shape, loc=0, scale=scale)
            sf_d_h = weibull_min.sf(d + horizon, shape, loc=0, scale=scale)
            if sf_d > 1e-10:
                # P(exit within horizon | survived to d) = 1 - S(d+h)/S(d)
                exit_prob.iloc[i] = float(np.clip(1.0 - sf_d_h / sf_d, 0, 1))
                # Expected remaining = integral of S(t)/S(d) from d to infinity
                # For Weibull: approximate as scale * Gamma(1 + 1/shape) - d (residual life)
                from math import gamma as gamma_fn
                mean_total = scale * gamma_fn(1.0 + 1.0 / max(shape, 0.1))
                expected_remaining.iloc[i] = max(mean_total - d, 0.0)
        except Exception:
            continue

    expected_remaining.name = "expected_remaining_days_in_mode"
    exit_prob.name = "mode_exit_probability_21d"

    n_valid = exit_prob.notna().sum()
    if n_valid > 0:
        logger.info(
            "Semi-Markov duration: %d modes fitted, mean exit prob=%.3f, "
            "mean expected remaining=%.0f days",
            len(weibull_params), exit_prob.mean(), expected_remaining.mean(),
        )

    return expected_remaining, exit_prob


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def compute_survival_timeline(
    daily_cache: pd.DataFrame,
) -> SurvivalTimelineResult:
    """Compute the full survival timeline for a daily cache.

    The input DataFrame must contain the three survival flag columns:
    - ``company_survival_mode_flag``
    - ``country_survival_mode_flag``
    - ``country_protected_flag``

    If any are missing, they default to 0 (normal).

    Parameters
    ----------
    daily_cache:
        Daily cache DataFrame (output of cache_builder + enrichment).

    Returns
    -------
    SurvivalTimelineResult
        Contains the enriched timeline DataFrame, switch point list,
        mode distribution, and stability statistics.
    """
    result = SurvivalTimelineResult()

    if daily_cache.empty:
        result.error = "Empty daily cache -- cannot compute survival timeline"
        logger.warning(result.error)
        return result

    try:
        # Extract flags (default to 0 if missing)
        company_flags = daily_cache.get(
            "company_survival_mode_flag",
            pd.Series(0, index=daily_cache.index),
        )
        country_flags = daily_cache.get(
            "country_survival_mode_flag",
            pd.Series(0, index=daily_cache.index),
        )
        protected_flags = daily_cache.get(
            "country_protected_flag",
            pd.Series(0, index=daily_cache.index),
        )

        # Also consider fuzzy protection degree if available:
        # treat protection_degree >= 0.5 as protected.
        if "fuzzy_protection_degree" in daily_cache.columns:
            fuzzy_prot = daily_cache["fuzzy_protection_degree"].fillna(0)
            # OR with binary flag: if either is protective, consider protected
            protected_flags = (
                protected_flags.fillna(0).astype(bool)
                | (fuzzy_prot >= 0.5)
            ).astype(int)

        # 1. Classify modes
        modes = _classify_series(company_flags, country_flags, protected_flags)
        modes.name = "survival_mode"

        # 2. Switch points
        switch_flags = _compute_switch_points(modes)
        switch_flags.name = "switch_point"

        # 3. Days in current mode
        days_counter = _compute_days_in_mode(modes)
        days_counter.name = "days_in_mode"

        # 4. Stability score
        stability = _compute_stability_score(modes)
        stability.name = "stability_score_21d"

        # 5. Mode code (integer encoding)
        mode_codes = modes.map(MODE_TO_CODE).astype(int)
        mode_codes.name = "survival_mode_code"

        # Write enriched columns back to the original cache so downstream
        # modules (prediction_aggregator, walk_forward) can read them
        # without needing to merge a separate DataFrame.
        daily_cache["survival_mode"] = modes
        daily_cache["survival_mode_code"] = mode_codes
        daily_cache["switch_point"] = switch_flags
        daily_cache["days_in_mode"] = days_counter
        daily_cache["stability_score_21d"] = stability

        # 6. Semi-Markov duration modeling (Barbu & Limnios 2008)
        # Computes duration-aware exit probability and expected remaining days
        try:
            exp_remaining, exit_prob = _compute_semi_markov_exit(
                modes, days_counter, switch_flags,
            )
            daily_cache["expected_remaining_days_in_mode"] = exp_remaining
            daily_cache["mode_exit_probability_21d"] = exit_prob
        except Exception as _sm_exc:
            logger.debug("Semi-Markov duration skipped: %s", _sm_exc)

        # Also keep a reference as the result timeline for backward compat.
        result.timeline = daily_cache

        # Extract structured switch points
        result.switch_points = _extract_switch_list(modes, switch_flags)
        result.n_switches = len(result.switch_points)

        # Mode distribution
        mode_counts = modes.value_counts(normalize=True)
        result.mode_distribution = {
            mode: float(mode_counts.get(mode, 0.0))
            for mode in SURVIVAL_MODES
        }

        # Mean stability
        result.mean_stability = float(stability.mean())

        result.fitted = True

        logger.info(
            "Survival timeline computed: %d days, %d switches, "
            "mean_stability=%.3f, distribution=%s",
            len(result.timeline),
            result.n_switches,
            result.mean_stability,
            {k: f"{v:.2%}" for k, v in result.mode_distribution.items() if v > 0},
        )

    except Exception as exc:
        result.error = f"Survival timeline computation failed: {exc}"
        logger.error(result.error)

    return result


def get_mode_at_date(
    timeline_result: SurvivalTimelineResult,
    target_date: pd.Timestamp,
) -> str:
    """Look up the survival mode for a specific date.

    Returns ``"unknown"`` if the date is not in the timeline.
    """
    if timeline_result.timeline.empty:
        return "unknown"
    if "survival_mode" not in timeline_result.timeline.columns:
        return "unknown"
    modes = timeline_result.timeline["survival_mode"]
    if target_date in modes.index:
        return str(modes.loc[target_date])
    return "unknown"


def get_switch_dates(
    timeline_result: SurvivalTimelineResult,
) -> list[pd.Timestamp]:
    """Return a list of dates where survival mode changes."""
    if timeline_result.timeline.empty:
        return []
    sp = timeline_result.timeline.get("switch_point")
    if sp is None:
        return []
    return list(sp.index[sp == 1])


# ---------------------------------------------------------------------------
# Enriched survival timeline (bridge between rule-based flags + HMM regimes)
# ---------------------------------------------------------------------------

# Combined state mapping: (survival_mode, market_regime) -> (state, intensity)
# Market regime comes from HMM (bull/bear/high_vol/low_vol) or "unknown".
# Intensity is a continuous [0, 1] score where 0 = safe, 1 = extreme crisis.
_COMBINED_STATE_MAP: dict[tuple[str, str], tuple[str, float]] = {
    # normal survival + market regimes
    ("normal", "bull"): ("stable_growth", 0.0),
    ("normal", "low_vol"): ("stable_growth", 0.05),
    ("normal", "high_vol"): ("elevated_risk", 0.25),
    ("normal", "bear"): ("market_stress", 0.30),
    ("normal", "unknown"): ("stable_growth", 0.05),
    # company_only + market regimes
    ("company_only", "bull"): ("company_distress_mild", 0.45),
    ("company_only", "low_vol"): ("company_distress_mild", 0.50),
    ("company_only", "high_vol"): ("company_distress_severe", 0.60),
    ("company_only", "bear"): ("company_distress_severe", 0.70),
    ("company_only", "unknown"): ("company_distress_mild", 0.55),
    # country_protected + market regimes
    ("country_protected", "bull"): ("protected_stress", 0.15),
    ("country_protected", "low_vol"): ("protected_stress", 0.20),
    ("country_protected", "high_vol"): ("protected_stress", 0.30),
    ("country_protected", "bear"): ("protected_stress", 0.35),
    ("country_protected", "unknown"): ("protected_stress", 0.25),
    # country_exposed + market regimes
    ("country_exposed", "bull"): ("country_crisis_mild", 0.40),
    ("country_exposed", "low_vol"): ("country_crisis_mild", 0.45),
    ("country_exposed", "high_vol"): ("country_crisis_severe", 0.60),
    ("country_exposed", "bear"): ("country_crisis_severe", 0.70),
    ("country_exposed", "unknown"): ("country_crisis_mild", 0.50),
    # both_unprotected + market regimes
    ("both_unprotected", "bull"): ("crisis", 0.70),
    ("both_unprotected", "low_vol"): ("crisis", 0.75),
    ("both_unprotected", "high_vol"): ("crisis", 0.90),
    ("both_unprotected", "bear"): ("extreme_crisis", 1.00),
    ("both_unprotected", "unknown"): ("crisis", 0.80),
    # both_protected + market regimes
    ("both_protected", "bull"): ("protected_crisis", 0.45),
    ("both_protected", "low_vol"): ("protected_crisis", 0.50),
    ("both_protected", "high_vol"): ("protected_crisis", 0.60),
    ("both_protected", "bear"): ("protected_crisis", 0.65),
    ("both_protected", "unknown"): ("protected_crisis", 0.55),
}

# All possible combined state labels (for documentation / validation).
COMBINED_STATES = sorted({v[0] for v in _COMBINED_STATE_MAP.values()})


@dataclass
class EnrichedTimelineResult:
    """Output of the enriched survival timeline computation.

    Extends ``SurvivalTimelineResult`` with regime-aware fields.
    """

    # Base survival timeline result.
    base: SurvivalTimelineResult = field(
        default_factory=SurvivalTimelineResult,
    )

    # Enriched timeline DataFrame (superset of base.timeline columns).
    timeline: pd.DataFrame = field(default_factory=pd.DataFrame)

    # Whether HMM regime data was available and incorporated.
    regime_available: bool = False

    # Combined state distribution: {state_label: fraction}.
    combined_state_distribution: dict[str, float] = field(default_factory=dict)

    # Mean survival intensity across all days.
    mean_intensity: float = float("nan")

    fitted: bool = False
    error: str | None = None


_WARNED_UNMAPPED_KEYS: set[tuple[str, str]] = set()


def _map_combined_state(
    survival_mode: str,
    market_regime: str,
) -> tuple[str, float]:
    """Look up combined state and intensity for a (survival_mode, market_regime) pair."""
    key = (survival_mode, market_regime)
    if key in _COMBINED_STATE_MAP:
        return _COMBINED_STATE_MAP[key]
    # Fallback: use the unknown-regime row for the survival mode.
    fallback_key = (survival_mode, "unknown")
    if fallback_key in _COMBINED_STATE_MAP:
        # Warn once per unmapped key so operators know they might want to
        # extend _COMBINED_STATE_MAP (e.g., when using n_regimes > 4).
        if key not in _WARNED_UNMAPPED_KEYS:
            _WARNED_UNMAPPED_KEYS.add(key)
            logger.warning(
                "Unmapped (survival_mode=%r, market_regime=%r) -- "
                "falling back to 'unknown' regime row. Consider extending "
                "_COMBINED_STATE_MAP if this regime label is expected.",
                survival_mode, market_regime,
            )
        return _COMBINED_STATE_MAP[fallback_key]
    # Last resort: both survival_mode and regime are unknown.
    if key not in _WARNED_UNMAPPED_KEYS:
        _WARNED_UNMAPPED_KEYS.add(key)
        logger.warning(
            "Completely unmapped (survival_mode=%r, market_regime=%r) -- "
            "returning ('unknown', 0.5)",
            survival_mode, market_regime,
        )
    return ("unknown", 0.5)


def compute_enriched_survival_timeline(
    daily_cache: pd.DataFrame,
    regime_labels: pd.Series | None = None,
    regime_confidence: pd.Series | None = None,
) -> EnrichedTimelineResult:
    """Compute the enriched survival timeline that bridges rule-based flags and HMM regimes.

    This function:
    1. Runs the base ``compute_survival_timeline()`` for rule-based mode classification.
    2. Overlays HMM regime labels (if provided) to create a combined state vector.
    3. Produces ``regime_state`` (categorical), ``survival_intensity`` (continuous 0-1),
       and ``regime_confidence`` (HMM posterior probability) columns.

    Parameters
    ----------
    daily_cache:
        Daily cache DataFrame with survival flag columns.
    regime_labels:
        Optional Series of market regime labels (e.g. "bull", "bear", "high_vol",
        "low_vol") aligned to the cache index. Typically from HMM/GMM output.
    regime_confidence:
        Optional Series of regime posterior probabilities (0-1) aligned to the
        cache index. Typically the max HMM posterior per day.

    Returns
    -------
    EnrichedTimelineResult
        Contains the enriched timeline DataFrame with combined state columns.
    """
    result = EnrichedTimelineResult()

    # Step 1: Run base survival timeline.
    base_result = compute_survival_timeline(daily_cache)
    result.base = base_result

    if not base_result.fitted:
        result.error = f"Base survival timeline failed: {base_result.error}"
        logger.warning(result.error)
        return result

    try:
        # Use the base timeline directly (which IS the original cache
        # after compute_survival_timeline now writes in-place).
        # We still copy here because the enriched timeline adds columns
        # that are specific to the enriched analysis, and we also write
        # the key columns back to the original daily_cache for downstream.
        timeline = base_result.timeline

        # Step 2: Merge market regime labels.
        if regime_labels is not None and not regime_labels.empty:
            # Align to timeline index, fill missing with "unknown".
            aligned_regimes = regime_labels.reindex(timeline.index).fillna("unknown")
            timeline["market_regime"] = aligned_regimes.astype(str)
            result.regime_available = True
        else:
            timeline["market_regime"] = "unknown"

        # Step 3: Compute combined state and survival intensity.
        survival_modes = timeline["survival_mode"]
        market_regimes = timeline["market_regime"]

        combined_states = []
        intensities = []
        for sm, mr in zip(survival_modes, market_regimes):
            state, intensity = _map_combined_state(str(sm), str(mr))
            combined_states.append(state)
            intensities.append(intensity)

        timeline["regime_state"] = combined_states
        timeline["survival_intensity"] = intensities

        # Step 4: Add regime confidence (HMM posterior or default).
        if regime_confidence is not None and not regime_confidence.empty:
            aligned_conf = regime_confidence.reindex(timeline.index).fillna(0.5)
            timeline["regime_confidence"] = aligned_conf
        else:
            # Default confidence: 1.0 if we have no HMM (rule-based is certain),
            # but reduce for "unknown" market regimes to signal uncertainty.
            timeline["regime_confidence"] = np.where(
                timeline["market_regime"] == "unknown", 0.5, 0.8
            )

        # Step 5: Compute regime transition probability estimate.
        # Simple empirical estimate: fraction of regime switches in a trailing window.
        _TRANSITION_WINDOW = 42  # ~2 months
        switch_flags = (
            timeline["regime_state"] != timeline["regime_state"].shift(1)
        ).astype(int)
        switch_flags.iloc[0] = 0
        timeline["regime_switch"] = switch_flags
        timeline["regime_transition_prob"] = (
            switch_flags
            .rolling(window=_TRANSITION_WINDOW, min_periods=1)
            .mean()
        )

        result.timeline = timeline

        # Summary statistics.
        state_counts = pd.Series(combined_states).value_counts(normalize=True)
        result.combined_state_distribution = {
            str(k): float(v) for k, v in state_counts.items()
        }
        result.mean_intensity = float(np.nanmean(intensities))
        result.fitted = True

        # Item 7: Log confidence range to help diagnose low-confidence situations.
        _conf = timeline["regime_confidence"]
        logger.info(
            "Enriched survival timeline: %d days, mean_intensity=%.3f, "
            "regime_available=%s, confidence=[min=%.3f, mean=%.3f, max=%.3f], "
            "states=%s",
            len(timeline),
            result.mean_intensity,
            result.regime_available,
            float(_conf.min()),
            float(_conf.mean()),
            float(_conf.max()),
            {k: f"{v:.1%}" for k, v in result.combined_state_distribution.items()
             if v > 0.01},
        )

    except Exception as exc:
        result.error = f"Enriched survival timeline failed: {exc}"
        logger.error(result.error)

    return result
