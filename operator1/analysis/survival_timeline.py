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

        # Build enriched timeline
        timeline = daily_cache.copy()
        timeline["survival_mode"] = modes
        timeline["survival_mode_code"] = mode_codes
        timeline["switch_point"] = switch_flags
        timeline["days_in_mode"] = days_counter
        timeline["stability_score_21d"] = stability

        result.timeline = timeline

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
