"""Product segment metrics for the daily cache.

Converts segment revenue data from ``extract_segment_data()`` into daily
cache columns consumed by temporal models, Monte Carlo concentration risk,
and the prediction aggregator.

Columns produced (10):
    segment_hhi           -- Herfindahl-Hirschman Index (0-1)
    segment_count         -- number of reporting segments
    dominant_segment_growth -- YoY growth of largest segment (NaN if single snapshot)
    estimated_market_share -- dominant segment revenue / sector revenue (if available)
    cannibalization_rate  -- overlap between new and old segment revenue (0 if single)
    net_new_revenue_pct   -- fraction from segments not in prior period
    network_effect_score  -- 0-1 proxy: accelerating dominant segment growth
    input_cost_pressure   -- COGS/revenue trend from cache
    growth_runway_quarters -- estimated quarters of above-avg growth remaining
    maturity_concentration -- fraction of revenue from decelerating segments

All columns are constant across the daily index (segment data is periodic,
not daily).  If computation fails, columns are set to NaN and the pipeline
continues unaffected.
"""

from __future__ import annotations

import logging
from typing import Any

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

EPSILON = 1e-9


def compute_product_metrics(
    cache: pd.DataFrame,
    segment_data: dict[str, Any],
    *,
    prior_segment_data: dict[str, Any] | None = None,
) -> pd.DataFrame:
    """Inject product segment metrics into the daily cache.

    Parameters
    ----------
    cache:
        Daily cache DataFrame (DatetimeIndex).
    segment_data:
        Dict from ``extract_segment_data()`` with keys:
        ``segments`` (name -> revenue float), ``descriptions``,
        ``n_segments``, ``has_revenue``, ``has_descriptions``.
    prior_segment_data:
        Optional prior-period segment data for YoY and cannibalization
        computation.  If None, YoY-dependent columns are NaN.

    Returns
    -------
    cache with 10 new columns added.
    """
    if cache.empty:
        return cache

    segments: dict[str, float] = segment_data.get("segments", {})
    n_segments = segment_data.get("n_segments", len(segments))

    # --- HHI (Herfindahl-Hirschman Index) ---
    hhi = _compute_hhi(segments)

    # --- Segment count ---
    seg_count = max(n_segments, len(segments))

    # --- Dominant segment ---
    dominant_name = ""
    dominant_share = 0.0
    total_rev = sum(abs(v) for v in segments.values()) if segments else 0.0
    if segments and total_rev > EPSILON:
        dominant_name = max(segments, key=lambda k: abs(segments[k]))
        dominant_share = abs(segments[dominant_name]) / total_rev

    # --- YoY growth of dominant segment ---
    dominant_growth = np.nan
    if prior_segment_data and dominant_name:
        prior_segs = prior_segment_data.get("segments", {})
        prior_val = prior_segs.get(dominant_name, 0.0)
        current_val = segments.get(dominant_name, 0.0)
        if abs(prior_val) > EPSILON:
            dominant_growth = (current_val - prior_val) / abs(prior_val)

    # --- Estimated market share (proxy: dominant share as fraction) ---
    # True market share requires sector total revenue from linked_agg.
    # Fallback: use dominant_share as a crude upper-bound proxy.
    estimated_market_share = dominant_share

    # --- Cannibalization rate ---
    cannibalization = 0.0
    if prior_segment_data:
        cannibalization = _compute_cannibalization(
            segments, prior_segment_data.get("segments", {}),
        )

    # --- Net new revenue percentage ---
    net_new_pct = 0.0
    if prior_segment_data:
        net_new_pct = _compute_net_new_revenue_pct(
            segments, prior_segment_data.get("segments", {}),
        )

    # --- Network effect score ---
    # Proxy: if dominant segment has > 50% share AND positive growth,
    # assume network effects.  Stronger with higher share + faster growth.
    network_score = 0.0
    if dominant_share > 0.5 and not np.isnan(dominant_growth) and dominant_growth > 0:
        # Scale: share contribution (0-0.5) + growth contribution (0-0.5)
        share_component = min(0.5, (dominant_share - 0.5) * 2.0)
        growth_component = min(0.5, dominant_growth * 2.5)
        network_score = share_component + growth_component
    network_score = float(np.clip(network_score, 0.0, 1.0))

    # --- Input cost pressure (from cache COGS/revenue trend) ---
    input_cost = _compute_input_cost_pressure(cache)

    # --- Growth runway quarters ---
    # Proxy: if dominant segment growth is above median, estimate how many
    # quarters at current deceleration rate it takes to converge.
    growth_runway = np.nan
    if not np.isnan(dominant_growth) and dominant_growth > 0.05:
        # Rough: runway = growth / assumed deceleration (5% per quarter)
        decel_rate = 0.05
        growth_runway = max(1.0, dominant_growth / decel_rate)
        growth_runway = min(growth_runway, 20.0)  # cap at 5 years

    # --- Maturity concentration ---
    # Fraction of revenue from segments with negative or zero growth.
    # Requires prior data; NaN if unavailable.
    maturity_conc = np.nan
    if prior_segment_data and segments and total_rev > EPSILON:
        prior_segs = prior_segment_data.get("segments", {})
        mature_rev = 0.0
        for name, rev in segments.items():
            prior_val = prior_segs.get(name, 0.0)
            if abs(prior_val) > EPSILON:
                growth = (rev - prior_val) / abs(prior_val)
                if growth <= 0:
                    mature_rev += abs(rev)
            else:
                # New segment -- not mature
                pass
        maturity_conc = mature_rev / total_rev

    # --- Inject into cache as constant daily columns ---
    cache["segment_hhi"] = hhi
    cache["segment_count"] = seg_count
    cache["dominant_segment_growth"] = dominant_growth
    cache["estimated_market_share"] = estimated_market_share
    cache["cannibalization_rate"] = cannibalization
    cache["net_new_revenue_pct"] = net_new_pct
    cache["network_effect_score"] = network_score
    cache["input_cost_pressure"] = input_cost
    cache["growth_runway_quarters"] = growth_runway
    cache["maturity_concentration"] = maturity_conc

    n_valid = sum(
        1 for c in [
            "segment_hhi", "dominant_segment_growth", "estimated_market_share",
            "cannibalization_rate", "net_new_revenue_pct", "network_effect_score",
            "input_cost_pressure", "growth_runway_quarters", "maturity_concentration",
        ]
        if c in cache.columns and cache[c].notna().any()
    )
    logger.info(
        "Product metrics: HHI=%.3f, segments=%d, dominant_share=%.1f%%, "
        "%d columns with data",
        hhi, seg_count, dominant_share * 100, n_valid,
    )

    return cache


# ---------------------------------------------------------------------------
# Helper functions
# ---------------------------------------------------------------------------

def _compute_hhi(segments: dict[str, float]) -> float:
    """Compute Herfindahl-Hirschman Index from segment revenue dict.

    HHI = sum of squared revenue shares.
    Range: 1/N (perfectly diversified) to 1.0 (single segment).
    """
    if not segments:
        return 1.0  # No data -> assume concentrated
    total = sum(abs(v) for v in segments.values())
    if total < EPSILON:
        return 1.0
    shares = [abs(v) / total for v in segments.values()]
    return float(sum(s ** 2 for s in shares))


def _compute_cannibalization(
    current: dict[str, float],
    prior: dict[str, float],
) -> float:
    """Compute cannibalization rate between two periods.

    Measures overlap: segments that grew at the expense of others.
    Returns 0-1 where 1 = all growth came from cannibalizing other segments.
    """
    if not current or not prior:
        return 0.0

    total_current = sum(abs(v) for v in current.values())
    if total_current < EPSILON:
        return 0.0

    # Compare share shifts: segments that gained share while others lost
    current_total = sum(abs(v) for v in current.values())
    prior_total = sum(abs(v) for v in prior.values())
    if current_total < EPSILON or prior_total < EPSILON:
        return 0.0

    gained = 0.0
    lost = 0.0
    for name in set(current) | set(prior):
        cur_share = abs(current.get(name, 0.0)) / current_total
        pri_share = abs(prior.get(name, 0.0)) / prior_total
        delta = cur_share - pri_share
        if delta > 0:
            gained += delta
        else:
            lost += abs(delta)

    # Cannibalization = min(gained, lost) / max(gained, lost)
    # High when gains and losses are symmetric (zero-sum reshuffling)
    if max(gained, lost) < EPSILON:
        return 0.0
    return float(min(gained, lost) / max(gained, lost))


def _compute_net_new_revenue_pct(
    current: dict[str, float],
    prior: dict[str, float],
) -> float:
    """Fraction of current revenue from segments not in prior period."""
    if not current:
        return 0.0

    total_current = sum(abs(v) for v in current.values())
    if total_current < EPSILON:
        return 0.0

    new_rev = sum(
        abs(v) for name, v in current.items()
        if name not in prior
    )
    return float(new_rev / total_current)


def _compute_input_cost_pressure(cache: pd.DataFrame) -> float:
    """Compute input cost pressure from COGS/revenue trend in cache.

    Rising COGS/revenue ratio = increasing input cost pressure.
    Returns the slope of the ratio over the last 252 days.
    """
    if "cost_of_revenue" not in cache.columns or "revenue" not in cache.columns:
        return np.nan

    cogs = cache["cost_of_revenue"].astype(float)
    rev = cache["revenue"].astype(float)

    # Safe ratio
    safe_rev = rev.where(rev.abs() > EPSILON)
    ratio = cogs / safe_rev

    # Drop NaN and compute slope over last 252 days
    valid = ratio.dropna().tail(252)
    if len(valid) < 20:
        return np.nan

    try:
        x = np.arange(len(valid), dtype=float)
        y = valid.values.astype(float)
        # Simple OLS slope
        x_mean = x.mean()
        y_mean = y.mean()
        slope = float(np.sum((x - x_mean) * (y - y_mean)) / max(np.sum((x - x_mean) ** 2), EPSILON))
        # Annualize: slope * 252
        return float(slope * 252)
    except Exception:
        return np.nan
