"""Stage 2: Frequency-First Pipeline -- each freq runs its own full pipeline.

Moves the multi-frequency pipeline from Stage 7.4 (after broken daily
ratios) to Stage 2 (before temporal models).  Each frequency (A/Q/S/M/W/D)
runs its own compute_derived_variables -> survival -> FH -> regime ->
forecast -> MC using frequency-aware formulas.

After all frequencies complete, the fusion step reconciles results and
forward-fills Q/A-computed ratios (PE, EV/EBITDA, ROA, etc.) into the
daily cache so downstream temporal models have correct values.

Sub-stages:
  2.0  Resample prep (build per-freq caches from raw filings)
  2.A  Annual pipeline
  2.Q  Quarterly pipeline
  2.M  Monthly pipeline
  2.W  Weekly pipeline
  2.D  Daily pipeline
  2.F  Fusion + forward-fill Q/A ratios to daily cache
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING

import numpy as np
import pandas as pd

if TYPE_CHECKING:
    from operator1.pipeline_state import PipelineState

logger = logging.getLogger("operator1.stages.stage2_freq")


# Ratios that are BROKEN on interpolated daily data (cross-type: market/flow,
# stock/flow) but CORRECT at native filing frequency (Q/A/S).
# After fusion, these are forward-filled from the Q or A pipeline result
# into the daily cache.
RATIOS_FROM_NATIVE_FREQ: tuple[str, ...] = (
    "pe_ratio_calc",
    "earnings_yield_calc",
    "ps_ratio_calc",
    "ev_to_ebitda",
    "fcf_yield",
    "roa",
    "roe",
    "eps_calc",
    "revenue_per_share",
    "net_debt_to_ebitda",
    "accruals",
    "accruals_signal",
    "dso",
    "dio",
    "dpo",
    "cash_conversion_cycle",
    "revenue_ttm_asof",
    "net_income_ttm_asof",
    "ebitda_ttm_asof",
    "revenue_growth_yoy",
    "earnings_growth_yoy",
    "gross_margin",
    "cash_burn_rate_monthly",
    "fh_runway_months",
)


def run_2_0_resample_prep(state: PipelineState) -> None:
    """2.0: Build per-frequency caches from raw filings.

    Reuses existing run_7_4_0_resample_prep() logic.
    """
    logger.info("Stage 2.0: Resample prep (frequency-first)")
    from operator1.stages.stage7_integration import run_7_4_0_resample_prep
    run_7_4_0_resample_prep(state)


def run_2_A_annual(state: PipelineState) -> None:
    """2.A: Run full pipeline at Annual frequency."""
    logger.info("Stage 2.A: Annual pipeline")
    from operator1.stages.stage7_integration import _run_7_4_single_freq
    _run_7_4_single_freq(state, "A")


def run_2_Q_quarterly(state: PipelineState) -> None:
    """2.Q: Run full pipeline at Quarterly (or Semi-Annual) frequency."""
    logger.info("Stage 2.Q: Quarterly pipeline")
    from operator1.stages.stage7_integration import _run_7_4_single_freq
    freqs = state.load_mf_frequencies()
    for f in freqs:
        if f in ("Q", "S"):
            _run_7_4_single_freq(state, f)


def run_2_M_monthly(state: PipelineState) -> None:
    """2.M: Run full pipeline at Monthly frequency."""
    logger.info("Stage 2.M: Monthly pipeline")
    from operator1.stages.stage7_integration import _run_7_4_single_freq
    _run_7_4_single_freq(state, "M")


def run_2_W_weekly(state: PipelineState) -> None:
    """2.W: Run full pipeline at Weekly frequency."""
    logger.info("Stage 2.W: Weekly pipeline")
    from operator1.stages.stage7_integration import _run_7_4_single_freq
    _run_7_4_single_freq(state, "W")


def run_2_D_daily(state: PipelineState) -> None:
    """2.D: Run full pipeline at Daily frequency."""
    logger.info("Stage 2.D: Daily pipeline")
    from operator1.stages.stage7_integration import _run_7_4_single_freq
    _run_7_4_single_freq(state, "D")


def run_2_F_fusion(state: PipelineState) -> None:
    """2.F: Fuse all frequency results + forward-fill Q/A ratios to daily.

    This is the critical step that makes the daily cache correct:
    1. Run existing 13-method frequency fusion.
    2. Load Q (or A fallback) pipeline cache.
    3. Forward-fill correct Q/A ratios into the daily cache.
    """
    logger.info("Stage 2.F: Frequency fusion + ratio forward-fill")

    # Step 1: Run existing fusion
    from operator1.stages.stage7_integration import run_7_4_6_fusion
    run_7_4_6_fusion(state)

    # Step 2: Forward-fill Q/A-computed ratios into the daily cache.
    # Prefer Q results (most granular filing frequency with native data).
    # Fall back to A if Q not available.
    cache = state.cache
    if cache is None or cache.empty:
        logger.warning("No daily cache to inject Q/A ratios into")
        return

    # Try Q first, then S, then A
    q_result = None
    for try_freq in ("Q", "S", "A"):
        q_result = state.load_mf_result(try_freq)
        if q_result is not None:
            logger.info("Using %s pipeline result for ratio forward-fill", try_freq)
            break

    if q_result is None:
        logger.warning("No Q/A pipeline result available for ratio forward-fill")
        return

    # Load the Q/A cache (has correct ratios at native filing scale)
    q_cache_obj = state.load_mf_cache(try_freq)
    if q_cache_obj is None or q_cache_obj.cache is None or q_cache_obj.cache.empty:
        logger.warning("Q/A cache empty, cannot forward-fill ratios")
        return

    q_cache = q_cache_obj.cache
    n_filled = 0

    for ratio_col in RATIOS_FROM_NATIVE_FREQ:
        if ratio_col not in q_cache.columns:
            continue
        q_series = q_cache[ratio_col].dropna()
        if q_series.empty:
            continue

        # Forward-fill Q/A values onto the daily index
        # This replaces the distorted daily-interpolated values with correct
        # filing-frequency values, forward-filled between filings.
        aligned = q_series.reindex(
            cache.index.union(q_series.index).sort_values()
        ).ffill().reindex(cache.index)

        if aligned.notna().any():
            cache[ratio_col] = aligned
            n_filled += 1

    state.cache = cache
    logger.info(
        "Forward-filled %d/%d Q/A ratios into daily cache",
        n_filled, len(RATIOS_FROM_NATIVE_FREQ),
    )


# Registry of all Stage 2 sub-stages in order
STAGE_2_FREQ_SUBSTAGES = [
    ("2.0", run_2_0_resample_prep),
    ("2.A", run_2_A_annual),
    ("2.Q", run_2_Q_quarterly),
    ("2.M", run_2_M_monthly),
    ("2.W", run_2_W_weekly),
    ("2.D", run_2_D_daily),
    ("2.F", run_2_F_fusion),
]
