"""Stage 2: Frequency-First Pipeline -- each freq runs its own full pipeline.

Moves the multi-frequency pipeline from Stage 7.4 (after broken daily
ratios) to Stage 2 (before temporal models).  Each frequency (A/Q/S/M/W/D)
runs its own compute_derived_variables -> survival -> FH -> regime ->
forecast -> MC using frequency-aware formulas.

After all frequencies complete, the fusion step reconciles results and
forward-fills Q/A-computed ratios (PE, EV/EBITDA, ROA, etc.) into the
daily cache so downstream temporal models have correct values.

Sub-stages (2-wave parallel):
  2.0   Resample prep (build per-freq caches from raw filings)
  2.W1  Wave 1: A + Q + S in parallel (native filing frequencies, no inter-deps)
  2.W2  Wave 2: M + W + D in parallel (interpolated, use Q context from Wave 1)
  2.F   Fusion + forward-fill Q/A ratios to daily cache
"""

from __future__ import annotations

import logging
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
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


# ---------------------------------------------------------------------------
# Internal: run one frequency in an isolated thread
# ---------------------------------------------------------------------------

def _run_freq_isolated(state: "PipelineState", freq: str, prior_context_freq: str | None) -> str:
    """Run a single frequency pipeline in an isolated thread.

    Each thread loads its own resampled cache from disk, runs the full
    per-frequency pipeline, and saves results back to disk via unique
    file paths (no shared mutable state between threads).

    Args:
        state: Shared PipelineState (only disk I/O methods used, which
               write to frequency-specific file paths -- no contention).
        freq: Frequency label ("A", "Q", "S", "M", "W", "D").
        prior_context_freq: Frequency to load cascading context from,
                            or None for Wave 1 (no prior context).

    Returns:
        freq label on success.

    Raises:
        Exception on pipeline failure (caught by caller).
    """
    from operator1.steps.multi_frequency_runner import run_single_frequency_pipeline
    from operator1.stages.stage7_integration import _get_mf_secrets

    resampled = state.load_mf_cache(freq)
    if resampled is None:
        logger.info("[%s] No resampled cache found -- skipping", freq)
        return freq

    # Load cascading context from a prior frequency (if specified)
    prior_context = None
    if prior_context_freq is not None:
        prior_context = state.load_mf_context(prior_context_freq)

    secrets = _get_mf_secrets()

    result = run_single_frequency_pipeline(
        resampled=resampled,
        prior_context=prior_context,
        secrets=secrets,
        market_id=state.market_id,
        ticker=state.company,
        skip_models=False,
    )

    # Save to disk (unique file paths per freq -- no contention)
    state.save_mf_result(freq, result)
    state.save_mf_context(freq, result.context_for_next)
    logger.info(
        "[%s] Pipeline complete: %d periods, survival=%.3f (%.1fs)",
        freq, result.n_periods, result.survival_probability, result.elapsed_seconds,
    )
    return freq


# ---------------------------------------------------------------------------
# Wave 1: Native filing frequencies in parallel (A + Q + S)
# ---------------------------------------------------------------------------

def run_2_wave1_native(state: PipelineState) -> None:
    """2.W1: Run A/Q/S pipelines in parallel (native filing frequencies).

    Wave 1 frequencies are native filing frequencies -- they use raw
    statement DataFrames directly, not interpolated daily data.  There
    are NO inter-dependencies between A, Q, and S: each uses its own
    filing data independently.

    All three run in parallel threads.  Each thread reads its own
    resampled cache from disk, runs the full per-frequency pipeline,
    and saves results to frequency-specific file paths.
    """
    logger.info("Stage 2.W1: Wave 1 -- native freq pipelines in parallel")

    freqs = state.load_mf_frequencies()
    native_freqs = [f for f in freqs if f in ("A", "Q", "S")]

    if not native_freqs:
        logger.info("No native filing frequencies to run in Wave 1")
        return

    t0 = time.time()

    with ThreadPoolExecutor(max_workers=len(native_freqs)) as pool:
        futures = {
            pool.submit(_run_freq_isolated, state, freq, None): freq
            for freq in native_freqs
        }

        for future in as_completed(futures):
            freq = futures[future]
            try:
                future.result(timeout=300)
            except Exception as exc:
                logger.warning("[%s] Wave 1 pipeline failed: %s", freq, exc)

    elapsed = time.time() - t0
    logger.info(
        "Wave 1 complete: %d native frequencies in %.1fs (parallel)",
        len(native_freqs), elapsed,
    )


# ---------------------------------------------------------------------------
# Wave 2: Interpolated frequencies in parallel (M + W + D)
# ---------------------------------------------------------------------------

def run_2_wave2_interpolated(state: PipelineState) -> None:
    """2.W2: Run M/W/D pipelines in parallel (interpolated frequencies).

    Wave 2 frequencies are derived from interpolation -- they don't have
    native filing data at their frequency.  They CAN benefit from
    cascading context from Wave 1 (A/Q) but NOT from each other.

    All three use the Q pipeline context as their prior (the most
    relevant native frequency).  If Q is unavailable, falls back to A.
    This avoids cross-dependency within Wave 2.
    """
    logger.info("Stage 2.W2: Wave 2 -- interpolated freq pipelines in parallel")

    freqs = state.load_mf_frequencies()
    interp_freqs = [f for f in freqs if f in ("M", "W", "D")]

    if not interp_freqs:
        logger.info("No interpolated frequencies to run in Wave 2")
        return

    # Determine best prior context: prefer Q, fall back to S, then A
    prior_context_freq = None
    for candidate in ("Q", "S", "A"):
        if state.load_mf_context(candidate) is not None:
            prior_context_freq = candidate
            break

    if prior_context_freq:
        logger.info("Wave 2 using %s context as prior for all interpolated freqs", prior_context_freq)

    t0 = time.time()

    with ThreadPoolExecutor(max_workers=len(interp_freqs)) as pool:
        futures = {
            pool.submit(_run_freq_isolated, state, freq, prior_context_freq): freq
            for freq in interp_freqs
        }

        for future in as_completed(futures):
            freq = futures[future]
            try:
                future.result(timeout=180)
            except Exception as exc:
                logger.warning("[%s] Wave 2 pipeline failed: %s", freq, exc)

    elapsed = time.time() - t0
    logger.info(
        "Wave 2 complete: %d interpolated frequencies in %.1fs (parallel)",
        len(interp_freqs), elapsed,
    )


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

    # Step 3: Re-run survival on daily cache with correct Q/A ratios.
    # After forward-fill, fcf_yield, revenue_growth_yoy, etc. are now
    # correct on the daily cache.  Re-run survival with freq="Q" trigger
    # set so the daily survival_probability reflects all triggers including
    # the now-correct flow-based ones (fcf_yield, revenue decline, Altman Z).
    if n_filled > 0:
        try:
            from operator1.analysis.survival_mode import (
                compute_company_survival_flag,
                compute_survival_probability,
            )
            from operator1.analysis.hierarchy_weights import compute_hierarchy_weights

            # Use "Q" trigger set since ratios are now at Q-quality
            cache["company_survival_mode_flag"] = compute_company_survival_flag(
                cache, freq="Q",
                sector=state.target_profile.get("sector", ""),
            )
            cache["survival_probability"] = compute_survival_probability(cache)
            cache = compute_hierarchy_weights(cache)
            state.cache = cache
            logger.info(
                "Post-fusion survival re-run (Q triggers): prob=%.3f, regime=%s",
                float(cache["survival_probability"].dropna().iloc[-1])
                if cache["survival_probability"].notna().any() else 0.0,
                str(cache["survival_regime"].dropna().iloc[-1])
                if "survival_regime" in cache.columns and cache["survival_regime"].notna().any()
                else "unknown",
            )
            # Also re-run financial health with correct Q/A ratios
            try:
                from operator1.models.financial_health import compute_financial_health
                cache, _fh = compute_financial_health(cache, freq="Q")
                state.fh_result = _fh
                state.cache = cache
                logger.info("Post-fusion FH re-run: composite=%.1f (%s)",
                            _fh.latest_composite, _fh.latest_label)
            except Exception as fh_exc:
                logger.debug("Post-fusion FH re-run failed: %s", fh_exc)
        except Exception as exc:
            logger.warning("Post-fusion survival re-run failed: %s", exc)


# Registry of all Stage 2 sub-stages in order (2-wave parallel)
STAGE_2_FREQ_SUBSTAGES = [
    ("2.0", run_2_0_resample_prep),
    ("2.W1", run_2_wave1_native),         # A + Q + S in parallel
    ("2.W2", run_2_wave2_interpolated),    # M + W + D in parallel
    ("2.F", run_2_F_fusion),
]
