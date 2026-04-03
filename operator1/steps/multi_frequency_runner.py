"""Multi-frequency sequential pipeline runner with cascading context.

Runs the full analytical pipeline at 5 frequencies (annual -> daily),
passing insights from slower frequencies to faster ones.

Execution order (slow to fast):
  1. Annual   (8yr, ~8 points)   -- secular trends, competitive moat
  2. Quarterly (6yr, ~24 points) -- earnings trajectory, fundamentals
  3. Monthly  (5yr, ~60 points)  -- macro cycles, sector rotation
  4. Weekly   (3yr, ~156 points) -- medium-term trends, earnings cycles
  5. Daily    (2yr, ~504 points) -- momentum, volatility, patterns

Each pipeline receives the prior pipeline's summary context (not time
series) to avoid look-ahead.  Context is limited to single-value
summaries: trend direction, secular regime label, latest survival
probability, and forecast bounds.
"""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass, field
from datetime import date
from pathlib import Path
from typing import Any

import pandas as pd

from operator1.features.frequency_resampler import (
    ResampledCache,
    build_cache_from_raw_filings,
    detect_all_filing_frequencies,
    detect_native_filing_frequency,
    get_frequencies_slow_to_fast,
    get_frequency_config,
    is_annual_only_market,
    resample_cache_to_frequency,
)

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Cascading context: what flows from slower to faster pipelines
# ---------------------------------------------------------------------------

@dataclass
class FrequencyContext:
    """Summary context passed from a slower frequency to the next faster one.

    Only single-value summaries are passed -- no time series, no per-period
    predictions, no regime label arrays.  This prevents look-ahead from
    slower-frequency full-history HMM fits.
    """

    frequency: str = ""
    trend_direction: str = "flat"         # "up", "down", "flat"
    secular_regime: str = "unknown"       # "bull", "bear", "sideways", "recovery"
    survival_probability_latest: float = 1.0
    survival_regime: str = "normal"
    forecast_bounds: dict[str, tuple[float, float]] = field(default_factory=dict)
    # Bounds format: {"close": (min, max), "revenue": (min, max), ...}
    confidence: float = 0.5              # overall confidence from this frequency


@dataclass
class FrequencyResult:
    """Result from one frequency pipeline run."""

    frequency: str
    label: str
    n_periods: int
    elapsed_seconds: float

    # Key outputs (summaries, not full objects -- those stay in pipeline state)
    regime_label: str = "unknown"
    survival_probability: float = 1.0
    survival_regime: str = "normal"
    trend_direction: str = "flat"
    forecast_summary: dict[str, Any] = field(default_factory=dict)
    walk_forward_mae: float | None = None

    # Context to pass to next frequency
    context_for_next: FrequencyContext = field(default_factory=FrequencyContext)


@dataclass
class MultiFrequencyResult:
    """Combined result from all frequency pipeline runs."""

    results: dict[str, FrequencyResult] = field(default_factory=dict)
    execution_order: list[str] = field(default_factory=list)
    total_elapsed_seconds: float = 0.0


# ---------------------------------------------------------------------------
# Single-frequency pipeline execution
# ---------------------------------------------------------------------------

def _extract_trend_direction(cache: pd.DataFrame) -> str:
    """Determine overall trend direction from a cache."""
    if cache.empty or "close" not in cache.columns:
        return "flat"
    closes = cache["close"].dropna()
    if len(closes) < 3:
        return "flat"
    first_third = closes.iloc[:len(closes) // 3].mean()
    last_third = closes.iloc[-len(closes) // 3:].mean()
    if last_third > first_third * 1.05:
        return "up"
    elif last_third < first_third * 0.95:
        return "down"
    return "flat"


def _extract_secular_regime(cache: pd.DataFrame) -> str:
    """Classify the secular regime from cache data."""
    if cache.empty or "close" not in cache.columns:
        return "unknown"
    closes = cache["close"].dropna()
    if len(closes) < 5:
        return "unknown"
    total_return = (closes.iloc[-1] / closes.iloc[0]) - 1 if closes.iloc[0] != 0 else 0
    # Check for drawdown
    peak = closes.cummax()
    drawdown = ((closes - peak) / peak).min()
    if total_return > 0.3 and drawdown > -0.2:
        return "bull"
    elif total_return < -0.2:
        return "bear"
    elif total_return > 0 and drawdown < -0.3:
        return "recovery"
    return "sideways"


def _extract_forecast_bounds(cache: pd.DataFrame) -> dict[str, tuple[float, float]]:
    """Extract historical min/max bounds for key variables.

    These are used to constrain faster-frequency predictions so they
    don't predict values outside the historical range seen at this
    slower frequency.
    """
    bounds = {}
    for col in ("close", "revenue", "total_assets", "net_income",
                "operating_cash_flow", "free_cash_flow"):
        if col in cache.columns:
            series = cache[col].dropna()
            if len(series) >= 3:
                bounds[col] = (float(series.min()), float(series.max()))
    return bounds


def run_single_frequency_pipeline(
    resampled: ResampledCache,
    prior_context: FrequencyContext | None = None,
    secrets: dict | None = None,
    market_id: str = "",
    ticker: str = "",
    skip_models: bool = False,
) -> FrequencyResult:
    """Run the analytical pipeline on a single-frequency cache.

    This runs derived variables, survival mode, regime detection,
    forecasting, and Monte Carlo on the resampled cache.

    Parameters
    ----------
    resampled:
        Resampled cache from ``resample_cache_to_frequency()``.
    prior_context:
        Context from the previous (slower) frequency pipeline.
        Used to constrain regime and survival detection.
    secrets:
        API keys dict.
    market_id:
        Market identifier for market-specific behavior.
    ticker:
        Company ticker for logging.
    skip_models:
        Skip temporal models (forecasting, MC, etc.).
    """
    start_time = time.time()
    cache = resampled.cache
    freq = resampled.frequency
    freq_label = resampled.label

    if cache.empty:
        return FrequencyResult(
            frequency=freq, label=freq_label,
            n_periods=0, elapsed_seconds=0,
        )

    logger.info(
        "--- %s Pipeline (%dyr, %d periods) ---",
        freq_label, resampled.lookback_years, resampled.n_periods,
    )

    # Step 1: Derived variables
    try:
        from operator1.features.derived_variables import compute_derived_variables
        cache = compute_derived_variables(cache)
        logger.info("[%s] Derived variables: %d columns", freq, len(cache.columns))
    except Exception as exc:
        logger.warning("[%s] Derived variables failed: %s", freq, exc)

    # Step 2: Survival mode
    survival_prob = 1.0
    survival_regime = "normal"
    try:
        from operator1.analysis.survival_mode import (
            compute_company_survival_flag,
            compute_survival_probability,
        )
        from operator1.analysis.hierarchy_weights import compute_hierarchy_weights
        cache["company_survival_mode_flag"] = compute_company_survival_flag(cache)
        cache["survival_probability"] = compute_survival_probability(cache)
        cache = compute_hierarchy_weights(cache)
        if "survival_probability" in cache.columns:
            sp = cache["survival_probability"].dropna()
            if len(sp) > 0:
                survival_prob = float(sp.iloc[-1])
        if "survival_regime" in cache.columns:
            sr = cache["survival_regime"].dropna()
            if len(sr) > 0:
                survival_regime = str(sr.iloc[-1])
        logger.info("[%s] Survival: prob=%.3f, regime=%s", freq, survival_prob, survival_regime)
    except Exception as exc:
        logger.warning("[%s] Survival mode failed: %s", freq, exc)

    # Step 3: Regime detection (skip for annual -- too few points)
    regime_label = "unknown"
    if resampled.n_periods >= 10 and not skip_models:
        try:
            from operator1.models.regime_detector import run_early_regime_detection
            target_var = "return_1d" if "return_1d" in cache.columns else None
            if target_var and cache[target_var].notna().sum() >= 10:
                cache, regime_result = run_early_regime_detection(
                    cache, target_variable=target_var,
                )
                if regime_result and regime_result.fitted:
                    if "regime_label" in cache.columns:
                        rl = cache["regime_label"].dropna()
                        if len(rl) > 0:
                            regime_label = str(rl.iloc[-1])
                    logger.info("[%s] Regime detection: current=%s", freq, regime_label)
        except Exception as exc:
            logger.debug("[%s] Regime detection skipped: %s", freq, exc)

    # Step 4: Forecasting (skip for annual and quarterly -- insufficient points for most models)
    forecast_summary = {}
    walk_forward_mae = None
    if resampled.n_periods >= 20 and not skip_models:
        try:
            from operator1.models.forecasting import run_forecasting
            cache, forecast_result = run_forecasting(cache)
            if forecast_result and forecast_result.forecasts:
                for var, horizons in forecast_result.forecasts.items():
                    if isinstance(horizons, dict):
                        for h, val in horizons.items():
                            forecast_summary[f"{var}_{h}"] = float(val) if val is not None else None
                logger.info("[%s] Forecasting: %d variables", freq, len(forecast_result.forecasts))
        except Exception as exc:
            logger.debug("[%s] Forecasting skipped: %s", freq, exc)

    # Step 4b: Walk-forward evaluation (need at least 30 periods)
    if resampled.n_periods >= 30 and not skip_models and forecast_summary:
        try:
            from operator1.models.walk_forward import run_walk_forward
            from operator1.analysis.survival_timeline import compute_survival_timeline
            _wf_tl = compute_survival_timeline(cache)
            _wf_modes = (
                _wf_tl.timeline["survival_mode"]
                if hasattr(_wf_tl, "timeline")
                and isinstance(_wf_tl.timeline, pd.DataFrame)
                and "survival_mode" in _wf_tl.timeline.columns
                else None
            )
            _wf_switches = (
                _wf_tl.timeline["switch_point"]
                if hasattr(_wf_tl, "timeline")
                and isinstance(_wf_tl.timeline, pd.DataFrame)
                and "switch_point" in _wf_tl.timeline.columns
                else None
            )
            _wf_result = run_walk_forward(cache, _wf_modes, _wf_switches)
            if _wf_result and _wf_result.fitted and not pd.isna(_wf_result.overall_mae):
                walk_forward_mae = _wf_result.overall_mae
                logger.info("[%s] Walk-forward MAE: %.6f", freq, walk_forward_mae)
        except Exception as exc:
            logger.debug("[%s] Walk-forward skipped: %s", freq, exc)

    # Step 5: Monte Carlo (need at least 15 periods)
    mc_survival = None
    if resampled.n_periods >= 15 and not skip_models:
        try:
            from operator1.models.monte_carlo import run_monte_carlo
            ret_col = "return_1d" if "return_1d" in cache.columns else None
            if ret_col and cache[ret_col].notna().sum() >= 10:
                mc_result = run_monte_carlo(cache, returns_col=ret_col)
                if mc_result and hasattr(mc_result, "survival_probability"):
                    mc_survival = mc_result.survival_probability
                    logger.info("[%s] Monte Carlo: survival=%.4f", freq, mc_survival or 0)
        except Exception as exc:
            logger.debug("[%s] Monte Carlo skipped: %s", freq, exc)

    # Log prior context disagreements (informational only)
    if prior_context is not None:
        _log_prior_context_disagreements(cache, prior_context, freq)

    # Extract summary for context passing
    trend = _extract_trend_direction(cache)
    secular = _extract_secular_regime(cache)
    bounds = _extract_forecast_bounds(cache)

    elapsed = time.time() - start_time

    # Build context for next frequency
    context_for_next = FrequencyContext(
        frequency=freq,
        trend_direction=trend,
        secular_regime=secular,
        survival_probability_latest=survival_prob,
        survival_regime=survival_regime,
        forecast_bounds=bounds,
        confidence=min(1.0, resampled.n_periods / 50),
    )

    result = FrequencyResult(
        frequency=freq,
        label=freq_label,
        n_periods=resampled.n_periods,
        elapsed_seconds=elapsed,
        regime_label=regime_label,
        survival_probability=survival_prob,
        survival_regime=survival_regime,
        trend_direction=trend,
        forecast_summary=forecast_summary,
        walk_forward_mae=walk_forward_mae,
        context_for_next=context_for_next,
    )

    logger.info(
        "[%s] Complete: %d periods, trend=%s, regime=%s, survival=%.3f (%.1fs)",
        freq, resampled.n_periods, trend, regime_label, survival_prob, elapsed,
    )

    return result


def _log_prior_context_disagreements(
    cache: pd.DataFrame,
    ctx: FrequencyContext,
    current_freq: str,
) -> None:
    """Log disagreements between slower-frequency context and current cache.

    Does NOT inject columns into the cache -- the FrequencyContext summary
    values are the cascading mechanism, not per-row cache columns.
    """
    if "company_survival_mode_flag" in cache.columns:
        current_survival = cache["company_survival_mode_flag"].iloc[-1] if len(cache) > 0 else 0
        if current_survival == 0 and ctx.survival_probability_latest < 0.5:
            logger.warning(
                "[%s] Frequency disagreement: %s says survival_prob=%.2f but "
                "%s shows no survival flag",
                current_freq, ctx.frequency,
                ctx.survival_probability_latest, current_freq,
            )


# ---------------------------------------------------------------------------
# Main sequential runner
# ---------------------------------------------------------------------------

def run_multi_frequency_pipeline(
    daily_cache: pd.DataFrame,
    secrets: dict | None = None,
    market_id: str = "",
    ticker: str = "",
    reference_date: date | None = None,
    skip_models: bool = False,
    frequencies: list[str] | None = None,
    income_df: pd.DataFrame | None = None,
    balance_df: pd.DataFrame | None = None,
    cashflow_df: pd.DataFrame | None = None,
    quotes_df: pd.DataFrame | None = None,
) -> MultiFrequencyResult:
    """Run the full pipeline sequentially across multiple frequencies.

    Execution order is always slow-to-fast (Annual -> Daily).  Each
    frequency pipeline receives context from the previous one.

    Parameters
    ----------
    daily_cache:
        Full daily cache (should have maximum available history).
    secrets:
        API keys dict.
    market_id:
        Market identifier.
    ticker:
        Company ticker.
    reference_date:
        Override "today" for backtesting.
    skip_models:
        Skip temporal models at all frequencies.
    frequencies:
        Override frequency list (default: all 5).
    income_df, balance_df, cashflow_df:
        Original wide-format statement DataFrames with ``report_date``.
        When provided, Q and A frequencies use these directly instead
        of resampling the interpolated daily cache.  This preserves
        actual reported values without interpolation artifacts.
    quotes_df:
        Original daily OHLCV DataFrame.  When provided alongside
        statement DataFrames, Q/A frequencies resample OHLCV from
        this source.
    """
    if frequencies is None:
        frequencies = get_frequencies_slow_to_fast()

    total_start = time.time()
    results: dict[str, FrequencyResult] = {}
    prior_context: FrequencyContext | None = None

    logger.info("=" * 60)
    logger.info("MULTI-FREQUENCY PIPELINE: %d frequencies", len(frequencies))
    logger.info("=" * 60)

    # Check if raw statement data is available for Q/A direct construction
    _has_raw_statements = any(
        df is not None and not df.empty
        for df in [income_df, balance_df, cashflow_df]
    )

    _is_annual_only = is_annual_only_market(market_id)

    # Detect ALL filing frequencies present in the raw data.
    # If both Q and S exist (e.g. JSE Sasol: quarterly metrics + semi-annual results),
    # run BOTH Q and S pipelines. If only S exists, replace Q with S.
    if _has_raw_statements and "Q" in frequencies:
        _all_freqs = detect_all_filing_frequencies(income_df, balance_df, cashflow_df)
        _native_freq = detect_native_filing_frequency(income_df, balance_df, cashflow_df)

        if "Q" in _all_freqs and "S" in _all_freqs:
            # Both quarterly and semi-annual filings exist.
            # Keep Q in the list AND add S (run both pipelines).
            # S goes BEFORE Q in slow-to-fast order (6-month > 3-month).
            if "S" not in frequencies:
                q_idx = frequencies.index("Q")
                frequencies.insert(q_idx, "S")  # insert S before Q
            logger.info(
                "Both Q and S filings detected -- running both pipelines (order: A,S,Q,M,W,D)"
            )
        elif _native_freq == "S" and "Q" not in _all_freqs:
            # Only semi-annual filings exist, no quarterly.
            # Replace Q with S in the frequency list.
            frequencies = [("S" if f == "Q" else f) for f in frequencies]
            logger.info(
                "Auto-switch: Q -> S (semi-annual filings only, median gap 120-250d)"
            )
        elif _native_freq == "A" and not _is_annual_only:
            # Data is annual but market is not in the annual-only registry.
            # This can happen when a company only files annually even though
            # other companies in the same market file quarterly.
            logger.info(
                "Filing data is annual for this company (but market %s is not annual-only). "
                "Q frequency will interpolate annual -> quarterly.",
                market_id,
            )

    # For ch_six at Q frequency: use Kalman-smoothed quarterly synthetic
    # financials instead of generic annual-to-quarterly interpolation.
    # The SIX proxy module produces higher-quality quarterly estimates
    # by leveraging 18 years of dividend history through a state-space model.
    _six_q_income = None
    _six_q_balance = None
    _six_q_cashflow = None
    if market_id == "ch_six" and _has_raw_statements:
        try:
            from operator1.features.six_derived_proxies import generate_synthetic_financials
            # Get the profile from the raw statement metadata
            _six_profile_cache = Path("cache/ch_six")
            _six_ticker = ticker or ""
            _six_profile = {}
            _profile_path = _six_profile_cache / _six_ticker.upper() / "profile.json"
            if _profile_path.exists():
                import json as _json
                _six_profile = _json.loads(_profile_path.read_text(encoding="utf-8"))
            if _six_profile:
                _six_q = generate_synthetic_financials(_six_profile, target_frequency="Q")
                _six_q_income = _six_q.get("income")
                _six_q_balance = _six_q.get("balance")
                _six_q_cashflow = _six_q.get("cashflow")
                _n_q = sum(
                    len(df) for df in [_six_q_income, _six_q_balance, _six_q_cashflow]
                    if df is not None and not df.empty
                )
                if _n_q > 0:
                    logger.info(
                        "[Q] SIX Kalman-smoothed quarterly financials: %d records",
                        _n_q,
                    )
        except Exception as _exc:
            logger.debug("SIX quarterly synthetic generation failed: %s", _exc)

    for freq in frequencies:
        # Data source selection per frequency:
        #   Q/A: raw filing data as-is (no interpolation artifacts)
        #   Q (ch_six): Kalman-smoothed quarterly from dividend data
        #   Q (other annual-only markets): interpolate annual filings to quarterly
        #   W/M: raw filing data with native frequency interpolation
        #        (stock=linear, flow=distribute to W/M periods)
        #   D:   use the daily cache directly (already interpolated)

        # Special case: SIX Q frequency uses Kalman-smoothed quarterly data
        if freq == "Q" and market_id == "ch_six" and _six_q_income is not None:
            resampled = build_cache_from_raw_filings(
                income_df=_six_q_income,
                balance_df=_six_q_balance,
                cashflow_df=_six_q_cashflow,
                quotes_df=quotes_df,
                frequency="Q",
                reference_date=reference_date,
            )
            logger.info(
                "[Q] Using SIX Kalman-smoothed quarterly: %d periods",
                resampled.n_periods,
            )
        elif freq in ("Q", "A", "W", "M", "S") and _has_raw_statements:
            # For annual-only markets at Q frequency: interpolate annual -> quarterly
            # instead of using raw Q data (which doesn't exist).
            # build_cache_from_raw_filings with freq="Q" + annual data will
            # produce interpolated quarterly values via the frequency interpolator.
            if freq == "Q" and _is_annual_only:
                logger.info(
                    "[Q] Annual-only market (%s) -- interpolating annual filings to quarterly",
                    market_id,
                )
            resampled = build_cache_from_raw_filings(
                income_df=income_df,
                balance_df=balance_df,
                cashflow_df=cashflow_df,
                quotes_df=quotes_df,
                frequency=freq,
                reference_date=reference_date,
            )
            if freq == "Q" and _is_annual_only:
                _method = "annual-to-quarterly interpolation"
            elif freq in ("Q", "A", "S"):
                _method = "raw filings"
            else:
                _method = "native interpolation"
            logger.info(
                "[%s] Using %s: %d periods",
                freq, _method, resampled.n_periods,
            )
        else:
            resampled = resample_cache_to_frequency(
                daily_cache, frequency=freq, reference_date=reference_date,
            )

        if resampled.n_periods < 3:
            logger.info(
                "[%s] Skipping -- only %d periods (need at least 3)",
                freq, resampled.n_periods,
            )
            continue

        # Run pipeline at this frequency
        result = run_single_frequency_pipeline(
            resampled=resampled,
            prior_context=prior_context,
            secrets=secrets,
            market_id=market_id,
            ticker=ticker,
            skip_models=skip_models,
        )

        results[freq] = result

        # Pass context to next (faster) frequency
        prior_context = result.context_for_next

    total_elapsed = time.time() - total_start

    logger.info("=" * 60)
    logger.info(
        "MULTI-FREQUENCY COMPLETE: %d/%d frequencies in %.1fs",
        len(results), len(frequencies), total_elapsed,
    )
    for freq, result in results.items():
        logger.info(
            "  %s: %d periods, trend=%s, survival=%.3f (%.1fs)",
            result.label, result.n_periods, result.trend_direction,
            result.survival_probability, result.elapsed_seconds,
        )
    logger.info("=" * 60)

    return MultiFrequencyResult(
        results=results,
        execution_order=list(results.keys()),
        total_elapsed_seconds=total_elapsed,
    )
