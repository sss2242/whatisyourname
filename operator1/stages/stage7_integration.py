"""Stage 7: Integration -- USS, retroactive calibration, diagnostics, multi-freq, hedge fund.

Sub-stages:
  7.1  USS forecast bounding + scenario engine
  7.2  Retroactive calibration
  7.3  Model diagnostics
  7.4  Multi-frequency pipeline + fusion
  7.5  Hedge fund analysis
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from operator1.pipeline_state import PipelineState

logger = logging.getLogger("operator1.stages.stage7")


def run_7_1_uss(state: PipelineState) -> None:
    """7.1: USS forecast bounding + scenario engine."""
    logger.info("Sub-stage 7.1: USS integration")
    cache = state.cache

    if state.survival_controller is None or not getattr(state.survival_controller, "is_survival", False):
        # Check early warning even if not in survival
        if state.survival_controller is not None and hasattr(state.survival_controller, "is_approaching_survival"):
            if state.survival_controller.is_approaching_survival():
                logger.warning(
                    "USS early warning: score=%.2f -- approaching survival triggers",
                    state.survival_controller.get_early_warning_latest(),
                )
        return

    # Forecast bounding
    if state.forecast_result is not None:
        try:
            from operator1.analysis.survival_regime_controller import bound_forecast_dict
            if hasattr(state.forecast_result, "forecasts") and state.forecast_result.forecasts:
                state.forecast_result.forecasts = bound_forecast_dict(
                    state.forecast_result.forecasts, cache,
                    state.survival_controller.current_regime,
                )
                logger.info("USS forecast bounding applied (regime=%s)",
                            state.survival_controller.current_regime)
        except Exception as exc:
            logger.debug("USS forecast bounding failed: %s", exc)

    # Scenario engine
    try:
        from operator1.analysis.scenario_engine import run_scenario_engine
        state.scenario_result = run_scenario_engine(
            cache,
            regime=state.survival_controller.current_regime,
            n_paths=state.survival_controller.model_config.mc_n_paths,
        )
        if state.scenario_result and state.scenario_result.available:
            logger.info(
                "Scenario engine: orderly=%.1f%% / muddle=%.1f%% / catastrophic=%.1f%%",
                state.scenario_result.orderly.survival_prob_252d * 100,
                state.scenario_result.muddle_through.survival_prob_252d * 100,
                state.scenario_result.catastrophic.survival_prob_252d * 100,
            )
    except Exception as exc:
        logger.warning("Scenario engine failed: %s", exc)

    # Reverse stress test (Basel III): minimum shock to trigger survival
    try:
        from operator1.analysis.scenario_engine import compute_reverse_stress_test
        _reverse = compute_reverse_stress_test(cache)
        if _reverse.available:
            # Attach to scenario_result for profile injection
            if state.scenario_result is not None:
                state.scenario_result.reverse_stress = _reverse
            logger.info(
                "Reverse stress: revenue=%.1f%%, margin=%.1fpp, rates=+%.0fbps -> %s",
                _reverse.revenue_shock_pct, _reverse.margin_shock_pp,
                _reverse.rate_shock_bps, _reverse.triggered_variable,
            )
    except Exception as exc:
        logger.debug("Reverse stress test skipped: %s", exc)


def run_7_2_retro_calibration(state: PipelineState) -> None:
    """7.2: Retroactive calibration (empirical Bayes: first-pass data -> second-pass priors)."""
    logger.info("Sub-stage 7.2: Retroactive calibration")
    try:
        from operator1.analysis.retroactive_calibration import run_retroactive_calibration

        _entity_groups = {}
        if state.relationships:
            for grp, ents in state.relationships.items():
                if isinstance(ents, list):
                    ids = []
                    for e in ents:
                        eid = ""
                        if isinstance(e, dict):
                            eid = e.get("isin", "") or e.get("ticker", "")
                        elif hasattr(e, "isin"):
                            eid = e.isin or getattr(e, "ticker", "")
                        if eid:
                            ids.append(eid)
                    _entity_groups[grp] = ids

        state.retro_params = run_retroactive_calibration(
            cache=state.cache,
            linked_caches=state.linked_caches if state.linked_caches else None,
            entity_groups=_entity_groups if _entity_groups else None,
            walk_forward_result=state.walk_forward_result,
            forecast_result=state.forecast_result,
            sobol_result=state.sobol_result,
            target_profile=state.target_profile,
        )
        if state.retro_params.n_calibrated > 0:
            logger.info("Retroactive calibration: %d groups calibrated",
                        state.retro_params.n_calibrated)
    except Exception as exc:
        logger.warning("Retroactive calibration failed: %s", exc)


def run_7_3_diagnostics(state: PipelineState) -> None:
    """7.3: Model diagnostics (expected path vs actual path)."""
    logger.info("Sub-stage 7.3: Model diagnostics")
    try:
        from operator1.monitoring.model_diagnostics import compute_model_diagnostics
        state.model_diagnostics_result = compute_model_diagnostics(
            state.cache,
            forecast_result=state.forecast_result,
            mc_result=state.mc_result,
            copula_result=state.copula_result,
            granger_result=state.granger_result,
            cycle_result=state.cycle_result,
            dtw_result=state.dtw_result,
            conformal_result=state.conformal_result,
        )
        if state.model_diagnostics_result and state.model_diagnostics_result.available:
            logger.info(
                "Model diagnostics: %d/%d on track",
                state.model_diagnostics_result.n_models_on_track,
                state.model_diagnostics_result.n_models_assessed,
            )
    except Exception as exc:
        logger.debug("Model diagnostics failed: %s", exc)


def run_7_4_multi_frequency(state: PipelineState) -> None:
    """7.4: Multi-frequency pipeline -- runs all sub-stages sequentially (backward compat)."""
    run_7_4_0_resample_prep(state)
    freqs = state.load_mf_frequencies()
    for freq in freqs:
        _run_7_4_single_freq(state, freq)
    run_7_4_6_fusion(state)


def _get_mf_ref_date(state: PipelineState):
    """Extract reference date for multi-frequency pipeline."""
    from datetime import datetime as _dt
    return _dt.strptime(state.end_date, "%Y-%m-%d").date() if state.end_date else None


def _get_mf_secrets() -> dict:
    """Load secrets for multi-frequency pipeline."""
    try:
        from operator1.secrets_loader import load_secrets
        return load_secrets()
    except Exception:
        return {}


def run_7_4_0_resample_prep(state: PipelineState) -> None:
    """7.4.0: Build all ResampledCache objects and save to disk."""
    logger.info("Sub-stage 7.4.0: Multi-frequency resample prep")

    from operator1.features.frequency_resampler import (
        ResampledCache,
        build_cache_from_raw_filings,
        detect_all_filing_frequencies,
        detect_native_filing_frequency,
        get_frequencies_slow_to_fast,
        is_annual_only_market,
        resample_cache_to_frequency,
    )

    cache = state.cache
    ref_date = _get_mf_ref_date(state)
    market_id = state.market_id

    income_df = state.income_df if not state.income_df.empty else None
    balance_df = state.balance_df if not state.balance_df.empty else None
    cashflow_df = state.cashflow_df if not state.cashflow_df.empty else None
    quotes_df = state.quotes_df if not state.quotes_df.empty else None

    _has_raw = any(df is not None and not df.empty for df in [income_df, balance_df, cashflow_df])
    _is_annual_only = is_annual_only_market(market_id)

    # Per-frequency statement groups from frequency separator (Step 3d).
    # These contain the ACTUAL filing data per frequency (quarterly, annual,
    # semiannual) before Chow-Lin reconciliation merged them together.
    _has_freq_groups = any(
        bool(getattr(state, grp, {}))
        for grp in ("income_freq_groups", "balance_freq_groups", "cashflow_freq_groups")
    )
    if _has_freq_groups:
        logger.info(
            "Per-frequency filing groups available: income=%s, balance=%s, cashflow=%s",
            list(state.income_freq_groups.keys()),
            list(state.balance_freq_groups.keys()),
            list(state.cashflow_freq_groups.keys()),
        )

    if not _has_raw and not _has_freq_groups:
        logger.warning(
            "MULTI-FREQUENCY DEGRADED: No raw financial statement data available "
            "(income_df, balance_df, cashflow_df are all empty, no freq_groups). "
            "MF pipeline will resample the daily cache instead of using actual "
            "filing data. Financial ratios at Q/A/M/W frequencies will be "
            "forward-filled interpolated values, NOT actual periodic filings. "
            "Results are OHLCV-only quality."
        )

    frequencies = get_frequencies_slow_to_fast()

    # Detect semi-annual filings and adjust frequency list.
    # Check freq_groups first (more accurate), then fall back to raw DFs.
    if _has_freq_groups:
        _available_freq_labels = set()
        for grp_name in ("income_freq_groups", "balance_freq_groups", "cashflow_freq_groups"):
            _available_freq_labels.update(getattr(state, grp_name, {}).keys())
        if "quarterly" in _available_freq_labels and "semiannual" in _available_freq_labels:
            if "S" not in frequencies:
                q_idx = frequencies.index("Q")
                frequencies.insert(q_idx, "S")
            logger.info("Both Q and S filings detected in freq_groups -- running both pipelines")
        elif "semiannual" in _available_freq_labels and "quarterly" not in _available_freq_labels:
            frequencies = [("S" if f == "Q" else f) for f in frequencies]
            logger.info("Auto-switch: Q -> S (semi-annual filings only in freq_groups)")
    elif _has_raw and "Q" in frequencies:
        _all_freqs = detect_all_filing_frequencies(income_df, balance_df, cashflow_df)
        _native = detect_native_filing_frequency(income_df, balance_df, cashflow_df)
        if "Q" in _all_freqs and "S" in _all_freqs:
            if "S" not in frequencies:
                q_idx = frequencies.index("Q")
                frequencies.insert(q_idx, "S")
            logger.info("Both Q and S filings detected -- running both pipelines")
        elif _native == "S" and "Q" not in _all_freqs:
            frequencies = [("S" if f == "Q" else f) for f in frequencies]
            logger.info("Auto-switch: Q -> S (semi-annual filings only)")

    # Save frequency list for later sub-stages
    state.save_mf_frequencies(frequencies)

    # Map pipeline freq code -> frequency separator label
    _FREQ_CODE_TO_LABEL = {"Q": "quarterly", "S": "semiannual", "A": "annual"}

    # Build and save each ResampledCache
    _degraded_freqs: list[str] = []
    for freq in frequencies:
        _sep_label = _FREQ_CODE_TO_LABEL.get(freq)

        if _sep_label and _has_freq_groups:
            # A/Q/S: Use frequency-specific raw filings (actual filing data
            # for this frequency only, before Chow-Lin reconciliation).
            _inc = state.income_freq_groups.get(_sep_label)
            _bal = state.balance_freq_groups.get(_sep_label)
            _cf = state.cashflow_freq_groups.get(_sep_label)
            _any_freq_data = any(
                df is not None and not df.empty
                for df in [_inc, _bal, _cf]
            )
            if _any_freq_data:
                resampled = build_cache_from_raw_filings(
                    income_df=_inc, balance_df=_bal,
                    cashflow_df=_cf, quotes_df=quotes_df,
                    frequency=freq, reference_date=ref_date,
                )
                logger.info(
                    "[%s] Built from native %s filings: inc=%d, bal=%d, cf=%d",
                    freq, _sep_label,
                    len(_inc) if _inc is not None and not _inc.empty else 0,
                    len(_bal) if _bal is not None and not _bal.empty else 0,
                    len(_cf) if _cf is not None and not _cf.empty else 0,
                )
            else:
                # No native filings for this specific frequency -- fall back
                # to reconciled DFs if available, otherwise daily cache.
                if _has_raw:
                    logger.info(
                        "[%s] No native %s filings in freq_groups -- "
                        "using reconciled DFs (interpolated from available frequencies)",
                        freq, _sep_label,
                    )
                    resampled = build_cache_from_raw_filings(
                        income_df=income_df, balance_df=balance_df,
                        cashflow_df=cashflow_df, quotes_df=quotes_df,
                        frequency=freq, reference_date=ref_date,
                    )
                else:
                    logger.warning(
                        "[%s] DEGRADED: No native %s filings and no raw DFs -- "
                        "using resampled daily cache",
                        freq, _sep_label,
                    )
                    resampled = resample_cache_to_frequency(
                        cache, frequency=freq, reference_date=ref_date,
                    )
                    _degraded_freqs.append(freq)
        elif freq in ("W", "M") and _has_raw:
            # W/M: No native filings exist at these frequencies.
            # Interpolate from reconciled (highest-freq) DFs.
            resampled = build_cache_from_raw_filings(
                income_df=income_df, balance_df=balance_df,
                cashflow_df=cashflow_df, quotes_df=quotes_df,
                frequency=freq, reference_date=ref_date,
            )
        elif freq == "D":
            # Daily: use daily cache as-is (always correct)
            resampled = resample_cache_to_frequency(
                cache, frequency=freq, reference_date=ref_date,
            )
        else:
            # Fallback: resample daily cache (degraded)
            resampled = resample_cache_to_frequency(
                cache, frequency=freq, reference_date=ref_date,
            )
            if freq != "D":
                _degraded_freqs.append(freq)
                logger.warning(
                    "[%s] DEGRADED: Using resampled daily cache (no raw %s filings). "
                    "Financial ratios are forward-filled interpolated values.",
                    freq, resampled.label,
                )

        if resampled.n_periods < 3:
            logger.info("[%s] Skipping -- only %d periods (need 3+)", freq, resampled.n_periods)
            continue

        state.save_mf_cache(freq, resampled)
        logger.info("[%s] Resampled: %d periods, source=%s, saved to disk",
                    freq, resampled.n_periods, resampled.data_source)

    logger.info("Resample prep complete: %d frequencies prepared", len(frequencies))


def _run_7_4_single_freq(state: PipelineState, freq: str) -> None:
    """Run the pipeline for a single frequency, loading context from prior."""
    from operator1.steps.multi_frequency_runner import run_single_frequency_pipeline

    resampled = state.load_mf_cache(freq)
    if resampled is None:
        logger.info("[%s] No resampled cache found -- skipping", freq)
        return

    # Load context from prior frequency (cascading)
    frequencies = state.load_mf_frequencies()
    idx = frequencies.index(freq) if freq in frequencies else -1
    prior_context = None
    if idx > 0:
        prior_freq = frequencies[idx - 1]
        prior_context = state.load_mf_context(prior_freq)

    secrets = _get_mf_secrets()

    result = run_single_frequency_pipeline(
        resampled=resampled,
        prior_context=prior_context,
        secrets=secrets,
        market_id=state.market_id,
        ticker=state.company,
        skip_models=False,
    )

    state.save_mf_result(freq, result)
    state.save_mf_context(freq, result.context_for_next)
    logger.info("[%s] Pipeline complete: %d periods, survival=%.3f (%.1fs)",
                freq, result.n_periods, result.survival_probability, result.elapsed_seconds)


def run_7_4_1_annual(state: PipelineState) -> None:
    """7.4.1: Annual frequency pipeline."""
    logger.info("Sub-stage 7.4.1: Annual pipeline")
    _run_7_4_single_freq(state, "A")


def run_7_4_2_quarterly(state: PipelineState) -> None:
    """7.4.2: Quarterly (or Semi-Annual) frequency pipeline."""
    logger.info("Sub-stage 7.4.2: Quarterly pipeline")
    freqs = state.load_mf_frequencies()
    # Run Q, S, or both depending on what prep detected
    for f in freqs:
        if f in ("Q", "S"):
            _run_7_4_single_freq(state, f)


def run_7_4_3_monthly(state: PipelineState) -> None:
    """7.4.3: Monthly frequency pipeline."""
    logger.info("Sub-stage 7.4.3: Monthly pipeline")
    _run_7_4_single_freq(state, "M")


def run_7_4_4_weekly(state: PipelineState) -> None:
    """7.4.4: Weekly frequency pipeline."""
    logger.info("Sub-stage 7.4.4: Weekly pipeline")
    _run_7_4_single_freq(state, "W")


def run_7_4_5_daily(state: PipelineState) -> None:
    """7.4.5: Daily frequency pipeline."""
    logger.info("Sub-stage 7.4.5: Daily pipeline")
    _run_7_4_single_freq(state, "D")


def run_7_4_6_fusion(state: PipelineState) -> None:
    """7.4.6: Fuse all frequency results into a single FusedMultiFreqResult."""
    logger.info("Sub-stage 7.4.6: Multi-frequency fusion")
    try:
        from operator1.models.frequency_fusion import fuse_multi_frequency_results
        from operator1.steps.multi_frequency_runner import MultiFrequencyResult

        available_freqs = state.list_mf_results()
        if not available_freqs:
            logger.info("No frequency results to fuse")
            return

        results = {}
        for freq in available_freqs:
            r = state.load_mf_result(freq)
            if r is not None:
                results[freq] = r

        if not results:
            return

        mf_result = MultiFrequencyResult(
            results=results,
            execution_order=list(results.keys()),
            total_elapsed_seconds=sum(r.elapsed_seconds for r in results.values()),
        )

        state.multi_frequency_result = fuse_multi_frequency_results(mf_result)
        logger.info(
            "Multi-frequency fusion: %d frequencies, survival=%.1f%%",
            state.multi_frequency_result.n_frequencies_used,
            state.multi_frequency_result.survival.fused_probability * 100,
        )
    except Exception as exc:
        logger.warning("Multi-frequency fusion failed: %s", exc)


def run_7_5_hedge_fund(state: PipelineState) -> None:
    """7.5: Hedge fund analysis (15 investment-grade metrics)."""
    logger.info("Sub-stage 7.5: Hedge fund analysis")
    try:
        from operator1.hedge_fund.engine import run_hedge_fund_analysis

        state.hf_result = run_hedge_fund_analysis(
            income_df=state.income_df,
            balance_df=state.balance_df,
            cashflow_df=state.cashflow_df,
            cache=state.cache,
            target_profile=state.target_profile,
            forecast_result=state.forecast_result,
            mc_result=state.mc_result,
            scenario_result=state.scenario_result,
            multi_frequency_result=state.multi_frequency_result,
            signal_ic_result=state.signal_ic_result,
            filing_calendar_result=state.filing_calendar_result,
            fh_result=state.fh_result,
            peer_ranking_result=state.peer_ranking_result,
            sentiment_result=state.sentiment_result,
            survival_controller=state.survival_controller,
            linked_caches=state.linked_caches,
            macro_data=state.macro_data,
        )
    except Exception as exc:
        logger.warning("Hedge fund analysis failed: %s", exc)

    # Anchor MC survival with Merton default probability + market-cap floor
    try:
        from operator1.models.monte_carlo import anchor_mc_survival
        _mcap = None
        if state.cache is not None and "market_cap" in state.cache.columns:
            if state.cache["market_cap"].notna().any():
                _mcap = float(state.cache["market_cap"].dropna().iloc[-1])
        _merton_pd = None
        if state.hf_result and hasattr(state.hf_result, "advanced_methods"):
            _adv = state.hf_result.advanced_methods
            if hasattr(_adv, "merton_default_probability") and isinstance(_adv.merton_default_probability, dict):
                _merton_pd = _adv.merton_default_probability.get("pd_1yr")
        anchor_mc_survival(state.mc_result, market_cap=_mcap, merton_pd=_merton_pd)
    except Exception:
        pass


# Registry
STAGE_7_SUBSTAGES = [
    ("7.1", run_7_1_uss),
    ("7.2", run_7_2_retro_calibration),
    ("7.3", run_7_3_diagnostics),
    ("7.4.0", run_7_4_0_resample_prep),
    ("7.4.1", run_7_4_1_annual),
    ("7.4.2", run_7_4_2_quarterly),
    ("7.4.3", run_7_4_3_monthly),
    ("7.4.4", run_7_4_4_weekly),
    ("7.4.5", run_7_4_5_daily),
    ("7.4.6", run_7_4_6_fusion),
    ("7.5", run_7_5_hedge_fund),
]
