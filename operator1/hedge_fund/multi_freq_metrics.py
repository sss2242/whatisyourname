"""Multi-frequency HF metric runners with per-frequency merge.

Three HF metrics benefit from multi-frequency analysis:
1. Momentum Composite -- revenue acceleration differs at Q vs A granularity
2. DCF Valuation -- growth assumptions differ by forecast horizon
3. Leverage Stress -- quarterly captures refinancing windows annual misses

Each metric runs independently at each available frequency (Annual,
Quarterly), then merges per-frequency outputs into a single fused result
via regression/curve-fitting. The fused result feeds back into the HF
pipeline as a standard result object -- downstream code sees it
identically to a single-frequency result.
"""

from __future__ import annotations

import logging
import math
from typing import Any

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# 1. Momentum Multi-Frequency
# ---------------------------------------------------------------------------

def compute_momentum_multi_freq(
    income_freq_groups: dict[str, pd.DataFrame],
    cashflow_freq_groups: dict[str, pd.DataFrame],
    cache: pd.DataFrame | None = None,
) -> Any:
    """Run momentum at each frequency, merge via weighted regression.

    Weight: quarterly=0.6, annual=0.4 (quarterly is more responsive).
    If only one frequency available, uses it directly.
    """
    from operator1.hedge_fund.engine import _compute_momentum
    from operator1.hedge_fund.types import MomentumCompositeResult

    freq_weights = {"quarterly": 0.6, "annual": 0.4, "semiannual": 0.5}
    freq_results: dict[str, MomentumCompositeResult] = {}

    for freq_label, income_df in income_freq_groups.items():
        if income_df is None or income_df.empty:
            continue
        cashflow_df = cashflow_freq_groups.get(freq_label, pd.DataFrame())
        try:
            result = _compute_momentum(income_df, cashflow_df, cache)
            freq_results[freq_label] = result
            logger.info("  Momentum @ %s: score=%.0f", freq_label, result.score)
        except Exception as exc:
            logger.debug("Momentum @ %s failed: %s", freq_label, exc)

    if not freq_results:
        return _compute_momentum(pd.DataFrame(), pd.DataFrame(), cache)

    if len(freq_results) == 1:
        result = next(iter(freq_results.values()))
        result.freq_scores = {k: v.score for k, v in freq_results.items()}
        result.merge_method = "single_freq"
        result.freq_agreement = 1.0
        return result

    # Weighted merge
    total_weight = 0.0
    weighted_score = 0.0
    scores = {}
    for freq_label, res in freq_results.items():
        w = freq_weights.get(freq_label, 0.5)
        weighted_score += w * res.score
        total_weight += w
        scores[freq_label] = res.score

    fused_score = weighted_score / total_weight if total_weight > 0 else 50.0

    # Agreement: do frequencies agree on direction (above/below 50)?
    directions = [1 if s > 50 else -1 for s in scores.values()]
    agreement = 1.0 if len(set(directions)) == 1 else 0.0

    # Use the quarterly result as base (more granular), override score
    base_freq = "quarterly" if "quarterly" in freq_results else next(iter(freq_results))
    merged = freq_results[base_freq]
    merged.score = fused_score
    merged.freq_scores = scores
    merged.merge_method = "weighted_regression"
    merged.freq_agreement = agreement

    logger.info(
        "  Momentum merged: score=%.0f (freqs=%s, agreement=%.0f%%)",
        fused_score, scores, agreement * 100,
    )
    return merged


# ---------------------------------------------------------------------------
# 2. DCF Multi-Frequency
# ---------------------------------------------------------------------------

def compute_dcf_multi_freq(
    cashflow_freq_groups: dict[str, pd.DataFrame],
    balance_freq_groups: dict[str, pd.DataFrame],
    cache: pd.DataFrame | None = None,
    target_profile: dict | None = None,
    mc_result: Any = None,
    macro_data: dict | None = None,
    income_freq_groups: dict[str, pd.DataFrame] | None = None,
) -> Any:
    """Run DCF at each frequency, merge via inverse-variance blend.

    Annual DCF: more stable growth estimates.
    Quarterly DCF: more responsive to recent changes.
    Fused: precision-weighted average of P50 intrinsic values.
    """
    from operator1.hedge_fund.engine import _compute_dcf
    from operator1.hedge_fund.types import DCFResult

    freq_results: dict[str, DCFResult] = {}

    for freq_label, cashflow_df in cashflow_freq_groups.items():
        if cashflow_df is None or cashflow_df.empty:
            continue
        balance_df = balance_freq_groups.get(freq_label, pd.DataFrame())
        income_df = (income_freq_groups or {}).get(freq_label, pd.DataFrame())
        try:
            result = _compute_dcf(
                cashflow_df, balance_df, cache, target_profile,
                mc_result, macro_data, income_df=income_df if not income_df.empty else None,
            )
            if result.available and result.intrinsic_p50 is not None:
                freq_results[freq_label] = result
                logger.info(
                    "  DCF @ %s: P50=$%.2f, upside=%.1f%%",
                    freq_label, result.intrinsic_p50, result.upside_pct or 0,
                )
        except Exception as exc:
            logger.debug("DCF @ %s failed: %s", freq_label, exc)

    if not freq_results:
        return _compute_dcf(pd.DataFrame(), pd.DataFrame(), cache, target_profile, mc_result, macro_data)

    if len(freq_results) == 1:
        result = next(iter(freq_results.values()))
        result.freq_intrinsics = {k: v.intrinsic_p50 for k, v in freq_results.items()}
        result.merge_method = "single_freq"
        result.freq_divergence = 0.0
        return result

    # Inverse-variance blend of P50 values
    p50s = {}
    spreads = {}
    for freq_label, res in freq_results.items():
        p50 = res.intrinsic_p50
        spread = (res.intrinsic_p90 or p50 * 1.2) - (res.intrinsic_p10 or p50 * 0.8)
        p50s[freq_label] = p50
        spreads[freq_label] = max(spread, 0.01)

    # Precision = 1 / variance (wider spread = lower precision)
    precisions = {f: 1.0 / (s ** 2) for f, s in spreads.items()}
    total_precision = sum(precisions.values())
    weights = {f: p / total_precision for f, p in precisions.items()}

    fused_p50 = sum(weights[f] * p50s[f] for f in p50s)

    # Divergence: how much do frequencies disagree?
    mean_p50 = sum(p50s.values()) / len(p50s)
    divergence = sum(abs(p - mean_p50) for p in p50s.values()) / len(p50s) / max(mean_p50, 1)

    # Conservative bands: union of P10-P90 ranges
    all_p10 = [r.intrinsic_p10 for r in freq_results.values() if r.intrinsic_p10 is not None]
    all_p90 = [r.intrinsic_p90 for r in freq_results.values() if r.intrinsic_p90 is not None]

    # Build merged result from quarterly (more granular) or first available
    base_freq = "quarterly" if "quarterly" in freq_results else next(iter(freq_results))
    merged = freq_results[base_freq]
    merged.intrinsic_p50 = fused_p50
    if all_p10:
        merged.intrinsic_p10 = min(all_p10)
    if all_p90:
        merged.intrinsic_p90 = max(all_p90)

    # Recalculate upside from fused P50
    if cache is not None and "close" in cache.columns and cache["close"].notna().any():
        current = float(cache["close"].dropna().iloc[-1])
        if current > 0:
            merged.upside_pct = (fused_p50 - current) / current * 100

    merged.freq_intrinsics = p50s
    merged.merge_method = "inverse_variance"
    merged.freq_divergence = divergence

    logger.info(
        "  DCF merged: P50=$%.2f (freqs=%s, divergence=%.2f, weights=%s)",
        fused_p50,
        {k: f"${v:.2f}" for k, v in p50s.items()},
        divergence,
        {k: f"{v:.2f}" for k, v in weights.items()},
    )
    return merged


# ---------------------------------------------------------------------------
# 3. Leverage Stress Multi-Frequency
# ---------------------------------------------------------------------------

def compute_leverage_stress_multi_freq(
    income_freq_groups: dict[str, pd.DataFrame],
    balance_freq_groups: dict[str, pd.DataFrame],
    mc_result: Any = None,
) -> Any:
    """Run leverage stress at each frequency, merge via worst-case envelope.

    Annual: sees long-term debt sustainability.
    Quarterly: sees near-term refinancing windows.
    Merged: takes WORSE outcome across frequencies for each scenario.
    """
    from operator1.hedge_fund.engine import _compute_leverage_stress
    from operator1.hedge_fund.types import LeverageStressResult

    freq_results: dict[str, LeverageStressResult] = {}

    for freq_label, income_df in income_freq_groups.items():
        if income_df is None or income_df.empty:
            continue
        balance_df = balance_freq_groups.get(freq_label, pd.DataFrame())
        if balance_df.empty:
            continue
        try:
            result = _compute_leverage_stress(income_df, balance_df, mc_result)
            if result.available:
                freq_results[freq_label] = result
                breach = "breach" if result.base_case.covenant_breach else "ok"
                logger.info("  LevStress @ %s: base=%s", freq_label, breach)
        except Exception as exc:
            logger.debug("LevStress @ %s failed: %s", freq_label, exc)

    if not freq_results:
        return _compute_leverage_stress(pd.DataFrame(), pd.DataFrame(), mc_result)

    if len(freq_results) == 1:
        result = next(iter(freq_results.values()))
        result.freq_results = {k: {"covenant_breach": v.base_case.covenant_breach}
                               for k, v in freq_results.items()}
        result.merge_method = "single_freq"
        result.freq_divergence = 0.0
        return result

    # Worst-case envelope: for each scenario, take the worse outcome
    base_freq = "quarterly" if "quarterly" in freq_results else next(iter(freq_results))
    merged = freq_results[base_freq]

    for freq_label, res in freq_results.items():
        if freq_label == base_freq:
            continue
        # For each scenario, take worse (higher) debt_to_ebitda and covenant breach
        for scenario_attr in ("base_case", "revenue_miss", "systemic_crisis"):
            merged_scenario = getattr(merged, scenario_attr, None)
            other_scenario = getattr(res, scenario_attr, None)
            if merged_scenario is None or other_scenario is None:
                continue
            # Covenant breach: any frequency saying breach = breach
            if getattr(other_scenario, "covenant_breach", False):
                merged_scenario.covenant_breach = True
            # Debt/EBITDA: take the higher (worse) value
            other_de = getattr(other_scenario, "debt_to_ebitda", None)
            merged_de = getattr(merged_scenario, "debt_to_ebitda", None)
            if other_de is not None and (merged_de is None or other_de > merged_de):
                merged_scenario.debt_to_ebitda = other_de

    # Refinancing risk: any frequency showing risk = risk
    merged.refinancing_risk = any(
        getattr(r, "refinancing_risk", False) for r in freq_results.values()
    )

    # Divergence: do frequencies agree on risk level?
    breaches = [r.base_case.covenant_breach for r in freq_results.values() if hasattr(r.base_case, "covenant_breach")]
    divergence = 0.0 if len(set(breaches)) <= 1 else 1.0

    merged.freq_results = {
        k: {"covenant_breach": v.base_case.covenant_breach if hasattr(v.base_case, "covenant_breach") else False}
        for k, v in freq_results.items()
    }
    merged.merge_method = "worst_case_envelope"
    merged.freq_divergence = divergence

    logger.info(
        "  LevStress merged: breach=%s, refinancing=%s (freqs=%s, divergence=%.0f%%)",
        merged.base_case.covenant_breach if hasattr(merged.base_case, "covenant_breach") else "?",
        merged.refinancing_risk,
        list(freq_results.keys()),
        divergence * 100,
    )
    return merged
