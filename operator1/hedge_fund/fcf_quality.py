"""HF-1.1 -- FCF Quality Scoring.

Measures whether reported earnings are backed by real cash generation.
Degrades 2-4 quarters before blowups when earnings become manufactured.

Formula:
    FCF_Quality = 0.40 * OCF_NI_Ratio_8Q
                + 0.30 * (1 - |Accruals_Ratio|)
                + 0.20 * FCF_Trend_Slope_8Q
                + 0.10 * (1 - CapEx_Volatility_8Q)

Primary data source: raw quarterly statement DataFrames (8 filings).
"""

from __future__ import annotations

import logging

import numpy as np
import pandas as pd

from operator1.hedge_fund.helpers import (
    extract_quarterly_series,
    compute_rolling_slope,
    get_hf_weight,
    normalize_score,
    safe_divide,
)
from operator1.hedge_fund.types import FCFQualityResult, _score_label

logger = logging.getLogger(__name__)


def compute_fcf_quality(
    income_df: pd.DataFrame,
    cashflow_df: pd.DataFrame,
    balance_df: pd.DataFrame | None = None,
    cache: pd.DataFrame | None = None,
) -> FCFQualityResult:
    """Compute FCF quality score from raw quarterly filings.

    Parameters
    ----------
    income_df:
        Raw income statement DataFrame (one row per filing period).
    cashflow_df:
        Raw cash flow statement DataFrame (one row per filing period).
    balance_df:
        Raw balance sheet DataFrame (for total_assets in accruals fallback).
    cache:
        Daily cache (optional, for pre-computed accruals column).

    Returns
    -------
    FCFQualityResult with score 0-100 (higher = better quality).
    """
    result = FCFQualityResult()
    n_periods = get_hf_weight("data_windows.quarterly_lookback", 8)
    weights = get_hf_weight("earnings_quality.fcf_quality", {})
    w_ocf = weights.get("ocf_ni_ratio_weight", 0.40)
    w_acc = weights.get("accruals_weight", 0.30)
    w_fcf = weights.get("fcf_trend_weight", 0.20)
    w_cap = weights.get("capex_volatility_weight", 0.10)

    try:
        # --- Component 1: OCF / NI ratio (target range 0.8-1.2) ---
        ocf_series = extract_quarterly_series(cashflow_df, "operating_cash_flow", n_periods)
        ni_series = extract_quarterly_series(income_df, "net_income", n_periods)

        ocf_ni_score = 50.0
        if len(ocf_series) >= 2 and len(ni_series) >= 2:
            # Align by index
            common = ocf_series.index.intersection(ni_series.index)
            if len(common) >= 2:
                ocf_vals = ocf_series.loc[common]
                ni_vals = ni_series.loc[common]
                mean_ni = float(ni_vals.mean())
                mean_ocf = float(ocf_vals.mean())

                # B1 FIX: Handle negative NI separately
                if mean_ni < 0:
                    # Loss-making: OCF/NI ratio is meaningless (neg/neg = pos)
                    if mean_ocf > 0:
                        ocf_ni_score = 35.0  # losing money but generating cash
                    else:
                        ocf_ni_score = 5.0   # losing money AND burning cash
                    result.ocf_ni_ratio_8q = None  # ratio not meaningful
                else:
                    # Normal path: positive NI
                    safe_ni = ni_vals.where(ni_vals.abs() > 1e-6)
                    ratios = ocf_vals / safe_ni
                    ratios = ratios.dropna()
                    if len(ratios) > 0:
                        mean_ratio = float(ratios.mean())
                        result.ocf_ni_ratio_8q = mean_ratio
                        if 0.8 <= mean_ratio <= 1.2:
                            ocf_ni_score = 100.0
                        elif mean_ratio > 1.2:
                            ocf_ni_score = max(0, 100 - (mean_ratio - 1.2) * 150)
                        else:
                            ocf_ni_score = max(0, mean_ratio / 0.8 * 100)

        # --- Component 2: Accruals (lower absolute value = better) ---
        accruals_score = 50.0
        # Try cache first (already computed in derived_variables)
        accruals_val = None
        if cache is not None and "accruals" in cache.columns:
            accruals_series = cache["accruals"].dropna()
            if len(accruals_series) > 0:
                accruals_val = float(accruals_series.iloc[-1])
        # Fallback: compute from raw DFs
        if accruals_val is None and len(ni_series) >= 2 and len(ocf_series) >= 2:
            # D1 FIX: total_assets comes from balance_df, not income_df
            ta_series = pd.Series(dtype=float)
            if balance_df is not None and not balance_df.empty:
                ta_series = extract_quarterly_series(balance_df, "total_assets", n_periods)
            if not ta_series.empty:
                common_all = ocf_series.index.intersection(ni_series.index).intersection(ta_series.index)
                if len(common_all) >= 1:
                    ni_last = float(ni_series.loc[common_all].iloc[-1])
                    ocf_last = float(ocf_series.loc[common_all].iloc[-1])
                    ta_last = float(ta_series.loc[common_all].iloc[-1])
                    if abs(ta_last) > 1e-6:
                        accruals_val = (ni_last - ocf_last) / ta_last

        if accruals_val is not None:
            result.accruals_component = accruals_val
            # Score: 100 when |accruals| = 0, degrades linearly
            abs_acc = abs(accruals_val)
            accruals_score = max(0, (1.0 - min(abs_acc, 0.5) / 0.5) * 100)

        # --- Component 3: FCF trend slope (positive = good) ---
        fcf_score = 50.0
        capex_series = extract_quarterly_series(cashflow_df, "capex", n_periods)
        if len(ocf_series) >= 3 and len(capex_series) >= 3:
            common_fc = ocf_series.index.intersection(capex_series.index)
            if len(common_fc) >= 3:
                fcf_series = ocf_series.loc[common_fc] - capex_series.loc[common_fc].abs()
                slope = compute_rolling_slope(fcf_series, window=len(fcf_series))
                if slope is not None:
                    result.fcf_trend_slope = slope
                    # Normalize: positive slope is good, negative is bad
                    # Scale by mean FCF to get relative slope
                    mean_fcf = fcf_series.abs().mean()
                    if mean_fcf > 1e-6:
                        relative_slope = slope / mean_fcf
                        fcf_score = normalize_score(50 + relative_slope * 500, 0, 100)

        # --- Component 4: CapEx volatility (lower = more consistent) ---
        capex_vol_score = 50.0
        if len(capex_series) >= 4:
            capex_abs = capex_series.abs()
            mean_capex = capex_abs.mean()
            if mean_capex > 1e-6:
                cv = float(capex_abs.std() / mean_capex)  # coefficient of variation
                result.capex_volatility = cv
                # Score: 100 when CV=0, degrades to 0 when CV>=1
                capex_vol_score = max(0, (1.0 - min(cv, 1.0)) * 100)

        # --- Composite ---
        composite = (
            w_ocf * ocf_ni_score
            + w_acc * accruals_score
            + w_fcf * fcf_score
            + w_cap * capex_vol_score
        )
        result.score = normalize_score(composite)
        result.label = _score_label(result.score)

        # Degradation flag: compare current to 2Q ago
        if len(ocf_series) >= 6:
            # Rough check: if OCF/NI trend is declining
            if result.fcf_trend_slope is not None and result.fcf_trend_slope < 0:
                result.degradation_flag = True

        # Narrative
        parts = []
        if result.ocf_ni_ratio_8q is not None:
            ratio_str = f"OCF/NI ratio {result.ocf_ni_ratio_8q:.2f}"
            if result.ocf_ni_ratio_8q < 0.7:
                parts.append(f"{ratio_str} (earnings not backed by cash)")
            elif result.ocf_ni_ratio_8q > 1.3:
                parts.append(f"{ratio_str} (possible revenue deferrals)")
            else:
                parts.append(f"{ratio_str} (healthy)")
        if result.degradation_flag:
            parts.append("FCF quality degrading over recent quarters")
        result.narrative = "; ".join(parts) if parts else "Insufficient data for FCF quality assessment"

        result.available = True

    except Exception as exc:
        logger.warning("FCF quality computation failed: %s", exc)
        result.error = str(exc)

    return result
