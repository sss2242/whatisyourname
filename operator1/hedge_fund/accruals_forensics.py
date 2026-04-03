"""HF-1.2 -- Accruals Forensics.

Extends existing Sloan accruals and Beneish M-Score with Modified Jones
Model discretionary accruals and cash conversion efficiency tracking.

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
from operator1.hedge_fund.types import AccrualsForensicResult, _risk_label

logger = logging.getLogger(__name__)


def compute_accruals_forensics(
    income_df: pd.DataFrame,
    balance_df: pd.DataFrame,
    cashflow_df: pd.DataFrame,
    cache: pd.DataFrame | None = None,
) -> AccrualsForensicResult:
    """Compute accruals forensic analysis from raw quarterly filings.

    Parameters
    ----------
    income_df, balance_df, cashflow_df:
        Raw statement DataFrames (one row per filing period).
    cache:
        Daily cache (optional, for pre-computed accruals_signal).
    """
    result = AccrualsForensicResult()
    n_periods = get_hf_weight("data_windows.quarterly_lookback", 8)
    weights = get_hf_weight("earnings_quality.accruals_forensics", {})
    w_sloan = weights.get("sloan_weight", 0.30)
    w_jones = weights.get("jones_weight", 0.25)
    w_div = weights.get("ni_ocf_divergence_weight", 0.20)
    w_wc = weights.get("working_capital_weight", 0.15)
    w_otc = weights.get("one_time_charges_weight", 0.10)

    try:
        ni_series = extract_quarterly_series(income_df, "net_income", n_periods)
        ocf_series = extract_quarterly_series(cashflow_df, "operating_cash_flow", n_periods)
        ta_series = extract_quarterly_series(balance_df, "total_assets", n_periods)
        rev_series = extract_quarterly_series(income_df, "revenue", n_periods)
        rec_series = extract_quarterly_series(balance_df, "receivables", n_periods)
        cogs_series = extract_quarterly_series(income_df, "cost_of_revenue", n_periods)

        # --- Component 1: Sloan accruals (NI - OCF) / TA ---
        sloan_score = 50.0
        if len(ni_series) >= 2 and len(ocf_series) >= 2 and len(ta_series) >= 2:
            common = ni_series.index.intersection(ocf_series.index).intersection(ta_series.index)
            if len(common) >= 2:
                accruals = (ni_series.loc[common] - ocf_series.loc[common]) / ta_series.loc[common].where(
                    ta_series.loc[common].abs() > 1e-6
                )
                accruals = accruals.dropna()
                if len(accruals) > 0:
                    latest = float(accruals.iloc[-1])
                    result.sloan_accruals = latest
                    # High absolute accruals = red flag. Score 0 at |0.3|, 100 at 0
                    sloan_score = max(0, (1.0 - min(abs(latest), 0.3) / 0.3) * 100)
                    # Invert: higher score = higher risk
                    sloan_score = 100 - sloan_score

        # --- Component 2: Modified Jones discretionary accruals ---
        jones_score = 50.0  # neutral if insufficient data
        jones_min = get_hf_weight("data_windows.jones_min_annual", 5)
        if len(ta_series) >= jones_min and len(rev_series) >= jones_min and len(rec_series) >= jones_min:
            try:
                # Total accruals / TA_lag
                ta_lag = ta_series.shift(1).dropna()
                common_j = ta_lag.index.intersection(rev_series.index).intersection(rec_series.index).intersection(
                    ni_series.index).intersection(ocf_series.index)
                if len(common_j) >= jones_min:
                    total_accruals = ni_series.loc[common_j] - ocf_series.loc[common_j]
                    dep_var = total_accruals / ta_lag.loc[common_j]
                    inv_ta = 1.0 / ta_lag.loc[common_j]
                    delta_rev = rev_series.loc[common_j].diff()
                    delta_rec = rec_series.loc[common_j].diff()
                    x_var = (delta_rev - delta_rec) / ta_lag.loc[common_j]

                    # OLS: dep_var = a * inv_ta + b * x_var + residual
                    dep_clean = dep_var.dropna()
                    common_clean = dep_clean.index.intersection(inv_ta.dropna().index).intersection(
                        x_var.dropna().index
                    )
                    if len(common_clean) >= 3:
                        X = np.column_stack([
                            inv_ta.loc[common_clean].values,
                            x_var.loc[common_clean].values,
                        ])
                        y = dep_clean.loc[common_clean].values
                        # Add intercept
                        X = np.column_stack([np.ones(len(X)), X])
                        try:
                            coeffs = np.linalg.lstsq(X, y, rcond=None)[0]
                            predicted = X @ coeffs
                            residuals = y - predicted
                            discretionary = float(residuals[-1])
                            result.modified_jones_discretionary = discretionary
                            # High |discretionary| = manipulation risk
                            jones_score = min(100, abs(discretionary) / 0.1 * 100)
                        except np.linalg.LinAlgError:
                            pass
            except Exception as exc:
                logger.debug("Modified Jones failed: %s", exc)

        # --- Component 3: NI-OCF divergence trend ---
        divergence_score = 50.0
        if len(ni_series) >= 4 and len(ocf_series) >= 4:
            common_d = ni_series.index.intersection(ocf_series.index)
            if len(common_d) >= 4:
                divergence = ni_series.loc[common_d] - ocf_series.loc[common_d]
                slope = compute_rolling_slope(divergence, window=len(divergence))
                if slope is not None:
                    result.ni_ocf_divergence_trend = slope
                    # Positive slope = NI growing faster than OCF = concerning
                    mean_abs = divergence.abs().mean()
                    if mean_abs > 1e-6:
                        rel_slope = slope / mean_abs
                        divergence_score = normalize_score(50 + rel_slope * 300, 0, 100)

        # --- Component 4: Working capital anomalies ---
        wc_score = 50.0
        ca_series = extract_quarterly_series(balance_df, "current_assets", n_periods)
        cl_series = extract_quarterly_series(balance_df, "current_liabilities", n_periods)
        if len(ca_series) >= 4 and len(cl_series) >= 4:
            common_wc = ca_series.index.intersection(cl_series.index)
            if len(common_wc) >= 4:
                wc = ca_series.loc[common_wc] - cl_series.loc[common_wc]
                wc_change = wc.diff().dropna()
                if len(wc_change) >= 2:
                    # Large unexplained WC swings = anomaly
                    wc_vol = float(wc_change.std())
                    wc_mean = float(wc.abs().mean())
                    if wc_mean > 1e-6:
                        wc_cv = wc_vol / wc_mean
                        result.working_capital_anomaly = wc_cv
                        wc_score = normalize_score(min(100, wc_cv / 0.5 * 100), 0, 100)

        # --- Component 5: Cash conversion efficiency ---
        cce_score = 50.0
        if len(ocf_series) >= 4 and len(rev_series) >= 4 and len(cogs_series) >= 4:
            common_c = ocf_series.index.intersection(rev_series.index).intersection(cogs_series.index)
            if len(common_c) >= 4:
                gross = rev_series.loc[common_c] - cogs_series.loc[common_c]
                safe_gross = gross.where(gross.abs() > 1e-6)
                cce = ocf_series.loc[common_c] / safe_gross
                cce = cce.dropna()
                if len(cce) >= 2:
                    result.cash_conversion_efficiency = float(cce.iloc[-1])
                    cce_slope = compute_rolling_slope(cce, window=len(cce))
                    if cce_slope is not None:
                        result.cce_trend = "improving" if cce_slope > 0.01 else "declining" if cce_slope < -0.01 else "stable"
                        # Declining CCE = risk. Score higher = worse
                        cce_score = normalize_score(50 - cce_slope * 1000, 0, 100)

        # One-time charges placeholder (needs more data than we typically have)
        otc_score = 50.0

        # --- Composite (higher = more risk) ---
        composite = (
            w_sloan * sloan_score
            + w_jones * jones_score
            + w_div * divergence_score
            + w_wc * wc_score
            + w_otc * otc_score
        )
        result.red_flag_score = normalize_score(composite)
        result.label = _risk_label(result.red_flag_score)

        # Narrative
        parts = []
        if result.sloan_accruals is not None:
            parts.append(f"Sloan accruals {result.sloan_accruals:.3f}")
        if result.modified_jones_discretionary is not None:
            if abs(result.modified_jones_discretionary) > 0.05:
                parts.append(f"High discretionary accruals ({result.modified_jones_discretionary:.3f})")
        if result.cce_trend == "declining":
            parts.append("Cash conversion efficiency declining")
        result.narrative = "; ".join(parts) if parts else "Insufficient data for accruals forensics"

        result.available = True

    except Exception as exc:
        logger.warning("Accruals forensics failed: %s", exc)
        result.error = str(exc)

    return result
