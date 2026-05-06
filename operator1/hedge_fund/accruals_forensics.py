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

        # --- Phase 1: Dechow F-Score (Dechow et al. 2011) ---
        try:
            _dechow = _compute_dechow_f_score(income_df, balance_df, cashflow_df, cache)
            result.dechow_f_score = _dechow.get("f_score")
            result.dechow_f_probability = _dechow.get("probability")
            result.dechow_f_label = _dechow.get("label", "")
            if result.dechow_f_probability is not None and result.dechow_f_probability > 0.5:
                parts.append(f"Dechow F-Score flags misstatement risk (P={result.dechow_f_probability:.0%})")
        except Exception as _exc:
            logger.debug("Dechow F-Score failed: %s", _exc)

        # --- Phase 1: Real activities manipulation (Roychowdhury 2006) ---
        try:
            _ram = _compute_real_activities_manipulation(income_df, balance_df, cashflow_df)
            result.real_manipulation_score = _ram.get("score")
            result.abnormal_cfo = _ram.get("abnormal_cfo")
            result.abnormal_production = _ram.get("abnormal_production")
            result.abnormal_discretionary = _ram.get("abnormal_discretionary")
            result.manipulation_type = _ram.get("type", "")
            if result.manipulation_type and result.manipulation_type != "clean":
                parts.append(f"Real activities manipulation: {result.manipulation_type}")
        except Exception as _exc:
            logger.debug("Real activities manipulation failed: %s", _exc)

        # --- Phase 8: Accounting conservatism (Basu 1997) ---
        try:
            _cons = _compute_accounting_conservatism(income_df, cache)
            result.conservatism_index = _cons.get("index")
            result.conservatism_label = _cons.get("label", "")
        except Exception as _exc:
            logger.debug("Conservatism index failed: %s", _exc)

        result.narrative = "; ".join(parts) if parts else "Insufficient data for accruals forensics"
        result.available = True

    except Exception as exc:
        logger.warning("Accruals forensics failed: %s", exc)
        result.error = str(exc)

    return result


# ---------------------------------------------------------------------------
# Dechow F-Score (Dechow, Ge, Larson & Sloan 2011, CAR)
# ---------------------------------------------------------------------------

def _compute_dechow_f_score(
    income_df: pd.DataFrame,
    balance_df: pd.DataFrame,
    cashflow_df: pd.DataFrame,
    cache: pd.DataFrame | None = None,
) -> dict:
    """Probit model predicting probability of financial misstatement.

    Uses exact coefficients from Dechow et al. (2011) Table 7.
    Returns dict with f_score, probability, label.
    """
    from scipy.stats import norm

    n = 8
    ni = extract_quarterly_series(income_df, "net_income", n)
    ocf = extract_quarterly_series(cashflow_df, "operating_cash_flow", n)
    ta = extract_quarterly_series(balance_df, "total_assets", n)
    rec = extract_quarterly_series(balance_df, "receivables", n)
    inv = extract_quarterly_series(balance_df, "inventory", n)

    if len(ni) < 2 or len(ta) < 2:
        return {}

    common = ni.index.intersection(ta.index)
    if len(common) < 2:
        return {}

    ta_vals = ta.loc[common]
    ta_lag = ta_vals.shift(1).dropna()
    if len(ta_lag) < 1:
        return {}

    last_idx = ta_lag.index[-1]
    ta_last = float(ta_vals.loc[last_idx])
    ta_lag_last = float(ta_lag.loc[last_idx])
    if abs(ta_lag_last) < 1e-6:
        return {}

    # RSST accruals proxy: (NI - OCF) / avg(TA)
    ocf_c = ocf.reindex(common).fillna(0)
    ni_c = ni.reindex(common).fillna(0)
    avg_ta = (ta_last + ta_lag_last) / 2.0
    rsst = float((ni_c.iloc[-1] - ocf_c.iloc[-1]) / avg_ta) if avg_ta > 1e-6 else 0.0

    # Change in receivables / avg(TA)
    rec_c = rec.reindex(common).fillna(0)
    d_rec = float(rec_c.diff().iloc[-1]) / avg_ta if len(rec_c) >= 2 and avg_ta > 1e-6 else 0.0

    # Change in inventory / avg(TA)
    inv_c = inv.reindex(common).fillna(0)
    d_inv = float(inv_c.diff().iloc[-1]) / avg_ta if len(inv_c) >= 2 and avg_ta > 1e-6 else 0.0

    # Soft assets ratio: (TA - cash - PP&E) / TA
    cash_s = extract_quarterly_series(balance_df, "cash_and_equivalents", n)
    cash_val = float(cash_s.iloc[-1]) if len(cash_s) > 0 else 0.0
    soft = (ta_last - cash_val) / ta_last if ta_last > 1e-6 else 0.5

    # Change in cash sales: proxy as change in (revenue - d_receivables)
    rev = extract_quarterly_series(income_df, "revenue", n)
    d_cash_sales = 0.0
    if len(rev) >= 2:
        cs = rev - rec.reindex(rev.index).fillna(0)
        cs_pct = cs.pct_change().dropna()
        if len(cs_pct) > 0:
            d_cash_sales = float(cs_pct.iloc[-1])

    # Change in ROA
    roa_series = ni.reindex(common) / ta_vals.loc[common]
    d_roa = float(roa_series.diff().iloc[-1]) if len(roa_series) >= 2 else 0.0

    # Issuance proxy: did shares outstanding increase?
    issuance = 0.0
    if cache is not None and "shares_outstanding" in cache.columns:
        so = cache["shares_outstanding"].dropna()
        if len(so) >= 252:
            issuance = 1.0 if float(so.iloc[-1]) > float(so.iloc[-252]) * 1.01 else 0.0

    # Probit coefficients from Dechow et al. (2011) Table 7 Model 1
    F = (-7.893
         + 0.790 * rsst
         + 2.518 * d_rec
         + 1.191 * d_inv
         + 1.979 * soft
         + 0.171 * d_cash_sales
         - 0.932 * d_roa
         + 1.029 * issuance)

    prob = float(norm.cdf(F))
    label = "flag" if prob > 0.50 else "watch" if prob > 0.25 else "clean"

    return {"f_score": round(F, 4), "probability": round(prob, 4), "label": label}


# ---------------------------------------------------------------------------
# Real Activities Manipulation (Roychowdhury 2006, JFE)
# ---------------------------------------------------------------------------

def _compute_real_activities_manipulation(
    income_df: pd.DataFrame,
    balance_df: pd.DataFrame,
    cashflow_df: pd.DataFrame,
) -> dict:
    """Detect overproduction, cost cutting, and sales pull-forward.

    Three OLS regressions estimate 'normal' levels; residuals = 'abnormal'.
    """
    from sklearn.linear_model import LinearRegression

    n = 12  # need more history for regressions
    rev = extract_quarterly_series(income_df, "revenue", n)
    ocf = extract_quarterly_series(cashflow_df, "operating_cash_flow", n)
    ta = extract_quarterly_series(balance_df, "total_assets", n)
    cogs = extract_quarterly_series(income_df, "cost_of_revenue", n)
    inv = extract_quarterly_series(balance_df, "inventory", n)
    sga = extract_quarterly_series(income_df, "sga_expenses", n)
    rd = extract_quarterly_series(income_df, "rd_expenses", n)

    common = rev.index.intersection(ta.index).intersection(ocf.index)
    if len(common) < 6:
        return {}

    # Align all series
    rev_c = rev.loc[common].astype(float)
    ta_c = ta.loc[common].astype(float)
    ocf_c = ocf.loc[common].astype(float)

    # Avoid division by zero
    ta_safe = ta_c.where(ta_c.abs() > 1e-6, 1e-6)
    inv_ta = ta_safe.reciprocal()
    sales_ta = rev_c / ta_safe
    d_sales = rev_c.diff().fillna(0) / ta_safe
    d_sales_lag = d_sales.shift(1).fillna(0)

    # Regression 1: Abnormal CFO
    abnormal_cfo = None
    try:
        y = (ocf_c / ta_safe).values.reshape(-1, 1)
        X = np.column_stack([inv_ta.values, sales_ta.values, d_sales.values])
        mask = np.isfinite(X).all(axis=1) & np.isfinite(y.ravel())
        if mask.sum() >= 4:
            lr = LinearRegression().fit(X[mask], y[mask])
            residuals = y[mask] - lr.predict(X[mask])
            abnormal_cfo = float(residuals[-1])
    except Exception:
        pass

    # Regression 2: Abnormal production costs
    abnormal_prod = None
    try:
        cogs_c = cogs.reindex(common).fillna(0).astype(float)
        inv_c = inv.reindex(common).fillna(0).astype(float)
        prod = cogs_c + inv_c.diff().fillna(0)
        y = (prod / ta_safe).values.reshape(-1, 1)
        X = np.column_stack([inv_ta.values, sales_ta.values, d_sales.values, d_sales_lag.values])
        mask = np.isfinite(X).all(axis=1) & np.isfinite(y.ravel())
        if mask.sum() >= 4:
            lr = LinearRegression().fit(X[mask], y[mask])
            residuals = y[mask] - lr.predict(X[mask])
            abnormal_prod = float(residuals[-1])
    except Exception:
        pass

    # Regression 3: Abnormal discretionary expenses
    abnormal_disc = None
    try:
        disc = sga.reindex(common).fillna(0).astype(float) + rd.reindex(common).fillna(0).astype(float)
        rev_lag = rev_c.shift(1).fillna(rev_c.iloc[0])
        sales_lag_ta = rev_lag / ta_safe
        y = (disc / ta_safe).values.reshape(-1, 1)
        X = np.column_stack([inv_ta.values, sales_lag_ta.values])
        mask = np.isfinite(X).all(axis=1) & np.isfinite(y.ravel())
        if mask.sum() >= 4:
            lr = LinearRegression().fit(X[mask], y[mask])
            residuals = y[mask] - lr.predict(X[mask])
            abnormal_disc = float(residuals[-1])
    except Exception:
        pass

    # Classification
    scores = []
    manip_type = "clean"
    if abnormal_cfo is not None:
        scores.append(abs(abnormal_cfo) * 200)
    if abnormal_prod is not None:
        scores.append(abs(abnormal_prod) * 200)
        if abnormal_prod > 0.05 and (abnormal_cfo is not None and abnormal_cfo < -0.02):
            manip_type = "overproduction"
    if abnormal_disc is not None:
        scores.append(abs(abnormal_disc) * 200)
        if abnormal_disc < -0.05 and manip_type == "clean":
            manip_type = "cost_cutting"

    if abnormal_cfo is not None and abnormal_cfo < -0.05 and manip_type == "clean":
        manip_type = "sales_pull_forward"

    composite = min(100, sum(scores) / max(len(scores), 1)) if scores else None

    return {
        "score": round(composite, 1) if composite is not None else None,
        "abnormal_cfo": round(abnormal_cfo, 4) if abnormal_cfo is not None else None,
        "abnormal_production": round(abnormal_prod, 4) if abnormal_prod is not None else None,
        "abnormal_discretionary": round(abnormal_disc, 4) if abnormal_disc is not None else None,
        "type": manip_type,
    }


# ---------------------------------------------------------------------------
# Accounting Conservatism (Basu 1997, JAE)
# ---------------------------------------------------------------------------

def _compute_accounting_conservatism(
    income_df: pd.DataFrame,
    cache: pd.DataFrame | None = None,
) -> dict:
    """Measure asymmetric timeliness: bad news reflected faster than good news.

    Negative index = aggressive accounting, positive = conservative.
    """
    if cache is None or "return_1d" not in cache.columns:
        return {}

    ni = extract_quarterly_series(income_df, "net_income", 12)
    if len(ni) < 4:
        return {}

    # Compute quarterly earnings changes
    ni_change = ni.pct_change().dropna()
    if len(ni_change) < 3:
        return {}

    # Get quarterly returns (sum daily returns within each quarter)
    returns = cache["return_1d"].dropna()
    if len(returns) < 63:
        return {}

    # Quarterly return aggregation aligned to filing dates
    q_returns = returns.resample("QE").sum()
    common = ni_change.index.intersection(q_returns.index)
    if len(common) < 3:
        # Try approximate alignment
        q_returns_approx = q_returns.reindex(ni_change.index, method="nearest", tolerance=pd.Timedelta("45D"))
        common = ni_change.index.intersection(q_returns_approx.dropna().index)
        if len(common) < 3:
            return {}
        q_returns = q_returns_approx

    ret_c = q_returns.loc[common].astype(float)
    ni_c = ni_change.loc[common].astype(float)

    # Basu asymmetric timeliness: split into positive and negative return quarters
    pos_mask = ret_c > 0
    neg_mask = ret_c <= 0

    if pos_mask.sum() < 2 or neg_mask.sum() < 2:
        return {}

    # Correlation of earnings changes with returns in each group
    corr_pos = float(ret_c[pos_mask].corr(ni_c[pos_mask])) if pos_mask.sum() >= 2 else 0.0
    corr_neg = float(ret_c[neg_mask].corr(ni_c[neg_mask])) if neg_mask.sum() >= 2 else 0.0

    # Handle NaN correlations
    if np.isnan(corr_pos):
        corr_pos = 0.0
    if np.isnan(corr_neg):
        corr_neg = 0.0

    # Conservatism index: higher correlation with bad news = more conservative
    index = corr_neg - corr_pos  # positive = conservative (bad news faster)
    index = max(-1.0, min(1.0, index))

    label = "conservative" if index > 0.15 else "aggressive" if index < -0.15 else "neutral"

    return {"index": round(index, 4), "label": label}
