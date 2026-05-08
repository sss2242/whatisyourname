"""T3.3 -- Derived decision variables (daily per entity).

Computes ~25 derived financial metrics from the as-of daily cache built
by ``cache_builder.py``.  Every derived variable gets a companion
``is_missing_<var>`` flag and, where a ratio is involved, an
``invalid_math_<var>`` flag.

The central abstraction is ``safe_ratio`` (Sec 15): if the denominator
is null, zero, or very small (< EPSILON), the result is null and the
appropriate flags are set.
"""

from __future__ import annotations

import logging
from typing import Sequence

import numpy as np
import pandas as pd

from operator1.constants import EPSILON

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Frequency context -- set by compute_derived_variables(freq=...) before
# running the stage pipeline.  Each stage reads this to choose the correct
# formula for the current frequency.
# ---------------------------------------------------------------------------

_CURRENT_FREQ: str = "D"  # default: daily

# Number of days in one period at each frequency (for DSO/DIO/DPO etc.)
_PERIOD_DAYS: dict[str, int] = {
    "D": 1,
    "W": 7,
    "M": 30,
    "Q": 90,
    "S": 180,
    "A": 365,
}

# Annualization multiplier for flow variables at each frequency.
# At Q frequency, quarterly EPS * 4 = annual EPS.  At A, no adjustment.
# At D/W/M: set to 1.0 (NO annualization) because daily cache has flow
# variables at MIXED scales (some daily rates from interpolator, some raw
# quarterly values from CompanyFacts).  Correct ratios come from Q/A
# pipeline results forward-filled by Stage 2.F fusion.
_ANNUALIZE_MULT: dict[str, float] = {
    "D": 1.0,    # no annualization -- mixed scale on daily cache
    "W": 1.0,    # no annualization -- interpolated from Q/A
    "M": 1.0,    # no annualization -- interpolated from Q/A
    "Q": 4.0,    # quarterly -> annual
    "S": 2.0,    # semi-annual -> annual
    "A": 1.0,    # already annual
}

# Frequencies where flow/stock and market/flow ratios are VALID
# (the input data is at native filing scale, not interpolated daily rates)
_NATIVE_RATIO_FREQS: set[str] = {"Q", "A", "S"}

# Frequencies where technical indicators (SMA, RSI, MACD, ADX, BB, OBV)
# are meaningful (need dense time series)
_TECHNICAL_FREQS: set[str] = {"D", "W"}


# ---------------------------------------------------------------------------
# Rolling 4-quarter TTM helper
# ---------------------------------------------------------------------------


def _rolling_4q_ttm(series: pd.Series) -> pd.Series:
    """Compute trailing-twelve-month (TTM) sums from as-of aligned data.

    As-of aligned data repeats each quarterly value daily until the next
    filing.  To compute a true TTM we:
      1. Detect quarter transitions (where the value changes).
      2. Extract the distinct quarterly values.
      3. Rolling-sum the last 4 quarters.
      4. Broadcast back onto the daily index.

    If fewer than 4 quarters are available, we scale up proportionally
    (e.g. 2 quarters * 2 = annualised estimate).

    Parameters
    ----------
    series:
        Daily-frequency series where periodic values are forward-filled.

    Returns
    -------
    pd.Series
        TTM (rolling 4-quarter sum) aligned to the original daily index.
    """
    if series.isna().all():
        return series.copy()

    # Detect transitions: where the value changes (new quarter reported)
    shifted = series.shift(1)
    is_new_quarter = (
        series.notna()
        & ((shifted.isna()) | (series != shifted))
    )

    # Extract quarterly values at transition points
    quarterly_vals = series[is_new_quarter].copy()

    if quarterly_vals.empty:
        # No transitions detected -- single constant value
        return series.copy()

    # Rolling sum of last 4 quarters
    if len(quarterly_vals) >= 4:
        q_ttm = quarterly_vals.rolling(window=4, min_periods=1).sum()
    else:
        q_ttm = quarterly_vals.rolling(window=len(quarterly_vals), min_periods=1).sum()

    # Scale up if fewer than 4 quarters available
    q_count = quarterly_vals.rolling(window=4, min_periods=1).count()
    scale = (4.0 / q_count).clip(upper=4.0)
    # Only scale when we have fewer than 4 quarters
    q_ttm = q_ttm.where(q_count >= 4, q_ttm * scale)

    # Broadcast quarterly TTM values back onto the daily index
    ttm_daily = q_ttm.reindex(series.index, method="ffill")

    return ttm_daily


# ---------------------------------------------------------------------------
# Safe ratio helper (Sec 15)
# ---------------------------------------------------------------------------


def safe_ratio(
    numerator: pd.Series,
    denominator: pd.Series,
    name: str,
    epsilon: float = EPSILON,
) -> tuple[pd.Series, pd.Series, pd.Series]:
    """Compute numerator / denominator with safety checks.

    Parameters
    ----------
    numerator, denominator:
        Aligned pandas Series.
    name:
        Variable name -- used for flag column naming.
    epsilon:
        Minimum absolute denominator value.

    Returns
    -------
    (result, is_missing, invalid_math)
        ``result`` is the ratio (null where unsafe).
        ``is_missing`` is 1 where result is null.
        ``invalid_math`` is 1 where denominator was 0/tiny/null.
    """
    denom_bad = (
        denominator.isna()
        | (denominator == 0)
        | (denominator.abs() < epsilon)
    )
    result = numerator / denominator
    result = result.where(~denom_bad, other=np.nan)

    is_missing = result.isna().astype(int)
    invalid_math = denom_bad.astype(int)

    return result, is_missing, invalid_math


def _set_ratio_columns(
    df: pd.DataFrame,
    name: str,
    result: pd.Series,
    is_missing: pd.Series,
    invalid_math: pd.Series,
) -> None:
    """Set the ratio result and companion flag columns in-place."""
    df[name] = result
    df[f"is_missing_{name}"] = is_missing
    df[f"invalid_math_{name}"] = invalid_math


# ---------------------------------------------------------------------------
# Returns & risk
# ---------------------------------------------------------------------------


def _compute_returns_and_risk(df: pd.DataFrame) -> pd.DataFrame:
    """Compute return/risk variables from ``close`` price.

    Variables: return_1d, log_return_1d, volatility_21d, drawdown_252d.
    """
    close = df["close"]

    # Daily return
    df["return_1d"] = close.pct_change()
    df["is_missing_return_1d"] = df["return_1d"].isna().astype(int)

    # Log return
    shifted = close.shift(1)
    with np.errstate(divide="ignore", invalid="ignore"):
        log_ret = np.log(close / shifted)
    df["log_return_1d"] = log_ret.where(shifted.notna() & (shifted > 0), other=np.nan)
    df["is_missing_log_return_1d"] = df["log_return_1d"].isna().astype(int)

    # Rolling 21-day volatility (std of daily returns)
    df["volatility_21d"] = df["return_1d"].rolling(window=21, min_periods=5).std()
    df["is_missing_volatility_21d"] = df["volatility_21d"].isna().astype(int)

    # EWMA volatility (RiskMetrics, JP Morgan 1996): gives more weight to
    # recent observations, adapting faster to regime changes than simple
    # rolling std.  Both are available for temporal models to choose from.
    df["volatility_ewma_21d"] = df["return_1d"].ewm(span=21, min_periods=5).std()
    df["is_missing_volatility_ewma_21d"] = df["volatility_ewma_21d"].isna().astype(int)

    # Rolling 252-day max drawdown
    rolling_max = close.rolling(window=252, min_periods=1).max()
    drawdown = (close - rolling_max) / rolling_max
    df["drawdown_252d"] = drawdown
    df["is_missing_drawdown_252d"] = drawdown.isna().astype(int)

    return df


# ---------------------------------------------------------------------------
# Solvency / leverage
# ---------------------------------------------------------------------------


def _compute_solvency(df: pd.DataFrame) -> pd.DataFrame:
    """Compute solvency/leverage variables.

    Variables: total_debt_asof, debt_to_equity_signed, debt_to_equity_abs,
    net_debt, net_debt_to_ebitda.
    """
    # Total debt as-of (short-term + long-term)
    st_debt = df.get("short_term_debt", pd.Series(np.nan, index=df.index))
    lt_debt = df.get("long_term_debt", pd.Series(np.nan, index=df.index))
    df["total_debt_asof"] = st_debt.fillna(0) + lt_debt.fillna(0)
    # Mark missing only when *both* components are null
    df["is_missing_total_debt_asof"] = (st_debt.isna() & lt_debt.isna()).astype(int)

    # Debt to equity -- signed (preserves negative equity)
    equity = df.get("total_equity", pd.Series(np.nan, index=df.index))
    result, ism, inv = safe_ratio(df["total_debt_asof"], equity, "debt_to_equity_signed")
    _set_ratio_columns(df, "debt_to_equity_signed", result, ism, inv)

    # Debt to equity -- absolute (for survival triggers)
    abs_equity = equity.abs()
    result, ism, inv = safe_ratio(df["total_debt_asof"], abs_equity, "debt_to_equity_abs")
    _set_ratio_columns(df, "debt_to_equity_abs", result, ism, inv)

    # Net debt = total_debt - cash
    cash = df.get("cash_and_equivalents", pd.Series(np.nan, index=df.index))
    df["net_debt"] = df["total_debt_asof"] - cash.fillna(0)
    df["is_missing_net_debt"] = (
        df["is_missing_total_debt_asof"].astype(bool) & cash.isna()
    ).astype(int)

    # Net debt to EBITDA (STOCK / FLOW -- needs annualized EBITDA)
    # Moody's uses 5.0x threshold for investment-grade boundary.
    # At Q: ebitda is quarterly -> annualize by *4.  At A: use as-is.
    # At D: ebitda may be daily rate or raw Q (inconsistent) -> use TTM if available.
    freq = _CURRENT_FREQ
    ebitda = df.get("ebitda", pd.Series(np.nan, index=df.index))
    if freq in _NATIVE_RATIO_FREQS:
        _ebitda_annual = ebitda * _ANNUALIZE_MULT.get(freq, 1.0)
        result, ism, inv = safe_ratio(df["net_debt"], _ebitda_annual, "net_debt_to_ebitda")
    else:
        # D/W/M: prefer TTM EBITDA if available (from _compute_ttm_and_growth)
        _ebitda_ttm = df.get("ebitda_ttm_asof", ebitda)
        result, ism, inv = safe_ratio(df["net_debt"], _ebitda_ttm, "net_debt_to_ebitda")
    _set_ratio_columns(df, "net_debt_to_ebitda", result, ism, inv)

    return df


# ---------------------------------------------------------------------------
# Liquidity / survival
# ---------------------------------------------------------------------------


def _compute_liquidity(df: pd.DataFrame) -> pd.DataFrame:
    """Compute liquidity ratios.

    Variables: current_ratio, quick_ratio, cash_ratio.
    """
    ca = df.get("current_assets", pd.Series(np.nan, index=df.index))
    cl = df.get("current_liabilities", pd.Series(np.nan, index=df.index))
    cash = df.get("cash_and_equivalents", pd.Series(np.nan, index=df.index))
    receivables = df.get("receivables", pd.Series(np.nan, index=df.index))

    # Current ratio = current_assets / current_liabilities
    result, ism, inv = safe_ratio(ca, cl, "current_ratio")
    _set_ratio_columns(df, "current_ratio", result, ism, inv)

    # Quick ratio = (cash + receivables) / current_liabilities
    quick_assets = cash.fillna(0) + receivables.fillna(0)
    # Mark quick assets as fully missing when both components are null
    quick_missing = cash.isna() & receivables.isna()
    quick_assets = quick_assets.where(~quick_missing, other=np.nan)
    result, ism, inv = safe_ratio(quick_assets, cl, "quick_ratio")
    _set_ratio_columns(df, "quick_ratio", result, ism, inv)

    # Cash ratio = cash / current_liabilities
    result, ism, inv = safe_ratio(cash, cl, "cash_ratio")
    _set_ratio_columns(df, "cash_ratio", result, ism, inv)

    return df


# ---------------------------------------------------------------------------
# Cash reality
# ---------------------------------------------------------------------------


def _compute_cash_reality(df: pd.DataFrame) -> pd.DataFrame:
    """Compute cash-flow reality variables.

    Variables: free_cash_flow, free_cash_flow_ttm_asof, fcf_yield.

    Frequency-aware:
    - At Q/A/S: FCF is at native filing scale.  TTM = sum of 4Q (or 1A).
      fcf_yield = FCF_annualized / market_cap.
    - At D/W/M: FCF is interpolated daily rate (garbage for ratios).
      free_cash_flow column still computed (for time-series models).
      fcf_yield SKIPPED (forward-filled from Q/A after fusion).
    """
    freq = _CURRENT_FREQ
    ocf = df.get("operating_cash_flow", pd.Series(np.nan, index=df.index))
    capex = df.get("capex", pd.Series(np.nan, index=df.index))
    market_cap = df.get("market_cap", pd.Series(np.nan, index=df.index))

    # Free cash flow = operating CF - capex (valid at any freq for time-series)
    capex_abs = capex.abs()
    df["free_cash_flow"] = ocf - capex_abs
    df["is_missing_free_cash_flow"] = (ocf.isna() & capex.isna()).astype(int)

    if freq in _NATIVE_RATIO_FREQS:
        # At Q/A/S: values are at native filing scale.
        # TTM for Q: sum 4 quarters.  For A: annual value IS the TTM.
        if freq == "A":
            df["free_cash_flow_ttm_asof"] = df["free_cash_flow"]
        elif freq == "S":
            # Sum 2 semi-annual periods = annual
            df["free_cash_flow_ttm_asof"] = _rolling_4q_ttm(df["free_cash_flow"])
        else:  # Q
            df["free_cash_flow_ttm_asof"] = _rolling_4q_ttm(df["free_cash_flow"])
        df["is_missing_free_cash_flow_ttm_asof"] = df["free_cash_flow_ttm_asof"].isna().astype(int)

        # FCF yield = FCF_TTM / market_cap (Greenblatt 2006)
        _fcf_for_yield = df["free_cash_flow_ttm_asof"]
        result, ism, inv = safe_ratio(_fcf_for_yield, market_cap, "fcf_yield")
        _set_ratio_columns(df, "fcf_yield", result, ism, inv)
    else:
        # D/W/M: TTM from interpolated daily rates is distorted.
        # Still compute TTM for backward compat but mark as unreliable.
        df["free_cash_flow_ttm_asof"] = _rolling_4q_ttm(df["free_cash_flow"])
        df["is_missing_free_cash_flow_ttm_asof"] = df["free_cash_flow_ttm_asof"].isna().astype(int)
        # fcf_yield: use TTM attempt but log warning
        _fcf_for_yield = df.get("free_cash_flow_ttm_asof", df["free_cash_flow"])
        result, ism, inv = safe_ratio(_fcf_for_yield, market_cap, "fcf_yield")
        _set_ratio_columns(df, "fcf_yield", result, ism, inv)
        logger.debug("fcf_yield computed at freq=%s (may be distorted by interpolation)", freq)

    return df


# ---------------------------------------------------------------------------
# Profitability
# ---------------------------------------------------------------------------


def _compute_profitability(df: pd.DataFrame) -> pd.DataFrame:
    """Compute profitability ratios.

    Variables: gross_margin, operating_margin, net_margin, roe.

    Frequency-aware:
    - At Q/A/S: all flow variables are at native filing scale from the
      SAME filing, so flow/flow ratios (margins) are correct.  ROE
      (flow/stock) is also correct because NI is at period scale.
    - At D/W/M: flow variables may come from different interpolation
      paths, producing incorrect margins (e.g. gross_margin = 0.775
      instead of 0.469 for AAPL).  We still compute them but apply
      stricter plausibility guards.  ROE is SKIPPED (flow/stock broken).
    """
    freq = _CURRENT_FREQ
    revenue = df.get("revenue", pd.Series(np.nan, index=df.index))
    gross_profit = df.get("gross_profit", pd.Series(np.nan, index=df.index))
    ebit = df.get("operating_income", pd.Series(np.nan, index=df.index))
    if ebit.isna().all():
        ebit = df.get("ebit", pd.Series(np.nan, index=df.index))
    net_income = df.get("net_income", pd.Series(np.nan, index=df.index))
    equity = df.get("total_equity", pd.Series(np.nan, index=df.index))

    # Gross margin (flow/flow -- safe at native freq, risky at D)
    result, ism, inv = safe_ratio(gross_profit, revenue, "gross_margin")
    _set_ratio_columns(df, "gross_margin", result, ism, inv)

    # Operating margin (flow/flow)
    result, ism, inv = safe_ratio(ebit, revenue, "operating_margin")
    _set_ratio_columns(df, "operating_margin", result, ism, inv)

    # Net margin (flow/flow)
    result, ism, inv = safe_ratio(net_income, revenue, "net_margin")
    _set_ratio_columns(df, "net_margin", result, ism, inv)

    # ROE = net_income / total_equity (FLOW/STOCK)
    if freq in _NATIVE_RATIO_FREQS:
        # At Q/A/S: NI is at native period scale, equity is snapshot.
        # For Q: annualize NI by multiplying by 4 for true annual ROE.
        # For A: NI is already annual.
        _ni_for_roe = net_income
        if freq == "Q":
            _ni_for_roe = net_income * 4  # annualize quarterly NI
        elif freq == "S":
            _ni_for_roe = net_income * 2  # annualize semi-annual NI
        result, ism, inv = safe_ratio(_ni_for_roe, equity, "roe")
        _set_ratio_columns(df, "roe", result, ism, inv)
    else:
        # D/W/M: NI is daily rate, equity is stock -> ratio is ~1000x too small
        # Still compute for backward compat but warn
        result, ism, inv = safe_ratio(net_income, equity, "roe")
        _set_ratio_columns(df, "roe", result, ism, inv)
        logger.debug("roe computed at freq=%s (flow/stock, likely distorted)", freq)

    # F5 guard: reject economically impossible margins caused by statement
    # frequency mismatch (e.g., annual gross_profit / quarterly revenue).
    _MARGIN_CAPS = {
        "gross_margin": 1.5,
        "operating_margin": 1.0,
        "net_margin": 1.0,
    }
    for _m_name, _m_cap in _MARGIN_CAPS.items():
        if _m_name in df.columns:
            _impossible = df[_m_name].abs() > _m_cap
            if _impossible.any():
                logger.warning(
                    "F5 plausibility: %d/%d days of %s exceed +/-%.0f%% "
                    "(likely statement frequency mismatch), setting to NaN",
                    int(_impossible.sum()), len(df), _m_name, _m_cap * 100,
                )
                df.loc[_impossible, _m_name] = np.nan
                df.loc[_impossible, f"is_missing_{_m_name}"] = 1
                df.loc[_impossible, f"invalid_math_{_m_name}"] = 1

    # EBITDA approximation
    if "ebitda" not in df.columns or df["ebitda"].isna().all():
        ebit_proxy = df.get("ebit", df.get("operating_income", pd.Series(np.nan, index=df.index)))
        df["ebitda"] = ebit_proxy
        df["is_missing_ebitda"] = df["ebitda"].isna().astype(int)

    return df


# ---------------------------------------------------------------------------
# Valuation (optional)
# ---------------------------------------------------------------------------


def _compute_valuation(df: pd.DataFrame) -> pd.DataFrame:
    """Compute valuation metrics with frequency-aware formulas.

    Variables: pe_ratio_calc, earnings_yield_calc, ps_ratio_calc,
    enterprise_value, ev_to_ebitda, pb_ratio.

    Frequency-aware (per-frequency-formulas-for-all-18-broken-variables.md):
    - At Q: PE = close / (sum(eps_diluted, 4Q)), P/S = mcap / rev_ttm_4Q,
      EV/EBITDA = EV / ebitda_ttm_4Q  (Graham & Dodd 1934, Damodaran 2012)
    - At A: PE = close / eps_annual, P/S = mcap / rev_annual,
      EV/EBITDA = EV / ebitda_annual
    - At S: PE = close / (eps_s * 2), annualize semi-annual
    - At D/W/M: PE, P/S, EV/EBITDA use TTM values (may be distorted;
      will be overwritten by Q/A values after fusion)
    """
    freq = _CURRENT_FREQ
    close = df.get("close", pd.Series(np.nan, index=df.index))
    shares = df.get("shares_outstanding", pd.Series(np.nan, index=df.index))
    market_cap = df.get("market_cap", pd.Series(np.nan, index=df.index))
    net_income = df.get("net_income", pd.Series(np.nan, index=df.index))
    revenue = df.get("revenue", pd.Series(np.nan, index=df.index))
    ebitda = df.get("ebitda", pd.Series(np.nan, index=df.index))
    total_debt = df.get("total_debt_asof", pd.Series(np.nan, index=df.index))
    cash = df.get("cash_and_equivalents", pd.Series(np.nan, index=df.index))

    # ---- EPS calculation (frequency-aware) ----
    eps_from_filings = df.get("eps_diluted", pd.Series(np.nan, index=df.index))
    if eps_from_filings.isna().all():
        eps_from_filings = df.get("eps", pd.Series(np.nan, index=df.index))

    if freq in _NATIVE_RATIO_FREQS and eps_from_filings.notna().any():
        # At Q/A/S: eps_diluted is at native filing scale.
        # Annualize for PE: Q eps * 4, S eps * 2, A eps * 1
        _mult = _ANNUALIZE_MULT.get(freq, 1.0)
        eps_calc = eps_from_filings * _mult
        logger.debug("PE: annualizing eps_diluted by %.0fx for freq=%s", _mult, freq)
    elif eps_from_filings.notna().any():
        # D/W/M with filing EPS: use as-is (forward-filled filing values)
        eps_calc = eps_from_filings
    else:
        # Fallback: recompute from net_income / shares
        eps_calc = net_income / shares.where(shares.abs() > EPSILON, other=np.nan)

    # ---- PE = close / annualized_EPS ----
    result, ism, inv = safe_ratio(close, eps_calc, "pe_ratio_calc")
    _set_ratio_columns(df, "pe_ratio_calc", result, ism, inv)

    # P9: Synthetic PE fallback from operating_income
    if df.get("pe_ratio_calc") is not None and df["pe_ratio_calc"].isna().all():
        _oi = df.get("operating_income", pd.Series(np.nan, index=df.index))
        if _oi.notna().any() and shares.notna().any():
            _oi_eps = _oi / shares.where(shares.abs() > EPSILON, other=np.nan)
            if freq in _NATIVE_RATIO_FREQS:
                _oi_eps = _oi_eps * _ANNUALIZE_MULT.get(freq, 1.0)
            _synth_pe, _synth_ism, _synth_inv = safe_ratio(close, _oi_eps, "pe_ratio_calc")
            if _synth_pe.notna().any():
                df["pe_ratio_calc"] = _synth_pe
                df["is_missing_pe_ratio_calc"] = _synth_ism
                df["invalid_math_pe_ratio_calc"] = _synth_inv
                logger.info("Synthetic PE from operating_income: %.1f (latest)",
                           float(_synth_pe.dropna().iloc[-1]) if _synth_pe.notna().any() else 0)

    # ---- Earnings yield = annualized_EPS / close ----
    result, ism, inv = safe_ratio(eps_calc, close, "earnings_yield_calc")
    _set_ratio_columns(df, "earnings_yield_calc", result, ism, inv)

    # ---- P/S = market_cap / revenue_annualized ----
    if freq in _NATIVE_RATIO_FREQS:
        # At Q/A/S: annualize revenue for P/S
        _rev_annual = revenue * _ANNUALIZE_MULT.get(freq, 1.0)
        result, ism, inv = safe_ratio(market_cap, _rev_annual, "ps_ratio_calc")
    else:
        # D/W/M: try TTM revenue (may be distorted)
        _revenue_for_ps = df.get("revenue_ttm_asof", revenue)
        result, ism, inv = safe_ratio(market_cap, _revenue_for_ps, "ps_ratio_calc")
    _set_ratio_columns(df, "ps_ratio_calc", result, ism, inv)

    # ---- Enterprise value = market_cap + total_debt - cash (stock/stock, always OK) ----
    ev = market_cap + total_debt.fillna(0) - cash.fillna(0)
    ev_missing = market_cap.isna()
    df["enterprise_value"] = ev.where(~ev_missing, other=np.nan)
    df["is_missing_enterprise_value"] = ev_missing.astype(int)

    # ---- EV/EBITDA = EV / annualized_EBITDA ----
    if freq in _NATIVE_RATIO_FREQS:
        # At Q/A/S: annualize EBITDA (Damodaran 2012)
        _ebitda_annual = ebitda * _ANNUALIZE_MULT.get(freq, 1.0)
        result, ism, inv = safe_ratio(df["enterprise_value"], _ebitda_annual, "ev_to_ebitda")
    else:
        # D/W/M: use TTM EBITDA (may be distorted)
        _ebitda_for_ev = df.get("ebitda_ttm_asof", ebitda)
        result, ism, inv = safe_ratio(df["enterprise_value"], _ebitda_for_ev, "ev_to_ebitda")
    _set_ratio_columns(df, "ev_to_ebitda", result, ism, inv)

    # ---- P/B = market_cap / total_equity (MARKET/STOCK, always OK) ----
    equity = df.get("total_equity", pd.Series(np.nan, index=df.index))
    result, ism, inv = safe_ratio(market_cap, equity, "pb_ratio")
    _set_ratio_columns(df, "pb_ratio", result, ism, inv)

    return df


# ---------------------------------------------------------------------------
# Interest coverage (Tier 2 -- spec Section C.3)
# ---------------------------------------------------------------------------


def _compute_interest_coverage(df: pd.DataFrame) -> pd.DataFrame:
    """Compute interest coverage ratio.

    Variable: interest_coverage = EBIT / interest_expense.
    """
    ebit = df.get("ebit", pd.Series(np.nan, index=df.index))
    if ebit.isna().all():
        ebit = df.get("operating_income", pd.Series(np.nan, index=df.index))
    interest_expense = df.get("interest_expense", pd.Series(np.nan, index=df.index))

    result, ism, inv = safe_ratio(ebit, interest_expense, "interest_coverage")
    _set_ratio_columns(df, "interest_coverage", result, ism, inv)

    return df


# ---------------------------------------------------------------------------
# Return on assets (Tier 4)
# ---------------------------------------------------------------------------


def _compute_roa(df: pd.DataFrame) -> pd.DataFrame:
    """Compute return on assets (frequency-aware).

    Variable: roa = annualized_net_income / total_assets.

    At Q: roa = (NI_q * 4) / TA  (annualize quarterly NI, DuPont analysis)
    At A: roa = NI_a / TA  (both from same annual filing)
    At D: roa = NI_daily / TA  (BROKEN: NI is daily rate ~1000x too small)
    """
    freq = _CURRENT_FREQ
    net_income = df.get("net_income", pd.Series(np.nan, index=df.index))
    total_assets = df.get("total_assets", pd.Series(np.nan, index=df.index))

    if freq in _NATIVE_RATIO_FREQS:
        # Annualize NI for true annual ROA (Palepu & Healy 2019)
        _ni_annual = net_income * _ANNUALIZE_MULT.get(freq, 1.0)
        result, ism, inv = safe_ratio(_ni_annual, total_assets, "roa")
    else:
        # D/W/M: NI is daily rate / stock TA -> ratio ~1000x too small
        result, ism, inv = safe_ratio(net_income, total_assets, "roa")
        logger.debug("roa at freq=%s (flow/stock, likely distorted)", freq)
    _set_ratio_columns(df, "roa", result, ism, inv)

    return df


# ---------------------------------------------------------------------------
# TTM aggregates and growth rates (Tier 5)
# ---------------------------------------------------------------------------


def _compute_ttm_and_growth(df: pd.DataFrame) -> pd.DataFrame:
    """Compute trailing-twelve-month aggregates and YoY growth rates.

    Frequency-aware TTM computation:
    - At Q: TTM = sum of last 4 quarterly values (standard)
    - At A: TTM = the annual value itself (annual IS the TTM)
    - At S: TTM = sum of last 2 semi-annual values
    - At D/W/M: TTM via _rolling_4q_ttm on interpolated data (may be
      distorted; overwritten by Q/A values after fusion)

    YoY growth:
    - At Q: compare current TTM to TTM 4 periods ago
    - At A: compare current annual to previous annual (shift 1)
    - At D: compare current TTM to TTM ~252 trading days ago
    """
    freq = _CURRENT_FREQ
    revenue = df.get("revenue", pd.Series(np.nan, index=df.index))
    net_income = df.get("net_income", pd.Series(np.nan, index=df.index))
    ebitda = df.get("ebitda", pd.Series(np.nan, index=df.index))

    # --- TTM computation (frequency-dependent) ---
    if freq == "A":
        # Annual value IS the TTM -- no summation needed
        df["revenue_ttm_asof"] = revenue
        df["net_income_ttm_asof"] = net_income
        df["ebitda_ttm_asof"] = ebitda
    elif freq == "S":
        # Semi-annual: sum 2 periods = annual
        df["revenue_ttm_asof"] = revenue.rolling(2, min_periods=1).sum()
        df["net_income_ttm_asof"] = net_income.rolling(2, min_periods=1).sum()
        df["ebitda_ttm_asof"] = ebitda.rolling(2, min_periods=1).sum()
    elif freq == "Q":
        # Quarterly: sum 4 periods = annual (standard TTM)
        df["revenue_ttm_asof"] = revenue.rolling(4, min_periods=1).sum()
        df["net_income_ttm_asof"] = net_income.rolling(4, min_periods=1).sum()
        df["ebitda_ttm_asof"] = ebitda.rolling(4, min_periods=1).sum()
    else:
        # D/W/M: use _rolling_4q_ttm on forward-filled daily data
        # (detects quarterly transitions and sums distinct values)
        df["revenue_ttm_asof"] = _rolling_4q_ttm(revenue)
        df["net_income_ttm_asof"] = _rolling_4q_ttm(net_income)
        df["ebitda_ttm_asof"] = _rolling_4q_ttm(ebitda)

    df["is_missing_revenue_ttm_asof"] = df["revenue_ttm_asof"].isna().astype(int)
    df["is_missing_net_income_ttm_asof"] = df["net_income_ttm_asof"].isna().astype(int)
    df["is_missing_ebitda_ttm_asof"] = df["ebitda_ttm_asof"].isna().astype(int)

    # --- YoY growth (frequency-dependent shift) ---
    # At Q: shift 4 periods back.  At A: shift 1.  At D: shift 252 days.
    _yoy_shift = {"A": 1, "S": 2, "Q": 4, "M": 12, "W": 52, "D": 252}.get(freq, 252)

    rev_ttm = df["revenue_ttm_asof"]
    rev_prev = rev_ttm.shift(_yoy_shift)
    result, ism, inv = safe_ratio(rev_ttm - rev_prev, rev_prev.abs(), "revenue_growth_yoy")
    _set_ratio_columns(df, "revenue_growth_yoy", result, ism, inv)

    ni_ttm = df["net_income_ttm_asof"]
    ni_prev = ni_ttm.shift(_yoy_shift)
    result, ism, inv = safe_ratio(ni_ttm - ni_prev, ni_prev.abs(), "earnings_growth_yoy")
    _set_ratio_columns(df, "earnings_growth_yoy", result, ism, inv)

    return df


# ---------------------------------------------------------------------------
# Volume average (Tier 3)
# ---------------------------------------------------------------------------


def _compute_volume_avg(df: pd.DataFrame) -> pd.DataFrame:
    """Compute 21-day average volume.

    Variable: volume_avg_21d.
    """
    volume = df.get("volume", pd.Series(np.nan, index=df.index))
    df["volume_avg_21d"] = volume.rolling(window=21, min_periods=5).mean()
    df["is_missing_volume_avg_21d"] = df["volume_avg_21d"].isna().astype(int)

    return df


# ---------------------------------------------------------------------------
# Per-share metrics
# ---------------------------------------------------------------------------


def _compute_per_share(df: pd.DataFrame) -> pd.DataFrame:
    """Compute per-share metrics (frequency-aware).

    Variables: eps_calc, book_value_per_share, revenue_per_share.

    eps_calc and revenue_per_share are FLOW/STOCK -- at D freq the flow
    numerator is a daily rate, producing values ~1000x too small.
    At Q/A/S: use native-scale values (no annualization needed for eps_calc
    since it's per-share per-period, matching eps_diluted from filings).
    """
    freq = _CURRENT_FREQ
    shares = df.get("shares_outstanding", pd.Series(np.nan, index=df.index))
    net_income = df.get("net_income", pd.Series(np.nan, index=df.index))
    equity = df.get("total_equity", pd.Series(np.nan, index=df.index))
    revenue = df.get("revenue", pd.Series(np.nan, index=df.index))

    # EPS = net_income / shares (FLOW/STOCK)
    # At Q: this gives quarterly EPS (matches eps_diluted from filings)
    # At D: this gives daily NI rate / shares (garbage)
    result, ism, inv = safe_ratio(net_income, shares, "eps_calc")
    _set_ratio_columns(df, "eps_calc", result, ism, inv)

    # Book value per share
    result, ism, inv = safe_ratio(equity, shares, "book_value_per_share")
    _set_ratio_columns(df, "book_value_per_share", result, ism, inv)

    # Revenue per share
    result, ism, inv = safe_ratio(revenue, shares, "revenue_per_share")
    _set_ratio_columns(df, "revenue_per_share", result, ism, inv)

    return df


# ---------------------------------------------------------------------------
# Technical indicators (familiar from TradingView, Yahoo Finance)
# ---------------------------------------------------------------------------


def _compute_technical_indicators(df: pd.DataFrame) -> pd.DataFrame:
    """Compute common technical indicators from close price.

    Variables: sma_50, sma_200, rsi_14, macd, macd_signal,
    bollinger_upper, bollinger_lower.
    """
    close = df.get("close", pd.Series(np.nan, index=df.index))

    # Collect all new columns in a dict to avoid DataFrame fragmentation
    # (repeated df["col"] = ... triggers PerformanceWarning).
    _new: dict[str, pd.Series] = {}

    # Simple Moving Averages (50-day and 200-day)
    _new["sma_50"] = close.rolling(window=50, min_periods=10).mean()
    _new["is_missing_sma_50"] = _new["sma_50"].isna().astype(int)

    _new["sma_200"] = close.rolling(window=200, min_periods=50).mean()
    _new["is_missing_sma_200"] = _new["sma_200"].isna().astype(int)

    # RSI (14-day) -- Relative Strength Index
    delta = close.diff()
    gain = delta.clip(lower=0)
    loss = (-delta).clip(lower=0)
    avg_gain = gain.rolling(window=14, min_periods=5).mean()
    avg_loss = loss.rolling(window=14, min_periods=5).mean()
    rs = avg_gain / avg_loss.where(avg_loss.abs() > EPSILON, other=np.nan)
    _new["rsi_14"] = 100.0 - (100.0 / (1.0 + rs))
    _new["is_missing_rsi_14"] = _new["rsi_14"].isna().astype(int)

    # MACD (12/26/9 standard parameters)
    ema_12 = close.ewm(span=12, min_periods=8, adjust=False).mean()
    ema_26 = close.ewm(span=26, min_periods=18, adjust=False).mean()
    _new["macd"] = ema_12 - ema_26
    _new["macd_signal"] = _new["macd"].ewm(span=9, min_periods=5, adjust=False).mean()
    _new["is_missing_macd"] = _new["macd"].isna().astype(int)
    _new["is_missing_macd_signal"] = _new["macd_signal"].isna().astype(int)

    # Bollinger Bands (20-day, 2 standard deviations)
    sma_20 = close.rolling(window=20, min_periods=5).mean()
    std_20 = close.rolling(window=20, min_periods=5).std()
    _new["bollinger_upper"] = sma_20 + 2 * std_20
    _new["bollinger_lower"] = sma_20 - 2 * std_20
    _new["is_missing_bollinger_upper"] = _new["bollinger_upper"].isna().astype(int)
    _new["is_missing_bollinger_lower"] = _new["bollinger_lower"].isna().astype(int)

    # MACD Histogram
    _new["macd_histogram"] = _new["macd"] - _new["macd_signal"]

    # Bollinger Band Width (normalized)
    bb_mid = sma_20
    _new["bb_width"] = ((sma_20 + 2 * std_20) - (sma_20 - 2 * std_20)) / bb_mid.where(
        bb_mid.abs() > EPSILON, other=np.nan
    )
    _new["is_missing_bb_width"] = _new["bb_width"].isna().astype(int)

    # ADX (Average Directional Index) via ta library when available
    try:
        import ta
        high = df.get("high")
        low = df.get("low")
        if high is not None and low is not None and close.notna().sum() > 30:
            adx_indicator = ta.trend.ADXIndicator(high, low, close, window=14)
            _new["adx"] = adx_indicator.adx()
            _new["adx_pos"] = adx_indicator.adx_pos()
            _new["adx_neg"] = adx_indicator.adx_neg()
            _new["is_missing_adx"] = _new["adx"].isna().astype(int)

            # OBV (On-Balance Volume)
            volume = df.get("volume")
            if volume is not None:
                _new["obv"] = ta.volume.OnBalanceVolumeIndicator(close, volume).on_balance_volume()
                _new["is_missing_obv"] = _new["obv"].isna().astype(int)
    except ImportError:
        pass  # ta library not installed, skip ADX/OBV
    except Exception as _exc:
        logger.debug("ta library indicators failed: %s", _exc)

    # Assign all columns at once to avoid fragmentation
    df = pd.concat([df, pd.DataFrame(_new, index=df.index)], axis=1)

    return df


# ---------------------------------------------------------------------------
# Beta vs market benchmark
# ---------------------------------------------------------------------------


def _compute_beta(df: pd.DataFrame) -> pd.DataFrame:
    """Compute rolling 252-day beta vs market benchmark.

    ``beta_252d = Cov(stock_return, benchmark_return, 252) / Var(benchmark_return, 252)``

    Requires ``benchmark_return_1d`` column in the cache (merged in main.py
    Step 4 before derived_variables runs).  If missing, beta is NaN.

    App core idea Section F.1 Tier 3: ``beta_252d``.
    """
    stock_ret = df.get("return_1d")
    bench_ret = df.get("benchmark_return_1d")

    if stock_ret is None or bench_ret is None or bench_ret.isna().all():
        df["beta_252d"] = np.nan
        df["is_missing_beta_252d"] = 1
        return df

    # Rolling covariance / variance (min 60 days for meaningful estimate)
    cov = stock_ret.rolling(252, min_periods=60).cov(bench_ret)
    var = bench_ret.rolling(252, min_periods=60).var()

    # Safe division: avoid div-by-zero when benchmark variance is tiny
    safe_var = var.where(var.abs() > EPSILON)
    beta = cov / safe_var

    df["beta_252d"] = beta
    df["is_missing_beta_252d"] = beta.isna().astype(int)

    return df


# ---------------------------------------------------------------------------
# Recovery time (days from trough to previous peak)
# ---------------------------------------------------------------------------


def _compute_recovery_time(df: pd.DataFrame) -> pd.DataFrame:
    """Compute average and maximum drawdown recovery time.

    Recovery time = number of trading days from a drawdown trough until
    the price recovers to the pre-drawdown peak.  Standard risk metric
    from App core idea Section F.1 Category 3.

    Variables: recovery_time_avg, recovery_time_max, n_recovery_episodes.
    """
    close = df.get("close")
    if close is None or close.isna().all():
        df["recovery_time_avg"] = np.nan
        df["recovery_time_max"] = np.nan
        df["n_recovery_episodes"] = 0
        df["is_missing_recovery_time_avg"] = 1
        df["is_missing_recovery_time_max"] = 1
        df["is_missing_n_recovery_episodes"] = 0
        return df

    close_clean = close.dropna()
    if len(close_clean) < 10:
        df["recovery_time_avg"] = np.nan
        df["recovery_time_max"] = np.nan
        df["n_recovery_episodes"] = 0
        df["is_missing_recovery_time_avg"] = 1
        df["is_missing_recovery_time_max"] = 1
        df["is_missing_n_recovery_episodes"] = 0
        return df

    # Track drawdown episodes: peak -> trough -> recovery
    running_max = close_clean.expanding().max()
    in_drawdown = close_clean < running_max

    recovery_times: list[int] = []
    dd_start_idx: int | None = None
    peak_value: float = 0.0

    for i in range(len(close_clean)):
        val = float(close_clean.iloc[i])
        peak = float(running_max.iloc[i])

        if in_drawdown.iloc[i] and dd_start_idx is None:
            # Entered a drawdown
            dd_start_idx = i
            peak_value = peak
        elif dd_start_idx is not None and val >= peak_value:
            # Recovered to pre-drawdown peak
            recovery_days = i - dd_start_idx
            if recovery_days > 0:
                recovery_times.append(recovery_days)
            dd_start_idx = None

    n_episodes = len(recovery_times)
    if n_episodes > 0:
        avg_recovery = float(np.mean(recovery_times))
        max_recovery = float(np.max(recovery_times))
    else:
        avg_recovery = np.nan
        max_recovery = np.nan

    # Store as scalar columns (same value for all days -- summary metric)
    df["recovery_time_avg"] = avg_recovery
    df["is_missing_recovery_time_avg"] = int(pd.isna(avg_recovery))
    df["recovery_time_max"] = max_recovery
    df["is_missing_recovery_time_max"] = int(pd.isna(max_recovery))
    df["n_recovery_episodes"] = n_episodes
    df["is_missing_n_recovery_episodes"] = 0

    return df


# ---------------------------------------------------------------------------
# Earnings quality and alpha signals (Sloan 1996, Foster/Olsen/Shevlin 1984)
# ---------------------------------------------------------------------------


def _compute_earnings_quality_signals(df: pd.DataFrame) -> pd.DataFrame:
    """Compute accruals quality, SUE, and PEAD signals.

    Accruals (Sloan 1996):
        accruals = (net_income - operating_cash_flow) / total_assets
        High accruals = low earnings quality (future underperformance).
        accruals_signal = -accruals (lower = better, positive = buy signal).

    SUE -- Standardized Unexpected Earnings (Foster/Olsen/Shevlin 1984):
        eps_surprise = eps_ttm - eps_ttm.shift(4_quarters)
        sue_score = eps_surprise / rolling_std(eps_surprise)
        Positive SUE = beat expectations -> drift up (PEAD).

    PEAD -- Post-Earnings Announcement Drift (Bernard & Thomas 1989):
        pead_signal = sue_score * alpha_decay(days_since_filing)
        Strongest 0-30 days post-filing, decays exponentially.
    """
    eps = 1e-9

    # --- Accruals signal (Sloan 1996, TAR) ---
    # accruals = (NI - OCF) / TA  (FLOW - FLOW) / STOCK
    # At Q/A/S: NI and OCF are at native period scale -> correct
    # At D: NI and OCF are daily rates, TA is stock -> ratio ~1000x too small
    freq = _CURRENT_FREQ
    ni = df.get("net_income")
    ocf = df.get("operating_cash_flow")
    ta = df.get("total_assets")
    if ni is not None and ocf is not None and ta is not None:
        safe_ta = ta.where(ta.abs() > eps)
        _ni_adj = ni
        _ocf_adj = ocf
        if freq in _NATIVE_RATIO_FREQS:
            # Annualize flow variables for comparable accruals ratio
            _mult = _ANNUALIZE_MULT.get(freq, 1.0)
            _ni_adj = ni * _mult
            _ocf_adj = ocf * _mult
        accruals = (_ni_adj - _ocf_adj) / safe_ta
        df["accruals"] = accruals
        df["accruals_signal"] = -accruals
    else:
        df["accruals"] = float("nan")
        df["accruals_signal"] = float("nan")

    # --- SUE score ---
    # Use EPS TTM (or net_income_ttm as proxy) shifted by ~252 days (4 quarters)
    eps_col = df.get("eps_calc")
    if eps_col is None:
        eps_col = df.get("net_income_ttm_asof")
    if eps_col is not None and eps_col.notna().sum() >= 10:
        eps_surprise = eps_col - eps_col.shift(252)
        surprise_std = eps_surprise.rolling(504, min_periods=60).std()
        safe_std = surprise_std.where(surprise_std.abs() > eps)
        sue = eps_surprise / safe_std
        df["eps_surprise_proxy"] = eps_surprise
        df["sue_score"] = sue
    else:
        df["eps_surprise_proxy"] = float("nan")
        df["sue_score"] = float("nan")

    # --- PEAD signal (decay-weighted SUE) ---
    # Decay based on distance from nearest filing date change
    sue = df.get("sue_score")
    if sue is not None and sue.notna().any():
        # Estimate days since last filing change
        # Use revenue changes as proxy for filing dates
        revenue = df.get("revenue")
        if revenue is not None:
            filing_change = (revenue.diff().abs() > eps).astype(int)
            days_since_filing = filing_change.groupby(
                filing_change.cumsum()
            ).cumcount()
        else:
            days_since_filing = pd.Series(30, index=df.index)  # default 30 days

        # Exponential decay: half-life of 30 business days
        decay = np.exp(-0.693 * days_since_filing / 30)
        df["pead_signal"] = sue * decay
    else:
        df["pead_signal"] = float("nan")

    return df


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

# ---------------------------------------------------------------------------
# A2: Realized vol decomposition (continuous + jump)
# Barndorff-Nielsen & Shephard (2004) bipower variation.
# Separates "normal" diffusion vol from sudden shock (jump) events.
# ---------------------------------------------------------------------------

def _compute_realized_vol_decomposition(df: pd.DataFrame) -> pd.DataFrame:
    """Decompose realized volatility into continuous and jump components.

    Uses bipower variation (BV) to estimate the continuous component.
    Jump variation (JV) = RV - BV captures sudden exogenous shocks
    (tariff announcements, earnings surprises, policy changes).

    Reference: Barndorff-Nielsen & Shephard (2004).
    """
    if "return_1d" not in df.columns:
        return df

    returns = df["return_1d"]
    if returns.notna().sum() < 30:
        return df

    # Realized variance: rolling sum of squared returns (21-day window)
    rv_21d = (returns ** 2).rolling(21, min_periods=10).sum()

    # Bipower variation: rolling sum of |r_t| * |r_{t-1}| * pi/2
    abs_ret = returns.abs()
    abs_ret_lag = abs_ret.shift(1)
    bv_21d = (np.pi / 2.0) * (abs_ret * abs_ret_lag).rolling(21, min_periods=10).sum()

    # Jump variation: max(RV - BV, 0)
    jv_21d = (rv_21d - bv_21d).clip(lower=0.0)

    # Realized vol (annualized sqrt of RV)
    df["realized_vol_21d"] = np.sqrt(rv_21d.clip(lower=0.0))

    # Continuous vol component (sqrt of BV)
    df["continuous_vol_21d"] = np.sqrt(bv_21d.clip(lower=0.0))

    # Jump vol component (sqrt of JV)
    df["jump_vol_21d"] = np.sqrt(jv_21d)

    # Jump ratio: what fraction of total vol comes from jumps
    # High jump_ratio = exogenous shock environment
    _total = rv_21d.clip(lower=1e-12)
    df["jump_ratio_21d"] = jv_21d / _total

    # Jump spike flag: jump_ratio above P90 of its own history
    if df["jump_ratio_21d"].notna().sum() > 50:
        _p90 = df["jump_ratio_21d"].quantile(0.90)
        df["jump_spike_flag"] = (df["jump_ratio_21d"] > _p90).astype(int)
    else:
        df["jump_spike_flag"] = 0

    return df


# ---------------------------------------------------------------------------
# G2: Merton distance-to-default
# Structural credit model: equity = call option on firm assets.
# Low DD = close to default boundary = equity vol will increase.
# ---------------------------------------------------------------------------

def _compute_merton_distance_to_default(df: pd.DataFrame) -> pd.DataFrame:
    """Compute Merton (1974) distance-to-default from equity and debt data.

    DD = (log(V/D) + (mu - 0.5*sigma_V^2)*T) / (sigma_V * sqrt(T))

    Where V = equity + debt (proxy for asset value), D = total debt,
    sigma_V is estimated from equity volatility via leverage adjustment.
    """
    has_required = all(
        c in df.columns and df[c].notna().any()
        for c in ("close", "volatility_21d")
    )
    has_debt = "total_debt" in df.columns or "total_debt_asof" in df.columns

    if not has_required or not has_debt:
        return df

    _debt_col = "total_debt" if "total_debt" in df.columns else "total_debt_asof"
    debt = df[_debt_col].ffill().fillna(0)
    equity_vol = df["volatility_21d"].ffill().fillna(0.02)

    # Market cap proxy (or use actual if available)
    if "market_cap" in df.columns and df["market_cap"].notna().any():
        mkt_cap = df["market_cap"].ffill()
    elif "shares_outstanding" in df.columns and df["shares_outstanding"].notna().any():
        mkt_cap = df["close"] * df["shares_outstanding"].ffill()
    else:
        # Rough proxy: last close * 1B shares (very approximate)
        mkt_cap = df["close"] * 1e9

    # Asset value proxy: equity + debt
    asset_value = mkt_cap + debt

    # Asset volatility via leverage adjustment (Merton approximation)
    # sigma_V = sigma_E * E / V
    leverage = mkt_cap / asset_value.clip(lower=1.0)
    asset_vol = equity_vol * leverage

    # Distance to default (T=1 year, mu=0 for risk-neutral)
    _T = 1.0
    _safe_debt = debt.clip(lower=1.0)
    _safe_vol = asset_vol.clip(lower=0.001)
    dd = (np.log(asset_value / _safe_debt) + (-0.5 * _safe_vol ** 2) * _T) / (_safe_vol * np.sqrt(_T))

    # Clip extreme values
    df["merton_dd"] = dd.clip(-10.0, 20.0)

    # DD change (declining DD = increasing default risk = rising vol)
    df["merton_dd_change_21d"] = df["merton_dd"] - df["merton_dd"].shift(21)

    return df


# ---------------------------------------------------------------------------
# Stage 18: Market Microstructure Signals
# Corwin-Schultz spread (2012, JF), Kyle lambda (1985), Parkinson vol
# (1980), Yang-Zhang vol (2000), volume clock intensity.
# Ref implementations: RiskLabAI/corwin_schultz.py, mgao6767/frds,
# Jensenberg/volatility-and-option, scikit-portfolio.
# ---------------------------------------------------------------------------

_CS_DENOM = 3 - 2 * (2 ** 0.5)  # Corwin-Schultz constant


def _compute_microstructure_signals(df: pd.DataFrame) -> pd.DataFrame:
    """Market microstructure features from daily OHLCV data.

    1. Corwin-Schultz bid-ask spread estimator (Corwin & Schultz 2012)
    2. Kyle lambda -- daily price impact proxy (Kyle 1985)
    3. Parkinson volatility -- range-based, 5x more efficient (Parkinson 1980)
    4. Yang-Zhang volatility -- overnight + intraday composite (Yang & Zhang 2000)
    5. Volume clock intensity -- z-scored volume anomaly (AFML concept)
    """
    has_ohlcv = all(
        c in df.columns and df[c].notna().sum() > 5
        for c in ("open", "high", "low", "close", "volume")
    )
    if not has_ohlcv:
        return df

    high = df["high"].astype(float)
    low = df["low"].astype(float)
    opn = df["open"].astype(float)
    close = df["close"].astype(float)
    volume = df["volume"].astype(float)
    eps = EPSILON

    # --- Corwin-Schultz spread (RiskLabAI pattern) ---
    log_hl_sq = np.log(high / low.clip(lower=eps)) ** 2
    cs_beta = log_hl_sq.rolling(2).sum().rolling(20).mean()
    h2 = high.rolling(2).max()
    l2 = low.rolling(2).min()
    cs_gamma = np.log(h2 / l2.clip(lower=eps)) ** 2
    cs_t1 = (2 ** 0.5 - 1) * np.sqrt(cs_beta.clip(lower=0)) / _CS_DENOM
    cs_t2 = np.sqrt((cs_gamma / _CS_DENOM).clip(lower=0))
    cs_alpha = (cs_t1 - cs_t2).clip(lower=0)
    df["corwin_schultz_spread"] = 2 * (np.exp(cs_alpha) - 1) / (1 + np.exp(cs_alpha))

    # --- Kyle lambda (frds pattern: OLS of |return| on signed volume) ---
    if "return_1d" in df.columns:
        ret = df["return_1d"].fillna(0)
        signed_vol = volume * np.sign(ret)
        # Rolling 21-day regression slope via cov/var
        cov_rv = ret.rolling(21, min_periods=10).cov(signed_vol)
        var_sv = signed_vol.rolling(21, min_periods=10).var()
        df["kyle_lambda"] = (cov_rv / var_sv.clip(lower=eps)) * 1e6

    # --- Parkinson volatility (scikit-portfolio pattern) ---
    log_hl = np.log(high / low.clip(lower=eps))
    df["parkinson_vol_21d"] = np.sqrt(
        (log_hl ** 2).rolling(21, min_periods=10).sum()
        / (4 * 21 * np.log(2))
    )

    # --- Yang-Zhang volatility (Jensenberg pattern) ---
    # Overnight: log(open_t / close_{t-1})
    oc = np.log(opn / close.shift(1).clip(lower=eps))
    oc_mean = oc.rolling(21, min_periods=10).mean()
    oc_var = ((oc - oc_mean) ** 2).rolling(21, min_periods=10).mean()
    # Close-to-open: log(close_t / open_t)
    co = np.log(close / opn.clip(lower=eps))
    co_mean = co.rolling(21, min_periods=10).mean()
    co_var = ((co - co_mean) ** 2).rolling(21, min_periods=10).mean()
    # Rogers-Satchell component
    rs = (
        np.log(high / close.clip(lower=eps)) * np.log(high / opn.clip(lower=eps))
        + np.log(low / close.clip(lower=eps)) * np.log(low / opn.clip(lower=eps))
    ).rolling(21, min_periods=10).mean()
    # Optimal blend factor (from Yang-Zhang 2000 paper)
    n = 21
    k = 0.34 / (1.34 + (n + 1) / max(n - 1, 1))
    yz_var = oc_var + k * co_var + (1 - k) * rs.clip(lower=0)
    df["yang_zhang_vol_21d"] = np.sqrt(yz_var.clip(lower=0))

    # --- Volume clock intensity (z-scored volume anomaly) ---
    if "volume_avg_21d" in df.columns:
        vol_std = volume.rolling(21, min_periods=5).std().clip(lower=eps)
        df["volume_clock_intensity"] = (
            (volume - df["volume_avg_21d"]) / vol_std
        )

    return df


# ---------------------------------------------------------------------------
# Stage 19: Stationarity & Time-Series Structure Features
# Fractional diff (Lopez de Prado 2018), rolling Hurst (Mandelbrot 1971),
# autocorrelation at lag 1 and 5 (Lo & MacKinlay 1988).
# Ref implementations: OmniQuant/advanced_features.py, Nixtla/tsfeatures.
# ---------------------------------------------------------------------------

def _frac_diff_weights(d: float, max_window: int = 100) -> np.ndarray:
    """Compute fractional differentiation weights (Lopez de Prado AFML Ch.5).

    Uses a fixed window approach (AFML Snippet 5.3) capped at max_window
    to keep the convolution practical for typical 504-day caches.
    """
    weights = [1.0]
    for k_idx in range(1, max_window):
        w = -weights[-1] * (d - k_idx + 1) / k_idx
        weights.append(w)
    return np.array(weights[::-1])


def _hurst_rs(x: np.ndarray) -> float:
    """Hurst exponent via R/S analysis (Nixtla/tsfeatures compact version)."""
    n = x.size
    if n < 20:
        return float("nan")
    t = np.arange(1, n + 1)
    y = x.cumsum()
    mean_t = y / t
    s_t = np.sqrt(
        np.array([np.mean((x[: i + 1] - mean_t[i]) ** 2) for i in range(n)])
    )
    r_t = np.array(
        [np.ptp(y[: i + 1] - t[: i + 1] * mean_t[i]) for i in range(n)]
    )
    with np.errstate(invalid="ignore", divide="ignore"):
        r_s = np.log(r_t / np.where(s_t > 0, s_t, np.nan))[1:]
    n_log = np.log(t)[1:]
    valid = np.isfinite(r_s) & np.isfinite(n_log)
    if valid.sum() < 5:
        return float("nan")
    a = np.column_stack((n_log[valid], np.ones(valid.sum())))
    result = np.linalg.lstsq(a, r_s[valid], rcond=-1)
    return float(np.clip(result[0][0], 0.0, 1.0))


def _compute_stationarity_features(df: pd.DataFrame) -> pd.DataFrame:
    """Time-series structure features: fractional diff, Hurst, autocorrelation.

    1. close_frac_diff -- fractionally differentiated close (d=0.4)
    2. hurst_exponent_rolling -- rolling 63d Hurst via R/S analysis
    3. autocorr_lag1 -- rolling 63d lag-1 autocorrelation
    4. autocorr_lag5 -- rolling 63d lag-5 autocorrelation (weekly reversal)
    """
    # --- Fractional differentiation of close (OmniQuant pattern) ---
    close = df.get("close")
    if close is not None and close.notna().sum() > 30:
        weights = _frac_diff_weights(d=0.4, max_window=100)
        w_len = len(weights)
        vals = close.values.astype(float)
        fd = np.full(len(vals), np.nan)
        for i in range(w_len, len(vals)):
            window = vals[i - w_len + 1: i + 1]
            if not np.any(np.isnan(window)):
                fd[i] = np.dot(weights, window)
        df["close_frac_diff"] = fd

    # --- Rolling Hurst exponent (tsfeatures pattern) ---
    ret = df.get("return_1d")
    if ret is not None and ret.notna().sum() > 70:
        _window = 63
        hurst_vals = np.full(len(df), np.nan)
        ret_arr = ret.fillna(0).values
        for i in range(_window, len(ret_arr)):
            hurst_vals[i] = _hurst_rs(ret_arr[i - _window: i])
        df["hurst_exponent_rolling"] = hurst_vals

    # --- Autocorrelation at lag 1 and 5 ---
    if ret is not None and ret.notna().sum() > 70:
        df["autocorr_lag1"] = ret.rolling(63, min_periods=30).apply(
            lambda x: x.autocorr(lag=1) if len(x) > 5 else float("nan"),
            raw=False,
        )
        df["autocorr_lag5"] = ret.rolling(63, min_periods=30).apply(
            lambda x: x.autocorr(lag=5) if len(x) > 10 else float("nan"),
            raw=False,
        )

    # --- Momentum 12-1 (Jegadeesh & Titman 1993; Novy-Marx 2012) ---
    if close is not None and close.notna().sum() > 260:
        ret_12m = close / close.shift(252) - 1
        ret_1m = close / close.shift(21) - 1
        df["momentum_12_1"] = ret_12m - ret_1m

    # --- Idiosyncratic volatility (Ang et al. 2006) ---
    if ret is not None and "beta_252d" in df.columns:
        bench = df.get("benchmark_return_1d")
        if bench is not None and bench.notna().sum() > 30:
            residual = ret - df["beta_252d"].fillna(1.0) * bench.fillna(0)
            df["idiosyncratic_vol_63d"] = residual.rolling(63, min_periods=20).std()

    # --- Earnings revision proxy (Chan, Jegadeesh & Lakonishok 1996) ---
    eps_col = df.get("eps_calc")
    if eps_col is None:
        eps_col = df.get("net_income_ttm_asof")
    if eps_col is not None and eps_col.notna().sum() > 70:
        shifted = eps_col.shift(63)
        safe_shifted = shifted.where(shifted.abs() > EPSILON)
        df["earnings_revision_proxy"] = (eps_col - shifted) / safe_shifted.abs()

    return df


# ---------------------------------------------------------------------------
# Stage 20: Credit Risk & Distress Signals
# Cash burn rate, debt maturity pressure, cash conversion cycle,
# Altman Z momentum, covenant proximity score.
# ---------------------------------------------------------------------------

def _compute_credit_signals(df: pd.DataFrame) -> pd.DataFrame:
    """Credit risk and distress early-warning features.

    1. cash_burn_rate_monthly -- monthly cash depletion when OCF negative
    2. debt_maturity_pressure -- fraction of debt due within 1 year
    3. cash_conversion_cycle -- DSO + DIO - DPO (Richards & Laughlin 1980)
    4. altman_z_momentum_63d -- 63-day change in Altman Z-score
    5. covenant_proximity_score -- max distance-to-threshold (0-1) across triggers
    """
    eps = EPSILON

    # --- Cash burn rate (monthly) ---
    # OCF is a flow variable: at Q = quarterly total, at A = annual total.
    # Monthly burn = -OCF / months_in_period.
    # At Q: -OCF_q / 3.  At A: -OCF_a / 12.  At D: -OCF_daily * 30 (scale up).
    freq = _CURRENT_FREQ
    _months_in_period = {"A": 12, "S": 6, "Q": 3, "M": 1, "W": 7/30, "D": 1/30}
    ocf = df.get("operating_cash_flow")
    if ocf is not None:
        _mip = _months_in_period.get(freq, 1/30)
        if freq in _NATIVE_RATIO_FREQS:
            # At Q/A/S: OCF is period total, divide by months in period
            df["cash_burn_rate_monthly"] = np.maximum(0.0, -ocf.astype(float)) / max(_mip, 0.01)
        else:
            # At D: OCF is daily rate, multiply by 30 for monthly
            df["cash_burn_rate_monthly"] = np.maximum(0.0, -ocf.astype(float)) * 30.0

    # --- Debt maturity pressure (short-term / total) ---
    std = df.get("short_term_debt")
    td = df.get("total_debt_asof")
    if std is not None and td is not None:
        safe_td = td.where(td.abs() > eps)
        df["debt_maturity_pressure"] = std.astype(float) / safe_td.astype(float)

    # --- Cash conversion cycle = DSO + DIO - DPO ---
    # Frequency-aware period days (Richards & Laughlin 1980):
    # At Q: revenue/90, At A: revenue/365, At S: revenue/180
    # At D/W/M: use 90 (quarterly approximation, may be distorted)
    freq = _CURRENT_FREQ
    _period_days = _PERIOD_DAYS.get(freq, 90)
    if freq == "D":
        _period_days = 90  # assume quarterly filing cadence for daily cache

    revenue = df.get("revenue")
    receivables = df.get("receivables")
    inventory = df.get("inventory")
    payables = df.get("payables")
    cogs = df.get("cost_of_revenue")
    if cogs is None or (cogs is not None and cogs.isna().all()):
        gp = df.get("gross_profit")
        if revenue is not None and gp is not None:
            cogs = (revenue.astype(float) - gp.astype(float)).clip(lower=eps)

    if revenue is not None:
        rev_per_day = revenue.astype(float) / float(_period_days)
        safe_rev_per_day = rev_per_day.where(rev_per_day.abs() > eps)
        if receivables is not None:
            df["dso"] = receivables.astype(float) / safe_rev_per_day
    if cogs is not None:
        cogs_per_day = cogs.astype(float) / float(_period_days)
        safe_cogs_per_day = cogs_per_day.where(cogs_per_day.abs() > eps)
        if inventory is not None:
            df["dio"] = inventory.astype(float) / safe_cogs_per_day
        if payables is not None:
            df["dpo"] = payables.astype(float) / safe_cogs_per_day

    dso = df.get("dso")
    dio = df.get("dio")
    dpo = df.get("dpo")
    if dso is not None and dio is not None and dpo is not None:
        df["cash_conversion_cycle"] = dso + dio - dpo
    elif dso is not None and dpo is not None:
        # No inventory data -- partial CCC
        df["cash_conversion_cycle"] = dso - dpo

    # --- Altman Z momentum (directional change over 63 days) ---
    az = df.get("fh_altman_z_score")
    if az is not None and az.notna().sum() > 70:
        df["altman_z_momentum_63d"] = az - az.shift(63)

    # --- Covenant proximity score ---
    # Max of normalized distances to each survival trigger threshold (0-1)
    _triggers = [
        ("current_ratio", 1.0, "below"),
        ("debt_to_equity_abs", 3.0, "above"),
        ("fcf_yield", 0.0, "below"),
        ("drawdown_252d", -0.40, "below"),
    ]
    try:
        from operator1.scoring_weights import get_weight
        _triggers = [
            ("current_ratio", float(get_weight("survival_thresholds.current_ratio", 1.0)), "below"),
            ("debt_to_equity_abs", float(get_weight("survival_thresholds.debt_to_equity", 3.0)), "above"),
            ("fcf_yield", float(get_weight("survival_thresholds.fcf_yield", 0.0)), "below"),
            ("drawdown_252d", float(get_weight("survival_thresholds.drawdown_252d", -0.40)), "below"),
        ]
    except Exception:
        pass  # use defaults

    proximity_parts = []
    for col_name, threshold, direction in _triggers:
        series = df.get(col_name)
        if series is None or series.isna().all():
            continue
        s = series.astype(float)
        if direction == "below":
            # Score = how close actual is to falling below threshold (0=safe, 1=at threshold)
            prox = ((threshold - s) / abs(threshold) if abs(threshold) > eps else (threshold - s)).clip(0, 1)
        else:
            # Score = how close actual is to exceeding threshold
            prox = ((s - threshold) / abs(threshold) if abs(threshold) > eps else (s - threshold)).clip(0, 1)
        proximity_parts.append(prox)

    if proximity_parts:
        stacked = pd.concat(proximity_parts, axis=1)
        df["covenant_proximity_score"] = stacked.max(axis=1)

    return df


# ---------------------------------------------------------------------------
# Stage 22: Tail Risk & Higher-Moment Features
# Skewness, kurtosis, tail ratio, max daily loss, vol-of-vol.
# Ref: Harvey & Siddique 2000, Dittmar 2002, quantstats/empyrical.
# ---------------------------------------------------------------------------

def _compute_tail_risk_features(df: pd.DataFrame) -> pd.DataFrame:
    """Higher-moment and tail-risk features from return distribution.

    1. skewness_63d -- rolling return skewness (Harvey & Siddique 2000)
    2. kurtosis_63d -- rolling return excess kurtosis (Dittmar 2002)
    3. tail_ratio_63d -- |P95/P5| asymmetry (empyrical convention)
    4. max_daily_loss_63d -- worst single-day return in 63d window
    5. vol_of_vol_21d -- volatility of volatility (Cont & da Fonseca 2002)
    """
    ret = df.get("return_1d")
    if ret is None or ret.notna().sum() < 70:
        return df

    df["skewness_63d"] = ret.rolling(63, min_periods=30).skew()
    df["kurtosis_63d"] = ret.rolling(63, min_periods=30).kurt()

    # Tail ratio: |P95 / P5| -- >1 means right tail larger (positive skew)
    p95 = ret.rolling(63, min_periods=30).quantile(0.95)
    p05 = ret.rolling(63, min_periods=30).quantile(0.05)
    safe_p05 = p05.where(p05.abs() > EPSILON)
    df["tail_ratio_63d"] = (p95 / safe_p05).abs()

    df["max_daily_loss_63d"] = ret.rolling(63, min_periods=10).min()

    # Vol-of-vol: std of volatility_21d over 21 days
    vol = df.get("volatility_21d")
    if vol is not None and vol.notna().sum() > 30:
        df["vol_of_vol_21d"] = vol.rolling(21, min_periods=10).std()

    return df


# ---------------------------------------------------------------------------
# Stage 24: Forensic Accounting Signals
# Revenue-receivables divergence (Lev & Thiagarajan 1993), capex/depreciation
# ratio (Sloan 1996), soft asset ratio (Barton & Simko 2002), OCF ratio
# (Dechow & Dichev 2002).
# ---------------------------------------------------------------------------

def _compute_forensic_signals(df: pd.DataFrame) -> pd.DataFrame:
    """Forensic accounting red-flag signals from financial statements.

    1. revenue_receivables_divergence -- channel stuffing detector
    2. capex_depreciation_ratio -- investment quality proxy
    3. soft_asset_ratio -- manipulation risk (high = easy to inflate)
    4. ocf_ratio -- cash conversion quality (OCF / NI)
    """
    eps = EPSILON

    # --- Revenue-receivables divergence (Lev & Thiagarajan 1993) ---
    # YoY pct_change: shift depends on frequency
    freq = _CURRENT_FREQ
    _yoy_shift = {"A": 1, "S": 2, "Q": 4, "M": 12, "W": 52, "D": 252}.get(freq, 252)
    revenue = df.get("revenue")
    receivables = df.get("receivables")
    if revenue is not None and receivables is not None:
        rev_g = revenue.astype(float).pct_change(_yoy_shift)
        rec_g = receivables.astype(float).pct_change(_yoy_shift)
        df["revenue_receivables_divergence"] = rev_g - rec_g  # positive = healthy

    # --- CapEx / Depreciation ratio (Sloan 1996) ---
    capex = df.get("capex")
    ebitda_val = df.get("ebitda")
    ebit_val = df.get("ebit")
    if ebit_val is None:
        ebit_val = df.get("operating_income")
    if capex is not None and ebitda_val is not None and ebit_val is not None:
        # Depreciation proxy = EBITDA - EBIT
        depreciation = (ebitda_val.astype(float) - ebit_val.astype(float)).clip(lower=eps)
        df["capex_depreciation_ratio"] = capex.astype(float).abs() / depreciation

    # --- Soft asset ratio (Barton & Simko 2002) ---
    total_assets = df.get("total_assets")
    cash = df.get("cash_and_equivalents")
    if total_assets is not None:
        hard_assets = cash.fillna(0).astype(float) if cash is not None else 0.0
        ta = total_assets.astype(float)
        safe_ta = ta.where(ta.abs() > eps)
        df["soft_asset_ratio"] = (ta - hard_assets) / safe_ta

    # --- OCF ratio -- operating cash flow / net income (Dechow & Dichev 2002) ---
    ocf = df.get("operating_cash_flow")
    ni = df.get("net_income")
    if ocf is not None and ni is not None:
        safe_ni = ni.astype(float).where(ni.astype(float).abs() > eps)
        df["ocf_ratio"] = ocf.astype(float) / safe_ni

    return df


# Ordered pipeline of computation stages
_COMPUTE_STAGES = (
    _compute_returns_and_risk,
    _compute_solvency,
    _compute_liquidity,
    _compute_interest_coverage,
    _compute_cash_reality,
    _compute_profitability,
    _compute_roa,
    _compute_valuation,
    _compute_ttm_and_growth,
    _compute_volume_avg,
    _compute_per_share,
    _compute_technical_indicators,
    _compute_recovery_time,
    _compute_beta,
    _compute_earnings_quality_signals,
    _compute_realized_vol_decomposition,  # A2: vol decomposition
    _compute_merton_distance_to_default,  # G2: Merton DD
    _compute_microstructure_signals,      # Stage 18: market microstructure
    _compute_stationarity_features,       # Stage 19: stationarity + factor
    _compute_credit_signals,              # Stage 20: credit risk
    _compute_tail_risk_features,          # Stage 22: tail risk
    _compute_forensic_signals,            # Stage 24: forensic accounting
)

# All derived variable names (for inspection / downstream reference)
DERIVED_VARIABLES: tuple[str, ...] = (
    # Returns / risk
    "return_1d", "log_return_1d", "volatility_21d", "drawdown_252d",
    # Solvency
    "total_debt_asof", "debt_to_equity_signed", "debt_to_equity_abs",
    "net_debt", "net_debt_to_ebitda",
    # Interest coverage
    "interest_coverage",
    # Liquidity
    "current_ratio", "quick_ratio", "cash_ratio",
    # Cash reality
    "free_cash_flow", "free_cash_flow_ttm_asof", "fcf_yield",
    # Profitability
    "gross_margin", "operating_margin", "net_margin", "roe", "roa",
    # Valuation
    "pe_ratio_calc", "earnings_yield_calc", "ps_ratio_calc",
    "enterprise_value", "ev_to_ebitda", "pb_ratio",
    # TTM & growth
    "revenue_ttm_asof", "net_income_ttm_asof", "ebitda_ttm_asof",
    "revenue_growth_yoy", "earnings_growth_yoy",
    # Volume
    "volume_avg_21d",
    # Per-share
    "eps_calc", "book_value_per_share", "revenue_per_share",
    # Technical indicators
    "sma_50", "sma_200", "rsi_14", "macd", "macd_signal",
    "bollinger_upper", "bollinger_lower",
    # Beta
    "beta_252d",
    # Recovery time
    "recovery_time_avg", "recovery_time_max", "n_recovery_episodes",
    # Earnings quality & alpha signals
    "accruals", "accruals_signal", "eps_surprise_proxy", "sue_score", "pead_signal",
    # Stage 18: Microstructure signals
    "corwin_schultz_spread", "kyle_lambda",
    "parkinson_vol_21d", "yang_zhang_vol_21d", "volume_clock_intensity",
    # Stage 19: Stationarity & factor features
    "close_frac_diff", "hurst_exponent_rolling", "autocorr_lag1", "autocorr_lag5",
    "momentum_12_1", "earnings_revision_proxy",
    # idiosyncratic_vol_63d: conditionally produced (needs benchmark_return_1d)
    # Stage 20: Credit signals
    "cash_burn_rate_monthly", "debt_maturity_pressure",
    "cash_conversion_cycle", "covenant_proximity_score",
    # altman_z_momentum_63d: conditionally produced (needs fh_altman_z_score from Step 5d)
    # Stage 22: Tail risk
    "skewness_63d", "kurtosis_63d", "tail_ratio_63d",
    "max_daily_loss_63d", "vol_of_vol_21d",
    # Stage 24: Forensic accounting
    "revenue_receivables_divergence", "capex_depreciation_ratio",
    "soft_asset_ratio", "ocf_ratio",
)


def compute_derived_variables(df: pd.DataFrame, freq: str = "D") -> pd.DataFrame:
    """Compute all derived decision variables for an entity cache.

    Parameters
    ----------
    df:
        DataFrame from cache builder.  At daily frequency this is the
        OHLCV spine with forward-filled financials.  At Q/A frequency
        this is a per-period cache from ``build_cache_from_raw_filings``.
    freq:
        Data frequency: ``"D"`` (daily), ``"W"`` (weekly), ``"M"``
        (monthly), ``"Q"`` (quarterly), ``"S"`` (semi-annual),
        ``"A"`` (annual).  Controls which ratio formulas run:

        - At **Q/A/S** (native filing freq): all ratios computed using
          filing-scale values (PE, EV/EBITDA, ROA, margins, TTM, etc.)
        - At **D/W/M** (interpolated): cross-type ratios (market/flow,
          stock/flow) are SKIPPED -- they produce garbage on interpolated
          daily rates.  Only OHLCV-native and stock/stock ratios run.
          The skipped ratios will be forward-filled from Q/A results
          after multi-frequency fusion.

    Returns
    -------
    pd.DataFrame
        Input DataFrame augmented with derived variables and their
        companion ``is_missing_*`` / ``invalid_math_*`` flags.
    """
    global _CURRENT_FREQ
    _CURRENT_FREQ = freq.upper() if freq else "D"
    logger.info("Computing derived variables at freq=%s", _CURRENT_FREQ)

    result = df.copy()
    for stage in _COMPUTE_STAGES:
        try:
            result = stage(result)
        except Exception as exc:
            logger.warning(
                "Derived variable stage %s failed: %s", stage.__name__, exc,
            )
    # Defragment after adding ~39 columns one-by-one across stages.
    # Without this, downstream consumers trigger PerformanceWarning from
    # pandas 2.x fragmented DataFrame internals.
    result = result.copy()

    # Deduplicate columns: when compute_derived_variables() is called
    # multiple times on the same cache (e.g. backtest_runner re-runs
    # features after loading state), technical indicator columns (sma_50,
    # rsi_14, macd, etc.) get added again.  Keep the first occurrence.
    if result.columns.duplicated().any():
        n_dupes = result.columns.duplicated().sum()
        result = result.loc[:, ~result.columns.duplicated()]
        logger.debug("Removed %d duplicate columns", n_dupes)

    # Ensure every var in DERIVED_VARIABLES exists and has an is_missing_*
    # companion.  Some stages require more data than available (e.g.,
    # momentum_12_1 needs 252 days) and may not create the column at all.
    # Newer stages (15-24) use direct assignment without companion flags.
    _n_vars_added = 0
    _n_flags_added = 0
    for _var in DERIVED_VARIABLES:
        if _var not in result.columns:
            result[_var] = np.nan
            _n_vars_added += 1
        _flag = f"is_missing_{_var}"
        if _flag not in result.columns:
            result[_flag] = result[_var].isna().astype(int)
            _n_flags_added += 1
    if _n_vars_added > 0 or _n_flags_added > 0:
        logger.debug(
            "Sweep: added %d missing var columns + %d is_missing_* flags",
            _n_vars_added, _n_flags_added,
        )

    logger.info(
        "Derived variables computed: %d new columns",
        len(result.columns) - len(df.columns),
    )
    return result
