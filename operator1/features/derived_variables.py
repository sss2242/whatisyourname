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

    # Net debt to EBITDA
    ebitda = df.get("ebitda", pd.Series(np.nan, index=df.index))
    result, ism, inv = safe_ratio(df["net_debt"], ebitda, "net_debt_to_ebitda")
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
    """
    ocf = df.get("operating_cash_flow", pd.Series(np.nan, index=df.index))
    capex = df.get("capex", pd.Series(np.nan, index=df.index))
    market_cap = df.get("market_cap", pd.Series(np.nan, index=df.index))

    # Free cash flow = operating CF - capex
    # Note: capex is often negative in API data (outflow), so we use abs
    capex_abs = capex.abs()
    df["free_cash_flow"] = ocf - capex_abs
    df["is_missing_free_cash_flow"] = (ocf.isna() & capex.isna()).astype(int)

    # Trailing 4-quarter FCF: rolling sum of last 4 distinct quarterly values.
    # The _rolling_4q_ttm helper detects quarter transitions in the
    # forward-filled daily data and sums the last 4 distinct values.
    df["free_cash_flow_ttm_asof"] = _rolling_4q_ttm(df["free_cash_flow"])
    df["is_missing_free_cash_flow_ttm_asof"] = df["free_cash_flow_ttm_asof"].isna().astype(int)

    # FCF yield = FCF / market_cap
    result, ism, inv = safe_ratio(
        df["free_cash_flow"], market_cap, "fcf_yield",
    )
    _set_ratio_columns(df, "fcf_yield", result, ism, inv)

    return df


# ---------------------------------------------------------------------------
# Profitability
# ---------------------------------------------------------------------------


def _compute_profitability(df: pd.DataFrame) -> pd.DataFrame:
    """Compute profitability ratios.

    Variables: gross_margin, operating_margin, net_margin, roe.
    """
    revenue = df.get("revenue", pd.Series(np.nan, index=df.index))
    gross_profit = df.get("gross_profit", pd.Series(np.nan, index=df.index))
    # Prefer ebit; fall back to operating_income (canonical name used by
    # many international data sources via canonical_translator).
    ebit = df.get("ebit", pd.Series(np.nan, index=df.index))
    if ebit.isna().all():
        ebit = df.get("operating_income", pd.Series(np.nan, index=df.index))
    net_income = df.get("net_income", pd.Series(np.nan, index=df.index))
    equity = df.get("total_equity", pd.Series(np.nan, index=df.index))

    # Gross margin
    result, ism, inv = safe_ratio(gross_profit, revenue, "gross_margin")
    _set_ratio_columns(df, "gross_margin", result, ism, inv)

    # Operating margin
    result, ism, inv = safe_ratio(ebit, revenue, "operating_margin")
    _set_ratio_columns(df, "operating_margin", result, ism, inv)

    # Net margin
    result, ism, inv = safe_ratio(net_income, revenue, "net_margin")
    _set_ratio_columns(df, "net_margin", result, ism, inv)

    # ROE = net_income / total_equity
    result, ism, inv = safe_ratio(net_income, equity, "roe")
    _set_ratio_columns(df, "roe", result, ism, inv)

    # EBITDA approximation: EBITDA is not a reported line item in most
    # GAAP/IFRS filings.  Best available proxy is operating_income (EBIT)
    # since depreciation/amortization are rarely available as separate items.
    if "ebitda" not in df.columns or df["ebitda"].isna().all():
        ebit_proxy = df.get("ebit", df.get("operating_income", pd.Series(np.nan, index=df.index)))
        df["ebitda"] = ebit_proxy
        df["is_missing_ebitda"] = df["ebitda"].isna().astype(int)

    return df


# ---------------------------------------------------------------------------
# Valuation (optional)
# ---------------------------------------------------------------------------


def _compute_valuation(df: pd.DataFrame) -> pd.DataFrame:
    """Compute optional valuation metrics.

    Variables: pe_ratio_calc, earnings_yield_calc, ps_ratio_calc,
    enterprise_value, ev_to_ebitda.
    """
    close = df.get("close", pd.Series(np.nan, index=df.index))
    shares = df.get("shares_outstanding", pd.Series(np.nan, index=df.index))
    market_cap = df.get("market_cap", pd.Series(np.nan, index=df.index))
    net_income = df.get("net_income", pd.Series(np.nan, index=df.index))
    revenue = df.get("revenue", pd.Series(np.nan, index=df.index))
    ebitda = df.get("ebitda", pd.Series(np.nan, index=df.index))
    total_debt = df.get("total_debt_asof", pd.Series(np.nan, index=df.index))
    cash = df.get("cash_and_equivalents", pd.Series(np.nan, index=df.index))

    # Earnings per share (calc)
    eps_calc = net_income / shares.where(shares.abs() > EPSILON, other=np.nan)

    # P/E ratio = close / EPS
    result, ism, inv = safe_ratio(close, eps_calc, "pe_ratio_calc")
    _set_ratio_columns(df, "pe_ratio_calc", result, ism, inv)

    # Earnings yield = EPS / close
    result, ism, inv = safe_ratio(eps_calc, close, "earnings_yield_calc")
    _set_ratio_columns(df, "earnings_yield_calc", result, ism, inv)

    # P/S ratio = market_cap / revenue
    result, ism, inv = safe_ratio(market_cap, revenue, "ps_ratio_calc")
    _set_ratio_columns(df, "ps_ratio_calc", result, ism, inv)

    # Enterprise value = market_cap + total_debt - cash
    ev = market_cap.fillna(0) + total_debt.fillna(0) - cash.fillna(0)
    ev_missing = market_cap.isna() & total_debt.isna() & cash.isna()
    df["enterprise_value"] = ev.where(~ev_missing, other=np.nan)
    df["is_missing_enterprise_value"] = ev_missing.astype(int)

    # EV/EBITDA
    result, ism, inv = safe_ratio(df["enterprise_value"], ebitda, "ev_to_ebitda")
    _set_ratio_columns(df, "ev_to_ebitda", result, ism, inv)

    # P/B ratio = market_cap / total_equity
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
    """Compute return on assets.

    Variable: roa = net_income / total_assets.
    """
    net_income = df.get("net_income", pd.Series(np.nan, index=df.index))
    total_assets = df.get("total_assets", pd.Series(np.nan, index=df.index))

    result, ism, inv = safe_ratio(net_income, total_assets, "roa")
    _set_ratio_columns(df, "roa", result, ism, inv)

    return df


# ---------------------------------------------------------------------------
# TTM aggregates and growth rates (Tier 5)
# ---------------------------------------------------------------------------


def _compute_ttm_and_growth(df: pd.DataFrame) -> pd.DataFrame:
    """Compute trailing-twelve-month aggregates and YoY growth rates.

    Variables:
    - revenue_ttm_asof, net_income_ttm_asof, ebitda_ttm_asof
      (approximated: since statements are as-of aligned, the value
      already reflects the latest period; we keep the name for clarity)
    - revenue_growth_yoy: YoY change in revenue
    - earnings_growth_yoy: YoY change in net_income

    TTM is computed by summing the last 4 distinct quarterly values
    using ``_rolling_4q_ttm``.  YoY growth compares the current TTM
    value to the TTM value ~252 business days ago.
    """
    revenue = df.get("revenue", pd.Series(np.nan, index=df.index))
    net_income = df.get("net_income", pd.Series(np.nan, index=df.index))
    ebitda = df.get("ebitda", pd.Series(np.nan, index=df.index))

    # Rolling 4-quarter TTM sums
    df["revenue_ttm_asof"] = _rolling_4q_ttm(revenue)
    df["is_missing_revenue_ttm_asof"] = df["revenue_ttm_asof"].isna().astype(int)

    df["net_income_ttm_asof"] = _rolling_4q_ttm(net_income)
    df["is_missing_net_income_ttm_asof"] = df["net_income_ttm_asof"].isna().astype(int)

    df["ebitda_ttm_asof"] = _rolling_4q_ttm(ebitda)
    df["is_missing_ebitda_ttm_asof"] = df["ebitda_ttm_asof"].isna().astype(int)

    # YoY growth rates: compare current TTM to TTM ~252 trading days ago
    rev_ttm = df["revenue_ttm_asof"]
    rev_prev = rev_ttm.shift(252)
    result, ism, inv = safe_ratio(rev_ttm - rev_prev, rev_prev.abs(), "revenue_growth_yoy")
    _set_ratio_columns(df, "revenue_growth_yoy", result, ism, inv)

    ni_ttm = df["net_income_ttm_asof"]
    ni_prev = ni_ttm.shift(252)
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
# Public API
# ---------------------------------------------------------------------------

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
)


def compute_derived_variables(df: pd.DataFrame) -> pd.DataFrame:
    """Compute all derived decision variables for an entity daily cache.

    Parameters
    ----------
    df:
        Daily DataFrame from ``build_entity_daily_cache()`` containing
        at minimum: ``close``, and as many Sec 8 direct fields as
        available.

    Returns
    -------
    pd.DataFrame
        Input DataFrame augmented with derived variables and their
        companion ``is_missing_*`` / ``invalid_math_*`` flags.
    """
    result = df.copy()
    for stage in _COMPUTE_STAGES:
        try:
            result = stage(result)
        except Exception as exc:
            logger.warning(
                "Derived variable stage %s failed: %s", stage.__name__, exc,
            )
    logger.info(
        "Derived variables computed: %d new columns",
        len(result.columns) - len(df.columns),
    )
    return result
