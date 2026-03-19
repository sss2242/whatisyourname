"""Market-data-only derived proxies for SIX Swiss Exchange companies.

When financial statement data (income, balance, cashflow) is unavailable
-- as is the case for SIX, which provides no free financial filing API --
this module computes proxy ratios from the data that IS available:

  - OHLCV price data (close, volume)
  - Dividend history (18 years from SIX share/dividend.json)
  - Shares outstanding (from SIX share/info.json)
  - Capital actions (from SIX official notices)
  - Insider transactions (from SIX management_transactions)

The proxies map to the pipeline's 5-tier survival hierarchy:

  Tier 1 (Liquidity):    Amihud illiquidity, dividend payout capacity
  Tier 2 (Solvency):     Capital return yield, capital action frequency
  Tier 3 (Stability):    Already from OHLCV (volatility, drawdown)
  Tier 4 (Profitability): Dividend growth consistency, earnings growth proxy
  Tier 5 (Valuation):    Dividend yield, PDG ratio, Gordon model return

Survival mode triggers are adapted:
  - current_ratio < 1.0    -> dividend_cut_flag (dividend decreased YoY)
  - debt_to_equity > 3.0   -> no_buyback_and_dividend_cut (double stress)
  - fcf_yield < 0           -> total_shareholder_return < 0
  - drawdown_252d < -0.40  -> same (from OHLCV, already works)

Top-level entry point:
    ``compute_six_proxies(cache, profile) -> SixProxyResult``
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Any

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

_EPS = 1e-10


# ---------------------------------------------------------------------------
# Result container
# ---------------------------------------------------------------------------

@dataclass
class SixProxyResult:
    """Container for SIX market-data-only proxy computations."""

    computed: bool = False
    n_proxies: int = 0
    dividend_data_years: int = 0
    error: str | None = None

    # Proxy values (latest)
    dividend_yield: float | None = None
    dividend_cagr: float | None = None
    dividend_cut_flag: bool = False
    dividend_stability: float | None = None
    total_shareholder_return: float | None = None
    amihud_illiquidity_mean: float | None = None
    insider_confidence: float | None = None
    gordon_implied_return: float | None = None
    pdg_ratio: float | None = None
    capital_action_count: int = 0


# ---------------------------------------------------------------------------
# Dividend-based proxies
# ---------------------------------------------------------------------------

def _compute_dividend_proxies(
    cache: pd.DataFrame,
    profile: dict[str, Any],
) -> dict[str, pd.Series | float]:
    """Compute dividend-based proxy ratios from SIX dividend history.

    Uses the dividend history stored in the profile (from SIX
    share/dividend.json endpoint) to derive:
    - dividend_yield (daily, from close price)
    - dividend_cagr (annualized growth rate)
    - dividend_cut_flag (1 if latest < previous)
    - dividend_stability (consistency score 0-1)
    - estimated_payout_capacity (minimum operating CF estimate)
    """
    results: dict[str, Any] = {}
    close = cache.get("close")
    if close is None or close.dropna().empty:
        return results

    shares = profile.get("shares_outstanding")
    latest_div = profile.get("latest_dividend_amount")
    div_years = profile.get("dividend_history_years", 0)

    if not latest_div or not shares:
        return results

    # --- Dividend yield (daily) ---
    annual_dividend = float(latest_div)
    div_yield = annual_dividend / close.replace(0, np.nan)
    results["six_proxy_dividend_yield"] = div_yield

    # --- Total annual dividend outflow ---
    total_div_outflow = annual_dividend * float(shares)
    results["_total_div_outflow"] = total_div_outflow

    # --- Dividend payout capacity proxy ---
    # A company paying X in dividends must have >= X in operating CF
    # Assume conservative payout ratio of 60% for Swiss blue chips
    payout_ratio_assumption = 0.60
    estimated_ocf = total_div_outflow / payout_ratio_assumption
    results["six_proxy_est_operating_cf"] = estimated_ocf

    # --- Dividend growth from history ---
    # Extract dividend time series from profile's extra data
    # We fetch the full dividend list from the SIX API
    dividends = _get_dividend_series_from_profile(profile)

    if len(dividends) >= 2:
        # CAGR
        first_val = dividends.iloc[-1]  # oldest
        last_val = dividends.iloc[0]    # newest
        n_years = len(dividends) - 1
        if first_val > 0 and n_years > 0:
            cagr = (last_val / first_val) ** (1.0 / n_years) - 1.0
            results["_dividend_cagr"] = cagr
        else:
            cagr = 0.0
            results["_dividend_cagr"] = 0.0

        # Dividend cut flag
        if len(dividends) >= 2:
            latest = dividends.iloc[0]
            previous = dividends.iloc[1]
            results["_dividend_cut_flag"] = bool(latest < previous)
        else:
            results["_dividend_cut_flag"] = False

        # Dividend stability (1 - CV of growth rates)
        if len(dividends) >= 3:
            growth_rates = dividends.pct_change().dropna()
            if growth_rates.std() > 0:
                cv = abs(growth_rates.std() / (growth_rates.mean() + _EPS))
                stability = max(0.0, 1.0 - cv)
            else:
                stability = 1.0  # zero variance = perfect stability
            results["six_proxy_dividend_stability"] = pd.Series(
                stability, index=cache.index, dtype=float,
            )

        # Gordon Growth Model implied return
        current_yield = div_yield.iloc[-1] if not div_yield.empty else 0.0
        gordon_return = current_yield + cagr
        results["_gordon_implied_return"] = gordon_return

        # Price-to-Dividend-Growth ratio
        if cagr > 0.001:
            price_to_div = close / annual_dividend
            pdg = price_to_div / (cagr * 100)  # normalize growth to percentage
            results["six_proxy_pdg_ratio"] = pdg

    return results


def _get_dividend_series_from_profile(profile: dict) -> pd.Series:
    """Extract dividend amounts as a Series from the profile.

    The SIX share/dividend.json returns items with exDividendDate and value.
    We fetch these directly from the SIX API since the profile only stores
    the latest values.
    """
    try:
        from operator1.clients.ch_six import _fetch_share_detail_list
        valor_id = profile.get("valor_id", "")
        if not valor_id:
            return pd.Series(dtype=float)

        dividends = _fetch_share_detail_list(valor_id, "share/dividend.json")
        if not dividends:
            return pd.Series(dtype=float)

        # Build Series: index=ex_date, values=dividend_amount
        records = []
        for d in dividends:
            val = d.get("value") or d.get("adjustedValue")
            ex_date = d.get("exDividendDate")
            if val and ex_date:
                # Parse YYYYMMDD int to date
                date_str = str(ex_date)
                if len(date_str) == 8:
                    records.append({
                        "date": pd.Timestamp(f"{date_str[:4]}-{date_str[4:6]}-{date_str[6:8]}"),
                        "dividend": float(val),
                    })

        if not records:
            return pd.Series(dtype=float)

        df = pd.DataFrame(records).sort_values("date", ascending=False)
        return df.set_index("date")["dividend"]

    except Exception as exc:
        logger.debug("Failed to extract dividend series: %s", exc)
        return pd.Series(dtype=float)


# ---------------------------------------------------------------------------
# Liquidity proxies (from OHLCV)
# ---------------------------------------------------------------------------

def _compute_liquidity_proxies(cache: pd.DataFrame) -> dict[str, pd.Series]:
    """Compute Amihud illiquidity ratio from OHLCV data.

    Amihud (2002): ratio of absolute return to dollar volume.
    High values = illiquid = potential stress.
    """
    results: dict[str, pd.Series] = {}

    close = cache.get("close")
    volume = cache.get("volume")
    if close is None or volume is None:
        return results

    returns = close.pct_change().abs()
    dollar_volume = close * volume

    # Avoid division by zero
    safe_dv = dollar_volume.replace(0, np.nan)
    amihud = returns / safe_dv

    # Scale to make values more interpretable (multiply by 1e6)
    amihud_scaled = amihud * 1e6

    results["six_proxy_amihud_illiquidity"] = amihud_scaled

    # Rolling 21-day average
    results["six_proxy_amihud_21d"] = amihud_scaled.rolling(21, min_periods=5).mean()

    # Liquidity score: inverse percentile rank (0=illiquid, 100=liquid)
    expanding_rank = amihud_scaled.expanding(min_periods=21).rank(pct=True)
    results["six_proxy_liquidity_score"] = (1.0 - expanding_rank) * 100

    return results


# ---------------------------------------------------------------------------
# Capital action proxies (from SIX notices)
# ---------------------------------------------------------------------------

def _compute_capital_proxies(
    cache: pd.DataFrame,
    profile: dict[str, Any],
) -> dict[str, Any]:
    """Compute buyback/capital return proxies from SIX official notices."""
    results: dict[str, Any] = {}

    shares = profile.get("shares_outstanding")
    market_cap = profile.get("market_cap")
    if not shares or not market_cap:
        return results

    # Count capital actions from notices
    try:
        from operator1.clients.ch_six import _six_search_notices
        isin = profile.get("isin", "")
        if isin:
            notices = _six_search_notices(isin=isin, years=2, notice_types="M")
            capital_actions = [
                n for n in notices
                if any(kw in (n.get("title", "") or "").lower()
                       for kw in ("kapitalvernichtung", "capital destruction",
                                  "kapitalherabsetzung", "capital reduction",
                                  "ruckkauf", "buyback"))
            ]
            results["_capital_action_count"] = len(capital_actions)

            # Buyback yield estimate: if shares were destroyed, estimate value
            # from the average price around the destruction date
            if capital_actions and len(notices) >= 2:
                # Simple proxy: assume ~2% of market cap per buyback event per year
                annual_buyback_events = len(capital_actions) / 2.0  # events per year
                buyback_yield = annual_buyback_events * 0.02  # rough estimate
                results["_buyback_yield_estimate"] = buyback_yield
    except Exception as exc:
        logger.debug("Capital proxy computation failed: %s", exc)

    return results


# ---------------------------------------------------------------------------
# Insider transaction proxies
# ---------------------------------------------------------------------------

def _compute_insider_proxies(profile: dict[str, Any]) -> dict[str, float]:
    """Compute insider confidence score from SIX management transactions.

    Uses the management_transactions/v1/overview.json endpoint to
    find buy/sell transactions for the company's ISIN.
    """
    results: dict[str, float] = {}

    try:
        import requests
        isin = profile.get("isin", "")
        if not isin:
            return results

        resp = requests.get(
            "https://www.six-group.com/sheldon/management_transactions/v1/overview.json",
            headers={"User-Agent": "Operator1/1.0", "Accept": "application/json"},
            timeout=30,
        )
        resp.raise_for_status()
        data = resp.json()

        items = data.get("itemList", [])
        buys = 0
        sells = 0
        buy_amount = 0.0
        sell_amount = 0.0

        for item in items:
            if item.get("ISIN") != isin:
                continue
            indicator = item.get("buySellIndicator", "")
            amount = item.get("transactionAmountCHF", 0.0) or 0.0
            if indicator == "1":  # buy
                buys += 1
                buy_amount += amount
            elif indicator == "2":  # sell
                sells += 1
                sell_amount += amount

        total = buys + sells
        if total > 0:
            buy_ratio = buys / total
            net_flow = buy_amount - sell_amount
            # Conviction score: buy ratio * log scale of net flow
            conviction = buy_ratio * np.log1p(abs(net_flow)) * (1 if net_flow >= 0 else -1)
            results["_insider_buy_ratio"] = buy_ratio
            results["_insider_net_flow"] = net_flow
            results["_insider_conviction"] = conviction
            results["_insider_transaction_count"] = total

    except Exception as exc:
        logger.debug("Insider proxy computation failed: %s", exc)

    return results


# ---------------------------------------------------------------------------
# Main entry point
# ---------------------------------------------------------------------------

def compute_six_proxies(
    cache: pd.DataFrame,
    profile: dict[str, Any],
) -> SixProxyResult:
    """Compute all SIX market-data-only proxy ratios.

    Injects proxy columns into the cache DataFrame in-place and
    returns a summary result object.

    Parameters
    ----------
    cache:
        Daily cache DataFrame with OHLCV data.
    profile:
        Company profile dict from CHSixClient.get_profile().

    Returns
    -------
    SixProxyResult with proxy values and metadata.
    """
    result = SixProxyResult()

    # Only run for ch_six market
    if profile.get("market_id") != "ch_six":
        return result

    try:
        n_proxies = 0

        # --- Dividend proxies ---
        div_proxies = _compute_dividend_proxies(cache, profile)
        for key, val in div_proxies.items():
            if key.startswith("six_proxy_") and isinstance(val, pd.Series):
                cache[key] = val
                n_proxies += 1

        result.dividend_yield = float(cache["six_proxy_dividend_yield"].iloc[-1]) if "six_proxy_dividend_yield" in cache.columns else None
        result.dividend_cagr = div_proxies.get("_dividend_cagr")
        result.dividend_cut_flag = div_proxies.get("_dividend_cut_flag", False)
        result.gordon_implied_return = div_proxies.get("_gordon_implied_return")
        result.total_shareholder_return = (
            (result.dividend_yield or 0) + div_proxies.get("_buyback_yield_estimate", 0)
        )

        # --- Liquidity proxies ---
        liq_proxies = _compute_liquidity_proxies(cache)
        for key, val in liq_proxies.items():
            if isinstance(val, pd.Series):
                cache[key] = val
                n_proxies += 1

        result.amihud_illiquidity_mean = float(cache["six_proxy_amihud_21d"].mean()) if "six_proxy_amihud_21d" in cache.columns else None

        # --- Capital action proxies ---
        cap_proxies = _compute_capital_proxies(cache, profile)
        result.capital_action_count = cap_proxies.get("_capital_action_count", 0)

        buyback_yield = cap_proxies.get("_buyback_yield_estimate", 0)
        if result.dividend_yield is not None:
            result.total_shareholder_return = result.dividend_yield + buyback_yield

        # --- Insider proxies ---
        insider_proxies = _compute_insider_proxies(profile)
        result.insider_confidence = insider_proxies.get("_insider_conviction")

        # --- Inject survival-mode proxy triggers ---
        # These replace the standard triggers when financial statement data is NaN
        _inject_survival_proxies(cache, div_proxies, cap_proxies)

        # --- Inject financial health proxy scores ---
        _inject_health_proxies(cache, div_proxies, liq_proxies, result)

        result.computed = True
        result.n_proxies = n_proxies
        result.dividend_data_years = profile.get("dividend_history_years", 0)

        logger.info(
            "SIX proxy computation: %d proxy columns injected, "
            "dividend_yield=%.2f%%, cagr=%.2f%%, insider_confidence=%.2f",
            n_proxies,
            (result.dividend_yield or 0) * 100,
            (result.dividend_cagr or 0) * 100,
            result.insider_confidence or 0,
        )

    except Exception as exc:
        result.error = str(exc)
        logger.warning("SIX proxy computation failed: %s", exc)

    return result


def _inject_survival_proxies(
    cache: pd.DataFrame,
    div_proxies: dict,
    cap_proxies: dict,
) -> None:
    """Inject proxy survival triggers into the cache.

    When standard survival triggers (current_ratio, debt_to_equity, fcf_yield)
    are NaN, these proxy triggers provide dividend-based alternatives.
    """
    # Proxy for current_ratio < 1.0: dividend was cut
    dividend_cut = div_proxies.get("_dividend_cut_flag", False)
    if "current_ratio" not in cache.columns or cache["current_ratio"].isna().all():
        # Use dividend cut as a proxy survival trigger
        cache["six_proxy_dividend_cut_flag"] = int(dividend_cut)

    # Proxy for fcf_yield < 0: total shareholder return < 0
    div_yield = div_proxies.get("six_proxy_dividend_yield")
    buyback_yield = cap_proxies.get("_buyback_yield_estimate", 0)
    if isinstance(div_yield, pd.Series):
        tsr = div_yield + buyback_yield
        cache["six_proxy_total_shareholder_return"] = tsr
        if "fcf_yield" not in cache.columns or cache["fcf_yield"].isna().all():
            cache["six_proxy_negative_tsr_flag"] = (tsr < 0).astype(int)


def _inject_health_proxies(
    cache: pd.DataFrame,
    div_proxies: dict,
    liq_proxies: dict,
    result: SixProxyResult,
) -> None:
    """Inject proxy financial health tier scores.

    Maps proxy ratios to the 5-tier structure so financial_health.py
    can use them when standard scores are NaN.
    """
    # Tier 1: Liquidity proxy score (from Amihud + dividend capacity)
    if "six_proxy_liquidity_score" in cache.columns:
        # Blend Amihud liquidity with dividend payout capacity
        liq_score = cache["six_proxy_liquidity_score"]
        # Dividend payout score: having a dividend at all = 50 pts, consistency adds more
        div_stability = div_proxies.get("six_proxy_dividend_stability")
        if isinstance(div_stability, pd.Series):
            div_score = div_stability * 50 + 50  # 50-100 range
            cache["six_proxy_tier1_score"] = (liq_score * 0.5 + div_score * 0.5).clip(0, 100)
        else:
            cache["six_proxy_tier1_score"] = liq_score

    # Tier 2: Solvency proxy score (from capital actions + dividend continuity)
    solvency_base = 50.0  # neutral starting point
    if result.capital_action_count > 0:
        solvency_base += min(result.capital_action_count * 10, 30)  # buybacks = strong
    if result.dividend_cut_flag:
        solvency_base -= 30  # dividend cut = solvency concern
    cache["six_proxy_tier2_score"] = pd.Series(
        max(0, min(100, solvency_base)), index=cache.index, dtype=float,
    )

    # Tier 4: Profitability proxy score (from dividend growth consistency)
    if result.dividend_cagr is not None:
        # Positive growth = profitable, negative = concerning
        growth_score = 50 + result.dividend_cagr * 500  # scale: 5% growth -> 75 pts
        growth_score = max(0, min(100, growth_score))
        cache["six_proxy_tier4_score"] = pd.Series(
            growth_score, index=cache.index, dtype=float,
        )

    # Tier 5: Valuation proxy score (from dividend yield relative to history)
    if "six_proxy_dividend_yield" in cache.columns:
        dy = cache["six_proxy_dividend_yield"]
        # Score based on yield percentile: higher yield = potentially undervalued
        yield_pctile = dy.expanding(min_periods=21).rank(pct=True) * 100
        cache["six_proxy_tier5_score"] = yield_pctile
