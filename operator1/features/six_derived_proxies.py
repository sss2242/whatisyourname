"""Market-data-only derived proxies for SIX Swiss Exchange companies.

When financial statement data (income, balance, cashflow) is unavailable
-- as is the case for SIX, which provides no free financial filing API --
this module computes proxy ratios from the data that IS available:

  - Close+volume price data (~5 months from SIX historic CSV)
  - Dividend history (18 years from SIX share/dividend.json)
  - Shares outstanding (from SIX share/info.json)
  - Capital actions (from SIX official notices)
  - Capital structure (from SIX issuer/capital_structure.json)
  - Insider transactions (from SIX management_transactions)

No yfinance dependency. All data comes from SIX PIT-compliant APIs.

The proxies map to the pipeline's 5-tier survival hierarchy:

  Tier 1 (Liquidity):    Amihud illiquidity, dividend payout capacity,
                          dividend coverage ratio
  Tier 2 (Solvency):     Capital return yield, dilution risk,
                          book equity floor, capital action frequency
  Tier 3 (Stability):    Volatility from SIX CSV close data, drawdown
  Tier 4 (Profitability): Dividend growth consistency, implied earnings
  Tier 5 (Valuation):    Dividend yield, PDG ratio, implied PE,
                          price-to-book floor

Accuracy improvements (v3 -- no yfinance):
  1. Adaptive payout ratio from sector + index membership
  2. Date-based CAGR (actual years elapsed, not count-1)
  3. Buyback yield from actual shares outstanding delta
  4. Swiss-market-calibrated tier percentile scores
  5. Exponentially weighted dividend stability
  6. Implied earnings from dividend + buyback cash returns
  7. Book equity floor from nominal_value + capital_structure
  8. Dilution risk from conditional/reported capital ratio
  9. Dividend coverage ratio
  10. Confidence tagging on all proxy columns
  11. Graceful handling of short/missing close data

Survival mode triggers are adapted:
  - current_ratio < 1.0    -> dividend_cut_flag OR dividend_coverage < 1.0
  - debt_to_equity > 3.0   -> high dilution_risk AND dividend_cut
  - fcf_yield < 0           -> total_shareholder_return < 0
  - drawdown_252d < -0.40  -> same (from SIX CSV close, if available)

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

# Swiss corporate tax rate (Zurich canton, conservative)
_SWISS_TAX_RATE = 0.15


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
    dividend_cagr_5y: float | None = None
    dividend_cut_flag: bool = False
    dividend_stability: float | None = None
    dividend_coverage: float | None = None
    total_shareholder_return: float | None = None
    amihud_illiquidity_mean: float | None = None
    insider_confidence: float | None = None
    gordon_implied_return: float | None = None
    pdg_ratio: float | None = None
    capital_action_count: int = 0
    estimated_payout_ratio: float | None = None
    buyback_yield: float | None = None
    implied_earnings: float | None = None
    implied_pe: float | None = None
    book_equity_floor: float | None = None
    price_to_book_floor: float | None = None
    dilution_risk: float | None = None


# ---------------------------------------------------------------------------
# Improvement 1: Adaptive payout ratio from sector + index
# ---------------------------------------------------------------------------

_SECTOR_PAYOUT_ADJUSTMENTS: dict[str, float] = {
    "Consumer Defensive": +0.20,
    "Consumer Cyclical": +0.10,
    "Healthcare": +0.05,
    "Communication Services": +0.05,
    "Utilities": +0.15,
    "Financial Services": -0.05,
    "Industrials": -0.05,
    "Basic Materials": -0.05,
    "Technology": -0.15,
    "Energy": 0.0,
    "Real Estate": +0.10,
}


def _estimate_payout_ratio(
    sector: str,
    index_memberships: list[str] | None,
) -> float:
    """Estimate payout ratio from sector and index membership.

    Swiss blue chips in defensive sectors (Nestle, Lindt) typically
    have 70-80% payout ratios. Tech/growth companies pay 20-40%.
    """
    base = 0.50

    # Sector adjustment
    adjustment = _SECTOR_PAYOUT_ADJUSTMENTS.get(sector, 0.0)
    base += adjustment

    # Index membership bonus (blue chips distribute more)
    if index_memberships:
        if "SMI" in index_memberships:
            base += 0.05
        elif "SLI" in index_memberships:
            base += 0.02

    return max(0.15, min(base, 0.90))


# ---------------------------------------------------------------------------
# Improvement 2: Date-based precise CAGR
# ---------------------------------------------------------------------------

def _precise_cagr(dividends: pd.Series, max_years: int | None = None) -> float:
    """Compute CAGR using actual date difference for the exponent.

    Parameters
    ----------
    dividends:
        Series indexed by date, values are dividend amounts.
        Sorted newest-first.
    max_years:
        If set, only use dividends within this many years.
    """
    if len(dividends) < 2:
        return 0.0

    series = dividends.copy()

    # Filter to max_years window if specified
    if max_years and len(series) > 2:
        cutoff = series.index[0] - pd.Timedelta(days=365.25 * max_years)
        series = series[series.index >= cutoff]
        if len(series) < 2:
            return 0.0

    first_date = series.index[-1]  # oldest
    last_date = series.index[0]    # newest
    years_elapsed = (last_date - first_date).days / 365.25

    first_val = float(series.iloc[-1])
    last_val = float(series.iloc[0])

    if first_val <= 0 or years_elapsed <= 0.5:
        return 0.0

    return (last_val / first_val) ** (1.0 / years_elapsed) - 1.0


# ---------------------------------------------------------------------------
# Improvement 3: Buyback yield from shares outstanding delta
# ---------------------------------------------------------------------------

def _compute_buyback_yield_from_notices(
    profile: dict[str, Any],
    avg_price: float,
) -> float:
    """Compute actual buyback yield from shares destroyed in SIX notices.

    Parses capital destruction notices to find the shares outstanding
    before and after each event, then computes the annual buyback value.
    """
    try:
        from operator1.clients.ch_six import (
            _six_search_notices, _six_get_notice_text, _parse_shares_outstanding,
        )

        isin = profile.get("isin", "")
        shares_current = profile.get("shares_outstanding")
        if not isin or not shares_current or avg_price <= 0:
            return 0.0

        notices = _six_search_notices(isin=isin, years=3, notice_types="M")
        capital_actions = [
            n for n in notices
            if any(kw in (n.get("title", "") or "").lower()
                   for kw in ("kapitalvernichtung", "capital destruction",
                              "kapitalherabsetzung", "capital reduction"))
        ]

        if not capital_actions:
            return 0.0

        # Parse shares outstanding from each notice
        shares_timeline: list[tuple[str, int]] = []
        for n in capital_actions:
            nid = n.get("noticeId")
            date_raw = str(n.get("date", ""))
            if nid and date_raw:
                text = _six_get_notice_text(nid)
                shares = _parse_shares_outstanding(text)
                if shares:
                    shares_timeline.append((date_raw, shares))

        if len(shares_timeline) < 2:
            # Only one data point -- use current shares vs notice shares
            if shares_timeline:
                _, notice_shares = shares_timeline[0]
                if notice_shares > shares_current:
                    destroyed = notice_shares - shares_current
                    buyback_value = destroyed * avg_price
                    market_cap = shares_current * avg_price
                    return buyback_value / max(market_cap, 1) if market_cap > 0 else 0.0
            return 0.0

        # Multiple data points: compute total shares destroyed
        shares_timeline.sort(key=lambda x: x[0], reverse=True)
        newest_shares = shares_timeline[0][1]
        oldest_shares = shares_timeline[-1][1]

        if oldest_shares <= newest_shares:
            return 0.0  # shares increased, not destroyed

        total_destroyed = oldest_shares - newest_shares

        # Compute time span
        newest_date = shares_timeline[0][0]
        oldest_date = shares_timeline[-1][0]
        if len(newest_date) == 8 and len(oldest_date) == 8:
            from datetime import date as dt_date
            d1 = dt_date(int(oldest_date[:4]), int(oldest_date[4:6]), int(oldest_date[6:8]))
            d2 = dt_date(int(newest_date[:4]), int(newest_date[4:6]), int(newest_date[6:8]))
            years = max((d2 - d1).days / 365.25, 0.5)
        else:
            years = len(shares_timeline) - 1

        annual_destroyed = total_destroyed / years
        annual_buyback_value = annual_destroyed * avg_price
        market_cap = shares_current * avg_price

        return annual_buyback_value / max(market_cap, 1)

    except Exception as exc:
        logger.debug("Buyback yield computation failed: %s", exc)
        return 0.0


# ---------------------------------------------------------------------------
# Improvement 4: Swiss-market-calibrated percentile scores
# ---------------------------------------------------------------------------

_SWISS_BENCHMARKS: dict[str, dict[str, float]] = {
    "dividend_yield": {"p25": 0.015, "p50": 0.028, "p75": 0.042},
    "dividend_cagr": {"p25": 0.01, "p50": 0.03, "p75": 0.06},
    "amihud_illiquidity": {"p25": 0.00001, "p50": 0.0001, "p75": 0.001},
    "buyback_yield": {"p25": 0.0, "p50": 0.005, "p75": 0.02},
    "dilution_risk": {"p25": 0.0, "p50": 0.05, "p75": 0.15},
    "dividend_coverage": {"p25": 1.0, "p50": 1.5, "p75": 2.5},
}


def _calibrated_score(value: float, metric: str, invert: bool = False) -> float:
    """Map a metric value to a 0-100 score using Swiss market percentiles."""
    bench = _SWISS_BENCHMARKS.get(metric)
    if not bench:
        return 50.0

    p25, p50, p75 = bench["p25"], bench["p50"], bench["p75"]

    if invert:
        value = -value
        p25, p50, p75 = -p75, -p50, -p25

    if value <= p25:
        score = 25.0 * (value / max(p25, _EPS)) if p25 > 0 else 25.0
    elif value <= p50:
        score = 25.0 + 25.0 * (value - p25) / max(p50 - p25, _EPS)
    elif value <= p75:
        score = 50.0 + 25.0 * (value - p50) / max(p75 - p50, _EPS)
    else:
        score = min(100.0, 75.0 + 25.0 * (value - p75) / max(p75 - p50, _EPS))

    return max(0.0, min(100.0, score))


# ---------------------------------------------------------------------------
# Improvement 5: Exponentially weighted dividend stability
# ---------------------------------------------------------------------------

def _exponential_stability(dividends: pd.Series, halflife_years: float = 5.0) -> float:
    """Compute dividend stability with exponential recency weighting.

    Recent dividend changes are weighted more heavily than old ones,
    since a recent cut matters more than one 10 years ago.
    """
    if len(dividends) < 3:
        return 0.5  # insufficient data

    growth_rates = dividends.sort_index().pct_change().dropna()
    if growth_rates.empty:
        return 0.5

    n = len(growth_rates)
    # Weights: index 0 = oldest (lowest weight), index n-1 = newest (highest)
    # Exponential decay from newest: weight_i = 2^(i/halflife) where i=0..n-1
    weights = np.exp(np.arange(n) * np.log(2) / max(halflife_years, 1.0))
    weights = weights / weights.sum()

    weighted_mean = np.average(growth_rates.values, weights=weights)
    weighted_var = np.average((growth_rates.values - weighted_mean) ** 2, weights=weights)
    weighted_std = np.sqrt(weighted_var)

    # Count positive growth rates (dividends that grew or stayed flat)
    positive_growth = (growth_rates.values >= -0.001).astype(float)
    positive_fraction = np.average(positive_growth, weights=weights)

    # Measure consistency: low std of growth rates = high stability
    # Use absolute scale (not CV) since mean can be near zero
    # A std of 0.05 (5% variation) is very stable for dividends
    consistency = max(0.0, 1.0 - weighted_std / 0.10)

    # Combine: 60% positive growth fraction + 40% consistency
    score = positive_fraction * 0.6 + consistency * 0.4
    return max(0.0, min(1.0, score))


# ---------------------------------------------------------------------------
# Improvement 6: Implied earnings from dividend + buyback
# ---------------------------------------------------------------------------

def _compute_implied_earnings(
    total_dividends: float,
    total_buyback_value: float,
    market_cap: float,
    tax_rate: float = _SWISS_TAX_RATE,
) -> dict[str, float]:
    """Derive implied earnings and PE from total cash returned to shareholders.

    Key insight: total cash returned (dividends + buybacks) cannot exceed
    after-tax earnings sustainably. For mature companies that distribute
    most of their earnings, this gives a tight lower bound.
    """
    total_returned = total_dividends + total_buyback_value
    if total_returned <= 0:
        return {}

    implied_after_tax = total_returned
    implied_pretax = implied_after_tax / max(1.0 - tax_rate, 0.5)

    results = {
        "implied_after_tax_earnings": implied_after_tax,
        "implied_pretax_earnings": implied_pretax,
    }

    if market_cap > 0:
        results["implied_pe"] = market_cap / implied_after_tax
        results["implied_earnings_yield"] = implied_after_tax / market_cap

    return results


# ---------------------------------------------------------------------------
# Improvement 7: Book equity floor from capital structure
# ---------------------------------------------------------------------------

def _compute_book_equity_floor(profile: dict[str, Any]) -> dict[str, float]:
    """Compute book equity floor from SIX capital structure data.

    SIX provides nominal_value * shares_outstanding = share capital.
    The issuer/capital_structure.json gives reported_share_capital.
    This is a PIT-compliant lower bound on book equity (actual book
    equity >= share capital since retained earnings are positive for
    profitable companies).
    """
    results: dict[str, float] = {}

    shares = profile.get("shares_outstanding")
    nominal = profile.get("nominal_value")
    reported_capital = profile.get("reported_share_capital")

    # Method 1: From capital_structure API
    if reported_capital:
        try:
            results["book_equity_floor"] = float(reported_capital)
        except (TypeError, ValueError):
            pass

    # Method 2: From nominal_value * shares (fallback)
    if "book_equity_floor" not in results and shares and nominal:
        try:
            results["book_equity_floor"] = float(shares) * float(nominal)
        except (TypeError, ValueError):
            pass

    return results


# ---------------------------------------------------------------------------
# Improvement 8: Dilution risk from conditional capital
# ---------------------------------------------------------------------------

def _compute_dilution_risk(profile: dict[str, Any]) -> float:
    """Compute dilution risk from conditional/reported capital ratio.

    SIX capital_structure.json provides both REPORTED_CAPITAL and
    CONDITIONAL_CAPITAL. The ratio conditional/reported indicates
    potential share dilution from warrants, convertibles, employee
    stock options, etc.

    Returns a ratio in [0, 1+] where:
      0.0 = no conditional capital (no dilution risk)
      0.05-0.10 = normal (most Swiss companies)
      0.15+ = elevated dilution risk
    """
    try:
        from operator1.clients.ch_six import _fetch_share_detail_list

        valor_id = profile.get("valor_id", "")
        if not valor_id:
            return 0.0

        cap_items = _fetch_share_detail_list(valor_id, "issuer/capital_structure.json")
        if not cap_items:
            return 0.0

        reported = 0.0
        conditional = 0.0

        for item in cap_items:
            category = item.get("category", "")
            for cap in item.get("capitals", []):
                if cap.get("capitalType") == "SHARE":
                    amount = cap.get("capital", 0) or 0
                    try:
                        amount = float(amount)
                    except (TypeError, ValueError):
                        continue
                    if category == "REPORTED_CAPITAL":
                        reported += amount
                    elif category == "CONDITIONAL_CAPITAL":
                        conditional += amount

        if reported > 0:
            return conditional / reported
        return 0.0

    except Exception as exc:
        logger.debug("Dilution risk computation failed: %s", exc)
        return 0.0


# ---------------------------------------------------------------------------
# Dividend series extraction
# ---------------------------------------------------------------------------

def _get_dividend_series_from_profile(profile: dict) -> pd.Series:
    """Extract dividend amounts as a Series from the SIX API."""
    try:
        from operator1.clients.ch_six import _fetch_share_detail_list
        valor_id = profile.get("valor_id", "")
        if not valor_id:
            return pd.Series(dtype=float)

        dividends = _fetch_share_detail_list(valor_id, "share/dividend.json")
        if not dividends:
            return pd.Series(dtype=float)

        records = []
        for d in dividends:
            val = d.get("value") or d.get("adjustedValue")
            ex_date = d.get("exDividendDate")
            if val and ex_date:
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
# Liquidity proxies (works with short close data)
# ---------------------------------------------------------------------------

def _compute_liquidity_proxies(cache: pd.DataFrame) -> dict[str, pd.Series]:
    """Compute Amihud illiquidity ratio from close+volume data.

    Works with the ~5 months of SIX historic CSV data. For days
    without close data, the Amihud values will be NaN (not fabricated).
    """
    results: dict[str, pd.Series] = {}

    close = cache.get("close")
    volume = cache.get("volume")
    if close is None or volume is None:
        return results

    # Only compute where we have actual data
    valid = close.notna() & volume.notna() & (volume > 0)
    if valid.sum() < 5:
        return results

    returns = close.pct_change().abs()
    dollar_volume = close * volume
    safe_dv = dollar_volume.replace(0, np.nan)
    amihud = returns / safe_dv
    amihud_scaled = amihud * 1e6

    results["six_proxy_amihud_illiquidity"] = amihud_scaled
    results["six_proxy_amihud_21d"] = amihud_scaled.rolling(21, min_periods=5).mean()

    # Confidence: high where we have actual data, zero where interpolated
    conf = valid.astype(float)
    results["six_proxy_amihud_confidence"] = conf

    # Liquidity score: calibrated against Swiss market benchmarks
    latest_amihud = float(amihud_scaled.dropna().iloc[-1]) if not amihud_scaled.dropna().empty else 0.0
    liq_score_val = _calibrated_score(latest_amihud, "amihud_illiquidity", invert=True)
    results["six_proxy_liquidity_score"] = pd.Series(liq_score_val, index=cache.index, dtype=float)

    return results


# ---------------------------------------------------------------------------
# Insider transaction proxies
# ---------------------------------------------------------------------------

def _compute_insider_proxies(profile: dict[str, Any]) -> dict[str, float]:
    """Compute insider confidence score from SIX management transactions."""
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
            if indicator == "1":
                buys += 1
                buy_amount += amount
            elif indicator == "2":
                sells += 1
                sell_amount += amount

        total = buys + sells
        if total > 0:
            buy_ratio = buys / total
            net_flow = buy_amount - sell_amount
            conviction = buy_ratio * np.log1p(abs(net_flow)) * (1 if net_flow >= 0 else -1)
            results["_insider_buy_ratio"] = buy_ratio
            results["_insider_net_flow"] = net_flow
            results["_insider_conviction"] = conviction
            results["_insider_transaction_count"] = total

    except Exception as exc:
        logger.debug("Insider proxy computation failed: %s", exc)

    return results


# ---------------------------------------------------------------------------
# Confidence tagging helper
# ---------------------------------------------------------------------------

def _set_with_confidence(
    cache: pd.DataFrame,
    col_name: str,
    value: pd.Series | float,
    confidence: float,
) -> None:
    """Set a proxy column and its companion confidence column."""
    if isinstance(value, (int, float)):
        cache[col_name] = pd.Series(value, index=cache.index, dtype=float)
    else:
        cache[col_name] = value
    cache[f"{col_name}_confidence"] = pd.Series(confidence, index=cache.index, dtype=float)


# ---------------------------------------------------------------------------
# Main entry point
# ---------------------------------------------------------------------------

def compute_six_proxies(
    cache: pd.DataFrame,
    profile: dict[str, Any],
) -> SixProxyResult:
    """Compute all SIX market-data-only proxy ratios.

    v3: No yfinance. All data from SIX PIT APIs + historic CSV.
    Decoupled: dividend-based proxies work even without price data.
    Injects proxy columns into the cache DataFrame in-place.
    """
    result = SixProxyResult()

    if profile.get("market_id") != "ch_six":
        return result

    try:
        n_proxies = 0
        close = cache.get("close")
        has_close = close is not None and close.notna().sum() >= 5

        shares = profile.get("shares_outstanding")
        latest_div = profile.get("latest_dividend_amount")
        sector = profile.get("sector", "")
        index_memberships = profile.get("index_memberships", [])

        # --- Improvement 1: Adaptive payout ratio ---
        payout_ratio = _estimate_payout_ratio(sector, index_memberships)
        result.estimated_payout_ratio = payout_ratio

        # === DIVIDEND-BASED PROXIES (work without close data) ===

        # --- Improvement 2: Date-based CAGR ---
        dividends = _get_dividend_series_from_profile(profile)
        if len(dividends) >= 2:
            cagr_full = _precise_cagr(dividends)
            result.dividend_cagr = cagr_full

            cagr_5y = _precise_cagr(dividends, max_years=5)
            result.dividend_cagr_5y = cagr_5y

            # Dividend cut flag: latest < previous
            result.dividend_cut_flag = bool(dividends.iloc[0] < dividends.iloc[1])

            # --- Improvement 5: Exponential stability ---
            stability = _exponential_stability(dividends)
            result.dividend_stability = stability
            _set_with_confidence(cache, "six_proxy_dividend_stability", stability, 0.85)
            n_proxies += 1

            result.dividend_data_years = profile.get("dividend_history_years", len(dividends))

        # --- Dividend yield (needs close OR latest_close from profile) ---
        latest_close = None
        if has_close:
            latest_close = float(close.dropna().iloc[-1])
        elif profile.get("latest_close"):
            latest_close = float(profile["latest_close"])

        if latest_div and latest_close and latest_close > 0:
            annual_dividend = float(latest_div)

            if has_close:
                # Daily dividend yield series from actual close prices
                div_yield = annual_dividend / close.replace(0, np.nan)
                _set_with_confidence(cache, "six_proxy_dividend_yield", div_yield, 0.95)
            else:
                # Single-point dividend yield from profile's latest_close
                div_yield_scalar = annual_dividend / latest_close
                _set_with_confidence(cache, "six_proxy_dividend_yield", div_yield_scalar, 0.80)

            result.dividend_yield = annual_dividend / latest_close
            n_proxies += 1

            # Estimated OCF using adaptive payout ratio
            if shares:
                total_div_outflow = annual_dividend * float(shares)
                estimated_ocf = total_div_outflow / payout_ratio
                _set_with_confidence(cache, "six_proxy_est_operating_cf", estimated_ocf, 0.50)
                n_proxies += 1

            # Gordon model
            if result.dividend_cagr_5y is not None:
                result.gordon_implied_return = result.dividend_yield + result.dividend_cagr_5y

            # PDG ratio
            if result.dividend_cagr_5y and result.dividend_cagr_5y > 0.001:
                pdg_scalar = (latest_close / annual_dividend) / (result.dividend_cagr_5y * 100)
                _set_with_confidence(cache, "six_proxy_pdg_ratio", pdg_scalar, 0.70)
                result.pdg_ratio = pdg_scalar
                n_proxies += 1

        # --- Improvement 3: Buyback yield from shares delta ---
        avg_price = latest_close or 0.0
        if has_close:
            avg_price = float(close.dropna().mean())
        buyback_yield = _compute_buyback_yield_from_notices(profile, avg_price)
        result.buyback_yield = buyback_yield

        # Total shareholder return
        if result.dividend_yield is not None:
            result.total_shareholder_return = result.dividend_yield + buyback_yield
            _set_with_confidence(
                cache, "six_proxy_total_shareholder_return",
                result.total_shareholder_return, 0.85,
            )
            n_proxies += 1

        # --- Capital action count ---
        try:
            from operator1.clients.ch_six import _six_search_notices
            isin = profile.get("isin", "")
            if isin:
                notices = _six_search_notices(isin=isin, years=2, notice_types="M")
                capital_actions = [
                    n for n in notices
                    if any(kw in (n.get("title", "") or "").lower()
                           for kw in ("kapitalvernichtung", "capital destruction",
                                      "kapitalherabsetzung", "ruckkauf", "buyback"))
                ]
                result.capital_action_count = len(capital_actions)
        except Exception:
            pass

        # --- Improvement 6: Implied earnings ---
        if latest_div and shares and latest_close:
            total_div = float(latest_div) * float(shares)
            annual_buyback = buyback_yield * (float(shares) * avg_price) if avg_price > 0 else 0.0
            market_cap = float(shares) * latest_close

            implied = _compute_implied_earnings(total_div, annual_buyback, market_cap)
            if implied:
                result.implied_earnings = implied.get("implied_after_tax_earnings")
                result.implied_pe = implied.get("implied_pe")

                if result.implied_pe:
                    _set_with_confidence(cache, "six_proxy_implied_pe", result.implied_pe, 0.70)
                    n_proxies += 1
                if implied.get("implied_earnings_yield"):
                    _set_with_confidence(
                        cache, "six_proxy_implied_earnings_yield",
                        implied["implied_earnings_yield"], 0.70,
                    )
                    n_proxies += 1

                # --- Improvement 9: Dividend coverage ---
                if total_div > 0 and result.implied_earnings:
                    coverage = result.implied_earnings / total_div
                    result.dividend_coverage = coverage
                    _set_with_confidence(cache, "six_proxy_dividend_coverage", coverage, 0.65)
                    n_proxies += 1

        # === PRICE-BASED PROXIES (need close data from SIX CSV) ===

        if has_close:
            # Liquidity proxies (Amihud)
            liq_proxies = _compute_liquidity_proxies(cache)
            for key, val in liq_proxies.items():
                if isinstance(val, pd.Series):
                    cache[key] = val
                    n_proxies += 1

            result.amihud_illiquidity_mean = (
                float(cache["six_proxy_amihud_21d"].mean())
                if "six_proxy_amihud_21d" in cache.columns else None
            )

        # === CAPITAL STRUCTURE PROXIES (from SIX API, no price needed) ===

        # --- Improvement 7: Book equity floor ---
        book_equity = _compute_book_equity_floor(profile)
        if book_equity.get("book_equity_floor"):
            result.book_equity_floor = book_equity["book_equity_floor"]
            _set_with_confidence(
                cache, "six_proxy_book_equity_floor",
                result.book_equity_floor, 0.90,
            )
            n_proxies += 1

            # Price-to-book floor (needs market cap)
            if shares and latest_close:
                market_cap = float(shares) * latest_close
                if result.book_equity_floor > 0:
                    ptb = market_cap / result.book_equity_floor
                    result.price_to_book_floor = ptb
                    _set_with_confidence(cache, "six_proxy_price_to_book_floor", ptb, 0.75)
                    n_proxies += 1

        # --- Improvement 8: Dilution risk ---
        dilution = _compute_dilution_risk(profile)
        result.dilution_risk = dilution
        _set_with_confidence(cache, "six_proxy_dilution_risk", dilution, 0.90)
        n_proxies += 1

        # --- Insider proxies ---
        insider_proxies = _compute_insider_proxies(profile)
        result.insider_confidence = insider_proxies.get("_insider_conviction")

        # --- Calibrated tier scores ---
        _inject_calibrated_tier_scores(cache, result)

        # --- Survival proxy triggers ---
        _inject_survival_proxies(cache, result)

        result.computed = True
        result.n_proxies = n_proxies

        logger.info(
            "SIX proxy v3: %d columns, yield=%.2f%%, cagr_5y=%.2f%%, "
            "buyback=%.2f%%, implied_pe=%.1f, payout=%.0f%%, "
            "book_floor=%.0f, dilution=%.3f, coverage=%.2f, "
            "has_close=%s (%d pts)",
            n_proxies,
            (result.dividend_yield or 0) * 100,
            (result.dividend_cagr_5y or 0) * 100,
            (result.buyback_yield or 0) * 100,
            result.implied_pe or 0,
            (result.estimated_payout_ratio or 0) * 100,
            result.book_equity_floor or 0,
            result.dilution_risk or 0,
            result.dividend_coverage or 0,
            has_close,
            int(close.notna().sum()) if has_close else 0,
        )

    except Exception as exc:
        result.error = str(exc)
        logger.warning("SIX proxy computation failed: %s", exc)

    return result


def _inject_calibrated_tier_scores(
    cache: pd.DataFrame,
    result: SixProxyResult,
) -> None:
    """Inject Swiss-market-calibrated tier proxy scores."""

    # Tier 1: Liquidity (Amihud + dividend payout capacity + coverage)
    liq_score = 50.0
    if "six_proxy_liquidity_score" in cache.columns:
        liq_score = float(cache["six_proxy_liquidity_score"].iloc[-1])
    div_capacity_bonus = 0.0
    if result.estimated_payout_ratio and result.dividend_yield:
        div_capacity_bonus = min(result.dividend_yield * 500, 25)
    coverage_bonus = 0.0
    if result.dividend_coverage is not None and result.dividend_coverage >= 1.5:
        coverage_bonus = min((result.dividend_coverage - 1.0) * 20, 15)
    _set_with_confidence(
        cache, "six_proxy_tier1_score",
        max(0, min(100, liq_score * 0.5 + (50 + div_capacity_bonus) * 0.3 + (50 + coverage_bonus) * 0.2)),
        0.60,
    )

    # Tier 2: Solvency (capital actions + dividend continuity + dilution risk)
    solvency = 50.0
    if result.capital_action_count > 0:
        solvency += _calibrated_score(result.buyback_yield or 0, "buyback_yield") * 0.2
    if result.dividend_cut_flag:
        solvency -= 30
    if not result.dividend_cut_flag and result.dividend_data_years >= 5:
        solvency += 10
    # Dilution risk penalty
    if result.dilution_risk is not None and result.dilution_risk > 0.10:
        solvency -= min((result.dilution_risk - 0.10) * 100, 25)
    # Book equity floor bonus (having substantial share capital is a solvency signal)
    if result.book_equity_floor and result.book_equity_floor > 0:
        solvency += 5
    _set_with_confidence(
        cache, "six_proxy_tier2_score",
        max(0, min(100, solvency)),
        0.60,
    )

    # Tier 4: Profitability (dividend growth + stability)
    profit_score = 50.0
    if result.dividend_cagr_5y is not None:
        profit_score = _calibrated_score(result.dividend_cagr_5y, "dividend_cagr")
    if result.dividend_stability is not None:
        stability_score = result.dividend_stability * 100
        profit_score = profit_score * 0.6 + stability_score * 0.4
    _set_with_confidence(
        cache, "six_proxy_tier4_score",
        max(0, min(100, profit_score)),
        0.55,
    )

    # Tier 5: Valuation (implied PE + dividend yield + P/B floor)
    valuation = 50.0
    if result.implied_pe is not None:
        if result.implied_pe < 12:
            valuation = 80
        elif result.implied_pe < 18:
            valuation = 60
        elif result.implied_pe < 25:
            valuation = 45
        else:
            valuation = 25
    if result.dividend_yield is not None:
        yield_score = _calibrated_score(result.dividend_yield, "dividend_yield")
        valuation = valuation * 0.4 + yield_score * 0.3
    if result.price_to_book_floor is not None:
        # P/B floor < 2 is cheap, > 5 is expensive
        if result.price_to_book_floor < 2:
            pb_score = 80
        elif result.price_to_book_floor < 4:
            pb_score = 55
        else:
            pb_score = 30
        valuation += pb_score * 0.3
    _set_with_confidence(
        cache, "six_proxy_tier5_score",
        max(0, min(100, valuation)),
        0.55,
    )


def _inject_survival_proxies(
    cache: pd.DataFrame,
    result: SixProxyResult,
) -> None:
    """Inject proxy survival triggers when standard triggers are NaN."""

    # Proxy for current_ratio < 1.0: dividend was cut OR dividend coverage < 1.0
    if "current_ratio" not in cache.columns or cache["current_ratio"].isna().all():
        liquidity_distress = result.dividend_cut_flag
        if result.dividend_coverage is not None and result.dividend_coverage < 1.0:
            liquidity_distress = True
        _set_with_confidence(
            cache, "six_proxy_liquidity_distress_flag",
            int(liquidity_distress), 0.60,
        )

    # Proxy for debt_to_equity > 3.0: high dilution + dividend cut
    if "debt_to_equity" not in cache.columns or cache["debt_to_equity"].isna().all():
        solvency_distress = (
            result.dividend_cut_flag
            and result.dilution_risk is not None
            and result.dilution_risk > 0.15
        )
        _set_with_confidence(
            cache, "six_proxy_solvency_distress_flag",
            int(solvency_distress), 0.50,
        )

    # Proxy for fcf_yield < 0: total shareholder return < 0
    if "six_proxy_total_shareholder_return" in cache.columns:
        tsr = cache["six_proxy_total_shareholder_return"]
        if "fcf_yield" not in cache.columns or cache["fcf_yield"].isna().all():
            _set_with_confidence(
                cache, "six_proxy_negative_tsr_flag",
                (tsr < 0).astype(int), 0.70,
            )
