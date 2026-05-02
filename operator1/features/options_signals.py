"""Options-derived forward-looking signals for equity prediction.

Fetches the full options surface for a stock and computes 6 features that
capture institutional positioning, tail risk pricing, and volatility
expectations.  These signals are leading indicators -- they move before
price does because informed traders express directional views through
options before the underlying.

Features computed:
  - put_call_ratio: put volume / call volume (>1.0 = bearish positioning)
  - risk_reversal_25d: 25-delta call IV minus 25-delta put IV (negative = bearish skew)
  - iv_skew: OTM put IV / ATM IV (>1.2 = elevated tail fear)
  - vix_term_structure: VIX / VIX3M ratio (>1.0 = backwardation = imminent stress)
  - skew_index: CBOE SKEW index (100-170, higher = more tail risk priced in)
  - variance_risk_premium: IV^2 - RV^2 (positive = market expects more vol than realized)

Data source: yfinance (already installed, no new packages).

References:
  - Bollen & Whaley (2004): risk reversal as directional predictor
  - Bollerslev, Tauchen & Zhou (2009): variance risk premium predicts returns
  - Squeezemetrics: gamma exposure (GEX) and market maker positioning

Pipeline integration:
  - Called in main.py Step 4a.7 (after IV fetch, before estimation)
  - All 6 features added to _extra_vars for temporal model consumption
  - Profile key: options_signals
"""

from __future__ import annotations

import logging
import math
from dataclasses import dataclass
from typing import Any

import pandas as pd

from operator1.scoring_weights import get_weight

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Result container
# ---------------------------------------------------------------------------

@dataclass
class OptionsSignalResult:
    """Container for options-derived signal computation results."""

    available: bool = False
    put_call_ratio: float | None = None
    risk_reversal_25d: float | None = None
    iv_skew: float | None = None
    vix_term_structure: float | None = None
    skew_index: float | None = None
    variance_risk_premium: float | None = None
    n_calls: int = 0
    n_puts: int = 0
    nearest_expiry: str = ""
    error: str = ""

    def to_profile_dict(self) -> dict[str, Any]:
        """Convert to profile-ready dict."""
        return {
            "available": self.available,
            "put_call_ratio": _safe_float(self.put_call_ratio),
            "risk_reversal_25d": _safe_float(self.risk_reversal_25d),
            "iv_skew": _safe_float(self.iv_skew),
            "vix_term_structure": _safe_float(self.vix_term_structure),
            "skew_index": _safe_float(self.skew_index),
            "variance_risk_premium": _safe_float(self.variance_risk_premium),
            "n_calls": self.n_calls,
            "n_puts": self.n_puts,
            "nearest_expiry": self.nearest_expiry,
        }


def _safe_float(val: float | None) -> float | None:
    """Return None for NaN/Inf, else rounded float."""
    if val is None:
        return None
    try:
        f = float(val)
        if math.isnan(f) or math.isinf(f):
            return None
        return round(f, 6)
    except (TypeError, ValueError):
        return None


# ---------------------------------------------------------------------------
# Black-Scholes delta approximation (for 25-delta strike interpolation)
# ---------------------------------------------------------------------------

def _bs_delta_call(S: float, K: float, T: float, sigma: float, r: float = 0.05) -> float:
    """Approximate Black-Scholes call delta.

    Parameters
    ----------
    S : spot price
    K : strike price
    T : time to expiry in years
    sigma : implied volatility (annualized)
    r : risk-free rate
    """
    if T <= 0 or sigma <= 0 or S <= 0 or K <= 0:
        return 0.0
    from scipy.stats import norm
    d1 = (math.log(S / K) + (r + 0.5 * sigma ** 2) * T) / (sigma * math.sqrt(T))
    return float(norm.cdf(d1))


def _interpolate_iv_at_delta(
    chain_df: pd.DataFrame,
    spot: float,
    T: float,
    target_delta: float,
    is_call: bool = True,
) -> float | None:
    """Interpolate IV at a target delta from an options chain.

    Finds the two strikes whose deltas bracket the target and linearly
    interpolates the IV between them.

    Parameters
    ----------
    chain_df : DataFrame with 'strike' and 'impliedVolatility' columns
    spot : current stock price
    T : time to expiry in years
    target_delta : desired delta (e.g. 0.25 for 25-delta)
    is_call : True for calls, False for puts

    Returns
    -------
    Interpolated IV at the target delta, or None if interpolation fails.
    """
    if chain_df.empty or "strike" not in chain_df.columns:
        return None

    iv_col = "impliedVolatility"
    if iv_col not in chain_df.columns:
        return None

    # Filter to rows with valid IV
    valid = chain_df[chain_df[iv_col].notna() & (chain_df[iv_col] > 0.01)].copy()
    if len(valid) < 2:
        return None

    # Compute delta for each strike
    deltas = []
    for _, row in valid.iterrows():
        K = float(row["strike"])
        sigma = float(row[iv_col])
        d = _bs_delta_call(spot, K, T, sigma)
        if not is_call:
            d = d - 1.0  # put delta = call delta - 1
            d = abs(d)   # work with absolute delta for puts
        deltas.append(d)

    valid = valid.copy()
    valid["delta"] = deltas

    # Sort by distance to target delta
    valid["delta_dist"] = (valid["delta"] - target_delta).abs()
    valid = valid.sort_values("delta_dist")

    if len(valid) < 2:
        return float(valid.iloc[0][iv_col])

    # Take two closest strikes and interpolate
    row1 = valid.iloc[0]
    row2 = valid.iloc[1]

    d1, iv1 = float(row1["delta"]), float(row1[iv_col])
    d2, iv2 = float(row2["delta"]), float(row2[iv_col])

    if abs(d2 - d1) < 1e-8:
        return float(iv1)

    # Linear interpolation
    weight = (target_delta - d1) / (d2 - d1)
    weight = max(0.0, min(1.0, weight))  # clamp
    iv = iv1 + weight * (iv2 - iv1)
    return float(iv)


# ---------------------------------------------------------------------------
# Individual signal computations
# ---------------------------------------------------------------------------

def _compute_put_call_ratio(
    calls_df: pd.DataFrame,
    puts_df: pd.DataFrame,
    min_volume: int = 100,
) -> float | None:
    """Compute put/call volume ratio from options chain.

    Parameters
    ----------
    calls_df : call options chain
    puts_df : put options chain
    min_volume : minimum total volume to consider valid

    Returns
    -------
    Put volume / call volume ratio, or None if insufficient data.
    """
    call_vol = 0
    put_vol = 0

    if "volume" in calls_df.columns:
        call_vol = int(calls_df["volume"].fillna(0).sum())
    if "volume" in puts_df.columns:
        put_vol = int(puts_df["volume"].fillna(0).sum())

    if call_vol < min_volume and put_vol < min_volume:
        return None

    if call_vol == 0:
        return 5.0  # cap at 5.0 for extreme bearish

    return round(put_vol / call_vol, 4)


def _compute_risk_reversal(
    calls_df: pd.DataFrame,
    puts_df: pd.DataFrame,
    spot: float,
    T: float,
) -> float | None:
    """Compute 25-delta risk reversal (call IV - put IV at 25-delta).

    Negative value = puts more expensive than calls = bearish expectations.
    This is the single best options-derived predictor of large moves.

    Parameters
    ----------
    calls_df : call options chain with strike and impliedVolatility
    puts_df : put options chain with strike and impliedVolatility
    spot : current stock price
    T : time to expiry in years
    """
    call_iv_25d = _interpolate_iv_at_delta(calls_df, spot, T, 0.25, is_call=True)
    put_iv_25d = _interpolate_iv_at_delta(puts_df, spot, T, 0.25, is_call=False)

    if call_iv_25d is None or put_iv_25d is None:
        return None

    return round(call_iv_25d - put_iv_25d, 6)


def _compute_iv_skew(
    puts_df: pd.DataFrame,
    spot: float,
) -> float | None:
    """Compute IV skew: OTM put IV / ATM IV.

    Values above 1.2 indicate elevated tail fear (puts are disproportionately
    expensive relative to ATM options).

    Parameters
    ----------
    puts_df : put options chain with strike and impliedVolatility
    spot : current stock price
    """
    iv_col = "impliedVolatility"
    if puts_df.empty or iv_col not in puts_df.columns:
        return None

    valid = puts_df[puts_df[iv_col].notna() & (puts_df[iv_col] > 0.01)].copy()
    if len(valid) < 3:
        return None

    # ATM: strike closest to spot
    valid["moneyness"] = (valid["strike"] / spot).astype(float)

    atm_mask = (valid["moneyness"] > 0.95) & (valid["moneyness"] < 1.05)
    atm_rows = valid[atm_mask]
    if atm_rows.empty:
        # Fallback: closest strike to spot
        valid["dist"] = (valid["strike"] - spot).abs()
        atm_iv = float(valid.sort_values("dist").iloc[0][iv_col])
    else:
        atm_iv = float(atm_rows[iv_col].mean())

    if atm_iv < 0.01:
        return None

    # OTM put: strike at ~90% of spot
    otm_mask = (valid["moneyness"] > 0.85) & (valid["moneyness"] < 0.95)
    otm_rows = valid[otm_mask]
    if otm_rows.empty:
        return None

    otm_iv = float(otm_rows[iv_col].mean())

    return round(otm_iv / atm_iv, 4)


def _fetch_vix_term_structure() -> float | None:
    """Fetch VIX term structure: VIX / VIX3M ratio.

    Above 1.0 = backwardation = market expects near-term stress to exceed
    longer-term expectations (historically precedes selloffs).
    """
    try:
        import yfinance as yf

        # Batch fetch VIX and VIX3M
        data = yf.download(
            ["^VIX", "^VIX3M"],
            period="5d",
            progress=False,
            threads=False,
        )

        if data.empty:
            return None

        # yfinance multi-ticker returns MultiIndex columns: (metric, ticker)
        close = data.get("Close", data)

        vix_val = None
        vix3m_val = None

        if isinstance(close.columns, pd.MultiIndex):
            if "^VIX" in close.columns.get_level_values(-1):
                vix_series = close.xs("^VIX", axis=1, level=-1) if close.columns.nlevels > 1 else close["^VIX"]
                vix_val = float(vix_series.dropna().iloc[-1])
            if "^VIX3M" in close.columns.get_level_values(-1):
                vix3m_series = close.xs("^VIX3M", axis=1, level=-1) if close.columns.nlevels > 1 else close["^VIX3M"]
                vix3m_val = float(vix3m_series.dropna().iloc[-1])
        else:
            if "^VIX" in close.columns:
                vix_val = float(close["^VIX"].dropna().iloc[-1])
            if "^VIX3M" in close.columns:
                vix3m_val = float(close["^VIX3M"].dropna().iloc[-1])

        if vix_val is None or vix3m_val is None or vix3m_val < 1.0:
            return None

        return round(vix_val / vix3m_val, 4)

    except Exception as exc:
        logger.debug("VIX term structure fetch failed: %s", exc)
        return None


def _fetch_skew_index() -> float | None:
    """Fetch CBOE SKEW index value.

    The SKEW index measures tail risk priced into S&P 500 options.
    Range: ~100 (no tail risk) to ~170 (extreme tail risk priced in).
    """
    try:
        import yfinance as yf

        skew = yf.Ticker("^SKEW")
        hist = skew.history(period="5d")
        if hist.empty:
            return None

        return round(float(hist["Close"].dropna().iloc[-1]), 2)

    except Exception as exc:
        logger.debug("SKEW index fetch failed: %s", exc)
        return None


def _compute_variance_risk_premium(
    iv30: float | None,
    realized_vol: float | None,
) -> float | None:
    """Compute variance risk premium: IV^2 - RV^2 (annualized).

    Positive = market expects more volatility than recently realized.
    Bollerslev, Tauchen & Zhou (2009) show this predicts equity returns
    1-6 months ahead.

    Parameters
    ----------
    iv30 : 30-day implied volatility (annualized, e.g. 0.25 for 25%)
    realized_vol : realized volatility (annualized, e.g. 0.20 for 20%)
    """
    if iv30 is None or realized_vol is None:
        return None
    if iv30 < 0.001 or realized_vol < 0.001:
        return None

    return round(iv30 ** 2 - realized_vol ** 2, 6)


# ---------------------------------------------------------------------------
# Main entry point
# ---------------------------------------------------------------------------

def compute_options_signals(
    cache: pd.DataFrame,
    ticker: str,
    market_id: str = "",
) -> tuple[pd.DataFrame, OptionsSignalResult]:
    """Compute options-derived forward-looking signals and inject into cache.

    Only works for US-listed stocks with liquid options chains (via yfinance).
    For non-US markets, returns the cache unchanged with an empty result.

    Parameters
    ----------
    cache : daily cache DataFrame
    ticker : stock ticker symbol (e.g. "AAPL")
    market_id : market identifier (options only available for US markets)

    Returns
    -------
    (cache, OptionsSignalResult) -- cache with 6 new constant columns
    """
    result = OptionsSignalResult()

    # Options chains are only reliably available for US-listed stocks
    # via yfinance. Non-US markets return empty chains.
    us_markets = {"us_sec_edgar", ""}
    if market_id and market_id not in us_markets:
        logger.debug("Options signals skipped for non-US market: %s", market_id)
        return cache, result

    if not ticker:
        return cache, result

    try:
        import yfinance as yf
    except ImportError:
        logger.debug("yfinance not available for options signals")
        return cache, result

    try:
        tick = yf.Ticker(ticker)

        # Get available expiration dates
        expirations = tick.options
        if not expirations:
            result.error = "no options expirations available"
            return cache, result

        # Use the nearest expiration (most liquid, most informative)
        nearest_exp = expirations[0]
        result.nearest_expiry = nearest_exp

        # Compute time to expiry in years
        from datetime import datetime
        exp_date = datetime.strptime(nearest_exp, "%Y-%m-%d")
        now = datetime.now()
        T = max((exp_date - now).days / 365.0, 0.001)

        # Fetch the options chain
        chain = tick.option_chain(nearest_exp)
        calls_df = chain.calls
        puts_df = chain.puts

        result.n_calls = len(calls_df)
        result.n_puts = len(puts_df)

        if calls_df.empty and puts_df.empty:
            result.error = "empty options chain"
            return cache, result

        # Get current spot price from cache
        spot = None
        if "close" in cache.columns and cache["close"].notna().any():
            spot = float(cache["close"].dropna().iloc[-1])

        if spot is None or spot <= 0:
            result.error = "no spot price available"
            return cache, result

        # ---- Compute all 6 signals ----

        # 1. Put/Call Ratio
        pcr = _compute_put_call_ratio(calls_df, puts_df)
        result.put_call_ratio = pcr
        if pcr is not None:
            cache["put_call_ratio"] = pcr

        # 2. Risk Reversal (25-delta)
        rr = _compute_risk_reversal(calls_df, puts_df, spot, T)
        result.risk_reversal_25d = rr
        if rr is not None:
            cache["risk_reversal_25d"] = rr

        # 3. IV Skew (OTM put IV / ATM IV)
        skew = _compute_iv_skew(puts_df, spot)
        result.iv_skew = skew
        if skew is not None:
            cache["iv_skew"] = skew

        # 4. VIX Term Structure
        vts = _fetch_vix_term_structure()
        result.vix_term_structure = vts
        if vts is not None:
            cache["vix_term_structure"] = vts

        # 5. SKEW Index
        skew_idx = _fetch_skew_index()
        result.skew_index = skew_idx
        if skew_idx is not None:
            cache["skew_index"] = skew_idx

        # 6. Variance Risk Premium
        iv30 = None
        if "iv30" in cache.columns and cache["iv30"].notna().any():
            iv30 = float(cache["iv30"].dropna().iloc[-1])

        rv = None
        if "volatility_21d" in cache.columns and cache["volatility_21d"].notna().any():
            rv = float(cache["volatility_21d"].dropna().iloc[-1])

        vrp = _compute_variance_risk_premium(iv30, rv)
        result.variance_risk_premium = vrp
        if vrp is not None:
            cache["variance_risk_premium"] = vrp

        result.available = True
        logger.info(
            "Options signals computed for %s: PCR=%.2f, RR25d=%s, skew=%s, VTS=%s, SKEW=%s, VRP=%s",
            ticker,
            pcr if pcr is not None else 0,
            f"{rr:.4f}" if rr is not None else "N/A",
            f"{skew:.3f}" if skew is not None else "N/A",
            f"{vts:.3f}" if vts is not None else "N/A",
            f"{skew_idx:.1f}" if skew_idx is not None else "N/A",
            f"{vrp:.6f}" if vrp is not None else "N/A",
        )

    except Exception as exc:
        result.error = str(exc)
        logger.warning("Options signals failed for %s: %s", ticker, exc)

    return cache, result
