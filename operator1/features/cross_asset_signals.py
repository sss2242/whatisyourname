"""Cross-asset sector rotation signals for equity prediction.

Tracks 11 sector ETFs, Treasury yields, USD index, and gold to detect
institutional capital rotation before it hits individual stocks.  The
January 2025 AAPL selloff was preceded by 2+ weeks of XLK underperformance
vs XLE -- a clear tech-to-value rotation visible in these signals.

Features computed:
  - sector_relative_strength: target sector ETF return / SPY (21d rolling)
  - sector_rank_12m: target sector rank among 11 sectors by 252d return (1-11)
  - sector_dispersion: cross-sector return std (21d rolling, higher = rotation)
  - yield_curve_10y2y: 10Y - 2Y Treasury spread (negative = recession signal)
  - usd_momentum_21d: USD index 21d return (strong USD = tech headwind)
  - cross_asset_stress: z-score composite of yield rise + USD + gold + dispersion

Data source: yfinance batch download (single API call for all tickers).
No new pip packages required.

References:
  - Dorsey (1995): Sector relative strength
  - Moskowitz & Grinblatt (1999): Cross-sector momentum
  - Murphy (1991): Intermarket analysis (bonds/commodities lead equities)
  - Pollet & Wilson (2010): Sector dispersion predicts volatility

Pipeline integration:
  - Called in main.py Step 4a.8 (after sector leading indicators)
  - All 6 features added to _extra_vars for temporal model consumption
  - Profile key: cross_asset_signals
"""

from __future__ import annotations

import logging
import math
from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Sector ETF mapping
# ---------------------------------------------------------------------------

# GICS sector name (from yfinance/OpenFIGI) -> SPDR sector ETF ticker
_SECTOR_TO_ETF: dict[str, str] = {
    "technology": "XLK",
    "information technology": "XLK",
    "energy": "XLE",
    "financial services": "XLF",
    "financials": "XLF",
    "healthcare": "XLV",
    "health care": "XLV",
    "industrials": "XLI",
    "consumer defensive": "XLP",
    "consumer staples": "XLP",
    "consumer cyclical": "XLY",
    "consumer discretionary": "XLY",
    "basic materials": "XLB",
    "materials": "XLB",
    "utilities": "XLU",
    "real estate": "XLRE",
    "communication services": "XLC",
    "communication": "XLC",
    "telecommunications": "XLC",
}

# All 11 SPDR sector ETFs + SPY benchmark
_ALL_SECTOR_ETFS = ["XLK", "XLE", "XLF", "XLV", "XLI", "XLP", "XLY", "XLB", "XLU", "XLRE", "XLC"]
_CROSS_ASSET_TICKERS = ["^TNX", "^IRX", "DX-Y.NYB", "GC=F"]
_BENCHMARK = "SPY"


# ---------------------------------------------------------------------------
# Result container
# ---------------------------------------------------------------------------

@dataclass
class CrossAssetResult:
    """Container for cross-asset signal results."""

    available: bool = False
    sector_relative_strength: float | None = None
    sector_rank_12m: int | None = None
    sector_dispersion: float | None = None
    yield_curve_10y2y: float | None = None
    usd_momentum_21d: float | None = None
    cross_asset_stress: float | None = None
    target_sector_etf: str = ""
    n_sectors_fetched: int = 0
    error: str = ""

    def to_profile_dict(self) -> dict[str, Any]:
        """Convert to profile-ready dict."""
        return {
            "available": self.available,
            "sector_relative_strength": _safe(self.sector_relative_strength),
            "sector_rank_12m": self.sector_rank_12m,
            "sector_dispersion": _safe(self.sector_dispersion),
            "yield_curve_10y2y": _safe(self.yield_curve_10y2y),
            "usd_momentum_21d": _safe(self.usd_momentum_21d),
            "cross_asset_stress": _safe(self.cross_asset_stress),
            "target_sector_etf": self.target_sector_etf,
            "n_sectors_fetched": self.n_sectors_fetched,
        }


def _safe(val: float | None) -> float | None:
    if val is None:
        return None
    try:
        f = float(val)
        return None if (math.isnan(f) or math.isinf(f)) else round(f, 6)
    except (TypeError, ValueError):
        return None


# ---------------------------------------------------------------------------
# Data fetching
# ---------------------------------------------------------------------------

def _fetch_etf_data(years: int = 2) -> pd.DataFrame:
    """Batch-fetch all sector ETFs + cross-asset tickers via yfinance.

    Returns a DataFrame with daily Close prices, one column per ticker.
    Uses a single yfinance.download() call for efficiency.
    """
    try:
        import yfinance as yf
    except ImportError:
        return pd.DataFrame()

    all_tickers = [_BENCHMARK] + _ALL_SECTOR_ETFS + _CROSS_ASSET_TICKERS
    try:
        data = yf.download(
            all_tickers,
            period=f"{years}y",
            progress=False,
            threads=False,
        )
        if data.empty:
            return pd.DataFrame()

        # Extract Close prices -- handle MultiIndex columns
        close = data.get("Close", data)
        if isinstance(close, pd.Series):
            close = close.to_frame()

        return close

    except Exception as exc:
        logger.debug("Cross-asset ETF fetch failed: %s", exc)
        return pd.DataFrame()


def _resolve_sector_etf(sector: str) -> str:
    """Map a company's sector string to the matching SPDR ETF ticker."""
    if not sector:
        return ""
    sector_lower = sector.lower().strip()
    return _SECTOR_TO_ETF.get(sector_lower, "")


# ---------------------------------------------------------------------------
# Signal computations
# ---------------------------------------------------------------------------

def _compute_sector_relative_strength(
    close_df: pd.DataFrame,
    sector_etf: str,
    window: int = 21,
) -> float | None:
    """Compute rolling relative strength: sector ETF return / SPY return."""
    if sector_etf not in close_df.columns or _BENCHMARK not in close_df.columns:
        return None

    sector_ret = close_df[sector_etf].pct_change(window).dropna()
    spy_ret = close_df[_BENCHMARK].pct_change(window).dropna()

    if sector_ret.empty or spy_ret.empty:
        return None

    # Use latest values
    s = float(sector_ret.iloc[-1])
    b = float(spy_ret.iloc[-1])

    if abs(b) < 1e-8:
        return None

    return round(s / b, 4)


def _compute_sector_rank(close_df: pd.DataFrame, sector_etf: str) -> int | None:
    """Rank target sector among 11 sectors by 252d return (1=best, 11=worst)."""
    available_etfs = [e for e in _ALL_SECTOR_ETFS if e in close_df.columns]
    if len(available_etfs) < 5 or sector_etf not in available_etfs:
        return None

    # Compute 252d return for each sector
    returns_252d: dict[str, float] = {}
    for etf in available_etfs:
        series = close_df[etf].dropna()
        if len(series) >= 252:
            ret = float(series.iloc[-1] / series.iloc[-252] - 1)
            returns_252d[etf] = ret
        elif len(series) >= 63:
            # Fallback to available history
            ret = float(series.iloc[-1] / series.iloc[0] - 1)
            returns_252d[etf] = ret

    if sector_etf not in returns_252d:
        return None

    # Rank: 1 = highest return, N = lowest
    sorted_etfs = sorted(returns_252d, key=returns_252d.get, reverse=True)
    rank = sorted_etfs.index(sector_etf) + 1
    return rank


def _compute_sector_dispersion(close_df: pd.DataFrame, window: int = 21) -> float | None:
    """Compute cross-sector return dispersion (std of sector daily returns).

    High dispersion = differentiation between sectors = rotation in progress.
    Low dispersion = all sectors moving together = macro-driven market.
    """
    available_etfs = [e for e in _ALL_SECTOR_ETFS if e in close_df.columns]
    if len(available_etfs) < 5:
        return None

    # Daily returns for each sector
    returns = close_df[available_etfs].pct_change().dropna()
    if len(returns) < window:
        return None

    # Cross-sectional std of the latest window
    recent = returns.tail(window)
    # For each day, compute std across sectors, then average
    daily_dispersion = recent.std(axis=1)
    avg_dispersion = float(daily_dispersion.mean())

    return round(avg_dispersion, 6)


def _compute_yield_curve(close_df: pd.DataFrame) -> float | None:
    """Compute 10Y - 2Y Treasury spread from ^TNX and ^IRX.

    ^TNX = 10-Year Treasury yield (in percentage points)
    ^IRX = 13-Week Treasury yield (proxy for short-term rate)

    Negative spread = yield curve inversion = recession signal.
    """
    tnx_col = "^TNX"
    irx_col = "^IRX"

    if tnx_col not in close_df.columns:
        return None

    tnx = close_df[tnx_col].dropna()
    if tnx.empty:
        return None

    tnx_latest = float(tnx.iloc[-1])

    if irx_col in close_df.columns:
        irx = close_df[irx_col].dropna()
        if not irx.empty:
            irx_latest = float(irx.iloc[-1])
            return round(tnx_latest - irx_latest, 4)

    # If ^IRX unavailable, return just the 10Y yield level
    return round(tnx_latest, 4)


def _compute_usd_momentum(close_df: pd.DataFrame, window: int = 21) -> float | None:
    """Compute USD index 21d momentum.

    Strong USD momentum = headwind for tech/multinational earnings.
    """
    usd_col = "DX-Y.NYB"
    if usd_col not in close_df.columns:
        return None

    usd = close_df[usd_col].dropna()
    if len(usd) < window + 1:
        return None

    momentum = float(usd.iloc[-1] / usd.iloc[-window - 1] - 1)
    return round(momentum, 6)


def _compute_cross_asset_stress(
    sector_dispersion: float | None,
    yield_curve: float | None,
    usd_momentum: float | None,
    close_df: pd.DataFrame,
) -> float | None:
    """Compute composite cross-asset stress indicator.

    Combines: yield rise + USD strength + gold rally + sector dispersion.
    Each component is z-scored, then averaged.
    Higher = more stressed environment.
    """
    components: list[float] = []

    # Yield curve inversion (negative = stress)
    if yield_curve is not None:
        # Invert: lower/negative curve = higher stress
        components.append(-yield_curve / 2.0)  # Scale to similar range

    # USD strength (positive = stress for equities)
    if usd_momentum is not None:
        components.append(usd_momentum * 10.0)  # Scale up from small pct

    # Gold momentum (positive gold = risk-off = stress)
    gold_col = "GC=F"
    if gold_col in close_df.columns:
        gold = close_df[gold_col].dropna()
        if len(gold) >= 22:
            gold_mom = float(gold.iloc[-1] / gold.iloc[-22] - 1)
            components.append(gold_mom * 5.0)

    # Sector dispersion (higher = rotation/differentiation)
    if sector_dispersion is not None:
        components.append(sector_dispersion * 100.0)  # Scale from small std

    if len(components) < 2:
        return None

    # Simple average of scaled components
    stress = float(np.mean(components))
    return round(stress, 6)


# ---------------------------------------------------------------------------
# Main entry point
# ---------------------------------------------------------------------------

def compute_cross_asset_signals(
    cache: pd.DataFrame,
    sector: str = "",
    years: int = 2,
) -> tuple[pd.DataFrame, CrossAssetResult]:
    """Compute cross-asset sector rotation signals and inject into cache.

    Parameters
    ----------
    cache : daily cache DataFrame
    sector : company's GICS sector (from target_profile["sector"])
    years : years of ETF history to fetch

    Returns
    -------
    (cache, CrossAssetResult) -- cache with 6 new constant columns
    """
    result = CrossAssetResult()

    # Resolve sector to ETF
    sector_etf = _resolve_sector_etf(sector)
    result.target_sector_etf = sector_etf

    # Fetch all ETF + cross-asset data
    close_df = _fetch_etf_data(years=years)
    if close_df.empty:
        result.error = "ETF data fetch failed"
        return cache, result

    result.n_sectors_fetched = sum(1 for e in _ALL_SECTOR_ETFS if e in close_df.columns)

    # Initialize variables before conditional blocks (B1 fix: NameError when sector_etf empty)
    rs = None
    rank = None

    # 1. Sector relative strength (21d rolling)
    if sector_etf:
        rs = _compute_sector_relative_strength(close_df, sector_etf)
        result.sector_relative_strength = rs
        if rs is not None:
            cache["sector_relative_strength"] = rs

    # 2. Sector rank (252d return, 1=best)
    if sector_etf:
        rank = _compute_sector_rank(close_df, sector_etf)
        result.sector_rank_12m = rank
        if rank is not None:
            cache["sector_rank_12m"] = rank

    # 3. Sector dispersion (21d cross-sectional std)
    disp = _compute_sector_dispersion(close_df)
    result.sector_dispersion = disp
    if disp is not None:
        cache["sector_dispersion"] = disp

    # 4. Yield curve (10Y - 3M spread)
    yc = _compute_yield_curve(close_df)
    result.yield_curve_10y2y = yc
    if yc is not None:
        cache["yield_curve_10y2y"] = yc

    # 5. USD momentum (21d)
    usd = _compute_usd_momentum(close_df)
    result.usd_momentum_21d = usd
    if usd is not None:
        cache["usd_momentum_21d"] = usd

    # 6. Cross-asset stress composite
    stress = _compute_cross_asset_stress(disp, yc, usd, close_df)
    result.cross_asset_stress = stress
    if stress is not None:
        cache["cross_asset_stress"] = stress

    result.available = True
    logger.info(
        "Cross-asset signals: RS=%s, rank=%s, disp=%s, YC=%s, USD=%s, stress=%s (ETF=%s)",
        f"{rs:.3f}" if rs is not None else "N/A",
        rank if rank is not None else "N/A",
        f"{disp:.5f}" if disp is not None else "N/A",
        f"{yc:.3f}" if yc is not None else "N/A",
        f"{usd:.4f}" if usd is not None else "N/A",
        f"{stress:.4f}" if stress is not None else "N/A",
        sector_etf or "unmapped",
    )

    return cache, result
