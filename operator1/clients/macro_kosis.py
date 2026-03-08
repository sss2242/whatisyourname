"""South Korea macro provider using FRED (Korean series) + KOSIS fallback.

Primary macro source for the Korean market.
Fallback: wbgapi (World Bank).

FRED hosts Korean macro data (via OECD). KOSIS API requires free registration.
"""

from __future__ import annotations

import logging
import os
from datetime import date, timedelta

import pandas as pd

logger = logging.getLogger(__name__)

# FRED hosts many Korean macro series via OECD data feeds.
_FRED_KR_SERIES: dict[str, str] = {
    "gdp_growth": "KORRGDPEXP",               # Korea Real GDP growth
    "inflation_rate_yoy": "FPCPITOTLZGKOR",   # Korea CPI inflation
    "unemployment_rate": "LRUNTTTTKOM156S",     # Korea unemployment rate (OECD)
    "interest_rate": "IRSTCI01KRM156N",         # Korea short-term interest rate
    "exchange_rate": "DEXKOUS",                 # KRW/USD exchange rate
}


def fetch_macro_kosis(
    years: int = 10,
) -> dict[str, pd.Series]:
    """Fetch Korean macro indicators.

    Uses FRED (which hosts Korean data via OECD) as primary path.
    Falls through to wbgapi if neither FRED nor KOSIS are available.

    Returns
    -------
    Dict mapping canonical indicator name -> pd.Series with DatetimeIndex.
    """
    results: dict[str, pd.Series] = {}

    # Path 1: FRED (has comprehensive Korean data via OECD)
    fred_key = os.environ.get("FRED_API_KEY", "")
    if fred_key:
        try:
            from fredapi import Fred
            fred = Fred(api_key=fred_key)
            start = date.today() - timedelta(days=365 * years)

            for canonical_name, series_id in _FRED_KR_SERIES.items():
                try:
                    series = fred.get_series(series_id, observation_start=start)
                    if series is not None and not series.empty:
                        series.name = canonical_name
                        results[canonical_name] = series
                        logger.debug("FRED-KR %s: %d obs", series_id, len(series))
                except Exception as exc:
                    logger.debug("FRED-KR failed for %s: %s", series_id, exc)

            if results:
                logger.info("Korea macro via FRED: %d/%d indicators", len(results), len(_FRED_KR_SERIES))
        except ImportError:
            logger.debug("fredapi not installed for KR macro")

    # Path 2: wbgapi fallback for any indicators FRED didn't return.
    _WB_KR_SERIES: dict[str, str] = {
        "gdp_growth": "NY.GDP.MKTP.KD.ZG",
        "inflation_rate_yoy": "FP.CPI.TOTL.ZG",
        "unemployment_rate": "SL.UEM.TOTL.ZS",
        "exchange_rate": "PA.NUS.FCRF",
    }
    missing = [k for k in _WB_KR_SERIES if k not in results]
    if missing:
        try:
            import wbgapi as wb
            start_year = date.today().year - years
            for canonical_name in missing:
                wb_id = _WB_KR_SERIES[canonical_name]
                try:
                    df = wb.data.DataFrame(wb_id, economy="KOR", time=range(start_year, date.today().year + 1))
                    if df is not None and not df.empty:
                        # wbgapi returns year columns like YR2020; transpose to series
                        series = df.iloc[0].dropna()
                        if not series.empty:
                            idx = pd.to_datetime([f"{str(y).replace('YR','')}-12-31" for y in series.index], errors="coerce")
                            s = pd.Series(series.values, index=idx, name=canonical_name, dtype=float)
                            s = s.dropna()
                            if not s.empty:
                                results[canonical_name] = s
                                logger.debug("wbgapi-KR %s: %d obs", wb_id, len(s))
                except Exception as exc:
                    logger.debug("wbgapi-KR failed for %s: %s", wb_id, exc)
            if any(k in results for k in missing):
                logger.info("Korea macro via wbgapi fallback: filled %d/%d missing indicators",
                            sum(1 for k in missing if k in results), len(missing))
        except ImportError:
            logger.debug("wbgapi not installed for KR macro fallback")

    if results:
        logger.info("Korea macro total: %d/%d indicators", len(results), len(_FRED_KR_SERIES))
    return results
