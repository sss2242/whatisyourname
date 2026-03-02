"""Taiwan macro provider using FRED + IMF fallback.

Primary: FRED (Taiwanese series via IMF/OECD feeds) -- needs FRED_API_KEY.
Secondary: IMF SDMX JSON API (free, no key) -- for GDP growth & unemployment.

Taiwan is not a World Bank member, so wbgapi has zero coverage.
The IMF includes Taiwan as "TW" (or "TWN" in some datasets).
"""

from __future__ import annotations

import logging
import os
from datetime import date, timedelta

import pandas as pd

logger = logging.getLogger(__name__)

# FRED series for Taiwan (via IMF, OECD, central bank sources)
_FRED_TW_SERIES: dict[str, str] = {
    "inflation_rate_yoy": "FPCPITOTLZGTWN",   # Taiwan CPI inflation (WB via FRED)
    "interest_rate": "IRSTCI01TWM156N",         # Taiwan short-term rate (OECD)
    "exchange_rate": "DEXTAUS",                  # TWD/USD exchange rate
}

# IMF SDMX JSON API -- free, no key needed
# Taiwan = "TW" in IMF WEO / IFS datasets
_IMF_BASE = "http://dataservices.imf.org/REST/SDMX_JSON.svc"
_IMF_TW_SERIES: dict[str, str] = {
    # GDP real growth rate (Annual, IFS dataset)
    "gdp_growth": f"{_IMF_BASE}/CompactData/IFS/A.TW.NGDP_R_PC_PP_PT",
    # Unemployment rate (Annual, IFS dataset)
    "unemployment_rate": f"{_IMF_BASE}/CompactData/IFS/A.TW.LUR_PT",
}


def _fetch_imf_series(url: str, canonical_name: str, years: int) -> pd.Series | None:
    """Fetch a single IMF SDMX JSON series."""
    try:
        import requests
        resp = requests.get(
            url,
            headers={"Accept": "application/json", "User-Agent": "Operator1/1.0"},
            timeout=30,
        )
        if resp.status_code != 200:
            logger.debug("IMF %s: HTTP %d", canonical_name, resp.status_code)
            return None

        data = resp.json()
        ds = data.get("CompactData", {}).get("DataSet", {})
        series_data = ds.get("Series", {})
        obs_list = series_data.get("Obs", []) if isinstance(series_data, dict) else []

        if not obs_list:
            return None

        # Filter to recent years
        cutoff_year = date.today().year - years
        values = []
        dates = []
        for obs in obs_list:
            year_str = obs.get("@TIME_PERIOD", "")
            val_str = obs.get("@OBS_VALUE", "")
            try:
                year = int(year_str[:4])
                if year >= cutoff_year:
                    values.append(float(val_str))
                    dates.append(pd.Timestamp(year=year, month=1, day=1))
            except (ValueError, TypeError):
                continue

        if not values:
            return None

        s = pd.Series(values, index=dates)
        s.name = canonical_name
        logger.debug("IMF-TW %s: %d obs", canonical_name, len(s))
        return s

    except Exception as exc:
        logger.debug("IMF-TW %s failed: %s", canonical_name, exc)
        return None


def fetch_macro_dgbas(
    years: int = 10,
) -> dict[str, pd.Series]:
    """Fetch Taiwanese macro indicators.

    Uses FRED as primary (inflation, interest rate, exchange rate).
    Uses IMF as secondary (GDP growth, unemployment -- not on FRED for Taiwan).
    Taiwan is not a World Bank member, so wbgapi has zero coverage.

    Returns
    -------
    Dict mapping canonical indicator name -> pd.Series with DatetimeIndex.
    """
    results: dict[str, pd.Series] = {}

    # Path 1: FRED for inflation, interest rate, exchange rate
    fred_key = os.environ.get("FRED_API_KEY", "")
    if fred_key:
        try:
            from fredapi import Fred
            fred = Fred(api_key=fred_key)
            start = date.today() - timedelta(days=365 * years)

            for canonical_name, series_id in _FRED_TW_SERIES.items():
                try:
                    series = fred.get_series(series_id, observation_start=start)
                    if series is not None and not series.empty:
                        series.name = canonical_name
                        results[canonical_name] = series
                        logger.debug("FRED-TW %s: %d obs", series_id, len(series))
                except Exception as exc:
                    logger.debug("FRED-TW failed for %s: %s", series_id, exc)

            if results:
                logger.info("Taiwan FRED: %d/%d indicators", len(results), len(_FRED_TW_SERIES))
        except ImportError:
            logger.debug("fredapi not installed for TW macro")

    # Path 2: IMF for GDP growth and unemployment (free, no key)
    for canonical_name, url in _IMF_TW_SERIES.items():
        if canonical_name not in results:
            s = _fetch_imf_series(url, canonical_name, years)
            if s is not None:
                results[canonical_name] = s

    if results:
        logger.info("Taiwan macro total: %d indicators (%s)",
                     len(results), ", ".join(sorted(results.keys())))
    else:
        logger.debug("Taiwan macro: no data from FRED or IMF")

    return results
