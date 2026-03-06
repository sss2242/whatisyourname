"""US macro provider using fredapi (Federal Reserve FRED).

Primary macro source for US market.
Fallback: wbgapi (World Bank).

Requires FRED_API_KEY (free registration).
Research: .roo/research/macro-per-region-2026-02-25.md
Package: fredapi 0.5.2
"""

from __future__ import annotations

import logging
import os

import pandas as pd

logger = logging.getLogger(__name__)

# FRED series IDs for canonical macro variables (US)
_FRED_SERIES: dict[str, str] = {
    "gdp_growth": "A191RL1Q225SBEA",  # Real GDP growth (quarterly, annualized)
    "inflation_rate_yoy": "CPIAUCSL",  # CPI for All Urban Consumers
    "unemployment_rate": "UNRATE",  # Unemployment Rate
    "interest_rate": "FEDFUNDS",  # Federal Funds Rate
    "exchange_rate": "DTWEXBGS",  # Trade Weighted US Dollar Index
}

# FRED series IDs for other countries (monthly/quarterly where available)
_FRED_COUNTRY_SERIES: dict[str, dict[str, str]] = {
    "CA": {
        "gdp_growth": "NAEXKP01CAQ189S",   # Canada GDP growth (quarterly)
        "inflation_rate_yoy": "CPALCY01CAM661N",  # Canada CPI YoY
        "unemployment_rate": "LRUNTTTTCAM156S",  # Canada unemployment
        "interest_rate": "IR3TIB01CAM156N",  # Canada 3-month interbank rate
        "exchange_rate": "DEXCAUS",  # USD/CAD exchange rate
    },
    "AU": {
        "gdp_growth": "NAEXKP01AUQ189S",   # Australia GDP growth
        "inflation_rate_yoy": "CPALCY01AUQ657N",  # Australia CPI YoY (quarterly)
        "unemployment_rate": "LRUNTTTTAUM156S",  # Australia unemployment
        "interest_rate": "IR3TIB01AUM156N",  # Australia 3-month rate
        "exchange_rate": "DEXUSAL",  # USD/AUD exchange rate
    },
    "HK": {
        "gdp_growth": "NYGDPMKTPCDLHKM",  # Hong Kong GDP (annual proxy)
        "inflation_rate_yoy": "FPCPITOTLZGHKM",  # Hong Kong CPI
        "unemployment_rate": "LRUNTTTTHAM156S",  # Hong Kong unemployment
        "interest_rate": "IR3TIB01HKM156N",  # HK 3-month HIBOR
        "exchange_rate": "DEXHKUS",  # USD/HKD exchange rate
    },
    "SG": {
        "gdp_growth": "NAEXKP01SGQ189S",   # Singapore GDP growth
        "inflation_rate_yoy": "CPALCY01SGM661N",  # Singapore CPI YoY
        "unemployment_rate": "LRUNTTTTSGQ156S",  # Singapore unemployment (quarterly)
        "interest_rate": "IR3TIB01SGM156N",  # Singapore 3-month rate
        "exchange_rate": "DEXSIUS",  # USD/SGD exchange rate
    },
    "ZA": {
        "gdp_growth": "NAEXKP01ZAQ189S",   # South Africa GDP growth
        "inflation_rate_yoy": "CPALCY01ZAM661N",  # South Africa CPI YoY
        "unemployment_rate": "LRUNTTTTNAM156S",  # South Africa unemployment proxy
        "interest_rate": "IR3TIB01ZAM156N",  # South Africa 3-month rate
        "exchange_rate": "DEXSFUS",  # USD/ZAR exchange rate
    },
    "CN": {
        "gdp_growth": "NAEXKP01CNQ189S",   # China GDP growth
        "inflation_rate_yoy": "CPALCY01CNM661N",  # China CPI YoY
        "unemployment_rate": "LRUNTTTTCNM156S",  # China unemployment
        "interest_rate": "IR3TIB01CNM156N",  # China 3-month rate
        "exchange_rate": "DEXCHUS",  # USD/CNY exchange rate
    },
    "IN": {
        "gdp_growth": "NAEXKP01INQ189S",   # India GDP growth
        "inflation_rate_yoy": "CPALCY01INM661N",  # India CPI YoY
        "unemployment_rate": "LRUNTTTTINM156S",  # India unemployment
        "interest_rate": "IR3TIB01INM156N",  # India 3-month rate
        "exchange_rate": "DEXINUS",  # USD/INR exchange rate
    },
}


def fetch_macro_fred(
    api_key: str = "",
    years: int = 10,
) -> dict[str, pd.Series]:
    """Fetch US macro indicators from FRED.

    Parameters
    ----------
    api_key:
        FRED API key. Falls back to FRED_API_KEY env var.
    years:
        Number of years of history.

    Returns
    -------
    Dict mapping canonical indicator name -> pd.Series with DatetimeIndex.
    """
    key = api_key or os.environ.get("FRED_API_KEY", "")
    if not key:
        logger.debug("FRED_API_KEY not set; falling back to wbgapi")
        return {}

    try:
        from fredapi import Fred
    except ImportError:
        logger.debug("fredapi not installed; falling back to wbgapi")
        return {}

    try:
        fred = Fred(api_key=key)
    except Exception as exc:
        logger.warning("fredapi init failed: %s", exc)
        return {}

    from datetime import date, timedelta
    start = date.today() - timedelta(days=365 * years)
    results: dict[str, pd.Series] = {}

    for canonical_name, series_id in _FRED_SERIES.items():
        try:
            series = fred.get_series(series_id, observation_start=start)
            if series is not None and not series.empty:
                series.name = canonical_name
                results[canonical_name] = series
                logger.debug("FRED %s: %d observations", series_id, len(series))
        except Exception as exc:
            logger.debug("FRED failed for %s: %s", series_id, exc)

    logger.info("FRED fetched %d/%d indicators", len(results), len(_FRED_SERIES))
    return results


def fetch_macro_fred_country(
    country_iso2: str,
    api_key: str = "",
    years: int = 10,
) -> dict[str, pd.Series]:
    """Fetch macro indicators from FRED for a non-US country.

    FRED hosts international economic data from OECD, IMF, and central
    banks.  This function uses country-specific series IDs defined in
    ``_FRED_COUNTRY_SERIES``.

    Parameters
    ----------
    country_iso2:
        ISO-2 country code (e.g. "CA", "AU", "HK").
    api_key:
        FRED API key. Falls back to FRED_API_KEY env var.
    years:
        Number of years of history.

    Returns
    -------
    Dict mapping canonical indicator name -> pd.Series.
    """
    cc = country_iso2.upper()
    series_map = _FRED_COUNTRY_SERIES.get(cc)
    if not series_map:
        logger.debug("No FRED series defined for country %s", cc)
        return {}

    key = api_key or os.environ.get("FRED_API_KEY", "")
    if not key:
        logger.debug("FRED_API_KEY not set; skipping FRED for %s", cc)
        return {}

    try:
        from fredapi import Fred
    except ImportError:
        logger.debug("fredapi not installed; skipping FRED for %s", cc)
        return {}

    try:
        fred = Fred(api_key=key)
    except Exception as exc:
        logger.warning("fredapi init failed for %s: %s", cc, exc)
        return {}

    from datetime import date, timedelta
    start = date.today() - timedelta(days=365 * years)
    results: dict[str, pd.Series] = {}

    for canonical_name, series_id in series_map.items():
        try:
            series = fred.get_series(series_id, observation_start=start)
            if series is not None and not series.empty:
                series.name = canonical_name
                results[canonical_name] = series
                logger.debug("FRED-%s %s: %d observations", cc, series_id, len(series))
        except Exception as exc:
            logger.debug("FRED-%s failed for %s: %s", cc, series_id, exc)

    logger.info("FRED-%s fetched %d/%d indicators", cc, len(results), len(series_map))
    return results
