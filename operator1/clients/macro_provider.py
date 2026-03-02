"""Macro provider dispatcher -- routes to per-region primary or wbgapi fallback.

Each market has a primary macro source (government central bank API).
If the primary fails or its library is not installed, falls back to
wbgapi (World Bank, global coverage, no key needed).

Research: .roo/research/macro-per-region-2026-02-25.md
"""

from __future__ import annotations

import logging

import pandas as pd

logger = logging.getLogger(__name__)

# Country code -> primary macro fetcher
_PRIMARY_FETCHERS: dict[str, str] = {
    "US": "fred",
    "EU": "ecb",
    "DE": "ecb",
    "FR": "ecb",
    "NL": "ecb",    # Netherlands -- eurozone, use ECB
    "ES": "ecb",    # Spain -- eurozone, use ECB
    "IT": "ecb",    # Italy -- eurozone, use ECB
    "BR": "bcb",
    "MX": "banxico",
    "GB": "ons",       # UK -- ONS (no key needed)
    "JP": "estat",     # Japan -- FRED (JP series) / e-Stat
    "KR": "kosis",     # Korea -- FRED (KR series) / KOSIS
    "TW": "dgbas",     # Taiwan -- FRED (TW series) / DGBAS
    "CL": "bcch",      # Chile -- FRED (CL series) / BCCh
}


def fetch_macro(
    country_iso2: str,
    market_id: str = "",
    secrets: dict | None = None,
    years: int = 10,
) -> dict[str, pd.Series]:
    """Fetch macro data for a country, using per-region primary + wbgapi fallback.

    Parameters
    ----------
    country_iso2:
        ISO-2 country code (e.g. "US", "JP", "BR").
    market_id:
        Market identifier (for logging).
    secrets:
        API key dictionary.
    years:
        Number of years of history.

    Returns
    -------
    Dict mapping canonical indicator name -> pd.Series with DatetimeIndex.
    """
    if secrets is None:
        secrets = {}

    results: dict[str, pd.Series] = {}
    cc = country_iso2.upper()

    # Try per-region primary first
    primary = _PRIMARY_FETCHERS.get(cc)

    if primary == "fred":
        try:
            from operator1.clients.macro_fredapi import fetch_macro_fred
            results = fetch_macro_fred(
                api_key=secrets.get("FRED_API_KEY", ""),
                years=years,
            )
        except Exception as exc:
            logger.debug("FRED primary failed: %s", exc)

    elif primary == "ecb":
        try:
            from operator1.clients.macro_sdmx import fetch_macro_ecb
            results = fetch_macro_ecb(years=years)
        except Exception as exc:
            logger.debug("ECB primary failed: %s", exc)

    elif primary == "bcb":
        try:
            from operator1.clients.macro_bcb import fetch_macro_bcb
            results = fetch_macro_bcb(years=years)
        except Exception as exc:
            logger.debug("BCB primary failed: %s", exc)

    elif primary == "banxico":
        try:
            from operator1.clients.macro_banxico import fetch_macro_banxico
            results = fetch_macro_banxico(
                api_token=secrets.get("BANXICO_TOKEN", ""),
                years=years,
            )
        except Exception as exc:
            logger.debug("Banxico primary failed: %s", exc)

    elif primary == "ons":
        try:
            from operator1.clients.macro_ons import fetch_macro_ons
            results = fetch_macro_ons(years=years)
        except Exception as exc:
            logger.debug("ONS primary failed: %s", exc)

    elif primary == "estat":
        try:
            from operator1.clients.macro_estat import fetch_macro_estat
            results = fetch_macro_estat(years=years)
        except Exception as exc:
            logger.debug("e-Stat primary failed: %s", exc)

    elif primary == "kosis":
        try:
            from operator1.clients.macro_kosis import fetch_macro_kosis
            results = fetch_macro_kosis(years=years)
        except Exception as exc:
            logger.debug("KOSIS primary failed: %s", exc)

    elif primary == "dgbas":
        try:
            from operator1.clients.macro_dgbas import fetch_macro_dgbas
            results = fetch_macro_dgbas(years=years)
        except Exception as exc:
            logger.debug("DGBAS primary failed: %s", exc)

    elif primary == "bcch":
        try:
            from operator1.clients.macro_bcch import fetch_macro_bcch
            results = fetch_macro_bcch(years=years)
        except Exception as exc:
            logger.debug("BCCh primary failed: %s", exc)

    # Fallback to wbgapi if primary returned nothing or insufficient data
    if len(results) < 3:  # Need at least GDP, inflation, unemployment
        if primary and results:
            logger.info(
                "Per-region macro (%s) returned only %d indicators; "
                "supplementing with wbgapi",
                primary, len(results),
            )
        elif primary:
            logger.info(
                "Per-region macro (%s) returned empty; falling back to wbgapi",
                primary,
            )

        try:
            from operator1.clients.macro_wbgapi import fetch_macro_wbgapi
            wb_results = fetch_macro_wbgapi(cc, years=years)

            # Merge: only add indicators not already fetched from primary
            for key, series in wb_results.items():
                if key not in results:
                    results[key] = series

        except Exception as exc:
            logger.warning("wbgapi fallback also failed for %s: %s", cc, exc)

    # Secondary fallback: IMF SDMX API (free, no key) for countries
    # not well covered by wbgapi (e.g. Taiwan)
    if len(results) < 3:
        try:
            from operator1.clients.macro_dgbas import _fetch_imf_series, _IMF_TW_SERIES
            # IMF uses ISO-2 codes; build URLs dynamically for any country
            _IMF_BASE = "http://dataservices.imf.org/REST/SDMX_JSON.svc"
            imf_series = {
                "gdp_growth": f"{_IMF_BASE}/CompactData/IFS/A.{cc}.NGDP_R_PC_PP_PT",
                "unemployment_rate": f"{_IMF_BASE}/CompactData/IFS/A.{cc}.LUR_PT",
            }
            for key, url in imf_series.items():
                if key not in results:
                    s = _fetch_imf_series(url, key, years)
                    if s is not None:
                        results[key] = s
            if results:
                logger.info("IMF supplemented %s with %d indicators", cc, len(results))
        except Exception as exc:
            logger.debug("IMF fallback failed for %s: %s", cc, exc)

    if not results:
        logger.warning("No macro data available for %s (market: %s)", cc, market_id)

    return results
