"""Global macro provider using wbgapi (World Bank).

Serves as both the primary provider for markets without region-specific
macro APIs and the fallback for all markets.

No API key required. World Bank open data.
Research: .roo/research/macro-wbgapi-2026-02-24.md
"""

from __future__ import annotations

import logging

import pandas as pd

logger = logging.getLogger(__name__)

# ISO-2 country code -> World Bank ISO-3 code
_COUNTRY_TO_WB: dict[str, str] = {
    "US": "USA",
    "GB": "GBR",
    "EU": "EMU",
    "FR": "FRA",
    "DE": "DEU",
    "JP": "JPN",
    "KR": "KOR",
    # Taiwan is not a World Bank member; no WB code available.
    # macro_provider will fall through to wbgapi with no match,
    # returning empty and letting the caller handle it.
    "BR": "BRA",
    "CL": "CHL",
    "CA": "CAN",
    "AU": "AUS",
    "IN": "IND",
    "CN": "CHN",
    "HK": "HKG",
    "SG": "SGP",
    "MX": "MEX",
    "ZA": "ZAF",
    "CH": "CHE",
    "SA": "SAU",
    "AE": "ARE",
}

# World Bank indicator IDs for canonical macro variables
_WB_INDICATORS: dict[str, str] = {
    "gdp_growth": "NY.GDP.MKTP.KD.ZG",
    "inflation_rate_yoy": "FP.CPI.TOTL.ZG",
    "unemployment_rate": "SL.UEM.TOTL.ZS",
    "interest_rate_real": "FR.INR.RINR",
    "exchange_rate": "PA.NUS.FCRF",
}


def fetch_macro_wbgapi(
    country_iso2: str,
    years: int = 10,
) -> dict[str, pd.Series]:
    """Fetch macro indicators from World Bank via wbgapi.

    Parameters
    ----------
    country_iso2:
        ISO-2 country code (e.g. "US", "JP", "BR").
    years:
        Number of most recent years to fetch.

    Returns
    -------
    Dict mapping canonical indicator name -> pd.Series with DatetimeIndex.
    """
    try:
        import wbgapi as wb
    except ImportError:
        logger.warning("wbgapi not installed. pip install wbgapi>=1.0.13")
        return {}

    wb_code = _COUNTRY_TO_WB.get(country_iso2.upper(), country_iso2.upper())
    results: dict[str, pd.Series] = {}

    for canonical_name, indicator_id in _WB_INDICATORS.items():
        try:
            # Verbatim from wbgapi README
            df = wb.data.DataFrame(
                indicator_id,
                economy=wb_code,
                mrv=years,
                numericTimeKeys=True,
            )

            if df is None or df.empty:
                continue

            # wbgapi returns DataFrame with years as columns, economy as rows
            # Transpose to get years as index
            if wb_code in df.index:
                row = df.loc[wb_code]
            elif len(df) == 1:
                row = df.iloc[0]
            else:
                continue

            # Convert to Series with year index
            series = row.dropna()
            if series.empty:
                continue

            # Convert year integers to datetime for downstream compatibility
            series.index = pd.to_datetime(series.index.astype(str), format="%Y")
            series.name = canonical_name
            results[canonical_name] = series

            logger.debug(
                "wbgapi %s/%s: %d observations",
                wb_code, canonical_name, len(series),
            )

        except Exception as exc:
            logger.debug("wbgapi failed for %s/%s: %s", wb_code, canonical_name, exc)

    logger.info(
        "wbgapi fetched %d/%d indicators for %s",
        len(results), len(_WB_INDICATORS), wb_code,
    )
    return results
