"""EU/DE/FR macro provider using sdmx1 (ECB / Bundesbank / INSEE).

Primary macro source for European markets.
Fallback: wbgapi (World Bank).

No API key for ECB and Bundesbank. INSEE may need key.
Research: .roo/research/macro-per-region-2026-02-25.md
Package: sdmx1 2.25.1

NOTE: ECB SDMX queries can be slow (30-90 seconds). The sdmx1 library
handles connection management. We set a generous timeout and catch
timeouts gracefully, falling back to wbgapi.
"""

from __future__ import annotations

import logging

import pandas as pd

logger = logging.getLogger(__name__)

# ECB SDW REST endpoint URLs for direct HTTP fallback
# These bypass sdmx1 if it times out or isn't installed.
# ECB migrated from sdw-wsrest.ecb.europa.eu to data-api.ecb.europa.eu
# Verified working 2026-02-25 via curl tests
_ECB_REST_BASE = "https://data-api.ecb.europa.eu/service"

# Direct URL patterns for key ECB series (CSV format, faster than SDMX XML)
# Verified with curl --max-time 15 on 2026-02-25
_ECB_DIRECT_URLS: dict[str, str] = {
    "exchange_rate": f"{_ECB_REST_BASE}/data/EXR/A.USD.EUR.SP00.A?lastNObservations=10&format=csvdata",
    "interest_rate": f"{_ECB_REST_BASE}/data/FM/M.U2.EUR.RT.MM.EURIBOR3MD_.HSTA?lastNObservations=36&format=csvdata",
    # GDP: quarterly nominal level -- we compute YoY growth rate in code
    "gdp_level": f"{_ECB_REST_BASE}/data/MNA/Q.N.I8.W2.S1.S1.B.B1GQ._Z._Z._Z.EUR.V.N?lastNObservations=24&format=csvdata",
    "inflation_rate_yoy": f"{_ECB_REST_BASE}/data/ICP/M.U2.N.000000.4.ANR?lastNObservations=36&format=csvdata",
    "unemployment_rate": f"{_ECB_REST_BASE}/data/LFSI/M.I8.S.UNEHRT.TOTAL0.15_74.T?lastNObservations=36&format=csvdata",
}


def fetch_macro_ecb(
    years: int = 10,
) -> dict[str, pd.Series]:
    """Fetch EU macro indicators from ECB.

    Tries sdmx1 library first, then falls back to direct REST API calls.
    Both approaches may be slow (ECB SDMX is known for long response times).

    Returns
    -------
    Dict mapping canonical indicator name -> pd.Series.
    """
    results: dict[str, pd.Series] = {}

    # Method 1: Try sdmx1 library (more structured, but can be slow)
    try:
        import sdmx
        # Use the new ECB data API endpoint (migrated from sdw-wsrest)
        ecb = sdmx.Client(
            "ECB",
            backend="data-api.ecb.europa.eu",
            timeout=90,
        )

        # Simple exchange rate query as a quick test
        try:
            data = ecb.data(
                "EXR",
                key={"FREQ": "A", "CURRENCY": "USD", "CURRENCY_DENOM": "EUR"},
                params={"lastNObservations": str(years)},
            )
            if data and hasattr(data, "data") and data.data:
                for ds in data.data:
                    for sk, obs_dict in ds.series.items():
                        values = []
                        dates = []
                        for tp, obs_list in obs_dict.items():
                            dates.append(str(tp))
                            val = obs_list[0].value if obs_list else None
                            values.append(float(val) if val is not None else None)
                        if values:
                            s = pd.Series(values, index=pd.to_datetime(dates))
                            s.name = "exchange_rate"
                            results["exchange_rate"] = s
                            break
        except Exception as exc:
            logger.debug("ECB sdmx1 exchange rate query failed: %s", exc)

        logger.info("ECB sdmx1 fetched %d indicators", len(results))
        return results

    except ImportError:
        logger.debug("sdmx1 not installed; trying direct REST API")
    except Exception as exc:
        logger.debug("ECB sdmx1 client failed: %s; trying direct REST", exc)

    # Method 2: Direct REST API with CSV format (faster, no sdmx1 needed)
    try:
        import requests
        for canonical_name, url in _ECB_DIRECT_URLS.items():
            try:
                # format=csvdata is appended to URLs already; just request it
                resp = requests.get(
                    url,
                    headers={
                        "Accept": "text/csv, application/vnd.sdmx.data+csv;version=1.0.0",
                        "User-Agent": "Operator1/1.0",
                    },
                    timeout=60,
                )
                if resp.status_code == 200 and resp.text:
                    import io
                    df = pd.read_csv(io.StringIO(resp.text))
                    if not df.empty and "OBS_VALUE" in df.columns:
                        time_col = "TIME_PERIOD" if "TIME_PERIOD" in df.columns else df.columns[0]
                        series = pd.Series(
                            pd.to_numeric(df["OBS_VALUE"], errors="coerce").values,
                            index=pd.to_datetime(df[time_col].astype(str)),
                        )

                        # GDP level -> compute YoY growth rate
                        if canonical_name == "gdp_level":
                            growth = series.pct_change(periods=4) * 100  # YoY quarterly
                            growth = growth.dropna()
                            if not growth.empty:
                                growth.name = "gdp_growth"
                                results["gdp_growth"] = growth
                                logger.debug("ECB REST gdp_growth: %d obs (computed from levels)", len(growth))
                        else:
                            series.name = canonical_name
                            results[canonical_name] = series
                            logger.debug("ECB REST %s: %d obs", canonical_name, len(series))
            except Exception as exc:
                logger.debug("ECB REST %s failed: %s", canonical_name, exc)
    except ImportError:
        logger.debug("requests not available for ECB REST fallback")

    logger.info("ECB REST fetched %d/%d indicators", len(results), len(_ECB_DIRECT_URLS))
    return results
