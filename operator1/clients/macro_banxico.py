"""Mexico macro provider using banxicoapi (Banxico).

Primary macro source for Mexican market.
Fallback: wbgapi (World Bank).

Requires BANXICO_TOKEN (free registration).
Research: .roo/research/macro-per-region-2026-02-25.md
Package: banxicoapi 1.0.2
"""

from __future__ import annotations

import logging
import os

import pandas as pd

logger = logging.getLogger(__name__)

# Banxico series IDs for canonical macro variables
_BANXICO_SERIES: dict[str, str] = {
    "interest_rate": "SF61745",      # Target rate
    "inflation_rate_yoy": "SP74665", # CPI annual change
    "exchange_rate": "SF63528",      # USD/MXN exchange rate
}


def fetch_macro_banxico(
    api_token: str = "",
    years: int = 10,
) -> dict[str, pd.Series]:
    """Fetch Mexican macro indicators from Banxico.

    Parameters
    ----------
    api_token:
        Banxico API token. Falls back to BANXICO_TOKEN env var.
    years:
        Number of years of history.

    Returns
    -------
    Dict mapping canonical indicator name -> pd.Series.
    """
    token = api_token or os.environ.get("BANXICO_TOKEN", "")
    if not token:
        logger.debug("BANXICO_TOKEN not set; falling back to wbgapi")
        return {}

    try:
        import banxicoapi
    except ImportError:
        logger.debug("banxicoapi not installed; falling back to wbgapi")
        return {}

    try:
        api = banxicoapi.BanxicoApi(token)
    except Exception as exc:
        logger.warning("banxicoapi init failed: %s", exc)
        return {}

    from datetime import date, timedelta
    start = (date.today() - timedelta(days=365 * years)).strftime("%Y-%m-%d")
    end = date.today().strftime("%Y-%m-%d")
    results: dict[str, pd.Series] = {}

    for canonical_name, series_id in _BANXICO_SERIES.items():
        try:
            # banxicoapi 1.0.2: get() takes a list of series IDs and date range
            raw = api.get([series_id], start_date=start, end_date=end)
            if not raw or not isinstance(raw, list):
                continue

            # Response format: [{"idSerie": "...", "datos": [{"fecha": "dd/mm/yyyy", "dato": "value"}, ...]}]
            datos = raw[0].get("datos", []) if raw else []
            if not datos:
                continue

            dates = []
            values = []
            for d in datos:
                try:
                    dates.append(pd.to_datetime(d["fecha"], dayfirst=True))
                    values.append(float(d["dato"].replace(",", "")))
                except (ValueError, KeyError):
                    continue

            if dates:
                series = pd.Series(values, index=pd.DatetimeIndex(dates), name=canonical_name)
                series = series.sort_index()
                results[canonical_name] = series
                logger.debug("Banxico %s: %d observations", series_id, len(series))
        except Exception as exc:
            logger.debug("Banxico failed for %s: %s", series_id, exc)

    logger.info("Banxico fetched %d/%d indicators", len(results), len(_BANXICO_SERIES))
    return results
