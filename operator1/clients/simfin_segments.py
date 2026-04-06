"""SimFin product segment data client.

Fetches quarterly product/geographic revenue segment breakdowns from
SimFin's free API.  Falls back gracefully when SimFin is unavailable
or the company has no segment data.

Usage::

    from operator1.clients.simfin_segments import fetch_simfin_segments
    segments = fetch_simfin_segments("AAPL")
    # {'iPhone': pd.Series, 'Services': pd.Series, ...}
"""

from __future__ import annotations

import logging
from typing import Any

import pandas as pd
import requests

logger = logging.getLogger(__name__)

SIMFIN_BASE = "https://backend.simfin.com/api/v3"

_cache: dict[str, dict[str, pd.Series]] = {}


def fetch_simfin_segments(
    ticker: str,
    api_key: str = "free",
    timeout: int = 15,
) -> dict[str, pd.Series]:
    """Fetch product segment revenue from SimFin.

    Parameters
    ----------
    ticker:
        Stock ticker (e.g. "AAPL", "MSFT").
    api_key:
        SimFin API key.  "free" for the free tier (2K calls/day).
    timeout:
        HTTP timeout in seconds.

    Returns
    -------
    Dict mapping segment name to a pd.Series of quarterly revenue
    values indexed by period-end date.  Empty dict if unavailable.
    """
    if ticker in _cache:
        return _cache[ticker]

    headers = {"Authorization": f"api-key {api_key}"}
    empty: dict[str, pd.Series] = {}

    try:
        # Step 1: Resolve ticker to SimFin company ID
        resp = requests.get(
            f"{SIMFIN_BASE}/companies/list",
            params={"ticker": ticker},
            headers=headers,
            timeout=timeout,
        )
        if resp.status_code != 200:
            logger.debug("SimFin company list failed: HTTP %d", resp.status_code)
            _cache[ticker] = empty
            return empty

        companies = resp.json()
        if not companies:
            logger.debug("SimFin: no company found for ticker %s", ticker)
            _cache[ticker] = empty
            return empty

        company_id = companies[0].get("id")
        if not company_id:
            _cache[ticker] = empty
            return empty

        # Step 2: Fetch segment data
        resp = requests.get(
            f"{SIMFIN_BASE}/companies/id/{company_id}/statements/derived",
            params={"statement": "segments", "period": "quarters"},
            headers=headers,
            timeout=timeout,
        )
        if resp.status_code != 200:
            logger.debug("SimFin segments failed: HTTP %d", resp.status_code)
            _cache[ticker] = empty
            return empty

        data = resp.json()
        rows = data if isinstance(data, list) else data.get("data", [])

        segments: dict[str, dict[pd.Timestamp, float]] = {}
        for row in rows:
            if not isinstance(row, dict):
                continue
            name = row.get("segment_name") or row.get("name") or "Unknown"
            period_end = row.get("period_end") or row.get("date")
            revenue = row.get("revenue") or row.get("value")

            if period_end and revenue is not None:
                try:
                    ts = pd.Timestamp(period_end)
                    val = float(revenue)
                    segments.setdefault(name, {})[ts] = val
                except (ValueError, TypeError):
                    continue

        result = {
            name: pd.Series(values).sort_index()
            for name, values in segments.items()
            if len(values) >= 2
        }

        if result:
            logger.info(
                "SimFin segments for %s: %d segments (%s)",
                ticker, len(result), ", ".join(result.keys()),
            )
        else:
            logger.debug("SimFin: no segment data for %s", ticker)

        _cache[ticker] = result
        return result

    except requests.RequestException as exc:
        logger.debug("SimFin request failed for %s: %s", ticker, exc)
        _cache[ticker] = empty
        return empty
    except Exception as exc:
        logger.debug("SimFin parsing failed for %s: %s", ticker, exc)
        _cache[ticker] = empty
        return empty
