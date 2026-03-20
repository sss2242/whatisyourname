"""Filing calendar awareness -- expected filing dates and frequency detection.

Tracks when financial statements should be available for a company based
on its market and filing frequency (quarterly, semiannual, annual).

This module addresses the gap where the pipeline didn't know how many
filings it should have received in a 2-year window.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from datetime import date, timedelta
from typing import Any

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


# Filing frequency per market (days between filings)
# Markets that require quarterly reports have ~90-day cycles.
# Markets with semiannual have ~180-day cycles.
_MARKET_FILING_FREQUENCY: dict[str, dict[str, Any]] = {
    # Quarterly reporters
    "us_sec_edgar": {"frequency": "quarterly", "days": 90, "deadline_days": 40, "expected_per_year": 4},
    "ca_sedar": {"frequency": "quarterly", "days": 90, "deadline_days": 45, "expected_per_year": 4},
    "br_cvm": {"frequency": "quarterly", "days": 90, "deadline_days": 45, "expected_per_year": 4},
    "kr_dart": {"frequency": "quarterly", "days": 90, "deadline_days": 45, "expected_per_year": 4},
    "in_bse": {"frequency": "quarterly", "days": 90, "deadline_days": 45, "expected_per_year": 4},
    "cn_sse": {"frequency": "quarterly", "days": 90, "deadline_days": 30, "expected_per_year": 4},
    "hk_hkex": {"frequency": "semiannual", "days": 180, "deadline_days": 60, "expected_per_year": 2},
    "sg_sgx": {"frequency": "semiannual", "days": 180, "deadline_days": 60, "expected_per_year": 2},
    "mx_bmv": {"frequency": "quarterly", "days": 90, "deadline_days": 40, "expected_per_year": 4},
    # Semiannual reporters (typical for European / ESEF markets)
    "uk_companies_house": {"frequency": "annual", "days": 365, "deadline_days": 270, "expected_per_year": 1},
    "eu_esef": {"frequency": "annual", "days": 365, "deadline_days": 120, "expected_per_year": 1},
    "fr_esef": {"frequency": "annual", "days": 365, "deadline_days": 120, "expected_per_year": 1},
    "de_esef": {"frequency": "annual", "days": 365, "deadline_days": 120, "expected_per_year": 1},
    "nl_esef": {"frequency": "annual", "days": 365, "deadline_days": 120, "expected_per_year": 1},
    "es_esef": {"frequency": "annual", "days": 365, "deadline_days": 120, "expected_per_year": 1},
    "it_esef": {"frequency": "annual", "days": 365, "deadline_days": 120, "expected_per_year": 1},
    "se_esef": {"frequency": "annual", "days": 365, "deadline_days": 120, "expected_per_year": 1},
    "ch_six": {"frequency": "annual", "days": 365, "deadline_days": 120, "expected_per_year": 1},
    # Mixed (quarterly interim + annual)
    "jp_jquants": {"frequency": "quarterly", "days": 90, "deadline_days": 45, "expected_per_year": 4},
    "tw_mops": {"frequency": "quarterly", "days": 90, "deadline_days": 45, "expected_per_year": 4},
    "cl_cmf": {"frequency": "quarterly", "days": 90, "deadline_days": 60, "expected_per_year": 4},
    # Annual only
    "au_asx": {"frequency": "semiannual", "days": 180, "deadline_days": 75, "expected_per_year": 2},
    "za_jse": {"frequency": "semiannual", "days": 180, "deadline_days": 90, "expected_per_year": 2},
    "sa_tadawul": {"frequency": "quarterly", "days": 90, "deadline_days": 45, "expected_per_year": 4},
    "ae_dfm": {"frequency": "quarterly", "days": 90, "deadline_days": 45, "expected_per_year": 4},
}


@dataclass
class FilingCalendarResult:
    """Summary of filing calendar analysis."""

    market_id: str = ""
    expected_frequency: str = "unknown"
    detected_frequency: str = "unknown"
    expected_filings_2yr: int = 0
    actual_filings_2yr: int = 0
    coverage_ratio: float = 0.0
    filing_dates: list[str] = field(default_factory=list)
    gaps: list[dict[str, str]] = field(default_factory=list)
    latest_filing_age_days: int = -1
    is_stale: bool = False
    stale_threshold_days: int = 0


def detect_filing_frequency(
    filing_dates: list[pd.Timestamp],
) -> str:
    """Detect filing frequency from a list of filing/report dates.

    Returns one of: "quarterly", "semiannual", "annual", "unknown".
    """
    if len(filing_dates) < 2:
        return "unknown"

    dates = sorted(filing_dates)
    gaps = [(dates[i + 1] - dates[i]).days for i in range(len(dates) - 1)]
    median_gap = float(np.median(gaps))

    if median_gap < 120:
        return "quarterly"
    elif median_gap < 250:
        return "semiannual"
    elif median_gap < 500:
        return "annual"
    return "unknown"


def analyze_filing_calendar(
    cache: pd.DataFrame,
    market_id: str = "",
    today: date | None = None,
) -> FilingCalendarResult:
    """Analyze the filing calendar for a company.

    Parameters
    ----------
    cache:
        Daily cache DataFrame with financial statement columns.
    market_id:
        PIT market identifier for expected frequency lookup.
    today:
        Override today's date (for testing).

    Returns
    -------
    FilingCalendarResult with expected vs actual filings, gaps, and staleness.
    """
    if today is None:
        today = date.today()

    result = FilingCalendarResult(market_id=market_id)

    # Get market filing config
    config = _MARKET_FILING_FREQUENCY.get(market_id, {})
    result.expected_frequency = config.get("frequency", "unknown")
    expected_per_year = config.get("expected_per_year", 4)
    result.expected_filings_2yr = expected_per_year * 2
    deadline_days = config.get("deadline_days", 90)
    result.stale_threshold_days = deadline_days + config.get("days", 90)

    # Find actual filing dates from the cache
    # Financial statement columns change value when a new filing arrives.
    # We detect filing dates by finding rows where key columns change.
    filing_dates = _detect_filing_dates_from_cache(cache)
    result.actual_filings_2yr = len(filing_dates)
    result.filing_dates = [str(d.date()) for d in filing_dates]

    if result.expected_filings_2yr > 0:
        result.coverage_ratio = result.actual_filings_2yr / result.expected_filings_2yr
    else:
        result.coverage_ratio = 1.0 if result.actual_filings_2yr > 0 else 0.0

    # Detect frequency from actual data
    result.detected_frequency = detect_filing_frequency(filing_dates)

    # Check staleness -- prefer filing_date column if available in cache,
    # otherwise fall back to detected value-change dates (which may correspond
    # to report_date rather than actual filing publication date).
    _latest_filing_ts = None
    if "filing_date" in cache.columns and cache["filing_date"].notna().any():
        _latest_filing_ts = pd.to_datetime(cache["filing_date"].dropna()).max()
    elif filing_dates:
        _latest_filing_ts = filing_dates[-1]

    if _latest_filing_ts is not None:
        result.latest_filing_age_days = (pd.Timestamp(today) - _latest_filing_ts).days
        result.is_stale = result.latest_filing_age_days > result.stale_threshold_days
    else:
        result.latest_filing_age_days = -1
        result.is_stale = True  # No filings at all = definitely stale

    # Find gaps (periods where expected filings were missing)
    if len(filing_dates) >= 2:
        expected_gap = config.get("days", 90)
        for i in range(len(filing_dates) - 1):
            gap_days = (filing_dates[i + 1] - filing_dates[i]).days
            if gap_days > expected_gap * 1.5:  # 50% tolerance
                result.gaps.append({
                    "from": str(filing_dates[i].date()),
                    "to": str(filing_dates[i + 1].date()),
                    "gap_days": str(gap_days),
                    "expected_days": str(expected_gap),
                })

    return result


def inject_filing_freshness(
    cache: pd.DataFrame,
    calendar_result: FilingCalendarResult,
    market_id: str = "",
) -> pd.DataFrame:
    """Inject filing freshness columns into the daily cache.

    Computes a continuous 0-1 freshness score for each day based on how
    recently a new filing was published.  Days right after a filing get
    freshness=1.0; freshness decays linearly toward 0.0 as the expected
    next filing date approaches and passes.

    New columns added:
      - ``filing_freshness``: 0.0-1.0 continuous score
      - ``filing_gap_flag``: 1 if beyond expected filing window
      - ``days_since_last_filing``: integer days since most recent filing

    Parameters
    ----------
    cache:
        Daily cache DataFrame (modified in place and returned).
    calendar_result:
        Result from ``analyze_filing_calendar()``.
    market_id:
        Market ID for filing frequency lookup.

    Returns
    -------
    cache with freshness columns added.
    """
    config = _MARKET_FILING_FREQUENCY.get(market_id, {})
    expected_gap_days = config.get("days", 90)
    # Grace period: data is still considered fully fresh for this many days
    # after expected gap; then decays to 0 over the same period.
    grace_days = expected_gap_days
    decay_days = expected_gap_days  # linear decay over one full cycle

    # Parse filing dates
    filing_dates_ts = []
    for d_str in calendar_result.filing_dates:
        try:
            filing_dates_ts.append(pd.Timestamp(d_str))
        except (ValueError, TypeError):
            continue

    if not filing_dates_ts:
        # No filings detected -- everything is stale
        cache["filing_freshness"] = 0.0
        cache["filing_gap_flag"] = 1
        cache["days_since_last_filing"] = -1
        logger.warning("No filing dates detected; all data marked as stale.")
        return cache

    filing_dates_ts = sorted(filing_dates_ts)

    # Compute per-day metrics
    freshness = pd.Series(0.0, index=cache.index, dtype=float)
    days_since = pd.Series(-1, index=cache.index, dtype=int)
    gap_flag = pd.Series(0, index=cache.index, dtype=int)

    for idx_date in cache.index:
        ts = pd.Timestamp(idx_date)

        # Find the most recent filing on or before this date
        recent_filings = [f for f in filing_dates_ts if f <= ts]
        if not recent_filings:
            # Before any filing in window
            freshness[idx_date] = 0.0
            gap_flag[idx_date] = 1
            days_since[idx_date] = -1
            continue

        last_filing = recent_filings[-1]
        days_elapsed = (ts - last_filing).days
        days_since[idx_date] = days_elapsed

        if days_elapsed <= grace_days:
            # Within expected window: fully fresh
            freshness[idx_date] = 1.0
            gap_flag[idx_date] = 0
        elif days_elapsed <= grace_days + decay_days:
            # Decay period: linear decay from 1.0 to 0.0
            decay_progress = (days_elapsed - grace_days) / decay_days
            freshness[idx_date] = max(1.0 - decay_progress, 0.0)
            gap_flag[idx_date] = 1
        else:
            # Severely stale
            freshness[idx_date] = 0.0
            gap_flag[idx_date] = 1

    cache["filing_freshness"] = freshness
    cache["filing_gap_flag"] = gap_flag
    cache["days_since_last_filing"] = days_since

    fresh_pct = (freshness > 0.5).mean() * 100
    gap_pct = gap_flag.mean() * 100
    logger.info(
        "Filing freshness: %.0f%% of days are fresh (>0.5), %.0f%% have gap flags",
        fresh_pct, gap_pct,
    )

    return cache


def _detect_filing_dates_from_cache(cache: pd.DataFrame) -> list[pd.Timestamp]:
    """Detect filing dates by finding rows where key financial columns change value.

    Financial statements are forward-filled onto daily rows. A new filing
    appears as a step change in columns like revenue, total_assets, etc.
    """
    key_cols = ["revenue", "total_assets", "net_income", "operating_cash_flow"]
    available = [c for c in key_cols if c in cache.columns and cache[c].notna().any()]

    if not available:
        return []

    # Use the first available column to detect changes
    col = available[0]
    series = cache[col].dropna()
    if series.empty:
        return []

    # Find rows where the value changes (new filing)
    changes = series.diff().abs() > 0
    change_dates = series.index[changes].tolist()

    # Add the first non-null date (first filing in the window)
    if len(series) > 0:
        first_date = series.index[0]
        if first_date not in change_dates:
            change_dates.insert(0, first_date)

    return sorted(change_dates)
