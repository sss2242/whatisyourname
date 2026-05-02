"""Event calendar features for uncertainty-adjusted predictions.

Tracks known upcoming events (FOMC meetings, earnings dates, political
events) and computes proximity features that adjust prediction confidence
and conformal interval width.  The January 2025 AAPL drop was driven by
known political events (inauguration, tariff announcements) -- an event
calendar would have widened prediction intervals preemptively.

Features computed:
  - days_to_next_event: minimum days until any known event (0-90)
  - event_uncertainty_premium: multiplier for conformal interval width (1.0-2.5)
  - fomc_proximity: days until next FOMC meeting (0-42)
  - earnings_proximity: days until next earnings (from filing_calendar)
  - event_density_30d: number of known events in the next 30 days (0-10)

Data sources:
  - config/event_calendar.json: FOMC dates + political events (static)
  - yfinance Ticker.calendar: next earnings date (dynamic)
  - filing_calendar_result: predicted next filing date (from pipeline)

References:
  - MacKinlay (1997): Event study methodology
  - Bloom (2009): Uncertainty channels (policy, monetary, idiosyncratic)
  - Rigobon & Sack (2004): Anticipation effect in options

Pipeline integration:
  - Called in main.py Step 4a.9 (after cross-asset signals)
  - event_uncertainty_premium consumed by conformal prediction
  - days_to_next_event consumed by prediction aggregator
  - Profile key: event_calendar_signals
"""

from __future__ import annotations

import json
import logging
import math
from dataclasses import dataclass
from datetime import date, timedelta
from pathlib import Path
from typing import Any

import pandas as pd

from operator1.scoring_weights import get_weight

logger = logging.getLogger(__name__)

_CONFIG_PATH = Path(__file__).resolve().parent.parent.parent / "config" / "event_calendar.json"


# ---------------------------------------------------------------------------
# Result container
# ---------------------------------------------------------------------------

@dataclass
class EventCalendarResult:
    """Container for event calendar signal results."""

    available: bool = False
    days_to_next_event: int | None = None
    event_uncertainty_premium: float | None = None
    fomc_proximity: int | None = None
    earnings_proximity: int | None = None
    event_density_30d: int | None = None
    next_event_type: str = ""
    next_event_date: str = ""
    n_fomc_loaded: int = 0
    n_political_loaded: int = 0
    error: str = ""

    def to_profile_dict(self) -> dict[str, Any]:
        """Convert to profile-ready dict."""
        return {
            "available": self.available,
            "days_to_next_event": self.days_to_next_event,
            "event_uncertainty_premium": _safe(self.event_uncertainty_premium),
            "fomc_proximity": self.fomc_proximity,
            "earnings_proximity": self.earnings_proximity,
            "event_density_30d": self.event_density_30d,
            "next_event_type": self.next_event_type,
            "next_event_date": self.next_event_date,
        }


def _safe(val: float | None) -> float | None:
    if val is None:
        return None
    try:
        f = float(val)
        return None if (math.isnan(f) or math.isinf(f)) else round(f, 4)
    except (TypeError, ValueError):
        return None


# ---------------------------------------------------------------------------
# Event loading
# ---------------------------------------------------------------------------

def _load_event_config() -> dict[str, Any]:
    """Load event calendar from config/event_calendar.json."""
    if not _CONFIG_PATH.exists():
        logger.debug("Event calendar config not found at %s", _CONFIG_PATH)
        return {}
    try:
        with open(_CONFIG_PATH, "r", encoding="utf-8") as f:
            return json.load(f)
    except Exception as exc:
        logger.debug("Failed to load event calendar: %s", exc)
        return {}


def _get_fomc_dates(config: dict) -> list[date]:
    """Extract all FOMC meeting dates from config."""
    dates: list[date] = []
    for key in sorted(config.keys()):
        if key.startswith("fomc_dates_"):
            for d_str in config[key]:
                try:
                    dates.append(date.fromisoformat(d_str))
                except (ValueError, TypeError):
                    continue
    return sorted(dates)


def _get_political_events(config: dict) -> list[dict[str, str]]:
    """Extract political events from config."""
    return config.get("political_events", [])


def _fetch_earnings_date(ticker: str) -> date | None:
    """Fetch next earnings date via yfinance Ticker.calendar."""
    try:
        import yfinance as yf
        t = yf.Ticker(ticker)
        cal = t.calendar
        if cal is not None:
            if isinstance(cal, dict):
                # yfinance returns dict with 'Earnings Date' key
                ed = cal.get("Earnings Date")
                if ed is not None:
                    if isinstance(ed, list) and ed:
                        return pd.Timestamp(ed[0]).date()
                    return pd.Timestamp(ed).date()
            elif isinstance(cal, pd.DataFrame) and not cal.empty:
                # Older yfinance format: DataFrame
                if "Earnings Date" in cal.columns:
                    val = cal["Earnings Date"].iloc[0]
                    return pd.Timestamp(val).date()
    except Exception as exc:
        logger.debug("Earnings date fetch failed for %s: %s", ticker, exc)
    return None


# ---------------------------------------------------------------------------
# Signal computations
# ---------------------------------------------------------------------------

def _compute_days_to_event(
    reference_date: date,
    all_event_dates: list[tuple[date, str]],
) -> tuple[int | None, str, str]:
    """Compute minimum days until any future event.

    Returns (days, event_type, event_date_str).
    """
    future_events = [
        (d, t) for d, t in all_event_dates
        if d >= reference_date
    ]
    if not future_events:
        return None, "", ""

    future_events.sort(key=lambda x: x[0])
    nearest_date, nearest_type = future_events[0]
    days = (nearest_date - reference_date).days
    return days, nearest_type, nearest_date.isoformat()


def _compute_uncertainty_premium(days_to_event: int | None) -> float:
    """Compute prediction interval widening factor based on event proximity.

    The closer we are to a known event, the wider the prediction intervals
    should be (higher uncertainty about the event's market impact).

    Returns a multiplier >= 1.0 for conformal interval width.
    """
    if days_to_event is None:
        return 1.0

    if days_to_event <= 0:
        return 2.5  # Event day: maximum uncertainty
    elif days_to_event <= 2:
        return 2.0  # 1-2 days before: high uncertainty
    elif days_to_event <= 5:
        return 1.5  # 3-5 days before: moderate uncertainty
    elif days_to_event <= 10:
        return 1.2  # 6-10 days before: slight uncertainty
    else:
        return 1.0  # >10 days: no adjustment


def _compute_event_density(
    reference_date: date,
    all_event_dates: list[tuple[date, str]],
    window_days: int = 30,
) -> int:
    """Count the number of known events in the next N days.

    High event density = information overload -> delayed price adjustment.
    """
    end_date = reference_date + timedelta(days=window_days)
    return sum(1 for d, _ in all_event_dates if reference_date <= d <= end_date)


def _compute_fomc_proximity(reference_date: date, fomc_dates: list[date]) -> int | None:
    """Compute days until the next FOMC meeting."""
    future = [d for d in fomc_dates if d >= reference_date]
    if not future:
        return None
    return (future[0] - reference_date).days


# ---------------------------------------------------------------------------
# Main entry point
# ---------------------------------------------------------------------------

def compute_event_calendar_features(
    cache: pd.DataFrame,
    ticker: str = "",
    filing_calendar_result: Any = None,
    reference_date: date | None = None,
) -> tuple[pd.DataFrame, EventCalendarResult]:
    """Compute event calendar features and inject into cache.

    Parameters
    ----------
    cache : daily cache DataFrame
    ticker : stock ticker for earnings date lookup
    filing_calendar_result : from filing_calendar.py (has next_expected_filing)
    reference_date : override for backtesting (default: today)

    Returns
    -------
    (cache, EventCalendarResult) -- cache with 5 new constant columns
    """
    result = EventCalendarResult()
    ref_date = reference_date or date.today()

    # Load static event config
    config = _load_event_config()
    fomc_dates = _get_fomc_dates(config)
    political_events = _get_political_events(config)

    result.n_fomc_loaded = len(fomc_dates)
    result.n_political_loaded = len(political_events)

    # Build unified event list: (date, type)
    all_events: list[tuple[date, str]] = []

    # FOMC meetings
    for d in fomc_dates:
        all_events.append((d, "fomc"))

    # Political events
    for evt in political_events:
        try:
            d = date.fromisoformat(evt.get("date", ""))
            all_events.append((d, evt.get("type", "political")))
        except (ValueError, TypeError):
            continue

    # Earnings date (dynamic from yfinance)
    earnings_date = _fetch_earnings_date(ticker) if ticker else None
    if earnings_date is not None:
        all_events.append((earnings_date, "earnings"))

    # Filing calendar predicted next filing
    if filing_calendar_result is not None:
        try:
            nef = getattr(filing_calendar_result, "next_expected_filing", None)
            if nef and isinstance(nef, dict) and nef.get("available"):
                pred_date = nef.get("predicted_date")
                if pred_date:
                    d = date.fromisoformat(str(pred_date)[:10])
                    all_events.append((d, "filing"))
        except Exception:
            pass

    # Sort all events
    all_events.sort(key=lambda x: x[0])

    # 1. Days to next event
    days, event_type, event_date = _compute_days_to_event(ref_date, all_events)
    result.days_to_next_event = days
    result.next_event_type = event_type
    result.next_event_date = event_date
    if days is not None:
        cache["days_to_next_event"] = days

    # 2. Uncertainty premium
    premium = _compute_uncertainty_premium(days)
    result.event_uncertainty_premium = premium
    cache["event_uncertainty_premium"] = premium

    # 3. FOMC proximity
    fomc_prox = _compute_fomc_proximity(ref_date, fomc_dates)
    result.fomc_proximity = fomc_prox
    if fomc_prox is not None:
        cache["fomc_proximity"] = fomc_prox

    # 4. Earnings proximity
    earnings_prox = None
    if earnings_date is not None:
        earnings_prox = max(0, (earnings_date - ref_date).days)
    result.earnings_proximity = earnings_prox
    if earnings_prox is not None:
        cache["earnings_proximity"] = earnings_prox

    # 5. Event density (next 30 days)
    density = _compute_event_density(ref_date, all_events)
    result.event_density_30d = density
    cache["event_density_30d"] = density

    result.available = True
    logger.info(
        "Event calendar: next=%s in %s days (%s), premium=%.1f, "
        "FOMC=%s days, earnings=%s days, density=%d events/30d",
        event_type or "none",
        days if days is not None else "N/A",
        event_date or "N/A",
        premium,
        fomc_prox if fomc_prox is not None else "N/A",
        earnings_prox if earnings_prox is not None else "N/A",
        density,
    )

    return cache, result
