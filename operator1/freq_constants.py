"""Frequency-aware constants for the multi-frequency pipeline.

Provides a central lookup for annualization factors, rolling window
sizes, and model compatibility that adapts to the current pipeline
frequency.  All models replace hardcoded ``252`` with calls to
:func:`get_periods_per_year` or :func:`get_vol_annualization`.

The current frequency is stored in a :class:`threading.local` so each
thread in the parallel frequency pipeline gets its own value.

Usage::

    from operator1.freq_constants import (
        get_periods_per_year,
        get_vol_annualization,
        get_rolling_quarter,
        should_skip,
    )

    ppy = get_periods_per_year()          # 252 at D, 4 at Q, 1 at A
    vol_ann = get_vol_annualization()     # sqrt(252) at D, 2.0 at Q
    rpq = get_rolling_quarter()           # 63 at D, 1 at Q
"""

from __future__ import annotations

import math
import threading

# ---------------------------------------------------------------------------
# Thread-local frequency context
# ---------------------------------------------------------------------------

_freq_context = threading.local()


def set_freq(freq: str) -> None:
    """Set the current pipeline frequency for this thread."""
    _freq_context.freq = freq.upper() if freq else "D"


def get_freq() -> str:
    """Return the current pipeline frequency (default ``"D"``)."""
    return getattr(_freq_context, "freq", "D")


# ---------------------------------------------------------------------------
# Frequency-indexed constant maps
# ---------------------------------------------------------------------------

PERIODS_PER_YEAR: dict[str, int] = {
    "D": 252,
    "W": 52,
    "M": 12,
    "Q": 4,
    "S": 2,
    "A": 1,
}

VOL_ANNUALIZATION: dict[str, float] = {
    "D": math.sqrt(252),
    "W": math.sqrt(52),
    "M": math.sqrt(12),
    "Q": 2.0,
    "S": math.sqrt(2),
    "A": 1.0,
}

PERIODS_PER_QUARTER: dict[str, int] = {
    "D": 63,
    "W": 13,
    "M": 3,
    "Q": 1,
    "S": 1,
    "A": 1,
}

PERIODS_PER_MONTH: dict[str, int] = {
    "D": 21,
    "W": 4,
    "M": 1,
    "Q": 1,
    "S": 1,
    "A": 1,
}

# Horizon label -> periods mapping per frequency
HORIZON_PERIODS: dict[str, dict[str, int]] = {
    "D": {"1d": 1, "5d": 5, "21d": 21, "252d": 252},
    "W": {"1p": 1, "short": 4, "medium": 13, "long": 52},
    "M": {"1p": 1, "short": 3, "medium": 6, "long": 12},
    "Q": {"1p": 1, "short": 2, "medium": 4, "long": 8},
    "S": {"1p": 1, "short": 2, "medium": 4, "long": 4},
    "A": {"1p": 1, "short": 2, "medium": 3, "long": 5},
}


# ---------------------------------------------------------------------------
# Model skip rules (models that produce nonsensical results at certain freqs)
# ---------------------------------------------------------------------------

SKIP_AT_FREQ: dict[str, set[str]] = {
    "pattern_detector": {"Q", "S", "A", "M"},
    "ohlc_predictor": {"Q", "S", "A", "M", "W"},
    "options_signals": {"Q", "S", "A", "M", "W"},
    "cross_asset_signals": {"Q", "S", "A", "M"},
    "behavioral_signals": {"Q", "S", "A", "M"},
    "news_sentiment": {"Q", "S", "A", "M"},
    "institutional_flow": {"Q", "S", "A", "M"},
    "accruals_forensics": {"D", "W", "M"},
    "earnings_smoothing": {"D", "W", "M"},
    "fcf_quality": {"D", "W", "M"},
}


# ---------------------------------------------------------------------------
# Accessor functions (use these instead of raw dict lookups)
# ---------------------------------------------------------------------------

def get_periods_per_year(freq: str | None = None) -> int:
    """Trading periods per year for the given or current frequency."""
    return PERIODS_PER_YEAR.get(freq or get_freq(), 252)


def get_vol_annualization(freq: str | None = None) -> float:
    """Volatility annualization factor: ``sqrt(periods_per_year)``."""
    return VOL_ANNUALIZATION.get(freq or get_freq(), math.sqrt(252))


def get_rolling_quarter(freq: str | None = None) -> int:
    """Number of periods in one quarter at the given frequency."""
    return PERIODS_PER_QUARTER.get(freq or get_freq(), 63)


def get_rolling_month(freq: str | None = None) -> int:
    """Number of periods in one month at the given frequency."""
    return PERIODS_PER_MONTH.get(freq or get_freq(), 21)


def get_rolling_year(freq: str | None = None) -> int:
    """Alias for :func:`get_periods_per_year`."""
    return get_periods_per_year(freq)


def get_hurst_vol_annualization(
    hurst: float = 0.5,
    freq: str | None = None,
) -> float:
    """Hurst-corrected volatility annualization factor.

    Standard ``sqrt(T)`` scaling assumes i.i.d. returns (H=0.5).
    Financial returns often exhibit persistence (H>0.5) or mean-reversion
    (H<0.5), making ``T^H`` more accurate than ``T^0.5``.

    Parameters
    ----------
    hurst:
        Hurst exponent (0-1).  0.5 = random walk (reduces to sqrt(T)).
        >0.5 = trending (vol scales faster). <0.5 = mean-reverting.
    freq:
        Frequency to compute for (default: current thread-local).

    Returns
    -------
    Annualization factor ``periods_per_year ** hurst``.
    """
    ppy = get_periods_per_year(freq)
    # Clamp H to [0.1, 0.9] for numerical safety
    h = max(0.1, min(0.9, hurst))
    return ppy ** h


def get_horizons(freq: str | None = None) -> dict[str, int]:
    """Return the horizon label -> period count map for the frequency."""
    return HORIZON_PERIODS.get(freq or get_freq(), HORIZON_PERIODS["D"])


def should_skip(model_name: str, freq: str | None = None) -> bool:
    """Check if a model should be skipped at the current frequency."""
    f = freq or get_freq()
    return f in SKIP_AT_FREQ.get(model_name, set())
