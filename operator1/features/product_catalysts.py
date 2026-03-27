"""Product catalyst detection for forward-looking signals.

Financial models are backward-looking. This module detects signals that
a company is about to launch new products or shift its revenue mix,
creating step-changes in market expectations that historical patterns
cannot capture.

Signals:
  1. R&D Acceleration -- consecutive quarters of above-average R&D growth
  2. News Catalyst Score -- product launch/announcement mentions in news
  3. Revenue Concentration Shift -- new segments or share changes > 5pp
  4. Earnings Surprise Momentum -- consecutive beat/miss streak

Entry point:
    detect_product_catalysts(cache, profile, news_articles)

Output columns:
    catalyst_score         -- composite catalyst signal (0 to 1)
    rnd_acceleration       -- R&D spending acceleration (ratio vs 5yr avg)
    revenue_diversification_delta -- change in segment concentration
    catalyst_type          -- product_launch, rnd_surge, earnings_momentum, none
"""

from __future__ import annotations

import logging
import re
from dataclasses import dataclass, field
from typing import Any

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Catalyst keyword patterns
# ---------------------------------------------------------------------------

_LAUNCH_PATTERNS = [
    re.compile(r"\b(launch|launches|launched|launching)\b", re.I),
    re.compile(r"\b(new product|new device|new service|new platform)\b", re.I),
    re.compile(r"\b(release|released|releases|releasing)\b", re.I),
    re.compile(r"\b(announce|announced|announces|unveil|unveiled)\b", re.I),
    re.compile(r"\b(next[- ]gen|generation|upgrade|redesign)\b", re.I),
    re.compile(r"\b(partnership|collaboration|deal|acquisition)\b", re.I),
    re.compile(r"\b(AI|artificial intelligence|machine learning)\b", re.I),
    re.compile(r"\b(expansion|enters? market|new market)\b", re.I),
]

_NEGATIVE_PATTERNS = [
    re.compile(r"\b(recall|lawsuit|fine|penalty|investigation)\b", re.I),
    re.compile(r"\b(decline|drop|fall|loss|warning|downgrade)\b", re.I),
    re.compile(r"\b(layoff|restructur|cut|close|shut)\b", re.I),
]


@dataclass
class CatalystResult:
    """Result container for product catalyst detection."""
    available: bool = False
    catalyst_score: float = 0.0        # 0-1 composite
    rnd_acceleration: float = 0.0      # ratio vs average
    revenue_diversification_delta: float = 0.0
    earnings_momentum: float = 0.0     # -1 to +1
    news_catalyst_score: float = 0.0   # 0-1
    catalyst_type: str = "none"
    n_catalyst_articles: int = 0
    n_negative_articles: int = 0
    detail: str = ""
    error: str = ""


# ---------------------------------------------------------------------------
# R&D Acceleration
# ---------------------------------------------------------------------------

def _compute_rnd_acceleration(cache: pd.DataFrame) -> float:
    """Detect R&D spending acceleration.

    When R&D grows faster than its own 2-year average for consecutive
    quarters, a product launch is likely within 6-12 months.

    Returns ratio > 1.0 when R&D is accelerating.
    """
    # Try rd_expenses or sga_expenses as proxy
    rnd_col = None
    for col in ["rd_expenses", "sga_expenses"]:
        if col in cache.columns and cache[col].notna().sum() > 10:
            rnd_col = col
            break

    if rnd_col is None:
        return 0.0

    series = cache[rnd_col].dropna()
    if len(series) < 60:
        return 0.0

    # Detect quarterly transitions (value changes)
    shifted = series.shift(1)
    transitions = series[(series != shifted) & series.notna()]

    if len(transitions) < 3:
        return 0.0

    # Recent quarter vs average of all prior quarters
    recent = float(transitions.iloc[-1])
    prior_avg = float(transitions.iloc[:-1].mean())

    if prior_avg <= 0:
        return 0.0

    ratio = recent / prior_avg
    return round(ratio, 3)


# ---------------------------------------------------------------------------
# News Catalyst Score
# ---------------------------------------------------------------------------

def _compute_news_catalyst_score(
    news_articles: list[dict] | None,
) -> tuple[float, int, int]:
    """Score news articles for product catalyst signals.

    Returns (score_0_to_1, n_catalyst, n_negative).
    """
    if not news_articles:
        return 0.0, 0, 0

    n_catalyst = 0
    n_negative = 0
    total_scored = 0

    for article in news_articles[:50]:  # cap at 50
        title = article.get("title", "") + " " + article.get("description", "")
        if not title.strip():
            continue

        total_scored += 1
        has_catalyst = any(p.search(title) for p in _LAUNCH_PATTERNS)
        has_negative = any(p.search(title) for p in _NEGATIVE_PATTERNS)

        if has_catalyst:
            n_catalyst += 1
        if has_negative:
            n_negative += 1

    if total_scored == 0:
        return 0.0, 0, 0

    # Score: fraction of catalyst articles, penalized by negative articles
    raw_score = n_catalyst / total_scored
    penalty = n_negative / total_scored * 0.5
    score = max(0.0, min(1.0, raw_score - penalty))

    return round(score, 3), n_catalyst, n_negative


# ---------------------------------------------------------------------------
# Revenue Diversification
# ---------------------------------------------------------------------------

def _compute_revenue_diversification(cache: pd.DataFrame) -> float:
    """Detect revenue concentration changes.

    Uses gross_margin and operating_margin trajectory as a proxy for
    product mix shifts (new segments typically have different margins).

    Returns delta in [0, 1] where higher = more diversification change.
    """
    if "gross_margin" not in cache.columns:
        return 0.0

    gm = cache["gross_margin"].dropna()
    if len(gm) < 120:
        return 0.0

    # Compare recent quarter's margin volatility vs historical
    recent = gm.iloc[-63:]  # ~1 quarter
    historical = gm.iloc[:-63]

    if len(historical) < 60 or len(recent) < 20:
        return 0.0

    recent_std = float(recent.std())
    hist_std = float(historical.std())

    if hist_std < 1e-8:
        return 0.0

    # Large change in margin dispersion suggests product mix shift
    delta = abs(recent_std - hist_std) / hist_std
    return round(min(delta, 1.0), 3)


# ---------------------------------------------------------------------------
# Earnings Momentum
# ---------------------------------------------------------------------------

def _compute_earnings_momentum(cache: pd.DataFrame) -> float:
    """Detect earnings surprise momentum.

    Consecutive quarters beating/missing expectations create momentum
    that historical price models don't capture.

    Returns score in [-1, +1].
    """
    if "net_income" not in cache.columns:
        return 0.0

    ni = cache["net_income"].dropna()
    if len(ni) < 120:
        return 0.0

    # Detect quarterly transitions
    shifted = ni.shift(1)
    transitions = ni[(ni != shifted) & ni.notna()]

    if len(transitions) < 4:
        return 0.0

    # Check if recent quarters are trending up or down
    recent_4 = transitions.iloc[-4:].values
    if len(recent_4) < 4:
        return 0.0

    # Count consecutive up or down moves
    diffs = np.diff(recent_4)
    n_up = sum(1 for d in diffs if d > 0)
    n_down = sum(1 for d in diffs if d < 0)

    if n_up >= 3:
        return 0.8  # strong positive momentum
    elif n_up >= 2:
        return 0.4
    elif n_down >= 3:
        return -0.8  # strong negative momentum
    elif n_down >= 2:
        return -0.4
    return 0.0


# ---------------------------------------------------------------------------
# Main entry point
# ---------------------------------------------------------------------------

def detect_product_catalysts(
    cache: pd.DataFrame,
    profile: dict | None = None,
    news_articles: list[dict] | None = None,
) -> tuple[pd.DataFrame, CatalystResult]:
    """Detect product catalysts and inject signals into cache.

    Parameters
    ----------
    cache:
        Daily cache DataFrame with financial statement columns.
    profile:
        Company profile dict (for sector/industry context).
    news_articles:
        List of recent news article dicts from news_sentiment module.

    Returns
    -------
    (cache, CatalystResult)
    """
    result = CatalystResult()

    try:
        # Component signals
        rnd_accel = _compute_rnd_acceleration(cache)
        news_score, n_cat, n_neg = _compute_news_catalyst_score(news_articles)
        rev_div = _compute_revenue_diversification(cache)
        earnings_mom = _compute_earnings_momentum(cache)

        result.rnd_acceleration = rnd_accel
        result.news_catalyst_score = news_score
        result.n_catalyst_articles = n_cat
        result.n_negative_articles = n_neg
        result.revenue_diversification_delta = rev_div
        result.earnings_momentum = earnings_mom

        # Composite catalyst score (0-1)
        # Weight: R&D acceleration (30%), news (25%), earnings momentum (25%),
        # revenue diversification (20%)
        rnd_signal = min(1.0, max(0, (rnd_accel - 1.0) * 2.0))  # >1.0 = accelerating
        earnings_signal = (earnings_mom + 1.0) / 2.0  # normalize to 0-1

        composite = (
            rnd_signal * 0.30
            + news_score * 0.25
            + earnings_signal * 0.25
            + rev_div * 0.20
        )
        composite = round(float(np.clip(composite, 0, 1)), 3)
        result.catalyst_score = composite

        # Determine catalyst type
        if rnd_signal > 0.5 and news_score > 0.3:
            result.catalyst_type = "product_launch"
        elif rnd_signal > 0.5:
            result.catalyst_type = "rnd_surge"
        elif earnings_mom > 0.5:
            result.catalyst_type = "earnings_momentum"
        elif news_score > 0.4:
            result.catalyst_type = "market_narrative"
        elif rev_div > 0.3:
            result.catalyst_type = "segment_shift"
        else:
            result.catalyst_type = "none"

        # Inject into cache
        cache["catalyst_score"] = composite
        cache["rnd_acceleration"] = rnd_accel
        cache["catalyst_type"] = result.catalyst_type

        result.available = True
        result.detail = (
            f"score={composite:.2f}, type={result.catalyst_type}, "
            f"rnd={rnd_accel:.2f}, news={news_score:.2f}, "
            f"earnings={earnings_mom:+.2f}, rev_div={rev_div:.2f}"
        )

        logger.info(
            "Product catalysts: score=%.2f, type=%s, "
            "rnd_accel=%.2f, news=%.2f (%d articles), earnings=%+.2f",
            composite, result.catalyst_type, rnd_accel,
            news_score, n_cat, earnings_mom,
        )

    except Exception as exc:
        result.error = str(exc)[:200]
        logger.warning("Product catalyst detection failed: %s", exc)

    return cache, result
