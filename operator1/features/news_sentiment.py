"""News Sentiment Scoring -- daily sentiment from stock news.

Fetches stock news from FMP (1 API call), scores sentiment via Gemini
(1 API call for all headlines), and injects daily sentiment columns
into the cache for temporal model learning.

Falls back to keyword-based scoring if Gemini is unavailable.

Top-level entry point:
    ``compute_news_sentiment(cache, symbol, gemini_client=None)``
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
# Constants
# ---------------------------------------------------------------------------

# Rolling windows for momentum and volatility
_MOMENTUM_SHORT: int = 5
_MOMENTUM_LONG: int = 21

# Keyword-based fallback scoring
_POSITIVE_KEYWORDS: set[str] = {
    "record", "growth", "beat", "profit", "upgrade", "buyback",
    "surge", "rally", "gain", "revenue", "strong", "positive",
    "outperform", "expand", "raise", "exceed", "dividend",
    "acquisition", "partnership", "breakthrough", "innovation",
    "approval", "launch", "bullish", "upside", "recovery",
}

_NEGATIVE_KEYWORDS: set[str] = {
    "decline", "loss", "downgrade", "lawsuit", "fine", "debt",
    "drop", "fall", "miss", "weak", "warning", "concern",
    "layoff", "restructuring", "investigation", "default",
    "bankruptcy", "sell", "bearish", "downside", "recession",
    "crash", "plunge", "violation", "fraud", "delay",
}

# Sentiment label thresholds
_LABEL_THRESHOLDS: list[tuple[float, str]] = [
    (-0.3, "Bearish"),
    (-0.1, "Slightly Bearish"),
    (0.1, "Neutral"),
    (0.3, "Slightly Bullish"),
    (1.1, "Bullish"),
]


# ---------------------------------------------------------------------------
# Result container
# ---------------------------------------------------------------------------


@dataclass
class SentimentResult:
    """Summary of news sentiment scoring."""

    n_articles_fetched: int = 0
    n_articles_scored: int = 0
    scoring_method: str = "none"  # "gemini", "keyword", "none"
    mean_sentiment: float = float("nan")
    latest_sentiment: float = float("nan")
    latest_label: str = "Unknown"
    columns_added: list[str] = field(default_factory=list)


# ---------------------------------------------------------------------------
# Keyword fallback scorer
# ---------------------------------------------------------------------------


def _keyword_score(text: str) -> float:
    """Score a headline using keyword matching. Returns -1.0 to +1.0."""
    if not text:
        return 0.0
    words = set(re.findall(r'\w+', text.lower()))
    pos = len(words & _POSITIVE_KEYWORDS)
    neg = len(words & _NEGATIVE_KEYWORDS)
    total = pos + neg
    if total == 0:
        return 0.0
    return (pos - neg) / total


def _sentiment_label(score: float) -> str:
    """Map a sentiment score to a label."""
    if np.isnan(score):
        return "Unknown"
    for threshold, label in _LABEL_THRESHOLDS:
        if score < threshold:
            return label
    return "Bullish"


# ---------------------------------------------------------------------------
# News fetchers (free APIs)
# ---------------------------------------------------------------------------


def _fetch_news_alpha_vantage(symbol: str) -> pd.DataFrame:
    """Alpha Vantage news endpoint -- removed (paid/commercial API).

    Returns an empty DataFrame. Previously used Alpha Vantage
    NEWS_SENTIMENT endpoint.
    """
    logger.debug("Alpha Vantage news removed -- returning empty DataFrame")
    return pd.DataFrame()


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def compute_news_sentiment(
    cache: pd.DataFrame,
    *,
    _legacy_fmp_client: Any = None,
    gemini_client: Any = None,
    symbol: str = "",
    news_df: pd.DataFrame | None = None,
) -> tuple[pd.DataFrame, SentimentResult]:
    """Compute daily news sentiment and inject into cache.

    Uses 1 FMP API call to fetch all news, then 1 Gemini call to
    batch-score all headlines. Falls back to keyword scoring if
    Gemini is unavailable.

    Parameters
    ----------
    cache:
        Daily cache DataFrame (DatetimeIndex).
    _legacy_fmp_client:
        Legacy parameter, ignored.
    gemini_client:
        GeminiClient instance for AI sentiment scoring. Optional.
    symbol:
        Trading symbol (e.g. 'AAPL') for FMP news query.
    news_df:
        Pre-fetched news DataFrame (for testing). If provided,
        FMP client is not called.

    Returns
    -------
    (cache, result)
        Cache with sentiment_* columns, and SentimentResult summary.
    """
    logger.info("Computing news sentiment for %s...", symbol or "target")

    result = SentimentResult()

    # Step 1: Fetch news
    # Priority: pre-fetched > Alpha Vantage (free) > legacy FMP > empty
    if news_df is not None:
        articles = news_df.copy()
    elif symbol:
        articles = _fetch_news_alpha_vantage(symbol)
        if articles.empty and _legacy_fmp_client is not None:
            try:
                articles = _legacy_fmp_client.get_stock_news(symbol, limit=1000)
            except Exception as exc:
                logger.warning("FMP news fetch failed: %s", exc)
                articles = pd.DataFrame()
    else:
        logger.warning("No symbol for sentiment -- skipping news fetch")
        articles = pd.DataFrame()

    if articles.empty or "title" not in articles.columns:
        logger.info("No news articles available for sentiment scoring")
        cache["sentiment_score"] = np.nan
        cache["sentiment_count"] = 0
        cache["sentiment_momentum_5d"] = np.nan
        cache["sentiment_momentum_21d"] = np.nan
        cache["sentiment_volatility_21d"] = np.nan
        cache["is_missing_sentiment"] = 1
        result.columns_added = [
            "sentiment_score", "sentiment_count",
            "sentiment_momentum_5d", "sentiment_momentum_21d",
            "sentiment_volatility_21d", "is_missing_sentiment",
        ]
        return cache, result

    result.n_articles_fetched = len(articles)

    # Step 2: Score headlines (1 Gemini API call, or keyword fallback)
    headlines = articles["title"].fillna("").tolist()

    scores: list[float] = []
    if gemini_client is not None and hasattr(gemini_client, "score_sentiment"):
        try:
            scores = gemini_client.score_sentiment(headlines)
            if len(scores) == len(headlines):
                result.scoring_method = "gemini"
                logger.info("Scored %d headlines via Gemini", len(scores))
            else:
                scores = []
        except Exception as exc:
            logger.warning("Gemini sentiment failed, falling back to keyword: %s", exc)
            scores = []

    if not scores:
        # Keyword fallback (0 API calls)
        scores = [_keyword_score(h) for h in headlines]
        result.scoring_method = "keyword"
        logger.info("Scored %d headlines via keyword fallback", len(scores))

    articles["sentiment"] = scores
    result.n_articles_scored = len(scores)

    # Step 3: Align to daily cache via as-of logic
    if "publishedDate" not in articles.columns:
        # Try common alternatives
        for alt in ("date", "published_date", "datetime"):
            if alt in articles.columns:
                articles = articles.rename(columns={alt: "publishedDate"})
                break

    if "publishedDate" in articles.columns:
        # Convert with utc=True then strip timezone to avoid
        # "Tz-aware datetime cannot be converted to datetime64" errors
        # when reindexing against the tz-naive cache index.
        _dt = pd.to_datetime(articles["publishedDate"], utc=True, errors="coerce")
        articles["date"] = _dt.dt.tz_localize(None).dt.normalize()
    else:
        # Last resort: assign today's date to all
        articles["date"] = pd.Timestamp.now().normalize()

    # Group by date: mean sentiment and article count
    daily = articles.groupby("date").agg(
        sentiment_mean=("sentiment", "mean"),
        article_count=("sentiment", "count"),
    )

    # Reindex to cache dates with forward-fill
    daily_sentiment = daily["sentiment_mean"].reindex(cache.index, method="ffill")
    daily_count = daily["article_count"].reindex(cache.index, fill_value=0)

    # Inject columns
    cache["sentiment_score"] = daily_sentiment
    cache["sentiment_count"] = daily_count.astype(int)
    cache["sentiment_momentum_5d"] = daily_sentiment.rolling(
        _MOMENTUM_SHORT, min_periods=1
    ).mean()
    cache["sentiment_momentum_21d"] = daily_sentiment.rolling(
        _MOMENTUM_LONG, min_periods=1
    ).mean()
    cache["sentiment_volatility_21d"] = daily_sentiment.rolling(
        _MOMENTUM_LONG, min_periods=2
    ).std()
    cache["is_missing_sentiment"] = daily_sentiment.isna().astype(int)

    result.columns_added = [
        "sentiment_score", "sentiment_count",
        "sentiment_momentum_5d", "sentiment_momentum_21d",
        "sentiment_volatility_21d", "is_missing_sentiment",
    ]

    # Summary
    valid = daily_sentiment.dropna()
    if len(valid) > 0:
        result.mean_sentiment = float(valid.mean())
        result.latest_sentiment = float(valid.iloc[-1])
        result.latest_label = _sentiment_label(result.latest_sentiment)

    logger.info(
        "News sentiment: %d articles, method=%s, mean=%.3f, latest=%.3f (%s)",
        result.n_articles_scored,
        result.scoring_method,
        result.mean_sentiment,
        result.latest_sentiment,
        result.latest_label,
    )

    return cache, result
