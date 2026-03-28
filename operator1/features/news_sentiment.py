"""News Sentiment Scoring -- daily sentiment from stock news.

Fetches stock news via GNews (Google News scraper, no API key) or RSS
fallback, scores sentiment via LLM (1 API call for all headlines),
and injects daily sentiment columns into the cache for temporal model
learning.

GNews targets the correct country for the market (English articles from
that country's news ecosystem), with local-language RSS feeds as
supplementary data scored by the LLM.

Falls back to keyword-based scoring if the LLM is unavailable.

Top-level entry point:
    ``compute_news_sentiment(cache, symbol, llm_client=None,
                             market_id="", company_name="")``
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
    # Scored article headlines for downstream modules (e.g., product_catalysts)
    articles: list[dict] = field(default_factory=list)
    mean_sentiment: float = float("nan")
    latest_sentiment: float = float("nan")
    latest_label: str = "Unknown"
    columns_added: list[str] = field(default_factory=list)


# ---------------------------------------------------------------------------
# Keyword fallback scorer
# ---------------------------------------------------------------------------


def _keyword_score(text: str) -> float:
    """Score a headline for sentiment. Returns -1.0 to +1.0.

    Prefers VADER (handles negation, intensity, context) when available.
    Falls back to simple keyword matching.
    """
    if not text:
        return 0.0

    # Try VADER first (better accuracy: handles negation, degree modifiers)
    try:
        from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
        analyzer = SentimentIntensityAnalyzer()
        return analyzer.polarity_scores(text)["compound"]
    except ImportError:
        pass

    # Fallback: simple keyword matching
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
# Market -> GNews country code mapping
# ---------------------------------------------------------------------------

# GNews country codes (ISO-2 uppercase).  We keep language="en" for all
# markets so keyword scoring works, but target the correct country so
# Google News returns locally relevant English articles.
_MARKET_TO_GNEWS_COUNTRY: dict[str, str] = {
    "us_sec_edgar":       "US",
    "uk_companies_house": "GB",
    "eu_esef":            "GB",   # pan-EU defaults to UK English news
    "fr_esef":            "FR",
    "de_esef":            "DE",
    "nl_esef":            "NL",
    "es_esef":            "ES",
    "it_esef":            "IT",
    "se_esef":            "SE",
    "jp_jquants":         "JP",
    "kr_dart":            "KR",
    "tw_mops":            "TW",
    "br_cvm":             "BR",
    "cl_cmf":             "CL",
    "cn_sse":             "CN",
    "in_bse":             "IN",
    "au_asx":             "AU",
    "hk_hkex":            "HK",
    "sg_sgx":             "SG",
    "za_jse":             "ZA",
    "sa_tadawul":         "SA",
    "ae_dfm":             "AE",
    "mx_bmv":             "MX",
    "ch_six":             "CH",
    "ca_sedar":           "CA",
}


# ---------------------------------------------------------------------------
# News fetchers (free APIs)
# ---------------------------------------------------------------------------


def _fetch_news_gnews(
    symbol: str,
    market_id: str = "",
) -> pd.DataFrame:
    """Fetch stock news via GNews (Google News scraper, no API key).

    Uses the gnews library to search Google News for recent English articles
    about the given stock symbol/company. The ``market_id`` determines which
    country's news ecosystem to search (e.g. Korean financial news in English
    for ``kr_dart``), improving relevance for non-US markets.

    Returns a DataFrame with columns: date, title, url, source.
    """
    try:
        from gnews import GNews
    except ImportError:
        logger.debug("gnews not installed; trying RSS fallback")
        return _fetch_news_rss(symbol, market_id=market_id)

    # Target the correct country but keep English for keyword scoring
    country = _MARKET_TO_GNEWS_COUNTRY.get(market_id, "US")

    try:
        gn = GNews(language="en", country=country, period="6m", max_results=50)
        articles = gn.get_news(f"{symbol} stock")
        if not articles:
            return pd.DataFrame()

        rows = []
        for art in articles:
            rows.append({
                "date": pd.to_datetime(art.get("published date", ""), errors="coerce"),
                "title": art.get("title", ""),
                "url": art.get("url", ""),
                "source": art.get("publisher", {}).get("title", "") if isinstance(art.get("publisher"), dict) else str(art.get("publisher", "")),
            })

        df = pd.DataFrame(rows)
        df = df.dropna(subset=["date"])
        logger.info("GNews fetched %d articles for %s (country=%s)", len(df), symbol, country)
        return df

    except Exception as exc:
        logger.warning("GNews fetch failed for %s: %s; trying RSS", symbol, exc)
        return _fetch_news_rss(symbol, market_id=market_id)


def _fetch_news_rss(symbol: str, market_id: str = "", company_name: str = "") -> pd.DataFrame:
    """Fetch stock news via regional RSS feeds with Google News fallback.

    Tries per-region news sources first (Naver for Korea, Yahoo JP for Japan,
    etc.), then falls back to Google News RSS.
    """
    try:
        import feedparser
    except ImportError:
        logger.debug("feedparser not installed; no news source available")
        return pd.DataFrame()

    import urllib.parse

    # Per-region RSS URLs -- local-language supplementary feeds.
    # These are scored by the LLM (keyword scoring is English-only).
    # English articles come from GNews; these add local-language coverage.
    _REGIONAL_RSS: dict[str, list[str]] = {
        # Asia
        "kr_dart": [
            "https://news.google.com/rss/search?q={symbol}+주식&hl=ko&gl=KR&ceid=KR:ko",
            "https://news.google.com/rss/search?q={name}+주가&hl=ko&gl=KR&ceid=KR:ko",
        ],
        "jp_jquants": [
            "https://news.google.com/rss/search?q={symbol}+株価&hl=ja&gl=JP&ceid=JP:ja",
            "https://news.google.com/rss/search?q={name}+株式&hl=ja&gl=JP&ceid=JP:ja",
        ],
        "cn_sse": [
            "https://news.google.com/rss/search?q={symbol}+股票&hl=zh-CN&gl=CN&ceid=CN:zh-Hans",
        ],
        "tw_mops": [
            "https://news.google.com/rss/search?q={symbol}+股價&hl=zh-TW&gl=TW&ceid=TW:zh-Hant",
        ],
        "hk_hkex": [
            "https://news.google.com/rss/search?q={symbol}+股價&hl=zh-TW&gl=HK&ceid=HK:zh-Hant",
        ],
        "in_bse": [
            "https://news.google.com/rss/search?q={symbol}+stock&hl=en&gl=IN&ceid=IN:en",
            "https://news.google.com/rss/search?q={name}+share+price&hl=en&gl=IN&ceid=IN:en",
        ],
        "sg_sgx": [
            "https://news.google.com/rss/search?q={symbol}+SGX&hl=en&gl=SG&ceid=SG:en",
        ],
        # South America
        "br_cvm": [
            "https://news.google.com/rss/search?q={symbol}+ações&hl=pt-BR&gl=BR&ceid=BR:pt-419",
            "https://news.google.com/rss/search?q={name}+bolsa&hl=pt-BR&gl=BR&ceid=BR:pt-419",
        ],
        "cl_cmf": [
            "https://news.google.com/rss/search?q={symbol}+acciones&hl=es&gl=CL&ceid=CL:es-419",
        ],
        # Europe
        "fr_esef": [
            "https://news.google.com/rss/search?q={symbol}+bourse&hl=fr&gl=FR&ceid=FR:fr",
            "https://news.google.com/rss/search?q={name}+actions&hl=fr&gl=FR&ceid=FR:fr",
        ],
        "de_esef": [
            "https://news.google.com/rss/search?q={symbol}+Aktie&hl=de&gl=DE&ceid=DE:de",
            "https://news.google.com/rss/search?q={name}+Boerse&hl=de&gl=DE&ceid=DE:de",
        ],
        "nl_esef": [
            "https://news.google.com/rss/search?q={symbol}+aandeel&hl=nl&gl=NL&ceid=NL:nl",
            "https://news.google.com/rss/search?q={name}+beurs&hl=nl&gl=NL&ceid=NL:nl",
        ],
        "es_esef": [
            "https://news.google.com/rss/search?q={symbol}+acciones&hl=es&gl=ES&ceid=ES:es",
            "https://news.google.com/rss/search?q={name}+bolsa&hl=es&gl=ES&ceid=ES:es",
        ],
        "it_esef": [
            "https://news.google.com/rss/search?q={symbol}+azioni&hl=it&gl=IT&ceid=IT:it",
            "https://news.google.com/rss/search?q={name}+borsa&hl=it&gl=IT&ceid=IT:it",
        ],
        "se_esef": [
            "https://news.google.com/rss/search?q={symbol}+aktie&hl=sv&gl=SE&ceid=SE:sv",
            "https://news.google.com/rss/search?q={name}+börs&hl=sv&gl=SE&ceid=SE:sv",
        ],
        "ch_six": [
            "https://news.google.com/rss/search?q={symbol}+Aktie&hl=de&gl=CH&ceid=CH:de",
        ],
        # Middle East
        "sa_tadawul": [
            "https://news.google.com/rss/search?q={symbol}+سهم&hl=ar&gl=SA&ceid=SA:ar",
        ],
        "ae_dfm": [
            "https://news.google.com/rss/search?q={symbol}+سهم&hl=ar&gl=AE&ceid=AE:ar",
        ],
        # Americas
        "mx_bmv": [
            "https://news.google.com/rss/search?q={symbol}+acciones&hl=es&gl=MX&ceid=MX:es-419",
        ],
        "ca_sedar": [
            "https://news.google.com/rss/search?q={symbol}+TSX&hl=en&gl=CA&ceid=CA:en",
        ],
        # Oceania
        "au_asx": [
            "https://news.google.com/rss/search?q={symbol}+ASX&hl=en&gl=AU&ceid=AU:en",
        ],
        # Africa
        "za_jse": [
            "https://news.google.com/rss/search?q={symbol}+JSE&hl=en&gl=ZA&ceid=ZA:en",
        ],
    }

    # Try regional sources first
    urls_to_try: list[str] = []
    if market_id:
        regional = _REGIONAL_RSS.get(market_id, [])
        for tpl in regional:
            url = tpl.format(
                symbol=urllib.parse.quote(symbol),
                name=urllib.parse.quote(company_name or symbol),
            )
            urls_to_try.append(url)

    # Always add English Google News as final fallback
    urls_to_try.append(
        f"https://news.google.com/rss/search?q={urllib.parse.quote(symbol)}+stock&hl=en-US&gl=US&ceid=US:en"
    )
    if company_name and company_name != symbol:
        urls_to_try.append(
            f"https://news.google.com/rss/search?q={urllib.parse.quote(company_name)}+stock&hl=en-US&gl=US&ceid=US:en"
        )

    for url in urls_to_try:
        try:
            feed = feedparser.parse(url)
            if not feed.entries:
                continue

            rows = []
            for entry in feed.entries[:50]:
                pub_date = entry.get("published", "")
                rows.append({
                    "date": pd.to_datetime(pub_date, errors="coerce"),
                    "title": entry.get("title", ""),
                    "url": entry.get("link", ""),
                    "source": entry.get("source", {}).get("title", "") if isinstance(entry.get("source"), dict) else "",
                })

            df = pd.DataFrame(rows)
            df = df.dropna(subset=["date"])
            if not df.empty:
                logger.info("RSS fetched %d articles for %s from %s", len(df), symbol, url[:60])
                return df
        except Exception:
            continue

    logger.debug("No RSS articles found for %s across %d sources", symbol, len(urls_to_try))
    return pd.DataFrame()





# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def compute_news_sentiment(
    cache: pd.DataFrame,
    *,
    _legacy_fmp_client: Any = None,
    llm_client: Any = None,
    symbol: str = "",
    market_id: str = "",
    company_name: str = "",
    news_df: pd.DataFrame | None = None,
) -> tuple[pd.DataFrame, SentimentResult]:
    """Compute daily news sentiment and inject into cache.

    Fetches English news via GNews (targeted to the market's country),
    supplements with local-language RSS feeds, then scores via LLM
    (1 API call) or keyword fallback.

    Parameters
    ----------
    cache:
        Daily cache DataFrame (DatetimeIndex).
    _legacy_fmp_client:
        Legacy parameter, ignored.
    llm_client:
        LLM client instance for AI sentiment scoring. Optional.
        Accepts GeminiClient, ClaudeClient, OpenRouterClient, or
        PooledLLMClient.
    symbol:
        Trading symbol (e.g. 'AAPL') for news query.
    market_id:
        Market identifier (e.g. 'kr_dart') for country-targeted
        news fetching. Derived from the user's region selection in
        run.py or from the ``--market`` CLI argument.
    company_name:
        Company name for supplementary RSS searches.
    news_df:
        Pre-fetched news DataFrame (for testing). If provided,
        news fetchers are not called.

    Returns
    -------
    (cache, result)
        Cache with sentiment_* columns, and SentimentResult summary.
    """
    logger.info("Computing news sentiment for %s...", symbol or "target")

    result = SentimentResult()

    # Step 1: Fetch news
    # Priority: pre-fetched > GNews (English, country-targeted) > RSS (local language) > empty
    if news_df is not None:
        articles = news_df.copy()
    elif symbol:
        # Primary: English articles from the target country
        articles = _fetch_news_gnews(symbol, market_id=market_id)
        # Supplementary: local-language RSS (if market has regional feeds)
        if market_id:
            rss_articles = _fetch_news_rss(
                symbol, market_id=market_id, company_name=company_name or symbol,
            )
            if not rss_articles.empty:
                if articles.empty:
                    articles = rss_articles
                else:
                    # Merge, deduplicate by title
                    articles = pd.concat([articles, rss_articles], ignore_index=True)
                    articles = articles.drop_duplicates(subset=["title"], keep="first")
                    logger.info(
                        "Merged %d RSS articles with GNews (total: %d)",
                        len(rss_articles), len(articles),
                    )
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
    if llm_client is not None and hasattr(llm_client, "score_sentiment"):
        try:
            scores = llm_client.score_sentiment(headlines)
            if len(scores) == len(headlines):
                result.scoring_method = "gemini"
                logger.info("Scored %d headlines via Gemini", len(scores))
            else:
                scores = []
        except Exception as exc:
            logger.warning("Gemini sentiment failed, falling back to keyword: %s", exc)
            scores = []

    if not scores or all(s == 0.0 for s in scores):
        # LLM failed or returned all zeros -- use VADER/keyword fallback
        scores = [_keyword_score(h) for h in headlines]
        result.scoring_method = "keyword"
        logger.info("Scored %d headlines via VADER/keyword fallback", len(scores))

    articles["sentiment"] = scores
    result.n_articles_scored = len(scores)

    # Store scored articles for downstream modules (e.g., product_catalysts)
    # Filter out NaN-scored articles so catalysts sees real scores only.
    try:
        valid_articles = articles[articles["sentiment"].notna()]
        result.articles = valid_articles[["title", "sentiment"]].to_dict("records")
    except Exception:
        result.articles = []

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
