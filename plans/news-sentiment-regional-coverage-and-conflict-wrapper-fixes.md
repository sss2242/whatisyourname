# News Sentiment Regional Coverage and Conflict Wrapper Fixes

## Problem Statement

The news sentiment module (`operator1/features/news_sentiment.py`) has significant regional coverage gaps -- GNews hardcodes English/US locale, only 5 of 25 markets have regional RSS feeds, and keyword scoring only works for English headlines. The conflict risk module (`operator1/features/conflict_risk.py`) works well but has one degraded data source (UCDP API requires auth since 2025).

---

## Current State

### News Sentiment (`news_sentiment.py`, 404 lines)

```
Fetch chain: GNews (English only) -> RSS (5 regional + English fallback) -> Empty
Score chain: LLM (any language) -> Keywords (English only) -> 0.0
```

**GNews** (primary fetcher): Hardcodes `language="en"` and `country="US"` at line 126. Every market gets English-only articles.

**RSS** (fallback): Has regional feeds for 5 markets only:
- `kr_dart` -- Korean Google News (ko)
- `jp_jquants` -- Japanese Google News (ja)
- `br_cvm` -- Portuguese Google News (pt-BR)
- `cn_sse` -- Chinese Google News (zh-CN)
- `tw_mops` -- Traditional Chinese Google News (zh-TW)

Missing: EU languages (fr, de, es, it, nl, sv), Arabic (ar), Hindi (hi), Cantonese (zh-HK)

**Keyword scoring**: 50 English keywords only. Non-English headlines always score 0.0.

**Dead code**: `_fetch_news_alpha_vantage()` is a redirect to GNews.

### Conflict Risk (`conflict_risk.py`, 931 lines)

**Static lists**: Complete coverage for all regions. Always works.

**UCDP GED API**: Returns 401 since 2025 (auth now required). Falls back gracefully to static lists, but loses real-time event counts.

**GDELT**: Works but rate-limited (5s between calls). Covers all markets.

---

## Implementation Plan

### Phase 1: Make GNews locale-aware (news_sentiment.py)

**File:** `operator1/features/news_sentiment.py`

Add a market-to-locale mapping and pass it to GNews:

```python
_MARKET_LOCALE: dict[str, tuple[str, str]] = {
    # market_id: (language, country)
    "us_sec_edgar":      ("en", "US"),
    "uk_companies_house": ("en", "GB"),
    "eu_esef_xbrl":      ("en", "GB"),  # pan-EU defaults to English
    "eu_esef_france":    ("fr", "FR"),
    "eu_esef_germany":   ("de", "DE"),
    "jp_jquants":        ("ja", "JP"),
    "kr_dart":           ("ko", "KR"),
    "tw_mops":           ("zh-Hant", "TW"),
    "br_cvm":            ("pt", "BR"),
    "cl_cmf":            ("es", "CL"),
    "cn_sse":            ("zh-Hans", "CN"),
    "in_bse":            ("en", "IN"),  # English is dominant for Indian financial news
    "au_asx":            ("en", "AU"),
    "hk_hkex":           ("zh-Hant", "HK"),
    "sg_sgx":            ("en", "SG"),
    "za_jse":            ("en", "ZA"),
    "sa_tadawul":        ("ar", "SA"),
    "ae_dfm":            ("ar", "AE"),
    "mx_bmv":            ("es", "MX"),
    "ch_six":            ("de", "CH"),
    "ca_sedar":          ("en", "CA"),
}
```

Update `_fetch_news_gnews()` signature to accept `market_id` and use the locale mapping.

### Phase 2: Expand regional RSS feeds (news_sentiment.py)

Add RSS feed templates for all missing markets:

```python
_REGIONAL_RSS: dict[str, list[str]] = {
    # Existing
    "kr_dart": [...],
    "jp_jquants": [...],
    "br_cvm": [...],
    "cn_sse": [...],
    "tw_mops": [...],
    # New
    "eu_esef_france": [
        "https://news.google.com/rss/search?q={symbol}+bourse&hl=fr&gl=FR&ceid=FR:fr",
        "https://news.google.com/rss/search?q={name}+actions&hl=fr&gl=FR&ceid=FR:fr",
    ],
    "eu_esef_germany": [
        "https://news.google.com/rss/search?q={symbol}+Aktie&hl=de&gl=DE&ceid=DE:de",
        "https://news.google.com/rss/search?q={name}+Boerse&hl=de&gl=DE&ceid=DE:de",
    ],
    "cl_cmf": [
        "https://news.google.com/rss/search?q={symbol}+acciones&hl=es&gl=CL&ceid=CL:es-419",
    ],
    "mx_bmv": [
        "https://news.google.com/rss/search?q={symbol}+acciones&hl=es&gl=MX&ceid=MX:es-419",
    ],
    "sa_tadawul": [
        "https://news.google.com/rss/search?q={symbol}+سهم&hl=ar&gl=SA&ceid=SA:ar",
    ],
    "ae_dfm": [
        "https://news.google.com/rss/search?q={symbol}+سهم&hl=ar&gl=AE&ceid=AE:ar",
    ],
    "hk_hkex": [
        "https://news.google.com/rss/search?q={symbol}+股價&hl=zh-TW&gl=HK&ceid=HK:zh-Hant",
    ],
    "ch_six": [
        "https://news.google.com/rss/search?q={symbol}+Aktie&hl=de&gl=CH&ceid=CH:de",
    ],
    "in_bse": [
        "https://news.google.com/rss/search?q={symbol}+stock&hl=en&gl=IN&ceid=IN:en",
        "https://news.google.com/rss/search?q={name}+share+price&hl=en&gl=IN&ceid=IN:en",
    ],
    "au_asx": [
        "https://news.google.com/rss/search?q={symbol}+ASX&hl=en&gl=AU&ceid=AU:en",
    ],
    "za_jse": [
        "https://news.google.com/rss/search?q={symbol}+JSE&hl=en&gl=ZA&ceid=ZA:en",
    ],
    "ca_sedar": [
        "https://news.google.com/rss/search?q={symbol}+TSX&hl=en&gl=CA&ceid=CA:en",
    ],
}
```

### Phase 3: Add multilingual keyword sets (news_sentiment.py)

Add keyword dictionaries for the 6 most common non-English languages in our markets:

```python
_POSITIVE_KEYWORDS_JA = {"上昇", "増益", "好調", "成長", "回復", "上方修正", ...}
_NEGATIVE_KEYWORDS_JA = {"下落", "減益", "低迷", "赤字", "下方修正", ...}

_POSITIVE_KEYWORDS_KO = {"상승", "흑자", "성장", "호실적", "상향", ...}
_NEGATIVE_KEYWORDS_KO = {"하락", "적자", "부진", "하향", "손실", ...}

_POSITIVE_KEYWORDS_ZH = {"上涨", "盈利", "增长", "利好", "突破", ...}
_NEGATIVE_KEYWORDS_ZH = {"下跌", "亏损", "下滑", "利空", "暴跌", ...}

_POSITIVE_KEYWORDS_PT = {"alta", "lucro", "crescimento", "valorização", ...}
_NEGATIVE_KEYWORDS_PT = {"queda", "prejuizo", "perda", "desvalorização", ...}

_POSITIVE_KEYWORDS_ES = {"alza", "ganancias", "crecimiento", "sube", ...}
_NEGATIVE_KEYWORDS_ES = {"baja", "perdidas", "caida", "desplome", ...}

_POSITIVE_KEYWORDS_AR = {"ارتفاع", "أرباح", "نمو", "صعود", ...}
_NEGATIVE_KEYWORDS_AR = {"انخفاض", "خسائر", "تراجع", "هبوط", ...}
```

Update `_keyword_score()` to accept a `language` parameter and select the right keyword set.

### Phase 4: Wire market_id through the sentiment pipeline (news_sentiment.py)

Update `compute_news_sentiment()` to accept `market_id` and `company_name` parameters:

```python
def compute_news_sentiment(
    cache, *, gemini_client=None, symbol="",
    market_id="", company_name="",  # NEW parameters
    news_df=None,
) -> tuple[pd.DataFrame, SentimentResult]:
```

Pass `market_id` to:
- `_fetch_news_gnews()` for locale selection
- `_fetch_news_rss()` for regional RSS feeds (already accepts it)
- `_keyword_score()` for language-appropriate keywords

### Phase 5: Clean up dead code (news_sentiment.py)

- Remove `_fetch_news_alpha_vantage()` (dead redirect)
- Update all callers to use `_fetch_news_gnews()` directly
- Remove misleading docstring references to FMP/Alpha Vantage

### Phase 6: UCDP API auth handling (conflict_risk.py)

Add optional UCDP API key support:

- Add `UCDP_API_KEY` to the `.env` loading in `secrets_loader.py` (optional key, not required)
- If present, pass as Bearer token in the UCDP request headers
- If not present, continue with the current graceful fallback to static lists
- Document in the `.env.example` that UCDP auth is optional but improves conflict data granularity

---

## Files Changed

| File | Changes |
|------|---------|
| `operator1/features/news_sentiment.py` | Phases 1-5: locale-aware GNews, expanded RSS, multilingual keywords, market_id wiring, dead code removal |
| `operator1/features/conflict_risk.py` | Phase 6: optional UCDP API key support |
| `operator1/secrets_loader.py` | Phase 6: add UCDP_API_KEY as optional key |
| `main.py` | Phase 4: pass market_id and company_name to compute_news_sentiment |

---

## Dependency Flow

```
Phase 1 (GNews locale) -- independent
Phase 2 (RSS feeds) -- independent
Phase 3 (multilingual keywords) -- independent
Phase 4 (wiring) -- depends on Phases 1-3
Phase 5 (cleanup) -- independent
Phase 6 (UCDP auth) -- independent
```

Phases 1, 2, 3, 5, 6 can be done in parallel. Phase 4 should come after 1-3.

---

## Testing Strategy

- All 33 existing conflict_risk tests must continue passing
- Add new tests for:
  - GNews locale selection per market
  - RSS feed URLs generated for each regional market
  - Multilingual keyword scoring (Japanese, Korean, Chinese, Portuguese, Spanish, Arabic)
  - `compute_news_sentiment()` with market_id parameter
  - UCDP API key header injection
