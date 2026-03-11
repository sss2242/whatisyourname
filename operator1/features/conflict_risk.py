"""War/conflict risk assessment for companies and countries.

Flags companies and their countries when affected by armed conflict, war,
sanctions, or geopolitical instability. Integrates into the survival
analysis framework -- a company in a war-affected country faces existential
risk that financial-only survival models do not capture.

Data sources (all free):
  1. UCDP GED API -- academic-grade conflict event data (no key)
  2. World Bank FCS list -- official fragile/conflict state classification
  3. Major sanctions lists -- OFAC/EU hardcoded country lists
  4. GDELT -- real-time news-based conflict monitoring (no key)

Top-level entry point:
    ``assess_conflict_risk(country_iso2, company_name=None)``

Output columns added to the daily cache:
  - ``country_conflict_flag``: 1 if active conflict in country
  - ``company_conflict_flag``: 1 if company directly affected
  - ``conflict_intensity_score``: 0.0 (peace) to 1.0 (active war)
  - ``sanctions_flag``: 1 if country under major sanctions
  - ``conflict_type``: none / low_intensity / civil_war / interstate_war
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from datetime import date, timedelta
from typing import Any

import requests

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Result container
# ---------------------------------------------------------------------------

@dataclass
class ConflictRiskResult:
    """Container for war/conflict risk assessment."""

    country_iso2: str = ""

    # Flags
    country_conflict_flag: bool = False
    company_conflict_flag: bool = False
    sanctions_flag: bool = False
    fragile_state_flag: bool = False

    # Scores
    conflict_intensity_score: float = 0.0   # 0.0 = peace, 1.0 = active war
    conflict_type: str = "none"             # none, low_intensity, civil_war, interstate_war, sanctions

    # Event-level detail (from UCDP)
    recent_events_30d: int = 0
    recent_events_90d: int = 0
    recent_fatalities_30d: int = 0
    recent_fatalities_90d: int = 0
    conflict_trend: str = "stable"          # escalating, stable, de-escalating

    # News-based (from GDELT)
    news_conflict_mentions_7d: int = 0
    news_conflict_tone: float = 0.0

    # Metadata
    data_sources_used: list[str] = field(default_factory=list)
    assessment_date: str = ""
    confidence: float = 0.0                 # 0-1, based on data availability
    errors: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        """Serialize to dict for profile storage."""
        return {
            "country_iso2": self.country_iso2,
            "country_conflict_flag": self.country_conflict_flag,
            "company_conflict_flag": self.company_conflict_flag,
            "sanctions_flag": self.sanctions_flag,
            "fragile_state_flag": self.fragile_state_flag,
            "conflict_intensity_score": round(self.conflict_intensity_score, 3),
            "conflict_type": self.conflict_type,
            "recent_events_30d": self.recent_events_30d,
            "recent_events_90d": self.recent_events_90d,
            "recent_fatalities_30d": self.recent_fatalities_30d,
            "recent_fatalities_90d": self.recent_fatalities_90d,
            "conflict_trend": self.conflict_trend,
            "news_conflict_mentions_7d": self.news_conflict_mentions_7d,
            "news_conflict_tone": round(self.news_conflict_tone, 3),
            "data_sources_used": self.data_sources_used,
            "assessment_date": self.assessment_date,
            "confidence": round(self.confidence, 2),
        }


# ---------------------------------------------------------------------------
# Static lists (updated periodically)
# ---------------------------------------------------------------------------

# World Bank FCS (Fragile and Conflict-affected Situations) list 2024-2025.
# Source: https://www.worldbank.org/en/topic/fragilityconflictviolence/brief/harmonized-list-of-fragile-situations
# ISO-2 codes for countries classified as fragile/conflict-affected.
FRAGILE_CONFLICT_STATES: set[str] = {
    "AF",  # Afghanistan
    "BF",  # Burkina Faso
    "BI",  # Burundi
    "CF",  # Central African Republic
    "TD",  # Chad
    "KM",  # Comoros
    "CD",  # DR Congo
    "ER",  # Eritrea
    "ET",  # Ethiopia
    "GN",  # Guinea
    "GW",  # Guinea-Bissau
    "HT",  # Haiti
    "IQ",  # Iraq
    "LB",  # Lebanon
    "LY",  # Libya
    "ML",  # Mali
    "MZ",  # Mozambique
    "MM",  # Myanmar
    "NE",  # Niger
    "NG",  # Nigeria
    "KP",  # North Korea
    "PS",  # Palestine
    "PG",  # Papua New Guinea
    "SO",  # Somalia
    "SS",  # South Sudan
    "SD",  # Sudan
    "SY",  # Syria
    "TL",  # Timor-Leste
    "VE",  # Venezuela
    "YE",  # Yemen
    "ZW",  # Zimbabwe
}

# Countries under major international sanctions (US OFAC + EU combined).
# These are country-wide or comprehensive sanctions programs.
SANCTIONED_COUNTRIES: set[str] = {
    "CU",  # Cuba -- US comprehensive sanctions
    "IR",  # Iran -- US + EU comprehensive sanctions
    "KP",  # North Korea -- US + EU + UN comprehensive sanctions
    "SY",  # Syria -- US + EU sanctions
    "RU",  # Russia -- US + EU extensive sanctions (since 2022)
    "BY",  # Belarus -- US + EU sanctions (since 2020)
    "VE",  # Venezuela -- US sector sanctions
    "MM",  # Myanmar -- US + EU targeted sanctions
    "SD",  # Sudan -- US comprehensive sanctions (partial lift)
    "SS",  # South Sudan -- US targeted sanctions
    "SO",  # Somalia -- US targeted sanctions
    "YE",  # Yemen -- US targeted sanctions (Houthis)
    "AF",  # Afghanistan -- Taliban sanctions
}

# Countries with active interstate or major civil wars (as of 2024-2025).
# Continuously updated; these are the most severe ongoing conflicts.
ACTIVE_WAR_COUNTRIES: set[str] = {
    "UA",  # Ukraine -- Russia-Ukraine war (since 2022)
    "PS",  # Palestine -- Israel-Palestine conflict
    "IL",  # Israel -- active military operations
    "SD",  # Sudan -- civil war (since 2023)
    "MM",  # Myanmar -- civil war (since 2021)
    "SY",  # Syria -- ongoing civil war
    "YE",  # Yemen -- civil war + Houthi conflict
    "ET",  # Ethiopia -- Tigray/Amhara conflicts
    "SO",  # Somalia -- Al-Shabaab insurgency
}

# Conflict start dates for time-varying historical flags.
# Before these dates, the country was relatively peaceful and the
# conflict flag should be 0 in the daily cache.
CONFLICT_START_DATES: dict[str, str] = {
    "UA": "2022-02-24",   # Russia-Ukraine full-scale invasion
    "PS": "2023-10-07",   # Israel-Palestine escalation
    "IL": "2023-10-07",   # Israel military operations post Oct 7
    "SD": "2023-04-15",   # Sudan RSF vs SAF civil war
    "MM": "2021-02-01",   # Myanmar military coup + civil war
    "ET": "2020-11-04",   # Tigray war start (Amhara ongoing)
    "SY": "2011-03-15",   # Syrian civil war (long-running)
    "YE": "2014-09-21",   # Houthi takeover of Sana'a
    "SO": "2006-12-24",   # Al-Shabaab insurgency (long-running)
}

# ISO-2 to UCDP country ID mapping (for major markets).
_ISO2_TO_UCDP_ID: dict[str, int] = {
    "AF": 700, "IQ": 645, "SY": 652, "YE": 679, "LY": 620,
    "UA": 369, "RU": 365, "IL": 666, "PS": 6661, "ET": 530,
    "SD": 625, "SS": 626, "MM": 775, "SO": 520, "NG": 475,
    "ML": 432, "BF": 439, "CD": 490, "CF": 482, "MZ": 541,
    "CO": 100,
    # Tier 1+2 market countries (expected low/zero conflict)
    "US": 2, "GB": 200, "JP": 740, "KR": 732, "TW": 713,
    "BR": 140, "CL": 155, "IN": 750, "CN": 710, "HK": 7101,
    "AU": 900, "SG": 830, "ZA": 560, "SA": 670, "AE": 696,
    "DE": 255, "FR": 220, "CA": 20, "CH": 225, "MX": 70,
    # EU ESEF market countries
    "ES": 230, "IT": 325, "NL": 210, "SE": 380,
    "EU": 2551,  # EU aggregate (uses DE as proxy)
}

# ISO-2 to country name for GDELT queries
_ISO2_TO_NAME: dict[str, str] = {
    "AF": "Afghanistan", "IQ": "Iraq", "SY": "Syria", "YE": "Yemen",
    "UA": "Ukraine", "RU": "Russia", "IL": "Israel", "PS": "Palestine",
    "ET": "Ethiopia", "SD": "Sudan", "SS": "South Sudan", "MM": "Myanmar",
    "SO": "Somalia", "NG": "Nigeria", "LY": "Libya", "ML": "Mali",
    "US": "United States", "GB": "United Kingdom", "JP": "Japan",
    "KR": "South Korea", "TW": "Taiwan", "BR": "Brazil", "CL": "Chile",
    "IN": "India", "CN": "China", "HK": "Hong Kong", "AU": "Australia",
    "SG": "Singapore", "ZA": "South Africa", "SA": "Saudi Arabia",
    "AE": "UAE", "DE": "Germany", "FR": "France", "CA": "Canada",
    "CH": "Switzerland", "MX": "Mexico",
    "ES": "Spain", "IT": "Italy", "NL": "Netherlands", "SE": "Sweden",
    "EU": "European Union",
    "BF": "Burkina Faso",
    "CD": "Congo", "CF": "Central African Republic", "MZ": "Mozambique",
    "VE": "Venezuela", "IR": "Iran", "KP": "North Korea", "CU": "Cuba",
    "BY": "Belarus", "LB": "Lebanon", "HT": "Haiti",
}


# ---------------------------------------------------------------------------
# UCDP GED API client (free, no key required)
# ---------------------------------------------------------------------------

_UCDP_BASE = "https://ucdpapi.pcr.uu.se/api/gedevents/24.0.10"


def _fetch_ucdp_events(
    country_iso2: str,
    days: int = 365,
    api_key: str = "",
) -> list[dict[str, Any]]:
    """Fetch recent conflict events from UCDP GED API.

    Since 2025, the UCDP API requires authentication. If a
    ``UCDP_API_KEY`` is provided (via .env or environment variable),
    it is sent as a Bearer token. Otherwise, the API returns 401
    and we fall back to static lists, which still provide reliable
    baseline conflict classification for all countries.

    Parameters
    ----------
    country_iso2:
        ISO-2 country code.
    days:
        How many days back to search.
    api_key:
        Optional UCDP API key for authenticated access.

    Returns
    -------
    List of UCDP event dicts with fields like type_of_violence,
    best_est (fatalities), date_start, etc.
    """
    ucdp_id = _ISO2_TO_UCDP_ID.get(country_iso2)
    if ucdp_id is None:
        logger.debug("No UCDP ID for country %s", country_iso2)
        return []

    start_date = (date.today() - timedelta(days=days)).isoformat()

    headers: dict[str, str] = {"Accept": "application/json"}
    if api_key:
        headers["Authorization"] = f"Bearer {api_key}"

    try:
        resp = requests.get(
            _UCDP_BASE,
            params={
                "Country": str(ucdp_id),
                "StartDate": start_date,
                "pagesize": 500,
            },
            headers=headers,
            timeout=15,
        )
        if resp.status_code == 401:
            if api_key:
                logger.warning(
                    "UCDP API rejected API key (401). Check UCDP_API_KEY. "
                    "Using static conflict lists as fallback."
                )
            else:
                logger.info(
                    "UCDP API requires authentication (401). "
                    "Set UCDP_API_KEY in .env for real-time conflict data. "
                    "Using static conflict lists as fallback."
                )
            return []
        resp.raise_for_status()
        data = resp.json()
        return data.get("Result", [])
    except Exception as exc:
        logger.warning("UCDP API failed for %s: %s", country_iso2, exc)
        return []


def _analyze_ucdp_events(events: list[dict]) -> dict[str, Any]:
    """Analyze UCDP events to compute conflict metrics.

    Returns dict with: events_30d, events_90d, fatalities_30d,
    fatalities_90d, conflict_type, trend.
    """
    if not events:
        return {
            "events_30d": 0, "events_90d": 0,
            "fatalities_30d": 0, "fatalities_90d": 0,
            "conflict_type": "none", "trend": "stable",
        }

    today = date.today()
    cutoff_30 = today - timedelta(days=30)
    cutoff_90 = today - timedelta(days=90)
    cutoff_prev_90 = today - timedelta(days=180)

    events_30d = 0
    events_90d = 0
    events_prev_90d = 0
    fatalities_30d = 0
    fatalities_90d = 0
    violence_types = set()

    for event in events:
        event_date_str = event.get("date_start", "")
        if not event_date_str:
            continue

        try:
            event_date = date.fromisoformat(event_date_str[:10])
        except (ValueError, TypeError):
            continue

        fatalities = int(event.get("best", event.get("best_est", 0)) or 0)
        violence_type = event.get("type_of_violence", 0)
        violence_types.add(violence_type)

        if event_date >= cutoff_30:
            events_30d += 1
            fatalities_30d += fatalities
        if event_date >= cutoff_90:
            events_90d += 1
            fatalities_90d += fatalities
        elif event_date >= cutoff_prev_90:
            events_prev_90d += 1

    # Classify conflict type based on UCDP violence types:
    # 1 = state-based, 2 = non-state, 3 = one-sided
    if 1 in violence_types:
        conflict_type = "interstate_war" if events_90d > 50 else "civil_war"
    elif 2 in violence_types or 3 in violence_types:
        conflict_type = "low_intensity"
    else:
        conflict_type = "none" if events_90d == 0 else "low_intensity"

    # Trend: compare last 90 days vs previous 90 days
    if events_prev_90d > 0:
        ratio = events_90d / events_prev_90d
        if ratio > 1.5:
            trend = "escalating"
        elif ratio < 0.5:
            trend = "de-escalating"
        else:
            trend = "stable"
    else:
        trend = "escalating" if events_90d > 10 else "stable"

    return {
        "events_30d": events_30d,
        "events_90d": events_90d,
        "fatalities_30d": fatalities_30d,
        "fatalities_90d": fatalities_90d,
        "conflict_type": conflict_type,
        "trend": trend,
    }


# ---------------------------------------------------------------------------
# GDELT news-based conflict monitoring (free, no key)
# ---------------------------------------------------------------------------

_GDELT_BASE = "https://api.gdeltproject.org/api/v2/doc/doc"

# Rate limiting: GDELT has aggressive rate limits (~1 req/5s for free tier)
_gdelt_last_call: float = 0.0
_GDELT_MIN_INTERVAL: float = 5.0  # seconds between calls


def _fetch_gdelt_conflict_news(
    country_name: str,
    days: int = 7,
) -> dict[str, Any]:
    """Query GDELT for recent conflict/war news about a country.

    Includes rate limiting (5s between calls) to avoid GDELT's 429 responses.

    Returns dict with: mentions_count, avg_tone.
    """
    global _gdelt_last_call

    if not country_name:
        return {"mentions_count": 0, "avg_tone": 0.0}

    # Rate limiting
    import time
    now = time.time()
    elapsed = now - _gdelt_last_call
    if elapsed < _GDELT_MIN_INTERVAL:
        sleep_time = _GDELT_MIN_INTERVAL - elapsed
        logger.debug("GDELT rate limit: sleeping %.1fs", sleep_time)
        time.sleep(sleep_time)

    query = f'"{country_name}" (war OR conflict OR military OR bombing OR attack)'

    try:
        resp = requests.get(
            _GDELT_BASE,
            params={
                "query": query,
                "mode": "ArtList",
                "maxrecords": 50,
                "format": "json",
                "timespan": f"{days}days",
            },
            headers={"User-Agent": "Operator1/1.0"},
            timeout=15,
        )
        _gdelt_last_call = time.time()

        if resp.status_code == 429:
            logger.info("GDELT rate limited (429). Skipping news data.")
            return {"mentions_count": 0, "avg_tone": 0.0}

        resp.raise_for_status()
        data = resp.json()
    except Exception as exc:
        logger.debug("GDELT query failed for %s: %s", country_name, exc)
        return {"mentions_count": 0, "avg_tone": 0.0}

    articles = data.get("articles", [])
    if not articles:
        return {"mentions_count": 0, "avg_tone": 0.0}

    # GDELT ArtList mode provides tone as a float in each article.
    # Negative = negative sentiment, positive = positive sentiment.
    # If tone is missing, estimate from title keywords.
    tones = []
    negative_keywords = {"war", "attack", "kill", "bomb", "death", "destroy",
                         "invasion", "missile", "casualt", "strike", "combat"}
    for article in articles:
        tone = article.get("tone")
        if tone is not None and isinstance(tone, (int, float)):
            tones.append(float(tone))
        else:
            # Estimate tone from title keywords
            title = (article.get("title") or "").lower()
            neg_count = sum(1 for kw in negative_keywords if kw in title)
            if neg_count > 0:
                tones.append(-2.0 * neg_count)
            else:
                tones.append(0.0)

    avg_tone = sum(tones) / len(tones) if tones else 0.0

    return {
        "mentions_count": len(articles),
        "avg_tone": avg_tone,
    }


# ---------------------------------------------------------------------------
# Conflict intensity scoring
# ---------------------------------------------------------------------------

def _compute_intensity_score(
    events_30d: int,
    events_90d: int,
    fatalities_30d: int,
    is_active_war: bool,
    is_sanctioned: bool,
    is_fragile: bool,
    news_mentions: int,
) -> float:
    """Compute a 0-1 conflict intensity score from multiple signals.

    Weighting:
      - UCDP events (40%): based on event count relative to thresholds
      - Fatalities (20%): log-scaled fatality count
      - Static flags (25%): active war (15%), sanctions (5%), fragile (5%)
      - News intensity (15%): GDELT mention count

    Returns float in [0.0, 1.0].
    """
    import math

    # Event intensity: 0 events = 0, 10+ events/30d = 0.5, 50+ = 0.8, 200+ = 1.0
    if events_30d == 0:
        event_score = 0.0
    elif events_30d < 10:
        event_score = events_30d / 20.0
    elif events_30d < 50:
        event_score = 0.5 + (events_30d - 10) / 80.0
    else:
        event_score = min(0.8 + (events_30d - 50) / 750.0, 1.0)

    # Fatality intensity: log-scaled
    if fatalities_30d == 0:
        fatality_score = 0.0
    else:
        fatality_score = min(math.log10(fatalities_30d + 1) / 4.0, 1.0)

    # Static flags
    flag_score = 0.0
    if is_active_war:
        flag_score += 0.6
    if is_sanctioned:
        flag_score += 0.2
    if is_fragile:
        flag_score += 0.2
    flag_score = min(flag_score, 1.0)

    # News intensity
    if news_mentions == 0:
        news_score = 0.0
    elif news_mentions < 10:
        news_score = news_mentions / 20.0
    else:
        news_score = min(0.5 + (news_mentions - 10) / 80.0, 1.0)

    # Weighted combination
    intensity = (
        0.40 * event_score
        + 0.20 * fatality_score
        + 0.25 * flag_score
        + 0.15 * news_score
    )

    return round(min(max(intensity, 0.0), 1.0), 3)


# ---------------------------------------------------------------------------
# Main entry point
# ---------------------------------------------------------------------------

def assess_conflict_risk(
    country_iso2: str,
    company_name: str | None = None,
    skip_ucdp: bool = False,
    skip_gdelt: bool = False,
    ucdp_api_key: str = "",
) -> ConflictRiskResult:
    """Assess war/conflict risk for a country and optionally a company.

    Parameters
    ----------
    country_iso2:
        ISO-2 country code (e.g. 'UA', 'US', 'IL').
    company_name:
        Optional company name for company-level conflict assessment.
    skip_ucdp:
        If True, skip the UCDP API call (use static lists only).
    skip_gdelt:
        If True, skip the GDELT API call.

    Returns
    -------
    ConflictRiskResult with all conflict risk fields populated.
    """
    result = ConflictRiskResult(
        country_iso2=country_iso2,
        assessment_date=date.today().isoformat(),
    )

    cc = country_iso2.upper()

    # ------------------------------------------------------------------
    # 1. Static list checks (always available, no API calls)
    # ------------------------------------------------------------------
    result.fragile_state_flag = cc in FRAGILE_CONFLICT_STATES
    result.sanctions_flag = cc in SANCTIONED_COUNTRIES
    is_active_war = cc in ACTIVE_WAR_COUNTRIES
    result.data_sources_used.append("static_lists")

    if result.fragile_state_flag:
        logger.info("Country %s is on World Bank FCS list", cc)
    if result.sanctions_flag:
        logger.info("Country %s is under major international sanctions", cc)
    if is_active_war:
        logger.info("Country %s has active armed conflict", cc)

    # ------------------------------------------------------------------
    # 2. UCDP event data (free, no key)
    # ------------------------------------------------------------------
    events_30d = 0
    events_90d = 0
    fatalities_30d = 0
    fatalities_90d = 0
    ucdp_conflict_type = "none"
    ucdp_trend = "stable"

    # Auto-detect UCDP key from environment if not provided
    if not ucdp_api_key:
        import os
        ucdp_api_key = os.environ.get("UCDP_API_KEY", "")

    if not skip_ucdp:
        ucdp_events = _fetch_ucdp_events(cc, days=365, api_key=ucdp_api_key)
        if ucdp_events:
            analysis = _analyze_ucdp_events(ucdp_events)
            events_30d = analysis["events_30d"]
            events_90d = analysis["events_90d"]
            fatalities_30d = analysis["fatalities_30d"]
            fatalities_90d = analysis["fatalities_90d"]
            ucdp_conflict_type = analysis["conflict_type"]
            ucdp_trend = analysis["trend"]
            result.data_sources_used.append("ucdp_ged")

    result.recent_events_30d = events_30d
    result.recent_events_90d = events_90d
    result.recent_fatalities_30d = fatalities_30d
    result.recent_fatalities_90d = fatalities_90d
    result.conflict_trend = ucdp_trend

    # ------------------------------------------------------------------
    # 3. GDELT news monitoring (free, no key)
    # ------------------------------------------------------------------
    news_mentions = 0
    news_tone = 0.0

    if not skip_gdelt:
        country_name = _ISO2_TO_NAME.get(cc, "")
        if country_name:
            gdelt = _fetch_gdelt_conflict_news(country_name, days=7)
            news_mentions = gdelt["mentions_count"]
            news_tone = gdelt["avg_tone"]
            result.data_sources_used.append("gdelt")

    result.news_conflict_mentions_7d = news_mentions
    result.news_conflict_tone = news_tone

    # ------------------------------------------------------------------
    # 4. Compute conflict intensity score
    # ------------------------------------------------------------------
    result.conflict_intensity_score = _compute_intensity_score(
        events_30d=events_30d,
        events_90d=events_90d,
        fatalities_30d=fatalities_30d,
        is_active_war=is_active_war,
        is_sanctioned=result.sanctions_flag,
        is_fragile=result.fragile_state_flag,
        news_mentions=news_mentions,
    )

    # ------------------------------------------------------------------
    # 5. Set flags and classify
    # ------------------------------------------------------------------
    # Country conflict flag: intensity > 0.3 OR active war OR sanctions OR recent events
    result.country_conflict_flag = (
        result.conflict_intensity_score > 0.3
        or is_active_war
        or result.sanctions_flag
        or events_30d > 5
    )

    # Conflict type: use UCDP classification if available, else infer
    if ucdp_conflict_type != "none":
        result.conflict_type = ucdp_conflict_type
    elif is_active_war:
        result.conflict_type = "interstate_war"
    elif result.sanctions_flag:
        result.conflict_type = "sanctions"
    elif result.fragile_state_flag and events_90d > 0:
        result.conflict_type = "low_intensity"
    else:
        result.conflict_type = "none"

    # Company conflict flag: if the company is in a conflict-affected country
    # Future enhancement: check if company operations/revenue are in conflict zones
    if company_name and result.country_conflict_flag:
        result.company_conflict_flag = True

    # ------------------------------------------------------------------
    # 6. Confidence score
    # ------------------------------------------------------------------
    # Based on how many data sources contributed
    source_count = len(result.data_sources_used)
    if source_count >= 3:
        result.confidence = 0.9
    elif source_count >= 2:
        result.confidence = 0.7
    else:
        result.confidence = 0.5

    logger.info(
        "Conflict risk for %s: intensity=%.2f, type=%s, flag=%s, sources=%s",
        cc, result.conflict_intensity_score, result.conflict_type,
        result.country_conflict_flag, result.data_sources_used,
    )

    return result


# ---------------------------------------------------------------------------
# Cache injection helper
# ---------------------------------------------------------------------------

def assess_linked_entity_conflict(
    linked_entities: dict[str, list[dict[str, Any]]],
    target_conflict: ConflictRiskResult,
) -> dict[str, Any]:
    """Assess conflict risk propagation from linked entities.

    For each linked entity group (suppliers, customers, competitors, etc.),
    checks whether entities operate in conflict zones and computes the
    impact on the target company.

    Logic:
      - Supplier in conflict zone -> RISK: supply chain disruption
      - Customer in conflict zone -> RISK: revenue loss
      - Competitor in conflict zone -> BENEFIT: competitive advantage
      - Financial institution in conflict zone -> RISK: credit/funding risk

    Parameters
    ----------
    linked_entities:
        Dict from entity discovery: {group_name: [entity_dict, ...]}.
        Each entity dict should have at minimum 'country' (ISO-2).
    target_conflict:
        The target company's own ConflictRiskResult.

    Returns
    -------
    Dict with linked entity conflict analysis:
      - supply_chain_risk_score (0-1)
      - revenue_exposure_score (0-1)
      - competitive_advantage_score (0-1)
      - linked_entities_in_conflict (list of affected entities)
      - linked_conflict_summary (human-readable text)
    """
    if not linked_entities:
        return {
            "supply_chain_risk_score": 0.0,
            "revenue_exposure_score": 0.0,
            "competitive_advantage_score": 0.0,
            "linked_entities_in_conflict": [],
            "linked_conflict_summary": "No linked entities available for conflict analysis.",
        }

    # Risk weights by relationship group
    _GROUP_RISK_TYPE = {
        "suppliers": "supply_chain",
        "customers": "revenue",
        "competitors": "competitive_advantage",
        "financial_institutions": "credit",
        "logistics": "supply_chain",
        "regulators": "regulatory",
    }

    supply_chain_risks = []
    revenue_risks = []
    competitive_advantages = []
    affected_entities = []

    for group_name, entities in linked_entities.items():
        risk_type = _GROUP_RISK_TYPE.get(group_name, "other")

        for entity in entities:
            entity_country = (entity.get("country") or "").upper()
            entity_name = entity.get("name", entity.get("ticker", "unknown"))

            if not entity_country:
                continue

            # Quick conflict check using static lists (no API calls)
            is_war = entity_country in ACTIVE_WAR_COUNTRIES
            is_sanctioned = entity_country in SANCTIONED_COUNTRIES
            is_fragile = entity_country in FRAGILE_CONFLICT_STATES
            is_affected = is_war or is_sanctioned or is_fragile

            if not is_affected:
                continue

            # Determine severity
            if is_war:
                severity = 1.0
                reason = "active armed conflict"
            elif is_sanctioned:
                severity = 0.7
                reason = "international sanctions"
            else:
                severity = 0.3
                reason = "fragile state"

            affected_entities.append({
                "name": entity_name,
                "country": entity_country,
                "group": group_name,
                "risk_type": risk_type,
                "severity": severity,
                "reason": reason,
            })

            if risk_type == "supply_chain":
                supply_chain_risks.append(severity)
            elif risk_type == "revenue":
                revenue_risks.append(severity)
            elif risk_type == "competitive_advantage":
                competitive_advantages.append(severity)
            elif risk_type == "credit":
                supply_chain_risks.append(severity * 0.5)  # credit risk is indirect

    # Compute aggregate scores (max severity across affected entities)
    supply_chain_score = max(supply_chain_risks) if supply_chain_risks else 0.0
    revenue_score = max(revenue_risks) if revenue_risks else 0.0
    competitive_score = max(competitive_advantages) if competitive_advantages else 0.0

    # Build summary
    summary_parts = []
    if supply_chain_risks:
        count = len(supply_chain_risks)
        summary_parts.append(
            f"{count} supplier/logistics {'partner' if count == 1 else 'partners'} "
            f"in conflict zones (supply chain risk: {supply_chain_score:.0%})"
        )
    if revenue_risks:
        count = len(revenue_risks)
        summary_parts.append(
            f"{count} {'customer' if count == 1 else 'customers'} "
            f"in conflict zones (revenue exposure: {revenue_score:.0%})"
        )
    if competitive_advantages:
        count = len(competitive_advantages)
        summary_parts.append(
            f"{count} {'competitor' if count == 1 else 'competitors'} "
            f"in conflict zones (potential competitive advantage)"
        )

    summary = "; ".join(summary_parts) if summary_parts else "No linked entities in conflict zones."

    logger.info(
        "Linked entity conflict: %d affected (%d supply chain, %d revenue, %d competitors)",
        len(affected_entities), len(supply_chain_risks), len(revenue_risks),
        len(competitive_advantages),
    )

    return {
        "supply_chain_risk_score": round(supply_chain_score, 3),
        "revenue_exposure_score": round(revenue_score, 3),
        "competitive_advantage_score": round(competitive_score, 3),
        "linked_entities_in_conflict": affected_entities,
        "linked_conflict_summary": summary,
    }


# ---------------------------------------------------------------------------
# Cache injection helper
# ---------------------------------------------------------------------------

def inject_conflict_risk_into_cache(
    cache: "pd.DataFrame",
    conflict_result: ConflictRiskResult,
    linked_conflict: dict[str, Any] | None = None,
) -> "pd.DataFrame":
    """Add conflict risk columns to the daily cache DataFrame.

    Parameters
    ----------
    cache:
        Daily cache DataFrame (with DatetimeIndex).
    conflict_result:
        Result from ``assess_conflict_risk()``.

    Returns
    -------
    cache with added columns: country_conflict_flag, company_conflict_flag,
    conflict_intensity_score, sanctions_flag.
    """
    cache["country_conflict_flag"] = int(conflict_result.country_conflict_flag)
    cache["company_conflict_flag"] = int(conflict_result.company_conflict_flag)
    cache["conflict_intensity_score"] = conflict_result.conflict_intensity_score
    cache["sanctions_flag"] = int(conflict_result.sanctions_flag)
    cache["fragile_state_flag"] = int(conflict_result.fragile_state_flag)
    cache["conflict_type"] = conflict_result.conflict_type

    # Linked entity conflict propagation columns
    if linked_conflict is not None:
        cache["supply_chain_risk_score"] = linked_conflict.get("supply_chain_risk_score", 0.0)
        cache["revenue_exposure_score"] = linked_conflict.get("revenue_exposure_score", 0.0)
        cache["competitive_advantage_score"] = linked_conflict.get("competitive_advantage_score", 0.0)

    # Apply time-varying conflict flags for historical accuracy
    cache = _apply_time_varying_conflict(cache, conflict_result)

    return cache


def _apply_time_varying_conflict(
    cache: "pd.DataFrame",
    conflict_result: ConflictRiskResult,
) -> "pd.DataFrame":
    """Set conflict flags to 0 before the conflict start date.

    For countries with known conflict start dates, the daily cache
    should show flag=0 (peaceful) before the war started, not the
    current conflict status for the entire 2-year window.

    This gives temporal models accurate historical context:
    a company in Ukraine was NOT in survival mode before Feb 2022.
    """
    import pandas as pd

    cc = conflict_result.country_iso2.upper()
    start_str = CONFLICT_START_DATES.get(cc)

    if not start_str or not conflict_result.country_conflict_flag:
        return cache

    try:
        start_ts = pd.Timestamp(start_str)
    except Exception:
        return cache

    if not hasattr(cache.index, 'dtype') or cache.index.empty:
        return cache

    pre_conflict = cache.index < start_ts

    if not pre_conflict.any():
        return cache  # all dates are after the conflict start

    # Zero out conflict flags for pre-conflict period
    conflict_cols = [
        "country_conflict_flag", "company_conflict_flag",
        "conflict_intensity_score",
    ]
    for col in conflict_cols:
        if col in cache.columns:
            cache.loc[pre_conflict, col] = 0

    # Keep sanctions and fragile state flags unchanged (they may predate the war)

    n_pre = pre_conflict.sum()
    n_post = (~pre_conflict).sum()
    logger.info(
        "Time-varying conflict for %s: %d days peaceful (before %s), %d days conflict",
        cc, n_pre, start_str, n_post,
    )

    return cache
