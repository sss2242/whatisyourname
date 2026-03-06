"""Global constants for Operator 1 pipeline."""

from datetime import date, timedelta

# ---------------------------------------------------------------------------
# Date window -- 2-year lookback from today
# ---------------------------------------------------------------------------


def get_date_window(years: int = 2) -> tuple[date, date]:
    """Return (start, end) date window computed at call time.

    Use this instead of the module-level DATE_START/DATE_END constants
    when running in a long-lived process or service where the date may
    change between pipeline runs.
    """
    end = date.today()
    start = end - timedelta(days=365 * years)
    return start, end


# Module-level constants kept for backward compatibility.
# These are evaluated once at import time.
DATE_END: date = date.today()
DATE_START: date = DATE_END - timedelta(days=730)

# ---------------------------------------------------------------------------
# Numerical safety
# ---------------------------------------------------------------------------
EPSILON: float = 1e-9

# ---------------------------------------------------------------------------
# PIT API base URLs (Point-in-Time data sources)
# ---------------------------------------------------------------------------

# United States -- SEC EDGAR (free, no key)
SEC_EDGAR_BASE_URL: str = "https://data.sec.gov"
SEC_EDGAR_SEARCH_URL: str = "https://efts.sec.gov/LATEST"

# European Union -- ESEF / XBRL Europe (free, no key)
ESEF_XBRL_BASE_URL: str = "https://filings.xbrl.org/api"

# United Kingdom -- Companies House (free, key required)
COMPANIES_HOUSE_BASE_URL: str = "https://api.company-information.service.gov.uk"

# Japan -- EDINET (free, no key)
EDINET_BASE_URL: str = "https://api.edinet-fsa.go.jp/api/v2"

# South Korea -- DART (free, key required)
DART_BASE_URL: str = "https://opendart.fss.or.kr/api"

# Taiwan -- MOPS (free, no key)
MOPS_BASE_URL: str = "https://mops.twse.com.tw"

# Brazil -- CVM (free, no key)
CVM_BASE_URL: str = "https://dados.cvm.gov.br/api/v1"

# Chile -- CMF (free, no key)
CMF_BASE_URL: str = "https://www.cmfchile.cl"

# ---------------------------------------------------------------------------
# LLM base URLs (report generation, entity discovery, sentiment)
# ---------------------------------------------------------------------------
GEMINI_BASE_URL: str = "https://generativelanguage.googleapis.com/v1beta"
CLAUDE_BASE_URL: str = "https://api.anthropic.com/v1"

# ---------------------------------------------------------------------------
# Match scoring thresholds (entity discovery)
# ---------------------------------------------------------------------------
MATCH_SCORE_THRESHOLD: int = 70
SECTOR_PEER_FALLBACK_COUNT: int = 5

# ---------------------------------------------------------------------------
# Cache directory
# ---------------------------------------------------------------------------
CACHE_DIR: str = "cache"
RAW_CACHE_DIR: str = "cache/raw"
