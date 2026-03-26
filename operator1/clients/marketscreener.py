"""MarketScreener/Zonebourse shareholder data scraper.

Fetches institutional shareholder data for European (and global) stocks
from MarketScreener (English) / Zonebourse (French) -- the same site
with different language frontends.

Data available:
  - Institutional holder names (Amundi, BlackRock, Vanguard, etc.)
  - Number of shares held
  - Percentage of outstanding shares
  - Value of holdings

Slug resolution uses DuckDuckGo HTML search (no API key needed) to
find the MarketScreener stock page URL from a company name.

No authentication required.  Data is embedded in server-rendered HTML
behind a CSS login overlay but fully parseable without login.

Usage:
    from operator1.clients.marketscreener import fetch_shareholders
    holders = fetch_shareholders("BNP Paribas")
"""

from __future__ import annotations

import logging
import re
import time
from typing import Any

logger = logging.getLogger(__name__)

_ZB_BASE = "https://www.zonebourse.com"
_MS_BASE = "https://www.marketscreener.com"

# Cache resolved slugs to avoid repeated DuckDuckGo searches
_slug_cache: dict[str, str] = {}


# ---------------------------------------------------------------------------
# Slug resolution via DuckDuckGo
# ---------------------------------------------------------------------------


def _resolve_slug(company_name: str) -> str:
    """Resolve a company name to a MarketScreener/Zonebourse stock slug.

    Uses DuckDuckGo HTML search with site: filter to find the stock page.
    The slug format is ``{COMPANY-NAME}-{ID}`` (e.g., ``BNP-PARIBAS-4618``).

    Parameters
    ----------
    company_name:
        Company name (e.g., "BNP Paribas", "TotalEnergies", "Siemens AG").

    Returns
    -------
    Slug string or empty string if not found.
    """
    if company_name in _slug_cache:
        return _slug_cache[company_name]

    try:
        from curl_cffi import requests as cf_requests
        s = cf_requests.Session(impersonate="chrome")

        # Search DuckDuckGo for the MarketScreener stock page
        r = s.get(
            "https://html.duckduckgo.com/html/",
            params={"q": f'site:marketscreener.com/quote/stock "{company_name}" shareholders'},
            timeout=15,
        )
        if r.ok:
            slugs = re.findall(
                r'marketscreener\.com/quote/stock/([A-Z0-9-]+-\d+)/',
                r.text,
            )
            if slugs:
                slug = slugs[0]
                _slug_cache[company_name] = slug
                logger.debug("MarketScreener slug for '%s': %s", company_name, slug)
                return slug

        # Fallback: try Zonebourse variant
        time.sleep(0.5)
        r2 = s.get(
            "https://html.duckduckgo.com/html/",
            params={"q": f'site:zonebourse.com/cours/action "{company_name}" societe'},
            timeout=15,
        )
        if r2.ok:
            slugs2 = re.findall(
                r'zonebourse\.com/cours/action/([A-Z0-9-]+-\d+)/',
                r2.text,
            )
            if slugs2:
                slug = slugs2[0]
                _slug_cache[company_name] = slug
                logger.debug("Zonebourse slug for '%s': %s", company_name, slug)
                return slug

    except ImportError:
        logger.debug("curl_cffi not available for MarketScreener slug resolution")
    except Exception as exc:
        logger.debug("MarketScreener slug resolution failed for '%s': %s", company_name, exc)

    return ""


# ---------------------------------------------------------------------------
# Shareholder data extraction
# ---------------------------------------------------------------------------


def _parse_number(text: str) -> int:
    """Parse a number string with non-breaking spaces and French formatting."""
    # Remove all whitespace/non-breaking space variants
    clean = re.sub(r'[\s\u00a0\u202f\u2009]', '', text.strip())
    # Replace comma decimal separator with dot
    clean = clean.replace(',', '.')
    try:
        return int(float(clean))
    except (ValueError, TypeError):
        return 0


def _parse_percentage(text: str) -> float:
    """Parse a percentage string (French format: 5,661 means 5.661%)."""
    clean = text.strip().replace('%', '').replace(',', '.').replace(' ', '')
    try:
        return round(float(clean), 4)
    except (ValueError, TypeError):
        return 0.0


def fetch_shareholders(
    company_name: str,
    slug: str = "",
) -> list[dict[str, Any]]:
    """Fetch institutional shareholders from MarketScreener/Zonebourse.

    Parameters
    ----------
    company_name:
        Company name for slug resolution (ignored if slug is provided).
    slug:
        Direct MarketScreener slug (e.g., "BNP-PARIBAS-4618").
        If empty, resolved via DuckDuckGo search.

    Returns
    -------
    List of dicts with: name, shares, percentage, holder_type,
    date_reported, source.
    """
    if not slug:
        slug = _resolve_slug(company_name)
    if not slug:
        logger.debug("Could not resolve MarketScreener slug for '%s'", company_name)
        return []

    holders: list[dict[str, Any]] = []

    try:
        from curl_cffi import requests as cf_requests
        s = cf_requests.Session(impersonate="chrome")

        # Fetch the company/societe page (has shareholder table)
        url = f"{_ZB_BASE}/cours/action/{slug}/societe/"
        r = s.get(url, timeout=20)

        if not r.ok:
            # Try MarketScreener English version
            url = f"{_MS_BASE}/quote/stock/{slug}/company/"
            r = s.get(url, timeout=20)

        if not r.ok:
            logger.debug("MarketScreener page not accessible for %s", slug)
            return []

        text = r.text

        # Parse shareholders-table sections
        # Each section has: data-collapse-group="shareholders-table"
        # followed by entity name, shares count, percentage, value
        for m in re.finditer(r'data-collapse-group="shareholders-table"', text):
            chunk = text[m.start():m.start() + 3000]

            # Extract holder name
            name_match = re.search(
                r'>([^<]+(?:Management|Investment|Asset|Bank|Capital|Group|Fund|'
                r'SA|SAS|SASU|Ltd|Inc|PLC|NV|AG|GmbH|LLC|LP|Trust|'
                r'Advisors|Partners|Holdings|Securities|Financial|'
                r'Insurance|Pension|Plan|State|Government|Employee)[^<]*)<',
                chunk, re.I,
            )
            if not name_match:
                # Try simpler: first link text
                name_match = re.search(r'<a[^>]+href="[^"]*shareholders[^"]*"[^>]*>([^<]{5,})</a>', chunk)
            if not name_match:
                continue

            name = name_match.group(1).strip()
            if not name or len(name) < 3:
                continue
            # Reject CSS selectors, HTML fragments, and other garbage
            if any(c in name for c in ('{', '}', '<', '>', 'not(', '.c-')):
                continue
            # Reject strings that start with special chars
            if name[0] in '.(#[':
                continue
            # Reject French/English labels that aren't holder names
            if name.lower().startswith(("liste des", "list of", "shareholders of")):
                continue

            # Extract numbers (shares counts)
            numbers = re.findall(r'>\s*([\d\u00a0\u202f\u2009\s,.]+)\s*<', chunk)
            clean_nums = []
            for n in numbers:
                n = n.strip()
                if n and any(c.isdigit() for c in n) and len(n) > 1:
                    clean_nums.append(n)

            # Extract percentages
            pcts = re.findall(r'([\d,.]+)\s*%', chunk)

            # Build holder record
            shares = _parse_number(clean_nums[0]) if clean_nums else 0
            percentage = _parse_percentage(pcts[0]) if pcts else 0.0

            if name and (shares > 0 or percentage > 0):
                holders.append({
                    "name": name,
                    "shares": shares,
                    "value": 0.0,
                    "percentage": percentage,
                    "holder_type": "institutional",
                    "date_reported": "",
                    "source": "marketscreener",
                })

        if holders:
            logger.info(
                "MarketScreener holders for %s (%s): %d found",
                company_name, slug, len(holders),
            )

    except ImportError:
        logger.debug("curl_cffi not available for MarketScreener")
    except Exception as exc:
        logger.debug("MarketScreener shareholder fetch failed for %s: %s", slug, exc)

    return holders
