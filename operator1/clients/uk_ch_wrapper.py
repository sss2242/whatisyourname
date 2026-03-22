"""UK Companies House PIT client -- wrapper with disk caching.

Uses the official Companies House REST API (free, key required) as
the primary source, with enhanced disk caching, iXBRL document
parsing for financial data, and scraper-based fallback.

Supports two authentication modes:

1. **API Key (Basic Auth)** -- default, suitable for most REST API
   endpoints (company search, filing history, iXBRL documents).
   Set via ``COMPANIES_HOUSE_API_KEY`` env var.

2. **OAuth2 (Authorization Code)** -- for user-delegated access to
   protected endpoints (e.g., user profile, filing on behalf of).
   Requires ``CH_OAUTH_CLIENT_ID``, ``CH_OAUTH_CLIENT_SECRET``, and
   ``CH_OAUTH_REDIRECT_URI`` env vars.  Tokens are cached to disk
   and refreshed automatically.

Primary: Companies House REST API (https://api.company-information.service.gov.uk)
Fallback: Website scraping via requests + BeautifulSoup

Coverage: ~4,000+ listed companies on LSE, $3.18T market cap.
"""

from __future__ import annotations

import json
import logging
import os
import time
from datetime import date
from pathlib import Path
from typing import Any

import pandas as pd

from operator1.http_utils import cached_get, HTTPError

logger = logging.getLogger(__name__)

_CH_BASE = "https://api.company-information.service.gov.uk"
_CH_SANDBOX_BASE = "https://api-sandbox.company-information.service.gov.uk"
_CH_SANDBOX_TEST_DATA = "https://test-data-sandbox.company-information.service.gov.uk"

# OAuth2 endpoints (same for production and sandbox -- identity service is shared).
_CH_OAUTH_AUTHORIZE = "https://identity.company-information.service.gov.uk/oauth2/authorise"
_CH_OAUTH_TOKEN = "https://identity.company-information.service.gov.uk/oauth2/token"
_CH_OAUTH_SANDBOX_AUTHORIZE = "https://identity-sandbox.company-information.service.gov.uk/oauth2/authorise"
_CH_OAUTH_SANDBOX_TOKEN = "https://identity-sandbox.company-information.service.gov.uk/oauth2/token"

# Default OAuth2 scope for user profile access.
_CH_OAUTH_DEFAULT_SCOPE = "https://identity.company-information.service.gov.uk/user/profile.read"

_CACHE_DIR = Path("cache/uk_companies_house")
_OAUTH_TOKEN_CACHE = Path("cache/uk_companies_house/.oauth_token.json")


class UKCompaniesHouseError(Exception):
    def __init__(self, endpoint: str, detail: str = "") -> None:
        self.endpoint = endpoint
        self.detail = detail
        super().__init__(f"UK CH error on {endpoint}: {detail}")


# ---------------------------------------------------------------------------
# OAuth2 helper
# ---------------------------------------------------------------------------


class CHOAuth2Manager:
    """Manages Companies House OAuth2 Authorization Code flow.

    This handles token acquisition, caching, and automatic refresh for
    user-delegated API access.  The authorization code must be obtained
    separately (via browser redirect) and passed to ``exchange_code()``.

    Environment variables:
        ``CH_OAUTH_CLIENT_ID``       -- OAuth2 client ID
        ``CH_OAUTH_CLIENT_SECRET``   -- OAuth2 client secret
        ``CH_OAUTH_REDIRECT_URI``    -- Registered redirect URI

    Usage::

        oauth = CHOAuth2Manager(sandbox=True)
        # Step 1: Get the authorization URL and open it in a browser.
        auth_url = oauth.get_authorization_url(state="mystate123")
        # Step 2: User authorizes, gets redirected with ?code=...
        oauth.exchange_code("the_authorization_code")
        # Step 3: Use the access token for API calls.
        headers = oauth.get_auth_headers()
    """

    def __init__(
        self,
        client_id: str = "",
        client_secret: str = "",
        redirect_uri: str = "",
        scope: str = _CH_OAUTH_DEFAULT_SCOPE,
        sandbox: bool = False,
        token_cache_path: Path | str = _OAUTH_TOKEN_CACHE,
    ) -> None:
        self.client_id = client_id or os.environ.get("CH_OAUTH_CLIENT_ID", "")
        self.client_secret = client_secret or os.environ.get("CH_OAUTH_CLIENT_SECRET", "")
        self.redirect_uri = redirect_uri or os.environ.get("CH_OAUTH_REDIRECT_URI", "")
        self.scope = scope
        self.sandbox = sandbox
        self._token_cache_path = Path(token_cache_path)

        self._authorize_url = _CH_OAUTH_SANDBOX_AUTHORIZE if sandbox else _CH_OAUTH_AUTHORIZE
        self._token_url = _CH_OAUTH_SANDBOX_TOKEN if sandbox else _CH_OAUTH_TOKEN

        self._access_token: str = ""
        self._refresh_token: str = ""
        self._expires_at: float = 0.0

        # Try to load cached token.
        self._load_cached_token()

    @property
    def is_configured(self) -> bool:
        """True if OAuth2 credentials are available."""
        return bool(self.client_id and self.client_secret and self.redirect_uri)

    @property
    def has_valid_token(self) -> bool:
        """True if we have a non-expired access token."""
        return bool(self._access_token and time.time() < self._expires_at - 30)

    def get_authorization_url(self, state: str = "") -> str:
        """Build the authorization URL for the browser redirect.

        Parameters
        ----------
        state:
            Opaque state parameter for CSRF protection.

        Returns
        -------
        Full URL to redirect the user to for authorization.
        """
        from urllib.parse import urlencode

        params = {
            "client_id": self.client_id,
            "redirect_uri": self.redirect_uri,
            "response_type": "code",
            "scope": self.scope,
        }
        if state:
            params["state"] = state

        return f"{self._authorize_url}?{urlencode(params)}"

    def exchange_code(self, authorization_code: str) -> dict[str, Any]:
        """Exchange an authorization code for access and refresh tokens.

        Parameters
        ----------
        authorization_code:
            The code returned by the OAuth2 callback.

        Returns
        -------
        Token response dict with ``access_token``, ``refresh_token``,
        ``expires_in``, ``token_type``.

        Raises
        ------
        UKCompaniesHouseError
            If the token exchange fails.
        """
        import requests

        data = {
            "grant_type": "authorization_code",
            "code": authorization_code,
            "client_id": self.client_id,
            "client_secret": self.client_secret,
            "redirect_uri": self.redirect_uri,
        }

        try:
            resp = requests.post(
                self._token_url,
                data=data,
                headers={"Content-Type": "application/x-www-form-urlencoded"},
                timeout=30,
            )
            resp.raise_for_status()
            token_data = resp.json()
        except Exception as exc:
            raise UKCompaniesHouseError(
                "oauth2/token",
                f"Token exchange failed: {exc}",
            ) from exc

        self._store_token(token_data)
        logger.info(
            "OAuth2 token acquired (expires_in=%ds)",
            token_data.get("expires_in", 0),
        )
        return token_data

    def refresh_access_token(self) -> dict[str, Any]:
        """Refresh the access token using the stored refresh token.

        Returns
        -------
        Token response dict.

        Raises
        ------
        UKCompaniesHouseError
            If no refresh token is available or refresh fails.
        """
        if not self._refresh_token:
            raise UKCompaniesHouseError(
                "oauth2/token",
                "No refresh token available -- re-authorize required",
            )

        import requests

        data = {
            "grant_type": "refresh_token",
            "refresh_token": self._refresh_token,
            "client_id": self.client_id,
            "client_secret": self.client_secret,
        }

        try:
            resp = requests.post(
                self._token_url,
                data=data,
                headers={"Content-Type": "application/x-www-form-urlencoded"},
                timeout=30,
            )
            resp.raise_for_status()
            token_data = resp.json()
        except Exception as exc:
            raise UKCompaniesHouseError(
                "oauth2/token",
                f"Token refresh failed: {exc}",
            ) from exc

        self._store_token(token_data)
        logger.info("OAuth2 token refreshed")
        return token_data

    def get_auth_headers(self) -> dict[str, str]:
        """Get HTTP headers with a valid Bearer token.

        Automatically refreshes the token if expired and a refresh
        token is available.

        Returns
        -------
        Headers dict with ``Authorization: Bearer <token>``.

        Raises
        ------
        UKCompaniesHouseError
            If no valid token can be obtained.
        """
        if not self.has_valid_token and self._refresh_token:
            try:
                self.refresh_access_token()
            except UKCompaniesHouseError:
                logger.warning("OAuth2 token refresh failed -- re-authorization needed")

        if not self.has_valid_token:
            raise UKCompaniesHouseError(
                "oauth2",
                "No valid OAuth2 access token -- authorize first via "
                "get_authorization_url() + exchange_code()",
            )

        return {"Authorization": f"Bearer {self._access_token}"}

    def _store_token(self, token_data: dict[str, Any]) -> None:
        """Store token in memory and on disk."""
        self._access_token = token_data.get("access_token", "")
        self._refresh_token = token_data.get("refresh_token", self._refresh_token)
        expires_in = token_data.get("expires_in", 3600)
        self._expires_at = time.time() + expires_in

        # Persist to disk cache.
        cache_data = {
            "access_token": self._access_token,
            "refresh_token": self._refresh_token,
            "expires_at": self._expires_at,
        }
        try:
            self._token_cache_path.parent.mkdir(parents=True, exist_ok=True)
            self._token_cache_path.write_text(
                json.dumps(cache_data, indent=2),
                encoding="utf-8",
            )
        except Exception as exc:
            logger.debug("Failed to cache OAuth2 token: %s", exc)

    def _load_cached_token(self) -> None:
        """Load token from disk cache if available and not expired."""
        if not self._token_cache_path.exists():
            return
        try:
            data = json.loads(self._token_cache_path.read_text(encoding="utf-8"))
            expires_at = data.get("expires_at", 0)
            if time.time() < expires_at - 30:
                self._access_token = data.get("access_token", "")
                self._refresh_token = data.get("refresh_token", "")
                self._expires_at = expires_at
                logger.debug("Loaded cached OAuth2 token (expires in %.0fs)", expires_at - time.time())
            else:
                # Token expired but we may have a refresh token.
                self._refresh_token = data.get("refresh_token", "")
        except Exception:
            pass


class UKCompaniesHouseClient:
    """Point-in-time client for UK Companies House with disk caching.

    Implements the ``PITClient`` protocol. Uses the official REST API
    with enhanced caching and iXBRL document parsing.

    Supports two auth modes:

    1. **API Key (Basic Auth)** -- pass ``api_key`` or set
       ``COMPANIES_HOUSE_API_KEY``.  Works for all public REST endpoints.

    2. **OAuth2 (Bearer token)** -- pass ``oauth_manager`` or set
       ``CH_OAUTH_CLIENT_ID`` + ``CH_OAUTH_CLIENT_SECRET`` +
       ``CH_OAUTH_REDIRECT_URI``.  Required for user-delegated access
       (e.g., filing on behalf of a user, accessing user profile).
       OAuth takes priority over API key when both are available.

    Parameters
    ----------
    api_key:
        Companies House API key. Also loads from COMPANIES_HOUSE_API_KEY env var.
    cache_dir:
        Local cache directory.
    sandbox:
        Use sandbox API endpoints.
    oauth_manager:
        Pre-configured ``CHOAuth2Manager``.  If not provided but OAuth
        env vars are set, one is created automatically.
    """

    def __init__(
        self,
        api_key: str = "",
        cache_dir: Path | str = _CACHE_DIR,
        sandbox: bool = False,
        oauth_manager: CHOAuth2Manager | None = None,
    ) -> None:
        self._api_key = api_key or os.environ.get("COMPANIES_HOUSE_API_KEY", "")
        self._cache_dir = Path(cache_dir)
        self._sandbox = sandbox
        self._base_url = _CH_SANDBOX_BASE if sandbox else _CH_BASE

        # OAuth2 setup: use provided manager, or auto-create if env vars present.
        if oauth_manager is not None:
            self._oauth = oauth_manager
        elif os.environ.get("CH_OAUTH_CLIENT_ID"):
            self._oauth = CHOAuth2Manager(sandbox=sandbox)
            if self._oauth.is_configured:
                logger.info(
                    "OAuth2 manager auto-configured (sandbox=%s, has_token=%s)",
                    sandbox, self._oauth.has_valid_token,
                )
        else:
            self._oauth: CHOAuth2Manager | None = None

    def _cache_path(self, identifier: str, filename: str) -> Path:
        safe_id = identifier.replace("/", "_").replace("\\", "_").upper()
        return self._cache_dir / safe_id / filename

    def _read_cache(self, identifier: str, filename: str) -> dict | None:
        path = self._cache_path(identifier, filename)
        if not path.exists():
            return None
        try:
            age_days = (date.today() - date.fromtimestamp(path.stat().st_mtime)).days
            if filename == "profile.json" and age_days > 7:
                return None
            return json.loads(path.read_text(encoding="utf-8"))
        except Exception:
            return None

    def _write_cache(self, identifier: str, filename: str, data: dict) -> None:
        path = self._cache_path(identifier, filename)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(data, default=str, indent=2), encoding="utf-8")

    def _get(self, path: str, params: dict | None = None) -> Any:
        url = f"{self._base_url}{path}"
        headers = {"Accept": "application/json", "User-Agent": "Operator1/1.0"}

        # Auth priority: OAuth2 Bearer token > API key Basic Auth.
        if self._oauth is not None and self._oauth.is_configured:
            try:
                oauth_headers = self._oauth.get_auth_headers()
                headers.update(oauth_headers)
            except UKCompaniesHouseError:
                # OAuth token unavailable/expired -- fall back to API key.
                logger.debug("OAuth2 unavailable for %s, falling back to API key", path)
                if self._api_key:
                    import base64
                    encoded = base64.b64encode(f"{self._api_key}:".encode()).decode()
                    headers["Authorization"] = f"Basic {encoded}"
        elif self._api_key:
            import base64
            encoded = base64.b64encode(f"{self._api_key}:".encode()).decode()
            headers["Authorization"] = f"Basic {encoded}"

        try:
            return cached_get(url, params=params, headers=headers)
        except HTTPError as exc:
            raise UKCompaniesHouseError(path, str(exc)) from exc

    @property
    def market_id(self) -> str:
        return "uk_companies_house"

    @property
    def market_name(self) -> str:
        return "United Kingdom (LSE) -- Companies House"

    def list_companies(self, query: str = "") -> list[dict[str, Any]]:
        if not query:
            return []
        try:
            data = self._get("/search/companies", params={"q": query, "items_per_page": 20})
            items = data.get("items", []) if isinstance(data, dict) else []
            return [
                {
                    "ticker": item.get("company_number", ""),
                    "name": item.get("title", ""),
                    "cik": item.get("company_number", ""),
                    "exchange": "LSE",
                    "country": "GB",
                    "market_id": self.market_id,
                }
                for item in items
            ]
        except Exception:
            return []

    def search_company(self, name: str) -> list[dict[str, Any]]:
        return self.list_companies(query=name)

    def get_profile(self, identifier: str) -> dict[str, Any]:
        cached = self._read_cache(identifier, "profile.json")
        if cached:
            return cached

        try:
            data = self._get(f"/company/{identifier}")
            raw_profile = {
                "name": data.get("company_name", ""),
                "ticker": identifier,
                "isin": "",
                "country": "GB",
                "sector": data.get("type", ""),
                "industry": data.get("sic_codes", [""])[0] if data.get("sic_codes") else "",
                "exchange": "LSE",
                "currency": "GBP",
                "cik": identifier,
                "company_status": data.get("company_status", ""),
                "date_of_creation": data.get("date_of_creation", ""),
            }
        except Exception as exc:
            raise UKCompaniesHouseError("get_profile", str(exc)) from exc

        from operator1.clients.canonical_translator import translate_profile
        profile = translate_profile(raw_profile, self.market_id)
        self._write_cache(identifier, "profile.json", profile)
        return profile

    def get_income_statement(self, identifier: str) -> pd.DataFrame:
        return self._fetch_financials_from_filings(identifier, "income")

    def get_balance_sheet(self, identifier: str) -> pd.DataFrame:
        return self._fetch_financials_from_filings(identifier, "balance")

    def get_cashflow_statement(self, identifier: str) -> pd.DataFrame:
        return self._fetch_financials_from_filings(identifier, "cashflow")

    def _fetch_financials_from_filings(self, identifier: str, statement_type: str) -> pd.DataFrame:
        """Fetch financial data from Companies House filing history + iXBRL parsing.

        Strategy (Research Log: .roo/research/uk-companies-house-2026-02-24.md):
        1. Get filing history from REST API (dates, transaction IDs)
        2. For each filing with accounts, try to download and parse the iXBRL document
        3. Extract financial values using ixbrl-parse (if installed) or regex fallback
        4. Map iXBRL tags to canonical fields via _UKGAAP_MAP

        Gov API endpoints verified correct as of 2026-02-24:
        - GET /company/{id}/filing-history?category=accounts
        """
        try:
            data = self._get(
                f"/company/{identifier}/filing-history",
                params={"category": "accounts", "items_per_page": 20},
            )
            items = data.get("items", []) if isinstance(data, dict) else []
        except Exception:
            return pd.DataFrame()

        rows: list[dict] = []
        for item in items:
            filing_date = item.get("date", "")
            description = item.get("description", "")
            period_end = item.get("action_date", filing_date)
            transaction_id = item.get("transaction_id", "")

            if "accounts" not in description.lower() and "annual" not in description.lower():
                continue

            # Try to extract financial values from iXBRL document
            ixbrl_values = {}
            if transaction_id:
                ixbrl_values = self._extract_ixbrl_values(identifier, transaction_id)

            row: dict = {
                "filing_date": filing_date,
                "report_date": period_end,
                "form": item.get("type", ""),
                "description": description,
                "transaction_id": transaction_id,
                "period_type": "annual",
            }
            # Merge extracted financial values into the row
            row.update(ixbrl_values)
            rows.append(row)

        if not rows:
            return pd.DataFrame()

        df = pd.DataFrame(rows)
        for col in ("filing_date", "report_date"):
            if col in df.columns:
                df[col] = pd.to_datetime(df[col], errors="coerce")

        from operator1.clients.canonical_translator import translate_financials
        return translate_financials(df, self.market_id, statement_type)

    def _extract_ixbrl_values(self, identifier: str, transaction_id: str) -> dict[str, float]:
        """Download and parse an iXBRL document from Companies House.

        Uses the Document API (document-api.company-information.service.gov.uk)
        with content negotiation to request XHTML format when available.
        Falls back to the find-and-update URL if the metadata link is
        unavailable.

        Uses ixbrl-parse library (if installed) for structured extraction,
        with regex fallback for common UK-GAAP tags.

        Research: .roo/research/uk-ixbrl-parse-2026-02-25.md
        Library: ixbrl-parse v0.10.1 (MIT license, no API key needed)
        """
        values: dict[str, float] = {}

        # Download the document via the Document API (supports content negotiation).
        try:
            import requests
            import base64

            auth_headers: dict[str, str] = {"User-Agent": "Operator1/1.0"}
            if self._api_key:
                encoded = base64.b64encode(f"{self._api_key}:".encode()).decode()
                auth_headers["Authorization"] = f"Basic {encoded}"

            # Step 1: Get document metadata to find the content URL and
            # check if XHTML format is available.
            meta_url = ""
            try:
                filing_data = self._get(
                    f"/company/{identifier}/filing-history/{transaction_id}",
                )
                meta_url = filing_data.get("links", {}).get("document_metadata", "")
            except Exception:
                pass

            html_content = ""

            if meta_url:
                # Check metadata for available formats.
                meta_resp = requests.get(
                    meta_url,
                    headers={**auth_headers, "Accept": "application/json"},
                    timeout=30,
                )
                if meta_resp.status_code == 200:
                    resources = meta_resp.json().get("resources", {})
                    content_url = meta_resp.json().get("links", {}).get("document", "")

                    if "application/xhtml+xml" in resources and content_url:
                        # Request iXBRL XHTML format via content negotiation.
                        xhtml_resp = requests.get(
                            content_url,
                            headers={**auth_headers, "Accept": "application/xhtml+xml"},
                            timeout=30,
                            allow_redirects=True,
                        )
                        if xhtml_resp.status_code == 200 and "xhtml" in xhtml_resp.headers.get("Content-Type", ""):
                            html_content = xhtml_resp.text
                            logger.debug(
                                "Downloaded iXBRL XHTML for %s/%s (%d bytes)",
                                identifier, transaction_id, len(html_content),
                            )

            # Fallback: try the find-and-update URL (old approach).
            if not html_content:
                doc_url = f"https://find-and-update.company-information.service.gov.uk/company/{identifier}/filing-history/{transaction_id}/document"
                resp = requests.get(
                    doc_url,
                    headers={**auth_headers, "Accept": "application/xhtml+xml, text/html"},
                    timeout=30,
                    allow_redirects=True,
                )
                ct = resp.headers.get("Content-Type", "")
                if resp.status_code == 200 and ("html" in ct or "xhtml" in ct):
                    html_content = resp.text

            if not html_content:
                return values

        except Exception as exc:
            logger.debug("iXBRL download failed for %s/%s: %s", identifier, transaction_id, exc)
            return values

        # Try ixbrl-parse library first (verified API from research log)
        # Docs: https://github.com/cybermaggedon/ixbrl-parse/blob/master/README.md
        try:
            from lxml import etree as ET
            from ixbrl_parse.ixbrl import parse as ixbrl_parse
            import io

            # Strip XML declaration to avoid lxml "Unicode strings with
            # encoding declaration are not supported" error.
            clean_content = html_content
            if clean_content.startswith("<?xml"):
                end_decl = clean_content.find("?>")
                if end_decl != -1:
                    clean_content = clean_content[end_decl + 2:].lstrip()

            tree = ET.parse(io.StringIO(clean_content))
            ixbrl = ixbrl_parse(tree)

            from operator1.clients.canonical_translator import _UKGAAP_MAP, _IFRS_MAP
            combined_map = {**_UKGAAP_MAP, **_IFRS_MAP}

            # Access parsed values via ixbrl.values dict
            for val_id, val_obj in ixbrl.values.items():
                concept = getattr(val_obj, "name", "")
                raw_value = val_obj.to_value() if hasattr(val_obj, "to_value") else None

                if concept and raw_value is not None:
                    # Try full concept name and bare name
                    canonical = combined_map.get(concept) or combined_map.get(
                        concept.split(":")[-1] if ":" in concept else concept
                    )
                    if canonical:
                        try:
                            values[canonical] = float(raw_value)
                        except (ValueError, TypeError):
                            continue

            if values:
                logger.debug("ixbrl-parse extracted %d values from %s/%s", len(values), identifier, transaction_id)
                return values

        except ImportError:
            logger.debug("ixbrl-parse not installed; trying regex fallback for iXBRL")
        except Exception as exc:
            logger.debug("ixbrl-parse failed for %s/%s: %s", identifier, transaction_id, exc)

        # Regex fallback: extract ix:nonFraction elements from the HTML
        try:
            import re
            from operator1.clients.canonical_translator import _UKGAAP_MAP, _IFRS_MAP
            combined_map = {**_UKGAAP_MAP, **_IFRS_MAP}

            # Match ix:nonFraction elements: <ix:nonFraction name="uk-gaap:Turnover" ...>VALUE</ix:nonFraction>
            pattern = re.compile(
                r'<ix:nonFraction[^>]*name="([^"]+)"[^>]*>([\d,.\-()]+)</ix:nonFraction>',
                re.IGNORECASE,
            )
            for match in pattern.finditer(html_content):
                concept = match.group(1)
                raw_value = match.group(2).replace(",", "").strip()

                canonical = combined_map.get(concept)
                if not canonical:
                    bare = concept.split(":")[-1] if ":" in concept else concept
                    canonical = combined_map.get(bare)

                if canonical and raw_value:
                    try:
                        val = float(raw_value.replace("(", "-").replace(")", ""))
                        values[canonical] = val
                    except (ValueError, TypeError):
                        continue

            if values:
                logger.debug("Regex extracted %d iXBRL values from %s/%s", len(values), identifier, transaction_id)

        except Exception as exc:
            logger.debug("iXBRL regex fallback failed: %s", exc)

        return values

    def get_quotes(self, identifier: str) -> pd.DataFrame:
        return pd.DataFrame()

    def get_peers(self, identifier: str) -> list[str]:
        profile = self.get_profile(identifier)
        sic = profile.get("industry", "")
        if not sic:
            return []
        try:
            data = self._get("/advanced-search/companies", params={
                "sic_codes": sic, "size": 10, "company_status": "active",
            })
            items = data.get("items", []) if isinstance(data, dict) else []
            return [
                item.get("company_number", "")
                for item in items
                if item.get("company_number", "") != identifier
            ][:10]
        except Exception:
            return []

    def get_executives(self, identifier: str) -> list[dict[str, Any]]:
        try:
            data = self._get(f"/company/{identifier}/officers")
            items = data.get("items", []) if isinstance(data, dict) else []
            return [
                {
                    "name": item.get("name", ""),
                    "title": item.get("officer_role", ""),
                    "appointed_on": item.get("appointed_on", ""),
                }
                for item in items[:10]
            ]
        except Exception:
            return []

    def get_holders(self, identifier: str) -> list[dict[str, Any]]:
        """Fetch persons with significant control (>25% shares/voting rights).

        Uses the Companies House PSC endpoint which returns anyone with
        >25% shares, >25% voting rights, or significant influence.

        Returns list of dicts with: name, shares, percentage, holder_type,
        date_reported, natures_of_control.
        """
        try:
            data = self._get(f"/company/{identifier}/persons-with-significant-control")
            items = data.get("items", []) if isinstance(data, dict) else []
        except Exception as exc:
            logger.debug("UK PSC fetch failed for %s: %s", identifier, exc)
            return []

        holders: list[dict[str, Any]] = []
        for item in items:
            # Build name from name or name_elements
            name = item.get("name", "")
            if not name:
                elems = item.get("name_elements", {})
                forename = elems.get("forename", "")
                surname = elems.get("surname", "")
                name = f"{forename} {surname}".strip() if surname else ""

            natures = item.get("natures_of_control", [])
            pct = 0.0
            holder_type = "individual"

            for nature in natures:
                nl = nature.lower()
                if "corporate" in nl:
                    holder_type = "corporate"
                if "75-to-100" in nl:
                    pct = max(pct, 87.5)
                elif "50-to-75" in nl:
                    pct = max(pct, 62.5)
                elif "25-to-50" in nl:
                    pct = max(pct, 37.5)
                elif "significant-influence" in nl or "significant-control" in nl:
                    pct = max(pct, 25.0)

            if name:
                holders.append({
                    "name": name,
                    "shares": 0,  # PSC doesn't provide exact share counts
                    "percentage": round(pct, 2),
                    "holder_type": holder_type,
                    "date_reported": item.get("notified_on", ""),
                    "natures_of_control": natures,
                })

        if holders:
            logger.info("UK holders for %s: %d from Companies House PSC", identifier, len(holders))
        return holders
