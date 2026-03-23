"""Saudi Arabia Tadawul PIT client -- native XBRL extraction (no yfinance).

Primary data sources (all via curl_cffi Chrome impersonation to bypass Akamai WAF):

  1. Company directory: ThemeSearchUtilityServlet (1,949 companies, ISIN)
  2. Sector-enriched list: WPS portal indices-performance page (269 TASI companies
     with sectorName)
  3. Trade data: TickerServlet (live prices, volume, change for all stocks)
  4. Financial statements: XBRL HTML from /Resources/XBRL_DOCS/ -- structured
     IFRS tables (income, balance, cashflow) with PIT filing dates from the
     statementsTabData portal resource. NO LLM NEEDED.
  5. Filing PDFs: /Resources/fsPdf/ links (fallback for LLM/fuzzy extraction)

Coverage: ~269 main-market (TASI) + ~93 parallel-market (NOMU) companies,
~$2.7T market cap.

OHLCV: handled separately via ohlcv_provider.py (yfinance .SR suffix).
The TickerServlet provides current-day snapshot only (no historical).

Key technical detail: All Tadawul endpoints require curl_cffi with Chrome
TLS fingerprint impersonation.  Plain requests gets 403 from Akamai WAF.
The statementsTabData URL is a WebSphere Portal resource identifier
(p0/IZ7_... pattern) that is relative to the company profile page URL.
"""
from __future__ import annotations

import base64
import json
import logging
import re
import time
from datetime import date
from io import StringIO
from pathlib import Path
from typing import Any

import pandas as pd

logger = logging.getLogger(__name__)

_TADAWUL_BASE = "https://www.saudiexchange.sa"
_CACHE_DIR = Path("cache/sa_tadawul")

# GitHub gist URL containing base64-encoded WPS portal URLs for TASI/NOMUC.
# Maintained by the NPM tadawul-symbol package community.
# Decodes to /wps/portal/saudiexchange/ourmarkets/... URLs that return JSON
# with {recordsFiltered, data: [{symbol, companyName, companyURL, sectorName}], recordsTotal}.
_TASI_GIST_URL = (
    "https://gist.githubusercontent.com/cyancaesar/"
    "f7ed54c1fb9f4b557a15fac33f66b801/raw/tadawul.json"
)

# WebSphere Portal resource path for financial statement tab data.
# This is a relative URL from the company profile page.
_STMT_TAB_PATH = (
    "p0/IZ7_5A602H80O0VC4060O4GML81G57="
    "CZ6_5A602H80OGF2E0QF9BQDEG10K4="
    "NJstatementsTabData=/"
)

# XBRL line item -> canonical field name mapping
_XBRL_INCOME_MAP: dict[str, str] = {
    "sales": "revenue",
    "total revenue": "revenue",
    "depreciation and amortisation": "depreciation_amortization",
    "finance costs": "interest_expense",
    "finance and other income": "other_income",
    "purchases": "cost_of_revenue",
    "producing and manufacturing": "manufacturing_costs",
    "selling, administrative and general": "sga_expenses",
    "research and development": "rd_expenses",
    "exploration": "exploration_costs",
    "royalties and other taxes": "royalties_and_taxes",
    "profit (loss) before zakat and income tax": "ebit",
    "income taxes and zakat": "taxes",
    "profit (loss) for period from continuing operations": "net_income",
    "profit (loss) for period": "net_income",
    "profit (loss), attributable to equity holders": "net_income_attributable",
    "profit (loss), attributable to non-controlling interests": "minority_interest",
    "basic earnings (loss) per share": "eps",
    "diluted earnings (loss) per share": "eps_diluted",
    "other income related to sales": "other_revenue",
}

_XBRL_BALANCE_MAP: dict[str, str] = {
    "total current assets": "current_assets",
    "total non-current assets": "noncurrent_assets",
    "total assets": "total_assets",
    "total current liabilities": "current_liabilities",
    "total non-current liabilities": "noncurrent_liabilities",
    "total liabilities": "total_liabilities",
    "total equity": "total_equity",
    "equity attributable to equity holders of parent": "equity_attributable",
    "non-controlling interests": "minority_interest_equity",
    "total liabilities and equity": "total_liabilities_and_equity",
    "cash and cash equivalents": "cash_and_equivalents",
    "bank balances and cash": "cash_and_equivalents",  # Tadawul wording
    "inventories": "inventory",
    "trade receivables": "receivables",
    "trade payables": "payables",
    "property, plant and equipment": "property_plant_equipment",
    "goodwill": "goodwill",
    "intangible assets": "intangible_assets",
    "retained earnings": "retained_earnings",
    "short-term borrowings": "short_term_debt",
    "long-term borrowings": "long_term_debt",
}

_XBRL_CASHFLOW_MAP: dict[str, str] = {
    "net cash from operating activities": "operating_cash_flow",
    "net cash used in investing activities": "investing_cf",
    "net cash used in financing activities": "financing_cf",
    "net cash from (used in) operating activities": "operating_cash_flow",
    "net cash from (used in) investing activities": "investing_cf",
    "net cash from (used in) financing activities": "financing_cf",
    # Tadawul XBRL uses "flows" in the line item names
    "net cash flows from (used in) operating activities": "operating_cash_flow",
    "net cash flows from (used in) investing activities": "investing_cf",
    "net cash flows from (used in) financing activities": "financing_cf",
    "net cash flows from operating activities": "operating_cash_flow",
    "net cash flows from (used in) operations": "operating_cash_flow",
    "capital expenditure": "capex",
    "capital expenditures": "capex",  # Tadawul uses plural
    "purchase of property, plant and equipment": "capex",
    "dividends paid": "dividends_paid",
    "dividends paid to shareholders": "dividends_paid",
}

# Statement type code -> (table section keyword, field map)
_STATEMENT_CONFIG: dict[str, tuple[str, dict[str, str]]] = {
    "income": ("statement of income", _XBRL_INCOME_MAP),
    "balance": ("financial position", _XBRL_BALANCE_MAP),
    "cashflow": ("cash flow", _XBRL_CASHFLOW_MAP),
}


# ---------------------------------------------------------------------------
# curl_cffi session helper
# ---------------------------------------------------------------------------

def _get_session():
    """Create a curl_cffi session with Chrome TLS fingerprint."""
    from curl_cffi import requests as cf_requests
    return cf_requests.Session(impersonate="chrome")


# ---------------------------------------------------------------------------
# Company directory (cached)
# ---------------------------------------------------------------------------

_company_cache: dict[str, dict] | None = None
_company_cache_time: float = 0.0
_COMPANY_CACHE_TTL = 86400  # 24 hours


def _fetch_company_directory() -> dict[str, dict]:
    """Fetch the full TASI company directory with sectors.

    Returns dict keyed by symbol: {symbol, companyName, sectorName, isin, ...}
    """
    global _company_cache, _company_cache_time

    now = time.time()
    if _company_cache is not None and (now - _company_cache_time) < _COMPANY_CACHE_TTL:
        return _company_cache

    s = _get_session()
    directory: dict[str, dict] = {}

    # Source 1: ThemeSearchUtilityServlet (1,949 companies with ISIN)
    try:
        r = s.get(
            f"{_TADAWUL_BASE}/tadawul.eportal.theme.helper/ThemeSearchUtilityServlet",
            params={"searchText": ""},
            timeout=20,
        )
        if r.status_code == 200:
            for item in r.json():
                sym = item.get("symbol", "")
                if sym:
                    directory[sym] = {
                        "ticker": sym,
                        "name": item.get("companyNameEN", item.get("tradingNameEn", "")),
                        "name_ar": item.get("companyNameAR", ""),
                        "trading_name": item.get("tradingNameEn", ""),
                        "isin": item.get("isin", ""),
                        "market_type": item.get("market_type", ""),
                        "cik": sym,
                        "exchange": "Tadawul",
                        "country": "SA",
                        "market_id": "sa_tadawul",
                        "sector": "",
                        "industry": "",
                    }
            logger.info("Tadawul directory (search): %d companies", len(directory))
    except Exception as exc:
        logger.warning("Tadawul ThemeSearchUtilityServlet failed: %s", exc)

    # Source 2: TASI indices page (269 main-market companies with sectors)
    # Fetch the gist to get the base64-encoded WPS portal URL, then decode
    try:
        gist_r = s.get(_TASI_GIST_URL, timeout=15)
        gist_data = gist_r.json()
        tasi_url = base64.b64decode(gist_data["TASI"]["EN"]).decode("utf-8")
        r = s.get(tasi_url, timeout=20)
        if r.status_code == 200:
            data = r.json()
            for item in data.get("data", []):
                sym = item.get("symbol", "")
                if sym and sym in directory:
                    directory[sym]["sector"] = item.get("sectorName", "")
                    # Store the companyURL for profile page access
                    directory[sym]["_profile_url"] = item.get("companyURL", "")
                elif sym:
                    directory[sym] = {
                        "ticker": sym,
                        "name": item.get("companyName", ""),
                        "isin": "",
                        "sector": item.get("sectorName", ""),
                        "industry": "",
                        "cik": sym,
                        "exchange": "Tadawul",
                        "country": "SA",
                        "market_id": "sa_tadawul",
                        "_profile_url": item.get("companyURL", ""),
                    }
            logger.info("Tadawul directory (TASI sectors): enriched with sector data")
    except Exception as exc:
        logger.debug("Tadawul TASI sector fetch failed: %s", exc)

    # Source 3: TickerServlet (live trade data for price enrichment)
    try:
        r = s.get(
            f"{_TADAWUL_BASE}/tadawul.eportal.theme.helper/TickerServlet",
            timeout=20,
        )
        if r.status_code == 200:
            for stock in r.json().get("stockData", []):
                sym = stock.get("pk_rf_company", "")
                if sym and sym in directory:
                    directory[sym]["last_price"] = stock.get("lastTradePrice")
                    directory[sym]["volume"] = stock.get("volumeTraded")
                    directory[sym]["change_pct"] = stock.get("changePercent")
    except Exception as exc:
        logger.debug("Tadawul TickerServlet failed: %s", exc)

    if directory:
        _company_cache = directory
        _company_cache_time = now

    return directory


# ---------------------------------------------------------------------------
# Foreign ownership table (cached, 411 companies)
# ---------------------------------------------------------------------------

_foreign_ownership_cache: dict[str, dict] | None = None
_foreign_ownership_cache_time: float = 0.0
_FOREIGN_OWNERSHIP_TTL = 3600  # 1 hour


def _fetch_foreign_ownership_table() -> dict[str, dict]:
    """Fetch and cache the full Tadawul foreign ownership table.

    The page at /newsandreports/reports-publications/foreign-ownership
    serves a server-rendered HTML table with 411 companies.  No JavaScript,
    no pagination needed.  curl_cffi Chrome impersonation required.

    Returns dict keyed by symbol: {company, max_foreign_pct,
    actual_foreign_pct, strategic_foreign_pct}.
    """
    global _foreign_ownership_cache, _foreign_ownership_cache_time

    now = time.time()
    if _foreign_ownership_cache is not None and (now - _foreign_ownership_cache_time) < _FOREIGN_OWNERSHIP_TTL:
        return _foreign_ownership_cache

    result: dict[str, dict] = {}
    try:
        s = _get_session()
        r = s.get(
            f"{_TADAWUL_BASE}/wps/portal/saudiexchange/newsandreports/"
            "reports-publications/foreign-ownership?locale=en",
            timeout=30,
        )
        if r.status_code != 200:
            logger.warning("Tadawul foreign ownership page: %d", r.status_code)
            return _foreign_ownership_cache or result

        tables = re.findall(r"<table[^>]*>(.*?)</table>", r.text, re.DOTALL | re.I)
        if not tables:
            return result

        rows = re.findall(r"<tr[^>]*>(.*?)</tr>", tables[0], re.DOTALL | re.I)
        for row in rows:
            cells = re.findall(r"<td[^>]*>(.*?)</td>", row, re.DOTALL)
            cells_clean = [re.sub(r"<[^>]+>", "", c).strip() for c in cells]
            if len(cells_clean) < 5 or not cells_clean[0] or not cells_clean[0][0].isdigit():
                continue

            symbol = cells_clean[0]
            try:
                max_pct = float(cells_clean[2].replace("%", "").strip())
            except (ValueError, TypeError):
                max_pct = 49.0
            try:
                actual_pct = float(cells_clean[3].replace("%", "").strip())
            except (ValueError, TypeError):
                actual_pct = 0.0
            try:
                strategic_pct = float(cells_clean[4].replace("%", "").strip())
            except (ValueError, TypeError):
                strategic_pct = 0.0

            result[symbol] = {
                "company": cells_clean[1],
                "max_foreign_pct": max_pct,
                "actual_foreign_pct": actual_pct,
                "strategic_foreign_pct": strategic_pct,
            }

        logger.info("Tadawul foreign ownership table: %d companies", len(result))
        _foreign_ownership_cache = result
        _foreign_ownership_cache_time = now

    except Exception as exc:
        logger.warning("Tadawul foreign ownership fetch failed: %s", exc)
        if _foreign_ownership_cache is not None:
            return _foreign_ownership_cache

    return result


# ---------------------------------------------------------------------------
# XBRL financial extraction
# ---------------------------------------------------------------------------


def _get_xbrl_links(symbol: str, s=None) -> list[dict]:
    """Get XBRL HTML download links for a company from statementsTabData.

    Returns list of {url, filing_date, period, period_type} sorted newest first.
    """
    if s is None:
        s = _get_session()

    directory = _fetch_company_directory()
    company = directory.get(symbol, {})
    profile_path = company.get("_profile_url", "")

    if not profile_path:
        # Construct a generic profile URL pattern
        logger.debug("No profile URL cached for %s, trying search", symbol)
        return []

    profile_url = (
        profile_path
        if profile_path.startswith("http")
        else f"{_TADAWUL_BASE}{profile_path}"
    )

    # Load the company profile page
    try:
        r = s.get(profile_url, timeout=25)
        if r.status_code != 200:
            logger.debug("Tadawul profile page failed for %s: %d", symbol, r.status_code)
            return []
    except Exception as exc:
        logger.debug("Tadawul profile page error for %s: %s", symbol, exc)
        return []

    # Hit the statementsTabData endpoint (relative to profile page)
    profile_base = profile_url.rsplit("/", 1)[0] + "/"
    try:
        r = s.get(
            profile_base + _STMT_TAB_PATH,
            params={
                "statementType": "6",
                "reportType": "Q",
                "requestLocale": "en",
                "symbol": symbol,
            },
            timeout=20,
        )
        if r.status_code != 200 or len(r.text) < 200:
            return []
    except Exception as exc:
        logger.debug("Tadawul statementsTabData failed for %s: %s", symbol, exc)
        return []

    # Parse XBRL HTML links from the response
    xbrl_links: list[dict] = []
    # Pattern: /Resources/XBRL_DOCS/{companyId}_{symbol}_{date}_Eng.html
    html_matches = re.findall(
        r'href="(/Resources/XBRL_DOCS/[^"]+_Eng\.html)"',
        r.text,
    )

    # Extract filing dates from the table structure
    # The table rows alternate: period label, then dates per year
    rows = re.findall(r"<tr[^>]*>(.*?)</tr>", r.text, re.DOTALL | re.I)
    period_dates: list[tuple[str, str]] = []  # (period_type, filing_date)

    in_xbrl_section = False
    current_period = ""
    for row in rows:
        text = re.sub(r"<[^>]+>", " ", row).strip()
        text = re.sub(r"\s+", " ", text)

        if "XBRL" in text and len(text) < 20:
            in_xbrl_section = True
            continue

        if in_xbrl_section:
            # Period type row (Annual, Q1, Q2, Q3, Q4)
            period_match = re.match(
                r"\s*(Annual|Q[1-4])\s+([\d-]+(?:\s+[\d-]+)*)",
                text,
            )
            if period_match:
                period_type = period_match.group(1)
                dates = re.findall(r"(\d{4}-\d{2}-\d{2})", text)
                for d in dates:
                    period_dates.append((period_type, d))

            if "Board Report" in text or "ESG Report" in text:
                in_xbrl_section = False

    # Match HTML links with their filing dates
    for html_url in html_matches:
        # Extract date from URL: ..._{date}_Eng.html
        date_match = re.search(r"_(\d{4}-\d{2}-\d{2})_", html_url)
        filing_date = date_match.group(1) if date_match else ""

        # Find the matching period type
        period_type = "unknown"
        for pt, fd in period_dates:
            if fd == filing_date:
                period_type = pt
                break

        xbrl_links.append({
            "url": f"{_TADAWUL_BASE}{html_url}",
            "filing_date": filing_date,
            "period_type": period_type,
        })

    # Sort newest first
    xbrl_links.sort(key=lambda x: x.get("filing_date", ""), reverse=True)
    logger.info(
        "Tadawul XBRL links for %s: %d files found",
        symbol, len(xbrl_links),
    )
    return xbrl_links


def _parse_xbrl_html(
    html_text: str,
    statement_type: str,
) -> list[dict]:
    """Parse structured financial data from Tadawul XBRL HTML.

    The XBRL HTML contains 42 tables. We identify the target statement
    by scanning table headers for keywords (e.g., "Statement of income",
    "financial position", "cash flow"), then extract line items + values.

    Returns list of {canonical_name, value, column_index}.
    """
    config = _STATEMENT_CONFIG.get(statement_type)
    if not config:
        return []

    section_keyword, field_map = config

    try:
        dfs = pd.read_html(StringIO(html_text))
    except Exception as exc:
        logger.debug("XBRL HTML parse failed: %s", exc)
        return []

    # Find the target data table by scanning for the section keyword.
    # Skip small tables (TOC, section headers) -- look for the table
    # that both matches the keyword AND has substantial data (>5 rows).
    target_idx = None
    for i, df in enumerate(dfs):
        if len(df) < 5:
            continue
        first_row_text = " ".join(str(v) for v in df.iloc[0].values).lower()
        if section_keyword in first_row_text:
            target_idx = i
            break

    if target_idx is None:
        return []

    # The data may be in the matched table itself or the next one.
    for data_idx in range(target_idx, min(target_idx + 3, len(dfs))):
        df = dfs[data_idx]
        if len(df) < 5:
            continue

        rows: list[dict] = []
        for _, row in df.iterrows():
            vals = list(row.values)
            if len(vals) < 2:
                continue

            # First column is the line item name
            line_item = str(vals[0]).strip().lower()
            if line_item in ("nan", "", "none"):
                continue

            # Remove [abstract] markers
            if "[abstract]" in line_item:
                continue

            # Match against field map (substring match)
            canonical = None
            for pattern, canon in field_map.items():
                if pattern in line_item:
                    canonical = canon
                    break

            if not canonical:
                continue

            # Extract numeric values from remaining columns
            # Typically: col1=current_quarter, col2=prior_quarter,
            # col3=current_ytd, col4=prior_ytd
            for col_idx in range(1, min(len(vals), 5)):
                val = vals[col_idx]
                if pd.isna(val):
                    continue
                try:
                    # Handle string numbers with commas
                    val_str = str(val).replace(",", "").replace("(", "-").replace(")", "")
                    val_str = val_str.replace("\u2011", "-").replace("\u2013", "-")
                    num = float(val_str)
                    rows.append({
                        "canonical_name": canonical,
                        "value": num,
                        "column_index": col_idx,
                    })
                    break  # Take first non-NaN value (current period)
                except (ValueError, TypeError):
                    continue

        if rows:
            return rows

    return []


def _fetch_xbrl_financials(
    symbol: str,
    statement_type: str,
    max_files: int = 8,
) -> pd.DataFrame:
    """Fetch structured financial data from Tadawul XBRL HTML files.

    Downloads XBRL HTML files for the given symbol, parses the structured
    IFRS tables, and returns a canonical long-format DataFrame.

    Parameters
    ----------
    symbol:
        Tadawul ticker (e.g. '2222', '2010').
    statement_type:
        One of 'income', 'balance', 'cashflow'.
    max_files:
        Maximum XBRL files to download (8 = ~2 years quarterly).
    """
    s = _get_session()
    xbrl_links = _get_xbrl_links(symbol, s=s)
    if not xbrl_links:
        return pd.DataFrame()

    all_rows: list[dict] = []

    for entry in xbrl_links[:max_files]:
        url = entry["url"]
        filing_date = entry.get("filing_date", "")
        period_type = entry.get("period_type", "")

        try:
            r = s.get(url, timeout=30)
            if r.status_code != 200:
                continue

            rows = _parse_xbrl_html(r.text, statement_type)
            for row in rows:
                row["filing_date"] = filing_date
                row["report_date"] = filing_date  # Will be refined below
                row["period_type"] = period_type
            all_rows.extend(rows)

            # Rate limiting
            time.sleep(0.5)

        except Exception as exc:
            logger.debug("Tadawul XBRL download failed for %s: %s", url[-50:], exc)

    if not all_rows:
        return pd.DataFrame()

    df = pd.DataFrame(all_rows)

    # Drop column_index (internal use only)
    if "column_index" in df.columns:
        df = df.drop(columns=["column_index"])

    # Deduplicate: keep one value per (canonical_name, filing_date)
    df = df.sort_values("filing_date", ascending=False)
    df = df.drop_duplicates(subset=["canonical_name", "filing_date"], keep="first")

    for col in ("filing_date", "report_date"):
        if col in df.columns:
            df[col] = pd.to_datetime(df[col], errors="coerce")

    logger.info(
        "Tadawul XBRL for %s/%s: %d records from %d/%d files",
        symbol, statement_type, len(df),
        min(max_files, len(xbrl_links)), len(xbrl_links),
    )
    return df


# ---------------------------------------------------------------------------
# SATadawulClient
# ---------------------------------------------------------------------------


class SATadawulClient:
    """PIT client for Saudi Tadawul equities.

    Uses curl_cffi with Chrome TLS impersonation to access Tadawul's
    native APIs (ThemeSearchUtilityServlet, TickerServlet, XBRL HTML).
    No yfinance dependency for search, profile, or financials.

    Implements the ``PITClient`` protocol.
    """

    def __init__(self, cache_dir: Path | str = _CACHE_DIR) -> None:
        self._cache_dir = Path(cache_dir)

    def _cache_path(self, identifier: str, fn: str) -> Path:
        return self._cache_dir / identifier.upper() / fn

    def _read_cache(self, identifier: str, fn: str) -> dict | None:
        p = self._cache_path(identifier, fn)
        if not p.exists():
            return None
        try:
            age_days = (date.today() - date.fromtimestamp(p.stat().st_mtime)).days
            if fn == "profile.json" and age_days > 7:
                return None
            if fn.endswith("_financials.json") and age_days > 30:
                return None
            return json.loads(p.read_text(encoding="utf-8"))
        except Exception:
            return None

    def _write_cache(self, identifier: str, fn: str, data: Any) -> None:
        p = self._cache_path(identifier, fn)
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(json.dumps(data, default=str, indent=2), encoding="utf-8")

    @property
    def market_id(self) -> str:
        return "sa_tadawul"

    @property
    def market_name(self) -> str:
        return "Saudi Arabia (Tadawul)"

    # -- Company discovery ---------------------------------------------------

    def list_companies(self, query: str = "") -> list[dict[str, Any]]:
        """List/search Tadawul companies from native directory."""
        directory = _fetch_company_directory()

        if not query:
            return list(directory.values())

        q = query.lower().strip()
        results = []

        # Exact symbol match
        if q in directory:
            return [directory[q]]

        # Substring search on name, trading name, symbol
        for sym, info in directory.items():
            name = info.get("name", "").lower()
            trading = info.get("trading_name", "").lower()
            if q in sym.lower() or q in name or q in trading:
                results.append(info)

        # Fuzzy match if no substring hits
        if not results and len(q) >= 3:
            try:
                from rapidfuzz import process, fuzz
                choices = {
                    sym: info.get("name", "").lower()
                    for sym, info in directory.items()
                }
                matches = process.extract(
                    q, choices, scorer=fuzz.WRatio,
                    score_cutoff=70.0, limit=10,
                )
                results = [directory[sym] for _, _, sym in matches]
            except ImportError:
                pass

        return results

    def search_company(self, name: str) -> list[dict[str, Any]]:
        return self.list_companies(query=name)

    # -- Company profile -----------------------------------------------------

    def get_profile(self, identifier: str) -> dict[str, Any]:
        cached = self._read_cache(identifier, "profile.json")
        if cached:
            return cached

        directory = _fetch_company_directory()
        company = directory.get(identifier.upper(), {})

        if not company:
            # Try search
            results = self.search_company(identifier)
            if results:
                company = results[0]

        profile = {
            "name": company.get("name", identifier),
            "ticker": company.get("ticker", identifier),
            "isin": company.get("isin", ""),
            "country": "SA",
            "sector": company.get("sector", ""),
            "industry": company.get("industry", ""),
            "exchange": "Tadawul",
            "currency": "SAR",
            "market_id": self.market_id,
            "last_price": company.get("last_price"),
            "volume": company.get("volume"),
        }

        if profile.get("name"):
            self._write_cache(identifier, "profile.json", profile)
        return profile

    # -- Financial statements ------------------------------------------------

    def get_income_statement(self, identifier: str) -> pd.DataFrame:
        return self._fetch_financials(identifier, "income")

    def get_balance_sheet(self, identifier: str) -> pd.DataFrame:
        return self._fetch_financials(identifier, "balance")

    def get_cashflow_statement(self, identifier: str) -> pd.DataFrame:
        return self._fetch_financials(identifier, "cashflow")

    def _fetch_financials(
        self, identifier: str, statement_type: str,
    ) -> pd.DataFrame:
        """Fetch financials from Tadawul XBRL (primary, fast, no LLM).

        Falls back to filing discovery + LLM/fuzzy extraction if XBRL
        is unavailable.
        """
        # Check cache
        cache_key = f"{statement_type}_financials.json"
        cached = self._read_cache(identifier, cache_key)
        if cached:
            try:
                df = pd.DataFrame(cached)
                if not df.empty:
                    for col in ("filing_date", "report_date"):
                        if col in df.columns:
                            df[col] = pd.to_datetime(df[col], errors="coerce")
                    return df
            except Exception:
                pass

        # Primary: XBRL HTML extraction (fast, structured, no LLM)
        try:
            df = _fetch_xbrl_financials(identifier, statement_type)
            if df is not None and not df.empty:
                self._write_cache(
                    identifier, cache_key,
                    df.to_dict(orient="records"),
                )
                logger.info(
                    "Tadawul %s %s: %d rows from XBRL (fast path)",
                    identifier, statement_type, len(df),
                )
                return df
        except Exception as exc:
            logger.debug("Tadawul XBRL extraction failed for %s: %s", identifier, exc)

        # Fallback: filing discovery + LLM/fuzzy extraction
        try:
            from operator1.clients.filing_discoverer import try_filing_extraction
            df = try_filing_extraction(
                ticker=identifier,
                market_id=self.market_id,
                statement_type=statement_type,
                llm_client=None,
            )
            if df is not None and not df.empty:
                logger.info(
                    "Tadawul %s %s: %d rows from filing discovery",
                    identifier, statement_type, len(df),
                )
                return df
        except Exception as exc:
            logger.debug("Tadawul filing discovery failed for %s: %s", identifier, exc)

        return pd.DataFrame()

    # -- Price data ----------------------------------------------------------

    def get_quotes(self, identifier: str) -> pd.DataFrame:
        """Tadawul does not provide historical OHLCV via API.

        Current-day snapshot is available via TickerServlet but historical
        data is handled by ohlcv_provider.py (yfinance .SR suffix).
        """
        return pd.DataFrame()

    def get_peers(self, identifier: str) -> list[str]:
        """Return peers from the same sector."""
        directory = _fetch_company_directory()
        company = directory.get(identifier.upper(), {})
        sector = company.get("sector", "")
        if not sector:
            return []
        return [
            sym for sym, info in directory.items()
            if info.get("sector") == sector and sym != identifier.upper()
        ][:10]

    def get_executives(self, identifier: str) -> list[dict[str, Any]]:
        return []

    # -- Institutional holders (native Tadawul foreign ownership page) --------

    def get_holders(self, identifier: str) -> list[dict[str, Any]]:
        """Fetch foreign ownership data from Tadawul's native HTML table.

        Data source: The Tadawul foreign-ownership page at
        ``/newsandreports/reports-publications/foreign-ownership``
        serves a server-rendered HTML table with **411 companies** containing:
          - Total Foreign Ownership Maximum Limit (%)
          - Total Foreign Ownership Actual (%)
          - Foreign Strategic Investors Ownership (%)

        The full table is fetched once and cached at module level (1-hour TTL)
        to avoid re-downloading 760KB for each company lookup.

        All fetched via curl_cffi Chrome impersonation (same as financials).
        No yfinance, no LLM needed.

        Probing notes (2026-03-23):
          - OwnershipServlet, ShareholderServlet: 404 (don't exist)
          - TickerServlet: no ownership fields (trade data only)
          - WPS profile page: 0 NJ resource paths for ownership tabs
          - statementsTabData type=4: filing calendar (dates only)
          - statementsTabData type=5: board report calendar (dates only)
          - Per-company major shareholder detail requires CMA authentication
          - The foreign-ownership HTML table is the ONLY public holder endpoint
        """
        holders: list[dict[str, Any]] = []

        # Use the cached foreign ownership table
        ownership = _fetch_foreign_ownership_table()
        symbol = identifier.upper().strip()
        company_data = ownership.get(symbol)

        if not company_data:
            logger.debug("Tadawul: no foreign ownership data for %s", symbol)
            return holders

        company_name = company_data["company"]
        actual_pct = company_data["actual_foreign_pct"]
        strategic_pct = company_data["strategic_foreign_pct"]
        max_pct = company_data["max_foreign_pct"]

        if actual_pct > 0:
            holders.append({
                "name": "Foreign Investors (aggregate)",
                "shares": 0,
                "value": 0.0,
                "percentage": round(actual_pct, 2),
                "holder_type": "foreign_aggregate",
                "date_reported": str(date.today()),
                "source": "tadawul_foreign_ownership",
                "max_foreign_limit_pct": round(max_pct, 2),
            })
        if strategic_pct > 0:
            holders.append({
                "name": "Foreign Strategic Investors",
                "shares": 0,
                "value": 0.0,
                "percentage": round(strategic_pct, 2),
                "holder_type": "foreign_strategic",
                "date_reported": str(date.today()),
                "source": "tadawul_foreign_ownership",
            })
        # Non-strategic foreign (retail foreign investors)
        retail_foreign = actual_pct - strategic_pct
        if retail_foreign > 0.01:
            holders.append({
                "name": "Foreign Retail Investors (implied)",
                "shares": 0,
                "value": 0.0,
                "percentage": round(retail_foreign, 2),
                "holder_type": "foreign_retail",
                "date_reported": str(date.today()),
                "source": "tadawul_foreign_ownership",
            })
        # Implied domestic ownership
        domestic_pct = 100.0 - actual_pct
        if domestic_pct > 0:
            holders.append({
                "name": "Domestic Investors (implied)",
                "shares": 0,
                "value": 0.0,
                "percentage": round(domestic_pct, 2),
                "holder_type": "domestic_aggregate",
                "date_reported": str(date.today()),
                "source": "tadawul_foreign_ownership",
            })

        if holders:
            logger.info(
                "Tadawul holders for %s (%s): foreign=%.2f%% (strategic=%.2f%%), "
                "domestic=%.2f%%, max_limit=%.0f%%",
                symbol, company_name, actual_pct, strategic_pct,
                domestic_pct, max_pct,
            )

        return holders

    def get_holder_history(self, identifier: str, years: int = 2) -> pd.DataFrame:
        """Return ownership metrics from Tadawul foreign ownership data.

        Currently returns a single-row snapshot.  Tadawul does not expose
        historical foreign ownership data via public API (would require
        scraping the page periodically and storing locally).
        """
        try:
            holders = self.get_holders(identifier)
            if not holders:
                return pd.DataFrame()

            foreign = next(
                (h for h in holders if h["holder_type"] == "foreign_aggregate"),
                None,
            )
            foreign_pct = foreign["percentage"] if foreign else 0.0
            strategic = next(
                (h for h in holders if h["holder_type"] == "foreign_strategic"),
                None,
            )
            strategic_pct = strategic["percentage"] if strategic else 0.0

            return pd.DataFrame([{
                "date_reported": pd.Timestamp(date.today()),
                "inst_ownership_pct": round(foreign_pct, 2),
                "inst_top5_concentration": 0.0,
                "inst_holder_count": len(holders),
                "foreign_strategic_pct": round(strategic_pct, 2),
            }])
        except Exception as exc:
            logger.debug("Tadawul holder history failed for %s: %s", identifier, exc)
        return pd.DataFrame()

    def get_insider_transactions(self, identifier: str) -> list[dict[str, Any]]:
        """Fetch board report filing dates from statementsTabData type=5.

        Tadawul does not expose per-transaction insider data via public API.
        However, statementsTabData with statementType=5 returns board report
        and ESG report filing dates, which indicate governance disclosure
        events.  Each date represents a board-approved filing.

        Probing confirmed: statementType=5 returns Annual/Quarterly board
        report dates + ESG report dates for each company.
        """
        transactions: list[dict[str, Any]] = []
        try:
            directory = _fetch_company_directory()
            company = directory.get(identifier.upper(), {})
            profile_path = company.get("_profile_url", "")

            if not profile_path:
                return transactions

            s = _get_session()
            profile_url = (
                profile_path
                if profile_path.startswith("http")
                else f"{_TADAWUL_BASE}{profile_path}"
            )

            # Load profile page first (needed for WPS session)
            s.get(profile_url, timeout=25)

            profile_base = profile_url.rsplit("/", 1)[0] + "/"
            r = s.get(
                profile_base + _STMT_TAB_PATH,
                params={
                    "statementType": "5",
                    "reportType": "Q",
                    "requestLocale": "en",
                    "symbol": identifier.upper(),
                },
                timeout=15,
            )
            if r.status_code != 200 or len(r.text) < 200:
                return transactions

            # Parse board report dates from the table
            rows = re.findall(r"<tr[^>]*>(.*?)</tr>", r.text, re.DOTALL | re.I)
            for row in rows:
                cells = re.findall(r"<td[^>]*>(.*?)</td>", row, re.DOTALL)
                cells_clean = [re.sub(r"<[^>]+>", "", c).strip() for c in cells]
                if not cells_clean:
                    continue
                report_type = cells_clean[0] if cells_clean else ""
                if report_type not in ("Annual", "Q1", "Q2", "Q3", "Q4", "Board Report", "ESG Report"):
                    continue
                # Each subsequent cell is a date for a year column
                for cell in cells_clean[1:]:
                    if cell and cell != "-" and re.match(r"\d{4}-\d{2}-\d{2}", cell):
                        transactions.append({
                            "insider_name": f"Board ({report_type})",
                            "position": "Board of Directors",
                            "date": cell,
                            "transaction": f"Board Report Filing ({report_type})",
                            "shares": 0,
                            "value": 0.0,
                            "source": "tadawul_statementsTabData_type5",
                        })

            if transactions:
                logger.info(
                    "Tadawul board report dates for %s: %d filings",
                    identifier, len(transactions),
                )
        except Exception as exc:
            logger.debug("Tadawul board report fetch failed for %s: %s", identifier, exc)
        return transactions
