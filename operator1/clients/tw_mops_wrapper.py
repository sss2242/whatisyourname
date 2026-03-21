"""Taiwan MOPS PIT client -- XBRL fast path (2 calls = all statements).

Uses the MOPS XBRL platform endpoint which returns ALL financial
statements (balance + income + cashflow) in a single inline XBRL HTML
response with standardized IFRS concept codes and English titles.

**Primary (XBRL fast path)**: 2 calls total for 2+ years of data
    ``mopsov.twse.com.tw/server-java/t164sb01``
    - GET with CO_ID, SYEAR, SSEASON, REPORT_ID=C
    - Returns ~780KB inline XBRL HTML with 3 statement tables
    - Each call has current year + prior year comparison columns
    - 2 calls (current + 2 years ago) = 3 unique fiscal years

**Fallback (JSON API)**: Annual data from new MOPS Vue.js API
    ``mops.twse.com.tw/mops/api/t164sb04`` etc.

Both require ``curl_cffi`` with Chrome TLS impersonation.
No API key required.  No geo-blocking.

Coverage: ~1,700+ listed companies on TWSE/TPEX, ~$1.2T market cap.
"""

from __future__ import annotations

import json
import logging
import re
import time
from datetime import date
from io import StringIO
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

_MOPS_BASE = "https://mops.twse.com.tw"
_MOPS_API = f"{_MOPS_BASE}/mops/api"
_MOPSOV_BASE = "https://mopsov.twse.com.tw"
_TWSE_BASE = "https://www.twse.com.tw"
_CACHE_DIR = Path("cache/tw_mops")

# XBRL platform URL template (single call = all 3 statements)
_XBRL_URL = (
    "{base}/server-java/t164sb01"
    "?step=1&CO_ID={co_id}&SYEAR={syear}&SSEASON={sseason}&REPORT_ID=C"
)

_REQUEST_DELAY_S = 1.5


class TWMopsError(Exception):
    def __init__(self, endpoint: str, detail: str = "") -> None:
        self.endpoint = endpoint
        self.detail = detail
        super().__init__(f"TW MOPS error on {endpoint}: {detail}")


def _roc_to_ce(roc_year: int) -> int:
    """Convert ROC year to Common Era year. ROC 113 = CE 2024."""
    return roc_year + 1911


def _ce_to_roc(ce_year: int) -> int:
    """Convert Common Era year to ROC year."""
    return ce_year - 1911


# ---------------------------------------------------------------------------
# XBRL concept code -> canonical field name mapping
# ---------------------------------------------------------------------------

# Balance sheet (Table 0 in XBRL response)
_XBRL_BALANCE_MAP: dict[str, str] = {
    "1100": "cash_and_equivalents",
    "1170": "receivables",
    "1180": "receivables",        # receivables from related parties
    "130X": "inventory",
    "11XX": "current_assets",
    "1600": "property_plant_equipment",
    "1780": "intangible_assets",
    "15XX": "noncurrent_assets",
    "1XXX": "total_assets",
    "2170": "payables",
    "2530": "bonds_payable",
    "2320": "short_term_debt",     # current portion of LT debt
    "2540": "long_term_debt",
    "2200": "other_payables",
    "21XX": "current_liabilities",
    "25XX": "noncurrent_liabilities",
    "2XXX": "total_liabilities",
    "3100": "share_capital",
    "3200": "total_capital_surplus",
    "3300": "retained_earnings",
    "31XX": "equity_attributable",
    "3XXX": "total_equity",
    "3X2X": "total_liabilities_and_equity",
    # Goodwill -- some companies have it, code may vary
    "1805": "goodwill",
    "1821": "goodwill",  # alternative code
}

# Income statement (Table 1 in XBRL response)
_XBRL_INCOME_MAP: dict[str, str] = {
    "4000": "revenue",
    "5000": "cost_of_revenue",
    "5900": "gross_profit",
    "5950": "gross_profit",        # gross profit net
    "6100": "selling_expenses",
    "6200": "admin_expenses",
    "6300": "rd_expenses",
    "6000": "sga_expenses",
    "6900": "operating_income",
    "7100": "interest_income",
    "7510": "interest_expense",
    "7000": "non_operating_income",
    "7900": "ebit",                # pre-tax income
    "7950": "taxes",
    "8200": "net_income",          # net income from continuing operations
    "8500": "comprehensive_income",
    "8610": "net_income_attributable",  # attributable to parent
    "8620": "minority_interest",   # attributable to NCI
    "9750": "eps",
    "9850": "eps_diluted",
}

# Cash flow statement (Table 2 in XBRL response)
_XBRL_CASHFLOW_MAP: dict[str, str] = {
    "A00010": "pretax_income",
    "A10000": "pretax_income",
    "A20100": "depreciation",
    "A20200": "amortization",
    "AAAA": "operating_cash_flow",
    "B02700": "capex",
    "BBBB": "investing_cf",
    "C04500": "dividends_paid",
    "C04900": "stock_buybacks",
    "C05500": "treasury_share_disposal",
    "A20900": "interest_expense_cf",
    "A21200": "interest_income_cf",
    "CCCC": "financing_cf",
    "EEEE": "net_change_in_cash",
    "E00100": "beginning_cash",
    "E00200": "ending_cash",
}

# Table index -> (statement type, field map)
_TABLE_CONFIG: dict[int, tuple[str, dict[str, str]]] = {
    0: ("balance", _XBRL_BALANCE_MAP),
    1: ("income", _XBRL_INCOME_MAP),
    2: ("cashflow", _XBRL_CASHFLOW_MAP),
}

# JSON API endpoint mapping (fallback)
_JSON_API_ENDPOINTS: dict[str, str] = {
    "income": "t164sb04",
    "balance": "t164sb03",
    "cashflow": "t164sb05",
}

# JSON API Chinese -> canonical (fallback only)
_JSON_INCOME_MAP: dict[str, str] = {
    "營業收入合計": "revenue", "營業成本合計": "cost_of_revenue",
    "營業毛利（毛損）": "gross_profit", "研究發展費用": "rd_expenses",
    "營業費用合計": "sga_expenses", "營業利益（損失）": "operating_income",
    "財務成本淨額": "interest_expense", "稅前淨利（淨損）": "ebit",
    "所得稅費用（利益）合計": "taxes", "本期淨利（淨損）": "net_income",
    "基本每股盈餘": "eps", "稀釋每股盈餘": "eps_diluted",
}
_JSON_BALANCE_MAP: dict[str, str] = {
    "現金及約當現金": "cash_and_equivalents", "流動資產合計": "current_assets",
    "資產總計": "total_assets", "資產總額": "total_assets",
    "流動負債合計": "current_liabilities", "負債總計": "total_liabilities",
    "負債總額": "total_liabilities", "保留盈餘合計": "retained_earnings",
    "權益總計": "total_equity", "權益總額": "total_equity",
}
_JSON_CASHFLOW_MAP: dict[str, str] = {
    "營業活動之淨現金流入（流出）": "operating_cash_flow",
    "投資活動之淨現金流入（流出）": "investing_cf",
    "籌資活動之淨現金流入（流出）": "financing_cf",
    "取得不動產、廠房及設備": "capex",
    "發放現金股利": "dividends_paid",
}
_JSON_STATEMENT_CONFIG: dict[str, tuple[str, dict[str, str]]] = {
    "income": ("t164sb04", _JSON_INCOME_MAP),
    "balance": ("t164sb03", _JSON_BALANCE_MAP),
    "cashflow": ("t164sb05", _JSON_CASHFLOW_MAP),
}


# ---------------------------------------------------------------------------
# curl_cffi session helper
# ---------------------------------------------------------------------------

def _get_session():
    """Create a curl_cffi session with Chrome TLS fingerprint."""
    from curl_cffi import requests as cf_requests
    return cf_requests.Session(impersonate="chrome")


def _parse_number(s: str) -> float | None:
    """Parse number from XBRL or MOPS format. Handles commas and parens."""
    if not s or not s.strip() or s.strip() == "nan":
        return None
    s = s.strip()
    neg = s.startswith("(") and s.endswith(")")
    if neg:
        s = s[1:-1]
    s = s.replace(",", "")
    try:
        val = float(s)
        return -val if neg else val
    except (ValueError, TypeError):
        return None


# ---------------------------------------------------------------------------
# Main client class
# ---------------------------------------------------------------------------

class TWMopsClient:
    """Point-in-time client for MOPS (Taiwanese equities).

    Uses the XBRL platform fast path: **2 API calls** return all 3
    financial statements (balance + income + cashflow) for 2+ years.

    Fallback: MOPS JSON API for annual data if XBRL fails.
    """

    def __init__(self, cache_dir: Path | str = _CACHE_DIR) -> None:
        self._cache_dir = Path(cache_dir)
        self._session = None
        self._xbrl_cache: dict[str, dict[str, pd.DataFrame]] = {}

    def _get_or_create_session(self):
        if self._session is None:
            self._session = _get_session()
        return self._session

    def _cache_path(self, identifier: str, filename: str) -> Path:
        safe_id = identifier.replace("/", "_").replace("\\", "_").upper()
        return self._cache_dir / safe_id / filename

    def _read_cache(self, identifier: str, filename: str) -> dict | None:
        path = self._cache_path(identifier, filename)
        if not path.exists():
            return None
        try:
            age_days = (date.today() - date.fromtimestamp(path.stat().st_mtime)).days
            if age_days > 1:
                return None
            return json.loads(path.read_text(encoding="utf-8"))
        except Exception:
            return None

    def _write_cache(self, identifier: str, filename: str, data: dict) -> None:
        path = self._cache_path(identifier, filename)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(
            json.dumps(data, default=str, indent=2, ensure_ascii=False),
            encoding="utf-8",
        )

    @property
    def market_id(self) -> str:
        return "tw_mops"

    @property
    def market_name(self) -> str:
        return "Taiwan (TWSE / TPEX) -- MOPS"

    # ------------------------------------------------------------------
    # Company listing and search
    # ------------------------------------------------------------------

    def list_companies(self, query: str = "") -> list[dict[str, Any]]:
        """List Taiwanese companies from TWSE JSON API."""
        session = self._get_or_create_session()
        try:
            r = session.get(
                f"{_TWSE_BASE}/exchangeReport/STOCK_DAY_ALL",
                params={"response": "json"},
                timeout=15,
            )
            data = r.json()
            items = data.get("data", []) if isinstance(data, dict) else []
            companies = []
            for item in items:
                if isinstance(item, (list, tuple)) and len(item) >= 2:
                    companies.append({
                        "ticker": str(item[0]),
                        "name": str(item[1]),
                        "cik": str(item[0]),
                        "exchange": "TWSE",
                        "country": "TW",
                        "market_id": self.market_id,
                    })
            if query:
                q = query.lower()
                companies = [
                    c for c in companies
                    if q in c["ticker"].lower() or q in c["name"].lower()
                ]
            return companies
        except Exception as exc:
            logger.warning("TWSE company list failed: %s", exc)
            return []

    def search_company(self, name: str) -> list[dict[str, Any]]:
        return self.list_companies(query=name)

    # ------------------------------------------------------------------
    # Company profile
    # ------------------------------------------------------------------

    def get_profile(self, identifier: str) -> dict[str, Any]:
        cached = self._read_cache(identifier, "profile.json")
        if cached:
            return cached

        matches: list[dict[str, Any]] = []
        for attempt in range(1, 4):
            try:
                matches = self.search_company(identifier)
                if matches:
                    break
            except Exception:
                pass
            time.sleep(min(2 ** (attempt - 1), 4))

        raw_profile = {
            "name": matches[0]["name"] if matches else "",
            "ticker": identifier,
            "isin": "",
            "country": "TW",
            "sector": "",
            "industry": "",
            "exchange": "TWSE",
            "currency": "TWD",
            "cik": identifier,
        }

        from operator1.clients.canonical_translator import translate_profile
        profile = translate_profile(raw_profile, self.market_id)
        self._write_cache(identifier, "profile.json", profile)
        return profile

    # ------------------------------------------------------------------
    # Financial statements (XBRL fast path: 2 calls = all statements)
    # ------------------------------------------------------------------

    def get_income_statement(self, identifier: str) -> pd.DataFrame:
        return self._get_statement(identifier, "income")

    def get_balance_sheet(self, identifier: str) -> pd.DataFrame:
        return self._get_statement(identifier, "balance")

    def get_cashflow_statement(self, identifier: str) -> pd.DataFrame:
        return self._get_statement(identifier, "cashflow")

    def _get_statement(self, identifier: str, statement_type: str) -> pd.DataFrame:
        """Return a single statement type from the XBRL cache.

        On first call for a company, fetches ALL statements via 2 XBRL
        calls and caches them.  Subsequent calls for the same company
        return from cache without additional API calls.
        """
        if identifier not in self._xbrl_cache:
            self._xbrl_cache[identifier] = self._fetch_xbrl_all(identifier)

        df = self._xbrl_cache[identifier].get(statement_type, pd.DataFrame())
        if not df.empty:
            return df

        # Fallback: JSON API
        logger.info("XBRL fast path empty for %s/%s, trying JSON fallback", identifier, statement_type)
        return self._fetch_json_fallback(identifier, statement_type)

    def _fetch_xbrl_all(self, identifier: str) -> dict[str, pd.DataFrame]:
        """Fetch ALL financial statements via 2 XBRL platform calls.

        Call 1: Current year Q4 -> current year + prior year data
        Call 2: 2 years ago Q4 -> that year + year before data

        Result: 3 unique fiscal years of data for all 3 statement types.
        """
        session = self._get_or_create_session()
        current_year = date.today().year

        all_rows: dict[str, list[dict]] = {
            "balance": [], "income": [], "cashflow": [],
        }

        # Make 2 calls: current year and 2 years ago
        for syear in [current_year, current_year - 2]:
            url = _XBRL_URL.format(
                base=_MOPSOV_BASE,
                co_id=identifier,
                syear=syear,
                sseason=4,
            )
            try:
                r = session.get(url, timeout=30)
                if len(r.content) < 10000:
                    logger.debug("XBRL response too small for %s/%s: %d bytes", identifier, syear, len(r.content))
                    continue

                dfs = pd.read_html(StringIO(r.text))
                if len(dfs) < 3:
                    continue

                # Parse tables 0 (balance), 1 (income), 2 (cashflow)
                for table_idx, (stmt_type, field_map) in _TABLE_CONFIG.items():
                    if table_idx >= len(dfs):
                        continue
                    rows = self._parse_xbrl_table(
                        dfs[table_idx], field_map, identifier, stmt_type,
                    )
                    all_rows[stmt_type].extend(rows)

                logger.info(
                    "XBRL %s/%d: parsed %d balance + %d income + %d cashflow rows",
                    identifier, syear,
                    sum(1 for r in all_rows["balance"] if str(syear) in r.get("report_date", "")),
                    sum(1 for r in all_rows["income"] if str(syear) in r.get("report_date", "")),
                    sum(1 for r in all_rows["cashflow"] if str(syear) in r.get("report_date", "")),
                )

            except Exception as exc:
                logger.warning("XBRL fetch failed for %s/%s: %s", identifier, syear, exc)

            time.sleep(_REQUEST_DELAY_S)

        # Convert to DataFrames and deduplicate
        result: dict[str, pd.DataFrame] = {}
        for stmt_type, rows in all_rows.items():
            if not rows:
                result[stmt_type] = pd.DataFrame()
                continue
            df = pd.DataFrame(rows)
            for col in ("filing_date", "report_date"):
                if col in df.columns:
                    df[col] = pd.to_datetime(df[col], errors="coerce")
            df = df.drop_duplicates(
                subset=["canonical_name", "report_date"], keep="first",
            )
            result[stmt_type] = df
            logger.info(
                "MOPS XBRL %s for %s: %d rows across %d periods",
                stmt_type, identifier, len(df),
                df["report_date"].nunique(),
            )

        return result

    def _parse_xbrl_table(
        self,
        df: pd.DataFrame,
        field_map: dict[str, str],
        identifier: str,
        statement_type: str,
    ) -> list[dict[str, Any]]:
        """Parse an XBRL inline HTML table into canonical rows.

        The table has columns:
        - Col 0: XBRL concept code (e.g. "1100", "4000", "AAAA")
        - Col 1: Accounting title (Chinese + English)
        - Col 2+: Value columns (current year, prior year, etc.)

        The column headers contain date strings like "2025/12/31" or
        "2024/1/1To12/31" that tell us which fiscal period each column
        represents.
        """
        if df.shape[0] < 3 or df.shape[1] < 3:
            return []

        # Extract year from column headers
        years: list[int] = []
        for col in df.columns[2:]:
            col_str = str(col[-1]) if isinstance(col, tuple) else str(col)
            # Match patterns like "2025/12/31" or "2024/1/1To12/31"
            year_match = re.search(r"(\d{4})/", col_str)
            if year_match:
                y = int(year_match.group(1))
                if y not in years:
                    years.append(y)

        if not years:
            return []

        rows: list[dict[str, Any]] = []
        for idx in range(df.shape[0]):
            code = str(df.iloc[idx, 0]).strip()
            if code == "nan":
                continue

            # Remove .0 from float codes (pandas may parse "4000" as 4000.0)
            code = code.replace(".0", "")

            canonical = field_map.get(code)
            if not canonical:
                continue

            # Extract values for each year column
            for yi, year in enumerate(years):
                col_idx = 2 + yi
                if col_idx >= df.shape[1]:
                    break

                val = df.iloc[idx, col_idx]
                value = _parse_number(str(val))
                if value is None:
                    continue

                report_date = f"{year}-12-31"

                rows.append({
                    "canonical_name": canonical,
                    "value": value,
                    "report_date": report_date,
                    "filing_date": report_date,
                    "source": "mops_xbrl",
                    "market_id": "tw_mops",
                    "identifier": identifier,
                    "xbrl_code": code,
                })

        return rows

    # ------------------------------------------------------------------
    # Fallback: JSON API (annual data)
    # ------------------------------------------------------------------

    def _fetch_json_fallback(
        self, identifier: str, statement_type: str,
    ) -> pd.DataFrame:
        """Fetch annual data from the new MOPS JSON API as fallback."""
        endpoint, field_map = _JSON_STATEMENT_CONFIG.get(
            statement_type, ("t164sb04", _JSON_INCOME_MAP),
        )
        session = self._get_or_create_session()

        # Initialize JSON API session
        try:
            session.get(f"{_MOPS_BASE}/", timeout=15)
        except Exception:
            pass

        current_roc = _ce_to_roc(date.today().year)
        for year_offset in range(0, 3):
            roc_year = str(current_roc - year_offset)
            try:
                r = session.post(
                    f"{_MOPS_API}/{endpoint}",
                    data=json.dumps({
                        "companyId": identifier,
                        "dataType": "1",
                        "season": "4",
                        "year": roc_year,
                        "subsidiaryCompanyId": "",
                    }),
                    headers={
                        "Content-Type": "application/json",
                        "Origin": _MOPS_BASE,
                        "Referer": f"{_MOPS_BASE}/mops/",
                    },
                    timeout=20,
                )
                j = r.json()
                if j.get("code") == 200 and j.get("result"):
                    return self._parse_json_report(
                        j["result"], field_map, identifier,
                    )
            except Exception as exc:
                logger.debug("JSON fallback failed for %s: %s", identifier, exc)
            time.sleep(_REQUEST_DELAY_S)

        return pd.DataFrame()

    def _parse_json_report(
        self,
        result: dict[str, Any],
        field_map: dict[str, str],
        identifier: str,
    ) -> pd.DataFrame:
        """Parse MOPS JSON API reportList into canonical DataFrame."""
        report_list = result.get("reportList", [])
        if not report_list:
            return pd.DataFrame()

        titles = result.get("titles", [])
        years: list[int] = []
        stride = 1
        for t in titles[1:]:
            main = t.get("main", "")
            subs = t.get("sub", [])
            year_match = re.search(r"(\d{2,3})年", main)
            if year_match:
                years.append(_roc_to_ce(int(year_match.group(1))))
                if len(subs) > 1:
                    stride = len(subs)

        if len(years) < 2:
            cy = date.today().year
            years = [cy, cy - 1]

        rows: list[dict[str, Any]] = []
        for row in report_list:
            if not row or len(row) < 3:
                continue
            label = row[0].strip().lstrip("\u3000 \t")
            canonical = field_map.get(label)
            if not canonical:
                continue
            for yi, year in enumerate(years):
                val_idx = 1 + yi * stride
                if val_idx >= len(row):
                    continue
                value = _parse_number(row[val_idx])
                if value is None:
                    continue
                rows.append({
                    "canonical_name": canonical,
                    "value": value,
                    "report_date": f"{year}-12-31",
                    "filing_date": f"{year}-12-31",
                    "source": "mops_json_api",
                    "market_id": "tw_mops",
                    "identifier": identifier,
                })

        if not rows:
            return pd.DataFrame()
        df = pd.DataFrame(rows)
        for col in ("filing_date", "report_date"):
            df[col] = pd.to_datetime(df[col], errors="coerce")
        return df.drop_duplicates(subset=["canonical_name", "report_date"], keep="first")

    # ------------------------------------------------------------------
    # OHLCV / Peers / Executives (not provided by MOPS)
    # ------------------------------------------------------------------

    def get_quotes(self, identifier: str) -> pd.DataFrame:
        return pd.DataFrame()

    def get_peers(self, identifier: str) -> list[str]:
        return []

    def get_executives(self, identifier: str) -> list[dict[str, Any]]:
        return []
