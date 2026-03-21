"""Taiwan MOPS PIT client -- dual-source: quarterly HTML + annual JSON API.

MOPS (Market Observation Post System) was redesigned as a Vue.js SPA in
early 2026.  The old form POST endpoints on ``mops.twse.com.tw`` are
WAF-blocked.  However, the legacy backend at ``mopsov.twse.com.tw`` still
serves quarterly HTML tables via the old form POST interface.

**Primary source** (quarterly data, HKEX-style session pattern):
    ``mopsov.twse.com.tw/mops/web/ajax_t164sb04`` etc.
    - GET index page first to acquire ``jcsession`` cookie
    - Form POST with ``co_id``, ``year`` (ROC), ``season`` (1-4)
    - Returns HTML tables parsed by ``pd.read_html()``
    - 8 quarters (2 years) of data per company

**Fallback** (annual data, Vue.js API):
    ``mops.twse.com.tw/mops/api/t164sb04`` etc.
    - JSON POST with ``companyId``, ``dataType``, ``season``, ``year``
    - Returns structured JSON with ``reportList`` arrays
    - Latest 2 annual years per call

Both sources require ``curl_cffi`` with Chrome TLS impersonation.
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

# Rate limit: pause between API requests (seconds).
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
# Chinese -> canonical field name mapping
# ---------------------------------------------------------------------------

# Income statement (綜合損益表)
_INCOME_MAP: dict[str, str] = {
    "營業收入合計": "revenue",
    "營業成本合計": "cost_of_revenue",
    "營業毛利（毛損）": "gross_profit",
    "營業毛利（毛損）淨額": "gross_profit",
    "推銷費用": "selling_expenses",
    "管理費用": "admin_expenses",
    "研究發展費用": "rd_expenses",
    "營業費用合計": "sga_expenses",
    "營業利益（損失）": "operating_income",
    "利息收入": "interest_income",
    "財務成本淨額": "interest_expense",
    "財務成本": "interest_expense",
    "稅前淨利（淨損）": "ebit",
    "所得稅費用（利益）合計": "taxes",
    "所得稅費用（利益）": "taxes",
    "繼續營業單位本期淨利（淨損）": "net_income",
    "本期淨利（淨損）": "net_income",
    "母公司業主（淨利∕損）": "net_income_attributable",
    "基本每股盈餘": "eps",
    "稀釋每股盈餘": "eps_diluted",
    "本期綜合損益總額": "comprehensive_income",
    "營業外收入及支出合計": "non_operating_income",
    "折舊費用": "depreciation",
    "攤銷費用": "amortization",
}

# Balance sheet (資產負債表)
_BALANCE_MAP: dict[str, str] = {
    "現金及約當現金": "cash_and_equivalents",
    "應收帳款淨額": "receivables",
    "應收票據及帳款淨額": "receivables",
    "存貨": "inventory",
    "流動資產合計": "current_assets",
    "非流動資產合計": "noncurrent_assets",
    "資產總計": "total_assets",
    "資產總額": "total_assets",
    "短期借款": "short_term_debt",
    "應付帳款": "payables",
    "應付票據及帳款": "payables",
    "流動負債合計": "current_liabilities",
    "非流動負債合計": "noncurrent_liabilities",
    "負債總計": "total_liabilities",
    "負債總額": "total_liabilities",
    "長期借款": "long_term_debt",
    "應付公司債": "bonds_payable",
    "股本合計": "share_capital",
    "保留盈餘合計": "retained_earnings",
    "歸屬於母公司業主之權益合計": "equity_attributable",
    "權益總計": "total_equity",
    "權益總額": "total_equity",
    "負債及權益總計": "total_liabilities_and_equity",
    "商譽": "goodwill",
    "無形資產": "intangible_assets",
    "不動產、廠房及設備": "property_plant_equipment",
}

# Cash flow statement (現金流量表)
_CASHFLOW_MAP: dict[str, str] = {
    "繼續營業單位稅前淨利（淨損）": "pretax_income",
    "本期稅前淨利（淨損）": "pretax_income",
    "折舊費用": "depreciation_cf",
    "攤銷費用": "amortization_cf",
    "營業活動之淨現金流入（流出）": "operating_cash_flow",
    "投資活動之淨現金流入（流出）": "investing_cf",
    "籌資活動之淨現金流入（流出）": "financing_cf",
    "取得不動產、廠房及設備": "capex",
    "發放現金股利": "dividends_paid",
    "本期現金及約當現金增加（減少）數": "net_change_in_cash",
    "期末現金及約當現金餘額": "ending_cash",
    "期初現金及約當現金餘額": "beginning_cash",
}

# Statement type -> (consolidated endpoint, individual endpoint, field map)
_STATEMENT_CONFIG: dict[str, tuple[str, str, dict[str, str]]] = {
    "income": ("ajax_t164sb04", "ajax_t163sb04", _INCOME_MAP),
    "balance": ("ajax_t164sb03", "ajax_t163sb05", _BALANCE_MAP),
    "cashflow": ("ajax_t164sb20", "ajax_t163sb20", _CASHFLOW_MAP),
}

# JSON API endpoint mapping (fallback)
_JSON_API_ENDPOINTS: dict[str, str] = {
    "income": "t164sb04",
    "balance": "t164sb03",
    "cashflow": "t164sb05",
}


# ---------------------------------------------------------------------------
# curl_cffi session helpers
# ---------------------------------------------------------------------------

def _get_mopsov_session():
    """Create a curl_cffi session for mopsov.twse.com.tw (legacy HTML API).

    HKEX-style pattern: GET the index page first to acquire a session
    cookie (``jcsession``), then POST to ajax endpoints.
    """
    from curl_cffi import requests as cf_requests
    session = cf_requests.Session(impersonate="chrome")
    try:
        session.get(f"{_MOPSOV_BASE}/mops/web/index", timeout=15)
        logger.debug("MOPSOV session initialized, cookies: %s", dict(session.cookies))
    except Exception as exc:
        logger.debug("MOPSOV session init failed: %s", exc)
    return session


def _get_json_api_session():
    """Create a curl_cffi session for the new MOPS JSON API (fallback)."""
    from curl_cffi import requests as cf_requests
    session = cf_requests.Session(impersonate="chrome")
    try:
        session.get(f"{_MOPS_BASE}/", timeout=15)
    except Exception as exc:
        logger.debug("MOPS JSON API session init failed: %s", exc)
    return session


def _parse_number(value_str: str) -> float | None:
    """Parse a Chinese/MOPS number string into a float.

    Handles comma-separated thousands, negative in parentheses,
    and empty strings.
    """
    if not value_str or not value_str.strip():
        return None
    s = value_str.strip()
    negative = False
    if s.startswith("(") and s.endswith(")"):
        negative = True
        s = s[1:-1]
    s = s.replace(",", "")
    try:
        val = float(s)
        return -val if negative else val
    except (ValueError, TypeError):
        return None


# ---------------------------------------------------------------------------
# Main client class
# ---------------------------------------------------------------------------

class TWMopsClient:
    """Point-in-time client for MOPS (Taiwanese equities).

    Implements the ``PITClient`` protocol.  Uses a dual-source approach:

    **Primary**: ``mopsov.twse.com.tw`` legacy HTML API (quarterly data).
    Discovered via HKEX-style session probing -- the old MOPS server is
    still running behind the new Vue.js SPA and serves quarterly financial
    statements via form POST with proper session cookies.

    **Fallback**: ``mops.twse.com.tw/mops/api/`` JSON API (annual data).
    The redesigned MOPS exposes a JSON API that returns the latest 2 fiscal
    years as a comparison table.

    Both require ``curl_cffi`` with Chrome TLS impersonation.
    """

    def __init__(self, cache_dir: Path | str = _CACHE_DIR) -> None:
        self._cache_dir = Path(cache_dir)
        self._mopsov_session = None
        self._json_session = None

    def _get_or_create_mopsov_session(self):
        if self._mopsov_session is None:
            self._mopsov_session = _get_mopsov_session()
        return self._mopsov_session

    def _get_or_create_json_session(self):
        if self._json_session is None:
            self._json_session = _get_json_api_session()
        return self._json_session

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
    # Company listing and search (via TWSE JSON API, not geo-blocked)
    # ------------------------------------------------------------------

    def list_companies(self, query: str = "") -> list[dict[str, Any]]:
        """List Taiwanese companies from TWSE stock day all JSON API."""
        session = self._get_or_create_json_session()
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
            except Exception as exc:
                logger.debug(
                    "MOPS get_profile attempt %d/3 for %s failed: %s",
                    attempt, identifier, exc,
                )
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
    # Financial statements (dual source: quarterly HTML + annual JSON)
    # ------------------------------------------------------------------

    def get_income_statement(self, identifier: str) -> pd.DataFrame:
        return self._fetch_financials(identifier, "income")

    def get_balance_sheet(self, identifier: str) -> pd.DataFrame:
        return self._fetch_financials(identifier, "balance")

    def get_cashflow_statement(self, identifier: str) -> pd.DataFrame:
        return self._fetch_financials(identifier, "cashflow")

    def _fetch_financials(
        self, identifier: str, statement_type: str,
    ) -> pd.DataFrame:
        """Fetch financial statements using dual source strategy.

        Primary: mopsov.twse.com.tw quarterly HTML (HKEX-style session).
        Fallback: mops.twse.com.tw/mops/api/ annual JSON.
        """
        # Try primary: quarterly HTML from mopsov
        df = self._fetch_quarterly_html(identifier, statement_type)
        if not df.empty:
            return df

        # Fallback: annual JSON from new MOPS API
        logger.info(
            "MOPS quarterly HTML failed for %s/%s, falling back to JSON API",
            identifier, statement_type,
        )
        return self._fetch_annual_json(identifier, statement_type)

    # ------------------------------------------------------------------
    # Primary: Quarterly HTML from mopsov.twse.com.tw
    # ------------------------------------------------------------------

    def _fetch_quarterly_html(
        self, identifier: str, statement_type: str,
    ) -> pd.DataFrame:
        """Fetch quarterly financial data from the legacy mopsov server.

        Uses HKEX-style session pattern: GET index first for ``jcsession``
        cookie, then form POST to ajax endpoints for each quarter.
        """
        consolidated_ep, _individual_ep, field_map = _STATEMENT_CONFIG[statement_type]
        session = self._get_or_create_mopsov_session()

        current_year = date.today().year
        current_roc = _ce_to_roc(current_year)
        current_quarter = max(1, (date.today().month - 1) // 3)

        all_rows: list[dict[str, Any]] = []

        # Fetch 2 years of quarterly data (up to 8 quarters)
        for roc_year in range(current_roc, current_roc - 3, -1):
            for season in range(4, 0, -1):
                ce_year = _roc_to_ce(roc_year)
                # Skip future quarters
                if ce_year == current_year and season > current_quarter:
                    continue

                try:
                    r = session.post(
                        f"{_MOPSOV_BASE}/mops/web/{consolidated_ep}",
                        data={
                            "encodeURIComponent": "1",
                            "step": "1",
                            "firstin": "1",
                            "off": "1",
                            "TYPEK": "sii",
                            "co_id": identifier,
                            "year": str(roc_year),
                            "season": str(season),
                        },
                        headers={
                            "Content-Type": "application/x-www-form-urlencoded",
                            "Referer": f"{_MOPSOV_BASE}/mops/web/{consolidated_ep.replace('ajax_', '')}",
                        },
                        timeout=20,
                    )
                except Exception as exc:
                    logger.debug(
                        "MOPSOV %s failed for %s Y%sQ%s: %s",
                        statement_type, identifier, roc_year, season, exc,
                    )
                    time.sleep(_REQUEST_DELAY_S)
                    continue

                if not r.text or "<table" not in r.text[:5000].lower():
                    time.sleep(_REQUEST_DELAY_S)
                    continue

                if "PAGE CANNOT" in r.text[:500]:
                    logger.debug("MOPSOV WAF block for %s", identifier)
                    time.sleep(_REQUEST_DELAY_S)
                    continue

                # Extract period from HTML header
                period_match = re.search(
                    r"民國(\d{2,3})年第(\d)季", r.text[:2000],
                )
                if period_match:
                    actual_roc = int(period_match.group(1))
                    actual_q = int(period_match.group(2))
                    actual_ce = _roc_to_ce(actual_roc)
                else:
                    actual_ce = ce_year
                    actual_q = season

                # Determine report_date from quarter
                month_end = {1: "03-31", 2: "06-30", 3: "09-30", 4: "12-31"}
                report_date = f"{actual_ce}-{month_end[actual_q]}"

                # Parse HTML table
                rows = self._parse_html_table(
                    r.text, field_map, identifier, report_date,
                )
                all_rows.extend(rows)

                time.sleep(_REQUEST_DELAY_S)

                # Stop after 8 quarters
                if len(all_rows) > 0 and len(set(
                    r["report_date"] for r in all_rows
                )) >= 8:
                    break
            else:
                continue
            break

        if not all_rows:
            return pd.DataFrame()

        df = pd.DataFrame(all_rows)
        for col in ("filing_date", "report_date"):
            if col in df.columns:
                df[col] = pd.to_datetime(df[col], errors="coerce")

        # Deduplicate: keep first per (canonical_name, report_date)
        df = df.drop_duplicates(
            subset=["canonical_name", "report_date"], keep="first",
        )

        logger.info(
            "MOPS quarterly %s for %s: %d rows across %d periods",
            statement_type, identifier, len(df),
            df["report_date"].nunique() if not df.empty else 0,
        )
        return df

    def _parse_html_table(
        self,
        html: str,
        field_map: dict[str, str],
        identifier: str,
        report_date: str,
    ) -> list[dict[str, Any]]:
        """Parse a MOPS HTML table response into canonical rows."""
        rows: list[dict[str, Any]] = []
        try:
            dfs = pd.read_html(StringIO(html))
        except Exception:
            return rows

        for df in dfs:
            if df.shape[0] < 5 or df.shape[1] < 2:
                continue

            for _, row in df.iterrows():
                label = str(row.iloc[0]).strip() if pd.notna(row.iloc[0]) else ""
                # Strip indentation characters
                label = label.lstrip("\u3000 \t")

                canonical = field_map.get(label)
                if not canonical:
                    continue

                # Current period value is in column 1
                val = row.iloc[1] if df.shape[1] > 1 else None
                if pd.isna(val):
                    continue

                value = _parse_number(str(val))
                if value is None:
                    # pandas may have already parsed it as float
                    try:
                        value = float(val)
                        if np.isnan(value):
                            continue
                    except (ValueError, TypeError):
                        continue

                rows.append({
                    "canonical_name": canonical,
                    "value": value,
                    "report_date": report_date,
                    "filing_date": report_date,
                    "source": "mopsov_quarterly",
                    "market_id": "tw_mops",
                    "identifier": identifier,
                })

            # Use only the first large table (financial data)
            if rows:
                break

        return rows

    # ------------------------------------------------------------------
    # Fallback: Annual JSON from new MOPS API
    # ------------------------------------------------------------------

    def _fetch_annual_json(
        self, identifier: str, statement_type: str,
    ) -> pd.DataFrame:
        """Fetch annual financials from the new MOPS JSON API (fallback).

        Returns the latest 2 fiscal years as a comparison table.
        """
        endpoint = _JSON_API_ENDPOINTS.get(statement_type, "t164sb04")
        field_map = _STATEMENT_CONFIG[statement_type][2]
        session = self._get_or_create_json_session()
        current_roc = _ce_to_roc(date.today().year)

        result = None
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
                    result = j["result"]
                    break
            except Exception as exc:
                logger.debug(
                    "MOPS JSON %s failed for %s (year=%s): %s",
                    statement_type, identifier, roc_year, exc,
                )
            time.sleep(_REQUEST_DELAY_S)

        if result is None:
            return pd.DataFrame()

        return self._parse_json_report_list(
            result, statement_type, field_map, identifier,
        )

    def _parse_json_report_list(
        self,
        result: dict[str, Any],
        statement_type: str,
        field_map: dict[str, str],
        identifier: str,
    ) -> pd.DataFrame:
        """Parse MOPS JSON API reportList into canonical long-format DataFrame."""
        report_list = result.get("reportList", [])
        if not report_list:
            return pd.DataFrame()

        # Extract year info and column stride from titles
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
            if col in df.columns:
                df[col] = pd.to_datetime(df[col], errors="coerce")
        df = df.drop_duplicates(
            subset=["canonical_name", "report_date"], keep="first",
        )

        logger.info(
            "MOPS JSON %s for %s: %d rows across %d years",
            statement_type, identifier, len(df), len(years),
        )
        return df

    # ------------------------------------------------------------------
    # OHLCV (not provided by MOPS -- handled by ohlcv_provider)
    # ------------------------------------------------------------------

    def get_quotes(self, identifier: str) -> pd.DataFrame:
        return pd.DataFrame()

    def get_peers(self, identifier: str) -> list[str]:
        return []

    def get_executives(self, identifier: str) -> list[dict[str, Any]]:
        return []
