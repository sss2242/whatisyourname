"""Taiwan MOPS PIT client -- new JSON API (2026 redesign).

MOPS (Market Observation Post System) was redesigned as a Vue.js SPA in
early 2026.  The old form POST endpoints (``/mops/web/ajax_t163sb04``)
are WAF-blocked from outside Taiwan.

This wrapper uses the **new JSON API** discovered by reverse-engineering
the Vue.js bundle (``/mops/assets/index.js``).  The API is at
``/mops/api/`` and accepts JSON POST requests with ``curl_cffi`` Chrome
TLS impersonation (plain ``requests`` gets 403 from the WAF).

Endpoints:
    - ``t164sb04`` -- Consolidated income statement (綜合損益表)
    - ``t164sb03`` -- Consolidated balance sheet (資產負債表)
    - ``t164sb05`` -- Consolidated cash flow statement (現金流量表)
    - ``redirectToOld`` -- Bridge to legacy HTML endpoints on mopsov.twse.com.tw

API contract::

    POST /mops/api/t164sb04
    Content-Type: application/json

    {"companyId": "2330", "dataType": "1", "season": "4",
     "year": "113", "subsidiaryCompanyId": ""}

Response::

    {"code": 200, "message": "查詢成功",
     "result": {
       "reportType": "合併",
       "companyAbbreviation": "台積電",
       "titles": [...],
       "reportList": [
         ["營業收入合計", "3,809,054,272", "100.00", "2,894,307,699", "100.00"],
         ...
       ]
     }}

Coverage: ~1,700+ listed companies on TWSE/TPEX, ~$1.2T market cap.
No API key required.  No geo-blocking on the new API.
"""

from __future__ import annotations

import json
import logging
import re
import time
from datetime import date
from pathlib import Path
from typing import Any

import pandas as pd

logger = logging.getLogger(__name__)

_MOPS_BASE = "https://mops.twse.com.tw"
_MOPS_API = f"{_MOPS_BASE}/mops/api"
_TWSE_BASE = "https://www.twse.com.tw"
_CACHE_DIR = Path("cache/tw_mops")

# Rate limit: pause between API requests (seconds).
_REQUEST_DELAY_S = 1.0


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

# Statement type -> (API endpoint, field map)
_STATEMENT_CONFIG: dict[str, tuple[str, dict[str, str]]] = {
    "income": ("t164sb04", _INCOME_MAP),
    "balance": ("t164sb03", _BALANCE_MAP),
    "cashflow": ("t164sb05", _CASHFLOW_MAP),
}


# ---------------------------------------------------------------------------
# curl_cffi session helper
# ---------------------------------------------------------------------------

def _get_session():
    """Create a curl_cffi session with Chrome TLS fingerprint.

    MOPS requires Chrome TLS impersonation -- plain requests gets 403
    from the WAF.  The session also loads the root page to acquire
    a JSESSIONID cookie.
    """
    from curl_cffi import requests as cf_requests
    session = cf_requests.Session(impersonate="chrome")
    try:
        session.get(f"{_MOPS_BASE}/", timeout=15)
    except Exception as exc:
        logger.debug("MOPS: failed to load root page: %s", exc)
    return session


def _parse_number(value_str: str) -> float | None:
    """Parse a Chinese/MOPS number string into a float.

    Handles:
    - Comma-separated thousands: "3,809,054,272"
    - Negative in parentheses: "(1,234)"
    - Empty strings -> None
    - Percentage strings (ignored if no digit)
    """
    if not value_str or not value_str.strip():
        return None
    s = value_str.strip()
    # Remove parentheses (negative)
    negative = False
    if s.startswith("(") and s.endswith(")"):
        negative = True
        s = s[1:-1]
    # Remove commas
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
    """Point-in-time client for MOPS (Taiwanese equities) using new JSON API.

    Implements the ``PITClient`` protocol.  Uses the redesigned MOPS
    JSON API (2026) with ``curl_cffi`` Chrome TLS impersonation.

    The API returns the latest 2 years of annual financial data per call
    (current year vs prior year comparison).  This aligns with the
    pipeline's 2-year lookback window.

    Note: MOPS uses ROC calendar (year 113 = CE 2024).
    """

    def __init__(self, cache_dir: Path | str = _CACHE_DIR) -> None:
        self._cache_dir = Path(cache_dir)
        self._session = None

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
        path.write_text(json.dumps(data, default=str, indent=2, ensure_ascii=False), encoding="utf-8")

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

        # Retry with backoff for transient failures
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
    # Financial statements (new MOPS JSON API)
    # ------------------------------------------------------------------

    def get_income_statement(self, identifier: str) -> pd.DataFrame:
        return self._fetch_mops_financials(identifier, "income")

    def get_balance_sheet(self, identifier: str) -> pd.DataFrame:
        return self._fetch_mops_financials(identifier, "balance")

    def get_cashflow_statement(self, identifier: str) -> pd.DataFrame:
        return self._fetch_mops_financials(identifier, "cashflow")

    def _fetch_mops_financials(
        self, identifier: str, statement_type: str,
    ) -> pd.DataFrame:
        """Fetch financial statements from the new MOPS JSON API.

        The API returns the latest 2 fiscal years as a comparison table.
        Each row in ``reportList`` is:
            [label, current_year_amount, current_year_pct,
             prior_year_amount, prior_year_pct]

        We extract both years and produce canonical long-format output.
        """
        endpoint, field_map = _STATEMENT_CONFIG.get(
            statement_type, ("t164sb04", _INCOME_MAP),
        )

        session = self._get_or_create_session()
        current_roc = _ce_to_roc(date.today().year)

        # Try current year first, then prior year
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
                    "MOPS %s fetch failed for %s (year=%s): %s",
                    statement_type, identifier, roc_year, exc,
                )
            time.sleep(_REQUEST_DELAY_S)

        if result is None:
            logger.warning(
                "MOPS %s: no data for %s after trying %d years",
                statement_type, identifier, 3,
            )
            return pd.DataFrame()

        # Parse the reportList into canonical rows
        return self._parse_report_list(
            result, statement_type, field_map, identifier,
        )

    def _parse_report_list(
        self,
        result: dict[str, Any],
        statement_type: str,
        field_map: dict[str, str],
        identifier: str,
    ) -> pd.DataFrame:
        """Parse MOPS JSON reportList into canonical long-format DataFrame.

        The ``titles`` structure tells us which fiscal years the columns
        represent.  Typically::

            titles[1].main = "114年度" (current year, ROC)
            titles[2].main = "113年度" (prior year, ROC)

        For balance sheets, titles may show dates like "114年12月31日".
        """
        report_list = result.get("reportList", [])
        if not report_list:
            return pd.DataFrame()

        # Extract year information from titles
        titles = result.get("titles", [])
        years: list[int] = []
        # Count sub-columns per year to determine stride
        # Income/balance: each year has [金額, %] -> stride 2
        # Cashflow: each year has [金額] only -> stride 1
        stride = 1
        for t in titles[1:]:  # skip first title (header label)
            main = t.get("main", "")
            subs = t.get("sub", [])
            # Extract ROC year from patterns like "114年度" or "114年12月31日"
            year_match = re.search(r"(\d{2,3})年", main)
            if year_match:
                roc = int(year_match.group(1))
                years.append(_roc_to_ce(roc))
                # Count sub-columns (金額 and optionally %)
                if len(subs) > 1:
                    stride = len(subs)

        if len(years) < 2:
            # Fallback: assume current year and prior year
            cy = date.today().year
            years = [cy, cy - 1]

        rows: list[dict[str, Any]] = []
        for row in report_list:
            if not row or len(row) < 3:
                continue

            label = row[0].strip()
            # Remove leading whitespace/indentation characters
            label = label.lstrip("\u3000 \t")

            canonical = field_map.get(label)
            if not canonical:
                continue

            # Extract values for each year
            # Income/balance: [label, y1_amount, y1_pct, y2_amount, y2_pct] (stride=2)
            # Cashflow:       [label, y1_amount, y2_amount]                 (stride=1)
            for yi, year in enumerate(years):
                val_idx = 1 + yi * stride
                if val_idx >= len(row):
                    continue

                value = _parse_number(row[val_idx])
                if value is None:
                    continue

                # Determine report_date from year
                if statement_type == "balance":
                    report_date = f"{year}-12-31"
                else:
                    report_date = f"{year}-12-31"

                rows.append({
                    "canonical_name": canonical,
                    "value": value,
                    "report_date": report_date,
                    "filing_date": report_date,
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

        # Deduplicate: keep first occurrence per (canonical_name, report_date)
        df = df.drop_duplicates(
            subset=["canonical_name", "report_date"], keep="first",
        )

        logger.info(
            "MOPS %s for %s: %d canonical rows across %d years",
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
