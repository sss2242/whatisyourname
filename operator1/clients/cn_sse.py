"""China SSE/SZSE PIT client -- uses baostock + yfinance.

Primary library: baostock (https://github.com/baostock/baostock)
  - Free, no API key, works globally (no geo-blocking)
  - Stock profiles, financial ratios, balance sheet ratios
  - Covers SSE + SZSE (~5,000+ companies)

Supplement: yfinance (profile enrichment: name, sector, industry, market_cap)

OHLCV: handled separately via ohlcv_baostock.py + ohlcv_provider.py

Coverage: ~5,000+ listed companies on SSE/SZSE, ~$10T market cap.
"""

from __future__ import annotations

import json
import logging
from datetime import date, timedelta
from pathlib import Path
from typing import Any

import pandas as pd

from operator1.http_utils import cached_get, HTTPError

logger = logging.getLogger(__name__)

_CACHE_DIR = Path("cache/cn_sse")


def _to_baostock_code(ticker: str) -> str:
    """Convert plain ticker to baostock format (sh.XXXXXX or sz.XXXXXX)."""
    ticker = ticker.strip()
    if ticker.startswith(("sh.", "sz.")):
        return ticker
    if ticker.upper().endswith(".SS"):
        return f"sh.{ticker[:-3]}"
    if ticker.upper().endswith(".SZ"):
        return f"sz.{ticker[:-3]}"
    if ticker.startswith("6"):
        return f"sh.{ticker}"
    elif ticker.startswith(("0", "3")):
        return f"sz.{ticker}"
    return f"sh.{ticker}"


def _to_yfinance_ticker(ticker: str) -> str:
    """Convert plain ticker to yfinance format (XXXXXX.SS or XXXXXX.SZ)."""
    ticker = ticker.strip()
    if ticker.endswith((".SS", ".SZ")):
        return ticker
    if ticker.startswith("sh."):
        return f"{ticker[3:]}.SS"
    if ticker.startswith("sz."):
        return f"{ticker[3:]}.SZ"
    if ticker.startswith("6"):
        return f"{ticker}.SS"
    elif ticker.startswith(("0", "3")):
        return f"{ticker}.SZ"
    return f"{ticker}.SS"


class CNSseClient:
    """PIT client for Chinese SSE/SZSE equities using baostock + yfinance.

    Implements the ``PITClient`` protocol. Uses baostock for financial
    ratios and yfinance for profile enrichment (name, sector, industry).
    """

    def __init__(self, cache_dir: Path | str = _CACHE_DIR) -> None:
        self._cache_dir = Path(cache_dir)
        self._headers = {
            "Accept": "application/json",
            "User-Agent": "Operator1/1.0",
            "Referer": "http://www.sse.com.cn/",
        }
        self._baostock_available = False
        self._company_cache: list[dict[str, Any]] | None = None

        try:
            import baostock
            self._baostock_available = True
            logger.info("baostock available for Chinese market data")
        except ImportError:
            logger.info("baostock not installed; Chinese market data limited")

    def _cache_path(self, identifier: str, fn: str) -> Path:
        return self._cache_dir / identifier.upper() / fn

    def _read_cache(self, identifier: str, fn: str) -> dict | None:
        p = self._cache_path(identifier, fn)
        if not p.exists():
            return None
        try:
            if fn == "profile.json" and (date.today() - date.fromtimestamp(p.stat().st_mtime)).days > 7:
                return None
            return json.loads(p.read_text(encoding="utf-8"))
        except Exception:
            return None

    def _write_cache(self, identifier: str, fn: str, data: dict) -> None:
        p = self._cache_path(identifier, fn)
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(json.dumps(data, default=str, indent=2), encoding="utf-8")

    @property
    def market_id(self) -> str:
        return "cn_sse"

    @property
    def market_name(self) -> str:
        return "China (SSE / SZSE) -- baostock + yfinance"

    # -- Company discovery ---------------------------------------------------

    def list_companies(self, query: str = "") -> list[dict[str, Any]]:
        """List Chinese listed companies via baostock."""
        if self._company_cache is None:
            self._company_cache = self._fetch_company_list()

        if not query:
            return self._company_cache

        q = query.lower()
        return [
            c for c in self._company_cache
            if q in c.get("ticker", "").lower()
            or q in c.get("name", "").lower()
        ]

    def search_company(self, name: str) -> list[dict[str, Any]]:
        return self.list_companies(query=name)

    def _fetch_company_list(self) -> list[dict[str, Any]]:
        """Build company list from baostock."""
        if not self._baostock_available:
            return []

        try:
            import baostock as bs
            lg = bs.login()
            if lg.error_code != "0":
                return []

            rs = bs.query_stock_basic()
            companies = []
            while rs.next():
                row = rs.get_row_data()
                # row: [code, code_name, ipoDate, outDate, type, status]
                if len(row) >= 2:
                    code = row[0]  # e.g. "sh.600519"
                    name = row[1]
                    ticker = code.split(".")[-1] if "." in code else code
                    companies.append({
                        "ticker": ticker,
                        "name": name,
                        "cik": ticker,
                        "exchange": "SSE" if code.startswith("sh") else "SZSE",
                        "country": "CN",
                        "market_id": self.market_id,
                    })
            bs.logout()
            return companies
        except Exception as exc:
            logger.debug("baostock company list failed: %s", exc)
            return []

    # -- Company profile -----------------------------------------------------

    def get_profile(self, identifier: str) -> dict[str, Any]:
        cached = self._read_cache(identifier, "profile.json")
        if cached:
            return cached

        exchange_code = identifier[:1] if len(identifier) >= 1 else ""
        exchange = "SSE" if exchange_code == "6" else "SZSE"

        raw: dict[str, Any] = {
            "name": "",
            "ticker": identifier,
            "isin": "",
            "country": "CN",
            "sector": "",
            "industry": "",
            "sub_industry": "",
            "exchange": exchange,
            "currency": "CNY",
            "cik": identifier,
            "market_cap": "",
            "shares_outstanding": "",
            "lei": "",
        }

        # Primary: baostock for basic info + financial ratios
        if self._baostock_available:
            self._enrich_from_baostock(identifier, raw)

        # Supplement: yfinance for name, sector, industry, market_cap
        self._enrich_from_yfinance(identifier, raw)

        # Try name from company list if still empty
        if not raw["name"]:
            matches = self.search_company(identifier)
            if matches:
                raw["name"] = matches[0].get("name", "")

        from operator1.clients.canonical_translator import translate_profile
        profile = translate_profile(raw, self.market_id)
        self._write_cache(identifier, "profile.json", profile)
        return profile

    def _enrich_from_baostock(self, identifier: str, raw: dict) -> None:
        """Enrich profile with baostock data (basic info + financial ratios)."""
        try:
            import baostock as bs
            bs_code = _to_baostock_code(identifier)

            lg = bs.login()
            if lg.error_code != "0":
                return

            # Basic stock info
            rs = bs.query_stock_basic(code=bs_code)
            while rs.next():
                row = rs.get_row_data()
                if len(row) >= 2 and not raw.get("name"):
                    raw["name"] = row[1]

            # Profit data (latest quarter)
            current_year = date.today().year
            for year in [current_year, current_year - 1]:
                for quarter in [4, 3, 2, 1]:
                    rs = bs.query_profit_data(code=bs_code, year=year, quarter=quarter)
                    rows = []
                    while rs.next():
                        rows.append(rs.get_row_data())
                    if rows:
                        row = rows[0]
                        fields = rs.fields
                        data = dict(zip(fields, row))
                        if data.get("totalShare"):
                            raw["shares_outstanding"] = data["totalShare"]
                        break
                if raw.get("shares_outstanding"):
                    break

            bs.logout()

        except Exception as exc:
            logger.debug("baostock profile enrichment failed for %s: %s", identifier, exc)
            try:
                bs.logout()
            except Exception:
                pass

    def _enrich_from_yfinance(self, identifier: str, raw: dict) -> None:
        """Enrich profile with yfinance data (name, sector, industry, market_cap)."""
        try:
            import yfinance as yf
            yf_ticker = _to_yfinance_ticker(identifier)
            t = yf.Ticker(yf_ticker)
            info = t.info or {}

            if not raw.get("name") or raw["name"] == "":
                raw["name"] = info.get("longName") or info.get("shortName") or ""
            if not raw.get("sector"):
                raw["sector"] = info.get("sector", "")
            if not raw.get("industry"):
                raw["industry"] = info.get("industry", "")
            if not raw.get("market_cap"):
                raw["market_cap"] = str(info.get("marketCap", ""))
            if not raw.get("shares_outstanding") or raw["shares_outstanding"] == "":
                shares = info.get("sharesOutstanding")
                if shares:
                    raw["shares_outstanding"] = str(shares)

        except Exception as exc:
            logger.debug("yfinance profile enrichment failed for %s: %s", identifier, exc)

    # -- Financial statements ------------------------------------------------

    def get_income_statement(self, identifier: str) -> pd.DataFrame:
        return self._fetch_financials(identifier, "income")

    def get_balance_sheet(self, identifier: str) -> pd.DataFrame:
        return self._fetch_financials(identifier, "balance")

    def get_cashflow_statement(self, identifier: str) -> pd.DataFrame:
        return self._fetch_financials(identifier, "cashflow")

    def _fetch_financials(self, identifier: str, statement_type: str) -> pd.DataFrame:
        """Fetch Chinese financial ratios via baostock.

        Note: baostock provides financial ratios (margins, ratios, EPS)
        rather than raw line items. For full financial statements,
        yfinance can supplement via Ticker.financials / .balance_sheet.
        """
        if not self._baostock_available:
            return pd.DataFrame()

        try:
            import baostock as bs
            bs_code = _to_baostock_code(identifier)

            lg = bs.login()
            if lg.error_code != "0":
                return pd.DataFrame()

            current_year = date.today().year
            all_rows = []

            query_fn = {
                "income": bs.query_profit_data,
                "balance": bs.query_balance_data,
                "cashflow": bs.query_cash_flow_data,
            }.get(statement_type)

            if not query_fn:
                bs.logout()
                return pd.DataFrame()

            # Fetch last 3 years of quarterly data
            for year in range(current_year - 2, current_year + 1):
                for quarter in [1, 2, 3, 4]:
                    rs = query_fn(code=bs_code, year=year, quarter=quarter)
                    while rs.next():
                        row = rs.get_row_data()
                        row_dict = dict(zip(rs.fields, row))
                        all_rows.append(row_dict)

            bs.logout()

            if not all_rows:
                return pd.DataFrame()

            df = pd.DataFrame(all_rows)

            # Add filing/report dates from statDate
            if "statDate" in df.columns:
                df["report_date"] = pd.to_datetime(df["statDate"], errors="coerce")
                df["filing_date"] = df["report_date"] + pd.Timedelta(days=45)

            if "pubDate" in df.columns:
                df["filing_date"] = pd.to_datetime(df["pubDate"], errors="coerce")

            from operator1.clients.canonical_translator import translate_financials
            return translate_financials(df, self.market_id, statement_type)

        except Exception as exc:
            logger.debug("baostock financials failed for %s/%s: %s", identifier, statement_type, exc)
            try:
                bs.logout()
            except Exception:
                pass
            return pd.DataFrame()

    # -- Price data -----------------------------------------------------------

    def get_quotes(self, identifier: str) -> pd.DataFrame:
        """SSE does not provide OHLCV data. Handled by ohlcv_provider."""
        return pd.DataFrame()

    # -- Peers / related entities --------------------------------------------

    def get_peers(self, identifier: str) -> list[str]:
        return []

    def get_executives(self, identifier: str) -> list[dict[str, Any]]:
        return []
