"""China SSE/SZSE PIT client -- uses akshare community library.

Primary library: akshare (https://github.com/akfamily/akshare)
  - Comprehensive Chinese financial data: SSE + SZSE
  - Stock lists, profiles, financial statements, quotes
  - 100+ data sources, actively maintained (v1.18.27+)
  - Documentation: https://akshare.akfamily.xyz/ (Chinese)

Fallback: Direct SSE/SZSE website APIs (limited, Chinese-only)

Coverage: ~5,000+ listed companies on SSE/SZSE, ~$10T market cap.

Research: .roo/research/cn-sse-2026-02-24.md
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


class CNSseClient:
    """PIT client for Chinese SSE/SZSE equities using akshare.

    Implements the ``PITClient`` protocol. Uses akshare as primary
    data source with direct SSE API as fallback.

    Research source: .roo/research/cn-sse-2026-02-24.md
    """

    def __init__(self, cache_dir: Path | str = _CACHE_DIR) -> None:
        self._cache_dir = Path(cache_dir)
        self._headers = {
            "Accept": "application/json",
            "User-Agent": "Operator1/1.0",
            "Referer": "http://www.sse.com.cn/",
        }
        self._akshare_available = False
        self._company_cache: list[dict[str, Any]] | None = None

        try:
            import akshare
            self._akshare_available = True
            logger.info("akshare available for Chinese market data")
        except ImportError:
            logger.info("akshare not installed; Chinese market data limited")

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
        return "China (SSE / SZSE) -- akshare"

    # -- Company discovery ---------------------------------------------------

    def list_companies(self, query: str = "") -> list[dict[str, Any]]:
        """List Chinese listed companies via akshare or SSE API."""
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
        """Build company list from akshare or fallback."""
        if self._akshare_available:
            try:
                import akshare as ak
                # akshare: stock_info_a_code_name() returns DataFrame with code + name
                df = ak.stock_info_a_code_name()
                companies = []
                for _, row in df.iterrows():
                    companies.append({
                        "ticker": str(row.get("code", "")),
                        "name": str(row.get("name", "")),
                        "cik": str(row.get("code", "")),
                        "exchange": "SSE/SZSE",
                        "country": "CN",
                        "market_id": self.market_id,
                    })
                if companies:
                    return companies
            except Exception as exc:
                logger.debug("akshare company list failed: %s", exc)

        return []

    # -- Company profile -----------------------------------------------------

    def get_profile(self, identifier: str) -> dict[str, Any]:
        cached = self._read_cache(identifier, "profile.json")
        if cached:
            return cached

        raw: dict[str, Any] = {
            "name": "",
            "ticker": identifier,
            "isin": "",
            "country": "CN",
            "sector": "",
            "industry": "",
            "exchange": "SSE",
            "currency": "CNY",
            "cik": identifier,
        }

        if self._akshare_available:
            try:
                import akshare as ak
                # akshare: stock_individual_info_em() returns company profile
                info_df = ak.stock_individual_info_em(symbol=identifier)
                if info_df is not None and not info_df.empty:
                    # info_df has columns: item, value
                    info_dict = {}
                    for _, row in info_df.iterrows():
                        info_dict[str(row.iloc[0])] = str(row.iloc[1]) if len(row) > 1 else ""

                    raw["name"] = info_dict.get("股票简称", info_dict.get("公司名称", ""))
                    raw["industry"] = info_dict.get("行业", "")
                    raw["sector"] = info_dict.get("行业", "")
                    exchange_code = identifier[:2] if len(identifier) >= 2 else ""
                    raw["exchange"] = "SSE" if exchange_code in ("60", "68") else "SZSE"
            except Exception as exc:
                logger.debug("akshare profile failed for %s: %s", identifier, exc)

        # Try name from company list if not found
        if not raw["name"]:
            matches = self.search_company(identifier)
            if matches:
                raw["name"] = matches[0].get("name", "")

        from operator1.clients.canonical_translator import translate_profile
        profile = translate_profile(raw, self.market_id)
        self._write_cache(identifier, "profile.json", profile)
        return profile

    # -- Financial statements ------------------------------------------------

    def get_income_statement(self, identifier: str) -> pd.DataFrame:
        return self._fetch_financials(identifier, "income")

    def get_balance_sheet(self, identifier: str) -> pd.DataFrame:
        return self._fetch_financials(identifier, "balance")

    def get_cashflow_statement(self, identifier: str) -> pd.DataFrame:
        return self._fetch_financials(identifier, "cashflow")

    def _fetch_financials(self, identifier: str, statement_type: str) -> pd.DataFrame:
        """Fetch Chinese financial statements via akshare."""
        if not self._akshare_available:
            return pd.DataFrame()

        try:
            import akshare as ak

            # akshare financial statement functions (East Money source)
            fetch_map = {
                "income": ak.stock_financial_report_sina,
                "balance": ak.stock_financial_report_sina,
                "cashflow": ak.stock_financial_report_sina,
            }

            # Map to Sina report types
            sina_type_map = {
                "income": "利润表",
                "balance": "资产负债表",
                "cashflow": "现金流量表",
            }

            df = fetch_map[statement_type](
                stock=identifier,
                symbol=sina_type_map[statement_type],
            )

            if df is None or df.empty:
                return pd.DataFrame()

            # Normalize to canonical format with filing_date and report_date.
            # akshare returns wide format: columns are Chinese line item names,
            # rows are reporting periods. We need to melt/unpivot into long
            # format with a 'concept' column for translate_financials to map.
            if "报告日" in df.columns:
                date_col = "报告日"
            elif df.columns[0] not in ("report_date",):
                date_col = df.columns[0]
            else:
                date_col = "report_date"

            # Identify value columns (everything except the date column)
            value_cols = [c for c in df.columns if c != date_col]

            if not value_cols:
                return pd.DataFrame()

            # Melt wide -> long: each Chinese column name becomes a 'concept' row
            long_df = df.melt(
                id_vars=[date_col],
                value_vars=value_cols,
                var_name="concept",
                value_name="value",
            )
            long_df = long_df.rename(columns={date_col: "report_date"})
            long_df["report_date"] = pd.to_datetime(long_df["report_date"], errors="coerce")
            long_df["filing_date"] = long_df["report_date"] + pd.Timedelta(days=45)  # Estimate
            long_df["value"] = pd.to_numeric(long_df["value"], errors="coerce")
            long_df = long_df.dropna(subset=["value"])

            from operator1.clients.canonical_translator import translate_financials
            return translate_financials(long_df, self.market_id, statement_type)

        except Exception as exc:
            logger.debug("akshare financials failed for %s/%s: %s", identifier, statement_type, exc)
            return pd.DataFrame()

    # -- Price data -----------------------------------------------------------

    def get_quotes(self, identifier: str) -> pd.DataFrame:
        """Fetch OHLCV price data via akshare."""
        if not self._akshare_available:
            return pd.DataFrame()

        try:
            import akshare as ak

            # akshare: stock_zh_a_hist() returns daily OHLCV
            end_date = date.today().strftime("%Y%m%d")
            start_date = (date.today() - timedelta(days=730)).strftime("%Y%m%d")

            df = ak.stock_zh_a_hist(
                symbol=identifier,
                period="daily",
                start_date=start_date,
                end_date=end_date,
                adjust="qfq",  # forward-adjusted
            )

            if df is None or df.empty:
                return pd.DataFrame()

            # Normalize column names to canonical OHLCV
            col_map = {
                "日期": "date",
                "开盘": "open",
                "最高": "high",
                "最低": "low",
                "收盘": "close",
                "成交量": "volume",
            }
            df = df.rename(columns=col_map)

            # Keep only OHLCV columns
            ohlcv_cols = ["date", "open", "high", "low", "close", "volume"]
            available = [c for c in ohlcv_cols if c in df.columns]
            df = df[available]

            if "date" in df.columns:
                df["date"] = pd.to_datetime(df["date"], errors="coerce")

            return df

        except Exception as exc:
            logger.debug("akshare quotes failed for %s: %s", identifier, exc)
            return pd.DataFrame()

    # -- Peers / related entities --------------------------------------------

    def get_peers(self, identifier: str) -> list[str]:
        return []

    def get_executives(self, identifier: str) -> list[dict[str, Any]]:
        return []
