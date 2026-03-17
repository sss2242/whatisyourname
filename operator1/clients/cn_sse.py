"""China SSE/SZSE PIT client -- akshare (primary) + baostock (fallback).

Primary financials: akshare (via Sina Finance API)
  - Returns full raw line items (revenue, total_assets, cash, etc.)
  - Includes 公告日期 (announcement date) for true PIT filing dates
  - 100 quarterly rows per statement (25 years of history)
  - Works globally (Sina Finance is not geo-blocked)
  - Chinese field names mapped to canonical schema

Fallback financials: baostock
  - Returns financial ratios (currentRatio, gpMargin, etc.)
  - Also works globally, covers SSE + SZSE (~8,600 securities)

Company search: baostock query_stock_basic (8,600+ securities)
Profile: baostock (basic info) + yfinance (sector/industry supplement)

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

# ---------------------------------------------------------------------------
# Chinese -> canonical field name mapping for akshare/Sina data
# ---------------------------------------------------------------------------

_INCOME_FIELD_MAP: dict[str, str] = {
    "营业收入": "revenue",
    "营业总收入": "total_revenue",
    "营业成本": "cost_of_revenue",
    "营业总成本": "total_operating_cost",
    "净利润": "net_income",
    "归属于母公司所有者的净利润": "net_income_to_parent",
    "营业利润": "operating_income",
    "利润总额": "income_before_tax",
    "所得税费用": "income_tax_expense",
    "销售费用": "selling_expense",
    "管理费用": "admin_expense",
    "研发费用": "rd_expense",
    "财务费用": "finance_expense",
    "营业税金及附加": "tax_and_surcharge",
    "基本每股收益": "eps_basic",
    "稀释每股收益": "eps_diluted",
    "少数股东损益": "minority_interest_income",
}

_BALANCE_FIELD_MAP: dict[str, str] = {
    "货币资金": "cash_and_equivalents",
    "交易性金融资产": "trading_financial_assets",
    "应收票据及应收账款": "accounts_receivable",
    "应收账款": "accounts_receivable_net",
    "预付款项": "prepayments",
    "存货": "inventory",
    "流动资产合计": "total_current_assets",
    "非流动资产合计": "total_non_current_assets",
    "固定资产净额": "net_fixed_assets",
    "无形资产": "intangible_assets",
    "商誉": "goodwill",
    "负债和所有者权益(或股东权益)总计": "total_assets",
    "短期借款": "short_term_debt",
    "应付票据及应付账款": "accounts_payable",
    "预收款项": "advance_receipts",
    "合同负债": "contract_liabilities",
    "流动负债合计": "total_current_liabilities",
    "长期借款": "long_term_debt",
    "非流动负债合计": "total_non_current_liabilities",
    "负债合计": "total_liabilities",
    "归属于母公司股东权益合计": "total_equity_to_parent",
    "所有者权益(或股东权益)合计": "total_equity",
    "少数股东权益": "minority_interest",
}

_CASHFLOW_FIELD_MAP: dict[str, str] = {
    "销售商品、提供劳务收到的现金": "cash_from_sales",
    "经营活动现金流入小计": "total_operating_cash_inflow",
    "经营活动现金流出小计": "total_operating_cash_outflow",
    "经营活动产生的现金流量净额": "operating_cash_flow",
    "购建固定资产、无形资产和其他长期资产所支付的现金": "capex",
    "投资活动现金流入小计": "total_investing_cash_inflow",
    "投资活动现金流出小计": "total_investing_cash_outflow",
    "投资活动产生的现金流量净额": "investing_cash_flow",
    "筹资活动现金流入小计": "total_financing_cash_inflow",
    "筹资活动现金流出小计": "total_financing_cash_outflow",
    "筹资活动产生的现金流量净额": "financing_cash_flow",
}

_STATEMENT_TYPE_MAP: dict[str, tuple[str, dict[str, str]]] = {
    "income": ("利润表", _INCOME_FIELD_MAP),
    "balance": ("资产负债表", _BALANCE_FIELD_MAP),
    "cashflow": ("现金流量表", _CASHFLOW_FIELD_MAP),
}


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
        """Fetch Chinese financial statements -- akshare primary, baostock fallback.

        Primary: akshare (via Sina Finance API)
          - Returns full raw line items (revenue, total_assets, cash, etc.)
          - Includes 公告日期 (announcement date) as true PIT filing date
          - Chinese field names mapped to canonical schema via _*_FIELD_MAP
          - Works globally (Sina Finance is not geo-blocked)

        Fallback: baostock
          - Returns financial ratios (currentRatio, gpMargin, etc.)
          - Less complete but still useful for derived variables
        """
        # Path 1: akshare (full raw line items with PIT dates)
        df = self._fetch_financials_akshare(identifier, statement_type)
        if df is not None and not df.empty:
            return df

        # Path 2: baostock fallback (ratios only)
        return self._fetch_financials_baostock(identifier, statement_type)

    def _fetch_financials_akshare(self, identifier: str, statement_type: str) -> pd.DataFrame:
        """Fetch full financial statements via akshare/Sina Finance.

        Returns a long-format DataFrame with canonical_name, value,
        report_date, and filing_date columns ready for the pipeline.
        """
        if statement_type not in _STATEMENT_TYPE_MAP:
            return pd.DataFrame()

        sina_symbol, field_map = _STATEMENT_TYPE_MAP[statement_type]

        try:
            import akshare as ak

            # akshare expects plain 6-digit ticker (no prefix)
            ticker = identifier.strip()
            for prefix in ("sh.", "sz.", "sh", "sz"):
                if ticker.startswith(prefix):
                    ticker = ticker[len(prefix):]
                    break
            ticker = ticker.split(".")[0]

            df = ak.stock_financial_report_sina(stock=ticker, symbol=sina_symbol)
            if df is None or df.empty:
                return pd.DataFrame()

            # Filter to last 2 years
            if "报告日" in df.columns:
                df["report_date"] = pd.to_datetime(df["报告日"].astype(str), format="%Y%m%d", errors="coerce")
                cutoff = pd.Timestamp.now() - pd.Timedelta(days=730)
                df = df[df["report_date"].notna() & (df["report_date"] >= cutoff)]

            # Extract filing_date from 公告日期 (announcement date = PIT date)
            if "公告日期" in df.columns:
                df["filing_date"] = pd.to_datetime(
                    df["公告日期"].astype(str), format="%Y%m%d", errors="coerce",
                )
            else:
                # Estimate: Chinese companies report within 1-2 months
                df["filing_date"] = df["report_date"] + pd.Timedelta(days=45)

            # Map Chinese field names to canonical names (long format)
            rows: list[dict] = []
            for _, record in df.iterrows():
                rd = record.get("report_date")
                fd = record.get("filing_date")
                for cn_name, canonical_name in field_map.items():
                    if cn_name in record.index:
                        val = record[cn_name]
                        if pd.notna(val):
                            try:
                                val = float(val)
                            except (ValueError, TypeError):
                                continue
                            rows.append({
                                "canonical_name": canonical_name,
                                "value": val,
                                "report_date": rd,
                                "filing_date": fd,
                            })

            if not rows:
                return pd.DataFrame()

            result = pd.DataFrame(rows)
            result["market_id"] = self.market_id
            result["currency"] = "CNY"
            result["statement_type"] = statement_type

            logger.info(
                "akshare %s/%s: %d rows (%d periods) from Sina Finance",
                identifier, statement_type, len(result),
                result["report_date"].nunique(),
            )
            return result

        except Exception as exc:
            logger.debug("akshare financials failed for %s/%s: %s", identifier, statement_type, exc)
            return pd.DataFrame()

    def _fetch_financials_baostock(self, identifier: str, statement_type: str) -> pd.DataFrame:
        """Fallback: fetch financial ratios via baostock.

        Returns ratios (not raw line items) with pubDate as filing_date.
        """
        if not self._baostock_available:
            return pd.DataFrame()

        try:
            import baostock as bs
            bs_code = _to_baostock_code(identifier)

            lg = bs.login()
            if lg.error_code != "0":
                logger.warning("baostock login failed: %s", lg.error_msg)
                return pd.DataFrame()

            query_fn = {
                "income": bs.query_profit_data,
                "balance": bs.query_balance_data,
                "cashflow": bs.query_cash_flow_data,
            }.get(statement_type)

            if not query_fn:
                bs.logout()
                return pd.DataFrame()

            current_year = date.today().year
            all_rows = []

            for year in range(current_year - 2, current_year + 1):
                for quarter in [1, 2, 3, 4]:
                    try:
                        rs = query_fn(code=bs_code, year=year, quarter=quarter)
                        while rs.next():
                            row = rs.get_row_data()
                            all_rows.append(dict(zip(rs.fields, row)))
                    except Exception:
                        continue

            bs.logout()

            if not all_rows:
                return pd.DataFrame()

            df = pd.DataFrame(all_rows)

            if "pubDate" in df.columns:
                df["filing_date"] = pd.to_datetime(df["pubDate"], errors="coerce")
            if "statDate" in df.columns:
                df["report_date"] = pd.to_datetime(df["statDate"], errors="coerce")
                if "filing_date" not in df.columns or df["filing_date"].isna().all():
                    df["filing_date"] = df["report_date"] + pd.Timedelta(days=45)

            skip_cols = {"code", "pubDate", "statDate", "report_date", "filing_date"}
            for col in df.columns:
                if col not in skip_cols:
                    df[col] = pd.to_numeric(df[col], errors="coerce")

            logger.info("baostock %s/%s: %d rows (fallback)", identifier, statement_type, len(df))

            from operator1.clients.canonical_translator import translate_financials
            return translate_financials(df, self.market_id, statement_type)

        except Exception as exc:
            logger.debug("baostock financials failed for %s/%s: %s", identifier, statement_type, exc)
            try:
                import baostock as bs
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
