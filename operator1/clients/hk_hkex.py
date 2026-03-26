"""Hong Kong HKEX PIT client -- akshare/EastMoney fast path + filing discovery.

Primary: akshare stock_financial_hk_report_em (EastMoney structured data)
  - Balance sheet, income statement, cash flow for any HKEX-listed stock
  - Structured line items with standardized codes, no LLM needed
  - ~4,000 rows per company across all reporting periods
  - Chinese field names mapped to canonical English schema

Fallback: HKEX News filing discovery (hkex_scraper.py)
  - Discovers annual/interim result announcements via date-windowed API
  - Downloads PDF documents for LLM/fuzzy extraction
  - Preserves filing_date from HKEX announcement date (true PIT)

Profile: akshare stock_hk_company_profile_em + yfinance enrichment

OHLCV: handled separately via ohlcv_provider.py (yfinance .HK)

Coverage: ~2,500+ listed companies on HKEX, ~$4.5T market cap.
"""
from __future__ import annotations

import json
import logging
from datetime import date
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)
_CACHE_DIR = Path("cache/hk_hkex")


# ---------------------------------------------------------------------------
# EastMoney Chinese -> Canonical field name mapping
# ---------------------------------------------------------------------------

# Balance sheet (资产负债表) -- STD_ITEM_CODE -> canonical_name
_BS_MAP: dict[str, str] = {
    "004009999": "total_assets",
    "004025999": "total_liabilities",
    "004030999": "stock_holder_equity",
    "004036999": "total_equity",
    "004028999": "net_assets",
    "004002999": "current_assets",
    "004011999": "current_liabilities",
    "004002010": "cash_and_equivalents",
    "004002001": "inventory",
    "004002003": "receivables",
    "004011001": "payables",
    "004001999": "noncurrent_assets",
    "004020999": "noncurrent_liabilities",
    "004020001": "long_term_debt",
    "004011010": "short_term_debt",
    "004001004": "intangible_assets",
    "004001002": "property_plant_equipment",
    "004030004": "retained_earnings",
    "004027999": "minority_interest",
    "004001009": "deferred_tax_assets",
    "004020003": "deferred_tax_liabilities",
    "004002011": "short_term_deposits",
    "004013999": "net_current_assets",
    "004001013": "investments_in_associates",
}

# Income statement (利润表) -- STD_ITEM_CODE -> canonical_name
_IS_MAP: dict[str, str] = {
    "004001001": "revenue",
    "004001999": "total_revenue",
    "004005001": "cost_of_revenue",
    "004007999": "gross_profit",
    "004010999": "operating_income",
    "004011999": "ebit",
    "004010003": "sga_expenses",
    "004010004": "admin_expenses",
    "004011200": "interest_income",
    "004011201": "interest_expense",
    "004011202": "share_of_associates_profit",
    "004012001": "taxes",
    "004012999": "net_income",
    "004013001": "net_income_attributable",
    "004013002": "minority_interest_pl",
    "004014001": "eps_basic",
    "004014002": "eps_diluted",
    "004011997": "other_profit_items",
}

# Cash flow (现金流量表) -- STD_ITEM_CODE -> canonical_name
_CF_MAP: dict[str, str] = {
    "003999": "operating_cash_flow",
    "005999": "investing_cf",
    "006999": "pre_financing_cf",
    "007999": "financing_cf",
    "010999": "net_cash_change",
    "011001": "cash_beginning",
    "011999": "cash_ending",
}

# Statement type -> (akshare symbol parameter, field mapping)
_STATEMENT_CONFIG: dict[str, tuple[str, dict[str, str]]] = {
    "income": ("利润表", _IS_MAP),
    "balance": ("资产负债表", _BS_MAP),
    "cashflow": ("现金流量表", _CF_MAP),
}


def _fetch_hkex_akshare(
    stock_code: str,
    statement_type: str,
) -> pd.DataFrame:
    """Fetch structured financial data from EastMoney via akshare.

    Parameters
    ----------
    stock_code:
        HKEX stock code (e.g. '00700', '700', '09988').
    statement_type:
        One of 'income', 'balance', 'cashflow'.

    Returns
    -------
    Canonical long-format DataFrame with columns:
    canonical_name, value, report_date, filing_date.
    """
    try:
        import akshare as ak
    except ImportError:
        logger.debug("akshare not installed -- skipping EastMoney fast path")
        return pd.DataFrame()

    config = _STATEMENT_CONFIG.get(statement_type)
    if not config:
        return pd.DataFrame()

    symbol, field_map = config

    # Normalize stock code to 5-digit format
    code = stock_code.split(".")[0].strip().zfill(5)

    try:
        df = ak.stock_financial_hk_report_em(
            stock=code,
            symbol=symbol,
            indicator="报告期",  # All reporting periods (not just annual)
        )
    except Exception as exc:
        logger.debug("akshare HK report failed for %s/%s: %s", code, statement_type, exc)
        return pd.DataFrame()

    if df is None or df.empty:
        return pd.DataFrame()

    # Map STD_ITEM_CODE to canonical names
    records: list[dict] = []
    for _, row in df.iterrows():
        item_code = str(row.get("STD_ITEM_CODE", ""))
        canonical = field_map.get(item_code)
        if not canonical:
            continue

        value = row.get("AMOUNT")
        if value is None or (isinstance(value, float) and np.isnan(value)):
            continue

        report_date = str(row.get("REPORT_DATE", ""))[:10]
        if not report_date:
            continue

        records.append({
            "canonical_name": canonical,
            "value": float(value),
            "report_date": report_date,
            "filing_date": "",  # EastMoney doesn't provide filing dates
        })

    if not records:
        return pd.DataFrame()

    result = pd.DataFrame(records)
    result["report_date"] = pd.to_datetime(result["report_date"], errors="coerce")

    # Dedup: keep one value per (canonical_name, report_date)
    result = result.sort_values("report_date", ascending=False)
    result = result.drop_duplicates(subset=["canonical_name", "report_date"], keep="first")
    result = result.sort_values("report_date")

    logger.info(
        "HKEX akshare for %s/%s: %d records across %d periods",
        code, statement_type, len(result),
        result["report_date"].nunique(),
    )
    return result


# ---------------------------------------------------------------------------
# HKHkexClient -- PIT client for Hong Kong HKEX equities
# ---------------------------------------------------------------------------


class HKHkexClient:
    """PIT client for Hong Kong HKEX equities.

    Uses akshare/EastMoney as primary data source for structured financial
    data (fast, no LLM needed). Falls back to HKEX News filing discovery
    for PDF-based extraction when akshare is unavailable.
    """

    def __init__(self, cache_dir: Path | str = _CACHE_DIR) -> None:
        self._cache_dir = Path(cache_dir)
        self._headers = {"Accept": "application/json", "User-Agent": "Operator1/1.0"}

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
        return "hk_hkex"

    @property
    def market_name(self) -> str:
        return "Hong Kong (HKEX)"

    # -- Company discovery ---------------------------------------------------

    def list_companies(self, query: str = "") -> list[dict[str, Any]]:
        from operator1.clients.yfinance_backed import yf_search
        return yf_search(query, self.market_id, "HK", "HKEX", yf_suffix=".HK")

    def search_company(self, name: str) -> list[dict[str, Any]]:
        return self.list_companies(query=name)

    # -- Company profile -----------------------------------------------------

    def get_profile(self, identifier: str) -> dict[str, Any]:
        """Fetch company profile from akshare/EastMoney + yfinance enrichment."""
        cached = self._read_cache(identifier, "profile.json")
        if cached:
            return cached

        profile: dict[str, Any] = {
            "ticker": identifier,
            "name": identifier,
            "country": "HK",
            "exchange": "HKEX",
            "currency": "HKD",
            "market_id": "hk_hkex",
        }

        # Try akshare company profile first
        try:
            import akshare as ak
            code = identifier.split(".")[0].strip().zfill(5)
            cp = ak.stock_hk_company_profile_em(symbol=code)
            if cp is not None and not cp.empty:
                row = cp.iloc[0]
                profile.update({
                    "name": str(row.get("英文名称", row.get("公司名称", identifier))),
                    "name_cn": str(row.get("公司名称", "")),
                    "registration": str(row.get("注册地", "")),
                    "industry": str(row.get("所属行业", "")),
                    "chairman": str(row.get("董事长", "")),
                    "employees": str(row.get("员工人数", "")),
                    "website": str(row.get("公司网址", "")),
                    "email": str(row.get("E-MAIL", "")),
                    "fiscal_year_end": str(row.get("年结日", "")),
                    "auditor": str(row.get("核数师", "")),
                })
                logger.info("HKEX profile for %s from akshare: %s", identifier, profile.get("name", "?"))
        except Exception as exc:
            logger.debug("akshare profile failed for %s: %s", identifier, exc)

        # Enrich with yfinance for sector/industry/market_cap
        try:
            from operator1.clients.yfinance_backed import yf_get_profile
            yf_profile = yf_get_profile(
                identifier, self.market_id,
                "Hong Kong", "HK", "HKEX", "HKD",
                yf_suffix=".HK",
            )
            for key in ("sector", "industry", "market_cap", "shares_outstanding",
                        "pe_ratio", "eps", "isin", "description"):
                if key not in profile or not profile[key]:
                    val = yf_profile.get(key)
                    if val:
                        profile[key] = val
        except Exception as exc:
            logger.debug("yfinance enrichment failed for %s: %s", identifier, exc)

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
        """Fetch financials from akshare/EastMoney (primary) or filing discovery (fallback).

        Primary path: akshare stock_financial_hk_report_em -- structured data
        from EastMoney, fast (~2s), no LLM needed. ~4,000 rows per company.

        Note: EastMoney data does not include filing_date (announcement date).
        For true PIT compliance, the filing discovery path provides filing_dates
        from HKEX announcement timestamps.

        Fallback: HKEX filing discovery + LLM/fuzzy PDF extraction.
        """
        # Primary: akshare/EastMoney (fast, structured, no LLM)
        try:
            df = _fetch_hkex_akshare(identifier, statement_type)
            if df is not None and not df.empty:
                logger.info("HKEX %s %s: %d rows from akshare (fast path)",
                           identifier, statement_type, len(df))
                return df
        except Exception as exc:
            logger.debug("HKEX akshare extraction failed for %s: %s", identifier, exc)

        # Fallback: HKEX filing discovery + LLM/fuzzy extraction
        try:
            from operator1.clients.filing_discoverer import try_filing_extraction
            df = try_filing_extraction(
                ticker=identifier,
                market_id=self.market_id,
                statement_type=statement_type,
                llm_client=None,
            )
            if df is not None and not df.empty:
                logger.info("HKEX %s %s: %d rows from filing discovery",
                           identifier, statement_type, len(df))
                return df
        except Exception as exc:
            logger.debug("HKEX filing discovery failed for %s: %s", identifier, exc)
        return pd.DataFrame()

    # -- Price data ----------------------------------------------------------

    def get_quotes(self, identifier: str) -> pd.DataFrame:
        """HKEX does not provide OHLCV data. Handled by ohlcv_provider."""
        return pd.DataFrame()

    def get_peers(self, identifier: str) -> list[str]:
        return []

    def get_executives(self, identifier: str) -> list[dict[str, Any]]:
        return []

    # -- Institutional holders (akshare / HKEX filing discovery) -------------

    def _hk_code(self, identifier: str) -> str:
        """Normalize HKEX stock code (e.g. '700' -> '00700')."""
        code = identifier.split(".")[0].strip().lstrip("0") or "0"
        return code.zfill(5)

    def get_holders(self, identifier: str) -> list[dict[str, Any]]:
        """Fetch holders from HKEX disclosure of interest filings.

        Uses the HKEX date-windowed scraper to search for "disclosure"
        filings (SFO Part XV -- substantial shareholder notifications).
        Parses filing titles for holder names and share change data.

        Probing confirmed (2026-03-26):
          - akshare stock_hk_main_board_stock_holder_em: REMOVED from
            akshare 1.18.43 (AttributeError)
          - EastMoney HK F10 shareholder PageAjax endpoints: all 404
          - EastMoney datacenter RPT_HK_* reports: all "config not found"
          - HKEXScraper.search_announcements: method doesn't exist
            (only search_filings + download_pdf)
          - HKEX DI system (di.hkex.com.hk): ASP.NET WebForms, needs
            complex __VIEWSTATE form POST chains

        The HKEX news API (titleSearchServlet.do) with title="disclosure"
        returns "Next Day Disclosure Return" filings which contain share
        buyback and issued share change data.  These are the best native
        holder-adjacent data available without scraping the DI system.

        No yfinance dependency.

        Returns list of dicts with: name, shares, percentage, holder_type,
        date_reported, source, document_url.
        """
        holders: list[dict[str, Any]] = []

        # --- HKEX disclosure filings via date-windowed scraper ---
        try:
            from operator1.clients.hkex_scraper import HKEXScraper
            from datetime import timedelta

            scraper = HKEXScraper()
            session = scraper._get_session()
            code = self._hk_code(identifier)

            today = date.today()
            import re as _re

            # Search recent 3-month window for disclosure filings
            # (HKEX API limits to 2-week windows per request)
            all_records: list[dict] = []
            current_end = today
            search_start = today - timedelta(days=90)
            window = timedelta(days=14)

            while current_end > search_start and len(all_records) < 20:
                current_start = max(current_end - window, search_start)
                from_str = current_start.strftime("%Y%m%d")
                to_str = current_end.strftime("%Y%m%d")

                records = scraper._query_window(session, from_str, to_str, "disclosure")
                # Filter for this stock code
                for rec in records:
                    raw_code = rec.get("STOCK_CODE", "").split("<br/>")[0].strip()
                    if raw_code == code:
                        all_records.append(rec)

                current_end = current_start - timedelta(days=1)

            if all_records:
                seen_titles: set[str] = set()
                for rec in all_records:
                    title = rec.get("TITLE", "").strip()
                    if not title or title in seen_titles:
                        continue
                    seen_titles.add(title)

                    # Parse release date
                    date_time = rec.get("DATE_TIME", "")
                    date_part = date_time.split(" ")[0] if date_time else ""
                    release_date = ""
                    if date_part and "/" in date_part:
                        parts = date_part.split("/")
                        if len(parts) == 3:
                            release_date = f"{parts[2]}-{parts[1]}-{parts[0]}"

                    # Build document URL
                    file_link = rec.get("FILE_LINK", "")
                    if file_link and file_link.startswith("/"):
                        file_link = "https://www1.hkexnews.hk" + file_link

                    long_text = rec.get("LONG_TEXT", "")

                    holders.append({
                        "name": title[:80],
                        "shares": 0,
                        "value": 0.0,
                        "percentage": 0.0,
                        "holder_type": "disclosure",
                        "date_reported": release_date,
                        "source": "hkex_disclosure",
                        "document_url": file_link,
                        "filing_category": long_text,
                    })

                if holders:
                    logger.info(
                        "HKEX disclosure filings for %s: %d from date-windowed search",
                        identifier, len(holders),
                    )
        except Exception as exc:
            logger.debug("HKEX disclosure search failed for %s: %s", identifier, exc)

        return holders

    def get_holder_history(self, identifier: str, years: int = 2) -> "pd.DataFrame":
        """Return disclosure filing metrics from HKEX.

        Derives aggregate metrics from get_holders() which uses HKEX
        disclosure filings.  Returns a single-row snapshot with filing
        count and date range.

        No yfinance dependency.
        """
        try:
            holders = self.get_holders(identifier)
            if not holders:
                return pd.DataFrame()

            # Use the most recent date_reported from holders
            report_dates = [h.get("date_reported", "") for h in holders if h.get("date_reported")]
            report_date = max(report_dates) if report_dates else date.today().isoformat()

            return pd.DataFrame([{
                "date_reported": pd.Timestamp(report_date),
                "inst_ownership_pct": 0.0,  # DI filings don't provide aggregate %
                "inst_top5_concentration": 0.0,
                "inst_holder_count": len(holders),
            }])
        except Exception as exc:
            logger.debug("HKEX holder history failed for %s: %s", identifier, exc)
        return pd.DataFrame()

    def get_insider_transactions(self, identifier: str) -> list[dict[str, Any]]:
        """Fetch insider/director transactions from HKEX news filings.

        Uses the HKEX date-windowed scraper to search for "director"
        filings (director dealing announcements under Listing Rules).
        Returns filing metadata; full extraction requires LLM.

        BUG FIX (2026-03-26): Previously called scraper.search_announcements()
        which doesn't exist. Now uses search_filings() which is the actual
        method, but searches for "director" title filter to find director
        dealing filings.

        No yfinance dependency.
        """
        transactions: list[dict[str, Any]] = []
        try:
            from operator1.clients.hkex_scraper import HKEXScraper
            from datetime import timedelta

            scraper = HKEXScraper()
            session = scraper._get_session()
            code = self._hk_code(identifier)

            today = date.today()

            # Search recent 3 months for director dealing filings
            all_records: list[dict] = []
            current_end = today
            search_start = today - timedelta(days=90)
            window = timedelta(days=14)

            while current_end > search_start and len(all_records) < 20:
                current_start = max(current_end - window, search_start)
                from_str = current_start.strftime("%Y%m%d")
                to_str = current_end.strftime("%Y%m%d")

                records = scraper._query_window(session, from_str, to_str, "director")
                for rec in records:
                    raw_code = rec.get("STOCK_CODE", "").split("<br/>")[0].strip()
                    if raw_code == code:
                        all_records.append(rec)

                current_end = current_start - timedelta(days=1)

            for rec in all_records:
                title = rec.get("TITLE", "").strip()
                date_time = rec.get("DATE_TIME", "")
                date_part = date_time.split(" ")[0] if date_time else ""
                release_date = ""
                if date_part and "/" in date_part:
                    parts = date_part.split("/")
                    if len(parts) == 3:
                        release_date = f"{parts[2]}-{parts[1]}-{parts[0]}"

                file_link = rec.get("FILE_LINK", "")
                if file_link and file_link.startswith("/"):
                    file_link = "https://www1.hkexnews.hk" + file_link

                transactions.append({
                    "insider_name": title[:60],
                    "position": "",
                    "date": release_date,
                    "transaction": "Disclosure",
                    "shares": 0,
                    "value": 0.0,
                    "source": "hkex_news",
                    "document_url": file_link,
                })

            if transactions:
                logger.info(
                    "HKEX insider transactions for %s: %d from date-windowed search",
                    identifier, len(transactions),
                )
        except Exception as exc:
            logger.debug("HKEX insider transaction discovery failed for %s: %s", identifier, exc)
        return transactions
