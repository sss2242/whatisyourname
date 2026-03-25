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
        """Fetch holders from akshare EastMoney (structured, no LLM needed).

        Uses akshare stock_hk_shareholders_em to get top-10 shareholders
        from EastMoney's aggregated HK shareholder data.  Falls back to
        HKEX filing discovery for disclosure of interest filings.

        Returns list of dicts with: name, shares, percentage, holder_type,
        date_reported, source.
        """
        holders: list[dict[str, Any]] = []

        # --- Path 1: akshare EastMoney shareholder data ---
        try:
            import akshare as ak
            code = self._hk_code(identifier)

            # Try top-10 shareholders
            try:
                df = ak.stock_hk_main_board_stock_holder_em(symbol=code)
                if df is not None and not df.empty:
                    for _, row in df.head(20).iterrows():
                        name = str(row.get("股东名称", row.get("holder_name", "")))
                        pct = 0.0
                        try:
                            pct_raw = row.get("持股比例", row.get("hold_ratio", 0))
                            if pct_raw is not None and str(pct_raw).strip() and str(pct_raw).strip() != "nan":
                                pct = float(str(pct_raw).replace("%", "").strip())
                                # akshare/EastMoney returns 持股比例 already as
                                # percentage (5.73 = 5.73%). Do NOT multiply by 100.
                        except (ValueError, TypeError):
                            pass
                        shares = 0
                        try:
                            shares = int(float(row.get("持股数量", row.get("hold_num", 0))))
                        except (ValueError, TypeError):
                            pass
                        date_str = str(row.get("公告日期", row.get("ann_date", "")))
                        if name and (pct > 0 or shares > 0):
                            holders.append({
                                "name": name,
                                "shares": shares,
                                "value": 0.0,
                                "percentage": round(pct, 2),
                                "holder_type": "institutional",
                                "date_reported": date_str,
                                "source": "akshare_eastmoney",
                            })
            except Exception as exc:
                logger.debug("akshare HK shareholders failed for %s: %s", identifier, exc)

            if holders:
                logger.info("HKEX holders for %s: %d from akshare/EastMoney", identifier, len(holders))
        except ImportError:
            logger.debug("akshare not available for HKEX holder lookup")
        except Exception as exc:
            logger.debug("akshare HK holder lookup failed for %s: %s", identifier, exc)

        # --- Path 2: HKEX filing discovery for disclosure of interest ---
        if not holders:
            try:
                from operator1.clients.hkex_scraper import HKEXScraper
                scraper = HKEXScraper()
                announcements = scraper.search_announcements(
                    stock_code=identifier,
                    category="disclosure",
                    max_results=10,
                )
                if announcements:
                    logger.info(
                        "HKEX disclosure filings for %s: %d found (LLM extraction available)",
                        identifier, len(announcements),
                    )
            except Exception as exc:
                logger.debug("HKEX filing discovery for holders failed: %s", exc)

        return holders

    def get_holder_history(self, identifier: str, years: int = 2) -> "pd.DataFrame":
        """Return institutional ownership metrics from akshare/EastMoney.

        Derives aggregate metrics from the holder data returned by
        get_holders().  Returns a single-row snapshot.
        """
        try:
            from datetime import date as _date

            holders = self.get_holders(identifier)
            if not holders:
                return pd.DataFrame()

            inst_holders = [h for h in holders if h.get("holder_type") == "institutional"]
            inst_pct = sum(h.get("percentage", 0) for h in inst_holders)

            hhi = 0.0
            top5 = inst_holders[:5]
            total_pct = sum(h.get("percentage", 0) for h in top5)
            if total_pct > 0:
                hhi = sum((h.get("percentage", 0) / total_pct) ** 2 for h in top5)

            return pd.DataFrame([{
                "date_reported": pd.Timestamp(_date.today()),
                "inst_ownership_pct": round(inst_pct, 2),
                "inst_top5_concentration": round(hhi, 4),
                "inst_holder_count": len(inst_holders),
            }])
        except Exception as exc:
            logger.debug("HKEX holder history failed for %s: %s", identifier, exc)
        return pd.DataFrame()

    def get_insider_transactions(self, identifier: str) -> list[dict[str, Any]]:
        """Fetch insider/director transactions from HKEX disclosure filings.

        Uses HKEX filing discovery to find director dealing announcements.
        Returns basic metadata; full extraction requires LLM.
        """
        transactions: list[dict[str, Any]] = []
        try:
            from operator1.clients.hkex_scraper import HKEXScraper
            scraper = HKEXScraper()
            announcements = scraper.search_announcements(
                stock_code=identifier,
                category="director",
                max_results=20,
            )
            for ann in (announcements or []):
                transactions.append({
                    "insider_name": ann.get("headline", "Unknown Director"),
                    "position": "",
                    "date": ann.get("date", ""),
                    "transaction": "Disclosure",
                    "shares": 0,
                    "value": 0.0,
                    "source": "hkex_filing_discovery",
                })
            if transactions:
                logger.info(
                    "HKEX insider transactions for %s: %d from filing discovery",
                    identifier, len(transactions),
                )
        except Exception as exc:
            logger.debug("HKEX insider transaction discovery failed for %s: %s", identifier, exc)
        return transactions
