"""Japan J-Quants PIT client -- powered by jquants-api-client SDK.

Replaces the original jp_edinet_wrapper.py. Uses the official J-Quants
API from JPX (Japan Exchange Group) for structured financial data.

Primary library: jquants-api-client 2.0.0 (https://github.com/J-Quants/jquants-api-client-python)
  - Equity master, daily bars, financial summary, earnings calendar
  - Official JPX SDK with V2 API key authentication

Coverage: ~3,800+ listed companies on TSE.
API: https://jpx-jquants.com/ (free plan available via email registration)

VERIFIED AGAINST OFFICIAL DOCS:
- Date: 2026-02-25
- Version: jquants-api-client@2.0.0
- Docs: https://github.com/J-Quants/jquants-api-client-python
- Research Log: .roo/research/jp-jquants-2026-02-25.md
"""

from __future__ import annotations

import logging
import os
import time
from datetime import date, datetime, timedelta
from pathlib import Path
from typing import Any

import pandas as pd

logger = logging.getLogger(__name__)

_CACHE_DIR = Path("cache/jp_jquants")

# J-Quants free plan: ~12 requests/minute.
# Enforce a 6-second minimum interval between API calls to stay safe.
_JQUANTS_MIN_INTERVAL = 6.0  # seconds between API calls
_jquants_last_call: float = 0.0


def _jquants_throttle() -> None:
    """Enforce minimum interval between J-Quants API calls."""
    global _jquants_last_call
    elapsed = time.monotonic() - _jquants_last_call
    if elapsed < _JQUANTS_MIN_INTERVAL:
        wait = _JQUANTS_MIN_INTERVAL - elapsed
        logger.debug("J-Quants rate limit: waiting %.1fs", wait)
        time.sleep(wait)
    _jquants_last_call = time.monotonic()

# V2 column name -> canonical field mapping
# get_eq_master() uses short column names (V2 API): CoName, CoNameEn, S33, S33Nm, Mkt, MktNm
# get_list() uses full column names: CompanyName, CompanyNameEnglish, Sector33Code, Sector33CodeName
# get_fin_summary() / get_fins_statements() use: NetSales, OperatingProfit, Profit, TotalAssets, Equity, etc.

_V2_INCOME_MAP = {
    # get_fin_summary columns (FINS_STATEMENTS format)
    "NetSales": "revenue",
    "OperatingProfit": "operating_income",
    "OrdinaryProfit": "pretax_income",
    "Profit": "net_income",
    "EarningsPerShare": "eps",
    "DilutedEarningsPerShare": "eps_diluted",
    # Legacy short names (kept for backward compat)
    "Sales": "revenue",
    "OP": "operating_income",
    "OdP": "pretax_income",
    "NP": "net_income",
    "EPS": "eps",
}

_V2_BALANCE_MAP = {
    # get_fin_summary columns
    "TotalAssets": "total_assets",
    "Equity": "total_equity",
    # Legacy short names
    "TA": "total_assets",
    "Eq": "total_equity",
    "CashEq": "cash_and_equivalents",
    "BPS": "book_value_per_share",
}

_V2_CASHFLOW_MAP = {
    # Legacy short names (cashflow not in base fin_summary)
    "CFO": "operating_cashflow",
    "CFI": "investing_cashflow",
    "CFF": "financing_cashflow",
}


class JPJquantsError(Exception):
    """Raised on Japan J-Quants wrapper failures."""
    def __init__(self, endpoint: str, detail: str = "") -> None:
        self.endpoint = endpoint
        self.detail = detail
        super().__init__(f"JP J-Quants error on {endpoint}: {detail}")


class JPJquantsClient:
    """Point-in-time client for J-Quants (Japanese equities).

    Implements the ``PITClient`` protocol. Uses jquants-api-client
    ClientV2 with API key authentication.

    Parameters
    ----------
    api_key:
        J-Quants API key. Also loads from JQUANTS_API_KEY env var.
        Get one free at https://jpx-jquants.com/login
    cache_dir:
        Local cache directory.
    """

    def __init__(
        self,
        api_key: str = "",
        cache_dir: Path | str = _CACHE_DIR,
    ) -> None:
        self._api_key = api_key or os.environ.get("JQUANTS_API_KEY", "")
        self._cache_dir = Path(cache_dir)
        self._cache_dir.mkdir(parents=True, exist_ok=True)

        self._client = None
        if self._api_key:
            try:
                import jquantsapi
                self._client = jquantsapi.ClientV2(api_key=self._api_key)
                logger.info("J-Quants ClientV2 initialized successfully")
            except ImportError:
                logger.warning(
                    "jquants-api-client not installed. "
                    "Install with: pip install jquants-api-client>=2.0.0"
                )
            except Exception as exc:
                logger.warning("Failed to initialize J-Quants client: %s", exc)
        else:
            logger.warning(
                "JQUANTS_API_KEY not set. J-Quants API requires an API key. "
                "Register free at https://jpx-jquants.com/login"
            )

    # ------------------------------------------------------------------
    # PITClient protocol methods
    # ------------------------------------------------------------------

    def list_companies(self, query: str = "") -> list[dict]:
        """Search or list Japanese companies via J-Quants equity master.

        Parameters
        ----------
        query:
            Company name or ticker code to search for. If empty, returns
            a sample of listed companies.

        Returns
        -------
        List of dicts with keys: identifier, name, exchange, sector.
        """
        if not self._client:
            logger.warning("J-Quants client not available")
            return []

        try:
            # Docs: jquantsapi/client_v2.py get_list() method
            _jquants_throttle()
            df = self._client.get_list()
            if df.empty:
                return []

            # Filter by query if provided
            if query:
                query_lower = query.lower()
                mask = (
                    df.get("Code", pd.Series(dtype=str)).astype(str).str.contains(query, case=False, na=False) |
                    df.get("CompanyName", pd.Series(dtype=str)).astype(str).str.contains(query, case=False, na=False) |
                    df.get("CompanyNameEnglish", pd.Series(dtype=str)).astype(str).str.contains(query, case=False, na=False)
                )
                df = df[mask]

            results = []
            for _, row in df.head(50).iterrows():
                # get_list() returns full names: CompanyName, CompanyNameEnglish, Sector33CodeName
                # get_eq_master() returns short: CoName, CoNameEn, S33Nm
                name = str(row.get("CompanyNameEnglish", "") or row.get("CoNameEn", "") or row.get("CompanyName", "") or row.get("CoName", ""))
                sector = str(row.get("Sector33CodeName", "") or row.get("S33Nm", "") or row.get("Sector17CodeName", "") or row.get("S17Nm", ""))
                results.append({
                    "identifier": str(row.get("Code", ""))[:4],
                    "name": name,
                    "exchange": "TSE",
                    "sector": sector,
                })
            return results

        except Exception as exc:
            logger.error("J-Quants list_companies failed: %s", exc)
            return []

    def get_profile(self, identifier: str) -> dict:
        """Get company profile from J-Quants equity master.

        Parameters
        ----------
        identifier:
            Ticker code (4 or 5 digits).

        Returns
        -------
        Dict with company profile fields.
        """
        if not self._client:
            return {}

        try:
            # Normalize to 5-digit code (J-Quants uses 5 digits)
            code = identifier.strip()
            if len(code) == 4:
                code = code + "0"

            # Docs: jquantsapi/client_v2.py get_eq_master() method
            _jquants_throttle()
            df = self._client.get_eq_master(code=code)
            if df.empty:
                return {}

            row = df.iloc[0]
            # V2 eq_master columns: CoName, CoNameEn, S33, S33Nm, Mkt, MktNm
            return {
                "name": str(row.get("CoNameEn", "") or row.get("CoName", "") or row.get("CompanyNameEnglish", "") or row.get("CompanyName", "")),
                "ticker": str(row.get("Code", ""))[:4],
                "exchange": "TSE",
                "sector": str(row.get("S33Nm", "") or row.get("S33", "") or row.get("Sector33CodeName", "") or row.get("Sector33Code", "")),
                "market_segment": str(row.get("MktNm", "") or row.get("Mkt", "") or row.get("MarketCodeName", "") or row.get("MarketCode", "")),
                "country": "Japan",
                "currency": "JPY",
            }

        except Exception as exc:
            logger.error("J-Quants get_profile failed for %s: %s", identifier, exc)
            return {}

    def get_financials(
        self,
        identifier: str,
        years: int = 5,
    ) -> dict[str, pd.DataFrame]:
        """Get financial statements from J-Quants financial summary.

        Uses get_fin_summary which is available on the FREE plan.
        Returns income statement, balance sheet, and cash flow data.

        Parameters
        ----------
        identifier:
            Ticker code (4 or 5 digits).
        years:
            Number of years of data to fetch.

        Returns
        -------
        Dict with keys 'income', 'balance', 'cashflow', each containing
        a PIT-compliant DataFrame.
        """
        empty = {
            "income": pd.DataFrame(),
            "balance": pd.DataFrame(),
            "cashflow": pd.DataFrame(),
        }

        if not self._client:
            return empty

        try:
            # Normalize to 5-digit code
            code = identifier.strip()
            if not code:
                logger.warning("J-Quants get_financials called with empty identifier")
                return empty
            if len(code) == 4:
                code = code + "0"

            # Fetch financial summary for the date range.
            # Use code parameter to avoid fetching ALL companies
            # (get_fin_summary_range without code returns thousands of rows
            # and exceeds free-tier rate limits).
            end_dt = datetime.now()
            start_dt = end_dt - timedelta(days=365 * years)

            # Docs: jquantsapi/client_v2.py get_fin_summary_range() method
            _jquants_throttle()
            try:
                # Try code-filtered fetch first (faster, less API load)
                df = self._client.get_fin_summary_range(
                    start_dt=start_dt,
                    end_dt=end_dt,
                    code=code,
                )
            except TypeError:
                # Fallback: older SDK versions may not support code param
                df = self._client.get_fin_summary_range(
                    start_dt=start_dt,
                    end_dt=end_dt,
                )

            if df.empty:
                return empty

            # Filter to this company (FINS uses LocalCode, others use Code)
            code_col = None
            for cc in ("LocalCode", "Code"):
                if cc in df.columns:
                    code_col = cc
                    break
            if code_col:
                df = df[df[code_col].astype(str).str.startswith(code[:4])]

            if df.empty:
                return empty

            # Only keep annual/4Q reports
            # FINS_STATEMENTS uses TypeOfCurrentPeriod (e.g. "FY", "1Q", "2Q", "3Q")
            # Older format may use CurPerType
            period_col = None
            for pc in ("TypeOfCurrentPeriod", "CurPerType", "TypeOfDocument"):
                if pc in df.columns:
                    period_col = pc
                    break

            if period_col:
                annual_mask = df[period_col].astype(str).str.contains("FY|Annual|4Q", case=False, na=True)
                df_annual = df[annual_mask] if annual_mask.any() else df
            else:
                df_annual = df

            # Determine date columns for PIT
            # FINS_STATEMENTS: DisclosedDate, CurrentPeriodEndDate
            # Older format: DiscDate, CurPerEn
            date_col = None
            for dc in ("DisclosedDate", "DiscDate"):
                if dc in df_annual.columns:
                    date_col = dc
                    break
            if not date_col:
                date_col = "DisclosedDate"  # fallback

            period_end_col = None
            for pec in ("CurrentPeriodEndDate", "CurPerEn"):
                if pec in df_annual.columns:
                    period_end_col = pec
                    break
            if not period_end_col:
                period_end_col = "CurrentPeriodEndDate"  # fallback

            # Build income statement
            income_rows = []
            for _, row in df_annual.iterrows():
                rec = {
                    "filing_date": str(row.get(date_col, "")),
                    "report_date": str(row.get(period_end_col, "")),
                }
                for v2_col, canonical in _V2_INCOME_MAP.items():
                    val = row.get(v2_col)
                    if pd.notna(val):
                        rec[canonical] = float(val)
                income_rows.append(rec)

            # Build balance sheet
            balance_rows = []
            for _, row in df_annual.iterrows():
                rec = {
                    "filing_date": str(row.get(date_col, "")),
                    "report_date": str(row.get(period_end_col, "")),
                }
                for v2_col, canonical in _V2_BALANCE_MAP.items():
                    val = row.get(v2_col)
                    if pd.notna(val):
                        rec[canonical] = float(val)
                balance_rows.append(rec)

            # Build cash flow statement
            cashflow_rows = []
            for _, row in df_annual.iterrows():
                rec = {
                    "filing_date": str(row.get(date_col, "")),
                    "report_date": str(row.get(period_end_col, "")),
                }
                for v2_col, canonical in _V2_CASHFLOW_MAP.items():
                    val = row.get(v2_col)
                    if pd.notna(val):
                        rec[canonical] = float(val)
                cashflow_rows.append(rec)

            return {
                "income": pd.DataFrame(income_rows) if income_rows else pd.DataFrame(),
                "balance": pd.DataFrame(balance_rows) if balance_rows else pd.DataFrame(),
                "cashflow": pd.DataFrame(cashflow_rows) if cashflow_rows else pd.DataFrame(),
            }

        except Exception as exc:
            logger.error("J-Quants get_financials failed for %s: %s", identifier, exc)
            return empty

    def get_peers(self, identifier: str, limit: int = 10) -> list[dict]:
        """Find peer companies by sector using J-Quants equity master.

        Parameters
        ----------
        identifier:
            Ticker code.
        limit:
            Maximum number of peers to return.

        Returns
        -------
        List of dicts with peer company info.
        """
        if not self._client:
            return []

        try:
            # Get the target company's sector
            profile = self.get_profile(identifier)
            if not profile or not profile.get("sector"):
                return []

            target_sector = profile["sector"]

            # Get all companies and filter by sector
            _jquants_throttle()
            df = self._client.get_list()
            if df.empty:
                return []

            # get_list() columns: Sector33CodeName, Sector17CodeName
            # get_eq_master() columns: S33Nm, S17Nm
            sector_col = None
            for sc in ("Sector33CodeName", "S33Nm", "Sector17CodeName", "S17Nm"):
                if sc in df.columns:
                    sector_col = sc
                    break
            if sector_col:
                peers_df = df[df[sector_col] == target_sector]
            else:
                return []

            # Exclude the target company
            code = identifier.strip()
            if "Code" in peers_df.columns:
                peers_df = peers_df[~peers_df["Code"].astype(str).str.startswith(code[:4])]

            results = []
            for _, row in peers_df.head(limit).iterrows():
                results.append({
                    "identifier": str(row.get("Code", ""))[:4],
                    "name": str(row.get("CompanyNameEnglish", row.get("CompanyName", ""))),
                    "exchange": "TSE",
                    "sector": str(row.get(sector_col, "")),
                })
            return results

        except Exception as exc:
            logger.error("J-Quants get_peers failed for %s: %s", identifier, exc)
            return []

    def get_executives(self, identifier: str) -> list[dict]:
        """J-Quants does not provide executive data.

        Returns empty list. Executive data would need a separate source.
        """
        return []
