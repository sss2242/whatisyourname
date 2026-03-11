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
from datetime import date, datetime, timedelta
from pathlib import Path
from typing import Any

import pandas as pd

logger = logging.getLogger(__name__)

_CACHE_DIR = Path("cache/jp_jquants")

# V2 column name -> canonical field mapping (from research log Section 6)
_V2_INCOME_MAP = {
    "Sales": "revenue",
    "OP": "operating_income",
    "OdP": "pretax_income",
    "NP": "net_income",
    "EPS": "eps",
}

_V2_BALANCE_MAP = {
    "TA": "total_assets",
    "TL": "total_liabilities",
    "Eq": "total_equity",
    "CA": "current_assets",
    "CL": "current_liabilities",
    "CashEq": "cash_and_equivalents",
    "BPS": "book_value_per_share",
    # J-Quants V2 condensed summary may not have all of these;
    # missing fields will be derived post-extraction or filled by
    # the estimation engine.
}

_V2_CASHFLOW_MAP = {
    "CFO": "operating_cash_flow",  # must match cache_builder.STATEMENT_FIELDS
    "CFI": "investing_cf",          # was "investing_cashflow" (wrong)
    "CFF": "financing_cf",          # was "financing_cashflow" (wrong)
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
                results.append({
                    "identifier": str(row.get("Code", ""))[:4],
                    "name": str(row.get("CompanyNameEnglish", row.get("CompanyName", ""))),
                    "exchange": "TSE",
                    "sector": str(row.get("S33NmEn", row.get("S17NmEn", ""))),
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
            df = self._client.get_eq_master(code=code)
            if df.empty:
                return {}

            row = df.iloc[0]
            return {
                "name": str(row.get("CompanyNameEnglish", row.get("CompanyName", ""))),
                "ticker": str(row.get("Code", ""))[:4],
                "exchange": "TSE",
                "sector": str(row.get("S33NmEn", row.get("S17NmEn", ""))),
                "market_segment": str(row.get("MktNmEn", row.get("Mkt", ""))),
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
            if len(code) == 4:
                code = code + "0"

            # Fetch financial summary for the date range
            end_dt = datetime.now()
            start_dt = end_dt - timedelta(days=365 * years)

            # Docs: jquantsapi/client_v2.py get_fin_summary_range() method
            df = self._client.get_fin_summary_range(
                start_dt=start_dt,
                end_dt=end_dt,
            )

            if df.empty:
                return empty

            # Filter to this company
            code_col = "Code" if "Code" in df.columns else "LocalCode"
            if code_col in df.columns:
                df = df[df[code_col].astype(str).str.startswith(code[:4])]

            if df.empty:
                return empty

            # Only keep annual reports (DocType containing "Annual" or FY end)
            # J-Quants V2: CurPerType field indicates period type
            if "CurPerType" in df.columns:
                # Keep FY (full year) entries
                annual_mask = df["CurPerType"].astype(str).str.contains("FY|Annual|4Q", case=False, na=True)
                df_annual = df[annual_mask] if annual_mask.any() else df
            elif "DocType" in df.columns:
                df_annual = df
            else:
                df_annual = df

            # Determine date column for PIT
            date_col = "DiscDate" if "DiscDate" in df_annual.columns else "DisclosedDate"
            period_end_col = "CurPerEn" if "CurPerEn" in df_annual.columns else "CurrentPeriodEndDate"

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
                # Derive missing fields from accounting identities
                ta = rec.get("total_assets")
                eq = rec.get("total_equity")
                if ta is not None and eq is not None and "total_liabilities" not in rec:
                    rec["total_liabilities"] = ta - eq
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

            # Log unmapped J-Quants V2 columns for diagnostic purposes.
            # J-Quants bypasses canonical_translator, so the global
            # unmapped concept logging doesn't cover this wrapper.
            _meta_cols = {
                date_col, period_end_col, "Code", "LocalCode",
                "CurPerType", "DocType", "DiscDate", "DisclosedDate",
                "CurPerEn", "CurrentPeriodEndDate",
            }
            _all_mapped = set(_V2_INCOME_MAP) | set(_V2_BALANCE_MAP) | set(_V2_CASHFLOW_MAP)
            _available = set(df_annual.columns) - _meta_cols
            _unmapped = _available - _all_mapped
            if _unmapped:
                logger.debug(
                    "J-Quants V2 unmapped columns (not in _V2_*_MAP): %s",
                    sorted(_unmapped)[:20],
                )

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
            df = self._client.get_list()
            if df.empty:
                return []

            sector_col = "S33NmEn" if "S33NmEn" in df.columns else "S17NmEn"
            if sector_col in df.columns:
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
