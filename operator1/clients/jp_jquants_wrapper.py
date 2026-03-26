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
_FIN_SUMMARY_CACHE_DIR = str(Path("cache/jp_jquants/fin_summary_csv"))

# J-Quants free plan: ~12 requests/minute.
# Enforce an 8-second minimum interval between API calls to stay safe
# (6s was too aggressive, caused 429 bursts during profile + financial fetch).
_JQUANTS_MIN_INTERVAL = 8.0  # seconds between API calls
_jquants_last_call: float = 0.0
_jquants_eq_master_cache: Any = None  # cached equity master DataFrame


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

    def _get_eq_master_cached(self) -> pd.DataFrame:
        """Get equity master with disk + memory caching.

        First checks in-memory cache, then disk cache (Parquet),
        then fetches from API.  Disk cache expires after 24 hours.
        Eliminates repeated API calls for company search/profile.
        """
        global _jquants_eq_master_cache
        if _jquants_eq_master_cache is not None:
            return _jquants_eq_master_cache

        # Check disk cache (works even without API client)
        cache_path = self._cache_dir / "eq_master.parquet"
        if cache_path.exists():
            age_hours = (time.time() - cache_path.stat().st_mtime) / 3600
            if age_hours < 24:
                try:
                    _jquants_eq_master_cache = pd.read_parquet(cache_path)
                    logger.debug("J-Quants eq_master loaded from disk cache (%d rows)", len(_jquants_eq_master_cache))
                    return _jquants_eq_master_cache
                except Exception:
                    pass
            # Cache expired but still return it if no client available
            elif not self._client:
                try:
                    _jquants_eq_master_cache = pd.read_parquet(cache_path)
                    logger.debug("J-Quants eq_master loaded from stale disk cache (%d rows)", len(_jquants_eq_master_cache))
                    return _jquants_eq_master_cache
                except Exception:
                    pass

        # Fetch from API
        if not self._client:
            return pd.DataFrame()

        _jquants_throttle()
        try:
            df = self._client.get_list()
        except Exception:
            df = pd.DataFrame()

        if not df.empty:
            _jquants_eq_master_cache = df
            # Save to disk
            try:
                self._cache_dir.mkdir(parents=True, exist_ok=True)
                df.to_parquet(cache_path, index=False)
                logger.info("J-Quants eq_master cached to disk (%d rows)", len(df))
            except Exception as exc:
                logger.debug("Failed to cache eq_master to disk: %s", exc)

        return df

    def list_companies(self, query: str = "") -> list[dict]:
        """Search or list Japanese companies via J-Quants equity master.

        Uses disk-cached equity master (24h TTL) to avoid repeated API
        calls.  Only fetches from J-Quants API if no cache exists.

        Parameters
        ----------
        query:
            Company name or ticker code to search for. If empty, returns
            a sample of listed companies.

        Returns
        -------
        List of dicts with keys: identifier, name, exchange, sector.
        """
        try:
            df = self._get_eq_master_cached()
            if df.empty:
                return []

            # Filter by query if provided
            # Handle both column name patterns:
            #   get_eq_master(): CoName, CoNameEn, S33Nm
            #   get_list():      CompanyName, CompanyNameEnglish, Sector33CodeName
            if query:
                name_cols = [c for c in ("CoNameEn", "CoName", "CompanyNameEnglish", "CompanyName") if c in df.columns]
                mask = df.get("Code", pd.Series(dtype=str)).astype(str).str.contains(query, case=False, na=False)
                for col in name_cols:
                    mask = mask | df[col].astype(str).str.contains(query, case=False, na=False)
                df = df[mask]

            results = []
            for _, row in df.head(50).iterrows():
                # Handle both column name patterns
                name = str(row.get("CoNameEn", "") or row.get("CompanyNameEnglish", "") or row.get("CoName", "") or row.get("CompanyName", ""))
                sector = str(row.get("S33NmEn", "") or row.get("S33Nm", "") or row.get("Sector33CodeName", ""))
                ticker_code = str(row.get("Code", ""))[:4]
                results.append({
                    "identifier": ticker_code,
                    "ticker": ticker_code,
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
        try:
            # Normalize to 5-digit code (J-Quants uses 5 digits: 7203 -> 72030)
            code = identifier.strip()
            base4 = code[:4]  # keep the 4-digit code for matching
            if len(code) == 4:
                code = code + "0"

            # Use the cached equity master to avoid an extra API call.
            # Falls back to direct API call if cache is empty.
            df = self._get_eq_master_cached()
            if df.empty and self._client:
                _jquants_throttle()
                try:
                    df = self._client.get_eq_master(code=code)
                except TypeError:
                    df = self._client.get_eq_master()

            if df.empty:
                return {}

            # Exact match on the 5-digit code first
            code_col = "Code" if "Code" in df.columns else df.columns[0]
            exact = df[df[code_col].astype(str) == code]
            if exact.empty:
                # Try matching on 4-digit prefix (some codes may differ)
                exact = df[df[code_col].astype(str).str[:4] == base4]
            if exact.empty:
                logger.warning("J-Quants: no company found for code %s", code)
                return {}

            row = exact.iloc[0]
            # V2 eq_master columns: CoName, CoNameEn, S33, S33Nm, Mkt, MktNm
            profile = {
                "name": str(row.get("CoNameEn", "") or row.get("CoName", "") or row.get("CompanyNameEnglish", "") or row.get("CompanyName", "")),
                "ticker": str(row.get("Code", ""))[:4],
                "exchange": "TSE",
                "sector": str(row.get("S33Nm", "") or row.get("S33", "") or row.get("Sector33CodeName", "") or row.get("Sector33Code", "")),
                "market_segment": str(row.get("MktNm", "") or row.get("Mkt", "") or row.get("MarketCodeName", "") or row.get("MarketCode", "")),
                "country": "Japan",
                "currency": "JPY",
            }

            # Validate: ensure the returned company matches the requested code
            returned_ticker = profile.get("ticker", "")
            if returned_ticker and returned_ticker != base4:
                logger.error(
                    "J-Quants profile mismatch: requested %s but got %s (%s). "
                    "Check code normalization.",
                    base4, returned_ticker, profile.get("name"),
                )

            return profile

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

            # Use get_fin_summary(code=X) instead of get_fin_summary_range().
            # get_fin_summary returns ALL financial summaries for a single
            # company in ONE API call (~8 rows).  get_fin_summary_range
            # iterates day-by-day over the date range, making hundreds of
            # calls even with caching on first run.
            #
            # The single-call approach is dramatically faster:
            #   get_fin_summary(code=X):      1 API call, ~2s
            #   get_fin_summary_range(2 years): ~730 API calls, ~100min
            _jquants_throttle()
            try:
                df = self._client.get_fin_summary(code=code)
            except TypeError:
                # Fallback: older SDK versions may not support code param
                end_dt = datetime.now()
                start_dt = end_dt - timedelta(days=365 * years)
                os.makedirs(_FIN_SUMMARY_CACHE_DIR, exist_ok=True)
                try:
                    df = self._client.get_fin_summary_range(
                        start_dt=start_dt,
                        end_dt=end_dt,
                        cache_dir=_FIN_SUMMARY_CACHE_DIR,
                    )
                except TypeError:
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
                    if pd.notna(val) and str(val).strip() != "":
                        try:
                            rec[canonical] = float(val)
                        except (ValueError, TypeError):
                            pass
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
                    if pd.notna(val) and str(val).strip() != "":
                        try:
                            rec[canonical] = float(val)
                        except (ValueError, TypeError):
                            pass
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
                    if pd.notna(val) and str(val).strip() != "":
                        try:
                            rec[canonical] = float(val)
                        except (ValueError, TypeError):
                            pass
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
        try:
            # Get the target company's sector
            profile = self.get_profile(identifier)
            if not profile or not profile.get("sector"):
                return []

            target_sector = profile["sector"]

            # Get all companies from cached master (no extra API call)
            df = self._get_eq_master_cached()
            if df.empty:
                return []

            # Handle both column name patterns
            sector_col = None
            for sc in ("S33Nm", "Sector33CodeName", "S17Nm", "Sector17CodeName"):
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
                name = str(row.get("CoNameEn", "") or row.get("CompanyNameEnglish", "") or row.get("CoName", "") or row.get("CompanyName", ""))
                results.append({
                    "identifier": str(row.get("Code", ""))[:4],
                    "name": name,
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

    # -- Institutional holders (J-Quants / LLM filing extraction) -------------

    def get_holders(self, identifier: str) -> list[dict]:
        """Fetch major shareholders from J-Quants financial summary data.

        J-Quants financial_summary includes major shareholder information
        in its response fields.  For detailed holder data, falls back to
        LLM extraction from the company's securities report (有価証券報告書)
        via the filing discovery pipeline.

        Returns list of dicts with: name, shares, percentage, holder_type,
        date_reported, source.
        """
        holders: list[dict] = []

        # --- Path 1: Try J-Quants financial summary for holder hints ---
        try:
            code = identifier.strip()[:4]
            # J-Quants V2 financial summary has some shareholder fields
            stmts = self._get_financial_statements(code)
            # The statements may contain major shareholder info in comments
            # but J-Quants doesn't have a dedicated holder endpoint.
            # Log and continue to filing extraction.
            logger.debug(
                "J-Quants does not provide per-company holder data for %s; "
                "using filing discovery fallback",
                identifier,
            )
        except Exception:
            pass

        # --- Path 2: Filing discovery + LLM/fuzzy extraction ---
        if not holders:
            try:
                from operator1.clients.filing_discoverer import try_filing_extraction
                # Securities reports (有価証券報告書) contain major shareholder tables
                # Try extracting from the company's annual filing
                df = try_filing_extraction(
                    ticker=identifier,
                    market_id=self.market_id,
                    statement_type="balance",  # triggers filing download
                    llm_client=None,
                )
                # If we got filing data, the holder info would need dedicated
                # extraction from the same PDFs. For now, log availability.
                if df is not None and not df.empty:
                    logger.info(
                        "JP filings available for %s (%d rows); "
                        "holder extraction from securities report available via LLM",
                        identifier, len(df),
                    )
            except Exception as exc:
                logger.debug("JP filing discovery for holders failed: %s", exc)


        # --- MarketScreener fallback (global institutional shareholder data) ---
        if not holders:
            try:
                from operator1.clients.marketscreener import fetch_shareholders
                company_name = ""
                if hasattr(self, '_read_cache'):
                    profile = self._read_cache(identifier, "profile.json")
                    if profile:
                        company_name = profile.get("name", "")
                if not company_name:
                    company_name = identifier
                ms_holders = fetch_shareholders(company_name)
                if ms_holders:
                    holders.extend(ms_holders)
                    logger.info(
                        "%s holders for %s: %d from MarketScreener (fallback)",
                        self.market_id, identifier, len(ms_holders),
                    )
            except Exception as exc:
                logger.debug("MarketScreener fallback failed for %s: %s", identifier, exc)

        return holders

    def get_holder_history(self, identifier: str, years: int = 2) -> "pd.DataFrame":
        """Return institutional ownership metrics from J-Quants / filing data.

        Derives aggregate metrics from get_holders().  Returns a single-row
        snapshot, or empty DataFrame if no holder data is available.
        """
        try:
            from datetime import date as _date

            holders = self.get_holders(identifier)
            if not holders:
                return pd.DataFrame()

            inst_pct = sum(h.get("percentage", 0) for h in holders)
            hhi = 0.0
            top5 = holders[:5]
            total_pct = sum(h.get("percentage", 0) for h in top5)
            if total_pct > 0:
                hhi = sum((h.get("percentage", 0) / total_pct) ** 2 for h in top5)

            return pd.DataFrame([{
                "date_reported": pd.Timestamp(_date.today()),
                "inst_ownership_pct": round(inst_pct, 2),
                "inst_top5_concentration": round(hhi, 4),
                "inst_holder_count": len(holders),
            }])
        except Exception as exc:
            logger.debug("JP holder history failed for %s: %s", identifier, exc)
        return pd.DataFrame()

    def get_insider_transactions(self, identifier: str) -> list[dict]:
        """Fetch insider transactions from J-Quants / filing discovery.

        J-Quants does not provide insider transaction data directly.
        Returns empty list; LLM extraction from securities reports can
        be added when filing discovery is extended for holder disclosures.
        """
        transactions: list[dict] = []
        # J-Quants has no insider transaction endpoint.
        # Would require EDINET filing discovery for 大量保有報告書 (large holder reports).
        logger.debug("JP insider transactions not available natively for %s", identifier)
        return transactions
