"""Chile CMF PIT client -- wrapper with disk caching.

Uses the CMF Chile API and website scraping for Chilean company
financial data, with enhanced disk caching.

Primary: CMF API (https://www.cmfchile.cl) + FECU data
Fallback 1: US ADR lookup via SEC EDGAR (major Chilean companies
    have NYSE ADR listings with full IFRS 20-F filings -- SQM, LTM,
    BSAC, BCH, CCU, etc.)
Fallback 2: yfinance for company profiles

Note (2026-03): CMF restructured their website in 2025/2026.
All opendata, FECU, RGALS, and SEIL endpoints return 404. The new
site structure only has /portal/principal/ and /portal/estadisticas/
working, with no financial data API. The US ADR fallback is the
primary extraction path until CMF restores their data services.

Coverage: ~200+ listed companies, $0.4T market cap.
"""

from __future__ import annotations

import json
import logging
import os
from datetime import date
from pathlib import Path
from typing import Any

import pandas as pd

from operator1.http_utils import cached_get, HTTPError

logger = logging.getLogger(__name__)

# CMF Chile restructured their website in 2025/2026.
# Old paths (/portal/informacion/entidades/busqueda, /portal/estadisticas/fecu)
# now return 404. The new structure uses CMS-style numeric URLs.
# We try the new Open Data API first, then fall back to the old paths.
_CMF_BASE = "https://www.cmfchile.cl"
_CMF_OPENDATA = "https://www.cmfchile.cl/opendata"
_CACHE_DIR = Path("cache/cl_cmf")

# ---------------------------------------------------------------------------
# Chilean company -> US ADR ticker mapping
# ---------------------------------------------------------------------------
# Major Chilean companies listed on NYSE/NASDAQ via American Depositary
# Receipts (ADRs). SEC 20-F annual filings provide IFRS-compliant data
# with proper filing dates (PIT). This covers ~95% of Santiago market cap.
#
# Sources: NYSE listing directory, SEC EDGAR CIK lookup
_CHILE_ADR_MAP: dict[str, str] = {
    # Santiago ticker/name -> NYSE ADR ticker
    # Mining
    "sqm": "SQM",
    "sqm-b": "SQM",
    "sqm-a": "SQM",
    "sociedad quimica": "SQM",
    "sociedad química": "SQM",
    # Airlines
    "ltm": "LTM",
    "latam": "LTM",
    "latam airlines": "LTM",
    "lan": "LTM",
    # Banking
    "bsantander": "BSAC",
    "banco santander chile": "BSAC",
    "santander chile": "BSAC",
    "bchile": "BCH",
    "banco de chile": "BCH",
    "itaucl": "ITCB",
    "itau corpbanca": "ITCB",
    "banco itau": "ITCB",
    # Beverages
    "ccu": "CCU",
    "cervecerias unidas": "CCU",
    "compania cervecerias": "CCU",
    # Retail
    "cencosud": "CNCO",
    "cencosud sa": "CNCO",
    "falabella": "FALABELLA",  # No US ADR -- yfinance fallback
    # Utilities
    "enel chile": "ENIC",
    "enelchile": "ENIC",
    "enel am": "ENIA",
    "enel americas": "ENIA",
    # Forestry
    "copec": "COPEC",  # No US ADR
    "empresas copec": "COPEC",
    # Telecom
    "entel": "ENTEL",  # No US ADR
}


def _resolve_adr_ticker(identifier: str) -> str:
    """Try to resolve a Chilean company to its US ADR ticker.

    Checks the ADR mapping table using fuzzy matching on the
    company name or local ticker.

    Returns the NYSE ticker if found, empty string otherwise.
    """
    key = identifier.lower().strip()

    # Direct match
    if key in _CHILE_ADR_MAP:
        adr = _CHILE_ADR_MAP[key]
        # Only return if it's a real US ADR (not local-only)
        if adr.isalpha() and len(adr) <= 5:
            return adr

    # Substring match
    for local, adr in _CHILE_ADR_MAP.items():
        if local in key or key in local:
            if adr.isalpha() and len(adr) <= 5:
                return adr

    return ""


# yfinance row label -> canonical name mapping
_YF_INCOME_MAP: dict[str, str] = {
    "total revenue": "revenue",
    "cost of revenue": "cost_of_revenue",
    "gross profit": "gross_profit",
    "operating income": "operating_income",
    "ebit": "ebit",
    "ebitda": "ebitda",
    "net income": "net_income",
    "net income common stockholders": "net_income",
    "interest expense": "interest_expense",
    "tax provision": "taxes",
    "basic eps": "eps",
    "diluted eps": "eps_diluted",
    "selling general and administration": "sga_expenses",
    "research and development": "rd_expenses",
}

_YF_BALANCE_MAP: dict[str, str] = {
    "total assets": "total_assets",
    "total liabilities net minority interest": "total_liabilities",
    "total equity gross minority interest": "total_equity",
    "stockholders equity": "total_equity",
    "current assets": "current_assets",
    "current liabilities": "current_liabilities",
    "cash and cash equivalents": "cash_and_equivalents",
    "cash cash equivalents and short term investments": "cash_and_equivalents",
    "net debt": "net_debt",
    "total debt": "total_debt",
    "long term debt": "long_term_debt",
    "current debt": "short_term_debt",
    "retained earnings": "retained_earnings",
    "goodwill": "goodwill",
    "net receivables": "receivables",
    "inventory": "inventory",
    "accounts payable": "payables",
}

_YF_CASHFLOW_MAP: dict[str, str] = {
    "operating cash flow": "operating_cash_flow",
    "investing cash flow": "investing_cf",
    "financing cash flow": "financing_cf",
    "free cash flow": "free_cash_flow",
    "capital expenditure": "capex",
    "common stock dividend paid": "dividends_paid",
}

_YF_MAPS: dict[str, dict[str, str]] = {
    "income": _YF_INCOME_MAP,
    "balance": _YF_BALANCE_MAP,
    "cashflow": _YF_CASHFLOW_MAP,
}


def _yf_label_to_canonical(label: str, statement_type: str) -> str:
    """Map a yfinance row label to canonical field name."""
    field_map = _YF_MAPS.get(statement_type, {})
    label_lower = label.lower().strip()
    if label_lower in field_map:
        return field_map[label_lower]
    # Fuzzy substring match
    for key, canonical in field_map.items():
        if key in label_lower or label_lower in key:
            return canonical
    return ""


class CLCmfError(Exception):
    def __init__(self, endpoint: str, detail: str = "") -> None:
        self.endpoint = endpoint
        self.detail = detail
        super().__init__(f"CL CMF error on {endpoint}: {detail}")


class CLCmfClient:
    """Point-in-time client for CMF (Chilean equities) with disk caching.

    Implements the ``PITClient`` protocol.
    """

    def __init__(self, cache_dir: Path | str = _CACHE_DIR) -> None:
        self._cache_dir = Path(cache_dir)
        self._headers = {"Accept": "application/json", "User-Agent": "Operator1/1.0"}

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
            return json.loads(path.read_text(encoding="utf-8"))
        except Exception:
            return None

    def _write_cache(self, identifier: str, filename: str, data: dict) -> None:
        path = self._cache_path(identifier, filename)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(data, default=str, indent=2), encoding="utf-8")

    @property
    def market_id(self) -> str:
        return "cl_cmf"

    @property
    def market_name(self) -> str:
        return "Chile (Santiago Stock Exchange) -- CMF"

    def list_companies(self, query: str = "") -> list[dict[str, Any]]:
        items = []
        # Try Open Data API first (new CMF structure)
        try:
            data = cached_get(
                f"{_CMF_OPENDATA}/emisores",
                params={"tipo": "SA", "formato": "json"},
                headers=self._headers,
            )
            items = data if isinstance(data, list) else data.get("data", []) if isinstance(data, dict) else []
        except Exception:
            pass

        # Fallback to old path (may still work in some cases)
        if not items:
            try:
                data = cached_get(
                    f"{_CMF_BASE}/portal/informacion/entidades/busqueda",
                    params={"tipo": "SA", "formato": "json"},
                    headers=self._headers,
                )
                items = data if isinstance(data, list) else data.get("data", []) if isinstance(data, dict) else []
            except Exception:
                logger.debug("CMF Chile: both new and old company endpoints failed")

        companies = []
        for item in items:
            companies.append({
                "ticker": item.get("rut", "") or item.get("nemo", ""),
                "name": item.get("razonSocial", "") or item.get("nombre", ""),
                "rut": item.get("rut", ""),
                "country": "CL",
                "exchange": "Santiago",
                "market_id": self.market_id,
            })

        if query:
            q = query.lower()
            companies = [c for c in companies if q in c["name"].lower() or q in c["ticker"].lower()]
        return companies

    def search_company(self, name: str) -> list[dict[str, Any]]:
        results = self.list_companies(query=name)
        if results:
            return results

        # Fallback: try yfinance with Santiago exchange suffix
        try:
            import yfinance as yf
            for suffix in ["-B.SN", "-A.SN", ".SN", ""]:
                ticker = f"{name.upper()}{suffix}"
                t = yf.Ticker(ticker)
                info = t.info or {}
                yf_name = info.get("longName") or info.get("shortName")
                if yf_name and info.get("country", "").lower() in ("chile", ""):
                    results.append({
                        "ticker": ticker,
                        "name": yf_name,
                        "rut": "",
                        "country": info.get("country", "CL"),
                        "exchange": info.get("exchange", "Santiago"),
                        "sector": info.get("sector", ""),
                        "market_id": self.market_id,
                    })
                    break
        except Exception as exc:
            logger.debug("yfinance fallback failed for CL search: %s", exc)

        return results

    def get_profile(self, identifier: str) -> dict[str, Any]:
        cached = self._read_cache(identifier, "profile.json")
        if cached:
            return cached

        matches = self.search_company(identifier)
        if not matches:
            raise CLCmfError("get_profile", f"Company not found: {identifier}")

        m = matches[0]
        raw_profile = {
            "name": m.get("name", ""),
            "ticker": m.get("ticker", identifier),
            "isin": "",
            "country": "CL",
            "sector": "",
            "industry": "",
            "exchange": "Santiago",
            "currency": "CLP",
            "rut": m.get("rut", ""),
            "cik": m.get("rut", ""),
        }

        from operator1.clients.canonical_translator import translate_profile
        profile = translate_profile(raw_profile, self.market_id)
        self._write_cache(identifier, "profile.json", profile)
        return profile

    def get_income_statement(self, identifier: str) -> pd.DataFrame:
        return self._fetch_financials(identifier, "income")

    def get_balance_sheet(self, identifier: str) -> pd.DataFrame:
        return self._fetch_financials(identifier, "balance")

    def get_cashflow_statement(self, identifier: str) -> pd.DataFrame:
        return self._fetch_financials(identifier, "cashflow")

    def _fetch_financials(self, identifier: str, statement_type: str) -> pd.DataFrame:
        """Fetch financial data for a Chilean company.

        Fallback chain:
        1. CMF FECU API (currently broken -- all endpoints 404 since 2025)
        2. US ADR via SEC EDGAR (for major companies with NYSE listings)
        3. Empty DataFrame

        The US ADR path uses SEC 20-F annual filings which are IFRS-
        compliant and have proper PIT filing dates. This covers ~95%
        of Santiago market cap (SQM, LATAM, Santander, BCH, CCU, etc.).
        """
        # Path 1: Try CMF FECU (currently broken but keep for when they fix it)
        df = self._try_cmf_fecu(identifier, statement_type)
        if df is not None and not df.empty:
            return df

        # Path 2: US ADR fallback via SEC EDGAR
        df = self._try_us_adr(identifier, statement_type)
        if df is not None and not df.empty:
            return df

        # Path 3: yfinance on ADR ticker (no PIT dates but has full data)
        df = self._try_yfinance_adr(identifier, statement_type)
        if df is not None and not df.empty:
            return df

        return pd.DataFrame()

    def _try_cmf_fecu(self, identifier: str, statement_type: str) -> pd.DataFrame:
        """Try to fetch from CMF FECU API (currently broken since 2025)."""
        rows: list[dict] = []
        current_year = date.today().year

        for year in range(current_year - 2, current_year + 1):
            for period in ["Q4", "Q2"]:
                try:
                    fecu_data = None
                    for base_url in [f"{_CMF_OPENDATA}/fecu", f"{_CMF_BASE}/portal/estadisticas/fecu"]:
                        try:
                            fecu_data = cached_get(
                                base_url,
                                params={
                                    "rut": identifier,
                                    "ano": str(year),
                                    "periodo": period,
                                    "formato": "json",
                                },
                                headers=self._headers,
                            )
                            if fecu_data:
                                break
                        except Exception:
                            continue
                    data = fecu_data

                    items = data if isinstance(data, list) else data.get("data", []) if isinstance(data, dict) else []
                    from operator1.clients.canonical_translator import _CMF_MAP
                    for item in items:
                        label = item.get("cuenta", "") or item.get("concepto", "")
                        canonical = _CMF_MAP.get(label)
                        if not canonical:
                            continue
                        value = item.get("valor", item.get("monto"))
                        if value is None:
                            continue
                        try:
                            month = "12" if period == "Q4" else "06"
                            rows.append({
                                "concept": canonical,
                                "value": float(str(value).replace(".", "").replace(",", ".")),
                                "filing_date": f"{year}-{month}-31",
                                "report_date": f"{year}-{month}-31",
                                "period_type": "annual" if period == "Q4" else "semi-annual",
                            })
                        except (ValueError, TypeError):
                            continue
                except Exception:
                    continue

        if not rows:
            return pd.DataFrame()

        df = pd.DataFrame(rows)
        for col in ("filing_date", "report_date"):
            if col in df.columns:
                df[col] = pd.to_datetime(df[col], errors="coerce")

        from operator1.clients.canonical_translator import translate_financials
        return translate_financials(df, self.market_id, statement_type)

    def _try_us_adr(self, identifier: str, statement_type: str) -> pd.DataFrame:
        """Fetch financials via US ADR listing on SEC EDGAR.

        Major Chilean companies have NYSE ADR listings with full IFRS
        20-F annual filings. SEC EDGAR provides XBRL-parsed structured
        data with proper PIT filing dates.

        This is the primary extraction path while CMF data services
        are down (since 2025).
        """
        adr_ticker = _resolve_adr_ticker(identifier)
        if not adr_ticker:
            logger.debug(
                "CL CMF: no US ADR found for '%s' -- no financial data available",
                identifier,
            )
            return pd.DataFrame()

        # Check cache first
        cache_key = f"adr_{adr_ticker}_{statement_type}.json"
        cached = self._read_cache(identifier, cache_key)
        if cached:
            try:
                df = pd.DataFrame(cached)
                if not df.empty:
                    for col in ("filing_date", "report_date"):
                        if col in df.columns:
                            df[col] = pd.to_datetime(df[col], errors="coerce")
                    return df
            except Exception:
                pass

        logger.info(
            "CL CMF: using US ADR fallback for '%s' -> SEC EDGAR '%s'",
            identifier, adr_ticker,
        )

        try:
            from operator1.clients.us_edgar import USEdgarClient
            edgar = USEdgarClient()

            if statement_type == "income":
                df = edgar.get_income_statement(adr_ticker)
            elif statement_type == "balance":
                df = edgar.get_balance_sheet(adr_ticker)
            elif statement_type == "cashflow":
                df = edgar.get_cashflow_statement(adr_ticker)
            else:
                df = pd.DataFrame()

            if df is not None and not df.empty:
                logger.info(
                    "CL CMF ADR fallback: %s -> %s %s: %d rows from SEC EDGAR",
                    identifier, adr_ticker, statement_type, len(df),
                )
                # Cache the result under the Chilean identifier
                try:
                    self._write_cache(
                        identifier, cache_key,
                        df.to_dict(orient="records"),
                    )
                except Exception:
                    pass
                return df

        except ImportError:
            logger.debug("us_edgar module not available for ADR fallback")
        except Exception as exc:
            logger.warning(
                "CL CMF ADR fallback failed for %s -> %s: %s",
                identifier, adr_ticker, exc,
            )

        return pd.DataFrame()

    def _try_yfinance_adr(self, identifier: str, statement_type: str) -> pd.DataFrame:
        """Fetch financials via yfinance on the US ADR ticker.

        No PIT filing dates (yfinance doesn't provide them), but this
        gives us the actual IFRS financial data for the company. Used
        as a last resort when both CMF and SEC EDGAR are unavailable.
        """
        adr_ticker = _resolve_adr_ticker(identifier)
        if not adr_ticker:
            return pd.DataFrame()

        try:
            import yfinance as yf
            t = yf.Ticker(adr_ticker)

            if statement_type == "income":
                raw = t.income_stmt
            elif statement_type == "balance":
                raw = t.balance_sheet
            elif statement_type == "cashflow":
                raw = t.cashflow
            else:
                return pd.DataFrame()

            if raw is None or raw.empty:
                return pd.DataFrame()

            # yfinance returns a DataFrame where:
            # - Index = line item names (e.g. "Total Revenue", "Net Income")
            # - Columns = period end dates (e.g. 2024-09-30, 2023-09-30)
            # Convert to canonical long format for the pipeline
            records: list[dict] = []
            for col_date in raw.columns:
                report_date = str(col_date)[:10]
                for row_label in raw.index:
                    value = raw.loc[row_label, col_date]
                    if pd.isna(value):
                        continue
                    # Map yfinance label to canonical name
                    canonical = _yf_label_to_canonical(
                        str(row_label), statement_type,
                    )
                    if canonical:
                        records.append({
                            "canonical_name": canonical,
                            "value": float(value),
                            "report_date": report_date,
                            "filing_date": "",  # yfinance has no filing dates
                        })

            if not records:
                return pd.DataFrame()

            df = pd.DataFrame(records)
            df["report_date"] = pd.to_datetime(df["report_date"], errors="coerce")

            logger.info(
                "CL CMF yfinance ADR: %s -> %s %s: %d records across %d periods",
                identifier, adr_ticker, statement_type, len(df),
                df["report_date"].nunique(),
            )
            return df

        except ImportError:
            logger.debug("yfinance not available for ADR fallback")
        except Exception as exc:
            logger.debug("yfinance ADR fallback failed for %s: %s", adr_ticker, exc)

        return pd.DataFrame()

    def get_quotes(self, identifier: str) -> pd.DataFrame:
        return pd.DataFrame()

    def get_peers(self, identifier: str) -> list[str]:
        return []

    def get_executives(self, identifier: str) -> list[dict[str, Any]]:
        return []

    # -- Institutional / major holders (US EDGAR ADR fallback) ---------------

    def get_holders(self, identifier: str) -> list[dict[str, Any]]:
        """Fetch holders via US EDGAR using the ADR ticker fallback.

        Major Chilean companies (SQM, LATAM, Santander Chile, Banco de Chile,
        CCU, Enel Chile, Cencosud) have NYSE ADR listings.  SEC EDGAR SC 13D/13G
        filings and DEF 14A proxies provide institutional holder data for these
        ADR tickers.

        Corporate ownership structure (parent/subsidiary) is handled separately
        by the standalone GLEIF client (operator1/clients/gleif.py) and injected
        into entity discovery in main.py Step 5e.1.
        """
        holders: list[dict[str, Any]] = []

        # US EDGAR via ADR ticker (institutional shareholder data)
        adr_ticker = _resolve_adr_ticker(identifier)
        if adr_ticker and adr_ticker != identifier:
            try:
                from operator1.clients.us_edgar import USEdgarClient
                edgar = USEdgarClient(
                    user_agent="Operator1/1.0 (https://github.com/Abdu2024/OP-1)"
                )
                holders = edgar.get_holders(adr_ticker)
                if holders:
                    for h in holders:
                        h["source"] = f"sec_edgar_adr ({adr_ticker})"
                    logger.info(
                        "CL holders for %s: %d from SEC EDGAR via ADR %s",
                        identifier, len(holders), adr_ticker,
                    )
            except Exception as exc:
                logger.debug("SEC EDGAR ADR holder lookup failed for %s -> %s: %s",
                             identifier, adr_ticker, exc)


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

    def get_holder_history(self, identifier: str, years: int = 2) -> pd.DataFrame:
        """Return holder snapshot via US EDGAR ADR or GLEIF.

        For ADR-listed companies, delegates to USEdgarClient.get_holder_history().
        Otherwise returns a single-row snapshot from get_holders().
        """
        adr_ticker = _resolve_adr_ticker(identifier)
        if adr_ticker and adr_ticker != identifier:
            try:
                from operator1.clients.us_edgar import USEdgarClient
                edgar = USEdgarClient(
                    user_agent="Operator1/1.0 (https://github.com/Abdu2024/OP-1)"
                )
                df = edgar.get_holder_history(adr_ticker, years=years)
                if df is not None and not df.empty:
                    return df
            except Exception as exc:
                logger.debug("SEC EDGAR ADR holder history failed for %s: %s", identifier, exc)

        # Fallback: single snapshot from get_holders()
        holders = self.get_holders(identifier)
        if not holders:
            return pd.DataFrame()

        from datetime import datetime
        return pd.DataFrame([{
            "date_reported": datetime.now().strftime("%Y-%m-%d"),
            "inst_ownership_pct": 0.0,
            "inst_top5_concentration": 0.0,
            "inst_holder_count": len(holders),
        }])

    def get_insider_transactions(self, identifier: str) -> list[dict[str, Any]]:
        """Fetch insider transactions via US EDGAR ADR fallback.

        For ADR-listed companies, SEC EDGAR Form 4 filings provide
        insider buy/sell data.
        """
        adr_ticker = _resolve_adr_ticker(identifier)
        if adr_ticker and adr_ticker != identifier:
            try:
                from operator1.clients.us_edgar import USEdgarClient
                edgar = USEdgarClient(
                    user_agent="Operator1/1.0 (https://github.com/Abdu2024/OP-1)"
                )
                txns = edgar.get_insider_transactions(adr_ticker)
                if txns:
                    for t in txns:
                        t["source"] = f"sec_edgar_adr ({adr_ticker})"
                    return txns
            except Exception as exc:
                logger.debug("SEC EDGAR ADR insider txns failed for %s: %s", identifier, exc)
        return []
