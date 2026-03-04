"""South Korea DART PIT client -- powered by dart-fss library.

Replaces the original dart.py with the dart-fss unofficial wrapper
for richer access to Korean financial disclosure data.

Primary library: dart-fss (https://github.com/josw123/dart-fss)
  - Corporate code lookup, financial statement extraction
  - XBRL parsing, structured DataFrames

Fallback: Direct DART Open API calls via requests

Coverage: ~2,500+ listed companies on KOSPI/KOSDAQ, $2.5T market cap.
API: https://opendart.fss.or.kr/api (free, API key required)
"""

from __future__ import annotations

import json
import logging
import os
import time
from datetime import date, timedelta
from pathlib import Path
from typing import Any

import pandas as pd

logger = logging.getLogger(__name__)

_DART_BASE = "https://opendart.fss.or.kr/api"
_CACHE_DIR = Path("cache/kr_dart")

# DART free plan: 10,000 req/day, ~1,000 req/min.
# We add a small delay between API calls to stay well within limits.
_DART_MIN_INTERVAL = 0.15  # seconds between API calls
_dart_last_call: float = 0.0


def _dart_throttle() -> None:
    """Enforce minimum interval between DART API calls."""
    global _dart_last_call
    elapsed = time.monotonic() - _dart_last_call
    if elapsed < _DART_MIN_INTERVAL:
        time.sleep(_DART_MIN_INTERVAL - elapsed)
    _dart_last_call = time.monotonic()


class KRDartError(Exception):
    """Raised on Korea DART wrapper failures."""
    def __init__(self, endpoint: str, detail: str = "") -> None:
        self.endpoint = endpoint
        self.detail = detail
        super().__init__(f"KR DART error on {endpoint}: {detail}")


class KRDartClient:
    """Point-in-time client for DART (South Korean equities) using dart-fss.

    Implements the ``PITClient`` protocol. Uses dart-fss as primary
    library with direct DART API as fallback.

    Parameters
    ----------
    api_key:
        DART Open API key (free registration at opendart.fss.or.kr).
        Also loads from DART_API_KEY env var.
    cache_dir:
        Local cache directory.
    """

    def __init__(
        self,
        api_key: str = "",
        cache_dir: Path | str = _CACHE_DIR,
    ) -> None:
        self._api_key = api_key or os.environ.get("DART_API_KEY", "")
        self._cache_dir = Path(cache_dir)
        self._dart_fss_available = False
        self._corp_list_cache: list[dict[str, Any]] | None = None

        # Initialize dart-fss if available
        if self._api_key:
            try:
                import dart_fss
                dart_fss.set_api_key(self._api_key)
                self._dart_fss_available = True
                logger.info("dart-fss initialized with API key")
            except ImportError:
                logger.warning("dart-fss not installed; using direct DART API")
            except Exception as exc:
                logger.warning("dart-fss init failed: %s", exc)

    # -- Cache helpers --------------------------------------------------------

    def _cache_path(self, identifier: str, filename: str) -> Path:
        safe_id = identifier.replace("/", "_").replace("\\", "_").upper()
        return self._cache_dir / safe_id / filename

    def _read_cache(self, identifier: str, filename: str) -> dict | None:
        path = self._cache_path(identifier, filename)
        if not path.exists():
            return None
        try:
            stat = path.stat()
            age_days = (date.today() - date.fromtimestamp(stat.st_mtime)).days
            if filename == "profile.json" and age_days > 7:
                return None
            return json.loads(path.read_text(encoding="utf-8"))
        except Exception:
            return None

    def _write_cache(self, identifier: str, filename: str, data: dict) -> None:
        path = self._cache_path(identifier, filename)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(data, default=str, indent=2), encoding="utf-8")

    # -- Protocol properties -------------------------------------------------

    @property
    def market_id(self) -> str:
        return "kr_dart"

    @property
    def market_name(self) -> str:
        return "South Korea (KOSPI / KOSDAQ) -- DART"

    # -- Company discovery ---------------------------------------------------

    def list_companies(self, query: str = "") -> list[dict[str, Any]]:
        """List Korean companies from DART corporate code list."""
        if self._corp_list_cache is None:
            self._corp_list_cache = self._fetch_corp_list()

        if not query:
            return self._corp_list_cache

        q = query.lower()
        return [
            c for c in self._corp_list_cache
            if q in c.get("name", "").lower()
            or q in c.get("ticker", "").lower()
            or q in c.get("corp_code", "").lower()
        ]

    def search_company(self, name: str) -> list[dict[str, Any]]:
        """Search for a Korean company by name, ticker, or corp code."""
        if self._dart_fss_available:
            try:
                import dart_fss
                _dart_throttle()
                corp_list = dart_fss.get_corp_list()
                results = corp_list.find_by_corp_name(name, exactly=False)
                if results:
                    return [
                        {
                            "ticker": getattr(r, "stock_code", "") or "",
                            "name": getattr(r, "corp_name", str(r)),
                            "corp_code": getattr(r, "corp_code", ""),
                            "cik": getattr(r, "corp_code", ""),
                            "exchange": "KRX",
                            "market_id": self.market_id,
                        }
                        for r in results[:20]
                    ]
            except Exception as exc:
                logger.debug("dart-fss search failed: %s", exc)

        return self.list_companies(query=name)

    def _fetch_corp_list(self) -> list[dict[str, Any]]:
        """Build company list from dart-fss or direct API."""
        if self._dart_fss_available:
            try:
                import dart_fss
                _dart_throttle()
                corp_list = dart_fss.get_corp_list()
                companies = []
                for corp in corp_list.corps:
                    stock_code = getattr(corp, "stock_code", "") or ""
                    if stock_code:  # Only listed companies
                        companies.append({
                            "ticker": stock_code,
                            "name": getattr(corp, "corp_name", ""),
                            "corp_code": getattr(corp, "corp_code", ""),
                            "cik": getattr(corp, "corp_code", ""),
                            "exchange": "KRX",
                            "market_id": self.market_id,
                        })
                if companies:
                    return companies
            except Exception as exc:
                logger.debug("dart-fss corp list failed: %s", exc)

        # Fallback: search via DART API
        return self._search_via_direct_api("")

    def _search_via_direct_api(self, query: str) -> list[dict[str, Any]]:
        """Search companies via direct DART API."""
        from operator1.http_utils import cached_get, HTTPError

        params: dict[str, Any] = {
            "crtfc_key": self._api_key,
            "page_count": 100,
            "page_no": 1,
        }
        if query:
            params["corp_name"] = query

        try:
            _dart_throttle()
            data = cached_get(f"{_DART_BASE}/list.json", params=params)
            items = data.get("list", []) if isinstance(data, dict) else []
            seen: dict[str, dict] = {}
            for item in items:
                corp_name = item.get("corp_name", "")
                corp_code = item.get("corp_code", "")
                stock_code = item.get("stock_code", "")
                if corp_code and corp_code not in seen:
                    seen[corp_code] = {
                        "ticker": stock_code or "",
                        "name": corp_name,
                        "corp_code": corp_code,
                        "cik": corp_code,
                        "exchange": "KRX",
                        "market_id": self.market_id,
                    }
            return list(seen.values())
        except Exception:
            return []

    # -- Company profile -----------------------------------------------------

    def get_profile(self, identifier: str) -> dict[str, Any]:
        """Fetch company profile from DART.

        Parameters
        ----------
        identifier:
            Stock code (e.g. "005930"), corp_code, or company name.
        """
        cached = self._read_cache(identifier, "profile.json")
        if cached:
            return cached

        raw_profile: dict[str, Any] = {}

        if self._dart_fss_available:
            try:
                import dart_fss
                _dart_throttle()
                corp_list = dart_fss.get_corp_list()
                # Try stock code first, then name
                corp = None
                if identifier.isdigit():
                    results = [c for c in corp_list.corps if getattr(c, "stock_code", "") == identifier]
                    corp = results[0] if results else None

                if corp is None:
                    results = corp_list.find_by_corp_name(identifier, exactly=False)
                    corp = results[0] if results else None

                if corp:
                    raw_profile = {
                        "name": getattr(corp, "corp_name", ""),
                        "ticker": getattr(corp, "stock_code", identifier) or identifier,
                        "isin": "",
                        "country": "KR",
                        "sector": "",
                        "industry": "",
                        "exchange": "KRX",
                        "currency": "KRW",
                        "corp_code": getattr(corp, "corp_code", ""),
                        "cik": getattr(corp, "corp_code", ""),
                    }
            except Exception as exc:
                logger.debug("dart-fss profile failed for %s: %s", identifier, exc)

        if not raw_profile:
            # Fallback: use DART company.json endpoint
            from operator1.http_utils import cached_get
            try:
                corp_code = self._resolve_corp_code(identifier)
                _dart_throttle()
                data = cached_get(
                    f"{_DART_BASE}/company.json",
                    params={"crtfc_key": self._api_key, "corp_code": corp_code},
                )
                if isinstance(data, dict) and data.get("status") == "000":
                    raw_profile = {
                        "name": data.get("corp_name", ""),
                        "ticker": data.get("stock_code", identifier) or identifier,
                        "isin": "",
                        "country": "KR",
                        "sector": data.get("induty_code", ""),
                        "industry": data.get("induty_code", ""),
                        "exchange": data.get("stock_name", "KRX"),
                        "currency": "KRW",
                        "corp_code": corp_code,
                        "cik": corp_code,
                    }
            except Exception as exc:
                logger.debug("DART direct profile failed: %s", exc)

        if not raw_profile:
            raise KRDartError("get_profile", f"Company not found: {identifier}")

        from operator1.clients.canonical_translator import translate_profile
        profile = translate_profile(raw_profile, self.market_id)
        self._write_cache(identifier, "profile.json", profile)
        return profile

    # -- Financial statements ------------------------------------------------

    def get_income_statement(self, identifier: str) -> pd.DataFrame:
        """Fetch income statements with filing_date and report_date."""
        return self._fetch_financials(identifier, "income")

    def get_balance_sheet(self, identifier: str) -> pd.DataFrame:
        """Fetch balance sheets with filing_date and report_date."""
        return self._fetch_financials(identifier, "balance")

    def get_cashflow_statement(self, identifier: str) -> pd.DataFrame:
        """Fetch cash flow statements with filing_date and report_date."""
        return self._fetch_financials(identifier, "cashflow")

    def _fetch_financials(self, identifier: str, statement_type: str) -> pd.DataFrame:
        """Extract financial statements using dart-fss or direct API."""
        # Try dart-fss first
        if self._dart_fss_available:
            try:
                df = self._fetch_via_dart_fss(identifier, statement_type)
                if df is not None and not df.empty:
                    return df
            except Exception as exc:
                logger.warning("dart-fss financials failed for %s: %s", identifier, exc)

        # Fallback: direct DART API (fnlttSinglAcntAll.json)
        return self._fetch_via_direct_api_financials(identifier, statement_type)

    def _fetch_via_dart_fss(self, identifier: str, statement_type: str) -> pd.DataFrame | None:
        """Use dart-fss to get structured financial statements."""
        import dart_fss

        _dart_throttle()
        corp_list = dart_fss.get_corp_list()
        corp = None

        if identifier.isdigit():
            results = [c for c in corp_list.corps if getattr(c, "stock_code", "") == identifier]
            corp = results[0] if results else None

        if corp is None:
            results = corp_list.find_by_corp_name(identifier, exactly=False)
            corp = results[0] if results else None

        if corp is None:
            return None

        # Map statement_type to dart-fss report type
        fs_type_map = {
            "income": "CFS",  # Consolidated Financial Statements
            "balance": "CFS",
            "cashflow": "CFS",
        }

        try:
            # Get financial statements for last 2 years
            fs = corp.extract_fs(
                bgn_de=(date.today() - timedelta(days=730)).strftime("%Y%m%d"),
                end_de=date.today().strftime("%Y%m%d"),
            )

            if fs is None:
                return None

            # dart-fss returns a dict of DataFrames keyed by report type
            rows: list[dict] = []
            for report_key, report_df in fs.items() if isinstance(fs, dict) else [(None, fs)]:
                if not isinstance(report_df, pd.DataFrame) or report_df.empty:
                    continue

                for _, row in report_df.iterrows():
                    account_name = str(row.get("account_nm", row.get("sj_nm", "")))
                    canonical = self._map_dart_concept(account_name, statement_type)
                    if not canonical:
                        continue

                    # Extract value columns (period columns)
                    for col in report_df.columns:
                        if col in ("account_nm", "sj_nm", "sj_div", "account_id", "thstrm_nm"):
                            continue
                        value = row.get(col)
                        if pd.isna(value):
                            continue
                        try:
                            rows.append({
                                "concept": canonical,
                                "value": float(str(value).replace(",", "")),
                                "filing_date": "",  # Will be filled from filings
                                "report_date": str(col),
                                "period_type": "annual",
                            })
                        except (ValueError, TypeError):
                            continue

            if not rows:
                return None

            df = pd.DataFrame(rows)
            for col in ("filing_date", "report_date"):
                if col in df.columns:
                    df[col] = pd.to_datetime(df[col], errors="coerce")

            from operator1.clients.canonical_translator import translate_financials
            return translate_financials(df, self.market_id, statement_type)

        except Exception as exc:
            logger.debug("dart-fss extract_fs failed: %s", exc)
            return None

    def _fetch_via_direct_api_financials(
        self, identifier: str, statement_type: str,
    ) -> pd.DataFrame:
        """Fallback: use DART fnlttSinglAcntAll.json endpoint."""
        from operator1.http_utils import cached_get

        corp_code = self._resolve_corp_code(identifier)
        if not corp_code:
            return pd.DataFrame()

        # DART financial statement types
        fs_div_map = {
            "income": "IS",   # Income Statement
            "balance": "BS",  # Balance Sheet
            "cashflow": "CF", # Cash Flow
        }
        fs_div = fs_div_map.get(statement_type, "IS")

        rows: list[dict] = []
        current_year = date.today().year

        for year in range(current_year - 2, current_year + 1):
            for reprt_code in ["11011", "11014", "11012", "11013"]:
                # 11011=annual, 11014=Q3, 11012=semi, 11013=Q1
                try:
                    data = cached_get(
                        f"{_DART_BASE}/fnlttSinglAcntAll.json",
                        params={
                            "crtfc_key": self._api_key,
                            "corp_code": corp_code,
                            "bsns_year": str(year),
                            "reprt_code": reprt_code,
                            "fs_div": "CFS",  # Consolidated
                        },
                    )

                    items = data.get("list", []) if isinstance(data, dict) else []
                    for item in items:
                        account_name = item.get("account_nm", "")
                        canonical = self._map_dart_concept(account_name, statement_type)
                        if not canonical:
                            continue

                        # Get current period amount
                        value_str = item.get("thstrm_amount", "")
                        if not value_str or value_str == "-":
                            continue

                        try:
                            value = float(value_str.replace(",", ""))
                        except (ValueError, TypeError):
                            continue

                        is_annual = reprt_code == "11011"
                        rows.append({
                            "concept": canonical,
                            "value": value,
                            "filing_date": item.get("rcept_no", ""),
                            "report_date": f"{year}-12-31" if is_annual else f"{year}-{['03','06','09','12'][int(reprt_code[-1])-1]}-30",
                            "period_type": "annual" if is_annual else "quarterly",
                            "form": "Annual" if is_annual else "Quarterly",
                        })
                except Exception:
                    continue

        if not rows:
            return pd.DataFrame()

        df = pd.DataFrame(rows)
        for col in ("filing_date", "report_date"):
            if col in df.columns:
                df[col] = pd.to_datetime(df[col], errors="coerce")

        # Cache filings
        self._cache_filings(identifier, df)

        from operator1.clients.canonical_translator import translate_financials
        return translate_financials(df, self.market_id, statement_type)

    def _map_dart_concept(self, concept: str, statement_type: str) -> str | None:
        """Map DART Korean account name to canonical name."""
        from operator1.clients.canonical_translator import _DART_MAP, _IFRS_MAP
        combined = {**_DART_MAP, **_IFRS_MAP}

        if concept in combined:
            return combined[concept]
        return None

    def _resolve_corp_code(self, identifier: str) -> str:
        """Resolve a stock code or name to a DART corp_code."""
        matches = self.list_companies(query=identifier)
        if matches:
            return matches[0].get("corp_code", "")

        # Try direct search
        if identifier.isdigit():
            all_companies = self.list_companies()
            for c in all_companies:
                if c.get("ticker") == identifier:
                    return c.get("corp_code", "")
        return ""

    def _cache_filings(self, identifier: str, df: pd.DataFrame) -> None:
        """Cache financial data as per-period JSON files."""
        if df.empty or "report_date" not in df.columns:
            return
        for period_end, group in df.groupby("report_date"):
            period_str = pd.Timestamp(period_end).strftime("%Y-%m-%d")
            records = group.to_dict(orient="records")
            for r in records:
                for k, v in r.items():
                    if isinstance(v, pd.Timestamp):
                        r[k] = v.isoformat()
            self._write_cache(identifier, f"filings/{period_str}.json",
                            {"period_end": period_str, "rows": records})

    # -- Price data ------------------------------------------------------------

    def get_quotes(self, identifier: str) -> pd.DataFrame:
        """DART does not provide OHLCV data. Returns empty DataFrame."""
        return pd.DataFrame()

    # -- Peers / executives --------------------------------------------------

    def get_peers(self, identifier: str) -> list[str]:
        """Return peer companies. Basic implementation using company list."""
        all_companies = self.list_companies()
        target_ticker = identifier
        peers = [c["ticker"] for c in all_companies if c.get("ticker") and c["ticker"] != target_ticker]
        return peers[:10]

    def get_executives(self, identifier: str) -> list[dict[str, Any]]:
        """Get executives from DART API if available."""
        if not self._api_key:
            return []

        corp_code = self._resolve_corp_code(identifier)
        if not corp_code:
            return []

        from operator1.http_utils import cached_get
        try:
            data = cached_get(
                f"{_DART_BASE}/exctvSttus.json",
                params={
                    "crtfc_key": self._api_key,
                    "corp_code": corp_code,
                    "bsns_year": str(date.today().year - 1),
                    "reprt_code": "11011",
                },
            )
            items = data.get("list", []) if isinstance(data, dict) else []
            return [
                {
                    "name": item.get("nm", ""),
                    "title": item.get("ofcps", ""),
                    "birth_year": item.get("birth_ym", ""),
                }
                for item in items
            ]
        except Exception:
            return []
