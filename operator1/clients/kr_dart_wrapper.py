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
import requests

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

    **Cross-statement caching**: The DART ``fnlttSinglAcntAll.json``
    endpoint returns ALL financial line items (income + balance + cashflow)
    in a single response.  On the first call for any statement type, we
    fetch all periods at once (8 API calls for 2 years x 4 periods) and
    cache the combined result.  Subsequent calls for the same company
    but different statement types are served from cache, reducing total
    API calls from 36 (3 types x 12 periods) to 8.

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
        # Cross-statement cache: identifier -> {all raw rows from fnlttSinglAcntAll}
        # Populated on first financial fetch, reused for all 3 statement types.
        self._financial_cache: dict[str, pd.DataFrame] = {}

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
        df = self._fetch_via_direct_api_financials(identifier, statement_type)

        # If still sparse (< 10 canonical fields), try LLM filing extraction
        # to parse the actual DART PDF filing for additional fields like
        # current_assets, current_liabilities, interest_expense.
        _KEY_FIELDS = {"current_assets", "current_liabilities", "interest_expense",
                       "cash_and_equivalents", "short_term_debt", "long_term_debt"}
        if not df.empty:
            _has = set(df.get("canonical_name", pd.Series()).unique()) if "canonical_name" in df.columns else set(df.columns)
            _missing_key = _KEY_FIELDS - _has
            if _missing_key:
                logger.info("DART %s sparse data: missing %s. Trying LLM filing extraction...", identifier, _missing_key)
                try:
                    llm_df = self._try_llm_filing_extraction(identifier, statement_type)
                    if llm_df is not None and not llm_df.empty:
                        df = pd.concat([df, llm_df], ignore_index=True).drop_duplicates(
                            subset=["canonical_name", "report_date"] if "canonical_name" in df.columns else None,
                            keep="first",
                        )
                        logger.info("LLM extraction added %d rows for %s", len(llm_df), identifier)
                except Exception as exc:
                    logger.debug("LLM filing extraction failed for %s: %s", identifier, exc)

        return df

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

    def _fetch_all_financials_bulk(self, identifier: str) -> pd.DataFrame:
        """Fetch ALL financial line items in one batch (cross-statement cache).

        Makes 8 API calls (2 years x 4 periods) and caches the combined
        result.  The ``fnlttSinglAcntAll.json`` endpoint returns income,
        balance sheet, and cash flow items together -- there's no need to
        call it separately per statement type.

        Returns the full raw DataFrame with ALL statement types combined.
        Subsequent calls for the same identifier return from cache.
        """
        # Return from cache if already fetched
        if identifier in self._financial_cache:
            cached = self._financial_cache[identifier]
            logger.debug("DART cross-statement cache hit for %s (%d rows)", identifier, len(cached))
            return cached

        from operator1.http_utils import cached_get

        corp_code = self._resolve_corp_code(identifier)
        if not corp_code:
            self._financial_cache[identifier] = pd.DataFrame()
            return pd.DataFrame()

        rows: list[dict] = []
        current_year = date.today().year
        api_calls = 0

        # Fetch 2 years x 4 periods = 8 API calls (was 12 per statement type)
        for year in range(current_year - 1, current_year + 1):
            for reprt_code in ["11011", "11014", "11012", "11013"]:
                # 11011=annual, 11014=Q3, 11012=semi, 11013=Q1
                try:
                    _dart_throttle()
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
                    api_calls += 1

                    items = data.get("list", []) if isinstance(data, dict) else []
                    for item in items:
                        account_name = item.get("account_nm", "")
                        sj_div = item.get("sj_div", "")  # BS, IS, CF, SCE

                        # Map to ALL statement types at once (not filtered)
                        for st in ("income", "balance", "cashflow"):
                            canonical = self._map_dart_concept(account_name, st)
                            if canonical:
                                break
                        else:
                            continue

                        value_str = item.get("thstrm_amount", "")
                        if not value_str or value_str == "-":
                            continue

                        try:
                            value = float(value_str.replace(",", ""))
                        except (ValueError, TypeError):
                            continue

                        is_annual = reprt_code == "11011"
                        filing_dt = item.get("rcept_dt", "") or ""
                        rows.append({
                            "concept": canonical,
                            "value": value,
                            "filing_date": filing_dt,
                            "report_date": f"{year}-12-31" if is_annual else f"{year}-{['03','06','09','12'][int(reprt_code[-1])-1]}-30",
                            "period_type": "annual" if is_annual else "quarterly",
                            "form": "Annual" if is_annual else "Quarterly",
                            "sj_div": sj_div,
                        })
                except Exception:
                    continue

        if not rows:
            self._financial_cache[identifier] = pd.DataFrame()
            return pd.DataFrame()

        df = pd.DataFrame(rows)
        for col in ("filing_date", "report_date"):
            if col in df.columns:
                df[col] = pd.to_datetime(df[col], errors="coerce")

        # Dedup: keep latest amendment per (concept, report_date)
        _before = len(df)
        if "concept" in df.columns and "report_date" in df.columns:
            df = (
                df.sort_values("filing_date", ascending=False, na_position="last")
                .drop_duplicates(subset=["concept", "report_date"], keep="first")
                .sort_values("report_date")
                .reset_index(drop=True)
            )
            _after = len(df)
            if _before != _after:
                logger.info("DART bulk dedup: %d -> %d rows (removed %d duplicates)",
                            _before, _after, _before - _after)

        self._financial_cache[identifier] = df
        logger.info(
            "DART bulk fetch for %s: %d rows from %d API calls (cached for all statement types)",
            identifier, len(df), api_calls,
        )

        # Cache filings to disk
        self._cache_filings(identifier, df)
        return df

    def _fetch_via_direct_api_financials(
        self, identifier: str, statement_type: str,
    ) -> pd.DataFrame:
        """Fetch financials via DART API with cross-statement caching.

        On first call, fetches ALL financial data in bulk (8 API calls
        for 2 years x 4 periods).  On subsequent calls for the same
        company, serves from the in-memory cache -- zero additional
        API calls.  This reduces total API usage from 36 calls
        (3 statement types x 12 periods) to 8 calls.
        """
        # Get or populate the bulk cache
        all_df = self._fetch_all_financials_bulk(identifier)
        if all_df.empty:
            return pd.DataFrame()

        from operator1.clients.canonical_translator import translate_financials
        return translate_financials(all_df, self.market_id, statement_type)

    def _map_dart_concept(self, concept: str, statement_type: str) -> str | None:
        """Map DART Korean account name to canonical name.

        Tries exact match first, then normalized match (strip whitespace
        and parenthetical content) to handle Korean label variations.
        """
        import re
        from operator1.clients.canonical_translator import _DART_MAP, _IFRS_MAP
        combined = {**_DART_MAP, **_IFRS_MAP}

        # Exact match first
        if concept in combined:
            return combined[concept]

        # Normalized match: strip whitespace and parenthetical content
        # Handles variations like "유동자산" vs "유 동 자 산" or "자본총계(지배)"
        normalized = re.sub(r'\s+', '', concept)
        normalized = re.sub(r'\(.*?\)', '', normalized)
        for key, canonical in combined.items():
            key_norm = re.sub(r'\s+', '', key)
            key_norm = re.sub(r'\(.*?\)', '', key_norm)
            if normalized == key_norm:
                return canonical

        return None

    def _resolve_corp_code(self, identifier: str) -> str:
        """Resolve a stock code or name to a DART corp_code.

        Resolution order:
        1. Search the cached corp list (from dart-fss or previous bulk download)
        2. Download and parse ``corpCode.xml`` bulk ZIP (same approach as dart-fss)
        3. Return empty string if all paths fail
        """
        # Try cached corp list first
        matches = self.list_companies(query=identifier)
        if matches:
            return matches[0].get("corp_code", "")

        # Try direct search by stock_code
        if identifier.isdigit():
            all_companies = self.list_companies()
            for c in all_companies:
                if c.get("ticker") == identifier:
                    return c.get("corp_code", "")

        # Fallback: download corpCode.xml bulk ZIP from DART API
        # This is the same approach dart-fss uses internally.
        # The ZIP contains ~100K companies with stock_code -> corp_code mapping.
        if self._api_key:
            corp_code = self._resolve_via_corp_code_xml(identifier)
            if corp_code:
                return corp_code

        return ""

    def _resolve_via_corp_code_xml(self, identifier: str) -> str:
        """Resolve stock_code to corp_code using DART filing search.

        Uses the HKEX date-windowed pattern: scan recent annual filings
        via ``list.json`` with narrow date windows.  Each filing result
        includes both ``stock_code`` and ``corp_code``, allowing us to
        build a mapping without downloading the slow 5MB corpCode.xml ZIP.

        Falls back to corpCode.xml bulk download if the filing search
        doesn't find the company (e.g. company filed outside the window).

        Resolution order:
        1. Disk cache (valid for 7 days)
        2. DART list.json filing search (fast, ~1s per page)
        3. DART corpCode.xml bulk ZIP download (slow, ~5MB)
        """
        import json as _json
        from datetime import timedelta

        cache_path = self._cache_dir / "_corpcode_map.json"

        # Try disk cache first (valid for 7 days)
        if cache_path.exists():
            try:
                age_days = (date.today() - date.fromtimestamp(cache_path.stat().st_mtime)).days
                if age_days < 7:
                    mapping = _json.loads(cache_path.read_text(encoding="utf-8"))
                    result = mapping.get(identifier, "")
                    if result:
                        logger.debug("corpCode cache hit: %s -> %s", identifier, result)
                        return result
                    # Search by name substring
                    id_lower = identifier.lower()
                    for key, code in mapping.items():
                        if id_lower in key.lower():
                            return code
            except Exception:
                pass

        # --- Fast path: DART list.json filing search (HKEX pattern) ---
        # Scan recent annual filings to build stock_code -> corp_code mapping.
        # Each page returns 100 filings with both stock_code and corp_code.
        # Most listed companies file annually, so scanning a few months
        # of filings covers the majority of active companies.
        mapping: dict[str, str] = {}

        try:
            today = date.today()
            for months_back in [1, 3, 6, 12]:
                start = today - timedelta(days=30 * months_back)
                end = today if months_back == 1 else today - timedelta(days=30 * (months_back - 1) - 1)

                for page in range(1, 11):  # Max 10 pages per window
                    _dart_throttle()
                    resp = requests.get(
                        f"{_DART_BASE}/list.json",
                        params={
                            "crtfc_key": self._api_key,
                            "bgn_de": start.strftime("%Y%m%d"),
                            "end_de": end.strftime("%Y%m%d"),
                            "pblntf_ty": "A",  # Annual reports
                            "page_count": "100",
                            "page_no": str(page),
                        },
                        timeout=15,
                    )
                    data = resp.json()
                    items = data.get("list", [])
                    if not items:
                        break

                    for item in items:
                        sc = item.get("stock_code", "").strip()
                        cc = item.get("corp_code", "").strip()
                        cn = item.get("corp_name", "").strip()
                        if sc and cc:
                            mapping[sc] = cc
                        if cn and cc:
                            mapping[cn] = cc

                    total_page = int(data.get("total_page", "1"))
                    if page >= total_page:
                        break

                # Check if we found the target
                if identifier in mapping:
                    logger.info(
                        "DART filing search resolved %s -> %s (%d mappings built)",
                        identifier, mapping[identifier], len(mapping),
                    )
                    break

        except Exception as exc:
            logger.debug("DART filing search for corp_code failed: %s", exc)

        # Populate corp_list_cache from the mappings we built
        if mapping:
            listed = []
            for key, code in mapping.items():
                if key.isdigit() and len(key) == 6:
                    corp_name = ""
                    for k2, c2 in mapping.items():
                        if c2 == code and not k2.isdigit():
                            corp_name = k2
                            break
                    listed.append({
                        "ticker": key,
                        "name": corp_name,
                        "corp_code": code,
                        "cik": code,
                        "exchange": "KRX",
                        "market_id": self.market_id,
                    })
            if listed:
                self._corp_list_cache = listed
                logger.info("DART filing search: %d listed companies mapped", len(listed))

            # Save to disk cache
            try:
                cache_path.parent.mkdir(parents=True, exist_ok=True)
                cache_path.write_text(
                    _json.dumps(mapping, ensure_ascii=False), encoding="utf-8",
                )
            except Exception:
                pass

        result = mapping.get(identifier, "")
        if result:
            return result

        # Substring search
        id_lower = identifier.lower()
        for key, code in mapping.items():
            if id_lower in key.lower():
                return code

        # --- Slow fallback: corpCode.xml bulk ZIP ---
        # Only used if filing search didn't find the company.
        logger.info("DART filing search didn't find %s; trying corpCode.xml bulk download...", identifier)
        return self._resolve_via_corp_code_xml_zip(identifier)

    def _resolve_via_corp_code_xml_zip(self, identifier: str) -> str:
        """Last-resort: download corpCode.xml bulk ZIP (~5MB) from DART."""
        import zipfile
        import io
        import json as _json
        import xml.etree.ElementTree as ET

        try:
            resp = requests.get(
                f"{_DART_BASE}/corpCode.xml",
                params={"crtfc_key": self._api_key},
                timeout=120,
            )
            resp.raise_for_status()
            if resp.content[:2] != b"PK":
                return ""
        except Exception as exc:
            logger.warning("corpCode.xml download failed: %s", exc)
            return ""

        mapping: dict[str, str] = {}
        try:
            with zipfile.ZipFile(io.BytesIO(resp.content)) as zf:
                xml_files = [n for n in zf.namelist() if n.endswith(".xml")]
                if not xml_files:
                    return ""
                with zf.open(xml_files[0]) as xf:
                    tree = ET.parse(xf)
                    root = tree.getroot()
                    for corp in root.findall(".//list"):
                        cc = corp.findtext("corp_code", "").strip()
                        sc = corp.findtext("stock_code", "").strip()
                        cn = corp.findtext("corp_name", "").strip()
                        if cc:
                            if sc:
                                mapping[sc] = cc
                            if cn:
                                mapping[cn] = cc
        except Exception as exc:
            logger.warning("corpCode.xml parse failed: %s", exc)
            return ""

        # Cache
        try:
            cache_path = self._cache_dir / "_corpcode_map.json"
            cache_path.parent.mkdir(parents=True, exist_ok=True)
            cache_path.write_text(_json.dumps(mapping, ensure_ascii=False), encoding="utf-8")
        except Exception:
            pass

        return mapping.get(identifier, "")

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

    # -- LLM filing extraction (for sparse DART data) -------------------------

    def _try_llm_filing_extraction(self, identifier: str, statement_type: str) -> pd.DataFrame | None:
        """Use LLM to extract missing financial fields from DART filing PDFs.

        Korean K-IFRS taxonomy keywords the LLM should search for:
        - 유동자산 (current_assets), 유동부채 (current_liabilities)
        - 이자비용/금융비용 (interest_expense)
        - 현금및현금성자산 (cash_and_equivalents)
        - 단기차입금 (short_term_debt), 장기차입금 (long_term_debt)
        - 매출채권 (receivables), 재고자산 (inventory)
        - 매입채무 (payables), 이익잉여금 (retained_earnings)

        Uses a single LLM call per filing to minimize API costs.
        """
        try:
            from operator1.clients.filing_discoverer import try_filing_extraction
        except ImportError:
            return None

        try:
            result = try_filing_extraction(
                ticker=identifier,
                market_id=self.market_id,
                statement_type=statement_type,
                llm_client=None,  # Will use factory default
            )
            return result if result is not None and not result.empty else None
        except Exception as exc:
            logger.debug("LLM filing extraction for DART %s failed: %s", identifier, exc)
            return None

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
