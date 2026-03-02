"""Chile CMF PIT client -- wrapper with disk caching.

Uses the CMF Chile API and website scraping for Chilean company
financial data, with enhanced disk caching.

Primary: CMF API (https://www.cmfchile.cl) + FECU data
Fallback: Website scraping for company profiles

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
        """Fetch FECU financial data from CMF.

        CMF provides FECU (standardized financial statements) in IFRS
        format for all listed Chilean companies.
        """
        rows: list[dict] = []
        current_year = date.today().year

        for year in range(current_year - 2, current_year + 1):
            for period in ["Q4", "Q2"]:  # Annual and semi-annual
                try:
                    # Try Open Data API first, then old path
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

    def get_quotes(self, identifier: str) -> pd.DataFrame:
        return pd.DataFrame()

    def get_peers(self, identifier: str) -> list[str]:
        return []

    def get_executives(self, identifier: str) -> list[dict[str, Any]]:
        return []
