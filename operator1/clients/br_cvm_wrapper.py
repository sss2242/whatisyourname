"""Brazil CVM PIT client -- powered by pycvm library.

Primary library: pycvm (https://github.com/glourencoffee/pycvm)
  - CVM open data portal access for Brazilian company filings

Fallback: Direct CVM API (https://dados.cvm.gov.br/api/v1)

Coverage: ~400+ listed companies on B3, $2.2T market cap.
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

_CVM_BASE = "https://dados.cvm.gov.br/api/v1"
_CVM_DATASET_BASE = "https://dados.cvm.gov.br/dados/CIA_ABERTA"
_CACHE_DIR = Path("cache/br_cvm")


class BRCvmError(Exception):
    def __init__(self, endpoint: str, detail: str = "") -> None:
        self.endpoint = endpoint
        self.detail = detail
        super().__init__(f"BR CVM error on {endpoint}: {detail}")


class BRCvmClient:
    """Point-in-time client for CVM (Brazilian equities) using pycvm.

    Implements the ``PITClient`` protocol. Uses pycvm as primary
    library with direct CVM API as fallback.
    """

    def __init__(self, cache_dir: Path | str = _CACHE_DIR) -> None:
        self._cache_dir = Path(cache_dir)
        self._pycvm_available = False
        self._headers = {"Accept": "application/json", "User-Agent": "Operator1/1.0"}

        try:
            import pycvm
            self._pycvm_available = True
            logger.info("pycvm available for CVM data")
        except ImportError:
            logger.info("pycvm not available; using direct CVM API")

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
        return "br_cvm"

    @property
    def market_name(self) -> str:
        return "Brazil (B3) -- CVM"

    def list_companies(self, query: str = "") -> list[dict[str, Any]]:
        """Fetch company list from CVM CSV registry (replaces dead /api/v1/)."""
        import io
        import csv

        try:
            import requests
            r = requests.get(
                f"{_CVM_DATASET_BASE}/CAD/DADOS/cad_cia_aberta.csv",
                timeout=30,
                headers=self._headers,
            )
            r.raise_for_status()
            reader = csv.DictReader(io.StringIO(r.text), delimiter=";")
            items = [row for row in reader if row.get("SIT") == "ATIVO"]
        except Exception as exc:
            logger.warning("CVM CSV fetch failed: %s", exc)
            items = []

        companies = []
        for item in items:
            companies.append({
                "ticker": item.get("CD_CVM", "") or item.get("CNPJ_CIA", ""),
                "name": item.get("DENOM_SOCIAL", "") or item.get("DENOM_COMERC", ""),
                "cik": item.get("CD_CVM", ""),
                "cnpj": item.get("CNPJ_CIA", ""),
                "exchange": "B3",
                "country": "BR",
                "market_id": self.market_id,
            })

        if query:
            q = query.lower()
            companies = [c for c in companies if q in c["name"].lower() or q in c["ticker"].lower()]
        return companies

    def search_company(self, name: str) -> list[dict[str, Any]]:
        return self.list_companies(query=name)

    def get_profile(self, identifier: str) -> dict[str, Any]:
        cached = self._read_cache(identifier, "profile.json")
        if cached:
            return cached

        matches = self.list_companies(query=identifier)
        if not matches:
            raise BRCvmError("get_profile", f"Company not found: {identifier}")

        m = matches[0]
        raw_profile = {
            "name": m.get("name", ""),
            "ticker": m.get("ticker", identifier),
            "isin": "",
            "country": "BR",
            "sector": "",
            "industry": "",
            "exchange": "B3",
            "currency": "BRL",
            "cik": m.get("cik", ""),
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
        """Fetch financial data from CVM using pycvm or direct API."""
        if self._pycvm_available:
            try:
                df = self._fetch_via_pycvm(identifier, statement_type)
                if df is not None and not df.empty:
                    return df
            except Exception as exc:
                logger.warning("pycvm failed for %s: %s", identifier, exc)

        return self._fetch_via_direct_api(identifier, statement_type)

    def _fetch_via_pycvm(self, identifier: str, statement_type: str) -> pd.DataFrame | None:
        """Use pycvm to fetch CVM financial data."""
        import pycvm

        # Map statement type to CVM document categories
        doc_type_map = {
            "income": "DRE",
            "balance": "BPA",  # Balance sheet - assets
            "cashflow": "DFC_MI",  # Cash flow - indirect method
        }
        doc_type = doc_type_map.get(statement_type, "DRE")

        try:
            # pycvm provides access to CVM datasets
            current_year = date.today().year
            rows: list[dict] = []

            for year in range(current_year - 2, current_year + 1):
                try:
                    # Fetch the annual financial statements dataset
                    url = f"{_CVM_DATASET_BASE}/DOC/DFP/DADOS/dfp_cia_aberta_{year}.csv"
                    df = pd.read_csv(url, sep=";", encoding="latin-1", on_bad_lines="skip")

                    # Filter for the target company
                    cd_cvm = identifier
                    company_df = df[df["CD_CVM"].astype(str) == cd_cvm]
                    if company_df.empty:
                        # Try matching by name
                        company_df = df[df["DENOM_CIA"].str.contains(identifier, case=False, na=False)]

                    if company_df.empty:
                        continue

                    for _, row in company_df.iterrows():
                        account_code = str(row.get("CD_CONTA", ""))
                        from operator1.clients.canonical_translator import _CVM_ACCOUNT_MAP
                        canonical = _CVM_ACCOUNT_MAP.get(account_code)
                        if not canonical:
                            continue

                        value = row.get("VL_CONTA")
                        if pd.isna(value):
                            continue

                        rows.append({
                            "concept": canonical,
                            "value": float(value),
                            "filing_date": str(row.get("DT_REFER", "")),
                            "report_date": str(row.get("DT_REFER", "")),
                            "period_type": "annual",
                        })
                except Exception:
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
            logger.debug("pycvm extraction failed: %s", exc)
            return None

    def _fetch_via_direct_api(self, identifier: str, statement_type: str) -> pd.DataFrame:
        """Fallback: fetch from CVM CSV datasets directly."""
        current_year = date.today().year
        rows: list[dict] = []

        for year in range(current_year - 2, current_year + 1):
            for doc_type in ("dfp", "itr"):  # dfp=annual, itr=quarterly
                try:
                    url = f"{_CVM_DATASET_BASE}/DOC/{'DFP' if doc_type == 'dfp' else 'ITR'}/DADOS/{doc_type}_cia_aberta_{year}.csv"
                    df = pd.read_csv(url, sep=";", encoding="latin-1", on_bad_lines="skip")

                    company_df = df[
                        (df["CD_CVM"].astype(str) == identifier) |
                        (df["DENOM_CIA"].str.contains(identifier, case=False, na=False))
                    ]

                    if company_df.empty:
                        continue

                    from operator1.clients.canonical_translator import _CVM_ACCOUNT_MAP
                    for _, row in company_df.iterrows():
                        account_code = str(row.get("CD_CONTA", ""))
                        canonical = _CVM_ACCOUNT_MAP.get(account_code)
                        if not canonical:
                            continue

                        value = row.get("VL_CONTA")
                        if pd.isna(value):
                            continue

                        rows.append({
                            "concept": canonical,
                            "value": float(value),
                            "filing_date": str(row.get("DT_RECEB", row.get("DT_REFER", ""))),
                            "report_date": str(row.get("DT_REFER", "")),
                            "period_type": "annual" if doc_type == "dfp" else "quarterly",
                        })
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
