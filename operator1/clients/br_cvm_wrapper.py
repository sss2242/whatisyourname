"""Brazil CVM PIT client -- ZIP-based dataset extraction.

Primary data source: CVM Open Data Portal (dados.cvm.gov.br)
  - Annual financial statements: DFP ZIP archives (10 markets, ~320 filings/year)
  - Quarterly financial statements: ITR ZIP archives (~2,200 filings/year)
  - Company registry: CSV (2,600+ active companies with CNPJ, CD_CVM)
  - Each ZIP contains per-statement CSVs with structured account codes
  - Filing receipt dates (DT_RECEB) provide true PIT timestamps
  - No API key required, no rate limiting, no auth

Architecture inspired by HKEX scraper pattern:
  - Year-windowed downloads (one ZIP per year, like HKEX 2-week windows)
  - Structured data extraction from within archives (no LLM needed)
  - Module-level cache to avoid redundant downloads across statement types

Coverage: ~400+ listed companies on B3, $2.2T market cap.

ZIP structure per year:
  dfp_cia_aberta_YYYY.zip (annual):
    - dfp_cia_aberta_YYYY.csv          (filing index with DT_RECEB = PIT date)
    - dfp_cia_aberta_DRE_con_YYYY.csv  (income statement, consolidated)
    - dfp_cia_aberta_BPA_con_YYYY.csv  (balance sheet assets, consolidated)
    - dfp_cia_aberta_BPP_con_YYYY.csv  (balance sheet liabilities, consolidated)
    - dfp_cia_aberta_DFC_MI_con_YYYY.csv (cash flow indirect, consolidated)

  itr_cia_aberta_YYYY.zip (quarterly):
    - Same structure as DFP but with quarterly data
"""

from __future__ import annotations

import io
import json
import logging
import os
import time
import zipfile
from datetime import date
from pathlib import Path
from typing import Any

import pandas as pd

from operator1.http_utils import cached_get, HTTPError

logger = logging.getLogger(__name__)

_CVM_BASE = "https://dados.cvm.gov.br/api/v1"
_CVM_DATASET_BASE = "https://dados.cvm.gov.br/dados/CIA_ABERTA"
_CACHE_DIR = Path("cache/br_cvm")

# ---------------------------------------------------------------------------
# Statement type -> CSV filename pattern within the ZIP
# Uses consolidated (_con) statements; falls back to individual (_ind)
# ---------------------------------------------------------------------------
_STATEMENT_CSV_MAP: dict[str, list[str]] = {
    "income": ["DRE_con", "DRE_ind"],
    "balance": ["BPA_con", "BPP_con", "BPA_ind", "BPP_ind"],  # Assets + Liabilities
    "cashflow": ["DFC_MI_con", "DFC_MD_con", "DFC_MI_ind", "DFC_MD_ind"],
}

# Module-level ZIP cache: {url: ZipFile bytes}
# Avoids re-downloading the same ZIP for income/balance/cashflow calls
_zip_cache: dict[str, bytes] = {}
_zip_cache_time: dict[str, float] = {}
_ZIP_CACHE_TTL = 3600  # 1 hour


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
            legal_name = item.get("DENOM_SOCIAL", "") or ""
            commercial_name = item.get("DENOM_COMERC", "") or ""
            companies.append({
                "ticker": item.get("CD_CVM", "") or item.get("CNPJ_CIA", ""),
                "name": legal_name or commercial_name,
                "name_commercial": commercial_name,
                "cik": item.get("CD_CVM", ""),
                "cnpj": item.get("CNPJ_CIA", ""),
                "exchange": "B3",
                "country": "BR",
                "market_id": self.market_id,
            })

        logger.debug("CVM registry loaded: %d active companies", len(companies))

        if query:
            q = query.lower()
            # Match against legal name, commercial name, CVM code, and CNPJ
            companies = [
                c for c in companies
                if q in c["name"].lower()
                or q in c.get("name_commercial", "").lower()
                or q in c["ticker"].lower()
                or q in c.get("cnpj", "").lower()
            ]
            logger.debug("CVM search '%s': %d matches", query, len(companies))
        return companies

    def search_company(self, name: str) -> list[dict[str, Any]]:
        return self.list_companies(query=name)

    def get_profile(self, identifier: str) -> dict[str, Any]:
        cached = self._read_cache(identifier, "profile.json")
        if cached:
            return cached

        matches = self.list_companies(query=identifier)

        # B3 tickers (e.g. PETR4, VALE3, ITUB4) use a 4-letter base + share
        # class digit.  CVM registry doesn't store B3 tickers, so strip the
        # trailing digit(s) and retry as a name search.
        if not matches and identifier.strip() and identifier[-1].isdigit():
            import re
            base = re.sub(r"\d+$", "", identifier).strip()
            if len(base) >= 3:
                matches = self.list_companies(query=base)
                if matches:
                    logger.info(
                        "Resolved B3 ticker '%s' -> CVM company '%s' (CD_CVM=%s)",
                        identifier, matches[0].get("name"), matches[0].get("cik"),
                    )

        if not matches:
            raise BRCvmError("get_profile", f"Company not found: {identifier}")

        m = matches[0]
        cd_cvm = m.get("cik", "")

        raw_profile = {
            "name": m.get("name", ""),
            "ticker": m.get("ticker", identifier),
            "isin": "",
            "country": "BR",
            "sector": "",
            "industry": "",
            "exchange": "B3",
            "currency": "BRL",
            "cik": cd_cvm,
            "cnpj": m.get("cnpj", ""),
        }

        # Enrich with FCA (Formulário Cadastral) data -- sector, industry,
        # founding date, ownership type, website, fiscal year.
        # FCA ZIP contains fca_cia_aberta_geral_YYYY.csv with rich profile data.
        fca = self._fetch_fca_profile(cd_cvm)
        if fca:
            raw_profile.update({
                "sector": fca.get("Setor_Atividade", ""),
                "industry": fca.get("Descricao_Atividade", ""),
                "sub_industry": "",
                "website": fca.get("Pagina_Web", ""),
                "founded": fca.get("Data_Constituicao", ""),
                "ownership_type": fca.get("Especie_Controle_Acionario", ""),
                "cvm_category": fca.get("Categoria_Registro_CVM", ""),
                "fiscal_year_end_month": fca.get("Mes_Encerramento_Exercicio_Social", ""),
                "fiscal_year_end_day": fca.get("Dia_Encerramento_Exercicio_Social", ""),
                "status": fca.get("Situacao_Emissor", ""),
                "country_origin": fca.get("Pais_Origem", "Brasil"),
            })
            logger.info(
                "CVM profile enriched for %s: sector=%s, ownership=%s",
                cd_cvm, fca.get("Setor_Atividade", "?"), fca.get("Especie_Controle_Acionario", "?"),
            )

        from operator1.clients.canonical_translator import translate_profile
        profile = translate_profile(raw_profile, self.market_id)
        self._write_cache(identifier, "profile.json", profile)
        return profile

    def _fetch_fca_profile(self, cd_cvm: str) -> dict[str, Any] | None:
        """Fetch company profile from FCA (Formulário Cadastral) ZIP.

        The FCA contains sector, industry description, ownership type,
        website, founding date, and fiscal year -- richer than the basic
        registry CSV.
        """
        import requests as req

        current_year = date.today().year

        # Try current year first, then previous (FCA may not be filed yet)
        for year in (current_year, current_year - 1):
            url = f"{_CVM_DATASET_BASE}/DOC/FCA/DADOS/fca_cia_aberta_{year}.zip"
            try:
                z = self._download_zip(url)
                csv_name = f"fca_cia_aberta_geral_{year}.csv"
                if csv_name not in z.namelist():
                    continue

                with z.open(csv_name) as f:
                    df = pd.read_csv(f, sep=";", encoding="latin-1", on_bad_lines="skip")

                company = df[df["Codigo_CVM"].astype(str) == str(cd_cvm)]
                if company.empty:
                    continue

                # Return the most recent version
                row = company.sort_values("Versao", ascending=False).iloc[0]
                return {col: str(row[col]) if pd.notna(row[col]) else "" for col in df.columns}

            except Exception as exc:
                logger.debug("FCA fetch failed for %s/%d: %s", cd_cvm, year, exc)
                continue

        return None

    def get_income_statement(self, identifier: str) -> pd.DataFrame:
        return self._fetch_financials(identifier, "income")

    def get_balance_sheet(self, identifier: str) -> pd.DataFrame:
        return self._fetch_financials(identifier, "balance")

    def get_cashflow_statement(self, identifier: str) -> pd.DataFrame:
        return self._fetch_financials(identifier, "cashflow")

    def _fetch_financials(self, identifier: str, statement_type: str) -> pd.DataFrame:
        """Fetch financial data from CVM ZIP archives (HKEX-inspired pattern).

        Downloads year-based ZIP archives from dados.cvm.gov.br, extracts
        the per-statement CSVs, and parses structured account codes into
        canonical long-format DataFrames.

        The filing index CSV inside each ZIP provides DT_RECEB (receipt date)
        which is used as filing_date for true PIT compliance.
        """
        try:
            df = self._fetch_via_zip(identifier, statement_type)
            if df is not None and not df.empty:
                return df
        except Exception as exc:
            logger.warning("CVM ZIP extraction failed for %s/%s: %s", identifier, statement_type, exc)

        return pd.DataFrame()

    def _download_zip(self, url: str) -> zipfile.ZipFile:
        """Download and cache a CVM ZIP archive.

        Uses a module-level cache (like HKEX session reuse) so that
        multiple statement type calls for the same company share the
        same downloaded ZIP.
        """
        import requests

        now = time.time()
        if url in _zip_cache and (now - _zip_cache_time.get(url, 0)) < _ZIP_CACHE_TTL:
            return zipfile.ZipFile(io.BytesIO(_zip_cache[url]))

        r = requests.get(url, timeout=60, headers=self._headers)
        r.raise_for_status()

        _zip_cache[url] = r.content
        _zip_cache_time[url] = now

        return zipfile.ZipFile(io.BytesIO(r.content))

    def _get_filing_dates(self, z: zipfile.ZipFile, doc_prefix: str, year: int, cd_cvm: str) -> dict[str, str]:
        """Extract filing receipt dates (DT_RECEB) from the index CSV.

        Returns dict mapping DT_REFER -> DT_RECEB for the target company.
        This provides true PIT filing dates.
        """
        index_name = f"{doc_prefix}_cia_aberta_{year}.csv"
        if index_name not in z.namelist():
            return {}

        try:
            with z.open(index_name) as f:
                idx_df = pd.read_csv(f, sep=";", encoding="latin-1")
            company_rows = idx_df[idx_df["CD_CVM"].astype(str) == cd_cvm]
            result = {}
            for _, row in company_rows.iterrows():
                dt_refer = str(row.get("DT_REFER", ""))
                dt_receb = str(row.get("DT_RECEB", ""))
                if dt_refer and dt_receb:
                    result[dt_refer] = dt_receb
            return result
        except Exception:
            return {}

    def _extract_from_zip(
        self,
        z: zipfile.ZipFile,
        doc_prefix: str,
        year: int,
        cd_cvm: str,
        statement_type: str,
    ) -> list[dict]:
        """Extract financial rows for a company from a CVM ZIP archive.

        Applies the HKEX pattern: structured data extraction from within
        archives, mapping account codes to canonical field names.
        """
        from operator1.clients.canonical_translator import _CVM_ACCOUNT_MAP

        csv_patterns = _STATEMENT_CSV_MAP.get(statement_type, [])
        filing_dates = self._get_filing_dates(z, doc_prefix, year, cd_cvm)

        rows: list[dict] = []
        for pattern in csv_patterns:
            csv_name = f"{doc_prefix}_cia_aberta_{pattern}_{year}.csv"
            if csv_name not in z.namelist():
                continue

            try:
                with z.open(csv_name) as f:
                    df = pd.read_csv(f, sep=";", encoding="latin-1", on_bad_lines="skip")
            except Exception as exc:
                logger.debug("Failed to read %s: %s", csv_name, exc)
                continue

            # Filter for the target company
            company_df = df[df["CD_CVM"].astype(str) == cd_cvm]
            if company_df.empty:
                continue

            # Use ORDEM_EXERC == 'ULTIMO' for the most recent exercise period
            # (avoids double-counting from 'PENULTIMO' comparative periods)
            if "ORDEM_EXERC" in company_df.columns:
                ultimo = company_df[company_df["ORDEM_EXERC"] == "ULTIMO"]
                if not ultimo.empty:
                    company_df = ultimo

            for _, row in company_df.iterrows():
                account_code = str(row.get("CD_CONTA", ""))
                canonical = _CVM_ACCOUNT_MAP.get(account_code)
                if not canonical:
                    continue

                value = row.get("VL_CONTA")
                if pd.isna(value):
                    continue

                dt_refer = str(row.get("DT_REFER", ""))
                dt_fim = str(row.get("DT_FIM_EXERC", dt_refer))

                # Use DT_RECEB from filing index as true PIT filing date
                filing_date = filing_dates.get(dt_refer, dt_refer)

                rows.append({
                    "canonical_name": canonical,
                    "value": float(value),
                    "filing_date": filing_date,
                    "report_date": dt_fim or dt_refer,
                    "period_type": "annual" if doc_prefix == "dfp" else "quarterly",
                    "account_code": account_code,
                    "account_desc": str(row.get("DS_CONTA", "")),
                })

            logger.debug(
                "CVM %s/%s %s/%d: %d rows from %s",
                cd_cvm, statement_type, doc_prefix, year,
                len([r for r in rows if r.get("period_type") == ("annual" if doc_prefix == "dfp" else "quarterly")]),
                csv_name,
            )

        return rows

    def _fetch_via_zip(self, identifier: str, statement_type: str) -> pd.DataFrame | None:
        """Fetch financials from CVM ZIP archives (primary, fast, no LLM).

        Downloads year-based ZIP files and extracts structured CSV data.
        Covers both annual (DFP) and quarterly (ITR) filings for 2-3 years.
        """
        # Resolve CD_CVM from identifier
        cd_cvm = self._resolve_cd_cvm(identifier)
        if not cd_cvm:
            return None

        current_year = date.today().year
        all_rows: list[dict] = []

        for year in range(current_year - 2, current_year + 1):
            for doc_prefix in ("itr", "dfp"):  # quarterly first (more granular)
                doc_type = "ITR" if doc_prefix == "itr" else "DFP"
                url = f"{_CVM_DATASET_BASE}/DOC/{doc_type}/DADOS/{doc_prefix}_cia_aberta_{year}.zip"

                try:
                    z = self._download_zip(url)
                    rows = self._extract_from_zip(z, doc_prefix, year, cd_cvm, statement_type)
                    all_rows.extend(rows)
                except Exception as exc:
                    logger.debug("CVM ZIP %s/%d failed: %s", doc_prefix, year, exc)
                    continue

        if not all_rows:
            return None

        df = pd.DataFrame(all_rows)

        # Deduplicate: keep latest version per (canonical_name, report_date)
        df = df.sort_values("filing_date", ascending=False)
        df = df.drop_duplicates(subset=["canonical_name", "report_date"], keep="first")

        for col in ("filing_date", "report_date"):
            if col in df.columns:
                df[col] = pd.to_datetime(df[col], errors="coerce")

        logger.info(
            "CVM ZIP for %s/%s: %d records from %d unique periods",
            identifier, statement_type, len(df),
            df["report_date"].nunique() if "report_date" in df.columns else 0,
        )
        return df

    def _resolve_cd_cvm(self, identifier: str) -> str:
        """Resolve any identifier (ticker, name, CVM code) to a CD_CVM code."""
        # If identifier looks like a CVM code (all digits), use directly
        if identifier.strip().isdigit():
            return identifier.strip()

        # Otherwise search the registry
        matches = self.list_companies(query=identifier)
        if matches:
            cd_cvm = matches[0].get("cik", "")
            if cd_cvm:
                logger.info("Resolved '%s' -> CD_CVM=%s (%s)", identifier, cd_cvm, matches[0].get("name", ""))
                return cd_cvm

        return ""



    def get_quotes(self, identifier: str) -> pd.DataFrame:
        return pd.DataFrame()

    def get_peers(self, identifier: str) -> list[str]:
        return []

    def get_executives(self, identifier: str) -> list[dict[str, Any]]:
        return []
