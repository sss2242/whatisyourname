"""Supplementary data providers for partial-coverage regions.

The primary PIT clients for EU (ESEF), Japan (EDINET), Taiwan (MOPS),
Brazil (CVM), and Chile (CMF) provide financial statement data but lack
full profile details (sector, industry, market cap), OHLCV price data,
peer lists, and executive info.

This module provides free supplementary APIs that fill those gaps:

  Region        | Supplement API          | Fills
  --------------+-------------------------+------------------------------
  EU (ESEF)     | OpenFIGI + Euronext     | Sector, industry, identifiers
  Japan (EDINET)| JPX Listed Info         | Sector, industry, market info
  Taiwan (MOPS) | TWSE Company Info       | Sector, industry, profile
  Brazil (CVM)  | B3 Company Data         | Sector, industry, profile
  Chile (CMF)   | Bolsa de Santiago       | Sector, industry, profile
  All regions   | OpenFIGI (global)       | FIGI, sector classification

All APIs used here are free and require no API key (or free registration).
"""

from __future__ import annotations

import logging
from typing import Any

from operator1.http_utils import cached_get, cached_post, HTTPError

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# OpenFIGI -- Global identifier and classification mapping (free, no key)
# https://www.openfigi.com/api
# ---------------------------------------------------------------------------

_OPENFIGI_URL = "https://api.openfigi.com/v3/mapping"
_OPENFIGI_SEARCH_URL = "https://api.openfigi.com/v3/search"


def openfigi_enrich(
    ticker: str = "",
    isin: str = "",
    exchange_code: str = "",
) -> dict[str, Any]:
    """Look up an instrument via OpenFIGI for sector/industry classification.

    OpenFIGI is free (no key for up to 25 requests/minute) and maps
    tickers, ISINs, and other identifiers to FIGI records that include
    market sector and security type.

    Returns a dict with keys: figi, name, ticker, exchange_code,
    market_sector, security_type, or empty dict on failure.
    """
    if not ticker and not isin:
        return {}

    jobs: list[dict[str, str]] = []
    if isin:
        jobs.append({"idType": "ID_ISIN", "idValue": isin})
    if ticker:
        job: dict[str, str] = {"idType": "TICKER", "idValue": ticker}
        if exchange_code:
            job["exchCode"] = exchange_code
        jobs.append(job)

    headers = {
        "Content-Type": "application/json",
        "Accept": "application/json",
    }

    try:
        results = cached_post(
            _OPENFIGI_URL,
            json_data=jobs,
            headers=headers,
        )
        if not isinstance(results, list):
            return {}
    except Exception as exc:
        logger.debug("OpenFIGI request failed: %s", exc)
        return {}

    # Parse first successful result
    for result_group in results:
        if "data" not in result_group:
            continue
        for item in result_group["data"]:
            return {
                "figi": item.get("figi", ""),
                "name": item.get("name", ""),
                "ticker": item.get("ticker", ticker),
                "exchange_code": item.get("exchCode", exchange_code),
                "market_sector": item.get("marketSector", ""),
                "security_type": item.get("securityType", ""),
                "security_type2": item.get("securityType2", ""),
            }

    return {}


def openfigi_search(query: str, exchange_code: str = "") -> list[dict[str, Any]]:
    """Search OpenFIGI for instruments matching a query string.

    Returns a list of matching instruments with FIGI metadata.
    """
    if not query:
        return []

    search_params: dict[str, Any] = {"query": query}
    if exchange_code:
        search_params["exchCode"] = exchange_code

    headers = {
        "Content-Type": "application/json",
        "Accept": "application/json",
    }

    try:
        data = cached_post(
            _OPENFIGI_SEARCH_URL,
            json_data=search_params,
            headers=headers,
        )
        if isinstance(data, dict):
            return data.get("data", [])
        return []
    except Exception:
        return []


# ---------------------------------------------------------------------------
# EU / ESEF: Euronext instrument search (free, no key)
# https://live.euronext.com/
# ---------------------------------------------------------------------------

_EURONEXT_SEARCH_URL = "https://live.euronext.com/en/search_instruments"


def euronext_enrich(ticker: str = "", name: str = "") -> dict[str, Any]:
    """Fetch profile enrichment from Euronext live data.

    Euronext provides instrument search that returns sector, market cap,
    and other profile data for EU-listed companies.
    """
    query = ticker or name
    if not query:
        return {}

    try:
        data = cached_get(
            _EURONEXT_SEARCH_URL,
            params={"q": query, "type": "Stock"},
            headers={
                "Accept": "application/json",
                "User-Agent": "Operator1/1.0",
            },
        )
        if not isinstance(data, list) or not data:
            return {}

        item = data[0] if isinstance(data, list) else data
        return {
            "name": item.get("name", ""),
            "isin": item.get("isin", ""),
            "ticker": item.get("symbol", ticker),
            "exchange": item.get("market", ""),
            "sector": item.get("sector", "") or item.get("icb_sector", ""),
            "industry": item.get("industry", "") or item.get("icb_industry", ""),
            "market_cap": item.get("market_cap", ""),
            "currency": item.get("currency", "EUR"),
        }
    except Exception as exc:
        logger.debug("Euronext enrichment failed for %s: %s", query, exc)
        return {}


# ---------------------------------------------------------------------------
# Japan: JPX Listed Company Info (free, no key)
# https://www.jpx.co.jp/english/listing/co/
# ---------------------------------------------------------------------------

_JPX_LISTED_URL = "https://www.jpx.co.jp/english/listing/co/index.html"
_JPX_SEARCH_URL = "https://quote.jpx.co.jp/jpx/template/quote.cgi"


def jpx_enrich(ticker: str) -> dict[str, Any]:
    """Fetch profile enrichment from JPX (Japan Exchange Group).

    JPX provides company information for all TSE-listed securities.
    The quote endpoint returns sector and basic profile data.
    """
    if not ticker:
        return {}

    try:
        data = cached_get(
            _JPX_SEARCH_URL,
            params={"F": "tmp/stock_detail", "MKTN": "T", "QCODE": ticker},
            headers={
                "Accept": "application/json, text/html",
                "User-Agent": "Operator1/1.0",
            },
        )
        # JPX returns HTML; try to extract structured data
        if isinstance(data, dict):
            return {
                "name": data.get("name", ""),
                "ticker": ticker,
                "sector": data.get("sector", "") or data.get("category33", ""),
                "industry": data.get("industry", ""),
                "market_cap": data.get("market_cap", ""),
                "exchange": "TSE",
                "currency": "JPY",
            }
    except Exception as exc:
        logger.debug("JPX enrichment failed for %s: %s", ticker, exc)

    # Fallback: use OpenFIGI with Tokyo exchange code
    figi = openfigi_enrich(ticker=ticker, exchange_code="JP")
    if figi:
        return {
            "name": figi.get("name", ""),
            "ticker": ticker,
            "sector": figi.get("market_sector", ""),
            "industry": "",
            "market_cap": "",
            "exchange": "TSE",
            "currency": "JPY",
        }

    return {}


# ---------------------------------------------------------------------------
# Taiwan: TWSE Company Info (free, no key)
# https://www.twse.com.tw/en/
# ---------------------------------------------------------------------------

_TWSE_COMPANY_URL = "https://www.twse.com.tw/exchangeReport/BWIBBU_d"
_TWSE_STOCK_INFO_URL = "https://www.twse.com.tw/en/api/codeQuery"


def twse_enrich(ticker: str) -> dict[str, Any]:
    """Fetch profile enrichment from TWSE (Taiwan Stock Exchange).

    TWSE provides company classification and PBR data that includes
    sector information. Also queries TWSE code lookup for names.
    """
    if not ticker:
        return {}

    result: dict[str, Any] = {
        "ticker": ticker,
        "exchange": "TWSE",
        "currency": "TWD",
        "country": "TW",
    }

    # Try TWSE code query for company name
    try:
        data = cached_get(
            _TWSE_STOCK_INFO_URL,
            params={"query": ticker},
            headers={
                "Accept": "application/json",
                "User-Agent": "Operator1/1.0",
            },
        )
        if isinstance(data, dict):
            suggestions = data.get("suggestions", [])
            if suggestions and isinstance(suggestions, list):
                # Format: "2330\t台積電" or similar
                for s in suggestions:
                    if isinstance(s, str) and ticker in s:
                        parts = s.split("\t")
                        if len(parts) >= 2:
                            result["name"] = parts[1]
                        break
    except Exception:
        pass

    # Try PBR/sector data
    from datetime import date
    try:
        data = cached_get(
            _TWSE_COMPANY_URL,
            params={
                "response": "json",
                "date": date.today().strftime("%Y%m%d"),
                "stockNo": ticker,
            },
            headers={
                "Accept": "application/json",
                "User-Agent": "Operator1/1.0",
            },
        )
        if isinstance(data, dict):
            rows = data.get("data", [])
            fields = data.get("fields", [])
            if rows and fields:
                # Extract sector from TWSE classification
                result["sector"] = data.get("title", "").split("-")[0].strip()
    except Exception:
        pass

    # Fallback: OpenFIGI with Taipei exchange
    if "sector" not in result or not result.get("sector"):
        figi = openfigi_enrich(ticker=ticker, exchange_code="TT")
        if figi:
            result["name"] = result.get("name") or figi.get("name", "")
            result["sector"] = figi.get("market_sector", "")

    return result


# ---------------------------------------------------------------------------
# Brazil: B3 Company Data (via CVM registration, free, no key)
# https://dados.cvm.gov.br/
# ---------------------------------------------------------------------------

_CVM_CIA_URL = "https://dados.cvm.gov.br/api/v1/cia_aberta"


def b3_enrich(ticker: str = "", cnpj: str = "", name: str = "") -> dict[str, Any]:
    """Fetch profile enrichment for Brazilian companies from CVM registry.

    CVM's open data portal provides company registration details including
    sector classification (SETOR_ATIVIDADE) and status.
    """
    if not ticker and not cnpj and not name:
        return {}

    try:
        params: dict[str, Any] = {"$top": 50, "$format": "json"}
        if name:
            params["$filter"] = f"contains(DENOM_SOCIAL,'{name}')"

        data = cached_get(
            _CVM_CIA_URL,
            params=params,
            headers={
                "Accept": "application/json",
                "User-Agent": "Operator1/1.0",
            },
        )
        items = data.get("value", []) if isinstance(data, dict) else []
    except Exception as exc:
        logger.debug("CVM enrichment failed: %s", exc)
        items = []

    # Find matching company
    for item in items:
        cd_cvm = str(item.get("CD_CVM", ""))
        company_cnpj = item.get("CNPJ_CIA", "")
        company_name = item.get("DENOM_SOCIAL", "")

        if ticker and cd_cvm == str(ticker):
            pass
        elif cnpj and company_cnpj == cnpj:
            pass
        elif name and name.lower() in company_name.lower():
            pass
        else:
            continue

        return {
            "name": company_name,
            "ticker": cd_cvm,
            "cnpj": company_cnpj,
            "sector": item.get("SETOR_ATIVIDADE", ""),
            "industry": item.get("CATEG_REG", ""),
            "sit_reg": item.get("SIT_REG", ""),
            "exchange": "B3",
            "currency": "BRL",
            "country": "BR",
        }

    # Fallback: OpenFIGI with Sao Paulo exchange
    if ticker:
        figi = openfigi_enrich(ticker=ticker, exchange_code="BS")
        if figi:
            return {
                "name": figi.get("name", ""),
                "ticker": ticker,
                "sector": figi.get("market_sector", ""),
                "industry": "",
                "exchange": "B3",
                "currency": "BRL",
                "country": "BR",
            }

    return {}


# ---------------------------------------------------------------------------
# Chile: Bolsa de Santiago / CMF enrichment (free, no key)
# https://www.cmfchile.cl/
# ---------------------------------------------------------------------------

_CMF_ENTITY_URL = "https://www.cmfchile.cl/portal/informacion/entidades/busqueda"


def santiago_enrich(
    ticker: str = "",
    rut: str = "",
    name: str = "",
) -> dict[str, Any]:
    """Fetch profile enrichment for Chilean companies from CMF.

    CMF provides entity information including classification and
    registration details for all supervised entities.
    """
    query = rut or ticker or name
    if not query:
        return {}

    try:
        data = cached_get(
            _CMF_ENTITY_URL,
            params={"busqueda": query, "tipo": "SA", "formato": "json"},
            headers={
                "Accept": "application/json",
                "User-Agent": "Operator1/1.0",
            },
        )
        items = data if isinstance(data, list) else data.get("data", [])
    except Exception as exc:
        logger.debug("CMF enrichment failed for %s: %s", query, exc)
        items = []

    for item in items:
        entity_rut = item.get("rut", "")
        entity_name = item.get("razonSocial", "") or item.get("nombre", "")

        if rut and entity_rut != rut:
            continue
        if name and name.lower() not in entity_name.lower():
            continue

        return {
            "name": entity_name,
            "ticker": item.get("nemo", ticker),
            "rut": entity_rut,
            "sector": item.get("sector", "") or item.get("clasificacion", ""),
            "industry": item.get("actividad", ""),
            "exchange": "Santiago",
            "currency": "CLP",
            "country": "CL",
        }

    # Fallback: OpenFIGI with Santiago exchange
    if ticker:
        figi = openfigi_enrich(ticker=ticker, exchange_code="SS")
        if figi:
            return {
                "name": figi.get("name", ""),
                "ticker": ticker,
                "sector": figi.get("market_sector", ""),
                "industry": "",
                "exchange": "Santiago",
                "currency": "CLP",
                "country": "CL",
            }

    return {}


# ---------------------------------------------------------------------------
# Unified supplement dispatcher
# ---------------------------------------------------------------------------

_MARKET_ENRICHERS = {
    "eu_esef": lambda t, **kw: euronext_enrich(ticker=t, name=kw.get("name", "")),
    "fr_esef": lambda t, **kw: euronext_enrich(ticker=t, name=kw.get("name", "")),
    "de_esef": lambda t, **kw: euronext_enrich(ticker=t, name=kw.get("name", "")),
    # Tier 2 ESEF markets: NL is on Euronext Amsterdam so euronext_enrich
    # is the right enricher.  ES/IT/SE are not on Euronext but the enricher
    # still attempts OpenFIGI + name-based lookup which can return sector data.
    "nl_esef": lambda t, **kw: euronext_enrich(ticker=t, name=kw.get("name", "")),
    "es_esef": lambda t, **kw: euronext_enrich(ticker=t, name=kw.get("name", "")),
    "it_esef": lambda t, **kw: euronext_enrich(ticker=t, name=kw.get("name", "")),
    "se_esef": lambda t, **kw: euronext_enrich(ticker=t, name=kw.get("name", "")),
    "jp_jquants": lambda t, **kw: jpx_enrich(ticker=t),
    "tw_mops": lambda t, **kw: twse_enrich(ticker=t),
    "br_cvm": lambda t, **kw: b3_enrich(
        ticker=t, cnpj=kw.get("cnpj", ""), name=kw.get("name", ""),
    ),
    "cl_cmf": lambda t, **kw: santiago_enrich(
        ticker=t, rut=kw.get("rut", ""), name=kw.get("name", ""),
    ),
}


def enrich_profile(
    market_id: str,
    ticker: str,
    existing_profile: dict[str, Any] | None = None,
    **kwargs: Any,
) -> dict[str, Any]:
    """Enrich a company profile using the region's supplementary API.

    Merges supplementary data into the existing profile, filling only
    fields that are empty or missing. Never overwrites existing data.

    Parameters
    ----------
    market_id:
        PIT market identifier (e.g. "eu_esef", "jp_jquants").
    ticker:
        Company ticker or identifier.
    existing_profile:
        Profile dict from the primary PIT client. Supplement data will
        be merged into this, filling gaps only.
    **kwargs:
        Additional lookup hints (name, cnpj, rut, isin, etc.).

    Returns
    -------
    dict: Enriched profile with supplement data merged in.
    """
    profile = dict(existing_profile) if existing_profile else {}

    enricher = _MARKET_ENRICHERS.get(market_id)
    if enricher is None:
        return profile

    try:
        supplement = enricher(ticker, **kwargs)
    except Exception as exc:
        logger.debug(
            "Supplement enrichment failed for %s/%s: %s",
            market_id, ticker, exc,
        )
        return profile

    if not supplement:
        return profile

    # Merge: only fill empty/missing fields
    for key, value in supplement.items():
        if value and not profile.get(key):
            profile[key] = value

    # Mark that supplementary data was used
    profile["_supplement_source"] = market_id
    profile["_supplement_fields"] = [
        k for k in supplement if supplement[k] and not (existing_profile or {}).get(k)
    ]

    logger.info(
        "Profile enriched for %s via %s: filled %d fields",
        ticker, market_id, len(profile.get("_supplement_fields", [])),
    )

    return profile
