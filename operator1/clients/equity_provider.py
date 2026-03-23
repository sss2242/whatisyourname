"""Equity data provider abstraction -- PIT API registry.

Global Point-in-Time (PIT) API registry.  Users select a region and
market, and the appropriate PIT client is instantiated.

Supported markets (all free, immutable PIT APIs):

  - US: SEC EDGAR ($50T)
  - UK: Companies House ($3.18T)
  - EU: ESEF/XBRL ($8-9T) -- covers France, Germany, etc.
  - Japan: EDINET ($6.5T)
  - South Korea: DART ($2.5T)
  - Taiwan: MOPS (~$1.2T)
  - Brazil: CVM ($2.2T)
  - Chile: CMF ($0.4T)

Total coverage: $91+ trillion in market cap.
"""

from __future__ import annotations

import logging
from typing import Any, Protocol, runtime_checkable

import pandas as pd

from operator1.clients.pit_base import PITClient, PITClientError
from operator1.clients.pit_registry import (
    MARKETS,
    MarketInfo,
    get_market,
    get_regions,
    get_markets_by_region,
)

logger = logging.getLogger(__name__)


# Keep the EquityProvider protocol for backward compatibility with
# downstream pipeline modules that type-check against it.
@runtime_checkable
class EquityProvider(Protocol):
    """Structural interface that any equity-data backend must satisfy.

    PIT clients implement these methods so the pipeline can treat them
    interchangeably.
    """

    def get_profile(self, identifier: str) -> dict[str, Any]: ...
    def get_quotes(self, identifier: str) -> pd.DataFrame: ...
    def get_income_statement(self, identifier: str) -> pd.DataFrame: ...
    def get_balance_sheet(self, identifier: str) -> pd.DataFrame: ...
    def get_cashflow_statement(self, identifier: str) -> pd.DataFrame: ...
    def get_peers(self, identifier: str) -> list[str]: ...
    def get_executives(self, identifier: str) -> list[dict[str, Any]]: ...


# Valid market IDs for the --market CLI flag
MARKET_CHOICES = tuple(MARKETS.keys())


class _JPJquantsAdapter:
    """Adapter that maps PITClient protocol to JPJquantsClient's API.

    JPJquantsClient exposes all financials via ``get_financials()`` as a
    dict of DataFrames.  This adapter exposes them as individual methods
    (``get_income_statement()``, ``get_balance_sheet()``, etc.) so that
    ``data_extraction.py`` can consume them through the standard PITClient
    interface without changing the J-Quants wrapper.
    """

    def __init__(self, client) -> None:
        self._client = client
        self._financials_cache: dict = {}

    @property
    def market_id(self) -> str:
        return "jp_jquants"

    @property
    def market_name(self) -> str:
        return "Japan (J-Quants / TSE)"

    def list_companies(self, query: str = "") -> list:
        return self._client.list_companies(query)

    def search_company(self, name: str) -> list:
        return self._client.list_companies(query=name)

    def get_profile(self, identifier: str) -> dict:
        return self._client.get_profile(identifier)

    def get_peers(self, identifier: str) -> list:
        return self._client.get_peers(identifier)

    def get_executives(self, identifier: str) -> list:
        return self._client.get_executives(identifier)

    def _get_financials(self, identifier: str) -> dict:
        if identifier not in self._financials_cache:
            self._financials_cache[identifier] = self._client.get_financials(identifier)
        return self._financials_cache[identifier]

    def get_income_statement(self, identifier: str) -> pd.DataFrame:
        return self._get_financials(identifier).get("income", pd.DataFrame())

    def get_balance_sheet(self, identifier: str) -> pd.DataFrame:
        return self._get_financials(identifier).get("balance", pd.DataFrame())

    def get_cashflow_statement(self, identifier: str) -> pd.DataFrame:
        return self._get_financials(identifier).get("cashflow", pd.DataFrame())

    def get_quotes(self, identifier: str) -> pd.DataFrame:
        # J-Quants doesn't provide OHLCV through the financial summary API.
        # The OHLCV provider (yfinance with .T suffix) handles this.
        return pd.DataFrame()

    def get_holders(self, identifier: str) -> list:
        if hasattr(self._client, "get_holders"):
            return self._client.get_holders(identifier)
        return []

    def get_holder_history(self, identifier: str, years: int = 2) -> pd.DataFrame:
        if hasattr(self._client, "get_holder_history"):
            return self._client.get_holder_history(identifier, years)
        return pd.DataFrame()

    def get_insider_transactions(self, identifier: str) -> list:
        if hasattr(self._client, "get_insider_transactions"):
            return self._client.get_insider_transactions(identifier)
        return []


def create_pit_client(
    market_id: str,
    secrets: dict[str, str] | None = None,
) -> PITClient:
    """Instantiate the PIT client for the given market.

    Parameters
    ----------
    market_id:
        Market identifier from the PIT registry (e.g. "us_sec_edgar").
    secrets:
        Dictionary of API keys.  Only needed for markets that require
        a key (UK Companies House, South Korea DART).

    Returns
    -------
    PITClient
        An initialised client satisfying the ``PITClient`` protocol.

    Raises
    ------
    SystemExit
        If the market is unknown or a required API key is missing.
    """
    if secrets is None:
        secrets = {}

    market = get_market(market_id)
    if market is None:
        available = ", ".join(MARKETS.keys())
        raise SystemExit(
            f"Unknown market '{market_id}'. Available markets: {available}"
        )

    # Import and instantiate the appropriate client
    if market_id == "us_sec_edgar":
        from operator1.clients.us_edgar import USEdgarClient
        edgar_identity = secrets.get("EDGAR_IDENTITY", "")
        if edgar_identity:
            user_agent = f"Operator1/1.0 ({edgar_identity})"
        else:
            user_agent = "Operator1/1.0 (https://github.com/Abdu2024/OP-1)"
        return USEdgarClient(user_agent=user_agent)

    if market_id == "uk_companies_house":
        api_key = secrets.get("COMPANIES_HOUSE_API_KEY", "")
        if not api_key:
            logger.warning(
                "COMPANIES_HOUSE_API_KEY not set. "
                "Companies House requests may be rate-limited."
            )
        from operator1.clients.uk_ch_wrapper import UKCompaniesHouseClient
        return UKCompaniesHouseClient(api_key=api_key)

    if market_id in ("eu_esef", "fr_esef", "de_esef"):
        country_code_map = {
            "eu_esef": "",
            "fr_esef": "FR",
            "de_esef": "DE",
        }
        from operator1.clients.eu_esef_wrapper import EUEsefClient
        return EUEsefClient(
            country_code=country_code_map[market_id],
            market_id=market_id,
        )

    if market_id == "jp_jquants":
        from operator1.clients.jp_jquants_wrapper import JPJquantsClient
        api_key = secrets.get("JQUANTS_API_KEY", "")
        if not api_key:
            logger.warning(
                "JQUANTS_API_KEY not set. J-Quants requests will fail. "
                "Register free at https://jpx-jquants.com/login"
            )
        raw_client = JPJquantsClient(api_key=api_key)
        return _JPJquantsAdapter(raw_client)

    if market_id == "kr_dart":
        api_key = secrets.get("DART_API_KEY", "")
        if not api_key:
            logger.warning(
                "DART_API_KEY not set. DART requests will fail. "
                "Register free at https://opendart.fss.or.kr/"
            )
        from operator1.clients.kr_dart_wrapper import KRDartClient
        return KRDartClient(api_key=api_key)

    if market_id == "tw_mops":
        from operator1.clients.tw_mops_wrapper import TWMopsClient
        return TWMopsClient()

    if market_id == "br_cvm":
        from operator1.clients.br_cvm_wrapper import BRCvmClient
        return BRCvmClient()

    if market_id == "cl_cmf":
        from operator1.clients.cl_cmf_wrapper import CLCmfClient
        return CLCmfClient()

    # --- Phase 2 markets ---

    if market_id == "ca_sedar":
        from operator1.clients.ca_sedar import CASedarClient
        return CASedarClient()

    if market_id == "au_asx":
        from operator1.clients.au_asx import AUAsxClient
        return AUAsxClient()

    if market_id == "in_bse":
        from operator1.clients.in_bse import INBseClient
        return INBseClient()

    if market_id == "cn_sse":
        from operator1.clients.cn_sse import CNSseClient
        return CNSseClient()

    if market_id == "hk_hkex":
        from operator1.clients.hk_hkex import HKHkexClient
        return HKHkexClient()

    if market_id == "sg_sgx":
        from operator1.clients.sg_sgx import SGSgxClient
        return SGSgxClient()

    if market_id == "mx_bmv":
        from operator1.clients.mx_bmv import MXBmvClient
        return MXBmvClient()

    if market_id == "za_jse":
        from operator1.clients.za_jse import ZAJseClient
        return ZAJseClient()

    if market_id == "ch_six":
        from operator1.clients.ch_six import CHSixClient
        return CHSixClient()

    # EU ESEF-based markets (NL, ES, IT, SE) -- reuse EUEsefClient
    if market_id in ("nl_esef", "es_esef", "it_esef", "se_esef"):
        country_code_map = {
            "nl_esef": "NL",
            "es_esef": "ES",
            "it_esef": "IT",
            "se_esef": "SE",
        }
        from operator1.clients.eu_esef_wrapper import EUEsefClient
        return EUEsefClient(
            country_code=country_code_map[market_id],
            market_id=market_id,
        )

    if market_id == "sa_tadawul":
        from operator1.clients.sa_tadawul import SATadawulClient
        return SATadawulClient()

    if market_id == "ae_dfm":
        from operator1.clients.ae_dfm import AEDfmClient
        return AEDfmClient()

    raise SystemExit(f"Market '{market_id}' is registered but has no client implementation.")


# Backward-compatible alias
def create_equity_provider(
    secrets: dict[str, str],
    provider: str = "auto",
    market_id: str = "us_sec_edgar",
) -> EquityProvider:
    """Create an equity provider (backward-compatible wrapper).

    For new code, prefer ``create_pit_client(market_id, secrets)`` directly.

    Parameters
    ----------
    secrets:
        API key dictionary.
    provider:
        Legacy provider name.  Ignored in new architecture; kept for
        backward compatibility with existing CLI flags.
    market_id:
        PIT market to use.  Defaults to US SEC EDGAR.
    """
    if provider != "auto":
        logger.info(
            "Legacy --provider flag '%s' ignored. "
            "Using PIT market '%s' instead.",
            provider,
            market_id,
        )
    return create_pit_client(market_id, secrets)
