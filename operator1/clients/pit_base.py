"""Base protocol and ABC for Point-in-Time (PIT) data clients.

Every regional PIT client (SEC EDGAR, ESEF, EDINET, DART, etc.) must
implement this interface so the pipeline can treat them interchangeably.

The key difference from the old ``EquityProvider`` protocol: PIT clients
return data with **filing dates** (the date the data became publicly
available), not just report-period dates.  This is what makes the data
truly point-in-time and prevents look-ahead bias.
"""

from __future__ import annotations

import logging
from typing import Any, Protocol, runtime_checkable

import pandas as pd

logger = logging.getLogger(__name__)


@runtime_checkable
class PITClient(Protocol):
    """Structural interface that every PIT data backend must satisfy.

    All financial statement DataFrames must include a ``filing_date``
    column (the date the filing was publicly available) in addition to
    the ``report_date`` (fiscal period end date).  The cache builder
    uses ``filing_date`` for as-of joins to guarantee no look-ahead.
    """

    # -- Market metadata -----------------------------------------------------

    @property
    def market_id(self) -> str:
        """Unique market identifier matching ``pit_registry.MARKETS`` keys."""
        ...

    @property
    def market_name(self) -> str:
        """Human-readable market/exchange name for display."""
        ...

    # -- Company discovery ---------------------------------------------------

    def list_companies(self, query: str = "") -> list[dict[str, Any]]:
        """Return available companies, optionally filtered by *query*.

        Each dict should contain at minimum:
            - ``ticker``: exchange ticker symbol
            - ``name``: company name
            - ``cik`` (or equivalent): filing entity identifier

        Optional fields: ``isin``, ``sector``, ``industry``, ``exchange``.
        """
        ...

    def search_company(self, name: str) -> list[dict[str, Any]]:
        """Search for a company by name.  Returns same format as list_companies."""
        ...

    # -- Company profile -----------------------------------------------------

    def get_profile(self, identifier: str) -> dict[str, Any]:
        """Fetch company profile / metadata.

        Parameters
        ----------
        identifier:
            Ticker, CIK, or other exchange-specific ID.

        Returns
        -------
        dict with keys: name, ticker, isin, country, sector, industry,
        exchange, currency, market_cap (if available), etc.
        """
        ...

    # -- Financial statements (PIT) ------------------------------------------

    def get_income_statement(self, identifier: str) -> pd.DataFrame:
        """Fetch income statements with filing_date and report_date columns."""
        ...

    def get_balance_sheet(self, identifier: str) -> pd.DataFrame:
        """Fetch balance sheets with filing_date and report_date columns."""
        ...

    def get_cashflow_statement(self, identifier: str) -> pd.DataFrame:
        """Fetch cash flow statements with filing_date and report_date columns."""
        ...

    # -- Price data -----------------------------------------------------------

    def get_quotes(self, identifier: str) -> pd.DataFrame:
        """Fetch OHLCV price data.

        Returns a DataFrame with columns: date, open, high, low, close,
        volume.  ``adjusted_close`` is optional.
        """
        ...

    # -- Peers / related entities --------------------------------------------

    def get_peers(self, identifier: str) -> list[str]:
        """Return a list of peer company identifiers."""
        ...

    def get_executives(self, identifier: str) -> list[dict[str, Any]]:
        """Return key executives / officers."""
        ...

    # -- Institutional / major holders ----------------------------------------

    def get_holders(self, identifier: str) -> list[dict[str, Any]]:
        """Return institutional or major shareholders.

        Each dict should contain at minimum:
            - ``name``: holder name (institution or individual)
            - ``shares``: number of shares held (0 if unknown)
            - ``percentage``: percentage of outstanding shares (0.0 if unknown)

        Optional fields: ``value``, ``holder_type``, ``date_reported``.

        Returns empty list if holder data is not available for this market.
        """
        ...

    def get_holder_history(self, identifier: str, years: int = 2) -> "pd.DataFrame":
        """Return quarterly institutional ownership metrics over time.

        Used for injecting ownership trends into the daily cache so
        temporal models can learn from institutional behavior patterns.

        Must return a DataFrame with columns:
            - ``date_reported``: datetime -- the filing/report date
            - ``inst_ownership_pct``: float -- total institutional ownership %
            - ``inst_top5_concentration``: float -- HHI of top 5 holders
            - ``inst_holder_count``: int -- number of institutional holders

        Returns empty DataFrame if historical data is not available.
        New markets: implement this method to plug into the ownership
        pipeline automatically.
        """
        ...

    def get_insider_transactions(self, identifier: str) -> list[dict[str, Any]]:
        """Return insider buy/sell transactions for the company.

        Each dict should contain at minimum:
            - ``insider_name``: str -- name of the insider
            - ``position``: str -- role (CEO, CFO, Director, etc.)
            - ``date``: str -- transaction date (ISO format)
            - ``transaction``: str -- type (Purchase, Sale, etc.)
            - ``shares``: int -- number of shares transacted
            - ``value``: float -- dollar value of transaction

        Returns empty list if insider data is not available.
        Used by the institutional flow predictor for smart money signals.
        """
        ...


class PITClientError(Exception):
    """Base exception for PIT client errors."""

    def __init__(self, market_id: str, endpoint: str, detail: str = "") -> None:
        self.market_id = market_id
        self.endpoint = endpoint
        self.detail = detail
        super().__init__(f"[{market_id}] API error on {endpoint}: {detail}")
