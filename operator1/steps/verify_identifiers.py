"""Step 0 / 0.1 -- Verify target identifiers and extract metadata.

Validates the user-provided company identifier (ticker, CIK, etc.)
via the selected PIT data source, then extracts the target company's
country and profile fields needed by every downstream module.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Any

from operator1.clients.pit_base import PITClientError
from operator1.clients.equity_provider import EquityProvider
from operator1.types import VerifiedTarget  # canonical definition in types.py

logger = logging.getLogger(__name__)


class VerificationError(Exception):
    """Raised when identifier verification fails.

    The pipeline should halt immediately when this is raised.
    """

    def __init__(self, source: str, detail: str) -> None:
        self.source = source
        self.detail = detail
        super().__init__(f"[{source}] Verification failed: {detail}")


def verify_identifiers(
    target_isin: str,
    fmp_symbol: str,
    pit_client: EquityProvider,
    fmp_client: Any = None,
) -> VerifiedTarget:
    """Verify identifiers and return a ``VerifiedTarget``.

    Parameters
    ----------
    target_isin:
        ISIN or company identifier.  May be empty if only a ticker
        is available.
    fmp_symbol:
        Ticker symbol for the target company.
    pit_client:
        Initialised PIT data client (satisfies EquityProvider protocol).
    fmp_client:
        Legacy parameter, kept for backward compatibility.  Ignored in
        the new PIT architecture.

    Returns
    -------
    VerifiedTarget
        Populated dataclass with all verified metadata.

    Raises
    ------
    VerificationError
        If the identifier is invalid or the API returns an error.
    """
    # ------------------------------------------------------------------
    # 1. Verify via PIT data provider (profile lookup)
    # ------------------------------------------------------------------
    lookup_key = target_isin if target_isin else fmp_symbol

    logger.info("Verifying identifier %s via PIT data provider ...", lookup_key)
    try:
        profile = pit_client.get_profile(lookup_key)
    except (PITClientError, Exception) as exc:
        raise VerificationError(
            "PITClient",
            f"Invalid identifier '{lookup_key}' or API error: {exc}",
        ) from exc

    country = profile.get("country")
    if not country:
        raise VerificationError(
            "PITClient",
            f"Profile for '{lookup_key}' has no country field. "
            "Cannot proceed without country information.",
        )

    logger.info(
        "Profile OK -- %s (%s), country=%s, sector=%s",
        profile.get("name"), profile.get("ticker"), country, profile.get("sector"),
    )

    # ------------------------------------------------------------------
    # 2. Build VerifiedTarget
    # ------------------------------------------------------------------
    resolved_isin = profile.get("isin") or target_isin or fmp_symbol

    return VerifiedTarget(
        isin=resolved_isin,
        ticker=profile.get("ticker", fmp_symbol),
        name=profile.get("name", ""),
        country=country,
        sector=profile.get("sector", ""),
        industry=profile.get("industry", ""),
        sub_industry=profile.get("sub_industry"),
        fmp_symbol=fmp_symbol,
        currency=profile.get("currency", ""),
        exchange=profile.get("exchange", ""),
        raw_profile=profile,
    )
