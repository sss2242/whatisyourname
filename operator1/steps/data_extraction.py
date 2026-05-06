"""Step C / C.1 -- Bulk data extraction for target and linked entities.

Fetches profiles, quotes, financial statements, and OHLCV data for all
entities via the selected PIT data source.  Raw data is persisted to
``cache/raw/{identifier}/`` as Parquet files.
Every API call is logged via the http_utils request log.
"""

from __future__ import annotations

import json
import logging
import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import pandas as pd

from operator1.clients.pit_base import PITClientError
from operator1.clients.equity_provider import EquityProvider
from operator1.clients.canonical_translator import (
    translate_financials,
    translate_profile,
    pivot_to_canonical_wide,
)
from operator1.config_loader import get_global_config
from operator1.constants import CACHE_DIR, RAW_CACHE_DIR, DATE_START, DATE_END
from operator1.types import VerifiedTarget, EntityData, ExtractionResult

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Disk cache helpers
# ---------------------------------------------------------------------------

def _entity_cache_dir(isin: str) -> Path:
    """Return the raw cache directory for an entity."""
    return Path(RAW_CACHE_DIR) / isin.replace("/", "_")


def _cache_file(isin: str, name: str) -> Path:
    return _entity_cache_dir(isin) / f"{name}.parquet"


def _json_cache_file(isin: str, name: str) -> Path:
    return _entity_cache_dir(isin) / f"{name}.json"


def _is_cached(isin: str, name: str, is_json: bool = False) -> bool:
    path = _json_cache_file(isin, name) if is_json else _cache_file(isin, name)
    return path.exists()


def _write_df(isin: str, name: str, df: pd.DataFrame) -> None:
    path = _cache_file(isin, name)
    os.makedirs(path.parent, exist_ok=True)
    df.to_parquet(path, index=False)


def _read_df(isin: str, name: str) -> pd.DataFrame:
    return pd.read_parquet(_cache_file(isin, name))


def _write_json(isin: str, name: str, data: Any) -> None:
    path = _json_cache_file(isin, name)
    os.makedirs(path.parent, exist_ok=True)
    with open(path, "w", encoding="utf-8") as fh:
        json.dump(data, fh, indent=2)


def _read_json(isin: str, name: str) -> Any:
    with open(_json_cache_file(isin, name), "r", encoding="utf-8") as fh:
        return json.load(fh)


# ---------------------------------------------------------------------------
# Long-to-wide format bridge
# ---------------------------------------------------------------------------

def _ensure_wide_format(df: pd.DataFrame) -> pd.DataFrame:
    """Bridge between wrapper output and cache_builder expectations.

    Some wrappers call translate_financials() internally, which returns
    long-format data (rows with ``canonical_name`` and ``value`` columns).
    The cache_builder's ``_asof_merge_statement()`` expects wide-format
    data (one column per canonical field, e.g. ``revenue``, ``net_income``).

    This function detects long-format output and pivots it to wide format,
    preserving ``filing_date`` and ``report_date`` columns for PIT joins.
    Wide-format data passes through unchanged.
    """
    if df is None or df.empty:
        return pd.DataFrame()

    # Detect long format: has both 'canonical_name' and 'value' columns
    if "canonical_name" in df.columns and "value" in df.columns:
        return pivot_to_canonical_wide(df)

    # Already wide format (e.g. J-Quants, or direct column names) -- pass through
    return df


# ---------------------------------------------------------------------------
# Per-entity extraction
# ---------------------------------------------------------------------------

def _fetch_entity_data(
    isin: str,
    pit_client: EquityProvider,
    is_target: bool = False,
    force_rebuild: bool = False,
) -> EntityData:
    """Fetch all PIT data for a single entity.

    Parameters
    ----------
    isin:
        Entity identifier (ISIN, ticker, CIK, etc.).
    pit_client:
        Initialised PIT data client.
    is_target:
        If True, also fetch peers and executives.
    force_rebuild:
        If True, bypass disk caches.
    """
    entity = EntityData(isin=isin)

    # Profile
    if not force_rebuild and _is_cached(isin, "profile", is_json=True):
        entity.profile = _read_json(isin, "profile")
    else:
        try:
            entity.profile = pit_client.get_profile(isin)
            _write_json(isin, "profile", entity.profile)
        except (PITClientError, Exception) as exc:
            logger.warning("Profile fetch failed for %s: %s", isin, exc)

    # Quotes (OHLCV)
    if not force_rebuild and _is_cached(isin, "quotes"):
        entity.quotes = _read_df(isin, "quotes")
    else:
        try:
            entity.quotes = pit_client.get_quotes(isin)
            if not entity.quotes.empty:
                _write_df(isin, "quotes", entity.quotes)
        except (PITClientError, Exception) as exc:
            logger.warning("Quotes fetch failed for %s: %s", isin, exc)

    # Income statement
    if not force_rebuild and _is_cached(isin, "income_statement"):
        entity.income_statement = _read_df(isin, "income_statement")
    else:
        try:
            raw_income = pit_client.get_income_statement(isin)
            # Bridge: if the wrapper returned long-format data (concept/value
            # columns from translate_financials), pivot to the wide format
            # that cache_builder expects (one column per canonical field).
            entity.income_statement = _ensure_wide_format(raw_income)
            if not entity.income_statement.empty:
                _write_df(isin, "income_statement", entity.income_statement)
        except (PITClientError, Exception) as exc:
            logger.warning("Income statement fetch failed for %s: %s", isin, exc)

    # Balance sheet
    if not force_rebuild and _is_cached(isin, "balance_sheet"):
        entity.balance_sheet = _read_df(isin, "balance_sheet")
    else:
        try:
            raw_balance = pit_client.get_balance_sheet(isin)
            entity.balance_sheet = _ensure_wide_format(raw_balance)
            if not entity.balance_sheet.empty:
                _write_df(isin, "balance_sheet", entity.balance_sheet)
        except (PITClientError, Exception) as exc:
            logger.warning("Balance sheet fetch failed for %s: %s", isin, exc)

    # Cash-flow statement
    if not force_rebuild and _is_cached(isin, "cashflow_statement"):
        entity.cashflow_statement = _read_df(isin, "cashflow_statement")
    else:
        try:
            raw_cashflow = pit_client.get_cashflow_statement(isin)
            entity.cashflow_statement = _ensure_wide_format(raw_cashflow)
            if not entity.cashflow_statement.empty:
                _write_df(isin, "cashflow_statement", entity.cashflow_statement)
        except (PITClientError, Exception) as exc:
            logger.warning("Cash-flow statement fetch failed for %s: %s", isin, exc)

    # Target-only extras
    if is_target:
        # Peers
        if not force_rebuild and _is_cached(isin, "peers", is_json=True):
            entity.peers = _read_json(isin, "peers")
        else:
            try:
                entity.peers = pit_client.get_peers(isin)
                _write_json(isin, "peers", entity.peers)
            except (PITClientError, Exception) as exc:
                logger.warning("Peers fetch failed for %s: %s", isin, exc)

        # Executives
        if not force_rebuild and _is_cached(isin, "executives", is_json=True):
            entity.executives = _read_json(isin, "executives")
        else:
            try:
                entity.executives = pit_client.get_executives(isin)
                _write_json(isin, "executives", entity.executives)
            except (PITClientError, Exception) as exc:
                logger.warning("Executives fetch failed for %s: %s", isin, exc)

    return entity


# ---------------------------------------------------------------------------
# Main extraction orchestrator
# ---------------------------------------------------------------------------

def extract_all_data(
    target: VerifiedTarget,
    linked_isins: list[str],
    pit_client: EquityProvider,
    _legacy_fmp_client: Any = None,
    force_rebuild: bool | None = None,
) -> ExtractionResult:
    """Fetch all raw data for the target and linked entities.

    Parameters
    ----------
    target:
        Verified target from Step 0.1.
    linked_isins:
        List of identifiers for linked entities (from discovery).
    pit_client:
        Initialised PIT data client.
    fmp_client:
        Legacy parameter, kept for backward compatibility.  Ignored.
    force_rebuild:
        Override FORCE_REBUILD config.

    Returns
    -------
    ExtractionResult
        Contains raw data for target and all linked entities.
    """
    cfg = get_global_config()
    if force_rebuild is None:
        force_rebuild = cfg.get("FORCE_REBUILD", False)

    result = ExtractionResult()

    # ------------------------------------------------------------------
    # 1. Target entity -- PIT data
    # ------------------------------------------------------------------
    logger.info("Extracting target data for %s (%s) ...", target.name, target.isin)
    result.target = _fetch_entity_data(
        target.isin, pit_client, is_target=True, force_rebuild=force_rebuild,
    )

    # ------------------------------------------------------------------
    # 2. Target entity -- OHLCV from PIT source or supplement
    # ------------------------------------------------------------------
    # In the new architecture, OHLCV comes from the PIT client's
    # get_quotes() method.  If the PIT source doesn't provide price data
    # (e.g. SEC EDGAR, ESEF, EDINET, CVM, CMF), we fall back to the
    # OHLCV provider (yfinance, pykrx, baostock, twstock, etc.).
    result.target.ohlcv = result.target.quotes
    if result.target.ohlcv.empty:
        market_id = getattr(pit_client, "market_id", "")
        ticker = result.target.profile.get("ticker", target.isin)
        logger.info(
            "PIT source has no OHLCV for %s; trying per-region OHLCV provider ...",
            target.isin,
        )
        try:
            from operator1.clients.ohlcv_provider import fetch_ohlcv
            result.target.ohlcv = fetch_ohlcv(
                ticker=ticker,
                market_id=market_id,
                years=5,
            )
            if not result.target.ohlcv.empty:
                logger.info(
                    "OHLCV provider fetched %d rows for %s",
                    len(result.target.ohlcv), ticker,
                )
        except Exception as exc:
            logger.warning("OHLCV provider failed for %s: %s", ticker, exc)

    if result.target.ohlcv.empty:
        result.errors.append({
            "entity": target.isin,
            "module": "pit_ohlcv",
            "error": "PIT source does not provide OHLCV -- price-dependent modules may be limited",
        })

    # ------------------------------------------------------------------
    # 3. Linked entities -- PIT data (batched)
    # ------------------------------------------------------------------
    total = len(linked_isins)
    for idx, isin in enumerate(linked_isins, 1):
        logger.info(
            "Extracting linked entity %d/%d: %s ...", idx, total, isin,
        )
        try:
            entity_data = _fetch_entity_data(
                isin, pit_client, is_target=False, force_rebuild=force_rebuild,
            )
            result.linked[isin] = entity_data
        except Exception as exc:
            logger.error("Extraction failed for linked entity %s: %s", isin, exc)
            result.errors.append({
                "entity": isin,
                "module": "pit_linked",
                "error": str(exc),
            })

    # Summary
    logger.info(
        "Extraction complete: target=%s, linked=%d, errors=%d",
        target.isin, len(result.linked), len(result.errors),
    )

    return result


def save_extraction_metadata(
    result: ExtractionResult,
    target: VerifiedTarget,
    output_path: str = "cache/extraction_metadata.json",
) -> None:
    """Persist extraction summary metadata to disk."""
    meta = {
        "target_isin": target.isin,
        "target_fmp_symbol": target.fmp_symbol,
        "target_data": {
            "profile": bool(result.target.profile),
            "quotes_rows": len(result.target.quotes),
            "income_statement_rows": len(result.target.income_statement),
            "balance_sheet_rows": len(result.target.balance_sheet),
            "cashflow_statement_rows": len(result.target.cashflow_statement),
            "ohlcv_rows": len(result.target.ohlcv),
            "peers_count": len(result.target.peers),
            "supply_chain_count": len(result.target.supply_chain),
            "executives_count": len(result.target.executives),
        },
        "linked_entities": {
            isin: {
                "profile": bool(e.profile),
                "quotes_rows": len(e.quotes),
                "income_statement_rows": len(e.income_statement),
                "balance_sheet_rows": len(e.balance_sheet),
                "cashflow_statement_rows": len(e.cashflow_statement),
            }
            for isin, e in result.linked.items()
        },
        "errors": result.errors,
    }
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as fh:
        json.dump(meta, fh, indent=2)
    logger.info("Extraction metadata saved to %s", output_path)
