"""T3.4 -- Linked entity aggregates (daily per relationship group).

For each day and each relationship group (competitors, suppliers,
customers, financial_institutions, sector_peers, industry_peers),
compute mean and median of key variables across group members.
Also compute relative strength and valuation premium measures
comparing the target to its peers.

Groups with zero members produce null columns with ``is_missing_*``
flags.  Output is persisted to ``cache/linked_aggregates_daily.parquet``.
"""

from __future__ import annotations

import logging
import os
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from operator1.constants import CACHE_DIR

logger = logging.getLogger(__name__)

# Key variables to aggregate across linked entities.
# These should exist in the entity daily caches after derived-variable
# computation.
AGGREGATE_VARIABLES: tuple[str, ...] = (
    "return_1d",
    "volatility_21d",
    "drawdown_252d",
    "current_ratio",
    "debt_to_equity_abs",
    "free_cash_flow",
    "fcf_yield",
    "gross_margin",
    "operating_margin",
    "net_margin",
    "roe",
    "pe_ratio_calc",
    "ev_to_ebitda",
    "enterprise_value",
    "market_cap",
)

# Relationship groups to process
RELATIONSHIP_GROUPS: tuple[str, ...] = (
    "competitors",
    "suppliers",
    "customers",
    "financial_institutions",
    "sector_peers",
    "industry_peers",
)


# ---------------------------------------------------------------------------
# Per-group aggregation
# ---------------------------------------------------------------------------


def _parse_relationship_year(value: str) -> pd.Timestamp | None:
    """Parse a relationship start/end year string to a Timestamp.

    Accepts: "2020", "2020-06", "current", "ongoing", "unknown".
    Returns None for non-parseable or open-ended values.
    """
    if not value or value in ("current", "ongoing", "unknown"):
        return None
    try:
        # Try full date first, then year-month, then year-only
        for fmt in ("%Y-%m-%d", "%Y-%m", "%Y"):
            try:
                return pd.Timestamp(value)
            except ValueError:
                continue
        return pd.Timestamp(f"{int(value)}-01-01")
    except (ValueError, TypeError):
        return None


def _apply_temporal_mask(
    series: pd.Series,
    start: pd.Timestamp | None,
    end: pd.Timestamp | None,
) -> pd.Series:
    """Mask a series to NaN outside the [start, end] relationship window."""
    if start is None and end is None:
        return series
    masked = series.copy()
    if start is not None:
        masked.loc[masked.index < start] = np.nan
    if end is not None:
        masked.loc[masked.index > end] = np.nan
    return masked


def _aggregate_group(
    group_name: str,
    entity_frames: list[pd.DataFrame],
    daily_index: pd.DatetimeIndex,
    temporal_ranges: list[tuple[pd.Timestamp | None, pd.Timestamp | None]] | None = None,
) -> pd.DataFrame:
    """Compute mean and median aggregates for a single relationship group.

    Parameters
    ----------
    group_name:
        E.g. ``"competitors"``.
    entity_frames:
        List of daily DataFrames for entities in this group.
    daily_index:
        The business-day index to align to.
    temporal_ranges:
        Optional list of ``(start, end)`` Timestamp pairs, one per
        entity_frame.  When provided, each entity's data is masked
        to NaN outside its relationship active window.  This prevents
        aggregating data from before a supplier relationship began or
        after a customer relationship ended.

    Returns
    -------
    pd.DataFrame
        Indexed by ``daily_index`` with columns like
        ``competitors_avg_return_1d``, ``competitors_median_volatility_21d``.
    """
    result = pd.DataFrame(index=daily_index)

    if not entity_frames:
        # No members -- produce null columns with missing flags
        for var in AGGREGATE_VARIABLES:
            avg_col = f"{group_name}_avg_{var}"
            med_col = f"{group_name}_median_{var}"
            result[avg_col] = np.nan
            result[med_col] = np.nan
            result[f"is_missing_{avg_col}"] = 1
            result[f"is_missing_{med_col}"] = 1
        logger.debug("Group '%s' has 0 members -- null columns", group_name)
        return result

    # Stack all entity frames for this group
    aligned: list[pd.DataFrame] = []
    masks: list[tuple[pd.Timestamp | None, pd.Timestamp | None]] = []
    for i, df in enumerate(entity_frames):
        # Reindex to the common daily index
        reindexed = df.reindex(daily_index)
        aligned.append(reindexed)
        if temporal_ranges and i < len(temporal_ranges):
            masks.append(temporal_ranges[i])
        else:
            masks.append((None, None))

    for var in AGGREGATE_VARIABLES:
        # Collect the variable from each entity, applying temporal mask
        series_list: list[pd.Series] = []
        for idx, df in enumerate(aligned):
            if var in df.columns:
                s = df[var]
                start, end = masks[idx]
                if start is not None or end is not None:
                    s = _apply_temporal_mask(s, start, end)
                series_list.append(s)

        if not series_list:
            avg_col = f"{group_name}_avg_{var}"
            med_col = f"{group_name}_median_{var}"
            result[avg_col] = np.nan
            result[med_col] = np.nan
            result[f"is_missing_{avg_col}"] = 1
            result[f"is_missing_{med_col}"] = 1
            continue

        combined = pd.concat(series_list, axis=1)

        avg_col = f"{group_name}_avg_{var}"
        med_col = f"{group_name}_median_{var}"

        result[avg_col] = combined.mean(axis=1)
        result[med_col] = combined.median(axis=1)
        result[f"is_missing_{avg_col}"] = result[avg_col].isna().astype(int)
        result[f"is_missing_{med_col}"] = result[med_col].isna().astype(int)

    n_masked = sum(1 for s, e in masks if s is not None or e is not None)
    logger.info(
        "Group '%s': aggregated %d variables across %d entities (%d temporally masked)",
        group_name, len(AGGREGATE_VARIABLES), len(entity_frames), n_masked,
    )

    return result


# ---------------------------------------------------------------------------
# Relative measures
# ---------------------------------------------------------------------------


def _compute_relative_measures(
    target_daily: pd.DataFrame,
    aggregates: pd.DataFrame,
) -> pd.DataFrame:
    """Compute relative strength and valuation premium vs peer groups.

    Variables:
    - ``rel_strength_vs_sector``: target return_1d / sector_peers_avg_return_1d
    - ``valuation_premium_vs_industry``: target pe_ratio_calc / industry_peers_avg_pe_ratio_calc

    These use safe division (null when denominator is zero/null).
    """
    result = pd.DataFrame(index=target_daily.index)

    # Relative strength vs sector peers
    sector_avg_ret = aggregates.get(
        "sector_peers_avg_return_1d",
        pd.Series(np.nan, index=target_daily.index),
    )
    target_ret = target_daily.get(
        "return_1d",
        pd.Series(np.nan, index=target_daily.index),
    )
    with np.errstate(divide="ignore", invalid="ignore"):
        rel = target_ret / sector_avg_ret
        rel = rel.replace([np.inf, -np.inf], np.nan)
    result["rel_strength_vs_sector"] = rel
    result["is_missing_rel_strength_vs_sector"] = rel.isna().astype(int)

    # Valuation premium vs industry peers
    industry_avg_pe = aggregates.get(
        "industry_peers_avg_pe_ratio_calc",
        pd.Series(np.nan, index=target_daily.index),
    )
    target_pe = target_daily.get(
        "pe_ratio_calc",
        pd.Series(np.nan, index=target_daily.index),
    )
    with np.errstate(divide="ignore", invalid="ignore"):
        premium = target_pe / industry_avg_pe
        premium = premium.replace([np.inf, -np.inf], np.nan)
    result["valuation_premium_vs_industry"] = premium
    result["is_missing_valuation_premium_vs_industry"] = premium.isna().astype(int)

    return result


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def compute_linked_aggregates(
    target_daily: pd.DataFrame,
    linked_daily: dict[str, pd.DataFrame],
    entity_groups: dict[str, list[str]],
    entity_temporal: dict[str, dict[str, tuple[str, str]]] | None = None,
) -> pd.DataFrame:
    """Compute all linked entity aggregates and relative measures.

    Parameters
    ----------
    target_daily:
        Target company daily cache (with derived variables).
    linked_daily:
        Dict of ``{isin: daily_df}`` for each linked entity.
    entity_groups:
        Dict of ``{relationship_group: [isin, ...]}`` mapping each
        group to its member ISINs.  This comes from
        ``DiscoveryResult.linked`` in entity_discovery.
    entity_temporal:
        Optional dict of ``{relationship_group: {isin: (start, end)}}``
        with relationship start/end year strings.  When provided,
        each entity's data is temporally masked so that aggregates
        only include data from within the active relationship window.
        This comes from ``LinkedEntity.relationship_start`` and
        ``LinkedEntity.relationship_end``.

    Returns
    -------
    pd.DataFrame
        Daily-indexed DataFrame with all aggregate columns and
        relative measures.
    """
    daily_index = target_daily.index

    # Build per-group frames
    all_groups: list[pd.DataFrame] = []

    for group_name in RELATIONSHIP_GROUPS:
        member_isins = entity_groups.get(group_name, [])
        member_frames = [
            linked_daily[isin]
            for isin in member_isins
            if isin in linked_daily
        ]

        # Build temporal ranges for this group
        temporal_ranges = None
        if entity_temporal and group_name in entity_temporal:
            group_temporal = entity_temporal[group_name]
            temporal_ranges = []
            for isin in member_isins:
                if isin in linked_daily and isin in group_temporal:
                    start_str, end_str = group_temporal[isin]
                    temporal_ranges.append((
                        _parse_relationship_year(start_str),
                        _parse_relationship_year(end_str),
                    ))
                elif isin in linked_daily:
                    temporal_ranges.append((None, None))

        group_agg = _aggregate_group(
            group_name, member_frames, daily_index,
            temporal_ranges=temporal_ranges,
        )
        all_groups.append(group_agg)

    # Combine all group aggregates
    aggregates = pd.concat(all_groups, axis=1)

    # Compute relative measures
    rel_measures = _compute_relative_measures(target_daily, aggregates)
    aggregates = pd.concat([aggregates, rel_measures], axis=1)

    logger.info(
        "Linked aggregates computed: %d columns across %d groups",
        len(aggregates.columns), len(RELATIONSHIP_GROUPS),
    )

    return aggregates


def persist_linked_aggregates(
    aggregates: pd.DataFrame,
    output_path: str | None = None,
) -> str:
    """Write linked aggregates to disk.

    Parameters
    ----------
    aggregates:
        Output of ``compute_linked_aggregates()``.
    output_path:
        Override output path.

    Returns
    -------
    str
        Path to the written Parquet file.
    """
    if output_path is None:
        output_path = str(Path(CACHE_DIR) / "linked_aggregates_daily.parquet")

    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    aggregates.to_parquet(output_path)
    logger.info(
        "Linked aggregates saved: %s (%d rows, %d cols)",
        output_path, len(aggregates), len(aggregates.columns),
    )
    return output_path


def compute_relative_metrics(
    target_cache: pd.DataFrame,
    linked_agg_df: pd.DataFrame,
) -> pd.DataFrame:
    """Compute company-relative metrics vs sector/industry aggregates.

    Returns a DataFrame with relative strength, valuation premium, and
    other company-vs-peer metrics from The_Apps_core_idea.pdf Section 5.1.
    """
    result = pd.DataFrame(index=target_cache.index)

    # rel_strength_vs_sector = company return - sector avg return
    _return_col = None
    for col in ("return_1d", "return_21d"):
        if col in target_cache.columns:
            _return_col = col
            break

    if _return_col is not None and linked_agg_df is not None:
        for prefix in ("competitors_avg_", "sector_peers_avg_", "industry_peers_avg_"):
            peer_col = f"{prefix}{_return_col}"
            if peer_col in linked_agg_df.columns:
                result["rel_strength_vs_sector"] = (
                    target_cache[_return_col] - linked_agg_df[peer_col]
                )
                break

    # valuation_premium_vs_industry = company PE - industry median PE
    if "pe_ratio_calc" in target_cache.columns and linked_agg_df is not None:
        for prefix in ("competitors_avg_", "industry_peers_avg_", "sector_peers_avg_"):
            peer_col = f"{prefix}pe_ratio_calc"
            if peer_col in linked_agg_df.columns:
                result["valuation_premium_vs_industry"] = (
                    target_cache["pe_ratio_calc"] - linked_agg_df[peer_col]
                )
                break

    # rel_volatility = company vol / sector avg vol
    if "volatility_21d" in target_cache.columns and linked_agg_df is not None:
        for prefix in ("competitors_avg_", "sector_peers_avg_"):
            peer_col = f"{prefix}volatility_21d"
            if peer_col in linked_agg_df.columns:
                peer_vol = linked_agg_df[peer_col].replace(0, float("nan"))
                result["rel_volatility_vs_sector"] = (
                    target_cache["volatility_21d"] / peer_vol
                )
                break

    n_cols = len([c for c in result.columns if result[c].notna().any()])
    if n_cols > 0:
        logger.info("Relative metrics computed: %d metrics with data", n_cols)
    return result
