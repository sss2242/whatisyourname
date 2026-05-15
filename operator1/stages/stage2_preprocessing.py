"""Stage 2: Data Preprocessing -- frequency separation, reconciliation, interpolation.

Sub-stages:
  2.1  Frequency separation (Bayesian classification + Chow-Lin/Denton disaggregation)
  2.2  Data reconciliation and quality checks

These sub-stages run AFTER Stage 1 (data fetch) populates the raw statement
DataFrames in PipelineState, and BEFORE Stage 3 (temporal models) which
requires the daily cache to be built from reconciled statements.
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from operator1.pipeline_state import PipelineState

logger = logging.getLogger("operator1.stages.stage2")


def run_2_1_frequency_separation(state: PipelineState) -> None:
    """2.1: Separate mixed-frequency statements and reconcile via Chow-Lin/Denton.

    Reads raw income_df, balance_df, cashflow_df from state.
    Writes back reconciled single-frequency DataFrames.
    """
    logger.info("Sub-stage 2.1: Frequency separation")

    # Ensure raw statement DFs are loaded from parquet (they may be
    # missing from pickle state if loaded from a checkpoint -- the
    # staged runner serializes DFs separately as parquet files).
    import pandas as pd
    from pathlib import Path
    _run_dir = Path(state.output_dir)
    for _df_name in ("income_df", "balance_df", "cashflow_df"):
        _df = getattr(state, _df_name, None)
        if _df is None or (isinstance(_df, pd.DataFrame) and _df.empty):
            _pq = _run_dir / f"{_df_name}.parquet"
            if _pq.exists():
                setattr(state, _df_name, pd.read_parquet(_pq))
                logger.info("Loaded %s from parquet: %d rows", _df_name, len(getattr(state, _df_name)))

    try:
        from operator1.clients.frequency_separator import (
            separate_by_period_type,
            build_highest_frequency_statement,
        )
    except ImportError as exc:
        logger.warning("Frequency separator not available: %s", exc)
        return

    market_id = state.market_id or ""

    for label, attr_name in [
        ("income", "income_df"),
        ("balance", "balance_df"),
        ("cashflow", "cashflow_df"),
    ]:
        stmt = getattr(state, attr_name, None)
        if stmt is None or stmt.empty:
            continue

        try:
            freq_groups = separate_by_period_type(stmt, market_id=market_id)
            if len(freq_groups) > 1:
                logger.info(
                    "Mixed-frequency %s detected: %s -- reconciling",
                    label,
                    {k: len(v) for k, v in freq_groups.items()},
                )
                reconciled = build_highest_frequency_statement(freq_groups)
                if not reconciled.empty:
                    setattr(state, attr_name, reconciled)
                    logger.info(
                        "Reconciled %s: %d rows x %d cols",
                        label, len(reconciled), len(reconciled.columns),
                    )
            else:
                logger.info(
                    "%s: single frequency (%s), no separation needed",
                    label,
                    list(freq_groups.keys())[0] if freq_groups else "unknown",
                )
        except Exception as exc:
            logger.warning("Frequency separation failed for %s: %s", label, exc)


def run_2_2_data_reconciliation(state: PipelineState) -> None:
    """2.2: Validate and clean reconciled statements.

    Runs accounting identity checks, field alias normalization,
    filing date validation, and duplicate removal on the reconciled
    statement DataFrames.
    """
    logger.info("Sub-stage 2.2: Data reconciliation")

    try:
        from operator1.quality.data_reconciliation import reconcile_financial_data
    except ImportError as exc:
        logger.warning("Data reconciliation not available: %s", exc)
        return

    income = state.income_df
    balance = state.balance_df
    cashflow = state.cashflow_df

    if income.empty and balance.empty and cashflow.empty:
        logger.info("No statement data to reconcile")
        return

    try:
        income_r, balance_r, cashflow_r, report = reconcile_financial_data(
            income, balance, cashflow,
        )
        state.income_df = income_r
        state.balance_df = balance_r
        state.cashflow_df = cashflow_r

        n_fixed = report.get("time_travel_fixes", 0) + report.get("duplicates_removed", 0)
        if n_fixed > 0:
            logger.info(
                "Reconciliation: %d time-travel fixes, %d duplicates removed",
                report.get("time_travel_fixes", 0),
                report.get("duplicates_removed", 0),
            )
    except Exception as exc:
        logger.warning("Data reconciliation failed: %s", exc)


# Registry of all Stage 2 sub-stages in order
STAGE_2_SUBSTAGES = [
    ("2.1", run_2_1_frequency_separation),
    ("2.2", run_2_2_data_reconciliation),
]
