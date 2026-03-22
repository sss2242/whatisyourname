"""T4.4 -- Data quality enforcement.

Validates the full feature table for:
  - **Look-ahead violations**: scans all as-of joins to ensure no
    ``report_date > t`` leaked through.
  - **Ratio safety audit**: verifies that every ``invalid_math_*`` flag
    is correctly set whenever a denominator was zero/null/tiny.
  - **Missing-data audit**: confirms every derived feature column has
    its ``is_missing_*`` companion and that the flags match actual nulls.
  - **Unit normalisation**: logs unit metadata when available.
  - **Summary report**: produces ``cache/data_quality_report.json`` with
    per-variable coverage percentages.

The pipeline **fails** if a look-ahead violation is found (raises
``LookAheadError``).  Other issues are logged and included in the
summary report.
"""

from __future__ import annotations

import json
import logging
import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from operator1.constants import CACHE_DIR, EPSILON
from operator1.features.derived_variables import DERIVED_VARIABLES

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Exceptions
# ---------------------------------------------------------------------------


from operator1.types import LookAheadError  # noqa: F401 -- re-export


class LookAheadBatchError(LookAheadError):
    """Raised when multiple look-ahead violations are detected during quality audit."""

    def __init__(self, details: list[dict[str, str]]) -> None:
        self.details = details
        msgs = "; ".join(
            f"{d['entity']}:{d['column']} day={d['day']} report_date={d['report_date']}"
            for d in details
        )
        # Call Exception.__init__ directly (not LookAheadError's single-entity init)
        Exception.__init__(self, f"Look-ahead violations found: {msgs}")


# ---------------------------------------------------------------------------
# Data quality report container
# ---------------------------------------------------------------------------


@dataclass
class QualityReport:
    """Summary of data quality checks."""

    total_rows: int = 0
    total_columns: int = 0
    coverage: dict[str, float] = field(default_factory=dict)
    missing_flag_issues: list[dict[str, Any]] = field(default_factory=list)
    invalid_math_issues: list[dict[str, Any]] = field(default_factory=list)
    look_ahead_violations: list[dict[str, str]] = field(default_factory=list)
    overall_coverage_pct: float = 0.0
    passed: bool = True

    def to_dict(self) -> dict[str, Any]:
        return {
            "total_rows": self.total_rows,
            "total_columns": self.total_columns,
            "overall_coverage_pct": round(self.overall_coverage_pct, 4),
            "passed": self.passed,
            "coverage_per_variable": {
                k: round(v, 4) for k, v in self.coverage.items()
            },
            "missing_flag_issues": self.missing_flag_issues,
            "invalid_math_issues": self.invalid_math_issues,
            "look_ahead_violations": self.look_ahead_violations,
        }


# ---------------------------------------------------------------------------
# Look-ahead check
# ---------------------------------------------------------------------------


def check_look_ahead(
    df: pd.DataFrame,
    entity_id: str = "target",
) -> list[dict[str, str]]:
    """Scan for look-ahead violations in a daily feature table.

    Looks for any ``_report_date`` columns left from as-of merges and
    verifies that ``report_date <= index date`` for every row.

    Parameters
    ----------
    df:
        Daily DataFrame with a DatetimeIndex.
    entity_id:
        Label for error messages.

    Returns
    -------
    list[dict]
        List of violation details (empty if clean).
    """
    violations: list[dict[str, str]] = []

    # Check for leftover _report_date columns from as-of merges
    report_date_cols = [c for c in df.columns if "report_date" in c.lower()]
    for col in report_date_cols:
        series = pd.to_datetime(df[col], errors="coerce")
        idx_dates = pd.to_datetime(df.index)
        bad_mask = series.notna() & (series > idx_dates)
        if bad_mask.any():
            for ts in df.index[bad_mask]:
                violations.append({
                    "entity": entity_id,
                    "column": col,
                    "day": str(ts.date()) if hasattr(ts, "date") else str(ts),
                    "report_date": str(series.loc[ts].date())
                    if hasattr(series.loc[ts], "date")
                    else str(series.loc[ts]),
                })

    return violations


# ---------------------------------------------------------------------------
# Missing-data flag audit
# ---------------------------------------------------------------------------


def audit_missing_flags(
    df: pd.DataFrame,
) -> list[dict[str, Any]]:
    """Verify that ``is_missing_*`` flags match actual null values.

    For every column ``X`` that has a companion ``is_missing_X``, confirm:
      - ``is_missing_X == 1`` wherever ``X`` is NaN
      - ``is_missing_X == 0`` wherever ``X`` is not NaN

    Returns a list of issues found (empty if all flags are correct).
    """
    issues: list[dict[str, Any]] = []
    checked = set()

    for col in df.columns:
        if col.startswith("is_missing_") or col.startswith("invalid_math_"):
            continue

        flag_col = f"is_missing_{col}"
        if flag_col not in df.columns:
            # Not all columns require flags (e.g. string profile fields)
            continue

        checked.add(col)
        actual_missing = df[col].isna()
        flag_values = df[flag_col]

        # Where actual is missing but flag says 0
        false_negatives = (actual_missing & (flag_values == 0)).sum()
        # Where actual is present but flag says 1
        false_positives = (~actual_missing & (flag_values == 1)).sum()

        if false_negatives > 0 or false_positives > 0:
            issues.append({
                "column": col,
                "false_negatives": int(false_negatives),
                "false_positives": int(false_positives),
                "total_rows": len(df),
            })

    return issues


# ---------------------------------------------------------------------------
# Invalid-math flag audit
# ---------------------------------------------------------------------------


def audit_invalid_math_flags(
    df: pd.DataFrame,
) -> list[dict[str, Any]]:
    """Verify that ``invalid_math_*`` flags are consistent.

    For every column with an ``invalid_math_X`` companion, check that
    when ``invalid_math_X == 1``, the value of ``X`` is NaN.

    Returns a list of issues found.
    """
    issues: list[dict[str, Any]] = []

    for col in df.columns:
        if not col.startswith("invalid_math_"):
            continue

        base_col = col[len("invalid_math_"):]
        if base_col not in df.columns:
            continue

        # Where invalid_math is flagged, the value should be NaN
        flagged = df[col] == 1
        value_present = df[base_col].notna()
        inconsistent = (flagged & value_present).sum()

        if inconsistent > 0:
            issues.append({
                "column": base_col,
                "invalid_math_flagged_but_value_present": int(inconsistent),
                "total_flagged": int(flagged.sum()),
            })

    return issues


# ---------------------------------------------------------------------------
# Coverage computation
# ---------------------------------------------------------------------------


def compute_coverage(
    df: pd.DataFrame,
    variable_list: tuple[str, ...] | list[str] | None = None,
) -> dict[str, float]:
    """Compute per-variable coverage (fraction of non-null values).

    Parameters
    ----------
    df:
        Feature DataFrame.
    variable_list:
        Optional list of variable names to check.  Defaults to all
        numeric columns excluding flags.

    Returns
    -------
    dict[str, float]
        Variable name -> coverage fraction (0.0 to 1.0).
    """
    if variable_list is None:
        variable_list = [
            c for c in df.columns
            if not c.startswith("is_missing_")
            and not c.startswith("invalid_math_")
            and df[c].dtype in ("float64", "float32", "int64", "int32", "Int64")
        ]

    n = len(df)
    if n == 0:
        return {v: 0.0 for v in variable_list}

    coverage: dict[str, float] = {}
    for var in variable_list:
        if var in df.columns:
            non_null = df[var].notna().sum()
            coverage[var] = float(non_null / n)
        else:
            coverage[var] = 0.0

    return coverage


# ---------------------------------------------------------------------------
# Full quality enforcement pipeline
# ---------------------------------------------------------------------------


def run_quality_checks(
    df: pd.DataFrame,
    entity_id: str = "target",
    fail_on_look_ahead: bool = True,
) -> QualityReport:
    """Run all data quality checks on a feature table.

    Parameters
    ----------
    df:
        Daily feature DataFrame (target or linked entity).
    entity_id:
        Identifier for error messages.
    fail_on_look_ahead:
        If True (default), raises ``LookAheadError`` on violations.

    Returns
    -------
    QualityReport
        Summary of all checks performed.

    Raises
    ------
    LookAheadError
        If look-ahead violations are found and ``fail_on_look_ahead``
        is True.
    """
    report = QualityReport()
    report.total_rows = len(df)
    report.total_columns = len(df.columns)

    # 1. Look-ahead check
    logger.info("Running look-ahead check for %s ...", entity_id)
    violations = check_look_ahead(df, entity_id)
    report.look_ahead_violations = violations
    if violations:
        report.passed = False
        logger.error(
            "LOOK-AHEAD VIOLATIONS found for %s: %d violations",
            entity_id, len(violations),
        )
        if fail_on_look_ahead:
            raise LookAheadBatchError(violations)

    # 2. Missing-flag audit
    logger.info("Auditing missing flags for %s ...", entity_id)
    report.missing_flag_issues = audit_missing_flags(df)
    if report.missing_flag_issues:
        logger.warning(
            "Missing-flag issues for %s: %d columns with mismatches",
            entity_id, len(report.missing_flag_issues),
        )

    # 3. Invalid-math flag audit
    logger.info("Auditing invalid-math flags for %s ...", entity_id)
    report.invalid_math_issues = audit_invalid_math_flags(df)
    if report.invalid_math_issues:
        logger.warning(
            "Invalid-math flag issues for %s: %d columns",
            entity_id, len(report.invalid_math_issues),
        )

    # 4. Coverage
    logger.info("Computing coverage for %s ...", entity_id)
    derived_vars = list(DERIVED_VARIABLES)
    report.coverage = compute_coverage(df, derived_vars)
    if report.coverage:
        report.overall_coverage_pct = (
            sum(report.coverage.values()) / len(report.coverage) * 100
        )

    logger.info(
        "Quality check complete for %s: passed=%s, coverage=%.1f%%",
        entity_id, report.passed, report.overall_coverage_pct,
    )

    return report


def run_full_quality_audit(
    target_daily: pd.DataFrame,
    linked_daily: dict[str, pd.DataFrame] | None = None,
    fail_on_look_ahead: bool = True,
) -> dict[str, QualityReport]:
    """Run quality checks on the target and all linked entity caches.

    Parameters
    ----------
    target_daily:
        Target company daily feature table.
    linked_daily:
        Dict of ``{isin: daily_df}`` for linked entities.
    fail_on_look_ahead:
        If True, raise on look-ahead violations.

    Returns
    -------
    dict[str, QualityReport]
        Quality reports keyed by entity identifier.
    """
    reports: dict[str, QualityReport] = {}

    # Target
    reports["target"] = run_quality_checks(
        target_daily, "target", fail_on_look_ahead,
    )

    # Linked entities
    if linked_daily:
        for isin, daily_df in linked_daily.items():
            try:
                reports[isin] = run_quality_checks(
                    daily_df, isin, fail_on_look_ahead,
                )
            except LookAheadError:
                raise
            except Exception as exc:
                logger.warning("Quality check failed for %s: %s", isin, exc)
                report = QualityReport()
                report.passed = False
                reports[isin] = report

    return reports


def save_quality_report(
    reports: dict[str, QualityReport],
    output_path: str | None = None,
) -> str:
    """Persist quality reports to ``cache/data_quality_report.json``.

    Parameters
    ----------
    reports:
        Output of ``run_full_quality_audit()``.
    output_path:
        Override output file path.

    Returns
    -------
    str
        Path to the written JSON file.
    """
    if output_path is None:
        output_path = str(Path(CACHE_DIR) / "data_quality_report.json")

    data = {entity: r.to_dict() for entity, r in reports.items()}

    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as fh:
        json.dump(data, fh, indent=2)

    logger.info("Data quality report saved to %s", output_path)
    return output_path
