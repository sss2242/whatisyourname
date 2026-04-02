"""Prediction log -- persistent IC tracking across pipeline runs.

Stores every prediction with metadata, fills in actual outcomes on
subsequent runs, and computes realized IC to close the learning loop.

Log format (JSONL, one record per prediction):
    {
        "ticker": "AAPL",
        "run_date": "2026-04-01",
        "variable": "return_5d",
        "horizon": "5d",
        "predicted_value": 0.0123,
        "predicted_direction": "up",
        "actual_value": null,        # filled on next run
        "actual_direction": null,
        "hit": null,                 # direction correct?
        "ic_contribution": null,     # Spearman rank contribution
        "model_used": "kalman",
        "confidence": 0.75,
        "survival_regime": "normal",
    }
"""

from __future__ import annotations

import json
import logging
import os
from datetime import date, datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

_LOG_PATH = Path("cache/prediction_log.jsonl")


def _load_log() -> list[dict]:
    """Load prediction log from disk."""
    if not _LOG_PATH.exists():
        return []
    records = []
    try:
        with open(_LOG_PATH, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line:
                    records.append(json.loads(line))
    except Exception as exc:
        logger.warning("Failed to load prediction log: %s", exc)
    return records


def _save_log(records: list[dict]) -> None:
    """Save prediction log to disk."""
    _LOG_PATH.parent.mkdir(parents=True, exist_ok=True)
    try:
        with open(_LOG_PATH, "w", encoding="utf-8") as f:
            for rec in records:
                f.write(json.dumps(rec, default=str) + "\n")
    except Exception as exc:
        logger.warning("Failed to save prediction log: %s", exc)


def store_predictions(
    ticker: str,
    predictions: dict[str, Any],
    model_used: dict[str, str] | None = None,
    survival_regime: str = "normal",
    run_date: date | None = None,
) -> int:
    """Store current run predictions to the log.

    Parameters
    ----------
    ticker:
        Company ticker.
    predictions:
        Nested dict: ``{variable: {horizon: HorizonPrediction_or_float}}``.
    model_used:
        Dict: ``{variable: model_name}``.
    survival_regime:
        Current survival regime label.
    run_date:
        Override for backtesting (default: today).

    Returns
    -------
    Number of predictions stored.
    """
    if run_date is None:
        run_date = date.today()

    records = _load_log()
    n_stored = 0

    for var, horizons in predictions.items():
        if not isinstance(horizons, dict):
            continue
        for h, pred_obj in horizons.items():
            # Extract point forecast from HorizonPrediction or raw float
            if hasattr(pred_obj, "point_forecast"):
                pf = pred_obj.point_forecast
                conf = getattr(pred_obj, "confidence", 0.5)
            elif isinstance(pred_obj, (int, float)):
                pf = float(pred_obj)
                conf = 0.5
            else:
                continue

            if pf is None or (isinstance(pf, float) and np.isnan(pf)):
                continue

            direction = "up" if pf > 0 else "down" if pf < 0 else "flat"

            record = {
                "ticker": ticker,
                "run_date": str(run_date),
                "variable": var,
                "horizon": h,
                "predicted_value": round(float(pf), 8),
                "predicted_direction": direction,
                "actual_value": None,
                "actual_direction": None,
                "hit": None,
                "ic_contribution": None,
                "model_used": (model_used or {}).get(var, "ensemble"),
                "confidence": round(float(conf), 4),
                "survival_regime": survival_regime,
            }
            records.append(record)
            n_stored += 1

    _save_log(records)
    logger.info("Prediction log: %d predictions stored for %s", n_stored, ticker)
    return n_stored


def fill_actuals(
    ticker: str,
    cache: pd.DataFrame,
    reference_date: date | None = None,
) -> dict[str, Any]:
    """Fill in actual outcomes for previous predictions.

    Scans the log for unfilled predictions from prior runs and
    computes actual returns from the cache.

    Returns
    -------
    Dict with summary: n_filled, n_hits, hit_rate, realized_ic.
    """
    if reference_date is None:
        reference_date = date.today()

    records = _load_log()
    if not records:
        return {"n_filled": 0, "n_hits": 0, "hit_rate": 0.0, "realized_ic": 0.0}

    n_filled = 0
    n_hits = 0
    n_total = 0

    for rec in records:
        if rec.get("ticker") != ticker:
            continue
        if rec.get("actual_value") is not None:
            continue  # already filled

        run_date_str = rec.get("run_date", "")
        variable = rec.get("variable", "")
        horizon = rec.get("horizon", "")

        # Parse horizon to days
        h_days = _parse_horizon_days(horizon)
        if h_days <= 0:
            continue

        # Find the target date (run_date + horizon)
        try:
            rd = datetime.strptime(run_date_str, "%Y-%m-%d").date()
        except (ValueError, TypeError):
            continue

        # Check if we have enough data to evaluate
        from datetime import timedelta
        target_date = rd + timedelta(days=int(h_days * 1.5))  # 1.5x for weekends
        if reference_date < target_date:
            continue  # not enough time has passed

        # Look up actual return in cache
        if variable not in cache.columns and "close" in cache.columns:
            # For return variables, compute from close
            if "return" in variable:
                actual = _compute_actual_return(cache, rd, h_days)
            else:
                actual = _lookup_actual(cache, variable, rd, h_days)
        else:
            actual = _lookup_actual(cache, variable, rd, h_days)

        if actual is not None and not np.isnan(actual):
            rec["actual_value"] = round(float(actual), 8)
            rec["actual_direction"] = "up" if actual > 0 else "down" if actual < 0 else "flat"
            rec["hit"] = (rec["predicted_direction"] == rec["actual_direction"])
            n_filled += 1
            n_total += 1
            if rec["hit"]:
                n_hits += 1

    _save_log(records)

    hit_rate = n_hits / n_total if n_total > 0 else 0.0

    # Compute realized IC (Spearman correlation of predicted vs actual)
    filled = [
        r for r in records
        if r.get("ticker") == ticker
        and r.get("actual_value") is not None
        and r.get("predicted_value") is not None
    ]
    realized_ic = 0.0
    if len(filled) >= 10:
        from scipy import stats
        predicted = [r["predicted_value"] for r in filled]
        actual = [r["actual_value"] for r in filled]
        ic, _ = stats.spearmanr(predicted, actual)
        if not np.isnan(ic):
            realized_ic = float(ic)

    summary = {
        "n_filled": n_filled,
        "n_hits": n_hits,
        "hit_rate": round(hit_rate, 4),
        "realized_ic": round(realized_ic, 4),
        "total_predictions": len([r for r in records if r.get("ticker") == ticker]),
    }

    if n_filled > 0:
        logger.info(
            "Prediction log: filled %d actuals, hit_rate=%.1f%%, IC=%.4f",
            n_filled, hit_rate * 100, realized_ic,
        )

    return summary


def _parse_horizon_days(horizon: str) -> int:
    """Parse horizon string to business days."""
    h = horizon.lower().strip()
    if h.endswith("d"):
        try:
            return int(h[:-1])
        except ValueError:
            pass
    if h.endswith("w"):
        try:
            return int(h[:-1]) * 5
        except ValueError:
            pass
    return 0


def _compute_actual_return(cache: pd.DataFrame, run_date: date, h_days: int) -> float | None:
    """Compute actual forward return from close price."""
    if "close" not in cache.columns:
        return None
    close = cache["close"].dropna()
    if close.empty:
        return None

    rd_ts = pd.Timestamp(run_date)
    # Find the closest date on or after run_date
    mask = close.index >= rd_ts
    if not mask.any():
        return None
    start_idx = close.index[mask][0]
    start_val = close[start_idx]

    # Find the value h_days later
    from datetime import timedelta
    target_ts = rd_ts + pd.Timedelta(days=int(h_days * 1.5))
    end_mask = close.index >= start_idx + pd.Timedelta(days=h_days)
    if not end_mask.any():
        return None
    end_idx = close.index[end_mask][0]
    end_val = close[end_idx]

    if start_val == 0:
        return None
    return float((end_val - start_val) / start_val)


def _lookup_actual(cache: pd.DataFrame, variable: str, run_date: date, h_days: int) -> float | None:
    """Look up actual value of a variable h_days after run_date."""
    if variable not in cache.columns:
        return None
    series = cache[variable].dropna()
    if series.empty:
        return None

    rd_ts = pd.Timestamp(run_date)
    target_ts = rd_ts + pd.Timedelta(days=int(h_days * 1.5))

    # Find closest value to target date
    mask = series.index >= target_ts
    if not mask.any():
        return None
    return float(series[series.index[mask][0]])
