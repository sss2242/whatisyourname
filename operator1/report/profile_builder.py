"""T7.1 -- Profile JSON builder.

Aggregates all pipeline outputs into a single ``company_profile.json``
that feeds report generation (T7.2) and the Gemini narrative (Sec 18).

Sections included:

1. **Target identity** -- verified ISIN, ticker, country, sector, etc.
2. **Survival flags** -- company, country, and protection flags (latest).
3. **Linked entity aggregates** -- summary stats per relationship group.
4. **Temporal results** -- regime labels, structural breaks, predictions,
   model metrics.
5. **Vanity percentage** -- latest and summary statistics.
6. **Monte Carlo** -- survival probability mean/p5/p95.
7. **Data quality** -- coverage percentages, failed modules.
8. **Estimation coverage** -- per-variable observed vs estimated stats.

Top-level entry point:
    ``build_company_profile(...)`` -> dict (also persisted as JSON).

Spec refs: Sec 18
"""

from __future__ import annotations

import json
import logging
import math
import os
from dataclasses import asdict, dataclass, field
from datetime import date, datetime
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from operator1.constants import CACHE_DIR, DATE_START, DATE_END

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _safe_float(val: Any) -> float | None:
    """Convert a value to a JSON-safe float (None for NaN/Inf)."""
    if val is None:
        return None
    try:
        f = float(val)
        if math.isnan(f) or math.isinf(f):
            return None
        return f
    except (TypeError, ValueError):
        return None


def _safe_str(val: Any) -> str | None:
    """Convert a value to a JSON-safe string (None for NaN/None)."""
    if val is None:
        return None
    if isinstance(val, float) and math.isnan(val):
        return None
    return str(val)


def _latest_row(df: pd.DataFrame) -> pd.Series | None:
    """Return the last row of a DataFrame, or None if empty."""
    if df is None or df.empty:
        return None
    return df.iloc[-1]


def _series_summary(s: pd.Series) -> dict[str, Any]:
    """Compute basic summary stats for a numeric Series."""
    if s is None or s.empty or s.isna().all():
        return {
            "mean": None,
            "median": None,
            "min": None,
            "max": None,
            "latest": None,
            "n_observed": 0,
            "n_missing": 0,
        }
    return {
        "mean": _safe_float(s.mean()),
        "median": _safe_float(s.median()),
        "min": _safe_float(s.min()),
        "max": _safe_float(s.max()),
        "latest": _safe_float(s.iloc[-1]),
        "n_observed": int(s.notna().sum()),
        "n_missing": int(s.isna().sum()),
    }


def _json_serialisable(obj: Any) -> Any:
    """Recursively convert numpy/pandas types for JSON serialisation."""
    if isinstance(obj, dict):
        return {k: _json_serialisable(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [_json_serialisable(v) for v in obj]
    if isinstance(obj, (np.integer,)):
        return int(obj)
    if isinstance(obj, (np.floating,)):
        return _safe_float(obj)
    if isinstance(obj, np.ndarray):
        return _json_serialisable(obj.tolist())
    if isinstance(obj, (np.bool_,)):
        return bool(obj)
    if isinstance(obj, (pd.Timestamp, datetime)):
        return obj.isoformat()
    if isinstance(obj, date):
        return obj.isoformat()
    if isinstance(obj, float) and (math.isnan(obj) or math.isinf(obj)):
        return None
    return obj


# ---------------------------------------------------------------------------
# Section builders
# ---------------------------------------------------------------------------


def _build_identity_section(
    verified_target: dict[str, Any] | None,
) -> dict[str, Any]:
    """Build the target identity section from VerifiedTarget data.

    Accepts either a VerifiedTarget dataclass (converted to dict) or
    a plain dict.
    """
    if not verified_target:
        return {"available": False}

    # Accept dataclass instances via asdict
    if hasattr(verified_target, "__dataclass_fields__"):
        verified_target = asdict(verified_target)  # type: ignore[arg-type]

    return {
        "available": True,
        "isin": verified_target.get("isin"),
        "ticker": verified_target.get("ticker"),
        "name": verified_target.get("name"),
        "country": verified_target.get("country"),
        "sector": verified_target.get("sector"),
        "industry": verified_target.get("industry"),
        "sub_industry": verified_target.get("sub_industry"),
        "fmp_symbol": verified_target.get("fmp_symbol"),
        "currency": verified_target.get("currency"),
        "exchange": verified_target.get("exchange"),
    }


def _build_current_state_section(cache: pd.DataFrame | None) -> dict[str, Any]:
    """Build the current state tier-by-tier snapshot (latest day values).

    Extracts the most recent value for each variable in every tier so
    the report and Gemini prompt can show actual ratios, not just scores.

    Spec ref: Sec F.1 Category 2
    """
    if cache is None or cache.empty:
        return {"available": False}

    latest = _latest_row(cache)
    if latest is None:
        return {"available": False}

    def _get(col: str) -> float | None:
        val = latest.get(col)
        return _safe_float(val) if val is not None else None

    section: dict[str, Any] = {
        "available": True,
        "date": str(latest.name.date()) if hasattr(latest.name, "date") else str(latest.name),
        "tier1_liquidity": {
            "cash_and_equivalents": _get("cash_and_equivalents"),
            "cash_ratio": _get("cash_ratio"),
            "current_ratio": _get("current_ratio"),
            "free_cash_flow_ttm": _get("free_cash_flow_ttm_asof"),
            "operating_cash_flow": _get("operating_cash_flow"),
        },
        "tier2_solvency": {
            "total_debt": _get("total_debt_asof"),
            "debt_to_equity": _get("debt_to_equity_abs"),
            "net_debt": _get("net_debt"),
            "net_debt_to_ebitda": _get("net_debt_to_ebitda"),
            "interest_coverage": _get("interest_coverage"),
            "total_equity": _get("total_equity"),
        },
        "tier3_stability": {
            "volatility_21d": _get("volatility_21d"),
            "drawdown_252d": _get("drawdown_252d"),
            "volume": _get("volume"),
            "volume_avg_21d": _get("volume_avg_21d"),
            "close": _get("close"),
        },
        "tier4_profitability": {
            "gross_margin": _get("gross_margin"),
            "operating_margin": _get("operating_margin"),
            "net_margin": _get("net_margin"),
            "roe": _get("roe"),
            "roa": _get("roa"),
            "revenue_ttm": _get("revenue_ttm_asof"),
            "net_income_ttm": _get("net_income_ttm_asof"),
            "ebitda_ttm": _get("ebitda_ttm_asof"),
        },
        "tier5_growth": {
            "pe_ratio": _get("pe_ratio_calc"),
            "earnings_yield": _get("earnings_yield_calc"),
            "ps_ratio": _get("ps_ratio_calc"),
            "pb_ratio": _get("pb_ratio"),
            "ev_to_ebitda": _get("ev_to_ebitda"),
            "enterprise_value": _get("enterprise_value"),
            "fcf_yield": _get("fcf_yield"),
            "revenue_growth_yoy": _get("revenue_growth_yoy"),
            "earnings_growth_yoy": _get("earnings_growth_yoy"),
        },
    }

    # Add regime info
    if "regime_label" in cache.columns:
        section["market_regime"] = _safe_str(latest.get("regime_label"))
    if "regime_fundamental" in cache.columns:
        section["fundamental_regime"] = _safe_str(latest.get("regime_fundamental"))
    if "survival_regime" in cache.columns:
        section["hierarchy_regime"] = _safe_str(latest.get("survival_regime"))

    return section


def _build_survival_episodes(cache: pd.DataFrame | None) -> dict[str, Any]:
    """Compute survival episode statistics from the cache.

    Identifies contiguous periods of survival mode and computes:
    - total survival days
    - episode count, lengths, longest, average

    Spec ref: Sec F.1 Category 4
    """
    if cache is None or cache.empty:
        return {"available": False}

    result: dict[str, Any] = {"available": True}

    for flag_name, label in [
        ("company_survival_mode_flag", "company"),
        ("country_survival_mode_flag", "country"),
    ]:
        if flag_name not in cache.columns:
            result[f"{label}_survival_days"] = 0
            result[f"{label}_episodes"] = 0
            continue

        flags = cache[flag_name].fillna(0).astype(int)
        total_days = int(flags.sum())
        result[f"{label}_survival_days"] = total_days

        # Detect episodes (contiguous runs of 1s)
        diff = flags.diff().fillna(flags.iloc[0])
        starts = flags.index[diff == 1].tolist()
        ends = flags.index[diff == -1].tolist()

        # Handle case where series ends in survival mode
        if len(starts) > len(ends):
            ends.append(flags.index[-1])

        episodes: list[dict[str, Any]] = []
        for s, e in zip(starts, ends):
            length = int((cache.index.get_loc(e) - cache.index.get_loc(s)) + 1)
            episodes.append({
                "start": str(s.date()) if hasattr(s, "date") else str(s),
                "end": str(e.date()) if hasattr(e, "date") else str(e),
                "length_days": length,
            })

        result[f"{label}_episodes"] = len(episodes)
        if episodes:
            lengths = [ep["length_days"] for ep in episodes]
            result[f"{label}_longest_episode"] = max(lengths)
            result[f"{label}_avg_episode_length"] = round(sum(lengths) / len(lengths), 1)
            result[f"{label}_episode_details"] = episodes[-5:]  # last 5 episodes
        else:
            result[f"{label}_longest_episode"] = 0
            result[f"{label}_avg_episode_length"] = 0
            result[f"{label}_episode_details"] = []

    # Both in survival simultaneously
    if "company_survival_mode_flag" in cache.columns and "country_survival_mode_flag" in cache.columns:
        both = (
            cache["company_survival_mode_flag"].fillna(0).astype(int)
            & cache["country_survival_mode_flag"].fillna(0).astype(int)
        )
        result["both_survival_days"] = int(both.sum())

    return result


def _build_survival_section(cache: pd.DataFrame | None) -> dict[str, Any]:
    """Extract latest survival flags and regime from the cache."""
    if cache is None or cache.empty:
        return {"available": False}

    latest = _latest_row(cache)
    if latest is None:
        return {"available": False}

    # Regime distribution over the entire window
    regime_col = "survival_regime"
    regime_dist: dict[str, Any] = {}
    if regime_col in cache.columns:
        vc = cache[regime_col].value_counts(normalize=True)
        regime_dist = {str(k): round(float(v), 4) for k, v in vc.items()}

    return {
        "available": True,
        "company_survival_mode_flag": int(latest.get("company_survival_mode_flag", 0)),
        "country_survival_mode_flag": int(latest.get("country_survival_mode_flag", 0)),
        "country_protected_flag": int(latest.get("country_protected_flag", 0)),
        "survival_regime": _safe_str(latest.get("survival_regime")),
        "regime_distribution_pct": regime_dist,
        "hierarchy_weights": {
            f"tier{i}": _safe_float(latest.get(f"hierarchy_tier{i}_weight"))
            for i in range(1, 6)
        },
    }


def _build_vanity_section(cache: pd.DataFrame | None) -> dict[str, Any]:
    """Summarise vanity percentage and v2 score from the cache."""
    if cache is None or cache.empty:
        return {"available": False}

    section: dict[str, Any] = {"available": False}

    # Legacy vanity_percentage
    if "vanity_percentage" in cache.columns:
        section.update({
            "available": True,
            **_series_summary(cache["vanity_percentage"]),
        })

    # V2 vanity score
    if "vanity_score" in cache.columns and cache["vanity_score"].notna().any():
        section["v2_available"] = True
        section["v2_score"] = _series_summary(cache["vanity_score"])
        section["v2_score_21d"] = _series_summary(
            cache["vanity_score_21d"],
        ) if "vanity_score_21d" in cache.columns else {}

        # Latest label and trend
        last_valid = cache["vanity_score"].last_valid_index()
        if last_valid is not None:
            section["v2_label"] = _safe_str(
                cache.at[last_valid, "vanity_label"]
                if "vanity_label" in cache.columns else None,
            )
            section["v2_trend"] = _safe_str(
                cache.at[last_valid, "vanity_trend"]
                if "vanity_trend" in cache.columns else None,
            )

        # Component breakdown (latest values)
        v2_components = [
            "vanity_rnd_mismatch",
            "vanity_sga_bloat_v2",
            "vanity_capital_misallocation",
            "vanity_competitive_decay",
            "vanity_sentiment_gap",
        ]
        breakdown = {}
        for comp in v2_components:
            if comp in cache.columns:
                s = cache[comp]
                last_idx = s.last_valid_index()
                breakdown[comp] = _safe_float(
                    s.at[last_idx] if last_idx is not None else None,
                )
        section["v2_breakdown"] = breakdown
    else:
        section["v2_available"] = False

    return section


def _build_linked_section(
    linked_aggregates: pd.DataFrame | None,
) -> dict[str, Any]:
    """Summarise linked entity aggregates by relationship group."""
    if linked_aggregates is None or linked_aggregates.empty:
        return {"available": False, "groups": {}}

    groups: dict[str, dict[str, Any]] = {}

    # Detect group columns by naming convention: {group}_avg_{var} etc.
    group_prefixes = set()
    for col in linked_aggregates.columns:
        parts = col.split("_")
        if len(parts) >= 3 and parts[-2] in ("avg", "median"):
            group_prefixes.add("_".join(parts[:-2]))

    for prefix in sorted(group_prefixes):
        group_info: dict[str, Any] = {}
        for col in linked_aggregates.columns:
            if col.startswith(prefix + "_"):
                suffix = col[len(prefix) + 1 :]
                group_info[suffix] = _series_summary(linked_aggregates[col])
        if group_info:
            groups[prefix] = group_info

    return {
        "available": bool(groups),
        "n_groups": len(groups),
        "groups": groups,
    }


def _build_regime_section(
    cache: pd.DataFrame | None,
    regime_result: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Summarise regime detection outputs."""
    if cache is None or cache.empty:
        return {"available": False}

    section: dict[str, Any] = {"available": True}

    # Current regime from cache
    if "regime_label" in cache.columns:
        latest = cache["regime_label"].iloc[-1]
        section["current_regime"] = _safe_str(latest)
        vc = cache["regime_label"].value_counts(normalize=True)
        section["regime_distribution_pct"] = {
            str(k): round(float(v), 4) for k, v in vc.items()
        }
    else:
        section["current_regime"] = None
        section["regime_distribution_pct"] = {}

    # Structural breaks
    if "structural_break" in cache.columns:
        break_days = cache.index[cache["structural_break"] == 1].tolist()
        section["n_structural_breaks"] = len(break_days)
        section["structural_break_dates"] = [
            str(d) for d in break_days[-10:]  # last 10
        ]
    else:
        section["n_structural_breaks"] = 0
        section["structural_break_dates"] = []

    # Model status flags from regime result
    if regime_result:
        section["hmm_fitted"] = regime_result.get("hmm_fitted", False)
        section["gmm_fitted"] = regime_result.get("gmm_fitted", False)
        section["pelt_fitted"] = regime_result.get("pelt_fitted", False)
        section["bcp_fitted"] = regime_result.get("bcp_fitted", False)

    return section


def _build_predictions_section(
    prediction_result: dict[str, Any] | None,
) -> dict[str, Any]:
    """Format prediction aggregation results for the profile."""
    if not prediction_result:
        return {"available": False}

    section: dict[str, Any] = {"available": True}

    # Extract flat prediction summaries per horizon
    predictions = prediction_result.get("predictions", {})
    horizon_summaries: dict[str, list[dict[str, Any]]] = {}

    for var_name, horizons in predictions.items():
        for h_label, pred in horizons.items():
            if h_label not in horizon_summaries:
                horizon_summaries[h_label] = []

            if isinstance(pred, dict):
                entry = {
                    "variable": var_name,
                    "point_forecast": _safe_float(pred.get("point_forecast")),
                    "lower_ci": _safe_float(pred.get("lower_ci")),
                    "upper_ci": _safe_float(pred.get("upper_ci")),
                    "confidence": _safe_float(pred.get("confidence")),
                }
            else:
                # Might be a dataclass
                entry = {
                    "variable": var_name,
                    "point_forecast": _safe_float(
                        getattr(pred, "point_forecast", None),
                    ),
                    "lower_ci": _safe_float(
                        getattr(pred, "lower_ci", None),
                    ),
                    "upper_ci": _safe_float(
                        getattr(pred, "upper_ci", None),
                    ),
                    "confidence": _safe_float(
                        getattr(pred, "confidence", None),
                    ),
                }
            horizon_summaries[h_label].append(entry)

    section["horizons"] = horizon_summaries

    # Technical Alpha
    ta = prediction_result.get("technical_alpha")
    if ta:
        if isinstance(ta, dict):
            section["technical_alpha"] = {
                "next_day_low": _safe_float(ta.get("next_day_low")),
                "mask_applied": ta.get("mask_applied", True),
            }
        else:
            section["technical_alpha"] = {
                "next_day_low": _safe_float(
                    getattr(ta, "next_day_low", None),
                ),
                "mask_applied": getattr(ta, "mask_applied", True),
            }

    # Ensemble weights
    section["ensemble_weights"] = prediction_result.get(
        "ensemble_weights", {},
    )
    section["n_models_available"] = prediction_result.get(
        "n_models_available", 0,
    )
    section["n_models_failed"] = prediction_result.get(
        "n_models_failed", 0,
    )

    return section


def _build_monte_carlo_section(
    mc_result: dict[str, Any] | None,
) -> dict[str, Any]:
    """Format Monte Carlo simulation results."""
    if not mc_result:
        return {"available": False}

    return {
        "available": True,
        "survival_probability_mean": _safe_float(
            mc_result.get("survival_probability_mean"),
        ),
        "survival_probability_p5": _safe_float(
            mc_result.get("survival_probability_p5"),
        ),
        "survival_probability_p95": _safe_float(
            mc_result.get("survival_probability_p95"),
        ),
        "n_paths": mc_result.get("n_paths", 0),
        "importance_sampling_used": mc_result.get(
            "importance_sampling_used", False,
        ),
        "current_regime": _safe_str(mc_result.get("current_regime")),
        "survival_by_horizon": {
            k: _safe_float(v)
            for k, v in mc_result.get("survival_probability", {}).items()
        },
    }


def _build_model_metrics_section(
    forecast_result: dict[str, Any] | None,
) -> dict[str, Any]:
    """Summarise model metrics from forecasting results."""
    if not forecast_result:
        return {"available": False}

    metrics = forecast_result.get("metrics", [])
    model_status: dict[str, Any] = {}

    # Collect model failure flags
    for key in (
        "model_failed_kalman",
        "model_failed_garch",
        "model_failed_var",
        "model_failed_lstm",
        "model_failed_tree",
    ):
        val = forecast_result.get(key)
        if val is not None:
            model_status[key] = bool(val)

    # Summarise per-model best RMSE
    model_best_rmse: dict[str, float | None] = {}
    for m in metrics:
        if isinstance(m, dict):
            name = m.get("model_name", "")
            fitted = m.get("fitted", False)
            rmse = m.get("rmse", float("nan"))
        else:
            name = getattr(m, "model_name", "")
            fitted = getattr(m, "fitted", False)
            rmse = getattr(m, "rmse", float("nan"))

        if fitted and name:
            rmse_val = _safe_float(rmse)
            if rmse_val is not None:
                if name not in model_best_rmse or (
                    model_best_rmse[name] is not None
                    and rmse_val < model_best_rmse[name]
                ):
                    model_best_rmse[name] = rmse_val

    return {
        "available": True,
        "model_status": model_status,
        "model_best_rmse": model_best_rmse,
        "model_used": forecast_result.get("model_used", {}),
    }


def _build_data_quality_section(
    quality_report_path: str | Path | None = None,
) -> dict[str, Any]:
    """Load data quality report from disk if available."""
    if quality_report_path is None:
        quality_report_path = Path(CACHE_DIR) / "data_quality_report.json"

    path = Path(quality_report_path)
    if not path.exists():
        return {"available": False}

    try:
        with open(path, "r", encoding="utf-8") as fh:
            report = json.load(fh)
        return {"available": True, **report}
    except (json.JSONDecodeError, OSError) as exc:
        logger.warning("Failed to load data quality report: %s", exc)
        return {"available": False, "error": str(exc)}


def _build_estimation_section(
    estimation_coverage_path: str | Path | None = None,
) -> dict[str, Any]:
    """Load estimation coverage stats from disk if available."""
    if estimation_coverage_path is None:
        estimation_coverage_path = (
            Path(CACHE_DIR) / "estimation_coverage.json"
        )

    path = Path(estimation_coverage_path)
    if not path.exists():
        return {"available": False}

    try:
        with open(path, "r", encoding="utf-8") as fh:
            coverage = json.load(fh)
        return {"available": True, **coverage}
    except (json.JSONDecodeError, OSError) as exc:
        logger.warning("Failed to load estimation coverage: %s", exc)
        return {"available": False, "error": str(exc)}


def _build_historical_section(
    cache: pd.DataFrame | None,
) -> dict[str, Any]:
    """Build the historical performance section from cache price data.

    Computes total return, annualised return/volatility, Sharpe ratio,
    max drawdown, and day-level statistics over the analysis window.
    """
    if cache is None or cache.empty or "close" not in cache.columns:
        return {"available": False}

    close = cache["close"].dropna()
    if len(close) < 2:
        return {"available": False}

    first_price = float(close.iloc[0])
    last_price = float(close.iloc[-1])
    n_days = len(close)

    # Total return
    total_return = (last_price - first_price) / first_price if first_price != 0 else None

    # Daily returns
    daily_returns = close.pct_change().dropna()
    if daily_returns.empty:
        return {"available": False}

    # Annualised return (252 trading days)
    trading_days = len(daily_returns)
    years = trading_days / 252.0
    annualised_return = _safe_float(
        ((1 + total_return) ** (1.0 / years) - 1) if total_return is not None and years > 0 else None
    )

    # Annualised volatility
    daily_vol = float(daily_returns.std())
    annualised_vol = _safe_float(daily_vol * (252 ** 0.5))

    # Sharpe ratio (assuming risk-free rate ~0 for simplicity)
    sharpe = _safe_float(
        (annualised_return / annualised_vol) if annualised_return is not None and annualised_vol and annualised_vol > 0 else None
    )

    # Maximum drawdown
    cummax = close.cummax()
    drawdown = (close - cummax) / cummax
    max_dd = _safe_float(float(drawdown.min()))

    # Real return (inflation-adjusted) if CPI data is in the cache
    real_return = None
    for col in ("cpi_annual", "inflation_annual", "inflation_rate"):
        if col in cache.columns:
            inf_series = cache[col].dropna()
            if not inf_series.empty:
                avg_inflation = float(inf_series.mean())
                if total_return is not None:
                    real_return = _safe_float(total_return - avg_inflation)
                break

    # Up/down day statistics
    up_days = float((daily_returns > 0).sum())
    down_days = float((daily_returns < 0).sum())
    total_counted = up_days + down_days

    section: dict[str, Any] = {
        "available": True,
        "date_range_start": str(close.index[0].date()) if hasattr(close.index[0], "date") else str(close.index[0]),
        "date_range_end": str(close.index[-1].date()) if hasattr(close.index[-1], "date") else str(close.index[-1]),
        "n_trading_days": n_days,
        "return_total": _safe_float(total_return),
        "return_real": real_return,
        "return_annualized": annualised_return,
        "volatility_annualized": annualised_vol,
        "sharpe_ratio": sharpe,
        "max_drawdown": max_dd,
        "up_days_percentage": _safe_float(up_days / total_counted) if total_counted > 0 else None,
        "down_days_percentage": _safe_float(down_days / total_counted) if total_counted > 0 else None,
        "best_day_return": _safe_float(float(daily_returns.max())),
        "worst_day_return": _safe_float(float(daily_returns.min())),
    }

    return section


def _build_ethical_filters_section(
    cache: pd.DataFrame | None,
) -> dict[str, Any]:
    """Run the four ethical filters and return the combined result."""
    if cache is None or cache.empty:
        return {
            "available": False,
            "purchasing_power": {"available": False, "verdict": "UNAVAILABLE"},
            "solvency": {"available": False, "verdict": "UNAVAILABLE"},
            "gharar": {"available": False, "verdict": "UNAVAILABLE"},
            "cash_is_king": {"available": False, "verdict": "UNAVAILABLE"},
        }
    try:
        from operator1.analysis.ethical_filters import compute_all_ethical_filters
        result = compute_all_ethical_filters(cache)
        result["available"] = True
        return result
    except Exception as exc:
        logger.warning("Ethical filter computation failed: %s", exc)
        return {
            "available": False,
            "error": str(exc),
            "purchasing_power": {"available": False, "verdict": "UNAVAILABLE"},
            "solvency": {"available": False, "verdict": "UNAVAILABLE"},
            "gharar": {"available": False, "verdict": "UNAVAILABLE"},
            "cash_is_king": {"available": False, "verdict": "UNAVAILABLE"},
        }


def _build_failed_modules_section(
    *,
    regime_result: dict[str, Any] | None = None,
    forecast_result: dict[str, Any] | None = None,
    mc_result: dict[str, Any] | None = None,
    prediction_result: dict[str, Any] | None = None,
) -> list[dict[str, str]]:
    """Compile a list of failed modules with mitigations applied."""
    failed: list[dict[str, str]] = []

    # Check regime detection
    if regime_result:
        for method, key in [
            ("HMM", "hmm_fitted"),
            ("GMM", "gmm_fitted"),
            ("PELT", "pelt_fitted"),
            ("BCP", "bcp_fitted"),
        ]:
            if not regime_result.get(key, False):
                error = regime_result.get(f"{key.split('_')[0]}_error", "")
                failed.append({
                    "module": f"Regime detection ({method})",
                    "error": error or "not fitted",
                    "mitigation": (
                        "Regime labels derived from available methods; "
                        "fallback to single-regime if all fail."
                    ),
                })

    # Check forecasting models
    if forecast_result:
        model_names = {
            "kalman": "Kalman filter",
            "garch": "GARCH",
            "var": "VAR",
            "lstm": "LSTM",
            "tree": "RF/GBM/XGB",
        }
        for key, name in model_names.items():
            if forecast_result.get(f"model_failed_{key}", False):
                error = forecast_result.get(f"{key}_error", "")
                failed.append({
                    "module": f"Forecasting ({name})",
                    "error": error or "model failed",
                    "mitigation": (
                        "Forecasts produced by next model in fallback "
                        "chain; baseline (last-value/EMA) always succeeds."
                    ),
                })

    # Check Monte Carlo
    if mc_result:
        mc_error = mc_result.get("error")
        if mc_error:
            failed.append({
                "module": "Monte Carlo simulation",
                "error": mc_error,
                "mitigation": "Survival probability set to NaN; uncertainty bands not adjusted.",
            })

    # Check prediction aggregation
    if prediction_result:
        pred_error = prediction_result.get("error")
        if pred_error:
            failed.append({
                "module": "Prediction aggregation",
                "error": pred_error,
                "mitigation": "Predictions unavailable; report limited to historical analysis.",
            })

    return failed


# ---------------------------------------------------------------------------
# Financial health section
# ---------------------------------------------------------------------------


def _build_financial_health_section(
    cache: pd.DataFrame | None,
    fh_result: dict[str, Any] | None,
) -> dict[str, Any]:
    """Build the financial health section from cache fh_* columns and result summary."""
    section: dict[str, Any] = {"available": False}

    # If we have a pre-computed result dict, use it as the base
    if fh_result:
        section = {
            "available": True,
            "latest_composite": _safe_float(fh_result.get("latest_composite")),
            "latest_label": fh_result.get("latest_label", "Unknown"),
            "mean_composite": _safe_float(fh_result.get("mean_composite")),
            "n_days_scored": fh_result.get("n_days_scored", 0),
            "tier_means": {
                k: _safe_float(v)
                for k, v in (fh_result.get("tier_means") or {}).items()
            },
            "columns_added": fh_result.get("columns_added", []),
        }

    # Enrich with time-series summary from cache columns if available
    if cache is not None and not cache.empty:
        for col in (
            "fh_liquidity_score", "fh_solvency_score", "fh_stability_score",
            "fh_profitability_score", "fh_growth_score", "fh_composite_score",
        ):
            if col in cache.columns:
                section.setdefault("series_summary", {})[col] = _series_summary(cache[col])
                if not section.get("available"):
                    section["available"] = True

        # Composite label distribution
        if "fh_composite_label" in cache.columns:
            vc = cache["fh_composite_label"].value_counts(normalize=True)
            section["label_distribution_pct"] = {
                str(k): round(float(v), 4) for k, v in vc.items()
            }

    return section


# ---------------------------------------------------------------------------
# Sentiment section
# ---------------------------------------------------------------------------


def _build_sentiment_section(
    cache: pd.DataFrame | None,
    sentiment_result: dict[str, Any] | None,
) -> dict[str, Any]:
    """Build news sentiment section."""
    section: dict[str, Any] = {"available": False}

    if sentiment_result:
        section = {
            "available": True,
            "n_articles_scored": sentiment_result.get("n_articles_scored", 0),
            "scoring_method": sentiment_result.get("scoring_method", "none"),
            "mean_sentiment": _safe_float(sentiment_result.get("mean_sentiment")),
            "latest_sentiment": _safe_float(sentiment_result.get("latest_sentiment")),
            "latest_label": sentiment_result.get("latest_label", "Unknown"),
        }

    if cache is not None and "sentiment_score" in cache.columns:
        section.setdefault("series_summary", {})["sentiment_score"] = _series_summary(cache["sentiment_score"])
        if not section.get("available"):
            section["available"] = True

    return section


# ---------------------------------------------------------------------------
# Peer ranking section
# ---------------------------------------------------------------------------


def _build_peer_ranking_section(
    cache: pd.DataFrame | None,
    peer_result: dict[str, Any] | None,
) -> dict[str, Any]:
    """Build peer percentile ranking section."""
    section: dict[str, Any] = {"available": False}

    if peer_result:
        section = {
            "available": True,
            "n_peers": peer_result.get("n_peers", 0),
            "n_variables_ranked": peer_result.get("n_variables_ranked", 0),
            "latest_composite_rank": _safe_float(peer_result.get("latest_composite_rank")),
            "latest_label": peer_result.get("latest_label", "Unknown"),
            "variable_ranks": {
                k: _safe_float(v)
                for k, v in (peer_result.get("variable_ranks") or {}).items()
            },
        }

    if cache is not None and "peer_rank_composite" in cache.columns:
        section.setdefault("series_summary", {})["peer_rank_composite"] = _series_summary(
            cache["peer_rank_composite"]
        )
        if not section.get("available"):
            section["available"] = True

    return section


# ---------------------------------------------------------------------------
# Macro quadrant section
# ---------------------------------------------------------------------------


def _build_macro_quadrant_section(
    cache: pd.DataFrame | None,
    macro_result: dict[str, Any] | None,
) -> dict[str, Any]:
    """Build macro quadrant classification section."""
    section: dict[str, Any] = {"available": False}

    if macro_result:
        section = {
            "available": True,
            "current_quadrant": macro_result.get("current_quadrant", "unknown"),
            "quadrant_distribution": macro_result.get("quadrant_distribution", {}),
            "n_transitions": macro_result.get("n_transitions", 0),
            "growth_trend": _safe_float(macro_result.get("growth_trend")),
            "inflation_target": _safe_float(macro_result.get("inflation_target")),
            "n_days_classified": macro_result.get("n_days_classified", 0),
        }

    if cache is not None and "macro_quadrant" in cache.columns:
        vc = cache["macro_quadrant"].value_counts(normalize=True)
        section["label_distribution_pct"] = {
            str(k): round(float(v), 4) for k, v in vc.items()
        }
        if not section.get("available"):
            section["available"] = True

    return section


# ---------------------------------------------------------------------------
# Main entry point
# ---------------------------------------------------------------------------


def build_company_profile(
    *,
    verified_target: dict[str, Any] | None = None,
    cache: pd.DataFrame | None = None,
    linked_aggregates: pd.DataFrame | None = None,
    regime_result: dict[str, Any] | None = None,
    forecast_result: dict[str, Any] | None = None,
    mc_result: dict[str, Any] | None = None,
    prediction_result: dict[str, Any] | None = None,
    quality_report_path: str | Path | None = None,
    estimation_coverage_path: str | Path | None = None,
    output_path: str | Path | None = None,
    # Advanced modules
    graph_risk_result: dict[str, Any] | None = None,
    game_theory_result: dict[str, Any] | None = None,
    fuzzy_protection_result: dict[str, Any] | None = None,
    pid_summary: dict[str, Any] | None = None,
    financial_health_result: dict[str, Any] | None = None,
    sentiment_result: dict[str, Any] | None = None,
    peer_ranking_result: dict[str, Any] | None = None,
    macro_quadrant_result: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Build the comprehensive company profile JSON.

    All parameters are optional -- missing sections are included with
    ``"available": False`` so the report generator can note gaps.

    Parameters
    ----------
    verified_target:
        Dict (or VerifiedTarget dataclass) from T2.1.
    cache:
        Full feature table (target daily cache with all derived
        variables, survival flags, regime labels, etc.).
    linked_aggregates:
        Linked entity aggregate DataFrame from T3.4.
    regime_result:
        Dict (or RegimeResult dataclass) from T6.1.
    forecast_result:
        Dict (or ForecastResult dataclass) from T6.2.
    mc_result:
        Dict (or MonteCarloResult dataclass) from T6.3.
    prediction_result:
        Dict (or PredictionAggregatorResult dataclass) from T6.4.
    quality_report_path:
        Path to ``data_quality_report.json`` (default: ``cache/``).
    estimation_coverage_path:
        Path to ``estimation_coverage.json`` (default: ``cache/``).
    output_path:
        Where to write ``company_profile.json``.  Defaults to
        ``cache/company_profile.json``.

    Returns
    -------
    dict
        The complete company profile as a Python dict.
        Also persisted to *output_path* as JSON.
    """
    logger.info("Building company profile JSON...")

    # Convert dataclass results to dicts if needed
    for name, obj in [
        ("regime_result", regime_result),
        ("forecast_result", forecast_result),
        ("mc_result", mc_result),
        ("prediction_result", prediction_result),
    ]:
        if obj is not None and hasattr(obj, "__dataclass_fields__"):
            try:
                converted = asdict(obj)
            except Exception:
                converted = obj.__dict__.copy() if hasattr(obj, "__dict__") else {}
            if name == "regime_result":
                regime_result = converted
            elif name == "forecast_result":
                forecast_result = converted
            elif name == "mc_result":
                mc_result = converted
            elif name == "prediction_result":
                prediction_result = converted

    # Assemble profile
    profile: dict[str, Any] = {
        "meta": {
            "generated_at": datetime.utcnow().isoformat() + "Z",
            "pipeline_version": "1.0.0",
            "date_range": {
                "start": DATE_START.isoformat(),
                "end": DATE_END.isoformat(),
            },
        },
        "identity": _build_identity_section(verified_target),
        "current_state": _build_current_state_section(cache),
        "historical": _build_historical_section(cache),
        "survival": _build_survival_section(cache),
        "survival_episodes": _build_survival_episodes(cache),
        "vanity": _build_vanity_section(cache),
        "linked_entities": _build_linked_section(linked_aggregates),
        "regimes": _build_regime_section(cache, regime_result),
        "predictions": _build_predictions_section(prediction_result),
        "monte_carlo": _build_monte_carlo_section(mc_result),
        "model_metrics": _build_model_metrics_section(forecast_result),
        "filters": _build_ethical_filters_section(cache),
        "graph_risk": graph_risk_result if graph_risk_result else {"available": False},
        "game_theory": game_theory_result if game_theory_result else {"available": False},
        "fuzzy_protection": fuzzy_protection_result if fuzzy_protection_result else {"available": False},
        "pid_controller": pid_summary if pid_summary else {"available": False},
        "financial_health": _build_financial_health_section(cache, financial_health_result),
        "sentiment": _build_sentiment_section(cache, sentiment_result),
        "peer_ranking": _build_peer_ranking_section(cache, peer_ranking_result),
        "macro_quadrant": _build_macro_quadrant_section(cache, macro_quadrant_result),
        "data_quality": _build_data_quality_section(quality_report_path),
        "estimation": _build_estimation_section(estimation_coverage_path),
        "failed_modules": _build_failed_modules_section(
            regime_result=regime_result,
            forecast_result=forecast_result,
            mc_result=mc_result,
            prediction_result=prediction_result,
        ),
    }

    # ------------------------------------------------------------------
    # Accounting standard caveat (cross-standard comparison warning)
    # ------------------------------------------------------------------
    try:
        from operator1.clients.canonical_translator import MARKET_ACCOUNTING_STANDARD
        market_id = ""
        if verified_target:
            market_id = verified_target.get("market_id", "") if isinstance(verified_target, dict) else getattr(verified_target, "market_id", "")
        accounting_standard = MARKET_ACCOUNTING_STANDARD.get(market_id, "Unknown")
        profile["meta"]["accounting_standard"] = accounting_standard
        if accounting_standard not in ("US-GAAP", "IFRS", "Unknown"):
            profile["meta"]["cross_standard_caveat"] = (
                f"This company reports under {accounting_standard}. "
                f"Cross-market comparisons with US-GAAP or IFRS companies "
                f"should note that similarly named metrics (e.g., EBIT, "
                f"operating income) may be defined differently across standards."
            )
    except Exception:
        pass

    # ------------------------------------------------------------------
    # Estimation transparency: mark which fields are estimated vs observed
    # ------------------------------------------------------------------
    if cache is not None and not cache.empty:
        estimated_fields: list[str] = []
        observed_fields: list[str] = []
        for col in cache.columns:
            missing_flag = f"is_missing_{col}"
            if missing_flag in cache.columns:
                pct_missing = float(cache[missing_flag].mean())
                if pct_missing > 0.5:
                    estimated_fields.append(col)
                else:
                    observed_fields.append(col)
        if estimated_fields:
            profile.setdefault("data_quality", {})
            profile["data_quality"]["estimated_fields"] = estimated_fields[:30]
            profile["data_quality"]["n_estimated"] = len(estimated_fields)
            profile["data_quality"]["n_observed"] = len(observed_fields)
            profile["data_quality"]["estimation_note"] = (
                f"{len(estimated_fields)} fields have >50% estimated values. "
                f"These are marked with [E] in the report where applicable."
            )

    # ------------------------------------------------------------------
    # Promote extended_models sub-keys to profile root so the report
    # generator can find them under the keys it already expects.
    # ------------------------------------------------------------------
    ext = profile.get("extended_models", {})
    if ext.get("conformal_prediction"):
        profile.setdefault("conformal_intervals", ext["conformal_prediction"])
    if ext.get("shap_explanations"):
        profile.setdefault("shap_explanations", ext["shap_explanations"])
    if ext.get("dtw_analogs"):
        profile.setdefault("historical_analogs", ext["dtw_analogs"])
    if ext.get("candlestick_patterns"):
        profile.setdefault("patterns", ext["candlestick_patterns"])
    if ext.get("cycle_decomposition"):
        profile.setdefault("cycle_decomposition", ext["cycle_decomposition"])
    if ext.get("sobol_sensitivity"):
        profile.setdefault("sobol_sensitivity", ext["sobol_sensitivity"])
    if ext.get("particle_filter"):
        profile.setdefault("particle_filter", ext["particle_filter"])
    if ext.get("transformer"):
        profile.setdefault("transformer", ext["transformer"])
    if ext.get("copula"):
        profile.setdefault("copula", ext["copula"])
    if ext.get("transfer_entropy"):
        profile.setdefault("transfer_entropy", ext["transfer_entropy"])
    if ext.get("granger_causality"):
        profile.setdefault("granger_causality", ext["granger_causality"])
    if ext.get("dual_regimes"):
        profile.setdefault("dual_regimes", ext["dual_regimes"])
    if ext.get("genetic_optimizer"):
        profile.setdefault("genetic_optimizer", ext["genetic_optimizer"])

    # Make JSON-safe
    profile = _json_serialisable(profile)

    # Persist to disk
    if output_path is None:
        output_path = Path(CACHE_DIR) / "company_profile.json"
    out = Path(output_path)
    out.parent.mkdir(parents=True, exist_ok=True)

    with open(out, "w", encoding="utf-8") as fh:
        json.dump(profile, fh, indent=2, default=str)

    logger.info("Company profile written to %s", out)
    return profile
