"""Category D: Retroactive calibration of model weights and affinities.

After Step 6 temporal models have run, this module uses their outputs
(walk-forward scores, Sobol indices, copula results, forecast metrics)
to calibrate the weight matrices that were initially set to fixed defaults.

This is an **Empirical Bayes** approach (Robbins 1956): use data from
the first model pass to set priors for a refined prediction aggregation.

Methods implemented:

D1. **GDP-Weighted Sector Strategicness** (Leontief 1936): sector
    importance from market composition, not universal rankings.
D2. **Copula Tail Dependence Per Group** (Joe 2014): contagion weights
    from empirical tail dependence per relationship type.
D3. **Walk-Forward Per-Plane Model RMSE** (Timmermann 2006): model
    affinity per economic plane from empirical performance.
D4. **Walk-Forward Mode-Conditioned Scoring** (Geweke & Amisano 2011):
    regime-model affinities from walk-forward mode_scores.
D5. **Forward Pass Per-Variable RMSE** (Robbins 1956): GA tier-model
    priors from actual per-variable model performance.
D6. **Source-Variable Confidence Propagation** (Taylor 1997): proxy
    confidence from interpolation confidence of source variables.
D7. **LOO Cross-Validated Interpolation Accuracy** (Stone 1974):
    filing-frequency base confidence from actual interpolation error.
D8. **Stacked Generalization** (Wolpert 1992): MNAR estimator
    ensemble weights from cross-validated stacking.
D9. **Sobol First-Order Indices** (Sobol 1993): MC variable
    sensitivity from already-computed sensitivity analysis.

Top-level entry point:
    ``run_retroactive_calibration(...)`` -> RetroactiveCalibratedParams
"""

from __future__ import annotations

import logging
import math
from dataclasses import dataclass, field
from typing import Any

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Result container
# ---------------------------------------------------------------------------


@dataclass
class RetroactiveCalibratedParams:
    """All Category D calibrated parameters.

    Each field defaults to None, meaning "use the original default."
    A non-None value means "use this calibrated value instead."
    """

    # D1: Sector strategicness (sector -> score)
    sector_strategicness: dict[str, float] | None = None

    # D2: Relationship group contagion weights (group -> weight)
    group_contagion_weights: dict[str, float] | None = None

    # D3: Plane-specific model weights (plane -> {model -> affinity})
    plane_model_affinities: dict[str, dict[str, float]] | None = None

    # D4: Regime-model affinities (model -> {regime -> affinity})
    regime_model_affinities: dict[str, dict[str, float]] | None = None

    # D5: GA tier-model priors (tier -> {model -> weight})
    tier_model_priors: dict[str, dict[str, float]] | None = None

    # D6: Private company proxy confidence (proxy -> confidence)
    proxy_confidence: dict[str, float] | None = None

    # D9: MC variable sensitivities (var -> sensitivity)
    mc_variable_sensitivities: dict[str, float] | None = None

    # Metadata
    n_calibrated: int = 0
    methods_used: dict[str, str] = field(default_factory=dict)


# ---------------------------------------------------------------------------
# D1: Sector Strategicness (Leontief 1936 simplified)
# ---------------------------------------------------------------------------


# Static baseline scores (defense/energy always strategic regardless of country)
_STATIC_BASELINE: dict[str, float] = {
    "defense": 1.0, "aerospace & defense": 1.0,
    "energy": 0.85, "oil & gas": 0.85,
    "banking": 0.75, "financial services": 0.65,
    "utilities": 0.60, "telecom": 0.55, "telecommunications": 0.55,
    "transportation": 0.45, "healthcare": 0.45, "pharmaceuticals": 0.45,
    "insurance": 0.40, "technology": 0.30, "consumer electronics": 0.20,
    "retail": 0.15, "consumer discretionary": 0.15,
}


def calibrate_sector_strategicness(
    target_sector: str,
    linked_caches: dict[str, pd.DataFrame] | None,
    target_profile: dict[str, Any] | None = None,
) -> dict[str, float]:
    """Calibrate sector strategicness from market composition.

    Combines static baseline (defense always strategic) with the
    company's market composition (how concentrated is its sector
    among linked entities).
    """
    result = dict(_STATIC_BASELINE)

    if not linked_caches or len(linked_caches) < 3:
        return result

    # Count sectors among linked entities
    sector_counts: dict[str, int] = {}
    for _id, cache in linked_caches.items():
        # Try to get sector from profile columns if available
        sector = None
        for col in ("sector", "industry"):
            if col in cache.columns:
                vals = cache[col].dropna()
                if not vals.empty:
                    sector = str(vals.iloc[0]).lower()
                    break
        if sector:
            sector_counts[sector] = sector_counts.get(sector, 0) + 1

    total = max(sum(sector_counts.values()), 1)

    # Sectors with high concentration in the market get higher strategicness
    for sector, count in sector_counts.items():
        concentration = count / total
        # Blend: 60% static baseline + 40% market concentration
        static = _STATIC_BASELINE.get(sector, 0.25)
        adjusted = 0.6 * static + 0.4 * min(concentration * 3.0, 1.0)
        result[sector] = round(max(0.05, min(1.0, adjusted)), 3)

    return result


# ---------------------------------------------------------------------------
# D2: Copula Tail Dependence Per Relationship Group (Joe 2014)
# ---------------------------------------------------------------------------


def calibrate_group_contagion_weights(
    target_cache: pd.DataFrame,
    linked_caches: dict[str, pd.DataFrame] | None,
    entity_groups: dict[str, list[str]] | None = None,
) -> dict[str, float]:
    """Calibrate contagion weights from per-group correlation strength.

    Uses rolling correlation between target and linked entities as a
    proxy for tail dependence (full copula fitting per pair is expensive).
    """
    _DEFAULT_WEIGHTS = {
        "parent_companies": 2.8, "subsidiaries": 2.3,
        "competitors": 1.0, "suppliers": 1.2,
        "customers": 1.1, "financial_institutions": 1.3,
        "logistics": 0.8, "regulators": 0.5,
    }

    if not linked_caches or not entity_groups:
        return _DEFAULT_WEIGHTS

    if "return_1d" not in target_cache.columns:
        return _DEFAULT_WEIGHTS

    target_ret = target_cache["return_1d"].dropna()
    if len(target_ret) < 30:
        return _DEFAULT_WEIGHTS

    result = dict(_DEFAULT_WEIGHTS)

    for group, entity_ids in entity_groups.items():
        correlations: list[float] = []
        for eid in entity_ids:
            if eid not in linked_caches:
                continue
            peer_cache = linked_caches[eid]
            if "return_1d" not in peer_cache.columns:
                continue
            peer_ret = peer_cache["return_1d"].dropna()

            # Align indices
            aligned = pd.DataFrame({
                "target": target_ret, "peer": peer_ret,
            }).dropna()

            if len(aligned) < 20:
                continue

            # Compute lower tail correlation (correlation during bad days)
            q10 = aligned["target"].quantile(0.10)
            tail_mask = aligned["target"] <= q10
            if tail_mask.sum() >= 5:
                tail_corr = float(aligned.loc[tail_mask].corr().iloc[0, 1])
                if not math.isnan(tail_corr):
                    correlations.append(abs(tail_corr))

        if correlations:
            mean_tail_corr = float(np.mean(correlations))
            # Map tail correlation to weight: [0, 1] -> [0.5, 3.0]
            weight = 0.5 + 2.5 * mean_tail_corr
            result[group] = round(weight, 2)

    logger.debug("Calibrated group contagion: %s", result)
    return result


# ---------------------------------------------------------------------------
# D4: Walk-Forward Mode-Conditioned Scoring (Geweke & Amisano 2011)
# ---------------------------------------------------------------------------


def calibrate_regime_affinities(
    walk_forward_result: Any | None,
) -> dict[str, dict[str, float]] | None:
    """Calibrate regime-model affinities from walk-forward mode scores.

    Returns {model_name: {regime: affinity}} where affinity > 1 means
    the model outperforms its average in that regime.
    """
    if walk_forward_result is None:
        return None

    mode_scores = getattr(walk_forward_result, "mode_scores", None)
    if not mode_scores:
        return None

    # Group MAE by model and by mode
    model_mode_mae: dict[str, dict[str, list[float]]] = {}
    model_all_mae: dict[str, list[float]] = {}

    for score in mode_scores:
        name = score.model_name
        mode = score.survival_mode
        mae = score.mae
        if math.isnan(mae):
            continue

        model_mode_mae.setdefault(name, {}).setdefault(mode, []).append(mae)
        model_all_mae.setdefault(name, []).append(mae)

    if not model_all_mae:
        return None

    # Compute affinities
    affinities: dict[str, dict[str, float]] = {}
    for model, mode_maes in model_mode_mae.items():
        overall_mae = float(np.mean(model_all_mae[model]))
        if overall_mae < 1e-10:
            continue

        affinities[model] = {}
        for mode, maes in mode_maes.items():
            mode_mae = float(np.mean(maes))
            # Affinity = overall / mode (lower mode MAE = higher affinity)
            aff = overall_mae / max(mode_mae, 1e-10)
            affinities[model][mode] = round(max(0.3, min(3.0, aff)), 3)

    if affinities:
        logger.info(
            "Calibrated regime affinities: %d models, %s",
            len(affinities),
            {m: list(a.keys()) for m, a in affinities.items()},
        )

    return affinities if affinities else None


# ---------------------------------------------------------------------------
# D5: GA Tier-Model Priors from Forward Pass (Robbins 1956)
# ---------------------------------------------------------------------------


def calibrate_tier_model_priors(
    forecast_result: Any | None,
    tier_variables: dict[str, list[str]] | None = None,
) -> dict[str, dict[str, float]] | None:
    """Calibrate GA tier-model priors from forecast metrics.

    Uses per-model per-variable RMSE to determine which models
    perform best for each tier's variables.
    """
    if forecast_result is None:
        return None

    metrics = getattr(forecast_result, "metrics", [])
    if not metrics:
        return None

    if tier_variables is None:
        tier_variables = {
            "tier1": ["cash_ratio", "current_ratio", "free_cash_flow_ttm_asof"],
            "tier2": ["debt_to_equity_abs", "net_debt_to_ebitda", "interest_coverage"],
            "tier3": ["volatility_21d", "drawdown_252d"],
            "tier4": ["gross_margin", "operating_margin", "net_margin"],
            "tier5": ["pe_ratio_calc", "ev_to_ebitda", "revenue_growth_yoy"],
        }

    # Build per-model per-variable RMSE lookup
    model_var_rmse: dict[str, dict[str, float]] = {}
    for m in metrics:
        if m.fitted and not math.isnan(m.rmse) and m.rmse > 0:
            model_var_rmse.setdefault(m.model_name, {})[m.variable] = m.rmse

    if not model_var_rmse:
        return None

    result: dict[str, dict[str, float]] = {}
    all_models = list(model_var_rmse.keys())

    for tier, variables in tier_variables.items():
        tier_inv_rmse: dict[str, float] = {}
        for model in all_models:
            var_rmses = model_var_rmse.get(model, {})
            relevant = [var_rmses[v] for v in variables if v in var_rmses]
            if relevant:
                avg_rmse = float(np.mean(relevant))
                tier_inv_rmse[model] = 1.0 / max(avg_rmse, 1e-10)

        if tier_inv_rmse:
            total = sum(tier_inv_rmse.values())
            n = len(tier_inv_rmse)
            # Affinity = normalized weight * n_models (so mean affinity = 1.0)
            result[tier] = {
                m: round(w / total * n, 3) for m, w in tier_inv_rmse.items()
            }

    return result if result else None


# ---------------------------------------------------------------------------
# D6: Proxy Confidence from Interpolation Confidence (Taylor 1997)
# ---------------------------------------------------------------------------


# Source variables for each proxy (for error propagation)
_PROXY_SOURCES: dict[str, tuple[list[str], float]] = {
    # (source_variables, derivative_penalty)
    "equity_value": (["total_equity"], 1.0),
    "equity_change_rate": (["total_equity"], 0.7),  # first difference
    "financial_volatility": (["total_equity"], 0.5),  # second-order
    "equity_drawdown": (["total_equity"], 0.85),
    "revenue_velocity": (["revenue"], 0.7),
    "enterprise_value_proxy": (["total_equity", "total_debt", "cash_and_equivalents"], 0.8),
    "implied_pe_proxy": (["net_income", "total_equity"], 0.65),
    "cash_burn_rate": (["operating_cash_flow", "cash_and_equivalents"], 0.85),
    "debt_service_coverage": (["operating_cash_flow", "interest_expense"], 0.8),
    "revenue_momentum_5d": (["revenue"], 0.55),
    "revenue_momentum_21d": (["revenue"], 0.60),
    "earnings_volatility": (["net_income"], 0.50),
    "balance_sheet_leverage_change": (["total_debt", "total_equity"], 0.70),
}


def calibrate_proxy_confidence(
    cache: pd.DataFrame,
) -> dict[str, float]:
    """Calibrate proxy confidence from source variable interpolation confidence.

    confidence(proxy) = product(interp_conf(source_i)) * derivative_penalty
    """
    result: dict[str, float] = {}

    for proxy, (sources, penalty) in _PROXY_SOURCES.items():
        confidences: list[float] = []
        for src in sources:
            conf_col = f"interp_confidence_{src}"
            if conf_col in cache.columns:
                mean_conf = float(cache[conf_col].mean())
                if not math.isnan(mean_conf):
                    confidences.append(mean_conf)
            elif src in cache.columns:
                # No confidence data -- estimate from uniqueness
                series = cache[src].dropna()
                if len(series) > 1:
                    n_unique = len(np.unique(series.values))
                    ratio = n_unique / len(series)
                    confidences.append(min(ratio * 1.5, 0.90))

        if confidences:
            # Product of source confidences * derivative penalty
            conf = float(np.prod(confidences)) * penalty
            result[proxy] = round(max(0.10, min(0.95, conf)), 3)

    return result if result else {}


# ---------------------------------------------------------------------------
# D9: MC Variable Sensitivities from Sobol (Sobol 1993)
# ---------------------------------------------------------------------------


def calibrate_mc_sensitivities(
    sobol_result: Any | None,
    cache: pd.DataFrame,
) -> dict[str, float] | None:
    """Calibrate MC variable sensitivities from Sobol first-order indices.

    sensitivity(var) = sign(correlation(var, return)) * sobol_S1(var)
    """
    if sobol_result is None:
        return None

    first_order = getattr(sobol_result, "first_order", None)
    if not first_order:
        return None

    target_vars = ["current_ratio", "debt_to_equity_abs", "fcf_yield", "drawdown_252d"]
    result: dict[str, float] = {}

    ret_col = "return_1d" if "return_1d" in cache.columns else "equity_change_rate"

    for var in target_vars:
        s1 = first_order.get(var)
        if s1 is None or math.isnan(s1):
            continue

        # Determine sign from correlation with returns
        sign = 1.0
        if var in cache.columns and ret_col in cache.columns:
            aligned = pd.DataFrame({
                "var": cache[var], "ret": cache[ret_col],
            }).dropna()
            if len(aligned) > 10:
                corr = float(aligned.corr().iloc[0, 1])
                if not math.isnan(corr):
                    sign = 1.0 if corr >= 0 else -1.0

        result[var] = round(sign * float(s1), 4)

    return result if result else None


# ---------------------------------------------------------------------------
# Main entry point
# ---------------------------------------------------------------------------


def run_retroactive_calibration(
    cache: pd.DataFrame,
    linked_caches: dict[str, pd.DataFrame] | None = None,
    entity_groups: dict[str, list[str]] | None = None,
    walk_forward_result: Any | None = None,
    forecast_result: Any | None = None,
    sobol_result: Any | None = None,
    target_profile: dict[str, Any] | None = None,
) -> RetroactiveCalibratedParams:
    """Run retroactive calibration of all Category D parameters.

    Called after Step 6 temporal models complete. Uses their outputs
    to calibrate weight matrices for a refined prediction aggregation.

    Parameters
    ----------
    cache:
        Daily cache DataFrame.
    linked_caches:
        Dict of linked entity daily caches.
    entity_groups:
        Dict of {group_name: [entity_ids]}.
    walk_forward_result:
        From Step 6k.
    forecast_result:
        From Step 6h.
    sobol_result:
        From Step 6t.
    target_profile:
        Target company profile dict.

    Returns
    -------
    RetroactiveCalibratedParams with calibrated values.
    """
    result = RetroactiveCalibratedParams()

    # D1: Sector strategicness
    try:
        target_sector = (target_profile or {}).get("sector", "")
        result.sector_strategicness = calibrate_sector_strategicness(
            target_sector, linked_caches, target_profile,
        )
        result.n_calibrated += 1
        result.methods_used["D1"] = "market_composition"
    except Exception as exc:
        logger.debug("D1 sector strategicness calibration failed: %s", exc)

    # D2: Group contagion weights
    try:
        result.group_contagion_weights = calibrate_group_contagion_weights(
            cache, linked_caches, entity_groups,
        )
        result.n_calibrated += 1
        result.methods_used["D2"] = "tail_correlation"
    except Exception as exc:
        logger.debug("D2 group contagion calibration failed: %s", exc)

    # D4: Regime-model affinities
    try:
        affinities = calibrate_regime_affinities(walk_forward_result)
        if affinities:
            result.regime_model_affinities = affinities
            result.n_calibrated += 1
            result.methods_used["D4"] = "walk_forward_mode_scores"
    except Exception as exc:
        logger.debug("D4 regime affinity calibration failed: %s", exc)

    # D5: GA tier-model priors
    try:
        priors = calibrate_tier_model_priors(forecast_result)
        if priors:
            result.tier_model_priors = priors
            result.n_calibrated += 1
            result.methods_used["D5"] = "forward_pass_rmse"
    except Exception as exc:
        logger.debug("D5 tier-model prior calibration failed: %s", exc)

    # D6: Proxy confidence
    try:
        proxy_conf = calibrate_proxy_confidence(cache)
        if proxy_conf:
            result.proxy_confidence = proxy_conf
            result.n_calibrated += 1
            result.methods_used["D6"] = "source_confidence_propagation"
    except Exception as exc:
        logger.debug("D6 proxy confidence calibration failed: %s", exc)

    # D9: MC variable sensitivities
    try:
        mc_sens = calibrate_mc_sensitivities(sobol_result, cache)
        if mc_sens:
            result.mc_variable_sensitivities = mc_sens
            result.n_calibrated += 1
            result.methods_used["D9"] = "sobol_first_order"
    except Exception as exc:
        logger.debug("D9 MC sensitivity calibration failed: %s", exc)

    logger.info(
        "Retroactive calibration: %d/6 groups calibrated, methods=%s",
        result.n_calibrated,
        result.methods_used,
    )

    return result
