"""3-layer feature selection: Boruta + Regime-Conditional PIMP + mRMR.

Replaces the Granger/PCMCI feature pruning that was killing all Gap 1-4
features due to linear-only testing and wrong variable pool.

Layer 1 (Boruta): Shadow feature comparison via Random Forest.
Layer 2 (PIMP): Permutation Importance with null distribution, per regime.
Layer 3 (mRMR): Minimum Redundancy Maximum Relevance via mutual information.

All implementations use existing sklearn/xgboost/numpy -- zero new deps.

References:
  - BorutaPy (scikit-learn-contrib/boruta_py)
  - Altmann 2010 (PIMP null distribution)
  - smazzanti/mrmr (greedy mRMR)
  - mlfinlab (Clustered Feature Importance)
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Any

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


@dataclass
class FeatureSelectionResult:
    """Container for 3-layer feature selection outputs."""

    # Layer 1: Boruta
    boruta_confirmed: list[str] = field(default_factory=list)
    boruta_tentative: list[str] = field(default_factory=list)
    boruta_rejected: list[str] = field(default_factory=list)
    boruta_n_iterations: int = 0

    # Layer 2: Regime-Conditional PIMP
    regime_importances: dict[str, dict[str, float]] = field(default_factory=dict)
    regime_selected: dict[str, list[str]] = field(default_factory=dict)
    pimp_p_values: dict[str, float] = field(default_factory=dict)

    # Layer 3: mRMR
    mrmr_selected: list[str] = field(default_factory=list)
    mrmr_scores: dict[str, float] = field(default_factory=dict)

    # Combined output
    final_selected: list[str] = field(default_factory=list)
    n_input: int = 0
    n_output: int = 0
    method_contributions: dict[str, int] = field(default_factory=dict)

    fitted: bool = False
    error: str | None = None


# ---------------------------------------------------------------------------
# Layer 1: Boruta (shadow feature comparison)
# ---------------------------------------------------------------------------

def select_features_boruta(
    cache: pd.DataFrame,
    target_col: str,
    candidates: list[str],
    *,
    n_trials: int = 50,
    alpha: float = 0.05,
    rf_max_depth: int = 7,
    rf_n_estimators: int = 100,
) -> tuple[list[str], list[str], list[str]]:
    """Layer 1: Boruta shadow feature comparison.

    Returns (confirmed, tentative, rejected) feature lists.
    """
    from sklearn.ensemble import RandomForestRegressor

    # Filter to valid candidates
    valid = [c for c in candidates if c in cache.columns and cache[c].notna().sum() > 10]
    if len(valid) < 2 or target_col not in cache.columns:
        return valid, [], []

    # Prepare data
    df = cache[valid + [target_col]].dropna()
    if len(df) < 30:
        return valid, [], []

    X = df[valid].values
    y = df[target_col].values
    n_features = X.shape[1]

    # Track hits: how many times each feature beats the best shadow
    hits = np.zeros(n_features, dtype=int)

    for trial in range(n_trials):
        # Create shadow features (permute each column independently)
        X_shadow = np.apply_along_axis(np.random.permutation, 0, X)
        X_combined = np.hstack([X, X_shadow])

        # Train RF on combined real + shadow
        rf = RandomForestRegressor(
            n_estimators=rf_n_estimators,
            max_depth=rf_max_depth,
            random_state=trial,
            n_jobs=-1,
        )
        rf.fit(X_combined, y)

        importances = rf.feature_importances_
        real_imp = importances[:n_features]
        shadow_imp = importances[n_features:]
        shadow_max = shadow_imp.max()

        # Count hits: real feature importance > best shadow
        hits[real_imp > shadow_max] += 1

    # Binomial test with Bonferroni correction
    from scipy.stats import binom

    confirmed = []
    tentative = []
    rejected = []
    alpha_corrected = alpha / max(n_features, 1)

    for i, feat in enumerate(valid):
        p_value = 1 - binom.cdf(hits[i] - 1, n_trials, 0.5)
        if p_value < alpha_corrected:
            confirmed.append(feat)
        elif p_value < alpha_corrected * 5:  # lenient threshold for tentative
            tentative.append(feat)
        else:
            rejected.append(feat)

    logger.info(
        "Boruta: %d confirmed, %d tentative, %d rejected (from %d candidates, %d trials)",
        len(confirmed), len(tentative), len(rejected), n_features, n_trials,
    )
    return confirmed, tentative, rejected


# ---------------------------------------------------------------------------
# Layer 2: Regime-Conditional PIMP (Permutation Importance with P-values)
# ---------------------------------------------------------------------------

def select_features_regime_importance(
    cache: pd.DataFrame,
    target_col: str,
    candidates: list[str],
    regime_labels: pd.Series | None = None,
    *,
    n_permutations: int = 50,
    significance: float = 0.05,
    min_regime_samples: int = 30,
) -> tuple[list[str], dict[str, dict[str, float]], dict[str, float]]:
    """Layer 2: Regime-conditional permutation importance with null distribution.

    Returns (selected_features, per_regime_importances, pimp_p_values).
    """
    valid = [c for c in candidates if c in cache.columns and cache[c].notna().sum() > 10]
    if len(valid) < 2 or target_col not in cache.columns:
        return valid, {}, {}

    try:
        from xgboost import XGBRegressor
        model_cls = XGBRegressor
        model_kwargs = {"n_estimators": 50, "max_depth": 4, "verbosity": 0, "n_jobs": -1}
    except ImportError:
        from sklearn.ensemble import GradientBoostingRegressor
        model_cls = GradientBoostingRegressor
        model_kwargs = {"n_estimators": 50, "max_depth": 4}

    # Determine regimes
    if regime_labels is not None and regime_labels.notna().sum() > 0:
        regimes = regime_labels.dropna().unique()
    else:
        regimes = ["all"]

    all_selected = set()
    regime_importances: dict[str, dict[str, float]] = {}
    global_p_values: dict[str, float] = {f: 1.0 for f in valid}

    for regime in regimes:
        if regime == "all":
            mask = cache.index
        else:
            if regime_labels is None:
                continue
            mask = cache.index[regime_labels == regime]

        segment = cache.loc[mask, valid + [target_col]].dropna()
        if len(segment) < min_regime_samples:
            continue

        X = segment[valid].values
        y = segment[target_col].values

        # Train model on this regime
        model = model_cls(**model_kwargs)
        model.fit(X, y)
        real_imp = model.feature_importances_

        # PIMP: build null distribution by permuting target
        null_importances = np.zeros((n_permutations, len(valid)))
        for p in range(n_permutations):
            y_perm = np.random.permutation(y)
            model_perm = model_cls(**model_kwargs)
            model_perm.fit(X, y_perm)
            null_importances[p] = model_perm.feature_importances_

        # Compute p-values
        regime_imp = {}
        for i, feat in enumerate(valid):
            p_value = float(np.mean(null_importances[:, i] >= real_imp[i]))
            regime_imp[feat] = float(real_imp[i])
            # Keep minimum p-value across regimes
            global_p_values[feat] = min(global_p_values[feat], p_value)
            if p_value < significance:
                all_selected.add(feat)

        regime_key = str(regime)
        regime_importances[regime_key] = regime_imp

    selected = sorted(all_selected)
    logger.info(
        "PIMP: %d features selected across %d regimes (%d permutations each)",
        len(selected), len(regimes), n_permutations,
    )
    return selected, regime_importances, global_p_values


# ---------------------------------------------------------------------------
# Layer 3: mRMR (Minimum Redundancy Maximum Relevance)
# ---------------------------------------------------------------------------

def select_features_mrmr(
    cache: pd.DataFrame,
    target_col: str,
    candidates: list[str],
    *,
    K: int = 30,
) -> tuple[list[str], dict[str, float]]:
    """Layer 3: Greedy mRMR selection via mutual information.

    Returns (selected_ordered, mrmr_scores).
    """
    from sklearn.feature_selection import mutual_info_regression

    valid = [c for c in candidates if c in cache.columns and cache[c].notna().sum() > 10]
    if len(valid) < 2 or target_col not in cache.columns:
        return valid[:K], {}

    df = cache[valid + [target_col]].dropna()
    if len(df) < 30:
        return valid[:K], {}

    X = df[valid].values
    y = df[target_col].values
    K = min(K, len(valid))

    # Compute relevance: MI(feature, target) for each feature
    relevance = mutual_info_regression(X, y, random_state=42)
    rel_dict = {valid[i]: float(relevance[i]) for i in range(len(valid))}

    # Greedy forward selection
    selected: list[str] = []
    remaining = list(range(len(valid)))
    scores: dict[str, float] = {}

    # First feature: highest relevance
    best_idx = int(np.argmax(relevance[remaining]))
    best_feat = valid[remaining[best_idx]]
    selected.append(best_feat)
    scores[best_feat] = float(relevance[remaining[best_idx]])
    remaining.pop(best_idx)

    for _ in range(K - 1):
        if not remaining:
            break

        best_score = -np.inf
        best_pos = 0

        for pos, idx in enumerate(remaining):
            rel = relevance[idx]
            # Redundancy: mean MI with already-selected features
            red = 0.0
            if selected:
                selected_indices = [valid.index(s) for s in selected]
                for s_idx in selected_indices:
                    # Approximate MI between features using correlation
                    corr = abs(np.corrcoef(X[:, idx], X[:, s_idx])[0, 1])
                    red += corr if np.isfinite(corr) else 0.0
                red /= len(selected)

            mrmr_score = rel - red
            if mrmr_score > best_score:
                best_score = mrmr_score
                best_pos = pos

        feat = valid[remaining[best_pos]]
        selected.append(feat)
        scores[feat] = float(best_score)
        remaining.pop(best_pos)

    logger.info("mRMR: selected %d of %d features (top: %s)", len(selected), len(valid),
                selected[0] if selected else "none")
    return selected, scores


# ---------------------------------------------------------------------------
# Orchestrator
# ---------------------------------------------------------------------------

def run_feature_selection(
    cache: pd.DataFrame,
    candidates: list[str],
    regime_labels: pd.Series | None = None,
    target_col: str = "return_1d",
    *,
    granger_result: Any = None,
) -> tuple[list[str], FeatureSelectionResult]:
    """Run 3-layer feature selection pipeline.

    Returns (selected_features, FeatureSelectionResult).
    """
    result = FeatureSelectionResult()
    result.n_input = len(candidates)

    if cache is None or cache.empty or len(candidates) < 2:
        result.error = "Insufficient data or candidates"
        result.final_selected = candidates
        result.n_output = len(candidates)
        return candidates, result

    if target_col not in cache.columns or cache[target_col].notna().sum() < 30:
        result.error = f"Target column {target_col} has insufficient data"
        result.final_selected = candidates
        result.n_output = len(candidates)
        return candidates, result

    # Load config
    try:
        from operator1.scoring_weights import get_weight
        n_trials = int(get_weight("feature_selection.boruta_n_trials", 50))
        alpha = float(get_weight("feature_selection.boruta_alpha", 0.05))
        rf_depth = int(get_weight("feature_selection.boruta_rf_max_depth", 7))
        pimp_n = int(get_weight("feature_selection.pimp_n_permutations", 50))
        pimp_sig = float(get_weight("feature_selection.pimp_significance", 0.05))
        mrmr_k = int(get_weight("feature_selection.mrmr_max_features", 30))
        min_regime = int(get_weight("feature_selection.min_regime_samples", 30))
    except Exception:
        n_trials, alpha, rf_depth = 50, 0.05, 7
        pimp_n, pimp_sig, mrmr_k, min_regime = 50, 0.05, 30, 30

    # --- Layer 1: Boruta ---
    try:
        confirmed, tentative, rejected = select_features_boruta(
            cache, target_col, candidates,
            n_trials=n_trials, alpha=alpha,
            rf_max_depth=rf_depth,
        )
        result.boruta_confirmed = confirmed
        result.boruta_tentative = tentative
        result.boruta_rejected = rejected
        result.boruta_n_iterations = n_trials
        # Pass confirmed + tentative to Layer 2
        layer1_survivors = confirmed + tentative
    except Exception as exc:
        logger.warning("Boruta failed, passing all candidates: %s", exc)
        layer1_survivors = candidates

    if not layer1_survivors:
        # Boruta rejected everything -- fall back to all candidates
        logger.warning("Boruta rejected all features -- falling back to full list")
        layer1_survivors = candidates

    # --- Layer 2: Regime-Conditional PIMP ---
    try:
        pimp_selected, regime_imp, p_values = select_features_regime_importance(
            cache, target_col, layer1_survivors,
            regime_labels=regime_labels,
            n_permutations=pimp_n,
            significance=pimp_sig,
            min_regime_samples=min_regime,
        )
        result.regime_importances = regime_imp
        result.regime_selected = {
            r: [f for f, imp in imps.items() if f in pimp_selected]
            for r, imps in regime_imp.items()
        }
        result.pimp_p_values = p_values
        layer2_survivors = pimp_selected if pimp_selected else layer1_survivors
    except Exception as exc:
        logger.warning("PIMP failed, using Boruta output: %s", exc)
        layer2_survivors = layer1_survivors

    if not layer2_survivors:
        layer2_survivors = layer1_survivors

    # --- Layer 3: mRMR ---
    try:
        mrmr_selected, mrmr_scores = select_features_mrmr(
            cache, target_col, layer2_survivors, K=mrmr_k,
        )
        result.mrmr_selected = mrmr_selected
        result.mrmr_scores = mrmr_scores
        final = mrmr_selected if mrmr_selected else layer2_survivors
    except Exception as exc:
        logger.warning("mRMR failed, using PIMP output: %s", exc)
        final = layer2_survivors

    if not final:
        final = candidates  # absolute fallback

    result.final_selected = final
    result.n_output = len(final)
    result.method_contributions = {
        "boruta": len(set(result.boruta_confirmed) & set(final)),
        "pimp": len(set(layer2_survivors) & set(final)),
        "mrmr": len(set(result.mrmr_selected) & set(final)),
    }
    result.fitted = True

    logger.info(
        "Feature selection: %d -> %d features (Boruta: %d confirmed, PIMP: %d, mRMR: %d)",
        result.n_input, result.n_output,
        len(result.boruta_confirmed), len(layer2_survivors), len(result.mrmr_selected),
    )

    return final, result
