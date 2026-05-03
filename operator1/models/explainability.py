"""F2 -- SHAP explainability for per-prediction feature attribution.

Provides model-agnostic SHAP (SHapley Additive exPlanations) values
that explain *why* a model made a specific prediction for a specific
company on a specific day.

Unlike Sobol sensitivity (which gives global variable importance across
the entire dataset), SHAP gives local explanations:

    "Model predicts debt-to-equity will rise 8% primarily because:
     +3.2% from rising long-term debt
     +2.1% from declining equity
     +1.5% from sector-wide leverage increase
     -0.8% from strong cash position"

These per-variable, per-prediction attributions feed directly into the
Gemini report narrative for richer, evidence-based explanations.

**Supported model types:**

- Tree ensembles (XGBoost, RF, GBM): uses ``TreeExplainer`` (exact, fast).
- Linear models (Kalman output, VAR): uses ``LinearExplainer``.
- Neural networks (LSTM, TFT): uses ``DeepExplainer`` or ``KernelExplainer``.
- Any model: falls back to ``KernelExplainer`` (model-agnostic, slower).

Top-level entry point:
    ``compute_shap_explanations(cache, model_bank, variables)``

Spec refs: Phase F enhancement (post D-E)
"""

from __future__ import annotations

import logging
import warnings
from dataclasses import dataclass, field
from typing import Any

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

# Maximum features to include in SHAP explanations (for readability).
MAX_TOP_FEATURES: int = 10

# Background sample size for KernelExplainer.
KERNEL_BACKGROUND_SIZE: int = 50

# Minimum data points for meaningful SHAP.
MIN_SAMPLES_FOR_SHAP: int = 30


# ---------------------------------------------------------------------------
# Result containers
# ---------------------------------------------------------------------------


@dataclass
class FeatureAttribution:
    """SHAP attribution for a single feature in a single prediction."""

    feature_name: str = ""
    shap_value: float = 0.0
    feature_value: float = 0.0
    direction: str = ""  # "positive" or "negative"
    contribution_pct: float = 0.0  # percentage of total absolute SHAP


@dataclass
class PredictionExplanation:
    """Full SHAP explanation for one variable's prediction."""

    variable: str = ""
    predicted_value: float = 0.0
    base_value: float = 0.0  # expected value (mean prediction)
    top_features: list[FeatureAttribution] = field(default_factory=list)
    total_shap_magnitude: float = 0.0
    explainer_type: str = ""  # "tree", "linear", "kernel", "deep"
    n_features_used: int = 0
    narrative: str = ""  # human-readable explanation string


@dataclass
class SHAPResult:
    """Collection of SHAP explanations across all predicted variables."""

    explanations: dict[str, PredictionExplanation] = field(
        default_factory=dict,
    )  # {variable_name: PredictionExplanation}
    global_importance: dict[str, float] = field(
        default_factory=dict,
    )  # {feature_name: mean_abs_shap}
    available: bool = False
    error: str | None = None


# ---------------------------------------------------------------------------
# Inline importance fallback (Fix 9: works without fitted model objects)
# ---------------------------------------------------------------------------


def from_inline_importance(
    inline_data: dict[str, dict[str, Any]],
    predictions: dict[str, float] | None = None,
) -> "SHAPResult":
    """Build a SHAPResult from forward-pass inline feature importance.

    This is the primary path when running in staged mode where model
    objects do not survive pickle serialization between sub-stages.
    The forward pass extracts ``feature_importances_`` / ``coef_`` from
    fitted models while they are still alive and stores them as a plain
    dict on ``ForwardPassResult.shap_inline``.

    Parameters
    ----------
    inline_data:
        ``{variable: {"top_drivers": [{"feature", "importance", "direction"}], "method": str}}``
    predictions:
        Optional ``{variable: predicted_value}`` for richer explanations.

    Returns
    -------
    SHAPResult with per-variable explanations and global importance.
    """
    result = SHAPResult()

    if not inline_data:
        result.error = "No inline importance data available"
        return result

    global_shap_sum: dict[str, float] = {}
    global_shap_count: dict[str, int] = {}

    for var, var_data in inline_data.items():
        drivers = var_data.get("top_drivers", [])
        method = var_data.get("method", "inline")
        if not drivers:
            continue

        total_magnitude = sum(d.get("importance", 0.0) for d in drivers)

        top_features: list[FeatureAttribution] = []
        for d in drivers[:MAX_TOP_FEATURES]:
            imp = d.get("importance", 0.0)
            pct = (imp / total_magnitude * 100) if total_magnitude > 0 else 0.0
            top_features.append(FeatureAttribution(
                feature_name=d.get("feature", ""),
                shap_value=imp,
                feature_value=0.0,
                direction=d.get("direction", "positive"),
                contribution_pct=pct,
            ))

        # Build narrative from top 5 drivers
        narrative_parts = []
        for fa in top_features[:5]:
            narrative_parts.append(f"{fa.shap_value:.4f} from {fa.feature_name}")
        narrative = f"Prediction driven by: {'; '.join(narrative_parts)}" if narrative_parts else ""

        predicted_value = predictions.get(var, 0.0) if predictions else 0.0

        result.explanations[var] = PredictionExplanation(
            variable=var,
            predicted_value=predicted_value,
            base_value=0.0,
            top_features=top_features,
            total_shap_magnitude=total_magnitude,
            explainer_type=method,
            n_features_used=len(drivers),
            narrative=narrative,
        )

        # Accumulate global importance
        for fa in top_features:
            global_shap_sum[fa.feature_name] = global_shap_sum.get(fa.feature_name, 0.0) + fa.shap_value
            global_shap_count[fa.feature_name] = global_shap_count.get(fa.feature_name, 0) + 1

    # Compute mean importance for global ranking
    for feat in global_shap_sum:
        count = global_shap_count.get(feat, 1)
        result.global_importance[feat] = global_shap_sum[feat] / count

    result.global_importance = dict(
        sorted(result.global_importance.items(), key=lambda x: -x[1])
    )

    result.available = len(result.explanations) > 0

    logger.info(
        "SHAP from inline importance: %d variables explained, method=%s",
        len(result.explanations),
        next(iter(inline_data.values()), {}).get("method", "unknown") if inline_data else "none",
    )

    return result


# ---------------------------------------------------------------------------
# SHAP computation (original library-based path)
# ---------------------------------------------------------------------------


def _try_tree_shap(
    model: Any,
    X: np.ndarray,
    feature_names: list[str],
) -> tuple[np.ndarray | None, float, str]:
    """Try TreeExplainer (exact SHAP for tree models)."""
    try:
        import shap  # type: ignore[import-untyped]

        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            explainer = shap.TreeExplainer(model)
            shap_values = explainer.shap_values(X)

        if isinstance(shap_values, list):
            shap_values = shap_values[0]

        base_value = float(explainer.expected_value)
        if isinstance(base_value, (list, np.ndarray)):
            base_value = float(base_value[0])

        return shap_values, base_value, "tree"
    except Exception:
        return None, 0.0, ""


def _try_kernel_shap(
    predict_fn: Any,
    X: np.ndarray,
    background: np.ndarray,
    feature_names: list[str],
) -> tuple[np.ndarray | None, float, str]:
    """Try KernelExplainer (model-agnostic, slower)."""
    try:
        import shap  # type: ignore[import-untyped]

        # Subsample background for speed
        if len(background) > KERNEL_BACKGROUND_SIZE:
            idx = np.random.choice(len(background), KERNEL_BACKGROUND_SIZE, replace=False)
            background = background[idx]

        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            explainer = shap.KernelExplainer(predict_fn, background)
            shap_values = explainer.shap_values(X, nsamples=100)

        if isinstance(shap_values, list):
            shap_values = shap_values[0]

        base_value = float(explainer.expected_value)
        if isinstance(base_value, (list, np.ndarray)):
            base_value = float(base_value[0])

        return shap_values, base_value, "kernel"
    except Exception:
        return None, 0.0, ""


def _build_explanation(
    variable: str,
    shap_values_row: np.ndarray,
    feature_values_row: np.ndarray,
    feature_names: list[str],
    base_value: float,
    predicted_value: float,
    explainer_type: str,
    max_features: int = MAX_TOP_FEATURES,
) -> PredictionExplanation:
    """Build a PredictionExplanation from raw SHAP values."""
    total_magnitude = float(np.sum(np.abs(shap_values_row)))

    # Sort features by absolute SHAP value
    indices = np.argsort(np.abs(shap_values_row))[::-1]

    top_features: list[FeatureAttribution] = []
    for idx in indices[:max_features]:
        sv = float(shap_values_row[idx])
        fv = float(feature_values_row[idx]) if idx < len(feature_values_row) else 0.0
        name = feature_names[idx] if idx < len(feature_names) else f"feature_{idx}"

        pct = (abs(sv) / total_magnitude * 100) if total_magnitude > 0 else 0.0

        top_features.append(FeatureAttribution(
            feature_name=name,
            shap_value=sv,
            feature_value=fv,
            direction="positive" if sv > 0 else "negative",
            contribution_pct=pct,
        ))

    # Build narrative string
    narrative_parts: list[str] = []
    for fa in top_features[:5]:
        sign = "+" if fa.shap_value > 0 else ""
        narrative_parts.append(
            f"{sign}{fa.shap_value:.4f} from {fa.feature_name}"
        )
    narrative = f"Prediction driven by: {'; '.join(narrative_parts)}"

    return PredictionExplanation(
        variable=variable,
        predicted_value=predicted_value,
        base_value=base_value,
        top_features=top_features,
        total_shap_magnitude=total_magnitude,
        explainer_type=explainer_type,
        n_features_used=len(feature_names),
        narrative=narrative,
    )


def explain_tree_prediction(
    model: Any,
    features: pd.DataFrame,
    target_variable: str,
    predicted_value: float,
) -> PredictionExplanation | None:
    """Compute SHAP explanation for a tree ensemble prediction.

    Parameters
    ----------
    model:
        Fitted tree model (XGBoost, RandomForest, GradientBoosting).
    features:
        Feature DataFrame. Last row is used for explanation.
    target_variable:
        Name of the variable being predicted.
    predicted_value:
        The model's prediction.

    Returns
    -------
    PredictionExplanation or None if SHAP computation fails.
    """
    if len(features) < MIN_SAMPLES_FOR_SHAP:
        return None

    feature_names = list(features.columns)
    X = features.values
    X_last = X[-1:]

    shap_values, base_value, explainer_type = _try_tree_shap(
        model, X_last, feature_names,
    )

    if shap_values is None:
        return None

    # Handle multi-output or single-row
    if shap_values.ndim == 1:
        sv_row = shap_values
    else:
        sv_row = shap_values[0]

    return _build_explanation(
        variable=target_variable,
        shap_values_row=sv_row,
        feature_values_row=X_last[0],
        feature_names=feature_names,
        base_value=base_value,
        predicted_value=predicted_value,
        explainer_type=explainer_type,
    )


def explain_any_prediction(
    predict_fn: Any,
    features: pd.DataFrame,
    target_variable: str,
    predicted_value: float,
) -> PredictionExplanation | None:
    """Compute SHAP explanation for any model using KernelExplainer.

    This is model-agnostic but slower than TreeExplainer.

    Parameters
    ----------
    predict_fn:
        A callable that takes a 2D numpy array and returns predictions.
    features:
        Feature DataFrame. Last row is explained.
    target_variable:
        Name of the variable being predicted.
    predicted_value:
        The model's prediction.

    Returns
    -------
    PredictionExplanation or None if SHAP computation fails.
    """
    if len(features) < MIN_SAMPLES_FOR_SHAP:
        return None

    feature_names = list(features.columns)
    X = features.values
    X_last = X[-1:]
    background = X[:-1] if len(X) > 1 else X

    shap_values, base_value, explainer_type = _try_kernel_shap(
        predict_fn, X_last, background, feature_names,
    )

    if shap_values is None:
        return None

    if shap_values.ndim == 1:
        sv_row = shap_values
    else:
        sv_row = shap_values[0]

    return _build_explanation(
        variable=target_variable,
        shap_values_row=sv_row,
        feature_values_row=X_last[0],
        feature_names=feature_names,
        base_value=base_value,
        predicted_value=predicted_value,
        explainer_type=explainer_type,
    )


# ---------------------------------------------------------------------------
# Pipeline-level function
# ---------------------------------------------------------------------------


def compute_shap_explanations(
    cache: pd.DataFrame,
    predictions: dict[str, float],
    tree_models: dict[str, Any] | None = None,
    predict_fns: dict[str, Any] | None = None,
    feature_columns: list[str] | None = None,
    max_variables: int = 20,
) -> SHAPResult:
    """Compute SHAP explanations for the current set of predictions.

    Tries TreeExplainer first (for tree models), falls back to
    KernelExplainer for other model types.

    Parameters
    ----------
    cache:
        Full feature table with numeric columns.
    predictions:
        ``{variable_name: predicted_value}`` for the current step.
    tree_models:
        ``{variable_name: fitted_tree_model}`` for tree-based explanations.
    predict_fns:
        ``{variable_name: callable}`` for model-agnostic explanations.
    feature_columns:
        Explicit list of feature columns.  If None, uses all numeric
        columns except the target variable.
    max_variables:
        Maximum number of variables to explain (for speed).

    Returns
    -------
    SHAPResult with per-variable explanations and global importance.
    """
    result = SHAPResult()

    try:
        import shap  # type: ignore[import-untyped]  # noqa: F401
    except ImportError:
        result.error = "shap library not installed (pip install shap)"
        logger.warning(result.error)
        return result

    if tree_models is None:
        tree_models = {}
    if predict_fns is None:
        predict_fns = {}

    # Determine feature columns
    if feature_columns is None:
        feature_columns = [
            c for c in cache.columns
            if cache[c].dtype in (np.float64, np.float32, np.int64)
        ]

    if len(feature_columns) < 2:
        result.error = "Insufficient feature columns for SHAP"
        logger.warning(result.error)
        return result

    # Clean features
    features = cache[feature_columns].dropna()
    if len(features) < MIN_SAMPLES_FOR_SHAP:
        result.error = f"Insufficient data for SHAP ({len(features)} < {MIN_SAMPLES_FOR_SHAP})"
        logger.warning(result.error)
        return result

    # Global SHAP importance accumulator
    global_shap_sum: dict[str, float] = {}
    global_shap_count: dict[str, int] = {}

    variables_to_explain = list(predictions.keys())[:max_variables]

    for var in variables_to_explain:
        predicted_value = predictions[var]

        # Prepare features (exclude target variable)
        feat_cols = [c for c in feature_columns if c != var]
        if not feat_cols:
            continue
        feat_df = features[feat_cols]

        explanation: PredictionExplanation | None = None

        # Try tree SHAP first
        if var in tree_models:
            explanation = explain_tree_prediction(
                model=tree_models[var],
                features=feat_df,
                target_variable=var,
                predicted_value=predicted_value,
            )

        # Fall back to kernel SHAP
        if explanation is None and var in predict_fns:
            explanation = explain_any_prediction(
                predict_fn=predict_fns[var],
                features=feat_df,
                target_variable=var,
                predicted_value=predicted_value,
            )

        if explanation is not None:
            result.explanations[var] = explanation

            # Accumulate global importance
            for fa in explanation.top_features:
                global_shap_sum[fa.feature_name] = (
                    global_shap_sum.get(fa.feature_name, 0.0) + abs(fa.shap_value)
                )
                global_shap_count[fa.feature_name] = (
                    global_shap_count.get(fa.feature_name, 0) + 1
                )

    # Compute mean absolute SHAP for global importance
    for feat in global_shap_sum:
        count = global_shap_count.get(feat, 1)
        result.global_importance[feat] = global_shap_sum[feat] / count

    # Sort by importance
    result.global_importance = dict(
        sorted(result.global_importance.items(), key=lambda x: -x[1])
    )

    result.available = len(result.explanations) > 0

    logger.info(
        "SHAP explanations computed for %d/%d variables, "
        "top global feature: %s",
        len(result.explanations),
        len(variables_to_explain),
        list(result.global_importance.keys())[:1] or ["none"],
    )

    return result


def format_shap_for_profile(shap_result: SHAPResult) -> dict[str, Any]:
    """Format SHAP results for inclusion in the company profile JSON.

    This produces a structure that Gemini can use to generate
    evidence-based narrative in the report.

    Returns
    -------
    Dict suitable for ``company_profile["shap_explanations"]``.
    """
    if not shap_result.available:
        return {"available": False, "error": shap_result.error}

    output: dict[str, Any] = {
        "available": True,
        "n_variables_explained": len(shap_result.explanations),
        "global_feature_importance": dict(
            list(shap_result.global_importance.items())[:15]
        ),
        "per_variable": {},
    }

    for var, exp in shap_result.explanations.items():
        output["per_variable"][var] = {
            "predicted": exp.predicted_value,
            "base_value": exp.base_value,
            "narrative": exp.narrative,
            "explainer": exp.explainer_type,
            "top_drivers": [
                {
                    "feature": fa.feature_name,
                    "shap_value": round(fa.shap_value, 6),
                    "direction": fa.direction,
                    "contribution_pct": round(fa.contribution_pct, 1),
                }
                for fa in exp.top_features[:5]
            ],
        }

    return output
