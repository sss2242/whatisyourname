"""T4.3 -- Survival hierarchy weight assignment.

For each day, selects a **regime** based on the survival flags computed
by T4.1 and looks up the corresponding tier weight vector from
``config/survival_hierarchy.yml``.

Regime selection logic (per The_Apps_core_idea.pdf Section C.4):
  - Neither company nor country flag              -> ``normal``
  - Company flag only                             -> ``company_survival``
  - Country flag only, company NOT protected      -> ``modified_survival``
  - Country flag only, company IS protected       -> ``normal``
  - Both flags                                    -> ``extreme_survival``

When ``vanity_percentage`` exceeds the configured threshold **and** the
company is in a survival regime, weight is shifted from Tier 4/5 to
Tier 1 (per Section C.6 of the spec).

Output columns:
  - ``survival_regime`` (categorical label)
  - ``hierarchy_tier1_weight`` ... ``hierarchy_tier5_weight``
"""

from __future__ import annotations

import logging
from typing import Any

import numpy as np
import pandas as pd

from operator1.config_loader import load_config

logger = logging.getLogger(__name__)

# Number of tiers in the hierarchy
_NUM_TIERS = 5


# ---------------------------------------------------------------------------
# Regime selection
# ---------------------------------------------------------------------------


def _select_regime(
    company_flag: int,
    country_flag: int,
    protected_flag: int,
) -> str:
    """Determine the survival regime for a single day.

    Parameters
    ----------
    company_flag:
        1 if company survival mode is active.
    country_flag:
        1 if country survival mode is active.
    protected_flag:
        1 if the company is country-protected.

    Returns
    -------
    str
        One of: ``"normal"``, ``"company_survival"``,
        ``"modified_survival"``, ``"extreme_survival"``.
    """
    company = bool(company_flag)
    country = bool(country_flag)
    protected = bool(protected_flag)

    if not company and not country:
        return "normal"
    if company and not country:
        return "company_survival"
    if not company and country:
        # Country crisis only.  Per spec Section C.4 Case 3/3b:
        # - If protected by government -> use standard (normal) weights.
        # - If NOT protected -> use modified_survival weights.
        if protected:
            return "normal"
        return "modified_survival"
    # Both company and country in crisis
    return "extreme_survival"


def select_regime_series(df: pd.DataFrame) -> pd.Series:
    """Vectorised regime selection for the full feature table.

    Parameters
    ----------
    df:
        Must contain ``company_survival_mode_flag``,
        ``country_survival_mode_flag``, ``country_protected_flag``.

    Returns
    -------
    pd.Series
        String series with regime labels.
    """
    company = df.get(
        "company_survival_mode_flag", pd.Series(0, index=df.index),
    ).fillna(0).astype(int)
    country = df.get(
        "country_survival_mode_flag", pd.Series(0, index=df.index),
    ).fillna(0).astype(int)
    protected = df.get(
        "country_protected_flag", pd.Series(0, index=df.index),
    ).fillna(0).astype(int)

    regimes = pd.Series("normal", index=df.index, dtype="object")

    # Company only
    regimes = regimes.where(
        ~((company == 1) & (country == 0)),
        other="company_survival",
    )
    # Country only, NOT protected -> modified_survival
    regimes = regimes.where(
        ~((company == 0) & (country == 1) & (protected == 0)),
        other="modified_survival",
    )
    # Country only, IS protected -> stays "normal" (no change needed,
    # already defaulted to "normal")

    # Both company and country in crisis
    regimes = regimes.where(
        ~((company == 1) & (country == 1)),
        other="extreme_survival",
    )

    regimes.name = "survival_regime"
    return regimes


# ---------------------------------------------------------------------------
# Weight lookup & vanity adjustment
# ---------------------------------------------------------------------------


def _get_weight_vector(
    regime: str,
    regimes_cfg: dict[str, Any],
) -> list[float]:
    """Look up tier weights for a given regime from config.

    Falls back to ``normal`` if the regime is not found.
    """
    preset = regimes_cfg.get(regime, regimes_cfg.get("normal", {}))
    weights = preset.get("weights", [0.2] * _NUM_TIERS)
    return weights[:_NUM_TIERS]


def _apply_vanity_adjustment(
    weights: list[float],
    vanity_pct: float,
    regime: str,
    vanity_cfg: dict[str, Any],
) -> list[float]:
    """Adjust tier weights based on vanity percentage.

    Per spec Section C.6: when ``vanity_pct`` exceeds the threshold
    **and** the company is in a survival regime, shift weight from
    Tier 4 and Tier 5 toward Tier 1 (liquidity).

    Parameters
    ----------
    weights:
        Current 5-element weight list.
    vanity_pct:
        The vanity percentage for this day.
    regime:
        The survival regime label for this day.
    vanity_cfg:
        The ``vanity_adjustment`` section from config.

    Returns
    -------
    list[float]
        Adjusted and normalised weights.
    """
    threshold = vanity_cfg.get("threshold", 10.0)
    if np.isnan(vanity_pct) or vanity_pct <= threshold:
        return weights

    # Only apply vanity adjustment in survival regimes (not normal mode).
    if regime == "normal":
        return weights

    tier1_delta = vanity_cfg.get("tier1_delta", 0.05)
    tier4_delta = vanity_cfg.get("tier4_delta", -0.02)
    tier5_delta = vanity_cfg.get("tier5_delta", -0.03)

    adjusted = list(weights)
    # tier1 is index 0, tier4 is index 3, tier5 is index 4
    adjusted[0] = adjusted[0] + tier1_delta
    adjusted[3] = adjusted[3] + tier4_delta
    adjusted[4] = adjusted[4] + tier5_delta

    # Clamp to non-negative
    adjusted = [max(0.0, w) for w in adjusted]

    # Normalise to sum to 1.0
    total = sum(adjusted)
    if total > 0:
        adjusted = [w / total for w in adjusted]

    return adjusted


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def _entropy_blend_weights(
    base_weights: list[float],
    forward_pass_errors: dict[str, list[float]] | None,
    alpha: float = 0.7,
) -> list[float]:
    """Blend regime-default weights with entropy-derived weights.

    Information-theoretic approach (Jaynes 1957): allocate more attention
    (weight) to tiers where prediction errors have higher entropy
    (more uncertainty = more information needed).

    Parameters
    ----------
    base_weights:
        Regime-default weights (sum to 1.0).
    forward_pass_errors:
        Dict of ``tier{i}`` -> list of prediction errors from a prior
        forward pass. When None or empty, returns base_weights unchanged.
    alpha:
        Blend factor: 1.0 = pure regime defaults, 0.0 = pure entropy.
        Decays toward 0.5 as more data accumulates.

    Returns
    -------
    list[float]
        Blended weights (sum to ~1.0).
    """
    if not forward_pass_errors:
        return base_weights

    import numpy as np

    n_tiers = len(base_weights)
    entropies = []
    for i in range(1, n_tiers + 1):
        errors = forward_pass_errors.get(f"tier{i}", [])
        if len(errors) < 10:
            entropies.append(1.0)  # neutral entropy when insufficient data
            continue
        arr = np.array(errors, dtype=float)
        arr = arr[np.isfinite(arr)]
        if len(arr) < 10:
            entropies.append(1.0)
            continue
        # Histogram-based Shannon entropy
        hist, _ = np.histogram(arr, bins=min(20, len(arr) // 3), density=True)
        hist = hist[hist > 0]
        if len(hist) < 2:
            entropies.append(1.0)
            continue
        hist = hist / hist.sum()
        entropies.append(float(-np.sum(hist * np.log2(hist + 1e-15))))

    total_entropy = sum(entropies)
    if total_entropy < 1e-10:
        return base_weights

    entropy_weights = [e / total_entropy for e in entropies]

    # Blend: alpha * regime_default + (1-alpha) * entropy_derived
    # Cap shift at +/- 10% from regime defaults
    blended = []
    for bw, ew in zip(base_weights, entropy_weights):
        shift = (1 - alpha) * (ew - bw)
        shift = max(-0.10, min(0.10, shift))
        blended.append(bw + shift)

    # Re-normalize to sum to 1.0
    s = sum(blended)
    if s > 1e-10:
        blended = [w / s for w in blended]

    return blended


def compute_hierarchy_weights(
    df: pd.DataFrame,
    config: dict[str, Any] | None = None,
    forward_pass_errors: dict[str, list[float]] | None = None,
) -> pd.DataFrame:
    """Compute survival hierarchy tier weights for every day.

    Parameters
    ----------
    df:
        Daily feature table with survival flags and vanity_percentage
        already computed.
    config:
        Override survival hierarchy config (loads YAML if None).

    Returns
    -------
    pd.DataFrame
        Input DataFrame augmented with:
        - ``survival_regime``
        - ``hierarchy_tier1_weight`` ... ``hierarchy_tier5_weight``
    """
    if config is None:
        config = load_config("survival_hierarchy")

    regimes_cfg = config.get("regimes", {})
    vanity_cfg = config.get("vanity_adjustment", {})
    vanity_v2_cfg = config.get("vanity_v2", {})

    result = df.copy()

    # 1. Select regime for each day
    result["survival_regime"] = select_regime_series(result)

    # 2. Assign weights per day
    # Prefer vanity_score (v2) if available, fall back to vanity_percentage
    if "vanity_score" in result.columns and result["vanity_score"].notna().any():
        vanity_pct = result["vanity_score"]
        vanity_trend = result.get("vanity_trend", pd.Series(np.nan, index=result.index))
        # Use v2 threshold if available
        v2_adj = vanity_v2_cfg.get("hierarchy_adjustment", {})
        if v2_adj:
            vanity_cfg = {
                "threshold": v2_adj.get("threshold", vanity_cfg.get("threshold", 10.0)),
                "tier1_delta": vanity_cfg.get("tier1_delta", 0.05),
                "tier4_delta": vanity_cfg.get("tier4_delta", -0.02),
                "tier5_delta": vanity_cfg.get("tier5_delta", -0.03),
                "rising_trend_multiplier": v2_adj.get("rising_trend_multiplier", 1.0),
            }
    else:
        vanity_pct = result.get(
            "vanity_percentage", pd.Series(np.nan, index=result.index),
        )
        vanity_trend = pd.Series(np.nan, index=result.index)

    # Pre-compute weight vectors for each regime
    regime_weights_cache: dict[str, list[float]] = {}
    for regime_name in ("normal", "company_survival", "modified_survival", "extreme_survival"):
        regime_weights_cache[regime_name] = _get_weight_vector(
            regime_name, regimes_cfg,
        )

    # Build weight columns
    tier_columns: dict[str, list[float]] = {
        f"hierarchy_tier{i+1}_weight": [] for i in range(_NUM_TIERS)
    }

    for idx in range(len(result)):
        regime = result["survival_regime"].iloc[idx]
        base_weights = list(regime_weights_cache.get(regime, [0.2] * _NUM_TIERS))

        # Apply entropy-based blending (Jaynes 1957 maximum entropy)
        # Shifts weight toward tiers with higher prediction uncertainty
        if forward_pass_errors:
            base_weights = _entropy_blend_weights(
                base_weights, forward_pass_errors, alpha=0.7,
            )

        # Apply vanity adjustment (only active in survival regimes)
        vp = vanity_pct.iloc[idx]
        # Amplify adjustment when vanity trend is rising
        effective_cfg = dict(vanity_cfg)
        trend_val = vanity_trend.iloc[idx]
        if trend_val == "rising":
            multiplier = vanity_cfg.get("rising_trend_multiplier", 1.0)
            effective_cfg["tier1_delta"] = vanity_cfg.get("tier1_delta", 0.05) * multiplier
            effective_cfg["tier4_delta"] = vanity_cfg.get("tier4_delta", -0.02) * multiplier
            effective_cfg["tier5_delta"] = vanity_cfg.get("tier5_delta", -0.03) * multiplier
        adjusted = _apply_vanity_adjustment(base_weights, vp, regime, effective_cfg)

        for tier_idx in range(_NUM_TIERS):
            col = f"hierarchy_tier{tier_idx+1}_weight"
            tier_columns[col].append(adjusted[tier_idx])

    for col, values in tier_columns.items():
        result[col] = values

    # Log regime distribution
    regime_counts = result["survival_regime"].value_counts()
    logger.info("Regime distribution:\n%s", regime_counts.to_string())

    # Verify weights sum to 1.0
    weight_cols = [f"hierarchy_tier{i+1}_weight" for i in range(_NUM_TIERS)]
    weight_sums = result[weight_cols].sum(axis=1)
    max_drift = (weight_sums - 1.0).abs().max()
    if max_drift > 1e-6:
        logger.warning(
            "Weight normalisation drift detected: max abs deviation = %.8f",
            max_drift,
        )

    logger.info(
        "Hierarchy weights computed: %d days, %d unique regimes",
        len(result), result["survival_regime"].nunique(),
    )

    return result
