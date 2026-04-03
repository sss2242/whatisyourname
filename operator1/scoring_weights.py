"""Centralized scoring weights loader.

Loads all tweakable model parameters from ``config/scoring_weights.yml``
and provides typed accessor functions used by the pipeline modules.

The config is cached in memory and can be reloaded at runtime (e.g.
from the dashboard Scoring Weights tab) via ``reload_scoring_weights()``.

Usage::

    from operator1.scoring_weights import get_scoring_weights, get_weight

    sw = get_scoring_weights()
    threshold = sw["survival_thresholds"]["current_ratio"]

    # Or use typed helpers:
    threshold = get_weight("survival_thresholds.current_ratio", default=1.0)
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)

_CONFIG_PATH = Path(__file__).resolve().parent.parent / "config" / "scoring_weights.yml"
_cache: dict[str, Any] | None = None


def _load() -> dict[str, Any]:
    """Load scoring weights from YAML, with defaults fallback."""
    try:
        import yaml
    except ImportError:
        logger.warning("PyYAML not available, using empty scoring weights")
        return {}

    if not _CONFIG_PATH.exists():
        logger.warning("Scoring weights config not found at %s", _CONFIG_PATH)
        return {}

    with open(_CONFIG_PATH, "r", encoding="utf-8") as fh:
        data = yaml.safe_load(fh) or {}

    logger.debug("Loaded scoring weights from %s (%d top-level keys)", _CONFIG_PATH, len(data))
    return data


def get_scoring_weights(*, reload: bool = False) -> dict[str, Any]:
    """Return the full scoring weights dict (cached).

    Parameters
    ----------
    reload:
        Force re-read from disk (e.g. after dashboard edit).
    """
    global _cache
    if _cache is None or reload:
        _cache = _load()
    return _cache


def reload_scoring_weights() -> dict[str, Any]:
    """Force reload from disk and return the new config."""
    return get_scoring_weights(reload=True)


def get_weight(dotted_path: str, default: Any = None) -> Any:
    """Get a nested value using dot notation.

    Example::

        get_weight("survival_thresholds.current_ratio", 1.0)
        get_weight("hierarchy_weights.normal", [20,20,20,20,20])
        get_weight("plane_weights.finance.copula", 1.0)
    """
    sw = get_scoring_weights()
    keys = dotted_path.split(".")
    current = sw
    for key in keys:
        if isinstance(current, dict) and key in current:
            current = current[key]
        else:
            return default
    return current


def save_scoring_weights(data: dict[str, Any]) -> None:
    """Save modified scoring weights back to disk.

    Called from the dashboard when the user edits parameters.
    """
    try:
        import yaml
    except ImportError:
        logger.error("PyYAML not available, cannot save scoring weights")
        return

    _CONFIG_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(_CONFIG_PATH, "w", encoding="utf-8") as fh:
        yaml.dump(data, fh, default_flow_style=False, sort_keys=False, allow_unicode=True)

    # Invalidate cache so next get_scoring_weights() picks up changes
    global _cache
    _cache = None
    logger.info("Scoring weights saved to %s", _CONFIG_PATH)


def get_scoring_weights_json() -> str:
    """Return scoring weights as a JSON string (for dashboard display)."""
    return json.dumps(get_scoring_weights(), indent=2, default=str)


# ---------------------------------------------------------------------------
# Typed accessors for common weight groups
# ---------------------------------------------------------------------------

def get_survival_thresholds() -> dict[str, float]:
    """Return survival trigger thresholds."""
    return get_weight("survival_thresholds", {
        "current_ratio": 1.0,
        "debt_to_equity": 3.0,
        "fcf_yield": 0.0,
        "drawdown_252d": -0.40,
        "conflict_intensity": 0.70,
        "inst_flow_momentum": -0.15,
    })


def get_hierarchy_weights(regime: str) -> list[float]:
    """Return tier weights for a survival regime."""
    defaults = {
        "normal": [20, 20, 20, 20, 20],
        "company_survival": [50, 30, 15, 4, 1],
        "modified_survival": [40, 35, 20, 4, 1],
        "extreme_survival": [60, 30, 10, 0, 0],
    }
    return get_weight(f"hierarchy_weights.{regime}", defaults.get(regime, [20, 20, 20, 20, 20]))


def get_plane_weights(plane: str) -> dict[str, float]:
    """Return model weight adjustments for an economic plane."""
    return get_weight(f"plane_weights.{plane}", {})


def get_graph_edge_weights() -> dict[str, float]:
    """Return edge weights for graph risk analysis."""
    return get_weight("graph_edge_weights", {
        "parent_companies": 2.8,
        "subsidiaries": 2.3,
        "suppliers": 1.2,
        "financial_institutions": 1.3,
        "customers": 1.1,
        "competitors": 1.0,
        "logistics": 0.8,
        "regulators": 0.5,
    })


def get_ensemble_params() -> dict[str, Any]:
    """Return ensemble aggregation parameters."""
    return get_weight("ensemble", {
        "min_rmse_for_weighting": 1e-10,
        "z_score_90": 1.645,
        "survival_risk_multiplier": 2.0,
        "transition_blend_halflife": 21,
        "fixed_share_parameter": 0.05,
        "fixed_share_eta": 0.1,
    })


def get_monte_carlo_params(regime: str = "normal") -> dict[str, Any]:
    """Return Monte Carlo parameters, optionally regime-overridden."""
    base = get_weight("monte_carlo", {
        "n_paths": 10000,
        "importance_tilt": 1.5,
        "horizons": [90, 252],
    })
    if regime != "normal":
        overrides = get_weight(f"monte_carlo_survival_overrides.{regime}", {})
        if overrides:
            merged = dict(base)
            merged.update(overrides)
            return merged
    return base


def get_conformal_params() -> dict[str, float]:
    """Return conformal prediction PID parameters."""
    return get_weight("conformal", {
        "target_coverage": 0.90,
        "pid_kp": 0.01,
        "pid_ki": 0.001,
        "pid_kd": 0.005,
        "copula_tail_amplification": 0.5,
    })


def get_conflict_weights() -> dict[str, float]:
    """Return conflict risk component weights."""
    return get_weight("conflict_weights", {
        "event_score": 0.40,
        "fatality_score": 0.20,
        "flag_score": 0.25,
        "news_score": 0.15,
    })


def get_vanity_weights() -> dict[str, float]:
    """Return vanity component weights."""
    return get_weight("vanity_weights", {
        "rnd_mismatch": 0.15,
        "sga_bloat": 0.25,
        "capital_misallocation": 0.30,
        "competitive_decay": 0.15,
        "sentiment_gap": 0.15,
    })


def get_frequency_fusion_weights() -> dict[str, dict[str, float]]:
    """Return horizon-to-frequency contribution weights."""
    return get_weight("frequency_fusion", {
        "1d": {"D": 1.0},
        "5d": {"D": 0.7, "W": 0.3},
        "21d": {"D": 0.3, "W": 0.4, "M": 0.3},
    })


def get_uss_params(regime: str) -> dict[str, Any]:
    """Return USS dimension parameters for a regime."""
    return get_weight(f"uss_model_switching.{regime}", {})
