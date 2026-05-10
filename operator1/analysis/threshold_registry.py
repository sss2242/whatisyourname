"""Centralized threshold registry -- single source of truth for survival triggers.

Replaces 6 fragmented threshold sources with one layered composition:
  Code defaults -> scoring_weights.yml -> sector overrides -> adaptive calibration

All consumers (survival_mode, monte_carlo, scenario_engine, USS) read from
the same registry via ``get_registry()``. Two view formats:

- ``survival_dict``: ``{"current_ratio_lt": 0.7, ...}`` for survival_mode.py
- ``mc_dict``: ``{"current_ratio": ("lt", 0.7), ...}`` for monte_carlo.py

Module singleton pattern (vectorbt). Frozen after init (Hydra).
Validated on construction (dynaconf). Layered priority (Hydra + dynaconf).
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, fields as dc_fields
from types import MappingProxyType
from typing import Any

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Threshold contract (frozen dataclass -- immutable after construction)
# ---------------------------------------------------------------------------

_FIELD_DEFAULTS = {
    "current_ratio_lt": 1.0,
    "debt_to_equity_abs_gt": 3.0,
    "fcf_yield_lt": 0.0,
    "drawdown_252d_lt": -0.40,
    "conflict_intensity_gt": 0.70,
    "inst_flow_momentum_lt": -0.15,
}

# Map scoring_weights.yml key names -> registry field names
_YML_KEY_MAP = {
    "current_ratio": "current_ratio_lt",
    "debt_to_equity": "debt_to_equity_abs_gt",
    "fcf_yield": "fcf_yield_lt",
    "drawdown_252d": "drawdown_252d_lt",
    "conflict_intensity": "conflict_intensity_gt",
    "inst_flow_momentum": "inst_flow_momentum_lt",
}

# Absolute bounds to prevent over-adaptation
_BOUNDS = {
    "current_ratio_lt": (0.1, 5.0),
    "debt_to_equity_abs_gt": (0.5, 50.0),
    "fcf_yield_lt": (-0.50, 0.50),
    "drawdown_252d_lt": (-0.90, -0.05),
    "conflict_intensity_gt": (0.1, 1.0),
    "inst_flow_momentum_lt": (-0.50, 0.0),
}


@dataclass(frozen=True)
class SurvivalThresholds:
    """Immutable survival trigger thresholds.

    Field names use ``_lt`` (less-than) and ``_gt`` (greater-than) suffixes
    to encode the comparison direction, matching survival_mode.py convention.
    """

    current_ratio_lt: float = 1.0
    debt_to_equity_abs_gt: float = 3.0
    fcf_yield_lt: float = 0.0
    drawdown_252d_lt: float = -0.40
    conflict_intensity_gt: float = 0.70
    inst_flow_momentum_lt: float = -0.15


# ---------------------------------------------------------------------------
# Registry
# ---------------------------------------------------------------------------


class ThresholdRegistry:
    """Single authoritative source for all survival thresholds.

    Created once after adaptive calibration (main.py Step 5j or
    backtest_runner equivalent). All consumers read via ``get_registry()``.

    Parameters
    ----------
    base_config:
        Full scoring_weights dict (from ``get_scoring_weights()``).
    sector:
        Company sector string for sector-aware overrides.
    adaptive_thresholds:
        ``ThresholdSet`` from ``compute_adaptive_thresholds()`` (optional).
    company_adjustments:
        Per-company overrides (optional, e.g. cash_adequacy_absolute).
    """

    def __init__(
        self,
        base_config: dict[str, Any] | None = None,
        sector: str = "",
        adaptive_thresholds: Any = None,
        company_adjustments: dict[str, float] | None = None,
    ) -> None:
        raw = self._compose(base_config, sector, adaptive_thresholds, company_adjustments)
        self._validate(raw)
        self._thresholds = SurvivalThresholds(**raw)
        self._sector = sector
        logger.debug(
            "ThresholdRegistry initialized: sector=%s, thresholds=%s",
            sector, self._thresholds,
        )

    @property
    def survival(self) -> SurvivalThresholds:
        """Frozen threshold object."""
        return self._thresholds

    @property
    def survival_dict(self) -> dict[str, float]:
        """Dict with ``_lt``/``_gt`` keys for survival_mode.py backward compat."""
        t = self._thresholds
        return {
            "current_ratio_lt": t.current_ratio_lt,
            "debt_to_equity_abs_gt": t.debt_to_equity_abs_gt,
            "fcf_yield_lt": t.fcf_yield_lt,
            "drawdown_252d_lt": t.drawdown_252d_lt,
            "conflict_intensity_gt": t.conflict_intensity_gt,
            "inst_flow_momentum_lt": t.inst_flow_momentum_lt,
        }

    @property
    def mc_dict(self) -> dict[str, tuple[str, float]]:
        """Dict with ``(operator, value)`` tuples for monte_carlo.py."""
        t = self._thresholds
        return {
            "current_ratio": ("lt", t.current_ratio_lt),
            "debt_to_equity_abs": ("gt", t.debt_to_equity_abs_gt),
            "fcf_yield": ("lt", t.fcf_yield_lt),
            "drawdown_252d": ("lt", t.drawdown_252d_lt),
        }

    @property
    def sector(self) -> str:
        return self._sector

    # ------------------------------------------------------------------
    # Composition logic (layered merge)
    # ------------------------------------------------------------------

    @staticmethod
    def _compose(
        base_config: dict[str, Any] | None,
        sector: str,
        adaptive: Any,
        company_adj: dict[str, float] | None,
    ) -> dict[str, float]:
        """Merge thresholds from 5 layers (lower overrides higher)."""
        # Layer 1: code defaults
        merged = dict(_FIELD_DEFAULTS)

        # Layer 2: scoring_weights.yml base values
        if base_config:
            st = base_config.get("survival_thresholds", {})
            for yml_key, field_name in _YML_KEY_MAP.items():
                if yml_key in st and not isinstance(st[yml_key], dict):
                    try:
                        merged[field_name] = float(st[yml_key])
                    except (TypeError, ValueError):
                        pass

        # Layer 3: sector overrides from scoring_weights.yml
        if sector and base_config:
            overrides_section = base_config.get("survival_thresholds", {}).get(
                "sector_overrides", {}
            )
            sector_key = sector.lower().replace(" ", "_")
            # Try exact match first, then substring match
            sector_cfg = overrides_section.get(sector_key, {})
            if not sector_cfg:
                for key, cfg in overrides_section.items():
                    if key in sector_key or sector_key in key:
                        sector_cfg = cfg
                        break
            if isinstance(sector_cfg, dict):
                for yml_key, value in sector_cfg.items():
                    field_name = _YML_KEY_MAP.get(yml_key)
                    if field_name:
                        try:
                            merged[field_name] = float(value)
                        except (TypeError, ValueError):
                            pass

        # Layer 4: adaptive calibration (runtime)
        if adaptive is not None and getattr(adaptive, "adapted", False):
            # ThresholdSet from adaptive_thresholds.py
            if hasattr(adaptive, "current_ratio_lt") and adaptive.current_ratio_lt is not None:
                merged["current_ratio_lt"] = float(adaptive.current_ratio_lt)
            if hasattr(adaptive, "debt_to_equity_abs_gt") and adaptive.debt_to_equity_abs_gt is not None:
                merged["debt_to_equity_abs_gt"] = float(adaptive.debt_to_equity_abs_gt)
            if hasattr(adaptive, "fcf_yield_lt") and adaptive.fcf_yield_lt is not None:
                merged["fcf_yield_lt"] = float(adaptive.fcf_yield_lt)
            if hasattr(adaptive, "drawdown_252d_lt") and adaptive.drawdown_252d_lt is not None:
                merged["drawdown_252d_lt"] = float(adaptive.drawdown_252d_lt)

        # Layer 5: company-specific adjustments
        if company_adj:
            for key, value in company_adj.items():
                if key in merged:
                    try:
                        merged[key] = float(value)
                    except (TypeError, ValueError):
                        pass

        return merged

    @staticmethod
    def _validate(raw: dict[str, float]) -> None:
        """Validate thresholds are within absolute bounds."""
        for field_name, (lo, hi) in _BOUNDS.items():
            val = raw.get(field_name)
            if val is not None and not (lo <= val <= hi):
                logger.warning(
                    "Threshold %s=%.4f outside bounds [%.2f, %.2f], clamping",
                    field_name, val, lo, hi,
                )
                raw[field_name] = max(lo, min(hi, val))


# ---------------------------------------------------------------------------
# Module singleton
# ---------------------------------------------------------------------------

_registry: ThresholdRegistry | None = None


def init_registry(
    base_config: dict[str, Any] | None = None,
    sector: str = "",
    adaptive_thresholds: Any = None,
    company_adjustments: dict[str, float] | None = None,
) -> ThresholdRegistry:
    """Initialize the global threshold registry.

    Called once in main.py after adaptive calibration (Step 5j).
    """
    global _registry
    _registry = ThresholdRegistry(
        base_config=base_config,
        sector=sector,
        adaptive_thresholds=adaptive_thresholds,
        company_adjustments=company_adjustments,
    )
    logger.info(
        "ThresholdRegistry: sector=%s, current_ratio_lt=%.2f, d/e_gt=%.2f",
        sector or "none",
        _registry.survival.current_ratio_lt,
        _registry.survival.debt_to_equity_abs_gt,
    )
    return _registry


def get_registry() -> ThresholdRegistry:
    """Return the global threshold registry.

    If not yet initialized (e.g. in tests), creates one with defaults.
    """
    global _registry
    if _registry is None:
        try:
            from operator1.scoring_weights import get_scoring_weights
            _registry = ThresholdRegistry(base_config=get_scoring_weights())
        except Exception:
            _registry = ThresholdRegistry()
    return _registry


def reset_registry() -> None:
    """Reset the singleton (for testing only)."""
    global _registry
    _registry = None
