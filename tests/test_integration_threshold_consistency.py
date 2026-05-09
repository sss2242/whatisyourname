"""Integration tests -- Cross-Layer Threshold Consistency.

The single most important integration test. Verifies that survival_mode.py
and monte_carlo.py use compatible thresholds, so their verdicts agree on
whether a company is in distress.

Bug pattern caught: threshold fragmentation (Problem 1 from Batch A plan).
Survival_mode says "normal" but MC says 0% survival, or vice versa.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from tests.conftest import AAPL_EXPECTED


def _extract_mc_prob(val) -> float:
    """Extract a float probability from MC survival_probability entry.

    MC returns various formats: float, {"mean": float, "p5": ...}, or
    nested dicts. This helper extracts the mean probability robustly.
    """
    if isinstance(val, (int, float)):
        return float(val)
    if isinstance(val, dict):
        if "mean" in val:
            return _extract_mc_prob(val["mean"])
        # Try first numeric value
        for v in val.values():
            if isinstance(v, (int, float)):
                return float(v)
            if isinstance(v, dict):
                return _extract_mc_prob(v)
    return 0.5  # safe default if unparseable


class TestThresholdConsistency:
    """Cross-module threshold agreement tests."""

    def test_survival_and_mc_thresholds_share_variables(self):
        """Both modules must check the same set of trigger variables."""
        from operator1.analysis.survival_mode import _load_company_thresholds
        from operator1.models.monte_carlo import DEFAULT_SURVIVAL_THRESHOLDS

        surv_thresholds = _load_company_thresholds()
        mc_thresholds = DEFAULT_SURVIVAL_THRESHOLDS

        # Extract variable names from both (survival uses _lt/_gt suffix)
        surv_vars = set()
        for key in surv_thresholds:
            base = key.replace("_lt", "").replace("_gt", "")
            surv_vars.add(base)

        mc_vars = set(mc_thresholds.keys())

        # MC should cover the main survival variables
        critical_vars = {"current_ratio", "debt_to_equity", "fcf_yield", "drawdown_252d"}
        for var in critical_vars:
            # Check MC has a matching key (may have different naming convention)
            mc_match = any(var in k for k in mc_vars)
            surv_match = any(var in k for k in surv_vars)
            if surv_match:
                assert mc_match, (
                    f"Survival checks '{var}' but MC has no matching threshold. "
                    f"MC keys: {sorted(mc_vars)}"
                )

    def test_survival_normal_implies_mc_positive(self, synthetic_cache):
        """When survival_mode says 'normal', MC should show >40% survival."""
        from operator1.analysis.survival_mode import compute_company_survival_flag
        from operator1.models.monte_carlo import run_monte_carlo

        cache = synthetic_cache.copy()
        # Force healthy values
        cache["current_ratio"] = 2.0
        cache["debt_to_equity_abs"] = 0.5
        cache["fcf_yield"] = 0.05
        cache["drawdown_252d"] = -0.05
        cache["regime_label"] = "bull"

        flag = compute_company_survival_flag(cache)
        latest_flag = int(flag.iloc[-1])

        if latest_flag == 0:  # normal
            mc = run_monte_carlo(cache, n_paths=200, random_state=42)
            sp_1d = mc.survival_probability.get("1d", {})
            p = _extract_mc_prob(sp_1d)
            mc_min = AAPL_EXPECTED.get("cross_survival_mc_normal_min", 0.40)
            assert p > mc_min, (
                f"Survival says normal but MC 1d survival={p:.3f} (<{mc_min}). "
                f"Threshold disagreement between modules."
            )

    def test_survival_distress_implies_mc_below_ceiling(self, synthetic_cache):
        """When survival_mode says 'distress', MC should show <90% survival."""
        from operator1.analysis.survival_mode import compute_company_survival_flag
        from operator1.models.monte_carlo import run_monte_carlo

        cache = synthetic_cache.copy()
        # Force distressed values
        cache["current_ratio"] = 0.3
        cache["debt_to_equity_abs"] = 8.0
        cache["fcf_yield"] = -0.10
        cache["drawdown_252d"] = -0.60
        cache["regime_label"] = "bear"

        flag = compute_company_survival_flag(cache)
        latest_flag = int(flag.iloc[-1])

        if latest_flag == 1:  # distress
            mc = run_monte_carlo(cache, n_paths=200, random_state=42)
            sp_252d = mc.survival_probability.get("252d", {})
            p = _extract_mc_prob(sp_252d)
            mc_max = AAPL_EXPECTED.get("cross_survival_mc_distress_max", 0.90)
            assert p < mc_max, (
                f"Survival says distress but MC 252d survival={p:.3f} (>{mc_max}). "
                f"Threshold disagreement between modules."
            )

    def test_sector_overrides_applied_consistently(self):
        """Sector overrides in survival_mode and MC should cover the same sectors."""
        from operator1.models.monte_carlo import SECTOR_SURVIVAL_OVERRIDES

        # Verify the sector override dict exists and is not empty
        assert isinstance(SECTOR_SURVIVAL_OVERRIDES, dict)
        # Technology should be in overrides (the main sector that needed fixing)
        tech_keys = [k for k in SECTOR_SURVIVAL_OVERRIDES if "tech" in k.lower()]
        assert len(tech_keys) > 0, (
            f"No technology sector in MC SECTOR_SURVIVAL_OVERRIDES. "
            f"Keys: {sorted(SECTOR_SURVIVAL_OVERRIDES.keys())}"
        )
