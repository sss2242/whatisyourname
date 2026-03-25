"""Fuzzy Logic government protection assessment.

Replaces the binary ``country_protected_flag`` (0 or 1) with a
continuous fuzzy membership score in [0, 1] that captures the
*degree* of government protection.

**Fuzzy variables and membership functions:**

1. **Sector strategicness** (0-1):
   - Core strategic (defense, energy, banking): 0.9-1.0
   - Semi-strategic (utilities, telecom, transport): 0.5-0.7
   - Non-strategic: 0.0-0.2

2. **Economic significance** (market_cap / GDP ratio):
   - Systemically important (>0.5%): 0.9-1.0
   - Significant (0.1-0.5%): 0.4-0.8
   - Minor (<0.1%): 0.0-0.3

3. **Policy responsiveness** (rate cut magnitude in lookback):
   - Emergency intervention (>2%): 0.8-1.0
   - Moderate adjustment (1-2%): 0.3-0.6
   - Normal policy (<1%): 0.0-0.2

The final protection score uses a fuzzy OR (maximum) of all three
dimensions, then applies a defuzzification step to produce a single
score.

Integration point: ``operator1/analysis/survival_mode.py``
Replaces: binary ``country_protected_flag``

Top-level entry points:
    ``compute_fuzzy_protection`` -- per-day fuzzy protection scores.
    ``FuzzyProtectionResult`` -- result container with per-dimension scores.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Fuzzy membership functions
# ---------------------------------------------------------------------------


def _sigmoid(x: float, center: float, steepness: float = 10.0) -> float:
    """Smooth sigmoid membership function in [0, 1]."""
    return float(1.0 / (1.0 + np.exp(-steepness * (x - center))))


def _trapezoidal(
    x: float,
    a: float, b: float, c: float, d: float,
) -> float:
    """Trapezoidal membership function.

    Returns 0 for x <= a or x >= d, rises linearly from a to b,
    flat 1.0 from b to c, falls linearly from c to d.
    """
    if x <= a or x >= d:
        return 0.0
    if a < x < b:
        return (x - a) / (b - a)
    if b <= x <= c:
        return 1.0
    if c < x < d:
        return (d - x) / (d - c)
    return 0.0


# ---------------------------------------------------------------------------
# Sector strategicness
# ---------------------------------------------------------------------------

# Sector -> fuzzy membership score.  Core strategic sectors get 0.9+,
# semi-strategic get 0.5-0.7, everything else gets 0.1.
_SECTOR_SCORES: dict[str, float] = {
    # Core strategic
    "defense": 1.0,
    "energy": 0.95,
    "banking": 0.90,
    "oil & gas": 0.90,
    "aerospace & defense": 1.0,
    # Semi-strategic
    "utilities": 0.70,
    "telecom": 0.65,
    "telecommunications": 0.65,
    "transportation": 0.55,
    "healthcare": 0.50,
    "pharmaceuticals": 0.50,
    "insurance": 0.45,
    "financial services": 0.60,
    # Lower priority
    "technology": 0.30,
    "consumer electronics": 0.20,
    "retail": 0.15,
    "consumer discretionary": 0.15,
    "media": 0.10,
    "entertainment": 0.10,
}


def _sector_membership(sector: str | None) -> float:
    """Return fuzzy membership for sector strategicness."""
    if not sector:
        return 0.1  # unknown -> low protection
    s = sector.lower().strip()
    # Exact match
    if s in _SECTOR_SCORES:
        return _SECTOR_SCORES[s]
    # Partial match
    for key, score in _SECTOR_SCORES.items():
        if key in s or s in key:
            return score
    return 0.1  # default: non-strategic


# ---------------------------------------------------------------------------
# Economic significance
# ---------------------------------------------------------------------------


def _economic_significance(
    market_cap: float | None,
    gdp: float | None,
) -> float:
    """Fuzzy membership for market_cap / GDP ratio."""
    if market_cap is None or gdp is None or gdp <= 0:
        return 0.0  # unknown -> no protection signal

    ratio = market_cap / gdp

    # Sigmoid centered at 0.001 (0.1% of GDP) with a smooth ramp
    # to 1.0 for very large companies (>0.5% of GDP)
    if ratio >= 0.005:
        return 1.0
    if ratio >= 0.001:
        # Linear ramp from 0.4 at 0.1% to 1.0 at 0.5%
        return 0.4 + 0.6 * (ratio - 0.001) / (0.005 - 0.001)
    if ratio >= 0.0001:
        # Linear ramp from 0.0 at 0.01% to 0.4 at 0.1%
        return 0.4 * (ratio - 0.0001) / (0.001 - 0.0001)
    return 0.0


# ---------------------------------------------------------------------------
# Policy responsiveness
# ---------------------------------------------------------------------------


def _policy_responsiveness(
    rate_cut_pct: float | None,
) -> float:
    """Fuzzy membership for central bank emergency intervention."""
    if rate_cut_pct is None:
        return 0.0  # no data -> no signal

    cut = abs(rate_cut_pct)
    if cut >= 2.0:
        return min(1.0, 0.8 + 0.1 * (cut - 2.0))
    if cut >= 1.0:
        return 0.3 + 0.5 * (cut - 1.0)
    if cut >= 0.25:
        return 0.1 * (cut - 0.25) / 0.75
    return 0.0


# ---------------------------------------------------------------------------
# Result container
# ---------------------------------------------------------------------------


@dataclass
class FuzzyProtectionResult:
    """Per-day fuzzy government protection assessment."""

    # Per-dimension scores (0-1)
    sector_score: float = 0.0
    economic_score: float = 0.0
    policy_score: float = 0.0

    # Aggregated protection degree (fuzzy OR = max)
    protection_degree: float = 0.0

    # Human-readable label
    label: str = "unprotected"

    def to_dict(self) -> dict[str, Any]:
        return {
            "sector_score": round(self.sector_score, 4),
            "economic_score": round(self.economic_score, 4),
            "policy_score": round(self.policy_score, 4),
            "protection_degree": round(self.protection_degree, 4),
            "label": self.label,
        }


def _label_from_degree(degree: float) -> str:
    """Map protection degree to a human-readable label."""
    if degree >= 0.8:
        return "strongly_protected"
    if degree >= 0.5:
        return "moderately_protected"
    if degree >= 0.25:
        return "weakly_protected"
    return "unprotected"


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def compute_fuzzy_protection(
    cache: pd.DataFrame,
    sector: str | None = None,
    gdp: float | None = None,
    parent_sector: str | None = None,
) -> pd.DataFrame:
    """Compute daily fuzzy government protection scores.

    Parameters
    ----------
    cache:
        Daily feature table with at least ``market_cap`` column.
    sector:
        Target company sector (from verified profile).
    gdp:
        Country GDP in USD (from macro API data).
    parent_sector:
        Parent company's sector (from GLEIF corporate structure).
        If provided, the target inherits 70% of the parent's sector
        strategicness as a protection floor.

    Returns
    -------
    pd.DataFrame
        Input DataFrame augmented with:
        - ``fuzzy_sector_score``
        - ``fuzzy_economic_score``
        - ``fuzzy_policy_score``
        - ``fuzzy_protection_degree`` (0-1, replaces binary flag)
        - ``fuzzy_protection_label``
        - ``fuzzy_parent_protection_floor`` (if parent_sector provided)
    """
    result = cache.copy()

    # Sector strategicness (constant for all days)
    sector_score = _sector_membership(sector)
    result["fuzzy_sector_score"] = sector_score

    # Economic significance (varies daily with market_cap)
    if "market_cap" in result.columns and gdp is not None:
        result["fuzzy_economic_score"] = result["market_cap"].apply(
            lambda mc: _economic_significance(mc, gdp) if pd.notna(mc) else 0.0
        )
    else:
        result["fuzzy_economic_score"] = 0.0

    # Policy responsiveness (from lending/policy rate changes)
    # Check if we have rate data in the cache
    rate_col = None
    for col_name in ("lending_interest_rate", "policy_rate", "real_interest_rate"):
        if col_name in result.columns:
            rate_col = col_name
            break

    if rate_col is not None:
        # Compute 3-month (~63 business days) rate change
        rate_change = result[rate_col] - result[rate_col].shift(63)
        # Negative change = rate cut
        result["fuzzy_policy_score"] = rate_change.apply(
            lambda rc: _policy_responsiveness(-rc) if pd.notna(rc) else 0.0
        )
    else:
        result["fuzzy_policy_score"] = 0.0

    # Aggregation: prefer scikit-fuzzy Mamdani rule engine (captures
    # dimension interactions), fall back to fuzzy OR (max) if unavailable.
    _used_skfuzzy = False
    try:
        import skfuzzy as fuzz
        from skfuzzy import control as ctrl

        # Define fuzzy variables (antecedents + consequent)
        sector_var = ctrl.Antecedent(np.arange(0, 1.01, 0.01), "sector")
        economic_var = ctrl.Antecedent(np.arange(0, 1.01, 0.01), "economic")
        policy_var = ctrl.Antecedent(np.arange(0, 1.01, 0.01), "policy")
        protection_var = ctrl.Consequent(np.arange(0, 1.01, 0.01), "protection")

        # Membership functions for each antecedent
        for var in (sector_var, economic_var, policy_var):
            var["low"] = fuzz.trimf(var.universe, [0, 0, 0.4])
            var["medium"] = fuzz.trimf(var.universe, [0.2, 0.5, 0.8])
            var["high"] = fuzz.trimf(var.universe, [0.6, 1.0, 1.0])

        protection_var["unprotected"] = fuzz.trimf(protection_var.universe, [0, 0, 0.3])
        protection_var["weak"] = fuzz.trimf(protection_var.universe, [0.1, 0.35, 0.55])
        protection_var["moderate"] = fuzz.trimf(protection_var.universe, [0.35, 0.55, 0.75])
        protection_var["strong"] = fuzz.trimf(protection_var.universe, [0.6, 0.8, 1.0])

        # Rules capturing dimension interactions
        rules = [
            ctrl.Rule(sector_var["high"] & economic_var["high"], protection_var["strong"]),
            ctrl.Rule(sector_var["high"] & economic_var["medium"], protection_var["strong"]),
            ctrl.Rule(sector_var["high"] & economic_var["low"], protection_var["moderate"]),
            ctrl.Rule(sector_var["medium"] & economic_var["high"], protection_var["strong"]),
            ctrl.Rule(sector_var["medium"] & economic_var["medium"], protection_var["moderate"]),
            ctrl.Rule(sector_var["medium"] & economic_var["low"], protection_var["weak"]),
            ctrl.Rule(sector_var["low"] & economic_var["high"], protection_var["moderate"]),
            ctrl.Rule(sector_var["low"] & economic_var["medium"], protection_var["weak"]),
            ctrl.Rule(sector_var["low"] & economic_var["low"], protection_var["unprotected"]),
            ctrl.Rule(policy_var["high"], protection_var["moderate"]),
            ctrl.Rule(sector_var["low"] & policy_var["high"], protection_var["moderate"]),
        ]

        protection_ctrl = ctrl.ControlSystem(rules)
        sim = ctrl.ControlSystemSimulation(protection_ctrl)

        # Evaluate row-by-row
        degrees = []
        for idx in range(len(result)):
            s = float(result["fuzzy_sector_score"].iloc[idx])
            e = float(result["fuzzy_economic_score"].iloc[idx])
            p = float(result["fuzzy_policy_score"].iloc[idx])
            try:
                sim.input["sector"] = np.clip(s, 0.001, 0.999)
                sim.input["economic"] = np.clip(e, 0.001, 0.999)
                sim.input["policy"] = np.clip(p, 0.001, 0.999)
                sim.compute()
                degrees.append(float(sim.output["protection"]))
            except Exception:
                degrees.append(max(s, e, p))  # fallback per-row

        result["fuzzy_protection_degree"] = degrees
        _used_skfuzzy = True
        logger.info("Fuzzy protection: using scikit-fuzzy Mamdani rule engine")
    except ImportError:
        logger.debug("scikit-fuzzy not installed, using fuzzy OR fallback")
    except Exception as exc:
        logger.debug("scikit-fuzzy rule engine failed, using fallback: %s", exc)

    if not _used_skfuzzy:
        # Fallback: fuzzy OR (element-wise maximum across dimensions)
        result["fuzzy_protection_degree"] = result[
            ["fuzzy_sector_score", "fuzzy_economic_score", "fuzzy_policy_score"]
        ].max(axis=1)

    # Parent protection inheritance: if the target is a subsidiary of a
    # strategically important parent, it inherits 70% of the parent's
    # sector protection score as a floor.  A subsidiary of a defense
    # contractor or systemically important bank inherits protection.
    if parent_sector:
        parent_protection = _sector_membership(parent_sector)
        parent_floor = parent_protection * 0.7
        if parent_floor > 0.1:
            result["fuzzy_parent_protection_floor"] = parent_floor
            # Apply floor: protection degree cannot be lower than parent floor
            result["fuzzy_protection_degree"] = result["fuzzy_protection_degree"].clip(
                lower=parent_floor
            )
            logger.info(
                "Parent protection inheritance: parent_sector=%s, "
                "parent_score=%.2f, floor=%.2f",
                parent_sector, parent_protection, parent_floor,
            )

    # Label
    result["fuzzy_protection_label"] = result["fuzzy_protection_degree"].apply(
        _label_from_degree
    )

    # Also update the binary flag for backward compatibility
    result["country_protected_flag"] = (
        result["fuzzy_protection_degree"] >= 0.5
    ).astype(int)

    logger.info(
        "Fuzzy protection: sector=%.2f, mean_economic=%.2f, mean_policy=%.2f, "
        "mean_degree=%.2f",
        sector_score,
        result["fuzzy_economic_score"].mean(),
        result["fuzzy_policy_score"].mean(),
        result["fuzzy_protection_degree"].mean(),
    )

    return result
