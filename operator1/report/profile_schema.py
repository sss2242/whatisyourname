"""Profile schema validation for Operator 1.

Defines the expected structure of the company profile dict that flows
between the profile builder (G1) and the report generator (H1).

Adding a new section to the pipeline requires:
1. Adding it to REQUIRED_PROFILE_KEYS here
2. Building it in profile_builder.py
3. Consuming it in report_generator.py

The validate() function catches mismatches at runtime rather than
letting them surface as silent "No data available" sections in the report.
"""

from __future__ import annotations

import logging
from typing import Any

logger = logging.getLogger(__name__)

# Every profile dict must contain these top-level keys.
# Each key maps to the module that produces it.
REQUIRED_PROFILE_KEYS: dict[str, str] = {
    "meta": "profile_builder (always present)",
    "identity": "_build_identity_section()",
    "current_state": "_build_current_state_section()",
    "historical": "_build_historical_section()",
    "survival": "_build_survival_section()",
    "survival_episodes": "_build_survival_episodes()",
    "vanity": "_build_vanity_section()",
    "linked_entities": "_build_linked_section()",
    "regimes": "_build_regime_section()",
    "predictions": "_build_predictions_section()",
    "monte_carlo": "_build_monte_carlo_section()",
    "model_metrics": "_build_model_metrics_section()",
    "filters": "_build_ethical_filters_section()",
    "graph_risk": "graph_risk_result or {available: False}",
    "game_theory": "game_theory_result or {available: False}",
    "fuzzy_protection": "fuzzy_protection_result or {available: False}",
    "pid_controller": "pid_summary or {available: False}",
    "financial_health": "_build_financial_health_section()",
    "sentiment": "_build_sentiment_section()",
    "peer_ranking": "_build_peer_ranking_section()",
    "macro_quadrant": "_build_macro_quadrant_section()",
    "conflict_risk": "_build_conflict_risk_profile_section()",
    "data_quality": "_build_data_quality_section()",
    "estimation": "_build_estimation_section()",
    "failed_modules": "_build_failed_modules_section()",
}

# Optional keys that may be present but are not required.
OPTIONAL_PROFILE_KEYS: set[str] = {
    "extended_models",
    "ohlc_predictions",
    "filing_calendar",
    "economic_plane",
    "linked_conflict",
    "conformal_intervals",
    "shap_explanations",
    "historical_analogs",
    "patterns",
    "cycle_decomposition",
    "institutional_holders",
}


def validate_profile(profile: dict[str, Any]) -> list[str]:
    """Validate that a profile dict contains all required keys.

    Parameters
    ----------
    profile:
        Company profile dict from ``build_company_profile()``.

    Returns
    -------
    List of issue strings. Empty list means the profile is valid.
    """
    issues: list[str] = []

    for key, producer in REQUIRED_PROFILE_KEYS.items():
        if key not in profile:
            issues.append(f"Missing required key '{key}' (produced by: {producer})")
        elif profile[key] is None:
            issues.append(f"Key '{key}' is None (produced by: {producer})")

    # Check that available sections have the 'available' flag
    for key in ("graph_risk", "game_theory", "fuzzy_protection",
                "pid_controller", "conflict_risk"):
        section = profile.get(key, {})
        if isinstance(section, dict) and "available" not in section:
            issues.append(
                f"Section '{key}' is missing 'available' flag "
                f"(should be True/False for optional sections)"
            )

    if issues:
        logger.warning(
            "Profile validation found %d issues:\n  %s",
            len(issues), "\n  ".join(issues),
        )
    else:
        logger.debug("Profile validation passed (%d keys present).", len(profile))

    return issues


def validate_profile_strict(profile: dict[str, Any]) -> None:
    """Validate profile and raise ValueError if invalid.

    Use this in test assertions.
    """
    issues = validate_profile(profile)
    if issues:
        raise ValueError(
            f"Profile validation failed with {len(issues)} issues:\n"
            + "\n".join(f"  - {i}" for i in issues)
        )
