"""Unified Survival System -- Central Regime Controller.

A single coherent survival system that merges the original spec concept with
insights from regime-switching theory, banking stress tests, military OODA,
medical triage, Knightian uncertainty, and graceful degradation.

Core Principle: Survival mode is a **discrete computational regime change**.
When activated, the pipeline reconfigures itself across five dimensions:

1. Variable Triage -- active/frozen/priority classification
2. Model Switching -- regime-aware parameter reconfiguration
3. Horizon Compression -- shortened forecast horizons
4. Correlation Switching -- crisis copula replacement
5. Forecast Bounding -- hard bounds on survival-mode predictions

The controller computes a regime timeline from cache data and provides
configuration to all downstream modules via a unified interface.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Any

import pandas as pd

from operator1.config_loader import load_config

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Regime enum and constants
# ---------------------------------------------------------------------------

REGIMES = ("normal", "company_survival", "modified_survival", "extreme_survival")

# Tier membership for variable triage
_TIER_MEMBERSHIP: dict[str, int] | None = None


def _get_tier_membership() -> dict[str, int]:
    """Load variable -> tier mapping from config (cached)."""
    global _TIER_MEMBERSHIP
    if _TIER_MEMBERSHIP is not None:
        return _TIER_MEMBERSHIP

    try:
        cfg = load_config("survival_hierarchy")
        tiers = cfg.get("tiers", {})
        membership: dict[str, int] = {}
        for tier_key, tier_data in tiers.items():
            tier_num = int(tier_key.replace("tier", ""))
            for var in tier_data.get("variables", []):
                membership[var] = tier_num
        _TIER_MEMBERSHIP = membership
    except Exception:
        _TIER_MEMBERSHIP = {}

    return _TIER_MEMBERSHIP


# ---------------------------------------------------------------------------
# Dimension 1: Variable Triage
# ---------------------------------------------------------------------------

# Triage status per (regime, tier) combination
# "active" = full computation, "priority" = extra resources, "frozen" = carry forward
# TODO: Tier 3 in company_survival and extreme_survival should enforce
# "reduced" model subset (Kalman + baseline only, skip LSTM/transformer/tree)
# per the architecture spec.  Currently all models run for Tier 3 "active".
_TRIAGE_TABLE: dict[str, dict[int, str]] = {
    "normal": {1: "active", 2: "active", 3: "active", 4: "active", 5: "active"},
    "company_survival": {1: "priority", 2: "priority", 3: "active", 4: "frozen", 5: "frozen"},
    "modified_survival": {1: "priority", 2: "priority", 3: "active", 4: "frozen", 5: "frozen"},
    "extreme_survival": {1: "priority", 2: "priority", 3: "active", 4: "frozen", 5: "frozen"},
}


@dataclass
class VariableTriageResult:
    """Classification of all variables into active/frozen/priority."""

    active_variables: list[str] = field(default_factory=list)
    priority_variables: list[str] = field(default_factory=list)
    frozen_variables: list[str] = field(default_factory=list)
    triage_map: dict[str, str] = field(default_factory=dict)


def triage_variables(regime: str, all_variables: list[str]) -> VariableTriageResult:
    """Classify variables into active/frozen/priority based on regime.

    From medical triage: classify variables into categories based on
    the survival regime to allocate computational resources.

    Parameters
    ----------
    regime:
        Current survival regime.
    all_variables:
        All variable names in the cache.

    Returns
    -------
    VariableTriageResult
        Classification of variables.
    """
    tier_map = _get_tier_membership()
    regime_triage = _TRIAGE_TABLE.get(regime, _TRIAGE_TABLE["normal"])

    result = VariableTriageResult()

    for var in all_variables:
        tier = tier_map.get(var, 3)  # default to tier 3 (active) for unmapped
        status = regime_triage.get(tier, "active")
        result.triage_map[var] = status

        if status == "priority":
            result.priority_variables.append(var)
            result.active_variables.append(var)  # priority is a subset of active
        elif status == "active":
            result.active_variables.append(var)
        elif status == "frozen":
            result.frozen_variables.append(var)

    return result


# ---------------------------------------------------------------------------
# Dimension 2: Model Switching
# ---------------------------------------------------------------------------

@dataclass
class ModelConfig:
    """Regime-specific model parameters for downstream wrappers."""

    # Kalman filter overrides
    kalman_process_noise_multiplier: float = 1.0
    kalman_observation_noise_multiplier: float = 1.0

    # LSTM / transformer overrides
    nn_lookback_cap: int | None = None  # None = use default
    nn_learning_rate_multiplier: float = 1.0

    # Monte Carlo overrides
    mc_n_paths: int = 10_000
    mc_stress_percentile: float = 0.95  # 95th for normal, 99th for crisis
    mc_regime_filter: str | None = None  # None = all, "crisis" = crisis-regime only

    # Tree ensemble overrides
    tree_max_depth_cap: int | None = None

    # General
    extra_variables_allowed: bool = True  # False in extreme survival


_MODEL_CONFIGS: dict[str, ModelConfig] = {
    "normal": ModelConfig(),
    "company_survival": ModelConfig(
        kalman_process_noise_multiplier=3.0,
        kalman_observation_noise_multiplier=2.0,
        nn_lookback_cap=10,
        nn_learning_rate_multiplier=2.0,
        mc_n_paths=20_000,
        mc_stress_percentile=0.99,
        mc_regime_filter="crisis",
        tree_max_depth_cap=4,
    ),
    "modified_survival": ModelConfig(
        kalman_process_noise_multiplier=2.0,
        kalman_observation_noise_multiplier=1.5,
        nn_lookback_cap=15,
        nn_learning_rate_multiplier=1.5,
        mc_n_paths=15_000,
        mc_stress_percentile=0.97,
    ),
    "extreme_survival": ModelConfig(
        kalman_process_noise_multiplier=5.0,
        kalman_observation_noise_multiplier=3.0,
        nn_lookback_cap=5,
        nn_learning_rate_multiplier=3.0,
        mc_n_paths=30_000,
        mc_stress_percentile=0.99,
        mc_regime_filter="crisis",
        tree_max_depth_cap=3,
        extra_variables_allowed=False,
    ),
}


def get_model_config(regime: str) -> ModelConfig:
    """Return model configuration for the given regime."""
    return _MODEL_CONFIGS.get(regime, _MODEL_CONFIGS["normal"])


def get_soft_transition_config(
    current_regime: str,
    previous_regime: str | None = None,
    days_since_switch: int = 0,
    halflife: int = 5,
) -> ModelConfig:
    """Return model config with gradual transition between regimes.

    Instead of hard-switching all parameters simultaneously when a regime
    changes, interpolates between old and new configs over a transition
    window. This prevents discontinuities in model behavior.

    Parameters
    ----------
    current_regime:
        Active regime label.
    previous_regime:
        Previous regime label (before the switch). When None or same as
        current_regime, returns the current config unchanged.
    days_since_switch:
        Number of days since the regime switch occurred.
    halflife:
        Transition half-life in days (default 5). At halflife days,
        parameters are 50% transitioned; at 2*halflife, 75%; at
        3*halflife, 87.5%.

    Returns
    -------
    ModelConfig
        Interpolated config (or current if no transition needed).
    """
    import numpy as np
    from dataclasses import asdict

    current_config = get_model_config(current_regime)

    if previous_regime is None or previous_regime == current_regime or days_since_switch <= 0:
        return current_config

    previous_config = get_model_config(previous_regime)

    # Exponential transition: lambda goes from 0 (old config) to 1 (new config)
    lam = float(1.0 - np.exp(-days_since_switch * np.log(2) / max(halflife, 1)))

    # Interpolate numeric fields only
    current_d = asdict(current_config)
    previous_d = asdict(previous_config)
    blended = {}
    for key in current_d:
        cv, pv = current_d[key], previous_d[key]
        if isinstance(cv, (int, float)) and isinstance(pv, (int, float)):
            blended[key] = (1 - lam) * pv + lam * cv
        else:
            # Non-numeric (strings, None): switch immediately to current
            blended[key] = cv

    return ModelConfig(**blended)


# ---------------------------------------------------------------------------
# Dimension 3: Horizon Compression
# ---------------------------------------------------------------------------

# From Boyd's OODA loop: compress the decision tempo in crisis.
_HORIZON_TABLE: dict[str, dict[str, int]] = {
    "normal": {"1d": 1, "5d": 5, "21d": 21, "252d": 252},
    "company_survival": {"1d": 1, "5d": 5, "21d": 21},
    "modified_survival": {"1d": 1, "5d": 5, "21d": 21, "252d": 252},
    "extreme_survival": {"1d": 1, "5d": 5},
}

_PRIMARY_HORIZON: dict[str, str] = {
    "normal": "252d",
    "company_survival": "5d",
    "modified_survival": "21d",
    "extreme_survival": "1d",
}


def get_active_horizons(regime: str) -> dict[str, int]:
    """Return active forecast horizons for the given regime.

    In extreme_survival, there is no point forecasting 252 days out.
    Every computational resource goes toward predicting what happens
    THIS WEEK.
    """
    return _HORIZON_TABLE.get(regime, _HORIZON_TABLE["normal"])


def get_primary_horizon(regime: str) -> str:
    """Return the primary (most important) horizon for the given regime."""
    return _PRIMARY_HORIZON.get(regime, "252d")


# ---------------------------------------------------------------------------
# Dimension 4: Correlation Switching
# ---------------------------------------------------------------------------

def get_crisis_correlation_override(regime: str) -> float | None:
    """Return correlation override for crisis regimes.

    From Basel stress testing: normal-mode correlations are dangerously
    optimistic during crisis. When survival activates, assume high
    correlation between all linked entities.

    Returns None for normal mode (use empirical correlations).
    Returns a float for crisis modes (override all pairwise correlations).
    """
    if regime == "extreme_survival":
        return 0.90
    if regime in ("company_survival", "modified_survival"):
        return 0.85
    return None


def get_copula_type_override(regime: str) -> str | None:
    """Return copula type override for crisis regimes.

    In crisis, replace Gaussian copula with Student-t or Clayton
    (lower tail dependence) for more realistic tail modeling.

    Returns None for normal mode (use best-fit AIC selection).
    """
    if regime in ("company_survival", "extreme_survival"):
        return "clayton"  # Lower tail dependence
    if regime == "modified_survival":
        return "student_t"  # Fat tails
    return None


# ---------------------------------------------------------------------------
# Dimension 5: Forecast Bounding
# ---------------------------------------------------------------------------

def bound_survival_forecast(
    variable: str,
    forecast: float,
    cache: pd.DataFrame,
    regime: str,
) -> float:
    """Apply hard bounds to forecasts in survival mode.

    From stress testing practice: in survival mode, apply conservative
    bounds to prevent unrealistic optimistic forecasts.

    Parameters
    ----------
    variable:
        Cache column name being forecast.
    forecast:
        Raw point forecast value.
    cache:
        Daily cache DataFrame.
    regime:
        Current survival regime.

    Returns
    -------
    float
        Bounded forecast value.
    """
    if regime not in ("company_survival", "modified_survival", "extreme_survival"):
        return forecast

    if variable not in cache.columns:
        return forecast

    last_series = cache[variable].dropna()
    if last_series.empty:
        return forecast

    last_actual = float(last_series.iloc[-1])

    # Revenue can't grow in survival mode -- cap at last actual
    if variable == "revenue":
        return min(forecast, last_actual)

    # Cash can only decline at most by the max observed burn rate
    if variable == "cash_and_equivalents":
        if "operating_cash_flow" in cache.columns:
            ocf = cache["operating_cash_flow"].dropna()
            if not ocf.empty:
                max_burn = float(ocf.min())  # Most negative OCF
                return max(forecast, last_actual + max_burn)
        return forecast

    # Debt doesn't decrease in crisis (no refinancing available)
    if variable in ("total_debt", "short_term_debt", "long_term_debt"):
        return max(forecast, last_actual)

    # Gross margin can't improve during crisis -- cap at last actual
    if variable in ("gross_margin", "operating_margin", "net_margin"):
        return min(forecast, last_actual)

    # Current ratio can't improve without new equity/refinancing
    if variable == "current_ratio":
        return min(forecast, last_actual)

    return forecast


def bound_forecast_dict(
    forecasts: dict[str, dict[str, float]],
    cache: pd.DataFrame,
    regime: str,
) -> dict[str, dict[str, float]]:
    """Apply survival bounds to all forecasts in a nested dict.

    Parameters
    ----------
    forecasts:
        {variable: {horizon: point_forecast}} nested dict.
    cache:
        Daily cache DataFrame.
    regime:
        Current survival regime.

    Returns
    -------
    Bounded forecasts dict (same structure).
    """
    if regime == "normal":
        return forecasts

    bounded: dict[str, dict[str, float]] = {}
    for var, horizons in forecasts.items():
        bounded[var] = {}
        if isinstance(horizons, dict):
            for h, val in horizons.items():
                try:
                    bounded[var][h] = bound_survival_forecast(
                        var, float(val), cache, regime,
                    )
                except (TypeError, ValueError):
                    bounded[var][h] = val
        else:
            bounded[var] = horizons
    return bounded


# ---------------------------------------------------------------------------
# Early Warning System (Transition Zone)
# ---------------------------------------------------------------------------

def compute_early_warning_score(cache: pd.DataFrame) -> pd.Series:
    """Compute continuous early warning score (0-1) for proximity to survival triggers.

    The one place where continuous scoring is useful: the boundary region.
    When current_ratio is 1.05, you are not in survival mode yet, but close.

    This does NOT change the computational mode -- it is informational only.
    The actual switch happens when the binary flag flips.
    """
    scores: list[pd.Series] = []

    # Current ratio proximity (trigger at 1.0)
    if "current_ratio" in cache.columns:
        cr = cache["current_ratio"].fillna(2.0)
        # Score: 0 when cr >= 2.0, 1.0 when cr <= 0.5
        cr_score = (2.0 - cr.clip(0.5, 2.0)) / 1.5
        scores.append(cr_score)

    # Debt-to-equity proximity (trigger at 3.0)
    if "debt_to_equity_abs" in cache.columns:
        de = cache["debt_to_equity_abs"].fillna(0.0)
        # Score: 0 when de <= 1.0, 1.0 when de >= 4.0
        de_score = (de.clip(1.0, 4.0) - 1.0) / 3.0
        scores.append(de_score)

    # FCF yield proximity (trigger at 0.0)
    if "fcf_yield" in cache.columns:
        fy = cache["fcf_yield"].fillna(0.1)
        # Score: 0 when fy >= 0.1, 1.0 when fy <= -0.1
        fy_score = (0.1 - fy.clip(-0.1, 0.1)) / 0.2
        scores.append(fy_score)

    # Drawdown proximity (trigger at -0.40)
    if "drawdown_252d" in cache.columns:
        dd = cache["drawdown_252d"].fillna(0.0)
        # Score: 0 when dd >= -0.15, 1.0 when dd <= -0.50
        dd_score = (-0.15 - dd.clip(-0.50, -0.15)) / 0.35
        scores.append(dd_score)

    if not scores:
        return pd.Series(0.0, index=cache.index, name="early_warning_score")

    # Take the maximum across all trigger proximities
    combined = pd.concat(scores, axis=1).max(axis=1)
    combined.name = "early_warning_score"
    return combined.clip(0.0, 1.0)


# ---------------------------------------------------------------------------
# SurvivalRegimeController -- central orchestrator
# ---------------------------------------------------------------------------

@dataclass
class SurvivalRegimeController:
    """Central controller that determines the current regime and configures
    all downstream modules.

    Attributes
    ----------
    regime_timeline : pd.Series
        Per-day regime label (index = cache DatetimeIndex).
    current_regime : str
        Latest regime label.
    early_warning : pd.Series
        Per-day early warning score (0-1).
    triage : VariableTriageResult
        Variable classification for current regime.
    model_config : ModelConfig
        Model parameters for current regime.
    horizons : dict[str, int]
        Active forecast horizons for current regime.
    primary_horizon : str
        Most important horizon for current regime.
    correlation_override : float | None
        Crisis correlation override (None = use empirical).
    copula_override : str | None
        Crisis copula type override (None = use AIC selection).
    is_survival : bool
        True if current regime is any survival mode.
    regime_switches : int
        Number of regime switches in the timeline.
    """

    regime_timeline: pd.Series = field(default_factory=lambda: pd.Series(dtype=str))
    current_regime: str = "normal"
    early_warning: pd.Series = field(default_factory=lambda: pd.Series(dtype=float))
    triage: VariableTriageResult = field(default_factory=VariableTriageResult)
    model_config: ModelConfig = field(default_factory=ModelConfig)
    horizons: dict[str, int] = field(default_factory=lambda: {"1d": 1, "5d": 5, "21d": 21, "252d": 252})
    primary_horizon: str = "252d"
    correlation_override: float | None = None
    copula_override: str | None = None
    is_survival: bool = False
    regime_switches: int = 0

    @classmethod
    def from_cache(cls, cache: pd.DataFrame) -> SurvivalRegimeController:
        """Build controller from a populated daily cache.

        The cache must already have survival flags computed (Step 5 in main.py):
        - company_survival_mode_flag
        - country_survival_mode_flag (optional)
        - country_protected_flag (optional)

        Parameters
        ----------
        cache:
            Daily cache DataFrame with survival flags.

        Returns
        -------
        SurvivalRegimeController
            Fully configured controller.
        """
        controller = cls()

        # Compute regime timeline
        try:
            from operator1.analysis.hierarchy_weights import select_regime_series
            controller.regime_timeline = select_regime_series(cache)
        except Exception as exc:
            logger.warning("Regime timeline computation failed: %s", exc)
            controller.regime_timeline = pd.Series(
                "normal", index=cache.index, name="survival_regime",
            )

        # Current regime = latest non-null value
        non_null = controller.regime_timeline.dropna()
        if not non_null.empty:
            controller.current_regime = str(non_null.iloc[-1])
        else:
            controller.current_regime = "normal"

        controller.is_survival = controller.current_regime != "normal"

        # Count regime switches
        if len(controller.regime_timeline) > 1:
            shifted = controller.regime_timeline.shift(1)
            controller.regime_switches = int(
                (controller.regime_timeline != shifted).sum() - 1
            )

        # Compute early warning
        controller.early_warning = compute_early_warning_score(cache)

        # Configure all 5 dimensions based on current regime
        all_vars = list(cache.columns)
        controller.triage = triage_variables(controller.current_regime, all_vars)
        controller.model_config = get_model_config(controller.current_regime)
        controller.horizons = get_active_horizons(controller.current_regime)
        controller.primary_horizon = get_primary_horizon(controller.current_regime)
        controller.correlation_override = get_crisis_correlation_override(
            controller.current_regime,
        )
        controller.copula_override = get_copula_type_override(controller.current_regime)

        logger.info(
            "SurvivalRegimeController: regime=%s, survival=%s, "
            "switches=%d, active_vars=%d, frozen_vars=%d, "
            "horizons=%s, primary=%s, corr_override=%s",
            controller.current_regime,
            controller.is_survival,
            controller.regime_switches,
            len(controller.triage.active_variables),
            len(controller.triage.frozen_variables),
            list(controller.horizons.keys()),
            controller.primary_horizon,
            controller.correlation_override,
        )

        return controller

    def get_regime(self, t: int | pd.Timestamp | None = None) -> str:
        """Get regime at a specific time index.

        Parameters
        ----------
        t:
            Integer positional index or Timestamp. None = latest.
        """
        if t is None:
            return self.current_regime

        if isinstance(t, int):
            if 0 <= t < len(self.regime_timeline):
                return str(self.regime_timeline.iloc[t])
            return self.current_regime

        if t in self.regime_timeline.index:
            return str(self.regime_timeline.loc[t])
        return self.current_regime

    def get_active_variables(self) -> list[str]:
        """Return only variables that should be computed (not frozen)."""
        return self.triage.active_variables

    def get_frozen_variables(self) -> list[str]:
        """Return variables that should carry forward last value."""
        return self.triage.frozen_variables

    def get_priority_variables(self) -> list[str]:
        """Return variables that deserve extra computational resources."""
        return self.triage.priority_variables

    def is_variable_frozen(self, variable: str) -> bool:
        """Check if a variable is frozen in the current regime."""
        return self.triage.triage_map.get(variable, "active") == "frozen"

    def should_forecast_horizon(self, horizon: str) -> bool:
        """Check if a horizon should be forecast in the current regime."""
        return horizon in self.horizons

    def bound_forecast(
        self,
        variable: str,
        forecast: float,
        cache: pd.DataFrame,
    ) -> float:
        """Apply survival-mode bounds to a single forecast."""
        return bound_survival_forecast(
            variable, forecast, cache, self.current_regime,
        )

    def bound_all_forecasts(
        self,
        forecasts: dict[str, dict[str, float]],
        cache: pd.DataFrame,
    ) -> dict[str, dict[str, float]]:
        """Apply survival-mode bounds to all forecasts."""
        return bound_forecast_dict(forecasts, cache, self.current_regime)

    def get_early_warning_latest(self) -> float:
        """Return the latest early warning score."""
        if self.early_warning.empty:
            return 0.0
        return float(self.early_warning.iloc[-1])

    def is_approaching_survival(self, threshold: float = 0.7) -> bool:
        """Check if early warning indicates approaching survival mode."""
        return self.get_early_warning_latest() >= threshold

    def detect_recovery_signal(self) -> dict[str, Any]:
        """Detect regime transitions from survival -> normal.

        Companies exiting distress dramatically outperform the market.
        This method flags such transitions as a high-conviction alpha signal.

        Returns
        -------
        dict with recovery signal metadata:
            - ``active``: True if a survival-to-normal transition detected
            - ``transition_from``: previous regime (e.g. "company_survival")
            - ``transition_to``: current regime (e.g. "normal")
            - ``days_since_recovery``: business days since the transition
            - ``conviction``: 0-1 score based on recovery speed and depth
            - ``position_signal_boost``: multiplier for position signal
        """
        result = {
            "active": False,
            "transition_from": "",
            "transition_to": "",
            "days_since_recovery": 0,
            "conviction": 0.0,
            "position_signal_boost": 1.0,
        }

        if self.regime_timeline.empty or len(self.regime_timeline) < 10:
            return result

        timeline = self.regime_timeline
        current = str(timeline.iloc[-1])

        # Look for survival -> normal transition in recent history
        _survival_regimes = {"company_survival", "extreme_survival", "modified_survival"}
        if current in _survival_regimes:
            return result  # still in survival, no recovery

        # Scan backward for the most recent survival regime
        for i in range(len(timeline) - 2, max(len(timeline) - 252, -1), -1):
            if i < 0:
                break
            regime_at_i = str(timeline.iloc[i])
            if regime_at_i in _survival_regimes:
                days_since = len(timeline) - 1 - i
                # Recovery conviction based on how recently the transition happened
                # Strongest signal: 0-30 days post-recovery
                # Decays with half-life of 63 days
                import math
                decay = math.exp(-0.693 * days_since / 63)
                conviction = min(1.0, decay)

                # Boost if recovery was from extreme_survival (deeper V)
                depth_boost = 1.0
                if regime_at_i == "extreme_survival":
                    depth_boost = 1.5
                elif regime_at_i == "company_survival":
                    depth_boost = 1.2

                result = {
                    "active": conviction > 0.1,
                    "transition_from": regime_at_i,
                    "transition_to": current,
                    "days_since_recovery": days_since,
                    "conviction": round(conviction * min(depth_boost, 1.0), 4),
                    "position_signal_boost": round(1.0 + 0.5 * conviction * depth_boost, 3),
                }
                break

        return result

    def to_profile_dict(self) -> dict[str, Any]:
        """Export controller state for inclusion in company profile."""
        regime_distribution = {}
        if not self.regime_timeline.empty:
            counts = self.regime_timeline.value_counts(normalize=True)
            regime_distribution = {str(k): round(float(v), 4) for k, v in counts.items()}

        return {
            "available": True,
            "current_regime": self.current_regime,
            "is_survival": self.is_survival,
            "regime_switches": self.regime_switches,
            "regime_distribution": regime_distribution,
            "active_horizons": list(self.horizons.keys()),
            "primary_horizon": self.primary_horizon,
            "n_active_variables": len(self.triage.active_variables),
            "n_frozen_variables": len(self.triage.frozen_variables),
            "n_priority_variables": len(self.triage.priority_variables),
            "correlation_override": self.correlation_override,
            "copula_override": self.copula_override,
            "early_warning_latest": self.get_early_warning_latest(),
            "approaching_survival": self.is_approaching_survival(),
            "recovery_signal": self.detect_recovery_signal(),
            "model_config": {
                "kalman_process_noise_mult": self.model_config.kalman_process_noise_multiplier,
                "kalman_obs_noise_mult": self.model_config.kalman_observation_noise_multiplier,
                "nn_lookback_cap": self.model_config.nn_lookback_cap,
                "nn_lr_mult": self.model_config.nn_learning_rate_multiplier,
                "mc_n_paths": self.model_config.mc_n_paths,
                "mc_stress_percentile": self.model_config.mc_stress_percentile,
            },
        }
