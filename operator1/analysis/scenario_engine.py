"""Unified Survival System -- Scenario Engine.

From Knightian uncertainty: in survival mode, point forecasts are unreliable
because the data-generating process has changed. Replace with scenario analysis.

Three scenarios computed via Monte Carlo with different distribution assumptions:

1. Orderly Resolution -- management executes restructuring
2. Muddle Through -- status quo continues
3. Catastrophic -- fire sale conditions

Each scenario produces: cash runway in days, probability of surviving 90 days,
probability of surviving 252 days, and terminal equity value.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Any

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

# Number of simulation paths per scenario
_DEFAULT_N_PATHS = 5_000
_DEFAULT_HORIZON_DAYS = 252


@dataclass
class ScenarioResult:
    """Result for a single scenario."""

    name: str = ""
    description: str = ""
    cash_runway_days: float = 0.0
    survival_prob_90d: float = 0.0
    survival_prob_252d: float = 0.0
    terminal_equity_value: float | None = None
    terminal_revenue: float | None = None
    terminal_cash: float | None = None
    median_return: float = 0.0
    p5_return: float = 0.0
    p95_return: float = 0.0
    max_drawdown_median: float = 0.0


@dataclass
class ScenarioEngineResult:
    """Container for all 3 scenario results."""

    available: bool = False
    regime: str = "normal"
    orderly: ScenarioResult = field(default_factory=ScenarioResult)
    muddle_through: ScenarioResult = field(default_factory=ScenarioResult)
    catastrophic: ScenarioResult = field(default_factory=ScenarioResult)
    n_paths: int = _DEFAULT_N_PATHS
    horizon_days: int = _DEFAULT_HORIZON_DAYS
    error: str = ""

    def to_dict(self) -> dict[str, Any]:
        """Export for profile inclusion."""
        if not self.available:
            return {"available": False, "error": self.error}

        def _sr(s: ScenarioResult) -> dict:
            return {
                "name": s.name,
                "description": s.description,
                "cash_runway_days": round(s.cash_runway_days, 1),
                "survival_prob_90d": round(s.survival_prob_90d, 4),
                "survival_prob_252d": round(s.survival_prob_252d, 4),
                "terminal_equity_value": s.terminal_equity_value,
                "terminal_revenue": s.terminal_revenue,
                "terminal_cash": s.terminal_cash,
                "median_return": round(s.median_return, 6),
                "p5_return": round(s.p5_return, 6),
                "p95_return": round(s.p95_return, 6),
                "max_drawdown_median": round(s.max_drawdown_median, 4),
            }

        return {
            "available": True,
            "regime": self.regime,
            "n_paths": self.n_paths,
            "horizon_days": self.horizon_days,
            "orderly_resolution": _sr(self.orderly),
            "muddle_through": _sr(self.muddle_through),
            "catastrophic": _sr(self.catastrophic),
        }


def _compute_cash_runway(
    cash: float,
    monthly_burn: float,
) -> float:
    """Compute days until cash hits zero at current burn rate."""
    if monthly_burn >= 0:
        return float("inf")  # Positive cash flow = no burn
    daily_burn = monthly_burn / 21.0  # Business days per month
    if daily_burn >= 0:
        return float("inf")
    return max(0.0, -cash / daily_burn)


def _simulate_paths(
    returns: np.ndarray,
    n_paths: int,
    horizon: int,
    shift: float = 0.0,
    percentile_range: tuple[float, float] = (0.0, 1.0),
    rng: np.random.Generator | None = None,
) -> np.ndarray:
    """Simulate price paths from historical return distribution.

    Parameters
    ----------
    returns:
        Historical daily returns array.
    n_paths:
        Number of simulation paths.
    horizon:
        Number of days to simulate.
    shift:
        Constant shift applied to sampled returns (scenario-specific).
    percentile_range:
        (lower, upper) percentile bounds for return filtering.
    rng:
        Random number generator.

    Returns
    -------
    np.ndarray of shape (n_paths, horizon) -- cumulative return paths.
    """
    if rng is None:
        rng = np.random.default_rng(42)

    clean = returns[np.isfinite(returns)]
    if len(clean) < 5:
        return np.zeros((n_paths, horizon))

    # Filter to percentile range
    lo = np.percentile(clean, percentile_range[0] * 100)
    hi = np.percentile(clean, percentile_range[1] * 100)
    filtered = clean[(clean >= lo) & (clean <= hi)]
    if len(filtered) < 3:
        filtered = clean

    # Sample with replacement + shift
    sampled = rng.choice(filtered, size=(n_paths, horizon), replace=True)
    sampled += shift

    # Cumulative return paths
    cum_paths = np.cumprod(1.0 + sampled, axis=1)
    return cum_paths


def _survival_probability(
    paths: np.ndarray,
    threshold: float = -0.40,
    up_to_day: int | None = None,
) -> float:
    """Fraction of paths that never breach the drawdown threshold."""
    if paths.size == 0:
        return 0.0

    if up_to_day is not None:
        paths = paths[:, :up_to_day]

    # Running max per path
    running_max = np.maximum.accumulate(paths, axis=1)
    drawdowns = (paths - running_max) / np.maximum(running_max, 1e-10)
    # Path survives if min drawdown never goes below threshold
    min_dd = drawdowns.min(axis=1)
    return float((min_dd > threshold).mean())


def _max_drawdown_paths(paths: np.ndarray) -> np.ndarray:
    """Compute max drawdown per path."""
    if paths.size == 0:
        return np.array([0.0])
    running_max = np.maximum.accumulate(paths, axis=1)
    drawdowns = (paths - running_max) / np.maximum(running_max, 1e-10)
    return drawdowns.min(axis=1)


def run_scenario_engine(
    cache: pd.DataFrame,
    regime: str = "normal",
    n_paths: int = _DEFAULT_N_PATHS,
    horizon_days: int = _DEFAULT_HORIZON_DAYS,
) -> ScenarioEngineResult:
    """Run 3-scenario Monte Carlo simulation for survival mode.

    Parameters
    ----------
    cache:
        Daily cache with return_1d, close, cash_and_equivalents,
        operating_cash_flow, revenue, total_equity columns.
    regime:
        Current survival regime.
    n_paths:
        Number of MC paths per scenario.
    horizon_days:
        Simulation horizon in trading days.

    Returns
    -------
    ScenarioEngineResult with all 3 scenario outcomes.
    """
    result = ScenarioEngineResult(
        regime=regime,
        n_paths=n_paths,
        horizon_days=horizon_days,
    )

    # Extract historical returns
    returns_col = "return_1d"
    if returns_col not in cache.columns:
        # Try private company proxy
        returns_col = "equity_change_rate"
    if returns_col not in cache.columns:
        result.error = "No returns column available"
        return result

    returns = cache[returns_col].dropna().values
    if len(returns) < 20:
        result.error = f"Insufficient return data ({len(returns)} < 20)"
        return result

    rng = np.random.default_rng(42)

    # Extract latest financial state
    last_cash = 0.0
    if "cash_and_equivalents" in cache.columns:
        cs = cache["cash_and_equivalents"].dropna()
        if not cs.empty:
            last_cash = float(cs.iloc[-1])

    monthly_burn = 0.0
    if "operating_cash_flow" in cache.columns:
        ocf = cache["operating_cash_flow"].dropna()
        if not ocf.empty:
            monthly_burn = float(ocf.iloc[-1]) / 12.0  # Annualized -> monthly

    last_close = 1.0
    if "close" in cache.columns:
        cs = cache["close"].dropna()
        if not cs.empty:
            last_close = float(cs.iloc[-1])

    last_revenue = None
    if "revenue" in cache.columns:
        rv = cache["revenue"].dropna()
        if not rv.empty:
            last_revenue = float(rv.iloc[-1])

    last_equity = None
    if "total_equity" in cache.columns:
        eq = cache["total_equity"].dropna()
        if not eq.empty:
            last_equity = float(eq.iloc[-1])

    # ---------------------------------------------------------------
    # Scenario 1: Orderly Resolution
    # Management executes a restructuring plan.
    # Cash burn reduces 30% over 90 days. Debt renegotiated.
    # Revenue declines 15% then stabilizes.
    # MC distribution: 25th-75th percentile of returns, shifted down 10%.
    # ---------------------------------------------------------------
    orderly_paths = _simulate_paths(
        returns, n_paths, horizon_days,
        shift=-0.001,  # ~0.1% daily drag
        percentile_range=(0.25, 0.75),
        rng=rng,
    )
    orderly_burn = monthly_burn * 0.7  # 30% reduction
    orderly_cash_runway = _compute_cash_runway(last_cash, orderly_burn)

    result.orderly = ScenarioResult(
        name="Orderly Resolution",
        description=(
            "Management executes restructuring: 30% burn rate reduction, "
            "debt renegotiation, 15% revenue decline then stabilization."
        ),
        cash_runway_days=min(orderly_cash_runway, 9999),
        survival_prob_90d=_survival_probability(orderly_paths, up_to_day=min(90, horizon_days)),
        survival_prob_252d=_survival_probability(orderly_paths, up_to_day=min(252, horizon_days)),
        terminal_equity_value=(
            float(last_equity * 0.85 * float(np.median(orderly_paths[:, -1])))
            if last_equity is not None else None
        ),
        terminal_revenue=(
            float(last_revenue * 0.85) if last_revenue is not None else None
        ),
        terminal_cash=(
            float(last_cash + orderly_burn * (horizon_days / 21.0))
            if last_cash > 0 else None
        ),
        median_return=float(np.median(orderly_paths[:, -1]) - 1.0),
        p5_return=float(np.percentile(orderly_paths[:, -1], 5) - 1.0),
        p95_return=float(np.percentile(orderly_paths[:, -1], 95) - 1.0),
        max_drawdown_median=float(np.median(_max_drawdown_paths(orderly_paths))),
    )

    # ---------------------------------------------------------------
    # Scenario 2: Muddle Through
    # Status quo continues. Current burn rate sustained.
    # No debt restructuring. Revenue follows current trend.
    # MC distribution: full historical distribution (as-is).
    # ---------------------------------------------------------------
    muddle_paths = _simulate_paths(
        returns, n_paths, horizon_days,
        shift=0.0,
        percentile_range=(0.0, 1.0),
        rng=rng,
    )

    result.muddle_through = ScenarioResult(
        name="Muddle Through",
        description=(
            "Status quo continues: current burn rate sustained, "
            "no restructuring, revenue follows existing trend."
        ),
        cash_runway_days=min(_compute_cash_runway(last_cash, monthly_burn), 9999),
        survival_prob_90d=_survival_probability(muddle_paths, up_to_day=min(90, horizon_days)),
        survival_prob_252d=_survival_probability(muddle_paths, up_to_day=min(252, horizon_days)),
        terminal_equity_value=(
            float(last_equity * float(np.median(muddle_paths[:, -1])))
            if last_equity is not None else None
        ),
        terminal_revenue=last_revenue,
        terminal_cash=(
            float(last_cash + monthly_burn * (horizon_days / 21.0))
            if last_cash > 0 else None
        ),
        median_return=float(np.median(muddle_paths[:, -1]) - 1.0),
        p5_return=float(np.percentile(muddle_paths[:, -1], 5) - 1.0),
        p95_return=float(np.percentile(muddle_paths[:, -1], 95) - 1.0),
        max_drawdown_median=float(np.median(_max_drawdown_paths(muddle_paths))),
    )

    # ---------------------------------------------------------------
    # Scenario 3: Catastrophic
    # Fire sale conditions. Assets marked down 40%.
    # All debt called (acceleration clauses triggered).
    # Revenue drops 40%. Key customers leave.
    # MC distribution: 1st-10th percentile of returns (tail).
    # ---------------------------------------------------------------
    catastrophic_paths = _simulate_paths(
        returns, n_paths, horizon_days,
        shift=-0.003,  # ~0.3% daily drag (severe)
        percentile_range=(0.0, 0.10),
        rng=rng,
    )
    catastrophic_burn = monthly_burn * 1.5  # 50% worse burn

    result.catastrophic = ScenarioResult(
        name="Catastrophic",
        description=(
            "Fire sale: assets marked down 40%, all debt called, "
            "revenue drops 40%, key customers leave."
        ),
        cash_runway_days=min(_compute_cash_runway(last_cash, catastrophic_burn), 9999),
        survival_prob_90d=_survival_probability(catastrophic_paths, up_to_day=min(90, horizon_days)),
        survival_prob_252d=_survival_probability(catastrophic_paths, up_to_day=min(252, horizon_days)),
        terminal_equity_value=(
            float(last_equity * 0.60 * float(np.median(catastrophic_paths[:, -1])))
            if last_equity is not None else None
        ),
        terminal_revenue=(
            float(last_revenue * 0.60) if last_revenue is not None else None
        ),
        terminal_cash=(
            float(max(0, last_cash + catastrophic_burn * (horizon_days / 21.0)))
            if last_cash > 0 else None
        ),
        median_return=float(np.median(catastrophic_paths[:, -1]) - 1.0),
        p5_return=float(np.percentile(catastrophic_paths[:, -1], 5) - 1.0),
        p95_return=float(np.percentile(catastrophic_paths[:, -1], 95) - 1.0),
        max_drawdown_median=float(np.median(_max_drawdown_paths(catastrophic_paths))),
    )

    result.available = True
    return result
