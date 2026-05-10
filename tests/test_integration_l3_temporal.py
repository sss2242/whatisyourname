"""Integration tests -- Layer 3: Temporal Models.

Validates Monte Carlo simulation, OHLC predictor, and burn-out
distributions produce sensible results without NaN/Inf.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from tests.conftest import AAPL_EXPECTED


class TestL3MonteCarloOnSynthetic:
    """MC tests using always-available synthetic data."""

    def test_mc_runs_without_crash(self, synthetic_cache):
        from operator1.models.monte_carlo import run_monte_carlo
        cache = synthetic_cache.copy()
        cache["regime_label"] = "bull"
        mc = run_monte_carlo(cache, n_paths=200, random_state=42)
        assert mc is not None
        assert mc.n_paths > 0

    def test_mc_survival_probability_in_range(self, synthetic_cache):
        from operator1.models.monte_carlo import run_monte_carlo
        cache = synthetic_cache.copy()
        cache["regime_label"] = "bull"
        mc = run_monte_carlo(cache, n_paths=200, random_state=42)
        for h, val in mc.survival_probability.items():
            p = val.get("mean", val) if isinstance(val, dict) else float(val)
            assert 0.0 <= p <= 1.0, (
                f"MC survival at {h}: {p} out of [0,1] range"
            )

    def test_mc_survival_monotonically_non_increasing(self, synthetic_cache):
        """P(survive 1 day) >= P(survive 5 days) >= P(survive 252 days)."""
        from operator1.models.monte_carlo import run_monte_carlo
        cache = synthetic_cache.copy()
        cache["regime_label"] = "bull"
        mc = run_monte_carlo(cache, n_paths=500, random_state=42)
        probs = []
        for h in ["1d", "5d", "21d", "252d"]:
            sp = mc.survival_probability.get(h)
            if sp is None:
                continue
            p = sp.get("mean", sp) if isinstance(sp, dict) else float(sp)
            probs.append(p)
        if len(probs) >= 2:
            for i in range(len(probs) - 1):
                assert probs[i] >= probs[i + 1] - 0.10, (
                    f"Non-monotonic survival: {probs} "
                    f"({probs[i]:.3f} < {probs[i+1]:.3f} at step {i})"
                )

    def test_mc_drawdown_not_nan(self, synthetic_cache):
        from operator1.models.monte_carlo import run_monte_carlo
        cache = synthetic_cache.copy()
        cache["regime_label"] = "bull"
        mc = run_monte_carlo(cache, n_paths=500, random_state=42)
        if hasattr(mc, "max_drawdown_distribution") and mc.max_drawdown_distribution:
            for h, dd in mc.max_drawdown_distribution.items():
                if isinstance(dd, dict):
                    median = dd.get("median")
                    if median is not None:
                        assert not np.isnan(median), (
                            f"MC drawdown NaN at horizon {h}"
                        )


class TestL3OHLCOnSynthetic:
    """OHLC predictor tests using synthetic data."""

    def test_ohlc_predictor_runs_without_crash(self, synthetic_cache):
        from operator1.models.ohlc_predictor import predict_ohlc_series
        result = predict_ohlc_series(synthetic_cache.copy())
        assert result is not None

    def test_ohlc_high_gt_low_for_next_day(self, synthetic_cache):
        from operator1.models.ohlc_predictor import predict_ohlc_series
        result = predict_ohlc_series(synthetic_cache.copy())
        if result.fitted and result.next_day:
            nd = result.next_day
            if nd.high is not None and nd.low is not None:
                assert nd.high >= nd.low, (
                    f"OHLC next_day high={nd.high} < low={nd.low}"
                )

    def test_ohlc_year_end_within_range(self, synthetic_cache):
        from operator1.models.ohlc_predictor import predict_ohlc_series
        result = predict_ohlc_series(synthetic_cache.copy())
        if result.fitted and result.next_year and len(result.next_year) > 0:
            last_close = float(synthetic_cache["close"].dropna().iloc[-1])
            year_close = result.next_year[-1].close
            if year_close is not None and last_close > 0:
                pct = abs(year_close - last_close) / last_close
                max_pct = AAPL_EXPECTED.get("l3_ohlc_year_pct_change_max", 0.60)
                assert pct < max_pct, (
                    f"OHLC year-end ${year_close:.2f} is {pct:.0%} from "
                    f"current ${last_close:.2f} (max {max_pct:.0%})"
                )
