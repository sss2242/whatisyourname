"""Phase 6 tests -- T6.3 Monte Carlo simulations.

Tests the regime-aware Monte Carlo simulation engine:
  - Regime distribution estimation
  - Transition matrix computation
  - Survival trigger checking
  - Path simulation (nominal and importance-sampled)
  - Variable evolution
  - Bootstrap survival probability
  - Full pipeline (run_monte_carlo)
  - Integration with regime detection and forecasting
"""

from __future__ import annotations

import unittest

import numpy as np
import pandas as pd


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_daily_index(days: int = 250) -> pd.DatetimeIndex:
    return pd.bdate_range(start="2023-01-02", periods=days, name="date")


def _make_mc_cache(days: int = 250, *, seed: int = 42) -> pd.DataFrame:
    """Build a synthetic daily cache with regimes and survival variables.

    Creates two distinct regimes:
      - First half: bull (positive returns, low vol)
      - Second half: bear (negative returns, high vol)
    """
    np.random.seed(seed)
    idx = _make_daily_index(days)
    mid = days // 2

    returns = np.concatenate([
        np.random.randn(mid) * 0.005 + 0.001,   # bull regime
        np.random.randn(days - mid) * 0.03 - 0.002,  # bear regime
    ])

    close = 100 + np.cumsum(returns * 100)
    vol = np.concatenate([
        np.full(mid, 0.08) + np.random.randn(mid) * 0.01,
        np.full(days - mid, 0.25) + np.random.randn(days - mid) * 0.02,
    ])

    regime_labels = np.concatenate([
        np.full(mid, "bull"),
        np.full(days - mid, "bear"),
    ])

    df = pd.DataFrame({
        "close": close,
        "return_1d": returns,
        "volatility_21d": vol,
        "regime_label": regime_labels,
        "current_ratio": 1.5 + np.cumsum(np.random.randn(days) * 0.005),
        "debt_to_equity_abs": 1.0 + np.cumsum(np.random.randn(days) * 0.003),
        "fcf_yield": 0.05 + np.cumsum(np.random.randn(days) * 0.001),
        "drawdown_252d": np.minimum(
            np.cumsum(np.random.randn(days) * 0.005),
            0.0,
        ) - 0.10,
    }, index=idx)

    return df


def _make_stressed_cache(days: int = 100, *, seed: int = 42) -> pd.DataFrame:
    """Cache where survival triggers are near breach thresholds."""
    np.random.seed(seed)
    idx = _make_daily_index(days)

    df = pd.DataFrame({
        "return_1d": np.random.randn(days) * 0.02,
        "regime_label": np.full(days, "bear"),
        "current_ratio": np.full(days, 1.05),  # close to 1.0 trigger
        "debt_to_equity_abs": np.full(days, 2.8),  # close to 3.0 trigger
        "fcf_yield": np.full(days, 0.01),  # close to 0.0 trigger
        "drawdown_252d": np.full(days, -0.35),  # close to -0.40 trigger
    }, index=idx)

    return df


# ===========================================================================
# Regime distribution estimation
# ===========================================================================


class TestEstimateRegimeDistributions(unittest.TestCase):
    """Test regime distribution parameter estimation."""

    def test_basic_two_regimes(self):
        from operator1.models.monte_carlo import estimate_regime_distributions

        np.random.seed(42)
        n = 200
        returns = np.concatenate([
            np.random.randn(100) * 0.005 + 0.001,  # bull
            np.random.randn(100) * 0.03 - 0.002,    # bear
        ])
        labels = np.array(["bull"] * 100 + ["bear"] * 100)

        dists = estimate_regime_distributions(returns, labels)

        self.assertIn("bull", dists)
        self.assertIn("bear", dists)

        # Bull should have positive mean, lower std.
        self.assertGreater(dists["bull"].mean, 0)
        self.assertLess(dists["bull"].std, dists["bear"].std)

        # Bear should have negative mean (or close to it).
        self.assertLess(dists["bear"].mean, dists["bull"].mean)

        # Observation counts should be correct.
        self.assertEqual(dists["bull"].n_obs, 100)
        self.assertEqual(dists["bear"].n_obs, 100)

    def test_explicit_regime_list(self):
        from operator1.models.monte_carlo import estimate_regime_distributions

        returns = np.random.randn(50) * 0.01
        labels = np.array(["a"] * 50)

        dists = estimate_regime_distributions(
            returns, labels, unique_regimes=["a", "b"],
        )

        self.assertIn("a", dists)
        self.assertIn("b", dists)
        # "b" has no observations, should use fallback.
        self.assertEqual(dists["b"].n_obs, 50)  # falls back to overall

    def test_handles_nan_returns(self):
        from operator1.models.monte_carlo import estimate_regime_distributions

        returns = np.array([0.01, np.nan, 0.02, 0.03, np.nan] * 20)
        labels = np.array(["a", "a", "a", "b", "b"] * 20)

        dists = estimate_regime_distributions(returns, labels)
        # Should not crash, and should have both regimes.
        self.assertIn("a", dists)
        self.assertIn("b", dists)

    def test_handles_nan_labels(self):
        from operator1.models.monte_carlo import estimate_regime_distributions

        returns = np.random.randn(100) * 0.01
        labels = np.array(["a"] * 50 + [np.nan] * 50, dtype=object)

        dists = estimate_regime_distributions(returns, labels)
        self.assertIn("a", dists)

    def test_all_nan_returns_fallback(self):
        from operator1.models.monte_carlo import estimate_regime_distributions

        returns = np.full(100, np.nan)
        labels = np.array(["a"] * 100)

        dists = estimate_regime_distributions(returns, labels)
        # Should use conservative defaults.
        self.assertIn("a", dists)
        self.assertEqual(dists["a"].n_obs, 0)

    def test_zero_std_handled(self):
        """Constant returns should not produce zero std."""
        from operator1.models.monte_carlo import estimate_regime_distributions

        returns = np.full(50, 0.01)
        labels = np.array(["a"] * 50)

        dists = estimate_regime_distributions(returns, labels)
        self.assertGreater(dists["a"].std, 0)


# ===========================================================================
# Transition matrix
# ===========================================================================


class TestEstimateTransitionMatrix(unittest.TestCase):
    """Test Markov transition matrix estimation."""

    def test_basic_two_regime(self):
        from operator1.models.monte_carlo import estimate_transition_matrix

        labels = np.array(["a"] * 50 + ["b"] * 50)
        matrix, order = estimate_transition_matrix(labels)

        self.assertEqual(len(order), 2)
        self.assertIn("a", order)
        self.assertIn("b", order)

        # Matrix should be row-stochastic (rows sum to 1).
        row_sums = matrix.sum(axis=1)
        np.testing.assert_allclose(row_sums, 1.0, atol=1e-10)

    def test_self_transitions_dominate(self):
        """In a stable sequence, self-transition prob should be high."""
        from operator1.models.monte_carlo import estimate_transition_matrix

        # 100 "a" followed by 100 "b" => only 1 transition a->b.
        labels = np.array(["a"] * 100 + ["b"] * 100)
        matrix, order = estimate_transition_matrix(labels)

        a_idx = order.index("a")
        # a->a should be very high, a->b should be low.
        self.assertGreater(matrix[a_idx, a_idx], 0.9)

    def test_handles_nan_labels(self):
        from operator1.models.monte_carlo import estimate_transition_matrix

        labels = np.array(["a", "a", np.nan, "b", "b"], dtype=object)
        matrix, order = estimate_transition_matrix(labels)

        row_sums = matrix.sum(axis=1)
        np.testing.assert_allclose(row_sums, 1.0, atol=1e-10)

    def test_single_regime(self):
        from operator1.models.monte_carlo import estimate_transition_matrix

        labels = np.array(["only"] * 50)
        matrix, order = estimate_transition_matrix(labels)

        self.assertEqual(matrix.shape, (1, 1))
        self.assertAlmostEqual(matrix[0, 0], 1.0)

    def test_smoothing_prevents_zero_probs(self):
        """Laplace smoothing should prevent zero transition probabilities."""
        from operator1.models.monte_carlo import estimate_transition_matrix

        # Only transitions a->a and b->b, never cross.
        labels = np.array(["a"] * 50 + ["b"] * 50)
        matrix, order = estimate_transition_matrix(labels, smoothing=1.0)

        # Cross-transitions should still be > 0 due to smoothing.
        for i in range(len(order)):
            for j in range(len(order)):
                self.assertGreater(matrix[i, j], 0)


# ===========================================================================
# Survival trigger checking
# ===========================================================================


class TestCheckSurvivalTriggers(unittest.TestCase):
    """Test survival trigger breach detection."""

    def test_no_breach(self):
        from operator1.models.monte_carlo import check_survival_triggers

        values = {
            "current_ratio": 1.5,
            "debt_to_equity_abs": 1.0,
            "fcf_yield": 0.05,
            "drawdown_252d": -0.10,
        }
        self.assertFalse(check_survival_triggers(values))

    def test_current_ratio_breach(self):
        from operator1.models.monte_carlo import check_survival_triggers

        values = {
            "current_ratio": 0.8,  # < 1.0
            "debt_to_equity_abs": 1.0,
            "fcf_yield": 0.05,
            "drawdown_252d": -0.10,
        }
        self.assertTrue(check_survival_triggers(values))

    def test_debt_equity_breach(self):
        from operator1.models.monte_carlo import check_survival_triggers

        values = {
            "current_ratio": 1.5,
            "debt_to_equity_abs": 3.5,  # > 3.0
            "fcf_yield": 0.05,
            "drawdown_252d": -0.10,
        }
        self.assertTrue(check_survival_triggers(values))

    def test_fcf_yield_breach(self):
        from operator1.models.monte_carlo import check_survival_triggers

        values = {
            "current_ratio": 1.5,
            "debt_to_equity_abs": 1.0,
            "fcf_yield": -0.02,  # < 0
            "drawdown_252d": -0.10,
        }
        self.assertTrue(check_survival_triggers(values))

    def test_drawdown_breach(self):
        from operator1.models.monte_carlo import check_survival_triggers

        values = {
            "current_ratio": 1.5,
            "debt_to_equity_abs": 1.0,
            "fcf_yield": 0.05,
            "drawdown_252d": -0.50,  # < -0.40
        }
        self.assertTrue(check_survival_triggers(values))

    def test_all_breach(self):
        from operator1.models.monte_carlo import check_survival_triggers

        values = {
            "current_ratio": 0.5,
            "debt_to_equity_abs": 5.0,
            "fcf_yield": -0.1,
            "drawdown_252d": -0.60,
        }
        self.assertTrue(check_survival_triggers(values))

    def test_missing_variable_ignored(self):
        from operator1.models.monte_carlo import check_survival_triggers

        # Only safe values provided.
        values = {"current_ratio": 1.5}
        self.assertFalse(check_survival_triggers(values))

    def test_nan_value_ignored(self):
        from operator1.models.monte_carlo import check_survival_triggers

        values = {"current_ratio": float("nan")}
        self.assertFalse(check_survival_triggers(values))

    def test_custom_thresholds(self):
        from operator1.models.monte_carlo import check_survival_triggers

        custom = {"custom_var": ("gt", 10.0)}
        values = {"custom_var": 15.0}
        self.assertTrue(check_survival_triggers(values, custom))

        values_ok = {"custom_var": 5.0}
        self.assertFalse(check_survival_triggers(values_ok, custom))

    def test_empty_values(self):
        from operator1.models.monte_carlo import check_survival_triggers

        self.assertFalse(check_survival_triggers({}))


# ===========================================================================
# Path simulation
# ===========================================================================


class TestSimulateReturnPaths(unittest.TestCase):
    """Test return path simulation."""

    def test_output_shape(self):
        from operator1.models.monte_carlo import (
            simulate_return_paths,
            RegimeDistribution,
        )

        rng = np.random.default_rng(42)
        dist = [RegimeDistribution(mean=0.001, std=0.01, n_obs=100)]
        tm = np.array([[1.0]])

        paths, lw = simulate_return_paths(
            n_paths=50, n_steps=10,
            current_regime_idx=0,
            transition_matrix=tm,
            regime_distributions=dist,
            rng=rng,
        )

        self.assertEqual(paths.shape, (50, 10))
        self.assertEqual(lw.shape, (50,))

    def test_no_tilt_zero_weights(self):
        """Without importance tilt, log weights should be 0."""
        from operator1.models.monte_carlo import (
            simulate_return_paths,
            RegimeDistribution,
        )

        rng = np.random.default_rng(42)
        dist = [RegimeDistribution(mean=0.0, std=0.01, n_obs=100)]
        tm = np.array([[1.0]])

        _, lw = simulate_return_paths(
            n_paths=100, n_steps=5,
            current_regime_idx=0,
            transition_matrix=tm,
            regime_distributions=dist,
            rng=rng,
            importance_tilt=0.0,
        )

        np.testing.assert_array_equal(lw, np.zeros(100))

    def test_tilt_produces_nonzero_weights(self):
        """With importance tilt, log weights should be non-zero."""
        from operator1.models.monte_carlo import (
            simulate_return_paths,
            RegimeDistribution,
        )

        rng = np.random.default_rng(42)
        dist = [RegimeDistribution(mean=0.001, std=0.01, n_obs=100)]
        tm = np.array([[1.0]])

        _, lw = simulate_return_paths(
            n_paths=100, n_steps=10,
            current_regime_idx=0,
            transition_matrix=tm,
            regime_distributions=dist,
            rng=rng,
            importance_tilt=1.5,
        )

        # At least some weights should be non-zero.
        self.assertTrue(np.any(lw != 0))

    def test_reproducibility(self):
        """Same seed should produce same paths."""
        from operator1.models.monte_carlo import (
            simulate_return_paths,
            RegimeDistribution,
        )

        dist = [RegimeDistribution(mean=0.0, std=0.01, n_obs=100)]
        tm = np.array([[1.0]])

        rng1 = np.random.default_rng(123)
        paths1, _ = simulate_return_paths(
            50, 10, 0, tm, dist, rng1,
        )

        rng2 = np.random.default_rng(123)
        paths2, _ = simulate_return_paths(
            50, 10, 0, tm, dist, rng2,
        )

        np.testing.assert_array_equal(paths1, paths2)

    def test_multi_regime_transitions(self):
        """Paths should use different distributions per regime."""
        from operator1.models.monte_carlo import (
            simulate_return_paths,
            RegimeDistribution,
        )

        rng = np.random.default_rng(42)
        # Two regimes: one very positive, one very negative.
        dists = [
            RegimeDistribution(mean=0.05, std=0.01, n_obs=100),
            RegimeDistribution(mean=-0.05, std=0.01, n_obs=100),
        ]
        # Equal transition probabilities.
        tm = np.array([[0.5, 0.5], [0.5, 0.5]])

        paths, _ = simulate_return_paths(
            1000, 20, 0, tm, dists, rng,
        )

        # With equal transitions, average return should be ~0.
        avg_return = paths.mean()
        self.assertAlmostEqual(avg_return, 0.0, places=1)


# ===========================================================================
# Variable evolution
# ===========================================================================


class TestEvolveVariables(unittest.TestCase):
    """Test survival variable evolution from return paths."""

    def test_basic_evolution(self):
        from operator1.models.monte_carlo import evolve_variables

        paths = np.array([[0.01, 0.02, -0.01]])  # 1 path, 3 steps
        initial = {"current_ratio": 1.5}

        result = evolve_variables(paths, initial)

        self.assertIn("current_ratio", result)
        self.assertEqual(result["current_ratio"].shape, (1, 3))

    def test_drawdown_always_negative(self):
        """Drawdown should always be <= 0."""
        from operator1.models.monte_carlo import evolve_variables

        np.random.seed(42)
        paths = np.random.randn(100, 50) * 0.02
        initial = {"drawdown_252d": -0.10}

        result = evolve_variables(paths, initial)

        self.assertTrue(np.all(result["drawdown_252d"] <= 0))

    def test_missing_variable_skipped(self):
        from operator1.models.monte_carlo import evolve_variables

        paths = np.random.randn(10, 5) * 0.01
        initial = {"current_ratio": 1.5}
        sensitivities = {
            "current_ratio": 0.5,
            "nonexistent_var": 1.0,
        }

        result = evolve_variables(paths, initial, sensitivities)

        self.assertIn("current_ratio", result)
        self.assertNotIn("nonexistent_var", result)

    def test_positive_returns_improve_current_ratio(self):
        """With default sensitivities, positive returns should increase
        current_ratio."""
        from operator1.models.monte_carlo import evolve_variables

        # All positive returns.
        paths = np.full((10, 50), 0.01)
        initial = {"current_ratio": 1.5}

        result = evolve_variables(paths, initial)

        # Terminal values should be above initial (positive returns + positive beta).
        terminal = result["current_ratio"][:, -1]
        self.assertTrue(np.all(terminal > 1.5))

    def test_negative_returns_worsen_debt_equity(self):
        """With default sensitivities, negative returns should increase
        debt_to_equity_abs (beta is negative, so negative returns
        increase the ratio)."""
        from operator1.models.monte_carlo import evolve_variables

        # All negative returns.
        paths = np.full((10, 50), -0.01)
        initial = {"debt_to_equity_abs": 1.0}

        result = evolve_variables(paths, initial)

        # Terminal values should be above initial (negative beta * negative returns = positive).
        terminal = result["debt_to_equity_abs"][:, -1]
        self.assertTrue(np.all(terminal > 1.0))


# ===========================================================================
# Run simulation
# ===========================================================================


class TestRunSimulation(unittest.TestCase):
    """Test the core simulation engine."""

    def test_returns_correct_shapes(self):
        from operator1.models.monte_carlo import (
            run_simulation,
            RegimeDistribution,
        )

        rng = np.random.default_rng(42)
        dist = [RegimeDistribution(mean=0.001, std=0.01, n_obs=100)]
        tm = np.array([[1.0]])
        initial = {
            "current_ratio": 1.5,
            "debt_to_equity_abs": 1.0,
            "fcf_yield": 0.05,
            "drawdown_252d": -0.10,
        }

        survival, weights, ess = run_simulation(
            n_paths=100,
            horizon_steps=21,
            current_regime_idx=0,
            transition_matrix=tm,
            regime_distributions=dist,
            initial_values=initial,
            rng=rng,
        )

        self.assertEqual(len(survival), 100)
        self.assertEqual(len(weights), 100)
        self.assertGreater(ess, 0)

    def test_survival_flags_binary(self):
        from operator1.models.monte_carlo import (
            run_simulation,
            RegimeDistribution,
        )

        rng = np.random.default_rng(42)
        dist = [RegimeDistribution(mean=0.0, std=0.01, n_obs=100)]
        tm = np.array([[1.0]])
        initial = {"current_ratio": 1.5}

        survival, _, _ = run_simulation(
            n_paths=50,
            horizon_steps=10,
            current_regime_idx=0,
            transition_matrix=tm,
            regime_distributions=dist,
            initial_values=initial,
            rng=rng,
        )

        # All values should be 0 or 1.
        unique_vals = set(survival)
        self.assertTrue(unique_vals.issubset({0.0, 1.0}))

    def test_weights_normalized(self):
        from operator1.models.monte_carlo import (
            run_simulation,
            RegimeDistribution,
        )

        rng = np.random.default_rng(42)
        dist = [RegimeDistribution(mean=0.0, std=0.02, n_obs=100)]
        tm = np.array([[1.0]])
        initial = {"current_ratio": 1.5}

        _, weights, _ = run_simulation(
            n_paths=200,
            horizon_steps=10,
            current_regime_idx=0,
            transition_matrix=tm,
            regime_distributions=dist,
            initial_values=initial,
            rng=rng,
        )

        # Weights should sum to ~1.
        self.assertAlmostEqual(weights.sum(), 1.0, places=5)

    def test_safe_initial_values_high_survival(self):
        """With very safe initial values and low vol, most paths should
        survive."""
        from operator1.models.monte_carlo import (
            run_simulation,
            RegimeDistribution,
        )

        rng = np.random.default_rng(42)
        dist = [RegimeDistribution(mean=0.001, std=0.001, n_obs=100)]
        tm = np.array([[1.0]])
        initial = {
            "current_ratio": 5.0,
            "debt_to_equity_abs": 0.2,
            "fcf_yield": 0.10,
            "drawdown_252d": -0.01,
        }

        survival, weights, _ = run_simulation(
            n_paths=500,
            horizon_steps=21,
            current_regime_idx=0,
            transition_matrix=tm,
            regime_distributions=dist,
            initial_values=initial,
            rng=rng,
            importance_fraction=0.0,  # no IS for this test
        )

        surv_prob = np.dot(weights, survival)
        self.assertGreater(surv_prob, 0.5)

    def test_importance_sampling_activates(self):
        """With importance fraction > 0, importance sampling should be used."""
        from operator1.models.monte_carlo import (
            run_simulation,
            RegimeDistribution,
        )

        rng = np.random.default_rng(42)
        dist = [RegimeDistribution(mean=0.0, std=0.01, n_obs=100)]
        tm = np.array([[1.0]])
        initial = {"current_ratio": 1.5}

        survival, weights, ess = run_simulation(
            n_paths=200,
            horizon_steps=10,
            current_regime_idx=0,
            transition_matrix=tm,
            regime_distributions=dist,
            initial_values=initial,
            rng=rng,
            importance_fraction=0.3,
            importance_tilt=1.5,
        )

        # ESS should be less than n_paths when IS is active.
        self.assertLess(ess, 200)
        self.assertGreater(ess, 0)


# ===========================================================================
# Bootstrap survival probability
# ===========================================================================


class TestBootstrapSurvivalProbability(unittest.TestCase):
    """Test bootstrap confidence interval computation."""

    def test_all_survive(self):
        from operator1.models.monte_carlo import bootstrap_survival_probability

        rng = np.random.default_rng(42)
        survival = np.ones(100)
        weights = np.ones(100) / 100

        stats = bootstrap_survival_probability(survival, weights, 500, rng)

        self.assertAlmostEqual(stats["mean"], 1.0, places=2)
        self.assertAlmostEqual(stats["p5"], 1.0, places=2)

    def test_none_survive(self):
        from operator1.models.monte_carlo import bootstrap_survival_probability

        rng = np.random.default_rng(42)
        survival = np.zeros(100)
        weights = np.ones(100) / 100

        stats = bootstrap_survival_probability(survival, weights, 500, rng)

        self.assertAlmostEqual(stats["mean"], 0.0, places=2)
        self.assertAlmostEqual(stats["p95"], 0.0, places=2)

    def test_mixed_survival(self):
        from operator1.models.monte_carlo import bootstrap_survival_probability

        rng = np.random.default_rng(42)
        survival = np.array([1.0] * 70 + [0.0] * 30)
        weights = np.ones(100) / 100

        stats = bootstrap_survival_probability(survival, weights, 1000, rng)

        # Mean should be close to 0.7.
        self.assertAlmostEqual(stats["mean"], 0.7, places=1)
        # p5 < mean < p95.
        self.assertLessEqual(stats["p5"], stats["mean"])
        self.assertGreaterEqual(stats["p95"], stats["mean"])

    def test_ci_width_reasonable(self):
        """Confidence interval should be narrower with more data."""
        from operator1.models.monte_carlo import bootstrap_survival_probability

        rng = np.random.default_rng(42)

        # Small sample.
        survival_small = np.array([1.0] * 7 + [0.0] * 3)
        weights_small = np.ones(10) / 10
        stats_small = bootstrap_survival_probability(
            survival_small, weights_small, 1000, rng,
        )
        width_small = stats_small["p95"] - stats_small["p5"]

        # Large sample.
        rng = np.random.default_rng(42)
        survival_large = np.array([1.0] * 700 + [0.0] * 300)
        weights_large = np.ones(1000) / 1000
        stats_large = bootstrap_survival_probability(
            survival_large, weights_large, 1000, rng,
        )
        width_large = stats_large["p95"] - stats_large["p5"]

        # Larger sample should have narrower CI.
        self.assertLess(width_large, width_small)

    def test_empty_input(self):
        from operator1.models.monte_carlo import bootstrap_survival_probability

        rng = np.random.default_rng(42)
        stats = bootstrap_survival_probability(
            np.array([]), np.array([]), 100, rng,
        )
        self.assertTrue(np.isnan(stats["mean"]))

    def test_returns_all_keys(self):
        from operator1.models.monte_carlo import bootstrap_survival_probability

        rng = np.random.default_rng(42)
        survival = np.ones(50)
        weights = np.ones(50) / 50

        stats = bootstrap_survival_probability(survival, weights, 100, rng)

        expected_keys = {"mean", "std", "p5", "p25", "median", "p75", "p95"}
        self.assertEqual(set(stats.keys()), expected_keys)


# ===========================================================================
# Initial value extraction
# ===========================================================================


class TestExtractInitialValues(unittest.TestCase):
    """Test initial variable value extraction from cache."""

    def test_extracts_latest_values(self):
        from operator1.models.monte_carlo import extract_initial_values

        cache = _make_mc_cache(100)
        values = extract_initial_values(cache)

        self.assertIn("current_ratio", values)
        self.assertIn("debt_to_equity_abs", values)
        # Should be the last non-NaN value.
        self.assertFalse(np.isnan(values["current_ratio"]))

    def test_custom_variables(self):
        from operator1.models.monte_carlo import extract_initial_values

        cache = _make_mc_cache(50)
        values = extract_initial_values(cache, ["current_ratio"])

        self.assertIn("current_ratio", values)
        self.assertNotIn("fcf_yield", values)

    def test_missing_column(self):
        from operator1.models.monte_carlo import extract_initial_values

        idx = pd.bdate_range("2024-01-02", periods=10)
        cache = pd.DataFrame({"other": np.random.randn(10)}, index=idx)

        values = extract_initial_values(cache)
        # No survival variables in cache.
        self.assertEqual(len(values), 0)


# ===========================================================================
# Current regime detection
# ===========================================================================


class TestDetectCurrentRegime(unittest.TestCase):
    """Test current regime detection from cache."""

    def test_returns_last_regime(self):
        from operator1.models.monte_carlo import detect_current_regime

        cache = _make_mc_cache(100)
        regime = detect_current_regime(cache)

        # Last half is "bear" in our synthetic data.
        self.assertEqual(regime, "bear")

    def test_no_regime_column(self):
        from operator1.models.monte_carlo import detect_current_regime

        idx = pd.bdate_range("2024-01-02", periods=10)
        cache = pd.DataFrame({"close": np.random.randn(10)}, index=idx)

        regime = detect_current_regime(cache)
        self.assertEqual(regime, "unknown")

    def test_all_nan_regime(self):
        from operator1.models.monte_carlo import detect_current_regime

        idx = pd.bdate_range("2024-01-02", periods=10)
        cache = pd.DataFrame({
            "regime_label": [np.nan] * 10,
        }, index=idx)

        regime = detect_current_regime(cache)
        self.assertEqual(regime, "unknown")


# ===========================================================================
# MonteCarloResult container
# ===========================================================================


class TestMonteCarloResult(unittest.TestCase):
    """Test the MonteCarloResult dataclass."""

    def test_default_state(self):
        from operator1.models.monte_carlo import MonteCarloResult

        r = MonteCarloResult()
        self.assertEqual(r.n_paths, 0)
        self.assertFalse(r.fitted)
        self.assertTrue(np.isnan(r.survival_probability_mean))
        self.assertEqual(r.survival_probability, {})
        self.assertIsNone(r.error)


# ===========================================================================
# Full pipeline: run_monte_carlo
# ===========================================================================


class TestRunMonteCarlo(unittest.TestCase):
    """Test the top-level Monte Carlo pipeline."""

    def test_returns_result(self):
        from operator1.models.monte_carlo import run_monte_carlo, MonteCarloResult

        cache = _make_mc_cache(200)
        result = run_monte_carlo(cache, n_paths=100, n_bootstrap=50)

        self.assertIsInstance(result, MonteCarloResult)
        self.assertTrue(result.fitted)
        self.assertIsNone(result.error)

    def test_survival_probabilities_per_horizon(self):
        from operator1.models.monte_carlo import (
            run_monte_carlo,
            DEFAULT_HORIZONS,
        )

        # Use 500 rows (>= 400 threshold) so DEFAULT_HORIZONS keys are used
        cache = _make_mc_cache(500)
        result = run_monte_carlo(cache, n_paths=100, n_bootstrap=50)

        # Should have probabilities for all default horizons.
        for h_label in DEFAULT_HORIZONS:
            self.assertIn(h_label, result.survival_probability)
            prob = result.survival_probability[h_label]
            self.assertGreaterEqual(prob, 0.0)
            self.assertLessEqual(prob, 1.0)

    def test_survival_stats_per_horizon(self):
        from operator1.models.monte_carlo import (
            run_monte_carlo,
            DEFAULT_HORIZONS,
        )

        # Use 500 rows (>= 400 threshold) so DEFAULT_HORIZONS keys are used
        cache = _make_mc_cache(500)
        result = run_monte_carlo(cache, n_paths=100, n_bootstrap=50)

        for h_label in DEFAULT_HORIZONS:
            self.assertIn(h_label, result.survival_stats)
            stats = result.survival_stats[h_label]
            self.assertIn("mean", stats)
            self.assertIn("p5", stats)
            self.assertIn("p95", stats)
            # p5 <= mean <= p95.
            self.assertLessEqual(stats["p5"], stats["mean"] + 0.01)
            self.assertGreaterEqual(stats["p95"], stats["mean"] - 0.01)

    def test_summary_stats(self):
        from operator1.models.monte_carlo import run_monte_carlo

        cache = _make_mc_cache(200)
        result = run_monte_carlo(cache, n_paths=100, n_bootstrap=50)

        self.assertFalse(np.isnan(result.survival_probability_mean))
        self.assertFalse(np.isnan(result.survival_probability_p5))
        self.assertFalse(np.isnan(result.survival_probability_p95))

        # Mean should be in [0, 1].
        self.assertGreaterEqual(result.survival_probability_mean, 0.0)
        self.assertLessEqual(result.survival_probability_mean, 1.0)

    def test_regime_distributions_populated(self):
        from operator1.models.monte_carlo import run_monte_carlo

        cache = _make_mc_cache(200)
        result = run_monte_carlo(cache, n_paths=50, n_bootstrap=50)

        self.assertGreater(len(result.regime_distributions), 0)
        # Our synthetic data has "bull" and "bear".
        self.assertIn("bear", result.regime_distributions)
        self.assertIn("bull", result.regime_distributions)

    def test_transition_matrix_present(self):
        from operator1.models.monte_carlo import run_monte_carlo

        cache = _make_mc_cache(200)
        result = run_monte_carlo(cache, n_paths=50, n_bootstrap=50)

        self.assertIsNotNone(result.transition_matrix)
        # Row-stochastic.
        row_sums = result.transition_matrix.sum(axis=1)
        np.testing.assert_allclose(row_sums, 1.0, atol=1e-10)

    def test_current_regime_detected(self):
        from operator1.models.monte_carlo import run_monte_carlo

        cache = _make_mc_cache(200)
        result = run_monte_carlo(cache, n_paths=50, n_bootstrap=50)

        self.assertEqual(result.current_regime, "bear")

    def test_importance_sampling_flag(self):
        from operator1.models.monte_carlo import run_monte_carlo

        cache = _make_mc_cache(100)

        # With IS.
        result_is = run_monte_carlo(
            cache,
            n_paths=50,
            importance_fraction=0.3,
            n_bootstrap=50,
        )
        self.assertTrue(result_is.importance_sampling_used)

        # Without IS.
        result_no_is = run_monte_carlo(
            cache,
            n_paths=50,
            importance_fraction=0.0,
            n_bootstrap=50,
        )
        self.assertFalse(result_no_is.importance_sampling_used)

    def test_reproducibility(self):
        """Same seed should produce same results."""
        from operator1.models.monte_carlo import run_monte_carlo

        cache = _make_mc_cache(100)

        r1 = run_monte_carlo(cache, n_paths=100, random_state=42, n_bootstrap=50)
        r2 = run_monte_carlo(cache, n_paths=100, random_state=42, n_bootstrap=50)

        self.assertAlmostEqual(
            r1.survival_probability_mean,
            r2.survival_probability_mean,
            places=10,
        )

        for h in r1.survival_probability:
            self.assertAlmostEqual(
                r1.survival_probability[h],
                r2.survival_probability[h],
                places=10,
            )

    def test_custom_horizons(self):
        from operator1.models.monte_carlo import run_monte_carlo

        cache = _make_mc_cache(100)
        custom = {"3d": 3, "10d": 10}

        result = run_monte_carlo(
            cache, n_paths=50, horizons=custom, n_bootstrap=50,
        )

        self.assertIn("3d", result.survival_probability)
        self.assertIn("10d", result.survival_probability)
        self.assertNotIn("1d", result.survival_probability)

    def test_missing_returns_column(self):
        from operator1.models.monte_carlo import run_monte_carlo

        idx = pd.bdate_range("2024-01-02", periods=50)
        cache = pd.DataFrame({"close": np.random.randn(50) + 100}, index=idx)

        result = run_monte_carlo(cache, n_paths=50, n_bootstrap=50)

        self.assertFalse(result.fitted)
        self.assertIsNotNone(result.error)

    def test_no_regime_column_uses_fallback(self):
        """Without regime labels, should use a single 'unknown' regime."""
        from operator1.models.monte_carlo import run_monte_carlo

        idx = pd.bdate_range("2024-01-02", periods=100)
        np.random.seed(42)
        cache = pd.DataFrame({
            "return_1d": np.random.randn(100) * 0.01,
            "current_ratio": np.full(100, 1.5),
            "debt_to_equity_abs": np.full(100, 1.0),
            "fcf_yield": np.full(100, 0.05),
            "drawdown_252d": np.full(100, -0.10),
        }, index=idx)

        result = run_monte_carlo(cache, n_paths=100, n_bootstrap=50)

        self.assertTrue(result.fitted)
        self.assertEqual(result.current_regime, "unknown")

    def test_stressed_cache_lower_survival(self):
        """With near-breach initial values, survival should be lower than
        safe values."""
        from operator1.models.monte_carlo import run_monte_carlo

        safe = _make_mc_cache(200)
        stressed = _make_stressed_cache(200)

        r_safe = run_monte_carlo(safe, n_paths=500, n_bootstrap=50, random_state=42)
        r_stressed = run_monte_carlo(stressed, n_paths=500, n_bootstrap=50, random_state=42)

        # Stressed should have lower (or equal) survival at the longest horizon.
        # Use 252d which amplifies differences.
        if "252d" in r_safe.survival_probability and "252d" in r_stressed.survival_probability:
            self.assertLessEqual(
                r_stressed.survival_probability["252d"],
                r_safe.survival_probability["252d"] + 0.1,  # small tolerance
            )

    def test_n_paths_recorded(self):
        from operator1.models.monte_carlo import run_monte_carlo

        cache = _make_mc_cache(100)
        result = run_monte_carlo(cache, n_paths=200, n_bootstrap=50)

        self.assertEqual(result.n_paths, 200)
        self.assertEqual(result.n_paths_importance, 60)  # 30% of 200


# ===========================================================================
# Integration: regime detection + monte carlo
# ===========================================================================


class TestRegimeAndMonteCarloIntegration(unittest.TestCase):
    """Integration test running regime detection then Monte Carlo."""

    def test_pipeline_sequence(self):
        """Run regime detection followed by Monte Carlo on the same cache."""
        from operator1.models.regime_detector import detect_regimes_and_breaks
        from operator1.models.monte_carlo import run_monte_carlo

        cache = _make_mc_cache(200)

        # Step 1: Regime detection.
        cache, detector = detect_regimes_and_breaks(cache, n_regimes=2)
        self.assertIn("regime_label", cache.columns)

        # Step 2: Monte Carlo.
        result = run_monte_carlo(cache, n_paths=100, n_bootstrap=50)

        self.assertTrue(result.fitted)
        self.assertGreater(len(result.survival_probability), 0)

    def test_regime_columns_survive_monte_carlo(self):
        """Monte Carlo should not remove regime columns."""
        from operator1.models.regime_detector import detect_regimes_and_breaks
        from operator1.models.monte_carlo import run_monte_carlo

        cache = _make_mc_cache(150)
        cache, _ = detect_regimes_and_breaks(cache, n_regimes=2)
        cols_before = set(cache.columns)

        run_monte_carlo(cache, n_paths=50, n_bootstrap=50)

        cols_after = set(cache.columns)
        self.assertTrue(cols_before.issubset(cols_after))


class TestFullPhase6Integration(unittest.TestCase):
    """Integration test running all three Phase 6 modules in sequence."""

    def test_regime_then_forecast_then_monte_carlo(self):
        """Full T6.1 -> T6.2 -> T6.3 pipeline."""
        from operator1.models.regime_detector import detect_regimes_and_breaks
        from operator1.models.forecasting import run_forecasting
        from operator1.models.monte_carlo import run_monte_carlo

        cache = _make_mc_cache(200)

        # T6.1: Regime detection.
        cache, detector = detect_regimes_and_breaks(cache, n_regimes=2)

        # T6.2: Forecasting.
        cache, forecast_result = run_forecasting(
            cache,
            ["current_ratio", "debt_to_equity_abs"],
            enable_burnout=False,
        )

        # T6.3: Monte Carlo.
        mc_result = run_monte_carlo(cache, n_paths=100, n_bootstrap=50)

        # All should succeed.
        self.assertIn("regime_label", cache.columns)
        self.assertGreater(len(forecast_result.forecasts), 0)
        self.assertTrue(mc_result.fitted)
        self.assertGreater(len(mc_result.survival_probability), 0)


# ===========================================================================
# Constants
# ===========================================================================


class TestConstants(unittest.TestCase):
    """Test module-level constants."""

    def test_default_n_paths(self):
        from operator1.models.monte_carlo import DEFAULT_N_PATHS
        self.assertEqual(DEFAULT_N_PATHS, 10_000)

    def test_default_horizons(self):
        from operator1.models.monte_carlo import DEFAULT_HORIZONS
        self.assertIn("1d", DEFAULT_HORIZONS)
        self.assertIn("252d", DEFAULT_HORIZONS)

    def test_default_survival_thresholds(self):
        from operator1.models.monte_carlo import DEFAULT_SURVIVAL_THRESHOLDS
        self.assertIn("current_ratio", DEFAULT_SURVIVAL_THRESHOLDS)
        self.assertIn("debt_to_equity_abs", DEFAULT_SURVIVAL_THRESHOLDS)
        self.assertIn("fcf_yield", DEFAULT_SURVIVAL_THRESHOLDS)
        self.assertIn("drawdown_252d", DEFAULT_SURVIVAL_THRESHOLDS)


if __name__ == "__main__":
    unittest.main()
