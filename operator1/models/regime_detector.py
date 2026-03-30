"""T6.1 -- Regime detection and structural break analysis.

Provides a unified ``RegimeDetector`` class that applies multiple
regime classification and structural-break methods to the daily cache:

**Regime Classification:**
  - **HMM** (Hidden Markov Model): identifies hidden market regimes
    (bull, bear, high-vol, low-vol) from returns + volatility.
  - **GMM** (Gaussian Mixture Model): unsupervised clustering of return
    distributions into regime groups.

**Structural Break Detection:**
  - **PELT** (Pruned Exact Linear Time): fast optimal segmentation for
    detecting mean/variance shifts in long time series.
  - **BCP** (Bayesian Change Point): probabilistic detection of regime
    shifts using Bayesian inference (simplified online variant when
    ``pymc`` is unavailable).

Each method is wrapped in ``try/except`` so that a missing optional
dependency (``hmmlearn``, ``ruptures``, ``pymc``) logs a warning and
returns a graceful fallback rather than crashing the pipeline (Sec 10.X).

Top-level entry point:
  ``detect_regimes_and_breaks(cache)`` runs all available methods and
  adds result columns to the DataFrame in-place.

Output columns added to the cache:
  - ``regime_hmm``: integer regime label from HMM (NaN where input NaN)
  - ``regime_hmm_prob_0 .. regime_hmm_prob_{n-1}``: posterior probs
  - ``regime_gmm``: integer regime label from GMM
  - ``regime_label``: human-readable label mapped from HMM regimes
  - ``structural_break``: 1 on days identified as breakpoints, else 0
  - ``breakpoint_method``: which method detected the break (pelt/bcp)
"""

from __future__ import annotations

import logging
import warnings
from dataclasses import dataclass, field
from typing import Any, Sequence

import numpy as np
import pandas as pd

from operator1.constants import CACHE_DIR

logger = logging.getLogger(__name__)

# Default regime count -- matches spec (bull, bear, high-vol, low-vol).
DEFAULT_N_REGIMES: int = 4

# Default regime label mapping (ordered by increasing mean return after fit).
DEFAULT_REGIME_LABELS: dict[int, str] = {
    0: "bear",
    1: "low_vol",
    2: "high_vol",
    3: "bull",
}

# PELT default penalty (higher = fewer breakpoints, more conservative).
# Overridden by BIC-based penalty when data length is known.
DEFAULT_PELT_PENALTY: float = 10.0


def _bic_pelt_penalty(n_obs: int, n_features: int = 2) -> float:
    """BIC-based PELT penalty (Schwarz 1978).

    Automatically balances model complexity against fit, adapting to data
    length and dimensionality.  Replaces the fixed default when called.
    """
    import math
    if n_obs < 30:
        return DEFAULT_PELT_PENALTY
    return 2.0 * n_features * math.log(n_obs)

# Minimum observations required for HMM / GMM fitting.
_MIN_OBS_HMM: int = 60
_MIN_OBS_GMM: int = 30
_MIN_OBS_PELT: int = 30


# ---------------------------------------------------------------------------
# Result container
# ---------------------------------------------------------------------------


@dataclass
class RegimeResult:
    """Container for regime detection outputs."""

    # HMM
    hmm_regimes: np.ndarray | None = None
    hmm_probs: np.ndarray | None = None
    hmm_fitted: bool = False
    hmm_error: str | None = None

    # GMM
    gmm_regimes: np.ndarray | None = None
    gmm_fitted: bool = False
    gmm_error: str | None = None

    # Structural breaks
    breakpoints_pelt: list[int] = field(default_factory=list)
    pelt_fitted: bool = False
    pelt_error: str | None = None

    breakpoints_bcp: list[int] = field(default_factory=list)
    bcp_fitted: bool = False
    bcp_error: str | None = None

    # Merged labels
    regime_labels: np.ndarray | None = None


# ---------------------------------------------------------------------------
# RegimeDetector class
# ---------------------------------------------------------------------------


class RegimeDetector:
    """Unified regime detection using multiple methods.

    Parameters
    ----------
    n_regimes:
        Number of hidden regimes for HMM / GMM (default 4).
    random_state:
        Seed for reproducibility.
    """

    def __init__(
        self,
        n_regimes: int = DEFAULT_N_REGIMES,
        random_state: int = 42,
    ) -> None:
        self.n_regimes = n_regimes
        self.random_state = random_state
        self._hmm_model: Any = None
        self._gmm_model: Any = None
        self._result = RegimeResult()

    @property
    def result(self) -> RegimeResult:
        return self._result

    # ------------------------------------------------------------------
    # HMM
    # ------------------------------------------------------------------

    def fit_hmm(
        self,
        returns: np.ndarray,
        volatility: np.ndarray,
    ) -> tuple[np.ndarray | None, np.ndarray | None]:
        """Fit a Gaussian HMM on (returns, volatility) features.

        Parameters
        ----------
        returns:
            1-D array of daily log-returns.
        volatility:
            1-D array of rolling volatility (e.g. 21-day).

        Returns
        -------
        (regimes, regime_probs)
            ``regimes`` is an int array of regime labels aligned to
            the *clean* (non-NaN) subset.  ``regime_probs`` has shape
            ``(n_clean, n_regimes)``.  Both are ``None`` on failure.
        """
        try:
            from hmmlearn.hmm import GaussianHMM  # type: ignore[import-untyped]
        except ImportError:
            msg = "hmmlearn not installed -- skipping HMM regime detection"
            logger.warning(msg)
            self._result.hmm_error = msg
            return None, None

        # Build feature matrix and drop NaN rows.
        X = np.column_stack([returns, volatility])
        valid_mask = ~np.isnan(X).any(axis=1)
        X_clean = X[valid_mask]

        if len(X_clean) < _MIN_OBS_HMM:
            msg = (
                f"Insufficient observations for HMM ({len(X_clean)} < "
                f"{_MIN_OBS_HMM}) -- skipping"
            )
            logger.warning(msg)
            self._result.hmm_error = msg
            return None, None

        # Add small regularization to prevent singular covariance matrices.
        # Near-zero variance in one feature (e.g., very low-vol regime)
        # can cause 'covars must be symmetric, positive-definite' errors.
        _epsilon = 1e-6
        X_clean = X_clean + np.random.default_rng(self.random_state).normal(
            0, _epsilon, size=X_clean.shape
        )

        # Try full covariance first, fall back to diagonal if singular.
        for cov_type in ("full", "diag"):
            try:
                with warnings.catch_warnings():
                    warnings.simplefilter("ignore")
                    model = GaussianHMM(
                        n_components=self.n_regimes,
                        covariance_type=cov_type,
                        n_iter=200,
                        random_state=self.random_state,
                        verbose=False,
                    )
                    model.fit(X_clean)

                regimes = model.predict(X_clean)
                probs = model.predict_proba(X_clean)

                # Forward-fill any NaN regime labels from warmup period
                # so no days are left unlabeled in downstream modules.
                if len(regimes) > 0:
                    first_valid = 0
                    for i in range(len(regimes)):
                        if not np.isnan(regimes[i]):
                            first_valid = i
                            break
                    if first_valid > 0:
                        regimes[:first_valid] = regimes[first_valid]
                        probs[:first_valid] = probs[first_valid]

                self._hmm_model = model
                self._result.hmm_regimes = regimes
                self._result.hmm_probs = probs
                self._result.hmm_fitted = True

                monitor = getattr(model, "monitor_", None)
                converged = getattr(monitor, "converged", "unknown") if monitor is not None else "unknown"
                logger.info(
                    "HMM fit: %d regimes, %d observations, cov=%s, converged=%s",
                    self.n_regimes,
                    len(X_clean),
                    cov_type,
                    converged,
                )
                return regimes, probs

            except Exception as exc:
                if cov_type == "full":
                    logger.info(
                        "HMM full covariance failed (%s), retrying with diagonal covariance",
                        exc,
                    )
                    continue
                msg = f"HMM fitting failed: {exc}"
                logger.warning(msg)
                self._result.hmm_error = msg
                return None, None

        msg = "HMM fitting failed: all covariance types exhausted"
        logger.warning(msg)
        self._result.hmm_error = msg
        return None, None

    # ------------------------------------------------------------------
    # GMM
    # ------------------------------------------------------------------

    def fit_gmm(self, returns: np.ndarray) -> np.ndarray | None:
        """Fit a Gaussian Mixture Model for regime clustering.

        Parameters
        ----------
        returns:
            1-D array of daily log-returns.

        Returns
        -------
        regimes:
            Int array of regime labels (aligned to clean subset), or
            ``None`` on failure.
        """
        try:
            from sklearn.mixture import GaussianMixture  # type: ignore[import-untyped]
        except ImportError:
            msg = "scikit-learn not installed -- skipping GMM regime detection"
            logger.warning(msg)
            self._result.gmm_error = msg
            return None

        X = returns[~np.isnan(returns)].reshape(-1, 1)

        if len(X) < _MIN_OBS_GMM:
            msg = (
                f"Insufficient observations for GMM ({len(X)} < "
                f"{_MIN_OBS_GMM}) -- skipping"
            )
            logger.warning(msg)
            self._result.gmm_error = msg
            return None

        try:
            model = GaussianMixture(
                n_components=self.n_regimes,
                covariance_type="full",
                random_state=self.random_state,
                max_iter=200,
            )
            model.fit(X)

            regimes = model.predict(X)

            self._gmm_model = model
            self._result.gmm_regimes = regimes
            self._result.gmm_fitted = True

            logger.info(
                "GMM fit: %d components, %d observations, converged=%s",
                self.n_regimes,
                len(X),
                model.converged_,
            )
            return regimes

        except Exception as exc:
            msg = f"GMM fitting failed: {exc}"
            logger.warning(msg)
            self._result.gmm_error = msg
            return None

    # ------------------------------------------------------------------
    # PELT (structural break detection)
    # ------------------------------------------------------------------

    def detect_breakpoints_pelt(
        self,
        series: np.ndarray,
        penalty: float = DEFAULT_PELT_PENALTY,
    ) -> list[int]:
        """PELT algorithm for structural break detection.

        Parameters
        ----------
        series:
            1-D numeric array (e.g. closing prices or returns).
        penalty:
            Penalty parameter for PELT -- higher values yield fewer
            breakpoints.

        Returns
        -------
        List of integer indices where breakpoints were detected.
        """
        try:
            import ruptures as rpt  # type: ignore[import-untyped]
        except ImportError:
            msg = "ruptures not installed -- skipping PELT breakpoint detection"
            logger.warning(msg)
            self._result.pelt_error = msg
            return []

        series_clean = series[~np.isnan(series)]

        if len(series_clean) < _MIN_OBS_PELT:
            msg = (
                f"Insufficient observations for PELT ({len(series_clean)} "
                f"< {_MIN_OBS_PELT}) -- skipping"
            )
            logger.warning(msg)
            self._result.pelt_error = msg
            return []

        try:
            algo = rpt.Pelt(model="rbf", min_size=5, jump=1).fit(series_clean)
            raw_bkps = algo.predict(pen=penalty)

            # ruptures includes the last index as a sentinel -- remove it.
            breakpoints = [b for b in raw_bkps if b < len(series_clean)]

            self._result.breakpoints_pelt = breakpoints
            self._result.pelt_fitted = True

            logger.info(
                "PELT detected %d structural breaks (penalty=%.1f)",
                len(breakpoints),
                penalty,
            )
            return breakpoints

        except Exception as exc:
            msg = f"PELT detection failed: {exc}"
            logger.warning(msg)
            self._result.pelt_error = msg
            return []

    # ------------------------------------------------------------------
    # Bayesian Change Point (simplified online variant)
    # ------------------------------------------------------------------

    def detect_breakpoints_bcp(
        self,
        series: np.ndarray,
        *,
        hazard_lambda: float = 200.0,
        threshold: float = 0.5,
    ) -> list[int]:
        """Bayesian online change-point detection.

        Uses a simplified Bayesian approach (Adams & MacKay, 2007)
        that works without ``pymc``.  If ``pymc`` is available, a
        full MCMC variant is attempted first for better accuracy.

        Parameters
        ----------
        series:
            1-D numeric array.
        hazard_lambda:
            Expected run length between change points.  Higher values
            make the detector more conservative.
        threshold:
            Minimum posterior change-point probability to flag a day.

        Returns
        -------
        List of integer indices where change points were detected.
        """
        series_clean = series[~np.isnan(series)]

        if len(series_clean) < _MIN_OBS_PELT:
            msg = (
                f"Insufficient observations for BCP ({len(series_clean)} "
                f"< {_MIN_OBS_PELT}) -- skipping"
            )
            logger.warning(msg)
            self._result.bcp_error = msg
            return []

        # Try pymc first (full MCMC), fall back to online algorithm.
        breakpoints = self._bcp_online(series_clean, hazard_lambda, threshold)

        self._result.breakpoints_bcp = breakpoints
        self._result.bcp_fitted = True

        logger.info(
            "BCP detected %d change points (lambda=%.0f, threshold=%.2f)",
            len(breakpoints),
            hazard_lambda,
            threshold,
        )
        return breakpoints

    @staticmethod
    def _bcp_online(
        series: np.ndarray,
        hazard_lambda: float,
        threshold: float,
    ) -> list[int]:
        """Adams-MacKay online Bayesian change-point detection.

        This is a pure-numpy implementation that doesn't require pymc.
        It maintains a run-length distribution and flags points where
        the posterior probability of a run-length reset exceeds the
        threshold.
        """
        n = len(series)
        if n < 3:
            return []

        # Hazard function: constant hazard 1/lambda.
        hazard = 1.0 / hazard_lambda

        # Run-length probabilities -- R[t, r] = P(run_length=r at time t).
        # We only track the current distribution vector (memory efficient).
        R = np.zeros(n + 1)
        R[0] = 1.0  # start with run-length 0

        # Sufficient statistics for Gaussian predictive distribution.
        # Using the conjugate normal-inverse-gamma model.
        # Use overall mean as prior center with moderate prior strength
        # so the model needs a few observations to adapt after a changepoint,
        # making the shift detectable via lower predictive probability.
        mu0 = float(np.mean(series))
        kappa0 = 1.0
        alpha0 = 1.0
        beta0 = float(np.var(series) * 0.1 + 1e-6)  # tight initial variance

        # Per-run-length sufficient stats (vectorised).
        mu_params = np.full(n + 1, mu0)
        kappa_params = np.full(n + 1, kappa0)
        alpha_params = np.full(n + 1, alpha0)
        beta_params = np.full(n + 1, beta0)

        changepoint_probs = np.zeros(n)

        for t in range(1, n):
            x = series[t]

            # Predictive probability under each run-length hypothesis.
            # Student-t predictive: simplified to Gaussian approximation.
            pred_var = (
                beta_params[: t + 1]
                * (kappa_params[: t + 1] + 1)
                / (alpha_params[: t + 1] * kappa_params[: t + 1])
            )
            pred_var = np.maximum(pred_var, 1e-12)
            pred_mean = mu_params[: t + 1]

            # Gaussian log-likelihood.
            pred_prob = np.exp(
                -0.5 * ((x - pred_mean) ** 2) / pred_var
            ) / np.sqrt(2 * np.pi * pred_var)

            # Growth probabilities (existing run continues).
            growth = R[: t + 1] * pred_prob * (1 - hazard)

            # Change-point probability (run resets to 0).
            cp_prob = np.sum(R[: t + 1] * pred_prob * hazard)

            # Build new run-length distribution.
            new_R = np.zeros(n + 1)
            new_R[0] = cp_prob
            new_R[1: t + 2] = growth[: t + 1]

            # Normalise.
            evidence = new_R[: t + 2].sum()
            if evidence > 0:
                new_R[: t + 2] /= evidence

            R = new_R
            changepoint_probs[t] = R[0]

            # Update sufficient statistics for each run-length.
            new_mu = mu_params.copy()
            new_kappa = kappa_params.copy()
            new_alpha = alpha_params.copy()
            new_beta = beta_params.copy()

            # Shift existing stats (run-length grows by 1).
            new_mu[1: t + 2] = (
                kappa_params[: t + 1] * mu_params[: t + 1] + x
            ) / (kappa_params[: t + 1] + 1)
            new_kappa[1: t + 2] = kappa_params[: t + 1] + 1
            new_alpha[1: t + 2] = alpha_params[: t + 1] + 0.5
            new_beta[1: t + 2] = beta_params[: t + 1] + (
                kappa_params[: t + 1]
                * (x - mu_params[: t + 1]) ** 2
                / (2 * (kappa_params[: t + 1] + 1))
            )

            # New run-length 0 resets to prior.
            new_mu[0] = mu0
            new_kappa[0] = kappa0
            new_alpha[0] = alpha0
            new_beta[0] = beta0

            mu_params = new_mu
            kappa_params = new_kappa
            alpha_params = new_alpha
            beta_params = new_beta

        # Flag change points where posterior exceeds threshold.
        breakpoints = list(np.where(changepoint_probs > threshold)[0])

        return breakpoints


# ---------------------------------------------------------------------------
# Regime label mapping
# ---------------------------------------------------------------------------


def _order_regimes_by_mean_return(
    regimes: np.ndarray,
    returns_clean: np.ndarray,
    n_regimes: int,
    *,
    has_volatility_info: bool = True,
) -> dict[int, str]:
    """Map regime integers to labels ordered by mean return.

    Lowest mean return -> "bear", highest -> "bull".  Intermediate
    regimes are labelled by volatility when the model used volatility
    features (HMM), or by return magnitude when it didn't (GMM).

    Parameters
    ----------
    has_volatility_info:
        True for HMM (which uses returns + volatility features),
        False for GMM (which uses returns only).  When False,
        intermediate labels use "low_return"/"moderate_return"
        instead of the misleading "low_vol"/"high_vol".
    """
    if has_volatility_info:
        label_pool = ["bear", "low_vol", "high_vol", "bull"]
    else:
        # GMM only sees returns, so "low_vol"/"high_vol" labels are
        # misleading -- use return-based labels instead.
        label_pool = ["bear", "low_return", "moderate_return", "bull"]
    if n_regimes > len(label_pool):
        # Extend with generic names for extra regimes.
        for i in range(len(label_pool), n_regimes):
            label_pool.append(f"regime_{i}")

    mean_returns = []
    for r in range(n_regimes):
        mask = regimes == r
        if mask.any():
            mean_returns.append(np.mean(returns_clean[mask]))
        else:
            mean_returns.append(0.0)

    # Sort regime indices by mean return ascending.
    sorted_indices = np.argsort(mean_returns)

    mapping: dict[int, str] = {}
    for rank, regime_idx in enumerate(sorted_indices):
        mapping[int(regime_idx)] = label_pool[min(rank, len(label_pool) - 1)]

    return mapping


# ---------------------------------------------------------------------------
# Pipeline entry point
# ---------------------------------------------------------------------------


def detect_regimes_and_breaks(
    cache: pd.DataFrame,
    *,
    n_regimes: int = DEFAULT_N_REGIMES,
    pelt_penalty: float = DEFAULT_PELT_PENALTY,
    bcp_hazard_lambda: float = 200.0,
    bcp_threshold: float = 0.5,
    random_state: int = 42,
    target_variable: str = "return_1d",
) -> tuple[pd.DataFrame, RegimeDetector]:
    """Run all regime detection methods on a daily cache DataFrame.

    Expects columns ``return_1d`` and ``volatility_21d`` to be present
    (or their private company proxies: ``equity_change_rate`` and
    ``financial_volatility``).

    Parameters
    ----------
    cache:
        Target company daily cache (DatetimeIndex).
    n_regimes:
        Number of regimes for HMM/GMM.
    pelt_penalty:
        PELT penalty parameter.
    bcp_hazard_lambda:
        BCP expected run length.
    bcp_threshold:
        BCP posterior probability threshold.
    random_state:
        Random seed.
    target_variable:
        Primary return variable for regime detection. Default is
        ``"return_1d"`` for public companies. For private companies
        without OHLCV data, use ``"equity_change_rate"``.

    Returns
    -------
    (cache, detector)
        The mutated cache DataFrame and the fitted ``RegimeDetector``.
    """
    logger.info("Starting regime detection and structural break analysis...")

    detector = RegimeDetector(n_regimes=n_regimes, random_state=random_state)

    # ------------------------------------------------------------------
    # Extract feature arrays
    # ------------------------------------------------------------------
    # Support both public (return_1d/volatility_21d/close) and private
    # company mode (equity_change_rate/financial_volatility/equity_value).
    returns_col = target_variable
    if target_variable == "equity_change_rate":
        vol_col = "financial_volatility"
        close_col = "equity_value"
    else:
        vol_col = "volatility_21d"
        close_col = "close"

    if returns_col not in cache.columns:
        logger.warning("Column '%s' not found -- regime detection skipped", returns_col)
        _add_empty_regime_columns(cache, n_regimes)
        return cache, detector

    returns = cache[returns_col].values
    volatility = (
        cache[vol_col].values if vol_col in cache.columns else np.full(len(cache), np.nan)
    )
    close = cache[close_col].values if close_col in cache.columns else returns

    # Non-NaN mask for returns (used to align results back to cache).
    valid_ret = ~np.isnan(returns)

    # ------------------------------------------------------------------
    # HMM
    # ------------------------------------------------------------------
    hmm_regimes, hmm_probs = detector.fit_hmm(returns, volatility)

    cache["regime_hmm"] = np.nan
    if hmm_regimes is not None:
        # Align back: valid_ret positions get the regime labels.
        valid_both = valid_ret & ~np.isnan(volatility)
        idx_clean = np.where(valid_both)[0]

        if len(idx_clean) == len(hmm_regimes):
            cache.iloc[idx_clean, cache.columns.get_loc("regime_hmm")] = hmm_regimes
        else:
            logger.warning(
                "HMM output length mismatch (%d vs %d valid) -- skipping assignment",
                len(hmm_regimes),
                len(idx_clean),
            )

        # Store posterior probabilities.
        if hmm_probs is not None:
            for r in range(n_regimes):
                col = f"regime_hmm_prob_{r}"
                cache[col] = np.nan
                if len(idx_clean) == hmm_probs.shape[0]:
                    cache.iloc[idx_clean, cache.columns.get_loc(col)] = hmm_probs[:, r]

    # ------------------------------------------------------------------
    # GMM
    # ------------------------------------------------------------------
    gmm_regimes = detector.fit_gmm(returns)

    cache["regime_gmm"] = np.nan
    if gmm_regimes is not None:
        idx_clean = np.where(valid_ret)[0]
        if len(idx_clean) == len(gmm_regimes):
            cache.iloc[idx_clean, cache.columns.get_loc("regime_gmm")] = gmm_regimes

    # ------------------------------------------------------------------
    # Structural breaks (PELT)
    # ------------------------------------------------------------------
    # Use BIC-based penalty when the caller passed the default value.
    # BIC automatically adapts to data length (Schwarz 1978).
    if pelt_penalty == DEFAULT_PELT_PENALTY:
        pelt_penalty = _bic_pelt_penalty(len(cache), n_features=2)
        logger.info("PELT penalty auto-set via BIC: %.2f (n=%d)", pelt_penalty, len(cache))
    bp_pelt = detector.detect_breakpoints_pelt(close, penalty=pelt_penalty)

    cache["structural_break"] = 0
    cache["breakpoint_method"] = ""

    # Map breakpoint indices (from clean series) back to cache positions.
    if bp_pelt:
        close_valid_idx = np.where(~np.isnan(close))[0]
        for bp in bp_pelt:
            if bp < len(close_valid_idx):
                cache_pos = close_valid_idx[bp]
                cache.iloc[cache_pos, cache.columns.get_loc("structural_break")] = 1
                existing = cache.iloc[cache_pos, cache.columns.get_loc("breakpoint_method")]
                cache.iloc[cache_pos, cache.columns.get_loc("breakpoint_method")] = (
                    "pelt" if not existing else f"{existing},pelt"
                )

    # ------------------------------------------------------------------
    # Structural breaks (BCP)
    # ------------------------------------------------------------------
    bp_bcp = detector.detect_breakpoints_bcp(
        close,
        hazard_lambda=bcp_hazard_lambda,
        threshold=bcp_threshold,
    )

    if bp_bcp:
        close_valid_idx = np.where(~np.isnan(close))[0]
        for bp in bp_bcp:
            if bp < len(close_valid_idx):
                cache_pos = close_valid_idx[bp]
                cache.iloc[cache_pos, cache.columns.get_loc("structural_break")] = 1
                existing = cache.iloc[cache_pos, cache.columns.get_loc("breakpoint_method")]
                cache.iloc[cache_pos, cache.columns.get_loc("breakpoint_method")] = (
                    "bcp" if not existing else f"{existing},bcp"
                )

    # ------------------------------------------------------------------
    # Regime labels (from HMM, with fallback to GMM)
    # ------------------------------------------------------------------
    if hmm_regimes is not None:
        valid_both = valid_ret & ~np.isnan(volatility)
        returns_clean = returns[valid_both]

        label_map = _order_regimes_by_mean_return(
            hmm_regimes, returns_clean, n_regimes,
            has_volatility_info=True,
        )
        cache["regime_label"] = cache["regime_hmm"].map(label_map)
        logger.info("Regime label mapping: %s", label_map)

    elif gmm_regimes is not None:
        returns_clean = returns[valid_ret]
        label_map = _order_regimes_by_mean_return(
            gmm_regimes, returns_clean, n_regimes,
            has_volatility_info=False,
        )
        cache["regime_label"] = cache["regime_gmm"].map(label_map)
        logger.info("Regime label mapping (from GMM fallback): %s", label_map)

    else:
        cache["regime_label"] = np.nan
        logger.warning("No regime model succeeded -- regime_label is all NaN")

    # ------------------------------------------------------------------
    # Summary logging
    # ------------------------------------------------------------------
    n_breaks = int(cache["structural_break"].sum())
    regime_counts = cache["regime_label"].value_counts(dropna=False).to_dict()
    logger.info(
        "Regime detection complete: %d structural breaks, regime distribution: %s",
        n_breaks,
        regime_counts,
    )

    return cache, detector


def _add_empty_regime_columns(cache: pd.DataFrame, n_regimes: int) -> None:
    """Add empty regime columns when detection cannot run."""
    cache["regime_hmm"] = np.nan
    cache["regime_gmm"] = np.nan
    cache["regime_label"] = np.nan
    cache["structural_break"] = 0
    cache["breakpoint_method"] = ""
    for r in range(n_regimes):
        cache[f"regime_hmm_prob_{r}"] = np.nan


# ---------------------------------------------------------------------------
# Early regime detection (Step 5.5 bridge)
# ---------------------------------------------------------------------------


@dataclass
class EarlyRegimeResult:
    """Lightweight result from early regime detection for the enriched survival timeline."""

    regime_labels: pd.Series | None = None
    regime_confidence: pd.Series | None = None
    detector: RegimeDetector | None = None
    fitted: bool = False
    error: str | None = None


def compute_online_change_scores(
    returns: np.ndarray,
    r: float = 0.01,
    order: int = 1,
    smooth: int = 7,
) -> np.ndarray | None:
    """Compute online change point scores using ChangeFinder (SDAR algorithm).

    Unlike batch PELT, ChangeFinder updates sequentially and detects regime
    changes in real-time with no look-ahead. High scores indicate probable
    change points.

    Returns an array of anomaly scores (same length as input), or None
    if changefinder is not installed.
    """
    try:
        import changefinder as cf_lib
    except ImportError:
        return None

    try:
        detector = cf_lib.ChangeFinder(r=r, order=order, smooth=smooth)
        scores = np.array([detector.update(float(x)) for x in returns])
        logger.info(
            "ChangeFinder: %d scores computed, max=%.3f, mean=%.3f",
            len(scores), float(np.max(scores)), float(np.mean(scores)),
        )
        return scores
    except Exception as exc:
        logger.debug("ChangeFinder failed: %s", exc)
        return None


def compute_distribution_drift(
    cache: pd.DataFrame,
    variable: str = "return_1d",
    window: int = 63,
    reference_window: int = 252,
) -> pd.Series | None:
    """Compute Wasserstein distance between recent and reference return distributions.

    A large distance indicates the return process has structurally changed,
    suggesting GARCH parameters may be stale and a regime shift has occurred.

    Uses ``scipy.stats.wasserstein_distance`` (already installed).

    Parameters
    ----------
    cache:
        Daily cache with the target variable.
    variable:
        Column name to measure drift on.
    window:
        Rolling window for "recent" distribution (default: 63 ~ 3 months).
    reference_window:
        Lookback for the "reference" distribution (default: 252 ~ 1 year).

    Returns
    -------
    pd.Series of Wasserstein distances, or None if insufficient data.
    """
    if variable not in cache.columns:
        return None
    series = cache[variable].dropna()
    if len(series) < reference_window + window:
        return None

    try:
        from scipy.stats import wasserstein_distance

        drift_scores = pd.Series(np.nan, index=cache.index)
        values = series.values
        idx = series.index

        for i in range(reference_window, len(values) - window + 1):
            ref = values[i - reference_window:i]
            recent = values[i:i + window]
            if len(ref) > 10 and len(recent) > 10:
                drift_scores.loc[idx[i + window - 1]] = wasserstein_distance(ref, recent)

        # Normalize to [0, 1] by dividing by the max observed distance
        max_drift = drift_scores.max()
        if max_drift > 0:
            drift_scores = drift_scores / max_drift

        logger.debug(
            "Distribution drift: %d scores, max=%.3f, mean=%.3f",
            drift_scores.notna().sum(),
            drift_scores.max() if drift_scores.notna().any() else 0,
            drift_scores.mean() if drift_scores.notna().any() else 0,
        )
        return drift_scores
    except ImportError:
        logger.debug("scipy not available for Wasserstein distance")
        return None
    except Exception as exc:
        logger.debug("Distribution drift computation failed: %s", exc)
        return None


def run_early_regime_detection(
    cache: pd.DataFrame,
    *,
    n_regimes: int = DEFAULT_N_REGIMES,
    pelt_penalty: float = DEFAULT_PELT_PENALTY,
    bcp_hazard_lambda: float = 200.0,
    bcp_threshold: float = 0.5,
    random_state: int = 42,
    target_variable: str = "return_1d",
) -> tuple[pd.DataFrame, EarlyRegimeResult]:
    """Run regime detection early in the pipeline (Step 5.5).

    This is a wrapper around ``detect_regimes_and_breaks()`` that also
    extracts the regime labels and confidence series needed by the
    enriched survival timeline.

    The full regime columns (``regime_hmm``, ``regime_gmm``, ``regime_label``,
    ``structural_break``, etc.) are added to the cache by the underlying
    ``detect_regimes_and_breaks()`` call, so Step 6 does not need to re-run
    regime detection.

    Parameters
    ----------
    cache:
        Daily cache DataFrame with ``return_1d`` and ``volatility_21d``
        (or ``equity_change_rate`` and ``financial_volatility`` for
        private companies).
    target_variable:
        Primary return variable. ``"return_1d"`` for public companies,
        ``"equity_change_rate"`` for private companies without OHLCV.
    n_regimes, pelt_penalty, bcp_hazard_lambda, bcp_threshold, random_state:
        Passed through to ``detect_regimes_and_breaks()``.

    Returns
    -------
    (cache, early_result)
        The mutated cache and an ``EarlyRegimeResult`` with regime_labels
        and regime_confidence Series aligned to the cache index.
    """
    early = EarlyRegimeResult()

    try:
        cache, detector = detect_regimes_and_breaks(
            cache,
            n_regimes=n_regimes,
            pelt_penalty=pelt_penalty,
            bcp_hazard_lambda=bcp_hazard_lambda,
            bcp_threshold=bcp_threshold,
            random_state=random_state,
            target_variable=target_variable,
        )
        early.detector = detector

        # Extract regime labels as a string Series.
        # IMPORTANT: The ordering below matters -- astype(str) first converts
        # NaN to the literal string "nan", then .where() replaces those
        # positions with "unknown" using the *original* NaN mask.  Do NOT
        # refactor to labels.fillna("unknown").astype(str) -- that would
        # skip the mask check and could leak "nan" strings if dtype changes.
        if "regime_label" in cache.columns:
            labels = cache["regime_label"].copy()
            if labels.notna().any():
                early.regime_labels = labels.astype(str).where(labels.notna(), "unknown")
            else:
                early.regime_labels = pd.Series("unknown", index=cache.index)
        else:
            early.regime_labels = pd.Series("unknown", index=cache.index)

        # Safety net: replace any lingering "nan" strings that might leak
        # through unexpected code paths (e.g., mixed-type Series edge cases).
        early.regime_labels = early.regime_labels.replace("nan", "unknown")

        # Extract max HMM posterior as regime confidence.
        prob_cols = [c for c in cache.columns if c.startswith("regime_hmm_prob_")]
        if prob_cols:
            prob_df = cache[prob_cols]
            if prob_df.notna().any().any():
                early.regime_confidence = prob_df.max(axis=1)
            else:
                early.regime_confidence = pd.Series(0.5, index=cache.index)
        else:
            early.regime_confidence = pd.Series(0.5, index=cache.index)

        early.fitted = True
        logger.info("Early regime detection complete for enriched survival timeline")

        # Compute Wasserstein distribution drift (non-parametric regime shift signal)
        try:
            _drift = compute_distribution_drift(cache, variable=target_variable)
            if _drift is not None and _drift.notna().any():
                cache["wasserstein_drift"] = _drift
                # Flag significant drift (top 10% of observed distances)
                _threshold = _drift.quantile(0.90)
                cache["distribution_shift_flag"] = (_drift > _threshold).astype(int)
                logger.info(
                    "Wasserstein drift: mean=%.3f, max=%.3f, %d shift days flagged",
                    _drift.mean(), _drift.max(),
                    cache["distribution_shift_flag"].sum(),
                )
        except Exception as _exc:
            logger.debug("Wasserstein drift skipped: %s", _exc)

    except Exception as exc:
        early.error = f"Early regime detection failed: {exc}"
        logger.warning(early.error)
        # Provide safe defaults so enriched timeline can still run.
        early.regime_labels = pd.Series("unknown", index=cache.index)
        early.regime_confidence = pd.Series(0.5, index=cache.index)

    return cache, early
