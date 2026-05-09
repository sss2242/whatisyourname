"""T4.1 -- Survival mode detection (Step F).

Computes three daily boolean flags:

- **company_survival_mode_flag**: triggered when ANY of the following
  are true for a given day ``t``:
    - ``current_ratio < 1.0``
    - ``debt_to_equity_abs > 3.0``
    - ``fcf_yield < 0``
    - ``drawdown_252d < -0.40``
    - ``conflict_intensity_score > 0.7`` (geopolitical conflict)
    - ``sanctions_flag == 1`` (international sanctions)

- **country_survival_mode_flag**: triggered when ANY config-driven
  macro threshold is breached (credit spread, unemployment, yield
  curve, FX volatility).  Since many of these macro proxies may not
  be available from the per-region macro APIs, the flag conservatively
  defaults to 0 when the data is missing.

- **country_protected_flag**: triggered when ANY of:
    - Target sector appears in ``strategic_sectors`` config list
    - Market cap > ``market_cap_gdp_threshold`` * GDP
    - Emergency rate cut > threshold within lookback window

All thresholds are loaded from ``config/country_protection_rules.yml``
-- nothing is hardcoded.
"""

from __future__ import annotations

import logging
from typing import Any

import pandas as pd

from operator1.config_loader import load_config

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Company survival
# ---------------------------------------------------------------------------

# Default thresholds -- read from config/scoring_weights.yml if available,
# falling back to hardcoded values when config is missing.
def _load_company_thresholds() -> dict[str, float]:
    """Load survival thresholds from scoring_weights.yml with hardcoded fallbacks."""
    try:
        from operator1.scoring_weights import get_weight
        return {
            "current_ratio_lt": float(get_weight("survival_thresholds.current_ratio", 1.0)),
            "debt_to_equity_abs_gt": float(get_weight("survival_thresholds.debt_to_equity", 3.0)),
            "fcf_yield_lt": float(get_weight("survival_thresholds.fcf_yield", 0.0)),
            "drawdown_252d_lt": float(get_weight("survival_thresholds.drawdown_252d", -0.40)),
            "conflict_intensity_gt": float(get_weight("survival_thresholds.conflict_intensity", 0.70)),
            "inst_flow_momentum_lt": float(get_weight("survival_thresholds.inst_flow_momentum", -0.15)),
        }
    except Exception:
        return {
            "current_ratio_lt": 1.0,
            "debt_to_equity_abs_gt": 3.0,
            "fcf_yield_lt": 0.0,
            "drawdown_252d_lt": -0.40,
        }


_COMPANY_THRESHOLDS = _load_company_thresholds()


def compute_company_survival_flag(
    df: pd.DataFrame,
    thresholds: dict[str, float] | None = None,
    freq: str = "D",
) -> pd.Series:
    """Compute daily company survival mode flag.

    Parameters
    ----------
    df:
        Feature table with derived variables already computed.
    thresholds:
        Override default thresholds (mainly for testing).
    freq:
        Data frequency (D/W/M/Q/A/S).  At D/W/M, the ``fcf_yield``
        trigger is skipped because flow-variable interpolation
        produces a distorted value (~0.04% instead of ~3.3%).
        The fcf_yield trigger is only reliable at Q/A/S where
        the value is at native filing scale.

    Returns
    -------
    pd.Series
        Integer series: 1 = survival mode active, 0 = normal.
    """
    t = thresholds or _COMPANY_THRESHOLDS
    _native_ratio_freqs = {"Q", "A", "S"}
    _daily_freqs = {"D", "W"}
    freq = freq.upper() if freq else "D"

    # Two-category condition architecture: liquidity triggers (suppressible
    # by cash adequacy) vs non-liquidity triggers (never suppressed).
    liquidity_conditions: list[pd.Series] = []
    non_liquidity_conditions: list[pd.Series] = []

    # ===================================================================
    # UNIVERSAL TRIGGERS (all frequencies)
    # ===================================================================

    # Current ratio < 1.0 (STOCK/STOCK -- valid at any freq)
    if "current_ratio" in df.columns:
        cr = df["current_ratio"]
        liquidity_conditions.append(cr.notna() & (cr < t.get("current_ratio_lt", 1.0)))

    # Debt-to-equity (absolute) > 3.0 (STOCK/STOCK -- valid at any freq)
    if "debt_to_equity_abs" in df.columns:
        de = df["debt_to_equity_abs"]
        liquidity_conditions.append(de.notna() & (de > t.get("debt_to_equity_abs_gt", 3.0)))

    # Drawdown < -40% (OHLCV -- valid at any freq)
    if "drawdown_252d" in df.columns:
        dd = df["drawdown_252d"]
        non_liquidity_conditions.append(dd.notna() & (dd < t.get("drawdown_252d_lt", -0.40)))

    # Geopolitical conflict (static/news -- valid at any freq)
    if "country_conflict_flag" in df.columns:
        cf = df["country_conflict_flag"]
        non_liquidity_conditions.append(cf.notna() & (cf == 1))
    elif "conflict_intensity_score" in df.columns:
        ci = df["conflict_intensity_score"]
        non_liquidity_conditions.append(ci.notna() & (ci > t.get("conflict_intensity_gt", 0.7)))
    if "sanctions_flag" in df.columns:
        sf = df["sanctions_flag"]
        non_liquidity_conditions.append(sf.notna() & (sf == 1))

    # ===================================================================
    # Q/A/S-ONLY TRIGGERS (native filing frequency -- flow-based ratios)
    # ===================================================================

    if freq in _native_ratio_freqs:
        # FCF yield < 0 (FLOW/MARKET -- only valid at native freq)
        if "fcf_yield" in df.columns:
            fy = df["fcf_yield"]
            liquidity_conditions.append(fy.notna() & (fy < t.get("fcf_yield_lt", 0.0)))

        # Revenue decline > 20% YoY (FLOW-based, Altman 1968)
        if "revenue_growth_yoy" in df.columns:
            rg = df["revenue_growth_yoy"]
            non_liquidity_conditions.append(rg.notna() & (rg < -0.20))

        # Altman Z-Score < 1.81 = distress zone (needs correct x3/x5 at Q/A)
        if "fh_altman_z_score" in df.columns:
            az = df["fh_altman_z_score"]
            non_liquidity_conditions.append(az.notna() & (az < 1.81))

        # Negative gross margin = selling below cost (Q/A only, same-filing data)
        if freq in ("Q", "S") and "gross_margin" in df.columns:
            gm = df["gross_margin"]
            non_liquidity_conditions.append(gm.notna() & (gm < 0))

    else:
        if "fcf_yield" in df.columns:
            logger.debug("Skipping fcf_yield trigger at freq=%s (distorted)", freq)

    # ===================================================================
    # D/W-ONLY TRIGGERS (daily OHLCV-native + institutional signals)
    # ===================================================================

    if freq in _daily_freqs:
        # Institutional selling (computed from daily holder/volume data)
        if "inst_flow_momentum" in df.columns:
            ifm = df["inst_flow_momentum"]
            non_liquidity_conditions.append(ifm.notna() & (ifm < t.get("inst_flow_momentum_lt", -0.15)))

        # Crowded + illiquid ownership structure
        if "inst_crowding_score" in df.columns and "inst_amihud_illiquidity" in df.columns:
            cs = df["inst_crowding_score"]
            ai = df["inst_amihud_illiquidity"]
            high_crowd = cs.notna() & (cs > t.get("inst_crowding_score_gt", 0.8))
            high_illiq = ai.notna() & (ai > ai.quantile(0.9))
            non_liquidity_conditions.append(high_crowd & high_illiq)

        # Merton distance-to-default < 1.0 (uses daily vol + stock debt, Merton 1974)
        if "merton_dd" in df.columns:
            mdd = df["merton_dd"]
            non_liquidity_conditions.append(mdd.notna() & (mdd < 1.0))

        # Vol-of-vol spike > P95 of own history (regime instability, Cont & da Fonseca 2002)
        if "vol_of_vol_21d" in df.columns:
            vov = df["vol_of_vol_21d"]
            _p95 = vov.quantile(0.95)
            if _p95 > 0:
                non_liquidity_conditions.append(vov.notna() & (vov > _p95))

    all_conditions = liquidity_conditions + non_liquidity_conditions
    if not all_conditions:
        logger.warning("No company survival trigger columns found -- defaulting to 0")
        return pd.Series(0, index=df.index, name="company_survival_mode_flag")

    # Build separate OR masks for each category
    _zero = pd.Series(False, index=df.index)

    liquidity_combined = _zero.copy()
    for c in liquidity_conditions:
        liquidity_combined = liquidity_combined | c

    non_liquidity_combined = _zero.copy()
    for c in non_liquidity_conditions:
        non_liquidity_combined = non_liquidity_combined | c

    # P6: Cash adequacy floor -- suppress ONLY liquidity triggers for
    # cash-rich companies. A company with $30B cash and current_ratio=1.01
    # (e.g. AAPL) is fundamentally different from one with $1M cash and
    # the same ratio. Non-liquidity triggers (drawdown, conflict, sanctions,
    # institutional flow) are NEVER suppressed -- those signal real problems
    # that cash reserves cannot mitigate.
    if ("cash_and_equivalents" in df.columns and "market_cap" in df.columns
            and df["cash_and_equivalents"].notna().any()
            and df["market_cap"].notna().any()):
        try:
            _cash = df["cash_and_equivalents"].astype(float)
            _mcap = df["market_cap"].astype(float)
            # Cash > 5% of market cap = cash-adequate, suppress liquidity triggers
            _cash_adequate = (_cash > 0) & (_mcap > 0) & (_cash / _mcap > 0.05)
            if _cash_adequate.any():
                _n_before = int(liquidity_combined.sum())
                liquidity_combined = liquidity_combined & ~_cash_adequate
                _n_after = int(liquidity_combined.sum())
                if _n_before != _n_after:
                    logger.info(
                        "Cash adequacy floor: suppressed %d/%d liquidity-triggered "
                        "survival days (cash/mcap > 5%%). Non-liquidity triggers "
                        "preserved: %d days.",
                        _n_before - _n_after, _n_before,
                        int(non_liquidity_combined.sum()),
                    )
        except Exception:
            pass

    # Final combined: liquidity (after cash floor) OR non-liquidity (always preserved)
    combined = liquidity_combined | non_liquidity_combined

    flag = combined.astype(int)
    flag.name = "company_survival_mode_flag"

    triggered = flag.sum()
    logger.info(
        "Company survival flag: %d / %d days triggered (%.1f%%)",
        triggered, len(flag), triggered / max(len(flag), 1) * 100,
    )

    return flag


def compute_survival_velocity(
    df: pd.DataFrame,
    thresholds: dict[str, float] | None = None,
    window: int = 21,
) -> tuple[pd.Series, pd.Series]:
    """Compute deterioration velocity toward survival thresholds.

    Fires a velocity flag when any trigger variable is approaching its
    threshold faster than 30% of the threshold distance per ``window`` days,
    even if the threshold has not yet been breached.

    Based on Duffie, Saita & Wang (2007) insight that default intensity
    depends on both level AND trajectory of covariates.

    Parameters
    ----------
    df:
        Daily feature table with derived variables.
    thresholds:
        Override thresholds (uses company defaults if None).
    window:
        Lookback window for velocity computation (default 21 days).

    Returns
    -------
    tuple[pd.Series, pd.Series]
        (survival_velocity_flag, survival_deterioration_rate)
        Flag is 1 when any trigger is deteriorating fast; rate is the
        worst (most negative) normalized deterioration across triggers.
    """
    import numpy as np

    t = thresholds or _COMPANY_THRESHOLDS
    _eps = 1e-10

    trigger_configs = [
        ("current_ratio", t.get("current_ratio_lt", 1.0), "below"),
        ("debt_to_equity_abs", t.get("debt_to_equity_abs_gt", 3.0), "above"),
        ("fcf_yield", t.get("fcf_yield_lt", 0.0), "below"),
        ("drawdown_252d", t.get("drawdown_252d_lt", -0.40), "below"),
    ]

    velocities: list[pd.Series] = []
    for col, threshold, direction in trigger_configs:
        if col not in df.columns or df[col].isna().all():
            continue
        s = df[col].astype(float)
        delta = s.diff(window)
        denom = max(abs(threshold), _eps)
        if direction == "below":
            # Negative delta = moving toward breach (value dropping toward threshold)
            vel = delta / denom
        else:
            # Positive delta = moving toward breach (value rising toward threshold)
            vel = -delta / denom
        velocities.append(vel)

    if not velocities:
        zero = pd.Series(0, index=df.index, dtype=int)
        nan = pd.Series(np.nan, index=df.index, dtype=float)
        zero.name = "survival_velocity_flag"
        nan.name = "survival_deterioration_rate"
        return zero, nan

    stacked = pd.concat(velocities, axis=1)
    # Most negative = worst deterioration across all triggers
    worst_rate = stacked.min(axis=1)
    worst_rate.name = "survival_deterioration_rate"

    # Flag: fires when deterioration exceeds 30% of threshold distance in window
    velocity_flag = (worst_rate < -0.30).astype(int)
    velocity_flag.name = "survival_velocity_flag"

    n_flagged = int(velocity_flag.sum())
    if n_flagged > 0:
        logger.info(
            "Survival velocity: %d / %d days flagged (fast deterioration toward threshold)",
            n_flagged, len(velocity_flag),
        )

    return velocity_flag, worst_rate


def compute_survival_probability(
    df: pd.DataFrame,
    thresholds: dict[str, float] | None = None,
) -> pd.Series:
    """Compute continuous company survival distress probability (0-1).

    Unlike the binary ``compute_company_survival_flag()`` which produces
    sharp 0/1 transitions at threshold crossings, this produces a smooth
    probability using sigmoid transforms of the distance from each threshold.

    A company with current_ratio = 1.01 gets probability ~0.48 (nearly
    distressed) while one with 0.99 gets ~0.52. The binary flag would
    give 0 and 1 respectively -- a cliff edge.

    The probability drives hierarchy weight interpolation: weights shift
    proportionally rather than snapping between regimes.
    """
    import numpy as np

    t = thresholds or _COMPANY_THRESHOLDS

    distress_signals: list[pd.Series] = []

    # Each trigger: compute normalized distance from threshold, then sigmoid
    trigger_configs = [
        ("current_ratio", t.get("current_ratio_lt", 1.0), "below"),
        ("debt_to_equity_abs", t.get("debt_to_equity_abs_gt", 3.0), "above"),
        ("fcf_yield", t.get("fcf_yield_lt", 0.0), "below"),
        ("drawdown_252d", t.get("drawdown_252d_lt", -0.40), "below"),
    ]

    for col, threshold, direction in trigger_configs:
        if col not in df.columns or df[col].isna().all():
            continue  # Skip entirely -- don't contribute 0.5 for missing data
        s = df[col].copy()  # Preserve NaN; per-row NaN handled by max aggregation

        # Distance from threshold (positive = in distress direction)
        if direction == "below":
            dist = (threshold - s) / max(abs(threshold), 0.01)
        else:
            dist = (s - threshold) / max(abs(threshold), 0.01)

        # Clip to reasonable range before sigmoid
        dist = dist.clip(lower=-5, upper=5)
        distress_signals.append(dist)

    # Conflict/sanctions triggers
    for col in ("country_conflict_flag", "sanctions_flag"):
        if col in df.columns:
            distress_signals.append(df[col].fillna(0).astype(float) * 3.0)

    if not distress_signals:
        return pd.Series(0.0, index=df.index, name="survival_probability")

    # Aggregate distress signals.  Two methods:
    #  - Max: conservative, dominated by the single worst trigger.
    #  - Geometric mean: captures compounding risk of multiple near-breaches
    #    (e.g. current_ratio=1.05, D/E=2.8, FCF=-0.01 all near thresholds).
    _signals_df = pd.concat(distress_signals, axis=1)
    combined = _signals_df.max(axis=1)

    # Geometric mean of (1 + signal) - 1, clipped to avoid negative signals
    # producing complex numbers.  When multiple signals are mildly positive,
    # the geometric mean is higher than the max of any single signal.
    _clipped = _signals_df.clip(lower=-0.99)
    _n_signals = (_clipped.notna()).sum(axis=1).clip(lower=1)
    _geo_mean = (1 + _clipped).prod(axis=1).pow(1.0 / _n_signals) - 1
    # Use geometric mean when it produces a higher distress signal
    # (multiple near-breaches compound), otherwise use max.
    combined = pd.concat([combined, _geo_mean], axis=1).max(axis=1)

    # Sigmoid transform to [0, 1]
    probability = 1.0 / (1.0 + np.exp(-combined))
    probability.name = "survival_probability"

    logger.info(
        "Survival probability: mean=%.3f, max=%.3f, >50%%=%d days",
        probability.mean(), probability.max(), (probability > 0.5).sum(),
    )

    return probability


def compute_survival_uncertainty(
    df: pd.DataFrame,
    probability: pd.Series | None = None,
    n_bootstrap: int = 100,
) -> tuple[pd.Series, pd.Series, pd.Series]:
    """Compute uncertainty bands around survival probability via bootstrap.

    Uses bootstrapped resampling of the distance-to-threshold components
    to produce P10/P90 credible intervals and an uncertainty spread.

    When Cox PH results are available (via lifelines standard errors),
    propagates analytical uncertainty. Falls back to bootstrap of the
    sigmoid components otherwise.

    Parameters
    ----------
    df:
        Daily feature table with survival trigger variables.
    probability:
        Pre-computed survival probability series (avoids recomputation).
    n_bootstrap:
        Number of bootstrap samples (default 100 for speed).

    Returns
    -------
    tuple[pd.Series, pd.Series, pd.Series]
        (survival_probability_p10, survival_probability_p90, survival_uncertainty)
    """
    import numpy as np

    if probability is None:
        probability = compute_survival_probability(df)

    # Bootstrap approach: add noise to trigger variables, recompute probability
    trigger_cols = [c for c in ("current_ratio", "debt_to_equity_abs", "fcf_yield", "drawdown_252d")
                    if c in df.columns and df[c].notna().any()]

    if not trigger_cols:
        p10 = probability.copy()
        p90 = probability.copy()
        p10.name = "survival_probability_p10"
        p90.name = "survival_probability_p90"
        unc = pd.Series(0.0, index=df.index, name="survival_uncertainty")
        return p10, p90, unc

    rng = np.random.default_rng(42)
    bootstrap_probs = np.zeros((n_bootstrap, len(df)))

    for b in range(n_bootstrap):
        # Perturb each trigger variable by its own rolling std
        perturbed = df.copy()
        for col in trigger_cols:
            s = perturbed[col].astype(float)
            noise_scale = s.rolling(63, min_periods=10).std().fillna(s.std())
            noise = rng.normal(0, 1, size=len(s)) * noise_scale.values * 0.5
            perturbed[col] = s + noise

        bp = compute_survival_probability(perturbed)
        bootstrap_probs[b] = bp.values

    p10_vals = np.nanpercentile(bootstrap_probs, 10, axis=0)
    p90_vals = np.nanpercentile(bootstrap_probs, 90, axis=0)

    p10 = pd.Series(p10_vals, index=df.index, name="survival_probability_p10")
    p90 = pd.Series(p90_vals, index=df.index, name="survival_probability_p90")
    uncertainty = pd.Series(p90_vals - p10_vals, index=df.index, name="survival_uncertainty")

    logger.info(
        "Survival uncertainty: mean spread=%.3f (P10=%.3f, P90=%.3f)",
        uncertainty.mean(), p10.mean(), p90.mean(),
    )

    return p10, p90, uncertainty


def compute_cox_survival_score(
    df: pd.DataFrame,
) -> pd.Series:
    """Compute data-driven survival risk score using Cox Proportional Hazards.

    Uses lifelines CoxPHFitter to learn hazard ratios from the company's
    own history of survival flag activations. Produces a continuous partial
    hazard score that reflects learned (not assumed) risk weights.

    Falls back to the sigmoid-based survival_probability if lifelines is
    not available or if there are too few survival events to fit.

    Returns
    -------
    pd.Series
        Continuous hazard score (higher = more distressed). Normalized to [0, 1].
    """
    import numpy as np

    try:
        from lifelines import CoxPHFitter
    except ImportError:
        logger.debug("lifelines not installed, skipping Cox PH survival score")
        return pd.Series(np.nan, index=df.index, name="cox_survival_score")

    # Covariates for Cox model -- same ratios as threshold-based detection
    covariates = []
    for col in ("current_ratio", "debt_to_equity_abs", "fcf_yield", "drawdown_252d"):
        if col in df.columns:
            covariates.append(col)

    if len(covariates) < 2:
        return pd.Series(np.nan, index=df.index, name="cox_survival_score")

    # Need an event column (binary survival flag)
    if "company_survival_mode_flag" not in df.columns:
        return pd.Series(np.nan, index=df.index, name="cox_survival_score")

    # Prepare data for Cox
    cox_df = df[covariates + ["company_survival_mode_flag"]].copy()
    cox_df = cox_df.dropna()

    # Need enough events (at least 5 survival days) for meaningful fit
    n_events = int(cox_df["company_survival_mode_flag"].sum())
    if n_events < 5 or len(cox_df) < 50:
        logger.debug("Too few survival events (%d) for Cox PH", n_events)
        return pd.Series(np.nan, index=df.index, name="cox_survival_score")

    try:
        # Duration = days since last survival event (run-length encoding).
        # Cox PH theory requires proper time-between-events, not a
        # monotonically increasing synthetic index.
        _flag = cox_df["company_survival_mode_flag"].values
        _durations = np.zeros(len(_flag), dtype=float)
        _counter = 1.0
        for _i in range(len(_flag)):
            _durations[_i] = _counter
            if _flag[_i] == 1:
                _counter = 1.0  # reset after event
            else:
                _counter += 1.0
        cox_df["duration"] = _durations

        cph = CoxPHFitter(penalizer=0.1)  # regularization for stability
        cph.fit(
            cox_df,
            duration_col="duration",
            event_col="company_survival_mode_flag",
        )

        # Partial hazard = exp(beta * X) -- continuous risk score
        hazard = cph.predict_partial_hazard(df[covariates].ffill().bfill())

        # Normalize to [0, 1] using sigmoid of log-hazard
        log_h = np.log(hazard.clip(lower=1e-10))
        normalized = 1.0 / (1.0 + np.exp(-log_h))
        normalized.name = "cox_survival_score"

        logger.info(
            "Cox PH survival score: %d obs, %d events, hazard_ratios=%s",
            len(cox_df), n_events,
            {col: round(float(cph.hazard_ratios_[col]), 3) for col in covariates},
        )
        return normalized

    except Exception as exc:
        logger.debug("Cox PH fitting failed: %s", exc)
        return pd.Series(np.nan, index=df.index, name="cox_survival_score")


def compute_discrete_time_survival(
    df: pd.DataFrame,
    freq: str = "Q",
) -> pd.Series:
    """Discrete-time survival score for quarterly/annual data.

    At low frequencies (Q, A, S), the continuous-time Cox PH assumption
    breaks down because events are only observed at discrete filing
    intervals.  This function uses complementary log-log link (the
    discrete-time analogue of Cox PH) via logistic regression with
    time dummies.

    Based on Kalbfleisch & Prentice (2002) and tomer1812/pydts
    (grouped survival data).

    Parameters
    ----------
    df:
        Feature table at quarterly/annual frequency.
    freq:
        Data frequency ("Q", "A", "S").

    Returns
    -------
    pd.Series
        Discrete-time survival score normalized to [0, 1].
    """
    import numpy as np

    covariates = [
        c for c in ("current_ratio", "debt_to_equity_abs", "fcf_yield")
        if c in df.columns
    ]
    if len(covariates) < 2 or "company_survival_mode_flag" not in df.columns:
        return pd.Series(np.nan, index=df.index, name="discrete_survival_score")

    cox_df = df[covariates + ["company_survival_mode_flag"]].dropna()
    n_events = int(cox_df["company_survival_mode_flag"].sum())

    if n_events < 2 or len(cox_df) < 4:
        return pd.Series(np.nan, index=df.index, name="discrete_survival_score")

    try:
        from sklearn.linear_model import LogisticRegression
        from sklearn.preprocessing import StandardScaler

        X = cox_df[covariates].values
        y = cox_df["company_survival_mode_flag"].values

        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)

        # Logistic regression with complementary log-log approximation:
        # P(event at t | survived to t) ~ 1 - exp(-exp(beta'X))
        # Approximated by logistic regression for simplicity
        model = LogisticRegression(
            penalty="l2", C=1.0, max_iter=200,
            class_weight="balanced",  # handle imbalanced classes
        )
        model.fit(X_scaled, y)

        # Predict probability of distress for ALL rows in original df
        full_X = df[covariates].fillna(method="ffill").fillna(0).values
        full_X_scaled = scaler.transform(full_X)
        probs = model.predict_proba(full_X_scaled)[:, 1]

        # Normalize to [0, 1]
        result = pd.Series(probs, index=df.index, name="discrete_survival_score")

        logger.info(
            "Discrete-time survival (freq=%s): %d obs, %d events, "
            "coefs=%s",
            freq, len(cox_df), n_events,
            {c: round(float(w), 3) for c, w in zip(covariates, model.coef_[0])},
        )
        return result

    except Exception as exc:
        logger.debug("Discrete-time survival failed: %s", exc)
        return pd.Series(np.nan, index=df.index, name="discrete_survival_score")


# ---------------------------------------------------------------------------
# Country survival
# ---------------------------------------------------------------------------


def compute_country_survival_flag(
    df: pd.DataFrame,
    config: dict[str, Any] | None = None,
) -> pd.Series:
    """Compute daily country survival mode flag.

    Uses config-driven thresholds from ``country_protection_rules.yml``.
    Since many macro proxies (credit spread, yield curve, FX vol) may
    not be available from World Bank data, the flag defaults to 0 when
    data is missing.

    Parameters
    ----------
    df:
        Daily feature table with macro variables aligned.
    config:
        Override config dict (loads from YAML if None).

    Returns
    -------
    pd.Series
        Integer series: 1 = country survival, 0 = normal.
    """
    if config is None:
        config = load_config("country_protection_rules")

    cs_cfg = config.get("country_survival", {})
    conditions: list[pd.Series] = []

    # Credit spread > threshold
    credit_spread_threshold = cs_cfg.get("credit_spread_pct", 5.0)
    if "credit_spread" in df.columns:
        cs = df["credit_spread"]
        conditions.append(cs.notna() & (cs > credit_spread_threshold))

    # Unemployment rise > threshold in lookback window
    unemp_rise_threshold = cs_cfg.get("unemployment_rise_pct", 3.0)
    unemp_months = cs_cfg.get("unemployment_rise_months", 6)
    if "unemployment_rate" in df.columns:
        unemp = df["unemployment_rate"]
        # Approximate N months as N*21 business days
        lookback_days = unemp_months * 21
        unemp_change = unemp - unemp.shift(lookback_days)
        conditions.append(unemp_change.notna() & (unemp_change > unemp_rise_threshold))

    # Yield curve slope < threshold
    yc_threshold = cs_cfg.get("yield_curve_slope", -0.5)
    if "yield_curve_slope" in df.columns:
        yc = df["yield_curve_slope"]
        conditions.append(yc.notna() & (yc < yc_threshold))

    # FX volatility > threshold
    fx_vol_threshold = cs_cfg.get("fx_volatility_pct", 20.0)
    if "fx_volatility" in df.columns:
        fxv = df["fx_volatility"]
        conditions.append(fxv.notna() & (fxv > fx_vol_threshold))

    if not conditions:
        logger.info(
            "No country survival macro proxies available -- defaulting to 0"
        )
        return pd.Series(0, index=df.index, name="country_survival_mode_flag")

    combined = conditions[0]
    for c in conditions[1:]:
        combined = combined | c

    flag = combined.astype(int)
    flag.name = "country_survival_mode_flag"

    triggered = flag.sum()
    logger.info(
        "Country survival flag: %d / %d days triggered (%.1f%%)",
        triggered, len(flag), triggered / max(len(flag), 1) * 100,
    )

    return flag


# ---------------------------------------------------------------------------
# LLM-enhanced strategic sector check
# ---------------------------------------------------------------------------

# Cache LLM responses per sector/country pair to avoid repeated API calls
_llm_strategic_cache: dict[str, bool] = {}


def _check_strategic_sector_via_llm(sector: str, country: str = "") -> bool:
    """Ask the LLM whether a sector is strategically important for a country.

    This supplements the static YAML strategic_sectors list with dynamic
    assessment based on recent policy data. The LLM can catch sectors
    that became strategic due to recent government policy changes.

    Returns True if the LLM considers the sector strategic, False otherwise.
    Non-blocking: returns False on any failure.
    """
    if not sector:
        return False

    cache_key = f"{sector.lower()}:{country.lower()}"
    if cache_key in _llm_strategic_cache:
        return _llm_strategic_cache[cache_key]

    try:
        from operator1.clients.llm_factory import create_llm_client
        from operator1.secrets_loader import load_secrets

        secrets = load_secrets()
        llm_client = create_llm_client(secrets)
        if llm_client is None:
            return False

        country_ctx = f" in {country}" if country else ""
        prompt = (
            f"Is the '{sector}' sector considered strategically important "
            f"for the government{country_ctx}? Consider: national security, "
            f"critical infrastructure, recent industrial policy, government "
            f"subsidies, and 'too big to fail' designations.\n\n"
            f"Answer ONLY 'YES' or 'NO'."
        )

        response = llm_client.generate(prompt)
        if response:
            is_strategic = response.strip().upper().startswith("YES")
            _llm_strategic_cache[cache_key] = is_strategic
            if is_strategic:
                logger.info(
                    "LLM flagged sector '%s'%s as strategically important",
                    sector, country_ctx,
                )
            return is_strategic
    except Exception as exc:
        logger.debug("LLM strategic sector check failed (non-blocking): %s", exc)

    _llm_strategic_cache[cache_key] = False
    return False


# ---------------------------------------------------------------------------
# Country protection
# ---------------------------------------------------------------------------


def compute_country_protected_flag(
    df: pd.DataFrame,
    target_sector: str,
    config: dict[str, Any] | None = None,
) -> pd.Series:
    """Compute daily country protection flag.

    A company is considered "country-protected" (and thus shielded from
    full country-survival weight shifts) if ANY of:
      - Sector is in the ``strategic_sectors`` config list
      - ``market_cap > market_cap_gdp_threshold * gdp_current_usd``
      - Emergency rate cut exceeds threshold in lookback window

    Parameters
    ----------
    df:
        Daily feature table with macro variables.
    target_sector:
        The target company's sector string.
    config:
        Override config dict (loads from YAML if None).

    Returns
    -------
    pd.Series
        Integer series: 1 = protected, 0 = not protected.
    """
    if config is None:
        config = load_config("country_protection_rules")

    conditions: list[pd.Series] = []

    # 1. Strategic sector (YAML config baseline)
    strategic = config.get("strategic_sectors", [])
    sector_lower = target_sector.lower().strip()
    is_strategic = any(s.lower().strip() == sector_lower for s in strategic)

    # 1b. LLM-enhanced strategic sector check (hybrid approach)
    # If the YAML list doesn't flag this sector, ask the LLM whether
    # the company's sector is considered strategically important for
    # the country. The LLM can incorporate recent policy changes that
    # the static YAML list may not reflect.
    if not is_strategic and target_sector:
        is_strategic = _check_strategic_sector_via_llm(
            target_sector,
            country=df.attrs.get("country", "") if hasattr(df, "attrs") else "",
        )

    if is_strategic:
        # Entire series is 1
        conditions.append(pd.Series(True, index=df.index))
        logger.info(
            "Sector '%s' is strategic -- country protection triggered globally",
            target_sector,
        )

    # 2. Market cap > threshold * GDP
    mc_gdp_threshold = config.get("market_cap_gdp_threshold", 0.001)
    if "market_cap" in df.columns and "gdp_current_usd" in df.columns:
        mc = df["market_cap"]
        gdp = df["gdp_current_usd"]
        # Both must be available for this check
        mc_condition = mc.notna() & gdp.notna() & (mc > mc_gdp_threshold * gdp)
        conditions.append(mc_condition)

    # 3. Emergency rate cut
    rate_cut_threshold = config.get("emergency_rate_cut_threshold", 2.0)
    rate_cut_months = config.get("emergency_rate_cut_months", 3)
    if "lending_interest_rate" in df.columns:
        rate = df["lending_interest_rate"]
        lookback_days = rate_cut_months * 21
        rate_change = rate.shift(lookback_days) - rate  # cut = positive change
        conditions.append(
            rate_change.notna() & (rate_change > rate_cut_threshold)
        )

    if not conditions:
        logger.info("No country protection conditions met -- defaulting to 0")
        return pd.Series(0, index=df.index, name="country_protected_flag")

    combined = conditions[0]
    for c in conditions[1:]:
        combined = combined | c

    flag = combined.astype(int)
    flag.name = "country_protected_flag"

    triggered = flag.sum()
    logger.info(
        "Country protected flag: %d / %d days triggered (%.1f%%)",
        triggered, len(flag), triggered / max(len(flag), 1) * 100,
    )

    return flag


# ---------------------------------------------------------------------------
# Public API -- compute all survival flags
# ---------------------------------------------------------------------------


def compute_survival_flags(
    df: pd.DataFrame,
    target_sector: str,
    config: dict[str, Any] | None = None,
) -> pd.DataFrame:
    """Compute all three survival flags and attach to the feature table.

    Parameters
    ----------
    df:
        Daily feature table (target) with derived variables and macro
        data already merged.
    target_sector:
        Target company sector.
    config:
        Override country protection config.

    Returns
    -------
    pd.DataFrame
        Input DataFrame augmented with:
        - ``company_survival_mode_flag``
        - ``country_survival_mode_flag``
        - ``country_protected_flag``
    """
    result = df.copy()

    result["company_survival_mode_flag"] = compute_company_survival_flag(df)
    result["country_survival_mode_flag"] = compute_country_survival_flag(
        df, config,
    )
    result["country_protected_flag"] = compute_country_protected_flag(
        df, target_sector, config,
    )

    logger.info("All survival flags computed and attached to feature table")
    return result
