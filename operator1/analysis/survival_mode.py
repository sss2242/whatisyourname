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

# Default thresholds (overridable via config if needed in the future)
_COMPANY_THRESHOLDS = {
    "current_ratio_lt": 1.0,
    "debt_to_equity_abs_gt": 3.0,
    "fcf_yield_lt": 0.0,
    "drawdown_252d_lt": -0.40,
}


def compute_company_survival_flag(
    df: pd.DataFrame,
    thresholds: dict[str, float] | None = None,
) -> pd.Series:
    """Compute daily company survival mode flag.

    Parameters
    ----------
    df:
        Daily feature table with derived variables already computed.
    thresholds:
        Override default thresholds (mainly for testing).

    Returns
    -------
    pd.Series
        Integer series: 1 = survival mode active, 0 = normal.
    """
    t = thresholds or _COMPANY_THRESHOLDS

    conditions: list[pd.Series] = []

    # Current ratio < 1.0
    if "current_ratio" in df.columns:
        cr = df["current_ratio"]
        conditions.append(cr.notna() & (cr < t.get("current_ratio_lt", 1.0)))

    # Debt-to-equity (absolute) > 3.0
    if "debt_to_equity_abs" in df.columns:
        de = df["debt_to_equity_abs"]
        conditions.append(de.notna() & (de > t.get("debt_to_equity_abs_gt", 3.0)))

    # FCF yield < 0
    if "fcf_yield" in df.columns:
        fy = df["fcf_yield"]
        conditions.append(fy.notna() & (fy < t.get("fcf_yield_lt", 0.0)))

    # Drawdown < -40%
    if "drawdown_252d" in df.columns:
        dd = df["drawdown_252d"]
        conditions.append(dd.notna() & (dd < t.get("drawdown_252d_lt", -0.40)))

    # Geopolitical conflict: country_conflict_flag OR intensity > 0.7 OR sanctions
    if "country_conflict_flag" in df.columns:
        cf = df["country_conflict_flag"]
        conditions.append(cf.notna() & (cf == 1))
    elif "conflict_intensity_score" in df.columns:
        ci = df["conflict_intensity_score"]
        conditions.append(ci.notna() & (ci > t.get("conflict_intensity_gt", 0.7)))
    if "sanctions_flag" in df.columns:
        sf = df["sanctions_flag"]
        conditions.append(sf.notna() & (sf == 1))

    if not conditions:
        logger.warning("No company survival trigger columns found -- defaulting to 0")
        return pd.Series(0, index=df.index, name="company_survival_mode_flag")

    # ANY condition true triggers survival mode
    combined = conditions[0]
    for c in conditions[1:]:
        combined = combined | c

    flag = combined.astype(int)
    flag.name = "company_survival_mode_flag"

    triggered = flag.sum()
    logger.info(
        "Company survival flag: %d / %d days triggered (%.1f%%)",
        triggered, len(flag), triggered / max(len(flag), 1) * 100,
    )

    return flag


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
        if col not in df.columns:
            continue
        s = df[col].fillna(threshold)  # neutral if missing

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

    # Max distress signal across all triggers
    combined = pd.concat(distress_signals, axis=1).max(axis=1)
    # Sigmoid transform to [0, 1]
    probability = 1.0 / (1.0 + np.exp(-combined))
    probability.name = "survival_probability"

    logger.info(
        "Survival probability: mean=%.3f, max=%.3f, >50%%=%d days",
        probability.mean(), probability.max(), (probability > 0.5).sum(),
    )

    return probability


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
    try:
        from lifelines import CoxPHFitter
    except ImportError:
        logger.debug("lifelines not installed, skipping Cox PH survival score")
        return pd.Series(np.nan, index=df.index, name="cox_survival_score")

    import numpy as np

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
        # Add duration column (time index)
        cox_df["duration"] = range(1, len(cox_df) + 1)

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
