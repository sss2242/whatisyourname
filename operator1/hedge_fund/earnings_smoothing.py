"""HF-1.3 -- Earnings Smoothing Detector.

Extends Beneish M-Score with Benford's Law digit analysis, earnings
vs cash flow volatility ratio, and sequential surprise pattern analysis.

Primary data source: raw quarterly statement DataFrames.
"""

from __future__ import annotations

import logging
from collections import Counter

import numpy as np
import pandas as pd

from operator1.hedge_fund.helpers import (
    extract_quarterly_series,
    get_hf_weight,
    normalize_score,
    safe_divide,
)
from operator1.hedge_fund.types import SmoothingResult, _risk_label

logger = logging.getLogger(__name__)


def _benford_chi_squared(values: np.ndarray) -> float | None:
    """Compute chi-squared deviation from Benford's Law first-digit distribution.

    Returns a chi-squared statistic. Higher = more deviation from expected.
    None if insufficient data (<30 values).
    """
    min_samples = get_hf_weight("data_windows.benford_min_samples", 30)
    # Filter to positive values and extract first digit
    positive = values[values > 0]
    if len(positive) < min_samples:
        return None

    first_digits = []
    for v in positive:
        s = f"{v:.0f}"
        if s and s[0].isdigit() and s[0] != "0":
            first_digits.append(int(s[0]))

    if len(first_digits) < min_samples:
        return None

    # Expected Benford distribution
    benford_expected = {d: np.log10(1 + 1 / d) for d in range(1, 10)}

    observed_counts = Counter(first_digits)
    n = len(first_digits)

    chi_sq = 0.0
    for digit in range(1, 10):
        observed = observed_counts.get(digit, 0)
        expected = benford_expected[digit] * n
        if expected > 0:
            chi_sq += (observed - expected) ** 2 / expected

    return chi_sq


def compute_earnings_smoothing(
    income_df: pd.DataFrame,
    cashflow_df: pd.DataFrame,
    cache: pd.DataFrame | None = None,
    fh_result: object | None = None,
) -> SmoothingResult:
    """Compute earnings smoothing index from raw quarterly filings.

    Parameters
    ----------
    income_df, cashflow_df:
        Raw statement DataFrames.
    cache:
        Daily cache (for filing-day price reactions).
    fh_result:
        FinancialHealthResult (for Beneish M-Score if available).
    """
    result = SmoothingResult()
    n_periods = get_hf_weight("data_windows.quarterly_lookback", 8)
    weights = get_hf_weight("earnings_quality.earnings_smoothing", {})
    w_ben = weights.get("beneish_weight", 0.25)
    w_vol = weights.get("vol_ratio_weight", 0.25)
    w_benford = weights.get("benford_weight", 0.20)
    w_surprise = weights.get("surprise_pattern_weight", 0.15)
    w_restate = weights.get("restatement_weight", 0.15)

    try:
        ni_series = extract_quarterly_series(income_df, "net_income", n_periods)
        ocf_series = extract_quarterly_series(cashflow_df, "operating_cash_flow", n_periods)
        rev_series = extract_quarterly_series(income_df, "revenue", n_periods * 4)  # more for Benford
        eps_series = extract_quarterly_series(income_df, "eps", n_periods)
        if eps_series.empty:
            eps_series = extract_quarterly_series(income_df, "eps_diluted", n_periods)

        # --- Component 1: Beneish M-Score probability ---
        beneish_score = 50.0
        if fh_result is not None:
            m_score = getattr(fh_result, "beneish_m_score", None)
            if m_score is None:
                # Try nested attribute
                beneish_dict = getattr(fh_result, "altman_z", None)
                if isinstance(fh_result, dict):
                    m_score = fh_result.get("beneish_m_score")
            if m_score is not None:
                result.beneish_probability = float(m_score)
                # M > -2.22 = likely manipulator (higher M = worse)
                # Score: 0-100 where 100 = high manipulation risk
                if m_score > -1.78:
                    beneish_score = 90.0  # very likely manipulator
                elif m_score > -2.22:
                    beneish_score = 70.0  # probable manipulator
                elif m_score > -2.76:
                    beneish_score = 40.0  # grey zone
                else:
                    beneish_score = 10.0  # unlikely manipulator

        # --- Component 2: Earnings vol / Cash flow vol ratio ---
        vol_ratio_score = 50.0
        if len(ni_series) >= 4 and len(ocf_series) >= 4:
            ni_vol = float(ni_series.std())
            ocf_vol = float(ocf_series.std())
            if ocf_vol > 1e-6:
                ratio = ni_vol / ocf_vol
                result.earnings_vol_ratio = ratio
                # Real earnings should be AT LEAST as volatile as cash flows
                # ratio < 0.5 = suspicious smoothing (earnings less volatile)
                # ratio > 1.0 = normal
                if ratio < 0.3:
                    vol_ratio_score = 90.0
                elif ratio < 0.5:
                    vol_ratio_score = 70.0
                elif ratio < 0.8:
                    vol_ratio_score = 40.0
                else:
                    vol_ratio_score = 15.0  # normal

        # --- Component 3: Benford's Law digit analysis ---
        benford_score = 50.0  # neutral if insufficient data
        all_rev = extract_quarterly_series(income_df, "revenue", n_periods * 4)
        if len(all_rev) >= 10:
            # Combine revenue, NI, and total_assets for larger sample
            all_values = all_rev.values
            chi_sq = _benford_chi_squared(all_values)
            if chi_sq is not None:
                result.benford_deviation = chi_sq
                # Chi-sq critical value for 8 df at 0.05 = 15.51
                # Higher chi-sq = more deviation from Benford
                if chi_sq > 25:
                    benford_score = 85.0
                elif chi_sq > 15.51:
                    benford_score = 65.0
                elif chi_sq > 10:
                    benford_score = 40.0
                else:
                    benford_score = 15.0

        # --- Component 4: Sequential surprise pattern ---
        surprise_score = 50.0
        if len(eps_series) >= 6:
            # Check for suspiciously consistent beats
            eps_changes = eps_series.diff().dropna()
            if len(eps_changes) >= 4:
                # Count consecutive positive changes
                signs = (eps_changes > 0).astype(int)
                max_consecutive = 0
                current_run = 0
                for s in signs:
                    if s:
                        current_run += 1
                        max_consecutive = max(max_consecutive, current_run)
                    else:
                        current_run = 0
                result.sequential_surprise_pattern = float(max_consecutive)
                # 6+ consecutive beats is statistically unlikely without smoothing
                if max_consecutive >= 6:
                    surprise_score = 75.0
                elif max_consecutive >= 4:
                    surprise_score = 55.0
                else:
                    surprise_score = 25.0

        # --- Component 5: Restatement risk (proxy) ---
        restatement_score = 50.0
        # Use accruals growth as a proxy (no restatement data from free APIs)
        if len(ni_series) >= 4 and len(ocf_series) >= 4:
            common = ni_series.index.intersection(ocf_series.index)
            if len(common) >= 4:
                accruals = ni_series.loc[common] - ocf_series.loc[common]
                accruals_growth = accruals.pct_change().dropna()
                if len(accruals_growth) >= 2:
                    # Rapidly growing accruals = restatement risk
                    max_growth = float(accruals_growth.abs().max())
                    result.restatement_probability = min(1.0, max_growth)
                    restatement_score = normalize_score(max_growth * 200, 0, 100)

        # --- Composite (higher = more smoothing risk) ---
        composite = (
            w_ben * beneish_score
            + w_vol * vol_ratio_score
            + w_benford * benford_score
            + w_surprise * surprise_score
            + w_restate * restatement_score
        )
        result.smoothing_index = normalize_score(composite)
        result.label = _risk_label(result.smoothing_index)

        # Narrative
        parts = []
        if result.earnings_vol_ratio is not None and result.earnings_vol_ratio < 0.5:
            parts.append(f"NI/OCF volatility ratio {result.earnings_vol_ratio:.2f} (earnings suspiciously smooth)")
        if result.benford_deviation is not None and result.benford_deviation > 15.51:
            parts.append(f"Benford deviation chi2={result.benford_deviation:.1f} (significant)")
        if result.beneish_probability is not None and result.beneish_probability > -2.22:
            parts.append(f"Beneish M={result.beneish_probability:.2f} (manipulation risk)")
        # --- Phase 1: Beneish M5-Score (5-variable variant, Beneish 1999) ---
        try:
            if fh_result is not None:
                m8 = getattr(fh_result, "fh_beneish_m_score", None)
                if m8 is None:
                    # Try attribute name variants
                    m8 = getattr(fh_result, "beneish_m_score", None)

                # M5 uses only 5 of the 8 variables: DSRI, GMI, AQI, SGI, DEPI
                # M5 = -6.065 + 0.823*DSRI + 0.906*GMI + 0.593*AQI + 0.717*SGI + 0.107*DEPI
                # We compute M5 from raw data if we can, otherwise approximate from M8
                _dsri = _gmii = _aqi = _sgi = _depi = None

                if len(rev_series) >= 2 and len(ni_series) >= 2:
                    rev_arr = rev_series.values
                    rec_s = extract_quarterly_series(income_df, "receivables", n_periods)
                    # DSRI: (Receivables_t / Revenue_t) / (Receivables_t-1 / Revenue_t-1)
                    if len(rec_s) >= 2 and len(rev_arr) >= 2:
                        r0 = safe_divide(float(rec_s.iloc[-1]), float(rev_arr[-1]))
                        r1 = safe_divide(float(rec_s.iloc[-2]), float(rev_arr[-2]))
                        if r0 is not None and r1 is not None and r1 > 0:
                            _dsri = r0 / r1

                    # SGI: Revenue_t / Revenue_t-1
                    if abs(float(rev_arr[-2])) > 1e-6:
                        _sgi = float(rev_arr[-1]) / float(rev_arr[-2])

                    # GMI: Gross_margin_t-1 / Gross_margin_t
                    gp_s = extract_quarterly_series(income_df, "gross_profit", n_periods)
                    if len(gp_s) >= 2 and len(rev_arr) >= 2:
                        gm_curr = safe_divide(float(gp_s.iloc[-1]), float(rev_arr[-1]))
                        gm_prev = safe_divide(float(gp_s.iloc[-2]), float(rev_arr[-2]))
                        if gm_curr is not None and gm_prev is not None and gm_curr > 0:
                            _gmii = gm_prev / gm_curr

                if _dsri is not None and _sgi is not None:
                    # Use defaults for missing components
                    _dsri = _dsri if _dsri is not None else 1.0
                    _gmii = _gmii if _gmii is not None else 1.0
                    _aqi = 1.0  # default (requires detailed asset quality data)
                    _depi = 1.0  # default (requires detailed depreciation data)

                    m5 = (-6.065
                          + 0.823 * _dsri
                          + 0.906 * _gmii
                          + 0.593 * _aqi
                          + 0.717 * _sgi
                          + 0.107 * _depi)
                    result.beneish_m5_score = round(m5, 4)

                    import math
                    prob_m5 = 1.0 / (1.0 + math.exp(-m5))
                    result.beneish_m5_probability = round(prob_m5, 4)

                    # Agreement: both M5 and M8 flag (or both clear)
                    if m8 is not None:
                        m8_flag = m8 > -2.22
                        m5_flag = m5 > -2.22
                        result.beneish_m5_m8_agreement = (m8_flag == m5_flag)

                    if result.beneish_m5_probability > 0.5:
                        parts.append(f"Beneish M5 P(manipulator)={result.beneish_m5_probability:.0%}")
        except Exception as _m5_exc:
            logger.debug("Beneish M5 computation failed: %s", _m5_exc)

        result.narrative = "; ".join(parts) if parts else "No significant smoothing detected"

        result.available = True

    except Exception as exc:
        logger.warning("Earnings smoothing detection failed: %s", exc)
        result.error = str(exc)

    return result
