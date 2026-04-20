"""Advanced hedge fund methods -- 15 additional analytical techniques.

P1 (7 methods): Piotroski F-Score, Balance Sheet Velocity, Earnings
    Persistence Transitions, Forensic Cash Flow (Mulford), OU Mean
    Reversion, Accruals Percentile Rank, Altman Z'' for non-US.

P2 (5 methods): GARCH Vol Term Structure, Insider Alignment Score,
    Capital Cycle Position, Earnings Torpedo Detection, VRP Proxy.

P3 (3 methods): Fama-French Factor Decomposition (stub -- needs
    external data), Implied Cost of Capital, Cross-Asset Regime.

All methods consume raw quarterly statement DFs or the daily cache.
Results are stored in HedgeFundResult.advanced dict and fed into
the thesis scorecard.
"""

from __future__ import annotations

import logging
import math
from dataclasses import dataclass, field
from typing import Any

import numpy as np
import pandas as pd

from operator1.hedge_fund.helpers import (
    extract_quarterly_series,
    extract_latest_value,
    extract_qoq_changes,
    compute_rolling_slope,
    safe_divide,
    normalize_score,
    get_cache_latest,
)

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Result container for all advanced methods
# ---------------------------------------------------------------------------

@dataclass
class AdvancedMethodsResult:
    """Combined result from all 15 advanced HF methods."""

    available: bool = False

    # P1 methods
    piotroski_f_score: int = 0          # 0-9
    piotroski_n_available: int = 0      # how many of 9 signals had data (0-9)
    piotroski_label: str = "unknown"    # strong/moderate/weak
    balance_sheet_velocity: dict = field(default_factory=dict)
    earnings_persistence: dict = field(default_factory=dict)
    forensic_cashflow: dict = field(default_factory=dict)
    ou_mean_reversion: dict = field(default_factory=dict)
    accruals_rank: float | None = None  # 0-100 percentile
    altman_z_double_prime: float | None = None
    altman_z_dp_zone: str = ""          # safe/grey/distress

    # P2 methods
    garch_vol_term_structure: dict = field(default_factory=dict)
    insider_alignment: dict = field(default_factory=dict)
    capital_cycle: dict = field(default_factory=dict)
    earnings_torpedo: dict = field(default_factory=dict)
    vrp_proxy: dict = field(default_factory=dict)

    # P3 methods
    fama_french_alpha: float | None = None
    implied_cost_of_capital: float | None = None
    cross_asset_regime: dict = field(default_factory=dict)


# ---------------------------------------------------------------------------
# P1: Piotroski F-Score (9 binary signals)
# ---------------------------------------------------------------------------

def compute_piotroski_f_score(
    income_df: pd.DataFrame,
    balance_df: pd.DataFrame,
    cashflow_df: pd.DataFrame,
) -> tuple[int, str]:
    """Compute Piotroski F-Score (0-9) from raw quarterly filings.

    Each signal is 1 (good) or 0 (bad). Score >= 7 = strong, <= 2 = weak.
    """
    score = 0
    n_available = 0  # track how many signals have sufficient data
    try:
        ni = extract_latest_value(income_df, "net_income")
        ocf = extract_latest_value(cashflow_df, "operating_cash_flow")
        ta = extract_latest_value(balance_df, "total_assets")
        ta_prev = None
        ta_series = extract_quarterly_series(balance_df, "total_assets", 5)
        if len(ta_series) >= 2:
            ta_prev = float(ta_series.iloc[-2])

        rev = extract_latest_value(income_df, "revenue")
        rev_prev = None
        rev_series = extract_quarterly_series(income_df, "revenue", 5)
        if len(rev_series) >= 2:
            rev_prev = float(rev_series.iloc[-2])

        gm = extract_latest_value(income_df, "gross_profit")
        gm_prev = None
        gm_series = extract_quarterly_series(income_df, "gross_profit", 5)
        if len(gm_series) >= 2:
            gm_prev = float(gm_series.iloc[-2])

        debt = extract_latest_value(balance_df, "total_debt")
        debt_prev = None
        debt_series = extract_quarterly_series(balance_df, "total_debt", 5)
        if len(debt_series) >= 2:
            debt_prev = float(debt_series.iloc[-2])

        cr = None
        ca = extract_latest_value(balance_df, "current_assets")
        cl = extract_latest_value(balance_df, "current_liabilities")
        if ca and cl and cl > 0:
            cr = ca / cl
        cr_prev = None
        ca_s = extract_quarterly_series(balance_df, "current_assets", 5)
        cl_s = extract_quarterly_series(balance_df, "current_liabilities", 5)
        if len(ca_s) >= 2 and len(cl_s) >= 2:
            ca_p, cl_p = float(ca_s.iloc[-2]), float(cl_s.iloc[-2])
            if cl_p > 0:
                cr_prev = ca_p / cl_p

        shares = extract_latest_value(balance_df, "total_equity")
        shares_prev = None
        eq_s = extract_quarterly_series(balance_df, "total_equity", 5)
        if len(eq_s) >= 2:
            shares_prev = float(eq_s.iloc[-2])

        # Signal 1: Positive net income
        if ni is not None:
            n_available += 1
            if ni > 0:
                score += 1
        # Signal 2: Positive OCF
        if ocf is not None:
            n_available += 1
            if ocf > 0:
                score += 1
        # Signal 3: Rising ROA (Y/Y)
        if ni is not None and ta is not None and ta_prev is not None and ta > 0 and ta_prev > 0:
            n_available += 1
            roa_curr = ni / ta
            ni_prev = None
            ni_s = extract_quarterly_series(income_df, "net_income", 5)
            if len(ni_s) >= 2:
                ni_prev = float(ni_s.iloc[-2])
            if ni_prev is not None:
                roa_prev = ni_prev / ta_prev
                if roa_curr > roa_prev:
                    score += 1
        # Signal 4: OCF > NI (quality)
        if ocf is not None and ni is not None:
            n_available += 1
            if ocf > ni:
                score += 1
        # Signal 5: Decreasing leverage
        if debt is not None and debt_prev is not None:
            n_available += 1
            if debt < debt_prev:
                score += 1
        # Signal 6: Increasing current ratio
        if cr is not None and cr_prev is not None:
            n_available += 1
            if cr > cr_prev:
                score += 1
        # Signal 7: No equity dilution
        if shares is not None and shares_prev is not None:
            n_available += 1
            if shares >= shares_prev:
                score += 1
        # Signal 8: Rising gross margin
        if gm is not None and gm_prev is not None and rev is not None and rev_prev is not None:
            if rev > 0 and rev_prev > 0:
                n_available += 1
                gm_pct = gm / rev
                gm_pct_prev = gm_prev / rev_prev
                if gm_pct > gm_pct_prev:
                    score += 1
        # Signal 9: Rising asset turnover
        if rev is not None and ta is not None and rev_prev is not None and ta_prev is not None:
            if ta > 0 and ta_prev > 0:
                n_available += 1
                at_curr = rev / ta
                at_prev = rev_prev / ta_prev
                if at_curr > at_prev:
                    score += 1

    except Exception as exc:
        logger.debug("Piotroski F-Score failed: %s", exc)

    # Normalize score by available signals to prevent NaN-as-zero penalty
    if n_available < 9 and n_available >= 4:
        normalized = round(score / n_available * 9)
        label = "strong" if normalized >= 7 else "weak" if normalized <= 2 else "moderate"
        label += f" ({n_available}/9 available)"
        return normalized, n_available, label
    elif n_available < 4:
        label = f"insufficient ({n_available}/9 available)"
        return score, n_available, label
    else:
        label = "strong" if score >= 7 else "weak" if score <= 2 else "moderate"
        return score, n_available, label


# ---------------------------------------------------------------------------
# P1: Balance Sheet Velocity
# ---------------------------------------------------------------------------

def compute_balance_sheet_velocity(
    income_df: pd.DataFrame,
    balance_df: pd.DataFrame,
    cashflow_df: pd.DataFrame,
    n_periods: int = 8,
) -> dict:
    """Rate-of-change analysis across most recent filings.

    Returns acceleration/deceleration flags for key BS items.
    """
    result = {"available": False, "items": {}}
    try:
        items = {
            "receivables": ("balance", "receivables"),
            "inventory": ("balance", "inventory"),
            "goodwill": ("balance", "goodwill"),
            "total_debt": ("balance", "total_debt"),
            "cash": ("balance", "cash_and_equivalents"),
            "revenue": ("income", "revenue"),
        }
        sources = {"balance": balance_df, "income": income_df}

        for name, (src_type, col) in items.items():
            df = sources.get(src_type)
            if df is None:
                continue
            series = extract_quarterly_series(df, col, n_periods)
            if len(series) >= 4:
                pct_changes = series.pct_change().dropna()
                if len(pct_changes) >= 2:
                    accel = float(pct_changes.diff().iloc[-1]) if len(pct_changes) >= 3 else 0.0
                    latest_change = float(pct_changes.iloc[-1])
                    result["items"][name] = {
                        "latest_qoq_pct": round(latest_change * 100, 1),
                        "acceleration": round(accel * 100, 1),
                        "flag": (
                            "accelerating" if accel > 0.02
                            else "decelerating" if accel < -0.02
                            else "stable"
                        ),
                    }

        # Composite velocity score: fast-deteriorating items count as risk
        risk_items = 0
        for name, data in result["items"].items():
            if name in ("receivables", "inventory", "goodwill", "total_debt"):
                if data["flag"] == "accelerating" and data["latest_qoq_pct"] > 5:
                    risk_items += 1
            elif name == "cash":
                if data["flag"] == "decelerating" or data["latest_qoq_pct"] < -5:
                    risk_items += 1

        result["velocity_risk_score"] = min(100, risk_items * 25)
        result["available"] = len(result["items"]) >= 2

    except Exception as exc:
        logger.debug("BS velocity failed: %s", exc)
    return result


# ---------------------------------------------------------------------------
# P1: Earnings Persistence Transitions
# ---------------------------------------------------------------------------

def compute_earnings_persistence(income_df: pd.DataFrame, n_periods: int = 12) -> dict:
    """Build a 3x3 Markov transition matrix from quarterly EPS surprises."""
    result = {"available": False, "transition_matrix": {}, "persistence_score": 0.0}
    try:
        eps = extract_quarterly_series(income_df, "eps", n_periods)
        if eps.empty:
            eps = extract_quarterly_series(income_df, "eps_diluted", n_periods)
        if len(eps) < 6:
            return result

        # Classify each quarter as beat/miss/inline (vs 4Q ago)
        surprises = eps - eps.shift(4)
        surprises = surprises.dropna()
        if len(surprises) < 4:
            return result

        threshold = float(surprises.std()) * 0.5
        states = []
        for s in surprises:
            if s > threshold:
                states.append("beat")
            elif s < -threshold:
                states.append("miss")
            else:
                states.append("inline")

        # Build transition matrix
        transitions = {"beat": {"beat": 0, "miss": 0, "inline": 0},
                       "miss": {"beat": 0, "miss": 0, "inline": 0},
                       "inline": {"beat": 0, "miss": 0, "inline": 0}}
        for i in range(len(states) - 1):
            transitions[states[i]][states[i + 1]] += 1

        # Normalize to probabilities
        for from_state in transitions:
            total = sum(transitions[from_state].values())
            if total > 0:
                for to_state in transitions[from_state]:
                    transitions[from_state][to_state] = round(
                        transitions[from_state][to_state] / total, 3
                    )

        result["transition_matrix"] = transitions
        result["current_state"] = states[-1] if states else "unknown"
        # Persistence = probability of staying in current state
        current = result["current_state"]
        result["persistence_score"] = transitions.get(current, {}).get(current, 0.0)
        result["available"] = True

    except Exception as exc:
        logger.debug("Earnings persistence failed: %s", exc)
    return result


# ---------------------------------------------------------------------------
# P1: Forensic Cash Flow Analysis (Mulford & Comiskey)
# ---------------------------------------------------------------------------

def compute_forensic_cashflow(
    income_df: pd.DataFrame,
    balance_df: pd.DataFrame,
    cashflow_df: pd.DataFrame,
    n_periods: int = 8,
) -> dict:
    """Detect 4 specific cash flow manipulation techniques."""
    result = {"available": False, "flags": [], "risk_score": 0}
    try:
        payables = extract_quarterly_series(balance_df, "payables", n_periods)
        receivables = extract_quarterly_series(balance_df, "receivables", n_periods)
        revenue = extract_quarterly_series(income_df, "revenue", n_periods)
        capex = extract_quarterly_series(cashflow_df, "capex", n_periods)
        ocf = extract_quarterly_series(cashflow_df, "operating_cash_flow", n_periods)

        flags = []
        # 1. Supplier financing: payables days increasing while revenue flat
        if len(payables) >= 4 and len(revenue) >= 4:
            common = payables.index.intersection(revenue.index)
            if len(common) >= 4:
                safe_rev = revenue.loc[common].where(revenue.loc[common].abs() > 1e-6)
                pay_days = payables.loc[common] / (safe_rev / 90)
                pay_days = pay_days.dropna()
                if len(pay_days) >= 4:
                    slope = compute_rolling_slope(pay_days, len(pay_days))
                    if slope is not None and slope > 1.0:
                        flags.append({"type": "supplier_financing", "detail": f"Payables days rising (slope={slope:.1f}d/Q)", "severity": "high"})

        # 2. Receivables factoring: sudden drop in receivables without revenue decline
        if len(receivables) >= 4 and len(revenue) >= 4:
            rec_change = receivables.pct_change().dropna()
            rev_change = revenue.pct_change().dropna()
            if len(rec_change) >= 2 and len(rev_change) >= 2:
                latest_rec = float(rec_change.iloc[-1])
                latest_rev = float(rev_change.iloc[-1])
                if latest_rec < -0.15 and latest_rev > -0.05:
                    flags.append({"type": "receivables_factoring", "detail": f"Receivables down {latest_rec:.0%} while revenue {latest_rev:+.0%}", "severity": "medium"})

        # 3. CapEx reclassification: declining capex/revenue ratio while assets grow
        if len(capex) >= 4 and len(revenue) >= 4:
            common_c = capex.index.intersection(revenue.index)
            if len(common_c) >= 4:
                safe_rev2 = revenue.loc[common_c].where(revenue.loc[common_c].abs() > 1e-6)
                capex_ratio = capex.loc[common_c].abs() / safe_rev2
                capex_ratio = capex_ratio.dropna()
                if len(capex_ratio) >= 4:
                    slope = compute_rolling_slope(capex_ratio, len(capex_ratio))
                    if slope is not None and slope < -0.005:
                        flags.append({"type": "capex_reclassification", "detail": "CapEx/Revenue declining (possible move to investing CF)", "severity": "medium"})

        # 4. Working capital smoothing: OCF volatility much lower than expected
        if len(ocf) >= 6 and len(revenue) >= 6:
            ocf_cv = float(ocf.std() / ocf.abs().mean()) if ocf.abs().mean() > 1e-6 else 0
            rev_cv = float(revenue.std() / revenue.abs().mean()) if revenue.abs().mean() > 1e-6 else 0
            if rev_cv > 0 and ocf_cv < rev_cv * 0.3:
                flags.append({"type": "wc_smoothing", "detail": f"OCF suspiciously smooth (CV={ocf_cv:.2f} vs Revenue CV={rev_cv:.2f})", "severity": "low"})

        result["flags"] = flags
        result["risk_score"] = min(100, len(flags) * 30)
        result["available"] = True

    except Exception as exc:
        logger.debug("Forensic cashflow failed: %s", exc)
    return result


# ---------------------------------------------------------------------------
# P1: OU Mean Reversion Speed
# ---------------------------------------------------------------------------

def compute_ou_mean_reversion(cache: pd.DataFrame | None, lookback: int = 252) -> dict:
    """Fit Ornstein-Uhlenbeck process to close prices.

    dX = theta*(mu - X)*dt + sigma*dW
    Half-life = ln(2) / theta
    """
    result = {"available": False}
    if cache is None or "close" not in cache.columns:
        return result
    try:
        closes = cache["close"].dropna()
        if len(closes) < 60:
            return result
        recent = closes.iloc[-lookback:] if len(closes) >= lookback else closes
        x = recent.values
        n = len(x)

        # OLS regression: x[t+1] - x[t] = a + b*x[t] + epsilon
        dx = np.diff(x)
        x_lag = x[:-1]
        X = np.column_stack([np.ones(n - 1), x_lag])
        y = dx

        coeffs = np.linalg.lstsq(X, y, rcond=None)[0]
        a, b = coeffs

        # OU parameters
        # b should be negative for mean-reversion
        if b >= 0:
            result["theta"] = 0.0
            result["half_life_days"] = float("inf")
            result["mean_reverting"] = False
        else:
            theta = -b  # reversion speed (positive)
            mu = -a / b  # long-run mean
            residuals = y - X @ coeffs
            sigma = float(np.std(residuals))
            half_life = np.log(2) / theta

            result["theta"] = round(theta, 6)
            result["mu"] = round(mu, 2)
            result["sigma"] = round(sigma, 4)
            result["half_life_days"] = round(half_life, 1)
            result["mean_reverting"] = True
            result["current_deviation_pct"] = round((float(x[-1]) - mu) / mu * 100, 1) if mu != 0 else 0

        result["available"] = True

    except Exception as exc:
        logger.debug("OU mean reversion failed: %s", exc)
    return result


# ---------------------------------------------------------------------------
# P1: Altman Z'' (non-manufacturing, non-US)
# ---------------------------------------------------------------------------

def compute_altman_z_double_prime(
    income_df: pd.DataFrame,
    balance_df: pd.DataFrame,
) -> tuple[float | None, str]:
    """Altman Z'' for non-manufacturing and non-US companies.

    Z'' = 6.56*(WC/TA) + 3.26*(RE/TA) + 6.72*(EBIT/TA) + 1.05*(BV_Equity/TL)
    Z'' > 2.60: Safe, < 1.10: Distress, Between: Grey
    """
    try:
        ta = extract_latest_value(balance_df, "total_assets")
        ca = extract_latest_value(balance_df, "current_assets")
        cl = extract_latest_value(balance_df, "current_liabilities")
        re = extract_latest_value(balance_df, "retained_earnings")
        ebit = extract_latest_value(income_df, "ebit")
        equity = extract_latest_value(balance_df, "total_equity")
        tl = extract_latest_value(balance_df, "total_liabilities")

        if ta is None or ta <= 0:
            return None, ""

        # Track which terms have valid inputs to normalize partial data
        z = 0.0
        n_terms = 0
        _coefficients = []

        wc = None
        if ca is not None and cl is not None:
            wc = ca - cl
        if wc is not None:
            z += 6.56 * safe_divide(wc, ta, 0)
            n_terms += 1
            _coefficients.append("WC/TA")
        if re is not None:
            z += 3.26 * safe_divide(re, ta, 0)
            n_terms += 1
            _coefficients.append("RE/TA")
        if ebit is not None:
            z += 6.72 * safe_divide(ebit, ta, 0)
            n_terms += 1
            _coefficients.append("EBIT/TA")
        if equity is not None and tl is not None and tl > 0:
            z += 1.05 * safe_divide(equity, tl, 0)
            n_terms += 1
            _coefficients.append("BV/TL")

        if n_terms == 0:
            return None, ""

        # Normalize if partial: scale up proportionally
        if n_terms < 4:
            z = z / n_terms * 4
            logger.debug("Altman Z'' partial: %d/4 terms (%s)", n_terms, ", ".join(_coefficients))

        zone = "safe" if z > 2.60 else "distress" if z < 1.10 else "grey"
        if n_terms < 4:
            zone += f" ({n_terms}/4 terms)"
        return round(z, 3), zone

    except Exception as exc:
        logger.debug("Altman Z'' failed: %s", exc)
        return None, ""


# ---------------------------------------------------------------------------
# P2: GARCH Vol Term Structure
# ---------------------------------------------------------------------------

def compute_garch_vol_term_structure(cache: pd.DataFrame | None) -> dict:
    """Synthetic vol term structure from multi-horizon realized vol.

    Inverted term structure (short > long) historically precedes large moves.
    """
    result = {"available": False}
    if cache is None or "return_1d" not in cache.columns:
        return result
    try:
        returns = cache["return_1d"].dropna()
        if len(returns) < 63:
            return result

        # Compute realized vol at multiple horizons
        vol_5d = float(returns.iloc[-5:].std() * np.sqrt(252))
        vol_21d = float(returns.iloc[-21:].std() * np.sqrt(252))
        vol_63d = float(returns.iloc[-63:].std() * np.sqrt(252))
        vol_252d = float(returns.std() * np.sqrt(252)) if len(returns) >= 252 else vol_63d

        result["vol_5d"] = round(vol_5d, 4)
        result["vol_21d"] = round(vol_21d, 4)
        result["vol_63d"] = round(vol_63d, 4)
        result["vol_252d"] = round(vol_252d, 4)

        # Term structure slope: positive = normal (long > short), negative = inverted
        result["slope_5d_63d"] = round(vol_63d - vol_5d, 4)
        result["inverted"] = vol_5d > vol_63d * 1.15  # 15% inversion threshold
        result["available"] = True

    except Exception as exc:
        logger.debug("GARCH vol TS failed: %s", exc)
    return result


# ---------------------------------------------------------------------------
# P2: Insider Alignment Score
# ---------------------------------------------------------------------------

def compute_insider_alignment(
    cache: pd.DataFrame | None,
    income_df: pd.DataFrame | None = None,
    balance_df: pd.DataFrame | None = None,
) -> dict:
    """Management-shareholder alignment score from insider behavior + SGA."""
    result = {"available": False, "score": 50, "label": "neutral"}
    try:
        components = []

        # Insider signal from cache (from institutional_flow.py)
        if cache is not None and "inst_insider_signal" in cache.columns:
            insider_sig = cache["inst_insider_signal"].dropna()
            if len(insider_sig) > 0:
                val = float(insider_sig.iloc[-1])
                # Positive = net buying = aligned
                components.append(("insider_buying", normalize_score(50 + val * 100, 0, 100)))

        # SGA trend vs revenue trend (empire-building)
        if income_df is not None:
            sga = extract_quarterly_series(income_df, "sga_expenses", 8)
            rev = extract_quarterly_series(income_df, "revenue", 8)
            if len(sga) >= 4 and len(rev) >= 4:
                sga_growth = float(sga.pct_change().dropna().mean())
                rev_growth = float(rev.pct_change().dropna().mean())
                # SGA growing faster than revenue = empire building
                gap = sga_growth - rev_growth
                components.append(("sga_discipline", normalize_score(50 - gap * 500, 0, 100)))

        # Share dilution check
        if balance_df is not None:
            eq = extract_quarterly_series(balance_df, "total_equity", 8)
            if len(eq) >= 4:
                eq_growth = float(eq.pct_change().dropna().mean())
                # Growing equity through issuance (not earnings) = dilution
                ni = extract_quarterly_series(income_df, "net_income", 8) if income_df is not None else pd.Series()
                if len(ni) >= 4:
                    ni_growth = float(ni.pct_change().dropna().mean())
                    if eq_growth > ni_growth + 0.02:
                        components.append(("no_dilution", 20.0))
                    else:
                        components.append(("no_dilution", 80.0))

        if components:
            result["score"] = round(sum(c[1] for c in components) / len(components), 1)
            result["components"] = {c[0]: round(c[1], 1) for c in components}
            result["label"] = "aligned" if result["score"] > 65 else "misaligned" if result["score"] < 35 else "neutral"
            result["available"] = True

    except Exception as exc:
        logger.debug("Insider alignment failed: %s", exc)
    return result


# ---------------------------------------------------------------------------
# P2: Capital Cycle Position
# ---------------------------------------------------------------------------

def compute_capital_cycle(
    income_df: pd.DataFrame,
    cashflow_df: pd.DataFrame,
    linked_caches: dict | None = None,
) -> dict:
    """Detect position in the industry capital investment cycle.

    Rising capex/revenue across sector = late cycle (over-investment).
    """
    result = {"available": False, "cycle_position": "unknown", "risk_score": 50}
    try:
        capex = extract_quarterly_series(cashflow_df, "capex", 8)
        rev = extract_quarterly_series(income_df, "revenue", 8)

        if len(capex) < 4 or len(rev) < 4:
            return result

        # Company's own capex intensity trend
        common = capex.index.intersection(rev.index)
        if len(common) < 4:
            return result
        safe_rev = rev.loc[common].where(rev.loc[common].abs() > 1e-6)
        capex_intensity = (capex.loc[common].abs() / safe_rev).dropna()

        if len(capex_intensity) < 4:
            return result

        own_slope = compute_rolling_slope(capex_intensity, len(capex_intensity))

        # Sector aggregate (from linked caches)
        sector_slope = None
        if linked_caches:
            sector_intensities = []
            for eid, lc in linked_caches.items():
                if "capex" in lc.columns and "revenue" in lc.columns:
                    c = lc["capex"].dropna()
                    r = lc["revenue"].dropna()
                    if len(c) >= 4 and len(r) >= 4:
                        safe_r = r.where(r.abs() > 1e-6)
                        ratio = (c.abs() / safe_r).dropna()
                        if len(ratio) >= 2:
                            sector_intensities.append(float(ratio.iloc[-1]))
            if sector_intensities:
                # Compare own vs sector
                own_latest = float(capex_intensity.iloc[-1])
                sector_median = float(np.median(sector_intensities))
                result["own_intensity"] = round(own_latest, 4)
                result["sector_median_intensity"] = round(sector_median, 4)

        # Classify cycle position
        if own_slope is not None:
            result["capex_trend_slope"] = round(own_slope, 6)
            if own_slope > 0.005:
                result["cycle_position"] = "late_cycle"  # increasing investment
                result["risk_score"] = 70
            elif own_slope < -0.005:
                result["cycle_position"] = "early_cycle"  # decreasing = approaching trough
                result["risk_score"] = 30
            else:
                result["cycle_position"] = "mid_cycle"
                result["risk_score"] = 50

        result["available"] = True

    except Exception as exc:
        logger.debug("Capital cycle failed: %s", exc)
    return result


# ---------------------------------------------------------------------------
# P2: Earnings Torpedo Detection
# ---------------------------------------------------------------------------

def compute_earnings_torpedo(
    income_df: pd.DataFrame,
    cache: pd.DataFrame | None = None,
    peer_ranking_result: Any = None,
) -> dict:
    """Detect "torpedo" setup: sky-high expectations + decelerating fundamentals."""
    result = {"available": False, "torpedo_risk": 0, "flags": []}
    try:
        # Check 1: PE > 2x sector median
        pe = get_cache_latest(cache, "pe_ratio_calc") if cache is not None else None
        peer_pe_rank = None
        if peer_ranking_result is not None:
            vr = getattr(peer_ranking_result, "variable_ranks", None)
            if isinstance(vr, dict):
                peer_pe_rank = vr.get("pe_ratio_calc")

        if pe is not None and pe > 30:
            result["flags"].append(f"High PE ({pe:.0f}x)")
        if peer_pe_rank is not None and peer_pe_rank > 80:
            result["flags"].append(f"PE in top {100-peer_pe_rank:.0f}% of peers")

        # Check 2: Revenue deceleration
        rev_changes = extract_qoq_changes(income_df, "revenue", 8)
        if len(rev_changes) >= 4:
            accel = rev_changes.diff().dropna()
            if len(accel) >= 2 and float(accel.iloc[-1]) < -0.01:
                result["flags"].append("Revenue growth decelerating")

        # Check 3: Sequential beats narrowing
        eps = extract_quarterly_series(income_df, "eps", 8)
        if len(eps) >= 6:
            surprises = eps - eps.shift(4)
            surprises = surprises.dropna()
            if len(surprises) >= 4:
                recent = surprises.iloc[-3:]
                if all(s > 0 for s in recent) and float(recent.iloc[-1]) < float(recent.iloc[0]):
                    result["flags"].append("Beats narrowing (smaller surprises each quarter)")

        # Check 4: Buyback-inflated EPS
        eps_growth = extract_qoq_changes(income_df, "eps", 8)
        rev_growth = extract_qoq_changes(income_df, "revenue", 8)
        if len(eps_growth) >= 2 and len(rev_growth) >= 2:
            latest_eps_g = float(eps_growth.iloc[-1])
            latest_rev_g = float(rev_growth.iloc[-1])
            if latest_eps_g > 0.05 and latest_rev_g < 0.02:
                result["flags"].append("EPS growing via buybacks, not revenue")

        result["torpedo_risk"] = min(100, len(result["flags"]) * 25)
        result["available"] = True

    except Exception as exc:
        logger.debug("Earnings torpedo failed: %s", exc)
    return result


# ---------------------------------------------------------------------------
# P2: VRP Proxy
# ---------------------------------------------------------------------------

def compute_vrp_proxy(cache: pd.DataFrame | None) -> dict:
    """Volatility risk premium proxy: GARCH forecast vs realized."""
    result = {"available": False}
    if cache is None or "return_1d" not in cache.columns:
        return result
    try:
        returns = cache["return_1d"].dropna()
        if len(returns) < 63:
            return result

        realized = float(returns.iloc[-21:].std() * np.sqrt(252))

        # Simple GARCH(1,1) forecast proxy: EWMA variance
        lambda_ewma = 0.94
        var_series = returns ** 2
        ewma_var = var_series.ewm(alpha=1 - lambda_ewma).mean()
        forecast = float(np.sqrt(ewma_var.iloc[-1] * 252))

        vrp = forecast - realized

        result["realized_vol"] = round(realized, 4)
        result["forecast_vol"] = round(forecast, 4)
        result["vrp"] = round(vrp, 4)
        result["regime"] = "risk_premium" if vrp > 0.02 else "tail_event" if vrp < -0.05 else "neutral"
        result["available"] = True

    except Exception as exc:
        logger.debug("VRP proxy failed: %s", exc)
    return result


# ---------------------------------------------------------------------------
# P3: Implied Cost of Capital (Ohlson-Juettner 2005)
# ---------------------------------------------------------------------------

def compute_implied_cost_of_capital(
    income_df: pd.DataFrame,
    cache: pd.DataFrame | None = None,
) -> float | None:
    """Reverse-engineer the discount rate from current price + earnings."""
    try:
        eps = extract_latest_value(income_df, "eps")
        if eps is None:
            eps = extract_latest_value(income_df, "eps_diluted")
        close = get_cache_latest(cache, "close") if cache is not None else None

        if eps is None or close is None or close <= 0 or eps <= 0:
            return None

        # Growth estimate from recent revenue trend
        rev = extract_quarterly_series(income_df, "revenue", 8)
        g = 0.05  # default 5%
        if len(rev) >= 5:
            yoy = rev.pct_change(4).dropna()
            if len(yoy) > 0:
                g = max(0.01, min(0.20, float(yoy.iloc[-1])))

        gamma = 1 + 0.03  # long-term growth ~3%
        ep_ratio = eps / close

        A = 0.5 * ((gamma - 1) + ep_ratio)
        inner = A ** 2 + ep_ratio * (g - (gamma - 1))
        if inner < 0:
            return None

        icc = A + np.sqrt(inner)
        return round(float(icc), 4) if np.isfinite(icc) else None

    except Exception as exc:
        logger.debug("ICC failed: %s", exc)
        return None


# ---------------------------------------------------------------------------
# P3: Cross-Asset Macro Regime
# ---------------------------------------------------------------------------

def compute_cross_asset_regime(macro_data: dict | None) -> dict:
    """Multi-signal macro regime indicator combining GDP, inflation, rates, FX."""
    result = {"available": False, "regime": "unknown", "signals": {}, "risk_score": 50}
    if not macro_data:
        return result
    try:
        n_danger = 0
        n_signals = 0

        for indicator, label, threshold_fn in [
            ("gdp", "gdp_declining", lambda s: float(s.iloc[-1]) < float(s.iloc[-2]) if len(s) >= 2 else False),
            ("inflation", "inflation_rising", lambda s: float(s.iloc[-1]) > float(s.iloc[-2]) if len(s) >= 2 else False),
            ("interest_rate", "rates_elevated", lambda s: float(s.iloc[-1]) > 0.04 if len(s) >= 1 else False),
            ("unemployment", "unemployment_rising", lambda s: float(s.iloc[-1]) > float(s.iloc[-2]) if len(s) >= 2 else False),
            ("currency", "fx_depreciating", lambda s: float(s.iloc[-1]) < float(s.iloc[-2]) * 0.95 if len(s) >= 2 else False),
        ]:
            series = macro_data.get(indicator)
            if series is not None and not series.empty:
                try:
                    danger = threshold_fn(series)
                    result["signals"][label] = danger
                    n_signals += 1
                    if danger:
                        n_danger += 1
                except Exception:
                    pass

        if n_signals >= 3:
            danger_ratio = n_danger / n_signals
            if danger_ratio >= 0.6:
                result["regime"] = "crisis"
                result["risk_score"] = 85
            elif danger_ratio >= 0.4:
                result["regime"] = "stress"
                result["risk_score"] = 65
            elif danger_ratio >= 0.2:
                result["regime"] = "caution"
                result["risk_score"] = 45
            else:
                result["regime"] = "benign"
                result["risk_score"] = 20
            result["available"] = True

    except Exception as exc:
        logger.debug("Cross-asset regime failed: %s", exc)
    return result


# ---------------------------------------------------------------------------
# Master function: run all 15 advanced methods
# ---------------------------------------------------------------------------

def run_advanced_methods(
    income_df: pd.DataFrame,
    balance_df: pd.DataFrame,
    cashflow_df: pd.DataFrame,
    cache: pd.DataFrame | None = None,
    linked_caches: dict | None = None,
    macro_data: dict | None = None,
    peer_ranking_result: Any = None,
) -> AdvancedMethodsResult:
    """Run all 15 advanced HF methods and return combined result."""
    result = AdvancedMethodsResult()

    # P1: Piotroski F-Score
    result.piotroski_f_score, result.piotroski_n_available, result.piotroski_label = compute_piotroski_f_score(
        income_df, balance_df, cashflow_df,
    )

    # P1: Balance Sheet Velocity
    result.balance_sheet_velocity = compute_balance_sheet_velocity(
        income_df, balance_df, cashflow_df,
    )

    # P1: Earnings Persistence
    result.earnings_persistence = compute_earnings_persistence(income_df)

    # P1: Forensic Cash Flow
    result.forensic_cashflow = compute_forensic_cashflow(
        income_df, balance_df, cashflow_df,
    )

    # P1: OU Mean Reversion
    result.ou_mean_reversion = compute_ou_mean_reversion(cache)

    # P1: Accruals Percentile Rank (uses peer_ranking if available)
    if peer_ranking_result is not None:
        vr = getattr(peer_ranking_result, "variable_ranks", None)
        if isinstance(vr, dict):
            # accruals_signal is lower = better quality
            # If peer_ranking has it, use it; otherwise derive from accruals
            result.accruals_rank = vr.get("accruals_signal")

    # P1: Altman Z''
    result.altman_z_double_prime, result.altman_z_dp_zone = compute_altman_z_double_prime(
        income_df, balance_df,
    )

    # P2: GARCH Vol Term Structure
    result.garch_vol_term_structure = compute_garch_vol_term_structure(cache)

    # P2: Insider Alignment
    result.insider_alignment = compute_insider_alignment(cache, income_df, balance_df)

    # P2: Capital Cycle
    result.capital_cycle = compute_capital_cycle(income_df, cashflow_df, linked_caches)

    # P2: Earnings Torpedo
    result.earnings_torpedo = compute_earnings_torpedo(income_df, cache, peer_ranking_result)

    # P2: VRP Proxy
    result.vrp_proxy = compute_vrp_proxy(cache)

    # P3: Implied Cost of Capital
    result.implied_cost_of_capital = compute_implied_cost_of_capital(income_df, cache)

    # P3: Cross-Asset Regime
    result.cross_asset_regime = compute_cross_asset_regime(macro_data)

    # P3: Fama-French Alpha (stub -- needs external factor data)
    # Would need Ken French CSV download. Setting to None for now.
    result.fama_french_alpha = None

    result.available = True
    return result
