"""Hedge Fund Analysis orchestrator.

Entry point: ``run_hedge_fund_analysis()`` runs all 15 HF metric
modules in dependency order, assembles the thesis scorecard, and
computes the position signal.

Called from main.py Step 6-HF (after multi-frequency runner, before
profile builder).
"""

from __future__ import annotations

import logging
import time
from typing import Any

import numpy as np
import pandas as pd

from operator1.hedge_fund.types import (
    HedgeFundResult,
    ThesisScorecard,
    TierScore,
    PositionSignalResult,
    FCFQualityResult,
    AccrualsForensicResult,
    SmoothingResult,
    DividendBurnResult,
    ReturnSpreadResult,
    OperatingLeverageResult,
    OBSRiskResult,
    AssetQualityResult,
    LeverageStressResult,
    MomentumCompositeResult,
    GrowthQualityResult,
    SurpriseResult,
    DCFResult,
    ValuationQualityResult,
    PEGCompositeResult,
    _score_label,
    _risk_label,
)
from operator1.hedge_fund.helpers import (
    get_hf_weight,
    normalize_score,
    safe_divide,
    get_cache_latest,
    extract_quarterly_series,
    extract_latest_value,
    extract_qoq_changes,
    compute_rolling_slope,
)

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Tier 2-5 inline computations (compact, same pattern as Tier 1)
# ---------------------------------------------------------------------------

def _compute_dividend_burn(
    income_df: pd.DataFrame,
    cashflow_df: pd.DataFrame,
    balance_df: pd.DataFrame,
) -> DividendBurnResult:
    """HF-2.1: Dividend burn risk from raw quarterly filings."""
    result = DividendBurnResult()
    try:
        n = get_hf_weight("data_windows.quarterly_lookback", 8)
        divs = extract_quarterly_series(cashflow_df, "dividends_paid", n)
        ocf = extract_quarterly_series(cashflow_df, "operating_cash_flow", n)
        capex = extract_quarterly_series(cashflow_df, "capex", n)
        ie = extract_quarterly_series(income_df, "interest_expense", n)

        if len(divs) < 2 or len(ocf) < 2:
            return result

        common = divs.index.intersection(ocf.index)
        if len(common) < 2:
            return result

        # FCF = OCF - |capex|
        capex_common = capex.reindex(common).fillna(0)
        fcf = ocf.loc[common] - capex_common.abs()
        div_abs = divs.loc[common].abs()

        # Dividend/FCF coverage
        safe_fcf = fcf.where(fcf.abs() > 1e-6)
        div_coverage = (div_abs / safe_fcf).dropna()
        if len(div_coverage) > 0:
            result.dividend_fcf_coverage = float(div_coverage.mean())

        # B2 FIX: Check for dividends paid from negative FCF
        mean_fcf = float(fcf.mean())
        mean_div = float(div_abs.mean())

        # Debt service coverage
        ie_common = ie.reindex(common).fillna(0)
        debt_service = ie_common + div_abs
        ds_coverage = (debt_service / safe_fcf).dropna()
        if len(ds_coverage) > 0:
            result.debt_service_coverage = float(ds_coverage.mean())

        # Working capital drain
        ca = extract_quarterly_series(balance_df, "current_assets", n)
        cl = extract_quarterly_series(balance_df, "current_liabilities", n)
        if len(ca) >= 4 and len(cl) >= 4:
            wc_common = ca.index.intersection(cl.index)
            if len(wc_common) >= 4:
                wc = ca.loc[wc_common] - cl.loc[wc_common]
                wc_slope = compute_rolling_slope(wc, window=len(wc))
                if wc_slope is not None and wc_slope < 0:
                    result.working_capital_drain = True

        # Composite risk score
        w = get_hf_weight("cash_flow.dividend_burn", {})
        # B2 FIX: negative FCF + positive dividends = automatic high risk
        if mean_fcf < 0 and mean_div > 0:
            div_score = 95.0  # paying dividends from debt/asset liquidation
        elif result.dividend_fcf_coverage is not None and result.dividend_fcf_coverage > 1.5:
            div_score = min(100, result.dividend_fcf_coverage * 40)
        elif result.dividend_fcf_coverage is not None:
            div_score = min(100, max(0, result.dividend_fcf_coverage * 60))
        else:
            div_score = 50
        ds_score = min(100, max(0, (result.debt_service_coverage or 0) * 50)) if result.debt_service_coverage else 50
        wc_score = 70 if result.working_capital_drain else 30
        vol_score = 50  # placeholder

        result.risk_score = normalize_score(
            w.get("div_fcf_weight", 0.4) * div_score
            + w.get("debt_service_weight", 0.3) * ds_score
            + w.get("wc_drain_weight", 0.2) * wc_score
            + w.get("earnings_vol_weight", 0.1) * vol_score
        )
        result.label = _risk_label(result.risk_score)
        result.narrative = f"Div/FCF coverage {result.dividend_fcf_coverage:.2f}" if result.dividend_fcf_coverage else ""
        result.available = True
    except Exception as exc:
        logger.debug("Dividend burn failed: %s", exc)
        result.error = str(exc)
    return result


def _compute_return_spread(
    income_df: pd.DataFrame,
    balance_df: pd.DataFrame,
    cashflow_df: pd.DataFrame,
) -> ReturnSpreadResult:
    """HF-2.2: CROA vs ROIC spread."""
    result = ReturnSpreadResult()
    try:
        n = get_hf_weight("data_windows.quarterly_lookback", 8)
        ocf = extract_latest_value(cashflow_df, "operating_cash_flow")
        ta = extract_latest_value(balance_df, "total_assets")
        ebit = extract_latest_value(income_df, "ebit")
        equity = extract_latest_value(balance_df, "total_equity")
        debt = extract_latest_value(balance_df, "total_debt")
        cash = extract_latest_value(balance_df, "cash_and_equivalents")
        taxes = extract_latest_value(income_df, "taxes")

        if ocf is not None and ta is not None and abs(ta) > 1e-6:
            result.croa = ocf / ta

        tax_rate = safe_divide(taxes, ebit, default=0.25) if taxes and ebit else 0.25
        nopat = ebit * (1 - min(abs(tax_rate), 0.5)) if ebit else None
        ic = (equity or 0) + (debt or 0) - (cash or 0)
        if nopat is not None and abs(ic) > 1e-6:
            result.roic = nopat / ic

        if result.croa is not None and result.roic is not None:
            result.spread_bps = (result.croa - result.roic) * 10000
            result.quality_label = "healthy" if result.spread_bps > 50 else "concerning" if result.spread_bps < -100 else "neutral"
            result.narrative = f"CROA {result.croa:.1%} vs ROIC {result.roic:.1%} = {result.spread_bps:.0f}bps"
            result.available = True

        # 5-factor DuPont decomposition (Palepu, Healy & Peek 2019)
        ni = extract_latest_value(income_df, "net_income")
        ebt = extract_latest_value(income_df, "ebt")
        revenue = extract_latest_value(income_df, "revenue")
        if ebt is None and ni is not None and taxes is not None:
            ebt = ni + abs(taxes)  # approximate EBT
        _eps_d = 1e-10
        if all(v is not None for v in [ni, ebt, ebit, revenue, ta, equity]):
            result.dupont_tax_burden = safe_divide(ni, ebt, default=None) if abs(ebt or 0) > _eps_d else None
            result.dupont_interest_burden = safe_divide(ebt, ebit, default=None) if abs(ebit or 0) > _eps_d else None
            result.dupont_asset_turnover = safe_divide(revenue, ta, default=None) if abs(ta or 0) > _eps_d else None
            result.dupont_equity_multiplier = safe_divide(ta, equity, default=None) if abs(equity or 0) > _eps_d else None
            # Identify quality driver: which component changed most?
            components = {
                "margin": abs(ebit / max(abs(revenue), _eps_d)) if revenue else 0,
                "leverage": abs(ta / max(abs(equity), _eps_d)) if equity else 0,
                "turnover": abs(revenue / max(abs(ta), _eps_d)) if ta else 0,
            }
            result.dupont_quality_driver = max(components, key=components.get) if components else ""
    except Exception as exc:
        logger.debug("Return spread failed: %s", exc)
        result.error = str(exc)
    return result


def _compute_operating_leverage(income_df: pd.DataFrame) -> OperatingLeverageResult:
    """HF-2.3: Operating leverage (DOL/DFL/DTL)."""
    result = OperatingLeverageResult()
    try:
        n = get_hf_weight("data_windows.quarterly_lookback", 8)
        rev_changes = extract_qoq_changes(income_df, "revenue", n)
        ebit_changes = extract_qoq_changes(income_df, "ebit", n)
        eps_changes = extract_qoq_changes(income_df, "eps", n)
        if eps_changes.empty:
            eps_changes = extract_qoq_changes(income_df, "eps_diluted", n)

        if len(rev_changes) >= 2 and len(ebit_changes) >= 2:
            common = rev_changes.index.intersection(ebit_changes.index)
            if len(common) >= 2:
                safe_rev = rev_changes.loc[common].where(rev_changes.loc[common].abs() > 0.001)
                dol = (ebit_changes.loc[common] / safe_rev).dropna()
                if len(dol) > 0:
                    result.dol = float(dol.median())

        if len(ebit_changes) >= 2 and len(eps_changes) >= 2:
            common2 = ebit_changes.index.intersection(eps_changes.index)
            if len(common2) >= 2:
                safe_ebit = ebit_changes.loc[common2].where(ebit_changes.loc[common2].abs() > 0.001)
                dfl = (eps_changes.loc[common2] / safe_ebit).dropna()
                if len(dfl) > 0:
                    result.dfl = float(dfl.median())

        if result.dol is not None and result.dfl is not None:
            result.dtl = result.dol * result.dfl

        # Classify sensitivity
        if result.dol is not None:
            if abs(result.dol) > 5:
                result.earnings_sensitivity = "extreme"
            elif abs(result.dol) > 3:
                result.earnings_sensitivity = "high"
            elif abs(result.dol) > 1.5:
                result.earnings_sensitivity = "moderate"
            else:
                result.earnings_sensitivity = "low"

        if result.dol is not None:
            result.narrative = f"DOL={result.dol:.1f}, DFL={result.dfl or 0:.1f}, sensitivity={result.earnings_sensitivity}"
            result.available = True
    except Exception as exc:
        logger.debug("Operating leverage failed: %s", exc)
        result.error = str(exc)
    return result


def _compute_obs_risk(balance_df: pd.DataFrame, income_df: pd.DataFrame) -> OBSRiskResult:
    """HF-3.1: Off-balance-sheet risk."""
    result = OBSRiskResult()
    try:
        ta = extract_latest_value(balance_df, "total_assets")
        gw = extract_latest_value(balance_df, "goodwill")
        intang = extract_latest_value(balance_df, "intangible_assets")
        sga = extract_latest_value(income_df, "sga_expenses")
        rev = extract_latest_value(income_df, "revenue")

        if ta and abs(ta) > 1e-6:
            result.goodwill_to_assets = safe_divide(gw, ta)
            result.intangibles_to_assets = safe_divide(intang, ta)

        if sga and rev and abs(rev) > 1e-6:
            sga_pct = sga / rev
            # SGA > 40% of revenue is anomalous for most sectors
            result.sga_anomaly_score = min(100, max(0, (sga_pct - 0.15) / 0.25 * 100))

        w = get_hf_weight("balance_sheet.obs_risk", {})
        gw_score = min(100, (result.goodwill_to_assets or 0) / 0.3 * 100)
        int_score = min(100, (result.intangibles_to_assets or 0) / 0.3 * 100)
        sga_score = result.sga_anomaly_score or 50

        result.risk_score = normalize_score(
            w.get("goodwill_weight", 0.35) * gw_score
            + w.get("intangibles_weight", 0.25) * int_score
            + w.get("sga_anomaly_weight", 0.20) * sga_score
            + w.get("keyword_weight", 0.10) * 50  # no filing text
            + w.get("policy_change_weight", 0.10) * 50
        )
        result.label = _risk_label(result.risk_score)
        result.available = True
    except Exception as exc:
        logger.debug("OBS risk failed: %s", exc)
        result.error = str(exc)
    return result


def _compute_asset_quality(
    income_df: pd.DataFrame,
    balance_df: pd.DataFrame,
    cashflow_df: pd.DataFrame,
) -> AssetQualityResult:
    """HF-3.2: Asset quality deterioration."""
    result = AssetQualityResult()
    try:
        n = get_hf_weight("data_windows.quarterly_lookback", 8)
        rec = extract_quarterly_series(balance_df, "receivables", n)
        rev = extract_quarterly_series(income_df, "revenue", n)
        inv = extract_quarterly_series(balance_df, "inventory", n)
        cogs = extract_quarterly_series(income_df, "cost_of_revenue", n)

        # DSO = receivables / (revenue / 365)
        if len(rec) >= 2 and len(rev) >= 2:
            common = rec.index.intersection(rev.index)
            if len(common) >= 2:
                safe_rev = rev.loc[common].where(rev.loc[common].abs() > 1e-6)
                dso = rec.loc[common] / (safe_rev / 90)  # quarterly ~ 90 days
                dso = dso.dropna()
                if len(dso) >= 2:
                    result.dso = float(dso.iloc[-1])
                    prev_dso = float(dso.iloc[-2])
                    if abs(prev_dso) > 1e-6:
                        result.dso_change_pct = (result.dso - prev_dso) / abs(prev_dso) * 100

        # Inventory days = inventory / (COGS / 90)
        if len(inv) >= 2 and len(cogs) >= 2:
            common_i = inv.index.intersection(cogs.index)
            if len(common_i) >= 2:
                safe_cogs = cogs.loc[common_i].where(cogs.loc[common_i].abs() > 1e-6)
                inv_days = inv.loc[common_i] / (safe_cogs / 90)
                inv_days = inv_days.dropna()
                if len(inv_days) >= 2:
                    result.inventory_days = float(inv_days.iloc[-1])
                    prev_inv = float(inv_days.iloc[-2])
                    if abs(prev_inv) > 1e-6:
                        result.inventory_days_change_pct = (result.inventory_days - prev_inv) / abs(prev_inv) * 100

        # Composite
        w = get_hf_weight("balance_sheet.asset_quality", {})
        dso_score = min(100, max(0, (result.dso_change_pct or 0) * 3 + 50))
        inv_score = min(100, max(0, (result.inventory_days_change_pct or 0) * 3 + 50))
        result.deterioration_score = normalize_score(
            w.get("dso_change_weight", 0.30) * dso_score
            + w.get("inventory_days_weight", 0.25) * inv_score
            + w.get("capitalization_weight", 0.20) * 50
            + w.get("allowance_weight", 0.15) * 50
            + w.get("impairment_weight", 0.10) * 50
        )
        result.label = _risk_label(result.deterioration_score)
        result.available = True
    except Exception as exc:
        logger.debug("Asset quality failed: %s", exc)
        result.error = str(exc)
    return result


def _compute_leverage_stress(
    income_df: pd.DataFrame,
    balance_df: pd.DataFrame,
    mc_result: Any = None,
) -> LeverageStressResult:
    """HF-3.3: 3-scenario leverage stress test."""
    from operator1.hedge_fund.types import StressScenario
    result = LeverageStressResult()
    try:
        cfg = get_hf_weight("balance_sheet.leverage_stress", {})
        debt = extract_latest_value(balance_df, "total_debt")
        ebitda = extract_latest_value(income_df, "ebitda")
        ebit = extract_latest_value(income_df, "ebit")
        ie = extract_latest_value(income_df, "interest_expense")
        rev = extract_latest_value(income_df, "revenue")
        margin = safe_divide(ebitda, rev) if ebitda and rev else None

        if not all([debt, rev]):
            return result
        if ebitda is None:
            return result

        covenant = cfg.get("covenant_threshold", 5.5)

        # B3 FIX: negative EBITDA = automatic distress
        if ebitda <= 0:
            base = StressScenario(
                name="base_case", debt_to_ebitda=None,
                interest_coverage=safe_divide(ebit, ie) if ebit and ie else None,
                covenant_breach=True,
            )
            cash_val = extract_latest_value(balance_df, "cash_and_equivalents") or 0
            if abs(ebitda) > 1e-6:
                base.cash_runway_months = safe_divide(cash_val, abs(ebitda) / 12)
            result.base_case = base
            result.revenue_miss = StressScenario(name="revenue_miss", covenant_breach=True)
            result.systemic_crisis = StressScenario(name="systemic_crisis", covenant_breach=True)
            result.refinancing_risk = True
            result.narrative = f"Negative EBITDA ({ebitda:.0f}): automatic distress"
            result.available = True
            return result

        # Base case (positive EBITDA path)
        base = StressScenario(name="base_case")
        base.debt_to_ebitda = safe_divide(debt, ebitda)
        base.interest_coverage = safe_divide(ebit, ie) if ebit and ie else None
        base.covenant_breach = (base.debt_to_ebitda or 0) > covenant
        result.base_case = base

        # Revenue miss (-15%)
        rev_miss_pct = cfg.get("revenue_miss_pct", 0.15)
        margin_miss_bps = cfg.get("margin_miss_bps", 200) / 10000
        stressed_rev = rev * (1 - rev_miss_pct)
        stressed_margin = (margin or 0.15) - margin_miss_bps
        stressed_ebitda = stressed_rev * max(stressed_margin, 0.01)
        miss = StressScenario(
            name="revenue_miss",
            revenue_shock_pct=rev_miss_pct * 100,
            margin_shock_bps=margin_miss_bps * 10000,
        )
        miss.debt_to_ebitda = safe_divide(debt, stressed_ebitda)
        miss.interest_coverage = safe_divide(stressed_ebitda * 0.85, ie) if ie else None
        miss.covenant_breach = (miss.debt_to_ebitda or 0) > covenant
        result.revenue_miss = miss

        # Systemic crisis (-25% rev, -400bps margin, +200bps rate)
        sys_rev_pct = cfg.get("systemic_revenue_pct", 0.25)
        sys_margin_bps = cfg.get("systemic_margin_bps", 400) / 10000
        sys_rev = rev * (1 - sys_rev_pct)
        sys_margin = (margin or 0.15) - sys_margin_bps
        sys_ebitda = sys_rev * max(sys_margin, 0.01)
        systemic = StressScenario(
            name="systemic_crisis",
            revenue_shock_pct=sys_rev_pct * 100,
            margin_shock_bps=sys_margin_bps * 10000,
            rate_shock_bps=200,
        )
        systemic.debt_to_ebitda = safe_divide(debt, sys_ebitda)
        # Rate shock on interest
        stressed_ie = (ie or 0) * 1.3  # +200bps on existing rate
        systemic.interest_coverage = safe_divide(sys_ebitda * 0.85, stressed_ie) if stressed_ie else None
        systemic.covenant_breach = (systemic.debt_to_ebitda or 0) > covenant
        result.systemic_crisis = systemic

        result.refinancing_risk = systemic.covenant_breach
        result.narrative = (
            f"Base D/EBITDA={base.debt_to_ebitda:.1f}x, "
            f"Stress={miss.debt_to_ebitda:.1f}x, "
            f"Crisis={systemic.debt_to_ebitda:.1f}x"
            if base.debt_to_ebitda else ""
        )
        result.available = True
    except Exception as exc:
        logger.debug("Leverage stress failed: %s", exc)
        result.error = str(exc)
    return result


def _compute_momentum(
    income_df: pd.DataFrame,
    cashflow_df: pd.DataFrame,
    cache: pd.DataFrame | None = None,
) -> MomentumCompositeResult:
    """HF-4.1: Multi-dimensional momentum composite."""
    result = MomentumCompositeResult()
    try:
        n = get_hf_weight("data_windows.quarterly_lookback", 8)
        w = get_hf_weight("inflection.momentum", {})

        # Revenue acceleration (2nd derivative)
        rev_changes = extract_qoq_changes(income_df, "revenue", n)
        rev_accel = None
        if len(rev_changes) >= 4:
            accel = rev_changes.diff().dropna()
            if len(accel) >= 2:
                rev_accel = float(accel.iloc[-1])
                result.revenue_acceleration = rev_accel

        # Margin trend slope
        margin_series = extract_quarterly_series(income_df, "net_income", n)
        rev_series = extract_quarterly_series(income_df, "revenue", n)
        margin_slope = None
        if len(margin_series) >= 4 and len(rev_series) >= 4:
            common = margin_series.index.intersection(rev_series.index)
            if len(common) >= 4:
                safe_rev = rev_series.loc[common].where(rev_series.loc[common].abs() > 1e-6)
                margins = (margin_series.loc[common] / safe_rev).dropna()
                margin_slope = compute_rolling_slope(margins, window=len(margins))
                result.margin_trend_slope = margin_slope

        # FCF conversion trend
        ocf = extract_quarterly_series(cashflow_df, "operating_cash_flow", n)
        fcf_conv_slope = None
        if len(ocf) >= 4 and len(rev_series) >= 4:
            common_f = ocf.index.intersection(rev_series.index)
            if len(common_f) >= 4:
                safe_rev2 = rev_series.loc[common_f].where(rev_series.loc[common_f].abs() > 1e-6)
                conv = (ocf.loc[common_f] / safe_rev2).dropna()
                fcf_conv_slope = compute_rolling_slope(conv, window=len(conv))
                result.fcf_conversion_trend = fcf_conv_slope

        # ROIC trajectory (simplified)
        roic_slope = None  # would need invested capital computation

        # Price momentum from cache (Jegadeesh & Titman 1993)
        price_mom_score = 50.0  # neutral default
        if cache is not None and "close" in cache.columns:
            closes = cache["close"].dropna()
            _pm_lookback = int(w.get("price_momentum_lookback_days", 63))
            _pm_scale = float(w.get("price_normalization_scale", 200))
            if len(closes) >= _pm_lookback:
                ret_63d = float(closes.iloc[-1] / closes.iloc[-_pm_lookback] - 1)
                result.price_momentum_63d = ret_63d
                if len(closes) >= 252:
                    result.price_momentum_252d = float(closes.iloc[-1] / closes.iloc[-252] - 1)
                price_mom_score = normalize_score(50 + ret_63d * _pm_scale, 0, 100)

        # Score components (normalize to 0-100)
        def _norm(val: float | None, scale: float = 100.0) -> float:
            if val is None:
                return 50.0
            return normalize_score(50 + val * scale, 0, 100)

        score = (
            w.get("revenue_accel_weight", 0.30) * _norm(rev_accel, 500)
            + w.get("margin_trend_weight", 0.25) * _norm(margin_slope, 5000)
            + w.get("fcf_conversion_weight", 0.15) * _norm(fcf_conv_slope, 5000)
            + w.get("roic_trajectory_weight", 0.05) * 50
            + w.get("price_momentum_weight", 0.25) * price_mom_score
        )
        result.score = normalize_score(score)
        result.label = _score_label(result.score)
        result.inflection_detected = (
            rev_accel is not None and rev_accel < -0.01
            and (margin_slope is not None and margin_slope < 0)
        )

        # Price-fundamental divergence detection
        _fund_score = (
            w.get("revenue_accel_weight", 0.30) * _norm(rev_accel, 500)
            + w.get("margin_trend_weight", 0.25) * _norm(margin_slope, 5000)
            + w.get("fcf_conversion_weight", 0.15) * _norm(fcf_conv_slope, 5000)
            + w.get("roic_trajectory_weight", 0.05) * 50
        ) / max(1 - w.get("price_momentum_weight", 0.25), 0.01)
        if price_mom_score > 65 and _fund_score < 40:
            result.price_fundamental_divergence = True
            result.divergence_direction = "price_leading"
        elif price_mom_score < 35 and _fund_score > 60:
            result.price_fundamental_divergence = True
            result.divergence_direction = "fundamentals_leading"
        result.available = True
    except Exception as exc:
        logger.debug("Momentum composite failed: %s", exc)
        result.error = str(exc)
    return result


def _compute_growth_quality(income_df: pd.DataFrame, balance_df: pd.DataFrame) -> GrowthQualityResult:
    """HF-4.2: Growth quality decomposition."""
    result = GrowthQualityResult()
    try:
        n = get_hf_weight("data_windows.quarterly_lookback", 8)
        rev = extract_quarterly_series(income_df, "revenue", n)
        gw = extract_quarterly_series(balance_df, "goodwill", n)

        if len(rev) >= 4:
            yoy = rev.pct_change(4).dropna()
            if len(yoy) > 0:
                total_growth = float(yoy.iloc[-1])

                # Estimate inorganic from goodwill jumps
                inorganic_pct = 0.0
                if len(gw) >= 4:
                    gw_change = gw.diff().dropna()
                    if len(gw_change) > 0:
                        max_gw_jump = float(gw_change.max())
                        mean_rev = float(rev.mean())
                        if mean_rev > 1e-6 and max_gw_jump > 0:
                            inorganic_pct = min(1.0, max_gw_jump / mean_rev)
                            result.inorganic_growth_pct = inorganic_pct * total_growth if total_growth > 0 else 0

                result.organic_growth_pct = total_growth * (1 - inorganic_pct)
                result.organic_fraction = 1 - inorganic_pct

                # Score
                w = get_hf_weight("inflection.growth_quality", {})
                organic_score = normalize_score(result.organic_fraction * 100, 0, 100)
                result.score = normalize_score(
                    w.get("organic_fraction_weight", 0.50) * organic_score
                    + w.get("margin_adjusted_weight", 0.25) * 50
                    + w.get("incremental_roic_weight", 0.15) * 50
                    + w.get("concentration_weight", 0.10) * 50
                )
                result.label = _score_label(result.score)
                result.available = True
    except Exception as exc:
        logger.debug("Growth quality failed: %s", exc)
        result.error = str(exc)
    return result


def _compute_earnings_surprise(
    income_df: pd.DataFrame,
    cache: pd.DataFrame | None = None,
    filing_calendar_result: Any = None,
) -> SurpriseResult:
    """HF-4.3: Earnings surprise probability."""
    result = SurpriseResult()
    try:
        n = get_hf_weight("data_windows.quarterly_lookback", 8)
        eps = extract_quarterly_series(income_df, "eps", n)
        if eps.empty:
            eps = extract_quarterly_series(income_df, "eps_diluted", n)

        if len(eps) >= 5:
            # Historical surprise = EPS - EPS(4Q ago)
            surprises = eps - eps.shift(4)
            surprises = surprises.dropna()
            if len(surprises) >= 3:
                mean_surprise = float(surprises.mean())
                std_surprise = float(surprises.std())
                if std_surprise > 1e-8:
                    # P(miss) = P(actual < expected - 1.5*sigma)
                    from scipy import stats
                    try:
                        z_miss = -1.5  # 1.5 sigma below
                        z_beat = 1.5   # 1.5 sigma above
                        result.p_miss = float(stats.norm.cdf(z_miss))
                        result.p_beat = float(1 - stats.norm.cdf(z_beat))
                        result.p_inline = 1.0 - result.p_miss - result.p_beat

                        # Adjust by recent trend
                        if len(surprises) >= 4:
                            recent_mean = float(surprises.iloc[-3:].mean())
                            if recent_mean > 0:
                                result.p_beat += 0.1
                                result.p_miss -= 0.1
                            elif recent_mean < 0:
                                result.p_miss += 0.1
                                result.p_beat -= 0.1
                            result.p_beat = max(0, min(1, result.p_beat))
                            result.p_miss = max(0, min(1, result.p_miss))
                            result.p_inline = max(0, 1.0 - result.p_beat - result.p_miss)
                    except ImportError:
                        result.p_beat = 0.33
                        result.p_miss = 0.33
                        result.p_inline = 0.34

                    result.expected_direction = (
                        "beat" if result.p_beat > 0.45
                        else "miss" if result.p_miss > 0.45
                        else "neutral"
                    )

        # Days to next filing
        if filing_calendar_result is not None:
            nf = getattr(filing_calendar_result, "next_expected_filing", None)
            if isinstance(nf, dict) and nf.get("available"):
                days = nf.get("days_until")
                if days is not None:
                    result.days_to_next_filing = int(days)

        result.available = len(eps) >= 5
    except Exception as exc:
        logger.debug("Earnings surprise failed: %s", exc)
        result.error = str(exc)
    return result


def _compute_dcf(
    cashflow_df: pd.DataFrame,
    balance_df: pd.DataFrame,
    cache: pd.DataFrame | None = None,
    target_profile: dict | None = None,
    mc_result: Any = None,
    macro_data: dict | None = None,
    income_df: pd.DataFrame | None = None,
) -> DCFResult:
    """HF-5.1: DCF Monte Carlo valuation (calibrated -- Gap 5 fixes).

    5 calibration fixes from v3 accuracy improvement plan:
    1. Growth prior from company's own 3-year revenue CAGR (not regime dists)
    2. WACC capped at sector median + 2% (prevents extreme discount rates)
    3. 3-stage model for high-growth companies (revenue CAGR > 10%)
    4. Reverse DCF: solve for implied growth rate from current price
    5. Sanity gate: flag as unreliable when DCF/price ratio > 5x or < 0.2x
    """
    result = DCFResult()
    try:
        cfg = get_hf_weight("valuation.dcf", {})
        fcf_latest = extract_latest_value(cashflow_df, "free_cash_flow")
        if fcf_latest is None:
            ocf = extract_latest_value(cashflow_df, "operating_cash_flow")
            capex = extract_latest_value(cashflow_df, "capex")
            if ocf is not None and capex is not None:
                fcf_latest = ocf - abs(capex)

        debt = extract_latest_value(balance_df, "total_debt") or 0
        cash_val = extract_latest_value(balance_df, "cash_and_equivalents") or 0
        shares = None
        if cache is not None and "shares_outstanding" in cache.columns:
            shares = get_cache_latest(cache, "shares_outstanding")
        if shares is None and target_profile:
            shares = target_profile.get("shares_outstanding")
        close = get_cache_latest(cache, "close") if cache is not None else None

        if fcf_latest is None or fcf_latest <= 0 or shares is None or shares <= 0:
            return result

        n_sims = cfg.get("n_simulations", 10000)
        years = cfg.get("explicit_period_years", 5)
        tg_mean = cfg.get("terminal_growth_mean", 0.025)
        tg_std = cfg.get("terminal_growth_std", 0.005)
        wacc_mean = cfg.get("wacc_mean", 0.09)
        wacc_std = cfg.get("wacc_std", 0.01)

        # --- Fix 1: Growth prior from company's own revenue CAGR ---
        # The old approach used MC regime distribution means, which can be
        # near-zero in high-vol regimes (causing $33 intrinsic for $250 stock).
        # Now use the company's actual 3-year revenue CAGR as the base prior.
        growth_mean = 0.05
        growth_std = 0.03
        _used_cagr = False
        if income_df is not None and not income_df.empty:
            try:
                rev_series = extract_quarterly_series(income_df, "revenue", 12)
                if len(rev_series) >= 8:
                    # 3-year CAGR: (latest / 12Q ago)^(1/3) - 1
                    _earliest = rev_series.iloc[0]
                    _latest = rev_series.iloc[-1]
                    if _earliest > 0 and _latest > 0:
                        cagr_3y = (_latest / _earliest) ** (1.0 / 3.0) - 1.0
                        growth_mean = max(0.02, min(cagr_3y, 0.30))
                        growth_std = max(0.01, min(abs(growth_mean) * 0.3, 0.08))
                        _used_cagr = True
                        logger.debug("DCF growth from 3yr CAGR: %.1f%%", growth_mean * 100)
            except Exception:
                pass

        # Fallback to MC regime distributions only if CAGR unavailable
        if not _used_cagr and mc_result is not None and hasattr(mc_result, "regime_distributions"):
            dists = mc_result.regime_distributions
            if dists:
                means = [d.get("mean", 0) for d in dists.values() if isinstance(d, dict)]
                if means:
                    growth_mean = max(0.02, min(0.20, np.mean(means) * 252))
                    growth_std = max(0.01, min(0.10, np.std(means) * 252)) if len(means) > 1 else 0.03

        # --- Fix 2: Cap WACC at sector median + 2% ---
        wacc_cap = cfg.get("wacc_cap", 0.14)  # default cap: 14%
        wacc_mean = min(wacc_mean, wacc_cap)

        # --- Fix 3: 3-stage model for high-growth companies ---
        use_3stage = growth_mean > 0.10
        transition_years = 5  # years 6-10: fade to terminal growth

        rng = np.random.default_rng(42)
        intrinsic_values = []
        for _ in range(n_sims):
            g = rng.normal(growth_mean, growth_std)
            g = max(-0.05, min(0.35, g))
            wacc = rng.normal(wacc_mean, wacc_std)
            wacc = max(0.04, min(wacc_cap, wacc))
            tg = rng.normal(tg_mean, tg_std)
            tg = max(0.01, min(wacc - 0.01, tg))

            if use_3stage:
                # Stage 1: High growth (years 1-5)
                pv_fcf = sum(
                    fcf_latest * (1 + g) ** t / (1 + wacc) ** t
                    for t in range(1, years + 1)
                )
                # Stage 2: Transition (years 6-10, linear fade to terminal)
                pv_transition = 0.0
                for t_offset in range(1, transition_years + 1):
                    t = years + t_offset
                    # Linear fade: growth decreases from g to tg
                    fade_frac = t_offset / transition_years
                    g_fade = g * (1 - fade_frac) + tg * fade_frac
                    pv_transition += fcf_latest * (1 + g) ** years * (1 + g_fade) ** t_offset / (1 + wacc) ** t

                # Stage 3: Terminal (year 11+)
                fcf_end_transition = fcf_latest * (1 + g) ** years * (1 + (g + tg) / 2) ** transition_years
                terminal = fcf_end_transition * (1 + tg) / max(wacc - tg, 0.01)
                pv_terminal = terminal / (1 + wacc) ** (years + transition_years)

                equity_value = pv_fcf + pv_transition + pv_terminal - debt + cash_val
            else:
                # Standard single-stage DCF
                pv_fcf = sum(fcf_latest * (1 + g) ** t / (1 + wacc) ** t for t in range(1, years + 1))
                terminal = fcf_latest * (1 + g) ** years * (1 + tg) / max(wacc - tg, 0.01)
                pv_terminal = terminal / (1 + wacc) ** years
                equity_value = pv_fcf + pv_terminal - debt + cash_val

            per_share = equity_value / shares
            if np.isfinite(per_share) and per_share > 0:
                intrinsic_values.append(per_share)

        if len(intrinsic_values) >= 100:
            arr = np.array(intrinsic_values)
            result.intrinsic_p10 = float(np.percentile(arr, 10))
            result.intrinsic_p25 = float(np.percentile(arr, 25))
            result.intrinsic_p50 = float(np.percentile(arr, 50))
            result.intrinsic_p75 = float(np.percentile(arr, 75))
            result.intrinsic_p90 = float(np.percentile(arr, 90))
            result.current_price = close
            result.n_simulations = len(intrinsic_values)

            if close and close > 0:
                result.upside_pct = (result.intrinsic_p50 - close) / close * 100
                downside = abs(result.intrinsic_p25 - close) if result.intrinsic_p25 else 0
                upside = abs(result.intrinsic_p75 - close) if result.intrinsic_p75 else 0
                result.risk_reward_ratio = safe_divide(upside, downside)

            # --- Fix 4: Reverse DCF (implied growth rate) ---
            if close and close > 0 and shares > 0:
                try:
                    target_ev = close * shares + debt - cash_val
                    if target_ev > 0:
                        # Binary search for implied growth rate
                        lo, hi = -0.05, 0.50
                        for _ in range(50):
                            mid = (lo + hi) / 2
                            pv = sum(fcf_latest * (1 + mid) ** t / (1 + wacc_mean) ** t for t in range(1, years + 1))
                            tv = fcf_latest * (1 + mid) ** years * (1 + tg_mean) / max(wacc_mean - tg_mean, 0.01)
                            pv_tv = tv / (1 + wacc_mean) ** years
                            if pv + pv_tv < target_ev:
                                lo = mid
                            else:
                                hi = mid
                        result.implied_growth_rate = round((lo + hi) / 2, 4)
                except Exception:
                    pass

            # --- Fix 5: Sanity gate ---
            if close and close > 0:
                ratio = result.intrinsic_p50 / close
                if ratio > 5.0 or ratio < 0.2:
                    result.reliable = False
                    result.warning = (
                        f"DCF/price ratio {ratio:.1f}x is extreme -- "
                        f"intrinsic ${result.intrinsic_p50:.2f} vs market ${close:.2f}. "
                        f"Growth assumption ({growth_mean*100:.1f}%) or WACC ({wacc_mean*100:.1f}%) "
                        f"may be miscalibrated."
                    )
                    logger.warning("DCF sanity gate: ratio=%.1fx, flagged as unreliable", ratio)
                else:
                    result.reliable = True

            _model_type = "3-stage" if use_3stage else "single-stage"
            _growth_src = "revenue CAGR" if _used_cagr else "MC regime"
            result.narrative = (
                f"Intrinsic value ${result.intrinsic_p50:.2f} "
                f"(range ${result.intrinsic_p25:.2f}-${result.intrinsic_p75:.2f})"
                + (f" vs current ${close:.2f}" if close else "")
                + f" [{_model_type}, growth={growth_mean*100:.1f}% from {_growth_src}]"
            )
            result.available = True
    except Exception as exc:
        logger.debug("DCF valuation failed: %s", exc)
        result.error = str(exc)
    return result


def _compute_valuation_quality(
    hf_results: dict[str, Any],
    cache: pd.DataFrame | None = None,
    peer_ranking_result: Any = None,
) -> ValuationQualityResult:
    """HF-5.2: Valuation-quality matrix positioning."""
    result = ValuationQualityResult()
    try:
        w = get_hf_weight("valuation.valuation_quality", {})

        # Quality composite from Tiers 1-3
        fcf_q = hf_results.get("fcf_quality")
        accruals = hf_results.get("accruals_forensic")
        asset_q = hf_results.get("asset_quality")
        lev_stress = hf_results.get("leverage_stress")
        ret_spread = hf_results.get("return_spread")
        div_burn = hf_results.get("dividend_burn")
        obs = hf_results.get("obs_risk")

        # Invert risk scores (lower risk = higher quality)
        quality = (
            w.get("fcf_quality_weight", 0.25) * (getattr(fcf_q, "score", 50) if fcf_q else 50)
            + w.get("accruals_inverse_weight", 0.20) * (100 - getattr(accruals, "red_flag_score", 50) if accruals else 50)
            + w.get("asset_quality_inverse_weight", 0.15) * (100 - getattr(asset_q, "deterioration_score", 50) if asset_q else 50)
            + w.get("leverage_stress_inverse_weight", 0.15) * (50 if not lev_stress or not getattr(lev_stress, "base_case", None)
                                                                else max(0, 100 - (getattr(lev_stress.base_case, "debt_to_ebitda", 3) or 3) * 15))
            + w.get("return_spread_weight", 0.10) * (normalize_score(50 + (getattr(ret_spread, "spread_bps", 0) or 0) / 50, 0, 100) if ret_spread else 50)
            + w.get("dividend_burn_inverse_weight", 0.10) * (100 - getattr(div_burn, "risk_score", 50) if div_burn else 50)
            + w.get("obs_risk_inverse_weight", 0.05) * (100 - getattr(obs, "risk_score", 50) if obs else 50)
        )
        result.quality_score = normalize_score(quality)

        # Valuation percentile from peer ranking
        if peer_ranking_result is not None:
            var_ranks = getattr(peer_ranking_result, "variable_ranks", None)
            if isinstance(var_ranks, dict):
                pe_rank = var_ranks.get("pe_ratio_calc")
                if pe_rank is not None:
                    result.valuation_percentile = float(pe_rank)

        # Quadrant classification
        cheap = (result.valuation_percentile or 50) < 40
        quality_high = result.quality_score > 60
        if cheap and quality_high:
            result.quadrant = "undervalued"
        elif cheap and not quality_high:
            result.quadrant = "value_trap"
        elif not cheap and quality_high:
            result.quadrant = "fair"
        else:
            result.quadrant = "overvalued"

        result.available = True
    except Exception as exc:
        logger.debug("Valuation-quality matrix failed: %s", exc)
        result.error = str(exc)
    return result


def _compute_peg(
    income_df: pd.DataFrame,
    cache: pd.DataFrame | None = None,
    vq_result: ValuationQualityResult | None = None,
) -> PEGCompositeResult:
    """HF-5.3: Quality-adjusted PEG + FCF yield spread."""
    result = PEGCompositeResult()
    try:
        pe = get_cache_latest(cache, "pe_ratio_calc") if cache is not None else None
        fcf_yield = get_cache_latest(cache, "fcf_yield") if cache is not None else None
        ie = extract_latest_value(income_df, "interest_expense")
        debt = None
        # Try to get debt from cache
        if cache is not None:
            debt = get_cache_latest(cache, "total_debt")

        # Growth rate from revenue Y/Y
        rev = extract_quarterly_series(income_df, "revenue", 8)
        growth = None
        if len(rev) >= 5:
            yoy = rev.pct_change(4).dropna()
            if len(yoy) > 0:
                growth = float(yoy.iloc[-1])

        if pe is not None and growth is not None and abs(growth) > 0.001:
            result.peg_raw = pe / (growth * 100)  # PEG = PE / growth%

            # Quality adjustment
            q_mult = 1.0
            cfg = get_hf_weight("valuation.peg", {})
            if vq_result and vq_result.quality_score:
                q_mult = max(
                    cfg.get("quality_multiplier_floor", 0.5),
                    min(cfg.get("quality_multiplier_ceiling", 1.5),
                        vq_result.quality_score / 50)
                )
            result.peg_adjusted = result.peg_raw / q_mult if q_mult > 0 else result.peg_raw

        if fcf_yield is not None:
            result.fcf_yield_pct = fcf_yield * 100
        if ie is not None and debt is not None and abs(debt) > 1e-6:
            result.debt_cost_pct = abs(ie / debt) * 100
        if result.fcf_yield_pct is not None and result.debt_cost_pct is not None:
            result.fcf_spread_bps = (result.fcf_yield_pct - result.debt_cost_pct) * 100
            result.cheap_flag = result.fcf_spread_bps > 200

        result.available = pe is not None or fcf_yield is not None
    except Exception as exc:
        logger.debug("PEG composite failed: %s", exc)
        result.error = str(exc)
    return result


# ---------------------------------------------------------------------------
# Scorecard synthesis
# ---------------------------------------------------------------------------

def _build_scorecard(hf: HedgeFundResult) -> ThesisScorecard:
    """Synthesize all 15 metrics into a 5-tier thesis scorecard."""
    sc = ThesisScorecard()
    w = get_hf_weight("thesis_scorecard.tier_weights", {})
    grades = get_hf_weight("thesis_scorecard.grade_thresholds", {})

    # Tier 1: Earnings Quality (higher = better)
    t1_score = (
        hf.fcf_quality.score * 0.40
        + (100 - hf.accruals_forensic.red_flag_score) * 0.35
        + (100 - hf.smoothing.smoothing_index) * 0.25
    )
    sc.earnings_quality = TierScore(
        tier_name="earnings_quality", score=normalize_score(t1_score),
        label=_score_label(t1_score),
        components={"fcf_quality": hf.fcf_quality.score,
                     "accruals_risk": hf.accruals_forensic.red_flag_score,
                     "smoothing_risk": hf.smoothing.smoothing_index},
    )

    # Tier 2: Cash Flow (higher = better, invert risk scores)
    t2_score = (
        (100 - hf.dividend_burn.risk_score) * 0.35
        + normalize_score(50 + (hf.return_spread.spread_bps or 0) / 50, 0, 100) * 0.35
        + (50 if hf.operating_leverage.earnings_sensitivity in ("low", "moderate") else 25) * 0.30
    )
    sc.cash_flow = TierScore(
        tier_name="cash_flow", score=normalize_score(t2_score),
        label=_score_label(t2_score),
    )

    # Tier 3: Balance Sheet (higher = better, invert risk scores)
    base_de = getattr(hf.leverage_stress.base_case, "debt_to_ebitda", None)
    lev_score = max(0, 100 - (base_de or 3) * 15) if base_de else 50
    t3_score = (
        (100 - hf.obs_risk.risk_score) * 0.30
        + (100 - hf.asset_quality.deterioration_score) * 0.35
        + lev_score * 0.35
    )
    sc.balance_sheet = TierScore(
        tier_name="balance_sheet", score=normalize_score(t3_score),
        label=_score_label(t3_score),
    )

    # Tier 4: Inflection (higher = better)
    t4_score = (
        hf.momentum.score * 0.40
        + hf.growth_quality.score * 0.35
        + (70 if hf.earnings_surprise.expected_direction == "beat" else 30 if hf.earnings_surprise.expected_direction == "miss" else 50) * 0.25
    )
    sc.inflection = TierScore(
        tier_name="inflection", score=normalize_score(t4_score),
        label=_score_label(t4_score),
    )

    # Tier 5: Valuation (higher = cheaper/better)
    val_score = 50.0
    if hf.dcf.available and hf.dcf.upside_pct is not None:
        val_score = normalize_score(50 + hf.dcf.upside_pct, 0, 100)
    elif hf.valuation_quality.available:
        val_score = 70 if hf.valuation_quality.quadrant == "undervalued" else 30 if hf.valuation_quality.quadrant == "overvalued" else 50
    sc.valuation = TierScore(
        tier_name="valuation", score=normalize_score(val_score),
        label=_score_label(val_score),
    )

    # Overall grade
    overall = (
        w.get("earnings_quality", 0.25) * sc.earnings_quality.score
        + w.get("cash_flow", 0.20) * sc.cash_flow.score
        + w.get("balance_sheet", 0.20) * sc.balance_sheet.score
        + w.get("inflection", 0.20) * sc.inflection.score
        + w.get("valuation", 0.15) * sc.valuation.score
    )

    # Grade letter
    if overall >= grades.get("a_plus", 85): sc.investment_grade = "A+"
    elif overall >= grades.get("a", 75): sc.investment_grade = "A"
    elif overall >= grades.get("b_plus", 65): sc.investment_grade = "B+"
    elif overall >= grades.get("b", 55): sc.investment_grade = "B"
    elif overall >= grades.get("c_plus", 45): sc.investment_grade = "C+"
    elif overall >= grades.get("c", 35): sc.investment_grade = "C"
    elif overall >= grades.get("d", 25): sc.investment_grade = "D"
    else: sc.investment_grade = "F"

    sc.conviction = min(10, max(0, int(overall / 10)))

    # B4 FIX: Distress override -- cap grade when company is clearly distressed
    _in_distress = (
        (hf.leverage_stress.available and hf.leverage_stress.base_case.covenant_breach)
        or hf.fcf_quality.score < 30
        or hf.dividend_burn.risk_score > 80
    )
    if _in_distress:
        if sc.investment_grade in ("A+", "A", "B+", "B"):
            sc.investment_grade = "D"
            sc.conviction = min(sc.conviction, 2)

    sc.available = True
    return sc


def _compute_position_signal(
    hf: HedgeFundResult,
    cache: pd.DataFrame | None = None,
    signal_ic_result: Any = None,
    survival_controller: Any = None,
    forecast_result: Any = None,
    filing_calendar_result: Any = None,
    mc_result: Any = None,
) -> PositionSignalResult:
    """Compute actionable position signal from thesis scorecard."""
    result = PositionSignalResult()
    try:
        if not hf.scorecard.available:
            return result

        # Alpha base from forecast OR fallback from scorecard
        alpha = 0.0
        if forecast_result is not None and hasattr(forecast_result, "forecasts"):
            r5d = forecast_result.forecasts.get("return_5d", {})
            if isinstance(r5d, dict):
                val = r5d.get("5d")
                if val is not None:
                    alpha = float(val) if isinstance(val, (int, float)) else 0.0

        # B5 FIX: Fallback alpha from scorecard grade when no forecast
        if abs(alpha) < 1e-8 and hf.scorecard.available:
            _grade_alpha = {
                "A+": 0.08, "A": 0.06, "B+": 0.03, "B": 0.01,
                "C+": -0.005, "C": -0.015, "D": -0.04, "F": -0.08,
            }
            alpha = _grade_alpha.get(hf.scorecard.investment_grade, 0.0)

        # Quality multiplier from scorecard
        q_mult = max(0.6, min(2.0, hf.scorecard.earnings_quality.score / 50))
        result.quality_multiplier = q_mult

        # Survival multiplier
        s_mult = 1.0
        cfg = get_hf_weight("position_sizing.survival_multipliers", {})
        if survival_controller is not None:
            regime = getattr(survival_controller, "current_regime", "normal")
            s_mult = cfg.get(regime, 1.0)
            # Recovery bonus
            recovery = getattr(survival_controller, "detect_recovery_signal", lambda: {})()
            if isinstance(recovery, dict) and recovery.get("active"):
                s_mult *= get_hf_weight("position_sizing.recovery_multiplier", 1.5)
        result.survival_multiplier = s_mult

        # Decay multiplier from filing freshness
        d_mult = 1.0
        if filing_calendar_result is not None:
            age = getattr(filing_calendar_result, "latest_filing_age_days", 0)
            if age > 60:
                d_mult = max(0.5, 1.0 - (age - 60) / 180)
        result.decay_multiplier = d_mult

        # IC-based conviction
        conviction = 0.5
        if signal_ic_result is not None and getattr(signal_ic_result, "available", False):
            best_ic = getattr(signal_ic_result, "best_ic", 0)
            conviction = min(1.0, max(0.1, abs(best_ic) * 10))
        result.conviction = conviction

        # Combine
        result.alpha_base = alpha
        raw = alpha * q_mult * s_mult * d_mult * conviction * 100
        result.signal = max(-1.0, min(1.0, raw))

        # Momentum clamp: prevent fighting strong price trends (config-driven)
        _clamp_cfg = get_hf_weight("position_sizing.momentum_clamp", {})
        if _clamp_cfg.get("enabled", True) and cache is not None and "close" in cache.columns:
            closes = cache["close"].dropna()
            _clamp_lookback = int(_clamp_cfg.get("lookback_days", 63))
            if len(closes) >= _clamp_lookback:
                _ret = float(closes.iloc[-1] / closes.iloc[-_clamp_lookback] - 1)
                _up_thresh = float(_clamp_cfg.get("uptrend_threshold", 0.08))
                _down_thresh = float(_clamp_cfg.get("downtrend_threshold", -0.15))
                _floor = float(_clamp_cfg.get("clamp_floor", -0.1))
                _ceiling = float(_clamp_cfg.get("clamp_ceiling", 0.1))

                if _ret > _up_thresh and result.signal < _floor:
                    result.momentum_override = True
                    result.momentum_override_reason = (
                        f"Signal clamped from {result.signal:.2f} to {_floor:.2f}: "
                        f"{_clamp_lookback}d return +{_ret*100:.1f}% overrides quality-driven sell"
                    )
                    result.signal = _floor
                elif _ret < _down_thresh and result.signal > _ceiling:
                    result.momentum_override = True
                    result.momentum_override_reason = (
                        f"Signal clamped from {result.signal:.2f} to {_ceiling:.2f}: "
                        f"{_clamp_lookback}d return {_ret*100:.1f}% overrides quality-driven buy"
                    )
                    result.signal = _ceiling

        # Label
        if result.signal > 0.5: result.label = "strong_buy"
        elif result.signal > 0.2: result.label = "buy"
        elif result.signal > -0.2: result.label = "hold"
        elif result.signal > -0.5: result.label = "sell"
        else: result.label = "strong_sell"

        # Entry/stop/target from recent price action
        if cache is not None and "close" in cache.columns:
            closes = cache["close"].dropna()
            lookback = get_hf_weight("position_sizing.support_resistance_lookback_days", 63)
            recent = closes.iloc[-lookback:] if len(closes) >= lookback else closes
            if len(recent) >= 10:
                current = float(recent.iloc[-1])
                result.entry_price = current * 0.98 if result.signal > 0 else current * 1.02
                result.stop_price = float(recent.min()) * 0.97
                result.target_price = float(recent.max()) * 1.03
                if result.stop_price and result.target_price and abs(current - result.stop_price) > 1e-6:
                    result.risk_reward_ratio = abs(result.target_price - current) / abs(current - result.stop_price)

        # Kelly criterion position sizing (Kelly 1956)
        # Uses MC terminal values to estimate P(win) and avg win/loss ratio
        if mc_result is not None and hasattr(mc_result, "terminal_values"):
            try:
                import numpy as _np
                _tv = mc_result.terminal_values
                # Get 252d terminal values if available, else longest horizon
                _horizon_key = "252d" if "252d" in _tv else (max(_tv.keys()) if _tv else None)
                if _horizon_key and _tv[_horizon_key] is not None:
                    _terminals = _np.array(_tv[_horizon_key])
                    _wins = _terminals[_terminals > 1.0] - 1.0  # returns > 0
                    _losses = 1.0 - _terminals[_terminals <= 1.0]  # returns < 0
                    if len(_wins) > 0 and len(_losses) > 0:
                        p_win = len(_wins) / len(_terminals)
                        avg_win = float(_np.mean(_wins))
                        avg_loss = float(_np.mean(_losses))
                        if avg_loss > 1e-10:
                            b = avg_win / avg_loss  # odds ratio
                            q = 1 - p_win
                            kelly = (p_win * b - q) / b
                            result.kelly_fraction = float(_np.clip(kelly, -1.0, 1.0))
                            result.half_kelly_size = float(_np.clip(kelly / 2, -0.5, 0.5))
                            result.kelly_edge = float(p_win * b - q)
            except Exception:
                pass

        result.available = True
    except Exception as exc:
        logger.debug("Position signal failed: %s", exc)
        result.error = str(exc)
    return result


# ---------------------------------------------------------------------------
# Master orchestrator
# ---------------------------------------------------------------------------

def run_hedge_fund_analysis(
    income_df: pd.DataFrame,
    balance_df: pd.DataFrame,
    cashflow_df: pd.DataFrame,
    cache: pd.DataFrame | None = None,
    target_profile: dict | None = None,
    forecast_result: Any = None,
    mc_result: Any = None,
    scenario_result: Any = None,
    multi_frequency_result: Any = None,
    signal_ic_result: Any = None,
    filing_calendar_result: Any = None,
    fh_result: Any = None,
    peer_ranking_result: Any = None,
    sentiment_result: Any = None,
    survival_controller: Any = None,
    linked_caches: dict | None = None,
    macro_data: dict | None = None,
) -> HedgeFundResult:
    """Run the complete Hedge Fund Analysis pipeline.

    Parameters
    ----------
    income_df, balance_df, cashflow_df:
        Raw statement DataFrames (primary data source, 8-24 rows each).
    cache:
        Daily cache (for price signals and derived variables).
    All other parameters:
        Upstream model results from the existing pipeline.

    Returns
    -------
    HedgeFundResult with all 15 metrics + scorecard + position signal.
    """
    t0 = time.time()
    hf = HedgeFundResult()

    if income_df is None or income_df.empty:
        logger.info("HF Analysis skipped: no income statement data")
        return hf

    logger.info("Running Hedge Fund Analysis (15 metrics + 15 advanced methods + scorecard)...")

    # --- Tier 1: Earnings Forensics ---
    from operator1.hedge_fund.fcf_quality import compute_fcf_quality
    from operator1.hedge_fund.accruals_forensics import compute_accruals_forensics
    from operator1.hedge_fund.earnings_smoothing import compute_earnings_smoothing

    hf.fcf_quality = compute_fcf_quality(income_df, cashflow_df, balance_df, cache)
    hf.accruals_forensic = compute_accruals_forensics(income_df, balance_df, cashflow_df, cache)
    hf.smoothing = compute_earnings_smoothing(income_df, cashflow_df, cache, fh_result)

    logger.info(
        "  Tier 1 (Earnings): FCF=%d, Accruals=%d, Smoothing=%d",
        int(hf.fcf_quality.score), int(hf.accruals_forensic.red_flag_score),
        int(hf.smoothing.smoothing_index),
    )

    # --- Tier 2: Cash Flow Stress Test ---
    hf.dividend_burn = _compute_dividend_burn(income_df, cashflow_df, balance_df)
    hf.return_spread = _compute_return_spread(income_df, balance_df, cashflow_df)
    hf.operating_leverage = _compute_operating_leverage(income_df)

    logger.info(
        "  Tier 2 (Cash Flow): DivBurn=%d, Spread=%sbps, DOL=%s",
        int(hf.dividend_burn.risk_score),
        f"{hf.return_spread.spread_bps:.0f}" if hf.return_spread.spread_bps else "N/A",
        f"{hf.operating_leverage.dol:.1f}" if hf.operating_leverage.dol else "N/A",
    )

    # --- Tier 3: Balance Sheet Risk ---
    hf.obs_risk = _compute_obs_risk(balance_df, income_df)
    hf.asset_quality = _compute_asset_quality(income_df, balance_df, cashflow_df)
    hf.leverage_stress = _compute_leverage_stress(income_df, balance_df, mc_result)

    logger.info(
        "  Tier 3 (Balance Sheet): OBS=%d, AssetQ=%d, LevStress=%s",
        int(hf.obs_risk.risk_score), int(hf.asset_quality.deterioration_score),
        "breach" if hf.leverage_stress.available and hf.leverage_stress.revenue_miss.covenant_breach else "ok",
    )

    # --- Tier 4: Inflection Detection ---
    hf.momentum = _compute_momentum(income_df, cashflow_df, cache)
    hf.growth_quality = _compute_growth_quality(income_df, balance_df)
    hf.earnings_surprise = _compute_earnings_surprise(income_df, cache, filing_calendar_result)

    logger.info(
        "  Tier 4 (Inflection): Momentum=%d, GrowthQ=%d, Surprise=%s",
        int(hf.momentum.score), int(hf.growth_quality.score),
        hf.earnings_surprise.expected_direction,
    )

    # --- Tier 5: Valuation Engine ---
    hf.dcf = _compute_dcf(cashflow_df, balance_df, cache, target_profile, mc_result, macro_data, income_df=income_df)
    hf_results_dict = {
        "fcf_quality": hf.fcf_quality, "accruals_forensic": hf.accruals_forensic,
        "asset_quality": hf.asset_quality, "leverage_stress": hf.leverage_stress,
        "return_spread": hf.return_spread, "dividend_burn": hf.dividend_burn,
        "obs_risk": hf.obs_risk,
    }
    hf.valuation_quality = _compute_valuation_quality(hf_results_dict, cache, peer_ranking_result)
    hf.peg_composite = _compute_peg(income_df, cache, hf.valuation_quality)

    logger.info(
        "  Tier 5 (Valuation): DCF=%s, Quality=%s, PEG=%s",
        f"${hf.dcf.intrinsic_p50:.2f}" if hf.dcf.available else "N/A",
        hf.valuation_quality.quadrant if hf.valuation_quality.available else "N/A",
        f"{hf.peg_composite.peg_adjusted:.1f}" if hf.peg_composite.peg_adjusted else "N/A",
    )

    # --- Scorecard + Position Signal ---
    hf.scorecard = _build_scorecard(hf)
    hf.position = _compute_position_signal(
        hf, cache, signal_ic_result, survival_controller,
        forecast_result, filing_calendar_result, mc_result,
    )

    # --- Advanced Methods (15 additional techniques) ---
    try:
        from operator1.hedge_fund.advanced_methods import run_advanced_methods
        _adv = run_advanced_methods(
            income_df=income_df,
            balance_df=balance_df,
            cashflow_df=cashflow_df,
            cache=cache,
            linked_caches=linked_caches,
            macro_data=macro_data,
            peer_ranking_result=peer_ranking_result,
        )
        if _adv.available:
            from dataclasses import asdict as _asdict_adv
            hf.advanced = _asdict_adv(_adv)
            logger.info(
                "  Advanced: Piotroski=%d/9, Z''=%s (%s), OU_hl=%s days, Torpedo=%d%%",
                _adv.piotroski_f_score,
                f"{_adv.altman_z_double_prime:.2f}" if _adv.altman_z_double_prime else "N/A",
                _adv.altman_z_dp_zone,
                f"{_adv.ou_mean_reversion.get('half_life_days', 'N/A')}",
                _adv.earnings_torpedo.get("torpedo_risk", 0),
            )
    except Exception as exc:
        logger.debug("Advanced HF methods failed: %s", exc)

    # --- Cross-Pipeline Fusion (8 methods) ---
    fusion_result = None
    try:
        from operator1.hedge_fund.fusion import run_fusion
        _hf_profile = hf.to_profile_dict()
        _main_signal = 0.0
        if forecast_result is not None and hasattr(forecast_result, "forecasts"):
            r5d = forecast_result.forecasts.get("return_5d", {})
            if isinstance(r5d, dict):
                val = r5d.get("5d")
                if val is not None and isinstance(val, (int, float)):
                    _main_signal = float(val) * 100  # scale to -1/+1 range

        fusion_result = run_fusion(
            hf_profile=_hf_profile,
            main_position_signal=max(-1, min(1, _main_signal)),
            multi_frequency_result=multi_frequency_result,
            signal_ic_result=signal_ic_result,
            filing_calendar_result=filing_calendar_result,
            survival_controller=survival_controller,
            cache=cache,
        )
        if fusion_result.available:
            hf.fusion = {
                "available": True,
                "fused_signal": fusion_result.fused_signal,
                "fused_label": fusion_result.fused_label,
                "fused_conviction": fusion_result.fused_conviction,
                "anomaly_overrides": fusion_result.anomaly_overrides,
                "anomaly_frozen": fusion_result.anomaly_frozen,
                "freq_disagreement": fusion_result.freq_disagreement_score,
                "meta_ensemble_agreement": fusion_result.meta_ensemble_agreement,
                "catalyst_regime": fusion_result.catalyst_weight_regime,
                "early_warnings": fusion_result.early_warnings,
                "belief_posterior": fusion_result.belief_network_posterior,
                "action": fusion_result.action,
                "primary_risk": fusion_result.primary_risk,
                "next_catalyst": fusion_result.next_catalyst,
            }
            logger.info(
                "  Fusion: signal=%+.2f (%s), conviction=%.0f%%, posterior=%.1f%%, warnings=%d",
                fusion_result.fused_signal, fusion_result.fused_label,
                fusion_result.fused_conviction * 100,
                fusion_result.belief_network_posterior * 100,
                len(fusion_result.early_warnings),
            )
    except Exception as exc:
        logger.debug("Cross-pipeline fusion failed: %s", exc)

    elapsed = time.time() - t0
    hf.available = True

    logger.info(
        "  HF Analysis complete in %.1fs: Grade=%s, Conviction=%d/10, Signal=%+.2f (%s)",
        elapsed, hf.scorecard.investment_grade, hf.scorecard.conviction,
        hf.position.signal, hf.position.label,
    )

    return hf
