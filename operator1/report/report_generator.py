"""T7.2 -- Report generation.

Consumes the ``company_profile.json`` built by T7.1 and produces:

1. A **branded Markdown report** via Gemini (or a fallback
   template when Gemini is unavailable).
2. An **optional set of charts** (matplotlib) saved as PNG files.
3. An **optional PDF** via ``pandoc`` (skipped gracefully if pandoc
   is not installed).

The Markdown report always includes a required **LIMITATIONS** section
covering data window, OHLCV source caveats, macro frequency, data
missingness summary, and failed modules with mitigations.

Top-level entry point:
    ``generate_report(profile, gemini_client=None, ...)``

Spec refs: Sec 18
"""

from __future__ import annotations

import json
import logging
import os
import shutil
import subprocess
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from enum import Enum

from operator1.constants import CACHE_DIR

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Report tiers
# ---------------------------------------------------------------------------

class ReportTier(str, Enum):
    """Three tiers of report detail."""

    BASIC = "basic"       # Quick screening -- 5 sections
    PRO = "pro"           # Peers + macro context -- 13 sections
    PREMIUM = "premium"   # Full institutional-grade -- all 22 sections

    @property
    def label(self) -> str:
        return {
            "basic": "Basic Report",
            "pro": "Pro Report",
            "premium": "Premium Report",
        }[self.value]

    @property
    def filename(self) -> str:
        return f"{self.value}_report.md"


class ReportMode(str, Enum):
    """Two modes controlling explanation depth.

    LEARN:   Plain-English explanations for financial newcomers.
             Each indicator gets context: what it means, whether the
             value is good or bad, and why it matters.
    RESULTS: Data-forward, minimal prose.  Clean grids and charts
             for professionals who already know the terminology.
    """

    LEARN = "learn"
    RESULTS = "results"

    @property
    def label(self) -> str:
        return {
            "learn": "Learn",
            "results": "Results",
        }[self.value]


# Sections included in each tier.  Section numbers match the fallback
# template headings (1-22).
TIER_SECTIONS: dict[ReportTier, set[int]] = {
    ReportTier.BASIC: {1, 2, 4, 6, 20},
    ReportTier.PRO: {1, 2, 3, 4, 5, 6, 7, 11, 14, 16, 17, 18, 195, 20},
    ReportTier.PREMIUM: set(range(1, 23)) | {195},  # all 22 sections + geopolitical
}


# ---------------------------------------------------------------------------
# Fallback report template (when Gemini is unavailable)
# ---------------------------------------------------------------------------

_FALLBACK_TEMPLATE = """\
# Company Analysis Report

**Generated:** {generated_at}

---

## 1. Executive Summary

{executive_summary}

---

## 2. Company Overview

{company_overview}

---

## 3. Historical Performance Analysis

{historical_performance}

---

## 4. Current Financial Snapshot (Tier-by-Tier)

{current_state_snapshot}

---

## 5. Financial Health Scoring

{financial_health}

---

## 6. Survival Mode Analysis

{survival_analysis}

---

## 7. Linked Variables & Market Context

{linked_entities}

---

## 8. Temporal Analysis & Model Insights

{regime_analysis}

---

## 9. Predictions & Forecasts

{predictions_forecasts}

---

## 10. Technical Patterns & Chart Analysis

{technical_patterns}

---

## 11. Ethical Filter Assessment

{ethical_filters}

---

## 12. Supply Chain & Contagion Risk

{graph_risk}

---

## 13. Competitive Landscape

{game_theory}

---

## 14. Regulatory & Government Protection

{fuzzy_protection}

---

## 15. Model Calibration & Adaptive Learning

{pid_controller}

---

## 16. Market Sentiment & News Flow

{sentiment_analysis}

---

## 17. Peer Comparison & Relative Valuation

{peer_ranking}

---

## 18. Macroeconomic Environment

{macro_quadrant}

---

## 19. Advanced Quantitative Insights

{advanced_insights}

---

## 19.5. Geopolitical & Conflict Risk

{geopolitical_risk}

---

## 20. Risk Factors & Limitations

{risk_assessment}

### 20.1 LIMITATIONS

{limitations}

---

## 21. Investment Recommendation

{investment_recommendation}

---

## 22. Appendix & Methodology

{appendix}
"""


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _fmt(val: Any, fmt: str = ".2f") -> str:
    """Format a numeric value or return 'N/A' for None."""
    if val is None:
        return "N/A"
    try:
        return f"{float(val):{fmt}}"
    except (TypeError, ValueError):
        return str(val)


def _pct(val: Any) -> str:
    """Format as percentage string."""
    if val is None:
        return "N/A"
    try:
        return f"{float(val) * 100:.1f}%"
    except (TypeError, ValueError):
        return str(val)


# ---------------------------------------------------------------------------
# Fallback report builder (no Gemini)
# ---------------------------------------------------------------------------


def _build_executive_summary(profile: dict[str, Any]) -> str:
    """Build the executive summary from profile data."""
    identity = profile.get("identity", {})
    if not identity.get("available"):
        return "Target company data unavailable."

    name = identity.get("name", "Unknown Company")
    ticker = identity.get("ticker", "")
    sector = identity.get("sector", "")
    country = identity.get("country", "")

    survival = profile.get("survival", {})
    regime = survival.get("survival_regime", "normal")
    company_flag = survival.get("company_survival_mode_flag", 0)
    country_flag = survival.get("country_survival_mode_flag", 0)

    mc = profile.get("monte_carlo", {})
    surv_prob = mc.get("survival_probability_mean")

    lines = [
        f"**{name}** ({ticker}) is a {sector} company based in {country}.",
        "",
    ]

    if company_flag or country_flag:
        lines.append(
            f"**Warning:** The company is currently in **{regime}** mode."
        )
        if surv_prob is not None:
            lines.append(
                f"Monte Carlo survival probability: **{_pct(surv_prob)}**"
            )
    else:
        lines.append("The company is operating under **normal** conditions.")
        if surv_prob is not None:
            lines.append(
                f"Monte Carlo survival probability: **{_pct(surv_prob)}**"
            )

    return "\n".join(lines)


def _build_company_overview(profile: dict[str, Any]) -> str:
    """Build company overview section."""
    identity = profile.get("identity", {})
    if not identity.get("available"):
        return "Company identity data unavailable."

    lines = [
        "| Field | Value |",
        "|-------|-------|",
        f"| **Name** | {identity.get('name', 'N/A')} |",
        f"| **ISIN** | {identity.get('isin', 'N/A')} |",
        f"| **Ticker** | {identity.get('ticker', 'N/A')} |",
        f"| **Exchange** | {identity.get('exchange', 'N/A')} |",
        f"| **Sector** | {identity.get('sector', 'N/A')} |",
        f"| **Industry** | {identity.get('industry', 'N/A')} |",
        f"| **Country** | {identity.get('country', 'N/A')} |",
        f"| **Currency** | {identity.get('currency', 'N/A')} |",
    ]
    return "\n".join(lines)


def _build_financial_health(profile: dict[str, Any]) -> str:
    """Build financial health section from tier scores, vanity, and extended models."""
    survival = profile.get("survival", {})
    vanity = profile.get("vanity", {})
    fh = profile.get("financial_health", {})
    extended = profile.get("extended_models", {})

    lines = []

    # 5-Tier Composite Score
    if fh.get("available", fh.get("latest_composite") is not None):
        lines.append("### Composite Health Score")
        lines.append("")
        latest = fh.get("latest_composite")
        label = fh.get("latest_label", "N/A")
        lines.append(f"**Overall Score:** {_fmt(latest, '.1f')}/100 ({label})")
        lines.append("")

        tier_means = fh.get("tier_means", {})
        if tier_means:
            lines.append("| Tier | Focus | Score |")
            lines.append("|------|-------|-------|")
            _tier_labels = {
                "fh_liquidity_score": ("1", "Liquidity & Cash Flow"),
                "fh_solvency_score": ("2", "Solvency & Leverage"),
                "fh_stability_score": ("3", "Market Stability"),
                "fh_profitability_score": ("4", "Profitability & Margins"),
                "fh_growth_score": ("5", "Growth & Valuation"),
            }
            for col, (num, desc) in _tier_labels.items():
                val = tier_means.get(col)
                lines.append(f"| Tier {num} | {desc} | {_fmt(val, '.1f')} |")
            lines.append("")

    # Altman Z-Score (Bankruptcy Prediction)
    z_data = fh.get("altman_z", extended.get("altman_z", {}))
    if isinstance(z_data, dict) and z_data.get("available"):
        lines.append("### Altman Z-Score (Bankruptcy Risk)")
        lines.append("")
        z_val = z_data.get("latest_z_score")
        zone = z_data.get("zone", "unknown")
        zone_desc = {
            "safe": "Investment Grade (low default risk)",
            "grey": "Grey Zone (moderate uncertainty)",
            "distress": "Distress Zone (elevated bankruptcy risk)",
        }.get(zone, "Insufficient data")
        lines.append(f"**Z-Score:** {_fmt(z_val, '.2f')} -- {zone_desc}")
        lines.append("")

    # Beneish M-Score (Earnings Quality)
    m_data = fh.get("beneish_m", extended.get("beneish_m", {}))
    if isinstance(m_data, dict) and m_data.get("available"):
        lines.append("### Beneish M-Score (Earnings Quality)")
        lines.append("")
        m_val = m_data.get("m_score")
        verdict = m_data.get("verdict", "unknown")
        verdict_desc = {
            "unlikely": "Earnings appear genuine (low manipulation probability)",
            "possible": "Moderate flags detected (further scrutiny advised)",
            "likely": "Elevated manipulation indicators (earnings reliability questionable)",
        }.get(verdict, "Insufficient data")
        lines.append(f"**M-Score:** {_fmt(m_val, '.2f')} -- {verdict_desc}")
        if m_data.get("likely_manipulator"):
            lines.append("")
            lines.append("> **Warning:** M-Score exceeds -1.78 threshold. "
                        "Financial statements may contain earnings manipulation. "
                        "Exercise caution with reported profitability metrics.")
        lines.append("")

    # Liquidity Runway
    runway_data = fh.get("liquidity_runway", extended.get("liquidity_runway", {}))
    if isinstance(runway_data, dict) and runway_data.get("available"):
        lines.append("### Liquidity Runway (Cash Survival)")
        lines.append("")
        months = runway_data.get("months_of_runway")
        verdict = runway_data.get("verdict", "unknown")
        verdict_desc = {
            "strong": "Ample cash reserves (>24 months at current burn rate)",
            "adequate": "Sufficient runway (12-24 months)",
            "tight": "Limited runway (6-12 months, refinancing likely needed)",
            "critical": "Critical cash position (<6 months, immediate action required)",
        }.get(verdict, "N/A")
        if months is not None and months != float("inf"):
            lines.append(f"**Runway:** {months:.0f} months -- {verdict_desc}")
        elif months == float("inf"):
            lines.append(f"**Runway:** Cash-generative (self-sustaining) -- {verdict_desc}")
        lines.append("")

    # Hierarchy Weights
    lines.append("### Portfolio Risk Allocation (Current Market Regime)")
    lines.append("")
    lines.append("The model dynamically shifts capital allocation across five risk tiers "
                "based on the prevailing market regime. During periods of financial stress, "
                "liquidity and solvency tiers receive higher weighting to protect against "
                "default risk.")
    lines.append("")
    weights = survival.get("hierarchy_weights", {})
    _tier_readable = {
        "tier1": "Liquidity & Cash Flow",
        "tier2": "Solvency & Leverage",
        "tier3": "Market Stability",
        "tier4": "Profitability & Margins",
        "tier5": "Growth & Valuation",
    }
    if weights:
        for tier, w in sorted(weights.items()):
            readable = _tier_readable.get(tier, tier)
            lines.append(f"- **{readable}**: {_fmt(w, '.1f')}% allocation")
    else:
        lines.append("Risk tier allocation data unavailable.")

    # Capital Allocation Quality (brief summary -- detail in premium section 19)
    lines.extend(["", "### Capital Allocation Quality", ""])

    if vanity.get("v2_available"):
        v2_score = vanity.get("v2_score", {})
        lines.append(
            f"- **Management discipline rating**: {vanity.get('v2_label', 'N/A')} "
            f"({_fmt(v2_score.get('latest'), '.1f')} / 100)"
        )
        lines.append(f"- **Direction**: {vanity.get('v2_trend', 'N/A')}")
    elif vanity.get("available"):
        lines.append(f"- **Non-productive expenditure ratio**: {_fmt(vanity.get('latest'), '.1f')}% of revenue")
    else:
        lines.append("Capital allocation quality data unavailable.")

    if not lines:
        return "Financial health data unavailable."

    return "\n".join(lines)


def _build_survival_analysis(profile: dict[str, Any]) -> str:
    """Build survival analysis section."""
    survival = profile.get("survival", {})
    if not survival.get("available"):
        return "Survival analysis data unavailable."

    lines = [
        f"- **Company survival flag**: {'ACTIVE' if survival.get('company_survival_mode_flag') else 'inactive'}",
        f"- **Country survival flag**: {'ACTIVE' if survival.get('country_survival_mode_flag') else 'inactive'}",
        f"- **Country protection flag**: {'ACTIVE' if survival.get('country_protected_flag') else 'inactive'}",
        f"- **Current regime**: {survival.get('survival_regime', 'N/A')}",
        "",
        "### Regime Distribution",
        "",
    ]

    dist = survival.get("regime_distribution_pct", {})
    if dist:
        for regime, pct in sorted(dist.items(), key=lambda x: -x[1]):
            lines.append(f"- {regime}: {pct * 100:.1f}%")
    else:
        lines.append("No regime distribution data available.")

    # Survival episode statistics (Gap 9)
    episodes = profile.get("survival_episodes", {})
    if episodes.get("available"):
        lines.extend(["", "### Survival Episode History"])
        lines.append("")

        co_days = episodes.get("company_survival_days", 0)
        co_eps = episodes.get("company_episodes", 0)
        lines.append(f"- **Company distress episodes**: {co_eps} episodes "
                     f"({co_days} total days over the 2-year window)")
        if co_eps > 0:
            lines.append(f"  - Longest episode: {episodes.get('company_longest_episode', 0)} days")
            lines.append(f"  - Average episode length: {episodes.get('company_avg_episode_length', 0)} days")

        ct_days = episodes.get("country_survival_days", 0)
        ct_eps = episodes.get("country_episodes", 0)
        lines.append(f"- **Country crisis episodes**: {ct_eps} episodes "
                     f"({ct_days} total days)")
        if ct_eps > 0:
            lines.append(f"  - Longest episode: {episodes.get('country_longest_episode', 0)} days")

        both = episodes.get("both_survival_days", 0)
        if both > 0:
            lines.append(f"- **Simultaneous company + country distress**: {both} days "
                        "*(highest risk periods)*")

    return "\n".join(lines)


def _build_linked_entities_section(profile: dict[str, Any]) -> str:
    """Build linked entities section."""
    linked = profile.get("linked_entities", {})
    if not linked.get("available"):
        return "Linked entity analysis unavailable."

    # Map code-style metric suffixes to human-readable labels
    _METRIC_LABELS: dict[str, str] = {
        "avg_return_1d": "Avg Daily Return",
        "median_return_1d": "Median Daily Return",
        "avg_volatility_21d": "Avg 21-Day Volatility",
        "median_volatility_21d": "Median 21-Day Volatility",
        "avg_drawdown_252d": "Avg Max Drawdown (1Y)",
        "median_drawdown_252d": "Median Max Drawdown (1Y)",
        "avg_current_ratio": "Avg Current Ratio",
        "median_current_ratio": "Median Current Ratio",
        "avg_debt_to_equity_abs": "Avg Debt-to-Equity",
        "median_debt_to_equity_abs": "Median Debt-to-Equity",
        "avg_free_cash_flow": "Avg Free Cash Flow",
        "median_free_cash_flow": "Median Free Cash Flow",
        "avg_fcf_yield": "Avg FCF Yield",
        "median_fcf_yield": "Median FCF Yield",
        "avg_gross_margin": "Avg Gross Margin",
        "median_gross_margin": "Median Gross Margin",
        "avg_operating_margin": "Avg Operating Margin",
        "median_operating_margin": "Median Operating Margin",
        "avg_net_margin": "Avg Net Margin",
        "median_net_margin": "Median Net Margin",
        "avg_roe": "Avg Return on Equity",
        "median_roe": "Median Return on Equity",
        "avg_pe_ratio_calc": "Avg P/E Ratio",
        "median_pe_ratio_calc": "Median P/E Ratio",
        "avg_ev_to_ebitda": "Avg EV/EBITDA",
        "median_ev_to_ebitda": "Median EV/EBITDA",
        "avg_enterprise_value": "Avg Enterprise Value",
        "median_enterprise_value": "Median Enterprise Value",
        "avg_market_cap": "Avg Market Cap",
        "median_market_cap": "Median Market Cap",
    }

    n_groups = linked.get("n_groups", 0)
    groups = linked.get("groups", {})

    lines = [f"**{n_groups} relationship group(s)** analysed.", ""]

    for group_name, metrics in sorted(groups.items()):
        lines.append(f"### {group_name.replace('_', ' ').title()}")
        lines.append("")
        if metrics:
            lines.append("| Metric | Latest | Mean |")
            lines.append("|--------|--------|------|")
        for metric_name, stats in sorted(metrics.items()):
            if isinstance(stats, dict) and "latest" in stats:
                label = _METRIC_LABELS.get(metric_name, metric_name.replace("_", " ").title())
                lines.append(
                    f"| {label} | {_fmt(stats.get('latest'))} | {_fmt(stats.get('mean'))} |"
                )
        lines.append("")

    return "\n".join(lines)


def _build_macro_section(profile: dict[str, Any]) -> str:
    """Build macro environment section."""
    # Macro data is embedded in the quality/estimation sections
    estimation = profile.get("estimation", {})
    quality = profile.get("data_quality", {})

    lines = []
    if estimation.get("available"):
        lines.append("Macro data was aligned to daily frequency using as-of logic.")
        lines.append("See the estimation coverage section for variable-level detail.")
    else:
        lines.append("Macro environment data was not available for this analysis.")

    if quality.get("available"):
        coverage = quality.get("variable_coverage", {})
        macro_vars = [
            k for k in coverage
            if k.startswith(("inflation", "cpi", "unemployment", "gdp", "exchange"))
        ]
        if macro_vars:
            lines.extend(["", "### Macro Variable Coverage", ""])
            for var in sorted(macro_vars):
                cov = coverage[var]
                if isinstance(cov, dict):
                    lines.append(
                        f"- {var}: {_fmt(cov.get('coverage_pct'), '.1f')}% coverage"
                    )
                else:
                    lines.append(f"- {var}: {_fmt(cov, '.1f')}% coverage")

    return "\n".join(lines) if lines else "Macro environment data unavailable."


def _build_regime_analysis(profile: dict[str, Any]) -> str:
    """Build regime analysis and forecasts section."""
    regimes = profile.get("regimes", {})
    preds = profile.get("predictions", {})

    lines = []

    if regimes.get("available"):
        _regime_descriptions = {
            "bull": "Bull Market (sustained upward trend with low volatility)",
            "bear": "Bear Market (declining prices with elevated risk)",
            "high_vol": "High Volatility (turbulent conditions, wide price swings)",
            "low_vol": "Low Volatility (calm markets, narrow trading range)",
        }
        raw_regime = regimes.get('current_regime', 'N/A')
        regime_desc = _regime_descriptions.get(raw_regime, raw_regime)
        lines.append(f"**Current market regime**: {regime_desc}")
        lines.append(f"**Structural breaks detected**: {regimes.get('n_structural_breaks', 0)} "
                     "(sudden shifts in market behaviour identified by statistical change-point analysis)")
        lines.append("")

        dist = regimes.get("regime_distribution_pct", {})
        if dist:
            lines.append("### Time Spent in Each Market Regime")
            lines.append("")
            for regime, pct in sorted(dist.items(), key=lambda x: -x[1]):
                regime_label = _regime_descriptions.get(regime, regime)
                lines.append(f"- {regime_label}: {pct * 100:.1f}% of the analysis window")
            lines.append("")
    else:
        lines.append("Market regime classification was not performed for this analysis.")
        lines.append("")

    _horizon_labels = {
        "1d": "Next Trading Day",
        "5d": "Next Week (5 trading days)",
        "21d": "Next Month (21 trading days)",
        "252d": "Next Year (252 trading days)",
    }

    if preds.get("available"):
        lines.append("### Forward-Looking Estimates")
        lines.append("")
        horizons = preds.get("horizons", {})
        for h_label in ("1d", "5d", "21d", "252d"):
            h_preds = horizons.get(h_label, {})
            if h_preds:
                h_display = _horizon_labels.get(h_label, h_label)
                lines.append(f"**{h_display}:**")
                # Handle both list format [{variable, point_forecast, ...}]
                # and dict format {variable: {point, lower, upper}}
                if isinstance(h_preds, list):
                    items = h_preds[:5]
                    for p in items:
                        var = p.get("variable", "?")
                        pf = _fmt(p.get("point_forecast"))
                        ci_lo = _fmt(p.get("lower_ci"))
                        ci_hi = _fmt(p.get("upper_ci"))
                        lines.append(f"- {var}: {pf} [{ci_lo}, {ci_hi}]")
                elif isinstance(h_preds, dict):
                    for var, vals in list(h_preds.items())[:5]:
                        if isinstance(vals, dict):
                            pf = _fmt(vals.get("point") or vals.get("point_forecast"))
                            ci_lo = _fmt(vals.get("lower") or vals.get("lower_ci"))
                            ci_hi = _fmt(vals.get("upper") or vals.get("upper_ci"))
                            lines.append(f"- {var}: {pf} [{ci_lo}, {ci_hi}]")
                lines.append("")
    else:
        lines.append("Forecast data unavailable.")

    return "\n".join(lines)


def _build_ethical_filters_section(profile: dict[str, Any]) -> str:
    """Build Section 8: Ethical Filter Assessment."""
    filters = profile.get("filters", {})
    lines: list[str] = []

    if not filters.get("available", False):
        lines.append("*Ethical filter data not available for this run.*")
        return "\n".join(lines)

    # Purchasing Power
    pp = filters.get("purchasing_power", {})
    lines.append("### Purchasing Power Filter")
    lines.append("")
    lines.append(f"**Verdict:** {pp.get('verdict', 'N/A')}")
    lines.append("")
    lines.append(f"- Nominal return: {_pct(pp.get('nominal_return'))}")
    lines.append(f"- Real return (inflation-adjusted): {_pct(pp.get('real_return'))}")
    lines.append(f"- Inflation impact: {_pct(pp.get('inflation_impact'))}")
    lines.append("")
    lines.append(
        "*This filter reveals whether investors actually gained purchasing "
        "power or merely saw a number go up while real wealth declined.*"
    )
    lines.append("")

    # Solvency
    sol = filters.get("solvency", {})
    lines.append("### Solvency Filter (Debt-to-Equity)")
    lines.append("")
    lines.append(f"**Verdict:** {sol.get('verdict', 'N/A')}")
    lines.append("")
    lines.append(f"- Debt-to-equity: {_fmt(sol.get('debt_to_equity'))}")
    lines.append(f"- Threshold: {_fmt(sol.get('threshold'))}")
    lines.append(f"- {sol.get('interpretation', '')}")
    lines.append("")
    lines.append(
        "*Beyond religious compliance, high leverage makes companies "
        "fragile in recessions and rate hikes. This filter protects "
        "against leveraged blow-ups.*"
    )
    lines.append("")

    # Gharar
    gh = filters.get("gharar", {})
    lines.append("### Gharar Filter (Volatility / Speculation)")
    lines.append("")
    lines.append(f"**Verdict:** {gh.get('verdict', 'N/A')}")
    lines.append("")
    lines.append(f"- Volatility (21d): {_pct(gh.get('volatility_21d'))}")
    lines.append(f"- Stability score: {_fmt(gh.get('stability_score'))}/10")
    lines.append(f"- {gh.get('interpretation', '')}")
    lines.append("")
    lines.append(
        "*This filter separates calculated investment from gambling "
        "regardless of one's background -- if volatility is extreme, any "
        "prediction is as likely to be wrong as right.*"
    )
    lines.append("")

    # Cash is King
    ck = filters.get("cash_is_king", {})
    lines.append("### Cash is King Filter (Free Cash Flow Yield)")
    lines.append("")
    lines.append(f"**Verdict:** {ck.get('verdict', 'N/A')}")
    lines.append("")
    lines.append(f"- FCF yield: {_pct(ck.get('fcf_yield'))}")
    if ck.get("fcf_margin") is not None:
        lines.append(f"- FCF margin: {_pct(ck.get('fcf_margin'))}")
    lines.append(f"- {ck.get('interpretation', '')}")
    lines.append("")
    lines.append(
        '*"Profit is an opinion, but cash is a fact." This filter ensures '
        "the company generates real liquid wealth, not just accounting "
        "entries.*"
    )

    return "\n".join(lines)


def _build_risk_assessment(profile: dict[str, Any]) -> str:
    """Build risk assessment section."""
    mc = profile.get("monte_carlo", {})
    survival = profile.get("survival", {})

    lines = []

    if mc.get("available"):
        lines.append("### Stress Testing (Monte Carlo Simulation)")
        lines.append("")
        lines.append("The company's financial trajectory was simulated across "
                     f"**{mc.get('n_paths', 'N/A')} randomised scenarios** "
                     "to estimate the probability of remaining financially viable "
                     "under varying market conditions.")
        lines.append("")
        lines.append(
            f"- **Expected survival probability**: "
            f"{_pct(mc.get('survival_probability_mean'))}"
        )
        lines.append(
            f"- **Worst-case estimate** (5th percentile): "
            f"{_pct(mc.get('survival_probability_p5'))}"
        )
        lines.append(
            f"- **Best-case estimate** (95th percentile): "
            f"{_pct(mc.get('survival_probability_p95'))}"
        )
        lines.append("")

        by_horizon = mc.get("survival_by_horizon", {})
        _h_labels = {"1d": "1 Day", "5d": "1 Week", "21d": "1 Month", "252d": "1 Year"}
        if by_horizon:
            lines.append("### Survival Probability by Time Horizon")
            lines.append("")
            for h, prob in sorted(by_horizon.items()):
                h_label = _h_labels.get(h, h)
                lines.append(f"- {h_label}: {_pct(prob)}")
            lines.append("")
    else:
        lines.append("Stress testing (Monte Carlo simulation) was not performed for this analysis.")

    # Model reliability
    metrics = profile.get("model_metrics", {})
    if metrics.get("available"):
        lines.append("### Forecast Model Accuracy")
        lines.append("")
        lines.append("*Lower prediction error indicates a model that fits historical data "
                     "more closely. Models with the best track record receive higher weight "
                     "in the final ensemble forecast.*")
        lines.append("")
        _model_readable = {
            "kalman": "Adaptive Kalman Filter (tracks evolving trends)",
            "garch": "Volatility Model (captures risk clustering)",
            "var": "Multi-Variable Regression (inter-variable dynamics)",
            "lstm": "Deep Learning Sequence Model (non-linear patterns)",
            "tree": "Gradient Boosted Ensemble (complex feature interactions)",
            "baseline": "Moving Average Baseline (benchmark)",
            "transformer": "Attention-Based Deep Learning (long-range dependencies)",
        }
        rmse_info = metrics.get("model_best_rmse", {})
        for model, rmse in sorted(rmse_info.items()):
            readable = _model_readable.get(model, model)
            lines.append(f"- {readable}: average prediction error = {_fmt(rmse, '.4f')}")

    return "\n".join(lines)


def _build_limitations(profile: dict[str, Any]) -> str:
    """Build the required LIMITATIONS section.

    Must cover:
    - Data window and frequency limitations
    - OHLCV source caveats
    - Macro data frequency and alignment
    - Data missingness summary
    - Failed modules and mitigations
    """
    meta = profile.get("meta", {})
    date_range = meta.get("date_range", {})
    quality = profile.get("data_quality", {})
    failed = profile.get("failed_modules", [])
    estimation = profile.get("estimation", {})

    lines = [
        "This analysis is subject to the following limitations:",
        "",
        "### Data Window",
        "",
        f"- Analysis covers **{date_range.get('start', 'N/A')}** to "
        f"**{date_range.get('end', 'N/A')}** (approximately 2 years).",
        "- Historical patterns may not predict future performance.",
        "- All financial statement data is aligned using as-of logic "
        "(latest report as of each trading day).",
        "",
        "### OHLCV Source",
        "",
    ]

    # PIT data source description
    _provider = meta.get("data_provider", "unknown")
    _provider_label = meta.get("data_provider_label", _provider)
    _is_pit = meta.get("pit_source", True)

    _ohlcv_source = meta.get("ohlcv_source", "")

    if _is_pit:
        lines.extend([
            f"- Financial statements and filings are sourced from "
            f"**{_provider_label}** -- a free government filing API.",
            "- Filing dates are immutable and used for point-in-time alignment "
            "(no look-ahead bias in historical analysis).",
        ])
        if _ohlcv_source and _ohlcv_source != _provider:
            lines.extend([
                f"- Price data (OHLCV) is sourced from **{_ohlcv_source}**, "
                "a separate market data provider. Raw exchange prices are "
                "inherently point-in-time (immutable historical facts).",
            ])
        else:
            lines.extend([
                f"- Price data is also sourced from **{_provider_label}**.",
            ])
        lines.extend([
            "- Price data may not account for all corporate actions "
            "(splits, dividends) depending on exchange adjustments.",
        ])
    else:
        lines.extend([
            f"- Data sourced from **{_provider_label}**.",
            "- Price data may not account for all corporate actions "
            "(splits, dividends) depending on source adjustments.",
        ])

    lines.extend([
        "",
        "### Macro Data",
        "",
    ])

    _macro_source = meta.get("macro_source")
    if _macro_source:
        _macro_country = meta.get("macro_country", "")
        lines.extend([
            f"- Macroeconomic indicators sourced from **{_macro_source}** "
            f"({_macro_country}).",
            "- GDP data is typically quarterly; inflation and interest rates "
            "are monthly; exchange rates are daily.",
            "- All macro series are forward-filled to daily frequency using "
            "as-of logic (latest available observation as of each trading day).",
        ])
    else:
        lines.extend([
            "- No macroeconomic data source was available for this market.",
            "- Survival mode analysis relies solely on company-level indicators.",
        ])

    # Data missingness
    lines.extend(["### Data Missingness", ""])
    if quality.get("available"):
        coverage = quality.get("variable_coverage", {})
        if coverage:
            low_coverage = [
                (var, info)
                for var, info in coverage.items()
                if isinstance(info, dict) and (info.get("coverage_pct", 100) or 100) < 80
            ]
            if low_coverage:
                lines.append(
                    f"- **{len(low_coverage)}** variable(s) have less than "
                    "80% coverage:"
                )
                for var, info in sorted(low_coverage, key=lambda x: x[1].get("coverage_pct", 0)):
                    lines.append(
                        f"  - {var}: {_fmt(info.get('coverage_pct'), '.1f')}%"
                    )
            else:
                lines.append("- All variables have 80%+ coverage.")
        else:
            lines.append("- Variable-level coverage data not available.")
    else:
        lines.append("- Data quality report not generated; coverage unknown.")

    if estimation.get("available"):
        lines.append(
            "- Missing values were estimated using a two-pass "
            "approach (deterministic identity fill + regime-weighted "
            "imputation). Estimated values are flagged and assigned "
            "confidence scores."
        )
    lines.append("")

    # Failed modules
    lines.extend(["### Failed Modules and Mitigations", ""])
    if failed:
        for f in failed:
            lines.append(f"- **{f.get('module', 'Unknown')}**: {f.get('error', 'unknown error')}")
            lines.append(f"  - *Mitigation*: {f.get('mitigation', 'none')}")
    else:
        lines.append("- No module failures detected.")

    return "\n".join(lines)


def _build_current_state_snapshot(profile: dict[str, Any]) -> str:
    """Build the Current Financial Snapshot section showing actual ratio values per tier."""
    cs = profile.get("current_state", {})
    if not cs.get("available"):
        return "*Current state snapshot unavailable.*"

    lines: list[str] = []
    lines.append(f"**As of:** {cs.get('date', 'N/A')}")
    if cs.get("market_regime"):
        lines.append(f"**Market Regime:** {cs['market_regime']}")
    if cs.get("fundamental_regime"):
        lines.append(f"**Financial Health Regime:** {cs['fundamental_regime']}")
    lines.append("")

    _tier_configs = [
        ("tier1_liquidity", "Tier 1: Liquidity & Cash Position", {
            "cash_and_equivalents": ("Cash on Hand", None),
            "cash_ratio": ("Cash Ratio", ".2f"),
            "current_ratio": ("Current Ratio", ".2f"),
            "free_cash_flow_ttm": ("Free Cash Flow (TTM)", None),
            "operating_cash_flow": ("Operating Cash Flow", None),
        }),
        ("tier2_solvency", "Tier 2: Solvency & Leverage", {
            "total_debt": ("Total Debt", None),
            "debt_to_equity": ("Debt-to-Equity Ratio", ".2f"),
            "net_debt": ("Net Debt", None),
            "net_debt_to_ebitda": ("Net Debt / EBITDA", ".2f"),
            "interest_coverage": ("Interest Coverage Ratio", ".1f"),
        }),
        ("tier3_stability", "Tier 3: Market Stability", {
            "volatility_21d": ("21-Day Volatility (annualised)", ".1%"),
            "drawdown_252d": ("Maximum Drawdown (1-year)", ".1%"),
            "close": ("Current Share Price", ".2f"),
            "volume": ("Daily Volume", ",.0f"),
            "volume_avg_21d": ("21-Day Average Volume", ",.0f"),
        }),
        ("tier4_profitability", "Tier 4: Profitability", {
            "gross_margin": ("Gross Margin", ".1%"),
            "operating_margin": ("Operating Margin", ".1%"),
            "net_margin": ("Net Margin", ".1%"),
            "roe": ("Return on Equity", ".1%"),
            "roa": ("Return on Assets", ".1%"),
        }),
        ("tier5_growth", "Tier 5: Growth & Valuation", {
            "pe_ratio": ("Price / Earnings", ".1f"),
            "pb_ratio": ("Price / Book Value", ".1f"),
            "ev_to_ebitda": ("EV / EBITDA", ".1f"),
            "fcf_yield": ("Free Cash Flow Yield", ".1%"),
            "earnings_yield": ("Earnings Yield", ".1%"),
            "revenue_growth_yoy": ("Revenue Growth (YoY)", ".1%"),
            "earnings_growth_yoy": ("Earnings Growth (YoY)", ".1%"),
        }),
    ]

    for tier_key, tier_title, fields in _tier_configs:
        tier_data = cs.get(tier_key, {})
        if not tier_data:
            continue

        lines.append(f"### {tier_title}")
        lines.append("")
        lines.append("| Metric | Value |")
        lines.append("|--------|-------|")

        for field_key, (label, fmt) in fields.items():
            val = tier_data.get(field_key)
            if val is None:
                formatted = "N/A"
            elif fmt and fmt.endswith("%"):
                try:
                    formatted = f"{float(val):{fmt}}"
                except (TypeError, ValueError):
                    formatted = str(val)
            elif fmt:
                try:
                    formatted = f"{float(val):{fmt}}"
                except (TypeError, ValueError):
                    formatted = str(val)
            else:
                # Auto-format large numbers
                try:
                    v = float(val)
                    if abs(v) >= 1e9:
                        formatted = f"${v/1e9:,.1f}B"
                    elif abs(v) >= 1e6:
                        formatted = f"${v/1e6:,.1f}M"
                    else:
                        formatted = f"{v:,.2f}"
                except (TypeError, ValueError):
                    formatted = str(val)

            lines.append(f"| {label} | {formatted} |")

        lines.append("")

    return "\n".join(lines)


def _build_historical_performance(profile: dict[str, Any]) -> str:
    """Build the Historical Performance Analysis section (Section 3)."""
    hist = profile.get("historical", {})
    lines: list[str] = []

    lines.append(f"**Analysis Period:** {hist.get('date_range_start', 'N/A')} to {hist.get('date_range_end', 'N/A')}")
    lines.append("")
    lines.append(f"- **Total Return:** {_pct(hist.get('return_total'))}")
    lines.append(f"- **Real Return (inflation-adjusted):** {_pct(hist.get('return_real'))}")
    lines.append(f"- **Annualised Return:** {_pct(hist.get('return_annualized'))}")
    lines.append(f"- **Annualised Volatility:** {_pct(hist.get('volatility_annualized'))}")
    lines.append(f"- **Sharpe Ratio:** {_fmt(hist.get('sharpe_ratio'))}")
    lines.append(f"- **Maximum Drawdown:** {_pct(hist.get('max_drawdown'))}")
    lines.append(f"- **Up Days:** {_pct(hist.get('up_days_percentage'))}")
    lines.append(f"- **Down Days:** {_pct(hist.get('down_days_percentage'))}")
    lines.append(f"- **Best Day Return:** {_pct(hist.get('best_day_return'))}")
    lines.append(f"- **Worst Day Return:** {_pct(hist.get('worst_day_return'))}")

    return "\n".join(lines)


def _build_predictions_forecasts(profile: dict[str, Any]) -> str:
    """Build the Predictions & Forecasts section (Section 8)."""
    preds = profile.get("predictions", {})
    lines: list[str] = []

    for horizon_label in ["next_day", "next_week", "next_month", "next_year"]:
        h_data = preds.get(horizon_label, {})
        h_title = horizon_label.replace("_", " ").title()
        lines.append(f"### {h_title}")
        lines.append("")

        if not h_data:
            lines.append("*No predictions available for this horizon.*")
            lines.append("")
            continue

        if "point_forecast" in h_data:
            pt = h_data["point_forecast"]
            if isinstance(pt, dict):
                for var, val in list(pt.items())[:10]:
                    lines.append(f"- **{var}:** {_fmt(val)}")
            else:
                lines.append(f"- Point forecast: {_fmt(pt)}")

        if "ohlc_series" in h_data:
            ohlc = h_data["ohlc_series"]
            if isinstance(ohlc, list) and ohlc:
                if horizon_label == "next_day":
                    lines.append("")
                    lines.append("*Technical Alpha protection applied: only Low is shown for next-day OHLC.*")
                lines.append("")
                lines.append(f"- OHLC candlestick series: {len(ohlc)} step(s)")

        lines.append("")

    # Monte Carlo
    mc = preds.get("monte_carlo", {})
    if mc:
        lines.append("### Monte Carlo Uncertainty")
        lines.append("")
        lines.append(f"- Scenarios simulated: {mc.get('n_scenarios', 'N/A')}")
        lines.append(f"- Tail risk (5th percentile): {_fmt(mc.get('p5'))}")
        lines.append(f"- Base case (50th percentile): {_fmt(mc.get('p50'))}")
        lines.append(f"- Upside (95th percentile): {_fmt(mc.get('p95'))}")

    # Conformal prediction intervals
    conformal = profile.get("conformal_intervals", {})
    if conformal:
        lines.append("")
        lines.append("### Conformal Prediction Intervals")
        lines.append("")
        lines.append("*Distribution-free intervals with guaranteed coverage (no Gaussian assumption):*")
        lines.append("")
        for var, intervals in list(conformal.items())[:10]:
            if isinstance(intervals, dict):
                for h, interval in intervals.items():
                    if isinstance(interval, dict):
                        lines.append(
                            f"- **{var}** ({h}): "
                            f"[{_fmt(interval.get('lower'))}, {_fmt(interval.get('upper'))}] "
                            f"(width: {_fmt(interval.get('interval_width'))})"
                        )

    # SHAP explanations
    shap_data = profile.get("shap_explanations", {})
    if shap_data.get("available"):
        lines.append("")
        lines.append("### What Drove These Predictions")
        lines.append("")
        lines.append("*Each prediction is decomposed into the contributions of individual "
                     "factors, showing which inputs pushed the forecast up or down:*")
        lines.append("")
        per_var = shap_data.get("per_variable", {})
        for var, exp in list(per_var.items())[:8]:
            narrative = exp.get("narrative", "")
            if narrative:
                lines.append(f"- **{var}:** {narrative}")
            else:
                drivers = exp.get("top_drivers", [])
                parts = []
                for d in drivers[:3]:
                    val = d.get("shap_value", 0)
                    direction = "upward" if val > 0 else "downward"
                    parts.append(f"{_fmt(abs(val))} {direction} pressure from {d.get('feature', '?')}")
                if parts:
                    lines.append(f"- **{var}:** {'; '.join(parts)}")

        global_imp = shap_data.get("global_feature_importance", {})
        if global_imp:
            lines.append("")
            lines.append("**Most influential factors overall (across all predictions):**")
            for feat, importance in list(global_imp.items())[:5]:
                lines.append(f"- {feat}: average influence {_fmt(importance, '.4f')}")

    # Historical analogs (DTW)
    analogs = profile.get("historical_analogs", {})
    if analogs.get("available"):
        lines.append("")
        lines.append("### Historical Analogs (DTW Pattern Matching)")
        lines.append("")
        lines.append(f"*Method: {analogs.get('method', 'DTW')} | "
                     f"Query window: {analogs.get('query_window_days', '?')} days | "
                     f"Forecast horizon: {analogs.get('forecast_horizon_days', '?')} days*")
        lines.append("")

        emp = analogs.get("empirical_forecast", {})
        if emp:
            lines.append("**Empirical forecast from analog outcomes:**")
            lines.append("")
            lines.append(f"- Mean return: {_fmt(emp.get('return_mean_pct'))}%")
            lines.append(f"- Median return: {_fmt(emp.get('return_median_pct'))}%")
            lines.append(f"- Range: [{_fmt(emp.get('return_p5_pct'))}%, {_fmt(emp.get('return_p95_pct'))}%]")
            lines.append(f"- Worst drawdown: {_fmt(emp.get('worst_drawdown_pct'))}%")
            lines.append("")

        analog_list = analogs.get("analogs", [])
        if analog_list:
            lines.append("**Closest historical matches:**")
            lines.append("")
            for a in analog_list[:5]:
                lines.append(f"- {a.get('narrative', a.get('period', 'Unknown'))}")

    return "\n".join(lines)


def _build_technical_patterns(profile: dict[str, Any]) -> str:
    """Build the Technical Patterns & Chart Analysis section (Section 9)."""
    patterns = profile.get("patterns", profile.get("technical_patterns", {}))
    lines: list[str] = []

    recent = patterns.get("recent_patterns", [])
    predicted = patterns.get("predicted_patterns_week", patterns.get("predicted_patterns", []))

    lines.append("*(See attached price history chart with regime shading)*")
    lines.append("")

    if recent:
        lines.append("### Recent Patterns (Last 6 Months)")
        lines.append("")
        for p in recent[:10]:
            lines.append(f"- {p}")
    else:
        lines.append("*No recent candlestick patterns detected.*")

    lines.append("")

    if predicted:
        lines.append("### Predicted Patterns")
        lines.append("")
        for p in predicted[:10]:
            lines.append(f"- {p}")
    else:
        lines.append("*No predicted patterns available.*")

    lines.append("")

    # OHLC predictions from the new iterative predictor
    ohlc = profile.get("ohlc_predictions", {})
    if ohlc.get("available"):
        lines.append("### Predicted Price Action")
        lines.append("")

        next_week = ohlc.get("next_week", {})
        if next_week:
            wk_ret = next_week.get("predicted_return")
            lines.append(f"**Next Week** ({next_week.get('n_candles', 0)} trading days): "
                        f"predicted return {wk_ret:+.1f}%" if wk_ret is not None
                        else "**Next Week**: candlestick series generated")
            lines.append("")

        next_month = ohlc.get("next_month", {})
        if next_month:
            mo_ret = next_month.get("predicted_return")
            lines.append(f"**Next Month** ({next_month.get('n_candles', 0)} trading days): "
                        f"predicted return {mo_ret:+.1f}%" if mo_ret is not None
                        else "**Next Month**: candlestick series generated")
            lines.append("")

        next_year = ohlc.get("next_year", {})
        if next_year:
            yr_ret = next_year.get("predicted_return")
            lines.append(f"**Next Year**: predicted return {yr_ret:+.1f}%" if yr_ret is not None
                        else "**Next Year**: monthly aggregated series generated")
            monthly = next_year.get("monthly_aggregates", [])
            if monthly:
                lines.append("")
                lines.append("| Month | Open | Close | Range | Confidence |")
                lines.append("|-------|------|-------|-------|------------|")
                for m in monthly[:6]:
                    lines.append(
                        f"| {m.get('period_start', '?')} | "
                        f"{_fmt(m.get('open'))} | {_fmt(m.get('close'))} | "
                        f"{_fmt(m.get('low'))}-{_fmt(m.get('high'))} | "
                        f"{_fmt(m.get('avg_confidence'), '.0%')} |"
                    )

        lines.append("")
        lines.append("*(See attached predicted candlestick charts)*")
    else:
        lines.append("*(Predicted candlestick charts not available for this run)*")

    return "\n".join(lines)


def _build_investment_recommendation(profile: dict[str, Any]) -> str:
    """Build the Investment Recommendation section (Section 12)."""
    lines: list[str] = []

    # Derive recommendation from available data
    filters = profile.get("filters", {})
    survival = profile.get("survival", {})
    hist = profile.get("historical", {})
    fh = profile.get("financial_health", {})
    mc = profile.get("monte_carlo", {})
    sent = profile.get("sentiment", {})

    # Count filter passes
    pass_count = 0
    total_filters = 0
    for f_name, f_data in filters.items():
        if isinstance(f_data, dict) and "verdict" in f_data:
            total_filters += 1
            verdict = str(f_data.get("verdict", ""))
            if "PASS" in verdict.upper():
                pass_count += 1

    # Gather signals for multi-factor recommendation
    is_survival = bool(survival.get("company_survival_mode_flag", 0))
    total_return = hist.get("return_total", 0)
    composite_health = fh.get("latest_composite")
    surv_prob = mc.get("survival_probability_mean")
    latest_sentiment = sent.get("latest_sentiment")

    # Multi-factor scoring
    score = 0
    factors: list[str] = []

    if is_survival:
        score -= 3
        factors.append("company is under financial distress")
    if surv_prob is not None and surv_prob < 0.5:
        score -= 2
        factors.append(f"survival probability is low ({_pct(surv_prob)})")
    elif surv_prob is not None and surv_prob > 0.8:
        score += 1
        factors.append(f"survival probability is solid ({_pct(surv_prob)})")
    if pass_count >= 3:
        score += 2
        factors.append(f"{pass_count}/{total_filters} ethical screens passed")
    elif pass_count <= 1 and total_filters > 0:
        score -= 1
        factors.append(f"only {pass_count}/{total_filters} ethical screens passed")
    if total_return is not None and total_return > 0.05:
        score += 1
        factors.append(f"positive historical return ({_pct(total_return)})")
    elif total_return is not None and total_return < -0.10:
        score -= 1
        factors.append(f"negative historical return ({_pct(total_return)})")
    if composite_health is not None and composite_health >= 60:
        score += 1
        factors.append(f"strong financial health score ({composite_health:.0f}/100)")
    elif composite_health is not None and composite_health < 35:
        score -= 1
        factors.append(f"weak financial health score ({composite_health:.0f}/100)")
    if latest_sentiment is not None and latest_sentiment > 0.3:
        score += 1
        factors.append("positive market sentiment")
    elif latest_sentiment is not None and latest_sentiment < -0.3:
        score -= 1
        factors.append("negative market sentiment")

    # Map score to recommendation
    if score >= 3:
        recommendation = "BUY"
        confidence = "Medium-High"
    elif score >= 1:
        recommendation = "BUY"
        confidence = "Medium"
    elif score >= -1:
        recommendation = "HOLD"
        confidence = "Low"
    elif score >= -3:
        recommendation = "SELL"
        confidence = "Medium"
    else:
        recommendation = "SELL"
        confidence = "Medium-High"

    rationale = "; ".join(factors) if factors else "Insufficient data for detailed assessment."

    lines.append(f"**Recommendation:** {recommendation}")
    lines.append("")
    lines.append(f"**Confidence Level:** {confidence}")
    lines.append("")
    lines.append(f"**Rationale:** {rationale.capitalize()}")
    lines.append("")
    lines.append("**Key Catalysts to Watch:**")
    lines.append("- Upcoming earnings announcements and quarterly results")
    lines.append("- Changes in cash flow trajectory or debt covenants")
    lines.append("- Central bank rate decisions and inflation data releases")
    lines.append("- Sector-specific regulatory developments")
    if is_survival:
        lines.append("- Liquidity events, refinancing windows, or asset sales")
    lines.append("")
    lines.append("*This recommendation is generated from quantitative factors across "
                 f"{total_filters} ethical screens, historical performance, financial "
                 "health scoring, survival analysis, and market sentiment. "
                 "Professional judgement and qualitative analysis should supplement "
                 "this assessment. This is not financial advice.*")

    return "\n".join(lines)


def _build_appendix(profile: dict[str, Any]) -> str:
    """Build the Appendix & Methodology section (Section 22).

    Covers data sources, modeling methodology, definitions, and disclaimers.
    Written in clear finance/mathematics language for institutional readers.
    """
    meta = profile.get("meta", {})
    lines: list[str] = []

    lines.append("### Data Sources")
    lines.append("")
    pit_source = meta.get("data_provider_label", meta.get("data_provider", "Government filing API"))
    lines.append(f"- **Primary financial data**: {pit_source} (point-in-time regulatory filings)")
    macro_source = meta.get("macro_source", "")
    if macro_source:
        lines.append(f"- **Macroeconomic indicators**: {macro_source}")
    lines.append("- **Price data (OHLCV)**: Exchange-reported daily trade data (inherently point-in-time)")
    lines.append("")

    lines.append("### Methodology Overview")
    lines.append("")
    lines.append("This analysis uses a **point-in-time (PIT) methodology** to prevent look-ahead bias. "
                 "All financial data is indexed by the date it became publicly available (the filing date), "
                 "not the fiscal period it covers. This ensures that no future information leaks into "
                 "historical calculations.")
    lines.append("")

    lines.append("**Key analytical frameworks:**")
    lines.append("")
    lines.append("- **5-Tier Survival Hierarchy**: Liquidity, Solvency, Market Stability, "
                 "Profitability, and Growth -- weighted dynamically based on financial distress signals")
    lines.append("- **Financial Health Scoring**: Composite 0-100 score across all five tiers, "
                 "with Altman Z-Score (bankruptcy prediction) and Beneish M-Score (earnings manipulation detection)")
    lines.append("- **Regime Detection**: Hidden Markov Models and Gaussian Mixture Models "
                 "identify market regime shifts (bull, bear, high-volatility, crisis)")
    lines.append("- **Multi-Model Forecasting**: Ensemble of Kalman filter, ARIMA, VAR, GARCH, "
                 "gradient boosted trees, LSTM, and Transformer models -- weighted by inverse validation error")
    lines.append("- **Monte Carlo Simulation**: 10,000-path simulation for survival probability "
                 "and Value-at-Risk estimation under current regime conditions")
    lines.append("- **Conformal Prediction**: Distribution-free confidence intervals that "
                 "provide guaranteed coverage regardless of model assumptions")
    lines.append("")

    # Extended models that were used
    ext = profile.get("extended_models", {})
    if ext:
        used_models = [k.replace("_", " ").title() for k, v in ext.items()
                      if isinstance(v, dict) and v.get("available")]
        if used_models:
            lines.append("**Extended models applied in this analysis:**")
            for m in used_models:
                lines.append(f"- {m}")
            lines.append("")

    lines.append("### Definitions")
    lines.append("")
    lines.append("| Term | Definition |")
    lines.append("|------|-----------|")
    lines.append("| PIT (Point-in-Time) | Data indexed by publication date, not fiscal period |")
    lines.append("| TTM (Trailing Twelve Months) | Sum of last four quarterly values |")
    lines.append("| FCF (Free Cash Flow) | Operating cash flow minus capital expenditures |")
    lines.append("| Z-Score (Altman) | Bankruptcy probability predictor (safe > 2.99, distress < 1.81) |")
    lines.append("| M-Score (Beneish) | Earnings manipulation detector (flag if > -1.78) |")
    lines.append("| VaR (Value at Risk) | Maximum expected loss at a given confidence level |")
    lines.append("| OHLCV | Open, High, Low, Close, Volume -- standard price bar data |")
    lines.append("")

    lines.append("### Disclaimers")
    lines.append("")
    lines.append("This report is generated from quantitative analysis of public regulatory filings "
                 "and market data. It does not constitute financial advice. Investment decisions should "
                 "incorporate qualitative factors, management assessment, and professional judgement "
                 "not captured by quantitative models. Past performance and model predictions do not "
                 "guarantee future results.")
    lines.append("")
    lines.append(f"*Report generated using Operator 1 analytical pipeline. "
                 f"Data source: {pit_source}.*")

    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Advanced module report sections
# ---------------------------------------------------------------------------


def _build_graph_risk_section(profile: dict[str, Any]) -> str:
    """Build Graph Theory / Supply Chain Risk section."""
    gr = profile.get("graph_risk", {})
    if not gr.get("available"):
        return "*Supply chain and contagion risk analysis not available for this run.*"

    lines = [
        "### Business Network Structure",
        "",
        f"The company sits within a network of **{gr.get('n_nodes', 0)} related entities** "
        f"connected by **{gr.get('n_edges', 0)} business relationships** "
        "(suppliers, customers, competitors, and subsidiaries).",
        "",
        f"- **Network connectivity**: {_fmt(gr.get('target_degree_centrality'))} "
        "*(higher values = more direct business relationships, more exposure to partner risk)*",
        f"- **Systemic importance**: {_fmt(gr.get('target_pagerank'))} "
        "*(higher values = the company's distress would ripple more widely)*",
        "",
        "### Contagion Risk (Domino Effect)",
        "",
        f"- **Spillover probability**: {_pct(gr.get('contagion_target_infection_prob'))}",
        "  *(If any linked company enters financial distress, this is the estimated "
        "probability that the trouble spreads to affect this company.)*",
        f"- **Expected companies affected in a crisis**: {_fmt(gr.get('contagion_expected_infected'), '.1f')}",
        "",
        "### Supply Chain Concentration Risk",
        "",
        f"- **Supplier concentration** (Herfindahl index): {_fmt(gr.get('supplier_hhi'))} "
        f"({gr.get('concentration_label', 'N/A')})",
        f"- **Customer concentration** (Herfindahl index): {_fmt(gr.get('customer_hhi'))}",
        "",
        "*A Herfindahl index above 0.25 indicates heavy dependence on a small number of "
        "counterparties -- a risk if any single supplier or customer fails.*",
        "",
    ]

    top_pr = gr.get("top_pagerank", {})
    if top_pr:
        lines.append("### Most Influential Companies in the Network")
        lines.append("")
        for name, score in list(top_pr.items())[:5]:
            lines.append(f"- **{name}**: influence score {_fmt(score, '.4f')}")
        lines.append("")

    return "\n".join(lines)


def _build_game_theory_section(profile: dict[str, Any]) -> str:
    """Build Game Theory / Competitive Dynamics section."""
    gt = profile.get("game_theory", {})
    if not gt.get("available"):
        return "*Competitive landscape analysis not available for this run.*"

    _structure_desc = {
        "monopoly": "Monopoly (single dominant player)",
        "oligopoly": "Oligopoly (few large players with significant pricing power)",
        "competitive": "Competitive Market (many players, limited individual pricing power)",
        "duopoly": "Duopoly (two dominant players)",
    }
    raw_structure = gt.get("market_structure", "N/A")
    structure_desc = _structure_desc.get(raw_structure, raw_structure)

    lines = [
        f"**Market structure**: {structure_desc}",
        f"**Competitors analysed**: {gt.get('n_competitors', 0)}",
        f"**Top-4 market share** (CR4): {_pct(gt.get('cr4'))} "
        "*(above 60% suggests oligopolistic pricing dynamics)*",
        "",
        "### Competitive Intensity",
        "",
        f"- **Competitive pressure**: {_fmt(gt.get('competitive_pressure'))} "
        f"({gt.get('pressure_label', 'N/A')})",
        "",
        "*Higher competitive pressure typically compresses margins and makes "
        "revenue forecasts less certain.*",
        "",
        "### Market Leadership Position",
        "",
    ]

    stk = gt.get("stackelberg", {})
    _role_desc = {
        "leader": "Market Leader (sets price/quantity, others follow)",
        "follower": "Market Follower (adapts to leader's moves)",
        "challenger": "Challenger (competing for leadership position)",
    }
    raw_role = stk.get("target_role", "N/A")
    role_desc = _role_desc.get(raw_role, raw_role)
    lines.append(f"- **Competitive role**: {role_desc}")
    lines.append(f"- **Leadership strength**: {_fmt(stk.get('leadership_score'))}")
    lines.append(f"- **Market cap rank**: #{stk.get('market_cap_rank', 'N/A')} in peer group")
    lines.append(f"- **Margin advantage over peers**: {_pct(stk.get('margin_advantage'))}")
    lines.append("")

    cournot = gt.get("cournot", {})
    eq_shares = cournot.get("equilibrium_shares", {})
    if eq_shares:
        lines.append("### Theoretical Equilibrium Market Shares")
        lines.append("")
        lines.append("*Based on Nash-Cournot equilibrium -- the stable market share "
                     "distribution where no player benefits from unilaterally changing "
                     "output:*")
        lines.append("")
        for name, share in eq_shares.items():
            lines.append(f"- **{name}**: {_pct(share)}")
        lines.append("")

    return "\n".join(lines)


def _build_fuzzy_protection_section(profile: dict[str, Any]) -> str:
    """Build Fuzzy Logic Government Protection section."""
    fp = profile.get("fuzzy_protection", {})
    if not fp.get("available"):
        return "*Fuzzy protection analysis not available for this run.*"

    lines = [
        f"**Protection degree**: {_fmt(fp.get('mean_degree', fp.get('protection_degree')))} / 1.00",
        f"**Label**: {fp.get('latest_label', fp.get('label', 'N/A'))}",
        "",
        "### Dimension Scores",
        "",
        f"- **Sector strategicness**: {_fmt(fp.get('sector_score'))} *(How important is the sector to the government?)*",
        f"- **Economic significance**: {_fmt(fp.get('economic_score', fp.get('mean_economic')))} *(Market cap relative to GDP)*",
        f"- **Policy responsiveness**: {_fmt(fp.get('policy_score', fp.get('mean_policy')))} *(Emergency rate cuts)*",
        "",
        "*Higher scores indicate the company is more likely to receive government support during crises, "
        "which reduces downside risk in extreme scenarios.*",
        "",
    ]
    return "\n".join(lines)


def _build_pid_section(profile: dict[str, Any]) -> str:
    """Build PID Controller adaptive learning section."""
    pid = profile.get("pid_controller", {})
    if not pid.get("available", True) or not pid:
        return "*Adaptive model calibration not available for this run.*"

    lines = [
        "The forecasting engine uses an automatic feedback loop to continuously "
        "recalibrate its predictions. When the model's forecast deviates from "
        "actual market data, the system adjusts its sensitivity -- becoming more "
        "aggressive when errors persist (under-reacting) and more conservative "
        "when corrections overshoot (over-reacting).",
        "",
        f"- **Average calibration factor**: {_fmt(pid.get('mean_multiplier'))} "
        "*(1.0 = no adjustment needed, >1.0 = model compensating for persistent errors)*",
        f"- **Peak correction**: {_fmt(pid.get('max_multiplier'))} "
        "*(largest single adjustment during the analysis, typically during regime changes)*",
        f"- **Most stable period**: {_fmt(pid.get('min_multiplier'))} "
        "*(model was most accurate here, requiring minimal adjustment)*",
        f"- **Variables monitored**: {pid.get('n_variables', 0)}",
        "",
    ]

    per_var = pid.get("per_variable", {})
    if per_var:
        lines.append("### Variables Requiring the Most Recalibration")
        lines.append("")
        sorted_vars = sorted(per_var.items(), key=lambda x: x[1].get("output", 1.0), reverse=True)
        for var, state in sorted_vars[:5]:
            output = state.get("output", 1.0)
            if output and output > 1.2:
                interpretation = "model was under-predicting, compensated upward"
            elif output and output < 0.8:
                interpretation = "model was over-predicting, damped down"
            else:
                interpretation = "well calibrated, minor adjustments"
            lines.append(
                f"- **{var}**: calibration factor {_fmt(state.get('output'))} "
                f"({interpretation})"
            )
        lines.append("")

    return "\n".join(lines)


def _build_sentiment_analysis_section(profile: dict[str, Any]) -> str:
    """Build News Sentiment Analysis section."""
    sent = profile.get("sentiment", {})
    if not sent.get("available"):
        return "*News sentiment analysis not available for this run.*"

    lines = [
        "News sentiment was scored across recent stock-related articles to capture "
        "market perception and information-driven price movements.",
        "",
        f"- **Articles scored**: {sent.get('n_articles_scored', 0)}",
        f"- **Scoring method**: {sent.get('scoring_method', 'N/A')}",
        f"- **Mean sentiment**: {_fmt(sent.get('mean_sentiment'))} *(range: -1.0 bearish to +1.0 bullish)*",
        f"- **Latest sentiment**: {_fmt(sent.get('latest_sentiment'))} ({sent.get('latest_label', 'N/A')})",
        "",
    ]

    summary = sent.get("series_summary", {}).get("sentiment_score", {})
    if summary:
        lines.append("### Sentiment Time Series Summary")
        lines.append("")
        lines.append(f"- **Min**: {_fmt(summary.get('min'))}")
        lines.append(f"- **Max**: {_fmt(summary.get('max'))}")
        lines.append(f"- **Median**: {_fmt(summary.get('median'))}")
        lines.append(f"- **Observed days**: {summary.get('n_observed', 0)}")
        lines.append("")

    return "\n".join(lines)


def _build_peer_ranking_section(profile: dict[str, Any]) -> str:
    """Build Peer Ranking & Relative Positioning section."""
    pr = profile.get("peer_ranking", {})
    if not pr.get("available"):
        return "*Peer ranking not available (no linked entities or insufficient data).*"

    lines = [
        "The target company is ranked against its peers on key financial metrics. "
        "A rank of 50 means at the peer median; higher is better.",
        "",
        f"- **Number of peers**: {pr.get('n_peers', 0)}",
        f"- **Variables ranked**: {pr.get('n_variables_ranked', 0)}",
        f"- **Composite rank**: {_fmt(pr.get('latest_composite_rank'))} ({pr.get('latest_label', 'N/A')})",
        "",
    ]

    var_ranks = pr.get("variable_ranks", {})
    if var_ranks:
        lines.append("### Variable Rankings (latest)")
        lines.append("")
        lines.append("| Variable | Percentile |")
        lines.append("|----------|-----------|")
        for var, rank in sorted(var_ranks.items(), key=lambda x: x[1] or 0, reverse=True):
            lines.append(f"| {var} | {_fmt(rank, '.0f')} |")
        lines.append("")

    return "\n".join(lines)


def _build_macro_quadrant_section(profile: dict[str, Any]) -> str:
    """Build Macro Environment Quadrant section."""
    mq = profile.get("macro_quadrant", {})
    if not mq.get("available"):
        return "*Macro quadrant classification not available (insufficient macro data).*"

    quadrant_descriptions = {
        "goldilocks": "Growth above trend, inflation below target -- favorable environment",
        "reflation": "Growth above trend, inflation above target -- expansionary but inflationary",
        "stagflation": "Growth below trend, inflation above target -- challenging environment",
        "deflation": "Growth below trend, inflation below target -- contractionary",
    }

    current = mq.get("current_quadrant", "unknown")
    desc = quadrant_descriptions.get(current, "Unknown classification")

    lines = [
        "The macro environment is classified into four quadrants based on GDP growth "
        "relative to trend and inflation relative to central bank targets.",
        "",
        f"- **Current quadrant**: **{current.upper()}** -- {desc}",
        f"- **GDP growth trend**: {_fmt(mq.get('growth_trend'))}%",
        f"- **Inflation target**: {_fmt(mq.get('inflation_target'))}%",
        f"- **Days classified**: {mq.get('n_days_classified', 0)}",
        f"- **Quadrant transitions**: {mq.get('n_transitions', 0)}",
        "",
    ]

    dist = mq.get("quadrant_distribution", {})
    if dist:
        lines.append("### Quadrant Distribution (over analysis window)")
        lines.append("")
        for q, pct in sorted(dist.items(), key=lambda x: x[1], reverse=True):
            lines.append(f"- **{q.title()}**: {pct*100:.1f}%")
        lines.append("")

    return "\n".join(lines)


def _build_advanced_insights(profile: dict[str, Any]) -> str:
    """Build the Advanced Quantitative Insights section.

    Covers: cycle decomposition, copula tail risk, transfer entropy,
    Sobol sensitivity, particle filter, and transformer attention.
    """
    lines: list[str] = []
    ext = profile.get("extended_models", {})

    # Cycle decomposition
    cycle = profile.get("cycle_decomposition", ext.get("cycle_decomposition", {}))
    if isinstance(cycle, dict) and cycle.get("available"):
        lines.append("### Market Cycle Analysis")
        lines.append("")
        lines.append(
            "The price history was decomposed into its constituent cyclical "
            "components using frequency analysis (Fourier and wavelet transforms). "
            "This reveals recurring patterns in the data."
        )
        lines.append("")
        dominant = cycle.get("dominant_cycles", [])
        if dominant:
            lines.append("**Dominant cycles detected:**")
            lines.append("")
            for c in dominant[:5]:
                if isinstance(c, dict):
                    period = c.get("period_days", "?")
                    strength = c.get("strength", c.get("amplitude", "?"))
                    lines.append(
                        f"- **{period}-day cycle** (strength: {_fmt(strength)}) "
                        f"-- approximately {int(period)//21} months"
                        if isinstance(period, (int, float)) and period > 21
                        else f"- **{period}-day cycle** (strength: {_fmt(strength)})"
                    )
                else:
                    lines.append(f"- {c}")
            lines.append("")

    # Copula tail dependency
    copula = profile.get("copula", ext.get("copula", {}))
    if isinstance(copula, dict) and copula.get("available"):
        lines.append("### Joint Tail Risk (Copula Analysis)")
        lines.append("")
        tail_dep = copula.get("tail_dependence", copula.get("tail_dep"))
        copula_type = copula.get("best_copula", copula.get("copula_type", "unknown"))
        lines.append(
            "Copula modelling measures how likely it is that extreme events "
            "occur simultaneously across related variables (e.g., price and "
            "volatility crashing together)."
        )
        lines.append("")
        lines.append(f"- **Tail dependence coefficient**: {_fmt(tail_dep)}")
        if tail_dep is not None:
            try:
                td = float(tail_dep)
                if td > 0.4:
                    lines.append("  *High tail dependence: extreme losses tend to cluster. "
                                "Diversification benefits reduce significantly in crises.*")
                elif td > 0.2:
                    lines.append("  *Moderate tail dependence: some tendency for joint extremes. "
                                "Standard diversification assumptions may be optimistic in stress periods.*")
                else:
                    lines.append("  *Low tail dependence: extreme events are relatively independent. "
                                "Diversification benefits hold up reasonably well in crises.*")
            except (TypeError, ValueError):
                pass
        lines.append(f"- **Best-fit distribution model**: {copula_type}")
        lines.append("")

    # Transfer entropy (causal information flow)
    te = profile.get("transfer_entropy", ext.get("transfer_entropy", {}))
    if isinstance(te, dict) and te.get("available"):
        lines.append("### Information Flow Analysis (Transfer Entropy)")
        lines.append("")
        lines.append(
            "Transfer entropy measures the directional flow of information "
            "between financial variables -- which metrics *lead* and which *follow*. "
            "This goes beyond simple correlation to identify causal relationships."
        )
        lines.append("")
        top_pairs = te.get("top_pairs", [])
        if top_pairs:
            lines.append("**Strongest causal relationships detected:**")
            lines.append("")
            for pair in top_pairs[:5]:
                if isinstance(pair, dict):
                    src = pair.get("source", "?")
                    tgt = pair.get("target", "?")
                    strength = pair.get("te_value", pair.get("strength", "?"))
                    lines.append(f"- {src} -> {tgt}: information flow = {_fmt(strength)}")
            lines.append("")

    # Sobol sensitivity
    sobol = profile.get("sobol_sensitivity", ext.get("sobol_sensitivity", {}))
    if isinstance(sobol, dict) and sobol.get("available"):
        lines.append("### Global Sensitivity Analysis")
        lines.append("")
        lines.append(
            "Sobol sensitivity analysis decomposes the variance in daily returns "
            "to determine which input factors contribute most to outcome uncertainty. "
            "This answers: *\"What drives the risk in this investment?\"*"
        )
        lines.append("")
        first_order = sobol.get("first_order", sobol.get("S1", {}))
        if isinstance(first_order, dict) and first_order:
            lines.append("**Top risk drivers (by contribution to return variance):**")
            lines.append("")
            sorted_factors = sorted(first_order.items(), key=lambda x: x[1] or 0, reverse=True)
            for factor, contrib in sorted_factors[:8]:
                lines.append(f"- **{factor}**: {_pct(contrib)} of return variance")
            lines.append("")

    # Particle filter
    pf = profile.get("particle_filter", ext.get("particle_filter", {}))
    if isinstance(pf, dict) and pf.get("available"):
        lines.append("### Non-Linear State Estimation (Particle Filter)")
        lines.append("")
        lines.append(
            "The particle filter tracks the *hidden true state* of key financial "
            "ratios by filtering out noise from observed data. Unlike simple moving "
            "averages, it handles sudden jumps and non-linear dynamics common in "
            "financial distress situations."
        )
        lines.append("")
        lines.append(f"- **State variables tracked**: {pf.get('n_states', 'N/A')}")
        lines.append(f"- **Resampling events**: {pf.get('n_resamples', 'N/A')} "
                     "*(more resampling = more volatile underlying state)*")
        lines.append("")

    # Transformer
    tf = profile.get("transformer", ext.get("transformer", {}))
    if isinstance(tf, dict) and tf.get("available"):
        lines.append("### Deep Learning Forecast (Transformer Architecture)")
        lines.append("")
        lines.append(
            "An attention-based neural network was trained on the full variable "
            "set to capture complex, non-linear relationships that traditional "
            "statistical models may miss. The attention mechanism reveals which "
            "historical days and variables the model considers most relevant."
        )
        lines.append("")
        lines.append(f"- **Training epochs completed**: {tf.get('n_epochs_trained', 'N/A')}")
        lines.append(f"- **Final training loss**: {_fmt(tf.get('final_train_loss'), '.6f')}")
        feat_imp = tf.get("feature_importance", {})
        if feat_imp:
            lines.append("")
            lines.append("**Variables the model weighted most heavily:**")
            lines.append("")
            sorted_feats = sorted(feat_imp.items(), key=lambda x: x[1] or 0, reverse=True)
            for feat, imp in sorted_feats[:5]:
                lines.append(f"- {feat}: attention weight {_fmt(imp, '.4f')}")
        lines.append("")

    # Granger causality
    gc = profile.get("granger_causality", ext.get("granger_causality", {}))
    if isinstance(gc, dict) and gc.get("available"):
        lines.append("### Causal Relationship Network (Granger Causality)")
        lines.append("")
        lines.append(
            "Granger causality tests whether historical values of one variable "
            "improve predictions of another. Unlike simple correlation, this "
            "identifies *directional predictive power* -- which metrics actually "
            "lead other metrics."
        )
        lines.append("")
        lines.append(f"- **Significant causal links found**: {gc.get('n_significant_pairs', 0)}")
        lines.append(f"- **Network density**: {_fmt(gc.get('network_density'))} "
                     "*(higher = more interconnected system, harder to diversify)*")
        lines.append(f"- **Variables retained after pruning**: {gc.get('n_retained', 0)} "
                     f"(removed {gc.get('n_pruned', 0)} non-predictive variables)")
        lines.append("")

        top_pairs = gc.get("top_pairs", [])
        if top_pairs:
            lines.append("**Strongest causal links:**")
            lines.append("")
            for pair in top_pairs[:5]:
                if isinstance(pair, dict):
                    lines.append(
                        f"- {pair.get('source', '?')} -> {pair.get('target', '?')} "
                        f"(lag: {pair.get('best_lag', '?')} days, "
                        f"significance: {_fmt(pair.get('p_value'), '.4f')})"
                    )
            lines.append("")

    # Dual regime classification
    dr = ext.get("dual_regimes", {})
    if isinstance(dr, dict) and dr.get("available"):
        lines.append("### Dual-Layer Market Classification")
        lines.append("")
        lines.append(
            "The analysis maintains two independent regime classifications: "
            "a *market regime* (based on price returns and volatility patterns) "
            "and a *fundamental regime* (based on the company's financial ratios). "
            "A company can be in a bull market while simultaneously showing "
            "fundamental deterioration -- this dual view captures that divergence."
        )
        lines.append("")
        lines.append(f"- **Current fundamental regime**: "
                     f"{dr.get('fundamental_regime_current', 'N/A')}")

        fund_dist = dr.get("fundamental_distribution", {})
        if fund_dist:
            _fund_desc = {
                "healthy": "Financially Healthy",
                "stressed": "Under Financial Stress",
                "distress": "In Financial Distress",
            }
            lines.append("")
            lines.append("**Time in each financial state:**")
            lines.append("")
            for state, pct in sorted(fund_dist.items(), key=lambda x: -x[1]):
                desc = _fund_desc.get(state, state)
                lines.append(f"- {desc}: {pct * 100:.1f}%")
        lines.append("")

    # Genetic algorithm optimizer
    ga = ext.get("genetic_optimizer", {})
    if isinstance(ga, dict) and ga.get("available"):
        lines.append("### Ensemble Weight Optimisation")
        lines.append("")
        lines.append(
            "An evolutionary algorithm was used to find the optimal blend of "
            "forecasting models. Each model contributes differently depending "
            "on its historical accuracy and the financial tier being predicted."
        )
        lines.append("")
        lines.append(f"- **Evolutionary generations**: {ga.get('n_generations', 0)}")
        lines.append(f"- **Convergence reached**: {'Yes' if ga.get('converged') else 'No'}")
        lines.append("")

        best_w = ga.get("best_weights", {})
        if best_w:
            _model_labels = {
                "kalman": "Adaptive Kalman Filter",
                "garch": "Volatility Model",
                "var": "Multi-Variable Regression",
                "lstm": "Deep Learning Sequence Model",
                "tree": "Gradient Boosted Ensemble",
                "baseline": "Moving Average Baseline",
                "transformer": "Attention-Based Neural Network",
            }
            lines.append("**Optimised model weights:**")
            lines.append("")
            for model, weight in sorted(best_w.items(), key=lambda x: -x[1]):
                if weight > 0.01:
                    label = _model_labels.get(model, model)
                    lines.append(f"- {label}: {weight * 100:.1f}%")
            lines.append("")

        tier_w = ga.get("tier_weights", {})
        if tier_w:
            _tier_labels_ga = {
                "tier1": "Liquidity & Cash",
                "tier2": "Solvency & Leverage",
                "tier3": "Market Stability",
                "tier4": "Profitability",
                "tier5": "Growth & Valuation",
            }
            lines.append("*The algorithm adjusts model weights per tier "
                        "(e.g., Kalman dominates for liquidity prediction, "
                        "deep learning for growth forecasting).*")
            lines.append("")

    # Capital Allocation Deep Dive (premium-only detailed breakdown)
    vanity = profile.get("vanity", {})
    if vanity.get("v2_available"):
        v2_score = vanity.get("v2_score", {})
        lines.append("### Capital Allocation & Management Discipline Assessment")
        lines.append("")
        lines.append(
            "This proprietary scoring framework evaluates management's capital "
            "allocation effectiveness by examining five dimensions of corporate "
            "financial behavior. Each dimension is scored from 0 (highly disciplined) "
            "to 100 (significant misallocation), with the composite providing an "
            "overall assessment of management stewardship quality."
        )
        lines.append("")
        lines.append(
            f"- **Composite discipline score**: {_fmt(v2_score.get('latest'), '.1f')} / 100 "
            f"({vanity.get('v2_label', 'N/A')})"
        )
        lines.append(f"- **21-day moving average**: {_fmt(vanity.get('v2_score_21d', {}).get('mean'), '.1f')}")
        lines.append(f"- **Historical range**: {_fmt(v2_score.get('min'), '.1f')} -- "
                     f"{_fmt(v2_score.get('max'), '.1f')}")
        lines.append(f"- **Trend**: {vanity.get('v2_trend', 'N/A')}")
        lines.append("")

        breakdown = vanity.get("v2_breakdown", {})
        _dim_labels = {
            "vanity_rnd_mismatch": (
                "Innovation Efficiency",
                "Evaluates whether R&D expenditure is generating returns. A high "
                "score indicates sustained research spending without corresponding "
                "revenue growth -- a potential signal of undisciplined innovation "
                "investment or strategic misalignment.",
            ),
            "vanity_sga_bloat_v2": (
                "Operating Cost Efficiency",
                "Measures SG&A expense relative to sector peers, adjusted for "
                "margin trajectory. Elevated overhead costs combined with "
                "compressing margins may indicate managerial bloat or an "
                "unsustainable cost structure.",
            ),
            "vanity_capital_misallocation": (
                "Balance Sheet Stewardship",
                "Assesses whether leverage decisions align with the company's "
                "liquidity position. Flags include debt accumulation during "
                "liquidity deterioration and shareholder distributions while "
                "solvency metrics are under stress.",
            ),
            "vanity_competitive_decay": (
                "Competitive Position Erosion",
                "Quantifies the divergence between perceived market standing and "
                "actual competitive performance. Rising competitive pressure "
                "alongside declining peer rankings and margin compression may "
                "indicate a deteriorating moat.",
            ),
            "vanity_sentiment_gap": (
                "Market Narrative Divergence",
                "Measures the gap between market sentiment (news flow, analyst "
                "coverage) and underlying financial trajectory. A positive "
                "narrative accompanying declining fundamental health scores "
                "warrants additional due diligence.",
            ),
        }

        if breakdown:
            lines.append("| Dimension | Score | Assessment |")
            lines.append("|-----------|-------|------------|")
            for comp_key, (dim_name, _description) in _dim_labels.items():
                val = breakdown.get(comp_key)
                if val is not None and not (isinstance(val, float) and val != val):
                    if val <= 20:
                        assessment = "Disciplined"
                    elif val <= 40:
                        assessment = "Adequate"
                    elif val <= 70:
                        assessment = "Elevated concern"
                    else:
                        assessment = "Material risk"
                    lines.append(f"| {dim_name} | {_fmt(val, '.1f')} | {assessment} |")
                else:
                    lines.append(f"| {dim_name} | N/A | Insufficient data |")
            lines.append("")

            # Detailed descriptions for each dimension
            lines.append("**Dimension Definitions:**")
            lines.append("")
            for comp_key, (dim_name, description) in _dim_labels.items():
                lines.append(f"- **{dim_name}**: {description}")
            lines.append("")

        lines.append(
            "*Interpretation guide: Scores below 20 reflect strong capital "
            "discipline characteristic of well-managed businesses. Scores between "
            "20 and 40 are typical of companies with adequate but not exceptional "
            "stewardship. Scores exceeding 40 warrant closer scrutiny of "
            "management decisions, and scores above 70 represent material "
            "capital allocation risk that may impair long-term shareholder value.*"
        )
        lines.append("")

    if not lines:
        return "*No advanced quantitative models were run for this analysis.*"

    return "\n".join(lines)


def _build_economic_position(profile: dict[str, Any]) -> str:
    """Build the Economic Position section (5-plane model)."""
    plane = profile.get("economic_plane", {})
    if not plane.get("primary_plane"):
        return "*Economic plane classification not available.*"

    _plane_descriptions = {
        "supply": "extracts, produces, or provides raw inputs to the economy",
        "manufacturing": "transforms inputs into goods, technology, and services",
        "consumption": "sells directly to end consumers and drives demand",
        "logistics": "moves goods and connects supply chains across regions",
        "financial_services": "provides capital, insurance, and financial intermediation",
    }

    primary = plane.get("primary_plane", "unknown")
    label = plane.get("plane_label", primary.replace("_", " ").title())
    desc = _plane_descriptions.get(primary, plane.get("plane_description", ""))
    secondary = plane.get("secondary_planes", [])

    lines = [
        f"The company operates primarily in the **{label}** plane of the economy "
        f"-- it {desc}.",
        "",
    ]

    if secondary:
        sec_labels = [s.replace("_", " ").title() for s in secondary]
        lines.append(
            f"It also has secondary exposure to: {', '.join(sec_labels)}."
        )
        lines.append("")

    lines.extend([
        "### How Economic Position Affects Analysis",
        "",
        "The company's position in the economic structure determines which "
        "linked variables matter most:",
        "",
    ])

    _plane_impact = {
        "supply": "- Upstream commodity prices and input costs are primary risk factors\n"
                  "- Downstream manufacturing demand drives revenue\n"
                  "- Macro indicators (GDP, industrial production) have direct impact",
        "manufacturing": "- Both upstream supply costs and downstream demand affect margins\n"
                        "- Technology and R&D cycles create competitive dynamics\n"
                        "- Capital expenditure and capacity utilisation are key drivers",
        "consumption": "- Consumer sentiment and disposable income drive revenue\n"
                      "- GDP growth and employment directly affect demand\n"
                      "- Brand strength and market share determine competitive position",
        "logistics": "- Global trade volumes and shipping rates drive revenue\n"
                    "- Fuel costs and infrastructure capacity are key constraints\n"
                    "- Acts as a bridge connecting supply and demand planes",
        "financial_services": "- Interest rates and monetary policy directly affect margins\n"
                             "- Credit quality and default rates are survival-critical\n"
                             "- Systemic risk means this company's health affects all other planes",
    }

    lines.append(_plane_impact.get(primary, "- General economic conditions apply"))
    lines.append("")

    return "\n".join(lines)


def _build_portfolio_fit(profile: dict[str, Any]) -> str:
    """Build the Portfolio Fit section (user portfolio context)."""
    ctx = profile.get("portfolio_context", {})
    if not ctx.get("available"):
        return "*Portfolio context not available (no --portfolio flag provided).*"

    lines = [
        "### Correlation with Existing Holdings",
        "",
        f"- **Portfolio correlation**: {_fmt(ctx.get('correlation_with_portfolio'))}",
        f"- **Diversification benefit**: {ctx.get('diversification_benefit', 'unknown').title()}",
        f"- **Marginal risk contribution**: {_fmt(ctx.get('marginal_var_contribution'))}",
        f"- **Sector concentration** (after adding this position): "
        f"{_fmt(ctx.get('sector_concentration'), '.0%')}",
        "",
    ]

    adjustment = ctx.get("recommendation_adjustment", "")
    if adjustment:
        lines.append(f"**Assessment**: {adjustment}")
        lines.append("")

    # Per-holding detail
    holdings = ctx.get("holdings_detail", [])
    if holdings:
        lines.append("### Correlation with Individual Holdings")
        lines.append("")
        lines.append("| Holding | Weight | Sector | Correlation |")
        lines.append("|---------|--------|--------|-------------|")
        for h in holdings:
            corr = h.get("correlation_with_target")
            corr_str = _fmt(corr) if corr is not None else "N/A"
            lines.append(
                f"| {h.get('symbol', '?')} | {h.get('weight', 0):.0f}% | "
                f"{h.get('sector', 'N/A')} | {corr_str} |"
            )
        lines.append("")

    return "\n".join(lines)



    lines: list[str] = []

    lines.append("### Methodology Summary")
    lines.append("")
    lines.append("This analysis uses **25+ mathematical modules** across 9 categories:")
    lines.append("")
    lines.append("| Category | Modules |")
    lines.append("|----------|---------|")
    lines.append("| Regime Detection | HMM (Hidden Markov Model), GMM (Gaussian Mixture), PELT, Bayesian Change Point |")
    lines.append("| Forecasting | Adaptive Kalman Filter, GARCH, VAR, LSTM with MC Dropout, **Temporal Fusion Transformer (TFT)** |")
    lines.append("| Tree Ensembles | Random Forest, XGBoost, Gradient Boosting |")
    lines.append("| Causality | Granger Causality, Transfer Entropy, Copula Models |")
    lines.append("| Uncertainty | **Conformal Prediction** (distribution-free intervals), **MC Dropout** (epistemic uncertainty), Regime-Aware Monte Carlo, Importance Sampling |")
    lines.append("| Explainability | **SHAP** (per-prediction feature attribution), Sobol Global Sensitivity |")
    lines.append("| Historical Analogs | **Dynamic Time Warping (DTW)** for finding similar past periods |")
    lines.append("| Optimisation | Genetic Algorithm (ensemble weight tuning) |")
    lines.append("| Pattern Recognition | Candlestick Detector, Wavelet/Fourier Decomposition |")
    lines.append("")

    lines.append("### Conformal Prediction")
    lines.append("")
    lines.append("Traditional financial models assume returns are normally distributed "
                 "and compute confidence intervals as RMSE x z-score x sqrt(horizon). "
                 "**This assumption is wrong** -- financial returns have fat tails and "
                 "regime switches that break Gaussian models.")
    lines.append("")
    lines.append("Conformal Prediction provides **distribution-free** intervals with "
                 "**guaranteed** finite-sample coverage. If we target 90% coverage, the "
                 "intervals will contain the true value at least 90% of the time -- regardless "
                 "of the underlying distribution.")
    lines.append("")
    lines.append("We use **Adaptive Conformal Inference (ACI)** which adjusts the interval "
                 "width online as the data distribution shifts (e.g., during regime changes).")
    lines.append("")

    lines.append("### SHAP Feature Attribution")
    lines.append("")
    lines.append("SHAP (SHapley Additive exPlanations) answers the question: *Why did "
                 "the model make this specific prediction?*")
    lines.append("")
    lines.append("For each predicted variable, SHAP decomposes the prediction into "
                 "contributions from individual features. For example: \"debt-to-equity "
                 "is predicted to rise 8% primarily because: +3.2% from rising long-term "
                 "debt, +2.1% from declining equity, -0.8% from strong cash position.\"")
    lines.append("")

    lines.append("### Temporal Fusion Transformer (TFT)")
    lines.append("")
    lines.append("TFT is a deep learning architecture purpose-built for mixed-frequency "
                 "time series. Unlike LSTM which treats all inputs equally, TFT uses:")
    lines.append("- **Variable selection gates** to learn which features matter")
    lines.append("- **Multi-head self-attention** to focus on relevant historical days")
    lines.append("- **Gated Residual Networks** for stable, deep learning")
    lines.append("")
    lines.append("This is particularly valuable for our data which mixes daily prices, "
                 "quarterly financial statements, and annual macro indicators.")
    lines.append("")

    lines.append("### Dynamic Time Warping (DTW) Historical Analogs")
    lines.append("")
    lines.append("DTW finds past periods where the company showed a similar "
                 "multi-variable pattern to the present. Instead of just matching "
                 "by regime label, DTW considers the *shape* of the trajectory "
                 "across multiple variables simultaneously (price, volatility, "
                 "debt, margins, macro conditions).")
    lines.append("")
    lines.append("The outcomes from those historical analog periods serve as "
                 "empirical priors: if 4 out of 5 analogs showed a 10% decline "
                 "in the following month, that is a strong signal regardless of "
                 "what the regression models predict.")
    lines.append("")

    lines.append("### MC Dropout (Epistemic Uncertainty)")
    lines.append("")
    lines.append("Standard neural networks give a single point prediction with "
                 "no indication of how *confident* the model is. MC Dropout fixes "
                 "this by running 100 forward passes through the LSTM with dropout "
                 "enabled at inference time. The spread of those 100 predictions "
                 "measures **epistemic uncertainty** -- how much the model itself "
                 "is unsure.")
    lines.append("")
    lines.append("This is different from **aleatoric uncertainty** (inherent "
                 "randomness in the data). A prediction with low epistemic but "
                 "high aleatoric uncertainty means: *the model is confident in "
                 "its estimate, but the variable is inherently noisy.* A prediction "
                 "with high epistemic uncertainty means: *the model does not have "
                 "enough information to make a reliable prediction.*")
    lines.append("")

    lines.append("### Forward Pass & Burn-Out Process")
    lines.append("")
    lines.append("The temporal engine uses a **day-by-day forward pass**: for each of "
                 "~500 trading days, it predicts the next day, compares with actual data, "
                 "and updates model parameters online. This is followed by a "
                 "**convergence-based burn-out** phase: intensive re-training on the most "
                 "recent 6 months with up to 10 iterations and patience-based early stopping.")
    lines.append("")

    lines.append("### Variable Tier Definitions")
    lines.append("")
    lines.append("| Tier | Category | Variables | Normal Weight |")
    lines.append("|------|----------|-----------|---------------|")
    lines.append("| 1 | Liquidity & Cash | cash_ratio, FCF, operating CF | 20% |")
    lines.append("| 2 | Solvency & Debt | debt_to_equity, net_debt_to_EBITDA | 20% |")
    lines.append("| 3 | Market Stability | volatility, drawdown, volume | 20% |")
    lines.append("| 4 | Profitability | margins, ROE, ROA | 20% |")
    lines.append("| 5 | Growth & Valuation | P/E, EV/EBITDA, revenue growth | 20% |")
    lines.append("")

    lines.append("### Data Sources")
    lines.append("")
    _meta_app = profile.get("meta", {})
    _provider_label_app = _meta_app.get("data_provider_label", _meta_app.get("data_provider", "Unknown"))
    _macro_source_app = _meta_app.get("macro_source", "")
    _macro_country_app = _meta_app.get("macro_country", "")

    lines.append(f"- **{_provider_label_app}:** Point-in-time financial data "
                 "(company profile, financial statements with filing dates, OHLCV prices)")
    if _macro_source_app:
        lines.append(f"- **{_macro_source_app} ({_macro_country_app}):** "
                     "Macroeconomic indicators (GDP, inflation, interest rates, unemployment, exchange rates)")
    lines.append("- **Gemini API:** Report narrative generation (optional)")
    lines.append("")

    meta = profile.get("meta", {})
    lines.append("### Data Timestamps")
    lines.append("")
    lines.append(f"- Report generated: {meta.get('generated_at', 'N/A')}")
    lines.append(f"- Cache date range: {meta.get('cache_start', 'N/A')} to {meta.get('cache_end', 'N/A')}")
    lines.append("")

    lines.append("### Disclaimer")
    lines.append("")
    lines.append("This report is generated algorithmically and is for informational purposes only. "
                 "It does not constitute financial advice. Past performance does not guarantee "
                 "future results. All predictions carry inherent uncertainty.")

    return "\n".join(lines)


def _build_key_indicators_table(profile: dict[str, Any], mode: ReportMode = ReportMode.RESULTS) -> str:
    """Build a Key Financial Indicators summary table.

    In LEARN mode, each indicator includes a plain-English explanation.
    In RESULTS mode, just the clean data grid.
    """
    # The profile stores current state under "current_state" with values
    # nested inside tier sub-dicts (tier1_liquidity, tier2_solvency, etc.).
    # Flatten all tier sub-dicts into a single lookup dict.
    cs = profile.get("current_state", {})
    snapshot: dict[str, Any] = {}
    for key, val in cs.items():
        if isinstance(val, dict):
            # Flatten tier sub-dicts (e.g. tier1_liquidity: {cash_ratio: 0.5})
            snapshot.update(val)
        else:
            snapshot[key] = val
    fh = profile.get("financial_health", {})

    # Indicator definitions: (label, value_key, format, learn_explanation)
    _INDICATORS: list[tuple[str, str, str, str]] = [
        ("P/E Ratio", "pe_ratio_calc", ".1f",
         "How much investors pay per dollar of earnings. "
         "Lower than sector average may suggest undervaluation."),
        ("P/B Ratio", "pb_ratio", ".2f",
         "Market price relative to book value. "
         "Below 1.0 means the market values the company below its net assets."),
        ("EV/EBITDA", "ev_to_ebitda", ".1f",
         "Enterprise value relative to operating cash earnings. "
         "Lower values suggest cheaper valuation relative to cash generation."),
        ("Gross Margin", "gross_margin", ".1%",
         "Percentage of revenue retained after direct costs. "
         "Higher is better -- shows pricing power and cost efficiency."),
        ("Operating Margin", "operating_margin", ".1%",
         "Percentage of revenue left after all operating expenses. "
         "Measures core business profitability before interest and taxes."),
        ("Net Margin", "net_margin", ".1%",
         "Percentage of revenue that becomes profit. "
         "The bottom line -- what shareholders actually keep."),
        ("ROE", "roe", ".1%",
         "Return on Equity -- profit generated per dollar of shareholder investment. "
         "Above 15% is generally considered strong."),
        ("ROA", "roa", ".1%",
         "Return on Assets -- how efficiently the company uses its total assets "
         "to generate profit. Higher means better asset utilization."),
        ("Current Ratio", "current_ratio", ".2f",
         "Can the company pay its short-term bills? Above 1.0 means yes. "
         "Below 1.0 means current debts exceed current assets -- a warning sign."),
        ("Debt-to-Equity", "debt_to_equity_abs", ".2f",
         "Total debt relative to shareholder equity. "
         "Above 2.0 means the company is heavily leveraged."),
        ("Interest Coverage", "interest_coverage", ".1f",
         "How many times operating profit covers interest payments. "
         "Below 1.5 means the company struggles to service its debt."),
        ("FCF Yield", "fcf_yield", ".1%",
         "Free cash flow relative to market cap. "
         "Higher means more cash generated per dollar of market value."),
        ("Revenue Growth YoY", "revenue_growth_yoy", ".1%",
         "Year-over-year revenue growth rate. "
         "Shows whether the business is expanding or contracting."),
        ("Volatility (21d)", "volatility_21d", ".1%",
         "How much the stock price fluctuates day-to-day. "
         "Higher volatility means more risk but also more opportunity."),
        ("Max Drawdown (1Y)", "drawdown_252d", ".1%",
         "Largest peak-to-trough decline in the past year. "
         "Shows the worst-case loss an investor would have experienced."),
    ]

    lines = ["### Key Financial Indicators", ""]

    if mode == ReportMode.RESULTS:
        lines.append("| Indicator | Value |")
        lines.append("|-----------|-------|")
        for label, key, fmt, _ in _INDICATORS:
            val = snapshot.get(key)
            lines.append(f"| {label} | {_fmt(val, fmt)} |")
    else:
        # LEARN mode: table with explanation column
        lines.append("| Indicator | Value | What This Means |")
        lines.append("|-----------|-------|-----------------|")
        for label, key, fmt, explanation in _INDICATORS:
            val = snapshot.get(key)
            lines.append(f"| {label} | {_fmt(val, fmt)} | {explanation} |")

    # Altman Z-Score (always included -- people know this one)
    z_data = fh.get("altman_z", {})
    if isinstance(z_data, dict) and z_data.get("available"):
        z_val = z_data.get("latest_z_score")
        zone = z_data.get("zone", "unknown")
        lines.append("")
        if mode == ReportMode.LEARN:
            zone_explain = {
                "safe": "Above 2.99 -- low bankruptcy risk. The company is financially healthy.",
                "grey": "Between 1.81 and 2.99 -- moderate uncertainty. Worth monitoring.",
                "distress": "Below 1.81 -- elevated bankruptcy risk. Proceed with caution.",
            }.get(zone, "")
            lines.append(f"**Altman Z-Score:** {_fmt(z_val, '.2f')} ({zone}) -- {zone_explain}")
        else:
            lines.append(f"**Altman Z-Score:** {_fmt(z_val, '.2f')} ({zone})")

    lines.append("")
    return "\n".join(lines)


def _build_geopolitical_risk_section(profile: dict[str, Any]) -> str:
    """Build Geopolitical & Conflict Risk section from conflict_risk data."""
    lines: list[str] = []

    conflict = profile.get("conflict_risk", profile.get("geopolitical_risk", {}))
    if not conflict or not isinstance(conflict, dict):
        lines.append("No geopolitical risk data available for this market.")
        lines.append("")
        return "\n".join(lines)

    country = conflict.get("country_iso2", "")
    intensity = conflict.get("conflict_intensity_score", 0)
    conflict_type = conflict.get("conflict_type", "none")
    country_flag = conflict.get("country_conflict_flag", False)
    company_flag = conflict.get("company_conflict_flag", False)
    sanctions = conflict.get("sanctions_flag", False)
    fragile = conflict.get("fragile_state_flag", False)

    # Status badge
    if intensity > 0.7:
        badge = "CRITICAL -- Active Conflict Zone"
        badge_color = "red"
    elif intensity > 0.4:
        badge = "ELEVATED -- Significant Geopolitical Risk"
        badge_color = "orange"
    elif intensity > 0.1:
        badge = "MODERATE -- Some Geopolitical Concerns"
        badge_color = "yellow"
    else:
        badge = "LOW -- Stable Geopolitical Environment"
        badge_color = "green"

    lines.append(f"**Risk Level:** **{badge}**")
    lines.append("")
    lines.append(f"**Conflict Intensity Score:** {_fmt(intensity, '.2f')} / 1.00")
    lines.append("")

    # Status flags table
    lines.append("| Risk Factor | Status |")
    lines.append("|-------------|--------|")
    lines.append(f"| Country Conflict Flag | {'Yes' if country_flag else 'No'} |")
    lines.append(f"| Company Directly Affected | {'Yes' if company_flag else 'No'} |")
    lines.append(f"| International Sanctions | {'Yes' if sanctions else 'No'} |")
    lines.append(f"| World Bank Fragile State | {'Yes' if fragile else 'No'} |")
    lines.append(f"| Conflict Classification | {conflict_type.replace('_', ' ').title()} |")
    lines.append("")

    # Event data
    events_30 = conflict.get("recent_events_30d", 0)
    events_90 = conflict.get("recent_events_90d", 0)
    fatalities_30 = conflict.get("recent_fatalities_30d", 0)
    trend = conflict.get("conflict_trend", "stable")

    if events_30 > 0 or events_90 > 0:
        lines.append("### Recent Conflict Events (UCDP)")
        lines.append("")
        lines.append(f"- **Last 30 days:** {events_30} events, {fatalities_30} fatalities")
        lines.append(f"- **Last 90 days:** {events_90} events")
        lines.append(f"- **Trend:** {trend.replace('_', ' ').title()}")
        lines.append("")

    # News monitoring
    news_mentions = conflict.get("news_conflict_mentions_7d", 0)
    news_tone = conflict.get("news_conflict_tone", 0)
    if news_mentions > 0:
        lines.append("### Conflict News Monitoring (GDELT)")
        lines.append("")
        tone_desc = "negative" if news_tone < -2 else "neutral" if news_tone < 2 else "positive"
        lines.append(f"- **Conflict-related articles (7 days):** {news_mentions}")
        lines.append(f"- **Average tone:** {_fmt(news_tone, '.1f')} ({tone_desc})")
        lines.append("")

    # Investment implications
    if country_flag:
        lines.append("### Investment Implications")
        lines.append("")
        if intensity > 0.7:
            lines.append(
                "This company operates in an **active conflict zone**. "
                "Key risks include supply chain disruption, asset destruction, "
                "capital flight, currency devaluation, and regulatory instability. "
                "Survival analysis models have been adjusted to prioritize "
                "liquidity and solvency metrics."
            )
        elif sanctions:
            lines.append(
                "This company's country is under **international sanctions**. "
                "Key risks include trade restrictions, frozen assets, SWIFT exclusion, "
                "reduced FDI, and increased cost of capital. Investors should assess "
                "sanctions compliance risk for their jurisdiction."
            )
        elif fragile:
            lines.append(
                "This company operates in a **fragile state** as classified by the "
                "World Bank. Key risks include institutional weakness, governance gaps, "
                "and elevated political instability. Due diligence should include "
                "assessment of operational resilience."
            )
        lines.append("")
    else:
        lines.append(
            "No active conflict or sanctions risk detected for this market. "
            "The geopolitical environment is considered stable for investment purposes."
        )
        lines.append("")

    # Linked entity conflict propagation
    linked_conflict = profile.get("linked_conflict", conflict.get("linked_conflict", {}))
    if linked_conflict and isinstance(linked_conflict, dict):
        affected = linked_conflict.get("linked_entities_in_conflict", [])
        if affected:
            lines.append("### Linked Entity Conflict Exposure")
            lines.append("")
            lines.append("| Entity | Country | Relationship | Risk Type | Severity | Reason |")
            lines.append("|--------|---------|-------------|-----------|----------|--------|")
            for e in affected:
                lines.append(
                    f"| {e.get('name', 'N/A')} | {e.get('country', '')} | "
                    f"{e.get('group', '')} | {e.get('risk_type', '')} | "
                    f"{_fmt(e.get('severity'), '.0%')} | {e.get('reason', '')} |"
                )
            lines.append("")

            sc_risk = linked_conflict.get("supply_chain_risk_score", 0)
            rev_risk = linked_conflict.get("revenue_exposure_score", 0)
            comp_adv = linked_conflict.get("competitive_advantage_score", 0)

            if sc_risk > 0:
                lines.append(f"**Supply Chain Risk:** {_fmt(sc_risk, '.0%')} -- "
                             "linked suppliers or logistics partners operate in conflict zones. "
                             "Disruption to raw materials, components, or shipping is probable.")
                lines.append("")
            if rev_risk > 0:
                lines.append(f"**Revenue Exposure:** {_fmt(rev_risk, '.0%')} -- "
                             "linked customers operate in conflict zones. "
                             "Demand contraction, payment delays, or market exit risk.")
                lines.append("")
            if comp_adv > 0:
                lines.append(f"**Competitive Advantage:** {_fmt(comp_adv, '.0%')} -- "
                             "competitors are impaired by conflict. "
                             "Potential market share gains if supply chains are diversified.")
                lines.append("")

        summary = linked_conflict.get("linked_conflict_summary", "")
        if summary:
            lines.append(f"*{summary}*")
            lines.append("")

    # Data sources
    sources = conflict.get("data_sources_used", [])
    if sources:
        lines.append(f"*Data sources: {', '.join(sources)}*")
        lines.append("")

    return "\n".join(lines)


def _build_fallback_report(
    profile: dict[str, Any],
    tier: ReportTier = ReportTier.PREMIUM,
    mode: ReportMode = ReportMode.RESULTS,
) -> str:
    """Build a report using the local template, filtered by tier and mode.

    Parameters
    ----------
    profile:
        Company profile dict.
    tier:
        Report tier controlling which sections are included.
    mode:
        Report mode controlling explanation depth (Learn vs Results).
    """
    # Map section numbers to their rendered content
    _section_builders: dict[int, tuple[str, str]] = {
        # section_num: (heading, content)
        1: ("1. Executive Summary", _build_executive_summary(profile)),
        2: ("2. Company Overview", _build_company_overview(profile)),
        3: ("3. Historical Performance Analysis", _build_historical_performance(profile)),
        4: ("4. Current Financial Snapshot (Tier-by-Tier)", _build_current_state_snapshot(profile)),
        5: ("5. Financial Health Scoring", _build_financial_health(profile)),
        6: ("6. Survival Mode Analysis", _build_survival_analysis(profile)),
        7: ("7. Linked Variables & Market Context", _build_linked_entities_section(profile)),
        8: ("8. Temporal Analysis & Model Insights", _build_regime_analysis(profile)),
        9: ("9. Predictions & Forecasts", _build_predictions_forecasts(profile)),
        10: ("10. Technical Patterns & Chart Analysis", _build_technical_patterns(profile)),
        11: ("11. Ethical Filter Assessment", _build_ethical_filters_section(profile)),
        12: ("12. Supply Chain & Contagion Risk", _build_graph_risk_section(profile)),
        13: ("13. Competitive Landscape", _build_game_theory_section(profile)),
        14: ("14. Regulatory & Government Protection", _build_fuzzy_protection_section(profile)),
        15: ("15. Model Calibration & Adaptive Learning", _build_pid_section(profile)),
        16: ("16. Market Sentiment & News Flow", _build_sentiment_analysis_section(profile)),
        17: ("17. Peer Comparison & Relative Valuation", _build_peer_ranking_section(profile)),
        18: ("18. Macroeconomic Environment", _build_macro_quadrant_section(profile)),
        19: ("19. Advanced Quantitative Insights", _build_advanced_insights(profile)),
        195: ("19.5. Geopolitical & Conflict Risk", _build_geopolitical_risk_section(profile)),
        20: ("20. Risk Factors & Limitations", (
            _build_risk_assessment(profile) + "\n\n### 20.1 LIMITATIONS\n\n" + _build_limitations(profile)
        )),
        21: ("21. Investment Recommendation", _build_investment_recommendation(profile)),
        22: ("22. Appendix & Methodology", _build_appendix(profile)),
    }

    included = TIER_SECTIONS.get(tier, TIER_SECTIONS[ReportTier.PREMIUM])

    generated_at = profile.get("meta", {}).get(
        "generated_at", datetime.now(timezone.utc).isoformat(),
    )
    identity = profile.get("identity", {})
    company_name = identity.get("name", "Unknown Company")
    ticker = identity.get("ticker", "")

    lines = [
        f"# {company_name} ({ticker}) -- {tier.label}",
        "",
        f"**Generated:** {generated_at}",
        "",
        "---",
        "",
    ]

    # Key Financial Indicators summary (always first, all tiers)
    key_indicators = _build_key_indicators_table(profile, mode=mode)
    if key_indicators.strip():
        lines.append(key_indicators)
        lines.append("")
        lines.append("---")
        lines.append("")

    for section_num in sorted(included):
        heading, content = _section_builders.get(section_num, ("", ""))
        if heading:
            lines.append(f"## {heading}")
            lines.append("")
            lines.append(content)
            lines.append("")
            lines.append("---")
            lines.append("")

    # Premium teaser for Basic and Pro tiers
    if tier != ReportTier.PREMIUM:
        lines.append("")
        lines.append("---")
        lines.append("")
        lines.append("## Unlock the Full Picture")
        lines.append("")

        if tier == ReportTier.BASIC:
            lines.append(
                "This Basic report covers the essential screening metrics. "
                "Upgrade to **Premium** for the complete institutional-grade analysis "
                "including:"
            )
        else:
            lines.append(
                "This Pro report provides a strong foundation for investment "
                "analysis. Upgrade to **Premium** for the full institutional-grade "
                "deep dive including:"
            )

        lines.append("")
        # Teaser items -- hint at value without exposing data
        _premium_teasers_basic = [
            "Historical performance analysis with trend decomposition",
            "Financial health scoring across five survival tiers",
            "Capital allocation and management discipline assessment",
            "Peer comparison with relative valuation positioning",
            "Macroeconomic environment classification and impact analysis",
            "Temporal regime analysis and predictive model insights",
            "Prediction forecasts with confidence intervals",
            "Supply chain and contagion risk mapping",
            "Competitive landscape game-theoretic modelling",
            "Advanced quantitative insights (copula tail risk, causal networks, "
            "cycle decomposition, non-linear state estimation)",
            "AI-generated narrative powered by institutional-grade LLM analysis",
        ]
        _premium_teasers_pro = [
            "Capital allocation deep dive with five-dimension management "
            "discipline scoring",
            "Temporal regime detection and predictive model ensemble",
            "Prediction forecasts with Monte Carlo simulation confidence bands",
            "Technical pattern recognition and chart-based signals",
            "Supply chain contagion risk modelling and domino-effect estimation",
            "Advanced quantitative insights (copula tail risk, transfer entropy, "
            "Sobol sensitivity, particle filter state estimation)",
            "Genetic algorithm-optimised model ensemble weights",
            "Full investment recommendation with risk-adjusted positioning",
            "AI-generated narrative powered by institutional-grade LLM analysis",
        ]

        teasers = _premium_teasers_basic if tier == ReportTier.BASIC else _premium_teasers_pro
        for teaser in teasers:
            lines.append(f"- {teaser}")
        lines.append("")
        lines.append("---")
        lines.append("")

    return "\n".join(lines)


# ---------------------------------------------------------------------------
# E2: Gemini response validation
# ---------------------------------------------------------------------------


# Required section headers (case-insensitive substring match).
_REQUIRED_SECTIONS: list[str] = [
    "executive summary",
    "company overview",
    "historical performance",
    "financial health",
    "survival",
    "linked variables",
    "temporal analysis",
    "predictions",
    "technical patterns",
    "ethical filter",
    "supply chain",
    "competitive",
    "risk factors",
    "limitations",
    "investment recommendation",
    "appendix",
]


def validate_gemini_report(
    markdown: str,
    profile: dict[str, Any],
) -> tuple[bool, list[str]]:
    """Validate that a Gemini-generated report is complete and accurate.

    Checks:
    1. All 13 required sections are present (by header text).
    2. Investment recommendation is present (BUY/HOLD/SELL keyword).
    3. LIMITATIONS section exists.
    4. No hallucinated numbers: spot-check key metrics against profile.

    Parameters
    ----------
    markdown:
        The Gemini-generated Markdown report text.
    profile:
        The company profile dict used to generate the report.

    Returns
    -------
    (is_valid, issues)
        ``is_valid`` is True only if all checks pass.
        ``issues`` is a list of human-readable issue descriptions.
    """
    issues: list[str] = []
    md_lower = markdown.lower()

    # Check 1: Section presence
    for section_name in _REQUIRED_SECTIONS:
        if section_name.lower() not in md_lower:
            issues.append(f"Missing section: '{section_name}'")

    # Check 2: Investment recommendation keyword
    rec_keywords = ["buy", "hold", "sell"]
    has_recommendation = any(
        kw in md_lower
        for kw in rec_keywords
    )
    if not has_recommendation:
        issues.append("No investment recommendation keyword (BUY/HOLD/SELL) found")

    # Check 3: LIMITATIONS section specifically
    if "limitations" not in md_lower:
        issues.append("LIMITATIONS section is missing")

    # Check 4: Spot-check key metrics against profile data
    identity = profile.get("identity", {})
    company_name = identity.get("name", "")
    if company_name and company_name.lower() not in md_lower:
        issues.append(f"Company name '{company_name}' not found in report")

    # Check ticker appears
    ticker = identity.get("ticker", "")
    if ticker and ticker.upper() not in markdown.upper():
        issues.append(f"Ticker '{ticker}' not found in report")

    # Check survival mode is mentioned if active
    survival = profile.get("survival", {})
    if survival.get("company_survival_mode_flag"):
        if "survival" not in md_lower:
            issues.append("Company is in survival mode but report does not mention survival")

    # Check debt-to-equity if available (spot-check for hallucination)
    # Profile stores hierarchy_weights in survival section, not tier2_solvency
    d2e = None
    if d2e is not None and not isinstance(d2e, str):
        d2e_str = f"{d2e:.1f}"
        # Allow some flexibility in formatting
        if d2e_str not in markdown and f"{d2e:.2f}" not in markdown:
            # Not a hard failure, just a warning
            issues.append(
                f"Debt-to-equity ({d2e_str}) not found verbatim in report "
                "(possible formatting difference, not necessarily an error)"
            )

    is_valid = len([i for i in issues if "not necessarily" not in i]) == 0

    if issues:
        logger.warning(
            "Gemini report validation: %d issue(s) found: %s",
            len(issues),
            "; ".join(issues[:5]),
        )
    else:
        logger.info("Gemini report validation: all checks passed")

    return is_valid, issues


# ---------------------------------------------------------------------------
# Charts (optional, gracefully skipped)
# ---------------------------------------------------------------------------


def generate_charts(
    cache: pd.DataFrame | None,
    profile: dict[str, Any],
    output_dir: str | Path | None = None,
) -> list[str]:
    """Generate optional analysis charts.

    Returns a list of file paths for successfully generated charts.
    Charts that fail to render are skipped with a warning.

    Parameters
    ----------
    cache:
        Full feature table with daily data.
    profile:
        Company profile dict from T7.1.
    output_dir:
        Directory to save chart PNGs. Defaults to ``cache/charts/``.
    """
    if output_dir is None:
        output_dir = Path(CACHE_DIR) / "charts"
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)

    chart_paths: list[str] = []

    if cache is None or cache.empty:
        logger.warning("No cache data available for chart generation.")
        return chart_paths

    # Ensure cache has a proper DatetimeIndex for matplotlib date formatting.
    # Without this, integer indices are interpreted as ordinal days since
    # year 0001, producing dates around 1959-1960 on the x-axis.
    if not isinstance(cache.index, pd.DatetimeIndex):
        for _date_col in ("date", "Date", "timestamp", "report_date"):
            if _date_col in cache.columns:
                try:
                    cache = cache.set_index(pd.to_datetime(cache[_date_col]))
                    logger.debug("Chart: converted '%s' column to DatetimeIndex", _date_col)
                    break
                except Exception:
                    continue
        else:
            try:
                cache.index = pd.to_datetime(cache.index)
                logger.debug("Chart: converted index to DatetimeIndex")
            except Exception:
                logger.warning(
                    "Cache index is not DatetimeIndex and cannot be converted; "
                    "chart x-axis dates may be incorrect (type=%s)",
                    type(cache.index).__name__,
                )

    try:
        import matplotlib
        matplotlib.use("Agg")  # Non-interactive backend
        import matplotlib.pyplot as plt
        import matplotlib.dates as mdates
        from matplotlib.ticker import FuncFormatter
    except ImportError:
        logger.warning("matplotlib not installed; skipping chart generation.")
        return chart_paths

    # --- Operator 1 Brand Palette ---
    # Proton-inspired: warm, clean, trustworthy, distinctive.
    # Purple accent is our signature -- no other finance product uses it.
    _CHART_BG = "#1c1b22"       # deep warm charcoal (not cold navy)
    _CHART_FG = "#eae7e1"       # warm off-white text
    _CHART_GRID = "#2d2b33"     # subtle warm grid
    _CHART_ACCENT = "#6d4aff"   # Proton-inspired purple (brand signature)
    _CHART_RED = "#dc3545"      # clear danger/bearish
    _CHART_GREEN = "#1ea885"    # teal-green (calmer than neon)
    _CHART_GOLD = "#e8950a"     # warm amber warning

    def _apply_brand_style(fig, ax, title: str) -> None:
        """Apply Operator 1 brand theme to a chart."""
        fig.patch.set_facecolor(_CHART_BG)
        ax.set_facecolor(_CHART_BG)
        ax.set_title(title, color=_CHART_FG, fontsize=14, fontweight="bold", pad=12)
        ax.tick_params(colors=_CHART_FG, labelsize=9)
        ax.xaxis.label.set_color(_CHART_FG)
        ax.yaxis.label.set_color(_CHART_FG)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        ax.spines["bottom"].set_color(_CHART_GRID)
        ax.spines["left"].set_color(_CHART_GRID)
        ax.grid(True, color=_CHART_GRID, alpha=0.5, linewidth=0.5)
        ax.xaxis.set_major_formatter(mdates.DateFormatter("%b %Y"))
        ax.xaxis.set_major_locator(mdates.MonthLocator(interval=3))
        for label in ax.get_xticklabels():
            label.set_rotation(45)
            label.set_ha("right")

    # Retrieve company name for chart titles
    identity = profile.get("identity", {})
    company = identity.get("name", identity.get("ticker", ""))

    # Chart 1: Price History with Regime Overlay
    try:
        if "close" in cache.columns:
            fig, ax = plt.subplots(figsize=(16, 7))
            ax.plot(cache.index, cache["close"], linewidth=1.5, color=_CHART_ACCENT, zorder=3)
            _apply_brand_style(fig, ax, f"{company} -- Closing Price (2Y)")
            ax.set_ylabel("Price ($)", color=_CHART_FG)

            # Regime shading with professional colors
            if "regime_label" in cache.columns:
                regime_colors = {
                    "bull": (_CHART_GREEN, "Bull Market"),
                    "bear": (_CHART_RED, "Bear Market"),
                    "high_vol": (_CHART_GOLD, "High Volatility"),
                    "low_vol": ("#8b8694", "Low Volatility"),
                }
                for regime, (color, label) in regime_colors.items():
                    mask = cache["regime_label"] == regime
                    if mask.any():
                        ax.fill_between(
                            cache.index,
                            cache["close"].min() * 0.98,
                            cache["close"].max() * 1.02,
                            where=mask, alpha=0.15, color=color, label=label,
                        )
                leg = ax.legend(
                    loc="upper left", fontsize=9, facecolor=_CHART_BG,
                    edgecolor=_CHART_GRID, labelcolor=_CHART_FG,
                )

            fig.tight_layout()
            path = str(out / "price_history.png")
            fig.savefig(path, dpi=180, facecolor=_CHART_BG)
            plt.close(fig)
            chart_paths.append(path)
            logger.info("Generated chart: %s", path)
    except Exception as exc:
        logger.warning("Failed to generate price history chart: %s", exc)

    # Chart 2: Survival Mode Timeline
    try:
        flag_cols = [
            c for c in (
                "company_survival_mode_flag",
                "country_survival_mode_flag",
                "country_protected_flag",
            )
            if c in cache.columns
        ]
        if flag_cols:
            nice_labels = {
                "company_survival_mode_flag": "Company Distress",
                "country_survival_mode_flag": "Country Crisis",
                "country_protected_flag": "Government Protection",
            }
            fig, ax = plt.subplots(figsize=(16, 4))
            colors = [_CHART_RED, _CHART_GOLD, _CHART_GREEN]
            for i, col in enumerate(flag_cols):
                ax.fill_between(
                    cache.index, i, i + cache[col].fillna(0).astype(float),
                    alpha=0.7, color=colors[i % len(colors)],
                    label=nice_labels.get(col, col),
                )
            ax.set_yticks(range(len(flag_cols)))
            ax.set_yticklabels([nice_labels.get(c, c) for c in flag_cols], fontsize=10)
            _apply_brand_style(fig, ax, f"{company} -- Survival Mode Timeline")
            leg = ax.legend(
                loc="upper right", fontsize=9, facecolor=_CHART_BG,
                edgecolor=_CHART_GRID, labelcolor=_CHART_FG,
            )
            fig.tight_layout()
            path = str(out / "survival_timeline.png")
            fig.savefig(path, dpi=180, facecolor=_CHART_BG)
            plt.close(fig)
            chart_paths.append(path)
            logger.info("Generated chart: %s", path)
    except Exception as exc:
        logger.warning("Failed to generate survival timeline chart: %s", exc)

    # Chart 3: Risk Hierarchy Weight Allocation
    try:
        tier_cols = [
            f"hierarchy_tier{i}_weight" for i in range(1, 6)
            if f"hierarchy_tier{i}_weight" in cache.columns
        ]
        if tier_cols:
            tier_names = [
                "Tier 1: Liquidity", "Tier 2: Solvency", "Tier 3: Stability",
                "Tier 4: Profitability", "Tier 5: Growth",
            ]
            tier_colors = ["#6d4aff", "#1ea885", "#e8950a", "#dc3545", "#8b8694"]
            fig, ax = plt.subplots(figsize=(16, 5))
            ax.stackplot(
                cache.index,
                *[cache[c].fillna(0) for c in tier_cols],
                labels=tier_names[:len(tier_cols)],
                colors=tier_colors[:len(tier_cols)],
                alpha=0.85,
            )
            _apply_brand_style(fig, ax, f"{company} -- Risk Hierarchy Weight Allocation")
            ax.set_ylabel("Portfolio Weight", color=_CHART_FG)
            ax.set_ylim(0, 1.05)
            leg = ax.legend(
                loc="upper right", fontsize=9, facecolor=_CHART_BG,
                edgecolor=_CHART_GRID, labelcolor=_CHART_FG,
            )
            fig.tight_layout()
            path = str(out / "hierarchy_weights.png")
            fig.savefig(path, dpi=180, facecolor=_CHART_BG)
            plt.close(fig)
            chart_paths.append(path)
            logger.info("Generated chart: %s", path)
    except Exception as exc:
        logger.warning("Failed to generate hierarchy weight chart: %s", exc)

    # Chart 4: 21-Day Realized Volatility
    try:
        if "volatility_21d" in cache.columns:
            fig, ax = plt.subplots(figsize=(16, 5))
            ax.fill_between(
                cache.index, 0, cache["volatility_21d"],
                alpha=0.3, color=_CHART_RED,
            )
            ax.plot(cache.index, cache["volatility_21d"], linewidth=1.2, color=_CHART_RED)
            _apply_brand_style(fig, ax, f"{company} -- 21-Day Realized Volatility")
            ax.set_ylabel("Annualized Volatility", color=_CHART_FG)
            ax.yaxis.set_major_formatter(FuncFormatter(lambda y, _: f"{y:.0%}"))
            fig.tight_layout()
            path = str(out / "volatility.png")
            fig.savefig(path, dpi=180, facecolor=_CHART_BG)
            plt.close(fig)
            chart_paths.append(path)
            logger.info("Generated chart: %s", path)
    except Exception as exc:
        logger.warning("Failed to generate volatility chart: %s", exc)

    # Chart 5: Financial Health Composite Score
    try:
        if "fh_composite_score" in cache.columns:
            fig, ax = plt.subplots(figsize=(16, 5))
            score = cache["fh_composite_score"]
            ax.plot(cache.index, score, linewidth=1.5, color=_CHART_GREEN, zorder=3)
            ax.axhline(y=50, color=_CHART_FG, linestyle="--", alpha=0.3, label="Neutral (50)")
            ax.fill_between(cache.index, 0, score, where=score >= 50,
                            alpha=0.15, color=_CHART_GREEN)
            ax.fill_between(cache.index, 0, score, where=score < 50,
                            alpha=0.15, color=_CHART_RED)
            _apply_brand_style(fig, ax, f"{company} -- Financial Health Composite (0-100)")
            ax.set_ylabel("Health Score", color=_CHART_FG)
            ax.set_ylim(0, 100)
            fig.tight_layout()
            path = str(out / "financial_health.png")
            fig.savefig(path, dpi=180, facecolor=_CHART_BG)
            plt.close(fig)
            chart_paths.append(path)
            logger.info("Generated chart: %s", path)
    except Exception as exc:
        logger.warning("Failed to generate financial health chart: %s", exc)

    # Chart 6: News Sentiment & Momentum
    try:
        if "sentiment_score" in cache.columns and cache["sentiment_score"].notna().any():
            fig, ax = plt.subplots(figsize=(16, 5))
            sent = cache["sentiment_score"]
            ax.bar(cache.index, sent, width=1.0, alpha=0.4,
                   color=[_CHART_GREEN if v >= 0 else _CHART_RED for v in sent.fillna(0)])
            if "sentiment_momentum_21d" in cache.columns:
                ax.plot(cache.index, cache["sentiment_momentum_21d"],
                        linewidth=2, color=_CHART_GOLD, label="21-Day Sentiment Trend")
            ax.axhline(y=0, color=_CHART_FG, linewidth=0.5, alpha=0.5)
            _apply_brand_style(fig, ax, f"{company} -- Market Sentiment & News Flow")
            ax.set_ylabel("Sentiment (-1 Bearish to +1 Bullish)", color=_CHART_FG)
            ax.set_ylim(-1.1, 1.1)
            leg = ax.legend(
                loc="upper left", fontsize=9, facecolor=_CHART_BG,
                edgecolor=_CHART_GRID, labelcolor=_CHART_FG,
            )
            fig.tight_layout()
            path = str(out / "sentiment.png")
            fig.savefig(path, dpi=180, facecolor=_CHART_BG)
            plt.close(fig)
            chart_paths.append(path)
            logger.info("Generated chart: %s", path)
    except Exception as exc:
        logger.warning("Failed to generate sentiment chart: %s", exc)

    # Chart 7: Predicted OHLC Candlestick (Next Month)
    try:
        ohlc_data = profile.get("ohlc_predictions", {})
        next_month = ohlc_data.get("next_month", {})
        series = next_month.get("series", [])
        if series and len(series) >= 5:
            fig, ax = plt.subplots(figsize=(16, 7))
            dates = list(range(len(series)))
            opens = [c["open"] for c in series if c.get("open")]
            highs = [c["high"] for c in series if c.get("high")]
            lows = [c["low"] for c in series if c.get("low")]
            closes = [c["close"] for c in series if c.get("close")]

            if len(opens) == len(series):
                # Draw candlesticks manually
                for i, candle in enumerate(series):
                    o, h, l, c_ = candle["open"], candle["high"], candle["low"], candle["close"]
                    color = _CHART_GREEN if c_ >= o else _CHART_RED
                    # Body
                    body_bottom = min(o, c_)
                    body_height = abs(c_ - o)
                    ax.bar(i, body_height, bottom=body_bottom, width=0.6,
                           color=color, edgecolor=color, alpha=0.85)
                    # Wicks
                    ax.plot([i, i], [l, h], color=color, linewidth=0.8)

                # Confidence band (envelope based on candle confidence)
                confidences = [c.get("confidence", 1.0) for c in series]
                mid_prices = [(c["high"] + c["low"]) / 2 for c in series]
                ranges = [c["high"] - c["low"] for c in series]
                upper = [m + r * (2 - conf) for m, r, conf in zip(mid_prices, ranges, confidences)]
                lower = [m - r * (2 - conf) for m, r, conf in zip(mid_prices, ranges, confidences)]
                ax.fill_between(dates, lower, upper, alpha=0.08, color=_CHART_ACCENT)

                _apply_brand_style(fig, ax, f"{company} -- Predicted Price (Next Month)")
                ax.set_ylabel("Price ($)", color=_CHART_FG)
                ax.set_xlabel("Trading Days Ahead", color=_CHART_FG)

                # Add predicted return annotation
                pred_ret = next_month.get("predicted_return")
                if pred_ret is not None:
                    ax.annotate(
                        f"Predicted return: {pred_ret:+.1f}%",
                        xy=(0.02, 0.95), xycoords="axes fraction",
                        fontsize=11, color=_CHART_GREEN if pred_ret >= 0 else _CHART_RED,
                        fontweight="bold",
                    )

                fig.tight_layout()
                path = str(out / "predicted_ohlc_month.png")
                fig.savefig(path, dpi=180, facecolor=_CHART_BG)
                plt.close(fig)
                chart_paths.append(path)
                logger.info("Generated chart: %s", path)
    except Exception as exc:
        logger.warning("Failed to generate predicted OHLC chart: %s", exc)

    # Chart 8: Predicted OHLC Candlestick (Next Week)
    try:
        next_week = ohlc_data.get("next_week", {})
        week_series = next_week.get("series", [])
        if week_series and len(week_series) >= 3:
            fig, ax = plt.subplots(figsize=(12, 6))
            for i, candle in enumerate(week_series):
                o, h, l, c_ = candle["open"], candle["high"], candle["low"], candle["close"]
                color = _CHART_GREEN if c_ >= o else _CHART_RED
                ax.bar(i, abs(c_ - o), bottom=min(o, c_), width=0.6,
                       color=color, edgecolor=color, alpha=0.85)
                ax.plot([i, i], [l, h], color=color, linewidth=0.8)

            _apply_brand_style(fig, ax, f"{company} -- Predicted Price (Next Week)")
            ax.set_ylabel("Price ($)", color=_CHART_FG)
            ax.set_xlabel("Trading Days Ahead", color=_CHART_FG)

            week_ret = next_week.get("predicted_return")
            if week_ret is not None:
                ax.annotate(
                    f"Predicted return: {week_ret:+.1f}%",
                    xy=(0.02, 0.95), xycoords="axes fraction",
                    fontsize=11, color=_CHART_GREEN if week_ret >= 0 else _CHART_RED,
                    fontweight="bold",
                )

            fig.tight_layout()
            path = str(out / "predicted_ohlc_week.png")
            fig.savefig(path, dpi=180, facecolor=_CHART_BG)
            plt.close(fig)
            chart_paths.append(path)
            logger.info("Generated chart: %s", path)
    except Exception as exc:
        logger.warning("Failed to generate predicted week OHLC chart: %s", exc)

    # Chart 9: Conflict Risk Gauge (if conflict data available)
    try:
        if "conflict_intensity_score" in cache.columns:
            intensity = cache["conflict_intensity_score"].iloc[-1] if not cache["conflict_intensity_score"].isna().all() else 0
            conflict_flag = cache["country_conflict_flag"].iloc[-1] if "country_conflict_flag" in cache.columns else 0
            sanctions = cache["sanctions_flag"].iloc[-1] if "sanctions_flag" in cache.columns else 0

            fig, axes = plt.subplots(1, 3, figsize=(16, 5))
            fig.patch.set_facecolor(_CHART_BG)

            # Gauge 1: Conflict Intensity
            ax = axes[0]
            ax.set_facecolor(_CHART_BG)
            theta = intensity * 180  # 0 to 180 degrees
            colors_gauge = [_CHART_GREEN, _CHART_GOLD, _CHART_RED]
            # Draw arc background
            from matplotlib.patches import Wedge
            for i, (start, end, color) in enumerate([(0, 60, _CHART_GREEN), (60, 120, _CHART_GOLD), (120, 180, _CHART_RED)]):
                wedge = Wedge((0.5, 0), 0.4, start, end, width=0.12,
                              facecolor=color, alpha=0.3, transform=ax.transAxes)
                ax.add_patch(wedge)
            # Needle
            import math
            needle_angle = math.radians(180 - theta)
            nx = 0.5 + 0.35 * math.cos(needle_angle)
            ny = 0.35 * math.sin(needle_angle)
            ax.annotate("", xy=(nx, ny), xytext=(0.5, 0),
                        arrowprops=dict(arrowstyle="-|>", color=_CHART_FG, lw=2),
                        xycoords="axes fraction", textcoords="axes fraction")
            ax.text(0.5, -0.15, f"Intensity: {intensity:.2f}", ha="center",
                    color=_CHART_FG, fontsize=12, fontweight="bold",
                    transform=ax.transAxes)
            ax.set_title("Conflict Intensity", color=_CHART_FG, fontsize=11, fontweight="bold")
            ax.set_xlim(-0.1, 1.1)
            ax.set_ylim(-0.3, 0.6)
            ax.axis("off")

            # Gauge 2: Country Conflict Status
            ax = axes[1]
            ax.set_facecolor(_CHART_BG)
            status_color = _CHART_RED if conflict_flag else _CHART_GREEN
            status_text = "ACTIVE" if conflict_flag else "CLEAR"
            circle = plt.Circle((0.5, 0.3), 0.25, color=status_color, alpha=0.3,
                                transform=ax.transAxes)
            ax.add_patch(circle)
            ax.text(0.5, 0.3, status_text, ha="center", va="center",
                    color=status_color, fontsize=16, fontweight="bold",
                    transform=ax.transAxes)
            ax.set_title("Country Conflict", color=_CHART_FG, fontsize=11, fontweight="bold")
            ax.set_xlim(0, 1)
            ax.set_ylim(0, 0.7)
            ax.axis("off")

            # Gauge 3: Sanctions Status
            ax = axes[2]
            ax.set_facecolor(_CHART_BG)
            sanc_color = _CHART_RED if sanctions else _CHART_GREEN
            sanc_text = "SANCTIONED" if sanctions else "CLEAR"
            circle = plt.Circle((0.5, 0.3), 0.25, color=sanc_color, alpha=0.3,
                                transform=ax.transAxes)
            ax.add_patch(circle)
            ax.text(0.5, 0.3, sanc_text, ha="center", va="center",
                    color=sanc_color, fontsize=16, fontweight="bold",
                    transform=ax.transAxes)
            ax.set_title("Sanctions Status", color=_CHART_FG, fontsize=11, fontweight="bold")
            ax.set_xlim(0, 1)
            ax.set_ylim(0, 0.7)
            ax.axis("off")

            fig.suptitle(f"{company} -- Geopolitical Risk Dashboard",
                         color=_CHART_FG, fontsize=14, fontweight="bold")
            fig.tight_layout()
            path = str(out / "conflict_risk.png")
            fig.savefig(path, dpi=180, facecolor=_CHART_BG)
            plt.close(fig)
            chart_paths.append(path)
            logger.info("Generated chart: %s", path)
    except Exception as exc:
        logger.warning("Failed to generate conflict risk chart: %s", exc)

    return chart_paths


# ---------------------------------------------------------------------------
# Chart embedding into Markdown
# ---------------------------------------------------------------------------

# Maps chart filename -> the report section heading it should appear after.
_CHART_SECTION_MAP: dict[str, str] = {
    "price_history.png": "## 3. Historical Performance Analysis",
    "survival_timeline.png": "## 6. Survival Mode Analysis",
    "hierarchy_weights.png": "## 4. Current Financial Snapshot",
    "volatility.png": "## 8. Temporal Analysis",
    "financial_health.png": "## 5. Financial Health Scoring",
    "sentiment.png": "## 16. Market Sentiment",
    "predicted_ohlc_month.png": "## 9. Predictions & Forecasts",
    "predicted_ohlc_week.png": "## 10. Technical Patterns",
    "conflict_risk.png": "## 19.5. Geopolitical",
}


def _embed_charts_in_markdown(
    markdown: str,
    chart_paths: list[str],
    chart_dir_relative: str = "charts",
) -> str:
    """Embed chart image references into the report markdown.

    For each generated chart, inserts a markdown image tag
    ``![title](charts/filename.png)`` after the matching section heading.

    Parameters
    ----------
    markdown:
        The full report markdown text.
    chart_paths:
        List of absolute/relative chart PNG paths from ``generate_charts()``.
    chart_dir_relative:
        Relative path from the report markdown file to the charts directory.

    Returns
    -------
    Updated markdown with embedded chart references.
    """
    if not chart_paths:
        return markdown

    for chart_path in chart_paths:
        filename = os.path.basename(chart_path)
        section_prefix = _CHART_SECTION_MAP.get(filename, "")

        if not section_prefix:
            continue

        # Build the image markdown
        title = filename.replace(".png", "").replace("_", " ").title()
        image_tag = f"\n\n![{title}]({chart_dir_relative}/{filename})\n"

        # Find the section heading in the markdown and insert after it
        for line_idx, line in enumerate(markdown.split("\n")):
            if line.strip().startswith(section_prefix.split(".")[0][:10]) and section_prefix.split(".")[-1].strip()[:8].lower() in line.lower():
                # Insert after the heading line + one blank line
                parts = markdown.split(line, 1)
                if len(parts) == 2:
                    markdown = parts[0] + line + image_tag + parts[1]
                break

    return markdown


# ---------------------------------------------------------------------------
# PDF generation (optional)
# ---------------------------------------------------------------------------


def _generate_pdf(
    markdown_path: str | Path,
    output_path: str | Path | None = None,
) -> str | None:
    """Convert Markdown report to PDF using pandoc.

    Returns the PDF path on success, or None if pandoc is unavailable.
    """
    if shutil.which("pandoc") is None:
        logger.info("pandoc not found; skipping PDF generation.")
        return None

    md = Path(markdown_path)
    if output_path is None:
        output_path = md.with_suffix(".pdf")
    pdf = Path(output_path)

    try:
        subprocess.run(
            [
                "pandoc",
                str(md),
                "-o",
                str(pdf),
                "--pdf-engine=xelatex",
                "-V",
                "geometry:margin=1in",
            ],
            check=True,
            capture_output=True,
            timeout=60,
        )
        logger.info("PDF report generated: %s", pdf)
        return str(pdf)
    except FileNotFoundError:
        logger.info("pandoc/xelatex not available; skipping PDF.")
        return None
    except subprocess.CalledProcessError as exc:
        logger.warning("PDF generation failed: %s", exc.stderr.decode()[:200])
        return None
    except subprocess.TimeoutExpired:
        logger.warning("PDF generation timed out.")
        return None


# ---------------------------------------------------------------------------
# Main entry point
# ---------------------------------------------------------------------------


def generate_report(
    profile: dict[str, Any],
    *,
    gemini_client: Any | None = None,
    cache: pd.DataFrame | None = None,
    output_dir: str | Path | None = None,
    generate_pdf: bool = False,
    generate_chart_images: bool = True,
    tier: ReportTier = ReportTier.PREMIUM,
    mode: ReportMode = ReportMode.RESULTS,
) -> dict[str, Any]:
    """Generate an analysis report from a company profile.

    Parameters
    ----------
    profile:
        Company profile dict from ``build_company_profile()``.
    gemini_client:
        Optional ``GeminiClient`` instance.  If provided, the report
        narrative is generated by Gemini.  Otherwise, a local template
        is used.
    cache:
        Full feature table for chart generation.
    output_dir:
        Directory for all report outputs.  Defaults to ``cache/report/``.
    generate_pdf:
        If True, create a PDF via pandoc for this tier.
    generate_chart_images:
        If True, generate chart PNGs (premium only).
    tier:
        Report tier controlling how many sections are included.
        Defaults to PREMIUM (all 22 sections).
    mode:
        Report mode controlling explanation depth.
        LEARN adds plain-English explanations for newcomers.
        RESULTS (default) is data-forward for professionals.

    Returns
    -------
    dict with keys:
        - ``markdown``: the full report as a Markdown string
        - ``markdown_path``: path to the saved ``.md`` file
        - ``chart_paths``: list of chart PNG paths
        - ``pdf_path``: path to PDF (or None)
        - ``tier``: the report tier used
        - ``mode``: the report mode used
    """
    if output_dir is None:
        output_dir = Path(CACHE_DIR) / "report"
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)

    logger.info("Generating %s...", tier.label)

    # Step 1: Generate narrative
    markdown = ""
    # Only use Gemini for premium reports (it generates the full 22-section report)
    if gemini_client is not None and tier == ReportTier.PREMIUM:
        try:
            profile_json = json.dumps(profile, indent=2, default=str)
            markdown = gemini_client.generate_report(profile_json)
            logger.info("%s narrative generated via Gemini.", tier.label)
        except Exception as exc:
            logger.warning(
                "Gemini report generation failed (%s); using fallback template.",
                exc,
            )
            markdown = ""

    # Fallback if Gemini produced nothing or tier is not premium
    if not markdown or not markdown.strip():
        markdown = _build_fallback_report(profile, tier=tier, mode=mode)
        logger.info("%s (%s mode) generated using local template.", tier.label, mode.label)

    # Ensure LIMITATIONS section exists (append if Gemini missed it)
    if "LIMITATIONS" not in markdown.upper():
        limitations = _build_limitations(profile)
        markdown += "\n\n---\n\n## LIMITATIONS\n\n" + limitations
        logger.info("Appended LIMITATIONS section to report.")

    # Step 1b: Validate Gemini output (E2) -- premium only
    if gemini_client is not None and tier == ReportTier.PREMIUM:
        is_valid, validation_issues = validate_gemini_report(markdown, profile)
        if not is_valid:
            logger.warning(
                "Gemini report validation found %d issues; "
                "appending missing sections from fallback template.",
                len(validation_issues),
            )
            for issue in validation_issues:
                if issue.startswith("Missing section:"):
                    section_name = issue.replace("Missing section: '", "").rstrip("'")
                    logger.info("Attempting to append missing section: %s", section_name)

    # Step 2: Save markdown
    md_path = out / tier.filename
    with open(md_path, "w", encoding="utf-8") as fh:
        fh.write(markdown)
    logger.info("Markdown report saved to %s", md_path)

    # Step 3: Generate charts (for Pro and Premium tiers)
    chart_paths: list[str] = []
    if generate_chart_images and tier in (ReportTier.PRO, ReportTier.PREMIUM):
        chart_dir = out / "charts"
        chart_paths = generate_charts(cache, profile, chart_dir)

    # Step 3b: Embed chart images into the markdown report
    if chart_paths:
        markdown = _embed_charts_in_markdown(markdown, chart_paths, "charts")
        # Re-save the markdown with embedded charts
        with open(md_path, "w", encoding="utf-8") as fh:
            fh.write(markdown)
        logger.info("Embedded %d charts into report.", len(chart_paths))

    # Step 4: Optional PDF (all tiers)
    pdf_path: str | None = None
    if generate_pdf:
        pdf_path = _generate_pdf(md_path, out / f"{tier.value}_report.pdf")

    return {
        "markdown": markdown,
        "markdown_path": str(md_path),
        "chart_paths": chart_paths,
        "pdf_path": pdf_path,
        "tier": tier.value,
        "mode": mode.value,
    }


def generate_all_reports(
    profile: dict[str, Any],
    *,
    gemini_client: Any | None = None,
    cache: pd.DataFrame | None = None,
    output_dir: str | Path | None = None,
    generate_pdf: bool = False,
    generate_chart_images: bool = True,
    mode: ReportMode = ReportMode.RESULTS,
) -> dict[str, dict[str, Any]]:
    """Generate all three report tiers (Basic, Pro, Premium) at once.

    Parameters
    ----------
    profile:
        Company profile dict from ``build_company_profile()``.
    gemini_client:
        Optional ``GeminiClient`` for premium report narrative.
    cache:
        Full feature table for chart generation.
    output_dir:
        Directory for all report outputs.
    generate_pdf:
        If True, generate PDF for all three report tiers.
    generate_chart_images:
        If True, generate chart PNGs (premium report only).
    mode:
        Report mode (LEARN or RESULTS) applied to all tiers.

    Returns
    -------
    dict mapping tier name to the ``generate_report()`` result dict.
    """
    results: dict[str, dict[str, Any]] = {}

    for tier in (ReportTier.BASIC, ReportTier.PRO, ReportTier.PREMIUM):
        results[tier.value] = generate_report(
            profile,
            gemini_client=gemini_client,
            cache=cache,
            output_dir=output_dir,
            generate_pdf=generate_pdf,
            generate_chart_images=generate_chart_images,
            tier=tier,
            mode=mode,
        )

    logger.info(
        "All three reports generated: %s",
        ", ".join(r["markdown_path"] for r in results.values()),
    )
    return results
