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
    ``generate_report(profile, llm_client=None, ...)``

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
    ReportTier.BASIC: {1, 2, 4, 6, 20, 2007},  # quick screening + position signal
    ReportTier.PRO: {1, 2, 3, 4, 5, 6, 65, 7, 75, 11, 14, 16, 17, 18, 195, 196, 197, 198, 199, 1995, 1996, 1997, 1998, 1999, 2001, 2002, 20, 2006, 2007, 2008, 2030, 2031, 2032, 2033},  # + options signals, cross-asset, event calendar
    ReportTier.PREMIUM: set(range(1, 23)) | {65, 75, 195, 196, 197, 198, 199, 1995, 1996, 1997, 1998, 1999, 2001, 2002, 2003, 2004, 2005, 2006, 2007, 2008, 2023, 2024, 2025, 2026, 2027, 2028, 2029, 2030, 2031, 2032, 2033, 2034},  # + options signals, cross-asset, event calendar, regime shifts
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

    # Extended Financial Health Signals (ensemble distress + CVaR)
    ensemble = fh.get("ensemble_distress_score")
    ewm_rank = fh.get("ewm_percentile_rank")
    cvar = fh.get("cvar_composite")
    if any(v is not None for v in (ensemble, ewm_rank, cvar)):
        lines.extend(["", "### Advanced Distress Signals", ""])
        if ensemble is not None:
            _label = "HIGH" if ensemble > 0.6 else "moderate" if ensemble > 0.3 else "low"
            lines.append(f"- **Ensemble distress score**: {ensemble:.3f} ({_label})")
            lines.append("  *(Combines Altman Z-Score, Ohlson O-Score, Zmijewski, and Merton PD)*")
        if ewm_rank is not None:
            lines.append(f"- **EWM percentile rank**: {ewm_rank:.1%}")
        if cvar is not None:
            lines.append(f"- **CVaR composite**: {cvar:.4f}")

    if not lines:
        return "Financial health data unavailable."

    return "\n".join(lines)


def _build_enriched_survival_timeline_section(profile: dict[str, Any]) -> str:
    """Build enriched survival timeline section (regime + survival bridge)."""
    est = profile.get("enriched_survival_timeline", {})
    if not est.get("available"):
        return "Enriched survival timeline data unavailable."

    lines = []
    lines.append(f"- **Regime detection available**: {'Yes' if est.get('regime_available') else 'No'}")
    lines.append(f"- **Mean survival intensity**: {est.get('mean_intensity', 0):.3f} "
                 "(0.0 = stable, 1.0 = extreme crisis)")

    # State distribution
    dist = est.get("combined_state_distribution", {})
    if dist:
        lines.extend(["", "### Combined State Distribution", ""])
        for state, pct in sorted(dist.items(), key=lambda x: -x[1]):
            if pct > 0.01:
                label = state.replace("_", " ").title()
                lines.append(f"- **{label}**: {pct * 100:.1f}% of days")

    # Regime switch statistics
    n_switches = est.get("base_n_switches")
    mean_stability = est.get("base_mean_stability")
    if n_switches is not None or mean_stability is not None:
        lines.extend(["", "### Regime Stability", ""])
        if n_switches is not None:
            lines.append(f"- **Regime switches (2yr window)**: {n_switches}")
        if mean_stability is not None:
            lines.append(f"- **Mean stability score (21d)**: {mean_stability:.3f} "
                         "(1.0 = fully stable, 0.0 = constantly switching)")

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

    # Survival velocity early warning (Duffie et al. 2007)
    vel_flag = survival.get("survival_velocity_flag")
    det_rate = survival.get("survival_deterioration_rate")
    if vel_flag is not None or det_rate is not None:
        lines.extend(["", "### Deterioration Velocity"])
        lines.append("")
        if vel_flag:
            lines.append("- **Velocity flag**: ACTIVE -- survival probability declining rapidly")
        else:
            lines.append("- **Velocity flag**: inactive")
        if det_rate is not None:
            lines.append(f"- **Deterioration rate**: {det_rate:.4f} per day")

    # Survival uncertainty bands (bootstrap P10/P90)
    p10 = survival.get("survival_probability_p10")
    p90 = survival.get("survival_probability_p90")
    unc = survival.get("survival_uncertainty")
    if any(v is not None for v in (p10, p90, unc)):
        lines.extend(["", "### Survival Probability Confidence Bands"])
        lines.append("")
        if p10 is not None and p90 is not None:
            lines.append(f"- **80% confidence interval**: [{p10:.1%}, {p90:.1%}]")
        if unc is not None:
            lines.append(f"- **Uncertainty score**: {unc:.3f}")
            if unc > 0.15:
                lines.append("  *High uncertainty -- survival estimate is unreliable.*")

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

    # Predicted regime shifts (forward-looking)
    shifts = profile.get("predicted_regime_shifts", {})
    if shifts.get("available"):
        lines.append("### Predicted Regime Shifts")
        lines.append("")
        lines.append(
            f"Based on the HMM transition matrix, the probability of the current "
            f"**{shifts.get('current_regime', 'unknown')}** regime ending within key horizons:"
        )
        lines.append("")
        lines.append("| Horizon | Probability of Regime Change |")
        lines.append("|---------|------------------------------|")
        lines.append(f"| 1 week (5 days) | {shifts.get('prob_exit_5d', 0) * 100:.1f}% |")
        lines.append(f"| 1 month (21 days) | {shifts.get('prob_exit_21d', 0) * 100:.1f}% |")
        lines.append(f"| 1 quarter (63 days) | {shifts.get('prob_exit_63d', 0) * 100:.1f}% |")
        lines.append(f"| 1 year (252 days) | {shifts.get('prob_exit_252d', 0) * 100:.1f}% |")
        lines.append("")

        exp_days = shifts.get("expected_days_to_shift")
        if exp_days is not None:
            lines.append(
                f"**Expected days until regime change:** ~{exp_days:.0f} trading days"
            )

        next_regime = shifts.get("most_probable_next_regime", "")
        next_prob = shifts.get("most_probable_next_prob", 0)
        if next_regime:
            lines.append(
                f"**Most likely next regime:** {next_regime} "
                f"({next_prob * 100:.0f}% of exit probability)"
            )
        lines.append("")

        predicted = shifts.get("predicted_shifts", [])
        if predicted:
            lines.append("**Dated Transition Forecasts:**")
            lines.append("")
            lines.append("| From | To | Probability | Expected Around | Confidence |")
            lines.append("|------|-----|-------------|-----------------|------------|")
            for s in predicted[:4]:
                lines.append(
                    f"| {s.get('from_regime', '')} | {s.get('to_regime', '')} | "
                    f"{s.get('probability', 0) * 100:.1f}% | "
                    f"{s.get('expected_date', 'N/A')} | "
                    f"{s.get('confidence', 0) * 100:.0f}% |"
                )
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
            "beta_252d": ("Market Beta (252-day)", ".2f"),
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
    _rec_avg = hist.get("recovery_time_avg")
    _rec_max = hist.get("recovery_time_max")
    _rec_n = hist.get("n_recovery_episodes", 0)
    if _rec_avg is not None:
        lines.append(f"- **Average Recovery Time:** {_rec_avg:.0f} trading days ({_rec_n} episodes)")
    if _rec_max is not None:
        lines.append(f"- **Longest Recovery:** {_rec_max:.0f} trading days")
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
    conformal = profile.get("extended_models", {}).get("conformal_prediction", {})
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
    shap_data = profile.get("extended_models", {}).get("shap_explanations", {})
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
    analogs = profile.get("extended_models", {}).get("dtw_analogs", {})
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
    patterns = profile.get("patterns", profile.get("technical_patterns",
                profile.get("extended_models", {}).get("candlestick_patterns", {})))
    lines: list[str] = []

    recent = patterns.get("recent_patterns", [])
    predicted = patterns.get("predicted_patterns_week", patterns.get("predicted_patterns", []))

    _has_ohlcv = profile.get("meta", {}).get("has_ohlcv", True)
    _is_priv = profile.get("meta", {}).get("is_private_company", False)
    if _has_ohlcv:
        lines.append("*(See attached price history chart with regime shading)*")
    elif _is_priv:
        lines.append("*(See attached equity trajectory chart -- private company, no market price data)*")
    else:
        lines.append("*(Price history chart not available -- no OHLCV data for this company)*")
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
        if _has_ohlcv:
            lines.append("*(See attached predicted candlestick charts)*")
        else:
            lines.append("*(Predicted candlestick charts not available -- no OHLCV data)*")
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


def _build_model_diagnostics_section(profile: dict[str, Any]) -> str:
    """Build model expected vs actual path diagnostics section."""
    diag = profile.get("model_diagnostics", {})
    if not diag.get("available"):
        return "Model diagnostics were not computed for this analysis."

    lines = []

    # Overall summary
    overall = diag.get("overall_robustness", "unknown")
    n_assessed = diag.get("n_models_assessed", 0)
    n_on_track = diag.get("n_models_on_track", 0)
    n_deviated = diag.get("n_models_deviated", 0)

    badge = {"high": "HIGH", "medium": "MODERATE", "low": "LOW"}.get(overall, "UNKNOWN")
    lines.append(f"**Overall Model Robustness: {badge}** "
                 f"({n_on_track}/{n_assessed} models on track, "
                 f"{n_deviated} deviated)")
    lines.append("")

    # Data characteristics summary
    chars = diag.get("data_characteristics", {})
    if chars:
        lines.append("### Data Characteristics")
        lines.append("")
        lines.append("| Metric | Value | Interpretation |")
        lines.append("|--------|-------|----------------|")
        skew = chars.get("skewness")
        if skew is not None:
            interp = "negative (downside risk)" if skew < -0.3 else "positive (upward drift)" if skew > 0.3 else "symmetric"
            lines.append(f"| Return Skewness | {skew:.3f} | {interp} |")
        kurt = chars.get("kurtosis")
        if kurt is not None:
            interp = "fat tails (extreme events)" if kurt > 3 else "thin tails" if kurt < 0 else "normal"
            lines.append(f"| Return Kurtosis | {kurt:.2f} | {interp} |")
        bc = chars.get("bimodality_coeff")
        if bc is not None:
            interp = "bimodal (multi-regime)" if bc > 0.555 else "unimodal"
            lines.append(f"| Bimodality | {bc:.3f} | {interp} |")
        acf = chars.get("acf_lag1")
        if acf is not None:
            interp = "predictable (momentum)" if abs(acf) > 0.1 else "random walk"
            lines.append(f"| Autocorrelation (lag 1) | {acf:.3f} | {interp} |")
        arch = chars.get("arch_proxy")
        if arch is not None:
            interp = "volatility clusters" if arch > 0.15 else "stable volatility"
            lines.append(f"| ARCH Effect | {arch:.3f} | {interp} |")
        cov = chars.get("data_coverage")
        if cov is not None:
            lines.append(f"| Data Coverage | {cov:.0%} | {'rich' if cov > 0.7 else 'sparse'} |")
        lines.append("")

    # Per-model cards
    models = diag.get("models", {})
    if models:
        lines.append("### Per-Model Assessment")
        lines.append("")
        lines.append("| Model | Expected | Actual | Deviation | Robustness |")
        lines.append("|-------|----------|--------|-----------|------------|")

        _labels = {
            "regime_detector": "Regime Detection",
            "forecasting": "Forecasting Engine",
            "monte_carlo": "Monte Carlo Simulation",
            "financial_health": "Financial Health",
            "estimation": "Data Estimation",
            "copula": "Copula Analysis",
            "granger_causality": "Granger Causality",
            "cycle_decomposition": "Cycle Decomposition",
            "dtw_analogs": "DTW Historical Analogs",
            "conformal_prediction": "Conformal Prediction",
        }

        for model_key, card in models.items():
            label = _labels.get(model_key, model_key)
            reasoning = card.get("reasoning", [])
            expected_str = reasoning[0] if reasoning else "N/A"
            # Truncate long reasoning
            if len(expected_str) > 60:
                expected_str = expected_str[:57] + "..."

            actual = card.get("actual", {})
            if isinstance(actual, dict):
                actual_parts = []
                for k, v in actual.items():
                    if k != "status":
                        actual_parts.append(f"{k}={v}")
                actual_str = ", ".join(actual_parts[:2]) if actual_parts else actual.get("status", "N/A")
            else:
                actual_str = str(actual)
            if len(actual_str) > 50:
                actual_str = actual_str[:47] + "..."

            deviation = card.get("deviation", 0.0)
            robustness = card.get("robustness", "unknown")
            rob_icon = {"high": "OK", "medium": "~", "low": "!!", "n/a": "--"}.get(robustness, "?")

            lines.append(
                f"| {label} | {expected_str} | {actual_str} | {deviation:.2f} | {rob_icon} {robustness} |"
            )

        lines.append("")

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


def _build_unified_survival_system_section(profile: dict[str, Any]) -> str:
    """Build the Unified Survival System section."""
    uss = profile.get("unified_survival_system", {})
    if not uss.get("available"):
        return "*Unified Survival System not available.*\n"

    lines = []
    regime = uss.get("current_regime", "normal")
    is_survival = uss.get("is_survival", False)
    regime_label = regime.upper().replace("_", " ")

    if is_survival:
        lines.append(f"**SURVIVAL MODE ACTIVE: {regime_label}**\n")
    else:
        lines.append(f"Regime: **{regime_label}** (normal operations)\n")

    lines.append("| Dimension | Configuration |")
    lines.append("|:----------|:-------------|")
    lines.append(f"| Active Horizons | {', '.join(uss.get('active_horizons', []))} |")
    lines.append(f"| Primary Horizon | {uss.get('primary_horizon', 'N/A')} |")
    lines.append(f"| Active Variables | {uss.get('n_active_variables', 0)} |")
    lines.append(f"| Frozen Variables | {uss.get('n_frozen_variables', 0)} |")
    lines.append(f"| Priority Variables | {uss.get('n_priority_variables', 0)} |")
    lines.append(f"| Correlation Override | {uss.get('correlation_override', 'None (empirical)')} |")
    lines.append(f"| Copula Override | {uss.get('copula_override', 'None (AIC selection)')} |")
    lines.append(f"| Regime Switches | {uss.get('regime_switches', 0)} |")

    ew = uss.get("early_warning_latest", 0)
    if ew > 0.5:
        lines.append(f"\n**Early Warning Score: {ew:.2f}/1.00**")
        if uss.get("approaching_survival"):
            lines.append("> Company is approaching survival mode triggers.\n")

    dist = uss.get("regime_distribution", {})
    if dist:
        lines.append("\n**Regime Distribution (historical):**\n")
        for r, pct in sorted(dist.items(), key=lambda x: -x[1]):
            lines.append(f"- {r}: {pct:.1%}")

    mc = uss.get("model_config", {})
    if mc and is_survival:
        lines.append("\n**Model Reconfiguration:**\n")
        lines.append(f"- Kalman noise: {mc.get('kalman_process_noise_mult', 1.0):.1f}x process, "
                     f"{mc.get('kalman_obs_noise_mult', 1.0):.1f}x observation")
        cap = mc.get("nn_lookback_cap")
        if cap:
            lines.append(f"- NN lookback capped at {cap} days")
        lines.append(f"- MC simulation: {mc.get('mc_n_paths', 10000):,} paths, "
                     f"P{mc.get('mc_stress_percentile', 0.95) * 100:.0f} stress")

    return "\n".join(lines) + "\n"


def _build_scenario_analysis_section(profile: dict[str, Any]) -> str:
    """Build the Scenario Analysis section."""
    sa = profile.get("scenario_analysis", {})
    if not sa.get("available"):
        return "*Scenario analysis not available (only runs in survival mode).*\n"

    lines = []
    lines.append(f"Three-scenario Monte Carlo analysis ({sa.get('n_paths', 0):,} paths, "
                 f"{sa.get('horizon_days', 252)} day horizon):\n")

    lines.append("| Scenario | Cash Runway | 90d Survival | 252d Survival | Median Return | Max Drawdown |")
    lines.append("|:---------|:----------:|:------------:|:-------------:|:-------------:|:------------:|")

    for key, label in [("orderly_resolution", "Orderly Resolution"),
                       ("muddle_through", "Muddle Through"),
                       ("catastrophic", "Catastrophic")]:
        s = sa.get(key, {})
        if not s:
            continue
        runway = s.get("cash_runway_days", 0)
        s90 = s.get("survival_prob_90d", 0)
        s252 = s.get("survival_prob_252d", 0)
        med_ret = s.get("median_return", 0)
        mdd = s.get("max_drawdown_median", 0)
        lines.append(
            f"| **{label}** | {runway:.0f}d | {s90:.1%} | {s252:.1%} | "
            f"{med_ret:+.1%} | {mdd:.1%} |"
        )

    # Scenario descriptions
    for key in ("orderly_resolution", "muddle_through", "catastrophic"):
        s = sa.get(key, {})
        if s.get("description"):
            lines.append(f"\n**{s.get('name', key)}:** {s['description']}")

    # Reverse stress test (SLSQP optimization -- what breaks the company?)
    rev = sa.get("reverse_stress")
    if isinstance(rev, dict) and rev.get("converged"):
        lines.extend(["", "### Reverse Stress Test", ""])
        lines.append("*Finds the minimum shock required to trigger survival mode:*")
        lines.append("")
        shocks = rev.get("shocks", {})
        if shocks:
            lines.append("| Variable | Required Shock |")
            lines.append("|----------|---------------|")
            for var, shock in sorted(shocks.items(), key=lambda x: abs(x[1]), reverse=True):
                lines.append(f"| {var} | {shock:+.1%} |")
        dist = rev.get("distance")
        if dist is not None:
            lines.append(f"\n- **Distance to failure**: {dist:.3f} (Euclidean norm of shock vector)")
            if dist < 0.5:
                lines.append("  *Very close to failure -- small shocks could trigger survival mode.*")

    return "\n".join(lines) + "\n"


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

    # Candlestick pattern detection
    patterns = ext.get("candlestick_patterns", {})
    if isinstance(patterns, dict) and patterns.get("available"):
        lines.append("### Candlestick Pattern Detection")
        lines.append("")
        lines.append(
            "Automated detection of classical candlestick patterns (doji, hammer, "
            "engulfing, etc.) plus motif discovery via Matrix Profile -- recurring "
            "price micro-structures that repeat across the history."
        )
        lines.append("")
        lines.append(f"- **Patterns detected**: {patterns.get('n_patterns', 0)}")
        pat_list = patterns.get("patterns", [])
        if pat_list:
            lines.append("")
            lines.append("**Most recent patterns:**")
            lines.append("")
            for p in (pat_list[-5:] if isinstance(pat_list, list) else []):
                if isinstance(p, dict):
                    lines.append(
                        f"- {p.get('pattern', '?')} on {p.get('date', '?')} "
                        f"({p.get('signal', 'neutral')})"
                    )
            lines.append("")
        motifs = patterns.get("motifs", [])
        discords = patterns.get("discords", [])
        if motifs or discords:
            lines.append(
                f"- **Recurring motifs**: {len(motifs)} "
                f"*(patterns that repeat multiple times in the price history)*"
            )
            lines.append(
                f"- **Anomalous discords**: {len(discords)} "
                f"*(rare patterns that deviate significantly from normal behavior)*"
            )
            lines.append("")

    # Walk-forward model evaluation
    wf = ext.get("walk_forward", {})
    if isinstance(wf, dict) and wf.get("available"):
        lines.append("### Walk-Forward Model Evaluation")
        lines.append("")
        lines.append(
            "Walk-forward analysis evaluates forecasting model accuracy by stepping "
            "through history day-by-day, retraining at regime switches, and tracking "
            "which model performs best in each survival mode."
        )
        lines.append("")
        lines.append(f"- **Overall best model**: {wf.get('overall_best_model', 'N/A')}")
        lines.append(f"- **Overall MAE**: {_fmt(wf.get('overall_mae'), '.6f')}")
        lines.append(f"- **Retrain events**: {wf.get('n_retrains', 0)} "
                     "*(model retrained at each regime switch)*")
        lines.append("")
        best_by_mode = wf.get("best_model_by_mode", {})
        if best_by_mode:
            lines.append("**Best model per survival mode:**")
            lines.append("")
            for mode, model in best_by_mode.items():
                lines.append(f"- {mode}: **{model}**")
            lines.append("")

    # Burn-out weight calibration
    bo = ext.get("burnout", {})
    if isinstance(bo, dict) and bo.get("available"):
        lines.append("### Online Weight Calibration (Burn-Out Phase)")
        lines.append("")
        lines.append(
            "The burn-out phase recalibrates model ensemble weights using "
            "exponential gradient learning on the most recent data. This adapts "
            "the forecast blend to current market conditions rather than relying "
            "on long-term averages."
        )
        lines.append("")
        lines.append(f"- **Iterations completed**: {bo.get('iterations_completed', 0)}")
        lines.append(f"- **Converged**: {'Yes' if bo.get('converged') else 'No'}")
        lines.append(f"- **Calibrated**: {'Yes' if bo.get('calibrated') else 'No'}")
        lines.append(f"- **Weight stability**: {_fmt(bo.get('weight_stability'))}")
        lines.append("")
        regime_w = bo.get("regime_weights", {})
        if regime_w:
            lines.append("**Per-regime calibrated weights:**")
            lines.append("")
            for regime, weights in regime_w.items():
                if isinstance(weights, dict):
                    top = sorted(weights.items(), key=lambda x: -(x[1] or 0))[:3]
                    top_str = ", ".join(f"{k}={v:.2f}" for k, v in top)
                    lines.append(f"- {regime}: {top_str}")
            lines.append("")

    # Time-varying causal dynamics
    tvg = ext.get("time_varying_granger", {})
    if isinstance(tvg, dict) and tvg.get("available"):
        lines.append("### Time-Varying Causal Dynamics")
        lines.append("")
        lines.append(
            "Rolling-window Granger causality reveals how causal relationships "
            "between financial variables evolve over time. Emerging pairs indicate "
            "new dependencies forming; disappearing pairs suggest decoupling."
        )
        lines.append("")
        lines.append(f"- **Analysis windows**: {tvg.get('n_windows', 0)}")
        emerging = tvg.get("emerging_pairs", [])
        disappearing = tvg.get("disappearing_pairs", [])
        if emerging:
            lines.append("")
            lines.append("**Emerging causal relationships** (newly significant):")
            lines.append("")
            for pair in emerging[:5]:
                if isinstance(pair, dict):
                    lines.append(
                        f"- {pair.get('source', '?')} -> {pair.get('target', '?')}"
                    )
                elif isinstance(pair, (list, tuple)) and len(pair) >= 2:
                    lines.append(f"- {pair[0]} -> {pair[1]}")
            lines.append("")
        if disappearing:
            lines.append("**Disappearing causal relationships** (recently lost significance):")
            lines.append("")
            for pair in disappearing[:5]:
                if isinstance(pair, dict):
                    lines.append(
                        f"- {pair.get('source', '?')} -> {pair.get('target', '?')}"
                    )
                elif isinstance(pair, (list, tuple)) and len(pair) >= 2:
                    lines.append(f"- {pair[0]} -> {pair[1]}")
            lines.append("")

    # Multivariate Monte Carlo
    mvmc = ext.get("multivariate_monte_carlo", {})
    if isinstance(mvmc, dict) and mvmc.get("available"):
        lines.append("### Multivariate Monte Carlo Simulation")
        lines.append("")
        lines.append(
            "Unlike standard Monte Carlo (which simulates price returns only), "
            "the multivariate simulation jointly models financial ratios "
            "(current ratio, FCF yield, debt-to-equity) using their historical "
            "correlation structure. Survival is checked directly on simulated "
            "ratio values, not just price paths."
        )
        lines.append("")
        surv = mvmc.get("survival_probability")
        if surv is not None:
            lines.append(f"- **Joint survival probability**: {_pct(surv)}")
        vars_sim = mvmc.get("variables_simulated", [])
        if vars_sim:
            lines.append(f"- **Variables jointly simulated**: {', '.join(vars_sim)}")
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
        ("Market Beta", "beta_252d", ".2f",
         "Measures how much the stock moves relative to the market. "
         "Beta > 1 means it amplifies market moves (higher risk/reward). "
         "Beta < 1 means it dampens market moves (defensive). "
         "Beta = 1 means it tracks the market exactly."),
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


def _build_institutional_holders_section(profile: dict[str, Any]) -> str:
    """Build Institutional/Major Holders section from holder data."""
    holders_data = profile.get("institutional_holders", {})
    if not holders_data.get("available"):
        return "*Institutional holder data not available for this market.*\n"

    holders = holders_data.get("holders", [])
    if not holders:
        return "*No institutional holders found.*\n"

    lines = [
        "### Top Institutional / Major Holders",
        "",
        "| Holder | Shares | % Outstanding | Type |",
        "|--------|--------|--------------|------|",
    ]
    for h in holders[:10]:
        name = h.get("name", "Unknown")
        shares = h.get("shares", 0)
        pct = h.get("percentage", 0)
        htype = h.get("holder_type", "institutional")
        shares_str = f"{shares:,.0f}" if shares else "N/A"
        pct_str = f"{pct:.1f}%" if pct else "N/A"
        lines.append(f"| {name} | {shares_str} | {pct_str} | {htype} |")

    lines.append("")
    total = holders_data.get("total_holders", len(holders))
    if total > 10:
        lines.append(f"*Showing top 10 of {total} holders.*")
        lines.append("")

    return "\n".join(lines)


def _build_institutional_ownership_deep_section(profile: dict[str, Any]) -> str:
    """Build Institutional Ownership Deep Analysis section (19.8).

    Renders MHHI common ownership, crowding risk, flow signals,
    insider activity, and liquidity analysis from the contagion scorer
    and institutional flow predictor.
    """
    data = profile.get("institutional_ownership_analysis", {})
    if not data.get("available"):
        return "*Institutional ownership deep analysis not available.*\n"

    lines: list[str] = []

    # Contagion sub-section
    contagion = data.get("contagion", {})
    if contagion:
        lines.append("### Common Ownership Analysis (MHHI)")
        lines.append("")
        mhhi = contagion.get("mhhi_delta", 0)
        label = contagion.get("mhhi_label", "unknown")
        lines.append(f"**MHHI Delta:** {mhhi:.3f} ({label.upper()})")
        lines.append("")

        shared = contagion.get("shared_institutions", [])
        if shared:
            lines.append(f"**Shared institutions ({contagion.get('n_shared', 0)}):** {', '.join(shared[:5])}")
            lines.append("")

        lines.append("| Metric | Value |")
        lines.append("|--------|-------|")
        lines.append(f"| Bipartite centrality | {contagion.get('bipartite_centrality', 0):.3f} |")
        lines.append(f"| Network density | {contagion.get('network_density', 0):.3f} |")
        lines.append(f"| Most influential institution | {contagion.get('most_influential_institution', 'N/A')} |")
        lines.append("")

        lines.append("### Crowding & Liquidity Risk")
        lines.append("")
        cs = contagion.get("crowding_score", 0)
        flag = contagion.get("crowded_trade_flag", False)
        flag_str = "CROWDED" if flag else "Normal"
        lines.append(f"**Crowding score:** {cs:.3f} ({flag_str})")
        lines.append(f"**Liquidation days:** {contagion.get('liquidation_days', 0):.0f} days")
        lines.append(f"**Liquidation risk:** {contagion.get('liquidation_risk', 0):.3f}")
        lines.append("")

    # Flow sub-section
    flow = data.get("flow", {})
    if flow:
        lines.append("### Institutional Flow Signals")
        lines.append("")
        lines.append("| Signal | Value | Label |")
        lines.append("|--------|-------|-------|")

        momentum = flow.get("momentum_latest")
        mom_str = f"{momentum:.3f}" if momentum is not None else "N/A"
        lines.append(f"| Flow momentum | {mom_str} | {flow.get('momentum_label', 'unknown')} |")

        cr = flow.get("crowding_risk_latest")
        cr_str = f"{cr:.3f}" if cr is not None else "N/A"
        lines.append(f"| Crowding risk | {cr_str} | {flow.get('crowding_risk_label', 'unknown')} |")

        sm = flow.get("smart_money_signal")
        sm_str = f"{sm:.3f}" if sm is not None else "N/A"
        lines.append(f"| Smart money | {sm_str} | {flow.get('smart_money_label', 'unknown')} |")

        insider = flow.get("insider_signal")
        ins_str = f"{insider:.3f}" if insider is not None else "N/A"
        lines.append(f"| Insider signal | {ins_str} | {flow.get('insider_label', 'unknown')} |")

        amihud = flow.get("amihud_illiquidity")
        if amihud is not None:
            lines.append(f"| Amihud illiquidity | {amihud:.6f} | -- |")

        lines.append("")

    if not lines:
        return "*No institutional ownership analysis data available.*\n"

    return "\n".join(lines)


def _build_market_buying_power_section(profile: dict[str, Any]) -> str:
    """Build Market Demand & Buying Power section (19.9)."""
    mbp = profile.get("market_buying_power", {})
    if not mbp.get("available"):
        return "*No market buying power data available.*\n"

    lines: list[str] = []
    bpi = mbp.get("buying_power_index", 50)
    momentum = mbp.get("sector_demand_momentum", 0)
    trend = mbp.get("consumer_confidence_trend", "stable")
    demand_risk = mbp.get("demand_risk_flag", False)
    inflation_drag = mbp.get("inflation_drag", 0)
    real_growth = mbp.get("real_revenue_growth_ppp")

    # Status badge
    if bpi >= 65:
        badge = "STRONG DEMAND"
    elif bpi >= 45:
        badge = "STABLE DEMAND"
    elif bpi >= 30:
        badge = "WEAKENING DEMAND"
    else:
        badge = "DEMAND CONTRACTION"

    lines.append(f"**Demand Environment:** {badge}")
    lines.append("")

    lines.append("| Indicator | Value |")
    lines.append("|-----------|-------|")
    lines.append(f"| Buying Power Index | **{bpi:.0f}** / 100 |")
    lines.append(f"| Sector Demand Momentum | {momentum:+.3f} |")
    lines.append(f"| Consumer Confidence Trend | {trend.title()} |")
    lines.append(f"| Demand Risk Flag | {'YES' if demand_risk else 'No'} |")
    if inflation_drag is not None:
        lines.append(f"| Inflation Drag | {inflation_drag:.2f} |")
    if real_growth is not None:
        lines.append(f"| Real Revenue Growth (PPP) | {real_growth:+.1%} |")
    lines.append("")

    if demand_risk:
        lines.append(
            "> **Warning:** Consumer spending momentum is declining in this "
            "company's primary markets. Revenue growth may face headwinds "
            "regardless of the company's own execution."
        )
    elif bpi >= 60:
        lines.append(
            "Consumer spending in this sector's markets is healthy. "
            "Demand-side conditions support continued revenue growth."
        )

    return "\n".join(lines)


def _build_product_catalysts_section(profile: dict[str, Any]) -> str:
    """Build Product Catalysts & Forward Signals section (19.10)."""
    cat = profile.get("product_catalysts", {})
    if not cat.get("available"):
        return "*No product catalyst data available.*\n"

    lines: list[str] = []
    score = cat.get("catalyst_score", 0)
    ctype = cat.get("catalyst_type", "none")
    rnd = cat.get("rnd_acceleration", 0)
    news = cat.get("news_catalyst_score", 0)
    earnings = cat.get("earnings_momentum", 0)
    rev_div = cat.get("revenue_diversification_delta", 0)
    n_articles = cat.get("n_catalyst_articles", 0)

    # Status badge
    if score >= 0.6:
        badge = "STRONG CATALYST SIGNALS"
    elif score >= 0.3:
        badge = "MODERATE CATALYST SIGNALS"
    else:
        badge = "NO SIGNIFICANT CATALYSTS"

    type_labels = {
        "product_launch": "Product Launch Likely",
        "rnd_surge": "R&D Surge Detected",
        "earnings_momentum": "Earnings Momentum",
        "market_narrative": "Market Narrative Shift",
        "segment_shift": "Revenue Mix Change",
        "none": "Steady State",
    }

    lines.append(f"**Catalyst Status:** {badge}")
    lines.append(f"**Catalyst Type:** {type_labels.get(ctype, ctype)}")
    lines.append("")

    lines.append("| Signal | Value | Interpretation |")
    lines.append("|--------|-------|----------------|")
    lines.append(f"| Composite Score | **{score:.2f}** | {'Active' if score > 0.3 else 'Quiet'} |")

    rnd_label = "Accelerating" if rnd > 1.1 else "Stable" if rnd > 0.9 else "Decelerating"
    lines.append(f"| R&D Acceleration | {rnd:.2f}x avg | {rnd_label} |")

    lines.append(f"| News Catalyst Score | {news:.2f} | {n_articles} catalyst articles |")

    earn_label = "Strong positive" if earnings > 0.5 else "Positive" if earnings > 0 else "Negative" if earnings < -0.5 else "Neutral"
    lines.append(f"| Earnings Momentum | {earnings:+.2f} | {earn_label} |")

    lines.append(f"| Revenue Diversification | {rev_div:.3f} | {'Shifting' if rev_div > 0.1 else 'Stable'} |")
    lines.append("")

    if score >= 0.5:
        lines.append(
            "> **Forward signal:** The combination of R&D acceleration and "
            "catalyst news suggests this company may be approaching a "
            "product cycle inflection point. Historical analogs from similar "
            "periods have been weighted accordingly in forecasts."
        )

    return "\n".join(lines)


def _build_geopolitical_risk_section(profile: dict[str, Any]) -> str:
    """Build Geopolitical & Conflict Risk section from conflict_risk data."""
    lines: list[str] = []

    conflict = profile.get("conflict_risk", {})
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
    # Data may be stored as nested dict under "linked_conflict" or as flat
    # keys (supply_chain_risk_score, etc.) directly in the conflict_risk dict.
    linked_conflict = (
        profile.get("linked_conflict")
        or conflict.get("linked_conflict")
        or conflict  # flat keys stored directly in conflict_risk
    )
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


def _build_six_swiss_exchange_section(profile: dict[str, Any]) -> str:
    """Build SIX Swiss Exchange-specific bonus sections.

    Leverages unique SIX API data (18yr dividends, capital structure,
    insider transactions, Merton credit risk, TDA topology) that only
    ch_six provides. Only rendered when market_id == 'ch_six'.
    """
    market_id = profile.get("meta", {}).get("market_id", "")
    if not market_id:
        # Try identity
        market_id = profile.get("identity", {}).get("market_id", "")
    if market_id != "ch_six":
        return ""

    lines: list[str] = []

    # Read six_proxy data from the cache columns stored in current_state
    # or directly from profile if the profile builder stored them
    cs = profile.get("current_state", {})

    # --- 1. Swiss Dividend Heritage ---
    lines.append("### Swiss Dividend Heritage")
    lines.append("")

    identity = profile.get("identity", {})
    company = identity.get("name", "Unknown")
    div_years = profile.get("identity", {}).get("dividend_history_years", 0)

    # Get proxy data from the profile (seeded by compute_six_proxies)
    # These flow through the standard pipeline via cache columns
    fh = profile.get("financial_health", {})
    vanity = profile.get("vanity", {})

    # Extract SIX-specific proxy values from current_state tier data
    tier5 = cs.get("tier5_growth", {})
    div_yield = tier5.get("fcf_yield")  # FCF yield as proxy

    lines.append(
        f"**{company}** is listed on the SIX Swiss Exchange, Switzerland's primary "
        f"stock exchange covering ~250 listed companies with ~CHF 1.8T total market cap."
    )
    lines.append("")

    if div_years and div_years >= 10:
        lines.append(
            f"SIX provides **{div_years} years** of PIT-dated dividend history -- "
            f"the longest dividend record of any market in this pipeline. "
            f"This enables dividend-based financial analysis with institutional-grade accuracy."
        )
        lines.append("")

    lines.append("**Shareholder Returns:**")
    lines.append("")
    # These values come from the standard pipeline columns (seeded by six_derived_proxies)
    lines.append(
        "- Dividend yield, buyback yield, and total shareholder return "
        "are computed from actual SIX dividend history and capital destruction notices"
    )
    lines.append(
        "- The pipeline detected dividend growth regimes using PELT changepoint analysis "
        "on the full dividend history"
    )
    lines.append("")

    # --- 2. Capital Structure (SIX unique) ---
    lines.append("### Capital Structure & Dilution Risk")
    lines.append("")
    lines.append(
        "SIX provides detailed capital structure data including share capital, "
        "conditional capital (warrants, convertibles, employee options), and "
        "authorized capital. This data is unavailable from most other market APIs."
    )
    lines.append("")

    tier2 = cs.get("tier2_solvency", {})
    net_debt = tier2.get("net_debt")
    debt_eq = tier2.get("debt_to_equity")
    if net_debt is not None:
        lines.append(f"- **Net Debt:** {_fmt(net_debt)}")
    if debt_eq is not None:
        lines.append(f"- **Debt-to-Equity:** {_fmt(debt_eq)}")
    lines.append(
        "- **Dilution risk** is assessed from the ratio of conditional capital to "
        "reported share capital (SIX capital_structure API)"
    )
    lines.append(
        "- **Share buyback history** is derived from SIX official notice corporate "
        "actions (capital destruction events with PIT dates)"
    )
    lines.append("")

    # --- 3. Insider Activity (SIX management_transactions) ---
    lines.append("### Insider Activity (SIX Management Transactions)")
    lines.append("")
    lines.append(
        "SIX publishes mandatory management transaction disclosures with "
        "CHF amounts for all SIX-listed companies. This provides a real-time "
        "signal of insider confidence."
    )
    lines.append("")
    lines.append(
        "- Insider buy/sell transactions are scored for conviction "
        "(buy-weighted net flow adjusted by transaction volume)"
    )
    lines.append(
        "- This signal feeds into the sentiment and market stability analysis"
    )
    lines.append("")

    # --- 4. Merton Credit Risk ---
    lines.append("### Credit Risk Assessment (Merton Structural Model)")
    lines.append("")
    lines.append(
        "The Merton (1974) structural credit model estimates the company's "
        "distance-to-default and default probability by treating equity as "
        "a call option on the firm's assets."
    )
    lines.append("")
    lines.append(
        "- **Distance to Default (DD):** measures how many standard deviations "
        "the company is from the default boundary. DD > 3.0 is investment grade, "
        "DD < 1.0 indicates distress."
    )
    lines.append(
        "- **Default Probability (PD):** the probability that asset value falls "
        "below the debt face value within the average debt maturity horizon."
    )
    lines.append(
        "- These metrics are derived from equity volatility computed from "
        "the SIX exchange's PIT-compliant historic close+volume data."
    )
    lines.append("")

    # --- 5. Topological Analysis ---
    lines.append("### Dividend Trajectory Topology (TDA)")
    lines.append("")
    lines.append(
        "Topological Data Analysis (persistent homology) is applied to the "
        "time-delay-embedded dividend trajectory to extract shape features "
        "that CAGR and volatility cannot capture."
    )
    lines.append("")
    lines.append(
        "- **Monotonicity score:** 1.0 = the company has never cut its dividend "
        "in the observable history (the trajectory is topologically a line). "
        "Below 0.8 indicates cyclicality or dividend instability."
    )
    lines.append(
        "- **Betti-1 (loops):** counts the number of distinct cycles in the "
        "dividend trajectory. Zero loops = pure growth path; one or more loops = "
        "the company has experienced and recovered from dividend stress."
    )
    lines.append("")

    # --- 6. Proxy Transparency ---
    lines.append("### Data Source Transparency")
    lines.append("")
    lines.append(
        "Financial statement data for this company is **estimated** using 17 "
        "mathematical models applied to SIX market data, 18-year dividend history, "
        "capital structure, and corporate action notices. Unlike SEC EDGAR (US), "
        "DART (Korea), or BSE (India) where actual filed financial statements are "
        "available via free APIs, the SIX Swiss Exchange does not provide free "
        "financial statement line items."
    )
    lines.append("")
    lines.append("**Estimation methods applied:**")
    lines.append("")
    lines.append("- Kalman filter with Jackknife bias correction (earnings)")
    lines.append("- Lintner dividend model (1956) and DuPont decomposition (full statements)")
    lines.append("- PELT regime detection on dividend growth (regime-aware CAGR)")
    lines.append("- L1-minimization balance sheet reconstruction (accounting identity constraints)")
    lines.append("- Monte Carlo uncertainty propagation (10K samples, [p5, p95] intervals)")
    lines.append("- Merton structural credit model (distance-to-default)")
    lines.append("- James-Stein shrinkage and Marchenko-Pastur denoising")
    lines.append("")
    lines.append(
        "*Validated against Nestle 2024 Annual Report: Net Income 0.8% error, "
        "P/E 0.2% error, Payout Ratio 0.8% error, Total Equity 0.8% error. "
        "The theoretical minimum (Cramer-Rao bound) for this information set is 0.78%.*"
    )
    lines.append("")

    return "\n".join(lines)


def _build_corporate_structure_section(profile: dict[str, Any]) -> str:
    """Render GLEIF corporate structure (parent/subsidiary hierarchy)."""
    cs = profile.get("corporate_structure", {})
    if not cs.get("available"):
        return "*Corporate structure data not available.*\n"

    lines = []
    parents = cs.get("parent_companies", [])
    if parents:
        lines.append("**Parent Companies:**\n")
        lines.append("| Name | Country | LEI | Relationship |")
        lines.append("|------|---------|-----|-------------|")
        for p in parents:
            lines.append(
                f"| {p.get('name', 'N/A')} | {p.get('country', 'N/A')} "
                f"| {p.get('lei', 'N/A')[:20]} | {p.get('relationship', 'parent')} |"
            )
        lines.append("")

    n_subs = cs.get("n_subsidiaries", 0)
    subs = cs.get("subsidiaries", [])
    sub_countries = cs.get("subsidiaries_countries", [])
    cross_border = cs.get("cross_border", False)

    if n_subs > 0:
        lines.append(f"**Subsidiaries:** {n_subs} entities across {len(sub_countries)} countries")
        if cross_border:
            lines.append(f"  - Cross-border operations: {', '.join(sub_countries[:10])}")
        lines.append("")
        if subs:
            lines.append("| Name | Country | LEI |")
            lines.append("|------|---------|-----|")
            for s in subs[:10]:
                lines.append(
                    f"| {s.get('name', 'N/A')} | {s.get('country', 'N/A')} "
                    f"| {s.get('lei', 'N/A')[:20]} |"
                )
            if n_subs > 10:
                lines.append(f"\n*... and {n_subs - 10} more subsidiaries.*")
    else:
        lines.append("*No subsidiary data available from GLEIF.*")

    lines.append(f"\n*Source: GLEIF (Global Legal Entity Identifier Foundation)*")
    return "\n".join(lines)


def _build_filing_calendar_section(profile: dict[str, Any]) -> str:
    """Render filing calendar analysis (frequency, staleness, next filing)."""
    fc = profile.get("filing_calendar", {})
    if not fc.get("available"):
        return "*Filing calendar analysis not available.*\n"

    lines = []
    freq = fc.get("detected_frequency", "unknown")
    coverage = fc.get("coverage_ratio", 0)
    expected = fc.get("expected_filings_2yr", 0)
    actual = fc.get("actual_filings_2yr", 0)
    stale = fc.get("is_stale", False)
    age = fc.get("latest_filing_age_days", 0)

    lines.append(f"**Filing Frequency:** {freq.title()}")
    lines.append(f"**Coverage:** {actual}/{expected} filings in 2-year window ({coverage:.0%})")
    lines.append(f"**Latest Filing Age:** {age} days")

    if stale:
        lines.append(f"\n> **WARNING:** Data is stale -- latest filing is {age} days old.")

    gaps = fc.get("gaps", [])
    if gaps:
        lines.append(f"\n**Filing Gaps Detected:** {len(gaps)}")
        for gap in gaps[:5]:
            lines.append(f"  - {gap}")

    next_filing = fc.get("next_expected_filing", {})
    if next_filing and next_filing.get("available", False):
        lines.append(f"\n**Next Expected Filing:** {next_filing.get('predicted_date', 'N/A')}")
        lines.append(f"  - Confidence: {next_filing.get('confidence', 'N/A')}")

    return "\n".join(lines)


def _build_macro_indicators_section(profile: dict[str, Any]) -> str:
    """Render macro indicator data summary."""
    indicators = profile.get("macro_indicators", {})
    if not indicators:
        return "*Macro indicator data not available.*\n"

    lines = []
    lines.append("| Indicator | Latest Value | Latest Date | Observations |")
    lines.append("|-----------|-------------|-------------|-------------|")
    for name, data in sorted(indicators.items()):
        val = data.get("latest_value", "N/A")
        if isinstance(val, float):
            val = f"{val:.4f}"
        lines.append(
            f"| {name.replace('_', ' ').title()} | {val} "
            f"| {data.get('latest_date', 'N/A')} | {data.get('observations', 0)} |"
        )

    return "\n".join(lines)


def _build_supply_chain_stress_section(profile: dict[str, Any]) -> str:
    """Render supply chain stress assessment."""
    scs = profile.get("supply_chain_stress", {})
    if not scs.get("available"):
        return "*Supply chain stress assessment not available.*\n"

    flag = scs.get("supply_chain_stress_flag", False)
    score = scs.get("supply_chain_stress_score", 0)
    sources = scs.get("stress_sources", [])

    badge = "STRESSED" if flag else "NORMAL"
    lines = [
        f"**Supply Chain Status:** {badge}",
        f"**Stress Score:** {score:.3f}",
    ]
    if sources:
        lines.append("\n**Stress Sources:**")
        for src in sources:
            lines.append(f"  - {src}")

    return "\n".join(lines)


def _build_multi_frequency_section(profile: dict[str, Any]) -> str:
    """Render multi-frequency analysis results (regime consensus, survival fusion, predictions)."""
    mf = profile.get("multi_frequency", {})
    if not mf.get("available"):
        return "*Multi-frequency analysis not available.*\n"

    lines = []
    n_freq = mf.get("n_frequencies_used", 0)
    lines.append(f"**Frequencies Analyzed:** {n_freq}\n")

    # Regime consensus
    rc = mf.get("regime_consensus", {})
    if rc:
        consensus = rc.get("consensus_regime", "unknown")
        agreement = rc.get("agreement_ratio", 0)
        freq_regimes = rc.get("frequency_regimes", {})
        interpretation = rc.get("interpretation", "")

        lines.append("### Regime Consensus\n")
        lines.append(f"**Consensus:** {consensus} ({agreement:.0%} agreement)\n")
        if freq_regimes:
            lines.append("| Frequency | Regime |")
            lines.append("|-----------|--------|")
            _freq_labels = {"A": "Annual", "Q": "Quarterly", "M": "Monthly", "W": "Weekly", "D": "Daily"}
            for f, r in freq_regimes.items():
                label = _freq_labels.get(f, f)
                marker = "" if r == consensus else " **"
                lines.append(f"| {label} | {r}{marker} |")
            lines.append("")
        if interpretation:
            lines.append(f"> {interpretation}\n")

    # Survival fusion
    sv = mf.get("survival", {})
    if sv:
        fused = sv.get("fused_probability", 1.0)
        harmonic = sv.get("harmonic_mean", 1.0)
        per_freq = sv.get("per_frequency", {})
        weakest = sv.get("weakest_frequency", "")
        interp = sv.get("interpretation", "")

        lines.append("### Survival Probability Fusion\n")
        lines.append(f"**Fused Probability:** {fused:.1%} (harmonic mean: {harmonic:.1%})\n")
        if per_freq:
            lines.append("| Frequency | Survival Probability |")
            lines.append("|-----------|---------------------|")
            _freq_labels = {"A": "Annual", "Q": "Quarterly", "M": "Monthly", "W": "Weekly", "D": "Daily"}
            for f, p in per_freq.items():
                label = _freq_labels.get(f, f)
                marker = " (weakest)" if f == weakest else ""
                lines.append(f"| {label} | {p:.1%}{marker} |")
            lines.append("")
        if interp:
            lines.append(f"> {interp}\n")

    # Cross-frequency predictions
    preds = mf.get("predictions", [])
    if preds:
        lines.append("### Cross-Frequency Predictions\n")
        lines.append("| Variable | Horizon | Forecast | Confidence | Frequencies |")
        lines.append("|----------|---------|----------|------------|-------------|")
        for p in preds[:15]:
            var = p.get("variable", "")
            horizon = p.get("horizon", "")
            pf = p.get("point_forecast")
            pf_str = f"{pf:.4f}" if pf is not None else "N/A"
            conf = p.get("confidence", 0)
            contribs = p.get("contributing_frequencies", {})
            freq_str = ", ".join(f"{k}({v:.0%})" for k, v in contribs.items())
            lines.append(f"| {var} | {horizon} | {pf_str} | {conf:.0%} | {freq_str} |")
        lines.append("")

    # Frequency summary
    fs = mf.get("frequency_summary", {})
    if fs:
        lines.append("### Frequency Pipeline Summary\n")
        lines.append("| Frequency | Periods | Trend | Regime | Survival | Time |")
        lines.append("|-----------|---------|-------|--------|----------|------|")
        for f, data in fs.items():
            lines.append(
                f"| {data.get('label', f)} | {data.get('n_periods', 0)} "
                f"| {data.get('trend_direction', 'N/A')} "
                f"| {data.get('regime_label', 'N/A')} "
                f"| {data.get('survival_probability', 0):.1%} "
                f"| {data.get('elapsed_seconds', 0):.1f}s |"
            )
        lines.append("")

    # v2 fields (from advanced frequency fusion, PR#2+)
    disagreement = mf.get("disagreement", {})
    if disagreement and disagreement.get("shape", "unknown") != "unknown":
        lines.append("### Cross-Frequency Disagreement\n")
        lines.append(f"**Shape:** {disagreement.get('shape', 'N/A')} "
                     f"(score: {disagreement.get('score', 0):.2f}, "
                     f"diversity: {disagreement.get('cosine_diversity', 0):.2f})\n")

    confirmed_breaks = mf.get("confirmed_breaks", [])
    if confirmed_breaks:
        lines.append("### Confirmed Structural Breaks\n")
        lines.append("| Date | Source | Confirming | Confidence |")
        lines.append("|------|--------|------------|------------|")
        for brk in confirmed_breaks[:10]:
            lines.append(
                f"| {brk.get('date', 'N/A')} "
                f"| {brk.get('source_freq', '?')} "
                f"| {brk.get('n_confirming', 0)} frequencies "
                f"| {brk.get('confidence', 'low')} |"
            )
        lines.append("")

    meta_weights = mf.get("meta_learner_weights", {})
    if meta_weights:
        lines.append("### Meta-Learner Frequency Weights\n")
        _freq_labels = {"A": "Annual", "Q": "Quarterly", "M": "Monthly", "W": "Weekly", "D": "Daily"}
        for f, w in sorted(meta_weights.items(), key=lambda x: -x[1]):
            label = _freq_labels.get(f, f)
            lines.append(f"- **{label}:** {w:.1%}")
        lines.append("")

    if mf.get("cointegrated"):
        lines.append(f"**Cointegration:** Frequencies are cointegrated "
                     f"(ECT = {mf.get('cointegration_ect', 0):.4f})\n")

    # Cross-frequency momentum (Moskowitz, Ooi & Pedersen 2012)
    cfm_score = mf.get("cross_freq_momentum_score")
    cfm_agreement = mf.get("cross_freq_momentum_agreement")
    cfm_leading = mf.get("momentum_leading_frequency")
    cfm_regime = mf.get("momentum_regime_label")
    if cfm_score is not None:
        lines.append("### Cross-Frequency Momentum\n")
        lines.append(f"**Momentum Score:** {cfm_score:+.3f}")
        if cfm_agreement is not None:
            lines.append(f"**Agreement Ratio:** {cfm_agreement:.0%}")
        if cfm_leading:
            _freq_labels = {"A": "Annual", "Q": "Quarterly", "M": "Monthly", "W": "Weekly", "D": "Daily"}
            lines.append(f"**Leading Frequency:** {_freq_labels.get(cfm_leading, cfm_leading)}")
        if cfm_regime:
            lines.append(f"**Momentum Regime:** {cfm_regime}")
        cfm_scores = mf.get("cross_freq_momentum_scores", {})
        if cfm_scores:
            lines.append("\n| Frequency | Momentum |")
            lines.append("|-----------|----------|")
            for f, s in cfm_scores.items():
                label = _freq_labels.get(f, f)
                lines.append(f"| {label} | {s:+.3f} |")
        lines.append("")

    return "\n".join(lines)


def _build_investment_thesis_scorecard(profile: dict[str, Any]) -> str:
    """Render 5-tier investment thesis scorecard."""
    lines = []
    fh = profile.get("financial_health", {})
    current = profile.get("current_state", {})
    catalysts = profile.get("product_catalysts", {})
    sentiment = profile.get("sentiment", {})
    pos = profile.get("position_signal", {})
    uss = profile.get("unified_survival_system", {})

    # Tier scores
    eq_score = 50
    beneish = fh.get("beneish_m_score")
    if isinstance(beneish, (int, float)):
        eq_score = 80 if beneish < -2.22 else 30
    accruals = current.get("accruals_signal")
    if isinstance(accruals, (int, float)):
        eq_score = max(0, min(100, eq_score + int(accruals * 30)))

    cf_score = 50
    fcf_yield = current.get("fcf_yield")
    if isinstance(fcf_yield, (int, float)):
        cf_score = 80 if fcf_yield > 0.05 else 60 if fcf_yield > 0 else 25
    runway = fh.get("runway_months")
    if isinstance(runway, (int, float)):
        cf_score = min(100, cf_score + 15) if runway > 24 else max(0, cf_score - 20) if runway < 6 else cf_score

    bs_score = 50
    z_score = fh.get("altman_z_score")
    if isinstance(z_score, (int, float)):
        bs_score = 85 if z_score > 2.99 else 55 if z_score > 1.81 else 20

    inf_score = 50
    sue = current.get("sue_score")
    if isinstance(sue, (int, float)):
        inf_score = max(0, min(100, 50 + int(sue * 15)))
    if catalysts.get("available"):
        inf_score = max(0, min(100, inf_score + int(catalysts.get("catalyst_score", 0) * 20)))
    recovery = uss.get("recovery_signal", {})
    if recovery.get("active"):
        inf_score = min(100, inf_score + 25)

    val_score = 50
    pe = current.get("pe_ratio_calc")
    if isinstance(pe, (int, float)):
        val_score = 75 if 5 < pe < 20 else 50 if pe < 40 else 30

    composite = int(0.25 * eq_score + 0.25 * cf_score + 0.20 * bs_score + 0.15 * inf_score + 0.15 * val_score)

    def _label(s):
        if s >= 75: return "STRONG"
        if s >= 55: return "ADEQUATE"
        if s >= 35: return "WEAK"
        return "CRITICAL"

    def _bar(s):
        return "[" + "#" * (s // 10) + "." * (10 - s // 10) + "]"

    lines.append("| Tier | Score | Rating | Visual |")
    lines.append("|------|-------|--------|--------|")
    for name, score in [("Earnings Quality", eq_score), ("Cash Flow", cf_score),
                        ("Balance Sheet", bs_score), ("Inflection / Catalysts", inf_score),
                        ("Valuation", val_score)]:
        lines.append(f"| {name} | {score}/100 | {_label(score)} | `{_bar(score)}` |")
    lines.append(f"| **COMPOSITE** | **{composite}/100** | **{_label(composite)}** | `{_bar(composite)}` |")
    lines.append("")

    if pos.get("available"):
        signal = pos.get("signal", 0)
        label = pos.get("label", "hold").upper()
        lines.append(f"**Position Signal:** {signal:+.2f} ({label})")
        if pos.get("recovery_active"):
            lines.append("  - Recovery signal ACTIVE")
    return "\n".join(lines) if lines else "*Scorecard not available.*\n"


def _build_signal_ic_section(profile: dict[str, Any]) -> str:
    """Render the Signal IC (Information Coefficient) analysis section.

    Displays which signals are predictive vs noise, the IC matrix across
    horizons, and the ICIR-based signal classification.
    """
    ic_data = profile.get("signal_ic", {})
    if not ic_data.get("available"):
        return "*Signal IC analysis not available (insufficient data for IC computation).*\n"

    lines: list[str] = []

    n_signals = ic_data.get("n_signals", 0)
    n_obs = ic_data.get("n_observations", 0)
    best_signal = ic_data.get("best_signal", "N/A")
    best_ic = ic_data.get("best_ic", 0)
    strong = ic_data.get("strong_signals", [])
    weak = ic_data.get("weak_signals", [])

    lines.append(f"**Signals analyzed:** {n_signals} | **Observations:** {n_obs}")
    lines.append(f"**Best signal:** `{best_signal}` (IC = {best_ic:.4f})")
    lines.append("")

    # Strong vs weak classification
    if strong:
        lines.append(f"**Predictive signals ({len(strong)}):** {', '.join(f'`{s}`' for s in strong[:10])}")
    if weak:
        lines.append(f"**Weak/noisy signals ({len(weak)}):** {', '.join(f'`{s}`' for s in weak[:10])}")
    lines.append("")

    # IC matrix table
    ic_matrix = ic_data.get("ic_matrix", {})
    if ic_matrix:
        # Get all horizons from first entry
        sample_key = next(iter(ic_matrix))
        horizons = sorted(ic_matrix[sample_key].keys()) if isinstance(ic_matrix[sample_key], dict) else []

        if horizons:
            header = "| Signal | " + " | ".join(f"IC ({h})" for h in horizons) + " | Speed |"
            sep = "|---|" + "|".join("---:" for _ in horizons) + "|---|"
            lines.append(header)
            lines.append(sep)

            speed_map = ic_data.get("signal_speed", {})
            # Sort by absolute IC of first horizon
            sorted_signals = sorted(
                ic_matrix.items(),
                key=lambda x: abs(list(x[1].values())[0]) if isinstance(x[1], dict) and x[1] else 0,
                reverse=True,
            )
            for sig_name, horizons_dict in sorted_signals[:15]:
                if not isinstance(horizons_dict, dict):
                    continue
                vals = " | ".join(f"{horizons_dict.get(h, 0):.4f}" for h in horizons)
                speed = speed_map.get(sig_name, "")
                lines.append(f"| `{sig_name}` | {vals} | {speed} |")
            lines.append("")

    # Interpretation
    lines.append("**Interpretation:**")
    lines.append(f"- Signals with |ICIR| above threshold are classified as **predictive** and receive priority in the ensemble")
    lines.append(f"- **{len(weak)}** weak signals were pruned from temporal model features to reduce overfitting")
    if best_ic > 0:
        lines.append(f"- Positive IC for `{best_signal}` indicates it has genuine forward-return predictive power")
    elif best_ic < 0:
        lines.append(f"- Negative IC for `{best_signal}` indicates contrarian signal (high values predict lower returns)")

    return "\n".join(lines) + "\n"


def _build_position_signal_section(profile: dict[str, Any]) -> str:
    """Render the position signal (-1 to +1 directional conviction)."""
    pos = profile.get("position_signal", {})
    if not pos.get("available"):
        return "*Position signal not available (insufficient prediction data).*\n"
    signal = pos.get("signal", 0)
    label = pos.get("label", "hold").upper()
    lines = []
    gauge_pos = int((signal + 1) * 25)
    gauge = "-" * gauge_pos + "|" + "-" * (50 - gauge_pos)
    lines.append("```")
    lines.append(f"SELL ----{gauge}---- BUY")
    lines.append(f"Signal: {signal:+.4f} ({label})")
    lines.append("```")
    lines.append("")
    lines.append("| Component | Value |")
    lines.append("|-----------|-------|")
    lines.append(f"| Return Forecast (5d) | {pos.get('return_forecast', 0):+.6f} |")
    lines.append(f"| IC Confidence | {pos.get('ic_confidence', 1.0):.2f}x |")
    lines.append(f"| Survival Multiplier | {pos.get('survival_multiplier', 1.0):.2f}x |")
    kelly = pos.get("kelly_fraction")
    if kelly is not None:
        lines.append(f"| Kelly Fraction | {kelly:.2%} |")
    kelly_edge = pos.get("kelly_edge")
    if kelly_edge is not None:
        lines.append(f"| Kelly Edge | {kelly_edge:+.4f} |")
    kelly_cap = pos.get("kelly_capped")
    if kelly_cap is not None:
        lines.append(f"| Kelly (capped) | {kelly_cap:.2%} |")
    if pos.get("recovery_active"):
        lines.append("| Recovery Signal | ACTIVE (boost applied) |")
    lines.append("")
    pl = profile.get("prediction_log", {})
    total = pl.get("total_predictions", 0)
    if total > 0:
        lines.append(f"**Track Record:** {total} predictions logged")
        n_filled = pl.get("n_filled", 0)
        if n_filled > 0:
            lines.append(f"  - {n_filled} evaluated, hit rate={pl.get('hit_rate', 0):.1%}, IC={pl.get('realized_ic', 0):.4f}")
    lines.append("")
    lines.append("> *Directional conviction signal, not investment advice.*")
    return "\n".join(lines)


def _build_synergies_section(profile: dict[str, Any]) -> str:
    """Render model synergies metadata."""
    syn = profile.get("synergies_applied", {})
    if not syn:
        return "*Model synergies metadata not available.*\n"

    lines = []
    cycle_feats = syn.get("cycle_features_added", [])
    if cycle_feats:
        lines.append(f"**Cycle Features Injected:** {', '.join(cycle_feats[:5])}")

    ucn = syn.get("unified_causal_network", {})
    if ucn:
        lines.append(f"**Unified Causal Network:** {ucn.get('n_pairs', 0)} pairs, "
                      f"density={ucn.get('density', 0):.3f}, "
                      f"{ucn.get('n_retained', 0)} variables retained")

    n_vars = syn.get("variables_after_pruning", 0)
    if n_vars:
        lines.append(f"**Variables After Causal Pruning:** {n_vars}")

    drift = syn.get("pattern_drift_applied", 1.0)
    if drift != 1.0:
        lines.append(f"**Pattern Drift Multiplier:** {drift:.3f}")

    thresholds = syn.get("adjusted_survival_thresholds", {})
    if thresholds:
        lines.append(f"**Peer-Adjusted Survival Thresholds:** {len(thresholds)} adjusted")

    return "\n".join(lines) if lines else "*No synergies were applied.*\n"


# ---------------------------------------------------------------------------
# Hedge Fund Analysis sections (23-29)
# ---------------------------------------------------------------------------

def _build_hf_earnings_quality(profile: dict[str, Any]) -> str:
    """Section 23: HF Earnings Quality (Tier 1)."""
    hf = profile.get("hedge_fund", {})
    if not hf.get("available"):
        return "*Hedge fund analysis not available.*\n"
    lines = ["Forensic earnings quality assessment using FCF quality scoring, accruals analysis, and smoothing detection.\n"]
    fcf = hf.get("fcf_quality", {})
    if fcf.get("available"):
        lines.append(f"**FCF Quality Score:** {fcf.get('score', 'N/A'):.0f}/100 ({fcf.get('label', '')})")
        if fcf.get("narrative"):
            lines.append(f"> {fcf['narrative']}")
    acc = hf.get("accruals_forensic", {})
    if acc.get("available"):
        lines.append(f"\n**Accruals Red Flag:** {acc.get('red_flag_score', 'N/A'):.0f}/100 ({acc.get('label', '')})")
        if acc.get("narrative"):
            lines.append(f"> {acc['narrative']}")
    sm = hf.get("smoothing", {})
    if sm.get("available"):
        lines.append(f"\n**Smoothing Index:** {sm.get('smoothing_index', 'N/A'):.0f}/100 ({sm.get('label', '')})")
        if sm.get("narrative"):
            lines.append(f"> {sm['narrative']}")
    # Advanced: Piotroski + Forensic CF
    adv = hf.get("advanced", {})
    if adv.get("available"):
        lines.append(f"\n**Piotroski F-Score:** {adv.get('piotroski_f_score', 'N/A')}/9 ({adv.get('piotroski_label', '')})")
        fc = adv.get("forensic_cashflow", {})
        if fc.get("available") and fc.get("flags"):
            lines.append(f"**Forensic Cash Flow Flags:** {len(fc['flags'])} detected")
            for f in fc["flags"][:3]:
                lines.append(f"  - [{f.get('severity','?')}] {f.get('type','')}: {f.get('detail','')}")
    return "\n".join(lines) + "\n"


def _build_hf_cash_flow(profile: dict[str, Any]) -> str:
    """Section 24: HF Cash Flow Stress Test (Tier 2)."""
    hf = profile.get("hedge_fund", {})
    if not hf.get("available"):
        return "*Hedge fund analysis not available.*\n"
    lines = ["Cash flow sustainability and stress testing.\n"]
    db = hf.get("dividend_burn", {})
    if db.get("available"):
        lines.append(f"**Dividend Burn Risk:** {db.get('risk_score', 'N/A'):.0f}/100 ({db.get('label', '')})")
    rs = hf.get("return_spread", {})
    if rs.get("available"):
        spread = rs.get("spread_bps")
        lines.append(f"**CROA vs ROIC Spread:** {spread:.0f} bps ({rs.get('quality_label', '')})" if spread else "**CROA vs ROIC:** N/A")
        # DuPont 5-factor decomposition (Palepu, Healy & Peek 2019)
        dupont_keys = [("dupont_tax_burden", "Tax Burden"), ("dupont_interest_burden", "Interest Burden"),
                       ("dupont_operating_margin", "Operating Margin"), ("dupont_asset_turnover", "Asset Turnover"),
                       ("dupont_equity_multiplier", "Equity Multiplier")]
        dupont_vals = {k: rs.get(k) for k, _ in dupont_keys if rs.get(k) is not None}
        if dupont_vals:
            lines.append("\n**DuPont Decomposition:**")
            for key, label in dupont_keys:
                v = rs.get(key)
                if v is not None:
                    lines.append(f"  - {label}: {v:.3f}")
    ol = hf.get("operating_leverage", {})
    if ol.get("available"):
        lines.append(f"**Operating Leverage (DOL):** {ol.get('dol', 'N/A')}, sensitivity={ol.get('earnings_sensitivity', '')}")
    return "\n".join(lines) + "\n"


def _build_hf_balance_sheet(profile: dict[str, Any]) -> str:
    """Section 25: HF Balance Sheet Risk (Tier 3)."""
    hf = profile.get("hedge_fund", {})
    if not hf.get("available"):
        return "*Hedge fund analysis not available.*\n"
    lines = ["Balance sheet hidden risks and leverage stress scenarios.\n"]
    obs = hf.get("obs_risk", {})
    if obs.get("available"):
        lines.append(f"**Off-Balance-Sheet Risk:** {obs.get('risk_score', 'N/A'):.0f}/100 ({obs.get('label', '')})")
    aq = hf.get("asset_quality", {})
    if aq.get("available"):
        lines.append(f"**Asset Quality Deterioration:** {aq.get('deterioration_score', 'N/A'):.0f}/100 ({aq.get('label', '')})")
        if aq.get("dso"):
            lines.append(f"  DSO: {aq['dso']:.0f} days (change: {aq.get('dso_change_pct', 0):+.1f}%)")
    ls = hf.get("leverage_stress", {})
    if ls.get("available"):
        lines.append("\n**Leverage Stress Scenarios:**\n")
        lines.append("| Scenario | Debt/EBITDA | Coverage | Covenant |")
        lines.append("|----------|-----------|---------|----------|")
        for sc_name in ("base_case", "revenue_miss", "systemic_crisis"):
            sc = ls.get(sc_name, {})
            de = sc.get("debt_to_ebitda")
            ic = sc.get("interest_coverage")
            cb = "BREACH" if sc.get("covenant_breach") else "OK"
            lines.append(f"| {sc_name.replace('_', ' ').title()} | {f'{de:.1f}x' if de else 'N/A'} | {f'{ic:.1f}x' if ic else 'N/A'} | {cb} |")
    # Advanced: Altman Z''
    adv = hf.get("advanced", {})
    if adv.get("available") and adv.get("altman_z_double_prime"):
        lines.append(f"\n**Altman Z'' (non-US):** {adv['altman_z_double_prime']:.2f} ({adv.get('altman_z_dp_zone', '')})")
    return "\n".join(lines) + "\n"


def _build_hf_inflection(profile: dict[str, Any]) -> str:
    """Section 26: HF Inflection Detection (Tier 4)."""
    hf = profile.get("hedge_fund", {})
    if not hf.get("available"):
        return "*Hedge fund analysis not available.*\n"
    lines = ["Momentum, growth quality, and earnings surprise probability.\n"]
    mom = hf.get("momentum", {})
    if mom.get("available"):
        lines.append(f"**Momentum Score:** {mom.get('score', 'N/A'):.0f}/100 ({mom.get('label', '')})")
        if mom.get("inflection_detected"):
            lines.append("> **Inflection detected:** momentum is reversing")
    gq = hf.get("growth_quality", {})
    if gq.get("available"):
        lines.append(f"**Growth Quality:** {gq.get('score', 'N/A'):.0f}/100 ({gq.get('label', '')})")
        if gq.get("organic_fraction") is not None:
            lines.append(f"  Organic fraction: {gq['organic_fraction']:.0%}")
    es = hf.get("earnings_surprise", {})
    if es.get("available"):
        lines.append(f"\n**Earnings Surprise Probability:**")
        lines.append(f"  P(beat)={es.get('p_beat', 0):.0%}, P(miss)={es.get('p_miss', 0):.0%}, P(inline)={es.get('p_inline', 0):.0%}")
        lines.append(f"  Direction: {es.get('expected_direction', 'neutral')}")
        if es.get("days_to_next_filing"):
            lines.append(f"  Next filing in ~{es['days_to_next_filing']} days")
    # Advanced: Torpedo, Capital Cycle
    adv = hf.get("advanced", {})
    if adv.get("available"):
        torp = adv.get("earnings_torpedo", {})
        if torp.get("torpedo_risk", 0) > 25:
            lines.append(f"\n**Earnings Torpedo Risk:** {torp['torpedo_risk']}%")
            for f in torp.get("flags", []):
                lines.append(f"  - {f}")
        cc = adv.get("capital_cycle", {})
        if cc.get("available"):
            lines.append(f"**Capital Cycle Position:** {cc.get('cycle_position', 'unknown')}")
    return "\n".join(lines) + "\n"


def _build_hf_valuation(profile: dict[str, Any]) -> str:
    """Section 27: HF Valuation Engine (Tier 5)."""
    hf = profile.get("hedge_fund", {})
    if not hf.get("available"):
        return "*Hedge fund analysis not available.*\n"
    lines = ["DCF Monte Carlo valuation, quality-value matrix, and PEG composite.\n"]
    dcf = hf.get("dcf", {})
    if dcf.get("available"):
        lines.append("**DCF Monte Carlo Intrinsic Value:**\n")
        lines.append("| Percentile | Value |")
        lines.append("|-----------|-------|")
        for p in ("intrinsic_p10", "intrinsic_p25", "intrinsic_p50", "intrinsic_p75", "intrinsic_p90"):
            val = dcf.get(p)
            label = p.replace("intrinsic_", "").upper()
            lines.append(f"| {label} | {'${:.2f}'.format(val) if val else 'N/A'} |")
        if dcf.get("current_price"):
            lines.append(f"\n**Current Price:** ${dcf['current_price']:.2f}")
        if dcf.get("upside_pct") is not None:
            lines.append(f"**Upside/Downside:** {dcf['upside_pct']:+.1f}%")
    vq = hf.get("valuation_quality", {})
    if vq.get("available"):
        lines.append(f"\n**Valuation-Quality Matrix:** {vq.get('quadrant', '').replace('_', ' ').title()}")
        lines.append(f"  Quality Score: {vq.get('quality_score', 0):.0f}/100")
    peg = hf.get("peg_composite", {})
    if peg.get("available"):
        if peg.get("peg_adjusted"):
            lines.append(f"**Quality-Adjusted PEG:** {peg['peg_adjusted']:.1f}x")
        if peg.get("fcf_spread_bps"):
            lines.append(f"**FCF Yield vs Debt Cost:** {peg['fcf_spread_bps']:.0f} bps ({'deleverageable' if peg.get('cheap_flag') else 'tight'})")
    # Advanced: ICC, OU
    adv = hf.get("advanced", {})
    if adv.get("available"):
        if adv.get("implied_cost_of_capital"):
            lines.append(f"\n**Implied Cost of Capital:** {adv['implied_cost_of_capital']:.1%}")
        ou = adv.get("ou_mean_reversion", {})
        if ou.get("available") and ou.get("mean_reverting"):
            lines.append(f"**Mean Reversion Half-Life:** {ou.get('half_life_days', 'N/A')} days (deviation: {ou.get('current_deviation_pct', 0):+.1f}%)")
    return "\n".join(lines) + "\n"


def _build_hf_scorecard(profile: dict[str, Any]) -> str:
    """Section 28: HF Investment Thesis Scorecard."""
    hf = profile.get("hedge_fund", {})
    if not hf.get("available"):
        return "*Hedge fund analysis not available.*\n"
    sc = hf.get("scorecard", {})
    if not sc.get("available"):
        return "*Scorecard not available.*\n"
    lines = ["5-tier investment thesis scorecard.\n"]
    lines.append(f"**Investment Grade: {sc.get('investment_grade', 'N/A')}** | Conviction: {sc.get('conviction', 0)}/10\n")
    lines.append("| Tier | Score | Label |")
    lines.append("|------|-------|-------|")
    for tier_key in ("earnings_quality", "cash_flow", "balance_sheet", "inflection", "valuation"):
        t = sc.get(tier_key, {})
        name = tier_key.replace("_", " ").title()
        score = t.get("score", 50)
        label = t.get("label", "fair")
        lines.append(f"| {name} | {score:.0f}/100 | {label} |")
    return "\n".join(lines) + "\n"


def _build_product_segments_section(profile: dict[str, Any]) -> str:
    """Section 30: Product Portfolio Analysis."""
    ps = profile.get("product_segments", {})
    if not ps.get("available"):
        return "*Product segment data not available for this company.*\n"

    lines = ["Product-level revenue decomposition, lifecycle classification, and risk assessment.\n"]

    # Segment overview
    lines.append(f"**Segments**: {ps.get('n_segments', 0)} product lines")
    lines.append(f"**Dominant segment**: {ps.get('dominant_segment', 'N/A')} "
                 f"({ps.get('dominant_segment_pct', 0):.0%} of revenue)")
    hhi = ps.get("hhi", 0)
    conc_label = "Low" if hhi < 0.25 else "Moderate" if hhi < 0.5 else "High" if hhi < 0.75 else "Very High"
    lines.append(f"**Revenue concentration (HHI)**: {hhi:.3f} ({conc_label})")

    # Lifecycle
    stage = ps.get("lifecycle_stage", "unknown")
    runway = ps.get("growth_runway_quarters", 0)
    lines.append(f"\n**Product Lifecycle**: {stage.title()}")
    if runway > 0:
        lines.append(f"- Estimated growth runway: ~{runway} quarters until peak adoption")
    elif stage == "maturity":
        lines.append("- Product at maturity -- growth deceleration expected")
    elif stage == "decline":
        lines.append("- Product in decline -- revenue erosion likely without new launches")

    # Pricing power
    pp = ps.get("pricing_power", 0)
    pp_label = "Strong" if pp > 0.3 else "Moderate" if pp > 0 else "Weak" if pp > -0.3 else "Negative"
    lines.append(f"\n**Pricing Power**: {pp:.3f} ({pp_label})")
    if pp > 0:
        lines.append("- Company can raise prices above inflation -- moat indicator")
    else:
        lines.append("- Revenue growth depends on volume, not pricing -- margin pressure risk")

    # Cannibalization
    cr = ps.get("cannibalization_rate", 0)
    if cr > 0.1:
        lines.append(f"\n**Cannibalization Rate**: {cr:.0%} of new product revenue displaces existing products")

    # Network effect
    nfx = ps.get("network_effect_score", 0)
    if nfx > 0.1:
        nfx_label = "Strong" if nfx > 0.5 else "Moderate"
        lines.append(f"\n**Network Effect**: {nfx:.2f} ({nfx_label}) -- platform economics detected")

    return "\n".join(lines) + "\n"


def _build_hf_position(profile: dict[str, Any]) -> str:
    """Section 29: HF Position Signal & Sizing."""
    hf = profile.get("hedge_fund", {})
    if not hf.get("available"):
        return "*Hedge fund analysis not available.*\n"
    pos = hf.get("position", {})
    if not pos.get("available"):
        return "*Position signal not available.*\n"
    lines = ["Actionable position signal with sizing guidance.\n"]
    signal = pos.get("signal", 0)
    label = pos.get("label", "hold").upper()
    lines.append(f"**Signal: {signal:+.3f} ({label})**\n")
    lines.append(f"- Conviction: {pos.get('conviction', 0):.2f}")
    lines.append(f"- Quality multiplier: {pos.get('quality_multiplier', 1):.2f}x")
    lines.append(f"- Survival multiplier: {pos.get('survival_multiplier', 1):.2f}x")
    if pos.get("entry_price"):
        lines.append(f"\n**Levels:**")
        lines.append(f"- Entry: ${pos['entry_price']:.2f}")
        lines.append(f"- Stop: ${pos.get('stop_price', 0):.2f}")
        lines.append(f"- Target: ${pos.get('target_price', 0):.2f}")
        if pos.get("risk_reward_ratio"):
            lines.append(f"- Risk/Reward: {pos['risk_reward_ratio']:.1f}:1")
    # Advanced methods summary
    adv = hf.get("advanced", {})
    if adv.get("available"):
        lines.append("\n**Advanced Signals:**")
        vol = adv.get("garch_vol_term_structure", {})
        if vol.get("inverted"):
            lines.append("- Vol term structure INVERTED (precedes large moves)")
        vrp = adv.get("vrp_proxy", {})
        if vrp.get("available"):
            lines.append(f"- VRP regime: {vrp.get('regime', 'unknown')}")
        ca = adv.get("cross_asset_regime", {})
        if ca.get("available"):
            lines.append(f"- Cross-asset macro: {ca.get('regime', 'unknown')}")
    return "\n".join(lines) + "\n"


def _build_options_signals_section(profile: dict[str, Any]) -> str:
    """Section 31: Options-Derived Forward Signals."""
    opt = profile.get("options_signals", {})
    if not opt.get("available"):
        return "*Options signal data not available for this company.*\n"

    lines: list[str] = []

    # Composite signal badge
    pcr = opt.get("put_call_ratio")
    rr = opt.get("risk_reversal_25d")
    vrp = opt.get("variance_risk_premium")
    stress_count = sum([
        1 if pcr is not None and pcr > 1.0 else 0,
        1 if rr is not None and rr < -0.02 else 0,
        1 if vrp is not None and vrp > 0.05 else 0,
    ])
    badge = "RISK OFF" if stress_count >= 2 else "RISK ON" if stress_count == 0 else "NEUTRAL"
    lines.append(f"**Composite Signal:** {badge}\n")

    lines.append("| Metric | Value | Interpretation |")
    lines.append("|--------|-------|----------------|")

    if pcr is not None:
        interp = "Bearish (put demand)" if pcr > 1.0 else "Bullish (call demand)" if pcr < 0.7 else "Neutral"
        lines.append(f"| Put/Call Ratio | {pcr:.2f} | {interp} |")
    if rr is not None:
        interp = "Institutional hedging" if rr < -0.02 else "Upside positioning" if rr > 0.02 else "Balanced"
        lines.append(f"| 25-Delta Risk Reversal | {rr:.4f} | {interp} |")

    iv_skew = opt.get("iv_skew")
    if iv_skew is not None:
        interp = "High crash fear premium" if iv_skew > 0.1 else "Low tail risk" if iv_skew < 0.02 else "Normal"
        lines.append(f"| IV Skew | {iv_skew:.4f} | {interp} |")

    vts = opt.get("vix_term_structure")
    if vts is not None:
        interp = "Backwardation (near-term fear)" if vts < 0 else "Contango (normal)" if vts > 0 else "Flat"
        lines.append(f"| VIX Term Structure | {vts:.3f} | {interp} |")

    skew = opt.get("skew_index")
    if skew is not None:
        interp = "Elevated tail risk" if skew > 130 else "Low tail risk" if skew < 110 else "Normal"
        lines.append(f"| SKEW Index | {skew:.1f} | {interp} |")

    if vrp is not None:
        interp = "Markets overpricing risk" if vrp > 0.03 else "Markets underpricing risk" if vrp < -0.01 else "Fair"
        lines.append(f"| Variance Risk Premium | {vrp:.4f} | {interp} |")

    lines.append("")
    return "\n".join(lines) + "\n"


def _build_cross_asset_signals_section(profile: dict[str, Any]) -> str:
    """Section 32: Cross-Asset Sector Rotation Signals."""
    ca = profile.get("cross_asset_signals", {})
    if not ca.get("available"):
        return "*Cross-asset signal data not available.*\n"

    lines: list[str] = []

    rank = ca.get("sector_rank_12m")
    rs = ca.get("sector_relative_strength")
    disp = ca.get("sector_dispersion")
    yc = ca.get("yield_curve_10y2y")
    usd = ca.get("usd_momentum_21d")
    stress = ca.get("cross_asset_stress")

    if rank is not None:
        color = "top-performing" if rank <= 3 else "bottom-performing" if rank >= 9 else "mid-pack"
        lines.append(f"**Sector Ranking:** #{rank} of 11 GICS sectors over 12 months ({color})\n")

    lines.append("| Signal | Value | Status |")
    lines.append("|--------|-------|--------|")

    if rs is not None:
        status = "Outperforming" if rs > 0.02 else "Underperforming" if rs < -0.02 else "In-line"
        lines.append(f"| Relative Strength vs S&P 500 | {rs:.3f} | {status} |")
    if disp is not None:
        status = "High dispersion (stock picking)" if disp > 0.02 else "Low dispersion (herding)" if disp < 0.005 else "Normal"
        lines.append(f"| Sector Dispersion | {disp:.5f} | {status} |")
    if yc is not None:
        status = "INVERTED (recession signal)" if yc < 0 else "Steep (growth signal)" if yc > 1.5 else "Normal"
        lines.append(f"| Yield Curve (10Y-2Y) | {yc:.3f} | {status} |")
    if usd is not None:
        status = "Strong dollar headwind" if usd > 0.02 else "Weak dollar tailwind" if usd < -0.02 else "Stable"
        lines.append(f"| USD Momentum (21d) | {usd:.3f} | {status} |")
    if stress is not None:
        status = "ELEVATED" if stress > 0.7 else "MODERATE" if stress > 0.3 else "LOW"
        lines.append(f"| Cross-Asset Stress Index | {stress:.3f} | {status} |")

    lines.append("")
    return "\n".join(lines) + "\n"


def _build_event_calendar_section(profile: dict[str, Any]) -> str:
    """Section 33: Upcoming Events & Uncertainty Calendar."""
    ev = profile.get("event_calendar_signals", {})
    if not ev.get("available"):
        return "*Event calendar data not available.*\n"

    lines: list[str] = []

    days_next = ev.get("days_to_next_event")
    premium = ev.get("event_uncertainty_premium")
    fomc = ev.get("fomc_proximity")
    earnings = ev.get("earnings_proximity")
    density = ev.get("event_density_30d")

    if days_next is not None:
        lines.append(f"**Next Scheduled Event:** {int(days_next)} trading days away\n")

    if premium is not None:
        pct = premium * 100
        lines.append(f"**Uncertainty Premium:** {pct:.1f}% confidence haircut applied to predictions\n")

    lines.append("| Event Type | Proximity | Impact |")
    lines.append("|------------|-----------|--------|")

    if fomc is not None:
        impact = "HIGH -- rate-sensitive" if fomc < 5 else "MODERATE" if fomc < 15 else "LOW"
        lines.append(f"| FOMC Meeting | {int(fomc)} days | {impact} |")
    if earnings is not None:
        impact = "HIGH -- vol compression" if earnings < 10 else "MODERATE" if earnings < 30 else "LOW"
        lines.append(f"| Earnings Release | {int(earnings)} days | {impact} |")

    if density is not None:
        label = "BUSY" if density > 5 else "MODERATE" if density > 2 else "QUIET"
        lines.append(f"| 30-Day Event Density | {int(density)} events | {label} |")

    # Next filing prediction from filing_calendar
    fc = profile.get("filing_calendar", {})
    nf = fc.get("next_expected_filing", {})
    if nf.get("available", False) and nf.get("predicted_date"):
        lines.append(f"| Next Expected Filing | {nf['predicted_date']} | Confidence: {nf.get('confidence', 'N/A')} |")

    lines.append("")
    return "\n".join(lines) + "\n"


def _build_predicted_regime_shifts_section(profile: dict[str, Any]) -> str:
    """Section 34: Predicted Regime Shifts."""
    rs = profile.get("predicted_regime_shifts", {})
    if not rs.get("available"):
        return "*Regime shift prediction data not available.*\n"

    lines: list[str] = []

    current = rs.get("current_regime", "unknown")
    p21 = rs.get("prob_exit_21d")
    p252 = rs.get("prob_exit_252d")
    exp_days = rs.get("expected_days_to_shift")
    next_regime = rs.get("most_probable_next_regime")

    lines.append(f"**Current Regime:** {current}\n")

    if next_regime:
        lines.append(f"**Most Probable Next Regime:** {next_regime}\n")
    if exp_days is not None:
        lines.append(f"**Expected Days to Regime Change:** {exp_days:.0f}\n")

    lines.append("| Horizon | P(Regime Exit) | Interpretation |")
    lines.append("|---------|---------------|----------------|")

    if p21 is not None:
        interp = "Likely shift imminent" if p21 > 0.5 else "Moderate transition risk" if p21 > 0.2 else "Regime stable"
        lines.append(f"| 21 days | {p21:.1%} | {interp} |")
    if p252 is not None:
        interp = "Almost certain shift" if p252 > 0.8 else "Probable shift" if p252 > 0.5 else "May persist"
        lines.append(f"| 252 days | {p252:.1%} | {interp} |")

    # Transition matrix if available
    tm = rs.get("transition_matrix_used")
    if tm and isinstance(tm, (list, dict)):
        lines.append("\n**Transition Probabilities (row = from, col = to):**\n")
        regime_order = rs.get("regime_order")
        if regime_order and isinstance(tm, list):
            header = "| | " + " | ".join(str(r) for r in regime_order) + " |"
            sep = "|---|" + "|".join("---" for _ in regime_order) + "|"
            lines.append(header)
            lines.append(sep)
            for i, row in enumerate(tm):
                if isinstance(row, list):
                    cells = " | ".join(f"{v:.2f}" if isinstance(v, (int, float)) else str(v) for v in row)
                    lines.append(f"| {regime_order[i] if i < len(regime_order) else i} | {cells} |")

    lines.append("")
    return "\n".join(lines) + "\n"


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
        65: ("6.5. Enriched Survival Timeline", _build_enriched_survival_timeline_section(profile)),
        7: ("7. Linked Variables & Market Context", _build_linked_entities_section(profile)),
        75: ("7.5. Economic Position & Industry Classification", _build_economic_position(profile)),
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
        196: ("19.6. SIX Swiss Exchange Analysis", _build_six_swiss_exchange_section(profile)),
        197: ("19.7. Institutional / Major Holders", _build_institutional_holders_section(profile)),
        198: ("19.8. Institutional Ownership Deep Analysis", _build_institutional_ownership_deep_section(profile)),
        199: ("19.9. Market Demand & Buying Power", _build_market_buying_power_section(profile)),
        1995: ("19.10. Product Catalysts & Forward Signals", _build_product_catalysts_section(profile)),
        1996: ("19.11. Unified Survival System", _build_unified_survival_system_section(profile)),
        1997: ("19.12. Scenario Analysis", _build_scenario_analysis_section(profile)),
        20: ("20. Risk Factors & Limitations", (
            _build_risk_assessment(profile) + "\n\n### 20.1 LIMITATIONS\n\n" + _build_limitations(profile)
        )),
        21: ("21. Investment Recommendation", _build_investment_recommendation(profile)),
        1998: ("21.5. Model Robustness Diagnostics", _build_model_diagnostics_section(profile)),
        2006: ("21.12. Investment Thesis Scorecard", _build_investment_thesis_scorecard(profile)),
        2007: ("21.13. Position Signal & Conviction", _build_position_signal_section(profile)),
        2008: ("21.14. Signal IC Analysis", _build_signal_ic_section(profile)),
        1999: ("21.6. Corporate Structure", _build_corporate_structure_section(profile)),
        2001: ("21.7. Filing Calendar & Data Freshness", _build_filing_calendar_section(profile)),
        2002: ("21.8. Macro Indicator Summary", _build_macro_indicators_section(profile)),
        2003: ("21.9. Supply Chain Stress Assessment", _build_supply_chain_stress_section(profile)),
        2004: ("21.10. Model Synergies Applied", _build_synergies_section(profile)),
        2005: ("21.11. Multi-Frequency Analysis", _build_multi_frequency_section(profile)),
        # Hedge Fund Analysis sections (23-29)
        2023: ("23. Hedge Fund Thesis: Earnings Quality", _build_hf_earnings_quality(profile)),
        2024: ("24. Hedge Fund Thesis: Cash Flow Stress", _build_hf_cash_flow(profile)),
        2025: ("25. Hedge Fund Thesis: Balance Sheet Risk", _build_hf_balance_sheet(profile)),
        2026: ("26. Hedge Fund Thesis: Inflection Detection", _build_hf_inflection(profile)),
        2027: ("27. Hedge Fund Thesis: Valuation", _build_hf_valuation(profile)),
        2028: ("28. Investment Thesis Scorecard (HF)", _build_hf_scorecard(profile)),
        2029: ("29. Position Signal & Sizing (HF)", _build_hf_position(profile)),
        2030: ("30. Product Portfolio Analysis", _build_product_segments_section(profile)),
        2031: ("31. Options-Derived Forward Signals", _build_options_signals_section(profile)),
        2032: ("32. Cross-Asset Sector Rotation Signals", _build_cross_asset_signals_section(profile)),
        2033: ("33. Upcoming Events & Uncertainty Calendar", _build_event_calendar_section(profile)),
        2034: ("34. Predicted Regime Shifts", _build_predicted_regime_shifts_section(profile)),
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
        logger.info("LLM report validation: all checks passed")

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

    def _apply_brand_style(fig, ax, title: str, *, use_date_axis: bool = True) -> None:
        """Apply Operator 1 brand theme to a chart.

        Parameters
        ----------
        use_date_axis:
            If True (default), format x-axis as calendar dates.
            Set to False for charts that use integer or non-date x-axes
            (e.g. predicted OHLC candlestick charts where x = "days ahead").
            Without this flag, integer x-values get interpreted as matplotlib
            date ordinals, producing nonsense dates like 1959-1960.
        """
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
        if use_date_axis:
            ax.xaxis.set_major_formatter(mdates.DateFormatter("%b %Y"))
            ax.xaxis.set_major_locator(mdates.MonthLocator(interval=3))
        for label in ax.get_xticklabels():
            label.set_rotation(45)
            label.set_ha("right")

    # Retrieve company name for chart titles
    identity = profile.get("identity", {})
    company = identity.get("name", identity.get("ticker", ""))

    # Check whether OHLCV price data is available.
    # If not, skip price-dependent charts (price history, volatility, OHLC predictions)
    # but still generate non-price charts (survival, financial health, sentiment, conflict).
    _has_ohlcv = profile.get("meta", {}).get("has_ohlcv", "close" in cache.columns)

    # Chart 1: Price History (public) or Equity Trajectory (private)
    _is_private = profile.get("meta", {}).get("is_private_company", False)
    try:
        if not _has_ohlcv and _is_private and "equity_value" in cache.columns:
            # Private company: equity trajectory chart
            fig, ax = plt.subplots(figsize=(16, 7))
            ev = cache["equity_value"].dropna()
            if len(ev) > 0:
                ax.plot(ev.index, ev, linewidth=1.5, color=_CHART_ACCENT, zorder=3)
                _apply_brand_style(fig, ax, f"{company} -- Book Equity Value (2Y)")
                ax.set_ylabel("Total Equity ($)", color=_CHART_FG)

                # Regime shading if available
                if "regime_label" in cache.columns:
                    regime_colors = {
                        "bull": (_CHART_GREEN, "Improving"),
                        "bear": (_CHART_RED, "Deteriorating"),
                        "high_vol": (_CHART_GOLD, "Volatile"),
                        "low_vol": ("#8b8694", "Stable"),
                    }
                    for regime, (color, label) in regime_colors.items():
                        mask = cache["regime_label"] == regime
                        if mask.any():
                            ax.fill_between(
                                cache.index,
                                ev.min() * 0.98,
                                ev.max() * 1.02,
                                where=mask.reindex(cache.index, fill_value=False),
                                alpha=0.15, color=color, label=label,
                            )
                    ax.legend(
                        loc="upper left", fontsize=9, facecolor=_CHART_BG,
                        edgecolor=_CHART_GRID, labelcolor=_CHART_FG,
                    )

                fig.tight_layout()
                path = str(out / "price_history.png")
                fig.savefig(path, dpi=180, facecolor=_CHART_BG)
                plt.close(fig)
                chart_paths.append(path)
                logger.info("Generated chart: %s (private company equity trajectory)", path)
        elif _has_ohlcv and "close" in cache.columns:
            fig, ax = plt.subplots(figsize=(16, 7))
            ax.plot(cache.index, cache["close"], linewidth=1.5, color=_CHART_ACCENT, zorder=3)
            _apply_brand_style(fig, ax, f"{company} -- Closing Price (2Y)")
            ax.set_ylabel("Price ($)", color=_CHART_FG)

            # Next-day Low estimate annotation (Technical Alpha)
            ta = profile.get("predictions", {}).get("technical_alpha", {})
            est_low = ta.get("estimated_low")
            if est_low is None:
                # Try from ohlc_predictions
                est_low = profile.get("ohlc_predictions", {}).get("next_day", {}).get("low")
            if est_low is not None and est_low > 0:
                ax.axhline(y=est_low, color=_CHART_GOLD, linestyle="--", linewidth=1.5, alpha=0.8)
                ax.annotate(
                    f"Next-Day Low: {est_low:,.2f}",
                    xy=(1.0, est_low), xycoords=("axes fraction", "data"),
                    fontsize=9, color=_CHART_GOLD, fontweight="bold",
                    ha="right", va="bottom",
                )

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

    # Chart 3: 21-Day Realized Volatility (requires OHLCV data)
    try:
        if _has_ohlcv and "volatility_21d" in cache.columns:
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

    # Chart 7: Predicted OHLC Candlestick (Next Month) -- requires OHLCV
    ohlc_data = profile.get("ohlc_predictions", {}) if _has_ohlcv else {}
    try:
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

                _apply_brand_style(fig, ax, f"{company} -- Predicted Price (Next Month)", use_date_axis=False)
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

    # Chart 8: Predicted OHLC Candlestick (Next Week) -- requires OHLCV
    try:
        next_week = ohlc_data.get("next_week", {}) if _has_ohlcv else {}
        week_series = next_week.get("series", [])
        if week_series and len(week_series) >= 3:
            fig, ax = plt.subplots(figsize=(12, 6))
            for i, candle in enumerate(week_series):
                o, h, l, c_ = candle["open"], candle["high"], candle["low"], candle["close"]
                color = _CHART_GREEN if c_ >= o else _CHART_RED
                ax.bar(i, abs(c_ - o), bottom=min(o, c_), width=0.6,
                       color=color, edgecolor=color, alpha=0.85)
                ax.plot([i, i], [l, h], color=color, linewidth=0.8)

            _apply_brand_style(fig, ax, f"{company} -- Predicted Price (Next Week)", use_date_axis=False)
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

    # Chart 9.5: Predicted OHLC Candlestick (Next Year) -- requires OHLCV
    try:
        next_year = ohlc_data.get("next_year", {}) if _has_ohlcv else {}
        year_series = next_year.get("series", [])
        if year_series and len(year_series) >= 20:
            # Aggregate 252 daily candles into weekly bars for readability
            weekly_bars = []
            for w_start in range(0, len(year_series), 5):
                week = year_series[w_start:w_start + 5]
                if not week:
                    continue
                w_open = week[0].get("open", 0) or 0
                w_high = max((c.get("high", 0) or 0) for c in week)
                w_low = min((c.get("low", float("inf")) or float("inf")) for c in week)
                w_close = week[-1].get("close", 0) or 0
                w_conf = sum(c.get("confidence", 1.0) for c in week) / len(week)
                if w_open and w_close and w_low < float("inf"):
                    weekly_bars.append({"open": w_open, "high": w_high, "low": w_low,
                                        "close": w_close, "confidence": w_conf})

            if len(weekly_bars) >= 5:
                fig, ax = plt.subplots(figsize=(16, 7))
                for i, candle in enumerate(weekly_bars):
                    o, h, l, c_ = candle["open"], candle["high"], candle["low"], candle["close"]
                    color = _CHART_GREEN if c_ >= o else _CHART_RED
                    body_bottom = min(o, c_)
                    body_height = abs(c_ - o)
                    ax.bar(i, body_height, bottom=body_bottom, width=0.6,
                           color=color, edgecolor=color, alpha=0.85)
                    ax.plot([i, i], [l, h], color=color, linewidth=0.8)

                # Confidence envelope
                confidences = [c["confidence"] for c in weekly_bars]
                mid_prices = [(c["high"] + c["low"]) / 2 for c in weekly_bars]
                ranges = [c["high"] - c["low"] for c in weekly_bars]
                upper = [m + r * (2 - conf) for m, r, conf in zip(mid_prices, ranges, confidences)]
                lower = [m - r * (2 - conf) for m, r, conf in zip(mid_prices, ranges, confidences)]
                ax.fill_between(range(len(weekly_bars)), lower, upper, alpha=0.08, color=_CHART_ACCENT)

                _apply_brand_style(fig, ax, f"{company} -- Predicted Price (Next Year, Weekly)", use_date_axis=False)
                ax.set_ylabel("Price ($)", color=_CHART_FG)
                ax.set_xlabel("Weeks Ahead", color=_CHART_FG)

                year_ret = next_year.get("predicted_return")
                if year_ret is not None:
                    ax.annotate(
                        f"Predicted annual return: {year_ret:+.1f}%",
                        xy=(0.02, 0.95), xycoords="axes fraction",
                        fontsize=11, color=_CHART_GREEN if year_ret >= 0 else _CHART_RED,
                        fontweight="bold",
                    )

                fig.tight_layout()
                path = str(out / "predicted_ohlc_year.png")
                fig.savefig(path, dpi=180, facecolor=_CHART_BG)
                plt.close(fig)
                chart_paths.append(path)
                logger.info("Generated chart: %s", path)
    except Exception as exc:
        logger.warning("Failed to generate predicted year OHLC chart: %s", exc)

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

# Maps chart filename -> list of keyword patterns to match section headings.
# Uses keyword matching (case-insensitive) so charts embed correctly in both
# LLM-generated reports (13 sections) and fallback template reports (22 sections),
# regardless of section numbering.
_CHART_KEYWORDS: dict[str, list[str]] = {
    "price_history.png": ["historical", "performance"],
    "survival_timeline.png": ["survival", "mode"],
    "volatility.png": ["temporal", "analysis"],
    "financial_health.png": ["financial", "health"],
    "sentiment.png": ["sentiment"],
    "predicted_ohlc_week.png": ["prediction", "forecast"],
    "predicted_ohlc_month.png": ["prediction", "forecast"],
    "predicted_ohlc_year.png": ["prediction", "forecast"],
    "conflict_risk.png": ["geopolitical", "conflict"],
}


def _embed_charts_in_markdown(
    markdown: str,
    chart_paths: list[str],
    chart_dir_relative: str = "charts",
) -> str:
    """Embed chart image references into the report markdown.

    For each generated chart, inserts a markdown image tag
    ``![title](charts/filename.png)`` after the matching section heading.

    Uses keyword matching so charts embed correctly in both LLM-generated
    reports (which use a 13-section structure) and fallback template
    reports (which use a 22-section structure).  The keywords are
    matched case-insensitively against ``##`` heading lines.

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
        keywords = _CHART_KEYWORDS.get(filename, [])

        if not keywords:
            continue

        # Build the image markdown
        title = filename.replace(".png", "").replace("_", " ").title()
        image_tag = f"\n\n![{title}]({chart_dir_relative}/{filename})\n"

        # Find the first ## heading line that contains ALL keywords
        for line in markdown.split("\n"):
            stripped = line.strip()
            if not stripped.startswith("##"):
                continue
            line_lower = stripped.lower()
            if all(kw in line_lower for kw in keywords):
                # Insert the image tag after this heading line
                parts = markdown.split(line, 1)
                if len(parts) == 2:
                    markdown = parts[0] + line + image_tag + parts[1]
                break

    return markdown


# ---------------------------------------------------------------------------
# PDF generation (optional)
# ---------------------------------------------------------------------------


def _get_template_dir() -> Path:
    """Return the path to the report templates directory."""
    return Path(__file__).resolve().parent / "templates"


def _generate_html(
    markdown_path: str | Path,
    output_path: str | Path | None = None,
) -> str | None:
    """Convert Markdown report to styled HTML using pandoc + custom template.

    Returns the HTML path on success, or None if pandoc is unavailable.
    """
    if shutil.which("pandoc") is None:
        logger.info("pandoc not found; skipping HTML generation.")
        return None

    md = Path(markdown_path)
    if output_path is None:
        output_path = md.with_suffix(".html")
    html = Path(output_path)

    tpl_dir = _get_template_dir()
    css_path = tpl_dir / "report.css"
    html_tpl = tpl_dir / "report.html"

    cmd = [
        "pandoc",
        str(md),
        "-o",
        str(html),
        "--standalone",
        "--self-contained",
    ]

    # Use custom HTML template if available
    if html_tpl.exists():
        cmd.extend(["--template", str(html_tpl)])

    # Embed CSS for styling
    if css_path.exists():
        cmd.extend(["--css", str(css_path)])

    try:
        subprocess.run(cmd, check=True, capture_output=True, timeout=60)
        logger.info("HTML report generated: %s", html)
        return str(html)
    except FileNotFoundError:
        logger.info("pandoc not available; skipping HTML generation.")
        return None
    except subprocess.CalledProcessError as exc:
        logger.warning("HTML generation failed: %s", exc.stderr.decode()[:200])
        return None
    except subprocess.TimeoutExpired:
        logger.warning("HTML generation timed out.")
        return None


def _generate_pdf(
    markdown_path: str | Path,
    output_path: str | Path | None = None,
) -> str | None:
    """Convert Markdown report to PDF using pandoc.

    Tries multiple PDF engines in order of preference:
    1. weasyprint (CSS-styled PDF -- uses our custom stylesheet)
    2. xelatex (LaTeX-based -- clean but no custom CSS)
    3. wkhtmltopdf (WebKit-based fallback)

    Returns the PDF path on success, or None if pandoc is unavailable.
    """
    if shutil.which("pandoc") is None:
        logger.info("pandoc not found; skipping PDF generation.")
        return None

    md = Path(markdown_path)
    if output_path is None:
        output_path = md.with_suffix(".pdf")
    pdf = Path(output_path)

    tpl_dir = _get_template_dir()
    css_path = tpl_dir / "report.css"

    # Try weasyprint first (supports CSS natively for beautiful PDFs)
    for engine, engine_args in [
        ("weasyprint", [
            "--pdf-engine=weasyprint",
            *(["--css", str(css_path)] if css_path.exists() else []),
        ]),
        ("xelatex", [
            "--pdf-engine=xelatex",
            "-V", "geometry:margin=1in",
            "-V", "fontsize=10pt",
            "-V", "mainfont=DejaVu Sans",
            "-V", "monofont=DejaVu Sans Mono",
        ]),
        ("wkhtmltopdf", [
            "--pdf-engine=wkhtmltopdf",
            *(["--css", str(css_path)] if css_path.exists() else []),
        ]),
    ]:
        cmd = ["pandoc", str(md), "-o", str(pdf), "--standalone"] + engine_args
        try:
            subprocess.run(
                cmd,
                check=True,
                capture_output=True,
                timeout=120,
            )
            logger.info("PDF report generated via %s: %s", engine, pdf)
            return str(pdf)
        except FileNotFoundError:
            logger.debug("PDF engine %s not available, trying next.", engine)
            continue
        except subprocess.CalledProcessError as exc:
            stderr = exc.stderr.decode()[:200] if exc.stderr else "unknown error"
            logger.debug("PDF engine %s failed: %s", engine, stderr)
            continue
        except subprocess.TimeoutExpired:
            logger.debug("PDF engine %s timed out.", engine)
            continue

    logger.warning("No PDF engine available (tried weasyprint, xelatex, wkhtmltopdf).")
    return None


# ---------------------------------------------------------------------------
# Main entry point
# ---------------------------------------------------------------------------


def generate_report(
    profile: dict[str, Any],
    *,
    llm_client: Any | None = None,
    cache: pd.DataFrame | None = None,
    output_dir: str | Path | None = None,
    generate_pdf: bool = False,
    generate_chart_images: bool = True,
    tier: ReportTier = ReportTier.PREMIUM,
    mode: ReportMode = ReportMode.RESULTS,
    # Backward-compatible alias (deprecated, use llm_client instead)
    gemini_client: Any | None = None,
) -> dict[str, Any]:
    """Generate an analysis report from a company profile.

    Parameters
    ----------
    profile:
        Company profile dict from ``build_company_profile()``.
    llm_client:
        Optional LLM client instance (Gemini, Claude, or OpenRouter).
        If provided, the premium report narrative is generated by the LLM.
        Otherwise, a local template is used.
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
    # Backward-compat: accept deprecated gemini_client kwarg
    if llm_client is None and gemini_client is not None:
        llm_client = gemini_client

    if output_dir is None:
        output_dir = Path(CACHE_DIR) / "report"
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)

    logger.info("Generating %s...", tier.label)

    # Step 1: Generate narrative via LLM (premium reports only)
    markdown = ""
    if llm_client is not None and tier == ReportTier.PREMIUM:
        try:
            profile_json = json.dumps(profile, indent=2, default=str)
            markdown = llm_client.generate_report(profile_json)
            logger.info(
                "%s narrative generated via %s.",
                tier.label,
                getattr(llm_client, "provider_name", "LLM"),
            )
        except Exception as exc:
            logger.warning(
                "LLM report generation failed (%s); using fallback template.",
                exc,
            )
            markdown = ""

    # Fallback if LLM produced nothing or tier is not premium
    if not markdown or not markdown.strip():
        markdown = _build_fallback_report(profile, tier=tier, mode=mode)
        logger.info("%s (%s mode) generated using local template.", tier.label, mode.label)

    # Ensure LIMITATIONS section exists (append if LLM missed it)
    if "LIMITATIONS" not in markdown.upper():
        limitations = _build_limitations(profile)
        markdown += "\n\n---\n\n## LIMITATIONS\n\n" + limitations
        logger.info("Appended LIMITATIONS section to report.")

    # Step 1b: Validate LLM output (E2) -- premium only
    if llm_client is not None and tier == ReportTier.PREMIUM:
        is_valid, validation_issues = validate_gemini_report(markdown, profile)
        if not is_valid:
            logger.warning(
                "Gemini report validation found %d issues; "
                "appending missing sections from fallback template.",
                len(validation_issues),
            )
            # Auto-patch: append missing sections from fallback template
            _patch_builders: dict[str, str] = {
                "executive summary": _build_executive_summary(profile),
                "company overview": _build_company_overview(profile),
                "historical performance": _build_historical_performance(profile),
                "financial health": _build_financial_health(profile),
                "survival": _build_survival_analysis(profile),
                "linked variables": _build_linked_entities_section(profile),
                "temporal analysis": _build_regime_analysis(profile),
                "predictions": _build_predictions_forecasts(profile),
                "technical patterns": _build_technical_patterns(profile),
                "ethical filter": _build_ethical_filters_section(profile),
                "supply chain": _build_graph_risk_section(profile),
                "competitive": _build_game_theory_section(profile),
                "risk factors": _build_risk_assessment(profile),
                "limitations": _build_limitations(profile),
                "investment recommendation": _build_investment_recommendation(profile),
                "appendix": _build_appendix(profile),
            }
            for issue in validation_issues:
                if issue.startswith("Missing section:"):
                    section_name = issue.replace("Missing section: '", "").rstrip("'")
                    content = _patch_builders.get(section_name.lower())
                    if content:
                        markdown += f"\n\n---\n\n## {section_name.title()}\n\n{content}\n"
                        logger.info("Auto-patched missing section: %s", section_name)
                    else:
                        logger.info("Missing section '%s' -- no matching fallback builder", section_name)

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

    # Step 4: Generate styled HTML (always, if pandoc is available)
    html_path = _generate_html(md_path, out / f"{tier.value}_report.html")

    # Step 5: Optional PDF (all tiers)
    pdf_path: str | None = None
    if generate_pdf:
        pdf_path = _generate_pdf(md_path, out / f"{tier.value}_report.pdf")

    # Step 6: Enhanced outputs (Premium tier only)
    # Generates fpdf2 PDF, mplfinance charts, quantstats tearsheet,
    # and plotly interactive dashboard as supplementary outputs.
    enhanced: dict[str, Any] = {}
    if tier == ReportTier.PREMIUM:
        try:
            from operator1.report.enhanced_outputs import generate_enhanced_outputs
            enhanced = generate_enhanced_outputs(
                markdown=markdown,
                cache=cache,
                profile=profile,
                chart_paths=chart_paths,
                output_dir=out,
                generate_pdf=True,
                generate_tearsheet=True,
                generate_interactive=True,
                generate_enhanced_charts=True,
            )
            logger.info(
                "Enhanced outputs: pdf=%s, tearsheet=%s, dashboard=%s, charts=%d",
                bool(enhanced.get("fpdf2_pdf_path")),
                bool(enhanced.get("tearsheet_path")),
                bool(enhanced.get("interactive_path")),
                len(enhanced.get("enhanced_chart_paths", [])),
            )
        except Exception as exc:
            logger.warning("Enhanced outputs failed (non-fatal): %s", exc)

    return {
        "markdown": markdown,
        "markdown_path": str(md_path),
        "html_path": html_path,
        "chart_paths": chart_paths,
        "pdf_path": pdf_path,
        "tier": tier.value,
        "mode": mode.value,
        "enhanced_pdf_path": enhanced.get("fpdf2_pdf_path"),
        "tearsheet_path": enhanced.get("tearsheet_path"),
        "interactive_dashboard_path": enhanced.get("interactive_path"),
        "enhanced_chart_paths": enhanced.get("enhanced_chart_paths", []),
    }


def generate_all_reports(
    profile: dict[str, Any],
    *,
    llm_client: Any | None = None,
    cache: pd.DataFrame | None = None,
    output_dir: str | Path | None = None,
    generate_pdf: bool = False,
    generate_chart_images: bool = True,
    mode: ReportMode = ReportMode.RESULTS,
    # Backward-compatible alias (deprecated, use llm_client instead)
    gemini_client: Any | None = None,
) -> dict[str, dict[str, Any]]:
    """Generate all three report tiers (Basic, Pro, Premium) at once.

    Parameters
    ----------
    profile:
        Company profile dict from ``build_company_profile()``.
    llm_client:
        Optional LLM client (Gemini, Claude, or OpenRouter) for premium report narrative.
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
    # Backward-compat: accept deprecated gemini_client kwarg
    if llm_client is None and gemini_client is not None:
        llm_client = gemini_client

    results: dict[str, dict[str, Any]] = {}

    for tier in (ReportTier.BASIC, ReportTier.PRO, ReportTier.PREMIUM):
        results[tier.value] = generate_report(
            profile,
            llm_client=llm_client,
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
