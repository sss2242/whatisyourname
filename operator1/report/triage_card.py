"""Unified Survival System -- Triage Card Output.

When survival mode is active, generates a single-page emergency brief
with 5 sections of immediately actionable information:

1. Cash Runway -- days of cash remaining, exact zero-cash date
2. Debt Maturity Schedule -- next 3 maturities, covenant test dates
3. Liquidity Sources -- credit lines, unencumbered assets, receivables
4. Competitive Survival -- competitor distress, supply chain risk
5. Scenario Matrix -- 3 scenarios with probability and cash impact

This is Output A from the Unified Survival System Architecture.
Output B (full survival analysis) is handled by the standard report
generator with survival-mode restructuring.
"""

from __future__ import annotations

import logging
import math
from datetime import date, timedelta
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)


def _safe_fmt(val: Any, fmt: str = ",.0f", fallback: str = "N/A") -> str:
    """Format a value safely, returning fallback for None/NaN."""
    if val is None:
        return fallback
    try:
        f = float(val)
        if math.isnan(f) or math.isinf(f):
            return fallback
        return f"{f:{fmt}}"
    except (TypeError, ValueError):
        return fallback


def _pct(val: Any, fallback: str = "N/A") -> str:
    """Format a float as percentage."""
    if val is None:
        return fallback
    try:
        f = float(val)
        if math.isnan(f) or math.isinf(f):
            return fallback
        return f"{f * 100:.1f}%"
    except (TypeError, ValueError):
        return fallback


def generate_triage_card(
    profile: dict[str, Any],
    cache: Any | None = None,
    scenario_result: Any | None = None,
    controller: Any | None = None,
    output_dir: Path | None = None,
) -> dict[str, Any]:
    """Generate a single-page emergency triage card for survival mode.

    Parameters
    ----------
    profile:
        Company profile dict from build_company_profile().
    cache:
        Daily cache DataFrame (optional, for additional detail).
    scenario_result:
        ScenarioEngineResult from run_scenario_engine().
    controller:
        SurvivalRegimeController instance.
    output_dir:
        Directory to write the triage card markdown.

    Returns
    -------
    Dict with 'markdown' (str), 'markdown_path' (str or None),
    and 'sections' (dict of section data).
    """
    identity = profile.get("identity", {})
    company_name = identity.get("name", "Unknown Company")
    ticker = identity.get("ticker", "")
    country = identity.get("country", "")

    survival = profile.get("survival", {})
    fh = profile.get("financial_health", {})
    mc = profile.get("monte_carlo", {})
    current = profile.get("current_state", {})

    regime = "unknown"
    if controller is not None:
        regime = controller.current_regime
    elif survival:
        regime = survival.get("survival_regime", "unknown")

    # ---------------------------------------------------------------
    # Section 1: Cash Runway
    # ---------------------------------------------------------------
    cash_section = _build_cash_runway_section(profile, cache)

    # ---------------------------------------------------------------
    # Section 2: Debt Maturity Schedule
    # ---------------------------------------------------------------
    debt_section = _build_debt_maturity_section(profile, cache)

    # ---------------------------------------------------------------
    # Section 3: Liquidity Sources
    # ---------------------------------------------------------------
    liquidity_section = _build_liquidity_sources_section(profile, cache)

    # ---------------------------------------------------------------
    # Section 4: Competitive Survival
    # ---------------------------------------------------------------
    competitive_section = _build_competitive_survival_section(profile)

    # ---------------------------------------------------------------
    # Section 5: Scenario Matrix
    # ---------------------------------------------------------------
    scenario_section = _build_scenario_matrix_section(scenario_result)

    # ---------------------------------------------------------------
    # Assemble markdown
    # ---------------------------------------------------------------
    lines = []
    lines.append(f"# TRIAGE CARD -- {company_name} ({ticker})")
    lines.append("")
    lines.append(f"**Regime:** `{regime.upper()}` | **Country:** {country} | **Date:** {date.today().isoformat()}")
    lines.append("")

    # Survival probability banner
    surv_prob = mc.get("survival_probability_mean")
    fh_score = fh.get("latest_composite")
    lines.append("---")
    lines.append("")
    lines.append(f"| Survival Probability | Health Score | Regime |")
    lines.append(f"|:---:|:---:|:---:|")
    lines.append(
        f"| **{_pct(surv_prob)}** | "
        f"**{_safe_fmt(fh_score, '.0f')}/100** ({fh.get('latest_label', 'N/A')}) | "
        f"**{regime.upper().replace('_', ' ')}** |"
    )
    lines.append("")
    lines.append("---")
    lines.append("")

    # Section 1
    lines.append("## 1. Cash Runway")
    lines.append("")
    lines.extend(cash_section["lines"])
    lines.append("")

    # Section 2
    lines.append("## 2. Debt Maturity Schedule")
    lines.append("")
    lines.extend(debt_section["lines"])
    lines.append("")

    # Section 3
    lines.append("## 3. Liquidity Sources")
    lines.append("")
    lines.extend(liquidity_section["lines"])
    lines.append("")

    # Section 4
    lines.append("## 4. Competitive Survival")
    lines.append("")
    lines.extend(competitive_section["lines"])
    lines.append("")

    # Section 5
    lines.append("## 5. Scenario Matrix")
    lines.append("")
    lines.extend(scenario_section["lines"])
    lines.append("")

    # Early warning
    if controller is not None and controller.is_approaching_survival():
        ew = controller.get_early_warning_latest()
        lines.append("---")
        lines.append("")
        lines.append(f"> **EARLY WARNING:** Score = {ew:.2f}/1.00. "
                      "Company is approaching survival mode triggers.")
        lines.append("")

    # Footer
    lines.append("---")
    lines.append("")
    lines.append(f"*Generated {date.today().isoformat()} by Operator 1 Unified Survival System. "
                  "All survival-mode forecasts carry LOW confidence and 3-5x wider uncertainty bands.*")

    markdown = "\n".join(lines)

    # Write to file
    markdown_path = None
    if output_dir is not None:
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        triage_path = output_dir / "triage_card.md"
        triage_path.write_text(markdown, encoding="utf-8")
        markdown_path = str(triage_path)
        logger.info("Triage card saved: %s", triage_path)

    return {
        "markdown": markdown,
        "markdown_path": markdown_path,
        "sections": {
            "cash_runway": cash_section.get("data", {}),
            "debt_maturity": debt_section.get("data", {}),
            "liquidity_sources": liquidity_section.get("data", {}),
            "competitive_survival": competitive_section.get("data", {}),
            "scenario_matrix": scenario_section.get("data", {}),
        },
    }


def _build_cash_runway_section(
    profile: dict[str, Any],
    cache: Any | None,
) -> dict[str, Any]:
    """Build Section 1: Cash Runway."""
    lines: list[str] = []
    data: dict[str, Any] = {}

    current = profile.get("current_state", {})
    tier1 = current.get("tier1_liquidity", {})
    fh = profile.get("financial_health", {})

    cash = tier1.get("cash_and_equivalents")
    ocf = tier1.get("operating_cash_flow")
    fcf = tier1.get("free_cash_flow")
    runway_months = fh.get("runway_months")

    # Compute runway if not available from FH
    runway_days = None
    zero_cash_date = None
    if runway_months is not None:
        try:
            rm = float(runway_months)
            if rm > 0 and rm < 999:
                runway_days = rm * 30
                zero_cash_date = date.today() + timedelta(days=int(runway_days))
        except (TypeError, ValueError):
            pass

    if runway_days is None and cash is not None and ocf is not None:
        try:
            c = float(cash)
            o = float(ocf)
            monthly_burn = o / 12.0
            if monthly_burn < 0 and c > 0:
                runway_days = -c / (monthly_burn / 30.0)  # In days
                zero_cash_date = date.today() + timedelta(days=int(runway_days))
        except (TypeError, ValueError):
            pass

    data["cash"] = cash
    data["operating_cash_flow"] = ocf
    data["free_cash_flow"] = fcf
    data["runway_days"] = runway_days
    data["zero_cash_date"] = str(zero_cash_date) if zero_cash_date else None

    lines.append("| Metric | Value |")
    lines.append("|:-------|------:|")
    lines.append(f"| Cash & Equivalents | ${_safe_fmt(cash)} |")
    lines.append(f"| Operating Cash Flow (annual) | ${_safe_fmt(ocf)} |")
    lines.append(f"| Free Cash Flow (annual) | ${_safe_fmt(fcf)} |")

    if runway_days is not None:
        urgency = "CRITICAL" if runway_days < 90 else "WARNING" if runway_days < 180 else "MONITOR"
        lines.append(f"| **Cash Runway** | **{runway_days:.0f} days** ({urgency}) |")
        if zero_cash_date:
            lines.append(f"| Zero-Cash Date | {zero_cash_date.isoformat()} |")
    else:
        lines.append("| Cash Runway | N/A (positive cash flow or missing data) |")

    return {"lines": lines, "data": data}


def _build_debt_maturity_section(
    profile: dict[str, Any],
    cache: Any | None,
) -> dict[str, Any]:
    """Build Section 2: Debt Maturity Schedule."""
    lines: list[str] = []
    data: dict[str, Any] = {}

    current = profile.get("current_state", {})
    tier2 = current.get("tier2_solvency", {})

    total_debt = tier2.get("total_debt") or tier2.get("total_debt_asof")
    st_debt = tier2.get("short_term_debt")
    lt_debt = tier2.get("long_term_debt")
    dte = tier2.get("debt_to_equity_abs")
    int_cov = tier2.get("interest_coverage")

    data["total_debt"] = total_debt
    data["short_term_debt"] = st_debt
    data["long_term_debt"] = lt_debt
    data["debt_to_equity"] = dte
    data["interest_coverage"] = int_cov

    lines.append("| Metric | Value |")
    lines.append("|:-------|------:|")
    lines.append(f"| Total Debt | ${_safe_fmt(total_debt)} |")
    lines.append(f"| Short-Term Debt (due < 1yr) | ${_safe_fmt(st_debt)} |")
    lines.append(f"| Long-Term Debt | ${_safe_fmt(lt_debt)} |")
    lines.append(f"| Debt-to-Equity | {_safe_fmt(dte, '.2f')}x |")
    lines.append(f"| Interest Coverage | {_safe_fmt(int_cov, '.2f')}x |")

    if int_cov is not None:
        try:
            ic = float(int_cov)
            if ic < 1.0:
                lines.append("")
                lines.append("> **CRITICAL:** Interest coverage < 1.0x. "
                             "Company cannot service its debt from operating income.")
            elif ic < 1.5:
                lines.append("")
                lines.append("> **WARNING:** Interest coverage < 1.5x. "
                             "Debt servicing capacity is severely constrained.")
        except (TypeError, ValueError):
            pass

    return {"lines": lines, "data": data}


def _build_liquidity_sources_section(
    profile: dict[str, Any],
    cache: Any | None,
) -> dict[str, Any]:
    """Build Section 3: Liquidity Sources."""
    lines: list[str] = []
    data: dict[str, Any] = {}

    current = profile.get("current_state", {})
    tier1 = current.get("tier1_liquidity", {})
    tier2 = current.get("tier2_solvency", {})

    cash = tier1.get("cash_and_equivalents")
    current_assets = tier2.get("current_assets") or tier1.get("current_assets")
    receivables = tier2.get("receivables")
    inventory = tier2.get("inventory")
    current_ratio = tier1.get("current_ratio") or tier2.get("current_ratio")

    data["cash"] = cash
    data["current_assets"] = current_assets
    data["receivables"] = receivables
    data["current_ratio"] = current_ratio

    lines.append("**Available liquidity sources (estimated):**")
    lines.append("")
    lines.append("| Source | Amount | Availability |")
    lines.append("|:-------|-------:|:-------------|")
    lines.append(f"| Cash & Equivalents | ${_safe_fmt(cash)} | Immediate |")

    if receivables is not None:
        try:
            r = float(receivables)
            factored = r * 0.85  # Typical factoring advance rate
            lines.append(f"| Receivables (factorable at 85%) | ${_safe_fmt(factored)} | 30-60 days |")
        except (TypeError, ValueError):
            pass

    if inventory is not None:
        try:
            i = float(inventory)
            liquidated = i * 0.50  # Distressed liquidation rate
            lines.append(f"| Inventory (distress liquidation at 50%) | ${_safe_fmt(liquidated)} | 60-90 days |")
        except (TypeError, ValueError):
            pass

    lines.append("")
    lines.append(f"Current Ratio: **{_safe_fmt(current_ratio, '.2f')}x**")

    return {"lines": lines, "data": data}


def _build_competitive_survival_section(
    profile: dict[str, Any],
) -> dict[str, Any]:
    """Build Section 4: Competitive Survival."""
    lines: list[str] = []
    data: dict[str, Any] = {}

    game_theory = profile.get("game_theory", {})
    conflict = profile.get("conflict_risk", {})
    linked = profile.get("linked_entities", {})

    market_structure = game_theory.get("market_structure", "unknown")
    competitive_pressure = game_theory.get("competitive_pressure")
    supply_chain_risk = conflict.get("supply_chain_risk_score", 0)
    revenue_exposure = conflict.get("revenue_exposure_score", 0)

    data["market_structure"] = market_structure
    data["competitive_pressure"] = competitive_pressure
    data["supply_chain_risk"] = supply_chain_risk

    lines.append("| Assessment | Value |")
    lines.append("|:-----------|------:|")
    lines.append(f"| Market Structure | {market_structure} |")
    lines.append(f"| Competitive Pressure | {_safe_fmt(competitive_pressure, '.3f')} |")
    lines.append(f"| Supply Chain Risk | {_safe_fmt(supply_chain_risk, '.2f')} |")
    lines.append(f"| Revenue Exposure | {_safe_fmt(revenue_exposure, '.2f')} |")

    # Competitor distress check
    peer_ranking = profile.get("peer_ranking", {})
    if isinstance(peer_ranking, dict) and peer_ranking.get("n_peers", 0) > 0:
        lines.append("")
        label = peer_ranking.get("latest_label", "unknown")
        rank = peer_ranking.get("latest_composite_rank")
        lines.append(f"Peer position: **{label}** (rank: {_safe_fmt(rank, '.1f')})")
        if label in ("Top Quartile", "Above Median"):
            lines.append("> Last-man-standing advantage: company ranks above peers.")

    return {"lines": lines, "data": data}


def _build_scenario_matrix_section(
    scenario_result: Any | None,
) -> dict[str, Any]:
    """Build Section 5: Scenario Matrix."""
    lines: list[str] = []
    data: dict[str, Any] = {}

    if scenario_result is None or not getattr(scenario_result, "available", False):
        lines.append("*Scenario engine did not run (not in survival mode or insufficient data).*")
        return {"lines": lines, "data": data}

    o = scenario_result.orderly
    m = scenario_result.muddle_through
    c = scenario_result.catastrophic

    lines.append("| Scenario | Cash Runway | 90d Survival | 252d Survival | Terminal Return |")
    lines.append("|:---------|:----------:|:------------:|:-------------:|:---------------:|")
    lines.append(
        f"| **Orderly Resolution** | {o.cash_runway_days:.0f}d | "
        f"{_pct(o.survival_prob_90d)} | {_pct(o.survival_prob_252d)} | "
        f"{_pct(o.median_return)} |"
    )
    lines.append(
        f"| **Muddle Through** | {m.cash_runway_days:.0f}d | "
        f"{_pct(m.survival_prob_90d)} | {_pct(m.survival_prob_252d)} | "
        f"{_pct(m.median_return)} |"
    )
    lines.append(
        f"| **Catastrophic** | {c.cash_runway_days:.0f}d | "
        f"{_pct(c.survival_prob_90d)} | {_pct(c.survival_prob_252d)} | "
        f"{_pct(c.median_return)} |"
    )

    lines.append("")
    lines.append("**Scenario Descriptions:**")
    lines.append(f"- **Orderly:** {o.description}")
    lines.append(f"- **Muddle Through:** {m.description}")
    lines.append(f"- **Catastrophic:** {c.description}")

    # Max drawdown comparison
    lines.append("")
    lines.append(f"Median max drawdown: Orderly {o.max_drawdown_median:.1%} / "
                  f"Muddle {m.max_drawdown_median:.1%} / "
                  f"Catastrophic {c.max_drawdown_median:.1%}")

    data["orderly"] = {
        "cash_runway": o.cash_runway_days,
        "survival_90d": o.survival_prob_90d,
        "survival_252d": o.survival_prob_252d,
    }
    data["muddle_through"] = {
        "cash_runway": m.cash_runway_days,
        "survival_90d": m.survival_prob_90d,
        "survival_252d": m.survival_prob_252d,
    }
    data["catastrophic"] = {
        "cash_runway": c.cash_runway_days,
        "survival_90d": c.survival_prob_90d,
        "survival_252d": c.survival_prob_252d,
    }

    return {"lines": lines, "data": data}
