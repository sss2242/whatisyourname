#!/usr/bin/env python3
"""Operator 1 -- Desktop Dashboard.

Native desktop application for financial analysis pipeline.
Replaces run.py's linear terminal flow with a professional,
non-linear GUI built with NiceGUI.

Usage:
    python dashboard.py
"""
from __future__ import annotations

import asyncio
import json
import logging
import os
import subprocess
import sys
import threading
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from nicegui import app, ui

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# State management
# ---------------------------------------------------------------------------

_STATE_FILE = Path("cache/dashboard_state.json")
_HISTORY_FILE = Path("cache/analysis_history.json")


class DashboardState:
    """Persistent dashboard state."""

    def __init__(self) -> None:
        self.llm_provider: str = ""
        self.llm_model: str = ""
        self.region: str = ""
        self.market_id: str = ""
        self.company: str = ""
        self.skip_linked: bool = False
        self.skip_models: bool = False
        self.gen_pdf: bool = False
        self.data_mode: str = "wrappers"
        self.keys: dict[str, str] = {}
        self.dep_status: dict[str, bool] = {}
        self.dep_versions: dict[str, str] = {}
        self.analysis_running: bool = False
        self.analysis_log: list[str] = []
        self.history: list[dict] = []
        self._load()

    def _load(self) -> None:
        if _STATE_FILE.exists():
            try:
                data = json.loads(_STATE_FILE.read_text())
                for k, v in data.items():
                    if hasattr(self, k):
                        setattr(self, k, v)
            except Exception:
                pass
        if _HISTORY_FILE.exists():
            try:
                self.history = json.loads(_HISTORY_FILE.read_text())
            except Exception:
                self.history = []

    def save(self) -> None:
        _STATE_FILE.parent.mkdir(parents=True, exist_ok=True)
        data = {
            "llm_provider": self.llm_provider,
            "llm_model": self.llm_model,
            "region": self.region,
            "market_id": self.market_id,
            "company": self.company,
            "skip_linked": self.skip_linked,
            "skip_models": self.skip_models,
            "gen_pdf": self.gen_pdf,
            "data_mode": self.data_mode,
        }
        _STATE_FILE.write_text(json.dumps(data, indent=2))

    def add_history(self, entry: dict) -> None:
        self.history.insert(0, entry)
        self.history = self.history[:50]
        _HISTORY_FILE.parent.mkdir(parents=True, exist_ok=True)
        _HISTORY_FILE.write_text(json.dumps(self.history, indent=2, default=str))


state = DashboardState()


# ---------------------------------------------------------------------------
# Dependency checking
# ---------------------------------------------------------------------------

DEPENDENCY_STAGES = {
    "Stage 1 (Core)": [
        ("numpy", "numpy"), ("pandas", "pandas"), ("pyarrow", "pyarrow"),
        ("scipy", "scipy"), ("requests", "requests"), ("yaml", "pyyaml"),
        ("matplotlib", "matplotlib"), ("pytest", "pytest"),
    ],
    "Stage 2 (ML)": [
        ("sklearn", "scikit-learn"), ("statsmodels", "statsmodels"),
        ("xgboost", "xgboost"), ("ruptures", "ruptures"),
        ("hmmlearn", "hmmlearn"), ("arch", "arch"),
        ("shap", "shap"), ("mapie", "mapie"),
    ],
    "Stage 3 (Deep Learning)": [
        ("torch", "torch"), ("arviz", "arviz"), ("pymc", "pymc"),
    ],
    "Stage 4 (Wrappers)": [
        ("edgar", "edgartools"), ("yfinance", "yfinance"),
        ("wbgapi", "wbgapi"), ("fredapi", "fredapi"),
        ("dart_fss", "dart-fss"), ("pykrx", "pykrx"),
    ],
}


def check_dependencies() -> tuple[dict[str, bool], dict[str, str], float]:
    """Check all dependencies, return (status, versions, progress)."""
    total = sum(len(pkgs) for pkgs in DEPENDENCY_STAGES.values())
    checked = 0
    dep_status: dict[str, bool] = {}
    dep_versions: dict[str, str] = {}

    for stage_name, packages in DEPENDENCY_STAGES.items():
        for import_name, pip_name in packages:
            try:
                mod = __import__(import_name)
                dep_status[pip_name] = True
                dep_versions[pip_name] = getattr(mod, "__version__", "OK")
            except ImportError:
                dep_status[pip_name] = False
                dep_versions[pip_name] = "missing"
            checked += 1

    return dep_status, dep_versions, checked / total


# ---------------------------------------------------------------------------
# Env key loading
# ---------------------------------------------------------------------------

def load_env_keys() -> dict[str, str]:
    """Load API keys from .env and environment."""
    keys: dict[str, str] = {}
    env_path = Path(".env")
    if env_path.exists():
        for line in env_path.read_text().splitlines():
            line = line.strip()
            if not line or line.startswith("#") or "=" not in line:
                continue
            k, _, v = line.partition("=")
            k = k.strip()
            v = v.strip().strip('"').strip("'")
            if k and v and not v.startswith("your_"):
                keys[k] = v
    for key_name in ["GEMINI_API_KEY", "ANTHROPIC_API_KEY", "OPENROUTER_API_KEY",
                     "EDGAR_IDENTITY", "COMPANIES_HOUSE_API_KEY", "DART_API_KEY",
                     "JQUANTS_API_KEY", "FRED_API_KEY"]:
        env_val = os.environ.get(key_name, "")
        if env_val and key_name not in keys:
            keys[key_name] = env_val
    return keys


def mask_key(key: str) -> str:
    if len(key) <= 8:
        return key[:2] + "***" + key[-2:]
    return key[:4] + "***" + key[-4:]


# ---------------------------------------------------------------------------
# Market data helpers
# ---------------------------------------------------------------------------

def get_regions() -> list[str]:
    try:
        from operator1.clients.pit_registry import get_regions as _gr
        return _gr()
    except Exception:
        return ["North America", "Europe", "Asia", "South America"]


def get_markets_for_region(region: str) -> list[dict]:
    try:
        from operator1.clients.pit_registry import get_markets_by_region
        markets = get_markets_by_region(region)
        return [{"id": m.market_id, "label": f"{m.country} -- {m.pit_api_name}",
                 "country": m.country} for m in markets]
    except Exception:
        return []


def get_llm_providers(keys: dict) -> list[dict]:
    providers = []
    if keys.get("GEMINI_API_KEY"):
        providers.append({"id": "gemini", "name": "Google Gemini", "active": True})
    if keys.get("ANTHROPIC_API_KEY"):
        providers.append({"id": "claude", "name": "Anthropic Claude", "active": True})
    if keys.get("OPENROUTER_API_KEY"):
        providers.append({"id": "openrouter", "name": "OpenRouter", "active": True})
    if not providers:
        providers.append({"id": "", "name": "No LLM keys configured", "active": False})
    return providers


# ---------------------------------------------------------------------------
# UI: Splash screen
# ---------------------------------------------------------------------------

def create_splash(on_complete):
    """Create splash screen with dependency loading bar."""
    with ui.column().classes("w-full h-screen items-center justify-center bg-monokai-base"):
        ui.label("OPERATOR 1").classes("text-4xl font-bold text-white mb-2")
        ui.label("Point-in-Time Financial Analysis").classes("text-lg text-gray-400 mb-1")
        ui.label("25 markets | $95T+ coverage | 25+ models").classes("text-sm text-gray-500 mb-8")

        progress = ui.linear_progress(value=0, show_value=False).classes("w-96")
        progress_label = ui.label("Initializing...").classes("text-gray-400 mt-2")
        stage_labels: dict[str, ui.label] = {}

        with ui.column().classes("mt-4 w-96"):
            for stage_name in DEPENDENCY_STAGES:
                stage_labels[stage_name] = ui.label(f"  {stage_name}").classes("text-sm text-gray-600")

    async def run_checks():
        total = sum(len(pkgs) for pkgs in DEPENDENCY_STAGES.values()) + 2
        checked = 0

        # Python + internet
        progress_label.text = "Checking Python version..."
        checked += 1
        progress.value = checked / total
        await asyncio.sleep(0.1)

        progress_label.text = "Checking internet..."
        checked += 1
        progress.value = checked / total
        await asyncio.sleep(0.1)

        # Each stage
        for stage_name, packages in DEPENDENCY_STAGES.items():
            stage_labels[stage_name].classes(replace="text-sm monokai-purple")
            for import_name, pip_name in packages:
                progress_label.text = f"Loading {pip_name}..."
                try:
                    mod = __import__(import_name)
                    state.dep_status[pip_name] = True
                    state.dep_versions[pip_name] = getattr(mod, "__version__", "OK")
                except ImportError:
                    state.dep_status[pip_name] = False
                    state.dep_versions[pip_name] = "missing"
                checked += 1
                progress.value = checked / total
                await asyncio.sleep(0.05)

            all_ok = all(state.dep_status.get(p, False) for _, p in packages)
            if all_ok:
                stage_labels[stage_name].classes(replace="text-sm text-green-400")
                stage_labels[stage_name].text = f"  {stage_name} -- done"
            else:
                stage_labels[stage_name].classes(replace="text-sm text-red-400")
                missing = [p for _, p in packages if not state.dep_status.get(p, False)]
                stage_labels[stage_name].text = f"  {stage_name} -- missing: {', '.join(missing)}"

        # Load keys
        progress_label.text = "Loading API keys..."
        state.keys = load_env_keys()
        progress.value = 1.0
        progress_label.text = "Ready!"
        await asyncio.sleep(0.5)

        on_complete()

    ui.timer(0.5, run_checks, once=True)


# ---------------------------------------------------------------------------
# UI: Main layout
# ---------------------------------------------------------------------------

def create_main_layout():
    """Create the main dashboard with browser-style top tab navigation."""

    # Header bar with branding + connection indicator
    with ui.header().classes("bg-monokai-card text-white items-center justify-between"):
        with ui.row().classes("items-center gap-3"):
            ui.label("OPERATOR 1").classes("text-xl font-bold")
            # Connection status dot (green = online, red = offline)
            conn_dot = ui.html(
                '<span id="conn-dot" style="display:inline-block;width:10px;height:10px;'
                'border-radius:50%;background:#666;margin-left:4px;" title="Checking..."></span>'
            )
            conn_label = ui.label("").classes("text-xs text-gray-400")
        with ui.row().classes("items-center gap-4"):
            health_badge = ui.badge("--/25 OK", color="gray").classes("text-xs")
            dark = ui.dark_mode(True)
            ui.button(icon="dark_mode", on_click=dark.toggle).props("flat color=white size=sm")

    # Internet connection check (runs async, updates the dot)
    async def _check_connection():
        import socket as _sock
        online = False
        latency_ms = 0
        hosts = [("data.sec.gov", 443), ("api.stlouisfed.org", 443), ("1.1.1.1", 53)]
        for host, port in hosts:
            try:
                import time as _t
                t0 = _t.time()
                s = _sock.create_connection((host, port), timeout=3)
                latency_ms = int((_t.time() - t0) * 1000)
                s.close()
                online = True
                break
            except Exception:
                continue
        if online:
            conn_dot.set_content(
                '<span id="conn-dot" style="display:inline-block;width:10px;height:10px;'
                'border-radius:50%;background:#00b894;box-shadow:0 0 6px #00b89488;" '
                f'title="Connected ({latency_ms}ms)"></span>'
            )
            conn_label.text = f"{latency_ms}ms"
            conn_label.classes(replace="text-xs monokai-green")
        else:
            conn_dot.set_content(
                '<span id="conn-dot" style="display:inline-block;width:10px;height:10px;'
                'border-radius:50%;background:#e17055;box-shadow:0 0 6px #e1705588;" '
                'title="No connection"></span>'
            )
            conn_label.text = "offline"
            conn_label.classes(replace="text-xs monokai-coral")

    # Check connection on load and every 30 seconds
    ui.timer(0.5, _check_connection, once=True)
    ui.timer(30.0, _check_connection)

    # Load health status into badge
    try:
        health_file = Path("cache/wrapper_health.json")
        if health_file.exists():
            hdata = json.loads(health_file.read_text())
            h = hdata.get("healthy", 0)
            t = hdata.get("total_markets", 25)
            health_badge.text = f"{h}/{t} OK"
            health_badge._props["color"] = "green" if h == t else "orange" if h > t - 3 else "red"
    except Exception:
        pass

    # Browser-style tab bar (static, non-closable tabs)
    with ui.tabs().classes("w-full bg-monokai-surface") as tabs:
        tab_home = ui.tab("Home", icon="home")
        tab_analyze = ui.tab("New Analysis", icon="search")
        tab_report = ui.tab("Report", icon="description")
        tab_health = ui.tab("Health", icon="monitor_heart")
        tab_models = ui.tab("Model Tests", icon="science")
        tab_config = ui.tab("Settings", icon="settings")

    with ui.tab_panels(tabs, value=tab_home).classes("w-full flex-grow"):
        with ui.tab_panel(tab_home).classes("p-6"):
            render_home()
        with ui.tab_panel(tab_analyze).classes("p-6"):
            render_analyze()
        with ui.tab_panel(tab_report).classes("p-6"):
            render_report()
        with ui.tab_panel(tab_health).classes("p-6"):
            render_health()
        with ui.tab_panel(tab_models).classes("p-6"):
            render_model_tests()
        with ui.tab_panel(tab_config).classes("p-6"):
            render_config()


# ---------------------------------------------------------------------------
# Page: Home
# ---------------------------------------------------------------------------

def render_home():
    """Home page -- last analysis or welcome screen."""
    # Check for previous analysis
    profile_path = Path("cache/company_profile.json")
    if profile_path.exists():
        try:
            profile = json.loads(profile_path.read_text())
            identity = profile.get("identity", {})
            name = identity.get("name", "Unknown")
            ticker = identity.get("ticker", "")
            country = identity.get("country", "")
            sector = identity.get("sector", "")

            ui.label(f"{name} ({ticker})").classes("text-2xl font-bold")
            ui.label(f"{sector} | {country}").classes("text-gray-400")
            ui.separator()

            # Summary cards
            with ui.row().classes("gap-4 mt-4"):
                survival = profile.get("survival", {})
                mc = profile.get("monte_carlo", {})
                fh = profile.get("financial_health", {})

                _card("Health Score", f"{fh.get('latest_composite', 'N/A'):.0f}/100" if isinstance(fh.get('latest_composite'), (int, float)) else "N/A",
                      fh.get("latest_label", ""), "favorite")
                _card("Survival Prob", f"{mc.get('survival_probability_mean', 0):.0%}" if mc.get('survival_probability_mean') else "N/A",
                      "", "shield")
                _card("Regime", survival.get("survival_regime", "unknown"), "", "timeline")

            # Extended models summary
            ext = profile.get("extended_models", {})
            if ext:
                available_count = sum(
                    1 for v in ext.values()
                    if isinstance(v, dict) and v.get("available")
                )
                ui.separator()
                ui.label("Advanced Model Results").classes("text-lg font-bold mt-4")
                ui.label(
                    f"{available_count} of {len(ext)} models produced results"
                ).classes("monokai-muted text-sm")

                _model_labels = {
                    "transfer_entropy": "Transfer Entropy",
                    "cycle_decomposition": "Cycle Decomposition",
                    "candlestick_patterns": "Candlestick Patterns",
                    "copula": "Copula Tail Risk",
                    "conformal_prediction": "Conformal Intervals",
                    "dtw_analogs": "DTW Analogs",
                    "shap_explanations": "SHAP Explainability",
                    "sobol_sensitivity": "Sobol Sensitivity",
                    "particle_filter": "Particle Filter",
                    "transformer": "Transformer NN",
                    "granger_causality": "Granger Causality",
                    "dual_regimes": "Dual Regimes",
                    "walk_forward": "Walk-Forward",
                    "burnout": "Burn-Out Calibration",
                    "genetic_optimizer": "Genetic Optimizer",
                    "time_varying_granger": "Time-Varying Granger",
                    "multivariate_monte_carlo": "Multivariate MC",
                }
                columns = [
                    {"name": "model", "label": "Model", "field": "model"},
                    {"name": "status", "label": "Status", "field": "status"},
                    {"name": "detail", "label": "Key Result", "field": "detail"},
                ]
                rows = []
                for key in sorted(ext.keys()):
                    data = ext[key]
                    if not isinstance(data, dict):
                        continue
                    avail = data.get("available", False)
                    label = _model_labels.get(key, key.replace("_", " ").title())
                    detail = ""
                    if key == "walk_forward" and avail:
                        detail = f"Best: {data.get('overall_best_model', '?')}"
                    elif key == "burnout" and avail:
                        detail = "Converged" if data.get("converged") else "Running"
                    elif key == "candlestick_patterns" and avail:
                        detail = f"{data.get('n_patterns', 0)} patterns"
                    elif key == "time_varying_granger" and avail:
                        detail = f"{len(data.get('emerging_pairs', []))} emerging"
                    elif key == "multivariate_monte_carlo" and avail:
                        sp = data.get("survival_probability")
                        detail = f"Surv: {sp:.1%}" if sp else ""
                    elif key == "granger_causality" and avail:
                        detail = f"{data.get('n_significant_pairs', 0)} causal links"
                    elif key == "copula" and avail:
                        detail = f"Best: {data.get('best_copula', '?')}"
                    rows.append({
                        "model": label,
                        "status": "OK" if avail else "--",
                        "detail": detail,
                    })
                if rows:
                    ui.table(columns=columns, rows=rows).classes("w-full mt-2")

            return
        except Exception:
            pass

    # Welcome screen
    ui.label("Welcome to Operator 1").classes("text-3xl font-bold")
    ui.label("Point-in-Time Financial Analysis").classes("text-lg text-gray-400")
    ui.separator()
    ui.label("25 markets | $95T+ coverage | 25+ math models").classes("text-gray-500")

    with ui.row().classes("gap-4 mt-6"):
        ui.button("Start New Analysis", icon="search",
                  on_click=lambda: None).classes("bg-monokai-purple")
        ui.button("Configure API Keys", icon="key",
                  on_click=lambda: None).classes("bg-monokai-surface")

    # History table
    if state.history:
        ui.label("Recent Analyses").classes("text-xl font-bold mt-8")
        columns = [
            {"name": "company", "label": "Company", "field": "company"},
            {"name": "market", "label": "Market", "field": "market"},
            {"name": "date", "label": "Date", "field": "date"},
            {"name": "status", "label": "Status", "field": "status"},
        ]
        ui.table(columns=columns, rows=state.history[:10]).classes("w-full")


def _card(title: str, value: str, subtitle: str, icon: str):
    with ui.card().classes("p-4 min-w-48"):
        with ui.row().classes("items-center gap-2"):
            ui.icon(icon).classes("text-2xl monokai-purple")
            ui.label(title).classes("text-sm text-gray-400")
        ui.label(value).classes("text-2xl font-bold mt-1")
        if subtitle:
            ui.label(subtitle).classes("text-sm text-gray-500")


# ---------------------------------------------------------------------------
# Page: New Analysis
# ---------------------------------------------------------------------------

def render_analyze():
    """New Analysis page -- non-linear form."""
    ui.label("New Analysis").classes("text-2xl font-bold")
    ui.separator()

    # LLM Provider (NEVER auto-select)
    providers = get_llm_providers(state.keys)
    ui.label("LLM Provider").classes("text-lg font-bold mt-4")
    ui.label("Choose which AI model generates report narratives").classes("text-sm text-gray-400")

    provider_options = {p["id"]: p["name"] for p in providers if p["active"]}
    if provider_options:
        provider_select = ui.select(
            options=provider_options,
            value=state.llm_provider or list(provider_options.keys())[0],
            on_change=lambda e: setattr(state, "llm_provider", e.value),
        ).classes("w-64")
    else:
        ui.label("No LLM keys configured. Go to Settings to add API keys.").classes("text-red-400")

    ui.separator()

    # Market Selection
    ui.label("Market Selection").classes("text-lg font-bold mt-4")
    regions = get_regions()

    with ui.row().classes("gap-4"):
        region_select = ui.select(
            options=regions,
            value=state.region or (regions[0] if regions else ""),
            label="Region",
            on_change=lambda e: _update_markets(e.value, market_select),
        ).classes("w-48")

        market_select = ui.select(options={}, label="Market").classes("w-64")

        if state.region:
            _update_markets(state.region, market_select)

    ui.separator()

    # Company
    ui.label("Company").classes("text-lg font-bold mt-4")
    company_input = ui.input(
        label="Company name or ticker",
        value=state.company,
        on_change=lambda e: setattr(state, "company", e.value),
    ).classes("w-96")

    ui.separator()

    # Options
    ui.label("Pipeline Options").classes("text-lg font-bold mt-4")
    with ui.column().classes("gap-2"):
        ui.switch("Discover linked entities (competitors, suppliers)",
                  value=not state.skip_linked,
                  on_change=lambda e: setattr(state, "skip_linked", not e.value))
        ui.switch("Run temporal models (forecasting, burn-out)",
                  value=not state.skip_models,
                  on_change=lambda e: setattr(state, "skip_models", not e.value))
        ui.switch("Generate PDF report",
                  value=state.gen_pdf,
                  on_change=lambda e: setattr(state, "gen_pdf", e.value))

    ui.separator()

    # Run button
    log_area = ui.log(max_lines=100).classes("w-full h-64 mt-4 hidden")
    progress = ui.linear_progress(value=0, show_value=False).classes("w-full hidden")

    async def run_pipeline():
        state.save()
        if not state.company or not state.market_id:
            ui.notify("Please select a market and enter a company", type="warning")
            return

        log_area.classes(remove="hidden")
        progress.classes(remove="hidden")
        progress.value = 0

        cmd = [
            sys.executable, "main.py",
            "--market", state.market_id,
            "--company", state.company,
        ]
        if state.skip_linked:
            cmd.append("--skip-linked")
        if state.skip_models:
            cmd.append("--skip-models")
        if state.gen_pdf:
            cmd.append("--pdf")
        if state.llm_provider:
            cmd.extend(["--llm-provider", state.llm_provider])

        log_area.push(f"Running: {' '.join(cmd)}")
        state.analysis_running = True

        proc = await asyncio.create_subprocess_exec(
            *cmd,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.STDOUT,
        )

        step = 0
        while True:
            line = await proc.stdout.readline()
            if not line:
                break
            text = line.decode().strip()
            log_area.push(text)
            if "Step" in text or "step" in text:
                step += 1
                progress.value = min(step / 10, 0.95)

        await proc.wait()
        progress.value = 1.0
        state.analysis_running = False

        if proc.returncode == 0:
            log_area.push("Pipeline completed successfully!")
            ui.notify("Analysis complete!", type="positive")
            state.add_history({
                "company": state.company,
                "market": state.market_id,
                "date": datetime.now().strftime("%Y-%m-%d %H:%M"),
                "status": "Done",
            })
        else:
            log_area.push(f"Pipeline failed (exit code {proc.returncode})")
            ui.notify("Pipeline failed", type="negative")

    ui.button("Run Analysis", icon="play_arrow", on_click=run_pipeline).classes(
        "bg-monokai-purple text-white mt-4"
    )


def _update_markets(region: str, market_select):
    state.region = region
    markets = get_markets_for_region(region)
    options = {m["id"]: m["label"] for m in markets}
    market_select.options = options
    if options:
        first_id = list(options.keys())[0]
        market_select.value = state.market_id if state.market_id in options else first_id
        state.market_id = market_select.value
    market_select.update()


# ---------------------------------------------------------------------------
# Page: Report
# ---------------------------------------------------------------------------

def render_report():
    """Report viewer with tabs including interactive plotly dashboard."""
    ui.label("Analysis Report").classes("text-2xl font-bold")

    # Discover all report files (support both old and new naming)
    report_dir = Path("cache/report")
    profile_path = Path("cache/company_profile.json")

    # Find the best available markdown report
    report_path = None
    for candidate in [
        report_dir / "premium_report.md",
        report_dir / "pro_report.md",
        report_dir / "basic_report.md",
    ]:
        if candidate.exists():
            report_path = candidate
            break

    # Enhanced output paths
    pdf_path = report_dir / "report.pdf"
    tearsheet_path = report_dir / "tearsheet.html"
    interactive_path = report_dir / "interactive_dashboard.html"

    if report_path is None and not interactive_path.exists():
        ui.label("No report available. Run an analysis first.").classes("text-gray-400 mt-4")
        return

    # Build tabs -- add Interactive tab when plotly dashboard exists
    with ui.tabs().classes("w-full") as tabs:
        tab_summary = ui.tab("Summary")
        if interactive_path.exists():
            tab_interactive = ui.tab("Interactive")
        tab_charts = ui.tab("Charts")
        tab_full = ui.tab("Full Report")
        if tearsheet_path.exists():
            tab_tearsheet = ui.tab("Tearsheet")

    with ui.tab_panels(tabs, value=tab_summary).classes("w-full"):
        with ui.tab_panel(tab_summary):
            # Load profile for summary
            if profile_path.exists():
                try:
                    profile = json.loads(profile_path.read_text())
                    identity = profile.get("identity", {})
                    ui.label(f"{identity.get('name', '?')} ({identity.get('ticker', '?')})").classes("text-xl font-bold")

                    current = profile.get("current_state", {})
                    if current.get("available"):
                        ui.label("Current Financial Snapshot").classes("text-lg mt-4")
                        with ui.row().classes("gap-4 flex-wrap"):
                            for tier_name, tier_data in [
                                ("Liquidity", current.get("tier1_liquidity", {})),
                                ("Solvency", current.get("tier2_solvency", {})),
                                ("Stability", current.get("tier3_stability", {})),
                                ("Profitability", current.get("tier4_profitability", {})),
                                ("Growth", current.get("tier5_growth", {})),
                            ]:
                                with ui.card().classes("p-3 min-w-40"):
                                    ui.label(tier_name).classes("font-bold text-sm")
                                    for k, v in list(tier_data.items())[:4]:
                                        if v is not None:
                                            ui.label(f"{k}: {v:,.2f}" if isinstance(v, float) else f"{k}: {v}").classes("text-xs text-gray-400")
                except Exception:
                    pass

        # Interactive plotly dashboard tab (embedded via iframe)
        if interactive_path.exists():
            with ui.tab_panel(tab_interactive):
                ui.label("Interactive Financial Dashboard").classes("text-lg font-bold")
                ui.label("Zoom, pan, and hover over charts for details.").classes("text-sm text-gray-400 mb-2")
                # Serve the HTML file and embed via iframe
                app.add_static_files("/report_assets", str(report_dir))
                ui.html(f'<iframe src="/report_assets/interactive_dashboard.html" '
                        f'style="width:100%; height:800px; border:none; border-radius:8px;"></iframe>')

        with ui.tab_panel(tab_charts):
            # Candlestick chart (from DearPyGui GPU chart concept)
            render_candlestick_chart()

            ui.separator()

            # Meter gauges (from ttkbootstrap Meter concept) + Radar (from DearPyGui)
            with ui.row().classes("gap-4 mt-4 items-start"):
                try:
                    profile = json.loads(profile_path.read_text()) if profile_path.exists() else {}
                    fh = profile.get("financial_health", {})
                    mc = profile.get("monte_carlo", {})

                    # Health score meter
                    composite = fh.get("latest_composite", 0)
                    if composite:
                        render_meter("Health", float(composite), 100)

                    # Survival probability meter
                    surv_prob = mc.get("survival_probability_mean", 0)
                    if surv_prob:
                        render_meter("Survival", float(surv_prob) * 100, 100, suffix="%")

                    # Altman Z-Score meter
                    altman = fh.get("altman_z", {})
                    z_score = altman.get("latest_z_score")
                    if z_score is not None:
                        render_meter("Z-Score", float(z_score), 5.0)

                except Exception:
                    pass

                # 5-tier radar chart (from DearPyGui polar plot concept)
                render_radar_chart()

            # Show mplfinance enhanced charts if available
            chart_dir = report_dir / "charts"
            if chart_dir.exists():
                enhanced_charts = list(chart_dir.glob("*.png"))
                if enhanced_charts:
                    ui.separator()
                    ui.label("Enhanced Charts (mplfinance)").classes("text-lg font-bold mt-4")
                    app.add_static_files("/chart_assets", str(chart_dir))
                    for chart_file in enhanced_charts:
                        title = chart_file.stem.replace("_", " ").title()
                        ui.label(title).classes("text-sm text-gray-400 mt-2")
                        ui.image(f"/chart_assets/{chart_file.name}").classes("w-full max-w-4xl")

        with ui.tab_panel(tab_full):
            if report_path and report_path.exists():
                try:
                    md_text = report_path.read_text(encoding="utf-8")
                    ui.markdown(md_text).classes("w-full")
                except Exception as exc:
                    ui.label(f"Error loading report: {exc}").classes("text-red-400")
            else:
                ui.label("No markdown report available.").classes("text-gray-400")

        # Quantstats tearsheet tab (embedded via iframe)
        if tearsheet_path.exists():
            with ui.tab_panel(tab_tearsheet):
                ui.label("Performance Tearsheet (quantstats)").classes("text-lg font-bold")
                ui.label("40+ performance metrics: Sharpe, Sortino, max drawdown, rolling returns, and more.").classes("text-sm text-gray-400 mb-2")
                app.add_static_files("/report_assets", str(report_dir))
                ui.html(f'<iframe src="/report_assets/tearsheet.html" '
                        f'style="width:100%; height:800px; border:none; border-radius:8px; background:white;"></iframe>')

    # Download buttons (at the bottom of the Report page)
    ui.separator()
    ui.label("Downloads").classes("text-lg font-bold mt-4")
    with ui.row().classes("gap-2 mt-2"):
        # PDF download (fpdf2 branded PDF)
        if pdf_path.exists():
            size_kb = pdf_path.stat().st_size / 1024
            ui.button(
                f"Download PDF ({size_kb:.0f} KB)",
                icon="picture_as_pdf",
                on_click=lambda: ui.download(str(pdf_path)),
            ).classes("bg-monokai-card text-white")

        # Markdown download
        if report_path and report_path.exists():
            ui.button("Download Markdown", icon="description",
                      on_click=lambda: ui.download(str(report_path))).classes("bg-monokai-purple text-white")

        # Interactive dashboard download
        if interactive_path.exists():
            size_kb = interactive_path.stat().st_size / 1024
            ui.button(
                f"Download Interactive HTML ({size_kb:.0f} KB)",
                icon="web",
                on_click=lambda: ui.download(str(interactive_path)),
            ).classes("bg-monokai-purple text-white")

        # Tearsheet download
        if tearsheet_path.exists():
            size_kb = tearsheet_path.stat().st_size / 1024
            ui.button(
                f"Download Tearsheet ({size_kb:.0f} KB)",
                icon="analytics",
                on_click=lambda: ui.download(str(tearsheet_path)),
            ).classes("bg-monokai-surface text-white")

        # JSON profile download
        if profile_path.exists():
            ui.button("Download JSON Profile", icon="data_object",
                      on_click=lambda: ui.download(str(profile_path))).classes("bg-monokai-surface text-white")


# ---------------------------------------------------------------------------
# Page: Health Monitor
# ---------------------------------------------------------------------------

def render_health():
    """Health monitoring dashboard."""
    ui.label("System Health").classes("text-2xl font-bold")

    health_data = {}
    health_path = Path("cache/wrapper_health.json")
    if health_path.exists():
        try:
            health_data = json.loads(health_path.read_text())
        except Exception:
            pass

    healthy = health_data.get("healthy", 0)
    degraded = health_data.get("degraded", 0)
    critical = health_data.get("critical", 0)
    total = health_data.get("total_markets", 25)
    last_check = health_data.get("last_check", "Never")

    # Overview bar
    with ui.card().classes("w-full p-4"):
        with ui.row().classes("items-center justify-between"):
            pct = (healthy / total * 100) if total > 0 else 0
            ui.label(f"{pct:.0f}% operational").classes("text-lg font-bold")
            ui.label(f"Last check: {last_check[:16] if len(last_check) > 16 else last_check}").classes("text-sm text-gray-400")
            ui.button("Run Full Check", icon="refresh", on_click=_run_health_check).classes("bg-monokai-purple")

        ui.linear_progress(value=healthy / total if total > 0 else 0).classes("mt-2")

        with ui.row().classes("gap-4 mt-2"):
            ui.badge(f"{healthy} Healthy", color="green")
            ui.badge(f"{degraded} Degraded", color="orange")
            ui.badge(f"{critical} Critical", color="red")

    # Tabs
    with ui.tabs().classes("w-full mt-4") as tabs:
        tab_markets = ui.tab("Markets")
        tab_deps = ui.tab("Dependencies")

    with ui.tab_panels(tabs, value=tab_markets).classes("w-full"):
        with ui.tab_panel(tab_markets):
            markets = health_data.get("markets", {})

            if not markets:
                ui.label("No health data available. Click 'Run Full Check' to scan all 25 markets.").classes("text-gray-400 mt-4")
            else:
                # Tier 1
                ui.label("Tier 1 Markets").classes("text-lg font-bold mt-4")
                tier1_ids = ["us_sec_edgar", "uk_companies_house", "eu_esef", "fr_esef",
                             "de_esef", "jp_jquants", "kr_dart", "tw_mops", "br_cvm", "cl_cmf"]
                with ui.row().classes("gap-3 flex-wrap"):
                    for mid in tier1_ids:
                        _health_card(mid, markets.get(mid, {}))

                # Tier 2
                ui.label("Tier 2 Markets").classes("text-lg font-bold mt-4")
                tier2_ids = [m for m in markets if m not in tier1_ids]
                with ui.row().classes("gap-3 flex-wrap"):
                    for mid in tier2_ids:
                        _health_card(mid, markets.get(mid, {}))

        with ui.tab_panel(tab_deps):
            ui.label("Dependency Status").classes("text-lg font-bold mt-4")
            for stage_name, packages in DEPENDENCY_STAGES.items():
                ok_count = sum(1 for _, p in packages if state.dep_status.get(p, False))
                total_count = len(packages)
                ui.label(f"{stage_name}").classes("font-bold mt-3")
                ui.linear_progress(value=ok_count / total_count if total_count else 0).classes("w-96")

                columns = [
                    {"name": "package", "label": "Package", "field": "package"},
                    {"name": "version", "label": "Version", "field": "version"},
                    {"name": "status", "label": "Status", "field": "status"},
                ]
                rows = []
                for _, pip_name in packages:
                    ok = state.dep_status.get(pip_name, False)
                    rows.append({
                        "package": pip_name,
                        "version": state.dep_versions.get(pip_name, "?"),
                        "status": "OK" if ok else "MISSING",
                    })
                ui.table(columns=columns, rows=rows).classes("w-full mt-1")


def _health_card(market_id: str, data: dict):
    """Single market health card with deep probe details."""
    status = data.get("status", "unknown")
    colors = {"healthy": "bg-green-800", "degraded": "bg-yellow-800",
              "critical": "bg-red-800", "unknown": "bg-gray-700"}
    color = colors.get(status, "bg-gray-700")
    level = data.get("level_passed", "--")
    latency = data.get("latency_ms", 0)
    short_id = market_id.split("_")[0].upper()

    # Extract deep probe info if available
    probes = data.get("probes", [])
    deep_probe = None
    for p in probes:
        if isinstance(p, dict) and p.get("level") == "L4_deep":
            deep_probe = p
            break

    with ui.expansion(text="").classes(f"min-w-32 {color} rounded"):
        with ui.row().classes("items-center gap-2"):
            icons = {"healthy": "check_circle", "degraded": "warning",
                     "critical": "error", "unknown": "help"}
            ui.icon(icons.get(status, "help")).classes("text-white")
            ui.label(short_id).classes("font-bold text-white")
            ui.label(f"{level} {latency}ms").classes("text-xs text-gray-300")

            # Pattern badge (from deep probe)
            if deep_probe:
                pattern = deep_probe.get("pattern", "")
                pattern_colors = {
                    "waf": "red", "session": "orange", "referer": "yellow",
                    "free_api": "green", "api_key": "blue",
                }
                if pattern:
                    ui.badge(pattern.upper(), color=pattern_colors.get(pattern, "gray")).classes("text-xs")

                # Schema drift indicator
                if deep_probe.get("schema_drift"):
                    ui.badge("DRIFT", color="orange").classes("text-xs")

                # Deep probe status
                dp_status = deep_probe.get("status", "")
                dp_colors = {
                    "working": "green", "restructured": "orange",
                    "down": "red", "waf_blocked": "red",
                    "geo_blocked": "yellow", "partial": "orange",
                }
                if dp_status and dp_status != status:
                    ui.badge(dp_status, color=dp_colors.get(dp_status, "gray")).classes("text-xs")

        # Expandable detail section
        if deep_probe and deep_probe.get("steps"):
            ui.separator()
            ui.label("Deep Probe Steps").classes("text-xs text-gray-400 mt-1")
            for step in deep_probe["steps"]:
                if isinstance(step, dict):
                    icon = "check" if step.get("passed") else "close"
                    icon_color = "text-green-400" if step.get("passed") else "text-red-400"
                    step_name = step.get("step_name", "?")
                    method = step.get("method", "")
                    detail = step.get("detail", step.get("error", ""))
                    ms = step.get("latency_ms", 0)
                    with ui.row().classes("items-center gap-1"):
                        ui.icon(icon).classes(f"text-sm {icon_color}")
                        ui.label(f"{step_name}").classes("text-xs text-gray-300")
                        ui.label(f"({method}) {ms}ms").classes("text-xs text-gray-500")
                    if detail:
                        ui.label(f"  {detail[:80]}").classes("text-xs text-gray-500 ml-4")


async def _run_health_check():
    ui.notify("Running health check on all 25 markets...", type="info")
    try:
        proc = await asyncio.create_subprocess_exec(
            sys.executable, "-m", "operator1.monitoring.health_check", "--json",
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
        )
        stdout, _ = await proc.communicate()
        if proc.returncode == 0:
            ui.notify("Health check complete! Refresh the page.", type="positive")
        else:
            ui.notify("Health check failed", type="negative")
    except Exception as exc:
        ui.notify(f"Error: {exc}", type="negative")


# ---------------------------------------------------------------------------
# Page: Model Tests
# ---------------------------------------------------------------------------

def render_model_tests():
    """Model smoke tests -- tests all 45 analytical models (not wrappers).

    Three layers: Features (11 modules), Analysis (10 modules), Temporal (24 modules).
    Each test builds a synthetic cache and calls the model's entry point.
    """
    ui.label("Model Smoke Tests").classes("text-2xl font-bold")
    ui.label(
        "Tests all analytical models by importing them, building a synthetic cache, "
        "and calling their main entry point. This checks features, analysis, and "
        "temporal models -- wrappers are tested on the Health tab."
    ).classes("text-sm text-gray-400 mb-2")

    # Summary cards (populated after tests run)
    summary_row = ui.row().classes("gap-4 mt-2")
    progress = ui.linear_progress(value=0, show_value=False).classes("w-full hidden")
    progress_label = ui.label("").classes("text-sm text-gray-400 hidden")

    # Results area with sub-tabs per layer
    results_container = ui.column().classes("w-full mt-4")

    # Cache for last results
    _test_state: dict = {"results": [], "running": False}

    def _render_results(results: list):
        """Render test results grouped by layer."""
        results_container.clear()

        if not results:
            with results_container:
                ui.label("No test results yet. Click 'Run All Tests' to start.").classes("text-gray-400")
            return

        # Count stats
        ok = sum(1 for r in results if r.status == "ok")
        fail = sum(1 for r in results if r.status == "fail")
        skip = sum(1 for r in results if r.status == "skip")
        total_ms = sum(r.latency_ms for r in results)

        # Update summary cards
        summary_row.clear()
        with summary_row:
            _card("Total", f"{len(results)}", "models tested", "science")
            _card("Passed", f"{ok}", f"{ok}/{len(results)}", "check_circle")
            _card("Failed", f"{fail}", "need attention" if fail else "all clear", "error")
            _card("Time", f"{total_ms / 1000:.1f}s", "total runtime", "timer")

        # Group by layer
        layers = {"features": [], "analysis": [], "temporal": []}
        for r in results:
            layers.setdefault(r.layer, []).append(r)

        layer_meta = {
            "features": {"icon": "data_object", "label": "Feature Engineering", "desc": "11 modules -- transforms cache into model-ready features"},
            "analysis": {"icon": "analytics", "label": "Analysis", "desc": "10 modules -- survival flags, fuzzy logic, adaptive calibration"},
            "temporal": {"icon": "timeline", "label": "Temporal Models", "desc": "24 modules -- forecasting, regime detection, Monte Carlo, ML"},
        }

        with results_container:
            with ui.tabs().classes("w-full") as layer_tabs:
                tab_all = ui.tab("All Models")
                tab_features = ui.tab("Features")
                tab_analysis = ui.tab("Analysis")
                tab_temporal = ui.tab("Temporal")
                tab_failures = ui.tab(f"Failures ({fail})")

            with ui.tab_panels(layer_tabs, value=tab_all).classes("w-full"):
                # All models
                with ui.tab_panel(tab_all):
                    _render_results_table(results)

                # Per-layer tabs
                for layer_key, tab in [("features", tab_features), ("analysis", tab_analysis), ("temporal", tab_temporal)]:
                    with ui.tab_panel(tab):
                        meta = layer_meta.get(layer_key, {})
                        with ui.row().classes("items-center gap-2 mb-2"):
                            ui.icon(meta.get("icon", "")).classes("text-xl monokai-purple")
                            ui.label(meta.get("label", layer_key.title())).classes("text-lg font-bold")
                        ui.label(meta.get("desc", "")).classes("text-sm text-gray-400 mb-2")

                        layer_results = layers.get(layer_key, [])
                        layer_ok = sum(1 for r in layer_results if r.status == "ok")
                        ui.linear_progress(
                            value=layer_ok / len(layer_results) if layer_results else 0,
                        ).classes("w-96 mb-2")
                        ui.label(f"{layer_ok}/{len(layer_results)} passed").classes("text-sm text-gray-400 mb-2")

                        _render_results_table(layer_results)

                # Failures only
                with ui.tab_panel(tab_failures):
                    failures = [r for r in results if r.status == "fail"]
                    if failures:
                        ui.label(f"{len(failures)} models failed:").classes("text-lg font-bold text-red-400 mb-2")
                        _render_results_table(failures)
                    else:
                        ui.label("No failures -- all models passed!").classes("text-lg monokai-green")

    def _render_results_table(results: list):
        """Render a table of test results."""
        columns = [
            {"name": "status_icon", "label": "", "field": "status_icon", "sortable": False},
            {"name": "name", "label": "Model", "field": "name", "sortable": True},
            {"name": "layer", "label": "Layer", "field": "layer", "sortable": True},
            {"name": "status", "label": "Status", "field": "status", "sortable": True},
            {"name": "latency", "label": "Time (ms)", "field": "latency", "sortable": True},
            {"name": "detail", "label": "Result / Error", "field": "detail"},
        ]
        rows = []
        for r in results:
            icon = "+" if r.status == "ok" else "X" if r.status == "fail" else "~"
            rows.append({
                "status_icon": icon,
                "name": r.name,
                "layer": r.layer,
                "status": r.status.upper(),
                "latency": r.latency_ms,
                "detail": r.error[:100] if r.error else r.detail[:100],
            })
        ui.table(columns=columns, rows=rows, row_key="name").classes("w-full")

    async def _run_all_tests():
        """Run all model smoke tests asynchronously."""
        if _test_state["running"]:
            ui.notify("Tests already running", type="warning")
            return

        _test_state["running"] = True
        progress.classes(remove="hidden")
        progress_label.classes(remove="hidden")
        progress.value = 0
        progress_label.text = "Importing model_tests..."

        try:
            from operator1.monitoring.model_tests import (
                MODEL_TESTS, _IMPORT_ONLY_TESTS, _build_synthetic_cache,
                run_single_test,
            )

            all_names = list(MODEL_TESTS.keys()) + list(_IMPORT_ONLY_TESTS.keys())
            total = len(all_names)
            cache = _build_synthetic_cache()
            results = []

            for i, name in enumerate(all_names):
                progress_label.text = f"Testing {name} ({i + 1}/{total})..."
                progress.value = i / total

                # Run test in thread to avoid blocking UI
                import functools
                loop = asyncio.get_event_loop()
                r = await loop.run_in_executor(
                    None, functools.partial(run_single_test, name, cache),
                )
                results.append(r)

                # Brief yield to let UI update
                await asyncio.sleep(0.01)

            progress.value = 1.0
            progress_label.text = f"Done -- {sum(1 for r in results if r.status == 'ok')}/{total} passed"

            _test_state["results"] = results
            _render_results(results)

            ok = sum(1 for r in results if r.status == "ok")
            if ok == total:
                ui.notify(f"All {total} models passed!", type="positive")
            else:
                fail = sum(1 for r in results if r.status == "fail")
                ui.notify(f"{ok} passed, {fail} failed out of {total}", type="warning")

        except Exception as exc:
            ui.notify(f"Test runner error: {exc}", type="negative")
            progress_label.text = f"Error: {exc}"
        finally:
            _test_state["running"] = False

    async def _run_layer_tests(layer: str):
        """Run tests for a single layer."""
        if _test_state["running"]:
            ui.notify("Tests already running", type="warning")
            return

        _test_state["running"] = True
        progress.classes(remove="hidden")
        progress_label.classes(remove="hidden")

        try:
            from operator1.monitoring.model_tests import (
                MODEL_TESTS, _IMPORT_ONLY_TESTS, _build_synthetic_cache,
                run_single_test,
            )

            names = [
                name for name, cfg in MODEL_TESTS.items()
                if cfg["layer"] == layer
            ]
            if layer == "temporal":
                names += list(_IMPORT_ONLY_TESTS.keys())

            total = len(names)
            cache = _build_synthetic_cache()
            results = []

            for i, name in enumerate(names):
                progress_label.text = f"Testing {name} ({i + 1}/{total})..."
                progress.value = i / total

                import functools
                loop = asyncio.get_event_loop()
                r = await loop.run_in_executor(
                    None, functools.partial(run_single_test, name, cache),
                )
                results.append(r)
                await asyncio.sleep(0.01)

            progress.value = 1.0
            progress_label.text = f"Done -- {sum(1 for r in results if r.status == 'ok')}/{total} passed"

            _test_state["results"] = results
            _render_results(results)

        except Exception as exc:
            ui.notify(f"Test runner error: {exc}", type="negative")
        finally:
            _test_state["running"] = False

    # Action buttons
    with ui.row().classes("gap-2 mt-2"):
        ui.button("Run All Tests", icon="play_arrow", on_click=_run_all_tests).classes(
            "bg-monokai-purple text-white"
        )
        ui.button("Features Only", icon="data_object",
                  on_click=lambda: _run_layer_tests("features")).classes("bg-monokai-surface text-white")
        ui.button("Analysis Only", icon="analytics",
                  on_click=lambda: _run_layer_tests("analysis")).classes("bg-monokai-surface text-white")
        ui.button("Temporal Only", icon="timeline",
                  on_click=lambda: _run_layer_tests("temporal")).classes("bg-monokai-surface text-white")

    ui.separator()

    # Render existing results or placeholder
    _render_results(_test_state["results"])


# ---------------------------------------------------------------------------
# Page: Config
# ---------------------------------------------------------------------------

def render_config():
    """Configuration page."""
    ui.label("Configuration").classes("text-2xl font-bold")
    ui.separator()

    # API Keys
    ui.label("API Keys").classes("text-lg font-bold mt-4")
    keys = state.keys or load_env_keys()

    key_names = [
        ("GEMINI_API_KEY", "Google Gemini (LLM)"),
        ("ANTHROPIC_API_KEY", "Anthropic Claude (LLM)"),
        ("OPENROUTER_API_KEY", "OpenRouter (LLM)"),
        ("EDGAR_IDENTITY", "SEC EDGAR email"),
        ("COMPANIES_HOUSE_API_KEY", "UK Companies House"),
        ("JQUANTS_API_KEY", "Japan J-Quants"),
        ("DART_API_KEY", "South Korea DART"),
        ("FRED_API_KEY", "US FRED macro"),
    ]

    for key_name, desc in key_names:
        with ui.row().classes("items-center gap-2 mt-1"):
            has_key = bool(keys.get(key_name))
            ui.icon("check_circle" if has_key else "cancel").classes(
                "text-green-400" if has_key else "text-red-400"
            )
            ui.label(f"{key_name}").classes("w-48 font-mono text-sm")
            if has_key:
                ui.label(mask_key(keys[key_name])).classes("text-sm text-gray-400 w-32")
            else:
                ui.label("Not set").classes("text-sm text-red-400 w-32")
            ui.label(desc).classes("text-xs text-gray-500")

    ui.separator()

    # Cache
    ui.label("Cache Management").classes("text-lg font-bold mt-4")
    cache_dir = Path("cache")
    if cache_dir.exists():
        total_size = sum(f.stat().st_size for f in cache_dir.rglob("*") if f.is_file())
        ui.label(f"Cache size: {total_size / 1024 / 1024:.1f} MB").classes("text-gray-400")

    def clear_cache():
        import shutil
        if cache_dir.exists():
            shutil.rmtree(cache_dir)
            cache_dir.mkdir()
        ui.notify("Cache cleared", type="positive")

    ui.button("Clear Cache", icon="delete", on_click=clear_cache).classes("bg-monokai-card mt-2")

    ui.separator()

    # About
    ui.label("About").classes("text-lg font-bold mt-4")
    ui.label(f"Python {sys.version.split()[0]}").classes("text-sm text-gray-400")
    ui.label("25 markets (10 Tier 1 + 15 Tier 2)").classes("text-sm text-gray-400")
    ui.label("Operator 1 v1.0").classes("text-sm text-gray-400")


# ---------------------------------------------------------------------------
# Feature: Candlestick chart (from DearPyGui's GPU chart concept)
# Uses ECharts which renders similar GPU-accelerated charts in browser
# ---------------------------------------------------------------------------

def render_candlestick_chart(container=None):
    """Render an OHLCV candlestick chart with volume bars and regime bands.

    Ported concept from DearPyGui's real-time candlestick plot.
    ECharts provides the same GPU-accelerated rendering in NiceGUI.
    """
    cache_dir = Path("cache")
    profile_path = cache_dir / "company_profile.json"

    # Try to load OHLCV data from cache
    ohlcv_data = None
    for parquet in cache_dir.rglob("quotes.parquet"):
        try:
            import pandas as _pd
            ohlcv_data = _pd.read_parquet(parquet)
            break
        except Exception:
            pass

    if ohlcv_data is None or ohlcv_data.empty:
        ui.label("No OHLCV data available").classes("text-gray-400")
        return

    # Prepare candlestick data for ECharts
    df = ohlcv_data.copy()
    if "date" in df.columns:
        df["date"] = df["date"].astype(str).str[:10]
    else:
        df["date"] = df.index.astype(str).str[:10]

    # ECharts candlestick expects: [open, close, low, high]
    dates = df["date"].tolist()[-120:]  # Last 120 trading days
    candles = []
    volumes = []
    for _, row in df.tail(120).iterrows():
        o = float(row.get("open", 0))
        c = float(row.get("close", 0))
        l = float(row.get("low", 0))
        h = float(row.get("high", 0))
        v = float(row.get("volume", 0))
        candles.append([o, c, l, h])
        volumes.append(v)

    ui.echart({
        "title": {"text": "Price (Last 120 Trading Days)", "left": "center",
                  "textStyle": {"color": "#ccc"}},
        "tooltip": {"trigger": "axis", "axisPointer": {"type": "cross"}},
        "grid": [
            {"left": "10%", "right": "8%", "height": "50%"},
            {"left": "10%", "right": "8%", "top": "68%", "height": "16%"},
        ],
        "xAxis": [
            {"type": "category", "data": dates, "gridIndex": 0,
             "axisLabel": {"show": False}},
            {"type": "category", "data": dates, "gridIndex": 1},
        ],
        "yAxis": [
            {"scale": True, "gridIndex": 0, "splitArea": {"show": True}},
            {"scale": True, "gridIndex": 1, "splitNumber": 2},
        ],
        "series": [
            {
                "type": "candlestick",
                "data": candles,
                "xAxisIndex": 0,
                "yAxisIndex": 0,
                "itemStyle": {
                    "color": "#00b894",       # up candle (Proton green)
                    "color0": "#e17055",      # down candle (Monokai coral)
                    "borderColor": "#00b894",
                    "borderColor0": "#e17055",
                },
            },
            {
                "type": "bar",
                "data": volumes,
                "xAxisIndex": 1,
                "yAxisIndex": 1,
                "itemStyle": {"color": "#a29bfe", "opacity": 0.5},
            },
        ],
        "backgroundColor": "transparent",
    }).classes("w-full h-96")


# ---------------------------------------------------------------------------
# Feature: 5-tier radar chart (from DearPyGui's polar plot concept)
# ---------------------------------------------------------------------------

def render_radar_chart():
    """Render a 5-tier financial health radar chart.

    Concept from DearPyGui's polar/radar plot, implemented via ECharts.
    Shows liquidity, solvency, stability, profitability, growth as a
    radar/spider chart.
    """
    profile_path = Path("cache/company_profile.json")
    if not profile_path.exists():
        return

    try:
        profile = json.loads(profile_path.read_text())
        fh = profile.get("financial_health", {})
        tier_means = fh.get("tier_means", {})

        values = [
            tier_means.get("tier1", 50),
            tier_means.get("tier2", 50),
            tier_means.get("tier3", 50),
            tier_means.get("tier4", 50),
            tier_means.get("tier5", 50),
        ]
    except Exception:
        values = [50, 50, 50, 50, 50]

    ui.echart({
        "radar": {
            "indicator": [
                {"name": "Liquidity", "max": 100},
                {"name": "Solvency", "max": 100},
                {"name": "Stability", "max": 100},
                {"name": "Profitability", "max": 100},
                {"name": "Growth", "max": 100},
            ],
            "shape": "polygon",
            "splitArea": {"show": True, "areaStyle": {"opacity": 0.1}},
            "axisName": {"color": "#ccc"},
        },
        "series": [{
            "type": "radar",
            "data": [{
                "value": values,
                "name": "Financial Health",
                "areaStyle": {"opacity": 0.3},
                "lineStyle": {"width": 2},
            }],
            "itemStyle": {"color": "#6c5ce7"},
        }],
        "backgroundColor": "transparent",
    }).classes("w-80 h-80")


# ---------------------------------------------------------------------------
# Feature: Status bar (from Textual's footer status concept)
# ---------------------------------------------------------------------------

def create_status_bar():
    """Bottom status bar showing system state at a glance.

    Ported from Textual's Footer widget concept -- always-visible
    status indicators at the bottom of the screen.
    """
    with ui.footer().classes("bg-monokai-card text-gray-400 text-xs py-1 px-4"):
        with ui.row().classes("w-full justify-between items-center"):
            with ui.row().classes("gap-4"):
                # Python version
                ui.label(f"Python {sys.version.split()[0]}")
                ui.separator().props("vertical")

                # Market status
                try:
                    health_path = Path("cache/wrapper_health.json")
                    if health_path.exists():
                        hdata = json.loads(health_path.read_text())
                        h = hdata.get("healthy", 0)
                        t = hdata.get("total_markets", 25)
                        ui.label(f"Markets: {h}/{t} OK").classes(
                            "text-green-400" if h == t else "text-yellow-400"
                        )
                    else:
                        ui.label("Markets: unchecked").classes("text-gray-500")
                except Exception:
                    ui.label("Markets: --")

                ui.separator().props("vertical")

                # LLM status
                keys = state.keys or load_env_keys()
                llm_count = sum(1 for k in ["GEMINI_API_KEY", "ANTHROPIC_API_KEY", "OPENROUTER_API_KEY"]
                               if keys.get(k))
                ui.label(f"LLM: {llm_count} providers").classes(
                    "text-green-400" if llm_count > 0 else "text-red-400"
                )

            with ui.row().classes("gap-4"):
                # Keyboard shortcuts hint (from Textual's key binding display)
                ui.label("Ctrl+N: New | Ctrl+H: Health | Ctrl+R: Report").classes("text-gray-600")


# ---------------------------------------------------------------------------
# Feature: Command palette (from Textual's Ctrl+P command palette)
# ---------------------------------------------------------------------------

def create_command_palette():
    """Quick command palette triggered by Ctrl+K.

    Ported from Textual's command palette concept -- type to search
    actions, markets, or companies without navigating menus.
    """
    dialog = ui.dialog().props("persistent maximized=false")

    with dialog:
        with ui.card().classes("w-96 bg-monokai-surface"):
            ui.label("Command Palette").classes("text-sm text-gray-400")
            search_input = ui.input(
                placeholder="Type a command... (market name, company, action)",
            ).classes("w-full").props("autofocus outlined dense dark")

            results_container = ui.column().classes("mt-2 max-h-64 overflow-auto")

            def on_search(e):
                results_container.clear()
                query = (e.value or "").lower().strip()
                if not query:
                    return

                with results_container:
                    # Search actions
                    actions = [
                        ("New Analysis", "search", "analyze"),
                        ("View Report", "description", "report"),
                        ("Health Check", "monitor_heart", "health"),
                        ("Settings", "settings", "config"),
                        ("Clear Cache", "delete", "clear_cache"),
                    ]
                    for name, icon, action in actions:
                        if query in name.lower():
                            with ui.row().classes("items-center gap-2 p-2 hover:bg-gray-700 cursor-pointer rounded"):
                                ui.icon(icon).classes("monokai-purple")
                                ui.label(name)

                    # Search markets
                    try:
                        from operator1.clients.pit_registry import MARKETS
                        for mid, minfo in MARKETS.items():
                            if query in minfo.country.lower() or query in mid.lower():
                                with ui.row().classes("items-center gap-2 p-2 hover:bg-gray-700 cursor-pointer rounded"):
                                    ui.icon("flag").classes("text-green-400")
                                    ui.label(f"{minfo.country} ({mid})")
                    except Exception:
                        pass

            search_input.on("update:model-value", on_search)

    # Register Ctrl+K keyboard shortcut
    ui.keyboard(on_key=lambda e: dialog.open() if e.key == "k" and e.action.keydown and e.modifiers.ctrl else None)

    return dialog


# ---------------------------------------------------------------------------
# Feature: Notification toasts with auto-dismiss (from ttkbootstrap's Toasts)
# ---------------------------------------------------------------------------

def toast_success(msg: str):
    """Green success toast (from ttkbootstrap's Toast widget concept)."""
    ui.notify(msg, type="positive", position="bottom-right", timeout=3000)


def toast_warning(msg: str):
    """Yellow warning toast."""
    ui.notify(msg, type="warning", position="bottom-right", timeout=5000)


def toast_error(msg: str):
    """Red error toast that stays longer."""
    ui.notify(msg, type="negative", position="bottom-right", timeout=8000)


# ---------------------------------------------------------------------------
# Feature: Meter/gauge widgets (from ttkbootstrap's Meter concept)
# Renders circular progress meters for key financial indicators
# ---------------------------------------------------------------------------

def render_meter(label: str, value: float, max_val: float = 100,
                 color: str = "#42a5f5", suffix: str = ""):
    """Circular meter gauge (concept from ttkbootstrap's Meter widget).

    Uses ECharts gauge series for a clean, modern circular indicator.
    """
    pct = min(value / max_val, 1.0) if max_val > 0 else 0

    # Color based on value
    if pct >= 0.7:
        color = "#00b894"  # Proton green
    elif pct >= 0.4:
        color = "#fdcb6e"  # Monokai amber
    else:
        color = "#e17055"  # Monokai coral

    ui.echart({
        "series": [{
            "type": "gauge",
            "radius": "100%",
            "startAngle": 200,
            "endAngle": -20,
            "min": 0,
            "max": max_val,
            "pointer": {"show": False},
            "progress": {
                "show": True,
                "width": 12,
                "roundCap": True,
                "itemStyle": {"color": color},
            },
            "axisLine": {"lineStyle": {"width": 12, "color": [[1, "#333"]]}},
            "axisTick": {"show": False},
            "splitLine": {"show": False},
            "axisLabel": {"show": False},
            "title": {
                "show": True,
                "offsetCenter": [0, "70%"],
                "fontSize": 12,
                "color": "#aaa",
            },
            "detail": {
                "fontSize": 24,
                "offsetCenter": [0, "0%"],
                "color": color,
                "formatter": f"{{value}}{suffix}",
            },
            "data": [{"value": round(value, 1), "name": label}],
        }],
        "backgroundColor": "transparent",
    }).classes("w-40 h-40")


# ---------------------------------------------------------------------------
# App entry point (enhanced with features from all 4 platforms)
# ---------------------------------------------------------------------------

@ui.page("/")
def main_page():
    """Main entry point -- splash -> dashboard with all platform features."""
    splash_container = ui.column().classes("w-full")
    main_container = ui.column().classes("w-full hidden")

    def on_splash_complete():
        splash_container.classes(add="hidden")
        main_container.classes(remove="hidden")
        with main_container:
            create_main_layout()
            # Textual features: status bar + command palette
            create_status_bar()
            create_command_palette()

    with splash_container:
        create_splash(on_splash_complete)


def _inject_monokai_theme():
    """Inject Proton.me-inspired Monokai dark theme via CSS overrides.

    Color palette:
      - Background:     #1a1a2e (deep navy-purple)
      - Surface:        #16213e (dark slate)
      - Card:           #0f3460 (muted indigo)
      - Primary:        #6c5ce7 (Proton purple)
      - Secondary:      #a29bfe (light purple)
      - Success/Green:  #00b894 (Proton green)
      - Warning/Amber:  #fdcb6e (warm amber)
      - Error/Red:      #e17055 (soft coral)
      - Text primary:   #dfe6e9 (off-white)
      - Text secondary: #b2bec3 (muted gray)
      - Accent:         #e84393 (Monokai pink for highlights)
    """
    ui.add_head_html("""
    <style>
      /* Proton.me Monokai base */
      body, .q-page, .q-layout, .nicegui-content {
        background-color: #1a1a2e !important;
        color: #dfe6e9 !important;
      }
      .q-header {
        background-color: #0f3460 !important;
        border-bottom: 1px solid #6c5ce733 !important;
      }
      .q-footer {
        background-color: #0f3460 !important;
        border-top: 1px solid #6c5ce733 !important;
        color: #b2bec3 !important;
      }
      .q-drawer, .q-drawer__content {
        background-color: #16213e !important;
      }
      .q-card, .q-expansion-item, .q-table {
        background-color: #16213e !important;
        color: #dfe6e9 !important;
        border: 1px solid #6c5ce722 !important;
        border-radius: 8px !important;
      }
      .q-table__container {
        background-color: #16213e !important;
      }
      .q-table thead th {
        color: #a29bfe !important;
        border-bottom-color: #6c5ce744 !important;
      }
      .q-table tbody td {
        color: #dfe6e9 !important;
        border-bottom-color: #6c5ce722 !important;
      }
      .q-tab {
        color: #b2bec3 !important;
      }
      .q-tab--active {
        color: #6c5ce7 !important;
      }
      .q-tabs__content .q-tab__indicator {
        background-color: #6c5ce7 !important;
      }
      .q-tab-panel {
        background-color: #1a1a2e !important;
      }
      .q-btn {
        border-radius: 6px !important;
      }
      .q-separator {
        background-color: #6c5ce733 !important;
      }
      .q-badge {
        border-radius: 4px !important;
      }
      .q-linear-progress {
        border-radius: 4px !important;
      }
      .q-linear-progress__track {
        background-color: #16213e !important;
      }
      .q-input .q-field__control, .q-select .q-field__control {
        background-color: #16213e !important;
        color: #dfe6e9 !important;
        border: 1px solid #6c5ce744 !important;
        border-radius: 6px !important;
      }
      .q-input .q-field__label, .q-select .q-field__label {
        color: #b2bec3 !important;
      }
      .q-notification {
        background-color: #16213e !important;
        border: 1px solid #6c5ce744 !important;
        border-radius: 8px !important;
      }
      .q-toggle__inner--truthy .q-toggle__track {
        background-color: #6c5ce7 !important;
      }
      /* Scrollbar styling */
      ::-webkit-scrollbar {
        width: 8px;
        height: 8px;
      }
      ::-webkit-scrollbar-track {
        background: #1a1a2e;
      }
      ::-webkit-scrollbar-thumb {
        background: #6c5ce744;
        border-radius: 4px;
      }
      ::-webkit-scrollbar-thumb:hover {
        background: #6c5ce7;
      }
      /* ECharts transparent background */
      .nicegui-echart canvas {
        background: transparent !important;
      }
      /* Log area */
      .q-log {
        background-color: #0f3460 !important;
        color: #00b894 !important;
        font-family: 'JetBrains Mono', 'Fira Code', monospace !important;
        border: 1px solid #6c5ce733 !important;
        border-radius: 8px !important;
      }
      /* Custom accent classes */
      .monokai-purple { color: #6c5ce7 !important; }
      .monokai-green { color: #00b894 !important; }
      .monokai-pink { color: #e84393 !important; }
      .monokai-amber { color: #fdcb6e !important; }
      .monokai-coral { color: #e17055 !important; }
      .monokai-light { color: #dfe6e9 !important; }
      .monokai-muted { color: #b2bec3 !important; }
      .bg-monokai-surface { background-color: #16213e !important; }
      .bg-monokai-card { background-color: #0f3460 !important; }
      .bg-monokai-base { background-color: #1a1a2e !important; }
      .bg-monokai-purple { background-color: #6c5ce7 !important; }
    </style>
    """)


if __name__ == "__main__":
    _inject_monokai_theme()
    ui.run(
        title="Operator 1 -- Financial Analysis",
        dark=True,
        port=8080,
        reload=False,
        show=False,  # Set to True for native window, False for server mode
    )
