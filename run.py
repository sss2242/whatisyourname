#!/usr/bin/env python3
"""Operator 1 -- Interactive Terminal Launcher.

A user-friendly interface that guides you through running the
financial analysis pipeline step by step.

Flow:
  1. System checks (Python version, internet)
  2. Dependency check
  3. LLM provider & model selection (always prompt first)
  4. Data source mode: wrappers-only vs API + wrappers
  5. If API + wrappers: prompt for market-specific API keys
  6. Company + country input
  7. LLM resolves the right market/client for the company
  8. Pipeline options + confirmation
  9. Run pipeline

Usage:
    python run.py
"""

from __future__ import annotations

import json
import os
import sys
import time
import socket
import subprocess
from pathlib import Path


# ---------------------------------------------------------------------------
# Terminal helpers
# ---------------------------------------------------------------------------

def _clear():
    os.system("cls" if os.name == "nt" else "clear")


def _color(text: str, code: str) -> str:
    """Wrap text in ANSI color codes (no-op on Windows without colorama)."""
    if os.name == "nt":
        return text
    return f"\033[{code}m{text}\033[0m"


def _green(t: str) -> str:
    return _color(t, "32")


def _yellow(t: str) -> str:
    return _color(t, "33")


def _red(t: str) -> str:
    return _color(t, "31")


def _cyan(t: str) -> str:
    return _color(t, "36")


def _bold(t: str) -> str:
    return _color(t, "1")


def _dim(t: str) -> str:
    return _color(t, "2")


def _banner():
    print("")
    print(_bold(_cyan("  ================================================================")))
    print(_bold(_cyan("     OPERATOR 1 -- Point-in-Time Financial Analysis")))
    print(_bold(_cyan("  ================================================================")))
    print(_dim("  Institutional-grade equity research from free government filings"))
    print(_dim("  10 markets | $91T+ coverage | 25+ math models | PIT-audited data"))
    print("")


def _separator():
    print(_dim("  " + "-" * 60))


def _step(num: int, title: str):
    print("")
    print(_bold(f"  [{num}] {title}"))
    print("")


def _ok(msg: str):
    print(f"  {_green('[OK]')} {msg}")


def _warn(msg: str):
    print(f"  {_yellow('[!]')} {msg}")


def _err(msg: str):
    print(f"  {_red('[ERROR]')} {msg}")


def _info(msg: str):
    print(f"  {_dim('[i]')} {msg}")


def _prompt(msg: str, default: str = "") -> str:
    suffix = f" [{default}]" if default else ""
    try:
        value = input(f"  > {msg}{suffix}: ").strip()
    except (EOFError, KeyboardInterrupt):
        print("")
        sys.exit(0)
    return value if value else default


def _yes_no(msg: str, default: bool = True) -> bool:
    suffix = "[Y/n]" if default else "[y/N]"
    try:
        value = input(f"  > {msg} {suffix}: ").strip().lower()
    except (EOFError, KeyboardInterrupt):
        print("")
        sys.exit(0)
    if not value:
        return default
    return value in ("y", "yes")


# ---------------------------------------------------------------------------
# Checks
# ---------------------------------------------------------------------------

def check_python_version() -> bool:
    v = sys.version_info
    if v.major >= 3 and v.minor >= 12:
        _ok(f"Python {v.major}.{v.minor}.{v.micro}")
        return True
    elif v.major >= 3 and v.minor >= 10:
        _warn(f"Python {v.major}.{v.minor}.{v.micro} (3.12+ required, some features may fail)")
        return True
    else:
        _err(f"Python {v.major}.{v.minor}.{v.micro} -- need 3.12+")
        _info("Download from: https://www.python.org/downloads/")
        return False


def check_internet() -> bool:
    """Check internet connectivity."""
    hosts = [
        ("data.sec.gov", 443),
        ("api.stlouisfed.org", 443),
        ("1.1.1.1", 53),
    ]
    for host, port in hosts:
        try:
            sock = socket.create_connection((host, port), timeout=5)
            sock.close()
            _ok(f"Internet connection active (reached {host})")
            return True
        except (socket.timeout, socket.error, OSError):
            continue
    _err("No internet connection detected")
    _info("This pipeline requires internet to fetch financial data.")
    return False


def check_dependencies() -> tuple[bool, list[str]]:
    """Check which key Python packages are installed."""
    required = {
        "requests": "HTTP requests",
        "pandas": "Data processing",
        "numpy": "Numerical computing",
        "yaml": "Config loading (pyyaml)",
    }
    optional = {
        "statsmodels": "Kalman filter, VAR models",
        "arch": "GARCH volatility models",
        "sklearn": "Tree ensembles, imputation",
        "torch": "LSTM deep learning",
        "hmmlearn": "Hidden Markov Models",
        "ruptures": "Structural break detection",
        "xgboost": "XGBoost tree ensemble",
        "matplotlib": "Chart generation",
        "dotenv": "python-dotenv (.env loading)",
    }

    missing_required: list[str] = []

    for pkg, desc in required.items():
        try:
            __import__(pkg)
            _ok(f"{pkg} -- {desc}")
        except ImportError:
            _err(f"{pkg} -- {desc} [MISSING]")
            missing_required.append(pkg)

    for pkg, desc in optional.items():
        try:
            __import__(pkg)
            _ok(f"{pkg} -- {desc}")
        except ImportError:
            _warn(f"{pkg} -- {desc} [not installed, some features limited]")

    return len(missing_required) == 0, missing_required


def _mask_key(key: str) -> str:
    """Show first 4 and last 4 chars of a key, mask the rest."""
    if len(key) <= 8:
        return key[:2] + "..." + key[-2:]
    return key[:4] + "..." + key[-4:]


# ---------------------------------------------------------------------------
# LLM key setup (always prompt first)
# ---------------------------------------------------------------------------

def _select_llm_model(provider: str) -> str:
    """Show available models for the chosen provider and let the user pick.

    Returns the model name string, or empty string to use the default.
    """
    try:
        from operator1.clients.llm_factory import get_available_models
        models = get_available_models(provider)
    except Exception:
        return ""

    if not models:
        return ""

    print("")
    print(_bold(f"  Available {provider.title()} models:"))
    print("")
    for idx, m in enumerate(models, 1):
        ctx = m["context_window"]
        out = m["max_output_tokens"]
        tier = m["tier"]
        # Human-readable context/output sizes
        ctx_str = f"{ctx // 1_000_000}M" if ctx >= 1_000_000 else f"{ctx // 1_000}K"
        out_str = f"{out // 1_000}K" if out >= 1_000 else str(out)
        default_marker = " (default)" if idx == 1 else ""
        print(f"    {_bold(str(idx))}. {m['name']}")
        print(f"       {_dim(f'{tier} | {ctx_str} context | {out_str} output')}{_green(default_marker)}")
    print("")

    choice = _prompt(f"Choose model (1-{len(models)})", "1")
    try:
        sel = int(choice) - 1
        if 0 <= sel < len(models):
            selected = models[sel]["name"]
            _ok(f"Model: {selected}")
            return selected
    except ValueError:
        # Try matching by name
        for m in models:
            if choice.lower() in m["name"].lower():
                _ok(f"Model: {m['name']}")
                return m["name"]

    # Default: first model
    _ok(f"Model: {models[0]['name']} (default)")
    return models[0]["name"]


def setup_llm_keys() -> dict[str, str]:
    """Prompt for LLM API keys. Always runs at the start of the flow.

    Returns a dict of all loaded keys (LLM + any from .env).
    """
    keys: dict[str, str] = {}
    env_path = Path(__file__).resolve().parent / ".env"

    # Load existing keys from .env if present
    if env_path.exists():
        with open(env_path, "r") as f:
            for line in f:
                line = line.strip()
                if not line or line.startswith("#") or "=" not in line:
                    continue
                k, _, v = line.partition("=")
                k = k.strip()
                v = v.strip().strip('"').strip("'").strip()
                if k and v and not v.startswith("your_"):
                    keys[k] = v

    # Also pull from environment variables
    for key_name in ["GEMINI_API_KEY", "ANTHROPIC_API_KEY", "OPENROUTER_API_KEY"]:
        if key_name not in keys:
            env_val = os.environ.get(key_name)
            if env_val and env_val.strip():
                keys[key_name] = env_val.strip()

    has_gemini = "GEMINI_API_KEY" in keys
    has_claude = "ANTHROPIC_API_KEY" in keys
    has_openrouter = "OPENROUTER_API_KEY" in keys

    if has_gemini:
        n_gemini = len([k for k in keys["GEMINI_API_KEY"].split(",") if k.strip()])
        key_info = f" ({n_gemini} keys for rotation)" if n_gemini > 1 else ""
        _ok(f"GEMINI_API_KEY: {_mask_key(keys['GEMINI_API_KEY'].split(',')[0].strip())}{key_info} (loaded from .env)")
    if has_claude:
        n_claude = len([k for k in keys["ANTHROPIC_API_KEY"].split(",") if k.strip()])
        key_info = f" ({n_claude} keys for rotation)" if n_claude > 1 else ""
        _ok(f"ANTHROPIC_API_KEY: {_mask_key(keys['ANTHROPIC_API_KEY'].split(',')[0].strip())}{key_info} (loaded from .env)")
    if has_openrouter:
        n_or = len([k for k in keys["OPENROUTER_API_KEY"].split(",") if k.strip()])
        key_info = f" ({n_or} keys for rotation)" if n_or > 1 else ""
        _ok(f"OPENROUTER_API_KEY: {_mask_key(keys['OPENROUTER_API_KEY'].split(',')[0].strip())}{key_info} (loaded from .env)")

    n_providers = sum([has_gemini, has_claude, has_openrouter])
    if n_providers >= 2:
        _ok(f"{n_providers} LLM providers available (cross-provider fallback enabled)")
    elif n_providers == 1:
        _ok("LLM provider available")
    else:
        # No LLM keys found -- prompt the user
        print(_dim("  An LLM API key is required for smart market routing and"))
        print(_dim("  AI-generated report narratives."))
        print("")
        print(_dim("  Tip: You can enter MULTIPLE keys (comma-separated) for automatic"))
        print(_dim("  key rotation. When one key hits rate limits or credit exhaustion,"))
        print(_dim("  the system rotates to the next key automatically."))
        print(_dim("  You can also provide keys for BOTH providers for cross-provider fallback."))
        print("")
        print(f"    {_bold('1')}. Google Gemini    -- https://aistudio.google.com/apikey")
        print(f"    {_bold('2')}. Anthropic Claude  -- https://console.anthropic.com/")
        print(f"    {_bold('3')}. OpenRouter (200+ models, free tier available) -- https://openrouter.ai/")
        print(f"    {_bold('4')}. Multiple providers (recommended for key rotation)")
        print(f"    {_bold('5')}. Skip (limited functionality, template reports only)")
        print("")

        llm_choice = _prompt("Choose LLM provider (1/2/3/4/5)", "1")
        if llm_choice in ("1", "4"):
            value = _prompt("Enter GEMINI_API_KEY(s) (comma-separate multiple keys)")
            if value:
                keys["GEMINI_API_KEY"] = value.strip()
                _save_key_to_env(env_path, "GEMINI_API_KEY", value.strip())
                n_keys = len([k for k in value.split(",") if k.strip()])
                _ok(f"GEMINI_API_KEY saved ({n_keys} key{'s' if n_keys > 1 else ''})")
        if llm_choice in ("2", "4"):
            value = _prompt("Enter ANTHROPIC_API_KEY(s) (comma-separate multiple keys)")
            if value:
                keys["ANTHROPIC_API_KEY"] = value.strip()
                _save_key_to_env(env_path, "ANTHROPIC_API_KEY", value.strip())
                n_keys = len([k for k in value.split(",") if k.strip()])
                _ok(f"ANTHROPIC_API_KEY saved ({n_keys} key{'s' if n_keys > 1 else ''})")
        if llm_choice in ("3", "4"):
            value = _prompt("Enter OPENROUTER_API_KEY")
            if value:
                keys["OPENROUTER_API_KEY"] = value.strip()
                _save_key_to_env(env_path, "OPENROUTER_API_KEY", value.strip())
                _ok("OPENROUTER_API_KEY saved")
        if llm_choice == "5":
            _warn("Skipping LLM setup. Smart routing disabled, template reports only.")

    # Determine which LLM provider to use
    has_gemini = "GEMINI_API_KEY" in keys
    has_claude = "ANTHROPIC_API_KEY" in keys
    has_openrouter = "OPENROUTER_API_KEY" in keys
    llm_provider = ""

    available_providers = []
    if has_gemini:
        available_providers.append(("gemini", "Google Gemini"))
    if has_claude:
        available_providers.append(("claude", "Anthropic Claude"))
    if has_openrouter:
        available_providers.append(("openrouter", "OpenRouter"))

    if len(available_providers) > 1:
        print("")
        print(_bold("  Which LLM provider to use for this session?"))
        for i, (pid, pname) in enumerate(available_providers, 1):
            print(f"    {_bold(str(i))}. {pname}")
        print("")
        prov = _prompt(f"Choose (1-{len(available_providers)})", "1")
        try:
            idx = int(prov) - 1
            if 0 <= idx < len(available_providers):
                llm_provider = available_providers[idx][0]
            else:
                llm_provider = available_providers[0][0]
        except ValueError:
            llm_provider = available_providers[0][0]
    elif len(available_providers) == 1:
        llm_provider = available_providers[0][0]

    # --- Model selection ---
    llm_model = ""
    if llm_provider:
        _ok(f"LLM provider: {llm_provider}")
        os.environ["LLM_PROVIDER"] = llm_provider
        keys["_llm_provider"] = llm_provider

        llm_model = _select_llm_model(llm_provider)
        if llm_model:
            os.environ["LLM_MODEL"] = llm_model
            keys["_llm_model"] = llm_model

    # Set keys in environment
    for k, v in keys.items():
        if not k.startswith("_"):
            os.environ[k] = v

    return keys


def _save_key_to_env(env_path: Path, key_name: str, value: str) -> None:
    """Append a key to the .env file."""
    try:
        with open(env_path, "a") as f:
            f.write(f"\n{key_name}={value}\n")
    except Exception:
        pass


# ---------------------------------------------------------------------------
# Data source mode selection
# ---------------------------------------------------------------------------

def choose_data_source_mode() -> str:
    """Let the user choose between wrappers-only or API + wrappers.

    Returns 'wrappers' or 'api_and_wrappers'.
    """
    print(_bold("  Choose data source mode:"))
    print("")
    print(f"    {_bold('1')}. {_green('Standard')} (recommended)")
    print("       Uses wrapper libraries with automatic gov API fallback")
    print(f"       {_dim('No extra API keys needed. Simplest setup.')}")
    print("")
    print(f"    {_bold('2')}. {_yellow('Enhanced (may need API keys)')}")
    print("       Prompts for market-specific API keys for direct gov API access")
    print(f"       {_dim('All API keys are free. Provides richer data coverage.')}")
    print("")

    choice = _prompt("Select mode (1/2)", "1")
    if choice == "2":
        _ok("Mode: Enhanced (API keys + wrappers)")
        return "api_and_wrappers"
    else:
        _ok("Mode: Standard (auto wrapper + gov API fallback)")
        return "wrappers"


def setup_market_api_keys(keys: dict[str, str]) -> dict[str, str]:
    """Prompt for market-specific API keys when using API + wrappers mode."""
    env_path = Path(__file__).resolve().parent / ".env"

    # Required market keys (needed for specific countries)
    required_keys = [
        ("EDGAR_IDENTITY", "US SEC EDGAR email identity", "e.g. your.name@example.com"),
        ("JQUANTS_API_KEY", "Japan J-Quants API key", "https://jpx-jquants.com/login"),
        ("COMPANIES_HOUSE_API_KEY", "UK Companies House", "https://developer.company-information.service.gov.uk/"),
        ("DART_API_KEY", "South Korea DART", "https://opendart.fss.or.kr/"),
    ]

    # Optional keys (for enhanced features)
    optional_keys = [
        ("openfigi_key", "OpenFIGI (higher rate limits)", "https://www.openfigi.com/api"),
        ("FRED_API_KEY", "US FRED macro data", "https://fred.stlouisfed.org/docs/api/api_key.html"),
    ]

    # Also pull from environment for all keys
    for key_name, _, _ in required_keys + optional_keys:
        if key_name not in keys:
            env_val = os.environ.get(key_name)
            if env_val and env_val.strip():
                keys[key_name] = env_val.strip()

    print(_bold("  Required API keys / identities:"))
    print("")

    for key_name, desc, url in required_keys:
        if key_name in keys:
            _ok(f"{key_name}: {_mask_key(keys[key_name])} ({desc})")
        else:
            _info(f"{key_name}: not set -- {desc}")
            _info(f"  Register: {url}")
            value = _prompt(f"Enter {desc} (or press Enter to skip)")
            if value:
                keys[key_name] = value.strip()
                os.environ[key_name] = value.strip()
                _save_key_to_env(env_path, key_name, value.strip())
                _ok(f"Saved {key_name}")

    print("")
    print(_bold("  Optional API keys:"))
    print("")

    for key_name, desc, url in optional_keys:
        if key_name in keys:
            _ok(f"{key_name}: {_mask_key(keys[key_name])} ({desc})")
        else:
            _info(f"{key_name}: not set -- {desc}")

    print("")
    if _yes_no("Enter any optional API keys now?", default=False):
        for key_name, desc, url in optional_keys:
            if key_name not in keys:
                value = _prompt(f"{desc} key (or press Enter to skip)")
                if value:
                    keys[key_name] = value.strip()
                    os.environ[key_name] = value.strip()
                    _save_key_to_env(env_path, key_name, value.strip())
                    _ok(f"Saved {key_name}")

    return keys


# ---------------------------------------------------------------------------
# LLM-driven market routing
# ---------------------------------------------------------------------------

def _try_company_lookup(
    company: str,
    market_id: str,
    keys: dict[str, str],
) -> bool:
    """Quick check whether the company can be found via the PIT wrapper.

    Returns True if the wrapper recognizes the company, False otherwise.
    This is a lightweight probe -- it does NOT fetch full financials.
    """
    try:
        from operator1.clients.equity_provider import create_pit_client
        secrets = {k: v for k, v in keys.items() if not k.startswith("_")}
        client = create_pit_client(market_id, secrets)
        if client is None:
            return False
        profile = client.get_profile(company)
        return bool(profile and profile.get("name"))
    except Exception:
        return False


def _llm_resolve_company(
    company: str,
    market_id: str,
    market,
    llm_provider: str,
    keys: dict[str, str],
) -> str | None:
    """Ask the LLM to resolve a company name/ticker for a specific market.

    Returns the corrected ticker/name string, or None if the LLM cannot help.
    """
    if not llm_provider:
        return None

    # Market-specific LLM resolution instructions
    _LLM_MARKET_HINTS: dict[str, str] = {
        "cn_sse": (
            "The data source (baostock) only accepts Chinese stock ticker codes "
            "(e.g. 600519 for Kweichow Moutai) or Chinese company names "
            "(e.g. 贵州茅台). If the user provided an English name, please "
            "return the 6-digit SSE/SZSE ticker code."
        ),
        "in_bse": (
            "The data source (BSE India) accepts BSE scrip codes "
            "(e.g. 500325 for Reliance), NSE ticker symbols (e.g. RELIANCE, TCS, INFY), "
            "or company names. Please return the BSE ticker symbol."
        ),
        "ca_sedar": (
            "The data source (TMX/TSX) accepts TSX ticker symbols "
            "(e.g. RY for Royal Bank, SHOP for Shopify, ENB for Enbridge). "
            "Please return the TSX ticker symbol."
        ),
    }
    extra_hint = _LLM_MARKET_HINTS.get(market_id, "")

    hint_block = f"\n\n{extra_hint}" if extra_hint else ""
    prompt = (
        f"I am searching for the company '{company}' on the "
        f"{market.country} {market.exchange} exchange (data source: {market.pit_api_name}).\n\n"
        f"The search failed. Can you tell me the correct ticker symbol or "
        f"official company name that this exchange/data source would recognize?"
        f"{hint_block}\n\n"
        f"Return ONLY the ticker or name -- no explanation, no punctuation, no quotes."
    )

    try:
        from operator1.clients.llm_factory import create_llm_client
        client = create_llm_client(keys, provider=llm_provider)
        if client is None:
            return None
        response = client.generate(prompt)
        if response:
            return response.strip().strip("'\"")
    except Exception as exc:
        _warn(f"LLM company resolution failed: {exc}")

    return None





# ---------------------------------------------------------------------------
# Pipeline options
# ---------------------------------------------------------------------------

def _check_user_input_pii(company: str, country: str, keys: dict[str, str]) -> None:
    """Use LLM + regex to check if user input contains personal data."""
    # PII guard removed -- user inputs API keys and emails at startup


def _check_market_pii(market_id: str, market) -> None:
    """Warn the user if the resolved market's API registration requires personal data."""
    # PII guard removed -- API key prompts happen at startup


def estimate_runtime(skip_linked: bool, skip_models: bool) -> str:
    """Rough time estimate based on options."""
    if skip_models and skip_linked:
        return "~2-5 minutes (data fetch + features only)"
    elif skip_models:
        return "~5-10 minutes (data + features + linked entities)"
    elif skip_linked:
        return "~15-30 minutes (models without linked entities)"
    else:
        return "~30-60 minutes (full analysis with all models)"


# ---------------------------------------------------------------------------
# Main interactive flow
# ---------------------------------------------------------------------------

def main() -> int:
    _clear()
    _banner()

    # ------------------------------------------------------------------
    # Step 1: System checks
    # ------------------------------------------------------------------
    _step(1, "System Checks")

    if not check_python_version():
        return 1

    _separator()

    if not check_internet():
        if not _yes_no("Continue without internet? (pipeline will likely fail)"):
            return 1

    # ------------------------------------------------------------------
    # Step 2: Dependencies
    # ------------------------------------------------------------------
    _step(2, "Checking Dependencies")

    deps_ok, missing = check_dependencies()

    if not deps_ok:
        print("")
        _err(f"Missing required packages: {', '.join(missing)}")
        if _yes_no("Install missing packages with pip?"):
            subprocess.run(
                [sys.executable, "-m", "pip", "install", "-r", "requirements.txt", "--quiet"],
                check=False,
            )
            print("")
            _info("Re-checking dependencies...")
            deps_ok, missing = check_dependencies()
            if not deps_ok:
                _err("Still missing packages. Install manually: pip install -r requirements.txt")
                return 1
        else:
            _info("Run: pip install -r requirements.txt")
            return 1

    # ------------------------------------------------------------------
    # Step 3: LLM Provider & Model Selection (always prompt first)
    # ------------------------------------------------------------------
    _step(3, "LLM Provider & Model Selection")

    keys = setup_llm_keys()
    llm_provider = keys.get("_llm_provider", "")

    # ------------------------------------------------------------------
    # Step 4: Data Source Mode
    # ------------------------------------------------------------------
    _step(4, "Data Source Mode")

    data_mode = choose_data_source_mode()

    # ------------------------------------------------------------------
    # Step 5: Market-specific API keys (only if API + wrappers)
    # ------------------------------------------------------------------
    if data_mode == "api_and_wrappers":
        _step(5, "Market API Keys")
        keys = setup_market_api_keys(keys)
    else:
        _step(5, "Data Source Check")
        _ok("Using community wrapper libraries only (no extra API keys needed).")

    # ------------------------------------------------------------------
    # Step 6: Region + Company Selection (region-first flow)
    # ------------------------------------------------------------------
    _step(6, "Region & Company Selection")

    # 6a: Region selection
    sys.path.insert(0, str(Path(__file__).resolve().parent))
    from operator1.clients.pit_registry import (
        get_regions, get_markets_by_region, MARKETS,
    )

    regions = get_regions()
    print(_bold("  Choose the region where the company is listed:"))
    print("")
    for i, region in enumerate(regions, 1):
        markets = get_markets_by_region(region)
        countries = ", ".join(m.country for m in markets)
        print(f"    {_bold(str(i))}. {region}")
        print(f"       {_dim(countries)}")
    print("")

    region_choice = _prompt(f"Select region (1-{len(regions)})", "1")
    try:
        region_idx = int(region_choice) - 1
        if 0 <= region_idx < len(regions):
            selected_region = regions[region_idx]
        else:
            selected_region = regions[0]
    except ValueError:
        selected_region = regions[0]

    region_markets = get_markets_by_region(selected_region)
    _ok(f"Region: {selected_region}")

    # 6b: Market selection within region
    if len(region_markets) > 1:
        print("")
        print(_bold(f"  Markets in {selected_region}:"))
        print("")
        for i, m in enumerate(region_markets, 1):
            key_note = _dim(" (API key required)") if m.requires_api_key else _dim(" (no key needed)")
            print(f"    {_bold(str(i))}. {m.country} -- {m.exchange}")
            print(f"       Data source: {m.pit_api_name}{key_note}")
        print("")
        mkt_choice = _prompt(f"Select market (1-{len(region_markets)})", "1")
        try:
            mkt_idx = int(mkt_choice) - 1
            if 0 <= mkt_idx < len(region_markets):
                selected_market = region_markets[mkt_idx]
            else:
                selected_market = region_markets[0]
        except ValueError:
            selected_market = region_markets[0]
    else:
        selected_market = region_markets[0]

    market_id = selected_market.market_id
    country = selected_market.country
    _ok(f"Market: {selected_market.country} ({selected_market.pit_api_name})")

    # 6c: Company input with market-specific hints
    _INPUT_HINTS: dict[str, str] = {
        "us_sec_edgar": "Enter a ticker (AAPL), CIK number (320193), or company name (Apple Inc)",
        "uk_companies_house": "Enter a company name (Unilever) or Companies House number (00041424)",
        "eu_esef": "Enter a company name (Siemens) or LEI code",
        "fr_esef": "Enter a company name (LVMH) or SIREN number",
        "de_esef": "Enter a company name (BMW) or LEI code",
        "jp_jquants": "Enter a Japanese ticker code (7203) or company name (Toyota Motor)",
        "kr_dart": "Enter a Korean stock code (005930) or company name (Samsung Electronics)",
        "tw_mops": "Enter a Taiwan stock code (2330) or company name (TSMC)",
        "br_cvm": "Enter a ticker (PETR4) or company name (Petrobras)",
        "cl_cmf": "Enter a ticker (SQM-B) or company name (SQM)",
        "cn_sse": "Enter a ticker code (600519) or Chinese name (贵州茅台). English names will be resolved via LLM.",
        "in_bse": "Enter a BSE scrip code (500325), ticker (RELIANCE), or company name (Tata Steel)",
        "ca_sedar": "Enter a TSX symbol (RY, SHOP, ENB) or company name (Royal Bank)",
    }
    hint = _INPUT_HINTS.get(market_id, "Enter a ticker or company name")

    print("")
    print(_bold("  Enter the company you want to analyze:"))
    print(f"  {_dim(hint)}")
    print("")
    company = _prompt("Company")
    if not company:
        _err("Company is required.")
        return 1
    _ok(f"Company: {company}")

    # --- Personal data check on user input ---
    _check_user_input_pii(company, country, keys)

    # ------------------------------------------------------------------
    # Step 7: LLM-assisted company validation (retry up to 2x)
    # ------------------------------------------------------------------
    _step(7, "Company Validation")

    _info(f"Validating '{company}' in {selected_market.country} ({selected_market.pit_api_name})...")

    # Try to verify the company exists via the wrapper first
    company_validated = False
    final_company = company
    for attempt in range(1, 3):  # max 2 attempts
        if _try_company_lookup(final_company, market_id, keys):
            company_validated = True
            _ok(f"Company '{final_company}' found via {selected_market.pit_api_name}")
            break

        if attempt == 1:
            _warn(f"Could not find '{final_company}' directly. Asking LLM for help...")
        else:
            _warn(f"LLM suggestion '{final_company}' also not found.")
            break

        # Ask LLM to resolve the company
        llm_suggestion = _llm_resolve_company(
            final_company, market_id, selected_market, llm_provider, keys,
        )
        if llm_suggestion and llm_suggestion.lower() != final_company.lower():
            _info(f"LLM suggests: '{llm_suggestion}'")
            final_company = llm_suggestion
        else:
            _warn("LLM could not resolve the company. Proceeding with original input.")
            break

    if company_validated:
        company = final_company
    else:
        _warn(f"Could not validate '{company}'. Proceeding anyway -- the pipeline will attempt to fetch data.")

    from operator1.clients.pit_registry import get_market, get_macro_api_for_market
    market = get_market(market_id)
    macro = get_macro_api_for_market(market_id)

    if not market:
        _err(f"Unknown market: {market_id}")
        return 1

    if macro:
        _info(f"Macro data: {macro.api_name}")
    _check_market_pii(market_id, market)

    # ------------------------------------------------------------------
    # Step 8: Pipeline Options
    # ------------------------------------------------------------------
    _step(8, "Pipeline Options")

    skip_linked = not _yes_no(
        "Discover linked entities? (competitors, suppliers)", default=True
    )
    skip_models = not _yes_no(
        "Run temporal models? (forecasting, burn-out)", default=True
    )
    skip_report = not _yes_no(
        "Generate reports? (Basic + Pro + Premium)", default=True
    )
    gen_pdf = _yes_no("Generate PDF report? (requires pandoc)", default=False)

    # Advanced options (collapsed by default)
    years = 2.0
    end_date = ""
    pit_mode = "report_date"
    output_dir = "cache"
    verbose = False

    if _yes_no("Show advanced options?", default=False):
        _years_str = _prompt("Lookback window in years", "2.0")
        try:
            years = float(_years_str)
        except ValueError:
            years = 2.0
        end_date = _prompt("End date for backtesting (YYYY-MM-DD, empty=today)", "")
        pit_mode = _prompt("PIT alignment mode (report_date or filing_date)", "report_date")
        if pit_mode not in ("report_date", "filing_date"):
            pit_mode = "report_date"
        output_dir = _prompt("Output directory", "cache")
        verbose = _yes_no("Verbose debug logging?", default=False)

    # ------------------------------------------------------------------
    # Step 9: Confirmation & Run
    # ------------------------------------------------------------------
    _step(9, "Confirmation")

    estimate = estimate_runtime(skip_linked, skip_models)

    print(f"  Market:           {_bold(market.country)} ({market.pit_api_name})")
    print(f"  Company:          {_bold(company)}")
    print(f"  Country:          {country}")
    print(f"  Data mode:        {'Enhanced (API keys)' if data_mode == 'api_and_wrappers' else 'Standard'}")
    if macro:
        print(f"  Macro source:     {macro.api_name}")
    print(f"  Linked entities:  {'Yes' if not skip_linked else 'Skip'}")
    print(f"  Temporal models:  {'Yes' if not skip_models else 'Skip'}")
    print(f"  Reports:          {'Yes' if not skip_report else 'Skip'}")
    print(f"  PDF output:       {'Yes' if gen_pdf else 'No'}")
    print(f"  Lookback:         {years} years")
    if end_date:
        print(f"  End date:         {end_date} (backtest mode)")
    print(f"  PIT mode:         {pit_mode}")
    print(f"  Output dir:       {output_dir}")
    llm_model = keys.get("_llm_model", "")
    _provider_labels = {"gemini": "Gemini", "claude": "Claude", "openrouter": "OpenRouter"}
    if llm_provider in _provider_labels:
        _llm_label = f"{_provider_labels[llm_provider]} / {llm_model or 'default'} (AI-generated)"
    else:
        _llm_label = "Template fallback (no LLM key)"
    print(f"  Report engine:    {_llm_label}")
    print(f"  Estimated time:   {_yellow(estimate)}")
    print("")

    if not _yes_no("Start the pipeline?"):
        _info("Cancelled.")
        return 0

    # ------------------------------------------------------------------
    # Run pipeline
    # ------------------------------------------------------------------
    print("")
    _step(10, "Running Pipeline")

    start_time = time.time()

    cmd = [
        sys.executable, "main.py",
        "--market", market_id,
        "--company", company,
        "--years", str(years),
        "--pit-mode", pit_mode,
        "--output-dir", output_dir,
    ]
    if skip_linked:
        cmd.append("--skip-linked")
    if skip_models:
        cmd.append("--skip-models")
    if skip_report:
        cmd.append("--skip-report")
    if gen_pdf:
        cmd.append("--pdf")
    if end_date:
        cmd.extend(["--end-date", end_date])
    if verbose:
        cmd.append("--verbose")
    if llm_provider:
        cmd.extend(["--llm-provider", llm_provider])
    llm_model = keys.get("_llm_model", "")
    if llm_model:
        cmd.extend(["--llm-model", llm_model])

    _info(f"Command: {' '.join(cmd)}")
    if llm_provider:
        _info(f"LLM: {llm_provider} / {llm_model or 'default'}")
    print("")
    _separator()
    print("")

    result = subprocess.run(cmd, check=False)

    elapsed = time.time() - start_time
    minutes = int(elapsed // 60)
    seconds = int(elapsed % 60)

    print("")
    _separator()
    print("")

    if result.returncode == 0:
        _ok(f"Pipeline completed successfully in {minutes}m {seconds}s")
        print("")
        _info("Output files:")

        output_dir = Path("cache")
        if output_dir.exists():
            for f in sorted(output_dir.rglob("*")):
                if f.is_file():
                    size = f.stat().st_size
                    size_str = f"{size / 1024:.1f} KB" if size > 1024 else f"{size} B"
                    print(f"    {f.relative_to('.')}  ({size_str})")

        print("")
        _info("Key files:")
        for key_file in [
            "cache/company_profile.json",
            "cache/report/premium_report.md",
            "cache/report/pro_report.md",
            "cache/report/basic_report.md",
        ]:
            p = Path(key_file)
            if p.exists():
                print(f"    {_green('[exists]')} {key_file}")
            else:
                print(f"    {_dim('[  --  ]')} {key_file}")
    else:
        _err(f"Pipeline failed (exit code {result.returncode}) after {minutes}m {seconds}s")
        _info("Check the log output above for details.")
        return 1

    print("")
    _bold("  Done! Review the report in cache/report/analysis_report.md")
    print("")

    return 0


if __name__ == "__main__":
    try:
        sys.exit(main())
    except KeyboardInterrupt:
        print("")
        print(_dim("  Interrupted."))
        sys.exit(130)
