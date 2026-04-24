#!/usr/bin/env python3
"""Operator 1 -- CLI entry point with global PIT API selection.

Usage (interactive):
    python main.py

Usage (non-interactive):
    python main.py --market us_sec_edgar --company AAPL
    python main.py --market jp_jquants --company 7203 --skip-models
    python main.py --market kr_dart --company 005930
    python main.py --report-only --output-dir cache
    python main.py --help

The user picks a region, then a market/exchange, then searches for a
company within that market.  All data comes from free, immutable
Point-in-Time (PIT) government filing APIs.  Gemini is used for
report generation only.

Supported PIT sources (Tier 1 -- $91T+ market cap coverage):
  - SEC EDGAR (US)          - Companies House (UK)
  - ESEF/XBRL (EU/FR/DE)   - J-Quants (Japan)
  - DART (South Korea)      - MOPS (Taiwan)
  - CVM (Brazil)            - CMF (Chile)

Requirements:
    pip install -r requirements.txt
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import sys
from datetime import date, timedelta
from pathlib import Path

import pandas as pd

# ---------------------------------------------------------------------------
# Early setup: configure logging before any operator1 imports
# ---------------------------------------------------------------------------

_LOG_DIR = Path("cache")
_LOG_FILE = _LOG_DIR / "pipeline_run.md"


def _setup_logging() -> None:
    """Configure logging to both console and a markdown log file.

    The markdown log is saved to ``cache/pipeline_run.md`` so that users
    can review the full pipeline output after the run completes.
    """
    root_logger = logging.getLogger()
    root_logger.setLevel(logging.INFO)

    # Console handler
    console = logging.StreamHandler()
    console.setLevel(logging.INFO)
    console.setFormatter(logging.Formatter(
        "%(asctime)s | %(levelname)-7s | %(name)s | %(message)s",
        datefmt="%H:%M:%S",
    ))
    root_logger.addHandler(console)

    # Markdown file handler -- writes a fenced code block for easy reading
    _LOG_DIR.mkdir(parents=True, exist_ok=True)
    file_handler = logging.FileHandler(_LOG_FILE, mode="w", encoding="utf-8")
    file_handler.setLevel(logging.DEBUG)
    file_handler.setFormatter(logging.Formatter(
        "%(asctime)s | %(levelname)-7s | %(name)s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    ))
    root_logger.addHandler(file_handler)

    # Write markdown header
    with open(_LOG_FILE, "w", encoding="utf-8") as f:
        f.write("# Operator 1 -- Pipeline Run Log\n\n")
        f.write(f"**Started:** {date.today().isoformat()}\n\n")
        f.write("```\n")


def _finalize_log() -> None:
    """Close the markdown fenced code block in the log file."""
    try:
        with open(_LOG_FILE, "a", encoding="utf-8") as f:
            f.write("```\n\n")
            f.write(f"**Log saved to:** `{_LOG_FILE}`\n")
    except Exception:
        pass


_setup_logging()
logger = logging.getLogger("operator1.main")


# ---------------------------------------------------------------------------
# Interactive selection helpers
# ---------------------------------------------------------------------------

def _select_region() -> str:
    """Display region menu and return the chosen region name."""
    from operator1.clients.pit_registry import get_regions, format_region_menu

    regions = get_regions()
    print(format_region_menu())

    while True:
        choice = input("Select a region (number or name): ").strip()
        if not choice:
            continue

        # Try numeric selection
        try:
            idx = int(choice) - 1
            if 0 <= idx < len(regions):
                return regions[idx]
        except ValueError:
            pass

        # Try name match (case-insensitive partial)
        for r in regions:
            if choice.lower() in r.lower():
                return r

        print(f"  Invalid choice: '{choice}'. Try again.")


def _select_market(region: str) -> str:
    """Display market menu for a region and return the chosen market_id."""
    from operator1.clients.pit_registry import (
        get_markets_by_region,
        format_market_menu,
    )

    markets = get_markets_by_region(region)
    if not markets:
        print(f"\nNo markets available for region: {region}")
        sys.exit(1)

    print(format_market_menu(region))

    while True:
        choice = input("Select a market (number or country name): ").strip()
        if not choice:
            continue

        # Numeric
        try:
            idx = int(choice) - 1
            if 0 <= idx < len(markets):
                return markets[idx].market_id
        except ValueError:
            pass

        # Name match
        for m in markets:
            if (choice.lower() in m.country.lower()
                    or choice.lower() in m.market_id.lower()):
                return m.market_id

        print(f"  Invalid choice: '{choice}'. Try again.")


def _select_company(pit_client) -> dict:
    """Search/browse companies within a PIT client and return selection."""
    print(f"\n--- Company Search ({pit_client.market_name}) ---")
    print("  Type a company name or ticker to search.")
    print("  Type 'list' to browse all companies.")
    print("  Type 'back' to go back.\n")

    while True:
        query = input("Search company: ").strip()
        if not query:
            continue
        if query.lower() == "back":
            return {}

        if query.lower() == "list":
            companies = pit_client.list_companies()
        else:
            companies = pit_client.search_company(query)

        if not companies:
            print(f"  No companies found for '{query}'. Try again.")
            continue

        # Show results (cap display at 25)
        display = companies[:25]
        print(f"\n  Found {len(companies)} companies"
              f"{' (showing first 25)' if len(companies) > 25 else ''}:\n")
        for idx, c in enumerate(display, 1):
            ticker = c.get("ticker", "")
            name = c.get("name", "Unknown")
            cik = c.get("cik", "")
            extra = f" (CIK: {cik})" if cik else ""
            print(f"    {idx:>3}. {ticker:<12} {name}{extra}")
        print()

        sel = input("Select a company (number, ticker, or 'search' again): ").strip()
        if not sel or sel.lower() == "search":
            continue

        # Numeric
        try:
            sel_idx = int(sel) - 1
            if 0 <= sel_idx < len(display):
                return display[sel_idx]
        except ValueError:
            pass

        # Match by ticker
        for c in companies:
            if sel.upper() == c.get("ticker", "").upper():
                return c

        print(f"  Invalid selection: '{sel}'. Try again.")


def _validate_api_data(
    market_info,
    macro_api_info,
    target_profile: dict,
    income_df: pd.DataFrame,
    balance_df: pd.DataFrame,
    cashflow_df: pd.DataFrame,
    quotes_df: pd.DataFrame,
    macro_data: dict,
    macro_dataset=None,
) -> None:
    """Log a diagnostic summary of what both micro and macro APIs returned.

    This helps confirm that region selection correctly triggers both the
    PIT (micro) client and the macro client, and that the data gathered
    is what the pipeline actually needs.
    """
    logger.info("")
    logger.info("--- API Data Validation Summary ---")

    # -- Micro (PIT) API --
    logger.info(
        "  PIT source : %s (%s, %s)",
        market_info.pit_api_name,
        market_info.country,
        market_info.exchange,
    )
    _micro_fields = {
        "profile": bool(target_profile),
        "income_statement": not income_df.empty,
        "balance_sheet": not balance_df.empty,
        "cashflow_statement": not cashflow_df.empty,
        "quotes_ohlcv": not quotes_df.empty,
    }
    for field_name, ok in _micro_fields.items():
        status = "OK" if ok else "MISSING"
        logger.info("    [%s] %s", status, field_name)

    # Key fields the pipeline needs from the profile
    _needed_profile_keys = [
        "name", "ticker", "country", "sector", "industry",
    ]
    _profile_present = [k for k in _needed_profile_keys if target_profile.get(k)]
    _profile_missing = [k for k in _needed_profile_keys if not target_profile.get(k)]
    if _profile_missing:
        logger.info(
            "    Profile fields present: %s | missing: %s",
            _profile_present, _profile_missing,
        )

    # -- Macro API --
    if macro_api_info:
        logger.info(
            "  Macro source: %s (%s)",
            macro_api_info.api_name,
            macro_api_info.country,
        )
        _expected = ["gdp", "inflation", "interest_rate", "unemployment", "currency"]
        for ind in _expected:
            series = macro_data.get(ind) if macro_data else None
            if series is not None and not series.empty:
                logger.info(
                    "    [OK]      macro.%s -- %d observations (%s to %s)",
                    ind, len(series),
                    series.index[0].date(), series.index[-1].date(),
                )
            else:
                logger.info("    [MISSING] macro.%s", ind)
    else:
        logger.info("  Macro source: none available for this market")

    # -- MacroDataset (downstream-ready) --
    if macro_dataset is not None:
        n_ind = len(macro_dataset.indicators)
        n_miss = len(macro_dataset.missing)
        logger.info(
            "  MacroDataset: %d indicators ready, %d missing %s",
            n_ind, n_miss,
            macro_dataset.missing if n_miss else "",
        )
    else:
        logger.info("  MacroDataset: not built (no macro data)")

    logger.info("--- End Validation Summary ---")
    logger.info("")


def _run_personal_data_checks(
    company: str,
    country: str,
    market_id: str,
    market_info,
    secrets: dict,
    interactive: bool = False,
) -> None:
    """Personal data guard removed -- API keys and emails are now prompted at startup."""
    pass


def _create_pit_client(market_id: str, secrets: dict):
    """Instantiate the PIT client for a given market_id.

    Delegates to the canonical factory in equity_provider.py to avoid
    duplicating client instantiation logic.
    """
    from operator1.clients.equity_provider import create_pit_client
    return create_pit_client(market_id, secrets)


# ---------------------------------------------------------------------------
# Main pipeline
# ---------------------------------------------------------------------------

def main() -> int:
    """Run the Operator 1 pipeline end-to-end."""

    parser = argparse.ArgumentParser(
        description="Operator 1 -- Point-in-Time Financial Analysis Pipeline",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""\
Interactive mode (no arguments):
  python main.py

Non-interactive examples:
  python main.py --market us_sec_edgar --company AAPL
  python main.py --market jp_jquants --company 7203
  python main.py --market kr_dart --company 005930
  python main.py --market eu_esef --company "Siemens"
  python main.py --list-markets
  python main.py --report-only --output-dir cache
""",
    )
    parser.add_argument(
        "--market", type=str, default="",
        help=(
            "Market ID to use (e.g. us_sec_edgar, jp_jquants, kr_dart). "
            "Use --list-markets to see all options."
        ),
    )
    parser.add_argument(
        "--company", type=str, default="",
        help=(
            "Company ticker or name to analyze. "
            "In interactive mode, you can search and browse."
        ),
    )
    parser.add_argument(
        "--list-markets", action="store_true",
        help="Display all supported PIT markets and exit",
    )
    parser.add_argument(
        "--list-regions", action="store_true",
        help="Display all supported regions and exit",
    )
    parser.add_argument(
        "--list-macro", action="store_true",
        help="Display all supported macro economic data APIs and exit",
    )
    parser.add_argument(
        "--years", type=float, default=2.0,
        help="Lookback window in years (default: 2.0)",
    )
    parser.add_argument(
        "--end-date", type=str, default="",
        help=(
            "Override end date for the analysis window (YYYY-MM-DD). "
            "Default: today. Use this for backtesting, e.g. --end-date 2024-12-31 "
            "to run on 2023-2024 data and validate predictions against 2025."
        ),
    )
    parser.add_argument(
        "--skip-linked", action="store_true",
        help="Skip linked entity discovery (faster, target-only analysis)",
    )
    parser.add_argument(
        "--skip-models", action="store_true",
        help="Skip temporal modeling / forecasting (cache + features only)",
    )
    parser.add_argument(
        "--skip-report", action="store_true",
        help="Skip report generation",
    )
    parser.add_argument(
        "--report-only", action="store_true",
        help="Only generate report from existing cache/profile",
    )
    parser.add_argument(
        "--output-dir", type=str, default="cache",
        help="Output directory for all artifacts (default: cache/)",
    )
    parser.add_argument(
        "--pdf", action="store_true",
        help="Generate PDF report (requires pandoc)",
    )
    parser.add_argument(
        "--llm-model", type=str, default="",
        help=(
            "LLM model name to use (e.g. gemini-2.0-flash, claude-sonnet-4-20250514). "
            "Overrides the default model for the selected provider. "
            "Use 'auto' to let the factory pick the best model."
        ),
    )
    parser.add_argument(
        "--llm-provider", type=str, default="",
        help=(
            "LLM provider to use: 'gemini' or 'claude'. "
            "If not set, auto-detected from available API keys."
        ),
    )
    parser.add_argument(
        "--pit-mode", type=str, default="report_date",
        choices=["report_date", "filing_date"],
        help=(
            "PIT alignment mode for financial statement merging. "
            "'report_date' (default): data appears on the fiscal period end date. "
            "'filing_date': strict PIT -- data appears only when publicly filed. "
            "Use 'filing_date' for backtesting to avoid look-ahead bias."
        ),
    )
    parser.add_argument(
        "--verbose", "-v", action="store_true",
        help="Enable debug logging",
    )
    parser.add_argument(
        "--stage", type=str, default="",
        help=(
            "Run specific pipeline stage(s) with checkpoint save/resume. "
            "Examples: '3' (temporal analysis), '4.1' (forecasting), "
            "'3-6' (stages 3 through 6), 'all' (all temporal stages 3-6). "
            "Requires --run-dir for state persistence. "
            "Stages: 3=temporal, 4=forecasting, 5=forward+MC, 6=ensemble."
        ),
    )
    parser.add_argument(
        "--run-dir", type=str, default="",
        help=(
            "Directory for staged pipeline state checkpoints. "
            "Used with --stage to save/resume between sub-stages. "
            "Auto-generated as cache/{company}_{end_date} if not set."
        ),
    )
    args = parser.parse_args()

    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)

    # ------------------------------------------------------------------
    # Parse --end-date for backtesting (override date.today())
    # ------------------------------------------------------------------
    _backtest_end_date = None
    if args.end_date:
        from datetime import datetime as _dt
        try:
            _backtest_end_date = _dt.strptime(args.end_date, "%Y-%m-%d").date()
            logger.info("Backtest mode: end_date=%s (predictions target future from this date)", _backtest_end_date)
        except ValueError:
            logger.error("Invalid --end-date format: %s (expected YYYY-MM-DD)", args.end_date)
            return 1

    # ------------------------------------------------------------------
    # Apply LLM CLI overrides to environment (before any LLM usage)
    # ------------------------------------------------------------------
    if args.llm_provider:
        os.environ["LLM_PROVIDER"] = args.llm_provider.strip().lower()
    if args.llm_model:
        os.environ["LLM_MODEL"] = args.llm_model.strip()

    # ------------------------------------------------------------------
    # Info-only commands
    # ------------------------------------------------------------------
    if args.list_markets:
        from operator1.clients.pit_registry import get_all_markets
        print("\n  OPERATOR 1 -- Supported PIT Markets\n")
        print(f"  {'ID':<22} {'Country':<20} {'Exchange':<30} {'API':<20} {'Cap':<8} {'Tier'}")
        print("  " + "-" * 108)
        for m in get_all_markets():
            print(
                f"  {m.market_id:<22} {m.country:<20} {m.exchange:<30} "
                f"{m.pit_api_name:<20} {m.market_cap:<8} {m.tier}"
            )
        print(f"\n  Total: {len(get_all_markets())} markets\n")
        return 0

    if args.list_regions:
        from operator1.clients.pit_registry import format_region_menu
        print(format_region_menu())
        return 0

    if args.list_macro:
        from operator1.clients.pit_registry import get_all_macro_apis
        print("\n  OPERATOR 1 -- Macro Economic Data APIs (Survival Mode)\n")
        print(f"  {'ID':<18} {'Country':<18} {'API':<40} {'Key?':<6} {'Indicators'}")
        print("  " + "-" * 110)
        for m in get_all_macro_apis():
            indicators = []
            if m.series_gdp: indicators.append("GDP")
            if m.series_inflation: indicators.append("CPI")
            if m.series_interest_rate: indicators.append("Rate")
            if m.series_unemployment: indicators.append("Jobs")
            if m.series_currency: indicators.append("FX")
            key_str = "Yes" if m.requires_api_key else "No"
            print(
                f"  {m.macro_id:<18} {m.country:<18} {m.api_name:<40} "
                f"{key_str:<6} {', '.join(indicators)}"
            )
        print(f"\n  Total: {len(get_all_macro_apis())} macro sources\n")
        return 0

    # ------------------------------------------------------------------
    # Step 0: Load secrets (only GEMINI_API_KEY is required now)
    # ------------------------------------------------------------------
    logger.info("=" * 60)
    logger.info("OPERATOR 1 -- Point-in-Time Financial Analysis")
    logger.info("=" * 60)

    try:
        from operator1.secrets_loader import load_secrets, validate_secrets
        secrets = load_secrets()
        # Pass market_id so only region-relevant keys are enforced
        validate_secrets(secrets, market_id=args.market)
    except SystemExit as exc:
        logger.error("Failed to load API keys: %s", exc)
        logger.error("Create a .env file from .env.example with ALL keys filled in")
        return 1

    # ------------------------------------------------------------------
    # Report-only mode
    # ------------------------------------------------------------------
    if args.report_only:
        return _generate_report_only(args, secrets)

    # ------------------------------------------------------------------
    # Step 1: Select region -> market -> company
    # ------------------------------------------------------------------
    logger.info("")
    logger.info("Step 1: Selecting data source and company...")

    # Determine market (interactive or CLI)
    market_id = args.market
    if not market_id:
        # Interactive region selection
        region = _select_region()
        market_id = _select_market(region)

    # Validate market exists
    from operator1.clients.pit_registry import get_market
    market_info = get_market(market_id)
    if market_info is None:
        logger.error("Unknown market ID: %s (use --list-markets to see options)", market_id)
        return 1

    logger.info(
        "Market: %s (%s) -- %s",
        market_info.country,
        market_info.exchange,
        market_info.pit_api_name,
    )

    # ------------------------------------------------------------------
    # Personal data guard: check user input and wrapper requirements
    # ------------------------------------------------------------------
    _run_personal_data_checks(
        company=args.company,
        country=getattr(market_info, "country", ""),
        market_id=market_id,
        market_info=market_info,
        secrets=secrets,
        interactive=not args.company,  # interactive if no --company CLI arg
    )

    # Create the PIT client for this market
    try:
        pit_client = _create_pit_client(market_id, secrets)
    except Exception as exc:
        logger.error("Failed to create PIT client for %s: %s", market_id, exc)
        return 1

    # Select company (interactive or CLI)
    company_info: dict = {}
    company_id = args.company

    if company_id:
        # Non-interactive: search for the company by ticker/name
        results = pit_client.search_company(company_id)
        if not results:
            # Try list_companies with query
            results = pit_client.list_companies(query=company_id)
        if results:
            company_info = results[0]
            logger.info(
                "Company found: %s (%s)",
                company_info.get("name", "Unknown"),
                company_info.get("ticker", company_id),
            )
        else:
            logger.warning(
                "Company '%s' not found via search, using as raw identifier",
                company_id,
            )
            company_info = {"ticker": company_id, "name": company_id}
    else:
        # Interactive company selection
        company_info = _select_company(pit_client)
        if not company_info:
            logger.info("No company selected. Exiting.")
            return 0

    ticker = company_info.get("ticker", "") or company_info.get("identifier", "")
    company_name = company_info.get("name", ticker)
    identifier = company_info.get("cik") or ticker or company_info.get("identifier", "")

    logger.info("Target: %s (%s) via %s", company_name, ticker, market_info.pit_api_name)

    # Identify the macro API for this market's region
    from operator1.clients.pit_registry import get_macro_api_for_market
    macro_api_info = get_macro_api_for_market(market_id)
    if macro_api_info:
        logger.info(
            "Macro source: %s (%s)",
            macro_api_info.api_name,
            macro_api_info.country,
        )
    else:
        logger.info("No macro API available for this market")

    # ------------------------------------------------------------------
    # Step 2: Fetch company profile from PIT API
    # ------------------------------------------------------------------
    logger.info("")
    logger.info("Step 2: Fetching company profile from %s...", market_info.pit_api_name)

    try:
        target_profile = pit_client.get_profile(identifier)
        target_profile.setdefault("name", company_name)
        target_profile.setdefault("ticker", ticker)
        target_profile.setdefault("country", market_info.country_code)
        target_profile.setdefault("market_id", market_id)
        target_profile.setdefault("pit_api", market_info.pit_api_name)
        logger.info(
            "Profile loaded: %s (%s), sector=%s",
            target_profile.get("name"),
            target_profile.get("ticker"),
            target_profile.get("sector", "N/A"),
        )
    except Exception as exc:
        logger.warning("Profile fetch failed (continuing with basic info): %s", exc)

    # W7 fix: Enrich profile for non-US markets via OpenFIGI/regional APIs.
    # This fills sector, industry, and identifier gaps without overwriting
    # existing data.
    try:
        from operator1.clients.supplement import enrich_profile
        target_profile = enrich_profile(
            market_id=market_id,
            ticker=ticker,
            existing_profile=target_profile,
        )
    except Exception as exc:
        logger.debug("Supplement enrichment skipped: %s", exc)
        target_profile = {
            "name": company_name,
            "ticker": ticker,
            "country": market_info.country_code,
            "market_id": market_id,
            "pit_api": market_info.pit_api_name,
            "sector": company_info.get("sector", ""),
            "industry": company_info.get("industry", ""),
        }

    # Step 2b: Fetch institutional/major holders (US/UK/KR only)
    target_holders: list[dict] = []
    try:
        if hasattr(pit_client, "get_holders"):
            target_holders = pit_client.get_holders(identifier)
            if target_holders:
                logger.info("Holders loaded: %d for %s", len(target_holders), ticker)
    except Exception as exc:
        logger.debug("Holder fetch skipped: %s", exc)

    # Step 2b.1: Fetch insider transactions (US only via yfinance)
    target_insiders: list[dict] = []
    try:
        if hasattr(pit_client, "get_insider_transactions"):
            target_insiders = pit_client.get_insider_transactions(identifier)
            if target_insiders:
                logger.info("Insider transactions loaded: %d for %s", len(target_insiders), ticker)
    except Exception as exc:
        logger.debug("Insider transaction fetch skipped: %s", exc)

    # ------------------------------------------------------------------
    # Step 3: Fetch PIT financial data
    # ------------------------------------------------------------------
    logger.info("")
    logger.info("Step 3: Fetching point-in-time financial data...")

    income_df = pd.DataFrame()
    balance_df = pd.DataFrame()
    cashflow_df = pd.DataFrame()
    quotes_df = pd.DataFrame()

    # Fetch all 4 data types in parallel (each hits different endpoints).
    from concurrent.futures import ThreadPoolExecutor, as_completed

    _fetch_tasks = {
        "income": pit_client.get_income_statement,
        "balance": pit_client.get_balance_sheet,
        "cashflow": pit_client.get_cashflow_statement,
        "quotes": pit_client.get_quotes,
    }

    with ThreadPoolExecutor(max_workers=4) as executor:
        futures = {
            executor.submit(fn, identifier): label
            for label, fn in _fetch_tasks.items()
        }
        for future in as_completed(futures):
            label = futures[future]
            try:
                result = future.result()
                if label == "income":
                    income_df = result
                elif label == "balance":
                    balance_df = result
                elif label == "cashflow":
                    cashflow_df = result
                elif label == "quotes":
                    quotes_df = result
                logger.info("%s: %d rows", label.capitalize(), len(result))
            except Exception as exc:
                logger.warning("%s fetch failed: %s", label.capitalize(), exc)

    # PIT filing APIs are regulatory filing systems and typically do not
    # carry market price data.  When OHLCV is empty, fetch from a free-tier
    # OHLCV source (Alpha Vantage or exchange-specific APIs).
    # Raw exchange OHLCV is inherently PIT: a trade at a given price on
    # a given date is an immutable fact that never changes retroactively.
    _ohlcv_source_label = market_info.pit_api_name  # default: same as PIT source
    if quotes_df.empty and ticker:
        logger.info(
            "PIT source %s does not provide OHLCV -- fetching from OHLCV provider.",
            market_info.pit_api_name,
        )
        try:
            from operator1.clients.ohlcv_provider import fetch_ohlcv
            quotes_df = fetch_ohlcv(ticker, market_id=args.market)
            if not quotes_df.empty:
                _ohlcv_source_label = "yfinance (Yahoo Finance)"
                logger.info("OHLCV fetched from yfinance provider: %d rows", len(quotes_df))
        except Exception as exc:
            logger.warning("OHLCV provider failed: %s", exc)

    # Step 3b: Reconcile financial data (normalize fields, validate dates)
    reconciliation_report = {}
    try:
        from operator1.quality.data_reconciliation import reconcile_financial_data
        income_df, balance_df, cashflow_df, reconciliation_report = (
            reconcile_financial_data(income_df, balance_df, cashflow_df)
        )
        if reconciliation_report.get("issues"):
            logger.warning(
                "Data reconciliation: %d issues found",
                len(reconciliation_report["issues"]),
            )
        else:
            logger.info("Data reconciliation: clean")
    except Exception as exc:
        logger.warning("Data reconciliation failed (continuing): %s", exc)

    # Step 3c: Pivot canonical long-format DataFrames to wide format.
    # The canonical translator outputs long format (one row per concept
    # per filing: canonical_name, value, filing_date, report_date).
    # Downstream modules (derived_variables, financial_health, estimator)
    # expect wide format (one row per date, columns = revenue, total_assets, etc.)
    try:
        from operator1.clients.canonical_translator import pivot_to_canonical_wide

        for label, stmt_df in [("income", income_df), ("balance", balance_df), ("cashflow", cashflow_df)]:
            if stmt_df.empty:
                continue
            logger.debug(
                "Pivot check for %s: columns=%s, has_canonical=%s, has_value=%s",
                label, list(stmt_df.columns)[:5],
                "canonical_name" in stmt_df.columns,
                "value" in stmt_df.columns,
            )
            if "canonical_name" in stmt_df.columns and "value" in stmt_df.columns:
                wide = pivot_to_canonical_wide(stmt_df, date_col="report_date")
                if not wide.empty:
                    # Preserve filing_date for PIT alignment by adding it
                    # from the latest filing per report_date
                    if "filing_date" in stmt_df.columns:
                        filing_dates = (
                            stmt_df.dropna(subset=["filing_date", "report_date"])
                            .sort_values("filing_date")
                            .drop_duplicates(subset=["report_date"], keep="last")
                            [["report_date", "filing_date"]]
                        )
                        wide = wide.merge(filing_dates, on="report_date", how="left")
                    if label == "income":
                        income_df = wide
                    elif label == "balance":
                        balance_df = wide
                    else:
                        cashflow_df = wide
                    logger.info(
                        "Pivoted %s to wide: %d periods x %d columns",
                        label, len(wide), len(wide.columns),
                    )
    except Exception as exc:
        logger.warning("Long-to-wide pivot failed (continuing with raw format): %s", exc)

    if quotes_df.empty and income_df.empty and balance_df.empty:
        logger.error(
            "No data retrieved for %s from %s. "
            "Check the identifier and try again.",
            identifier,
            market_info.pit_api_name,
        )
        return 1

    # ------------------------------------------------------------------
    # Step 4: Build unified daily cache
    # ------------------------------------------------------------------
    logger.info("")
    logger.info("Step 4: Building daily cache from PIT data...")

    # Build a time-indexed cache from OHLCV as the spine
    if not quotes_df.empty:
        if "date" in quotes_df.columns:
            quotes_df["date"] = pd.to_datetime(quotes_df["date"])
            cache = quotes_df.set_index("date").sort_index()
        elif quotes_df.index.name == "date" or hasattr(quotes_df.index, "date"):
            cache = quotes_df.sort_index()
        else:
            cache = quotes_df.copy()
    else:
        # Fallback: create empty cache with date range
        end = _backtest_end_date if _backtest_end_date else date.today()
        start = end - timedelta(days=int(args.years * 365))
        idx = pd.date_range(start, end, freq="B", name="date")
        cache = pd.DataFrame(index=idx)

    # Merge financial statement data using frequency-aware interpolation.
    # Periodic filings (quarterly/semi-annual/annual) are interpolated
    # to daily frequency respecting the nature of each variable:
    #   - Stock variables (balance sheet): linear interpolation
    #   - Flow variables (income/cashflow): distribute period totals
    # Columns are merged WITHOUT prefixes so that derived_variables.py
    # finds the expected canonical names (revenue, total_assets, etc.).
    # If the same column name exists in multiple statements, the first
    # non-null value wins (income > balance > cashflow priority).
    _merged_cols: set = set(cache.columns)
    _interp_confidence: dict[str, pd.Series] = {}

    # Import the frequency-aware interpolator
    try:
        from operator1.estimation.frequency_interpolator import (
            interpolate_statement_to_daily,
        )
        _use_interpolator = True
    except ImportError:
        logger.warning("Frequency interpolator not available -- falling back to flat ffill")
        _use_interpolator = False

    for label, stmt_df in [
        ("income", income_df),
        ("balance", balance_df),
        ("cashflow", cashflow_df),
    ]:
        if stmt_df.empty:
            continue
        try:
            # PIT alignment mode:
            # - report_date (default): data appears on fiscal period end.
            # - filing_date (strict PIT): data appears only when publicly filed.
            #   Use --pit-mode filing_date for backtesting to avoid look-ahead bias.
            _preferred_date_col = getattr(args, "pit_mode", "report_date")
            if _preferred_date_col in stmt_df.columns:
                date_col = _preferred_date_col
            elif "report_date" in stmt_df.columns:
                date_col = "report_date"
            elif "filing_date" in stmt_df.columns:
                date_col = "filing_date"
            else:
                date_col = ""
            if date_col not in stmt_df.columns:
                logger.warning("No date column in %s data, skipping merge", label)
                continue

            stmt_df[date_col] = pd.to_datetime(stmt_df[date_col])
            stmt_df = stmt_df.sort_values(date_col)
            # Deduplicate: keep last row per date (most recent filing)
            stmt_df = stmt_df.drop_duplicates(subset=[date_col], keep="last")

            # Extract numeric columns for merge
            numeric_cols = stmt_df.select_dtypes(include=["number"]).columns.tolist()
            # Exclude date-like columns from numeric merge
            numeric_cols = [c for c in numeric_cols if c != date_col and "date" not in c.lower()]
            if not numeric_cols:
                continue

            stmt_indexed = stmt_df.set_index(date_col)[numeric_cols]

            if _use_interpolator and len(stmt_indexed) >= 2:
                # Frequency-aware interpolation: stock variables get linear
                # interpolation, flow variables get period distribution.
                stmt_aligned, conf_df = interpolate_statement_to_daily(
                    stmt_indexed,
                    daily_index=cache.index,
                    market_id=market_id,
                )
                # Store confidence scores for later use
                for col in conf_df.columns:
                    _interp_confidence[col] = conf_df[col]
            else:
                # Fallback for single-filing or missing interpolator:
                # flat forward-fill (original behavior).
                combined_idx = cache.index.union(stmt_indexed.index).sort_values()
                stmt_aligned = stmt_indexed.reindex(combined_idx).ffill()
                stmt_aligned = stmt_aligned.reindex(cache.index)

            # Skip columns already in cache (first statement wins)
            new_cols = [c for c in stmt_aligned.columns if c not in _merged_cols]
            if new_cols:
                cache = cache.join(stmt_aligned[new_cols], how="left")
                _merged_cols.update(new_cols)

            logger.info("Merged %s data: %d columns (%d new)", label, len(numeric_cols), len(new_cols))
        except Exception as exc:
            logger.warning("Failed to merge %s data: %s", label, exc)

    # Store interpolation confidence in the cache for downstream models
    for col, conf_series in _interp_confidence.items():
        conf_col = f"interp_confidence_{col}"
        if conf_col not in cache.columns:
            cache[conf_col] = conf_series

    logger.info("Cache built: %d rows x %d columns", len(cache), len(cache.columns))

    # ------------------------------------------------------------------
    # Backtest date filter: trim cache to end at --end-date
    # ------------------------------------------------------------------
    if _backtest_end_date is not None:
        _bt_end_ts = pd.Timestamp(_backtest_end_date)
        _bt_start_ts = _bt_end_ts - pd.Timedelta(days=int(args.years * 365))
        _pre_len = len(cache)
        cache = cache[(cache.index >= _bt_start_ts) & (cache.index <= _bt_end_ts)]
        logger.info(
            "Backtest filter: %d -> %d rows (window %s to %s)",
            _pre_len, len(cache),
            cache.index[0].date() if len(cache) > 0 else "N/A",
            cache.index[-1].date() if len(cache) > 0 else "N/A",
        )

    # ------------------------------------------------------------------
    # Step 4.bench: Fetch benchmark index returns for beta computation
    # ------------------------------------------------------------------
    try:
        from operator1.clients.ohlcv_provider import fetch_benchmark_returns
        _bench_returns = fetch_benchmark_returns(market_id, years=int(getattr(args, "years", 2)))
        if not _bench_returns.empty:
            _bench_aligned = _bench_returns.reindex(cache.index, method="ffill")
            cache["benchmark_return_1d"] = _bench_aligned
            logger.info(
                "Benchmark returns merged: %d non-null days for %s",
                int(_bench_aligned.notna().sum()), market_id,
            )
    except Exception as exc:
        logger.debug("Benchmark fetch skipped: %s", exc)

    # ------------------------------------------------------------------
    # Step 4.bench.1: Fetch options-implied volatility (D3)
    # IV-RV spread is the best single predictor of vol regime changes.
    # ------------------------------------------------------------------
    try:
        from operator1.clients.ohlcv_provider import fetch_implied_volatility
        _iv_series = fetch_implied_volatility(ticker)
        if not _iv_series.empty:
            _iv_val = float(_iv_series.iloc[0])
            cache["iv30"] = _iv_val  # constant (current snapshot)
            # IV-RV spread: implied minus realized vol
            if "volatility_21d" in cache.columns:
                _rv = cache["volatility_21d"].iloc[-1] if cache["volatility_21d"].notna().any() else 0.0
                cache["iv_rv_spread"] = _iv_val - float(_rv)
                logger.info("IV30=%.4f, RV=%.4f, IV-RV spread=%.4f", _iv_val, float(_rv), _iv_val - float(_rv))
    except Exception as exc:
        logger.debug("Implied volatility fetch skipped: %s", exc)

    # ------------------------------------------------------------------
    # Step 4.bench.2: Fetch sector leading indicators (D4)
    # Cross-asset ETFs that historically lead the target's sector.
    # ------------------------------------------------------------------
    try:
        from operator1.clients.ohlcv_provider import fetch_sector_leading_indicators
        _sector = target_profile.get("sector", "")
        _leader_df = fetch_sector_leading_indicators(_sector, years=int(getattr(args, "years", 2)))
        if not _leader_df.empty:
            # Merge leading indicator returns into cache
            _leader_aligned = _leader_df.reindex(cache.index, method="ffill")
            for _ldr_col in _leader_aligned.columns:
                _col_name = f"sector_leader_{_ldr_col}"
                if _col_name not in cache.columns:
                    cache[_col_name] = _leader_aligned[_ldr_col]
            logger.info("Sector leading indicators merged: %d ETFs for sector '%s'", len(_leader_df.columns), _sector)
    except Exception as exc:
        logger.debug("Sector leading indicators skipped: %s", exc)

    # ------------------------------------------------------------------
    # Step 4a.8: Cross-asset sector rotation signals (Gap 3)
    # Tracks 11 sector ETFs + Treasury yields + USD + gold to detect
    # institutional capital rotation before it hits individual stocks.
    # ------------------------------------------------------------------
    cross_asset_result = None
    try:
        from operator1.features.cross_asset_signals import compute_cross_asset_signals
        cache, cross_asset_result = compute_cross_asset_signals(
            cache, sector=target_profile.get("sector", ""),
        )
        if cross_asset_result and cross_asset_result.available:
            logger.info(
                "Cross-asset signals: RS=%s, rank=%s, disp=%s, YC=%s",
                f"{cross_asset_result.sector_relative_strength:.3f}"
                if cross_asset_result.sector_relative_strength is not None else "N/A",
                cross_asset_result.sector_rank_12m or "N/A",
                f"{cross_asset_result.sector_dispersion:.5f}"
                if cross_asset_result.sector_dispersion is not None else "N/A",
                f"{cross_asset_result.yield_curve_10y2y:.3f}"
                if cross_asset_result.yield_curve_10y2y is not None else "N/A",
            )
    except Exception as exc:
        logger.debug("Cross-asset signals skipped: %s", exc)
    # Step 4a.7: Options-derived forward-looking signals (Gap 1)
    # Fetches full options surface and computes 6 features: put/call ratio,
    # 25-delta risk reversal, IV skew, VIX term structure, SKEW index,
    # and variance risk premium. These are leading indicators that move
    # before price (institutional positioning visible in options first).
    # ------------------------------------------------------------------
    options_signal_result = None
    try:
        from operator1.features.options_signals import compute_options_signals
        cache, options_signal_result = compute_options_signals(
            cache, ticker=ticker, market_id=market_id,
        )
        if options_signal_result and options_signal_result.available:
            logger.info(
                "Options signals: PCR=%.2f, RR25d=%s, VTS=%s",
                options_signal_result.put_call_ratio or 0,
                f"{options_signal_result.risk_reversal_25d:.4f}"
                if options_signal_result.risk_reversal_25d is not None else "N/A",
                f"{options_signal_result.vix_term_structure:.3f}"
                if options_signal_result.vix_term_structure is not None else "N/A",
            )
    except Exception as exc:
        logger.debug("Options signals skipped: %s", exc)

    # ------------------------------------------------------------------
    # Step 4a: Fetch macro data for survival mode analysis
    # ------------------------------------------------------------------
    macro_data = {}
    macro_dataset = None
    macro_quadrant_result = None

    if macro_api_info:
        logger.info("")
        logger.info("Step 4a: Fetching macro data from %s...", macro_api_info.api_name)
        try:
            from operator1.clients.macro_provider import fetch_macro
            macro_data = fetch_macro(
                market_info.country_code,
                secrets=secrets,
                years=int(getattr(args, "years", 2)),
            )
            if macro_data:
                logger.info("Macro data: %d indicators fetched", len(macro_data))
                for name, series in macro_data.items():
                    logger.info("    [OK] %s: %d observations", name, len(series))
            else:
                logger.warning("Macro data: no indicators returned (APIs may need keys)")
        except Exception as exc:
            logger.warning("Macro data fetch failed (continuing without macro): %s", exc)

    # Build MacroDataset from raw macro dict (structured container for downstream)
    if macro_data:
        try:
            from operator1.steps.macro_mapping import fetch_macro_data as _build_macro_ds
            macro_dataset = _build_macro_ds(
                country_iso2=market_info.country_code,
                macro_raw=macro_data,
            )
            logger.info(
                "MacroDataset built: %d indicators, %d missing",
                len(macro_dataset.indicators),
                len(macro_dataset.missing),
            )
        except Exception as exc:
            logger.warning("MacroDataset construction failed: %s", exc)

    # Compute macro quadrant classification
    if macro_data:
        try:
            from operator1.features.macro_quadrant import compute_macro_quadrant
            cache, macro_quadrant_result = compute_macro_quadrant(
                cache,
                macro_data=macro_dataset,
            )
            logger.info(
                "Macro quadrant: %s, stability=%.3f",
                getattr(macro_quadrant_result, "latest_quadrant", "N/A"),
                getattr(macro_quadrant_result, "stability_score", 0.0),
            )
        except Exception as exc:
            logger.warning("Macro quadrant classification failed: %s", exc)

    # ------------------------------------------------------------------
    # Step 4a-validate: Log what both APIs returned for diagnostics
    # ------------------------------------------------------------------
    _validate_api_data(
        market_info=market_info,
        macro_api_info=macro_api_info,
        target_profile=target_profile,
        income_df=income_df,
        balance_df=balance_df,
        cashflow_df=cashflow_df,
        quotes_df=quotes_df,
        macro_data=macro_data,
        macro_dataset=macro_dataset,
    )

    # ------------------------------------------------------------------
    # Step 4d: Institutional ownership history -> cache merge
    # ------------------------------------------------------------------
    try:
        if hasattr(pit_client, "get_holder_history"):
            _holder_hist_df = pit_client.get_holder_history(identifier, years=int(getattr(args, "years", 2)))
            if _holder_hist_df is not None and not _holder_hist_df.empty and "date_reported" in _holder_hist_df.columns:
                _holder_hist_df["date_reported"] = pd.to_datetime(_holder_hist_df["date_reported"])
                _holder_hist_df = _holder_hist_df.sort_values("date_reported")
                _inst_cols = [c for c in _holder_hist_df.columns if c.startswith("inst_")]
                if _inst_cols:
                    _inst_indexed = _holder_hist_df.set_index("date_reported")[_inst_cols]
                    if _use_interpolator and len(_inst_indexed) >= 2:
                        _inst_aligned, _ = interpolate_statement_to_daily(
                            _inst_indexed, daily_index=cache.index, market_id=market_id,
                        )
                    else:
                        # Single snapshot or no interpolator: flat forward-fill
                        _combined = cache.index.union(_inst_indexed.index).sort_values()
                        _inst_aligned = _inst_indexed.reindex(_combined).ffill()
                        _inst_aligned = _inst_aligned.reindex(cache.index)
                    _new_inst = [c for c in _inst_aligned.columns if c not in cache.columns]
                    if _new_inst:
                        cache = cache.join(_inst_aligned[_new_inst], how="left")
                    logger.info(
                        "Institutional ownership merged: %d columns from %d snapshots",
                        len(_new_inst), len(_holder_hist_df),
                    )
    except Exception as exc:
        logger.debug("Institutional ownership history skipped: %s", exc)

    # ------------------------------------------------------------------
    # Step 4a.3: Conflict risk assessment
    # Must run BEFORE survival mode (Step 5) because country_conflict_flag
    # and sanctions_flag are survival triggers.
    # ------------------------------------------------------------------
    conflict_result = None
    linked_conflict = None
    try:
        from operator1.features.conflict_risk import (
            assess_conflict_risk,
            inject_conflict_risk_into_cache,
        )

        conflict_result = assess_conflict_risk(
            country_iso2=market_info.country_code,
            company_name=company_name,
        )
        cache = inject_conflict_risk_into_cache(cache, conflict_result)
        logger.info(
            "Conflict risk: flag=%s, intensity=%.3f, type=%s, sources=%s",
            conflict_result.country_conflict_flag,
            conflict_result.conflict_intensity_score,
            conflict_result.conflict_type,
            conflict_result.data_sources_used,
        )
    except Exception as exc:
        logger.warning("Conflict risk assessment failed: %s", exc)

    # ------------------------------------------------------------------
    # Step 4a.4: Market buying power (demand-side signal)
    # ------------------------------------------------------------------
    buying_power_result = None
    try:
        from operator1.features.market_buying_power import compute_market_buying_power
        cache, buying_power_result = compute_market_buying_power(
            cache,
            sector=target_profile.get("sector"),
            country_iso2=market_info.country_code,
            macro_data=macro_data,
        )
        if buying_power_result and buying_power_result.available:
            logger.info(
                "Market buying power: index=%.0f, momentum=%+.3f, demand_risk=%s",
                buying_power_result.buying_power_index,
                buying_power_result.sector_demand_momentum,
                buying_power_result.demand_risk_flag,
            )
    except Exception as exc:
        logger.warning("Market buying power failed: %s", exc)

    # ------------------------------------------------------------------
    # Step 4a.5: SIX proxy computation + canonical column seeding
    # Must run BEFORE estimation so proxy values flow into the estimator.
    # ------------------------------------------------------------------
    six_proxy_result = None
    if market_id == "ch_six":
        try:
            from operator1.features.six_derived_proxies import (
                compute_six_proxies, seed_canonical_columns,
            )
            six_proxy_result = compute_six_proxies(cache, target_profile)
            if six_proxy_result.computed:
                logger.info(
                    "SIX proxies: %d columns, yield=%.2f%%, implied_pe=%.1f",
                    six_proxy_result.n_proxies,
                    (six_proxy_result.dividend_yield or 0) * 100,
                    six_proxy_result.implied_pe or 0,
                )
                # Seed canonical columns so the estimator can cascade-fill
                seed_canonical_columns(cache, target_profile, six_proxy_result)
            elif six_proxy_result.error:
                logger.warning("SIX proxy computation failed: %s", six_proxy_result.error)
        except Exception as exc:
            logger.warning("SIX proxy module failed: %s", exc)

    # ------------------------------------------------------------------
    # Step 4a.6: Lightweight pre-estimation ratio computation
    # Compute basic financial ratios from raw statement data BEFORE
    # estimation, so the estimator can use them as features and the FH
    # calibration (interest_coverage > 10, cash > debt) sees non-NaN values.
    # ------------------------------------------------------------------
    try:
        from operator1.constants import EPSILON as _EPS
        _pre_ratios = {
            "current_ratio": ("current_assets", "current_liabilities"),
            "interest_coverage": ("ebit", "interest_expense"),
            "cash_ratio": ("cash_and_equivalents", "current_liabilities"),
            "gross_margin": ("gross_profit", "revenue"),
            "net_margin": ("net_income", "revenue"),
            "operating_margin": ("operating_income", "revenue"),
            "debt_to_equity_abs": ("total_debt", "total_equity"),
        }
        _n_pre = 0
        for ratio_name, (num_col, den_col) in _pre_ratios.items():
            if (ratio_name not in cache.columns
                    and num_col in cache.columns
                    and den_col in cache.columns):
                _num = cache[num_col].astype(float)
                _den = cache[den_col].astype(float)
                _safe_den = _den.where(_den.abs() > _EPS)
                cache[ratio_name] = _num / _safe_den
                _n_pre += 1
        if _n_pre > 0:
            logger.info("Pre-estimation ratios computed: %d ratios", _n_pre)
    except Exception as exc:
        logger.debug("Pre-estimation ratio computation skipped: %s", exc)

    # ------------------------------------------------------------------
    # Step 4b: Estimation -- fill missing financials
    # ------------------------------------------------------------------
    estimation_coverage = None
    try:
        logger.info("")
        logger.info("Step 4b: Running estimation (Sudoku inference)...")

        from operator1.estimation.estimator import run_estimation
        from operator1.config_loader import load_config

        _global_cfg = load_config("global_config")
        _imputer_method = _global_cfg.get("estimation_imputer", "bayesian_ridge")

        cache, estimation_coverage = run_estimation(
            cache,
            imputer_method=_imputer_method,
        )
        logger.info(
            "Estimation complete: method=%s, variables=%d",
            _imputer_method,
            len(estimation_coverage.coverage_before) if estimation_coverage else 0,
        )

        # Persist coverage report
        if estimation_coverage is not None:
            coverage_path = Path(args.output_dir) / "estimation_coverage.json"
            coverage_path.parent.mkdir(parents=True, exist_ok=True)
            import json as _json
            with open(coverage_path, "w", encoding="utf-8") as _f:
                _json.dump({
                    "coverage_before": estimation_coverage.coverage_before,
                    "coverage_after": estimation_coverage.coverage_after,
                }, _f, indent=2, default=str)
            logger.info("Estimation coverage saved: %s", coverage_path)
    except Exception as exc:
        logger.warning("Estimation failed (continuing with raw data): %s", exc)

    # ------------------------------------------------------------------
    # Step 4b.1: Unified confidence metric (precision-weighted pooling)
    # Combines interpolation confidence (distance from filing) with
    # estimation confidence (model agreement) into a single metric.
    # ------------------------------------------------------------------
    try:
        from operator1.estimation.estimator import ESTIMABLE_VARIABLES as _EST_VARS
        _n_combined = 0
        for col in _EST_VARS:
            interp_col = f"interp_confidence_{col}"
            est_col = f"{col}_confidence"
            combined_col = f"{col}_combined_confidence"
            if interp_col in cache.columns and est_col in cache.columns:
                ic = cache[interp_col].fillna(0.5)
                ec = cache[est_col].fillna(0.5)
                # Precision-weighted pooling: 1 / (1/a + 1/b)
                # Clamp to avoid division by zero
                ic_safe = ic.clip(lower=0.01)
                ec_safe = ec.clip(lower=0.01)
                combined = 1.0 / ((1.0 / ic_safe) + (1.0 / ec_safe))
                # Normalize to [0, 1]
                max_val = combined.max()
                if max_val > 0:
                    combined = combined / max_val
                cache[combined_col] = combined
                _n_combined += 1
        if _n_combined > 0:
            logger.info("Unified confidence: %d variables with combined confidence metric", _n_combined)
    except Exception as exc:
        logger.debug("Unified confidence computation skipped: %s", exc)

    # ------------------------------------------------------------------
    # Step 4c: Filing calendar analysis
    # ------------------------------------------------------------------
    filing_calendar_result = None
    try:
        from operator1.features.filing_calendar import analyze_filing_calendar
        filing_calendar_result = analyze_filing_calendar(cache, market_id=market_id)
        logger.info(
            "Filing calendar: expected=%d, actual=%d (%.0f%%), freq=%s, stale=%s (age=%dd)",
            filing_calendar_result.expected_filings_2yr,
            filing_calendar_result.actual_filings_2yr,
            filing_calendar_result.coverage_ratio * 100,
            filing_calendar_result.detected_frequency,
            filing_calendar_result.is_stale,
            filing_calendar_result.latest_filing_age_days,
        )
        if filing_calendar_result.is_stale:
            logger.warning(
                "STALE DATA: Latest filing is %d days old (threshold: %d days for %s)",
                filing_calendar_result.latest_filing_age_days,
                filing_calendar_result.stale_threshold_days,
                market_id,
            )
        if filing_calendar_result.gaps:
            logger.warning(
                "Filing gaps detected: %d gaps in 2-year window",
                len(filing_calendar_result.gaps),
            )
    except Exception as exc:
        logger.warning("Filing calendar analysis failed: %s", exc)

    # Step 4c.1: Inject filing freshness into cache (if calendar succeeded)
    if filing_calendar_result is not None:
        try:
            from operator1.features.filing_calendar import inject_filing_freshness
            cache = inject_filing_freshness(cache, filing_calendar_result, market_id=market_id)
        except Exception as exc:
            logger.warning("Filing freshness injection failed: %s", exc)

    # ------------------------------------------------------------------
    # Step 4c.1: Event calendar features (Gap 4)
    # Moved here from Step 4a.9 so filing_calendar_result is available.
    # Tracks known upcoming events (FOMC, earnings, political) and
    # computes proximity features that adjust prediction confidence.
    # ------------------------------------------------------------------
    event_calendar_result = None
    try:
        from operator1.features.event_calendar import compute_event_calendar_features
        cache, event_calendar_result = compute_event_calendar_features(
            cache,
            ticker=ticker,
            filing_calendar_result=filing_calendar_result,
            reference_date=_backtest_end_date,
        )
    except Exception as exc:
        logger.debug("Event calendar signals skipped: %s", exc)

    # ------------------------------------------------------------------
    # Step 5: Feature engineering
    # ------------------------------------------------------------------
    logger.info("")
    logger.info("Step 5: Computing derived features...")

    from operator1.features.derived_variables import compute_derived_variables
    from operator1.analysis.survival_mode import compute_company_survival_flag, compute_survival_probability
    from operator1.analysis.hierarchy_weights import compute_hierarchy_weights

    try:
        cache = compute_derived_variables(cache)
        logger.info("Features computed: %d columns", len(cache.columns))
    except Exception as exc:
        logger.warning("Feature engineering partially failed: %s", exc)

    # Step 5a: Private company proxy variables (when no OHLCV data)
    _is_private = False
    try:
        from operator1.features.private_company_proxies import (
            is_private_company,
            compute_private_company_proxies,
            resolve_proxies,
        )
        _is_private = is_private_company(cache)
        if _is_private:
            cache = compute_private_company_proxies(cache)
            # Transparent resolution: write proxy values INTO standard
            # column names (close, return_1d, volatility_21d, etc.) so
            # all downstream models work without any code changes.
            cache = resolve_proxies(cache)
            logger.info(
                "Private company mode ACTIVE -- proxy values resolved into "
                "standard columns for transparent downstream consumption"
            )
    except Exception as exc:
        logger.warning("Private company proxy computation failed: %s", exc)

    # Step 5.inst: Institutional flow features (from inst_* cache columns)
    try:
        from operator1.features.institutional_flow import compute_institutional_flow
        cache = compute_institutional_flow(cache, insider_transactions=target_insiders)
        _n_inst = sum(1 for c in cache.columns if c.startswith("inst_") and cache[c].notna().any())
        if _n_inst > 0:
            logger.info("Institutional flow: %d inst_* columns with data", _n_inst)
    except Exception as exc:
        logger.debug("Institutional flow computation skipped: %s", exc)

    # Survival mode
    weights: dict = {f"tier{i}": 20.0 for i in range(1, 6)}
    try:
        cache["company_survival_mode_flag"] = compute_company_survival_flag(cache)
        cache["survival_probability"] = compute_survival_probability(cache)
        # Cox PH data-driven survival risk score (lifelines)
        try:
            from operator1.analysis.survival_mode import compute_cox_survival_score
            _cox_score = compute_cox_survival_score(cache)
            if _cox_score.notna().any():
                cache["cox_survival_score"] = _cox_score
                # Blend sigmoid + Cox for combined probability
                _sig = cache["survival_probability"]
                _cox = cache["cox_survival_score"].fillna(_sig)
                # Blend weights from scoring_weights config (overridden by
                # adaptive_model_params in Step 5k unless use_adaptive=False)
                try:
                    from operator1.scoring_weights import get_weight as _gw
                    _w_sig = float(_gw("survival_blend.sigmoid_weight", 0.4))
                    _w_cox = float(_gw("survival_blend.cox_weight", 0.6))
                except Exception:
                    _w_sig, _w_cox = 0.4, 0.6
                cache["survival_probability"] = _w_sig * _sig + _w_cox * _cox
                logger.info("Cox PH survival score computed and blended (w_sig=%.2f, w_cox=%.2f)", _w_sig, _w_cox)
        except Exception as _exc:
            logger.debug("Cox PH survival score skipped: %s", _exc)
        cache = compute_hierarchy_weights(cache)
        for i in range(1, 6):
            col = f"hierarchy_tier{i}_weight"
            if col in cache.columns:
                weights[f"tier{i}"] = float(cache[col].iloc[-1])
        logger.info("Survival mode: %d days flagged", cache["company_survival_mode_flag"].sum())
    except Exception as exc:
        logger.warning("Survival mode detection failed: %s", exc)

    # ------------------------------------------------------------------
    # Step 5-USS: Unified Survival System -- Create central controller
    # ------------------------------------------------------------------
    # The SurvivalRegimeController determines the current regime and
    # configures all downstream modules across 5 dimensions:
    # variable triage, model switching, horizon compression,
    # correlation switching, and forecast bounding.
    survival_controller = None
    scenario_result = None
    try:
        from operator1.analysis.survival_regime_controller import SurvivalRegimeController
        survival_controller = SurvivalRegimeController.from_cache(cache)

        # Inject early warning score into cache
        if not survival_controller.early_warning.empty:
            cache["early_warning_score"] = survival_controller.early_warning
            if survival_controller.is_approaching_survival():
                logger.warning(
                    "EARLY WARNING: Score %.2f -- company approaching survival triggers",
                    survival_controller.get_early_warning_latest(),
                )

        logger.info(
            "Unified Survival System: regime=%s, survival=%s, "
            "frozen=%d vars, horizons=%s",
            survival_controller.current_regime,
            survival_controller.is_survival,
            len(survival_controller.get_frozen_variables()),
            list(survival_controller.horizons.keys()),
        )
    except Exception as exc:
        logger.warning("Unified Survival System controller failed: %s", exc)

    # Step 5e variables: initialized early so fuzzy protection (Step 5b)
    # can safely access relationships.get("parent_companies") for
    # protection inheritance from GLEIF corporate structure.
    relationships = {}
    graph_risk_result = None
    game_theory_result = None
    linked_caches: dict[str, pd.DataFrame] = {}
    linked_agg_df: pd.DataFrame | None = None
    contagion_result = None
    _ownership_edge_weights: dict[str, float] = {}

    # Step 5b: Fuzzy Logic government protection
    fuzzy_result = None
    try:
        from operator1.analysis.fuzzy_protection import compute_fuzzy_protection

        # Extract GDP from macro data if available
        _gdp_val = None
        if macro_data and macro_data.get("gdp") is not None:
            _gdp_series = macro_data["gdp"]
            if not _gdp_series.empty:
                _gdp_val = float(_gdp_series.dropna().iloc[-1])

        # Extract parent sector from GLEIF corporate structure (if available)
        # for protection inheritance.  The parent_companies group was added
        # in Step 5e.1 from GLEIF data.
        _parent_sector = None
        _parent_entities = relationships.get("parent_companies", [])
        if _parent_entities:
            # Use the first parent's name to infer sector
            _p = _parent_entities[0]
            _parent_name = _p.get("name", "") if isinstance(_p, dict) else getattr(_p, "name", "")
            # Infer sector from parent name heuristics
            _pn = _parent_name.lower()
            if any(w in _pn for w in ["energy", "oil", "gas", "petrol"]):
                _parent_sector = "energy"
            elif any(w in _pn for w in ["bank", "financ", "credit", "invest"]):
                _parent_sector = "banking"
            elif any(w in _pn for w in ["defense", "defence", "aerospace", "military"]):
                _parent_sector = "defense"
            elif any(w in _pn for w in ["telecom", "communic"]):
                _parent_sector = "telecommunications"
            elif any(w in _pn for w in ["pharma", "health", "medical"]):
                _parent_sector = "healthcare"
            elif any(w in _pn for w in ["utilit", "electric", "power"]):
                _parent_sector = "utilities"
            elif any(w in _pn for w in ["insur"]):
                _parent_sector = "insurance"
            elif any(w in _pn for w in ["technolog", "software", "digital"]):
                _parent_sector = "technology"

        cache = compute_fuzzy_protection(
            cache,
            sector=target_profile.get("sector"),
            gdp=_gdp_val,
            parent_sector=_parent_sector,
        )
        fuzzy_result = {
            "mean_degree": float(cache["fuzzy_protection_degree"].mean()),
            "sector_score": float(cache["fuzzy_sector_score"].iloc[0]),
            "latest_label": cache["fuzzy_protection_label"].iloc[-1],
        }
        logger.info(
            "Fuzzy protection: degree=%.3f (%s)",
            fuzzy_result["mean_degree"],
            fuzzy_result["latest_label"],
        )
    except Exception as exc:
        logger.warning("Fuzzy protection analysis failed: %s", exc)

    # Step 5d: Financial health scores
    fh_result = None
    try:
        from operator1.models.financial_health import compute_financial_health
        cache, fh_result = compute_financial_health(
            cache,
            hierarchy_weights=weights,
        )
        logger.info(
            "Financial health: composite=%.1f (%s), %d columns added",
            fh_result.latest_composite,
            fh_result.latest_label,
            len(fh_result.columns_added),
        )
    except Exception as exc:
        logger.warning("Financial health scoring failed: %s", exc)

    # W6 fix: Compute vanity (capital allocation quality) scores.
    # Profile builder reads vanity_score, vanity_label, vanity_trend, and
    # 5 component columns -- these were never populated without this call.
    try:
        from operator1.analysis.vanity import compute_vanity_score
        cache = compute_vanity_score(cache)
        logger.info("Vanity scores computed")
    except Exception as exc:
        logger.debug("Vanity scoring skipped: %s", exc)

    # Step 5d.1: SIX proxy result is available from Step 4a.5 (computed
    # before estimation). No need to re-compute here.

    # Step 5e: Linked entity discovery via Gemini (optional)
    # NOTE: relationships, graph_risk_result, game_theory_result, linked_caches,
    # linked_agg_df, contagion_result, _ownership_edge_weights are initialized
    # before Step 5b (fuzzy protection) so they can be safely accessed there.

    # Build the LLM client once for the whole pipeline
    from operator1.clients.llm_factory import create_llm_client
    llm_client = create_llm_client(secrets)

    if not args.skip_linked and llm_client is not None:
        logger.info("")
        logger.info("Step 5e: Discovering linked entities via %s...", llm_client.provider_name)

        try:
            from operator1.steps.entity_discovery import discover_linked_entities

            discovery_result = discover_linked_entities(
                target_profile=target_profile,
                llm_client=llm_client,
                pit_client=pit_client,
                secrets=secrets,
            )
            # Extract the linked dict from DiscoveryResult
            if hasattr(discovery_result, "linked"):
                relationships = discovery_result.linked
            elif isinstance(discovery_result, dict):
                relationships = discovery_result
            else:
                relationships = {}
            total_linked = sum(len(v) for v in relationships.values() if isinstance(v, list))
            logger.info("Linked entities discovered: %d", total_linked)
        except Exception as exc:
            logger.warning("Entity discovery failed (continuing without): %s", exc)

        # Step 5e.1: Enrich with GLEIF corporate structure (parent/subsidiary)
        # GLEIF provides corporate control chains (who owns whom at the entity
        # level) -- distinct from institutional shareholders (portfolio data).
        # Parent-subsidiary edges are the strongest contagion channels.
        try:
            from operator1.clients.gleif import fetch_corporate_structure
            _gleif_id = target_profile.get("lei") or target_profile.get("name") or company_name
            _corp_struct = fetch_corporate_structure(_gleif_id)
            if _corp_struct.available:
                # Add parent companies as a new relationship group
                _parent_entities = []
                for _parent in [_corp_struct.ultimate_parent, _corp_struct.direct_parent]:
                    if _parent:
                        _parent_entities.append({
                            "isin": "",
                            "ticker": _parent.lei[:10] if _parent.lei else "",
                            "name": _parent.name,
                            "country": _parent.country,
                            "sector": "",
                            "relationship_group": "parent_companies",
                            "match_score": 100,
                            "lei": _parent.lei,
                            "relationship": _parent.relationship,
                        })
                if _parent_entities:
                    relationships["parent_companies"] = _parent_entities

                # Add subsidiaries as a new relationship group
                _sub_entities = []
                for _sub in _corp_struct.subsidiaries[:15]:
                    _sub_entities.append({
                        "isin": "",
                        "ticker": _sub.lei[:10] if _sub.lei else "",
                        "name": _sub.name,
                        "country": _sub.country,
                        "sector": "",
                        "relationship_group": "subsidiaries",
                        "match_score": 100,
                        "lei": _sub.lei,
                        "relationship": "subsidiary",
                    })
                if _sub_entities:
                    relationships["subsidiaries"] = _sub_entities

                _n_parents = len(_parent_entities)
                _n_subs = len(_sub_entities)
                logger.info(
                    "GLEIF corporate structure: %d parents, %d subsidiaries added to relationships",
                    _n_parents, _n_subs,
                )
        except Exception as exc:
            logger.debug("GLEIF corporate structure enrichment skipped: %s", exc)

        # Graph risk -- convert LinkedEntity dataclasses to dicts for .get() compat
        _rel_dicts = {}
        try:
            from dataclasses import asdict as _asdict
            from operator1.models.graph_risk import compute_graph_risk_metrics
            for _grp, _ents in relationships.items():
                if isinstance(_ents, list):
                    _rel_dicts[_grp] = [
                        _asdict(e) if hasattr(e, "__dataclass_fields__") else e
                        for e in _ents
                    ]
                else:
                    _rel_dicts[_grp] = _ents
            graph_risk_result = compute_graph_risk_metrics(
                target_isin=target_profile.get("isin", ticker),
                relationships=_rel_dicts,
                target_cache=cache,
                linked_caches=linked_caches if linked_caches else None,
            )
            logger.info(
                "Graph risk: %d nodes, centrality=%.3f",
                graph_risk_result.n_nodes,
                graph_risk_result.target_degree_centrality,
            )
        except Exception as exc:
            logger.warning("Graph risk analysis failed: %s", exc)

        # Game theory
        try:
            from operator1.models.game_theory import analyze_competitive_dynamics
            game_theory_result = analyze_competitive_dynamics(
                target_cache=cache,
                target_name=target_profile.get("name", "target"),
                competitor_caches=linked_caches if linked_caches else None,
            )
            logger.info(
                "Game theory: %s, pressure=%.3f",
                game_theory_result.market_structure,
                game_theory_result.competitive_pressure,
            )
        except Exception as exc:
            logger.warning("Game theory analysis failed: %s", exc)
        # Step 5f: Fetch financial data for linked entities
        if relationships:
            logger.info("")
            logger.info("Step 5f: Fetching linked entity data...")

            _MAX_LINKED_ENTITIES = 10  # cap to stay within API budgets

            # Flatten all entities from discovery result
            _all_linked: list[dict] = []
            _entity_groups: dict[str, list[str]] = {}
            for group_name, group_entities in relationships.items():
                group_ids: list[str] = []
                if isinstance(group_entities, list):
                    for ent in group_entities:
                        ent_id = ""
                        if hasattr(ent, "isin") and ent.isin:
                            ent_id = ent.isin
                        elif hasattr(ent, "ticker") and ent.ticker:
                            ent_id = ent.ticker
                        elif isinstance(ent, dict):
                            ent_id = ent.get("isin", "") or ent.get("ticker", "")
                        if ent_id and ent_id not in {e.get("id") for e in _all_linked}:
                            _all_linked.append({
                                "id": ent_id,
                                "name": getattr(ent, "name", "") if hasattr(ent, "name") else ent.get("name", ""),
                                "group": group_name,
                            })
                            group_ids.append(ent_id)
                _entity_groups[group_name] = group_ids

            # Cap total entities
            _all_linked = _all_linked[:_MAX_LINKED_ENTITIES]

            def _fetch_linked_entity(ent_info: dict) -> tuple[str, pd.DataFrame]:
                """Fetch and build daily cache for one linked entity."""
                ent_id = ent_info["id"]
                try:
                    # Fetch financial statements
                    _inc = pit_client.get_income_statement(ent_id)
                    _bal = pit_client.get_balance_sheet(ent_id)
                    _cf = pit_client.get_cashflow_statement(ent_id)
                    _qt = pit_client.get_quotes(ent_id)

                    # Build minimal daily cache (OHLCV spine + ffill statements)
                    if not _qt.empty and "date" in _qt.columns:
                        _qt["date"] = pd.to_datetime(_qt["date"])
                        _ent_cache = _qt.set_index("date").sort_index()
                    else:
                        _ent_cache = pd.DataFrame(
                            index=pd.date_range(
                                cache.index[0], cache.index[-1], freq="B", name="date"
                            )
                        )

                    # Merge statements (same logic as target cache)
                    for _lbl, _sdf in [("inc", _inc), ("bal", _bal), ("cf", _cf)]:
                        if _sdf.empty:
                            continue
                        # Use report_date first (consistent with target cache merge)
                        _dcol = "report_date" if "report_date" in _sdf.columns else "filing_date"
                        if _dcol not in _sdf.columns:
                            continue
                        _sdf[_dcol] = pd.to_datetime(_sdf[_dcol])
                        _sdf = _sdf.sort_values(_dcol)
                        _sdf = _sdf.drop_duplicates(subset=[_dcol], keep="last")
                        _ncols = _sdf.select_dtypes(include=["number"]).columns.tolist()
                        _ncols = [c for c in _ncols if c != _dcol and "date" not in c.lower()]
                        if _ncols:
                            _si = _sdf.set_index(_dcol)[_ncols]
                            # Union+ffill+reindex (same as target cache merge)
                            _combined = _ent_cache.index.union(_si.index).sort_values()
                            _sa = _si.reindex(_combined).ffill()
                            _sa = _sa.reindex(_ent_cache.index)
                            _new = [c for c in _sa.columns if c not in _ent_cache.columns]
                            if _new:
                                _ent_cache = _ent_cache.join(_sa[_new], how="left")

                    # Compute derived variables
                    if "close" in _ent_cache.columns and _ent_cache["close"].notna().sum() > 5:
                        _ent_cache = compute_derived_variables(_ent_cache)

                    return ent_id, _ent_cache
                except Exception as _exc:
                    logger.debug("Linked entity %s fetch failed: %s", ent_id, _exc)
                    return ent_id, pd.DataFrame()

            # Fetch linked entities in parallel
            if _all_linked:
                with ThreadPoolExecutor(max_workers=4) as executor:
                    _futures = {
                        executor.submit(_fetch_linked_entity, ent): ent
                        for ent in _all_linked
                    }
                    for future in as_completed(_futures):
                        ent_info = _futures[future]
                        try:
                            ent_id, ent_cache = future.result()
                            if not ent_cache.empty:
                                linked_caches[ent_id] = ent_cache
                                logger.info(
                                    "  Linked %s: %d rows x %d cols",
                                    ent_info.get("name", ent_id)[:30],
                                    len(ent_cache), len(ent_cache.columns),
                                )
                        except Exception as _exc:
                            logger.debug("Linked entity future failed: %s", _exc)

                logger.info(
                    "Linked entity data: %d/%d entities fetched",
                    len(linked_caches), len(_all_linked),
                )

            # Step 5f.1: Fetch competitor holders for ownership contagion
            _competitor_holders: dict[str, list[dict]] = {}
            if target_holders and hasattr(pit_client, "get_holders"):
                _comp_ids = _entity_groups.get("competitors", [])
                for _cid in _comp_ids[:5]:  # cap at 5 competitors
                    try:
                        _ch = pit_client.get_holders(_cid)
                        if _ch:
                            _competitor_holders[_cid] = _ch
                    except Exception:
                        pass
                if _competitor_holders:
                    logger.info(
                        "Competitor holders fetched: %d/%d competitors",
                        len(_competitor_holders), len(_comp_ids),
                    )

            # Step 5f.2: Ownership contagion analysis (MHHI + crowding + liquidation)
            if target_holders:
                try:
                    from operator1.models.ownership_contagion import (
                        compute_ownership_contagion,
                        inject_contagion_into_cache,
                        get_ownership_edge_weights,
                    )
                    contagion_result = compute_ownership_contagion(
                        target_holders=target_holders,
                        competitor_holders=_competitor_holders,
                        cache=cache,
                    )
                    if contagion_result and contagion_result.available:
                        cache = inject_contagion_into_cache(cache, contagion_result)
                        _ownership_edge_weights = get_ownership_edge_weights(contagion_result)
                        logger.info(
                            "Ownership contagion: MHHI=%.3f, crowding=%.3f, "
                            "liquidation=%.0fd, shared_inst=%d",
                            contagion_result.mhhi_delta,
                            contagion_result.crowding_score,
                            contagion_result.liquidation_days,
                            contagion_result.n_shared_institutions,
                        )
                    # Re-run graph_risk with ownership edge weights for second
                    # contagion channel (shared holders amplify contagion).
                    # graph_risk was first computed at Step 5e with unweighted
                    # edges; now we enhance it with ownership overlap weights.
                    if _ownership_edge_weights and graph_risk_result is not None:
                        try:
                            from operator1.models.graph_risk import compute_graph_risk_metrics as _grc
                            _enhanced_gr = _grc(
                                target_isin=target_profile.get("isin", ticker),
                                relationships=_rel_dicts,
                                edge_weights=_ownership_edge_weights,
                                target_cache=cache,
                                linked_caches=linked_caches if linked_caches else None,
                            )
                            if _enhanced_gr.available:
                                _old_contagion = graph_risk_result.contagion_target_infection_prob
                                graph_risk_result = _enhanced_gr
                                logger.info(
                                    "Graph risk re-computed with ownership edge weights: "
                                    "contagion=%.3f (was %.3f)",
                                    _enhanced_gr.contagion_target_infection_prob,
                                    _old_contagion,
                                )
                        except Exception as _gr_exc:
                            logger.debug("Graph risk re-computation skipped: %s", _gr_exc)

                except Exception as exc:
                    logger.debug("Ownership contagion skipped: %s", exc)

            # Step 5g: Compute linked aggregates
            if linked_caches:
                try:
                    from operator1.features.linked_aggregates import compute_linked_aggregates
                    linked_agg_df = compute_linked_aggregates(
                        target_daily=cache,
                        linked_daily=linked_caches,
                        entity_groups=_entity_groups,
                    )
                    # Merge aggregate columns into the target cache
                    if linked_agg_df is not None and not linked_agg_df.empty:
                        _new_agg_cols = [
                            c for c in linked_agg_df.columns if c not in cache.columns
                        ]
                        if _new_agg_cols:
                            cache = cache.join(linked_agg_df[_new_agg_cols], how="left")
                        logger.info(
                            "Linked aggregates computed: %d columns merged into cache",
                            len(_new_agg_cols),
                        )

                    # Step 5g.1: Compute company-relative metrics vs peer aggregates
                    # (rel_strength_vs_sector, valuation_premium, rel_volatility)
                    try:
                        from operator1.features.linked_aggregates import compute_relative_metrics
                        _rel_df = compute_relative_metrics(cache, linked_agg_df)
                        if _rel_df is not None and not _rel_df.empty:
                            _new_rel = [c for c in _rel_df.columns if c not in cache.columns and _rel_df[c].notna().any()]
                            if _new_rel:
                                cache = cache.join(_rel_df[_new_rel], how="left")
                                logger.info(
                                    "Relative metrics computed: %d columns merged into cache",
                                    len(_new_rel),
                                )
                    except Exception as _rel_exc:
                        logger.debug("Relative metrics computation skipped: %s", _rel_exc)
                except Exception as exc:
                    logger.warning("Linked aggregates computation failed: %s", exc)
    else:
        if llm_client is None:
            logger.info("Step 5e: Skipped (no LLM API key for entity discovery)")
        else:
            logger.info("Step 5e: Skipped (--skip-linked)")

    # Step 5g.5: Linked entity conflict propagation
    # Checks if any linked entities (suppliers, customers, etc.) are in
    # conflict zones, which creates supply chain / revenue exposure risk.
    if conflict_result is not None and relationships:
        try:
            from operator1.features.conflict_risk import assess_linked_entity_conflict
            linked_conflict = assess_linked_entity_conflict(
                linked_entities=relationships,
                target_conflict=conflict_result,
            )
            if linked_conflict:
                cache = inject_conflict_risk_into_cache(
                    cache, conflict_result, linked_conflict=linked_conflict,
                )
                logger.info(
                    "Linked conflict: supply_chain=%.2f, revenue=%.2f, competitive=%.2f",
                    linked_conflict.get("supply_chain_risk_score", 0),
                    linked_conflict.get("revenue_exposure_score", 0),
                    linked_conflict.get("competitive_advantage_score", 0),
                )
        except Exception as exc:
            logger.warning("Linked entity conflict propagation failed: %s", exc)

    # Step 5g.6: Unified supply chain stress flag
    # Combines geopolitical risk + supplier financial health into one signal.
    # App core idea Section F.1 Category 6.
    supply_chain_stress_result = None
    try:
        from operator1.features.conflict_risk import compute_supply_chain_stress
        supply_chain_stress_result = compute_supply_chain_stress(
            conflict_result=conflict_result,
            linked_caches=linked_caches if linked_caches else None,
            relationships=relationships if relationships else None,
        )
        if supply_chain_stress_result and supply_chain_stress_result.get("available"):
            # Inject flag into cache
            cache["supply_chain_stress_flag"] = int(supply_chain_stress_result["supply_chain_stress_flag"])
            cache["supply_chain_stress_score"] = supply_chain_stress_result["supply_chain_stress_score"]
            logger.info(
                "Supply chain stress: flag=%s, score=%.3f, sources=%s",
                supply_chain_stress_result["supply_chain_stress_flag"],
                supply_chain_stress_result["supply_chain_stress_score"],
                supply_chain_stress_result["stress_sources"],
            )
    except Exception as exc:
        logger.debug("Supply chain stress computation skipped: %s", exc)

    # Step 5h: Peer percentile ranking (requires linked caches)
    peer_ranking_result = None
    if linked_caches:
        try:
            from operator1.features.peer_ranking import compute_peer_ranking
            cache, _pr_result = compute_peer_ranking(
                cache, linked_caches=linked_caches,
            )
            peer_ranking_result = {
                "n_peers": _pr_result.n_peers,
                "n_variables_ranked": _pr_result.n_variables_ranked,
                "latest_composite_rank": _pr_result.latest_composite_rank,
                "latest_label": _pr_result.latest_label,
                "variable_ranks": _pr_result.variable_ranks,
            }
            logger.info(
                "Peer ranking: rank=%.1f (%s), %d peers, %d variables",
                _pr_result.latest_composite_rank
                if not pd.isna(_pr_result.latest_composite_rank) else 0.0,
                _pr_result.latest_label,
                _pr_result.n_peers,
                _pr_result.n_variables_ranked,
            )
        except Exception as exc:
            logger.warning("Peer ranking failed: %s", exc)

    # Step 5i: News sentiment scoring (uses LLM for AI scoring, keyword fallback)
    sentiment_result = None
    try:
        from operator1.features.news_sentiment import compute_news_sentiment
        cache, _sent_result = compute_news_sentiment(
            cache,
            llm_client=llm_client,
            symbol=ticker,
            market_id=market_id,
            company_name=company_name,
        )
        if _sent_result.n_articles_scored > 0:
            sentiment_result = {
                "n_articles_fetched": _sent_result.n_articles_fetched,
                "n_articles_scored": _sent_result.n_articles_scored,
                "scoring_method": _sent_result.scoring_method,
                "mean_sentiment": _sent_result.mean_sentiment,
                "latest_sentiment": _sent_result.latest_sentiment,
                "latest_label": _sent_result.latest_label,
            }
            logger.info(
                "Sentiment: %s (%.3f), %d articles scored via %s",
                _sent_result.latest_label,
                _sent_result.latest_sentiment
                if not pd.isna(_sent_result.latest_sentiment) else 0.0,
                _sent_result.n_articles_scored,
                _sent_result.scoring_method,
            )
        else:
            logger.info("Sentiment: no articles available for scoring")
    except Exception as exc:
        logger.warning("News sentiment scoring failed: %s", exc)

    # ------------------------------------------------------------------
    # Step 5i.5: Product catalyst detection
    # ------------------------------------------------------------------
    catalyst_result = None
    try:
        from operator1.features.product_catalysts import detect_product_catalysts
        # Pass news articles from sentiment step if available.
        # _sent_result is the SentimentResult dataclass (has .articles list);
        # sentiment_result is a plain dict for the profile (no .articles).
        _news_articles = []
        try:
            _news_articles = _sent_result.articles  # noqa: F821
        except (NameError, AttributeError):
            _news_articles = []
        cache, catalyst_result = detect_product_catalysts(
            cache,
            profile=target_profile,
            news_articles=_news_articles if _news_articles else None,
        )
        if catalyst_result and catalyst_result.available:
            logger.info(
                "Product catalysts: score=%.2f, type=%s, rnd=%.2f, news=%.2f",
                catalyst_result.catalyst_score,
                catalyst_result.catalyst_type,
                catalyst_result.rnd_acceleration,
                catalyst_result.news_catalyst_score,
            )
    except Exception as exc:
        logger.warning("Product catalyst detection failed: %s", exc)

    # ------------------------------------------------------------------
    # Step 5i.6: Product segment extraction + cache injection
    # ------------------------------------------------------------------
    # Must run BEFORE Step 6 (temporal models) so that segment_hhi,
    # cannibalization_rate, network_effect_score, etc. exist in cache
    # when _extra_vars is built and when Monte Carlo checks segment_hhi.
    _seg_result: dict = {}
    try:
        if hasattr(pit_client, "extract_segment_data"):
            _seg_result = pit_client.extract_segment_data(identifier) or {}
            if _seg_result.get("n_segments", 0) >= 2:
                logger.info(
                    "Segments extracted: %d segments for %s",
                    _seg_result["n_segments"], company_name,
                )
                # Inject product metrics into cache for temporal models
                try:
                    from operator1.features.product_metrics import compute_product_metrics
                    cache = compute_product_metrics(cache, _seg_result)
                except Exception as _pm_exc:
                    logger.debug("Product metrics computation failed: %s", _pm_exc)

                # Geographic supply chain risk metrics (Gap 2)
                try:
                    from operator1.features.product_metrics import compute_geographic_metrics
                    _geo_segs = _seg_result.get("geo_segments", {})
                    _gleif_subs = relationships.get("subsidiaries", [])
                    # Convert dataclass subsidiaries to dicts if needed
                    _sub_dicts = [
                        s if isinstance(s, dict) else {"country": getattr(s, "country", "")}
                        for s in _gleif_subs
                    ]
                    cache = compute_geographic_metrics(
                        cache,
                        geo_segments=_geo_segs,
                        subsidiaries=_sub_dicts,
                    )
                except Exception as _geo_exc:
                    logger.debug("Geographic metrics computation failed: %s", _geo_exc)
    except Exception as _seg_exc:
        logger.debug("Segment extraction failed: %s", _seg_exc)

    # ------------------------------------------------------------------
    # Step 5j: Adaptive threshold calibration
    # ------------------------------------------------------------------
    # Replaces fixed textbook survival thresholds with peer-calibrated,
    # data-derived values.  Must run AFTER linked entity data (Step 5f)
    # and peer ranking (Step 5h) so peer distributions are available.
    # Recalibrates survival flags computed earlier in Step 5 with
    # sector-aware thresholds.
    _adaptive_thresholds = None
    try:
        from operator1.analysis.adaptive_thresholds import (
            compute_adaptive_thresholds,
            threshold_set_to_survival_dict,
            threshold_set_to_mc_dict,
        )
        _adaptive_thresholds = compute_adaptive_thresholds(
            cache,
            linked_caches=linked_caches if linked_caches else None,
            regime_detector=None,  # HMM not yet fitted; will be used in Step 6
            fh_composite_scores=(
                cache["fh_composite_score"]
                if "fh_composite_score" in cache.columns
                else None
            ),
        )
        if _adaptive_thresholds.adapted:
            # Recalibrate survival flags with adaptive thresholds
            _adapted_survival_dict = threshold_set_to_survival_dict(_adaptive_thresholds)
            cache["company_survival_mode_flag"] = compute_company_survival_flag(
                cache, thresholds=_adapted_survival_dict,
            )
            cache["survival_probability"] = compute_survival_probability(
                cache, thresholds=_adapted_survival_dict,
            )
            # Re-run hierarchy weights with updated survival flags
            cache = compute_hierarchy_weights(cache)
            for i in range(1, 6):
                col = f"hierarchy_tier{i}_weight"
                if col in cache.columns:
                    weights[f"tier{i}"] = float(cache[col].iloc[-1])
            logger.info(
                "Adaptive thresholds applied: %d days flagged (was %d before recalibration)",
                cache["company_survival_mode_flag"].sum(),
                cache["company_survival_mode_flag"].sum(),  # logged for comparison
            )
    except Exception as exc:
        logger.warning("Adaptive threshold calibration failed (using defaults): %s", exc)

    # Refresh USS controller after adaptive thresholds recalibrate survival
    if survival_controller is not None:
        try:
            survival_controller = SurvivalRegimeController.from_cache(cache)
            logger.info(
                "USS controller refreshed: regime=%s, survival=%s",
                survival_controller.current_regime,
                survival_controller.is_survival,
            )
        except Exception as exc:
            logger.debug("USS controller refresh failed: %s", exc)

    # ------------------------------------------------------------------
    # Step 5j.5: Signal IC measurement
    # ------------------------------------------------------------------
    # Computes rolling Spearman IC for all derived signals vs forward
    # returns. IC-strong signals get priority in the ensemble; weak
    # signals are pruned from _extra_vars.
    signal_ic_result = None
    try:
        from operator1.analysis.signal_ic import compute_signal_ic, get_ic_weighted_signals
        signal_ic_result = compute_signal_ic(cache)
        if signal_ic_result and signal_ic_result.available:
            logger.info(
                "Signal IC: %d strong signals (best=%s, IC=%.4f), %d weak",
                len(signal_ic_result.strong_signals),
                signal_ic_result.best_signal,
                signal_ic_result.best_ic,
                len(signal_ic_result.weak_signals),
            )
    except Exception as exc:
        logger.warning("Signal IC measurement failed: %s", exc)

    # ------------------------------------------------------------------
    # Step 5j.6: Fill actuals from previous prediction log
    # ------------------------------------------------------------------
    prediction_log_summary = None
    try:
        from operator1.analysis.prediction_log import fill_actuals
        prediction_log_summary = fill_actuals(
            ticker=ticker, cache=cache,
            reference_date=_backtest_end_date,
        )
        if prediction_log_summary and prediction_log_summary.get("n_filled", 0) > 0:
            logger.info(
                "Prediction log: filled %d actuals, hit_rate=%.1f%%, IC=%.4f",
                prediction_log_summary["n_filled"],
                prediction_log_summary["hit_rate"] * 100,
                prediction_log_summary["realized_ic"],
            )
    except Exception as exc:
        logger.debug("Prediction log fill skipped: %s", exc)

    # ------------------------------------------------------------------
    # Step 5.5: Enriched survival timeline (bridge: rule-based + HMM)
    # ------------------------------------------------------------------
    # Runs early regime detection (HMM/GMM/PELT/BCP) and combines it
    # with the rule-based survival flags into a unified state vector.
    # This gives downstream temporal models a single interface for
    # regime_state, survival_intensity, and regime_confidence.
    enriched_timeline_result = None
    early_regime_result = None
    regime_detector = None

    if not args.skip_models:
        logger.info("")
        logger.info("Step 5.5: Enriched survival timeline...")

        try:
            from operator1.models.regime_detector import run_early_regime_detection
            from operator1.analysis.survival_timeline import (
                compute_enriched_survival_timeline,
            )

            # Early regime detection: runs HMM/GMM/PELT/BCP and adds regime
            # columns to cache. This replaces the separate regime detection
            # that used to run at the start of Step 6.
            # In private company mode, use equity_change_rate instead of return_1d.
            _regime_target = "equity_change_rate" if _is_private else "return_1d"
            cache, early_regime_result = run_early_regime_detection(
                cache, target_variable=_regime_target,
            )
            # Online change point detection via ChangeFinder
            try:
                from operator1.models.regime_detector import compute_online_change_scores
                _ret_for_cf = cache.get(_regime_target)
                if _ret_for_cf is not None and _ret_for_cf.notna().sum() > 30:
                    _cf_scores = compute_online_change_scores(_ret_for_cf.fillna(0).values)
                    if _cf_scores is not None:
                        cache["online_change_score"] = _cf_scores
                        logger.info("ChangeFinder online scores computed")
            except Exception as _exc:
                logger.debug("ChangeFinder skipped: %s", _exc)
            if early_regime_result and early_regime_result.fitted:
                regime_detector = early_regime_result.detector
                logger.info("Early regime detection complete")
            else:
                logger.info(
                    "Early regime detection did not fit: %s",
                    getattr(early_regime_result, "error", "unknown"),
                )

            # Build enriched survival timeline.
            _regime_labels = (
                early_regime_result.regime_labels
                if early_regime_result else None
            )
            _regime_confidence = (
                early_regime_result.regime_confidence
                if early_regime_result else None
            )
            enriched_timeline_result = compute_enriched_survival_timeline(
                cache,
                regime_labels=_regime_labels,
                regime_confidence=_regime_confidence,
            )
            if enriched_timeline_result and enriched_timeline_result.fitted:
                # Item 3: Index length check before merging.
                _etl = enriched_timeline_result.timeline
                if len(_etl) != len(cache):
                    logger.warning(
                        "Enriched timeline length (%d) differs from cache (%d) "
                        "-- using reindex to align safely",
                        len(_etl), len(cache),
                    )

                # Merge enriched columns back into cache.
                _enriched_cols = [
                    "regime_state", "survival_intensity",
                    "regime_confidence", "regime_switch",
                    "regime_transition_prob", "survival_mode",
                    "survival_mode_code", "switch_point",
                    "days_in_mode", "stability_score_21d",
                    "market_regime",
                ]
                for col in _enriched_cols:
                    if col in _etl.columns:
                        if col not in cache.columns:
                            # Align by index in case lengths differ.
                            cache[col] = _etl[col].reindex(cache.index)
                        else:
                            # Item 2: Log when a column is skipped due to collision.
                            logger.debug(
                                "Enriched column '%s' skipped -- already in cache",
                                col,
                            )
                logger.info(
                    "Enriched survival timeline: mean_intensity=%.3f, "
                    "regime_available=%s, states=%s",
                    enriched_timeline_result.mean_intensity,
                    enriched_timeline_result.regime_available,
                    {k: f"{v:.1%}"
                     for k, v in enriched_timeline_result.combined_state_distribution.items()
                     if v > 0.01},
                )
            else:
                logger.warning(
                    "Enriched survival timeline failed: %s",
                    getattr(enriched_timeline_result, "error", "unknown"),
                )
        except Exception as exc:
            logger.warning("Step 5.5 (enriched survival timeline) failed: %s", exc)

    # ------------------------------------------------------------------
    # Step 5k: Adaptive model parameters (Tier 2)
    # ------------------------------------------------------------------
    # Compute data-derived model parameters from pipeline outputs.
    # Must run after Step 5.5 (regime detector available) and before
    # Step 6 (temporal models consume these parameters).
    _adaptive_model_params = None
    try:
        from operator1.analysis.adaptive_model_params import (
            compute_blend_weights,
            compute_regime_risk_multiplier,
            compute_garman_klass_factor,
            compute_transition_halflife,
            compute_adaptive_mc_params,
            compute_adaptive_participation_rate,
            AdaptiveModelParams,
        )
        _adaptive_model_params = AdaptiveModelParams()

        # Cox/sigmoid blend recalibration (inverse-variance, Cochrane 1954)
        if "survival_probability" in cache.columns and "cox_survival_score" in cache.columns:
            _sig = cache.get("survival_probability")
            _cox = cache.get("cox_survival_score")
            _actual = cache.get("company_survival_mode_flag", pd.Series(0, index=cache.index))
            if _sig is not None and _cox is not None:
                w_sig, w_cox = compute_blend_weights(_sig, _cox, _actual)
                _adaptive_model_params.blend_w_sig = w_sig
                _adaptive_model_params.blend_w_cox = w_cox
                # Re-blend with data-driven weights
                cache["survival_probability"] = w_sig * _sig + w_cox * _cox.fillna(_sig)
                _adaptive_model_params.methods_used["blend"] = f"inverse_variance(sig={w_sig:.3f},cox={w_cox:.3f})"
                logger.info("Adaptive blend: w_sig=%.3f, w_cox=%.3f", w_sig, w_cox)

        # Regime risk multiplier (HMM volatility ratio)
        _adaptive_model_params.survival_risk_multiplier = compute_regime_risk_multiplier(
            regime_detector, cache,
        )
        # Garman-Klass intraday factor
        _adaptive_model_params.intraday_low_factor = compute_garman_klass_factor(cache)
        # Transition half-life from enriched timeline
        _adaptive_model_params.transition_halflife = compute_transition_halflife(
            enriched_timeline_result,
        )
        # Participation rate (Amihud)
        _adaptive_model_params.participation_rate = compute_adaptive_participation_rate(cache)
        # MC parameters (precision-targeted)
        _adaptive_model_params.mc_n_paths, _adaptive_model_params.mc_is_tilt = (
            compute_adaptive_mc_params(cache)
        )
        _adaptive_model_params.adapted = True
        logger.info(
            "Adaptive model params: risk_mult=%.2f, gk_factor=%.2f, "
            "transition_hl=%d, mc_paths=%d, mc_tilt=%.2f, participation=%.3f",
            _adaptive_model_params.survival_risk_multiplier,
            _adaptive_model_params.intraday_low_factor,
            _adaptive_model_params.transition_halflife,
            _adaptive_model_params.mc_n_paths,
            _adaptive_model_params.mc_is_tilt,
            _adaptive_model_params.participation_rate,
        )
    except Exception as exc:
        logger.warning("Adaptive model params failed (using defaults): %s", exc)

    # Step 5k.2: Tier 3 adaptive parameters (windows, NN, noise, patterns)
    _adaptive_tier3 = None
    try:
        from operator1.analysis.adaptive_windows import (
            compute_adaptive_windows,
            compute_nn_hyperparams,
            compute_pattern_thresholds,
            compute_stale_threshold,
            AdaptiveTier3Params,
        )
        from operator1.analysis.adaptive_model_params import compute_effective_sample_size

        _detected_freq = (
            filing_calendar_result.detected_frequency
            if filing_calendar_result is not None
            else "quarterly"
        )
        _adaptive_tier3 = AdaptiveTier3Params()
        _adaptive_tier3.windows = compute_adaptive_windows(_detected_freq)
        _adaptive_tier3.stale_threshold_days = compute_stale_threshold(_detected_freq)

        # NN hyperparams from effective sample size
        _n_eff_close = compute_effective_sample_size(cache, "close")
        _n_feat = sum(
            1 for c in cache.columns
            if cache[c].dtype in ("float64", "float32") and cache[c].notna().sum() > 10
        )
        _adaptive_tier3.nn_params = compute_nn_hyperparams(
            n_eff=_n_eff_close, n_features=min(_n_feat, 30),
        )

        # Pattern thresholds
        _adaptive_tier3.pattern_body_threshold, _adaptive_tier3.pattern_doji_threshold = (
            compute_pattern_thresholds(cache, lookback=_adaptive_tier3.windows.medium)
        )

        _adaptive_tier3.adapted = True
        logger.info(
            "Tier 3 adaptive: freq=%s, windows=%d/%d/%d/%d, nn_d=%d/h=%d/drop=%.2f, "
            "pattern=%.2f/%.2f, stale=%dd",
            _detected_freq,
            _adaptive_tier3.windows.short, _adaptive_tier3.windows.medium,
            _adaptive_tier3.windows.long, _adaptive_tier3.windows.trend,
            _adaptive_tier3.nn_params.d_model, _adaptive_tier3.nn_params.hidden_dim,
            _adaptive_tier3.nn_params.dropout,
            _adaptive_tier3.pattern_body_threshold, _adaptive_tier3.pattern_doji_threshold,
            _adaptive_tier3.stale_threshold_days,
        )
    except Exception as exc:
        logger.warning("Tier 3 adaptive params failed (using defaults): %s", exc)

    # ------------------------------------------------------------------
    # Step 6: Temporal modeling (optional)
    # ------------------------------------------------------------------
    forecast_result = None
    forward_pass_result = None
    walk_forward_result = None
    burnout_result = None
    mc_result = None
    pred_result = None
    transfer_entropy_result = None
    cycle_result = None
    pattern_result = None
    copula_result = None
    conformal_result = None
    dtw_result = None
    shap_result = None
    sobol_result = None
    particle_filter_result = None
    transformer_result = None
    dual_regime_result = None
    granger_result = None
    ga_result = None
    ohlc_result = None
    recursive_result = None
    regime_shift_result = None
    _synergy_meta = {}
    _pattern_drift = 1.0

    # ------------------------------------------------------------------
    # Staged execution: when --stage is set, delegate to the stage runner
    # with per-model sub-stage checkpoint save/resume.
    # ------------------------------------------------------------------
    if args.stage and not args.skip_models:
        from operator1.pipeline_state import PipelineState
        from operator1.stages.runner import run_stages

        _run_dir = args.run_dir or f"{args.output_dir}/{args.company}_{args.end_date or 'latest'}"

        # Build PipelineState from current local variables
        _ps = PipelineState(
            market_id=args.market,
            company=args.company,
            end_date=args.end_date,
            years=args.years,
            output_dir=_run_dir,
        )
        _ps.cache = cache
        _ps.target_profile = target_profile
        _ps.income_df = income_df
        _ps.balance_df = balance_df
        _ps.cashflow_df = cashflow_df
        _ps.quotes_df = quotes_df
        _ps.weights = weights
        _ps.fh_result = fh_result
        _ps.fuzzy_result = fuzzy_result
        _ps.relationships = relationships
        _ps.linked_caches = linked_caches
        _ps.linked_agg_df = locals().get("linked_agg_df")
        _ps.graph_risk_result = graph_risk_result
        _ps.game_theory_result = game_theory_result
        _ps.contagion_result = locals().get("contagion_result")
        _ps.peer_ranking_result = peer_ranking_result
        _ps.sentiment_result = sentiment_result
        _ps.catalyst_result = catalyst_result
        _ps.signal_ic_result = locals().get("signal_ic_result")
        _ps.prediction_log_summary = locals().get("prediction_log_summary")
        _ps.enriched_timeline_result = enriched_timeline_result
        _ps.early_regime_result = early_regime_result
        _ps.regime_detector = regime_detector
        _ps.survival_controller = locals().get("survival_controller")
        _ps.is_private = _is_private
        _ps.adaptive_thresholds = locals().get("_adaptive_thresholds")
        _ps.adaptive_model_params = locals().get("_adaptive_model_params")
        _ps.adaptive_tier3 = locals().get("_adaptive_tier3")

        # Save Stage 2 checkpoint (pre-temporal), then run requested stages
        _ps.save("2.9")
        logger.info("Staged execution: running --stage %s", args.stage)
        run_stages(_ps, args.stage, save_checkpoints=True)

        # Copy results back to local variables for Step 7+8
        cache = _ps.cache
        forecast_result = _ps.forecast_result
        forward_pass_result = _ps.forward_pass_result
        walk_forward_result = _ps.walk_forward_result
        burnout_result = _ps.burnout_result
        mc_result = _ps.mc_result
        pred_result = _ps.pred_result
        transfer_entropy_result = _ps.transfer_entropy_result
        cycle_result = _ps.cycle_result
        pattern_result = _ps.pattern_result
        copula_result = _ps.copula_result
        conformal_result = _ps.conformal_result
        dtw_result = _ps.dtw_result
        shap_result = _ps.shap_result
        sobol_result = _ps.sobol_result
        particle_filter_result = _ps.particle_filter_result
        transformer_result = _ps.transformer_result
        dual_regime_result = _ps.dual_regime_result
        granger_result = _ps.granger_result
        ga_result = _ps.ga_result
        ohlc_result = _ps.ohlc_result
        regime_shift_result = _ps.regime_shift_result
        _synergy_meta = _ps.synergy_meta
        _pattern_drift = _ps.pattern_drift
        weights = _ps.weights
        regime_detector = _ps.regime_detector
        _economic_plane = _ps.economic_plane
        # Stage 7 results (if stage spec included them)
        scenario_result = _ps.scenario_result
        _retro_params = _ps.retro_params
        model_diagnostics_result = _ps.model_diagnostics_result
        multi_frequency_result = _ps.multi_frequency_result
        hf_result = _ps.hf_result
        _mode_weights = _ps.mode_weights
        _tv_granger_result = _ps.tv_granger_result
        _mv_mc_result = _ps.mv_mc_result

        logger.info("Staged execution complete -- continuing to output stages")

    elif not args.skip_models:
        logger.info("")
        logger.info("Step 6: Running temporal models...")

        from operator1.models.regime_detector import detect_regimes_and_breaks
        from operator1.models.forecasting import (
            run_forecasting,
            run_forward_pass,
            run_burnout,
        )
        from operator1.models.monte_carlo import run_monte_carlo
        from operator1.models.prediction_aggregator import run_prediction_aggregation

        # Collect injected feature columns for temporal model learning.
        # Include linked aggregate columns (competitors_avg_*, suppliers_median_*,
        # etc.) so that temporal models can learn from cross-entity signals.
        #
        # LOOK-AHEAD GUARD: survival_intensity, regime_confidence, and
        # regime_transition_prob are EXCLUDED because they are derived from
        # HMM regime labels fitted on the FULL 2-year cache (Step 5.5).
        # Including them would let the forward pass "see" future regime
        # information, violating the no-look-ahead principle (Spec J, 6.1).
        # The online_change_score (ChangeFinder) IS safe -- it uses only
        # past data for each score.  stability_score_21d is safe -- it
        # uses a backward-looking 21-day rolling window on rule-based flags.
        _linked_prefixes = (
            "competitors_", "suppliers_", "customers_",
            "financial_institutions_", "sector_peers_", "industry_peers_",
            "rel_", "valuation_premium_",
        )
        # Columns derived from full-cache HMM that would cause look-ahead
        # bias if fed as features to the forward pass temporal models.
        _hmm_lookahead_cols = {
            "survival_intensity",    # blends rule-based (clean) + HMM (look-ahead)
            "regime_confidence",     # from HMM posteriors fitted on full cache
            "regime_transition_prob", # from enriched timeline using HMM labels
        }
        _extra_vars = [
            c for c in cache.columns
            if (c.startswith("fh_") or c.startswith("sentiment_")
                or c.startswith("peer_") or c.startswith("macro_")
                or c.startswith("inst_")
                or c.startswith("buying_power_") or c.startswith("catalyst_")
                or c.startswith("conflict_") or c.startswith("demand_")
                or c.startswith("merton_") or c.startswith("rv_")
                or c.startswith("policy_risk_") or c.startswith("sector_leader_")
                or c.startswith("segment_") or c.startswith("product_")
                or c.startswith("pricing_") or c.startswith("margin_")
                or c.startswith("som_") or c.startswith("customer_")
                or c in ("stability_score_21d",
                         "buying_power_index", "sector_demand_momentum",
                         "catalyst_score", "online_change_score",
                         "iv30", "iv_rv_spread",
                         "days_to_next_event", "event_uncertainty_premium",
                         "fomc_proximity", "earnings_proximity",
                         "event_density_30d",
                         "sector_relative_strength", "sector_rank_12m",
                         "sector_dispersion", "yield_curve_10y2y",
                         "usd_momentum_21d", "cross_asset_stress",
                         "geo_hhi", "china_revenue_pct",
                         "supply_chain_geo_hhi", "trade_policy_uncertainty",
                         "tariff_exposure_score",
                         "put_call_ratio", "risk_reversal_25d",
                         "iv_skew", "vix_term_structure",
                         "skew_index", "variance_risk_premium",
                         "cannibalization_rate", "net_new_revenue_pct",
                         "network_effect_score", "input_cost_pressure",
                         "growth_runway_quarters", "maturity_concentration",
                         "estimated_market_share", "dominant_segment_growth",
                     # Raw macro indicators (Gap C)
                     "gdp_growth", "inflation_rate_yoy", "real_interest_rate",
                     "unemployment_rate", "official_exchange_rate_lcu_per_usd")
                or any(c.startswith(p) for p in _linked_prefixes)
                # Estimation confidence columns (Gap B)
                or c.endswith("_confidence")
                or c.startswith("interp_confidence_"))
            and cache[c].dtype in ("float64", "float32", "int64")
            and not c.startswith("is_missing_")
            and c not in _hmm_lookahead_cols
        ]
        # IC-based signal filtering: prune weak signals before temporal models
        if signal_ic_result is not None and signal_ic_result.available:
            try:
                _extra_vars = get_ic_weighted_signals(signal_ic_result, _extra_vars)
            except Exception as _ic_exc:
                logger.debug("IC signal filtering skipped: %s", _ic_exc)

        if _extra_vars:
            logger.info("Extra variables for temporal models (%d): %s", len(_extra_vars), _extra_vars[:10])

        # Item 4: Regime detection -- skip if already run in Step 5.5.
        # Check both that detector exists AND regime columns are in cache
        # to guard against partial state from a Step 5.5 exception.
        _regime_ready = (
            regime_detector is not None
            and hasattr(regime_detector, "result")
            and "regime_label" in cache.columns
        )
        if not _regime_ready:
            try:
                cache, regime_detector = detect_regimes_and_breaks(cache)
                logger.info("Regimes detected")
            except Exception as exc:
                logger.warning("Regime detection failed: %s", exc)
        else:
            logger.info("Regime detection: using results from Step 5.5 (early detection)")

        # Dual regime classification
        try:
            from operator1.models.regime_mixer import compute_dual_regimes
            from operator1.analysis.adaptive_thresholds import threshold_set_to_regime_dict
            _regime_thresholds = (
                threshold_set_to_regime_dict(_adaptive_thresholds)
                if _adaptive_thresholds is not None and _adaptive_thresholds.adapted
                else None
            )
            dual_regime_result = compute_dual_regimes(
                cache, thresholds=_regime_thresholds,
            )
            if dual_regime_result and dual_regime_result.fitted:
                logger.info("Dual regime classification complete")
        except Exception as exc:
            logger.warning("Dual regime classification failed: %s", exc)

        # Granger causality (informational -- pruning replaced by feature selection below)
        _gc_vars = []
        try:
            from operator1.models.granger_causality import (
                compute_granger_causality,
            )
            _gc_vars = [
                c for c in cache.columns
                if cache[c].dtype in ("float64", "float32")
                and cache[c].notna().sum() > 50
            ][:25]
            if len(_gc_vars) >= 3:
                granger_result = compute_granger_causality(
                    cache, variables=_gc_vars,
                )
                if granger_result and granger_result.fitted:
                    logger.info(
                        "Granger causality: %d significant pairs (informational, no pruning)",
                        len(granger_result.significant_pairs),
                    )
        except Exception as exc:
            logger.warning("Granger causality analysis failed: %s", exc)

        # Transfer entropy
        try:
            from operator1.models.causality import compute_transfer_entropy
            _te_vars = [c for c in cache.columns if cache[c].dtype in ("float64", "float32") and cache[c].notna().sum() > 30][:20]
            if len(_te_vars) >= 2:
                transfer_entropy_result = compute_transfer_entropy(cache, variables=_te_vars)
                logger.info("Transfer entropy computed: %d variable pairs", len(_te_vars) * (len(_te_vars) - 1))
        except Exception as exc:
            logger.warning("Transfer entropy failed: %s", exc)

        # Cycle decomposition
        # In private mode, use revenue or equity instead of close price.
        _cycle_var = "equity_value" if _is_private else "close"
        if _is_private and _cycle_var not in cache.columns:
            _cycle_var = "revenue" if "revenue" in cache.columns else "total_equity"
        try:
            from operator1.models.cycle_decomposition import run_cycle_decomposition
            cycle_result = run_cycle_decomposition(cache, variable=_cycle_var)
            logger.info("Cycle decomposition complete (variable=%s)", _cycle_var)
        except Exception as exc:
            logger.warning("Cycle decomposition failed: %s", exc)

        # Candlestick pattern detection
        try:
            from operator1.models.pattern_detector import detect_patterns
            pattern_result = detect_patterns(cache)
            logger.info("Candlestick patterns detected")
        except Exception as exc:
            logger.warning("Candlestick pattern detection failed: %s", exc)

        # Classify economic plane for plane-aware model weighting
        _economic_plane = None
        try:
            from operator1.analysis.economic_planes import classify_economic_plane
            _economic_plane = classify_economic_plane(
                sector=target_profile.get("sector"),
                industry=target_profile.get("industry"),
            )
        except Exception:
            pass

        # Pre-forecasting synergies (including plane-aware weighting)
        try:
            from operator1.models.model_synergies import apply_pre_forecasting_synergies
            cache, _extra_vars, _synergy_meta = apply_pre_forecasting_synergies(
                cache,
                cycle_result=cycle_result,
                granger_result=granger_result,
                transfer_entropy_result=transfer_entropy_result,
                peer_result=None,
                linked_caches=linked_caches or None,
                extra_variables=_extra_vars,
                economic_plane=_economic_plane,
            )
            logger.info("Pre-forecasting synergies applied")
        except Exception as exc:
            logger.warning("Pre-forecasting synergies failed: %s", exc)

        # Feature selection (Boruta + Regime-Conditional PIMP + mRMR)
        # Replaces the old Granger-based pruning with a 3-layer system that
        # handles non-linear effects and regime-conditional importance.
        feature_selection_result = None
        try:
            from operator1.models.feature_selector import run_feature_selection
            _fs_target = "equity_change_rate" if _is_private else "return_1d"
            _fs_regime = cache.get("regime_label") if "regime_label" in cache.columns else None
            _extra_vars, feature_selection_result = run_feature_selection(
                cache, _extra_vars, regime_labels=_fs_regime,
                target_col=_fs_target, granger_result=granger_result,
            )
            if feature_selection_result and feature_selection_result.fitted:
                logger.info(
                    "Feature selection: %d -> %d features (Boruta: %d, PIMP: %d, mRMR: %d)",
                    feature_selection_result.n_input,
                    feature_selection_result.n_output,
                    len(feature_selection_result.boruta_confirmed),
                    sum(len(v) for v in feature_selection_result.regime_selected.values()),
                    len(feature_selection_result.mrmr_selected),
                )
        except Exception as exc:
            logger.warning("Feature selection failed (keeping all features): %s", exc)

        # Standard forecasting
        try:
            cache, forecast_result = run_forecasting(
                cache,
                extra_variables=_extra_vars,
                windows=_adaptive_tier3.windows if _adaptive_tier3 is not None and _adaptive_tier3.adapted else None,
            )
            logger.info("Forecasting complete")
        except Exception as exc:
            logger.warning("Forecasting failed: %s", exc)

        # Forward pass
        regime_labels = cache.get("regime_label") if "regime_label" in cache.columns else None
        try:
            forward_pass_result = run_forward_pass(
                cache,
                hierarchy_weights=weights,
                regime_labels=regime_labels,
                extra_variables=_extra_vars,
            )
            logger.info("Forward pass complete: %d steps", forward_pass_result.total_days)
        except Exception as exc:
            import traceback as _tb
            logger.warning("Forward pass failed: %s\n%s", exc, _tb.format_exc())

        # Burn-out (weight calibration via exponential gradient learning)
        try:
            burnout_result = run_burnout(
                cache,
                hierarchy_weights=weights,
                regime_labels=regime_labels,
                extra_variables=_extra_vars,
                forward_pass_result=forward_pass_result,
            )
            logger.info(
                "Burn-out complete: %d iterations, converged=%s, calibrated=%s",
                burnout_result.iterations_completed,
                burnout_result.converged,
                getattr(burnout_result, "calibrated", False),
            )
            if getattr(burnout_result, "regime_weights", None):
                for regime, wts in burnout_result.regime_weights.items():
                    logger.info(
                        "  Regime '%s' weights: %s",
                        regime,
                        {k: f"{v:.3f}" for k, v in wts.items()},
                    )
        except Exception as exc:
            logger.warning("Burn-out failed: %s", exc)

        # Walk-forward evaluation (produces WalkForwardResult for recency-weighted RMSE)
        try:
            from operator1.models.walk_forward import run_walk_forward
            from operator1.analysis.survival_timeline import compute_survival_timeline
            _wf_timeline_result = compute_survival_timeline(cache)
            # Pass the .timeline DataFrame (not the result wrapper) -- walk_forward
            # calls len() on it, and SurvivalTimelineResult has no __len__.
            _wf_timeline_df = (
                _wf_timeline_result.timeline
                if hasattr(_wf_timeline_result, "timeline")
                else _wf_timeline_result
            )
            # Extract individual Series for survival_modes and switch_points.
            # run_walk_forward expects pd.Series (not a DataFrame) for these
            # parameters -- passing the full DataFrame would produce garbage
            # mode labels (str(row) instead of str(value)) and break
            # mode-conditioned scoring.
            _wf_modes = (
                _wf_timeline_df["survival_mode"]
                if isinstance(_wf_timeline_df, pd.DataFrame)
                and "survival_mode" in _wf_timeline_df.columns
                else None
            )
            _wf_switches = (
                _wf_timeline_df["switch_point"]
                if isinstance(_wf_timeline_df, pd.DataFrame)
                and "switch_point" in _wf_timeline_df.columns
                else None
            )
            walk_forward_result = run_walk_forward(
                cache, _wf_modes, _wf_switches,
            )
            if walk_forward_result and walk_forward_result.fitted:
                logger.info(
                    "Walk-forward: %d days evaluated, best=%s (MAE=%.6f)",
                    walk_forward_result.total_days_evaluated,
                    walk_forward_result.overall_best_model,
                    walk_forward_result.overall_mae
                    if not pd.isna(walk_forward_result.overall_mae) else 0.0,
                )
        except Exception as exc:
            logger.warning("Walk-forward evaluation failed: %s", exc)

        # Forward pass error aggregation + MCS + Fixed Share (Part 2)
        _mode_confidence_sets = None
        _fixed_share = None
        try:
            if forward_pass_result is not None and hasattr(forward_pass_result, "predictions_log"):
                from operator1.models.walk_forward import (
                    aggregate_forward_pass_errors,
                    compute_mode_confidence_sets,
                )
                from operator1.models.prediction_aggregator import FixedShareForecaster

                _fp_log = getattr(forward_pass_result, "predictions_log", [])
                if _fp_log:
                    _mode_errors = aggregate_forward_pass_errors(_fp_log, cache)
                    if _mode_errors:
                        _mode_confidence_sets = compute_mode_confidence_sets(_mode_errors)
                        logger.info(
                            "Mode confidence sets: %s",
                            {m: len(v) for m, v in _mode_confidence_sets.items()},
                        )

                    # Initialize Fixed Share with all model names from forward pass
                    _all_model_names = set()
                    for mode_models in _mode_errors.values():
                        _all_model_names.update(mode_models.keys())
                    if _all_model_names:
                        _fixed_share = FixedShareForecaster(sorted(_all_model_names))
                        # Feed historical errors to warm up weights
                        for mode_models in _mode_errors.values():
                            _min_len = min(len(v) for v in mode_models.values()) if mode_models else 0
                            for step in range(min(_min_len, 50)):
                                step_losses = {
                                    name: errs[step]
                                    for name, errs in mode_models.items()
                                    if step < len(errs)
                                }
                                _fixed_share.update(step_losses)
                        logger.info(
                            "Fixed Share weights: %s",
                            {k: f"{v:.3f}" for k, v in _fixed_share.get_weights().items()},
                        )
        except Exception as exc:
            logger.debug("Mode error aggregation / Fixed Share failed: %s", exc)

        # Derive mode_weights from Fixed Share for prediction aggregator
        _mode_weights = None
        if _fixed_share is not None:
            try:
                _mode_weights = {"global": _fixed_share.get_weights()}
            except Exception:
                pass

        # Monte Carlo
        # In private mode, use equity_change_rate instead of return_1d.
        try:
            _mc_returns = "equity_change_rate" if _is_private else "return_1d"
            _mc_thresholds = (
                threshold_set_to_mc_dict(_adaptive_thresholds)
                if _adaptive_thresholds is not None and _adaptive_thresholds.adapted
                else None
            )
            _mc_n = (
                _adaptive_model_params.mc_n_paths
                if _adaptive_model_params is not None and _adaptive_model_params.adapted
                else 10_000
            )
            _mc_tilt = (
                _adaptive_model_params.mc_is_tilt
                if _adaptive_model_params is not None and _adaptive_model_params.adapted
                else 1.5
            )
            # Pass burn-out calibrated distributions when available
            _burnout_dists = (
                burnout_result.regime_distributions
                if burnout_result is not None and getattr(burnout_result, "calibrated", False)
                else None
            )
            mc_result = run_monte_carlo(
                cache, returns_col=_mc_returns,
                n_paths=_mc_n,
                importance_tilt=_mc_tilt,
                survival_thresholds=_mc_thresholds,
                burnout_distributions=_burnout_dists,
            )
            logger.info("Monte Carlo simulation complete")

            # Set product concentration risk flag on MC result
            if mc_result is not None and "segment_hhi" in cache.columns:
                _seg_hhi = float(cache["segment_hhi"].iloc[-1]) if cache["segment_hhi"].notna().any() else 0
                mc_result.segment_hhi = _seg_hhi
                mc_result.concentration_risk_flag = _seg_hhi > 0.5

            # E2: Forward-looking (path-wise) survival trigger checking.
            # Computes fraction of MC paths that trigger ANY survival condition
            # at ANY point along the path (not just terminal). Standard in
            # credit risk as 'first-passage-time' but novel in equity.
            try:
                from operator1.models.monte_carlo import compute_anticipated_survival
                for _as_horizon_label, _as_horizon_days in [("63d", 63), ("252d", 252)]:
                    _as_prob = compute_anticipated_survival(
                        cache, mc_result, horizon_days=_as_horizon_days,
                    )
                    if mc_result is not None:
                        mc_result.anticipated_survival[_as_horizon_label] = _as_prob
                if mc_result is not None and mc_result.anticipated_survival:
                    logger.info(
                        "E2 anticipated survival: %s",
                        {k: f"{v:.1%}" for k, v in mc_result.anticipated_survival.items()},
                    )
            except Exception as _as_exc:
                logger.debug("E2 anticipated survival skipped: %s", _as_exc)
        except Exception as exc:
            logger.warning("Monte Carlo failed: %s", exc)

        # Predicted regime shifts (Section E.5 from core idea)
        # Uses the HMM transition matrix from MC to predict when the
        # current regime is likely to change and to which regime.
        regime_shift_result = None
        try:
            from operator1.models.regime_shift_predictor import predict_regime_shifts
            _mc_transition = mc_result.transition_matrix if mc_result is not None else None
            _mc_regime_order = None
            if mc_result is not None and hasattr(mc_result, "regime_order"):
                _mc_regime_order = mc_result.regime_order
            _stab_score = None
            if "stability_score_21d" in cache.columns:
                _ss = cache["stability_score_21d"].dropna()
                if len(_ss) > 0:
                    _stab_score = float(_ss.iloc[-1])
            _trans_hl = (
                _adaptive_model_params.transition_halflife
                if _adaptive_model_params is not None and _adaptive_model_params.adapted
                else None
            )
            regime_shift_result = predict_regime_shifts(
                cache,
                transition_matrix=_mc_transition,
                regime_order=_mc_regime_order,
                stability_score=_stab_score,
                transition_halflife=_trans_hl,
                reference_date=_backtest_end_date,
            )
            if regime_shift_result and regime_shift_result.available:
                logger.info(
                    "Regime shift prediction: P(exit 21d)=%.1f%%, P(exit 252d)=%.1f%%, "
                    "expected_days=%.0f, next=%s",
                    regime_shift_result.prob_exit_21d * 100,
                    regime_shift_result.prob_exit_252d * 100,
                    regime_shift_result.expected_days_to_shift,
                    regime_shift_result.most_probable_next_regime,
                )
        except Exception as exc:
            logger.warning("Regime shift prediction failed: %s", exc)

        # Copula
        try:
            from operator1.models.copula import run_copula_analysis
            copula_result = run_copula_analysis(cache)
            logger.info("Copula analysis complete")
        except Exception as exc:
            logger.warning("Copula analysis failed: %s", exc)

        # Transformer forecaster
        try:
            from operator1.models.transformer_forecaster import train_transformer
            _tf_vars = [c for c in cache.columns
                        if cache[c].dtype in ("float64", "float32")
                        and cache[c].notna().sum() > 100][:15]
            if len(_tf_vars) >= 2:
                transformer_result = train_transformer(cache, variables=_tf_vars)
                if (
                    transformer_result
                    and transformer_result.fitted
                    and forecast_result is not None
                    and transformer_result.forecasts
                ):
                    from operator1.models.forecasting import ModelMetrics
                    for var, val in transformer_result.forecasts.items():
                        forecast_result.forecasts.setdefault(var, {})
                        if "1d" not in forecast_result.forecasts[var]:
                            forecast_result.forecasts[var]["1d"] = val
                        forecast_result.metrics.append(
                            ModelMetrics(
                                model_name="transformer",
                                variable=var,
                                rmse=transformer_result.final_train_loss
                                if transformer_result.final_train_loss > 0
                                else 0.01,
                                fitted=True,
                            )
                        )
                    for var in transformer_result.forecasts:
                        forecast_result.model_used.setdefault(var, "transformer")
                    logger.info("Transformer forecasts injected: %d vars", len(transformer_result.forecasts))
        except Exception as exc:
            logger.warning("Transformer forecaster failed: %s", exc)

        # Particle Filter
        try:
            from operator1.models.particle_filter import run_particle_filter
            _pf_vars = [v for v in ["cash_ratio", "free_cash_flow_ttm", "current_ratio", "debt_to_equity"]
                        if v in cache.columns]
            if _pf_vars:
                particle_filter_result = run_particle_filter(cache, variables=_pf_vars)
                logger.info("Particle filter complete")
        except Exception as exc:
            logger.warning("Particle filter failed: %s", exc)

        # Conformal prediction (before aggregation so results feed in)
        # Prefer ConformalPIDCalibrator (PID-controlled + Mondrian partitioning)
        # with fallback to standard ConformalCalibrator.
        try:
            from operator1.models.conformal import ConformalPIDCalibrator, ConformalCalibrator, build_conformal_result
            if forecast_result is not None:
                # Prefer the forward pass calibrator which has per-variable
                # per-survival-mode scores (Mondrian partitioning).
                calibrator = None
                if forward_pass_result is not None and hasattr(forward_pass_result, "conformal_calibrator") and forward_pass_result.conformal_calibrator is not None:
                    calibrator = forward_pass_result.conformal_calibrator
                    logger.info("Reusing forward pass conformal calibrator (per-variable per-mode scores)")
                else:
                    try:
                        calibrator = ConformalPIDCalibrator(target_coverage=0.9)
                        logger.info("Using new ConformalPIDCalibrator (PID + Mondrian)")
                    except Exception:
                        calibrator = ConformalCalibrator(coverage=0.9, adaptive=True)
                    if hasattr(forecast_result, "residuals") and forecast_result.residuals is not None:
                        for r in forecast_result.residuals:
                            calibrator.update(r)
                # build_conformal_result expects nested dict:
                # {variable: {horizon_label: point_forecast}}
                # Use forecast_result.forecasts (available now) instead of
                # pred_result.predictions (set later by run_prediction_aggregation).
                _nested_forecasts: dict[str, dict[str, float]] = {}
                if forecast_result is not None and hasattr(forecast_result, "forecasts"):
                    for var, var_forecasts in forecast_result.forecasts.items():
                        if isinstance(var_forecasts, dict):
                            _nested_forecasts[var] = {}
                            for h, val in var_forecasts.items():
                                try:
                                    _nested_forecasts[var][h] = float(val)
                                except (TypeError, ValueError):
                                    pass
                            if not _nested_forecasts[var]:
                                del _nested_forecasts[var]
                # Compute regime transition probability for interval widening
                _conf_trans_prob = None
                _conf_vol_ratio = None
                if regime_detector is not None and "regime_label" in cache.columns:
                    try:
                        _rl = cache["regime_label"].dropna()
                        if len(_rl) >= 2:
                            _current = str(_rl.iloc[-1])
                            _transitions = sum(
                                1 for i in range(max(0, len(_rl) - 63), len(_rl) - 1)
                                if str(_rl.iloc[i]) != str(_rl.iloc[i + 1])
                            )
                            _conf_trans_prob = min(1.0, _transitions / 63.0)
                        if "return_1d" in cache.columns:
                            _regime_vols = cache.groupby("regime_label")["return_1d"].std()
                            if len(_regime_vols) >= 2:
                                _conf_vol_ratio = float(_regime_vols.max() / max(_regime_vols.min(), 1e-8))
                    except Exception:
                        pass
                # Extract event uncertainty premium from cache for interval widening
                _conf_event_premium = None
                if "event_uncertainty_premium" in cache.columns:
                    _eup = cache["event_uncertainty_premium"].dropna()
                    if len(_eup) > 0:
                        _conf_event_premium = float(_eup.iloc[-1])
                # Extract geographic concentration for interval widening
                _conf_geo_hhi = None
                if "geo_hhi" in cache.columns:
                    _gh = cache["geo_hhi"].dropna()
                    if len(_gh) > 0:
                        _conf_geo_hhi = float(_gh.iloc[-1])

                conformal_result = build_conformal_result(
                    calibrator,
                    forecasts=_nested_forecasts,
                    horizons={"1d": 1, "5d": 5, "21d": 21, "252d": 252},
                    regime_transition_prob=_conf_trans_prob,
                    regime_vol_ratio=_conf_vol_ratio,
                    event_uncertainty_premium=_conf_event_premium,
                    geo_concentration_hhi=_conf_geo_hhi,
                )
                logger.info("Conformal prediction intervals computed")

                # G1: Quantile Regression for asymmetric prediction intervals.
                # Augments conformal intervals with regime-conditioned asymmetry
                # (crisis = wider downside, bull = wider upside).
                try:
                    from operator1.models.conformal import QuantileRegressionCalibrator
                    _qr_cal = QuantileRegressionCalibrator(lower_quantile=0.05, upper_quantile=0.95)
                    _residuals_list = []
                    if hasattr(forecast_result, "residuals") and forecast_result.residuals is not None:
                        _residuals_list = list(forecast_result.residuals)
                    if len(_residuals_list) >= _qr_cal._min_samples:
                        _qr_fitted = _qr_cal.fit(_residuals_list)
                        if _qr_fitted:
                            logger.info("G1 Quantile Regression calibrator fitted (%d residuals)", len(_residuals_list))
                            # Produce asymmetric intervals for each variable
                            if conformal_result is not None and hasattr(conformal_result, "intervals"):
                                _n_asym = 0
                                for var, horizons_dict in conformal_result.intervals.items():
                                    if isinstance(horizons_dict, dict):
                                        for h, interval in horizons_dict.items():
                                            pf = getattr(interval, "point_forecast", None) or getattr(interval, "forecast", None)
                                            if pf is not None:
                                                _lo, _hi = _qr_cal.predict_interval(float(pf))
                                                if _lo is not None and _hi is not None:
                                                    if hasattr(interval, "lower"):
                                                        interval.lower = _lo
                                                    if hasattr(interval, "upper"):
                                                        interval.upper = _hi
                                                    _n_asym += 1
                                if _n_asym > 0:
                                    logger.info("G1 asymmetric intervals applied to %d predictions", _n_asym)
                except Exception as _qr_exc:
                    logger.debug("G1 Quantile Regression skipped: %s", _qr_exc)
        except Exception as exc:
            logger.warning("Conformal prediction failed: %s", exc)

        # DTW analogs (before aggregation so results feed in)
        try:
            from operator1.models.dtw_analogs import find_historical_analogs
            _dtw_vars = None
            if _is_private:
                _dtw_vars = [c for c in ["equity_value", "revenue", "net_income",
                             "total_debt", "operating_cash_flow"]
                             if c in cache.columns and cache[c].notna().sum() > 30]
            _dtw_catalyst = (
                catalyst_result.catalyst_score
                if catalyst_result is not None and catalyst_result.available
                else None
            )
            dtw_result = find_historical_analogs(
                cache, variables=_dtw_vars,
                linked_caches=linked_caches if linked_caches else None,
                catalyst_score=_dtw_catalyst,
            )
            logger.info("DTW analogs complete")
        except Exception as exc:
            logger.warning("DTW historical analogs failed: %s", exc)

        # Prediction aggregation (now receives conformal + DTW results)
        # Merge burn-out regime weights into mode_weights (primary source)
        if (
            burnout_result is not None
            and getattr(burnout_result, "calibrated", False)
            and burnout_result.regime_weights
        ):
            if _mode_weights is None:
                _mode_weights = {}
            # Burn-out regime weights override FixedShare per-regime weights
            for regime, model_weights in burnout_result.regime_weights.items():
                _mode_weights[regime] = model_weights
            logger.info(
                "Prediction aggregator: using burn-out calibrated weights for %d regimes",
                len(burnout_result.regime_weights),
            )

        if forecast_result is not None:
            try:
                pred_result = run_prediction_aggregation(
                    cache, forecast_result, mc_result,
                    mode_weights=_mode_weights,
                    signal_ic_result=signal_ic_result,
                    prediction_log_summary=prediction_log_summary,
                    conformal_result=conformal_result,
                    dual_regime_result=dual_regime_result,
                    copula_result=copula_result,
                    dtw_result=dtw_result,
                    granger_result=granger_result,
                    shap_result=shap_result,
                    walk_forward_result=walk_forward_result,
                    feature_selection_result=feature_selection_result,
                    event_calendar_result=event_calendar_result,
                )
                logger.info("Predictions aggregated (with %d sibling module results)",
                    sum(1 for r in [conformal_result, dual_regime_result,
                        copula_result, dtw_result, granger_result,
                        shap_result, forward_pass_result] if r is not None)
                )
            except Exception as exc:
                logger.warning("Prediction aggregation failed: %s", exc)

        # USS: Bound aggregated predictions (not just raw forecasts)
        if (survival_controller is not None
                and survival_controller.is_survival
                and pred_result is not None
                and hasattr(pred_result, "predictions")):
            try:
                from operator1.analysis.survival_regime_controller import bound_survival_forecast
                _n_bounded = 0
                for var, horizons_dict in pred_result.predictions.items():
                    if isinstance(horizons_dict, dict):
                        for h, hp in horizons_dict.items():
                            pf = getattr(hp, "point_forecast", None)
                            if pf is not None:
                                bounded = bound_survival_forecast(
                                    var, float(pf), cache,
                                    survival_controller.current_regime,
                                )
                                if bounded != float(pf):
                                    hp.point_forecast = bounded
                                    _n_bounded += 1
                if _n_bounded > 0:
                    logger.info(
                        "USS: bounded %d aggregated prediction points", _n_bounded,
                    )
            except Exception as exc:
                logger.debug("USS aggregated prediction bounding failed: %s", exc)

        # SHAP explainability (after aggregation -- needs pred_result)
        try:
            from operator1.models.explainability import compute_shap_explanations
            if pred_result is not None:
                _shap_preds: dict[str, float] = {}
                if hasattr(pred_result, "predictions"):
                    for var, horizons_dict in pred_result.predictions.items():
                        if isinstance(horizons_dict, dict):
                            hp_1d = horizons_dict.get("1d")
                            if hp_1d is not None:
                                pf = getattr(hp_1d, "point_forecast", None)
                                if pf is not None:
                                    _shap_preds[var] = pf
                # W9 fix: Extract predict functions from forward pass model states
                # so SHAP can generate actual explanations instead of returning empty.
                _shap_predict_fns: dict[str, Any] = {}
                if forward_pass_result is not None and hasattr(forward_pass_result, "model_states"):
                    for var, wrapper in forward_pass_result.model_states.items():
                        if hasattr(wrapper, "predict"):
                            _shap_predict_fns[var] = wrapper.predict
                shap_result = compute_shap_explanations(
                    cache,
                    predictions=_shap_preds,
                    predict_fns=_shap_predict_fns if _shap_predict_fns else None,
                )
                logger.info("SHAP explanations computed")
        except Exception as exc:
            logger.warning("SHAP explainability failed: %s", exc)

        # Sobol sensitivity
        try:
            from operator1.models.sensitivity import run_sensitivity_analysis
            _sobol_target = "equity_change_rate" if _is_private else "return_1d"
            sobol_result = run_sensitivity_analysis(cache, target_variable=_sobol_target)
            logger.info("Sobol sensitivity analysis complete")
        except Exception as exc:
            logger.warning("Sobol sensitivity failed: %s", exc)

        # Sobol -> Hierarchy feedback loop (Proposal 1.5)
        # Adjusts hierarchy weights toward data-driven Sobol importance
        try:
            from operator1.models.sensitivity import adjust_hierarchy_from_sobol
            _adjusted_weights = adjust_hierarchy_from_sobol(sobol_result, weights)
            if _adjusted_weights != weights:
                weights = _adjusted_weights
                logger.info("Hierarchy weights adjusted from Sobol: %s", weights)
        except Exception as exc:
            logger.debug("Sobol hierarchy feedback skipped: %s", exc)

        # Time-varying Granger causality (Proposal 3.5)
        _tv_granger_result = None
        try:
            from operator1.models.granger_causality import compute_time_varying_granger
            _tv_granger_result = compute_time_varying_granger(cache, variables=_gc_vars[:15] if _gc_vars else None)
            if _tv_granger_result and _tv_granger_result.get("emerging_pairs"):
                logger.info(
                    "Time-varying Granger: %d windows, %d emerging, %d disappearing",
                    _tv_granger_result.get("n_windows", 0),
                    len(_tv_granger_result.get("emerging_pairs", [])),
                    len(_tv_granger_result.get("disappearing_pairs", [])),
                )
        except Exception as exc:
            logger.debug("Time-varying Granger failed: %s", exc)

        # Multivariate Monte Carlo (Proposal 3.3)
        _mv_mc_result = None
        try:
            from operator1.models.monte_carlo import run_multivariate_monte_carlo
            _copula_corr = None
            if copula_result is not None and hasattr(copula_result, "copula_correlation"):
                # Extract correlation matrix from dict format
                _cop_vars = list(copula_result.copula_correlation.keys())
                if _cop_vars:
                    import numpy as _np
                    _copula_corr = _np.array([
                        [copula_result.copula_correlation[vi].get(vj, 0.0) for vj in _cop_vars]
                        for vi in _cop_vars
                    ])
            _mv_mc_result = run_multivariate_monte_carlo(
                cache, copula_correlation=_copula_corr,
            )
            if _mv_mc_result and _mv_mc_result.get("available"):
                logger.info(
                    "Multivariate MC: survival=%.4f, vars=%s",
                    _mv_mc_result.get("survival_probability", 0),
                    _mv_mc_result.get("variables_simulated", []),
                )
        except Exception as exc:
            logger.debug("Multivariate Monte Carlo failed: %s", exc)

        # Genetic Algorithm
        try:
            from operator1.models.genetic_optimizer import run_genetic_optimization
            ga_result = run_genetic_optimization(
                cache,
                forecast_result=forecast_result,
            )
            if ga_result and ga_result.fitted:
                logger.info("GA optimization complete")
        except Exception as exc:
            logger.warning("Genetic algorithm optimization failed: %s", exc)

        # OHLC candlestick prediction
        try:
            from operator1.models.ohlc_predictor import predict_ohlc_series
            from operator1.models.model_synergies import compute_pattern_drift_adjustment
            _pattern_drift = compute_pattern_drift_adjustment(pattern_result)
            ohlc_result = predict_ohlc_series(
                cache,
                forecast_result=forecast_result,
                mc_result=mc_result,
                pattern_drift_multiplier=_pattern_drift,
                cycle_result=cycle_result,
            )
            if ohlc_result and ohlc_result.fitted:
                logger.info("OHLC prediction complete")
                # Run pattern detection on predicted OHLC candles
                # (App core idea Section E.5: specific dated pattern formations)
                try:
                    from operator1.models.pattern_detector import detect_patterns_on_predicted_ohlc
                    _last_candle = None
                    if "close" in cache.columns and "open" in cache.columns:
                        _last_candle = {
                            "open": float(cache["open"].iloc[-1]) if cache["open"].notna().any() else None,
                            "high": float(cache["high"].iloc[-1]) if "high" in cache.columns and cache["high"].notna().any() else None,
                            "low": float(cache["low"].iloc[-1]) if "low" in cache.columns and cache["low"].notna().any() else None,
                            "close": float(cache["close"].iloc[-1]) if cache["close"].notna().any() else None,
                        }
                    _pred_patterns = detect_patterns_on_predicted_ohlc(ohlc_result, _last_candle)
                    if _pred_patterns and pattern_result is not None:
                        pattern_result.predicted_patterns_week = _pred_patterns
                        logger.info("Predicted OHLC patterns: %d formations detected", len(_pred_patterns))
                except Exception as _pp_exc:
                    logger.debug("Predicted OHLC pattern detection failed: %s", _pp_exc)
        except Exception as exc:
            logger.warning("OHLC candlestick prediction failed: %s", exc)

        # Recursive day-by-day predictions (sub-stage 6.11)
        recursive_result = None
        try:
            from operator1.models.recursive_aggregator import run_recursive_predictions
            if forward_pass_result is not None and hasattr(forward_pass_result, "model_states") and forward_pass_result.model_states:
                _rc_transition_matrix = None
                _rc_regime_order = None
                if mc_result is not None:
                    _rc_transition_matrix = getattr(mc_result, "transition_matrix", None)
                    _rc_regime_order = getattr(mc_result, "regime_order", None)
                recursive_result = run_recursive_predictions(
                    cache=cache,
                    model_states=forward_pass_result.model_states,
                    transition_matrix=_rc_transition_matrix,
                    regime_order=_rc_regime_order,
                )
                if recursive_result and recursive_result.available:
                    logger.info("Recursive predictions complete: %d steps, %d snapshots",
                               recursive_result.total_steps, len(recursive_result.snapshots))
        except Exception as exc:
            logger.warning("Recursive day-by-day predictions failed: %s", exc)
    else:
        logger.info("Step 6: Skipped (--skip-models)")
        # regime_detector may have been set in Step 5.5; keep it if so.

    # ------------------------------------------------------------------
    # Step 6-USS: Unified Survival System -- Post-model integration
    # ------------------------------------------------------------------
    # Apply forecast bounding (Dimension 5) and run scenario engine
    # when survival mode is active.
    # SKIP when --stage was used: the staged runner (Stage 7) already
    # handled USS, retro calibration, diagnostics, multi-freq, and HF.
    if args.stage:
        logger.info("Steps 6-USS through 6-HF: handled by staged runner")
    elif survival_controller is not None and not args.skip_models:
        # Forecast bounding: apply hard bounds to survival-mode forecasts
        if survival_controller.is_survival and forecast_result is not None:
            try:
                from operator1.analysis.survival_regime_controller import bound_forecast_dict
                if hasattr(forecast_result, "forecasts") and forecast_result.forecasts:
                    forecast_result.forecasts = bound_forecast_dict(
                        forecast_result.forecasts, cache, survival_controller.current_regime,
                    )
                    logger.info(
                        "USS forecast bounding applied (regime=%s)",
                        survival_controller.current_regime,
                    )
            except Exception as exc:
                logger.debug("USS forecast bounding failed: %s", exc)

        # Scenario engine: 3-scenario MC simulation for survival mode
        if survival_controller.is_survival:
            try:
                from operator1.analysis.scenario_engine import run_scenario_engine
                scenario_result = run_scenario_engine(
                    cache,
                    regime=survival_controller.current_regime,
                    n_paths=survival_controller.model_config.mc_n_paths,
                )
                if scenario_result and scenario_result.available:
                    logger.info(
                        "Scenario engine: orderly=%.1f%% / muddle=%.1f%% / catastrophic=%.1f%% (252d survival)",
                        scenario_result.orderly.survival_prob_252d * 100,
                        scenario_result.muddle_through.survival_prob_252d * 100,
                        scenario_result.catastrophic.survival_prob_252d * 100,
                    )
            except Exception as exc:
                logger.warning("Scenario engine failed: %s", exc)
        else:
            # Not in survival but check early warning
            if survival_controller.is_approaching_survival():
                logger.warning(
                    "USS early warning: score=%.2f -- approaching survival triggers",
                    survival_controller.get_early_warning_latest(),
                )

    # ------------------------------------------------------------------
    # Step 6.5: Retroactive calibration (Category D)
    # ------------------------------------------------------------------
    # After all temporal models have run, use their outputs to calibrate
    # model weight matrices that were initially set to fixed defaults.
    # Empirical Bayes: use first-pass data to set second-pass priors.
    _retro_params = None
    if not args.skip_models and not args.stage:
        try:
            from operator1.analysis.retroactive_calibration import run_retroactive_calibration

            _entity_groups_for_retro = {}
            if relationships:
                for grp, ents in relationships.items():
                    if isinstance(ents, list):
                        ids = []
                        for e in ents:
                            eid = ""
                            if isinstance(e, dict):
                                eid = e.get("isin", "") or e.get("ticker", "")
                            elif hasattr(e, "isin"):
                                eid = e.isin or getattr(e, "ticker", "")
                            if eid:
                                ids.append(eid)
                        _entity_groups_for_retro[grp] = ids

            _retro_params = run_retroactive_calibration(
                cache=cache,
                linked_caches=linked_caches if linked_caches else None,
                entity_groups=_entity_groups_for_retro if _entity_groups_for_retro else None,
                walk_forward_result=walk_forward_result,
                forecast_result=forecast_result,
                sobol_result=sobol_result,
                target_profile=target_profile,
            )
            if _retro_params.n_calibrated > 0:
                logger.info(
                    "Step 6.5: Retroactive calibration complete (%d groups calibrated)",
                    _retro_params.n_calibrated,
                )
        except Exception as exc:
            logger.warning("Retroactive calibration failed: %s", exc)

    # ------------------------------------------------------------------
    # Step 6.6: Model expected path vs actual path diagnostics
    # ------------------------------------------------------------------
    # For each model, pre-computes what it SHOULD produce based on data
    # characteristics, then compares against what it actually produced.
    # Produces per-model robustness ratings for the profile and report.
    model_diagnostics_result = None
    try:
        from operator1.monitoring.model_diagnostics import compute_model_diagnostics
        model_diagnostics_result = compute_model_diagnostics(
            cache,
            forecast_result=forecast_result,
            mc_result=mc_result,
            copula_result=copula_result,
            granger_result=granger_result,
            cycle_result=cycle_result,
            dtw_result=dtw_result,
            conformal_result=conformal_result,
        )
        if model_diagnostics_result and model_diagnostics_result.available:
            logger.info(
                "Model diagnostics: %d/%d on track, overall=%s",
                model_diagnostics_result.n_models_on_track,
                model_diagnostics_result.n_models_assessed,
                model_diagnostics_result.overall_robustness,
            )
    except Exception as exc:
        logger.debug("Model diagnostics failed: %s", exc)

    # ------------------------------------------------------------------
    # Step 6.7: Multi-frequency forecasting (sequential, slow-to-fast)
    # ------------------------------------------------------------------
    # Runs the full analytical pipeline at 5 frequencies (Annual -> Daily)
    # with cascading context. Each slower frequency's insights constrain
    # the next faster frequency's predictions.
    multi_frequency_result = None
    if not args.skip_models and not args.stage:
        try:
            from operator1.steps.multi_frequency_runner import run_multi_frequency_pipeline
            from operator1.models.frequency_fusion import fuse_multi_frequency_results

            logger.info("")
            logger.info("Step 6.7: Multi-frequency forecasting (5 frequencies)...")

            multi_frequency_result = run_multi_frequency_pipeline(
                daily_cache=cache,
                secrets=secrets,
                market_id=market_id,
                ticker=ticker,
                reference_date=_backtest_end_date,
                skip_models=False,
                income_df=income_df,
                balance_df=balance_df,
                cashflow_df=cashflow_df,
                quotes_df=quotes_df,
            )

            # Fuse results across all frequencies
            if multi_frequency_result and multi_frequency_result.results:
                multi_frequency_result = fuse_multi_frequency_results(multi_frequency_result)
                logger.info(
                    "Multi-frequency fusion: %d frequencies, regime=%s (%.0f%% agreement), "
                    "survival=%.1f%%",
                    multi_frequency_result.n_frequencies_used,
                    multi_frequency_result.regime_consensus.consensus_regime,
                    multi_frequency_result.regime_consensus.agreement_ratio * 100,
                    multi_frequency_result.survival.fused_probability * 100,
                )
        except Exception as exc:
            logger.warning("Multi-frequency pipeline failed: %s", exc)

    # ------------------------------------------------------------------
    # Step 6-HF: Hedge Fund Normal Analysis
    # ------------------------------------------------------------------
    # Parallel analytical track: 15 investment-grade metrics across 5 tiers.
    # Reads from raw statement DFs (not the daily cache) for most metrics.
    hf_result = None
    if not args.skip_models and not args.stage:
        try:
            from operator1.hedge_fund.engine import run_hedge_fund_analysis

            hf_result = run_hedge_fund_analysis(
                income_df=income_df,
                balance_df=balance_df,
                cashflow_df=cashflow_df,
                cache=cache,
                target_profile=target_profile,
                forecast_result=forecast_result,
                mc_result=mc_result,
                scenario_result=scenario_result,
                multi_frequency_result=multi_frequency_result,
                signal_ic_result=signal_ic_result,
                filing_calendar_result=filing_calendar_result,
                fh_result=fh_result,
                peer_ranking_result=peer_ranking_result,
                sentiment_result=sentiment_result,
                survival_controller=survival_controller,
                linked_caches=linked_caches,
                macro_data=macro_data,
            )
        except Exception as exc:
            logger.warning("Hedge Fund Analysis failed: %s", exc)

    # ------------------------------------------------------------------
    # Step 7: Build company profile
    # ------------------------------------------------------------------
    # Item 5: When --skip-models is used, the following variables are None:
    #   enriched_timeline_result, early_regime_result, regime_detector,
    #   forecast_result, forward_pass_result, burnout_result, mc_result,
    #   pred_result, transfer_entropy_result, cycle_result, pattern_result,
    #   copula_result, conformal_result, dtw_result, shap_result,
    #   sobol_result, particle_filter_result, transformer_result,
    #   dual_regime_result, granger_result, ga_result, ohlc_result.
    # All downstream code must check for None before accessing these.
    logger.info("")
    logger.info("Step 7: Building company profile...")

    from operator1.report.profile_builder import build_company_profile
    from dataclasses import asdict as _asdict

    def _safe_float(val) -> float | None:
        """Convert to JSON-safe float (None for NaN/Inf/None)."""
        import math as _math
        if val is None:
            return None
        try:
            f = float(val)
            if _math.isnan(f) or _math.isinf(f):
                return None
            return round(f, 6)
        except (TypeError, ValueError):
            return None

    def _to_dict(obj):
        """Convert a dataclass or dict-like object to a plain dict."""
        if obj is None:
            return None
        if isinstance(obj, dict):
            return obj
        if hasattr(obj, "__dataclass_fields__"):
            try:
                return _asdict(obj)
            except Exception:
                pass
        if hasattr(obj, "__dict__"):
            return obj.__dict__.copy()
        return None

    def _available_dict(obj):
        d = _to_dict(obj)
        if d is not None:
            d.setdefault("available", True)
        return d

    fh_dict = _to_dict(fh_result)

    # Build regime result dict
    regime_result_dict = None
    if not args.skip_models and regime_detector is not None:
        try:
            _rd = regime_detector.result
            regime_result_dict = {
                "hmm_fitted": _rd.hmm_fitted,
                "gmm_fitted": _rd.gmm_fitted,
                "pelt_fitted": _rd.pelt_fitted,
                "bcp_fitted": _rd.bcp_fitted,
                "hmm_error": _rd.hmm_error,
                "gmm_error": _rd.gmm_error,
                "pelt_error": _rd.pelt_error,
                "bcp_error": _rd.bcp_error,
            }
        except Exception:
            pass

    # Run data quality audit (feeds into profile data_quality section)
    _quality_path = None
    try:
        from operator1.quality.data_quality import run_quality_checks, save_quality_report
        _qr = run_quality_checks(cache, entity_id=ticker or "target")
        _quality_path = str(Path(args.output_dir) / "data_quality_report.json")
        save_quality_report({"target": _qr}, output_path=_quality_path)
    except Exception as exc:
        logger.debug("Quality audit skipped: %s", exc)

    try:
        profile = build_company_profile(
            verified_target=target_profile,
            cache=cache,
            linked_aggregates=linked_agg_df,
            regime_result=regime_result_dict,
            forecast_result=forecast_result,
            mc_result=mc_result,
            prediction_result=pred_result,
            quality_report_path=_quality_path,
            estimation_coverage_path=str(Path(args.output_dir) / "estimation_coverage.json"),
            graph_risk_result=_available_dict(graph_risk_result),
            game_theory_result=_available_dict(game_theory_result),
            fuzzy_protection_result=_available_dict(fuzzy_result),
            pid_summary=(
                getattr(forward_pass_result, "pid_summary", None)
                if forward_pass_result is not None
                else None
            ),
            financial_health_result=fh_dict,
            sentiment_result=sentiment_result,
            peer_ranking_result=peer_ranking_result,
            macro_quadrant_result=_to_dict(macro_quadrant_result),
        )

        # Inject enriched survival timeline summary
        if enriched_timeline_result and enriched_timeline_result.fitted:
            profile["enriched_survival_timeline"] = {
                "available": True,
                "regime_available": enriched_timeline_result.regime_available,
                "mean_intensity": enriched_timeline_result.mean_intensity,
                "combined_state_distribution": (
                    enriched_timeline_result.combined_state_distribution
                ),
                "base_n_switches": enriched_timeline_result.base.n_switches,
                "base_mean_stability": enriched_timeline_result.base.mean_stability,
            }
        else:
            profile["enriched_survival_timeline"] = {"available": False}

        # Inject filing calendar analysis
        if filing_calendar_result is not None:
            _fc_dict = {
                "available": True,
                "expected_frequency": filing_calendar_result.expected_frequency,
                "detected_frequency": filing_calendar_result.detected_frequency,
                "expected_filings_2yr": filing_calendar_result.expected_filings_2yr,
                "actual_filings_2yr": filing_calendar_result.actual_filings_2yr,
                "coverage_ratio": round(filing_calendar_result.coverage_ratio, 3),
                "latest_filing_age_days": filing_calendar_result.latest_filing_age_days,
                "is_stale": filing_calendar_result.is_stale,
                "gaps": filing_calendar_result.gaps,
            }
            # Predicted next filing date (Section E.5 from core idea)
            try:
                from operator1.features.filing_calendar import predict_next_filing_date
                _ref_date = pd.Timestamp(_backtest_end_date) if _backtest_end_date else pd.Timestamp.now()
                _next_filing = predict_next_filing_date(filing_calendar_result, reference_date=_ref_date)
                _fc_dict["next_expected_filing"] = _next_filing
            except Exception as _nf_exc:
                logger.debug("Next filing prediction failed: %s", _nf_exc)
                _fc_dict["next_expected_filing"] = {"available": False}
            profile["filing_calendar"] = _fc_dict
        else:
            profile["filing_calendar"] = {"available": False}

        # Inject economic plane classification
        try:
            plane_info = classify_economic_plane(
                sector=target_profile.get("sector"),
                industry=target_profile.get("industry"),
            )
            profile["economic_plane"] = plane_info
        except Exception as exc:
            logger.warning("Economic plane classification failed: %s", exc)
            profile["economic_plane"] = {"primary_plane": "unknown", "secondary_planes": []}

        # Inject PIT data source metadata
        profile.setdefault("meta", {})
        profile["meta"]["data_provider"] = market_info.pit_api_name
        profile["meta"]["data_provider_label"] = (
            f"{market_info.pit_api_name} ({market_info.country} -- "
            f"{market_info.exchange})"
        )
        profile["meta"]["market_id"] = market_id
        profile["meta"]["pit_source"] = True
        profile["meta"]["using_pit_only"] = True

        # Track the actual OHLCV source separately from the filing source.
        # SEC EDGAR and most PIT filing APIs don't provide price data --
        # OHLCV typically comes from yfinance or a per-region wrapper.
        profile["meta"]["ohlcv_source"] = _ohlcv_source_label

        # Flag whether OHLCV data is available in the cache.
        # Used by the report generator to decide whether to generate
        # price-based charts and reference them in the report narrative.
        _has_ohlcv = (
            "close" in cache.columns
            and cache["close"].notna().sum() >= 5
        )
        profile["meta"]["has_ohlcv"] = _has_ohlcv
        profile["meta"]["is_private_company"] = _is_private
        if not _has_ohlcv:
            logger.warning(
                "No usable OHLCV data in cache -- price charts and "
                "price-dependent models will be skipped in the report."
            )

        # Inject macro data summary
        if macro_api_info:
            profile["meta"]["macro_source"] = macro_api_info.api_name
            profile["meta"]["macro_country"] = macro_api_info.country
        if macro_data:
            macro_summary = {}
            for indicator, series in macro_data.items():
                if series is not None and not series.empty:
                    macro_summary[indicator] = {
                        "latest_value": float(series.iloc[-1]),
                        "latest_date": str(series.index[-1].date()),
                        "observations": len(series),
                    }
            profile["macro_indicators"] = macro_summary

        # Inject market buying power
        if buying_power_result is not None and buying_power_result.available:
            profile["market_buying_power"] = {
                "available": True,
                "buying_power_index": buying_power_result.buying_power_index,
                "sector_demand_momentum": buying_power_result.sector_demand_momentum,
                "real_revenue_growth_ppp": buying_power_result.real_revenue_growth_ppp,
                "demand_risk_flag": buying_power_result.demand_risk_flag,
                "consumer_confidence_trend": buying_power_result.consumer_confidence_trend,
                "inflation_drag": buying_power_result.inflation_drag,
            }
        else:
            profile["market_buying_power"] = {"available": False}

        # Inject supply chain stress flag
        if supply_chain_stress_result is not None and supply_chain_stress_result.get("available"):
            profile["supply_chain_stress"] = supply_chain_stress_result
        else:
            profile["supply_chain_stress"] = {"available": False}

        # Inject product catalyst signals
        if catalyst_result is not None and catalyst_result.available:
            profile["product_catalysts"] = {
                "available": True,
                "catalyst_score": catalyst_result.catalyst_score,
                "catalyst_type": catalyst_result.catalyst_type,
                "rnd_acceleration": catalyst_result.rnd_acceleration,
                "news_catalyst_score": catalyst_result.news_catalyst_score,
                "earnings_momentum": catalyst_result.earnings_momentum,
                "revenue_diversification_delta": catalyst_result.revenue_diversification_delta,
                "n_catalyst_articles": catalyst_result.n_catalyst_articles,
            }
        else:
            profile["product_catalysts"] = {"available": False}

        # Inject event calendar signals (Gap 4)
        if event_calendar_result is not None and event_calendar_result.available:
            profile["event_calendar_signals"] = event_calendar_result.to_profile_dict()
        else:
            profile["event_calendar_signals"] = {"available": False}
        # Inject cross-asset rotation signals (Gap 3)
        if cross_asset_result is not None and cross_asset_result.available:
            profile["cross_asset_signals"] = cross_asset_result.to_profile_dict()
        else:
            profile["cross_asset_signals"] = {"available": False}
        # Inject options-derived signals (Gap 1)
        if options_signal_result is not None and options_signal_result.available:
            profile["options_signals"] = options_signal_result.to_profile_dict()
        else:
            profile["options_signals"] = {"available": False}

        # Product segment data -- reuse _seg_result from Step 5i.6
        # (extraction + cache injection already happened before temporal models).
        try:
            if _seg_result and _seg_result.get("n_segments", 0) >= 2:
                _seg_rev = _seg_result.get("segments", {})
                _dominant = max(_seg_rev, key=_seg_rev.get) if _seg_rev else ""
                _total_rev = sum(_seg_rev.values()) if _seg_rev else 0
                _dom_pct = _seg_rev.get(_dominant, 0) / _total_rev if _total_rev > 0 else 0

                profile["product_segments"] = {
                    "available": True,
                    "segments": _seg_result.get("segments", {}),
                    "descriptions": _seg_result.get("descriptions", {}),
                    "n_segments": _seg_result.get("n_segments", 0),
                    "has_revenue": _seg_result.get("has_revenue", False),
                    "has_descriptions": _seg_result.get("has_descriptions", False),
                    "dominant_segment": _dominant,
                    "dominant_segment_pct": round(_dom_pct, 4),
                    "source": _seg_result.get("source", "unknown"),
                }
            else:
                profile["product_segments"] = {"available": False}
        except Exception as _seg_exc:
            logger.debug("Product segment profile injection failed: %s", _seg_exc)
            profile["product_segments"] = {"available": False}

        # Inject reconciliation report
        if reconciliation_report:
            profile["meta"]["reconciliation"] = reconciliation_report

        # Inject GLEIF corporate structure into profile
        _parent_list = relationships.get("parent_companies", [])
        _sub_list = relationships.get("subsidiaries", [])
        if _parent_list or _sub_list:
            _parent_dicts = []
            for _p in _parent_list:
                if isinstance(_p, dict):
                    _parent_dicts.append({
                        "name": _p.get("name", ""),
                        "lei": _p.get("lei", ""),
                        "country": _p.get("country", ""),
                        "relationship": _p.get("relationship", "parent"),
                    })
            _sub_dicts = []
            _sub_countries = set()
            for _s in _sub_list:
                if isinstance(_s, dict):
                    _sub_dicts.append({
                        "name": _s.get("name", ""),
                        "lei": _s.get("lei", ""),
                        "country": _s.get("country", ""),
                    })
                    if _s.get("country"):
                        _sub_countries.add(_s["country"])
            profile["corporate_structure"] = {
                "available": True,
                "source": "gleif",
                "parent_companies": _parent_dicts,
                "n_parents": len(_parent_dicts),
                "subsidiaries": _sub_dicts[:10],
                "n_subsidiaries": len(_sub_dicts),
                "subsidiaries_countries": sorted(_sub_countries),
                "cross_border": len(_sub_countries) > 1,
            }
        else:
            profile["corporate_structure"] = {"available": False}

        # Inject linked entity conflict data into profile
        # (Bug fix: linked_conflict was passed to inject_conflict_risk_into_cache
        # but never injected into the profile dict, so report_generator's
        # geopolitical risk section silently returned empty for linked fields.)
        if linked_conflict and isinstance(linked_conflict, dict):
            profile.setdefault("conflict_risk", {})["linked_conflict"] = linked_conflict

        # Inject institutional/major holders
        if target_holders:
            profile["institutional_holders"] = {
                "available": True,
                "holders": target_holders[:10],
                "total_holders": len(target_holders),
            }
        else:
            profile["institutional_holders"] = {"available": False}

        # Inject institutional ownership deep analysis (contagion + flow)
        _inst_analysis: dict[str, Any] = {"available": False}
        try:
            _has_contagion = contagion_result is not None and contagion_result.available
            _has_flow = "inst_flow_momentum" in cache.columns and cache["inst_flow_momentum"].notna().any()

            if _has_contagion or _has_flow:
                _inst_analysis = {"available": True}

                if _has_contagion:
                    _inst_analysis["contagion"] = {
                        "mhhi_delta": _safe_float(contagion_result.mhhi_delta),
                        "mhhi_label": (
                            "high" if contagion_result.mhhi_delta > 0.3
                            else "moderate" if contagion_result.mhhi_delta > 0.1
                            else "low"
                        ),
                        "shared_institutions": contagion_result.shared_institutions[:5],
                        "n_shared": contagion_result.n_shared_institutions,
                        "bipartite_centrality": _safe_float(contagion_result.target_bipartite_centrality),
                        "network_density": _safe_float(contagion_result.ownership_network_density),
                        "most_influential_institution": contagion_result.most_connected_institution,
                        "crowding_score": _safe_float(contagion_result.crowding_score),
                        "crowded_trade_flag": contagion_result.crowded_trade_flag,
                        "liquidation_days": _safe_float(contagion_result.liquidation_days),
                        "liquidation_risk": _safe_float(contagion_result.liquidation_risk),
                    }

                if _has_flow:
                    _latest = cache.iloc[-1]
                    _inst_analysis["flow"] = {
                        "momentum_latest": _safe_float(_latest.get("inst_flow_momentum")),
                        "momentum_label": str(_latest.get("inst_flow_momentum_label", "unknown")),
                        "crowding_risk_latest": _safe_float(_latest.get("inst_crowding_risk")),
                        "crowding_risk_label": str(_latest.get("inst_crowding_risk_label", "unknown")),
                        "smart_money_signal": _safe_float(_latest.get("inst_smart_money_signal")),
                        "smart_money_label": str(_latest.get("inst_smart_money_label", "unknown")),
                        "insider_signal": _safe_float(_latest.get("inst_insider_signal")),
                        "insider_label": str(_latest.get("inst_insider_label", "unknown")),
                        "amihud_illiquidity": _safe_float(_latest.get("inst_amihud_illiquidity")),
                    }
        except Exception as _exc:
            logger.debug("Institutional ownership analysis profile section failed: %s", _exc)

        profile["institutional_ownership_analysis"] = _inst_analysis

        # Inject extended model results into profile
        if "extended_models" not in profile:
            profile["extended_models"] = {}

        if transfer_entropy_result is not None:
            profile["extended_models"]["transfer_entropy"] = _available_dict(transfer_entropy_result)
        if cycle_result is not None:
            profile["extended_models"]["cycle_decomposition"] = _available_dict(cycle_result)
        if pattern_result is not None:
            profile["extended_models"]["candlestick_patterns"] = _available_dict(pattern_result)
        if copula_result is not None:
            profile["extended_models"]["copula"] = _available_dict(copula_result)
        if conformal_result is not None:
            profile["extended_models"]["conformal_prediction"] = _available_dict(conformal_result)
        if dtw_result is not None:
            try:
                from operator1.models.dtw_analogs import format_analogs_for_profile
                profile["extended_models"]["dtw_analogs"] = format_analogs_for_profile(dtw_result)
            except Exception:
                profile["extended_models"]["dtw_analogs"] = _available_dict(dtw_result)
        if shap_result is not None:
            try:
                from operator1.models.explainability import format_shap_for_profile
                profile["extended_models"]["shap_explanations"] = format_shap_for_profile(shap_result)
            except Exception:
                profile["extended_models"]["shap_explanations"] = _available_dict(shap_result)
        if sobol_result is not None:
            profile["extended_models"]["sobol_sensitivity"] = _available_dict(sobol_result)
        if particle_filter_result is not None:
            pf_dict = _available_dict(particle_filter_result)
            if pf_dict:
                for key in ("filtered_states", "particles_final", "weights_final"):
                    if key in pf_dict and hasattr(pf_dict[key], "tolist"):
                        pf_dict[key] = "<%d values>" % len(pf_dict[key])
                if "percentiles" in pf_dict and isinstance(pf_dict["percentiles"], dict):
                    pf_dict["percentiles"] = {
                        k: v.tolist() if hasattr(v, "tolist") else v
                        for k, v in pf_dict["percentiles"].items()
                    }
            profile["extended_models"]["particle_filter"] = pf_dict
        if transformer_result is not None:
            tf_dict = _available_dict(transformer_result)
            if tf_dict:
                tf_dict.pop("attention_weights", None)
                tf_dict.pop("train_loss_history", None)
            profile["extended_models"]["transformer"] = tf_dict

        # OHLC predictions
        if ohlc_result is not None and ohlc_result.fitted:
            try:
                from operator1.models.ohlc_predictor import format_ohlc_for_profile
                profile["ohlc_predictions"] = format_ohlc_for_profile(ohlc_result)
            except Exception as exc:
                logger.warning("OHLC profile formatting failed: %s", exc)

        # Granger causality
        if granger_result is not None and granger_result.fitted:
            profile["extended_models"]["granger_causality"] = {
                "available": True,
                "n_significant_pairs": len(granger_result.significant_pairs),
                "network_density": granger_result.network_density,
                "n_retained": len(granger_result.retained_variables),
                "n_pruned": len(granger_result.pruned_variables),
                "top_pairs": granger_result.significant_pairs[:10],
            }

        # Time-varying Granger causality (Proposal 3.5)
        if _tv_granger_result is not None and _tv_granger_result.get("n_windows", 0) > 0:
            profile["extended_models"]["time_varying_granger"] = {
                "available": True,
                "n_windows": _tv_granger_result["n_windows"],
                "emerging_pairs": _tv_granger_result.get("emerging_pairs", []),
                "disappearing_pairs": _tv_granger_result.get("disappearing_pairs", []),
            }

        # Multivariate Monte Carlo (Proposal 3.3)
        if _mv_mc_result is not None and _mv_mc_result.get("available"):
            profile["extended_models"]["multivariate_monte_carlo"] = _mv_mc_result

        # Predicted regime shifts (Section E.5 from core idea)
        if regime_shift_result is not None and regime_shift_result.available:
            profile["predicted_regime_shifts"] = regime_shift_result.to_dict()
        else:
            profile["predicted_regime_shifts"] = {"available": False}

        # Dual regimes
        if dual_regime_result is not None and dual_regime_result.fitted:
            profile["extended_models"]["dual_regimes"] = {
                "available": True,
                "fundamental_regime_current": (
                    str(dual_regime_result.fund_regime_labels.iloc[-1])
                    if dual_regime_result.fund_regime_labels is not None
                    and len(dual_regime_result.fund_regime_labels) > 0
                    else None
                ),
                "fundamental_distribution": (
                    dual_regime_result.fund_regime_labels.value_counts(normalize=True)
                    .round(4).to_dict()
                    if dual_regime_result.fund_regime_labels is not None
                    else {}
                ),
            }

        # Walk-forward results
        if walk_forward_result is not None:
            profile["extended_models"]["walk_forward"] = {
                "available": True,
                "overall_mae": getattr(walk_forward_result, "overall_mae", None),
                "overall_best_model": getattr(walk_forward_result, "overall_best_model", None),
                "n_retrains": len(getattr(walk_forward_result, "retrain_dates", [])),
                "mode_scores": getattr(walk_forward_result, "mode_scores", {}),
                "best_model_by_mode": getattr(walk_forward_result, "best_model_by_mode", {}),
            }

        # Burn-out results
        if burnout_result is not None:
            profile["extended_models"]["burnout"] = {
                "available": True,
                "iterations_completed": burnout_result.iterations_completed,
                "converged": burnout_result.converged,
                "calibrated": getattr(burnout_result, "calibrated", False),
                "weight_stability": getattr(burnout_result, "weight_stability", None),
                "calibration_steps": getattr(burnout_result, "calibration_steps", 0),
                "regime_weights": getattr(burnout_result, "regime_weights", {}),
                "regime_distributions": getattr(burnout_result, "regime_distributions", {}),
            }

        # GA optimization
        if ga_result is not None and ga_result.fitted:
            profile["extended_models"]["genetic_optimizer"] = {
                "available": True,
                "best_weights": ga_result.best_weights,
                "tier_weights": ga_result.tier_weights,
                "n_generations": ga_result.n_generations,
                "converged": ga_result.converged,
            }

        # Recursive day-by-day predictions
        if recursive_result is not None and getattr(recursive_result, "available", False):
            try:
                profile["extended_models"]["recursive_predictions"] = recursive_result.to_dict()
            except Exception:
                profile["extended_models"]["recursive_predictions"] = {"available": True}

        # Synergy metadata
        # Module contribution scores (Section F.1 Category 7 from core idea)
        if pred_result is not None and hasattr(pred_result, "module_contributions") and pred_result.module_contributions:
            profile.setdefault("model_metrics", {})["module_contributions"] = pred_result.module_contributions

        # Inject feature selection results
        if feature_selection_result is not None and feature_selection_result.fitted:
            profile["feature_selection"] = {
                "available": True,
                "n_input": feature_selection_result.n_input,
                "n_output": feature_selection_result.n_output,
                "boruta_confirmed": feature_selection_result.boruta_confirmed[:20],
                "boruta_tentative": feature_selection_result.boruta_tentative[:10],
                "regime_selected": {
                    k: v[:10] for k, v in feature_selection_result.regime_selected.items()
                },
                "mrmr_selected": feature_selection_result.mrmr_selected[:15],
                "method_contributions": feature_selection_result.method_contributions,
            }
        else:
            profile["feature_selection"] = {"available": False}

        if _synergy_meta:
            profile["synergies_applied"] = {
                "cycle_features_added": _synergy_meta.get("cycle_features_added", []),
                "unified_causal_network": {
                    "n_pairs": len(_synergy_meta.get("unified_causal_network", {}).get("all_pairs", [])),
                    "n_retained": len(_synergy_meta.get("unified_causal_network", {}).get("retained_variables", [])),
                    "density": _synergy_meta.get("unified_causal_network", {}).get("network_density", 0),
                },
                "variables_after_pruning": _synergy_meta.get("variables_after_pruning", 0),
                "adjusted_survival_thresholds": _synergy_meta.get("adjusted_survival_thresholds", {}),
                "pattern_drift_applied": _pattern_drift if not args.skip_models else 1.0,
            }

        # Inject Unified Survival System data into profile
        if survival_controller is not None:
            profile["unified_survival_system"] = survival_controller.to_profile_dict()
        else:
            profile["unified_survival_system"] = {"available": False}

        # Inject model diagnostics (expected path vs actual path)
        if model_diagnostics_result is not None and model_diagnostics_result.available:
            profile["model_diagnostics"] = model_diagnostics_result.to_dict()
        else:
            profile["model_diagnostics"] = {"available": False}

        # Inject scenario engine results
        if scenario_result is not None and scenario_result.available:
            profile["scenario_analysis"] = scenario_result.to_dict()
        else:
            profile["scenario_analysis"] = {"available": False}

        # Inject multi-frequency fusion results
        if multi_frequency_result is not None and hasattr(multi_frequency_result, "available") and multi_frequency_result.available:
            profile["multi_frequency"] = multi_frequency_result.to_profile_dict()
        else:
            profile["multi_frequency"] = {"available": False}

        # Inject Signal IC results
        if signal_ic_result is not None and signal_ic_result.available:
            profile["signal_ic"] = signal_ic_result.to_profile_dict()
        else:
            profile["signal_ic"] = {"available": False}

        # Inject prediction log summary (from previous runs)
        if prediction_log_summary is not None:
            profile["prediction_log"] = prediction_log_summary
        else:
            profile["prediction_log"] = {"n_filled": 0, "total_predictions": 0}

        # Compute position signal (-1 to +1 directional conviction)
        _position_signal = 0.0
        try:
            # Base: return_5d forecast direction + magnitude
            _return_forecast = 0.0
            if pred_result is not None and hasattr(pred_result, "predictions"):
                _r5d = pred_result.predictions.get("return_5d", {}).get("5d")
                if _r5d is None:
                    _r5d = pred_result.predictions.get("close", {}).get("5d")
                if _r5d is not None:
                    pf = getattr(_r5d, "point_forecast", None)
                    if pf is not None:
                        _return_forecast = float(pf)

            # IC confidence multiplier
            _ic_conf = 1.0
            if signal_ic_result and signal_ic_result.available:
                _ic_conf = min(2.0, max(0.5, abs(signal_ic_result.best_ic) * 20))

            # Survival regime multiplier
            _surv_mult = 1.0
            if survival_controller is not None:
                if survival_controller.is_survival:
                    _surv_mult = 0.3  # dampen during distress
                # Recovery boost
                _recovery = survival_controller.detect_recovery_signal()
                if _recovery.get("active"):
                    _surv_mult *= _recovery.get("position_signal_boost", 1.0)

            # Combine: forecast * IC confidence * survival multiplier
            _raw_signal = _return_forecast * _ic_conf * _surv_mult
            _position_signal = max(-1.0, min(1.0, _raw_signal * 100))  # scale to -1/+1

            # Label
            if _position_signal > 0.3:
                _pos_label = "buy"
            elif _position_signal < -0.3:
                _pos_label = "sell"
            else:
                _pos_label = "hold"

            profile["position_signal"] = {
                "available": True,
                "signal": round(_position_signal, 4),
                "label": _pos_label,
                "return_forecast": round(_return_forecast, 6),
                "ic_confidence": round(_ic_conf, 4),
                "survival_multiplier": round(_surv_mult, 4),
                "recovery_active": _recovery.get("active", False) if survival_controller else False,
            }
        except Exception as _ps_exc:
            logger.debug("Position signal computation failed: %s", _ps_exc)
            profile["position_signal"] = {"available": False}

        # W2: Inject Hedge Fund Analysis results into profile
        if hf_result is not None and hf_result.available:
            profile["hedge_fund"] = hf_result.to_profile_dict()
        else:
            profile["hedge_fund"] = {"available": False}

        # Save profile -- sanitize dict keys (some model results use tuple keys)
        def _sanitize_keys(obj):
            """Recursively convert non-string dict keys to strings for JSON."""
            if isinstance(obj, dict):
                return {str(k): _sanitize_keys(v) for k, v in obj.items()}
            if isinstance(obj, list):
                return [_sanitize_keys(i) for i in obj]
            return obj

        profile = _sanitize_keys(profile)
        profile_path = Path(args.output_dir) / "company_profile.json"
        profile_path.parent.mkdir(parents=True, exist_ok=True)
        with open(profile_path, "w", encoding="utf-8") as fh:
            json.dump(profile, fh, indent=2, default=str)
        logger.info("Profile saved: %s", profile_path)

        # Store predictions to log for future IC evaluation
        if pred_result is not None and hasattr(pred_result, "predictions"):
            try:
                from operator1.analysis.prediction_log import store_predictions
                _model_used = {}
                if forecast_result is not None and hasattr(forecast_result, "model_used"):
                    _model_used = forecast_result.model_used
                _surv_regime = "normal"
                if "survival_regime" in cache.columns:
                    _sr = cache["survival_regime"].dropna()
                    if len(_sr) > 0:
                        _surv_regime = str(_sr.iloc[-1])
                store_predictions(
                    ticker=ticker,
                    predictions=pred_result.predictions,
                    model_used=_model_used,
                    survival_regime=_surv_regime,
                    run_date=_backtest_end_date,
                )
            except Exception as _pl_exc:
                logger.debug("Prediction log storage failed: %s", _pl_exc)

        # Validate profile completeness before report generation
        try:
            from operator1.report.profile_schema import validate_profile
            _profile_issues = validate_profile(profile)
            if _profile_issues:
                logger.warning(
                    "Profile validation: %d issues (report may have missing sections)",
                    len(_profile_issues),
                )
        except Exception as exc:
            logger.debug("Profile validation skipped: %s", exc)

    except Exception as exc:
        logger.error("Profile building failed: %s", exc)
        return 1

    # ------------------------------------------------------------------
    # Step 8: Generate report via Gemini (optional)
    # ------------------------------------------------------------------
    if not args.skip_report:
        logger.info("")
        logger.info("Step 8: Generating reports (Basic + Pro + Premium)...")

        from operator1.report.report_generator import generate_all_reports

        try:
            all_reports = generate_all_reports(
                profile=profile,
                llm_client=llm_client,
                cache=cache,
                output_dir=Path(args.output_dir) / "report",
                generate_pdf=args.pdf,
            )
            for tier_name, report_output in all_reports.items():
                logger.info(
                    "  %s report: %s",
                    tier_name.capitalize(),
                    report_output.get("markdown_path"),
                )
            premium = all_reports.get("premium", {})
            if premium.get("pdf_path"):
                logger.info("PDF saved: %s", premium["pdf_path"])
        except Exception as exc:
            logger.error("Report generation failed: %s", exc)

        # Step 8-USS: Generate triage card when in company distress
        # Only for company_survival and extreme_survival (not modified_survival,
        # where the company itself is healthy but country is in crisis)
        if (survival_controller is not None
                and survival_controller.current_regime in ("company_survival", "extreme_survival")):
            try:
                from operator1.report.triage_card import generate_triage_card
                triage_output = generate_triage_card(
                    profile=profile,
                    cache=cache,
                    scenario_result=scenario_result,
                    controller=survival_controller,
                    output_dir=Path(args.output_dir) / "report",
                )
                if triage_output.get("markdown_path"):
                    logger.info("Triage card saved: %s", triage_output["markdown_path"])
            except Exception as exc:
                logger.warning("Triage card generation failed: %s", exc)
    else:
        logger.info("Step 8: Skipped (--skip-report)")

    # ------------------------------------------------------------------
    # Done
    # ------------------------------------------------------------------
    logger.info("")
    logger.info("=" * 60)
    logger.info("PIPELINE COMPLETE")
    logger.info("=" * 60)
    logger.info("Market: %s (%s)", market_info.country, market_info.pit_api_name)
    logger.info("Company: %s (%s)", company_name, ticker)
    logger.info("Output directory: %s", args.output_dir)

    return 0


def _generate_report_only(args: argparse.Namespace, secrets: dict) -> int:
    """Generate report from an existing profile JSON."""
    profile_path = Path(args.output_dir) / "company_profile.json"
    if not profile_path.exists():
        logger.error("No existing profile found at %s. Run full pipeline first.", profile_path)
        return 1

    with open(profile_path, "r", encoding="utf-8") as fh:
        profile = json.load(fh)

    from operator1.clients.llm_factory import create_llm_client
    llm_client = create_llm_client(secrets)

    from operator1.report.report_generator import generate_report

    report_output = generate_report(
        profile=profile,
        llm_client=llm_client,
        output_dir=Path(args.output_dir) / "report",
        generate_pdf=args.pdf,
    )
    logger.info("Report saved: %s", report_output.get("markdown_path"))
    return 0


if __name__ == "__main__":
    try:
        exit_code = main()
    finally:
        _finalize_log()
    sys.exit(exit_code)
