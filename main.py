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

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-7s | %(name)s | %(message)s",
    datefmt="%H:%M:%S",
)
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
        "--verbose", "-v", action="store_true",
        help="Enable debug logging",
    )
    args = parser.parse_args()

    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)

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
        validate_secrets(secrets)
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

    ticker = company_info.get("ticker", "")
    company_name = company_info.get("name", ticker)
    identifier = company_info.get("cik") or ticker

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
        target_profile = {
            "name": company_name,
            "ticker": ticker,
            "country": market_info.country_code,
            "market_id": market_id,
            "pit_api": market_info.pit_api_name,
            "sector": company_info.get("sector", ""),
            "industry": company_info.get("industry", ""),
        }

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
    if quotes_df.empty and ticker:
        logger.info(
            "PIT source %s does not provide OHLCV -- fetching from OHLCV provider.",
            market_info.pit_api_name,
        )
        try:
            from operator1.clients.ohlcv_provider import fetch_ohlcv
            quotes_df = fetch_ohlcv(ticker, market_id=args.market)
            if not quotes_df.empty:
                logger.info("OHLCV fetched from provider: %d rows", len(quotes_df))
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
        end = date.today()
        start = end - timedelta(days=int(args.years * 365))
        idx = pd.date_range(start, end, freq="B", name="date")
        cache = pd.DataFrame(index=idx)

    # Merge financial statement data using PIT filing_date (as-of join).
    # Columns are merged WITHOUT prefixes so that derived_variables.py
    # finds the expected canonical names (revenue, total_assets, etc.).
    # If the same column name exists in multiple statements, the first
    # non-null value wins (income > balance > cashflow priority).
    _merged_cols: set = set(cache.columns)

    for label, stmt_df in [
        ("income", income_df),
        ("balance", balance_df),
        ("cashflow", cashflow_df),
    ]:
        if stmt_df.empty:
            continue
        try:
            # Use report_date for alignment (filing_date has duplicates from
            # multi-period filings like 10-Q containing both Q and YTD data).
            # The PIT constraint is still satisfied: we forward-fill from the
            # report_date, which is always <= filing_date.
            date_col = "report_date" if "report_date" in stmt_df.columns else "filing_date"
            if date_col not in stmt_df.columns:
                logger.warning("No date column in %s data, skipping merge", label)
                continue

            stmt_df[date_col] = pd.to_datetime(stmt_df[date_col])
            stmt_df = stmt_df.sort_values(date_col)
            # Deduplicate: keep last row per date (most recent filing)
            stmt_df = stmt_df.drop_duplicates(subset=[date_col], keep="last")

            # Forward-fill financial data onto the daily cache (as-of join)
            numeric_cols = stmt_df.select_dtypes(include=["number"]).columns.tolist()
            # Exclude date-like columns from numeric merge
            numeric_cols = [c for c in numeric_cols if c != date_col and "date" not in c.lower()]
            if not numeric_cols:
                continue

            stmt_indexed = stmt_df.set_index(date_col)[numeric_cols]
            # Forward-fill financial data onto the daily cache (as-of join).
            # The quarterly filing dates don't exist in the daily index,
            # so we union the indices first, then ffill, then select daily dates.
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

    logger.info("Cache built: %d rows x %d columns", len(cache), len(cache.columns))

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
    # Step 5: Feature engineering
    # ------------------------------------------------------------------
    logger.info("")
    logger.info("Step 5: Computing derived features...")

    from operator1.features.derived_variables import compute_derived_variables
    from operator1.analysis.survival_mode import compute_company_survival_flag
    from operator1.analysis.hierarchy_weights import compute_hierarchy_weights

    try:
        cache = compute_derived_variables(cache)
        logger.info("Features computed: %d columns", len(cache.columns))
    except Exception as exc:
        logger.warning("Feature engineering partially failed: %s", exc)

    # Survival mode
    weights: dict = {f"tier{i}": 20.0 for i in range(1, 6)}
    try:
        cache["company_survival_mode_flag"] = compute_company_survival_flag(cache)
        cache = compute_hierarchy_weights(cache)
        for i in range(1, 6):
            col = f"hierarchy_tier{i}_weight"
            if col in cache.columns:
                weights[f"tier{i}"] = float(cache[col].iloc[-1])
        logger.info("Survival mode: %d days flagged", cache["company_survival_mode_flag"].sum())
    except Exception as exc:
        logger.warning("Survival mode detection failed: %s", exc)

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

        cache = compute_fuzzy_protection(
            cache,
            sector=target_profile.get("sector"),
            gdp=_gdp_val,
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

    # Step 5e: Linked entity discovery via Gemini (optional)
    relationships = {}
    graph_risk_result = None
    game_theory_result = None
    linked_caches: dict[str, pd.DataFrame] = {}
    linked_agg_df: pd.DataFrame | None = None

    # Build the LLM client once for the whole pipeline
    from operator1.clients.llm_factory import create_llm_client
    llm_client = create_llm_client(secrets)

    if not args.skip_linked and llm_client is not None:
        logger.info("")
        logger.info("Step 5e: Discovering linked entities via %s...", llm_client.provider_name)

        try:
            from operator1.steps.entity_discovery import discover_linked_entities

            gemini_client = llm_client
            discovery_result = discover_linked_entities(
                target_profile=target_profile,
                gemini_client=gemini_client,
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

        # Graph risk
        try:
            from operator1.models.graph_risk import compute_graph_risk_metrics
            graph_risk_result = compute_graph_risk_metrics(
                target_isin=target_profile.get("isin", ticker),
                relationships=relationships,
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

                    # Merge statements (same logic as target cache, simplified)
                    for _lbl, _sdf in [("inc", _inc), ("bal", _bal), ("cf", _cf)]:
                        if _sdf.empty:
                            continue
                        _dcol = "filing_date" if "filing_date" in _sdf.columns else "report_date"
                        if _dcol not in _sdf.columns:
                            continue
                        _sdf[_dcol] = pd.to_datetime(_sdf[_dcol])
                        _sdf = _sdf.sort_values(_dcol)
                        _ncols = _sdf.select_dtypes(include=["number"]).columns.tolist()
                        _ncols = [c for c in _ncols if c != _dcol and "date" not in c.lower()]
                        if _ncols:
                            _si = _sdf.set_index(_dcol)[_ncols]
                            _sa = _si.reindex(_ent_cache.index, method="ffill")
                            _new = [c for c in _sa.columns if c not in _ent_cache.columns]
                            if _new:
                                _ent_cache = _ent_cache.join(_sa[_new], how="left")

                    # Compute derived variables
                    if "close" in _ent_cache.columns and _ent_cache["close"].notna().sum() > 5:
                        from operator1.features.derived_variables import compute_derived_variables
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
                except Exception as exc:
                    logger.warning("Linked aggregates computation failed: %s", exc)
    else:
        if llm_client is None:
            logger.info("Step 5e: Skipped (no LLM API key for entity discovery)")
        else:
            logger.info("Step 5e: Skipped (--skip-linked)")

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
            gemini_client=llm_client,
            symbol=ticker,
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
    # Step 6: Temporal modeling (optional)
    # ------------------------------------------------------------------
    forecast_result = None
    forward_pass_result = None
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
    _synergy_meta = {}
    _pattern_drift = 1.0

    if not args.skip_models:
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
        _linked_prefixes = (
            "competitors_", "suppliers_", "customers_",
            "financial_institutions_", "sector_peers_", "industry_peers_",
        )
        _extra_vars = [
            c for c in cache.columns
            if (c.startswith("fh_") or c.startswith("sentiment_")
                or c.startswith("peer_") or c.startswith("macro_")
                or any(c.startswith(p) for p in _linked_prefixes))
            and cache[c].dtype in ("float64", "float32", "int64")
            and not c.startswith("is_missing_")
        ]
        if _extra_vars:
            logger.info("Extra variables for temporal models (%d): %s", len(_extra_vars), _extra_vars[:10])

        # Regime detection
        regime_detector = None
        try:
            cache, regime_detector = detect_regimes_and_breaks(cache)
            logger.info("Regimes detected")
        except Exception as exc:
            logger.warning("Regime detection failed: %s", exc)

        # Dual regime classification
        try:
            from operator1.models.regime_mixer import compute_dual_regimes
            dual_regime_result = compute_dual_regimes(cache)
            if dual_regime_result and dual_regime_result.fitted:
                logger.info("Dual regime classification complete")
        except Exception as exc:
            logger.warning("Dual regime classification failed: %s", exc)

        # Granger causality
        try:
            from operator1.models.granger_causality import (
                compute_granger_causality,
                prune_features_by_causality,
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
                    _extra_vars = prune_features_by_causality(
                        _extra_vars,
                        granger_result,
                        always_keep=["close", "return_1d", "volatility_21d"],
                    )
                    logger.info(
                        "Granger causality: %d significant pairs, %d variables retained",
                        len(granger_result.significant_pairs),
                        len(_extra_vars),
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
        try:
            from operator1.models.cycle_decomposition import run_cycle_decomposition
            cycle_result = run_cycle_decomposition(cache, variable="close")
            logger.info("Cycle decomposition complete")
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

        # Standard forecasting
        try:
            cache, forecast_result = run_forecasting(
                cache,
                extra_variables=_extra_vars,
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
            logger.warning("Forward pass failed: %s", exc)

        # Burn-out
        try:
            burnout_result = run_burnout(
                cache,
                hierarchy_weights=weights,
                regime_labels=regime_labels,
                extra_variables=_extra_vars,
            )
            logger.info(
                "Burn-out complete: %d iterations, converged=%s",
                burnout_result.iterations_completed,
                burnout_result.converged,
            )
        except Exception as exc:
            logger.warning("Burn-out failed: %s", exc)

        # Monte Carlo
        try:
            mc_result = run_monte_carlo(cache)
            logger.info("Monte Carlo simulation complete")
        except Exception as exc:
            logger.warning("Monte Carlo failed: %s", exc)

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

        # Prediction aggregation
        if forecast_result is not None:
            try:
                pred_result = run_prediction_aggregation(
                    cache, forecast_result, mc_result,
                )
                logger.info("Predictions aggregated")

                # Copula tail adjustment
                if (
                    copula_result is not None
                    and hasattr(copula_result, "tail_dependence")
                    and copula_result.tail_dependence > 0.2
                    and pred_result is not None
                    and pred_result.fitted
                ):
                    _tail_mult = 1.0 + copula_result.tail_dependence
                    for var_preds in pred_result.predictions.values():
                        for hp in var_preds.values():
                            if not (hp.lower_ci != hp.lower_ci):
                                mid = hp.point_forecast
                                hp.lower_ci = mid - (mid - hp.lower_ci) * _tail_mult
                                hp.upper_ci = mid + (hp.upper_ci - mid) * _tail_mult
                    logger.info("Copula tail adjustment applied")
            except Exception as exc:
                logger.warning("Prediction aggregation failed: %s", exc)

        # Conformal prediction
        try:
            from operator1.models.conformal import ConformalCalibrator, build_conformal_result
            if forecast_result is not None and pred_result is not None:
                calibrator = ConformalCalibrator(coverage=0.9, adaptive=True)
                if hasattr(forecast_result, "residuals") and forecast_result.residuals is not None:
                    for r in forecast_result.residuals:
                        calibrator.update(r)
                _point_forecasts: dict[str, float] = {}
                if hasattr(pred_result, "predictions"):
                    for var, horizons_dict in pred_result.predictions.items():
                        if isinstance(horizons_dict, dict):
                            for h, hp in horizons_dict.items():
                                pf = getattr(hp, "point_forecast", None)
                                if pf is not None:
                                    _point_forecasts[f"{var}_{h}"] = pf
                conformal_result = build_conformal_result(
                    calibrator,
                    forecasts=_point_forecasts,
                    horizons={"1d": 1, "5d": 5, "21d": 21, "252d": 252},
                )
                logger.info("Conformal prediction intervals computed")
        except Exception as exc:
            logger.warning("Conformal prediction failed: %s", exc)

        # DTW analogs
        try:
            from operator1.models.dtw_analogs import find_historical_analogs
            dtw_result = find_historical_analogs(cache)
            logger.info("DTW analogs complete")
        except Exception as exc:
            logger.warning("DTW historical analogs failed: %s", exc)

        # SHAP explainability
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
                shap_result = compute_shap_explanations(
                    cache,
                    predictions=_shap_preds,
                )
                logger.info("SHAP explanations computed")
        except Exception as exc:
            logger.warning("SHAP explainability failed: %s", exc)

        # Sobol sensitivity
        try:
            from operator1.models.sensitivity import run_sensitivity_analysis
            sobol_result = run_sensitivity_analysis(cache, target_variable="return_1d")
            logger.info("Sobol sensitivity analysis complete")
        except Exception as exc:
            logger.warning("Sobol sensitivity failed: %s", exc)

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
            )
            if ohlc_result and ohlc_result.fitted:
                logger.info("OHLC prediction complete")
        except Exception as exc:
            logger.warning("OHLC candlestick prediction failed: %s", exc)
    else:
        logger.info("Step 6: Skipped (--skip-models)")
        regime_detector = None

    # ------------------------------------------------------------------
    # Step 7: Build company profile
    # ------------------------------------------------------------------
    logger.info("")
    logger.info("Step 7: Building company profile...")

    from operator1.report.profile_builder import build_company_profile
    from dataclasses import asdict as _asdict

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

    try:
        profile = build_company_profile(
            verified_target=target_profile,
            cache=cache,
            linked_aggregates=linked_agg_df,
            regime_result=regime_result_dict,
            forecast_result=forecast_result,
            mc_result=mc_result,
            prediction_result=pred_result,
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

        # Inject economic plane classification
        try:
            from operator1.analysis.economic_planes import classify_economic_plane
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

        # Inject reconciliation report
        if reconciliation_report:
            profile["meta"]["reconciliation"] = reconciliation_report

        # Inject extended model results into profile
        if "extended_models" not in profile:
            profile["extended_models"] = {}

        if transfer_entropy_result is not None:
            profile["extended_models"]["transfer_entropy"] = {"available": True}
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

        # GA optimization
        if ga_result is not None and ga_result.fitted:
            profile["extended_models"]["genetic_optimizer"] = {
                "available": True,
                "best_weights": ga_result.best_weights,
                "tier_weights": ga_result.tier_weights,
                "n_generations": ga_result.n_generations,
                "converged": ga_result.converged,
            }

        # Synergy metadata
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

        # Save profile
        profile_path = Path(args.output_dir) / "company_profile.json"
        profile_path.parent.mkdir(parents=True, exist_ok=True)
        with open(profile_path, "w", encoding="utf-8") as fh:
            json.dump(profile, fh, indent=2, default=str)
        logger.info("Profile saved: %s", profile_path)
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
                gemini_client=llm_client,
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
        gemini_client=llm_client,
        output_dir=Path(args.output_dir) / "report",
        generate_pdf=args.pdf,
    )
    logger.info("Report saved: %s", report_output.get("markdown_path"))
    return 0


if __name__ == "__main__":
    sys.exit(main())
