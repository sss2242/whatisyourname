#!/usr/bin/env python3
"""Generate frozen AAPL test fixtures from live pipeline.

Run: python tests/generate_golden_data.py
Requires: EDGAR_IDENTITY env var (SEC email) + internet.
Run manually when test data needs refreshing (rarely).

Produces:
  tests/fixtures/aapl_cache.parquet
  tests/fixtures/aapl_income.parquet
  tests/fixtures/aapl_balance.parquet
  tests/fixtures/aapl_cashflow.parquet
  tests/fixtures/aapl_profile.json
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import pandas as pd

# Add project root to path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

FIXTURE_DIR = Path(__file__).parent / "fixtures"


def main() -> int:
    FIXTURE_DIR.mkdir(parents=True, exist_ok=True)

    from operator1.secrets_loader import load_secrets
    secrets = load_secrets()

    from operator1.clients.equity_provider import create_pit_client
    pit_client = create_pit_client("us_sec_edgar", secrets)

    # Search and profile
    results = pit_client.search_company("AAPL")
    company_info = results[0] if results else {"ticker": "AAPL", "name": "Apple Inc"}
    identifier = company_info.get("cik") or company_info.get("ticker", "AAPL")

    profile = pit_client.get_profile(identifier)
    profile.setdefault("name", "Apple Inc")
    profile.setdefault("ticker", "AAPL")
    profile.setdefault("sector", "Technology")
    profile.setdefault("country", "US")

    with open(FIXTURE_DIR / "aapl_profile.json", "w") as f:
        json.dump(profile, f, indent=2, default=str)
    print(f"Saved: aapl_profile.json")

    # Fetch statements
    income_df = pit_client.get_income_statement(identifier)
    balance_df = pit_client.get_balance_sheet(identifier)
    cashflow_df = pit_client.get_cashflow_statement(identifier)
    quotes_df = pit_client.get_quotes(identifier)

    for name, df in [("income", income_df), ("balance", balance_df), ("cashflow", cashflow_df)]:
        if not df.empty:
            df.to_parquet(FIXTURE_DIR / f"aapl_{name}.parquet")
            print(f"Saved: aapl_{name}.parquet ({len(df)} rows)")

    # OHLCV fallback
    if quotes_df.empty:
        from operator1.clients.ohlcv_provider import fetch_ohlcv
        quotes_df = fetch_ohlcv("AAPL", market_id="us_sec_edgar")

    # Build cache (simplified version of main.py Step 4)
    if not quotes_df.empty:
        if "date" in quotes_df.columns:
            quotes_df["date"] = pd.to_datetime(quotes_df["date"])
            cache = quotes_df.set_index("date").sort_index()
        else:
            cache = quotes_df.sort_index()
    else:
        cache = pd.DataFrame(index=pd.bdate_range("2023-01-01", periods=504, freq="B"))

    # Merge statements via ffill
    for label, stmt_df in [("income", income_df), ("balance", balance_df), ("cashflow", cashflow_df)]:
        if stmt_df.empty:
            continue
        date_col = "report_date" if "report_date" in stmt_df.columns else "filing_date"
        if date_col not in stmt_df.columns:
            continue
        stmt_df[date_col] = pd.to_datetime(stmt_df[date_col])
        stmt_df = stmt_df.sort_values(date_col).drop_duplicates(subset=[date_col], keep="last")
        ncols = [c for c in stmt_df.select_dtypes(include=["number"]).columns if "date" not in c.lower()]
        if ncols:
            si = stmt_df.set_index(date_col)[ncols]
            ci = cache.index.union(si.index).sort_values()
            sa = si.reindex(ci).ffill().reindex(cache.index)
            new = [c for c in sa.columns if c not in cache.columns]
            if new:
                cache = cache.join(sa[new], how="left")

    # Inject shares_outstanding + market_cap from profile
    shares = profile.get("shares_outstanding")
    if shares and "shares_outstanding" not in cache.columns:
        cache["shares_outstanding"] = float(shares)
    if "close" in cache.columns and "shares_outstanding" in cache.columns:
        if "market_cap" not in cache.columns:
            cache["market_cap"] = cache["close"] * cache["shares_outstanding"]

    # Run derived variables
    from operator1.features.derived_variables import compute_derived_variables
    cache = compute_derived_variables(cache)

    cache.to_parquet(FIXTURE_DIR / "aapl_cache.parquet")
    print(f"Saved: aapl_cache.parquet ({len(cache)} rows x {len(cache.columns)} cols)")

    return 0


if __name__ == "__main__":
    sys.exit(main())
