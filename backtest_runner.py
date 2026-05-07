#!/usr/bin/env python3
"""Staged Backtest Runner for Roo Cloud / CI environments.

Breaks the Operator 1 pipeline into 3 independent stages that can be
run sequentially across separate process invocations, avoiding timeout
limits in constrained environments.

Usage:
    # Run all 3 stages in sequence:
    python backtest_runner.py --market us_sec_edgar --company AAPL \
        --end-date 2024-12-31 --years 2 --stage all

    # Run one stage at a time (feed-through via disk state):
    python backtest_runner.py --stage 1 --market us_sec_edgar --company AAPL --end-date 2024-12-31
    python backtest_runner.py --stage 2 --run-dir cache/backtest_AAPL_2024-12-31
    python backtest_runner.py --stage 3 --run-dir cache/backtest_AAPL_2024-12-31

    # Validate predictions against actual 2025 data:
    python backtest_runner.py --validate --run-dir cache/backtest_AAPL_2024-12-31

Stages:
    1 - Data fetch + cache build + feature engineering + survival + linked entities
    2 - Temporal models (regime, forecasting, Monte Carlo, walk-forward, etc.)
    3 - Profile build + report generation + prediction extraction

Each stage saves its full state to disk so the next stage can resume
without re-running prior work.
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import pickle
import sys
import time
from datetime import date, datetime, timedelta
from pathlib import Path

import numpy as np
import pandas as pd

# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-7s | %(name)s | %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("backtest_runner")


# ---------------------------------------------------------------------------
# State container -- serialized between stages
# ---------------------------------------------------------------------------

from operator1.pipeline_state import PipelineState

# BacktestState is now PipelineState. The backtest_runner uses PipelineState
# for all state management, with transient runtime objects stored as simple
# attributes (_secrets, _llm_client, _pit_client) that aren't serialized.
BacktestState = PipelineState  # backward compat alias


# ---------------------------------------------------------------------------
# Stage 1: Data fetch + cache build + features
# ---------------------------------------------------------------------------

def run_stage1(state: PipelineState, substage: str = "all") -> None:
    """Fetch data, build cache, compute features, survival, linked entities.

    When substage is "all", runs everything (original behavior).
    When substage is "1.1" through "1.6", runs only that portion and saves
    a checkpoint so the next sub-stage can resume from disk.

    Sub-stages:
        1.1  -- Data fetch (PIT client, profile, holders, statements, OHLCV)
        1.2  -- Reconciliation + pivot + frequency separation + cache build
        1.3  -- OHLCV fallback + holders + segments
        1.4a -- Cache build (OHLCV spine, merge, benchmark, IV, cross-asset, options)
        1.4b -- Macro + risk (macro fetch, quadrant, conflict, buying power, pre-ratios)
        1.5  -- Estimation + SIX proxies + derived variables + survival + FH
        1.6  -- Entity discovery + graph risk + sentiment + catalysts
        1.7  -- Adaptive calibration (thresholds, model params, windows, signal IC)
        1.8a -- Regime detection + enriched timeline (HMM/GMM/PELT/BCP/ChangeFinder)
        1.8b -- Finalization (linked conflict, aggregates, peer ranking, behavioral, normalization)
    """
    logger.info("=" * 60)
    logger.info("STAGE 1: Data Fetch + Cache Build + Features (substage=%s)", substage)
    logger.info("=" * 60)

    from operator1.secrets_loader import load_secrets
    state._secrets = load_secrets()

    # Parse end date
    end_dt = datetime.strptime(state.end_date, "%Y-%m-%d").date()
    start_dt = end_dt - timedelta(days=int(state.years * 365))
    logger.info("Window: %s to %s (%.1f years)", start_dt, end_dt, state.years)

    # Get market info
    from operator1.clients.pit_registry import get_market, get_macro_api_for_market
    market_info = get_market(state.market_id)
    if market_info is None:
        raise ValueError(f"Unknown market: {state.market_id}")
    logger.info("Market: %s (%s)", market_info.country, market_info.pit_api_name)

    # Create PIT client
    from operator1.clients.equity_provider import create_pit_client
    pit_client = create_pit_client(state.market_id, state._secrets)
    state._pit_client = pit_client

    # Search company
    results = pit_client.search_company(state.company)
    if not results:
        results = pit_client.list_companies(query=state.company)
    if results:
        company_info = results[0]
        logger.info("Company: %s (%s)", company_info.get("name"), company_info.get("ticker"))
    else:
        company_info = {"ticker": state.company, "name": state.company}

    ticker = company_info.get("ticker", "") or company_info.get("identifier", "")
    company_name = company_info.get("name", ticker)
    identifier = company_info.get("cik") or ticker or company_info.get("identifier", "")

    # Profile
    try:
        state.target_profile = pit_client.get_profile(identifier)
        state.target_profile.setdefault("name", company_name)
        state.target_profile.setdefault("ticker", ticker)
        state.target_profile.setdefault("country", market_info.country_code)
        state.target_profile.setdefault("market_id", state.market_id)
        state.target_profile.setdefault("pit_api", market_info.pit_api_name)
    except Exception as exc:
        logger.warning("Profile fetch failed: %s", exc)
        state.target_profile = {
            "name": company_name, "ticker": ticker,
            "country": market_info.country_code, "market_id": state.market_id,
        }

    # Supplement enrichment
    try:
        from operator1.clients.supplement import enrich_profile
        state.target_profile = enrich_profile(
            market_id=state.market_id, ticker=ticker,
            existing_profile=state.target_profile,
        )
    except Exception as exc:
        logger.debug("Supplement skipped: %s", exc)

    # -- CHECKPOINT 1.1: Profile + company search complete --
    state.save("1.1")
    logger.info("Checkpoint 1.1 saved (profile + company search)")
    if substage == "1.1":
        return

    # Fetch financial data (parallel) -- each download is a separate network call
    from concurrent.futures import ThreadPoolExecutor, as_completed

    income_df = balance_df = cashflow_df = quotes_df = pd.DataFrame()
    tasks = {
        "income": pit_client.get_income_statement,
        "balance": pit_client.get_balance_sheet,
        "cashflow": pit_client.get_cashflow_statement,
        "quotes": pit_client.get_quotes,
    }
    with ThreadPoolExecutor(max_workers=4) as ex:
        futures = {ex.submit(fn, identifier): lbl for lbl, fn in tasks.items()}
        for fut in as_completed(futures):
            lbl = futures[fut]
            try:
                r = fut.result()
                if lbl == "income": income_df = r
                elif lbl == "balance": balance_df = r
                elif lbl == "cashflow": cashflow_df = r
                elif lbl == "quotes": quotes_df = r
                logger.info("%s: %d rows", lbl.capitalize(), len(r))
            except Exception as exc:
                logger.warning("%s failed: %s", lbl, exc)

    # Data reconciliation
    try:
        from operator1.quality.data_reconciliation import reconcile_financial_data
        income_df, balance_df, cashflow_df, _ = reconcile_financial_data(
            income_df, balance_df, cashflow_df
        )
    except Exception as exc:
        logger.warning("Reconciliation failed: %s", exc)

    # Pivot to wide
    try:
        from operator1.clients.canonical_translator import pivot_to_canonical_wide
        for label, stmt_df in [("income", income_df), ("balance", balance_df), ("cashflow", cashflow_df)]:
            if stmt_df.empty:
                continue
            if "canonical_name" in stmt_df.columns and "value" in stmt_df.columns:
                wide = pivot_to_canonical_wide(stmt_df, date_col="report_date")
                if not wide.empty:
                    if "filing_date" in stmt_df.columns:
                        fd = (stmt_df.dropna(subset=["filing_date", "report_date"])
                              .sort_values("filing_date")
                              .drop_duplicates(subset=["report_date"], keep="last")
                              [["report_date", "filing_date"]])
                        wide = wide.merge(fd, on="report_date", how="left")
                    if label == "income": income_df = wide
                    elif label == "balance": balance_df = wide
                    else: cashflow_df = wide
                    logger.info("Pivoted %s: %d x %d", label, len(wide), len(wide.columns))
    except Exception as exc:
        logger.warning("Pivot failed: %s", exc)

    # Frequency separation: resolve mixed annual+quarterly statement data.
    # Mirrors main.py Step 3d -- prevents annual totals from being
    # distributed over quarterly windows (F5 bug).
    try:
        from operator1.clients.frequency_separator import (
            separate_by_period_type,
            build_highest_frequency_statement,
        )
        for label, stmt_ref in [("income", "income_df"), ("balance", "balance_df"), ("cashflow", "cashflow_df")]:
            stmt = locals()[stmt_ref]
            if stmt.empty:
                continue
            freq_groups = separate_by_period_type(stmt, market_id=state.market_id)
            if len(freq_groups) > 1:
                logger.info(
                    "Mixed-frequency %s: %s",
                    label, {k: len(v) for k, v in freq_groups.items()},
                )
                reconciled = build_highest_frequency_statement(freq_groups)
                if not reconciled.empty:
                    if label == "income":
                        income_df = reconciled
                    elif label == "balance":
                        balance_df = reconciled
                    else:
                        cashflow_df = reconciled
    except Exception as exc:
        logger.warning("Frequency separation failed: %s", exc)

    # Save raw statement DataFrames for multi-frequency Q/A direct construction
    state.income_df = income_df
    state.balance_df = balance_df
    state.cashflow_df = cashflow_df
    state.quotes_df = quotes_df

    # -- CHECKPOINT 1.2: Financial statements fetched --
    state.save("1.2")
    logger.info("Checkpoint 1.2 saved (financial statements)")
    if substage == "1.2":
        return

    # OHLCV fallback + holders + segments
    state.ohlcv_source_label = market_info.pit_api_name
    if quotes_df.empty and ticker:
        try:
            from operator1.clients.ohlcv_provider import fetch_ohlcv
            quotes_df = fetch_ohlcv(ticker, market_id=state.market_id)
            if not quotes_df.empty:
                state.ohlcv_source_label = "yfinance"
                logger.info("OHLCV from yfinance: %d rows", len(quotes_df))
        except Exception as exc:
            logger.warning("OHLCV fallback failed: %s", exc)

    # Holders
    try:
        if hasattr(pit_client, "get_holders"):
            state.target_holders = pit_client.get_holders(identifier) or []
    except Exception:
        pass
    try:
        if hasattr(pit_client, "get_insider_transactions"):
            state.target_insiders = pit_client.get_insider_transactions(identifier) or []
    except Exception:
        pass

    # Product segment extraction
    _seg_result: dict = {}
    try:
        if hasattr(pit_client, "extract_segment_data"):
            _seg_result = pit_client.extract_segment_data(identifier) or {}
            if _seg_result.get("n_segments", 0) >= 2:
                logger.info("Segments: %d segments extracted", _seg_result["n_segments"])
    except Exception as exc:
        logger.debug("Segment extraction skipped: %s", exc)

    # -- CHECKPOINT 1.3: OHLCV + holders + segments complete --
    state.quotes_df = quotes_df
    state.save("1.3")
    logger.info("Checkpoint 1.3 saved (OHLCV + holders + segments)")
    if substage == "1.3":
        return

    # Build cache
    if not quotes_df.empty:
        if "date" in quotes_df.columns:
            quotes_df["date"] = pd.to_datetime(quotes_df["date"])
            cache = quotes_df.set_index("date").sort_index()
        elif quotes_df.index.name == "date" or hasattr(quotes_df.index, "date"):
            cache = quotes_df.sort_index()
        else:
            cache = quotes_df.copy()
    else:
        idx = pd.date_range(start_dt, end_dt, freq="B", name="date")
        cache = pd.DataFrame(index=idx)

    # Benchmark returns for beta_252d
    try:
        from operator1.clients.ohlcv_provider import fetch_benchmark_returns
        _bench = fetch_benchmark_returns(state.market_id, years=int(state.years))
        if not _bench.empty:
            cache["benchmark_return_1d"] = _bench.reindex(cache.index, method="ffill")
            logger.info("Benchmark returns merged for beta_252d")
    except Exception as exc:
        logger.debug("Benchmark fetch skipped: %s", exc)

    # Fix 10: Implied volatility (IV-RV spread)
    try:
        from operator1.clients.ohlcv_provider import fetch_implied_volatility
        _iv_series = fetch_implied_volatility(ticker)
        if not _iv_series.empty:
            _iv_val = float(_iv_series.iloc[0])
            cache["iv30"] = _iv_val
            if "volatility_21d" in cache.columns:
                _rv = cache["volatility_21d"].iloc[-1] if cache["volatility_21d"].notna().any() else 0.0
                cache["iv_rv_spread"] = _iv_val - float(_rv)
    except Exception:
        pass

    # Fix 10: Sector leading indicators
    try:
        from operator1.clients.ohlcv_provider import fetch_sector_leading_indicators
        _sector = state.target_profile.get("sector", "")
        _leader_df = fetch_sector_leading_indicators(_sector, years=int(state.years))
        if not _leader_df.empty:
            _leader_aligned = _leader_df.reindex(cache.index, method="ffill")
            for _ldr_col in _leader_aligned.columns:
                _col_name = f"sector_leader_{_ldr_col}"
                if _col_name not in cache.columns:
                    cache[_col_name] = _leader_aligned[_ldr_col]
    except Exception:
        pass

    # Cross-asset sector rotation signals (Gap 3)
    try:
        from operator1.features.cross_asset_signals import compute_cross_asset_signals
        cache, _ca_result = compute_cross_asset_signals(
            cache, sector=state.target_profile.get("sector", ""),
        )
        if _ca_result and _ca_result.available:
            state.cross_asset_result = _ca_result
            logger.info("Cross-asset signals: rank=%s, disp=%s",
                        _ca_result.sector_rank_12m or "N/A",
                        f"{_ca_result.sector_dispersion:.5f}" if _ca_result.sector_dispersion else "N/A")
    except Exception:
        pass
    # Options-derived forward-looking signals (Gap 1)
    try:
        from operator1.features.options_signals import compute_options_signals
        cache, _opt_result = compute_options_signals(
            cache, ticker=ticker, market_id=state.market_id,
        )
        if _opt_result and _opt_result.available:
            state.options_signal_result = _opt_result
            logger.info(
                "Options signals: PCR=%.2f, RR25d=%s",
                _opt_result.put_call_ratio or 0,
                f"{_opt_result.risk_reversal_25d:.4f}" if _opt_result.risk_reversal_25d is not None else "N/A",
            )
    except Exception:
        pass

    # Merge statements
    try:
        from operator1.estimation.frequency_interpolator import interpolate_statement_to_daily
        _use_interp = True
    except ImportError:
        _use_interp = False

    _merged_cols = set(cache.columns)
    for label, stmt_df in [("income", income_df), ("balance", balance_df), ("cashflow", cashflow_df)]:
        if stmt_df.empty:
            continue
        try:
            for dc in ("report_date", "filing_date"):
                if dc in stmt_df.columns:
                    date_col = dc
                    break
            else:
                continue
            stmt_df[date_col] = pd.to_datetime(stmt_df[date_col])
            stmt_df = stmt_df.sort_values(date_col).drop_duplicates(subset=[date_col], keep="last")
            ncols = [c for c in stmt_df.select_dtypes(include=["number"]).columns
                     if c != date_col and "date" not in c.lower()]
            if not ncols:
                continue
            si = stmt_df.set_index(date_col)[ncols]
            if _use_interp and len(si) >= 2:
                sa, _ = interpolate_statement_to_daily(si, daily_index=cache.index, market_id=state.market_id)
            else:
                ci = cache.index.union(si.index).sort_values()
                sa = si.reindex(ci).ffill().reindex(cache.index)
            new = [c for c in sa.columns if c not in _merged_cols]
            if new:
                cache = cache.join(sa[new], how="left")
                _merged_cols.update(new)
            logger.info("Merged %s: %d new cols", label, len(new))
        except Exception as exc:
            logger.warning("Merge %s failed: %s", label, exc)

    # --- CompanyFacts fallback for missing critical balance sheet fields ---
    # edgartools XBRL extraction sometimes returns incomplete data due to
    # SEC rate limiting (429 responses) during concurrent ThreadPoolExecutor
    # calls. When critical balance sheet fields are missing, fill them from
    # the SEC CompanyFacts API which provides ALL reported XBRL facts.
    _critical_balance_fields = {
        "current_assets": ["AssetsCurrent"],
        "current_liabilities": ["LiabilitiesCurrent"],
        "cash_and_equivalents": [
            "CashCashEquivalentsRestrictedCashAndRestrictedCashEquivalents",
            "CashAndCashEquivalentsAtCarryingValue",
            "CashAndCashEquivalents",
        ],
        "total_liabilities": ["Liabilities"],
        "retained_earnings": ["RetainedEarningsAccumulatedDeficit"],
        "short_term_debt": ["ShortTermBorrowings", "DebtCurrent", "CommercialPaper"],
        "receivables": ["AccountsReceivableNetCurrent"],
        "inventory": ["InventoryNet"],
        "payables": ["AccountsPayableCurrent"],
        "shares_outstanding": [
            "EntityCommonStockSharesOutstanding",
            "CommonStockSharesOutstanding",
            "WeightedAverageNumberOfShareOutstandingBasicAndDiluted",
            "WeightedAverageNumberOfDilutedSharesOutstanding",
        ],
    }
    # P2/P3/P9: Extend CompanyFacts fallback to income statement fields.
    # Missing net_income/EPS disables PE anchor, full Piotroski, and HF
    # valuation tier. These are the highest-impact fields for prediction
    # accuracy improvement.
    _critical_income_fields = {
        "net_income": [
            "NetIncomeLoss",
            "ProfitLoss",
            "NetIncomeLossAvailableToCommonStockholdersBasic",
        ],
        "eps_diluted": [
            "EarningsPerShareDiluted",
            "EarningsPerShareBasic",
        ],
        "operating_income": [
            "OperatingIncomeLoss",
            "IncomeLossFromContinuingOperationsBeforeIncomeTaxesExtraordinaryItemsNoncontrollingInterest",
        ],
        "gross_profit": ["GrossProfit"],
        "ebit": [
            "OperatingIncomeLoss",
            "IncomeLossFromContinuingOperationsBeforeIncomeTaxesMinorityInterestAndIncomeLossFromEquityMethodInvestments",
        ],
        "interest_expense": [
            "InterestExpense",
            "InterestExpenseDebt",
            "InterestPaid",
        ],
    }
    _all_critical_fields = {**_critical_balance_fields, **_critical_income_fields}
    _missing_critical = [f for f in _all_critical_fields if f not in cache.columns
                         or (f in cache.columns and cache[f].isna().all())]
    if _missing_critical and state.market_id == "us_sec_edgar":
        try:
            import requests as _req
            _cik = state.target_profile.get("cik", "")
            if not _cik:
                _cik = pit_client._resolve_cik_fallback(identifier)
            _cik_padded = str(_cik).zfill(10)
            _headers = {"User-Agent": pit_client._user_agent, "Accept": "application/json"}
            _facts_resp = _req.get(
                f"https://data.sec.gov/api/xbrl/companyfacts/CIK{_cik_padded}.json",
                headers=_headers, timeout=30,
            )
            if _facts_resp.status_code == 200:
                _usgaap = _facts_resp.json().get("facts", {}).get("us-gaap", {})
                _dei = _facts_resp.json().get("facts", {}).get("dei", {})
                _filled = 0
                for _field, _concepts in _all_critical_fields.items():
                    if _field in cache.columns and not cache[_field].isna().all():
                        continue
                    for _concept in _concepts:
                        # shares_outstanding concepts live in DEI namespace;
                        # check DEI first, then us-gaap
                        _cdata = _dei.get(_concept, {}) or _usgaap.get(_concept, {})
                        # Field-specific unit keys
                        _unit_key = "USD/shares" if _field in ("eps_diluted",) else (
                            "shares" if _field == "shares_outstanding" else "USD"
                        )
                        _entries = _cdata.get("units", {}).get(_unit_key, [])
                        if not _entries and _unit_key == "USD/shares":
                            _entries = _cdata.get("units", {}).get("USD", [])
                        if not _entries:
                            continue
                        # Build a Series from filing data, aligned to report_date
                        _rows = []
                        for _e in _entries:
                            if _e.get("form") in ("10-K", "10-Q") and _e.get("val") is not None and _e.get("end"):
                                _rows.append({"date": pd.Timestamp(_e["end"]), "value": float(_e["val"])})
                        if _rows:
                            _fdf = pd.DataFrame(_rows).drop_duplicates(subset=["date"], keep="last").set_index("date").sort_index()
                            # Forward-fill to daily index
                            _ci = cache.index.union(_fdf.index).sort_values()
                            _aligned = _fdf["value"].reindex(_ci).ffill().reindex(cache.index)
                            if _aligned.notna().sum() > 0:
                                cache[_field] = _aligned
                                _filled += 1
                                logger.info("CompanyFacts filled '%s' from %s: %d non-NaN days",
                                           _field, _concept, _aligned.notna().sum())
                            break  # stop trying alternative concepts
                # P10: Auto-discovery of company-specific XBRL concepts.
                # When explicit concept names don't match, scan ALL us-gaap
                # concepts for substring matches. This catches non-standard
                # filers like Apple who use extended taxonomy entries.
                if _filled < len(_missing_critical):
                    _KEYWORD_MAP = {
                        "net_income": ["NetIncome", "ProfitLoss", "NetEarnings"],
                        "gross_profit": ["GrossProfit", "GrossMargin"],
                        "operating_income": ["OperatingIncome", "OperatingProfit"],
                        "interest_expense": ["InterestExpense", "InterestCost"],
                        "eps_diluted": ["EarningsPerShare"],
                        "current_assets": ["AssetsCurrent"],
                        "current_liabilities": ["LiabilitiesCurrent"],
                        "cash_and_equivalents": ["CashAndCashEquivalent", "CashCashEquivalent"],
                    }
                    _still_missing = [
                        f for f in _all_critical_fields
                        if (f not in cache.columns or cache[f].isna().all())
                        and f in _KEYWORD_MAP
                    ]
                    for _field in _still_missing:
                        _keywords = _KEYWORD_MAP[_field]
                        # Search all us-gaap concepts for keyword matches
                        for _concept_name, _cdata in _usgaap.items():
                            if any(kw.lower() in _concept_name.lower() for kw in _keywords):
                                _unit_key = "USD/shares" if _field == "eps_diluted" else "USD"
                                _entries = _cdata.get("units", {}).get(_unit_key, [])
                                if not _entries and _unit_key == "USD/shares":
                                    _entries = _cdata.get("units", {}).get("USD", [])
                                _rows = []
                                for _e in _entries:
                                    if _e.get("form") in ("10-K", "10-Q") and _e.get("val") is not None and _e.get("end"):
                                        _rows.append({"date": pd.Timestamp(_e["end"]), "value": float(_e["val"])})
                                if _rows:
                                    _fdf = pd.DataFrame(_rows).drop_duplicates(subset=["date"], keep="last").set_index("date").sort_index()
                                    _ci = cache.index.union(_fdf.index).sort_values()
                                    _aligned = _fdf["value"].reindex(_ci).ffill().reindex(cache.index)
                                    if _aligned.notna().sum() > 0:
                                        cache[_field] = _aligned
                                        _filled += 1
                                        logger.info("Auto-discovered '%s' from concept '%s': %d non-NaN days",
                                                   _field, _concept_name, _aligned.notna().sum())
                                        break  # found a match, stop searching

                if _filled > 0:
                    logger.info("CompanyFacts fallback: filled %d/%d missing critical fields", _filled, len(_missing_critical))
        except Exception as exc:
            logger.debug("CompanyFacts fallback failed: %s", exc)

    # Backtest date filter
    bt_end = pd.Timestamp(end_dt)
    bt_start = bt_end - pd.Timedelta(days=int(state.years * 365))
    cache = cache[(cache.index >= bt_start) & (cache.index <= bt_end)]
    logger.info("Cache after backtest filter: %d rows x %d cols", len(cache), len(cache.columns))

    # -- Inject shares_outstanding from profile if missing from cache --
    # shares_outstanding lives in the profile dict (from edgartools
    # company.shares_outstanding or yfinance info) but is NOT in any
    # financial statement DataFrame. Without it, market_cap, PE, EV,
    # fcf_yield, and the cash adequacy survival floor all break.
    if ("shares_outstanding" not in cache.columns
            or cache.get("shares_outstanding") is None
            or (cache["shares_outstanding"].isna().all() if "shares_outstanding" in cache.columns else True)):
        _shares = state.target_profile.get("shares_outstanding")
        if _shares is not None:
            try:
                _shares_val = float(_shares)
                if _shares_val > 0:
                    cache["shares_outstanding"] = _shares_val
                    logger.info(
                        "Injected shares_outstanding from profile: %.0f",
                        _shares_val,
                    )
            except (TypeError, ValueError):
                pass

    # Compute market_cap from close * shares_outstanding if not already present
    if ("close" in cache.columns
            and "shares_outstanding" in cache.columns
            and cache["shares_outstanding"].notna().any()):
        if "market_cap" not in cache.columns or cache["market_cap"].isna().all():
            cache["market_cap"] = cache["close"] * cache["shares_outstanding"]
            logger.info(
                "Computed market_cap: latest=%.0f",
                cache["market_cap"].dropna().iloc[-1] if cache["market_cap"].notna().any() else 0,
            )

    # -- CHECKPOINT 1.4a: Cache built (OHLCV + statements merged) --
    state.cache = cache
    state.save("1.4a")
    logger.info("Checkpoint 1.4a saved (cache built, pre-macro)")
    if substage == "1.4a":
        return

    # Macro data
    macro_api_info = get_macro_api_for_market(state.market_id)
    if macro_api_info:
        try:
            from operator1.clients.macro_provider import fetch_macro
            state.macro_data = fetch_macro(market_info.country_code, secrets=state._secrets, years=int(state.years))
            if state.macro_data:
                logger.info("Macro: %d indicators", len(state.macro_data))
        except Exception as exc:
            logger.warning("Macro fetch failed: %s", exc)

    if state.macro_data:
        try:
            from operator1.steps.macro_mapping import fetch_macro_data
            state.macro_dataset = fetch_macro_data(country_iso2=market_info.country_code, macro_raw=state.macro_data)
        except Exception:
            pass
        try:
            from operator1.features.macro_quadrant import compute_macro_quadrant
            cache, state.macro_quadrant_result = compute_macro_quadrant(cache, macro_data=state.macro_dataset)
        except Exception:
            pass

    # Conflict risk
    try:
        from operator1.features.conflict_risk import assess_conflict_risk, inject_conflict_risk_into_cache
        state.conflict_result = assess_conflict_risk(country_iso2=market_info.country_code, company_name=company_name)
        cache = inject_conflict_risk_into_cache(cache, state.conflict_result)
    except Exception:
        pass

    # Market buying power
    try:
        from operator1.features.market_buying_power import compute_market_buying_power
        cache, state.buying_power_result = compute_market_buying_power(
            cache, sector=state.target_profile.get("sector"),
            country_iso2=market_info.country_code, macro_data=state.macro_data,
        )
    except Exception:
        pass

    # Pre-estimation ratios
    try:
        from operator1.constants import EPSILON
        _pre_ratios = {
            "current_ratio": ("current_assets", "current_liabilities"),
            "interest_coverage": ("ebit", "interest_expense"),
            "cash_ratio": ("cash_and_equivalents", "current_liabilities"),
            "gross_margin": ("gross_profit", "revenue"),
            "net_margin": ("net_income", "revenue"),
        }
        for rn, (nc, dc) in _pre_ratios.items():
            if rn not in cache.columns and nc in cache.columns and dc in cache.columns:
                _den = cache[dc].astype(float).where(cache[dc].astype(float).abs() > EPSILON)
                cache[rn] = cache[nc].astype(float) / _den
    except Exception:
        pass

    # -- CHECKPOINT 1.4: Cache built + macro + conflict + pre-ratios --
    state.cache = cache
    state.save("1.4")
    state.save("1.4b")  # alias for clarity
    logger.info("Checkpoint 1.4 saved (cache + macro + conflict)")
    if substage in ("1.4", "1.4b"):
        return

    # Estimation
    # SIX proxy computation (Switzerland only -- must run BEFORE estimation)
    if state.market_id == "ch_six":
        try:
            from operator1.features.six_derived_proxies import (
                compute_six_proxies, seed_canonical_columns,
            )
            state.six_proxy_result = compute_six_proxies(cache, state.target_profile)
            if state.six_proxy_result.computed:
                seed_canonical_columns(cache, state.target_profile, state.six_proxy_result)
                logger.info(
                    "SIX proxies: %d columns, yield=%.2f%%",
                    state.six_proxy_result.n_proxies,
                    (state.six_proxy_result.dividend_yield or 0) * 100,
                )
        except Exception as exc:
            logger.warning("SIX proxy computation failed: %s", exc)

    try:
        from operator1.estimation.estimator import run_estimation
        from operator1.config_loader import load_config
        cfg = load_config("global_config")
        cache, state.estimation_coverage = run_estimation(cache, imputer_method=cfg.get("estimation_imputer", "bayesian_ridge"))
        logger.info("Estimation complete")
    except Exception as exc:
        logger.warning("Estimation failed: %s", exc)

    # Filing calendar
    try:
        from operator1.features.filing_calendar import analyze_filing_calendar
        state.filing_calendar_result = analyze_filing_calendar(cache, market_id=state.market_id)
    except Exception:
        pass

    # Event calendar features (Gap 4)
    try:
        from operator1.features.event_calendar import compute_event_calendar_features
        from datetime import datetime as _dt
        _ref = _dt.strptime(state.end_date, "%Y-%m-%d").date() if state.end_date else None
        cache, _evt_result = compute_event_calendar_features(
            cache, ticker=ticker, filing_calendar_result=state.filing_calendar_result,
            reference_date=_ref,
        )
        if _evt_result and _evt_result.available:
            state.event_calendar_result = _evt_result
    except Exception:
        pass

    # Derived variables
    try:
        from operator1.features.derived_variables import compute_derived_variables
        cache = compute_derived_variables(cache)
        logger.info("Features: %d cols", len(cache.columns))
    except Exception as exc:
        logger.warning("Features failed: %s", exc)

    # Private company proxies
    try:
        from operator1.features.private_company_proxies import is_private_company, compute_private_company_proxies, resolve_proxies
        state.is_private = is_private_company(cache)
        if state.is_private:
            cache = compute_private_company_proxies(cache)
            cache = resolve_proxies(cache)
    except Exception:
        pass

    # Institutional flow
    try:
        from operator1.features.institutional_flow import compute_institutional_flow
        cache = compute_institutional_flow(cache, insider_transactions=state.target_insiders)
    except Exception:
        pass

    # (Estimation + features done, continuing to survival...)

    # Survival mode
    from operator1.analysis.survival_mode import compute_company_survival_flag, compute_survival_probability
    from operator1.analysis.hierarchy_weights import compute_hierarchy_weights
    state.weights = {f"tier{i}": 20.0 for i in range(1, 6)}
    try:
        cache["company_survival_mode_flag"] = compute_company_survival_flag(cache)
        cache["survival_probability"] = compute_survival_probability(cache)
        try:
            from operator1.analysis.survival_mode import compute_cox_survival_score
            cox = compute_cox_survival_score(cache)
            if cox.notna().any():
                cache["cox_survival_score"] = cox
                sig = cache["survival_probability"]
                cache["survival_probability"] = 0.4 * sig + 0.6 * cox.fillna(sig)
        except Exception:
            pass
        # Gradient-based early warning: deterioration velocity
        try:
            from operator1.analysis.survival_mode import compute_survival_velocity
            _vel_flag, _vel_rate = compute_survival_velocity(cache)
            cache["survival_velocity_flag"] = _vel_flag
            cache["survival_deterioration_rate"] = _vel_rate
        except Exception:
            pass
        # Survival uncertainty bands (bootstrap P10/P90)
        try:
            from operator1.analysis.survival_mode import compute_survival_uncertainty
            _p10, _p90, _unc = compute_survival_uncertainty(cache, probability=cache.get("survival_probability"))
            cache["survival_probability_p10"] = _p10
            cache["survival_probability_p90"] = _p90
            cache["survival_uncertainty"] = _unc
        except Exception:
            pass
        cache = compute_hierarchy_weights(cache)
        for i in range(1, 6):
            col = f"hierarchy_tier{i}_weight"
            if col in cache.columns:
                state.weights[f"tier{i}"] = float(cache[col].iloc[-1])
        logger.info("Survival: %d flagged days", cache["company_survival_mode_flag"].sum())
    except Exception as exc:
        logger.warning("Survival failed: %s", exc)

    # Fuzzy protection
    try:
        from operator1.analysis.fuzzy_protection import compute_fuzzy_protection
        gdp_val = None
        if state.macro_data and state.macro_data.get("gdp") is not None:
            gs = state.macro_data["gdp"]
            if not gs.empty:
                gdp_val = float(gs.dropna().iloc[-1])
        cache = compute_fuzzy_protection(cache, sector=state.target_profile.get("sector"), gdp=gdp_val)
        state.fuzzy_result = {
            "mean_degree": float(cache["fuzzy_protection_degree"].mean()),
            "sector_score": float(cache["fuzzy_sector_score"].iloc[0]),
            "latest_label": cache["fuzzy_protection_label"].iloc[-1],
        }
    except Exception:
        pass

    # Financial health
    try:
        from operator1.models.financial_health import compute_financial_health
        cache, state.fh_result = compute_financial_health(cache, hierarchy_weights=state.weights)
        logger.info("FH: %.1f (%s)", state.fh_result.latest_composite, state.fh_result.latest_label)
    except Exception as exc:
        logger.warning("FH failed: %s", exc)

    # Vanity
    try:
        from operator1.analysis.vanity import compute_vanity_score
        cache = compute_vanity_score(cache)
    except Exception:
        pass

    # -- CHECKPOINT 1.5: Derived vars + survival + FH + vanity complete --
    state.cache = cache
    state.save("1.5")
    logger.info("Checkpoint 1.5 saved (features + survival + health)")
    if substage == "1.5":
        return

    # LLM client for entity discovery
    from operator1.clients.llm_factory import create_llm_client
    state._llm_client = create_llm_client(state._secrets)

    # Entity discovery + linked entity fetch
    if state._llm_client is not None:
        try:
            from operator1.steps.entity_discovery import discover_linked_entities
            discovery_result = discover_linked_entities(
                target_profile=state.target_profile,
                llm_client=state._llm_client,
                pit_client=pit_client,
                secrets=state._secrets,
            )
            if hasattr(discovery_result, "linked"):
                state.relationships = discovery_result.linked
            elif isinstance(discovery_result, dict):
                state.relationships = discovery_result
            logger.info("Linked entities: %d", sum(len(v) for v in state.relationships.values() if isinstance(v, list)))
        except Exception as exc:
            logger.warning("Entity discovery failed: %s", exc)

        # GLEIF
        try:
            from operator1.clients.gleif import fetch_corporate_structure
            gleif_id = state.target_profile.get("lei") or state.target_profile.get("name") or company_name
            cs = fetch_corporate_structure(gleif_id)
            if cs.available:
                parents = []
                for p in [cs.ultimate_parent, cs.direct_parent]:
                    if p:
                        parents.append({"name": p.name, "country": p.country, "lei": p.lei,
                                        "relationship": p.relationship, "relationship_group": "parent_companies"})
                if parents:
                    state.relationships["parent_companies"] = parents
                subs = [{"name": s.name, "country": s.country, "lei": s.lei,
                         "relationship": "subsidiary", "relationship_group": "subsidiaries"}
                        for s in cs.subsidiaries[:15]]
                if subs:
                    state.relationships["subsidiaries"] = subs
        except Exception:
            pass

        # Graph risk
        try:
            from operator1.models.graph_risk import compute_graph_risk_metrics
            from dataclasses import asdict
            rel_dicts = {}
            for grp, ents in state.relationships.items():
                if isinstance(ents, list):
                    rel_dicts[grp] = [asdict(e) if hasattr(e, "__dataclass_fields__") else e for e in ents]
            state.graph_risk_result = compute_graph_risk_metrics(
                target_isin=state.target_profile.get("isin", ticker),
                relationships=rel_dicts, target_cache=cache,
            )
        except Exception:
            pass

        # Game theory
        try:
            from operator1.models.game_theory import analyze_competitive_dynamics
            state.game_theory_result = analyze_competitive_dynamics(
                target_cache=cache, target_name=state.target_profile.get("name", "target"),
                competitor_caches=state.linked_caches or None,
            )
        except Exception:
            pass

    # News sentiment
    try:
        from operator1.features.news_sentiment import compute_news_sentiment
        cache, _sr = compute_news_sentiment(
            cache, llm_client=state._llm_client, symbol=ticker,
            market_id=state.market_id, company_name=company_name,
        )
        if _sr.n_articles_scored > 0:
            state.sentiment_result = {
                "n_articles_fetched": _sr.n_articles_fetched,
                "n_articles_scored": _sr.n_articles_scored,
                "scoring_method": _sr.scoring_method,
                "mean_sentiment": _sr.mean_sentiment,
                "latest_sentiment": _sr.latest_sentiment,
                "latest_label": _sr.latest_label,
            }
    except Exception:
        pass

    # Product catalysts (Fix 12: pass news_articles from sentiment step)
    try:
        from operator1.features.product_catalysts import detect_product_catalysts
        _news_articles = []
        try:
            _news_articles = _sr.articles if '_sr' in dir() and hasattr(_sr, 'articles') else []
        except Exception:
            _news_articles = []
        cache, state.catalyst_result = detect_product_catalysts(
            cache, profile=state.target_profile,
            news_articles=_news_articles if _news_articles else None,
        )
    except Exception:
        pass

    # Peer ranking
    if state.linked_caches:
        try:
            from operator1.features.peer_ranking import compute_peer_ranking
            cache, pr = compute_peer_ranking(cache, linked_caches=state.linked_caches)
            state.peer_ranking_result = {
                "n_peers": pr.n_peers, "n_variables_ranked": pr.n_variables_ranked,
                "latest_composite_rank": pr.latest_composite_rank,
                "latest_label": pr.latest_label,
            }
        except Exception:
            pass

    # -- CHECKPOINT 1.6: Entity discovery + sentiment complete --
    state.cache = cache
    state.save("1.6")
    logger.info("Checkpoint 1.6 saved (entities + sentiment)")
    if substage == "1.6":
        return

    # Adaptive thresholds
    _t_17a = time.time()
    try:
        from operator1.analysis.adaptive_thresholds import compute_adaptive_thresholds, threshold_set_to_survival_dict
        state.adaptive_thresholds = compute_adaptive_thresholds(
            cache, linked_caches=state.linked_caches or None,
            fh_composite_scores=cache.get("fh_composite_score"),
        )
        if state.adaptive_thresholds.adapted:
            adapted = threshold_set_to_survival_dict(state.adaptive_thresholds)
            cache["company_survival_mode_flag"] = compute_company_survival_flag(cache, thresholds=adapted)
            cache["survival_probability"] = compute_survival_probability(cache, thresholds=adapted)
            cache = compute_hierarchy_weights(cache)
    except Exception:
        pass
    logger.info("  1.7a adaptive_thresholds: %.1fs", time.time() - _t_17a)

    # Adaptive model parameters (Tier 2)
    _t_17b = time.time()
    try:
        from operator1.analysis.adaptive_model_params import (
            compute_blend_weights, compute_regime_risk_multiplier,
            compute_garman_klass_factor, compute_transition_halflife,
            compute_adaptive_mc_params, compute_adaptive_participation_rate,
            AdaptiveModelParams,
        )
        state.adaptive_model_params = AdaptiveModelParams()
        if "survival_probability" in cache.columns and "cox_survival_score" in cache.columns:
            _sig = cache.get("survival_probability")
            _cox = cache.get("cox_survival_score")
            _actual = cache.get("company_survival_mode_flag", pd.Series(0, index=cache.index))
            if _sig is not None and _cox is not None:
                w_sig, w_cox = compute_blend_weights(_sig, _cox, _actual)
                state.adaptive_model_params.blend_w_sig = w_sig
                state.adaptive_model_params.blend_w_cox = w_cox
                cache["survival_probability"] = w_sig * _sig + w_cox * _cox.fillna(_sig)
        state.adaptive_model_params.survival_risk_multiplier = compute_regime_risk_multiplier(
            state.regime_detector, cache,
        )
        state.adaptive_model_params.intraday_low_factor = compute_garman_klass_factor(cache)
        state.adaptive_model_params.mc_n_paths, state.adaptive_model_params.mc_is_tilt = (
            compute_adaptive_mc_params(cache)
        )
        state.adaptive_model_params.participation_rate = compute_adaptive_participation_rate(cache)
        state.adaptive_model_params.adapted = True
        logger.info("Adaptive model params computed")
    except Exception as exc:
        logger.debug("Adaptive model params skipped: %s", exc)
    logger.info("  1.7b adaptive_model_params: %.1fs", time.time() - _t_17b)

    # Adaptive windows (Tier 3)
    _t_17c = time.time()
    try:
        from operator1.analysis.adaptive_windows import (
            compute_adaptive_windows, compute_nn_hyperparams,
            compute_pattern_thresholds, compute_stale_threshold,
            AdaptiveTier3Params,
        )
        from operator1.analysis.adaptive_model_params import compute_effective_sample_size
        _freq = (
            state.filing_calendar_result.detected_frequency
            if state.filing_calendar_result is not None
            else "quarterly"
        )
        state.adaptive_tier3 = AdaptiveTier3Params()
        state.adaptive_tier3.windows = compute_adaptive_windows(_freq)
        state.adaptive_tier3.stale_threshold_days = compute_stale_threshold(_freq)
        _n_eff = compute_effective_sample_size(cache, "close")
        _n_feat = sum(1 for c in cache.columns if cache[c].dtype in ("float64", "float32") and cache[c].notna().sum() > 10)
        state.adaptive_tier3.nn_params = compute_nn_hyperparams(n_eff=_n_eff, n_features=min(_n_feat, 30))
        state.adaptive_tier3.pattern_body_threshold, state.adaptive_tier3.pattern_doji_threshold = (
            compute_pattern_thresholds(cache, lookback=state.adaptive_tier3.windows.medium)
        )
        state.adaptive_tier3.adapted = True
        logger.info("Adaptive windows computed")
    except Exception as exc:
        logger.debug("Adaptive windows skipped: %s", exc)
    logger.info("  1.7c adaptive_windows: %.1fs", time.time() - _t_17c)

    # Signal IC measurement
    _t_17d = time.time()
    try:
        from operator1.analysis.signal_ic import compute_signal_ic, get_ic_weighted_signals
        state.signal_ic_result = compute_signal_ic(cache)
        if state.signal_ic_result and state.signal_ic_result.available:
            logger.info(
                "Signal IC: %d strong, best=%s (IC=%.4f)",
                len(state.signal_ic_result.strong_signals),
                state.signal_ic_result.best_signal, state.signal_ic_result.best_ic,
            )
    except Exception as exc:
        logger.debug("Signal IC skipped: %s", exc)
    logger.info("  1.7d signal_ic: %.1fs", time.time() - _t_17d)

    # Fill actuals from previous prediction log
    try:
        from operator1.analysis.prediction_log import fill_actuals
        state.prediction_log_summary = fill_actuals(
            ticker=state.company, cache=cache,
            reference_date=datetime.strptime(state.end_date, "%Y-%m-%d").date() if state.end_date else None,
        )
    except Exception as exc:
        logger.debug("Prediction log fill skipped: %s", exc)

    # -- CHECKPOINT 1.7: Adaptive calibration + Signal IC complete --
    state.cache = cache
    state.save("1.7")
    logger.info("Checkpoint 1.7 saved (adaptive calibration)")
    if substage == "1.7":
        return

    # Enriched survival timeline
    try:
        from operator1.models.regime_detector import run_early_regime_detection
        from operator1.analysis.survival_timeline import compute_enriched_survival_timeline
        target_var = "equity_change_rate" if state.is_private else "return_1d"
        cache, state.early_regime_result = run_early_regime_detection(cache, target_variable=target_var)
        if state.early_regime_result and state.early_regime_result.fitted:
            state.regime_detector = state.early_regime_result.detector
        rl = state.early_regime_result.regime_labels if state.early_regime_result else None
        rc = state.early_regime_result.regime_confidence if state.early_regime_result else None
        state.enriched_timeline_result = compute_enriched_survival_timeline(cache, regime_labels=rl, regime_confidence=rc)
        if state.enriched_timeline_result and state.enriched_timeline_result.fitted:
            etl = state.enriched_timeline_result.timeline
            for col in ["regime_state", "survival_intensity", "regime_confidence",
                        "regime_switch", "regime_transition_prob", "survival_mode",
                        "survival_mode_code", "switch_point", "days_in_mode", "stability_score_21d"]:
                if col in etl.columns and col not in cache.columns:
                    cache[col] = etl[col].reindex(cache.index)
            logger.info("Enriched timeline: intensity=%.3f", state.enriched_timeline_result.mean_intensity)

        # ChangeFinder online change point scores (no look-ahead)
        try:
            from operator1.models.regime_detector import compute_online_change_scores
            _regime_target = "equity_change_rate" if state.is_private else "return_1d"
            _ret_for_cf = cache.get(_regime_target)
            if _ret_for_cf is not None and _ret_for_cf.notna().sum() > 30:
                _cf_scores = compute_online_change_scores(_ret_for_cf.fillna(0).values)
                if _cf_scores is not None:
                    cache["online_change_score"] = _cf_scores
        except Exception:
            pass
    except Exception as exc:
        logger.warning("Enriched timeline failed: %s", exc)

    # -- CHECKPOINT 1.8a: Regime detection + enriched timeline --
    state.cache = cache
    state.save("1.8a")
    logger.info("Checkpoint 1.8a saved (regime detection + enriched timeline)")
    if substage == "1.8a":
        return

    # Linked entity conflict propagation
    if state.conflict_result is not None and state.relationships:
        try:
            from operator1.features.conflict_risk import assess_linked_entity_conflict
            state.linked_conflict = assess_linked_entity_conflict(
                linked_entities=state.relationships,
                target_conflict=state.conflict_result,
            )
        except Exception:
            pass

    # Supply chain stress
    try:
        from operator1.features.conflict_risk import compute_supply_chain_stress
        _scs = compute_supply_chain_stress(
            conflict_result=state.conflict_result,
            linked_caches=state.linked_caches if state.linked_caches else None,
            relationships=state.relationships if state.relationships else None,
        )
        if _scs and _scs.get("available"):
            cache["supply_chain_stress_flag"] = int(_scs["supply_chain_stress_flag"])
            cache["supply_chain_stress_score"] = _scs["supply_chain_stress_score"]
    except Exception:
        pass

    # Linked aggregates (requires linked_caches from entity fetch above)
    if state.linked_caches:
        try:
            from operator1.features.linked_aggregates import compute_linked_aggregates, compute_relative_metrics
            _entity_groups = {}
            for grp, ents in state.relationships.items():
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
                    _entity_groups[grp] = ids
            state.linked_agg_df = compute_linked_aggregates(
                target_daily=cache, linked_daily=state.linked_caches,
                entity_groups=_entity_groups,
            )
            if state.linked_agg_df is not None and not state.linked_agg_df.empty:
                _new_agg = [c for c in state.linked_agg_df.columns if c not in cache.columns]
                if _new_agg:
                    cache = cache.join(state.linked_agg_df[_new_agg], how="left")
                logger.info("Linked aggregates: %d columns merged", len(_new_agg))
                # Relative metrics
                _rel = compute_relative_metrics(cache, state.linked_agg_df)
                if _rel is not None and not _rel.empty:
                    _new_rel = [c for c in _rel.columns if c not in cache.columns and _rel[c].notna().any()]
                    if _new_rel:
                        cache = cache.join(_rel[_new_rel], how="left")
        except Exception as exc:
            logger.debug("Linked aggregates skipped: %s", exc)

    # Ownership contagion
    if state.target_holders:
        try:
            from operator1.models.ownership_contagion import compute_ownership_contagion, inject_contagion_into_cache
            state.contagion_result = compute_ownership_contagion(
                target_holders=state.target_holders,
                competitor_holders={},
                cache=cache,
            )
            if state.contagion_result and state.contagion_result.available:
                cache = inject_contagion_into_cache(cache, state.contagion_result)
                logger.info("Ownership contagion: MHHI=%.3f", state.contagion_result.mhhi_delta)
        except Exception as exc:
            logger.debug("Ownership contagion skipped: %s", exc)

    # Save segment result for Stage 3 profile injection
    state.seg_result = _seg_result

    # Product segment metrics (for _extra_vars + MC concentration risk)
    if _seg_result and _seg_result.get("n_segments", 0) >= 2:
        try:
            from operator1.features.product_metrics import compute_product_metrics
            cache = compute_product_metrics(cache, _seg_result)
        except Exception as exc:
            logger.debug("Product metrics computation failed: %s", exc)

    # Geographic supply chain risk metrics (Gap 2)
    if _seg_result:
        try:
            from operator1.features.product_metrics import compute_geographic_metrics
            _geo_segs = _seg_result.get("geo_segments", {})
            _gleif_subs = state.relationships.get("subsidiaries", [])
            _sub_dicts = [
                s if isinstance(s, dict) else {"country": getattr(s, "country", "")}
                for s in _gleif_subs
            ]
            cache = compute_geographic_metrics(
                cache, geo_segments=_geo_segs, subsidiaries=_sub_dicts,
            )
        except Exception as exc:
            logger.debug("Geographic metrics computation failed: %s", exc)

    # Operational efficiency features
    try:
        from operator1.features.product_metrics import compute_operational_efficiency
        cache = compute_operational_efficiency(cache)
    except Exception:
        pass

    # Behavioral finance signals
    try:
        from operator1.features.behavioral_signals import compute_behavioral_signals
        cache = compute_behavioral_signals(cache)
    except Exception:
        pass

    # Complexity / entropy signals
    try:
        from operator1.features.complexity_signals import compute_complexity_signals
        cache = compute_complexity_signals(cache)
    except Exception:
        pass

    # Feature normalization (MUST run last before temporal models)
    try:
        from operator1.features.feature_normalization import compute_feature_normalization
        cache = compute_feature_normalization(cache)
    except Exception:
        pass

    state.cache = cache
    state.save("1.8")
    state.save("1")  # backward compat
    logger.info("STAGE 1 COMPLETE: %d rows x %d cols", len(cache), len(cache.columns))


# ---------------------------------------------------------------------------
# Stage 2: Temporal models
# ---------------------------------------------------------------------------

def run_stage2(state, stage_spec: str = "all") -> None:
    """Run temporal model stages via the staged runner.
    
    Args:
        state: PipelineState with cache populated from Stage 1.
        stage_spec: Which sub-stages to run. Examples:
            'all'  -- all stages 3-7 (default)
            '3'    -- all regime/causality/pattern sub-stages
            '3.1'  -- just regime detection
            '4.1'  -- just forecasting
            '5.4'  -- just Monte Carlo
            '6.11' -- just recursive predictions
            '7.4'  -- just multi-frequency pipeline
            '3-6'  -- stages 3 through 6
    """
    from operator1.stages.runner import run_stages
    
    # Ensure PipelineState has output_dir set for checkpoints
    if not state.output_dir:
        state.output_dir = 'cache'
    
    run_stages(state, stage_spec, save_checkpoints=True)

    # Post-pipeline hierarchy weight recalibration with forward pass errors
    if state.forward_pass_result is not None and hasattr(state.forward_pass_result, 'errors_by_tier'):
        try:
            _fp_errors = {
                f"tier{k}": v
                for k, v in state.forward_pass_result.errors_by_tier.items()
                if v
            }
            if _fp_errors:
                from operator1.analysis.hierarchy_weights import compute_hierarchy_weights
                state.cache = compute_hierarchy_weights(state.cache, forward_pass_errors=_fp_errors)
                for i in range(1, 6):
                    col = f"hierarchy_tier{i}_weight"
                    if col in state.cache.columns:
                        state.weights[f"tier{i}"] = float(state.cache[col].iloc[-1])
                logger.info("Hierarchy weights recalibrated with forward-pass error feedback")
        except Exception as _hw_exc:
            logger.debug("Hierarchy weight recalibration skipped: %s", _hw_exc)

    logger.info("Stage 2 complete via staged runner (spec=%s)", stage_spec)

def run_stage3(state: PipelineState) -> None:
    """Build profile, extract predictions, optionally validate against actuals."""
    logger.info("=" * 60)
    logger.info("STAGE 3: Profile Build + Prediction Extraction")
    logger.info("=" * 60)

    cache = state.cache
    from operator1.report.profile_builder import build_company_profile
    from dataclasses import asdict

    def _safe_float(val):
        import math
        if val is None: return None
        try:
            f = float(val)
            return None if (math.isnan(f) or math.isinf(f)) else round(f, 6)
        except (TypeError, ValueError):
            return None

    def _to_dict(obj):
        if obj is None: return None
        if isinstance(obj, dict): return obj
        if hasattr(obj, "__dataclass_fields__"):
            try: return asdict(obj)
            except Exception: pass
        if hasattr(obj, "__dict__"): return obj.__dict__.copy()
        return None

    def _available_dict(obj):
        d = _to_dict(obj)
        if d is not None: d.setdefault("available", True)
        return d

    fh_dict = _to_dict(state.fh_result)
    regime_result_dict = None
    if state.regime_detector is not None:
        try:
            rd = state.regime_detector.result
            regime_result_dict = {
                "hmm_fitted": rd.hmm_fitted, "gmm_fitted": rd.gmm_fitted,
                "pelt_fitted": rd.pelt_fitted, "bcp_fitted": rd.bcp_fitted,
            }
        except Exception:
            pass

    # Hedge Fund Analysis -- check if already run by mf.fuse (new flow).
    # Only run HF here if mf.fuse did NOT already produce a result (backward compat).
    hf_result = None
    _hf_from_fuse = state.profile.get("hedge_fund", {})
    if _hf_from_fuse.get("available"):
        logger.info("HF Analysis: using result from mf.fuse (already ran after fusion)")
    else:
        try:
            from operator1.hedge_fund.engine import run_hedge_fund_analysis
            hf_result = run_hedge_fund_analysis(
                income_df=state.income_df,
                balance_df=state.balance_df,
                cashflow_df=state.cashflow_df,
                cache=cache,
                target_profile=state.target_profile,
                forecast_result=state.forecast_result,
                mc_result=state.mc_result,
                scenario_result=state.scenario_result,
                multi_frequency_result=state.multi_frequency_result,
                signal_ic_result=state.signal_ic_result,
                filing_calendar_result=state.filing_calendar_result,
                fh_result=state.fh_result,
                peer_ranking_result=state.peer_ranking_result if isinstance(state.peer_ranking_result, dict) else None,
                sentiment_result=state.sentiment_result,
                survival_controller=state.survival_controller,
                linked_caches=state.linked_caches,
                macro_data=state.macro_data,
                seg_result=state.seg_result or None,
                prediction_log_summary=state.prediction_log_summary or None,
            )
            if hf_result and hf_result.available:
                logger.info("HF Analysis: grade=%s, signal=%+.2f",
                            hf_result.scorecard.investment_grade, hf_result.position.signal)
        except Exception as exc:
            logger.debug("HF analysis skipped: %s", exc)

    try:
        profile = build_company_profile(
            verified_target=state.target_profile,
            cache=cache,
            linked_aggregates=state.linked_agg_df,
            regime_result=regime_result_dict,
            forecast_result=state.forecast_result,
            mc_result=state.mc_result,
            prediction_result=state.pred_result,
            graph_risk_result=_available_dict(state.graph_risk_result),
            game_theory_result=_available_dict(state.game_theory_result),
            fuzzy_protection_result=_available_dict(state.fuzzy_result),
            financial_health_result=fh_dict,
            sentiment_result=state.sentiment_result,
            peer_ranking_result=state.peer_ranking_result if isinstance(state.peer_ranking_result, dict) else None,
            macro_quadrant_result=_to_dict(state.macro_quadrant_result),
        )
        state.profile = profile

        # Save profile
        run_dir = Path(state.output_dir)
        profile_path = run_dir / "company_profile.json"

        def _sanitize(obj):
            if isinstance(obj, dict): return {str(k): _sanitize(v) for k, v in obj.items()}
            if isinstance(obj, list): return [_sanitize(i) for i in obj]
            return obj

        # Inject USS data into profile
        if state.survival_controller is not None:
            profile["unified_survival_system"] = state.survival_controller.to_profile_dict()
        else:
            profile["unified_survival_system"] = {"available": False}

        # Hedge Fund Analysis -- use mf.fuse result if available, else Stage 3 result
        if _hf_from_fuse.get("available"):
            profile["hedge_fund"] = _hf_from_fuse
        elif hf_result is not None and hf_result.available:
            profile["hedge_fund"] = hf_result.to_profile_dict()
        else:
            profile["hedge_fund"] = {"available": False}
        if state.scenario_result is not None and state.scenario_result.available:
            profile["scenario_analysis"] = state.scenario_result.to_dict()
        else:
            profile["scenario_analysis"] = {"available": False}

        # Multi-frequency fusion
        if (state.multi_frequency_result is not None
                and hasattr(state.multi_frequency_result, "available")
                and state.multi_frequency_result.available):
            profile["multi_frequency"] = state.multi_frequency_result.to_profile_dict()
        else:
            profile["multi_frequency"] = {"available": False}

        # Signal IC results
        if state.signal_ic_result is not None and state.signal_ic_result.available:
            profile["signal_ic"] = state.signal_ic_result.to_profile_dict()
        else:
            profile["signal_ic"] = {"available": False}

        # Prediction log summary
        if state.prediction_log_summary is not None:
            profile["prediction_log"] = state.prediction_log_summary
        else:
            profile["prediction_log"] = {"n_filled": 0, "total_predictions": 0}

        # ---------------------------------------------------------------
        # Fix 13: Inject 17 missing profile sections (parity with main.py)
        # ---------------------------------------------------------------

        # 1. meta
        profile.setdefault("meta", {})
        profile["meta"]["market_id"] = state.market_id
        profile["meta"]["pit_source"] = True
        profile["meta"]["ohlcv_source"] = state.ohlcv_source_label
        profile["meta"]["is_private_company"] = state.is_private
        _has_ohlcv = "close" in cache.columns and cache["close"].notna().sum() >= 5
        profile["meta"]["has_ohlcv"] = _has_ohlcv

        # 2. enriched_survival_timeline
        if state.enriched_timeline_result and state.enriched_timeline_result.fitted:
            profile["enriched_survival_timeline"] = {
                "available": True,
                "regime_available": state.enriched_timeline_result.regime_available,
                "mean_intensity": state.enriched_timeline_result.mean_intensity,
                "combined_state_distribution": state.enriched_timeline_result.combined_state_distribution,
                "base_n_switches": state.enriched_timeline_result.base.n_switches,
                "base_mean_stability": state.enriched_timeline_result.base.mean_stability,
            }
        else:
            profile["enriched_survival_timeline"] = {"available": False}

        # 3. filing_calendar
        if state.filing_calendar_result is not None:
            profile["filing_calendar"] = {
                "available": True,
                "expected_frequency": state.filing_calendar_result.expected_frequency,
                "detected_frequency": state.filing_calendar_result.detected_frequency,
                "expected_filings_2yr": state.filing_calendar_result.expected_filings_2yr,
                "actual_filings_2yr": state.filing_calendar_result.actual_filings_2yr,
                "coverage_ratio": round(state.filing_calendar_result.coverage_ratio, 3),
                "latest_filing_age_days": state.filing_calendar_result.latest_filing_age_days,
                "is_stale": state.filing_calendar_result.is_stale,
                "gaps": state.filing_calendar_result.gaps,
            }
        else:
            profile["filing_calendar"] = {"available": False}

        # 4. economic_plane
        try:
            from operator1.analysis.economic_planes import classify_economic_plane
            profile["economic_plane"] = classify_economic_plane(
                sector=state.target_profile.get("sector"),
                industry=state.target_profile.get("industry"),
            )
        except Exception:
            profile["economic_plane"] = {"primary_plane": "unknown", "secondary_planes": []}

        # 5. corporate_structure
        _parent_list = state.relationships.get("parent_companies", [])
        _sub_list = state.relationships.get("subsidiaries", [])
        if _parent_list or _sub_list:
            _parent_dicts = [{"name": p.get("name", ""), "lei": p.get("lei", ""), "country": p.get("country", "")} for p in _parent_list if isinstance(p, dict)]
            _sub_dicts = [{"name": s.get("name", ""), "lei": s.get("lei", ""), "country": s.get("country", "")} for s in _sub_list if isinstance(s, dict)]
            _sub_countries = sorted({s.get("country", "") for s in _sub_list if isinstance(s, dict) and s.get("country")})
            profile["corporate_structure"] = {
                "available": True, "source": "gleif",
                "parent_companies": _parent_dicts, "n_parents": len(_parent_dicts),
                "subsidiaries": _sub_dicts[:10], "n_subsidiaries": len(_sub_dicts),
                "subsidiaries_countries": _sub_countries, "cross_border": len(_sub_countries) > 1,
            }
        else:
            profile["corporate_structure"] = {"available": False}

        # 6. institutional_holders
        if state.target_holders:
            profile["institutional_holders"] = {"available": True, "holders": state.target_holders[:10], "total_holders": len(state.target_holders)}
        else:
            profile["institutional_holders"] = {"available": False}

        # 7. institutional_ownership_analysis
        _inst_analysis = {"available": False}
        try:
            _has_contagion = state.contagion_result is not None and state.contagion_result.available
            _has_flow = "inst_flow_momentum" in cache.columns and cache["inst_flow_momentum"].notna().any()
            if _has_contagion or _has_flow:
                _inst_analysis = {"available": True}
                if _has_contagion:
                    _inst_analysis["contagion"] = {
                        "mhhi_delta": _safe_float(state.contagion_result.mhhi_delta),
                        "crowding_score": _safe_float(state.contagion_result.crowding_score),
                        "n_shared": state.contagion_result.n_shared_institutions,
                    }
                if _has_flow:
                    _latest = cache.iloc[-1]
                    _inst_analysis["flow"] = {
                        "momentum_latest": _safe_float(_latest.get("inst_flow_momentum")),
                        "momentum_label": str(_latest.get("inst_flow_momentum_label", "unknown")),
                    }
        except Exception:
            pass
        profile["institutional_ownership_analysis"] = _inst_analysis

        # 8. macro_indicators
        if state.macro_data:
            macro_summary = {}
            for indicator, series in state.macro_data.items():
                if series is not None and not series.empty:
                    macro_summary[indicator] = {
                        "latest_value": float(series.iloc[-1]),
                        "latest_date": str(series.index[-1].date()),
                        "observations": len(series),
                    }
            profile["macro_indicators"] = macro_summary

        # 9. market_buying_power
        if state.buying_power_result is not None and state.buying_power_result.available:
            profile["market_buying_power"] = {
                "available": True,
                "buying_power_index": state.buying_power_result.buying_power_index,
                "sector_demand_momentum": state.buying_power_result.sector_demand_momentum,
                "demand_risk_flag": state.buying_power_result.demand_risk_flag,
            }
        else:
            profile["market_buying_power"] = {"available": False}

        # 10. supply_chain_stress
        if "supply_chain_stress_flag" in cache.columns:
            profile["supply_chain_stress"] = {
                "available": True,
                "supply_chain_stress_flag": bool(cache["supply_chain_stress_flag"].iloc[-1]),
                "supply_chain_stress_score": _safe_float(cache.get("supply_chain_stress_score", pd.Series([0])).iloc[-1]),
            }
        else:
            profile["supply_chain_stress"] = {"available": False}

        # 10b. options_signals (Gap 1)
        if state.options_signal_result is not None and state.options_signal_result.available:
            profile["options_signals"] = state.options_signal_result.to_profile_dict()
        else:
            profile["options_signals"] = {"available": False}

        # 10c. cross_asset_signals (Gap 3)
        if state.cross_asset_result is not None and state.cross_asset_result.available:
            profile["cross_asset_signals"] = state.cross_asset_result.to_profile_dict()
        else:
            profile["cross_asset_signals"] = {"available": False}

        # 10d. event_calendar_signals (Gap 4)
        if state.event_calendar_result is not None and state.event_calendar_result.available:
            profile["event_calendar_signals"] = state.event_calendar_result.to_profile_dict()
        else:
            profile["event_calendar_signals"] = {"available": False}

        # 11. product_catalysts
        if state.catalyst_result is not None and state.catalyst_result.available:
            profile["product_catalysts"] = {
                "available": True,
                "catalyst_score": state.catalyst_result.catalyst_score,
                "catalyst_type": state.catalyst_result.catalyst_type,
                "rnd_acceleration": state.catalyst_result.rnd_acceleration,
                "news_catalyst_score": state.catalyst_result.news_catalyst_score,
            }
        else:
            profile["product_catalysts"] = {"available": False}

        # 11b. behavioral signals
        try:
            _beh_cols = [c for c in cache.columns if c.startswith("anchoring_") or c.startswith("disposition_") or c.startswith("attention_") or c.startswith("lottery_")]
            if _beh_cols:
                _latest = cache.iloc[-1]
                profile["behavioral_signals"] = {
                    "available": True,
                    "anchoring_52w_high": float(_latest.get("anchoring_52w_high", 0) or 0),
                    "anchoring_52w_low": float(_latest.get("anchoring_52w_low", 0) or 0),
                    "disposition_effect_proxy": float(_latest.get("disposition_effect_proxy", 0) or 0),
                    "attention_spike": int(_latest.get("attention_spike", 0) or 0),
                    "lottery_characteristics": float(_latest.get("lottery_characteristics", 0) or 0),
                }
            else:
                profile["behavioral_signals"] = {"available": False}
        except Exception:
            profile["behavioral_signals"] = {"available": False}

        # 11c. complexity signals
        try:
            _cpx_cols = [c for c in cache.columns if c.startswith("sample_entropy") or c.startswith("perm_entropy") or c.startswith("lz_") or c.startswith("approx_entropy")]
            if _cpx_cols:
                _latest = cache.iloc[-1]
                profile["complexity_signals"] = {
                    "available": True,
                    "sample_entropy_21d": float(_latest.get("sample_entropy_21d", 0) or 0),
                    "perm_entropy_21d": float(_latest.get("perm_entropy_21d", 0) or 0),
                    "lz_complexity": float(_latest.get("lz_complexity", 0) or 0),
                    "approx_entropy_price": float(_latest.get("approx_entropy_price", 0) or 0),
                }
            else:
                profile["complexity_signals"] = {"available": False}
        except Exception:
            profile["complexity_signals"] = {"available": False}

        # 12. product_segments
        _seg = state.seg_result
        if _seg and _seg.get("n_segments", 0) >= 2:
            _seg_rev = _seg.get("segments", {})
            _dominant = max(_seg_rev, key=_seg_rev.get) if _seg_rev else ""
            _total_rev = sum(_seg_rev.values()) if _seg_rev else 0
            _dom_pct = _seg_rev.get(_dominant, 0) / _total_rev if _total_rev > 0 else 0
            profile["product_segments"] = {
                "available": True, "segments": _seg.get("segments", {}),
                "n_segments": _seg.get("n_segments", 0),
                "dominant_segment": _dominant, "dominant_segment_pct": round(_dom_pct, 4),
            }
        else:
            profile["product_segments"] = {"available": False}

        # 13. model_diagnostics
        if state.model_diagnostics_result is not None and getattr(state.model_diagnostics_result, "available", False):
            profile["model_diagnostics"] = state.model_diagnostics_result.to_dict()
        else:
            profile["model_diagnostics"] = {"available": False}

        # 14. predicted_regime_shifts
        if state.regime_shift_result is not None and getattr(state.regime_shift_result, "available", False):
            profile["predicted_regime_shifts"] = state.regime_shift_result.to_dict()
        else:
            profile["predicted_regime_shifts"] = {"available": False}

        # 15. feature_selection
        if state.feature_selection_result is not None and getattr(state.feature_selection_result, "fitted", False):
            profile["feature_selection"] = {
                "available": True,
                "n_input": state.feature_selection_result.n_input,
                "n_output": state.feature_selection_result.n_output,
                "boruta_confirmed": state.feature_selection_result.boruta_confirmed[:20],
                "boruta_tentative": getattr(state.feature_selection_result, "boruta_tentative", [])[:10],
                "regime_selected": {
                    k: v[:10] for k, v in getattr(state.feature_selection_result, "regime_selected", {}).items()
                },
                "mrmr_selected": state.feature_selection_result.mrmr_selected[:15],
                "method_contributions": getattr(state.feature_selection_result, "method_contributions", {}),
            }
        else:
            profile["feature_selection"] = {"available": False}

        # 16. recursive_predictions
        if state.recursive_result is not None and getattr(state.recursive_result, "available", False):
            try:
                profile.setdefault("extended_models", {})["recursive_predictions"] = state.recursive_result.to_dict()
            except Exception:
                profile.setdefault("extended_models", {})["recursive_predictions"] = {"available": True}

        # 14. ohlc_predictions
        if state.ohlc_result is not None and state.ohlc_result.fitted:
            try:
                from operator1.models.ohlc_predictor import format_ohlc_for_profile
                profile["ohlc_predictions"] = format_ohlc_for_profile(state.ohlc_result)
            except Exception:
                profile["ohlc_predictions"] = {"available": False}
        else:
            profile["ohlc_predictions"] = {"available": False}

        # 15. predicted_regime_shifts (computed in Stage 2b, stored as local)
        profile.setdefault("predicted_regime_shifts", {"available": False})

        # 16. extended_models
        if "extended_models" not in profile:
            profile["extended_models"] = {}
        if state.transfer_entropy_result is not None:
            profile["extended_models"]["transfer_entropy"] = _available_dict(state.transfer_entropy_result)
        if state.cycle_result is not None:
            profile["extended_models"]["cycle_decomposition"] = _available_dict(state.cycle_result)
        if state.pattern_result is not None:
            profile["extended_models"]["candlestick_patterns"] = _available_dict(state.pattern_result)
        if state.copula_result is not None:
            profile["extended_models"]["copula"] = _available_dict(state.copula_result)
        if state.conformal_result is not None:
            profile["extended_models"]["conformal_prediction"] = _available_dict(state.conformal_result)
        if state.dtw_result is not None:
            profile["extended_models"]["dtw_analogs"] = _available_dict(state.dtw_result)
        if state.shap_result is not None:
            profile["extended_models"]["shap_explanations"] = _available_dict(state.shap_result)
        if state.sobol_result is not None:
            profile["extended_models"]["sobol_sensitivity"] = _available_dict(state.sobol_result)
        if state.particle_filter_result is not None:
            profile["extended_models"]["particle_filter"] = _available_dict(state.particle_filter_result)
        if state.transformer_result is not None:
            profile["extended_models"]["transformer"] = _available_dict(state.transformer_result)
        if state.granger_result is not None and state.granger_result.fitted:
            profile["extended_models"]["granger_causality"] = {
                "available": True,
                "n_significant_pairs": len(state.granger_result.significant_pairs),
                "network_density": state.granger_result.network_density,
            }
        if state.dual_regime_result is not None and state.dual_regime_result.fitted:
            profile["extended_models"]["dual_regimes"] = {"available": True}
        if state.walk_forward_result is not None:
            profile["extended_models"]["walk_forward"] = {
                "available": True,
                "overall_mae": getattr(state.walk_forward_result, "overall_mae", None),
                "overall_best_model": getattr(state.walk_forward_result, "overall_best_model", None),
            }
        if state.burnout_result is not None:
            profile["extended_models"]["burnout"] = {
                "available": True,
                "converged": state.burnout_result.converged,
                "calibrated": getattr(state.burnout_result, "calibrated", False),
            }
        if state.ga_result is not None and state.ga_result.fitted:
            profile["extended_models"]["genetic_optimizer"] = {"available": True}
        if state.tv_granger_result is not None and state.tv_granger_result.get("n_windows", 0) > 0:
            profile["extended_models"]["time_varying_granger"] = {
                "available": True, "n_windows": state.tv_granger_result["n_windows"],
            }
        if state.mv_mc_result is not None and state.mv_mc_result.get("available"):
            profile["extended_models"]["multivariate_monte_carlo"] = state.mv_mc_result

        # 17. synergies_applied
        if state.synergy_meta:
            profile["synergies_applied"] = {
                "cycle_features_added": state.synergy_meta.get("cycle_features_added", []),
                "variables_after_pruning": state.synergy_meta.get("variables_after_pruning", 0),
            }

        # Linked conflict injection
        if state.linked_conflict and isinstance(state.linked_conflict, dict):
            profile.setdefault("conflict_risk", {})["linked_conflict"] = state.linked_conflict

        # ---------------------------------------------------------------
        # End Fix 13
        # ---------------------------------------------------------------

        # Position signal
        try:
            _position_signal = 0.0
            _return_forecast = 0.0
            if state.pred_result is not None and hasattr(state.pred_result, "predictions"):
                _r5d = state.pred_result.predictions.get("return_5d", {}).get("5d")
                if _r5d is None:
                    _r5d = state.pred_result.predictions.get("close", {}).get("5d")
                if _r5d is not None:
                    pf = getattr(_r5d, "point_forecast", None)
                    if pf is not None:
                        _return_forecast = float(pf)
            _ic_conf = 1.0
            if state.signal_ic_result and state.signal_ic_result.available:
                _ic_conf = min(2.0, max(0.5, abs(state.signal_ic_result.best_ic) * 20))
            _surv_mult = 1.0
            _recovery_active = False
            if state.survival_controller is not None:
                if state.survival_controller.is_survival:
                    _surv_mult = 0.3
                _recovery = state.survival_controller.detect_recovery_signal()
                if _recovery.get("active"):
                    _surv_mult *= _recovery.get("position_signal_boost", 1.0)
                    _recovery_active = True
            _raw = _return_forecast * _ic_conf * _surv_mult
            _position_signal = max(-1.0, min(1.0, _raw * 100))
            _pos_label = "buy" if _position_signal > 0.3 else "sell" if _position_signal < -0.3 else "hold"
            profile["position_signal"] = {
                "available": True,
                "signal": round(_position_signal, 4),
                "label": _pos_label,
                "return_forecast": round(_return_forecast, 6),
                "ic_confidence": round(_ic_conf, 4),
                "survival_multiplier": round(_surv_mult, 4),
                "recovery_active": _recovery_active,
            }
        except Exception:
            profile["position_signal"] = {"available": False}

        profile = _sanitize(profile)
        with open(profile_path, "w") as f:
            json.dump(profile, f, indent=2, default=str)
        logger.info("Profile saved: %s", profile_path)

        # Store predictions to log
        if state.pred_result is not None and hasattr(state.pred_result, "predictions"):
            try:
                from operator1.analysis.prediction_log import store_predictions
                _model_used = {}
                if state.forecast_result is not None and hasattr(state.forecast_result, "model_used"):
                    _model_used = state.forecast_result.model_used
                _surv_regime = "normal"
                if "survival_regime" in cache.columns:
                    _sr = cache["survival_regime"].dropna()
                    if len(_sr) > 0:
                        _surv_regime = str(_sr.iloc[-1])
                store_predictions(
                    ticker=state.company,
                    predictions=state.pred_result.predictions,
                    model_used=_model_used,
                    survival_regime=_surv_regime,
                    run_date=datetime.strptime(state.end_date, "%Y-%m-%d").date() if state.end_date else None,
                )
            except Exception:
                pass
    except Exception as exc:
        logger.error("Profile build failed: %s", exc)
        import traceback
        traceback.print_exc()

    # Extract predictions summary
    preds_summary = extract_predictions(state)
    state.predictions_summary = preds_summary
    preds_path = Path(state.output_dir) / "predictions_summary.json"
    with open(preds_path, "w") as f:
        json.dump(preds_summary, f, indent=2, default=str)
    logger.info("Predictions saved: %s", preds_path)

    # Report generation (optional, requires LLM client)
    try:
        from operator1.report.report_generator import generate_all_reports
        from operator1.clients.llm_factory import create_llm_client

        llm_client = create_llm_client(state._secrets) if state._secrets else None
        report_dir = Path(state.output_dir) / "report"
        all_reports = generate_all_reports(
            profile=state.profile,
            llm_client=llm_client,
            cache=state.cache,
            output_dir=report_dir,
            generate_pdf=False,
        )
        for tier_name, report_output in all_reports.items():
            logger.info(
                "  %s report: %s",
                tier_name.capitalize(),
                report_output.get("markdown_path"),
            )
    except Exception as exc:
        logger.info("Report generation skipped: %s", exc)

    state.save("3")
    logger.info("STAGE 3 COMPLETE")


def extract_predictions(state: PipelineState) -> dict:
    """Extract all predictions into a flat summary dict for validation."""
    summary = {
        "market_id": state.market_id,
        "company": state.company,
        "end_date": state.end_date,
        "years": state.years,
    }

    cache = state.cache
    if cache is not None and len(cache) > 0:
        last = cache.iloc[-1]
        summary["last_close"] = float(last.get("close", 0)) if "close" in cache.columns else None
        summary["last_date"] = str(cache.index[-1].date())

    # Forecast results
    if state.forecast_result is not None:
        fc = {}
        for var, horizons in (state.forecast_result.forecasts or {}).items():
            if isinstance(horizons, dict):
                fc[var] = {h: float(v) if isinstance(v, (int, float)) else str(v) for h, v in horizons.items()}
        summary["forecasts"] = fc
        summary["model_used"] = state.forecast_result.model_used or {}

    # Monte Carlo
    if state.mc_result is not None:
        mc = {}
        if hasattr(state.mc_result, "survival_probability"):
            mc["survival_probability"] = {k: float(v) for k, v in (state.mc_result.survival_probability or {}).items()}
        if hasattr(state.mc_result, "max_drawdown_distribution") and state.mc_result.max_drawdown_distribution:
            mc["max_drawdown_distribution"] = state.mc_result.max_drawdown_distribution
        summary["monte_carlo"] = mc

    # Prediction aggregator
    if state.pred_result is not None and hasattr(state.pred_result, "predictions"):
        agg = {}
        for var, horizons in state.pred_result.predictions.items():
            if isinstance(horizons, dict):
                for h, hp in horizons.items():
                    key = f"{var}_{h}"
                    agg[key] = {
                        "point_forecast": getattr(hp, "point_forecast", None),
                        "lower_bound": getattr(hp, "lower_ci", None),
                        "upper_bound": getattr(hp, "upper_ci", None),
                    }
        summary["aggregated_predictions"] = agg

    # Financial health
    if state.fh_result is not None:
        summary["financial_health"] = {
            "composite": getattr(state.fh_result, "latest_composite", None),
            "label": getattr(state.fh_result, "latest_label", None),
        }

    # Survival
    if cache is not None and "survival_probability" in cache.columns:
        summary["survival_probability_latest"] = float(cache["survival_probability"].iloc[-1])

    # OHLC predictions
    if state.ohlc_result is not None and state.ohlc_result.fitted:
        nd = getattr(state.ohlc_result, "next_day", None)
        if nd is not None:
            summary["ohlc_next_day"] = {
                "open": getattr(nd, "open", None),
                "high": getattr(nd, "high", None),
                "low": getattr(nd, "low", None),
                "close": getattr(nd, "close", None),
            }
        # Also extract week/month/year series summaries
        for period, attr in [("next_week", "next_week"), ("next_month", "next_month"), ("next_year", "next_year")]:
            series = getattr(state.ohlc_result, attr, None)
            if series and len(series) > 0:
                last_candle = series[-1]
                summary[f"ohlc_{period}_end"] = {
                    "close": getattr(last_candle, "close", None),
                    "high": max((getattr(c, "high", 0) or 0) for c in series),
                    "low": min((getattr(c, "low", float("inf")) or float("inf")) for c in series),
                }

    return summary


# ---------------------------------------------------------------------------
# Validation: compare predictions to actual 2025 data
# ---------------------------------------------------------------------------

def validate_predictions(state: PipelineState) -> dict:
    """Fetch actual data for the prediction period and compare."""
    logger.info("=" * 60)
    logger.info("VALIDATION: Comparing predictions to actuals")
    logger.info("=" * 60)

    end_dt = datetime.strptime(state.end_date, "%Y-%m-%d").date()
    # Fetch 1 year of actual data after the backtest window
    actual_start = end_dt + timedelta(days=1)
    actual_end = end_dt + timedelta(days=365)

    ticker = state.target_profile.get("ticker", state.company)

    # Fetch actual OHLCV
    try:
        import yfinance as yf
        actual_df = yf.download(ticker, start=str(actual_start), end=str(min(actual_end, date.today())), progress=False)
        if actual_df.empty:
            logger.warning("No actual data available for validation period")
            return {"error": "no_actual_data"}
        # Flatten MultiIndex columns if present
        if isinstance(actual_df.columns, pd.MultiIndex):
            actual_df.columns = [c[0] if isinstance(c, tuple) else c for c in actual_df.columns]
        logger.info("Actual data: %d trading days (%s to %s)",
                     len(actual_df), actual_df.index[0].date(), actual_df.index[-1].date())
    except Exception as exc:
        logger.error("Failed to fetch actual data: %s", exc)
        return {"error": str(exc)}

    # Load predictions
    preds_path = Path(state.output_dir) / "predictions_summary.json"
    if preds_path.exists():
        with open(preds_path) as f:
            preds = json.load(f)
    else:
        preds = state.predictions_summary

    # Compute actual metrics
    validation = {
        "prediction_date": state.end_date,
        "actual_data_start": str(actual_df.index[0].date()),
        "actual_data_end": str(actual_df.index[-1].date()),
        "actual_trading_days": len(actual_df),
    }

    # Price comparison
    last_predicted_close = preds.get("last_close")
    if last_predicted_close and "Close" in actual_df.columns:
        actual_close = actual_df["Close"]

        # 1-day actual
        if len(actual_close) >= 1:
            validation["actual_1d_close"] = float(actual_close.iloc[0])
            validation["actual_1d_return"] = float((actual_close.iloc[0] - last_predicted_close) / last_predicted_close)

        # 5-day actual
        if len(actual_close) >= 5:
            validation["actual_5d_close"] = float(actual_close.iloc[4])
            validation["actual_5d_return"] = float((actual_close.iloc[4] - last_predicted_close) / last_predicted_close)

        # 21-day actual
        if len(actual_close) >= 21:
            validation["actual_21d_close"] = float(actual_close.iloc[20])
            validation["actual_21d_return"] = float((actual_close.iloc[20] - last_predicted_close) / last_predicted_close)

        # 252-day actual (1 year)
        if len(actual_close) >= 252:
            validation["actual_252d_close"] = float(actual_close.iloc[251])
            validation["actual_252d_return"] = float((actual_close.iloc[251] - last_predicted_close) / last_predicted_close)

        # Max drawdown
        cummax = actual_close.cummax()
        dd = (actual_close - cummax) / cummax
        validation["actual_max_drawdown"] = float(dd.min())

        # Actual volatility (21-day)
        returns = actual_close.pct_change().dropna()
        if len(returns) >= 21:
            validation["actual_volatility_21d"] = float(returns.iloc[:21].std())
        if len(returns) >= 252:
            validation["actual_volatility_252d"] = float(returns.std())

        # Latest actual close
        validation["actual_latest_close"] = float(actual_close.iloc[-1])
        validation["actual_total_return"] = float((actual_close.iloc[-1] - last_predicted_close) / last_predicted_close)

    # Compare with predictions
    comparisons = []
    forecasts = preds.get("forecasts", {})
    for var, horizons in forecasts.items():
        if var == "close" and isinstance(horizons, dict):
            for h, pred_val in horizons.items():
                actual_key = f"actual_{h}_close"
                if actual_key in validation:
                    error = float(pred_val) - validation[actual_key]
                    pct_error = error / validation[actual_key] * 100 if validation[actual_key] != 0 else 0
                    comparisons.append({
                        "variable": var, "horizon": h,
                        "predicted": float(pred_val),
                        "actual": validation[actual_key],
                        "error": round(error, 4),
                        "pct_error": round(pct_error, 2),
                    })

    agg_preds = preds.get("aggregated_predictions", {})
    for key, pred_data in agg_preds.items():
        if "close" in key and pred_data.get("point_forecast") is not None:
            h = key.replace("close_", "")
            actual_key = f"actual_{h}_close"
            if actual_key in validation:
                pf = float(pred_data["point_forecast"])
                actual = validation[actual_key]
                error = pf - actual
                comparisons.append({
                    "variable": "close (aggregated)", "horizon": h,
                    "predicted": pf, "actual": actual,
                    "error": round(error, 4),
                    "pct_error": round(error / actual * 100, 2) if actual != 0 else 0,
                    "lower_bound": pred_data.get("lower_bound"),
                    "upper_bound": pred_data.get("upper_bound"),
                    "actual_in_bounds": (
                        pred_data.get("lower_bound") is not None
                        and pred_data.get("upper_bound") is not None
                        and pred_data["lower_bound"] <= actual <= pred_data["upper_bound"]
                    ),
                })

    validation["comparisons"] = comparisons

    # Print summary
    logger.info("")
    logger.info("--- VALIDATION RESULTS ---")
    logger.info("Backtest window: %s to %s", preds.get("last_date", "?"), state.end_date)
    logger.info("Last predicted close: $%.2f", last_predicted_close or 0)
    for c in comparisons:
        bounds_str = ""
        if "actual_in_bounds" in c:
            lb = c.get('lower_bound')
            ub = c.get('upper_bound')
            if lb is not None and ub is not None:
                bounds_str = f" [{'IN' if c.get('actual_in_bounds') else 'OUT'} bounds: {lb:.2f}-{ub:.2f}]"
            else:
                bounds_str = ""
        logger.info("  %s %s: predicted=$%.2f, actual=$%.2f, error=%.2f%%$%s",
                     c["variable"], c["horizon"], c["predicted"], c["actual"], c["pct_error"], bounds_str)

    if "actual_max_drawdown" in validation:
        logger.info("  Actual max drawdown: %.2f%%", validation["actual_max_drawdown"] * 100)
    if "actual_volatility_21d" in validation:
        logger.info("  Actual 21d volatility: %.4f", validation["actual_volatility_21d"])
    if "actual_total_return" in validation:
        logger.info("  Actual total return to date: %.2f%%", validation["actual_total_return"] * 100)

    # Save validation
    val_path = Path(state.output_dir) / "validation_results.json"
    with open(val_path, "w") as f:
        json.dump(validation, f, indent=2, default=str)
    logger.info("Validation saved: %s", val_path)

    return validation


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main() -> int:
    parser = argparse.ArgumentParser(
        description="Staged Backtest Runner for Operator 1",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""\
Examples:
  # Full run (all stages):
  python backtest_runner.py --market us_sec_edgar --company AAPL --end-date 2024-12-31 --stage all

  # Stage by stage:
  python backtest_runner.py --stage 1 --market us_sec_edgar --company AAPL --end-date 2024-12-31
  python backtest_runner.py --stage 2 --run-dir cache/backtest_AAPL_2024-12-31
  python backtest_runner.py --stage 3 --run-dir cache/backtest_AAPL_2024-12-31

  # Validate predictions:
  python backtest_runner.py --validate --run-dir cache/backtest_AAPL_2024-12-31

  # Different company/window:
  python backtest_runner.py --market kr_dart --company 005930 --end-date 2024-06-30 --years 1.5 --stage all
""",
    )
    parser.add_argument("--stage", type=str, default="all",
                        help="Which stage to run. Top-level: 1 (data), 2 (temporal), 3 (profile), all. "
                             "Sub-stages: 3.1 (regime), 4.1 (forecast), 5.4 (MC), 6.11 (recursive), "
                             "7.4 (multi-freq), 7.5 (HF). Ranges: 3-6. "
                             "and 2a2 (forecasting). Use '2a' to run both, '2' for all sub-stages.")
    parser.add_argument("--market", type=str, default="us_sec_edgar",
                        help="Market ID (e.g. us_sec_edgar, kr_dart, jp_jquants)")
    parser.add_argument("--company", type=str, default="AAPL",
                        help="Company ticker or name")
    parser.add_argument("--end-date", type=str, default="2024-12-31",
                        help="Backtest end date (YYYY-MM-DD)")
    parser.add_argument("--years", type=float, default=2.0,
                        help="Lookback window in years")
    parser.add_argument("--run-dir", type=str, default="",
                        help="Run directory (auto-generated if not set)")
    parser.add_argument("--validate", action="store_true",
                        help="Validate predictions against actual data")
    parser.add_argument("--verbose", "-v", action="store_true",
                        help="Enable debug logging")
    args = parser.parse_args()

    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)

    # Auto-generate run dir
    if not args.run_dir:
        args.run_dir = f"cache/backtest_{args.company}_{args.end_date}"

    state = BacktestState()
    state.market_id = args.market
    state.company = args.company
    state.end_date = args.end_date
    state.years = args.years
    state.output_dir = args.run_dir

    def _find_latest_checkpoint(run_dir: str, target_sub: str) -> str | None:
        """Find the most recent checkpoint before the target sub-stage.

        Scans all state_*.pkl files in run_dir, sorts by modification time,
        and returns the latest one that is NOT the target sub-stage itself.
        This ensures sub-stage 5.1 loads from 4.1 (not from 1).
        """
        rd = Path(run_dir)
        if not rd.exists():
            return None
        candidates = []
        for pkl in rd.glob("state_*.pkl"):
            sub = pkl.stem.replace("state_", "")
            if sub != target_sub:
                candidates.append((pkl.stat().st_mtime, sub))
        if not candidates:
            return None
        candidates.sort(reverse=True)
        return candidates[0][1]

    # Validation mode
    if args.validate:
        state.load_checkpoint("3")
        validate_predictions(state)
        return 0

    # Build ordered list of stages to run.
    # Stage 1: Data fetch + cache build + features (backtest-specific)
    # Stage 2: Temporal models via staged runner (stages 3-7, shared with main.py)
    #   Supports sub-stage specs: 3.1, 4.1, 5.4, 6.11, 7.4, etc.
    # Stage 3: Profile build + prediction extraction (backtest-specific)
    # Stage 1 sub-stage IDs
    _STAGE1_SUBSTAGES = {"1.1", "1.2", "1.3", "1.4", "1.5", "1.6", "1.7", "1.8"}

    if args.stage == "all":
        stages = ["1", "2", "3"]
    elif args.stage in ("1", "2", "3"):
        stages = [args.stage]
    elif args.stage in _STAGE1_SUBSTAGES:
        # Stage 1 sub-stage: route to run_stage1 with substage param
        stages = [args.stage]
    else:
        # Sub-stage spec (e.g., "3.1", "4.1", "6.11", "3-6")
        # Route through Stage 2 with the spec passed to the staged runner
        stages = [f"2:{args.stage}"]

    _STAGE_FUNCS = {
        "1": lambda s: run_stage1(s, substage="all"),
        "1.1": lambda s: run_stage1(s, substage="1.1"),
        "1.2": lambda s: run_stage1(s, substage="1.2"),
        "1.3": lambda s: run_stage1(s, substage="1.3"),
        "1.4": lambda s: run_stage1(s, substage="1.4"),
        "1.5": lambda s: run_stage1(s, substage="1.5"),
        "1.6": lambda s: run_stage1(s, substage="1.6"),
        "1.7": lambda s: run_stage1(s, substage="1.7"),
        "1.8": lambda s: run_stage1(s, substage="1.8"),
        "2": lambda s: run_stage2(s, "all"),
        "3": lambda s: run_stage3(s),
    }
    # Map stages to their dependency for state loading
    _STAGE_DEPS = {
        "1": None,
        "1.1": None,
        "1.2": "1.1",
        "1.3": "1.2",
        "1.4": "1.3",
        "1.5": "1.4",
        "1.6": "1.5",
        "1.7": "1.6",
        "1.8": "1.7",
        "2": "1",   # temporal models depend on Stage 1 (data fetch)
        "3": None,  # profile build loads latest checkpoint dynamically
    }

    # Handle sub-stage routing: "2:3.1" means "load latest checkpoint, run sub-stage 3.1"
    for i, stage_key in enumerate(stages):
        if ":" in stage_key:
            parent, sub_spec = stage_key.split(":", 1)
            _STAGE_FUNCS[stage_key] = lambda s, spec=sub_spec: run_stage2(s, spec)
            # Find the best dependency: look for the previous sub-stage checkpoint
            # (e.g., 5.1 should load from 4.1, not from Stage 1)
            _best_dep = _find_latest_checkpoint(state.output_dir, sub_spec)
            _STAGE_DEPS[stage_key] = _best_dep or "1"
            stages[i] = stage_key

    for stage_key in stages:
        dep = _STAGE_DEPS[stage_key]
        # Stage 3 (profile build) needs the latest temporal checkpoint
        if dep is None and stage_key == "3":
            _latest = _find_latest_checkpoint(state.output_dir, "3")
            if _latest:
                logger.info("Stage 3: loading latest checkpoint '%s'", _latest)
                state.load_checkpoint(_latest)
        if dep is not None:
            # Load state from the dependency stage, with fallback chain
            dep_key = str(dep)
            _loaded = False
            _dep_pkl = Path(state.output_dir) / f"state_{dep_key}.pkl"
            if _dep_pkl.exists():
                state.load_checkpoint(dep_key)
                _loaded = True
            else:
                # Fallback: walk backwards through the dependency chain
                # to find the latest available checkpoint
                _fallback = dep_key
                while _fallback in _STAGE_DEPS and _STAGE_DEPS.get(_fallback) is not None:
                    _fallback = str(_STAGE_DEPS[_fallback])
                    _fb_pkl = Path(state.output_dir) / f"state_{_fallback}.pkl"
                    if _fb_pkl.exists():
                        logger.warning(
                            "Stage %s dep '%s' not found, falling back to '%s'",
                            stage_key, dep_key, _fallback,
                        )
                        state.load_checkpoint(_fallback)
                        _loaded = True
                        break
                if not _loaded:
                    logger.warning(
                        "No checkpoint found for stage %s (dep=%s), running with current state",
                        stage_key, dep_key,
                    )

        t0 = time.time()
        try:
            _STAGE_FUNCS[stage_key](state)
        except Exception as exc:
            logger.error("Stage %s FAILED: %s", stage_key, exc)
            import traceback
            traceback.print_exc()
            return 1

        elapsed = time.time() - t0
        logger.info("Stage %s completed in %.1fs", stage_key, elapsed)

    return 0


if __name__ == "__main__":
    sys.exit(main())
