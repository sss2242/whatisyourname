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

class BacktestState:
    """Mutable state bag passed between stages and persisted to disk."""

    def __init__(self) -> None:
        # Config
        self.market_id: str = ""
        self.company: str = ""
        self.end_date: str = ""
        self.years: float = 2.0
        self.run_dir: str = ""

        # Stage 1 outputs
        self.cache: pd.DataFrame | None = None
        self.target_profile: dict = {}
        self.relationships: dict = {}
        self.linked_caches: dict = {}
        self.macro_data: dict = {}
        self.macro_dataset = None
        self.weights: dict = {}
        self.fh_result = None
        self.conflict_result = None
        self.buying_power_result = None
        self.catalyst_result = None
        self.sentiment_result = None
        self.peer_ranking_result = None
        self.filing_calendar_result = None
        self.estimation_coverage = None
        self.fuzzy_result = None
        self.macro_quadrant_result = None
        self.graph_risk_result = None
        self.game_theory_result = None
        self.contagion_result = None
        self.six_proxy_result = None
        self.target_holders: list = []
        self.target_insiders: list = []
        self.signal_ic_result = None
        self.prediction_log_summary = None
        self.linked_agg_df = None
        self.linked_conflict = None
        self.enriched_timeline_result = None
        self.early_regime_result = None
        self.regime_detector = None
        self._is_private: bool = False
        self._adaptive_thresholds = None
        self._adaptive_model_params = None
        self._adaptive_tier3 = None
        self._ohlcv_source_label: str = ""
        self._seg_result: dict = {}
        self._mode_weights = None
        # Raw statement DataFrames (for multi-frequency Q/A direct construction)
        self._income_df: pd.DataFrame = pd.DataFrame()
        self._balance_df: pd.DataFrame = pd.DataFrame()
        self._cashflow_df: pd.DataFrame = pd.DataFrame()
        self._quotes_df: pd.DataFrame = pd.DataFrame()

        # USS (Unified Survival System) outputs
        self.survival_controller = None
        self.scenario_result = None

        # Stage 2 outputs
        self.forecast_result = None
        self.forward_pass_result = None
        self.walk_forward_result = None
        self.burnout_result = None
        self.mc_result = None
        self.pred_result = None
        self.transfer_entropy_result = None
        self.cycle_result = None
        self.pattern_result = None
        self.copula_result = None
        self.conformal_result = None
        self.dtw_result = None
        self.shap_result = None
        self.sobol_result = None
        self.particle_filter_result = None
        self.transformer_result = None
        self.dual_regime_result = None
        self.granger_result = None
        self.ga_result = None
        self.ohlc_result = None
        self._synergy_meta: dict = {}
        self._pattern_drift: float = 1.0
        self._retro_params = None
        self._tv_granger_result = None
        self._mv_mc_result = None
        self._extra_vars: list = []

        # Stage 2.5 outputs
        self.multi_frequency_result = None

        # Stage 3 outputs
        self.profile: dict = {}
        self.predictions_summary: dict = {}

        # Secrets (not persisted)
        self._secrets: dict = {}
        self._llm_client = None
        self._pit_client = None

    def save(self, stage: int) -> None:
        """Persist state after a stage completes."""
        run_dir = Path(self.run_dir)
        run_dir.mkdir(parents=True, exist_ok=True)

        # Save cache as parquet (fast, compact)
        if self.cache is not None:
            self.cache.to_parquet(run_dir / "cache.parquet")

        # Save linked caches
        if self.linked_caches:
            lc_dir = run_dir / "linked_caches"
            lc_dir.mkdir(exist_ok=True)
            for eid, lc in self.linked_caches.items():
                safe_name = eid.replace("/", "_").replace("\\", "_")[:50]
                lc.to_parquet(lc_dir / f"{safe_name}.parquet")
            # Save the ID mapping
            with open(lc_dir / "_ids.json", "w") as f:
                json.dump(list(self.linked_caches.keys()), f)

        # Save linked_agg_df
        if self.linked_agg_df is not None and not self.linked_agg_df.empty:
            self.linked_agg_df.to_parquet(run_dir / "linked_agg.parquet")

        # Save non-DataFrame state as pickle
        # Exclude large DataFrames and non-picklable objects
        state_dict = {}
        skip_keys = {
            "cache", "linked_caches", "linked_agg_df",
            "_secrets", "_llm_client", "_pit_client",
        }
        for k, v in self.__dict__.items():
            if k in skip_keys:
                continue
            try:
                pickle.dumps(v)
                state_dict[k] = v
            except (pickle.PicklingError, TypeError, AttributeError) as exc:
                logger.debug("Skipping non-picklable state key %s: %s", k, exc)

        with open(run_dir / f"state_stage{stage}.pkl", "wb") as f:
            pickle.dump(state_dict, f)

        # Save config as JSON for human readability
        config = {
            "market_id": self.market_id,
            "company": self.company,
            "end_date": self.end_date,
            "years": self.years,
            "run_dir": self.run_dir,
            "stage_completed": stage,
            "timestamp": datetime.now().isoformat(),
        }
        with open(run_dir / "config.json", "w") as f:
            json.dump(config, f, indent=2)

        logger.info("State saved to %s (stage %d)", run_dir, stage)

    def save_sub(self, sub_stage: str) -> None:
        """Persist state after a sub-stage completes (e.g. '2a', '2b')."""
        run_dir = Path(self.run_dir)
        run_dir.mkdir(parents=True, exist_ok=True)

        if self.cache is not None:
            self.cache.to_parquet(run_dir / "cache.parquet")

        if self.linked_caches:
            lc_dir = run_dir / "linked_caches"
            lc_dir.mkdir(exist_ok=True)
            for eid, lc in self.linked_caches.items():
                safe_name = eid.replace("/", "_").replace("\\", "_")[:50]
                lc.to_parquet(lc_dir / f"{safe_name}.parquet")
            with open(lc_dir / "_ids.json", "w") as f:
                json.dump(list(self.linked_caches.keys()), f)

        if self.linked_agg_df is not None and not self.linked_agg_df.empty:
            self.linked_agg_df.to_parquet(run_dir / "linked_agg.parquet")

        state_dict = {}
        skip_keys = {"cache", "linked_caches", "linked_agg_df", "_secrets", "_llm_client", "_pit_client"}
        for k, v in self.__dict__.items():
            if k in skip_keys:
                continue
            try:
                pickle.dumps(v)
                state_dict[k] = v
            except (pickle.PicklingError, TypeError, AttributeError):
                pass

        with open(run_dir / f"state_{sub_stage}.pkl", "wb") as f:
            pickle.dump(state_dict, f)

        config = {
            "market_id": self.market_id, "company": self.company,
            "end_date": self.end_date, "years": self.years,
            "run_dir": self.run_dir, "sub_stage_completed": sub_stage,
            "timestamp": datetime.now().isoformat(),
        }
        with open(run_dir / "config.json", "w") as f:
            json.dump(config, f, indent=2)

        logger.info("State saved to %s (sub-stage %s)", run_dir, sub_stage)

    def load_sub(self, sub_stage: str) -> None:
        """Load state from a previous sub-stage (e.g. '1', '2a', '2b')."""
        run_dir = Path(self.run_dir)

        config_path = run_dir / "config.json"
        if config_path.exists():
            with open(config_path) as f:
                config = json.load(f)
            self.market_id = config.get("market_id", self.market_id)
            self.company = config.get("company", self.company)
            self.end_date = config.get("end_date", self.end_date)
            self.years = config.get("years", self.years)

        cache_path = run_dir / "cache.parquet"
        if cache_path.exists():
            self.cache = pd.read_parquet(cache_path)
            logger.info("Loaded cache: %d rows x %d cols", len(self.cache), len(self.cache.columns))

        lc_dir = run_dir / "linked_caches"
        ids_path = lc_dir / "_ids.json"
        if ids_path.exists():
            with open(ids_path) as f:
                ids = json.load(f)
            self.linked_caches = {}
            for eid in ids:
                safe_name = eid.replace("/", "_").replace("\\", "_")[:50]
                pq = lc_dir / f"{safe_name}.parquet"
                if pq.exists():
                    self.linked_caches[eid] = pd.read_parquet(pq)

        lag_path = run_dir / "linked_agg.parquet"
        if lag_path.exists():
            self.linked_agg_df = pd.read_parquet(lag_path)

        # Try the exact sub-stage pickle, then fall back to stage number pickles
        pkl_candidates = [
            run_dir / f"state_{sub_stage}.pkl",
            run_dir / f"state_stage{sub_stage}.pkl",
        ]
        # Also try numeric stages for backward compat
        if sub_stage.isdigit():
            pkl_candidates.append(run_dir / f"state_stage{sub_stage}.pkl")

        for pkl_path in pkl_candidates:
            if pkl_path.exists():
                with open(pkl_path, "rb") as f:
                    state_dict = pickle.load(f)
                for k, v in state_dict.items():
                    if hasattr(self, k):
                        setattr(self, k, v)
                logger.info("Loaded state from %s (%d keys)", pkl_path.name, len(state_dict))
                break

        # Reload secrets (not persisted)
        try:
            from operator1.secrets_loader import load_secrets
            self._secrets = load_secrets()
        except Exception:
            pass

    def load(self, stage: int) -> None:
        """Load state from a previous stage."""
        run_dir = Path(self.run_dir)

        # Load config
        config_path = run_dir / "config.json"
        if config_path.exists():
            with open(config_path) as f:
                config = json.load(f)
            self.market_id = config.get("market_id", self.market_id)
            self.company = config.get("company", self.company)
            self.end_date = config.get("end_date", self.end_date)
            self.years = config.get("years", self.years)

        # Load cache
        cache_path = run_dir / "cache.parquet"
        if cache_path.exists():
            self.cache = pd.read_parquet(cache_path)
            logger.info("Loaded cache: %d rows x %d cols", len(self.cache), len(self.cache.columns))

        # Load linked caches
        lc_dir = run_dir / "linked_caches"
        ids_path = lc_dir / "_ids.json"
        if ids_path.exists():
            with open(ids_path) as f:
                ids = json.load(f)
            self.linked_caches = {}
            for eid in ids:
                safe_name = eid.replace("/", "_").replace("\\", "_")[:50]
                pq = lc_dir / f"{safe_name}.parquet"
                if pq.exists():
                    self.linked_caches[eid] = pd.read_parquet(pq)
            logger.info("Loaded %d linked caches", len(self.linked_caches))

        # Load linked_agg
        lag_path = run_dir / "linked_agg.parquet"
        if lag_path.exists():
            self.linked_agg_df = pd.read_parquet(lag_path)

        # Load pickle state (try latest stage first, then earlier)
        for s in range(stage, 0, -1):
            pkl_path = run_dir / f"state_stage{s}.pkl"
            if pkl_path.exists():
                with open(pkl_path, "rb") as f:
                    state_dict = pickle.load(f)
                for k, v in state_dict.items():
                    if hasattr(self, k):
                        setattr(self, k, v)
                logger.info("Loaded state from stage %d (%d keys)", s, len(state_dict))
                break


# ---------------------------------------------------------------------------
# Stage 1: Data fetch + cache build + features
# ---------------------------------------------------------------------------

def run_stage1(state: BacktestState) -> None:
    """Fetch data, build cache, compute features, survival, linked entities."""
    logger.info("=" * 60)
    logger.info("STAGE 1: Data Fetch + Cache Build + Features")
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

    # Product segment extraction (for product_metrics cache columns)
    _seg_result: dict = {}
    try:
        if hasattr(pit_client, "extract_segment_data"):
            _seg_result = pit_client.extract_segment_data(identifier) or {}
            if _seg_result.get("n_segments", 0) >= 2:
                logger.info("Segments: %d segments extracted", _seg_result["n_segments"])
    except Exception as exc:
        logger.debug("Segment extraction skipped: %s", exc)

    # Fetch financial data (parallel)
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

    # OHLCV fallback
    state._ohlcv_source_label = market_info.pit_api_name
    if quotes_df.empty and ticker:
        try:
            from operator1.clients.ohlcv_provider import fetch_ohlcv
            quotes_df = fetch_ohlcv(ticker, market_id=state.market_id)
            if not quotes_df.empty:
                state._ohlcv_source_label = "yfinance"
                logger.info("OHLCV from yfinance: %d rows", len(quotes_df))
        except Exception as exc:
            logger.warning("OHLCV fallback failed: %s", exc)

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

    # Save raw statement DataFrames for multi-frequency Q/A direct construction
    state._income_df = income_df
    state._balance_df = balance_df
    state._cashflow_df = cashflow_df
    state._quotes_df = quotes_df

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

    # Backtest date filter
    bt_end = pd.Timestamp(end_dt)
    bt_start = bt_end - pd.Timedelta(days=int(state.years * 365))
    cache = cache[(cache.index >= bt_start) & (cache.index <= bt_end)]
    logger.info("Cache after backtest filter: %d rows x %d cols", len(cache), len(cache.columns))

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
        state._is_private = is_private_company(cache)
        if state._is_private:
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

    # Adaptive thresholds
    try:
        from operator1.analysis.adaptive_thresholds import compute_adaptive_thresholds, threshold_set_to_survival_dict
        state._adaptive_thresholds = compute_adaptive_thresholds(
            cache, linked_caches=state.linked_caches or None,
            fh_composite_scores=cache.get("fh_composite_score"),
        )
        if state._adaptive_thresholds.adapted:
            adapted = threshold_set_to_survival_dict(state._adaptive_thresholds)
            cache["company_survival_mode_flag"] = compute_company_survival_flag(cache, thresholds=adapted)
            cache["survival_probability"] = compute_survival_probability(cache, thresholds=adapted)
            cache = compute_hierarchy_weights(cache)
    except Exception:
        pass

    # Adaptive model parameters (Tier 2)
    try:
        from operator1.analysis.adaptive_model_params import (
            compute_blend_weights, compute_regime_risk_multiplier,
            compute_garman_klass_factor, compute_transition_halflife,
            compute_adaptive_mc_params, compute_adaptive_participation_rate,
            AdaptiveModelParams,
        )
        state._adaptive_model_params = AdaptiveModelParams()
        if "survival_probability" in cache.columns and "cox_survival_score" in cache.columns:
            _sig = cache.get("survival_probability")
            _cox = cache.get("cox_survival_score")
            _actual = cache.get("company_survival_mode_flag", pd.Series(0, index=cache.index))
            if _sig is not None and _cox is not None:
                w_sig, w_cox = compute_blend_weights(_sig, _cox, _actual)
                state._adaptive_model_params.blend_w_sig = w_sig
                state._adaptive_model_params.blend_w_cox = w_cox
                cache["survival_probability"] = w_sig * _sig + w_cox * _cox.fillna(_sig)
        state._adaptive_model_params.survival_risk_multiplier = compute_regime_risk_multiplier(
            state.regime_detector, cache,
        )
        state._adaptive_model_params.intraday_low_factor = compute_garman_klass_factor(cache)
        state._adaptive_model_params.mc_n_paths, state._adaptive_model_params.mc_is_tilt = (
            compute_adaptive_mc_params(cache)
        )
        state._adaptive_model_params.participation_rate = compute_adaptive_participation_rate(cache)
        state._adaptive_model_params.adapted = True
        logger.info("Adaptive model params computed")
    except Exception as exc:
        logger.debug("Adaptive model params skipped: %s", exc)

    # Adaptive windows (Tier 3)
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
        state._adaptive_tier3 = AdaptiveTier3Params()
        state._adaptive_tier3.windows = compute_adaptive_windows(_freq)
        state._adaptive_tier3.stale_threshold_days = compute_stale_threshold(_freq)
        _n_eff = compute_effective_sample_size(cache, "close")
        _n_feat = sum(1 for c in cache.columns if cache[c].dtype in ("float64", "float32") and cache[c].notna().sum() > 10)
        state._adaptive_tier3.nn_params = compute_nn_hyperparams(n_eff=_n_eff, n_features=min(_n_feat, 30))
        state._adaptive_tier3.pattern_body_threshold, state._adaptive_tier3.pattern_doji_threshold = (
            compute_pattern_thresholds(cache, lookback=state._adaptive_tier3.windows.medium)
        )
        state._adaptive_tier3.adapted = True
        logger.info("Adaptive windows computed")
    except Exception as exc:
        logger.debug("Adaptive windows skipped: %s", exc)

    # Signal IC measurement
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

    # Fill actuals from previous prediction log
    try:
        from operator1.analysis.prediction_log import fill_actuals
        state.prediction_log_summary = fill_actuals(
            ticker=state.company, cache=cache,
            reference_date=datetime.strptime(state.end_date, "%Y-%m-%d").date() if state.end_date else None,
        )
    except Exception as exc:
        logger.debug("Prediction log fill skipped: %s", exc)

    # Enriched survival timeline
    try:
        from operator1.models.regime_detector import run_early_regime_detection
        from operator1.analysis.survival_timeline import compute_enriched_survival_timeline
        target_var = "equity_change_rate" if state._is_private else "return_1d"
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
            _regime_target = "equity_change_rate" if state._is_private else "return_1d"
            _ret_for_cf = cache.get(_regime_target)
            if _ret_for_cf is not None and _ret_for_cf.notna().sum() > 30:
                _cf_scores = compute_online_change_scores(_ret_for_cf.fillna(0).values)
                if _cf_scores is not None:
                    cache["online_change_score"] = _cf_scores
        except Exception:
            pass
    except Exception as exc:
        logger.warning("Enriched timeline failed: %s", exc)

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
    state._seg_result = _seg_result

    # Product segment metrics (for _extra_vars + MC concentration risk)
    if _seg_result and _seg_result.get("n_segments", 0) >= 2:
        try:
            from operator1.features.product_metrics import compute_product_metrics
            cache = compute_product_metrics(cache, _seg_result)
        except Exception as exc:
            logger.debug("Product metrics computation failed: %s", exc)

    state.cache = cache
    state.save_sub("1")
    logger.info("STAGE 1 COMPLETE: %d rows x %d cols", len(cache), len(cache.columns))


# ---------------------------------------------------------------------------
# Stage 2: Temporal models
# ---------------------------------------------------------------------------

def _init_extra_vars(state: BacktestState) -> None:
    """Build the _extra_vars list from cache columns.

    Mirrors main.py Step 6 _extra_vars construction (lines 2437-2473).
    """
    cache = state.cache
    _linked_prefixes = (
        "competitors_", "suppliers_", "customers_",
        "financial_institutions_", "sector_peers_", "industry_peers_",
        "rel_", "valuation_premium_",
    )
    # Columns derived from full-cache HMM that would cause look-ahead
    # bias if fed as features to the forward pass temporal models.
    _hmm_lookahead_cols = {
        "survival_intensity",       # blends rule-based (clean) + HMM (look-ahead)
        "regime_confidence",        # from HMM posteriors fitted on full cache
        "regime_transition_prob",   # from enriched timeline using HMM labels
    }
    state._extra_vars = [
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
                     "cannibalization_rate", "net_new_revenue_pct",
                     "network_effect_score", "input_cost_pressure",
                     "growth_runway_quarters", "maturity_concentration",
                     "estimated_market_share", "dominant_segment_growth")
            or any(c.startswith(p) for p in _linked_prefixes))
        and cache[c].dtype in ("float64", "float32", "int64")
        and not c.startswith("is_missing_")
        and c not in _hmm_lookahead_cols
    ]


def run_stage2a1(state: BacktestState) -> None:
    """Stage 2a1: Regime detection + causality + cycle/pattern + synergies."""
    logger.info("=" * 60)
    logger.info("STAGE 2a1: Regime + Causality + Patterns")
    logger.info("=" * 60)

    cache = state.cache
    if cache is None or cache.empty:
        raise ValueError("No cache data -- run Stage 1 first")

    _init_extra_vars(state)

    # Regime detection
    if state.regime_detector is None or "regime_label" not in cache.columns:
        try:
            from operator1.models.regime_detector import detect_regimes_and_breaks
            cache, state.regime_detector = detect_regimes_and_breaks(cache)
        except Exception as exc:
            logger.warning("Regime detection failed: %s", exc)

    # Dual regimes
    try:
        from operator1.models.regime_mixer import compute_dual_regimes
        state.dual_regime_result = compute_dual_regimes(cache)
    except Exception:
        pass

    # Granger causality
    try:
        from operator1.models.granger_causality import compute_granger_causality, prune_features_by_causality
        gc_vars = [c for c in cache.columns if cache[c].dtype in ("float64", "float32") and cache[c].notna().sum() > 50][:25]
        if len(gc_vars) >= 3:
            state.granger_result = compute_granger_causality(cache, variables=gc_vars)
            if state.granger_result and state.granger_result.fitted:
                keep = ["equity_value", "equity_change_rate", "financial_volatility"] if state._is_private else ["close", "return_1d", "volatility_21d"]
                state._extra_vars = prune_features_by_causality(state._extra_vars, state.granger_result, always_keep=keep)
    except Exception:
        pass

    # Transfer entropy
    try:
        from operator1.models.causality import compute_transfer_entropy
        te_vars = [c for c in cache.columns if cache[c].dtype in ("float64", "float32") and cache[c].notna().sum() > 30][:20]
        if len(te_vars) >= 2:
            state.transfer_entropy_result = compute_transfer_entropy(cache, variables=te_vars)
    except Exception:
        pass

    # Cycle decomposition
    try:
        from operator1.models.cycle_decomposition import run_cycle_decomposition
        cv = "equity_value" if state._is_private else "close"
        if cv not in cache.columns:
            cv = "revenue" if "revenue" in cache.columns else "total_equity"
        state.cycle_result = run_cycle_decomposition(cache, variable=cv)
    except Exception:
        pass

    # Pattern detection
    try:
        from operator1.models.pattern_detector import detect_patterns
        state.pattern_result = detect_patterns(cache)
    except Exception:
        pass

    # Pre-forecasting synergies
    try:
        from operator1.models.model_synergies import apply_pre_forecasting_synergies
        from operator1.analysis.economic_planes import classify_economic_plane
        ep = classify_economic_plane(sector=state.target_profile.get("sector"), industry=state.target_profile.get("industry"))
        cache, state._extra_vars, state._synergy_meta = apply_pre_forecasting_synergies(
            cache, cycle_result=state.cycle_result, granger_result=state.granger_result,
            transfer_entropy_result=state.transfer_entropy_result,
            linked_caches=state.linked_caches or None, extra_variables=state._extra_vars,
            economic_plane=ep,
        )
    except Exception:
        pass

    state.cache = cache
    state.save_sub("2a1")
    logger.info("STAGE 2a1 COMPLETE")


def run_stage2a2(state: BacktestState) -> None:
    """Stage 2a2: Forecasting (heavyweight, may take >5min)."""
    logger.info("=" * 60)
    logger.info("STAGE 2a2: Forecasting")
    logger.info("=" * 60)

    cache = state.cache
    if cache is None or cache.empty:
        raise ValueError("No cache data -- run Stage 2a1 first")

    if not state._extra_vars:
        _init_extra_vars(state)

    try:
        from operator1.models.forecasting import run_forecasting
        cache, state.forecast_result = run_forecasting(
            cache, extra_variables=state._extra_vars,
            windows=state._adaptive_tier3.windows if state._adaptive_tier3 is not None and state._adaptive_tier3.adapted else None,
        )
        logger.info("Forecasting complete")
    except Exception as exc:
        logger.warning("Forecasting failed: %s", exc)

    state.cache = cache
    state.save_sub("2a2")
    logger.info("STAGE 2a2 COMPLETE")


def run_stage2a(state: BacktestState) -> None:
    """Stage 2a: Regime + causality + forecasting (runs 2a1 + 2a2)."""
    run_stage2a1(state)
    run_stage2a2(state)


def run_stage2b(state: BacktestState) -> None:
    """Stage 2b: Forward pass + burnout + walk-forward + Monte Carlo."""
    logger.info("=" * 60)
    logger.info("STAGE 2b: Forward Pass + Burnout + Walk-Forward + MC")
    logger.info("=" * 60)

    cache = state.cache
    if cache is None or cache.empty:
        raise ValueError("No cache data -- run Stage 2a first")

    if not state._extra_vars:
        _init_extra_vars(state)

    from operator1.models.forecasting import run_forward_pass, run_burnout
    from operator1.models.monte_carlo import run_monte_carlo

    regime_labels = cache.get("regime_label") if "regime_label" in cache.columns else None

    # Forward pass
    try:
        state.forward_pass_result = run_forward_pass(
            cache, hierarchy_weights=state.weights, regime_labels=regime_labels,
            extra_variables=state._extra_vars,
        )
        logger.info("Forward pass: %d days", state.forward_pass_result.total_days)
    except Exception as exc:
        logger.warning("Forward pass failed: %s", exc)

    # Burnout
    try:
        state.burnout_result = run_burnout(
            cache, hierarchy_weights=state.weights, regime_labels=regime_labels,
            extra_variables=state._extra_vars, forward_pass_result=state.forward_pass_result,
        )
    except Exception:
        pass

    # Walk-forward
    try:
        from operator1.models.walk_forward import run_walk_forward
        from operator1.analysis.survival_timeline import compute_survival_timeline
        tl = compute_survival_timeline(cache)
        tl_df = tl.timeline if hasattr(tl, "timeline") else tl
        modes = tl_df["survival_mode"] if isinstance(tl_df, pd.DataFrame) and "survival_mode" in tl_df.columns else None
        switches = tl_df["switch_point"] if isinstance(tl_df, pd.DataFrame) and "switch_point" in tl_df.columns else None
        state.walk_forward_result = run_walk_forward(cache, modes, switches)
        if state.walk_forward_result and state.walk_forward_result.fitted:
            logger.info("Walk-forward: best=%s, MAE=%.6f",
                        state.walk_forward_result.overall_best_model,
                        state.walk_forward_result.overall_mae if not pd.isna(state.walk_forward_result.overall_mae) else 0)
    except Exception as exc:
        logger.warning("Walk-forward failed: %s", exc)

    # Monte Carlo
    try:
        mc_ret = "equity_change_rate" if state._is_private else "return_1d"
        _mc_n = (
            state._adaptive_model_params.mc_n_paths
            if state._adaptive_model_params is not None and getattr(state._adaptive_model_params, "adapted", False)
            else 10_000
        )
        _mc_tilt = (
            state._adaptive_model_params.mc_is_tilt
            if state._adaptive_model_params is not None and getattr(state._adaptive_model_params, "adapted", False)
            else 1.5
        )
        _mc_thresholds = None
        if state._adaptive_thresholds is not None and state._adaptive_thresholds.adapted:
            try:
                from operator1.analysis.adaptive_thresholds import threshold_set_to_mc_dict
                _mc_thresholds = threshold_set_to_mc_dict(state._adaptive_thresholds)
            except Exception:
                pass
        _burnout_dists = (
            state.burnout_result.regime_distributions
            if state.burnout_result is not None and getattr(state.burnout_result, "calibrated", False)
            else None
        )
        state.mc_result = run_monte_carlo(
            cache, returns_col=mc_ret,
            n_paths=_mc_n,
            importance_tilt=_mc_tilt,
            survival_thresholds=_mc_thresholds,
            burnout_distributions=_burnout_dists,
        )
        logger.info("Monte Carlo complete")
    except Exception as exc:
        logger.warning("Monte Carlo failed: %s", exc)

    # Fix 11: segment_hhi MC injection
    if state.mc_result is not None and "segment_hhi" in cache.columns:
        _seg_hhi = float(cache["segment_hhi"].iloc[-1]) if cache["segment_hhi"].notna().any() else 0
        state.mc_result.segment_hhi = _seg_hhi
        state.mc_result.concentration_risk_flag = _seg_hhi > 0.5

    # Fix 6: FixedShare/MCS pipeline (mode-conditioned weights)
    state._mode_weights = None
    try:
        if state.forward_pass_result is not None and hasattr(state.forward_pass_result, "predictions_log"):
            from operator1.models.walk_forward import (
                aggregate_forward_pass_errors,
                compute_mode_confidence_sets,
            )
            from operator1.models.prediction_aggregator import FixedShareForecaster

            _fp_log = getattr(state.forward_pass_result, "predictions_log", [])
            if _fp_log:
                _mode_errors = aggregate_forward_pass_errors(_fp_log, cache)
                if _mode_errors:
                    _mode_confidence_sets = compute_mode_confidence_sets(_mode_errors)
                    _all_model_names = set()
                    for mode_models in _mode_errors.values():
                        _all_model_names.update(mode_models.keys())
                    if _all_model_names:
                        _fixed_share = FixedShareForecaster(sorted(_all_model_names))
                        for mode_models in _mode_errors.values():
                            _min_len = min(len(v) for v in mode_models.values()) if mode_models else 0
                            for step in range(min(_min_len, 50)):
                                step_losses = {
                                    name: errs[step]
                                    for name, errs in mode_models.items()
                                    if step < len(errs)
                                }
                                _fixed_share.update(step_losses)
                        state._mode_weights = {"global": _fixed_share.get_weights()}
    except Exception:
        pass

    # Merge burn-out regime weights into mode_weights
    if (
        state.burnout_result is not None
        and getattr(state.burnout_result, "calibrated", False)
        and state.burnout_result.regime_weights
    ):
        if state._mode_weights is None:
            state._mode_weights = {}
        for regime, model_weights in state.burnout_result.regime_weights.items():
            state._mode_weights[regime] = model_weights

    # Regime shift prediction
    try:
        from operator1.models.regime_shift_predictor import predict_regime_shifts
        _mc_trans = state.mc_result.transition_matrix if state.mc_result is not None else None
        _mc_order = getattr(state.mc_result, "regime_order", None) if state.mc_result else None
        _stab = None
        if "stability_score_21d" in cache.columns:
            _ss = cache["stability_score_21d"].dropna()
            if len(_ss) > 0:
                _stab = float(_ss.iloc[-1])
        _thl = (
            state._adaptive_model_params.transition_halflife
            if state._adaptive_model_params is not None and getattr(state._adaptive_model_params, "adapted", False)
            else None
        )
        from datetime import datetime as _dt
        _ref = _dt.strptime(state.end_date, "%Y-%m-%d").date() if state.end_date else None
        regime_shift_result = predict_regime_shifts(
            cache, transition_matrix=_mc_trans, regime_order=_mc_order,
            stability_score=_stab, transition_halflife=_thl, reference_date=_ref,
        )
        if regime_shift_result and regime_shift_result.available:
            logger.info("Regime shift: P(exit 21d)=%.1f%%", regime_shift_result.prob_exit_21d * 100)
    except Exception as exc:
        logger.debug("Regime shift prediction skipped: %s", exc)

    # Copula
    try:
        from operator1.models.copula import run_copula_analysis
        state.copula_result = run_copula_analysis(cache)
    except Exception:
        pass

    state.cache = cache
    state.save_sub("2b")
    logger.info("STAGE 2b COMPLETE")


def run_stage2c(state: BacktestState) -> None:
    """Stage 2c: Transformer + Particle + Conformal + DTW + Aggregation + SHAP + Sobol + GA + OHLC."""
    logger.info("=" * 60)
    logger.info("STAGE 2c: Ensemble Models + Aggregation")
    logger.info("=" * 60)

    cache = state.cache
    if cache is None or cache.empty:
        raise ValueError("No cache data -- run Stage 2b first")

    from operator1.models.prediction_aggregator import run_prediction_aggregation

    # Transformer
    try:
        from operator1.models.transformer_forecaster import train_transformer
        tf_vars = [c for c in cache.columns if cache[c].dtype in ("float64", "float32") and cache[c].notna().sum() > 100][:15]
        if len(tf_vars) >= 2:
            state.transformer_result = train_transformer(cache, variables=tf_vars)
    except Exception:
        pass

    # Particle filter
    try:
        from operator1.models.particle_filter import run_particle_filter
        pf_vars = [v for v in ["cash_ratio", "free_cash_flow_ttm", "current_ratio", "debt_to_equity"] if v in cache.columns]
        if pf_vars:
            state.particle_filter_result = run_particle_filter(cache, variables=pf_vars)
    except Exception:
        pass

    # Conformal prediction (Fix 7: reuse forward pass calibrator when available)
    try:
        from operator1.models.conformal import ConformalPIDCalibrator, ConformalCalibrator, build_conformal_result
        if state.forecast_result is not None:
            calibrator = None
            if (state.forward_pass_result is not None
                    and hasattr(state.forward_pass_result, "conformal_calibrator")
                    and state.forward_pass_result.conformal_calibrator is not None):
                calibrator = state.forward_pass_result.conformal_calibrator
            else:
                try:
                    calibrator = ConformalPIDCalibrator(target_coverage=0.9)
                except Exception:
                    calibrator = ConformalCalibrator(coverage=0.9, adaptive=True)
                if hasattr(state.forecast_result, "residuals") and state.forecast_result.residuals:
                    for r in state.forecast_result.residuals:
                        calibrator.update(r)
            nested = {}
            if hasattr(state.forecast_result, "forecasts"):
                for var, vf in state.forecast_result.forecasts.items():
                    if isinstance(vf, dict):
                        nested[var] = {h: float(v) for h, v in vf.items() if isinstance(v, (int, float))}
            state.conformal_result = build_conformal_result(
                calibrator, forecasts=nested, horizons={"1d": 1, "5d": 5, "21d": 21, "252d": 252},
            )

            # Fix 8: G1 Quantile Regression for asymmetric intervals
            try:
                from operator1.models.conformal import QuantileRegressionCalibrator
                _qr_cal = QuantileRegressionCalibrator(lower_quantile=0.05, upper_quantile=0.95)
                _residuals_list = list(state.forecast_result.residuals) if hasattr(state.forecast_result, "residuals") and state.forecast_result.residuals else []
                if len(_residuals_list) >= _qr_cal._min_samples:
                    if _qr_cal.fit(_residuals_list):
                        if state.conformal_result is not None and hasattr(state.conformal_result, "intervals"):
                            for var, horizons_dict in state.conformal_result.intervals.items():
                                if isinstance(horizons_dict, dict):
                                    for h, interval in horizons_dict.items():
                                        pf = getattr(interval, "point_forecast", None) or getattr(interval, "forecast", None)
                                        if pf is not None:
                                            _lo, _hi = _qr_cal.predict_interval(float(pf))
                                            if _lo is not None and _hi is not None:
                                                if hasattr(interval, "lower"): interval.lower = _lo
                                                if hasattr(interval, "upper"): interval.upper = _hi
            except Exception:
                pass
    except Exception:
        pass

    # DTW analogs
    try:
        from operator1.models.dtw_analogs import find_historical_analogs
        dtw_vars = None
        if state._is_private:
            dtw_vars = [c for c in ["equity_value", "revenue", "net_income"] if c in cache.columns and cache[c].notna().sum() > 30]
        state.dtw_result = find_historical_analogs(
            cache, variables=dtw_vars, linked_caches=state.linked_caches or None,
        )
    except Exception:
        pass

    # Prediction aggregation (Fix 3: pass mode_weights, signal_ic, prediction_log)
    if state.forecast_result is not None:
        try:
            state.pred_result = run_prediction_aggregation(
                cache, state.forecast_result, state.mc_result,
                mode_weights=getattr(state, "_mode_weights", None),
                signal_ic_result=state.signal_ic_result,
                prediction_log_summary=state.prediction_log_summary,
                conformal_result=state.conformal_result,
                dual_regime_result=state.dual_regime_result,
                copula_result=state.copula_result,
                dtw_result=state.dtw_result,
                granger_result=state.granger_result,
                shap_result=state.shap_result,
                walk_forward_result=state.walk_forward_result,
            )
            logger.info("Predictions aggregated")
        except Exception as exc:
            logger.warning("Prediction aggregation failed: %s", exc)

    # Fix 9: USS aggregated prediction bounding
    if (state.survival_controller is not None
            and state.survival_controller.is_survival
            and state.pred_result is not None
            and hasattr(state.pred_result, "predictions")):
        try:
            from operator1.analysis.survival_regime_controller import bound_survival_forecast
            for var, horizons_dict in state.pred_result.predictions.items():
                if isinstance(horizons_dict, dict):
                    for h, hp in horizons_dict.items():
                        pf = getattr(hp, "point_forecast", None)
                        if pf is not None:
                            bounded = bound_survival_forecast(
                                var, float(pf), cache,
                                state.survival_controller.current_regime,
                            )
                            if bounded != float(pf):
                                hp.point_forecast = bounded
        except Exception:
            pass

    # SHAP
    try:
        from operator1.models.explainability import compute_shap_explanations
        if state.pred_result is not None:
            preds = {}
            if hasattr(state.pred_result, "predictions"):
                for var, hd in state.pred_result.predictions.items():
                    if isinstance(hd, dict):
                        hp = hd.get("1d")
                        if hp is not None:
                            pf = getattr(hp, "point_forecast", None)
                            if pf is not None:
                                preds[var] = pf
            state.shap_result = compute_shap_explanations(cache, predictions=preds)
    except Exception:
        pass

    # Sobol
    try:
        from operator1.models.sensitivity import run_sensitivity_analysis
        sv = "equity_change_rate" if state._is_private else "return_1d"
        state.sobol_result = run_sensitivity_analysis(cache, target_variable=sv)
    except Exception:
        pass

    # GA
    try:
        from operator1.models.genetic_optimizer import run_genetic_optimization
        state.ga_result = run_genetic_optimization(cache, forecast_result=state.forecast_result)
    except Exception:
        pass

    # Sobol -> Hierarchy feedback loop
    try:
        from operator1.models.sensitivity import adjust_hierarchy_from_sobol
        _adj = adjust_hierarchy_from_sobol(state.sobol_result, state.weights)
        if _adj != state.weights:
            state.weights = _adj
    except Exception:
        pass

    # Time-varying Granger causality
    try:
        from operator1.models.granger_causality import compute_time_varying_granger
        _gc_vars = [c for c in cache.columns if cache[c].dtype in ("float64", "float32") and cache[c].notna().sum() > 50][:15]
        if _gc_vars:
            state._tv_granger_result = compute_time_varying_granger(cache, variables=_gc_vars)
    except Exception:
        pass

    # Multivariate Monte Carlo
    try:
        from operator1.models.monte_carlo import run_multivariate_monte_carlo
        state._mv_mc_result = run_multivariate_monte_carlo(cache)
    except Exception:
        pass

    # OHLC predictor
    try:
        from operator1.models.ohlc_predictor import predict_ohlc_series
        from operator1.models.model_synergies import compute_pattern_drift_adjustment
        state._pattern_drift = compute_pattern_drift_adjustment(state.pattern_result)
        state.ohlc_result = predict_ohlc_series(
            cache, forecast_result=state.forecast_result, mc_result=state.mc_result,
            pattern_drift_multiplier=state._pattern_drift, cycle_result=state.cycle_result,
        )
        # Predicted OHLC patterns
        if state.ohlc_result and state.ohlc_result.fitted and state.pattern_result is not None:
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
                _pred_patterns = detect_patterns_on_predicted_ohlc(state.ohlc_result, _last_candle)
                if _pred_patterns:
                    state.pattern_result.predicted_patterns_week = _pred_patterns
            except Exception:
                pass
    except Exception:
        pass

    # E2: Anticipated survival (path-wise MC trigger checking)
    if state.mc_result is not None:
        try:
            from operator1.models.monte_carlo import compute_anticipated_survival
            for _h_label, _h_days in [("63d", 63), ("252d", 252)]:
                _as_prob = compute_anticipated_survival(cache, state.mc_result, horizon_days=_h_days)
                state.mc_result.anticipated_survival[_h_label] = _as_prob
        except Exception:
            pass

    state.cache = cache
    state.save_sub("2c")
    logger.info("STAGE 2c COMPLETE")


def run_stage2d(state: BacktestState) -> None:
    """Stage 2d: USS + Multi-frequency + Retroactive calibration + Diagnostics."""
    logger.info("=" * 60)
    logger.info("STAGE 2d: USS + Multi-Frequency + Calibration")
    logger.info("=" * 60)

    cache = state.cache
    if cache is None or cache.empty:
        raise ValueError("No cache data -- run Stage 2c first")

    # USS: Unified Survival System integration
    try:
        from operator1.analysis.survival_regime_controller import (
            SurvivalRegimeController, bound_forecast_dict,
        )
        state.survival_controller = SurvivalRegimeController.from_cache(cache)
        if state.survival_controller.is_survival:
            if state.forecast_result is not None and hasattr(state.forecast_result, "forecasts"):
                state.forecast_result.forecasts = bound_forecast_dict(
                    state.forecast_result.forecasts, cache,
                    state.survival_controller.current_regime,
                )
                logger.info("USS forecast bounding applied")
            from operator1.analysis.scenario_engine import run_scenario_engine
            state.scenario_result = run_scenario_engine(
                cache, regime=state.survival_controller.current_regime,
                n_paths=state.survival_controller.model_config.mc_n_paths,
            )
            if state.scenario_result and state.scenario_result.available:
                logger.info(
                    "USS scenario engine: orderly=%.1f%% / muddle=%.1f%% / catastrophic=%.1f%%",
                    state.scenario_result.orderly.survival_prob_252d * 100,
                    state.scenario_result.muddle_through.survival_prob_252d * 100,
                    state.scenario_result.catastrophic.survival_prob_252d * 100,
                )
    except Exception as exc:
        logger.debug("USS integration skipped: %s", exc)

    # Multi-frequency forecasting (per-frequency sub-stages)
    try:
        _run_bt_mf_all(state)
    except Exception as exc:
        logger.warning("Multi-frequency failed: %s", exc)

    # Retroactive calibration
    try:
        from operator1.analysis.retroactive_calibration import run_retroactive_calibration
        _entity_groups_for_retro = {}
        if state.relationships:
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
                    _entity_groups_for_retro[grp] = ids
        state._retro_params = run_retroactive_calibration(
            cache=cache,
            linked_caches=state.linked_caches if state.linked_caches else None,
            entity_groups=_entity_groups_for_retro if _entity_groups_for_retro else None,
            walk_forward_result=state.walk_forward_result,
            forecast_result=state.forecast_result,
            sobol_result=state.sobol_result,
            target_profile=state.target_profile,
        )
        if state._retro_params.n_calibrated > 0:
            logger.info("Retroactive calibration: %d groups calibrated", state._retro_params.n_calibrated)
    except Exception as exc:
        logger.warning("Retroactive calibration failed: %s", exc)

    # Model diagnostics
    try:
        from operator1.monitoring.model_diagnostics import compute_model_diagnostics
        _diag = compute_model_diagnostics(
            cache, forecast_result=state.forecast_result, mc_result=state.mc_result,
            copula_result=state.copula_result, granger_result=state.granger_result,
            cycle_result=state.cycle_result, dtw_result=state.dtw_result,
            conformal_result=state.conformal_result,
        )
        if _diag and _diag.available:
            logger.info("Model diagnostics: %d/%d on track", _diag.n_models_on_track, _diag.n_models_assessed)
    except Exception as exc:
        logger.debug("Model diagnostics failed: %s", exc)

    state.cache = cache
    state.save_sub("2d")
    logger.info("STAGE 2d COMPLETE")


# Keep backward compat: run_stage2 runs all sub-stages
def run_stage2(state: BacktestState) -> None:
    """Run all Stage 2 sub-stages sequentially (backward compat)."""
    run_stage2a(state)
    run_stage2b(state)
    run_stage2c(state)
    run_stage2d(state)


# ---------------------------------------------------------------------------
# Multi-frequency per-frequency sub-stages (used by run_stage2d and dispatch)
# ---------------------------------------------------------------------------

def _bt_mf_prep(state: BacktestState) -> None:
    """Build all ResampledCache objects and save to disk."""
    from operator1.features.frequency_resampler import (
        build_cache_from_raw_filings, detect_all_filing_frequencies,
        detect_native_filing_frequency, get_frequencies_slow_to_fast,
        is_annual_only_market, resample_cache_to_frequency,
    )
    cache = state.cache
    _bt_end = datetime.strptime(state.end_date, "%Y-%m-%d").date()
    _has_raw = any(not df.empty for df in [state._income_df, state._balance_df, state._cashflow_df])
    _is_ann = is_annual_only_market(state.market_id)
    freqs = get_frequencies_slow_to_fast()
    # Detect semi-annual
    if _has_raw and "Q" in freqs:
        _all_f = detect_all_filing_frequencies(state._income_df, state._balance_df, state._cashflow_df)
        _nat = detect_native_filing_frequency(state._income_df, state._balance_df, state._cashflow_df)
        if "Q" in _all_f and "S" in _all_f:
            if "S" not in freqs:
                freqs.insert(freqs.index("Q"), "S")
        elif _nat == "S" and "Q" not in _all_f:
            freqs = [("S" if f == "Q" else f) for f in freqs]
    # Save freq list
    import json as _json
    _mf_dir = Path(state.run_dir) / "mf"
    _mf_dir.mkdir(parents=True, exist_ok=True)
    (_mf_dir / "frequencies.json").write_text(_json.dumps(freqs))
    # Build each cache
    for freq in freqs:
        inc = state._income_df if not state._income_df.empty else None
        bal = state._balance_df if not state._balance_df.empty else None
        cf = state._cashflow_df if not state._cashflow_df.empty else None
        qt = state._quotes_df if not state._quotes_df.empty else None
        if freq in ("Q", "A", "W", "M", "S") and _has_raw:
            resampled = build_cache_from_raw_filings(
                income_df=inc, balance_df=bal, cashflow_df=cf, quotes_df=qt,
                frequency=freq, reference_date=_bt_end,
            )
        else:
            resampled = resample_cache_to_frequency(cache, frequency=freq, reference_date=_bt_end)
        if resampled.n_periods < 3:
            logger.info("[%s] Skipping -- %d periods (need 3+)", freq, resampled.n_periods)
            continue
        resampled.cache.to_parquet(_mf_dir / f"{freq}_cache.parquet")
        meta = {"frequency": resampled.frequency, "label": resampled.label,
                "n_periods": resampled.n_periods, "lookback_years": resampled.lookback_years,
                "is_partial_last_period": resampled.is_partial_last_period,
                "original_daily_rows": resampled.original_daily_rows,
                "resampled_rows": resampled.resampled_rows}
        (_mf_dir / f"{freq}_meta.json").write_text(_json.dumps(meta, indent=2))
        logger.info("[%s] Resampled: %d periods", freq, resampled.n_periods)


def _bt_mf_run_freq(state: BacktestState, freq: str) -> None:
    """Run the FULL temporal pipeline (2a1->2a2->2b->2c) for a single frequency.

    Each frequency gets its own BacktestState with the resampled cache.
    The existing stage functions operate on whatever state.cache contains,
    so no changes needed to them.  Results are saved to {run_dir}/mf/{freq}/.
    """
    import json as _json
    _mf_dir = Path(state.run_dir) / "mf"
    cp = _mf_dir / f"{freq}_cache.parquet"
    if not cp.exists():
        logger.info("[%s] No cache -- skipping", freq)
        return

    logger.info("=" * 60)
    logger.info("FULL TEMPORAL PIPELINE: frequency=%s", freq)
    logger.info("=" * 60)

    # Create a per-frequency state by copying the shared Stage 1 state
    freq_state = BacktestState()
    freq_state.market_id = state.market_id
    freq_state.company = state.company
    freq_state.end_date = state.end_date
    freq_state.years = state.years
    freq_state.run_dir = str(_mf_dir / freq)  # per-freq output dir
    freq_state._secrets = state._secrets
    freq_state._llm_client = state._llm_client
    freq_state._pit_client = state._pit_client

    # Copy Stage 1 results that don't depend on frequency
    freq_state.target_profile = state.target_profile
    freq_state.relationships = state.relationships
    freq_state.linked_caches = state.linked_caches
    freq_state.macro_data = state.macro_data
    freq_state.macro_dataset = state.macro_dataset
    freq_state.weights = state.weights.copy() if state.weights else {}
    freq_state.fh_result = state.fh_result
    freq_state.conflict_result = state.conflict_result
    freq_state.buying_power_result = state.buying_power_result
    freq_state.filing_calendar_result = state.filing_calendar_result
    freq_state.estimation_coverage = state.estimation_coverage
    freq_state.fuzzy_result = state.fuzzy_result
    freq_state.macro_quadrant_result = state.macro_quadrant_result
    freq_state.target_holders = state.target_holders
    freq_state.target_insiders = state.target_insiders
    freq_state.signal_ic_result = state.signal_ic_result
    freq_state._adaptive_thresholds = state._adaptive_thresholds
    freq_state._adaptive_model_params = state._adaptive_model_params
    freq_state._adaptive_tier3 = state._adaptive_tier3
    freq_state._is_private = state._is_private
    freq_state._ohlcv_source_label = state._ohlcv_source_label
    freq_state._seg_result = state._seg_result
    freq_state._income_df = state._income_df
    freq_state._balance_df = state._balance_df
    freq_state._cashflow_df = state._cashflow_df
    freq_state._quotes_df = state._quotes_df

    # Load the resampled cache for this frequency
    freq_state.cache = pd.read_parquet(cp)
    logger.info("[%s] Loaded resampled cache: %d rows x %d cols",
                freq, len(freq_state.cache), len(freq_state.cache.columns))

    # Run the full temporal pipeline (2a1 -> 2a2 -> 2b -> 2c)
    t0 = time.time()
    try:
        run_stage2a1(freq_state)
    except Exception as exc:
        logger.warning("[%s] Stage 2a1 failed: %s", freq, exc)

    try:
        run_stage2a2(freq_state)
    except Exception as exc:
        logger.warning("[%s] Stage 2a2 failed: %s", freq, exc)

    try:
        run_stage2b(freq_state)
    except Exception as exc:
        logger.warning("[%s] Stage 2b failed: %s", freq, exc)

    try:
        run_stage2c(freq_state)
    except Exception as exc:
        logger.warning("[%s] Stage 2c failed: %s", freq, exc)

    elapsed = time.time() - t0

    # Save full per-frequency state
    freq_state.save_sub("2c")

    logger.info("[%s] Full temporal pipeline complete: %.1fs", freq, elapsed)


def _bt_mf_fuse(state: BacktestState) -> None:
    """Fuse all per-frequency full states, then run USS + diagnostics + HF.

    1. Load each frequency's BacktestState from mf/{freq}/
    2. Extract FrequencyResult summaries for the fusion engine
    3. Fuse into FusedMultiFreqResult
    4. Run USS (forecast bounding + scenario engine)
    5. Run retroactive calibration + model diagnostics
    6. Run HF analysis on fused result
    7. Save everything back to the main state
    """
    import json as _json
    from operator1.models.frequency_fusion import fuse_multi_frequency_results
    from operator1.steps.multi_frequency_runner import (
        MultiFrequencyResult, FrequencyResult, FrequencyContext,
    )

    _mf_dir = Path(state.run_dir) / "mf"
    fp = _mf_dir / "frequencies.json"
    freqs = _json.loads(fp.read_text()) if fp.exists() else []

    # Load each frequency's state and build FrequencyResult summaries
    freq_states: dict[str, BacktestState] = {}
    freq_results: dict[str, FrequencyResult] = {}

    for freq in freqs:
        freq_dir = _mf_dir / freq
        state_pkl = freq_dir / "state_2c.pkl"
        if not state_pkl.exists():
            logger.info("[%s] No state_2c.pkl -- skipping", freq)
            continue

        # Load per-frequency state
        fs = BacktestState()
        fs.run_dir = str(freq_dir)
        fs.market_id = state.market_id
        fs.company = state.company
        fs.end_date = state.end_date
        fs.years = state.years
        fs.load_sub("2c")
        freq_states[freq] = fs

        # Build FrequencyResult summary from the full state
        _surv_prob = 1.0
        _surv_regime = "normal"
        _trend = "flat"
        _regime = "unknown"
        _forecast_summary = {}
        _wf_mae = None

        if fs.cache is not None:
            if "survival_probability" in fs.cache.columns:
                sp = fs.cache["survival_probability"].dropna()
                if len(sp) > 0:
                    _surv_prob = float(sp.iloc[-1])
            if "survival_regime" in fs.cache.columns:
                sr = fs.cache["survival_regime"].dropna()
                if len(sr) > 0:
                    _surv_regime = str(sr.iloc[-1])
            if "regime_label" in fs.cache.columns:
                rl = fs.cache["regime_label"].dropna()
                if len(rl) > 0:
                    _regime = str(rl.iloc[-1])
            # Trend from close
            if "close" in fs.cache.columns:
                closes = fs.cache["close"].dropna()
                if len(closes) >= 3:
                    first = closes.iloc[:len(closes)//3].mean()
                    last = closes.iloc[-len(closes)//3:].mean()
                    _trend = "up" if last > first * 1.05 else ("down" if last < first * 0.95 else "flat")

        if fs.forecast_result is not None and hasattr(fs.forecast_result, "forecasts"):
            for var, horizons in fs.forecast_result.forecasts.items():
                if isinstance(horizons, dict):
                    for h, val in horizons.items():
                        _forecast_summary[f"{var}_{h}"] = float(val) if val is not None else None

        if fs.walk_forward_result is not None and hasattr(fs.walk_forward_result, "overall_mae"):
            if not pd.isna(fs.walk_forward_result.overall_mae):
                _wf_mae = fs.walk_forward_result.overall_mae

        fr = FrequencyResult(
            frequency=freq, label=freq,
            n_periods=len(fs.cache) if fs.cache is not None else 0,
            elapsed_seconds=0,
            regime_label=_regime,
            survival_probability=_surv_prob,
            survival_regime=_surv_regime,
            trend_direction=_trend,
            forecast_summary=_forecast_summary,
            walk_forward_mae=_wf_mae,
            context_for_next=FrequencyContext(
                frequency=freq, trend_direction=_trend,
                survival_probability_latest=_surv_prob,
                survival_regime=_surv_regime,
            ),
        )
        freq_results[freq] = fr
        logger.info("[%s] Loaded: %d periods, regime=%s, survival=%.3f, forecasts=%d",
                    freq, fr.n_periods, _regime, _surv_prob, len(_forecast_summary))

    if not freq_results:
        logger.warning("No per-frequency results to fuse")
        return

    # Fuse all frequency results
    mfr = MultiFrequencyResult(
        results=freq_results,
        execution_order=list(freq_results.keys()),
        total_elapsed_seconds=0,
    )
    state.multi_frequency_result = fuse_multi_frequency_results(mfr)
    logger.info("MF fusion: %d freqs, survival=%.1f%%",
                state.multi_frequency_result.n_frequencies_used,
                state.multi_frequency_result.survival.fused_probability * 100)

    # Use the Daily frequency's state as the base for downstream (USS, HF, profile)
    # since it has the most complete data (504 days, all models)
    daily_state = freq_states.get("D")
    if daily_state is not None:
        # Copy Daily's temporal model results into the main state
        for attr in ("cache", "forecast_result", "forward_pass_result",
                     "walk_forward_result", "burnout_result", "mc_result",
                     "pred_result", "conformal_result", "copula_result",
                     "transformer_result", "particle_filter_result",
                     "shap_result", "sobol_result", "ga_result",
                     "ohlc_result", "dual_regime_result", "granger_result",
                     "transfer_entropy_result", "cycle_result", "pattern_result",
                     "regime_detector", "enriched_timeline_result",
                     "_mode_weights", "_synergy_meta", "_pattern_drift",
                     "_tv_granger_result", "_mv_mc_result", "_extra_vars"):
            val = getattr(daily_state, attr, None)
            if val is not None:
                setattr(state, attr, val)
        logger.info("Daily state merged into main state")

    # USS: forecast bounding + scenario engine
    cache = state.cache
    if cache is not None:
        try:
            from operator1.analysis.survival_regime_controller import (
                SurvivalRegimeController, bound_forecast_dict,
            )
            state.survival_controller = SurvivalRegimeController.from_cache(cache)
            if state.survival_controller.is_survival:
                if state.forecast_result is not None and hasattr(state.forecast_result, "forecasts"):
                    state.forecast_result.forecasts = bound_forecast_dict(
                        state.forecast_result.forecasts, cache,
                        state.survival_controller.current_regime,
                    )
                from operator1.analysis.scenario_engine import run_scenario_engine
                state.scenario_result = run_scenario_engine(
                    cache, regime=state.survival_controller.current_regime,
                    n_paths=state.survival_controller.model_config.mc_n_paths,
                )
        except Exception as exc:
            logger.debug("USS skipped: %s", exc)

    # Retroactive calibration
    try:
        from operator1.analysis.retroactive_calibration import run_retroactive_calibration
        _eg = {}
        if state.relationships:
            for grp, ents in state.relationships.items():
                if isinstance(ents, list):
                    ids = [
                        (e.get("isin", "") or e.get("ticker", "")) if isinstance(e, dict)
                        else (getattr(e, "isin", "") or getattr(e, "ticker", ""))
                        for e in ents
                    ]
                    _eg[grp] = [i for i in ids if i]
        state._retro_params = run_retroactive_calibration(
            cache=cache, linked_caches=state.linked_caches or None,
            entity_groups=_eg or None, walk_forward_result=state.walk_forward_result,
            forecast_result=state.forecast_result, sobol_result=state.sobol_result,
            target_profile=state.target_profile,
        )
    except Exception as exc:
        logger.debug("Retro-cal skipped: %s", exc)

    # Model diagnostics
    try:
        from operator1.monitoring.model_diagnostics import compute_model_diagnostics
        compute_model_diagnostics(
            cache, forecast_result=state.forecast_result, mc_result=state.mc_result,
            copula_result=state.copula_result, granger_result=state.granger_result,
            cycle_result=state.cycle_result, dtw_result=state.dtw_result,
            conformal_result=state.conformal_result,
        )
    except Exception as exc:
        logger.debug("Diagnostics skipped: %s", exc)

    # HF analysis (runs on fused result)
    try:
        from operator1.hedge_fund.engine import run_hedge_fund_analysis
        from operator1.clients.llm_factory import create_llm_client
        if state._llm_client is None:
            state._llm_client = create_llm_client(state._secrets)
        hf_result = run_hedge_fund_analysis(
            income_df=state._income_df,
            balance_df=state._balance_df,
            cashflow_df=state._cashflow_df,
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
        )
        if hf_result and hf_result.available:
            # Store HF result -- will be picked up by Stage 3
            state.profile["hedge_fund"] = hf_result.to_profile_dict()
            logger.info("HF Analysis: grade=%s, signal=%+.2f",
                        hf_result.scorecard.investment_grade, hf_result.position.signal)
    except Exception as exc:
        logger.warning("HF analysis failed: %s", exc)

    logger.info("Fusion + USS + HF complete")


def _run_bt_mf_all(state: BacktestState) -> None:
    """Run full per-frequency pipeline: prep -> all freqs -> fuse + USS + HF."""
    import json as _json
    _bt_mf_prep(state)
    _mf_dir = Path(state.run_dir) / "mf"
    fp = _mf_dir / "frequencies.json"
    freqs = _json.loads(fp.read_text()) if fp.exists() else []
    for freq in freqs:
        if (_mf_dir / f"{freq}_cache.parquet").exists():
            _bt_mf_run_freq(state, freq)
    _bt_mf_fuse(state)


# Per-frequency dispatch wrappers for CLI --stage
def _run_bt_mf_prep(state: BacktestState) -> None:
    _bt_mf_prep(state)
    state.save_sub("2d.mf.prep")

def _run_bt_mf_A(state: BacktestState) -> None:
    _bt_mf_run_freq(state, "A")
    state.save_sub("2d.mf.A")

def _run_bt_mf_Q(state: BacktestState) -> None:
    # Run Q and/or S depending on what prep detected
    import json as _json
    _mf_dir = Path(state.run_dir) / "mf"
    fp = _mf_dir / "frequencies.json"
    freqs = _json.loads(fp.read_text()) if fp.exists() else []
    for f in freqs:
        if f in ("Q", "S") and (_mf_dir / f"{f}_cache.parquet").exists():
            _bt_mf_run_freq(state, f)
    state.save_sub("2d.mf.Q")

def _run_bt_mf_M(state: BacktestState) -> None:
    _bt_mf_run_freq(state, "M")
    state.save_sub("2d.mf.M")

def _run_bt_mf_W(state: BacktestState) -> None:
    _bt_mf_run_freq(state, "W")
    state.save_sub("2d.mf.W")

def _run_bt_mf_D(state: BacktestState) -> None:
    _bt_mf_run_freq(state, "D")
    state.save_sub("2d.mf.D")

def _run_bt_mf_fuse(state: BacktestState) -> None:
    _bt_mf_fuse(state)
    state.save_sub("2d.mf.fuse")


# ---------------------------------------------------------------------------
# Stage 3: Profile build + prediction extraction + validation
# ---------------------------------------------------------------------------

def run_stage3(state: BacktestState) -> None:
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

    # Hedge Fund Analysis (before profile build)
    hf_result = None
    try:
        from operator1.hedge_fund.engine import run_hedge_fund_analysis
        hf_result = run_hedge_fund_analysis(
            income_df=state._income_df,
            balance_df=state._balance_df,
            cashflow_df=state._cashflow_df,
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
        run_dir = Path(state.run_dir)
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

        # Hedge Fund Analysis
        if hf_result is not None and hf_result.available:
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
        profile["meta"]["ohlcv_source"] = state._ohlcv_source_label
        profile["meta"]["is_private_company"] = state._is_private
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

        # 12. product_segments
        _seg = state._seg_result
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

        # 13. model_diagnostics (computed in Stage 2d)
        # model_diagnostics_result is local to run_stage2d; use profile injection if available
        profile.setdefault("model_diagnostics", {"available": False})

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
        if state._tv_granger_result is not None and state._tv_granger_result.get("n_windows", 0) > 0:
            profile["extended_models"]["time_varying_granger"] = {
                "available": True, "n_windows": state._tv_granger_result["n_windows"],
            }
        if state._mv_mc_result is not None and state._mv_mc_result.get("available"):
            profile["extended_models"]["multivariate_monte_carlo"] = state._mv_mc_result

        # 17. synergies_applied
        if state._synergy_meta:
            profile["synergies_applied"] = {
                "cycle_features_added": state._synergy_meta.get("cycle_features_added", []),
                "variables_after_pruning": state._synergy_meta.get("variables_after_pruning", 0),
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
    preds_path = Path(state.run_dir) / "predictions_summary.json"
    with open(preds_path, "w") as f:
        json.dump(preds_summary, f, indent=2, default=str)
    logger.info("Predictions saved: %s", preds_path)

    # Report generation (optional, requires LLM client)
    try:
        from operator1.report.report_generator import generate_all_reports
        from operator1.clients.llm_factory import create_llm_client

        llm_client = create_llm_client(state._secrets) if state._secrets else None
        report_dir = Path(state.run_dir) / "report"
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

    state.save_sub("3")
    logger.info("STAGE 3 COMPLETE")


def extract_predictions(state: BacktestState) -> dict:
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

def validate_predictions(state: BacktestState) -> dict:
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
    preds_path = Path(state.run_dir) / "predictions_summary.json"
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
    val_path = Path(state.run_dir) / "validation_results.json"
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
                        choices=["1", "2", "2a", "2a1", "2a2", "2b", "2c", "2d",
                                 "2d.mf.prep", "2d.mf.A", "2d.mf.Q", "2d.mf.M",
                                 "2d.mf.W", "2d.mf.D", "2d.mf.fuse",
                                 "mf.prep", "mf.A", "mf.Q", "mf.M",
                                 "mf.W", "mf.D", "mf.fuse",
                                 "3", "all"],
                        help="Which stage to run. Stage 2a is split into 2a1 (regime+causality+patterns) "
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
    state.run_dir = args.run_dir

    # Validation mode
    if args.validate:
        state.load(stage=3)
        validate_predictions(state)
        return 0

    # Build ordered list of stages to run
    if args.stage == "all":
        # New flow: Stage 1 (shared) -> per-frequency full pipelines -> fusion + HF -> profile
        stages = ["1", "2d.mf.prep", "2d.mf.A", "2d.mf.Q", "2d.mf.M", "2d.mf.W", "2d.mf.D", "2d.mf.fuse", "3"]
    elif args.stage == "2":
        stages = ["2d.mf.prep", "2d.mf.A", "2d.mf.Q", "2d.mf.M", "2d.mf.W", "2d.mf.D", "2d.mf.fuse"]
    elif args.stage == "2a":
        stages = ["2a1", "2a2"]
    else:
        # Map short aliases: mf.prep -> 2d.mf.prep, mf.A -> 2d.mf.A, etc.
        _alias = args.stage
        if _alias.startswith("mf."):
            _alias = f"2d.{_alias}"
        stages = [_alias]

    _STAGE_FUNCS = {
        "1": run_stage1,
        "2a": run_stage2a,
        "2a1": run_stage2a1,
        "2a2": run_stage2a2,
        "2b": run_stage2b,
        "2c": run_stage2c,
        "2d": run_stage2d,
        # Per-frequency multi-frequency sub-stages
        "2d.mf.prep": _run_bt_mf_prep,
        "2d.mf.A": _run_bt_mf_A,
        "2d.mf.Q": _run_bt_mf_Q,
        "2d.mf.M": _run_bt_mf_M,
        "2d.mf.W": _run_bt_mf_W,
        "2d.mf.D": _run_bt_mf_D,
        "2d.mf.fuse": _run_bt_mf_fuse,
        "3": run_stage3,
    }
    # Map sub-stages to the stage they depend on for loading state
    _STAGE_DEPS = {
        "1": None,
        "2a": 1,
        "2a1": 1,
        "2a2": "2a1",
        "2b": "2a2",
        "2c": "2b",
        "2d": "2c",
        "2d.mf.prep": "2c",
        "2d.mf.A": "2d.mf.prep",
        "2d.mf.Q": "2d.mf.A",
        "2d.mf.M": "2d.mf.Q",
        "2d.mf.W": "2d.mf.M",
        "2d.mf.D": "2d.mf.W",
        "2d.mf.fuse": "2d.mf.D",
        "3": "2d",
    }

    for stage_key in stages:
        dep = _STAGE_DEPS[stage_key]
        if dep is not None:
            # Load state from the dependency stage
            dep_key = str(dep)
            state.load_sub(dep_key)

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
