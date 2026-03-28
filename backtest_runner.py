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

    # Product catalysts
    try:
        from operator1.features.product_catalysts import detect_product_catalysts
        cache, state.catalyst_result = detect_product_catalysts(
            cache, profile=state.target_profile,
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
    except Exception as exc:
        logger.warning("Enriched timeline failed: %s", exc)

    state.cache = cache
    state.save(stage=1)
    logger.info("STAGE 1 COMPLETE: %d rows x %d cols", len(cache), len(cache.columns))


# ---------------------------------------------------------------------------
# Stage 2: Temporal models
# ---------------------------------------------------------------------------

def run_stage2(state: BacktestState) -> None:
    """Run all temporal models on the cached data from Stage 1."""
    logger.info("=" * 60)
    logger.info("STAGE 2: Temporal Models")
    logger.info("=" * 60)

    cache = state.cache
    if cache is None or cache.empty:
        raise ValueError("No cache data -- run Stage 1 first")

    from operator1.models.regime_detector import detect_regimes_and_breaks
    from operator1.models.forecasting import run_forecasting, run_forward_pass, run_burnout
    from operator1.models.monte_carlo import run_monte_carlo
    from operator1.models.prediction_aggregator import run_prediction_aggregation

    # Extra variables for temporal models
    _linked_prefixes = ("competitors_", "suppliers_", "customers_", "financial_institutions_", "sector_peers_")
    state._extra_vars = [
        c for c in cache.columns
        if (c.startswith("fh_") or c.startswith("sentiment_") or c.startswith("peer_")
            or c.startswith("macro_") or c.startswith("inst_")
            or c.startswith("buying_power_") or c.startswith("catalyst_")
            or c.startswith("conflict_") or c.startswith("demand_")
            or c in ("survival_intensity", "regime_confidence", "regime_transition_prob",
                     "stability_score_21d", "buying_power_index", "catalyst_score")
            or any(c.startswith(p) for p in _linked_prefixes))
        and cache[c].dtype in ("float64", "float32", "int64")
        and not c.startswith("is_missing_")
    ]

    # Regime detection (skip if already done)
    if state.regime_detector is None or "regime_label" not in cache.columns:
        try:
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

    # Forecasting
    try:
        cache, state.forecast_result = run_forecasting(cache, extra_variables=state._extra_vars)
        logger.info("Forecasting complete")
    except Exception as exc:
        logger.warning("Forecasting failed: %s", exc)

    # Forward pass
    regime_labels = cache.get("regime_label") if "regime_label" in cache.columns else None
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
        state.mc_result = run_monte_carlo(cache, returns_col=mc_ret)
        logger.info("Monte Carlo complete")
    except Exception as exc:
        logger.warning("Monte Carlo failed: %s", exc)

    # Copula
    try:
        from operator1.models.copula import run_copula_analysis
        state.copula_result = run_copula_analysis(cache)
    except Exception:
        pass

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

    # Conformal prediction
    try:
        from operator1.models.conformal import ConformalPIDCalibrator, ConformalCalibrator, build_conformal_result
        if state.forecast_result is not None:
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

    # Prediction aggregation
    if state.forecast_result is not None:
        try:
            state.pred_result = run_prediction_aggregation(
                cache, state.forecast_result, state.mc_result,
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

    # OHLC predictor
    try:
        from operator1.models.ohlc_predictor import predict_ohlc_series
        from operator1.models.model_synergies import compute_pattern_drift_adjustment
        state._pattern_drift = compute_pattern_drift_adjustment(state.pattern_result)
        state.ohlc_result = predict_ohlc_series(
            cache, forecast_result=state.forecast_result, mc_result=state.mc_result,
            pattern_drift_multiplier=state._pattern_drift, cycle_result=state.cycle_result,
        )
    except Exception:
        pass

    state.cache = cache
    state.save(stage=2)
    logger.info("STAGE 2 COMPLETE")


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
            except: pass
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

        profile = _sanitize(profile)
        with open(profile_path, "w") as f:
            json.dump(profile, f, indent=2, default=str)
        logger.info("Profile saved: %s", profile_path)
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

    state.save(stage=3)
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
                        "lower_bound": getattr(hp, "lower_bound", None),
                        "upper_bound": getattr(hp, "upper_bound", None),
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
            bounds_str = f" [{'IN' if c['actual_in_bounds'] else 'OUT'} bounds: {c.get('lower_bound', '?'):.2f}-{c.get('upper_bound', '?'):.2f}]"
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
    parser.add_argument("--stage", type=str, default="all", choices=["1", "2", "3", "all"],
                        help="Which stage to run (1, 2, 3, or all)")
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

    stages = [1, 2, 3] if args.stage == "all" else [int(args.stage)]

    for stage_num in stages:
        # Load state from prior stage if resuming
        if stage_num > 1:
            state.load(stage=stage_num - 1)

        t0 = time.time()
        try:
            if stage_num == 1:
                run_stage1(state)
            elif stage_num == 2:
                run_stage2(state)
            elif stage_num == 3:
                run_stage3(state)
        except Exception as exc:
            logger.error("Stage %d FAILED: %s", stage_num, exc)
            import traceback
            traceback.print_exc()
            return 1

        elapsed = time.time() - t0
        logger.info("Stage %d completed in %.1fs", stage_num, elapsed)

    return 0


if __name__ == "__main__":
    sys.exit(main())
