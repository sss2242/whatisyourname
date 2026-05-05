"""PipelineState -- shared mutable state bag for the staged pipeline.

Replaces the hundreds of local variables in main.py's main() function.
Each sub-stage reads from and writes to this object. Between sub-stages,
it serializes to disk (cache.parquet + state_{sub_stage}.pkl) so the next
sub-stage can resume without re-running prior work.

Usage:
    state = PipelineState(market_id="us_sec_edgar", company="AAPL",
                          end_date="2024-12-31", output_dir="cache/AAPL")
    run_stage_3_1(state)   # regime detection
    state.save("3.1")      # persist to disk
    # ... new process invocation ...
    state = PipelineState.load_from("3.1", run_dir="cache/AAPL")
    run_stage_3_2(state)   # dual regimes
"""

from __future__ import annotations

import json
import logging
import pickle
from datetime import datetime
from pathlib import Path
from typing import Any

import pandas as pd

logger = logging.getLogger("operator1.pipeline_state")


class PipelineState:
    """Mutable state bag passed between pipeline sub-stages."""

    def __init__(
        self,
        market_id: str = "",
        company: str = "",
        end_date: str = "",
        years: float = 2.0,
        output_dir: str = "cache",
    ) -> None:
        # ---- Config ----
        self.market_id = market_id
        self.company = company
        self.end_date = end_date
        self.years = years
        self.output_dir = output_dir

        # ---- Stage 1: Data Acquisition ----
        self.cache: pd.DataFrame | None = None
        self.target_profile: dict = {}
        self.income_df: pd.DataFrame = pd.DataFrame()
        self.balance_df: pd.DataFrame = pd.DataFrame()
        self.cashflow_df: pd.DataFrame = pd.DataFrame()
        self.quotes_df: pd.DataFrame = pd.DataFrame()
        self.macro_data: dict = {}
        self.macro_dataset: Any = None
        self.macro_quadrant_result: Any = None
        self.conflict_result: Any = None
        self.buying_power_result: Any = None
        self.estimation_coverage: Any = None
        self.filing_calendar_result: Any = None
        self.event_calendar_result: Any = None
        self.options_signal_result: Any = None
        self.cross_asset_result: Any = None

        # ---- Stage 2: Feature Engineering ----
        self.weights: dict = {}
        self.fh_result: Any = None
        self.fuzzy_result: Any = None
        self.relationships: dict = {}
        self.linked_caches: dict = {}
        self.linked_agg_df: pd.DataFrame | None = None
        self.graph_risk_result: Any = None
        self.game_theory_result: Any = None
        self.contagion_result: Any = None
        self.peer_ranking_result: Any = None
        self.sentiment_result: Any = None
        self.catalyst_result: Any = None
        self.signal_ic_result: Any = None
        self.prediction_log_summary: Any = None
        self.enriched_timeline_result: Any = None
        self.early_regime_result: Any = None
        self.regime_detector: Any = None
        self.survival_controller: Any = None
        self.target_holders: list = []
        self.target_insiders: list = []
        self.six_proxy_result: Any = None
        self.linked_conflict: Any = None
        self.seg_result: dict = {}
        self.supply_chain_stress_result: Any = None
        self.is_private: bool = False
        self.ohlcv_source_label: str = ""
        self.adaptive_thresholds: Any = None
        self.adaptive_model_params: Any = None
        self.adaptive_tier3: Any = None
        self.mode_weights: Any = None

        # ---- Per-frequency raw statement groups (from frequency separator) ----
        # Keys: "quarterly", "semiannual", "annual"
        # Values: DataFrames with actual filing data for that frequency only.
        # Preserved from main.py Step 3d so Stage 7.4 MF pipeline can use
        # frequency-appropriate source data instead of the reconciled
        # highest-freq version.
        self.income_freq_groups: dict[str, pd.DataFrame] = {}
        self.balance_freq_groups: dict[str, pd.DataFrame] = {}
        self.cashflow_freq_groups: dict[str, pd.DataFrame] = {}

        # ---- Stage 3: Temporal Analysis ----
        self.feature_selection_result: Any = None
        self.dual_regime_result: Any = None
        self.granger_result: Any = None
        self.transfer_entropy_result: Any = None
        self.cycle_result: Any = None
        self.pattern_result: Any = None
        self.synergy_meta: dict = {}
        self.extra_vars: list = []
        self.economic_plane: Any = None

        # ---- Stage 4: Forecasting ----
        self.forecast_result: Any = None

        # ---- Stage 5: Forward Modeling ----
        self.forward_pass_result: Any = None
        self.burnout_result: Any = None
        self.walk_forward_result: Any = None
        self.mc_result: Any = None
        self.copula_result: Any = None
        self.regime_shift_result: Any = None

        # ---- Stage 6: Ensemble & Aggregation ----
        self.transformer_result: Any = None
        self.particle_filter_result: Any = None
        self.conformal_result: Any = None
        self.dtw_result: Any = None
        self.pred_result: Any = None
        self.shap_result: Any = None
        self.sobol_result: Any = None
        self.ga_result: Any = None
        self.ohlc_result: Any = None
        self.pattern_drift: float = 1.0
        self.tv_granger_result: Any = None
        self.mv_mc_result: Any = None
        self.recursive_result: Any = None

        # ---- Stage 7: Integration ----
        self.scenario_result: Any = None
        self.retro_params: Any = None
        self.model_diagnostics_result: Any = None
        self.multi_frequency_result: Any = None
        self.hf_result: Any = None

        # ---- Stage 8: Output ----
        self.profile: dict = {}
        self.predictions_summary: dict = {}

        # ---- Transient runtime objects (not serialized) ----
        self._secrets: dict = {}
        self._llm_client: Any = None
        self._pit_client: Any = None

    # ------------------------------------------------------------------
    # Serialization
    # ------------------------------------------------------------------

    def save(self, sub_stage: str) -> None:
        """Persist state to disk after a sub-stage completes."""
        run_dir = Path(self.output_dir)
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
            with open(lc_dir / "_ids.json", "w") as f:
                json.dump(list(self.linked_caches.keys()), f)

        # Save linked_agg_df
        if self.linked_agg_df is not None and not self.linked_agg_df.empty:
            self.linked_agg_df.to_parquet(run_dir / "linked_agg.parquet")

        # Save non-DataFrame state as pickle
        skip_keys = {
            "cache", "linked_caches", "linked_agg_df",
            "_secrets", "_llm_client", "_pit_client",
            "income_df", "balance_df", "cashflow_df", "quotes_df",
            "income_freq_groups", "balance_freq_groups", "cashflow_freq_groups",
        }
        state_dict = {}
        for k, v in self.__dict__.items():
            if k in skip_keys:
                continue
            try:
                pickle.dumps(v)
                state_dict[k] = v
            except (pickle.PicklingError, TypeError, AttributeError) as exc:
                # Try to salvage the result by stripping non-picklable sub-fields
                # (e.g., ForwardPassResult.model_states contains fitted sklearn/torch models)
                _salvaged = False
                if hasattr(v, "__dict__"):
                    try:
                        _cleaned = type(v).__new__(type(v))
                        for attr_name, attr_val in v.__dict__.items():
                            try:
                                pickle.dumps(attr_val)
                                setattr(_cleaned, attr_name, attr_val)
                            except (pickle.PicklingError, TypeError, AttributeError):
                                setattr(_cleaned, attr_name, None)
                                logger.debug(
                                    "Stripped non-picklable sub-field %s.%s",
                                    k, attr_name,
                                )
                        pickle.dumps(_cleaned)
                        state_dict[k] = _cleaned
                        _salvaged = True
                        logger.info(
                            "Salvaged %s by stripping non-picklable sub-fields", k,
                        )
                    except Exception:
                        pass
                if not _salvaged:
                    logger.warning("Skipping non-picklable key: %s (%s)", k, exc)

        # Save raw statement DFs separately (they can be large)
        for df_name in ("income_df", "balance_df", "cashflow_df", "quotes_df"):
            df = getattr(self, df_name, None)
            if df is not None and not df.empty:
                df.to_parquet(run_dir / f"{df_name}.parquet")

        # Save per-frequency statement groups (from frequency separator)
        for grp_name in ("income_freq_groups", "balance_freq_groups", "cashflow_freq_groups"):
            grp = getattr(self, grp_name, {})
            if grp:
                grp_dir = run_dir / "freq_groups" / grp_name
                grp_dir.mkdir(parents=True, exist_ok=True)
                for freq_label, df in grp.items():
                    if df is not None and not df.empty:
                        df.to_parquet(grp_dir / f"{freq_label}.parquet")

        with open(run_dir / f"state_{sub_stage}.pkl", "wb") as f:
            pickle.dump(state_dict, f)

        # Human-readable config
        config = {
            "market_id": self.market_id,
            "company": self.company,
            "end_date": self.end_date,
            "years": self.years,
            "output_dir": self.output_dir,
            "sub_stage_completed": sub_stage,
            "timestamp": datetime.now().isoformat(),
        }
        with open(run_dir / "config.json", "w") as f:
            json.dump(config, f, indent=2)

        logger.info("State saved to %s (sub-stage %s)", run_dir, sub_stage)

    @classmethod
    def load_from(cls, sub_stage: str, run_dir: str) -> "PipelineState":
        """Load state from a previous sub-stage checkpoint."""
        state = cls()
        state.output_dir = run_dir
        state._load_state(sub_stage)
        return state

    def load_checkpoint(self, sub_stage: str) -> None:
        """Load state from a checkpoint in-place."""
        self._load_state(sub_stage)

    def _load_state(self, sub_stage: str) -> None:
        """Internal: load state from disk."""
        run_dir = Path(self.output_dir)

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
            logger.info("Loaded cache: %d rows x %d cols",
                        len(self.cache), len(self.cache.columns))

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

        # Load linked_agg
        lag_path = run_dir / "linked_agg.parquet"
        if lag_path.exists():
            self.linked_agg_df = pd.read_parquet(lag_path)

        # Load raw statement DFs
        for df_name in ("income_df", "balance_df", "cashflow_df", "quotes_df"):
            df_path = run_dir / f"{df_name}.parquet"
            if df_path.exists():
                setattr(self, df_name, pd.read_parquet(df_path))

        # Load per-frequency statement groups (from frequency separator)
        for grp_name in ("income_freq_groups", "balance_freq_groups", "cashflow_freq_groups"):
            grp_dir = run_dir / "freq_groups" / grp_name
            if grp_dir.exists():
                grp = {}
                for pq in grp_dir.glob("*.parquet"):
                    grp[pq.stem] = pd.read_parquet(pq)
                if grp:
                    setattr(self, grp_name, grp)

        # Find the state pickle -- try exact sub-stage, then fall back
        pkl_candidates = [
            run_dir / f"state_{sub_stage}.pkl",
        ]
        # Also try numeric stage pickles for backward compat
        if "." not in sub_stage and sub_stage.replace("a", "").replace("b", "").replace("c", "").replace("d", "").replace("e", "").isdigit():
            pkl_candidates.append(run_dir / f"state_stage{sub_stage}.pkl")

        for pkl_path in pkl_candidates:
            if pkl_path.exists():
                with open(pkl_path, "rb") as f:
                    state_dict = pickle.load(f)
                for k, v in state_dict.items():
                    if hasattr(self, k):
                        setattr(self, k, v)
                logger.info("Loaded state from %s (%d keys)",
                            pkl_path.name, len(state_dict))
                break

    # ------------------------------------------------------------------
    # Multi-frequency per-frequency disk helpers
    # ------------------------------------------------------------------

    def _mf_dir(self) -> Path:
        """Return the multi-frequency sub-directory, creating if needed."""
        d = Path(self.output_dir) / "mf"
        d.mkdir(parents=True, exist_ok=True)
        return d

    def save_mf_cache(self, freq: str, resampled) -> None:
        """Save a ResampledCache to disk for a single frequency."""
        d = self._mf_dir()
        resampled.cache.to_parquet(d / f"{freq}_cache.parquet")
        import json as _json
        meta = {
            "frequency": resampled.frequency,
            "label": resampled.label,
            "n_periods": resampled.n_periods,
            "lookback_years": resampled.lookback_years,
            "is_partial_last_period": resampled.is_partial_last_period,
            "original_daily_rows": resampled.original_daily_rows,
            "resampled_rows": resampled.resampled_rows,
            "data_source": getattr(resampled, "data_source", "unknown"),
        }
        (d / f"{freq}_meta.json").write_text(_json.dumps(meta, indent=2))

    def load_mf_cache(self, freq: str):
        """Load a ResampledCache from disk for a single frequency."""
        d = self._mf_dir()
        cache_path = d / f"{freq}_cache.parquet"
        meta_path = d / f"{freq}_meta.json"
        if not cache_path.exists():
            return None
        import json as _json
        from operator1.features.frequency_resampler import ResampledCache
        cache_df = pd.read_parquet(cache_path)
        meta = _json.loads(meta_path.read_text()) if meta_path.exists() else {}
        return ResampledCache(
            cache=cache_df,
            frequency=meta.get("frequency", freq),
            label=meta.get("label", freq),
            n_periods=meta.get("n_periods", len(cache_df)),
            lookback_years=meta.get("lookback_years", 2),
            is_partial_last_period=meta.get("is_partial_last_period", False),
            original_daily_rows=meta.get("original_daily_rows", 0),
            resampled_rows=meta.get("resampled_rows", len(cache_df)),
            data_source=meta.get("data_source", "unknown"),
        )

    def save_mf_result(self, freq: str, result) -> None:
        """Save a FrequencyResult to disk."""
        d = self._mf_dir()
        with open(d / f"{freq}_result.pkl", "wb") as f:
            pickle.dump(result, f)

    def load_mf_result(self, freq: str):
        """Load a FrequencyResult from disk."""
        d = self._mf_dir()
        pkl = d / f"{freq}_result.pkl"
        if not pkl.exists():
            return None
        with open(pkl, "rb") as f:
            return pickle.load(f)

    def save_mf_context(self, freq: str, context) -> None:
        """Save a FrequencyContext to disk (for cascading to next freq)."""
        d = self._mf_dir()
        with open(d / f"{freq}_context.pkl", "wb") as f:
            pickle.dump(context, f)

    def load_mf_context(self, freq: str):
        """Load a FrequencyContext from disk."""
        d = self._mf_dir()
        pkl = d / f"{freq}_context.pkl"
        if not pkl.exists():
            return None
        with open(pkl, "rb") as f:
            return pickle.load(f)

    def list_mf_results(self) -> list[str]:
        """List all frequencies that have saved FrequencyResults."""
        d = self._mf_dir()
        return [
            p.stem.replace("_result", "")
            for p in sorted(d.glob("*_result.pkl"))
        ]

    def save_mf_frequencies(self, frequencies: list[str]) -> None:
        """Save the ordered list of frequencies to run."""
        import json as _json
        d = self._mf_dir()
        (d / "frequencies.json").write_text(_json.dumps(frequencies))

    def load_mf_frequencies(self) -> list[str]:
        """Load the ordered frequency list."""
        import json as _json
        d = self._mf_dir()
        p = d / "frequencies.json"
        if not p.exists():
            return []
        return _json.loads(p.read_text())

    def find_latest_checkpoint(self) -> str | None:
        """Find the latest completed sub-stage checkpoint in run_dir."""
        run_dir = Path(self.output_dir)
        if not run_dir.exists():
            return None

        # Look for state_*.pkl files and find the latest by timestamp
        checkpoints = []
        for pkl in run_dir.glob("state_*.pkl"):
            sub = pkl.stem.replace("state_", "")
            checkpoints.append((pkl.stat().st_mtime, sub))

        if not checkpoints:
            return None

        checkpoints.sort(reverse=True)
        return checkpoints[0][1]
