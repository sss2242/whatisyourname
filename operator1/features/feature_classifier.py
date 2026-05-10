"""Classify cache columns by temporal resolution for model routing.

Separates features into three classes based on how frequently they change:

- **TICK**: Changes most trading days (return_1d, volume, MACD, z-scores).
  Suitable for LSTM, Tree, VAR -- temporal models need daily variation.
- **PERIODIC**: Changes at filing frequency (current_ratio, PE, margins).
  Forward-filled for 63+ days. NOT suitable for temporal models directly.
  Use normalized versions (_zscore_63d, _change_21d) instead.
- **STATIC**: Changes rarely or never (conflict flags, sector scores, geo_hhi).
  Conditioning/regime classification only.

Per-model routing:
- LSTM/Transformer: TICK only (~20 features with daily variation)
- Tree ensemble: TICK + normalized PERIODIC (~45 features)
- VAR/Kalman: TICK only (stationarity preferred)
- Baseline: no features

Three-tier classification: name-based -> suffix-based -> variance-based.
"""

from __future__ import annotations

import logging
from enum import Enum
from typing import Any

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


class FeatureClass(Enum):
    """Temporal resolution class for a cache feature."""

    TICK = "tick"          # changes most trading days
    PERIODIC = "periodic"  # changes at filing frequency (~63-252 day intervals)
    STATIC = "static"      # changes rarely or never


# ---------------------------------------------------------------------------
# Name-based heuristics (covers ~90% of known features)
# ---------------------------------------------------------------------------

TICK_PREFIXES = (
    "return_", "log_return", "volatility_", "volatility_ewma",
    "volume", "close_frac_diff", "macd", "rsi", "adx", "obv",
    "bb_", "sma_", "benchmark_return", "online_change_score",
    "sentiment_score", "sentiment_momentum", "sentiment_volatility",
    "parkinson_vol", "yang_zhang_vol", "corwin_schultz",
    "kyle_lambda", "volume_clock", "hurst_exponent",
    "autocorr_lag", "momentum_12_1", "jump_ratio", "jump_spike",
    "realized_vol", "continuous_vol", "jump_vol",
    "skewness_63d", "kurtosis_63d", "tail_ratio", "max_daily_loss",
    "vol_of_vol", "sample_entropy", "perm_entropy", "lz_complexity",
    "approx_entropy", "anchoring_", "disposition_", "attention_spike",
    "lottery_characteristics",
)

STATIC_PREFIXES = (
    "country_conflict", "company_conflict", "sanctions_flag",
    "fragile_state", "conflict_type", "conflict_intensity",
    "geo_hhi", "china_revenue", "macro_quadrant",
    "fuzzy_protection", "fuzzy_sector", "fuzzy_economic",
    "fuzzy_policy", "fuzzy_parent", "sector_strategicness",
    "country_protected", "country_survival",
    "supply_chain_stress_flag",
)

# Suffixes that convert periodic features to tick-frequency signals
TICK_SUFFIXES = (
    "_zscore_63d", "_percentile_252d", "_change_21d", "_regime_zscore",
    "_delta_5d", "_delta_21d",
)


class FeatureClassifier:
    """Classify cache columns by temporal resolution for model routing."""

    def classify(self, col_name: str, series: pd.Series | None = None) -> FeatureClass:
        """Classify a single feature.

        Three-tier classification (in priority order):
        1. Name-based heuristics (instant, ~90% coverage)
        2. Suffix-based inference (normalized versions of periodic are TICK)
        3. Variance-based detection (for unknown features)
        """
        col_lower = col_name.lower()

        # Tier 1: Name-based (TICK)
        if any(col_lower.startswith(p) for p in TICK_PREFIXES):
            return FeatureClass.TICK

        # Tier 1: Name-based (STATIC)
        if any(col_lower.startswith(p) for p in STATIC_PREFIXES):
            return FeatureClass.STATIC

        # Tier 2: Suffix-based (normalized periodic -> TICK)
        if any(col_lower.endswith(s) for s in TICK_SUFFIXES):
            return FeatureClass.TICK

        # Companion flags are same class as their parent
        if col_lower.startswith("is_missing_") or col_lower.startswith("invalid_math_"):
            return FeatureClass.STATIC  # flags are binary, rarely change

        # Tier 3: Variance-based detection (for unknown features)
        if series is not None:
            recent = series.tail(63).dropna()
            if len(recent) < 10:
                return FeatureClass.STATIC
            n_distinct = recent.nunique()
            change_rate = n_distinct / len(recent)
            if change_rate > 0.30:
                return FeatureClass.TICK
            elif change_rate > 0.02:
                return FeatureClass.PERIODIC
            return FeatureClass.STATIC

        # Default: assume PERIODIC (safe -- won't be sent to LSTM)
        return FeatureClass.PERIODIC

    def classify_all(
        self,
        cache: pd.DataFrame,
        variables: list[str],
    ) -> dict[str, FeatureClass]:
        """Classify a list of variables, returning a dict of classifications."""
        result: dict[str, FeatureClass] = {}
        for var in variables:
            series = cache.get(var) if cache is not None else None
            result[var] = self.classify(var, series)
        return result

    def build_model_feature_sets(
        self,
        cache: pd.DataFrame,
        extra_vars: list[str],
    ) -> dict[str, list[str]]:
        """Build per-model-type feature sets from classified features.

        Returns a dict with model-type keys and feature-list values:
        - ``"lstm"``: TICK features only (daily variation for gradient learning)
        - ``"tree"``: TICK + normalized PERIODIC (z-scores change daily)
        - ``"var"``: TICK features (stationarity preferred)
        - ``"kalman"``: empty (single-variable, no exogenous)
        - ``"baseline"``: empty
        - ``"all"``: everything (backward compat)
        """
        classifications = self.classify_all(cache, extra_vars)

        tick: list[str] = []
        periodic_normalized: list[str] = []

        for var, cls in classifications.items():
            if cls == FeatureClass.TICK:
                tick.append(var)
            elif cls == FeatureClass.PERIODIC:
                # Check if a normalized version exists in cache
                found_norm = False
                for suffix in TICK_SUFFIXES:
                    norm_var = var + suffix
                    if norm_var in cache.columns and norm_var in classifications:
                        # Already classified as TICK via suffix rule
                        found_norm = True
                        break
                    elif norm_var in cache.columns:
                        periodic_normalized.append(norm_var)
                        found_norm = True
                        break
                if not found_norm:
                    # No normalized version -- keep raw for tree (can handle)
                    periodic_normalized.append(var)
            # STATIC features excluded from all temporal models

        model_sets = {
            "lstm": tick,
            "tree": tick + periodic_normalized,
            "var": tick,
            "kalman": [],
            "baseline": [],
            "all": extra_vars,
        }

        logger.info(
            "Feature classifier: %d tick, %d periodic_norm, %d static "
            "(from %d total extra_vars)",
            len(tick),
            len(periodic_normalized),
            sum(1 for c in classifications.values() if c == FeatureClass.STATIC),
            len(extra_vars),
        )

        return model_sets
