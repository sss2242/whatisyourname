"""Behavioral finance signals -- anomalies from investor psychology.

Features:
  1. anchoring_52w_high: Proximity to 52-week high (George & Hwang 2004, JF)
  2. anchoring_52w_low: Proximity to 52-week low
  3. disposition_effect_proxy: Institutional tendency to sell winners (Shefrin & Statman 1985)
  4. attention_spike: Abnormal volume flag (Barber & Odean 2008, RFS)
  5. lottery_characteristics: High idio-vol + positive skew + low price (Bali et al. 2011, JFE)

Pipeline step: Step 5i.7 (after product_catalysts, before complexity_signals)
All signals gracefully return NaN when input data is insufficient.
"""

from __future__ import annotations

import logging

import numpy as np
import pandas as pd

from operator1.constants import EPSILON
from operator1.scoring_weights import get_weight

logger = logging.getLogger(__name__)

# Load tweakable constants from config (fallback to hardcoded defaults)
_ATTENTION_SPIKE_THRESHOLD = float(get_weight("behavioral_signals.attention_spike_threshold", 2.0))
_DISPOSITION_WINDOW = int(get_weight("behavioral_signals.disposition_window", 63))
_LOTTERY_LOW_PRICE = float(get_weight("behavioral_signals.lottery_low_price_threshold", 5.0))


def compute_behavioral_signals(cache: pd.DataFrame) -> pd.DataFrame:
    """Compute behavioral finance features from cache data.

    Parameters
    ----------
    cache:
        Daily cache with at minimum ``close`` and ``volume``.
        Optional: ``return_1d``, ``inst_flow_momentum``, ``beta_252d``,
        ``benchmark_return_1d``, ``volume_avg_21d``.

    Returns
    -------
    pd.DataFrame
        Cache augmented with behavioral signal columns.
    """
    result = cache.copy()
    close = result.get("close")

    # ------------------------------------------------------------------
    # 1-2. Anchoring to 52-week high/low (George & Hwang 2004)
    # Stocks near 52w high continue rising; near 52w low continue falling.
    # The ratio is bounded [0, 1] for high and [1, inf) for low.
    # ------------------------------------------------------------------
    if close is not None and close.notna().sum() > 30:
        high_252 = close.rolling(252, min_periods=20).max()
        low_252 = close.rolling(252, min_periods=20).min()
        result["anchoring_52w_high"] = close / high_252.clip(lower=EPSILON)
        safe_low = low_252.clip(lower=EPSILON)
        result["anchoring_52w_low"] = close / safe_low

    # ------------------------------------------------------------------
    # 3. Disposition effect proxy (Shefrin & Statman 1985; Frazzini 2006)
    # Correlation between institutional flow and returns over 63 days.
    # Strongly negative = institutions sell winners (disposition effect).
    # ------------------------------------------------------------------
    ret = result.get("return_1d")
    inst_flow = result.get("inst_flow_momentum")
    if ret is not None and inst_flow is not None:
        if ret.notna().sum() > 70 and inst_flow.notna().sum() > 10:
            result["disposition_effect_proxy"] = ret.rolling(
                63, min_periods=30
            ).corr(inst_flow)

    # ------------------------------------------------------------------
    # 4. Attention spike (Barber & Odean 2008)
    # Z-scored volume > 2.0 flags abnormal attention days.
    # Attention-driven buying predicts short-term overreaction.
    # ------------------------------------------------------------------
    volume = result.get("volume")
    vol_avg = result.get("volume_avg_21d")
    if volume is not None and vol_avg is not None:
        vol_std = volume.rolling(21, min_periods=5).std().clip(lower=EPSILON)
        vol_z = (volume - vol_avg) / vol_std
        result["attention_spike"] = (vol_z > 2.0).astype(int)

    # ------------------------------------------------------------------
    # 5. Lottery characteristics (Bali, Cakici & Whitelaw 2011)
    # Composite of: high idiosyncratic vol + positive skewness + low price.
    # Lottery stocks have negative expected returns (overpriced by
    # retail investors seeking asymmetric payoffs).
    # ------------------------------------------------------------------
    if ret is not None and close is not None and ret.notna().sum() > 70:
        # Idiosyncratic vol: residual after removing beta * benchmark
        beta = result.get("beta_252d")
        bench = result.get("benchmark_return_1d")
        if beta is not None and bench is not None and bench.notna().sum() > 30:
            residual = ret - beta.fillna(1.0) * bench.fillna(0)
        else:
            residual = ret  # total vol as fallback when no benchmark

        idio_vol = residual.rolling(63, min_periods=20).std()

        # Return skewness (positive skew = lottery-like payoff)
        skew = ret.rolling(63, min_periods=30).skew()

        # Low price flag (below 20th percentile of own history)
        price_p20 = close.expanding(min_periods=20).quantile(0.20)
        low_price = (close < price_p20).astype(float)

        # Normalize each to [0, 1] range via expanding percentile rank
        def _expanding_pctrank(s: pd.Series) -> pd.Series:
            return s.expanding(min_periods=20).apply(
                lambda x: pd.Series(x).rank(pct=True).iloc[-1],
                raw=False,
            )

        idio_pct = _expanding_pctrank(idio_vol)
        skew_pct = _expanding_pctrank(skew)

        result["lottery_characteristics"] = (idio_pct + skew_pct + low_price) / 3.0

    n_new = len(result.columns) - len(cache.columns)
    if n_new > 0:
        logger.info("Behavioral signals computed: %d columns", n_new)

    return result
