"""Institutional Flow Predictor -- derived signals from ownership time series.

Computes daily features from the ``inst_*`` columns already in the cache
(populated at Step 4d from ``get_holder_history()``):

1. **Flow Momentum** (Brunnermeier & Nagel 2004): EMA-smoothed quarter-
   over-quarter change in institutional ownership percentage.
2. **Crowding Risk** (dynamic): concentration * ownership level, amplified
   by negative flow momentum.
3. **Smart Money Signal**: divergence between top-5 concentration change
   and total ownership change.
4. **Insider Signal**: net insider buying/selling from insider transaction
   data (US only via yfinance).
5. **Amihud Illiquidity** (Amihud 2002): daily ``|return| / dollar_volume``
   ratio -- the only time-varying institutional column.

Academic refs:
  - Brunnermeier & Nagel 2004 (hedge fund flows)
  - Lakonishok, Shleifer & Vishny 1992 (LSV herding measure)
  - Amihud 2002 (illiquidity ratio)

All signals gracefully return NaN when input data is insufficient.
"""

from __future__ import annotations

import logging
from typing import Any

import numpy as np
import pandas as pd

from operator1.constants import EPSILON

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Label thresholds
# ---------------------------------------------------------------------------

_FLOW_LABELS: list[tuple[float, str]] = [
    (-0.10, "distributing"),   # < -10% QoQ
    (-0.03, "reducing"),       # -10% to -3%
    (0.03, "stable"),          # -3% to +3%
    (0.10, "accumulating"),    # +3% to +10%
    (float("inf"), "strong_accumulating"),
]

_CROWDING_LABELS: list[tuple[float, str]] = [
    (0.3, "low"),
    (0.6, "moderate"),
    (0.8, "high"),
    (float("inf"), "extreme"),
]

_SMART_MONEY_LABELS: list[tuple[float, str]] = [
    (-0.15, "conviction_sell"),
    (-0.05, "mild_sell"),
    (0.05, "neutral"),
    (0.15, "mild_buy"),
    (float("inf"), "conviction_buy"),
]

_INSIDER_LABELS: list[tuple[float, str]] = [
    (-0.3, "insider_selling"),
    (-0.05, "mild_selling"),
    (0.05, "neutral"),
    (0.3, "mild_buying"),
    (float("inf"), "insider_buying"),
]


def _apply_labels(
    series: pd.Series,
    thresholds: list[tuple[float, str]],
    default: str = "unknown",
) -> pd.Series:
    """Map a continuous series to categorical labels via thresholds."""
    labels = pd.Series(default, index=series.index, dtype="object")
    for threshold, label in thresholds:
        mask = series.notna() & (series <= threshold)
        labels = labels.where(~mask | (labels != default), other=label)
    # Apply in order: first match wins
    result = pd.Series(default, index=series.index, dtype="object")
    for threshold, label in thresholds:
        still_default = result == default
        in_range = series.notna() & (series <= threshold)
        result = result.where(~(still_default & in_range), other=label)
    return result


# ---------------------------------------------------------------------------
# 1. Flow Momentum (Brunnermeier & Nagel 2004)
# ---------------------------------------------------------------------------


def _compute_flow_momentum(cache: pd.DataFrame) -> pd.Series:
    """EMA-smoothed quarter-over-quarter change in institutional ownership.

    Uses a 63-business-day lookback (approx 1 quarter).  Returns NaN where
    insufficient data exists (US/UK single snapshot -> constant -> delta=0).
    """
    pct = cache.get("inst_ownership_pct")
    if pct is None or pct.notna().sum() < 2:
        return pd.Series(np.nan, index=cache.index, name="inst_flow_momentum")

    # Quarter-over-quarter percentage change
    shift_period = min(63, len(pct) - 1)
    if shift_period < 1:
        return pd.Series(np.nan, index=cache.index, name="inst_flow_momentum")

    shifted = pct.shift(shift_period)
    with np.errstate(divide="ignore", invalid="ignore"):
        delta = (pct - shifted) / shifted.clip(lower=EPSILON)
    delta = delta.replace([np.inf, -np.inf], np.nan)

    # EMA smoothing to reduce noise from interpolation artifacts
    momentum = delta.ewm(span=21, min_periods=5).mean()
    momentum.name = "inst_flow_momentum"
    return momentum


# ---------------------------------------------------------------------------
# 2. Crowding Risk (dynamic)
# ---------------------------------------------------------------------------


def _compute_crowding_risk(cache: pd.DataFrame) -> pd.Series:
    """Dynamic crowding risk: concentration * ownership * flow stress.

    Unlike the static crowding_score in ownership_contagion.py (which
    uses current-quarter snapshot data), this uses the time-series
    evolution of concentration.
    """
    hhi = cache.get("inst_top5_concentration")
    pct = cache.get("inst_ownership_pct")

    if hhi is None or pct is None:
        return pd.Series(np.nan, index=cache.index, name="inst_crowding_risk")

    # Base crowding = concentration * ownership fraction
    base = hhi * (pct / 100.0).clip(lower=0, upper=1)

    # Amplify when flow momentum is negative (institutions leaving)
    flow = cache.get("inst_flow_momentum")
    if flow is not None and flow.notna().any():
        negative_flow = (-flow).clip(lower=0)
        amplifier = 1.0 + negative_flow * 2.0  # up to 3x amplification
        crowding = (base * amplifier).clip(lower=0, upper=1)
    else:
        crowding = base.clip(lower=0, upper=1)

    crowding.name = "inst_crowding_risk"
    return crowding


# ---------------------------------------------------------------------------
# 3. Smart Money Signal
# ---------------------------------------------------------------------------


def _compute_smart_money_signal(cache: pd.DataFrame) -> pd.Series:
    """Divergence between top-5 concentration change and total ownership change.

    Positive = top holders accumulating while total ownership declining
    (informed buying).  Negative = top holders reducing while total
    increasing (informed selling, retail buying the dip).
    """
    hhi = cache.get("inst_top5_concentration")
    pct = cache.get("inst_ownership_pct")

    if hhi is None or pct is None:
        return pd.Series(np.nan, index=cache.index, name="inst_smart_money_signal")

    if hhi.notna().sum() < 2 or pct.notna().sum() < 2:
        return pd.Series(np.nan, index=cache.index, name="inst_smart_money_signal")

    shift_period = min(63, max(1, len(hhi) - 1))

    # Change in top-5 concentration (positive = top holders gaining share)
    hhi_delta = hhi - hhi.shift(shift_period)

    # Change in total ownership (positive = total institutional buying)
    pct_delta = pct - pct.shift(shift_period)

    # Divergence: positive when top holders move opposite to total
    # Normalize by typical magnitude
    divergence = hhi_delta - (pct_delta / 100.0)
    signal = divergence.clip(-1, 1)
    signal.name = "inst_smart_money_signal"
    return signal


# ---------------------------------------------------------------------------
# 4. Insider Signal (from insider_transactions data)
# ---------------------------------------------------------------------------


def _compute_insider_signal(
    cache: pd.DataFrame,
    insider_transactions: list[dict[str, Any]] | None = None,
) -> pd.Series:
    """Net insider buying/selling signal from management transactions.

    Positive = insiders buying (bullish conviction from people who
    know the company best).  Negative = insiders selling.
    """
    if not insider_transactions:
        return pd.Series(np.nan, index=cache.index, name="inst_insider_signal")

    try:
        df = pd.DataFrame(insider_transactions)
        if df.empty or "date" not in df.columns or "shares" not in df.columns:
            return pd.Series(np.nan, index=cache.index, name="inst_insider_signal")

        df["date"] = pd.to_datetime(df["date"], errors="coerce")
        df = df.dropna(subset=["date"])
        if df.empty:
            return pd.Series(np.nan, index=cache.index, name="inst_insider_signal")

        # Classify transaction type
        if "transaction" in df.columns:
            # yfinance insider_transactions has a "Transaction" column
            # "Sale" = negative, "Purchase" / "Buy" = positive
            def _sign(tx: str) -> int:
                tx_lower = str(tx).lower()
                if "sale" in tx_lower or "sell" in tx_lower:
                    return -1
                return 1

            df["signed_shares"] = df["shares"].abs() * df["transaction"].apply(_sign)
        else:
            df["signed_shares"] = df["shares"]

        # Aggregate by date
        daily = df.groupby("date")["signed_shares"].sum()
        daily = daily.reindex(cache.index, fill_value=0)

        # Rolling 90-day net insider activity
        rolling = daily.rolling(window=90, min_periods=1).sum()

        # Normalize to [-1, 1]
        max_abs = rolling.abs().max()
        if max_abs > 0:
            signal = rolling / max_abs
        else:
            signal = rolling * 0.0

        signal.name = "inst_insider_signal"
        return signal

    except Exception as exc:
        logger.debug("Insider signal computation failed: %s", exc)
        return pd.Series(np.nan, index=cache.index, name="inst_insider_signal")


# ---------------------------------------------------------------------------
# 5. Amihud Illiquidity (Amihud 2002)
# ---------------------------------------------------------------------------


def _compute_amihud_illiquidity(
    cache: pd.DataFrame,
    window: int = 21,
) -> pd.Series:
    """Daily Amihud illiquidity ratio: avg(|return| / dollar_volume).

    Higher values = less liquid = more vulnerable to large institutional
    trades moving the price.  This is the only time-varying institutional
    column (computed from daily returns and volume, not quarterly ownership
    snapshots).

    Ref: Amihud, Y. (2002). Illiquidity and stock returns.
    """
    ret = cache.get("return_1d")
    close = cache.get("close")
    volume = cache.get("volume")

    if ret is None or close is None or volume is None:
        return pd.Series(np.nan, index=cache.index, name="inst_amihud_illiquidity")

    dollar_volume = (close * volume).clip(lower=1.0)

    with np.errstate(divide="ignore", invalid="ignore"):
        daily_illiq = ret.abs() / dollar_volume

    daily_illiq = daily_illiq.replace([np.inf, -np.inf], np.nan)

    # Rolling mean over window
    amihud = daily_illiq.rolling(window=window, min_periods=5).mean()
    amihud.name = "inst_amihud_illiquidity"
    return amihud


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def compute_institutional_flow(
    cache: pd.DataFrame,
    insider_transactions: list[dict[str, Any]] | None = None,
) -> pd.DataFrame:
    """Compute all institutional flow features and add them to the cache.

    Parameters
    ----------
    cache:
        Daily cache DataFrame with ``inst_ownership_pct``,
        ``inst_top5_concentration``, ``inst_holder_count`` columns
        (from ``get_holder_history()`` merged at Step 4d).
        Also needs ``return_1d``, ``close``, ``volume`` for Amihud.
    insider_transactions:
        Optional list of insider transaction dicts (from
        ``get_insider_transactions()``).  US only.

    Returns
    -------
    pd.DataFrame
        Input cache with institutional flow columns added.
    """
    result = cache.copy()

    # Check if we have any institutional data at all
    has_inst = any(
        col in result.columns and result[col].notna().any()
        for col in ("inst_ownership_pct", "inst_top5_concentration")
    )

    if not has_inst:
        logger.info("No inst_* columns in cache -- skipping institutional flow computation")
        return result

    # 1. Flow momentum
    momentum = _compute_flow_momentum(result)
    result["inst_flow_momentum"] = momentum
    result["inst_flow_momentum_label"] = _apply_labels(momentum, _FLOW_LABELS)

    # 2. Crowding risk (needs flow momentum computed first)
    crowding = _compute_crowding_risk(result)
    result["inst_crowding_risk"] = crowding
    result["inst_crowding_risk_label"] = _apply_labels(crowding, _CROWDING_LABELS)

    # 3. Smart money signal
    smart = _compute_smart_money_signal(result)
    result["inst_smart_money_signal"] = smart
    result["inst_smart_money_label"] = _apply_labels(smart, _SMART_MONEY_LABELS)

    # 4. Insider signal
    insider = _compute_insider_signal(result, insider_transactions)
    result["inst_insider_signal"] = insider
    result["inst_insider_label"] = _apply_labels(insider, _INSIDER_LABELS)

    # 5. Amihud illiquidity (daily time-varying)
    amihud = _compute_amihud_illiquidity(result)
    result["inst_amihud_illiquidity"] = amihud

    # Summary log
    n_flow = momentum.notna().sum()
    n_crowd = crowding.notna().sum()
    n_smart = smart.notna().sum()
    n_insider = insider.notna().sum()
    n_amihud = amihud.notna().sum()

    logger.info(
        "Institutional flow: momentum=%d, crowding=%d, smart_money=%d, "
        "insider=%d, amihud=%d non-NaN days",
        n_flow, n_crowd, n_smart, n_insider, n_amihud,
    )

    return result
