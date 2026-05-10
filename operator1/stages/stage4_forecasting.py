"""Stage 4: Forecasting -- the heavyweight step, isolated for background execution.

Sub-stages:
  4.1  Forecasting (Kalman + GARCH + VAR + LSTM + Tree + Baseline + ETS)

This is the slowest stage (~5-10min). It is intentionally isolated as a
single sub-stage so it can be run via nohup or in a background process
without blocking subsequent lightweight stages.
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from operator1.pipeline_state import PipelineState

logger = logging.getLogger("operator1.stages.stage4")


def run_4_1_forecasting(state: PipelineState) -> None:
    """4.1: Forecasting (6 model types per variable -- Kalman/GARCH/VAR/LSTM/Tree/Baseline)."""
    logger.info("Sub-stage 4.1: Forecasting")
    cache = state.cache
    if cache is None or cache.empty:
        raise ValueError("No cache -- run Stage 3 first")

    if not state.extra_vars:
        from operator1.stages.stage3_temporal import _init_extra_vars
        _init_extra_vars(state)

    try:
        from operator1.models.forecasting import run_forecasting
        cache, state.forecast_result = run_forecasting(
            cache,
            extra_variables=state.extra_vars,
            model_feature_sets=getattr(state, "model_feature_sets", None),
            windows=(
                state.adaptive_tier3.windows
                if state.adaptive_tier3 is not None
                and getattr(state.adaptive_tier3, "adapted", False)
                else None
            ),
        )
        state.cache = cache
        logger.info("Forecasting complete")
    except Exception as exc:
        logger.warning("Forecasting failed: %s", exc)


# Registry
STAGE_4_SUBSTAGES = [
    ("4.1", run_4_1_forecasting),
]
