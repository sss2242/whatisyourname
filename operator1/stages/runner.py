"""Stage runner -- dispatches sub-stages with checkpoint save/resume.

Usage from CLI:
    python -m operator1.stages.runner --stage 3.1 --run-dir cache/AAPL
    python -m operator1.stages.runner --stage 3-6 --run-dir cache/AAPL
    python -m operator1.stages.runner --stage all --run-dir cache/AAPL

Usage from main.py:
    from operator1.stages.runner import run_stages
    run_stages(state, "3", "6")   # run stages 3 through 6
    run_stages(state, "4.1")      # run just forecasting
"""

from __future__ import annotations

import logging
import time
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from operator1.pipeline_state import PipelineState

logger = logging.getLogger("operator1.stages.runner")


# ---------------------------------------------------------------------------
# Registry: all sub-stages in execution order
# ---------------------------------------------------------------------------

def _build_registry() -> list[tuple[str, callable]]:
    """Build the full ordered registry of sub-stages.

    Imports are deferred to avoid circular imports and to only load
    stage modules when they are actually needed.
    """
    from operator1.stages.stage3_temporal import STAGE_3_SUBSTAGES
    from operator1.stages.stage4_forecasting import STAGE_4_SUBSTAGES
    from operator1.stages.stage5_forward import STAGE_5_SUBSTAGES
    from operator1.stages.stage6_ensemble import STAGE_6_SUBSTAGES
    from operator1.stages.stage7_integration import STAGE_7_SUBSTAGES

    return (
        STAGE_3_SUBSTAGES
        + STAGE_4_SUBSTAGES
        + STAGE_5_SUBSTAGES
        + STAGE_6_SUBSTAGES
        + STAGE_7_SUBSTAGES
    )


# Map sub-stage IDs to the previous sub-stage for state loading
def _build_dep_map(registry: list[tuple[str, callable]]) -> dict[str, str | None]:
    """Build dependency map: sub_stage -> previous sub_stage."""
    deps: dict[str, str | None] = {}
    prev = None
    for sub_id, _ in registry:
        deps[sub_id] = prev
        prev = sub_id
    return deps


def _parse_stage_spec(spec: str) -> tuple[str | None, str | None]:
    """Parse a stage spec like '3', '4.1', '3-6', 'all'.

    Returns (start, end) where each is a stage/sub-stage ID or None.
    """
    spec = spec.strip()
    if spec == "all":
        return (None, None)
    if "-" in spec:
        parts = spec.split("-", 1)
        return (parts[0].strip(), parts[1].strip())
    return (spec, spec)


def _matches_stage(sub_id: str, stage_spec: str) -> bool:
    """Check if a sub-stage ID matches a stage spec.

    '3' matches '3.1', '3.2', etc.
    '3.1' matches only '3.1'.
    '4' matches '4.1'.
    '7.4' matches '7.4.0', '7.4.1', etc.
    """
    if sub_id == stage_spec:
        return True
    # Prefix match: spec "3" matches "3.1", spec "7.4" matches "7.4.0"
    if sub_id.startswith(stage_spec + "."):
        return True
    return False


def _filter_substages(
    registry: list[tuple[str, callable]],
    start: str | None,
    end: str | None,
) -> list[tuple[str, callable]]:
    """Filter registry to include sub-stages between start and end (inclusive).

    Returns an empty list if the spec matches nothing, preventing
    accidental execution of all stages on typos like ``--stage foo``.
    """
    if start is None and end is None:
        return registry  # all

    # Find start index
    start_idx = -1
    if start is not None:
        for i, (sub_id, _) in enumerate(registry):
            if _matches_stage(sub_id, start):
                start_idx = i
                break
        if start_idx == -1:
            logger.warning("Stage spec '%s' matched no sub-stages", start)
            return []
    else:
        start_idx = 0

    # Find end index
    end_idx = -1
    if end is not None:
        for i in range(len(registry) - 1, -1, -1):
            if _matches_stage(registry[i][0], end):
                end_idx = i
                break
        if end_idx == -1:
            logger.warning("Stage spec '%s' matched no sub-stages", end)
            return []
    else:
        end_idx = len(registry) - 1

    return registry[start_idx:end_idx + 1]


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

# ---------------------------------------------------------------------------
# Infrastructure: graceful degradation, timeout, validation
# ---------------------------------------------------------------------------

# Critical sub-stages that MUST succeed -- pipeline aborts if they fail.
# Non-critical sub-stages log a warning and continue if they fail.
_CRITICAL_SUBSTAGES: set[str] = {
    "3.1",   # regime detection (regime_label needed by everything)
    "4.1",   # forecasting (predictions needed for aggregation)
    "5.1",   # forward pass (model states, PID, calibrator)
    "5.4",   # Monte Carlo (survival probability is core output)
    "6.5",   # prediction aggregation (final ensemble)
    "7.5",   # hedge fund (parallel track but core for profile)
}

# Per-sub-stage timeout in seconds. Prevents hanging models from blocking pipeline.
_SUBSTAGE_TIMEOUTS: dict[str, int] = {
    "4.1": 300,   # forecasting: 5 min (LSTM + tree cascade)
    "5.4": 300,   # Monte Carlo: 5 min (10K paths)
    "6.1": 180,   # transformer: 3 min
    "6.9": 120,   # genetic optimizer: 2 min
    "7.4.1": 180, # MF annual: 3 min
    "7.4.2": 180, # MF quarterly: 3 min
    "7.4.3": 120, # MF monthly: 2 min
    "7.5": 300,   # hedge fund: 5 min
}
_DEFAULT_TIMEOUT: int = 120  # 2 min for everything else

# Required state fields per sub-stage. Validated before dispatch.
_SUBSTAGE_REQUIREMENTS: dict[str, list[str]] = {
    "3.1": ["cache"],
    "4.1": ["cache", "extra_vars"],
    "5.1": ["cache", "forecast_result", "weights"],
    "5.4": ["cache"],
    "6.3": ["cache", "forward_pass_result"],
    "6.5": ["cache", "forecast_result"],
    "7.5": ["cache", "income_df", "balance_df", "cashflow_df"],
}


def _validate_state_for_substage(state: "PipelineState", sub_id: str) -> None:
    """Check that required fields are populated before running a sub-stage."""
    required = _SUBSTAGE_REQUIREMENTS.get(sub_id, [])
    missing = [f for f in required if getattr(state, f, None) is None]
    if missing:
        raise ValueError(
            f"Sub-stage {sub_id} requires state fields {missing} "
            f"but they are None. Run prior stages first."
        )


def run_stages(
    state: "PipelineState",
    stage_spec: str = "all",
    *,
    save_checkpoints: bool = True,
) -> None:
    """Run sub-stages matching the spec, with optional checkpoint save.

    Args:
        state: PipelineState object (must have cache populated for stages 3+)
        stage_spec: Which stages to run. Examples:
            'all'  -- run all stages 3-6
            '3'    -- run all stage 3 sub-stages (3.1-3.7)
            '4.1'  -- run just forecasting
            '3-6'  -- run stages 3 through 6
            '5.3'  -- run just walk-forward
        save_checkpoints: If True, save state to disk after each sub-stage
    """
    registry = _build_registry()
    start, end = _parse_stage_spec(stage_spec)
    substages = _filter_substages(registry, start, end)

    if not substages:
        logger.warning("No sub-stages matched spec '%s'", stage_spec)
        return

    dep_map = _build_dep_map(registry)

    # If resuming (first sub-stage has a dependency and state.cache is empty),
    # load from the previous checkpoint
    first_id = substages[0][0]
    if state.cache is None and dep_map.get(first_id) is not None:
        prev = dep_map[first_id]
        logger.info("Loading checkpoint from sub-stage %s", prev)
        state.load_checkpoint(prev)

    total = len(substages)
    skipped: list[str] = []

    for i, (sub_id, func) in enumerate(substages, 1):
        logger.info("=" * 60)
        logger.info("[%d/%d] Running sub-stage %s", i, total, sub_id)
        logger.info("=" * 60)

        # Pre-flight validation: check required state fields exist
        try:
            _validate_state_for_substage(state, sub_id)
        except ValueError as val_exc:
            if sub_id in _CRITICAL_SUBSTAGES:
                logger.error("CRITICAL validation failed for %s: %s", sub_id, val_exc)
                raise
            logger.warning("Validation failed for %s (non-critical, skipping): %s", sub_id, val_exc)
            skipped.append(sub_id)
            continue

        t0 = time.time()
        timeout = _SUBSTAGE_TIMEOUTS.get(sub_id, _DEFAULT_TIMEOUT)

        try:
            # Run with timeout via concurrent.futures
            from concurrent.futures import ThreadPoolExecutor, TimeoutError as FuturesTimeout
            with ThreadPoolExecutor(max_workers=1) as pool:
                future = pool.submit(func, state)
                future.result(timeout=timeout)
        except FuturesTimeout:
            elapsed = time.time() - t0
            msg = f"Sub-stage {sub_id} timed out after {timeout}s (elapsed: {elapsed:.1f}s)"
            if sub_id in _CRITICAL_SUBSTAGES:
                logger.error("CRITICAL timeout: %s", msg)
                if save_checkpoints:
                    state.save(f"{sub_id}_failed")
                raise TimeoutError(msg)
            logger.warning("Non-critical timeout: %s (continuing)", msg)
            skipped.append(sub_id)
            continue
        except Exception as exc:
            elapsed = time.time() - t0
            # Graceful degradation: non-critical sub-stages can fail
            if sub_id not in _CRITICAL_SUBSTAGES:
                logger.warning(
                    "Non-critical sub-stage %s failed after %.1fs (continuing): %s",
                    sub_id, elapsed, exc,
                )
                skipped.append(sub_id)
                if save_checkpoints:
                    state.save(f"{sub_id}_skipped")
                continue
            # Critical failure: abort pipeline
            logger.error("CRITICAL sub-stage %s FAILED after %.1fs: %s", sub_id, elapsed, exc)
            import traceback
            traceback.print_exc()
            if save_checkpoints:
                state.save(f"{sub_id}_failed")
            raise

        elapsed = time.time() - t0
        logger.info("Sub-stage %s completed in %.1fs", sub_id, elapsed)

        if save_checkpoints:
            state.save(sub_id)

    if skipped:
        logger.info("Pipeline completed with %d skipped sub-stages: %s", len(skipped), skipped)


def get_available_stages() -> list[str]:
    """Return list of all available sub-stage IDs."""
    return [sub_id for sub_id, _ in _build_registry()]


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------

def main() -> int:
    """Run stages from the command line."""
    import argparse

    parser = argparse.ArgumentParser(
        description="Run pipeline sub-stages with checkpoint save/resume",
    )
    parser.add_argument(
        "--stage", type=str, default="all",
        help="Stage spec: 'all', '3', '4.1', '3-6', etc.",
    )
    parser.add_argument(
        "--run-dir", type=str, required=True,
        help="Directory containing pipeline state checkpoints",
    )
    parser.add_argument(
        "--no-save", action="store_true",
        help="Disable checkpoint saving (run in-memory only)",
    )
    parser.add_argument(
        "--list", action="store_true", dest="list_stages",
        help="List all available sub-stages and exit",
    )
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)-7s | %(name)s | %(message)s",
        datefmt="%H:%M:%S",
    )

    if args.list_stages:
        for sid in get_available_stages():
            print(sid)
        return 0

    from operator1.pipeline_state import PipelineState

    state = PipelineState(output_dir=args.run_dir)

    # Load the latest checkpoint or the dependency of the first requested stage
    registry = _build_registry()
    start, _ = _parse_stage_spec(args.stage)
    if start is not None:
        dep_map = _build_dep_map(registry)
        # Find the first matching sub-stage
        for sub_id, _ in registry:
            if _matches_stage(sub_id, start):
                prev = dep_map.get(sub_id)
                if prev is not None:
                    state.load_checkpoint(prev)
                else:
                    # First stage -- try loading whatever exists
                    latest = state.find_latest_checkpoint()
                    if latest:
                        state.load_checkpoint(latest)
                break
    else:
        latest = state.find_latest_checkpoint()
        if latest:
            state.load_checkpoint(latest)

    try:
        run_stages(state, args.stage, save_checkpoints=not args.no_save)
    except Exception:
        return 1

    return 0


if __name__ == "__main__":
    import sys
    sys.exit(main())
