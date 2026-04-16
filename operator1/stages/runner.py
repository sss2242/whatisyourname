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
    """
    if sub_id == stage_spec:
        return True
    if "." not in stage_spec and sub_id.startswith(stage_spec + "."):
        return True
    return False


def _filter_substages(
    registry: list[tuple[str, callable]],
    start: str | None,
    end: str | None,
) -> list[tuple[str, callable]]:
    """Filter registry to include sub-stages between start and end (inclusive)."""
    if start is None and end is None:
        return registry  # all

    # Find start index
    start_idx = 0
    if start is not None:
        for i, (sub_id, _) in enumerate(registry):
            if _matches_stage(sub_id, start):
                start_idx = i
                break

    # Find end index
    end_idx = len(registry) - 1
    if end is not None:
        for i in range(len(registry) - 1, -1, -1):
            if _matches_stage(registry[i][0], end):
                end_idx = i
                break

    return registry[start_idx:end_idx + 1]


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

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
    for i, (sub_id, func) in enumerate(substages, 1):
        logger.info("=" * 60)
        logger.info("[%d/%d] Running sub-stage %s", i, total, sub_id)
        logger.info("=" * 60)

        t0 = time.time()
        try:
            func(state)
        except Exception as exc:
            logger.error("Sub-stage %s FAILED: %s", sub_id, exc)
            import traceback
            traceback.print_exc()
            # Save what we have so far
            if save_checkpoints:
                state.save(f"{sub_id}_failed")
            raise

        elapsed = time.time() - t0
        logger.info("Sub-stage %s completed in %.1fs", sub_id, elapsed)

        if save_checkpoints:
            state.save(sub_id)


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
