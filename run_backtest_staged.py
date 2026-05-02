#!/usr/bin/env python3
"""Staged Backtest Compiler -- runs each sub-stage as a separate process.

Calls backtest_runner.py for each sub-stage individually, waits for
completion, shows real-time status, and tracks progress. If a sub-stage
fails, saves the failure point so you can resume from where it stopped.

Usage:
    python run_backtest_staged.py --market us_sec_edgar --company AAPL --end-date 2024-12-31
    python run_backtest_staged.py --resume --run-dir cache/backtest_AAPL_2024-12-31
    python run_backtest_staged.py --market us_sec_edgar --company AAPL --end-date 2024-12-31 --start-from 7.4.5
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path


# ---------------------------------------------------------------------------
# All sub-stages in execution order
# ---------------------------------------------------------------------------

ALL_STAGES = [
    # Backtest Stage 1: Data fetch + cache build + features
    ("1", "Data Fetch + Cache Build + Features"),
    # Backtest Stage 2: Data preprocessing (frequency separation + reconciliation)
    ("2.1", "Frequency Separation (Bayesian + Chow-Lin/Denton)"),
    ("2.2", "Data Reconciliation (identity checks + dedup)"),
    # Backtest Stage 3+: Temporal models (36 sub-stages via staged runner)
    ("3.1", "Regime Detection (HMM/GMM/PELT/BCP/ChangeFinder)"),
    ("3.2", "Dual Regime Mixer"),
    ("3.3", "Granger Causality (PCMCI)"),
    ("3.4", "Transfer Entropy"),
    ("3.5", "Cycle Decomposition (CEEMDAN/FFT)"),
    ("3.6", "Pattern Detection (Candlestick + Matrix Profile)"),
    ("3.7", "Pre-Forecasting Synergies"),
    ("3.8", "Feature Selection (Boruta + PIMP + mRMR)"),
    ("4.1", "Forecasting (Kalman/GARCH/VAR/LSTM/Tree/ETS)"),
    ("5.1", "Forward Pass (PID-controlled walk)"),
    ("5.2", "Burn-Out Calibration"),
    ("5.3", "Walk-Forward Evaluation (MCS + FixedShare)"),
    ("5.4", "Monte Carlo Simulation (10K paths)"),
    ("5.5", "Copula Analysis (Gaussian/Student-t/Clayton)"),
    ("6.1", "Transformer Forecaster"),
    ("6.2", "Particle Filter"),
    ("6.3", "Conformal Prediction (PID + Mondrian + QR)"),
    ("6.4", "DTW Historical Analogs"),
    ("6.5", "Prediction Aggregation (Ensemble)"),
    ("6.6", "SHAP Explainability"),
    ("6.7", "Sobol Sensitivity + Hierarchy Feedback"),
    ("6.8", "Time-Varying Granger + Multivariate MC"),
    ("6.9", "Genetic Optimizer (Optuna TPE)"),
    ("6.10", "OHLC Predictor + Predicted Patterns"),
    ("6.11", "Recursive Day-by-Day Predictions"),
    ("7.1", "USS + Scenario Engine"),
    ("7.2", "Retroactive Calibration"),
    ("7.3", "Model Diagnostics"),
    ("7.4.0", "Multi-Frequency: Resample Prep"),
    ("7.4.1", "Multi-Frequency: Annual Pipeline"),
    ("7.4.2", "Multi-Frequency: Quarterly Pipeline"),
    ("7.4.3", "Multi-Frequency: Monthly Pipeline"),
    ("7.4.4", "Multi-Frequency: Weekly Pipeline"),
    ("7.4.5", "Multi-Frequency: Daily Pipeline"),
    ("7.4.6", "Multi-Frequency: Fusion (13 methods)"),
    ("7.5", "Hedge Fund Analysis (15 metrics + fusion)"),
    # Backtest Stage 3: Profile build + prediction extraction
    ("3:profile", "Profile Build + Report Generation"),
]


# ---------------------------------------------------------------------------
# Terminal helpers
# ---------------------------------------------------------------------------

def _color(text: str, code: str) -> str:
    if os.name == "nt":
        return text
    return f"\033[{code}m{text}\033[0m"

def _green(t: str) -> str: return _color(t, "32")
def _yellow(t: str) -> str: return _color(t, "33")
def _red(t: str) -> str: return _color(t, "31")
def _cyan(t: str) -> str: return _color(t, "36")
def _bold(t: str) -> str: return _color(t, "1")
def _dim(t: str) -> str: return _color(t, "2")


def _progress_bar(current: int, total: int, width: int = 40) -> str:
    filled = int(width * current / total) if total > 0 else 0
    bar = "=" * filled + "-" * (width - filled)
    pct = current / total * 100 if total > 0 else 0
    return f"[{bar}] {pct:.0f}% ({current}/{total})"


# ---------------------------------------------------------------------------
# Stage execution
# ---------------------------------------------------------------------------

def run_single_stage(
    stage_id: str,
    stage_name: str,
    run_dir: str,
    market: str = "",
    company: str = "",
    end_date: str = "",
    years: float = 2.0,
) -> tuple[bool, float]:
    """Run a single stage via backtest_runner.py subprocess.

    Returns (success, elapsed_seconds).
    """
    cmd = [sys.executable, "backtest_runner.py"]

    if stage_id == "1":
        # Stage 1 needs market/company/end-date
        cmd.extend([
            "--stage", "1",
            "--market", market,
            "--company", company,
            "--end-date", end_date,
            "--years", str(years),
        ])
    elif stage_id == "3:profile":
        # Stage 3 (profile build)
        cmd.extend(["--stage", "3", "--run-dir", run_dir])
    else:
        # Sub-stage spec routed through Stage 2
        cmd.extend(["--stage", stage_id, "--run-dir", run_dir])

    t0 = time.time()
    result = subprocess.run(cmd, capture_output=True, text=True)
    elapsed = time.time() - t0

    return result.returncode == 0, elapsed


def find_resume_point(run_dir: str) -> str | None:
    """Find the latest completed checkpoint in run_dir."""
    rd = Path(run_dir)
    if not rd.exists():
        return None

    checkpoints = []
    for pkl in rd.glob("state_*.pkl"):
        sub = pkl.stem.replace("state_", "")
        checkpoints.append((pkl.stat().st_mtime, sub))

    if not checkpoints:
        return None

    checkpoints.sort(reverse=True)
    return checkpoints[0][1]


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> int:
    parser = argparse.ArgumentParser(
        description="Staged Backtest Compiler -- runs each sub-stage separately",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""\
Examples:
  python run_backtest_staged.py --market us_sec_edgar --company AAPL --end-date 2024-12-31
  python run_backtest_staged.py --resume --run-dir cache/backtest_AAPL_2024-12-31
  python run_backtest_staged.py --market us_sec_edgar --company AAPL --end-date 2024-12-31 --start-from 7.4.5
""",
    )
    parser.add_argument("--market", type=str, default="us_sec_edgar")
    parser.add_argument("--company", type=str, default="AAPL")
    parser.add_argument("--end-date", type=str, default="2024-12-31")
    parser.add_argument("--years", type=float, default=2.0)
    parser.add_argument("--run-dir", type=str, default="")
    parser.add_argument("--resume", action="store_true",
                        help="Resume from last checkpoint")
    parser.add_argument("--start-from", type=str, default="",
                        help="Start from a specific sub-stage (e.g., 7.4.5)")
    parser.add_argument("--validate", action="store_true",
                        help="Run validation after completion")
    args = parser.parse_args()

    run_dir = args.run_dir or f"cache/backtest_{args.company}_{args.end_date}"

    print()
    print(_bold(_cyan("  OPERATOR 1 -- Staged Backtest Compiler")))
    print(_dim(f"  Market: {args.market} | Company: {args.company} | End: {args.end_date}"))
    print(_dim(f"  Run dir: {run_dir}"))
    print()

    # Determine start point
    start_idx = 0
    if args.resume:
        last_checkpoint = find_resume_point(run_dir)
        if last_checkpoint:
            # Find the stage AFTER the last checkpoint
            for i, (sid, _) in enumerate(ALL_STAGES):
                if sid == last_checkpoint:
                    start_idx = i + 1
                    break
            print(_yellow(f"  Resuming from after checkpoint: {last_checkpoint}"))
        else:
            print(_yellow("  No checkpoints found, starting from beginning"))
    elif args.start_from:
        for i, (sid, _) in enumerate(ALL_STAGES):
            if sid == args.start_from:
                start_idx = i
                break
        else:
            print(_red(f"  Unknown stage: {args.start_from}"))
            return 1
        print(_yellow(f"  Starting from: {args.start_from}"))

    # Run stages
    total = len(ALL_STAGES)
    completed = start_idx
    failed_stage = None
    total_time = 0.0

    results: list[dict] = []

    for i in range(start_idx, total):
        stage_id, stage_name = ALL_STAGES[i]

        # Status display
        print()
        print(_bold(f"  [{i + 1}/{total}] {_progress_bar(i, total)}"))
        print(f"  {_cyan('RUNNING')} {_bold(stage_id)}: {stage_name}")
        print(_dim(f"  Started: {datetime.now().strftime('%H:%M:%S')}"))

        success, elapsed = run_single_stage(
            stage_id=stage_id,
            stage_name=stage_name,
            run_dir=run_dir,
            market=args.market,
            company=args.company,
            end_date=args.end_date,
            years=args.years,
        )

        total_time += elapsed

        if success:
            completed += 1
            status = _green("DONE")
            results.append({
                "stage": stage_id,
                "name": stage_name,
                "status": "done",
                "elapsed": round(elapsed, 1),
            })
        else:
            status = _red("FAILED")
            failed_stage = stage_id
            results.append({
                "stage": stage_id,
                "name": stage_name,
                "status": "failed",
                "elapsed": round(elapsed, 1),
            })

        print(f"  {status} {stage_id}: {stage_name} ({elapsed:.1f}s)")

        if not success:
            print()
            print(_red(f"  Stage {stage_id} failed after {elapsed:.1f}s"))
            print(_yellow(f"  Resume with: python run_backtest_staged.py --resume --run-dir {run_dir}"))
            break

    # Summary
    print()
    print(_bold("  " + "=" * 60))
    print(_bold("  BACKTEST SUMMARY"))
    print(_bold("  " + "=" * 60))
    print(f"  Stages completed: {completed}/{total}")
    print(f"  Total time: {total_time:.0f}s ({total_time / 60:.1f} min)")
    if failed_stage:
        print(f"  {_red('Failed at')}: {failed_stage}")
    else:
        print(f"  {_green('Status')}: All stages complete")
    print()

    # Show per-stage timing
    print(_bold("  Per-stage timing:"))
    for r in results:
        icon = _green("+") if r["status"] == "done" else _red("X")
        print(f"    {icon} {r['stage']:>8} | {r['elapsed']:>6.1f}s | {r['name']}")

    # Save results
    results_path = Path(run_dir) / "staged_compiler_results.json"
    results_path.parent.mkdir(parents=True, exist_ok=True)
    with open(results_path, "w") as f:
        json.dump({
            "market": args.market,
            "company": args.company,
            "end_date": args.end_date,
            "total_stages": total,
            "completed": completed,
            "total_time_s": round(total_time, 1),
            "failed_stage": failed_stage,
            "stages": results,
            "timestamp": datetime.now().isoformat(),
        }, f, indent=2)

    # Validate if requested
    if args.validate and not failed_stage:
        print()
        print(_bold("  Running validation..."))
        cmd = [sys.executable, "backtest_runner.py", "--validate", "--run-dir", run_dir]
        subprocess.run(cmd)

    return 0 if not failed_stage else 1


if __name__ == "__main__":
    sys.exit(main())
