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
    # Backtest Stage 1: Data fetch + cache build + features (8 sub-stages)
    ("1.1", "Profile + Company Search (PIT client, SEC EDGAR profile)"),
    ("1.2", "Financial Statements (CompanyFacts API: income, balance, cashflow)"),
    ("1.3", "OHLCV + Holders + Segments (yfinance, holders, product segments)"),
    ("1.4a", "Cache Build (OHLCV spine, merge, benchmark, IV, cross-asset, options)"),
    ("1.4b", "Macro + Risk (macro fetch, quadrant, conflict, buying power, pre-ratios)"),
    ("1.5", "Estimation + Derived Variables + Survival (SIX proxies, estimation, features, FH)"),
    ("1.6", "Entity Discovery + Sentiment (LLM entities, graph risk, sentiment)"),
    ("1.7", "Adaptive Calibration (thresholds, model params, windows, signal IC)"),
    ("1.8a", "Regime Detection + Timeline (HMM/GMM/PELT/BCP, ChangeFinder, enriched timeline)"),
    ("1.8b", "Finalization (linked conflict, aggregates, peer ranking, behavioral, normalization)"),
    # Backtest Stage 2: Data preprocessing (frequency separation + reconciliation)
    ("2.1", "Frequency Separation (Bayesian + Chow-Lin/Denton)"),
    ("2.2", "Data Reconciliation (identity checks + dedup)"),
    # Backtest Stage 2-freq: Frequency-first pipeline (each freq runs own full pipeline)
    ("2.0", "Freq Pipeline: Resample Prep (build per-freq caches from raw filings)"),
    ("2.A", "Freq Pipeline: Annual (derived vars + survival + FH + regime + forecast + MC)"),
    ("2.Q", "Freq Pipeline: Quarterly (derived vars + survival + FH + regime + forecast + MC)"),
    ("2.M", "Freq Pipeline: Monthly (interpolated from Q/A)"),
    ("2.W", "Freq Pipeline: Weekly (interpolated from Q/A)"),
    ("2.D", "Freq Pipeline: Daily (OHLCV + stock/stock ratios only)"),
    ("2.F", "Freq Pipeline: Fusion (13-method fusion + forward-fill Q/A ratios to daily cache)"),
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
    ("7.4.7", "Multi-Frequency: Hierarchical Reconciliation (MinTrace mint_shrink)"),
    ("7.5.1", "Hedge Fund: Base Metrics + Scorecard (15 metrics + EVA + SOTP + CVaR + SGR)"),
    ("7.5.2", "Hedge Fund: Advanced Methods (22 techniques: Piotroski, Merton, moat, factors)"),
    ("7.5.3", "Hedge Fund: Multi-Frequency Variants (FCF MF, accruals MF, growth MF)"),
    ("7.5.4", "Hedge Fund: Cross-Pipeline Fusion (11 methods: HRP, Brier, anomaly routing)"),
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

    # First check our own progress file
    progress = _load_progress(run_dir)
    last_idx = progress.get("last_completed_idx")
    if last_idx is not None and isinstance(last_idx, int):
        next_idx = last_idx + 1
        if next_idx < len(ALL_STAGES):
            return ALL_STAGES[next_idx][0]
        return None  # all done

    # Fallback: check PipelineState checkpoints
    checkpoints = []
    for pkl in rd.glob("state_*.pkl"):
        sub = pkl.stem.replace("state_", "")
        checkpoints.append((pkl.stat().st_mtime, sub))

    if not checkpoints:
        return None

    checkpoints.sort(reverse=True)
    return checkpoints[0][1]


# ---------------------------------------------------------------------------
# Progress persistence (for self-restart pattern)
# ---------------------------------------------------------------------------

_PROGRESS_FILENAME = "staged_progress.json"


def _load_progress(run_dir: str) -> dict:
    """Load accumulated progress from prior self-restart invocations."""
    path = Path(run_dir) / _PROGRESS_FILENAME
    if path.exists():
        try:
            with open(path) as f:
                return json.load(f)
        except (json.JSONDecodeError, OSError):
            pass
    return {}


def _save_progress(run_dir: str, progress: dict) -> None:
    """Persist progress to disk between self-restart invocations."""
    rd = Path(run_dir)
    rd.mkdir(parents=True, exist_ok=True)
    progress["timestamp"] = datetime.now().isoformat()
    with open(rd / _PROGRESS_FILENAME, "w") as f:
        json.dump(progress, f, indent=2)


def _print_summary(run_dir: str) -> None:
    """Print accumulated results from all prior invocations."""
    progress = _load_progress(run_dir)
    stages = progress.get("stages", [])
    total_elapsed = progress.get("total_elapsed", 0)
    total = len(ALL_STAGES)
    completed = sum(1 for s in stages if s["status"] == "done")
    failed = [s for s in stages if s["status"] == "failed"]

    print(_bold("  " + "=" * 60))
    print(_bold("  BACKTEST SUMMARY"))
    print(_bold("  " + "=" * 60))
    print(f"  Stages completed: {completed}/{total}")
    print(f"  Total time: {total_elapsed:.0f}s ({total_elapsed / 60:.1f} min)")
    if failed:
        print(f"  {_red('Failed at')}: {failed[-1]['stage']}")
    elif completed >= total:
        print(f"  {_green('Status')}: All stages complete")
    else:
        print(f"  {_yellow('Status')}: In progress ({total - completed} remaining)")
    print()

    if stages:
        print(_bold("  Per-stage timing:"))
        for r in stages:
            icon = _green("+") if r["status"] == "done" else _red("X")
            print(f"    {icon} {r['stage']:>8} | {r['elapsed']:>6.1f}s | {r['name']}")

    # Also save the compiler results JSON
    results_path = Path(run_dir) / "staged_compiler_results.json"
    with open(results_path, "w") as f:
        json.dump({
            "market": progress.get("market", ""),
            "company": progress.get("company", ""),
            "end_date": progress.get("end_date", ""),
            "total_stages": total,
            "completed": completed,
            "total_time_s": round(total_elapsed, 1),
            "failed_stage": failed[-1]["stage"] if failed else None,
            "stages": stages,
            "timestamp": datetime.now().isoformat(),
        }, f, indent=2)


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

    # ------------------------------------------------------------------
    # Self-restart architecture: run ONE sub-stage, save progress,
    # then os.execv() to restart at the next sub-stage.
    # This gives each sub-stage a fresh process timeout window.
    # ------------------------------------------------------------------

    total = len(ALL_STAGES)

    if start_idx >= total:
        print(_green("  All stages already completed."))
        _print_summary(run_dir)
        return 0

    # Load accumulated results from prior sub-stages
    progress = _load_progress(run_dir)

    # Run ONE sub-stage
    i = start_idx
    stage_id, stage_name = ALL_STAGES[i]

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

    # Save progress
    stage_result = {
        "stage": stage_id,
        "name": stage_name,
        "status": "done" if success else "failed",
        "elapsed": round(elapsed, 1),
    }
    progress.setdefault("stages", []).append(stage_result)
    progress["last_completed_idx"] = i if success else i - 1
    progress["last_completed_id"] = stage_id if success else (ALL_STAGES[i - 1][0] if i > 0 else None)
    progress["total_elapsed"] = round(
        sum(s["elapsed"] for s in progress["stages"]), 1,
    )
    progress["market"] = args.market
    progress["company"] = args.company
    progress["end_date"] = args.end_date
    progress["validate"] = args.validate
    _save_progress(run_dir, progress)

    if success:
        print(f"  {_green('DONE')} {stage_id}: {stage_name} ({elapsed:.1f}s)")
    else:
        print(f"  {_red('FAILED')} {stage_id}: {stage_name} ({elapsed:.1f}s)")
        print()
        print(_red(f"  Stage {stage_id} failed after {elapsed:.1f}s"))
        print(_yellow(f"  Resume: python run_backtest_staged.py --resume --run-dir {run_dir}"))
        _print_summary(run_dir)
        return 1

    # If there are more sub-stages, spawn a fully detached process and exit.
    # This gives a true terminal separation: the current process terminates
    # completely, and a brand-new process with its own session starts for
    # the next sub-stage. Platform timeouts reset because the old session
    # is gone.
    next_idx = i + 1
    if next_idx < total:
        next_id = ALL_STAGES[next_idx][0]
        next_name = ALL_STAGES[next_idx][1]

        # Build command for the next sub-stage
        cmd = [
            sys.executable, __file__,
            "--market", args.market,
            "--company", args.company,
            "--end-date", args.end_date,
            "--years", str(args.years),
            "--run-dir", run_dir,
            "--start-from", next_id,
        ]
        if args.validate:
            cmd.append("--validate")

        # Log file for the detached process
        log_dir = Path(run_dir)
        log_dir.mkdir(parents=True, exist_ok=True)
        log_path = log_dir / f"stage_{next_id.replace('.', '_')}.log"

        print()
        print(_bold(f"  Spawning detached process for sub-stage {next_id}: {next_name}"))
        print(_dim(f"  Log: {log_path}"))
        print(_dim(f"  This terminal will now exit. Next sub-stage runs independently."))
        print()
        sys.stdout.flush()
        sys.stderr.flush()

        # Spawn a fully detached process with its own session.
        # start_new_session=True creates a new process group (setsid on Linux),
        # so the child is not killed when this terminal/session closes.
        with open(log_path, "w") as log_file:
            subprocess.Popen(
                cmd,
                stdout=log_file,
                stderr=subprocess.STDOUT,
                start_new_session=True,
                close_fds=True,
            )

        # Exit this process -- the new one runs independently
        print(_green(f"  Detached process spawned for {next_id}. Exiting current session."))
        return 0
    else:
        # All stages done
        print()
        _print_summary(run_dir)

        # Validate if requested
        if args.validate:
            print()
            print(_bold("  Running validation..."))
            cmd = [sys.executable, "backtest_runner.py", "--validate", "--run-dir", run_dir]
            subprocess.run(cmd)

        return 0


if __name__ == "__main__":
    sys.exit(main())
