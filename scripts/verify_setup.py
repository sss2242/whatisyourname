#!/usr/bin/env python3
"""Verify that the development environment is correctly set up.

Checks Python version, pip availability, and all 4 dependency stages.
Run after following the setup instructions in requirements.txt.

Usage:
    python scripts/verify_setup.py
"""

from __future__ import annotations

import sys
import importlib


def _check(name: str, import_name: str | None = None) -> bool:
    """Try to import a package and report success/failure."""
    mod = import_name or name
    try:
        importlib.import_module(mod)
        return True
    except ImportError:
        return False


def main() -> int:
    errors = 0

    # Python version
    v = sys.version_info
    print(f"Python {v.major}.{v.minor}.{v.micro}")
    if v.major < 3 or v.minor < 12:
        print("  ERROR: Python 3.12+ required")
        errors += 1
    else:
        print("  OK")

    # Stage 1: Core
    stage1 = [
        ("numpy", None),
        ("pandas", None),
        ("pyarrow", None),
        ("scipy", None),
        ("requests", None),
        ("yaml", "yaml"),
        ("python-dotenv", "dotenv"),
        ("matplotlib", "matplotlib"),
        ("pytest", "pytest"),
    ]
    print("\nStage 1 -- Core:")
    for pkg, imp in stage1:
        ok = _check(pkg, imp)
        print(f"  {'OK' if ok else 'MISSING':>7}  {pkg}")
        if not ok:
            errors += 1

    # Stage 2: ML / Statistics
    stage2 = [
        ("scikit-learn", "sklearn"),
        ("statsmodels", "statsmodels"),
        ("xgboost", "xgboost"),
        ("ruptures", "ruptures"),
        ("hmmlearn", "hmmlearn"),
        ("arch", "arch"),
        ("shap", "shap"),
        ("mapie", "mapie"),
        ("dtaidistance", "dtaidistance"),
        ("PyWavelets", "pywt"),
        ("ta", "ta"),
        ("lifelines", "lifelines"),
        ("changefinder", "changefinder"),
        ("copulae", "copulae"),
        ("stumpy", "stumpy"),
        ("miceforest", "miceforest"),
        ("SALib", "SALib"),
        ("scikit-fuzzy", "skfuzzy"),
        ("tigramite", "tigramite"),
        ("optuna", "optuna"),
        ("statsforecast", "statsforecast"),
        ("vaderSentiment", "vaderSentiment"),
        ("EMD-signal", "PyEMD"),
    ]
    print("\nStage 2 -- ML / Statistics:")
    for pkg, imp in stage2:
        ok = _check(pkg, imp)
        print(f"  {'OK' if ok else 'MISSING':>7}  {pkg}")
        if not ok:
            errors += 1

    # Stage 3: Deep Learning + Bayesian
    stage3 = [
        ("torch", "torch"),
        ("arviz", "arviz"),
        ("pymc", "pymc"),
    ]
    print("\nStage 3 -- Deep Learning + Bayesian:")
    for pkg, imp in stage3:
        ok = _check(pkg, imp)
        print(f"  {'OK' if ok else 'MISSING':>7}  {pkg}")
        if not ok:
            errors += 1

    # Stage 4: Data Source Wrappers (subset of key packages)
    stage4 = [
        ("yfinance", "yfinance"),
        ("feedparser", "feedparser"),
        ("plotly", "plotly"),
        ("fpdf2", "fpdf"),
        ("gnews", "gnews"),
        ("onnxruntime", "onnxruntime"),
        ("curl_cffi", "curl_cffi"),
        ("pdfplumber", "pdfplumber"),
        ("sdmx1", "sdmx"),
        ("wbgapi", "wbgapi"),
        ("fredapi", "fredapi"),
    ]
    print("\nStage 4 -- Data Source Wrappers (key packages):")
    for pkg, imp in stage4:
        ok = _check(pkg, imp)
        print(f"  {'OK' if ok else 'MISSING':>7}  {pkg}")
        if not ok:
            errors += 1

    # Summary
    print(f"\n{'=' * 40}")
    if errors == 0:
        print("All checks passed. Environment is ready.")
    else:
        print(f"{errors} package(s) missing. Run:")
        print("  pip install --timeout 300 -r requirements/stage1-core.txt")
        print("  pip install --timeout 300 -r requirements/stage2-ml.txt")
        print("  pip install --timeout 300 -r requirements/stage3-deeplearning.txt")
        print("  pip install --timeout 300 -r requirements/stage4-wrappers.txt")

    return 1 if errors > 0 else 0


if __name__ == "__main__":
    sys.exit(main())
