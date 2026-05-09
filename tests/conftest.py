"""Session-scoped fixtures for integration testing.

Expensive computations (FH scoring, survival detection) run once per
pytest session via session-scoped fixtures and are reused across all
integration test files.

Golden AAPL data is loaded from frozen parquet fixtures in tests/fixtures/.
If fixtures don't exist, tests that depend on them are skipped.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import yaml

logger = logging.getLogger(__name__)

FIXTURE_DIR = Path(__file__).parent / "fixtures"


def _load_expected_ranges() -> dict:
    """Load expected output ranges from YAML config."""
    path = FIXTURE_DIR / "expected_ranges.yml"
    if not path.exists():
        return {}
    with open(path, "r", encoding="utf-8") as fh:
        return yaml.safe_load(fh) or {}


EXPECTED = _load_expected_ranges()
AAPL_EXPECTED = EXPECTED.get("aapl", {})


# ---------------------------------------------------------------------------
# Synthetic cache fixture (always available, no golden data needed)
# ---------------------------------------------------------------------------


@pytest.fixture(scope="session")
def synthetic_cache() -> pd.DataFrame:
    """Build a minimal synthetic daily cache for degradation tests.

    Contains just enough columns for survival_mode, financial_health,
    and monte_carlo to run without crashing.
    """
    rng = np.random.RandomState(42)
    n = 252
    dates = pd.bdate_range("2023-01-02", periods=n, freq="B")
    close = 100.0 + np.cumsum(rng.randn(n) * 0.5)
    returns = np.diff(close, prepend=close[0]) / np.maximum(close, 1e-9)

    df = pd.DataFrame(
        {
            "close": close,
            "open": close * (1 + rng.randn(n) * 0.002),
            "high": close * (1 + np.abs(rng.randn(n) * 0.01)),
            "low": close * (1 - np.abs(rng.randn(n) * 0.01)),
            "volume": rng.uniform(1e6, 1e8, n),
            "return_1d": returns,
            "volatility_21d": pd.Series(returns).rolling(21).std().values,
            "current_ratio": rng.uniform(0.8, 2.5, n),
            "debt_to_equity_abs": rng.uniform(0.5, 3.0, n),
            "fcf_yield": rng.uniform(-0.02, 0.08, n),
            "drawdown_252d": rng.uniform(-0.3, 0.0, n),
            "gross_margin": rng.uniform(0.3, 0.8, n),
            "net_margin": rng.uniform(0.05, 0.3, n),
            "operating_margin": rng.uniform(0.1, 0.4, n),
            "revenue": rng.uniform(5e9, 1e11, n),
            "net_income": rng.uniform(1e9, 3e10, n),
            "total_assets": rng.uniform(1e10, 4e11, n),
            "total_liabilities": rng.uniform(5e9, 3e11, n),
            "total_equity": rng.uniform(5e9, 1e11, n),
            "cash_and_equivalents": rng.uniform(1e9, 5e10, n),
            "total_debt": rng.uniform(1e9, 1e11, n),
            "operating_cash_flow": rng.uniform(1e9, 3e10, n),
            "interest_expense": rng.uniform(1e7, 1e9, n),
            "ebit": rng.uniform(1e9, 3e10, n),
            "shares_outstanding": np.full(n, 15e9),
            "market_cap": close * 15e9,
            "pe_ratio_calc": rng.uniform(15, 35, n),
        },
        index=dates,
    )
    df.index.name = "date"
    return df


# ---------------------------------------------------------------------------
# Golden AAPL fixtures (require generate_golden_data.py to have been run)
# ---------------------------------------------------------------------------


@pytest.fixture(scope="session")
def aapl_cache() -> pd.DataFrame:
    """Frozen AAPL daily cache after L1 feature engineering."""
    path = FIXTURE_DIR / "aapl_cache.parquet"
    if not path.exists():
        pytest.skip(
            "Golden AAPL fixture not found. "
            "Run: python tests/generate_golden_data.py"
        )
    return pd.read_parquet(path)


@pytest.fixture(scope="session")
def aapl_statements() -> dict[str, pd.DataFrame]:
    """Frozen AAPL raw statement DataFrames."""
    stmts: dict[str, pd.DataFrame] = {}
    for name in ("income", "balance", "cashflow"):
        path = FIXTURE_DIR / f"aapl_{name}.parquet"
        stmts[name] = pd.read_parquet(path) if path.exists() else pd.DataFrame()
    return stmts


@pytest.fixture(scope="session")
def aapl_profile() -> dict:
    """Frozen AAPL target profile dict."""
    path = FIXTURE_DIR / "aapl_profile.json"
    if not path.exists():
        return {"name": "Apple Inc", "ticker": "AAPL", "sector": "Technology"}
    return json.loads(path.read_text())
