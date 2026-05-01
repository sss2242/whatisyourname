"""Information theory / complexity signals for time-series predictability.

Features:
  1. sample_entropy_21d: Regularity measure (Richman & Moorman 2000)
  2. approx_entropy_price: Price predictability (Pincus 1991, PNAS)
  3. lz_complexity: Randomness measure (Lempel & Ziv 1976)
  4. perm_entropy_21d: Ordinal pattern entropy (Bandt & Pompe 2002, PRL)

These features tell temporal models HOW predictable a series is, enabling
adaptive confidence: high entropy = low predictability = wider conformal bands.

Implementation inlined from raphaelvallat/antropy (367 stars) and
blue-yonder/tsfresh (9,183 stars) to avoid adding dependencies.

Pipeline step: Step 5i.8 (after behavioral_signals, before feature_normalization)
"""

from __future__ import annotations

import logging
from math import factorial, log

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Inlined entropy implementations (from antropy + tsfresh patterns)
# ---------------------------------------------------------------------------

def _sample_entropy(x: np.ndarray, order: int = 2, r: float | None = None) -> float:
    """Sample entropy (Richman & Moorman 2000).

    Inlined from tsfresh pattern -- dependency-free, works for short windows.
    """
    x = np.asarray(x, dtype=float)
    if np.isnan(x).any() or len(x) < 10:
        return float("nan")

    if r is None:
        r = 0.2 * np.std(x)
    if r < 1e-12:
        return float("nan")

    n = len(x)
    # Build template vectors of length m and m+1
    # Count matches within tolerance r (Chebyshev / max distance)
    def _count_matches(template_len: int) -> int:
        count = 0
        templates = np.array([x[i: i + template_len] for i in range(n - template_len + 1)])
        for i in range(len(templates)):
            for j in range(i + 1, len(templates)):
                if np.max(np.abs(templates[i] - templates[j])) < r:
                    count += 1
        return count

    a = _count_matches(order + 1)
    b = _count_matches(order)

    if b == 0:
        return float("nan")
    if a == 0:
        # No matches at m+1 -- maximum entropy for this data
        return float("nan")

    return -log(a / b)


def _approx_entropy(x: np.ndarray, order: int = 2, r: float | None = None) -> float:
    """Approximate entropy (Pincus 1991).

    Similar to sample entropy but counts self-matches (biased upward).
    Used on PRICE series (not returns) for different signal.
    """
    x = np.asarray(x, dtype=float)
    if np.isnan(x).any() or len(x) < 10:
        return float("nan")

    if r is None:
        r = 0.2 * np.std(x)
    if r < 1e-12:
        return float("nan")

    n = len(x)

    def _phi(m: int) -> float:
        templates = np.array([x[i: i + m] for i in range(n - m + 1)])
        n_t = len(templates)
        counts = np.zeros(n_t)
        for i in range(n_t):
            for j in range(n_t):
                if np.max(np.abs(templates[i] - templates[j])) <= r:
                    counts[i] += 1
        # Avoid log(0) -- if any count is 0 (shouldn't happen since self-match)
        counts = np.maximum(counts, 1)
        return np.mean(np.log(counts / n_t))

    return abs(_phi(order) - _phi(order + 1))


def _lz_complexity(binary_string: str, normalize: bool = True) -> float:
    """Lempel-Ziv 1976 complexity.

    Counts distinct substrings encountered scanning left to right.
    Input MUST be a binary string (e.g. "10110010").
    Uses the Kaspar & Schuster (1987) implementation which is robust
    against index boundary issues.
    """
    n = len(binary_string)
    if n < 2:
        return float("nan")

    s = binary_string + "0"  # sentinel to handle boundary
    vocab: set[str] = set()
    w = ""
    c = 0
    for char in s[:n]:  # only iterate original length
        wc = w + char
        if wc in vocab:
            w = wc
        else:
            vocab.add(wc)
            c += 1
            w = ""
    if w:
        c += 1

    if normalize:
        normalizer = n / max(np.log2(max(n, 2)), 1e-12)
        return c / normalizer
    return float(c)


def _perm_entropy(
    x: np.ndarray, order: int = 3, delay: int = 1, normalize: bool = True
) -> float:
    """Permutation entropy (Bandt & Pompe 2002).

    Counts ordinal pattern frequencies. Most robust entropy for short,
    noisy financial series (uses only rank order, not magnitudes).
    """
    x = np.asarray(x, dtype=float)
    n = len(x)
    n_patterns = n - (order - 1) * delay

    if n_patterns < 5:
        return float("nan")

    # Build ordinal patterns
    pattern_counts: dict[tuple, int] = {}
    for i in range(n_patterns):
        pattern = tuple(np.argsort(x[i: i + order * delay: delay]))
        pattern_counts[pattern] = pattern_counts.get(pattern, 0) + 1

    # Convert to probabilities
    total = sum(pattern_counts.values())
    probs = np.array([c / total for c in pattern_counts.values()])

    # Shannon entropy
    h = -np.sum(probs * np.log2(probs + 1e-15))

    if normalize:
        h_max = np.log2(factorial(order))
        if h_max > 0:
            h /= h_max

    return float(h)


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def compute_complexity_signals(
    cache: pd.DataFrame,
    window: int = 63,
    price_window: int = 126,
) -> pd.DataFrame:
    """Compute information-theoretic complexity features.

    Parameters
    ----------
    cache:
        Daily cache with ``return_1d`` and optionally ``close``.
    window:
        Rolling window for return-based entropy (default 63 = 1 quarter).
    price_window:
        Rolling window for price-based entropy (default 126 = 2 quarters).

    Returns
    -------
    pd.DataFrame
        Cache augmented with 4 entropy/complexity columns.
    """
    result = cache.copy()
    ret = result.get("return_1d")

    if ret is None or ret.notna().sum() < window + 10:
        logger.info("Complexity signals: insufficient data (need %d+ return observations)", window + 10)
        return result

    ret_arr = ret.fillna(0).values

    # ------------------------------------------------------------------
    # 1. Sample entropy of returns (rolling)
    # High = complex/unpredictable, Low = regular/predictable
    # ------------------------------------------------------------------
    se_vals = np.full(len(ret_arr), np.nan)
    for i in range(window, len(ret_arr)):
        chunk = ret_arr[i - window: i]
        se_vals[i] = _sample_entropy(chunk, order=2)
    result["sample_entropy_21d"] = se_vals

    # ------------------------------------------------------------------
    # 2. Permutation entropy of returns (rolling)
    # Most robust for short noisy financial series
    # ------------------------------------------------------------------
    pe_vals = np.full(len(ret_arr), np.nan)
    for i in range(window, len(ret_arr)):
        chunk = ret_arr[i - window: i]
        pe_vals[i] = _perm_entropy(chunk, order=3, delay=1)
    result["perm_entropy_21d"] = pe_vals

    # ------------------------------------------------------------------
    # 3. Lempel-Ziv complexity of binarized returns (rolling)
    # Near 1.0 = near-random walk, near 0.0 = highly patterned
    # ------------------------------------------------------------------
    lz_vals = np.full(len(ret_arr), np.nan)
    for i in range(window, len(ret_arr)):
        chunk = ret_arr[i - window: i]
        binary = "".join(["1" if v > 0 else "0" for v in chunk])
        lz_vals[i] = _lz_complexity(binary, normalize=True)
    result["lz_complexity"] = lz_vals

    # ------------------------------------------------------------------
    # 4. Approximate entropy of PRICE (not returns -- different signal)
    # Captures price-level predictability vs return predictability
    # ------------------------------------------------------------------
    close = result.get("close")
    if close is not None and close.notna().sum() > price_window + 10:
        close_arr = close.fillna(method="ffill").values
        ae_vals = np.full(len(close_arr), np.nan)
        for i in range(price_window, len(close_arr)):
            chunk = close_arr[i - price_window: i]
            ae_vals[i] = _approx_entropy(chunk, order=2)
        result["approx_entropy_price"] = ae_vals

    n_new = len(result.columns) - len(cache.columns)
    if n_new > 0:
        logger.info("Complexity signals computed: %d columns", n_new)

    return result
