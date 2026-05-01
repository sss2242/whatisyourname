# GitHub Research: Layer 5 Multi-Frequency Pipeline Implementation Findings

*Researched 2026-05-01 -- GitHub code search for each proposed Layer 5 enhancement*

---

## Enhancement 5.3C: Cross-Frequency Momentum Signal

**Best implementation found:** [`AQR-Research/TSMP`](https://github.com/topics/time-series-momentum) -- Moskowitz, Ooi & Pedersen (2012) implementations in several repos

**Most practical:** No dedicated library needed. The concept is simple:

```python
def cross_frequency_momentum(frequency_results: dict) -> dict:
    # Sign of trend at each frequency (+1 = up, -1 = down, 0 = flat)
    signs = {}
    weights = {"A": 5, "Q": 4, "M": 3, "W": 2, "D": 1}  # slower = more weight
    
    for freq, result in frequency_results.items():
        td = getattr(result, 'trend_direction', 'flat')
        signs[freq] = 1 if td == 'up' else (-1 if td == 'down' else 0)
    
    # Weighted momentum score
    total_weight = sum(weights.get(f, 1) for f in signs)
    score = sum(signs[f] * weights.get(f, 1) for f in signs) / max(total_weight, 1)
    
    # Direction agreement: fraction of frequencies with same sign as majority
    majority_sign = 1 if score > 0 else (-1 if score < 0 else 0)
    n_agree = sum(1 for s in signs.values() if s == majority_sign)
    agreement = n_agree / max(len(signs), 1)
    
    # Potential reversal: long-term and short-term disagree
    long_term = signs.get('A', signs.get('Q', 0))
    short_term = signs.get('D', signs.get('W', 0))
    reversal = long_term != 0 and short_term != 0 and long_term != short_term
    
    return {
        'cross_freq_momentum_score': float(np.clip(score, -1, 1)),
        'cross_freq_direction_agreement': float(agreement),
        'potential_reversal_flag': reversal,
    }
```

**Found in:** [`Stefan-Jansen/machine-learning-for-trading`](https://github.com/Stefan-Jansen/machine-learning-for-trading) (17,213 stars) -- Chapter 4 covers momentum across timeframes. The key insight: momentum strategies that work at monthly frequency also work at weekly and quarterly, but the CONVERGENCE across frequencies is more predictive than any single frequency.

**No new dependency.** ~25 lines.

---

## Enhancement 5.3B: Weighted Regime Voting

**No dedicated GitHub implementation.** Standard weighted voting:

```python
def weighted_regime_vote(frequency_results: dict) -> dict:
    votes = {}  # regime -> total weight
    vote_weights = {}
    
    for freq, result in frequency_results.items():
        regime = getattr(result, 'regime_label', 'unknown')
        mae = getattr(result, 'walk_forward_mae', None)
        n_periods = getattr(result, 'n_periods', 1)
        
        # Weight: inverse MAE * log(data points)
        w = (1.0 / max(mae, 0.01)) * max(np.log(n_periods), 0.1) if mae else 1.0
        vote_weights[freq] = w
        votes[regime] = votes.get(regime, 0) + w
    
    # Normalize weights
    total_w = sum(vote_weights.values())
    vote_weights = {f: w / total_w for f, w in vote_weights.items()}
    
    consensus = max(votes, key=votes.get) if votes else 'unknown'
    return {
        'weighted_regime_consensus': consensus,
        'regime_vote_weights': vote_weights,
    }
```

**No new dependency.** ~20 lines.

---

## Enhancement 5.2A: Uncertainty Propagation in Cascading Context

**Found in:** [`uber/orbit`](https://github.com/uber/orbit) (1,900+ stars) -- Bayesian time series with posterior predictive propagation. The key pattern: pass the DISTRIBUTION, not just the point estimate.

**Also found:** [`pyro-ppl/pyro`](https://github.com/pyro-ppl/pyro) (8,500+ stars) -- probabilistic programming with automatic posterior propagation. Too heavy for our use case.

**Practical implementation for our cascade:**
```python
@dataclass
class FrequencyContext:
    # ... existing fields ...
    # NEW: uncertainty propagation
    forecast_uncertainty: dict[str, float] = field(default_factory=dict)
    # Per-variable standard deviation from MC or conformal at this frequency
    regime_posterior: dict[str, float] = field(default_factory=dict)
    # Distribution over regimes: {"bull": 0.6, "bear": 0.2, "high_vol": 0.2}
```

The next faster frequency uses these as soft priors: `prior_weight = confidence * (1 / freq_rank)`.

**No new dependency.** ~15 lines of context field additions + ~10 lines of prior blending in the runner.

---

## Enhancement 5.3A: Temporal Hierarchical Reconciliation (MinT)

**Best implementation found:** [`Nixtla/hierarchicalforecast`](https://github.com/Nixtla/hierarchicalforecast) (790 stars)
- `hierarchicalforecast.methods.MinTrace` -- optimal MinT reconciliation
- `hierarchicalforecast.methods.BottomUp` -- simple bottom-up aggregation
- Handles both cross-sectional (geographic) and temporal hierarchies

**Key temporal summing matrix pattern:**
```python
def temporal_reconcile(monthly_forecasts, quarterly_forecasts, annual_forecast):
    # Temporal summing matrix S:
    # annual = sum(Q1, Q2, Q3, Q4) = sum(M1...M12)
    # quarterly = sum(M1, M2, M3), sum(M4, M5, M6), etc.
    
    # Simple bottom-up reconciliation (no library needed):
    # Adjust annual to match sum of quarters
    q_sum = sum(quarterly_forecasts)
    if abs(annual_forecast) > 1e-10:
        scale = q_sum / annual_forecast
        # Shrink toward reconciled: blend
        reconciled_annual = 0.7 * annual_forecast + 0.3 * q_sum
    
    # MinT optimal (with hierarchicalforecast):
    # from hierarchicalforecast.methods import MinTrace
    # reconciled = MinTrace().reconcile(Y_hat, S, tags)
    
    return reconciled_annual, quarterly_forecasts  # monthly unchanged (bottom level)
```

**Also found:** [`statsforecast`](https://github.com/Nixtla/statsforecast) (4,000+ stars, already in our deps) -- has reconciliation utilities in its hierarchical extension.

**Recommendation:** Inline bottom-up reconciliation (~20 lines). MinT optimal via hierarchicalforecast as optional upgrade.

---

## Enhancement 5.1A: Adaptive Frequency Selection

**Found in:** [`scikit-learn mutual_info_regression`](https://github.com/scikit-learn/scikit-learn) for information gain computation.

**Practical approach for our pipeline:**
```python
def should_run_frequency(freq, filing_frequency, available_data_points):
    # Rule-based selection (fast, no ML):
    # 1. Annual: always run (baseline)
    # 2. Quarterly: skip if filing_frequency == 'annual' AND < 4 filings
    # 3. Monthly: skip if < 12 data points at monthly
    # 4. Weekly: always run (price data always available)
    # 5. Daily: always run (native frequency)
    
    MIN_USEFUL_PERIODS = {"A": 3, "Q": 4, "M": 12, "W": 26, "D": 60}
    threshold = MIN_USEFUL_PERIODS.get(freq, 10)
    
    if available_data_points < threshold:
        return False, f"Only {available_data_points} periods (need {threshold})"
    
    # Skip semi-annual for quarterly filers (no additional information)
    if freq == 'S' and filing_frequency == 'quarterly':
        return False, "Semi-annual redundant for quarterly filer"
    
    return True, "sufficient data"
```

**No new dependency.** ~15 lines.

---

## Enhancement 5.2B: Model Complexity Matching

**Found in:** Academic literature on model selection with limited data (Burnham & Anderson 2002).

```python
# Frequency -> recommended model subset
FREQ_MODEL_MAP = {
    "A": ["baseline", "ema", "linear_trend"],      # 3-10 points: simple only
    "Q": ["baseline", "ema", "kalman", "ets"],      # 8-24 points: add Kalman, ETS
    "M": ["kalman", "ets", "var", "tree"],           # 24-60 points: add VAR, tree
    "W": ["kalman", "garch", "var", "lstm", "tree"], # 100+ points: add GARCH, LSTM
    "D": None,  # Full cascade (504+ points)
}
```

**No new dependency.** ~10 lines (config table).

---

## Enhancement 5.3D: Multi-Frequency Conformal Intervals

**Found in:** [`MAPIE`](https://github.com/scikit-learn-contrib/MAPIE) (1,300 stars) -- already used in Layer 3 for daily conformal.

**Implementation pattern:**
```python
def fuse_multifreq_intervals(per_freq_intervals: dict, conservatism: float = 0.5):
    # per_freq_intervals: {"D": (lo, hi), "W": (lo, hi), ...}
    all_lowers = [v[0] for v in per_freq_intervals.values() if v[0] is not None]
    all_uppers = [v[1] for v in per_freq_intervals.values() if v[1] is not None]
    
    if not all_lowers:
        return None, None
    
    # Blend: conservative uses union (widest), aggressive uses intersection (tightest)
    fused_lower = conservatism * min(all_lowers) + (1 - conservatism) * max(all_lowers)
    fused_upper = conservatism * max(all_uppers) + (1 - conservatism) * min(all_uppers)
    
    return fused_lower, fused_upper
```

**No new dependency.** ~15 lines. But requires conformal to run at each frequency (P4 due to computational cost).

---

## Enhancement 5.1B: Quality-Weighted Resampling

**No GitHub implementation.** Simple weighting by estimation confidence:

```python
def quality_weighted_resample(cache, confidence_cols):
    # Weight each period by average confidence across estimation columns
    conf_cols = [c for c in cache.columns if c.endswith('_confidence')]
    if conf_cols:
        quality = cache[conf_cols].mean(axis=1)
    else:
        quality = pd.Series(1.0, index=cache.index)
    return quality
```

**No new dependency.** ~10 lines.

---

## Summary

| Enhancement | Implementation | New Dependencies | Lines |
|-------------|---------------|-----------------|-------|
| 5.3C: Cross-freq momentum | Weighted sign agreement | None | ~25 |
| 5.3B: Weighted regime vote | Inverse-MAE * log(n) | None | ~20 |
| 5.2A: Uncertainty propagation | Context fields + prior blend | None | ~25 |
| 5.3A: MinT reconciliation | Inline bottom-up | None (optional: hierarchicalforecast) | ~20 |
| 5.1A: Frequency selection | Rule-based min periods | None | ~15 |
| 5.2B: Model complexity | Config table lookup | None | ~10 |
| 5.3D: Multi-freq conformal | Interval union/intersection | None | ~15 |
| 5.1B: Quality weighting | Mean confidence scoring | None | ~10 |
| **Total** | | **0 new dependencies** | **~140 lines** |

All 8 enhancements: **zero new dependencies**, ~140 total lines. All pure numpy/pandas computation on existing multi-frequency results.
