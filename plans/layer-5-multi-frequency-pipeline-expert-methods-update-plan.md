# Layer 5: Multi-Frequency Pipeline -- Expert Methods Update Plan

*Researched 2026-05-01 -- domain expert methods for enhancing the 3 Layer 5 multi-frequency modules*

Layer 5 runs the full analytical pipeline at 5 frequencies (A/Q/M/W/D) and fuses results. It already has a sophisticated 13-method fusion architecture. Enhancements focus on improving individual fusion methods, the per-frequency pipeline, and cross-frequency signal quality.

---

## Current State Summary

| Module | Lines | Current Methods | Enhancement Opportunity |
|--------|-------|----------------|------------------------|
| 5.1 Resampler | 704 | Rule-based OHLCV/stock/flow resampling | Adaptive frequency selection, quality-aware resampling |
| 5.2 Runner | 620 | Fixed A->Q->M->W->D cascade | Selective frequency skipping, parallel execution |
| 5.3 Fusion | 1,283 | 13 fusion methods (M1-M13) | Enhance M5 (MinT), M7 (copula), M13 (meta-learner); add information-theoretic fusion |

---

## Enhancement 5.1A: Adaptive Frequency Selection

**Current:** Always runs all 5 frequencies (A/Q/M/W/D).

**Expert method:** **Information-Theoretic Frequency Selection** -- before running each frequency, compute the expected information gain from that frequency given the data available. Skip frequencies that add negligible information (e.g., semi-annual for a quarterly filer has no new data vs quarterly).

**Formula:** `IG(freq) = H(prediction) - H(prediction | freq_data)` where H is entropy. Skip freq if `IG < threshold`.

**New result fields:** `skipped_frequencies` (list), `information_gain_per_freq` (dict)

### Enhancement 5.1B: Quality-Weighted Resampling

**Current:** Equal treatment of all resampled periods.

**Expert method:** Weight resampled periods by **data quality** (from estimation confidence). Periods with high interpolation (low filing_freshness) get lower weight in the per-frequency pipeline.

**New result fields:** `quality_weights_per_period` (per-frequency array)

---

## Enhancement 5.2A: Cascading Context Enhancement

**Current:** Context passes trend_direction, secular_regime, survival_probability, forecast_bounds.

**Expert method:** Add **prediction uncertainty propagation** -- each frequency passes not just point forecasts but uncertainty bands. The next faster frequency uses these as Bayesian priors: `posterior = prior * likelihood`.

**New context fields:** `forecast_uncertainty` (per-variable std from MC/conformal), `regime_posterior` (distribution over regimes, not just point label)

### Enhancement 5.2B: Frequency-Specific Model Selection

**Current:** Same model cascade (Kalman/GARCH/VAR/LSTM/Tree/ETS) runs at every frequency.

**Expert method:** Different frequencies favor different models. Annual data (8-10 points) works best with simple models (EMA, linear trend). Weekly data (150+ points) can support LSTM/Tree. Match model complexity to data availability via **Akaike Frequency Selection**: `optimal_model_complexity = f(n_periods)`.

**New result fields:** `model_selection_per_freq` (dict of freq -> recommended model subset)

---

## Enhancement 5.3A: Enhanced MinT Reconciliation (M5)

**Current:** M5 applies MinT with disagreement penalty.

**Expert method:** **Temporal Hierarchical Reconciliation** (Athanasopoulos et al. 2017) -- enforce that forecasts at different frequencies are mathematically consistent: monthly forecasts should sum to quarterly, quarterly to annual. Current M5 penalizes disagreement but doesn't enforce strict consistency.

**Formula:** `y_tilde = S @ (S.T @ W^{-1} @ S)^{-1} @ S.T @ W^{-1} @ y_hat` where S is the temporal summing matrix.

**Implementation:** `hierarchicalforecast` library (Nixtla, 790 stars) or inline ~30 lines.

### Enhancement 5.3B: Frequency Regime Voting with Confidence Weighting

**Current:** Regime consensus uses majority vote.

**Expert method:** Weight each frequency's regime vote by its **prediction accuracy** (walk_forward_mae) and **data quality** (n_periods). A quarterly frequency with MAE=2% should have more regime voting power than an annual frequency with MAE=5%.

**Formula:** `vote_weight_f = (1 / mae_f) * log(n_periods_f)` normalized across frequencies.

**New result fields:** `weighted_regime_consensus` (string), `regime_vote_weights` (dict)

### Enhancement 5.3C: Cross-Frequency Momentum Signal

**Current:** No explicit cross-frequency momentum comparison.

**Expert method:** **Temporal Momentum** (Moskowitz, Ooi & Pedersen 2012) -- when ALL frequencies agree on direction (annual trend up, quarterly trend up, monthly trend up, weekly trend up, daily trend up), this is a much stronger signal than any single frequency. Disagreement (annual up but daily down) = potential reversal.

**Formula:** `cross_freq_momentum = sum(sign(trend_f) * weight_f for f in frequencies)` where weight = 1/freq_rank.

**New result fields:** `cross_freq_momentum_score` (-1 to +1), `cross_freq_direction_agreement` (0-1), `potential_reversal_flag` (boolean when long-term and short-term disagree)

### Enhancement 5.3D: Frequency-Specific Conformal Intervals

**Current:** Conformal intervals computed only at daily frequency.

**Expert method:** Compute conformal intervals at EACH frequency, then fuse them. Lower-frequency intervals (quarterly, annual) are naturally wider but capture different risks. The fused interval = intersection for high-confidence, union for conservative.

**Formula:** `fused_lower = max(lower_D, lower_W, lower_M) * alpha + min(lower_D, lower_W, lower_M) * (1-alpha)` where alpha depends on desired conservatism.

**New result fields:** `per_freq_intervals` (dict of freq -> lower/upper), `fused_interval_method` (string)

---

## Implementation Priority Matrix

| Enhancement | Impact | Complexity | Dependencies | Priority |
|-------------|--------|------------|-------------|----------|
| 5.3C: Cross-frequency momentum | HIGH | LOW | Existing trend_direction | P1 |
| 5.3B: Weighted regime voting | MEDIUM | LOW | walk_forward_mae | P1 |
| 5.2A: Uncertainty propagation | HIGH | MEDIUM | MC/conformal results | P2 |
| 5.3A: MinT temporal reconciliation | MEDIUM | MEDIUM | hierarchicalforecast or inline | P2 |
| 5.1A: Adaptive frequency selection | MEDIUM | MEDIUM | Information gain computation | P3 |
| 5.2B: Model complexity matching | LOW | LOW | n_periods threshold table | P3 |
| 5.3D: Multi-frequency conformal | LOW | HIGH | Conformal at each frequency | P4 |
| 5.1B: Quality-weighted resampling | LOW | MEDIUM | Estimation confidence | P4 |

---

## Proposed New Variables (~8 result fields)

| # | Variable | Source | Type |
|---|----------|--------|------|
| 1 | `cross_freq_momentum_score` | 5.3 Fusion | Float (-1 to +1) |
| 2 | `cross_freq_direction_agreement` | 5.3 Fusion | Float (0-1) |
| 3 | `potential_reversal_flag` | 5.3 Fusion | Boolean |
| 4 | `weighted_regime_consensus` | 5.3 Fusion | String |
| 5 | `regime_vote_weights` | 5.3 Fusion | Dict |
| 6 | `forecast_uncertainty_propagated` | 5.2 Runner | Dict |
| 7 | `skipped_frequencies` | 5.1 Resampler | List |
| 8 | `information_gain_per_freq` | 5.1 Resampler | Dict |

All result objects -- 0 new cache columns. Consistent with Layer 5 architecture.

---

## GitHub Research Highlights

- **Temporal hierarchical reconciliation:** [`Nixtla/hierarchicalforecast`](https://github.com/Nixtla/hierarchicalforecast) (790 stars) -- MinT, BottomUp, TopDown methods
- **Cross-frequency momentum:** [`AQR capital management research`](https://github.com/topics/time-series-momentum) -- Moskowitz et al. 2012 implementation patterns
- **Information-theoretic selection:** [`scikit-learn mutual_info_regression`](https://github.com/scikit-learn/scikit-learn) for IG computation
- **Conformal at multiple frequencies:** [`MAPIE`](https://github.com/scikit-learn-contrib/MAPIE) already used in Layer 3

All P1+P2 enhancements require **zero new dependencies** (~80 lines total).

---

## Implementation Checklist

```
[ ] Phase 1: P1 Enhancements
    [ ] 5.3C: Cross-frequency momentum signal in frequency_fusion.py
    [ ] 5.3B: Weighted regime voting in frequency_fusion.py

[ ] Phase 2: P2 Enhancements
    [ ] 5.2A: Uncertainty propagation in multi_frequency_runner.py
    [ ] 5.3A: MinT temporal reconciliation in frequency_fusion.py

[ ] Phase 3: P3 Enhancements
    [ ] 5.1A: Adaptive frequency selection in frequency_resampler.py
    [ ] 5.2B: Model complexity matching in multi_frequency_runner.py

[ ] Phase 4: P4 (deferred)
    [ ] 5.3D: Multi-frequency conformal intervals
    [ ] 5.1B: Quality-weighted resampling
```
