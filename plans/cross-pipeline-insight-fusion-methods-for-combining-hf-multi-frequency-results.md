# Cross-Pipeline Insight Fusion

*How to meaningfully combine the 30 HF metrics + 5-frequency pipeline outputs into superior actionable intelligence.*

---

## The Problem

Right now the system has two powerful analytical tracks running in parallel:

1. **Main pipeline** (50 modules): Produces regime labels, survival probabilities, forecasts, MC simulations, conformal intervals -- all at 5 frequencies (D/W/M/Q/A)
2. **HF pipeline** (30 methods): Produces earnings quality, cash flow stress, balance sheet risk, inflection detection, valuation, position signal

But these two tracks barely talk to each other. The HF pipeline reads FROM the main pipeline (mc_result, forecast_result, etc.) but the insights don't flow BACK or get CROSS-VALIDATED. The multi-frequency results are available but only lightly consumed.

The question is: what are the proven methods for fusing these heterogeneous outputs into something greater than the sum of its parts?

---

## Category 1: Bayesian Belief Network Fusion

### What experts do

Professional quant teams (AQR, Two Sigma, Bridgewater) don't just average signals. They build directed acyclic graphs (DAGs) where each node is a signal and edges represent causal/informational relationships. The network propagates evidence: when one signal fires, it updates the posterior probability of related signals.

### How it applies here

```
Multi-Freq Regime Consensus (A/Q/M/W/D)
    |
    v
Macro Quadrant + Cross-Asset Regime
    |
    +---> Survival Probability (weighted by regime confidence)
    |         |
    |         v
    |     HF Leverage Stress (conditioned on macro regime)
    |         |
    |         v
    |     HF Earnings Torpedo (amplified in late-cycle regime)
    |
    +---> HF Momentum Composite (validated by freq consensus)
    |         |
    |         v
    |     HF Earnings Surprise (adjusted by momentum direction)
    |
    +---> Position Signal (Bayesian posterior from all ancestors)
```

**Method:** Each HF metric becomes a conditional probability: `P(metric | regime, frequency_consensus)`. When the quarterly frequency shows "bear" but the monthly shows "recovery," the Bayesian network propagates this disagreement as uncertainty, widening the position signal confidence interval.

**Implementation:** Use `pgmpy` (Python Bayesian network library) or a simpler hand-coded belief propagation. Each node has a conditional probability table (CPT) learned from historical data.

**Complexity:** HIGH (needs historical labeled data to train CPTs)
**Impact:** VERY HIGH (this is how the best multi-strategy funds work)

---

## Category 2: Hierarchical Signal Aggregation (Slow-to-Fast Constraint)

### What experts do

The "Cascading Context" pattern from the multi-frequency runner (annual -> quarterly -> monthly -> weekly -> daily) is the right idea, but it's currently one-directional. Expert implementations use bidirectional constraint propagation:

- **Top-down:** Slower frequencies constrain faster frequencies (annual trend bounds quarterly forecasts)
- **Bottom-up:** Faster frequencies provide early warning to slower frequencies (daily regime break updates quarterly survival assessment)

### How it applies here

**Top-down (already partially implemented):**
- Annual secular trend -> bounds DCF terminal growth rate (HF-5.1)
- Quarterly regime -> constrains weekly forecast range
- Monthly macro quadrant -> gates HF momentum interpretation

**Bottom-up (NOT implemented -- this is the gap):**
- Daily pattern detection (torpedo, vol inversion) -> should OVERRIDE weekly/monthly signal if severity is extreme
- Weekly PEAD signal -> should accelerate quarterly earnings surprise update
- Daily OU mean reversion break -> should trigger quarterly leverage stress re-evaluation

**The key insight:** Most quant systems are purely top-down. The ones that also flow bottom-up (early warning propagation) catch inflection points 2-4 weeks earlier.

**Implementation:** Add an "early warning propagation" step after the HF analysis that checks:
1. If any daily/weekly signal exceeds a severity threshold
2. If so, re-evaluate the affected quarterly/annual metric with updated priors

**Complexity:** MEDIUM
**Impact:** HIGH

---

## Category 3: Regime-Conditioned Signal Weighting

### What experts do

Signal predictive power varies dramatically across market regimes. A value signal (low PE) works in recovery but fails in crisis. A momentum signal works in trends but fails in mean-reverting regimes. The Hurst exponent and HMM regime labels should modulate which signals get weight.

### How it applies here

Each of the 30 HF metrics should have a **per-regime IC table**:

```
                     Bull    Bear    High_Vol  Low_Vol  Survival
FCF Quality          0.08    0.12    0.06      0.09     0.15
Accruals Forensic    0.05    0.09    0.04      0.06     0.11
Momentum             0.12    0.03    0.02      0.10     0.01
Piotroski            0.06    0.10    0.08      0.07     0.13
OU Reversion         0.10    0.04    0.02      0.11     0.01
Earnings Torpedo     0.02    0.15    0.12      0.03     0.08
```

The scorecard and position signal should use the IC from the CURRENT regime, not a static weight.

**Implementation:** During the walk-forward evaluation, track IC of each HF metric against forward returns, partitioned by regime. Build the IC table from historical data. Use the appropriate column at runtime.

**Complexity:** MEDIUM (needs walk-forward integration)
**Impact:** VERY HIGH (the single most impactful fusion technique)

---

## Category 4: Cross-Frequency Disagreement as a Signal

### What experts do

When multiple timeframes disagree, that disagreement itself is informative. A company where the annual trend is bullish but the weekly trend is bearish is in a potential inflection point. The degree of cross-frequency agreement/disagreement is a meta-signal.

### How it applies here

The multi-frequency fusion already computes `regime_consensus.agreement_ratio`. But this isn't used by the HF pipeline.

**Enhanced disagreement signals:**

1. **Temporal divergence score:** Count how many frequencies disagree on regime. 0 disagreements = high conviction, 3+ = inflection zone.

2. **Quality-price divergence:** HF earnings quality score (quarterly-native) vs daily price momentum. High quality + falling price = buy. Low quality + rising price = sell. This is the classic "quality-value" combination that outperforms either signal alone.

3. **Survival consensus:** If the quarterly survival probability is 0.95 (safe) but the weekly survival probability is 0.65 (stressed), the weekly signal is detecting something the quarterly misses. Take the MINIMUM (weakest-link principle from harmonic mean).

4. **Filing-freshness weighted disagreement:** When the most recent filing is stale (>60 days), give MORE weight to weekly/daily signals (they reflect current market pricing) and LESS to quarterly signals (based on old data).

**Implementation:** A `compute_cross_frequency_disagreement()` function that takes all frequency results + HF results and produces a set of meta-signals.

**Complexity:** LOW
**Impact:** HIGH

---

## Category 5: Ensemble of Ensembles (Meta-Learning)

### What experts do

The main pipeline has an ensemble (prediction_aggregator with inverse-RMSE + FixedShare + MCS). The HF pipeline has an ensemble (thesis scorecard with weighted tiers). Running a meta-ensemble on TOP of both produces better results than either alone.

### How it applies here

**Level 1 ensembles (already exist):**
- Main pipeline: Kalman/GARCH/VAR/LSTM/XGB/Baseline -> prediction_aggregator
- HF pipeline: 15 metrics -> thesis scorecard

**Level 2 meta-ensemble (new):**
- Take the main pipeline position_signal and the HF position_signal
- Weight them by their respective IC (from signal_ic.py)
- When they agree (both buy or both sell), increase conviction
- When they disagree (main says buy, HF says sell), reduce conviction and flag as "conflicted"

**The stacking approach** (Wolpert 1992): Train a simple logistic regression on historical outputs of both Level 1 ensembles to predict forward returns. The stacker learns which ensemble to trust in which conditions.

**Implementation:** Add a `meta_ensemble.py` that combines main pipeline position_signal with HF position_signal using IC-weighted averaging + disagreement detection.

**Complexity:** LOW
**Impact:** MEDIUM-HIGH

---

## Category 6: Temporal Cascade for Catalyst-Driven Events

### What experts do

Event-driven funds don't use signals the same way every day. They identify CATALYSTS (earnings, FDA approval, merger, regime change) and position BEFORE the catalyst. After the catalyst fires, the signal regime changes completely.

### How it applies here

We already have:
- `filing_calendar_result.next_expected_filing` (date of next earnings)
- `product_catalysts.catalyst_score` (forward-looking catalyst signals)
- `regime_shift_predictor.expected_days_to_shift` (HMM-predicted regime change)

The fusion should create a **catalyst calendar** that maps upcoming events to expected signal regime changes:

```
Days to catalyst: 30+   -> Weight: slow fundamentals (HF Tier 1-3)
Days to catalyst: 10-30  -> Weight: momentum + pre-earnings drift (HF Tier 4)
Days to catalyst: 0-10   -> Weight: vol regime + positioning (HF Tier 5)
Days after catalyst: 0-5  -> Weight: PEAD signal + surprise reaction
Days after catalyst: 5+   -> Return to normal signal weights
```

This is the "time-to-event" modulation that converts the static signal weights into dynamic, event-aware weights.

**Implementation:** A `catalyst_aware_weighting()` function that adjusts HF tier weights based on proximity to the next filing/catalyst event.

**Complexity:** MEDIUM
**Impact:** HIGH (this is the edge for timing entries/exits)

---

## Category 7: Anomaly-Driven Priority Routing

### What experts do

When everything looks normal, the standard signal pipeline is fine. But when an anomaly is detected at ANY frequency, the system should switch to anomaly-investigation mode:

1. **Matrix Profile discord** (from pattern_detector) fires -> prioritize technical signals
2. **ChangeFinder online score** spikes -> re-run regime detection immediately
3. **Forensic CF flag** detected -> freeze the position signal until the flag is resolved
4. **Earnings torpedo** fires -> override momentum with caution

### How it applies here

Add an "anomaly priority router" that scans all outputs for high-severity signals:

```python
ANOMALY_OVERRIDES = {
    "forensic_cf_flag": {"action": "freeze_position", "severity": "high"},
    "earnings_torpedo_risk > 75": {"action": "cap_signal_at_hold", "severity": "high"},
    "vol_term_structure_inverted": {"action": "widen_stops_2x", "severity": "medium"},
    "matrix_profile_discord": {"action": "flag_for_review", "severity": "low"},
    "covenant_breach_any_scenario": {"action": "set_signal_sell", "severity": "critical"},
}
```

When a critical anomaly fires, it OVERRIDES the normal signal computation. This prevents the system from recommending "buy" when a covenant breach is imminent.

**Complexity:** LOW
**Impact:** VERY HIGH (prevents catastrophic recommendations)

---

## Category 8: Multi-Frequency IC Attribution

### What experts do

For each HF metric, measure IC separately at each frequency's native horizon:

```
FCF Quality:
  Annual IC (1Y horizon):  0.12  -- strongest (quarterly data, annual outcome)
  Quarterly IC (3M horizon): 0.09
  Monthly IC (1M horizon):  0.04
  Weekly IC (1W horizon):   0.01  -- noise at this timescale
  Daily IC (1D horizon):    0.00  -- meaningless

Momentum:
  Annual IC:  0.03  -- momentum reverts at long horizons
  Quarterly IC: 0.08
  Monthly IC:  0.11  -- strongest (momentum is a medium-term signal)
  Weekly IC:   0.09
  Daily IC:    0.05
```

This IC attribution tells you WHICH frequency provides the most information for EACH metric. The fusion then routes each metric to its optimal frequency.

**Implementation:** Extend `signal_ic.py` to compute IC at multiple horizons (return_5d, return_21d, return_63d, return_252d) for each HF metric. Use the per-horizon IC to set the natural frequency weight in `hf_frequency_fusion.py`.

**Complexity:** MEDIUM
**Impact:** HIGH (replaces hardcoded frequency assignments with data-driven ones)

---

## Recommended Implementation Priority

| # | Method | Complexity | Impact | Dependencies |
|---|--------|-----------|--------|-------------|
| 1 | **Anomaly-Driven Priority Routing** | LOW | VERY HIGH | HF results + pattern_detector |
| 2 | **Cross-Frequency Disagreement Signals** | LOW | HIGH | Multi-freq results + HF results |
| 3 | **Regime-Conditioned Signal Weighting** | MEDIUM | VERY HIGH | Walk-forward IC per regime |
| 4 | **Meta-Ensemble (Stacking)** | LOW | MEDIUM-HIGH | Both position signals + IC |
| 5 | **Catalyst-Driven Temporal Weighting** | MEDIUM | HIGH | Filing calendar + catalyst detector |
| 6 | **Bottom-Up Early Warning Propagation** | MEDIUM | HIGH | Daily signals -> quarterly re-eval |
| 7 | **Multi-Frequency IC Attribution** | MEDIUM | HIGH | signal_ic at multiple horizons |
| 8 | **Bayesian Belief Network** | HIGH | VERY HIGH | Historical labeled data + pgmpy |

Methods 1-4 can be implemented with ~400 lines total and have the highest ROI. Methods 5-7 add ~300 more lines. Method 8 is a research project.

---

## The Master Fusion Flow

```
Main Pipeline (50 modules)          HF Pipeline (30 methods)
        |                                    |
        v                                    v
Multi-Freq Results (5 freqs)      Thesis Scorecard + Position
        |                                    |
        +----------+            +------------+
                   |            |
                   v            v
         Cross-Frequency Disagreement Signals
                        |
                        v
              Regime-Conditioned Weighting
                        |
                        v
              Anomaly Priority Router
                        |
                        v
              Catalyst-Aware Temporal Modulation
                        |
                        v
              Meta-Ensemble: Stacked Position Signal
                        |
                        v
              FINAL OUTPUT: Action + Conviction + Timing
```

This transforms the system from "here are 80 numbers" to "buy AAPL at $138 with 7/10 conviction, stop at $125, target $155, next catalyst in 6 weeks, primary risk is Q3 debt refinancing."
