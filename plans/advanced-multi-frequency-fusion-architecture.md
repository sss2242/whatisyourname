# Advanced Multi-Frequency Fusion Architecture v2

*Refined with community implementations from Nixtla, M2FMoE, WPMixer, Merlion, Darts, Orbit*

Upgrade plan for `operator1/models/frequency_fusion.py`. The HF fusion layer (`hedge_fund/fusion.py`) remains unchanged.

---

## Current State vs Target

| Aspect | Current (499 lines, 3 methods) | Target (~1,400 lines, 13 methods) |
|--------|-------------------------------|----------------------------------|
| Regime consensus | Majority vote | Wavelet coherence + Granger cascade + gating |
| Prediction fusion | Static horizon weight table | MinT reconciliation + orthogonal residuals + meta-learner |
| Survival fusion | Harmonic mean | Copula joint uncertainty + sigmoid gating |
| Confidence | `n_freqs / 3` | AIC selection + cointegration + break alignment |
| Disagreement | Text description | Signal curve shape classification + diversity scoring |
| New | N/A | Orthogonal decomposition, frequency band gating, expert alignment |

---

## 13-Method Architecture

### Layer A: Frequency-Specific Insight Extraction

#### Method 1: Structural Break Alignment (Bai-Perron 2003)

Cross-frequency break confirmation. Each frequency detects breaks independently via PELT (already computed per-frequency in stage 2a1). This method aligns break dates across frequencies.

```python
def align_structural_breaks(frequency_results):
    # Collect breaks from each frequency
    breaks = {}
    for freq, result in frequency_results.items():
        if hasattr(result, 'structural_breaks'):
            breaks[freq] = result.structural_breaks  # list of dates

    confirmed = []
    for freq, break_dates in breaks.items():
        for bd in break_dates:
            # Check if any other frequency has a break within tolerance
            tolerance = _FREQ_TOLERANCE[freq]  # A=90d, Q=30d, M=10d, W=3d, D=1d
            n_confirming = sum(
                1 for f2, bd2_list in breaks.items()
                if f2 != freq
                for bd2 in bd2_list
                if abs((bd - bd2).days) <= tolerance
            )
            confirmed.append(BreakConfirmation(
                date=bd, source_freq=freq,
                n_confirming=n_confirming,
                confidence='high' if n_confirming >= 2 else 'medium' if n_confirming == 1 else 'low',
                is_structural=n_confirming >= 2,
            ))
    return BreakAlignment(confirmed=confirmed)
```

---

#### Method 2: Orthogonal Wavelet Decomposition (adapted from WPMixer)

**Source:** WPMixer `wavelet_patch_mixer.py` -- wavelet packet decomposition creates orthogonal frequency bands, guaranteeing non-redundant information per resolution.

Instead of naively averaging correlated forecasts (Daily and Weekly may be >90% correlated for trending stocks), decompose each frequency's forecast into orthogonal wavelet bands. Only fuse the non-overlapping component from each frequency.

```python
def orthogonal_decomposition(frequency_forecasts):
    import pywt

    # Each frequency owns its natural wavelet band
    # D -> detail coefficients level 0 (highest frequency)
    # W -> detail coefficients level 1
    # M -> detail coefficients level 2
    # Q -> detail coefficients level 3
    # A -> approximation coefficients (lowest frequency)

    for freq, forecast_series in frequency_forecasts.items():
        # DWT decomposition
        coeffs = pywt.wavedec(forecast_series, 'db4', level=4)
        # Zero out bands NOT owned by this frequency
        for level in range(len(coeffs)):
            if level != _FREQ_TO_LEVEL[freq]:
                coeffs[level] = np.zeros_like(coeffs[level])
        # Reconstruct: only the owned band survives
        orthogonal_component[freq] = pywt.waverec(coeffs, 'db4')

    # Fused = sum of orthogonal components (guaranteed non-redundant)
    fused = sum(orthogonal_component.values())
    return fused, orthogonal_component
```

**Key insight from WPMixer:** The wavelet packet basis is orthogonal, so the fused signal has no double-counting. This replaces the horizon weight table for variables where all 5 frequencies produce forecasts.

---

#### Method 3: Granger Cascade Direction (Breitung & Candelon 2006)

Tests whether slow frequencies lead fast or vice versa. When market prices (Daily) Granger-cause fundamentals (Quarterly), the market "knows something" -- amplify Daily weight. When fundamentals lead prices (normal), amplify Quarterly/Annual weight.

```python
def compute_cascade_direction(frequency_results):
    from statsmodels.tsa.stattools import grangercausalitytests

    directions = {}
    freq_order = ['A', 'Q', 'M', 'W', 'D']

    for i, slow in enumerate(freq_order[:-1]):
        fast = freq_order[i + 1]
        slow_series = frequency_results[slow].forecast_path
        fast_series = frequency_results[fast].forecast_path

        # Align to slower frequency's index
        aligned = pd.DataFrame({'slow': slow_series, 'fast': fast_series}).dropna()
        if len(aligned) < 10:
            continue

        # Test both directions
        try:
            slow_to_fast = grangercausalitytests(aligned[['fast', 'slow']], maxlag=3)
            fast_to_slow = grangercausalitytests(aligned[['slow', 'fast']], maxlag=3)

            p_s2f = min(slow_to_fast[lag][0]['ssr_ftest'][1] for lag in slow_to_fast)
            p_f2s = min(fast_to_slow[lag][0]['ssr_ftest'][1] for lag in fast_to_slow)

            if p_s2f < 0.05 and p_f2s >= 0.05:
                directions[(slow, fast)] = 'fundamental_lead'  # Expected
            elif p_f2s < 0.05 and p_s2f >= 0.05:
                directions[(slow, fast)] = 'market_lead'  # Market knows something
            elif p_s2f < 0.05 and p_f2s < 0.05:
                directions[(slow, fast)] = 'bidirectional'  # Feedback loop
            else:
                directions[(slow, fast)] = 'independent'
        except Exception:
            directions[(slow, fast)] = 'unknown'

    return CascadeDirection(directions=directions)
```

---

#### Method 4: Spectral Gating (adapted from M2FMoE FreqMoE)

**Source:** M2FMoE `FreqMoE` class (lines 28-104) -- learns to gate frequency band contributions based on spectral magnitude.

Adapted for our non-neural context: instead of learnable gating weights, use the spectral power density of each frequency's forecast error to determine how much weight that frequency should receive. Frequencies with large forecast errors in a particular spectral band get less weight for that band.

```python
def spectral_gating(frequency_forecasts, frequency_errors):
    # Compute FFT power spectrum of each frequency's forecast errors
    spectral_weights = {}
    for freq, errors in frequency_errors.items():
        fft_errors = np.fft.rfft(errors)
        power = np.abs(fft_errors) ** 2
        # Inverse power = weight (low error = high weight)
        spectral_weights[freq] = 1.0 / (power + 1e-8)

    # Normalize per spectral band
    for band_idx in range(len(next(iter(spectral_weights.values())))):
        total = sum(sw[band_idx] for sw in spectral_weights.values())
        for freq in spectral_weights:
            spectral_weights[freq][band_idx] /= total

    # Apply spectral gating: each frequency contributes to bands where it has low error
    fused_spectrum = np.zeros_like(next(iter(frequency_forecasts.values())))
    for freq, forecast in frequency_forecasts.items():
        fft_forecast = np.fft.rfft(forecast)
        gated = fft_forecast * spectral_weights[freq]
        fused_spectrum += gated

    return np.fft.irfft(fused_spectrum), spectral_weights
```

---

### Layer B: Cross-Frequency Reconciliation

#### Method 5: MinT Hierarchical Reconciliation (from Nixtla/hierarchicalforecast)

**Source:** Nixtla `hierarchicalforecast` methods.py -- production-grade MinT implementation with WLS, OLS, and Structural scaling.

Our temporal hierarchy: `A = sum(4Q)`, `Q = sum(3M)`, `M = sum(~4.3W)`, `W = sum(5D)`.

```python
def mint_reconciliation(frequency_forecasts, frequency_errors):
    # Build summing matrix S for temporal hierarchy
    # S maps bottom level (Daily) to all levels
    n_bottom = 252  # trading days in a year
    n_levels = 5    # A, Q, M, W, D

    # S matrix: each row is a constraint
    # Row 0: Annual = sum of all 252 daily values
    # Rows 1-4: Q1 = sum of D1..D63, Q2 = sum of D64..D126, etc.
    # ...
    S = build_temporal_summing_matrix(n_bottom, hierarchy='temporal_AQMWD')

    # W = covariance of base (daily) forecast errors
    # WLS variant: W = diag(var(errors_daily))
    W_diag = np.var(frequency_errors.get('D', np.zeros(n_bottom)))
    W = np.eye(n_bottom) * max(W_diag, 1e-8)

    # MinT reconciliation: y_tilde = S @ inv(S.T @ inv(W) @ S) @ S.T @ inv(W) @ y_hat
    W_inv = np.linalg.inv(W)
    G = np.linalg.inv(S.T @ W_inv @ S) @ S.T @ W_inv
    y_hat = stack_all_level_forecasts(frequency_forecasts)
    y_reconciled = S @ G @ y_hat

    return unstack_reconciled(y_reconciled, hierarchy='temporal_AQMWD')
```

**Nixtla improvement:** Use their Structural scaling variant which estimates W from the structure of S, avoiding the need for a full error covariance matrix (which we may not have enough data to estimate reliably).

---

#### Method 6: Resolution Accumulation Gating (from M2FMoE + Orbit)

**Source:** M2FMoE `ResolutionLinearAccumulateFusion` (lines 185-205) for the accumulation pattern. Orbit KTR for kernel-based time-varying weights.

Coarse-to-fine additive fusion where each resolution contributes through a projection that captures its unique information.

```python
def resolution_gating(frequency_results, regime_detector):
    freq_order = ['A', 'Q', 'M', 'W', 'D']  # coarse to fine

    # Walk-forward-learned per-regime weights
    # (from Kats/Facebook backtesting + selection pattern)
    regime_weights = {}
    for freq, result in frequency_results.items():
        if hasattr(result, 'walk_forward_mae_by_regime'):
            for regime, mae in result.walk_forward_mae_by_regime.items():
                regime_weights.setdefault(regime, {})[freq] = 1.0 / max(mae, 1e-8)

    # Normalize per regime
    for regime in regime_weights:
        total = sum(regime_weights[regime].values())
        regime_weights[regime] = {f: w/total for f, w in regime_weights[regime].items()}

    current_regime = regime_detector.current_regime if regime_detector else 'normal'
    weights = regime_weights.get(current_regime, {f: 0.2 for f in freq_order})

    # M2FMoE accumulation: coarse to fine, additive
    accumulated = None
    for freq in freq_order:
        if freq not in frequency_results:
            continue
        contribution = frequency_results[freq].point_forecast * weights.get(freq, 0.1)
        if accumulated is None:
            accumulated = contribution
        else:
            # GatingUnit sigmoid handoff (from M2FMoE line 236)
            gate = 1.0 / (1.0 + np.exp(-2.94))  # learned bias from M2FMoE
            accumulated = gate * contribution + (1 - gate) * accumulated

    return accumulated, weights
```

---

#### Method 7: Copula Joint Uncertainty (Patton 2012)

Cross-frequency error correlation for proper prediction intervals.

```python
def copula_joint_uncertainty(frequency_errors):
    from copulae import StudentCopula, GaussianCopula

    # Stack errors from all available frequencies
    error_matrix = np.column_stack([
        frequency_errors[f] for f in sorted(frequency_errors.keys())
        if len(frequency_errors[f]) > 20
    ])

    if error_matrix.shape[1] < 2:
        return None  # not enough frequencies

    # Fit Student-t copula (captures tail dependence)
    try:
        cop = StudentCopula(dim=error_matrix.shape[1])
        cop.fit(error_matrix)
        tail_dep = cop.dtau  # lower tail dependence

        # Simulate joint errors
        simulated = cop.random(1000)
        # Joint P5/P95 from copula
        joint_p5 = np.percentile(simulated.sum(axis=1), 5)
        joint_p95 = np.percentile(simulated.sum(axis=1), 95)

        return CopulaUncertainty(
            tail_dependence=float(tail_dep),
            joint_p5=float(joint_p5),
            joint_p95=float(joint_p95),
            copula_type='student_t',
        )
    except Exception:
        # Fallback to Gaussian
        cop = GaussianCopula(dim=error_matrix.shape[1])
        cop.fit(error_matrix)
        return CopulaUncertainty(copula_type='gaussian', tail_dependence=0.0)
```

---

#### Method 8: Disagreement Signal Curve (Vanna-Volga + M2FMoE ExpertAlignmentLoss)

**Source:** M2FMoE `ExpertAlignmentLoss` (lines 273-335) -- diversity scoring that penalizes expert collapse.

```python
def compute_disagreement_signal(frequency_signals):
    # frequency_signals: {freq: float} -- position signal per frequency (-1 to +1)
    signals = np.array([frequency_signals[f] for f in ['A', 'Q', 'M', 'W', 'D']
                        if f in frequency_signals])
    freq_ranks = np.arange(len(signals))  # 0=slowest, N=fastest

    if len(signals) < 2:
        return DisagreementSignal(shape='insufficient', score=0.0)

    # Disagreement score (from M2FMoE diversity: norm_std)
    score = float(np.std(signals))

    # Direction: correlation of signal with frequency rank
    if np.std(signals) > 0.01:
        direction = float(np.corrcoef(signals, freq_ranks)[0, 1])
    else:
        direction = 0.0

    # Shape classification
    if score < 0.1:
        if np.mean(signals) > 0.2:
            shape = 'monotonic_bullish'
        elif np.mean(signals) < -0.2:
            shape = 'monotonic_bearish'
        else:
            shape = 'flat_neutral'
    elif direction > 0.5:
        shape = 'smirk_momentum'  # faster freqs more bullish
    elif direction < -0.5:
        shape = 'reverse_smirk_reversion'  # faster freqs more bearish
    elif signals[0] * signals[-1] > 0 and np.min(np.abs(signals[1:-1])) < 0.1:
        shape = 'smile_transition'  # extremes agree, middle uncertain
    else:
        shape = 'mixed'

    # M2FMoE cosine diversity: penalize low diversity
    if len(signals) >= 3:
        cosine_div = 1.0 - np.mean([
            np.dot(signals, np.roll(signals, k)) / (np.linalg.norm(signals) ** 2 + 1e-8)
            for k in range(1, len(signals))
        ])
    else:
        cosine_div = score

    return DisagreementSignal(
        shape=shape, score=score, direction=direction,
        cosine_diversity=cosine_div,
        confidence_adjustment=1.0 - 0.5 * score,  # high disagreement = low confidence
    )
```

---

### Layer C: Unique Insight Preservation

#### Method 9: Frequency Anomaly Injection with Sigmoid Gating (from M2FMoE GatingUnit)

**Source:** M2FMoE `GatingUnit` (lines 230-240) -- sigmoid handoff between new signal and prior.

```python
def inject_frequency_anomalies(fused_forecast, frequency_forecasts, cross_freq_mean):
    anomalies = []
    freq_order = ['A', 'Q', 'M', 'W', 'D']

    for freq, forecast in frequency_forecasts.items():
        deviation = abs(forecast - cross_freq_mean)
        threshold = 2.0 * np.std([f for f in frequency_forecasts.values()])

        if deviation > threshold:
            freq_rank = freq_order.index(freq) if freq in freq_order else 2
            is_slow = freq_rank <= 2  # A, Q, M are "slow"

            if is_slow:
                # Structural anomaly: slow frequency sees something others don't
                # GatingUnit sigmoid: bias toward the anomaly (init_gate_bias=2.94 from M2FMoE)
                gate = 1.0 / (1.0 + np.exp(-2.94 * (deviation / threshold - 1.0)))
                adjusted = gate * forecast + (1 - gate) * fused_forecast
                anomalies.append(FrequencyAnomaly(
                    freq=freq, type='structural', gate_value=gate,
                    original_fused=fused_forecast, adjusted_fused=adjusted,
                    interpretation=f'{freq} frequency detects structural shift not visible at faster timescales',
                ))
                fused_forecast = adjusted
            else:
                # Fast frequency anomaly: flag but don't adjust
                anomalies.append(FrequencyAnomaly(
                    freq=freq, type='noise_candidate', gate_value=0.0,
                    interpretation=f'{freq} frequency shows anomalous signal, possible noise',
                ))

    return fused_forecast, anomalies
```

---

#### Method 10: Temporal Horizon Ownership with Resolution Projection

**Source:** M2FMoE `ResolutionLinearAccumulateFusion` -- per-resolution linear projection + accumulation.

Each horizon is "owned" by its natural frequency. Other frequencies provide additive residual adjustments (from Method 2 orthogonal decomposition), weighted by their information ratio at that horizon.

```python
def horizon_ownership_fusion(frequency_results, residual_decomposition):
    ownership = {
        '1d': 'D', '5d': 'W', '21d': 'M', '63d': 'Q', '252d': 'A'
    }

    fused = {}
    for horizon, owner_freq in ownership.items():
        if owner_freq not in frequency_results:
            continue

        # Owner provides the base forecast (100% weight)
        base = frequency_results[owner_freq].forecasts.get(horizon)
        if base is None:
            continue

        # Other frequencies provide residual adjustments
        adjustment = 0.0
        for freq, result in frequency_results.items():
            if freq == owner_freq:
                continue
            residual = residual_decomposition.get(freq, {}).get(horizon, 0.0)
            info_ratio = residual_decomposition.get(freq, {}).get('info_ratio', 0.0)
            # Only add adjustment if this frequency has useful residual info
            if abs(info_ratio) > 0.1:
                adjustment += residual * min(info_ratio, 1.0) * 0.2  # cap at 20% adjustment

        fused[horizon] = FusedPrediction(
            variable='close', horizon=horizon,
            point_forecast=base + adjustment,
            owner_freq=owner_freq,
            adjustment_from_others=adjustment,
        )

    return fused
```

---

### Layer D: Meta-Level Quality

#### Method 11: AIC Frequency Selection (from Merlion ModelSelector)

**Source:** Merlion `ModelSelector` class -- selects best model via validation metric.

```python
def select_informative_frequencies(frequency_results, frequency_errors):
    # Compute AIC-like score for including each frequency
    # AIC = 2k - 2ln(L) where k = parameters, L = likelihood
    # For our purposes: k = 1 (one frequency), L approximated from MSE

    n = len(next(iter(frequency_errors.values())))
    included = {}
    excluded = {}

    for freq, errors in frequency_errors.items():
        mse = np.mean(errors ** 2)
        aic_with = 2 * 1 + n * np.log(max(mse, 1e-12))

        # AIC without = AIC of the mean of all OTHER frequencies
        other_mean = np.mean([
            frequency_results[f].point_forecast
            for f in frequency_results if f != freq
        ])
        mse_without = np.mean((other_mean - actuals) ** 2) if actuals is not None else mse * 1.1
        aic_without = n * np.log(max(mse_without, 1e-12))

        delta = aic_with - aic_without
        if delta < -2:
            included[freq] = {'delta_aic': delta, 'reason': 'improves_fusion'}
        elif delta > 2:
            excluded[freq] = {'delta_aic': delta, 'reason': 'hurts_fusion'}
        else:
            included[freq] = {'delta_aic': delta, 'reason': 'marginal'}

    return FrequencySelection(included=included, excluded=excluded)
```

---

#### Method 12: Cointegration Long-Run Anchor (Johansen 1991)

```python
def test_frequency_cointegration(frequency_forecast_paths):
    from statsmodels.tsa.vector_ar.vecm import coint_johansen

    # Stack forecast paths from frequencies that have enough data
    paths = {}
    for freq, result in frequency_forecast_paths.items():
        if hasattr(result, 'forecast_path') and len(result.forecast_path) >= 10:
            paths[freq] = result.forecast_path

    if len(paths) < 2:
        return CointegrationResult(available=False)

    # Align to common index
    df = pd.DataFrame(paths).dropna()
    if len(df) < 15:
        return CointegrationResult(available=False)

    try:
        result = coint_johansen(df.values, det_order=0, k_ar_diff=1)
        # Trace statistic test
        rank = sum(result.lr1 > result.cvt[:, 1])  # 5% critical values

        if rank > 0:
            # Cointegrated: extract error correction term
            beta = result.evec[:, :rank]  # cointegrating vectors
            ect = df.values @ beta  # error correction terms
            latest_ect = float(ect[-1, 0])

            return CointegrationResult(
                available=True, cointegrated=True,
                rank=rank, error_correction_term=latest_ect,
                interpretation=(
                    'Frequencies share long-run equilibrium. '
                    f'ECT={latest_ect:.4f} -- {"diverged, expect reversion" if abs(latest_ect) > 0.1 else "near equilibrium"}.'
                ),
            )
        else:
            return CointegrationResult(
                available=True, cointegrated=False,
                interpretation='Frequencies are structurally independent -- no shared equilibrium.',
            )
    except Exception as exc:
        return CointegrationResult(available=False, error=str(exc))
```

---

#### Method 13: Meta-Learner Stacking (from Darts EnsembleModel)

**Source:** Darts `EnsembleModel` -- trains a regression model on stacked per-frequency outputs.

```python
def train_meta_learner(frequency_walk_forward_results):
    from sklearn.linear_model import Ridge

    # Build training data: X = per-frequency forecasts, y = actuals
    X_rows = []
    y_rows = []

    for day_idx in range(len(actuals)):
        row = []
        for freq in ['A', 'Q', 'M', 'W', 'D']:
            forecast = frequency_walk_forward_results.get(freq, {}).get(day_idx)
            row.append(forecast if forecast is not None else np.nan)
        if not any(np.isnan(row)):
            X_rows.append(row)
            y_rows.append(actuals[day_idx])

    if len(X_rows) < 30:
        return None  # insufficient data

    X = np.array(X_rows)
    y = np.array(y_rows)

    # Ridge regression (L2 regularization prevents overfitting to few frequencies)
    model = Ridge(alpha=1.0)
    model.fit(X, y)

    # The coefficients ARE the learned combination weights
    weights = dict(zip(['A', 'Q', 'M', 'W', 'D'], model.coef_))
    intercept = model.intercept_

    return MetaLearner(
        weights=weights, intercept=intercept,
        r2_score=model.score(X, y),
        interpretation=f'Learned weights: {", ".join(f"{k}={v:.3f}" for k,v in sorted(weights.items(), key=lambda x: -abs(x[1])))}',
    )
```

---

## Execution Order

```
Input: 5 FrequencyResult objects from per-frequency pipelines

LAYER D (first -- decides which frequencies to include):
  M11: AIC Frequency Selection
       -> prune uninformative frequencies before fusion

LAYER A (extract unique insights from included frequencies):
  M1:  Structural Break Alignment
  M2:  Orthogonal Wavelet Decomposition (WPMixer-inspired)
  M3:  Granger Cascade Direction
  M4:  Spectral Gating (M2FMoE FreqMoE-inspired)

LAYER B (reconcile and combine):
  M5:  MinT Hierarchical Reconciliation (Nixtla-inspired)
  M6:  Resolution Accumulation Gating (M2FMoE + Orbit-inspired)
  M7:  Copula Joint Uncertainty
  M8:  Disagreement Signal Curve (M2FMoE ExpertAlignmentLoss-inspired)
  M13: Meta-Learner Stacking (Darts-inspired)

LAYER C (preserve unique insights post-fusion):
  M9:  Frequency Anomaly Injection (M2FMoE GatingUnit-inspired)
  M10: Temporal Horizon Ownership with Resolution Projection
  M12: Cointegration Long-Run Anchor

Output: Enhanced FrequencyFusionResult
  -> feeds into HF fusion.run_fusion() as multi_frequency_result
```

---

## Community Code References

| Method | Community Source | Stars | What We Adapt |
|--------|----------------|-------|--------------|
| M2 | WPMixer `wavelet_patch_mixer.py` | 82 | Orthogonal wavelet band ownership per resolution |
| M4 | M2FMoE `FreqMoE` class | 5 | Spectral magnitude gating for frequency band weights |
| M5 | Nixtla `hierarchicalforecast` | 745 | MinT matrix reconciliation with WLS error covariance |
| M6 | M2FMoE `ResolutionLinearAccumulateFusion` | 5 | Coarse-to-fine additive accumulation pattern |
| M6 | Orbit KTR model | 2,047 | Kernel-based time-varying regression for regime weights |
| M8 | M2FMoE `ExpertAlignmentLoss` | 5 | Cosine diversity scoring + norm-std diversity |
| M9 | M2FMoE `GatingUnit` | 5 | Sigmoid handoff with learned bias (init=2.94) |
| M11 | Merlion `ModelSelector` | 4,473 | Select-or-discard pattern based on validation metric |
| M13 | Darts `EnsembleModel` | 9,331 | Ridge regression meta-learner for combination weights |
| M13 | Kats (Facebook) | 6,296 | Backtesting + selection for weight calibration |

---

## Dependencies

All already installed:
- `PyWavelets>=1.7` -- wavelet decomposition (M2)
- `copulae>=0.7` -- copula fitting (M7)
- `statsmodels` -- Granger tests (M3), Johansen cointegration (M12)
- `scikit-learn` -- Ridge regression meta-learner (M13)
- `numpy`, `scipy` -- MinT matrix operations (M5), spectral analysis (M4)

No new dependencies needed.

---

## File Changes

| File | Change |
|------|--------|
| `operator1/models/frequency_fusion.py` | Rewrite: 499 -> ~1,400 lines. Add 13 methods. Keep `FrequencyFusionResult` interface. |
| `operator1/stages/stage7_integration.py` | Pass walk-forward errors and regime detector to fusion. |
| `backtest_runner.py` | Pass additional context to `_bt_mf_fuse`. |
| `config/scoring_weights.yml` | Add `frequency_fusion_v2` with per-method tunables. |
| `operator1/report/report_generator.py` | Update Multi-Frequency section with new fusion insights. |

What does NOT change: `hedge_fund/fusion.py`, `multi_frequency_runner.py`, `frequency_resampler.py`, per-frequency pipeline execution.
