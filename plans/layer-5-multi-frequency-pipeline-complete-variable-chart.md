# Layer 5: Multi-Frequency Pipeline -- Complete Variable Chart

Every variable produced by the 3 Layer 5 modules. This layer runs the full analytical pipeline at 5 frequencies (Annual -> Daily) with cascading context, then fuses results via a 13-method architecture. All outputs are result objects stored in `profile["multi_frequency"]` -- no direct cache columns added to the daily cache.

---

## 5.1 Frequency Resampler

**File:** `operator1/features/frequency_resampler.py` (704 lines)
**Pipeline step:** Step 6.7 (pre-processing)
**Profile key:** Consumed internally by multi_frequency_runner

**Purpose:** Resamples the daily cache to lower frequencies while preserving PIT constraints and correctly handling OHLCV vs financial statement data.

### Frequency Configuration

| Freq | Label | Resample Rule | Lookback |
|------|-------|---------------|----------|
| D | Daily | None (native) | 2 years |
| W | Weekly | W-FRI | 3 years |
| M | Monthly | ME | 5 years |
| Q | Quarterly | QE | 6 years |
| S | Semi-Annual | 2QE | 7 years |
| A | Annual | YE | 8 years |

### Resampling Rules

| Data Type | Aggregation | Examples |
|-----------|------------|---------|
| OHLCV | Open=first, High=max, Low=min, Close=last, Volume=sum | open, high, low, close, volume |
| Flow (I/S, C/F) | Sum over period | revenue, net_income, operating_cash_flow, capex |
| Stock (B/S) | Last value in period | total_assets, total_debt, cash_and_equivalents |

### ResampledCache Output Fields

| # | Variable | Type | Description |
|---|----------|------|-------------|
| 1 | `frequency` | String | Target frequency code (D/W/M/Q/S/A) |
| 2 | `label` | String | Human-readable label (Daily/Weekly/Monthly/etc.) |
| 3 | `lookback_years` | Integer | How many years of history included |
| 4 | `cache` | DataFrame | Resampled DataFrame with all columns |
| 5 | `n_periods` | Integer | Number of resampled periods |
| 6 | `is_partial_last_period` | Boolean | True if last period is incomplete (truncated to today) |
| 7 | `original_daily_rows` | Integer | Row count of input daily cache |
| 8 | `resampled_rows` | Integer | Row count after resampling |

**Annual-only market handling:** Markets in `ANNUAL_ONLY_MARKETS` (EU ESEF, UK Companies House, CH SIX, CL CMF, AU ASX, SG SGX) have Q-frequency caches built by interpolating annual filings (stock=linear, flow=distributed to quarters).

**Direct construction:** `build_cache_from_raw_filings()` constructs Q/A caches directly from raw statement DataFrames without going through daily forward-fill, avoiding interpolation artifacts.

---

## 5.2 Multi-Frequency Runner

**File:** `operator1/steps/multi_frequency_runner.py` (620 lines)
**Pipeline step:** Step 6.7
**Profile key:** `multi_frequency` (via fusion)

**Purpose:** Runs the full analytical pipeline at each frequency in slow-to-fast order (A -> Q -> M -> W -> D). Each frequency passes cascading context to the next.

### FrequencyContext (Cascading -- passed from slower to faster)

| # | Variable | Type | Description |
|---|----------|------|-------------|
| 1 | `frequency` | String | Source frequency code |
| 2 | `trend_direction` | String | "up" / "down" / "flat" from close price slope |
| 3 | `secular_regime` | String | "bull" / "bear" / "sideways" / "recovery" |
| 4 | `survival_probability_latest` | Float (0-1) | Latest survival probability at this frequency |
| 5 | `survival_regime` | String | "normal" / "company_survival" / "modified_survival" / "extreme_survival" |
| 6 | `forecast_bounds` | Dict | Per-variable (min, max) from historical range: e.g. {"close": (95.0, 210.5)} |
| 7 | `confidence` | Float (0-1) | Overall confidence from this frequency |

### FrequencyResult (Per-frequency output)

| # | Variable | Type | Description |
|---|----------|------|-------------|
| 1 | `frequency` | String | Frequency code (D/W/M/Q/A) |
| 2 | `label` | String | Human-readable label |
| 3 | `n_periods` | Integer | Number of data periods at this frequency |
| 4 | `elapsed_seconds` | Float | Wall-clock time for this frequency's pipeline |
| 5 | `regime_label` | String | Detected regime at this frequency |
| 6 | `survival_probability` | Float (0-1) | Survival probability at this frequency |
| 7 | `survival_regime` | String | Survival regime label |
| 8 | `trend_direction` | String | Trend direction detected |
| 9 | `forecast_summary` | Dict | Per-variable forecast values |
| 10 | `walk_forward_mae` | Float or None | Walk-forward mean absolute error |
| 11 | `context_for_next` | FrequencyContext | Context object to pass to next faster frequency |

### MultiFrequencyResult (Combined output)

| # | Variable | Type | Description |
|---|----------|------|-------------|
| 1 | `results` | Dict | Per-frequency FrequencyResult objects keyed by freq code |
| 2 | `execution_order` | List | Order frequencies were executed (e.g. ["A", "Q", "M", "W", "D"]) |
| 3 | `total_elapsed_seconds` | Float | Total wall-clock time across all frequencies |

### Per-Frequency Pipeline Steps

For each `ResampledCache`, the runner executes:
1. `compute_derived_variables(cache)` -- feature engineering
2. `compute_company_survival_flag(cache)` + `compute_survival_probability(cache)` -- survival detection
3. `detect_regimes_and_breaks(cache)` -- HMM/GMM/PELT regime detection
4. `run_forecasting(cache)` -- Kalman/GARCH/VAR/LSTM/Tree/ETS (if not skip_models)
5. `run_monte_carlo(cache)` -- regime-switching MC simulation (if not skip_models)
6. `anchor_mc_survival()` -- propagate market-cap survival floor
7. Extract summary context for next frequency

---

## 5.3 Frequency Fusion

**File:** `operator1/models/frequency_fusion.py` (1,283 lines)
**Pipeline step:** Step 6.7 (post-processing)
**Profile key:** `multi_frequency`

**Purpose:** Reconciles predictions, regimes, and survival probabilities from all frequency pipelines into a single coherent output using a 13-method fusion architecture.

### 13 Fusion Methods

| # | Method | Function | Purpose |
|---|--------|----------|---------|
| M1 | Break Alignment | `_m1_break_alignment()` | Align structural breaks across frequencies; confirmed if 2+ agree |
| M2 | Orthogonal Decomposition | `_m2_orthogonal_decomposition()` | Wavelet-band decomposition of each frequency's contribution |
| M3 | Granger Cascade | `_m3_granger_cascade()` | Test Granger causality direction between adjacent frequencies |
| M4 | Spectral Gating | `_m4_spectral_gating()` | Filter forecasts by frequency-appropriate spectral band |
| M5 | MINT Reconciliation | `_m5_mint_reconciliation()` | MinT optimal reconciliation with disagreement penalty for outlier frequencies |
| M6 | Resolution Gating | `_m6_resolution_gating()` | Gate frequencies by data quality (n_periods, walk_forward_mae) |
| M7 | Copula Joint Uncertainty | `_m7_copula_joint_uncertainty()` | Model joint survival uncertainty across frequencies using copula |
| M8 | Disagreement Signal | `_m8_disagreement_signal()` | Classify shape of cross-frequency disagreement (convergent/divergent/mixed) |
| M9 | Anomaly Injection | `_m9_anomaly_injection()` | Inject anomaly/outlier adjustments into fused predictions |
| M10 | Horizon Ownership | `_m10_horizon_ownership()` | Assign each horizon to its most informative frequency |
| M11 | AIC Frequency Selection | `_m11_aic_frequency_selection()` | Select best-fitting frequency per variable via AIC |
| M12 | Cointegration Anchor | `_m12_cointegration_anchor()` | Test if frequency forecasts share a long-run equilibrium |
| M13 | Meta-Learner | `_m13_meta_learner()` | Ridge regression meta-learner for combination weights |

### Horizon-to-Frequency Weights

| Horizon | D | W | M | Q | A |
|---------|---|---|---|---|---|
| 1d | 1.0 | | | | |
| 5d | 0.7 | 0.3 | | | |
| 21d | 0.3 | 0.4 | 0.3 | | |
| 3m | | 0.15 | 0.35 | 0.50 | |
| 1y | | | 0.15 | 0.35 | 0.50 |
| 2y | | | | 0.30 | 0.70 |

### RegimeConsensus Result

| # | Variable | Type | Description |
|---|----------|------|-------------|
| 1 | `consensus_regime` | String | Majority-vote regime label across frequencies |
| 2 | `agreement_ratio` | Float (0-1) | Fraction of frequencies agreeing on consensus |
| 3 | `frequency_regimes` | Dict | Per-frequency regime label: {"A": "bull", "Q": "bull", ...} |
| 4 | `disagreements` | List | Frequencies that disagree with consensus |
| 5 | `interpretation` | String | Human-readable interpretation text |
| 6 | `cascade_direction` | Dict | Granger cascade direction between adjacent frequencies |

### FusedSurvival Result

| # | Variable | Type | Description |
|---|----------|------|-------------|
| 1 | `fused_probability` | Float (0-1) | Final fused survival probability |
| 2 | `harmonic_mean` | Float (0-1) | Harmonic mean of per-frequency probabilities (weakest-link) |
| 3 | `per_frequency` | Dict | Per-frequency survival probability: {"A": 0.95, "Q": 0.88, ...} |
| 4 | `weakest_frequency` | String | Frequency showing lowest survival probability |
| 5 | `interpretation` | String | Human-readable interpretation text |
| 6 | `copula_tail_dependence` | Float | Lower tail dependence from copula joint uncertainty (M7) |
| 7 | `joint_p5` | Float or None | 5th percentile of joint survival distribution |
| 8 | `joint_p95` | Float or None | 95th percentile of joint survival distribution |

### FusedPrediction Result (per variable per horizon)

| # | Variable | Type | Description |
|---|----------|------|-------------|
| 1 | `variable` | String | Variable name (e.g. "close", "revenue") |
| 2 | `horizon` | String | Prediction horizon (e.g. "1d", "5d", "21d", "63d", "252d") |
| 3 | `point_forecast` | Float or None | Reconciled point forecast |
| 4 | `lower_bound` | Float or None | Lower confidence bound |
| 5 | `upper_bound` | Float or None | Upper confidence bound |
| 6 | `contributing_frequencies` | Dict | Per-frequency weight: {"D": 0.7, "W": 0.3} |
| 7 | `confidence` | Float (0-1) | Confidence in this fused prediction |
| 8 | `owner_freq` | String | Primary frequency for this horizon (from M10) |
| 9 | `adjustment_from_others` | Float | Size of adjustment from non-owner frequencies |

### DisagreementSignal Result

| # | Variable | Type | Description |
|---|----------|------|-------------|
| 1 | `shape` | String | "convergent" / "divergent" / "mixed" / "unknown" |
| 2 | `score` | Float (0-1) | Disagreement magnitude (0=full agreement) |
| 3 | `direction` | Float (-1 to +1) | Net directional bias of disagreement |
| 4 | `cosine_diversity` | Float | Cosine diversity measure across frequency forecasts |
| 5 | `confidence_adjustment` | Float | Multiplier to apply to confidence (< 1 when disagreement high) |

### BreakConfirmation Result (per confirmed break)

| # | Variable | Type | Description |
|---|----------|------|-------------|
| 1 | `date` | Date | Date of the structural break |
| 2 | `source_freq` | String | Frequency that first detected the break |
| 3 | `n_confirming` | Integer | Number of frequencies confirming this break |
| 4 | `confidence` | String | "low" / "medium" / "high" |
| 5 | `is_structural` | Boolean | True if confirmed as structural (not noise) |

### FrequencyFusionResult (Master output)

| # | Variable | Type | Profile Key |
|---|----------|------|-------------|
| 1 | `regime_consensus` | RegimeConsensus | `multi_frequency.regime_consensus` |
| 2 | `survival` | FusedSurvival | `multi_frequency.survival` |
| 3 | `predictions` | List of FusedPrediction | `multi_frequency.predictions` |
| 4 | `frequency_summary` | Dict | `multi_frequency.frequency_summary` |
| 5 | `n_frequencies_used` | Integer | `multi_frequency.n_frequencies_used` |
| 6 | `available` | Boolean | `multi_frequency.available` |
| 7 | `disagreement` | DisagreementSignal | `multi_frequency.disagreement` |
| 8 | `confirmed_breaks` | List of BreakConfirmation | `multi_frequency.confirmed_breaks` (count only in profile) |
| 9 | `excluded_frequencies` | List | `multi_frequency.excluded_frequencies` |
| 10 | `meta_learner_weights` | Dict | `multi_frequency.meta_learner_weights` |
| 11 | `cointegrated` | Boolean | `multi_frequency.cointegrated` |
| 12 | `cointegration_ect` | Float | (internal, not serialized to profile) |
| 13 | `methods_applied` | List | `multi_frequency.methods_applied` |

---

## Layer 5 Grand Total

| Module | Result Fields | Type |
|--------|--------------|------|
| 5.1 Frequency Resampler | 8 (per ResampledCache) | Internal (consumed by runner) |
| 5.2 Multi-Frequency Runner | 11 per freq + 3 combined | Internal + Result |
| 5.3 Frequency Fusion | 13 top-level + 6 RegimeConsensus + 8 FusedSurvival + 9 per FusedPrediction + 5 DisagreementSignal + 5 per BreakConfirmation | Result |
| **Profile output** | **~13 top-level fields** | **Stored in `profile["multi_frequency"]`** |

**0 new cache columns added to daily cache. ~13 top-level result fields + nested sub-objects** stored in `profile["multi_frequency"]` via `to_profile_dict()`.

---

## Running Total Across All 5 Layers

| Layer | Cache Columns | Result Fields |
|-------|--------------|---------------|
| Layer 1: Features | ~311 | ~9 |
| Layer 2: Analysis | ~61 | ~71 |
| Layer 3: Temporal | ~17 | ~121 |
| Layer 4: Hedge Fund | 0 | ~107 |
| Layer 5: Multi-Frequency | 0 | ~13 top-level + nested |
| **Total** | **~389** | **~321+** |
