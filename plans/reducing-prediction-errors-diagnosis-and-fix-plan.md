# Reducing Prediction Errors: Diagnosis and Fix Plan

## Current Baseline (AAPL 2024-12-31 cutoff, validated against 2025 actuals)

| Source | Horizon | Predicted | Actual | Error |
|--------|---------|-----------|--------|-------|
| Raw cascade | 1d | $250.68 | $242.53 | 3.36% |
| Raw cascade | 5d | $251.35 | $241.38 | 4.13% |
| Raw cascade | 21d | $252.22 | $226.77 | 11.22% |
| **Aggregated ensemble** | **1d** | **$245.67** | **$242.53** | **1.30%** |
| **Aggregated ensemble** | **5d** | **$246.32** | **$241.38** | **2.05%** |
| **Aggregated ensemble** | **21d** | **$247.18** | **$226.77** | **9.00%** |
| MF fusion | 1d | $250.76 | $242.53 | 3.40% |
| MF fusion | 5d | $255.63 | $241.38 | 5.91% |
| MF fusion | 21d | $254.95 | $226.77 | 12.43% |
| 252d forecast | -- | $264.01 | $272.82 EOY | 3.23% |

Key context: AAPL dropped ~9.4% in the first 21 trading days of Jan 2025, then recovered to +8.95% by year-end. The actual 21d close of $226.77 included the early-January selloff driven by macro events (tariff fears, DeepSeek AI disruption).

---

## Error Decomposition: Where the Errors Come From

### Problem 1: Close confidence = 0.0 (zero model confidence)

The aggregated close prediction has `confidence: 0.0` at all horizons. This means the prediction aggregator assigned zero confidence to the close forecast, which disables several downstream adjustments (survival-intensity correction, scenario bounding, feature-weighted confidence scaling). The root cause: the close variable was predicted by the baseline EMA model (not Kalman/GARCH/Tree), and the `ConformalCalibrator` received no valid residuals for close.

**Fix:** Ensure the close variable gets Kalman or Tree model fit, not just baseline EMA. The cascade in [`forecasting.py`](operator1/models/forecasting.py) tries models in order but close may be failing early models due to non-stationarity. Adding a differencing step (predict return, transform back to price) would help Kalman/GARCH converge on the close series.

### Problem 2: HF Scorecard shows C grade with critical valuation tier (0/100)

The HF scorecard rated AAPL as "C" with conviction=4. The valuation tier scored **0/100** (critical). PEG composite is unavailable. PE ratio, EV/EBITDA, earnings yield are all null/zero in the profile. This means the entire valuation layer -- which should inform the prediction aggregator's fundamental fair value anchor -- is missing.

**Root cause:** `net_income` and `eps` are not in the cache (null net_margin, null PE). The edgartools XBRL extraction returned revenue and balance sheet but missed income statement items (net_income, eps_diluted). The CompanyFacts fallback only covers 4 balance sheet fields.

**Fix:** Extend the CompanyFacts fallback to also fetch income statement fields: `NetIncomeLoss`, `EarningsPerShareDiluted`, `EarningsPerShareBasic`, `OperatingIncomeLoss`, `GrossProfit`.

### Problem 3: Piotroski F-Score only 2/9 available

Only 2 of 9 Piotroski components could be computed. This severely limits the HF forensic analysis quality. The missing components all depend on income statement fields that are absent.

**Fix:** Same as Problem 2 -- extend CompanyFacts fallback.

### Problem 4: MF fusion close predictions WORSE than daily-only

The MF fusion close 5d prediction ($255.63) was 5.91% off vs the aggregated ensemble (2.05%). The MF fusion applies horizon-to-frequency weights (5d: 70% daily + 30% weekly) but the weekly pipeline may have different regime detection (M says "moderate_return" while D/W say "bull"). The cascade direction shows M->W is "divergent", meaning the monthly frequency disagrees with the weekly.

**Fix:** The MF fusion should weight by inverse prediction variance (not fixed horizon weights) when per-frequency predictions diverge significantly. When `freq_disagreement > 0.5` (currently 0.63), reduce the contribution of disagreeing frequencies rather than averaging them.

### Problem 5: Degenerate HMM regime detection (100% bull)

The HMM detected 4 regimes but the "bull" state had only 0.4% of observations, triggering the PELT fallback. However, the PELT fallback classified ALL 502 days as "bull" (3 segments, all bull). This means the regime-switching Monte Carlo and regime-conditional weighting are effectively disabled -- everything runs in a single regime.

**Root cause:** AAPL had a strong 2023-2024 run (+100% total return, Sharpe 1.96). The return distribution was genuinely unimodal-positive. The PELT fallback uses mean return > 0.0005/day as the bull threshold, which was met for all 3 segments.

**Fix:** (a) Use the GMM fit (which detected 4 components) as a secondary regime source when PELT produces a single regime. (b) Add a volatility-clustering regime overlay: even in a bull market, there are high-vol vs low-vol periods that should inform MC path generation and confidence interval widths.

### Problem 6: Survival mode over-triggering (63.75% of days)

The profile shows 320 survival days out of 502 (63.75%) with 5 episodes. AAPL's `current_ratio = 1.012` is just barely above the 1.0 threshold. Any day where the forward-filled ratio dips below 1.0 triggers survival mode. This is a false positive -- AAPL is the most cash-rich company in the world ($30B cash, but also $131B current liabilities in their treasury operations).

**Impact on predictions:** Survival mode shifts hierarchy weights from equal (20/20/20/20/20) to liquidity-focused (50/30/15/4/1), which downweights growth/profitability features and biases predictions toward conservative/defensive postures. This explains the slight downward bias in predictions.

**Fix:** The adaptive thresholds module should account for the absolute magnitude of cash (not just the ratio). A company with $30B cash and current_ratio of 1.01 is in a fundamentally different position than one with $1M cash and the same ratio. Add a "cash adequacy floor" that prevents survival triggering when absolute cash exceeds a market-cap-relative threshold (e.g., cash > 5% of market cap).

### Problem 7: Confidence interval width explosion at 21d

The 21d aggregated bounds are $120.19 to $384.25 -- a 107% width relative to the predicted price. This is far too wide to be useful. The 5d bounds (45% width) are marginally useful, and 1d bounds (20% width) are reasonable.

**Root cause:** The conformal calibrator widens intervals by `sqrt(horizon_days)` scaling. Additionally, the copula tail multiplier and regime transition probability both add multiplicative widening. With a single-regime environment (Problem 5), the regime transition probability is near-zero, so that widening channel is disabled -- the width comes entirely from the conformal residuals and copula.

**Fix:** (a) Cap the maximum CI width at a configurable threshold (e.g., 40% for 21d). (b) Use the OU mean-reversion parameters already computed (theta=0.002, mu=$332.35, half-life=337 days) to bound the 21d interval: a mean-reverting process cannot deviate beyond a statistically bounded range in 21 days. (c) The quantile regression calibrator (already wired in stage 2c) should provide tighter asymmetric intervals if it received enough residuals.

### Problem 8: MF daily survival at 50% drags fused survival down

The MF fused survival probability is 0.8 (harmonic mean), pulled down by the D frequency showing 50% survival. The Q, M, W frequencies all show 100%. This means the daily pipeline's MC simulation is still producing marginal survival results despite the CompanyFacts fix improving current_ratio from NaN to 1.012.

**Root cause:** current_ratio=1.012 is just barely above the survival threshold. The MC simulation has path variance that pushes some paths below 1.0 at the daily frequency (not at Q/M/W where the ratio is smoothed). The market-cap survival floor added in PR #2 should address this, but the floor may not be activating correctly for the daily MF sub-pipeline.

**Fix:** Verify the market-cap survival floor propagates to the per-frequency MC runs inside the MF pipeline. Currently the floor is applied in [`stage5_forward.py:run_5_4_monte_carlo`](operator1/stages/stage5_forward.py) but the MF sub-pipeline calls `run_monte_carlo()` directly from [`multi_frequency_runner.py`](operator1/steps/multi_frequency_runner.py) without the floor logic.

### Problem 9: No net_income/EPS means no PE anchor

Without PE ratio, the prediction aggregator cannot use its fundamental fair value anchor (Step B4 in the aggregation pipeline). This anchor pulls extreme price predictions toward a PE-implied fair value. With it disabled, predictions are purely technical/statistical with no fundamental gravity.

**Fix:** Same as Problem 2 -- extend CompanyFacts. Additionally, compute a synthetic PE from `operating_income / shares_outstanding` as a fallback when net_income is unavailable.

### Problem 10: Linked entities unavailable

The profile shows `linked_entities: available=false`. Entity discovery failed to find linked entities (competitors, suppliers, etc.), which means:
- No peer-adjusted survival thresholds
- No cross-entity DTW analogs (peer analog weight = 0)
- No competitive pressure index
- No linked aggregate features in _extra_vars

**Root cause:** Entity discovery requires an LLM client. The OpenRouter key was provided, but the LLM client creation may have failed silently or the entity discovery call may have timed out.

**Fix:** Debug the LLM client creation path and ensure entity discovery completes. As a fallback, add a static competitor list for well-known companies (AAPL competitors: MSFT, GOOG, AMZN, META, SAMSUNG).

---

## Prioritized Fix Plan

### Tier 1: High Impact (directly reduces prediction error)

- [ ] **Extend CompanyFacts fallback to income statement fields** -- fills net_income, EPS, operating_income, gross_profit. Enables PE anchor, full Piotroski, and HF valuation tier. Estimated impact: 1-3% error reduction on 21d+
- [ ] **Fix close variable model selection** -- ensure Kalman or Tree model fits close (predict returns, transform back). Estimated impact: 0.5-1% error reduction
- [ ] **Fix degenerate regime detection** -- use GMM as secondary source when PELT produces single regime; add volatility-clustering overlay. Estimated impact: improves MC path generation and regime-conditional weighting
- [ ] **Fix MF daily survival floor propagation** -- ensure market-cap floor applies in per-frequency MC runs. Estimated impact: fixes fused survival from 0.8 to ~0.95+

### Tier 2: Medium Impact (improves uncertainty quantification)

- [ ] **Cap confidence interval width** -- max 40% at 21d; use OU mean-reversion bounds. Estimated impact: makes 21d intervals actionable
- [ ] **Fix MF fusion to use inverse-variance weighting** -- when frequency disagreement > 0.5, reduce disagreeing frequency contribution. Estimated impact: fixes MF close 5d from 5.91% to ~2% error
- [ ] **Add cash adequacy floor to survival triggers** -- prevent over-triggering for cash-rich companies. Estimated impact: reduces false survival episodes from 63.75% to ~10%

### Tier 3: Important but Lower Impact

- [ ] **Debug entity discovery / LLM client** -- ensure linked entities populate. Estimated impact: enables peer features, better DTW analogs
- [ ] **Compute synthetic PE fallback** -- operating_income / shares from CompanyFacts. Estimated impact: enables fundamental fair value anchor even without net_income
- [ ] **Add AAPL-specific XBRL concept mapping** -- Apple uses `RevenueFromContractWithCustomerExcludingAssessedTax` not `Revenues`. Some CompanyFacts concepts may need LLM taxonomy resolution for non-standard filers.

---

## Expected Outcome After Fixes

| Source | Horizon | Current Error | Target Error |
|--------|---------|--------------|--------------|
| Aggregated ensemble | 1d | 1.30% | under 1.0% |
| Aggregated ensemble | 5d | 2.05% | under 1.5% |
| Aggregated ensemble | 21d | 9.00% | under 5.0% |
| MF fusion | 5d | 5.91% | under 2.5% |
| 252d | -- | 3.23% | under 3.0% |

The biggest single wins will come from extending CompanyFacts (enables PE anchor + full HF scoring) and fixing the degenerate regime detection (enables regime-conditional MC and weighting). Together these should halve the 21d error.
