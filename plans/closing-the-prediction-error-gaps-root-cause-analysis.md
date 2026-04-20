# Closing the Prediction Error Gaps -- Root Cause Analysis

*2026-04-20*

## The Errors

| Horizon | Predicted | Actual | Error | Direction |
|---------|-----------|--------|-------|-----------|
| 1 day   | $250.25   | $242.53 | +3.19% | Predicted too high |
| 5 day   | $249.85   | $241.38 | +3.51% | Predicted too high |
| 21 day  | $249.50   | $226.77 | +10.02% | Predicted too high |
| 252 day | ~$250     | $272.82 | ~-8.9% | Would have predicted too low |

Additional context:
- **21d conformal interval:** $-60.22 to $559.21 -- absurdly wide, useless
- **HF signal:** strong_sell (-1.00) -- correct about the drawdown, wrong about the recovery
- **MC survival 252d:** 12.03% -- way too pessimistic (AAPL ended +8.95%)
- **Regime:** high_vol dominant (495/502 days) -- but 2024 AAPL was not particularly volatile

---

## Error #1: Systematic Upward Bias in Short-Term Predictions

**What happened:** All 3 horizons predicted the price would stay near $250 (the 2024-12-31 close). AAPL actually dropped 3.1% on the first day and kept falling to $226.77 by day 21.

**Root cause: Kalman filter mean-reversion bias.** The Kalman local-level model is essentially a random walk with drift. When it sees a stock that ended 2024 near its all-time high ($250.42), the one-step-ahead prediction is just slightly below the last observation. It has no mechanism to predict a directional move.

**What's missing:**

1. **Momentum signals not flowing to predictions.** The cross-asset rotation signals (Gap 3) detected sector_relative_strength and sector_dispersion, but these were only 2 of 76 candidate features. After feature selection, only fh_liquidity_score survived. The momentum-relevant features (return_5d, return_21d, MACD histogram) were excluded from extra_vars because they're already in the cache as target-adjacent variables, not features.

2. **No earnings calendar adjustment.** AAPL reports Q1 earnings in late January. The event_calendar module computed days_to_next_event and event_uncertainty_premium, but these were pruned by feature selection (Boruta rejected them -- they're near-constant for 500 days and only spike near events).

3. **No January seasonality.** The "January effect" and tax-loss selling recovery are well-documented seasonal patterns. The pipeline has no explicit seasonality model for calendar effects.

**Fix needed:** Add a **momentum overlay** to the Kalman/baseline predictions. When the last 5d/21d return is negative and accelerating, shift the prediction downward. This could be as simple as blending with a momentum model:
```
adjusted = 0.7 * kalman_pred + 0.3 * (last_close * (1 + return_21d/21 * horizon))
```

---

## Error #2: 21-Day Conformal Interval Is Useless (-$60 to $559)

**What happened:** The 21d interval spans $619 (-$60 to $559), which is wider than the stock price itself. This provides zero actionable information.

**Root cause: Conformal calibrator has too few residuals at the 21d horizon.** The ConformalPIDCalibrator adjusts its quantile based on residual history. At the 1d horizon it has 335 residuals (one per day). At the 21d horizon it has ~24 residuals (one per 21-day window). With so few samples, the quantile estimate is noisy and defaults to wide intervals.

Additionally, the **regime_vol_ratio widening** is compounding the problem. The code at [`stage6_ensemble.py:131-143`](operator1/stages/stage6_ensemble.py:131) computes _vol_ratio from HMM emission distributions. When one regime has very few observations (bull had only 2 days), the vol ratio becomes extreme, causing massive interval widening.

**What's missing:**

1. **Horizon-specific calibration.** The calibrator feeds all residuals into one pool. It should maintain separate calibrators per horizon (1d, 5d, 21d, 252d) so short-horizon residuals don't contaminate long-horizon intervals.

2. **Vol ratio floor.** When a regime has <10 observations, the vol_ratio should be clamped to a reasonable maximum (e.g., 3.0) instead of allowing degenerate values.

3. **Interval sanity bounds.** No interval should be wider than some fraction of the stock price (e.g., +/-50% at most for 21d). A hard floor/ceiling would catch degenerate cases.

---

## Error #3: MC Survival 252d = 12% Is Way Too Pessimistic

**What happened:** The Monte Carlo estimated 12% survival probability at 252 days. AAPL ended 2025 at +8.95% total return. The company was never in genuine distress.

**Root cause: Survival triggers are calibrated for all companies, not mega-caps.** The survival thresholds include `drawdown_252d < -0.40` and `fcf_yield < 0`. For AAPL:
- The drawdown trigger (-40%) is reasonable but the MC simulation path samples from the high_vol regime distribution, which has fat tails (Student-t df=7.1). Over 252 steps, many paths breach -40%.
- The survival threshold adaptive calibration only adapted 2 of 12 thresholds (drawdown_252d and fh_label_breaks) due to 0 available peers.

**What's missing:**

1. **Market-cap-aware survival calibration.** A $3T company has fundamentally different survival dynamics than a $1B company. Mega-caps almost never breach drawdown_252d < -40% because they ARE the market. The peer-based calibration (adaptive_thresholds) tried to adjust but found 0 peers because SEC rate limiting blocked company search.

2. **Peer data.** Entity discovery found 0 linked entities (SEC returned 429 Too Many Requests on company_tickers.json). Without peers, the adaptive threshold calibration falls back to textbook defaults, which are calibrated for average companies, not mega-caps.

3. **Sector-level calibration.** Even without peers, the system could calibrate thresholds using sector-wide distributions from FRED, yfinance sector ETFs, or Fama-French factor returns.

---

## Error #4: HF Strong Sell Was Wrong

**What happened:** HF analysis gave Grade=C, Signal=-1.00 (strong_sell). AAPL ended 2025 at +8.95%.

**Root cause: Structural data gap in EDGAR XBRL extraction.** Several critical ratios came back as NaN:
- `cash_ratio`: NaN (all 502 days)
- `current_ratio`: NaN (all 502 days)
- `interest_coverage`: NaN (all 502 days)
- `net_margin`: NaN (all 502 days)
- `pe_ratio_calc`: NaN (all 502 days)

Without these, the HF pipeline sees:
- Piotroski F-Score: 2/9 (most signals are NaN -> score 0)
- Altman Z'': -2.28 (NaN ratios default to 0 in the formula)
- DCF: $33.05 (sanity-flagged as unreliable)

The XBRL extraction got `total_assets`, `total_liabilities`, `total_equity`, `revenue`, `net_income`, `operating_cash_flow` etc., but missed `current_assets`, `current_liabilities`, and `cash_and_equivalents` -- the building blocks of liquidity ratios.

**What's missing:**

1. **XBRL concept coverage gap.** The US-GAAP concept map in [`us_edgar.py:41-83`](operator1/clients/us_edgar.py:41) maps ~45 concepts. But Apple uses non-standard XBRL tags for some items (e.g., `us-gaap:CashCashEquivalentsRestrictedCashAndRestrictedCashEquivalents` instead of the mapped `CashAndCashEquivalentsAtCarryingValue`). The concept map needs expanding.

2. **Estimation engine fallback.** When XBRL extraction misses these fields, the estimation engine should derive them from related fields. For example: `current_assets = total_assets - non_current_assets`, or `cash_and_equivalents` can be derived from the cashflow statement's ending cash balance. The identity fill phase has some of these, but not all.

3. **NaN-aware scoring.** The HF modules (Piotroski, Altman) should distinguish between "ratio is genuinely bad" and "ratio is unavailable". A NaN current_ratio should not default to 0 (which looks like bankruptcy) -- it should be excluded from the scoring denominator.

---

## Error #5: Regime Detection Overfit to High-Vol

**What happened:** HMM classified 495/502 days as high_vol, with only 2 days as bull. This is unreasonable for AAPL in 2023-2024, which was in a clear uptrend (+37% in 2024).

**Root cause: 4-state HMM with insufficient data per state.** The HMM fitted 4 regimes on 502 observations of (return_1d, volatility_21d). With only 2 observations in the "bull" state, the emission distribution is degenerate. The "high_vol" state absorbed almost everything because AAPL's 2023-2024 returns had moderate volatility that the HMM classified as elevated compared to its learned "low_vol" state.

**What's missing:**

1. **Regime label calibration.** The labels "bull", "bear", "high_vol", "low_vol" are assigned post-hoc based on emission means. A state with positive mean return but moderate volatility gets labeled "high_vol" if the volatility mean is above the median across states. The labeling should incorporate both return AND volatility jointly.

2. **BIC-based regime count selection.** The HMM always uses 4 states. It should cross-validate with BIC/AIC across 2-6 states and pick the best. For a 502-day series, 2-3 states might be more appropriate.

3. **Rolling HMM.** The current HMM fits on the full 502-day history (with look-ahead), then the labels are applied retrospectively. The forward pass uses these labels for weighting, creating a subtle look-ahead bias. A rolling HMM (refit every 63 days) would produce purely online regime labels.

---

## Summary: 6 Improvements to Close the Gaps

| # | Issue | Impact | Complexity | Files |
|---|-------|--------|-----------|-------|
| 1 | **Expand XBRL concept coverage** | HIGH -- fixes NaN ratios, fixes HF scoring | Low (~20 concept additions) | `us_edgar.py` |
| 2 | **NaN-aware HF scoring** | HIGH -- Piotroski/Altman stop penalizing missing data | Medium (~30 lines per module) | `hedge_fund/engine.py`, `hedge_fund/advanced_methods.py` |
| 3 | **Momentum overlay for Kalman** | MEDIUM -- reduces directional bias | Low (~15 lines) | `forecasting.py` |
| 4 | **Horizon-specific conformal calibration** | MEDIUM -- fixes useless 21d intervals | Medium (~40 lines) | `conformal.py` |
| 5 | **Market-cap-aware survival thresholds** | MEDIUM -- fixes overly pessimistic MC | Medium (~50 lines) | `adaptive_thresholds.py` |
| 6 | **BIC-based HMM regime count** | LOW -- better regime detection | Medium (~30 lines) | `regime_detector.py` |

Items 1 and 2 are the highest-impact fixes -- they directly address why the HF analysis produced a strong_sell on a +8.95% stock. The XBRL concept coverage gap means AAPL's liquidity data was invisible to the pipeline, making a healthy $3T company look like it had no cash.
