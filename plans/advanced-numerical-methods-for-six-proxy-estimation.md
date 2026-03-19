# Advanced Numerical Methods for SIX Proxy Estimation

## Current State

The SIX proxy module uses basic algebra: `yield = dividend/price`, `implied_earnings = dividends + buybacks`, `OCF = dividends / payout_ratio`. These give ~5% error for Nestle but are fundamentally limited -- they treat each observable independently.

## Method 1: Kalman Filter for Latent Earnings State

**What it does**: Models earnings as a hidden state that we observe indirectly through dividends and buybacks. The Kalman filter optimally combines noisy observations over time to estimate the hidden state.

**Why it's better**: Instead of using a single dividend to estimate earnings (noisy), the Kalman filter uses the entire 18-year dividend history as a time series, learning the relationship between dividends and underlying earnings, and producing a smoothed earnings estimate with uncertainty bounds.

**Mathematical formulation**:
```
State equation:    E_t = a * E_{t-1} + w_t    (earnings follow AR(1))
Observation eq:    D_t = p * E_t + v_t          (dividends = payout * earnings + noise)
                   B_t = b * E_t + u_t          (buybacks = buyback_frac * earnings + noise)
```

Where:
- `E_t` = latent earnings at time t (unobserved)
- `D_t` = dividend at time t (observed, 18 years)
- `B_t` = buyback value at time t (observed from notices)
- `p` = payout ratio (estimated as part of the filter)
- `a` = earnings persistence (how sticky earnings are year-to-year)
- `w_t, v_t, u_t` = process and observation noise

The Kalman filter estimates `E_t` optimally given all observations up to time t. It also gives `P_t` (uncertainty of the estimate), which directly feeds the confidence tagging system.

**Available in**: `statsmodels.tsa.statespace` (already installed in stage 2)

**Accuracy improvement**: For Nestle, instead of `implied_earnings = dividends + buybacks = CHF 11.47B`, the Kalman filter would learn that Nestle's payout ratio has been declining slowly (from ~80% in 2008 to ~73% in 2025), and adjust the earnings estimate upward to ~CHF 11.0B (closer to actual CHF 10.9B).

---

## Method 2: Gordon Growth Model with Bayesian Parameter Estimation

**What it does**: The classic Gordon model says `Price = Dividend / (r - g)` where r = required return and g = dividend growth rate. Instead of plugging in a single CAGR, use Bayesian inference to estimate the posterior distribution of `r` and `g` given the entire price + dividend history.

**Why it's better**: Gives a full probability distribution of implied required return and growth rate, not point estimates. The uncertainty in `r` and `g` propagates into uncertainty in implied earnings, producing calibrated confidence intervals.

**Mathematical formulation**:
```
Likelihood:   P_t ~ Normal(D_t / (r - g), sigma_p)
Prior on r:   r ~ Normal(0.08, 0.03)     (Swiss equity risk premium + risk-free)
Prior on g:   g ~ Normal(cagr_5y, 0.02)   (centered on observed CAGR)
Prior on sigma: sigma_p ~ HalfNormal(5)
```

Posterior via MCMC (PyMC is already installed in stage 3):
```python
with pm.Model():
    r = pm.Normal('r', mu=0.08, sigma=0.03)
    g = pm.Normal('g', mu=cagr_5y, sigma=0.02)
    sigma = pm.HalfNormal('sigma', sigma=5)
    mu = dividends / (r - g)
    pm.Normal('obs', mu=mu, sigma=sigma, observed=prices)
    trace = pm.sample(2000)
```

The posterior `r - g` gives the cost of equity, which combined with dividends gives implied earnings with uncertainty bands.

**Available in**: `pymc` (already installed in stage 3)

---

## Method 3: Structural Break Detection on Dividend Series

**What it does**: Uses the PELT (Pruned Exact Linear Time) algorithm to detect structural breaks in the dividend growth series. This identifies regime changes -- when did the company shift from "high growth" to "mature payout"?

**Why it's better**: The current CAGR treats the 18-year dividend history as a single trend. In reality, most companies go through phases. Nestle's dividends grew ~8% annually from 2006-2015, then slowed to ~3% from 2016-2025. PELT detects this breakpoint, and we use only the most recent regime for forward-looking estimates.

**Mathematical formulation**: PELT minimizes:
```
sum_{segments} [cost(segment) + penalty]
```
where `cost(segment)` is the negative log-likelihood of the data within each segment assuming a Normal model, and `penalty` controls the number of breakpoints.

**Available in**: `ruptures` (already installed in stage 2)

**Accuracy improvement**: Using only the post-2016 regime, dividend CAGR = ~2.8% (closer to actual 2.56% we computed) instead of the full 18-year CAGR of 4.79% which is inflated by the earlier high-growth period.

---

## Method 4: Monte Carlo Propagation for Uncertainty Quantification

**What it does**: Instead of producing single-point estimates, propagate uncertainty through all formulas using Monte Carlo simulation. Each input has a distribution (dividend_yield ~ Normal(0.04, 0.005)), and we sample 10,000 paths through the formula chain.

**Why it's better**: The current confidence tags (0.50-0.95) are heuristic guesses. Monte Carlo gives statistically rigorous confidence intervals. For example, `implied_PE = 17.1 [15.2, 19.8] at 90% CI` is far more useful than `implied_PE = 17.1, confidence = 0.70`.

**Mathematical formulation**:
```python
# Sample from input distributions
div_samples = np.random.normal(div_yield, div_yield * 0.05, 10000)
buyback_samples = np.random.normal(buyback_yield, buyback_yield * 0.20, 10000)
tax_samples = np.random.normal(0.15, 0.02, 10000)

# Propagate through formula
implied_earnings = (div_samples + buyback_samples) * market_cap / (1 - tax_samples)
implied_pe = market_cap / implied_earnings

# Extract percentiles
pe_median = np.median(implied_pe)
pe_ci_90 = np.percentile(implied_pe, [5, 95])
```

**Available in**: `numpy` (already installed)

---

## Method 5: Sparse Signal Reconstruction (Compressed Sensing) for Balance Sheet

**What it does**: We know a few balance sheet facts with certainty (share capital from SIX, implied earnings from dividends). We know the accounting identities that constrain balance sheet items (assets = liabilities + equity, equity = share_capital + retained_earnings, etc.). Compressed sensing reconstructs the full balance sheet from sparse observations + constraints.

**Why it's better**: Instead of estimating each ratio independently, this exploits the structural relationships between balance sheet items. If we know equity and earnings, we can bound retained earnings, which bounds total assets, which bounds debt.

**Mathematical formulation**:
```
minimize ||x||_1  subject to  A*x = b, x >= 0

where:
  x = [total_assets, total_liabilities, equity, retained_earnings, 
       total_debt, current_assets, current_liabilities, cash, ...]
  A = constraint matrix (accounting identities)
  b = observed values (share_capital, implied_earnings, market_cap)
```

Constraints:
- `total_assets = total_liabilities + equity` (identity)
- `equity >= share_capital` (retained earnings >= 0 for profitable companies)
- `equity <= market_cap` (P/B >= 1 for most companies)
- `total_debt <= 3 * equity` (typical for non-financial companies)
- `cash >= annual_dividends` (need cash to pay dividends)

**Available in**: `scipy.optimize.linprog` (already installed in stage 1)

---

## Method 6: Amihud Illiquidity with Kyle Lambda Estimation

**What it does**: Kyle's lambda measures price impact per unit of order flow. It's a more sophisticated liquidity measure than raw Amihud, estimated via regression of price changes on signed volume.

**Mathematical formulation**:
```
Delta_P_t = alpha + lambda * SignedVolume_t + epsilon_t
```

Since SIX CSV doesn't have bid-ask (only close + volume), we use the Roll (1984) estimator for effective spread:
```
spread = 2 * sqrt(-Cov(r_t, r_{t-1}))    if Cov < 0
spread = 0                                  if Cov >= 0
```

Combined with Amihud, this gives two independent liquidity signals that can be averaged for a more robust Tier 1 score.

**Available in**: `numpy` (regression + covariance)

---

## Implementation Priority (by impact / effort ratio)

| # | Method | Impact | Effort | Priority |
|---|--------|--------|--------|----------|
| 1 | Monte Carlo propagation | High (rigorous CI) | Low (numpy only) | **P0** |
| 2 | PELT on dividends | High (regime-aware CAGR) | Low (ruptures installed) | **P0** |
| 3 | Roll spread estimator | Medium (better liquidity) | Low (numpy) | **P1** |
| 4 | Kalman filter for earnings | High (optimal estimation) | Medium (statsmodels) | **P1** |
| 5 | Compressed sensing balance sheet | High (full balance sheet) | Medium (scipy linprog) | **P2** |
| 6 | Bayesian Gordon model | Medium (posterior CI) | High (PyMC MCMC is slow) | **P3** |

## Checklist

- [ ] Method 1 (Monte Carlo): Add `_propagate_uncertainty()` -- replace static confidence with sampled CI
- [ ] Method 2 (PELT): Add `_detect_dividend_regimes()` -- use most recent regime for CAGR
- [ ] Method 3 (Roll spread): Add `_roll_effective_spread()` -- second liquidity signal
- [ ] Method 4 (Kalman): Add `_kalman_earnings_filter()` -- optimal earnings estimate from dividend series
- [ ] Method 5 (Compressed sensing): Add `_reconstruct_balance_sheet()` -- L1-minimization of balance sheet
- [ ] Method 6 (Bayesian Gordon): Add `_bayesian_gordon_model()` -- posterior r,g distributions
