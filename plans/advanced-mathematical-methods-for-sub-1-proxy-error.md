# Advanced Mathematical Methods for Sub-1% Proxy Error

## Current Error Analysis (Nestle Baseline)

| Metric | Current Error | Root Cause |
|--------|--------------|------------|
| Net Income | 2.3% | Kalman local-level model assumes random walk; ignores mean-reverting profitability |
| P/E Ratio | 1.3% | Derived from NI error + price noise |
| Payout | 2.3% | Denominator (Kalman NI) propagates 2.3% bias |
| Total Equity | 1.9% | Market-implied median of 3 estimates; P/B ratio is sector-average, not company-specific |
| Total Assets | 32.4% | Goodwill/intangibles from acquisitions unobservable from market data alone |

The fundamental constraint: we observe dividends, prices, volume, capital structure, and corporate actions. We do NOT observe income statements, balance sheets, or cash flow statements. Every method below is a different way to extract maximum information from what we DO observe.

---

## Method 1: Two-Factor Kalman with Payout State (Lintner-Kalman Hybrid)

**Current limitation:** The Kalman filter models earnings as a single local-level state. But Lintner (1956) showed dividends depend on TWO latent states: earnings AND the target payout ratio. The payout ratio itself drifts over time (Nestle: 80% in 2008 to 73% in 2024).

**Mathematical formulation:**

```
State vector: x_t = [E_t, p_t]'    (latent earnings, latent payout ratio)

Transition:   x_t = F * x_{t-1} + w_t
where F = [[a, 0],                   -- earnings persistence
            [0, rho]]                 -- payout ratio persistence

Observation:  D_t = p_t * E_t + v_t  (dividend = payout * earnings + noise)

This is nonlinear (p_t * E_t product), so use:
- Extended Kalman Filter (EKF), or
- Unscented Kalman Filter (UKF) from filterpy
```

The UKF handles the nonlinear observation equation without linearization error. The two-state model simultaneously estimates earnings AND payout ratio, eliminating the circularity in the current approach (where we estimate payout from sector, then use it to estimate earnings, then re-estimate payout from earnings).

**Expected improvement:** The payout ratio error (2.3%) should drop to <1% because the filter learns the company-specific payout trajectory rather than using a sector average.

**Dependency:** `filterpy` (pip installable, ~50KB) or hand-coded UKF (numpy only)

---

## Method 2: Ohlson Residual Income Model (1995) for Equity Valuation

**Current limitation:** Market-implied equity uses sector-average P/B ratios. The Ohlson (1995) model relates book equity to market price through residual income, giving a company-specific P/B that can be inverted.

**Mathematical formulation:**

```
P_t = BV_t + sum_{k=1}^{inf} R^{-k} * E_t[RI_{t+k}]

where:
  RI_t = NI_t - r * BV_{t-1}          (residual income)
  BV_t = BV_{t-1} + NI_t - D_t        (clean surplus)
  R = 1 + r                            (discount factor)

Linear information dynamics (Ohlson):
  RI_{t+1} = omega * RI_t + v_t + eps_{t+1}
  v_{t+1} = gamma * v_t + eta_{t+1}

Closed form:
  P_t = BV_t + alpha_1 * RI_t + alpha_2 * v_t

where:
  alpha_1 = omega / (R - omega)
  alpha_2 = R / ((R - omega)(R - gamma))
```

We observe P_t (price) and D_t (dividends). We estimate NI_t from Kalman. We can therefore solve for BV_t by inverting:

```
BV_t = (P_t - alpha_2 * v_t) / (1 + alpha_1 * (NI_t/BV_t - r))
```

This is a fixed-point equation solved iteratively. The result is a company-specific book value that is consistent with both market price and residual income dynamics.

**Expected improvement:** Equity error from 1.9% to <1%, because the model uses the actual market price (which embeds all public information) rather than a sector-average P/B.

**Dependency:** `numpy` (iterative solver)

---

## Method 3: Feltham-Ohlson (1995) for Total Assets with Growth

**Current limitation:** Total assets (32.4% error) are derived from revenue/asset_turnover using sector averages. Nestle's actual asset base is inflated by CHF ~45B of goodwill from acquisitions -- invisible from market data.

**Mathematical formulation:**

The Feltham-Ohlson model distinguishes between operating assets (OA) and financial assets (FA):

```
P_t = FA_t + OA_t + sum R^{-k} E[OI_{t+k} - r*OA_{t+k-1}]

OA grows at rate g (related to capex and depreciation):
  OA_{t+1} = (1 + g) * OA_t + capex_t - depreciation_t
```

The key insight: if we know equity (from Method 2) and total_debt (from the L1 optimizer), then:

```
total_assets = total_equity + total_liabilities
             = BV_t + total_debt + non_debt_liabilities
```

The L1 optimizer already gets equity right (35.5B vs 36.2B). The error is in total_liabilities. For Nestle, total_liabilities = 82B, but L1 estimates ~44B because it uses the debt constraint `total_debt <= 3 * equity`.

**Fix:** Replace the rigid `3x equity` constraint with the Merton structural model for debt capacity:

```
D = V * N(-d2)  -- Merton (1974)
where:
  V = asset value (from Ohlson)
  d2 = (ln(V/K) + (r - sigma^2/2)*T) / (sigma * sqrt(T))
  K = face value of debt
  sigma = asset volatility (from equity volatility via leverage ratio)
```

This gives a theoretically grounded debt estimate rather than an arbitrary ratio.

**Expected improvement:** Total assets error from 32% to ~15% (fundamental limit: goodwill from acquisitions is unobservable).

**Dependency:** `scipy.stats.norm` for N(d2)

---

## Method 4: Particle Filter with Nonlinear Balance Sheet Dynamics

**Current limitation:** The Kalman filter assumes linear Gaussian dynamics. Real earnings have fat tails (Fama & French 2000), regime-dependent volatility, and nonlinear dependence on macro conditions.

**Mathematical formulation:**

```
# State: theta_t = [E_t, p_t, g_t, sigma_t]
#   E_t = earnings, p_t = payout, g_t = growth rate, sigma_t = volatility

# Transition (nonlinear):
E_t = E_{t-1} * (1 + g_{t-1}) + sigma_t * w_t
p_t = p_{t-1} + eta_t                             -- random walk payout
g_t = mu_g + rho_g * (g_{t-1} - mu_g) + zeta_t   -- mean-reverting growth

# Observation:
D_t = p_t * E_t + v_t                              -- dividend
P_t = E_t * PE_t + u_t                             -- price (where PE = f(g, r, sigma))
V_t = volume_func(sigma_t) + noise                  -- volume
```

The particle filter (Sequential Monte Carlo) handles this fully nonlinear, non-Gaussian system. With 10,000 particles and 18 observations (annual dividends), it produces posterior distributions over all 4 latent states simultaneously.

**Expected improvement:** Net income error from 2.3% to ~1.5% by capturing the nonlinear dynamics (growth mean-reversion + volatility regime switching).

**Dependency:** `numpy` (SMC is simple to implement; or use the existing `operator1/models/particle_filter.py`)

---

## Method 5: Cross-Sectional Regression from SIX FQS Universe

**Current limitation:** All estimates use only the target company's data. The SIX FQS endpoint has 110K+ securities -- we can compute dividend yields for ALL SIX-listed dividend payers and use cross-sectional relationships to calibrate estimates.

**Mathematical formulation:**

```
# Step 1: Fetch dividend yield for all ~250 SIX equity issuers
# Step 2: Cross-sectional regression:
#   log(PE_i) = alpha + beta_1 * DivYield_i + beta_2 * DivGrowth_i 
#             + beta_3 * Illiquidity_i + eps_i
#
# Step 3: Use the fitted model to predict PE for the target company
# Step 4: This is a "comparable companies" approach with statistical rigor
```

The Fama-MacBeth (1973) two-pass regression ensures that the cross-sectional relationship is time-stable. The first pass estimates betas per period, the second pass averages betas across periods.

**Expected improvement:** PE error from 1.3% to <0.5% by using information from 250+ peer companies rather than just the target.

**Dependency:** `numpy` for OLS + the existing FQS search endpoint

---

## Method 6: Bayesian Hierarchical Model for Swiss Market

**Current limitation:** Sector ratios (net_margin, ROE, P/B, etc.) are point estimates. A Bayesian hierarchical model treats each company as drawn from a sector-level distribution, with the sector parameters themselves drawn from a market-level distribution. This provides optimally shrunk estimates.

**Mathematical formulation:**

```
# Market level (hyperprior):
mu_margin ~ Normal(0.12, 0.05)
sigma_margin ~ HalfNormal(0.03)

# Sector level:
margin_sector ~ Normal(mu_margin, sigma_margin)

# Company level:
margin_company ~ Normal(margin_sector, tau_company)

# Observation:
D_t / P_t ~ function(margin_company, payout, growth)
```

With PyMC, this is a ~15-line model that produces posterior distributions for ALL sector ratios simultaneously, automatically borrowing information from related companies.

**Expected improvement:** All sector-ratio-derived estimates improve by ~30% because the Bayesian model optimally shrinks company estimates toward sector means when data is sparse.

**Dependency:** `pymc` (installed in stage 3). Note: MCMC sampling takes 30-60 seconds per company. Best run once and cached.

---

## Method 7: Information-Theoretic Bounds (Entropy-Based Confidence)

**Current limitation:** Monte Carlo CI widths are based on assumed input distributions (Normal with guessed CVs). There's no theoretical guarantee these are calibrated.

**Mathematical formulation:**

The maximum entropy principle (Jaynes 1957) gives the least-informative distribution consistent with known constraints:

```
# Known: E[X] = mu, E[(X-mu)^2] = sigma^2
# MaxEnt solution: Normal(mu, sigma)     -- validates current assumption

# But if we also know: X >= 0 (earnings must be positive for dividend payers)
# MaxEnt solution: Truncated Normal or Log-Normal

# And if we know: X <= market_cap (earnings can't exceed market cap)
# MaxEnt solution: Beta-scaled distribution on [0, market_cap]
```

Using the correct MaxEnt distribution with all known constraints produces calibrated CIs that are neither too wide (wasteful) nor too narrow (overconfident).

The **Kullback-Leibler divergence** between the proxy distribution and the true distribution provides an upper bound on the expected error:

```
E[|proxy - true|] <= sqrt(2 * KL(proxy || true))  -- Pinsker's inequality
```

This lets us compute the theoretical minimum achievable error given our information set.

**Expected improvement:** More honest confidence intervals. May reveal that 2.3% error on NI is actually near-optimal given the information available (18 dividends + prices + capital actions).

**Dependency:** `scipy.stats` for distribution fitting

---

## Method 8: Edwards-Bell-Ohlson (EBO) Equity Valuation with Analyst Consensus

**Current limitation:** We have no forward-looking earnings estimates. But the market price ITSELF embeds forward expectations. The EBO model decomposes price into:

```
P = BV + (1-year RI) / (1+r) + (terminal RI) / (r * (1+r))
```

If we observe P, estimate BV (from Method 2), and know r (from CAPM or Fama-French), we can back out the market's implied forward earnings. This is the "reverse engineering" approach used by institutional equity research.

**Key insight:** The implied forward earnings from EBO + the historical Kalman earnings give two independent estimates. Their weighted average (inverse-variance weighted) is the minimum-variance estimator by the Gauss-Markov theorem.

**Expected improvement:** Combines backward-looking (Kalman from dividends) with forward-looking (market-implied from price) for an optimally weighted estimate. Could push NI error to <1%.

---

## Priority Ranking

| # | Method | Expected NI Error | Expected Equity Error | Complexity | Dependencies |
|---|--------|------------------|-----------------------|------------|-------------|
| 1 | Two-Factor Kalman (UKF) | 1.5% | -- | Medium | filterpy or numpy |
| 2 | Ohlson Residual Income | -- | <1% | Medium | numpy |
| 3 | Cross-Sectional Regression | <0.5% PE | -- | Low | numpy + FQS |
| 4 | EBO Reverse Engineering | <1% | -- | Low | numpy |
| 5 | Merton Structural Debt | -- | ~15% TA | Medium | scipy.stats |
| 6 | Particle Filter 4-State | 1.5% | -- | High | numpy |
| 7 | Bayesian Hierarchical | All improve ~30% | All improve ~30% | High | pymc (slow) |
| 8 | MaxEnt Confidence | No accuracy change | No accuracy change | Low | scipy.stats |

## Recommended Implementation Order

1. **EBO reverse-engineered earnings** (Method 8) -- simplest, highest impact on NI. Uses only price and dividend data we already have. Low complexity.
2. **Two-Factor UKF** (Method 1) -- fixes the fundamental Kalman limitation (single-state). Medium complexity.
3. **Cross-sectional regression from FQS** (Method 5) -- uses the SIX 110K security universe to calibrate PE. Low complexity but requires FQS batch query.
4. **Ohlson equity** (Method 2) -- company-specific book value from residual income model. Medium complexity.
5. **Merton debt** (Method 3) -- fixes the total assets error. Medium complexity.

Methods 6-7 (Bayesian, Particle) are high-complexity with diminishing returns given the information constraint. Method 8 (MaxEnt) is a diagnostic tool, not an accuracy improvement.

## Theoretical Lower Bound on Error

Given that we observe:
- 18 annual dividends (noisy observation of earnings * payout)
- ~108 daily prices (noisy observation of market's earnings expectation)  
- Capital structure (partial balance sheet information)

The Fisher information matrix for the earnings parameter gives:

```
Var(E_hat) >= 1 / I(E) = sigma_D^2 / (p^2 * T)
```

Where sigma_D is dividend noise, p is payout ratio, T is number of observations. For Nestle: sigma_D ~ 0.10 CHF, p ~ 0.73, T = 18:

```
Var(E_hat) >= 0.01 / (0.53 * 18) = 0.00105
std(E_hat) >= 0.032 CHF per share
```

With earnings ~4.13 CHF/share, the minimum achievable CV = 0.032 / 4.13 = **0.78%**. Our current 2.3% error is ~3x the Cramer-Rao bound. There is room for improvement, but below ~1% we are approaching the theoretical minimum.

## Checklist

- [ ] Implement EBO reverse-engineered earnings (Method 8)
- [ ] Implement Two-Factor UKF for earnings + payout (Method 1)
- [ ] Implement Cross-sectional FQS regression for PE (Method 5)
- [ ] Implement Ohlson residual income for equity (Method 2)
- [ ] Implement Merton structural debt model (Method 3)
- [ ] Compute Cramer-Rao bound and report theoretical minimum error
