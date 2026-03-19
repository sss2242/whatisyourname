# SIX Proxy v4: Seed Canonical Columns for Estimator

## Problem

The estimator (`estimator.py`) only processes variables that exist as columns in the cache (line 755). For SIX, financial statement columns like `revenue`, `net_income`, `total_assets`, `total_equity`, `operating_cash_flow` don't exist because `get_income_statement()` etc. return empty DataFrames.

The current `six_derived_proxies.py` injects `six_proxy_*` prefixed columns which the estimator ignores entirely. The proxy module does useful math but its output never reaches the estimation engine.

## Solution

Add a new function `seed_canonical_columns()` in `six_derived_proxies.py` that:
1. Uses the advanced numerical methods to compute best estimates of canonical variables
2. Injects them directly into the cache under canonical names (`net_income`, `total_equity`, etc.)
3. Tags them with `_source = "six_proxy"` so the estimator knows they're estimates, not observations
4. The estimator's Phase 1 accounting identities then cascade-fill the rest

## What We Can Derive

From the SIX data we have, using numerical methods:

### Tier A: Direct computation (high confidence)
```
total_equity >= share_capital                    # from capital_structure (floor)
market_cap = shares * close                       # from SIX CSV
```

### Tier B: Kalman-filtered earnings (medium-high confidence)
```
net_income = KalmanFilter(dividends, buybacks)   # 18yr series -> optimal estimate
operating_cash_flow = net_income / (1 - tax) + depreciation_estimate
```

Using the Kalman filter state-space model:
- State: `E_t` (latent earnings)
- Observation 1: `D_t = payout * E_t + noise` (dividends observed 18yr)
- Observation 2: `B_t = buyback_frac * E_t + noise` (buybacks from notices)

The Kalman filter estimates `E_t` optimally and gives `P_t` (variance), which maps directly to the estimator's confidence system.

### Tier C: Constrained reconstruction (medium confidence)
Using scipy.optimize with accounting identity constraints:
```
total_assets = total_liabilities + total_equity    # identity
total_equity = share_capital + retained_earnings   # identity
retained_earnings >= cumulative_net_income * (1 - payout_ratio) # bound
total_debt <= 3 * total_equity                      # typical constraint
cash >= annual_dividends                            # must have cash to pay
current_assets >= cash                              # identity
```

### Tier D: PELT regime-aware projections (medium confidence)
```
revenue = net_income / net_margin                   # need margin estimate
```
Use PELT on the dividend growth series to detect the current regime. Companies in stable regimes have predictable margins (Consumer Defensive ~12-15% net margin). Use the regime + sector to estimate net_margin, then derive revenue.

## Implementation

### Step 1: Add `_kalman_earnings_estimate()` to six_derived_proxies.py

```python
def _kalman_earnings_estimate(dividends, buyback_values, payout_prior):
    from statsmodels.tsa.statespace.mlemodel import MLEModel
    
    # Or simpler: use UnobservedComponents with level + irregular
    from statsmodels.tsa.statespace.structural import UnobservedComponents
    
    # Total cash returned = dividends + buybacks per year
    total_returned = dividends + buyback_values  # annual series
    
    # Earnings = total_returned / payout_ratio
    # Model total_returned as noisy observation of latent earnings * payout
    model = UnobservedComponents(total_returned, level='local level')
    result = model.fit(disp=False)
    
    # Smoothed state = optimal estimate of latent level
    smoothed_earnings = result.smoothed_state[0] / payout_prior
    smoothed_variance = result.smoothed_state_cov[0, 0] / (payout_prior ** 2)
    
    return smoothed_earnings, smoothed_variance
```

### Step 2: Add `_reconstruct_balance_sheet()` using scipy.optimize

```python
def _reconstruct_balance_sheet(known_values):
    from scipy.optimize import linprog
    
    # Variables: [total_assets, total_liabilities, total_equity,
    #             retained_earnings, total_debt, cash, current_assets,
    #             current_liabilities]
    
    # Minimize sum of variables (L1 norm = sparsest solution)
    # Subject to accounting identities as equality constraints
    # And domain-specific inequalities as bounds
```

### Step 3: Add `_pelt_regime_margin()` using ruptures

```python
def _pelt_regime_margin(dividends, sector):
    import ruptures
    
    growth = dividends.pct_change().dropna().values
    algo = ruptures.Pelt(model="rbf").fit(growth.reshape(-1, 1))
    breakpoints = algo.predict(pen=2)
    
    # Use only the latest regime
    latest_start = breakpoints[-2] if len(breakpoints) > 1 else 0
    recent_growth = growth[latest_start:]
    
    # Map sector to net margin range
    sector_margins = {
        'Consumer Defensive': (0.10, 0.15),
        'Healthcare': (0.15, 0.25),
        'Financial Services': (0.20, 0.30),
        'Technology': (0.12, 0.20),
        'Industrials': (0.06, 0.10),
    }
    lo, hi = sector_margins.get(sector, (0.08, 0.15))
    
    # Stable dividend growth -> higher margin estimate
    stability = 1 - np.std(recent_growth) / (np.abs(np.mean(recent_growth)) + 1e-10)
    margin = lo + (hi - lo) * max(0, min(1, stability))
    
    return margin
```

### Step 4: Add `seed_canonical_columns()` -- the bridge function

```python
def seed_canonical_columns(cache, profile, proxy_result):
    """Inject estimated canonical columns so the estimator can process them.
    
    Called BEFORE run_estimation() in main.py.
    """
    # net_income (from Kalman or implied earnings)
    if proxy_result.implied_earnings and 'net_income' not in cache.columns:
        cache['net_income'] = proxy_result.implied_earnings
    
    # total_equity (from capital structure floor)
    if proxy_result.book_equity_floor and 'total_equity' not in cache.columns:
        cache['total_equity'] = proxy_result.book_equity_floor
    
    # operating_cash_flow (from estimated OCF)
    if 'six_proxy_est_operating_cf' in cache.columns and 'operating_cash_flow' not in cache.columns:
        cache['operating_cash_flow'] = cache['six_proxy_est_operating_cf']
    
    # cash_and_equivalents (>= annual dividends, from dividend data)
    if proxy_result.dividend_yield and 'cash_and_equivalents' not in cache.columns:
        shares = profile.get('shares_outstanding', 0)
        div = profile.get('latest_dividend_amount', 0)
        if shares and div:
            cache['cash_and_equivalents'] = float(shares) * float(div)
    
    # The estimator's Phase 1 will then cascade:
    # total_assets = total_liabilities + total_equity
    # net_debt = total_debt - cash
    # free_cash_flow = operating_cash_flow - capex
```

### Step 5: Wire in main.py

In main.py Step 4b (before `run_estimation()`):
```python
if market_id == 'ch_six' and six_proxy_result and six_proxy_result.computed:
    from operator1.features.six_derived_proxies import seed_canonical_columns
    seed_canonical_columns(cache, target_profile, six_proxy_result)
    logger.info("SIX canonical columns seeded for estimator")
```

## Checklist

- [ ] Add `_kalman_earnings_estimate()` using statsmodels UnobservedComponents
- [ ] Add `_pelt_regime_margin()` using ruptures for sector-aware margin
- [ ] Add `_reconstruct_balance_sheet()` using scipy.optimize.linprog
- [ ] Add `seed_canonical_columns()` bridge function
- [ ] Wire `seed_canonical_columns()` in main.py before run_estimation()
- [ ] Test: verify estimator Phase 1 cascades from seeded SIX values
