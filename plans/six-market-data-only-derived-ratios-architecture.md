# SIX Market-Data-Only Derived Ratios Architecture

## Available Data from SIX APIs (No Financial Statements)

| Source | Data | Update Frequency |
|--------|------|-----------------|
| FQS ref.json | shares_outstanding, nominal_value, ValorId, ISIN | Static (changes at capital events) |
| share/info.json | numberInIssue, index memberships, regulatory standard | Static |
| share/dividend.json | 18 years of ex-dividend dates + CHF amounts per share | Annual |
| issuer/capital_structure.json | share capital, conditional capital | Static |
| official_notices | capital destruction dates, shares destroyed per event | Event-driven |
| management_transactions | insider buy/sell with amounts, dates, ISIN | Daily |
| OHLCV (yfinance) | close, open, high, low, volume (2 years daily) | Daily |
| Macro (FRED/wbgapi) | GDP, inflation, interest rates, unemployment, FX | Quarterly/Annual |

## Creative Derivation Methods

### Tier 1: Liquidity and Cash (from dividends + price)

**1. Dividend Payout Capacity Proxy**
- `total_annual_dividend = dividend_per_share * shares_outstanding`
- For Nestle: CHF 3.10 * 2.58B = CHF 7.99B annual dividend outflow
- A company paying CHF 8B in dividends must have at least that much in operating cash flow
- `min_operating_cf_estimate = total_annual_dividend / payout_ratio_assumption`
- Conservative assumption: payout_ratio = 0.60 (typical for Swiss blue chips)
- `estimated_operating_cf = 7.99B / 0.60 = CHF 13.3B`

**2. Dividend Coverage Ratio (inverse payout)**
- `dividend_coverage = 1 / estimated_payout_ratio`
- If dividend has grown consistently for 18 years, payout ratio is likely stable
- `dividend_growth_rate = (latest_dividend / earliest_dividend)^(1/n_years) - 1`
- Nestle: (3.10 / 1.60)^(1/18) - 1 = 3.7% CAGR
- Consistent dividend growth implies earnings growth at similar or higher rate

**3. Liquidity Stress Indicator (from volume + price)**
- `amihud_illiquidity = abs(return_1d) / (volume * close)` -- Amihud measure
- Daily illiquidity ratio: high values = low liquidity, potential stress
- `liquidity_score = 100 * (1 - percentile_rank(amihud_illiquidity))`

### Tier 2: Solvency (from capital actions + dividends)

**4. Capital Return Yield (buybacks + dividends)**
- From SIX official notices: shares destroyed per year
- `buyback_yield = (shares_destroyed * avg_price_at_destruction) / market_cap`
- `total_shareholder_return = dividend_yield + buyback_yield`
- This is a proxy for excess cash generation capacity

**5. Nominal Capital Ratio**
- `nominal_capital_ratio = share_capital / market_cap`
- Nestle: 257.6M / 199.2B = 0.0013 (0.13%)
- Low ratio = heavily appreciated equity (good sign)
- Declining over time = share buybacks reducing capital

**6. Capital Action Frequency (survival signal)**
- Count capital destruction events over 2 years from notices
- Companies doing regular buybacks are signaling financial strength
- `capital_action_confidence = n_buyback_events / expected_events`

### Tier 3: Market Stability (from OHLCV -- already works)

These already work from the OHLCV data pipeline:
- `return_1d`, `log_return_1d`, `volatility_21d`, `drawdown_252d`
- `return_5d`, `return_21d` (multi-horizon returns)
- Volume-based metrics

**7. Insider Confidence Score (from management_transactions)**
- `insider_buy_ratio = insider_buys / (insider_buys + insider_sells)`
- `insider_net_flow = sum(buy_amounts) - sum(sell_amounts)`
- `insider_conviction = insider_buy_ratio * log(1 + abs(insider_net_flow))`
- Insiders buying = confidence signal; selling = caution

**8. Index Membership Stability Score**
- SMI membership = top 20 Swiss companies by market cap
- SLI membership = top 30 by liquidity
- `index_stability = 1.0 if SMI member, 0.7 if SLI only, 0.3 otherwise`
- This is a proxy for market standing and institutional support

### Tier 4: Profitability Proxies (from dividend trajectory)

**9. Earnings Growth Proxy (from dividend growth)**
- Sustainable dividend growth requires earnings growth
- `implied_earnings_growth = dividend_cagr * (1 + dividend_cagr)`
- Companies that cut dividends are in earnings trouble
- `dividend_streak_score = consecutive_years_of_growth / total_years`

**10. Profitability Consistency (from dividend variance)**
- `dividend_stability = 1 - (std(dividend_growth_rates) / mean(dividend_growth_rates))`
- Low variance in dividend growth = consistent profitability
- A company with 18 years of growing dividends has stable margins

### Tier 5: Valuation (from price + dividends)

**11. Dividend Yield (direct)**
- `dividend_yield = annual_dividend / close`
- Historical percentile ranking against own history
- `yield_zscore = (current_yield - mean_yield) / std_yield`
- High yield relative to history = potentially undervalued (or in trouble)

**12. Price-to-Dividend Growth (PDG) Ratio**
- `pdg_ratio = (close / dividend_per_share) / dividend_cagr`
- Analogous to PEG ratio but using dividends instead of earnings
- Lower = cheaper relative to dividend growth capacity

**13. Gordon Growth Model Implied Return**
- `implied_return = dividend_yield + dividend_cagr`
- Nestle: 4.01% + 3.7% = 7.7% implied total return
- Compare to risk-free rate (Swiss 10Y bond) for equity risk premium

## Survival Mode Adaptation

The standard survival triggers need adaptation for SIX:

| Standard Trigger | SIX Proxy |
|-----------------|-----------|
| `current_ratio < 1.0` | `dividend_cut_flag` (dividend decreased YoY) |
| `debt_to_equity > 3.0` | `capital_action_stress` (no buybacks + dividend cut) |
| `fcf_yield < 0` | `total_shareholder_return < 0` (negative buyback+dividend yield) |
| `drawdown_252d < -0.40` | Same (from OHLCV -- already works) |

## Implementation Approach

1. Create `operator1/features/six_derived_proxies.py` -- new module
2. Compute the 13 proxy ratios above from SIX data + OHLCV
3. Map the proxies to the existing tier structure for financial health scoring
4. Inject into the daily cache as `six_proxy_*` columns
5. Modify `survival_mode.py` to accept proxy triggers when standard ratios are NaN
6. Modify `financial_health.py` to use proxy scores when standard scores are NaN

## What This Achieves

| Pipeline Module | Before (empty) | After (proxies) |
|----------------|----------------|-----------------|
| Survival mode detection | Always "normal" (NaN triggers) | Dividend-based stress detection |
| Financial health Tier 1 (liquidity) | NaN | Amihud liquidity + dividend capacity |
| Financial health Tier 2 (solvency) | NaN | Capital return yield + nominal ratio |
| Financial health Tier 3 (stability) | Already works | Same + insider confidence |
| Financial health Tier 4 (profitability) | NaN | Dividend growth consistency |
| Financial health Tier 5 (valuation) | NaN | Yield + PDG + Gordon model |
| Monte Carlo survival | No thresholds to test | Dividend-based thresholds |
| Ethical filters | Partial (only purchasing power) | + solvency proxy + cash quality proxy |

The key insight: **18 years of dividend history IS financial data** -- it reveals earnings capacity, cash generation, solvency, and management discipline, all without a single line of a balance sheet.
