# Fix SIX Proxy Formulas and Remove yfinance from ch_six.py

## Problem

The `ch_six.py` wrapper has 3 yfinance dependencies that violate PIT compliance:
1. `list_companies()` falls back to yfinance when FQS returns empty (line 338, 361)
2. `get_profile()` enriches with yfinance for sector/industry/market_cap (line 466-479)
3. `get_quotes()` delegates to yfinance via ohlcv_provider (line 549-557)

The `six_derived_proxies.py` has formulas that reference OHLCV `close` and `volume` columns -- which currently come from yfinance. Without yfinance, these columns won't exist unless we use the SIX historic CSV.

## Available PIT Data from SIX APIs (no yfinance)

| Source | Data | PIT? |
|--------|------|------|
| FQS ref.json | ValorId, ISIN, ticker, name (110K securities) | Yes |
| share/info.json | shares_outstanding, nominal_value, indices, regulatory_standard, first_trading_date, issuedBy | Yes |
| share/dividend.json | 18+ years ex-dividend dates and amounts | Yes |
| issuer/financial_reports.json | auditor, accounting_standard, annual_closing_date | Yes |
| issuer/capital_structure.json | share capital, conditional capital | Yes |
| overview/info.json | website, issuer_code, next_gm_date | Yes |
| official_notices find/details | 349K corporate actions with PIT dates, notice text | Yes |
| management_transactions overview.json | insider buy/sell with amounts in CHF | Yes |
| market_data/v1/{ValorId}/historic.csv | ~5 months close+volume (no O/H/L) | Yes (exchange data) |

## Plan

### 1. Remove yfinance from ch_six.py

**`list_companies()`** -- Remove yfinance fallback. FQS covers 110K securities. If FQS is down, return empty list (graceful degradation, same as other PIT clients).

**`get_profile()`** -- Remove yfinance enrichment step (Step 7). For sector/industry, we have two PIT-compliant alternatives:
- Use the `regulatoryStandard` and `indexSymbols` from share/info.json to infer sector. SMI index membership + regulatory standard gives a reasonable sector classification.
- Use the `supplement.py` enrichment (OpenFIGI) which is already wired in main.py after profile fetch -- OpenFIGI returns sector/industry from FIGI data.
- For market_cap: compute from `shares_outstanding * latest_close` where latest_close comes from the SIX historic CSV.

**`get_quotes()`** -- Implement SIX historic CSV fetcher. The endpoint `/sheldon/market_data/v1/{ValorId}/historic.csv` returns ~5 months of close+volume. This is insufficient for the full 2-year pipeline window, BUT:
- The pipeline already handles short OHLCV gracefully (creates empty cache with business day index, forward-fills statements)
- 5 months of close+volume is enough for: Amihud illiquidity, short-term volatility, recent drawdown
- The SIX proxy module fills the analytical gaps for the periods without price data
- Mark the data with `ohlcv_source = "SIX historic CSV (5mo)"` so the report generator knows

### 2. Fix six_derived_proxies.py Formulas

The proxy formulas are mathematically sound but have wiring issues and can be improved:

**Fix A: Handle missing close/volume gracefully**
Currently `compute_six_proxies()` bails out if `close` is None. With SIX CSV, we'll have ~5 months. The formulas should:
- Compute Amihud/volatility from whatever close data exists (even 5 months)
- For periods without close data, use the proxy-only path (dividend-based metrics still work without any price data since they come from SIX dividend API)

**Fix B: Decouple dividend proxies from price data**
The strongest proxies (dividend yield, CAGR, stability, implied earnings) only need dividends + shares_outstanding + latest_close. Even if close data is short, these are computed from the dividend history API (18 years) and a single price point. Restructure so these compute first and independently.

**Fix C: Add nominal value-based book equity proxy**
SIX provides `nominalValue` and `shares_outstanding` from share/info.json. `nominal_value * shares_outstanding = minimum book equity` (share capital). Combined with `reported_share_capital` from capital_structure.json, this gives a PIT-compliant book value floor. Use this as:
- `book_equity_floor = reported_share_capital` (from capital_structure)
- `price_to_book_floor = market_cap / book_equity_floor` (upper bound on P/B since book equity >= share capital)

**Fix D: Add capital structure solvency proxy**
From capital_structure.json, SIX provides `REPORTED_CAPITAL` and `CONDITIONAL_CAPITAL`. The ratio `conditional_capital / reported_capital` indicates dilution risk. High conditional capital relative to reported capital means potential share dilution (warrants, convertibles). This is a direct solvency signal.

**Fix E: Compute market cap from SIX data only**
`market_cap = shares_outstanding * latest_close_from_six_csv`. If no SIX CSV data, use `shares_outstanding * nominal_value` as an absolute floor (par value, not market value -- flag as "par_value_floor" in metadata).

**Fix F: Add dividend coverage ratio**
A new formula using the implied earnings:
`dividend_coverage = implied_after_tax_earnings / total_dividends`
This tells you how many times the company can cover its dividend from estimated earnings. Values < 1.0 suggest unsustainable dividends. This is a direct solvency/liquidity signal that feeds Tier 1 and Tier 2.

**Fix G: Confidence tagging**
Every proxy column should have a companion `_confidence` column (0-1) indicating how reliable the estimate is:
- Dividend yield from actual dividends + actual close: confidence 0.95
- Implied PE from dividends + buybacks: confidence 0.70 (depends on payout assumption)
- Estimated OCF from adaptive payout ratio: confidence 0.50 (sector average)
- Tier scores from calibrated benchmarks: confidence 0.60

### 3. Wire compute_six_proxies into main.py

Currently `compute_six_proxies` is not called from main.py. It needs to be wired in Step 5 (feature engineering) after `compute_derived_variables()`, gated on `market_id == "ch_six"`.

### 4. Update plans/six-proxy-formula-accuracy-improvements.md

Update the plan doc to reflect the implemented changes and remove references to yfinance.

## Implementation Checklist

- [ ] **ch_six.py**: Remove yfinance imports and fallbacks from `list_companies()`
- [ ] **ch_six.py**: Remove yfinance enrichment from `get_profile()` Step 7
- [ ] **ch_six.py**: Implement `get_quotes()` using SIX historic CSV endpoint
- [ ] **ch_six.py**: Add `_fetch_historic_csv()` helper for SIX market data
- [ ] **ch_six.py**: Compute market_cap from shares_outstanding * latest_close in profile
- [ ] **six_derived_proxies.py**: Decouple dividend proxies from price data (Fix B)
- [ ] **six_derived_proxies.py**: Add book equity floor proxy from nominal_value + capital_structure (Fix C)
- [ ] **six_derived_proxies.py**: Add dilution risk from conditional/reported capital ratio (Fix D)
- [ ] **six_derived_proxies.py**: Add dividend coverage ratio (Fix F)
- [ ] **six_derived_proxies.py**: Add confidence tagging to all proxy columns (Fix G)
- [ ] **six_derived_proxies.py**: Handle short/missing close data gracefully (Fix A)
- [ ] **main.py**: Wire `compute_six_proxies()` call in Step 5 for ch_six market
- [ ] **plans/six-proxy-formula-accuracy-improvements.md**: Update to reflect changes
