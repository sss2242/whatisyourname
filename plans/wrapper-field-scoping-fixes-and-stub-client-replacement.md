# Wrapper Field Scoping Fixes and Stub Client Replacement (v2)

## Context

Detailed audit of each stub client reveals three categories of completeness and two cross-field bugs to fix.

---

## Fix 1: Remove OHLCV from IN and CN PIT clients (Bugs 12-13)

### 1a. `in_bse.py` -- remove get_quotes OHLCV delegation
BSE API does not provide OHLCV. Currently delegates to `ohlcv_provider.fetch_ohlcv()`. Change to return empty. The pipeline fallback in `main.py:675` already calls the OHLCV provider when quotes are empty.

### 1b. `cn_sse.py` -- remove get_quotes OHLCV delegation
SSE does not provide OHLCV. Currently delegates to `ohlcv_baostock`. Change to return empty. Same pipeline fallback handles it.

---

## Fix 2: Add yfinance-backed financials to stub clients

Since none of these 9 exchanges have free public financial statement APIs, we use yfinance as the data source for financial statements. yfinance provides `Ticker.income_stmt`, `Ticker.balance_sheet`, and `Ticker.cashflow` for global stocks.

### Shared helper: `_yfinance_financials_mixin`

Create a reusable mixin or helper function in a new file `operator1/clients/yfinance_backed.py` that provides:

```python
def _yf_get_profile(ticker_symbol: str, market_id: str, country: str, exchange: str, currency: str) -> dict
def _yf_get_financials(ticker_symbol: str, market_id: str, statement_type: str) -> pd.DataFrame
def _yf_search(query: str, exchange_suffix: str) -> list[dict]
```

Each stub client imports and calls these helpers, keeping its own market-specific config (country, currency, exchange suffix).

### Per-client changes (ONE BY ONE):

#### Client 1: `ca_sedar.py` (Canada)
- **Profile**: Already works via SEDAR+ API -- keep as-is
- **Search**: Already works via SEDAR+ API -- keep as-is
- **Financials**: Add yfinance with `.TO` suffix for TSX tickers
- **OHLCV**: Already empty -- correct

#### Client 2: `au_asx.py` (Australia)
- **Profile**: Already works via ASX API -- keep as-is
- **Search**: Already works via ASX API -- keep as-is
- **Financials**: Add yfinance with `.AX` suffix
- **OHLCV**: Already empty -- correct

#### Client 3: `hk_hkex.py` (Hong Kong)
- **Profile**: Skeleton only -- add yfinance `.HK` profile enrichment
- **Search**: Returns empty -- add yfinance search
- **Financials**: Add yfinance with `.HK` suffix
- **OHLCV**: Already empty -- correct

#### Client 4: `sg_sgx.py` (Singapore)
- **Profile**: Skeleton only -- add yfinance `.SI` profile enrichment
- **Search**: Returns empty -- add yfinance search
- **Financials**: Add yfinance with `.SI` suffix
- **OHLCV**: Already empty -- correct

#### Client 5: `mx_bmv.py` (Mexico)
- **Profile**: Skeleton only -- add yfinance `.MX` profile enrichment
- **Search**: Returns empty -- add yfinance search
- **Financials**: Add yfinance with `.MX` suffix
- **OHLCV**: Already empty -- correct

#### Client 6: `za_jse.py` (South Africa)
- **Profile**: Skeleton only -- add yfinance `.JO` profile enrichment
- **Search**: Returns empty -- add yfinance search
- **Financials**: Add yfinance with `.JO` suffix
- **OHLCV**: Already empty -- correct

#### Client 7: `ch_six.py` (Switzerland)
- **Profile**: Skeleton only -- add yfinance `.SW` profile enrichment
- **Search**: Returns empty -- add yfinance search
- **Financials**: Add yfinance with `.SW` suffix
- **OHLCV**: Already empty -- correct

#### Client 8: `sa_tadawul.py` (Saudi Arabia)
- **Profile**: Skeleton only -- add yfinance `.SR` profile enrichment
- **Search**: Returns empty -- add yfinance search
- **Financials**: Add yfinance with `.SR` suffix
- **OHLCV**: Already empty -- correct

#### Client 9: `ae_dfm.py` (UAE)
- **Profile**: Skeleton only -- add yfinance profile enrichment (UAE tickers use various suffixes)
- **Search**: Returns empty -- add yfinance search
- **Financials**: Add yfinance financials
- **OHLCV**: Already empty -- correct

---

## Fix 3: Add OHLCV suffix mappings

Verify `ohlcv_yfinance.py` has correct suffix mappings for all 9 markets so the OHLCV fallback works:

| market_id | yfinance suffix |
|-----------|-----------------|
| ca_sedar | .TO |
| au_asx | .AX |
| hk_hkex | .HK |
| sg_sgx | .SI |
| mx_bmv | .MX |
| za_jse | .JO |
| ch_six | .SW |
| sa_tadawul | .SR |
| ae_dfm | (varies) |

---

## Fix 4: Add macro provider entries

Add FRED or wbgapi primary entries for markets missing from `macro_provider.py`:

```python
"CA": "fred",  # Canada
"AU": "fred",  # Australia  
"HK": "fred",  # Hong Kong
"SG": "fred",  # Singapore
"ZA": "fred",  # South Africa
"CN": "fred",  # China
"IN": "fred",  # India
```

SA, AE, CH fall through to wbgapi (acceptable, World Bank covers these).

---

## Implementation Order

1. Create `yfinance_backed.py` with shared helper functions
2. Fix `in_bse.py` -- remove OHLCV from get_quotes
3. Fix `cn_sse.py` -- remove OHLCV from get_quotes
4. Update `ca_sedar.py` -- add financials (keep existing profile/search)
5. Update `au_asx.py` -- add financials (keep existing profile/search)
6. Update `hk_hkex.py` -- add profile, search, financials
7. Update `sg_sgx.py` -- add profile, search, financials
8. Update `mx_bmv.py` -- add profile, search, financials
9. Update `za_jse.py` -- add profile, search, financials
10. Update `ch_six.py` -- add profile, search, financials
11. Update `sa_tadawul.py` -- add profile, search, financials
12. Update `ae_dfm.py` -- add profile, search, financials
13. Verify OHLCV suffix mappings in `ohlcv_yfinance.py`
14. Add macro provider entries
15. Test each client individually

## Verification per client

After each client update, run:
1. The existing PIT test for that market (if one exists)
2. A quick import + instantiation check
3. Verify `get_quotes()` returns empty (OHLCV handled by provider)
4. Verify financials return non-empty DataFrames for a known ticker
