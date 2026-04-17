# AAPL Backtest Validation: 2024 Data Predicting 2025

**Run date:** 2026-04-17
**Market:** US SEC EDGAR (`us_sec_edgar`)
**Company:** Apple Inc. (AAPL, CIK 320193)
**Backtest window:** 2023-01-01 to 2024-12-31 (2 years)
**Prediction target:** 2025 actuals

---

## Pipeline Execution Summary

| Stage | Duration | Details |
|-------|----------|---------|
| Stage 1 (Data + Features) | 120.4s | 502 rows x 318 cols, SEC EDGAR XBRL + yfinance OHLCV |
| Stage 2 (Temporal Models) | ~4min | 5-frequency MF pipeline (A/Q/M/W/D), 25+ models |
| Stage 3 (Profile + Report) | 74.6s | 3-tier reports (Basic/Pro/Premium) + PDF + tearsheet + interactive dashboard |
| **Total** | **~6min** | |

## Data Sources Used

- **PIT financials:** SEC EDGAR XBRL (edgartools) -- income, balance, cashflow with filing_date + report_date
- **OHLCV:** yfinance (Yahoo Finance)
- **Macro:** FRED API (GDP, inflation, interest rate, unemployment)
- **Holders:** SEC EDGAR SC 13D/13G + DEF 14A proxy (5 holders, 15.17% ownership)
- **Segments:** XBRL OperatingSegmentsMember (18 segments, HHI=0.078)
- **News:** gnews + VADER sentiment scoring
- **LLM:** OpenRouter (nvidia/nemotron-3-super-120b-a12b:free) for entity discovery

## Prediction vs Actual Results

| Horizon | Predicted Close | Actual Close | Error ($) | Error (%) |
|---------|----------------|--------------|-----------|-----------|
| 1 day | $250.11 | $242.53 | $7.59 | **3.13%** |
| 5 day | $249.39 | $241.38 | $8.00 | **3.32%** |
| 21 day | $248.74 | $226.77 | $21.97 | **9.69%** |

## Actual 2025 Market Outcome

| Metric | Value |
|--------|-------|
| Actual total return (2025 full year) | **+8.95%** |
| Actual max drawdown | **-30.22%** |
| Actual 21d volatility | 0.0200 |
| Year-end close (2025-12-30) | $272.82 |

## Model Results

### Financial Health
- Composite score: computed across 5 tiers (Liquidity, Solvency, Stability, Profitability, Growth)
- Ethical filters: Purchasing Power=PASS, Solvency=PASS, Gharar=LOW

### Survival Analysis
- Survival mode: 26/502 days flagged (5.18% company_only)
- Enriched timeline: mean intensity=0.265, states={elevated_risk: 93.4%, company_distress_severe: 5.2%, stable_growth: 1.4%}
- Regime: high_vol dominant (495/502 days)

### Hedge Fund Analysis
- Grade: C, Conviction: 4/10
- Signal: -1.00 (strong_sell)
- Piotroski F-Score: 2/9
- Altman Z'': -2.28 (distress zone)
- DCF intrinsic value: $33.05
- Fusion signal: -0.32 (sell), conviction 49%

### Multi-Frequency Fusion
- 5 frequencies processed (A/Q/M/W/D)
- Regime consensus: high_vol (100% agreement)
- Fused survival probability: 80.0%
- 11 fusion methods applied, 276 predictions generated

### Model Diagnostics
- 10 models assessed, 7 on track, 0 deviated
- Overall robustness: high

## Key Observations

1. **Short-term accuracy was strong** -- 1d and 5d predictions were within 3.3% of actuals
2. **21d prediction missed the January 2025 selloff** -- AAPL dropped to $226.77 (a 9.7% miss), driven by the broader tech correction in early 2025
3. **Max drawdown was severe** -- -30.22% actual drawdown in 2025, which the survival analysis partially flagged (5.2% of days in company_distress_severe)
4. **HF strong_sell signal was premature** -- while AAPL did experience significant drawdown, it recovered to +8.95% YTD by year end
5. **Multi-frequency fusion provided robust regime consensus** -- all 5 frequencies agreed on high_vol regime

## Generated Artifacts

- `company_profile.json` -- full 25-section profile
- `predictions_summary.json` -- per-variable per-horizon forecasts
- `validation_results.json` -- prediction vs actual comparison
- `report/basic_report.md` -- 5-section screening report
- `report/pro_report.md` -- 18-section professional report with 7 embedded charts
- `report/premium_report.md` -- 27-section institutional report with full model details
- `report/report.pdf` -- branded PDF (fpdf2)
- `report/tearsheet.html` -- quantstats performance tearsheet
- `report/interactive_dashboard.html` -- plotly zoomable dashboard
