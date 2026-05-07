# Per-Frequency Formulas for All 18 Broken Variables

For each broken variable, this specifies the exact formula to use at each frequency. The principle: at native frequencies (Q/A), use the standard textbook formula. At interpolated frequencies (D/W/M), either skip, adapt the formula, or pull the value from the nearest native-frequency pipeline.

---

## Variable 1: `pe_ratio_calc` (Price-to-Earnings)

**Standard formula:** PE = Price / EPS_TTM

| Freq | Formula | Method | Reference |
|------|---------|--------|-----------|
| **Q** | `close_q / (sum(eps_diluted, 4Q))` | TTM EPS from 4 quarterly EPS values | Graham & Dodd (1934) |
| **A** | `close_a / eps_diluted_annual` | Annual EPS directly from filing | Standard |
| **S** | `close_s / (eps_diluted_s * 2)` | Annualize semi-annual EPS | Standard |
| **M** | Forward-fill from Q | Not computed, taken from Q pipeline | N/A |
| **W** | Forward-fill from Q | Not computed | N/A |
| **D** | Forward-fill from Q | Not computed at daily; D pipeline skips PE | N/A |

**Implementation:** `derived_variables.py:412` -- when `freq in ("Q", "A", "S")`, compute PE using `eps_diluted` from the filing (annualized if Q or S). When `freq="D"`, skip.

**Expert note:** Shiller CAPE (cyclically-adjusted PE using 10-year average earnings, Shiller 2000) is a better long-term valuation anchor. Could add as `pe_cape` at A frequency: `close / mean(eps_annual, 10Y)`. Not implementing now but noting for future.

---

## Variable 2: `ev_to_ebitda` (Enterprise Value / EBITDA)

**Standard formula:** EV/EBITDA = (market_cap + total_debt - cash) / EBITDA_TTM

| Freq | Formula | Method |
|------|---------|--------|
| **Q** | `EV / sum(ebitda, 4Q)` | TTM EBITDA from 4 quarters |
| **A** | `EV / ebitda_annual` | Annual EBITDA directly |
| **S** | `EV / (ebitda_s * 2)` | Annualize |
| **D** | Forward-fill from Q | Skip |

**Expert note:** Damodaran (2012) recommends using EBITDA before stock compensation for tech companies. The `ebitda` field in the cache is `ebit` (proxy), not true EBITDA. At Q/A freq this is still correct because both ebit and D&A are at native scale.

---

## Variable 3: `ps_ratio_calc` (Price-to-Sales)

| Freq | Formula | Method |
|------|---------|--------|
| **Q** | `market_cap / sum(revenue, 4Q)` | TTM revenue |
| **A** | `market_cap / revenue_annual` | Annual revenue |
| **D** | Forward-fill from Q | Skip |

---

## Variable 4: `fcf_yield` (Free Cash Flow Yield)

**Standard formula:** FCF Yield = FCF_TTM / Market Cap

| Freq | Formula | Method |
|------|---------|--------|
| **Q** | `sum(fcf, 4Q) / market_cap` | TTM FCF from 4 quarters |
| **A** | `fcf_annual / market_cap` | Annual FCF |
| **D** | Forward-fill from Q | Skip |

**Expert note:** Greenblatt (2006, "The Little Book") uses earnings yield (EBIT/EV) instead of FCF yield for his "Magic Formula." Both are valid; FCF yield is preferred by activist investors (Icahn) because it strips out capex manipulation.

---

## Variable 5: `roa` (Return on Assets)

**Standard formula:** ROA = Net Income_TTM / Total Assets

| Freq | Formula | Method | Reference |
|------|---------|--------|-----------|
| **Q** | `sum(net_income, 4Q) / total_assets_latest` | TTM NI / latest B/S | DuPont analysis |
| **A** | `net_income_annual / total_assets_annual` | Both from same annual filing | Standard |
| **D** | Forward-fill from Q | Skip |

**Expert note:** Some practitioners use average total assets `(TA_begin + TA_end) / 2` (Palepu & Healy 2019). At Q frequency: `sum(NI, 4Q) / mean(TA_Q0, TA_Q4)`.

---

## Variable 6: `roe` (Return on Equity)

| Freq | Formula | Method |
|------|---------|--------|
| **Q** | `sum(net_income, 4Q) / total_equity_latest` | TTM NI / latest equity |
| **A** | `net_income_annual / total_equity_annual` | Standard |
| **D** | Forward-fill from Q | Skip |

**Expert note:** DuPont 5-factor decomposition (Palepu, Healy & Peek 2019): ROE = Tax Burden * Interest Burden * EBIT Margin * Asset Turnover * Equity Multiplier. Already partially implemented in HF engine (`hedge_fund/engine.py:145` CROA/ROIC spread). The decomposition works correctly at Q/A frequency because all components use native-frequency values.

---

## Variable 7: `eps_calc` (Earnings Per Share, computed)

| Freq | Formula | Method |
|------|---------|--------|
| **Q** | `net_income_q / shares_outstanding` | Quarterly NI / latest shares |
| **A** | `net_income_a / shares_outstanding` | Annual NI / shares |
| **D** | Use `eps_diluted` from filing (forward-filled) | Not recomputed |

**Expert note:** `eps_diluted` from CompanyFacts is the authoritative source. `eps_calc` should only be computed as a fallback when `eps_diluted` is missing.

---

## Variable 8: `revenue_per_share`

| Freq | Formula |
|------|---------|
| **Q** | `revenue_q / shares_outstanding` |
| **A** | `revenue_a / shares_outstanding` |
| **D** | Forward-fill from Q |

---

## Variable 9: `net_debt_to_ebitda`

| Freq | Formula |
|------|---------|
| **Q** | `net_debt / sum(ebitda, 4Q)` -- TTM EBITDA |
| **A** | `net_debt / ebitda_annual` |
| **D** | Forward-fill from Q |

**Expert note:** Moody's uses Net Debt / EBITDA with a 5.0x threshold for investment-grade boundary. Already used in HF leverage stress (`engine.py:350`) with 5.5x covenant threshold.

---

## Variable 10: `accruals` (Sloan Accruals Ratio)

**Standard formula:** Accruals = (Net Income - Operating Cash Flow) / Total Assets

| Freq | Formula | Reference |
|------|---------|-----------|
| **Q** | `(NI_q - OCF_q) / TA_q` | Sloan (1996, TAR) |
| **A** | `(NI_a - OCF_a) / TA_a` | Standard |
| **D** | Forward-fill from Q | Skip |

**Expert note:** Richardson et al. (2005) extended Sloan to decompose accruals into working capital, non-current operating, and financial components. The simple Sloan ratio is sufficient for the survival signal but the decomposition is useful for the HF forensics module (already in `accruals_forensics.py`).

---

## Variable 11: `dso` (Days Sales Outstanding)

**Standard formula:** DSO = Receivables / (Revenue / Days_in_Period)

| Freq | Formula | Method |
|------|---------|--------|
| **Q** | `receivables / (revenue_q / 90)` | 90 days in quarter |
| **A** | `receivables / (revenue_a / 365)` | 365 days in year |
| **S** | `receivables / (revenue_s / 180)` | 180 days |
| **D** | Forward-fill from Q | Skip |

**Expert note:** Richards & Laughlin (1980) Cash Conversion Cycle = DSO + DIO - DPO. All three components have the same frequency sensitivity. The CCC is a powerful working capital efficiency metric used by Warren Buffett (DSO decline = improving collections).

---

## Variable 12: `dio` (Days Inventory Outstanding)

| Freq | Formula |
|------|---------|
| **Q** | `inventory / (COGS_q / 90)` |
| **A** | `inventory / (COGS_a / 365)` |
| **D** | Forward-fill from Q |

---

## Variable 13: `dpo` (Days Payable Outstanding)

| Freq | Formula |
|------|---------|
| **Q** | `payables / (COGS_q / 90)` |
| **A** | `payables / (COGS_a / 365)` |
| **D** | Forward-fill from Q |

---

## Variable 14: `altman_x3` (EBIT / Total Assets)

**Standard formula:** x3 = EBIT / Total Assets (Altman Z-Score component)

| Freq | Formula | Reference |
|------|---------|-----------|
| **Q** | `sum(ebit, 4Q) / TA` | TTM EBIT for annualized ratio |
| **A** | `ebit_a / TA_a` | Altman (1968) original |
| **D** | Forward-fill from Q | Skip |

**Expert note:** Altman Z-Score has 5 components. x1 (WC/TA) and x4 (MVE/TL) use stock variables only -- they're fine at any frequency. x3 (EBIT/TA) and x5 (Revenue/TA) are FLOW/STOCK -- broken on daily. x2 (RE/TA) is STOCK/STOCK -- fine. So Altman Z on Q/A cache will be fully correct; on D cache, only x1/x2/x4 should be computed.

---

## Variable 15: `altman_x5` (Revenue / Total Assets)

| Freq | Formula |
|------|---------|
| **Q** | `sum(revenue, 4Q) / TA` |
| **A** | `revenue_a / TA_a` |
| **D** | Forward-fill from Q |

---

## Variable 16: `fh_runway_months` (Cash Runway)

**Standard formula:** Runway = Cash / Monthly Burn Rate = Cash / (|OCF| / months_in_period)

| Freq | Formula | Method |
|------|---------|--------|
| **Q** | `cash / (|OCF_q| / 3)` | 3 months in quarter |
| **A** | `cash / (|OCF_a| / 12)` | 12 months in year |
| **D** | Forward-fill from Q | Skip |

---

## Variable 17: `gross_margin` (Gross Profit / Revenue)

**Surprise finding:** This SHOULD be flow/flow (same scale), but the debug scan showed cache value of 0.775 vs Apple actual 0.469. Root cause: `gross_profit` and `revenue` were interpolated from DIFFERENT quarterly source values.

| Freq | Formula | Status |
|------|---------|--------|
| **Q** | `gross_profit_q / revenue_q` | Correct at Q (both from same filing) |
| **A** | `gross_profit_a / revenue_a` | Correct at A |
| **D** | `gross_profit_daily / revenue_daily` | ONLY correct if both from same interpolation source |

**Fix:** Compute gross_margin ONLY at Q/A frequency where both numerator and denominator come from the same filing. Forward-fill to D.

---

## Variable 18: TTM Values (revenue_ttm, net_income_ttm, ebitda_ttm)

| Freq | Formula | Method |
|------|---------|--------|
| **Q** | `sum(value, last 4 quarters)` | Rolling window over raw Q values |
| **A** | `= annual_value` | Annual IS the TTM |
| **S** | `sum(value, last 2 semesters)` | |
| **D** | Forward-fill from Q | Skip computation |

---

## Implementation Location Map

| Variable | Compute at Q/A in | Forward-fill to D in | Skip on D in |
|----------|-------------------|---------------------|-------------|
| pe_ratio_calc | `derived_variables.py:412` | `stage2_freq_pipeline.py:run_2_F_fusion` | `derived_variables.py:379` |
| ev_to_ebitda | `derived_variables.py:454` | same | same |
| ps_ratio_calc | `derived_variables.py:438` | same | same |
| fcf_yield | `derived_variables.py:295` | same | `derived_variables.py:269` |
| roa | `derived_variables.py:499` | same | `derived_variables.py:491` |
| roe | `derived_variables.py:339` | same | `derived_variables.py:309` |
| eps_calc | `derived_variables.py:585` | same | same |
| gross_margin | `derived_variables.py:327` | same | `derived_variables.py:309` |
| TTMs | `derived_variables.py:529-534` | same | `derived_variables.py:510` |
| accruals | `derived_variables.py:809` | same | skipped stage |
| dso/dio/dpo | `derived_variables.py:1245-1252` | same | skipped stage |
| altman_x3/x5 | `financial_health.py:658,664` | same | `financial_health.py:633` |
| fh_runway | `financial_health.py:850` | same | `financial_health.py:809` |
