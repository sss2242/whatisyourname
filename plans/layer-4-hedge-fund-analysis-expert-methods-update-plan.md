# Layer 4: Hedge Fund Analysis -- Expert Methods Update Plan

*Researched 2026-05-01 -- domain expert methods for enhancing the 20 Layer 4 hedge fund modules*

Layer 4 is the parallel investment analysis track. It consumes raw quarterly statements (8-24 rows) and produces ~107 result fields answering "Can I make money on this, when, and how much?" All outputs are result objects in `profile["hedge_fund"]` -- no cache columns.

---

## Current State Summary

| Tier | Modules | Current Focus | Enhancement Opportunity |
|------|---------|--------------|------------------------|
| **Tier 1: Earnings Quality** (4.1-4.3) | FCF Quality, Accruals Forensics, Earnings Smoothing | Static scoring composites | Temporal degradation tracking, peer-relative forensics |
| **Tier 2: Cash Flow** (4.4-4.6) | Dividend Burn, CROA/ROIC, Operating Leverage | Point-in-time metrics | DuPont decomposition, cash flow duration |
| **Tier 3: Balance Sheet** (4.7-4.9) | OBS Risk, Asset Quality, Leverage Stress | Scenario-based | Hidden leverage detection, distress distance |
| **Tier 4: Inflection** (4.10-4.12) | Momentum, Growth Quality, Earnings Surprise | Backward-looking | Leading indicator composites, regime-conditional momentum |
| **Tier 5: Valuation** (4.13-4.15) | DCF, Val-Quality, PEG | Standard DCF | Residual income, real options, market-implied growth |
| **Synthesis** (4.16-4.20) | Scorecard, Position Signal, Advanced Methods, Fusion | Weighted composites | Kelly criterion sizing, Bayesian conviction updating |

---

## Tier 1 Enhancements: Earnings Quality

### Enhancement 4.1A: FCF Quality Temporal Degradation Tracker

**Current:** Single-point score (0-100). `degradation_flag` is binary (declining over 4Q).

**Expert method:** Track FCF quality score as a **time series** over 8-12 quarters. Fit a linear trend and compute the **rate of change** and **quarters until quality drops below 50**. This is analogous to the "cash runway" concept but for earnings quality.

**Formula:** `quality_runway_quarters = (current_score - 50) / abs(quarterly_slope)` when slope is negative.

**New variables:** `fcf_quality_slope` (quarterly rate of change), `quality_runway_quarters` (time to critical), `quality_acceleration` (2nd derivative -- accelerating or decelerating degradation)

### Enhancement 4.2A: Peer-Relative Forensic Scoring

**Current:** Absolute thresholds for accruals, Jones model residuals.

**Expert method:** Score accruals and discretionary accruals relative to **sector peers** (from linked_caches). A company with 5% accruals in a sector where the median is 8% is actually BETTER than average, while the same 5% in a sector with 2% median is a red flag.

**Formula:** `peer_relative_accruals = (company_accruals - sector_P50) / sector_MAD`

**New variables:** `peer_relative_red_flag` (0-100, peer-adjusted), `sector_accruals_percentile` (0-100)

### Enhancement 4.3A: Benford's Law on Revenue Digits

**Current:** Benford analysis on unspecified "reported revenue" digits.

**Expert method:** Apply Benford's Law separately to **revenue line items** AND **expense line items**. Companies that manipulate typically inflate revenue digits (first digits skew toward 1-2) while expenses remain Benford-compliant. The **divergence between revenue and expense Benford compliance** is a stronger manipulation signal than either alone.

**New variables:** `benford_revenue_chisq`, `benford_expense_chisq`, `benford_divergence_score` (revenue chi-sq minus expense chi-sq -- positive = revenue manipulation)

---

## Tier 2 Enhancements: Cash Flow

### Enhancement 4.5A: Extended DuPont Decomposition

**Current:** CROA vs ROIC spread (2 metrics).

**Expert method:** **5-factor DuPont decomposition** (Palepu, Healy & Peek 2019): `ROE = Tax_Burden * Interest_Burden * Operating_Margin * Asset_Turnover * Equity_Multiplier`. Decompose changes in profitability into WHICH component is driving it -- a margin-driven improvement is higher quality than leverage-driven.

**New variables:** `dupont_tax_burden` (NI/EBT), `dupont_interest_burden` (EBT/EBIT), `dupont_asset_turnover` (Rev/TA), `dupont_equity_multiplier` (TA/Equity), `dupont_quality_driver` (which factor changed most -- margin vs leverage vs turnover)

### Enhancement 4.6A: Cash Flow Duration

**Current:** Operating leverage (DOL, DFL, DTL).

**Expert method:** **Cash flow duration** (Dechow, Ge & Schrand 2010) -- measure the average time between cash collection and cash disbursement. Short duration = more resilient cash flows; long duration = more fragile (vulnerable to revenue timing).

**Formula:** `cf_duration = weighted_avg(DSO, DPO, DIO)` where weights are proportional to magnitude. Net CF duration = DSO + DIO - DPO (similar to CCC but weighted by variability).

**New variables:** `cf_duration_days` (weighted average), `cf_duration_stability` (rolling std of duration), `cf_fragility_score` (high duration + high variability = fragile)

---

## Tier 3 Enhancements: Balance Sheet

### Enhancement 4.7A: Hidden Leverage Detection

**Current:** OBS risk scores goodwill/intangibles ratio.

**Expert method:** Detect **hidden leverage** from operating leases, purchase commitments, and pension obligations that don't appear on-balance-sheet under US GAAP (pre-ASC 842 for historical data). Use the **Moody's lease-equivalent debt** adjustment: `adjusted_debt = total_debt + 8 * annual_rent_expense`.

**New variables:** `adjusted_debt_to_equity` (including lease-equivalent), `hidden_leverage_ratio` (off-balance / on-balance), `pension_underfunding_risk` (if pension data available)

### Enhancement 4.9A: Distress Distance Matrix

**Current:** Leverage stress tests 3 scenarios with fixed assumptions.

**Expert method:** Instead of fixed scenarios, compute a **distress distance matrix** showing how many quarters of each type of shock the company can absorb:

| Shock Type | Quarters to Covenant Breach | Quarters to Cash Exhaustion |
|---|---|---|
| Revenue -10% | ? | ? |
| Revenue -20% | ? | ? |
| Margin -200bps | ? | ? |
| Rate +100bps | ? | ? |

This is a generalization of the reverse stress test (Layer 2) applied to the quarterly HF statement data.

**New variables:** `distress_distance_matrix` (4x2 matrix), `weakest_shock` (which combination triggers fastest), `quarters_of_buffer` (minimum across all shocks)

---

## Tier 4 Enhancements: Inflection

### Enhancement 4.10A: Regime-Conditional Fundamental Momentum

**Current:** Momentum composite with fixed weights across all regimes.

**Expert method:** Weight momentum components differently by survival regime. In `company_survival`, revenue acceleration matters 70% (can the company grow out of trouble?). In `normal`, margin slope matters 40% (are they becoming more efficient?). Use the regime labels from Layer 2 to condition the weights.

**New variables:** `regime_adjusted_momentum` (0-100), `momentum_regime_bias` (which component drives the score in this regime)

### Enhancement 4.12A: Earnings Torpedo Risk Enhancement

**Current:** P(beat/miss/inline) from historical SUE distribution. `earnings_torpedo` exists in advanced_methods but is market-data-driven.

**Expert method:** Combine fundamental torpedo risk with market-implied expectations. A company with HIGH analyst expectations (high PE, low implied vol pre-earnings) AND declining fundamental momentum has the highest torpedo risk. **Market-Fundamental Torpedo Score:**
```
torpedo_risk = market_expectations_percentile * (1 - fundamental_momentum / 100)
```

**New variables:** `market_fundamental_torpedo` (0-100), `implied_expectations_percentile` (from PE rank + IV rank)

---

## Tier 5 Enhancements: Valuation

### Enhancement 4.13A: Residual Income Valuation (RIV)

**Current:** Standard DCF with MC.

**Expert method:** **Residual Income Model** (Ohlson 1995) -- values the company as book value + present value of excess earnings above cost of equity. More robust than DCF for companies with volatile cash flows because it anchors on book value.

**Formula:** `Intrinsic = BV + sum(RI_t / (1+r)^t)` where `RI_t = NI_t - r * BV_{t-1}` (earnings minus required return on equity).

**New variables:** `riv_intrinsic` (residual income value), `riv_excess_return` (latest RI / BV -- is the company creating value?), `riv_vs_dcf_divergence` (if RIV and DCF disagree, one model is more reliable)

### Enhancement 4.13B: Market-Implied Growth Rate

**Current:** DCF uses sampled growth rates from regime distributions.

**Expert method:** **Reverse-engineer the growth rate implied by the current stock price**. If the market price = DCF with growth = g*, then g* is the "market-implied growth rate." Compare g* to the company's actual growth -- if g* >> actual, the stock is priced for perfection (torpedo risk).

**Formula:** Solve `P = FCF * (1+g) / (WACC - g)` for g given P and FCF.

**New variables:** `implied_growth_rate` (%), `growth_gap` (implied minus actual growth), `priced_for_perfection_flag` (growth gap > 2x actual)

### Enhancement 4.15A: Relative Value vs Absolute Value Reconciliation

**Current:** PEG composite compares PE to growth.

**Expert method:** Reconcile **relative value** (PE vs peers from peer_ranking) with **absolute value** (DCF intrinsic). When they agree (both say cheap), conviction is high. When they disagree (PE says cheap but DCF says expensive), flag as "relative value trap" -- cheap for a reason.

**New variables:** `relative_absolute_agreement` (-1 to +1), `value_trap_flag` (relative cheap + absolute expensive), `conviction_boost` (agreement amplifies conviction)

---

## Synthesis Enhancements

### Enhancement 4.17A: Kelly Criterion Position Sizing

**Current:** Position signal uses `alpha * quality_mult * survival_mult * decay_mult * conviction`. Sizing is heuristic.

**Expert method:** **Half-Kelly sizing** (Kelly 1956) -- the mathematically optimal bet size given the edge and odds:
```
kelly_fraction = (p * b - q) / b
half_kelly = kelly_fraction / 2  # conservative: half-Kelly reduces variance of returns
```
Where `p = P(positive return from DCF MC)`, `b = median upside / median downside`, `q = 1 - p`.

**New variables:** `kelly_fraction` (full Kelly), `half_kelly_size` (conservative sizing), `kelly_edge` (p*b - q -- is there an edge at all?)

### Enhancement 4.20A: Bayesian Conviction Updating

**Current:** Fusion has a simplified Bayesian belief network.

**Expert method:** Proper **Bayesian updating** where each new HF metric is treated as evidence that updates the posterior probability of positive return:
```
P(positive | evidence_1, ..., evidence_n) propto P(positive) * prod(P(evidence_i | positive))
```
Likelihoods estimated from historical backtests: "When FCF quality > 80, what fraction of companies had positive 1-year returns?"

**New variables:** `bayesian_posterior_updated` (0-1, after all evidence), `strongest_evidence` (which HF metric shifted the posterior the most), `evidence_concordance` (do all evidence sources point the same way?)

---

## Implementation Priority Matrix

| Enhancement | Impact | Complexity | Dependencies | Priority |
|-------------|--------|------------|-------------|----------|
| 4.13B: Market-implied growth | HIGH | LOW | DCF result, close price | P1 |
| 4.17A: Kelly criterion sizing | HIGH | LOW | MC survival probability, DCF | P1 |
| 4.10A: Regime-conditional momentum | HIGH | LOW | Layer 2 survival_regime | P1 |
| 4.5A: DuPont decomposition | MEDIUM | LOW | Raw statements | P1 |
| 4.1A: FCF quality degradation tracker | MEDIUM | LOW | Historical FCF quality scores | P2 |
| 4.15A: Relative-absolute reconciliation | MEDIUM | LOW | DCF + peer_ranking | P2 |
| 4.2A: Peer-relative forensics | MEDIUM | MEDIUM | linked_caches | P2 |
| 4.13A: Residual income valuation | HIGH | MEDIUM | BV, NI, cost of equity | P2 |
| 4.9A: Distress distance matrix | MEDIUM | MEDIUM | Quarterly statements | P3 |
| 4.7A: Hidden leverage detection | MEDIUM | MEDIUM | Lease/rent data (may be NaN) | P3 |
| 4.6A: Cash flow duration | LOW | MEDIUM | CCC components | P3 |
| 4.20A: Bayesian conviction updating | MEDIUM | HIGH | Historical backtest data | P4 |
| 4.12A: Market-fundamental torpedo | LOW | MEDIUM | Advanced methods torpedo + PE | P4 |
| 4.3A: Revenue-expense Benford divergence | LOW | LOW | Granular revenue data | P4 |

---

## Proposed New Variables (~30 total)

| # | Variable | Module | Type |
|---|----------|--------|------|
| 1 | `fcf_quality_slope` | 4.1 | Float |
| 2 | `quality_runway_quarters` | 4.1 | Float |
| 3 | `peer_relative_red_flag` | 4.2 | 0-100 |
| 4 | `sector_accruals_percentile` | 4.2 | 0-100 |
| 5 | `benford_divergence_score` | 4.3 | Float |
| 6 | `dupont_tax_burden` | 4.5 | Float |
| 7 | `dupont_interest_burden` | 4.5 | Float |
| 8 | `dupont_asset_turnover` | 4.5 | Float |
| 9 | `dupont_equity_multiplier` | 4.5 | Float |
| 10 | `dupont_quality_driver` | 4.5 | Categorical |
| 11 | `cf_duration_days` | 4.6 | Float |
| 12 | `cf_fragility_score` | 4.6 | 0-100 |
| 13 | `adjusted_debt_to_equity` | 4.7 | Float |
| 14 | `hidden_leverage_ratio` | 4.7 | Float |
| 15 | `distress_distance_matrix` | 4.9 | 4x2 matrix |
| 16 | `quarters_of_buffer` | 4.9 | Float |
| 17 | `regime_adjusted_momentum` | 4.10 | 0-100 |
| 18 | `market_fundamental_torpedo` | 4.12 | 0-100 |
| 19 | `riv_intrinsic` | 4.13 | Float ($) |
| 20 | `riv_excess_return` | 4.13 | Float |
| 21 | `riv_vs_dcf_divergence` | 4.13 | Float |
| 22 | `implied_growth_rate` | 4.13 | Float (%) |
| 23 | `growth_gap` | 4.13 | Float (%) |
| 24 | `priced_for_perfection_flag` | 4.13 | Boolean |
| 25 | `relative_absolute_agreement` | 4.15 | Float (-1 to +1) |
| 26 | `value_trap_flag` | 4.15 | Boolean |
| 27 | `kelly_fraction` | 4.17 | Float |
| 28 | `half_kelly_size` | 4.17 | Float |
| 29 | `kelly_edge` | 4.17 | Float |
| 30 | `bayesian_posterior_updated` | 4.20 | Float (0-1) |

All result objects -- 0 new cache columns.

---

## Implementation Checklist

```
[ ] Phase 1: P1 Enhancements (high impact, low complexity)
    [ ] 4.13B: Market-implied growth rate (reverse-engineer from price)
    [ ] 4.17A: Half-Kelly position sizing
    [ ] 4.10A: Regime-conditional momentum weights
    [ ] 4.5A: 5-factor DuPont decomposition

[ ] Phase 2: P2 Enhancements
    [ ] 4.1A: FCF quality temporal degradation tracker
    [ ] 4.15A: Relative-absolute value reconciliation
    [ ] 4.2A: Peer-relative forensic scoring
    [ ] 4.13A: Residual income valuation

[ ] Phase 3: P3 Enhancements
    [ ] 4.9A: Distress distance matrix
    [ ] 4.7A: Hidden leverage detection
    [ ] 4.6A: Cash flow duration

[ ] Phase 4: P4 Enhancements (deferred)
    [ ] 4.20A: Bayesian conviction updating
    [ ] 4.12A: Market-fundamental torpedo
    [ ] 4.3A: Revenue-expense Benford divergence
```
