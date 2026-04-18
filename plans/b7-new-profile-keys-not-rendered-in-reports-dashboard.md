# B7: New Profile Keys Not Rendered in Reports/Dashboard

## Gap Analysis

Cross-referencing profile keys injected in `main.py` Step 7 against report section builders in `report_generator.py` and dashboard display in `dashboard.py`.

### Profile Keys with NO Report Section

These 3 profile keys are injected into the profile JSON but have **zero rendering** in any report tier:

| Profile Key | Source Module | Data Available | Report Section | Dashboard |
|---|---|---|---|---|
| `options_signals` | `features/options_signals.py` (Gap 1) | PCR, risk reversal, IV skew, VIX term structure, SKEW index, variance risk premium | **MISSING** -- no builder, no section ID | **MISSING** |
| `cross_asset_signals` | `features/cross_asset_signals.py` (Gap 3) | Sector RS, rank, dispersion, yield curve, USD momentum, stress index | **MISSING** -- no builder, no section ID | **MISSING** |
| `event_calendar_signals` | `features/event_calendar.py` (Gap 4) | Days to next event, event uncertainty premium, FOMC/earnings proximity, event density | **MISSING** -- no builder, no section ID | **MISSING** |

### Profile Keys with Report Sections but NO Dashboard Display

These keys have report builders wired into `TIER_SECTIONS` but are not shown on the dashboard home page or report summary tab:

| Profile Key | Report Section | Section ID | Dashboard |
|---|---|---|---|
| `corporate_structure` | 21.6. Corporate Structure | 1999 | **MISSING** |
| `model_diagnostics` | 21.5. Model Robustness Diagnostics | 1998 | **MISSING** |
| `scenario_analysis` | 19.12. Scenario Analysis | 1997 | **MISSING** |
| `multi_frequency` | 21.11. Multi-Frequency Analysis | 2005 | Partial -- shows regime/survival in summary tab but not frequency breakdown |
| `predicted_regime_shifts` | Not in section builders but referenced in advanced insights | N/A | **MISSING** |

### Profile Keys Already Working in Both

These are correctly rendered in reports AND dashboard:

| Profile Key | Report | Dashboard |
|---|---|---|
| `position_signal` | 21.13 | Home page card |
| `signal_ic` | 21.14 | Home page card |
| `market_buying_power` | 19.9 | Home page card |
| `product_catalysts` | 19.10 | Home page card |
| `supply_chain_stress` | 21.9 | Home page card |
| `prediction_log` | In position signal section | Home page card |
| `product_segments` | 30 | Not on dashboard -- low priority |
| `hedge_fund` | Sections 23-29 | Not on dashboard -- large, report-only |
| `unified_survival_system` | 19.11 | Not needed -- survival info shown elsewhere |

---

## Implementation Plan

### Part 1: Report Section Builders (report_generator.py)

#### 1.1 Options Signals Section (Gap 1)

Add `_build_options_signals_section()` to render the 6 options-derived features.

**Section ID:** 2031
**Title:** "31. Options-Derived Forward Signals"
**Tier:** Pro + Premium
**Profile key:** `options_signals`

**Content to render:**
- Put/Call Ratio with interpretation -- above 1.0 = bearish sentiment, below 0.7 = bullish
- 25-Delta Risk Reversal -- negative = demand for puts, institutional hedging
- IV Skew -- steep = crash fear premium
- VIX Term Structure -- contango vs backwardation
- SKEW Index -- tail risk pricing
- Variance Risk Premium -- positive = markets overpricing risk

**Format:** Status badge at top -- RISK ON / NEUTRAL / RISK OFF based on composite. Then 6-row metric table with value, interpretation, and signal direction arrow.

#### 1.2 Cross-Asset Signals Section (Gap 3)

Add `_build_cross_asset_signals_section()`.

**Section ID:** 2032
**Title:** "32. Cross-Asset Sector Rotation Signals"
**Tier:** Pro + Premium
**Profile key:** `cross_asset_signals`

**Content to render:**
- Sector Relative Strength vs S&P 500
- Sector rank among 11 GICS sectors over 12 months
- Sector dispersion -- low = herding, high = stock picking opportunity
- Yield curve 10Y-2Y -- inversion warning
- USD momentum -- strong dollar headwind for multinationals
- Cross-asset stress index -- composite

**Format:** Sector rotation heatmap description -- "Your sector ranks #X of 11 over 12 months". Traffic light for yield curve and USD. Stress index gauge.

#### 1.3 Event Calendar Section (Gap 4)

Add `_build_event_calendar_section()`.

**Section ID:** 2033
**Title:** "33. Upcoming Events & Uncertainty Calendar"
**Tier:** Pro + Premium
**Profile key:** `event_calendar_signals`

**Content to render:**
- Days to next event -- countdown
- Event uncertainty premium -- confidence haircut percentage
- FOMC proximity -- impact on rate-sensitive sectors
- Earnings proximity -- volatility compression before release
- Event density over next 30 days -- "busy" vs "quiet" calendar
- Predicted next filing date from `filing_calendar`

**Format:** Timeline-style list of upcoming events with days remaining and impact rating. Uncertainty premium shown as a confidence modifier percentage.

#### 1.4 Predicted Regime Shifts Section

Add `_build_predicted_regime_shifts_section()` as a dedicated section rather than buried in advanced insights.

**Section ID:** 2034
**Title:** "34. Predicted Regime Shifts"
**Tier:** Premium only
**Profile key:** `predicted_regime_shifts`

**Content to render:**
- Current regime label
- P(exit 21d) and P(exit 252d) with interpretation
- Expected days to shift
- Most probable next regime
- Transition matrix visualization as text table

#### 1.5 Wire New Sections into TIER_SECTIONS

Update `TIER_SECTIONS` dict:
- **Pro:** Add 2031, 2032, 2033
- **Premium:** Add 2031, 2032, 2033, 2034

### Part 2: Dashboard Rendering (dashboard.py)

#### 2.1 Options Signals Card (Home Page)

Add a card in the home page profile display section showing:
- Composite signal: RISK ON / NEUTRAL / RISK OFF badge
- PCR value with small arrow
- IV Skew value

#### 2.2 Cross-Asset Card (Home Page)

Add a card showing:
- Sector rank: "#X of 11" with color
- Stress index value with gauge color

#### 2.3 Event Calendar Card (Home Page)

Add a card showing:
- "Next event in X days" countdown
- Event density badge: quiet / moderate / busy
- Uncertainty premium percentage

#### 2.4 Regime Shift Card (Home Page)

Add a card showing:
- "P(shift 21d) = X%" 
- "Next regime: Y" with color

#### 2.5 Report Summary Tab Enhancement

In the Report page Summary tab, add a new "Forward Signals" row that shows:
- Options composite signal
- Cross-asset stress
- Event uncertainty premium
- Regime shift probability

### Part 3: Backtest Runner Parity

#### 3.1 Inject New Profile Keys in backtest_runner.py

Add the same 3 profile key injections that `main.py` has:
- `profile["options_signals"]`
- `profile["cross_asset_signals"]`  
- `profile["event_calendar_signals"]`

These are currently computed in backtest_runner.py Step 1 but not injected into the profile dict in the profile-building section.

---

## File Changes Summary

| File | Changes |
|---|---|
| `operator1/report/report_generator.py` | Add 4 section builder functions (~120 lines each). Update `TIER_SECTIONS` to include new IDs. |
| `dashboard.py` | Add 4 cards to home page profile display (~60 lines). Add forward signals row to summary tab (~20 lines). |
| `backtest_runner.py` | Add 3 profile key injections in the profile-building section (~15 lines). |

---

## Execution Order

```
1. report_generator.py -- add _build_options_signals_section()
2. report_generator.py -- add _build_cross_asset_signals_section()
3. report_generator.py -- add _build_event_calendar_section()
4. report_generator.py -- add _build_predicted_regime_shifts_section()
5. report_generator.py -- update TIER_SECTIONS with new IDs
6. dashboard.py -- add 4 new cards to home page
7. dashboard.py -- add forward signals to summary tab
8. backtest_runner.py -- add 3 profile key injections
9. Verify: run AST parse on all 3 files
10. Push and create PR
```
