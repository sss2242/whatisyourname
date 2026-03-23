# Institutional Ownership x Existing 42 Models: Cooperation Map

How the two new modules (Ownership Contagion Scorer + Institutional Flow Predictor) cooperate with every relevant existing model to produce finer results. Organized by the 42-model map layers.

---

## New Columns Produced

From **institutional_flow.py** (Module 2):
- `inst_flow_momentum` -- EMA-smoothed QoQ ownership change
- `inst_crowding_risk` -- dynamic concentration x ownership
- `inst_smart_money_signal` -- top-5 vs total divergence
- `inst_insider_signal` -- net insider buying/selling
- `inst_herding_measure` -- LSV herding coefficient

From **ownership_contagion.py** (Module 1):
- `inst_mhhi_delta` -- common ownership with competitors
- `inst_bipartite_centrality` -- ownership network centrality
- `inst_ownership_network_density` -- network interconnectedness
- `inst_crowding_score` -- static crowded trade fragility
- `inst_crowded_trade_flag` -- binary crowding flag
- `inst_amihud_illiquidity` -- daily Amihud ratio (time-varying)
- `inst_liquidation_days` -- days to exit top-5 positions
- `inst_liquidation_risk` -- normalized liquidation pressure

---

## Layer 1: Feature Engineering -- Direct Cooperation

### #1 derived_variables.py -- Amihud illiquidity as a new derived variable

The `inst_amihud_illiquidity` column is computed from `|return_1d| / dollar_volume`. This is the only daily time-varying institutional column. It should be computed inside `compute_institutional_flow()` but follows the same `safe_ratio()` pattern as existing derived variables.

**Cooperation:** Amihud feeds directly into:
- Liquidation pressure calculation (contagion scorer)
- Survival mode triggers (high illiquidity + high crowding = danger)
- Volatility models (Amihud correlates with future volatility spikes)

### #4 linked_aggregates.py -- Ownership overlap as an aggregate

**Enhancement:** Add `inst_ownership_overlap` to `AGGREGATE_VARIABLES`. When computing linked aggregates, for each entity group, compute the mean MHHI overlap coefficient between the target and group members. This produces:

- `competitors_avg_inst_ownership_overlap` -- how much the target shares holders with its competitors (average across competitor group)
- `suppliers_avg_inst_ownership_overlap` -- same for suppliers

**Why this matters:** A company whose suppliers are all owned by the same institutions as the company itself has correlated supply chain risk through ownership channels, not just business channels.

### #7 news_sentiment.py -- Institutional selling as sentiment divergence

**Cooperation via vanity.py Component 5 (sentiment-reality gap):**

When `inst_flow_momentum < -0.1` (institutions selling) AND `news_sentiment_score > 0.3` (positive news), this is a classic "dumb money" signal -- retail investors buying on good headlines while institutions quietly exit. The existing vanity component 5 (sentiment-reality gap) captures part of this, but we can make it sharper:

- vanity.py `_sentiment_reality_gap()` currently compares `news_sentiment_score` vs `fh_composite_score`
- **Enhancement:** Also compare `news_sentiment_score` vs `inst_flow_momentum` -- divergence between public sentiment and institutional behavior is a stronger vanity signal than sentiment vs financial health alone

### #8 peer_ranking.py -- Ownership concentration ranking

**Enhancement:** Add institutional metrics to the peer ranking variables. When linked caches have `inst_ownership_pct` or `inst_top5_concentration`, the peer ranker can compute:

- `peer_rank_inst_concentration` -- is the target more or less concentrated than peers?
- `peer_rank_inst_liquidity` -- is the target more or less liquid (by Amihud) than peers?

High concentration + low liquidity relative to peers = the target is the most fragile stock in its group.

---

## Layer 2: Analysis Modules -- Direct Cooperation

### #11 survival_mode.py -- Two new survival triggers

**New trigger 1: Extreme institutional selling**
```
inst_flow_momentum < -0.15
```
When institutions reduce their stake by more than 15% in a quarter, that is a survival-level event. Institutions have information advantages -- their mass exit signals deep problems.

**New trigger 2: Crowded + illiquid**
```
inst_crowding_score > 0.8 AND inst_amihud_illiquidity > 90th percentile
```
When ownership is highly concentrated AND the stock is illiquid, any single large holder selling creates a cascading price crash. This is the Khandani and Lo 2011 crowded trade scenario.

**Impact:** These triggers flow into `company_survival_mode_flag`, which in turn drives hierarchy weight shifts (Tier 1 liquidity gets 50-60% weight in survival mode), survival timeline mode classification, and all downstream models.

### #12 hierarchy_weights.py -- Sobol feedback with institutional variables

The Sobol sensitivity analysis (#36) already feeds back into hierarchy weights via `adjust_hierarchy_from_sobol()`. With the new institutional columns in the cache, Sobol will now measure how sensitive `return_1d` is to:
- `inst_flow_momentum` (institutional selling pressure)
- `inst_crowding_score` (ownership fragility)
- `inst_amihud_illiquidity` (market microstructure liquidity)

If Sobol finds that `inst_flow_momentum` has high first-order sensitivity, the hierarchy weights will automatically shift toward Tier 3 (Market Stability) where liquidity-related variables live. This is data-driven weight calibration -- no hardcoded rules needed.

### #13 survival_timeline.py -- Institutional regime-state interaction

The enriched survival timeline combines rule-based survival modes with HMM market regimes into 11 combined states (stable_growth through extreme_crisis). With the new institutional triggers in survival_mode:

- A company in `normal` survival mode but with `inst_flow_momentum < -0.15` will now flip to `company_only` survival mode
- Combined with a `bear` HMM regime, this becomes `company_distress_severe` (intensity 0.55-0.75)
- The survival intensity drives hierarchy weight interpolation in real-time

**Finer result:** The timeline captures a new class of distress -- "institutional exit" -- that was previously invisible to the rule-based system which only looked at financial ratios.

### #14 fuzzy_protection.py -- MHHI as protection input

**Enhancement:** Common ownership (MHHI delta > 0.3) between the target and competitors reduces competitive incentives. In the fuzzy protection framework, this is analogous to government protection -- the company is "protected" by the fact that its largest shareholders also own its competitors and have no incentive to let any single company fail.

Add `inst_mhhi_delta` as a fourth fuzzy input:
- High MHHI (> 0.3) -> `institutional_protection` membership score 0.6-0.9
- This does NOT increase the `fuzzy_protection_degree` directly, but it modifies the policy responsiveness signal: if both government AND institutional protection are high, the overall protection degree is amplified.

### #17 vanity.py -- Capital misallocation with institutional context

Component 3 (capital misallocation) checks if the company is paying dividends while solvency is critical, or taking on debt while liquidity is declining.

**Enhancement:** If `inst_flow_momentum < 0` (institutions selling) AND the company is doing stock buybacks (`stock_buybacks > 0`), that is extreme vanity. Management is using cash to buy back shares while their most sophisticated investors are dumping. This is a new sub-signal within Component 3:

```
buyback_while_selling = (stock_buybacks > 0) AND (inst_flow_momentum < -0.05)
vanity_penalty = min(20, abs(inst_flow_momentum) * 100)
```

---

## Layer 3: Temporal Models -- Automatic + Explicit Cooperation

### Automatic consumption via _extra_vars

All `inst_*` columns are automatically included in `_extra_vars` via the prefix match `c.startswith("inst_")` in main.py line 1765. This means the following models consume them without any code changes:

### #20 granger_causality.py -- Causal testing of institutional variables

Granger causality will test whether `inst_flow_momentum` Granger-causes `return_1d` (do institutional flows predict returns?), whether `inst_crowding_score` Granger-causes `volatility_21d` (does crowding predict vol spikes?), etc.

**Expected findings:**
- `inst_flow_momentum -> return_1d`: likely significant (institutional flows lead returns by 1-5 days)
- `inst_amihud_illiquidity -> volatility_21d`: likely significant (illiquidity predicts volatility)
- `inst_crowding_score -> drawdown_252d`: likely significant at longer lags (crowding predicts eventual crashes)

If Granger finds these causal links significant, the variables are retained in `_extra_vars`. If not, they are pruned (correct behavior -- the data tells us whether institutional signals matter for this specific company).

### #21 causality.py (transfer entropy) -- Information flow from institutional behavior

Transfer entropy measures *nonlinear* information flow. While Granger is limited to linear relationships, TE will capture cases like: institutional selling (`inst_flow_momentum < 0`) causes volatility spikes (`volatility_21d` increases) through a nonlinear threshold effect.

### #22 model_synergies.py -- Unified causal network with ownership

The pre-forecasting synergies module builds a unified causal network from Granger + transfer entropy results. With institutional variables included, the causal network will show:
- Whether institutional flows are a *root cause* or a *symptom* of other variables
- Whether the ownership overlap (MHHI) creates causal pathways between the target and competitors

### #23 forecasting.py -- Institutional features improve forecasts

**VAR model:** Including `inst_flow_momentum` and `inst_amihud_illiquidity` as exogenous variables in the VAR model improves multi-step forecasts because institutional behavior is a leading indicator.

**LSTM model:** The LSTM can learn the nonlinear relationship between institutional selling pressure and return sequences. When `inst_flow_momentum` drops below a threshold, the LSTM learns to predict more negative returns.

**Tree ensembles (XGBoost/RF/GBM):** These can capture interaction effects: `inst_crowding_score * inst_amihud_illiquidity` as a compound risk factor.

### #25 monte_carlo.py -- Drift adjustment from institutional flows

**Explicit cooperation (requires code change):**

```python
# In run_monte_carlo(): adjust per-regime drift
if "inst_flow_momentum" in cache.columns:
    flow = float(cache["inst_flow_momentum"].iloc[-1]) if cache["inst_flow_momentum"].notna().any() else 0.0
    if flow < -0.05:
        # Institutions selling -> reduce expected return
        drift_adjustment = flow * 0.5  # 50% pass-through
        for regime_key in regime_params:
            regime_params[regime_key]["mean"] += drift_adjustment
```

**Multivariate MC enhancement:**

`run_multivariate_monte_carlo()` jointly simulates (return, delta_current_ratio, delta_fcf_yield, delta_debt_to_equity). With `inst_amihud_illiquidity` in the cache, add it as a 5th simulated variable. This allows the MC to simulate scenarios where illiquidity increases simultaneously with institutional selling, producing more realistic tail distributions.

### #30 conformal.py -- Wider intervals under institutional stress

**Explicit cooperation (requires code change):**

The `ConformalPIDCalibrator` uses Mondrian partitioning by survival mode. We can add an additional partition dimension: institutional crowding state.

```python
# When inst_crowding_score > 0.8: use a wider quantile
# This is automatic if inst_crowded_trade_flag is used as a Mondrian partition variable
```

Alternatively, the simpler approach: the prediction_aggregator already widens bands when model uncertainty is high. With `inst_crowding_risk` in the cache, the aggregator can scale band width proportionally.

### #31 copula.py -- Tail dependence with institutional variables

The copula analysis models tail dependencies between variables. Including `inst_flow_momentum` and `inst_amihud_illiquidity` in the copula will reveal:
- Whether institutional selling and market drawdowns have asymmetric tail dependence (Clayton copula detects lower-tail dependence)
- The probability of a joint crisis where institutional exit coincides with liquidity drying up

This feeds into the `joint_crisis_probability` metric which is already used by the prediction aggregator.

### #33 cycle_decomposition.py -- Ownership cycles

If `inst_flow_momentum` has enough non-NaN observations (KR market with 8 quarterly points interpolated to daily), the cycle decomposition can detect:
- Whether institutional buying/selling follows a seasonal cycle (e.g., quarter-end rebalancing)
- The dominant period of institutional flow cycles

This is marginal for US/UK (single snapshot) but valuable for KR.

### #35 explainability.py (SHAP) -- Ownership as prediction driver

SHAP will now explain predictions in terms of institutional behavior:
- "inst_flow_momentum contributed -2.3% to the current_ratio forecast (institutional selling depresses the stock price, reducing market cap and indirectly affecting equity ratios)"
- "inst_amihud_illiquidity contributed +1.1% to the volatility_21d forecast (illiquid stocks have higher realized volatility)"

These explanations go into the report narrative, giving the user actionable insight about *why* the model predicts what it predicts.

### #36 sensitivity.py (Sobol) -- Quantifying institutional influence

Sobol sensitivity analysis decomposes the variance of `return_1d` into contributions from each input variable. With institutional columns:
- First-order `S1(inst_flow_momentum)` tells us what fraction of return variance is explained by institutional flows alone
- Total-order `ST(inst_flow_momentum)` tells us the contribution including interactions with other variables

If `ST(inst_flow_momentum)` is high (say > 0.1, meaning 10% of return variance), the hierarchy weights will be adjusted toward liquidity/stability tiers via the Sobol feedback loop in `adjust_hierarchy_from_sobol()`.

### #38 graph_risk.py -- Ownership contagion channel

**Explicit cooperation (requires code change):**

Currently graph_risk models contagion through business relationships only (supply chain, customer/competitor links). The contagion scorer adds a second channel: ownership overlap.

```python
# In main.py Step 5e, after computing contagion result:
if contagion_result and contagion_result.mhhi_pairwise:
    ownership_weights = {}
    for ent_id, overlap in contagion_result.mhhi_pairwise.items():
        ownership_weights[ent_id] = 0.3 + 0.7 * overlap
        # Base 30% contagion + 70% scaled by ownership overlap
    
    graph_risk_result = compute_graph_risk_metrics(
        target_isin=...,
        relationships=...,
        edge_weights=ownership_weights,  # NEW: ownership-weighted edges
        target_cache=cache,
        linked_caches=linked_caches,
    )
```

This means graph_risk's SIR contagion simulation will spread distress more aggressively through ownership-connected nodes. If Vanguard owns 10% of both AAPL and MSFT, distress in MSFT is more likely to infect AAPL (through correlated selling by shared holders).

### #39 game_theory.py -- Common ownership reduces competition

**Explicit cooperation (requires code change):**

The Cournot/Bertrand game theory analysis assumes independent profit-maximizing firms. When MHHI delta is high, firms share owners who internalize the externality of competition. The competitive pressure index should be reduced by the MHHI factor:

```python
# In game_theory.py: adjust competitive_pressure
if mhhi_delta is not None and mhhi_delta > 0.1:
    # Common ownership softens competition (Azar et al 2018)
    competitive_pressure *= (1.0 - mhhi_delta * 0.5)
    # 50% pass-through: MHHI=0.3 reduces pressure by 15%
```

This produces a more realistic assessment of competitive dynamics -- a company in an industry with high common ownership faces less competitive pressure than the raw market structure suggests.

### #40 dtw_analogs.py -- Institutional regime as analog filter

**Explicit cooperation (optional, requires code change):**

When searching for historical analogs, filter or weight by institutional regime:
- If the current `inst_crowding_score > 0.7`, prioritize analogs from past periods that also had high crowding
- Analogs from similar institutional environments produce more relevant forecasts

### #41 prediction_aggregator.py -- Band widening under institutional stress

**Explicit cooperation (requires code change):**

```python
# In run_prediction_aggregation(): widen uncertainty bands
if "inst_crowding_risk" in cache.columns:
    cr = float(cache["inst_crowding_risk"].iloc[-1]) if cache["inst_crowding_risk"].notna().any() else 0.0
    if cr > 0.7:
        # Crowded positions create fat-tailed risk
        band_multiplier = 1.0 + (cr - 0.7) * 1.5  # up to 45% wider at max crowding
        for var in result.predictions:
            for horizon in result.predictions[var]:
                hp = result.predictions[var][horizon]
                if hp.lower_bound is not None and hp.upper_bound is not None:
                    center = hp.point_forecast
                    width = (hp.upper_bound - hp.lower_bound) / 2
                    hp.lower_bound = center - width * band_multiplier
                    hp.upper_bound = center + width * band_multiplier
```

---

## Summary: Cooperation Matrix

| Existing Model | Cooperation Type | What Changes | Impact |
|---------------|-----------------|--------------|--------|
| #1 derived_variables | Pattern (Amihud follows safe_ratio) | None (separate module) | Amihud feeds liquidation + survival |
| #4 linked_aggregates | Enhancement | Add ownership overlap to AGGREGATE_VARIABLES | Cross-entity ownership risk metric |
| #7 news_sentiment | Via vanity #17 | Compare sentiment vs inst_flow | Sharper sentiment-reality gap signal |
| #8 peer_ranking | Enhancement | Add inst concentration + Amihud to ranking | Relative fragility measure |
| #11 survival_mode | **Explicit** | 2 new triggers (selling + crowded/illiquid) | Catches institutional exit distress |
| #12 hierarchy_weights | Via Sobol #36 | Automatic (Sobol adjusts) | Data-driven tier reweighting |
| #13 survival_timeline | Automatic | New triggers cascade through | New distress mode captured |
| #14 fuzzy_protection | Enhancement | MHHI as 4th fuzzy input | Institutional protection layer |
| #17 vanity | Enhancement | Buyback-while-selling signal | Sharper capital misallocation |
| #20 granger_causality | **Automatic** | inst_* in _extra_vars | Tests if flows cause returns |
| #21 transfer_entropy | **Automatic** | inst_* in _extra_vars | Nonlinear information flow |
| #22 model_synergies | Automatic | Unified causal network | Ownership in causal graph |
| #23 forecasting | **Automatic** | inst_* in _extra_vars | VAR/LSTM/tree use as features |
| #25 monte_carlo | **Explicit** | Drift adjustment from flows | Realistic return simulation |
| #27 transformer | **Automatic** | inst_* in _extra_vars | Attention learns flow patterns |
| #30 conformal | **Explicit** | Wider intervals under crowding | Fat-tail coverage |
| #31 copula | **Automatic** | inst_* in variable selection | Tail dependence with ownership |
| #33 cycle_decomposition | **Automatic** | Detects rebalancing cycles | Seasonal ownership patterns |
| #35 SHAP | **Automatic** | inst_* as feature importance | Explains ownership impact |
| #36 Sobol sensitivity | **Automatic** | inst_* sensitivity measured | Quantifies ownership influence |
| #38 graph_risk | **Explicit** | Ownership edge weights | Second contagion channel |
| #39 game_theory | **Explicit** | MHHI reduces competitive pressure | Realistic competition model |
| #40 dtw_analogs | Optional explicit | Filter by institutional regime | Better analog selection |
| #41 prediction_aggregator | **Explicit** | Band widening under crowding | Fat-tail uncertainty |

**Cooperation count:**
- **Automatic** (zero code changes, _extra_vars mechanism): 10 models
- **Explicit** (targeted code changes in existing modules): 7 models
- **Enhancement** (small additions to existing modules): 5 models
- **Total models cooperating:** 22 out of 42

The other 20 models either operate on fixed inputs that correctly should not see institutional data (regime detector uses return_1d only; particle filter uses specific survival variables) or are utility/infrastructure modules.

---

## Expected Result Quality Improvements

| Metric | Before | After | Why |
|--------|--------|-------|-----|
| Survival flag accuracy | Catches financial distress only | Also catches institutional exit distress | Two new triggers: mass selling, crowded+illiquid |
| Forecasting RMSE | Baseline | ~5-10% improvement for stocks with high institutional ownership | inst_flow_momentum is a leading indicator |
| Monte Carlo tail accuracy | Symmetric tails | Asymmetric (heavier left tail when institutions selling) | Drift adjustment from flows |
| Uncertainty band calibration | Based on model residuals only | Wider under crowding (more honest) | Crowding creates fat tails the base models miss |
| Graph contagion realism | Business relationships only | Business + ownership channels | Shared-holder selling is a real contagion path |
| Competitive analysis accuracy | Assumes independent firms | Adjusts for common ownership | MHHI reduces competitive pressure when warranted |
| SHAP explanations | No ownership narrative | Explains institutional flow impact | Actionable insight for investors |
| Report depth | No institutional deep analysis | Full section 19.8 with MHHI, crowding, flow, insider signals | Professional-grade ownership analysis |
