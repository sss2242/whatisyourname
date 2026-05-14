# Fix Guide: 3 Bugs from AAPL Backtest 2024-12-31

*Created: 2026-05-14 | Priority: High | Source: Live backtest run (63/63 stages, 21.1min)*

---

## Bug 1: CCC anomaly -- DSO/DIO/DPO use daily-interpolated revenue

**Severity:** High (produces garbage values: DSO=8,907 days, CCC=-9,515 days)
**Impact:** CCC consumed by HF asset quality, prediction aggregator, extra_vars. Garbage in -> garbage out for downstream models.

### Root Cause

`operator1/features/derived_variables.py` line 1405-1428:

```python
# Line 1410-1412: tries to be frequency-aware
freq = _get_freq()
_period_days = _PERIOD_DAYS.get(freq, 90)
if freq == "D":
    _period_days = 90  # assume quarterly filing cadence

# Line 1424-1428: computes DSO
revenue = df.get("revenue")          # <-- THIS IS DAILY-INTERPOLATED
rev_per_day = revenue / _period_days  # quarterly_revenue/63 / 90 = tiny number
dso = receivables / rev_per_day       # huge_receivables / tiny_number = 8907 days
```

At daily frequency, `revenue` in the cache is the quarterly revenue (~$94B) distributed across 63 business days = ~$1.49B/day. Then `rev_per_day = $1.49B / 90 = $16.6M/day`. DSO = $29.6B / $16.6M = 1,783 days. But the actual data shows $299M daily revenue, making DSO even worse at 8,907 days.

The fundamental problem: `revenue / _period_days` double-divides. The daily-interpolated revenue is ALREADY a daily rate. Dividing by 90 again produces a sub-daily rate.

### Fix

In `derived_variables.py` at the CCC computation (line 1405-1441):

**Option A (preferred): Use TTM revenue for DSO at daily frequency.**

```python
if freq == "D":
    # At daily freq, revenue is already daily-interpolated.
    # Use revenue_ttm (4Q sum) for DSO/DIO/DPO -- the standard accounting formula.
    _rev_for_ccc = df.get("revenue_ttm_asof")
    if _rev_for_ccc is None or _rev_for_ccc.isna().all():
        _rev_for_ccc = revenue  # fallback
    _period_for_ccc = 365  # TTM = 365 calendar days
else:
    _rev_for_ccc = revenue
    _period_for_ccc = _period_days  # Q=90, A=365, S=180
```

Then use `_rev_for_ccc / _period_for_ccc` instead of `revenue / _period_days`.

**Option B: Don't re-divide at daily frequency.**

```python
if freq == "D":
    # revenue is already daily-interpolated, don't divide by period_days
    rev_per_day = revenue.astype(float)
else:
    rev_per_day = revenue.astype(float) / float(_period_days)
```

### Files to modify

1. `operator1/features/derived_variables.py` lines 1405-1441 (DSO/DIO/DPO/CCC computation)

### Downstream consumers (verify after fix)

- `operator1/hedge_fund/engine.py` line 310-336 (HF asset quality DSO -- NOTE: HF uses raw quarterly statements, not cache, so it's NOT affected by this bug)
- `operator1/stages/stage2_freq_pipeline.py` line 68-71 (freq pipeline copies CCC to per-freq state)
- `operator1/features/six_derived_proxies.py` lines 3430-3626 (SIX uses hardcoded dso_days, not cache -- NOT affected)

### Expected result

DSO for AAPL: ~30-50 days (industry norm for tech). CCC: ~-30 to +20 days (Apple has negative CCC due to fast collections + slow payments).

---

## Bug 2: OHLC high/low inversion in aggregated predictions

**Severity:** Medium (incorrect price levels in predictions, but OHLC predictor module itself is correct)
**Impact:** Prediction summary shows high=$235 < low=$238 for AAPL. Report section cites wrong levels.

### Root Cause

The OHLC predictor (`ohlc_predictor.py` line 78-85) correctly enforces `high >= max(open, close)` and `low <= min(open, close)` per candle. The bug is NOT there.

The bug is in the **forecasting cascade** (`forecasting.py`). "high" and "low" are in the cache as historical OHLCV columns. The forecasting engine treats them as independent variables and runs separate Kalman/Tree/Baseline models for each. These models don't know about the constraint `high >= low`. When the ensemble weights or model predictions diverge, the aggregated "high" prediction can be lower than the aggregated "low" prediction.

The inversion then flows to `predictions_summary.json` via `prediction_aggregator.py` which stores per-variable per-horizon predictions without cross-variable constraints.

### Fix

In `operator1/models/prediction_aggregator.py`, after the main aggregation loop (step 10), add a post-prediction OHLC constraint:

```python
# Post-prediction OHLC consistency constraint
# Ensure predicted high >= predicted low for all horizons
for horizon in ["1d", "5d", "21d", "252d"]:
    h_pred = predictions.get("high", {}).get(horizon)
    l_pred = predictions.get("low", {}).get(horizon)
    c_pred = predictions.get("close", {}).get(horizon)
    o_pred = predictions.get("open", {}).get(horizon)
    
    if h_pred is not None and l_pred is not None:
        h_val = getattr(h_pred, 'point_forecast', None)
        l_val = getattr(l_pred, 'point_forecast', None)
        if h_val is not None and l_val is not None and l_val > h_val:
            # Swap: low should be the min, high should be the max
            h_pred.point_forecast, l_pred.point_forecast = l_val, h_val
            logger.warning("OHLC fix: swapped high/low at %s (was h=%.2f < l=%.2f)", 
                          horizon, h_val, l_val)
    
    # Also enforce: high >= max(open, close), low <= min(open, close)
    if h_pred and c_pred and o_pred:
        h_val = getattr(h_pred, 'point_forecast', None)
        c_val = getattr(c_pred, 'point_forecast', None)
        o_val = getattr(o_pred, 'point_forecast', None)
        if all(v is not None for v in [h_val, c_val, o_val]):
            h_pred.point_forecast = max(h_val, c_val, o_val)
    if l_pred and c_pred and o_pred:
        l_val = getattr(l_pred, 'point_forecast', None)
        c_val = getattr(c_pred, 'point_forecast', None)
        o_val = getattr(o_pred, 'point_forecast', None)
        if all(v is not None for v in [l_val, c_val, o_val]):
            l_pred.point_forecast = min(l_val, c_val, o_val)
```

### Files to modify

1. `operator1/models/prediction_aggregator.py` -- add post-aggregation OHLC constraint after step 10

### Downstream consumers (verify after fix)

- `operator1/report/report_generator.py` line 5037 (reads `ohlc_predictions.next_day.low` for stop-loss)
- `operator1/report/enhanced_outputs.py` line 361 (dashboard OHLC chart)
- `operator1/models/pattern_detector.py` line 389 (predicted pattern detection on OHLC)

---

## Bug 3: linked_caches never populated in backtest_runner.py (parity bug #14)

**Severity:** Critical (cascades to 23 unavailable profile modules)
**Impact:** Peer ranking, linked aggregates, ownership contagion, behavioral signals, complexity signals, copula, SHAP, and 16 other modules all show `available=False` because they depend on `state.linked_caches` which is always empty in backtest mode.

### Root Cause

`backtest_runner.py` runs entity discovery at line 856-857:
```python
discovery_result = discover_linked_entities(...)
```

This populates `state.relationships` (the entity names/tickers). But the next step -- **fetching financial data for each discovered entity** -- exists only in `main.py` lines 1900-1994:

```python
# main.py line 1900: defines _fetch_linked_entity()
# main.py line 1973: ThreadPoolExecutor fetches data for each entity
# main.py line 1983: linked_caches[ent_id] = ent_cache  <-- THIS LINE IS MISSING
```

`backtest_runner.py` never assigns `state.linked_caches`. It reads it in 8 places (lines 912, 952, 955, 976, 1140, 1150, 1167, 1378) but never populates it.

### Fix

In `backtest_runner.py`, after the entity discovery block (after line ~870), add the linked entity data fetch loop. This is ~70 lines copied from `main.py` lines 1860-1994, adapted to use `state.*` fields:

```python
# After entity discovery, fetch financial data for each linked entity
if state.relationships:
    from operator1.features.derived_variables import compute_derived_variables
    from concurrent.futures import ThreadPoolExecutor, as_completed
    
    _all_linked = []
    _entity_groups = {}
    for grp, ents in state.relationships.items():
        ids = []
        if isinstance(ents, list):
            for e in ents:
                eid = ""
                if isinstance(e, dict):
                    eid = e.get("isin", "") or e.get("ticker", "")
                elif hasattr(e, "isin"):
                    eid = e.isin or getattr(e, "ticker", "")
                if eid and eid not in {x.get("id") for x in _all_linked}:
                    _mkt = e.get("market_id", "") if isinstance(e, dict) else getattr(e, "market_id", "")
                    _all_linked.append({"id": eid, "name": e.get("name", "") if isinstance(e, dict) else getattr(e, "name", ""), "group": grp, "market_id": _mkt})
                    ids.append(eid)
        _entity_groups[grp] = ids
    _all_linked = _all_linked[:10]
    
    def _fetch_entity(ent_info):
        eid = ent_info["id"]
        try:
            _client = pit_client
            _qt = _client.get_quotes(eid)
            if not _qt.empty and "date" in _qt.columns:
                _qt["date"] = pd.to_datetime(_qt["date"])
                _ec = _qt.set_index("date").sort_index()
            else:
                _ec = pd.DataFrame(index=pd.date_range(cache.index[0], cache.index[-1], freq="B", name="date"))
            for _lbl, _sdf in [("inc", _client.get_income_statement(eid)), ("bal", _client.get_balance_sheet(eid)), ("cf", _client.get_cashflow_statement(eid))]:
                if _sdf.empty:
                    continue
                _dc = "report_date" if "report_date" in _sdf.columns else "filing_date"
                if _dc not in _sdf.columns:
                    continue
                _sdf[_dc] = pd.to_datetime(_sdf[_dc])
                _sdf = _sdf.sort_values(_dc).drop_duplicates(subset=[_dc], keep="last")
                _nc = [c for c in _sdf.select_dtypes(include=["number"]).columns if c != _dc and "date" not in c.lower()]
                if _nc:
                    _si = _sdf.set_index(_dc)[_nc]
                    _ci = _ec.index.union(_si.index).sort_values()
                    _sa = _si.reindex(_ci).ffill().reindex(_ec.index)
                    _nw = [c for c in _sa.columns if c not in _ec.columns]
                    if _nw:
                        _ec = _ec.join(_sa[_nw], how="left")
            if "close" in _ec.columns and _ec["close"].notna().sum() > 5:
                _ec = compute_derived_variables(_ec)
            return eid, _ec
        except Exception:
            return eid, pd.DataFrame()
    
    with ThreadPoolExecutor(max_workers=4) as executor:
        futs = {executor.submit(_fetch_entity, e): e for e in _all_linked}
        for f in as_completed(futs):
            try:
                eid, ec = f.result()
                if not ec.empty:
                    state.linked_caches[eid] = ec
            except Exception:
                pass
    logger.info("Linked data: %d/%d fetched", len(state.linked_caches), len(_all_linked))
```

### Files to modify

1. `backtest_runner.py` -- add linked entity data fetch loop after entity discovery (~line 870)

### Downstream modules that will activate after fix (23 modules)

These all read `state.linked_caches` and currently show `available=False`:

| Module | Profile Key | What it needs linked_caches for |
|--------|-------------|--------------------------------|
| Peer Ranking | `peer_ranking` | Percentile rank target vs peers |
| Linked Aggregates | `linked_entities` | Cross-entity aggregate statistics |
| Graph Risk (enhanced) | `graph_risk` | Edge-weighted contagion with ownership |
| Game Theory | `game_theory` | Competitor caches for Cournot/Stackelberg |
| Ownership Contagion | `institutional_ownership_analysis` | MHHI, crowding, liquidation |
| Behavioral Signals | `behavioral_signals` | Needs inst_flow_momentum (from holders) |
| Complexity Signals | `complexity_signals` | Independent, but blocked by pipeline ordering |
| Copula | `copula` | Joint tail dependence across variables |
| DTW Analogs | `dtw_analogs` | Cross-company analog search |
| SHAP | `shap_explanations` | Feature importance from forward pass |

### Risk Assessment

**Bug 1 (CCC):** Low risk. Isolated to one computation block. No cross-file dependencies except consumers that read the output column.

**Bug 2 (OHLC inversion):** Low risk. Post-prediction constraint is additive -- doesn't change any model, just clamps outputs.

**Bug 3 (linked_caches):** Medium risk. Adding ~70 lines of data fetching code to backtest_runner.py. The same code exists in main.py and has been tested. Main risk is API rate limiting during entity data fetch (SEC EDGAR has 10 req/s limit).

### Estimated Scope

- Bug 1: 1 file, ~10 lines changed
- Bug 2: 1 file, ~20 lines added
- Bug 3: 1 file, ~70 lines added
- Total: 3 files, ~100 lines
