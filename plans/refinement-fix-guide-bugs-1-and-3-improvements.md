# Refinement Fix Guide: Bugs 1 and 3

*Created: 2026-05-14 | Priority: Medium | Source: Architecture review of initial bug fixes*

---

## Bug 1 Refinement: Simplify COGS Scaling in CCC Formula

### Problem with Current Fix

Line 1441 in `derived_variables.py` has a complex COGS scaling expression:

```python
_cogs_for_ccc = (_ttm_rev.astype(float) - _ttm_gp.astype(float) * 
    (_ttm_rev / revenue.where(revenue.abs() > eps)).fillna(1.0)).clip(lower=eps)
```

This scales daily gross_profit to TTM level via `GP_daily * (TTM_Rev / Daily_Rev)`, then subtracts from TTM_Rev. It's mathematically valid but:
- Hard to read and maintain
- Prone to NaN propagation when daily revenue has gaps
- The ratio `_ttm_rev / revenue` can produce infinity when daily revenue is near zero

### Proposed Simplification

Replace with gross-margin-ratio approach:

```python
# Simplified COGS TTM scaling:
# COGS = Revenue * (1 - gross_margin), where gross_margin = GP / Revenue
if freq in ("D", "W", "M"):
    _ttm_rev = df.get("revenue_ttm_asof")
    if _ttm_rev is not None and _ttm_rev.notna().any():
        _rev_for_ccc = _ttm_rev
        _ccc_period = 365.0
        
        # COGS_ttm via gross margin ratio (simpler, no NaN propagation)
        if revenue is not None and gp is not None:
            _gm = (gp.astype(float) / revenue.astype(float).where(revenue.abs() > eps)).fillna(0.5)
            _gm_smoothed = _gm.rolling(63, min_periods=1).median()  # smooth out quarterly jumps
            _cogs_for_ccc = (_ttm_rev.astype(float) * (1.0 - _gm_smoothed)).clip(lower=eps)
    else:
        _rev_for_ccc = revenue
        _ccc_period = 1.0
```

**Why this is better:**
- `gross_margin = GP / Revenue` is a ratio, not affected by interpolation scale
- Rolling median smooths out quarterly filing jumps
- No infinity risk from dividing TTM by daily revenue
- Same result when data is clean, more robust when data has gaps

### Files to Modify

1. `operator1/features/derived_variables.py` lines 1437-1441 -- replace COGS scaling block

### Expected Result

Same DSO/DIO/DPO values when data is clean. More robust values when revenue has NaN gaps or near-zero periods.

---

## Bug 3 Refinement: Cross-Region Routing + Ownership Contagion

### Problem with Current Fix

The linked entity data fetch in `backtest_runner.py` has 2 gaps vs `main.py`:

**Gap A: No cross-region routing.** All entities use the target's PIT client. For AAPL (US target), this works because most linked entities (MSFT, GOOGL, Samsung ADR) are accessible via SEC EDGAR. But for non-US targets (Samsung on kr_dart), linked entities like TSMC (tw_mops) or Apple (us_sec_edgar) need their own market's PIT client.

Main.py handles this at lines 1908-1916:
```python
_ent_market_id = ent_info.get("market_id", "")
if _ent_market_id and _ent_market_id != market_id:
    try:
        _ent_client = _create_pit_client(_ent_market_id, secrets)
    except Exception:
        pass  # fall back to target's client
```

**Gap B: No ownership contagion re-run.** After fetching linked entity data, main.py (lines 2014-2065) does:
1. Fetch competitor holders
2. Compute ownership contagion (MHHI, crowding, liquidation)
3. Re-run graph risk with ownership edge weights

The backtest_runner fix skips all 3.

### Proposed Fix

Add to `backtest_runner.py` after the entity data fetch loop:

**Part A: Cross-region routing (~10 lines)**
```python
def _fetch_linked(ent_info):
    _eid = ent_info["id"]
    try:
        _cl = pit_client  # default: target's client
        _mkt = ent_info.get("market_id", "")
        if _mkt and _mkt != state.market_id:
            try:
                from operator1.clients.equity_provider import create_pit_client as _cpc
                _cl = _cpc(_mkt, state._secrets)
            except Exception:
                pass  # fall back to target's client
        # ... rest of fetch logic using _cl ...
```

**Part B: Ownership contagion (~25 lines, after data fetch)**
```python
# Ownership contagion (parity with main.py lines 2014-2065)
if state.target_holders:
    try:
        _comp_holders = {}
        _comp_ids = _entity_groups.get("competitors", [])
        for _cid in _comp_ids[:5]:
            try:
                _ch = pit_client.get_holders(_cid)
                if _ch:
                    _comp_holders[_cid] = _ch
            except Exception:
                pass
        from operator1.models.ownership_contagion import (
            compute_ownership_contagion, inject_contagion_into_cache,
            get_ownership_edge_weights,
        )
        state.contagion_result = compute_ownership_contagion(
            target_holders=state.target_holders,
            competitor_holders=_comp_holders, cache=cache,
        )
        if state.contagion_result and state.contagion_result.available:
            cache = inject_contagion_into_cache(cache, state.contagion_result)
            # Re-run graph risk with ownership edge weights
            _ow = get_ownership_edge_weights(state.contagion_result)
            if _ow and state.graph_risk_result:
                from operator1.models.graph_risk import compute_graph_risk_metrics as _grc
                state.graph_risk_result = _grc(
                    target_isin=state.target_profile.get("isin", ticker),
                    relationships=rel_dicts, edge_weights=_ow,
                    target_cache=cache, linked_caches=state.linked_caches or None,
                )
    except Exception:
        pass
```

### Files to Modify

1. `backtest_runner.py` -- add cross-region routing to `_fetch_linked` function (~10 lines)
2. `backtest_runner.py` -- add ownership contagion block after data fetch (~25 lines)

### Impact

- Cross-region routing: enables non-US backtests to fetch linked entity data correctly
- Ownership contagion: activates `institutional_ownership_analysis` in profile, enables enhanced graph risk with ownership edge weights

### Risk

**Low.** Both additions are wrapped in try/except. Cross-region client creation failure falls back to target's client. Contagion failure is logged and skipped.

---

## Implementation Order

1. Bug 1 refinement (COGS simplification): 1 file, ~5 lines changed
2. Bug 3 Part A (cross-region routing): 1 file, ~10 lines added
3. Bug 3 Part B (ownership contagion): 1 file, ~25 lines added
4. Total: 1-2 files, ~40 lines
