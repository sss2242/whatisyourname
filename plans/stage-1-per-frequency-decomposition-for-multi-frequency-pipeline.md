# Stage 1 Per-Frequency Decomposition for Multi-Frequency Pipeline

## Problem

The multi-frequency pipeline (Stage 7.4) needs frequency-specific financial statement data -- actual annual filings for the Annual pipeline, actual quarterly filings for the Quarterly pipeline. But today, main.py Step 3d runs `separate_by_period_type()` which correctly splits mixed-freq DFs into per-frequency groups, then **throws them away** after building a reconciled highest-freq version via `build_highest_frequency_statement()`. Only the reconciled DF (quarterly Chow-Lin disaggregated from annual + original quarterly) survives to Stage 7.4.

This means:
- **Annual pipeline** receives quarterly-reconciled data, not actual annual filings. `build_cache_from_raw_filings(frequency="A")` sees quarterly-spaced dates and doesn't interpolate, producing a weird hybrid.
- **Quarterly pipeline** receives Chow-Lin disaggregated rows mixed with actual quarterly rows. Can't distinguish real vs synthetic.
- **The daily cache** (main.py Step 4) correctly uses the reconciled highest-freq version -- that's the right choice for daily forward-fill.

## Key Insight

The frequency separator ALREADY does the hard work. `separate_by_period_type()` at line 849 produces:
```python
freq_groups = {"quarterly": df_quarterly, "annual": df_annual}  
# or {"quarterly": df_q, "semiannual": df_s, "annual": df_a}
```

We just need to **preserve these groups** alongside the reconciled version, and pass them to Stage 7.4 so each frequency pipeline gets its native source data.

## Design

### Current Flow
```
PIT Client -> mixed-freq DFs
    |
    v
Step 3d: separate_by_period_type() -> freq_groups (DISCARDED)
    |                                       |
    v                                       v
build_highest_frequency_statement()    (thrown away)
    |
    v
reconciled income_df/balance_df/cashflow_df
    |
    v
Step 4: Build daily cache from reconciled DFs
    |
    v
Stage 7.4: build_cache_from_raw_filings(reconciled DFs, freq=A/Q/M/W)
           ^-- wrong: annual pipeline gets quarterly data
```

### New Flow
```
PIT Client -> mixed-freq DFs
    |
    v
Step 3d: separate_by_period_type() -> freq_groups (PRESERVED)
    |                                       |
    v                                       v
build_highest_frequency_statement()    Saved to PipelineState:
    |                                  income_freq_groups = {"quarterly": df, "annual": df}
    v                                  balance_freq_groups = {...}
reconciled income_df/balance_df/cashflow_df    cashflow_freq_groups = {...}
    |
    v
Step 4: Build daily cache from reconciled DFs (unchanged)
    |
    v
Stage 7.4.0: For each freq, use the CORRECT frequency group:
  - freq="A" -> use freq_groups["annual"]
  - freq="Q" -> use freq_groups["quarterly"]
  - freq="S" -> use freq_groups["semiannual"]
  - freq="W" -> interpolate from reconciled DFs (no native W filings)
  - freq="M" -> interpolate from reconciled DFs (no native M filings)
  - freq="D" -> daily cache as-is
```

## Implementation Steps

### Step 1: Add freq_groups fields to PipelineState

**File:** `operator1/pipeline_state.py`

Add 3 new fields to store the per-frequency statement groups:
```python
# Per-frequency raw statement groups (from frequency separator Step 3d)
# Keys: "quarterly", "semiannual", "annual"
# Values: DataFrames with actual filing data for that frequency only
self.income_freq_groups: dict[str, pd.DataFrame] = {}
self.balance_freq_groups: dict[str, pd.DataFrame] = {}
self.cashflow_freq_groups: dict[str, pd.DataFrame] = {}
```

Add serialization: save each group as a separate parquet file in `freq_groups/` subdirectory. Skip in pickle (DataFrames too large).

### Step 2: Preserve freq_groups in main.py Step 3d

**File:** `main.py` (around line 844)

Before the reconciliation loop, initialize storage:
```python
_income_freq_groups = {}
_balance_freq_groups = {}
_cashflow_freq_groups = {}
```

Inside the loop, save the groups before reconciling:
```python
freq_groups = separate_by_period_type(stmt, market_id=market_id)
# Preserve per-frequency groups for Stage 7.4 MF pipeline
if label == "income":
    _income_freq_groups = freq_groups
elif label == "balance":
    _balance_freq_groups = freq_groups
else:
    _cashflow_freq_groups = freq_groups
# Continue with reconciliation as before...
```

Copy to PipelineState before Stage 6:
```python
_ps.income_freq_groups = _income_freq_groups
_ps.balance_freq_groups = _balance_freq_groups
_ps.cashflow_freq_groups = _cashflow_freq_groups
```

### Step 3: Use freq_groups in run_7_4_0_resample_prep()

**File:** `operator1/stages/stage7_integration.py`

Replace the current logic that passes the same DFs to all frequencies:

```python
# Map pipeline freq code -> separator freq label
_FREQ_CODE_TO_LABEL = {"Q": "quarterly", "S": "semiannual", "A": "annual"}

for freq in frequencies:
    _sep_label = _FREQ_CODE_TO_LABEL.get(freq)
    
    if _sep_label and _has_freq_groups:
        # A/Q/S: Use frequency-specific raw filings (actual filing data)
        _inc = state.income_freq_groups.get(_sep_label)
        _bal = state.balance_freq_groups.get(_sep_label)
        _cf = state.cashflow_freq_groups.get(_sep_label)
        _any_freq_data = any(
            df is not None and not df.empty 
            for df in [_inc, _bal, _cf]
        )
        if _any_freq_data:
            resampled = build_cache_from_raw_filings(
                income_df=_inc, balance_df=_bal,
                cashflow_df=_cf, quotes_df=quotes_df,
                frequency=freq, reference_date=ref_date,
            )
            logger.info(
                "[%s] Built from native %s filings: inc=%d, bal=%d, cf=%d",
                freq, _sep_label,
                len(_inc) if _inc is not None else 0,
                len(_bal) if _bal is not None else 0,
                len(_cf) if _cf is not None else 0,
            )
        else:
            # No native filings for this frequency
            # For A: this is unexpected (most markets have annual filings)
            # For Q: possible for annual-only markets
            # For S: possible for quarterly-only markets
            logger.warning(
                "[%s] No native %s filings found in freq_groups -- "
                "falling back to resampled daily cache",
                freq, _sep_label,
            )
            resampled = resample_cache_to_frequency(cache, frequency=freq, ...)
            
    elif freq in ("W", "M") and _has_raw:
        # W/M: No native filings exist at these frequencies.
        # Interpolate from reconciled (highest-freq) DFs.
        resampled = build_cache_from_raw_filings(
            income_df=income_df, balance_df=balance_df,
            cashflow_df=cashflow_df, quotes_df=quotes_df,
            frequency=freq, reference_date=ref_date,
        )
    elif freq == "D":
        # Daily: use daily cache as-is
        resampled = resample_cache_to_frequency(cache, frequency="D", ...)
    else:
        # Fallback: resample daily cache (degraded)
        resampled = resample_cache_to_frequency(cache, frequency=freq, ...)
```

### Step 4: Serialize freq_groups through checkpoints

**File:** `operator1/pipeline_state.py`

In `save()`, serialize freq_groups alongside other statement DFs:
```python
for grp_name in ("income_freq_groups", "balance_freq_groups", "cashflow_freq_groups"):
    grp = getattr(self, grp_name, {})
    if grp:
        grp_dir = run_dir / "freq_groups" / grp_name
        grp_dir.mkdir(parents=True, exist_ok=True)
        for freq_label, df in grp.items():
            if df is not None and not df.empty:
                df.to_parquet(grp_dir / f"{freq_label}.parquet")
```

In `_load_state()`, restore:
```python
for grp_name in ("income_freq_groups", "balance_freq_groups", "cashflow_freq_groups"):
    grp_dir = run_dir / "freq_groups" / grp_name
    if grp_dir.exists():
        grp = {}
        for pq in grp_dir.glob("*.parquet"):
            grp[pq.stem] = pd.read_parquet(pq)
        setattr(self, grp_name, grp)
```

### Step 5: Add _has_freq_groups check

**File:** `operator1/stages/stage7_integration.py`

```python
_has_freq_groups = any(
    bool(getattr(state, grp, {}))
    for grp in ("income_freq_groups", "balance_freq_groups", "cashflow_freq_groups")
)
if _has_freq_groups:
    logger.info(
        "Per-frequency filing groups available: income=%s, balance=%s, cashflow=%s",
        list(state.income_freq_groups.keys()),
        list(state.balance_freq_groups.keys()),
        list(state.cashflow_freq_groups.keys()),
    )
```

## Files Modified

| File | Change |
|------|--------|
| `operator1/pipeline_state.py` | Add 3 freq_groups fields + save/load serialization |
| `main.py` | Preserve freq_groups from Step 3d, copy to PipelineState |
| `operator1/stages/stage7_integration.py` | Use freq-specific groups in resample prep |

## Backward Compatibility

- When freq_groups are empty (old checkpoints, single-freq markets), the existing fallback paths still work
- The reconciled DFs (income_df, balance_df, cashflow_df) are unchanged -- daily cache build is not affected
- Weekly/Monthly frequencies continue to use interpolation from reconciled DFs (correct behavior, no native W/M filings exist)

## What This Fixes

Before: Annual MF pipeline runs on quarterly-reconciled data (Chow-Lin disaggregated + original quarterly merged). Revenue growth = 0% because every annual period sees the same quarterly-total value.

After: Annual MF pipeline runs on actual annual filings. Revenue growth computed from real annual reported values. Survival mode sees actual annual ratios, not interpolated quarterly approximations.
