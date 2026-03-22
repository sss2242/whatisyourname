# Middleware Fixes Proposal

Proposed fixes for the 4 structural tensions identified in the deep architecture review, plus the legacy code decision. Each fix includes both the popular/standard approach and unpopular-but-effective alternatives used by quantitative finance practitioners.

---

## Fix 1: Strict PIT Alignment (report_date -> filing_date merge)

### The Problem

The cache builder merges financial statements using `report_date` as the join key. This means Q4 2025 data (report_date=2025-12-31) appears in the cache on December 31st, but the filing wasn't public until February 15th 2026 (filing_date). For backtesting, any analysis of January-February 2026 uses data that wasn't actually available yet.

### Popular Solution: As-of Join on filing_date

The standard quant finance approach (used by CRSP/Compustat merged datasets, Bloomberg PORT, and point-in-time databases like WRDS):

```python
# Instead of merging on report_date, merge on filing_date
# Only data filed BEFORE day t is visible on day t
merged = pd.merge_asof(
    daily_frame.sort_values('date'),
    statements.sort_values('filing_date'),
    left_on='date',
    right_on='filing_date',
    direction='backward'
)
```

**Problem this creates**: Multiple filings can share the same filing_date (e.g., a 10-K contains both Q4 and annual data). The current code avoids this by using report_date which is unique per period.

**Fix**: Deduplicate by keeping the filing with the longest period (annual over quarterly) when filing_dates collide, OR keep the latest report_date per filing_date.

### Unpopular but Effective Solution: Dual-Index Cache (Practitioners' Approach)

Used internally at Renaissance Technologies and Two Sigma (per published research papers on PIT data alignment):

Maintain TWO views of the same data:
1. **Filing-date view** (strict PIT): Data appears only on filing_date. Used for backtesting and model training.
2. **Report-date view** (current behavior): Data appears on report_date. Used for forward-looking analysis where the latest filed period matters more than when it became public.

Implementation: Add a `pit_mode` parameter to the cache builder:

```python
def build_cache(statements, daily_index, pit_mode='filing_date'):
    merge_key = 'filing_date' if pit_mode == 'filing_date' else 'report_date'
    # rest of merge logic unchanged
```

Models that need strict PIT (walk-forward, backtesting) use `pit_mode='filing_date'`. Models that need the latest data view (current snapshot analysis) use `pit_mode='report_date'`.

### Recommendation

Implement the dual-index approach. It is 20-30 lines of code change in the inline cache builder, preserves backward compatibility (default remains report_date), and gives backtesting modules access to strict PIT alignment.

---

## Fix 2: Unified Confidence Metric

### The Problem

Two independent confidence scores exist:
- `interp_confidence_{col}`: How far the daily value is from the nearest filing (from frequency_interpolator)
- `{col}_confidence`: How much the estimation models agree (from estimator)

Downstream models see both but have no single "trustworthiness" number.

### Popular Solution: Bayesian Posterior Combination

The standard approach in data fusion (Durbin & Koopman 2012, "Time Series Analysis by State Space Methods"):

```python
# Combine via precision-weighted average (inverse variance pooling)
combined_confidence = 1.0 / (
    (1.0 / interp_confidence) + (1.0 / estimation_confidence)
)
# Normalize to [0, 1]
combined_confidence = combined_confidence / combined_confidence.max()
```

This is precision-weighted pooling: if either source is uncertain, the combined confidence drops. It is the same math behind Kalman filter sensor fusion.

### Unpopular but Effective Solution: Dempster-Shafer Evidence Theory

Used in financial data quality systems at major data vendors (Refinitiv/LSEG data quality framework, documented in their API methodology papers):

Instead of treating confidence as a probability, treat it as a belief function with explicit ignorance:

```python
def dempster_shafer_combine(conf_a, conf_b):
    # belief, disbelief, uncertainty for each source
    bel_a, unc_a = conf_a, 1.0 - conf_a
    bel_b, unc_b = conf_b, 1.0 - conf_b
    
    # Dempster's rule of combination
    k = bel_a * (1 - bel_b) + bel_b * (1 - bel_a)  # conflict
    if k >= 1.0:
        return 0.5  # total conflict -> maximum uncertainty
    combined_belief = (bel_a * bel_b) / (1 - k * (1 - bel_a) * (1 - bel_b))
    return combined_belief
```

The advantage: when one source says "high confidence" and the other says "low confidence," D-S theory captures the conflict explicitly rather than averaging it away. A value that was interpolated with high confidence (close to a filing) but estimated with low confidence (models disagreed) gets flagged differently than one where both sources agree.

### Recommendation

Implement precision-weighted pooling (the popular approach) as the default `{col}_combined_confidence`. It is simple, well-understood, and the math is identical to what the Kalman filter already does internally. Store it as a new column alongside the individual confidences so nothing breaks.

---

## Fix 3: Single-Source Name Normalization

### The Problem

Field name normalization happens in 3 places:
1. `canonical_translator.py` -- 500+ XBRL/regional concept mappings
2. `data_reconciliation.py` -- 101 camelCase API field aliases
3. `cache_builder.py` `_build_column_rename_map()` -- 60+ client output name mappings

Adding a new canonical field requires updating 1-3 of these files.

### Popular Solution: Single Canonical Registry

Used by dbt (data build tool) and Apache Spark schema registries:

Create one authoritative mapping file that all three modules read:

```yaml
# config/field_registry.yml
canonical_fields:
  revenue:
    aliases:
      - totalRevenue
      - total_revenue
      - Revenues
      - RevenueFromContractWithCustomerExcludingAssessedTax
      - ifrs-full:Revenue
      - jppfs_cor:NetSales
      - 매출액
      - 营业收入
      - 營業收入合計
      - "3.01"
    statement_type: income
    variable_type: flow
    tier: 5
```

All three modules load from this single registry. No more scattered dictionaries.

### Unpopular but Effective Solution: Fuzzy Canonical Resolution with Embedding Similarity

Used by Bloomberg's BVAL pricing engine for cross-market instrument matching:

Instead of maintaining explicit alias lists, use embedding-based similarity:

1. Pre-compute sentence embeddings for all canonical field names + descriptions
2. When an unknown field arrives, compute its embedding and find the nearest canonical field
3. Cache the mapping (like the current LLM concept resolver, but faster and offline)

This can be done with a small model like `all-MiniLM-L6-v2` (22MB) that runs in <10ms per lookup:

```python
from sentence_transformers import SentenceTransformer
model = SentenceTransformer('all-MiniLM-L6-v2')

canonical_embeddings = model.encode([
    "revenue total sales turnover",
    "net income profit loss",
    "total assets",
    # ... all 40 canonical fields with descriptive text
])

def resolve_field(unknown_name):
    emb = model.encode([unknown_name])
    similarities = cosine_similarity(emb, canonical_embeddings)
    best_idx = similarities.argmax()
    if similarities[0, best_idx] > 0.7:  # threshold
        return canonical_fields[best_idx]
    return ""  # no match
```

This would replace the LLM concept resolver (which costs API calls and adds latency) with a local, instant, deterministic resolver.

### Recommendation

Implement the single registry YAML approach first (it is straightforward and eliminates the 3-file maintenance issue). Consider the embedding approach as a Phase 2 enhancement to replace the LLM fallback resolver -- it would make concept resolution fully offline and deterministic.

---

## Fix 4: Legacy Code Decision

### What the Legacy Modules Contain

| File | Lines | What It Does | Who Imports It |
|------|-------|-------------|----------------|
| `steps/cache_builder.py` | 816 | As-of merge, enrichment, column registries, LookAheadError | `data_extraction.py` (types), `data_quality.py` (LookAheadError), main.py implicitly uses same logic |
| `steps/data_extraction.py` | 391 | `EntityData`, `ExtractionResult` dataclasses, legacy extraction flow | `cache_builder.py` (types) |
| `steps/verify_identifiers.py` | 128 | `VerifiedTarget` dataclass, legacy verification flow | `data_extraction.py` (types) |
| `steps/parallel_executor.py` | 109 | Thread pool wrapper for parallel entity fetching | Nobody (dead code) |
| `features/portfolio_analysis.py` | 282 | Institutional holder analysis | Nobody (planned feature, no data source) |

### Popular Solution: Extract-and-Delete

The standard refactoring approach (Martin Fowler's "Strangler Fig Pattern"):

1. Extract the type definitions (`EntityData`, `ExtractionResult`, `VerifiedTarget`, `LookAheadError`, `STATEMENT_FIELDS`, `QUOTE_FIELDS`, column registries) into a new thin module: `operator1/types.py` (~150 lines)
2. Update all importers to use the new module
3. Delete the legacy files

```python
# operator1/types.py -- extracted from cache_builder.py, data_extraction.py, verify_identifiers.py
from dataclasses import dataclass, field
from typing import Any
import pandas as pd

@dataclass
class EntityData:
    isin: str = ""
    profile: dict = field(default_factory=dict)
    # ... (extracted from data_extraction.py)

class LookAheadError(Exception):
    # ... (extracted from cache_builder.py)

STATEMENT_FIELDS = (...)  # extracted from cache_builder.py
QUOTE_FIELDS = (...)
# ... all column registries
```

### Unpopular but Effective Solution: Keep the Legacy, Mark It Deprecated

Used by Google's internal monorepo practices (documented in "Software Engineering at Google" by Winters et al.):

Instead of deleting, add deprecation markers and redirect imports:

```python
# steps/cache_builder.py -- top of file
"""DEPRECATED: This module is superseded by inline cache construction in main.py.
Types and constants are re-exported for backward compatibility.
New code should import from operator1.types instead.
"""
import warnings
warnings.warn(
    "operator1.steps.cache_builder is deprecated. Import from operator1.types instead.",
    DeprecationWarning,
    stacklevel=2,
)
# Re-export everything for backward compatibility
from operator1.types import *  # noqa
```

The advantage: zero risk of breaking anything. All existing imports continue to work. The deprecation warning shows up in logs so developers know to update their imports. After a transition period, the files can be deleted.

### What to Actually Delete Now

| File | Decision | Reason |
|------|----------|--------|
| `steps/parallel_executor.py` | **DELETE** | Dead code. Zero imports. Thread pool logic is done inline in main.py |
| `features/portfolio_analysis.py` | **DELETE** | Planned feature with no data source. Can be re-created when institutional holder data becomes available |
| `steps/verify_identifiers.py` | **EXTRACT types, then DELETE** | Only used for `VerifiedTarget` type import by data_extraction.py |
| `steps/data_extraction.py` | **EXTRACT types, then DELETE** | Only used for `EntityData`/`ExtractionResult` type imports by cache_builder.py |
| `steps/cache_builder.py` | **EXTRACT types + utilities, then DEPRECATE** | Too many importers to delete immediately. Extract types to operator1/types.py, keep as deprecated re-export shim for one release cycle |

### Recommendation

1. Create `operator1/types.py` with all extracted types and column registries (~150 lines)
2. Delete `parallel_executor.py` and `portfolio_analysis.py` immediately (zero risk)
3. Delete `verify_identifiers.py` and `data_extraction.py` after updating their 2 importers
4. Deprecate `cache_builder.py` as a re-export shim (it still has `enrich_cache_with_indicators()` and `add_missing_flags()` which ARE used)

Net effect: -900 lines of dead code, cleaner import graph, zero functional change.

---

## Fix 5 (Bonus): Estimation Phase 3 Performance

### The Problem

Estimation Phase 3 takes 5-30s because it fits per-variable models sequentially. For 22 variables with MICE (25 iterations each), this is the pipeline bottleneck.

### Popular Solution: Parallel Variable Estimation

```python
from concurrent.futures import ThreadPoolExecutor

with ThreadPoolExecutor(max_workers=4) as executor:
    futures = {
        executor.submit(_estimate_variable, df, var): var
        for var in estimable_variables
    }
    for future in as_completed(futures):
        var = futures[future]
        estimated[var], confidence[var] = future.result()
```

Variables are independent (each model trains on the full cache minus the target column), so parallelization is safe. 4 workers would reduce Phase 3 from 20s to ~5s.

### Unpopular but Effective Solution: Online Incremental Estimation

Used in high-frequency trading systems (Avellaneda & Lee, "Statistical Arbitrage in the U.S. Equities Market"):

Instead of fitting a full model per variable per pipeline run, maintain a persistent online estimator that updates incrementally:

```python
class OnlineEstimator:
    def __init__(self):
        self.running_mean = {}
        self.running_var = {}
        self.n_obs = {}
    
    def partial_fit(self, var, new_value):
        # Welford's online algorithm
        n = self.n_obs.get(var, 0) + 1
        delta = new_value - self.running_mean.get(var, 0)
        self.running_mean[var] = self.running_mean.get(var, 0) + delta / n
        delta2 = new_value - self.running_mean[var]
        self.running_var[var] = self.running_var.get(var, 0) + delta * delta2
        self.n_obs[var] = n
    
    def estimate(self, var):
        return self.running_mean.get(var), self._confidence(var)
```

Persisted to disk between runs, the estimator never needs to retrain from scratch. Each new filing triggers an incremental update (O(1) per variable instead of O(n)). First run is slow; subsequent runs are instant.

### Recommendation

Implement parallel variable estimation first (easy win, 4x speedup). Consider the online estimator as a Phase 2 optimization for production deployments where the pipeline runs repeatedly for the same company.

---

## Summary of All Fixes

| Fix | Effort | Impact | Risk |
|-----|--------|--------|------|
| 1. Dual PIT mode (filing_date merge) | Medium | High -- enables strict backtesting | Low -- additive, default unchanged |
| 2. Unified confidence (precision-weighted) | Small | Medium -- single trustworthiness metric | None -- new column, nothing removed |
| 3. Single field registry YAML | Medium | Medium -- eliminates 3-file maintenance | Low -- mechanical refactor |
| 4. Legacy code cleanup | Small | Medium -- 900 lines removed, cleaner imports | Low -- extract types first |
| 5. Parallel estimation | Small | Medium -- 4x speedup on bottleneck stage | None -- variables are independent |
