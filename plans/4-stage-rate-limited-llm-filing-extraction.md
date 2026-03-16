# 4-Stage Rate-Limited LLM Filing Extraction

## Problem

The filing extraction pipeline sends 8 LLM requests in quick succession (one per discovered filing PDF), which exhausts all 5 Gemini free-tier keys within seconds. The Gemini free tier allows 15 RPM per key, so 5 keys = 75 RPM total. But the extraction loop fires all 8 requests as fast as possible with only the per-host rate limiter (0.25 req/s = 15 RPM for a single key) providing backpressure.

With the `PooledLLMClient` rotating keys on 429, the effective rate is 5x faster (1.25 req/s), which works in theory. But the inner retry loop in `llm_base.py` consumes 5 retries per 429 before the pool rotates, burning 30-60 seconds per failed key.

## Current Flow

```
try_filing_extraction() -- called once per ticker, serialized by lock
  for filing in discovery.filings[:8]:     # up to 8 filings
    pdf_bytes = discoverer.download_filing(filing)  # ~1s each, no rate limit issue
    extraction = extractor.extract_from_pdf(pdf_bytes)
      -> pdfplumber text extraction (local, ~2s per 50-page PDF)
      -> _select_top_pages() (local, instant)
      -> _extract_via_llm(text) (1 LLM call per filing)
        -> self._llm.generate(prompt)  # THE BOTTLENECK
```

Each `generate()` call goes through:
1. `PooledLLMClient._call_with_rotation()` -- tries active key
2. `LLMClient._execute_with_retry()` -- 5 retries with 2/4/8/16/32s backoff on 429
3. `_rate_limit_sleep()` -- enforces 0.25 req/s per host (4s between calls)

## Solution: 4-Stage Extraction with Per-Key Budgeting

Split the 8 filings into 4 stages of 2 filings each. Between stages, wait for the rate limit window to reset. This distributes the load evenly across keys and time.

### Stage Design

```
Stage 1: Extract filings [0, 1] using keys [1, 2]    -- ~15s
  wait 15s for rate limit reset
Stage 2: Extract filings [2, 3] using keys [3, 4]    -- ~15s
  wait 15s for rate limit reset
Stage 3: Extract filings [4, 5] using keys [5/1, 1/2] -- ~15s
  wait 15s for rate limit reset
Stage 4: Extract filings [6, 7] using keys [2/3, 3/4] -- ~15s
```

Total time: ~2 minutes (vs current ~5+ minutes of retries and failures)

### Implementation Changes

#### File: `operator1/clients/filing_discoverer.py`

**Change in `try_filing_extraction()`**: Replace the simple loop over 8 filings with staged extraction:

```python
# Current code:
for filing in discovery.filings[:8]:
    pdf_bytes = discoverer.download_filing(filing)
    extraction = extractor.extract_from_pdf(pdf_bytes, market_id=market_id)
    ...

# New code:
STAGE_SIZE = 2          # filings per stage
STAGE_PAUSE_S = 15.0    # seconds between stages (rate limit reset)
MAX_FILINGS = 8         # total filings to extract

filings_to_extract = discovery.filings[:MAX_FILINGS]
stages = [filings_to_extract[i:i + STAGE_SIZE]
          for i in range(0, len(filings_to_extract), STAGE_SIZE)]

for stage_num, stage_filings in enumerate(stages):
    if stage_num > 0:
        logger.info(
            "Filing extraction stage %d/%d: pausing %.0fs for rate limit reset",
            stage_num + 1, len(stages), STAGE_PAUSE_S,
        )
        time.sleep(STAGE_PAUSE_S)

    for filing in stage_filings:
        pdf_bytes = discoverer.download_filing(filing)
        extraction = extractor.extract_from_pdf(pdf_bytes, market_id=market_id)
        ...
```

#### File: `operator1/clients/llm_base.py`

**Reduce inner retry count for 429s**: Currently 5 retries with exponential backoff = up to 62 seconds wasted per failed key. Since the `PooledLLMClient` handles rotation, the inner loop should fail fast on 429:

```python
# Current: max_retries = 5, backoff = 2.0
# Change for 429 specifically: fail after 2 retries (2s + 4s = 6s max)
# so the pool can rotate to the next key quickly.

# In _execute_with_retry():
if resp.status_code == 429:
    # Fail fast on rate limit -- let PooledLLMClient rotate to next key
    if attempt >= 2:
        raise RuntimeError(
            f"{self.provider_name}: rate limited after {attempt} attempts"
        )
```

#### File: `operator1/clients/llm_factory.py`

**Round-robin key selection**: Instead of always starting from key 1, start each `generate()` call from a rotating index:

```python
# In PooledLLMClient:
def _call_with_rotation(self, method_name, *args, **kwargs):
    # Round-robin: start from next key after last successful call
    # This distributes load even when no exhaustion occurs
    start_idx = self._current_idx
    ...
```

### Configuration

Add to `config/global_config.yml`:

```yaml
# Filing extraction rate limiting
filing_extraction_stage_size: 2      # filings per stage
filing_extraction_stage_pause_s: 15  # seconds between stages
filing_extraction_max_filings: 8     # max filings to extract per ticker
```

### Expected Behavior

With 5 Gemini keys and the staged approach:

| Stage | Filings | Keys Used | Time | Cumulative |
|-------|---------|-----------|------|------------|
| 1 | Filing 0, 1 | Key 1, 2 | ~15s | 15s |
| pause | -- | -- | 15s | 30s |
| 2 | Filing 2, 3 | Key 3, 4 | ~15s | 45s |
| pause | -- | -- | 15s | 60s |
| 3 | Filing 4, 5 | Key 5, 1 | ~15s | 75s |
| pause | -- | -- | 15s | 90s |
| 4 | Filing 6, 7 | Key 2, 3 | ~15s | 105s |

Total: ~105 seconds for 8 filings. With the current approach it takes 5+ minutes of retries and still fails.

### Fallback: Prioritize Most Recent Filings

When rate limits are severe (1 key, free tier), extract fewer filings in priority order:

1. Most recent annual results (critical -- has full-year data)
2. Most recent interim results (important -- H1 data)
3. Second most recent annual results (useful -- YoY comparison)
4. Everything else (nice to have)

This ensures even with 1 key at 15 RPM, the 2-3 most important filings get extracted.

### Task Checklist

- [ ] Add staged extraction loop to `try_filing_extraction()` in `filing_discoverer.py`
- [ ] Add stage pause configuration to `global_config.yml`
- [ ] Reduce inner retry count for 429 in `llm_base.py` (fail fast for pool rotation)
- [ ] Add filing priority sorting (annual first, then interim, then quarterly)
- [ ] Test with 5 Gemini free-tier keys on Tencent (00700)
- [ ] Commit and push to PR
