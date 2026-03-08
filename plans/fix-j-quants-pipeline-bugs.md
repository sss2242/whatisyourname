# Fix J-Quants Pipeline Bugs

4 bugs identified from the Toyota Motor (7203) pipeline run on 2026-03-07.

---

## Bug 1: Wrong company returned (CRITICAL)

**Symptom:** User requests Toyota (7203), gets KYOKUYO (1301).

**Root cause:** `get_profile()` at line 214 normalizes the 4-digit ticker to 5 digits by appending "0" (7203 -> 72030). Then it searches the equity master DataFrame. The V2 API returns codes like "72030" for Toyota, but the search logic may be matching on partial strings or sorting incorrectly, returning code "13010" (KYOKUYO) first.

**Fix:**

```python
# In get_profile(), line 214-216:
# Current (broken):
code = identifier.strip()
if len(code) == 4:
    code = code + "0"  # 7203 -> 72030

# The issue is likely in the master DataFrame filtering.
# Fix: use exact match on the 5-digit code, not startswith/contains.
master = self._client.get_eq_master()
match = master[master["Code"] == code]  # EXACT match, not partial
if match.empty and len(code) == 5:
    # Try without trailing zero (some codes don't follow the pattern)
    match = master[master["Code"].str[:4] == identifier.strip()[:4]]
```

Also add a validation check after profile fetch:

```python
# After fetching profile, verify the returned company matches the requested ticker
if profile.get("ticker") and profile["ticker"] != identifier[:4]:
    logger.error(
        "Profile mismatch: requested %s but got %s (%s)",
        identifier, profile["ticker"], profile.get("name"),
    )
```

**Files:** `operator1/clients/jp_jquants_wrapper.py` (get_profile method)

---

## Bug 2: Empty ticker in downstream calls (CRITICAL)

**Symptom:** `Target: TOYOTA MOTOR CORPORATION () via J-Quants` -- empty ticker; all financial calls fail with empty identifier.

**Root cause chain:**
1. Company search returns `{"name": "TOYOTA MOTOR CORPORATION", "identifier": "7203", "ticker": ""}` -- ticker field empty
2. `main.py:578` sets `ticker = company_info.get("ticker", "")` = ""
3. `main.py:580` sets `identifier = company_info.get("cik") or ticker` = "" (no CIK for J-Quants)
4. All subsequent calls use empty identifier

**Fix (2 parts):**

**Part A** -- Fix the company search to include ticker in results:

```python
# In search_companies(), around line 186:
results.append({
    "identifier": str(row.get("Code", ""))[:4],
    "name": name,
    "ticker": str(row.get("Code", ""))[:4],  # ADD THIS
    ...
})
```

**Part B** -- Fix main.py identifier fallback:

```python
# main.py line 580:
# Current:
identifier = company_info.get("cik") or ticker
# Fix: also fall back to the identifier field from search results
identifier = company_info.get("cik") or ticker or company_info.get("identifier", "")
```

**Files:** `operator1/clients/jp_jquants_wrapper.py` (search_companies), `main.py` (identifier resolution)

---

## Bug 3: 429 rate limit exhaustion (HIGH)

**Symptom:** `too many 429 error responses` after 6+ minutes of requests.

**Root cause:** The pipeline sends rapid requests during initialization (equity master fetch, profile, company list), exhausting the free plan budget of ~12 req/min. The financial data calls then hit 429s with no recovery.

**Fix:**

1. Add exponential backoff on 429 responses (currently the library just raises):

```python
# In get_financials(), wrap the API call:
for attempt in range(3):
    try:
        _jquants_throttle()
        summary = self._client.get_fin_summary(code=code)
        break
    except requests.HTTPError as e:
        if "429" in str(e):
            wait = _JQUANTS_MIN_INTERVAL * (2 ** attempt)
            logger.warning("J-Quants 429: backing off %.0fs (attempt %d/3)", wait, attempt + 1)
            time.sleep(wait)
        else:
            raise
```

2. Increase `_JQUANTS_MIN_INTERVAL` from 6.0 to 8.0 seconds to give more headroom.

3. Cache the equity master so it's only fetched once per pipeline run (currently fetched on every `get_profile` and `search_companies` call).

**Files:** `operator1/clients/jp_jquants_wrapper.py`

---

## Bug 4: Date-based query instead of company-code-based (HIGH)

**Symptom:** `url: /v2/fins/summary?date=20210308` -- fetches ALL companies for a date.

**Root cause:** The `get_financials()` method calls `self._client.get_fin_summary(date=start_date)` which fetches financial summaries for ALL companies on that date. This is bandwidth-wasteful and causes the 429 errors (each call returns thousands of rows).

**Fix:**

The J-Quants V2 API `get_fin_summary` supports a `code` parameter:

```python
# Current (broken):
summary = self._client.get_fin_summary(date=date_str)

# Fix: filter by company code
summary = self._client.get_fin_summary(code=code)
```

This fetches only the target company's financial summaries across all available dates, instead of all companies for a single date. This:
- Reduces response size from thousands of rows to ~8-10 rows
- Reduces API calls from ~730 (one per day in 2yr window) to 1
- Eliminates the 429 problem

**Files:** `operator1/clients/jp_jquants_wrapper.py` (get_financials method)

---

## Implementation Order

```
Bug 4 (date -> code query)    -- fixes 429 AND reduces calls from 730 to 1
Bug 1 (wrong company)         -- exact match on 5-digit code
Bug 2 (empty ticker)          -- ticker in search results + main.py fallback
Bug 3 (rate limiting)         -- backoff + master caching (partially fixed by Bug 4)
```

Bug 4 should be done first because it's the root cause of Bug 3 (429 errors). Once financial queries use `code=` instead of `date=`, the rate limit problem largely disappears.

---

## Files Changed

| File | Bug | Changes |
|------|-----|---------|
| `operator1/clients/jp_jquants_wrapper.py` | 1, 2, 3, 4 | Fix profile matching, ticker in search results, code-based query, backoff, master cache |
| `main.py` | 2 | Identifier fallback to company_info identifier field |

---

## Testing

- Existing tests: `tests/test_pit_jp_jquants.py` (4 tests, all pass)
- Add new tests:
  - `test_profile_returns_correct_company` -- verify 7203 returns Toyota, not KYOKUYO
  - `test_search_includes_ticker` -- verify search results have non-empty ticker field
  - `test_financials_uses_code_parameter` -- verify API call uses `code=` not `date=`
  - `test_429_backoff` -- verify exponential backoff on rate limit errors
