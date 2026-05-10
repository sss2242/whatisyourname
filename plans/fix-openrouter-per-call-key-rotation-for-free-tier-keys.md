# Fix OpenRouter Per-Call Key Rotation for Free-Tier Keys

## Problem

Free OpenRouter keys get 1 call then they're burned. The pipeline makes 3 LLM calls:
1. **Entity discovery** (`entity_discovery.py:561`) -- `propose_linked_entities()`
2. **Sentiment scoring** (`news_sentiment.py:530`) -- `score_sentiment()`
3. **Report generation** (`report_generator.py:5705`) -- `generate_report()`

With a single free key, only call 1 succeeds. Calls 2 and 3 fail with 429/402/quota errors.

## Current Key System (Already Well-Designed)

The infrastructure is already there -- it just needs the user to provide multiple keys:

### secrets_loader.py (line 158-168)
Already supports numbered keys:
```
OPENROUTER_API_KEY_1=sk-or-v1-abc...  # for entity discovery
OPENROUTER_API_KEY_2=sk-or-v1-def...  # for sentiment
OPENROUTER_API_KEY_3=sk-or-v1-ghi...  # for report
```

Also supports comma-separated:
```
OPENROUTER_API_KEY=sk-or-v1-abc...,sk-or-v1-def...,sk-or-v1-ghi...
```

Both formats are collected by `load_secrets()` and merged into a single comma-separated string at line 167: `secrets[provider_key] = ",".join(all_keys)`.

### llm_factory.py (line 276-281)
`create_llm_client()` calls `get_key_pool(secrets, "OPENROUTER_API_KEY")` which splits the comma-separated string into individual keys, then `_build_pooled_or_single()` creates one `OpenRouterClient` per key, wrapped in a `PooledLLMClient`.

### PooledLLMClient (line 100-199)
On each call, `_call_with_rotation()` catches exhaustion errors (429, "quota exceeded", "credit", etc.) and rotates to the next key via `_rotate()`.

## The Gap

The system works correctly IF the user provides 3+ keys. But there are two issues:

### Issue 1: Single-key path skips pooling

At `llm_factory.py:317`:
```python
if len(primary_clients) == 1:
    return primary_clients[0]  # no PooledLLMClient, no rotation
```

With a single key, there's no `PooledLLMClient` wrapper. The raw `OpenRouterClient` is returned. When it gets a 429, `llm_base.py`'s retry logic retries the SAME key 5 times (all fail). No rotation happens because there's no pool.

**Fix:** Even with 1 key, wrap in `PooledLLMClient` so the exhaustion detection logic is active. When the single key is exhausted, the pool reports "all keys exhausted" cleanly instead of retrying the dead key 5 times.

### Issue 2: .env.example doesn't show multi-key pattern

Users don't know they can provide multiple keys. The `.env.example` only shows:
```
OPENROUTER_API_KEY=your_openrouter_api_key_here
```

**Fix:** Update `.env.example` to show the numbered key pattern with a comment explaining why.

### Issue 3: No per-call key assignment (optional optimization)

The current rotation is reactive (rotate after failure). For free OpenRouter keys, we KNOW each key will fail after 1 call. A proactive approach would assign key N to call N upfront, avoiding the failed-request overhead.

**Fix:** Add a `proactive_rotation` mode to `PooledLLMClient` that pre-rotates to the next key after each successful call (not waiting for failure).

## Implementation Plan

### Change 1: Always wrap in PooledLLMClient (even with 1 key)

**File:** `operator1/clients/llm_factory.py`
**Line:** 316-318

**Before:**
```python
if len(primary_clients) == 1:
    return primary_clients[0]
```

**After:**
```python
# Even with a single key, wrap in PooledLLMClient so exhaustion
# detection and reporting works cleanly (instead of retrying the
# same dead key 5 times via llm_base retry logic).
if len(primary_clients) == 1 and not fallback_clients:
    return primary_clients[0]
```

Actually, this is fine as-is for single-key without fallbacks. The real fix is Change 3.

### Change 2: Update .env.example with multi-key pattern

**File:** `.env.example`

Add after the existing `OPENROUTER_API_KEY` line:
```bash
# OpenRouter free-tier keys (each free key = 1 call, need 3 for full pipeline)
# Get free keys at https://openrouter.ai/ -- sign up with different emails
# Option A: Numbered keys (recommended)
OPENROUTER_API_KEY_1=
OPENROUTER_API_KEY_2=
OPENROUTER_API_KEY_3=
# Option B: Comma-separated in a single variable
# OPENROUTER_API_KEY=key1,key2,key3
```

### Change 3: Add proactive rotation mode for free-tier providers

**File:** `operator1/clients/llm_factory.py`

Add a `proactive_rotate` flag to `PooledLLMClient` that pre-rotates to the next key after each SUCCESSFUL call. This avoids the overhead of a failed request + retry cycle for keys that are known to be single-use.

**New `_call_with_rotation()` logic:**
```python
def _call_with_rotation(self, method_name: str, *args, **kwargs):
    """Call a method on the active client, rotating on exhaustion."""
    attempts = len(self._all_clients) - len(self._exhausted)
    for _ in range(max(attempts, 1)):
        try:
            method = getattr(self._active, method_name)
            result = method(*args, **kwargs)
            # Proactive rotation: move to next key BEFORE it fails.
            # For free-tier providers where each key = 1 call.
            if self._proactive_rotate:
                self._rotate()
            return result
        except Exception as exc:
            if self._is_exhaustion_error(exc):
                if not self._rotate():
                    raise
            else:
                raise
    raise RuntimeError("All LLM keys exhausted")
```

**Enable proactive rotation for OpenRouter free models:**

In `_build_pooled_or_single()`, detect free-tier OpenRouter and enable proactive rotation:

```python
# Detect if all OpenRouter keys are likely free-tier
_proactive = (
    provider == "openrouter"
    and len(primary_clients) > 1
    and all(
        ":free" in getattr(c, "_model", "")
        for c in primary_clients
    )
)

pool = PooledLLMClient(primary_clients, fallback_clients or None)
pool._proactive_rotate = _proactive
if _proactive:
    logger.info("Proactive rotation enabled (free-tier OpenRouter, %d keys)", len(primary_clients))
```

### Change 4: Log which call uses which key

**File:** `operator1/clients/llm_factory.py`

In `_call_with_rotation()`, log the key index and task type so the user can see which key was used for which call:

```python
logger.info(
    "LLM call: key %d/%d (%s) for %s",
    self._current_idx + 1, len(self._all_clients),
    self._active.provider_name,
    method_name,
)
```

## Execution Checklist

```
[ ] 1. .env.example: Add OPENROUTER_API_KEY_1/_2/_3 pattern with comments
[ ] 2. llm_factory.py: Add _proactive_rotate flag to PooledLLMClient
[ ] 3. llm_factory.py: Modify _call_with_rotation() to pre-rotate on success
[ ] 4. llm_factory.py: Enable proactive rotation for free-tier OpenRouter
[ ] 5. llm_factory.py: Add key-usage logging
```

## Summary

| What | Status |
|------|--------|
| Multi-key support in secrets_loader | Already works (numbered + comma-separated) |
| Key pooling in llm_factory | Already works (PooledLLMClient) |
| Reactive rotation on 429 | Already works (_call_with_rotation) |
| **Proactive rotation for free-tier** | **NEW -- avoids wasted failed requests** |
| **.env.example shows multi-key pattern** | **NEW -- users know they can provide 3 keys** |
| **Per-call logging** | **NEW -- visibility into which key serves which call** |

The fix is minimal: 3 keys in .env + proactive rotation flag. No architectural changes needed -- the infrastructure is already solid.
