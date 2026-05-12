# LLM Key Rotation Strategy -- User-Configurable Mode

## Problem

The current `PooledLLMClient` in [`llm_factory.py`](operator1/clients/llm_factory.py) has two rotation strategies hardcoded with auto-detection:

1. **Reactive rotation** (`proactive_rotate=False`): Rotates to next key only when the current key fails (429/quota exceeded). Best for **paid keys** -- maximizes usage of each key before switching.

2. **Proactive rotation** (`proactive_rotate=True`): Rotates to next key after every successful call. Best for **free-tier keys** -- avoids wasting a failed request + 5 retry cycles per key transition.

The auto-detection at [`llm_factory.py:384-395`](operator1/clients/llm_factory.py:384) only enables proactive rotation when ALL conditions are met:
- Provider is `openrouter`
- Multiple keys exist
- All models contain `:free` in the name

This means:
- Users with free Gemini/Claude keys get reactive rotation (wrong -- should be proactive)
- Users with paid OpenRouter keys using free models get proactive (potentially wrong)
- Users cannot override the behavior

## Solution

Add `llm_key_rotation` config to [`global_config.yml`](config/global_config.yml) with three modes:

| Mode | Behavior | Best For |
|------|----------|---------|
| `auto` | Current auto-detection logic (`:free` model check) | Default, backward compatible |
| `per_call` | Proactive rotation after every successful call | Free-tier keys (Gemini free, OpenRouter free, Claude trial) |
| `on_failure` | Reactive rotation only on 429/exhaustion | Paid keys with generous rate limits |

## Current Code Flow

```
create_llm_client(secrets)
  -> resolve provider (config/env/auto-detect)
  -> resolve model (config/env/default)
  -> get key pool (comma-separated keys from .env)
  -> _build_pooled_or_single(provider, keys, model, secrets)
       -> if single key: return plain LLMClient (no pooling)
       -> if multiple keys:
            -> build primary_clients (one per key)
            -> build fallback_clients (alt provider keys)
            -> auto-detect _proactive from ":free" in model name  <-- THIS CHANGES
            -> return PooledLLMClient(primary, fallback, proactive_rotate=_proactive)
```

## Files to Change (2 files, ~15 lines)

### File 1: [`config/global_config.yml`](config/global_config.yml) -- Add config key

Add after the `llm_provider` / `llm_model` section:

```yaml
# LLM key rotation strategy for multi-key setups (comma-separated keys in .env):
#   auto       -- auto-detect: proactive for free-tier models, reactive for paid
#   per_call   -- rotate to next key after every successful call (best for free-tier keys)
#   on_failure -- rotate only when a key gets 429/exhausted (best for paid keys)
llm_key_rotation: "auto"
```

### File 2: [`operator1/clients/llm_factory.py`](operator1/clients/llm_factory.py:384-395) -- Read config

Replace the hardcoded auto-detection block at lines 384-395:

**Before:**
```python
    # Detect free-tier OpenRouter: enable proactive rotation so each
    # successful call advances to the next key before it gets a 429.
    _proactive = (
        provider == "openrouter"
        and len(primary_clients) > 1
        and all(
            ":free" in getattr(c, "_model", "")
            for c in primary_clients
        )
    )
```

**After:**
```python
    # Determine key rotation strategy from config
    cfg = get_global_config()
    _rotation_mode = str(cfg.get("llm_key_rotation", "auto")).strip().lower()

    if _rotation_mode == "per_call":
        _proactive = True
    elif _rotation_mode == "on_failure":
        _proactive = False
    else:  # "auto" -- original auto-detection logic
        _proactive = (
            provider == "openrouter"
            and len(primary_clients) > 1
            and all(
                ":free" in getattr(c, "_model", "")
                for c in primary_clients
            )
        )
```

## Data Flow

```
global_config.yml
  llm_key_rotation: "per_call" | "on_failure" | "auto"
       |
       v
_build_pooled_or_single()  [llm_factory.py:384]
       |
       v
PooledLLMClient(proactive_rotate=True/False)  [llm_factory.py:402]
       |
       v
_call_with_rotation()  [llm_factory.py:198]
  - if proactive: _proactive_rotate_next() after every success
  - if reactive: _rotate() only on exhaustion error
```

## No Changes Needed

| File | Why |
|------|-----|
| `PooledLLMClient` class | Already accepts `proactive_rotate` parameter -- no changes |
| `llm_base.py` | Does not handle rotation -- delegates up to PooledLLMClient |
| `openrouter.py` / `gemini.py` / `claude.py` | Individual clients unchanged -- pooling is external |
| `secrets_loader.py` | Key parsing unchanged |
| `run.py` / `main.py` | Call `create_llm_client(secrets)` unchanged |

## Verification Steps

1. With `llm_key_rotation: "auto"` -- behavior identical to current (backward compat)
2. With `llm_key_rotation: "per_call"` -- proactive=True even for paid Gemini keys
3. With `llm_key_rotation: "on_failure"` -- proactive=False even for free OpenRouter keys
4. With single key -- no PooledLLMClient created regardless of config (line 364)

## Execution Checklist

```
[ ] 1. Add llm_key_rotation config to global_config.yml
[ ] 2. Update _build_pooled_or_single() in llm_factory.py to read config
[ ] 3. Syntax check + import verification
[ ] 4. Commit on same branch + push
```
