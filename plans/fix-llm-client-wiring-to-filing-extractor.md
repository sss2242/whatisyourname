# Fix LLM Client Wiring to Filing Extractor

## Problem

The LLM client created in main.py never reaches `try_filing_extraction()`. All 9 Tier 2 market clients pass `llm_client=None`, causing every PDF extraction to fail silently and fall back to yfinance.

## Approach: Auto-Create LLM Client in `try_filing_extraction()`

The cleanest fix with minimal code change. When `llm_client=None`, `try_filing_extraction()` auto-creates one from environment secrets via the existing `llm_factory`.

This avoids:
- Threading `llm_client` through `create_pit_client()` and 15+ constructors
- Changing the signature of every Tier 2 client class
- Modifying main.py's PIT client instantiation

## Changes Required

### 1. `filing_discoverer.py` -- Auto-create LLM client when None

In `try_filing_extraction()` at line 1741, add:

```python
if llm_client is None:
    try:
        from operator1.clients.llm_factory import create_llm_client
        from operator1.secrets_loader import load_secrets
        llm_client = create_llm_client(load_secrets())
    except Exception:
        pass  # Still None -- extraction will fail gracefully
```

This tries to create an LLM client from whatever API keys are in the environment. If no keys exist, it stays None and the extraction fails gracefully as before.

### 2. No other files need to change

The 9 Tier 2 clients continue passing `llm_client=None`, but now `try_filing_extraction()` handles it internally. Zero changes to client code.

## Safety

- If no LLM API keys are configured, behavior is identical to before (returns empty, falls back to yfinance)
- The `load_secrets()` call only reads from `.env` or environment variables -- no user interaction
- The `create_llm_client()` call is idempotent and stateless
- The auto-created client is used only for the filing extraction, not stored globally
