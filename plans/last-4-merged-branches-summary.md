# Last 4 Merged Branches Summary

Generated: 2026-02-28

## 1. `feature/uk-ixbrl-parse` (PR #2 from sos0240)

**Commit:** `feat: add ixbrl-parse for UK Companies House financial data extraction`

- Added the `ixbrl-parse` library integration for parsing UK Companies House iXBRL financial documents
- New test file: `tests/test_pit_uk_ixbrl.py` (155 lines)
- 36 files changed, 195 insertions, 1945 deletions (significant cleanup alongside the feature)

## 2. `feature/test-all-wrappers` (PR #1 from sos0240)

**Commit:** `test: add individual test files for all wrappers (OHLCV, macro, PIT, LLM)`

- Added comprehensive test coverage for all wrapper modules
- Individual test files for each OHLCV provider (baostock, pykrx, twstock, yfinance)
- Individual test files for each macro provider (banxico, bcb, bcch, dgbas, estat, fredapi, kosis, ons, sdmx, wbgapi)
- Individual test files for each PIT wrapper (br_cvm, cl_cmf, eu_esef, jp_jquants, kr_dart, tw_mops, uk_ch, us_edgar)
- Individual test files for LLM modules (base, claude, gemini, factory, model_selection)

## 3. `fix/translator-bridge-and-field-name-alignment` (PR #1 from sos4002)

**Commits:**
- `fix: wire translator bridge between wrappers and cache_builder, align field names`
- `fix: make all API keys required, add validate_secrets() to main.py entry point`
- `fix: close coverage gaps -- J-Quants adapter, yfinance suffixes, EU macro routing`
- `feat: add 5 macro fetcher modules for UK, Japan, Korea, Taiwan, Chile`
- `fix: yfinance MultiIndex columns + rewrite ONS macro to use FRED`

Key changes:
- Wired the canonical translator bridge between data wrappers and the cache builder
- Aligned field names across all wrappers for consistency
- Added API key validation at startup
- Added macro fetcher modules for 5 new regions (UK/ONS, Japan/e-Stat, Korea/KOSIS, Taiwan/DGBAS, Chile/BCCH)
- Fixed yfinance MultiIndex column handling

## 4. `feature/replace-edinet-ch-research` (PR #1 from sos0024)

**Commits:**
- `research: replace EDINET with J-Quants, add ixbrl-parse for UK CH, verify all API keys`
- `feat: replace EDINET with J-Quants, add ixbrl-parse, add key prompts, remove PII guard`
- `feat: add per-region OHLCV and macro wrappers with global fallbacks`
- `fix: add timeouts and fallback paths for slow/broken APIs`
- `fix: update ECB domain to data-api.ecb.europa.eu (verified working)`
- `fix: macro canonical name mismatch in macro_mapping.py`

Key changes:
- Replaced EDINET (archived) with J-Quants for Japan financial data
- Added per-region OHLCV and macro wrappers with yfinance/wbgapi global fallbacks
- Added timeouts and fallback paths for unreliable APIs
- Updated ECB API domain to the new verified endpoint
- Fixed macro canonical name mismatches in the mapping layer

---

## Environment Setup

- **Python version**: 3.12.3 (pinned in `.python-version`)
- **All dependencies from `requirements.txt`** install and import successfully
- **Smoke tests**: 23/23 passing (`tests/test_phase1_smoke.py`)
