# Dashboard Toggles: LLM Key Rotation + Frequency Pipeline Mode

## What We're Adding

Two new `ui.select` dropdowns in the dashboard's "Advanced Options" expansion panel on the New Analysis page, plus saving their values to `global_config.yml` so they take effect when the pipeline runs.

---

## Current Dashboard Structure (relevant areas)

### New Analysis page (`dashboard.py:880-921`)

```
Pipeline Options
  [switch] Discover linked entities
  [switch] Run temporal models
  [switch] Generate PDF report

Advanced Options (expandable)
  [number] Lookback years
  [input]  End date (backtest)
  [select] PIT alignment (report_date / filing_date)
  [input]  Output directory
  [switch] Skip report generation
  [switch] Verbose debug logging
```

### Config mechanism

The advanced options use a local `_adv` dict (line 894) that feeds into the `cmd` list (line 945+) for the subprocess call. Config values are NOT persisted to YAML from the New Analysis page -- they're passed as CLI flags to `main.py`.

However, `llm_key_rotation` and `frequency_pipeline.mode` are read from `global_config.yml` at runtime (not CLI flags). So we need a different approach: **write to the YAML file** when the user changes the toggle, similar to how Scoring Weights works.

---

## Implementation Plan

### File: `dashboard.py` -- 1 file, ~30 lines added

#### Change 1: Add two `ui.select` widgets inside the Advanced Options expansion

**Location:** After line 920 (after "Verbose debug logging" switch), inside the `with ui.expansion("Advanced Options")` block.

```python
            ui.separator().classes("my-2")
            ui.label("Performance Settings").classes("text-sm font-bold text-gray-300")
            with ui.row().classes("items-center gap-4"):
                ui.label("Frequency pipeline").classes("w-40 text-sm")
                ui.select(
                    options={
                        "parallel": "Parallel (fast, 4+ cores)",
                        "sequential": "Sequential (low memory)",
                    },
                    value=_get_config_value("frequency_pipeline", "mode", "parallel"),
                    on_change=lambda e: _set_config_value("frequency_pipeline", "mode", e.value),
                ).classes("w-56")
            with ui.row().classes("items-center gap-4"):
                ui.label("LLM key rotation").classes("w-40 text-sm")
                ui.select(
                    options={
                        "auto": "Auto-detect (default)",
                        "per_call": "Per-call (free-tier keys)",
                        "on_failure": "On-failure (paid keys)",
                    },
                    value=_get_config_value("llm_key_rotation", None, "auto"),
                    on_change=lambda e: _set_config_value("llm_key_rotation", None, e.value),
                ).classes("w-56")
```

#### Change 2: Add helper functions for reading/writing global_config.yml

**Location:** Near the top of the file, after the `DashboardState` class (after line ~100).

```python
def _get_config_value(key: str, subkey: str | None, default: str) -> str:
    """Read a value from global_config.yml."""
    try:
        from operator1.config_loader import get_global_config
        cfg = get_global_config()
        if subkey:
            return str(cfg.get(key, {}).get(subkey, default))
        return str(cfg.get(key, default))
    except Exception:
        return default


def _set_config_value(key: str, subkey: str | None, value: str) -> None:
    """Write a value to global_config.yml and invalidate cache."""
    try:
        import yaml
        config_path = Path("config/global_config.yml")
        with open(config_path, "r") as f:
            cfg = yaml.safe_load(f) or {}
        if subkey:
            if key not in cfg or not isinstance(cfg[key], dict):
                cfg[key] = {}
            cfg[key][subkey] = value
        else:
            cfg[key] = value
        with open(config_path, "w") as f:
            yaml.dump(cfg, f, default_flow_style=False, sort_keys=False)
        # Invalidate config cache so next pipeline run picks up new value
        from operator1.config_loader import _cache
        _cache.pop("global_config", None)
    except Exception as exc:
        logger.warning("Failed to save config: %s", exc)
```

---

## Why This Approach

1. **Writes to YAML immediately** -- The pipeline reads `global_config.yml` at runtime via `get_global_config()`. Writing to YAML + invalidating cache means the next pipeline run (even in the same process) picks up the new value.

2. **Same pattern as Scoring Weights** -- The Scoring Weights panel at lines 1274-1733 already reads from YAML, modifies, and saves back with `save_scoring_weights()`. We're doing the same but simpler (no complex nested structure).

3. **No CLI flag changes needed** -- `llm_key_rotation` and `frequency_pipeline.mode` are read from config inside the pipeline code, not from CLI arguments. The dashboard just needs to update the config file before running the pipeline subprocess.

4. **Toggles are in Advanced Options** -- These are power-user settings, so they belong in the expandable Advanced Options section, not as top-level switches.

---

## Data Flow

```
User clicks "Sequential" in Frequency Pipeline dropdown
  |
  v
_set_config_value("frequency_pipeline", "mode", "sequential")
  |
  v
config/global_config.yml updated on disk
config_loader._cache invalidated
  |
  v
User clicks Run button -> main.py subprocess starts
  |
  v
main.py -> run_stages() -> _build_registry() -> build_freq_substages()
  |
  v
build_freq_substages() calls get_global_config() -> reads "sequential"
  |
  v
Returns 8 sequential sub-stages instead of 4 parallel sub-stages
```

---

## Files Changed

| File | Lines Added | What |
|------|------------|------|
| `dashboard.py` | ~30 | 2 `ui.select` widgets + 2 helper functions |

## No Changes Needed

| File | Why |
|------|-----|
| `global_config.yml` | Already has both config keys from our earlier commits |
| `llm_factory.py` | Already reads `llm_key_rotation` from config |
| `stage2_freq_pipeline.py` | Already reads `frequency_pipeline.mode` from config |
| `runner.py` | Already calls `build_freq_substages()` dynamically |

---

## Execution Checklist

```
[ ] 1. Add _get_config_value() and _set_config_value() helpers
[ ] 2. Add two ui.select widgets in Advanced Options
[ ] 3. Syntax check
[ ] 4. Commit + push
```
