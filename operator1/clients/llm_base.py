"""Abstract base class for LLM clients.

Defines the interface that both GeminiClient and ClaudeClient implement.
All methods return sensible fallbacks on failure -- the LLM is never a
hard dependency for the pipeline.

Includes shared retry logic with exponential backoff and per-host rate
limiting, consistent with the project's http_utils pattern.
"""

from __future__ import annotations

import json
import logging
import time
from abc import ABC, abstractmethod
from typing import Any

import requests

from operator1.config_loader import get_global_config

logger = logging.getLogger(__name__)


class LLMNonRetryableError(Exception):
    """Raised for non-retryable LLM API errors (e.g. 400 invalid key).

    This intentionally does NOT inherit from ``requests.RequestException``
    so that the retry catch block will not intercept it.
    """

    def __init__(self, provider: str, status_code: int, detail: str) -> None:
        self.provider = provider
        self.status_code = status_code
        self.detail = detail
        super().__init__(f"{provider} API error {status_code}: {detail}")


# ---------------------------------------------------------------------------
# Per-host rate limiting (mirrors http_utils._rate_limit_sleep)
# ---------------------------------------------------------------------------

_last_request_time_by_host: dict[str, float] = {}

# LLM-specific rate limits (requests per second).
# Gemini free tier: 15 RPM = 0.25/s.  Paid: 1000 RPM = ~16/s.
# Claude: Tier 1 = 50 RPM = ~0.83/s.  Higher tiers scale up.
# These are conservative defaults; override via global_config.yml.
_LLM_HOST_RATE_LIMITS: dict[str, float] = {
    "generativelanguage.googleapis.com": 0.25,  # Gemini free tier
    "api.anthropic.com": 0.8,                   # Claude Tier 1
    "openrouter.ai": 0.15,                      # OpenRouter free tier: ~10 RPM = 0.17/s (conservative)
}


from operator1.http_utils import _extract_host, _sanitise_url  # shared utilities


# ---------------------------------------------------------------------------
# System prompt loading (from config/llm_system_prompts.yml)
# ---------------------------------------------------------------------------

_system_prompt_cache: dict[str, Any] | None = None


def _load_system_prompts() -> dict[str, Any]:
    """Load and cache system prompts from config/llm_system_prompts.yml."""
    global _system_prompt_cache
    if _system_prompt_cache is not None:
        return _system_prompt_cache

    try:
        from operator1.config_loader import load_config
        _system_prompt_cache = load_config("llm_system_prompts")
    except Exception as exc:
        logger.debug("System prompts config not found: %s (using defaults)", exc)
        _system_prompt_cache = {}
    return _system_prompt_cache


def get_system_prompt(task_type: str = "") -> str:
    """Build a system prompt for the given task type.

    Composes: base_system_prompt + task-specific overlay.
    Returns empty string if config is not available (graceful fallback).
    """
    config = _load_system_prompts()
    base = config.get("base_system_prompt", "")
    overlay = config.get(task_type, "") if task_type else ""
    parts = [p for p in (base, overlay) if p]
    return "\n\n".join(parts)


def get_task_temperature(task_type: str = "") -> float | None:
    """Get the enforced temperature for a task type, or None for default."""
    config = _load_system_prompts()
    defaults = config.get("temperature_defaults", {})
    if task_type and task_type in defaults:
        return float(defaults[task_type])
    return None


def _rate_limit_sleep(host: str, calls_per_second: float | None = None) -> None:
    """Per-host rate limiter: sleep if requests are too fast."""
    effective_rate = calls_per_second or _LLM_HOST_RATE_LIMITS.get(host, 1.0)
    if effective_rate <= 0:
        return

    min_interval = 1.0 / effective_rate
    now = time.time()
    last_time = _last_request_time_by_host.get(host, 0.0)
    elapsed = now - last_time
    if elapsed < min_interval:
        sleep_time = min_interval - elapsed
        logger.debug("LLM rate limiting [%s]: sleeping %.2fs", host, sleep_time)
        time.sleep(sleep_time)
    _last_request_time_by_host[host] = time.time()


# _sanitise_url imported from operator1.http_utils above


# ---------------------------------------------------------------------------
# Model capability definitions
# ---------------------------------------------------------------------------

# Each entry: (model_name, max_output_tokens, context_window, report_capable)
# "report_capable" means it can generate 8000-12000 word reports (~16k tokens)
GEMINI_MODELS: dict[str, dict[str, Any]] = {
    "gemini-2.5-flash-preview-05-20": {
        "max_output_tokens": 65536,
        "context_window": 1048576,
        "report_capable": True,
        "tier": "preview",
    },
    "gemini-2.5-pro-preview-05-06": {
        "max_output_tokens": 65536,
        "context_window": 1048576,
        "report_capable": True,
        "tier": "preview",
    },
    "gemini-2.0-flash": {
        "max_output_tokens": 8192,
        "context_window": 1048576,
        "report_capable": True,
        "tier": "stable",
    },
    # NOTE: gemini-1.5-pro and gemini-1.5-flash removed -- deprecated from
    # Gemini v1beta API as of early 2026.  Use gemini-2.0-flash or newer.
}

CLAUDE_MODELS: dict[str, dict[str, Any]] = {
    "claude-opus-4-20250514": {
        "max_output_tokens": 32000,
        "context_window": 200000,
        "report_capable": True,
        "tier": "flagship",
    },
    "claude-sonnet-4-20250514": {
        "max_output_tokens": 64000,
        "context_window": 200000,
        "report_capable": True,
        "tier": "balanced",
    },
    "claude-3-5-sonnet-20241022": {
        "max_output_tokens": 8192,
        "context_window": 200000,
        "report_capable": True,
        "tier": "stable",
    },
    "claude-3-5-haiku-20241022": {
        "max_output_tokens": 8192,
        "context_window": 200000,
        "report_capable": True,
        "tier": "fast",
    },
}


def get_best_model(provider: str, model_registry: dict[str, dict[str, Any]]) -> str:
    """Pick the best cost-effective report-capable model from a provider's registry.

    Prefers stable/balanced tiers over flagship (Opus is $15/$75 per MTok --
    too expensive for free-tier keys). The "fast" tier (Haiku at $0.25/$1.25)
    is preferred over "flagship" (Opus) for cost efficiency.

    Returns the model name string.
    """
    candidates = [
        (name, info)
        for name, info in model_registry.items()
        if info.get("report_capable", False)
    ]
    if not candidates:
        return next(iter(model_registry))

    # Cost-aware priority: balanced > stable > fast > preview > flagship
    # This ensures we pick Sonnet ($3/$15) or Haiku ($0.25/$1.25) over
    # Opus ($15/$75) by default, saving 5-60x on API costs.
    tier_priority = {"balanced": 0, "stable": 1, "fast": 2, "preview": 3, "flagship": 4}
    candidates.sort(
        key=lambda x: (
            tier_priority.get(x[1].get("tier", "stable"), 5),
            -x[1].get("max_output_tokens", 0),
        )
    )
    return candidates[0][0]


def validate_model(
    model: str,
    model_registry: dict[str, dict[str, Any]],
    provider: str,
) -> str:
    """Validate a model name against the registry.

    If the model is in the registry, return it with full metadata logging.
    If the model is NOT in the registry, accept it anyway with default
    capabilities (4096 max tokens, 32768 context). This allows users to
    specify any model their provider supports without being locked to
    the hardcoded registry.
    """
    if model in model_registry:
        info = model_registry[model]
        logger.info(
            "%s model '%s': max_output=%d, context=%d, report_capable=%s",
            provider, model,
            info["max_output_tokens"],
            info["context_window"],
            info["report_capable"],
        )
        return model

    # Accept unknown models with conservative defaults instead of
    # falling back to a different (potentially expensive) model.
    model_registry[model] = {
        "max_output_tokens": 4096,
        "context_window": 32768,
        "report_capable": True,
        "tier": "unknown",
    }
    logger.info(
        "%s model '%s' not in known registry -- accepting with default "
        "capabilities (max_output=4096, context=32768). "
        "Known models: %s",
        provider, model, ", ".join(model_registry.keys()),
    )
    return model


# ---------------------------------------------------------------------------
# Abstract base class
# ---------------------------------------------------------------------------

class LLMClient(ABC):
    """Abstract LLM client interface.

    Subclasses must implement ``_do_request`` (the actual HTTP call)
    and ``_build_request_args`` (provider-specific request construction).
    All higher-level methods (entity discovery, report generation,
    sentiment scoring) are implemented here using shared prompts.

    Includes built-in retry logic with exponential backoff and per-host
    rate limiting, following the project's existing ``http_utils`` pattern.
    """

    # ------------------------------------------------------------------
    # Abstract methods -- subclasses must implement
    # ------------------------------------------------------------------

    @abstractmethod
    def _build_request_args(
        self,
        prompt: str,
        *,
        max_output_tokens: int = 8192,
        temperature: float = 0.7,
        system_prompt: str = "",
    ) -> dict[str, Any]:
        """Build the request kwargs for requests.post().

        Must return a dict with keys: url, json, headers (optional),
        and any other kwargs for requests.post().

        Parameters
        ----------
        system_prompt:
            System-level instructions loaded from config/llm_system_prompts.yml.
            Provider-specific: Gemini uses ``systemInstruction``, Claude uses
            ``system``, OpenRouter uses a system role message.
        """

    @abstractmethod
    def _parse_response(self, data: dict[str, Any]) -> str:
        """Extract the text content from a provider's JSON response."""

    @property
    @abstractmethod
    def provider_name(self) -> str:
        """Human-readable provider name (e.g. 'Gemini', 'Claude')."""

    @property
    @abstractmethod
    def model_name(self) -> str:
        """The model identifier currently in use."""

    @property
    @abstractmethod
    def max_output_tokens(self) -> int:
        """Maximum output tokens the current model supports."""

    # ------------------------------------------------------------------
    # Shared request execution with retry + rate limiting
    # ------------------------------------------------------------------

    def _execute_request(
        self,
        prompt: str,
        *,
        max_output_tokens: int = 8192,
        temperature: float = 0.7,
        timeout: int = 60,
        task_type: str = "",
    ) -> str:
        """Execute an LLM request with retry logic and rate limiting.

        This is the core execution method that handles:
        - System prompt composition (base + task overlay)
        - Per-task temperature enforcement
        - Per-host rate limiting
        - Exponential backoff on 429/5xx errors
        - Retry-After header respect
        - Request logging
        """
        # Compose system prompt from config (base + task overlay)
        system_prompt = get_system_prompt(task_type)

        # Enforce per-task temperature from config (overrides caller)
        task_temp = get_task_temperature(task_type)
        if task_temp is not None:
            temperature = task_temp

        cfg = get_global_config()
        max_retries: int = cfg.get("max_retries", 5)
        backoff: float = cfg.get("backoff_factor", 2.0)
        retryable_codes: set = set(cfg.get("retry_on_status", [429, 500, 502, 503, 504]))

        req_args = self._build_request_args(
            prompt,
            max_output_tokens=max_output_tokens,
            temperature=temperature,
            system_prompt=system_prompt,
        )
        url = req_args.pop("url")
        host = _extract_host(url)

        last_exc: Exception | None = None

        for attempt in range(1, max_retries + 1):
            # Rate limit before each attempt
            _rate_limit_sleep(host)

            t0 = time.time()
            try:
                resp = requests.post(url, timeout=timeout, **req_args)
                elapsed = time.time() - t0

                if resp.status_code == 200:
                    logger.debug(
                        "%s request OK (%.1fs, attempt %d)",
                        self.provider_name, elapsed, attempt,
                    )
                    data = resp.json()
                    return self._parse_response(data)

                # Check if retryable
                if resp.status_code not in retryable_codes:
                    # Non-retryable error -- raise immediately
                    error_detail = resp.text[:500]
                    logger.error(
                        "%s API error %d (non-retryable): %s",
                        self.provider_name, resp.status_code, error_detail,
                    )
                    raise LLMNonRetryableError(
                        self.provider_name, resp.status_code, error_detail,
                    )

                # Retryable error -- backoff.
                # For 429 (rate limit), give enough attempts for the rate
                # limiter to cool down (especially for free-tier OpenRouter).
                # For 5xx (server errors), use the full retry budget.
                _max_attempts_for_code = 4 if resp.status_code == 429 else max_retries

                retry_after = resp.headers.get("Retry-After")
                if retry_after:
                    try:
                        wait = float(retry_after)
                    except ValueError:
                        wait = backoff ** attempt
                else:
                    wait = backoff ** attempt

                logger.warning(
                    "%s HTTP %d on attempt %d/%d for %s -- retrying in %.1fs",
                    self.provider_name, resp.status_code, attempt,
                    _max_attempts_for_code, _sanitise_url(url), wait,
                )
                last_exc = requests.HTTPError(
                    f"HTTP {resp.status_code}", response=resp,
                )

                if attempt >= _max_attempts_for_code:
                    # Exhausted retries for this error type -- raise so
                    # PooledLLMClient can rotate to the next key.
                    raise RuntimeError(
                        f"{self.provider_name}: HTTP {resp.status_code} "
                        f"after {attempt} attempts (rate limited)"
                    )

                time.sleep(wait)

            except requests.RequestException as exc:
                elapsed = time.time() - t0
                last_exc = exc
                wait = backoff ** attempt
                logger.warning(
                    "%s request error on attempt %d/%d (%.1fs): %s -- retrying in %.1fs",
                    self.provider_name, attempt, max_retries, elapsed, exc, wait,
                )
                time.sleep(wait)

        # All retries exhausted
        raise RuntimeError(
            f"{self.provider_name}: all {max_retries} retries exhausted. "
            f"Last error: {last_exc}"
        )

    # ------------------------------------------------------------------
    # High-level generate methods (use _execute_request internally)
    # ------------------------------------------------------------------

    def generate(self, prompt: str, *, task_type: str = "") -> str:
        """Send a prompt and return the raw text response.

        This is the public interface used by ``run.py`` for LLM-based
        market routing and by ``PooledLLMClient`` for key rotation.

        Parameters
        ----------
        task_type:
            Task identifier for system prompt selection and temperature
            enforcement.  Valid values: ``"report_generation"``,
            ``"entity_discovery"``, ``"sentiment_scoring"``,
            ``"data_extraction"``, ``"filing_extraction"``,
            ``"macro_mapping"``.  Empty string uses base prompt only.
        """
        cfg = get_global_config()
        timeout = cfg.get("timeout_s", 30)
        return self._execute_request(prompt, timeout=timeout, task_type=task_type)

    # Keep private alias for backward compatibility with internal callers
    def _generate(self, prompt: str, *, task_type: str = "") -> str:
        """Send a prompt and return the raw text response."""
        return self.generate(prompt, task_type=task_type)

    def _generate_with_config(
        self,
        prompt: str,
        *,
        max_output_tokens: int = 8192,
        temperature: float = 0.7,
        timeout: int = 60,
        task_type: str = "",
    ) -> str:
        """Send a prompt with generation config and return the raw text."""
        return self._execute_request(
            prompt,
            max_output_tokens=max_output_tokens,
            temperature=temperature,
            timeout=timeout,
            task_type=task_type,
        )

    # ------------------------------------------------------------------
    # Shared helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _parse_json_response(text: str) -> Any:
        """Best-effort extraction of JSON from an LLM response.

        Handles responses that wrap JSON in markdown code fences,
        preamble text before/after JSON, and other common LLM quirks.
        Uses regex extraction as a fallback when direct parsing fails.
        """
        import re

        cleaned = text.strip()
        # Strip markdown code fences if present
        if cleaned.startswith("```"):
            lines = cleaned.split("\n")
            # Remove first and last fence lines
            lines = [l for l in lines if not l.strip().startswith("```")]
            cleaned = "\n".join(lines).strip()

        # Attempt 1: Direct JSON parse
        try:
            return json.loads(cleaned)
        except json.JSONDecodeError:
            pass

        # Attempt 2: Extract JSON array via regex (handles preamble/postamble)
        array_match = re.search(r'\[[\s\S]*?\]', cleaned)
        if array_match:
            try:
                return json.loads(array_match.group())
            except json.JSONDecodeError:
                pass

        # Attempt 3: Extract JSON object via regex
        obj_match = re.search(r'\{[\s\S]*?\}', cleaned)
        if obj_match:
            try:
                return json.loads(obj_match.group())
            except json.JSONDecodeError:
                pass

        # Attempt 4: Try to extract comma-separated numbers (sentiment scores)
        numbers = re.findall(r'-?\d+\.?\d*', cleaned)
        if len(numbers) >= 3:
            try:
                return [float(n) for n in numbers]
            except ValueError:
                pass

        logger.warning("Failed to parse LLM JSON response; returning None")
        return None

    # ------------------------------------------------------------------
    # Linked entity discovery (Sec 5)
    # ------------------------------------------------------------------

    _LINKED_ENTITIES_PROMPT = """\
You are a financial analyst. Given the following company profile, suggest
related entities grouped by relationship type.

Company profile:
{profile_json}

Sector hints: {sector_hints}

Return a JSON object with these keys. Each value is a list of objects:

- competitors: direct competitors in the same industry
- suppliers: known major suppliers
- customers: known major customers
- financial_institutions: primary banks or lenders
- logistics: key logistics or distribution partners
- regulators: relevant regulatory bodies (if publicly listed)

Each entity should be an object with:
  "name": "Company Name" (publicly traded, common English trading name),
  "market_id": "the market_id from the AVAILABLE MARKETS list below",
  "ticker": "the company's ticker symbol in that market",
  "relationship_start": "YYYY" or "ongoing" or "unknown",
  "relationship_end": "current" or "YYYY" or "unknown",
  "stability": "stable" or "volatile" or "new"

AVAILABLE MARKETS (use these exact market_id values):
{available_markets}

Example format:
{{"competitors": [{{"name": "Taiwan Semiconductor", "market_id": "tw_mops", "ticker": "2330", "relationship_start": "2020", "relationship_end": "current", "stability": "stable"}}]}}

IMPORTANT:
- Relationships change over time. Only include entities with CURRENT or
  RECENT relationships (within the last 2 years).
- Use the market_id where the company is PRIMARILY listed.
- If unsure of the market_id or ticker, omit them and set to empty string.
- Only include companies you are reasonably confident about.

Return valid JSON only, no markdown.
"""

    def propose_linked_entities(
        self,
        target_profile: dict[str, Any],
        sector_hints: str = "",
        available_markets: str = "",
    ) -> dict[str, list]:
        """Ask LLM to propose linked entities for a target company.

        Returns dict mapping relationship_group -> list of entity items.
        Each item is either a plain name string (backward compat) or a dict
        with ``name``, ``market_id``, and ``ticker`` for direct routing.
        Returns empty dict on any failure.
        """
        try:
            prompt = self._LINKED_ENTITIES_PROMPT.format(
                profile_json=json.dumps(target_profile, indent=2),
                sector_hints=sector_hints or "none",
                available_markets=available_markets or "Not available -- omit market_id",
            )
            text = self._generate(prompt, task_type="entity_discovery")
            parsed = self._parse_json_response(text)
            if isinstance(parsed, dict):
                result: dict[str, list] = {}
                for k, vs in parsed.items():
                    if not isinstance(vs, list):
                        continue
                    entities: list = []
                    for v in vs:
                        if isinstance(v, dict):
                            name = v.get("name", "")
                            if name:
                                # Preserve market routing info from LLM
                                entities.append({
                                    "name": str(name),
                                    "market_id": str(v.get("market_id", "") or ""),
                                    "ticker": str(v.get("ticker", "") or ""),
                                })
                        elif isinstance(v, str):
                            entities.append(v)
                    if entities:
                        result[k] = entities
                # Store the raw parsed data for temporal extraction
                self._last_entity_proposals_raw = parsed
                return result
            return {}
        except Exception as exc:
            logger.warning("%s linked-entity proposal failed: %s", self.provider_name, exc)
            return {}

    # ------------------------------------------------------------------
    # 3-call entity discovery prompts (international + local + gap-fill)
    # ------------------------------------------------------------------

    _LINKED_ENTITIES_INTL_PROMPT = """\
You are a financial analyst specializing in global competitive landscapes.

Company profile:
{profile_json}

TASK: First determine if this company has significant international operations,
revenue, exports, or supply chains outside its home country ({country}).

Then list INTERNATIONAL (non-{country}) entities across these groups:
- competitors: international competitors from OTHER countries
- suppliers: international/cross-border suppliers
- customers: international customers or export markets

Rules:
- Only include companies headquartered OUTSIDE {country}
- Only include publicly traded companies
- Include 5-8 entities per group if the company is truly international
- If the company is primarily domestic, say so and list fewer entities
- Include the entity's country of headquarters

Return JSON:
{{
  "is_international": true/false,
  "international_revenue_estimate": "high/medium/low/unknown",
  "competitors": [{{"name": "Company Name", "country": "US", "relationship_start": "2020", "relationship_end": "current", "stability": "stable"}}],
  "suppliers": [...],
  "customers": [...]
}}

Only include entities you are reasonably confident about.
Return valid JSON only, no markdown.
"""

    _LINKED_ENTITIES_LOCAL_PROMPT = """\
You are a financial analyst specializing in the {country} market.

Company profile:
{profile_json}

We already found these INTERNATIONAL entities (do NOT repeat them):
{already_found}

TASK: Focus on the DOMESTIC {country} market only. List:
- competitors: at least 5 domestic competitors in the same sector/industry
- suppliers: at least 3 domestic suppliers
- customers: major domestic customers
- financial_institutions: primary banks, lenders, or financial partners in {country}
- logistics: key domestic logistics or distribution partners
- regulators: relevant regulatory bodies (if publicly listed)

Rules:
- Only include entities headquartered IN {country}
- Only include publicly traded companies (except regulators)
- Be thorough: list MORE entities than you think necessary
- Each entity should have the temporal context fields

Return JSON with the same format:
{{"competitors": [{{"name": "Company Name", "country": "{country}", "relationship_start": "2020", "relationship_end": "current", "stability": "stable"}}], ...}}

Return valid JSON only, no markdown.
"""

    _LINKED_ENTITIES_GAPFILL_PROMPT = """\
You are a financial analyst reviewing entity discovery results.

Company: {company_name} ({ticker}), {sector} sector, {country}

Entities found so far:
{found_summary}

Groups with few or no entities:
{thin_groups}

TASK: Fill the gaps. For each thin/empty group listed above, suggest
additional entities that we missed. Also consider:
- Any RECENT changes (last 12 months) in the competitive landscape
- New market entrants or departures
- Supply chain shifts or new partnerships
- Any major customer wins or losses

Return JSON with ONLY the additional entities (do not repeat existing ones):
{{"competitors": [{{"name": "Company Name", "country": "XX", "relationship_start": "2024", "relationship_end": "current", "stability": "new"}}], ...}}

Return valid JSON only, no markdown.
"""

    def propose_linked_entities_3call(
        self,
        target_profile: dict[str, Any],
        sector_hints: str = "",
    ) -> dict[str, list[str]]:
        """3-call LLM entity discovery for thicker linked entity caches.

        Call 1: International scope check + global peers
        Call 2: Domestic deep dive (avoids duplicating Call 1 results)
        Call 3: Gap-fill for underrepresented groups

        Returns dict mapping relationship_group -> list of company names.
        Falls back to single-call propose_linked_entities() on failure.
        """
        country = (target_profile.get("country") or "Unknown").upper()
        company_name = target_profile.get("name", "Unknown")
        ticker = target_profile.get("ticker", "")
        sector = target_profile.get("sector", "Unknown")
        profile_json = json.dumps(target_profile, indent=2)

        all_entities: dict[str, list[str]] = {}
        all_raw: dict[str, list] = {}

        def _merge_results(parsed: dict) -> None:
            """Merge parsed LLM response into all_entities."""
            for group, entities in parsed.items():
                if group.startswith("is_") or group.endswith("_estimate"):
                    continue  # skip metadata fields
                if not isinstance(entities, list):
                    continue
                if group not in all_entities:
                    all_entities[group] = []
                    all_raw[group] = []
                existing_names = {n.lower() for n in all_entities[group]}
                for ent in entities:
                    name = ""
                    if isinstance(ent, dict):
                        name = ent.get("name", "")
                    elif isinstance(ent, str):
                        name = ent
                    if name and name.lower() not in existing_names:
                        all_entities[group].append(name)
                        all_raw[group].append(ent)
                        existing_names.add(name.lower())

        # ---- Call 1: International scope + global peers ----
        try:
            prompt1 = self._LINKED_ENTITIES_INTL_PROMPT.format(
                profile_json=profile_json,
                country=country,
            )
            text1 = self._generate(prompt1, task_type="entity_discovery")
            parsed1 = self._parse_json_response(text1)
            if isinstance(parsed1, dict):
                is_intl = parsed1.get("is_international", False)
                _merge_results(parsed1)
                logger.info(
                    "3-call discovery: Call 1 (international) done. "
                    "is_international=%s, entities=%d",
                    is_intl,
                    sum(len(v) for k, v in all_entities.items()),
                )
        except Exception as exc:
            logger.warning("3-call discovery: Call 1 failed: %s", exc)

        # ---- Call 2: Domestic deep dive ----
        try:
            already_found = json.dumps(
                {g: [n for n in names] for g, names in all_entities.items()},
                indent=2,
            )
            prompt2 = self._LINKED_ENTITIES_LOCAL_PROMPT.format(
                profile_json=profile_json,
                country=country,
                already_found=already_found,
            )
            text2 = self._generate(prompt2, task_type="entity_discovery")
            parsed2 = self._parse_json_response(text2)
            if isinstance(parsed2, dict):
                _merge_results(parsed2)
                logger.info(
                    "3-call discovery: Call 2 (local) done. total entities=%d",
                    sum(len(v) for v in all_entities.values()),
                )
        except Exception as exc:
            logger.warning("3-call discovery: Call 2 failed: %s", exc)

        # ---- Call 3: Gap-fill ----
        try:
            # Identify thin groups
            target_counts = {
                "competitors": 5, "suppliers": 3, "customers": 3,
                "financial_institutions": 2, "logistics": 2, "regulators": 1,
            }
            thin_groups = []
            for group, target_count in target_counts.items():
                actual = len(all_entities.get(group, []))
                if actual < target_count:
                    thin_groups.append(f"{group}: {actual} found (want {target_count}+)")

            if thin_groups:
                found_summary = "\n".join(
                    f"  {g}: {', '.join(names[:5])}"
                    + (f" (+{len(names)-5} more)" if len(names) > 5 else "")
                    for g, names in all_entities.items()
                    if names
                )
                prompt3 = self._LINKED_ENTITIES_GAPFILL_PROMPT.format(
                    company_name=company_name,
                    ticker=ticker,
                    sector=sector,
                    country=country,
                    found_summary=found_summary or "  (none found)",
                    thin_groups="\n".join(f"  - {tg}" for tg in thin_groups),
                )
                text3 = self._generate(prompt3, task_type="entity_discovery")
                parsed3 = self._parse_json_response(text3)
                if isinstance(parsed3, dict):
                    before = sum(len(v) for v in all_entities.values())
                    _merge_results(parsed3)
                    after = sum(len(v) for v in all_entities.values())
                    logger.info(
                        "3-call discovery: Call 3 (gap-fill) done. "
                        "added %d entities, total=%d",
                        after - before, after,
                    )
            else:
                logger.info("3-call discovery: Call 3 skipped (all groups sufficient)")
        except Exception as exc:
            logger.warning("3-call discovery: Call 3 failed: %s", exc)

        # Store raw data for temporal extraction
        self._last_entity_proposals_raw = all_raw

        total = sum(len(v) for v in all_entities.values())
        logger.info(
            "3-call discovery complete: %d total entities across %d groups",
            total, len(all_entities),
        )

        if total == 0:
            # Fall back to single-call if 3-call produced nothing
            logger.warning("3-call discovery returned 0 entities; falling back to single call")
            return self.propose_linked_entities(target_profile, sector_hints=sector_hints)

        return all_entities

    # ------------------------------------------------------------------
    # World Bank mapping suggestions (Sec 4)
    # ------------------------------------------------------------------

    _WB_MAPPING_PROMPT = """\
You are a macroeconomic data analyst. For the country "{country}" and
sector "{sector}", suggest the most relevant macroeconomic indicator codes
for these canonical variables:

- inflation_rate_yoy
- cpi_index
- unemployment_rate
- gdp_growth
- gdp_current_usd
- official_exchange_rate_lcu_per_usd
- current_account_balance_pct_gdp
- reserves_months_of_imports
- real_interest_rate
- lending_interest_rate
- deposit_interest_rate

Return a JSON object mapping each variable name to a macro indicator
code string. If unsure, omit the key.
Return valid JSON only, no markdown.
"""

    def propose_macro_indicator_mappings(
        self,
        country: str,
        sector: str = "",
    ) -> dict[str, str]:
        """Ask LLM to suggest macro indicator mappings.

        These are *suggestions only* -- not used at runtime without
        human review.  Returns empty dict on failure.
        """
        try:
            prompt = self._WB_MAPPING_PROMPT.format(
                country=country,
                sector=sector or "general",
            )
            text = self._generate(prompt, task_type="macro_mapping")
            parsed = self._parse_json_response(text)
            if isinstance(parsed, dict):
                return {k: str(v) for k, v in parsed.items()}
            return {}
        except Exception as exc:
            logger.warning("%s WB mapping proposal failed: %s", self.provider_name, exc)
            return {}

    # ------------------------------------------------------------------
    # Report generation (Sec 18) -- Full 13-section prompt (Phase E1)
    # ------------------------------------------------------------------

    _REPORT_PROMPT = """\
You are generating a data-grounded equity research report. Your role is to \
synthesize the quantitative pipeline outputs below into clear, actionable \
analysis. Every claim must cite data from the profile. Do not speculate \
beyond what the models produced.

You have been provided with a complete company profile that includes:
- 2 years of historical financial and market data
- Advanced temporal analysis using 25+ mathematical models (HMM, Kalman, \
GARCH, VAR, LSTM with MC Dropout, Temporal Fusion Transformer, Random Forest, XGBoost, etc.)
- Survival mode analysis (company + country)
- Ethical filter assessments (Purchasing Power, Solvency, Gharar, Cash is King)
- Multi-horizon predictions with Conformal Prediction intervals (distribution-free, \
guaranteed coverage -- not Gaussian assumptions)
- SHAP feature attribution explaining what drove each prediction
- MC Dropout epistemic uncertainty (separates "model does not know" from "inherent randomness")
- Dynamic Time Warping (DTW) historical analogs ("the last time this pattern occurred was...")
- Regime detection and structural break analysis
- Linked variables (sector, industry, competitors, macro indicators)
- Financial health composite scoring (0-100 scale across 5 tiers)
- News sentiment analysis (AI-scored from recent stock news articles)
- Peer percentile ranking (target vs peer group on all key metrics)
- Macro environment quadrant classification (goldilocks/reflation/stagflation/deflation)

Your task is to generate a professional investment report that synthesizes \
this information into actionable insights for sophisticated investors.

---

REPORT STRUCTURE (MUST INCLUDE ALL 14 SECTIONS):

1. EXECUTIVE SUMMARY
   - 3 bullet points summarizing key findings (cite specific metrics from the profile)
   - Investment recommendation derived from pipeline signals:
     Use hedge_fund.position.signal if available (>0.3=BUY, -0.3 to 0.3=HOLD, <-0.3=SELL).
     State confidence level based on model_diagnostics.overall_robustness.
   - 12-month price range from the pipeline's conformal prediction intervals (NOT your own estimate)

2. COMPANY OVERVIEW
   - Company identity and classification
   - Current market position and capitalization
   - Sector and industry context

3. HISTORICAL PERFORMANCE ANALYSIS (2 Years)
   - Total return vs real return (Purchasing Power filter applied)
   - Risk-adjusted performance: Sharpe ratio, maximum drawdown
   - Regime breakdown: time spent in bull/bear/high-volatility regimes
   - Structural breaks and major market events detected
   - Up days vs down days distribution

4. CURRENT FINANCIAL HEALTH (Tier-by-Tier Breakdown)

   **Explain the Tier Hierarchy:**
   - Why the 5-tier system matters
   - How weights change in different survival regimes
   - Current hierarchy weights and what they mean

   **Tier 1: Liquidity & Cash** -- Cash and equivalents, cash ratio, free cash flow; Cash is King filter results
   **Tier 2: Solvency & Debt** -- Debt-to-equity, net debt to EBITDA, interest coverage; Solvency filter results
   **Tier 3: Market Stability** -- Volatility, drawdown, volume; Gharar filter results
   **Tier 4: Profitability** -- Margins (gross, operating, net), ROE, ROA
   **Tier 5: Growth & Valuation** -- Revenue/earnings growth, P/E, EV/EBITDA

5. SURVIVAL MODE ANALYSIS
   - Current survival status (company, country, protection)
   - Historical survival episodes (count and duration)
   - Capital allocation quality (vanity score): composite 0-100 score measuring \
management discipline. Components: R&D vs Growth Mismatch, SGA Bloat (peer-relative), \
Capital Misallocation (debt/dividend signals), Competitive Decay (losing ground), \
Sentiment-Reality Gap (positive PR vs declining fundamentals). Labels: Disciplined (<20), \
Moderate (20-40), Wasteful (40-70), Reckless (>70). Interpret the trend (rising/falling/stable) \
and explain which components drive the score. Relate to management quality for investors.

6. LINKED VARIABLES & MARKET CONTEXT
   - **Peer Ranking**: percentile rank of target vs peer group on key metrics. \
A composite rank above 60 = above average; below 40 = below average. Show the top \
variable rankings in a table. Explain what relative positioning means for investors.
   - Sector performance: relative strength vs sector median
   - Industry positioning: valuation premium/discount vs industry
   - Competitor health assessment: how does the company compare?
   - Supply chain risk analysis (if applicable)
   - **Macro Environment Quadrant**: classify as Goldilocks (growth + low inflation), \
Reflation (growth + high inflation), Stagflation (low growth + high inflation), or \
Deflation (low growth + low inflation). Explain how the current quadrant affects this stock. \
Show the quadrant distribution over the analysis window.
   - **News Sentiment**: summarize AI-scored sentiment from recent news articles. \
Include mean sentiment, latest reading, and momentum trend. \
Explain what the sentiment trajectory means for near-term price action.
   - Macro indicators from per-region APIs (inflation, GDP, unemployment, FX)

7. TEMPORAL ANALYSIS & MODEL INSIGHTS
   - Current market regime and regime distribution over 2 years
   - Regime transitions and what they signal
   - Structural breaks detected
   - Model performance summary by tier (accuracy percentages)
   - Best performing module (from 23+ model ensemble including Kalman, GARCH, \
VAR, LSTM, Temporal Fusion Transformer, tree ensembles)
   - Conformal prediction calibration quality (empirical coverage vs target)
   - SHAP global feature importance: which variables matter most across all predictions
   - Confidence levels in predictions

8. PREDICTIONS & FORECASTS
   **Next Day:** NOTE: Due to Technical Alpha protection, only Low price \
is shown for next-day OHLC. Include Tier 1-5 variable predictions with \
Conformal Prediction intervals (distribution-free, guaranteed 90% coverage). \
Survival probability for next day.
   **Next Week:** Expected return and volatility, full OHLC candlestick series, predicted technical patterns.
   **Next Month:** Price target range (conformal 5th-95th percentile bands), key events to watch, regime shift predictions.
   **Next Year:** Annual outlook with widening uncertainty, predicted regime changes by quarter, long-term trajectory.
   **Monte Carlo Uncertainty:** Tail risk scenarios (worst 5%), base case (50th percentile), upside scenarios (top 5%).
   **SHAP Feature Drivers:** For key predictions, explain WHAT drove the forecast \
(e.g. "next-day cash_ratio prediction driven primarily by: +0.03 from declining \
short-term debt, -0.01 from rising volatility"). Use the SHAP narratives from the profile data.
   **Historical Analogs (DTW):** If available, describe the closest historical analog \
periods found via Dynamic Time Warping. Example: "The last time this company showed \
a similar pattern of rising debt + falling margins + high macro stress was [date]. \
In the following month, the stock [outcome]." Include the empirical return distribution \
from analog outcomes.
   **Epistemic vs Aleatoric Uncertainty:** Where MC Dropout data is available, \
distinguish between model uncertainty ("the model is unsure about this prediction") \
and inherent randomness ("even a perfect model would see variance here"). This helps \
investors understand the *quality* of each prediction.

9. TECHNICAL PATTERNS & CHART ANALYSIS
   - Describe the 2-year price chart with regime shading
   - Historical candlestick patterns detected (last 6 months)
   - Predicted patterns for next week/month
   - Support and resistance levels

10. ETHICAL FILTER ASSESSMENT
    **Purchasing Power Filter:** Verdict, nominal vs real return, inflation impact.
    **Solvency Filter:** Verdict, debt-to-equity ratio, threshold.
    **Gharar Filter (Uncertainty/Speculation):** Verdict, volatility, stability score.
    **Cash is King Filter:** Verdict, FCF yield.
    **Overall Ethical Score:** Combine all filters; is this investment suitable for ethical/Islamic investors?
    Universal lessons: Why these filters matter for ALL investors.

11. GEOPOLITICAL & CONFLICT RISK
    - Country conflict status (active conflict zones, fragile state classification)
    - Sanctions exposure (OFAC, EU, UN sanctions lists)
    - Conflict intensity score (0-1 scale) and trend
    - Supply chain exposure to conflict zones (linked entities in war zones)
    - Revenue exposure to conflict regions
    - Investment implications of geopolitical risk

12. RISK FACTORS & LIMITATIONS
    - Model assumptions and their limitations
    - Key risks: company-specific, industry/sector, macro/country
    - Scenarios that could invalidate predictions
    - Black swan events not captured by models
    11.1 LIMITATIONS (SHORT, REQUIRED):
    Provide 5-10 bullets covering: data window, OHLCV source (yfinance/regional wrapper) caveats, \
macro frequency reality (macro APIs provide monthly/quarterly data, aligned daily via as-of logic), \
missingness summary, any modules that failed and how the report compensated. \
Must be easy for a non-technical client to understand.

13. INVESTMENT RECOMMENDATION
    **Recommendation:** Derived from hedge_fund.position.signal (>0.3=BUY, -0.3..0.3=HOLD, <-0.3=SELL).
    If position signal is not available, derive from survival_probability and financial_health composite.
    Do NOT generate your own recommendation independent of the pipeline signals.
    **Confidence Level:** Based on model_diagnostics.overall_robustness (>0.7=High, 0.5-0.7=Medium, <0.5=Low).
    **12-Month Price Range:** From pipeline conformal intervals at 252d horizon. State point estimate + 90% CI.
    **Key Catalysts to Watch:** from event_calendar and filing_calendar data in the profile.
    **Risk Factors:** from survival_probability, conflict_risk, scenario_analysis data.
    **IMPORTANT:** If survival_probability < 0.85, lead with "ELEVATED RISK" warning.
    If scorecard grade is D or F, the tone must reflect distress.

14. APPENDIX
    - Methodology summary: all 23+ temporal modules used:
      * Regime Detection: HMM, GMM, PELT, Bayesian Change Point
      * Forecasting: Adaptive Kalman, GARCH, VAR, LSTM, Temporal Fusion Transformer (TFT)
      * Tree Ensembles: Random Forest, XGBoost, Gradient Boosting
      * Causality: Granger Causality, Transfer Entropy, Copula Models
      * Uncertainty: Conformal Prediction (distribution-free intervals), \
MC Dropout (epistemic uncertainty), Regime-Aware Monte Carlo, Importance Sampling
      * Explainability: SHAP (per-prediction feature attribution), Sobol Sensitivity
      * Historical Analogs: Dynamic Time Warping (DTW) for finding similar past periods
      * Optimisation: Genetic Algorithm for ensemble weights
      * Pattern Recognition: Candlestick Detector, Wavelet/Fourier Decomposition
    - Forward pass + burn-out process explanation (day-by-day predict-compare-update)
    - Conformal Prediction explanation: why distribution-free intervals are more \
reliable than Gaussian assumptions for financial data
    - SHAP explanation: how per-prediction drivers are computed
    - Ensemble weighting approach (inverse-RMSE + GA optimisation)
    - Variable tier definitions (Tier 1-5 explained)
    - Glossary of technical terms (including Conformal Prediction, SHAP, TFT, \
regime, structural break, nonconformity score)
    - Data sources: PIT government filings (financials + OHLCV), per-region macro APIs, LLM (relationships)
    - Data timestamps and coverage
    - Disclaimer and limitations

---

FORMATTING REQUIREMENTS:
- Use markdown formatting with clear section headers (##, ###)
- Use tables for financial data where appropriate
- Use bullet points and numbered lists for clarity
- Bold key metrics and verdicts
- Italicize interpretive commentary
- Include placeholders for charts: [CHART: Description]
- Keep language professional but accessible
- Explain technical concepts when first introduced
- Total length: aim for 8,000-12,000 words (comprehensive but readable)

---

COMPLETE COMPANY PROFILE DATA:
{profile_json}

---

Generate the complete Bloomberg-style investment report now.
"""

    def generate_report(
        self,
        company_profile_json: str,
        *,
        max_output_tokens: int = 16000,
        temperature: float = 0.3,
        timeout: int = 120,
    ) -> str:
        """Generate a Bloomberg-style analysis report from profile data.

        Automatically caps max_output_tokens to the model's limit if
        the requested value exceeds it.

        Parameters
        ----------
        company_profile_json:
            JSON string of the full company profile.
        max_output_tokens:
            Maximum tokens for the response (default 16000 for
            comprehensive reports).
        temperature:
            Sampling temperature (lower = more factual).
        timeout:
            Request timeout in seconds (report generation is slow).

        Returns
        -------
        Markdown report string, or a fallback message on failure.
        """
        # Cap to model's actual limit
        effective_tokens = min(max_output_tokens, self.max_output_tokens)
        if effective_tokens < max_output_tokens:
            logger.info(
                "%s model '%s' max output is %d tokens (requested %d); "
                "capping to model limit.",
                self.provider_name, self.model_name,
                self.max_output_tokens, max_output_tokens,
            )

        try:
            prompt = self._REPORT_PROMPT.format(
                profile_json=company_profile_json,
            )
            return self._generate_with_config(
                prompt,
                max_output_tokens=effective_tokens,
                temperature=temperature,
                timeout=timeout,
                task_type="report_generation",
            )

        except Exception as exc:
            logger.error("%s report generation failed: %s", self.provider_name, exc)
            return (
                "# Report Generation Failed\n\n"
                "The automated report could not be generated. "
                "Please review the raw data in the cache artifacts.\n\n"
                f"Error: {exc}\n"
            )

    # ------------------------------------------------------------------
    # Sentiment scoring (batch)
    # ------------------------------------------------------------------

    def score_sentiment(
        self,
        headlines: list[str],
        *,
        batch_size: int = 500,
    ) -> list[float]:
        """Score sentiment for a batch of news headlines.

        Sends all headlines in a single call (or splits into batches of
        ``batch_size`` if too many). Returns a list of scores from -1.0
        (very bearish) to +1.0 (very bullish).

        Parameters
        ----------
        headlines:
            List of headline strings.
        batch_size:
            Maximum headlines per API call.

        Returns
        -------
        list[float]
            Sentiment scores aligned 1:1 with input headlines.
            Returns empty list on failure.
        """
        if not headlines:
            return []

        all_scores: list[float] = []

        for i in range(0, len(headlines), batch_size):
            batch = headlines[i:i + batch_size]
            numbered = "\n".join(
                f"{j + 1}. {h}" for j, h in enumerate(batch)
            )
            prompt = (
                "Score the sentiment of each financial news headline below "
                "from -1.0 (very bearish/negative for the stock) to +1.0 "
                "(very bullish/positive for the stock). 0.0 means neutral.\n\n"
                "Return ONLY a JSON array of numbers in the same order, "
                "nothing else. Example: [-0.3, 0.8, 0.0, -0.5]\n\n"
                f"Headlines:\n{numbered}"
            )

            try:
                text = self._generate(prompt, task_type="sentiment_scoring")
                scores = self._parse_json_response(text)

                if isinstance(scores, list) and len(scores) == len(batch):
                    # Validate all are numbers in [-1, 1]
                    validated = []
                    for s in scores:
                        try:
                            val = float(s)
                            validated.append(max(-1.0, min(1.0, val)))
                        except (TypeError, ValueError):
                            validated.append(0.0)
                    all_scores.extend(validated)
                else:
                    logger.warning(
                        "%s sentiment: expected %d scores, got %s",
                        self.provider_name,
                        len(batch),
                        type(scores).__name__,
                    )
                    all_scores.extend([0.0] * len(batch))

            except Exception as exc:
                logger.warning("%s sentiment scoring failed: %s", self.provider_name, exc)
                all_scores.extend([0.0] * len(batch))

        return all_scores
