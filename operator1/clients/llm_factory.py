"""Factory for creating LLM client instances with multi-key rotation.

Reads ``llm_provider`` from global config (or the ``LLM_PROVIDER`` env
variable) and returns the matching client.  Supported values:

- ``gemini`` (default) -- Google Gemini via generativelanguage API
- ``claude`` -- Anthropic Claude via Messages API

**Multi-key support**: Comma-separated API keys in ``.env`` are split
into a pool.  On credit exhaustion (HTTP 400/402) or rate limiting
(HTTP 429), the factory automatically rotates to the next key.  When
all keys for one provider are exhausted, it falls back to the other
provider.

**Cost-aware model selection**: By default picks the most cost-effective
model (Sonnet/Haiku over Opus) to maximize runs per free-tier credit.

Model selection can be configured via ``llm_model`` in global_config.yml
or the ``LLM_MODEL`` environment variable. Use ``"auto"`` to let the
factory pick the best model for the provider.
"""

from __future__ import annotations

import logging
import os
from typing import Any

from operator1.clients.llm_base import LLMClient
from operator1.config_loader import get_global_config

logger = logging.getLogger(__name__)

# Supported provider identifiers (case-insensitive)
_SUPPORTED_PROVIDERS = ("gemini", "claude", "openrouter")

# HTTP status codes that indicate key exhaustion (rotate to next key)
_EXHAUSTION_ERRORS = ("credit balance is too low", "quota exceeded",
                      "rate limit", "429", "402", "insufficient")


def get_available_models(provider: str) -> list[dict[str, Any]]:
    """Return a list of available models for the given provider.

    Each entry is a dict with keys: name, max_output_tokens, context_window,
    report_capable, tier.

    Parameters
    ----------
    provider:
        ``"gemini"`` or ``"claude"``.

    Returns
    -------
    List of model info dicts, sorted by tier priority then output capacity.
    """
    from operator1.clients.llm_base import GEMINI_MODELS, CLAUDE_MODELS
    from operator1.clients.openrouter import OPENROUTER_MODELS

    if provider == "gemini":
        registry = GEMINI_MODELS
    elif provider == "openrouter":
        registry = OPENROUTER_MODELS
    else:
        registry = CLAUDE_MODELS

    tier_priority = {"balanced": 0, "stable": 1, "fast": 2, "preview": 3, "flagship": 4}
    models = []
    for name, info in registry.items():
        models.append({
            "name": name,
            "max_output_tokens": info.get("max_output_tokens", 0),
            "context_window": info.get("context_window", 0),
            "report_capable": info.get("report_capable", False),
            "tier": info.get("tier", "unknown"),
        })
    models.sort(key=lambda m: (
        tier_priority.get(m["tier"], 5),
        -m["max_output_tokens"],
    ))
    return models


class PooledLLMClient:
    """LLM client wrapper with multi-key rotation and provider fallback.

    On credit exhaustion or rate limiting, automatically rotates to the
    next API key in the pool.  When all keys for the primary provider are
    exhausted, falls back to the alternate provider.

    Implements the same interface as ``LLMClient`` by delegating all calls
    to the active underlying client.
    """

    def __init__(
        self,
        primary_clients: list[LLMClient],
        fallback_clients: list[LLMClient] | None = None,
    ) -> None:
        self._primary = primary_clients
        self._fallback = fallback_clients or []
        self._all_clients = self._primary + self._fallback
        self._current_idx = 0
        self._exhausted: set[int] = set()

        if not self._all_clients:
            raise ValueError("No LLM clients available")

    @property
    def _active(self) -> LLMClient:
        return self._all_clients[self._current_idx]

    @property
    def provider_name(self) -> str:
        return self._active.provider_name

    @property
    def model_name(self) -> str:
        return self._active.model_name

    @property
    def max_output_tokens(self) -> int:
        return self._active.max_output_tokens

    def _rotate(self) -> bool:
        """Rotate to the next non-exhausted client. Returns False if all exhausted."""
        self._exhausted.add(self._current_idx)
        for i in range(len(self._all_clients)):
            if i not in self._exhausted:
                old_provider = self._all_clients[self._current_idx].provider_name
                self._current_idx = i
                new_provider = self._all_clients[i].provider_name
                if old_provider != new_provider:
                    logger.info(
                        "LLM key exhausted -- switching provider: %s -> %s (key %d/%d)",
                        old_provider, new_provider, i + 1, len(self._all_clients),
                    )
                else:
                    logger.info(
                        "LLM key exhausted -- rotating to next key (%d/%d) for %s",
                        i + 1, len(self._all_clients), new_provider,
                    )
                return True
        logger.warning("All %d LLM keys exhausted -- falling back to templates", len(self._all_clients))
        return False

    @staticmethod
    def _is_exhaustion_error(exc: Exception) -> bool:
        """Check if an exception indicates credit exhaustion or rate limiting."""
        msg = str(exc).lower()
        return any(indicator in msg for indicator in _EXHAUSTION_ERRORS)

    def _call_with_rotation(self, method_name: str, *args, **kwargs):
        """Call a method on the active client, rotating on exhaustion."""
        attempts = len(self._all_clients) - len(self._exhausted)
        for _ in range(max(attempts, 1)):
            try:
                method = getattr(self._active, method_name)
                return method(*args, **kwargs)
            except Exception as exc:
                if self._is_exhaustion_error(exc):
                    if not self._rotate():
                        raise  # all exhausted, re-raise
                else:
                    raise  # non-exhaustion error, don't rotate
        raise RuntimeError("All LLM keys exhausted")

    # -- Delegate all LLMClient interface methods --

    def generate(self, prompt: str, **kwargs) -> str:
        return self._call_with_rotation("generate", prompt, **kwargs)

    def generate_report(self, company_profile_json: str, **kwargs) -> str:
        return self._call_with_rotation("generate_report", company_profile_json, **kwargs)

    def propose_linked_entities(self, profile: dict, **kwargs) -> dict:
        return self._call_with_rotation("propose_linked_entities", profile, **kwargs)

    def score_sentiment(self, headlines: list[str], **kwargs) -> list[float]:
        return self._call_with_rotation("score_sentiment", headlines, **kwargs)


def create_llm_client(
    secrets: dict[str, str],
    *,
    provider: str | None = None,
    model: str | None = None,
) -> LLMClient | PooledLLMClient | None:
    """Create and return an LLM client based on configuration.

    If multiple API keys are provided (comma-separated in .env), returns
    a ``PooledLLMClient`` that auto-rotates on key exhaustion and falls
    back to the alternate provider.

    Resolution order for provider selection:
    1. Explicit ``provider`` argument
    2. ``LLM_PROVIDER`` environment variable
    3. ``llm_provider`` key in ``global_config.yml``
    4. Auto-detect: use whichever API key is available (Gemini first)

    Resolution order for model selection:
    1. Explicit ``model`` argument
    2. ``LLM_MODEL`` environment variable
    3. ``llm_model`` key in ``global_config.yml``
    4. Provider's default model (cost-optimized: Sonnet > Haiku > Opus)

    Parameters
    ----------
    secrets:
        Dict of loaded API secrets (from ``load_secrets()``).
    provider:
        Explicit provider override.  If ``None``, resolved from config.
    model:
        Explicit model override.  If ``None``, resolved from config.
        Use ``"auto"`` to let the provider pick its best model.

    Returns
    -------
    An ``LLMClient`` or ``PooledLLMClient`` instance, or ``None`` if no
    suitable API key is available.
    """
    from operator1.secrets_loader import get_key_pool

    # Resolve provider
    if provider is None:
        provider = os.environ.get("LLM_PROVIDER", "").strip().lower()
    if not provider:
        cfg = get_global_config()
        provider = str(cfg.get("llm_provider", "")).strip().lower()
    if not provider:
        provider = _auto_detect_provider(secrets)

    if provider not in _SUPPORTED_PROVIDERS:
        logger.warning(
            "Unknown llm_provider '%s'; supported: %s. Falling back to auto-detect.",
            provider,
            ", ".join(_SUPPORTED_PROVIDERS),
        )
        provider = _auto_detect_provider(secrets)

    # Resolve model
    if model is None:
        model = os.environ.get("LLM_MODEL", "").strip()
    if not model:
        cfg = get_global_config()
        model = str(cfg.get("llm_model", "")).strip()

    # Build primary clients (one per key in the pool)
    _KEY_MAP = {"gemini": "GEMINI_API_KEY", "claude": "ANTHROPIC_API_KEY", "openrouter": "OPENROUTER_API_KEY"}
    primary_key_name = _KEY_MAP.get(provider, "GEMINI_API_KEY")

    # Determine fallback providers (exclude the primary)
    _fallback_order = [p for p in ("gemini", "claude", "openrouter") if p != provider]

    primary_keys = get_key_pool(secrets, primary_key_name)

    primary_clients = []
    for key in primary_keys:
        client = _build_single_client(provider, key, model)
        if client:
            primary_clients.append(client)

    # Build fallback clients from all other providers that have keys
    fallback_clients = []
    fallback_provider = _fallback_order[0] if _fallback_order else "gemini"
    for fb_provider in _fallback_order:
        fb_key_name = _KEY_MAP.get(fb_provider, "")
        if not fb_key_name:
            continue
        fb_keys = get_key_pool(secrets, fb_key_name)
        for key in fb_keys:
            client = _build_single_client(fb_provider, key, "")
            if client:
                fallback_clients.append(client)

    all_clients = primary_clients + fallback_clients
    if not all_clients:
        logger.info("No LLM API keys available; LLM features disabled.")
        return None

    if len(all_clients) == 1:
        # Single key -- no need for pooling overhead
        client = all_clients[0]
        logger.info(
            "Using %s as LLM provider (model: %s, max_output: %d tokens)",
            client.provider_name, client.model_name, client.max_output_tokens,
        )
        return client

    # Multiple keys -- wrap in pooled client
    pooled = PooledLLMClient(primary_clients, fallback_clients)
    logger.info(
        "LLM key pool: %d primary (%s) + %d fallback (%s) keys. "
        "Active: %s / %s",
        len(primary_clients), provider,
        len(fallback_clients), fallback_provider,
        pooled.provider_name, pooled.model_name,
    )
    return pooled


def _auto_detect_provider(secrets: dict[str, str]) -> str:
    """Pick the first provider that has an API key available."""
    from operator1.secrets_loader import get_key_pool
    if get_key_pool(secrets, "GEMINI_API_KEY"):
        return "gemini"
    if get_key_pool(secrets, "ANTHROPIC_API_KEY"):
        return "claude"
    if get_key_pool(secrets, "OPENROUTER_API_KEY"):
        return "openrouter"
    return "gemini"


def _build_single_client(
    provider: str,
    api_key: str,
    model: str = "",
) -> LLMClient | None:
    """Build a single LLM client for a given provider and key."""
    if not api_key:
        return None

    if provider == "gemini":
        try:
            from operator1.clients.gemini import GeminiClient
            kwargs: dict = {"api_key": api_key}
            if model:
                kwargs["model"] = model
            return GeminiClient(**kwargs)
        except Exception as exc:
            logger.warning("Failed to build Gemini client: %s", exc)
            return None

    if provider == "claude":
        try:
            from operator1.clients.claude import ClaudeClient
            kwargs = {"api_key": api_key}
            if model:
                kwargs["model"] = model
            return ClaudeClient(**kwargs)
        except Exception as exc:
            logger.warning("Failed to build Claude client: %s", exc)
            return None

    if provider == "openrouter":
        try:
            from operator1.clients.openrouter import OpenRouterClient
            kwargs = {"api_key": api_key}
            if model:
                kwargs["model"] = model
            return OpenRouterClient(**kwargs)
        except Exception as exc:
            logger.warning("Failed to build OpenRouter client: %s", exc)
            return None

    return None
