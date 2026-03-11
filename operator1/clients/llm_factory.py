"""Factory for creating LLM client instances.

Reads ``llm_provider`` from global config (or the ``LLM_PROVIDER`` env
variable) and returns the matching client.  Supported values:

- ``gemini`` (default) -- Google Gemini via generativelanguage API
- ``claude`` -- Anthropic Claude via Messages API

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

    registry = GEMINI_MODELS if provider == "gemini" else CLAUDE_MODELS

    tier_priority = {"flagship": 0, "balanced": 1, "stable": 2, "fast": 3, "preview": 4}
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


def create_llm_client(
    secrets: dict[str, str],
    *,
    provider: str | None = None,
    model: str | None = None,
) -> LLMClient | None:
    """Create and return an LLM client based on configuration.

    Resolution order for provider selection:
    1. Explicit ``provider`` argument
    2. ``LLM_PROVIDER`` environment variable
    3. ``llm_provider`` key in ``global_config.yml``
    4. Auto-detect: use whichever API key is available (Gemini first)

    Resolution order for model selection:
    1. Explicit ``model`` argument
    2. ``LLM_MODEL`` environment variable
    3. ``llm_model`` key in ``global_config.yml``
    4. Provider's default model

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
    An ``LLMClient`` instance (GeminiClient or ClaudeClient), or ``None``
    if no suitable API key is available.
    """
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
    # If still empty, let the client use its default (don't pass model kwarg)

    # Build the client
    if provider == "claude":
        return _build_claude(secrets, model)
    elif provider == "openrouter":
        return _build_openrouter(secrets, model)
    else:
        return _build_gemini(secrets, model)


def _auto_detect_provider(secrets: dict[str, str]) -> str:
    """Pick the first provider that has an API key available."""
    if secrets.get("GEMINI_API_KEY"):
        return "gemini"
    if secrets.get("ANTHROPIC_API_KEY"):
        return "claude"
    if secrets.get("OPENROUTER_API_KEY"):
        return "openrouter"
    # Default to gemini even without a key (caller handles None)
    return "gemini"


def _build_gemini(secrets: dict[str, str], model: str = "") -> LLMClient | None:
    """Build a GeminiClient if an API key is present."""
    api_key = secrets.get("GEMINI_API_KEY", "")
    if not api_key:
        logger.info("No GEMINI_API_KEY found; LLM features disabled.")
        return None
    from operator1.clients.gemini import GeminiClient

    kwargs: dict = {"api_key": api_key}
    if model:
        kwargs["model"] = model

    client = GeminiClient(**kwargs)
    logger.info(
        "Using Gemini as LLM provider (model: %s, max_output: %d tokens)",
        client.model_name, client.max_output_tokens,
    )
    return client


def _build_claude(secrets: dict[str, str], model: str = "") -> LLMClient | None:
    """Build a ClaudeClient if an API key is present."""
    api_key = secrets.get("ANTHROPIC_API_KEY", "")
    if not api_key:
        logger.info("No ANTHROPIC_API_KEY found; LLM features disabled.")
        return None
    from operator1.clients.claude import ClaudeClient

    kwargs: dict = {"api_key": api_key}
    if model:
        kwargs["model"] = model

    client = ClaudeClient(**kwargs)
    logger.info(
        "Using Claude as LLM provider (model: %s, max_output: %d tokens)",
        client.model_name, client.max_output_tokens,
    )
    return client


def _build_openrouter(secrets: dict[str, str], model: str = "") -> LLMClient | None:
    """Build an OpenRouterClient if an API key is present."""
    api_key = secrets.get("OPENROUTER_API_KEY", "")
    if not api_key:
        logger.info("No OPENROUTER_API_KEY found; LLM features disabled.")
        return None
    from operator1.clients.openrouter import OpenRouterClient

    kwargs: dict = {"api_key": api_key}
    if model:
        kwargs["model"] = model

    client = OpenRouterClient(**kwargs)
    logger.info(
        "Using OpenRouter as LLM provider (model: %s, max_output: %d tokens)",
        client.model_name, client.max_output_tokens,
    )
    return client
