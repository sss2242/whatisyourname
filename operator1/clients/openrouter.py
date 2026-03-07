"""OpenRouter API client -- unified gateway to 200+ LLM models.

Drop-in alternative to GeminiClient/ClaudeClient for linked entity
discovery, report generation, and sentiment scoring. Uses the
OpenAI-compatible chat completions API through OpenRouter's gateway.

OpenRouter provides access to models from Google, Anthropic, Meta,
Mistral, DeepSeek, and others through a single API key.

API docs: https://openrouter.ai/docs
"""

from __future__ import annotations

import logging
from typing import Any

from operator1.clients.llm_base import (
    LLMClient,
    validate_model,
    get_best_model,
)
from operator1.constants import OPENROUTER_BASE_URL

logger = logging.getLogger(__name__)

# Model registry -- curated set of report-capable models available on
# OpenRouter.  The ":free" suffix indicates free-tier models (no cost).
OPENROUTER_MODELS: dict[str, dict[str, Any]] = {
    # --- Free tier (zero cost) ---
    "qwen/qwen3-coder:free": {
        "max_output_tokens": 262000,
        "context_window": 262000,
        "report_capable": True,
        "tier": "free",
    },
    "openai/gpt-oss-120b:free": {
        "max_output_tokens": 131072,
        "context_window": 131072,
        "report_capable": True,
        "tier": "free",
    },
    "mistralai/mistral-small-3.1-24b-instruct:free": {
        "max_output_tokens": 32768,
        "context_window": 96000,
        "report_capable": True,
        "tier": "free",
    },
    "google/gemma-3-4b-it:free": {
        "max_output_tokens": 8192,
        "context_window": 131072,
        "report_capable": True,
        "tier": "free",
    },
    # --- Paid models (available with credits) ---
    "anthropic/claude-sonnet-4": {
        "max_output_tokens": 64000,
        "context_window": 200000,
        "report_capable": True,
        "tier": "balanced",
    },
    "google/gemini-2.0-flash-001": {
        "max_output_tokens": 8192,
        "context_window": 1048576,
        "report_capable": True,
        "tier": "stable",
    },
    "deepseek/deepseek-chat": {
        "max_output_tokens": 8192,
        "context_window": 65536,
        "report_capable": True,
        "tier": "fast",
    },
}

# Default: pick a free-tier model with high output for report generation
_DEFAULT_MODEL = "qwen/qwen3-coder:free"


class OpenRouterClient(LLMClient):
    """Wrapper around the OpenRouter chat completions API.

    Uses the OpenAI-compatible endpoint at openrouter.ai. All methods
    are wrapped in try/except and return sensible fallbacks on failure --
    the LLM is never a hard dependency for the pipeline.

    Parameters
    ----------
    api_key:
        OpenRouter API key (from secrets).
    base_url:
        Override for testing.
    model:
        Model name to use. If ``"auto"``, picks the best
        report-capable model from the registry.
    """

    def __init__(
        self,
        api_key: str,
        base_url: str = OPENROUTER_BASE_URL,
        model: str = _DEFAULT_MODEL,
    ) -> None:
        self._api_key = api_key
        self._base_url = base_url.rstrip("/")
        self._last_entity_proposals_raw: dict = {}

        # Model validation / auto-selection
        if model == "auto":
            model = get_best_model("OpenRouter", OPENROUTER_MODELS)
            logger.info("OpenRouter auto-selected model: %s", model)

        self._model = validate_model(model, OPENROUTER_MODELS, "OpenRouter")

        # Cache model capabilities
        model_info = OPENROUTER_MODELS.get(self._model, {})
        self._max_output_tokens = model_info.get("max_output_tokens", 8192)

    @property
    def provider_name(self) -> str:
        return "OpenRouter"

    @property
    def model_name(self) -> str:
        return self._model

    @property
    def max_output_tokens(self) -> int:
        return self._max_output_tokens

    # ------------------------------------------------------------------
    # Provider-specific request building
    # ------------------------------------------------------------------

    def _build_request_args(
        self,
        prompt: str,
        *,
        max_output_tokens: int = 8192,
        temperature: float = 0.7,
    ) -> dict[str, Any]:
        """Build OpenRouter API request arguments.

        OpenRouter uses the OpenAI-compatible chat completions endpoint.
        Authentication is via Bearer token in the Authorization header.
        The HTTP-Referer and X-Title headers are required by OpenRouter
        for app identification.
        """
        url = f"{self._base_url}/chat/completions"
        headers = {
            "Authorization": f"Bearer {self._api_key}",
            "HTTP-Referer": "https://github.com/oso2424/githubu-isu-meanu",
            "X-Title": "Operator1 Financial Analysis Pipeline",
            "Content-Type": "application/json",
        }
        payload = {
            "model": self._model,
            "messages": [
                {"role": "user", "content": prompt},
            ],
            "max_tokens": max_output_tokens,
            "temperature": temperature,
        }
        return {"url": url, "json": payload, "headers": headers}

    def _parse_response(self, data: dict[str, Any]) -> str:
        """Extract text from OpenRouter's OpenAI-compatible response.

        Response format: choices[0].message.content
        """
        # Check for API errors
        if "error" in data:
            error = data["error"]
            error_msg = error.get("message", str(error)) if isinstance(error, dict) else str(error)
            logger.error("OpenRouter API error: %s", error_msg)
            return ""

        choices = data.get("choices", [])
        if not choices:
            logger.warning("OpenRouter returned no choices")
            return ""

        message = choices[0].get("message", {})
        content = message.get("content", "")

        # Check for truncation
        finish_reason = choices[0].get("finish_reason", "")
        if finish_reason == "length":
            logger.warning(
                "OpenRouter response truncated (finish_reason=length). "
                "Consider increasing max_tokens or using a model with higher limits."
            )

        return content or ""
