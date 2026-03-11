"""OpenRouter API client -- 200+ models via a single API key.

OpenRouter provides access to Google, Anthropic, Meta, Mistral, DeepSeek
and many other models through a unified OpenAI-compatible API.

Free tier available at https://openrouter.ai/

Implements the LLMClient interface with OpenRouter-specific API handling.
Uses the OpenAI-compatible chat completions endpoint.
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

# Default model: free tier Gemini via OpenRouter
_DEFAULT_MODEL = "google/gemini-2.0-flash-exp:free"

# Registry of known OpenRouter models with capabilities
OPENROUTER_MODELS: dict[str, dict[str, Any]] = {
    "google/gemini-2.0-flash-exp:free": {
        "max_output_tokens": 8192,
        "context_window": 1048576,
        "report_capable": True,
        "tier": "balanced",
    },
    "anthropic/claude-sonnet-4": {
        "max_output_tokens": 64000,
        "context_window": 200000,
        "report_capable": True,
        "tier": "balanced",
    },
    "meta-llama/llama-3.1-405b-instruct": {
        "max_output_tokens": 4096,
        "context_window": 131072,
        "report_capable": True,
        "tier": "stable",
    },
    "mistralai/mistral-large-latest": {
        "max_output_tokens": 8192,
        "context_window": 128000,
        "report_capable": True,
        "tier": "stable",
    },
    "deepseek/deepseek-chat": {
        "max_output_tokens": 8192,
        "context_window": 64000,
        "report_capable": True,
        "tier": "fast",
    },
}


class OpenRouterClient(LLMClient):
    """Wrapper around the OpenRouter API (OpenAI-compatible).

    All methods are wrapped in try/except and return sensible fallbacks
    on failure -- the LLM is never a hard dependency for the pipeline.

    Parameters
    ----------
    api_key:
        OpenRouter API key.
    base_url:
        Override for testing.
    model:
        Model identifier (e.g. "google/gemini-2.0-flash-exp:free").
        Use "auto" to pick the best available model.
    """

    def __init__(
        self,
        api_key: str,
        base_url: str = OPENROUTER_BASE_URL,
        model: str = _DEFAULT_MODEL,
    ) -> None:
        self._api_key = api_key
        self._base_url = base_url.rstrip("/")

        # Model validation / auto-selection
        if model == "auto" or not model:
            model = get_best_model("OpenRouter", OPENROUTER_MODELS)
            logger.info("OpenRouter auto-selected model: %s", model)

        self._model = validate_model(model, OPENROUTER_MODELS, "OpenRouter")

        # Cache model capabilities
        model_info = OPENROUTER_MODELS.get(self._model, {})
        self._max_output_tokens = model_info.get("max_output_tokens", 8192)

        logger.info(
            "OpenRouter client initialized: model=%s, max_output=%d",
            self._model, self._max_output_tokens,
        )

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
        Auth is via Bearer token in the Authorization header.
        """
        url = f"{self._base_url}/chat/completions"
        payload = {
            "model": self._model,
            "messages": [{"role": "user", "content": prompt}],
            "max_tokens": min(max_output_tokens, self._max_output_tokens),
            "temperature": temperature,
        }
        headers = {
            "Authorization": f"Bearer {self._api_key}",
            "Content-Type": "application/json",
            "HTTP-Referer": "https://github.com/Abdu2024/OP-1",
            "X-Title": "Operator1 Financial Analysis",
        }
        return {"url": url, "json": payload, "headers": headers}

    def _parse_response(self, data: dict[str, Any]) -> str:
        """Extract text from OpenRouter's OpenAI-compatible response.

        Response format: choices[0].message.content
        """
        choices = data.get("choices", [])
        if not choices:
            logger.warning("OpenRouter returned no choices")
            return ""

        message = choices[0].get("message", {})
        content = message.get("content", "")

        if not content:
            # Check for error in response
            error = data.get("error", {})
            if error:
                logger.warning("OpenRouter error: %s", error.get("message", ""))
            return ""

        return content
