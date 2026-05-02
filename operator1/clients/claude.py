"""Anthropic Claude API client.

Drop-in alternative to GeminiClient for linked entity discovery,
report generation, and sentiment scoring. Implements the LLMClient
interface with Claude-specific API handling.
"""

from __future__ import annotations

import logging
from typing import Any

from operator1.clients.llm_base import (
    LLMClient,
    CLAUDE_MODELS,
    validate_model,
    get_best_model,
)
from operator1.constants import CLAUDE_BASE_URL

logger = logging.getLogger(__name__)

# Default model: claude-sonnet-4 is a good balance of quality and speed,
# with high output token support for report generation.
_DEFAULT_MODEL = "claude-sonnet-4-20250514"


class ClaudeClient(LLMClient):
    """Wrapper around the Anthropic Messages API.

    All methods are wrapped in try/except and return sensible fallbacks
    on failure -- the LLM is never a hard dependency for the pipeline.

    The client automatically validates the chosen model against a
    registry of known models and their capabilities. If the model is
    not recognized, it falls back to the best available model.

    Rate limiting and retry logic are handled by the LLMClient base
    class (_execute_request), which respects per-host rate limits and
    Retry-After headers.

    Parameters
    ----------
    api_key:
        Anthropic API key (from secrets).
    base_url:
        Override for testing.
    model:
        Model name to use for generation. If ``"auto"``, picks the best
        report-capable model from the registry.
    """

    def __init__(
        self,
        api_key: str,
        base_url: str = CLAUDE_BASE_URL,
        model: str = _DEFAULT_MODEL,
    ) -> None:
        self._api_key = api_key
        self._base_url = base_url.rstrip("/")
        self._last_entity_proposals_raw: dict = {}

        # Model validation / auto-selection
        if model == "auto":
            model = get_best_model("Claude", CLAUDE_MODELS)
            logger.info("Claude auto-selected model: %s", model)

        self._model = validate_model(model, CLAUDE_MODELS, "Claude")

        # Cache model capabilities
        model_info = CLAUDE_MODELS.get(self._model, {})
        self._max_output_tokens = model_info.get("max_output_tokens", 8192)

    @property
    def provider_name(self) -> str:
        return "Claude"

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
        system_prompt: str = "",
    ) -> dict[str, Any]:
        """Build Claude API request arguments.

        Claude uses the Messages API at /v1/messages. Authentication
        is via the ``x-api-key`` header (NOT a query parameter -- unlike
        Gemini, the key is never in the URL). The ``anthropic-version``
        header is required and specifies the API version.

        When ``system_prompt`` is provided, it is sent via the
        ``system`` top-level parameter in the Messages API payload.
        """
        url = f"{self._base_url}/messages"
        headers = {
            "x-api-key": self._api_key,
            "anthropic-version": "2023-06-01",
            "content-type": "application/json",
        }
        payload: dict[str, Any] = {
            "model": self._model,
            "max_tokens": max_output_tokens,
            "temperature": temperature,
            "messages": [
                {"role": "user", "content": prompt},
            ],
        }
        if system_prompt:
            payload["system"] = system_prompt
        return {"url": url, "json": payload, "headers": headers}

    def _parse_response(self, data: dict[str, Any]) -> str:
        """Extract text from Claude's response structure.

        Claude returns: content[0].text (where content[0].type == "text")

        Also checks for stop_reason to detect truncation or errors.
        """
        # Check stop reason
        stop_reason = data.get("stop_reason", "")
        if stop_reason == "max_tokens":
            logger.warning(
                "Claude response truncated (stop_reason=max_tokens). "
                "The output hit the max_tokens limit. Consider increasing "
                "max_output_tokens or using a model with higher limits."
            )
        elif stop_reason not in ("end_turn", "stop_sequence", ""):
            logger.warning("Claude unexpected stop_reason: %s", stop_reason)

        # Check for errors in the response body
        if data.get("type") == "error":
            error_msg = data.get("error", {}).get("message", "unknown error")
            logger.error("Claude API error: %s", error_msg)
            return ""

        # Extract text from content blocks
        content_blocks = data.get("content", [])
        if not content_blocks:
            logger.warning("Claude returned no content blocks")
            return ""

        for block in content_blocks:
            if block.get("type") == "text":
                return block.get("text", "")

        logger.warning("Claude returned no text blocks in response")
        return ""
