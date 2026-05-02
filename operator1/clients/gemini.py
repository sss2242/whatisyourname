"""Google Gemini API client.

Used for linked entity discovery and report generation.
Implements the LLMClient interface with Gemini-specific API handling.
"""

from __future__ import annotations

import logging
from typing import Any

from operator1.clients.llm_base import (
    LLMClient,
    GEMINI_MODELS,
    validate_model,
    get_best_model,
)
from operator1.constants import GEMINI_BASE_URL

logger = logging.getLogger(__name__)

# Default model: gemini-2.0-flash is stable and fast with 1M context
_DEFAULT_MODEL = "gemini-2.0-flash"


class GeminiClient(LLMClient):
    """Wrapper around the Gemini generative-language API.

    All methods are wrapped in try/except and return sensible fallbacks
    on failure -- Gemini is never a hard dependency for the pipeline.

    The client automatically validates the chosen model against a
    registry of known models and their capabilities. If the model is
    not recognized, it falls back to the best available model.

    Rate limiting and retry logic are handled by the LLMClient base
    class (_execute_request), which respects per-host rate limits and
    Retry-After headers.

    Parameters
    ----------
    api_key:
        Gemini API key (from secrets).
    base_url:
        Override for testing.
    model:
        Model name to use for generation. If ``"auto"``, picks the best
        report-capable model from the registry.
    """

    def __init__(
        self,
        api_key: str,
        base_url: str = GEMINI_BASE_URL,
        model: str = _DEFAULT_MODEL,
    ) -> None:
        self._api_key = api_key
        self._base_url = base_url.rstrip("/")
        self._last_entity_proposals_raw: dict = {}

        # Model validation / auto-selection
        if model == "auto":
            model = get_best_model("Gemini", GEMINI_MODELS)
            logger.info("Gemini auto-selected model: %s", model)

        self._model = validate_model(model, GEMINI_MODELS, "Gemini")

        # Cache model capabilities
        model_info = GEMINI_MODELS.get(self._model, {})
        self._max_output_tokens = model_info.get("max_output_tokens", 8192)

    @property
    def provider_name(self) -> str:
        return "Gemini"

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
        """Build Gemini API request arguments.

        Gemini uses the generateContent endpoint with API key as a
        query parameter. The payload uses the ``contents`` array with
        ``parts`` containing the prompt text. Generation config is
        passed via ``generationConfig``.

        When ``system_prompt`` is provided, it is sent via the
        ``systemInstruction`` top-level field (Gemini v1beta API).
        """
        url = (
            f"{self._base_url}/models/{self._model}:generateContent"
            f"?key={self._api_key}"
        )
        payload: dict[str, Any] = {
            "contents": [{"parts": [{"text": prompt}]}],
            "generationConfig": {
                "temperature": temperature,
                "maxOutputTokens": max_output_tokens,
            },
        }
        if system_prompt:
            payload["systemInstruction"] = {
                "parts": [{"text": system_prompt}],
            }
        return {"url": url, "json": payload}

    def _parse_response(self, data: dict[str, Any]) -> str:
        """Extract text from Gemini's response structure.

        Gemini returns: candidates[0].content.parts[0].text
        """
        candidates = data.get("candidates", [])
        if not candidates:
            logger.warning("Gemini returned no candidates")
            return ""

        # Check for safety filtering
        finish_reason = candidates[0].get("finishReason", "")
        if finish_reason == "SAFETY":
            logger.warning(
                "Gemini response blocked by safety filter (finishReason=SAFETY)"
            )
            return ""

        content = candidates[0].get("content", {})
        parts = content.get("parts", [])
        if not parts:
            logger.warning("Gemini returned empty parts in response")
            return ""
        return parts[0].get("text", "")
