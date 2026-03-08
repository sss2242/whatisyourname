"""Tests for operator1.clients.openrouter.OpenRouterClient."""

from __future__ import annotations

import pytest


class TestOpenRouterClient:
    def test_import(self):
        from operator1.clients.openrouter import OpenRouterClient
        assert OpenRouterClient is not None

    def test_instantiation(self):
        from operator1.clients.openrouter import OpenRouterClient
        client = OpenRouterClient(api_key="test_key_123")
        assert client is not None

    def test_provider_name(self):
        from operator1.clients.openrouter import OpenRouterClient
        client = OpenRouterClient(api_key="test")
        assert client.provider_name == "OpenRouter"

    def test_model_name(self):
        from operator1.clients.openrouter import OpenRouterClient
        client = OpenRouterClient(api_key="test")
        # Default model should be set
        assert client.model_name
        assert isinstance(client.model_name, str)

    def test_max_output_tokens(self):
        from operator1.clients.openrouter import OpenRouterClient
        client = OpenRouterClient(api_key="test")
        assert client.max_output_tokens > 0

    def test_model_registry_exists(self):
        from operator1.clients.openrouter import OPENROUTER_MODELS
        assert isinstance(OPENROUTER_MODELS, dict)
        assert len(OPENROUTER_MODELS) > 0

    def test_model_registry_has_free_tier(self):
        from operator1.clients.openrouter import OPENROUTER_MODELS
        free_models = [k for k, v in OPENROUTER_MODELS.items() if v.get("tier") == "free"]
        assert len(free_models) >= 1, "Should have at least one free-tier model"

    def test_all_models_report_capable(self):
        from operator1.clients.openrouter import OPENROUTER_MODELS
        for name, info in OPENROUTER_MODELS.items():
            assert info.get("report_capable") is True, f"{name} should be report_capable"

    def test_all_models_have_required_keys(self):
        from operator1.clients.openrouter import OPENROUTER_MODELS
        required = {"max_output_tokens", "context_window", "report_capable", "tier"}
        for name, info in OPENROUTER_MODELS.items():
            missing = required - set(info.keys())
            assert not missing, f"{name} missing keys: {missing}"

    def test_build_request_args_format(self):
        from operator1.clients.openrouter import OpenRouterClient
        client = OpenRouterClient(api_key="test_key")
        args = client._build_request_args("Hello world", max_output_tokens=1024)
        assert "url" in args
        assert "json" in args
        assert "headers" in args
        # URL should point to chat completions
        assert "/chat/completions" in args["url"]
        # Headers should have Bearer auth
        assert "Authorization" in args["headers"]
        assert args["headers"]["Authorization"] == "Bearer test_key"
        # Payload should be OpenAI-compatible
        payload = args["json"]
        assert "model" in payload
        assert "messages" in payload
        assert isinstance(payload["messages"], list)
        assert payload["messages"][0]["role"] == "user"
        assert payload["messages"][0]["content"] == "Hello world"
        assert payload["max_tokens"] == 1024

    def test_parse_response_success(self):
        from operator1.clients.openrouter import OpenRouterClient
        client = OpenRouterClient(api_key="test")
        response = {
            "choices": [
                {
                    "message": {"content": "Hello from OpenRouter"},
                    "finish_reason": "stop",
                }
            ]
        }
        result = client._parse_response(response)
        assert result == "Hello from OpenRouter"

    def test_parse_response_empty_choices(self):
        from operator1.clients.openrouter import OpenRouterClient
        client = OpenRouterClient(api_key="test")
        result = client._parse_response({"choices": []})
        assert result == ""

    def test_parse_response_error(self):
        from operator1.clients.openrouter import OpenRouterClient
        client = OpenRouterClient(api_key="test")
        result = client._parse_response({"error": {"message": "Rate limited"}})
        assert result == ""

    def test_parse_response_truncated(self):
        from operator1.clients.openrouter import OpenRouterClient
        client = OpenRouterClient(api_key="test")
        response = {
            "choices": [
                {
                    "message": {"content": "Partial text..."},
                    "finish_reason": "length",
                }
            ]
        }
        # Should still return the content even if truncated
        result = client._parse_response(response)
        assert result == "Partial text..."

    def test_specific_model_selection(self):
        from operator1.clients.openrouter import OpenRouterClient, OPENROUTER_MODELS
        # Pick first model from registry
        model_name = next(iter(OPENROUTER_MODELS))
        client = OpenRouterClient(api_key="test", model=model_name)
        assert client.model_name == model_name

    def test_auto_model_selection(self):
        from operator1.clients.openrouter import OpenRouterClient
        client = OpenRouterClient(api_key="test", model="auto")
        assert client.model_name  # Should pick something

    def test_factory_integration(self):
        """Test that llm_factory recognizes openrouter provider."""
        from operator1.clients.llm_factory import get_available_models
        models = get_available_models("openrouter")
        assert len(models) > 0
        for m in models:
            assert "name" in m or "model" in m or isinstance(m, dict)
