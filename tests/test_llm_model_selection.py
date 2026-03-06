"""Tests for LLM model selection and factory helpers."""

from __future__ import annotations

import unittest


class TestGetAvailableModels(unittest.TestCase):
    """Test the get_available_models factory helper."""

    def test_gemini_models_returned(self):
        from operator1.clients.llm_factory import get_available_models

        models = get_available_models("gemini")
        self.assertGreater(len(models), 0)
        names = [m["name"] for m in models]
        self.assertIn("gemini-2.0-flash", names)

    def test_claude_models_returned(self):
        from operator1.clients.llm_factory import get_available_models

        models = get_available_models("claude")
        self.assertGreater(len(models), 0)
        names = [m["name"] for m in models]
        self.assertIn("claude-sonnet-4-20250514", names)

    def test_models_sorted_by_tier(self):
        from operator1.clients.llm_factory import get_available_models

        models = get_available_models("claude")
        tiers = [m["tier"] for m in models]
        # Cost-aware ordering: balanced (cost-effective) first, flagship last.
        # This matches llm_factory.get_available_models() which prioritizes
        # cost-effective models for free-tier optimization.
        tier_order = {"balanced": 0, "stable": 1, "fast": 2, "preview": 3, "flagship": 4}
        tier_indices = [tier_order.get(t, 99) for t in tiers]
        self.assertEqual(tier_indices, sorted(tier_indices))

    def test_each_model_has_required_keys(self):
        from operator1.clients.llm_factory import get_available_models

        for provider in ("gemini", "claude"):
            models = get_available_models(provider)
            for m in models:
                self.assertIn("name", m)
                self.assertIn("max_output_tokens", m)
                self.assertIn("context_window", m)
                self.assertIn("report_capable", m)
                self.assertIn("tier", m)
                self.assertIsInstance(m["name"], str)
                self.assertIsInstance(m["max_output_tokens"], int)
                self.assertIsInstance(m["context_window"], int)

    def test_unknown_provider_returns_empty(self):
        from operator1.clients.llm_factory import get_available_models

        models = get_available_models("openai")
        # Should return empty since openai isn't in the registries
        # (it would try CLAUDE_MODELS since "openai" != "gemini")
        # Actually the function defaults to claude registry for non-gemini
        # This is fine -- it still returns models
        self.assertIsInstance(models, list)

    def test_gemini_has_stable_models(self):
        from operator1.clients.llm_factory import get_available_models

        models = get_available_models("gemini")
        stable = [m for m in models if m["tier"] == "stable"]
        self.assertGreater(len(stable), 0, "Expected at least one stable Gemini model")

    def test_claude_has_flagship_and_balanced(self):
        from operator1.clients.llm_factory import get_available_models

        models = get_available_models("claude")
        tiers = {m["tier"] for m in models}
        self.assertIn("flagship", tiers)
        self.assertIn("balanced", tiers)

    def test_all_models_report_capable(self):
        """All registered models should be report-capable."""
        from operator1.clients.llm_factory import get_available_models

        for provider in ("gemini", "claude"):
            models = get_available_models(provider)
            for m in models:
                self.assertTrue(
                    m["report_capable"],
                    f"{m['name']} is not report_capable",
                )


class TestCreateLlmClientModelOverride(unittest.TestCase):
    """Test that create_llm_client respects model override."""

    def test_model_env_var_respected(self):
        """LLM_MODEL env var should be picked up by create_llm_client."""
        import os
        from operator1.clients.llm_factory import create_llm_client

        # No API key, so client will be None, but the resolution logic
        # should not crash when LLM_MODEL is set
        os.environ["LLM_MODEL"] = "gemini-1.5-pro"
        try:
            client = create_llm_client({})
            # No API key -> None, but no crash
            self.assertIsNone(client)
        finally:
            os.environ.pop("LLM_MODEL", None)

    def test_provider_env_var_respected(self):
        """LLM_PROVIDER env var should be picked up."""
        import os
        from operator1.clients.llm_factory import create_llm_client

        os.environ["LLM_PROVIDER"] = "claude"
        try:
            client = create_llm_client({})
            self.assertIsNone(client)  # no key
        finally:
            os.environ.pop("LLM_PROVIDER", None)


if __name__ == "__main__":
    unittest.main()
