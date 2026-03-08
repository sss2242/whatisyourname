"""Tests for per-market taxonomy hints in the LLM filing extractor.

Verifies that:
1. Every registered PIT market has a taxonomy hint entry
2. The hint is correctly injected into the extraction prompt
3. Unknown markets produce an empty hint (graceful fallback)
4. The get_taxonomy_hint() public API works
"""

from __future__ import annotations

import pytest

from operator1.clients.llm_filing_extractor import (
    ExtractionResult,
    LLMFilingExtractor,
    _EXTRACTION_PROMPT,
    _MARKET_TAXONOMY_HINTS,
    get_taxonomy_hint,
)
from operator1.clients.pit_registry import MARKETS


# ---------------------------------------------------------------------------
# 1. Coverage: every registered market should have a taxonomy hint
# ---------------------------------------------------------------------------

class TestTaxonomyHintCoverage:
    """Ensure taxonomy hints exist for all PIT registry markets."""

    def test_all_registered_markets_have_hints(self):
        missing = [
            mid for mid in MARKETS
            if mid not in _MARKET_TAXONOMY_HINTS
        ]
        assert not missing, (
            f"Markets missing taxonomy hints: {missing}. "
            "Add entries to _MARKET_TAXONOMY_HINTS in llm_filing_extractor.py."
        )

    def test_no_empty_hints(self):
        for market_id, hint in _MARKET_TAXONOMY_HINTS.items():
            assert hint.strip(), f"Taxonomy hint for '{market_id}' is empty"

    def test_hints_mention_currency(self):
        """Each hint should mention the filing currency."""
        for market_id, hint in _MARKET_TAXONOMY_HINTS.items():
            assert "Currency:" in hint, (
                f"Taxonomy hint for '{market_id}' does not mention Currency"
            )

    def test_hints_mention_standard(self):
        """Each hint should mention the accounting standard."""
        for market_id, hint in _MARKET_TAXONOMY_HINTS.items():
            assert "Standard:" in hint, (
                f"Taxonomy hint for '{market_id}' does not mention accounting Standard"
            )

    def test_hints_mention_language(self):
        """Each hint should mention the filing language."""
        for market_id, hint in _MARKET_TAXONOMY_HINTS.items():
            assert "Language:" in hint, (
                f"Taxonomy hint for '{market_id}' does not mention Language"
            )


# ---------------------------------------------------------------------------
# 2. get_taxonomy_hint() API
# ---------------------------------------------------------------------------

class TestGetTaxonomyHint:
    """Test the public get_taxonomy_hint() function."""

    def test_known_market(self):
        hint = get_taxonomy_hint("kr_dart")
        assert "K-IFRS" in hint
        assert "KRW" in hint
        assert "Korean" in hint

    def test_unknown_market_returns_empty(self):
        hint = get_taxonomy_hint("xx_unknown")
        assert hint == ""

    def test_empty_market_returns_empty(self):
        hint = get_taxonomy_hint("")
        assert hint == ""

    @pytest.mark.parametrize("market_id", [
        "us_sec_edgar", "jp_jquants", "cn_sse", "br_cvm",
        "tw_mops", "hk_hkex", "sa_tadawul",
    ])
    def test_non_english_markets_have_local_terms(self, market_id):
        """Markets with non-English filings should include local-language keywords."""
        hint = get_taxonomy_hint(market_id)
        # These markets should have non-ASCII characters in their hints
        # (Japanese, Korean, Chinese, Arabic, etc.) or specific local terms
        assert len(hint) > 100, f"Hint for {market_id} seems too short"


# ---------------------------------------------------------------------------
# 3. Prompt injection
# ---------------------------------------------------------------------------

class TestPromptInjection:
    """Verify taxonomy hints are injected into the extraction prompt."""

    def test_prompt_has_taxonomy_placeholder(self):
        """The prompt template must contain {taxonomy_hint}."""
        assert "{taxonomy_hint}" in _EXTRACTION_PROMPT

    def test_prompt_format_with_hint(self):
        """Formatting the prompt with a hint should embed it."""
        hint = get_taxonomy_hint("kr_dart")
        prompt = _EXTRACTION_PROMPT.format(
            taxonomy_hint=f"\n{hint}\n",
            text="sample filing text",
        )
        assert "K-IFRS" in prompt
        assert "sample filing text" in prompt

    def test_prompt_format_without_hint(self):
        """Formatting with empty hint should still produce valid prompt."""
        prompt = _EXTRACTION_PROMPT.format(
            taxonomy_hint="",
            text="sample filing text",
        )
        assert "sample filing text" in prompt
        assert "Return ONLY a valid JSON" in prompt


# ---------------------------------------------------------------------------
# 4. Integration: _extract_via_llm uses the hint
# ---------------------------------------------------------------------------

class _FakeLLM:
    """Fake LLM client that captures the prompt it receives."""

    def __init__(self):
        self.last_prompt = ""

    def generate(self, prompt: str) -> str:
        self.last_prompt = prompt
        return '{"revenue": 1000, "net_income": 100, "report_date": "2024-12-31", "currency": "KRW", "period_type": "annual"}'


class TestExtractViaLLMIntegration:
    """Test that _extract_via_llm actually injects taxonomy hints."""

    def test_korean_hint_injected(self):
        fake_llm = _FakeLLM()
        extractor = LLMFilingExtractor(llm_client=fake_llm)
        result = extractor._extract_via_llm(
            text="sample Korean filing text",
            market_id="kr_dart",
            source_format="pdf",
        )
        assert result.success
        # The prompt sent to the LLM should contain Korean taxonomy hints
        assert "\ub9e4\ucd9c\uc561" in fake_llm.last_prompt  # Korean for "revenue"
        assert "K-IFRS" in fake_llm.last_prompt

    def test_unknown_market_still_works(self):
        fake_llm = _FakeLLM()
        extractor = LLMFilingExtractor(llm_client=fake_llm)
        result = extractor._extract_via_llm(
            text="sample filing text",
            market_id="xx_unknown",
            source_format="pdf",
        )
        assert result.success
        # Should still have the core prompt structure
        assert "Return ONLY a valid JSON" in fake_llm.last_prompt

    def test_no_llm_client_returns_error(self):
        extractor = LLMFilingExtractor(llm_client=None)
        result = extractor._extract_via_llm(
            text="sample text",
            market_id="us_sec_edgar",
            source_format="pdf",
        )
        assert not result.success
        assert "No LLM client" in result.error
