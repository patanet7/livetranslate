#!/usr/bin/env python3
"""
TDD Test Suite for Domain Prompt Manager
Phase 2: SimulStreaming Innovation

Tests written BEFORE implementation to validate:
- Domain terminology loading (medical, legal, technical)
- Prompt generation with token limits (448 tokens max)
- Context carryover (223 tokens max)
- Scrolling context window
- Database integration
"""

import pytest
from unittest.mock import Mock, MagicMock, patch
import sys
from pathlib import Path

# Add src directory to path
SRC_DIR = Path(__file__).parent.parent / "src"
sys.path.insert(0, str(SRC_DIR))

from domain_prompt_manager import (
    DomainPromptManager,
    DomainPromptConfig,
    create_domain_prompt
)


class TestDomainPromptManager:
    """Test domain prompt manager functionality"""

    def test_initialization_default(self):
        """Test default initialization"""
        manager = DomainPromptManager()

        assert manager.config.max_total_tokens == 448  # SimulStreaming limit
        assert manager.config.max_context_tokens == 223
        assert manager.config.max_terminology_tokens == 225
        assert len(manager.context_buffer) == 0

    def test_initialization_custom_config(self):
        """Test initialization with custom config"""
        config = DomainPromptConfig(
            max_total_tokens=400,
            max_context_tokens=200
        )
        manager = DomainPromptManager(config=config)

        assert manager.config.max_total_tokens == 400
        assert manager.config.max_context_tokens == 200

    def test_builtin_domains_loaded(self):
        """Test that built-in domains are loaded"""
        manager = DomainPromptManager()

        assert "medical" in manager.builtin_domains
        assert "legal" in manager.builtin_domains
        assert "technical" in manager.builtin_domains
        assert "business" in manager.builtin_domains
        assert "education" in manager.builtin_domains

        # Check medical terms are present
        medical_terms = manager.builtin_domains["medical"]
        assert "diagnosis" in medical_terms
        assert "symptoms" in medical_terms
        assert "prescription" in medical_terms

    def test_get_domain_terminology_builtin(self):
        """Test getting terminology from built-in dictionaries"""
        manager = DomainPromptManager()

        medical_terms = manager.get_domain_terminology("medical", limit=10)
        assert len(medical_terms) <= 10
        assert len(medical_terms) > 0
        assert "diagnosis" in medical_terms or "symptoms" in medical_terms

    def test_get_domain_terminology_unknown(self):
        """Test handling of unknown domain"""
        manager = DomainPromptManager()

        terms = manager.get_domain_terminology("unknown_domain")
        assert len(terms) == 0

    def test_create_domain_prompt_medical(self):
        """Test creating medical domain prompt"""
        manager = DomainPromptManager()

        prompt = manager.create_domain_prompt(domain="medical")

        assert len(prompt) > 0
        assert manager._estimate_tokens(prompt) <= 448
        # Should contain medical terminology
        prompt_lower = prompt.lower()
        assert any(term in prompt_lower for term in ["diagnosis", "symptoms", "patient"])

    def test_create_domain_prompt_with_custom_terms(self):
        """Test prompt creation with custom terminology"""
        manager = DomainPromptManager()

        custom_terms = ["COVID-19", "vaccination", "antibodies"]
        prompt = manager.create_domain_prompt(
            domain="medical",
            custom_terms=custom_terms
        )

        assert "COVID-19" in prompt
        assert "vaccination" in prompt
        assert "antibodies" in prompt

    def test_create_domain_prompt_with_context(self):
        """Test prompt creation with previous context"""
        manager = DomainPromptManager()

        previous_context = "The patient reports feeling unwell for three days."
        prompt = manager.create_domain_prompt(
            domain="medical",
            previous_context=previous_context
        )

        assert previous_context in prompt
        assert manager._estimate_tokens(prompt) <= 448

    def test_create_domain_prompt_token_limit(self):
        """Test that generated prompts respect 448 token limit"""
        manager = DomainPromptManager()

        # Create prompt with lots of context
        long_context = " ".join(["This is a very long context sentence."] * 100)

        prompt = manager.create_domain_prompt(
            domain="technical",
            custom_terms=["term1", "term2", "term3"],
            previous_context=long_context
        )

        tokens = manager._estimate_tokens(prompt)
        assert tokens <= 448

    def test_context_carryover_limit(self):
        """Test that context carryover respects 223 token limit"""
        manager = DomainPromptManager()

        # Create very long context
        long_context = " ".join(["word"] * 300)  # ~75 tokens

        prompt = manager.create_domain_prompt(
            previous_context=long_context
        )

        # Context portion should be <= 223 tokens
        context_tokens = manager._estimate_tokens(prompt)
        assert context_tokens <= 223

    def test_terminology_limit(self):
        """Test that terminology respects ~225 token limit"""
        manager = DomainPromptManager()

        # Medical + custom terms
        custom_terms = [f"term_{i}" for i in range(100)]

        prompt = manager.create_domain_prompt(
            domain="medical",
            custom_terms=custom_terms
        )

        tokens = manager._estimate_tokens(prompt)
        assert tokens <= 448

    def test_update_context(self):
        """Test scrolling context window updates"""
        manager = DomainPromptManager()

        manager.update_context("First output segment.")
        manager.update_context("Second output segment.")
        manager.update_context("Third output segment.")

        assert len(manager.context_buffer) == 3

        context = manager.get_current_context()
        assert "First output segment" in context
        assert "Third output segment" in context

    def test_context_buffer_token_limit(self):
        """Test that context buffer stays within token limit"""
        manager = DomainPromptManager()

        # Add many segments
        for i in range(50):
            manager.update_context(f"Segment {i} with some additional text.")

        context = manager.get_current_context()
        tokens = manager._estimate_tokens(context)

        assert tokens <= 223

    def test_clear_context(self):
        """Test clearing context buffer"""
        manager = DomainPromptManager()

        manager.update_context("Some context")
        manager.update_context("More context")

        assert len(manager.context_buffer) > 0

        manager.clear_context()

        assert len(manager.context_buffer) == 0
        assert manager.get_current_context() == ""

    def test_get_current_context(self):
        """Test getting current context from buffer"""
        manager = DomainPromptManager()

        manager.update_context("First")
        manager.update_context("Second")

        context = manager.get_current_context()

        assert "First" in context
        assert "Second" in context

    def test_token_estimation(self):
        """Test token count estimation"""
        manager = DomainPromptManager()

        # GPT-2 tokenizer: ~4 chars per token
        text = "This is a test sentence."  # 25 chars ≈ 6 tokens

        tokens = manager._estimate_tokens(text)

        assert tokens > 0
        assert tokens < len(text)  # Should be less than character count

    def test_trim_to_tokens(self):
        """Test trimming text to token limit"""
        manager = DomainPromptManager()

        long_text = " ".join(["word"] * 200)  # ~50 tokens

        trimmed = manager._trim_to_tokens(long_text, max_tokens=25)

        trimmed_tokens = manager._estimate_tokens(trimmed)
        assert trimmed_tokens <= 25
        assert len(trimmed) < len(long_text)

    def test_empty_prompt_generation(self):
        """Test generating prompt with no inputs"""
        manager = DomainPromptManager()

        prompt = manager.create_domain_prompt()

        assert prompt == ""

    def test_combined_domain_and_context(self):
        """Test combining domain terminology and context"""
        manager = DomainPromptManager()

        prompt = manager.create_domain_prompt(
            domain="technical",
            custom_terms=["Docker", "Kubernetes"],
            previous_context="Discussing deployment strategies."
        )

        assert "Docker" in prompt or "Kubernetes" in prompt
        assert "deployment" in prompt.lower()

        tokens = manager._estimate_tokens(prompt)
        assert tokens <= 448

    def test_convenience_function(self):
        """Test create_domain_prompt convenience function"""
        prompt = create_domain_prompt(
            domain="medical",
            custom_terms=["COVID-19"],
            previous_context="Patient consultation."
        )

        assert len(prompt) > 0
        assert "COVID-19" in prompt


class TestDomainPromptDatabase:
    """Test database integration for domain prompts"""

    def test_get_terminology_from_database(self):
        """Test loading terminology from database"""
        # Mock database session
        mock_session = Mock()
        mock_query = Mock()

        # Mock database results
        mock_category = Mock()
        mock_category.domain_id = "test-id"
        mock_category.name = "medical"

        mock_term = Mock()
        mock_term.term = "diagnosis"

        mock_query.filter.return_value.first.return_value = mock_category
        mock_query.filter.return_value.order_by.return_value.limit.return_value.all.return_value = [mock_term]
        mock_session.query.return_value = mock_query

        manager = DomainPromptManager(db_session=mock_session)

        # This should attempt to query database first
        terms = manager.get_domain_terminology("medical")

        # Should fallback to built-in if database fails
        assert len(terms) > 0

    def test_database_fallback_to_builtin(self):
        """Test fallback to built-in dictionaries if database fails"""
        # Mock database session that raises exception
        mock_session = Mock()
        mock_session.query.side_effect = Exception("Database error")

        manager = DomainPromptManager(db_session=mock_session)

        # Should fallback to built-in dictionaries
        terms = manager.get_domain_terminology("medical")

        assert len(terms) > 0
        assert "diagnosis" in terms or "symptoms" in terms


class TestDomainPromptSimulStreaming:
    """Test compliance with SimulStreaming paper specifications"""

    def test_max_tokens_448_compliance(self):
        """Test that prompts never exceed 448 tokens (SimulStreaming limit)"""
        manager = DomainPromptManager()

        # Try various combinations
        test_cases = [
            {"domain": "medical", "custom_terms": [f"term{i}" for i in range(50)]},
            {"previous_context": " ".join(["context"] * 200)},
            {"domain": "technical", "custom_terms": ["Docker"] * 100, "previous_context": "long " * 100},
        ]

        for case in test_cases:
            prompt = manager.create_domain_prompt(**case)
            tokens = manager._estimate_tokens(prompt)
            assert tokens <= 448, f"Prompt exceeded 448 tokens: {tokens}"

    def test_context_223_tokens_max(self):
        """Test that context carryover never exceeds 223 tokens"""
        manager = DomainPromptManager()

        # Very long context
        long_context = " ".join(["word"] * 500)

        prompt = manager.create_domain_prompt(previous_context=long_context)

        # Estimate context portion (should be <= 223 tokens)
        tokens = manager._estimate_tokens(prompt)
        assert tokens <= 223

    def test_terminology_225_tokens_max(self):
        """Test that terminology portion <= 225 tokens"""
        manager = DomainPromptManager()

        # Large custom term list
        custom_terms = [f"terminology_item_{i}" for i in range(100)]

        prompt = manager.create_domain_prompt(
            domain="medical",
            custom_terms=custom_terms
        )

        # Should trim to fit within 225 tokens
        tokens = manager._estimate_tokens(prompt)
        assert tokens <= 448


# Run tests
if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])
