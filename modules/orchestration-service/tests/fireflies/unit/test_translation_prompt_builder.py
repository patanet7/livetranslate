#!/usr/bin/env python3
"""
Translation Prompt Builder Tests

Verifies speaker identity injection, context window formatting,
glossary injection, and template selection logic.
"""

import os
import sys
from pathlib import Path

import pytest

os.environ["SKIP_MAIN_FASTAPI_IMPORT"] = "1"

orchestration_root = Path(__file__).parent.parent.parent.parent
src_path = orchestration_root / "src"
sys.path.insert(0, str(orchestration_root))
sys.path.insert(0, str(src_path))

import importlib.util

_tpb_spec = importlib.util.spec_from_file_location(
    "translation_prompt_builder",
    src_path / "services" / "translation_prompt_builder.py",
)
_tpb_module = importlib.util.module_from_spec(_tpb_spec)
_tpb_spec.loader.exec_module(_tpb_module)

TranslationPromptBuilder = _tpb_module.TranslationPromptBuilder
PromptContext = _tpb_module.PromptContext
BuiltPrompt = _tpb_module.BuiltPrompt
create_prompt_builder = _tpb_module.create_prompt_builder


class TestSpeakerNameInPrompt:
    """Speaker identity should appear in translation prompts for per-speaker consistency."""

    def test_prompt_includes_speaker_name(self):
        """Translation prompt should include speaker identity when provided."""
        builder = TranslationPromptBuilder()
        result = builder.build(
            PromptContext(
                current_sentence="We need to fix the diarization pipeline",
                target_language="zh",
                speaker_name="Thomas Patane",
                previous_sentences=["The audio is working now"],
            )
        )
        assert "Thomas Patane" in result.prompt
        assert "Current Speaker: Thomas Patane" in result.prompt

    def test_prompt_without_speaker_name(self):
        """Prompt should not contain 'Current Speaker' when no speaker provided."""
        builder = TranslationPromptBuilder()
        result = builder.build(
            PromptContext(
                current_sentence="Hello world",
                target_language="es",
                previous_sentences=["Good morning"],
            )
        )
        assert "Current Speaker" not in result.prompt

    def test_speaker_name_uses_full_template(self):
        """Speaker name only appears when full template is used (has context/glossary)."""
        builder = TranslationPromptBuilder()
        # No context or glossary → simple template → no speaker section
        result = builder.build(
            PromptContext(
                current_sentence="Hello",
                target_language="es",
                speaker_name="Alice",
            )
        )
        # Simple template doesn't have speaker_section placeholder
        assert result.template_used == "simple"
        assert "Current Speaker" not in result.prompt


class TestTemplateSelection:
    """Template selection based on available context and glossary."""

    def test_full_template_with_context(self):
        """Uses full template when previous sentences are available."""
        builder = TranslationPromptBuilder()
        result = builder.build(
            PromptContext(
                current_sentence="Continue",
                target_language="fr",
                previous_sentences=["First sentence"],
            )
        )
        assert result.template_used == "full"
        assert result.has_context is True

    def test_full_template_with_glossary(self):
        """Uses full template when glossary terms are available."""
        builder = TranslationPromptBuilder()
        result = builder.build(
            PromptContext(
                current_sentence="Check the API endpoint",
                target_language="es",
                glossary_terms={"API": "API", "endpoint": "punto de acceso"},
            )
        )
        assert result.template_used == "full"
        assert result.has_glossary is True
        assert result.glossary_term_count == 2

    def test_simple_template_when_no_context(self):
        """Uses simple template when no context or glossary."""
        builder = TranslationPromptBuilder()
        result = builder.build(
            PromptContext(
                current_sentence="Hello",
                target_language="de",
            )
        )
        assert result.template_used == "simple"
        assert result.has_context is False
        assert result.has_glossary is False

    def test_minimal_template_when_enabled(self):
        """Uses minimal template when use_minimal_when_empty is True."""
        builder = TranslationPromptBuilder(use_minimal_when_empty=True)
        result = builder.build(
            PromptContext(
                current_sentence="Hi",
                target_language="ja",
            )
        )
        assert result.template_used == "minimal"


class TestContextWindow:
    """Context window formatting in the prompt."""

    def test_context_sentences_numbered(self):
        """Previous sentences appear numbered in the prompt."""
        builder = TranslationPromptBuilder()
        result = builder.build(
            PromptContext(
                current_sentence="What do you think?",
                target_language="ko",
                previous_sentences=["First point", "Second point", "Third point"],
            )
        )
        assert "1. First point" in result.prompt
        assert "2. Second point" in result.prompt
        assert "3. Third point" in result.prompt
        assert result.context_sentence_count == 3

    def test_no_context_placeholder(self):
        """Shows placeholder when no context (but still using full template via glossary)."""
        builder = TranslationPromptBuilder()
        result = builder.build(
            PromptContext(
                current_sentence="Translate this",
                target_language="fr",
                glossary_terms={"test": "essai"},
            )
        )
        assert "(no previous context)" in result.prompt


class TestGlossaryFormatting:
    """Glossary terms injection in the prompt."""

    def test_glossary_terms_formatted(self):
        """Glossary terms appear as source = target pairs."""
        builder = TranslationPromptBuilder()
        result = builder.build(
            PromptContext(
                current_sentence="Check the API",
                target_language="es",
                glossary_terms={"API": "API", "endpoint": "punto de acceso"},
            )
        )
        assert "- API = API" in result.prompt
        assert "- endpoint = punto de acceso" in result.prompt
        assert "Glossary (use these exact translations)" in result.prompt


class TestFactoryFunction:
    """create_prompt_builder factory with configuration."""

    def test_default_factory(self):
        """Factory creates builder with default settings."""
        builder = create_prompt_builder()
        assert isinstance(builder, TranslationPromptBuilder)

    def test_factory_with_minimal_enabled(self):
        """Factory respects use_minimal_when_empty config."""
        builder = create_prompt_builder({"use_minimal_when_empty": True})
        result = builder.build(
            PromptContext(current_sentence="Hi", target_language="es")
        )
        assert result.template_used == "minimal"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
