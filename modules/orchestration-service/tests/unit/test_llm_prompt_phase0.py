"""Tests for Phase 0 LLM prompt redesign.

Verifies:
- Bilingual context pairs format ([lang] source + [lang] translation)
- Context-present/absent system prompt variants
- User message format differs based on context presence
"""

import pytest
from livetranslate_common.models import TranslationContext

from translation.llm_client import LLMClient


@pytest.fixture
def client(llm_config):
    return LLMClient(llm_config)


class TestBilingualContext:
    """Task 0.4: Context shows [lang] source + [lang] translation pairs."""

    def test_context_includes_source_and_translation(self, client):
        context = [
            TranslationContext(text="之前的话", translation="Previous words"),
        ]
        messages = client._build_messages(
            text="这是新的",
            source_language="zh",
            target_language="en",
            context=context,
        )
        user_msg = messages[1]["content"]
        # Both source and translation must appear with language labels
        assert "[Chinese] 之前的话" in user_msg
        assert "[English] Previous words" in user_msg

    def test_context_uses_prior_label(self, client):
        context = [
            TranslationContext(text="Hello", translation="你好"),
        ]
        messages = client._build_messages(
            text="How are you?",
            source_language="en",
            target_language="zh",
            context=context,
        )
        user_msg = messages[1]["content"]
        assert "[Prior:]" in user_msg

    def test_context_newlines_stripped(self, client):
        context = [
            TranslationContext(text="line\none", translation="line\ntwo"),
        ]
        messages = client._build_messages(
            text="test",
            source_language="en",
            target_language="zh",
            context=context,
        )
        user_msg = messages[1]["content"]
        # Newlines within context entries should be replaced with spaces
        assert "line one" in user_msg
        assert "line two" in user_msg

    def test_multiple_context_pairs(self, client):
        context = [
            TranslationContext(text="第一句", translation="First sentence"),
            TranslationContext(text="第二句", translation="Second sentence"),
        ]
        messages = client._build_messages(
            text="第三句",
            source_language="zh",
            target_language="en",
            context=context,
        )
        user_msg = messages[1]["content"]
        assert "[Chinese] 第一句" in user_msg
        assert "[English] First sentence" in user_msg
        assert "[Chinese] 第二句" in user_msg
        assert "[English] Second sentence" in user_msg


class TestDraftFinalSystemPrompt:
    """Task 0.5: System prompt variant selected by context presence."""

    def test_final_with_context_has_never_repeat(self, client):
        """With context present, system prompt includes 'Never repeat context' guard."""
        context = [
            TranslationContext(text="Hello", translation="你好"),
        ]
        messages = client._build_messages(
            text="Goodbye",
            source_language="en",
            target_language="zh",
            context=context,
        )
        system_msg = messages[0]["content"]
        assert "Never repeat context" in system_msg

    def test_draft_no_context_shorter_prompt(self, client):
        """Without context (draft path), system prompt omits the repeat guard."""
        messages = client._build_messages(
            text="Hello",
            source_language="en",
            target_language="zh",
            context=[],
        )
        system_msg = messages[0]["content"]
        # No context → shorter prompt, no "Never repeat context"
        assert "Never repeat context" not in system_msg
        # Should still have core translation instruction
        assert "English" in system_msg
        assert "Chinese" in system_msg

    def test_final_no_context_no_repeat_guard(self, client):
        """Without context, system prompt does not include 'Never repeat context'."""
        messages = client._build_messages(
            text="Hello",
            source_language="en",
            target_language="zh",
            context=[],
        )
        system_msg = messages[0]["content"]
        assert "Never repeat context" not in system_msg

    def test_system_prompt_uses_speech_wording(self, client):
        """System prompt says 'speech' not 'spoken'."""
        messages = client._build_messages(
            text="Hello",
            source_language="en",
            target_language="zh",
            context=[],
        )
        system_msg = messages[0]["content"]
        assert "speech" in system_msg.lower()


class TestDraftFinalUserMessage:
    """Task 0.6: User message format differs based on context presence."""

    def test_final_with_context_uses_new_label(self, client):
        """With context, user message uses [New:] label before the text."""
        context = [
            TranslationContext(text="Hello", translation="你好"),
        ]
        messages = client._build_messages(
            text="Goodbye",
            source_language="en",
            target_language="zh",
            context=context,
        )
        user_msg = messages[1]["content"]
        assert "[New:]" in user_msg
        assert "Goodbye" in user_msg

    def test_draft_no_context_uses_translate_prefix(self, client):
        """Without context, user message uses compact 'Translate: ...' format."""
        messages = client._build_messages(
            text="Hello world",
            source_language="en",
            target_language="zh",
            context=[],
        )
        user_msg = messages[1]["content"]
        assert "Translate: Hello world" in user_msg

    def test_no_translate_this_label(self, client):
        """The old [Translate this:] label should no longer appear."""
        messages = client._build_messages(
            text="Hello",
            source_language="en",
            target_language="zh",
            context=[],
        )
        user_msg = messages[1]["content"]
        assert "[Translate this:]" not in user_msg

    def test_old_context_format_gone(self, client):
        """The old '[Context -- previous conversation, do NOT translate:]' is gone."""
        context = [
            TranslationContext(text="Hello", translation="你好"),
        ]
        messages = client._build_messages(
            text="Goodbye",
            source_language="en",
            target_language="zh",
            context=context,
        )
        user_msg = messages[1]["content"]
        assert "do NOT translate" not in user_msg
