"""Tests for LLM client cross-direction context in prompt building."""
import inspect

from livetranslate_common.models import TranslationContext
from translation.config import TranslationConfig
from translation.llm_client import LLMClient


class TestLLMClientCrossContext:
    def _build(self, **kwargs):
        config = TranslationConfig()
        client = LLMClient(config)
        return client._build_messages(**kwargs)

    def test_cross_context_absent_no_header(self):
        """When cross_context is None, no cross-direction header appears."""
        messages = self._build(
            text="你好",
            source_language="zh",
            target_language="en",
            context=[],
        )
        user_content = messages[1]["content"]
        assert "[Recent context (other speaker):]" not in user_content

    def test_cross_context_empty_list_no_header(self):
        """When cross_context is empty list, no cross-direction header appears."""
        messages = self._build(
            text="你好",
            source_language="zh",
            target_language="en",
            context=[],
            cross_context=[],
        )
        user_content = messages[1]["content"]
        assert "[Recent context (other speaker):]" not in user_content

    def test_cross_context_renders_in_prompt(self):
        """Cross-direction context entries appear in the user message."""
        cross = [
            TranslationContext(text="Hello", translation="你好"),
            TranslationContext(text="How are you?", translation="你好吗？"),
        ]
        messages = self._build(
            text="我很好",
            source_language="zh",
            target_language="en",
            context=[TranslationContext(text="你好", translation="Hello")],
            cross_context=cross,
        )
        user_content = messages[1]["content"]

        assert "[Recent context (other speaker):]" in user_content
        # Cross-context comes from en→zh, so ctx.text is English, ctx.translation is Chinese.
        # Labels are swapped relative to the current direction (zh→en):
        # tgt_name (English) labels ctx.text, src_name (Chinese) labels ctx.translation.
        assert "[English] Hello" in user_content
        assert "[Chinese] 你好" in user_content
        assert "[English] How are you?" in user_content
        assert "[Chinese] 你好吗？" in user_content

    def test_cross_context_appears_after_prior_context(self):
        """Cross-direction context appears between [Prior:] and [New:]."""
        context = [TranslationContext(text="你好", translation="Hello")]
        cross = [TranslationContext(text="Thank you", translation="谢谢")]

        messages = self._build(
            text="不客气",
            source_language="zh",
            target_language="en",
            context=context,
            cross_context=cross,
        )
        user_content = messages[1]["content"]

        prior_idx = user_content.index("[Prior:]")
        cross_idx = user_content.index("[Recent context (other speaker):]")
        new_idx = user_content.index("[New:]")
        assert prior_idx < cross_idx < new_idx

    def test_cross_context_newlines_sanitized(self):
        """Newlines in cross-context entries are replaced with spaces."""
        cross = [TranslationContext(text="Hello\nworld", translation="你好\n世界")]
        messages = self._build(
            text="测试",
            source_language="zh",
            target_language="en",
            context=[],
            cross_context=cross,
        )
        user_content = messages[1]["content"]
        assert "Hello world" in user_content
        assert "你好 世界" in user_content

    def test_translate_stream_accepts_cross_context(self):
        """translate_stream() has cross_context parameter wired to _build_messages."""
        sig = inspect.signature(LLMClient.translate_stream)
        assert "cross_context" in sig.parameters, (
            "translate_stream() must accept cross_context parameter"
        )

    def test_translate_stream_passes_cross_context_to_build_messages(self):
        """Verify translate_stream passes cross_context to _build_messages.

        We inspect the source of translate_stream to confirm cross_context
        is forwarded, since calling it would require a live LLM server.
        """
        source = inspect.getsource(LLMClient.translate_stream)
        assert "cross_context=cross_context" in source, (
            "translate_stream must pass cross_context to _build_messages"
        )
