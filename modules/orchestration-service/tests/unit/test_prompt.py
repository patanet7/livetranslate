"""Prompt-building unit tests.

`translation.prompt.build_messages` is the pure function the merged LLMClient
is driven by. It produces a 2-element [system, user] messages list with
several optional inclusions (context, glossary, cross-context). These tests
pin the prompt structure so we notice if anyone accidentally drifts the
format the model was tuned for.
"""

from __future__ import annotations

import pytest

from livetranslate_common.models import TranslationContext
from translation.prompt import build_messages


class TestSystemPrompt:
    def test_no_context_short_prompt(self) -> None:
        msgs = build_messages("hi", "zh", "en")
        sys_msg = msgs[0]["content"]
        assert "Translate Chinese speech to English" in sys_msg
        assert "Never repeat context" not in sys_msg

    def test_with_context_includes_repeat_guard(self) -> None:
        ctx = [TranslationContext(text="你好", translation="Hello")]
        msgs = build_messages("hi", "zh", "en", context=ctx)
        sys_msg = msgs[0]["content"]
        assert "Never repeat context" in sys_msg

    def test_extra_instruction_en_to_zh(self) -> None:
        msgs = build_messages("hi", "en", "zh")
        assert "simplified characters" in msgs[0]["content"]

    def test_unknown_language_passthrough(self) -> None:
        msgs = build_messages("hi", "xyz", "abc")
        assert "Translate xyz speech to abc" in msgs[0]["content"]


class TestUserMessage:
    def test_no_context_uses_translate_prefix(self) -> None:
        msgs = build_messages("hello world", "en", "zh")
        assert msgs[1]["content"] == "Translate: hello world"

    def test_with_context_uses_new_marker(self) -> None:
        ctx = [TranslationContext(text="prior", translation="anterior")]
        msgs = build_messages("hello", "en", "es", context=ctx)
        user = msgs[1]["content"]
        assert "[New:]" in user
        assert user.endswith("hello")

    def test_context_bilingual_pairs(self) -> None:
        ctx = [TranslationContext(text="你好", translation="Hello")]
        msgs = build_messages("再见", "zh", "en", context=ctx)
        user = msgs[1]["content"]
        assert "[Chinese] 你好" in user
        assert "[English] Hello" in user

    def test_cross_context_labels_swapped(self) -> None:
        """Cross-direction context has the opposite-direction labels."""
        # cross_context represents prior translations from the OPPOSITE direction
        # so labels swap: ctx.text in target_language, ctx.translation in source.
        cross = [TranslationContext(text="Reply text", translation="回复")]
        msgs = build_messages("新话", "zh", "en", cross_context=cross)
        user = msgs[1]["content"]
        assert "Recent context (other speaker)" in user
        # Cross context labels: tgt_name (English) first, then src_name (Chinese)
        assert "[English] Reply text" in user
        assert "[Chinese] 回复" in user

    def test_glossary_terms_injected(self) -> None:
        msgs = build_messages(
            "hello", "en", "zh", glossary_terms={"hello": "你好"}
        )
        user = msgs[1]["content"]
        assert "Terms: hello=你好" in user

    def test_glossary_terms_capped_at_50(self) -> None:
        big = {f"k{i}": f"v{i}" for i in range(100)}
        msgs = build_messages("hi", "en", "zh", glossary_terms=big)
        user = msgs[1]["content"]
        # Truncation: only 50 entries make it into the prompt.
        assert user.count("=") == 50

    def test_glossary_term_lengths_capped_at_100(self) -> None:
        big_key = "k" * 500
        big_val = "v" * 500
        msgs = build_messages("hi", "en", "zh", glossary_terms={big_key: big_val})
        user = msgs[1]["content"]
        # Each side capped at 100 chars (the cap is keys[:100] / vals[:100])
        assert "k" * 100 in user
        assert "k" * 101 not in user

    def test_glossary_newlines_sanitized(self) -> None:
        """Newlines in glossary entries are replaced with spaces to prevent
        prompt-structure injection (e.g. user inserting `\\n\\nIgnore...`)."""
        msgs = build_messages(
            "hi", "en", "zh",
            glossary_terms={"foo\nbar": "baz\nquux"},
        )
        user = msgs[1]["content"]
        assert "foo bar=baz quux" in user
        assert "foo\nbar" not in user


def test_returns_two_message_list() -> None:
    msgs = build_messages("x", "en", "zh")
    assert len(msgs) == 2
    assert msgs[0]["role"] == "system"
    assert msgs[1]["role"] == "user"
