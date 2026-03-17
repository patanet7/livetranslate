"""QA expert review — additional tests for coverage gaps.

Gaps identified during review of Phase 0/1/2 test suite:
1. SegmentStore: empty text, whitespace-only text, eviction preserves pending IDs
2. DirectionalContextStore: token eviction via directional wrapper, get_cross_direction case-insensitivity
3. LLMClient._build_messages: glossary injection, is_draft=True with context present (system prompt gate)
4. LLMClient._extract_translation: quote stripping, prefix stripping (response cleaning)
5. SegmentStore: on_final_received called twice for same segment_id (idempotency)
"""
import pytest
from translation.segment_record import SegmentPhase, SegmentRecord
from translation.segment_store import SegmentStore
from translation.context_store import DirectionalContextStore
from translation.llm_client import LLMClient
from translation.config import TranslationConfig
from livetranslate_common.models import TranslationContext


# ---------------------------------------------------------------------------
# SegmentStore: edge cases not covered by test_segment_store.py
# ---------------------------------------------------------------------------


class TestSegmentStoreEmptyAndWhitespace:
    def test_empty_text_on_final_received_is_final_true(self):
        """Empty-text is_final=True segment returns empty translate_text."""
        store = SegmentStore()
        rec, text = store.on_final_received(1, "", True, "en", "zh")
        assert rec.phase == SegmentPhase.FINAL_RECEIVED
        assert text == ""

    def test_whitespace_only_text_is_final_true(self):
        """Whitespace-only is_final=True with no pending returns empty string."""
        store = SegmentStore()
        rec, text = store.on_final_received(1, "   ", True, "en", "zh")
        assert text == ""

    def test_whitespace_only_accumulates_as_empty(self):
        """Whitespace-only non-final does not add visible content to pending."""
        store = SegmentStore()
        store.on_final_received(1, "   ", False, "en", "zh")
        # Pending may accumulate whitespace, but flush returns empty after strip
        text = store.flush_pending()
        assert text.strip() == ""

    def test_non_final_then_final_with_empty_final(self):
        """Accumulated text + empty final still flushes accumulated."""
        store = SegmentStore()
        store.on_final_received(1, "Hello", False, "en", "zh")
        rec, text = store.on_final_received(2, "", True, "en", "zh")
        # The accumulated "Hello" should be returned even though final text is empty
        assert "Hello" in text

    def test_on_final_received_twice_same_id_overwrites_phase(self):
        """Calling on_final_received twice with same segment_id updates source_text."""
        store = SegmentStore()
        store.on_final_received(1, "initial", False, "en", "zh")
        rec, text = store.on_final_received(1, "updated", True, "en", "zh")
        # On second call with is_final=True: rec.source_text is "updated"
        assert rec.source_text == "updated"
        assert rec.phase == SegmentPhase.FINAL_RECEIVED

    def test_evict_old_does_not_remove_pending_segment_ids_records(self):
        """evict_old protects records referenced by _pending_segment_ids.

        Pending records are mid-accumulation — evicting them would orphan
        the sentence buffer. evict_old skips any record whose segment_id
        appears in _pending_segment_ids.
        """
        store = SegmentStore()
        # Add 8 drafts with low IDs
        for i in range(8):
            store.on_draft_received(i, f"text {i}", "en", "zh")
        # Add 2 pending finals with low IDs that would normally be evicted
        store.on_final_received(1, "pending one", False, "en", "zh")
        store.on_final_received(2, "pending two", False, "en", "zh")
        assert store._pending_segment_ids == [1, 2]

        store.evict_old(keep_last=3)
        # Pending records 1 and 2 must survive eviction
        assert store.get(1) is not None, "pending segment 1 was evicted"
        assert store.get(2) is not None, "pending segment 2 was evicted"
        # Recent records also survive
        assert store.get(5) is not None
        assert store.get(6) is not None
        assert store.get(7) is not None
        # Old non-pending records evicted
        assert store.get(0) is None
        assert store.get(3) is None

    def test_draft_then_two_finals_correct_accumulation(self):
        """Draft for seg 1 is registered; finals for 2 and 3 accumulate normally."""
        store = SegmentStore()
        store.on_draft_received(1, "draft text", "en", "zh")
        store.on_final_received(2, "Part one", False, "en", "zh")
        store.on_final_received(3, "part two", False, "en", "zh")
        rec, text = store.on_final_received(4, "part three.", True, "en", "zh")
        assert text == "Part one part two part three."
        assert store._pending_sentence == ""


class TestSegmentStoreGetAfterReset:
    def test_get_after_reset_returns_none(self):
        store = SegmentStore()
        store.on_draft_received(1, "hello", "en", "zh")
        store.reset()
        assert store.get(1) is None

    def test_flush_pending_after_reset_returns_empty(self):
        store = SegmentStore()
        store.on_final_received(1, "Hello", False, "en", "zh")
        store.reset()
        assert store.flush_pending() == ""


# ---------------------------------------------------------------------------
# DirectionalContextStore: gaps not covered by test_directional_context_store.py
# ---------------------------------------------------------------------------


class TestDirectionalContextStoreTokenEviction:
    def test_token_eviction_through_directional_wrapper(self):
        """DirectionalContextStore max_tokens evicts oldest entries.

        This exercises the token-level eviction path in RollingContextWindow
        through the DirectionalContextStore wrapper — not just count eviction.
        """
        # max_tokens=20: each long entry should trigger eviction
        store = DirectionalContextStore(max_entries=100, max_tokens=20)
        store.add("en", "zh", "short", "短")  # ~6 tokens
        store.add("en", "zh", "medium entry here", "中等文本内容")  # ~10 tokens
        store.add("en", "zh", "another long entry text", "又一个长条目文字")  # ~10 tokens

        ctx = store.get("en", "zh")
        from translation.context import RollingContextWindow
        w = RollingContextWindow(max_entries=100, max_tokens=20)
        total_tokens = sum(w._estimate_tokens(c.text + c.translation) for c in ctx)
        assert total_tokens <= 20

    def test_token_eviction_does_not_affect_other_direction(self):
        """Token eviction in one direction does not touch the other direction."""
        store = DirectionalContextStore(max_entries=100, max_tokens=20)
        # Fill zh->en to trigger token eviction
        store.add("zh", "en", "很长的中文句子需要更多的词符", "A longer English sentence needing tokens")
        store.add("zh", "en", "第二个中文句子也很长", "Second English sentence is also long enough")
        store.add("zh", "en", "第三个很长", "Third also long entry here today")

        # en->zh stays untouched
        store.add("en", "zh", "hello", "你好")
        en_zh = store.get("en", "zh")
        assert len(en_zh) == 1
        assert en_zh[0].text == "hello"


class TestDirectionalContextStoreCrossCaseInsensitive:
    def test_get_cross_direction_case_insensitive_source(self):
        """get_cross_direction should work regardless of key case."""
        store = DirectionalContextStore()
        store.add("EN", "ZH", "Hello", "你好")
        store.add("EN", "ZH", "Goodbye", "再见")

        # Cross-direction for zh->en should find entries from en->zh (stored as EN->ZH)
        cross = store.get_cross_direction("zh", "en")
        assert len(cross) == 2

    def test_get_cross_direction_returns_last_two_after_many_adds(self):
        """Cross direction returns exactly 2 entries (last 2) regardless of total count."""
        store = DirectionalContextStore()
        entries = [
            ("Hello", "你好"), ("How are you?", "你好吗？"),
            ("I am fine", "我很好"), ("Thank you", "谢谢"),
            ("Goodbye", "再见"),
        ]
        for src, tgt in entries:
            store.add("en", "zh", src, tgt)

        cross = store.get_cross_direction("zh", "en")
        assert len(cross) == 2
        assert cross[0].text == "Thank you"
        assert cross[1].text == "Goodbye"


# ---------------------------------------------------------------------------
# LLMClient._build_messages: missing cases
# ---------------------------------------------------------------------------


class TestBuildMessagesGlossary:
    """Glossary injection in _build_messages — not covered in existing tests."""

    def _make_client(self) -> LLMClient:
        config = TranslationConfig(
            base_url="http://localhost:11434/v1",
            model="test",
        )
        return LLMClient(config)

    def test_glossary_terms_appear_in_user_message(self):
        """Glossary terms are included in the prompt in 'Terms: k=v' format."""
        client = self._make_client()
        messages = client._build_messages(
            text="The engineer mentioned the SDK",
            source_language="en",
            target_language="zh",
            context=[],
            glossary_terms={"SDK": "软件开发工具包", "engineer": "工程师"},
        )
        user_msg = messages[1]["content"]
        assert "Terms:" in user_msg
        assert "SDK=软件开发工具包" in user_msg or "engineer=工程师" in user_msg

    def test_glossary_newlines_are_sanitized(self):
        """Newlines in glossary keys/values are replaced with spaces (injection guard)."""
        client = self._make_client()
        messages = client._build_messages(
            text="test",
            source_language="en",
            target_language="zh",
            context=[],
            glossary_terms={"key\ninjection": "val\ninjection"},
        )
        user_msg = messages[1]["content"]
        assert "key\ninjection" not in user_msg
        assert "val\ninjection" not in user_msg

    def test_glossary_truncated_to_50_terms(self):
        """Oversized glossary is truncated to first 50 terms (injection guard)."""
        client = self._make_client()
        huge_glossary = {f"term{i}": f"val{i}" for i in range(60)}
        messages = client._build_messages(
            text="test",
            source_language="en",
            target_language="zh",
            context=[],
            glossary_terms=huge_glossary,
        )
        user_msg = messages[1]["content"]
        # Should not have term50 through term59 (truncated at 50)
        term_count = user_msg.count("term")
        assert term_count <= 50

    def test_no_glossary_no_terms_line(self):
        """Without glossary, 'Terms:' line is absent."""
        client = self._make_client()
        messages = client._build_messages(
            text="hello",
            source_language="en",
            target_language="zh",
            context=[],
        )
        user_msg = messages[1]["content"]
        assert "Terms:" not in user_msg


class TestBuildMessagesSystemPromptBranching:
    """Verify system prompt branches on has_context (bool(context)).

    The implementation gates 'Never repeat context' on bool(context).
    Tests confirm this so a refactor doesn't accidentally change it.
    """

    def _make_client(self) -> LLMClient:
        config = TranslationConfig(base_url="http://localhost:11434/v1", model="test")
        return LLMClient(config)

    def test_with_context_gets_never_repeat(self):
        """With context present, system prompt includes 'Never repeat context' guard."""
        client = self._make_client()
        context = [TranslationContext(text="Hello", translation="你好")]
        messages = client._build_messages(
            text="Goodbye",
            source_language="en",
            target_language="zh",
            context=context,
        )
        system_msg = messages[0]["content"]
        assert "Never repeat context" in system_msg

    def test_without_context_no_repeat_guard(self):
        """Without context, system prompt omits the 'Never repeat context' guard."""
        client = self._make_client()
        messages = client._build_messages(
            text="Hello",
            source_language="en",
            target_language="zh",
            context=[],
        )
        system_msg = messages[0]["content"]
        assert "Never repeat context" not in system_msg

    def test_nothink_suffix_always_appended(self):
        """The /nothink suffix is always appended for Qwen3 compatibility."""
        client = self._make_client()
        # With context
        messages_ctx = client._build_messages(
            text="hello",
            source_language="en",
            target_language="zh",
            context=[TranslationContext(text="hi", translation="嗨")],
        )
        assert "/nothink" in messages_ctx[1]["content"]

        # Without context
        messages_no_ctx = client._build_messages(
            text="hello",
            source_language="en",
            target_language="zh",
            context=[],
        )
        assert "/nothink" in messages_no_ctx[1]["content"]


class TestExtractTranslation:
    """Tests for LLMClient._extract_translation response cleaning."""

    def _make_client(self) -> LLMClient:
        config = TranslationConfig(base_url="http://localhost:11434/v1", model="test")
        return LLMClient(config)

    def test_plain_translation_returned_as_is(self):
        client = self._make_client()
        assert client._extract_translation("你好世界") == "你好世界"

    def test_think_block_stripped(self):
        client = self._make_client()
        result = client._extract_translation("<think>Let me think</think>你好")
        assert result == "你好"

    def test_unclosed_think_block_discarded(self):
        client = self._make_client()
        result = client._extract_translation("<think>reasoning that never closes")
        assert result == ""

    def test_double_quotes_stripped(self):
        client = self._make_client()
        assert client._extract_translation('"你好世界"') == "你好世界"

    def test_single_quotes_stripped(self):
        client = self._make_client()
        assert client._extract_translation("'你好世界'") == "你好世界"

    def test_curly_quotes_stripped(self):
        client = self._make_client()
        # Unicode left/right double quotation marks
        assert client._extract_translation("\u201c你好世界\u201d") == "你好世界"

    def test_translation_prefix_stripped(self):
        client = self._make_client()
        assert client._extract_translation("Translation: 你好世界") == "你好世界"

    def test_output_prefix_stripped(self):
        client = self._make_client()
        assert client._extract_translation("Output: 你好") == "你好"

    def test_chinese_translation_prefix_stripped(self):
        client = self._make_client()
        assert client._extract_translation("翻译: 你好") == "你好"

    def test_whitespace_stripped(self):
        client = self._make_client()
        assert client._extract_translation("  你好  ") == "你好"

    def test_empty_string_returned_as_empty(self):
        client = self._make_client()
        assert client._extract_translation("") == ""

    def test_think_block_followed_by_content(self):
        client = self._make_client()
        result = client._extract_translation("<think>reasoning</think>\n\n你好世界")
        assert result == "你好世界"


# ---------------------------------------------------------------------------
# SegmentStore: is_final_translated idempotency and phase guard
# ---------------------------------------------------------------------------


class TestSegmentStoreFinalTranslatedGuard:
    def test_is_final_translated_false_before_translation(self):
        """is_final_translated is False right after final is received."""
        store = SegmentStore()
        store.on_final_received(1, "Hello.", True, "en", "zh")
        assert not store.is_final_translated(1)

    def test_on_final_translated_twice_does_not_change_phase(self):
        """Calling on_final_translated twice with different translations keeps last."""
        store = SegmentStore()
        store.on_final_received(1, "Hello.", True, "en", "zh")
        store.on_final_translated(1, "你好。")
        store.on_final_translated(1, "你好！")  # second call
        rec = store.get(1)
        assert rec is not None
        assert rec.final_translation == "你好！"
        assert rec.phase == SegmentPhase.FINAL_TRANSLATED

    def test_is_final_translated_after_reset(self):
        """is_final_translated returns False after reset, even for previously translated."""
        store = SegmentStore()
        store.on_final_received(1, "Hello.", True, "en", "zh")
        store.on_final_translated(1, "你好。")
        assert store.is_final_translated(1)
        store.reset()
        assert not store.is_final_translated(1)


# ---------------------------------------------------------------------------
# RollingContextWindow: token estimation
# ---------------------------------------------------------------------------


class TestRollingContextWindowTokenEstimation:
    """Token estimation logic — CJK vs Latin split."""

    def test_cjk_chars_count_as_one_token_each(self):
        from translation.context import RollingContextWindow
        # Pure CJK: 5 chars → 5 tokens
        assert RollingContextWindow._estimate_tokens("你好世界啊") == 5

    def test_latin_chars_count_as_four_per_token(self):
        from translation.context import RollingContextWindow
        # 4 latin chars → 1 token, 8 latin → 2 tokens
        assert RollingContextWindow._estimate_tokens("abcd") == 1
        assert RollingContextWindow._estimate_tokens("abcdefgh") == 2

    def test_mixed_text_combines_cjk_and_latin(self):
        from translation.context import RollingContextWindow
        # 2 CJK + 4 Latin = 2 + 1 = 3 tokens
        assert RollingContextWindow._estimate_tokens("你好abcd") == 3

    def test_minimum_one_token(self):
        from translation.context import RollingContextWindow
        # Even empty string should return at least 1
        assert RollingContextWindow._estimate_tokens("") >= 1
        assert RollingContextWindow._estimate_tokens("a") >= 1
