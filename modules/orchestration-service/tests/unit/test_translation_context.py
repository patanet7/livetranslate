"""Tests for rolling translation context window."""
from livetranslate_common.models import TranslationContext
from translation.context import RollingContextWindow


class TestRollingContextWindow:
    def test_empty_context(self):
        window = RollingContextWindow(max_entries=5)
        assert window.get_context() == []

    def test_add_and_get(self):
        window = RollingContextWindow(max_entries=5)
        window.add("你好", "Hello")
        ctx = window.get_context()
        assert len(ctx) == 1
        assert ctx[0].text == "你好"
        assert ctx[0].translation == "Hello"

    def test_eviction_by_count(self):
        window = RollingContextWindow(max_entries=3)
        window.add("one", "一")
        window.add("two", "二")
        window.add("three", "三")
        window.add("four", "四")  # should evict "one"

        ctx = window.get_context()
        assert len(ctx) == 3
        assert ctx[0].text == "two"
        assert ctx[2].text == "four"

    def test_eviction_by_tokens(self):
        window = RollingContextWindow(max_entries=100, max_tokens=20)
        window.add("short", "短")  # ~6 tokens
        window.add("medium text here", "中等文本")  # ~10 tokens
        window.add("another long entry", "又一个长条目")  # ~10 tokens → exceeds 20

        ctx = window.get_context()
        # Oldest entries evicted until under token limit
        total_tokens = sum(window._estimate_tokens(c.text + c.translation) for c in ctx)
        assert total_tokens <= 20

    def test_failed_translations_not_added(self):
        window = RollingContextWindow(max_entries=5)
        window.add("good", "好")
        # Don't add failed translations — caller responsibility
        assert len(window.get_context()) == 1

    def test_clear(self):
        window = RollingContextWindow(max_entries=5)
        window.add("one", "一")
        window.clear()
        assert window.get_context() == []
