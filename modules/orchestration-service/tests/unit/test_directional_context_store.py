"""Tests for DirectionalContextStore — Phase 1 of context redesign.

Verifies:
- Per-(source_lang, target_lang) independent windows
- Case-insensitive language keys
- Cross-direction retrieval for interpreter mode
- clear_direction only clears one direction
- clear_all clears everything
"""
from translation.context_store import DirectionalContextStore


class TestDirectionalContextStore:
    def test_empty_store_returns_empty_list(self):
        store = DirectionalContextStore()
        assert store.get("zh", "en") == []

    def test_add_and_get_single_direction(self):
        store = DirectionalContextStore()
        store.add("zh", "en", "你好", "Hello")
        ctx = store.get("zh", "en")
        assert len(ctx) == 1
        assert ctx[0].text == "你好"
        assert ctx[0].translation == "Hello"

    def test_independent_directions(self):
        """Two directions (zh->en and en->zh) maintain separate windows."""
        store = DirectionalContextStore()
        store.add("zh", "en", "你好", "Hello")
        store.add("en", "zh", "Goodbye", "再见")

        zh_en = store.get("zh", "en")
        en_zh = store.get("en", "zh")

        assert len(zh_en) == 1
        assert zh_en[0].text == "你好"
        assert len(en_zh) == 1
        assert en_zh[0].text == "Goodbye"

    def test_case_insensitive_keys(self):
        store = DirectionalContextStore()
        store.add("ZH", "EN", "你好", "Hello")
        ctx = store.get("zh", "en")
        assert len(ctx) == 1
        assert ctx[0].text == "你好"

    def test_case_insensitive_mixed(self):
        store = DirectionalContextStore()
        store.add("Zh", "En", "你好", "Hello")
        store.add("zh", "en", "谢谢", "Thank you")
        ctx = store.get("ZH", "EN")
        assert len(ctx) == 2

    def test_cross_direction_retrieval(self):
        """get_cross_direction returns entries from the opposite direction."""
        store = DirectionalContextStore()
        store.add("en", "zh", "Hello", "你好")
        store.add("en", "zh", "How are you?", "你好吗？")
        store.add("en", "zh", "I am fine", "我很好")

        # Cross-direction for zh->en should return last 2 entries of en->zh
        cross = store.get_cross_direction("zh", "en")
        assert len(cross) == 2
        assert cross[0].text == "How are you?"
        assert cross[1].text == "I am fine"

    def test_cross_direction_empty_when_no_opposite(self):
        store = DirectionalContextStore()
        store.add("zh", "en", "你好", "Hello")
        cross = store.get_cross_direction("zh", "en")
        assert cross == []

    def test_cross_direction_returns_at_most_two(self):
        store = DirectionalContextStore()
        for i in range(5):
            store.add("en", "zh", f"text {i}", f"翻译 {i}")
        cross = store.get_cross_direction("zh", "en")
        assert len(cross) == 2

    def test_clear_direction_only_clears_one(self):
        store = DirectionalContextStore()
        store.add("zh", "en", "你好", "Hello")
        store.add("en", "zh", "Goodbye", "再见")

        store.clear_direction("zh", "en")

        assert store.get("zh", "en") == []
        assert len(store.get("en", "zh")) == 1

    def test_clear_direction_nonexistent_is_noop(self):
        store = DirectionalContextStore()
        store.clear_direction("zh", "en")  # should not raise

    def test_clear_all(self):
        store = DirectionalContextStore()
        store.add("zh", "en", "你好", "Hello")
        store.add("en", "zh", "Goodbye", "再见")
        store.add("ja", "en", "こんにちは", "Hello")

        store.clear_all()

        assert store.get("zh", "en") == []
        assert store.get("en", "zh") == []
        assert store.get("ja", "en") == []

    def test_eviction_by_count(self):
        store = DirectionalContextStore(max_entries=2, max_tokens=10000)
        store.add("zh", "en", "one", "一")
        store.add("zh", "en", "two", "二")
        store.add("zh", "en", "three", "三")

        ctx = store.get("zh", "en")
        assert len(ctx) == 2
        assert ctx[0].text == "two"
        assert ctx[1].text == "three"

    def test_direction_flip_preserves_both_contexts(self):
        """Interpreter mode: direction flip is a no-op for context."""
        store = DirectionalContextStore()

        # Simulate interpreter: speaker A talks zh->en
        store.add("zh", "en", "你好", "Hello")
        store.add("zh", "en", "我是小明", "I am Xiaoming")

        # Speaker B responds en->zh (direction flip)
        store.add("en", "zh", "Nice to meet you", "很高兴认识你")

        # Both directions still have their context
        assert len(store.get("zh", "en")) == 2
        assert len(store.get("en", "zh")) == 1

    def test_custom_max_entries_and_tokens(self):
        store = DirectionalContextStore(max_entries=3, max_tokens=500)
        for i in range(5):
            store.add("zh", "en", f"text {i}", f"translation {i}")
        ctx = store.get("zh", "en")
        assert len(ctx) <= 3
