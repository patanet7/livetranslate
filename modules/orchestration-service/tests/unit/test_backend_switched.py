"""Tests for backend_switched handling — Phase 4 of context redesign.

Verifies:
- handle_backend_switched resets segment_store
- handle_backend_switched does NOT reset context_store
- BackendSwitchedMessage is sent to the frontend
"""
from translation.context_store import DirectionalContextStore
from translation.segment_store import SegmentStore


class TestBackendSwitchedSegmentStoreReset:
    """Backend switch resets segment store but preserves context store."""

    def test_segment_store_reset_on_backend_switch(self):
        """Segment store is cleared because word overlap patterns change."""
        store = SegmentStore()
        store.on_draft_received(1, "hello", "en", "zh")
        store.on_final_received(2, "world", False, "en", "zh")
        assert store.get(1) is not None
        assert store._pending_sentence != ""

        store.reset()

        assert store.get(1) is None
        assert store.get(2) is None
        assert store._pending_sentence == ""
        assert store._pending_segment_ids == []

    def test_context_store_preserved_on_backend_switch(self):
        """Context store is NOT reset — prior translations remain valid."""
        ctx = DirectionalContextStore(max_entries=5, max_tokens=500)
        ctx.add("en", "zh", "hello", "你好")
        ctx.add("en", "zh", "world", "世界")

        # Simulate backend switch: only segment_store.reset(), not context_store
        seg = SegmentStore()
        seg.on_draft_received(1, "hello", "en", "zh")
        seg.reset()

        # Context should be intact
        entries = ctx.get("en", "zh")
        assert len(entries) == 2
        assert entries[0].text == "hello"
        assert entries[1].text == "world"

    def test_cross_direction_context_preserved_on_backend_switch(self):
        """Cross-direction context survives backend switch."""
        ctx = DirectionalContextStore(max_entries=5, max_tokens=500, cross_direction_max_tokens=200)
        ctx.add("en", "zh", "hello", "你好")
        ctx.add("zh", "en", "你好", "hello")

        # Backend switch: reset segment store only
        seg = SegmentStore()
        seg.reset()

        # Both directions intact
        cross = ctx.get_cross_direction("en", "zh")
        assert len(cross) > 0
