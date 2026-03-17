"""Tests for SegmentRecord and SegmentStore — Phase 2 of context redesign.

Verifies:
- SegmentRecord phase transitions (draft/final lifecycle)
- SegmentStore sentence accumulation (non-final finals accumulate, is_final flushes)
- flush_pending returns accumulated text and clears state
- reset clears everything
- evict_old keeps only the N most recent records
- Draft path does not interfere with final path
- Already-translated segments are detected via is_final_translated
"""
from translation.segment_record import SegmentPhase, SegmentRecord
from translation.segment_store import SegmentStore


class TestSegmentRecord:
    def test_default_phase_is_draft_received(self):
        rec = SegmentRecord(
            segment_id=1, source_text="hello", source_lang="en", target_lang="zh",
        )
        assert rec.phase == SegmentPhase.DRAFT_RECEIVED
        assert rec.draft_translation is None
        assert rec.final_translation is None

    def test_phase_is_string_enum(self):
        assert SegmentPhase.DRAFT_RECEIVED == "draft_received"
        assert SegmentPhase.FINAL_TRANSLATED == "final_translated"

    def test_all_phases_exist(self):
        phases = {p.value for p in SegmentPhase}
        assert phases == {
            "draft_received", "draft_translated",
            "final_received", "final_translated",
        }


class TestSegmentStoreDraftLifecycle:
    def test_on_draft_received_creates_record(self):
        store = SegmentStore()
        rec = store.on_draft_received(1, "hello", "en", "zh")
        assert rec.segment_id == 1
        assert rec.source_text == "hello"
        assert rec.phase == SegmentPhase.DRAFT_RECEIVED

    def test_on_draft_translated_updates_record(self):
        store = SegmentStore()
        store.on_draft_received(1, "hello", "en", "zh")
        rec = store.on_draft_translated(1, "你好")
        assert rec is not None
        assert rec.draft_translation == "你好"
        assert rec.phase == SegmentPhase.DRAFT_TRANSLATED

    def test_on_draft_translated_unknown_id_returns_none(self):
        store = SegmentStore()
        assert store.on_draft_translated(999, "translation") is None

    def test_draft_does_not_affect_pending_sentence(self):
        store = SegmentStore()
        store.on_draft_received(1, "hello world", "en", "zh")
        assert store._pending_sentence == ""


class TestSegmentStoreFinalLifecycle:
    def test_final_with_is_final_true_returns_text(self):
        store = SegmentStore()
        rec, text = store.on_final_received(1, "Hello world.", True, "en", "zh")
        assert rec.phase == SegmentPhase.FINAL_RECEIVED
        assert text == "Hello world."

    def test_final_with_is_final_false_accumulates(self):
        store = SegmentStore()
        rec, text = store.on_final_received(1, "Hello", False, "en", "zh")
        assert text == ""
        assert store._pending_sentence == "Hello"

    def test_accumulated_sentence_flushes_on_is_final(self):
        store = SegmentStore()
        store.on_final_received(1, "Hello", False, "en", "zh")
        store.on_final_received(2, "world", False, "en", "zh")
        rec, text = store.on_final_received(3, "how are you?", True, "en", "zh")
        assert text == "Hello world how are you?"
        assert store._pending_sentence == ""

    def test_on_final_translated_updates_record(self):
        store = SegmentStore()
        store.on_final_received(1, "Hello.", True, "en", "zh")
        rec = store.on_final_translated(1, "你好。")
        assert rec is not None
        assert rec.final_translation == "你好。"
        assert rec.phase == SegmentPhase.FINAL_TRANSLATED

    def test_on_final_translated_unknown_id_returns_none(self):
        store = SegmentStore()
        assert store.on_final_translated(999, "translation") is None

    def test_is_final_translated(self):
        store = SegmentStore()
        store.on_final_received(1, "Hello.", True, "en", "zh")
        assert not store.is_final_translated(1)
        store.on_final_translated(1, "你好。")
        assert store.is_final_translated(1)

    def test_is_final_translated_unknown_id(self):
        store = SegmentStore()
        assert not store.is_final_translated(999)

    def test_draft_then_final_overwrites_phase(self):
        """A segment can arrive as draft first, then final."""
        store = SegmentStore()
        store.on_draft_received(1, "Hello", "en", "zh")
        rec, text = store.on_final_received(1, "Hello world.", True, "en", "zh")
        assert rec.phase == SegmentPhase.FINAL_RECEIVED
        assert rec.source_text == "Hello world."
        assert text == "Hello world."


class TestSegmentStoreSentenceAccumulation:
    def test_multiple_non_final_then_final(self):
        store = SegmentStore()
        store.on_final_received(10, "The quick", False, "en", "zh")
        store.on_final_received(11, "brown fox", False, "en", "zh")
        rec, text = store.on_final_received(12, "jumps.", True, "en", "zh")
        assert text == "The quick brown fox jumps."

    def test_consecutive_is_final_true(self):
        """Each is_final=True segment stands alone if no pending text."""
        store = SegmentStore()
        _, text1 = store.on_final_received(1, "First sentence.", True, "en", "zh")
        _, text2 = store.on_final_received(2, "Second sentence.", True, "en", "zh")
        assert text1 == "First sentence."
        assert text2 == "Second sentence."

    def test_whitespace_handling(self):
        """Text is stripped and joined with single spaces."""
        store = SegmentStore()
        store.on_final_received(1, "  Hello  ", False, "en", "zh")
        _, text = store.on_final_received(2, "  world.  ", True, "en", "zh")
        assert text == "Hello world."

    def test_pending_segment_ids_tracked(self):
        store = SegmentStore()
        store.on_final_received(10, "Hello", False, "en", "zh")
        store.on_final_received(11, "world", False, "en", "zh")
        assert store._pending_segment_ids == [10, 11]

        store.on_final_received(12, "done.", True, "en", "zh")
        assert store._pending_segment_ids == []


class TestSegmentStoreFlushPending:
    def test_flush_returns_pending_text(self):
        store = SegmentStore()
        store.on_final_received(1, "Hello", False, "en", "zh")
        store.on_final_received(2, "world", False, "en", "zh")
        text = store.flush_pending()
        assert text == "Hello world"

    def test_flush_clears_state(self):
        store = SegmentStore()
        store.on_final_received(1, "Hello", False, "en", "zh")
        store.flush_pending()
        assert store._pending_sentence == ""
        assert store._pending_segment_ids == []

    def test_flush_empty_returns_empty_string(self):
        store = SegmentStore()
        assert store.flush_pending() == ""

    def test_double_flush_returns_empty(self):
        store = SegmentStore()
        store.on_final_received(1, "Hello", False, "en", "zh")
        store.flush_pending()
        assert store.flush_pending() == ""


class TestSegmentStoreReset:
    def test_reset_clears_everything(self):
        store = SegmentStore()
        store.on_draft_received(1, "draft", "en", "zh")
        store.on_final_received(2, "final", False, "en", "zh")
        store.reset()

        assert store.get(1) is None
        assert store.get(2) is None
        assert store._pending_sentence == ""
        assert store._pending_segment_ids == []


class TestSegmentStoreEviction:
    def test_evict_old_keeps_recent(self):
        store = SegmentStore()
        for i in range(10):
            store.on_draft_received(i, f"text {i}", "en", "zh")

        store.evict_old(keep_last=3)
        assert len(store._records) == 3
        # Should keep segments 7, 8, 9 (highest IDs)
        assert store.get(7) is not None
        assert store.get(8) is not None
        assert store.get(9) is not None
        assert store.get(6) is None

    def test_evict_old_noop_when_under_limit(self):
        store = SegmentStore()
        store.on_draft_received(1, "text", "en", "zh")
        store.evict_old(keep_last=50)
        assert len(store._records) == 1

    def test_evict_old_exact_count(self):
        store = SegmentStore()
        for i in range(5):
            store.on_draft_received(i, f"text {i}", "en", "zh")
        store.evict_old(keep_last=5)
        assert len(store._records) == 5

    def test_evict_old_protects_pending_segment_ids(self):
        """Eviction must not remove records referenced by _pending_segment_ids."""
        store = SegmentStore()
        # Create 10 segments, accumulate segments 0-2 as non-final (pending)
        for i in range(10):
            store.on_draft_received(i, f"text {i}", "en", "zh")
        # Simulate non-final finals that accumulate into pending
        store.on_final_received(0, "word one", is_final=False, source_lang="en", target_lang="zh")
        store.on_final_received(1, "word two", is_final=False, source_lang="en", target_lang="zh")
        store.on_final_received(2, "word three", is_final=False, source_lang="en", target_lang="zh")
        # Now evict with keep_last=3 — would normally evict 0-6, but 0,1,2 are pending
        store.evict_old(keep_last=3)
        # Pending records must survive eviction
        assert store.get(0) is not None, "pending segment 0 was evicted"
        assert store.get(1) is not None, "pending segment 1 was evicted"
        assert store.get(2) is not None, "pending segment 2 was evicted"
        # Recent records also survive
        assert store.get(7) is not None
        assert store.get(8) is not None
        assert store.get(9) is not None
        # Non-pending old records should be evicted
        assert store.get(3) is None
        assert store.get(4) is None


class TestSegmentStoreDraftTranslated:
    """Verify on_draft_translated updates phase and stores translation."""

    def test_draft_translated_updates_record(self):
        store = SegmentStore()
        store.on_draft_received(1, "hello world", "en", "zh")
        rec = store.on_draft_translated(1, "你好世界")
        assert rec is not None
        assert rec.phase == SegmentPhase.DRAFT_TRANSLATED
        assert rec.draft_translation == "你好世界"

    def test_draft_translated_unknown_id_returns_none(self):
        store = SegmentStore()
        assert store.on_draft_translated(999, "translation") is None

    def test_draft_then_final_preserves_draft_translation(self):
        """Draft translation is preserved when final arrives for same segment_id."""
        store = SegmentStore()
        store.on_draft_received(1, "hello", "en", "zh")
        store.on_draft_translated(1, "draft: 你好")
        rec, text = store.on_final_received(1, "hello world.", is_final=True, source_lang="en", target_lang="zh")
        assert rec.draft_translation == "draft: 你好"
        assert rec.phase == SegmentPhase.FINAL_RECEIVED
        assert text == "hello world."


class TestSegmentStoreGet:
    def test_get_existing(self):
        store = SegmentStore()
        store.on_draft_received(42, "hello", "en", "zh")
        rec = store.get(42)
        assert rec is not None
        assert rec.segment_id == 42

    def test_get_nonexistent(self):
        store = SegmentStore()
        assert store.get(999) is None
