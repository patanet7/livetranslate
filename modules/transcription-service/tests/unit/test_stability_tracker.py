"""Tests for SimpleStabilityTracker and draft→final segment lifecycle.

Bug 1: Finals should have ALL text as stable (unstable_text="").
  The stability tracker was splitting at the overlap boundary even for
  final passes, leaving the tail of every segment permanently "gray".

Bug 2: is_draft and is_final are independent flags — is_final means
  "ends with punctuation" (Whisper), is_draft means "first-pass preview"
  (two-pass system). A draft CAN have is_final=True.
"""
import pytest
from dataclasses import dataclass


@dataclass
class FakeSegment:
    text: str
    start_ms: int
    end_ms: int


# Import the class under test
# api.py is in src/ which is on sys.path when running from transcription-service
from api import SimpleStabilityTracker


class TestStabilityTrackerSplit:
    """SimpleStabilityTracker.split() should respect draft vs final semantics."""

    def test_draft_has_unstable_tail(self):
        """Draft pass: text in the overlap window should be unstable."""
        tracker = SimpleStabilityTracker(overlap_s=1.0)
        segments = [
            FakeSegment(text="Hello everyone", start_ms=0, end_ms=2000),
            FakeSegment(text="welcome to the demo", start_ms=2000, end_ms=4000),
            FakeSegment(text="of our system", start_ms=4000, end_ms=5000),
        ]
        text = "Hello everyone welcome to the demo of our system"
        stable, unstable = tracker.split(text, segments)

        # Last segment ends at 5000ms, overlap=1000ms → cutoff at 4000ms
        # Segments at or before 4000ms are stable, after are unstable
        assert "Hello everyone" in stable
        assert "welcome to the demo" in stable
        assert "of our system" in unstable

    def test_final_all_text_stable(self):
        """Final pass: ALL text should be stable — the next segment handles its own overlap."""
        tracker = SimpleStabilityTracker(overlap_s=1.0)
        segments = [
            FakeSegment(text="Hello everyone", start_ms=0, end_ms=2000),
            FakeSegment(text="welcome to the demo", start_ms=2000, end_ms=4000),
            FakeSegment(text="of our system", start_ms=4000, end_ms=5000),
        ]
        text = "Hello everyone welcome to the demo of our system"
        # On a final pass, we should be able to mark everything as stable
        stable, unstable = tracker.split(text, segments, is_final=True)

        assert stable == text
        assert unstable == ""

    def test_draft_short_text_all_unstable(self):
        """Very short text (≤2 words) should be entirely unstable on drafts."""
        tracker = SimpleStabilityTracker(overlap_s=1.0)
        stable, unstable = tracker.split("Hello world", [])
        assert stable == ""
        assert unstable == "Hello world"

    def test_final_short_text_all_stable(self):
        """Even short text should be fully stable on finals."""
        tracker = SimpleStabilityTracker(overlap_s=1.0)
        stable, unstable = tracker.split("Hello world", [], is_final=True)
        assert stable == "Hello world"
        assert unstable == ""


class TestDedupOverlap:
    """_dedup_overlap correctly strips repeated text from consecutive segments."""

    def test_word_overlap_removed(self):
        from api import _dedup_overlap
        prev = "Hello everyone welcome to the"
        new = "to the demo of our system"
        result = _dedup_overlap(prev, new)
        assert result == "demo of our system"

    def test_no_overlap_unchanged(self):
        from api import _dedup_overlap
        prev = "Hello everyone"
        new = "completely different text"
        result = _dedup_overlap(prev, new)
        assert result == "completely different text"

    def test_empty_prev_unchanged(self):
        from api import _dedup_overlap
        result = _dedup_overlap("", "Hello world")
        assert result == "Hello world"

    def test_cjk_overlap_removed(self):
        from api import _dedup_overlap
        prev = "今天我们要展示实时翻译"
        new = "实时翻译系统的功能"
        result = _dedup_overlap(prev, new)
        assert "实时翻译" not in result  # overlap should be stripped
        assert "系统的功能" in result
