"""Tests for refactored VACOnlineProcessor, SimpleStabilityTracker, and RMS gate."""
import asyncio
from dataclasses import dataclass

import numpy as np
import pytest

from api import SimpleStabilityTracker
from vac_online_processor import VACOnlineProcessor


def _speech_audio(n_samples: int) -> np.ndarray:
    """Generate non-silent audio (sine wave) that passes the RMS speech gate."""
    t = np.arange(n_samples, dtype=np.float32) / 16000.0
    return (0.1 * np.sin(2 * np.pi * 440 * t)).astype(np.float32)


class TestVACProcessorQueue:
    @pytest.mark.asyncio
    async def test_uses_asyncio_queue(self):
        """Processor should use asyncio.Queue instead of list + flag."""
        proc = VACOnlineProcessor(
            prebuffer_s=0.3, overlap_s=0.5, stride_s=4.5,
        )
        assert hasattr(proc, "_audio_queue")
        assert isinstance(proc._audio_queue, asyncio.Queue)

    @pytest.mark.asyncio
    async def test_retains_overlap_after_inference(self):
        """After inference, last overlap_s seconds should be retained in buffer."""
        proc = VACOnlineProcessor(
            prebuffer_s=0.3, overlap_s=0.5, stride_s=4.5,
        )
        # Feed 1 second of audio at 16kHz
        audio = np.zeros(16000, dtype=np.float32)
        await proc.feed_audio(audio)

        # Get inference audio — this triggers overlap retention
        inference_audio = proc.get_inference_audio()
        assert len(inference_audio) == 16000  # full buffer returned

        # After get_inference_audio, buffer should retain last overlap_s = 0.5s = 8000 samples
        assert proc._buffer_samples == 8000

    @pytest.mark.asyncio
    async def test_first_inference_at_prebuffer(self):
        """First inference should fire at prebuffer_s, not stride_s."""
        proc = VACOnlineProcessor(
            prebuffer_s=0.3, overlap_s=0.5, stride_s=4.5,
        )
        # 0.3s at 16kHz = 4800 samples (must be non-silent to pass RMS gate)
        audio = _speech_audio(4800)
        await proc.feed_audio(audio)
        assert proc.ready_for_inference() is True

    @pytest.mark.asyncio
    async def test_not_ready_before_prebuffer(self):
        """Should not be ready for inference before prebuffer threshold."""
        proc = VACOnlineProcessor(prebuffer_s=0.3, overlap_s=0.5, stride_s=4.5)
        # Feed 0.2s = 3200 samples (less than 0.3s prebuffer)
        audio = np.zeros(3200, dtype=np.float32)
        await proc.feed_audio(audio)
        assert proc.ready_for_inference() is False

    @pytest.mark.asyncio
    async def test_subsequent_inference_at_stride(self):
        """After first inference, next should wait for stride_s of new audio."""
        proc = VACOnlineProcessor(prebuffer_s=0.3, overlap_s=0.5, stride_s=4.5)
        # Feed prebuffer amount and trigger first inference (non-silent)
        audio = _speech_audio(4800)  # 0.3s
        await proc.feed_audio(audio)
        assert proc.ready_for_inference() is True
        proc.get_inference_audio()  # consume

        # Feed less than stride_s — should not be ready
        audio2 = _speech_audio(16000)  # 1.0s < 4.5s stride
        await proc.feed_audio(audio2)
        assert proc.ready_for_inference() is False

    @pytest.mark.asyncio
    async def test_silence_suppressed_by_rms_gate(self):
        """Pure silence should not trigger inference even with enough samples."""
        proc = VACOnlineProcessor(prebuffer_s=0.3, overlap_s=0.5, stride_s=4.5)
        # Feed silent audio exceeding prebuffer threshold
        audio = np.zeros(4800, dtype=np.float32)  # 0.3s of silence
        await proc.feed_audio(audio)
        assert proc.ready_for_inference() is False  # silence gated


# ---------------------------------------------------------------------------
# Helpers shared by the new test classes below
# ---------------------------------------------------------------------------

@dataclass
class FakeSegment:
    """Minimal stand-in for a real Segment that only needs text and end_ms."""
    text: str
    start_ms: int
    end_ms: int


class TestSimpleStabilityTracker:
    """Behavioral tests for SimpleStabilityTracker.split()."""

    def test_split_empty_text_returns_empty_pair(self):
        """split('', []) must return ('', '')."""
        tracker = SimpleStabilityTracker(overlap_s=1.0)
        stable, unstable = tracker.split("", [])
        assert stable == ""
        assert unstable == ""

    def test_split_short_text_no_segments_returns_all_unstable(self):
        """With 2 or fewer words and no segments, everything is unstable."""
        tracker = SimpleStabilityTracker(overlap_s=1.0)
        stable, unstable = tracker.split("short", [])
        assert stable == ""
        assert unstable == "short"

        stable2, unstable2 = tracker.split("two words", [])
        assert stable2 == ""
        assert unstable2 == "two words"

    def test_split_segments_all_after_cutoff_are_unstable(self):
        """When all segment end_ms values exceed the cutoff, everything is unstable."""
        tracker = SimpleStabilityTracker(overlap_s=2.0)
        # last_end_ms = 3000, overlap_ms = 2000, cutoff = 1000
        # Both segments end after 1000 ms → both unstable
        segs = [
            FakeSegment(text="hello", start_ms=0, end_ms=1500),
            FakeSegment(text="world", start_ms=1500, end_ms=3000),
        ]
        stable, unstable = tracker.split("hello world", segs)
        assert stable == ""
        assert "hello" in unstable
        assert "world" in unstable

    def test_split_segments_all_before_cutoff_are_stable(self):
        """When all segment end_ms values are at or before the cutoff, everything is stable."""
        tracker = SimpleStabilityTracker(overlap_s=0.5)
        # last_end_ms = 5000, overlap_ms = 500, cutoff = 4500
        # Both content segments end at 1000 and 2000 (< 4500) → both stable
        # A trailing silent/anchor segment sets last_end_ms to 5000
        segs = [
            FakeSegment(text="hello", start_ms=0, end_ms=1000),
            FakeSegment(text="world", start_ms=1000, end_ms=2000),
            FakeSegment(text="", start_ms=2000, end_ms=5000),
        ]
        stable, unstable = tracker.split("hello world", segs)
        assert "hello" in stable
        assert "world" in stable
        assert unstable == ""

    def test_split_empty_segments_falls_back_to_word_count(self):
        """With an empty segments list, the word-count fallback must be used."""
        tracker = SimpleStabilityTracker(overlap_s=1.0)
        # 10 words — fallback keeps last ~20% (2 words) unstable
        text = "one two three four five six seven eight nine ten"
        stable, unstable = tracker.split(text, [])
        assert stable != ""
        assert unstable != ""
        # Together they reconstruct the full text
        reconstructed = (stable + " " + unstable).strip()
        assert reconstructed == text

    def test_split_overlap_zero_cutoff_equals_last_end_all_stable(self):
        """With overlap_s=0.0, cutoff equals last_end_ms so all segments are stable."""
        tracker = SimpleStabilityTracker(overlap_s=0.0)
        segs = [
            FakeSegment(text="alpha", start_ms=0, end_ms=1000),
            FakeSegment(text="beta", start_ms=1000, end_ms=2000),
        ]
        stable, unstable = tracker.split("alpha beta", segs)
        # cutoff_ms = 2000 - 0 = 2000; end_ms <= 2000 for both → stable
        assert "alpha" in stable
        assert "beta" in stable
        assert unstable == ""


class TestHasSpeechRMSBoundary:
    """Boundary tests for VACOnlineProcessor._has_speech RMS gate."""

    @pytest.mark.asyncio
    async def test_rms_exactly_at_threshold_is_silence(self):
        """Audio with RMS exactly 0.005 must be treated as silence (threshold is >0.005)."""
        proc = VACOnlineProcessor(prebuffer_s=0.3, overlap_s=0.5, stride_s=4.5)
        # A constant-value array has RMS == abs(value)
        n = 8000
        audio = np.full(n, 0.005, dtype=np.float32)
        actual_rms = float(np.sqrt(np.mean(audio ** 2)))
        assert abs(actual_rms - 0.005) < 1e-6
        await proc.feed_audio(audio)
        # RMS is exactly at threshold; gate is strictly > so this is silence
        result = proc._has_speech(rms_threshold=0.005)
        assert result is False

    @pytest.mark.asyncio
    async def test_rms_above_threshold_is_speech(self):
        """Audio with RMS of 0.006 must be treated as speech (above threshold 0.005)."""
        proc = VACOnlineProcessor(prebuffer_s=0.3, overlap_s=0.5, stride_s=4.5)
        n = 8000
        value = 0.006
        audio = np.full(n, value, dtype=np.float32)
        actual_rms = float(np.sqrt(np.mean(audio ** 2)))
        assert actual_rms > 0.005
        await proc.feed_audio(audio)
        result = proc._has_speech(rms_threshold=0.005)
        assert result is True
