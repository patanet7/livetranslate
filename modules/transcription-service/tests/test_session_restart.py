"""Behavioral tests for session restart on language switch.

Tests verify:
- VACOnlineProcessor.reset() clears buffer and counters
- HallucinationFilter.reset() clears cross-segment state
- _dedup_overlap with empty prev preserves full text
"""

import numpy as np
import pytest

from vac_online_processor import VACOnlineProcessor


@pytest.mark.asyncio
class TestVACProcessorReset:
    """VACOnlineProcessor must support reset() for session restart."""

    def test_vac_has_reset_method(self):
        vac = VACOnlineProcessor(prebuffer_s=0.5, overlap_s=1.0, stride_s=4.5)
        assert hasattr(vac, "reset"), "VACOnlineProcessor must have reset()"

    async def test_reset_clears_buffer_and_counters(self):
        vac = VACOnlineProcessor(prebuffer_s=0.5, overlap_s=1.0, stride_s=4.5)

        # Feed 1 second of audio
        audio = np.zeros(16000, dtype=np.float32)
        await vac.feed_audio(audio)
        assert vac._buffer_samples > 0

        vac.reset()
        assert vac._buffer_samples == 0
        assert vac._new_samples_since_inference == 0
        assert vac._first_inference_done is False

    async def test_reset_makes_not_ready_for_inference(self):
        vac = VACOnlineProcessor(prebuffer_s=0.5, overlap_s=1.0, stride_s=4.5)

        # Feed enough audio for prebuffer readiness
        audio = np.zeros(8000, dtype=np.float32)  # 0.5s
        await vac.feed_audio(audio)

        vac.reset()
        assert not vac.ready_for_inference()

    async def test_reset_re_enables_prebuffer_threshold(self):
        """After reset, first inference should use prebuffer_s, not stride_s."""
        vac = VACOnlineProcessor(prebuffer_s=0.5, overlap_s=0.5, stride_s=4.5)

        # First session: feed prebuffer amount → should be ready
        audio = np.zeros(8000, dtype=np.float32)  # 0.5s = prebuffer_s
        await vac.feed_audio(audio)
        assert vac.ready_for_inference()

        # Consume the inference audio (marks first_inference_done)
        vac.get_inference_audio()

        # Now need full stride (4.5s) for next inference
        small_audio = np.zeros(8000, dtype=np.float32)  # only 0.5s
        await vac.feed_audio(small_audio)
        assert not vac.ready_for_inference()  # needs 4.5s, only has 0.5s

        # Reset → back to prebuffer threshold
        vac.reset()
        audio2 = np.zeros(8000, dtype=np.float32)  # 0.5s = prebuffer_s
        await vac.feed_audio(audio2)
        assert vac.ready_for_inference()  # prebuffer threshold again!


class TestHallucinationFilterReset:
    """HallucinationFilter.reset() already exists — verify it works."""

    def test_reset_clears_recent_texts(self):
        from transcription.hallucination_filter import HallucinationFilter

        hf = HallucinationFilter()
        hf.reset()
        assert len(hf._recent_texts) == 0

    def test_reset_callable_without_error(self):
        from transcription.hallucination_filter import HallucinationFilter

        hf = HallucinationFilter()
        hf.reset()  # should not raise


class TestDedupAfterRestart:
    """After session restart, dedup must not match against pre-restart text."""

    def test_empty_prev_preserves_cjk(self):
        from api import _dedup_overlap

        assert _dedup_overlap("", "你好世界") == "你好世界"

    def test_empty_prev_preserves_english(self):
        from api import _dedup_overlap

        assert _dedup_overlap("", "Hello world and more") == "Hello world and more"
