"""Tests for refactored VACOnlineProcessor."""
import asyncio

import numpy as np
import pytest

from vac_online_processor import VACOnlineProcessor


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
        # 0.3s at 16kHz = 4800 samples
        audio = np.zeros(4800, dtype=np.float32)
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
        # Feed prebuffer amount and trigger first inference
        audio = np.zeros(4800, dtype=np.float32)  # 0.3s
        await proc.feed_audio(audio)
        assert proc.ready_for_inference() is True
        proc.get_inference_audio()  # consume

        # Feed less than stride_s — should not be ready
        audio2 = np.zeros(16000, dtype=np.float32)  # 1.0s < 4.5s stride
        await proc.feed_audio(audio2)
        assert proc.ready_for_inference() is False
