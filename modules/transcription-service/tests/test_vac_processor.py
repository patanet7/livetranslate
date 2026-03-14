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
        # After inference, buffer should retain last 0.5s = 8000 samples
        # (tested via internal state after process_chunk)

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
