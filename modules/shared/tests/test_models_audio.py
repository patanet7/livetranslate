"""Behavioral tests for AudioChunk and MeetingAudioStream."""

from __future__ import annotations

import pytest

from livetranslate_common.models.audio import AudioChunk, MeetingAudioStream


class TestAudioChunk:
    def test_audio_chunk_creation(self) -> None:
        raw = b"\x00\x01\x02\x03" * 256
        chunk = AudioChunk(
            data=raw,
            timestamp_ms=123456,
            sequence_number=0,
            source_id="mic-device-01",
        )
        assert chunk.data == raw
        assert chunk.timestamp_ms == 123456
        assert chunk.sequence_number == 0
        assert chunk.source_id == "mic-device-01"

    def test_audio_chunk_json_roundtrip(self) -> None:
        raw = b"\x01\x02\x03\x04" * 100
        original = AudioChunk(
            data=raw,
            timestamp_ms=999000,
            sequence_number=42,
            source_id="browser-tab-7",
        )
        json_str = original.model_dump_json()
        restored = AudioChunk.model_validate_json(json_str)
        assert restored.data == original.data
        assert restored.timestamp_ms == original.timestamp_ms
        assert restored.sequence_number == original.sequence_number
        assert restored.source_id == original.source_id


class TestMeetingAudioStreamProtocol:
    def test_protocol_compliance(self) -> None:
        """A class with the required attrs and read_chunk coroutine satisfies the protocol."""

        class FakeStream:
            source_type = "browser"
            sample_rate = 48000
            channels = 1
            encoding = "pcm_s16le"

            async def read_chunk(self) -> AudioChunk | None:
                return None

        stream = FakeStream()
        assert isinstance(stream, MeetingAudioStream)

    @pytest.mark.asyncio
    async def test_read_chunks_until_none(self) -> None:
        """A conforming stream can be exhausted by reading until None."""
        chunks = [
            AudioChunk(data=b"\x00" * 32, timestamp_ms=i * 20, sequence_number=i, source_id="test")
            for i in range(3)
        ]

        class FiniteStream:
            source_type = "test"
            sample_rate = 16000
            channels = 1
            encoding = "pcm_s16le"

            def __init__(self, items: list[AudioChunk]) -> None:
                self._items = list(items)

            async def read_chunk(self) -> AudioChunk | None:
                if self._items:
                    return self._items.pop(0)
                return None

        stream = FiniteStream(chunks)
        assert isinstance(stream, MeetingAudioStream)

        received: list[AudioChunk] = []
        while True:
            chunk = await stream.read_chunk()
            if chunk is None:
                break
            received.append(chunk)

        assert len(received) == 3
        assert received[0].sequence_number == 0
        assert received[2].sequence_number == 2
