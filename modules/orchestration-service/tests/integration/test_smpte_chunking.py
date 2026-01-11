"""
SMPTE Timecode Audio Chunking Test
===================================

Demonstrates and tests audio chunking with SMPTE timecode for proper overlap handling.
This test runs standalone without database dependencies.
"""

import pytest
import numpy as np
from typing import List, Dict, Any
from timecode import Timecode


def generate_test_audio(
    duration_seconds: float = 1.0, frequency: int = 440, sample_rate: int = 16000
) -> bytes:
    """Generate sine wave audio for testing"""
    num_samples = int(duration_seconds * sample_rate)
    t = np.linspace(0, duration_seconds, num_samples, False)
    audio_signal = np.sin(2 * np.pi * frequency * t)
    audio_signal = (audio_signal * 32767).astype(np.int16)
    return audio_signal.tobytes()


def chunk_audio_with_smpte(
    audio_data: bytes,
    chunk_size_ms: int = 500,
    overlap_ms: int = 100,
    sample_rate: int = 16000,
    sample_width: int = 2,
    framerate: str = "30",
) -> List[Dict[str, Any]]:
    """
    Split audio into overlapping chunks with SMPTE timecode.

    Args:
        audio_data: Raw PCM audio bytes
        chunk_size_ms: Chunk duration in milliseconds
        overlap_ms: Overlap duration in milliseconds
        sample_rate: Audio sample rate
        sample_width: Bytes per sample
        framerate: SMPTE framerate

    Returns:
        List of chunks with SMPTE timecode metadata
    """
    samples_per_chunk = int((chunk_size_ms / 1000.0) * sample_rate)
    overlap_samples = int((overlap_ms / 1000.0) * sample_rate)
    bytes_per_chunk = samples_per_chunk * sample_width

    stride_samples = samples_per_chunk - overlap_samples
    stride_bytes = stride_samples * sample_width

    chunks = []

    for i in range(0, len(audio_data), stride_bytes):
        chunk_end = min(i + bytes_per_chunk, len(audio_data))
        chunk = audio_data[i:chunk_end]

        if len(chunk) > 0:
            start_time_seconds = i / (sample_rate * sample_width)
            end_time_seconds = chunk_end / (sample_rate * sample_width)

            # Generate SMPTE timecode
            # Calculate frame numbers from seconds
            fps = int(framerate)
            start_frame = max(1, int(start_time_seconds * fps))
            end_frame = max(1, int(end_time_seconds * fps))

            start_tc = Timecode(framerate, frames=start_frame)
            end_tc = Timecode(framerate, frames=end_frame)

            chunk_metadata = {
                "data": chunk,
                "chunk_index": len(chunks),
                "start_byte": i,
                "end_byte": chunk_end,
                "start_time_seconds": start_time_seconds,
                "end_time_seconds": end_time_seconds,
                "duration_ms": (end_time_seconds - start_time_seconds) * 1000,
                "has_overlap": len(chunks) > 0,
                "overlap_ms": overlap_ms if len(chunks) > 0 else 0,
                "smpte_timecode": {
                    "start": str(start_tc),
                    "end": str(end_tc),
                    "start_frames": start_tc.frames,
                    "end_frames": end_tc.frames,
                    "framerate": framerate,
                },
            }

            chunks.append(chunk_metadata)

    return chunks


@pytest.mark.asyncio
class TestSMPTEChunking:
    """Test SMPTE timecode-based audio chunking"""

    def test_basic_chunking(self):
        """TEST: Basic audio chunking without overlap"""
        audio = generate_test_audio(duration_seconds=2.0)
        chunks = chunk_audio_with_smpte(
            audio,
            chunk_size_ms=500,
            overlap_ms=0,  # No overlap
            framerate="30",
        )

        # Should have 4 chunks (2 seconds / 0.5 seconds)
        assert len(chunks) == 4, f"Expected 4 chunks, got {len(chunks)}"

        # Verify timecodes
        assert chunks[0]["smpte_timecode"]["start"] == "00:00:00:00"
        assert chunks[0]["has_overlap"] == False

        print(f"\n✅ Basic chunking: {len(chunks)} chunks")
        for i, chunk in enumerate(chunks):
            print(
                f"   Chunk {i}: {chunk['smpte_timecode']['start']} -> {chunk['smpte_timecode']['end']}"
            )

    def test_overlapping_chunks(self):
        """TEST: Audio chunking with 20% overlap"""
        audio = generate_test_audio(duration_seconds=5.0)
        chunks = chunk_audio_with_smpte(
            audio,
            chunk_size_ms=500,
            overlap_ms=100,  # 20% overlap
            framerate="30",
        )

        assert len(chunks) > 0, "Should have chunks"

        # Verify overlaps
        for i in range(1, len(chunks)):
            current = chunks[i]
            previous = chunks[i - 1]

            assert current["has_overlap"] == True
            assert current["start_time_seconds"] < previous["end_time_seconds"], (
                f"Chunk {i} should overlap with previous"
            )

            # Verify overlap amount
            overlap_time = previous["end_time_seconds"] - current["start_time_seconds"]
            overlap_ms = overlap_time * 1000
            assert 95 <= overlap_ms <= 105, (
                f"Overlap should be ~100ms, got {overlap_ms:.1f}ms"
            )

        print(
            f"\n✅ Overlapping chunks: {len(chunks)} chunks with {chunks[1]['overlap_ms']}ms overlap"
        )
        print(
            f"   First chunk: {chunks[0]['smpte_timecode']['start']} -> {chunks[0]['smpte_timecode']['end']}"
        )
        print(
            f"   Second chunk: {chunks[1]['smpte_timecode']['start']} -> {chunks[1]['smpte_timecode']['end']}"
        )
        print(f"   Overlap verified: {overlap_ms:.1f}ms")

    def test_smpte_framerate_accuracy(self):
        """TEST: SMPTE timecode frame accuracy"""
        audio = generate_test_audio(duration_seconds=1.0)

        # Test different framerates
        for framerate in ["24", "25", "30", "60"]:
            chunks = chunk_audio_with_smpte(
                audio,
                chunk_size_ms=100,
                overlap_ms=0,
                framerate=framerate,
            )

            assert len(chunks) == 10, f"Should have 10 chunks for {framerate}fps"

            # Verify frame numbers increase
            for i in range(1, len(chunks)):
                assert (
                    chunks[i]["smpte_timecode"]["start_frames"]
                    > chunks[i - 1]["smpte_timecode"]["start_frames"]
                )

            print(
                f"\n✅ {framerate}fps: {len(chunks)} chunks, frames {chunks[0]['smpte_timecode']['start_frames']}-{chunks[-1]['smpte_timecode']['end_frames']}"
            )

    def test_timeline_reconstruction(self):
        """TEST: Reconstruct timeline from SMPTE timecodes"""
        audio = generate_test_audio(duration_seconds=3.0)
        chunks = chunk_audio_with_smpte(
            audio,
            chunk_size_ms=500,
            overlap_ms=100,
            framerate="30",
        )

        # Verify no gaps in timeline
        for i in range(1, len(chunks)):
            gap = chunks[i]["start_time_seconds"] - chunks[i - 1]["end_time_seconds"]
            assert gap <= 0, f"No gaps allowed (gap: {gap:.3f}s at chunk {i})"

        # Verify total duration
        total_duration = (
            chunks[-1]["end_time_seconds"] - chunks[0]["start_time_seconds"]
        )
        assert abs(total_duration - 3.0) < 0.1, (
            f"Total duration should be ~3.0s, got {total_duration:.2f}s"
        )

        print("\n✅ Timeline reconstruction successful")
        print(f"   Total chunks: {len(chunks)}")
        print(f"   Duration: {total_duration:.2f}s")
        print(f"   Start: {chunks[0]['smpte_timecode']['start']}")
        print(f"   End: {chunks[-1]['smpte_timecode']['end']}")


if __name__ == "__main__":
    # Run tests directly
    test = TestSMPTEChunking()
    print("\n" + "=" * 70)
    print("SMPTE Timecode Audio Chunking Tests")
    print("=" * 70)

    test.test_basic_chunking()
    test.test_overlapping_chunks()
    test.test_smpte_framerate_accuracy()
    test.test_timeline_reconstruction()

    print("\n" + "=" * 70)
    print("✅ All SMPTE chunking tests passed!")
    print("=" * 70)
