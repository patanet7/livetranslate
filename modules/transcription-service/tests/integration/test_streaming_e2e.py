"""End-to-end streaming transcription tests.

These tests stream real audio through the /api/stream WebSocket endpoint
in 100ms chunks at real-time pace — exactly how the browser sends audio.

The audio passes through the full pipeline:
  Binary chunks → Queue → VACOnlineProcessor → Backend.transcribe() → Segments

Requires: transcription service running locally on port 5001.
  Start with: uv run python src/main.py --registry config/model_registry.local.yaml

Run:
  uv run pytest modules/transcription-service/tests/integration/test_streaming_e2e.py -v -s
"""
from __future__ import annotations

import asyncio
import json
import sys
from pathlib import Path

import numpy as np
import pytest
import soundfile as sf

sys.path.insert(0, str(Path(__file__).parent.parent.parent / "src"))

FIXTURES_DIR = Path(__file__).parent.parent / "fixtures" / "audio"
SERVICE_URL = "ws://localhost:5001/api/stream"


async def _stream_audio(
    audio: np.ndarray,
    sample_rate: int = 16000,
    language: str | None = "en",
    chunk_duration_ms: int = 100,
    pace_factor: float = 1.0,
    recv_timeout_s: float = 60.0,
) -> list[dict]:
    """Stream audio through the transcription service at real-time pace.

    Args:
        audio: float32 PCM audio at ``sample_rate``.
        sample_rate: Audio sample rate in Hz.
        language: BCP-47 language hint (None for auto-detect).
        chunk_duration_ms: Size of each chunk in milliseconds.
        pace_factor: 1.0 = real-time, 0.5 = 2x speed, 2.0 = half speed.
        recv_timeout_s: How long to wait for results after all audio is sent.

    Returns:
        List of received JSON messages from the service.
    """
    import websockets

    ws = await websockets.connect(SERVICE_URL, ping_interval=None, close_timeout=30)

    # Send config
    config: dict = {"type": "config"}
    if language:
        config["language"] = language
    await ws.send(json.dumps(config))

    # Stream audio in chunks at paced rate
    chunk_samples = int(sample_rate * chunk_duration_ms / 1000)
    chunk_sleep = (chunk_duration_ms / 1000) * pace_factor
    chunks_sent = 0

    for i in range(0, len(audio), chunk_samples):
        chunk = audio[i : i + chunk_samples]
        if len(chunk) < 160:  # skip tiny tail fragments
            break
        await ws.send(chunk.astype(np.float32).tobytes())
        chunks_sent += 1
        await asyncio.sleep(chunk_sleep)

    # Signal end of stream
    await ws.send(json.dumps({"type": "end"}))

    # Collect all results
    messages = []
    deadline = asyncio.get_event_loop().time() + recv_timeout_s
    while asyncio.get_event_loop().time() < deadline:
        try:
            msg = await asyncio.wait_for(ws.recv(), timeout=min(30, recv_timeout_s))
            data = json.loads(msg)
            messages.append(data)
        except asyncio.TimeoutError:
            break
        except Exception:
            break

    await ws.close()
    return messages


def _load_fixture(name: str) -> tuple[np.ndarray, int]:
    """Load an audio fixture file. Returns (audio, sample_rate)."""
    path = FIXTURES_DIR / name
    if not path.exists():
        pytest.skip(f"Audio fixture not found: {path}")
    data, sr = sf.read(str(path))
    # Mix to mono if stereo
    if data.ndim == 2:
        data = data.mean(axis=1)
    return data.astype(np.float32), sr


def _extract_segments(messages: list[dict]) -> list[dict]:
    return [m for m in messages if m.get("type") == "segment"]


def _extract_full_text(messages: list[dict]) -> str:
    segments = _extract_segments(messages)
    return " ".join(s.get("text", "") for s in segments).strip()


async def _check_service():
    """Check if transcription service is running."""
    try:
        import websockets
        ws = await asyncio.wait_for(
            websockets.connect(SERVICE_URL, ping_interval=None),
            timeout=3,
        )
        await ws.close()
        return True
    except Exception:
        return False


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


@pytest.fixture(autouse=True)
async def require_service():
    """Skip all tests if transcription service isn't running."""
    if not await _check_service():
        pytest.skip(
            "Transcription service not running on port 5001. "
            "Start with: uv run python src/main.py --registry config/model_registry.local.yaml"
        )


@pytest.mark.asyncio
@pytest.mark.e2e
class TestJFKStreaming:
    """Stream the JFK inaugural address and verify transcription."""

    async def test_jfk_produces_segments(self):
        """Streaming JFK audio at real-time pace produces transcription segments."""
        audio, sr = _load_fixture("jfk.wav")
        messages = await _stream_audio(audio, sr, language="en", pace_factor=1.0)

        segments = _extract_segments(messages)
        assert len(segments) >= 1, f"Expected segments, got: {[m['type'] for m in messages]}"

    async def test_jfk_text_contains_speech_content(self):
        """JFK transcription contains recognizable speech words."""
        audio, sr = _load_fixture("jfk.wav")
        messages = await _stream_audio(audio, sr, language="en", pace_factor=1.0)

        full_text = _extract_full_text(messages).lower()
        assert len(full_text) > 10, f"Transcription too short: '{full_text}'"

        # Should contain real English words (not hallucination)
        has_real_words = any(
            word in full_text
            for word in ["country", "ask", "fellow", "american", "do", "what", "your", "can"]
        )
        assert has_real_words, f"Expected speech content, got: '{full_text[:200]}'"

    async def test_jfk_language_detected_before_segment(self):
        """language_detected must fire before the first segment."""
        audio, sr = _load_fixture("jfk.wav")
        messages = await _stream_audio(audio, sr, language="en", pace_factor=1.0)

        types = [m["type"] for m in messages]
        if "segment" in types:
            seg_idx = types.index("segment")
            assert "language_detected" in types[:seg_idx], (
                f"language_detected must come before first segment. Order: {types}"
            )

    async def test_jfk_segments_have_timing(self):
        """Segments should have start_ms and end_ms fields from the expanded wire format."""
        audio, sr = _load_fixture("jfk.wav")
        messages = await _stream_audio(audio, sr, language="en", pace_factor=1.0)

        segments = _extract_segments(messages)
        if not segments:
            pytest.skip("No segments produced")

        for seg in segments:
            # Wire format should include start_ms/end_ms (Task 1.3 fix)
            assert "start_ms" in seg, f"Missing start_ms in segment: {seg.keys()}"
            assert "end_ms" in seg, f"Missing end_ms in segment: {seg.keys()}"
            assert "stable_text" in seg, f"Missing stable_text in segment: {seg.keys()}"
            assert "unstable_text" in seg, f"Missing unstable_text in segment: {seg.keys()}"


@pytest.mark.asyncio
@pytest.mark.e2e
class TestStreamingBehavior:
    """Test streaming/chunking behavior with various audio inputs."""

    async def test_long_audio_produces_multiple_segments(self):
        """30s audio should produce multiple transcription segments."""
        audio, sr = _load_fixture("long_30s_speech.wav")
        messages = await _stream_audio(
            audio, sr, language="en", pace_factor=0.5, recv_timeout_s=90
        )
        segments = _extract_segments(messages)
        # 30s of audio with 5s chunk_duration should produce ~6 inference windows
        assert len(segments) >= 2, (
            f"Expected multiple segments for 30s audio, got {len(segments)}"
        )

    async def test_short_audio_handled_gracefully(self):
        """0.5s audio should not crash (may produce 0 or 1 segments)."""
        audio, sr = _load_fixture("short_500ms.wav")
        messages = await _stream_audio(audio, sr, language="en", pace_factor=1.0, recv_timeout_s=15)
        # Should not crash — 0 segments is OK for audio shorter than prebuffer
        errors = [m for m in messages if m.get("type") == "error" and not m.get("recoverable")]
        assert len(errors) == 0, f"Unrecoverable error: {errors}"

    async def test_silence_produces_no_hallucination(self):
        """Pure silence should produce empty or minimal text."""
        audio, sr = _load_fixture("silence.wav")
        messages = await _stream_audio(audio, sr, language="en", pace_factor=1.0, recv_timeout_s=15)
        full_text = _extract_full_text(messages)
        # Silence should produce little to no text
        assert len(full_text) < 50, f"Silence produced too much text: '{full_text}'"

    async def test_noisy_audio_still_produces_output(self):
        """Speech+noise should still produce segments (not crash or hang)."""
        audio, sr = _load_fixture("noisy.wav")
        messages = await _stream_audio(audio, sr, language="en", pace_factor=1.0, recv_timeout_s=30)
        # Should not crash
        errors = [m for m in messages if m.get("type") == "error" and not m.get("recoverable")]
        assert len(errors) == 0, f"Unrecoverable error: {errors}"

    async def test_2x_speed_still_works(self):
        """Streaming at 2x real-time should work (some frames may be dropped by backpressure)."""
        audio, sr = _load_fixture("jfk.wav")
        messages = await _stream_audio(
            audio, sr, language="en", pace_factor=0.5, recv_timeout_s=30
        )
        # Should get at least some results even at 2x speed
        segments = _extract_segments(messages)
        # At 2x, some frames drop — but we should still get results
        types = [m["type"] for m in messages]
        assert "language_detected" in types or len(segments) >= 1, (
            f"Expected at least language_detected or segments at 2x speed, got: {types}"
        )

    async def test_auto_language_detection(self):
        """Streaming without language hint triggers auto-detection."""
        audio, sr = _load_fixture("jfk.wav")
        messages = await _stream_audio(
            audio, sr, language=None, pace_factor=1.0, recv_timeout_s=60
        )
        lang_msgs = [m for m in messages if m.get("type") == "language_detected"]
        if lang_msgs:
            detected = lang_msgs[0]["language"]
            assert detected == "en", f"Expected 'en', got '{detected}'"


@pytest.mark.asyncio
@pytest.mark.e2e
class TestChunkingEdgeCases:
    """Test chunking/VAC edge cases."""

    async def test_varying_chunk_sizes(self):
        """Sending different chunk sizes should not crash the pipeline."""
        audio, sr = _load_fixture("jfk.wav")

        import websockets
        ws = await websockets.connect(SERVICE_URL, ping_interval=None)
        await ws.send(json.dumps({"type": "config", "language": "en"}))

        # Send chunks of varying sizes: 50ms, 100ms, 200ms, 150ms
        sizes = [800, 1600, 3200, 2400]  # samples at 16kHz
        offset = 0
        for size in sizes * 10:
            if offset >= len(audio):
                break
            chunk = audio[offset : offset + size]
            await ws.send(chunk.astype(np.float32).tobytes())
            offset += size
            await asyncio.sleep(0.05)

        await ws.send(json.dumps({"type": "end"}))

        messages = []
        while True:
            try:
                msg = await asyncio.wait_for(ws.recv(), timeout=30)
                messages.append(json.loads(msg))
            except Exception:
                break

        await ws.close()

        # Should not crash — any segments are a bonus
        errors = [m for m in messages if m.get("type") == "error" and not m.get("recoverable")]
        assert len(errors) == 0, f"Unrecoverable error with varying chunks: {errors}"

    async def test_rapid_reconnect(self):
        """Connecting, sending a few chunks, disconnecting, and reconnecting should work."""
        import websockets

        audio, sr = _load_fixture("jfk.wav")

        for i in range(3):
            ws = await websockets.connect(SERVICE_URL, ping_interval=None)
            await ws.send(json.dumps({"type": "config", "language": "en"}))
            # Send 1 second of audio
            chunk = audio[: sr].astype(np.float32)
            await ws.send(chunk.tobytes())
            await asyncio.sleep(0.5)  # give service time to process
            await ws.close()
            await asyncio.sleep(0.5)  # breathing room between reconnects

        # Final connection should still work
        ws = await websockets.connect(SERVICE_URL, ping_interval=None)
        await ws.send(json.dumps({"type": "config", "language": "en"}))
        await ws.send(audio.astype(np.float32).tobytes())
        await asyncio.sleep(0.5)
        await ws.send(json.dumps({"type": "end"}))

        messages = []
        while True:
            try:
                msg = await asyncio.wait_for(ws.recv(), timeout=30)
                messages.append(json.loads(msg))
            except Exception:
                break
        await ws.close()

        # Service should still be responsive after rapid reconnects
        types = [m["type"] for m in messages]
        assert len(types) > 0, "Service stopped responding after rapid reconnects"
