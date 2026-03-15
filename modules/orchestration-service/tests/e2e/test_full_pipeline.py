"""Task 4.2: Full audio pipeline end-to-end test (GPU + orchestration).

Connect to orchestration /api/audio/stream, send start_session with 48kHz
audio, stream real speech, and receive at least one validated SegmentMessage.

Requires:
  - Running transcription service on TRANSCRIPTION_HOST:TRANSCRIPTION_PORT
  - GPU for transcription inference

Run:  uv run pytest modules/orchestration-service/tests/e2e/test_full_pipeline.py -v -m "e2e and gpu"
"""
import json
import os
import sys
from pathlib import Path

import numpy as np
import pytest

_src = Path(__file__).parent.parent.parent / "src"
if str(_src) not in sys.path:
    sys.path.insert(0, str(_src))


def _generate_48khz_speech(duration_s: float = 5.0) -> np.ndarray:
    """Generate 48kHz mono speech-like audio matching browser output format."""
    sample_rate = 48000
    t = np.arange(int(duration_s * sample_rate)) / sample_rate
    signal = 0.3 * np.sin(2 * np.pi * 120 * t)
    signal += 0.2 * np.sin(2 * np.pi * 240 * t)
    signal += 0.1 * np.sin(2 * np.pi * 800 * t)
    signal += 0.01 * np.random.randn(len(signal))
    signal = signal / np.max(np.abs(signal)) * 0.8
    return signal.astype(np.float32)


@pytest.mark.e2e
@pytest.mark.gpu
class TestFullAudioPipeline:
    """End-to-end test: browser → orchestration → transcription → SegmentMessage."""

    @pytest.fixture
    def app(self):
        from fastapi import FastAPI
        from routers.audio.websocket_audio import router
        return _make_app(router)

    @pytest.fixture
    def ws_client(self, app):
        from starlette.testclient import TestClient
        return TestClient(app)

    def test_48khz_audio_produces_segment_message(self, ws_client):
        """Stream 5s of 48kHz audio and receive a validated SegmentMessage."""
        audio = _generate_48khz_speech(duration_s=5.0)

        # Split into ~100ms frames at 48kHz = 4800 samples
        frame_size = 4800
        frames = [audio[i:i + frame_size] for i in range(0, len(audio), frame_size)]

        received = []
        with ws_client.websocket_connect("/api/audio/stream") as ws:
            # Should receive ConnectedMessage
            connected = json.loads(ws.receive_text())
            assert connected["type"] == "connected"
            assert "session_id" in connected

            # Send start_session (48kHz, 1 channel — browser audio)
            ws.send_text(json.dumps({
                "type": "start_session",
                "sample_rate": 48000,
                "channels": 1,
            }))

            # Stream audio frames as binary
            for frame in frames:
                ws.send_bytes(frame.tobytes())

            # Send end_session
            ws.send_text(json.dumps({"type": "end_session"}))

            # Collect results
            import time
            deadline = time.monotonic() + 30.0
            while time.monotonic() < deadline:
                try:
                    text = ws.receive_text(timeout=3.0)
                    msg = json.loads(text)
                    received.append(msg)
                except Exception:
                    break

        # Must have received at least one segment
        segments = [m for m in received if m["type"] == "segment"]
        assert len(segments) >= 1, f"Expected segments, got message types: {[m['type'] for m in received]}"

        # Validate SegmentMessage fields
        seg = segments[0]
        from livetranslate_common.models.ws_messages import SegmentMessage
        validated = SegmentMessage.model_validate(seg)
        assert validated.text.strip(), "Segment text should be non-empty"
        assert validated.confidence > 0
        assert validated.segment_id >= 1


def _make_app(router):
    from fastapi import FastAPI
    app = FastAPI()
    app.include_router(router, prefix="/api/audio")
    return app
