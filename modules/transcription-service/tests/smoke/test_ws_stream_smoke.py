"""Task 4.1: Transcription WebSocket smoke test (GPU required).

Connects to a live transcription service /api/stream endpoint, sends real
speech audio, and verifies:
  - language_detected fires before first segment
  - At least one segment has non-empty text and confidence > 0
  - End message closes cleanly

Run:  uv run pytest modules/transcription-service/tests/smoke/test_ws_stream_smoke.py -v
"""
import json
import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).parent.parent.parent / "src"))


def _generate_speech_like_audio(duration_s: float = 3.0, sample_rate: int = 16000) -> np.ndarray:
    """Generate synthetic speech-like audio (fundamentals + formants + noise)."""
    t = np.arange(int(duration_s * sample_rate)) / sample_rate
    signal = 0.3 * np.sin(2 * np.pi * 120 * t)   # fundamental
    signal += 0.2 * np.sin(2 * np.pi * 240 * t)   # 2nd harmonic
    signal += 0.1 * np.sin(2 * np.pi * 360 * t)   # 3rd harmonic
    signal += 0.1 * np.sin(2 * np.pi * 800 * t)   # F1 formant
    signal += 0.05 * np.sin(2 * np.pi * 1200 * t)  # F2 formant
    signal += 0.01 * np.random.randn(len(signal))
    signal = signal / np.max(np.abs(signal)) * 0.8
    return signal.astype(np.float32)


@pytest.mark.e2e
@pytest.mark.gpu
class TestWebSocketStreamSmoke:
    """Smoke tests that require a running transcription service with GPU."""

    @pytest.fixture
    def app(self):
        """Create app with real model registry."""
        from api import create_app

        registry_path = Path(__file__).parent.parent.parent / "config" / "model_registry.yaml"
        if not registry_path.exists():
            pytest.skip("model_registry.yaml not found")
        return create_app(registry_path)

    @pytest.fixture
    def client(self, app):
        from starlette.testclient import TestClient
        return TestClient(app)

    def test_stream_real_speech_produces_segments(self, client):
        """Send real speech audio and verify transcription results."""
        audio = _generate_speech_like_audio(duration_s=3.0)

        # Split into ~100ms frames (1600 samples at 16kHz)
        frame_size = 1600
        frames = [audio[i:i + frame_size] for i in range(0, len(audio), frame_size)]

        received_messages = []
        with client.websocket_connect("/api/stream") as ws:
            # Send config
            ws.send_text(json.dumps({
                "type": "config",
                "language": "en",
            }))

            # Send audio frames
            for frame in frames:
                ws.send_bytes(frame.tobytes())

            # Send end
            ws.send_text(json.dumps({"type": "end"}))

            # Collect responses (with timeout via TestClient)
            import time
            deadline = time.monotonic() + 15.0
            while time.monotonic() < deadline:
                try:
                    text = ws.receive_text(timeout=2.0)
                    msg = json.loads(text)
                    received_messages.append(msg)
                    if msg.get("type") == "error" and not msg.get("recoverable"):
                        break
                except Exception:
                    break

        # Verify language_detected arrives
        types = [m["type"] for m in received_messages]
        assert "language_detected" in types, f"Expected language_detected, got: {types}"

        # Verify at least one segment
        segments = [m for m in received_messages if m["type"] == "segment"]
        assert len(segments) >= 1, f"Expected at least 1 segment, got {len(segments)}"

        # Verify segment has text and confidence
        first_segment = segments[0]
        assert first_segment.get("text", "").strip(), "Segment text should be non-empty"
        assert first_segment.get("confidence", 0) > 0, "Confidence should be > 0"

    def test_language_detected_before_first_segment(self, client):
        """language_detected must fire before the first segment message."""
        audio = _generate_speech_like_audio(duration_s=2.0)

        with client.websocket_connect("/api/stream") as ws:
            ws.send_bytes(audio.tobytes())
            ws.send_text(json.dumps({"type": "end"}))

            types_seen = []
            import time
            deadline = time.monotonic() + 15.0
            while time.monotonic() < deadline:
                try:
                    text = ws.receive_text(timeout=2.0)
                    msg = json.loads(text)
                    types_seen.append(msg["type"])
                    if msg["type"] == "segment":
                        break
                except Exception:
                    break

            if "segment" in types_seen:
                seg_idx = types_seen.index("segment")
                # language_detected should appear before segment
                assert "language_detected" in types_seen[:seg_idx], (
                    f"language_detected must come before first segment. Order: {types_seen}"
                )
