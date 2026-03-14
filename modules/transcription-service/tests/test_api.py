"""Tests for transcription service WebSocket API."""
import json

import numpy as np
import pytest
from fastapi.testclient import TestClient

from api import create_app


@pytest.fixture
def app():
    return create_app(registry_path=None)


@pytest.fixture
def client(app):
    return TestClient(app)


class TestHealthEndpoint:
    def test_health(self, client):
        resp = client.get("/health")
        assert resp.status_code == 200
        data = resp.json()
        assert "status" in data
        assert "loaded_backends" in data


class TestModelsEndpoint:
    def test_models_empty(self, client):
        resp = client.get("/api/models")
        assert resp.status_code == 200
        assert isinstance(resp.json(), list)


class TestWebSocketStream:
    """Tests for WebSocket /api/stream endpoint."""

    def test_config_message_accepted(self, client):
        """Config messages should be accepted without error."""
        with client.websocket_connect("/api/stream") as ws:
            ws.send_text(json.dumps({
                "type": "config",
                "language": "zh",
                "initial_prompt": "Technical meeting about AI",
                "glossary_terms": ["GPT", "transformer", "attention"],
            }))
            # Send end to close cleanly
            ws.send_text(json.dumps({"type": "end"}))

    def test_end_message_closes_stream(self, client):
        """End message should close the WebSocket cleanly."""
        with client.websocket_connect("/api/stream") as ws:
            ws.send_text(json.dumps({"type": "end"}))
            # Connection should close without error

    def test_binary_audio_no_registry_continues(self, client):
        """Binary audio with no registry should be silently ignored."""
        with client.websocket_connect("/api/stream") as ws:
            audio = np.zeros(16000, dtype=np.float32)
            ws.send_bytes(audio.tobytes())
            # Send end — should not have crashed
            ws.send_text(json.dumps({"type": "end"}))

    def test_malformed_json_continues(self, client):
        """Malformed JSON control frames should not crash the connection."""
        with client.websocket_connect("/api/stream") as ws:
            ws.send_text("not valid json {{{")
            # Connection should still work
            ws.send_text(json.dumps({"type": "end"}))

    def test_undersized_audio_ignored(self, client):
        """Audio frames < 100ms (1600 samples) should be silently dropped."""
        with client.websocket_connect("/api/stream") as ws:
            tiny_audio = np.zeros(100, dtype=np.float32)  # ~6ms
            ws.send_bytes(tiny_audio.tobytes())
            ws.send_text(json.dumps({"type": "end"}))
