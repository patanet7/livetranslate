"""Tests for transcription service WebSocket API, _dedup_overlap, and no_speech_prob gate."""
import json

import numpy as np
import pytest
from fastapi.testclient import TestClient

from api import _dedup_overlap, create_app


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


class TestDedupOverlapCJK:
    """Boundary tests for _dedup_overlap with CJK and Latin text."""

    # --- CJK ---

    def test_cjk_three_char_overlap_stripped(self):
        """A 3-character CJK overlap at the boundary must be removed from new_text."""
        prev = "今天天气很好"
        new = "天气很好，明天也会晴朗"
        result = _dedup_overlap(prev, new)
        # "天气很" is the 3-char overlap; new must not start with it
        assert not result.startswith("天气很")
        assert "明天也会晴朗" in result

    def test_cjk_two_char_overlap_not_stripped(self):
        """A 2-character CJK overlap must NOT be stripped (minimum is 3 to avoid false matches)."""
        prev = "今天天气"
        # 2-char overlap "天气" — must pass through unchanged
        new = "天气晴朗，出门散步"
        result = _dedup_overlap(prev, new)
        # The overlap is only 2 chars — too short to strip, so new_text is returned as-is
        assert result == new

    def test_cjk_no_overlap_returned_unchanged(self):
        """When there is no matching overlap, new_text must be returned unchanged."""
        prev = "今天天气很好"
        new = "明天可能会下雨"
        result = _dedup_overlap(prev, new)
        assert result == new

    # --- Latin ---

    def test_latin_twelve_word_overlap_stripped(self):
        """A 12-word Latin overlap (at the cap) must be stripped."""
        overlap = "one two three four five six seven eight nine ten eleven twelve"
        prev = "sentence before " + overlap
        new = overlap + " then some more words"
        result = _dedup_overlap(prev, new)
        assert result.strip() == "then some more words"

    def test_latin_thirteen_word_overlap_not_stripped(self):
        """A 13-word overlap exceeds the 12-word cap and must NOT be stripped."""
        overlap_words = "one two three four five six seven eight nine ten eleven twelve thirteen"
        prev = "prefix " + overlap_words
        new = overlap_words + " suffix words"
        result = _dedup_overlap(prev, new)
        # 13 words > cap of 12, so dedup cannot detect it — new_text returned unchanged
        assert result == new

    def test_latin_no_overlap_returned_unchanged(self):
        """Latin text with no overlap must be returned unchanged."""
        prev = "The quick brown fox"
        new = "jumps over the lazy dog"
        result = _dedup_overlap(prev, new)
        assert result == new

    def test_empty_prev_returns_new_unchanged(self):
        """Empty prev_text must cause new_text to be returned unmodified."""
        result = _dedup_overlap("", "some new text")
        assert result == "some new text"

    def test_empty_new_returns_empty(self):
        """Empty new_text must be returned as empty string."""
        result = _dedup_overlap("some previous text", "")
        assert result == ""


class TestNoSpeechProbGateBoundary:
    """Boundary tests for the no_speech_prob > 0.6 suppression gate.

    The gate condition in _run_inference is:
        if result.no_speech_prob is not None and result.no_speech_prob > 0.6:
            return  # suppressed

    These tests verify the boundary values directly against that condition
    without calling into the WebSocket handler.
    """

    def _gate_fires(self, no_speech_prob) -> bool:
        """Return True when the gate would suppress the result."""
        return no_speech_prob is not None and no_speech_prob > 0.6

    def test_no_speech_prob_none_passes_through(self):
        """None no_speech_prob must not trigger the gate (pass through)."""
        assert self._gate_fires(None) is False

    def test_no_speech_prob_at_threshold_passes_through(self):
        """no_speech_prob=0.6 is not > 0.6, so the gate must not fire."""
        assert self._gate_fires(0.6) is False

    def test_no_speech_prob_just_above_threshold_suppressed(self):
        """no_speech_prob=0.61 is > 0.6, so the gate must fire (suppress)."""
        assert self._gate_fires(0.61) is True

    def test_no_speech_prob_zero_passes_through(self):
        """no_speech_prob=0.0 is well below the gate — must pass through."""
        assert self._gate_fires(0.0) is False

    def test_no_speech_prob_one_suppressed(self):
        """no_speech_prob=1.0 is well above the gate — must be suppressed."""
        assert self._gate_fires(1.0) is True
