"""Unit tests for WebSocketTranscriptionClient.

Tests the WebSocket client that connects to the transcription service's
/api/stream endpoint.  External connections (websockets.connect) are mocked
with AsyncMock — everything else uses real objects.
"""
from __future__ import annotations

import asyncio
import json
import struct
import sys
from pathlib import Path
from unittest.mock import AsyncMock, patch

import pytest
import websockets

# Ensure src is importable without installed package
sys.path.insert(0, str(Path(__file__).parent.parent.parent / "src"))

from clients.transcription_client import WebSocketTranscriptionClient


# ---------------------------------------------------------------------------
# Helpers — lightweight fakes that satisfy the websockets ClientConnection
# protocol without pulling in AsyncMock's __aiter__ pitfalls.
# ---------------------------------------------------------------------------

class _FakeWs:
    """Minimal fake websockets ClientConnection.

    Supports ``async for message in ws`` iteration, plus ``send()`` and
    ``close()`` as async callables whose calls can be inspected.
    """

    def __init__(self, messages: list[str | bytes] | None = None):
        self._messages = messages or []
        self.send = AsyncMock()
        self.close = AsyncMock()

    def __aiter__(self):
        return self._iter_messages()

    async def _iter_messages(self):
        for msg in self._messages:
            yield msg


class _DroppingWs(_FakeWs):
    """Fake ws that raises ConnectionClosed on first iteration — simulates a dropped connection."""

    async def _iter_messages(self):
        raise websockets.ConnectionClosed(None, None)
        yield  # noqa: unreachable — makes this an async generator


class _BlockingWs(_FakeWs):
    """Fake ws that blocks indefinitely — simulates a long-lived idle connection."""

    async def _iter_messages(self):
        await asyncio.sleep(3600)
        yield "never reached"


def _make_mock_ws(messages: list[str | bytes] | None = None) -> _FakeWs:
    """Build a fake WebSocket that yields ``messages`` when iterated."""
    return _FakeWs(messages)


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

class TestConnectLifecycle:
    """Verify connect/close transitions."""

    @pytest.mark.asyncio
    async def test_connect_sets_connected_true(self):
        client = WebSocketTranscriptionClient(host="127.0.0.1", port=9999)
        mock_ws = _make_mock_ws()

        with patch("clients.transcription_client.websockets.connect", new_callable=AsyncMock, return_value=mock_ws):
            await client.connect()

        assert client.connected is True
        assert client._ws is mock_ws
        await client.close()

    @pytest.mark.asyncio
    async def test_close_sets_connected_false(self):
        client = WebSocketTranscriptionClient(host="127.0.0.1", port=9999)
        mock_ws = _make_mock_ws()

        with patch("clients.transcription_client.websockets.connect", new_callable=AsyncMock, return_value=mock_ws):
            await client.connect()
            assert client.connected is True

            await client.close()

        assert client.connected is False
        assert client._ws is None

    @pytest.mark.asyncio
    async def test_url_property_formats_correctly(self):
        client = WebSocketTranscriptionClient(host="myhost", port=4242)
        assert client.url == "ws://myhost:4242/api/stream"


class TestSendAudioBinaryFormat:
    """Verify audio frames are sent as raw bytes, not JSON-wrapped."""

    @pytest.mark.asyncio
    async def test_send_audio_passes_bytes_directly(self):
        client = WebSocketTranscriptionClient()
        mock_ws = _make_mock_ws()

        with patch("clients.transcription_client.websockets.connect", new_callable=AsyncMock, return_value=mock_ws):
            await client.connect()

        # 4 float32 samples (16 bytes)
        audio_bytes = struct.pack("<4f", 0.1, -0.2, 0.3, -0.4)
        await client.send_audio(audio_bytes)

        mock_ws.send.assert_awaited_once_with(audio_bytes)
        # Confirm it was NOT wrapped in JSON
        sent_arg = mock_ws.send.call_args[0][0]
        assert isinstance(sent_arg, bytes)

        await client.close()

    @pytest.mark.asyncio
    async def test_send_audio_raises_when_not_connected(self):
        client = WebSocketTranscriptionClient()

        with pytest.raises(RuntimeError, match="Not connected"):
            await client.send_audio(b"\x00" * 16)


class TestSendConfigJson:
    """Verify config messages are well-formed JSON with type='config'."""

    @pytest.mark.asyncio
    async def test_send_config_with_all_fields(self):
        client = WebSocketTranscriptionClient()
        mock_ws = _make_mock_ws()

        with patch("clients.transcription_client.websockets.connect", new_callable=AsyncMock, return_value=mock_ws):
            await client.connect()

        await client.send_config(
            language="en",
            initial_prompt="Meeting about Q4 results",
            glossary_terms=["EBITDA", "synergy"],
        )

        sent_raw = mock_ws.send.call_args[0][0]
        payload = json.loads(sent_raw)

        assert payload["type"] == "config"
        assert payload["language"] == "en"
        assert payload["initial_prompt"] == "Meeting about Q4 results"
        assert payload["glossary_terms"] == ["EBITDA", "synergy"]

        await client.close()

    @pytest.mark.asyncio
    async def test_send_config_omits_none_fields(self):
        client = WebSocketTranscriptionClient()
        mock_ws = _make_mock_ws()

        with patch("clients.transcription_client.websockets.connect", new_callable=AsyncMock, return_value=mock_ws):
            await client.connect()

        await client.send_config(language="fr")

        payload = json.loads(mock_ws.send.call_args[0][0])
        assert payload == {"type": "config", "language": "fr"}
        assert "initial_prompt" not in payload
        assert "glossary_terms" not in payload

        await client.close()

    @pytest.mark.asyncio
    async def test_send_config_raises_when_not_connected(self):
        client = WebSocketTranscriptionClient()

        with pytest.raises(RuntimeError, match="Not connected"):
            await client.send_config(language="en")


class TestSendEnd:
    """Verify end-of-stream signalling."""

    @pytest.mark.asyncio
    async def test_send_end_sends_json_end_type(self):
        client = WebSocketTranscriptionClient()
        mock_ws = _make_mock_ws()

        with patch("clients.transcription_client.websockets.connect", new_callable=AsyncMock, return_value=mock_ws):
            await client.connect()

        await client.send_end()

        payload = json.loads(mock_ws.send.call_args[0][0])
        assert payload == {"type": "end"}

        await client.close()

    @pytest.mark.asyncio
    async def test_send_end_swallows_connection_closed(self):
        client = WebSocketTranscriptionClient()
        mock_ws = _make_mock_ws()
        mock_ws.send.side_effect = websockets.ConnectionClosed(None, None)

        with patch("clients.transcription_client.websockets.connect", new_callable=AsyncMock, return_value=mock_ws):
            await client.connect()

        # Should not raise
        await client.send_end()
        await client.close()


class TestReceiveValidSegment:
    """Verify segment messages dispatch to registered callbacks."""

    @pytest.mark.asyncio
    async def test_segment_callback_receives_parsed_json(self):
        segment_msg = json.dumps({
            "type": "segment",
            "text": "Hello world",
            "start": 0.0,
            "end": 1.5,
            "is_final": True,
        })

        mock_ws = _make_mock_ws(messages=[segment_msg])
        client = WebSocketTranscriptionClient()

        received = []

        async def capture(data):
            received.append(data)

        client.on_segment(capture)

        with patch("clients.transcription_client.websockets.connect", new_callable=AsyncMock, return_value=mock_ws):
            await client.connect()
            # Let the receive loop process the message
            await asyncio.sleep(0.05)

        assert len(received) == 1
        assert received[0]["type"] == "segment"
        assert received[0]["text"] == "Hello world"

        await client.close()

    @pytest.mark.asyncio
    async def test_language_detected_callback_fires(self):
        lang_msg = json.dumps({
            "type": "language_detected",
            "language": "ja",
            "confidence": 0.97,
        })

        mock_ws = _make_mock_ws(messages=[lang_msg])
        client = WebSocketTranscriptionClient()

        received = []

        async def capture(data):
            received.append(data)

        client.on_language_detected(capture)

        with patch("clients.transcription_client.websockets.connect", new_callable=AsyncMock, return_value=mock_ws):
            await client.connect()
            await asyncio.sleep(0.05)

        assert len(received) == 1
        assert received[0]["language"] == "ja"

        await client.close()


class TestReceiveInvalidJson:
    """Verify malformed JSON does not crash the receive loop."""

    @pytest.mark.asyncio
    async def test_malformed_json_is_skipped_gracefully(self):
        messages = [
            "this is not json {{{",
            json.dumps({"type": "segment", "text": "after bad json"}),
        ]

        mock_ws = _make_mock_ws(messages=messages)
        client = WebSocketTranscriptionClient()

        received = []

        async def capture(data):
            received.append(data)

        client.on_segment(capture)

        with patch("clients.transcription_client.websockets.connect", new_callable=AsyncMock, return_value=mock_ws):
            await client.connect()
            await asyncio.sleep(0.05)

        # The valid segment after the bad JSON should still arrive
        assert len(received) == 1
        assert received[0]["text"] == "after bad json"

        await client.close()


class TestReceiveUnknownType:
    """Verify unknown message types do not trigger any callback."""

    @pytest.mark.asyncio
    async def test_unknown_type_dispatches_no_callback(self):
        unknown_msg = json.dumps({"type": "unknown_future_event", "data": 42})

        mock_ws = _make_mock_ws(messages=[unknown_msg])
        client = WebSocketTranscriptionClient()

        segment_received = []
        error_received = []

        async def seg_cb(data):
            segment_received.append(data)

        async def err_cb(data):
            error_received.append(data)

        client.on_segment(seg_cb)
        client.on_error(err_cb)

        with patch("clients.transcription_client.websockets.connect", new_callable=AsyncMock, return_value=mock_ws):
            await client.connect()
            await asyncio.sleep(0.05)

        assert segment_received == []
        assert error_received == []

        await client.close()


class TestCallbackRegistration:
    """Verify on_segment / on_error / on_language_detected registration."""

    def test_on_segment_registers_callback(self):
        client = WebSocketTranscriptionClient()

        async def handler(data):
            pass

        client.on_segment(handler)
        assert handler in client._callbacks["segment"]

    def test_on_error_registers_callback(self):
        client = WebSocketTranscriptionClient()

        async def handler(data):
            pass

        client.on_error(handler)
        assert handler in client._callbacks["error"]

    def test_on_language_detected_registers_callback(self):
        client = WebSocketTranscriptionClient()

        async def handler(data):
            pass

        client.on_language_detected(handler)
        assert handler in client._callbacks["language_detected"]

    def test_on_backend_switched_registers_callback(self):
        client = WebSocketTranscriptionClient()

        async def handler(data):
            pass

        client.on_backend_switched(handler)
        assert handler in client._callbacks["backend_switched"]

    def test_multiple_callbacks_per_event_type(self):
        client = WebSocketTranscriptionClient()

        async def handler_a(data):
            pass

        async def handler_b(data):
            pass

        client.on_segment(handler_a)
        client.on_segment(handler_b)

        assert len(client._callbacks["segment"]) == 2
        assert handler_a in client._callbacks["segment"]
        assert handler_b in client._callbacks["segment"]


class TestCloseCancelsReceiveTask:
    """Verify close() cancels the background receive loop task."""

    @pytest.mark.asyncio
    async def test_receive_task_is_cancelled_on_close(self):
        mock_ws = _BlockingWs()
        client = WebSocketTranscriptionClient()

        with patch("clients.transcription_client.websockets.connect", new_callable=AsyncMock, return_value=mock_ws):
            await client.connect()

        assert client._receive_task is not None
        task = client._receive_task
        assert not task.done()

        await client.close()

        assert task.cancelled() or task.done()
        assert client._ws is None
        assert client.connected is False


class TestReconnectOnConnectionLost:
    """Verify reconnect with exponential backoff when the connection drops."""

    @pytest.mark.asyncio
    async def test_reconnect_attempts_on_unexpected_close(self):
        # First connect drops immediately; reconnect succeeds with a blocking ws.
        call_count = 0

        async def fake_connect(url):
            nonlocal call_count
            call_count += 1
            if call_count == 1:
                return _DroppingWs()
            # Return a blocking ws so the receive loop stays alive after reconnect
            return _BlockingWs()

        client = WebSocketTranscriptionClient(
            host="127.0.0.1",
            port=9999,
            max_reconnect_attempts=3,
            reconnect_base_delay_s=0.01,
        )

        with patch("clients.transcription_client.websockets.connect", side_effect=fake_connect):
            await client.connect()
            # Wait for the receive loop to hit ConnectionClosed and reconnect
            await asyncio.sleep(0.15)

        assert call_count >= 2
        assert client.connected is True

        await client.close()

    @pytest.mark.asyncio
    async def test_reconnect_exhausted_fires_error_callback(self):
        """When all reconnect attempts fail, the error callback should fire."""
        client = WebSocketTranscriptionClient(
            host="127.0.0.1",
            port=9999,
            max_reconnect_attempts=2,
            reconnect_base_delay_s=0.01,
        )

        error_received = []

        async def error_handler(data):
            error_received.append(data)

        client.on_error(error_handler)

        call_count = 0

        async def always_fail(url):
            nonlocal call_count
            call_count += 1
            if call_count == 1:
                return _DroppingWs()
            raise OSError("Connection refused")

        with patch("clients.transcription_client.websockets.connect", side_effect=always_fail):
            await client.connect()
            # Wait for reconnect attempts to exhaust
            await asyncio.sleep(0.3)

        assert len(error_received) == 1
        assert error_received[0]["recoverable"] is False
        assert "connection lost" in error_received[0]["message"].lower()

        await client.close()

    @pytest.mark.asyncio
    async def test_no_reconnect_when_closing_intentionally(self):
        """When close() is called, ConnectionClosed should not trigger reconnect."""
        connect_count = 0

        async def counting_connect(url):
            nonlocal connect_count
            connect_count += 1
            return _BlockingWs()

        client = WebSocketTranscriptionClient(
            host="127.0.0.1",
            port=9999,
            reconnect_base_delay_s=0.01,
        )

        with patch("clients.transcription_client.websockets.connect", side_effect=counting_connect):
            await client.connect()
            await client.close()
            await asyncio.sleep(0.1)

        # Only the initial connect, no reconnect attempts
        assert connect_count == 1


class TestCallbackErrorIsolation:
    """Verify that a failing callback does not crash the receive loop."""

    @pytest.mark.asyncio
    async def test_exception_in_callback_does_not_stop_processing(self):
        messages = [
            json.dumps({"type": "segment", "text": "first"}),
            json.dumps({"type": "segment", "text": "second"}),
        ]

        mock_ws = _make_mock_ws(messages=messages)
        client = WebSocketTranscriptionClient()

        results = []

        async def exploding_callback(data):
            if data["text"] == "first":
                raise ValueError("Boom")
            results.append(data)

        client.on_segment(exploding_callback)

        with patch("clients.transcription_client.websockets.connect", new_callable=AsyncMock, return_value=mock_ws):
            await client.connect()
            await asyncio.sleep(0.05)

        # The second message should still be processed despite the first callback exploding
        assert len(results) == 1
        assert results[0]["text"] == "second"

        await client.close()


class TestBinaryFramesFromServerAreIgnored:
    """Verify binary frames from the server are silently skipped."""

    @pytest.mark.asyncio
    async def test_binary_server_frame_does_not_trigger_callback(self):
        messages = [
            b"\x00\x01\x02\x03",  # binary frame — should be skipped
            json.dumps({"type": "segment", "text": "real segment"}),
        ]

        mock_ws = _make_mock_ws(messages=messages)
        client = WebSocketTranscriptionClient()

        received = []

        async def capture(data):
            received.append(data)

        client.on_segment(capture)

        with patch("clients.transcription_client.websockets.connect", new_callable=AsyncMock, return_value=mock_ws):
            await client.connect()
            await asyncio.sleep(0.05)

        assert len(received) == 1
        assert received[0]["text"] == "real segment"

        await client.close()
