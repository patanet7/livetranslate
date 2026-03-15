"""WebSocket client for the transcription service.

Replaces SocketIOWhisperClient with a plain WebSocket client that speaks
the same binary-audio + JSON-control protocol as /api/stream.

Designed for dual-pipe use: both the loopback WebSocket handler and the
Fireflies/Google Meet adapters can create instances to forward audio.

Protocol:
  Client → Service:
    - Binary frames: raw float32 PCM audio at 16 kHz mono
    - JSON text:     {"type": "config", ...} or {"type": "end"}
  Service → Client:
    - JSON text:     segment, language_detected, backend_switched, error
"""
from __future__ import annotations

import asyncio
import json
from collections.abc import Awaitable, Callable
from typing import Any

import websockets
from websockets.asyncio.client import ClientConnection
from livetranslate_common.logging import get_logger

logger = get_logger()

# Type alias for async callbacks that receive a parsed JSON dict
MessageCallback = Callable[[dict[str, Any]], Awaitable[None]]


class WebSocketTranscriptionClient:
    """Plain WebSocket client for the transcription service's /api/stream endpoint.

    Usage::

        client = WebSocketTranscriptionClient(host="localhost", port=5001)
        client.on_segment(my_segment_handler)
        await client.connect()
        await client.send_config(language="en")
        await client.send_audio(audio_bytes)
        await client.send_end()
        await client.close()
    """

    def __init__(
        self,
        host: str = "localhost",
        port: int = 5001,
        max_reconnect_attempts: int = 5,
        reconnect_base_delay_s: float = 0.5,
    ):
        self.host = host
        self.port = port
        self.max_reconnect_attempts = max_reconnect_attempts
        self.reconnect_base_delay_s = reconnect_base_delay_s

        self._ws: ClientConnection | None = None
        self._receive_task: asyncio.Task | None = None
        self._connected = False
        self._closing = False

        self._callbacks: dict[str, list[MessageCallback]] = {
            "segment": [],
            "language_detected": [],
            "backend_switched": [],
            "error": [],
        }

    @property
    def url(self) -> str:
        return f"ws://{self.host}:{self.port}/api/stream"

    @property
    def connected(self) -> bool:
        return self._connected and self._ws is not None

    def on_segment(self, callback: MessageCallback) -> None:
        """Register callback for transcription segment messages."""
        self._callbacks["segment"].append(callback)

    def on_language_detected(self, callback: MessageCallback) -> None:
        """Register callback for language detection messages."""
        self._callbacks["language_detected"].append(callback)

    def on_backend_switched(self, callback: MessageCallback) -> None:
        """Register callback for backend switch messages."""
        self._callbacks["backend_switched"].append(callback)

    def on_error(self, callback: MessageCallback) -> None:
        """Register callback for error messages from the transcription service."""
        self._callbacks["error"].append(callback)

    async def connect(self) -> None:
        """Connect to the transcription service and start the receive loop."""
        self._closing = False
        try:
            self._ws = await websockets.connect(
                self.url, ping_interval=20, ping_timeout=10
            )
            self._connected = True
            self._receive_task = asyncio.create_task(self._receive_loop())
            logger.info("transcription_client_connected", url=self.url)
        except (OSError, websockets.WebSocketException) as exc:
            self._connected = False
            logger.error("transcription_client_connect_failed", url=self.url, error=str(exc))
            raise

    async def send_config(
        self,
        language: str | None = None,
        initial_prompt: str | None = None,
        glossary_terms: list[str] | None = None,
    ) -> None:
        """Send a config message to the transcription service."""
        if not self._ws:
            raise RuntimeError("Not connected to transcription service")

        msg: dict[str, Any] = {"type": "config"}
        if language is not None:
            msg["language"] = language
        if initial_prompt is not None:
            msg["initial_prompt"] = initial_prompt
        if glossary_terms is not None:
            msg["glossary_terms"] = glossary_terms

        await self._ws.send(json.dumps(msg))

    async def send_audio(self, audio_bytes: bytes) -> None:
        """Send raw float32 PCM audio bytes to the transcription service.

        The transcription service expects 16 kHz mono float32 frames
        (``np.ndarray.tobytes()``).
        """
        if not self._ws:
            raise RuntimeError("Not connected to transcription service")
        await self._ws.send(audio_bytes)

    async def send_end(self) -> None:
        """Signal end of the audio stream (graceful close)."""
        if self._ws:
            try:
                await self._ws.send(json.dumps({"type": "end"}))
            except websockets.ConnectionClosed:
                pass

    async def close(self) -> None:
        """Close the connection and stop the receive loop."""
        self._closing = True
        if self._receive_task and not self._receive_task.done():
            self._receive_task.cancel()
            try:
                await self._receive_task
            except asyncio.CancelledError:
                pass
        if self._ws:
            try:
                await self._ws.close()
            except Exception:
                pass
        self._ws = None
        self._connected = False
        logger.info("transcription_client_closed")

    async def _receive_loop(self) -> None:
        """Background loop: read messages from the transcription service and dispatch callbacks.

        Owns reconnect logic inline — never spawns a second task.
        """
        while not self._closing:
            try:
                async for message in self._ws:
                    if isinstance(message, bytes):
                        continue  # transcription service only sends text frames
                    try:
                        data = json.loads(message)
                    except json.JSONDecodeError:
                        logger.warning("transcription_client_invalid_json")
                        continue

                    msg_type = data.get("type")
                    callbacks = self._callbacks.get(msg_type, [])
                    for cb in callbacks:
                        try:
                            await cb(data)
                        except Exception:
                            logger.exception(
                                "transcription_callback_error",
                                msg_type=msg_type,
                            )

                # Server closed cleanly — attempt reconnect if not shutting down
                if not self._closing:
                    self._connected = False
                    logger.warning("transcription_client_connection_lost")
                    reconnected = await self._attempt_reconnect()
                    if not reconnected:
                        break
                else:
                    break

            except websockets.ConnectionClosed:
                self._connected = False
                if not self._closing:
                    logger.warning("transcription_client_connection_lost")
                    reconnected = await self._attempt_reconnect()
                    if not reconnected:
                        break
                else:
                    break

            except asyncio.CancelledError:
                raise

            except Exception:
                self._connected = False
                logger.exception("transcription_client_receive_error")
                break

    async def _attempt_reconnect(self) -> bool:
        """Attempt to reconnect with exponential backoff.

        Returns True if reconnected, False if all attempts exhausted.
        Does NOT spawn a new receive task — the caller (_receive_loop) continues.
        """
        for attempt in range(1, self.max_reconnect_attempts + 1):
            delay = self.reconnect_base_delay_s * (2 ** (attempt - 1))
            logger.info(
                "transcription_client_reconnecting",
                attempt=attempt,
                max_attempts=self.max_reconnect_attempts,
                delay_s=delay,
            )
            await asyncio.sleep(delay)

            try:
                self._ws = await websockets.connect(
                self.url, ping_interval=20, ping_timeout=10
            )
                self._connected = True
                logger.info("transcription_client_reconnected", attempt=attempt)
                return True
            except (OSError, websockets.WebSocketException) as exc:
                logger.warning(
                    "transcription_client_reconnect_failed",
                    attempt=attempt,
                    error=str(exc),
                )

        # All attempts exhausted
        logger.error("transcription_client_reconnect_exhausted")
        for cb in self._callbacks.get("error", []):
            try:
                await cb({
                    "type": "error",
                    "message": "Transcription service connection lost after all reconnect attempts",
                    "recoverable": False,
                })
            except Exception:
                pass
        return False
