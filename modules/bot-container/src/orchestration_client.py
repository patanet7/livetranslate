#!/usr/bin/env python3
"""
WebSocket Client for Bot → Orchestration Communication

Bot containers use the SAME WebSocket protocol as frontend!

Architecture:
    Bot Container (this client)
        ↓ ws://orchestration:3000/ws
    Orchestration Service (websocket_frontend_handler.py)
        ↓ authenticate, track session
    Streaming Coordinator (streaming_coordinator.py)
        ↓ deduplicate, group speakers
    Whisper Service
        ↓ segments back

Key Benefits:
- Reuses existing orchestration infrastructure
- Same authentication and session management
- Consistent segment processing (deduplication, speaker grouping)
- Account ID tracking flows through orchestration
"""

import asyncio
import json
import base64
import logging
from datetime import datetime, timezone
from typing import Optional, Callable, Dict, Any
import websockets
from websockets.asyncio.client import ClientConnection

logger = logging.getLogger(__name__)


class OrchestrationClient:
    """
    Bot's WebSocket client to orchestration service.

    Uses the SAME protocol as frontend clients:
    - Authenticate with user token
    - Start streaming session
    - Stream audio chunks (base64 encoded)
    - Receive transcription segments (already processed)

    Usage:
        client = OrchestrationClient(
            orchestration_url="ws://orchestration:3000/ws",
            user_token="user-api-token",
            meeting_id="meeting-123",
            connection_id="bot-connection-456"
        )

        # Register segment handler
        client.on_segment(lambda seg: print(f"Segment: {seg['text']}"))

        # Connect and authenticate
        await client.connect()

        # Stream audio
        await client.send_audio_chunk(audio_bytes)
    """

    def __init__(
        self,
        orchestration_url: str,
        user_token: str,
        meeting_id: str,
        connection_id: str,
        auto_reconnect: bool = True
    ):
        """
        Initialize orchestration client

        Args:
            orchestration_url: WebSocket URL (e.g., "ws://orchestration:3000/ws")
            user_token: User API token for authentication
            meeting_id: Meeting identifier
            connection_id: Unique bot connection ID
            auto_reconnect: Enable automatic reconnection on disconnect
        """
        self.orchestration_url = orchestration_url
        self.user_token = user_token
        self.meeting_id = meeting_id
        self.connection_id = connection_id
        self.auto_reconnect = auto_reconnect

        # Connection state
        self.websocket: Optional[ClientConnection] = None
        self.connected = False
        self.authenticated = False
        self.session_started = False

        # Callbacks
        self.segment_callback: Optional[Callable[[Dict[str, Any]], None]] = None
        self.error_callback: Optional[Callable[[str], None]] = None
        self.connection_callback: Optional[Callable[[bool], None]] = None

        # Background tasks
        self.receiver_task: Optional[asyncio.Task] = None

    async def connect(self) -> bool:
        """
        Connect to orchestration WebSocket server and authenticate

        Follows the same flow as frontend:
        1. Connect to WebSocket
        2. Send authenticate message
        3. Receive authenticated response
        4. Send start_session message
        5. Receive session_started response
        6. Start receiving segments in background

        Returns:
            True if connection successful, False otherwise
        """
        try:
            logger.info(f"Connecting to orchestration at {self.orchestration_url}")

            # Connect to WebSocket
            self.websocket = await websockets.connect(
                self.orchestration_url,
                ping_interval=30,
                ping_timeout=10
            )

            self.connected = True
            logger.info("WebSocket connection established")

            # Step 1: Authenticate (handle connection:established)
            await self._authenticate()

            # Step 2: Start session
            await self._start_session()

            # Step 3: Start background receiver for segments
            self.receiver_task = asyncio.create_task(self._receive_messages())

            # Notify connection callback
            if self.connection_callback:
                await self.connection_callback(True)

            logger.info(f"✅ Bot connected and authenticated: {self.connection_id}")
            return True

        except Exception as e:
            logger.error(f"Failed to connect to orchestration: {e}", exc_info=True)
            self.connected = False

            if self.error_callback:
                # Check if callback is async or sync
                if asyncio.iscoroutinefunction(self.error_callback):
                    await self.error_callback(f"Connection failed: {e}")
                else:
                    self.error_callback(f"Connection failed: {e}")

            return False

    async def _authenticate(self):
        """
        Handle WebSocket connection establishment

        The orchestration service sends 'connection:established' immediately upon connection.
        We treat this as successful authentication for bot connections.
        """
        # Wait for connection:established message
        response_raw = await self.websocket.recv()
        response = json.loads(response_raw)

        if response.get("type") != "connection:established":
            raise RuntimeError(f"Connection failed: Expected 'connection:established', got {response}")

        logger.info(f"✅ Connection established: {response.get('data', {}).get('connectionId')}")
        self.authenticated = True

    async def _start_session(self):
        """
        Start streaming session (same format as frontend)

        Message format:
        {
            "type": "start_session",
            "session_id": "{connection_id}",
            "config": {
                "model": "large-v3",
                "language": "en",
                "enable_vad": true,
                "enable_cif": true
            }
        }
        """
        session_message = {
            "type": "start_session",
            "session_id": self.connection_id,
            "config": {
                "model": "large-v3",
                "language": "en",
                "enable_vad": True,
                "enable_cif": True,
                "enable_rolling_context": True
            }
        }

        logger.info(f"Starting session: {self.connection_id}")
        await self.websocket.send(json.dumps(session_message))

        # Wait for session_started response
        response_raw = await self.websocket.recv()
        response = json.loads(response_raw)

        if response.get("type") != "session_started":
            raise RuntimeError(f"Session start failed: {response}")

        self.session_started = True
        logger.info(f"✅ Session started: {response.get('session_id')}")

    async def send_audio_chunk(self, audio_data: bytes, timestamp: Optional[datetime] = None):
        """
        Send audio chunk to orchestration (same format as frontend)

        Message format:
        {
            "type": "audio_chunk",
            "audio": "<base64-encoded>",
            "timestamp": "2025-01-15T10:30:00.000Z"
        }

        Args:
            audio_data: Raw audio bytes (PCM, 16kHz, mono, float32)
            timestamp: Optional timestamp (defaults to now)
        """
        if not self.websocket or not self.session_started:
            logger.warning("Cannot send audio: not connected or session not started")
            return

        # Encode audio to base64 (same as frontend)
        audio_base64 = base64.b64encode(audio_data).decode('utf-8')

        # Format timestamp (same as frontend)
        if timestamp is None:
            timestamp = datetime.now(timezone.utc)

        timestamp_str = timestamp.isoformat().replace('+00:00', 'Z')

        # Create message (same as frontend)
        message = {
            "type": "audio_chunk",
            "audio": audio_base64,
            "timestamp": timestamp_str
        }

        await self.websocket.send(json.dumps(message))
        logger.debug(f"Sent audio chunk: {len(audio_data)} bytes")

    async def _receive_messages(self):
        """
        Background task to receive messages from orchestration

        Expected message types:
        - segment: Transcription segment (deduplicated, speaker grouped)
        - error: Error message
        - ping: Heartbeat
        """
        try:
            async for message_raw in self.websocket:
                try:
                    data = json.loads(message_raw)
                    msg_type = data.get("type")

                    if msg_type == "segment":
                        # Segment from orchestration (already processed!)
                        # - Deduplicated by absolute_start_time
                        # - Speaker grouped
                        # - Timestamped
                        await self._handle_segment(data)

                    elif msg_type == "translation":
                        # Translation result (if enabled)
                        await self._handle_translation(data)

                    elif msg_type == "error":
                        # Error from orchestration
                        error_msg = data.get("error", "Unknown error")
                        logger.error(f"Error from orchestration: {error_msg}")
                        if self.error_callback:
                            await self.error_callback(error_msg)

                    elif msg_type == "ping":
                        # Heartbeat - send pong
                        await self.websocket.send(json.dumps({"type": "pong"}))

                    elif msg_type == "pong":
                        # Pong response (ignore)
                        pass

                    else:
                        logger.warning(f"Unknown message type: {msg_type}")

                except json.JSONDecodeError as e:
                    logger.error(f"Invalid JSON from orchestration: {e}")
                except Exception as e:
                    logger.error(f"Error processing message: {e}", exc_info=True)

        except websockets.exceptions.ConnectionClosed:
            logger.warning("WebSocket connection closed by orchestration")
            self.connected = False
            self.authenticated = False
            self.session_started = False

            if self.connection_callback:
                await self.connection_callback(False)

            # Auto-reconnect if enabled
            if self.auto_reconnect:
                logger.info("Attempting to reconnect...")
                await asyncio.sleep(5)
                await self.connect()

        except Exception as e:
            logger.error(f"Error in receive loop: {e}", exc_info=True)
            if self.error_callback:
                await self.error_callback(f"Receive error: {e}")

    async def _handle_segment(self, segment: Dict[str, Any]):
        """Handle received segment from orchestration"""
        logger.debug(f"Received segment: {segment.get('text', '')[:50]}")

        if self.segment_callback:
            try:
                # Call callback (can be sync or async)
                result = self.segment_callback(segment)
                if asyncio.iscoroutine(result):
                    await result
            except Exception as e:
                logger.error(f"Error in segment callback: {e}", exc_info=True)

    async def _handle_translation(self, translation: Dict[str, Any]):
        """Handle received translation from orchestration"""
        logger.debug(f"Received translation: {translation.get('text', '')[:50]}")
        # TODO: Add translation callback if needed

    async def disconnect(self):
        """Cleanly disconnect from orchestration"""
        if not self.websocket:
            return

        logger.info(f"Disconnecting bot {self.connection_id}")

        # Send end_session message
        if self.session_started:
            try:
                await self.websocket.send(json.dumps({
                    "type": "end_session",
                    "session_id": self.connection_id
                }))
            except Exception as e:
                logger.warning(f"Error sending end_session: {e}")

        # Cancel receiver task
        if self.receiver_task:
            self.receiver_task.cancel()
            try:
                await self.receiver_task
            except asyncio.CancelledError:
                pass

        # Close WebSocket
        try:
            await self.websocket.close()
        except Exception as e:
            logger.warning(f"Error closing websocket: {e}")

        self.connected = False
        self.authenticated = False
        self.session_started = False

        logger.info(f"✅ Disconnected: {self.connection_id}")

    # Callback registration methods

    def on_segment(self, callback: Callable[[Dict[str, Any]], None]):
        """
        Register callback for transcription segments

        Args:
            callback: Function(segment_dict) called when segment arrives
        """
        self.segment_callback = callback

    def on_error(self, callback: Callable[[str], None]):
        """
        Register callback for errors

        Args:
            callback: Function(error_message) called on errors
        """
        self.error_callback = callback

    def on_connection_change(self, callback: Callable[[bool], None]):
        """
        Register callback for connection state changes

        Args:
            callback: Function(connected: bool) called on state change
        """
        self.connection_callback = callback

    # Status methods

    def is_connected(self) -> bool:
        """Check if connected to orchestration"""
        return self.connected and self.websocket is not None

    def is_authenticated(self) -> bool:
        """Check if authenticated"""
        return self.authenticated

    def is_session_active(self) -> bool:
        """Check if session is active"""
        return self.session_started

    def get_status(self) -> Dict[str, Any]:
        """Get client status"""
        return {
            "connection_id": self.connection_id,
            "connected": self.connected,
            "authenticated": self.authenticated,
            "session_started": self.session_started,
            "websocket_open": self.websocket.open if self.websocket else False
        }


# Example usage
async def example_usage():
    """Example of using OrchestrationClient"""

    client = OrchestrationClient(
        orchestration_url="ws://localhost:3000/ws",
        user_token="test-token-123",
        meeting_id="meeting-456",
        connection_id="bot-789"
    )

    # Register callbacks
    def on_segment(segment):
        print(f"📄 Segment: {segment.get('text')}")
        print(f"   Speaker: {segment.get('speaker')}")
        print(f"   Time: {segment.get('absolute_start_time')}")

    def on_error(error):
        print(f"❌ Error: {error}")

    client.on_segment(on_segment)
    client.on_error(on_error)

    # Connect
    connected = await client.connect()
    if not connected:
        print("Failed to connect")
        return

    # Simulate audio streaming
    import numpy as np
    for i in range(10):
        # Generate 1 second of test audio
        audio = np.random.randn(16000).astype(np.float32)
        await client.send_audio_chunk(audio.tobytes())
        await asyncio.sleep(1)

    # Disconnect
    await client.disconnect()


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    asyncio.run(example_usage())
