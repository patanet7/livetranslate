#!/usr/bin/env python3
"""
Fireflies Mock Server

Simulates Fireflies.ai GraphQL API and Socket.IO Realtime API for testing.
Provides configurable scenarios for integration testing without real API calls.

Uses Socket.IO protocol (matching the real Fireflies API), NOT raw WebSocket.

Usage:
    # Start the mock server
    server = FirefliesMockServer(host="localhost", port=8080)
    await server.start()

    # Configure test scenarios
    server.add_scenario(MockTranscriptScenario(...))

    # Connect client to mock server (Socket.IO)
    client = FirefliesRealtimeClient(
        api_key="test-key",
        transcript_id="test-transcript",
        endpoint="http://localhost:8080",
        socketio_path="/ws/realtime",
    )

    # Clean up
    await server.stop()
"""

import asyncio
import contextlib
import json
import logging
import uuid
from dataclasses import dataclass, field
from datetime import UTC, datetime
from typing import Any

import socketio
from aiohttp import web

logger = logging.getLogger(__name__)


# =============================================================================
# Data Classes
# =============================================================================


@dataclass
class MockMeeting:
    """Mock meeting data for GraphQL responses."""

    id: str = field(default_factory=lambda: f"meeting_{uuid.uuid4().hex[:8]}")
    title: str = "Test Meeting"
    organizer_email: str = "test@example.com"
    meeting_link: str | None = "https://meet.example.com/test"
    start_time: datetime | None = field(default_factory=lambda: datetime.now(UTC))
    end_time: datetime | None = None
    privacy: str = "private"
    state: str = "active"

    def to_graphql_dict(self) -> dict[str, Any]:
        """Convert to GraphQL response format."""
        return {
            "id": self.id,
            "title": self.title,
            "organizer_email": self.organizer_email,
            "meeting_link": self.meeting_link,
            "start_time": self.start_time.isoformat() + "Z" if self.start_time else None,
            "end_time": self.end_time.isoformat() + "Z" if self.end_time else None,
            "privacy": self.privacy,
            "state": self.state,
        }


@dataclass
class MockChunk:
    """Mock transcript chunk data.

    Matches the REAL Fireflies Realtime API payload which sends exactly 5 fields:
      chunk_id, text, speaker_name, start_time, end_time

    Real behavior discovered from captured stream data (/tmp/orchestration.log):
    - chunk_id is an integer (e.g. 65174), converted to str by our client
    - Same chunk_id gets progressive text updates (word-by-word ASR streaming)
    - Text is corrected mid-stream as ASR refines ("key cat" -> "kitty cat")
    - No confidence or is_final fields exist in the real API
    """

    chunk_id: str = field(default_factory=lambda: str(uuid.uuid4().int % 100000))
    text: str = ""
    speaker_name: str = "Speaker"
    start_time: float = 0.0
    end_time: float = 0.0

    def to_websocket_dict(self, transcript_id: str) -> dict[str, Any]:
        """Convert to raw WebSocket message format (legacy, for backward compat)."""
        return {
            "type": "transcription.broadcast",
            "data": {
                "chunk_id": self.chunk_id,
                "text": self.text,
                "speaker_name": self.speaker_name,
                "start_time": self.start_time,
                "end_time": self.end_time,
            },
        }

    def to_socketio_dict(self, transcript_id: str) -> dict[str, Any]:
        """Convert to Socket.IO event data (no 'type' wrapper — event name is separate).

        Note: Real Fireflies does NOT include transcript_id in event data.
        It's established during Socket.IO auth handshake. We omit it here
        to match real behavior. The client uses self.transcript_id instead.
        """
        return {
            "chunk_id": self.chunk_id,
            "text": self.text,
            "speaker_name": self.speaker_name,
            "start_time": self.start_time,
            "end_time": self.end_time,
        }


@dataclass
class MockTranscriptScenario:
    """
    Defines a transcript scenario for testing.

    Models real Fireflies streaming behavior discovered from captured data:
    - Same chunk_id sent repeatedly with progressively longer text
    - ASR text corrected mid-stream (words change as model refines)
    - Multiple chunk_ids can be active simultaneously (interleaving)
    - New chunk_id signals the previous one is "done" (finalization trigger)
    - Typical ~200-500ms between word updates within a chunk
    """

    transcript_id: str = field(default_factory=lambda: f"transcript_{uuid.uuid4().hex[:8]}")
    meeting: MockMeeting = field(default_factory=MockMeeting)
    chunks: list[MockChunk] = field(default_factory=list)

    # Timing configuration
    chunk_delay_ms: float = 500.0  # Delay between chunks (when new chunk_id starts)
    word_delay_ms: float = 300.0  # Delay per word update (same chunk_id)
    stream_mode: str = "chunks"  # "chunks" (final text only) or "words" (word-by-word)

    # Error simulation
    simulate_disconnect_after_chunks: int | None = None
    simulate_error_message: str | None = None

    @classmethod
    def conversation_scenario(
        cls,
        speakers: list[str] | None = None,
        num_exchanges: int = 10,
        chunk_delay_ms: float = 500.0,
        word_stream: bool = True,
    ) -> "MockTranscriptScenario":
        """
        Create a realistic conversation scenario.

        When word_stream=True (default), each sentence is streamed word-by-word
        using the same chunk_id — matching real Fireflies behavior.

        When word_stream=False, each sentence is emitted as a single chunk
        (useful for faster tests that don't need to test dedup logic).

        Args:
            speakers: List of speaker names
            num_exchanges: Number of conversational exchanges
            chunk_delay_ms: Delay between chunks
            word_stream: Whether to stream word-by-word within each chunk

        Returns:
            Configured MockTranscriptScenario
        """
        if speakers is None:
            speakers = ["Alice", "Bob", "Charlie"]

        conversation_templates = [
            "Hello everyone, thank you for joining.",
            "Good morning! Happy to be here.",
            "Let's get started with the agenda.",
            "I have a question about the budget.",
            "That's a great point.",
            "Could you elaborate on that?",
            "I think we should consider the alternatives.",
            "The deadline is next Friday.",
            "Does anyone have concerns?",
            "I agree with that approach.",
            "Let me share my screen.",
            "Can everyone see this?",
            "The numbers look promising.",
            "We need to address the bottleneck.",
            "What are the next steps?",
            "I'll follow up on that.",
            "Any other questions?",
            "Thanks for the update.",
            "Let's schedule a follow-up.",
            "Great meeting everyone!",
        ]

        chunks = []
        current_time = 0.0
        base_chunk_id = 65000  # Match real Fireflies integer chunk_id range

        for i in range(num_exchanges):
            speaker = speakers[i % len(speakers)]
            text = conversation_templates[i % len(conversation_templates)]
            chunk_id = str(base_chunk_id + i)

            if word_stream:
                # Stream word-by-word with same chunk_id (real behavior)
                words = text.split()
                accumulated = ""
                for w_idx, word in enumerate(words):
                    accumulated = (accumulated + " " + word).strip()
                    word_count = w_idx + 1
                    end_time = current_time + word_count * 0.3

                    chunk = MockChunk(
                        chunk_id=chunk_id,
                        text=accumulated,
                        speaker_name=speaker,
                        start_time=current_time,
                        end_time=end_time,
                    )
                    chunks.append(chunk)
            else:
                # Single chunk per sentence (faster for simple tests)
                word_count = len(text.split())
                duration = word_count * 0.3

                chunk = MockChunk(
                    chunk_id=chunk_id,
                    text=text,
                    speaker_name=speaker,
                    start_time=current_time,
                    end_time=current_time + duration,
                )
                chunks.append(chunk)

            current_time += len(text.split()) * 0.3 + 0.2  # Gap between speakers

        return cls(
            chunks=chunks,
            chunk_delay_ms=chunk_delay_ms,
            stream_mode="words" if word_stream else "chunks",
        )

    @classmethod
    def word_by_word_scenario(
        cls,
        text: str,
        speaker: str = "Speaker",
        word_delay_ms: float = 300.0,
        chunk_id: str | None = None,
    ) -> "MockTranscriptScenario":
        """
        Create a word-by-word streaming scenario for a single utterance.

        Matches real Fireflies behavior: SAME chunk_id with progressively
        longer text as ASR processes speech word-by-word.

        Example from real Fireflies stream (chunk_id=65174):
          "Well."
          "Well, let's"
          "Well, let's check."
          "Well, let's check and see."
          ... (up to 18 updates per chunk_id)

        Args:
            text: Full sentence to stream word-by-word
            speaker: Speaker name
            word_delay_ms: Delay between word updates (~300ms is realistic)
            chunk_id: Explicit chunk_id (auto-generated if None)
        """
        if chunk_id is None:
            chunk_id = str(65000 + uuid.uuid4().int % 1000)

        words = text.split()
        chunks = []
        word_duration = word_delay_ms / 1000.0

        accumulated_text = ""
        for word in words:
            accumulated_text = (accumulated_text + " " + word).strip()

            chunk = MockChunk(
                chunk_id=chunk_id,
                text=accumulated_text,
                speaker_name=speaker,
                start_time=0.0,
                end_time=len(accumulated_text.split()) * word_duration,
            )
            chunks.append(chunk)

        return cls(
            chunks=chunks,
            word_delay_ms=word_delay_ms,
            stream_mode="words",
        )

    @classmethod
    def captured_realtime_scenario(
        cls,
        word_delay_ms: float = 300.0,
    ) -> "MockTranscriptScenario":
        """
        Scenario built from REAL captured Fireflies stream data.

        Extracted from /tmp/orchestration.log (2026-02-20T23:13:00Z).
        This is actual Fireflies Realtime API behavior, not synthetic.

        Key behaviors this models:
        - Word-by-word streaming with same chunk_id
        - ASR text corrections mid-stream ("key cat" -> "kitty cat")
        - Multiple chunk_ids active simultaneously (interleaving)
        - ~16 interim updates per chunk before finalization
        """
        chunks = []

        # chunk_id=65174: "Well, let's check and see what's going on..."
        # Real progression from captured data (18 updates)
        chunk_65174_progression = [
            "Well.",
            "Well, let's",
            "Well, let's check.",
            "Well, let's check and",
            "Well, let's check and see.",
            "well, let's check and see what's",
            "Well, let's check and see what's going.",
            "Well, let's check and see what's going on.",
            "Well, let's check and see what's going on. What?",
            "Well, let's check and see what's going on. What is",
            "Well, let's check and see what's going on. What is going?",
            "Well, let's check and see what's going on. What is going on?",
            "Well, let's check and see what's going on. What is going on with",
            "Well, let's check and see what's going on. What is going on with my",
            "Well, let's check and see what's going on. What is going on with my key.",
            "Well, let's check and see what's going on. What is going on with my key cat?",
            "Well, let's check and see what's going on. What is going on with my key? Okay.",
            "Well, let's check and see what's going on. What is going on with my kitty cat,",
        ]

        for text in chunk_65174_progression:
            chunks.append(
                MockChunk(
                    chunk_id="65174",
                    text=text,
                    speaker_name="Speaker",
                    start_time=0.0,
                    end_time=len(text.split()) * 0.3,
                )
            )

        # chunk_id=65175: starts while 65174 is still updating (interleaving!)
        # Real progression: "but" -> "What is" -> "But is" -> ...
        chunk_65175_progression = [
            "but",
            "What is",
            "But is",
            "What is going?",
            "What is going on?",
            "What is going on? We",
            "What is going on with my",
            "What is going on with my key?",
            "What is going on with my key to",
            "What is going on with my key to kill?",
            "What is going on with my key to get",
            "What is going on with my key to kill it?",
            "What is going on with my key to get it?",
            "What is going on with my key to kill? It should",
            "What is going on with my key to kill? It should be",
            "What is going on with my key to kill? It should be working",
            "What is going on with my key to kill, it should be working in",
            "What is going on with my key to kill? It should be working now.",
        ]

        for text in chunk_65175_progression:
            chunks.append(
                MockChunk(
                    chunk_id="65175",
                    text=text,
                    speaker_name="Speaker",
                    start_time=6.0,
                    end_time=6.0 + len(text.split()) * 0.3,
                )
            )

        # chunk_id=65176: starts while 65175 still active
        chunk_65176_progression = [
            "here.",
            "here working.",
            "here working now",
            "here, working now, working",
            "here, working now working again.",
        ]

        for text in chunk_65176_progression:
            chunks.append(
                MockChunk(
                    chunk_id="65176",
                    text=text,
                    speaker_name="Speaker",
                    start_time=14.0,
                    end_time=14.0 + len(text.split()) * 0.3,
                )
            )

        # chunk_id=65177: short utterance
        chunk_65177_progression = [
            "let's,",
            "Let's go.",
        ]

        for text in chunk_65177_progression:
            chunks.append(
                MockChunk(
                    chunk_id="65177",
                    text=text,
                    speaker_name="Speaker",
                    start_time=23.0,
                    end_time=23.0 + len(text.split()) * 0.3,
                )
            )

        return cls(
            chunks=chunks,
            word_delay_ms=word_delay_ms,
            stream_mode="words",
        )


# =============================================================================
# Mock Server (Socket.IO)
# =============================================================================


class FirefliesMockServer:
    """
    Mock server for Fireflies.ai API testing.

    Uses Socket.IO protocol (matching the real Fireflies API):
    - Socket.IO endpoint at /ws/realtime for realtime transcript streaming
    - GraphQL endpoint at /graphql for active_meetings queries
    - Health check at /health

    Authentication: Token and transcriptId are passed in the Socket.IO
    auth payload (matching fireflies_client.py:522-525):
        auth = {
            "token": "Bearer <api_key>",
            "transcriptId": "<transcript_id>",
        }
    """

    def __init__(
        self,
        host: str = "localhost",
        port: int = 8080,
        valid_api_keys: set[str] | None = None,
    ):
        self.host = host
        self.port = port
        self.valid_api_keys = valid_api_keys or {"test-api-key", "ff-test-key"}

        # Scenarios by transcript ID
        self._scenarios: dict[str, MockTranscriptScenario] = {}

        # Active meetings (for GraphQL queries)
        self._meetings: list[MockMeeting] = []

        # Connected Socket.IO clients: sid -> transcript_id
        self._sid_to_transcript: dict[str, str] = {}

        # Server state
        self._app: web.Application | None = None
        self._runner: web.AppRunner | None = None
        self._site: web.TCPSite | None = None
        self._sio: socketio.AsyncServer | None = None

        # Statistics
        self._stats = {
            "graphql_requests": 0,
            "websocket_connections": 0,
            "chunks_sent": 0,
            "errors": 0,
        }

        # Background task tracking (prevents fire-and-forget)
        self._background_tasks: set[asyncio.Task] = set()

    @property
    def graphql_url(self) -> str:
        """GraphQL endpoint URL."""
        return f"http://{self.host}:{self.port}/graphql"

    @property
    def websocket_url(self) -> str:
        """Socket.IO endpoint URL (use http:// for Socket.IO clients)."""
        return f"http://{self.host}:{self.port}"

    @property
    def stats(self) -> dict[str, int]:
        """Server statistics."""
        return self._stats.copy()

    # =========================================================================
    # Scenario Management
    # =========================================================================

    def add_scenario(self, scenario: MockTranscriptScenario) -> str:
        """Add a test scenario. Returns transcript ID."""
        self._scenarios[scenario.transcript_id] = scenario
        self._meetings.append(scenario.meeting)
        logger.info(f"Added scenario: {scenario.transcript_id}")
        return scenario.transcript_id

    def remove_scenario(self, transcript_id: str) -> bool:
        """Remove a scenario by transcript ID."""
        if transcript_id in self._scenarios:
            scenario = self._scenarios.pop(transcript_id)
            self._meetings = [m for m in self._meetings if m.id != scenario.meeting.id]
            return True
        return False

    def clear_scenarios(self):
        """Remove all scenarios."""
        self._scenarios.clear()
        self._meetings.clear()

    def get_scenario(self, transcript_id: str) -> MockTranscriptScenario | None:
        """Get scenario by transcript ID."""
        return self._scenarios.get(transcript_id)

    # =========================================================================
    # Server Lifecycle
    # =========================================================================

    async def start(self):
        """Start the mock server with Socket.IO."""
        # Create Socket.IO server (async mode = aiohttp)
        self._sio = socketio.AsyncServer(
            async_mode="aiohttp",
            cors_allowed_origins="*",
            logger=False,
            engineio_logger=False,
        )

        # Register Socket.IO event handlers
        self._register_socketio_handlers()

        # Create aiohttp app
        self._app = web.Application()

        # Attach Socket.IO to aiohttp app at /ws/realtime path
        self._sio.attach(self._app, socketio_path="/ws/realtime")

        # Add HTTP routes (GraphQL + health)
        self._app.router.add_post("/graphql", self._handle_graphql)
        self._app.router.add_get("/health", self._handle_health)

        self._runner = web.AppRunner(self._app)
        await self._runner.setup()

        self._site = web.TCPSite(self._runner, self.host, self.port)
        await self._site.start()

        logger.info(f"Fireflies mock server started at {self.host}:{self.port}")
        logger.info(f"  GraphQL: {self.graphql_url}")
        logger.info(f"  Socket.IO: {self.websocket_url} (path=/ws/realtime)")

    async def stop(self):
        """Stop the mock server."""
        # Cancel background tasks
        for task in list(self._background_tasks):
            task.cancel()
            with contextlib.suppress(asyncio.CancelledError):
                await task
        self._background_tasks.clear()

        # Disconnect all Socket.IO clients
        if self._sio:
            for sid in list(self._sid_to_transcript.keys()):
                with contextlib.suppress(Exception):
                    await self._sio.disconnect(sid)
        self._sid_to_transcript.clear()

        # Stop the server
        if self._site:
            await self._site.stop()
        if self._runner:
            await self._runner.cleanup()

        self._site = None
        self._runner = None
        self._app = None
        self._sio = None

        logger.info("Fireflies mock server stopped")

    # =========================================================================
    # Socket.IO Event Handlers
    # =========================================================================

    def _register_socketio_handlers(self):
        """Register Socket.IO event handlers on the server."""
        sio = self._sio

        @sio.event
        async def connect(sid, environ, auth):
            """Handle Socket.IO connection with auth payload."""
            self._stats["websocket_connections"] += 1

            # Auth payload from FirefliesRealtimeClient:
            #   {"token": "Bearer <api_key>", "transcriptId": "<transcript_id>"}
            if not auth or not isinstance(auth, dict):
                logger.warning(f"Socket.IO connect: no auth payload from {sid}")
                await sio.emit("auth.failed", {"message": "Missing auth payload"}, to=sid)
                await sio.disconnect(sid)
                return False

            token = auth.get("token", "")
            transcript_id = auth.get("transcriptId", "")

            # Validate API key
            if not self._validate_api_key(token):
                logger.warning(f"Socket.IO auth failed for {sid}: invalid token")
                await sio.emit("auth.failed", {"message": "Invalid API key"}, to=sid)
                await sio.disconnect(sid)
                return False

            # Validate transcript_id
            if not transcript_id:
                logger.warning(f"Socket.IO auth failed for {sid}: missing transcriptId")
                await sio.emit("auth.failed", {"message": "Missing transcriptId"}, to=sid)
                await sio.disconnect(sid)
                return False

            # Auth success
            self._sid_to_transcript[sid] = transcript_id
            logger.info(f"Socket.IO client {sid} authenticated for transcript: {transcript_id}")

            # Emit auth success + connection established
            await sio.emit("auth.success", {}, to=sid)
            await sio.emit("connection.established", {}, to=sid)

            # Start streaming scenario if available
            scenario = self._scenarios.get(transcript_id)
            if scenario:
                task = asyncio.create_task(self._stream_scenario(sid, scenario))
                self._background_tasks.add(task)
                task.add_done_callback(self._background_tasks.discard)

            return True

        @sio.event
        async def disconnect(sid):
            """Handle Socket.IO disconnection."""
            transcript_id = self._sid_to_transcript.pop(sid, None)
            logger.info(
                f"Socket.IO client {sid} disconnected"
                + (f" (transcript: {transcript_id})" if transcript_id else "")
            )

        @sio.on("ping")
        async def on_ping(sid, data=None):
            """Handle ping/pong for keepalive."""
            await sio.emit("pong", {}, to=sid)

    # =========================================================================
    # Scenario Streaming
    # =========================================================================

    async def _stream_scenario(
        self,
        sid: str,
        scenario: MockTranscriptScenario,
    ):
        """Stream a scenario's chunks to a Socket.IO client."""
        sio = self._sio
        try:
            for i, chunk in enumerate(scenario.chunks):
                # Check if client still connected
                if sid not in self._sid_to_transcript:
                    break

                # Check for disconnect simulation
                if (
                    scenario.simulate_disconnect_after_chunks
                    and i >= scenario.simulate_disconnect_after_chunks
                ):
                    await sio.disconnect(sid)
                    break

                # Emit as Socket.IO event (event name = "transcription.broadcast")
                data = chunk.to_socketio_dict(scenario.transcript_id)
                await sio.emit("transcription.broadcast", data, to=sid)
                self._stats["chunks_sent"] += 1

                # Delay based on stream mode
                if scenario.stream_mode == "words":
                    await asyncio.sleep(scenario.word_delay_ms / 1000.0)
                else:
                    await asyncio.sleep(scenario.chunk_delay_ms / 1000.0)

            # Send error if configured
            if scenario.simulate_error_message and sid in self._sid_to_transcript:
                await sio.emit(
                    "connection.error",
                    {"message": scenario.simulate_error_message},
                    to=sid,
                )

        except Exception as e:
            logger.error(f"Error streaming scenario to {sid}: {e}")
            self._stats["errors"] += 1

    # =========================================================================
    # HTTP Request Handlers
    # =========================================================================

    async def _handle_health(self, request: web.Request) -> web.Response:
        """Health check endpoint."""
        return web.json_response(
            {
                "status": "ok",
                "scenarios": len(self._scenarios),
                "active_connections": len(self._sid_to_transcript),
            }
        )

    async def _handle_graphql(self, request: web.Request) -> web.Response:
        """Handle GraphQL requests."""
        self._stats["graphql_requests"] += 1

        # Validate API key
        auth_header = request.headers.get("Authorization", "")
        if not self._validate_api_key(auth_header):
            return web.json_response(
                {"errors": [{"message": "Invalid API key"}]},
                status=401,
            )

        try:
            body = await request.json()
            query = body.get("query", "")
            variables = body.get("variables", {})

            # Handle active_meetings query
            if "active_meetings" in query:
                return await self._handle_active_meetings_query(variables)

            # Unknown query
            return web.json_response(
                {"errors": [{"message": "Unknown query"}]},
                status=400,
            )

        except json.JSONDecodeError:
            self._stats["errors"] += 1
            return web.json_response(
                {"errors": [{"message": "Invalid JSON"}]},
                status=400,
            )

    async def _handle_active_meetings_query(
        self,
        variables: dict[str, Any],
    ) -> web.Response:
        """Handle active_meetings GraphQL query."""
        email_filter = variables.get("email")
        states_filter = variables.get("states", ["active", "paused"])

        # Filter meetings
        filtered_meetings = []
        for meeting in self._meetings:
            if email_filter and meeting.organizer_email != email_filter:
                continue
            if meeting.state not in states_filter:
                continue
            filtered_meetings.append(meeting.to_graphql_dict())

        return web.json_response(
            {
                "data": {
                    "active_meetings": filtered_meetings,
                }
            }
        )

    # =========================================================================
    # Utility Methods
    # =========================================================================

    def _validate_api_key(self, auth_value: str) -> bool:
        """Validate API key from Authorization header or Socket.IO auth token."""
        if not auth_value:
            return False
        # Strip "Bearer " prefix if present
        if auth_value.startswith("Bearer "):
            api_key = auth_value[7:]
        else:
            api_key = auth_value
        return api_key in self.valid_api_keys

    async def broadcast_to_transcript(
        self,
        transcript_id: str,
        event_name: str,
        data: dict[str, Any],
    ):
        """Broadcast a Socket.IO event to all clients connected to a transcript."""
        if not self._sio:
            return
        for sid, tid in list(self._sid_to_transcript.items()):
            if tid == transcript_id:
                await self._sio.emit(event_name, data, to=sid)

    async def inject_chunk(
        self,
        transcript_id: str,
        text: str,
        speaker_name: str = "Speaker",
    ):
        """Inject a chunk into an active session (for testing)."""
        chunk = MockChunk(
            text=text,
            speaker_name=speaker_name,
            start_time=0.0,
            end_time=len(text.split()) * 0.3,
        )

        data = chunk.to_socketio_dict(transcript_id)
        await self.broadcast_to_transcript(transcript_id, "transcription.broadcast", data)
        self._stats["chunks_sent"] += 1


# =============================================================================
# Pytest Fixtures
# =============================================================================


async def create_mock_server(
    host: str = "localhost",
    port: int = 8080,
    scenarios: list[MockTranscriptScenario] | None = None,
) -> FirefliesMockServer:
    """
    Create and start a mock server for testing.

    Args:
        host: Server host
        port: Server port
        scenarios: Initial scenarios to add

    Returns:
        Started mock server
    """
    server = FirefliesMockServer(host=host, port=port)

    if scenarios:
        for scenario in scenarios:
            server.add_scenario(scenario)

    await server.start()
    return server


# Example usage
if __name__ == "__main__":

    async def main():
        # Create server with conversation scenario
        server = FirefliesMockServer()

        # Add a conversation scenario
        scenario = MockTranscriptScenario.conversation_scenario(
            speakers=["Alice", "Bob"],
            num_exchanges=5,
            chunk_delay_ms=1000,
        )
        server.add_scenario(scenario)

        print("Starting mock server...")
        await server.start()

        print(f"GraphQL URL: {server.graphql_url}")
        print(f"Socket.IO URL: {server.websocket_url}")
        print(f"Transcript ID: {scenario.transcript_id}")
        print("\nPress Ctrl+C to stop...")

        try:
            while True:
                await asyncio.sleep(1)
        except KeyboardInterrupt:
            print("\nStopping...")
        finally:
            await server.stop()

    asyncio.run(main())
