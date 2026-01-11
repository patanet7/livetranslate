#!/usr/bin/env python3
"""
Fireflies Mock Server

Simulates Fireflies.ai GraphQL API and WebSocket Realtime API for testing.
Provides configurable scenarios for integration testing without real API calls.

Usage:
    # Start the mock server
    server = FirefliesMockServer(host="localhost", port=8080)
    await server.start()

    # Configure test scenarios
    server.add_scenario(MockTranscriptScenario(...))

    # Connect client to mock server
    client = FirefliesRealtimeClient(
        api_key="test-key",
        transcript_id="test-transcript",
        endpoint=f"ws://localhost:8080/realtime"
    )

    # Clean up
    await server.stop()
"""

import logging
import asyncio
import json
import uuid
from datetime import datetime, timezone
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Any, Callable, Set
from aiohttp import web, WSMsgType

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
    meeting_link: Optional[str] = "https://meet.example.com/test"
    start_time: Optional[datetime] = field(default_factory=lambda: datetime.now(timezone.utc))
    end_time: Optional[datetime] = None
    privacy: str = "private"
    state: str = "active"

    def to_graphql_dict(self) -> Dict[str, Any]:
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
    """Mock transcript chunk data."""

    chunk_id: str = field(default_factory=lambda: f"chunk_{uuid.uuid4().hex[:8]}")
    text: str = ""
    speaker_name: str = "Speaker"
    start_time: float = 0.0
    end_time: float = 0.0
    confidence: float = 0.95
    is_final: bool = True

    def to_websocket_dict(self, transcript_id: str) -> Dict[str, Any]:
        """Convert to WebSocket message format."""
        return {
            "type": "transcription.broadcast",
            "data": {
                "transcript_id": transcript_id,
                "chunk_id": self.chunk_id,
                "text": self.text,
                "speaker_name": self.speaker_name,
                "start_time": self.start_time,
                "end_time": self.end_time,
                "confidence": self.confidence,
                "is_final": self.is_final,
            }
        }


@dataclass
class MockTranscriptScenario:
    """
    Defines a transcript scenario for testing.

    Includes chunks to be streamed with timing information.
    """

    transcript_id: str = field(default_factory=lambda: f"transcript_{uuid.uuid4().hex[:8]}")
    meeting: MockMeeting = field(default_factory=MockMeeting)
    chunks: List[MockChunk] = field(default_factory=list)

    # Timing configuration
    chunk_delay_ms: float = 500.0  # Delay between chunks
    word_delay_ms: float = 100.0   # Delay per word (for word-by-word streaming)
    stream_mode: str = "chunks"     # "chunks" or "words"

    # Error simulation
    simulate_disconnect_after_chunks: Optional[int] = None
    simulate_error_message: Optional[str] = None

    @classmethod
    def conversation_scenario(
        cls,
        speakers: List[str] = None,
        num_exchanges: int = 10,
        chunk_delay_ms: float = 500.0,
    ) -> "MockTranscriptScenario":
        """
        Create a realistic conversation scenario.

        Args:
            speakers: List of speaker names
            num_exchanges: Number of conversational exchanges
            chunk_delay_ms: Delay between chunks

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

        for i in range(num_exchanges):
            speaker = speakers[i % len(speakers)]
            text = conversation_templates[i % len(conversation_templates)]

            # Estimate duration based on word count
            word_count = len(text.split())
            duration = word_count * 0.3  # ~300ms per word

            chunk = MockChunk(
                chunk_id=f"chunk_{i+1:04d}",
                text=text,
                speaker_name=speaker,
                start_time=current_time,
                end_time=current_time + duration,
            )
            chunks.append(chunk)

            current_time += duration + 0.2  # Small gap between speakers

        return cls(
            chunks=chunks,
            chunk_delay_ms=chunk_delay_ms,
        )

    @classmethod
    def word_by_word_scenario(
        cls,
        text: str,
        speaker: str = "Speaker",
        word_delay_ms: float = 150.0,
    ) -> "MockTranscriptScenario":
        """
        Create a word-by-word streaming scenario.

        Simulates real-time transcription where words arrive incrementally.
        """
        words = text.split()
        chunks = []
        current_time = 0.0
        word_duration = word_delay_ms / 1000.0

        accumulated_text = ""
        for i, word in enumerate(words):
            accumulated_text = (accumulated_text + " " + word).strip()
            is_final = (i == len(words) - 1)

            chunk = MockChunk(
                chunk_id=f"chunk_{i+1:04d}",
                text=accumulated_text,
                speaker_name=speaker,
                start_time=0.0,
                end_time=current_time + word_duration,
                is_final=is_final,
            )
            chunks.append(chunk)
            current_time += word_duration

        return cls(
            chunks=chunks,
            word_delay_ms=word_delay_ms,
            stream_mode="words",
        )


# =============================================================================
# Mock Server
# =============================================================================


class FirefliesMockServer:
    """
    Mock server for Fireflies.ai API testing.

    Provides:
    - GraphQL endpoint for active_meetings queries
    - WebSocket endpoint for realtime transcript streaming
    - Configurable scenarios for different test cases
    """

    def __init__(
        self,
        host: str = "localhost",
        port: int = 8080,
        valid_api_keys: Optional[Set[str]] = None,
    ):
        """
        Initialize the mock server.

        Args:
            host: Server host
            port: Server port
            valid_api_keys: Set of valid API keys (default: accepts any)
        """
        self.host = host
        self.port = port
        self.valid_api_keys = valid_api_keys or {"test-api-key", "ff-test-key"}

        # Scenarios by transcript ID
        self._scenarios: Dict[str, MockTranscriptScenario] = {}

        # Active meetings (for GraphQL queries)
        self._meetings: List[MockMeeting] = []

        # Connected WebSocket clients by transcript ID
        self._ws_clients: Dict[str, Set[web.WebSocketResponse]] = {}

        # Server state
        self._app: Optional[web.Application] = None
        self._runner: Optional[web.AppRunner] = None
        self._site: Optional[web.TCPSite] = None

        # Statistics
        self._stats = {
            "graphql_requests": 0,
            "websocket_connections": 0,
            "chunks_sent": 0,
            "errors": 0,
        }

    @property
    def graphql_url(self) -> str:
        """GraphQL endpoint URL."""
        return f"http://{self.host}:{self.port}/graphql"

    @property
    def websocket_url(self) -> str:
        """WebSocket endpoint URL."""
        return f"ws://{self.host}:{self.port}/realtime"

    @property
    def stats(self) -> Dict[str, int]:
        """Server statistics."""
        return self._stats.copy()

    # =========================================================================
    # Scenario Management
    # =========================================================================

    def add_scenario(self, scenario: MockTranscriptScenario) -> str:
        """
        Add a test scenario.

        Args:
            scenario: The scenario to add

        Returns:
            Transcript ID for the scenario
        """
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

    def get_scenario(self, transcript_id: str) -> Optional[MockTranscriptScenario]:
        """Get scenario by transcript ID."""
        return self._scenarios.get(transcript_id)

    # =========================================================================
    # Server Lifecycle
    # =========================================================================

    async def start(self):
        """Start the mock server."""
        self._app = web.Application()
        self._app.router.add_post("/graphql", self._handle_graphql)
        self._app.router.add_get("/realtime", self._handle_websocket)
        self._app.router.add_get("/health", self._handle_health)

        self._runner = web.AppRunner(self._app)
        await self._runner.setup()

        self._site = web.TCPSite(self._runner, self.host, self.port)
        await self._site.start()

        logger.info(f"Fireflies mock server started at {self.host}:{self.port}")
        logger.info(f"  GraphQL: {self.graphql_url}")
        logger.info(f"  WebSocket: {self.websocket_url}")

    async def stop(self):
        """Stop the mock server."""
        # Close all WebSocket connections
        for transcript_id, clients in self._ws_clients.items():
            for ws in clients:
                await ws.close()
        self._ws_clients.clear()

        # Stop the server
        if self._site:
            await self._site.stop()
        if self._runner:
            await self._runner.cleanup()

        self._site = None
        self._runner = None
        self._app = None

        logger.info("Fireflies mock server stopped")

    # =========================================================================
    # Request Handlers
    # =========================================================================

    async def _handle_health(self, request: web.Request) -> web.Response:
        """Health check endpoint."""
        return web.json_response({
            "status": "ok",
            "scenarios": len(self._scenarios),
            "active_connections": sum(len(c) for c in self._ws_clients.values()),
        })

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
        variables: Dict[str, Any],
    ) -> web.Response:
        """Handle active_meetings GraphQL query."""
        email_filter = variables.get("email")
        states_filter = variables.get("states", ["active", "paused"])

        # Filter meetings
        filtered_meetings = []
        for meeting in self._meetings:
            # Apply email filter
            if email_filter and meeting.organizer_email != email_filter:
                continue

            # Apply state filter
            if meeting.state not in states_filter:
                continue

            filtered_meetings.append(meeting.to_graphql_dict())

        return web.json_response({
            "data": {
                "active_meetings": filtered_meetings,
            }
        })

    async def _handle_websocket(self, request: web.Request) -> web.WebSocketResponse:
        """Handle WebSocket connections."""
        ws = web.WebSocketResponse()
        await ws.prepare(request)

        self._stats["websocket_connections"] += 1
        transcript_id = None
        authenticated = False

        try:
            async for msg in ws:
                if msg.type == WSMsgType.TEXT:
                    try:
                        data = json.loads(msg.data)
                        msg_type = data.get("type", "")

                        # Handle authentication
                        if msg_type == "auth":
                            api_key = data.get("api_key", "")
                            transcript_id = data.get("transcript_id", "")

                            if not self._validate_api_key(f"Bearer {api_key}"):
                                await ws.send_json({
                                    "type": "auth.failed",
                                    "message": "Invalid API key",
                                })
                                break

                            authenticated = True

                            # Register client
                            if transcript_id not in self._ws_clients:
                                self._ws_clients[transcript_id] = set()
                            self._ws_clients[transcript_id].add(ws)

                            await ws.send_json({"type": "auth.success"})
                            await ws.send_json({"type": "connection.established"})

                            # Start streaming scenario if available
                            scenario = self._scenarios.get(transcript_id)
                            if scenario:
                                asyncio.create_task(
                                    self._stream_scenario(ws, scenario)
                                )

                        # Handle ping/pong
                        elif msg_type == "ping":
                            await ws.send_json({"type": "pong"})

                    except json.JSONDecodeError:
                        logger.warning("Received invalid JSON on WebSocket")

                elif msg.type == WSMsgType.ERROR:
                    logger.error(f"WebSocket error: {ws.exception()}")
                    self._stats["errors"] += 1

        finally:
            # Cleanup
            if transcript_id and transcript_id in self._ws_clients:
                self._ws_clients[transcript_id].discard(ws)
                if not self._ws_clients[transcript_id]:
                    del self._ws_clients[transcript_id]

        return ws

    async def _stream_scenario(
        self,
        ws: web.WebSocketResponse,
        scenario: MockTranscriptScenario,
    ):
        """Stream a scenario's chunks to a WebSocket client."""
        try:
            for i, chunk in enumerate(scenario.chunks):
                if ws.closed:
                    break

                # Check for disconnect simulation
                if (
                    scenario.simulate_disconnect_after_chunks
                    and i >= scenario.simulate_disconnect_after_chunks
                ):
                    await ws.close()
                    break

                # Send chunk
                message = chunk.to_websocket_dict(scenario.transcript_id)
                await ws.send_json(message)
                self._stats["chunks_sent"] += 1

                # Delay based on stream mode
                if scenario.stream_mode == "words":
                    await asyncio.sleep(scenario.word_delay_ms / 1000.0)
                else:
                    await asyncio.sleep(scenario.chunk_delay_ms / 1000.0)

            # Send error if configured
            if scenario.simulate_error_message:
                await ws.send_json({
                    "type": "error",
                    "message": scenario.simulate_error_message,
                })

        except Exception as e:
            logger.error(f"Error streaming scenario: {e}")
            self._stats["errors"] += 1

    # =========================================================================
    # Utility Methods
    # =========================================================================

    def _validate_api_key(self, auth_header: str) -> bool:
        """Validate API key from Authorization header."""
        if not auth_header.startswith("Bearer "):
            return False

        api_key = auth_header[7:]  # Remove "Bearer " prefix
        return api_key in self.valid_api_keys

    async def broadcast_to_transcript(
        self,
        transcript_id: str,
        message: Dict[str, Any],
    ):
        """Broadcast a message to all clients connected to a transcript."""
        clients = self._ws_clients.get(transcript_id, set())
        for ws in clients:
            if not ws.closed:
                await ws.send_json(message)

    async def inject_chunk(
        self,
        transcript_id: str,
        text: str,
        speaker_name: str = "Speaker",
    ):
        """
        Inject a chunk into an active session (for testing).

        Args:
            transcript_id: Transcript to inject into
            text: Chunk text
            speaker_name: Speaker name
        """
        chunk = MockChunk(
            text=text,
            speaker_name=speaker_name,
            start_time=0.0,
            end_time=len(text.split()) * 0.3,
        )

        message = chunk.to_websocket_dict(transcript_id)
        await self.broadcast_to_transcript(transcript_id, message)
        self._stats["chunks_sent"] += 1


# =============================================================================
# Pytest Fixtures
# =============================================================================


async def create_mock_server(
    host: str = "localhost",
    port: int = 8080,
    scenarios: Optional[List[MockTranscriptScenario]] = None,
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

        print(f"Starting mock server...")
        await server.start()

        print(f"GraphQL URL: {server.graphql_url}")
        print(f"WebSocket URL: {server.websocket_url}")
        print(f"Transcript ID: {scenario.transcript_id}")
        print(f"\nPress Ctrl+C to stop...")

        try:
            while True:
                await asyncio.sleep(1)
        except KeyboardInterrupt:
            print("\nStopping...")
        finally:
            await server.stop()

    asyncio.run(main())
