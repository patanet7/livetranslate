"""
Fireflies Demo Server

Production-ready copy of the mock Fireflies server for demo mode.
Simulates Fireflies.ai GraphQL API and Socket.IO Realtime API without
requiring a real Fireflies account.

Copied from tests/fireflies/mocks/fireflies_mock_server.py to avoid
importing test code in production.
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
    organizer_email: str = "demo@livetranslate.local"
    meeting_link: str | None = "https://meet.example.com/demo"
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
    """Mock transcript chunk data."""

    chunk_id: str = field(default_factory=lambda: f"chunk_{uuid.uuid4().hex[:8]}")
    text: str = ""
    speaker_name: str = "Speaker"
    start_time: float = 0.0
    end_time: float = 0.0
    confidence: float = 0.95
    is_final: bool = True

    def to_socketio_dict(self, transcript_id: str) -> dict[str, Any]:
        """Convert to Socket.IO event data."""
        return {
            "transcript_id": transcript_id,
            "chunk_id": self.chunk_id,
            "text": self.text,
            "speaker_name": self.speaker_name,
            "start_time": self.start_time,
            "end_time": self.end_time,
            "confidence": self.confidence,
            "is_final": self.is_final,
        }


@dataclass
class MockTranscriptScenario:
    """Defines a transcript scenario for demo streaming."""

    transcript_id: str = field(default_factory=lambda: f"transcript_{uuid.uuid4().hex[:8]}")
    meeting: MockMeeting = field(default_factory=MockMeeting)
    chunks: list[MockChunk] = field(default_factory=list)
    chunk_delay_ms: float = 500.0
    word_delay_ms: float = 100.0
    stream_mode: str = "chunks"
    simulate_disconnect_after_chunks: int | None = None
    simulate_error_message: str | None = None

    @classmethod
    def conversation_scenario(
        cls,
        speakers: list[str] | None = None,
        num_exchanges: int = 10,
        chunk_delay_ms: float = 500.0,
    ) -> "MockTranscriptScenario":
        """Create a realistic conversation scenario."""
        if speakers is None:
            speakers = ["Alice", "Bob", "Charlie"]

        conversation_templates = [
            "Hello everyone, thank you for joining today's meeting.",
            "Good morning! Happy to be here. Let's make this productive.",
            "Let's get started with the agenda for today.",
            "I have a question about the quarterly budget projections.",
            "That's a great point. I think we should explore that further.",
            "Could you elaborate on the timeline for the rollout?",
            "I think we should consider the alternatives before deciding.",
            "The deadline is next Friday, so we need to move quickly.",
            "Does anyone have concerns about the proposed approach?",
            "I agree with that approach. It aligns with our goals.",
            "Let me share my screen so everyone can see the data.",
            "Can everyone see this chart? The numbers look promising.",
            "The Q3 numbers are showing a fifteen percent improvement.",
            "We need to address the performance bottleneck in the pipeline.",
            "What are the next steps for the implementation phase?",
            "I'll follow up with the engineering team on that item.",
            "Any other questions before we move to the next topic?",
            "Thanks for the update. That was very informative.",
            "Let's schedule a follow-up meeting for next week.",
            "Great meeting everyone! Thanks for your time today.",
            "One more thing — can we align on the testing strategy?",
            "I'll send out the meeting notes by end of day.",
            "The client feedback has been overwhelmingly positive.",
            "We should prioritize the security audit before launch.",
            "I'll prepare a detailed proposal for the next session.",
            "Let's make sure we document all the action items.",
            "The beta testing phase starts on Monday morning.",
            "We've seen a significant reduction in error rates.",
            "I'll coordinate with the design team on the mockups.",
            "Excellent progress this week. Keep up the great work!",
        ]

        chunks = []
        current_time = 0.0

        for i in range(num_exchanges):
            speaker = speakers[i % len(speakers)]
            text = conversation_templates[i % len(conversation_templates)]
            word_count = len(text.split())
            duration = word_count * 0.3

            chunk = MockChunk(
                chunk_id=f"chunk_{i + 1:04d}",
                text=text,
                speaker_name=speaker,
                start_time=current_time,
                end_time=current_time + duration,
            )
            chunks.append(chunk)
            current_time += duration + 0.2

        return cls(
            chunks=chunks,
            chunk_delay_ms=chunk_delay_ms,
        )


# =============================================================================
# Demo Server (Socket.IO)
# =============================================================================


class FirefliesDemoServer:
    """
    Mock server for Fireflies.ai demo mode.

    Provides:
    - Socket.IO endpoint at /ws/realtime for realtime transcript streaming
    - GraphQL endpoint at /graphql for active_meetings queries
    - Health check at /health
    """

    def __init__(
        self,
        host: str = "localhost",
        port: int = 8090,
        valid_api_keys: set[str] | None = None,
    ):
        self.host = host
        self.port = port
        self.valid_api_keys = valid_api_keys or {"demo-api-key", "test-api-key", "ff-test-key"}

        self._scenarios: dict[str, MockTranscriptScenario] = {}
        self._meetings: list[MockMeeting] = []
        self._sid_to_transcript: dict[str, str] = {}

        self._app: web.Application | None = None
        self._runner: web.AppRunner | None = None
        self._site: web.TCPSite | None = None
        self._sio: socketio.AsyncServer | None = None

        self._stats = {
            "graphql_requests": 0,
            "websocket_connections": 0,
            "chunks_sent": 0,
            "errors": 0,
        }

        self._background_tasks: set[asyncio.Task] = set()

    @property
    def base_url(self) -> str:
        """Base URL for the demo server."""
        return f"http://{self.host}:{self.port}"

    @property
    def stats(self) -> dict[str, int]:
        """Server statistics."""
        return self._stats.copy()

    # =========================================================================
    # Scenario Management
    # =========================================================================

    def add_scenario(self, scenario: MockTranscriptScenario) -> str:
        """Add a scenario. Returns transcript ID."""
        self._scenarios[scenario.transcript_id] = scenario
        self._meetings.append(scenario.meeting)
        logger.info(f"Demo server: added scenario {scenario.transcript_id}")
        return scenario.transcript_id

    def clear_scenarios(self):
        """Remove all scenarios."""
        self._scenarios.clear()
        self._meetings.clear()

    # =========================================================================
    # Server Lifecycle
    # =========================================================================

    async def start(self):
        """Start the demo server with Socket.IO."""
        self._sio = socketio.AsyncServer(
            async_mode="aiohttp",
            cors_allowed_origins="*",
            logger=False,
            engineio_logger=False,
        )

        self._register_socketio_handlers()

        self._app = web.Application()
        self._sio.attach(self._app, socketio_path="/ws/realtime")

        self._app.router.add_post("/graphql", self._handle_graphql)
        self._app.router.add_get("/health", self._handle_health)

        self._runner = web.AppRunner(self._app)
        await self._runner.setup()

        self._site = web.TCPSite(self._runner, self.host, self.port)
        await self._site.start()

        logger.info(f"Demo server started at {self.base_url}")

    async def stop(self):
        """Stop the demo server."""
        for task in list(self._background_tasks):
            task.cancel()
            with contextlib.suppress(asyncio.CancelledError):
                await task
        self._background_tasks.clear()

        if self._sio:
            for sid in list(self._sid_to_transcript.keys()):
                with contextlib.suppress(Exception):
                    await self._sio.disconnect(sid)
        self._sid_to_transcript.clear()

        if self._site:
            await self._site.stop()
        if self._runner:
            await self._runner.cleanup()

        self._site = None
        self._runner = None
        self._app = None
        self._sio = None

        logger.info("Demo server stopped")

    # =========================================================================
    # Socket.IO Event Handlers
    # =========================================================================

    def _register_socketio_handlers(self):
        """Register Socket.IO event handlers."""
        sio = self._sio

        @sio.event
        async def connect(sid, environ, auth):
            self._stats["websocket_connections"] += 1

            if not auth or not isinstance(auth, dict):
                await sio.emit("auth.failed", {"message": "Missing auth payload"}, to=sid)
                await sio.disconnect(sid)
                return False

            token = auth.get("token", "")
            transcript_id = auth.get("transcriptId", "")

            if not self._validate_api_key(token):
                await sio.emit("auth.failed", {"message": "Invalid API key"}, to=sid)
                await sio.disconnect(sid)
                return False

            if not transcript_id:
                await sio.emit("auth.failed", {"message": "Missing transcriptId"}, to=sid)
                await sio.disconnect(sid)
                return False

            self._sid_to_transcript[sid] = transcript_id
            logger.info(f"Demo client {sid} connected for transcript: {transcript_id}")

            await sio.emit("auth.success", {}, to=sid)
            await sio.emit("connection.established", {}, to=sid)

            scenario = self._scenarios.get(transcript_id)
            if scenario:
                task = asyncio.create_task(self._stream_scenario(sid, scenario))
                self._background_tasks.add(task)
                task.add_done_callback(self._background_tasks.discard)

            return True

        @sio.event
        async def disconnect(sid):
            self._sid_to_transcript.pop(sid, None)

        @sio.on("ping")
        async def on_ping(sid, data=None):
            await sio.emit("pong", {}, to=sid)

    # =========================================================================
    # Scenario Streaming
    # =========================================================================

    async def _stream_scenario(self, sid: str, scenario: MockTranscriptScenario):
        """Stream a scenario's chunks to a Socket.IO client."""
        sio = self._sio
        try:
            for i, chunk in enumerate(scenario.chunks):
                if sid not in self._sid_to_transcript:
                    break

                if (
                    scenario.simulate_disconnect_after_chunks
                    and i >= scenario.simulate_disconnect_after_chunks
                ):
                    await sio.disconnect(sid)
                    break

                data = chunk.to_socketio_dict(scenario.transcript_id)
                await sio.emit("transcription.broadcast", data, to=sid)
                self._stats["chunks_sent"] += 1

                if scenario.stream_mode == "words":
                    await asyncio.sleep(scenario.word_delay_ms / 1000.0)
                else:
                    await asyncio.sleep(scenario.chunk_delay_ms / 1000.0)

            if scenario.simulate_error_message and sid in self._sid_to_transcript:
                await sio.emit(
                    "connection.error",
                    {"message": scenario.simulate_error_message},
                    to=sid,
                )

        except Exception as e:
            logger.error(f"Error streaming demo scenario to {sid}: {e}")
            self._stats["errors"] += 1

    # =========================================================================
    # HTTP Request Handlers
    # =========================================================================

    async def _handle_health(self, request: web.Request) -> web.Response:
        return web.json_response(
            {
                "status": "ok",
                "mode": "demo",
                "scenarios": len(self._scenarios),
                "active_connections": len(self._sid_to_transcript),
            }
        )

    async def _handle_graphql(self, request: web.Request) -> web.Response:
        self._stats["graphql_requests"] += 1

        auth_header = request.headers.get("Authorization", "")
        if not self._validate_api_key(auth_header):
            return web.json_response(
                {"errors": [{"message": "Invalid API key"}]},
                status=401,
            )

        try:
            body = await request.json()
            query = body.get("query", "")

            if "active_meetings" in query:
                filtered_meetings = [m.to_graphql_dict() for m in self._meetings]
                return web.json_response({"data": {"active_meetings": filtered_meetings}})

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

    # =========================================================================
    # Utility
    # =========================================================================

    def _validate_api_key(self, auth_value: str) -> bool:
        if not auth_value:
            return False
        if auth_value.startswith("Bearer "):
            api_key = auth_value[7:]
        else:
            api_key = auth_value
        return api_key in self.valid_api_keys
