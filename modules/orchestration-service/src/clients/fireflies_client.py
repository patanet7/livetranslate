"""
Fireflies.ai Realtime Client

Handles communication with Fireflies.ai API:
- GraphQL API for querying active meetings
- Socket.IO API for realtime transcript streaming

Reference:
- https://docs.fireflies.ai/realtime-api/overview
- https://docs.fireflies.ai/realtime-api/event-schema
- https://docs.fireflies.ai/graphql-api/query/active-meetings

NOTE: Fireflies uses Socket.IO protocol, NOT raw WebSocket.
The endpoint is wss://api.fireflies.ai with path /ws/realtime.
"""

import asyncio
import logging
import uuid
from collections.abc import Awaitable, Callable
from datetime import datetime
from typing import Any

import aiohttp
import socketio
from models.fireflies import (
    FirefliesChunk,
    FirefliesConnectionStatus,
    FirefliesMeeting,
    MeetingState,
)

logger = logging.getLogger(__name__)


# =============================================================================
# Constants
# =============================================================================

DEFAULT_GRAPHQL_ENDPOINT = "https://api.fireflies.ai/graphql"
# Socket.IO endpoint (NOT raw WebSocket)
DEFAULT_WEBSOCKET_ENDPOINT = "wss://api.fireflies.ai"
DEFAULT_WEBSOCKET_PATH = "/ws/realtime"

# Reconnection settings
MAX_RECONNECTION_ATTEMPTS = 5
INITIAL_RECONNECTION_DELAY = 1.0  # seconds
MAX_RECONNECTION_DELAY = 60.0  # seconds
RECONNECTION_BACKOFF_MULTIPLIER = 2.0


# =============================================================================
# GraphQL Queries
# =============================================================================

ACTIVE_MEETINGS_QUERY = """
query ActiveMeetings($email: String, $states: [MeetingState!]) {
  active_meetings(input: { email: $email, states: $states }) {
    id
    title
    organizer_email
    meeting_link
    start_time
    end_time
    privacy
    state
  }
}
"""

PAST_TRANSCRIPTS_QUERY = """
query Transcripts($limit: Int, $skip: Int) {
  transcripts(limit: $limit, skip: $skip) {
    id
    title
    date
    duration
    organizer_email
    participants
    summary {
      overview
      action_items
    }
  }
}
"""

TRANSCRIPT_DETAIL_QUERY = """
query Transcript($id: String!) {
  transcript(id: $id) {
    id
    title
    date
    duration
    sentences {
      text
      start_time
      end_time
      speaker_name
    }
    participants
  }
}
"""


# =============================================================================
# Event Callback Types
# =============================================================================

# Callback for receiving transcript chunks
TranscriptCallback = Callable[[FirefliesChunk], Awaitable[None]]

# Callback for connection status changes
StatusCallback = Callable[[FirefliesConnectionStatus, str | None], Awaitable[None]]

# Callback for errors
ErrorCallback = Callable[[str, Exception | None], Awaitable[None]]


# =============================================================================
# GraphQL Client
# =============================================================================


class FirefliesGraphQLClient:
    """
    GraphQL client for Fireflies.ai API.

    Used for querying active meetings to get transcript IDs
    for WebSocket connections.
    """

    def __init__(
        self,
        api_key: str,
        endpoint: str = DEFAULT_GRAPHQL_ENDPOINT,
        timeout: float = 30.0,
    ):
        self.api_key = api_key
        self.endpoint = endpoint
        self.timeout = aiohttp.ClientTimeout(total=timeout)
        self._session: aiohttp.ClientSession | None = None

    async def _get_session(self) -> aiohttp.ClientSession:
        """Get or create aiohttp session"""
        if self._session is None or self._session.closed:
            self._session = aiohttp.ClientSession(
                timeout=self.timeout,
                headers={
                    "Authorization": f"Bearer {self.api_key}",
                    "Content-Type": "application/json",
                },
            )
        return self._session

    async def close(self):
        """Close the client session"""
        if self._session and not self._session.closed:
            await self._session.close()
            self._session = None

    async def execute_query(
        self,
        query: str,
        variables: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        """
        Execute a GraphQL query.

        Args:
            query: GraphQL query string
            variables: Query variables

        Returns:
            Query result data

        Raises:
            FirefliesAPIError: If the API returns an error
        """
        session = await self._get_session()

        payload = {"query": query}
        if variables:
            payload["variables"] = variables

        try:
            async with session.post(self.endpoint, json=payload) as response:
                result = await response.json()

                if response.status != 200:
                    error_msg = result.get("errors", [{"message": "Unknown error"}])[0].get(
                        "message"
                    )
                    raise FirefliesAPIError(
                        f"GraphQL error: {error_msg}", status_code=response.status
                    )

                if "errors" in result:
                    error_msg = result["errors"][0].get("message", "Unknown GraphQL error")
                    raise FirefliesAPIError(f"GraphQL error: {error_msg}")

                return result.get("data", {})

        except aiohttp.ClientError as e:
            logger.error(f"Fireflies API request failed: {e}")
            raise FirefliesAPIError(f"Request failed: {e}") from e

    async def get_active_meetings(
        self,
        email: str | None = None,
        states: list[MeetingState] | None = None,
    ) -> list[FirefliesMeeting]:
        """
        Get active meetings from Fireflies.

        Args:
            email: Filter by organizer email (optional)
            states: Filter by meeting states (default: active and paused)

        Returns:
            List of active meetings
        """
        variables = {}
        if email:
            variables["email"] = email
        if states:
            variables["states"] = [s.value for s in states]

        try:
            data = await self.execute_query(ACTIVE_MEETINGS_QUERY, variables)
            meetings_data = data.get("active_meetings", [])

            meetings = []
            for m in meetings_data:
                meeting = FirefliesMeeting(
                    id=m["id"],
                    title=m.get("title"),
                    organizer_email=m.get("organizer_email"),
                    meeting_link=m.get("meeting_link"),
                    start_time=datetime.fromisoformat(m["start_time"].replace("Z", "+00:00"))
                    if m.get("start_time")
                    else None,
                    end_time=datetime.fromisoformat(m["end_time"].replace("Z", "+00:00"))
                    if m.get("end_time")
                    else None,
                    privacy=m.get("privacy"),
                    state=MeetingState(m.get("state", "active")),
                )
                meetings.append(meeting)

            logger.info(f"Found {len(meetings)} active meetings")
            return meetings

        except Exception as e:
            logger.error(f"Failed to get active meetings: {e}")
            raise

    async def get_transcripts(
        self,
        limit: int = 20,
        skip: int = 0,
    ) -> list[dict[str, Any]]:
        """
        Get past transcripts from Fireflies.

        Args:
            limit: Maximum number of transcripts to return (default: 20)
            skip: Number of transcripts to skip (for pagination)

        Returns:
            List of transcript dictionaries
        """
        variables = {"limit": limit, "skip": skip}

        try:
            data = await self.execute_query(PAST_TRANSCRIPTS_QUERY, variables)
            transcripts = data.get("transcripts", [])

            logger.info(f"Found {len(transcripts)} past transcripts")
            return transcripts

        except Exception as e:
            logger.error(f"Failed to get transcripts: {e}")
            raise

    async def get_transcript_detail(
        self,
        transcript_id: str,
    ) -> dict[str, Any] | None:
        """
        Get detailed transcript including sentences.

        Args:
            transcript_id: Transcript ID to fetch

        Returns:
            Transcript detail with sentences, or None if not found
        """
        variables = {"id": transcript_id}

        try:
            data = await self.execute_query(TRANSCRIPT_DETAIL_QUERY, variables)
            return data.get("transcript")

        except Exception as e:
            logger.error(f"Failed to get transcript detail: {e}")
            raise


# =============================================================================
# Socket.IO Client (Fireflies Realtime API)
# =============================================================================


class FirefliesRealtimeClient:
    """
    Socket.IO client for Fireflies.ai Realtime API.

    Connects to Fireflies Socket.IO endpoint to receive live transcription events.
    Handles authentication, reconnection, and event parsing.

    Events received:
    - auth.success: Authentication succeeded
    - auth.failed: Authentication failed (socket will disconnect)
    - connection.established: Ready to receive transcripts
    - connection.error: Connection or authorization error
    - transcription.broadcast: New transcript segment

    NOTE: Fireflies uses Socket.IO protocol, NOT raw WebSocket.
    Previous implementation using aiohttp ws_connect returned 404.
    """

    def __init__(
        self,
        api_key: str,
        transcript_id: str,
        endpoint: str = DEFAULT_WEBSOCKET_ENDPOINT,
        socketio_path: str = DEFAULT_WEBSOCKET_PATH,
        on_transcript: TranscriptCallback | None = None,
        on_status_change: StatusCallback | None = None,
        on_error: ErrorCallback | None = None,
        auto_reconnect: bool = True,
        max_reconnect_attempts: int = MAX_RECONNECTION_ATTEMPTS,
    ):
        """
        Initialize the Fireflies realtime client.

        Args:
            api_key: Fireflies API key
            transcript_id: Transcript ID to connect to (from active_meetings)
            endpoint: Socket.IO endpoint URL (wss://api.fireflies.ai)
            socketio_path: Socket.IO path (/ws/realtime)
            on_transcript: Callback for transcript chunks
            on_status_change: Callback for connection status changes
            on_error: Callback for errors
            auto_reconnect: Whether to automatically reconnect on disconnect
            max_reconnect_attempts: Maximum reconnection attempts
        """
        self.api_key = api_key
        self.transcript_id = transcript_id
        self.endpoint = endpoint
        self.socketio_path = socketio_path

        # Callbacks
        self.on_transcript = on_transcript
        self.on_status_change = on_status_change
        self.on_error = on_error

        # Reconnection settings
        self.auto_reconnect = auto_reconnect
        self.max_reconnect_attempts = max_reconnect_attempts

        # State
        self._status = FirefliesConnectionStatus.DISCONNECTED
        self._reconnect_attempts = 0
        self._reconnect_delay = INITIAL_RECONNECTION_DELAY
        self._running = False

        # Socket.IO client
        self.sio = socketio.AsyncClient(
            reconnection=auto_reconnect,
            reconnection_attempts=max_reconnect_attempts,
            reconnection_delay=INITIAL_RECONNECTION_DELAY,
            reconnection_delay_max=MAX_RECONNECTION_DELAY,
            logger=False,  # Use our own logging
            engineio_logger=False,
        )

        # Register event handlers
        self._register_handlers()

        # Chunk tracking for deduplication
        self._last_chunk_id: str | None = None
        self._processed_chunk_ids: set = set()

    def _register_handlers(self):
        """Register Socket.IO event handlers"""

        @self.sio.event
        async def connect():
            """Called when Socket.IO connects"""
            logger.info("Socket.IO connected to Fireflies")
            await self._set_status(FirefliesConnectionStatus.AUTHENTICATING)

        @self.sio.event
        async def disconnect():
            """Called when Socket.IO disconnects"""
            logger.info("Socket.IO disconnected from Fireflies")
            if self._running:
                await self._set_status(FirefliesConnectionStatus.RECONNECTING)
            else:
                await self._set_status(FirefliesConnectionStatus.DISCONNECTED)

        @self.sio.event
        async def connect_error(data):
            """Called on connection error"""
            error_msg = str(data) if data else "Connection error"
            logger.error(f"Socket.IO connection error: {error_msg}")
            await self._notify_error(f"Connection error: {error_msg}")
            await self._set_status(FirefliesConnectionStatus.ERROR, error_msg)

        # Fireflies-specific events
        @self.sio.on("auth.success")
        async def on_auth_success(data=None):
            """Called when authentication succeeds"""
            logger.info("Fireflies authentication successful")
            await self._set_status(FirefliesConnectionStatus.CONNECTED, "Authenticated")
            self._reconnect_attempts = 0
            self._reconnect_delay = INITIAL_RECONNECTION_DELAY

        @self.sio.on("auth.failed")
        async def on_auth_failed(data=None):
            """Called when authentication fails"""
            error_msg = (
                data.get("message", "Authentication failed")
                if isinstance(data, dict)
                else "Authentication failed"
            )
            logger.error(f"Fireflies authentication failed: {error_msg}")
            await self._notify_error(error_msg)
            await self._set_status(FirefliesConnectionStatus.ERROR, error_msg)
            self._running = False
            self.auto_reconnect = False  # Don't reconnect on auth failure

        @self.sio.on("connection.established")
        async def on_connection_established(data=None):
            """Called when connection is fully established"""
            logger.info("Fireflies connection established, ready for transcripts")

        @self.sio.on("connection.error")
        async def on_connection_error(data=None):
            """Called on Fireflies connection error"""
            error_msg = (
                data.get("message", "Connection error")
                if isinstance(data, dict)
                else "Connection error"
            )
            logger.error(f"Fireflies connection error: {error_msg}")
            await self._notify_error(f"Connection error: {error_msg}")

        @self.sio.on("transcription.broadcast")
        async def on_transcription_broadcast(data):
            """Called when new transcription data is received"""
            await self._handle_transcript(data)

        # Also handle generic 'transcript' event (some APIs use this)
        @self.sio.on("transcript")
        async def on_transcript_event(data):
            """Fallback handler for transcript events"""
            await self._handle_transcript(data)

    @property
    def status(self) -> FirefliesConnectionStatus:
        """Current connection status"""
        return self._status

    @property
    def is_connected(self) -> bool:
        """Whether currently connected and authenticated"""
        return self._status == FirefliesConnectionStatus.CONNECTED

    async def _set_status(self, status: FirefliesConnectionStatus, message: str | None = None):
        """Update status and notify callback"""
        self._status = status
        logger.info(
            f"Fireflies connection status: {status.value}" + (f" - {message}" if message else "")
        )

        if self.on_status_change:
            try:
                await self.on_status_change(status, message)
            except Exception as e:
                logger.error(f"Error in status callback: {e}")

    async def _notify_error(self, message: str, exception: Exception | None = None):
        """Notify error callback"""
        logger.error(f"Fireflies error: {message}" + (f" - {exception}" if exception else ""))

        if self.on_error:
            try:
                await self.on_error(message, exception)
            except Exception as e:
                logger.error(f"Error in error callback: {e}")

    async def connect(self) -> bool:
        """
        Connect to Fireflies realtime Socket.IO endpoint.

        Returns:
            True if connection and authentication succeeded
        """
        if self._running:
            logger.warning("Already connected or connecting")
            return self.is_connected

        self._running = True
        await self._set_status(FirefliesConnectionStatus.CONNECTING)

        try:
            # Socket.IO auth payload
            auth = {
                "token": f"Bearer {self.api_key}",
                "transcriptId": self.transcript_id,
            }

            logger.info(
                f"Connecting to Fireflies Socket.IO: {self.endpoint} "
                f"(path: {self.socketio_path}, transcript: {self.transcript_id})"
            )

            # Connect using Socket.IO protocol
            await self.sio.connect(
                self.endpoint,
                socketio_path=self.socketio_path,
                auth=auth,
                transports=["websocket"],  # Prefer WebSocket transport
                wait_timeout=30,
            )

            # Wait briefly for auth response
            await asyncio.sleep(0.5)

            return self.is_connected

        except socketio.exceptions.ConnectionError as e:
            error_msg = f"Socket.IO connection failed: {e}"
            logger.error(error_msg)
            await self._notify_error(error_msg, e)
            await self._set_status(FirefliesConnectionStatus.ERROR, str(e))
            self._running = False
            return False
        except Exception as e:
            error_msg = f"Unexpected connection error: {e}"
            logger.error(error_msg)
            await self._notify_error(error_msg, e)
            await self._set_status(FirefliesConnectionStatus.ERROR, str(e))
            self._running = False
            return False

    async def disconnect(self):
        """Disconnect from Fireflies Socket.IO"""
        self._running = False
        self.auto_reconnect = False  # Prevent reconnection

        # Update Socket.IO client reconnection setting
        self.sio.reconnection = False

        try:
            if self.sio.connected:
                await self.sio.disconnect()
        except Exception as e:
            logger.warning(f"Error during disconnect: {e}")

        await self._set_status(FirefliesConnectionStatus.DISCONNECTED)
        logger.info("Disconnected from Fireflies")

    async def _handle_transcript(self, message: dict[str, Any]):
        """Handle transcription.broadcast event"""
        try:
            # Extract chunk data - may be nested in 'data' or at top level
            chunk_data = message.get("data", message) if isinstance(message, dict) else message

            if not isinstance(chunk_data, dict):
                logger.warning(f"Unexpected transcript data format: {type(chunk_data)}")
                return

            chunk_id = chunk_data.get("chunk_id") or chunk_data.get("id") or uuid.uuid4().hex[:12]

            # Log raw data for debugging (first few chunks per session)
            logger.info(
                f"Fireflies chunk received: keys={list(chunk_data.keys())}, "
                f"chunk_id={chunk_id}, text={str(chunk_data.get('text', ''))[:80]}"
            )

            # Create FirefliesChunk model
            chunk = FirefliesChunk(
                transcript_id=chunk_data.get("transcript_id", self.transcript_id),
                chunk_id=chunk_id,
                text=chunk_data.get("text", chunk_data.get("content", "")),
                speaker_name=chunk_data.get("speaker_name", chunk_data.get("speaker", "Unknown")),
                start_time=float(chunk_data.get("start_time", chunk_data.get("startTime", 0.0))),
                end_time=float(chunk_data.get("end_time", chunk_data.get("endTime", 0.0))),
            )

            # Track last chunk for deduplication reference
            self._last_chunk_id = chunk_id

            logger.debug(
                f"Transcript chunk: [{chunk.speaker_name}] {chunk.text[:50]}... "
                f"({chunk.start_time:.2f}s - {chunk.end_time:.2f}s)"
            )

            # Call transcript callback
            if self.on_transcript:
                await self.on_transcript(chunk)

        except Exception as e:
            logger.error(f"Error processing transcript chunk: {e}")


# =============================================================================
# Unified Client
# =============================================================================


class FirefliesClient:
    """
    Unified Fireflies.ai client combining GraphQL and Socket.IO functionality.

    Usage:
        client = FirefliesClient(api_key="your-api-key")

        # Get active meetings
        meetings = await client.get_active_meetings()

        # Connect to a meeting's realtime transcript
        await client.connect_realtime(
            transcript_id=meetings[0].id,
            on_transcript=handle_transcript,
        )
    """

    def __init__(
        self,
        api_key: str,
        graphql_endpoint: str = DEFAULT_GRAPHQL_ENDPOINT,
        websocket_endpoint: str = DEFAULT_WEBSOCKET_ENDPOINT,
        socketio_path: str = DEFAULT_WEBSOCKET_PATH,
    ):
        """
        Initialize the Fireflies client.

        Args:
            api_key: Fireflies API key
            graphql_endpoint: GraphQL API endpoint
            websocket_endpoint: Socket.IO endpoint URL
            socketio_path: Socket.IO path for realtime
        """
        self.api_key = api_key
        self.graphql_endpoint = graphql_endpoint
        self.websocket_endpoint = websocket_endpoint
        self.socketio_path = socketio_path

        # GraphQL client
        self._graphql = FirefliesGraphQLClient(
            api_key=api_key,
            endpoint=graphql_endpoint,
        )

        # Active realtime connections (transcript_id -> client)
        self._realtime_clients: dict[str, FirefliesRealtimeClient] = {}

    async def close(self):
        """Close all connections"""
        # Close GraphQL client
        await self._graphql.close()

        # Close all realtime connections
        for client in self._realtime_clients.values():
            await client.disconnect()
        self._realtime_clients.clear()

    # -------------------------------------------------------------------------
    # GraphQL Methods
    # -------------------------------------------------------------------------

    async def get_active_meetings(
        self,
        email: str | None = None,
        states: list[MeetingState] | None = None,
    ) -> list[FirefliesMeeting]:
        """
        Get active meetings from Fireflies.

        Args:
            email: Filter by organizer email (optional)
            states: Filter by meeting states

        Returns:
            List of active meetings
        """
        return await self._graphql.get_active_meetings(email, states)

    async def get_transcripts(
        self,
        limit: int = 20,
        skip: int = 0,
    ) -> list[dict[str, Any]]:
        """
        Get past transcripts from Fireflies.

        Args:
            limit: Maximum number of transcripts to return
            skip: Number of transcripts to skip (pagination)

        Returns:
            List of transcript dictionaries
        """
        return await self._graphql.get_transcripts(limit, skip)

    async def get_transcript_detail(
        self,
        transcript_id: str,
    ) -> dict[str, Any] | None:
        """
        Get detailed transcript including sentences.

        Args:
            transcript_id: Transcript ID to fetch

        Returns:
            Transcript detail with sentences
        """
        return await self._graphql.get_transcript_detail(transcript_id)

    # -------------------------------------------------------------------------
    # Realtime Methods
    # -------------------------------------------------------------------------

    async def connect_realtime(
        self,
        transcript_id: str,
        on_transcript: TranscriptCallback | None = None,
        on_status_change: StatusCallback | None = None,
        on_error: ErrorCallback | None = None,
        auto_reconnect: bool = True,
    ) -> FirefliesRealtimeClient:
        """
        Connect to a realtime transcript stream.

        Args:
            transcript_id: Transcript ID to connect to
            on_transcript: Callback for transcript chunks
            on_status_change: Callback for status changes
            on_error: Callback for errors
            auto_reconnect: Whether to auto-reconnect on disconnect

        Returns:
            The realtime client instance
        """
        # Check if already connected to this transcript
        if transcript_id in self._realtime_clients:
            existing = self._realtime_clients[transcript_id]
            if existing.is_connected:
                logger.warning(f"Already connected to transcript: {transcript_id}")
                return existing
            else:
                # Clean up old connection
                await existing.disconnect()

        # Create new realtime client
        client = FirefliesRealtimeClient(
            api_key=self.api_key,
            transcript_id=transcript_id,
            endpoint=self.websocket_endpoint,
            socketio_path=self.socketio_path,
            on_transcript=on_transcript,
            on_status_change=on_status_change,
            on_error=on_error,
            auto_reconnect=auto_reconnect,
        )

        # Connect
        success = await client.connect()

        if success:
            self._realtime_clients[transcript_id] = client
            logger.info(f"Connected to Fireflies realtime for transcript: {transcript_id}")
        else:
            logger.error(f"Failed to connect to Fireflies realtime for transcript: {transcript_id}")

        return client

    async def disconnect_realtime(self, transcript_id: str):
        """
        Disconnect from a realtime transcript stream.

        Args:
            transcript_id: Transcript ID to disconnect from
        """
        if transcript_id in self._realtime_clients:
            client = self._realtime_clients.pop(transcript_id)
            await client.disconnect()
            logger.info(f"Disconnected from Fireflies realtime: {transcript_id}")
        else:
            logger.warning(f"No active connection for transcript: {transcript_id}")

    def get_realtime_status(self, transcript_id: str) -> FirefliesConnectionStatus | None:
        """
        Get the connection status for a transcript.

        Args:
            transcript_id: Transcript ID to check

        Returns:
            Connection status or None if not connected
        """
        if transcript_id in self._realtime_clients:
            return self._realtime_clients[transcript_id].status
        return None

    def get_active_connections(self) -> dict[str, FirefliesConnectionStatus]:
        """
        Get all active realtime connections and their statuses.

        Returns:
            Dictionary of transcript_id -> connection status
        """
        return {tid: client.status for tid, client in self._realtime_clients.items()}


# =============================================================================
# Exceptions
# =============================================================================


class FirefliesAPIError(Exception):
    """Exception for Fireflies API errors"""

    def __init__(self, message: str, status_code: int | None = None):
        super().__init__(message)
        self.status_code = status_code


class FirefliesConnectionError(Exception):
    """Exception for Fireflies WebSocket connection errors"""

    pass


class FirefliesAuthError(Exception):
    """Exception for Fireflies authentication errors"""

    pass
