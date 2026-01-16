"""
Fireflies.ai Realtime Client

Handles communication with Fireflies.ai API:
- GraphQL API for querying active meetings
- WebSocket API for realtime transcript streaming

Reference:
- https://docs.fireflies.ai/realtime-api/overview
- https://docs.fireflies.ai/realtime-api/event-schema
- https://docs.fireflies.ai/graphql-api/query/active-meetings
"""

import logging
import asyncio
import json
from datetime import datetime
from typing import Dict, Any, Optional, List, Callable, Awaitable
import aiohttp
from aiohttp import WSMsgType, ClientWebSocketResponse

from models.fireflies import (
    FirefliesChunk,
    FirefliesMeeting,
    FirefliesConnectionStatus,
    MeetingState,
)

logger = logging.getLogger(__name__)


# =============================================================================
# Constants
# =============================================================================

DEFAULT_GRAPHQL_ENDPOINT = "https://api.fireflies.ai/graphql"
DEFAULT_WEBSOCKET_ENDPOINT = "wss://api.fireflies.ai/realtime"

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
StatusCallback = Callable[[FirefliesConnectionStatus, Optional[str]], Awaitable[None]]

# Callback for errors
ErrorCallback = Callable[[str, Optional[Exception]], Awaitable[None]]


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
        self._session: Optional[aiohttp.ClientSession] = None

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
        variables: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
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
                    error_msg = result.get("errors", [{"message": "Unknown error"}])[
                        0
                    ].get("message")
                    raise FirefliesAPIError(
                        f"GraphQL error: {error_msg}", status_code=response.status
                    )

                if "errors" in result:
                    error_msg = result["errors"][0].get(
                        "message", "Unknown GraphQL error"
                    )
                    raise FirefliesAPIError(f"GraphQL error: {error_msg}")

                return result.get("data", {})

        except aiohttp.ClientError as e:
            logger.error(f"Fireflies API request failed: {e}")
            raise FirefliesAPIError(f"Request failed: {e}") from e

    async def get_active_meetings(
        self,
        email: Optional[str] = None,
        states: Optional[List[MeetingState]] = None,
    ) -> List[FirefliesMeeting]:
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
                    start_time=datetime.fromisoformat(
                        m["start_time"].replace("Z", "+00:00")
                    )
                    if m.get("start_time")
                    else None,
                    end_time=datetime.fromisoformat(
                        m["end_time"].replace("Z", "+00:00")
                    )
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
    ) -> List[Dict[str, Any]]:
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
    ) -> Optional[Dict[str, Any]]:
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
# WebSocket Client
# =============================================================================


class FirefliesRealtimeClient:
    """
    WebSocket client for Fireflies.ai Realtime API.

    Connects to Fireflies WebSocket to receive live transcription events.
    Handles authentication, reconnection, and event parsing.

    Events emitted:
    - auth.success: Authentication succeeded
    - auth.failed: Authentication failed (socket will disconnect)
    - connection.established: Ready to receive transcripts
    - connection.error: Connection or authorization error
    - transcription.broadcast: New transcript segment
    """

    def __init__(
        self,
        api_key: str,
        transcript_id: str,
        endpoint: str = DEFAULT_WEBSOCKET_ENDPOINT,
        on_transcript: Optional[TranscriptCallback] = None,
        on_status_change: Optional[StatusCallback] = None,
        on_error: Optional[ErrorCallback] = None,
        auto_reconnect: bool = True,
        max_reconnect_attempts: int = MAX_RECONNECTION_ATTEMPTS,
    ):
        """
        Initialize the Fireflies realtime client.

        Args:
            api_key: Fireflies API key
            transcript_id: Transcript ID to connect to (from active_meetings)
            endpoint: WebSocket endpoint URL
            on_transcript: Callback for transcript chunks
            on_status_change: Callback for connection status changes
            on_error: Callback for errors
            auto_reconnect: Whether to automatically reconnect on disconnect
            max_reconnect_attempts: Maximum reconnection attempts
        """
        self.api_key = api_key
        self.transcript_id = transcript_id
        self.endpoint = endpoint

        # Callbacks
        self.on_transcript = on_transcript
        self.on_status_change = on_status_change
        self.on_error = on_error

        # Reconnection settings
        self.auto_reconnect = auto_reconnect
        self.max_reconnect_attempts = max_reconnect_attempts

        # State
        self._ws: Optional[ClientWebSocketResponse] = None
        self._session: Optional[aiohttp.ClientSession] = None
        self._status = FirefliesConnectionStatus.DISCONNECTED
        self._reconnect_attempts = 0
        self._reconnect_delay = INITIAL_RECONNECTION_DELAY
        self._running = False
        self._receive_task: Optional[asyncio.Task] = None

        # Chunk tracking for deduplication
        self._last_chunk_id: Optional[str] = None
        self._processed_chunk_ids: set = set()

    @property
    def status(self) -> FirefliesConnectionStatus:
        """Current connection status"""
        return self._status

    @property
    def is_connected(self) -> bool:
        """Whether currently connected and authenticated"""
        return self._status == FirefliesConnectionStatus.CONNECTED

    async def _set_status(
        self, status: FirefliesConnectionStatus, message: Optional[str] = None
    ):
        """Update status and notify callback"""
        self._status = status
        logger.info(
            f"Fireflies connection status: {status.value}"
            + (f" - {message}" if message else "")
        )

        if self.on_status_change:
            try:
                await self.on_status_change(status, message)
            except Exception as e:
                logger.error(f"Error in status callback: {e}")

    async def _notify_error(self, message: str, exception: Optional[Exception] = None):
        """Notify error callback"""
        logger.error(
            f"Fireflies error: {message}" + (f" - {exception}" if exception else "")
        )

        if self.on_error:
            try:
                await self.on_error(message, exception)
            except Exception as e:
                logger.error(f"Error in error callback: {e}")

    async def connect(self) -> bool:
        """
        Connect to Fireflies realtime WebSocket.

        Returns:
            True if connection and authentication succeeded
        """
        if self._running:
            logger.warning("Already connected or connecting")
            return self.is_connected

        self._running = True
        await self._set_status(FirefliesConnectionStatus.CONNECTING)

        try:
            # Create session if needed
            if self._session is None or self._session.closed:
                self._session = aiohttp.ClientSession()

            # Build WebSocket URL with authentication
            ws_url = f"{self.endpoint}?token={self.api_key}&transcript_id={self.transcript_id}"

            await self._set_status(FirefliesConnectionStatus.AUTHENTICATING)

            # Connect to WebSocket
            self._ws = await self._session.ws_connect(
                ws_url,
                heartbeat=30.0,
                receive_timeout=60.0,
            )

            logger.info(
                f"WebSocket connected to Fireflies for transcript: {self.transcript_id}"
            )

            # Start receiving messages
            self._receive_task = asyncio.create_task(self._receive_loop())

            # Wait briefly for auth response
            await asyncio.sleep(0.5)

            return self.is_connected

        except aiohttp.ClientError as e:
            await self._notify_error(f"Connection failed: {e}", e)
            await self._set_status(FirefliesConnectionStatus.ERROR, str(e))
            self._running = False
            return False
        except Exception as e:
            await self._notify_error(f"Unexpected connection error: {e}", e)
            await self._set_status(FirefliesConnectionStatus.ERROR, str(e))
            self._running = False
            return False

    async def disconnect(self):
        """Disconnect from Fireflies WebSocket"""
        self._running = False
        self.auto_reconnect = False  # Prevent reconnection

        if self._receive_task:
            self._receive_task.cancel()
            try:
                await self._receive_task
            except asyncio.CancelledError:
                pass
            self._receive_task = None

        if self._ws and not self._ws.closed:
            await self._ws.close()
            self._ws = None

        if self._session and not self._session.closed:
            await self._session.close()
            self._session = None

        await self._set_status(FirefliesConnectionStatus.DISCONNECTED)
        logger.info("Disconnected from Fireflies")

    async def _receive_loop(self):
        """Main loop for receiving WebSocket messages"""
        try:
            async for msg in self._ws:
                if msg.type == WSMsgType.TEXT:
                    await self._handle_message(msg.data)
                elif msg.type == WSMsgType.ERROR:
                    await self._notify_error(f"WebSocket error: {self._ws.exception()}")
                    break
                elif msg.type == WSMsgType.CLOSED:
                    logger.info("WebSocket closed by server")
                    break

        except asyncio.CancelledError:
            logger.debug("Receive loop cancelled")
            raise
        except Exception as e:
            await self._notify_error(f"Receive loop error: {e}", e)
        finally:
            if self._running and self.auto_reconnect:
                await self._attempt_reconnect()
            else:
                await self._set_status(FirefliesConnectionStatus.DISCONNECTED)

    async def _handle_message(self, data: str):
        """Parse and handle incoming WebSocket message"""
        try:
            message = json.loads(data)
            event_type = message.get("type") or message.get("event")

            # Handle different event types
            if event_type == "auth.success":
                await self._set_status(
                    FirefliesConnectionStatus.CONNECTED, "Authenticated"
                )
                self._reconnect_attempts = 0
                self._reconnect_delay = INITIAL_RECONNECTION_DELAY

            elif event_type == "auth.failed":
                await self._notify_error("Authentication failed")
                await self._set_status(
                    FirefliesConnectionStatus.ERROR, "Authentication failed"
                )
                self._running = False
                self.auto_reconnect = False  # Don't reconnect on auth failure

            elif event_type == "connection.established":
                logger.info("Fireflies connection established, ready for transcripts")

            elif event_type == "connection.error":
                error_msg = message.get("message", "Unknown connection error")
                await self._notify_error(f"Connection error: {error_msg}")

            elif event_type == "transcription.broadcast":
                await self._handle_transcript(message)

            else:
                logger.debug(f"Unhandled Fireflies event type: {event_type}")

        except json.JSONDecodeError as e:
            logger.warning(f"Failed to parse Fireflies message: {e}")
        except Exception as e:
            logger.error(f"Error handling Fireflies message: {e}")

    async def _handle_transcript(self, message: Dict[str, Any]):
        """Handle transcription.broadcast event"""
        try:
            # Extract chunk data - may be nested in 'data' or at top level
            chunk_data = message.get("data", message)

            chunk_id = chunk_data.get("chunk_id")

            # Skip if we've already processed this exact chunk
            # (Note: same chunk_id with different text = update, which we DO want)

            # Create FirefliesChunk model
            chunk = FirefliesChunk(
                transcript_id=chunk_data.get("transcript_id", self.transcript_id),
                chunk_id=chunk_id,
                text=chunk_data.get("text", ""),
                speaker_name=chunk_data.get("speaker_name", "Unknown"),
                start_time=float(chunk_data.get("start_time", 0.0)),
                end_time=float(chunk_data.get("end_time", 0.0)),
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

    async def _attempt_reconnect(self):
        """Attempt to reconnect with exponential backoff"""
        if not self.auto_reconnect:
            return

        if self._reconnect_attempts >= self.max_reconnect_attempts:
            await self._notify_error(
                f"Max reconnection attempts ({self.max_reconnect_attempts}) reached"
            )
            await self._set_status(
                FirefliesConnectionStatus.ERROR, "Max reconnection attempts reached"
            )
            self._running = False
            return

        self._reconnect_attempts += 1
        await self._set_status(
            FirefliesConnectionStatus.RECONNECTING,
            f"Attempt {self._reconnect_attempts}/{self.max_reconnect_attempts}",
        )

        logger.info(
            f"Reconnecting to Fireflies in {self._reconnect_delay:.1f}s "
            f"(attempt {self._reconnect_attempts}/{self.max_reconnect_attempts})"
        )

        await asyncio.sleep(self._reconnect_delay)

        # Increase delay for next attempt (exponential backoff)
        self._reconnect_delay = min(
            self._reconnect_delay * RECONNECTION_BACKOFF_MULTIPLIER,
            MAX_RECONNECTION_DELAY,
        )

        # Close existing connections
        if self._ws and not self._ws.closed:
            await self._ws.close()

        # Attempt reconnection
        try:
            ws_url = f"{self.endpoint}?token={self.api_key}&transcript_id={self.transcript_id}"
            self._ws = await self._session.ws_connect(
                ws_url,
                heartbeat=30.0,
                receive_timeout=60.0,
            )

            logger.info("Reconnected to Fireflies WebSocket")

            # Restart receive loop
            self._receive_task = asyncio.create_task(self._receive_loop())

        except Exception as e:
            logger.error(f"Reconnection failed: {e}")
            await self._attempt_reconnect()


# =============================================================================
# Unified Client
# =============================================================================


class FirefliesClient:
    """
    Unified Fireflies.ai client combining GraphQL and WebSocket functionality.

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
    ):
        """
        Initialize the Fireflies client.

        Args:
            api_key: Fireflies API key
            graphql_endpoint: GraphQL API endpoint
            websocket_endpoint: WebSocket API endpoint
        """
        self.api_key = api_key
        self.graphql_endpoint = graphql_endpoint
        self.websocket_endpoint = websocket_endpoint

        # GraphQL client
        self._graphql = FirefliesGraphQLClient(
            api_key=api_key,
            endpoint=graphql_endpoint,
        )

        # Active realtime connections (transcript_id -> client)
        self._realtime_clients: Dict[str, FirefliesRealtimeClient] = {}

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
        email: Optional[str] = None,
        states: Optional[List[MeetingState]] = None,
    ) -> List[FirefliesMeeting]:
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
    ) -> List[Dict[str, Any]]:
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
    ) -> Optional[Dict[str, Any]]:
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
        on_transcript: Optional[TranscriptCallback] = None,
        on_status_change: Optional[StatusCallback] = None,
        on_error: Optional[ErrorCallback] = None,
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
            on_transcript=on_transcript,
            on_status_change=on_status_change,
            on_error=on_error,
            auto_reconnect=auto_reconnect,
        )

        # Connect
        success = await client.connect()

        if success:
            self._realtime_clients[transcript_id] = client
            logger.info(
                f"Connected to Fireflies realtime for transcript: {transcript_id}"
            )
        else:
            logger.error(
                f"Failed to connect to Fireflies realtime for transcript: {transcript_id}"
            )

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

    def get_realtime_status(
        self, transcript_id: str
    ) -> Optional[FirefliesConnectionStatus]:
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

    def get_active_connections(self) -> Dict[str, FirefliesConnectionStatus]:
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

    def __init__(self, message: str, status_code: Optional[int] = None):
        super().__init__(message)
        self.status_code = status_code


class FirefliesConnectionError(Exception):
    """Exception for Fireflies WebSocket connection errors"""

    pass


class FirefliesAuthError(Exception):
    """Exception for Fireflies authentication errors"""

    pass
