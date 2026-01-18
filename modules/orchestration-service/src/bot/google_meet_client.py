#!/usr/bin/env python3
"""
Google Meet API Client

Official Google Meet API integration for the orchestration service.
Provides authenticated access to Google Meet spaces, conference records,
and participant data for the LiveTranslate bot system.

Features:
- OAuth 2.0 authentication with required scopes
- Space management (create, join, monitor)
- Conference record access and participant data
- Real-time meeting event monitoring
- Transcript and recording access when available
- Integration with GoogleMeetBotManager

API Documentation: https://developers.google.com/workspace/meet/api
"""

import asyncio
import json
import logging
import os
import time
from dataclasses import dataclass
from collections.abc import Callable
from typing import Any

from google.auth.transport.requests import Request
from google.oauth2.credentials import Credentials
from google_auth_oauthlib.flow import InstalledAppFlow
from googleapiclient.discovery import build
from googleapiclient.errors import HttpError

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Required OAuth 2.0 scopes for Google Meet API
REQUIRED_SCOPES = [
    "https://www.googleapis.com/auth/meetings.space.created",
    "https://www.googleapis.com/auth/meetings.space.readonly",
    "https://www.googleapis.com/auth/meetings.space.settings",
]


@dataclass
class GoogleMeetConfig:
    """Configuration for Google Meet API client."""

    credentials_path: str
    token_path: str = "token.json"
    application_name: str = "LiveTranslate Bot"
    api_version: str = "v2"
    timeout: float = 30.0
    retry_attempts: int = 3
    retry_delay: float = 1.0


@dataclass
class MeetingSpace:
    """Google Meet space information."""

    name: str  # Format: spaces/{space_id}
    meeting_uri: str
    meeting_code: str
    config: dict[str, Any]
    active_conference: str | None = None
    created_time: str | None = None
    end_time: str | None = None


@dataclass
class ConferenceRecord:
    """Google Meet conference record information."""

    name: str  # Format: conferenceRecords/{conference_id}
    start_time: str
    end_time: str | None = None
    expire_time: str | None = None
    space: str | None = None


@dataclass
class Participant:
    """Meeting participant information."""

    name: str  # Format: conferenceRecords/{conference_id}/participants/{participant_id}
    earliest_start_time: str | None = None
    latest_end_time: str | None = None
    participant_id: str | None = None
    display_name: str | None = None


@dataclass
class TranscriptEntry:
    """Meeting transcript entry."""

    name: str  # Format: conferenceRecords/{conference_id}/transcripts/{transcript_id}/entries/{entry_id}
    participant: str | None = None
    text: str | None = None
    language_code: str | None = None
    start_time: str | None = None
    end_time: str | None = None


class GoogleMeetAuthenticator:
    """Handles Google Meet API authentication."""

    def __init__(self, config: GoogleMeetConfig):
        self.config = config
        self.credentials = None
        self.service = None

    async def authenticate(self) -> bool:
        """Authenticate with Google Meet API."""
        try:
            # Load existing credentials
            if os.path.exists(self.config.token_path):
                self.credentials = Credentials.from_authorized_user_file(
                    self.config.token_path, REQUIRED_SCOPES
                )

            # Refresh or obtain new credentials
            if not self.credentials or not self.credentials.valid:
                if self.credentials and self.credentials.expired and self.credentials.refresh_token:
                    logger.info("Refreshing Google Meet API credentials")
                    self.credentials.refresh(Request())
                else:
                    logger.info("Obtaining new Google Meet API credentials")
                    flow = InstalledAppFlow.from_client_secrets_file(
                        self.config.credentials_path, REQUIRED_SCOPES
                    )
                    self.credentials = flow.run_local_server(port=0)

                # Save credentials for next run
                with open(self.config.token_path, "w") as token:
                    token.write(self.credentials.to_json())

            # Build service
            self.service = build(
                "meet",
                self.config.api_version,
                credentials=self.credentials,
                cache_discovery=False,
            )

            logger.info("Google Meet API authentication successful")
            return True

        except Exception as e:
            logger.error(f"Google Meet API authentication failed: {e}")
            return False

    def is_authenticated(self) -> bool:
        """Check if currently authenticated."""
        return self.credentials is not None and self.credentials.valid and self.service is not None


class GoogleMeetSpaceManager:
    """Manages Google Meet spaces."""

    def __init__(self, service, config: GoogleMeetConfig):
        self.service = service
        self.config = config

    async def create_space(self, config: dict[str, Any] | None = None) -> MeetingSpace | None:
        """Create a new Google Meet space."""
        try:
            space_config = config or {
                "access_type": "OPEN",
                "entry_point_access": "ALL",
            }

            space = self.service.spaces().create(body={"config": space_config}).execute()

            meeting_space = MeetingSpace(
                name=space["name"],
                meeting_uri=space["meetingUri"],
                meeting_code=space["meetingCode"],
                config=space["config"],
                created_time=space.get("createTime"),
            )

            logger.info(f"Created Google Meet space: {meeting_space.meeting_code}")
            return meeting_space

        except HttpError as e:
            logger.error(f"Failed to create Google Meet space: {e}")
            return None

    async def get_space(self, space_name: str) -> MeetingSpace | None:
        """Get existing Google Meet space."""
        try:
            space = self.service.spaces().get(name=space_name).execute()

            return MeetingSpace(
                name=space["name"],
                meeting_uri=space["meetingUri"],
                meeting_code=space["meetingCode"],
                config=space["config"],
                active_conference=space.get("activeConference"),
                created_time=space.get("createTime"),
                end_time=space.get("endTime"),
            )

        except HttpError as e:
            logger.error(f"Failed to get Google Meet space {space_name}: {e}")
            return None

    async def end_active_conference(self, space_name: str) -> bool:
        """End active conference in a space."""
        try:
            self.service.spaces().endActiveConference(name=space_name).execute()
            logger.info(f"Ended active conference in space: {space_name}")
            return True

        except HttpError as e:
            logger.error(f"Failed to end conference in space {space_name}: {e}")
            return False


class GoogleMeetConferenceManager:
    """Manages Google Meet conference records."""

    def __init__(self, service, config: GoogleMeetConfig):
        self.service = service
        self.config = config

    async def get_conference_record(self, conference_name: str) -> ConferenceRecord | None:
        """Get conference record information."""
        try:
            conference = self.service.conferenceRecords().get(name=conference_name).execute()

            return ConferenceRecord(
                name=conference["name"],
                start_time=conference["startTime"],
                end_time=conference.get("endTime"),
                expire_time=conference.get("expireTime"),
                space=conference.get("space"),
            )

        except HttpError as e:
            logger.error(f"Failed to get conference record {conference_name}: {e}")
            return None

    async def list_conference_records(self, filter_query: str | None = None) -> list[ConferenceRecord]:
        """List conference records with optional filtering."""
        try:
            request = self.service.conferenceRecords().list()
            if filter_query:
                request = request.filter(filter_query)

            conferences = []
            while request is not None:
                response = request.execute()

                for conference in response.get("conferenceRecords", []):
                    conferences.append(
                        ConferenceRecord(
                            name=conference["name"],
                            start_time=conference["startTime"],
                            end_time=conference.get("endTime"),
                            expire_time=conference.get("expireTime"),
                            space=conference.get("space"),
                        )
                    )

                request = self.service.conferenceRecords().list_next(request, response)

            return conferences

        except HttpError as e:
            logger.error(f"Failed to list conference records: {e}")
            return []

    async def get_participants(self, conference_name: str) -> list[Participant]:
        """Get participants for a conference."""
        try:
            request = self.service.conferenceRecords().participants().list(parent=conference_name)

            participants = []
            while request is not None:
                response = request.execute()

                for participant in response.get("participants", []):
                    participants.append(
                        Participant(
                            name=participant["name"],
                            earliest_start_time=participant.get("earliestStartTime"),
                            latest_end_time=participant.get("latestEndTime"),
                            participant_id=participant.get("signedinUser", {}).get("user"),
                            display_name=participant.get("signedinUser", {}).get("displayName")
                            or participant.get("anonymousUser", {}).get("displayName"),
                        )
                    )

                request = (
                    self.service.conferenceRecords().participants().list_next(request, response)
                )

            return participants

        except HttpError as e:
            logger.error(f"Failed to get participants for {conference_name}: {e}")
            return []


class GoogleMeetTranscriptManager:
    """Manages Google Meet transcripts."""

    def __init__(self, service, config: GoogleMeetConfig):
        self.service = service
        self.config = config

    async def list_transcripts(self, conference_name: str) -> list[str]:
        """List available transcripts for a conference."""
        try:
            response = (
                self.service.conferenceRecords()
                .transcripts()
                .list(parent=conference_name)
                .execute()
            )

            return [transcript["name"] for transcript in response.get("transcripts", [])]

        except HttpError as e:
            logger.error(f"Failed to list transcripts for {conference_name}: {e}")
            return []

    async def get_transcript_entries(self, transcript_name: str) -> list[TranscriptEntry]:
        """Get transcript entries."""
        try:
            request = (
                self.service.conferenceRecords()
                .transcripts()
                .entries()
                .list(parent=transcript_name)
            )

            entries = []
            while request is not None:
                response = request.execute()

                for entry in response.get("entries", []):
                    entries.append(
                        TranscriptEntry(
                            name=entry["name"],
                            participant=entry.get("participant"),
                            text=entry.get("text"),
                            language_code=entry.get("languageCode"),
                            start_time=entry.get("startTime"),
                            end_time=entry.get("endTime"),
                        )
                    )

                request = (
                    self.service.conferenceRecords()
                    .transcripts()
                    .entries()
                    .list_next(request, response)
                )

            return entries

        except HttpError as e:
            logger.error(f"Failed to get transcript entries for {transcript_name}: {e}")
            return []


class GoogleMeetClient:
    """
    Main Google Meet API client for the orchestration service.

    Provides comprehensive access to Google Meet API functionality
    with integration support for the GoogleMeetBotManager.
    """

    def __init__(self, config: GoogleMeetConfig):
        self.config = config

        # Authentication
        self.authenticator = GoogleMeetAuthenticator(config)

        # Service managers (initialized after authentication)
        self.space_manager = None
        self.conference_manager = None
        self.transcript_manager = None

        # State tracking
        self.authenticated = False
        self.monitored_spaces: dict[str, dict] = {}
        self.active_conferences: dict[str, dict] = {}

        # Background task tracking (prevents garbage collection and enables cleanup)
        self._background_tasks: set[asyncio.Task] = set()

        logger.info("GoogleMeetClient initialized")
        logger.info(f"  Application: {config.application_name}")
        logger.info(f"  API Version: {config.api_version}")

    async def initialize(self) -> bool:
        """Initialize the client with authentication."""
        try:
            # Authenticate
            success = await self.authenticator.authenticate()
            if not success:
                return False

            # Initialize service managers
            service = self.authenticator.service
            self.space_manager = GoogleMeetSpaceManager(service, self.config)
            self.conference_manager = GoogleMeetConferenceManager(service, self.config)
            self.transcript_manager = GoogleMeetTranscriptManager(service, self.config)

            self.authenticated = True
            logger.info("Google Meet client initialized successfully")
            return True

        except Exception as e:
            logger.error(f"Failed to initialize Google Meet client: {e}")
            return False

    def is_ready(self) -> bool:
        """Check if client is ready for use."""
        return (
            self.authenticated
            and self.authenticator.is_authenticated()
            and self.space_manager is not None
        )

    async def create_meeting_space(
        self, meeting_config: dict[str, Any] | None = None
    ) -> MeetingSpace | None:
        """Create a new meeting space for bot integration."""
        if not self.is_ready():
            logger.error("Google Meet client not ready")
            return None

        return await self.space_manager.create_space(meeting_config)

    async def join_existing_meeting(self, meeting_uri: str) -> dict[str, Any] | None:
        """
        Get information about an existing meeting for bot integration.

        Note: The Google Meet API doesn't support programmatic joining of meetings.
        This method provides meeting information for manual or browser-based joining.
        """
        try:
            # Parse meeting URI to extract meeting code
            meeting_code = self._extract_meeting_code(meeting_uri)
            if not meeting_code:
                logger.error(f"Invalid meeting URI: {meeting_uri}")
                return None

            # Try to find the space by meeting code
            # Note: This requires the space to be created through our API
            spaces = await self._search_spaces_by_code(meeting_code)

            if spaces:
                space = spaces[0]
                return {
                    "space": space,
                    "meeting_uri": meeting_uri,
                    "meeting_code": meeting_code,
                    "join_method": "browser_automation",  # Requires browser automation
                    "api_access": "limited",  # Limited to conference records after meeting
                }
            else:
                return {
                    "meeting_uri": meeting_uri,
                    "meeting_code": meeting_code,
                    "join_method": "browser_automation",
                    "api_access": "post_meeting_only",  # Only conference records after meeting
                    "note": "Meeting not created through API - limited access",
                }

        except Exception as e:
            logger.error(f"Error processing meeting URI {meeting_uri}: {e}")
            return None

    async def monitor_active_conference(self, space_name: str, callback: Callable | None = None) -> bool:
        """Monitor an active conference for events."""
        try:
            if not self.is_ready():
                return False

            space = await self.space_manager.get_space(space_name)
            if not space or not space.active_conference:
                logger.warning(f"No active conference in space: {space_name}")
                return False

            # Store monitoring info
            self.monitored_spaces[space_name] = {
                "space": space,
                "callback": callback,
                "start_time": time.time(),
                "last_check": time.time(),
            }

            # Start monitoring task
            task = asyncio.create_task(
                self._monitor_conference_loop(space_name)
            )
            self._background_tasks.add(task)
            task.add_done_callback(self._background_tasks.discard)

            logger.info(f"Started monitoring conference: {space.active_conference}")
            return True

        except Exception as e:
            logger.error(f"Failed to monitor conference in space {space_name}: {e}")
            return False

    async def get_live_participants(self, conference_name: str) -> list[Participant]:
        """Get current participants in an active conference."""
        if not self.is_ready():
            return []

        return await self.conference_manager.get_participants(conference_name)

    async def get_meeting_transcript(self, conference_name: str) -> list[TranscriptEntry]:
        """Get transcript entries for a conference (available after meeting ends)."""
        if not self.is_ready():
            return []

        try:
            # List available transcripts
            transcripts = await self.transcript_manager.list_transcripts(conference_name)

            if not transcripts:
                logger.info(f"No transcripts available for conference: {conference_name}")
                return []

            # Get entries from the first available transcript
            transcript_name = transcripts[0]
            entries = await self.transcript_manager.get_transcript_entries(transcript_name)

            logger.info(f"Retrieved {len(entries)} transcript entries from {transcript_name}")
            return entries

        except Exception as e:
            logger.error(f"Error getting transcript for {conference_name}: {e}")
            return []

    async def end_meeting(self, space_name: str) -> bool:
        """End an active meeting."""
        if not self.is_ready():
            return False

        success = await self.space_manager.end_active_conference(space_name)

        # Stop monitoring if active
        if space_name in self.monitored_spaces:
            del self.monitored_spaces[space_name]

        return success

    async def _monitor_conference_loop(self, space_name: str):
        """Background loop for monitoring conference events."""
        try:
            while space_name in self.monitored_spaces:
                monitor_info = self.monitored_spaces[space_name]

                # Check space status
                space = await self.space_manager.get_space(space_name)
                if not space or not space.active_conference:
                    logger.info(f"Conference ended in space: {space_name}")
                    break

                # Get current participants
                participants = await self.conference_manager.get_participants(
                    space.active_conference
                )

                # Call callback if provided
                if monitor_info["callback"]:
                    try:
                        await monitor_info["callback"](
                            {
                                "event_type": "participant_update",
                                "space": space,
                                "participants": participants,
                                "timestamp": time.time(),
                            }
                        )
                    except Exception as e:
                        logger.error(f"Error in monitor callback: {e}")

                # Update last check time
                monitor_info["last_check"] = time.time()

                # Wait before next check
                await asyncio.sleep(30)  # Check every 30 seconds

        except Exception as e:
            logger.error(f"Error in conference monitoring loop: {e}")
        finally:
            # Cleanup
            if space_name in self.monitored_spaces:
                del self.monitored_spaces[space_name]

    def _extract_meeting_code(self, meeting_uri: str) -> str | None:
        """Extract meeting code from Google Meet URI."""
        try:
            # Handle various Google Meet URI formats
            if "meet.google.com/" in meeting_uri:
                return meeting_uri.split("meet.google.com/")[-1].split("?")[0]
            return None
        except Exception:
            return None

    async def _search_spaces_by_code(self, meeting_code: str) -> list[MeetingSpace]:
        """Search for spaces by meeting code (limited API functionality)."""
        # Note: Google Meet API doesn't provide direct search by meeting code
        # This would require maintaining our own mapping of created spaces
        return []

    def get_client_statistics(self) -> dict[str, Any]:
        """Get comprehensive client statistics."""
        return {
            "authenticated": self.authenticated,
            "api_ready": self.is_ready(),
            "monitored_spaces": len(self.monitored_spaces),
            "active_conferences": len(self.active_conferences),
            "config": {
                "application_name": self.config.application_name,
                "api_version": self.config.api_version,
                "timeout": self.config.timeout,
            },
            "credentials_valid": self.authenticator.is_authenticated()
            if self.authenticator
            else False,
        }


# Factory functions
def create_google_meet_client(credentials_path: str, **config_kwargs) -> GoogleMeetClient:
    """Create a Google Meet client with configuration."""
    config = GoogleMeetConfig(credentials_path=credentials_path, **config_kwargs)
    return GoogleMeetClient(config)


# Integration helper for GoogleMeetBotManager
class BotManagerIntegration:
    """Integration helper for GoogleMeetBotManager."""

    def __init__(self, meet_client: GoogleMeetClient, bot_manager):
        self.meet_client = meet_client
        self.bot_manager = bot_manager

    async def create_bot_meeting(self, meeting_request) -> dict[str, Any] | None:
        """Create a meeting space for a bot."""
        space = await self.meet_client.create_meeting_space()
        if space:
            return {
                "space": space,
                "meeting_info": {
                    "meeting_id": space.meeting_code,
                    "meeting_uri": space.meeting_uri,
                    "space_name": space.name,
                },
            }
        return None

    async def join_external_meeting(self, meeting_uri: str) -> dict[str, Any] | None:
        """Get information for joining an external meeting."""
        return await self.meet_client.join_existing_meeting(meeting_uri)

    async def monitor_bot_meeting(self, space_name: str, bot_id: str) -> bool:
        """Monitor a meeting for a specific bot."""

        async def bot_callback(event_data):
            # Forward events to bot manager
            if hasattr(self.bot_manager, "handle_meeting_event"):
                await self.bot_manager.handle_meeting_event(bot_id, event_data)

        return await self.meet_client.monitor_active_conference(space_name, bot_callback)


# Example usage
async def main():
    """Example usage of the Google Meet client."""
    # Create client
    client = create_google_meet_client(
        credentials_path="path/to/credentials.json",
        application_name="LiveTranslate Bot System",
    )

    # Initialize
    success = await client.initialize()
    if not success:
        print("Failed to initialize Google Meet client")
        return

    # Create a meeting space
    space = await client.create_meeting_space()
    if space:
        print(f"Created meeting: {space.meeting_uri}")
        print(f"Meeting code: {space.meeting_code}")

        # Monitor the meeting
        def meeting_callback(event):
            print(f"Meeting event: {event}")

        await client.monitor_active_conference(space.name, meeting_callback)

        # Simulate meeting duration
        await asyncio.sleep(60)

        # End meeting
        await client.end_meeting(space.name)

    # Get client statistics
    stats = client.get_client_statistics()
    print(f"Client statistics: {json.dumps(stats, indent=2, default=str)}")


if __name__ == "__main__":
    asyncio.run(main())
