"""
Meeting Bot Service Client

Python client for calling the Node.js meeting-bot-service API.
Uses ScreenApp's battle-tested GoogleMeetBot via HTTP.
"""

import logging
from typing import Dict, Any, Optional
import httpx
from pydantic import BaseModel


class JoinRequest(BaseModel):
    """Request to join a meeting"""
    meetingUrl: str
    botName: str
    botId: str
    userId: str
    teamId: Optional[str] = "livetranslate-team"
    timezone: Optional[str] = "UTC"
    eventId: Optional[str] = None
    bearerToken: Optional[str] = None


class JoinResponse(BaseModel):
    """Response from join request"""
    success: bool
    botId: str
    correlationId: str
    message: Optional[str] = None
    error: Optional[str] = None


class BotStatusResponse(BaseModel):
    """Response from status check"""
    success: bool
    botId: Optional[str] = None
    state: Optional[str] = None
    error: Optional[str] = None


class MeetingBotServiceClient:
    """
    Client for the meeting-bot-service HTTP API.

    This service runs the battle-tested ScreenApp GoogleMeetBot that successfully
    bypasses Google's bot detection.
    """

    def __init__(
        self,
        base_url: str = "http://localhost:5005",
        timeout: float = 30.0
    ):
        """
        Initialize the meeting bot service client.

        Args:
            base_url: Base URL of the meeting-bot-service
            timeout: HTTP request timeout in seconds
        """
        self.base_url = base_url.rstrip("/")
        self.timeout = timeout
        self.logger = logging.getLogger(__name__)

    async def join_meeting(
        self,
        meeting_url: str,
        bot_name: str,
        bot_id: str,
        user_id: str,
        team_id: str = "livetranslate-team",
        timezone: str = "UTC",
        event_id: Optional[str] = None,
        bearer_token: Optional[str] = None
    ) -> JoinResponse:
        """
        Request a bot to join a Google Meet meeting.

        Args:
            meeting_url: Google Meet URL
            bot_name: Display name for the bot
            bot_id: Unique bot identifier
            user_id: User identifier
            team_id: Team identifier
            timezone: Timezone for the bot
            event_id: Event identifier
            bearer_token: Authentication token

        Returns:
            JoinResponse with success status and bot information

        Raises:
            httpx.HTTPError: If the HTTP request fails
        """
        request = JoinRequest(
            meetingUrl=meeting_url,
            botName=bot_name,
            botId=bot_id,
            userId=user_id,
            teamId=team_id,
            timezone=timezone,
            eventId=event_id,
            bearerToken=bearer_token
        )

        self.logger.info(
            f"Requesting bot to join meeting: {meeting_url}",
            extra={
                "bot_id": bot_id,
                "bot_name": bot_name,
                "user_id": user_id
            }
        )

        async with httpx.AsyncClient(timeout=self.timeout) as client:
            response = await client.post(
                f"{self.base_url}/api/bot/join",
                json=request.model_dump(exclude_none=True)
            )
            response.raise_for_status()

            data = response.json()
            result = JoinResponse(**data)

            if result.success:
                self.logger.info(
                    f"Bot join request successful: {result.correlationId}",
                    extra={"bot_id": bot_id}
                )
            else:
                self.logger.error(
                    f"Bot join request failed: {result.error}",
                    extra={"bot_id": bot_id}
                )

            return result

    async def get_bot_status(self, bot_id: str) -> BotStatusResponse:
        """
        Get the status of a bot.

        Args:
            bot_id: Bot identifier

        Returns:
            BotStatusResponse with bot state

        Raises:
            httpx.HTTPError: If the HTTP request fails
        """
        async with httpx.AsyncClient(timeout=self.timeout) as client:
            response = await client.get(
                f"{self.base_url}/api/bot/status/{bot_id}"
            )
            response.raise_for_status()

            data = response.json()
            return BotStatusResponse(**data)

    async def leave_meeting(self, bot_id: str) -> Dict[str, Any]:
        """
        Request a bot to leave a meeting.

        Args:
            bot_id: Bot identifier

        Returns:
            Response dict with success status

        Raises:
            httpx.HTTPError: If the HTTP request fails
        """
        self.logger.info(f"Requesting bot to leave meeting", extra={"bot_id": bot_id})

        async with httpx.AsyncClient(timeout=self.timeout) as client:
            response = await client.post(
                f"{self.base_url}/api/bot/leave/{bot_id}"
            )
            response.raise_for_status()

            data = response.json()

            if data.get("success"):
                self.logger.info(
                    "Bot leave request successful",
                    extra={"bot_id": bot_id}
                )
            else:
                self.logger.error(
                    f"Bot leave request failed: {data.get('error')}",
                    extra={"bot_id": bot_id}
                )

            return data

    async def health_check(self) -> Dict[str, Any]:
        """
        Check if the meeting-bot-service is healthy.

        Returns:
            Health check response

        Raises:
            httpx.HTTPError: If the HTTP request fails
        """
        async with httpx.AsyncClient(timeout=10.0) as client:
            response = await client.get(f"{self.base_url}/api/health")
            response.raise_for_status()
            return response.json()
