#!/usr/bin/env python3
"""
Docker Bot Management API Router

Public API for Docker-based bot management operations:
- Start bot (join meeting)
- Stop bot (leave meeting)
- Get bot status
- List bots
- Send commands

Moved from standalone routers/bot_management.py for package consolidation.
"""

import logging
from typing import Any

from bot.docker_bot_manager import BotStatus, DockerBotManager, get_bot_manager
from fastapi import APIRouter, Depends, HTTPException, Query
from pydantic import BaseModel, Field, HttpUrl

logger = logging.getLogger(__name__)

router = APIRouter(tags=["bot-docker-management"])


class StartBotRequest(BaseModel):
    """Request to start a bot"""

    meeting_url: HttpUrl = Field(..., description="Google Meet URL")
    user_token: str = Field(..., description="User API token for orchestration auth")
    user_id: str = Field(..., description="User ID")
    language: str = Field("en", description="Transcription language (en, es, fr, de, zh, etc.)")
    task: str = Field("transcribe", description="Task: transcribe or translate")
    enable_virtual_webcam: bool = Field(False, description="Enable virtual webcam output")
    metadata: dict[str, Any] | None = Field(default_factory=dict, description="Additional metadata")


class StartBotResponse(BaseModel):
    """Response from starting a bot"""

    connection_id: str = Field(..., description="Bot connection ID")
    status: str = Field(..., description="Bot status")
    message: str = Field(..., description="Status message")


class StopBotRequest(BaseModel):
    """Request to stop a bot"""

    timeout: int = Field(30, description="Timeout in seconds")


class BotStatusResponse(BaseModel):
    """Bot status response"""

    connection_id: str
    user_id: str
    meeting_url: str
    status: str
    created_at: float
    started_at: float | None
    active_at: float | None
    stopped_at: float | None
    uptime_seconds: float
    is_healthy: bool
    container_id: str | None
    container_name: str | None
    error_message: str | None
    metadata: dict[str, Any]


class BotListResponse(BaseModel):
    """List of bots response"""

    bots: list[BotStatusResponse]
    total: int


class BotCommandRequest(BaseModel):
    """Request to send command to bot"""

    action: str = Field(..., description="Command action (leave, reconfigure, status)")
    data: dict[str, Any] | None = Field(default_factory=dict, description="Command data")


@router.post("/start", response_model=StartBotResponse)
async def start_bot(request: StartBotRequest, manager: DockerBotManager = Depends(get_bot_manager)):
    """
    Start a bot to join Google Meet

    The bot will:
    1. Join the specified Google Meet URL
    2. Capture and stream audio to orchestration service
    3. Receive transcription segments back
    4. Optionally display translations on virtual webcam

    Returns:
        connection_id: Unique bot ID for tracking
    """
    logger.info(f"Starting bot for meeting: {request.meeting_url}")

    try:
        connection_id = await manager.start_bot(
            meeting_url=str(request.meeting_url),
            user_token=request.user_token,
            user_id=request.user_id,
            language=request.language,
            task=request.task,
            enable_virtual_webcam=request.enable_virtual_webcam,
            metadata=request.metadata,
        )

        return StartBotResponse(
            connection_id=connection_id,
            status="spawning",
            message=f"Bot {connection_id} is starting. It will send callbacks as it progresses.",
        )

    except Exception as e:
        logger.error(f"Failed to start bot: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Failed to start bot: {e!s}") from e


@router.post("/stop/{connection_id}")
async def stop_bot(
    connection_id: str,
    request: StopBotRequest = StopBotRequest(),
    manager: DockerBotManager = Depends(get_bot_manager),
):
    """
    Stop a bot (leave meeting)

    Sends a leave command via Redis to the bot container.
    Bot will:
    1. Stop audio capture
    2. Leave Google Meet
    3. Send 'completed' callback
    4. Exit cleanly
    """
    logger.info(f"Stopping bot: {connection_id}")

    try:
        await manager.stop_bot(connection_id, timeout=request.timeout)

        return {
            "status": "success",
            "message": f"Stop command sent to bot {connection_id}",
        }

    except ValueError as e:
        raise HTTPException(status_code=404, detail=str(e)) from e
    except Exception as e:
        logger.error(f"Failed to stop bot: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Failed to stop bot: {e!s}") from e


@router.get("/status/{connection_id}", response_model=BotStatusResponse)
async def get_bot_status(connection_id: str, manager: DockerBotManager = Depends(get_bot_manager)):
    """
    Get bot status

    Returns:
        Bot instance with full status information
    """
    bot = manager.get_bot(connection_id)

    if not bot:
        raise HTTPException(status_code=404, detail=f"Bot not found: {connection_id}")

    return BotStatusResponse(**bot.to_dict())


@router.get("/list", response_model=BotListResponse)
async def list_bots(
    status: str | None = Query(None, description="Filter by status"),
    user_id: str | None = Query(None, description="Filter by user ID"),
    manager: DockerBotManager = Depends(get_bot_manager),
):
    """
    List bots with optional filters

    Query parameters:
    - status: Filter by bot status (spawning, active, completed, failed, etc.)
    - user_id: Filter by user ID
    """
    # Parse status filter
    status_enum = None
    if status:
        try:
            status_enum = BotStatus(status)
        except ValueError as e:
            raise HTTPException(
                status_code=400,
                detail=f"Invalid status: {status}. Valid values: {[s.value for s in BotStatus]}",
            ) from e

    # List bots
    bots = manager.list_bots(status=status_enum, user_id=user_id)

    return BotListResponse(
        bots=[BotStatusResponse(**bot.to_dict()) for bot in bots], total=len(bots)
    )


@router.post("/command/{connection_id}")
async def send_bot_command(
    connection_id: str,
    request: BotCommandRequest,
    manager: DockerBotManager = Depends(get_bot_manager),
):
    """
    Send command to bot via Redis

    Supported commands:
    - leave: Stop bot and leave meeting
    - reconfigure: Update bot configuration (language, task)
    - status: Request status update

    Example:
    ```json
    {
        "action": "reconfigure",
        "data": {
            "language": "es",
            "task": "translate"
        }
    }
    ```
    """
    logger.info(f"Sending command to bot {connection_id}: {request.action}")

    try:
        command = {"action": request.action, **request.data}
        await manager.send_command(connection_id, command)

        return {
            "status": "success",
            "message": f"Command '{request.action}' sent to bot {connection_id}",
        }

    except ValueError as e:
        raise HTTPException(status_code=404, detail=str(e)) from e
    except RuntimeError as e:
        raise HTTPException(status_code=503, detail=str(e)) from e
    except Exception as e:
        logger.error(f"Failed to send command: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Failed to send command: {e!s}") from e


@router.get("/stats")
async def get_manager_stats(manager: DockerBotManager = Depends(get_bot_manager)):
    """
    Get bot manager statistics

    Returns:
        - Total bots
        - Active bots
        - Success/failure rates
        - Bots by status
    """
    return manager.get_stats()
