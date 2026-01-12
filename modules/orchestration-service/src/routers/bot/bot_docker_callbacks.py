#!/usr/bin/env python3
"""
Bot Docker Callback API Router

Handles HTTP callbacks from bot containers.

Bot containers send status updates via HTTP POST:
- POST /callback/started
- POST /callback/joining
- POST /callback/active
- POST /callback/completed
- POST /callback/failed

Moved from standalone routers/bot_callbacks.py for package consolidation.
"""

import logging
from typing import Dict, Any, Optional
from fastapi import APIRouter, HTTPException, Depends
from pydantic import BaseModel, Field

from bot.docker_bot_manager import get_bot_manager, DockerBotManager

logger = logging.getLogger(__name__)

router = APIRouter(tags=["bot-docker-callbacks"])


class BotCallbackPayload(BaseModel):
    """Payload from bot container callback"""

    connection_id: str = Field(..., description="Bot connection ID")
    container_id: str = Field(..., description="Docker container ID")
    error: Optional[str] = Field(None, description="Error message (for failed status)")
    exit_code: Optional[int] = Field(None, description="Exit code (for failed status)")
    metadata: Optional[Dict[str, Any]] = Field(
        default_factory=dict, description="Additional metadata"
    )


@router.post("/started")
async def bot_started_callback(
    payload: BotCallbackPayload, manager: DockerBotManager = Depends(get_bot_manager)
):
    """
    Handle bot 'started' callback

    Bot container sends this after:
    1. Container started
    2. Connected to orchestration WebSocket

    Next: Bot will send 'joining' callback
    """
    logger.info(
        f"Bot {payload.connection_id} started (container: {payload.container_id})"
    )

    try:
        await manager.handle_bot_callback(
            connection_id=payload.connection_id,
            status="started",
            data={"container_id": payload.container_id, **payload.metadata},
        )

        return {
            "status": "success",
            "message": f"Bot {payload.connection_id} started callback received",
        }

    except Exception as e:
        logger.error(f"Error handling started callback: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/joining")
async def bot_joining_callback(
    payload: BotCallbackPayload, manager: DockerBotManager = Depends(get_bot_manager)
):
    """
    Handle bot 'joining' callback

    Bot container sends this when:
    - About to join Google Meet

    Next: Bot will send 'active' callback
    """
    logger.info(f"Bot {payload.connection_id} joining meeting")

    try:
        await manager.handle_bot_callback(
            connection_id=payload.connection_id,
            status="joining",
            data={"container_id": payload.container_id, **payload.metadata},
        )

        return {
            "status": "success",
            "message": f"Bot {payload.connection_id} joining callback received",
        }

    except Exception as e:
        logger.error(f"Error handling joining callback: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/active")
async def bot_active_callback(
    payload: BotCallbackPayload, manager: DockerBotManager = Depends(get_bot_manager)
):
    """
    Handle bot 'active' callback

    Bot container sends this when:
    - Successfully joined Google Meet
    - Audio streaming active

    Bot is now fully operational
    """
    logger.info(f"Bot {payload.connection_id} active in meeting")

    try:
        await manager.handle_bot_callback(
            connection_id=payload.connection_id,
            status="active",
            data={"container_id": payload.container_id, **payload.metadata},
        )

        return {
            "status": "success",
            "message": f"Bot {payload.connection_id} active callback received",
        }

    except Exception as e:
        logger.error(f"Error handling active callback: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/completed")
async def bot_completed_callback(
    payload: BotCallbackPayload, manager: DockerBotManager = Depends(get_bot_manager)
):
    """
    Handle bot 'completed' callback

    Bot container sends this on clean exit:
    - Received leave command
    - Successfully left meeting
    - Cleaned up resources
    """
    logger.info(f"Bot {payload.connection_id} completed")

    try:
        await manager.handle_bot_callback(
            connection_id=payload.connection_id,
            status="completed",
            data={"container_id": payload.container_id, **payload.metadata},
        )

        return {
            "status": "success",
            "message": f"Bot {payload.connection_id} completed callback received",
        }

    except Exception as e:
        logger.error(f"Error handling completed callback: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/failed")
async def bot_failed_callback(
    payload: BotCallbackPayload, manager: DockerBotManager = Depends(get_bot_manager)
):
    """
    Handle bot 'failed' callback

    Bot container sends this on error:
    - Failed to join meeting
    - Audio capture error
    - Orchestration disconnect
    - Unexpected error
    """
    logger.error(f"Bot {payload.connection_id} failed: {payload.error}")

    try:
        await manager.handle_bot_callback(
            connection_id=payload.connection_id,
            status="failed",
            data={
                "container_id": payload.container_id,
                "error": payload.error,
                "exit_code": payload.exit_code,
                **payload.metadata,
            },
        )

        return {
            "status": "success",
            "message": f"Bot {payload.connection_id} failed callback received",
        }

    except Exception as e:
        logger.error(f"Error handling failed callback: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))
