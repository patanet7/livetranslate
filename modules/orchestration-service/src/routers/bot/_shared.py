"""
Shared components for bot router modules

Common imports, utilities, and configurations used across all bot router components.
"""

import logging
from typing import Dict, Any, Optional, List
from datetime import datetime

from fastapi import APIRouter, HTTPException, status, Depends
from fastapi.responses import JSONResponse, StreamingResponse
from pydantic import BaseModel

# Dependencies
from dependencies import get_bot_manager

# Models
from models.bot import (
    BotSpawnRequest,
    BotResponse,
    BotInstance,
    BotStats,
    BotConfiguration,
)

# Shared logger
logger = logging.getLogger(__name__)


# Shared response models
class BotListResponse(BaseModel):
    """Response model for bot list"""
    bots: List[BotResponse]
    total: int
    active: int
    inactive: int


class BotConfigUpdateRequest(BaseModel):
    """Request model for bot configuration update"""
    bot_id: str
    config: Dict[str, Any]


class BotAnalyticsResponse(BaseModel):
    """Response model for bot analytics"""
    bot_id: str
    analytics: Dict[str, Any]
    timestamp: str


class SystemStatsResponse(BaseModel):
    """Response model for system statistics"""
    stats: Dict[str, Any]
    timestamp: str


class VirtualWebcamConfigRequest(BaseModel):
    """Request model for virtual webcam configuration"""
    bot_id: str
    config: Dict[str, Any]


def create_bot_router(prefix: str = "") -> APIRouter:
    """Create a standardized bot router with common configuration."""
    return APIRouter(
        prefix=prefix,
        tags=["bots"],
        responses={
            404: {"description": "Bot not found"},
            422: {"description": "Validation error"},
            500: {"description": "Internal server error"}
        }
    )


def get_error_response(status_code: int, message: str, details: Optional[Dict[str, Any]] = None) -> HTTPException:
    """Create standardized error response"""
    error_detail = {"message": message}
    if details:
        error_detail["details"] = details
    
    return HTTPException(
        status_code=status_code,
        detail=error_detail
    )


async def validate_bot_exists(bot_id: str, bot_manager) -> BotInstance:
    """Validate that a bot exists and return its instance"""
    try:
        bot_instance = await bot_manager.get_bot(bot_id)
        if not bot_instance:
            raise get_error_response(
                status.HTTP_404_NOT_FOUND,
                f"Bot {bot_id} not found",
                {"bot_id": bot_id}
            )
        return bot_instance
    except Exception as e:
        logger.error(f"Error validating bot {bot_id}: {e}")
        raise get_error_response(
            status.HTTP_500_INTERNAL_SERVER_ERROR,
            f"Failed to validate bot: {str(e)}",
            {"bot_id": bot_id}
        )