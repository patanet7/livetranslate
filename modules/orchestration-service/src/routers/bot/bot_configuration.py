"""
Bot Configuration Management Router

Bot configuration endpoints including:
- Bot config updates (/config)
- Bot config retrieval (/config)
"""

from typing import Any

from dependencies import get_bot_manager
from fastapi import Depends, HTTPException, status

from ._shared import (
    BotConfigUpdateRequest,
    create_bot_router,
)

# Create router for bot configuration management
router = create_bot_router()


@router.post("/{bot_id}/config")
async def update_bot_config(
    bot_id: str, request: BotConfigUpdateRequest, bot_manager=Depends(get_bot_manager)
) -> dict[str, Any]:
    """
    Update bot configuration

    Updates the configuration of a running bot. Some settings may require
    bot restart to take effect.
    """
    raise HTTPException(
        status_code=status.HTTP_501_NOT_IMPLEMENTED,
        detail="Bot configuration update not yet implemented",
    )


@router.get("/{bot_id}/config")
async def get_bot_config(bot_id: str, bot_manager=Depends(get_bot_manager)) -> dict[str, Any]:
    """
    Get bot configuration

    Returns the current configuration of a bot instance.
    """
    raise HTTPException(
        status_code=status.HTTP_501_NOT_IMPLEMENTED,
        detail="Bot configuration retrieval not yet implemented",
    )
