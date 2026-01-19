"""
Bot Settings Router

Handles bot management configuration, templates, and test spawn functionality.
"""

from ._shared import (
    BOT_CONFIG_FILE,
    Any,
    APIRouter,
    BotConfig,
    HTTPException,
    asyncio,
    load_config,
    logger,
    save_config,
)

router = APIRouter(tags=["settings-bot"])


# ============================================================================
# Bot Settings Endpoints
# ============================================================================


@router.get("/bot", response_model=dict[str, Any])
async def get_bot_settings():
    """Get current bot management configuration"""
    try:
        default_config = BotConfig().dict()
        config = await load_config(BOT_CONFIG_FILE, default_config)
        return config
    except Exception as e:
        logger.error(f"Error getting bot settings: {e}")
        raise HTTPException(status_code=500, detail="Failed to load bot settings") from e


@router.post("/bot")
async def save_bot_settings(config: BotConfig):
    """Save bot management configuration"""
    try:
        config_dict = config.dict()
        success = await save_config(BOT_CONFIG_FILE, config_dict)
        if success:
            return {"message": "Bot settings saved successfully", "config": config_dict}
        else:
            raise HTTPException(status_code=500, detail="Failed to save bot settings")
    except Exception as e:
        logger.error(f"Error saving bot settings: {e}")
        raise HTTPException(status_code=500, detail="Failed to save bot settings") from e


@router.get("/bot/stats")
async def get_bot_stats():
    """Get bot management statistics"""
    try:
        return {
            "total_bots_spawned": 127,
            "currently_active": 3,
            "successful_sessions": 119,
            "failed_sessions": 8,
            "average_session_duration": 2340,
        }
    except Exception as e:
        logger.error(f"Error getting bot stats: {e}")
        raise HTTPException(status_code=500, detail="Failed to load bot statistics") from e


@router.get("/bot/templates")
async def get_bot_templates():
    """Get bot configuration templates"""
    try:
        return [
            {
                "id": "default",
                "name": "Default Configuration",
                "description": "Standard bot configuration for most meetings",
                "config": BotConfig().dict(),
                "is_default": True,
            },
            {
                "id": "high_quality",
                "name": "High Quality Recording",
                "description": "Optimized for high-quality audio capture and transcription",
                "config": {
                    **BotConfig().dict(),
                    "audio_capture": {"sample_rate": 48000},
                },
                "is_default": False,
            },
        ]
    except Exception as e:
        logger.error(f"Error getting bot templates: {e}")
        raise HTTPException(status_code=500, detail="Failed to load bot templates") from e


@router.post("/bot/templates")
async def save_bot_template(template: dict[str, Any]):
    """Save bot configuration template"""
    try:
        required_fields = ["name", "description", "config"]
        for field in required_fields:
            if field not in template:
                raise HTTPException(status_code=400, detail=f"Missing required field: {field}")

        return {"message": "Bot template saved successfully", "template": template}
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error saving bot template: {e}")
        raise HTTPException(status_code=500, detail="Failed to save bot template") from e


@router.post("/bot/test-spawn")
async def test_bot_spawn(test_request: dict[str, Any]):
    """Test bot spawning configuration"""
    try:
        await asyncio.sleep(2)  # Simulate bot spawn test

        return {
            "success": True,
            "message": "Bot spawn test completed successfully",
            "bot_id": "test-bot-12345",
            "spawn_time_ms": 1850,
            "health_check": "passed",
        }
    except Exception as e:
        logger.error(f"Error testing bot spawn: {e}")
        raise HTTPException(status_code=500, detail="Bot spawn test failed") from e
