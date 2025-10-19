"""
Bot Virtual Webcam Management Router

Virtual webcam endpoints including:
- Webcam management (/webcam)
- Webcam status (/webcam/status)
- Virtual webcam frame streaming (/virtual-webcam/frame)
- Virtual webcam configuration (/virtual-webcam/config)
"""

import time
from ._shared import *

# Create router for virtual webcam management
router = create_bot_router()


@router.post("/{bot_id}/webcam")
async def manage_virtual_webcam(
    bot_id: str,
    request: VirtualWebcamConfigRequest,
    bot_manager=Depends(get_bot_manager)
) -> Dict[str, Any]:
    """
    Manage virtual webcam output

    Controls the virtual webcam that displays translation output.
    Actions: start, stop, update settings.
    """
    try:
        logger.info(f"Managing virtual webcam for bot {bot_id}: {request.config.get('action', 'unknown')}")

        # Validate bot exists
        await validate_bot_exists(bot_id, bot_manager)

        # Execute webcam action
        # TODO: Implement virtual webcam methods in bot_manager
        action = request.config.get("action", "update")
        if action in ["start", "stop", "update"]:
            result = {"message": f"Virtual webcam {action} not yet implemented"}
        else:
            raise get_error_response(
                status.HTTP_400_BAD_REQUEST,
                f"Invalid action: {action}",
                {"bot_id": bot_id, "action": action}
            )

        return {
            "message": f"Virtual webcam {action} completed",
            "result": result,
        }

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to manage virtual webcam: {e}")
        raise get_error_response(
            status.HTTP_500_INTERNAL_SERVER_ERROR,
            f"Failed to manage virtual webcam: {str(e)}",
            {"bot_id": bot_id}
        )


@router.get("/{bot_id}/webcam/status")
async def get_webcam_status(
    bot_id: str,
    bot_manager=Depends(get_bot_manager)
) -> Dict[str, Any]:
    """
    Get virtual webcam status

    Returns the current status and configuration of the virtual webcam.
    """
    try:
        # Validate bot exists
        await validate_bot_exists(bot_id, bot_manager)

        # TODO: Implement get_virtual_webcam_status method in bot_manager
        webcam_status = {"message": "Virtual webcam status not yet implemented"}

        return {"bot_id": bot_id, "webcam_status": webcam_status}

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to get webcam status: {e}")
        raise get_error_response(
            status.HTTP_500_INTERNAL_SERVER_ERROR,
            f"Failed to get webcam status: {str(e)}",
            {"bot_id": bot_id}
        )


@router.get("/virtual-webcam/frame/{bot_id}")
async def get_virtual_webcam_frame(
    bot_id: str,
    bot_manager=Depends(get_bot_manager)
) -> Dict[str, Any]:
    """Get current virtual webcam frame as base64 image."""
    try:
        bot_instance = bot_manager.get_bot(bot_id)
        if not bot_instance:
            raise get_error_response(
                status.HTTP_404_NOT_FOUND,
                "Bot not found",
                {"bot_id": bot_id}
            )
            
        # Get current frame from virtual webcam
        if hasattr(bot_instance, 'virtual_webcam') and bot_instance.virtual_webcam:
            frame_base64 = bot_instance.virtual_webcam.get_current_frame_base64()
            if frame_base64:
                return {
                    "bot_id": bot_id,
                    "frame_base64": frame_base64,
                    "timestamp": time.time(),
                    "webcam_stats": bot_instance.virtual_webcam.get_webcam_stats()
                }
            else:
                raise get_error_response(
                    status.HTTP_404_NOT_FOUND,
                    "No frame available",
                    {"bot_id": bot_id}
                )
        else:
            raise get_error_response(
                status.HTTP_404_NOT_FOUND,
                "Virtual webcam not enabled",
                {"bot_id": bot_id}
            )
            
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error getting virtual webcam frame: {e}")
        raise get_error_response(
            status.HTTP_500_INTERNAL_SERVER_ERROR,
            f"Error getting virtual webcam frame: {str(e)}",
            {"bot_id": bot_id}
        )


@router.get("/virtual-webcam/config/{bot_id}")
async def get_virtual_webcam_config(
    bot_id: str,
    bot_manager=Depends(get_bot_manager)
) -> Dict[str, Any]:
    """Get virtual webcam configuration."""
    try:
        bot_instance = bot_manager.get_bot(bot_id)
        if not bot_instance:
            raise get_error_response(
                status.HTTP_404_NOT_FOUND,
                "Bot not found",
                {"bot_id": bot_id}
            )
            
        if hasattr(bot_instance, 'virtual_webcam') and bot_instance.virtual_webcam:
            return {
                "bot_id": bot_id,
                "config": bot_instance.virtual_webcam.config.__dict__,
                "speakers": {
                    speaker_id: {
                        "speaker_name": info.speaker_name,
                        "color": info.color,
                        "last_active": info.last_active.isoformat()
                    }
                    for speaker_id, info in bot_instance.virtual_webcam.speakers.items()
                },
                "is_streaming": bot_instance.virtual_webcam.is_streaming
            }
        else:
            raise get_error_response(
                status.HTTP_404_NOT_FOUND,
                "Virtual webcam not enabled",
                {"bot_id": bot_id}
            )
            
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error getting virtual webcam config: {e}")
        raise get_error_response(
            status.HTTP_500_INTERNAL_SERVER_ERROR,
            f"Error getting virtual webcam config: {str(e)}",
            {"bot_id": bot_id}
        )


@router.post("/virtual-webcam/config/{bot_id}")
async def update_virtual_webcam_config(
    bot_id: str,
    config_update: Dict[str, Any],
    bot_manager=Depends(get_bot_manager)
) -> Dict[str, Any]:
    """Update virtual webcam configuration."""
    try:
        bot_instance = bot_manager.get_bot(bot_id)
        if not bot_instance:
            raise get_error_response(
                status.HTTP_404_NOT_FOUND,
                "Bot not found",
                {"bot_id": bot_id}
            )
            
        if hasattr(bot_instance, 'virtual_webcam') and bot_instance.virtual_webcam:
            # Import DisplayMode and Theme from virtual_webcam module
            import sys
            import os
            sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'bot'))
            from virtual_webcam import DisplayMode, Theme
            
            # Update configurable properties
            webcam = bot_instance.virtual_webcam
            config = webcam.config
            
            # Update allowed configuration properties
            if "display_mode" in config_update:
                try:
                    config.display_mode = DisplayMode(config_update["display_mode"])
                except ValueError:
                    raise get_error_response(
                        status.HTTP_400_BAD_REQUEST,
                        "Invalid display mode",
                        {"bot_id": bot_id, "invalid_value": config_update["display_mode"]}
                    )
            
            if "theme" in config_update:
                try:
                    config.theme = Theme(config_update["theme"])
                except ValueError:
                    raise get_error_response(
                        status.HTTP_400_BAD_REQUEST,
                        "Invalid theme",
                        {"bot_id": bot_id, "invalid_value": config_update["theme"]}
                    )
            
            if "max_translations_displayed" in config_update:
                config.max_translations_displayed = max(1, min(10, int(config_update["max_translations_displayed"])))
            
            if "translation_duration_seconds" in config_update:
                config.translation_duration_seconds = max(5.0, min(60.0, float(config_update["translation_duration_seconds"])))
            
            if "show_speaker_names" in config_update:
                config.show_speaker_names = bool(config_update["show_speaker_names"])
            
            if "show_confidence" in config_update:
                config.show_confidence = bool(config_update["show_confidence"])
            
            if "show_timestamps" in config_update:
                config.show_timestamps = bool(config_update["show_timestamps"])
            
            return {
                "bot_id": bot_id,
                "message": "Virtual webcam configuration updated",
                "config": config.__dict__
            }
        else:
            raise get_error_response(
                status.HTTP_404_NOT_FOUND,
                "Virtual webcam not enabled",
                {"bot_id": bot_id}
            )
            
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error updating virtual webcam config: {e}")
        raise get_error_response(
            status.HTTP_500_INTERNAL_SERVER_ERROR,
            f"Error updating virtual webcam config: {str(e)}",
            {"bot_id": bot_id}
        )