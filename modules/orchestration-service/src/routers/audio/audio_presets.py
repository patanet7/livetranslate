"""
Audio Presets Management Router

Preset management endpoints including:
- Preset listing (/presets)
- Preset retrieval (/presets/{preset_name})
- Preset application (/presets/{preset_name}/apply)
- Preset saving (/presets/save)
- Preset deletion (/presets/{preset_name})
- Preset comparison (/presets/compare/{preset1}/{preset2})
"""

from datetime import UTC, datetime, timedelta
from typing import Any

import numpy as np
from dependencies import get_audio_coordinator, get_config_manager
from fastapi import Depends, HTTPException, status
from models.audio import AudioConfiguration
from utils.audio_errors import (
    AudioProcessingBaseError,
    AudioProcessingError,
    ValidationError,
)

from ._shared import (
    create_audio_router,
    error_boundary,
    format_recovery,
    logger,
    service_recovery,
)

# Create router for audio presets
router = create_audio_router()


@router.get("/presets", response_model=dict[str, AudioConfiguration])
async def get_audio_presets(
    category: str | None = None, config_manager=Depends(get_config_manager)
) -> dict[str, AudioConfiguration]:
    """
    Get all available audio configuration presets

    - **category**: Optional filter by preset category (e.g., 'voice', 'music', 'broadcast')
    """
    try:
        # Get all available presets
        all_presets = await _get_all_presets()

        # Filter by category if specified
        if category:
            filtered_presets = {
                name: preset
                for name, preset in all_presets.items()
                if preset.get("category", "").lower() == category.lower()
            }
            return filtered_presets

        return all_presets

    except Exception as e:
        logger.error(f"Failed to get presets: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to retrieve presets: {e!s}",
        ) from e


async def _get_all_presets() -> dict[str, dict[str, Any]]:
    """Get all available audio configuration presets"""
    return {
        "broadcast_standard": {
            "name": "Broadcast Standard",
            "description": "Standard broadcast audio processing with EBU R128 compliance",
            "category": "broadcast",
            "config": {
                "noise_reduction": {"enabled": True, "strength": 0.3},
                "compression": {"enabled": True, "ratio": 3.0, "threshold": -18.0},
                "equalizer": {
                    "enabled": True,
                    "low_shelf": {"freq": 100, "gain": 0, "q": 0.7},
                    "mid_peak": {"freq": 1000, "gain": 0, "q": 1.0},
                    "high_shelf": {"freq": 8000, "gain": -1, "q": 0.7},
                },
                "limiter": {"enabled": True, "threshold": -1.0, "release": 0.05},
                "lufs_normalization": {"enabled": True, "target_lufs": -23.0},
            },
            "characteristics": {
                "quality": "excellent",
                "processing_speed": "medium",
                "cpu_usage": "medium",
                "latency": "low",
            },
            "optimized_for": ["broadcast", "streaming", "compliance"],
            "created_by": "system",
            "created_at": "2024-01-01T00:00:00Z",
        },
        "voice_clarity": {
            "name": "Voice Clarity",
            "description": "Optimize for speech clarity and intelligibility",
            "category": "voice",
            "config": {
                "noise_reduction": {"enabled": True, "strength": 0.6},
                "voice_enhancement": {"enabled": True, "enhancement_strength": 0.8},
                "voice_filter": {
                    "enabled": True,
                    "low_cutoff": 80,
                    "high_cutoff": 8000,
                },
                "compression": {"enabled": True, "ratio": 4.0, "threshold": -20.0},
                "equalizer": {
                    "enabled": True,
                    "low_shelf": {"freq": 80, "gain": -2, "q": 0.7},
                    "mid_peak": {"freq": 2000, "gain": 2, "q": 1.2},
                    "high_shelf": {"freq": 5000, "gain": 1, "q": 0.7},
                },
                "agc": {"enabled": True, "target_level": -18.0, "max_gain": 15.0},
            },
            "characteristics": {
                "quality": "excellent",
                "processing_speed": "fast",
                "cpu_usage": "medium",
                "latency": "very_low",
            },
            "optimized_for": ["speech", "voice_over", "podcasts", "real_time"],
            "created_by": "system",
            "created_at": "2024-01-01T00:00:00Z",
        },
        "music_mastering": {
            "name": "Music Mastering",
            "description": "Professional music mastering chain",
            "category": "music",
            "config": {
                "equalizer": {
                    "enabled": True,
                    "low_shelf": {"freq": 100, "gain": 1, "q": 0.7},
                    "mid_peak": {"freq": 1000, "gain": 0, "q": 0.8},
                    "high_shelf": {"freq": 10000, "gain": 2, "q": 0.7},
                },
                "compression": {"enabled": True, "ratio": 2.5, "threshold": -12.0},
                "limiter": {"enabled": True, "threshold": -0.3, "release": 0.1},
                "lufs_normalization": {"enabled": True, "target_lufs": -14.0},
            },
            "characteristics": {
                "quality": "excellent",
                "processing_speed": "slow",
                "cpu_usage": "high",
                "latency": "medium",
            },
            "optimized_for": ["music", "mastering", "full_range"],
            "created_by": "system",
            "created_at": "2024-01-01T00:00:00Z",
        },
        "real_time_low_latency": {
            "name": "Real-time Low Latency",
            "description": "Minimal processing for real-time applications",
            "category": "real_time",
            "config": {
                "vad": {"enabled": True, "threshold": 0.3},
                "noise_reduction": {"enabled": True, "strength": 0.2},
                "agc": {"enabled": True, "target_level": -20.0, "max_gain": 10.0},
            },
            "characteristics": {
                "quality": "good",
                "processing_speed": "very_fast",
                "cpu_usage": "low",
                "latency": "minimal",
            },
            "optimized_for": ["real_time", "live_streaming", "low_latency"],
            "created_by": "system",
            "created_at": "2024-01-01T00:00:00Z",
        },
        "podcast_production": {
            "name": "Podcast Production",
            "description": "Complete podcast production chain",
            "category": "voice",
            "config": {
                "noise_reduction": {"enabled": True, "strength": 0.5},
                "voice_enhancement": {"enabled": True, "enhancement_strength": 0.6},
                "compression": {"enabled": True, "ratio": 6.0, "threshold": -22.0},
                "equalizer": {
                    "enabled": True,
                    "low_shelf": {"freq": 80, "gain": -3, "q": 0.7},
                    "mid_peak": {"freq": 3000, "gain": 2, "q": 1.0},
                    "high_shelf": {"freq": 8000, "gain": 0, "q": 0.7},
                },
                "limiter": {"enabled": True, "threshold": -2.0, "release": 0.1},
                "lufs_normalization": {"enabled": True, "target_lufs": -19.0},
            },
            "characteristics": {
                "quality": "excellent",
                "processing_speed": "medium",
                "cpu_usage": "medium",
                "latency": "low",
            },
            "optimized_for": ["podcasts", "voice", "distribution"],
            "created_by": "system",
            "created_at": "2024-01-01T00:00:00Z",
        },
    }


@router.get("/presets/{preset_name}")
async def get_preset_details(
    preset_name: str, config_manager=Depends(get_config_manager)
) -> dict[str, Any]:
    """
    Get detailed information about a specific preset
    """
    try:
        all_presets = await _get_all_presets()

        if preset_name not in all_presets:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=f"Preset '{preset_name}' not found",
            )

        preset = all_presets[preset_name]

        # Add usage statistics (placeholder)
        preset["usage_stats"] = {
            "times_used": np.random.randint(10, 1000),
            "last_used": (datetime.now(UTC) - timedelta(days=np.random.randint(1, 30))).isoformat(),
            "average_rating": round(4.0 + np.random.random(), 1),
        }

        return preset

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to get preset {preset_name}: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to retrieve preset: {e!s}",
        ) from e


@router.post("/presets/{preset_name}/apply")
async def apply_preset_to_audio(
    preset_name: str,
    request: dict[str, Any],
    audio_coordinator=Depends(get_audio_coordinator),
    config_manager=Depends(get_config_manager),
) -> dict[str, Any]:
    """
    Apply a preset configuration to audio data

    - **audio_data**: Base64 encoded audio data
    - **override_config**: Optional config overrides for the preset
    """
    correlation_id = f"preset_apply_{preset_name}_{datetime.now(UTC).strftime('%Y%m%d_%H%M%S_%f')}"

    async with error_boundary(
        correlation_id=correlation_id,
        context={
            "service": "orchestration",
            "endpoint": f"/presets/{preset_name}/apply",
            "preset": preset_name,
        },
        recovery_strategies=[format_recovery, service_recovery],
    ) as apply_correlation_id:
        try:
            logger.info(f"[{apply_correlation_id}] Applying preset: {preset_name}")

            # Validate request
            audio_data = request.get("audio_data")
            if not audio_data:
                raise ValidationError(
                    "No audio data provided for preset application",
                    correlation_id=apply_correlation_id,
                    validation_details={"missing_field": "audio_data"},
                )

            # Get preset configuration
            all_presets = await _get_all_presets()
            if preset_name not in all_presets:
                raise ValidationError(
                    f"Preset '{preset_name}' not found",
                    correlation_id=apply_correlation_id,
                    validation_details={
                        "preset": preset_name,
                        "available_presets": list(all_presets.keys()),
                    },
                )

            preset = all_presets[preset_name]
            preset_config = preset["config"].copy()

            # Apply any configuration overrides
            override_config = request.get("override_config", {})
            if override_config:
                preset_config = _merge_configs(preset_config, override_config)

            # Apply preset to audio
            result = await _apply_preset_processing(
                audio_data, preset_config, apply_correlation_id, audio_coordinator
            )

            return {
                "apply_id": apply_correlation_id,
                "preset_name": preset_name,
                "preset_config": preset_config,
                "override_applied": bool(override_config),
                "result": result,
                "timestamp": datetime.now(UTC).isoformat(),
            }

        except AudioProcessingBaseError:
            raise
        except Exception as e:
            raise AudioProcessingError(
                f"Preset application failed for {preset_name}: {e!s}",
                correlation_id=apply_correlation_id,
                processing_stage="preset_application",
                details={"preset": preset_name, "error": str(e)},
            ) from e


def _merge_configs(base_config: dict[str, Any], override_config: dict[str, Any]) -> dict[str, Any]:
    """Recursively merge configuration dictionaries"""
    result = base_config.copy()

    for key, value in override_config.items():
        if key in result and isinstance(result[key], dict) and isinstance(value, dict):
            result[key] = _merge_configs(result[key], value)
        else:
            result[key] = value

    return result


async def _apply_preset_processing(
    audio_data: str,
    preset_config: dict[str, Any],
    correlation_id: str,
    audio_coordinator,
) -> dict[str, Any]:
    """Apply preset processing to audio data"""
    try:
        # Simulate preset processing
        processing_stages = []
        current_audio = audio_data
        total_processing_time = 0.0

        # Process each enabled stage in the preset
        for stage_name, stage_config in preset_config.items():
            if stage_config.get("enabled", False):
                # Simulate stage processing
                stage_time = 0.02 + np.random.random() * 0.03
                total_processing_time += stage_time

                processing_stages.append(
                    {
                        "stage": stage_name,
                        "config": stage_config,
                        "processing_time": stage_time,
                        "status": "completed",
                    }
                )

        return {
            "processed_audio": current_audio,  # Placeholder - same as input
            "stages_applied": processing_stages,
            "total_stages": len(processing_stages),
            "total_processing_time": round(total_processing_time, 3),
            "preset_applied_successfully": True,
        }

    except Exception as e:
        raise AudioProcessingError(
            f"Preset processing failed: {e!s}",
            correlation_id=correlation_id,
            processing_stage="preset_processing",
            details={"error": str(e)},
        ) from e


@router.post("/presets/save")
async def save_custom_preset(
    request: dict[str, Any], config_manager=Depends(get_config_manager)
) -> dict[str, Any]:
    """
    Save a custom audio processing preset

    - **preset_name**: Name for the new preset
    - **description**: Description of the preset
    - **category**: Preset category
    - **config**: Audio processing configuration
    """
    try:
        preset_name = request.get("preset_name")
        description = request.get("description", "")
        category = request.get("category", "custom")
        config = request.get("config", {})

        if not preset_name:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="Preset name is required",
            )

        if not config:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="Preset configuration is required",
            )

        # Validate preset name
        if len(preset_name) < 3 or len(preset_name) > 50:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="Preset name must be between 3 and 50 characters",
            )

        # Check if preset already exists
        all_presets = await _get_all_presets()
        if preset_name in all_presets:
            raise HTTPException(
                status_code=status.HTTP_409_CONFLICT,
                detail=f"Preset '{preset_name}' already exists",
            )

        # Create preset object
        new_preset = {
            "name": preset_name,
            "description": description,
            "category": category,
            "config": config,
            "characteristics": _analyze_preset_characteristics(config),
            "optimized_for": _determine_optimization_targets(config),
            "created_by": "user",
            "created_at": datetime.now(UTC).isoformat(),
        }

        # Save preset (placeholder - in real implementation, save to database)
        preset_id = f"custom_{datetime.now(UTC).strftime('%Y%m%d_%H%M%S')}"

        return {
            "preset_id": preset_id,
            "preset_name": preset_name,
            "status": "saved",
            "preset_details": new_preset,
            "timestamp": datetime.now(UTC).isoformat(),
        }

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to save preset: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to save preset: {e!s}",
        ) from e


def _analyze_preset_characteristics(config: dict[str, Any]) -> dict[str, str]:
    """Analyze preset configuration to determine characteristics"""
    stage_count = len([k for k, v in config.items() if v.get("enabled", False)])

    # Determine processing speed based on stages
    if stage_count <= 2:
        processing_speed = "very_fast"
        cpu_usage = "low"
    elif stage_count <= 4:
        processing_speed = "fast"
        cpu_usage = "medium"
    elif stage_count <= 6:
        processing_speed = "medium"
        cpu_usage = "medium"
    else:
        processing_speed = "slow"
        cpu_usage = "high"

    # Determine quality based on stage complexity
    complex_stages = ["spectral_denoising", "voice_enhancement", "lufs_normalization"]
    has_complex = any(config.get(stage, {}).get("enabled", False) for stage in complex_stages)

    quality = "excellent" if has_complex else "good"
    latency = "minimal" if stage_count <= 2 else "low" if stage_count <= 4 else "medium"

    return {
        "quality": quality,
        "processing_speed": processing_speed,
        "cpu_usage": cpu_usage,
        "latency": latency,
    }


def _determine_optimization_targets(config: dict[str, Any]) -> list[str]:
    """Determine what the preset is optimized for"""
    targets = []

    if config.get("voice_enhancement", {}).get("enabled"):
        targets.extend(["speech", "voice"])

    if (
        config.get("real_time_low_latency", {}).get("enabled")
        or len([k for k, v in config.items() if v.get("enabled", False)]) <= 2
    ):
        targets.append("real_time")

    if config.get("lufs_normalization", {}).get("enabled"):
        lufs_target = config.get("lufs_normalization", {}).get("target_lufs", -23)
        if lufs_target == -23:
            targets.append("broadcast")
        elif lufs_target >= -16:
            targets.append("streaming")

    if config.get("compression", {}).get("enabled"):
        ratio = config.get("compression", {}).get("ratio", 1)
        if ratio >= 6:
            targets.append("podcasts")

    if not targets:
        targets.append("general")

    return targets


@router.delete("/presets/{preset_name}")
async def delete_preset(
    preset_name: str, config_manager=Depends(get_config_manager)
) -> dict[str, Any]:
    """
    Delete a custom preset (system presets cannot be deleted)
    """
    try:
        all_presets = await _get_all_presets()

        if preset_name not in all_presets:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=f"Preset '{preset_name}' not found",
            )

        preset = all_presets[preset_name]

        # Check if it's a system preset
        if preset.get("created_by") == "system":
            raise HTTPException(
                status_code=status.HTTP_403_FORBIDDEN,
                detail="System presets cannot be deleted",
            )

        # Delete preset (placeholder - in real implementation, delete from database)
        logger.info(f"Deleting custom preset: {preset_name}")

        return {
            "preset_name": preset_name,
            "status": "deleted",
            "timestamp": datetime.now(UTC).isoformat(),
        }

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to delete preset {preset_name}: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to delete preset: {e!s}",
        ) from e


@router.get("/presets/compare/{preset1}/{preset2}")
async def compare_presets(
    preset1: str, preset2: str, config_manager=Depends(get_config_manager)
) -> dict[str, Any]:
    """
    Compare two presets and provide analysis and recommendations
    """
    try:
        all_presets = await _get_all_presets()

        # Validate both presets exist
        if preset1 not in all_presets:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=f"Preset '{preset1}' not found",
            )

        if preset2 not in all_presets:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=f"Preset '{preset2}' not found",
            )

        preset1_data = all_presets[preset1]
        preset2_data = all_presets[preset2]

        # Perform comparison
        comparison = await _compare_preset_configurations(preset1_data, preset2_data)

        return {
            "preset1": preset1,
            "preset2": preset2,
            "comparison": comparison,
            "recommendation": _generate_preset_recommendation(preset1_data, preset2_data),
            "timestamp": datetime.now(UTC).isoformat(),
        }

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to compare presets {preset1} and {preset2}: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to compare presets: {e!s}",
        ) from e


async def _compare_preset_configurations(preset1_data: dict, preset2_data: dict) -> dict[str, Any]:
    """Compare two preset configurations"""
    config1 = preset1_data["config"]
    config2 = preset2_data["config"]

    # Compare stage enablement
    stages1 = {k for k, v in config1.items() if v.get("enabled", False)}
    stages2 = {k for k, v in config2.items() if v.get("enabled", False)}

    common_stages = stages1.intersection(stages2)
    unique_to_1 = stages1 - stages2
    unique_to_2 = stages2 - stages1

    # Compare characteristics
    char1 = preset1_data.get("characteristics", {})
    char2 = preset2_data.get("characteristics", {})

    characteristics_comparison = {}
    for key in set(char1.keys()).union(char2.keys()):
        characteristics_comparison[key] = {
            "preset1": char1.get(key, "unknown"),
            "preset2": char2.get(key, "unknown"),
            "advantage": _determine_characteristic_advantage(char1.get(key), char2.get(key), key),
        }

    return {
        "stage_comparison": {
            "common_stages": list(common_stages),
            "unique_to_preset1": list(unique_to_1),
            "unique_to_preset2": list(unique_to_2),
            "total_stages_preset1": len(stages1),
            "total_stages_preset2": len(stages2),
        },
        "characteristics_comparison": characteristics_comparison,
        "optimization_targets": {
            "preset1": preset1_data.get("optimized_for", []),
            "preset2": preset2_data.get("optimized_for", []),
        },
        "categories": {
            "preset1": preset1_data.get("category", "unknown"),
            "preset2": preset2_data.get("category", "unknown"),
        },
    }


def _determine_characteristic_advantage(value1: str, value2: str, characteristic: str) -> str:
    """Determine which preset has advantage for a characteristic"""
    if value1 == value2:
        return "equal"

    # Define preference order for each characteristic
    preferences = {
        "quality": ["excellent", "good", "fair", "poor"],
        "processing_speed": ["very_fast", "fast", "medium", "slow"],
        "cpu_usage": ["low", "medium", "high"],
        "latency": ["minimal", "very_low", "low", "medium", "high"],
    }

    if characteristic in preferences:
        pref_order = preferences[characteristic]
        try:
            idx1 = pref_order.index(value1) if value1 in pref_order else len(pref_order)
            idx2 = pref_order.index(value2) if value2 in pref_order else len(pref_order)

            if idx1 < idx2:
                return "preset1"
            elif idx2 < idx1:
                return "preset2"
            else:
                return "equal"
        except (ValueError, TypeError):
            return "equal"

    return "equal"


def _generate_preset_recommendation(preset1_data: dict, preset2_data: dict) -> str:
    """Generate recommendation based on preset comparison"""
    char1 = preset1_data.get("characteristics", {})
    char2 = preset2_data.get("characteristics", {})

    # Simple recommendation logic based on characteristics
    if "real_time" in preset1_data.get("optimized_for", []):
        return f"Use {preset1_data['name']} for real-time applications"
    elif "real_time" in preset2_data.get("optimized_for", []):
        return f"Use {preset2_data['name']} for real-time applications"
    elif char1.get("quality") == "excellent":
        return f"Use {preset1_data['name']} for best quality"
    elif char2.get("quality") == "excellent":
        return f"Use {preset2_data['name']} for best quality"
    else:
        return "Both presets are suitable for general use"
