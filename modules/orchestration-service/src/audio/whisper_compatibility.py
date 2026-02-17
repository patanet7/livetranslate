#!/usr/bin/env python3
"""
Whisper Service Compatibility Layer - Audio Chunking Migration

Ensures the orchestration service chunking configuration matches and is compatible
with what we're replacing in the whisper service's internal chunking system.

This module provides:
- Configuration migration from whisper service settings
- Compatibility validation
- Easy configuration mapping
- Seamless transition support
"""

import os
from dataclasses import asdict, dataclass
from typing import Any

from livetranslate_common.logging import get_logger

from .models import AudioChunkingConfig

logger = get_logger()


@dataclass
class WhisperServiceConfig:
    """Configuration structure matching whisper service defaults."""

    # Audio settings from whisper service (whisper_service.py:838-853)
    sample_rate: int = 16000
    buffer_duration: float = 4.0  # Reduced from 6.0 in whisper service
    inference_interval: float = 3.0
    overlap_duration: float = 0.2  # Minimal overlap in whisper service
    enable_vad: bool = True

    # Device settings
    device: str | None = None

    # Performance settings
    min_inference_interval: float = 0.2
    max_concurrent_requests: int = 10

    # Model settings
    models_dir: str = ""
    default_model: str = "whisper-base.en"

    # Session settings
    session_dir: str | None = None


class WhisperCompatibilityManager:
    """
    Manages compatibility between orchestration service chunking and whisper service.
    Ensures seamless migration with preserved functionality.
    """

    def __init__(self):
        self.whisper_config = self._load_whisper_config()
        self.orchestration_config = self._create_compatible_orchestration_config()

    def _load_whisper_config(self) -> WhisperServiceConfig:
        """Load current whisper service configuration from environment and defaults."""
        try:
            config = WhisperServiceConfig(
                # Load from environment variables (matching whisper_service.py)
                sample_rate=int(os.getenv("SAMPLE_RATE", "16000")),
                buffer_duration=float(os.getenv("BUFFER_DURATION", "4.0")),
                inference_interval=float(os.getenv("INFERENCE_INTERVAL", "3.0")),
                overlap_duration=float(os.getenv("OVERLAP_DURATION", "0.2")),
                enable_vad=os.getenv("ENABLE_VAD", "true").lower() == "true",
                device=os.getenv("OPENVINO_DEVICE"),
                min_inference_interval=float(os.getenv("MIN_INFERENCE_INTERVAL", "0.2")),
                max_concurrent_requests=int(os.getenv("MAX_CONCURRENT_REQUESTS", "10")),
                models_dir=os.getenv("WHISPER_MODELS_DIR", ""),
                default_model=os.getenv("WHISPER_DEFAULT_MODEL", "whisper-base.en"),
                session_dir=os.getenv("SESSION_DIR"),
            )

            logger.info(f"Loaded whisper service configuration: {asdict(config)}")
            return config

        except Exception as e:
            logger.warning(f"Failed to load whisper config from environment: {e}")
            logger.info("Using whisper service defaults")
            return WhisperServiceConfig()

    def _create_compatible_orchestration_config(self) -> AudioChunkingConfig:
        """Create orchestration config that's compatible with whisper service settings."""

        # Map whisper service settings to orchestration service
        orchestration_config = AudioChunkingConfig(
            # Core chunking parameters - match whisper service
            chunk_duration=self.whisper_config.inference_interval,  # 3.0s default
            overlap_duration=self.whisper_config.overlap_duration,  # 0.2s default
            processing_interval=self.whisper_config.inference_interval * 0.8,  # Slightly faster
            buffer_duration=max(5.0, self.whisper_config.buffer_duration),  # Ensure >= 5.0s minimum
            # Quality thresholds - optimized for whisper compatibility
            min_quality_threshold=0.3,
            silence_threshold=0.01,  # Match whisper's silence detection
            noise_threshold=0.02,
            # Speaker correlation - enhanced beyond whisper service
            speaker_correlation_enabled=True,
            correlation_confidence_threshold=0.7,
            correlation_temporal_window=self.whisper_config.inference_interval,
            # Database integration - new in orchestration
            store_audio_files=True,
            store_transcripts=True,
            store_translations=True,
            store_correlations=True,
            track_chunk_lineage=True,
            # Performance settings - match whisper service
            max_concurrent_chunks=self.whisper_config.max_concurrent_requests,
            chunk_processing_timeout=30.0,
            database_batch_size=100,
            # File storage settings
            audio_storage_path=self.whisper_config.session_dir or "/data/audio",
            file_compression_enabled=True,
            cleanup_old_files=True,
            file_retention_days=30,
        )

        logger.info(f"Created compatible orchestration config: {orchestration_config.dict()}")
        return orchestration_config

    def validate_compatibility(self) -> dict[str, Any]:
        """Validate that orchestration config is compatible with whisper service."""

        validation_result = {
            "compatible": True,
            "issues": [],
            "warnings": [],
            "migration_notes": [],
        }

        # Check timing compatibility
        if self.orchestration_config.chunk_duration != self.whisper_config.inference_interval:
            validation_result["warnings"].append(
                f"Chunk duration ({self.orchestration_config.chunk_duration}s) differs from "
                f"whisper inference interval ({self.whisper_config.inference_interval}s)"
            )

        if self.orchestration_config.overlap_duration != self.whisper_config.overlap_duration:
            validation_result["warnings"].append(
                f"Overlap duration ({self.orchestration_config.overlap_duration}s) differs from "
                f"whisper overlap ({self.whisper_config.overlap_duration}s)"
            )

        # Check buffer compatibility
        if self.orchestration_config.buffer_duration < self.whisper_config.buffer_duration:
            validation_result["issues"].append(
                f"Orchestration buffer ({self.orchestration_config.buffer_duration}s) is smaller than "
                f"whisper buffer ({self.whisper_config.buffer_duration}s)"
            )
            validation_result["compatible"] = False

        # Check performance compatibility
        if (
            self.orchestration_config.max_concurrent_chunks
            < self.whisper_config.max_concurrent_requests
        ):
            validation_result["warnings"].append(
                f"Max concurrent chunks ({self.orchestration_config.max_concurrent_chunks}) is less than "
                f"whisper max requests ({self.whisper_config.max_concurrent_requests})"
            )

        # Migration notes
        validation_result["migration_notes"].extend(
            [
                "Orchestration service adds database persistence for all audio chunks",
                "Speaker correlation is now centralized and enhanced",
                "Chunk lineage tracking provides better debugging capabilities",
                "File storage includes compression and retention management",
                "Processing intervals can be tuned independently of chunk duration",
            ]
        )

        logger.info(f"Compatibility validation result: {validation_result}")
        return validation_result

    def get_migration_config(self) -> dict[str, Any]:
        """Get configuration for smooth migration from whisper service."""

        return {
            "whisper_service": {
                "endpoint": "http://localhost:5001/api/process-chunk",
                "compatibility_mode": True,
                "preserve_timing": True,
                "chunk_metadata_support": True,
            },
            "orchestration_service": {
                "chunking_config": self.orchestration_config.dict(),
                "enable_database_storage": True,
                "enable_speaker_correlation": True,
                "enable_chunk_lineage": True,
            },
            "migration_settings": {
                "gradual_transition": True,
                "validate_chunk_consistency": True,
                "preserve_session_continuity": True,
                "enable_comparison_mode": False,  # Set to True for A/B testing
            },
        }

    def create_chunk_metadata_for_whisper(
        self,
        chunk_id: str,
        session_id: str,
        sequence_number: int,
        start_time: float,
        end_time: float,
        audio_data: bytes,
    ) -> dict[str, Any]:
        """
        Create chunk metadata compatible with whisper service expectations.
        This ensures whisper service receives all necessary context.
        """

        duration = end_time - start_time

        return {
            "chunk_id": chunk_id,
            "session_id": session_id,
            "sequence_number": sequence_number,
            "start_time": start_time,
            "end_time": end_time,
            "duration": duration,
            "overlap_start": max(0, start_time - self.orchestration_config.overlap_duration),
            "overlap_end": min(
                end_time + self.orchestration_config.overlap_duration,
                start_time + self.orchestration_config.buffer_duration,
            ),
            "sample_rate": self.whisper_config.sample_rate,
            "enable_vad": False,  # VAD already applied by orchestration
            "enable_enhancement": False,  # Enhancement already applied
            "timestamp_mode": "word",
            "processing_context": {
                "source": "orchestration_service",
                "pipeline_version": "1.0",
                "chunking_algorithm": "centralized_overlap_coordination",
                "quality_threshold": self.orchestration_config.min_quality_threshold,
            },
        }

    def get_whisper_service_request(
        self,
        chunk_id: str,
        session_id: str,
        audio_data: bytes,
        chunk_metadata: dict[str, Any],
        model_name: str | None = None,
    ) -> dict[str, Any]:
        """
        Create a complete request for the whisper service that matches the new API.
        This ensures seamless integration with the updated whisper service.
        """

        import base64

        return {
            "chunk_id": chunk_id,
            "session_id": session_id,
            "audio_data": base64.b64encode(audio_data).decode("utf-8"),
            "chunk_metadata": chunk_metadata,
            "model_name": model_name or self.whisper_config.default_model,
        }

    def validate_whisper_response(self, response: dict[str, Any]) -> dict[str, Any]:
        """
        Validate and normalize whisper service response for orchestration service.
        Ensures consistent response format regardless of whisper service version.
        """

        validation_result = {"valid": True, "normalized_response": {}, "issues": []}

        try:
            # Required fields validation
            required_fields = ["chunk_id", "session_id", "status"]
            for field in required_fields:
                if field not in response:
                    validation_result["issues"].append(f"Missing required field: {field}")
                    validation_result["valid"] = False

            # Normalize response structure
            if validation_result["valid"]:
                validation_result["normalized_response"] = {
                    "chunk_id": response["chunk_id"],
                    "session_id": response["session_id"],
                    "status": response["status"],
                    "transcription": response.get("transcription", {}),
                    "processing_info": response.get("processing_info", {}),
                    "chunk_sequence": response.get("chunk_sequence", 0),
                    "chunk_timing": response.get("chunk_timing", {}),
                    "error": response.get("error"),
                    "error_type": response.get("error_type"),
                }

                # Ensure transcription has required fields
                transcription = validation_result["normalized_response"]["transcription"]
                if transcription:
                    transcription.setdefault("text", "")
                    transcription.setdefault("language", "unknown")
                    transcription.setdefault("confidence_score", 0.0)
                    transcription.setdefault("segments", [])

        except Exception as e:
            validation_result["valid"] = False
            validation_result["issues"].append(f"Response validation error: {e!s}")

        return validation_result


def create_compatibility_manager() -> WhisperCompatibilityManager:
    """Factory function to create a configured compatibility manager."""
    return WhisperCompatibilityManager()


def get_compatible_chunking_config() -> AudioChunkingConfig:
    """Get chunking configuration compatible with whisper service."""
    manager = create_compatibility_manager()
    return manager.orchestration_config


def validate_migration_readiness() -> dict[str, Any]:
    """Validate that the system is ready for migration from whisper to orchestration chunking."""
    manager = create_compatibility_manager()

    compatibility_check = manager.validate_compatibility()
    migration_config = manager.get_migration_config()

    return {
        "ready_for_migration": compatibility_check["compatible"],
        "compatibility_issues": compatibility_check["issues"],
        "compatibility_warnings": compatibility_check["warnings"],
        "migration_config": migration_config,
        "recommended_actions": [
            "Test orchestration chunking with existing audio samples",
            "Validate transcription quality matches whisper service output",
            "Ensure database storage is properly configured",
            "Test speaker correlation functionality",
            "Verify chunk lineage tracking",
            "Run performance comparison tests",
        ],
    }


# Easy configuration access for frontend settings
def get_frontend_compatible_config() -> dict[str, Any]:
    """Get configuration in format suitable for frontend settings pages."""
    manager = create_compatibility_manager()

    return {
        "whisper_service_settings": {
            "sample_rate": manager.whisper_config.sample_rate,
            "buffer_duration": manager.whisper_config.buffer_duration,
            "inference_interval": manager.whisper_config.inference_interval,
            "overlap_duration": manager.whisper_config.overlap_duration,
            "enable_vad": manager.whisper_config.enable_vad,
            "max_concurrent_requests": manager.whisper_config.max_concurrent_requests,
        },
        "orchestration_chunking_settings": {
            "chunk_duration": manager.orchestration_config.chunk_duration,
            "overlap_duration": manager.orchestration_config.overlap_duration,
            "processing_interval": manager.orchestration_config.processing_interval,
            "buffer_duration": manager.orchestration_config.buffer_duration,
            "max_concurrent_chunks": manager.orchestration_config.max_concurrent_chunks,
            "speaker_correlation_enabled": manager.orchestration_config.speaker_correlation_enabled,
        },
        "migration_status": {
            "compatibility_validated": True,
            "migration_ready": True,
            "database_integration": "enabled",
            "chunk_lineage_tracking": "enabled",
            "speaker_correlation": "enhanced",
        },
    }


# Configuration templates for different scenarios
CONFIGURATION_PRESETS = {
    "exact_whisper_match": {
        "description": "Exactly match current whisper service settings",
        "chunk_duration": 3.0,
        "overlap_duration": 0.2,
        "processing_interval": 3.0,
        "buffer_duration": 4.0,
    },
    "optimized_performance": {
        "description": "Optimized for better performance with minimal overlap",
        "chunk_duration": 2.5,
        "overlap_duration": 0.3,
        "processing_interval": 2.2,
        "buffer_duration": 6.0,
    },
    "high_accuracy": {
        "description": "Higher accuracy with more overlap and longer chunks",
        "chunk_duration": 4.0,
        "overlap_duration": 0.8,
        "processing_interval": 3.2,
        "buffer_duration": 8.0,
    },
    "real_time_optimized": {
        "description": "Optimized for real-time processing with minimal latency",
        "chunk_duration": 2.0,
        "overlap_duration": 0.2,
        "processing_interval": 1.8,
        "buffer_duration": 4.0,
    },
}


def apply_configuration_preset(preset_name: str) -> AudioChunkingConfig:
    """Apply a configuration preset for specific use cases."""
    if preset_name not in CONFIGURATION_PRESETS:
        raise ValueError(
            f"Unknown preset: {preset_name}. Available: {list(CONFIGURATION_PRESETS.keys())}"
        )

    preset = CONFIGURATION_PRESETS[preset_name]
    manager = create_compatibility_manager()

    # Update orchestration config with preset values
    config_dict = manager.orchestration_config.dict()
    config_dict.update(
        {
            "chunk_duration": preset["chunk_duration"],
            "overlap_duration": preset["overlap_duration"],
            "processing_interval": preset["processing_interval"],
            "buffer_duration": preset["buffer_duration"],
        }
    )

    return AudioChunkingConfig(**config_dict)
