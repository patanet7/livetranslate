"""
Embedded audio/Whisper service facade.

This module mirrors the functionality of the standalone whisper-service so the
orchestration layer can run transcription work in-process when desired. The
facade is intentionally defensive: production deployments with the dedicated
microservice still work, and local environments without heavy dependencies
degrade gracefully instead of crashing on import errors.
"""

from __future__ import annotations

import asyncio
import logging
import os
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional

# Add shared module to path for model registry (append to avoid conflicts)
_SHARED_PATH = Path(__file__).parent.parent.parent.parent.parent / "shared" / "src"
if str(_SHARED_PATH) not in sys.path:
    sys.path.append(str(_SHARED_PATH))

try:
    from model_registry import ModelRegistry
    _MODEL_REGISTRY_AVAILABLE = True
except ImportError:
    _MODEL_REGISTRY_AVAILABLE = False

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Attempt to import whisper-service components. We keep this optional so the
# orchestration service can boot even when the audio stack is unavailable.
# ---------------------------------------------------------------------------

# EMBEDDED MODE DISABLED - Using remote whisper-service only
AUDIO_MODULE_AVAILABLE = False
AUDIO_IMPORT_ERROR: Optional[BaseException] = None
_AUDIO_SOURCE_PATH: Optional[Path] = None

# Embedded mode disabled to avoid import warnings
# The orchestration service will connect to whisper-service via Socket.IO on port 5001
logger.info("Embedded audio mode disabled - using remote whisper-service")


class UnifiedAudioError(RuntimeError):
    """Raised when the embedded audio service is unavailable or fails."""


class UnifiedAudioService:
    """Facade around whisper-service that exposes a thin async API."""

    DEFAULT_SAMPLE_RATE = 16000
    # Use model registry if available, fallback to hardcoded default
    DEFAULT_MODEL = (
        ModelRegistry.DEFAULT_WHISPER_MODEL
        if _MODEL_REGISTRY_AVAILABLE
        else "whisper-base"
    )

    def __init__(self) -> None:
        self._service: Optional[_WhisperService] = None
        self._init_lock = asyncio.Lock()
        self._last_error: Optional[str] = None
        self._metrics: Dict[str, Any] = {
            "total_requests": 0,
            "successful_requests": 0,
            "failed_requests": 0,
            "total_processing_time": 0.0,
        }

    def is_available(self) -> bool:
        return AUDIO_MODULE_AVAILABLE

    @property
    def last_error(self) -> Optional[str]:
        return self._last_error

    async def _ensure_service(self) -> Optional[_WhisperService]:
        if not AUDIO_MODULE_AVAILABLE:
            return None

        if self._service is not None:
            return self._service

        async with self._init_lock:
            if self._service is not None:
                return self._service

            if os.getenv("DISABLE_EMBEDDED_AUDIO", "").lower() in {"1", "true", "yes"}:
                self._last_error = "Embedded audio disabled via environment"
                return None

            try:
                logger.info("Initializing embedded Whisper service ...")
                self._service = _WhisperService()
                self._last_error = None
                logger.info("Embedded Whisper service ready")
            except Exception as exc:  # pragma: no cover - depends on environment
                self._last_error = str(exc)
                logger.warning(
                    "Embedded Whisper service initialization failed: %s",
                    exc,
                    exc_info=True,
                )
                self._service = None

        return self._service

    async def transcribe_bytes(
        self,
        *,
        audio_bytes: bytes,
        language: Optional[str],
        model: Optional[str],
        enable_vad: bool,
        session_id: Optional[str] = None,
        sample_rate: Optional[int] = None,
    ) -> Dict[str, Any]:
        """Transcribe raw audio bytes using the embedded service."""
        service = await self._ensure_service()
        if service is None:
            raise UnifiedAudioError(
                self._last_error or "Embedded audio service unavailable"
            )

        request = _TranscriptionRequest(
            audio_data=audio_bytes,
            model_name=model or self.DEFAULT_MODEL,
            language=language,
            session_id=session_id,
            streaming=False,
            sample_rate=sample_rate or self.DEFAULT_SAMPLE_RATE,
            enable_vad=enable_vad,
        )

        try:
            result: _TranscriptionResult = await service.transcribe(request)
        except Exception as exc:
            self._metrics["total_requests"] += 1
            self._metrics["failed_requests"] += 1
            self._last_error = str(exc)
            raise UnifiedAudioError(str(exc)) from exc

        processing_time = getattr(result, "processing_time", 0.0)
        self._metrics["total_requests"] += 1
        self._metrics["successful_requests"] += 1
        self._metrics["total_processing_time"] += processing_time

        return {
            "text": getattr(result, "text", ""),
            "language": getattr(result, "language", language or "auto"),
            "segments": getattr(result, "segments", []),
            "speakers": getattr(result, "speakers", None),
            "confidence": getattr(result, "confidence_score", 0.0),
            "processing_time": processing_time,
            "session_id": getattr(result, "session_id", session_id),
            "timestamp": getattr(result, "timestamp", datetime.now(timezone.utc).isoformat()),
        }

    async def health(self) -> Dict[str, Any]:
        """Return health information for diagnostics."""
        available = self.is_available()
        service = await self._ensure_service() if available else None

        status = "healthy" if service else "degraded"
        return {
            "status": status,
            "embedded": True,
            "module_available": available,
            "source_path": _AUDIO_SOURCE_PATH.as_posix()
            if _AUDIO_SOURCE_PATH
            else None,
            "last_error": self._last_error,
        }

    async def get_models(self) -> List[str]:
        """Return list of available models (fallback to defaults)."""
        if self._service and hasattr(self._service, "model_manager"):
            manager = getattr(self._service, "model_manager")
            if hasattr(manager, "list_models"):
                try:
                    return list(manager.list_models())
                except Exception as exc:
                    logger.debug("Failed to list models from embedded service: %s", exc)
        return [self.DEFAULT_MODEL]

    async def get_device_info(self) -> Dict[str, Any]:
        """Return device information if available."""
        device = None
        if self._service and hasattr(self._service, "model_manager"):
            manager = getattr(self._service, "model_manager")
            device = getattr(manager, "device", None)

        return {
            "device": device or "cpu",
            "mode": "embedded",
            "module_available": self.is_available(),
            "last_error": self._last_error,
        }

    async def get_statistics(self) -> Dict[str, Any]:
        """Return aggregated metrics."""
        total = self._metrics.get("total_requests", 0)
        total_time = self._metrics.get("total_processing_time", 0.0)
        avg_time = total_time / total if total else 0.0

        return {
            **self._metrics,
            "average_processing_time": avg_time,
            "timestamp": datetime.now(timezone.utc).isoformat(),
        }


_AUDIO_SINGLETON: Optional[UnifiedAudioService] = None


def get_unified_audio_service() -> UnifiedAudioService:
    global _AUDIO_SINGLETON
    if _AUDIO_SINGLETON is None:
        _AUDIO_SINGLETON = UnifiedAudioService()
    return _AUDIO_SINGLETON


def reset_unified_audio_service() -> None:
    """Reset the singleton (for testing only)"""
    global _AUDIO_SINGLETON
    _AUDIO_SINGLETON = None


__all__ = [
    "UnifiedAudioService",
    "UnifiedAudioError",
    "get_unified_audio_service",
    "reset_unified_audio_service",
]
