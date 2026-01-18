"""
Embedded translation service facade.

This module loads the existing `translation-service` package directly inside
the orchestration process so we can run everything as a single service.
If the heavy translation stack (models, OpenVINO/vLLM, etc.) is unavailable,
the facade falls back to lightweight mock responses rather than crashing.
"""

from __future__ import annotations

import asyncio
import logging
import os
import sys
from datetime import UTC, datetime
from difflib import SequenceMatcher
from pathlib import Path
from typing import Any
from uuid import uuid4

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Locate translation-service source and attempt to import it. We keep this
# defensive so unified deployments without the optional dependencies can
# still boot and degrade gracefully.
# ---------------------------------------------------------------------------

TRANSLATION_MODULE_AVAILABLE = False
TRANSLATION_IMPORT_ERROR: BaseException | None = None
_TRANSLATION_SOURCE_PATH: Path | None = None

try:
    repo_root = Path(__file__).resolve()
    for _ in range(5):
        repo_root = repo_root.parent
    translation_src = (repo_root / "modules" / "translation-service" / "src").resolve()
    if translation_src.exists():
        if str(translation_src) not in sys.path:
            sys.path.insert(0, str(translation_src))
        _TRANSLATION_SOURCE_PATH = translation_src

        from translation_service import (  # type: ignore
            TranslationRequest as _TranslationRequest,
            TranslationResult as _TranslationResult,
            TranslationService as _TranslationService,
            create_translation_service as _create_translation_service,
        )

        TRANSLATION_MODULE_AVAILABLE = True
        logger.info("Embedded translation module available at %s", translation_src.as_posix())
    else:
        TRANSLATION_IMPORT_ERROR = FileNotFoundError(
            f"translation-service source directory missing at {translation_src}"
        )
        logger.warning("%s", TRANSLATION_IMPORT_ERROR)
except Exception as exc:  # pragma: no cover - import failures are environment dependent
    TRANSLATION_IMPORT_ERROR = exc
    TRANSLATION_MODULE_AVAILABLE = False
    logger.warning("Failed to import embedded translation-service module: %s", exc, exc_info=True)


class UnifiedTranslationError(RuntimeError):
    """Raised when the embedded translation service is unavailable or fails."""


class UnifiedTranslationService:
    """
    Thin facade around the original translation-service so orchestration can call it
    in-process. All public methods return plain dicts or primitives, leaving response
    modelling to the FastAPI layer.
    """

    def __init__(self) -> None:
        self._service: _TranslationService | None = None
        self._init_lock = asyncio.Lock()
        self._last_error: str | None = None
        self._default_languages: list[dict[str, str]] = [
            {"code": "en", "name": "English"},
            {"code": "es", "name": "Spanish"},
            {"code": "fr", "name": "French"},
            {"code": "de", "name": "German"},
            {"code": "it", "name": "Italian"},
            {"code": "pt", "name": "Portuguese"},
            {"code": "ru", "name": "Russian"},
            {"code": "ja", "name": "Japanese"},
            {"code": "ko", "name": "Korean"},
            {"code": "zh", "name": "Chinese"},
        ]
        self._metrics: dict[str, Any] = {
            "total_translations": 0,
            "successful_translations": 0,
            "failed_translations": 0,
            "total_processing_time": 0.0,
            "language_pairs_processed": {},
            "model_performance": {},
        }
        self._sessions: dict[str, dict[str, Any]] = {}

    # ------------------------------------------------------------------
    # Service bootstrapping helpers
    # ------------------------------------------------------------------

    def is_available(self) -> bool:
        """Return True when the translation module could be imported."""
        return TRANSLATION_MODULE_AVAILABLE

    @property
    def last_error(self) -> str | None:
        """Return the last initialization or runtime error message, if any."""
        return self._last_error

    async def _ensure_service(self) -> _TranslationService | None:
        """
        Lazily create the underlying translation service. This matches the async
        factory provided by the microservice so model loading happens once.
        """
        if not TRANSLATION_MODULE_AVAILABLE:
            return None

        if self._service is not None:
            return self._service

        async with self._init_lock:
            if self._service is not None:
                return self._service

            try:
                # Allow callers to skip heavy initialization via environment variable.
                if os.getenv("DISABLE_EMBEDDED_TRANSLATION", "").lower() in {
                    "1",
                    "true",
                    "yes",
                }:
                    self._last_error = "Embedded translation disabled via environment"
                    return None

                logger.info("Initializing embedded translation service ...")
                service = await _create_translation_service()
                self._service = service
                self._last_error = None
                logger.info(
                    "Embedded translation service ready (fallback=%s)",
                    getattr(service, "fallback_mode", False),
                )
            except Exception as exc:  # pragma: no cover - highly environment dependent
                self._last_error = str(exc)
                logger.warning("Embedded translation initialization failed: %s", exc, exc_info=True)
                self._service = None

        return self._service

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    async def translate(
        self,
        *,
        text: str,
        target_language: str,
        source_language: str | None = None,
        session_id: str | None = None,
        quality: str | None = None,
        model: str | None = None,
        confidence_threshold: float = 0.8,
    ) -> dict[str, Any]:
        """
        Perform a translation request using the embedded service.
        Returns a dict aligned with the orchestration translation response.
        """
        service = await self._ensure_service()
        if service is None:
            raise UnifiedTranslationError(
                self._last_error or "Embedded translation service unavailable"
            )

        try:
            request = _TranslationRequest(
                text=text,
                source_language=source_language or "auto",
                target_language=target_language or "en",
                session_id=session_id,
                streaming=False,
                confidence_threshold=confidence_threshold,
            )

            result: _TranslationResult = await service.translate(request)
            backend_used = getattr(result, "backend_used", None)

            # Update basic metrics for analytics-style queries
            self._metrics["total_translations"] = self._metrics.get("total_translations", 0) + 1
            self._metrics["successful_translations"] = (
                self._metrics.get("successful_translations", 0) + 1
            )
            self._metrics["total_processing_time"] = self._metrics.get(
                "total_processing_time", 0.0
            ) + getattr(result, "processing_time", 0.0)

            pair_key = f"{(result.source_language or source_language or 'auto').lower()}->{(result.target_language or target_language).lower()}"
            pair_stats = self._metrics.setdefault("language_pairs_processed", {})
            pair_stats[pair_key] = pair_stats.get(pair_key, 0) + 1

            model_stats = self._metrics.setdefault("model_performance", {})
            model_key = (backend_used or model or "embedded").lower()
            model_entry = model_stats.setdefault(
                model_key, {"count": 0, "total_processing_time": 0.0}
            )
            model_entry["count"] += 1
            model_entry["total_processing_time"] += getattr(result, "processing_time", 0.0)

            return {
                "translated_text": result.translated_text,
                "source_language": result.source_language or source_language or "auto",
                "target_language": result.target_language or target_language,
                "confidence": getattr(result, "confidence_score", 0.0),
                "processing_time": getattr(result, "processing_time", 0.0),
                "backend_used": backend_used or (model or "embedded"),
                "model_used": model or backend_used or "embedded",
                "session_id": getattr(result, "session_id", session_id),
                "timestamp": getattr(result, "timestamp", datetime.now(UTC).isoformat()),
            }
        except Exception as exc:
            self._last_error = str(exc)
            self._metrics["total_translations"] = self._metrics.get("total_translations", 0) + 1
            self._metrics["failed_translations"] = self._metrics.get("failed_translations", 0) + 1
            logger.warning("Embedded translation failed: %s", exc, exc_info=True)
            raise UnifiedTranslationError(str(exc)) from exc

    async def health(self) -> dict[str, Any]:
        """
        Return health diagnostics for the embedded service. The structure mirrors
        the HTTP health endpoint used by the original microservice.
        """
        available = self.is_available()
        service = await self._ensure_service() if available else None

        status = {
            "status": "healthy" if service else "degraded",
            "embedded": True,
            "module_available": available,
            "source_path": _TRANSLATION_SOURCE_PATH.as_posix()
            if _TRANSLATION_SOURCE_PATH
            else None,
            "fallback_mode": bool(getattr(service, "fallback_mode", False)) if service else True,
            "last_error": self._last_error,
        }
        return status

    async def get_supported_languages(self) -> list[dict[str, str]]:
        """
        Return supported languages. When the embedded service exposes a richer
        list we attempt to use it, otherwise a conservative default is returned.
        """
        service = await self._ensure_service()
        if service and hasattr(service, "config"):
            languages = getattr(service.config, "supported_languages", None)
            if isinstance(languages, list) and languages:
                return languages
        return list(self._default_languages)

    async def detect_language(self, text: str) -> dict[str, Any]:
        """
        Provide a lightweight language detection fallback. If the embedded service
        exposes custom detectors in the future this can be expanded.
        """
        if not text:
            raise UnifiedTranslationError("Text required for language detection")

        try:
            # Try optional langdetect if installed
            import langdetect  # type: ignore

            detected = langdetect.detect_langs(text)
            primary = detected[0]
            alternatives = [{candidate.lang: candidate.prob} for candidate in detected[1:3]]
            return {
                "language": primary.lang,
                "confidence": primary.prob,
                "alternatives": alternatives,
            }
        except Exception:
            # Graceful fallback with minimal signal
            return {
                "language": "en",
                "confidence": 0.5,
                "alternatives": [],
            }

    # ------------------------------------------------------------------
    # Session & analytics helpers used by orchestration routers
    # ------------------------------------------------------------------

    async def start_session(self, config: dict[str, Any] | None = None) -> str:
        """Create a lightweight real-time translation session."""
        session_id = (config or {}).get("session_id") or uuid4().hex
        self._sessions[session_id] = {
            "config": config or {},
            "created_at": datetime.now(UTC).isoformat(),
            "last_activity": datetime.now(UTC).isoformat(),
            "translations": 0,
        }
        return session_id

    async def stop_session(self, session_id: str) -> dict[str, Any]:
        """Stop a real-time session and return summary stats."""
        session = self._sessions.pop(session_id, None)
        if not session:
            return {"status": "not_found", "session_id": session_id}

        duration = (
            datetime.now(UTC) - datetime.fromisoformat(session["created_at"])
        ).total_seconds()

        return {
            "status": "stopped",
            "session_id": session_id,
            "duration_seconds": duration,
            "translations": session.get("translations", 0),
        }

    async def realtime_translate(
        self, session_id: str, text: str, target_language: str
    ) -> dict[str, Any]:
        """Translate text within a session context."""
        if session_id not in self._sessions:
            await self.start_session({"session_id": session_id})

        result = await self.translate(
            text=text,
            source_language=None,
            target_language=target_language,
            session_id=session_id,
        )

        session = self._sessions[session_id]
        session["translations"] = session.get("translations", 0) + 1
        session["last_activity"] = datetime.now(UTC).isoformat()

        return result

    async def get_statistics(self) -> dict[str, Any]:
        """Expose accumulated translation metrics."""
        total = self._metrics.get("total_translations", 0)
        total_time = self._metrics.get("total_processing_time", 0.0)
        avg_time = total_time / total if total else 0.0

        return {
            **self._metrics,
            "average_processing_time": avg_time,
            "active_sessions": len(self._sessions),
            "timestamp": datetime.now(UTC).isoformat(),
        }

    async def get_models(self) -> list[dict[str, Any]]:
        """Return available models (basic defaults for embedded mode)."""
        models = []
        if self._service and hasattr(self._service, "backend_priority"):
            for backend in getattr(self._service, "backend_priority", []):
                models.append(
                    {
                        "name": backend,
                        "description": f"Embedded backend: {backend}",
                        "mode": "embedded",
                    }
                )

        if not models:
            models.append(
                {
                    "name": "embedded",
                    "description": "Embedded translation backend",
                    "mode": "embedded",
                }
            )
        return models

    async def get_device_info(self) -> dict[str, Any]:
        """Return device/runtime info for diagnostics."""
        device = getattr(self._service, "inference_client", None)
        backend = None
        if device:
            backend = getattr(device, "backend", None)

        return {
            "device": backend or "cpu",
            "mode": "embedded",
            "details": {
                "module_imported": TRANSLATION_MODULE_AVAILABLE,
                "fallback_mode": bool(
                    getattr(self._service, "fallback_mode", False) if self._service else True
                ),
            },
        }

    async def get_translation_quality(
        self, original: str, translated: str, source_lang: str, target_lang: str
    ) -> dict[str, Any]:
        """
        Compute a simple quality estimate using sequence matching. This is not a
        substitute for dedicated quality models but gives quick feedback.
        """
        matcher = SequenceMatcher(None, original.lower().strip(), translated.lower().strip())
        score = matcher.ratio()
        return {
            "score": score,
            "method": "sequence_matcher",
            "source_language": source_lang,
            "target_language": target_lang,
        }


_TRANSLATION_SINGLETON: UnifiedTranslationService | None = None


def get_unified_translation_service() -> UnifiedTranslationService:
    """Return the module-level singleton facade."""
    global _TRANSLATION_SINGLETON
    if _TRANSLATION_SINGLETON is None:
        _TRANSLATION_SINGLETON = UnifiedTranslationService()
    return _TRANSLATION_SINGLETON


def reset_unified_translation_service() -> None:
    """Reset the singleton (for testing only)"""
    global _TRANSLATION_SINGLETON
    _TRANSLATION_SINGLETON = None


__all__ = [
    "UnifiedTranslationError",
    "UnifiedTranslationService",
    "get_unified_translation_service",
    "reset_unified_translation_service",
]
