"""
Translation Service Client

Handles communication with the translation service.
Provides methods for text translation, language detection, and quality assessment.
"""

import logging
import asyncio
import json
import os
from datetime import datetime
from typing import Dict, Any, Optional, List
import aiohttp
from pydantic import BaseModel

from internal_services.translation import (
    UnifiedTranslationError,
    get_unified_translation_service,
)

logger = logging.getLogger(__name__)


class TranslationRequest(BaseModel):
    """Request model for translation"""

    text: str
    source_language: Optional[str] = None  # auto-detect if None
    target_language: str
    model: str = "default"
    quality: str = "balanced"  # fast, balanced, quality
    session_id: Optional[str] = None


class TranslationResponse(BaseModel):
    """Response model for translation"""

    translated_text: str
    source_language: Optional[str] = "auto"  # Allow None, default to "auto"
    target_language: str
    confidence: float = 0.95  # Default confidence
    processing_time: float = 0.0  # Default processing time
    model_used: str = "default"  # Default model
    backend_used: Optional[str] = None  # Backend that performed translation
    session_id: Optional[str] = None  # Session tracking
    timestamp: Optional[str] = None  # Request timestamp

    model_config = {"protected_namespaces": (), "extra": "ignore"}


class LanguageDetectionResponse(BaseModel):
    """Response model for language detection"""

    language: str
    confidence: float
    alternatives: List[Dict[str, float]]


class TranslationServiceClient:
    """Client for the Translation service"""

    def __init__(
        self,
        base_url: Optional[str] = None,
        timeout: Optional[float] = None,
        config_manager=None,
    ):
        self.config_manager = config_manager
        resolved_base_url = base_url or self._get_base_url()
        # Treat special markers as request to use embedded mode exclusively.
        if resolved_base_url and resolved_base_url.lower() in {"embedded", "internal", "local"}:
            resolved_base_url = None

        self.base_url = resolved_base_url
        timeout_seconds = timeout or self._get_timeout()
        self.session: Optional[aiohttp.ClientSession] = None
        self.timeout = aiohttp.ClientTimeout(total=timeout_seconds)
        # EMBEDDED SERVICE DISABLED - Use only remote translation service
        self._embedded_service = None
        self._prefer_embedded = False
        self._embedded_failure_logged = False

    # ------------------------------------------------------------------
    # Mode helpers
    # ------------------------------------------------------------------

    def _embedded_enabled(self) -> bool:
        return (
            self._prefer_embedded
            and self._embedded_service is not None
            and self._embedded_service.is_available()
        )

    def _remote_enabled(self) -> bool:
        return bool(self.base_url and self.base_url.startswith("http"))

    async def _translate_embedded(self, request: TranslationRequest) -> Optional[TranslationResponse]:
        if not self._embedded_enabled():
            return None
        try:
            result = await self._embedded_service.translate(
                text=request.text,
                source_language=request.source_language,
                target_language=request.target_language,
                session_id=request.session_id,
                quality=request.quality,
                model=request.model,
            )
            return TranslationResponse(
                translated_text=result["translated_text"],
                source_language=result.get("source_language", request.source_language or "auto"),
                target_language=result.get("target_language", request.target_language),
                confidence=float(result.get("confidence", 0.0)),
                processing_time=float(result.get("processing_time", 0.0)),
                model_used=result.get("model_used", request.model),
                backend_used=result.get("backend_used"),
                session_id=result.get("session_id", request.session_id),
                timestamp=result.get("timestamp"),
            )
        except UnifiedTranslationError as exc:
            if not self._embedded_failure_logged:
                logger.warning("Embedded translation unavailable: %s", exc)
                self._embedded_failure_logged = True
            return None
        except Exception as exc:
            logger.exception("Embedded translation failed unexpectedly: %s", exc)
            return None

    def _get_base_url(self) -> str:
        """Get the translation service base URL from configuration"""
        if self.config_manager:
            return self.config_manager.get_service_url(
                "translation", "http://localhost:5003"
            )
        return "http://localhost:5003"

    def _get_timeout(self) -> int:
        """Resolve translation timeout"""
        if self.config_manager:
            return self.config_manager.get("services.translation.timeout", 60)
        return 60

    async def _get_session(self) -> aiohttp.ClientSession:
        """Get or create aiohttp session"""
        if not self._remote_enabled():
            raise RuntimeError("Remote translation service not configured")
        if self.session is None or self.session.closed:
            self.session = aiohttp.ClientSession(timeout=self.timeout)
        return self.session

    async def close(self):
        """Close the client session"""
        if self.session and not self.session.closed:
            await self.session.close()

    async def health_check(self) -> Dict[str, Any]:
        """Check translation service health"""
        if self._embedded_enabled():
            try:
                details = await self._embedded_service.health()
                return {
                    "status": details.get("status", "healthy"),
                    "service": "translation",
                    "mode": "embedded",
                    "details": details,
                }
            except Exception as exc:
                if not self._embedded_failure_logged:
                    logger.warning("Embedded translation health check failed: %s", exc)
                    self._embedded_failure_logged = True
        if not self._remote_enabled():
            return {
                "status": "unavailable",
                "service": "translation",
                "mode": "embedded" if self._embedded_enabled() else "disabled",
                "error": self._embedded_service.last_error if self._embedded_enabled() else "no backend configured",
            }
        try:
            session = await self._get_session()
            async with session.get(f"{self.base_url}/api/health") as response:
                if response.status == 200:
                    data = await response.json()
                    return {
                        "status": "healthy",
                        "service": "translation",
                        "url": self.base_url,
                        "details": data,
                    }
                else:
                    return {
                        "status": "unhealthy",
                        "service": "translation",
                        "url": self.base_url,
                        "error": f"HTTP {response.status}",
                    }
        except Exception as e:
            logger.error(f"Translation service health check failed: {e}")
            return {
                "status": "unhealthy",
                "service": "translation",
                "url": self.base_url,
                "error": str(e),
            }

    async def get_supported_languages(self) -> List[Dict[str, str]]:
        """Get supported languages"""
        if self._embedded_enabled():
            try:
                return await self._embedded_service.get_supported_languages()
            except Exception as exc:
                logger.debug("Embedded supported language fetch failed: %s", exc)

        if not self._remote_enabled():
            return self._get_default_languages()

        try:
            session = await self._get_session()
            async with session.get(f"{self.base_url}/api/languages") as response:
                if response.status == 200:
                    data = await response.json()
                    return data.get("languages", [])
                else:
                    logger.error(f"Failed to get languages: HTTP {response.status}")
                    return self._get_default_languages()
        except Exception as e:
            logger.error(f"Failed to get languages: {e}")
            return self._get_default_languages()

    def _get_default_languages(self) -> List[Dict[str, str]]:
        """Get default supported languages"""
        return [
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

    async def get_device_info(self) -> Dict[str, Any]:
        """Get current device information (CPU/GPU) from translation service"""
        if self._embedded_enabled():
            try:
                return await self._embedded_service.get_device_info()
            except Exception as exc:
                logger.debug("Embedded device info lookup failed: %s", exc)
        if not self._remote_enabled():
            return {"device": "cpu", "mode": "embedded", "status": "fallback"}
        try:
            session = await self._get_session()
            async with session.get(f"{self.base_url}/api/device-info") as response:
                if response.status == 200:
                    data = await response.json()
                    return data
                else:
                    logger.error(f"Failed to get device info: HTTP {response.status}")
                    return {"device": "unknown", "status": "error"}
        except Exception as e:
            logger.error(f"Failed to get device info: {e}")
            return {"device": "unknown", "status": "error", "error": str(e)}

    async def detect_language(self, text: str) -> LanguageDetectionResponse:
        """Detect language of text"""
        if self._embedded_enabled():
            try:
                result = await self._embedded_service.detect_language(text)
                alternatives_raw = result.get("alternatives", [])
                alternatives: List[Dict[str, float]] = []
                for item in alternatives_raw:
                    if isinstance(item, dict):
                        alternatives.append(item)
                return LanguageDetectionResponse(
                    language=result.get("language", "en"),
                    confidence=float(result.get("confidence", 0.5)),
                    alternatives=alternatives,
                )
            except Exception as exc:
                logger.debug("Embedded language detection failed: %s", exc)

        if not self._remote_enabled():
            return LanguageDetectionResponse(
                language="en", confidence=0.5, alternatives=[{"en": 0.5}]
            )
        try:
            session = await self._get_session()

            request_data = {"text": text}

            async with session.post(
                f"{self.base_url}/api/detect", json=request_data
            ) as response:
                if response.status == 200:
                    result = await response.json()
                    return LanguageDetectionResponse(**result)
                else:
                    error_text = await response.text()
                    raise Exception(
                        f"Language detection failed: HTTP {response.status} - {error_text}"
                    )

        except Exception as e:
            logger.error(f"Language detection failed: {e}")
            # Return default detection
            return LanguageDetectionResponse(
                language="en", confidence=0.5, alternatives=[{"en": 0.5}]
            )

    async def translate(self, request: TranslationRequest) -> TranslationResponse:
        """Translate text"""
        embedded_response = await self._translate_embedded(request)
        if embedded_response is not None:
            return embedded_response

        if not self._remote_enabled():
            raise UnifiedTranslationError(
                "Remote translation service unavailable and embedded translation failed"
            )
        try:
            session = await self._get_session()

            request_data = {
                "text": request.text,
                "source_language": request.source_language,
                "target_language": request.target_language,
                "model": request.model,
                "quality": request.quality,
            }

            async with session.post(
                f"{self.base_url}/api/translate", json=request_data
            ) as response:
                if response.status == 200:
                    result = await response.json()
                    return TranslationResponse(**result)
                else:
                    error_text = await response.text()
                    raise Exception(
                        f"Translation failed: HTTP {response.status} - {error_text}"
                    )

        except Exception as e:
            logger.error(f"Translation failed: {e}")
            raise

    async def translate_batch(
        self, requests: List[TranslationRequest]
    ) -> List[TranslationResponse]:
        """Translate multiple texts in batch"""
        if self._embedded_enabled():
            results: List[TranslationResponse] = []
            for req in requests:
                response = await self._translate_embedded(req)
                if response is None:
                    results = []
                    break
                results.append(response)
            if results:
                return results

        if not self._remote_enabled():
            raise UnifiedTranslationError(
                "Batch translation unavailable without embedded or remote backend"
            )
        try:
            session = await self._get_session()

            request_data = {"requests": [req.dict() for req in requests]}

            async with session.post(
                f"{self.base_url}/api/translate/batch", json=request_data
            ) as response:
                if response.status == 200:
                    results = await response.json()
                    return [
                        TranslationResponse(**result)
                        for result in results["translations"]
                    ]
                else:
                    error_text = await response.text()
                    raise Exception(
                        f"Batch translation failed: HTTP {response.status} - {error_text}"
                    )

        except Exception as e:
            logger.error(f"Batch translation failed: {e}")
            raise

    async def get_translation_quality(
        self, original: str, translated: str, source_lang: str, target_lang: str
    ) -> Dict[str, Any]:
        """Get translation quality metrics"""
        if self._embedded_enabled():
            try:
                return await self._embedded_service.get_translation_quality(
                    original, translated, source_lang, target_lang
                )
            except Exception as exc:
                logger.debug("Embedded quality assessment failed: %s", exc)
        if not self._remote_enabled():
            return {
                "score": 0.0,
                "method": "fallback",
                "error": "No translation backend available for quality scoring",
            }
        try:
            session = await self._get_session()

            request_data = {
                "original": original,
                "translated": translated,
                "source_language": source_lang,
                "target_language": target_lang,
            }

            async with session.post(
                f"{self.base_url}/api/quality", json=request_data
            ) as response:
                if response.status == 200:
                    return await response.json()
                else:
                    error_text = await response.text()
                    logger.error(
                        f"Quality assessment failed: HTTP {response.status} - {error_text}"
                    )
                    return {"error": error_text}

        except Exception as e:
            logger.error(f"Quality assessment failed: {e}")
            return {"error": str(e)}

    async def start_realtime_session(self, session_config: Dict[str, Any]) -> str:
        """Start a real-time translation session"""
        if self._embedded_enabled():
            return await self._embedded_service.start_session(session_config)

        if not self._remote_enabled():
            raise UnifiedTranslationError("Realtime translation session not available without backend")
        try:
            session = await self._get_session()

            async with session.post(
                f"{self.base_url}/api/realtime/start", json=session_config
            ) as response:
                if response.status == 200:
                    result = await response.json()
                    return result["session_id"]
                else:
                    error_text = await response.text()
                    raise Exception(
                        f"Failed to start realtime session: HTTP {response.status} - {error_text}"
                    )

        except Exception as e:
            logger.error(f"Failed to start realtime session: {e}")
            raise

    async def translate_realtime(
        self, session_id: str, text: str, target_language: str
    ) -> Optional[Dict[str, Any]]:
        """Translate text in real-time session"""
        if self._embedded_enabled():
            try:
                return await self._embedded_service.realtime_translate(
                    session_id=session_id,
                    text=text,
                    target_language=target_language,
                )
            except Exception as exc:
                logger.debug("Embedded realtime translation failed: %s", exc)

        if not self._remote_enabled():
            return None
        try:
            session = await self._get_session()

            request_data = {
                "session_id": session_id,
                "text": text,
                "target_language": target_language,
            }

            async with session.post(
                f"{self.base_url}/api/realtime/translate", json=request_data
            ) as response:
                if response.status == 200:
                    return await response.json()
                else:
                    error_text = await response.text()
                    logger.error(
                        f"Realtime translation failed: HTTP {response.status} - {error_text}"
                    )
                    return None

        except Exception as e:
            logger.error(f"Realtime translation failed: {e}")
            return None

    async def stop_realtime_session(self, session_id: str) -> Dict[str, Any]:
        """Stop a real-time translation session"""
        if self._embedded_enabled():
            return await self._embedded_service.stop_session(session_id)

        if not self._remote_enabled():
            return {"status": "not_available", "session_id": session_id}
        try:
            session = await self._get_session()

            async with session.post(
                f"{self.base_url}/api/realtime/stop", json={"session_id": session_id}
            ) as response:
                if response.status == 200:
                    return await response.json()
                else:
                    error_text = await response.text()
                    logger.error(
                        f"Failed to stop realtime session: HTTP {response.status} - {error_text}"
                    )
                    return {"status": "error", "message": error_text}

        except Exception as e:
            logger.error(f"Failed to stop realtime session: {e}")
            return {"status": "error", "message": str(e)}

    async def get_statistics(self) -> Dict[str, Any]:
        """Get service statistics"""
        if self._embedded_enabled():
            try:
                return await self._embedded_service.get_statistics()
            except Exception as exc:
                logger.debug("Embedded statistics retrieval failed: %s", exc)

        if not self._remote_enabled():
            return {
                "error": "No translation backend available",
                "timestamp": datetime.utcnow().isoformat(),
            }
        try:
            session = await self._get_session()

            # Try the health endpoint first as it might have some stats
            async with session.get(f"{self.base_url}/api/health") as response:
                if response.status == 200:
                    return await response.json()
                else:
                    return {"error": f"HTTP {response.status}"}

        except Exception as e:
            logger.error(f"Failed to get statistics: {e}")
            return {"error": str(e)}

    async def get_models(self) -> List[Dict[str, Any]]:
        """Get available translation models"""
        if self._embedded_enabled():
            try:
                return await self._embedded_service.get_models()
            except Exception as exc:
                logger.debug("Embedded model list retrieval failed: %s", exc)

        if not self._remote_enabled():
            return [{"name": "embedded", "description": "Embedded translation backend"}]
        try:
            session = await self._get_session()

            async with session.get(f"{self.base_url}/api/models") as response:
                if response.status == 200:
                    data = await response.json()
                    return data.get("models", [])
                else:
                    logger.error(f"Failed to get models: HTTP {response.status}")
                    return [{"name": "default", "description": "Default model"}]

        except Exception as e:
            logger.error(f"Failed to get models: {e}")
            return [{"name": "default", "description": "Default model"}]

    async def translate_to_multiple_languages(
        self,
        text: str,
        source_language: Optional[str],
        target_languages: List[str],
        quality: str = "balanced",
        session_id: Optional[str] = None
    ) -> Dict[str, TranslationResponse]:
        """
        OPTIMIZED: Translate text to multiple target languages using batched endpoint.

        This uses the new /api/translate/multi endpoint which processes all languages
        in a single HTTP request, significantly reducing overhead.

        Args:
            text: Text to translate
            source_language: Source language (auto-detect if None)
            target_languages: List of target language codes
            quality: Translation quality level
            session_id: Optional session ID for tracking

        Returns:
            Dictionary mapping language codes to translation responses
        """
        # Try embedded service first (with parallel processing)
        if self._embedded_enabled():
            return await self._translate_multi_embedded(
                text, source_language, target_languages, session_id
            )

        # Use optimized multi-language endpoint
        if self._remote_enabled():
            try:
                logger.info(f"Using optimized multi-language endpoint for {len(target_languages)} languages: {target_languages}")

                session = await self._get_session()

                request_data = {
                    "text": text,
                    "source_language": source_language or "auto",
                    "target_languages": target_languages,
                    "quality": quality
                }

                # Add optional parameters
                if session_id:
                    request_data["session_id"] = session_id
                if model:
                    request_data["model"] = model

                logger.info(f"Multi-language request: model={model}, quality={quality}")

                response = await session.post(
                    f"{self.base_url}/api/translate/multi",
                    json=request_data
                )

                if response.status == 200:
                    data = await response.json()

                    # Convert to TranslationResponse objects
                    result_dict = {}
                    for lang, translation_data in data["translations"].items():
                        if "error" in translation_data:
                            logger.error(f"Translation to {lang} had error: {translation_data['error']}")
                            result_dict[lang] = TranslationResponse(
                                translated_text=f"Error: {translation_data['error']}",
                                source_language=data.get("source_language", source_language or "auto"),
                                target_language=lang,
                                confidence=0.0,
                                processing_time=translation_data.get("processing_time", 0.0),
                                model_used="error",
                                backend_used="error"
                            )
                        else:
                            result_dict[lang] = TranslationResponse(
                                translated_text=translation_data["translated_text"],
                                source_language=data.get("source_language", source_language or "auto"),
                                target_language=lang,
                                confidence=translation_data.get("confidence", 0.0),
                                processing_time=translation_data.get("processing_time", 0.0),
                                model_used=translation_data.get("backend_used", "unknown"),
                                backend_used=translation_data.get("backend_used"),
                                session_id=session_id,
                                timestamp=data.get("timestamp")
                            )

                    logger.info(f"Multi-language translation successful: {len(result_dict)}/{len(target_languages)} languages")
                    return result_dict

                else:
                    error_text = await response.text()
                    logger.error(f"Multi-language endpoint failed with status {response.status}: {error_text}")
                    # Fall through to fallback

            except Exception as e:
                logger.warning(f"Multi-language endpoint failed: {e}, falling back to individual translations")

        # Fallback: individual translations
        return await self._fallback_individual_translations(
            text, source_language, target_languages, quality, session_id
        )

    async def _translate_multi_embedded(
        self,
        text: str,
        source_language: Optional[str],
        target_languages: List[str],
        session_id: Optional[str]
    ) -> Dict[str, TranslationResponse]:
        """Translate to multiple languages using embedded service (parallel)"""
        if not self._embedded_enabled():
            return {}

        try:
            # Parallel translation to all languages
            async def translate_one(target_lang: str) -> tuple[str, TranslationResponse]:
                result = await self._translate_embedded(TranslationRequest(
                    text=text,
                    source_language=source_language,
                    target_language=target_lang,
                    session_id=session_id
                ))
                return target_lang, result

            # Use semaphore to limit concurrency
            semaphore = asyncio.Semaphore(10)

            async def translate_with_limit(target_lang: str):
                async with semaphore:
                    return await translate_one(target_lang)

            results = await asyncio.gather(
                *[translate_with_limit(lang) for lang in target_languages],
                return_exceptions=True
            )

            # Collect results
            result_dict = {}
            for result in results:
                if isinstance(result, Exception):
                    logger.error(f"Embedded translation task failed: {result}")
                    continue

                target_lang, translation_response = result
                result_dict[target_lang] = translation_response

            return result_dict

        except Exception as e:
            logger.error(f"Embedded multi-language translation failed: {e}")
            return {}

    async def _fallback_individual_translations(
        self,
        text: str,
        source_language: Optional[str],
        target_languages: List[str],
        quality: str,
        session_id: Optional[str]
    ) -> Dict[str, TranslationResponse]:
        """Fallback: translate each language individually"""
        logger.warning("Using fallback: individual translation requests")

        result_dict = {}
        for target_lang in target_languages:
            try:
                request = TranslationRequest(
                    text=text,
                    source_language=source_language,
                    target_language=target_lang,
                    quality=quality,
                    session_id=session_id
                )
                translation_result = await self.translate(request)
                result_dict[target_lang] = translation_result

            except Exception as individual_error:
                logger.error(f"Individual translation to {target_lang} failed: {individual_error}")
                result_dict[target_lang] = TranslationResponse(
                    translated_text=f"Translation failed: {str(individual_error)}",
                    source_language=source_language or "auto",
                    target_language=target_lang,
                    confidence=0.0,
                    processing_time=0.0,
                    model_used="error",
                    backend_used="error"
                )

        return result_dict

    async def get_analytics(self) -> Dict[str, Any]:
        """Get translation analytics for analytics API"""
        stats = await self.get_statistics()
        if "error" not in stats:
            supported_languages = await self.get_supported_languages()
            return {
                "total_translations": stats.get("total_translations", 0),
                "successful_translations": stats.get("successful_translations", 0),
                "failed_translations": stats.get("failed_translations", 0),
                "average_processing_time_ms": stats.get("average_processing_time", 0) * 1000,
                "translation_quality_score": stats.get("translation_quality_score", 0.0),
                "language_pairs_processed": stats.get("language_pairs_processed", {}),
                "model_performance": stats.get("model_performance", {}),
                "error_rate": stats.get("failed_translations", 0) / max(1, stats.get("total_translations", 1)),
                "throughput_per_minute": stats.get("throughput_per_minute", 0),
                "active_sessions": stats.get("active_sessions", 0),
                "supported_languages": len(supported_languages),
                "timestamp": stats.get("timestamp", datetime.utcnow().isoformat()),
            }

        return {
            "total_translations": 0,
            "successful_translations": 0,
            "failed_translations": 0,
            "average_processing_time_ms": 0,
            "translation_quality_score": 0.0,
            "language_pairs_processed": {},
            "model_performance": {},
            "error_rate": 0.0,
            "throughput_per_minute": 0,
            "active_sessions": 0,
            "supported_languages": len(await self.get_supported_languages()),
            "timestamp": datetime.utcnow().isoformat(),
            "error": stats.get("error", "statistics unavailable"),
        }
