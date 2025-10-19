"""
Translation Service Client

Handles communication with the translation service.
Provides methods for text translation, language detection, and quality assessment.
"""

import logging
import asyncio
import json
from typing import Dict, Any, Optional, List
import aiohttp
from pydantic import BaseModel

logger = logging.getLogger(__name__)


class TranslationRequest(BaseModel):
    """Request model for translation"""

    text: str
    source_language: Optional[str] = None  # auto-detect if None
    target_language: str
    model: str = "default"
    quality: str = "balanced"  # fast, balanced, quality


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
        self.base_url = base_url or self._get_base_url()
        timeout_seconds = timeout or self._get_timeout()
        self.session: Optional[aiohttp.ClientSession] = None
        self.timeout = aiohttp.ClientTimeout(total=timeout_seconds)

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
        if self.session is None or self.session.closed:
            self.session = aiohttp.ClientSession(timeout=self.timeout)
        return self.session

    async def close(self):
        """Close the client session"""
        if self.session and not self.session.closed:
            await self.session.close()

    async def health_check(self) -> Dict[str, Any]:
        """Check translation service health"""
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
        quality: str = "balanced"
    ) -> Dict[str, TranslationResponse]:
        """
        Translate text to multiple target languages
        
        Args:
            text: Text to translate
            source_language: Source language (auto-detect if None)
            target_languages: List of target language codes
            quality: Translation quality level
            
        Returns:
            Dictionary mapping language codes to translation responses
        """
        try:
            # Create translation requests for each target language
            requests = []
            for target_lang in target_languages:
                requests.append(TranslationRequest(
                    text=text,
                    source_language=source_language,
                    target_language=target_lang,
                    quality=quality
                ))
            
            # If translation service supports batch translation, use it
            try:
                logger.info(f"Attempting batch translation for {len(target_languages)} languages: {target_languages}")
                batch_results = await self.translate_batch(requests)
                logger.info(f"Batch translation successful, got {len(batch_results)} results")
                result_dict = {}
                for i, target_lang in enumerate(target_languages):
                    if i < len(batch_results):
                        result_dict[target_lang] = batch_results[i]
                    else:
                        # Fallback for missing results
                        logger.warning(f"Missing batch result for {target_lang} (index {i})")
                        result_dict[target_lang] = TranslationResponse(
                            translated_text="Translation failed",
                            source_language=source_language or "auto",
                            target_language=target_lang,
                            confidence=0.0,
                            processing_time=0.0,
                            model_used="error"
                        )
                return result_dict
                
            except Exception as batch_error:
                logger.warning(f"Batch translation failed, falling back to individual: {batch_error}")
                
                # Fallback to individual translations
                result_dict = {}
                for target_lang in target_languages:
                    try:
                        logger.info(f"Attempting individual translation to {target_lang}")
                        request = TranslationRequest(
                            text=text,
                            source_language=source_language,
                            target_language=target_lang,
                            quality=quality
                        )
                        translation_result = await self.translate(request)
                        logger.info(f"Individual translation to {target_lang} successful: {translation_result.translated_text[:50]}...")
                        result_dict[target_lang] = translation_result
                    except Exception as individual_error:
                        logger.error(f"Translation to {target_lang} failed: {individual_error}")
                        # Add error placeholder
                        result_dict[target_lang] = TranslationResponse(
                            translated_text=f"Translation to {target_lang} failed: {str(individual_error)}",
                            source_language=source_language or "auto",
                            target_language=target_lang,
                            confidence=0.0,
                            processing_time=0.0,
                            model_used="error"
                        )
                
                return result_dict
                
        except Exception as e:
            logger.error(f"Multi-language translation failed: {e}")
            # Return error responses for all target languages
            result_dict = {}
            for target_lang in target_languages:
                result_dict[target_lang] = TranslationResponse(
                    translated_text=f"Translation failed: {str(e)}",
                    source_language=source_language or "auto",
                    target_language=target_lang,
                    confidence=0.0,
                    processing_time=0.0,
                    model_used="error"
                )
            return result_dict

    async def get_analytics(self) -> Dict[str, Any]:
        """Get translation analytics for analytics API"""
        try:
            session = await self._get_session()
            
            # Try the health endpoint as translation service might not have dedicated analytics
            async with session.get(f"{self.base_url}/api/health") as response:
                if response.status == 200:
                    result = await response.json()
                    return result
                else:
                    # Try to get basic stats instead
                    stats = await self.get_statistics()
                    if "error" not in stats:
                        # Convert basic stats to analytics format
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
                            "supported_languages": len(await self.get_supported_languages()),
                            "timestamp": stats.get("timestamp", 0)
                        }
                    else:
                        # Return default analytics
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
                            "supported_languages": 0,
                            "timestamp": 0,
                            "error": f"HTTP {response.status}"
                        }
                    
        except Exception as e:
            logger.error(f"Failed to get translation analytics: {e}")
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
                "supported_languages": 0,
                "timestamp": 0,
                "error": str(e)
            }
