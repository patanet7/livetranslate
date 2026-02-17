#!/usr/bin/env python3
"""
Translation Service Module

Provides translation capabilities using local inference (vLLM, Ollama) with fallback to external APIs.
Integrates with the shared inference infrastructure for efficient model management.
"""

import asyncio
import json
import os

# Import shared inference clients
import sys
from collections.abc import AsyncGenerator
from dataclasses import dataclass
from datetime import datetime

sys.path.append(os.path.join(os.path.dirname(__file__), "..", "..", "shared", "src"))

from collections import deque

from inference import BaseInferenceClient, get_inference_client_async
from livetranslate_common.logging import get_logger
from local_translation import LocalTranslationService

logger = get_logger()


class TranslationContinuityManager:
    """
    Manages translation context and overlap detection for continuous conversation translation.
    Optimized for Chinese→English translation with smart sentence boundary detection.
    """

    def __init__(self, max_context_items: int = 5, max_context_seconds: int = 30):
        self.max_context_items = max_context_items
        self.max_context_seconds = max_context_seconds

        # Context storage (text overlap detection removed - handled by whisper service)
        self.translation_history = deque(maxlen=max_context_items)
        self.sentence_buffer = ""

        # Chinese sentence patterns for buffering complete sentences
        # Use unicode escapes to avoid RUF001 ambiguous character warnings
        self.chinese_sentence_endings = ["\u3002", "\uff01", "\uff1f", "\uff1b", "\u2026\u2026"]
        self.chinese_pause_markers = ["\uff0c", "\u3001", "\uff1a", '"', '"', "\u201c", "\u201d"]

        logger.info(
            f"Translation continuity manager initialized (context: {max_context_items} items, {max_context_seconds}s)"
        )

    def detect_sentence_boundaries(self, text: str) -> tuple[str, str]:
        """
        Detect complete sentences in Chinese text.

        Returns:
            Tuple of (complete_sentences, remaining_partial_text)
        """
        if not text:
            return "", ""

        # Find the last sentence ending
        last_ending_pos = -1
        for ending in self.chinese_sentence_endings:
            pos = text.rfind(ending)
            if pos > last_ending_pos:
                last_ending_pos = pos

        if last_ending_pos >= 0:
            # Found complete sentence(s)
            complete = text[: last_ending_pos + 1]
            remaining = text[last_ending_pos + 1 :].strip()
            logger.debug(f"Complete sentences: '{complete}', Remaining: '{remaining}'")
            return complete, remaining

        # No complete sentences, check for natural pause points
        for pause in self.chinese_pause_markers:
            pos = text.rfind(pause)
            if pos > len(text) * 0.7:  # Pause point in last 30% of text
                complete = text[: pos + 1]
                remaining = text[pos + 1 :].strip()
                logger.debug(f"Natural break at pause: '{complete}', Remaining: '{remaining}'")
                return complete, remaining

        # Return everything as partial if no good break point
        return "", text

    def build_context_prompt(self, current_text: str) -> str:
        """Build context prompt from recent translation history."""
        if not self.translation_history:
            return current_text

        # Build context from recent translations
        context_parts = []
        for item in list(self.translation_history)[-3:]:  # Last 3 for context
            context_parts.append(f"Chinese: {item['source']}\nEnglish: {item['translation']}")

        context_str = "\n\n".join(context_parts)

        prompt = f"""Continue translating this Chinese conversation to natural English. Maintain conversational flow and context.

Previous conversation context:
{context_str}

Current Chinese text to translate:
{current_text}

Provide a natural English translation that flows smoothly from the previous context:"""

        return prompt

    def process_streaming_text(self, clean_text: str, chunk_id: str | None = None) -> dict:
        """
        Process clean streaming text with context management and sentence buffering.
        Note: Text deduplication is now handled by whisper service.

        Args:
            clean_text: Clean transcribed text (already deduplicated by whisper service)
            chunk_id: Optional chunk identifier

        Returns:
            Dict with translation result or buffering status
        """
        logger.info(f"Processing clean chunk {chunk_id}: '{clean_text[:50]}...'")

        # Add to sentence buffer (text is already clean from whisper service)
        self.sentence_buffer += clean_text

        # Check for complete sentences
        complete_sentences, remaining = self.detect_sentence_boundaries(self.sentence_buffer)

        if complete_sentences:
            # Update buffer with remaining text
            self.sentence_buffer = remaining

            return {
                "status": "ready_for_translation",
                "text": complete_sentences,
                "context_prompt": self.build_context_prompt(complete_sentences),
                "chunk_id": chunk_id,
            }
        else:
            # Buffer incomplete text
            buffer_length = len(self.sentence_buffer)
            logger.debug(
                f"Buffering incomplete text ({buffer_length} chars): '{self.sentence_buffer[:30]}...'"
            )

            return {"status": "buffering", "buffer_length": buffer_length, "chunk_id": chunk_id}

    def store_translation_result(self, source_text: str, translation: str):
        """Store translation result in context history."""
        self.translation_history.append(
            {
                "source": source_text,
                "translation": translation,
                "timestamp": datetime.now().timestamp(),
            }
        )
        logger.debug(
            f"Stored translation in context: '{source_text[:30]}...' → '{translation[:30]}...'"
        )

    def clear_context(self):
        """Clear all context and buffers."""
        self.translation_history.clear()
        self.sentence_buffer = ""
        logger.info("Translation context cleared")


@dataclass
class TranslationRequest:
    """Translation request data structure"""

    text: str
    source_language: str = "auto"
    target_language: str = "en"
    session_id: str | None = None
    streaming: bool = False
    confidence_threshold: float = 0.8
    preserve_formatting: bool = True
    context: str | None = None


@dataclass
class TranslationResult:
    """Translation result data structure"""

    translated_text: str
    source_language: str
    target_language: str
    confidence_score: float
    processing_time: float
    backend_used: str
    session_id: str | None = None
    timestamp: str = None

    def __post_init__(self):
        if self.timestamp is None:
            self.timestamp = datetime.now().isoformat()


class TranslationService:
    """
    Main Translation Service

    Provides unified translation interface with multiple backends:
    - Local inference (vLLM, Ollama) for privacy and performance
    - Basic fallback for unsupported cases
    """

    def __init__(self, config: dict | None = None):
        """Initialize translation service with configuration"""
        self.config = config or self._load_config()
        self.inference_client: BaseInferenceClient | None = None
        self.local_service: LocalTranslationService | None = None
        self.sessions: dict[str, dict] = {}

        # Translation continuity management for streaming
        self.continuity_managers: dict[str, TranslationContinuityManager] = {}

        # Backend priority order (Triton first for production)
        self.backend_priority = [
            "local_inference",  # Triton/vLLM/Ollama
            "fallback",  # Basic fallback
        ]

        logger.info("Translation Service initialized")

    async def initialize(self):
        """Initialize all translation backends"""
        try:
            # Initialize local inference client
            if self.config.get("use_local_inference", True):
                self.inference_client = await get_inference_client_async()

                if self.inference_client:
                    self.local_service = LocalTranslationService(self.inference_client)
                    await self.local_service.initialize()
                    logger.info("Local inference translation service initialized")

            # Could add external API clients here if needed in the future
            # (OpenAI, Anthropic, Google Translate, etc.)

            logger.info("Translation service fully initialized")

        except Exception as e:
            logger.error(f"Failed to initialize translation service: {e}")
            raise

    async def translate(self, request: TranslationRequest) -> TranslationResult:
        """
        Translate text using the best available backend

        Args:
            request: Translation request with text and parameters

        Returns:
            Translation result with confidence and metadata
        """
        start_time = datetime.now()

        # Check if running in fallback mode
        if hasattr(self, "fallback_mode") and self.fallback_mode:
            logger.info("Running in fallback mode - returning mock translation")
            return TranslationResult(
                translated_text=f"[MOCK TRANSLATION] {request.text} -> {request.target_language}",
                source_language=request.source_language or "auto",
                target_language=request.target_language,
                confidence_score=0.8,
                backend_used="fallback",
                processing_time=(datetime.now() - start_time).total_seconds(),
                session_id=request.session_id,
            )

        # Try backends in priority order
        for backend_name in self.backend_priority:
            try:
                result = await self._try_backend(backend_name, request)
                if result and result.confidence_score >= request.confidence_threshold:
                    result.processing_time = (datetime.now() - start_time).total_seconds()
                    result.session_id = request.session_id

                    # Update session if provided
                    if request.session_id:
                        await self._update_session(request.session_id, request, result)

                    return result

            except Exception as e:
                logger.warning(f"Backend {backend_name} failed: {e}")
                continue

        # If all backends fail, return error result
        raise Exception("All translation backends failed")

    async def translate_stream(self, request: TranslationRequest) -> AsyncGenerator[str, None]:
        """
        Stream translation results in real-time

        Args:
            request: Translation request with streaming enabled

        Yields:
            Partial translation results as they become available
        """
        if not request.streaming:
            # Non-streaming fallback
            result = await self.translate(request)
            yield result.translated_text
            return

        # Try streaming with local inference first
        if self.local_service:
            try:
                async for chunk in self.local_service.translate_stream(request):
                    yield chunk
                return
            except Exception as e:
                logger.warning(f"Local streaming failed: {e}")

        # Fallback to non-streaming
        result = await self.translate(request)
        yield result.translated_text

    async def translate_with_continuity(
        self,
        text: str,
        session_id: str,
        target_language: str = "en",
        source_language: str = "auto",
        chunk_id: str | None = None,
    ) -> dict:
        """
        Translate with context management and sentence buffering for streaming conversations.
        Receives clean text (already deduplicated by whisper service).

        Args:
            text: Clean transcribed text to translate (already deduplicated)
            session_id: Session identifier for context management
            target_language: Target language code (default: "en")
            source_language: Source language code (default: "auto")
            chunk_id: Optional chunk identifier for logging

        Returns:
            Dict with translation result or buffering status
        """
        # Get or create continuity manager for this session
        if session_id not in self.continuity_managers:
            self.continuity_managers[session_id] = TranslationContinuityManager()
            logger.info(f"Created new continuity manager for session {session_id}")

        continuity_mgr = self.continuity_managers[session_id]

        # Process clean text with sentence buffering (no deduplication needed)
        process_result = continuity_mgr.process_streaming_text(text, chunk_id)

        if process_result["status"] == "buffering":
            # Not ready for translation yet
            return {
                "status": "buffering",
                "buffer_length": process_result["buffer_length"],
                "chunk_id": chunk_id,
                "message": "Waiting for complete sentence...",
            }

        elif process_result["status"] == "ready_for_translation":
            # Ready to translate complete sentences
            text_to_translate = process_result["text"]
            context_prompt = process_result["context_prompt"]

            logger.info(f"Translating complete text: '{text_to_translate[:100]}...'")

            try:
                # Create translation request with context
                translation_request = TranslationRequest(
                    text=context_prompt,  # Use context-enhanced prompt
                    source_language=source_language,
                    target_language=target_language,
                    session_id=session_id,
                    context="\n".join(
                        [
                            item["translation"]
                            for item in list(continuity_mgr.translation_history)[-2:]
                        ]
                    ),
                )

                # Perform translation
                result = await self.translate(translation_request)

                # Extract just the new translation (remove context prompt response)
                # This is a simple extraction - could be improved with better prompt engineering
                translated_text = result.translated_text
                if "Current Chinese text to translate:" in translated_text:
                    parts = translated_text.split("Provide a natural English translation")
                    if len(parts) > 1:
                        translated_text = parts[-1].strip().strip(":").strip()

                # Store in context history
                continuity_mgr.store_translation_result(text_to_translate, translated_text)

                return {
                    "status": "translated",
                    "source_text": text_to_translate,
                    "translated_text": translated_text,
                    "confidence_score": result.confidence_score,
                    "processing_time": result.processing_time,
                    "backend_used": result.backend_used,
                    "chunk_id": chunk_id,
                    "context_items": len(continuity_mgr.translation_history),
                }

            except Exception as e:
                logger.error(f"Translation failed for session {session_id}: {e}")
                return {"status": "error", "error": str(e), "chunk_id": chunk_id}

    def clear_session_context(self, session_id: str):
        """Clear translation context for a specific session."""
        if session_id in self.continuity_managers:
            self.continuity_managers[session_id].clear_context()
            del self.continuity_managers[session_id]
            logger.info(f"Cleared context for session {session_id}")

    def get_session_context_info(self, session_id: str) -> dict:
        """Get context information for a session."""
        if session_id not in self.continuity_managers:
            return {"status": "no_context", "session_id": session_id}

        mgr = self.continuity_managers[session_id]
        return {
            "status": "active",
            "session_id": session_id,
            "context_items": len(mgr.translation_history),
            "buffer_length": len(mgr.sentence_buffer),
            "has_previous_text": bool(mgr.previous_source_text),
        }

    async def detect_language(self, text: str) -> tuple[str, float]:
        """
        Detect the language of input text

        Args:
            text: Text to analyze

        Returns:
            Tuple of (language_code, confidence_score)
        """
        # Try local inference first
        if self.local_service:
            try:
                return await self.local_service.detect_language(text)
            except Exception as e:
                logger.warning(f"Local language detection failed: {e}")

        # Additional language detection backends can be added here

        # Basic fallback - assume English
        return ("en", 0.5)

    async def get_supported_languages(self) -> list[dict[str, str]]:
        """Get list of supported languages across all backends"""
        languages = set()

        # Get from local service
        if self.local_service:
            try:
                local_langs = await self.local_service.get_supported_languages()
                languages.update((lang["code"], lang["name"]) for lang in local_langs)
            except Exception as e:
                logger.warning(f"Failed to get local supported languages: {e}")

        # Additional language providers can be added here

        # Convert to list of dicts
        return [{"code": code, "name": name} for code, name in languages]

    async def get_service_status(self) -> dict[str, dict]:
        """Get status of all translation backends"""
        status = {}

        # Local inference status
        if self.inference_client:
            try:
                status["local_inference"] = {
                    "available": True,
                    "backend": self.inference_client.get_backend_info(),
                    "health": await self.inference_client.health_check(),
                }
            except Exception as e:
                status["local_inference"] = {"available": False, "error": str(e)}

        # Additional backends can be added here in the future

        return status

    async def create_session(self, session_id: str, config: dict | None = None) -> dict:
        """Create a new translation session with specific configuration"""
        session_config = {
            "created_at": datetime.now().isoformat(),
            "config": config or {},
            "stats": {"translations": 0, "total_chars": 0, "avg_confidence": 0.0},
        }

        self.sessions[session_id] = session_config
        logger.info(f"Created translation session: {session_id}")
        return session_config

    async def get_session(self, session_id: str) -> dict | None:
        """Get session information"""
        return self.sessions.get(session_id)

    async def close_session(self, session_id: str) -> dict:
        """Close and return final session statistics"""
        session = self.sessions.pop(session_id, None)
        if session:
            session["closed_at"] = datetime.now().isoformat()
            logger.info(f"Closed translation session: {session_id}")
        return session

    async def _try_backend(
        self, backend_name: str, request: TranslationRequest
    ) -> TranslationResult | None:
        """Try a specific backend for translation"""
        if backend_name == "local_inference" and self.local_service:
            return await self.local_service.translate(request)

        # Additional backends can be added here as elif clauses

        elif backend_name == "fallback":
            # Basic fallback - return original text
            return TranslationResult(
                translated_text=request.text,
                source_language=request.source_language,
                target_language=request.target_language,
                confidence_score=0.1,
                processing_time=0.0,
                backend_used="fallback",
            )

        return None

    async def _update_session(
        self, session_id: str, request: TranslationRequest, result: TranslationResult
    ):
        """Update session statistics"""
        if session_id not in self.sessions:
            await self.create_session(session_id)

        session = self.sessions[session_id]
        stats = session["stats"]

        stats["translations"] += 1
        stats["total_chars"] += len(request.text)

        # Update average confidence
        old_avg = stats["avg_confidence"]
        count = stats["translations"]
        stats["avg_confidence"] = (old_avg * (count - 1) + result.confidence_score) / count

    def _load_config(self) -> dict:
        """Load configuration from environment and config files"""
        config = {
            # Local inference settings
            "use_local_inference": os.getenv("USE_LOCAL_INFERENCE", "true").lower() == "true",
            "use_legacy_apis": os.getenv("USE_LEGACY_APIS", "true").lower() == "true",
            # Model settings
            "translation_model": os.getenv("TRANSLATION_MODEL", "meta-llama/Llama-3.1-8B-Instruct"),
            "max_tokens": int(os.getenv("MAX_TOKENS", "1024")),
            "temperature": float(os.getenv("TEMPERATURE", "0.1")),
            # Service settings
            "timeout": int(os.getenv("TRANSLATION_TIMEOUT", "30")),
            "retry_attempts": int(os.getenv("RETRY_ATTEMPTS", "3")),
            "confidence_threshold": float(os.getenv("CONFIDENCE_THRESHOLD", "0.8")),
        }

        # Load from config file if exists
        config_file = os.path.join(os.path.dirname(__file__), "..", "config.json")
        if os.path.exists(config_file):
            try:
                with open(config_file) as f:
                    file_config = json.load(f)
                    config.update(file_config)
            except Exception as e:
                logger.warning(f"Failed to load config file: {e}")

        return config

    async def shutdown(self):
        """Shutdown the translation service and cleanup resources"""
        try:
            if self.local_service:
                await self.local_service.shutdown()

            # Additional backends shutdown can be added here

            if self.inference_client:
                await self.inference_client.close()

            logger.info("Translation service shutdown complete")

        except Exception as e:
            logger.error(f"Error during shutdown: {e}")


# Factory function for easy service creation
async def create_translation_service(config: dict | None = None) -> TranslationService:
    """
    Factory function to create and initialize a translation service

    Args:
        config: Optional configuration dict

    Returns:
        Initialized TranslationService instance
    """
    service = TranslationService(config)
    try:
        await service.initialize()
        logger.info("Translation service initialized with inference backends")
    except Exception as e:
        logger.warning(f"Failed to initialize inference backends: {e}")
        logger.info("Running in fallback mode - translations will return mock responses")
        # Set a flag to indicate fallback mode
        service.fallback_mode = True
        service.inference_client = None
    return service


# CLI interface for testing
if __name__ == "__main__":
    import argparse

    async def main():
        parser = argparse.ArgumentParser(description="Translation Service")
        parser.add_argument("--text", required=True, help="Text to translate")
        parser.add_argument("--source", default="auto", help="Source language")
        parser.add_argument("--target", default="en", help="Target language")
        parser.add_argument("--streaming", action="store_true", help="Use streaming")

        args = parser.parse_args()

        # Create service
        service = await create_translation_service()

        try:
            # Create request
            request = TranslationRequest(
                text=args.text,
                source_language=args.source,
                target_language=args.target,
                streaming=args.streaming,
            )

            if args.streaming:
                print("Streaming translation:")
                async for chunk in service.translate_stream(request):
                    print(chunk, end="", flush=True)
                print()
            else:
                result = await service.translate(request)
                print(f"Translation: {result.translated_text}")
                print(f"Confidence: {result.confidence_score:.2f}")
                print(f"Backend: {result.backend_used}")
                print(f"Time: {result.processing_time:.2f}s")

        finally:
            await service.shutdown()

    asyncio.run(main())
