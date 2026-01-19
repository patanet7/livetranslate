#!/usr/bin/env python3
"""
Audio Coordinator - Orchestration Service

Central coordination class for all audio processing in the LiveTranslate system.
Manages audio chunking, database integration, service communication, and speaker correlation.
Replaces scattered chunking logic across frontend and whisper service with centralized control.

Features:
- Centralized audio chunking with configurable overlap management
- Real-time coordination with whisper and translation services
- Complete database integration with bot_sessions schema
- Speaker correlation between whisper and Google Meet speakers
- WebSocket and chunk-based processing support
- Performance monitoring and quality assurance
- Concurrent session management with resource pooling
"""

import asyncio
import json
import logging
import os
import time
from collections import defaultdict
from collections.abc import AsyncGenerator, Callable
from contextlib import asynccontextmanager
from datetime import UTC, datetime
from typing import Any, Optional

import httpx
import numpy as np
from clients.audio_service_client import (
    AudioServiceClient,
    TranscriptionRequest as ClientTranscriptionRequest,
)
from clients.translation_service_client import (
    TranslationRequest as ClientTranslationRequest,
    TranslationServiceClient,
)

from .chunk_manager import ChunkManager, create_chunk_manager
from .database_adapter import AudioDatabaseAdapter
from .models import (
    AudioChunkingConfig,
    AudioChunkMetadata,
    AudioStreamingSession,
    SourceType,
    create_audio_chunk_metadata,
    get_default_chunking_config,
)

# Import data pipeline for modern database operations
try:
    from pipeline.data_pipeline import (
        AudioChunkMetadata as PipelineAudioChunkMetadata,
        TranscriptionDataPipeline,
        TranscriptionResult,
        TranslationResult,
    )

    _PIPELINE_AVAILABLE = True
except ImportError:
    TranscriptionDataPipeline = None
    TranscriptionResult = None
    TranslationResult = None
    PipelineAudioChunkMetadata = None
    _PIPELINE_AVAILABLE = False
from .audio_processor import AudioPipelineProcessor, create_audio_pipeline_processor
from .config import (
    AudioProcessingConfig,
    create_audio_config_manager,
)
from .translation_cache import TranslationResultCache

# Optional import for database optimization tracking
try:
    from database.translation_optimization_adapter import (
        TranslationOptimizationAdapter,
    )
except ImportError:
    TranslationOptimizationAdapter = None

logger = logging.getLogger(__name__)


class ServiceClientPool:
    """
    HTTP client pool for communicating with whisper and translation services.
    Provides connection pooling, retry logic, and circuit breaker patterns.
    Falls back to embedded service clients when available.
    """

    def __init__(
        self,
        service_urls: dict[str, str],
        pool_config: dict | None = None,
        audio_client: AudioServiceClient | None = None,
        translation_client: TranslationServiceClient | None = None,
    ):
        self.service_urls = service_urls
        pool_config = pool_config or {}
        self.audio_client = audio_client
        self.translation_client = translation_client

        # HTTP client configuration
        self.timeout = httpx.Timeout(
            connect=pool_config.get("connect_timeout", 10.0),
            read=pool_config.get("read_timeout", 30.0),
            write=pool_config.get("write_timeout", 10.0),
            pool=pool_config.get("pool_timeout", 5.0),
        )

        self.limits = httpx.Limits(
            max_keepalive_connections=pool_config.get("max_keepalive", 20),
            max_connections=pool_config.get("max_connections", 100),
            keepalive_expiry=pool_config.get("keepalive_expiry", 30.0),
        )

        # Circuit breaker state
        self.service_health = {
            url: {"healthy": True, "failures": 0, "last_failure": 0}
            for url in service_urls.values()
        }
        self.max_failures = pool_config.get("max_failures", 5)
        self.failure_timeout = pool_config.get("failure_timeout", 60.0)

        # HTTP client
        self.client: httpx.AsyncClient | None = None

    async def initialize(self) -> bool:
        """Initialize the HTTP client pool with robust error handling."""
        try:
            # First try with HTTP/2 enabled
            self.client = httpx.AsyncClient(
                timeout=self.timeout,
                limits=self.limits,
                http2=True,  # Enable HTTP/2 for better performance
                headers={"User-Agent": "LiveTranslate-AudioCoordinator/1.0"},
            )
            logger.info("Service client pool initialized with HTTP/2 support")
            return True
        except ImportError as e:
            if "h2" in str(e):
                logger.warning(
                    "HTTP/2 support not available (h2 package missing), falling back to HTTP/1.1"
                )
                try:
                    # Fallback to HTTP/1.1
                    self.client = httpx.AsyncClient(
                        timeout=self.timeout,
                        limits=self.limits,
                        http2=False,
                        headers={"User-Agent": "LiveTranslate-AudioCoordinator/1.0"},
                    )
                    logger.info("Service client pool initialized with HTTP/1.1")
                    return True
                except Exception as fallback_e:
                    logger.error(
                        f"Failed to initialize service client pool even with HTTP/1.1 fallback: {fallback_e}"
                    )
                    return False
            else:
                logger.error(f"Failed to initialize service client pool due to import error: {e}")
                return False
        except Exception as e:
            logger.error(f"Failed to initialize service client pool: {e}")
            # Try one more time without HTTP/2 as a last resort
            try:
                self.client = httpx.AsyncClient(
                    timeout=self.timeout,
                    limits=self.limits,
                    http2=False,
                    headers={"User-Agent": "LiveTranslate-AudioCoordinator/1.0"},
                )
                logger.warning(
                    "Service client pool initialized with HTTP/1.1 as fallback after general error"
                )
                return True
            except Exception as final_e:
                logger.error(f"Complete failure to initialize service client pool: {final_e}")
                return False

    async def close(self):
        """Close the HTTP client pool."""
        if self.client:
            await self.client.aclose()
            logger.info("Service client pool closed")

    def _is_remote_service(self, service_url: str | None) -> bool:
        return bool(service_url and service_url.startswith("http"))

    @asynccontextmanager
    async def get_client(self) -> AsyncGenerator[httpx.AsyncClient, None]:
        """Get HTTP client with circuit breaker protection."""
        if not self.client:
            raise RuntimeError("Service client pool not initialized")
        yield self.client

    async def send_to_whisper_service(
        self,
        session_id: str,
        chunk_metadata: AudioChunkMetadata,
        audio_data: np.ndarray,
        model: str = "whisper-tiny",
    ) -> dict[str, Any] | None:
        """Send audio chunk to whisper service for transcription."""
        if self.audio_client:
            return await self._transcribe_with_audio_client(
                session_id, chunk_metadata, audio_data, model
            )

        service_url = self.service_urls.get("whisper_service")
        if not self._is_remote_service(service_url) or not self._is_service_healthy(service_url):
            return None

        try:
            # Convert audio to bytes
            audio_bytes = self._audio_to_bytes(audio_data, chunk_metadata.sample_rate)

            # Prepare request data
            files = {"audio": ("chunk.wav", audio_bytes, "audio/wav")}
            data = {
                "session_id": session_id,
                "chunk_id": chunk_metadata.chunk_id,
                "chunk_sequence": chunk_metadata.chunk_sequence,
                "chunk_start_time": chunk_metadata.chunk_start_time,
                "chunk_end_time": chunk_metadata.chunk_end_time,
                "metadata": json.dumps(
                    {
                        "source_type": chunk_metadata.source_type.value,
                        "quality_score": chunk_metadata.audio_quality_score,
                        "overlap_metadata": chunk_metadata.overlap_metadata,
                    }
                ),
            }

            async with self.get_client() as client:
                response = await client.post(
                    f"{service_url}/transcribe/chunk", files=files, data=data
                )

                if response.status_code == 200:
                    result = response.json()
                    self._mark_service_healthy(service_url)
                    return result
                else:
                    logger.warning(f"Whisper service error: {response.status_code}")
                    self._mark_service_failure(service_url)
                    return None

        except Exception as e:
            logger.error(f"Failed to send chunk to whisper service: {e}")
            self._mark_service_failure(service_url)
            return None

    async def send_to_translation_service(
        self, session_id: str, transcript_data: dict[str, Any], target_language: str
    ) -> dict[str, Any] | None:
        """Send transcript to translation service."""
        if self.translation_client:
            return await self._translate_with_client(transcript_data, target_language)

        service_url = self.service_urls.get("translation_service")
        if not self._is_remote_service(service_url) or not self._is_service_healthy(service_url):
            return None

        try:
            translation_request = {
                "session_id": session_id,
                "text": transcript_data["text"],
                "source_language": transcript_data.get("language", "auto"),
                "target_language": target_language,
                "speaker_id": transcript_data.get("speaker_id"),
                "speaker_name": transcript_data.get("speaker_name"),
                "start_timestamp": transcript_data["start_timestamp"],
                "end_timestamp": transcript_data["end_timestamp"],
                "metadata": {
                    "transcript_id": transcript_data.get("transcript_id"),
                    "chunk_id": transcript_data.get("chunk_id"),
                    "confidence": transcript_data.get("confidence", 0.0),
                },
            }

            async with self.get_client() as client:
                response = await client.post(
                    f"{service_url}/api/translate", json=translation_request
                )

                if response.status_code == 200:
                    result = response.json()
                    self._mark_service_healthy(service_url)
                    return result
                else:
                    logger.warning(f"Translation service error: {response.status_code}")
                    self._mark_service_failure(service_url)
                    return None

        except Exception as e:
            logger.error(f"Failed to send to translation service: {e}")
            self._mark_service_failure(service_url)
            return None

    async def _translate_with_client(
        self,
        transcript_data: dict[str, Any],
        target_language: str,
    ) -> dict[str, Any] | None:
        if not self.translation_client:
            return None

        try:
            request = ClientTranslationRequest(
                text=transcript_data["text"],
                source_language=transcript_data.get("language", "auto"),
                target_language=target_language,
            )
            response = await self.translation_client.translate(request)
            metadata = response.model_dump(
                exclude={
                    "translated_text",
                    "target_language",
                    "confidence",
                    "processing_time",
                }
            )
            metadata.update(
                {
                    "processing_time": response.processing_time,
                    "backend": getattr(response, "backend_used", None),
                    "model": getattr(response, "model_used", None),
                }
            )
            return {
                "translated_text": response.translated_text,
                "confidence": response.confidence,
                "processing_time": response.processing_time,
                "service": getattr(response, "backend_used", "embedded"),
                "metadata": metadata,
            }
        except Exception as exc:
            logger.error(f"Embedded translation failed: {exc}")
            return None

    async def _transcribe_with_audio_client(
        self,
        session_id: str,
        chunk_metadata: AudioChunkMetadata,
        audio_data: np.ndarray,
        model: str = "whisper-tiny",
    ) -> dict[str, Any] | None:
        if not self.audio_client:
            return None

        try:
            audio_bytes = self._audio_to_bytes(audio_data, chunk_metadata.sample_rate)
            request = ClientTranscriptionRequest(
                language=None,
                task="transcribe",
                enable_diarization=True,
                enable_vad=True,
                model=model,
            )
            response = await self.audio_client.transcribe_stream(audio_bytes, request)
            if response is None:
                return None

            metadata = {
                "processing_time": response.processing_time,
                "session_id": session_id,
                "chunk_id": chunk_metadata.chunk_id,
            }
            metadata.update(
                response.model_dump(
                    exclude={"text", "language", "segments", "speakers", "confidence"}
                )
            )

            speaker_info: dict[str, Any] = {}
            if response.speakers:
                speaker_info["speakers"] = response.speakers

            return {
                "text": response.text,
                "language": response.language,
                "confidence": response.confidence,
                "segments": response.segments,
                "speaker_info": speaker_info,
                "metadata": metadata,
            }
        except Exception as exc:
            logger.error(f"Embedded audio transcription failed: {exc}")
            return None

    def _audio_to_bytes(self, audio_data: np.ndarray, sample_rate: int) -> bytes:
        """Convert audio data to WAV bytes."""
        import io

        import soundfile as sf

        # Convert to int16 for efficient transmission
        if audio_data.dtype != np.int16:
            normalized = np.clip(audio_data, -1.0, 1.0)
            audio_int16 = (normalized * 32767).astype(np.int16)
        else:
            audio_int16 = audio_data

        # Create WAV file in memory
        buffer = io.BytesIO()
        sf.write(buffer, audio_int16, sample_rate, format="WAV")
        buffer.seek(0)
        return buffer.read()

    def _is_service_healthy(self, service_url: str | None) -> bool:
        """Check if service is healthy based on circuit breaker state."""
        if not self._is_remote_service(service_url):
            return True
        health = self.service_health.get(
            service_url, {"healthy": True, "failures": 0, "last_failure": 0}
        )

        if health["healthy"]:
            return True

        # Check if enough time has passed to retry
        if time.time() - health["last_failure"] > self.failure_timeout:
            health["healthy"] = True
            health["failures"] = 0
            return True

        return False

    def _mark_service_healthy(self, service_url: str | None):
        """Mark service as healthy."""
        if not self._is_remote_service(service_url):
            return
        if service_url in self.service_health:
            self.service_health[service_url]["healthy"] = True
            self.service_health[service_url]["failures"] = 0

    def _mark_service_failure(self, service_url: str | None):
        """Mark service failure and update circuit breaker state."""
        if not self._is_remote_service(service_url):
            return
        if service_url not in self.service_health:
            return

        health = self.service_health[service_url]
        health["failures"] += 1
        health["last_failure"] = time.time()

        if health["failures"] >= self.max_failures:
            health["healthy"] = False
            logger.warning(f"Circuit breaker opened for {service_url}")


class SessionManager:
    """
    Manages multiple audio streaming sessions with resource pooling.
    Handles session lifecycle, resource allocation, and cleanup.
    """

    def __init__(self, max_concurrent_sessions: int = 10):
        self.max_concurrent_sessions = max_concurrent_sessions
        self.active_sessions: dict[str, AudioStreamingSession] = {}
        self.session_managers: dict[str, ChunkManager] = {}
        self.session_callbacks: dict[str, dict[str, Callable]] = defaultdict(dict)

        # Resource tracking
        self.total_sessions_created = 0
        self.total_chunks_processed = 0
        self.average_session_duration = 0.0

    def can_create_session(self) -> bool:
        """Check if new session can be created."""
        return len(self.active_sessions) < self.max_concurrent_sessions

    async def create_session(
        self,
        session_id: str,
        bot_session_id: str,
        source_type: SourceType,
        config: AudioChunkingConfig,
        target_languages: list[str],
        database_adapter: AudioDatabaseAdapter,
    ) -> AudioStreamingSession | None:
        """Create new audio streaming session."""
        if not self.can_create_session():
            logger.warning(f"Cannot create session {session_id}: max concurrent sessions reached")
            return None

        try:
            # Create streaming session
            streaming_session = AudioStreamingSession(
                streaming_session_id=session_id,
                bot_session_id=bot_session_id,
                source_type=source_type,
                chunk_config=config,
                target_languages=target_languages,
            )

            # Create chunk manager
            chunk_manager = create_chunk_manager(
                config, database_adapter, bot_session_id, source_type
            )

            # Store session
            self.active_sessions[session_id] = streaming_session
            self.session_managers[session_id] = chunk_manager
            self.total_sessions_created += 1

            logger.info(f"Created audio session {session_id} for bot session {bot_session_id}")
            return streaming_session

        except Exception as e:
            logger.error(f"Failed to create session {session_id}: {e}")
            return None

    async def start_session(self, session_id: str) -> bool:
        """Start audio processing for a session."""
        if session_id not in self.session_managers:
            return False

        try:
            chunk_manager = self.session_managers[session_id]
            success = await chunk_manager.start()

            if success:
                self.active_sessions[session_id].stream_status = "active"
                logger.info(f"Started audio session {session_id}")

            return success

        except Exception as e:
            logger.error(f"Failed to start session {session_id}: {e}")
            return False

    async def stop_session(self, session_id: str) -> dict[str, Any]:
        """Stop audio processing for a session and return statistics."""
        if session_id not in self.session_managers:
            return {}

        try:
            chunk_manager = self.session_managers[session_id]
            stats = await chunk_manager.stop()

            # Update session
            if session_id in self.active_sessions:
                session = self.active_sessions[session_id]
                session.stream_status = "ended"
                session.ended_at = datetime.now(UTC)
                session.chunks_processed = stats.get("chunks_processed", 0)
                session.total_duration = stats.get("total_audio_duration", 0.0)
                session.average_processing_time = stats.get("average_processing_time", 0.0)
                session.average_quality_score = stats.get("average_quality_score", 0.0)

                # Update global statistics
                self.total_chunks_processed += session.chunks_processed
                session_duration = (session.ended_at - session.started_at).total_seconds()
                self.average_session_duration = (
                    self.average_session_duration * (self.total_sessions_created - 1)
                    + session_duration
                ) / self.total_sessions_created

            logger.info(f"Stopped audio session {session_id}: {stats}")
            return stats

        except Exception as e:
            logger.error(f"Failed to stop session {session_id}: {e}")
            return {}
        finally:
            # Cleanup
            self.session_managers.pop(session_id, None)
            self.active_sessions.pop(session_id, None)
            self.session_callbacks.pop(session_id, None)

    def get_session(self, session_id: str) -> AudioStreamingSession | None:
        """Get session by ID."""
        return self.active_sessions.get(session_id)

    def get_chunk_manager(self, session_id: str) -> ChunkManager | None:
        """Get chunk manager for session."""
        return self.session_managers.get(session_id)

    def set_session_callbacks(self, session_id: str, callbacks: dict[str, Callable]):
        """Set callbacks for a session."""
        if session_id in self.session_managers:
            chunk_manager = self.session_managers[session_id]

            if "on_chunk_ready" in callbacks:
                chunk_manager.set_chunk_ready_callback(callbacks["on_chunk_ready"])
            if "on_quality_alert" in callbacks:
                chunk_manager.set_quality_alert_callback(callbacks["on_quality_alert"])
            if "on_error" in callbacks:
                chunk_manager.set_error_callback(callbacks["on_error"])

            self.session_callbacks[session_id] = callbacks

    def get_all_sessions(self) -> list[AudioStreamingSession]:
        """Get all active sessions."""
        return list(self.active_sessions.values())

    def get_session_statistics(self) -> dict[str, Any]:
        """Get overall session management statistics."""
        active_sessions = len(self.active_sessions)

        return {
            "active_sessions": active_sessions,
            "max_concurrent_sessions": self.max_concurrent_sessions,
            "utilization": active_sessions / self.max_concurrent_sessions,
            "total_sessions_created": self.total_sessions_created,
            "total_chunks_processed": self.total_chunks_processed,
            "average_session_duration": self.average_session_duration,
            "average_chunks_per_session": self.total_chunks_processed
            / max(1, self.total_sessions_created),
        }


class AudioCoordinator:
    """
    Central audio coordination class for the LiveTranslate system.
    Orchestrates all audio processing from reception through chunking, transcription, and translation.
    Includes complete audio processing pipeline with persistent configuration management.
    """

    def __init__(
        self,
        config: AudioChunkingConfig,
        database_url: str | None,
        service_urls: dict[str, str],
        max_concurrent_sessions: int = 10,
        audio_config_file: str | None = None,
        audio_client: AudioServiceClient | None = None,
        translation_client: TranslationServiceClient | None = None,
        data_pipeline: Optional["TranscriptionDataPipeline"] = None,
    ):
        self.config = config
        self.database_url = database_url
        self.service_urls = service_urls
        self.audio_client = audio_client
        self.translation_client = translation_client

        # Core components - Prefer data pipeline over legacy AudioDatabaseAdapter
        self.data_pipeline = data_pipeline
        if data_pipeline:
            logger.info(
                "AudioCoordinator using TranscriptionDataPipeline for database operations (production-ready)"
            )
            # Keep AudioDatabaseAdapter as None when using pipeline
            self.database_adapter = None
        elif database_url:
            logger.warning(
                "AudioCoordinator using legacy AudioDatabaseAdapter (deprecated - migrate to data_pipeline)"
            )
            self.database_adapter = AudioDatabaseAdapter(database_url)
        else:
            logger.info("AudioCoordinator running without database persistence")
            self.database_adapter = None

        self.service_client = ServiceClientPool(
            service_urls,
            audio_client=audio_client,
            translation_client=translation_client,
        )
        self.session_manager = SessionManager(max_concurrent_sessions)

        # Audio processing configuration and pipeline
        self.audio_config_manager = create_audio_config_manager(
            config_file_path=audio_config_file,
            database_adapter=self.database_adapter,
            auto_reload=True,
        )
        self.audio_processors: dict[str, AudioPipelineProcessor] = {}  # Per-session processors

        # Translation optimization components
        self.translation_opt_adapter = None
        self.translation_cache = None

        # Initialize translation optimization if database available
        if self.database_adapter and TranslationOptimizationAdapter:
            try:
                self.translation_opt_adapter = TranslationOptimizationAdapter(
                    self.database_adapter.db_manager
                )
                logger.info("Translation optimization adapter initialized")
            except Exception as e:
                logger.warning(f"Failed to initialize translation optimization adapter: {e}")

        # Initialize translation cache
        redis_url = os.getenv("REDIS_URL", "redis://localhost:6379/1")
        cache_enabled = os.getenv("TRANSLATION_CACHE_ENABLED", "true").lower() == "true"
        cache_ttl = int(os.getenv("TRANSLATION_CACHE_TTL", "3600"))  # 1 hour default

        if cache_enabled:
            try:
                self.translation_cache = TranslationResultCache(
                    redis_url=redis_url,
                    ttl=cache_ttl,
                    db_adapter=self.translation_opt_adapter,
                    session_id=None,  # Will be set per-session
                )
                logger.info(f"Translation cache enabled: TTL={cache_ttl}s, Redis={redis_url}")
            except Exception as e:
                logger.warning(f"Failed to initialize translation cache: {e}")
                self.translation_cache = None
        else:
            logger.info("Translation cache disabled by configuration")

        # Processing state
        self.is_running = False
        self.start_time: float | None = None

        # Performance tracking
        self.total_audio_processed = 0.0
        self.total_chunks_created = 0
        self.total_transcripts_generated = 0
        self.total_translations_generated = 0
        self.processing_errors = 0

        # Event callbacks
        self.on_transcript_ready: Callable | None = None
        self.on_translation_ready: Callable | None = None
        self.on_session_event: Callable | None = None
        self.on_error: Callable | None = None

        logger.info("AudioCoordinator initialized")

    # ==================== Format Conversion Helper Methods ====================

    def _create_transcription_result(
        self, transcript_data: dict[str, Any]
    ) -> "TranscriptionResult":
        """
        Convert transcript dictionary to TranscriptionResult dataclass for pipeline.

        Args:
            transcript_data: Transcript dictionary with text, timestamps, language, etc.

        Returns:
            TranscriptionResult instance for pipeline processing
        """
        if not TranscriptionResult:
            raise ImportError("TranscriptionResult not available - pipeline not imported")

        speaker_info = transcript_data.get("speaker_info", {})

        return TranscriptionResult(
            text=transcript_data.get("text", ""),
            language=transcript_data.get("language", "en"),
            start_time=transcript_data.get("start_timestamp", 0.0),
            end_time=transcript_data.get("end_timestamp", 0.0),
            speaker=speaker_info.get("speaker_id") if speaker_info else None,
            speaker_name=speaker_info.get("speaker_name") if speaker_info else None,
            confidence=transcript_data.get("confidence", 0.0),
            segment_index=transcript_data.get("segment_index", 0),
            is_final=transcript_data.get("is_final", True),
            words=transcript_data.get("words"),
        )

    def _create_translation_result(self, translation_data: dict[str, Any]) -> "TranslationResult":
        """
        Convert translation dictionary to TranslationResult dataclass for pipeline.

        Args:
            translation_data: Translation dictionary with translated_text, languages, etc.

        Returns:
            TranslationResult instance for pipeline processing
        """
        if not TranslationResult:
            raise ImportError("TranslationResult not available - pipeline not imported")

        return TranslationResult(
            text=translation_data.get("translated_text", ""),
            source_language=translation_data.get("source_language", "auto"),
            target_language=translation_data.get("target_language", "en"),
            speaker=translation_data.get("speaker_id"),
            speaker_name=translation_data.get("speaker_name"),
            confidence=translation_data.get("confidence", 0.0),
            translation_service=translation_data.get("translation_service", "translation_service"),
            model_name=translation_data.get("model"),
        )

    async def _store_transcript_via_pipeline(
        self,
        session_id: str,
        transcript_data: dict[str, Any],
        audio_file_id: str | None = None,
    ) -> str | None:
        """
        Store transcript using data pipeline with proper format conversion.

        Args:
            session_id: Session identifier
            transcript_data: Transcript dictionary from whisper service
            audio_file_id: Optional audio file ID for linking

        Returns:
            Transcript ID if successful, None otherwise
        """
        try:
            transcription_result = self._create_transcription_result(transcript_data)
            return await self.data_pipeline.process_transcription_result(
                session_id=session_id,
                file_id=audio_file_id,
                transcription=transcription_result,
                source_type=transcript_data.get("source_type", "whisper_service"),
            )
        except Exception as e:
            logger.error(f"Failed to store transcript via pipeline: {e}", exc_info=True)
            return None

    async def _store_translation_via_pipeline(
        self,
        session_id: str,
        transcript_id: str,
        translation_data: dict[str, Any],
        start_timestamp: float,
        end_timestamp: float,
    ) -> str | None:
        """
        Store translation using data pipeline with proper format conversion.

        Args:
            session_id: Session identifier
            transcript_id: Source transcript ID
            translation_data: Translation dictionary from translation service
            start_timestamp: Start time of translation
            end_timestamp: End time of translation

        Returns:
            Translation ID if successful, None otherwise
        """
        try:
            translation_result = self._create_translation_result(translation_data)
            return await self.data_pipeline.process_translation_result(
                session_id=session_id,
                transcript_id=transcript_id,
                translation=translation_result,
                start_time=start_timestamp,
                end_time=end_timestamp,
            )
        except Exception as e:
            logger.error(f"Failed to store translation via pipeline: {e}", exc_info=True)
            return None

    # ==================== Initialization and Lifecycle ====================

    async def initialize(self) -> bool:
        """Initialize the audio coordinator and all components."""
        try:
            # Initialize database adapter (optional for development)
            if self.database_adapter:
                success = await self.database_adapter.initialize()
                if not success:
                    logger.error("Failed to initialize database adapter")
                    return False
            else:
                logger.warning("Database adapter disabled - running without persistence")

            # Initialize service client pool
            success = await self.service_client.initialize()
            if not success:
                logger.error("Failed to initialize service client pool")
                return False

            # Initialize audio configuration manager
            success = await self.audio_config_manager.initialize()
            if not success:
                logger.error("Failed to initialize audio configuration manager")
                return False

            # Set up configuration change callback
            self.audio_config_manager.add_config_change_callback(self._on_audio_config_change)

            self.is_running = True
            self.start_time = time.time()

            logger.info("AudioCoordinator initialized successfully")
            return True

        except Exception as e:
            logger.error(f"Failed to initialize AudioCoordinator: {e}")
            return False

    async def shutdown(self) -> dict[str, Any]:
        """Shutdown the audio coordinator and return final statistics."""
        try:
            self.is_running = False

            # Stop all active sessions
            session_stats = []
            for session_id in list(self.session_manager.active_sessions.keys()):
                stats = await self.session_manager.stop_session(session_id)
                session_stats.append(stats)

            # Close components
            await self.service_client.close()

            # Close translation cache
            if self.translation_cache:
                try:
                    cache_stats = self.translation_cache.get_stats()
                    logger.info(f"Translation cache final stats: {cache_stats}")
                    await self.translation_cache.close()
                except Exception as e:
                    logger.error(f"Error closing translation cache: {e}")

            if self.database_adapter:
                await self.database_adapter.close()
            await self.audio_config_manager.shutdown()

            # Clear audio processors
            self.audio_processors.clear()

            # Calculate final statistics
            total_runtime = time.time() - self.start_time if self.start_time else 0

            final_stats = {
                "total_runtime": total_runtime,
                "total_audio_processed": self.total_audio_processed,
                "total_chunks_created": self.total_chunks_created,
                "total_transcripts_generated": self.total_transcripts_generated,
                "total_translations_generated": self.total_translations_generated,
                "processing_errors": self.processing_errors,
                "session_statistics": self.session_manager.get_session_statistics(),
                "database_statistics": self.database_adapter.get_operation_statistics()
                if self.database_adapter
                else {},
                "session_final_stats": session_stats,
            }

            logger.info(f"AudioCoordinator shutdown complete: {final_stats}")
            return final_stats

        except Exception as e:
            logger.error(f"Error during AudioCoordinator shutdown: {e}")
            return {}

    async def cleanup(self):
        """Cleanup method (alias for shutdown for test compatibility)."""
        return await self.shutdown()

    # Event callback setters
    def set_transcript_ready_callback(self, callback: Callable[[dict[str, Any]], None]):
        """Set callback for when transcripts are ready."""
        self.on_transcript_ready = callback

    def set_translation_ready_callback(self, callback: Callable[[dict[str, Any]], None]):
        """Set callback for when translations are ready."""
        self.on_translation_ready = callback

    def set_session_event_callback(self, callback: Callable[[str, dict[str, Any]], None]):
        """Set callback for session events."""
        self.on_session_event = callback

    def set_error_callback(self, callback: Callable[[str], None]):
        """Set callback for error notifications."""
        self.on_error = callback

    async def process_audio_chunk(
        self,
        audio_chunk: bytes,
        pipeline_config: dict[str, Any] | None = None,
        session_id: str | None = None,
    ) -> dict[str, Any]:
        """
        Process a single audio chunk through the pipeline.

        Args:
            audio_chunk: Raw audio bytes
            pipeline_config: Optional pipeline configuration
            session_id: Optional session ID for tracking

        Returns:
            Dictionary with processed_audio and metrics
        """
        try:
            # For now, return the audio as-is with basic metrics
            # TODO: Implement actual pipeline processing when stages are defined
            import numpy as np

            # Convert bytes to numpy array for basic analysis
            audio_array = np.frombuffer(audio_chunk, dtype=np.int16)

            # Calculate basic metrics
            rms = np.sqrt(np.mean(np.square(audio_array.astype(float))))
            rms_db = 20 * np.log10(rms / 32768.0) if rms > 0 else -96.0

            return {
                "processed_audio": audio_chunk,  # Pass through for now
                "rms": rms_db,
                "snr": 0.0,  # Placeholder
                "cpu_usage": 0.0,  # Placeholder
                "stage_latencies": {},
            }

        except Exception as e:
            logger.error(f"Failed to process audio chunk: {e}")
            raise

    # Audio processing configuration management

    async def _on_audio_config_change(
        self,
        change_type: str,
        config: AudioProcessingConfig,
        session_id: str | None = None,
    ):
        """Handle audio configuration changes."""
        try:
            if session_id:
                # Update session-specific processor
                if session_id in self.audio_processors:
                    self.audio_processors[session_id].update_config(config)
                    logger.info(f"Updated audio processor config for session {session_id}")
            else:
                # Update all processors with new default config
                for processor in self.audio_processors.values():
                    processor.update_config(config)
                logger.info("Updated all audio processors with new default config")

            # Emit configuration change event
            if self.on_session_event:
                self.on_session_event(
                    "audio_config_changed",
                    {
                        "change_type": change_type,
                        "session_id": session_id,
                        "preset_name": config.preset_name,
                    },
                )

        except Exception as e:
            logger.error(f"Failed to handle audio config change: {e}")

    def get_audio_processing_config(self, session_id: str | None = None) -> AudioProcessingConfig:
        """Get audio processing configuration for session or default."""
        if session_id:
            return self.audio_config_manager.get_session_config(session_id)
        else:
            return self.audio_config_manager.get_default_config()

    async def update_audio_processing_config(
        self,
        config_updates: dict[str, Any],
        session_id: str | None = None,
        save_persistent: bool = True,
    ) -> AudioProcessingConfig:
        """Update audio processing configuration."""
        return await self.audio_config_manager.update_config_from_dict(
            config_updates, session_id, save_persistent
        )

    async def apply_audio_preset(self, preset_name: str, session_id: str | None = None) -> bool:
        """Apply an audio processing preset."""
        return await self.audio_config_manager.apply_preset(preset_name, session_id)

    def get_available_audio_presets(self) -> list[str]:
        """Get list of available audio processing presets."""
        return self.audio_config_manager.get_available_presets()

    def get_audio_config_schema(self) -> dict[str, Any]:
        """Get audio configuration schema for frontend validation."""
        return self.audio_config_manager.get_config_schema()

    def _get_or_create_audio_processor(self, session_id: str) -> AudioPipelineProcessor:
        """Get or create audio processor for session."""
        if session_id not in self.audio_processors:
            # Get session-specific config or use default
            audio_config = self.audio_config_manager.get_session_config(session_id)
            self.audio_processors[session_id] = create_audio_pipeline_processor(audio_config)
            logger.info(
                f"Created audio processor for session {session_id} with preset: {audio_config.preset_name}"
            )

        return self.audio_processors[session_id]

    # Session management API
    async def create_audio_session(
        self,
        bot_session_id: str,
        source_type: SourceType = SourceType.BOT_AUDIO,
        target_languages: list[str] | None = None,
        custom_config: dict[str, Any] | None = None,
    ) -> str | None:
        """
        Create new audio streaming session.

        Returns:
            str: Streaming session ID if successful, None if failed
        """
        if not self.is_running:
            logger.error("AudioCoordinator not running")
            return None

        try:
            # Generate session ID
            session_id = f"audio_{bot_session_id}_{int(time.time() * 1000)}"

            # Prepare configuration
            config = self.config.copy()
            if custom_config:
                for key, value in custom_config.items():
                    if hasattr(config, key):
                        setattr(config, key, value)

            # Use default target languages if not provided
            if not target_languages:
                target_languages = ["en", "es", "fr", "de"]

            # Create session
            streaming_session = await self.session_manager.create_session(
                session_id,
                bot_session_id,
                source_type,
                config,
                target_languages,
                self.database_adapter,
            )

            if streaming_session:
                # Set up session callbacks
                callbacks = {
                    "on_chunk_ready": lambda metadata, audio_data: asyncio.create_task(
                        self._handle_chunk_ready(session_id, metadata, audio_data)
                    ),
                    "on_quality_alert": lambda alert: self._handle_quality_alert(session_id, alert),
                    "on_error": lambda error: self._handle_session_error(session_id, error),
                }

                self.session_manager.set_session_callbacks(session_id, callbacks)

                # Emit session event
                if self.on_session_event:
                    self.on_session_event(
                        "audio_session_created",
                        {
                            "session_id": session_id,
                            "bot_session_id": bot_session_id,
                            "source_type": source_type.value,
                            "target_languages": target_languages,
                            "config": config.dict(),
                        },
                    )

                logger.info(f"Created audio session {session_id}")
                return session_id

            return None

        except Exception as e:
            logger.error(f"Failed to create audio session: {e}")
            if self.on_error:
                self.on_error(f"Session creation failed: {e}")
            return None

    async def start_audio_session(self, session_id: str) -> bool:
        """Start audio processing for a session."""
        try:
            success = await self.session_manager.start_session(session_id)

            if success and self.on_session_event:
                self.on_session_event("audio_session_started", {"session_id": session_id})

            return success

        except Exception as e:
            logger.error(f"Failed to start audio session {session_id}: {e}")
            return False

    async def stop_audio_session(self, session_id: str) -> dict[str, Any]:
        """Stop audio processing for a session."""
        try:
            stats = await self.session_manager.stop_session(session_id)

            if self.on_session_event:
                self.on_session_event(
                    "audio_session_stopped",
                    {"session_id": session_id, "final_stats": stats},
                )

            return stats

        except Exception as e:
            logger.error(f"Failed to stop audio session {session_id}: {e}")
            return {}

    async def add_audio_data(self, session_id: str, audio_data: np.ndarray) -> bool:
        """
        Add audio data to a session for processing.
        Includes complete audio processing pipeline before chunking.

        Args:
            session_id: Audio streaming session ID
            audio_data: Audio samples as numpy array

        Returns:
            bool: True if data was successfully added
        """
        try:
            chunk_manager = self.session_manager.get_chunk_manager(session_id)
            if not chunk_manager:
                logger.warning(f"No chunk manager found for session {session_id}")
                return False

            # Apply audio processing pipeline
            audio_processor = self._get_or_create_audio_processor(session_id)
            processed_audio, processing_metadata = audio_processor.process_audio_chunk(audio_data)

            # Log processing results if significant
            if processing_metadata.get("bypassed", False):
                logger.debug(
                    f"Audio processing bypassed for session {session_id}: {processing_metadata.get('bypass_reason', 'unknown')}"
                )
            elif processing_metadata.get("stages_applied"):
                logger.debug(
                    f"Applied audio processing stages for session {session_id}: {processing_metadata['stages_applied']}"
                )

            # Add processed audio to chunk manager
            samples_added = await chunk_manager.add_audio_data(processed_audio)

            if samples_added > 0:
                # Update session activity
                session = self.session_manager.get_session(session_id)
                if session:
                    session.last_activity_at = datetime.now(UTC)

                # Update global statistics
                self.total_audio_processed += samples_added / 16000  # Convert to seconds

                # Store processing metadata for analytics
                if self.on_session_event:
                    self.on_session_event(
                        "audio_processed",
                        {
                            "session_id": session_id,
                            "samples_processed": len(audio_data),
                            "processing_metadata": processing_metadata,
                        },
                    )

            return samples_added > 0

        except Exception as e:
            logger.error(f"Failed to add audio data to session {session_id}: {e}")
            self.processing_errors += 1
            return False

    async def _handle_chunk_ready(
        self, session_id: str, metadata: AudioChunkMetadata, audio_data: np.ndarray
    ):
        """Handle when audio chunk is ready for processing."""
        try:
            self.total_chunks_created += 1

            # Send to whisper service
            transcript_result = await self.service_client.send_to_whisper_service(
                session_id, metadata, audio_data
            )

            if transcript_result and transcript_result.get("text"):
                # Store transcript in database (if available)
                transcript_id = None
                if self.data_pipeline:
                    # Use production-ready pipeline with all fixes (NULL safety, caching, transactions, rate limiting)
                    transcript_id = await self._store_transcript_via_pipeline(
                        metadata.session_id,
                        {
                            "text": transcript_result["text"],
                            "start_timestamp": metadata.chunk_start_time,
                            "end_timestamp": metadata.chunk_end_time,
                            "language": transcript_result.get("language", "en"),
                            "confidence": transcript_result.get("confidence", 0.0),
                            "speaker_info": transcript_result.get("speaker_info", {}),
                            "chunk_id": metadata.chunk_id,
                            "source_type": "whisper_service",
                            "metadata": transcript_result.get("metadata", {}),
                        },
                        audio_file_id=metadata.chunk_id,
                    )
                elif self.database_adapter:
                    # Fallback to legacy adapter (deprecated)
                    transcript_id = await self.database_adapter.store_transcript(
                        metadata.session_id,
                        {
                            "text": transcript_result["text"],
                            "start_timestamp": metadata.chunk_start_time,
                            "end_timestamp": metadata.chunk_end_time,
                            "language": transcript_result.get("language", "en"),
                            "confidence": transcript_result.get("confidence", 0.0),
                            "speaker_info": transcript_result.get("speaker_info", {}),
                            "chunk_id": metadata.chunk_id,
                            "source_type": "whisper_service",
                            "metadata": transcript_result.get("metadata", {}),
                        },
                        audio_file_id=metadata.chunk_id,
                    )
                else:
                    # Generate fake ID for non-persistent mode
                    transcript_id = f"transcript_{metadata.chunk_id}"

                if transcript_id:
                    self.total_transcripts_generated += 1

                    # Emit transcript ready event
                    if self.on_transcript_ready:
                        self.on_transcript_ready(
                            {
                                "session_id": session_id,
                                "transcript_id": transcript_id,
                                "chunk_id": metadata.chunk_id,
                                "text": transcript_result["text"],
                                "start_timestamp": metadata.chunk_start_time,
                                "end_timestamp": metadata.chunk_end_time,
                                "language": transcript_result.get("language", "en"),
                                "speaker_info": transcript_result.get("speaker_info", {}),
                                "confidence": transcript_result.get("confidence", 0.0),
                            }
                        )

                    # Request translations
                    session = self.session_manager.get_session(session_id)
                    if session and session.target_languages:
                        await self._request_translations(
                            session_id,
                            transcript_id,
                            transcript_result,
                            session.target_languages,
                        )
                else:
                    logger.warning(f"Failed to store transcript for chunk {metadata.chunk_id}")
            else:
                logger.debug(f"No text in transcript result for chunk {metadata.chunk_id}")

        except Exception as e:
            logger.error(f"Failed to handle chunk ready for {metadata.chunk_id}: {e}")
            self.processing_errors += 1

    async def _request_translations(
        self,
        session_id: str,
        transcript_id: str,
        transcript_result: dict[str, Any],
        target_languages: list[str],
    ):
        """
        OPTIMIZED: Request translations with caching and multi-language batching.

        Flow:
        1. Check cache for all target languages
        2. Translate only uncached languages using multi-language endpoint
        3. Store new translations in cache
        4. Record batch metadata in database
        5. Store translations and emit events
        """
        source_language = transcript_result.get("language", "auto")
        text = transcript_result["text"]

        # Filter out source language
        target_langs = [lang for lang in target_languages if lang != source_language]

        if not target_langs:
            logger.debug(f"No target languages to translate to (source={source_language})")
            return

        logger.info(
            f"Requesting translations: {len(target_langs)} languages "
            f"for '{text[:50]}...' ({source_language}→{target_langs})"
        )

        # Start timing for performance tracking
        start_time = time.time()

        # STEP 1: Check cache for all target languages (if cache enabled)
        cached_results = {}
        needs_translation = []

        if self.translation_cache:
            try:
                # Set session ID for database tracking
                self.translation_cache.session_id = session_id

                # Multi-get from cache
                cache_results = await self.translation_cache.get_multi(
                    text=text, source_lang=source_language, target_langs=target_langs
                )

                # Separate cached vs needs translation
                for lang in target_langs:
                    if cache_results.get(lang):
                        cached_results[lang] = cache_results[lang]
                        logger.debug(f"Cache HIT: {source_language}→{lang}")
                    else:
                        needs_translation.append(lang)
                        logger.debug(f"Cache MISS: {source_language}→{lang}")

            except Exception as cache_error:
                logger.error(f"Cache lookup failed: {cache_error}, proceeding without cache")
                needs_translation = target_langs
        else:
            # No cache, translate all
            needs_translation = target_langs

        # STEP 2: Translate only what's not cached (using multi-language endpoint)
        new_translations = {}

        if needs_translation:
            logger.info(
                f"Translating {len(needs_translation)} uncached languages: {needs_translation}"
            )

            try:
                # Use optimized multi-language endpoint via translation client
                if self.translation_client:
                    translation_results = (
                        await self.translation_client.translate_to_multiple_languages(
                            text=text,
                            source_language=source_language,
                            target_languages=needs_translation,
                            session_id=session_id,
                        )
                    )

                    # Convert TranslationResponse objects to dict format
                    for lang, response in translation_results.items():
                        if response and hasattr(response, "translated_text"):
                            new_translations[lang] = {
                                "translated_text": response.translated_text,
                                "confidence": response.confidence,
                                "metadata": {
                                    "backend_used": response.backend_used,
                                    "model_used": response.model_used,
                                    "processing_time": response.processing_time,
                                },
                            }
                else:
                    # Fallback to service client pool (sequential)
                    logger.warning("Translation client not available, using service pool")
                    for lang in needs_translation:
                        translation_result = await self.service_client.send_to_translation_service(
                            session_id, transcript_result, lang
                        )
                        if translation_result:
                            new_translations[lang] = translation_result

                # STEP 3: Store new translations in cache
                if self.translation_cache and new_translations:
                    try:
                        await self.translation_cache.set_multi(
                            text=text,
                            source_lang=source_language,
                            translations=new_translations,
                        )
                        logger.debug(f"Cached {len(new_translations)} new translations")
                    except Exception as cache_error:
                        logger.error(f"Failed to cache translations: {cache_error}")

            except Exception as translation_error:
                logger.error(f"Translation failed: {translation_error}")
                self.processing_errors += 1
                return

        # STEP 4: Combine cached + new translations
        all_translations = {**cached_results, **new_translations}

        # Calculate performance metrics
        total_time_ms = (time.time() - start_time) * 1000
        cache_hits = len(cached_results)
        cache_misses = len(new_translations)
        cache_hit_rate = cache_hits / len(target_langs) if target_langs else 0

        logger.info(
            f"Translation complete: {len(all_translations)} languages in {total_time_ms:.2f}ms "
            f"(cache: {cache_hits} hits, {cache_misses} misses, {cache_hit_rate:.1%} hit rate)"
        )

        # STEP 5: Record batch metadata in database
        if self.translation_opt_adapter:
            try:
                # Generate batch ID
                batch_id = f"batch_{session_id}_{transcript_id}"

                # Record batch performance
                await self.translation_opt_adapter.record_translation_batch(
                    session_id=session_id,
                    source_text=text,
                    source_language=source_language,
                    target_languages=target_langs,
                    total_time_ms=total_time_ms,
                    cache_hits=cache_hits,
                    cache_misses=cache_misses,
                    success_count=len(all_translations),
                    error_count=len(target_langs) - len(all_translations),
                    results=all_translations,
                    batch_id=batch_id,
                )
            except Exception as db_error:
                logger.error(f"Failed to record batch in database: {db_error}")

        # STEP 6: Store each translation in database and emit events
        for target_lang, translation_data in all_translations.items():
            try:
                await self._store_and_emit_translation(
                    session_id=session_id,
                    transcript_id=transcript_id,
                    transcript_result=transcript_result,
                    target_language=target_lang,
                    translation_data=translation_data,
                    was_cached=(target_lang in cached_results),
                )
            except Exception as store_error:
                logger.error(f"Failed to store translation {target_lang}: {store_error}")

    async def _process_single_translation(
        self,
        session_id: str,
        transcript_id: str,
        transcript_result: dict[str, Any],
        target_language: str,
    ):
        """Process a single translation request."""
        try:
            translation_result = await self.service_client.send_to_translation_service(
                session_id, transcript_result, target_language
            )

            if translation_result and translation_result.get("translated_text"):
                # Store translation in database (if available)
                translation_id = None
                if self.data_pipeline:
                    # Use production-ready pipeline with all fixes
                    translation_id = await self._store_translation_via_pipeline(
                        session_id,
                        transcript_id,
                        {
                            "translated_text": translation_result["translated_text"],
                            "source_language": transcript_result.get("language", "auto"),
                            "target_language": target_language,
                            "confidence": translation_result.get("confidence", 0.0),
                            "translation_service": translation_result.get("service", "unknown"),
                            "speaker_id": transcript_result.get("speaker_info", {}).get(
                                "speaker_id"
                            ),
                            "speaker_name": transcript_result.get("speaker_info", {}).get(
                                "speaker_name"
                            ),
                            "metadata": translation_result.get("metadata", {}),
                        },
                        start_timestamp=transcript_result.get("start_timestamp", 0.0),
                        end_timestamp=transcript_result.get("end_timestamp", 0.0),
                    )
                elif self.database_adapter:
                    # Fallback to legacy adapter (deprecated)
                    translation_id = await self.database_adapter.store_translation(
                        session_id,
                        transcript_id,
                        {
                            "translated_text": translation_result["translated_text"],
                            "source_language": transcript_result.get("language", "auto"),
                            "target_language": target_language,
                            "confidence": translation_result.get("confidence", 0.0),
                            "translation_service": translation_result.get("service", "unknown"),
                            "speaker_id": transcript_result.get("speaker_info", {}).get(
                                "speaker_id"
                            ),
                            "speaker_name": transcript_result.get("speaker_info", {}).get(
                                "speaker_name"
                            ),
                            "start_timestamp": transcript_result.get("start_timestamp", 0.0),
                            "end_timestamp": transcript_result.get("end_timestamp", 0.0),
                            "metadata": translation_result.get("metadata", {}),
                        },
                    )
                else:
                    # Generate fake ID for non-persistent mode
                    translation_id = f"translation_{transcript_id}_{target_language}"

                if translation_id:
                    self.total_translations_generated += 1

                    # Emit translation ready event
                    if self.on_translation_ready:
                        self.on_translation_ready(
                            {
                                "session_id": session_id,
                                "translation_id": translation_id,
                                "transcript_id": transcript_id,
                                "original_text": transcript_result["text"],
                                "translated_text": translation_result["translated_text"],
                                "source_language": transcript_result.get("language", "auto"),
                                "target_language": target_language,
                                "speaker_info": transcript_result.get("speaker_info", {}),
                                "confidence": translation_result.get("confidence", 0.0),
                                "start_timestamp": transcript_result.get("start_timestamp", 0.0),
                                "end_timestamp": transcript_result.get("end_timestamp", 0.0),
                            }
                        )
                else:
                    logger.warning(f"Failed to store translation for transcript {transcript_id}")

        except Exception as e:
            logger.error(
                f"Failed to process translation {target_language} for transcript {transcript_id}: {e}"
            )
            self.processing_errors += 1

    async def _store_and_emit_translation(
        self,
        session_id: str,
        transcript_id: str,
        transcript_result: dict[str, Any],
        target_language: str,
        translation_data: dict[str, Any],
        was_cached: bool,
    ):
        """
        Store translation in database and emit event.

        Args:
            session_id: Bot session ID
            transcript_id: Transcript ID
            transcript_result: Original transcript data
            target_language: Target language code
            translation_data: Translation result
            was_cached: Whether result came from cache
        """
        # Store translation in database (if available)
        translation_id = None

        if self.data_pipeline:
            # Use production-ready pipeline with all fixes
            translation_id = await self._store_translation_via_pipeline(
                session_id,
                transcript_id,
                {
                    "translated_text": translation_data.get("translated_text", ""),
                    "source_language": transcript_result.get("language", "auto"),
                    "target_language": target_language,
                    "confidence": translation_data.get("confidence", 0.0),
                    "translation_service": translation_data.get("metadata", {}).get(
                        "backend_used", "unknown"
                    ),
                    "speaker_id": transcript_result.get("speaker_info", {}).get("speaker_id"),
                    "speaker_name": transcript_result.get("speaker_info", {}).get("speaker_name"),
                    "metadata": translation_data.get("metadata", {}),
                },
                start_timestamp=transcript_result.get("start_timestamp", 0.0),
                end_timestamp=transcript_result.get("end_timestamp", 0.0),
            )
        elif self.database_adapter:
            # Fallback to legacy adapter (deprecated)
            translation_id = await self.database_adapter.store_translation(
                session_id,
                transcript_id,
                {
                    "translated_text": translation_data.get("translated_text", ""),
                    "source_language": transcript_result.get("language", "auto"),
                    "target_language": target_language,
                    "confidence": translation_data.get("confidence", 0.0),
                    "translation_service": translation_data.get("metadata", {}).get(
                        "backend_used", "unknown"
                    ),
                    "speaker_id": transcript_result.get("speaker_info", {}).get("speaker_id"),
                    "speaker_name": transcript_result.get("speaker_info", {}).get("speaker_name"),
                    "start_timestamp": transcript_result.get("start_timestamp", 0.0),
                    "end_timestamp": transcript_result.get("end_timestamp", 0.0),
                    "metadata": translation_data.get("metadata", {}),
                },
            )

            # Update translation with optimization metadata
            if translation_id and self.translation_opt_adapter:
                try:
                    await self.translation_opt_adapter.update_translation_optimization_metadata(
                        translation_id=translation_id,
                        model_name=translation_data.get("metadata", {}).get("model_used"),
                        model_backend=translation_data.get("metadata", {}).get("backend_used"),
                        was_cached=was_cached,
                        optimization_metadata={
                            "was_cached": was_cached,
                            "processing_time_ms": translation_data.get("metadata", {}).get(
                                "processing_time"
                            ),
                        },
                    )
                except Exception as opt_error:
                    logger.error(f"Failed to update optimization metadata: {opt_error}")
        else:
            # Generate fake ID for non-persistent mode
            translation_id = f"translation_{transcript_id}_{target_language}"

        if translation_id:
            self.total_translations_generated += 1

            # Emit translation ready event
            if self.on_translation_ready:
                self.on_translation_ready(
                    {
                        "session_id": session_id,
                        "translation_id": translation_id,
                        "transcript_id": transcript_id,
                        "original_text": transcript_result["text"],
                        "translated_text": translation_data.get("translated_text", ""),
                        "source_language": transcript_result.get("language", "auto"),
                        "target_language": target_language,
                        "speaker_info": transcript_result.get("speaker_info", {}),
                        "confidence": translation_data.get("confidence", 0.0),
                        "start_timestamp": transcript_result.get("start_timestamp", 0.0),
                        "end_timestamp": transcript_result.get("end_timestamp", 0.0),
                        "was_cached": was_cached,
                    }
                )
        else:
            logger.warning(f"Failed to store translation for transcript {transcript_id}")

    def _handle_quality_alert(self, session_id: str, alert: dict[str, Any]):
        """Handle quality alerts from chunk processing."""
        logger.info(f"Quality alert for session {session_id}: {alert}")

        if self.on_session_event:
            self.on_session_event("quality_alert", {"session_id": session_id, "alert": alert})

    def _handle_session_error(self, session_id: str, error: str):
        """Handle errors from session processing."""
        logger.error(f"Session error for {session_id}: {error}")
        self.processing_errors += 1

        if self.on_error:
            self.on_error(f"Session {session_id} error: {error}")

    # Status and monitoring API
    def get_coordinator_status(self) -> dict[str, Any]:
        """Get comprehensive coordinator status."""
        runtime = time.time() - self.start_time if self.start_time else 0

        # Get audio processing statistics
        audio_processing_stats = {}
        for session_id, processor in self.audio_processors.items():
            audio_processing_stats[session_id] = processor.get_processing_statistics()

        return {
            "is_running": self.is_running,
            "runtime_seconds": runtime,
            "total_audio_processed": self.total_audio_processed,
            "total_chunks_created": self.total_chunks_created,
            "total_transcripts_generated": self.total_transcripts_generated,
            "total_translations_generated": self.total_translations_generated,
            "processing_errors": self.processing_errors,
            "success_rate": 1.0 - (self.processing_errors / max(1, self.total_chunks_created)),
            "session_statistics": self.session_manager.get_session_statistics(),
            "database_statistics": self.database_adapter.get_operation_statistics()
            if self.database_adapter
            else {},
            "audio_processing_statistics": audio_processing_stats,
            "audio_config": {
                "default_preset": self.audio_config_manager.get_default_config().preset_name,
                "available_presets": self.get_available_audio_presets(),
                "active_processors": len(self.audio_processors),
            },
            "config": self.config.dict(),
        }

    def get_session_status(self, session_id: str) -> dict[str, Any] | None:
        """Get status for a specific session."""
        session = self.session_manager.get_session(session_id)
        chunk_manager = self.session_manager.get_chunk_manager(session_id)
        audio_processor = self.audio_processors.get(session_id)

        if not session or not chunk_manager:
            return None

        status = {
            "session": session.dict(),
            "chunk_manager_status": chunk_manager.get_status(),
        }

        # Add audio processing status if available
        if audio_processor:
            status["audio_processing_status"] = audio_processor.get_processing_statistics()
            status["audio_config"] = {
                "preset_name": audio_processor.config.preset_name,
                "enabled_stages": audio_processor.config.enabled_stages,
                "version": audio_processor.config.version,
            }

        return status

    def get_all_sessions_status(self) -> list[dict[str, Any]]:
        """Get status for all active sessions."""
        statuses = []

        for session_id in self.session_manager.active_sessions:
            status = self.get_session_status(session_id)
            if status:
                statuses.append(status)

        return statuses

    # File processing API
    async def process_audio_file(
        self,
        session_id: str,
        audio_file_path: str,
        config: dict[str, Any],
        request_id: str,
    ) -> dict[str, Any]:
        """
        Process an audio file through the orchestration pipeline.

        Args:
            session_id: Audio session identifier
            audio_file_path: Path to the input audio file
            config: Audio processing configuration (supports enable_transcription, enable_translation,
                    enable_diarization, target_languages, whisper_model, translation_quality)
            request_id: Request tracking ID

        Returns:
            Dict[str, Any]: Processing results with transcription, translations, and metadata
        """
        try:
            logger.info(f"[{request_id}] Processing audio file through orchestration pipeline")

            # Load audio file - handle WebM and other formats that soundfile doesn't support
            import os

            import soundfile as sf

            # Check if file is WebM or other unsupported format
            file_ext = os.path.splitext(audio_file_path)[1].lower()

            if file_ext in [".webm", ".mp4", ".m4a", ".mp3"]:
                # Convert to WAV using ffmpeg directly
                logger.info(f"[{request_id}] Converting {file_ext} to WAV for processing")
                import subprocess
                import tempfile

                # Get target sample rate from config (default to 16kHz for Whisper)
                target_sample_rate = config.get("sample_rate", 16000)

                # Create temporary WAV file
                with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as temp_wav:
                    temp_wav_path = temp_wav.name

                try:
                    # Use ffmpeg to convert to WAV (mono, target sample rate)
                    ffmpeg_cmd = [
                        "ffmpeg",
                        "-i",
                        audio_file_path,  # Input file
                        "-ac",
                        "1",  # Convert to mono (1 channel)
                        "-ar",
                        str(target_sample_rate),  # Set sample rate
                        "-y",  # Overwrite output file
                        temp_wav_path,  # Output file
                    ]

                    # Run ffmpeg
                    result = subprocess.run(ffmpeg_cmd, capture_output=True, text=True, check=True)

                    logger.info(
                        f"[{request_id}] Converted to mono @ {target_sample_rate}Hz using ffmpeg"
                    )

                    # Read the WAV file
                    audio_data, sample_rate = sf.read(temp_wav_path)

                except subprocess.CalledProcessError as e:
                    logger.error(f"[{request_id}] ffmpeg conversion failed: {e.stderr}")
                    raise RuntimeError(f"Failed to convert {file_ext} to WAV: {e.stderr}") from e
                finally:
                    # Clean up temp file
                    if os.path.exists(temp_wav_path):
                        os.unlink(temp_wav_path)
            else:
                # soundfile can handle this format directly
                audio_data, sample_rate = sf.read(audio_file_path)

            # Convert to numpy array if needed
            if not isinstance(audio_data, np.ndarray):
                audio_data = np.array(audio_data)

            # Ensure mono audio (take first channel if stereo)
            if len(audio_data.shape) > 1:
                audio_data = audio_data[:, 0]

            # Convert to float32 if not already
            if audio_data.dtype != np.float32:
                audio_data = audio_data.astype(np.float32)

            logger.info(
                f"[{request_id}] Loaded audio file: {len(audio_data)} samples at {sample_rate}Hz"
            )

            # Check if audio processing should be applied
            enable_audio_processing = config.get("audio_processing", True)

            if enable_audio_processing:
                # Build modular enabled stages based on frontend flags
                enable_noise_reduction = config.get("noise_reduction", False)
                enable_speech_enhancement = config.get("speech_enhancement", False)

                enabled_stages = []

                # Always enable VAD for voice activity detection
                enabled_stages.append("vad")

                # Noise reduction stages
                if enable_noise_reduction:
                    enabled_stages.extend(
                        [
                            "noise_reduction",
                            "spectral_denoising",
                            "conventional_denoising",
                        ]
                    )
                    logger.info(f"[{request_id}] Noise reduction enabled")

                # Speech enhancement stages
                if enable_speech_enhancement:
                    enabled_stages.extend(
                        [
                            "voice_filter",
                            "voice_enhancement",
                            "equalizer",
                            "lufs_normalization",
                            "agc",
                            "compression",
                            "limiter",
                        ]
                    )
                    logger.info(f"[{request_id}] Speech enhancement enabled")

                if len(enabled_stages) > 1:  # More than just VAD
                    # Get base session config and customize enabled stages
                    from .audio_processor import create_audio_pipeline_processor
                    from .config import AudioProcessingConfig

                    # Get or create base config
                    base_config = self.audio_config_manager.get_session_config(session_id)

                    # Create custom config with selected stages
                    custom_config = AudioProcessingConfig(
                        enabled_stages=enabled_stages,
                        vad=base_config.vad,
                        voice_filter=base_config.voice_filter,
                        noise_reduction=base_config.noise_reduction,
                        voice_enhancement=base_config.voice_enhancement,
                        equalizer=base_config.equalizer,
                        spectral_denoising=base_config.spectral_denoising,
                        conventional_denoising=base_config.conventional_denoising,
                        lufs_normalization=base_config.lufs_normalization,
                        agc=base_config.agc,
                        compression=base_config.compression,
                        limiter=base_config.limiter,
                        quality=base_config.quality,
                        sample_rate=sample_rate,
                        preset_name=f"modular_{session_id}",
                    )

                    # Create processor with custom config
                    audio_processor = create_audio_pipeline_processor(custom_config, sample_rate)
                    logger.info(
                        f"[{request_id}] Created modular audio processor with stages: {enabled_stages}"
                    )

                    # Process audio through pipeline
                    processed_audio, processing_metadata = audio_processor.process_audio_chunk(
                        audio_data
                    )

                    if processing_metadata.get("bypassed", False):
                        logger.info(
                            f"[{request_id}] Audio processing bypassed: {processing_metadata.get('bypass_reason', 'unknown')}"
                        )
                    elif processing_metadata.get("stages_applied"):
                        logger.info(
                            f"[{request_id}] Applied audio processing stages: {processing_metadata['stages_applied']}"
                        )
                else:
                    # Only VAD enabled - use raw audio
                    logger.info(f"[{request_id}] Only VAD enabled - using raw audio")
                    processed_audio = audio_data
                    processing_metadata = {
                        "bypassed": True,
                        "bypass_reason": "only_vad_enabled",
                    }
            else:
                # Skip audio processing - use raw audio
                logger.info(f"[{request_id}] Audio processing DISABLED - using raw audio")
                processed_audio = audio_data
                processing_metadata = {
                    "bypassed": True,
                    "bypass_reason": "disabled_by_config",
                }

            # Create chunk metadata for tracking
            audio_duration = len(audio_data) / sample_rate

            # Get file size
            import os

            file_size = os.path.getsize(audio_file_path) if os.path.exists(audio_file_path) else 0

            chunk_metadata = create_audio_chunk_metadata(
                session_id=session_id,
                file_path=audio_file_path,
                file_size=file_size,
                duration_seconds=audio_duration,
                chunk_sequence=0,
                chunk_start_time=0.0,
                source_type=SourceType.MANUAL_UPLOAD,
                sample_rate=sample_rate,
                audio_quality_score=processing_metadata.get("quality_score", 0.9),
            )

            # Initialize result
            result = {
                "status": "processed",
                "file_path": audio_file_path,
                "duration": audio_duration,
                "processing_time": 0.0,
            }

            start_time = time.time()

            # Send to whisper service if transcription enabled
            if config.get("enable_transcription", True):
                logger.info(f"[{request_id}] Sending to whisper service for transcription")

                # DEBUG: Log audio quality before sending to Whisper
                audio_rms = np.sqrt(np.mean(processed_audio**2))
                audio_max = np.max(np.abs(processed_audio))
                logger.info(
                    f"[{request_id}] Audio quality before Whisper: RMS={audio_rms:.4f}, Max={audio_max:.4f}, Samples={len(processed_audio)}"
                )

                # Optionally dump audio for debugging
                if config.get("debug_audio", False):
                    import tempfile

                    with tempfile.NamedTemporaryFile(suffix="_pre_whisper.wav", delete=False) as f:
                        sf.write(f.name, processed_audio, sample_rate)
                        logger.info(f"[{request_id}] DEBUG: Saved pre-whisper audio to {f.name}")

                whisper_model = config.get("whisper_model", "whisper-tiny")
                transcript_result = await self.service_client.send_to_whisper_service(
                    session_id, chunk_metadata, processed_audio, whisper_model
                )

                if transcript_result and transcript_result.get("text"):
                    result["transcription"] = transcript_result.get("text", "")
                    result["language"] = transcript_result.get("language", "en")
                    result["confidence"] = transcript_result.get("confidence", 0.0)
                    result["segments"] = transcript_result.get("segments", [])

                    # Add speaker information if available
                    speaker_info = transcript_result.get("speaker_info", {})
                    if speaker_info.get("speakers"):
                        result["speakers"] = speaker_info["speakers"]

                    logger.info(
                        f"[{request_id}] Transcription complete: {len(result['transcription'])} chars, "
                        f"language={result['language']}, confidence={result['confidence']:.2f}"
                    )

                    # Store transcript in database if available
                    if (self.data_pipeline or self.database_adapter) and config.get("session_id"):
                        try:
                            if self.data_pipeline:
                                # Use production-ready pipeline
                                transcript_id = await self._store_transcript_via_pipeline(
                                    config.get("session_id"),
                                    {
                                        "text": result["transcription"],
                                        "start_timestamp": 0.0,
                                        "end_timestamp": audio_duration,
                                        "language": result["language"],
                                        "confidence": result["confidence"],
                                        "speaker_info": speaker_info,
                                        "chunk_id": chunk_metadata.chunk_id,
                                        "source_type": "file_upload",
                                        "metadata": transcript_result.get("metadata", {}),
                                    },
                                    audio_file_id=chunk_metadata.chunk_id,
                                )
                            else:
                                # Fallback to legacy adapter (deprecated)
                                transcript_id = await self.database_adapter.store_transcript(
                                    config.get("session_id"),
                                    {
                                        "text": result["transcription"],
                                        "start_timestamp": 0.0,
                                        "end_timestamp": audio_duration,
                                        "language": result["language"],
                                        "confidence": result["confidence"],
                                        "speaker_info": speaker_info,
                                        "chunk_id": chunk_metadata.chunk_id,
                                        "source_type": "file_upload",
                                        "metadata": transcript_result.get("metadata", {}),
                                    },
                                    audio_file_id=chunk_metadata.chunk_id,
                                )
                            result["transcript_id"] = transcript_id
                            logger.info(
                                f"[{request_id}] Stored transcript in database: {transcript_id}"
                            )
                        except Exception as db_error:
                            logger.warning(
                                f"[{request_id}] Failed to store transcript in database: {db_error}"
                            )

                    # Request translations if enabled
                    if config.get("enable_translation", False) and config.get("target_languages"):
                        target_languages = config["target_languages"]

                        # Handle string or list format
                        if isinstance(target_languages, str):
                            import json

                            try:
                                target_languages = json.loads(target_languages)
                            except json.JSONDecodeError:
                                target_languages = [
                                    lang.strip() for lang in target_languages.split(",")
                                ]

                        logger.info(
                            f"[{request_id}] Requesting translations for languages: {target_languages}"
                        )

                        translations = {}
                        source_language = result["language"]

                        # Process translations concurrently
                        translation_tasks = []
                        for target_lang in target_languages:
                            if target_lang != source_language:  # Skip same language
                                task = asyncio.create_task(
                                    self._translate_single_file(
                                        session_id,
                                        transcript_result,
                                        target_lang,
                                        request_id,
                                        config,
                                    )
                                )
                                translation_tasks.append((target_lang, task))

                        # Wait for all translations
                        for target_lang, task in translation_tasks:
                            try:
                                translation_result = await task
                                if translation_result and translation_result.get("translated_text"):
                                    translations[target_lang] = {
                                        "text": translation_result["translated_text"],
                                        "confidence": translation_result.get("confidence", 0.0),
                                        "service": translation_result.get("service", "unknown"),
                                    }
                                    logger.info(
                                        f"[{request_id}] Translation complete ({target_lang}): "
                                        f"{len(translation_result['translated_text'])} chars"
                                    )
                            except Exception as trans_error:
                                logger.error(
                                    f"[{request_id}] Translation failed for {target_lang}: {trans_error}"
                                )
                                translations[target_lang] = {"error": str(trans_error)}

                        result["translations"] = translations
                        logger.info(f"[{request_id}] Completed {len(translations)} translations")
                else:
                    logger.warning(f"[{request_id}] Whisper service returned no transcription")
                    result["status"] = "error"
                    result["error"] = "Transcription service returned empty result"
            else:
                logger.info(f"[{request_id}] Transcription disabled by configuration")

            # Calculate total processing time
            result["processing_time"] = time.time() - start_time

            logger.info(
                f"[{request_id}] Audio file processing complete in {result['processing_time']:.2f}s: "
                f"status={result['status']}"
            )

            return result

        except Exception as e:
            logger.error(
                f"[{request_id}] Failed to process audio file through orchestration: {e}",
                exc_info=True,
            )
            return {
                "status": "error",
                "error": str(e),
                "file_path": audio_file_path,
                "processing_time": 0.0,
            }

    async def _translate_single_file(
        self,
        session_id: str,
        transcript_result: dict[str, Any],
        target_language: str,
        request_id: str,
        config: dict[str, Any],
    ) -> dict[str, Any] | None:
        """Translate a single file's transcription to target language."""
        try:
            translation_result = await self.service_client.send_to_translation_service(
                session_id, transcript_result, target_language
            )

            if translation_result and translation_result.get("translated_text"):
                # Store translation in database if available
                if (
                    (self.data_pipeline or self.database_adapter)
                    and config.get("session_id")
                    and transcript_result.get("transcript_id")
                ):
                    try:
                        if self.data_pipeline:
                            # Use production-ready pipeline
                            translation_id = await self._store_translation_via_pipeline(
                                config.get("session_id"),
                                transcript_result.get("transcript_id"),
                                {
                                    "translated_text": translation_result["translated_text"],
                                    "source_language": transcript_result.get("language", "auto"),
                                    "target_language": target_language,
                                    "confidence": translation_result.get("confidence", 0.0),
                                    "translation_service": translation_result.get(
                                        "service", "unknown"
                                    ),
                                    "speaker_id": transcript_result.get("speaker_info", {}).get(
                                        "speaker_id"
                                    ),
                                    "speaker_name": transcript_result.get("speaker_info", {}).get(
                                        "speaker_name"
                                    ),
                                    "metadata": translation_result.get("metadata", {}),
                                },
                                start_timestamp=0.0,
                                end_timestamp=config.get("duration", 0.0),
                            )
                        else:
                            # Fallback to legacy adapter (deprecated)
                            translation_id = await self.database_adapter.store_translation(
                                config.get("session_id"),
                                transcript_result.get("transcript_id"),
                                {
                                    "translated_text": translation_result["translated_text"],
                                    "source_language": transcript_result.get("language", "auto"),
                                    "target_language": target_language,
                                    "confidence": translation_result.get("confidence", 0.0),
                                    "translation_service": translation_result.get(
                                        "service", "unknown"
                                    ),
                                    "speaker_id": transcript_result.get("speaker_info", {}).get(
                                        "speaker_id"
                                    ),
                                    "speaker_name": transcript_result.get("speaker_info", {}).get(
                                        "speaker_name"
                                    ),
                                    "start_timestamp": 0.0,
                                    "end_timestamp": config.get("duration", 0.0),
                                    "metadata": translation_result.get("metadata", {}),
                                },
                            )
                        translation_result["translation_id"] = translation_id
                        logger.info(
                            f"[{request_id}] Stored translation in database: {translation_id}"
                        )
                    except Exception as db_error:
                        logger.warning(
                            f"[{request_id}] Failed to store translation in database: {db_error}"
                        )

                return translation_result

            return None

        except Exception as e:
            logger.error(f"[{request_id}] Translation failed for {target_language}: {e}")
            return None


# Factory function for creating audio coordinator
def create_audio_coordinator(
    database_url: str | None,
    service_urls: dict[str, str],
    config: AudioChunkingConfig | None = None,
    max_concurrent_sessions: int = 10,
    audio_config_file: str | None = None,
    audio_client: AudioServiceClient | None = None,
    translation_client: TranslationServiceClient | None = None,
    data_pipeline: Optional["TranscriptionDataPipeline"] = None,
) -> AudioCoordinator:
    """
    Create and return an AudioCoordinator instance.

    Args:
        database_url: Database URL (deprecated if data_pipeline provided)
        service_urls: Dictionary of downstream service URLs
        config: Audio chunking configuration
        max_concurrent_sessions: Maximum concurrent audio sessions
        audio_config_file: Path to audio configuration file
        audio_client: Optional audio service client (for embedded mode)
        translation_client: Optional translation service client (for embedded mode)
        data_pipeline: Production-ready TranscriptionDataPipeline (RECOMMENDED)

    Returns:
        Configured AudioCoordinator instance

    Note:
        Prefer passing data_pipeline over database_url for production deployments.
        The data_pipeline includes NULL safety, LRU caching, transactions, and rate limiting.
    """
    if not config:
        config = get_default_chunking_config()

    return AudioCoordinator(
        config,
        database_url,
        service_urls,
        max_concurrent_sessions,
        audio_config_file,
        audio_client=audio_client,
        translation_client=translation_client,
        data_pipeline=data_pipeline,
    )


# Example usage and testing
async def main():
    """Example usage of the audio coordinator."""
    import os

    # Configuration
    database_url = os.getenv(
        "DATABASE_URL", "postgresql://postgres:password@localhost:5432/livetranslate"
    )
    service_urls = {
        "whisper_service": "http://localhost:5001",
        "translation_service": "http://localhost:5003",
    }

    # Create coordinator
    coordinator = create_audio_coordinator(database_url, service_urls)

    # Set callbacks
    def on_transcript_ready(data):
        print(f"Transcript ready: {data['text'][:50]}...")

    def on_translation_ready(data):
        print(f"Translation ready ({data['target_language']}): {data['translated_text'][:50]}...")

    def on_session_event(event_type, data):
        print(f"Session event: {event_type}")

    coordinator.set_transcript_ready_callback(on_transcript_ready)
    coordinator.set_translation_ready_callback(on_translation_ready)
    coordinator.set_session_event_callback(on_session_event)

    try:
        # Initialize
        success = await coordinator.initialize()
        if not success:
            print("Failed to initialize coordinator")
            return

        # Create session
        session_id = await coordinator.create_audio_session(
            bot_session_id="test-bot-session-123",
            source_type=SourceType.BOT_AUDIO,
            target_languages=["en", "es", "fr"],
        )

        if session_id:
            print(f"Created session: {session_id}")

            # Start session
            await coordinator.start_audio_session(session_id)

            # Simulate audio data
            sample_rate = 16000
            for i in range(50):  # 5 seconds of audio
                # Generate test audio
                t = np.arange(1600) / sample_rate + i * 0.1
                audio = 0.1 * np.sin(2 * np.pi * 440 * t) + 0.01 * np.random.randn(1600)

                await coordinator.add_audio_data(session_id, audio.astype(np.float32))
                await asyncio.sleep(0.1)  # Real-time simulation

            # Stop session
            final_stats = await coordinator.stop_audio_session(session_id)
            print(f"Session final stats: {final_stats}")

            # Get coordinator status
            status = coordinator.get_coordinator_status()
            print(f"Coordinator status: {status}")

    finally:
        # Shutdown
        await coordinator.shutdown()


if __name__ == "__main__":
    asyncio.run(main())
